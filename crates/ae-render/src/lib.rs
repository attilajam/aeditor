use std::fmt;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::thread;

use ae_core::{AeError, Rational, Resolution, Result, RgbaFrame, Timeline, VideoCodec};
use ae_media::{LIGHT_CLEANUP_FILTER, MediaBackend, VideoWriterConfig, stretch_frame_lanczos};
use ae_transitions::{TransitionCtx, build_transition};
use ae_transitions_gpu::GpuTransitionEngine;
use crossbeam_channel::bounded;
use lru::LruCache;
use sha2::{Digest, Sha256};

mod enhance;
pub use enhance::{enhance_quality, enhance_video, prepare_quality_enhancer};

mod reframe;
pub use reframe::{
    CenterFallback, CharacterAnalysisDetector, CharacterDetection, CropRect, FaceBox, FaceDetector,
    NamedHead, ReframeConfig, Reframer, YUNET_INPUT, YUNET_STRIDES, YunetDetector, YunetFace,
    clamp_crop, cover_decode_res, crop_frame, decode_yunet, open_detector, reframe_detected_frame,
    reframe_fingerprint, reframe_frame, yunet_bgr_input,
};

pub type RenderProgressCallback = Arc<dyn Fn(u32, &str) + Send + Sync>;

#[derive(Clone)]
pub struct RenderConfig {
    pub video_path: PathBuf,
    pub audio_path: Option<PathBuf>,
    pub output_path: PathBuf,
    pub resolution: Option<Resolution>,
    pub fps: Option<Rational>,
    pub codec: VideoCodec,
    pub crf: u32,
    pub preset: String,
    pub use_gpu: bool,
    pub quality_enhance: bool,
    /// Fast cleanup and color polish in the existing encoder pass.
    pub light_cleanup: bool,
    /// Opt-in subject locking with an aspect-preserving crop. Default off.
    pub reframe_faces: bool,
    /// Optional character or YuNet ONNX model; missing file => centered CLI fallback.
    pub face_model_path: Option<PathBuf>,
    pub shaders_dir: Option<PathBuf>,
    pub cache_capacity: usize,
    pub progress_cb: Option<RenderProgressCallback>,
}

impl fmt::Debug for RenderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RenderConfig")
            .field("video_path", &self.video_path)
            .field("output_path", &self.output_path)
            .field("resolution", &self.resolution)
            .field("reframe_faces", &self.reframe_faces)
            .finish()
    }
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            video_path: PathBuf::new(),
            audio_path: None,
            output_path: PathBuf::from("out.mp4"),
            resolution: None,
            fps: None,
            codec: VideoCodec::X264,
            crf: 18,
            preset: "fast".to_string(),
            use_gpu: false,
            quality_enhance: false,
            light_cleanup: false,
            reframe_faces: false,
            face_model_path: None,
            shaders_dir: None,
            cache_capacity: 16,
            progress_cb: None,
        }
    }
}

/// Compute exact frame counts per timeline slot using error-diffusion rounding so cumulative
/// cut timestamps never drift from the beat grid by more than ±0.5 frame over any song length.
pub fn compute_frame_accurate_slot_lengths(timeline: &Timeline, fps: Rational) -> Vec<usize> {
    let fps_f = fps.as_secs_f64();
    let mut lengths = Vec::with_capacity(timeline.slots.len());
    let mut cumulative_frames = 0i64;

    for slot in &timeline.slots {
        let target_end_frames = (slot.dst_end.as_secs_f64() * fps_f).round() as i64;
        let slot_frames = (target_end_frames - cumulative_frames).max(1);
        cumulative_frames += slot_frames;
        lengths.push(slot_frames as usize);
    }

    lengths
}

/// Render a `Timeline` to `config.output_path`, stretching every frame to fill the target
/// resolution and applying modular CPU or GPU transitions at each beat boundary.
pub fn render_timeline(
    backend: &dyn MediaBackend,
    timeline: &Timeline,
    config: &RenderConfig,
) -> Result<()> {
    if config.quality_enhance {
        if let Some(cb) = &config.progress_cb {
            cb(
                2,
                "Loading AI quality enhancer (first use downloads 86 MB)...",
            );
        }
        prepare_quality_enhancer()?;
    }
    let target_res = config.resolution.unwrap_or(timeline.resolution);
    let fps = config.fps.unwrap_or(timeline.fps);
    let slot_frame_counts = compute_frame_accurate_slot_lengths(timeline, fps);
    let total_frames: usize = slot_frame_counts.iter().sum();
    let total_duration = Rational::from_frame_index(total_frames as i64, fps);

    let video_path = if !config.video_path.as_os_str().is_empty() {
        config.video_path.clone()
    } else {
        PathBuf::from(&timeline.source_video)
    };

    let audio_path = config.audio_path.clone().or_else(|| {
        if !timeline.source_audio.is_empty() && Path::new(&timeline.source_audio).exists() {
            Some(PathBuf::from(&timeline.source_audio))
        } else {
            None
        }
    });

    let gpu_engine = if config.use_gpu {
        Some(GpuTransitionEngine::new(config.shaders_dir.as_deref()))
    } else {
        None
    };

    // Opt-in reframe path decodes at a cover-fit size (source aspect preserved,
    // so the stretch inside decode is a no-op) and crops per frame later.
    // Default rendering keeps decoding straight to the stretched target.
    let decode_res = if config.reframe_faces {
        let probe = backend.probe_video(&video_path)?;
        let (sw, sh) = probe.oriented_dimensions();
        cover_decode_res(sw, sh, target_res)
    } else {
        target_res
    };

    // Pre-decode slots with an LRU cache so reused shots or overlapping handles are cached
    let cap = NonZeroUsize::new(config.cache_capacity.max(2)).unwrap();
    let mut seg_cache: LruCache<_, Vec<RgbaFrame>> = LruCache::new(cap);

    // For each slot, also decode any trailing handle frames needed by the NEXT slot's transition
    let mut decoded_slots: Vec<Vec<RgbaFrame>> = Vec::with_capacity(timeline.slots.len());

    let total_slots = timeline.slots.len().max(1);
    for (idx, slot) in timeline.slots.iter().enumerate() {
        if let Some(ref cb) = config.progress_cb {
            let pct = 10 + ((idx * 45) / total_slots) as u32;
            cb(
                pct,
                &format!(
                    "Decoding slot {}/{} (Scene #{})...",
                    idx + 1,
                    total_slots,
                    slot.shot_id
                ),
            );
        }
        let base_frames = slot_frame_counts[idx];
        let next_trans_frames = if idx + 1 < timeline.slots.len() {
            let next_spec = &timeline.slots[idx + 1].transition_in;
            if next_spec.id == "cut" {
                0usize
            } else {
                ((next_spec.duration.as_secs_f64() * fps.as_secs_f64()).round() as usize)
                    .min(base_frames / 2)
            }
        } else {
            0usize
        };

        let needed_frames = base_frames + next_trans_frames;
        let needed_dur = Rational::from_frame_index(needed_frames as i64, fps);
        let cache_key = (
            slot.shot_id,
            slot.src_in,
            slot.src_out,
            slot.speed.to_bits(),
            needed_frames,
            decode_res.width,
            decode_res.height,
            config.reframe_faces,
        );

        let frames = if let Some(cached) = seg_cache.get(&cache_key) {
            cached.clone()
        } else {
            let mut seg = backend.decode_segment(
                &video_path,
                slot.src_in,
                needed_dur,
                decode_res,
                fps,
                slot.speed,
                Some(slot.src_out),
            )?;
            // Ensure exact stretch-to-fill target resolution via SIMD Lanczos3
            for f in &mut seg {
                if f.width != decode_res.width || f.height != decode_res.height {
                    *f = stretch_frame_lanczos(f, decode_res)?;
                }
            }
            seg_cache.put(cache_key, seg.clone());
            seg
        };

        decoded_slots.push(frames);
    }

    // Per-shot subject locking, then aspect-preserving crop + resize to the target.
    if config.reframe_faces {
        let rcfg = ReframeConfig {
            enabled: true,
            model_path: config.face_model_path.clone(),
            ..ReframeConfig::default()
        };
        let mut detector = open_detector(&rcfg);
        for slot_frames in &mut decoded_slots {
            let mut reframer =
                Reframer::new(&rcfg, decode_res.width, decode_res.height, target_res);
            for f in slot_frames.iter_mut() {
                *f = reframe_detected_frame(&mut *detector, &mut reframer, f, target_res)?;
            }
            detector.finish_shot()?;
        }
    }

    // Stream composited frames through a bounded channel to the encoder thread
    let writer_cfg = VideoWriterConfig {
        output_path: config.output_path.clone(),
        audio_path,
        resolution: target_res,
        fps,
        codec: config.codec,
        crf: config.crf,
        preset: config.preset.clone(),
        total_duration: Some(total_duration),
        audio_fade_out_secs: 0.35,
        light_cleanup: config.light_cleanup,
    };

    let mut writer = backend.create_writer(writer_cfg)?;
    let (tx, rx) = bounded::<RgbaFrame>(32);

    thread::scope(|s| -> Result<()> {
        let encode_handle = s.spawn(move || -> Result<()> {
            for (index, frame) in rx.iter().enumerate() {
                if config.quality_enhance {
                    writer.write_frame(&enhance_quality(&frame)?)?;
                    if let Some(cb) = &config.progress_cb {
                        cb(
                            55 + ((index + 1) * 43 / total_frames.max(1)) as u32,
                            &format!("AI enhancement: frame {}/{}", index + 1, total_frames),
                        );
                    }
                } else {
                    writer.write_frame(&frame)?;
                }
            }
            writer.finish()
        });

        let mut out_frame_idx = 0i64;
        for (idx, slot) in timeline.slots.iter().enumerate() {
            if !config.quality_enhance
                && let Some(ref cb) = config.progress_cb
            {
                let pct = 55 + (((idx + 1) * 43) / total_slots) as u32;
                cb(
                    pct,
                    &format!(
                        "Encoding MP4 slot {}/{} ({})...",
                        idx + 1,
                        total_slots,
                        target_res
                    ),
                );
            }
            let slot_len = slot_frame_counts[idx];
            let cur_frames = &decoded_slots[idx];

            let trans_frames = if idx > 0 && slot.transition_in.id != "cut" {
                ((slot.transition_in.duration.as_secs_f64() * fps.as_secs_f64()).round() as usize)
                    .min(slot_len / 2)
                    .min(
                        decoded_slots[idx - 1]
                            .len()
                            .saturating_sub(slot_frame_counts[idx - 1]),
                    )
            } else {
                0usize
            };

            let transition_opt = if trans_frames > 0 {
                let spec = &slot.transition_in;
                if let Some(ref gpu) = gpu_engine {
                    Some(gpu.resolve(&spec.id, &spec.params)?)
                } else {
                    Some(build_transition(&spec.id, &spec.params)?)
                }
            } else {
                None
            };

            let ctx = TransitionCtx {
                width: target_res.width,
                height: target_res.height,
                fps,
                duration: slot.transition_in.duration,
                easing: slot.transition_in.easing.clone(),
                beat_strength: slot.beat_strength,
            };

            for f_in_slot in 0..slot_len {
                let pts = Rational::from_frame_index(out_frame_idx, fps);
                out_frame_idx += 1;

                let to_frame = &cur_frames[f_in_slot.min(cur_frames.len() - 1)];

                if f_in_slot < trans_frames
                    && let Some(ref tr) = transition_opt
                {
                    let prev_slot_len = slot_frame_counts[idx - 1];
                    let prev_frames = &decoded_slots[idx - 1];
                    let from_idx = (prev_slot_len + f_in_slot).min(prev_frames.len() - 1);
                    let from_frame = &prev_frames[from_idx];
                    let progress = ((f_in_slot + 1) as f32) / ((trans_frames + 1) as f32);

                    let mut blended = RgbaFrame::new(target_res.width, target_res.height, pts);
                    tr.render(
                        &ctx,
                        from_frame.as_ref(),
                        to_frame.as_ref(),
                        progress,
                        blended.as_mut(),
                    );
                    if tx.send(blended).is_err() {
                        break;
                    }
                } else {
                    let mut out_f = to_frame.clone();
                    out_f.pts = pts;
                    if tx.send(out_f).is_err() {
                        break;
                    }
                }
            }
        }

        drop(tx);
        encode_handle.join().expect("encoder thread panicked")?;
        Ok(())
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IncrementalDraftReport {
    pub output_path: PathBuf,
    pub resolution: Resolution,
    pub total_slots: usize,
    pub cached_slots: usize,
    pub encoded_slots: usize,
}

/// Phase P3: Incremental slot-level draft renderer.
/// Hashes each `TimelineSlot` into `.ae-cache/draft_slots/<hash>.ts`. Unchanged slots are reused
/// from cache, changed slots are encoded with `libx264 -preset ultrafast`, and all
/// slots are stitched without re-encoding via `ffmpeg -f concat -c:v copy`.
pub fn render_timeline_incremental_draft(
    backend: &dyn MediaBackend,
    timeline: &Timeline,
    config: &RenderConfig,
    cache_dir: &Path,
    scale_divisor: u32,
) -> Result<IncrementalDraftReport> {
    if config.quality_enhance {
        if let Some(cb) = &config.progress_cb {
            cb(
                2,
                "Loading AI quality enhancer (first use downloads 86 MB)...",
            );
        }
        prepare_quality_enhancer()?;
    }
    let base_res = config.resolution.unwrap_or(timeline.resolution);
    let div = scale_divisor.clamp(1, 4);
    let draft_res = Resolution::new(
        (base_res.width / div).max(64),
        (base_res.height / div).max(64),
    );
    let fps = config.fps.unwrap_or(timeline.fps);
    let slot_frame_counts = compute_frame_accurate_slot_lengths(timeline, fps);
    let total_frames: usize = slot_frame_counts.iter().sum();
    let total_duration = Rational::from_frame_index(total_frames as i64, fps);

    let video_path = if !config.video_path.as_os_str().is_empty() {
        config.video_path.clone()
    } else {
        PathBuf::from(&timeline.source_video)
    };

    // Opt-in preview reframe: decode at a cover-fit size (source aspect
    // preserved) and crop per frame later, exactly like the final renderer.
    // The fingerprint below keeps caches from reusing stale segments when the
    // toggle or model changes.
    let reframe_fp = reframe_fingerprint(config.reframe_faces, config.face_model_path.as_deref());
    let decode_res = if config.reframe_faces {
        let probe = backend.probe_video(&video_path)?;
        let (sw, sh) = probe.oriented_dimensions();
        cover_decode_res(sw, sh, draft_res)
    } else {
        draft_res
    };

    let slots_dir = cache_dir.join("draft_slots");
    std::fs::create_dir_all(&slots_dir)?;

    let total_slots = timeline.slots.len();
    let mut cached_slots = 0usize;
    let mut encoded_slots = 0usize;
    let mut segment_paths: Vec<PathBuf> = Vec::with_capacity(total_slots);
    let mut jobs: Vec<(usize, PathBuf, usize)> = Vec::new();
    let mut seen: std::collections::HashMap<PathBuf, usize> = std::collections::HashMap::new();
    let mut initially_cached = 0usize;

    for (idx, slot) in timeline.slots.iter().enumerate() {
        let frames_count = slot_frame_counts[idx];
        let mut hasher = Sha256::new();
        hasher.update(b"draft-transition-v4");
        hasher.update([u8::from(config.quality_enhance)]);
        if config.light_cleanup {
            hasher.update(LIGHT_CLEANUP_FILTER);
        }
        if config.quality_enhance {
            hasher.update(enhance::CACHE_VERSION);
        }
        hasher.update([u8::from(config.use_gpu)]);
        hasher.update(reframe_fp.as_bytes());
        hasher.update(video_path.to_string_lossy().as_bytes());
        hasher.update(b"|");
        hasher.update(slot.shot_id.to_le_bytes());
        hasher.update(slot.src_in.numer().to_le_bytes());
        hasher.update(slot.src_in.denom().to_le_bytes());
        hasher.update(slot.src_out.numer().to_le_bytes());
        hasher.update(slot.src_out.denom().to_le_bytes());
        hasher.update((frames_count as u64).to_le_bytes());
        hasher.update(draft_res.width.to_le_bytes());
        hasher.update(draft_res.height.to_le_bytes());
        hasher.update(fps.numer().to_le_bytes());
        hasher.update(fps.denom().to_le_bytes());
        hasher.update(slot.speed.to_bits().to_le_bytes());
        hasher.update(slot.transition_in.id.as_bytes());
        hasher.update(slot.transition_in.duration.numer().to_le_bytes());
        hasher.update(slot.transition_in.duration.denom().to_le_bytes());
        hasher.update(slot.transition_in.easing.as_bytes());
        hasher.update(slot.transition_in.params.to_string().as_bytes());
        hasher.update(slot.beat_strength.to_bits().to_le_bytes());
        if idx > 0 && slot.transition_in.id != "cut" {
            let prev = &timeline.slots[idx - 1];
            hasher.update(prev.shot_id.to_le_bytes());
            hasher.update(prev.src_in.numer().to_le_bytes());
            hasher.update(prev.src_in.denom().to_le_bytes());
            hasher.update(prev.src_out.numer().to_le_bytes());
            hasher.update(prev.src_out.denom().to_le_bytes());
            hasher.update(prev.speed.to_bits().to_le_bytes());
            hasher.update(slot_frame_counts[idx - 1].to_le_bytes());
        }
        let hash_hex = hex::encode(hasher.finalize());
        let ts_path = slots_dir.join(format!("slot_{hash_hex}.ts"));

        segment_paths.push(ts_path.clone());
        if ts_path.exists() {
            cached_slots += 1;
            initially_cached += 1;
        } else if let Some(&job) = seen.get(&ts_path) {
            cached_slots += 1;
            jobs[job].2 += 1;
        } else {
            encoded_slots += 1;
            seen.insert(ts_path.clone(), jobs.len());
            jobs.push((idx, ts_path, 1usize));
        }
    }

    let completed = std::sync::atomic::AtomicUsize::new(initially_cached);
    if initially_cached > 0 {
        if let Some(cb) = &config.progress_cb {
            cb(
                10 + (initially_cached * 70 / total_slots.max(1)) as u32,
                &format!("Quick Render: {initially_cached}/{total_slots} slots cached"),
            );
        }
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let progress_lock = std::sync::Mutex::new(());
    thread::scope(|scope| -> Result<()> {
        let mut workers = Vec::new();
        for _ in 0..jobs.len().min(4) {
            let jobs = &jobs;
            let next = &next;
            let completed = &completed;
            let progress_lock = &progress_lock;
            let video_path = &video_path;
            let slot_frame_counts = &slot_frame_counts;
            let decode_res = &decode_res;
            workers.push(scope.spawn(move || -> Result<()> {
                loop {
                    let job = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((idx, ts_path, count)) = jobs.get(job) else {
                        break;
                    };
                    let idx = *idx;
                    let slot = &timeline.slots[idx];
                    let frames_count = slot_frame_counts[idx];
                    let transition_frames = if idx > 0 && slot.transition_in.id != "cut" {
                        ((slot.transition_in.duration.as_secs_f64() * fps.as_secs_f64()).round()
                            as usize)
                            .min(frames_count / 2)
                            .min(slot_frame_counts[idx - 1] / 2)
                    } else {
                        0
                    };
                    static TEMP_ID: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(0);
                    let temp_path = ts_path.with_extension(format!(
                        "{}.{}.tmp",
                        std::process::id(),
                        TEMP_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    ));
                    let result = (|| -> Result<()> {
                        if transition_frames == 0
                            && !config.quality_enhance
                            && !config.light_cleanup
                            && !config.reframe_faces
                        {
                            return encode_hard_cut_mpegts(
                                &video_path,
                                &temp_path,
                                slot,
                                frames_count,
                                draft_res,
                                fps,
                            );
                        }
                        render_raw_slot(
                            backend,
                            video_path,
                            &temp_path,
                            timeline,
                            idx,
                            frames_count,
                            transition_frames,
                            slot_frame_counts,
                            draft_res,
                            *decode_res,
                            fps,
                            config,
                        )
                    })()
                    .and_then(|()| std::fs::rename(&temp_path, &ts_path).map_err(Into::into));
                    if result.is_err() {
                        let _ = std::fs::remove_file(&temp_path);
                    }
                    result?;
                    let _guard = progress_lock.lock().unwrap();
                    let done =
                        completed.fetch_add(*count, std::sync::atomic::Ordering::Relaxed) + count;
                    if let Some(cb) = &config.progress_cb {
                        cb(
                            10 + (done * 70 / total_slots.max(1)) as u32,
                            &format!("Quick Render: {done}/{total_slots} slots complete"),
                        );
                    }
                }
                Ok(())
            }));
        }
        let mut result = Ok(());
        for worker in workers {
            match worker.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) if result.is_ok() => result = Err(e),
                Err(_) if result.is_ok() => {
                    result = Err(AeError::Config("slot worker panicked".into()))
                }
                _ => {}
            }
        }
        result
    })?;

    if let Some(ref cb) = config.progress_cb {
        cb(
            88,
            &format!(
                "Stitching {} slots ({} cached, {} encoded)...",
                total_slots, cached_slots, encoded_slots
            ),
        );
    }

    // Write ffmpeg concat demuxer list
    let concat_list_path = slots_dir.join("concat_list.txt");
    let mut list_content = String::new();
    for seg in &segment_paths {
        let abs = std::fs::canonicalize(seg).unwrap_or_else(|_| seg.clone());
        list_content.push_str(&format!("file '{}'\n", abs.display()));
    }
    std::fs::write(&concat_list_path, list_content)?;

    if let Some(parent) = config.output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let audio_path = config.audio_path.clone().or_else(|| {
        if !timeline.source_audio.is_empty() && Path::new(&timeline.source_audio).exists() {
            Some(PathBuf::from(&timeline.source_audio))
        } else {
            None
        }
    });

    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
    ])
    .arg(&concat_list_path);

    if let Some(ref aud) = audio_path {
        cmd.arg("-i").arg(aud);
        cmd.args([
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-t",
            &format!("{:.6}", total_duration.as_secs_f64()),
        ]);
    } else {
        cmd.args(["-c:v", "copy"]);
    }

    cmd.args(["-movflags", "+faststart"])
        .arg(&config.output_path);

    let out = cmd.output().map_err(|e| AeError::Encode {
        path: config.output_path.clone(),
        message: format!("failed running ffmpeg concat stitch: {e}"),
    })?;
    if !out.status.success() {
        return Err(AeError::Encode {
            path: config.output_path.clone(),
            message: format!(
                "ffmpeg concat stitch failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ),
        });
    }

    Ok(IncrementalDraftReport {
        output_path: config.output_path.clone(),
        resolution: draft_res,
        total_slots,
        cached_slots,
        encoded_slots,
    })
}

fn encode_hard_cut_mpegts(
    video_path: &Path,
    ts_path: &Path,
    slot: &ae_core::TimelineSlot,
    frames_count: usize,
    res: Resolution,
    fps: Rational,
) -> Result<()> {
    if slot.src_out <= slot.src_in {
        return Err(AeError::Config(
            "source end must follow source start".into(),
        ));
    }
    if fps <= Rational::ZERO || !slot.speed.is_finite() || slot.speed <= 0.0 {
        return Err(AeError::Config(
            "segment frame rate and speed must be positive".into(),
        ));
    }
    let duration = Rational::from_frame_index(frames_count as i64, fps).as_secs_f64();
    let source_duration =
        (duration * slot.speed.max(0.1)).min((slot.src_out - slot.src_in).as_secs_f64());
    let setpts = if (slot.speed - 1.0).abs() > 1e-3 {
        format!("setpts=PTS/{:.6},", slot.speed)
    } else {
        String::new()
    };
    let vf = format!(
        "trim=end={:.6},{setpts}fps={}/{},scale=iw*sar:ih,scale={}:{}:flags=lanczos+accurate_rnd,setsar=1:1,tpad=stop_mode=clone:stop=-1,format=yuv420p",
        ((slot.src_out - slot.src_in).as_secs_f64() - 0.000001).max(0.0),
        fps.numer(),
        fps.denom(),
        res.width,
        res.height
    );
    let output = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "2",
            "-filter_threads",
            "1",
            "-ss",
            &format!("{:.6}", slot.src_in.as_secs_f64().max(0.0)),
            "-t",
            &format!("{:.6}", (source_duration + 0.15).max(0.05)),
            "-i",
        ])
        .arg(video_path)
        .args([
            "-an",
            "-vf",
            &vf,
            "-frames:v",
            &frames_count.to_string(),
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-crf",
            "18",
            "-g",
            "15",
            "-f",
            "mpegts",
        ])
        .arg(ts_path)
        .output()
        .map_err(|e| AeError::Encode {
            path: ts_path.to_path_buf(),
            message: format!("failed running slot encoder: {e}"),
        })?;
    if !output.status.success() || std::fs::metadata(ts_path).map_or(true, |m| m.len() == 0) {
        return Err(AeError::Encode {
            path: ts_path.to_path_buf(),
            message: format!(
                "slot encoder failed or produced no frames: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        });
    }
    Ok(())
}

fn render_raw_slot(
    backend: &dyn MediaBackend,
    video_path: &Path,
    temp_path: &Path,
    timeline: &Timeline,
    idx: usize,
    frames_count: usize,
    transition_frames: usize,
    slot_frame_counts: &[usize],
    res: Resolution,
    decode_res: Resolution,
    fps: Rational,
    config: &RenderConfig,
) -> Result<()> {
    use std::io::{Read, Seek, Write};
    let slot = &timeline.slots[idx];
    // Preview reframe mirrors the final renderer: decode wide at a cover-fit
    // size, then crop per frame. The tail spool belongs to the previous shot,
    // so it gets its own freshly-reset smoother (per-shot reset semantics).
    let mut reframing = if config.reframe_faces {
        let rcfg = ReframeConfig {
            enabled: true,
            model_path: config.face_model_path.clone(),
            ..ReframeConfig::default()
        };
        let tail = Reframer::new(&rcfg, decode_res.width, decode_res.height, res);
        let main = Reframer::new(&rcfg, decode_res.width, decode_res.height, res);
        Some((open_detector(&rcfg), main, tail))
    } else {
        None
    };
    let mut outgoing = None;
    let mut offset = 0;
    if transition_frames > 0 {
        let previous = &timeline.slots[idx - 1];
        let frame_time = Rational::from_secs_f64(previous.speed / fps.as_secs_f64());
        let tail_start = (previous.src_in
            + Rational::from_secs_f64(
                slot_frame_counts[idx - 1] as f64 * previous.speed / fps.as_secs_f64(),
            ))
        .min((previous.src_out - frame_time).max(previous.src_in));
        // shortcut: spool transition tails to disk for safe retries; use replayable streaming to remove this I/O.
        let spool_path = temp_path.with_extension("frames.tmp");
        struct Spool(PathBuf);
        impl Drop for Spool {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let guard = Spool(spool_path);
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&guard.0)?;
        let mut decode = |start, count| -> Result<()> {
            file.set_len(0)?;
            file.rewind()?;
            backend.decode_segment_stream(
                video_path,
                start,
                Rational::from_frame_index(count as i64, fps),
                decode_res,
                fps,
                previous.speed,
                Some(previous.src_out),
                &mut |frame| {
                    let frame = match reframing.as_mut() {
                        Some((detector, _, tail)) => {
                            reframe_detected_frame(detector.as_mut(), tail, &frame, res)?
                        }
                        None => frame,
                    };
                    file.write_all(&frame.width.to_le_bytes())?;
                    file.write_all(&frame.height.to_le_bytes())?;
                    file.write_all(&frame.data)?;
                    Ok(())
                },
            )
        };
        if decode(tail_start, transition_frames).is_err() {
            let earlier = (tail_start - frame_time * 4).max(previous.src_in);
            offset =
                ((tail_start - earlier).as_secs_f64() / frame_time.as_secs_f64()).round() as usize;
            decode(earlier, transition_frames + offset)?;
        }
        file.rewind()?;
        outgoing = Some((file, guard));
    }
    let transition = if transition_frames > 0 {
        Some(if config.use_gpu {
            GpuTransitionEngine::new(config.shaders_dir.as_deref())
                .resolve(&slot.transition_in.id, &slot.transition_in.params)?
        } else {
            build_transition(&slot.transition_in.id, &slot.transition_in.params)?
        })
    } else {
        None
    };
    let ctx = TransitionCtx {
        width: res.width,
        height: res.height,
        fps,
        duration: slot.transition_in.duration,
        easing: slot.transition_in.easing.clone(),
        beat_strength: slot.beat_strength,
    };
    let mut last_outgoing: Option<RgbaFrame> = None;
    let mut read_outgoing = || -> Result<Option<RgbaFrame>> {
        let file = &mut outgoing.as_mut().unwrap().0;
        let mut dims = [0u8; 8];
        match file.read(&mut dims[..1])? {
            0 => return Ok(None),
            _ => file.read_exact(&mut dims[1..])?,
        }
        let width = u32::from_le_bytes(dims[..4].try_into().unwrap());
        let height = u32::from_le_bytes(dims[4..].try_into().unwrap());
        let mut frame = RgbaFrame::new(width, height, Rational::ZERO);
        file.read_exact(&mut frame.data)?;
        Ok(Some(frame))
    };
    for _ in 0..offset {
        if let Some(frame) = read_outgoing()? {
            last_outgoing = Some(frame);
        }
    }
    encode_slot_mpegts(temp_path, res, fps, config.light_cleanup, |emit| {
        let mut i = 0;
        backend.decode_segment_stream(
            video_path,
            slot.src_in,
            Rational::from_frame_index(frames_count as i64, fps),
            decode_res,
            fps,
            slot.speed,
            Some(slot.src_out),
            &mut |mut frame| {
                if reframing.is_some() {
                    let (detector, main, _) = reframing.as_mut().unwrap();
                    frame = reframe_detected_frame(detector.as_mut(), main, &frame, res)?;
                }
                if i < transition_frames {
                    if let Some(frame) = read_outgoing()? {
                        last_outgoing = Some(frame);
                    }
                    let from = last_outgoing.as_ref().ok_or_else(|| AeError::Decode {
                        path: video_path.to_path_buf(),
                        message: "no outgoing frames".into(),
                    })?;
                    let mut blended = RgbaFrame::new(res.width, res.height, frame.pts);
                    transition.as_ref().unwrap().render(
                        &ctx,
                        from.as_ref(),
                        frame.as_ref(),
                        (i + 1) as f32 / (transition_frames + 1) as f32,
                        blended.as_mut(),
                    );
                    frame = blended;
                }
                i += 1;
                if config.quality_enhance {
                    frame = enhance_quality(&frame)?;
                }
                emit(frame)
            },
        )?;
        if let Some((detector, _, _)) = &reframing {
            detector.finish_shot()?;
        }
        Ok(())
    })
}

fn encode_slot_mpegts(
    ts_path: &Path,
    res: Resolution,
    fps: Rational,
    light_cleanup: bool,
    stream: impl FnOnce(&mut dyn FnMut(RgbaFrame) -> Result<()>) -> Result<()>,
) -> Result<()> {
    use std::io::{Read, Write};
    use std::process::Stdio;
    let vf = if light_cleanup {
        format!("setsar=1:1,format=yuv420p,{LIGHT_CLEANUP_FILTER}")
    } else {
        "setsar=1:1,format=yuv420p".into()
    };
    let mut child = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-filter_threads",
            "1",
            "-threads",
            "2",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &format!("{}x{}", res.width, res.height),
            "-r",
            &format!("{}/{}", fps.numer(), fps.denom()),
            "-i",
            "pipe:0",
            "-vf",
            &vf,
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-crf",
            "18",
            "-g",
            "15",
            "-f",
            "mpegts",
        ])
        .arg(ts_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| AeError::Encode {
            path: ts_path.to_path_buf(),
            message: format!("failed spawning slot mpegts encoder: {e}"),
        })?;
    let mut stdin = child.stdin.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let reader = thread::spawn(move || {
        let mut captured = Vec::new();
        let mut buf = [0; 4096];
        while let Ok(n) = stderr.read(&mut buf) {
            if n == 0 {
                break;
            }
            captured.extend_from_slice(&buf[..n.min(65536usize.saturating_sub(captured.len()))]);
        }
        captured
    });
    let result = stream(&mut |frame| {
        let stretched;
        let bytes = if frame.width == res.width && frame.height == res.height {
            &frame.data
        } else {
            stretched = stretch_frame_lanczos(&frame, res)?;
            &stretched.data
        };
        stdin.write_all(bytes).map_err(|e| AeError::Encode {
            path: ts_path.to_path_buf(),
            message: format!("failed writing slot frame: {e}"),
        })
    });
    drop(stdin);
    if result.is_err() {
        let _ = child.kill();
    }
    let status = child.wait();
    let stderr = reader.join().unwrap_or_default();
    result?;
    let status = status.map_err(|e| AeError::Encode {
        path: ts_path.to_path_buf(),
        message: format!("failed waiting on slot mpegts encoder: {e}"),
    })?;
    if !status.success() || std::fs::metadata(ts_path).map_or(true, |m| m.len() == 0) {
        return Err(AeError::Encode {
            path: ts_path.to_path_buf(),
            message: format!(
                "slot mpegts encoder failed: {}",
                String::from_utf8_lossy(&stderr).trim()
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ae_core::{TimelineSlot, TransitionSpec};
    use ae_media::{FfmpegBackend, generate_synthetic_multi_shot_video};

    #[test]
    #[ignore = "downloads the 86 MB model on first use"]
    fn enhanced_export_matches_ai_restoration_without_changing_size_or_frame_count() {
        let tmp = tempfile::tempdir().unwrap();
        let resolution = Resolution::new(64, 36);
        let fps = Rational::from_secs(30);
        let duration = Rational::from_frame_index(2, fps);
        let backend = FfmpegBackend::new();
        let source = tmp.path().join("source.mp4");
        let mut writer = backend
            .create_writer(VideoWriterConfig {
                output_path: source.clone(),
                resolution,
                fps,
                crf: 0,
                ..Default::default()
            })
            .unwrap();
        let mut frame = RgbaFrame::solid(64, 36, [100, 120, 140, 255], Rational::ZERO);
        for (i, p) in frame.data.chunks_exact_mut(4).enumerate() {
            for c in &mut p[..3] {
                *c += ((i * 31 + i / 64 * 17) % 41) as u8;
            }
        }
        writer.write_frame(&frame).unwrap();
        writer.write_frame(&frame).unwrap();
        writer.finish().unwrap();
        let timeline = Timeline {
            version: 1,
            source_video: source.to_string_lossy().into_owned(),
            source_audio: String::new(),
            resolution,
            fps,
            total_duration: duration,
            mean_boundary_similarity: 1.0,
            slots: vec![TimelineSlot {
                slot_index: 0,
                dst_start: Rational::ZERO,
                dst_end: duration,
                shot_id: 0,
                src_in: Rational::ZERO,
                src_out: duration,
                speed: 1.0,
                transition_in: TransitionSpec::default(),
                boundary_similarity: 1.0,
                beat_strength: 1.0,
                locked: false,
            }],
        };
        let output = tmp.path().join("enhanced.mp4");
        render_timeline(
            &backend,
            &timeline,
            &RenderConfig {
                output_path: output.clone(),
                quality_enhance: true,
                crf: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let original = backend
            .decode_segment(
                &source,
                Rational::ZERO,
                duration,
                resolution,
                fps,
                1.0,
                None,
            )
            .unwrap();
        let expected = enhance_quality(&original[0]).unwrap();
        let rendered = backend
            .decode_segment(
                &output,
                Rational::ZERO,
                duration,
                resolution,
                fps,
                1.0,
                None,
            )
            .unwrap();
        assert_eq!(rendered.len(), 2);
        assert_eq!((rendered[0].width, rendered[0].height), (64, 36));
        let difference = |a: &RgbaFrame, b: &RgbaFrame| -> f64 {
            a.data
                .iter()
                .zip(&b.data)
                .map(|(a, b)| a.abs_diff(*b) as f64)
                .sum::<f64>()
                / a.data.len() as f64
        };
        assert!(
            difference(&expected, &rendered[0]) < 3.0,
            "export differs from AI restoration"
        );
        assert!(
            difference(&expected, &original[0]) > 1.0,
            "enhancer did not change the source"
        );
    }

    #[test]
    fn test_error_diffusion_zero_drift_over_5_minutes() {
        let fps = Rational::from_secs(30);
        // 600 slots of 0.5123s each (~5.12 minutes)
        let mut slots = Vec::new();
        for i in 0..600 {
            let s = Rational::from_secs_f64((i as f64) * 0.5123);
            let e = Rational::from_secs_f64(((i + 1) as f64) * 0.5123);
            slots.push(TimelineSlot {
                slot_index: i,
                dst_start: s,
                dst_end: e,
                shot_id: 0,
                src_in: Rational::ZERO,
                src_out: e - s,
                speed: 1.0,
                transition_in: TransitionSpec::default(),
                boundary_similarity: 1.0,
                beat_strength: 0.8,
                locked: false,
            });
        }
        let tl = Timeline {
            version: 1,
            source_video: String::new(),
            source_audio: String::new(),
            resolution: Resolution::P720,
            fps,
            total_duration: slots.last().unwrap().dst_end,
            mean_boundary_similarity: 1.0,
            slots,
        };

        let counts = compute_frame_accurate_slot_lengths(&tl, fps);
        let mut cum = 0usize;
        for (i, &c) in counts.iter().enumerate() {
            cum += c;
            let actual_end_sec = (cum as f64) / 30.0;
            let ideal_end_sec = tl.slots[i].dst_end.as_secs_f64();
            let err_frames = (actual_end_sec - ideal_end_sec).abs() * 30.0;
            assert!(
                err_frames <= 0.501,
                "slot {i} drifted by {err_frames:.3} frames (> 0.5 frame)"
            );
        }
    }

    #[test]
    fn test_render_holds_source_end_until_the_timeline_transition() {
        for source_fps in [Rational::from_secs(30), Rational::new(30_000, 1_001)] {
            let tmp = tempfile::tempdir().unwrap();
            let source = tmp.path().join("source_bounds.mp4");
            let fps = Rational::from_secs(30);
            let resolution = Resolution::new(64, 64);
            generate_synthetic_multi_shot_video(
                &source,
                resolution,
                source_fps,
                &[
                    ([220, 10, 10], [220, 10, 10], 12),
                    ([10, 220, 10], [10, 220, 10], 30),
                    ([10, 10, 220], [10, 10, 220], 30),
                ],
            )
            .unwrap();
            let backend = FfmpegBackend::new();
            let mut timeline = Timeline {
                version: 1,
                source_video: source.to_string_lossy().into_owned(),
                source_audio: String::new(),
                resolution,
                fps,
                total_duration: Rational::from_secs(2),
                mean_boundary_similarity: 1.0,
                slots: vec![
                    TimelineSlot {
                        slot_index: 0,
                        dst_start: Rational::ZERO,
                        dst_end: Rational::ONE,
                        shot_id: 0,
                        src_in: Rational::ZERO,
                        src_out: Rational::from_frame_index(12, source_fps),
                        speed: 1.0,
                        transition_in: TransitionSpec::default(),
                        boundary_similarity: 1.0,
                        beat_strength: 1.0,
                        locked: false,
                    },
                    TimelineSlot {
                        slot_index: 1,
                        dst_start: Rational::ONE,
                        dst_end: Rational::from_secs(2),
                        shot_id: 2,
                        src_in: Rational::from_frame_index(42, source_fps),
                        src_out: Rational::from_frame_index(72, source_fps),
                        speed: 1.0,
                        transition_in: TransitionSpec {
                            id: "crossfade".into(),
                            duration: Rational::from_millis(200),
                            easing: "linear".into(),
                            params: toml::Value::Table(toml::map::Map::new()),
                        },
                        boundary_similarity: 1.0,
                        beat_strength: 1.0,
                        locked: false,
                    },
                ],
            };
            for speed in [0.8, 1.0, 1.25] {
                timeline.slots[1].transition_in.easing = "linear".into();
                timeline.slots[0].speed = speed;
                timeline.slots[1].src_out = Rational::from_frame_index(72, source_fps);
                let config = RenderConfig {
                    video_path: source.clone(),
                    output_path: tmp.path().join("bounded.mp4"),
                    resolution: Some(resolution),
                    ..Default::default()
                };
                render_timeline(&backend, &timeline, &config).unwrap();
                let cache = tmp.path().join(format!("cache_{speed}"));
                let draft = RenderConfig {
                    output_path: tmp.path().join("bounded_draft.mp4"),
                    ..config.clone()
                };
                render_timeline_incremental_draft(&backend, &timeline, &draft, &cache, 1).unwrap();
                for path in [&config.output_path, &draft.output_path] {
                    let frames = backend
                        .decode_segment(
                            path,
                            Rational::ZERO,
                            Rational::from_secs(2),
                            resolution,
                            fps,
                            1.0,
                            None,
                        )
                        .unwrap();
                    assert_eq!(frames.len(), 60);
                    let middle = frames[32].pixel(32, 32);
                    assert!(
                        middle[0] > 35 && middle[2] > 35,
                        "crossfade missing at speed {speed}: {middle:?}"
                    );
                    for (i, frame) in frames.iter().enumerate() {
                        let pixel = frame.pixel(32, 32);
                        assert!(
                            pixel[1] < 70,
                            "unselected green scene leaked into frame {i} at speed {speed}: {pixel:?}"
                        );
                        if i < 30 {
                            assert!(
                                pixel[0] > 150,
                                "outgoing shot changed before its transition"
                            );
                        }
                        if i >= 36 {
                            assert!(pixel[2] > 150, "incoming shot was not shown");
                        }
                    }
                }
                timeline.slots[1].src_out = Rational::from_frame_index(69, source_fps);
                let report =
                    render_timeline_incremental_draft(&backend, &timeline, &draft, &cache, 1)
                        .unwrap();
                assert_eq!(
                    report.encoded_slots, 1,
                    "changing only src_out must invalidate the slot cache"
                );
                assert_eq!(report.cached_slots, 1);
                timeline.slots[1].transition_in.easing = "quad_in".into();
                let report =
                    render_timeline_incremental_draft(&backend, &timeline, &draft, &cache, 1)
                        .unwrap();
                assert_eq!(
                    (report.cached_slots, report.encoded_slots),
                    (1, 1),
                    "changing transition easing must invalidate its slot"
                );
                timeline.slots[0].speed = speed + 0.1;
                let report =
                    render_timeline_incremental_draft(&backend, &timeline, &draft, &cache, 1)
                        .unwrap();
                assert_eq!(
                    (report.cached_slots, report.encoded_slots),
                    (0, 2),
                    "changing the outgoing clip must invalidate its transition"
                );
            }
        }
    }

    #[test]
    fn test_render_stretched_resolution_and_verify_output() {
        let tmp = tempfile::tempdir().unwrap();
        let src_vid = tmp.path().join("src.mp4");
        let out_vid = tmp.path().join("out_vertical.mp4");
        let fps = Rational::from_secs(30);

        generate_synthetic_multi_shot_video(
            &src_vid,
            Resolution::new(320, 180),
            fps,
            &[
                ([220, 40, 40], [20, 20, 20], 20),
                ([40, 220, 40], [20, 20, 20], 20),
            ],
        )
        .unwrap();

        let tl = Timeline {
            version: 1,
            source_video: src_vid.to_string_lossy().to_string(),
            source_audio: String::new(),
            resolution: Resolution::new(240, 426), // vertical stretched
            fps,
            total_duration: Rational::from_secs(1),
            mean_boundary_similarity: 0.85,
            slots: vec![
                TimelineSlot {
                    slot_index: 0,
                    dst_start: Rational::ZERO,
                    dst_end: Rational::from_millis(500),
                    shot_id: 0,
                    src_in: Rational::ZERO,
                    src_out: Rational::from_millis(500),
                    speed: 1.0,
                    transition_in: TransitionSpec::default(),
                    boundary_similarity: 1.0,
                    beat_strength: 0.8,
                    locked: false,
                },
                TimelineSlot {
                    slot_index: 1,
                    dst_start: Rational::from_millis(500),
                    dst_end: Rational::from_secs(1),
                    shot_id: 1,
                    src_in: Rational::from_millis(700),
                    src_out: Rational::from_millis(1200),
                    speed: 1.0,
                    transition_in: TransitionSpec {
                        id: "crossfade".into(),
                        duration: Rational::from_millis(100),
                        easing: "cubic_in_out".into(),
                        params: toml::Value::Table(toml::map::Map::new()),
                    },
                    boundary_similarity: 0.85,
                    beat_strength: 0.9,
                    locked: false,
                },
            ],
        };

        let backend = FfmpegBackend::new();
        render_timeline(
            &backend,
            &tl,
            &RenderConfig {
                video_path: src_vid,
                output_path: out_vid.clone(),
                resolution: Some(Resolution::new(240, 426)),
                use_gpu: true,
                ..Default::default()
            },
        )
        .unwrap();

        let probe = backend.probe_video(&out_vid).unwrap();
        assert_eq!(probe.width, 240);
        assert_eq!(probe.height, 426);
        assert_eq!(probe.sar_num, 1);
        assert_eq!(probe.sar_den, 1);
    }

    #[test]
    fn test_quick_render_hard_cut_speed_padding_and_failed_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source.mp4");
        let cache = tmp.path().join("cache");
        let fps = Rational::from_secs(10);
        let resolution = Resolution::new(64, 64);
        let draft_resolution = Resolution::new(96, 80);
        let backend = FfmpegBackend::new();
        let timeline = Timeline {
            version: 1,
            source_video: source.to_string_lossy().into_owned(),
            source_audio: String::new(),
            resolution,
            fps,
            total_duration: Rational::ONE,
            mean_boundary_similarity: 1.0,
            slots: vec![TimelineSlot {
                slot_index: 0,
                dst_start: Rational::ZERO,
                dst_end: Rational::ONE,
                shot_id: 0,
                src_in: Rational::from_millis(500),
                src_out: Rational::from_millis(1500),
                speed: 2.0,
                transition_in: TransitionSpec::default(),
                boundary_similarity: 1.0,
                beat_strength: 1.0,
                locked: false,
            }],
        };
        let config = RenderConfig {
            output_path: tmp.path().join("draft.mp4"),
            resolution: Some(draft_resolution),
            ..Default::default()
        };
        for _ in 0..2 {
            assert!(
                render_timeline_incremental_draft(&backend, &timeline, &config, &cache, 1).is_err()
            );
            assert_eq!(
                std::fs::read_dir(cache.join("draft_slots"))
                    .unwrap()
                    .count(),
                0
            );
        }
        let mut writer = backend
            .create_writer(VideoWriterConfig {
                output_path: source.clone(),
                resolution,
                fps,
                crf: 0,
                ..Default::default()
            })
            .unwrap();
        for i in 0..25 {
            let color = if i < 5 {
                [0, 0, 220, 255]
            } else if i < 15 {
                [220, (i - 5) as u8 * 20, 0, 255]
            } else {
                [0, 0, 220, 255]
            };
            writer
                .write_frame(&RgbaFrame::solid(
                    64,
                    64,
                    color,
                    Rational::from_frame_index(i, fps),
                ))
                .unwrap();
        }
        writer.finish().unwrap();
        let report =
            render_timeline_incremental_draft(&backend, &timeline, &config, &cache, 1).unwrap();
        assert_eq!((report.cached_slots, report.encoded_slots), (0, 1));
        let probe = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_read_frames",
                "-of",
                "csv=p=0",
            ])
            .arg(&config.output_path)
            .output()
            .unwrap();
        assert!(probe.status.success());
        assert_eq!(String::from_utf8_lossy(&probe.stdout).trim(), "96,80,10");
        let expected = backend
            .decode_segment(
                &source,
                timeline.slots[0].src_in,
                Rational::ONE,
                draft_resolution,
                fps,
                2.0,
                Some(timeline.slots[0].src_out),
            )
            .unwrap();
        let actual = backend
            .decode_segment(
                &config.output_path,
                Rational::ZERO,
                Rational::ONE,
                draft_resolution,
                fps,
                1.0,
                None,
            )
            .unwrap();
        for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            let pixel = actual.pixel(32, 32);
            let reference = expected.pixel(32, 32);
            assert!(
                pixel[0] > 180 && pixel[2] < 40,
                "scene leakage at frame {i}: {pixel:?}"
            );
            assert!(
                pixel[1].abs_diff(reference[1]) < 25,
                "speed mismatch at frame {i}: {pixel:?} vs {reference:?}"
            );
        }
        assert!(actual[4].pixel(32, 32)[1] > actual[0].pixel(32, 32)[1] + 70);
        assert!(actual[9].pixel(32, 32)[1].abs_diff(actual[4].pixel(32, 32)[1]) < 10);
        let mut empty = timeline.clone();
        empty.slots[0].src_in = Rational::from_secs(4);
        empty.slots[0].src_out = Rational::from_secs(5);
        for _ in 0..2 {
            assert!(
                render_timeline_incremental_draft(&backend, &empty, &config, &cache, 1).is_err()
            );
        }
        assert_eq!(
            std::fs::read_dir(cache.join("draft_slots"))
                .unwrap()
                .filter_map(|entry| {
                    let path = entry.unwrap().path();
                    (path.extension().is_some_and(|ext| ext == "ts")).then_some(path)
                })
                .count(),
            1
        );
    }

    #[test]
    fn quick_render_parallel_streams_and_deduplicates_in_order() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct StreamingBackend {
            active: AtomicUsize,
            peak: AtomicUsize,
            calls: AtomicUsize,
            retried: std::sync::atomic::AtomicBool,
        }
        impl MediaBackend for StreamingBackend {
            fn probe_video(&self, _: &Path) -> Result<ae_media::VideoProbeInfo> {
                unreachable!()
            }
            fn probe_audio(&self, _: &Path) -> Result<ae_media::AudioProbeInfo> {
                unreachable!()
            }
            fn decode_scaled_stream(
                &self,
                _: &Path,
                _: u32,
                _: u32,
                _: Option<Rational>,
                _: crossbeam_channel::Sender<RgbaFrame>,
            ) -> Result<()> {
                unreachable!()
            }
            fn decode_segment(
                &self,
                _: &Path,
                _: Rational,
                _: Rational,
                _: Resolution,
                _: Rational,
                _: f64,
                _: Option<Rational>,
            ) -> Result<Vec<RgbaFrame>> {
                panic!("must stream")
            }
            fn decode_segment_stream(
                &self,
                _: &Path,
                start: Rational,
                duration: Rational,
                res: Resolution,
                fps: Rational,
                _: f64,
                _: Option<Rational>,
                emit: &mut dyn FnMut(RgbaFrame) -> Result<()>,
            ) -> Result<()> {
                let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
                self.peak.fetch_max(active, Ordering::SeqCst);
                self.calls.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(80));
                let count = (duration.as_secs_f64() * fps.as_secs_f64()).round() as usize;
                let color = if start.as_secs_f64() < 15.0 {
                    [220, 0, 0, 255]
                } else {
                    [0, 0, 220, 255]
                };
                let result = if (start.as_secs_f64() - 9.9).abs() < 0.001
                    && !self.retried.swap(true, Ordering::SeqCst)
                {
                    emit(RgbaFrame::solid(
                        res.width,
                        res.height,
                        [0, 220, 0, 255],
                        start,
                    ))?;
                    Err(AeError::Config("test late decode failure".into()))
                } else {
                    (0..count).try_for_each(|i| {
                        emit(RgbaFrame::solid(
                            res.width,
                            res.height,
                            color,
                            Rational::from_frame_index(i as i64, fps),
                        ))
                    })
                };
                self.active.fetch_sub(1, Ordering::SeqCst);
                result
            }
            fn run_scdet_baseline(&self, _: &Path, _: f32) -> Result<Vec<Rational>> {
                unreachable!()
            }
            fn create_writer(
                &self,
                _: VideoWriterConfig,
            ) -> Result<Box<dyn ae_media::VideoWriter>> {
                unreachable!()
            }
            fn passthrough_remux(
                &self,
                _: &Path,
                _: &Path,
                _: &Path,
                _: Option<Resolution>,
                _: VideoCodec,
            ) -> Result<()> {
                unreachable!()
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source.mp4");
        let fps = Rational::from_secs(10);
        let res = Resolution::new(64, 64);
        generate_synthetic_multi_shot_video(&source, res, fps, &[([220, 0, 0], [220, 0, 0], 10)])
            .unwrap();
        let mut slots = Vec::new();
        for i in 0..5 {
            let shot = if i == 4 { 0 } else { i };
            slots.push(TimelineSlot {
                slot_index: i,
                dst_start: Rational::from_secs(i as i64 * 10),
                dst_end: Rational::from_secs((i + 1) as i64 * 10),
                shot_id: shot as _,
                src_in: Rational::from_secs(shot as i64 * 10),
                src_out: Rational::from_secs(shot as i64 * 10 + 10),
                speed: 1.0,
                transition_in: if i == 0 || i == 4 {
                    TransitionSpec::default()
                } else {
                    TransitionSpec {
                        id: "crossfade".into(),
                        duration: Rational::from_secs(5),
                        easing: "linear".into(),
                        params: toml::Value::Table(toml::map::Map::new()),
                    }
                },
                boundary_similarity: 1.0,
                beat_strength: 1.0,
                locked: false,
            });
        }
        let timeline = Timeline {
            version: 1,
            source_video: source.to_string_lossy().into_owned(),
            source_audio: String::new(),
            resolution: res,
            fps,
            total_duration: Rational::from_secs(50),
            mean_boundary_similarity: 1.0,
            slots,
        };
        let progress = Arc::new(std::sync::Mutex::new(Vec::new()));
        let updates = progress.clone();
        let config = RenderConfig {
            output_path: tmp.path().join("draft.mp4"),
            progress_cb: Some(Arc::new(move |pct, _| updates.lock().unwrap().push(pct))),
            ..Default::default()
        };
        let backend = StreamingBackend {
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
            retried: std::sync::atomic::AtomicBool::new(false),
        };
        let report = render_timeline_incremental_draft(
            &backend,
            &timeline,
            &config,
            &tmp.path().join("cache"),
            1,
        )
        .unwrap();
        assert_eq!((report.encoded_slots, report.cached_slots), (4, 1));
        assert!(backend.peak.load(Ordering::SeqCst) >= 2);
        assert!(backend.peak.load(Ordering::SeqCst) <= 4);
        assert_eq!(backend.calls.load(Ordering::SeqCst), 7);
        assert!(
            std::fs::read_dir(tmp.path().join("cache/draft_slots"))
                .unwrap()
                .all(|entry| !entry.unwrap().path().to_string_lossy().ends_with(".tmp"))
        );
        let updates = progress.lock().unwrap();
        assert!(updates.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(updates.iter().filter(|&&pct| pct == 80).count(), 1);
        let frames = FfmpegBackend::new()
            .decode_segment(
                &config.output_path,
                Rational::ZERO,
                Rational::from_secs(50),
                res,
                fps,
                1.0,
                None,
            )
            .unwrap();
        assert_eq!(frames.len(), 500);
        assert!(
            frames[100].pixel(32, 32)[1] < 60,
            "failed outgoing decode leaked into the retry"
        );
        for (index, red) in [
            (20, true),
            (170, true),
            (270, false),
            (370, false),
            (470, true),
        ] {
            let pixel = frames[index].pixel(32, 32);
            assert!(
                if red {
                    pixel[0] > 150 && pixel[2] < 60
                } else {
                    pixel[2] > 150 && pixel[0] < 60
                },
                "frame {index}: {pixel:?}"
            );
        }
    }

    #[test]
    fn test_incremental_draft_render_reuses_unchanged_slots() {
        let tmp = tempfile::tempdir().unwrap();
        let src_vid = tmp.path().join("src_draft.mp4");
        let out_draft = tmp.path().join("draft.mp4");
        let cache_dir = tmp.path().join("cache");
        let fps = Rational::from_secs(30);

        generate_synthetic_multi_shot_video(
            &src_vid,
            Resolution::new(320, 180),
            fps,
            &[
                ([220, 40, 40], [20, 20, 20], 20),
                ([40, 220, 40], [20, 20, 20], 20),
                ([40, 60, 220], [20, 20, 20], 20),
            ],
        )
        .unwrap();

        let mut tl = Timeline {
            version: 1,
            source_video: src_vid.to_string_lossy().to_string(),
            source_audio: String::new(),
            resolution: Resolution::new(360, 640),
            fps,
            total_duration: Rational::from_secs(1),
            mean_boundary_similarity: 0.9,
            slots: vec![
                TimelineSlot {
                    slot_index: 0,
                    dst_start: Rational::ZERO,
                    dst_end: Rational::from_millis(500),
                    shot_id: 0,
                    src_in: Rational::ZERO,
                    src_out: Rational::from_millis(500),
                    speed: 1.0,
                    transition_in: TransitionSpec::default(),
                    boundary_similarity: 1.0,
                    beat_strength: 0.8,
                    locked: false,
                },
                TimelineSlot {
                    slot_index: 1,
                    dst_start: Rational::from_millis(500),
                    dst_end: Rational::from_secs(1),
                    shot_id: 1,
                    src_in: Rational::from_millis(700),
                    src_out: Rational::from_millis(1200),
                    speed: 1.0,
                    transition_in: TransitionSpec::default(),
                    boundary_similarity: 0.9,
                    beat_strength: 0.8,
                    locked: false,
                },
            ],
        };

        let backend = FfmpegBackend::new();
        let cfg = RenderConfig {
            video_path: src_vid,
            output_path: out_draft.clone(),
            resolution: Some(Resolution::new(360, 640)),
            ..Default::default()
        };

        // First run: 0 cached, 2 encoded
        let r1 = render_timeline_incremental_draft(&backend, &tl, &cfg, &cache_dir, 2).unwrap();
        assert_eq!(r1.cached_slots, 0);
        assert_eq!(r1.encoded_slots, 2);
        assert!(out_draft.exists());

        // Modify only Slot #1 (swap Shot #1 -> Shot #2): Slot #0 must be a cache hit!
        tl.slots[1].shot_id = 2;
        tl.slots[1].src_in = Rational::from_millis(1400);
        tl.slots[1].src_out = Rational::from_millis(1900);

        let r2 = render_timeline_incremental_draft(&backend, &tl, &cfg, &cache_dir, 2).unwrap();
        assert_eq!(r2.cached_slots, 1, "unchanged Slot #0 must be cached!");
        assert_eq!(
            r2.encoded_slots, 1,
            "only changed Slot #1 should be encoded!"
        );
    }

    #[test]
    fn test_incremental_draft_reframe_toggle_invalidates_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let src_vid = tmp.path().join("src_reframe.mp4");
        let out_draft = tmp.path().join("draft.mp4");
        let cache_dir = tmp.path().join("cache");
        let fps = Rational::from_secs(30);

        generate_synthetic_multi_shot_video(
            &src_vid,
            Resolution::new(320, 180),
            fps,
            &[
                ([220, 40, 40], [20, 20, 20], 20),
                ([40, 220, 40], [20, 20, 20], 20),
                ([40, 60, 220], [20, 20, 20], 20),
            ],
        )
        .unwrap();

        // Second slot joins with a crossfade so both the hard-cut fast path
        // and the transition-blend path render reframed.
        let tl = Timeline {
            version: 1,
            source_video: src_vid.to_string_lossy().to_string(),
            source_audio: String::new(),
            resolution: Resolution::new(240, 400),
            fps,
            total_duration: Rational::from_secs(1),
            mean_boundary_similarity: 0.9,
            slots: vec![
                TimelineSlot {
                    slot_index: 0,
                    dst_start: Rational::ZERO,
                    dst_end: Rational::from_millis(500),
                    shot_id: 0,
                    src_in: Rational::ZERO,
                    src_out: Rational::from_millis(500),
                    speed: 1.0,
                    transition_in: TransitionSpec::default(),
                    boundary_similarity: 1.0,
                    beat_strength: 0.8,
                    locked: false,
                },
                TimelineSlot {
                    slot_index: 1,
                    dst_start: Rational::from_millis(500),
                    dst_end: Rational::from_secs(1),
                    shot_id: 1,
                    src_in: Rational::from_millis(700),
                    src_out: Rational::from_millis(1200),
                    speed: 1.0,
                    transition_in: TransitionSpec {
                        id: "crossfade".into(),
                        duration: Rational::from_millis(100),
                        easing: "cubic_in_out".into(),
                        params: toml::Value::Table(toml::map::Map::new()),
                    },
                    boundary_similarity: 0.9,
                    beat_strength: 0.8,
                    locked: false,
                },
            ],
        };

        let backend = FfmpegBackend::new();
        // Baseline without reframing: both slots encode once, then hit cache.
        let plain = RenderConfig {
            video_path: src_vid.clone(),
            output_path: out_draft.clone(),
            resolution: Some(Resolution::new(240, 400)),
            ..Default::default()
        };
        let p1 = render_timeline_incremental_draft(&backend, &tl, &plain, &cache_dir, 2).unwrap();
        assert_eq!((p1.cached_slots, p1.encoded_slots), (0, 2));
        let p2 = render_timeline_incremental_draft(&backend, &tl, &plain, &cache_dir, 2).unwrap();
        assert_eq!((p2.cached_slots, p2.encoded_slots), (2, 0));

        // Enabling reframe (no model file => centered fallback) must not reuse
        // the stretched segments: everything re-encodes at the same slot count.
        let reframed = RenderConfig {
            video_path: src_vid,
            output_path: out_draft.clone(),
            resolution: Some(Resolution::new(240, 400)),
            reframe_faces: true,
            face_model_path: Some(tmp.path().join("missing-yunet.onnx")),
            ..Default::default()
        };
        let r1 =
            render_timeline_incremental_draft(&backend, &tl, &reframed, &cache_dir, 2).unwrap();
        assert_eq!((r1.cached_slots, r1.encoded_slots), (0, 2));
        assert!(out_draft.exists());
        let r2 =
            render_timeline_incremental_draft(&backend, &tl, &reframed, &cache_dir, 2).unwrap();
        assert_eq!((r2.cached_slots, r2.encoded_slots), (2, 0));

        // Reframed output keeps the draft resolution and plays the full slot.
        let frames = backend
            .decode_segment(
                &out_draft,
                Rational::ZERO,
                Rational::from_secs(1),
                Resolution::new(240, 400),
                fps,
                1.0,
                None,
            )
            .unwrap();
        assert_eq!(frames.len(), 30);
        assert!(frames.iter().all(|f| (f.width, f.height) == (240, 400)));
    }
}
