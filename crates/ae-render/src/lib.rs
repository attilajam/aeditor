use std::fmt;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::thread;

use ae_core::{AeError, Rational, Resolution, Result, RgbaFrame, Timeline, VideoCodec};
use ae_media::{MediaBackend, VideoWriterConfig, stretch_frame_lanczos};
use ae_transitions::{TransitionCtx, build_transition};
use ae_transitions_gpu::GpuTransitionEngine;
use crossbeam_channel::bounded;
use lru::LruCache;
use sha2::{Digest, Sha256};

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

    // Pre-decode slots with an LRU cache so reused shots or overlapping handles are cached
    let cap = NonZeroUsize::new(config.cache_capacity.max(2)).unwrap();
    let mut seg_cache: LruCache<(usize, i64, usize, u32, u32), Vec<RgbaFrame>> = LruCache::new(cap);

    // For each slot, also decode any trailing handle frames needed by the NEXT slot's transition
    let mut decoded_slots: Vec<Vec<RgbaFrame>> = Vec::with_capacity(timeline.slots.len());

    let total_slots = timeline.slots.len().max(1);
    for (idx, slot) in timeline.slots.iter().enumerate() {
        if let Some(ref cb) = config.progress_cb {
            let pct = 10 + ((idx * 45) / total_slots) as u32;
            cb(
                pct,
                &format!("Decoding slot {}/{} (Scene #{})...", idx + 1, total_slots, slot.shot_id),
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
            (slot.src_in.as_secs_f64() * 1_000.0).round() as i64,
            needed_frames,
            target_res.width,
            target_res.height,
        );

        let frames = if let Some(cached) = seg_cache.get(&cache_key) {
            cached.clone()
        } else {
            let mut seg = backend.decode_segment(
                &video_path,
                slot.src_in,
                needed_dur,
                target_res,
                fps,
                slot.speed,
            )?;
            // Ensure exact stretch-to-fill target resolution via SIMD Lanczos3
            for f in &mut seg {
                if f.width != target_res.width || f.height != target_res.height {
                    *f = stretch_frame_lanczos(f, target_res)?;
                }
            }
            seg_cache.put(cache_key, seg.clone());
            seg
        };

        decoded_slots.push(frames);
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
    };

    let mut writer = backend.create_writer(writer_cfg)?;
    let (tx, rx) = bounded::<RgbaFrame>(32);

    thread::scope(|s| -> Result<()> {
        let encode_handle = s.spawn(move || -> Result<()> {
            while let Ok(frame) = rx.recv() {
                writer.write_frame(&frame)?;
            }
            writer.finish()
        });

        let mut out_frame_idx = 0i64;
        for (idx, slot) in timeline.slots.iter().enumerate() {
            if let Some(ref cb) = config.progress_cb {
                let pct = 55 + (((idx + 1) * 43) / total_slots) as u32;
                cb(
                    pct,
                    &format!("Encoding MP4 slot {}/{} ({})...", idx + 1, total_slots, target_res),
                );
            }
            let slot_len = slot_frame_counts[idx];
            let cur_frames = &decoded_slots[idx];

            let trans_frames = if idx > 0 && slot.transition_in.id != "cut" {
                ((slot.transition_in.duration.as_secs_f64() * fps.as_secs_f64()).round() as usize)
                    .min(slot_len / 2)
                    .min(decoded_slots[idx - 1].len().saturating_sub(slot_frame_counts[idx - 1]))
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
/// instantaneously (`0 ms`), changed slots are encoded with `libx264 -preset ultrafast`, and all
/// slots are stitched without re-encoding via `ffmpeg -f concat -c:v copy`.
pub fn render_timeline_incremental_draft(
    backend: &dyn MediaBackend,
    timeline: &Timeline,
    config: &RenderConfig,
    cache_dir: &Path,
    scale_divisor: u32,
) -> Result<IncrementalDraftReport> {
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

    let slots_dir = cache_dir.join("draft_slots");
    std::fs::create_dir_all(&slots_dir)?;

    let total_slots = timeline.slots.len();
    let mut cached_slots = 0usize;
    let mut encoded_slots = 0usize;
    let mut segment_paths: Vec<PathBuf> = Vec::with_capacity(total_slots);

    for (idx, slot) in timeline.slots.iter().enumerate() {
        let frames_count = slot_frame_counts[idx];
        let mut hasher = Sha256::new();
        hasher.update(video_path.to_string_lossy().as_bytes());
        hasher.update(b"|");
        hasher.update(slot.shot_id.to_le_bytes());
        hasher.update(slot.src_in.numer().to_le_bytes());
        hasher.update(slot.src_in.denom().to_le_bytes());
        hasher.update((frames_count as u64).to_le_bytes());
        hasher.update(draft_res.width.to_le_bytes());
        hasher.update(draft_res.height.to_le_bytes());
        hasher.update(fps.numer().to_le_bytes());
        hasher.update(fps.denom().to_le_bytes());
        hasher.update(((slot.speed * 1000.0).round() as i64).to_le_bytes());
        hasher.update(slot.transition_in.id.as_bytes());
        hasher.update(slot.transition_in.duration.numer().to_le_bytes());
        if idx > 0 && slot.transition_in.id != "cut" {
            let prev = &timeline.slots[idx - 1];
            hasher.update(prev.shot_id.to_le_bytes());
            hasher.update(prev.src_out.numer().to_le_bytes());
        }
        let hash_hex = hex::encode(hasher.finalize());
        let ts_path = slots_dir.join(format!("slot_{hash_hex}.ts"));

        if ts_path.exists() {
            cached_slots += 1;
            if let Some(ref cb) = config.progress_cb {
                let pct = 10 + (((idx + 1) * 70) / total_slots.max(1)) as u32;
                cb(
                    pct,
                    &format!(
                        "Quick Render: slot {}/{} cached (Scene #{})",
                        idx + 1,
                        total_slots,
                        slot.shot_id
                    ),
                );
            }
            segment_paths.push(ts_path);
            continue;
        }

        encoded_slots += 1;
        if let Some(ref cb) = config.progress_cb {
            let pct = 10 + (((idx + 1) * 70) / total_slots.max(1)) as u32;
            cb(
                pct,
                &format!(
                    "Quick Render: encoding changed slot {}/{} (Scene #{})...",
                    idx + 1,
                    total_slots,
                    slot.shot_id
                ),
            );
        }

        let slot_dur = Rational::from_frame_index(frames_count as i64, fps);
        let frames = backend.decode_segment(
            &video_path,
            slot.src_in,
            slot_dur,
            draft_res,
            fps,
            slot.speed,
        )?;

        // Encode this single slot directly to MPEG-TS (`ultrafast`) so it can be concat-copied
        encode_slot_mpegts(&ts_path, &frames, draft_res, fps)?;
        segment_paths.push(ts_path);
    }

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

    cmd.args(["-movflags", "+faststart"]).arg(&config.output_path);

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

fn encode_slot_mpegts(
    ts_path: &Path,
    frames: &[RgbaFrame],
    res: Resolution,
    fps: Rational,
) -> Result<()> {
    use std::io::Write;
    use std::process::Stdio;

    let mut child = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
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
            "setsar=1:1,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-crf",
            "26",
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

    if let Some(mut stdin) = child.stdin.take() {
        for frame in frames {
            let stretched;
            let bytes = if frame.width == res.width && frame.height == res.height {
                &frame.data
            } else {
                stretched = stretch_frame_lanczos(frame, res)?;
                &stretched.data
            };
            stdin.write_all(bytes).map_err(|e| AeError::Encode {
                path: ts_path.to_path_buf(),
                message: format!("failed writing slot frame: {e}"),
            })?;
        }
    }

    let out = child.wait_with_output().map_err(|e| AeError::Encode {
        path: ts_path.to_path_buf(),
        message: format!("failed waiting on slot mpegts encoder: {e}"),
    })?;
    if !out.status.success() {
        return Err(AeError::Encode {
            path: ts_path.to_path_buf(),
            message: format!(
                "slot mpegts encoder failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
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
        assert_eq!(r2.encoded_slots, 1, "only changed Slot #1 should be encoded!");
    }
}
