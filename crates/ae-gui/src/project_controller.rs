use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::thread;

use ae_beats::write_debug_artifacts;
use ae_core::{CacheManager, Rational, Resolution, Result, RgbaFrame, Shot, Timeline, VideoCodec};
use ae_media::{FfmpegBackend, MediaBackend};
use ae_render::{
    IncrementalDraftReport, RenderConfig, RenderProgressCallback, enhance_video, render_timeline,
    render_timeline_incremental_draft,
};
use ae_shots::{
    ClassicShotDetector, ShotDetectorConfig, detect_shots_classic,
    refine_shot_boundaries_from_frame_file, refine_shot_boundaries_with_progress,
};
use ae_similarity::{ShotDescriptorBank, SimilarityConfig, build_descriptor_bank_from_frames};
use crossbeam_channel::bounded;
use serde::{Deserialize, Serialize};

use crate::beat_editor_model::{BeatEditorModel, BeatViewItem};
use crate::character_analysis::{CharacterAnalysisOutput, build_character_portraits};
use crate::character_edits::{CharacterEdit, CharacterEdits, load_edits, save_edits};
use crate::character_model::{
    CharacterAnalysis, CharacterFilter, CharacterFilterMode, CharacterId, SceneCharacterTags,
    character_matches,
};
use crate::image_provider::{global_media_cache, set_gui_progress};
use crate::shot_pool_model::{ShotPoolModel, ShotViewItem};
use crate::timeline_model::{AlternativeShotItem, TimelineModel, TimelineSlotViewItem};

struct RefinementFrames(PathBuf);

impl RefinementFrames {
    fn new() -> std::io::Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "ae-refinement-{}-{}.rgba",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(_) => return Ok(Self(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
    }
}

impl Drop for RefinementFrames {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[derive(Serialize, Deserialize)]
struct CachedAnalysis {
    shots: Vec<Shot>,
    bank: ShotDescriptorBank,
}

impl CachedAnalysis {
    fn valid(&self, duration: Rational, fps: Rational) -> bool {
        let frame_duration = if fps.as_secs_f64() > 0.0 {
            1.0 / fps.as_secs_f64()
        } else {
            1.0
        };
        let limit = duration.as_secs_f64().max(1.0) + frame_duration;
        let timestamp = |t: Rational| {
            t.denom() > 0
                && t.denom() <= 1_000_000_000
                && t.numer() >= 0
                && t.numer() <= 1_000_000_000_000_000
                && t.as_secs_f64() <= limit
        };
        !self.shots.is_empty()
            && self.bank.by_shot.len() == self.shots.len()
            && self.shots.iter().enumerate().all(|(id, shot)| {
                let times = [shot.start, shot.usable_start, shot.usable_end, shot.end];
                shot.id == id
                    && shot.enabled
                    && !shot.locked
                    && shot.cut_confidence.is_finite()
                    && shot.mean_luma.is_finite()
                    && times.iter().copied().all(timestamp)
                    && times
                        .windows(2)
                        .all(|pair| pair[0].as_secs_f64() <= pair[1].as_secs_f64())
                    && shot.start.as_secs_f64() < shot.end.as_secs_f64()
                    && shot.usable_start.as_secs_f64() < shot.usable_end.as_secs_f64()
                    && (id == 0 || self.shots[id - 1].end.as_secs_f64() <= shot.start.as_secs_f64())
                    && !self.bank.by_shot[id].is_empty()
                    && self.bank.by_shot[id].iter().all(|d| {
                        d.shot_id == id
                            && timestamp(d.pts)
                            && d.oklab_thumb.len() == 192
                            && (d.phash.len() == 5 || d.phash.len() == 8)
                            && d.combined_embedding.len() == 200
                            && d.ssim_thumb_rgba.len() == 32 * 18 * 4
                            && d.oklab_thumb
                                .iter()
                                .chain(d.oklab_stats.iter())
                                .chain(d.motion_vec.iter())
                                .chain(d.combined_embedding.iter())
                                .all(|value| value.is_finite())
                            && d.mean_luma.is_finite()
                            && d.subject_pos.iter().all(|value| value.is_finite())
                    })
            })
            && [
                &self.bank.oklab_z,
                &self.bank.stats_z,
                &self.bank.phash_z,
                &self.bank.motion_z,
            ]
            .iter()
            .all(|z| z.mean.is_finite() && z.std.is_finite())
    }
}

/// Namespace for the GUI scene-analysis cache (shots + `ShotDescriptorBank`).
/// `v2` invalidates pre-continuity entries whose descriptors predate `subject_pos`
/// screen-position continuity and motion-direction compatibility, mirroring the CLI
/// `descriptors_v2_continuity` namespace bump. Without the bump, stale banks still
/// deserialize (serde defaults `subject_pos` to center) but silently neutralize the
/// position-continuity signal for every cut.
pub const GUI_SCENE_ANALYSIS_CACHE_TAG: &str = "gui-scene-analysis-v2";

/// Retired namespace from before the continuity fix; must never collide with a live key.
#[cfg(test)]
const LEGACY_GUI_SCENE_ANALYSIS_CACHE_TAG: &str = "gui-scene-analysis-v1";

fn analysis_params(
    detector: &ShotDetectorConfig,
    similarity: &SimilarityConfig,
    analysis_fps: Rational,
    source_fps: Rational,
    duration: Rational,
    size: (u32, u32),
) -> String {
    format!("{detector:?}|{similarity:?}|{analysis_fps:?}|{source_fps:?}|{duration:?}|{size:?}")
}

fn analysis_key(
    cache: &CacheManager,
    path: &Path,
    detector: &ShotDetectorConfig,
    similarity: &SimilarityConfig,
    analysis_fps: Rational,
    source_fps: Rational,
    duration: Rational,
    size: (u32, u32),
) -> Option<String> {
    cache
        .file_cache_key(
            path,
            GUI_SCENE_ANALYSIS_CACHE_TAG,
            &analysis_params(
                detector,
                similarity,
                analysis_fps,
                source_fps,
                duration,
                size,
            ),
        )
        .ok()
}

fn provisional_shots(frames: &[RgbaFrame], duration: Rational) -> Vec<Shot> {
    let minimum = ShotDetectorConfig::default().min_shot_duration;
    let mut edges = vec![Rational::ZERO];
    for frame in frames {
        if frame.pts - *edges.last().unwrap() >= minimum && duration - frame.pts >= minimum {
            edges.push(frame.pts);
        }
    }
    edges.push(duration);
    edges
        .windows(2)
        .enumerate()
        .map(|(id, range)| Shot {
            id,
            start: range[0],
            end: range[1],
            usable_start: range[0],
            usable_end: range[1],
            cut_confidence: 0.0,
            boundary_kind: if id == 0 {
                ae_core::ShotBoundaryKind::Initial
            } else {
                ae_core::ShotBoundaryKind::HardCut
            },
            mean_luma: 0.5,
            enabled: true,
            locked: false,
        })
        .collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CharacterViewItem {
    pub id: CharacterId,
    pub label: String,
    pub provisional: bool,
    pub mode: CharacterFilterMode,
    pub scene_count: usize,
}

/// Full snapshot of the GUI state whenever an action completes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GuiStateSnapshot {
    pub video_path: String,
    pub audio_path: String,
    pub click_track_path: String,
    pub draft_video_path: String,
    pub resolution: String,
    pub target_width: u32,
    pub target_height: u32,
    pub allow_reuse: bool,
    pub bass_low_hz: f32,
    pub bass_high_hz: f32,
    pub separation_mode: String,
    pub beat_mode: String,
    pub sensitivity: f32,
    pub audio_duration_sec: f64,
    pub timeline_duration_sec: f64,
    pub spectrogram_url: String,
    pub preview_revision: u64,
    pub mean_similarity: f32,
    pub status_message: String,
    #[serde(default)]
    pub characters: Vec<CharacterViewItem>,
    #[serde(default)]
    pub character_filters: Vec<CharacterFilter>,
    #[serde(default)]
    pub character_analysis_error: String,
    #[serde(default)]
    pub character_revision: u64,
    #[serde(default)]
    pub character_analysis_pending: bool,
    pub shots: Vec<ShotViewItem>,
    pub beats: Vec<BeatViewItem>,
    pub slots: Vec<TimelineSlotViewItem>,
}

fn character_models_dir() -> PathBuf {
    let directory = PathBuf::from("models");
    if directory.join("manifest.toml").is_file() {
        directory
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
    }
}

fn shot_has_character(
    shot: &Shot,
    frames: &[RgbaFrame],
    resolution: Resolution,
    detector: &mut dyn ae_render::FaceDetector,
) -> Result<bool> {
    detector.reset();
    let start = frames.partition_point(|frame| frame.pts < shot.usable_start);
    let end = frames.partition_point(|frame| frame.pts < shot.usable_end);
    let frames = &frames[start..end];
    let mut detected = false;
    if !frames.is_empty() {
        // shortcut: sample three proxy frames per scene; sample more for brief appearances.
        for index in [0, frames.len() / 2, frames.len() - 1] {
            let frame = ae_media::stretch_frame_lanczos(&frames[index], resolution)?;
            if detector.detect(&frame).is_some() {
                detected = true;
                break;
            }
        }
    }
    detector.finish_shot()?;
    Ok(detected)
}

/// Top-level GUI controller coordinating the Shot Pool, Bass Beat Editor, Match-Cut Sequencer,
/// Stretched Resolution settings, and Video Exporter.
pub struct ProjectController {
    pub backend: FfmpegBackend,
    pub video_path: Option<PathBuf>,
    pub audio_path: Option<PathBuf>,
    pub click_track_path: Option<PathBuf>,
    pub draft_video_path: Option<PathBuf>,
    pub resolution: Resolution,
    pub quality_enhance: bool,
    pub light_cleanup: bool,
    /// Opt-in full animated-character locking for preview and final export.
    pub reframe_faces: bool,
    character_detections: HashMap<(Rational, Rational), bool>,
    pub character_analysis: Option<CharacterAnalysis>,
    pub character_filters: Vec<CharacterFilter>,
    pub character_analysis_error: String,
    pub character_revision: u64,
    character_base_analysis: Option<CharacterAnalysis>,
    character_edits: CharacterEdits,
    character_source_key: String,
    character_analysis_key: String,
    character_edits_root: PathBuf,
    character_edits_error: String,
    pub fps: Rational,
    pub shot_pool: ShotPoolModel,
    pub beat_editor: BeatEditorModel,
    pub timeline_model: TimelineModel,
    pub preview_revision: u64,
    pub status_message: String,
}

impl Default for ProjectController {
    fn default() -> Self {
        Self::new()
    }
}

impl ProjectController {
    pub fn new() -> Self {
        Self {
            backend: FfmpegBackend::new(),
            video_path: None,
            audio_path: None,
            click_track_path: None,
            draft_video_path: None,
            resolution: Resolution::VERTICAL, // 1080x1920 default stretch target
            quality_enhance: false,
            light_cleanup: false,
            reframe_faces: false,
            character_detections: HashMap::new(),
            character_analysis: None,
            character_filters: Vec::new(),
            character_analysis_error: String::new(),
            character_revision: 0,
            character_base_analysis: None,
            character_edits: CharacterEdits::default(),
            character_source_key: String::new(),
            character_analysis_key: String::new(),
            character_edits_root: PathBuf::from(".ae-project/character-edits"),
            character_edits_error: String::new(),
            fps: Rational::from_secs(30),
            shot_pool: ShotPoolModel::new(),
            beat_editor: BeatEditorModel::new(),
            timeline_model: TimelineModel::new(),
            preview_revision: 1,
            status_message: "Ready — Import a video clip and an audio track to begin.".to_string(),
        }
    }

    /// Unified Single-Pass Streaming Video Ingest:
    /// Streams `path` once from FFmpeg with hardware acceleration (`-hwaccel auto -threads 4`)
    /// at an adaptive frame rate, scanning long videos in parallel without blocking the UI thread.
    pub fn import_video(&mut self, path: &Path) -> Result<()> {
        self.import_video_streaming(path, |_, _| {})
    }

    /// Streaming video ingest that invokes `on_partial_snapshot(snapshot, is_final)` progressively
    /// as frames and scenes are discovered so the GUI stays 100% live and interactive during import.
    pub fn import_video_streaming<F>(&mut self, path: &Path, mut on_update: F) -> Result<()>
    where
        F: FnMut(GuiStateSnapshot, bool),
    {
        set_gui_progress(true, 4, "Probing video clip...");
        let probe = self.backend.probe_video(path)?;
        self.fps = probe.fps;
        self.video_path = Some(path.to_path_buf());
        self.timeline_model.config.source_video = path.to_string_lossy().to_string();
        self.timeline_model.config.fps = probe.fps;
        self.timeline_model.config.resolution = self.resolution;
        self.shot_pool.shots.clear();
        self.character_detections.clear();
        self.character_analysis = None;
        self.character_filters.clear();
        self.character_analysis_error.clear();
        self.character_revision = self.character_revision.wrapping_add(1);
        self.character_base_analysis = None;
        self.character_edits = CharacterEdits::default();
        self.character_source_key.clear();
        self.character_analysis_key.clear();
        self.character_edits_error.clear();
        if let Ok(mut cache) = global_media_cache().write() {
            cache.character_portraits.clear();
            cache.character_revision = self.character_revision;
        }
        self.timeline_model.timeline = None;
        self.draft_video_path = None;
        self.sync_timeline_to_preview_cache();

        let dur_sec = probe.duration.as_secs_f64();
        // Adaptive analysis & proxy FPS so long 20-minute episodes don't decode 36,000 frames
        let ingest_fps = if dur_sec > 300.0 {
            Rational::from_secs(6)
        } else if dur_sec > 60.0 {
            Rational::from_secs(10)
        } else if dur_sec > 8.0 && probe.fps.as_secs_f64() > 15.0 {
            Rational::from_secs(15)
        } else {
            probe.fps
        };
        let expected_frames = ((dur_sec * ingest_fps.as_secs_f64()).round() as usize).max(1);

        // Use 240x136 for long videos (> 60s) and 320x180 for short clips to keep RAM and pipe bandwidth low
        let (ing_w, ing_h) = if dur_sec > 60.0 {
            (240, 136)
        } else {
            (320, 180)
        };

        let detector = ShotDetectorConfig::default();
        let similarity = SimilarityConfig::default();
        let cache = CacheManager::default_local();
        let cache_key = analysis_key(
            &cache,
            path,
            &detector,
            &similarity,
            ingest_fps,
            probe.fps,
            probe.duration,
            (ing_w, ing_h),
        );
        let cached = cache_key
            .as_deref()
            .and_then(|key| cache.get::<CachedAnalysis>(key))
            .filter(|entry| entry.valid(probe.duration, ingest_fps));
        let cache_hit = cached.is_some();
        let mut provisional = false;
        if !cache_hit && dur_sec > 60.0 {
            set_gui_progress(
                true,
                6,
                "Reading keyframes for a provisional scene preview...",
            );
            // Keyframe hints are disposable; unsupported indexes fall back to the accurate scan.
            if let Ok(frames) = self.backend.decode_keyframes(path, ing_w, ing_h) {
                if frames.len() > 1 {
                    self.shot_pool.shots = provisional_shots(&frames, probe.duration);
                    self.populate_thumbnails_from_ram(&frames);
                    self.status_message = format!(
                        "{} provisional scenes — verifying all cuts...",
                        self.shot_pool.shots.len()
                    );
                    on_update(self.snapshot(), true);
                    provisional = true;
                }
            }
        }
        // Capture native thumbnails once; separate seeks are expensive on long-GOP sources.
        let refinement_frames = if !cache_hit && ingest_fps < probe.fps {
            RefinementFrames::new().ok()
        } else {
            None
        };

        set_gui_progress(
            true,
            8,
            if cache_hit {
                "Streaming video frames from cached scene analysis..."
            } else {
                "Streaming video frames & detecting scenes live..."
            },
        );
        let (tx, rx) = bounded::<RgbaFrame>(64);
        let path_owned = path.to_path_buf();
        let backend = self.backend.clone();
        let analysis_path = refinement_frames.as_ref().map(|frames| frames.0.clone());

        let (decoded_frames, small_64) =
            thread::scope(|s| -> Result<(Vec<RgbaFrame>, Vec<RgbaFrame>)> {
                let handle = s.spawn(move || {
                    backend.decode_scaled_stream_parallel(
                        &path_owned,
                        ing_w,
                        ing_h,
                        ingest_fps,
                        analysis_path.as_deref(),
                        tx,
                    )
                });
                let mut decoded = Vec::with_capacity(expected_frames);
                let mut small = Vec::with_capacity(expected_frames);
                let mut live_detector = ClassicShotDetector::new(detector.clone());
                let mut last_partial_emit = std::time::Instant::now();
                let mut cached_shots_count = 0usize;

                while let Ok(f) = rx.recv() {
                    if !cache_hit {
                        let frame = downsample_64x36_fast(&f);
                        live_detector.push(&frame);
                        small.push(frame);
                    }
                    decoded.push(f);

                    if decoded.len().is_multiple_of(16) || decoded.len() == expected_frames {
                        let ratio = (decoded.len() as f32 / expected_frames as f32).clamp(0.0, 1.0);
                        let pct = 8 + (ratio * 66.0) as u32;
                        set_gui_progress(
                            true,
                            pct,
                            format!(
                                "Streaming video frames ({}/{} frames)...",
                                decoded.len(),
                                expected_frames
                            ),
                        );

                        // Every 450ms, emit newly discovered scenes incrementally
                        if !cache_hit
                            && !provisional
                            && last_partial_emit.elapsed().as_millis() >= 450
                            && small.len() >= 12
                        {
                            let cur_dur = decoded.last().map(|fr| fr.pts).unwrap_or(probe.duration);
                            if let Ok(partial_shots) =
                                live_detector.detect(cur_dur, ingest_fps, false)
                            {
                                // Only populate thumbnails for newly discovered shots!
                                let completed = partial_shots.len().saturating_sub(1);
                                if completed > cached_shots_count {
                                    self.shot_pool.shots = partial_shots;
                                    self.populate_thumbnails_range_from_ram(
                                        &decoded,
                                        cached_shots_count,
                                        completed,
                                    );
                                    cached_shots_count = completed;
                                    self.status_message = format!(
                                        "Splitting clip... {} scenes found so far ({:.0}%)",
                                        self.shot_pool.shots.len(),
                                        ratio * 100.0
                                    );
                                    on_update(self.snapshot(), false);
                                }
                            }
                            last_partial_emit = std::time::Instant::now();
                        }
                    }
                }
                handle.join().map_err(|_| ae_core::AeError::Decode {
                    path: path.to_path_buf(),
                    message: "single-pass ingest decode thread panicked".into(),
                })??;
                Ok((decoded, small))
            })?;

        if let Some(cached) = cached {
            self.shot_pool.shots = cached.shots;
            self.timeline_model.bank = cached.bank;
        } else {
            set_gui_progress(true, 76, "Finalizing shot boundaries...");
            self.shot_pool.shots =
                detect_shots_classic(&small_64, probe.duration, ingest_fps, &detector)?;
            let mut progress = |fraction: f32| {
                set_gui_progress(
                    true,
                    76 + (fraction.clamp(0.0, 1.0) * 6.0) as u32,
                    format!("Finalizing shot boundaries ({:.0}%)...", fraction * 100.0),
                );
            };
            if let Some(frames) = &refinement_frames {
                refine_shot_boundaries_from_frame_file(
                    path,
                    &frames.0,
                    &mut self.shot_pool.shots,
                    ingest_fps,
                    probe.fps,
                    &mut progress,
                )?;
            } else {
                refine_shot_boundaries_with_progress(
                    &self.backend,
                    path,
                    &mut self.shot_pool.shots,
                    ingest_fps,
                    probe.fps,
                    &mut progress,
                )?;
            }
        }
        drop(refinement_frames);

        // Slice start/mid/end thumbnails for every shot directly from `decoded_frames` in RAM
        self.status_message = format!(
            "Verified {} scenes — indexing thumbnails...",
            self.shot_pool.shots.len()
        );
        set_gui_progress(true, 83, "Indexing scene thumbnails in RAM...");
        self.populate_thumbnails_from_ram(&decoded_frames);
        on_update(self.snapshot(), true);

        // Sample 3 frames per shot (start, mid, end) for `build_descriptor_bank_from_frames`
        if !cache_hit {
            set_gui_progress(true, 89, "Computing visual similarity descriptors...");
            let sampled_for_sim = self.sample_three_frames_per_shot(&small_64);
            self.timeline_model.bank = build_descriptor_bank_from_frames(
                &sampled_for_sim,
                &self.shot_pool.shots,
                &similarity,
            )?;
            if let Some(key) = cache_key.filter(|key| {
                analysis_key(
                    &cache,
                    path,
                    &detector,
                    &similarity,
                    ingest_fps,
                    probe.fps,
                    probe.duration,
                    (ing_w, ing_h),
                )
                .as_ref()
                    == Some(key)
            }) {
                let _ = cache.put(
                    &key,
                    &CachedAnalysis {
                        shots: self.shot_pool.shots.clone(),
                        bank: self.timeline_model.bank.clone(),
                    },
                );
            }
        }

        // Store proxy reel in GuiMediaCache
        set_gui_progress(true, 95, "Populating instant RAM preview reel...");
        if let Ok(mut cache) = global_media_cache().write() {
            cache.set_proxy_reel(decoded_frames);
        }

        self.detect_characters()?;
        self.status_message = format!(
            "Kept {} of {} scenes with detected characters in '{}' ({:.1}s).",
            self.character_filtered_shots()
                .iter()
                .filter(|shot| shot.enabled)
                .count(),
            self.shot_pool.shots.len(),
            path.file_name().and_then(|s| s.to_str()).unwrap_or("clip"),
            probe.duration.as_secs_f64()
        );

        if self.audio_path.is_some() {
            set_gui_progress(true, 98, "Sequencing match cuts...");
            let _ = self.resequence();
        }
        set_gui_progress(false, 100, self.status_message.clone());
        on_update(self.snapshot(), true);
        Ok(())
    }

    /// Extract 3 key frames per shot (usable_start, mid, usable_end) from `frames` in RAM so
    /// visual similarity extraction scales with the number of scenes ($3N$) rather than video length.
    fn sample_three_frames_per_shot(&self, frames: &[RgbaFrame]) -> Vec<RgbaFrame> {
        if frames.is_empty() {
            return Vec::new();
        }
        let find_idx = |target_pts: Rational| -> usize {
            match frames.binary_search_by(|f| f.pts.cmp(&target_pts)) {
                Ok(i) => i,
                Err(i) => i.min(frames.len() - 1),
            }
        };
        let mut out = Vec::with_capacity(self.shot_pool.shots.len() * 3);
        for shot in &self.shot_pool.shots {
            let mid = Rational::from_secs_f64(
                0.5 * (shot.usable_start.as_secs_f64() + shot.usable_end.as_secs_f64()),
            );
            let end_pts = (shot.usable_end - Rational::from_millis(40)).max(shot.usable_start);
            for pts in [shot.usable_start, mid, end_pts] {
                let mut f = frames[find_idx(pts)].clone();
                f.pts = pts;
                out.push(f);
            }
        }
        out
    }

    fn populate_thumbnails_from_ram(&self, frames: &[RgbaFrame]) {
        self.populate_thumbnails_range_from_ram(frames, 0, self.shot_pool.shots.len());
    }

    fn populate_thumbnails_range_from_ram(
        &self,
        frames: &[RgbaFrame],
        start_shot_idx: usize,
        end_shot_idx: usize,
    ) {
        if frames.is_empty() || start_shot_idx >= end_shot_idx {
            return;
        }
        let find_nearest = |target_pts: Rational| -> RgbaFrame {
            match frames.binary_search_by(|f| f.pts.cmp(&target_pts)) {
                Ok(i) => frames[i].clone(),
                Err(i) => {
                    if i == 0 {
                        frames[0].clone()
                    } else if i >= frames.len() {
                        frames[frames.len() - 1].clone()
                    } else {
                        let d_prev = (target_pts - frames[i - 1].pts).abs();
                        let d_curr = (frames[i].pts - target_pts).abs();
                        if d_prev <= d_curr {
                            frames[i - 1].clone()
                        } else {
                            frames[i].clone()
                        }
                    }
                }
            }
        };

        let frame_duration = frames
            .get(1)
            .map_or(Rational::ZERO, |f| f.pts - frames[0].pts);
        let slice_end = end_shot_idx.min(self.shot_pool.shots.len());
        let mut prepared = Vec::with_capacity(slice_end - start_shot_idx);
        for shot in &self.shot_pool.shots[start_shot_idx..slice_end] {
            let mid = Rational::from_secs_f64(
                0.5 * (shot.usable_start.as_secs_f64() + shot.usable_end.as_secs_f64()),
            );
            let end_pts = (shot.usable_end - frame_duration).max(shot.usable_start);
            let t0 = find_nearest(shot.usable_start);
            let t1 = find_nearest(mid);
            let t2 = find_nearest(end_pts);
            prepared.push((shot.id, [t0, t1, t2]));
        }

        if let Ok(mut cache) = global_media_cache().write() {
            for (shot_id, thumbs) in prepared {
                cache.store_shot_thumbnails(shot_id, thumbs);
            }
        }
    }

    /// Import an audio track, isolate the bass instrument (`ae-beats`), populate draggable beat pins
    /// and the spectrogram texture (`image://spectrogram/<rev>`), and auto-sequence if video is loaded.
    pub fn import_audio(&mut self, path: &Path) -> Result<()> {
        set_gui_progress(true, 20, "Decoding audio & isolating bass frequencies...");
        self.audio_path = Some(path.to_path_buf());
        self.timeline_model.config.source_audio = path.to_string_lossy().to_string();
        self.beat_editor.load_audio(path)?;
        self.refresh_click_track_wav();

        let active_count = self.beat_editor.beats.iter().filter(|b| b.enabled).count();
        self.status_message = format!(
            "Detected {} bass beats in '{}' ({:.1}s).",
            active_count,
            path.file_name().and_then(|s| s.to_str()).unwrap_or("audio"),
            self.beat_editor.duration_secs()
        );

        if !self.shot_pool.shots.is_empty() {
            set_gui_progress(true, 85, "Sequencing match cuts to bass beats...");
            let _ = self.resequence();
        }
        set_gui_progress(false, 100, self.status_message.clone());
        Ok(())
    }

    pub fn import_transition_style(&mut self, path: &Path) -> Result<()> {
        let policy = ae_transitions::StyleTomlConfig::load(path)?;
        if !policy.match_cut_threshold.is_finite()
            || !policy.strong_beat_threshold.is_finite()
            || policy.default_duration_ms < 0
        {
            return Err(ae_core::AeError::Config(
                "Style thresholds must be finite and duration must be nonnegative".into(),
            ));
        }
        let params = toml::Value::Table(toml::map::Map::new());
        ae_transitions::build_transition(&policy.strong_beat_transition, &params)?;
        ae_transitions::build_transition(&policy.low_similarity_transition, &params)?;
        let previous = std::mem::replace(&mut self.timeline_model.policy, policy);
        if !self.shot_pool.shots.is_empty() && self.audio_path.is_some() {
            if let Err(error) = self.resequence() {
                self.timeline_model.policy = previous;
                return Err(error);
            }
        }
        self.status_message = format!("Imported transition style from {}", path.display());
        Ok(())
    }

    pub fn set_resolution_str(&mut self, res_str: &str) -> Result<Resolution> {
        let res: Resolution = res_str.parse()?;
        self.resolution = res;
        self.timeline_model.set_resolution(res);
        self.status_message = format!("Output resolution set to {} (stretch to fill).", res);
        Ok(res)
    }

    pub fn set_allow_reuse(&mut self, allow: bool) -> Result<()> {
        self.timeline_model.config.allow_reuse = allow;
        if self.shot_pool.shots.is_empty() {
            self.status_message = if allow {
                "Round-robin clip reuse enabled — clips may repeat once the pool is exhausted."
                    .to_string()
            } else {
                "Clip reuse disabled — every scene is used at most once.".to_string()
            };
            return Ok(());
        }
        self.resequence()
    }

    fn detect_characters(&mut self) -> Result<()> {
        if self.character_analysis.is_some() {
            return Ok(());
        }
        let pending: Vec<_> = self
            .shot_pool
            .shots
            .iter()
            .filter(|shot| {
                !self
                    .character_detections
                    .contains_key(&(shot.usable_start, shot.usable_end))
            })
            .cloned()
            .collect();
        if pending.is_empty() {
            return Ok(());
        }
        let (was_busy, percent, _) = crate::image_provider::get_gui_progress();
        let result = (|| -> Result<()> {
            self.prepare_character_model(&character_models_dir())?;
            let video = self.video_path.as_deref().ok_or_else(|| {
                ae_core::AeError::Config("Import a video before detecting characters.".into())
            })?;
            let info = self.backend.probe_video(video)?;
            let (width, height) = info.oriented_dimensions();
            let longest = width.max(height).max(1) as f64;
            let resolution = Resolution::new(
                (width as f64 * 640.0 / longest).round().max(1.0) as u32,
                (height as f64 * 640.0 / longest).round().max(1.0) as u32,
            );
            let mut detector = ae_render::open_detector(&ae_render::ReframeConfig {
                enabled: true,
                model_path: Some(character_models_dir().join("anime_character.onnx")),
                ..Default::default()
            });
            for (index, shot) in pending.iter().enumerate() {
                set_gui_progress(
                    true,
                    95 + (index * 3 / pending.len()) as u32,
                    format!(
                        "Finding character scenes ({}/{})…",
                        index + 1,
                        pending.len()
                    ),
                );
                let samples = {
                    let cache = global_media_cache().read().map_err(|_| {
                        ae_core::AeError::Config(
                            "Could not read source frames for character detection.".into(),
                        )
                    })?;
                    let start = cache
                        .proxy_reel
                        .partition_point(|frame| frame.pts < shot.usable_start);
                    let half_sample = cache
                        .proxy_reel
                        .windows(2)
                        .next()
                        .map(|pair| (pair[1].pts - pair[0].pts) / 2)
                        .unwrap_or(Rational::ZERO);
                    let end = cache
                        .proxy_reel
                        .partition_point(|frame| frame.pts < shot.usable_end - half_sample);
                    let frames = cache.proxy_reel.get(start..end).unwrap_or(&[]);
                    if frames.is_empty() {
                        Vec::new()
                    } else {
                        [0, frames.len() / 2, frames.len() - 1]
                            .map(|index| frames[index].clone())
                            .to_vec()
                    }
                };
                let detected = shot_has_character(shot, &samples, resolution, detector.as_mut())?;
                self.character_detections
                    .insert((shot.usable_start, shot.usable_end), detected);
            }
            Ok(())
        })();
        if let Err(error) = &result {
            self.status_message = error.to_string();
        }
        set_gui_progress(was_busy, percent, self.status_message.clone());
        result
    }

    pub fn scene_character_tags(&self, shot: &Shot) -> SceneCharacterTags {
        if let Some(analysis) = &self.character_analysis {
            return self
                .character_edits
                .tags_for(analysis, shot.usable_start, shot.usable_end);
        }
        let detected = self
            .character_detections
            .get(&(shot.usable_start, shot.usable_end));
        SceneCharacterTags {
            unresolved: detected == Some(&true),
            complete: detected.is_some(),
            ..Default::default()
        }
    }

    pub fn cycle_character_filter(&mut self, character_id: CharacterId) -> Result<()> {
        if !self.character_analysis.as_ref().is_some_and(|analysis| {
            analysis
                .records
                .iter()
                .any(|record| record.id == character_id)
        }) {
            return Err(ae_core::AeError::Config("Unknown character.".into()));
        }
        if let Some(filter) = self
            .character_filters
            .iter_mut()
            .find(|f| f.character_id == character_id)
        {
            filter.mode = filter.mode.next();
            self.character_filters
                .retain(|f| f.mode != CharacterFilterMode::Neutral);
        } else {
            self.character_filters.push(CharacterFilter {
                character_id,
                mode: CharacterFilterMode::With,
            });
        }
        self.draft_video_path = None;
        self.resequence()
    }

    pub fn clear_character_filters(&mut self) -> Result<()> {
        self.character_filters.clear();
        self.draft_video_path = None;
        self.resequence()
    }

    pub fn set_character_analysis(&mut self, analysis: CharacterAnalysis) -> Result<()> {
        analysis.validate()?;
        self.character_base_analysis = Some(analysis.clone());
        self.character_analysis = Some(analysis);
        self.character_analysis_error.clear();
        self.character_revision = self.character_revision.wrapping_add(1);
        self.draft_video_path = None;
        self.resequence()
    }

    fn revised_character_portraits(
        &self,
        before: &CharacterAnalysis,
        after: &CharacterAnalysis,
        mut portraits: HashMap<u64, RgbaFrame>,
    ) -> Result<HashMap<u64, RgbaFrame>> {
        portraits.retain(|id, _| after.records.iter().any(|record| record.id == *id));
        let changed = after
            .records
            .iter()
            .filter(|record| {
                !portraits.contains_key(&record.id)
                    || before
                        .records
                        .iter()
                        .find(|old| old.id == record.id)
                        .and_then(|old| old.exemplars.first())
                        != record.exemplars.first()
            })
            .map(|record| record.id)
            .collect::<Vec<_>>();
        if !changed.is_empty() {
            let video = self
                .video_path
                .as_deref()
                .ok_or_else(|| ae_core::AeError::Config("Import a video first.".into()))?;
            portraits.extend(build_character_portraits(
                &self.backend,
                video,
                after,
                &changed,
                &std::sync::atomic::AtomicBool::new(false),
            )?);
        }
        Ok(portraits)
    }

    pub fn install_character_analysis(&mut self, output: CharacterAnalysisOutput) -> Result<()> {
        output.analysis.validate()?;
        let mut corrected = output.analysis.clone();
        let saved = load_edits(&self.character_edits_root, &output.source_key).and_then(|edits| {
            let edits = edits.unwrap_or_else(|| CharacterEdits { analysis_key: output.identity_key.clone(), ..Default::default() });
            if edits.analysis_key != output.identity_key {
                return Err(ae_core::AeError::Config("Saved character corrections need review after analysis changed; the saved file was preserved.".into()));
            }
            edits.apply(&mut corrected)?;
            Ok(edits)
        });
        let (edits, edit_error) = match saved {
            Ok(edits) => (edits, String::new()),
            Err(error) => (
                CharacterEdits {
                    analysis_key: output.identity_key.clone(),
                    ..Default::default()
                },
                error.to_string(),
            ),
        };
        let portraits =
            self.revised_character_portraits(&output.analysis, &corrected, output.portraits)?;
        self.character_edits = edits;
        self.character_edits_error = edit_error.clone();
        self.character_base_analysis = Some(output.analysis);
        self.character_analysis = Some(corrected);
        self.character_source_key = output.source_key;
        self.character_analysis_key = output.identity_key;
        self.character_filters.retain(|filter| {
            self.character_analysis
                .as_ref()
                .unwrap()
                .records
                .iter()
                .any(|record| record.id == filter.character_id)
        });
        self.character_revision = self.character_revision.wrapping_add(1);
        self.character_analysis_error = edit_error;
        if let Ok(mut cache) = global_media_cache().write() {
            cache.character_portraits = portraits;
            cache.character_revision = self.character_revision;
        }
        self.draft_video_path = None;
        self.resequence()?;
        if !output.warning.is_empty() {
            self.status_message
                .push_str(&format!(" {}", output.warning));
        }
        Ok(())
    }

    pub fn apply_character_edit(&mut self, edit: CharacterEdit) -> Result<()> {
        if !self.character_edits_error.is_empty() {
            return Err(ae_core::AeError::Config(self.character_edits_error.clone()));
        }
        let mut edits = self.character_edits.clone();
        if edits.analysis_key != self.character_analysis_key {
            return Err(ae_core::AeError::Config(
                "Character correction version mismatch.".into(),
            ));
        }
        edits.operations.push(edit.clone());
        let mut analysis = self
            .character_base_analysis
            .clone()
            .ok_or_else(|| ae_core::AeError::Config("Analyze characters first.".into()))?;
        edits.apply(&mut analysis)?;
        let existing = global_media_cache()
            .read()
            .map_err(|_| ae_core::AeError::Config("Could not read character portraits.".into()))?
            .character_portraits
            .clone();
        let portraits = self.revised_character_portraits(
            self.character_analysis.as_ref().unwrap(),
            &analysis,
            existing,
        )?;
        save_edits(
            &self.character_edits_root,
            &self.character_source_key,
            &edits,
        )?;
        if let CharacterEdit::Merge { from, into } = edit {
            if self
                .character_filters
                .iter()
                .any(|filter| filter.character_id == into)
            {
                self.character_filters
                    .retain(|filter| filter.character_id != from);
            } else {
                for filter in &mut self.character_filters {
                    if filter.character_id == from {
                        filter.character_id = into;
                    }
                }
            }
        }
        self.character_filters.retain(|filter| {
            analysis
                .records
                .iter()
                .any(|record| record.id == filter.character_id)
        });
        self.character_edits = edits;
        self.character_analysis = Some(analysis);
        self.character_revision = self.character_revision.wrapping_add(1);
        if let Ok(mut cache) = global_media_cache().write() {
            cache.character_portraits = portraits;
            cache.character_revision = self.character_revision;
        }
        self.draft_video_path = None;
        self.resequence()
    }

    pub fn rename_character(&mut self, id: CharacterId, label: String) -> Result<()> {
        self.apply_character_edit(CharacterEdit::Rename { id, label })
    }

    pub fn merge_characters(&mut self, from: CharacterId, into: CharacterId) -> Result<()> {
        self.apply_character_edit(CharacterEdit::Merge { from, into })
    }

    pub fn review_scene_characters(
        &mut self,
        shot_id: usize,
        present: Vec<CharacterId>,
    ) -> Result<()> {
        let shot = self
            .shot_pool
            .shots
            .iter()
            .find(|shot| shot.id == shot_id)
            .ok_or_else(|| ae_core::AeError::Config("Scene not found.".into()))?;
        self.apply_character_edit(CharacterEdit::Interval {
            start: shot.usable_start,
            end: shot.usable_end,
            present,
            reviewed: true,
        })
    }

    fn character_filtered_shots(&self) -> Vec<Shot> {
        self.shot_pool
            .shots
            .iter()
            .cloned()
            .map(|mut shot| {
                shot.enabled &=
                    character_matches(&self.scene_character_tags(&shot), &self.character_filters);
                shot
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn seed_character_detections_for_test(&mut self) {
        for shot in &self.shot_pool.shots {
            self.character_detections
                .insert((shot.usable_start, shot.usable_end), true);
        }
        self.character_analysis = Some(CharacterAnalysis {
            observations: self
                .shot_pool
                .shots
                .iter()
                .map(|shot| crate::character_model::CharacterObservation {
                    id: shot.id as u64 + 1,
                    pts: shot.usable_start + (shot.usable_end - shot.usable_start) / 2,
                    bounds: [0.0, 0.0, 1.0, 1.0],
                    score: 1.0,
                    feature: vec![0.0],
                    character_id: None,
                    possible: Vec::new(),
                })
                .collect(),
            scanned_ranges: self
                .shot_pool
                .shots
                .iter()
                .map(|shot| (shot.usable_start, shot.usable_end))
                .collect(),
            ..Default::default()
        });
    }

    pub fn resequence(&mut self) -> Result<()> {
        if self.shot_pool.shots.is_empty() {
            return Ok(());
        }
        self.detect_characters()?;
        let shots = self.character_filtered_shots();
        if !shots.iter().any(|shot| shot.enabled) {
            self.timeline_model.timeline = None;
            self.draft_video_path = None;
            self.sync_timeline_to_preview_cache();
            self.status_message = if self.character_filters.is_empty() {
                "No enabled scenes with detected characters. No edit generated."
            } else {
                "No enabled scenes match these character filters. Clear filters or review unresolved characters."
            }.into();
            return Ok(());
        }
        if self.timeline_model.bank.by_shot.iter().all(Vec::is_empty)
            && !self.beat_editor.beats.iter().any(|beat| beat.enabled)
        {
            self.timeline_model.timeline = None;
            self.draft_video_path = None;
            self.sync_timeline_to_preview_cache();
            self.status_message =
                "Import audio to generate an edit from the matching scenes.".into();
            return Ok(());
        }
        let excluded_locks = self
            .timeline_model
            .timeline
            .as_ref()
            .map(|timeline| {
                timeline
                    .slots
                    .iter()
                    .filter(|slot| {
                        slot.locked && !shots.get(slot.shot_id).is_some_and(|shot| shot.enabled)
                    })
                    .count()
            })
            .unwrap_or(0);
        let tl = self
            .timeline_model
            .resequence(&shots, &self.beat_editor.beats);
        let tl = match tl {
            Ok(timeline) => timeline.clone(),
            Err(error) => {
                self.timeline_model.timeline = None;
                self.draft_video_path = None;
                self.sync_timeline_to_preview_cache();
                return Err(error);
            }
        };
        self.sync_timeline_to_preview_cache();
        let reuse_note = if self.timeline_model.config.allow_reuse {
            "round-robin reuse"
        } else {
            "no reuse"
        };
        self.status_message = format!(
            "Sequenced {} slots ({}, hard cut default, mean match similarity {:.1}%).",
            tl.slots.len(),
            reuse_note,
            tl.mean_boundary_similarity * 100.0
        );
        if excluded_locks > 0 {
            self.status_message.push_str(&format!(
                " Replaced {excluded_locks} locked slots excluded by scene selection."
            ));
        }
        Ok(())
    }

    fn sync_timeline_to_preview_cache(&mut self) {
        if let Ok(mut cache) = global_media_cache().write() {
            self.preview_revision = cache.set_active_timeline(self.timeline_model.timeline.clone());
        }
    }

    fn refresh_click_track_wav(&mut self) {
        if let Some(ref session) = self.beat_editor.session {
            let preview_dir = PathBuf::from(".ae-cache/preview_audio");
            let active_beats: Vec<_> = self
                .beat_editor
                .beats
                .iter()
                .filter(|b| b.enabled)
                .cloned()
                .collect();
            if write_debug_artifacts(
                &preview_dir,
                &session.mono_22k,
                &session.last_bass_spec,
                &session.last_odf,
                &active_beats,
            )
            .is_ok()
            {
                self.click_track_path = Some(preview_dir.join("click_track.wav"));
            }
        }
    }

    pub fn toggle_shot(&mut self, shot_id: usize) -> Result<()> {
        if let Some(enabled) = self.shot_pool.toggle_shot(shot_id) {
            self.status_message = format!(
                "Scene #{shot_id:02} {}.",
                if enabled { "included" } else { "excluded" }
            );
            let _ = self.resequence();
        }
        Ok(())
    }

    pub fn set_all_shots_enabled(&mut self, enabled: bool) -> Result<()> {
        self.shot_pool.set_all_enabled(enabled);
        let _ = self.resequence();
        Ok(())
    }

    pub fn invert_shots(&mut self) -> Result<()> {
        self.shot_pool.invert_selection();
        let _ = self.resequence();
        Ok(())
    }

    pub fn split_shot(&mut self, shot_id: usize, split_sec: f64) -> Result<()> {
        self.shot_pool.split_shot_at(shot_id, split_sec)?;
        let proxy_frames = global_media_cache()
            .read()
            .map(|c| c.proxy_reel.clone())
            .unwrap_or_default();
        if !proxy_frames.is_empty() {
            self.populate_thumbnails_from_ram(&proxy_frames);
            let sampled = self.sample_three_frames_per_shot(&proxy_frames);
            self.timeline_model.bank = build_descriptor_bank_from_frames(
                &sampled,
                &self.shot_pool.shots,
                &SimilarityConfig::default(),
            )?;
        }
        let _ = self.resequence();
        Ok(())
    }

    pub fn move_beat(&mut self, beat_index: usize, new_time_sec: f64) -> Result<()> {
        let clamped = self.beat_editor.move_beat(beat_index, new_time_sec)?;
        self.status_message = format!("Moved beat #{beat_index} to {clamped:.3}s (manual lock).");
        let _ = self.resequence();
        Ok(())
    }

    pub fn toggle_beat(&mut self, beat_index: usize) -> Result<()> {
        if beat_index < self.beat_editor.beats.len() {
            let enabled = self.beat_editor.toggle_beat(beat_index)?;
            self.status_message = format!(
                "Beat #{beat_index} {}.",
                if enabled { "enabled" } else { "disabled" }
            );
        } else if let Some(ref session) = self.beat_editor.session {
            // Clicked a ghost peak -> promote to active manual beat
            let ghost_idx = beat_index - self.beat_editor.beats.len();
            if let Some(ghost) = session.ghost_peaks.get(ghost_idx) {
                let t = ghost.time.as_secs_f64();
                self.beat_editor.add_beat_at(t)?;
                self.status_message = format!("Promoted ghost peak at {t:.3}s to active beat.");
            }
        }
        let _ = self.resequence();
        Ok(())
    }

    pub fn add_beat_at(&mut self, time_sec: f64) -> Result<()> {
        let idx = self.beat_editor.add_beat_at(time_sec)?;
        self.status_message = format!("Added manual beat #{idx} at {time_sec:.3}s.");
        let _ = self.resequence();
        Ok(())
    }

    pub fn remove_beat(&mut self, beat_index: usize) -> Result<()> {
        self.beat_editor.remove_beat(beat_index)?;
        self.status_message = format!("Removed beat #{beat_index}.");
        let _ = self.resequence();
        Ok(())
    }

    pub fn set_bass_band(&mut self, low_hz: f32, high_hz: f32) -> Result<()> {
        self.beat_editor.set_bass_band(low_hz, high_hz)?;
        self.status_message = format!(
            "Updated bass band to {:.0}-{:.0} Hz ({} active beats).",
            self.beat_editor.config.bass_band.low_hz,
            self.beat_editor.config.bass_band.high_hz,
            self.beat_editor.beats.iter().filter(|b| b.enabled).count()
        );
        let _ = self.resequence();
        Ok(())
    }

    pub fn set_sensitivity(&mut self, peak_delta: f32) -> Result<()> {
        self.beat_editor.set_sensitivity(peak_delta)?;
        let _ = self.resequence();
        Ok(())
    }

    pub fn swap_slot_shot(&mut self, slot_index: usize, new_shot_id: usize) -> Result<()> {
        self.detect_characters()?;
        let shot =
            self.shot_pool.shots.get(new_shot_id).ok_or_else(|| {
                ae_core::AeError::Config(format!("Scene #{new_shot_id} not found."))
            })?;
        if !character_matches(&self.scene_character_tags(shot), &self.character_filters) {
            return Err(ae_core::AeError::Config(
                if self.character_filters.is_empty() {
                    "This scene has no detected character."
                } else {
                    "This scene does not match the character filters."
                }
                .into(),
            ));
        }
        // Explicit assignment may re-enable a scene, but cannot bypass the character filter.
        self.shot_pool.set_shot_enabled(new_shot_id, true);
        let shots = self.character_filtered_shots();
        self.timeline_model
            .swap_slot_shot(&shots, slot_index, new_shot_id)?;
        self.sync_timeline_to_preview_cache();
        self.status_message = format!("Assigned Scene #{new_shot_id:02} to Slot #{slot_index}.");
        Ok(())
    }

    pub fn slip_slot_in_point(&mut self, slot_index: usize, delta_sec: f64) -> Result<()> {
        let new_in =
            self.timeline_model
                .slip_slot_in_point(&self.shot_pool.shots, slot_index, delta_sec)?;
        self.sync_timeline_to_preview_cache();
        self.status_message = format!("Slipped Slot #{slot_index} in-point to {new_in:.2}s.");
        Ok(())
    }

    pub fn set_slot_transition(
        &mut self,
        slot_index: usize,
        transition_id: &str,
        duration_ms: i64,
    ) -> Result<()> {
        self.timeline_model
            .set_slot_transition(slot_index, transition_id, duration_ms)?;
        self.sync_timeline_to_preview_cache();
        self.status_message =
            format!("Slot #{slot_index} transition set to '{transition_id}' ({duration_ms}ms).");
        Ok(())
    }

    /// Set the same transition on every slot boundary (slot 0 always stays a hard cut).
    pub fn set_all_slot_transitions(
        &mut self,
        transition_id: &str,
        duration_ms: i64,
    ) -> Result<()> {
        let n = self
            .timeline_model
            .timeline
            .as_ref()
            .map(|tl| tl.slots.len())
            .unwrap_or(0);
        for i in 0..n {
            self.timeline_model
                .set_slot_transition(i, transition_id, duration_ms)?;
        }
        self.sync_timeline_to_preview_cache();
        self.status_message =
            format!("Applied '{transition_id}' between all clips ({duration_ms}ms).");
        Ok(())
    }
    /// Apply fast cleanup and color polish to the completed preview.
    pub fn enhance_draft(&mut self) -> Result<PathBuf> {
        let input = self.draft_video_path.as_deref().ok_or_else(|| {
            ae_core::AeError::Config("render the original preview before enhancing it".into())
        })?;
        set_gui_progress(true, 2, "Filtering the preview video…");
        let progress: RenderProgressCallback = std::sync::Arc::new(|pct, message| {
            set_gui_progress(true, pct, message.to_string());
        });
        let output = enhance_video(
            &self.backend,
            input,
            Path::new(".ae-cache"),
            Some(&progress),
        )?;
        self.status_message = "Filtered preview ready for playback and comparison.".into();
        set_gui_progress(false, 100, self.status_message.clone());
        Ok(output)
    }

    /// Incremental slot-cached render into `.ae-cache/quick_preview.mp4`.
    pub fn quick_render_draft(&mut self, scale_divisor: u32) -> Result<IncrementalDraftReport> {
        let Some(ref tl) = self.timeline_model.timeline else {
            return Err(ae_core::AeError::Config(
                "no timeline to quick-render — import video and audio first".into(),
            ));
        };
        let cache_dir = PathBuf::from(".ae-cache");
        let out_path = cache_dir.join("quick_preview.mp4");

        set_gui_progress(true, 8, "Starting incremental Quick Render...");
        if self.reframe_faces {
            self.prepare_character_model(&character_models_dir())?;
        }
        let cfg = self.build_quick_render_config(tl, &out_path);

        let report =
            render_timeline_incremental_draft(&self.backend, tl, &cfg, &cache_dir, scale_divisor)?;
        self.draft_video_path = Some(out_path);
        self.status_message = format!(
            "Quick Render ready ({}) — {}/{} slots reused from cache, {} encoded.",
            report.resolution, report.cached_slots, report.total_slots, report.encoded_slots
        );
        set_gui_progress(false, 100, self.status_message.clone());
        Ok(report)
    }

    pub fn rank_slot_alternatives(
        &self,
        slot_index: usize,
        top_k: usize,
    ) -> Vec<AlternativeShotItem> {
        self.timeline_model
            .rank_alternatives(&self.character_filtered_shots(), slot_index, top_k)
    }

    /// Export the current timeline to `output_path` stretched to `self.resolution`.
    pub fn export_video(&mut self, output_path: &Path, use_gpu: bool) -> Result<()> {
        self.export_video_with_quality(output_path, use_gpu, 18, "fast")
    }

    fn prepare_character_model(&self, models_dir: &Path) -> Result<()> {
        let manifest = ae_core::ModelManifest::load(&models_dir.join("manifest.toml"))?;
        let entry = manifest.models.get("anime_character").ok_or_else(|| {
            ae_core::AeError::Config(
                "Animated character detector is missing from models/manifest.toml".into(),
            )
        })?;
        let path = models_dir.join(&entry.filename);
        if path.exists() {
            manifest.verify_file("anime_character", models_dir)?;
            return Ok(());
        }
        set_gui_progress(true, 8, "Downloading animated character detector (36 MB)…");
        static TEMP_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let temp = path.with_extension(format!(
            "{}.{}.download",
            std::process::id(),
            TEMP_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let result = (|| -> Result<()> {
            let output = std::process::Command::new("curl")
                .args([
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--max-time",
                    "180",
                ])
                .arg(&entry.url)
                .arg("--output")
                .arg(&temp)
                .output()?;
            if !output.status.success() {
                return Err(ae_core::AeError::Config(format!(
                    "Could not download animated character detector: {}",
                    String::from_utf8_lossy(&output.stderr).trim(),
                )));
            }
            let mut downloaded = manifest.clone();
            downloaded
                .models
                .get_mut("anime_character")
                .unwrap()
                .filename = temp.file_name().unwrap().to_string_lossy().into_owned();
            downloaded.verify_file("anime_character", models_dir)?;
            std::fs::rename(&temp, &path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        result
    }

    /// Full-character model for animated subject locking in previews and exports.
    /// Downloaded and verified before rendering so missing files cannot disable tracking.
    fn reframe_model_path(&self) -> Option<PathBuf> {
        self.reframe_faces
            .then(|| character_models_dir().join("anime_character.onnx"))
    }

    /// Final-export render config. Face-following crop is opt-in (default off).
    pub fn build_export_config(
        &self,
        tl: &Timeline,
        output_path: &Path,
        use_gpu: bool,
        crf: u32,
        preset: &str,
    ) -> RenderConfig {
        let video_path = self
            .video_path
            .clone()
            .unwrap_or_else(|| PathBuf::from(&tl.source_video));
        RenderConfig {
            video_path,
            audio_path: self.audio_path.clone(),
            output_path: output_path.to_path_buf(),
            resolution: Some(self.resolution),
            fps: Some(self.fps),
            codec: VideoCodec::X264,
            crf,
            preset: preset.to_string(),
            use_gpu,
            quality_enhance: self.quality_enhance,
            light_cleanup: self.light_cleanup,
            reframe_faces: self.reframe_faces,
            face_model_path: self.reframe_model_path(),
            shaders_dir: Some(PathBuf::from("transitions")),
            cache_capacity: 16,
            progress_cb: Some(std::sync::Arc::new(|pct, msg| {
                set_gui_progress(true, pct, msg.to_string());
            })),
            ..Default::default()
        }
    }

    /// Quick Render preview config. Honors the same face-following toggle as
    /// final export so the preview shows the reframed crop.
    pub fn build_quick_render_config(&self, tl: &Timeline, out_path: &Path) -> RenderConfig {
        let video_path = self
            .video_path
            .clone()
            .unwrap_or_else(|| PathBuf::from(&tl.source_video));
        RenderConfig {
            video_path,
            audio_path: self.audio_path.clone(),
            output_path: out_path.to_path_buf(),
            resolution: Some(self.resolution),
            fps: Some(self.fps),
            codec: VideoCodec::X264,
            crf: 18,
            preset: "ultrafast".to_string(),
            use_gpu: false,
            quality_enhance: false,
            light_cleanup: false,
            reframe_faces: self.reframe_faces,
            face_model_path: self.reframe_model_path(),
            shaders_dir: Some(PathBuf::from("transitions")),
            cache_capacity: 16,
            progress_cb: Some(std::sync::Arc::new(|pct, msg| {
                set_gui_progress(true, pct, msg.to_string());
            })),
            ..Default::default()
        }
    }

    pub fn export_video_with_quality(
        &mut self,
        output_path: &Path,
        use_gpu: bool,
        crf: u32,
        preset: &str,
    ) -> Result<()> {
        if crf > 51
            || ![
                "ultrafast",
                "superfast",
                "veryfast",
                "faster",
                "fast",
                "medium",
                "slow",
                "slower",
                "veryslow",
                "placebo",
            ]
            .contains(&preset)
        {
            return Err(ae_core::AeError::Config(
                "invalid x264 CRF or preset".into(),
            ));
        }
        let Some(ref tl) = self.timeline_model.timeline else {
            return Err(ae_core::AeError::Config(
                "no timeline to export — import video and audio first".into(),
            ));
        };
        set_gui_progress(
            true,
            25,
            if self.reframe_faces {
                format!(
                    "Rendering {} slots at {} (character lock)...",
                    tl.slots.len(),
                    self.resolution
                )
            } else {
                format!(
                    "Rendering {} slots at {} (stretch to fill)...",
                    tl.slots.len(),
                    self.resolution
                )
            },
        );
        if self.reframe_faces {
            self.prepare_character_model(&character_models_dir())?;
        }
        let cfg = self.build_export_config(tl, output_path, use_gpu, crf, preset);

        let res = render_timeline(&self.backend, tl, &cfg);
        if let Err(e) = res {
            set_gui_progress(false, 0, format!("Export failed: {e}"));
            return Err(e);
        }
        self.status_message = format!(
            "Exported {} ({} slots) -> {}",
            self.resolution,
            tl.slots.len(),
            output_path.display()
        );
        set_gui_progress(false, 100, self.status_message.clone());
        Ok(())
    }

    pub fn snapshot(&self) -> GuiStateSnapshot {
        let assignments = self.timeline_model.slot_assignments();
        let mut shots = self.shot_pool.to_view_items(&assignments);
        let mut character_counts: HashMap<CharacterId, usize> = HashMap::new();
        shots.retain_mut(|item| {
            let tags = self.scene_character_tags(&self.shot_pool.shots[item.id]);
            let visible = character_matches(&tags, &self.character_filters)
                || (self.character_filters.is_empty() && !tags.complete && !tags.failed);
            for id in &tags.present {
                *character_counts.entry(*id).or_default() += 1;
            }
            item.character_ids = tags.present;
            item.characters_unresolved =
                tags.unresolved || !tags.possible.is_empty() || !tags.complete;
            visible
        });
        let beats = self.beat_editor.to_view_items();
        let slots = self.timeline_model.to_view_items();
        let mean_similarity = self
            .timeline_model
            .timeline
            .as_ref()
            .map(|tl| tl.mean_boundary_similarity)
            .unwrap_or(0.0);

        let timeline_duration_sec = self
            .timeline_model
            .timeline
            .as_ref()
            .map(|tl| tl.total_duration.as_secs_f64())
            .unwrap_or_else(|| self.beat_editor.duration_secs());

        GuiStateSnapshot {
            video_path: self
                .video_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            audio_path: self
                .audio_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            click_track_path: self
                .click_track_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            draft_video_path: self
                .draft_video_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            resolution: self.resolution.to_string(),
            target_width: self.resolution.width,
            target_height: self.resolution.height,
            allow_reuse: self.timeline_model.config.allow_reuse,
            bass_low_hz: self.beat_editor.config.bass_band.low_hz,
            bass_high_hz: self.beat_editor.config.bass_band.high_hz,
            separation_mode: format!("{:?}", self.beat_editor.config.separation).to_lowercase(),
            beat_mode: format!("{:?}", self.beat_editor.config.beat_mode).to_lowercase(),
            sensitivity: self.beat_editor.config.peak_delta,
            audio_duration_sec: self.beat_editor.duration_secs(),
            timeline_duration_sec,
            spectrogram_url: format!(
                "image://spectrogram/{}",
                self.beat_editor.spectrogram_revision
            ),
            preview_revision: self.preview_revision,
            mean_similarity,
            status_message: self.status_message.clone(),
            characters: self
                .character_analysis
                .as_ref()
                .map(|analysis| {
                    analysis
                        .records
                        .iter()
                        .enumerate()
                        .map(|(index, record)| CharacterViewItem {
                            id: record.id,
                            label: if record.label.is_empty() {
                                format!("Character {}", index + 1)
                            } else {
                                record.label.clone()
                            },
                            provisional: record.provisional,
                            mode: self
                                .character_filters
                                .iter()
                                .find(|f| f.character_id == record.id)
                                .map(|f| f.mode)
                                .unwrap_or_default(),
                            scene_count: character_counts.get(&record.id).copied().unwrap_or(0),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            character_filters: self.character_filters.clone(),
            character_analysis_error: self.character_analysis_error.clone(),
            character_revision: self.character_revision,
            character_analysis_pending: self.character_analysis_error.is_empty()
                && (self.character_analysis.is_none()
                    || self.shot_pool.shots.iter().any(|shot| {
                        !self
                            .character_analysis
                            .as_ref()
                            .unwrap()
                            .tags_for(shot.usable_start, shot.usable_end)
                            .complete
                    })),
            shots,
            beats,
            slots,
        }
    }

    pub fn snapshot_json(&self) -> String {
        serde_json::to_string(&self.snapshot()).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Ultra-fast stride downsample to `64x36` (~0.003 ms/frame) so `detect_shots_classic` runs at its
/// calibrated `64x36` scale without any Lanczos convolution overhead.
fn downsample_64x36_fast(frame: &RgbaFrame) -> RgbaFrame {
    if frame.width == 64 && frame.height == 36 {
        return frame.clone();
    }
    let mut out = RgbaFrame::new(64, 36, frame.pts);
    let sw = frame.width as usize;
    let sh = frame.height as usize;
    for y in 0..36usize {
        let sy = (y * sh) / 36;
        let src_row = sy * sw;
        let dst_row = y * 64;
        for x in 0..64usize {
            let sx = (x * sw) / 64;
            let s_idx = (src_row + sx) * 4;
            let d_idx = (dst_row + x) * 4;
            out.data[d_idx..d_idx + 4].copy_from_slice(&frame.data[s_idx..s_idx + 4]);
        }
    }
    out
}

#[cfg(test)]
mod character_filter_tests {
    use super::*;
    use ae_core::ShotBoundaryKind;
    use ae_render::{FaceBox, FaceDetector};

    fn shot(id: usize) -> Shot {
        Shot {
            id,
            start: Rational::from_secs(id as i64),
            end: Rational::from_secs(id as i64 + 1),
            usable_start: Rational::from_secs(id as i64),
            usable_end: Rational::from_secs(id as i64 + 1),
            cut_confidence: 1.0,
            boundary_kind: ShotBoundaryKind::HardCut,
            mean_luma: 0.5,
            enabled: true,
            locked: false,
        }
    }

    fn identified_controller() -> ProjectController {
        use crate::character_model::{CharacterObservation, CharacterRecord};
        let mut controller = ProjectController::new();
        controller.shot_pool.shots = (0..3).map(shot).collect();
        controller.character_analysis = Some(CharacterAnalysis {
            records: (1..=2)
                .map(|id| CharacterRecord {
                    id,
                    label: String::new(),
                    exemplars: vec![id],
                    provisional: false,
                })
                .collect(),
            observations: (1..=2)
                .map(|id| CharacterObservation {
                    id,
                    pts: Rational::new(2 * id as i64 - 1, 2),
                    bounds: [0.0, 0.0, 0.5, 0.5],
                    score: 1.0,
                    feature: vec![0.0],
                    character_id: Some(id),
                    possible: Vec::new(),
                })
                .collect(),
            scanned_ranges: vec![(Rational::ZERO, Rational::from_secs(3))],
            ..Default::default()
        });
        controller
    }

    #[test]
    fn character_identity_filters_share_browser_and_candidate_rules() {
        let _guard = crate::tests::GUI_TEST_LOCK.lock().unwrap();
        let mut controller = identified_controller();
        controller.shot_pool.shots[1].enabled = false;
        controller.cycle_character_filter(1).unwrap();
        assert_eq!(
            controller
                .snapshot()
                .characters
                .iter()
                .map(|character| character.scene_count)
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
        assert_eq!(
            controller
                .snapshot()
                .shots
                .iter()
                .map(|shot| shot.id)
                .collect::<Vec<_>>(),
            vec![0]
        );
        controller.cycle_character_filter(1).unwrap();
        assert_eq!(
            controller.character_filters[0].mode,
            CharacterFilterMode::Only
        );
        controller.cycle_character_filter(1).unwrap();
        assert_eq!(
            controller
                .snapshot()
                .shots
                .iter()
                .map(|shot| shot.id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        let candidates = controller.character_filtered_shots();
        assert_eq!(
            candidates.iter().map(|shot| shot.id).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(!candidates[0].enabled && !candidates[1].enabled && candidates[2].enabled);
        assert!(
            controller
                .swap_slot_shot(0, 0)
                .unwrap_err()
                .to_string()
                .contains("character filters")
        );
        controller.cycle_character_filter(1).unwrap();
        assert!(controller.character_filters.is_empty());
        assert!(!controller.shot_pool.shots[1].enabled);
        assert_eq!(
            controller
                .snapshot()
                .shots
                .iter()
                .map(|shot| shot.id)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(controller.snapshot().characters.len(), 2);
    }

    #[test]
    fn character_identity_conflicts_clear_stale_draft() {
        let _guard = crate::tests::GUI_TEST_LOCK.lock().unwrap();
        let mut controller = identified_controller();
        controller.cycle_character_filter(1).unwrap();
        controller.draft_video_path = Some(PathBuf::from("old.mp4"));
        controller.cycle_character_filter(2).unwrap();
        assert!(controller.snapshot().shots.is_empty());
        assert!(
            controller.draft_video_path.is_none() && controller.timeline_model.timeline.is_none()
        );
        assert!(
            controller
                .status_message
                .contains("No enabled scenes match")
        );
        assert!(controller.cycle_character_filter(999).is_err());
        controller.clear_character_filters().unwrap();
        assert_eq!(controller.snapshot().shots.len(), 2);
    }

    #[test]
    fn character_pending_scenes_are_visible_but_not_candidates() {
        let mut controller = ProjectController::new();
        controller.shot_pool.shots = vec![shot(0)];
        let snapshot = controller.snapshot();
        assert_eq!(snapshot.shots.len(), 1);
        assert!(snapshot.shots[0].characters_unresolved);
        assert!(!controller.character_filtered_shots()[0].enabled);
        controller.character_filters.push(CharacterFilter {
            character_id: 1,
            mode: CharacterFilterMode::Without,
        });
        assert!(controller.snapshot().shots.is_empty());
    }

    #[test]
    #[ignore = "requires pinned models, ffmpeg, and .ae-cache/character-fixtures/anime-example.jpg"]
    fn character_identity_real_media_roundtrip() {
        let _guard = crate::tests::GUI_TEST_LOCK.lock().unwrap();
        let directory = tempfile::tempdir().unwrap();
        let example = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../.ae-cache/character-fixtures/anime-example.jpg");
        assert!(
            example.is_file(),
            "Cache the upstream detector example image first"
        );
        let video = directory.path().join("characters.mp4");
        let status = std::process::Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-loop", "1", "-i"])
            .arg(&example)
            .args([
                "-vf",
                "crop=488:368:306:66,scale=640:-2",
                "-t",
                "0.5",
                "-r",
                "6",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-y",
            ])
            .arg(&video)
            .status()
            .unwrap();
        assert!(status.success());
        let mut controller = ProjectController::new();
        controller.character_edits_root = directory.path().join("corrections");
        controller.video_path = Some(video.clone());
        let info = controller.backend.probe_video(&video).unwrap();
        let mut scene = shot(0);
        scene.end = info.duration;
        scene.usable_end = info.duration;
        controller.shot_pool.shots = vec![scene];
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let output = crate::character_analysis::analyze_characters(
            &controller.backend,
            &video,
            &controller.shot_pool.shots,
            &cancel,
        )
        .unwrap();
        assert!(
            !output.analysis.observations.is_empty(),
            "real detector found no characters"
        );
        assert!(
            !output.analysis.records.is_empty(),
            "real identity grouping produced no portraits"
        );
        assert!(
            output
                .analysis
                .tags_for(Rational::ZERO, info.duration)
                .complete
        );
        assert_eq!(output.portraits.len(), output.analysis.records.len());
        let repeated = crate::character_analysis::analyze_characters(
            &controller.backend,
            &video,
            &controller.shot_pool.shots,
            &cancel,
        )
        .unwrap();
        assert_eq!(repeated.analysis, output.analysis);
        let frames = controller
            .backend
            .decode_segment(
                &video,
                Rational::ZERO,
                info.duration,
                Resolution::new(160, 120),
                info.fps,
                1.0,
                Some(info.duration),
            )
            .unwrap();
        controller.timeline_model.bank = build_descriptor_bank_from_frames(
            &frames,
            &controller.shot_pool.shots,
            &SimilarityConfig::default(),
        )
        .unwrap();
        controller.resolution = Resolution::new(160, 120);
        controller.timeline_model.config.resolution = controller.resolution;
        controller.timeline_model.config.fps = info.fps;
        controller.timeline_model.config.source_video = video.to_string_lossy().into_owned();
        controller
            .install_character_analysis(output.clone())
            .unwrap();
        let id = output.analysis.records[0].id;
        controller.cycle_character_filter(id).unwrap();
        assert_eq!(controller.snapshot().shots.len(), 1);
        controller
            .rename_character(id, "Test character".into())
            .unwrap();
        let rendered = directory.path().join("edit.mp4");
        controller.export_video(&rendered, false).unwrap();
        assert!(controller.backend.probe_video(&rendered).unwrap().duration > Rational::ZERO);
        controller.character_analysis = None;
        controller.install_character_analysis(output).unwrap();
        assert_eq!(
            controller
                .snapshot()
                .characters
                .iter()
                .find(|character| character.id == id)
                .unwrap()
                .label,
            "Test character"
        );
        controller.cycle_character_filter(id).unwrap();
        controller.cycle_character_filter(id).unwrap();
        assert!(controller.snapshot().shots.is_empty());
        assert!(controller.timeline_model.timeline.is_none());
        set_gui_progress(false, 100, "Character media test completed.");
    }

    #[test]
    fn character_sampling_excludes_adjacent_scenes_and_needs_no_face() {
        struct Detector;
        impl FaceDetector for Detector {
            fn detect(&mut self, frame: &RgbaFrame) -> Option<FaceBox> {
                (frame.data[0] == 255).then_some(FaceBox {
                    cx: 2.0,
                    cy: 2.0,
                    score: 1.0,
                })
            }
        }
        let frames: Vec<_> = [0, 500, 1000, 1500]
            .into_iter()
            .map(|ms| {
                RgbaFrame::solid(
                    4,
                    4,
                    [if ms >= 1000 { 255 } else { 0 }, 0, 0, 255],
                    Rational::from_millis(ms),
                )
            })
            .collect();
        let resolution = Resolution::new(4, 4);
        assert!(!shot_has_character(&shot(0), &frames, resolution, &mut Detector).unwrap());
        assert!(shot_has_character(&shot(1), &frames, resolution, &mut Detector).unwrap());
        assert!(!shot_has_character(&shot(2), &frames, resolution, &mut Detector).unwrap());
    }

    #[test]
    fn character_filter_is_default_and_independent_of_tracking_checkbox() {
        let mut controller = ProjectController::new();
        assert!(!controller.reframe_faces);
        controller.shot_pool.shots = vec![shot(0), shot(1)];
        controller
            .character_detections
            .insert((Rational::ZERO, Rational::from_secs(1)), false);
        controller
            .character_detections
            .insert((Rational::from_secs(1), Rational::from_secs(2)), true);
        for tracking in [false, true, false] {
            controller.reframe_faces = tracking;
            controller.detect_characters().unwrap(); // Reuses results; no model or video needed.
            let filtered = controller.character_filtered_shots();
            assert!(!filtered[0].enabled);
            assert!(filtered[1].enabled);
            assert_eq!(
                filtered[1].id, 1,
                "filtering must preserve scene/descriptor IDs"
            );
            assert_eq!(
                controller
                    .snapshot()
                    .shots
                    .iter()
                    .map(|shot| shot.id)
                    .collect::<Vec<_>>(),
                vec![1]
            );
            assert!(
                controller
                    .swap_slot_shot(0, 0)
                    .unwrap_err()
                    .to_string()
                    .contains("no detected character")
            );
        }
        assert!(
            controller.shot_pool.shots.iter().all(|shot| shot.enabled),
            "filter must preserve manual scene selections"
        );
        controller.shot_pool.shots[1].enabled = false;
        assert!(!controller.character_filtered_shots()[1].enabled);
    }

    #[test]
    fn no_character_scenes_clears_the_previous_edit_without_render_error() {
        let _guard = crate::tests::GUI_TEST_LOCK.lock().unwrap();
        let mut controller = ProjectController::new();
        controller.shot_pool.shots = vec![shot(0)];
        controller
            .character_detections
            .insert((Rational::ZERO, Rational::from_secs(1)), false);
        controller.draft_video_path = Some(PathBuf::from("stale-preview.mp4"));
        controller.timeline_model.timeline = Some(Timeline {
            version: 1,
            source_video: "unused.mp4".into(),
            source_audio: String::new(),
            resolution: Resolution::VERTICAL,
            fps: Rational::from_secs(30),
            total_duration: Rational::from_secs(1),
            mean_boundary_similarity: 1.0,
            slots: vec![ae_core::TimelineSlot {
                slot_index: 0,
                dst_start: Rational::ZERO,
                dst_end: Rational::from_secs(1),
                shot_id: 0,
                src_in: Rational::ZERO,
                src_out: Rational::from_secs(1),
                speed: 1.0,
                transition_in: ae_core::TransitionSpec::default(),
                boundary_similarity: 1.0,
                beat_strength: 1.0,
                locked: true,
            }],
        });
        controller.resequence().unwrap();
        assert!(controller.timeline_model.timeline.is_none());
        assert!(controller.draft_video_path.is_none());
        assert!(controller.snapshot().shots.is_empty());
        assert!(
            controller
                .status_message
                .contains("No enabled scenes with detected characters")
        );
    }
}

#[cfg(test)]
mod analysis_cache_tests {
    use super::*;
    use ae_media::generate_synthetic_multi_shot_video;

    #[test]
    fn keyframe_hints_stay_provisional_and_have_valid_ranges() {
        let frames: Vec<_> = [0, 40, 1000, 1000, 1200, 2400, 2990]
            .into_iter()
            .map(|ms| RgbaFrame::new(64, 36, Rational::from_millis(ms)))
            .collect();
        let shots = provisional_shots(&frames, Rational::from_secs(3));
        assert_eq!(shots.len(), 3);
        assert_eq!(shots[0].start, Rational::ZERO);
        assert_eq!(shots.last().unwrap().end, Rational::from_secs(3));
        assert!(shots.iter().enumerate().all(|(id, s)| s.id == id
            && s.cut_confidence == 0.0
            && s.start == s.usable_start
            && s.end == s.usable_end
            && s.end > s.start));
        assert!(shots.windows(2).all(|s| s[0].end == s[1].start));
    }

    #[test]
    fn long_import_replaces_keyframe_hints_with_sequentially_verified_scenes() {
        let _guard = crate::tests::GUI_TEST_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("long.mp4");
        let fps = Rational::from_secs(30);
        generate_synthetic_multi_shot_video(
            &path,
            Resolution::new(64, 36),
            fps,
            &[
                ([220, 20, 20], [20, 10, 10], 660),
                ([20, 220, 20], [10, 20, 10], 660),
                ([20, 20, 220], [10, 10, 20], 660),
            ],
        )
        .unwrap();
        let mut controller = ProjectController::new();
        let native = dir.path().join("native.rgba");
        let (tx, rx) = crossbeam_channel::unbounded();
        controller
            .backend
            .decode_scaled_stream_with_analysis(
                &path,
                240,
                136,
                Some(Rational::from_secs(10)),
                &native,
                tx,
            )
            .unwrap();
        let small: Vec<_> = rx.iter().map(|f| downsample_64x36_fast(&f)).collect();
        let probe = controller.backend.probe_video(&path).unwrap();
        let mut expected = detect_shots_classic(
            &small,
            probe.duration,
            Rational::from_secs(10),
            &ShotDetectorConfig::default(),
        )
        .unwrap();
        refine_shot_boundaries_from_frame_file(
            &path,
            &native,
            &mut expected,
            Rational::from_secs(10),
            fps,
            &mut |_| {},
        )
        .unwrap();
        let mut saw_provisional = false;
        controller
            .import_video_streaming(&path, |snapshot, _| {
                if snapshot.status_message.contains("provisional") {
                    saw_provisional = true;
                    assert!(snapshot.shots.iter().all(|s| s.confidence == 0.0));
                }
            })
            .unwrap();
        assert!(saw_provisional);
        assert_eq!(controller.shot_pool.shots, expected);
        let cache = CacheManager::default_local();
        let key = analysis_key(
            &cache,
            &path,
            &ShotDetectorConfig::default(),
            &SimilarityConfig::default(),
            Rational::from_secs(10),
            fps,
            probe.duration,
            (240, 136),
        )
        .unwrap();
        let cached: CachedAnalysis = cache.get(&key).unwrap();
        assert_eq!(cached.shots, expected);
        assert!(cached.shots.iter().all(|s| s.cut_confidence > 0.0));
        std::fs::remove_file(Path::new(".ae-cache").join(format!("{key}.json"))).unwrap();
    }

    #[test]
    fn refinement_scratch_is_private_and_removed() {
        let frames = RefinementFrames::new().unwrap();
        let path = frames.0.clone();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o077,
                0
            );
        }
        drop(frames);
        assert!(!path.exists());
    }

    #[test]
    fn import_reuses_raw_analysis_and_recovers_from_corruption() {
        let _guard = crate::tests::GUI_TEST_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clip.mp4");
        let fps = Rational::from_secs(10);
        generate_synthetic_multi_shot_video(
            &path,
            Resolution::new(64, 36),
            fps,
            &[([220, 20, 20], [220, 20, 20], 20)],
        )
        .unwrap();
        let mut controller = ProjectController::new();
        controller.import_video(&path).unwrap();
        let probe = controller.backend.probe_video(&path).unwrap();
        let detector = ShotDetectorConfig::default();
        let similarity = SimilarityConfig::default();
        let cache = CacheManager::default_local();
        let key = analysis_key(
            &cache,
            &path,
            &detector,
            &similarity,
            fps,
            fps,
            probe.duration,
            (320, 180),
        )
        .unwrap();
        let mut entry: CachedAnalysis = cache.get(&key).unwrap();
        assert!(entry.valid(probe.duration, fps));
        let mut changed_detector = detector.clone();
        changed_detector.mad_k_high += 1.0;
        assert_ne!(
            Some(key.clone()),
            analysis_key(
                &cache,
                &path,
                &changed_detector,
                &similarity,
                fps,
                fps,
                probe.duration,
                (320, 180)
            )
        );
        let mut changed_similarity = similarity.clone();
        changed_similarity.weights.motion += 1.0;
        assert_ne!(
            Some(key.clone()),
            analysis_key(
                &cache,
                &path,
                &detector,
                &changed_similarity,
                fps,
                fps,
                probe.duration,
                (320, 180)
            )
        );
        assert_ne!(
            Some(key.clone()),
            analysis_key(
                &cache,
                &path,
                &detector,
                &similarity,
                Rational::from_secs(5),
                fps,
                probe.duration,
                (320, 180)
            )
        );

        entry.shots[0].usable_start = Rational::from_millis(500);
        entry.shots[0].mean_luma = 0.123;
        entry.bank.by_shot[0][0].mean_luma = 0.234;
        cache.put(&key, &entry).unwrap();
        controller.shot_pool.shots[0].enabled = false;
        controller.shot_pool.shots[0].locked = true;
        controller.import_video(&path).unwrap();
        assert_eq!(controller.shot_pool.shots[0].mean_luma, 0.123);
        assert!(controller.shot_pool.shots[0].enabled);
        assert!(!controller.shot_pool.shots[0].locked);
        assert_eq!(
            controller.timeline_model.bank.by_shot[0][0].mean_luma,
            0.234
        );
        assert_eq!(
            global_media_cache()
                .read()
                .unwrap()
                .get_shot_thumbnail(0, 0)
                .pts,
            Rational::from_millis(500)
        );

        entry.shots[0].id = 4;
        cache.put(&key, &entry).unwrap();
        controller.import_video(&path).unwrap();
        assert_eq!(controller.shot_pool.shots[0].id, 0);
        assert_ne!(controller.shot_pool.shots[0].mean_luma, 0.123);
        let mut corrupt: CachedAnalysis = cache.get(&key).unwrap();
        corrupt.bank.by_shot[0].clear();
        cache.put(&key, &corrupt).unwrap();
        controller.import_video(&path).unwrap();
        assert!(!controller.timeline_model.bank.by_shot[0].is_empty());
        let cache_path = Path::new(".ae-cache").join(format!("{key}.json"));
        std::fs::write(&cache_path, b"not json").unwrap();
        controller.import_video(&path).unwrap();
        assert!(
            cache
                .get::<CachedAnalysis>(&key)
                .unwrap()
                .valid(probe.duration, fps)
        );
        std::fs::remove_file(cache_path).unwrap();
    }

    fn continuity_shot(id: usize) -> Shot {
        let start = Rational::from_secs(id as i64 * 10);
        Shot {
            id,
            start,
            end: start + Rational::from_secs(10),
            usable_start: start,
            usable_end: start + Rational::from_secs(10),
            cut_confidence: 0.9,
            boundary_kind: ae_core::ShotBoundaryKind::HardCut,
            mean_luma: 0.4,
            enabled: true,
            locked: false,
        }
    }

    fn continuity_bank_entry(
        shot_id: usize,
        pos: [f32; 2],
        motion: [f32; 2],
    ) -> ae_similarity::FrameDescriptor {
        // Appearance signals held identical so only continuity decides.
        let mut emb = vec![0.0f32; 200];
        emb[0] = 1.0;
        ae_similarity::FrameDescriptor {
            shot_id,
            pts: Rational::from_secs(shot_id as i64 * 10),
            oklab_thumb: vec![0.5; 192],
            oklab_stats: [0.5, 0.0, 0.0, 0.1, 0.0, 0.0],
            phash: vec![0xA5u8; 8],
            motion_vec: motion,
            mean_luma: 0.4,
            subject_pos: pos,
            combined_embedding: emb,
            ssim_thumb_rgba: vec![128u8; 32 * 18 * 4],
        }
    }

    #[test]
    fn scene_analysis_cache_key_is_continuity_versioned() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clip.mp4");
        std::fs::write(&path, b"fake-video-bytes").unwrap();
        let cache = CacheManager::default_local();
        let detector = ShotDetectorConfig::default();
        let similarity = SimilarityConfig::default();
        let fps = Rational::from_secs(10);
        let params = analysis_params(
            &detector,
            &similarity,
            fps,
            fps,
            Rational::from_secs(2),
            (320, 180),
        );
        let legacy = cache
            .file_cache_key(&path, LEGACY_GUI_SCENE_ANALYSIS_CACHE_TAG, &params)
            .unwrap();
        let current = analysis_key(
            &cache,
            &path,
            &detector,
            &similarity,
            fps,
            fps,
            Rational::from_secs(2),
            (320, 180),
        )
        .unwrap();
        assert_eq!(GUI_SCENE_ANALYSIS_CACHE_TAG, "gui-scene-analysis-v2");
        assert_ne!(
            legacy, current,
            "pre-continuity v1 cache entries must never hit the live lookup"
        );
    }

    #[test]
    fn cached_analysis_rejects_non_finite_subject_pos() {
        let shot = continuity_shot(0);
        let bank_with = |pos: [f32; 2]| ShotDescriptorBank {
            by_shot: vec![vec![continuity_bank_entry(0, pos, [0.2, 0.0])]],
            ..Default::default()
        };
        let good = CachedAnalysis {
            shots: vec![shot.clone()],
            bank: bank_with([0.3, 0.5]),
        };
        assert!(good.valid(Rational::from_secs(10), Rational::from_secs(10)));
        let bad = CachedAnalysis {
            shots: vec![shot],
            bank: bank_with([f32::NAN, 0.5]),
        };
        assert!(
            !bad.valid(Rational::from_secs(10), Rational::from_secs(10)),
            "corrupt subject centroids must force descriptor recomputation"
        );
    }

    #[test]
    fn default_alternative_ranking_prefers_continuity_compatible_clip() {
        // No opt-in: a stock `TimelineModel` must rank the
        // position/motion-compatible candidate first.
        let shots: Vec<Shot> = (0..3).map(continuity_shot).collect();
        let bank = ShotDescriptorBank {
            by_shot: vec![
                vec![continuity_bank_entry(0, [0.3, 0.5], [0.3, 0.0])],
                vec![continuity_bank_entry(1, [0.32, 0.5], [0.28, 0.02])],
                vec![continuity_bank_entry(2, [0.8, 0.5], [-0.3, 0.0])],
            ],
            ..Default::default()
        };
        let slot = |slot_index: usize, shot_id: usize| ae_core::TimelineSlot {
            slot_index,
            dst_start: Rational::from_secs(slot_index as i64),
            dst_end: Rational::from_secs(slot_index as i64 + 1),
            shot_id,
            src_in: Rational::from_secs(shot_id as i64 * 10),
            src_out: Rational::from_secs(shot_id as i64 * 10 + 1),
            speed: 1.0,
            transition_in: ae_core::TransitionSpec::default(),
            boundary_similarity: 1.0,
            beat_strength: 0.7,
            locked: false,
        };
        let mut model = TimelineModel::new();
        model.bank = bank;
        model.timeline = Some(Timeline {
            version: 1,
            source_video: String::new(),
            source_audio: String::new(),
            resolution: Resolution::P1080,
            fps: Rational::from_secs(30),
            total_duration: Rational::from_secs(2),
            mean_boundary_similarity: 1.0,
            slots: vec![slot(0, 0), slot(1, 2)],
        });
        let ranked = model.rank_alternatives(&shots, 1, 2);
        assert_eq!(ranked.len(), 2);
        assert_eq!(
            ranked[0].shot_id, 1,
            "continuity-compatible clip must win by default: {ranked:?}"
        );
        assert!(
            ranked[0].similarity > ranked[1].similarity + 0.02,
            "scores must separate compatible from clashing cuts: {ranked:?}"
        );
    }
}

#[cfg(test)]
mod export_config_tests {
    use super::*;

    fn empty_timeline() -> Timeline {
        Timeline {
            version: 1,
            source_video: "clip.mp4".to_string(),
            source_audio: String::new(),
            resolution: Resolution::VERTICAL,
            fps: Rational::from_secs(30),
            total_duration: Rational::ZERO,
            mean_boundary_similarity: 1.0,
            slots: Vec::new(),
        }
    }

    #[test]
    fn export_enhancement_choices_leave_quick_preview_unfiltered() {
        let mut ctrl = ProjectController::new();
        assert!(!ctrl.light_cleanup);
        assert!(!ctrl.quality_enhance);
        let tl = empty_timeline();
        for (light, ai) in [(false, false), (true, false), (false, true)] {
            ctrl.light_cleanup = light;
            ctrl.quality_enhance = ai;
            let cfg = ctrl.build_export_config(&tl, Path::new("out.mp4"), false, 23, "medium");
            assert_eq!((cfg.light_cleanup, cfg.quality_enhance), (light, ai));
            assert_eq!((cfg.crf, cfg.preset.as_str()), (23, "medium"));
            let preview = ctrl.build_quick_render_config(&tl, Path::new("preview.mp4"));
            assert!(!preview.light_cleanup);
            assert!(!preview.quality_enhance);
        }
    }

    #[test]
    fn face_following_crop_is_opt_in_with_established_model_path() {
        let ctrl = ProjectController::new();
        assert!(!ctrl.reframe_faces, "toggle must default off");

        let tl = empty_timeline();
        let out = PathBuf::from("out.mp4");
        let cfg = ctrl.build_export_config(&tl, &out, false, 18, "fast");
        assert!(!cfg.reframe_faces);
        assert_eq!(cfg.face_model_path, None);
        assert_eq!(cfg.video_path, PathBuf::from("clip.mp4"));

        let mut enabled = ProjectController::new();
        enabled.video_path = Some(PathBuf::from("custom.mp4"));
        enabled.reframe_faces = true;
        let cfg = enabled.build_export_config(&tl, &out, true, 23, "medium");
        assert!(cfg.reframe_faces);
        assert_eq!(
            cfg.face_model_path,
            Some(character_models_dir().join("anime_character.onnx")),
            "enabled toggle must use the established model path"
        );
        assert_eq!(cfg.video_path, PathBuf::from("custom.mp4"));
        assert!(cfg.use_gpu);
        assert_eq!((cfg.crf, cfg.preset), (23, "medium".to_string()));
    }

    #[test]
    fn face_model_download_rejects_corrupt_files_and_removes_partial_downloads() {
        let _guard = crate::tests::GUI_TEST_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let mut ctrl = ProjectController::new();
        assert!(
            ctrl.prepare_character_model(&tmp.path().join("missing"))
                .is_err()
        );
        ctrl.reframe_faces = true;
        let source = tmp.path().join("corrupt.onnx");
        std::fs::write(&source, b"not an ONNX model").unwrap();
        let mut manifest: ae_core::ModelManifest =
            toml::from_str(include_str!("../../../models/manifest.toml")).unwrap();
        manifest.models.get_mut("anime_character").unwrap().url =
            format!("file://{}", source.display());
        std::fs::write(
            tmp.path().join("manifest.toml"),
            toml::to_string(&manifest).unwrap(),
        )
        .unwrap();
        let error = ctrl
            .prepare_character_model(tmp.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("SHA-256 mismatch"), "{error}");
        assert!(!tmp.path().join("anime_character.onnx").exists());
        assert_eq!(
            std::fs::read_dir(tmp.path()).unwrap().count(),
            2,
            "failed download must remove its temporary file"
        );
        std::fs::write(
            tmp.path().join("anime_character.onnx"),
            b"corrupt existing model",
        )
        .unwrap();
        assert!(
            ctrl.prepare_character_model(tmp.path())
                .unwrap_err()
                .to_string()
                .contains("SHA-256 mismatch")
        );
        set_gui_progress(false, 0, "");
    }

    #[test]
    fn quick_render_preview_honors_the_same_toggle() {
        let ctrl = ProjectController::new();
        let tl = empty_timeline();
        let out = PathBuf::from(".ae-cache/quick_preview.mp4");

        let cfg = ctrl.build_quick_render_config(&tl, &out);
        assert!(!cfg.reframe_faces, "preview must default to no reframe");
        assert_eq!(cfg.face_model_path, None);
        // Preview keeps its fast/quality profile independent of the toggle.
        assert_eq!(
            (cfg.crf, cfg.preset.as_str(), cfg.use_gpu),
            (18, "ultrafast", false)
        );
        assert!(!cfg.quality_enhance);

        let mut enabled = ProjectController::new();
        enabled.reframe_faces = true;
        let cfg = enabled.build_quick_render_config(&tl, &out);
        assert!(cfg.reframe_faces, "preview must follow the toggle");
        assert_eq!(
            cfg.face_model_path,
            Some(character_models_dir().join("anime_character.onnx")),
            "preview must use the established model path"
        );
    }
}
