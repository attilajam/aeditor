use std::path::{Path, PathBuf};
use std::thread;

use ae_beats::write_debug_artifacts;
use ae_core::{Rational, Resolution, Result, RgbaFrame, VideoCodec};
use ae_media::{FfmpegBackend, MediaBackend};
use ae_render::{
    IncrementalDraftReport, RenderConfig, render_timeline, render_timeline_incremental_draft,
};
use ae_shots::{ShotDetectorConfig, detect_shots_classic};
use ae_similarity::{SimilarityConfig, build_descriptor_bank_from_frames};
use crossbeam_channel::bounded;
use serde::{Deserialize, Serialize};

use crate::beat_editor_model::{BeatEditorModel, BeatViewItem};
use crate::image_provider::{global_media_cache, set_gui_progress};
use crate::shot_pool_model::{ShotPoolModel, ShotViewItem};
use crate::timeline_model::{AlternativeShotItem, TimelineModel, TimelineSlotViewItem};

/// Full snapshot of the GUI state serialized to QML whenever an action completes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GuiStateSnapshot {
    pub video_path: String,
    pub audio_path: String,
    pub click_track_path: String,
    pub draft_video_path: String,
    pub resolution: String,
    pub target_width: u32,
    pub target_height: u32,
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
    pub shots: Vec<ShotViewItem>,
    pub beats: Vec<BeatViewItem>,
    pub slots: Vec<TimelineSlotViewItem>,
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
    /// at an adaptive frame rate so even 25-minute episodes import in a few seconds with bounded
    /// RAM and without starving the UI thread.
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

        set_gui_progress(true, 8, "Streaming video frames & detecting scenes live...");
        let (tx, rx) = bounded::<RgbaFrame>(64);
        let path_owned = path.to_path_buf();
        let backend = self.backend.clone();

        let (decoded_frames, small_64) =
            thread::scope(|s| -> Result<(Vec<RgbaFrame>, Vec<RgbaFrame>)> {
                let handle = s.spawn(move || {
                    backend.decode_scaled_stream(&path_owned, ing_w, ing_h, Some(ingest_fps), tx)
                });
                let mut decoded = Vec::with_capacity(expected_frames);
                let mut small = Vec::with_capacity(expected_frames);
                let mut last_partial_emit = std::time::Instant::now();
                let mut cached_shots_count = 0usize;

                while let Ok(f) = rx.recv() {
                    small.push(downsample_64x36_fast(&f));
                    decoded.push(f);

                    if decoded.len().is_multiple_of(16) || decoded.len() == expected_frames {
                        let ratio =
                            (decoded.len() as f32 / expected_frames as f32).clamp(0.0, 1.0);
                        let pct = 8 + (ratio * 66.0) as u32;
                        set_gui_progress(
                            true,
                            pct,
                            format!(
                                "Scanning video & splitting scenes ({}/{} frames)...",
                                decoded.len(),
                                expected_frames
                            ),
                        );

                        // Every 450ms, emit newly discovered scenes incrementally
                        if last_partial_emit.elapsed().as_millis() >= 450 && small.len() >= 12 {
                            let cur_dur = decoded.last().map(|fr| fr.pts).unwrap_or(probe.duration);
                            if let Ok(partial_shots) = detect_shots_classic(
                                &small,
                                cur_dur,
                                ingest_fps,
                                &ShotDetectorConfig::default(),
                            ) {
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

        // Finalize shot detection across all frames
        set_gui_progress(true, 76, "Finalizing shot boundaries...");
        self.shot_pool.shots = detect_shots_classic(
            &small_64,
            probe.duration,
            ingest_fps,
            &ShotDetectorConfig::default(),
        )?;

        // Slice start/mid/end thumbnails for every shot directly from `decoded_frames` in RAM
        set_gui_progress(true, 83, "Indexing scene thumbnails in RAM...");
        self.populate_thumbnails_from_ram(&decoded_frames);
        on_update(self.snapshot(), true);

        // Sample 3 frames per shot (start, mid, end) for `build_descriptor_bank_from_frames`
        set_gui_progress(true, 89, "Computing visual similarity descriptors...");
        let sampled_for_sim = self.sample_three_frames_per_shot(&small_64);
        self.timeline_model.bank = build_descriptor_bank_from_frames(
            &sampled_for_sim,
            &self.shot_pool.shots,
            &SimilarityConfig::default(),
        )?;

        // Store proxy reel in GuiMediaCache
        set_gui_progress(true, 95, "Populating instant RAM preview reel...");
        if let Ok(mut cache) = global_media_cache().write() {
            cache.set_proxy_reel(decoded_frames);
        }

        self.status_message = format!(
            "Detected {} scenes in '{}' ({:.1}s) — Instant RAM Preview ready.",
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

        let slice_end = end_shot_idx.min(self.shot_pool.shots.len());
        let mut prepared = Vec::with_capacity(slice_end - start_shot_idx);
        for shot in &self.shot_pool.shots[start_shot_idx..slice_end] {
            let mid = Rational::from_secs_f64(
                0.5 * (shot.usable_start.as_secs_f64() + shot.usable_end.as_secs_f64()),
            );
            let end_pts = (shot.usable_end - Rational::from_millis(50)).max(shot.usable_start);
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

    pub fn set_resolution_str(&mut self, res_str: &str) -> Result<Resolution> {
        let res: Resolution = res_str.parse()?;
        self.resolution = res;
        self.timeline_model.set_resolution(res);
        self.status_message = format!("Output resolution set to {} (stretch to fill).", res);
        Ok(res)
    }

    pub fn resequence(&mut self) -> Result<()> {
        if self.shot_pool.shots.is_empty() {
            return Ok(());
        }
        let tl = self
            .timeline_model
            .resequence(&self.shot_pool.shots, &self.beat_editor.beats)?
            .clone();
        self.sync_timeline_to_preview_cache();
        self.status_message = format!(
            "Sequenced {} unique slots (hard cut default, mean match similarity {:.1}%).",
            tl.slots.len(),
            tl.mean_boundary_similarity * 100.0
        );
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
        // Automatically enable the shot if the user explicitly dragged/swapped it into the timeline
        self.shot_pool.set_shot_enabled(new_shot_id, true);
        self.timeline_model
            .swap_slot_shot(&self.shot_pool.shots, slot_index, new_shot_id)?;
        self.sync_timeline_to_preview_cache();
        self.status_message = format!("Assigned Scene #{new_shot_id:02} to Slot #{slot_index}.");
        Ok(())
    }

    pub fn slip_slot_in_point(&mut self, slot_index: usize, delta_sec: f64) -> Result<()> {
        let new_in = self
            .timeline_model
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
        self.status_message = format!(
            "Slot #{slot_index} transition set to '{transition_id}' ({duration_ms}ms)."
        );
        Ok(())
    }

    /// Phase P3: Incremental slot-cached Quick Render into `.ae-cache/quick_preview.mp4`.
    pub fn quick_render_draft(&mut self, scale_divisor: u32) -> Result<IncrementalDraftReport> {
        let Some(ref tl) = self.timeline_model.timeline else {
            return Err(ae_core::AeError::Config(
                "no timeline to quick-render — import video and audio first".into(),
            ));
        };
        let cache_dir = PathBuf::from(".ae-cache");
        let out_path = cache_dir.join("quick_preview.mp4");
        let video_path = self
            .video_path
            .clone()
            .unwrap_or_else(|| PathBuf::from(&tl.source_video));

        set_gui_progress(true, 8, "Starting incremental Quick Render...");
        let cfg = RenderConfig {
            video_path,
            audio_path: self.audio_path.clone(),
            output_path: out_path.clone(),
            resolution: Some(self.resolution),
            fps: Some(self.fps),
            codec: VideoCodec::X264,
            crf: 26,
            preset: "ultrafast".to_string(),
            use_gpu: false,
            shaders_dir: Some(PathBuf::from("transitions")),
            cache_capacity: 16,
            progress_cb: Some(std::sync::Arc::new(|pct, msg| {
                set_gui_progress(true, pct, msg.to_string());
            })),
        };

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
            .rank_alternatives(&self.shot_pool.shots, slot_index, top_k)
    }

    /// Export the current timeline to `output_path` stretched to `self.resolution`.
    pub fn export_video(&mut self, output_path: &Path, use_gpu: bool) -> Result<()> {
        let Some(ref tl) = self.timeline_model.timeline else {
            return Err(ae_core::AeError::Config(
                "no timeline to export — import video and audio first".into(),
            ));
        };
        set_gui_progress(
            true,
            25,
            format!(
                "Rendering {} slots at {} (stretch to fill)...",
                tl.slots.len(),
                self.resolution
            ),
        );
        let video_path = self
            .video_path
            .clone()
            .unwrap_or_else(|| PathBuf::from(&tl.source_video));

        let cfg = RenderConfig {
            video_path,
            audio_path: self.audio_path.clone(),
            output_path: output_path.to_path_buf(),
            resolution: Some(self.resolution),
            fps: Some(self.fps),
            codec: VideoCodec::X264,
            crf: 18,
            preset: "fast".to_string(),
            use_gpu,
            shaders_dir: Some(PathBuf::from("transitions")),
            cache_capacity: 16,
            progress_cb: Some(std::sync::Arc::new(|pct, msg| {
                set_gui_progress(true, pct, msg.to_string());
            })),
        };

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
        let shots = self.shot_pool.to_view_items(&assignments);
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
