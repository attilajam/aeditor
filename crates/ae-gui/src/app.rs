use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use ae_core::{BeatMode, Rational, Resolution, RgbaFrame, SeparationMode};
use ae_media::{FfmpegBackend, MediaBackend};
use crossbeam_channel::{Receiver, Sender, unbounded};
use egui::{
    Align, Color32, ColorImage, Context, CornerRadius, FontId, Key, Layout, Modifiers, Pos2, Rect,
    RichText, ScrollArea, Sense, Stroke, StrokeKind, TextureHandle, TextureOptions, Ui, Vec2,
};

use crate::character_analysis::{CharacterAnalysisOutput, analyze_characters};
use crate::character_edits::CharacterEdit;
use crate::image_provider::{get_gui_progress, global_media_cache, set_gui_progress};
use crate::project_controller::{GuiStateSnapshot, ProjectController};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileDialogKind {
    ImportVideo,
    ImportAudio,
    ExportVideo,
    ImportStyle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeftTab {
    Media,
    Audio,
    Transitions,
    Settings,
}

impl LeftTab {
    fn label(&self) -> &'static str {
        match self {
            LeftTab::Media => "Media",
            LeftTab::Audio => "Audio",
            LeftTab::Transitions => "Transitions",
            LeftTab::Settings => "Settings",
        }
    }
}

pub struct AutoEditorEguiApp {
    pub controller: Arc<Mutex<ProjectController>>,
    pub snapshot: GuiStateSnapshot,
    snapshot_tx: Sender<(u64, GuiStateSnapshot, bool)>, // (action_seq, new_snapshot, clear_shot_textures)
    snapshot_rx: Receiver<(u64, GuiStateSnapshot, bool)>,
    action_seq: u64,
    pending_resequence: bool,
    preview_needs_render: bool,
    pub selected_shot_id: usize,
    pub selected_slot_idx: usize,
    selected_transition: Option<&'static str>,
    pub live_timeline_mode: bool,
    pub res_input: String,
    pub use_gpu_export: bool,
    quality_enhance_export: bool,
    light_cleanup_export: bool,
    reframe_faces_export: bool,
    compare_enhancement: bool,
    enhancement_split: f32,
    enhanced_video_path: Option<PathBuf>,
    enhancement_tx: Sender<(u64, Result<PathBuf, String>)>,
    enhancement_rx: Receiver<(u64, Result<PathBuf, String>)>,
    enhancement_pending: bool,
    enhancement_error: Option<String>,

    // 5K / HiDPI UI scaling
    pub ui_zoom: f32,
    pub zoom_initialized: bool,

    // Live playback & loop-cut state
    pub playing: bool,
    pub playhead_sec: f64,
    pub last_tick: Instant,
    pub click_track_enabled: bool,
    pub loop_enabled: bool,
    pub loop_start_sec: f64,
    pub loop_end_sec: f64,
    audio_child: Option<Child>,

    // Dragging state on the Bass Beat Canvas & Drag-and-Drop from Cut-Up Scenes to Timeline
    dragging_beat_idx: Option<usize>,
    dragging_shot_id: Option<usize>,

    // Native file dialogs remember the last browsed directory.
    last_browse_dir: Option<PathBuf>,

    // GPU Texture caches & per-frame upload budget so 260+ scenes never spike frame time
    shot_textures: HashMap<(usize, u8), TextureHandle>,
    character_textures: HashMap<(u64, u64), TextureHandle>,
    character_rename: Option<(u64, String)>,
    character_review_open: bool,
    character_scene_review: Option<(usize, Vec<u64>)>,
    character_tx: Sender<(u64, u64, PathBuf, Result<CharacterAnalysisOutput, String>)>,
    character_rx: Receiver<(u64, u64, PathBuf, Result<CharacterAnalysisOutput, String>)>,
    character_generation: u64,
    character_cancel: Arc<AtomicBool>,
    character_analysis_running: bool,
    placeholder_tex: Option<TextureHandle>,
    uploads_this_frame: usize,
    preview_tex: Option<(u64, u64, TextureHandle)>,
    draft_preview: Option<(u64, u64, TextureHandle, Option<TextureHandle>)>,
    draft_frame_tx: Sender<(u64, Option<PathBuf>, Result<DraftPreviewFrames, String>)>,
    draft_frame_rx: Receiver<(u64, Option<PathBuf>, Result<DraftPreviewFrames, String>)>,
    draft_frames: VecDeque<DraftPreviewFrames>,
    draft_frame_pending: bool,

    // Parity layout state (CapCut-style studio)
    pub left_tab: LeftTab,
    pub search_query: String,
    pub sort_asc: bool,
    pub timeline_zoom: f32,
    selected_beat_idx: Option<usize>,
    spectrogram_tex: Option<(u64, TextureHandle)>,
    scrub_tex: Option<(usize, ae_core::Rational, u64, TextureHandle)>,
    export_crf: u32,
    export_preset: String,
}

impl AutoEditorEguiApp {
    pub fn new(ctrl: ProjectController) -> Self {
        let snap = ctrl.snapshot();
        let res_input = snap.resolution.clone();
        let (snapshot_tx, snapshot_rx) = unbounded();
        let (draft_frame_tx, draft_frame_rx) = unbounded();
        let (enhancement_tx, enhancement_rx) = unbounded();
        let (character_tx, character_rx) = unbounded();
        let quality_enhance_export = ctrl.quality_enhance;
        let light_cleanup_export = ctrl.light_cleanup;
        let reframe_faces_export = ctrl.reframe_faces;
        Self {
            controller: Arc::new(Mutex::new(ctrl)),
            snapshot: snap,
            snapshot_tx,
            snapshot_rx,
            action_seq: 0,
            pending_resequence: false,
            preview_needs_render: true,
            selected_shot_id: 0,
            selected_slot_idx: 0,
            selected_transition: None,
            live_timeline_mode: true,
            res_input,
            use_gpu_export: false,
            quality_enhance_export,
            light_cleanup_export,
            reframe_faces_export,
            compare_enhancement: false,
            enhancement_split: 0.5,
            enhanced_video_path: None,
            enhancement_tx,
            enhancement_rx,
            enhancement_pending: false,
            enhancement_error: None,
            ui_zoom: 1.0, // Refined for larger monitors on the first frame
            zoom_initialized: false,
            playing: false,
            playhead_sec: 0.0,
            last_tick: Instant::now(),
            click_track_enabled: false,
            loop_enabled: false,
            loop_start_sec: 0.0,
            loop_end_sec: 0.0,
            audio_child: None,
            dragging_beat_idx: None,
            dragging_shot_id: None,
            last_browse_dir: None,
            shot_textures: HashMap::new(),
            character_textures: HashMap::new(),
            character_rename: None,
            character_review_open: false,
            character_scene_review: None,
            character_tx,
            character_rx,
            character_generation: 0,
            character_cancel: Arc::new(AtomicBool::new(false)),
            character_analysis_running: false,
            placeholder_tex: None,
            uploads_this_frame: 0,
            preview_tex: None,
            draft_preview: None,
            draft_frame_tx,
            draft_frame_rx,
            draft_frame_pending: false,
            draft_frames: VecDeque::new(),
            left_tab: LeftTab::Media,
            search_query: String::new(),
            sort_asc: true,
            timeline_zoom: 1.0,
            selected_beat_idx: None,
            spectrogram_tex: None,
            scrub_tex: None,
            export_crf: 18,
            export_preset: "fast".to_string(),
        }
    }

    fn rgba_to_color_image(frame: &RgbaFrame) -> ColorImage {
        ColorImage::from_rgba_unmultiplied(
            [frame.width as usize, frame.height as usize],
            &frame.data,
        )
    }

    fn get_placeholder_texture(&mut self, ctx: &Context) -> TextureHandle {
        if let Some(ref tex) = self.placeholder_tex {
            return tex.clone();
        }
        let frame = RgbaFrame::solid(64, 36, [28, 32, 44, 255], ae_core::Rational::ZERO);
        let tex = ctx.load_texture(
            "shot_placeholder",
            Self::rgba_to_color_image(&frame),
            TextureOptions::LINEAR,
        );
        self.placeholder_tex = Some(tex.clone());
        tex
    }

    fn get_character_texture(&mut self, ctx: &Context, id: u64) -> TextureHandle {
        let revision = self.snapshot.character_revision;
        let key = (revision, id);
        if let Some(texture) = self.character_textures.get(&key) {
            return texture.clone();
        }
        if self.uploads_this_frame >= 8 {
            ctx.request_repaint();
            return self.get_placeholder_texture(ctx);
        }
        let frame = if let Ok(cache) = global_media_cache().try_read() {
            if cache.character_revision == revision {
                cache.character_portraits.get(&id).cloned()
            } else {
                None
            }
        } else {
            ctx.request_repaint();
            None
        };
        if let Some(frame) = frame {
            self.uploads_this_frame += 1;
            let texture = ctx.load_texture(
                format!("character_{revision}_{id}"),
                Self::rgba_to_color_image(&frame),
                TextureOptions::LINEAR,
            );
            self.character_textures.insert(key, texture.clone());
            texture
        } else {
            self.get_placeholder_texture(ctx)
        }
    }

    fn get_shot_texture(&mut self, ctx: &Context, shot_id: usize, pos: u8) -> TextureHandle {
        let key = (shot_id, pos.min(2));
        if let Some(tex) = self.shot_textures.get(&key) {
            return tex.clone();
        }
        // Budget GPU texture uploads to at most 8 per frame so scrolling 260+ scenes stays at 60fps
        if self.uploads_this_frame >= 8 {
            ctx.request_repaint();
            return self.get_placeholder_texture(ctx);
        }
        let Ok(cache) = global_media_cache().try_read() else {
            ctx.request_repaint();
            return self.get_placeholder_texture(ctx);
        };
        let frame = cache.get_shot_thumbnail(shot_id, pos);
        drop(cache);

        self.uploads_this_frame += 1;
        let img = Self::rgba_to_color_image(&frame);
        let tex = ctx.load_texture(format!("shot_{shot_id}_{pos}"), img, TextureOptions::LINEAR);
        self.shot_textures.insert(key, tex.clone());
        tex
    }

    fn get_scrub_texture(&mut self, ctx: &Context, shot_id: usize, position: f32) -> TextureHandle {
        let Some(shot) = self.snapshot.shots.iter().find(|shot| shot.id == shot_id) else {
            return self.get_shot_texture(ctx, shot_id, 1);
        };
        let start = ae_core::Rational::from_secs_f64(shot.start_sec);
        let end = ae_core::Rational::from_secs_f64(shot.end_sec);
        let Ok(cache) = global_media_cache().try_read() else {
            return self.get_shot_texture(ctx, shot_id, 1);
        };
        let revision = cache.preview_revision;
        let frame = cache.source_preview_frame(shot_id, start, end, position);
        drop(cache);
        if let Some((id, pts, rev, texture)) = &self.scrub_tex {
            if *id == shot_id && *pts == frame.pts && *rev == revision {
                return texture.clone();
            }
        }
        let image = Self::rgba_to_color_image(&frame);
        let texture = if let Some((_, _, _, mut texture)) = self.scrub_tex.take() {
            texture.set(image, TextureOptions::LINEAR);
            texture
        } else {
            ctx.load_texture("source_scrub", image, TextureOptions::LINEAR)
        };
        self.scrub_tex = Some((shot_id, frame.pts, revision, texture.clone()));
        texture
    }

    fn get_spectrogram_texture(&mut self, ctx: &Context) -> TextureHandle {
        let Ok(cache) = global_media_cache().try_read() else {
            return self.get_placeholder_texture(ctx);
        };
        let revision = cache.spectrogram_revision;
        if let Some((rev, texture)) = &self.spectrogram_tex {
            if *rev == revision {
                return texture.clone();
            }
        }
        let frame = cache.get_spectrogram();
        drop(cache);
        let texture = ctx.load_texture(
            "audio_spectrogram",
            Self::rgba_to_color_image(&frame),
            TextureOptions::LINEAR,
        );
        self.spectrogram_tex = Some((revision, texture.clone()));
        texture
    }

    fn get_preview_texture(
        &mut self,
        ctx: &Context,
        playhead_ms: u64,
    ) -> (TextureHandle, Option<TextureHandle>) {
        self.draft_frames.retain(|chunk| {
            chunk.seq == self.action_seq && chunk.enhanced_path == self.enhanced_video_path
        });
        while let Ok((seq, enhanced_path, result)) = self.draft_frame_rx.try_recv() {
            self.draft_frame_pending = false;
            if seq == self.action_seq
                && enhanced_path == self.enhanced_video_path
                && !self.snapshot.draft_video_path.is_empty()
            {
                match result {
                    Ok(chunk) => {
                        self.draft_frames.push_back(chunk);
                        while self.draft_frames.len() > 2 {
                            self.draft_frames.pop_front();
                        }
                    }
                    Err(error) => {
                        self.snapshot.status_message = format!("Could not preview draft: {error}");
                        if self.enhanced_video_path.is_some() {
                            self.enhancement_error = Some(self.snapshot.status_message.clone());
                            self.enhanced_video_path = None;
                            self.compare_enhancement = false;
                            self.draft_preview = None;
                        } else {
                            self.snapshot.draft_video_path.clear();
                        }
                    }
                }
            }
        }
        if !self.snapshot.draft_video_path.is_empty() {
            let last_frame = (self.snapshot.timeline_duration_sec * 30.0).round() as u64;
            let frame_index = (playhead_ms * 30 / 1000).min(last_frame.saturating_sub(1));
            let chunk_start = frame_index / 30 * 30;
            if let Some(chunk) = self
                .draft_frames
                .iter()
                .find(|c| c.start_frame == chunk_start)
            {
                if self
                    .draft_preview
                    .as_ref()
                    .is_none_or(|(seq, index, _, after)| {
                        *seq != self.action_seq
                            || *index != frame_index
                            || after.is_some() != self.enhanced_video_path.is_some()
                    })
                {
                    let index = (frame_index - chunk_start) as usize;
                    // Hold the last frame at the end of the rendered video.
                    let frame = &chunk.frames[index.min(chunk.frames.len() - 1)];
                    let texture = ctx.load_texture(
                        "draft_preview",
                        Self::rgba_to_color_image(frame),
                        TextureOptions::LINEAR,
                    );
                    let after = chunk.enhanced.as_ref().map(|frames| {
                        ctx.load_texture(
                            "enhanced_preview",
                            Self::rgba_to_color_image(&frames[index.min(frames.len() - 1)]),
                            TextureOptions::LINEAR,
                        )
                    });
                    self.draft_preview = Some((self.action_seq, frame_index, texture, after));
                }
            }
            let requested_start = if self
                .draft_frames
                .iter()
                .any(|c| c.start_frame == chunk_start)
            {
                chunk_start + 30
            } else {
                chunk_start
            };
            if !self.draft_frame_pending
                && (requested_start == chunk_start || self.playing)
                && (requested_start as f64 / 30.0) < self.snapshot.timeline_duration_sec
                && !self
                    .draft_frames
                    .iter()
                    .any(|c| c.start_frame == requested_start)
            {
                self.draft_frame_pending = true;
                let path = PathBuf::from(&self.snapshot.draft_video_path);
                let enhanced_path = self.enhanced_video_path.clone();
                let seq = self.action_seq;
                let tx = self.draft_frame_tx.clone();
                let ctx = ctx.clone();
                let duration = self.snapshot.timeline_duration_sec;
                thread::spawn(move || {
                    let result = decode_preview_chunk(
                        &path,
                        enhanced_path.as_deref(),
                        requested_start,
                        duration,
                    )
                    .map(|mut chunk| {
                        chunk.seq = seq;
                        chunk.enhanced_path = enhanced_path.clone();
                        chunk
                    })
                    .map_err(|error| error.to_string());
                    let _ = tx.send((seq, enhanced_path, result));
                    ctx.request_repaint();
                });
            }
            if let Some((seq, _, texture, after)) = &self.draft_preview
                && *seq == self.action_seq
                && after.is_some() == self.enhanced_video_path.is_some()
            {
                return (texture.clone(), after.clone());
            }
            // Never substitute an uncropped proxy while a rendered frame is loading.
            return (self.get_placeholder_texture(ctx), None);
        }
        let rev = self.snapshot.preview_revision;
        let bucket_ms = (playhead_ms / 33) * 33; // 30 fps frame quantization
        if let Some((c_ms, c_rev, tex)) = &self.preview_tex
            && *c_ms == bucket_ms
            && *c_rev == rev
        {
            return (tex.clone(), None);
        }
        let Ok(cache) = global_media_cache().try_read() else {
            return self
                .preview_tex
                .as_ref()
                .map(|(_, _, t)| (t.clone(), None))
                .unwrap_or_else(|| (self.get_placeholder_texture(ctx), None));
        };
        let frame = cache.resolve_timeline_preview_frame(bucket_ms).0;
        drop(cache);

        let img = Self::rgba_to_color_image(&frame);
        let tex = ctx.load_texture("live_preview", img, TextureOptions::LINEAR);
        self.preview_tex = Some((bucket_ms, rev, tex.clone()));
        (tex, None)
    }

    fn poll_enhanced_render(&mut self) {
        while let Ok((seq, result)) = self.enhancement_rx.try_recv() {
            self.enhancement_pending = false;
            if seq != self.action_seq {
                continue;
            }
            match result {
                Ok(path) => {
                    self.enhanced_video_path = Some(path);
                    self.draft_preview = None;
                    self.snapshot.status_message = "Filtered preview ready.".into();
                }
                Err(error) => {
                    self.enhancement_error = Some(format!("Preview filtering failed: {error}"));
                    self.compare_enhancement = false;
                }
            }
        }
    }

    fn spawn_enhanced_render(&mut self, ctx: Context) {
        self.enhancement_pending = true;
        self.playing = false;
        self.stop_audio_companion();
        let seq = self.action_seq;
        let controller = Arc::clone(&self.controller);
        let tx = self.enhancement_tx.clone();
        set_gui_progress(true, 2, "Enhancing the whole preview video…");
        thread::spawn(move || {
            let result = controller
                .lock()
                .map_err(|e| e.to_string())
                .and_then(|mut ctrl| ctrl.enhance_draft().map_err(|e| e.to_string()));
            if let Err(error) = &result {
                set_gui_progress(false, 0, format!("Preview filtering failed: {error}"));
            }
            let _ = tx.send((seq, result));
            ctx.request_repaint();
        });
    }

    fn paint_enhancement_comparison(&mut self, ui: &mut Ui, rect: Rect, after: &TextureHandle) {
        let response = ui
            .interact(
                rect,
                ui.scope_id().with("enhancement_divider"),
                Sense::click_and_drag(),
            )
            .on_hover_cursor(egui::CursorIcon::ResizeHorizontal)
            .on_hover_text("Drag left or right to compare before and after");
        if (response.clicked() || response.dragged())
            && let Some(pos) = response.interact_pointer_pos()
        {
            self.enhancement_split = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
        }
        let x = rect.left() + rect.width() * self.enhancement_split;
        let after_rect = Rect::from_min_max(Pos2::new(x, rect.top()), rect.max);
        ui.painter().with_clip_rect(after_rect).image(
            after.id(),
            rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
        ui.painter().line_segment(
            [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
            Stroke::new(3.0, Color32::BLACK),
        );
        ui.painter().line_segment(
            [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
            Stroke::new(1.5, Color32::WHITE),
        );
        let handle = Pos2::new(x, rect.center().y);
        ui.painter()
            .circle_filled(handle, 13.0, Color32::from_rgb(30, 34, 42));
        ui.painter()
            .circle_stroke(handle, 13.0, Stroke::new(1.5, Color32::WHITE));
        ui.painter().text(
            handle,
            egui::Align2::CENTER_CENTER,
            "↔",
            FontId::proportional(15.0),
            Color32::WHITE,
        );
        for (label, pos, align) in [
            (
                "Before",
                rect.left_top() + Vec2::splat(8.0),
                egui::Align2::LEFT_TOP,
            ),
            (
                "After · AI",
                rect.right_top() + Vec2::new(-8.0, 8.0),
                egui::Align2::RIGHT_TOP,
            ),
        ] {
            ui.painter().text(
                pos,
                align,
                label,
                FontId::proportional(13.0),
                Color32::WHITE,
            );
        }
    }

    fn stop_audio_companion(&mut self) {
        if let Some(mut child) = self.audio_child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    fn restart_audio_companion(&mut self) {
        self.stop_audio_companion();
        if !self.playing {
            return;
        }
        let audio_file = if self.click_track_enabled && !self.snapshot.click_track_path.is_empty() {
            &self.snapshot.click_track_path
        } else {
            &self.snapshot.audio_path
        };
        if audio_file.is_empty() {
            return;
        }
        if let Ok(child) = Command::new("ffplay")
            .args([
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                "-ss",
                &format!("{:.3}", self.playhead_sec.max(0.0)),
                audio_file,
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            self.audio_child = Some(child);
        }
    }

    fn toggle_play_pause(&mut self) {
        if self.snapshot.slots.is_empty() && self.snapshot.audio_duration_sec <= 0.0 {
            return;
        }
        self.playing = !self.playing;
        self.last_tick = Instant::now();
        if self.playing {
            self.restart_audio_companion();
        } else {
            self.stop_audio_companion();
        }
    }

    fn seek_to(&mut self, sec: f64) {
        let max_dur = self
            .snapshot
            .timeline_duration_sec
            .max(self.snapshot.audio_duration_sec)
            .max(0.1);
        self.playhead_sec = sec.clamp(0.0, max_dur);
        self.last_tick = Instant::now();
        if self.playing {
            self.restart_audio_companion();
        }
    }

    fn fmt_tc(sec: f64) -> String {
        let s = sec.max(0.0);
        let h = (s / 3600.0).floor() as u64;
        let m = ((s % 3600.0) / 60.0).floor() as u64;
        let ss = (s % 60.0).floor() as u64;
        let ff = ((s.fract() * 30.0).round() as u64).min(29);
        format!("{h:02}:{m:02}:{ss:02}:{ff:02}")
    }

    fn fmt_dur(sec: f64) -> String {
        let s = sec.max(0.0);
        let m = (s / 60.0).floor() as u64;
        let ss = (s % 60.0).floor() as u64;
        format!("{m}:{ss:02}")
    }

    fn media_display_name(&self, shot_id: usize) -> String {
        let stem = std::path::Path::new(&self.snapshot.video_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("scene");
        format!("{stem}_s{shot_id:02}.mp4")
    }

    fn audio_display_name(&self) -> String {
        if self.snapshot.audio_path.is_empty() {
            "no_audio.mp3".to_string()
        } else {
            std::path::Path::new(&self.snapshot.audio_path)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("audio.mp3")
                .to_string()
        }
    }

    /// Dispatch any controller mutation + re-sequencing to a background worker thread so    /// the egui UI thread never blocks for even 1 millisecond when clicking/dragging beats or scenes.
    /// Uses a monotonic `action_seq` so stale background results never overwrite newer user edits.
    fn spawn_controller_action<F>(&mut self, ctx: Context, clear_shot_tex: bool, action: F)
    where
        F: FnOnce(&mut ProjectController) + Send + 'static,
    {
        self.action_seq = self.action_seq.wrapping_add(1);
        let seq = self.action_seq;
        self.pending_resequence = true;
        self.preview_needs_render = true;
        self.snapshot.draft_video_path.clear();
        self.enhanced_video_path = None;
        self.enhancement_error = None;
        let ctrl_arc = Arc::clone(&self.controller);
        let tx = self.snapshot_tx.clone();
        thread::spawn(move || {
            if let Ok(mut ctrl) = ctrl_arc.lock() {
                ctrl.draft_video_path = None;
                action(&mut ctrl);
                let _ = tx.send((seq, ctrl.snapshot(), clear_shot_tex));
                ctx.request_repaint();
            }
        });
    }

    fn set_reframe_faces(&mut self, ctx: Context, enabled: bool) {
        self.reframe_faces_export = enabled;
        self.spawn_controller_action(ctx, false, move |ctrl| {
            ctrl.reframe_faces = enabled;
        });
    }

    fn start_character_analysis(&mut self, ctx: Context) {
        if self.character_analysis_running {
            return;
        }
        let inputs = self.controller.try_lock().ok().and_then(|controller| {
            controller
                .video_path
                .clone()
                .filter(|_| !controller.shot_pool.shots.is_empty())
                .map(|path| {
                    (
                        controller.backend.clone(),
                        path,
                        controller.shot_pool.shots.clone(),
                    )
                })
        });
        let Some((backend, path, shots)) = inputs else {
            return;
        };
        self.character_cancel = Arc::new(AtomicBool::new(false));
        let cancel = Arc::clone(&self.character_cancel);
        let generation = self.character_generation;
        let revision = self.snapshot.character_revision;
        let tx = self.character_tx.clone();
        self.character_analysis_running = true;
        set_gui_progress(true, 1, "Analyzing animated characters…");
        thread::spawn(move || {
            let result = analyze_characters(&backend, &path, &shots, &cancel)
                .map_err(|error| error.to_string());
            let _ = tx.send((generation, revision, path, result));
            ctx.request_repaint();
        });
    }

    fn poll_character_analysis(&mut self, ctx: &Context) {
        while let Ok((generation, revision, path, result)) = self.character_rx.try_recv() {
            if generation != self.character_generation
                || path.to_string_lossy() != self.snapshot.video_path
            {
                continue;
            }
            self.character_analysis_running = false;
            set_gui_progress(false, 100, "Character analysis finished.");
            if revision != self.snapshot.character_revision {
                continue;
            }
            self.spawn_controller_action(ctx.clone(), false, move |controller| {
                let result = result
                    .map_err(ae_core::AeError::Config)
                    .and_then(|output| controller.install_character_analysis(output));
                if let Err(error) = result {
                    controller.character_analysis_error = error.to_string();
                    controller.status_message = format!("Character analysis: {error}");
                }
            });
        }
    }

    fn start_import_video(&mut self, ctx: Context, path: PathBuf) {
        self.character_cancel.store(true, Ordering::Relaxed);
        self.character_generation = self.character_generation.wrapping_add(1);
        self.character_analysis_running = false;
        self.character_textures.clear();
        self.character_rename = None;
        self.character_review_open = false;
        self.character_scene_review = None;
        self.preview_needs_render = true;
        self.snapshot.draft_video_path.clear();
        self.enhanced_video_path = None;
        self.enhancement_error = None;
        self.last_browse_dir = path.parent().map(|p| p.to_path_buf());
        self.action_seq = self.action_seq.wrapping_add(1);
        let seq = self.action_seq;
        let ctrl_arc = Arc::clone(&self.controller);
        let tx = self.snapshot_tx.clone();
        set_gui_progress(true, 2, "Starting video import...");
        ctx.request_repaint();
        thread::spawn(move || {
            if let Ok(mut ctrl) = ctrl_arc.lock() {
                ctrl.draft_video_path = None;
                let tx_progress = tx.clone();
                let ctx_progress = ctx.clone();
                if let Err(e) = ctrl.import_video_streaming(&path, move |snap, clear_tex| {
                    let _ = tx_progress.send((seq, snap, clear_tex));
                    ctx_progress.request_repaint();
                }) {
                    ctrl.status_message = format!("Import video error: {e}");
                    set_gui_progress(false, 0, ctrl.status_message.clone());
                    let _ = tx.send((seq, ctrl.snapshot(), false));
                }
            }
            ctx.request_repaint();
        });
    }

    fn start_import_audio(&mut self, ctx: Context, path: PathBuf) {
        self.preview_needs_render = true;
        self.snapshot.draft_video_path.clear();
        self.enhanced_video_path = None;
        self.enhancement_error = None;
        self.last_browse_dir = path.parent().map(|p| p.to_path_buf());
        self.action_seq = self.action_seq.wrapping_add(1);
        let seq = self.action_seq;
        let ctrl_arc = Arc::clone(&self.controller);
        let tx = self.snapshot_tx.clone();
        set_gui_progress(true, 5, "Starting audio import...");
        ctx.request_repaint();
        thread::spawn(move || {
            if let Ok(mut ctrl) = ctrl_arc.lock() {
                ctrl.draft_video_path = None;
                if let Err(e) = ctrl.import_audio(&path) {
                    ctrl.status_message = format!("Import audio error: {e}");
                    set_gui_progress(false, 0, ctrl.status_message.clone());
                }
                let _ = tx.send((seq, ctrl.snapshot(), false));
            }
            ctx.request_repaint();
        });
    }

    fn spawn_quick_render(&mut self, ctx: Context) {
        self.snapshot.draft_video_path.clear();
        self.enhanced_video_path = None;
        self.enhancement_error = None;
        self.action_seq = self.action_seq.wrapping_add(1);
        let seq = self.action_seq;
        let ctrl_arc = Arc::clone(&self.controller);
        let tx = self.snapshot_tx.clone();
        set_gui_progress(true, 4, "Rendering preview...");
        let reframe_faces = self.reframe_faces_export;
        thread::spawn(move || {
            if let Ok(mut ctrl) = ctrl_arc.lock() {
                ctrl.draft_video_path = None;
                ctrl.reframe_faces = reframe_faces;
                if let Err(e) = ctrl.quick_render_draft(2) {
                    ctrl.status_message = format!("Quick render error: {e}");
                    set_gui_progress(false, 0, ctrl.status_message.clone());
                }
                let _ = tx.send((seq, ctrl.snapshot(), false));
            }
            ctx.request_repaint();
        });
    }

    fn start_export_video(&mut self, ctx: Context, path: PathBuf) {
        self.last_browse_dir = path.parent().map(|p| p.to_path_buf());
        self.action_seq = self.action_seq.wrapping_add(1);
        let seq = self.action_seq;
        let ctrl_arc = Arc::clone(&self.controller);
        let tx = self.snapshot_tx.clone();
        let use_gpu = self.use_gpu_export;
        let quality_enhance = self.quality_enhance_export;
        let light_cleanup = self.light_cleanup_export;
        let reframe_faces = self.reframe_faces_export;
        let crf = self.export_crf;
        let preset = self.export_preset.clone();
        set_gui_progress(true, 4, "Starting MP4 export...");
        ctx.request_repaint();
        thread::spawn(move || {
            if let Ok(mut ctrl) = ctrl_arc.lock() {
                ctrl.quality_enhance = quality_enhance;
                ctrl.light_cleanup = light_cleanup;
                ctrl.reframe_faces = reframe_faces;
                if let Err(e) = ctrl.export_video_with_quality(&path, use_gpu, crf, &preset) {
                    ctrl.status_message = format!("Export error: {e}");
                    set_gui_progress(false, 0, ctrl.status_message.clone());
                }
                let _ = tx.send((seq, ctrl.snapshot(), false));
            }
            ctx.request_repaint();
        });
    }

    fn open_file_dialog(&mut self, ctx: &Context, kind: FileDialogKind) {
        self.playing = false;
        self.stop_audio_companion();
        let mut dialog = rfd::FileDialog::new();
        if let Some(dir) = &self.last_browse_dir {
            dialog = dialog.set_directory(dir);
        }
        let path = match kind {
            FileDialogKind::ImportVideo => dialog
                .set_title("Import video")
                .add_filter("Video", &["mp4", "mkv", "mov", "webm", "avi", "m4v", "ts"])
                .pick_file(),
            FileDialogKind::ImportAudio => dialog
                .set_title("Import audio")
                .add_filter(
                    "Audio",
                    &["mp3", "wav", "flac", "ogg", "m4a", "aac", "opus"],
                )
                .pick_file(),
            FileDialogKind::ExportVideo => dialog
                .set_title("Export MP4")
                .add_filter("MP4 video", &["mp4"])
                .set_file_name("edited_out.mp4")
                .save_file(),
            FileDialogKind::ImportStyle => dialog
                .set_title("Import transition style preset")
                .add_filter("Transition style", &["toml"])
                .pick_file(),
        };
        let Some(path) = path else {
            return;
        };
        self.last_browse_dir = path.parent().map(Path::to_path_buf);
        match kind {
            FileDialogKind::ImportVideo => self.start_import_video(ctx.clone(), path),
            FileDialogKind::ImportAudio => self.start_import_audio(ctx.clone(), path),
            FileDialogKind::ExportVideo => self.start_export_video(ctx.clone(), path),
            FileDialogKind::ImportStyle => {
                self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                    if let Err(error) = ctrl.import_transition_style(&path) {
                        ctrl.status_message = format!("Could not import style: {error}");
                    }
                })
            }
        }
    }

    fn open_rendered_draft(&mut self) {
        let path = if self.compare_enhancement {
            self.enhanced_video_path
                .clone()
                .unwrap_or_else(|| PathBuf::from(&self.snapshot.draft_video_path))
        } else {
            PathBuf::from(&self.snapshot.draft_video_path)
        };
        #[cfg(target_os = "macos")]
        let result = Command::new("open").arg(&path).spawn();
        #[cfg(target_os = "windows")]
        let result = Command::new("explorer").arg(&path).spawn();
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        let result = Command::new("xdg-open").arg(&path).spawn();
        if let Err(error) = result {
            self.snapshot.status_message = format!("Could not open draft: {error}");
        }
    }

    // ---------- top bar ----------
    fn render_studio_topbar(&mut self, ctx: &Context, ui: &mut Ui, busy: bool) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = Vec2::new(8.0, 4.0);
            ui.label(RichText::new("auto-editor").size(15.0).strong());
            ui.label(RichText::new("Beat-synced editor").weak());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let (op_busy, pct, _pmsg) = get_gui_progress();
                if op_busy || busy {
                    ui.label(
                        RichText::new(format!("{pct}%"))
                            .size(12.0)
                            .color(Color32::from_rgb(255, 209, 102)),
                    );
                    ctx.request_repaint_after(Duration::from_millis(100));
                }
                if ui
                    .add_enabled(
                        !busy && !self.snapshot.slots.is_empty(),
                        egui::Button::new(RichText::new("Export MP4").size(13.0).strong()),
                    )
                    .on_hover_text(format!(
                        "{} · compression {} · {} encoding. Change these in Settings.",
                        self.snapshot.resolution, self.export_crf, self.export_preset
                    ))
                    .clicked()
                {
                    self.open_file_dialog(ctx, FileDialogKind::ExportVideo);
                }
            });
        });
    }

    // ---------- left rail: backend-enabled tabs ----------
    fn render_studio_rail(&mut self, ui: &mut Ui) {
        ui.vertical_centered(|ui| {
            for tab in [
                LeftTab::Media,
                LeftTab::Audio,
                LeftTab::Transitions,
                LeftTab::Settings,
            ] {
                let selected = self.left_tab == tab;
                let response = ui.add_sized([88.0, 70.0], egui::Button::new("").selected(selected));
                response.widget_info(|| {
                    egui::WidgetInfo::selected(egui::Role::Tab, true, selected, tab.label())
                });
                let center = Pos2::new(response.rect.center().x, response.rect.top() + 22.0);
                let color = ui.visuals().text_color();
                paint_tab_icon(ui.painter(), tab, center, color);
                ui.painter().text(
                    Pos2::new(center.x, response.rect.bottom() - 17.0),
                    egui::Align2::CENTER_CENTER,
                    tab.label(),
                    FontId::proportional(11.5),
                    color,
                );
                if response.clicked() {
                    self.left_tab = tab;
                }
                ui.add_space(4.0);
            }
        });
    }

    fn browser_items(&self) -> Vec<BrowserItem> {
        let mut items: Vec<BrowserItem> = self
            .snapshot
            .shots
            .iter()
            .map(|s| BrowserItem {
                shot_id: Some(s.id),
                name: self.media_display_name(s.id),
                dur_sec: s.duration_sec,
                is_audio: false,
            })
            .collect();
        if !self.snapshot.audio_path.is_empty() || self.snapshot.audio_duration_sec > 0.0 {
            items.push(BrowserItem {
                shot_id: None,
                name: self.audio_display_name(),
                dur_sec: self.snapshot.audio_duration_sec,
                is_audio: true,
            });
        }
        // tab filter: Media shows video scenes only, Audio shows the track only
        // (Transitions / Settings tabs render their own panels, not the grid)
        if self.left_tab == LeftTab::Audio {
            items.retain(|i| i.is_audio);
        } else {
            items.retain(|i| !i.is_audio);
        }
        // search
        let q = self.search_query.trim().to_lowercase();
        if !q.is_empty() {
            items.retain(|i| i.name.to_lowercase().contains(&q));
        }
        // sort by name
        items.sort_by(|a, b| {
            let ord = a.name.to_lowercase().cmp(&b.name.to_lowercase());
            if self.sort_asc { ord } else { ord.reverse() }
        });
        items
    }

    fn render_media_card(
        &mut self,
        ctx: &Context,
        ui: &mut Ui,
        item: &BrowserItem,
        card_w: f32,
        busy: bool,
    ) {
        let thumb_h = (card_w * 9.0 / 16.0).ceil();
        let sense = if busy {
            Sense::hover()
        } else if item.is_audio {
            Sense::click()
        } else {
            Sense::drag()
        };
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(card_w, thumb_h), sense);
        let resp = resp.on_hover_text(if item.is_audio {
            "Edit beat markers"
        } else {
            "Move left or right to preview. Drag onto a timeline clip to replace its scene."
        });
        if resp.drag_started()
            && let Some(sid) = item.shot_id
        {
            self.dragging_shot_id = Some(sid);
        }
        if item.is_audio {
            ui.painter()
                .rect_filled(rect, CornerRadius::same(8), Color32::from_rgb(10, 10, 12));
            ui.painter().rect_stroke(
                rect,
                CornerRadius::same(8),
                Stroke::new(1.0, Color32::from_rgb(55, 58, 72)),
                StrokeKind::Inside,
            );
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "♪",
                FontId::proportional(30.0),
                Color32::from_rgb(90, 92, 110),
            );
        } else if let Some(sid) = item.shot_id {
            let position = resp
                .hover_pos()
                .map(|pos| ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0));
            let tex = if let Some(position) = position {
                self.get_scrub_texture(ctx, sid, position)
            } else {
                self.get_shot_texture(ctx, sid, 1)
            };
            // rounded look: fill + image inset slightly
            ui.painter()
                .rect_filled(rect, CornerRadius::same(8), Color32::from_rgb(20, 20, 24));
            ui.painter().image(
                tex.id(),
                rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }
        // bottom-right duration badge
        let position = resp
            .hover_pos()
            .map(|pos| ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0));
        let dur = if !item.is_audio && position.is_some() {
            Self::fmt_dur(item.dur_sec * position.unwrap() as f64)
        } else {
            Self::fmt_dur(item.dur_sec)
        };
        let br = Rect::from_min_size(
            Pos2::new(rect.right() - 52.0, rect.bottom() - 24.0),
            Vec2::new(46.0, 18.0),
        );
        ui.painter().rect_filled(
            br,
            CornerRadius::same(5),
            Color32::from_rgba_unmultiplied(8, 8, 10, 210),
        );
        ui.painter().text(
            br.center(),
            egui::Align2::CENTER_CENTER,
            dur,
            FontId::monospace(10.5),
            Color32::WHITE,
        );
        if !item.is_audio {
            if let Some(position) = position {
                let x = rect.left() + rect.width() * position;
                ui.painter().line_segment(
                    [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
                    Stroke::new(1.5, Color32::WHITE),
                );
            }
        }
        // name + duration row
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = Vec2::new(4.0, 2.0);
            let name_w = card_w - 52.0;
            ui.add_sized(
                [name_w, 18.0],
                egui::Label::new(RichText::new(&item.name).size(12.0)).truncate(),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(
                    RichText::new(Self::fmt_dur(item.dur_sec))
                        .size(11.5)
                        .color(Color32::from_rgb(140, 142, 160)),
                );
            });
        });
        if let Some(shot) = item
            .shot_id
            .and_then(|id| self.snapshot.shots.iter().find(|shot| shot.id == id))
            .cloned()
        {
            ui.horizontal(|ui| {
                for id in &shot.character_ids {
                    let name = self
                        .snapshot
                        .characters
                        .iter()
                        .find(|character| character.id == *id)
                        .map(|character| character.label.clone())
                        .unwrap_or_default();
                    let texture = self.get_character_texture(ctx, *id);
                    ui.image((texture.id(), Vec2::splat(20.0)))
                        .on_hover_text(name);
                }
                if shot.characters_unresolved {
                    ui.small("Needs review");
                }
            });
            resp.context_menu(|ui| {
                if ui
                    .add_enabled(!busy, egui::Button::new("Review scene characters"))
                    .clicked()
                {
                    self.character_scene_review = Some((shot.id, shot.character_ids));
                    ui.close();
                }
            });
        }
    }

    fn render_character_review(&mut self, ctx: &Context, ui: &mut Ui, busy: bool) {
        if let Some((shot_id, mut ids)) = self.character_scene_review.take() {
            let shot = self
                .snapshot
                .shots
                .iter()
                .find(|shot| shot.id == shot_id)
                .cloned();
            let mut keep = true;
            ui.group(|ui| {
                ui.label(format!("Complete character list for Scene #{shot_id}"));
                for character in &self.snapshot.characters {
                    let mut present = ids.contains(&character.id);
                    if ui.checkbox(&mut present, &character.label).changed() {
                        ids.retain(|id| *id != character.id);
                        if present {
                            ids.push(character.id);
                        }
                    }
                }
                if ui
                    .add_enabled(
                        !busy && shot.is_some(),
                        egui::Button::new("Confirm complete list"),
                    )
                    .clicked()
                {
                    let shot = shot.unwrap();
                    let present = ids.clone();
                    self.spawn_controller_action(ctx.clone(), false, move |controller| {
                        if let Err(error) = controller.review_scene_characters(shot.id, present) {
                            controller.status_message = error.to_string();
                        }
                    });
                    keep = false;
                }
                if ui.button("Cancel review").clicked() {
                    keep = false;
                }
            });
            if keep {
                self.character_scene_review = Some((shot_id, ids));
            }
        }
        if !self.character_review_open {
            return;
        }
        let (appearances, next_id) = self
            .controller
            .try_lock()
            .ok()
            .and_then(|controller| {
                controller.character_analysis.as_ref().map(|analysis| {
                    let next_id = analysis
                        .records
                        .iter()
                        .map(|record| record.id)
                        .chain(
                            analysis
                                .observations
                                .iter()
                                .map(|observation| observation.id),
                        )
                        .max()
                        .unwrap_or(0)
                        .checked_add(1);
                    let appearances = analysis
                        .observations
                        .iter()
                        .map(|observation| {
                            (observation.id, observation.pts, observation.character_id)
                        })
                        .collect::<Vec<_>>();
                    (appearances, next_id)
                })
            })
            .unwrap_or_default();
        let characters = self.snapshot.characters.clone();
        if !self.snapshot.character_analysis_error.is_empty() {
            ui.colored_label(Color32::LIGHT_RED, &self.snapshot.character_analysis_error);
        }
        ui.small("Review individual source appearances. Separate a mistaken identity with New character, or move it to an existing group.");
        ScrollArea::vertical()
            .id_salt("character_appearances")
            .max_height(180.0)
            .show_rows(ui, 26.0, appearances.len(), |ui, range| {
                for &(observation_id, pts, character_id) in &appearances[range] {
                    ui.horizontal(|ui| {
                        ui.label(Self::fmt_tc(pts.as_secs_f64()));
                        let name = characters
                            .iter()
                            .find(|character| Some(character.id) == character_id)
                            .map(|character| character.label.as_str())
                            .unwrap_or("Unresolved");
                        let mut operation = None;
                        ui.add_enabled_ui(!busy, |ui| {
                            ui.menu_button(name, |ui| {
                                for character in &characters {
                                    if ui.button(&character.label).clicked() {
                                        operation = Some(CharacterEdit::Assign {
                                            observation_id,
                                            character_id: Some(character.id),
                                        });
                                        ui.close();
                                    }
                                }
                                if ui.button("Unresolved").clicked() {
                                    operation = Some(CharacterEdit::Assign {
                                        observation_id,
                                        character_id: None,
                                    });
                                    ui.close();
                                }
                            });
                            if ui
                                .add_enabled(next_id.is_some(), egui::Button::new("New character"))
                                .clicked()
                            {
                                operation = Some(CharacterEdit::Assign {
                                    observation_id,
                                    character_id: next_id,
                                });
                            }
                            if ui.button("Ignore detection").clicked() {
                                operation = Some(CharacterEdit::Ignore { observation_id });
                            }
                            if let Some(id) = character_id {
                                if ui.button("Use as portrait").clicked() {
                                    operation =
                                        Some(CharacterEdit::Portrait { id, observation_id });
                                }
                            }
                        });
                        if let Some(operation) = operation {
                            self.spawn_controller_action(ctx.clone(), false, move |controller| {
                                if let Err(error) = controller.apply_character_edit(operation) {
                                    controller.status_message = error.to_string();
                                }
                            });
                        }
                    });
                }
            });
    }

    // ---------- media browser column ----------
    fn render_studio_browser(&mut self, ctx: &Context, ui: &mut Ui, busy: bool) {
        if self.left_tab == LeftTab::Transitions {
            ui.heading("Transitions");
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!busy, egui::Button::new("Import style preset…"))
                    .on_hover_text("Load a TOML preset that chooses among the built-in effects")
                    .clicked()
                {
                    self.open_file_dialog(ctx, FileDialogKind::ImportStyle);
                }
                if ui
                    .add_enabled(
                        !busy
                            && !self.snapshot.slots.is_empty()
                            && self.selected_transition.is_some(),
                        egui::Button::new("Apply to all"),
                    )
                    .clicked()
                {
                    if let Some(id) = self.selected_transition {
                        self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                            if let Err(error) = ctrl.set_all_slot_transitions(id, 120) {
                                ctrl.status_message = error.to_string();
                            }
                        });
                    }
                }
            });
            ui.label(if self.snapshot.slots.is_empty() {
                "Import video and audio to create a timeline first."
            } else {
                "Select an effect, then apply it to the selected clip or every cut."
            });
            ui.add_space(8.0);
            ScrollArea::vertical()
                .id_salt("studio_trans_list")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for (id, description) in ae_transitions::list_registered_transitions() {
                        let selected = self.selected_transition == Some(id);
                        let title = id.replace('_', " ");
                        let title = format!("{}{}", title[..1].to_uppercase(), &title[1..]);
                        let width = ui.available_width();
                        ui.add_enabled_ui(!busy && !self.snapshot.slots.is_empty(), |ui| {
                            ui.horizontal(|ui| {
                                let effect = egui::Button::new("")
                                    .left_text(title)
                                    .selected(selected)
                                    .truncate();
                                if ui
                                    .add_sized([width, 36.0], effect)
                                    .on_hover_text(description)
                                    .clicked()
                                {
                                    self.selected_transition = Some(id);
                                    let index = self.selected_slot_idx;
                                    let id = id.to_owned();
                                    self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                                        if let Err(error) =
                                            ctrl.set_slot_transition(index, &id, 120)
                                        {
                                            ctrl.status_message = error.to_string();
                                        }
                                    });
                                }
                            });
                        });
                    }
                });
            return;
        }

        if self.left_tab == LeftTab::Audio {
            self.render_audio_panel(ctx, ui, busy);
            return;
        }

        // Settings has its own controls.
        if self.left_tab == LeftTab::Settings {
            ui.add_enabled_ui(!busy, |ui| self.render_studio_settings(ctx, ui, busy));
            return;
        }

        self.character_textures.retain(|(revision, id), _| {
            *revision == self.snapshot.character_revision
                && self
                    .snapshot
                    .characters
                    .iter()
                    .any(|character| character.id == *id)
        });
        ui.horizontal(|ui| {
            ui.heading("Characters");
            if ui
                .add_enabled(!busy, egui::Button::new("Review appearances"))
                .clicked()
            {
                self.character_review_open = !self.character_review_open;
            }
            if self.character_analysis_running && ui.button("Cancel analysis").clicked() {
                self.character_cancel.store(true, Ordering::Relaxed);
            }
            if !busy
                && !self.snapshot.character_analysis_error.is_empty()
                && ui.button("Retry analysis").clicked()
            {
                self.spawn_controller_action(ctx.clone(), false, |controller| {
                    controller.character_analysis = None;
                    controller.character_analysis_error.clear();
                });
            }
            if !self.snapshot.character_filters.is_empty()
                && ui
                    .add_enabled(
                        !busy && !self.pending_resequence,
                        egui::Button::new("Clear filters"),
                    )
                    .clicked()
            {
                self.spawn_controller_action(ctx.clone(), false, |ctrl| {
                    if let Err(error) = ctrl.clear_character_filters() {
                        ctrl.status_message = error.to_string();
                    }
                });
            }
        });
        let characters = self.snapshot.characters.clone();
        let no_portraits = global_media_cache()
            .try_read()
            .map(|cache| {
                cache.character_revision != self.snapshot.character_revision
                    || cache.character_portraits.is_empty()
            })
            .unwrap_or(true);
        if !self.snapshot.character_analysis_error.is_empty() {
            ui.colored_label(Color32::LIGHT_RED, &self.snapshot.character_analysis_error);
        }
        if no_portraits {
            if self.snapshot.character_analysis_error.is_empty() {
                ui.small("Character analysis pending or no characters found.");
            } else {
                ui.small(&self.snapshot.character_analysis_error);
            }
        }
        ScrollArea::horizontal()
            .id_salt("character_row")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for character in &characters {
                        ui.vertical(|ui| {
                            let texture = self.get_character_texture(ctx, character.id);
                            let portrait = ui
                                .add_enabled(
                                    !busy && !self.pending_resequence,
                                    egui::Button::image(egui::Image::new((
                                        texture.id(),
                                        Vec2::new(64.0, 64.0),
                                    ))),
                                )
                                .on_hover_text(format!(
                                    "{}: click for With, Only, Without, then clear",
                                    character.label
                                ));
                            let mode = match character.mode {
                                crate::character_model::CharacterFilterMode::Neutral => "All",
                                crate::character_model::CharacterFilterMode::With => "With",
                                crate::character_model::CharacterFilterMode::Only => "Only",
                                crate::character_model::CharacterFilterMode::Without => "Without",
                            };
                            let label = format!(
                                "{} · {} ({}){}",
                                character.label,
                                mode,
                                character.scene_count,
                                if character.provisional {
                                    " · Review"
                                } else {
                                    ""
                                }
                            );
                            let response = ui.add_enabled(
                                !busy && !self.pending_resequence,
                                egui::Button::new(label),
                            );
                            if response.clicked() || portrait.clicked() {
                                let id = character.id;
                                self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                                    if let Err(error) = ctrl.cycle_character_filter(id) {
                                        ctrl.status_message = error.to_string();
                                    }
                                });
                            }
                            response.context_menu(|ui| {
                                if ui.button("Rename").clicked() {
                                    self.character_rename =
                                        Some((character.id, character.label.clone()));
                                    ui.close();
                                }
                                ui.menu_button("Merge into", |ui| {
                                    for other in
                                        characters.iter().filter(|other| other.id != character.id)
                                    {
                                        if ui.button(&other.label).clicked() {
                                            let from = character.id;
                                            let into = other.id;
                                            self.spawn_controller_action(
                                                ctx.clone(),
                                                false,
                                                move |ctrl| {
                                                    if let Err(error) =
                                                        ctrl.merge_characters(from, into)
                                                    {
                                                        ctrl.status_message = error.to_string();
                                                    }
                                                },
                                            );
                                            ui.close();
                                        }
                                    }
                                });
                            });
                        });
                    }
                });
            });
        if let Some((id, mut label)) = self.character_rename.take() {
            let mut keep = true;
            ui.horizontal(|ui| {
                ui.label("Rename character:");
                ui.text_edit_singleline(&mut label);
                if ui
                    .add_enabled(!busy && !self.pending_resequence, egui::Button::new("Save"))
                    .clicked()
                {
                    let name = label.clone();
                    self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                        if let Err(error) = ctrl.rename_character(id, name) {
                            ctrl.status_message = error.to_string();
                        }
                    });
                    keep = false;
                }
                if ui.button("Cancel").clicked() {
                    keep = false;
                }
            });
            if keep {
                self.character_rename = Some((id, label));
            }
        }

        self.render_character_review(ctx, ui, busy);
        ui.heading(self.left_tab.label());
        ui.add_space(6.0);
        // search row
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = Vec2::new(8.0, 6.0);
            let search_w = ui.available_width() - 44.0;
            ui.add_sized(
                [search_w.max(80.0), 30.0],
                egui::TextEdit::singleline(&mut self.search_query).hint_text("Search by name"),
            );
            if ui
                .add_enabled(
                    !busy,
                    egui::Button::new(RichText::new("+").size(16.0)).min_size(Vec2::splat(30.0)),
                )
                .on_hover_text(if self.left_tab == LeftTab::Audio {
                    "Import audio"
                } else {
                    "Import video"
                })
                .clicked()
            {
                let kind = if self.left_tab == LeftTab::Audio {
                    FileDialogKind::ImportAudio
                } else {
                    FileDialogKind::ImportVideo
                };
                self.open_file_dialog(ctx, kind);
            }
        });
        ui.add_space(6.0);
        // filter row
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = Vec2::new(8.0, 4.0);
            let sort_label = if self.sort_asc {
                "Name A-Z"
            } else {
                "Name Z-A"
            };
            if ui
                .button(RichText::new(format!("{sort_label} ▾")).size(12.0))
                .clicked()
            {
                self.sort_asc = !self.sort_asc;
            }
        });
        ui.add_space(6.0);

        let items = self.browser_items();
        if items.is_empty() {
            ui.add_space(30.0);
            ui.vertical_centered(|ui| {
                ui.label(
                    RichText::new(if !self.search_query.trim().is_empty() {
                        "No matches. Clear the search to see your media."
                    } else if !self.snapshot.character_filters.is_empty() && self.left_tab != LeftTab::Audio {
                        "No scenes match these character filters. Clear filters or review appearances."
                    } else if self.character_analysis_running && self.left_tab != LeftTab::Audio {
                        "Analyzing animated characters…"
                    } else if !self.snapshot.video_path.is_empty() && self.left_tab != LeftTab::Audio {
                        "No character scenes found. Review appearances or retry character analysis."
                    } else if self.left_tab == LeftTab::Audio {
                        "Import an audio track to find its beats."
                    } else {
                        "Import a video to split it into scenes."
                    })
                    .size(12.5)
                    .color(Color32::from_rgb(140, 145, 165)),
                );
                ui.add_space(8.0);
                let is_audio = self.left_tab == LeftTab::Audio;
                let import_label = if is_audio {
                    "Import audio"
                } else {
                    "Import video"
                };
                if ui
                    .add_enabled(!busy, egui::Button::new(import_label))
                    .clicked()
                {
                    self.open_file_dialog(
                        ctx,
                        if is_audio {
                            FileDialogKind::ImportAudio
                        } else {
                            FileDialogKind::ImportVideo
                        },
                    );
                }
            });
            return;
        }
        let avail_w = ui.available_width().max(120.0);
        let card_w = ((avail_w - 10.0) / 2.0).clamp(120.0, 400.0);
        ScrollArea::vertical()
            .id_salt("studio_media_grid")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing = Vec2::new(10.0, 10.0);
                for row in items.chunks(2) {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing = Vec2::new(10.0, 10.0);
                        for item in row {
                            ui.vertical(|ui| {
                                ui.set_width(card_w);
                                self.render_media_card(ctx, ui, item, card_w, busy);
                            });
                        }
                    });
                }
            });
    }

    // ---------- settings panel ----------
    fn render_studio_settings(&mut self, ctx: &Context, ui: &mut Ui, _busy: bool) {
        ScrollArea::vertical()
            .id_salt("studio_settings")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing = Vec2::new(4.0, 8.0);
                ui.heading("Settings");
                ui.label(RichText::new("Output").strong().size(13.0));
                egui::ComboBox::from_id_salt("set_res_preset")
                    .selected_text(&self.snapshot.resolution)
                    .width(180.0)
                    .show_ui(ui, |ui| {
                        for (label, val) in [
                            ("1080x1920 (Vertical)", "1080x1920"),
                            ("1920x1080 (1080p)", "1920x1080"),
                            ("1280x720 (720p)", "1280x720"),
                            ("3840x2160 (4K)", "3840x2160"),
                            ("2160x3840 (Vertical 4K)", "2160x3840"),
                            ("720x1280 (Vertical 720p)", "720x1280"),
                            ("1080x1080 (Square)", "1080x1080"),
                            ("1440x1080 (4:3)", "1440x1080"),
                        ] {
                            if ui.selectable_label(self.snapshot.resolution == val, label).clicked() {
                                self.res_input = val.to_string();
                                let owned = val.to_string();
                                self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                                    if let Err(error) = ctrl.set_resolution_str(&owned) {
                                        ctrl.status_message = format!("Invalid output size: {error}");
                                    }
                                });
                            }
                        }
                    });
                ui.horizontal(|ui| {
                    ui.label("Custom (WxH):");
                    let res_resp = ui.add(
                        egui::TextEdit::singleline(&mut self.res_input)
                            .desired_width(100.0)
                            .hint_text("WxH"),
                    );
                    if res_resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                        let owned = self.res_input.clone();
                        self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                            if let Err(error) = ctrl.set_resolution_str(&owned) {
                                ctrl.status_message = format!("Invalid output size: {error}");
                            }
                        });
                    }
                });
                let mut allow_reuse = self.snapshot.allow_reuse;
                if ui
                    .checkbox(&mut allow_reuse, "Reuse clips")
                    .on_hover_text(
                        "After every scene has been used once, the pool reopens for another round.\n\
                         Neighbouring scenes are never the same.",
                    )
                    .changed()
                {
                    self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                        let _ = ctrl.set_allow_reuse(allow_reuse);
                    });
                }
                ui.label("Export enhancement");
                let mut enhancement = (self.light_cleanup_export, self.quality_enhance_export);
                egui::ComboBox::from_id_salt("export_enhancement")
                    .selected_text(if enhancement.1 { "AI restoration" } else if enhancement.0 { "Light cleanup (fast)" } else { "Off" })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut enhancement, (false, false), "Off");
                        ui.selectable_value(&mut enhancement, (true, false), "Light cleanup (fast)");
                        ui.selectable_value(&mut enhancement, (false, true), "AI restoration");
                    });
                (self.light_cleanup_export, self.quality_enhance_export) = enhancement;
                if self.quality_enhance_export {
                    ui.small("AI restores noise, compression artifacts and mild blur. First use downloads 86 MB; processing stays local.");
                } else if self.light_cleanup_export {
                    ui.small("Stronger denoising and sharpening with richer contrast and color, matching preview cleanup. No model download.");
                }
                if ui.checkbox(&mut self.reframe_faces_export, "Keep animated character fixed")
                    .on_hover_text("Follow the whole animated character with a small, fixed crop in previews and exports. Keeps a wide view and stops panning at image edges without zooming tighter. Character filtering stays active whether this is checked or not.")
                    .changed()
                {
                    self.set_reframe_faces(ctx.clone(), self.reframe_faces_export);
                }
                ui.small("Character filters choose scenes. This checkbox only enables tracking and cropping.");
                ui.separator();
                ui.label(RichText::new("Preview").strong());
                if ui.add_enabled(!self.snapshot.slots.is_empty(), egui::Button::new(
                    if self.compare_enhancement { "Hide before/after" } else { "Apply light preview cleanup" }
                )).on_hover_text("Apply stronger denoising, sharpening, contrast and color, then cache the result for playback and comparison. No model download. Matches Light cleanup export; AI restoration is a separate option.").clicked() {
                    self.compare_enhancement = !self.compare_enhancement;
                    self.draft_preview = None;
                    self.enhancement_error = None;
                }
                if self.compare_enhancement {
                    ui.add(egui::Slider::new(&mut self.enhancement_split, 0.0..=1.0)
                        .text("Before / after").show_value(false));
                    ui.small("Drag the line to compare light cleanup; Export offers light cleanup or AI restoration.");
                    if self.enhancement_pending {
                        ui.horizontal(|ui| { ui.spinner(); ui.label("Filtering preview…"); });
                    }
                }
                if self.compare_enhancement && self.enhanced_video_path.is_some() {
                    ui.small("Filtered preview ready for playback.");
                }
                if let Some(error) = &self.enhancement_error {
                    ui.colored_label(Color32::from_rgb(255, 150, 130), error);
                }
                ui.label("The preview follows the output aspect ratio.");
                if ui.add_enabled(!self.snapshot.draft_video_path.is_empty(), egui::Button::new("Play rendered preview")).clicked() {
                    self.open_rendered_draft();
                }
                ui.separator();
                ui.label(RichText::new("MP4 export quality").strong());
                egui::ComboBox::from_id_salt("export_quality")
                    .selected_text(format!("Compression {}", self.export_crf))
                    .show_ui(ui, |ui| {
                        for (crf, label) in [(28, "Smaller file"), (23, "Standard"), (18, "High quality"), (14, "Very high quality"), (0, "Lossless (large file)")] {
                            ui.selectable_value(&mut self.export_crf, crf, label);
                        }
                    });
                ui.add(egui::Slider::new(&mut self.export_crf, 0..=51).text("Compression"));
                ui.label(RichText::new("Lower compression values preserve more detail and make larger files.").small().weak());
                egui::ComboBox::from_id_salt("export_preset")
                    .selected_text(format!("Encoding: {}", self.export_preset))
                    .show_ui(ui, |ui| {
                        for preset in ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"] {
                            ui.selectable_value(&mut self.export_preset, preset.to_owned(), preset);
                        }
                    });
                ui.label(RichText::new("Slower encoding reduces file size at the same quality.").small().weak());
                ui.separator();
                ui.label(RichText::new("Interface").strong().size(13.0));
                egui::ComboBox::from_id_salt("set_ui_scale")
                    .selected_text(format!("{}%", (self.ui_zoom * 100.0).round() as u32))
                    .width(110.0)
                    .show_ui(ui, |ui| {
                        for (label, z) in [
                            ("100%", 1.0f32),
                            ("125%", 1.25),
                            ("150% (5K)", 1.50),
                            ("175% (5K)", 1.75),
                            ("200%", 2.00),
                        ] {
                            if ui
                                .selectable_label((self.ui_zoom - z).abs() < 0.05, label)
                                .clicked()
                            {
                                self.ui_zoom = z;
                            }
                        }
                    });
            });
    }

    fn render_audio_panel(&mut self, ctx: &Context, ui: &mut Ui, busy: bool) {
        ScrollArea::vertical().id_salt("audio_controls").auto_shrink([false, false]).show(ui, |ui| {
            ui.heading("Audio");
            if ui.add_enabled(!busy, egui::Button::new(if self.snapshot.audio_path.is_empty() { "Import audio…" } else { "Replace audio…" })).clicked() {
                self.open_file_dialog(ctx, FileDialogKind::ImportAudio);
            }
            if self.snapshot.audio_path.is_empty() {
                ui.label("Import a track to detect beats and place cuts on the music.");
                return;
            }
            ui.add(egui::Label::new(self.audio_display_name()).truncate()).on_hover_text(&self.snapshot.audio_path);
            let active = self.snapshot.beats.iter().filter(|b| !b.is_ghost && b.enabled).count();
            ui.label(format!("{} of {} beats enabled", active, self.snapshot.beats.iter().filter(|b| !b.is_ghost).count()));
            ui.label("Edit markers on the music track below. Click to select, drag to move, or double-click to add.");
            ui.add_enabled_ui(!busy, |ui| {
                if ui.button("Add beat at playhead").clicked() {
                    let time = self.playhead_sec.min(self.snapshot.audio_duration_sec);
                    self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                        if let Err(error) = ctrl.add_beat_at(time) { ctrl.status_message = error.to_string(); }
                    });
                }
                ui.separator();
                ui.label(RichText::new("Selected beat").strong());
                egui::ComboBox::from_id_salt("selected_beat")
                    .selected_text(self.selected_beat_idx.and_then(|index| self.snapshot.beats.iter().find(|b| !b.is_ghost && b.index == index)).map(|beat| format!("Beat {} at {:.3}s", beat.index + 1, beat.time_sec)).unwrap_or_else(|| "Select a marker".to_owned()))
                    .show_ui(ui, |ui| {
                        for beat in self.snapshot.beats.iter().filter(|b| !b.is_ghost) {
                            ui.selectable_value(&mut self.selected_beat_idx, Some(beat.index), format!("Beat {} at {:.3}s", beat.index + 1, beat.time_sec));
                        }
                    });
                if let Some(beat) = self.selected_beat_idx.and_then(|index| self.snapshot.beats.iter().find(|b| !b.is_ghost && b.index == index)).cloned() {
                    let index = beat.index;
                    ui.horizontal(|ui| {
                        ui.label("Time");
                        let mut time = beat.time_sec;
                        let response = ui.add(egui::DragValue::new(&mut time).speed(0.01).range(0.01..=(self.snapshot.audio_duration_sec - 0.01).max(0.01)).suffix(" s"));
                        if response.changed() { self.snapshot.beats.iter_mut().find(|b| !b.is_ghost && b.index == index).unwrap().time_sec = time; }
                        if response.drag_stopped() || (response.changed() && !response.dragged()) {
                            self.selected_beat_idx = None;
                            self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                                if let Err(error) = ctrl.move_beat(index, time) { ctrl.status_message = error.to_string(); }
                            });
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.button(if beat.enabled { "Disable beat" } else { "Enable beat" }).clicked() {
                            self.spawn_controller_action(ctx.clone(), false, move |ctrl| { if let Err(error) = ctrl.toggle_beat(index) { ctrl.status_message = error.to_string(); } });
                        }
                        if ui.button("Delete beat").clicked() {
                            self.selected_beat_idx = None;
                            self.spawn_controller_action(ctx.clone(), false, move |ctrl| { if let Err(error) = ctrl.remove_beat(index) { ctrl.status_message = error.to_string(); } });
                        }
                    });
                }
                ui.separator();
                ui.label(RichText::new("Beat detection").strong());
                ui.label("Higher thresholds find fewer beats. Your manual edits are preserved.");
                let mut sensitivity = self.snapshot.sensitivity;
                let response = ui.add(egui::Slider::new(&mut sensitivity, 0.25..=2.20).text("Threshold"));
                if response.changed() { self.snapshot.sensitivity = sensitivity; }
                if response.drag_stopped() || (response.changed() && !response.dragged()) {
                    self.spawn_controller_action(ctx.clone(), false, move |ctrl| { if let Err(error) = ctrl.set_sensitivity(sensitivity) { ctrl.status_message = error.to_string(); } });
                }
                ui.collapsing("Detection settings", |ui| {
                    let mut low = self.snapshot.bass_low_hz;
                    let mut high = self.snapshot.bass_high_hz;
                    let lo = ui.add(egui::Slider::new(&mut low, 20.0..=150.0).text("Low frequency (Hz)"));
                    let hi = ui.add(egui::Slider::new(&mut high, 80.0..=500.0).text("High frequency (Hz)"));
                    if lo.changed() || hi.changed() { self.snapshot.bass_low_hz = low; self.snapshot.bass_high_hz = high; }
                    if lo.drag_stopped() || hi.drag_stopped() || (lo.changed() && !lo.dragged()) || (hi.changed() && !hi.dragged()) {
                        self.spawn_controller_action(ctx.clone(), false, move |ctrl| { if let Err(error) = ctrl.set_bass_band(low, high) { ctrl.status_message = error.to_string(); } });
                    }
                    egui::ComboBox::from_id_salt("audio_isolation").selected_text(if self.snapshot.separation_mode == "hpss" { "Harmonic isolation" } else { "Frequency band" }).show_ui(ui, |ui| {
                        for (label, value, mode) in [("Harmonic isolation", "hpss", SeparationMode::Hpss), ("Frequency band", "band", SeparationMode::Band)] {
                            if ui.selectable_label(self.snapshot.separation_mode == value, label).clicked() {
                                self.spawn_controller_action(ctx.clone(), false, move |ctrl| { if let Err(error) = ctrl.beat_editor.set_separation_mode(mode).and_then(|_| ctrl.resequence()) { ctrl.status_message = error.to_string(); } });
                            }
                        }
                    });
                    egui::ComboBox::from_id_salt("audio_beat_mode").selected_text(&self.snapshot.beat_mode).show_ui(ui, |ui| {
                        for (label, value, mode) in [("Hybrid", "hybrid", BeatMode::Hybrid), ("Regular grid", "grid", BeatMode::Grid), ("Onsets", "onsets", BeatMode::Onsets)] {
                            if ui.selectable_label(self.snapshot.beat_mode == value, label).clicked() {
                                self.spawn_controller_action(ctx.clone(), false, move |ctrl| { if let Err(error) = ctrl.beat_editor.set_beat_mode(mode).and_then(|_| ctrl.resequence()) { ctrl.status_message = error.to_string(); } });
                            }
                        }
                    });
                });
            });
        });
    }

    fn render_audio_lane(&mut self, ctx: &Context, ui: &mut Ui, rect: Rect, busy: bool) {
        let duration = self.snapshot.audio_duration_sec;
        if duration <= 0.0 {
            return;
        }
        let texture = self.get_spectrogram_texture(ctx);
        let painter = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));
        painter.image(
            texture.id(),
            rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
        let response = ui.interact(rect, ui.make_persistent_id("audio_canvas"), Sense::click());
        let position = ui.input(|input| input.pointer.interact_pos());
        let time_at = |position: Pos2| {
            (((position.x - rect.left()) / rect.width()).clamp(0.0, 1.0) as f64 * duration)
                .clamp(0.01, (duration - 0.01).max(0.01))
        };
        let mut interacted = false;
        let mut move_beat = None;
        for beat in self
            .snapshot
            .beats
            .clone()
            .into_iter()
            .filter(|beat| !beat.is_ghost)
        {
            let x = rect.left() + (beat.time_sec / duration) as f32 * rect.width();
            let selected = self.selected_beat_idx == Some(beat.index);
            let color = if selected {
                Color32::WHITE
            } else if beat.enabled {
                Color32::from_rgb(103, 197, 172)
            } else {
                Color32::GRAY
            };
            let handle =
                Rect::from_center_size(Pos2::new(x, rect.top() + 10.0), Vec2::new(14.0, 18.0));
            painter.line_segment(
                [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
                Stroke::new(if selected { 2.0 } else { 1.0 }, color),
            );
            painter.rect_filled(handle, 3.0, color);
            if !beat.enabled {
                painter.text(
                    handle.center(),
                    egui::Align2::CENTER_CENTER,
                    "×",
                    FontId::proportional(12.0),
                    Color32::BLACK,
                );
            }
            let marker = ui
                .interact(
                    Rect::from_center_size(
                        Pos2::new(x, rect.center().y),
                        Vec2::new(12.0, rect.height()),
                    )
                    .intersect(rect),
                    ui.make_persistent_id(("beat_marker", beat.index)),
                    if busy {
                        Sense::hover()
                    } else {
                        Sense::click_and_drag()
                    },
                )
                .on_hover_text(format!(
                    "Beat {} at {:.3}s{}",
                    beat.index + 1,
                    beat.time_sec,
                    if beat.enabled { "" } else { " (disabled)" }
                ));
            if marker.clicked() {
                interacted = true;
                self.selected_beat_idx = Some(beat.index);
                self.left_tab = LeftTab::Audio;
            }
            if marker.drag_started() || marker.dragged() {
                interacted = true;
                self.dragging_beat_idx = Some(beat.index);
                self.selected_beat_idx = Some(beat.index);
                if let Some(position) = position {
                    if let Some(local) = self
                        .snapshot
                        .beats
                        .iter_mut()
                        .find(|b| !b.is_ghost && b.index == beat.index)
                    {
                        local.time_sec = time_at(position);
                    }
                }
            }
            if marker.drag_stopped() {
                interacted = true;
                move_beat = Some((beat.index, position.map(time_at).unwrap_or(beat.time_sec)));
            }
        }
        if let Some((index, time)) = move_beat {
            self.dragging_beat_idx = None;
            self.selected_beat_idx = None;
            self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                if let Err(error) = ctrl.move_beat(index, time) {
                    ctrl.status_message = error.to_string();
                }
            });
        } else if !interacted && response.double_clicked() && !busy {
            if let Some(position) = position {
                let time = time_at(position);
                self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                    if let Err(error) = ctrl.add_beat_at(time) {
                        ctrl.status_message = error.to_string();
                    }
                });
            }
        } else if !interacted && response.clicked() {
            if let Some(position) = position {
                self.seek_to(time_at(position));
            }
            self.left_tab = LeftTab::Audio;
        }
        let x = rect.left() + (self.playhead_sec / duration) as f32 * rect.width();
        if x >= rect.left() && x <= rect.right() {
            painter.line_segment(
                [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
                Stroke::new(1.5, Color32::from_rgb(255, 209, 102)),
            );
        }
    }

    // ---------- center preview ----------
    fn render_studio_preview(&mut self, ctx: &Context, ui: &mut Ui) {
        let avail = ui.available_size();
        let (outer, _) = ui.allocate_exact_size(avail, Sense::hover());
        ui.painter()
            .rect_filled(outer, CornerRadius::ZERO, Color32::from_rgb(8, 8, 10));
        let aw = self.snapshot.target_width.max(1) as f32;
        let ah = self.snapshot.target_height.max(1) as f32;
        let mut bw = outer.width().min(outer.height() * aw / ah);
        let mut bh = bw * ah / aw;
        if bh > outer.height() {
            bh = outer.height();
            bw = bh * aw / ah;
        }
        let rect = Rect::from_center_size(outer.center(), Vec2::new(bw.max(40.0), bh.max(24.0)));
        if self.snapshot.slots.is_empty() {
            ui.painter()
                .rect_filled(rect, CornerRadius::same(4), Color32::from_rgb(16, 16, 20));
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Import a clip + audio to preview the beat-synced edit",
                FontId::proportional(13.0),
                Color32::from_rgb(130, 134, 155),
            );
        } else {
            let playhead_ms = (self.playhead_sec.max(0.0) * 1000.0).round() as u64;
            let (tex, enhanced) = self.get_preview_texture(ctx, playhead_ms);
            ui.painter()
                .rect_filled(rect, CornerRadius::same(4), Color32::from_rgb(16, 16, 20));
            // slight inset so corners look rounded
            let img_rect = rect.shrink(1.0);
            ui.painter().image(
                tex.id(),
                img_rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
            if self.compare_enhancement
                && let Some(after) = enhanced
            {
                self.paint_enhancement_comparison(ui, img_rect, &after);
            } else if self.compare_enhancement {
                ui.painter().text(
                    img_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "Preparing filtered preview…",
                    FontId::proportional(14.0),
                    Color32::WHITE,
                );
            }
            ui.painter().rect_stroke(
                rect,
                CornerRadius::same(4),
                Stroke::new(1.0, Color32::from_rgb(60, 62, 78)),
                StrokeKind::Outside,
            );
        }
        // status overlay (bottom-left): keeps background-task feedback visible
        let (op_busy, pct, pmsg) = get_gui_progress();
        let msg = if op_busy && !pmsg.is_empty() {
            format!("{pct}% — {pmsg}")
        } else {
            self.snapshot.status_message.clone()
        };
        if !msg.is_empty() {
            ui.painter().text(
                Pos2::new(outer.left() + 10.0, outer.bottom() - 10.0),
                egui::Align2::LEFT_BOTTOM,
                trunc_chars(&msg, 90),
                FontId::proportional(11.5),
                if op_busy {
                    Color32::from_rgb(255, 209, 102)
                } else {
                    Color32::from_rgb(130, 133, 152)
                },
            );
            if op_busy {
                ctx.request_repaint_after(Duration::from_millis(100));
            }
        }
    }

    // ---------- timeline ----------
    fn track_header(&self, ui: &mut Ui, name: &str, count: usize, h: f32) {
        let (rect, _) = ui.allocate_exact_size(Vec2::new(196.0, h), Sense::hover());
        let p = ui.painter();
        p.text(
            Pos2::new(rect.left() + 26.0, rect.top() + 4.0),
            egui::Align2::LEFT_TOP,
            name,
            FontId::proportional(13.0),
            Color32::from_rgb(225, 227, 236),
        );
        p.text(
            Pos2::new(rect.right() - 8.0, rect.top() + 6.0),
            egui::Align2::RIGHT_TOP,
            format!("{count}"),
            FontId::proportional(12.0),
            Color32::from_rgb(130, 133, 152),
        );
    }

    fn render_studio_timeline(&mut self, ctx: &Context, ui: &mut Ui, busy: bool) {
        // transport row
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = Vec2::new(8.0, 4.0);
            let play = if self.playing { "❚❚" } else { "▶" };
            if ui
                .add_enabled(
                    !self.snapshot.slots.is_empty() || self.snapshot.audio_duration_sec > 0.0,
                    egui::Button::new(play).min_size(Vec2::new(34.0, 28.0)),
                )
                .on_hover_text("Play / pause (Space)")
                .clicked()
            {
                self.toggle_play_pause();
            }
            if ui
                .add_enabled(
                    !busy
                        && !self.snapshot.shots.is_empty()
                        && self.snapshot.audio_duration_sec > 0.0,
                    egui::Button::new("Sequence to beats"),
                )
                .clicked()
            {
                self.spawn_controller_action(ctx.clone(), false, |ctrl| {
                    let _ = ctrl.resequence();
                });
            }
            ui.label(
                RichText::new(Self::fmt_tc(self.playhead_sec))
                    .size(12.5)
                    .monospace(),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(RichText::new(format!("{:.0}%", self.timeline_zoom * 100.0)).size(12.0));
                ui.add_sized(
                    [120.0, 18.0],
                    egui::Slider::new(&mut self.timeline_zoom, 0.4..=3.0).show_value(false),
                );
                ui.label("Timeline zoom");
                ui.label(RichText::new(&self.snapshot.resolution).weak());
            });
        });
        ui.add_space(2.0);

        let total_dur = self
            .snapshot
            .timeline_duration_sec
            .max(self.snapshot.audio_duration_sec)
            .max(8.0);
        let base_pps = (ui.available_width() - 210.0).max(200.0) / total_dur as f32;
        let pps = base_pps * self.timeline_zoom;
        let content_w = total_dur as f32 * pps + 30.0;
        let audio_height = (ui.available_height() - 22.0 - 52.0 - 78.0 - 6.0).max(68.0);
        let ptr = ui.input(|i| i.pointer.interact_pos());
        let released = ui.input(|i| i.pointer.any_released());
        let mut dropped: Option<(usize, usize)> = None;
        let mut seek_to: Option<f64> = None;

        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = Vec2::new(0.0, 0.0);
            // fixed header column
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing = Vec2::new(0.0, 0.0);
                ui.allocate_exact_size(Vec2::new(196.0, 22.0), Sense::hover());
                self.track_header(ui, "Transitions", self.trans_count(), 52.0);
                self.track_header(ui, "Clips", self.snapshot.slots.len(), 78.0);
                self.track_header(
                    ui,
                    "Music",
                    usize::from(
                        !self.snapshot.audio_path.is_empty()
                            || self.snapshot.audio_duration_sec > 0.0,
                    ),
                    audio_height,
                );
            });
            // scrolling lanes
            ScrollArea::horizontal()
                .id_salt("studio_lanes")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing = Vec2::new(0.0, 0.0);
                        // ruler
                        let (rrect, rresp) = ui.allocate_exact_size(
                            Vec2::new(content_w, 22.0),
                            Sense::click_and_drag(),
                        );
                        {
                            let p = ui.painter();
                            let mut t = 0.0f64;
                            while t <= total_dur + 0.01 {
                                let x = rrect.left() + t as f32 * pps;
                                p.line_segment(
                                    [
                                        Pos2::new(x, rrect.top() + 14.0),
                                        Pos2::new(x, rrect.bottom()),
                                    ],
                                    Stroke::new(1.0, Color32::from_rgb(70, 72, 90)),
                                );
                                p.text(
                                    Pos2::new(x + 3.0, rrect.top()),
                                    egui::Align2::LEFT_TOP,
                                    format!("00:{t:02.0}"),
                                    FontId::monospace(10.5),
                                    Color32::from_rgb(130, 133, 152),
                                );
                                t += 2.0;
                            }
                            // playhead diamond
                            let px = rrect.left() + self.playhead_sec as f32 * pps;
                            p.circle_filled(Pos2::new(px, rrect.top() + 5.0), 4.5, Color32::WHITE);
                            p.line_segment(
                                [
                                    Pos2::new(px, rrect.top()),
                                    Pos2::new(px, rrect.bottom() + 200.0),
                                ],
                                Stroke::new(1.5, Color32::from_white_alpha(220)),
                            );
                        }
                        if (rresp.clicked() || rresp.dragged())
                            && let Some(pos) = ptr
                        {
                            seek_to =
                                Some(((pos.x - rrect.left()) / pps).clamp(0.0, total_dur as f32)
                                    as f64);
                        }
                        // transitions lane
                        {
                            let (lrect, _) =
                                ui.allocate_exact_size(Vec2::new(content_w, 52.0), Sense::hover());
                            let p = ui.painter();
                            let mut any = false;
                            for s in &self.snapshot.slots {
                                if s.slot_index == 0 || s.transition_id == "cut" {
                                    continue;
                                }
                                any = true;
                                let bx = lrect.left() + s.dst_start_sec as f32 * pps;
                                let pill = Rect::from_center_size(
                                    Pos2::new(bx, lrect.center().y),
                                    Vec2::new(108.0, 24.0),
                                );
                                p.rect_filled(
                                    pill,
                                    CornerRadius::same(6),
                                    Color32::from_rgb(122, 95, 30),
                                );
                                p.text(
                                    pill.center(),
                                    egui::Align2::CENTER_CENTER,
                                    trunc_chars(&s.transition_id, 15),
                                    FontId::proportional(11.0),
                                    Color32::from_rgb(245, 235, 200),
                                );
                            }
                            if !any {
                                p.text(
                                    Pos2::new(lrect.left() + 8.0, lrect.center().y - 7.0),
                                    egui::Align2::LEFT_TOP,
                                    "hard cuts",
                                    FontId::proportional(11.0),
                                    Color32::from_rgba_unmultiplied(130, 133, 152, 120),
                                );
                            }
                        }
                        // clips lane
                        {
                            let (lrect, _) =
                                ui.allocate_exact_size(Vec2::new(content_w, 78.0), Sense::hover());
                            let p = ui.painter();
                            for s in self.snapshot.slots.clone() {
                                let x = lrect.left() + s.dst_start_sec as f32 * pps;
                                let w = (s.duration_sec as f32 * pps).max(26.0);
                                let crect = Rect::from_min_size(
                                    Pos2::new(x + 1.0, lrect.top() + 4.0),
                                    Vec2::new(w - 2.0, 70.0),
                                );
                                let selected = self.selected_slot_idx == s.slot_index;
                                p.rect_filled(
                                    crect,
                                    CornerRadius::same(6),
                                    Color32::from_rgb(49, 84, 145),
                                );
                                p.rect_stroke(
                                    crect,
                                    CornerRadius::same(6),
                                    Stroke::new(
                                        if selected { 2.0 } else { 1.0 },
                                        if selected {
                                            Color32::WHITE
                                        } else {
                                            Color32::from_rgb(30, 70, 160)
                                        },
                                    ),
                                    StrokeKind::Inside,
                                );
                                let max_chars = ((w - 12.0) / 6.4).clamp(3.0, 30.0) as usize;
                                p.text(
                                    Pos2::new(crect.left() + 7.0, crect.top() + 6.0),
                                    egui::Align2::LEFT_TOP,
                                    trunc_chars(&self.media_display_name(s.shot_id), max_chars),
                                    FontId::proportional(11.5),
                                    Color32::WHITE,
                                );
                                p.text(
                                    Pos2::new(crect.left() + 7.0, crect.top() + 24.0),
                                    egui::Align2::LEFT_TOP,
                                    format!(
                                        "{:02}:{:02}–{:02}:{:02} · {:.1}s",
                                        (s.src_in_sec / 60.0).floor() as u64,
                                        (s.src_in_sec % 60.0).floor() as u64,
                                        (s.src_out_sec / 60.0).floor() as u64,
                                        (s.src_out_sec % 60.0).floor() as u64,
                                        s.duration_sec
                                    ),
                                    FontId::monospace(10.0),
                                    Color32::from_rgb(215, 225, 255),
                                );
                                // beat strength ticks
                                let ticks = (s.beat_strength * 8.0).round() as usize;
                                if w > 70.0 {
                                    p.text(
                                        Pos2::new(crect.left() + 7.0, crect.bottom() - 18.0),
                                        egui::Align2::LEFT_TOP,
                                        "▮".repeat(ticks.min(8)),
                                        FontId::proportional(9.0),
                                        Color32::from_rgba_unmultiplied(255, 255, 255, 130),
                                    );
                                }
                                let resp = ui.interact(
                                    crect,
                                    ui.make_persistent_id(("lane_clip", s.slot_index)),
                                    Sense::click(),
                                );
                                if resp.clicked() {
                                    self.selected_slot_idx = s.slot_index;
                                    self.selected_shot_id = s.shot_id;
                                    seek_to = Some(s.dst_start_sec);
                                }
                                if let Some(drag_id) = self.dragging_shot_id
                                    && let Some(pos) = ptr
                                    && crect.contains(pos)
                                {
                                    p.rect_stroke(
                                        crect,
                                        CornerRadius::same(6),
                                        Stroke::new(2.5, Color32::from_rgb(76, 224, 179)),
                                        StrokeKind::Inside,
                                    );
                                    if released {
                                        dropped = Some((s.slot_index, drag_id));
                                    }
                                }
                            }
                            // playhead
                            let px = lrect.left() + self.playhead_sec as f32 * pps;
                            if px >= lrect.left() && px <= lrect.right() {
                                p.line_segment(
                                    [Pos2::new(px, lrect.top()), Pos2::new(px, lrect.bottom())],
                                    Stroke::new(1.5, Color32::from_white_alpha(220)),
                                );
                            }
                        }
                        // Beat markers share the timeline ruler and zoom.
                        let (lane, _) = ui.allocate_exact_size(
                            Vec2::new(content_w, audio_height),
                            Sense::hover(),
                        );
                        let audio_width = self.snapshot.audio_duration_sec as f32 * pps;
                        let audio_rect = Rect::from_min_size(
                            lane.min + Vec2::new(0.0, 4.0),
                            Vec2::new(audio_width, (audio_height - 8.0).max(40.0)),
                        );
                        self.render_audio_lane(ctx, ui, audio_rect, busy);
                    });
                });
        });
        if let Some(t) = seek_to {
            self.live_timeline_mode = true;
            self.seek_to(t);
        }
        if let Some((slot_idx, shot_id)) = dropped {
            self.dragging_shot_id = None;
            if let Some(s) = self.snapshot.slots.get_mut(slot_idx) {
                s.shot_id = shot_id;
                s.locked = true;
            }
            self.selected_slot_idx = slot_idx;
            self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                let _ = ctrl.swap_slot_shot(slot_idx, shot_id);
            });
        }
        if released && self.dragging_shot_id.is_some() && dropped.is_none() {
            // released outside any clip — cancel drag
        }
    }

    fn trans_count(&self) -> usize {
        self.snapshot
            .slots
            .iter()
            .filter(|s| s.slot_index > 0 && s.transition_id != "cut")
            .count()
    }
}

struct DraftPreviewFrames {
    seq: u64,
    enhanced_path: Option<PathBuf>,
    start_frame: u64,
    frames: Vec<RgbaFrame>,
    enhanced: Option<Vec<RgbaFrame>>,
}

fn decode_preview_chunk(
    original: &Path,
    enhanced: Option<&Path>,
    start_frame: u64,
    duration: f64,
) -> ae_core::Result<DraftPreviewFrames> {
    let backend = FfmpegBackend::new();
    let start = Rational::from_frame_index(start_frame as i64, Rational::from_secs(30));
    let length = Rational::from_secs_f64((duration - start.as_secs_f64()).clamp(1.0 / 30.0, 1.0));
    let decode = |path: &Path| -> ae_core::Result<Vec<RgbaFrame>> {
        let info = backend.probe_video(path)?;
        let (width, height) = info.oriented_dimensions();
        let frames = backend.decode_segment(
            path,
            start,
            length,
            Resolution::new(width, height),
            Rational::from_secs(30),
            1.0,
            None,
        )?;
        if frames.is_empty() {
            return Err(ae_core::AeError::Config(
                "draft contains no frame at this time".into(),
            ));
        }
        Ok(frames)
    };
    Ok(DraftPreviewFrames {
        seq: 0,
        enhanced_path: enhanced.map(Path::to_path_buf),
        start_frame,
        frames: decode(original)?,
        enhanced: enhanced.map(decode).transpose()?,
    })
}

#[cfg(test)]
pub(crate) fn decode_draft_frame(
    path: &Path,
    playhead_ms: u64,
    duration: f64,
) -> ae_core::Result<RgbaFrame> {
    let backend = FfmpegBackend::new();
    let info = backend.probe_video(path)?;
    let (width, height) = info.oriented_dimensions();
    let start =
        Rational::from_secs_f64((playhead_ms as f64 / 1000.0).min((duration - 0.04).max(0.0)));
    backend
        .decode_segment(
            path,
            start,
            Rational::new(1, 30),
            Resolution::new(width, height),
            Rational::from_secs(30),
            1.0,
            None,
        )?
        .into_iter()
        .next()
        .ok_or_else(|| ae_core::AeError::Config("draft contains no frame at this time".into()))
}

fn paint_tab_icon(painter: &egui::Painter, tab: LeftTab, center: Pos2, color: Color32) {
    let stroke = Stroke::new(1.5, color);
    let point = |x, y| center + Vec2::new(x, y);
    match tab {
        LeftTab::Media => {
            painter.rect_stroke(
                Rect::from_center_size(center, Vec2::splat(16.0)),
                1.0,
                stroke,
                StrokeKind::Inside,
            );
            for x in [-3.0, 3.0] {
                painter.line_segment([point(x, -7.0), point(x, 7.0)], stroke);
            }
            for y in [-3.0, 3.0] {
                painter.line_segment([point(-7.0, y), point(7.0, y)], stroke);
            }
        }
        LeftTab::Audio => {
            painter.line_segment([point(-2.0, 5.0), point(-2.0, -6.0)], stroke);
            painter.line_segment([point(-2.0, -6.0), point(6.0, -8.0)], stroke);
            painter.line_segment([point(6.0, -8.0), point(6.0, 3.0)], stroke);
            painter.circle_filled(point(-5.0, 6.0), 3.0, color);
            painter.circle_filled(point(3.0, 4.0), 3.0, color);
        }
        LeftTab::Transitions => {
            painter.line_segment([point(-8.0, -4.0), point(8.0, -4.0)], stroke);
            painter.line_segment([point(4.0, -8.0), point(8.0, -4.0)], stroke);
            painter.line_segment([point(4.0, 0.0), point(8.0, -4.0)], stroke);
            painter.line_segment([point(8.0, 4.0), point(-8.0, 4.0)], stroke);
            painter.line_segment([point(-4.0, 0.0), point(-8.0, 4.0)], stroke);
            painter.line_segment([point(-4.0, 8.0), point(-8.0, 4.0)], stroke);
        }
        LeftTab::Settings => {
            painter.circle_stroke(center, 5.5, stroke);
            painter.circle_stroke(center, 2.0, stroke);
            for n in 0..8 {
                let angle = n as f32 * std::f32::consts::TAU / 8.0;
                let direction = Vec2::new(angle.cos(), angle.sin());
                painter.line_segment([center + direction * 5.5, center + direction * 8.5], stroke);
            }
        }
    }
}

fn trunc_chars(s: &str, n: usize) -> String {
    let count = s.chars().count();
    if count <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

#[derive(Clone)]
struct BrowserItem {
    shot_id: Option<usize>,
    name: String,
    dur_sec: f64,
    is_audio: bool,
}

impl Drop for AutoEditorEguiApp {
    fn drop(&mut self) {
        self.character_cancel.store(true, Ordering::Relaxed);
        self.stop_audio_companion();
    }
}

impl eframe::App for AutoEditorEguiApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // Initialize 5K-optimized UI zoom on first frame if monitor size is known
        if !self.zoom_initialized {
            if let Some(mon_size) = ctx.input(|i| i.viewport().monitor_size) {
                if mon_size.x >= 3840.0 {
                    self.ui_zoom = 1.50;
                } else if mon_size.x >= 2560.0 {
                    self.ui_zoom = 1.25;
                }
            }
            ctx.set_theme(egui::Theme::Dark);
            let mut style = (*ctx.style_of(egui::Theme::Dark)).clone();
            style.visuals = egui::Visuals::dark();
            style.visuals.panel_fill = Color32::from_rgb(29, 32, 39);
            style.visuals.window_fill = style.visuals.panel_fill;
            style.visuals.extreme_bg_color = Color32::from_rgb(21, 23, 28);
            style.visuals.faint_bg_color = Color32::from_rgb(34, 38, 47);
            style.visuals.override_text_color = Some(Color32::from_rgb(237, 240, 245));
            style.visuals.widgets.noninteractive.bg_stroke =
                Stroke::new(1.0, Color32::from_rgb(52, 58, 70));
            style.visuals.selection.bg_fill = Color32::from_rgb(49, 68, 103);
            style.visuals.selection.stroke = Stroke::new(1.0, Color32::from_rgb(137, 172, 255));
            style.spacing.button_padding = Vec2::new(10.0, 6.0);
            style
                .text_styles
                .insert(egui::TextStyle::Body, FontId::proportional(13.0));
            style
                .text_styles
                .insert(egui::TextStyle::Button, FontId::proportional(13.0));
            style
                .text_styles
                .insert(egui::TextStyle::Heading, FontId::proportional(17.0));
            ctx.set_style_of(egui::Theme::Dark, style);
            self.zoom_initialized = true;
        }

        // Keyboard shortcuts: Ctrl+=/Ctrl+-/Ctrl+0 for UI zoom, Space for Play/Pause
        ctx.input(|i| {
            if i.key_pressed(Key::Equals) && i.modifiers.matches_logically(Modifiers::COMMAND) {
                self.ui_zoom = (self.ui_zoom + 0.125).min(2.25);
            }
            if i.key_pressed(Key::Minus) && i.modifiers.matches_logically(Modifiers::COMMAND) {
                self.ui_zoom = (self.ui_zoom - 0.125).max(0.85);
            }
            if i.key_pressed(Key::Num0) && i.modifiers.matches_logically(Modifiers::COMMAND) {
                self.ui_zoom = 1.25;
            }
        });
        if ctx.input(|i| i.key_pressed(Key::Space)) && !ctx.egui_wants_keyboard_input() {
            self.toggle_play_pause();
        }

        ctx.set_zoom_factor(self.ui_zoom);
        self.uploads_this_frame = 0;

        // Receive completed snapshots from background workers only if they match the latest action_seq
        // and the user is not mid-drag on a beat pin!
        while let Ok((seq, new_snap, clear_shot_tex)) = self.snapshot_rx.try_recv() {
            if clear_shot_tex {
                self.shot_textures.clear();
            }
            if seq == self.action_seq && self.dragging_beat_idx.is_none() {
                self.res_input = new_snap.resolution.clone();
                self.snapshot = new_snap;
                self.pending_resequence = false;
            }
        }

        self.poll_enhanced_render();
        self.poll_character_analysis(&ctx);

        // Poll background task progress (lock-free atomic read)
        let (raw_busy, _pct, _progress_msg) = get_gui_progress();
        if !raw_busy
            && !self.pending_resequence
            && !self.enhancement_pending
            && self.snapshot.character_analysis_pending
            && !self.snapshot.video_path.is_empty()
        {
            self.start_character_analysis(ctx.clone());
        }
        let mut busy = raw_busy
            || self.character_analysis_running
            || self.pending_resequence
            || self.enhancement_pending;
        if !busy
            && self.preview_needs_render
            && !self.snapshot.slots.is_empty()
            && self
                .snapshot
                .beats
                .iter()
                .any(|beat| beat.enabled && !beat.is_ghost)
        {
            self.preview_needs_render = false;
            self.spawn_quick_render(ctx.clone());
            busy = true;
        }
        if !busy
            && self.compare_enhancement
            && self.enhanced_video_path.is_none()
            && !self.snapshot.draft_video_path.is_empty()
            && !self.preview_needs_render
        {
            self.spawn_enhanced_render(ctx.clone());
            busy = true;
        }
        if busy {
            ctx.request_repaint_after(Duration::from_millis(33));
        }

        // Advance playhead during live preview playback
        if self.playing {
            let dt = self.last_tick.elapsed().as_secs_f64();
            self.last_tick = Instant::now();
            self.playhead_sec += dt;

            let max_dur = self
                .snapshot
                .timeline_duration_sec
                .max(self.snapshot.audio_duration_sec)
                .max(0.1);
            if self.loop_enabled && self.loop_end_sec > self.loop_start_sec {
                if self.playhead_sec >= self.loop_end_sec {
                    let start = self.loop_start_sec;
                    self.seek_to(start);
                }
            } else if self.playhead_sec >= max_dur {
                self.seek_to(0.0);
            }
            ctx.request_repaint_after(Duration::from_millis(16));
        }

        // Studio header and export
        egui::Panel::top("studio_topbar")
            .exact_size(44.0)
            .show(ui, |ui| {
                ui.add_space(5.0);
                self.render_studio_topbar(&ctx, ui, busy);
            });

        // Timeline and playback
        egui::Panel::bottom("studio_timeline")
            .resizable(true)
            .default_size(280.0)
            .min_size(250.0)
            .show(ui, |ui| {
                ui.add_space(4.0);
                self.render_studio_timeline(&ctx, ui, busy);
                ui.take_available_height();
            });

        // Navigation rail
        egui::Panel::left("studio_rail")
            .exact_size(104.0)
            .show(ui, |ui| {
                self.render_studio_rail(ui);
            });

        // Media browser column
        egui::Panel::left("studio_browser")
            .resizable(true)
            .default_size(400.0)
            .min_size(290.0)
            .show(ui, |ui| {
                ui.add_space(6.0);
                self.render_studio_browser(&ctx, ui, busy);
            });

        // Center preview
        egui::CentralPanel::default().show(ui, |ui| {
            self.render_studio_preview(&ctx, ui);
        });

        // Render floating drag ghost when dragging a scene card toward the timeline
        if let Some(drag_shot_id) = self.dragging_shot_id {
            if let Some(pos) = ctx.input(|i| i.pointer.interact_pos()) {
                let tex = self.get_shot_texture(&ctx, drag_shot_id, 1);
                let layer = ctx.layer_painter(egui::LayerId::new(
                    egui::Order::Tooltip,
                    egui::Id::unique("scene_drag_ghost"),
                ));
                let ghost_rect =
                    Rect::from_min_size(pos + Vec2::new(14.0, 14.0), Vec2::new(148.0, 102.0));
                layer.rect_filled(
                    ghost_rect,
                    CornerRadius::same(6),
                    Color32::from_rgba_unmultiplied(20, 24, 34, 235),
                );
                layer.rect_stroke(
                    ghost_rect,
                    CornerRadius::same(6),
                    Stroke::new(2.0, Color32::from_rgb(76, 224, 179)),
                    StrokeKind::Outside,
                );
                let img_rect = Rect::from_min_size(
                    ghost_rect.min + Vec2::new(6.0, 6.0),
                    Vec2::new(136.0, 72.0),
                );
                layer.image(
                    tex.id(),
                    img_rect,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
                layer.text(
                    Pos2::new(ghost_rect.center().x, ghost_rect.bottom() - 11.0),
                    egui::Align2::CENTER_CENTER,
                    format!("Scene #{drag_shot_id:02} → Drop on Slot"),
                    FontId::proportional(11.0),
                    Color32::from_rgb(76, 224, 179),
                );
            }
            if ctx.input(|i| i.pointer.any_released()) {
                self.dragging_shot_id = None;
            }
        }

        // OS drag-and-drop support

        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .collect()
        });
        for path in dropped {
            if !busy {
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if matches!(ext.as_str(), "mp3" | "wav" | "flac" | "ogg" | "m4a" | "aac") {
                    self.start_import_audio(ctx.clone(), path);
                } else {
                    self.start_import_video(ctx.clone(), path);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn character_analysis_discards_replaced_sources_and_revisions() {
        let _guard = crate::tests::GUI_TEST_LOCK.lock().unwrap();
        let mut app = AutoEditorEguiApp::new(ProjectController::new());
        app.snapshot.video_path = "new.mp4".into();
        app.snapshot.character_revision = 3;
        app.character_generation = 2;
        app.character_analysis_running = true;
        for (generation, revision, source) in [(1, 3, "new.mp4"), (2, 3, "old.mp4")] {
            app.character_tx
                .send((
                    generation,
                    revision,
                    PathBuf::from(source),
                    Err("stale".into()),
                ))
                .unwrap();
        }
        app.poll_character_analysis(&Context::default());
        assert!(app.character_analysis_running);
        assert!(!app.pending_resequence);
        assert!(app.snapshot.character_analysis_error.is_empty());
        app.character_tx
            .send((2, 2, PathBuf::from("new.mp4"), Err("stale revision".into())))
            .unwrap();
        app.poll_character_analysis(&Context::default());
        assert!(!app.character_analysis_running);
        assert!(!app.pending_resequence);
        assert!(
            app.controller
                .lock()
                .unwrap()
                .character_analysis_error
                .is_empty()
        );
    }

    #[test]
    fn character_portraits_respect_revision_and_upload_budget() {
        let _guard = crate::tests::GUI_TEST_LOCK.lock().unwrap();
        let ctx = Context::default();
        let mut app = AutoEditorEguiApp::new(ProjectController::new());
        app.snapshot.character_revision = 123;
        let mut cache = global_media_cache().write().unwrap();
        let old_portraits = std::mem::take(&mut cache.character_portraits);
        let old_revision = cache.character_revision;
        cache.character_revision = 123;
        cache.character_portraits.insert(
            1,
            RgbaFrame::solid(10, 10, [255, 0, 0, 255], Rational::ZERO),
        );
        drop(cache);
        let first = app.get_character_texture(&ctx, 1);
        assert_eq!(app.uploads_this_frame, 1);
        assert_eq!(app.get_character_texture(&ctx, 1).id(), first.id());
        assert_eq!(app.uploads_this_frame, 1);
        app.snapshot.character_revision = 124;
        assert_ne!(app.get_character_texture(&ctx, 1).id(), first.id());
        app.snapshot.character_revision = 123;
        app.character_textures.clear();
        app.uploads_this_frame = 8;
        assert_ne!(app.get_character_texture(&ctx, 1).id(), first.id());
        assert!(app.character_textures.is_empty());
        let mut cache = global_media_cache().write().unwrap();
        cache.character_portraits = old_portraits;
        cache.character_revision = old_revision;
    }

    #[test]
    fn enhancement_divider_drags_and_clamps_to_preview_edges() {
        let ctx = Context::default();
        let mut app = AutoEditorEguiApp::new(ProjectController::new());
        let texture = app.get_placeholder_texture(&ctx);
        let rect = Rect::from_min_size(Pos2::new(100.0, 100.0), Vec2::new(400.0, 200.0));
        for (x, pressed, expected) in [
            (300.0, None, 0.5),
            (300.0, Some(true), 0.5),
            (400.0, None, 0.75),
            (600.0, None, 1.0),
            (50.0, None, 0.0),
            (50.0, Some(false), 0.0),
        ] {
            let pos = Pos2::new(x, 200.0);
            let mut events = vec![egui::Event::PointerMoved(pos)];
            if let Some(pressed) = pressed {
                events.push(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Modifiers::NONE,
                });
            }
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0))),
                    events,
                    ..Default::default()
                },
                |ui| app.paint_enhancement_comparison(ui, rect, &texture),
            );
            output.textures_delta.clear();
            assert!((app.enhancement_split - expected).abs() < 0.01);
        }
    }

    #[test]
    fn enhanced_render_rejects_old_edits_and_preserves_export_choice() {
        let mut app = AutoEditorEguiApp::new(ProjectController::new());
        app.action_seq = 2;
        app.compare_enhancement = true;
        app.quality_enhance_export = true;
        app.enhancement_pending = true;
        app.enhancement_tx
            .send((1, Ok(PathBuf::from("stale.mp4"))))
            .unwrap();
        app.poll_enhanced_render();
        assert!(app.enhanced_video_path.is_none());
        assert!(!app.enhancement_pending);

        let path = PathBuf::from("enhanced.mp4");
        app.enhancement_tx.send((2, Ok(path.clone()))).unwrap();
        app.poll_enhanced_render();
        assert_eq!(app.enhanced_video_path, Some(path.clone()));
        // Hiding and reopening the comparison retains the completed render.
        app.compare_enhancement = false;
        app.poll_enhanced_render();
        app.compare_enhancement = true;
        assert_eq!(app.enhanced_video_path, Some(path));
        assert!(!app.enhancement_pending);

        app.enhanced_video_path = None;
        app.enhancement_tx
            .send((2, Err("filter failed".into())))
            .unwrap();
        app.poll_enhanced_render();
        assert!(!app.compare_enhancement);
        assert!(app.quality_enhance_export);
        assert!(
            app.enhancement_error
                .as_ref()
                .unwrap()
                .contains("filter failed")
        );
    }

    #[test]
    fn face_tracking_toggle_invalidates_preview_and_updates_controller() {
        let ctx = Context::default();
        let mut app = AutoEditorEguiApp::new(ProjectController::new());
        for enabled in [true, false] {
            app.preview_needs_render = false;
            app.snapshot.draft_video_path = "old-preview.mp4".into();
            app.enhanced_video_path = Some(PathBuf::from("old-cleanup.mp4"));
            app.set_reframe_faces(ctx.clone(), enabled);
            assert!(app.preview_needs_render);
            assert!(app.snapshot.draft_video_path.is_empty());
            assert!(app.enhanced_video_path.is_none());
            let (seq, _, _) = app
                .snapshot_rx
                .recv_timeout(Duration::from_secs(5))
                .unwrap();
            assert_eq!(seq, app.action_seq);
            let ctrl = app.controller.lock().unwrap();
            assert_eq!(ctrl.reframe_faces, enabled);
            assert_eq!(
                ctrl.build_quick_render_config(
                    &ae_core::Timeline {
                        version: 1,
                        total_duration: Rational::ZERO,
                        mean_boundary_similarity: 1.0,
                        source_video: String::new(),
                        source_audio: String::new(),
                        resolution: Resolution::new(320, 180),
                        fps: Rational::from_secs(30),
                        slots: vec![]
                    },
                    Path::new("preview.mp4"),
                )
                .reframe_faces,
                enabled
            );
        }
    }

    #[test]
    fn rendered_playback_reuses_chunks_and_holds_the_rendered_image() {
        let ctx = Context::default();
        let mut app = AutoEditorEguiApp::new(ProjectController::new());
        app.snapshot.draft_video_path = "not-opened.mp4".into();
        app.snapshot.timeline_duration_sec = 2.0;
        app.playing = true;
        for start_frame in [0, 30] {
            app.draft_frame_tx
                .send((
                    0,
                    None,
                    Ok(DraftPreviewFrames {
                        seq: 0,
                        enhanced_path: None,
                        start_frame,
                        frames: (0..30)
                            .map(|i| {
                                RgbaFrame::solid(
                                    4,
                                    4,
                                    [i, 0, 0, 255],
                                    Rational::new((start_frame + i as u64) as i64, 30),
                                )
                            })
                            .collect(),
                        enhanced: None,
                    }),
                ))
                .unwrap();
        }
        for ms in [0, 34, 100, 999, 1000, 1034, 1999, 2000, 0] {
            let (texture, _) = app.get_preview_texture(&ctx, ms);
            let (_, index, cached, _) = app.draft_preview.as_ref().unwrap();
            assert_eq!(*index, (ms * 30 / 1000).min(59));
            assert_eq!(texture.id(), cached.id());
            assert!(
                !app.draft_frame_pending,
                "cached playback launched a decoder"
            );
        }
        // A cache miss must hold the rendered crop, never switch to the source proxy.
        let held = app.draft_preview.as_ref().unwrap().2.id();
        app.draft_frames.clear();
        app.draft_frame_pending = true;
        assert_eq!(app.get_preview_texture(&ctx, 1000).0.id(), held);
        // Old edit results cannot repopulate the cache.
        app.action_seq += 1;
        app.draft_frame_tx
            .send((0, None, Err("stale failure".into())))
            .unwrap();
        app.snapshot.timeline_duration_sec = 0.0;
        let (texture, _) = app.get_preview_texture(&ctx, 0);
        assert_ne!(texture.id(), held);
        assert!(app.draft_frames.is_empty());
        assert!(!app.snapshot.draft_video_path.is_empty());
    }

    #[test]
    fn cached_comparison_decodes_matching_times_across_all_clips() {
        let tmp = tempfile::tempdir().unwrap();
        let before = tmp.path().join("before.mp4");
        let after = tmp.path().join("after.mp4");
        for (path, colors) in [
            (&before, [[140, 20, 20], [20, 140, 20]]),
            (&after, [[200, 30, 30], [30, 200, 30]]),
        ] {
            ae_media::generate_synthetic_multi_shot_video(
                path,
                Resolution::new(64, 36),
                Rational::from_secs(30),
                &[(colors[0], colors[0], 36), (colors[1], colors[1], 36)],
            )
            .unwrap();
        }
        for (ms, channel) in [(0, 0), (100, 0), (1150, 0), (1250, 1), (2350, 1), (0, 0)] {
            let start_frame = ms / 1000 * 30;
            let chunk = decode_preview_chunk(&before, Some(&after), start_frame, 2.4).unwrap();
            assert_eq!(chunk.frames.len(), if start_frame == 60 { 12 } else { 30 });
            let index = (ms * 30 / 1000 - start_frame) as usize;
            let original = &chunk.frames[index];
            let enhanced = &chunk.enhanced.as_ref().unwrap()[index];
            assert!(original.pixel(32, 18)[channel] > 100);
            assert!(enhanced.pixel(32, 18)[channel] > 180);
            assert_eq!(
                (original.width, original.height, original.pts),
                (enhanced.width, enhanced.height, enhanced.pts)
            );
        }
    }

    #[test]
    fn whole_video_enhancement_is_cached_until_the_rendered_edit_changes() {
        use ae_core::{Timeline, TimelineSlot, TransitionSpec};
        use eframe::App;
        let _guard = crate::tests::GUI_TEST_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source.mp4");
        let fps = Rational::from_secs(30);
        ae_media::generate_synthetic_multi_shot_video(
            &source,
            Resolution::new(64, 64),
            fps,
            &[
                ([150, 20, 20], [150, 20, 20], 6),
                ([20, 150, 20], [20, 150, 20], 6),
            ],
        )
        .unwrap();
        let mut ctrl = ProjectController::new();
        ctrl.video_path = Some(source.clone());
        let audio = tmp.path().join("audio.wav");
        ae_beats::generate_synthetic_bass_and_kick_wav(&audio, 0.4, &[(0.05, 65.41)], &[]).unwrap();
        ctrl.audio_path = Some(audio);
        ctrl.resolution = Resolution::new(128, 128);
        ctrl.timeline_model.timeline = Some(Timeline {
            version: 1,
            source_video: source.to_string_lossy().into_owned(),
            source_audio: String::new(),
            resolution: ctrl.resolution,
            fps,
            total_duration: Rational::from_frame_index(12, fps),
            mean_boundary_similarity: 1.0,
            slots: (0..2)
                .map(|i| TimelineSlot {
                    slot_index: i,
                    dst_start: Rational::from_frame_index(i as i64 * 6, fps),
                    dst_end: Rational::from_frame_index(i as i64 * 6 + 6, fps),
                    shot_id: i,
                    src_in: Rational::from_frame_index(i as i64 * 6, fps),
                    src_out: Rational::from_frame_index(i as i64 * 6 + 6, fps),
                    speed: 1.0,
                    transition_in: TransitionSpec::default(),
                    boundary_similarity: 1.0,
                    beat_strength: 1.0,
                    locked: false,
                })
                .collect(),
        });
        ctrl.quick_render_draft(2).unwrap();
        let mut app = AutoEditorEguiApp::new(ctrl);
        app.preview_needs_render = false;
        app.compare_enhancement = true;
        let ctx = Context::default();
        let mut frame = eframe::Frame::_new_kittest();
        let input = || egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1200.0, 760.0))),
            ..Default::default()
        };
        let mut output = ctx.run_ui(input(), |ui| app.ui(ui, &mut frame));
        output.textures_delta.clear();
        assert!(app.enhancement_pending);
        let result = app
            .enhancement_rx
            .recv_timeout(Duration::from_secs(60))
            .unwrap();
        assert!(result.1.is_ok(), "{:?}", result.1);
        app.enhancement_tx.send(result).unwrap();
        let mut output = ctx.run_ui(input(), |ui| app.ui(ui, &mut frame));
        output.textures_delta.clear();
        assert!(!app.enhancement_pending);
        let path = app.enhanced_video_path.clone().unwrap();
        let backend = FfmpegBackend::new();
        let frames = backend
            .decode_segment(
                &path,
                Rational::ZERO,
                Rational::from_frame_index(12, fps),
                Resolution::new(64, 64),
                fps,
                1.0,
                None,
            )
            .unwrap();
        assert_eq!(frames.len(), 12);
        assert_eq!(backend.probe_video(&path).unwrap().fps, fps);
        let audio = backend.probe_audio(&path).unwrap();
        assert!(
            (audio.duration.as_secs_f64() - 0.4).abs() < 0.05,
            "audio timing changed"
        );
        assert!(frames[0].pixel(32, 32)[0] > 100);
        assert!(frames[11].pixel(32, 32)[1] > 100);
        for showing in [false, true, false, true] {
            app.compare_enhancement = showing;
            let mut output = ctx.run_ui(input(), |ui| app.ui(ui, &mut frame));
            output.textures_delta.clear();
            assert!(
                !app.enhancement_pending,
                "comparison reran preview filtering"
            );
            assert_eq!(app.enhanced_video_path.as_ref(), Some(&path));
        }
        let mut ctrl = app.controller.lock().unwrap();
        let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
        assert_eq!(ctrl.enhance_draft().unwrap(), path);
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            modified
        );
        let slot = &mut ctrl.timeline_model.timeline.as_mut().unwrap().slots[1];
        slot.shot_id = 9;
        slot.src_in = Rational::ZERO;
        slot.src_out = Rational::from_frame_index(6, fps);
        ctrl.quick_render_draft(2).unwrap();
        let edited = ctrl.enhance_draft().unwrap();
        assert_ne!(edited, path);
        let last = decode_draft_frame(&edited, 350, 0.4).unwrap();
        assert!(last.pixel(32, 32)[0] > 100, "updated edit was not enhanced");
        assert!(path.is_file(), "previously completed cache was overwritten");
    }

    #[test]
    fn automatically_renders_the_selected_output_resolution() {
        use ae_core::Resolution;
        use eframe::App;
        let _guard = crate::tests::GUI_TEST_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let video = tmp.path().join("video.mp4");
        let audio = tmp.path().join("audio.wav");
        ae_media::generate_synthetic_multi_shot_video(
            &video,
            Resolution::new(64, 36),
            Rational::from_secs(30),
            &[
                ([200, 20, 20], [200, 20, 20], 30),
                ([20, 200, 20], [20, 200, 20], 30),
            ],
        )
        .unwrap();
        ae_beats::generate_synthetic_bass_and_kick_wav(
            &audio,
            1.8,
            &[(0.5, 65.41), (1.0, 82.41)],
            &[],
        )
        .unwrap();
        let mut controller = ProjectController::new();
        controller.set_resolution_str("120x200").unwrap();
        controller.import_video(&video).unwrap();
        controller.seed_character_detections_for_test();
        controller.import_audio(&audio).unwrap();
        let mut app = AutoEditorEguiApp::new(controller);
        assert!(!app.snapshot.slots.is_empty());
        let ctx = Context::default();
        let mut frame = eframe::Frame::_new_kittest();
        let input = || egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1200.0, 760.0))),
            ..Default::default()
        };
        let beats = app.snapshot.beats.clone();
        assert!(beats.iter().any(|beat| beat.enabled && !beat.is_ghost));
        for inactive_beats in [
            Vec::new(),
            beats
                .iter()
                .cloned()
                .map(|mut beat| {
                    beat.enabled = false;
                    beat
                })
                .collect(),
            beats
                .iter()
                .cloned()
                .map(|mut beat| {
                    beat.is_ghost = true;
                    beat
                })
                .collect(),
        ] {
            app.snapshot.beats = inactive_beats;
            let seq = app.action_seq;
            let mut output = ctx.run_ui(input(), |ui| app.ui(ui, &mut frame));
            output.textures_delta.clear();
            assert_eq!(
                app.action_seq, seq,
                "inactive beats triggered a preview render"
            );
            assert!(app.preview_needs_render);
        }
        app.snapshot.beats = beats;
        let mut output = ctx.run_ui(input(), |ui| app.ui(ui, &mut frame));
        output.textures_delta.clear();
        let (seq, snapshot, clear) = app
            .snapshot_rx
            .recv_timeout(Duration::from_secs(30))
            .unwrap();
        assert!(
            !snapshot.draft_video_path.is_empty(),
            "{}",
            snapshot.status_message
        );
        let rendered = decode_draft_frame(
            Path::new(&snapshot.draft_video_path),
            100,
            snapshot.timeline_duration_sec,
        )
        .unwrap();
        assert_eq!((rendered.width, rendered.height), (64, 100));
        assert_eq!((snapshot.target_width, snapshot.target_height), (120, 200));
        app.snapshot_tx.send((seq, snapshot, clear)).unwrap();
        let mut output = ctx.run_ui(input(), |ui| app.ui(ui, &mut frame));
        output.textures_delta.clear();
        assert_eq!(app.action_seq, seq, "unchanged timeline was rendered twice");
        app.spawn_controller_action(ctx.clone(), false, |controller| {
            controller.set_resolution_str("160x240").unwrap();
        });
        let (seq, snapshot, clear) = app
            .snapshot_rx
            .recv_timeout(Duration::from_secs(30))
            .unwrap();
        app.snapshot_tx.send((seq, snapshot, clear)).unwrap();
        let mut output = ctx.run_ui(input(), |ui| app.ui(ui, &mut frame));
        output.textures_delta.clear();
        let (_, snapshot, _) = app
            .snapshot_rx
            .recv_timeout(Duration::from_secs(30))
            .unwrap();
        let rendered = decode_draft_frame(
            Path::new(&snapshot.draft_video_path),
            100,
            snapshot.timeline_duration_sec,
        )
        .unwrap();
        assert_eq!((rendered.width, rendered.height), (80, 120));
        assert_eq!((snapshot.target_width, snapshot.target_height), (160, 240));
    }

    #[test]
    fn editing_invalidates_the_rendered_preview() {
        let mut app = AutoEditorEguiApp::new(ProjectController::new());
        app.snapshot.draft_video_path = "old-draft.mp4".into();
        app.enhanced_video_path = Some(PathBuf::from("old-enhanced.mp4"));
        app.spawn_controller_action(Context::default(), false, |_| {});
        assert!(app.snapshot.draft_video_path.is_empty());
        assert!(app.enhanced_video_path.is_none());
    }

    #[test]
    fn playback_requires_a_timeline_or_audio() {
        let mut app = AutoEditorEguiApp::new(ProjectController::new());
        app.toggle_play_pause();
        assert!(!app.playing);

        app.snapshot.audio_duration_sec = 12.0;
        app.toggle_play_pause();
        assert!(app.playing);
        app.toggle_play_pause();
        assert!(!app.playing);
    }

    #[test]
    fn selecting_a_beat_does_not_disable_it() {
        let ctx = Context::default();
        let mut app = AutoEditorEguiApp::new(ProjectController::new());
        app.snapshot.audio_duration_sec = 2.0;
        app.snapshot.beats = vec![crate::BeatViewItem {
            index: 0,
            time_sec: 0.5,
            strength: 1.0,
            bass_freq_hz: 60.0,
            is_downbeat: true,
            enabled: true,
            is_manual: false,
            is_ghost: false,
        }];
        let position = Pos2::new(300.0, 150.0);
        for pressed in [None, Some(true), Some(false)] {
            let mut events = vec![egui::Event::PointerMoved(position)];
            if let Some(pressed) = pressed {
                events.push(egui::Event::PointerButton {
                    pos: position,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Modifiers::NONE,
                });
            }
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1000.0, 400.0))),
                    events,
                    ..Default::default()
                },
                |ui| {
                    app.render_audio_lane(
                        &ctx,
                        ui,
                        Rect::from_min_size(Pos2::new(100.0, 100.0), Vec2::new(800.0, 100.0)),
                        false,
                    )
                },
            );
            output.textures_delta.clear();
        }
        assert_eq!(app.selected_beat_idx, Some(0));
        assert_eq!(app.left_tab, LeftTab::Audio);
        assert!(app.snapshot.beats[0].enabled);
        assert_eq!(
            app.action_seq, 0,
            "selecting must not trigger a backend edit"
        );
    }

    #[test]
    fn resized_timeline_keeps_its_height_across_frames_and_tabs() {
        use eframe::App;
        let ctx = Context::default();
        let mut app = AutoEditorEguiApp::new(ProjectController::new());
        let mut frame = eframe::Frame::_new_kittest();
        for pass in 0..8 {
            app.left_tab = if pass % 2 == 0 {
                LeftTab::Media
            } else {
                LeftTab::Audio
            };
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1200.0, 900.0))),
                    ..Default::default()
                },
                |ui| {
                    let id = ui.scope_id().with("studio_timeline");
                    if pass == 0 {
                        ctx.data_mut(|data| {
                            data.insert_persisted(
                                id,
                                egui::containers::panel::PanelState {
                                    outer_rect: Rect::from_min_size(
                                        Pos2::new(0.0, 480.0),
                                        Vec2::new(1200.0, 420.0),
                                    ),
                                },
                            )
                        });
                    }
                    app.ui(ui, &mut frame);
                    let state = egui::containers::panel::PanelState::load(&ctx, id).unwrap();
                    assert!(
                        (state.size().y - 420.0).abs() < 1.0,
                        "timeline reset on frame {pass}: {:?}",
                        state.size()
                    );
                },
            );
            output.textures_delta.clear();
        }
    }

    #[test]
    fn transition_titles_share_alignment_and_row_spacing() {
        let ctx = Context::default();
        let mut app = AutoEditorEguiApp::new(ProjectController::new());
        app.left_tab = LeftTab::Transitions;
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(400.0, 700.0))),
                ..Default::default()
            },
            |ui| app.render_studio_browser(&ctx, ui, false),
        );
        output.textures_delta.clear();
        let titles: Vec<_> = output
            .shapes
            .iter()
            .filter_map(|shape| match &shape.shape {
                egui::Shape::Text(text)
                    if ["Crossfade", "Cut", "Dip", "Flash"].contains(&text.galley.text()) =>
                {
                    Some(text.pos)
                }
                _ => None,
            })
            .collect();
        assert_eq!(titles.len(), 4);
        for pair in titles.windows(2) {
            assert!(
                (pair[0].x - pair[1].x).abs() < 0.1,
                "transition text is not aligned: {titles:?}"
            );
        }
        let spacing = titles[1].y - titles[0].y;
        for pair in titles.windows(2) {
            assert!((pair[1].y - pair[0].y - spacing).abs() < 0.1);
        }
    }

    #[test]
    fn editor_panels_fit_minimum_window() {
        use eframe::App;

        for tab in [
            LeftTab::Media,
            LeftTab::Audio,
            LeftTab::Transitions,
            LeftTab::Settings,
        ] {
            let ctx = Context::default();
            let mut app = AutoEditorEguiApp::new(ProjectController::new());
            app.left_tab = tab;
            let mut frame = eframe::Frame::_new_kittest();
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1200.0, 760.0))),
                    ..Default::default()
                },
                |ui| app.ui(ui, &mut frame),
            );
            output.textures_delta.clear();
            let preview = output
                .shapes
                .iter()
                .find_map(|shape| match &shape.shape {
                    egui::Shape::Rect(rect) if rect.fill == Color32::from_rgb(8, 8, 10) => {
                        Some(rect.rect)
                    }
                    _ => None,
                })
                .expect("preview must remain visible on every tab");
            assert!(
                preview.width() > 300.0 && preview.height() > 200.0,
                "preview squeezed out on {tab:?}: {preview:?}"
            );
            assert!(preview.right() <= 1200.0 && preview.bottom() <= 760.0);
        }
    }

    #[test]
    fn preview_uses_output_dimensions() {
        let ctx = Context::default();
        let mut app = AutoEditorEguiApp::new(ProjectController::new());
        for resolution in ["1080x1920", "1920x1080", "1080x1080", "1000x600"] {
            let mut controller = app.controller.lock().unwrap();
            controller.set_resolution_str(resolution).unwrap();
            app.snapshot = controller.snapshot();
            drop(controller);
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0))),
                    ..Default::default()
                },
                |ui| app.render_studio_preview(&ctx, ui),
            );
            output.textures_delta.clear();
            let preview = output
                .shapes
                .iter()
                .find_map(|shape| match &shape.shape {
                    egui::Shape::Rect(rect) if rect.fill == Color32::from_rgb(16, 16, 20) => {
                        Some(rect.rect)
                    }
                    _ => None,
                })
                .expect("preview canvas must be painted");
            let ratio = app.snapshot.target_width as f32 / app.snapshot.target_height as f32;
            assert!(
                (preview.aspect_ratio() - ratio).abs() < 0.001,
                "wrong preview ratio for {resolution}"
            );
            assert!(preview.width() <= 800.0 && preview.height() <= 600.0);
        }
    }
}
