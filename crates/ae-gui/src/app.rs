use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use ae_core::{BeatMode, RgbaFrame, SeparationMode};
use crossbeam_channel::{Receiver, Sender, unbounded};
use egui::{
    Align, Color32, ColorImage, Context, CornerRadius, CursorIcon, DragValue, FontId, Key, Layout,
    Modifiers, Pos2, ProgressBar, Rect, RichText, ScrollArea, Sense, Stroke, StrokeKind,
    TextureHandle, TextureOptions, Ui, Vec2,
};

use crate::image_provider::{get_gui_progress, global_media_cache, set_gui_progress};
use crate::project_controller::{GuiStateSnapshot, ProjectController};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileDialogKind {
    ImportVideo,
    ImportAudio,
    ExportVideo,
}

#[derive(Clone, Debug)]
pub struct DirEntryItem {
    pub name: String,
    pub path: PathBuf,
    pub is_dir: bool,
    pub size_mb: f64,
}

#[derive(Clone, Debug)]
pub struct InAppFileDialog {
    pub kind: FileDialogKind,
    pub current_dir: PathBuf,
    pub entries: Vec<DirEntryItem>,
    pub selected_path: Option<PathBuf>,
    pub filename_input: String,
    pub path_bar_input: String,
}

impl InAppFileDialog {
    pub fn open(kind: FileDialogKind, start_dir: Option<PathBuf>) -> Self {
        let dir = start_dir
            .filter(|d| d.is_dir())
            .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("/"));
        let mut dlg = Self {
            kind,
            current_dir: dir.clone(),
            entries: Vec::new(),
            selected_path: None,
            filename_input: if kind == FileDialogKind::ExportVideo {
                "edited_out.mp4".to_string()
            } else {
                String::new()
            },
            path_bar_input: dir.to_string_lossy().to_string(),
        };
        dlg.refresh_entries();
        dlg
    }

    pub fn navigate_to(&mut self, dir: PathBuf) {
        if dir.is_dir() {
            self.current_dir = dir;
            self.path_bar_input = self.current_dir.to_string_lossy().to_string();
            self.selected_path = None;
            self.refresh_entries();
        }
    }

    fn matches_filter(&self, path: &Path) -> bool {
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            return false;
        };
        let ext_lower = ext.to_ascii_lowercase();
        match self.kind {
            FileDialogKind::ImportVideo => {
                matches!(
                    ext_lower.as_str(),
                    "mp4" | "mkv" | "mov" | "webm" | "avi" | "m4v" | "ts"
                )
            }
            FileDialogKind::ImportAudio => {
                matches!(
                    ext_lower.as_str(),
                    "mp3" | "wav" | "flac" | "ogg" | "m4a" | "aac" | "opus"
                )
            }
            FileDialogKind::ExportVideo => ext_lower == "mp4",
        }
    }

    pub fn refresh_entries(&mut self) {
        self.entries.clear();
        let Ok(read_dir) = std::fs::read_dir(&self.current_dir) else {
            return;
        };
        for entry in read_dir.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') {
                continue;
            }
            let meta = entry.metadata().ok();
            let is_dir = meta.as_ref().map(|m| m.is_dir()).unwrap_or(false);
            if !is_dir && !self.matches_filter(&path) {
                continue;
            }
            let size_mb = meta
                .as_ref()
                .map(|m| (m.len() as f64) / (1024.0 * 1024.0))
                .unwrap_or(0.0);
            self.entries.push(DirEntryItem {
                name,
                path,
                is_dir,
                size_mb,
            });
        }
        self.entries.sort_by(|a, b| {
            b.is_dir
                .cmp(&a.is_dir)
                .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
        });
    }
}

pub struct AutoEditorEguiApp {
    pub controller: Arc<Mutex<ProjectController>>,
    pub snapshot: GuiStateSnapshot,
    snapshot_tx: Sender<(u64, GuiStateSnapshot, bool)>, // (action_seq, new_snapshot, clear_shot_textures)
    snapshot_rx: Receiver<(u64, GuiStateSnapshot, bool)>,
    action_seq: u64,
    pending_resequence: bool,
    pub selected_shot_id: usize,
    pub selected_slot_idx: usize,
    pub live_timeline_mode: bool,
    pub res_input: String,
    pub draft_divisor: u32,
    pub use_gpu_export: bool,

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

    // In-app zero-stall file browser modal + last browsed directory
    file_dialog: Option<InAppFileDialog>,
    last_browse_dir: Option<PathBuf>,

    // GPU Texture caches & per-frame upload budget so 260+ scenes never spike frame time
    shot_textures: HashMap<(usize, u8), TextureHandle>,
    placeholder_tex: Option<TextureHandle>,
    uploads_this_frame: usize,
    spectrogram_tex: Option<(u64, TextureHandle)>,
    preview_tex: Option<(u64, u64, TextureHandle)>,
}

impl AutoEditorEguiApp {
    pub fn new(ctrl: ProjectController) -> Self {
        let snap = ctrl.snapshot();
        let res_input = snap.resolution.clone();
        let (snapshot_tx, snapshot_rx) = unbounded();
        Self {
            controller: Arc::new(Mutex::new(ctrl)),
            snapshot: snap,
            snapshot_tx,
            snapshot_rx,
            action_seq: 0,
            pending_resequence: false,
            selected_shot_id: 0,
            selected_slot_idx: 0,
            live_timeline_mode: true,
            res_input,
            draft_divisor: 3,
            use_gpu_export: false,
            ui_zoom: 1.35, // 5K-friendly default zoom, auto-refined on first frame
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
            file_dialog: None,
            last_browse_dir: None,
            shot_textures: HashMap::new(),
            placeholder_tex: None,
            uploads_this_frame: 0,
            spectrogram_tex: None,
            preview_tex: None,
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
        let tex = ctx.load_texture(
            format!("shot_{shot_id}_{pos}"),
            img,
            TextureOptions::LINEAR,
        );
        self.shot_textures.insert(key, tex.clone());
        tex
    }

    fn get_spectrogram_texture(&mut self, ctx: &Context) -> TextureHandle {
        let Ok(cache) = global_media_cache().try_read() else {
            return self
                .spectrogram_tex
                .as_ref()
                .map(|(_, t)| t.clone())
                .unwrap_or_else(|| self.get_placeholder_texture(ctx));
        };
        let rev = cache.spectrogram_revision;
        if let Some((cached_rev, ref tex)) = self.spectrogram_tex
            && cached_rev == rev
        {
            return tex.clone();
        }
        let frame = cache.get_spectrogram();
        drop(cache);

        let img = Self::rgba_to_color_image(&frame);
        let tex = ctx.load_texture(
            format!("spectrogram_{rev}"),
            img,
            TextureOptions::LINEAR,
        );
        self.spectrogram_tex = Some((rev, tex.clone()));
        tex
    }

    fn get_preview_texture(&mut self, ctx: &Context, playhead_ms: u64) -> TextureHandle {
        let rev = self.snapshot.preview_revision;
        let bucket_ms = (playhead_ms / 33) * 33; // 30 fps frame quantization
        if let Some((c_ms, c_rev, ref tex)) = self.preview_tex
            && c_ms == bucket_ms
            && c_rev == rev
        {
            return tex.clone();
        }
        let Ok(cache) = global_media_cache().try_read() else {
            return self
                .preview_tex
                .as_ref()
                .map(|(_, _, t)| t.clone())
                .unwrap_or_else(|| self.get_placeholder_texture(ctx));
        };
        let frame = cache.resolve_timeline_preview_frame(bucket_ms).0;
        drop(cache);

        let img = Self::rgba_to_color_image(&frame);
        let tex = ctx.load_texture("live_preview", img, TextureOptions::LINEAR);
        self.preview_tex = Some((bucket_ms, rev, tex.clone()));
        tex
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
        self.playing = !self.playing;
        self.last_tick = Instant::now();
        if self.playing {
            self.restart_audio_companion();
        } else {
            self.stop_audio_companion();
        }
    }

    fn seek_to(&mut self, sec: f64) {
        let max_dur = self.snapshot.timeline_duration_sec.max(self.snapshot.audio_duration_sec).max(0.1);
        self.playhead_sec = sec.clamp(0.0, max_dur);
        self.last_tick = Instant::now();
        if self.playing {
            self.restart_audio_companion();
        }
    }

    fn loop_around_cut(&mut self, slot_idx: usize, half_win: f64) {
        if let Some(slot) = self.snapshot.slots.get(slot_idx) {
            let cut = slot.dst_start_sec;
            let max_dur = self.snapshot.timeline_duration_sec.max(1.0);
            self.loop_start_sec = (cut - half_win).max(0.0);
            self.loop_end_sec = (cut + half_win).min(max_dur).max(self.loop_start_sec + 0.4);
            self.loop_enabled = true;
            self.live_timeline_mode = true;
            self.playing = true;
            self.seek_to(self.loop_start_sec);
        }
    }

    /// Dispatch any controller mutation + re-sequencing to a background worker thread so
    /// the egui UI thread never blocks for even 1 millisecond when clicking/dragging beats or scenes.
    /// Uses a monotonic `action_seq` so stale background results never overwrite newer user edits.
    fn spawn_controller_action<F>(&mut self, ctx: Context, clear_shot_tex: bool, action: F)
    where
        F: FnOnce(&mut ProjectController) + Send + 'static,
    {
        self.action_seq = self.action_seq.wrapping_add(1);
        let seq = self.action_seq;
        self.pending_resequence = true;
        let ctrl_arc = Arc::clone(&self.controller);
        let tx = self.snapshot_tx.clone();
        thread::spawn(move || {
            if let Ok(mut ctrl) = ctrl_arc.lock() {
                action(&mut ctrl);
                let _ = tx.send((seq, ctrl.snapshot(), clear_shot_tex));
                ctx.request_repaint();
            }
        });
    }

    fn start_import_video(&mut self, ctx: Context, path: PathBuf) {
        self.last_browse_dir = path.parent().map(|p| p.to_path_buf());
        self.action_seq = self.action_seq.wrapping_add(1);
        let seq = self.action_seq;
        let ctrl_arc = Arc::clone(&self.controller);
        let tx = self.snapshot_tx.clone();
        set_gui_progress(true, 2, "Starting video import...");
        ctx.request_repaint();
        thread::spawn(move || {
            if let Ok(mut ctrl) = ctrl_arc.lock() {
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
        self.last_browse_dir = path.parent().map(|p| p.to_path_buf());
        self.action_seq = self.action_seq.wrapping_add(1);
        let seq = self.action_seq;
        let ctrl_arc = Arc::clone(&self.controller);
        let tx = self.snapshot_tx.clone();
        set_gui_progress(true, 5, "Starting audio import...");
        ctx.request_repaint();
        thread::spawn(move || {
            if let Ok(mut ctrl) = ctrl_arc.lock() {
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
        self.action_seq = self.action_seq.wrapping_add(1);
        let seq = self.action_seq;
        let ctrl_arc = Arc::clone(&self.controller);
        let tx = self.snapshot_tx.clone();
        let div = self.draft_divisor;
        set_gui_progress(true, 4, "Starting incremental Quick Render...");
        thread::spawn(move || {
            if let Ok(mut ctrl) = ctrl_arc.lock() {
                if let Err(e) = ctrl.quick_render_draft(div) {
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
        set_gui_progress(true, 4, "Starting MP4 export...");
        ctx.request_repaint();
        thread::spawn(move || {
            if let Ok(mut ctrl) = ctrl_arc.lock() {
                if let Err(e) = ctrl.export_video(&path, use_gpu) {
                    ctrl.status_message = format!("Export error: {e}");
                    set_gui_progress(false, 0, ctrl.status_message.clone());
                }
                let _ = tx.send((seq, ctrl.snapshot(), false));
            }
            ctx.request_repaint();
        });
    }

    fn render_file_dialog_modal(&mut self, ctx: &Context) {
        let Some(mut dlg) = self.file_dialog.take() else {
            return;
        };

        let title = match dlg.kind {
            FileDialogKind::ImportVideo => "📂 Select Video Clip (.mp4, .mkv, .mov, .webm)",
            FileDialogKind::ImportAudio => "🎵 Select Audio Track (.mp3, .wav, .flac, .ogg, .m4a)",
            FileDialogKind::ExportVideo => "🎬 Export Edited MP4 Video",
        };

        let mut keep_open = true;
        let mut cancel_clicked = false;
        let mut confirmed_path: Option<(FileDialogKind, PathBuf)> = None;

        egui::Window::new(title)
            .open(&mut keep_open)
            .collapsible(false)
            .resizable(true)
            .default_size([740.0, 480.0])
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                // Top navigation & editable path bar
                ui.horizontal(|ui| {
                    if ui.button("⬆ Up").clicked()
                        && let Some(parent) = dlg.current_dir.parent().map(|p| p.to_path_buf())
                    {
                        dlg.navigate_to(parent);
                    }
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut dlg.path_bar_input)
                            .desired_width(ui.available_width() - 65.0),
                    );
                    if (resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)))
                        || ui.button("Go").clicked()
                    {
                        let p = PathBuf::from(dlg.path_bar_input.trim());
                        if p.is_file() {
                            confirmed_path = Some((dlg.kind, p));
                        } else if p.is_dir() {
                            dlg.navigate_to(p);
                        }
                    }
                });

                ui.separator();

                // Main split: Quick-access sidebar on left + directory listing on right
                ui.columns(2, |cols| {
                    cols[0].set_max_width(145.0);
                    cols[0].vertical(|ui| {
                        ui.label(RichText::new("Quick Places").strong());
                        ui.add_space(4.0);
                        if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
                            for (lbl, sub) in [
                                ("🏠 Home", ""),
                                ("🎬 Videos", "Videos"),
                                ("📥 Downloads", "Downloads"),
                                ("🎵 Music", "Music"),
                                ("📄 Documents", "Documents"),
                                ("🖥 Desktop", "Desktop"),
                            ] {
                                let target = if sub.is_empty() {
                                    home.clone()
                                } else {
                                    home.join(sub)
                                };
                                if target.is_dir() && ui.button(lbl).clicked() {
                                    dlg.navigate_to(target);
                                }
                            }
                        }
                        if let Ok(cwd) = std::env::current_dir()
                            && ui.button("📁 Project Dir").clicked()
                        {
                            dlg.navigate_to(cwd);
                        }
                    });

                    cols[1].vertical(|ui| {
                        ScrollArea::vertical()
                            .id_salt("in_app_file_browser_scroll")
                            .max_height(330.0)
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                let entries = dlg.entries.clone();
                                for item in entries {
                                    let is_selected = dlg
                                        .selected_path
                                        .as_ref()
                                        .is_some_and(|s| s == &item.path);
                                    let label = if item.is_dir {
                                        format!("📁 {}/", item.name)
                                    } else {
                                        format!("🎞 {}  ({:.1} MB)", item.name, item.size_mb)
                                    };
                                    let resp = ui.selectable_label(is_selected, label);
                                    if resp.clicked() {
                                        if item.is_dir {
                                            dlg.selected_path = None;
                                        } else {
                                            dlg.filename_input = item.name.clone();
                                            dlg.selected_path = Some(item.path.clone());
                                        }
                                    }
                                    if resp.double_clicked() {
                                        if item.is_dir {
                                            dlg.navigate_to(item.path);
                                        } else {
                                            confirmed_path = Some((dlg.kind, item.path));
                                        }
                                    }
                                }
                            });
                    });
                });

                ui.separator();

                // Bottom confirmation bar
                ui.horizontal(|ui| {
                    ui.label("File:");
                    ui.add(
                        egui::TextEdit::singleline(&mut dlg.filename_input).desired_width(340.0),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let confirm_label = if dlg.kind == FileDialogKind::ExportVideo {
                            "💾 Save MP4"
                        } else {
                            "✅ Open"
                        };
                        if ui
                            .button(
                                RichText::new(confirm_label)
                                    .strong()
                                    .color(Color32::from_rgb(20, 24, 32)),
                            )
                            .clicked()
                        {
                            if let Some(ref sel) = dlg.selected_path {
                                confirmed_path = Some((dlg.kind, sel.clone()));
                            } else if !dlg.filename_input.trim().is_empty() {
                                let candidate = dlg.current_dir.join(dlg.filename_input.trim());
                                confirmed_path = Some((dlg.kind, candidate));
                            }
                        }
                        if ui.button("Cancel").clicked() {
                            cancel_clicked = true;
                        }
                    });
                });
            });

        if let Some((kind, path)) = confirmed_path {
            // Close the confirmation modal immediately on this frame and dispatch background task!
            match kind {
                FileDialogKind::ImportVideo => self.start_import_video(ctx.clone(), path),
                FileDialogKind::ImportAudio => self.start_import_audio(ctx.clone(), path),
                FileDialogKind::ExportVideo => self.start_export_video(ctx.clone(), path),
            }
        } else if keep_open && !cancel_clicked {
            self.file_dialog = Some(dlg);
        }
    }

    fn render_top_toolbar(&mut self, ctx: &Context, ui: &mut Ui, busy: bool) {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = Vec2::new(8.0, 6.0);

            if ui
                .add_enabled(!busy, egui::Button::new(RichText::new("📂 Import Clip...").strong()))
                .clicked()
            {
                self.file_dialog = Some(InAppFileDialog::open(
                    FileDialogKind::ImportVideo,
                    self.last_browse_dir.clone(),
                ));
            }

            if ui
                .add_enabled(!busy, egui::Button::new(RichText::new("🎵 Import Audio...").strong()))
                .clicked()
            {
                self.file_dialog = Some(InAppFileDialog::open(
                    FileDialogKind::ImportAudio,
                    self.last_browse_dir.clone(),
                ));
            }

            ui.separator();
            ui.label("Resolution (Stretch):");

            egui::ComboBox::from_id_salt("res_preset_combo")
                .selected_text(&self.snapshot.resolution)
                .width(145.0)
                .show_ui(ui, |ui| {
                    for (label, val) in [
                        ("1080x1920 (Vertical)", "1080x1920"),
                        ("1920x1080 (1080p)", "1920x1080"),
                        ("1280x720 (720p)", "1280x720"),
                        ("3840x2160 (4K)", "3840x2160"),
                    ] {
                        if ui.selectable_label(self.snapshot.resolution == val, label).clicked() {
                            self.res_input = val.to_string();
                            let val_owned = val.to_string();
                            self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                                let _ = ctrl.set_resolution_str(&val_owned);
                            });
                        }
                    }
                });

            let res_resp = ui.add(
                egui::TextEdit::singleline(&mut self.res_input)
                    .desired_width(86.0)
                    .hint_text("WxH"),
            );
            if res_resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                let val_owned = self.res_input.clone();
                self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                    let _ = ctrl.set_resolution_str(&val_owned);
                });
            }

            ui.separator();
            ui.label("UI Scale (5K):");
            egui::ComboBox::from_id_salt("ui_scale_combo")
                .selected_text(format!("{}%", (self.ui_zoom * 100.0).round() as u32))
                .width(90.0)
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

            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui
                    .add_enabled(
                        !busy,
                        egui::Button::new(
                            RichText::new("🎬 Export MP4...")
                                .strong()
                                .color(Color32::from_rgb(20, 24, 32)),
                        )
                        .fill(Color32::from_rgb(76, 224, 179)),
                    )
                    .clicked()
                {
                    self.file_dialog = Some(InAppFileDialog::open(
                        FileDialogKind::ExportVideo,
                        self.last_browse_dir.clone(),
                    ));
                }

                if ui
                    .add_enabled(!busy, egui::Button::new("🔄 Re-Sequence"))
                    .clicked()
                {
                    self.spawn_controller_action(ctx.clone(), false, |ctrl| {
                        let _ = ctrl.resequence();
                    });
                }

                if ui
                    .add_enabled(!busy, egui::Button::new("⚡ Quick Render"))
                    .clicked()
                {
                    self.spawn_quick_render(ctx.clone());
                }

                egui::ComboBox::from_id_salt("draft_div_combo")
                    .selected_text(if self.draft_divisor == 3 {
                        "Draft 1/3 Res"
                    } else {
                        "Half 1/2 Res"
                    })
                    .width(115.0)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.draft_divisor, 3, "Draft 1/3 Res (Fastest)");
                        ui.selectable_value(&mut self.draft_divisor, 2, "Half 1/2 Res (Sharp)");
                    });
            });
        });
    }

    fn render_shot_pool(&mut self, ctx: &Context, ui: &mut Ui, busy: bool) {
        ui.horizontal(|ui| {
            ui.heading(format!(
                "1. Cut-Up Scenes ({} detected)",
                self.snapshot.shots.len()
            ));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.add_enabled(!busy, egui::Button::new("Invert")).clicked() {
                    for s in &mut self.snapshot.shots {
                        s.enabled = !s.enabled;
                    }
                    self.spawn_controller_action(ctx.clone(), false, |ctrl| {
                        let _ = ctrl.invert_shots();
                    });
                }
                if ui
                    .add_enabled(!busy, egui::Button::new("Deselect All"))
                    .clicked()
                {
                    for s in &mut self.snapshot.shots {
                        s.enabled = false;
                    }
                    self.spawn_controller_action(ctx.clone(), false, |ctrl| {
                        let _ = ctrl.set_all_shots_enabled(false);
                    });
                }
                if ui
                    .add_enabled(!busy, egui::Button::new("Select All"))
                    .clicked()
                {
                    for s in &mut self.snapshot.shots {
                        s.enabled = true;
                    }
                    self.spawn_controller_action(ctx.clone(), false, |ctrl| {
                        let _ = ctrl.set_all_shots_enabled(true);
                    });
                }
            });
        });
        ui.add_space(4.0);

        let shots = self.snapshot.shots.clone();
        if shots.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label(
                    RichText::new("Click 'Import Clip...' to detect scenes and generate thumbnails.")
                        .color(Color32::from_rgb(140, 150, 178)),
                );
            });
            return;
        }

        let card_w = 200.0f32;
        let thumb_h = 112.0f32;
        let card_outer_w = card_w + 14.0; // inner_margin(6.0)*2 + stroke
        let row_h = thumb_h + 48.0;
        let spacing_x = 10.0f32;
        let avail_w = ui.available_width().max(card_outer_w);
        let cols = ((avail_w + spacing_x) / (card_outer_w + spacing_x))
            .floor()
            .max(1.0) as usize;
        let total_rows = shots.len().div_ceil(cols);

        ScrollArea::vertical()
            .id_salt("shot_pool_scroll")
            .auto_shrink([false, false])
            .show_rows(ui, row_h, total_rows, |ui, row_range| {
                ui.spacing_mut().item_spacing = Vec2::new(spacing_x, 10.0);
                for row_idx in row_range {
                    let start_idx = row_idx * cols;
                    let end_idx = (start_idx + cols).min(shots.len());
                    ui.horizontal(|ui| {
                        for shot in &shots[start_idx..end_idx] {
                            let is_used = shot.used_in_slot >= 0;
                            let border_col = if is_used {
                                Color32::from_rgb(76, 224, 179)
                            } else if shot.enabled {
                                Color32::from_rgb(60, 70, 98)
                            } else {
                                Color32::from_rgb(38, 42, 56)
                            };
                            let bg_col = if shot.enabled {
                                Color32::from_rgb(32, 37, 52)
                            } else {
                                Color32::from_rgb(20, 23, 32)
                            };

                            egui::Frame::NONE
                                .fill(bg_col)
                                .stroke(Stroke::new(if is_used { 2.0 } else { 1.0 }, border_col))
                                .corner_radius(CornerRadius::same(6))
                                .inner_margin(6.0)
                                .show(ui, |ui| {
                                    ui.set_width(card_w);
                                    ui.vertical(|ui| {
                                        let (rect, resp) = ui.allocate_exact_size(
                                            Vec2::new(card_w, thumb_h),
                                            Sense::click_and_drag(),
                                        );
                                        if resp.drag_started() {
                                            self.dragging_shot_id = Some(shot.id);
                                        }
                                        let scrub_pos = if let Some(ptr) = resp.hover_pos() {
                                            let rel = ((ptr.x - rect.left())
                                                / rect.width().max(1.0))
                                            .clamp(0.0, 1.0);
                                            if rel < 0.33 {
                                                0
                                            } else if rel > 0.66 {
                                                2
                                            } else {
                                                1
                                            }
                                        } else {
                                            1
                                        };

                                        let tex = self.get_shot_texture(ctx, shot.id, scrub_pos);
                                        let uv =
                                            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
                                        let tint = if shot.enabled {
                                            Color32::WHITE
                                        } else {
                                            Color32::from_white_alpha(120)
                                        };
                                        ui.painter().image(tex.id(), rect, uv, tint);

                                        if is_used {
                                            let badge_rect = Rect::from_min_size(
                                                Pos2::new(rect.right() - 62.0, rect.top() + 4.0),
                                                Vec2::new(58.0, 18.0),
                                            );
                                            ui.painter().rect_filled(
                                                badge_rect,
                                                CornerRadius::same(4),
                                                Color32::from_rgba_unmultiplied(20, 130, 100, 220),
                                            );
                                            ui.painter().text(
                                                badge_rect.center(),
                                                egui::Align2::CENTER_CENTER,
                                                format!("Slot #{}", shot.used_in_slot),
                                                FontId::proportional(11.0),
                                                Color32::WHITE,
                                            );
                                        }

                                        if resp.clicked() {
                                            self.selected_shot_id = shot.id;
                                        }

                                        ui.horizontal(|ui| {
                                            let mut enabled = shot.enabled;
                                            if ui.checkbox(&mut enabled, "").changed() {
                                                let sid = shot.id;
                                                if let Some(s) = self
                                                    .snapshot
                                                    .shots
                                                    .iter_mut()
                                                    .find(|s| s.id == sid)
                                                {
                                                    s.enabled = enabled;
                                                }
                                                self.spawn_controller_action(
                                                    ctx.clone(),
                                                    false,
                                                    move |ctrl| {
                                                        let _ = ctrl.toggle_shot(sid);
                                                    },
                                                );
                                            }
                                            ui.vertical(|ui| {
                                                ui.label(
                                                    RichText::new(format!(
                                                        "Scene #{:02} ({:.2}s)",
                                                        shot.id, shot.duration_sec
                                                    ))
                                                    .strong()
                                                    .size(12.0),
                                                );
                                                ui.label(
                                                    RichText::new(format!(
                                                        "{:.2}s – {:.2}s",
                                                        shot.usable_start_sec, shot.usable_end_sec
                                                    ))
                                                    .color(Color32::from_rgb(145, 155, 182))
                                                    .size(10.5),
                                                );
                                            });
                                        });
                                    });
                                });
                        }
                    });
                }
            });
    }

    fn render_preview_monitor(&mut self, ctx: &Context, ui: &mut Ui, busy: bool) {
        ui.horizontal(|ui| {
            let title = if self.live_timeline_mode {
                format!(
                    "2. Live Edit Preview ({}×{})",
                    self.snapshot.target_width, self.snapshot.target_height
                )
            } else {
                format!(
                    "2. Scene #{:02} Inspector ({}×{})",
                    self.selected_shot_id, self.snapshot.target_width, self.snapshot.target_height
                )
            };
            ui.heading(title);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let toggle_label = if self.live_timeline_mode {
                    format!("Inspect Scene #{:02}", self.selected_shot_id)
                } else {
                    "Switch to Timeline Edit".to_string()
                };
                if ui.button(toggle_label).clicked() {
                    self.live_timeline_mode = !self.live_timeline_mode;
                }
            });
        });

        let avail = ui.available_size();
        let canvas_h = (avail.y - 68.0).max(120.0);
        let (outer_rect, _) =
            ui.allocate_exact_size(Vec2::new(avail.x, canvas_h), Sense::hover());

        // Compute stretched target aspect box inside outer_rect
        let target_aspect = (self.snapshot.target_width.max(1) as f32)
            / (self.snapshot.target_height.max(1) as f32);
        let mut box_w = outer_rect.width().min(outer_rect.height() * target_aspect);
        let mut box_h = box_w / target_aspect;
        if box_h > outer_rect.height() {
            box_h = outer_rect.height();
            box_w = box_h * target_aspect;
        }
        let preview_rect = Rect::from_center_size(outer_rect.center(), Vec2::new(box_w, box_h));

        let playhead_ms = (self.playhead_sec.max(0.0) * 1000.0).round() as u64;
        let tex = if self.live_timeline_mode {
            self.get_preview_texture(ctx, playhead_ms)
        } else {
            self.get_shot_texture(ctx, self.selected_shot_id, 1)
        };

        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        ui.painter().rect_filled(
            outer_rect,
            CornerRadius::same(4),
            Color32::from_rgb(12, 14, 20),
        );
        ui.painter()
            .image(tex.id(), preview_rect, uv, Color32::WHITE);
        ui.painter().rect_stroke(
            preview_rect,
            CornerRadius::same(2),
            Stroke::new(
                if self.playing { 2.0 } else { 1.0 },
                if self.playing {
                    Color32::from_rgb(76, 224, 179)
                } else {
                    Color32::from_rgb(75, 85, 120)
                },
            ),
            StrokeKind::Outside,
        );

        // Timecode overlay
        let max_dur = self
            .snapshot
            .timeline_duration_sec
            .max(self.snapshot.audio_duration_sec);
        let tc = format!(
            "{:.2}s / {:.2}s{}",
            self.playhead_sec,
            max_dur,
            if self.loop_enabled { " [LOOP]" } else { "" }
        );
        ui.painter().text(
            Pos2::new(preview_rect.left() + 8.0, preview_rect.bottom() - 8.0),
            egui::Align2::LEFT_BOTTOM,
            tc,
            FontId::monospace(12.0),
            Color32::from_rgb(76, 224, 179),
        );

        // Scrub slider
        let mut scrub = self.playhead_sec;
        let slider_max = max_dur.max(0.1);
        if ui
            .add(
                egui::Slider::new(&mut scrub, 0.0..=slider_max)
                    .show_value(false)
                    .trailing_fill(true),
            )
            .changed()
        {
            self.live_timeline_mode = true;
            self.seek_to(scrub);
        }

        // Transport controls
        ui.horizontal(|ui| {
            if ui.button("|<").clicked() {
                self.live_timeline_mode = true;
                self.seek_to(0.0);
            }
            if ui.button("-1f").clicked() {
                self.live_timeline_mode = true;
                self.seek_to(self.playhead_sec - 1.0 / 30.0);
            }
            let play_label = if self.playing {
                "⏸ Pause"
            } else {
                "▶ Play Edit"
            };
            if ui.button(RichText::new(play_label).strong()).clicked() {
                self.live_timeline_mode = true;
                self.toggle_play_pause();
            }
            if ui.button("+1f").clicked() {
                self.live_timeline_mode = true;
                self.seek_to(self.playhead_sec + 1.0 / 30.0);
            }

            if ui
                .checkbox(&mut self.click_track_enabled, "Click Track")
                .changed()
                && self.playing
            {
                self.restart_audio_companion();
            }

            if self.loop_enabled && ui.button("Clear Loop").clicked() {
                self.loop_enabled = false;
            }

            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui
                    .add_enabled(!busy, egui::Button::new("✂ Split Scene"))
                    .clicked()
                    && let Some(s) = self.snapshot.shots.get(self.selected_shot_id)
                {
                    let sid = self.selected_shot_id;
                    let mid = 0.5 * (s.start_sec + s.end_sec);
                    self.spawn_controller_action(ctx.clone(), true, move |ctrl| {
                        let _ = ctrl.split_shot(sid, mid);
                    });
                }
            });
        });
    }

    fn render_bass_beat_canvas(&mut self, ctx: &Context, ui: &mut Ui, busy: bool) {
        ui.horizontal(|ui| {
            ui.heading(
                "3. Bass-Instrument Beat Editor (Drag pins to adjust, Click to toggle, Right-click to delete, Double-click to add)",
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let mut sens = self.snapshot.sensitivity;
                let sens_resp = ui.add_enabled(
                    !busy,
                    egui::Slider::new(&mut sens, 0.25..=2.20).text("Threshold"),
                );
                if sens_resp.changed() {
                    self.snapshot.sensitivity = sens;
                }
                if sens_resp.drag_stopped() || (sens_resp.changed() && !sens_resp.dragged()) {
                    self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                        let _ = ctrl.set_sensitivity(sens);
                    });
                }

                let mut low_hz = self.snapshot.bass_low_hz;
                let mut high_hz = self.snapshot.bass_high_hz;
                let hi_resp = ui.add_enabled(
                    !busy,
                    DragValue::new(&mut high_hz)
                        .range(80.0..=500.0)
                        .speed(1.0)
                        .suffix(" Hz"),
                );
                ui.label("–");
                let lo_resp = ui.add_enabled(
                    !busy,
                    DragValue::new(&mut low_hz)
                        .range(20.0..=150.0)
                        .speed(1.0)
                        .suffix(" Hz"),
                );
                ui.label("Bass Band:");

                if lo_resp.changed() || hi_resp.changed() {
                    self.snapshot.bass_low_hz = low_hz;
                    self.snapshot.bass_high_hz = high_hz;
                }
                if lo_resp.drag_stopped()
                    || hi_resp.drag_stopped()
                    || ((lo_resp.changed() || hi_resp.changed())
                        && !lo_resp.dragged()
                        && !hi_resp.dragged())
                {
                    self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                        let _ = ctrl.set_bass_band(low_hz, high_hz);
                    });
                }

                egui::ComboBox::from_id_salt("sep_mode_combo")
                    .selected_text(self.snapshot.separation_mode.to_uppercase())
                    .width(78.0)
                    .show_ui(ui, |ui| {
                        for (lbl, m) in [
                            ("HPSS", SeparationMode::Hpss),
                            ("BAND", SeparationMode::Band),
                            ("DEMUCS", SeparationMode::Demucs),
                        ] {
                            if ui
                                .selectable_label(
                                    self.snapshot.separation_mode.eq_ignore_ascii_case(lbl),
                                    lbl,
                                )
                                .clicked()
                            {
                                self.snapshot.separation_mode = lbl.to_lowercase();
                                self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                                    let _ = ctrl.beat_editor.set_separation_mode(m);
                                    let _ = ctrl.resequence();
                                });
                            }
                        }
                    });

                egui::ComboBox::from_id_salt("beat_mode_combo")
                    .selected_text(self.snapshot.beat_mode.to_uppercase())
                    .width(82.0)
                    .show_ui(ui, |ui| {
                        for (lbl, m) in [
                            ("HYBRID", BeatMode::Hybrid),
                            ("GRID", BeatMode::Grid),
                            ("ONSETS", BeatMode::Onsets),
                        ] {
                            if ui
                                .selectable_label(
                                    self.snapshot.beat_mode.eq_ignore_ascii_case(lbl),
                                    lbl,
                                )
                                .clicked()
                            {
                                self.snapshot.beat_mode = lbl.to_lowercase();
                                self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                                    let _ = ctrl.beat_editor.set_beat_mode(m);
                                    let _ = ctrl.resequence();
                                });
                            }
                        }
                    });
            });
        });

        let avail = ui.available_size();
        let (rect, canvas_resp) =
            ui.allocate_exact_size(Vec2::new(avail.x, avail.y.max(90.0)), Sense::click_and_drag());

        let spec_tex = self.get_spectrogram_texture(ctx);
        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        ui.painter().image(spec_tex.id(), rect, uv, Color32::WHITE);

        let dur = self.snapshot.audio_duration_sec.max(0.001);

        // Draw Loop highlight region if active
        if self.loop_enabled && self.snapshot.audio_duration_sec > 0.0 {
            let x0 = rect.left() + ((self.loop_start_sec / dur) as f32) * rect.width();
            let x1 = rect.left() + ((self.loop_end_sec / dur) as f32) * rect.width();
            let loop_rect =
                Rect::from_min_max(Pos2::new(x0, rect.top()), Pos2::new(x1, rect.bottom()));
            ui.painter().rect_filled(
                loop_rect,
                CornerRadius::ZERO,
                Color32::from_rgba_unmultiplied(76, 224, 179, 45),
            );
        }

        // Interactive Beat Pins using dedicated per-pin `ui.interact` widgets so fast drags never lose focus!
        let beats = self.snapshot.beats.clone();
        let pointer_pos = ui.input(|i| i.pointer.interact_pos());
        let mut clicked_pin_idx: Option<usize> = None;
        let mut right_clicked_pin_idx: Option<usize> = None;
        let mut dragged_pin: Option<(usize, f64)> = None;
        let mut drag_finished_pin: Option<(usize, f64)> = None;
        let mut any_pin_interacted = false;

        for beat in &beats {
            let bx = rect.left() + ((beat.time_sec / dur) as f32) * rect.width();
            let is_being_dragged = self.dragging_beat_idx == Some(beat.index);
            let pin_color = if is_being_dragged {
                Color32::from_rgb(255, 230, 100)
            } else if beat.is_ghost {
                Color32::from_rgba_unmultiplied(136, 146, 176, 110)
            } else if !beat.enabled {
                Color32::from_rgb(120, 130, 155)
            } else if beat.is_manual {
                Color32::from_rgb(255, 209, 102)
            } else if beat.is_downbeat {
                Color32::from_rgb(255, 92, 92)
            } else {
                Color32::from_rgb(56, 239, 125)
            };

            let hit_rect =
                Rect::from_center_size(Pos2::new(bx, rect.center().y), Vec2::new(18.0, rect.height()));
            let pin_id = ui.make_persistent_id(("beat_pin_widget", beat.index, beat.is_ghost));
            let pin_resp = ui.interact(hit_rect, pin_id, Sense::click_and_drag());

            ui.painter().line_segment(
                [Pos2::new(bx, rect.top()), Pos2::new(bx, rect.bottom())],
                Stroke::new(
                    if is_being_dragged {
                        3.0
                    } else if beat.enabled {
                        2.0
                    } else {
                        1.0
                    },
                    pin_color,
                ),
            );

            let handle_rect =
                Rect::from_center_size(Pos2::new(bx, rect.top() + 11.0), Vec2::new(18.0, 22.0));
            ui.painter()
                .rect_filled(handle_rect, CornerRadius::same(4), pin_color);
            let sym = if beat.is_ghost {
                "+"
            } else if beat.enabled {
                "▼"
            } else {
                "×"
            };
            ui.painter().text(
                handle_rect.center(),
                egui::Align2::CENTER_CENTER,
                sym,
                FontId::proportional(10.5),
                Color32::from_rgb(14, 16, 23),
            );

            if pin_resp.hovered() || pin_resp.dragged() || is_being_dragged {
                ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
                let tooltip_rect =
                    Rect::from_min_size(Pos2::new(bx + 8.0, rect.top() + 24.0), Vec2::new(72.0, 20.0));
                ui.painter().rect_filled(
                    tooltip_rect,
                    CornerRadius::same(4),
                    Color32::from_rgba_unmultiplied(12, 15, 22, 235),
                );
                ui.painter().text(
                    tooltip_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    format!("{:.2}s{}", beat.time_sec, if beat.is_manual { " ★" } else { "" }),
                    FontId::monospace(11.0),
                    Color32::from_rgb(255, 230, 100),
                );
            }

            if !beat.is_ghost && (pin_resp.drag_started() || pin_resp.dragged()) {
                any_pin_interacted = true;
                self.dragging_beat_idx = Some(beat.index);
                if let Some(pos) = pointer_pos {
                    let rel = ((pos.x - rect.left()) / rect.width().max(1.0)).clamp(0.01, 0.99) as f64;
                    dragged_pin = Some((beat.index, rel * dur));
                }
            }
            if !beat.is_ghost && pin_resp.drag_stopped() {
                any_pin_interacted = true;
                let final_sec = if let Some(pos) = pointer_pos {
                    let rel = ((pos.x - rect.left()) / rect.width().max(1.0)).clamp(0.01, 0.99) as f64;
                    rel * dur
                } else {
                    beat.time_sec
                };
                drag_finished_pin = Some((beat.index, final_sec));
            } else if pin_resp.secondary_clicked() && !beat.is_ghost {
                any_pin_interacted = true;
                right_clicked_pin_idx = Some(beat.index);
            } else if pin_resp.clicked() {
                any_pin_interacted = true;
                clicked_pin_idx = Some(beat.index);
            }
        }

        // Apply live drag or click updates
        if let Some((drag_idx, new_sec)) = dragged_pin {
            if let Some(b) = self
                .snapshot
                .beats
                .iter_mut()
                .find(|b| !b.is_ghost && b.index == drag_idx)
            {
                b.time_sec = new_sec;
                b.is_manual = true;
                b.enabled = true;
            }
            self.snapshot.status_message =
                format!("Dragging Beat #{drag_idx} → {new_sec:.3}s (release to re-align clips)");
        }

        if let Some((drag_idx, final_sec)) = drag_finished_pin {
            self.dragging_beat_idx = None;
            if let Some(b) = self
                .snapshot
                .beats
                .iter_mut()
                .find(|b| !b.is_ghost && b.index == drag_idx)
            {
                b.time_sec = final_sec;
                b.is_manual = true;
                b.enabled = true;
            }
            self.snapshot.status_message =
                format!("Moved Beat #{drag_idx} to {final_sec:.3}s — re-aligning timeline...");
            self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                let _ = ctrl.move_beat(drag_idx, final_sec);
            });
        } else if let Some(idx) = right_clicked_pin_idx {
            self.snapshot.beats.retain(|b| b.index != idx);
            self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                let _ = ctrl.remove_beat(idx);
            });
        } else if let Some(idx) = clicked_pin_idx {
            if let Some(b) = self.snapshot.beats.iter_mut().find(|b| b.index == idx) {
                b.enabled = !b.enabled;
                b.is_manual = true;
                b.is_ghost = false;
            }
            self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                let _ = ctrl.toggle_beat(idx);
            });
        } else if !any_pin_interacted
            && canvas_resp.double_clicked()
            && let Some(pos) = pointer_pos
            && self.snapshot.audio_duration_sec > 0.0
        {
            let rel = ((pos.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0) as f64;
            let t = rel * dur;
            self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                let _ = ctrl.add_beat_at(t);
            });
        } else if !any_pin_interacted
            && canvas_resp.clicked()
            && let Some(pos) = pointer_pos
            && self.snapshot.audio_duration_sec > 0.0
        {
            let rel = ((pos.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0) as f64;
            self.seek_to(rel * dur);
        }

        // Synchronized red Playhead line
        if self.snapshot.audio_duration_sec > 0.0 {
            let px = rect.left() + ((self.playhead_sec / dur) as f32) * rect.width();
            ui.painter().line_segment(
                [Pos2::new(px, rect.top()), Pos2::new(px, rect.bottom())],
                Stroke::new(2.5, Color32::from_rgb(255, 51, 102)),
            );
        }
    }

    fn render_match_timeline(&mut self, ctx: &Context, ui: &mut Ui, busy: bool) {
        ui.horizontal(|ui| {
            ui.heading(format!(
                "4. Sequenced Match-Cut Timeline ({} beat slots — No clip reuse, Hard Cut default)",
                self.snapshot.slots.len()
            ));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(
                    RichText::new(
                        "Drag any scene from Cut-Up Scenes onto a slot to swap; click any % CUT badge to loop ±1.2s",
                    )
                    .color(Color32::from_rgb(140, 150, 178))
                    .size(11.5),
                );
            });
        });
        ui.add_space(4.0);

        let slots = self.snapshot.slots.clone();
        if slots.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label(
                    RichText::new("Import both a video clip and an audio track to sequence match cuts.")
                        .color(Color32::from_rgb(140, 150, 178)),
                );
            });
            return;
        }

        let pointer_pos = ui.input(|i| i.pointer.interact_pos());
        let pointer_released = ui.input(|i| i.pointer.any_released());
        let mut dropped_on_slot: Option<(usize, usize)> = None;

        ScrollArea::horizontal()
            .id_salt("timeline_slots_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing = Vec2::new(8.0, 6.0);
                    let card_h = (ui.available_height() - 6.0).max(110.0);

                    for slot in &slots {
                        // Cut boundary similarity badge between slot i-1 and slot i
                        if slot.slot_index > 0 {
                            let sim_pct = (slot.boundary_similarity * 100.0).round() as u32;
                            let badge_bg = if slot.boundary_similarity >= 0.75 {
                                Color32::from_rgb(27, 67, 50)
                            } else {
                                Color32::from_rgb(61, 44, 46)
                            };
                            let (b_rect, b_resp) =
                                ui.allocate_exact_size(Vec2::new(48.0, card_h), Sense::click());
                            ui.painter()
                                .rect_filled(b_rect, CornerRadius::same(5), badge_bg);
                            ui.painter().text(
                                Pos2::new(b_rect.center().x, b_rect.center().y - 14.0),
                                egui::Align2::CENTER_CENTER,
                                format!("{sim_pct}%"),
                                FontId::proportional(13.0),
                                Color32::WHITE,
                            );
                            ui.painter().text(
                                b_rect.center(),
                                egui::Align2::CENTER_CENTER,
                                slot.transition_id.to_uppercase(),
                                FontId::proportional(10.0),
                                Color32::from_rgb(168, 218, 220),
                            );
                            ui.painter().text(
                                Pos2::new(b_rect.center().x, b_rect.center().y + 14.0),
                                egui::Align2::CENTER_CENTER,
                                "🔁 LOOP",
                                FontId::proportional(9.5),
                                Color32::from_rgb(255, 209, 102),
                            );
                            if b_resp.clicked() {
                                self.loop_around_cut(slot.slot_index, 1.2);
                            }
                        }

                        // Slot card
                        let is_playhead_here = self.playhead_sec >= slot.dst_start_sec
                            && self.playhead_sec < slot.dst_end_sec;
                        let border_col = if is_playhead_here {
                            Color32::from_rgb(255, 51, 102)
                        } else if slot.locked {
                            Color32::from_rgb(255, 209, 102)
                        } else {
                            Color32::from_rgb(59, 68, 100)
                        };

                        let frame_resp = egui::Frame::NONE
                            .fill(Color32::from_rgb(34, 40, 58))
                            .stroke(Stroke::new(
                                if is_playhead_here || slot.locked {
                                    2.0
                                } else {
                                    1.0
                                },
                                border_col,
                            ))
                            .corner_radius(CornerRadius::same(6))
                            .inner_margin(6.0)
                            .show(ui, |ui| {
                                ui.set_width(185.0);
                                ui.vertical(|ui| {
                                    let thumb_h = (card_h - 56.0).max(54.0);
                                    let (t_rect, t_resp) = ui.allocate_exact_size(
                                        Vec2::new(173.0, thumb_h),
                                        Sense::click(),
                                    );
                                    let tex = self.get_shot_texture(ctx, slot.shot_id, 1);
                                    let uv =
                                        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
                                    ui.painter().image(tex.id(), t_rect, uv, Color32::WHITE);

                                    if t_resp.clicked() {
                                        self.selected_slot_idx = slot.slot_index;
                                        self.selected_shot_id = slot.shot_id;
                                        self.seek_to(slot.dst_start_sec);
                                    }

                                    ui.horizontal(|ui| {
                                        ui.label(
                                            RichText::new(format!(
                                                "#{}: Scene #{:02}",
                                                slot.slot_index, slot.shot_id
                                            ))
                                            .strong()
                                            .size(11.5),
                                        );
                                        ui.with_layout(
                                            Layout::right_to_left(Align::Center),
                                            |ui| {
                                                ui.add_enabled_ui(!busy, |ui| {
                                                    ui.menu_button("Swap ▾", |ui| {
                                                        ui.set_min_width(170.0);
                                                        let alts = self
                                                            .controller
                                                            .try_lock()
                                                            .map(|c| {
                                                                c.rank_slot_alternatives(
                                                                    slot.slot_index,
                                                                    5,
                                                                )
                                                            })
                                                            .unwrap_or_default();
                                                        if alts.is_empty() {
                                                            ui.label("No unused enabled scenes");
                                                        } else {
                                                            for alt in alts {
                                                                if ui
                                                                    .button(format!(
                                                                        "Scene #{:02} ({:.0}% match)",
                                                                        alt.shot_id,
                                                                        alt.similarity * 100.0
                                                                    ))
                                                                    .clicked()
                                                                {
                                                                    let s_idx = slot.slot_index;
                                                                    let a_id = alt.shot_id;
                                                                    self.spawn_controller_action(
                                                                        ctx.clone(),
                                                                        false,
                                                                        move |ctrl| {
                                                                            let _ = ctrl
                                                                                .swap_slot_shot(
                                                                                    s_idx, a_id,
                                                                                );
                                                                        },
                                                                    );
                                                                    ui.close();
                                                                }
                                                            }
                                                        }
                                                    });
                                                });
                                            },
                                        );
                                    });

                                    ui.horizontal(|ui| {
                                        ui.label(
                                            RichText::new(format!(
                                                "{:.2}s–{:.2}s (in {:.2}s)",
                                                slot.dst_start_sec,
                                                slot.dst_end_sec,
                                                slot.src_in_sec
                                            ))
                                            .color(Color32::from_rgb(143, 155, 184))
                                            .size(10.0),
                                        );
                                        ui.with_layout(
                                            Layout::right_to_left(Align::Center),
                                            |ui| {
                                                if ui
                                                    .add_enabled(
                                                        !busy,
                                                        egui::Button::new("+0.1s").small(),
                                                    )
                                                    .clicked()
                                                {
                                                    let s_idx = slot.slot_index;
                                                    self.spawn_controller_action(
                                                        ctx.clone(),
                                                        false,
                                                        move |ctrl| {
                                                            let _ = ctrl.slip_slot_in_point(
                                                                s_idx, 0.10,
                                                            );
                                                        },
                                                    );
                                                }
                                                if ui
                                                    .add_enabled(
                                                        !busy,
                                                        egui::Button::new("-0.1s").small(),
                                                    )
                                                    .clicked()
                                                {
                                                    let s_idx = slot.slot_index;
                                                    self.spawn_controller_action(
                                                        ctx.clone(),
                                                        false,
                                                        move |ctrl| {
                                                            let _ = ctrl.slip_slot_in_point(
                                                                s_idx, -0.10,
                                                            );
                                                        },
                                                    );
                                                }
                                            },
                                        );
                                    });
                                });
                            });

                        // Check if a dragged scene from Cut-Up Scenes is hovering over or dropped onto this slot
                        if let Some(drag_shot_id) = self.dragging_shot_id
                            && let Some(pos) = pointer_pos
                            && frame_resp.response.rect.contains(pos)
                        {
                            let drop_rect = frame_resp.response.rect;
                            ui.painter().rect_filled(
                                drop_rect,
                                CornerRadius::same(6),
                                Color32::from_rgba_unmultiplied(76, 224, 179, 45),
                            );
                            ui.painter().rect_stroke(
                                drop_rect,
                                CornerRadius::same(6),
                                Stroke::new(3.0, Color32::from_rgb(76, 224, 179)),
                                StrokeKind::Inside,
                            );
                            ui.painter().text(
                                drop_rect.center(),
                                egui::Align2::CENTER_CENTER,
                                format!("Drop Scene #{drag_shot_id:02} → Slot #{}", slot.slot_index),
                                FontId::proportional(13.0),
                                Color32::WHITE,
                            );
                            if pointer_released {
                                dropped_on_slot = Some((slot.slot_index, drag_shot_id));
                            }
                        }
                    }
                });
            });

        if let Some((slot_idx, new_shot_id)) = dropped_on_slot {
            self.dragging_shot_id = None;
            // Optimistic UI update in 0ms
            if let Some(s) = self.snapshot.slots.get_mut(slot_idx) {
                s.shot_id = new_shot_id;
                s.locked = true;
            }
            self.spawn_controller_action(ctx.clone(), false, move |ctrl| {
                let _ = ctrl.swap_slot_shot(slot_idx, new_shot_id);
            });
        }
    }
}

impl Drop for AutoEditorEguiApp {
    fn drop(&mut self) {
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
            ctx.set_visuals(egui::Visuals::dark());
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

        // Poll background task progress (lock-free atomic read)
        let (raw_busy, pct, progress_msg) = get_gui_progress();
        let busy = raw_busy || self.pending_resequence;
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

        // 1. Top Toolbar
        egui::Panel::top("top_toolbar")
            .exact_size(46.0)
            .show(ui, |ui| {
                ui.add_space(4.0);
                self.render_top_toolbar(&ctx, ui, busy);
            });

        // 2. Bottom Status & Progress Bar
        egui::Panel::bottom("bottom_status_bar")
            .exact_size(34.0)
            .show(ui, |ui| {
                ui.horizontal_centered(|ui| {
                    let dot_color = if busy {
                        Color32::from_rgb(255, 209, 102)
                    } else {
                        Color32::from_rgb(76, 224, 179)
                    };
                    let (dot_rect, _) =
                        ui.allocate_exact_size(Vec2::new(10.0, 10.0), Sense::hover());
                    ui.painter().circle_filled(dot_rect.center(), 4.5, dot_color);

                    let msg = if busy && !progress_msg.is_empty() {
                        &progress_msg
                    } else {
                        &self.snapshot.status_message
                    };
                    ui.label(
                        RichText::new(msg)
                            .color(if busy {
                                Color32::from_rgb(255, 209, 102)
                            } else {
                                Color32::from_rgb(201, 210, 232)
                            })
                            .strong(),
                    );

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!(
                                "Mean Match Similarity: {:.1}%  |  Target: {} (Stretch)",
                                self.snapshot.mean_similarity * 100.0,
                                self.snapshot.resolution
                            ))
                            .color(Color32::from_rgb(107, 230, 193))
                            .strong(),
                        );
                        ui.separator();
                        ui.add_sized(
                            [220.0, 18.0],
                            ProgressBar::new((pct as f32) / 100.0)
                                .show_percentage()
                                .animate(busy),
                        );
                    });
                });
            });

        let avail_h = ui.available_height();

        // 3. Sequenced Match-Cut Timeline (bottom ~26% of workspace)
        egui::Panel::bottom("match_timeline_panel")
            .resizable(true)
            .default_size((avail_h * 0.26).max(165.0))
            .min_size(135.0)
            .show(ui, |ui| {
                self.render_match_timeline(&ctx, ui, busy);
            });

        // 4. Interactive Bass Beat & Frequency Editor (middle ~26% of workspace)
        egui::Panel::bottom("bass_beat_panel")
            .resizable(true)
            .default_size((avail_h * 0.26).max(165.0))
            .min_size(130.0)
            .show(ui, |ui| {
                self.render_bass_beat_canvas(&ctx, ui, busy);
            });

        // 5. Right Panel: Stretched Output Preview Monitor (~42% width)
        let avail_w = ui.available_width();
        egui::Panel::right("preview_monitor_panel")
            .resizable(true)
            .default_size((avail_w * 0.42).max(380.0))
            .min_size(300.0)
            .show(ui, |ui| {
                self.render_preview_monitor(&ctx, ui, busy);
            });

        // 6. Central Panel: Cut-Up Scenes (Shot Pool Grid)
        egui::CentralPanel::default().show(ui, |ui| {
            self.render_shot_pool(&ctx, ui, busy);
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

        // 7. In-app zero-stall File Picker Modal + OS drag-and-drop support
        self.render_file_dialog_modal(&ctx);

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
