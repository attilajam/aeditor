use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{OnceLock, RwLock};

use ae_core::{Rational, RgbaFrame, Timeline};
use ae_transitions::{TransitionCtx, build_transition};

/// Thread-safe progress state polled by the Qt6 UI thread while background import/export tasks run.
#[derive(Default)]
pub struct GuiProgressTracker {
    pub busy: AtomicBool,
    pub percent: AtomicU32,
    pub message: RwLock<String>,
}

static GLOBAL_PROGRESS: OnceLock<GuiProgressTracker> = OnceLock::new();

pub fn global_progress() -> &'static GuiProgressTracker {
    GLOBAL_PROGRESS.get_or_init(GuiProgressTracker::default)
}

pub fn set_gui_progress(busy: bool, percent: u32, message: impl Into<String>) {
    let p = global_progress();
    p.busy.store(busy, Ordering::SeqCst);
    p.percent.store(percent.min(100), Ordering::SeqCst);
    if let Ok(mut guard) = p.message.write() {
        *guard = message.into();
    }
}

pub fn get_gui_progress() -> (bool, u32, String) {
    let p = global_progress();
    let busy = p.busy.load(Ordering::SeqCst);
    let pct = p.percent.load(Ordering::SeqCst);
    let msg = p
        .message
        .read()
        .map(|g| g.clone())
        .unwrap_or_default();
    (busy, pct, msg)
}

/// In-memory RGBA image cache backing Qt6's `QQuickImageProvider`:
/// - `image://shots/<shot_id>/<0|1|2>` (start, midpoint, and end thumbnails of each cut-up scene)
/// - `image://spectrogram/<revision>` (live bass-band spectrogram with ODF curve and beat pins)
/// - `image://preview/<playhead_ms>/<revision>` (0ms-wait virtual timeline frame resolver from proxy reel)
#[derive(Default)]
pub struct GuiMediaCache {
    pub shot_thumbnails: HashMap<(usize, u8), RgbaFrame>,
    pub spectrogram: Option<RgbaFrame>,
    pub spectrogram_revision: u64,
    pub proxy_reel: Vec<RgbaFrame>,
    pub active_timeline: Option<Timeline>,
    pub preview_revision: u64,
}

static GLOBAL_MEDIA_CACHE: OnceLock<RwLock<GuiMediaCache>> = OnceLock::new();

pub fn global_media_cache() -> &'static RwLock<GuiMediaCache> {
    GLOBAL_MEDIA_CACHE.get_or_init(|| RwLock::new(GuiMediaCache::default()))
}

impl GuiMediaCache {
    pub fn store_shot_thumbnails(&mut self, shot_id: usize, thumbs: [RgbaFrame; 3]) {
        let [t0, t1, t2] = thumbs;
        self.shot_thumbnails.insert((shot_id, 0), t0);
        self.shot_thumbnails.insert((shot_id, 1), t1);
        self.shot_thumbnails.insert((shot_id, 2), t2);
    }

    pub fn get_shot_thumbnail(&self, shot_id: usize, pos: u8) -> RgbaFrame {
        self.shot_thumbnails
            .get(&(shot_id, pos.min(2)))
            .or_else(|| self.shot_thumbnails.get(&(shot_id, 1)))
            .cloned()
            .unwrap_or_else(|| RgbaFrame::solid(320, 180, [32, 36, 46, 255], Rational::ZERO))
    }

    pub fn update_spectrogram(&mut self, frame: RgbaFrame) -> u64 {
        self.spectrogram = Some(frame);
        self.spectrogram_revision = self.spectrogram_revision.wrapping_add(1);
        self.spectrogram_revision
    }

    pub fn get_spectrogram(&self) -> RgbaFrame {
        self.spectrogram
            .clone()
            .unwrap_or_else(|| RgbaFrame::solid(640, 160, [20, 24, 32, 255], Rational::ZERO))
    }

    pub fn set_proxy_reel(&mut self, frames: Vec<RgbaFrame>) {
        self.proxy_reel = frames;
        self.preview_revision = self.preview_revision.wrapping_add(1);
    }

    pub fn set_active_timeline(&mut self, timeline: Option<Timeline>) -> u64 {
        self.active_timeline = timeline;
        self.preview_revision = self.preview_revision.wrapping_add(1);
        self.preview_revision
    }

    /// Resolve a source timestamp `src_pts` to the closest proxy frame in RAM (or shot thumbnail fallback).
    fn sample_proxy_at(&self, shot_id: usize, src_pts: Rational) -> RgbaFrame {
        if !self.proxy_reel.is_empty() {
            // Binary search proxy_reel by pts
            let idx = match self
                .proxy_reel
                .binary_search_by(|f| f.pts.cmp(&src_pts))
            {
                Ok(i) => i,
                Err(i) => {
                    if i == 0 {
                        0
                    } else if i >= self.proxy_reel.len() {
                        self.proxy_reel.len() - 1
                    } else {
                        let d_prev = (src_pts - self.proxy_reel[i - 1].pts).abs();
                        let d_curr = (self.proxy_reel[i].pts - src_pts).abs();
                        if d_prev <= d_curr { i - 1 } else { i }
                    }
                }
            };
            return self.proxy_reel[idx].clone();
        }
        self.get_shot_thumbnail(shot_id, 1)
    }

    /// Phase P1: Instant `0 ms` virtual timeline frame resolver for `image://preview/<playhead_ms>`.
    /// Maps playhead timestamp `playhead_ms` -> active `TimelineSlot` -> source timestamp `src_pts`
    /// -> proxy frame in RAM, applying real-time CPU transitions if inside a transition window.
    pub fn resolve_timeline_preview_frame(&self, playhead_ms: u64) -> (RgbaFrame, i32, i32) {
        let Some(ref tl) = self.active_timeline else {
            return (
                self.get_shot_thumbnail(0, 1),
                -1,
                0,
            );
        };
        if tl.slots.is_empty() {
            return (self.get_shot_thumbnail(0, 1), -1, 0);
        }

        let t = Rational::from_millis(playhead_ms as i64);
        let slot_idx = tl
            .slots
            .iter()
            .position(|s| t >= s.dst_start && t < s.dst_end)
            .unwrap_or(tl.slots.len() - 1);

        let slot = &tl.slots[slot_idx];
        let elapsed_dst = (t - slot.dst_start).max(Rational::ZERO);
        let scaled_elapsed =
            Rational::from_secs_f64(elapsed_dst.as_secs_f64() * slot.speed.max(0.1));
        let src_pts = (slot.src_in + scaled_elapsed).min(slot.src_out);

        let to_frame = self.sample_proxy_at(slot.shot_id, src_pts);

        // If inside a non-cut transition window at the beginning of this slot, composite live!
        if slot_idx > 0
            && slot.transition_in.id != "cut"
            && slot.transition_in.duration > Rational::ZERO
            && elapsed_dst < slot.transition_in.duration
            && let Ok(tr) = build_transition(&slot.transition_in.id, &slot.transition_in.params)
        {
            let prev_slot = &tl.slots[slot_idx - 1];
            let prev_pts = prev_slot.src_out + elapsed_dst;
            let from_frame = self.sample_proxy_at(prev_slot.shot_id, prev_pts);
            if from_frame.width == to_frame.width && from_frame.height == to_frame.height {
                let progress = (elapsed_dst.as_secs_f32()
                    / slot.transition_in.duration.as_secs_f32().max(1e-4))
                .clamp(0.0, 1.0);
                let ctx = TransitionCtx {
                    width: to_frame.width,
                    height: to_frame.height,
                    fps: tl.fps,
                    duration: slot.transition_in.duration,
                    easing: slot.transition_in.easing.clone(),
                    beat_strength: slot.beat_strength,
                };
                let mut blended = RgbaFrame::new(to_frame.width, to_frame.height, t);
                tr.render(
                    &ctx,
                    from_frame.as_ref(),
                    to_frame.as_ref(),
                    progress,
                    blended.as_mut(),
                );
                return (blended, slot_idx as i32, slot.shot_id as i32);
            }
        }

        (to_frame, slot_idx as i32, slot.shot_id as i32)
    }
}
