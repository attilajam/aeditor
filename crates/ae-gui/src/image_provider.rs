use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{OnceLock, RwLock};

use ae_core::{Rational, RgbaFrame, Timeline, TimelineSlot};
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
    let msg = p.message.read().map(|g| g.clone()).unwrap_or_default();
    (busy, pct, msg)
}

/// In-memory RGBA image cache backing Qt6's `QQuickImageProvider`:
/// - `image://shots/<shot_id>/<0|1|2>` (start, midpoint, and end thumbnails of each cut-up scene)
/// - `image://spectrogram/<revision>` (live bass-band spectrogram with ODF curve and beat pins)
/// - `image://preview/<playhead_ms>/<revision>` (0ms-wait virtual timeline frame resolver from proxy reel)
#[derive(Default)]
pub struct GuiMediaCache {
    pub shot_thumbnails: HashMap<(usize, u8), RgbaFrame>,
    pub character_portraits: HashMap<u64, RgbaFrame>,
    pub character_revision: u64,
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

    /// Resolve a source timestamp within the selected slot, or use its thumbnail fallback.
    fn sample_proxy_at(&self, slot: &TimelineSlot, src_pts: Rational) -> RgbaFrame {
        self.sample_source_range(slot.shot_id, slot.src_in, slot.src_out, src_pts)
    }

    pub fn source_preview_frame(
        &self,
        shot_id: usize,
        start: Rational,
        end: Rational,
        position: f32,
    ) -> RgbaFrame {
        let time = start
            + Rational::from_secs_f64(
                (end - start).as_secs_f64() * position.clamp(0.0, 1.0) as f64,
            );
        self.sample_source_range(shot_id, start, end, time)
    }

    fn sample_source_range(
        &self,
        shot_id: usize,
        start_time: Rational,
        end_time: Rational,
        src_pts: Rational,
    ) -> RgbaFrame {
        // Proxy PTS can precede the actual sampled source frame by half a sample interval.
        let half_sample = self
            .proxy_reel
            .windows(2)
            .next()
            .map(|pair| (pair[1].pts - pair[0].pts) / 2)
            .unwrap_or(Rational::ZERO);
        let start = self
            .proxy_reel
            .partition_point(|frame| frame.pts < start_time);
        let end = self
            .proxy_reel
            .partition_point(|frame| frame.pts < end_time - half_sample);
        if let Some(frames) = self
            .proxy_reel
            .get(start..end)
            .filter(|frames| !frames.is_empty())
        {
            let idx = match frames.binary_search_by(|f| f.pts.cmp(&src_pts)) {
                Ok(i) => i,
                Err(0) => 0,
                Err(i) if i >= frames.len() => frames.len() - 1,
                Err(i) => {
                    let d_prev = (src_pts - frames[i - 1].pts).abs();
                    let d_curr = (frames[i].pts - src_pts).abs();
                    if d_prev <= d_curr { i - 1 } else { i }
                }
            };
            return frames[idx].clone();
        }
        self.get_shot_thumbnail(shot_id, 1)
    }

    /// Phase P1: Instant `0 ms` virtual timeline frame resolver for `image://preview/<playhead_ms>`.
    /// Maps playhead timestamp `playhead_ms` -> active `TimelineSlot` -> source timestamp `src_pts`
    /// -> proxy frame in RAM, applying real-time CPU transitions if inside a transition window.
    pub fn resolve_timeline_preview_frame(&self, playhead_ms: u64) -> (RgbaFrame, i32, i32) {
        let Some(ref tl) = self.active_timeline else {
            return (self.get_shot_thumbnail(0, 1), -1, 0);
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

        let to_frame = self.sample_proxy_at(slot, src_pts);

        // If inside a non-cut transition window at the beginning of this slot, composite live!
        if slot_idx > 0
            && slot.transition_in.id != "cut"
            && slot.transition_in.duration > Rational::ZERO
            && elapsed_dst < slot.transition_in.duration
            && let Ok(tr) = build_transition(&slot.transition_in.id, &slot.transition_in.params)
        {
            let prev_slot = &tl.slots[slot_idx - 1];
            let from_frame = self.sample_proxy_at(prev_slot, prev_slot.src_out);
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

#[cfg(test)]
mod tests {
    use super::*;
    use ae_core::{Resolution, TimelineSlot, TransitionSpec};

    #[test]
    fn source_scrubbing_covers_the_shot_without_crossing_its_boundaries() {
        let mut cache = GuiMediaCache::default();
        cache.proxy_reel = (0..30)
            .map(|i| RgbaFrame::solid(2, 2, [i as u8, 0, 0, 255], Rational::new(i, 10)))
            .collect();
        let start = Rational::ONE;
        let end = Rational::from_secs(2);
        for (position, expected) in [(0.0, 10), (0.5, 15), (1.0, 19), (-1.0, 10), (2.0, 19)] {
            assert_eq!(
                cache
                    .source_preview_frame(0, start, end, position)
                    .pixel(0, 0)[0],
                expected
            );
        }
    }

    #[test]
    fn test_preview_stays_in_the_selected_source_range_until_transition() {
        let fps = Rational::from_secs(30);
        let mut cache = GuiMediaCache::default();
        cache.proxy_reel = (0..90)
            .map(|i| {
                let color = if i < 12 {
                    [220, 10, 10, 255]
                } else if i < 42 {
                    [10, 220, 10, 255]
                } else {
                    [10, 10, 220, 255]
                };
                RgbaFrame::solid(2, 2, color, Rational::from_frame_index(i, fps))
            })
            .collect();
        cache.active_timeline = Some(Timeline {
            version: 1,
            source_video: String::new(),
            source_audio: String::new(),
            resolution: Resolution::new(2, 2),
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
                    src_out: Rational::from_millis(400),
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
                    src_in: Rational::from_millis(1400),
                    src_out: Rational::from_millis(2400),
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
        });
        for ms in [0, 400, 800, 999, 1000, 1100, 1199, 1500] {
            let (frame, _, _) = cache.resolve_timeline_preview_frame(ms);
            let pixel = frame.pixel(0, 0);
            assert!(
                pixel[1] < 70,
                "unselected green scene leaked at {ms}ms: {pixel:?}"
            );
            if ms < 1000 {
                assert!(pixel[0] > 150);
            }
            if ms >= 1200 {
                assert!(pixel[2] > 150);
            }
        }

        // The fps filter may label a frame just before the source cut with the preceding sample PTS.
        cache.proxy_reel = (0..30)
            .map(|i| {
                let color = if i < 23 {
                    [220, 10, 10, 255]
                } else {
                    [10, 220, 10, 255]
                };
                RgbaFrame::solid(2, 2, color, Rational::new(i, 10))
            })
            .collect();
        let timeline = cache.active_timeline.as_mut().unwrap();
        timeline.slots.truncate(1);
        timeline.slots[0].src_out = Rational::new(70, 30);
        timeline.slots[0].dst_end = Rational::from_secs(3);
        for ms in [2200, 2300, 2500, 2999] {
            let frame = cache.resolve_timeline_preview_frame(ms).0;
            assert!(
                frame.pixel(0, 0)[0] > 150,
                "sampled proxy crossed the source end at {ms}ms"
            );
        }
    }
}
