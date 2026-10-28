use std::path::Path;

use ae_core::{AeError, Rational, Resolution, Result, RgbaFrame, Shot, ShotBoundaryKind};
use ae_media::MediaBackend;
use ae_shots::{ShotDetectorConfig, detect_shots};
use serde::{Deserialize, Serialize};

use crate::image_provider::{global_media_cache, set_gui_progress};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShotViewItem {
    pub id: usize,
    pub start_sec: f64,
    pub end_sec: f64,
    pub usable_start_sec: f64,
    pub usable_end_sec: f64,
    pub duration_sec: f64,
    pub enabled: bool,
    pub locked: bool,
    pub boundary_kind: String,
    pub confidence: f32,
    pub used_in_slot: i32,
    pub thumb_start_url: String,
    pub thumb_mid_url: String,
    pub thumb_end_url: String,
}

#[derive(Clone, Debug, Default)]
pub struct ShotPoolModel {
    pub shots: Vec<Shot>,
    pub thumb_res: Resolution,
}

impl ShotPoolModel {
    pub fn new() -> Self {
        Self {
            shots: Vec::new(),
            thumb_res: Resolution::new(320, 180), // HiDPI 5K-ready thumbnail resolution
        }
    }

    /// Detect scenes in `video_path` and decode 3 representative thumbnails (start, mid, end)
    /// per scene into `GuiMediaCache` for instant display and hover-scrubbing in QML.
    pub fn load_from_video(
        &mut self,
        backend: &dyn MediaBackend,
        video_path: &Path,
        config: &ShotDetectorConfig,
    ) -> Result<()> {
        let detected = detect_shots(backend, video_path, config)?;
        self.shots = detected;
        self.populate_thumbnails(backend, video_path)?;
        Ok(())
    }

    pub fn populate_thumbnails(
        &self,
        backend: &dyn MediaBackend,
        video_path: &Path,
    ) -> Result<()> {
        let fps = Rational::from_secs(10);
        let one_frame_dur = Rational::new(1, 10);

        let total = self.shots.len().max(1);
        for (idx, shot) in self.shots.iter().enumerate() {
            let pct = 30 + ((idx * 40) / total) as u32;
            set_gui_progress(
                true,
                pct,
                format!("Generating scene thumbnails ({}/{})...", idx + 1, total),
            );
            let mid = Rational::from_secs_f64(
                0.5 * (shot.usable_start.as_secs_f64() + shot.usable_end.as_secs_f64()),
            );
            let end_sample = (shot.usable_end - one_frame_dur).max(shot.usable_start);

            let t0 = backend
                .decode_segment(
                    video_path,
                    shot.usable_start,
                    one_frame_dur,
                    self.thumb_res,
                    fps,
                    1.0,
                )
                .ok()
                .and_then(|mut v| v.pop())
                .unwrap_or_else(|| {
                    RgbaFrame::solid(
                        self.thumb_res.width,
                        self.thumb_res.height,
                        [45, 50, 65, 255],
                        shot.usable_start,
                    )
                });

            let t1 = backend
                .decode_segment(video_path, mid, one_frame_dur, self.thumb_res, fps, 1.0)
                .ok()
                .and_then(|mut v| v.pop())
                .unwrap_or_else(|| t0.clone());

            let t2 = backend
                .decode_segment(
                    video_path,
                    end_sample,
                    one_frame_dur,
                    self.thumb_res,
                    fps,
                    1.0,
                )
                .ok()
                .and_then(|mut v| v.pop())
                .unwrap_or_else(|| t1.clone());

            if let Ok(mut cache) = global_media_cache().write() {
                cache.store_shot_thumbnails(shot.id, [t0, t1, t2]);
            }
        }
        Ok(())
    }

    /// Toggle whether `shot_id` is included in the candidate pool for `ae-sequencer`.
    pub fn toggle_shot(&mut self, shot_id: usize) -> Option<bool> {
        let shot = self.shots.iter_mut().find(|s| s.id == shot_id)?;
        shot.enabled = !shot.enabled;
        Some(shot.enabled)
    }

    pub fn set_shot_enabled(&mut self, shot_id: usize, enabled: bool) -> bool {
        if let Some(shot) = self.shots.iter_mut().find(|s| s.id == shot_id) {
            shot.enabled = enabled;
            true
        } else {
            false
        }
    }

    pub fn set_all_enabled(&mut self, enabled: bool) {
        for s in &mut self.shots {
            s.enabled = enabled;
        }
    }

    pub fn invert_selection(&mut self) {
        for s in &mut self.shots {
            s.enabled = !s.enabled;
        }
    }

    /// Trim a shot's usable range `[usable_start_sec, usable_end_sec]`.
    pub fn trim_shot_usable(
        &mut self,
        shot_id: usize,
        usable_start_sec: f64,
        usable_end_sec: f64,
    ) -> Result<()> {
        let shot = self
            .shots
            .iter_mut()
            .find(|s| s.id == shot_id)
            .ok_or_else(|| AeError::Config(format!("shot {shot_id} not found")))?;

        let new_start = Rational::from_secs_f64(usable_start_sec).max(shot.start);
        let new_end = Rational::from_secs_f64(usable_end_sec).min(shot.end);
        if new_end <= new_start + Rational::from_millis(80) {
            return Err(AeError::Config(
                "trimmed usable range must be at least 80ms".into(),
            ));
        }
        shot.usable_start = new_start;
        shot.usable_end = new_end;
        shot.locked = true;
        Ok(())
    }

    /// Split `shot_id` at `split_sec` into two independent shots.
    pub fn split_shot_at(&mut self, shot_id: usize, split_sec: f64) -> Result<()> {
        let pos = self
            .shots
            .iter()
            .position(|s| s.id == shot_id)
            .ok_or_else(|| AeError::Config(format!("shot {shot_id} not found")))?;

        let split_pts = Rational::from_secs_f64(split_sec);
        let orig = self.shots[pos].clone();
        if split_pts <= orig.start + Rational::from_millis(100)
            || split_pts >= orig.end - Rational::from_millis(100)
        {
            return Err(AeError::Config(
                "split point must be at least 100ms inside the shot boundaries".into(),
            ));
        }

        let first = Shot {
            id: orig.id,
            start: orig.start,
            end: split_pts,
            usable_start: orig.usable_start.min(split_pts),
            usable_end: split_pts,
            cut_confidence: orig.cut_confidence,
            boundary_kind: orig.boundary_kind,
            mean_luma: orig.mean_luma,
            enabled: orig.enabled,
            locked: true,
        };
        let second = Shot {
            id: orig.id + 1,
            start: split_pts,
            end: orig.end,
            usable_start: split_pts,
            usable_end: orig.usable_end.max(split_pts),
            cut_confidence: 1.0,
            boundary_kind: ShotBoundaryKind::HardCut,
            mean_luma: orig.mean_luma,
            enabled: orig.enabled,
            locked: true,
        };

        self.shots[pos] = first;
        self.shots.insert(pos + 1, second);
        for (idx, s) in self.shots.iter_mut().enumerate() {
            s.id = idx;
        }
        Ok(())
    }

    /// Merge `shot_id` with the immediately following shot.
    pub fn merge_with_next(&mut self, shot_id: usize) -> Result<()> {
        let pos = self
            .shots
            .iter()
            .position(|s| s.id == shot_id)
            .ok_or_else(|| AeError::Config(format!("shot {shot_id} not found")))?;
        if pos + 1 >= self.shots.len() {
            return Err(AeError::Config("cannot merge the last shot".into()));
        }
        let next = self.shots.remove(pos + 1);
        let cur = &mut self.shots[pos];
        cur.end = next.end;
        cur.usable_end = next.usable_end;
        cur.locked = true;
        for (idx, s) in self.shots.iter_mut().enumerate() {
            s.id = idx;
        }
        Ok(())
    }

    pub fn to_view_items(&self, slot_assignments: &[(usize, usize)]) -> Vec<ShotViewItem> {
        // slot_assignments is &(shot_id, slot_index)
        self.shots
            .iter()
            .map(|s| {
                let used_in_slot = slot_assignments
                    .iter()
                    .find(|&&(sid, _)| sid == s.id)
                    .map(|&(_, slot_idx)| slot_idx as i32)
                    .unwrap_or(-1);
                ShotViewItem {
                    id: s.id,
                    start_sec: s.start.as_secs_f64(),
                    end_sec: s.end.as_secs_f64(),
                    usable_start_sec: s.usable_start.as_secs_f64(),
                    usable_end_sec: s.usable_end.as_secs_f64(),
                    duration_sec: s.usable_duration().as_secs_f64(),
                    enabled: s.enabled,
                    locked: s.locked,
                    boundary_kind: format!("{:?}", s.boundary_kind),
                    confidence: s.cut_confidence,
                    used_in_slot,
                    thumb_start_url: format!("image://shots/{}/0", s.id),
                    thumb_mid_url: format!("image://shots/{}/1", s.id),
                    thumb_end_url: format!("image://shots/{}/2", s.id),
                }
            })
            .collect()
    }
}
