use ae_core::{AeError, Beat, Rational, Resolution, Result, Shot, Timeline, TransitionSpec};
use ae_sequencer::{SequencerConfig, rank_alternative_shots_for_slot, sequence_timeline};
use ae_similarity::ShotDescriptorBank;
use ae_transitions::StyleTomlConfig;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimelineSlotViewItem {
    pub slot_index: usize,
    pub dst_start_sec: f64,
    pub dst_end_sec: f64,
    pub duration_sec: f64,
    pub shot_id: usize,
    pub src_in_sec: f64,
    pub src_out_sec: f64,
    pub speed: f64,
    pub transition_id: String,
    pub transition_duration_ms: i64,
    pub boundary_similarity: f32,
    pub beat_strength: f32,
    pub locked: bool,
    pub thumb_url: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlternativeShotItem {
    pub shot_id: usize,
    pub similarity: f32,
    pub thumb_url: String,
}

#[derive(Clone, Debug, Default)]
pub struct TimelineModel {
    pub config: SequencerConfig,
    pub policy: StyleTomlConfig,
    pub bank: ShotDescriptorBank,
    pub timeline: Option<Timeline>,
}

impl TimelineModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Re-run the match-cut beam search sequencer over enabled shots and enabled beats,
    /// preserving any user-locked (`locked == true`) slots and strictly avoiding clip reuse.
    pub fn resequence(&mut self, shots: &[Shot], beats: &[Beat]) -> Result<&Timeline> {
        let locked_slots: Vec<_> = self
            .timeline
            .as_ref()
            .map(|tl| {
                tl.slots
                    .iter()
                    .filter(|s| s.locked)
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let mut new_tl = sequence_timeline(shots, beats, &self.bank, &self.policy, &self.config)?;

        // Restore locked slots if their shot is still enabled and unused in other slots
        for locked in locked_slots {
            if locked.slot_index < new_tl.slots.len()
                && shots.get(locked.shot_id).is_some_and(|s| s.enabled)
            {
                let conflict_idx = new_tl
                    .slots
                    .iter()
                    .enumerate()
                    .find(|&(i, other)| i != locked.slot_index && other.shot_id == locked.shot_id)
                    .map(|(i, _)| i);

                let prev_shot_id = new_tl.slots[locked.slot_index].shot_id;
                let dst_dur = new_tl.slots[locked.slot_index].dst_duration();
                {
                    let slot = &mut new_tl.slots[locked.slot_index];
                    slot.shot_id = locked.shot_id;
                    slot.src_in = locked.src_in;
                    slot.src_out = locked.src_in + dst_dur;
                    slot.transition_in = locked.transition_in;
                    slot.locked = true;
                }
                if let Some(ci) = conflict_idx {
                    new_tl.slots[ci].shot_id = prev_shot_id;
                }
            }
        }

        self.recalculate_similarities(&mut new_tl);
        self.timeline = Some(new_tl);
        Ok(self.timeline.as_ref().unwrap())
    }

    pub fn set_resolution(&mut self, res: Resolution) {
        self.config.resolution = res;
        if let Some(ref mut tl) = self.timeline {
            tl.resolution = res;
        }
    }

    pub fn toggle_lock_slot(&mut self, slot_index: usize) -> Result<bool> {
        let tl = self
            .timeline
            .as_mut()
            .ok_or_else(|| AeError::Config("no active timeline".into()))?;
        let slot = tl
            .slots
            .get_mut(slot_index)
            .ok_or_else(|| AeError::Config(format!("slot {slot_index} out of bounds")))?;
        slot.locked = !slot.locked;
        Ok(slot.locked)
    }

    /// Return top-`k` unused enabled alternative shots ranked by visual match similarity for `slot_index`.
    pub fn rank_alternatives(
        &self,
        shots: &[Shot],
        slot_index: usize,
        top_k: usize,
    ) -> Vec<AlternativeShotItem> {
        let Some(ref tl) = self.timeline else {
            return Vec::new();
        };
        rank_alternative_shots_for_slot(shots, &self.bank, tl, slot_index, top_k)
            .into_iter()
            .map(|(shot_id, similarity)| AlternativeShotItem {
                shot_id,
                similarity,
                thumb_url: format!("image://shots/{shot_id}/0"),
            })
            .collect()
    }

    /// Swap `slot_index` to use `new_shot_id`. If `new_shot_id` was already used in another slot
    /// `other_idx`, swaps the two slots' shots so no clip is ever duplicated in the timeline.
    pub fn swap_slot_shot(
        &mut self,
        shots: &[Shot],
        slot_index: usize,
        new_shot_id: usize,
    ) -> Result<()> {
        let shot = shots
            .get(new_shot_id)
            .ok_or_else(|| AeError::Config(format!("shot {new_shot_id} not found")))?;

        let tl = self
            .timeline
            .as_mut()
            .ok_or_else(|| AeError::Config("no active timeline".into()))?;

        if slot_index >= tl.slots.len() {
            return Err(AeError::Config(format!("slot {slot_index} out of bounds")));
        }

        let prev_shot_id = tl.slots[slot_index].shot_id;
        let existing_other_slot = tl
            .slots
            .iter()
            .position(|s| s.slot_index != slot_index && s.shot_id == new_shot_id);

        if let Some(other_idx) = existing_other_slot
            && let Some(prev_shot) = shots.get(prev_shot_id)
        {
            let other_dur = tl.slots[other_idx].dst_duration();
            tl.slots[other_idx].shot_id = prev_shot_id;
            tl.slots[other_idx].src_in = prev_shot.usable_start;
            tl.slots[other_idx].src_out =
                (prev_shot.usable_start + other_dur).min(prev_shot.usable_end);
        }

        let slot = &mut tl.slots[slot_index];
        let dur = slot.dst_duration();
        slot.shot_id = new_shot_id;
        slot.src_in = shot.usable_start;
        slot.src_out = (shot.usable_start + dur).min(shot.usable_end);
        slot.locked = true;

        let mut tl_clone = tl.clone();
        self.recalculate_similarities(&mut tl_clone);
        self.timeline = Some(tl_clone);
        Ok(())
    }

    /// Slip the source in-point of `slot_index` by `delta_sec` within its shot's usable range.
    pub fn slip_slot_in_point(
        &mut self,
        shots: &[Shot],
        slot_index: usize,
        delta_sec: f64,
    ) -> Result<f64> {
        let tl = self
            .timeline
            .as_mut()
            .ok_or_else(|| AeError::Config("no active timeline".into()))?;
        let slot = tl
            .slots
            .get_mut(slot_index)
            .ok_or_else(|| AeError::Config(format!("slot {slot_index} out of bounds")))?;
        let shot = shots
            .get(slot.shot_id)
            .ok_or_else(|| AeError::Config("slot shot_id missing".into()))?;

        let src_dur = slot.src_duration();
        let max_in = (shot.usable_end - src_dur).max(shot.usable_start);
        let desired = Rational::from_secs_f64(slot.src_in.as_secs_f64() + delta_sec);
        let new_in = desired.max(shot.usable_start).min(max_in);
        slot.src_in = new_in;
        slot.src_out = new_in + src_dur;
        slot.locked = true;
        let out_sec = new_in.as_secs_f64();

        let mut tl_clone = tl.clone();
        self.recalculate_similarities(&mut tl_clone);
        self.timeline = Some(tl_clone);
        Ok(out_sec)
    }

    /// Set the transition for `slot_index` (defaults to `"cut"`, can be changed to `"crossfade"`, `"zoom_punch"`, etc.).
    pub fn set_slot_transition(
        &mut self,
        slot_index: usize,
        transition_id: &str,
        duration_ms: i64,
    ) -> Result<()> {
        let tl = self
            .timeline
            .as_mut()
            .ok_or_else(|| AeError::Config("no active timeline".into()))?;
        let slot = tl
            .slots
            .get_mut(slot_index)
            .ok_or_else(|| AeError::Config(format!("slot {slot_index} out of bounds")))?;

        if transition_id == "cut" || duration_ms <= 0 || slot_index == 0 {
            slot.transition_in = TransitionSpec::default();
        } else {
            slot.transition_in = TransitionSpec {
                id: transition_id.to_string(),
                duration: Rational::from_millis(duration_ms),
                easing: "cubic_in_out".to_string(),
                params: toml::Value::Table(toml::map::Map::new()),
            };
        }
        Ok(())
    }

    fn recalculate_similarities(&self, tl: &mut Timeline) {
        if tl.slots.len() <= 1 {
            tl.mean_boundary_similarity = 1.0;
            return;
        }
        let mut sum = 0.0f32;
        for i in 1..tl.slots.len() {
            let prev = &tl.slots[i - 1];
            let curr = &tl.slots[i];
            let sim = match (
                self.bank.descriptor_at(prev.shot_id, prev.src_out),
                self.bank.descriptor_at(curr.shot_id, curr.src_in),
            ) {
                (Some(d_out), Some(d_in)) => self.bank.similarity(d_out, d_in).clamp(0.0, 1.0),
                _ => curr.boundary_similarity,
            };
            tl.slots[i].boundary_similarity = sim;
            sum += sim;
        }
        tl.mean_boundary_similarity = sum / ((tl.slots.len() - 1) as f32);
    }

    pub fn slot_assignments(&self) -> Vec<(usize, usize)> {
        self.timeline
            .as_ref()
            .map(|tl| {
                tl.slots
                    .iter()
                    .map(|s| (s.shot_id, s.slot_index))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn to_view_items(&self) -> Vec<TimelineSlotViewItem> {
        let Some(ref tl) = self.timeline else {
            return Vec::new();
        };
        tl.slots
            .iter()
            .map(|s| TimelineSlotViewItem {
                slot_index: s.slot_index,
                dst_start_sec: s.dst_start.as_secs_f64(),
                dst_end_sec: s.dst_end.as_secs_f64(),
                duration_sec: s.dst_duration().as_secs_f64(),
                shot_id: s.shot_id,
                src_in_sec: s.src_in.as_secs_f64(),
                src_out_sec: s.src_out.as_secs_f64(),
                speed: s.speed,
                transition_id: s.transition_in.id.clone(),
                transition_duration_ms: (s.transition_in.duration.as_secs_f64() * 1000.0).round()
                    as i64,
                boundary_similarity: s.boundary_similarity,
                beat_strength: s.beat_strength,
                locked: s.locked,
                thumb_url: format!("image://shots/{}/1", s.shot_id),
            })
            .collect()
    }
}
