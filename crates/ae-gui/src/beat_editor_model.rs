use std::path::Path;

use ae_beats::{BeatAnalysisSession, BeatDetectorConfig};
use ae_core::{AeError, Beat, BeatMode, FreqBand, Rational, Result, SeparationMode};
use serde::{Deserialize, Serialize};

use crate::image_provider::global_media_cache;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BeatViewItem {
    pub index: usize,
    pub time_sec: f64,
    pub strength: f32,
    pub bass_freq_hz: f32,
    pub is_downbeat: bool,
    pub enabled: bool,
    pub is_manual: bool,
    pub is_ghost: bool,
}

#[derive(Clone, Debug)]
pub struct BeatEditorModel {
    pub config: BeatDetectorConfig,
    pub session: Option<BeatAnalysisSession>,
    pub beats: Vec<Beat>,
    pub spectrogram_revision: u64,
}

impl Default for BeatEditorModel {
    fn default() -> Self {
        Self::new()
    }
}

impl BeatEditorModel {
    pub fn new() -> Self {
        Self {
            config: BeatDetectorConfig::default(),
            session: None,
            beats: Vec::new(),
            spectrogram_revision: 0,
        }
    }

    /// Load an audio file, compute and cache its STFT spectrogram, and run initial bass beat detection.
    pub fn load_audio(&mut self, path: &Path) -> Result<()> {
        let mut session = BeatAnalysisSession::from_audio_file(path)?;
        self.beats.clear();
        let detected = session.recompute_beats(&self.config, &[])?;
        self.beats = detected;
        self.session = Some(session);
        self.refresh_spectrogram_cache();
        Ok(())
    }

    pub fn load_from_session(&mut self, mut session: BeatAnalysisSession) -> Result<()> {
        let detected = session.recompute_beats(&self.config, &self.beats)?;
        self.beats = detected;
        self.session = Some(session);
        self.refresh_spectrogram_cache();
        Ok(())
    }

    pub fn duration_secs(&self) -> f64 {
        self.session
            .as_ref()
            .map(|s| s.duration.as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Drag beat `index` to `new_time_sec` and mark it `is_manual = true`.
    /// Allows free positioning within the song duration (re-sorting beats if dragged across another beat).
    pub fn move_beat(&mut self, index: usize, new_time_sec: f64) -> Result<f64> {
        if index >= self.beats.len() {
            return Err(AeError::Config(format!("beat index {index} out of bounds")));
        }
        let dur = self.duration_secs().max(new_time_sec).max(0.1);
        let clamped_sec = new_time_sec.clamp(0.01, dur - 0.01);

        self.beats[index].time = Rational::from_secs_f64(clamped_sec);
        self.beats[index].is_manual = true;
        self.beats[index].enabled = true;
        self.beats.sort_by_key(|b| b.time);
        for (i, b) in self.beats.iter_mut().enumerate() {
            b.index = i;
        }
        Ok(clamped_sec)
    }

    /// Toggle a beat active/inactive by clicking its pin in the GUI.
    pub fn toggle_beat(&mut self, index: usize) -> Result<bool> {
        let beat = self
            .beats
            .get_mut(index)
            .ok_or_else(|| AeError::Config(format!("beat index {index} out of bounds")))?;
        beat.enabled = !beat.enabled;
        beat.is_manual = true;
        Ok(beat.enabled)
    }

    /// Add a manual beat at `time_sec` (or promote a nearby ghost peak).
    pub fn add_beat_at(&mut self, time_sec: f64) -> Result<usize> {
        let t = Rational::from_secs_f64(time_sec.max(0.0));
        // If there is already a beat within 40ms, enable it and move it to t
        if let Some((idx, existing)) = self
            .beats
            .iter_mut()
            .enumerate()
            .find(|(_, b)| (b.time - t).abs().as_secs_f64() < 0.04)
        {
            existing.time = t;
            existing.enabled = true;
            existing.is_manual = true;
            self.refresh_spectrogram_cache();
            return Ok(idx);
        }

        self.beats.push(Beat {
            index: 0,
            time: t,
            strength: 0.85,
            bass_freq_hz: 80.0,
            is_downbeat: false,
            enabled: true,
            is_manual: true,
        });
        self.beats.sort_by_key(|b| b.time);
        let mut inserted_idx = 0;
        for (i, b) in self.beats.iter_mut().enumerate() {
            b.index = i;
            if b.time == t && b.is_manual {
                inserted_idx = i;
            }
        }
        self.refresh_spectrogram_cache();
        Ok(inserted_idx)
    }

    /// Delete a beat marker at `index`.
    pub fn remove_beat(&mut self, index: usize) -> Result<()> {
        if index >= self.beats.len() {
            return Err(AeError::Config(format!("beat index {index} out of bounds")));
        }
        self.beats.remove(index);
        for (i, b) in self.beats.iter_mut().enumerate() {
            b.index = i;
        }
        self.refresh_spectrogram_cache();
        Ok(())
    }

    /// Update the bass frequency isolation band (`low_hz..high_hz`) and recompute beats in < 15 ms
    /// from the cached STFT while preserving `is_manual` beats.
    pub fn set_bass_band(&mut self, low_hz: f32, high_hz: f32) -> Result<()> {
        let lo = low_hz.clamp(15.0, 600.0);
        let hi = high_hz.clamp(lo + 15.0, 1200.0);
        self.config.bass_band = FreqBand {
            low_hz: lo,
            high_hz: hi,
        };
        self.recompute_preserving_manual()
    }

    pub fn set_sensitivity(&mut self, peak_delta: f32) -> Result<()> {
        self.config.peak_delta = peak_delta.clamp(0.15, 3.0);
        self.recompute_preserving_manual()
    }

    pub fn set_separation_mode(&mut self, mode: SeparationMode) -> Result<()> {
        self.config.separation = mode;
        self.recompute_preserving_manual()
    }

    pub fn set_beat_mode(&mut self, mode: BeatMode) -> Result<()> {
        self.config.beat_mode = mode;
        self.recompute_preserving_manual()
    }

    pub fn recompute_preserving_manual(&mut self) -> Result<()> {
        if let Some(ref mut session) = self.session {
            self.beats = session.recompute_beats(&self.config, &self.beats)?;
            self.refresh_spectrogram_cache();
        }
        Ok(())
    }

    fn refresh_spectrogram_cache(&mut self) {
        if let Some(ref session) = self.session {
            let rgba = session.render_spectrogram_rgba(&self.beats);
            if let Ok(mut cache) = global_media_cache().write() {
                self.spectrogram_revision = cache.update_spectrogram(rgba);
            }
        }
    }

    pub fn to_view_items(&self) -> Vec<BeatViewItem> {
        let mut out: Vec<BeatViewItem> = self
            .beats
            .iter()
            .map(|b| BeatViewItem {
                index: b.index,
                time_sec: b.time.as_secs_f64(),
                strength: b.strength,
                bass_freq_hz: b.bass_freq_hz,
                is_downbeat: b.is_downbeat,
                enabled: b.enabled,
                is_manual: b.is_manual,
                is_ghost: false,
            })
            .collect();

        if let Some(ref session) = self.session {
            let base = out.len();
            for (g_idx, g) in session.ghost_peaks.iter().enumerate() {
                out.push(BeatViewItem {
                    index: base + g_idx,
                    time_sec: g.time.as_secs_f64(),
                    strength: g.strength,
                    bass_freq_hz: g.bass_freq_hz,
                    is_downbeat: false,
                    enabled: false,
                    is_manual: false,
                    is_ghost: true,
                });
            }
        }

        out
    }
}
