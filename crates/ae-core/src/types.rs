use std::fmt;
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{AeError, Result};
use crate::time::Rational;

/// Target output resolution. Supports presets (`720p`, `1080p`, `4k`, `vertical`) or `WxH`.
/// When rendering, frames are stretched to fill (axes scaled independently, SAR 1:1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

impl Resolution {
    pub const fn new(width: u32, height: u32) -> Self {
        // Ensure even dimensions for YUV420p encoders
        let w = if width.is_multiple_of(2) {
            width
        } else {
            width + 1
        };
        let h = if height.is_multiple_of(2) {
            height
        } else {
            height + 1
        };
        Self {
            width: w,
            height: h,
        }
    }

    pub const P720: Self = Self {
        width: 1280,
        height: 720,
    };
    pub const P1080: Self = Self {
        width: 1920,
        height: 1080,
    };
    pub const P4K: Self = Self {
        width: 3840,
        height: 2160,
    };
    pub const VERTICAL: Self = Self {
        width: 1080,
        height: 1920,
    };

    pub fn aspect_ratio(self) -> f32 {
        (self.width as f32) / (self.height.max(1) as f32)
    }
}

impl Default for Resolution {
    fn default() -> Self {
        Self::P1080
    }
}

impl FromStr for Resolution {
    type Err = AeError;

    fn from_str(s: &str) -> Result<Self> {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "720p" | "hd" => Ok(Self::P720),
            "1080p" | "fhd" => Ok(Self::P1080),
            "4k" | "2160p" | "uhd" => Ok(Self::P4K),
            "vertical" | "9:16" | "portrait" | "reels" | "shorts" => Ok(Self::VERTICAL),
            _ => {
                let (w_str, h_str) = lower.split_once('x').ok_or_else(|| {
                    AeError::Config(format!(
                        "invalid resolution '{s}': expected WxH (e.g. 1080x1920) or preset (720p, 1080p, 4k, vertical)"
                    ))
                })?;
                let w: u32 = w_str
                    .trim()
                    .parse()
                    .map_err(|_| AeError::Config(format!("invalid width in resolution '{s}'")))?;
                let h: u32 = h_str
                    .trim()
                    .parse()
                    .map_err(|_| AeError::Config(format!("invalid height in resolution '{s}'")))?;
                if w < 16 || h < 16 || w > 16384 || h > 16384 {
                    return Err(AeError::Config(format!(
                        "resolution {w}x{h} out of valid range (16..=16384)"
                    )));
                }
                Ok(Self::new(w, h))
            }
        }
    }
}

impl fmt::Display for Resolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ShotBoundaryKind {
    #[default]
    HardCut,
    Dissolve,
    Fade,
    Initial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ShotDetectorMode {
    #[default]
    Classic,
    Neural,
}

impl FromStr for ShotDetectorMode {
    type Err = AeError;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "classic" => Ok(Self::Classic),
            "neural" | "transnet" | "transnetv2" => Ok(Self::Neural),
            other => Err(AeError::Config(format!(
                "unknown shot detector mode '{other}' (expected classic|neural)"
            ))),
        }
    }
}

fn default_true() -> bool {
    true
}

/// A detected shot interval `[start, end)` within the source video, along with a trimmed
/// `[usable_start, usable_end)` range free of mid-dissolve or black/static edge frames.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Shot {
    pub id: usize,
    pub start: Rational,
    pub end: Rational,
    pub usable_start: Rational,
    pub usable_end: Rational,
    pub cut_confidence: f32,
    pub boundary_kind: ShotBoundaryKind,
    pub mean_luma: f32,
    /// Whether this shot is enabled/checked in the GUI Shot Pool for sequencing.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Whether the user manually trimmed or locked this shot's boundaries.
    #[serde(default)]
    pub locked: bool,
}

impl Shot {
    pub fn duration(&self) -> Rational {
        (self.end - self.start).max(Rational::ZERO)
    }

    pub fn usable_duration(&self) -> Rational {
        (self.usable_end - self.usable_start).max(Rational::ZERO)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SeparationMode {
    Band,
    #[default]
    Hpss,
    Demucs,
}

impl FromStr for SeparationMode {
    type Err = AeError;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "band" | "bandpass" => Ok(Self::Band),
            "hpss" | "harmonic" => Ok(Self::Hpss),
            "demucs" | "htdemucs" => Ok(Self::Demucs),
            other => Err(AeError::Config(format!(
                "unknown separation mode '{other}' (expected band|hpss|demucs)"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BeatMode {
    Onsets,
    Grid,
    #[default]
    Hybrid,
}

impl FromStr for BeatMode {
    type Err = AeError;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "onsets" | "onset" => Ok(Self::Onsets),
            "grid" => Ok(Self::Grid),
            "hybrid" => Ok(Self::Hybrid),
            other => Err(AeError::Config(format!(
                "unknown beat mode '{other}' (expected onsets|grid|hybrid)"
            ))),
        }
    }
}

/// Frequency band `[low_hz, high_hz]` for bass instrument isolation (default 30–250 Hz).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FreqBand {
    pub low_hz: f32,
    pub high_hz: f32,
}

impl Default for FreqBand {
    fn default() -> Self {
        Self {
            low_hz: 30.0,
            high_hz: 250.0,
        }
    }
}

impl FromStr for FreqBand {
    type Err = AeError;
    fn from_str(s: &str) -> Result<Self> {
        let (lo_s, hi_s) = s.trim().split_once('-').ok_or_else(|| {
            AeError::Config(format!("invalid frequency band '{s}': expected LOW-HIGH (e.g. 30-250)"))
        })?;
        let low_hz: f32 = lo_s
            .trim()
            .parse()
            .map_err(|_| AeError::Config(format!("invalid low frequency in '{s}'")))?;
        let high_hz: f32 = hi_s
            .trim().trim_end_matches("hz").trim_end_matches("Hz")
            .parse()
            .map_err(|_| AeError::Config(format!("invalid high frequency in '{s}'")))?;
        if low_hz <= 5.0 || high_hz <= low_hz || high_hz > 10_000.0 {
            return Err(AeError::Config(format!(
                "frequency band {low_hz}-{high_hz} Hz out of valid range"
            )));
        }
        Ok(Self { low_hz, high_hz })
    }
}

impl fmt::Display for FreqBand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.0}-{:.0}", self.low_hz, self.high_hz)
    }
}

/// Detected bass-instrument beat / onset event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Beat {
    pub index: usize,
    pub time: Rational,
    /// Normalized strength in `[0.0, 1.0]`.
    pub strength: f32,
    /// Estimated fundamental / centroid bass frequency in Hz.
    pub bass_freq_hz: f32,
    /// True if this beat falls on a bar/downbeat accent.
    pub is_downbeat: bool,
    /// True if this beat is active (toggled on in the GUI beat editor).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// True if the user manually dragged or inserted this beat (preserved across re-detection).
    #[serde(default)]
    pub is_manual: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SimilarityMode {
    #[default]
    Fast,
    Neural,
}

impl FromStr for SimilarityMode {
    type Err = AeError;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fast" | "classic" => Ok(Self::Fast),
            "neural" | "dinov2" | "clip" => Ok(Self::Neural),
            other => Err(AeError::Config(format!(
                "unknown similarity mode '{other}' (expected fast|neural)"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SequenceOrder {
    #[default]
    Similarity,
    Chronological,
    Mixed,
}

impl FromStr for SequenceOrder {
    type Err = AeError;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "similarity" => Ok(Self::Similarity),
            "chronological" | "chrono" => Ok(Self::Chronological),
            "mixed" | "hybrid" => Ok(Self::Mixed),
            other => Err(AeError::Config(format!(
                "unknown sequence order '{other}' (expected similarity|chronological|mixed)"
            ))),
        }
    }
}

/// Allowed playback speed range for fitting slightly shorter shots into beat slots.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpeedRange {
    pub min: f64,
    pub max: f64,
}

impl Default for SpeedRange {
    fn default() -> Self {
        Self { min: 1.0, max: 1.0 }
    }
}

impl FromStr for SpeedRange {
    type Err = AeError;
    fn from_str(s: &str) -> Result<Self> {
        if let Some((lo, hi)) = s.trim().split_once('-') {
            let min: f64 = lo
                .trim()
                .parse()
                .map_err(|_| AeError::Config(format!("invalid min speed in '{s}'")))?;
            let max: f64 = hi
                .trim()
                .parse()
                .map_err(|_| AeError::Config(format!("invalid max speed in '{s}'")))?;
            if min <= 0.1 || max < min || max > 4.0 {
                return Err(AeError::Config(format!(
                    "speed range {min}..{max} out of valid range (0.1..=4.0)"
                )));
            }
            Ok(Self { min, max })
        } else {
            let v: f64 = s
                .trim()
                .parse()
                .map_err(|_| AeError::Config(format!("invalid speed '{s}'")))?;
            Ok(Self { min: v, max: v })
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VideoCodec {
    #[default]
    X264,
    X265,
    Av1,
    Nvenc,
    Vaapi,
}

impl FromStr for VideoCodec {
    type Err = AeError;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "x264" | "h264" | "libx264" => Ok(Self::X264),
            "x265" | "h265" | "hevc" | "libx265" => Ok(Self::X265),
            "av1" | "svtav1" | "libsvtav1" => Ok(Self::Av1),
            "nvenc" | "h264_nvenc" => Ok(Self::Nvenc),
            "vaapi" | "h264_vaapi" => Ok(Self::Vaapi),
            other => Err(AeError::UnsupportedCodec {
                codec: other.to_string(),
                details: "expected one of: x264, x265, av1, nvenc, vaapi".to_string(),
            }),
        }
    }
}

impl VideoCodec {
    pub fn ffmpeg_encoder_name(self) -> &'static str {
        match self {
            Self::X264 => "libx264",
            Self::X265 => "libx265",
            Self::Av1 => "libsvtav1",
            Self::Nvenc => "h264_nvenc",
            Self::Vaapi => "h264_vaapi",
        }
    }
}

/// Specification for a transition applied at the start of a timeline slot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransitionSpec {
    pub id: String,
    pub duration: Rational,
    pub easing: String,
    #[serde(default = "default_toml_table")]
    pub params: toml::Value,
}

fn default_toml_table() -> toml::Value {
    toml::Value::Table(toml::map::Map::new())
}

impl Default for TransitionSpec {
    fn default() -> Self {
        Self {
            id: "cut".to_string(),
            duration: Rational::ZERO,
            easing: "linear".to_string(),
            params: default_toml_table(),
        }
    }
}

/// A single beat-aligned edit slot in the output `Timeline`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimelineSlot {
    pub slot_index: usize,
    /// Timeline output start time `[dst_start, dst_end)`.
    pub dst_start: Rational,
    pub dst_end: Rational,
    /// Source shot identifier.
    pub shot_id: usize,
    /// In-point and out-point in the source video clip `[src_in, src_out)`.
    pub src_in: Rational,
    pub src_out: Rational,
    /// Playback speed factor (`1.0` = normal speed).
    pub speed: f64,
    /// Transition leading into this slot from `slot_index - 1`.
    pub transition_in: TransitionSpec,
    /// Cosine / composite visual similarity in `[0.0, 1.0]` between the previous slot's
    /// outgoing frame and this slot's incoming frame.
    pub boundary_similarity: f32,
    /// Beat strength `[0.0, 1.0]` at the cut point.
    pub beat_strength: f32,
    /// True if the user pinned/locked this slot in the GUI timeline.
    #[serde(default)]
    pub locked: bool,
}

impl TimelineSlot {
    pub fn dst_duration(&self) -> Rational {
        (self.dst_end - self.dst_start).max(Rational::ZERO)
    }

    pub fn src_duration(&self) -> Rational {
        (self.src_out - self.src_in).max(Rational::ZERO)
    }
}

/// Complete serializable edit decision timeline (JSON).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Timeline {
    pub version: u32,
    pub source_video: String,
    pub source_audio: String,
    pub resolution: Resolution,
    pub fps: Rational,
    pub total_duration: Rational,
    pub mean_boundary_similarity: f32,
    pub slots: Vec<TimelineSlot>,
}

impl Timeline {
    pub fn save_json(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load_json(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let tl: Self = serde_json::from_str(&content)?;
        Ok(tl)
    }
}
