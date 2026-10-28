use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AeError {
    #[error("unsupported or missing codec '{codec}': {details}")]
    UnsupportedCodec { codec: String, details: String },

    #[error(
        "audio duration ({audio_secs:.2}s) exceeds available footage ({footage_secs:.2}s) and reuse/retiming could not fill all {missing_slots} remaining slots"
    )]
    InsufficientFootage {
        audio_secs: f64,
        footage_secs: f64,
        missing_slots: usize,
    },

    #[error("too few usable shots detected ({found}, minimum required: {min_required})")]
    TooFewShots { found: usize, min_required: usize },

    #[error("media decode error for '{path}': {message}")]
    Decode { path: PathBuf, message: String },

    #[error("media encode error for '{path}': {message}")]
    Encode { path: PathBuf, message: String },

    #[error("audio processing error: {0}")]
    Audio(String),

    #[error("ML model error ({model}): {message}")]
    Model { model: String, message: String },

    #[error("transition error ({id}): {message}")]
    Transition { id: String, message: String },

    #[error("configuration error: {0}")]
    Config(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("TOML parse error: {0}")]
    TomlDe(#[from] toml::de::Error),
}

pub type Result<T> = std::result::Result<T, AeError>;
