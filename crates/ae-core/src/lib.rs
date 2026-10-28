pub mod cache;
pub mod error;
pub mod frame;
pub mod models;
pub mod time;
pub mod types;

pub use cache::CacheManager;
pub use error::{AeError, Result};
pub use frame::{
    FrameMut, FrameRef, LinearRgbaFrame, RgbaFrame, bt709_u8_to_linear, bt709_u8_to_linear_lut,
    linear_to_bt709_u8,
};
pub use models::{ModelEntry, ModelManifest};
pub use time::Rational;
pub use types::{
    Beat, BeatMode, FreqBand, Resolution, SeparationMode, SequenceOrder, Shot, ShotBoundaryKind,
    ShotDetectorMode, SimilarityMode, SpeedRange, Timeline, TimelineSlot, TransitionSpec,
    VideoCodec,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_time_and_frames_roundtrip() {
        let fps = Rational::new(30000, 1001); // 29.97 NTSC
        let t = Rational::from_frame_index(300, fps);
        assert!((t.as_secs_f64() - 10.01).abs() < 1e-4);
        assert_eq!(t.to_nearest_frame(fps), 300);

        let parsed = Rational::parse_duration("0.25s").unwrap();
        assert_eq!(parsed, Rational::new(1, 4));
    }

    #[test]
    fn resolution_presets_and_custom() {
        assert_eq!("1080p".parse::<Resolution>().unwrap(), Resolution::P1080);
        assert_eq!(
            "vertical".parse::<Resolution>().unwrap(),
            Resolution::new(1080, 1920)
        );
        assert_eq!(
            "1080x1920".parse::<Resolution>().unwrap(),
            Resolution::new(1080, 1920)
        );
        // Odd dimensions are rounded up to even for YUV420p compatibility
        assert_eq!(Resolution::new(853, 481), Resolution::new(854, 482));
    }

    #[test]
    fn linear_light_bt709_roundtrip() {
        for v in 0..=255u8 {
            let lin = bt709_u8_to_linear(v);
            let back = linear_to_bt709_u8(lin);
            assert_eq!(v, back, "roundtrip failed for u8 value {v}");
        }
    }
}
