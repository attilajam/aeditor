use std::path::Path;

use ae_core::{AeError, Result, RgbaFrame};
use image::{RgbImage, imageops::FilterType};
use ndarray::{Array2, Array4};
use ort::{session::Session, value::TensorRef};

fn error(message: impl ToString) -> AeError {
    AeError::Model {
        model: "ccip".into(),
        message: message.to_string(),
    }
}

pub struct CharacterEncoder {
    feature: Session,
    metric: Session,
}

impl CharacterEncoder {
    pub fn load(feature_model: &Path, metric_model: &Path) -> Result<Self> {
        let env = ae_core::onnx_environment()?;
        let feature = Session::builder(env)
            .map_err(error)?
            .commit_from_file(feature_model)
            .map_err(error)?;
        let metric = Session::builder(env)
            .map_err(error)?
            .commit_from_file(metric_model)
            .map_err(error)?;
        Ok(Self { feature, metric })
    }

    pub fn encode(&mut self, crop: &RgbaFrame) -> Result<Vec<f32>> {
        let input = preprocess(crop)?;
        let tensor = TensorRef::from_array_view(&input).map_err(error)?;
        let outputs = self.feature.run(ort::inputs![tensor]).map_err(error)?;
        let (_, value) = outputs
            .iter()
            .next()
            .ok_or_else(|| error("missing feature output"))?;
        let (shape, values) = value.try_extract_tensor::<f32>().map_err(error)?;
        validate_output(shape, values, &[1, 768])?;
        Ok(values.to_vec())
    }

    pub fn distances(&mut self, features: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        validate_features(features)?;
        if features.is_empty() {
            return Ok(Vec::new());
        }
        let n = features.len();
        let input = Array2::from_shape_vec((n, 768), features.iter().flatten().copied().collect())
            .map_err(error)?;
        let tensor = TensorRef::from_array_view(&input).map_err(error)?;
        let outputs = self.metric.run(ort::inputs![tensor]).map_err(error)?;
        let (_, value) = outputs
            .iter()
            .next()
            .ok_or_else(|| error("missing metric output"))?;
        let (shape, values) = value.try_extract_tensor::<f32>().map_err(error)?;
        validate_output(shape, values, &[n as i64, n as i64])?;
        if values.iter().any(|distance| *distance < -1e-5)
            || (0..n).any(|i| (0..n).any(|j| (values[i * n + j] - values[j * n + i]).abs() > 1e-4))
        {
            return Err(error("invalid character distance matrix"));
        }
        Ok(values
            .chunks(n)
            .map(|row| row.iter().map(|distance| distance.max(0.0)).collect())
            .collect())
    }
}

fn preprocess(frame: &RgbaFrame) -> Result<Array4<f32>> {
    let size = (frame.width as usize)
        .checked_mul(frame.height as usize)
        .and_then(|n| n.checked_mul(4));
    if frame.width == 0 || frame.height == 0 || size != Some(frame.data.len()) {
        return Err(error("invalid RGBA frame dimensions or buffer"));
    }
    let rgb: Vec<u8> = frame
        .data
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    let image = RgbImage::from_raw(frame.width, frame.height, rgb)
        .ok_or_else(|| error("invalid RGB frame"))?;
    let image = image::imageops::resize(&image, 384, 384, FilterType::Triangle);
    let mean = [0.48145466, 0.4578275, 0.40821073];
    let std = [0.26862954, 0.26130258, 0.27577711];
    let mut input = Array4::zeros((1, 3, 384, 384));
    for (x, y, pixel) in image.enumerate_pixels() {
        for c in 0..3 {
            input[[0, c, y as usize, x as usize]] = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
        }
    }
    Ok(input)
}

fn validate_features(features: &[Vec<f32>]) -> Result<()> {
    if features.len() > 64
        || features
            .iter()
            .any(|v| v.len() != 768 || v.iter().any(|x| !x.is_finite()))
    {
        return Err(error("expected at most 64 finite 768-value features"));
    }
    Ok(())
}

fn validate_output(shape: &[i64], values: &[f32], expected: &[i64]) -> Result<()> {
    if shape != expected
        || values.len() != expected.iter().product::<i64>() as usize
        || values.iter().any(|v| !v.is_finite())
    {
        return Err(error("invalid model output shape or values"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ae_core::Rational;

    #[test]
    fn character_encoder_preprocess_rgb_and_normalizes() {
        let frame = RgbaFrame::solid(1, 1, [255, 0, 128, 0], Rational::new(0, 1));
        let input = preprocess(&frame).unwrap();
        assert_eq!(input.shape(), &[1, 3, 384, 384]);
        assert!((input[[0, 0, 0, 0]] - (1.0 - 0.48145466) / 0.26862954).abs() < 1e-6);
        assert!((input[[0, 2, 383, 383]] - (128.0 / 255.0 - 0.40821073) / 0.27577711).abs() < 1e-6);
    }
    #[test]
    fn character_encoder_rejects_invalid_frames() {
        let mut frame = RgbaFrame::new(1, 1, Rational::new(0, 1));
        frame.data.pop();
        assert!(preprocess(&frame).is_err());
        frame.width = 0;
        assert!(preprocess(&frame).is_err());
    }
    #[test]
    fn character_encoder_validates_features() {
        assert!(validate_features(&vec![vec![0.0; 768]; 64]).is_ok());
        assert!(validate_features(&vec![vec![0.0; 768]; 65]).is_err());
        assert!(validate_features(&[vec![0.0; 767]]).is_err());
        assert!(validate_features(&[vec![f32::NAN; 768]]).is_err());
    }
    #[test]
    fn character_encoder_validates_outputs() {
        assert!(validate_output(&[2, 2], &[0.0; 4], &[2, 2]).is_ok());
        assert!(validate_output(&[4], &[0.0; 4], &[2, 2]).is_err());
        assert!(validate_output(&[2, 2], &[f32::INFINITY; 4], &[2, 2]).is_err());
    }

    #[test]
    #[ignore = "requires the pinned CCIP models in models/"]
    fn character_encoder_real_models() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        let manifest = ae_core::ModelManifest::load(&root.join("manifest.toml")).unwrap();
        let feature = manifest
            .verify_file("anime_identity_features", &root)
            .unwrap();
        let metric = manifest
            .verify_file("anime_identity_metric", &root)
            .unwrap();
        let mut encoder = CharacterEncoder::load(&feature, &metric).unwrap();
        let frame = RgbaFrame::solid(128, 96, [150, 50, 80, 255], Rational::ZERO);
        let first = encoder.encode(&frame).unwrap();
        let second = encoder.encode(&frame).unwrap();
        assert_eq!(first.len(), 768);
        assert!(first.iter().zip(&second).all(|(a, b)| (a - b).abs() < 1e-5));
        let distances = encoder.distances(&[first, second]).unwrap();
        assert_eq!(distances.len(), 2);
        assert!(
            distances
                .iter()
                .flatten()
                .all(|distance| distance.abs() < 1e-4)
        );
    }
}
