use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use ae_core::{
    AeError, CacheManager, ModelManifest, Rational, Resolution, Result, RgbaFrame, Shot,
};
use ae_media::MediaBackend;
use ae_render::{CharacterAnalysisDetector, crop_frame};
use ae_similarity::CharacterEncoder;
use crossbeam_channel::bounded;
use serde::{Deserialize, Serialize};

use crate::character_model::{CharacterAnalysis, CharacterObservation};
use crate::image_provider::set_gui_progress;

#[derive(Clone, Serialize, Deserialize)]
pub struct CharacterAnalysisOutput {
    pub analysis: CharacterAnalysis,
    pub portraits: HashMap<u64, RgbaFrame>,
    pub cache_key: String,
    pub identity_key: String,
    pub source_key: String,
    pub warning: String,
}

pub fn models_directory() -> PathBuf {
    let local = PathBuf::from("models");
    if local.join("manifest.toml").is_file() {
        local
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
    }
}

fn cancelled(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        Err(AeError::Config("Character analysis cancelled.".into()))
    } else {
        Ok(())
    }
}

fn prepare_model(
    manifest: &ModelManifest,
    directory: &Path,
    key: &str,
    cancel: &AtomicBool,
) -> Result<PathBuf> {
    cancelled(cancel)?;
    let entry = manifest
        .models
        .get(key)
        .ok_or_else(|| AeError::Config(format!("Missing model {key}")))?;
    let destination = directory.join(&entry.filename);
    if destination.exists() {
        return manifest.verify_file(key, directory);
    }
    std::fs::create_dir_all(directory)?;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let temporary = destination.with_extension(format!(
        "{}.{}.download",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        set_gui_progress(true, 1, format!("Downloading {}…", entry.name));
        let mut child = std::process::Command::new("curl")
            .args([
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--max-time",
                "300",
            ])
            .arg(&entry.url)
            .arg("--output")
            .arg(&temporary)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()?;
        let status = loop {
            if cancel.load(Ordering::Relaxed) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(AeError::Config("Character analysis cancelled.".into()));
            }
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(50)),
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(error.into());
                }
            }
        };
        if !status.success() {
            let mut stderr = String::new();
            if let Some(pipe) = child.stderr.take() {
                let _ = pipe.take(65536).read_to_string(&mut stderr);
            }
            return Err(AeError::Config(format!(
                "Model download failed: {}",
                stderr
            )));
        }
        let mut verification = manifest.clone();
        verification.models.get_mut(key).unwrap().filename = temporary
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        verification.verify_file(key, directory)?;
        std::fs::rename(&temporary, &destination)?;
        Ok(destination)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

fn validate_cached(output: &CharacterAnalysisOutput) -> bool {
    output.analysis.validate().is_ok()
        && output
            .analysis
            .observations
            .iter()
            .all(|o| o.feature.len() == 768)
        && output.portraits.iter().all(|(id, frame)| {
            output.analysis.records.iter().any(|r| r.id == *id)
                && frame.width > 0
                && frame.height > 0
                && frame.width <= 160
                && frame.height <= 160
                && frame.data.len() == frame.width as usize * frame.height as usize * 4
        })
}

pub fn analyze_characters(
    backend: &dyn MediaBackend,
    video: &Path,
    shots: &[Shot],
    cancel: &AtomicBool,
) -> Result<CharacterAnalysisOutput> {
    cancelled(cancel)?;
    let directory = models_directory();
    let manifest = ModelManifest::load(&directory.join("manifest.toml"))?;
    let keys = [
        "anime_character",
        "anime_identity_features",
        "anime_identity_metric",
    ];
    let models = keys
        .iter()
        .map(|key| prepare_model(&manifest, &directory, key, cancel))
        .collect::<Result<Vec<_>>>()?;
    cancelled(cancel)?;
    let info = backend.probe_video(video)?;
    let (width, height) = info.oriented_dimensions();
    let longest = width.max(height).max(1) as f64;
    let resolution = Resolution::new(
        (width as f64 * 640.0 / longest).round().max(2.0) as u32,
        (height as f64 * 640.0 / longest).round().max(2.0) as u32,
    );
    let cache = CacheManager::default_local();
    let params = format!(
        "v2-6fps-640-nms0.5-score0.3-ccip0.178475-margin0.02|{:?}|{:?}",
        keys.map(|key| &manifest.models[key].sha256),
        shots
            .iter()
            .map(|s| (s.usable_start, s.usable_end))
            .collect::<Vec<_>>()
    );
    let cache_key = cache.file_cache_key(video, "gui-character-analysis-v2", &params)?;
    let source_key = cache.file_cache_key(video, "gui-character-edits-v1", "")?;
    let identity_key = cache.file_cache_key(
        video,
        "gui-character-identities-v2",
        &format!(
            "6fps-640-score0.3-ccip0.178475-margin0.02|{:?}",
            keys.map(|key| &manifest.models[key].sha256)
        ),
    )?;
    if let Some(output) = cache
        .get::<CharacterAnalysisOutput>(&cache_key)
        .filter(|output| {
            output.cache_key == cache_key
                && output.source_key == source_key
                && validate_cached(output)
        })
    {
        return Ok(output);
    }
    let mut detector = CharacterAnalysisDetector::new(&models[0], 0.3, 0.5);
    let mut encoder = CharacterEncoder::load(&models[1], &models[2])?;
    let mut analysis = CharacterAnalysis::default();
    let mut sampled = Vec::new();
    let mut process = |frame: RgbaFrame| -> Result<()> {
        cancelled(cancel)?;
        if frame.pts < Rational::ZERO || frame.pts >= info.duration {
            return Ok(());
        }
        sampled.push(frame.pts);
        observe_frame(&mut analysis, &mut detector, &mut encoder, &frame)?;
        set_gui_progress(
            true,
            (5.0 + 75.0 * frame.pts.as_secs_f64() / info.duration.as_secs_f64().max(0.001)) as u32,
            "Finding animated characters…",
        );
        Ok(())
    };
    // A bounded channel keeps decode memory independent of the source length.
    std::thread::scope(|scope| -> Result<()> {
        let (tx, rx) = bounded(2);
        let decoder = scope.spawn(|| {
            backend.decode_scaled_stream(
                video,
                resolution.width,
                resolution.height,
                Some(Rational::from_secs(6)),
                tx,
            )
        });
        let mut result = Ok(());
        for frame in &rx {
            if let Err(error) = process(frame) {
                result = Err(error);
                break;
            }
        }
        drop(rx);
        let decoded = decoder
            .join()
            .map_err(|_| AeError::Config("Character decoder panicked.".into()))?;
        result?;
        decoded
    })?;
    drop(process);
    // Short scenes can lie between samples; decode one native-rate frame inside them.
    for shot in shots {
        if sampled
            .iter()
            .any(|pts| *pts >= shot.usable_start && *pts < shot.usable_end)
        {
            continue;
        }
        cancelled(cancel)?;
        let midpoint = shot.usable_start + (shot.usable_end - shot.usable_start) / 2;
        let frames = backend.decode_segment(
            video,
            midpoint,
            Rational::new(info.fps.denom(), info.fps.numer()),
            resolution,
            info.fps,
            1.0,
            Some(shot.usable_end),
        )?;
        let frame = frames
            .into_iter()
            .next()
            .ok_or_else(|| AeError::Config("No frame inside short scene.".into()))?;
        sampled.push(frame.pts);
        observe_frame(&mut analysis, &mut detector, &mut encoder, &frame)?;
    }
    analysis.sampled_pts = sampled;
    if analysis.sampled_pts.is_empty() {
        return Err(AeError::Config(
            "No source frames for character analysis.".into(),
        ));
    }
    analysis
        .scanned_ranges
        .push((Rational::ZERO, info.duration));
    set_gui_progress(true, 85, "Grouping character appearances…");
    // shortcut: published CCIP threshold is uncalibrated on video; calibrate with labeled clips before claiming accuracy.
    analysis.assign_groups(
        |a, b| {
            cancelled(cancel)?;
            Ok(encoder.distances(&[a.to_vec(), b.to_vec()])?[0][1].max(0.0))
        },
        0.17847511429108218,
        0.02,
    )?;
    analysis.validate()?;
    let portrait_ids = analysis
        .records
        .iter()
        .map(|record| record.id)
        .collect::<Vec<_>>();
    let portraits = build_character_portraits(backend, video, &analysis, &portrait_ids, cancel)?;
    cancelled(cancel)?;
    if cache.file_cache_key(video, "gui-character-analysis-v2", &params)? != cache_key {
        return Err(AeError::Config(
            "Source changed during character analysis.".into(),
        ));
    }
    let mut output = CharacterAnalysisOutput {
        analysis,
        portraits,
        cache_key,
        identity_key,
        source_key,
        warning: String::new(),
    };
    if let Err(error) = cache.put(&output.cache_key, &output) {
        output.warning = format!("Character analysis could not be cached: {error}");
    }
    Ok(output)
}

fn observe_frame(
    analysis: &mut CharacterAnalysis,
    detector: &mut CharacterAnalysisDetector,
    encoder: &mut CharacterEncoder,
    frame: &RgbaFrame,
) -> Result<()> {
    let hits = detector.detect_all(&frame)?;
    for (index, hit) in hits.iter().enumerate() {
        let crop = crop_frame(&frame, hit.rect);
        let feature = encoder.encode(&crop)?;
        let mut score = hit.score;
        if hit.rect.w < 24
            || hit.rect.h < 24
            || hits.iter().enumerate().any(|(other_index, other)| {
                if index == other_index {
                    return false;
                }
                let a = hit.rect;
                let b = other.rect;
                let overlap = (a.x + a.w).min(b.x + b.w).saturating_sub(a.x.max(b.x)) as f64
                    * (a.y + a.h).min(b.y + b.h).saturating_sub(a.y.max(b.y)) as f64;
                overlap > a.w as f64 * a.h as f64 * 0.2
            })
        {
            score = 0.0;
        }
        let id = ((frame.pts.as_secs_f64() * 1_000_000.0).round() as u64)
            .checked_mul(1024)
            .and_then(|id| id.checked_add(index as u64 + 1))
            .ok_or_else(|| AeError::Config("Character timestamp exceeds ID range.".into()))?;
        analysis.observations.push(CharacterObservation {
            id,
            pts: frame.pts,
            bounds: [
                hit.rect.x as f32 / frame.width as f32,
                hit.rect.y as f32 / frame.height as f32,
                hit.rect.w as f32 / frame.width as f32,
                hit.rect.h as f32 / frame.height as f32,
            ],
            score,
            feature,
            character_id: None,
            possible: Vec::new(),
        });
    }
    Ok(())
}

pub fn build_character_portraits(
    backend: &dyn MediaBackend,
    video: &Path,
    analysis: &CharacterAnalysis,
    ids: &[u64],
    cancel: &AtomicBool,
) -> Result<HashMap<u64, RgbaFrame>> {
    let info = backend.probe_video(video)?;
    let (width, height) = info.oriented_dimensions();
    let long = width.max(height).max(1) as f64;
    let resolution = Resolution::new(
        (width as f64 * 640.0 / long).round().max(2.0) as u32,
        (height as f64 * 640.0 / long).round().max(2.0) as u32,
    );
    let mut portraits = HashMap::new();
    for record in &analysis.records {
        if !ids.contains(&record.id) {
            continue;
        }
        cancelled(cancel)?;
        let Some(observation) = record
            .exemplars
            .first()
            .and_then(|id| analysis.observations.iter().find(|o| o.id == *id))
        else {
            continue;
        };
        let frame = backend
            .decode_segment(
                video,
                observation.pts,
                Rational::new(info.fps.denom(), info.fps.numer()),
                resolution,
                info.fps,
                1.0,
                Some(info.duration),
            )?
            .into_iter()
            .next()
            .ok_or_else(|| AeError::Config("No portrait source frame.".into()))?;
        let [x, y, w, h] = observation.bounds;
        let left = (x * frame.width as f32)
            .round()
            .clamp(0.0, (frame.width - 1) as f32) as u32;
        let top = (y * frame.height as f32)
            .round()
            .clamp(0.0, (frame.height - 1) as f32) as u32;
        let rect = ae_render::CropRect {
            x: left,
            y: top,
            w: (w * frame.width as f32)
                .round()
                .max(1.0)
                .min((frame.width - left) as f32) as u32,
            h: (h * frame.height as f32)
                .round()
                .max(1.0)
                .min((frame.height - top) as f32) as u32,
        };
        let crop = crop_frame(&frame, rect);
        let long = crop.width.max(crop.height) as f64;
        portraits.insert(
            record.id,
            ae_media::stretch_frame_lanczos(
                &crop,
                Resolution::new(
                    (crop.width as f64 * 160.0 / long).round().max(2.0) as u32,
                    (crop.height as f64 * 160.0 / long).round().max(2.0) as u32,
                ),
            )?,
        );
    }
    Ok(portraits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn character_cache_rejects_invalid_portraits() {
        let mut output = CharacterAnalysisOutput {
            analysis: CharacterAnalysis::default(),
            portraits: HashMap::new(),
            cache_key: String::new(),
            identity_key: String::new(),
            source_key: String::new(),
            warning: String::new(),
        };
        assert!(validate_cached(&output));
        output
            .portraits
            .insert(123, RgbaFrame::solid(10, 10, [0; 4], Rational::ZERO));
        assert!(!validate_cached(&output));
        assert!(cancelled(&AtomicBool::new(true)).is_err());
    }
}
