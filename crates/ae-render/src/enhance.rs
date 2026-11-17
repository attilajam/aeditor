use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicU64, Ordering},
};

use ae_core::{AeError, CacheManager, Result, RgbaFrame};
use ae_media::{LIGHT_CLEANUP_FILTER, MediaBackend};
use ort::{session::Session, value::TensorRef};
use rayon::prelude::*;

// Philip Hofmann's 1× GaterV3 restoration model, CC BY 4.0; see models/QUALITY_ENHANCER.md.
const MODEL_URL: &str = "https://github.com/Phhofm/models/releases/download/1xgaterv3_r_restore/1xgaterv3_r_restore_fp32_op17_onnxslim.onnx";
static SESSION: OnceLock<Mutex<Option<Session>>> = OnceLock::new();
// 256px tiles amortize the 32px halo: 1080p needs ~40 inferences instead of
// ~135 with 128px tiles, at 1.56x halo overhead instead of 2.25x.
const TILE: usize = 256;
const HALO: usize = 32;
const SIDE: usize = TILE + HALO * 2;
// Batched tiles per ONNX run. Fixed so macOS Core ML static shapes stay valid;
// the trailing batch is padded by repeating the last tile.
const BATCH: usize = 8;
pub(crate) const CACHE_VERSION: &str = "gater-detail-v3";

/// Fast denoising, sharpening and color polish for a preview, caching only completed output.
/// AI restoration remains exclusive to export via `enhance_quality`.
pub fn enhance_video(
    backend: &dyn MediaBackend,
    input: &Path,
    cache_dir: &Path,
    progress: Option<&crate::RenderProgressCallback>,
) -> Result<PathBuf> {
    let key = CacheManager::new(cache_dir, true).file_cache_key(
        input,
        "filtered-preview-v1",
        LIGHT_CLEANUP_FILTER,
    )?;
    let directory = cache_dir.join("enhanced_videos");
    let output = directory.join(format!("{key}.mp4"));
    if output.is_file() && backend.probe_video(&output).is_ok() {
        if let Some(cb) = progress {
            cb(100, "Filtered preview reused from cache.");
        }
        return Ok(output);
    }
    let info = backend.probe_video(input)?;
    let (width, height) = info.oriented_dimensions();
    if let Some(cb) = progress {
        cb(5, "Applying preview cleanup and color polish…");
    }
    std::fs::create_dir_all(&directory)?;
    static TEMP_ID: AtomicU64 = AtomicU64::new(0);
    let temp = output.with_extension(format!(
        "{}.{}.tmp.mp4",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<()> {
        let filtered = Command::new("ffmpeg")
            .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
            .arg(input)
            .args([
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-vf",
                &format!("scale={width}:{height},setsar=1,{LIGHT_CLEANUP_FILTER}"),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
            ])
            .arg(&temp)
            .stdin(Stdio::null())
            .output()?;
        if !filtered.status.success() {
            return Err(AeError::Encode {
                path: temp.clone(),
                message: format!(
                    "preview filtering failed: {}",
                    String::from_utf8_lossy(&filtered.stderr).trim()
                ),
            });
        }
        backend.probe_video(&temp)?;
        std::fs::rename(&temp, &output)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result?;
    if let Some(cb) = progress {
        cb(100, "Filtered preview ready.");
    }
    Ok(output)
}

fn model_error(error: impl std::fmt::Display) -> AeError {
    AeError::Model {
        model: "quality enhancer".into(),
        message: error.to_string(),
    }
}

/// Loads once, downloading and caching the model on first use. No media leaves the machine.
pub fn prepare_quality_enhancer() -> Result<()> {
    let mut session = SESSION
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(model_error)?;
    if session.is_none() {
        let env = ae_core::onnx_environment()?;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(2);
        let builder = Session::builder(env)
            .map_err(model_error)?
            .with_dimension_override("batch_size", BATCH as i64)
            .map_err(model_error)?
            .with_dimension_override("width", SIDE as i64)
            .map_err(model_error)?
            .with_dimension_override("height", SIDE as i64)
            .map_err(model_error)?;
        #[cfg(target_os = "macos")]
        let builder = builder
            .with_execution_providers([ort::ep::CoreML::default()
                .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                .with_static_input_shapes(true)
                .build()])
            .map_err(model_error)?;
        *session = Some(
            builder
                .with_intra_threads(threads)
                .map_err(model_error)?
                .commit_from_url(MODEL_URL)
                .map_err(model_error)?,
        );
    }
    Ok(())
}

/// Neural restoration at the input resolution, preserving alpha and timestamps.
pub fn enhance_quality(frame: &RgbaFrame) -> Result<RgbaFrame> {
    validate_frame(frame)?;
    Ok(enhance_frames(&[frame.clone()])?.remove(0))
}

/// Batched restoration for a group of frames. All tiles from all frames share
/// one session lock and run as `[BATCH, 3, SIDE, SIDE]` inferences, so a 1080p
/// frame needs ~5 runs instead of ~135 single-tile runs.
fn enhance_frames(frames: &[RgbaFrame]) -> Result<Vec<RgbaFrame>> {
    for frame in frames {
        validate_frame(frame)?;
    }
    if frames.is_empty() {
        return Ok(Vec::new());
    }
    prepare_quality_enhancer()?;
    // Flat (frame, left, top) list for every tile.
    let mut flat: Vec<(usize, usize, usize)> = Vec::new();
    for (fi, frame) in frames.iter().enumerate() {
        for top in (0..frame.height as usize).step_by(TILE) {
            for left in (0..frame.width as usize).step_by(TILE) {
                flat.push((fi, left, top));
            }
        }
    }
    let total_tiles = flat.len();
    debug_assert!(total_tiles > 0);
    let tile_elems = 3 * SIDE * SIDE;
    let padded = total_tiles.div_ceil(BATCH) * BATCH;
    let mut inputs = vec![0.0f32; padded * tile_elems];
    // Fill tiles in parallel; the padded tail repeats the last tile.
    let last = *flat.last().unwrap();
    inputs
        .par_chunks_exact_mut(tile_elems)
        .enumerate()
        .for_each(|(idx, slot)| {
            let (frame_idx, left, top) = if idx < total_tiles { flat[idx] } else { last };
            pack_tile(&frames[frame_idx], left, top, slot);
        });
    let mut guard = SESSION.get().unwrap().lock().map_err(model_error)?;
    let session = guard.as_mut().unwrap();
    let mut outputs = vec![0.0f32; padded * tile_elems];
    for (batch_in, batch_out) in inputs
        .chunks_exact(BATCH * tile_elems)
        .zip(outputs.chunks_exact_mut(BATCH * tile_elems))
    {
        let tensor =
            TensorRef::from_array_view(([BATCH, 3, SIDE, SIDE], batch_in)).map_err(model_error)?;
        let out = session.run(ort::inputs![tensor]).map_err(model_error)?;
        let (shape, values) = out[0].try_extract_tensor::<f32>().map_err(model_error)?;
        if shape.as_ref() != [BATCH as i64, 3, SIDE as i64, SIDE as i64]
            || values.len() != batch_out.len()
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(model_error(
                "model must return finite RGB pixels at the original resolution",
            ));
        }
        batch_out.copy_from_slice(&values);
    }
    drop(guard);
    // Scatter tiles back; padded outputs are ignored.
    let mut restored: Vec<RgbaFrame> = frames.to_vec();
    for (i, &(fi, left, top)) in flat.iter().enumerate() {
        let base = i * tile_elems;
        unpack_tile(
            &outputs[base..base + tile_elems],
            &mut restored[fi],
            left,
            top,
        );
    }
    // Blend back source detail (rayon-parallel inside).
    for (out, src) in restored.iter_mut().zip(frames.iter()) {
        sharpen_detail(out, src);
    }
    Ok(restored)
}

fn validate_frame(frame: &RgbaFrame) -> Result<()> {
    if frame.width == 0
        || frame.height == 0
        || frame.data.len() != frame.width as usize * frame.height as usize * 4
    {
        return Err(model_error("invalid RGBA frame"));
    }
    Ok(())
}

fn pack_tile(frame: &RgbaFrame, left: usize, top: usize, slot: &mut [f32]) {
    let width = frame.width as usize;
    let height = frame.height as usize;
    for y in 0..SIDE {
        let sy = (top + y).saturating_sub(HALO).min(height - 1);
        for x in 0..SIDE {
            let sx = (left + x).saturating_sub(HALO).min(width - 1);
            let px = (sy * width + sx) * 4;
            for c in 0..3 {
                slot[c * SIDE * SIDE + y * SIDE + x] = frame.data[px + c] as f32 / 255.0;
            }
        }
    }
}

fn unpack_tile(tile: &[f32], frame: &mut RgbaFrame, left: usize, top: usize) {
    let width = frame.width as usize;
    let height = frame.height as usize;
    for y in 0..TILE.min(height - top) {
        for x in 0..TILE.min(width - left) {
            for c in 0..3 {
                let value = tile[c * SIDE * SIDE + (y + HALO) * SIDE + x + HALO];
                frame.data[((top + y) * width + left + x) * 4 + c] =
                    (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
}

fn sharpen_detail(restored: &mut RgbaFrame, original: &RgbaFrame) {
    // Retain fine source detail the restoration network can soften.
    for (dst, src) in restored
        .data
        .chunks_exact_mut(4)
        .zip(original.data.chunks_exact(4))
    {
        for c in 0..3 {
            dst[c] = (0.75 * dst[c] as f32 + 0.25 * src[c] as f32).round() as u8;
        }
    }
    let (width, height) = (restored.width as usize, restored.height as usize);
    let luma: Vec<f32> = restored
        .data
        .chunks_exact(4)
        .map(|p| 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32)
        .collect();
    restored
        .data
        .par_chunks_exact_mut(4)
        .enumerate()
        .for_each(|(i, pixel)| {
            let (x, y) = (i % width, i / width);
            let mut blurred = 0.0;
            for dy in 0..3 {
                let sy = (y + dy).saturating_sub(1).min(height - 1);
                for dx in 0..3 {
                    let sx = (x + dx).saturating_sub(1).min(width - 1);
                    let weight = [1.0, 2.0, 1.0][dx] * [1.0, 2.0, 1.0][dy];
                    blurred += luma[sy * width + sx] * weight / 16.0;
                }
            }
            let detail = luma[i] - blurred;
            // A two-level noise gate and eight-level cap limit grain amplification and edge halos.
            let boost = (0.6 * detail.signum() * (detail.abs() - 2.0).max(0.0)).clamp(-8.0, 8.0);
            for channel in &mut pixel[..3] {
                *channel = (*channel as f32 + boost).round().clamp(0.0, 255.0) as u8;
            }
        });
}

#[allow(dead_code)]
fn restore_tiles(
    frame: &RgbaFrame,
    mut infer: impl FnMut(&[f32], usize) -> Result<Vec<f32>>,
) -> Result<RgbaFrame> {
    let (width, height) = (frame.width as usize, frame.height as usize);
    let mut result = frame.clone();
    let mut input = vec![0.0; 3 * SIDE * SIDE];
    // Overlapping context keeps memory bounded without scaling pixels or exposing tile edges.
    for top in (0..height).step_by(TILE) {
        for left in (0..width).step_by(TILE) {
            for y in 0..SIDE {
                let sy = (top + y).saturating_sub(HALO).min(height - 1);
                for x in 0..SIDE {
                    let sx = (left + x).saturating_sub(HALO).min(width - 1);
                    for c in 0..3 {
                        input[c * SIDE * SIDE + y * SIDE + x] =
                            frame.data[(sy * width + sx) * 4 + c] as f32 / 255.0;
                    }
                }
            }
            let output = infer(&input, SIDE)?;
            if output.len() != input.len() {
                return Err(model_error("invalid restoration tile size"));
            }
            for y in 0..TILE.min(height - top) {
                for x in 0..TILE.min(width - left) {
                    for c in 0..3 {
                        let value = output[c * SIDE * SIDE + (y + HALO) * SIDE + x + HALO];
                        result.data[((top + y) * width + left + x) * 4 + c] =
                            (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                    }
                }
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ae_core::{Rational, Resolution, Timeline, TimelineSlot, TransitionSpec};
    use ae_media::{FfmpegBackend, VideoWriterConfig};

    #[test]
    fn light_cleanup_preview_and_export_preserve_frames_and_fractional_fps() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("source.mp4");
        let backend = FfmpegBackend::new();
        let resolution = Resolution::new(64, 64);
        let fps = Rational::new(30_000, 1001);
        let duration = Rational::from_frame_index(6, fps);
        let mut writer = backend
            .create_writer(VideoWriterConfig {
                output_path: input.clone(),
                resolution,
                fps,
                crf: 0,
                ..Default::default()
            })
            .unwrap();
        let mut frame = RgbaFrame::solid(64, 64, [100, 120, 140, 255], Rational::ZERO);
        for (i, pixel) in frame.data.chunks_exact_mut(4).enumerate() {
            for channel in &mut pixel[..3] {
                *channel += ((i * 31 + i / 64 * 17) % 41) as u8;
            }
        }
        for _ in 0..6 {
            writer.write_frame(&frame).unwrap();
        }
        writer.finish().unwrap();
        let output = enhance_video(&backend, &input, tmp.path(), None).unwrap();
        let info = backend.probe_video(&output).unwrap();
        assert_eq!((info.width, info.height, info.fps), (64, 64, fps));
        assert!((info.duration - duration).as_secs_f64().abs() < 0.001);
        assert!(backend.probe_audio(&output).is_err());
        let decode = |path: &Path| {
            backend
                .decode_segment(path, Rational::ZERO, duration, resolution, fps, 1.0, None)
                .unwrap()
        };
        let original = decode(&input);
        let filtered = decode(&output);
        assert_eq!(filtered.len(), 6);
        let difference: usize = original[0]
            .data
            .iter()
            .zip(&filtered[0].data)
            .map(|(a, b)| a.abs_diff(*b) as usize)
            .sum();
        assert!(
            difference > original[0].data.len(),
            "cleanup did not change pixels"
        );
        let modified = std::fs::metadata(&output).unwrap().modified().unwrap();
        assert_eq!(
            enhance_video(&backend, &input, tmp.path(), None).unwrap(),
            output
        );
        assert_eq!(
            std::fs::metadata(&output).unwrap().modified().unwrap(),
            modified
        );

        let timeline = Timeline {
            version: 1,
            source_video: input.to_string_lossy().into_owned(),
            source_audio: String::new(),
            resolution,
            fps,
            total_duration: duration,
            mean_boundary_similarity: 1.0,
            slots: vec![TimelineSlot {
                slot_index: 0,
                dst_start: Rational::ZERO,
                dst_end: duration,
                shot_id: 0,
                src_in: Rational::ZERO,
                src_out: duration,
                speed: 1.0,
                transition_in: TransitionSpec::default(),
                boundary_similarity: 1.0,
                beat_strength: 1.0,
                locked: false,
            }],
        };
        let mut config = crate::RenderConfig {
            output_path: tmp.path().join("export.mp4"),
            light_cleanup: true,
            ..Default::default()
        };
        crate::render_timeline(&backend, &timeline, &config).unwrap();
        let exported = decode(&config.output_path);
        assert_eq!(exported.len(), 6);
        let info = backend.probe_video(&config.output_path).unwrap();
        assert_eq!((info.width, info.height, info.fps), (64, 64, fps));
        let difference: usize = exported[0]
            .data
            .iter()
            .zip(&filtered[0].data)
            .map(|(a, b)| a.abs_diff(*b) as usize)
            .sum();
        assert!(
            difference < filtered[0].data.len() * 3,
            "export differs from light preview"
        );

        config.output_path = tmp.path().join("draft.mp4");
        config.light_cleanup = false;
        let render = |config: &crate::RenderConfig| {
            crate::render_timeline_incremental_draft(&backend, &timeline, config, tmp.path(), 1)
                .unwrap()
        };
        assert_eq!(render(&config).encoded_slots, 1);
        config.light_cleanup = true;
        assert_eq!(
            render(&config).encoded_slots,
            1,
            "cleanup reused an unfiltered slot"
        );
        assert_eq!(render(&config).cached_slots, 1);
        let draft = decode(&config.output_path);
        assert_eq!(draft.len(), 6);
        let difference: usize = draft[0]
            .data
            .iter()
            .zip(&original[0].data)
            .map(|(a, b)| a.abs_diff(*b) as usize)
            .sum();
        assert!(
            difference > original[0].data.len(),
            "draft cleanup did not change pixels"
        );
    }

    #[test]
    fn sharpening_boosts_edges_without_changing_flat_areas_alpha_or_size() {
        for (w, h) in [(1, 1), (9, 3)] {
            let flat = RgbaFrame::solid(w, h, [100, 120, 140, 77], Rational::from_millis(12));
            let mut sharp = flat.clone();
            sharpen_detail(&mut sharp, &flat);
            assert_eq!(sharp, flat);
        }
        let mut edge = RgbaFrame::solid(9, 3, [80, 80, 80, 77], Rational::from_millis(12));
        for y in 0..3 {
            for x in 4..9 {
                let i = ((y * 9 + x) * 4) as usize;
                edge.data[i..i + 3].fill(160);
            }
        }
        let mut sharp = edge.clone();
        sharpen_detail(&mut sharp, &edge);
        assert!(sharp.pixel(3, 1)[0] < edge.pixel(3, 1)[0]);
        assert!(sharp.pixel(4, 1)[0] > edge.pixel(4, 1)[0]);
        assert!(sharp.data.chunks_exact(4).all(|p| p[3] == 77));
        assert_eq!((sharp.width, sharp.height, sharp.pts), (9, 3, edge.pts));
        for (a, b) in sharp.data.iter().zip(&edge.data) {
            assert!(a.abs_diff(*b) <= 8, "sharpening exceeded its halo limit");
        }
        let mut grain = RgbaFrame::solid(5, 5, [100, 100, 100, 255], Rational::ZERO);
        grain.data[(2 * 5 + 2) * 4..(2 * 5 + 2) * 4 + 3].fill(101);
        let mut sharp = grain.clone();
        sharpen_detail(&mut sharp, &grain);
        assert_eq!(sharp, grain, "one-level grain should not be amplified");
    }

    #[test]
    fn tiles_preserve_every_pixel_including_edges_alpha_and_pts() {
        for (w, h) in [(1, 1), (129, 257), (256, 128)] {
            let data = (0..w * h * 4).map(|i| (i % 251) as u8).collect();
            let frame = RgbaFrame::from_rgba(w, h, data, Rational::from_millis(123));
            assert_eq!(
                restore_tiles(&frame, |input, _| Ok(input.to_vec())).unwrap(),
                frame
            );
        }
    }

    #[test]
    fn rejects_invalid_frames_and_tile_output() {
        assert!(enhance_quality(&RgbaFrame::new(0, 0, Rational::ZERO)).is_err());
        let frame = RgbaFrame::solid(1, 1, [12, 24, 36, 78], Rational::ZERO);
        assert!(restore_tiles(&frame, |_, _| Ok(vec![])).is_err());
    }

    #[test]
    #[ignore = "downloads the 86 MB model on first use"]
    fn neural_restoration_keeps_resolution_and_changes_noisy_pixels() {
        let mut frame = RgbaFrame::solid(33, 25, [100, 120, 140, 77], Rational::from_millis(123));
        for (i, pixel) in frame.data.chunks_exact_mut(4).enumerate() {
            pixel[0] += (i % 17) as u8;
        }
        let restored = enhance_quality(&frame).unwrap();
        assert_eq!(
            (restored.width, restored.height, restored.pts),
            (33, 25, frame.pts)
        );
        assert!(restored.data.chunks_exact(4).all(|p| p[3] == 77));
        assert_ne!(restored.data, frame.data);
        let start = std::time::Instant::now();
        let repeated = enhance_quality(&frame).unwrap();
        assert_eq!(restored, repeated);
        eprintln!("Warm restoration tile: {:?}", start.elapsed());
    }
}
