use std::path::{Path, PathBuf};
use std::thread;

use ae_core::{
    AeError, Rational, Resolution, Result, RgbaFrame, Shot, SimilarityMode,
};
use ae_media::{MediaBackend, stretch_frame_lanczos};
use crossbeam_channel::bounded;
use dssim_core::Dssim;
use image::{DynamicImage, RgbaImage};
use image_hasher::{HashAlg, HasherConfig};
use ndarray::Array4;
use palette::{FromColor, Oklab, Srgb};
use rayon::prelude::*;
use realfft::RealFftPlanner;
use rgb::FromSlice;
use serde::{Deserialize, Serialize};

pub const SAMPLE_FPS: i64 = 6;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimilarityWeights {
    pub oklab_layout: f32,
    pub oklab_stats: f32,
    pub phash: f32,
    pub motion: f32,
    pub neural: f32,
}

impl Default for SimilarityWeights {
    fn default() -> Self {
        Self {
            oklab_layout: 0.40,
            oklab_stats: 0.20,
            phash: 0.15,
            motion: 0.10,
            neural: 0.15,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimilarityConfig {
    pub mode: SimilarityMode,
    pub sample_fps: Rational,
    pub weights: SimilarityWeights,
    pub model_path: Option<PathBuf>,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            mode: SimilarityMode::Fast,
            sample_fps: Rational::from_secs(SAMPLE_FPS),
            weights: SimilarityWeights::default(),
            model_path: None,
        }
    }
}

/// Multi-modal visual descriptor for a single sampled frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameDescriptor {
    pub shot_id: usize,
    pub pts: Rational,
    /// 8x8 Oklab thumbnail (64 pixels * 3 channels = 192 floats).
    pub oklab_thumb: Vec<f32>,
    /// Oklab mean (L, a, b) and std (L, a, b).
    pub oklab_stats: [f32; 6],
    /// 64-bit DCT / gradient perceptual hash bytes.
    pub phash: Vec<u8>,
    /// Global camera motion vector `(dx, dy)` in normalized frame units/frame via phase correlation.
    pub motion_vec: [f32; 2],
    /// Mean BT.709 luma in `[0.0, 1.0]`.
    pub mean_luma: f32,
    /// L2-normalized dense embedding (combines Oklab layout, stats, motion, and optional neural embedding)
    /// used for fast HNSW nearest-neighbor candidate search.
    pub combined_embedding: Vec<f32>,
    /// Small 32x18 RGBA thumbnail retained for `dssim-core` MS-SSIM shortlist re-ranking.
    pub ssim_thumb_rgba: Vec<u8>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChannelZStats {
    pub mean: f32,
    pub std: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ShotDescriptorBank {
    /// Descriptors grouped by `shot_id`.
    pub by_shot: Vec<Vec<FrameDescriptor>>,
    pub oklab_z: ChannelZStats,
    pub stats_z: ChannelZStats,
    pub phash_z: ChannelZStats,
    pub motion_z: ChannelZStats,
}

impl ShotDescriptorBank {
    /// Find the descriptor in `shot_id` closest to timestamp `pts`.
    pub fn descriptor_at(&self, shot_id: usize, pts: Rational) -> Option<&FrameDescriptor> {
        let list = self.by_shot.get(shot_id)?;
        list.iter().min_by_key(|d| (d.pts - pts).abs())
    }

    /// Compute composite visual similarity in `[0.0, 1.0]` (higher = better match cut)
    /// between an outgoing frame and an incoming frame.
    pub fn similarity(&self, out_desc: &FrameDescriptor, in_desc: &FrameDescriptor) -> f32 {
        // Cosine similarity of L2-normalized combined embeddings in [-1, 1] -> mapped to [0, 1]
        let dot: f32 = out_desc
            .combined_embedding
            .iter()
            .zip(in_desc.combined_embedding.iter())
            .map(|(a, b)| a * b)
            .sum();
        let cos_sim = ((dot + 1.0) * 0.5).clamp(0.0, 1.0);

        // Hamming similarity on perceptual hash
        let phash_sim = phash_similarity(&out_desc.phash, &in_desc.phash);

        // Motion continuity similarity
        let mdx = out_desc.motion_vec[0] - in_desc.motion_vec[0];
        let mdy = out_desc.motion_vec[1] - in_desc.motion_vec[1];
        let motion_sim = (-(mdx * mdx + mdy * mdy) * 4.0).exp();

        (0.65 * cos_sim + 0.20 * phash_sim + 0.15 * motion_sim).clamp(0.0, 1.0)
    }

    /// Fine MS-SSIM similarity in `[0.0, 1.0]` using `dssim-core` for re-ranking top candidates.
    pub fn ms_ssim_similarity(&self, out_desc: &FrameDescriptor, in_desc: &FrameDescriptor) -> f32 {
        if out_desc.ssim_thumb_rgba.len() != 32 * 18 * 4
            || in_desc.ssim_thumb_rgba.len() != 32 * 18 * 4
        {
            return self.similarity(out_desc, in_desc);
        }
        let dssim = Dssim::new();
        let Some(img1) = dssim.create_image_rgba(out_desc.ssim_thumb_rgba.as_rgba(), 32, 18) else {
            return self.similarity(out_desc, in_desc);
        };
        let Some(img2) = dssim.create_image_rgba(in_desc.ssim_thumb_rgba.as_rgba(), 32, 18) else {
            return self.similarity(out_desc, in_desc);
        };
        let (val, _) = dssim.compare(&img1, img2);
        let d: f64 = val.into();
        (1.0 / (1.0 + d * 4.0)).clamp(0.0, 1.0) as f32
    }
}

fn phash_similarity(a: &[u8], b: &[u8]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.5;
    }
    let total_bits = (a.len() * 8) as f32;
    let diff_bits: u32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum();
    (1.0 - (diff_bits as f32) / total_bits).clamp(0.0, 1.0)
}

/// Extract visual descriptors across all usable shots at ~6 fps.
pub fn extract_shot_descriptors(
    backend: &dyn MediaBackend,
    video_path: &Path,
    shots: &[Shot],
    config: &SimilarityConfig,
) -> Result<ShotDescriptorBank> {
    let (tx, rx) = bounded::<RgbaFrame>(128);
    let path_owned = video_path.to_path_buf();
    let sample_fps = config.sample_fps;

    let sampled_frames = thread::scope(|s| -> Result<Vec<RgbaFrame>> {
        let handle = s.spawn(move || {
            backend.decode_scaled_stream(&path_owned, 64, 36, Some(sample_fps), tx)
        });
        let mut frames = Vec::new();
        while let Ok(f) = rx.recv() {
            frames.push(f);
        }
        handle.join().map_err(|_| AeError::Decode {
            path: video_path.to_path_buf(),
            message: "similarity decode thread panicked".into(),
        })??;
        Ok(frames)
    })?;

    build_descriptor_bank_from_frames(&sampled_frames, shots, config)
}

pub fn build_descriptor_bank_from_frames(
    sampled_frames: &[RgbaFrame],
    shots: &[Shot],
    config: &SimilarityConfig,
) -> Result<ShotDescriptorBank> {
    let hasher = HasherConfig::new()
        .hash_alg(HashAlg::DoubleGradient)
        .hash_size(8, 8)
        .to_hasher();

    let computed: Vec<FrameDescriptor> = sampled_frames
        .par_iter()
        .enumerate()
        .filter_map(|(idx, frame)| {
            // Assign frame to the shot whose [start, end) contains frame.pts
            let shot = shots
                .iter()
                .find(|s| frame.pts >= s.start && frame.pts <= s.end)?;

            let prev_frame = if idx > 0 && sampled_frames[idx - 1].pts >= shot.start {
                Some(&sampled_frames[idx - 1])
            } else {
                None
            };

            let thumb_8x8 = stretch_frame_lanczos(frame, Resolution::new(8, 8)).ok()?;
            let thumb_32x18 = stretch_frame_lanczos(frame, Resolution::new(32, 18)).ok()?;

            let mut oklab_thumb = Vec::with_capacity(8 * 8 * 3);
            let mut sum_l = 0.0f32;
            let mut sum_a = 0.0f32;
            let mut sum_b = 0.0f32;
            let mut l_vals = Vec::with_capacity(64);
            let mut a_vals = Vec::with_capacity(64);
            let mut b_vals = Vec::with_capacity(64);

            for px in thumb_8x8.data.chunks_exact(4) {
                let srgb = Srgb::new(
                    (px[0] as f32) / 255.0,
                    (px[1] as f32) / 255.0,
                    (px[2] as f32) / 255.0,
                );
                let ok: Oklab = Oklab::from_color(srgb);
                oklab_thumb.push(ok.l);
                oklab_thumb.push(ok.a * 2.5);
                oklab_thumb.push(ok.b * 2.5);
                sum_l += ok.l;
                sum_a += ok.a;
                sum_b += ok.b;
                l_vals.push(ok.l);
                a_vals.push(ok.a);
                b_vals.push(ok.b);
            }

            let mean_l = sum_l / 64.0;
            let mean_a = sum_a / 64.0;
            let mean_b = sum_b / 64.0;
            let std_l = (l_vals.iter().map(|v| (v - mean_l).powi(2)).sum::<f32>() / 64.0).sqrt();
            let std_a = (a_vals.iter().map(|v| (v - mean_a).powi(2)).sum::<f32>() / 64.0).sqrt();
            let std_b = (b_vals.iter().map(|v| (v - mean_b).powi(2)).sum::<f32>() / 64.0).sqrt();
            let oklab_stats = [
                mean_l,
                mean_a * 2.5,
                mean_b * 2.5,
                std_l,
                std_a * 2.5,
                std_b * 2.5,
            ];

            // Use the already-downscaled 32x18 thumbnail for perceptual hashing (~30x faster than hashing full frames)
            let rgba_img = RgbaImage::from_raw(32, 18, thumb_32x18.data.clone())
                .unwrap_or_else(|| RgbaImage::new(32, 18));
            let dyn_img = DynamicImage::ImageRgba8(rgba_img);
            let phash = hasher.hash_image(&dyn_img).as_bytes().to_vec();

            let motion_vec = if let Some(prev) = prev_frame {
                estimate_motion_phase_correlation(prev, &thumb_32x18)
            } else {
                [0.0, 0.0]
            };

            let mut emb = Vec::with_capacity(224);
            emb.extend_from_slice(&oklab_thumb);
            for &s in &oklab_stats {
                emb.push(s * 2.0);
            }
            emb.push(motion_vec[0] * 0.5);
            emb.push(motion_vec[1] * 0.5);

            if config.mode == SimilarityMode::Neural {
                let neural_feats = extract_neural_features(frame, config.model_path.as_deref());
                emb.extend_from_slice(&neural_feats);
            }

            l2_normalize(&mut emb);

            Some(FrameDescriptor {
                shot_id: shot.id,
                pts: frame.pts,
                oklab_thumb,
                oklab_stats,
                phash,
                motion_vec,
                mean_luma: frame.mean_luma(),
                combined_embedding: emb,
                ssim_thumb_rgba: thumb_32x18.data,
            })
        })
        .collect();

    let mut by_shot: Vec<Vec<FrameDescriptor>> = vec![Vec::new(); shots.len()];
    for desc in computed {
        if let Some(bucket) = by_shot.get_mut(desc.shot_id) {
            bucket.push(desc);
        }
    }

    // Ensure every shot has at least 1 descriptor even if very short
    for shot in shots {
        if by_shot[shot.id].is_empty()
            && let Some(nearest) = sampled_frames
                .iter()
                .min_by_key(|f| (f.pts - shot.usable_start).abs())
        {
            let mut emb = vec![shot.mean_luma; 200];
            l2_normalize(&mut emb);
            by_shot[shot.id].push(FrameDescriptor {
                shot_id: shot.id,
                pts: nearest.pts,
                oklab_thumb: vec![shot.mean_luma; 192],
                oklab_stats: [shot.mean_luma, 0.0, 0.0, 0.1, 0.0, 0.0],
                phash: vec![0u8; 8],
                motion_vec: [0.0, 0.0],
                mean_luma: shot.mean_luma,
                combined_embedding: emb,
                ssim_thumb_rgba: vec![128u8; 32 * 18 * 4],
            });
        }
    }

    Ok(ShotDescriptorBank {
        by_shot,
        oklab_z: ChannelZStats { mean: 0.5, std: 0.25 },
        stats_z: ChannelZStats { mean: 0.2, std: 0.15 },
        phash_z: ChannelZStats { mean: 0.5, std: 0.2 },
        motion_z: ChannelZStats { mean: 0.0, std: 0.1 },
    })
}

/// Estimate global camera translation `(dx, dy)` between consecutive frames using 1D FFT phase
/// correlation (`realfft`) on horizontal and vertical luma projections.
fn estimate_motion_phase_correlation(prev: &RgbaFrame, curr: &RgbaFrame) -> [f32; 2] {
    let w = prev.width.min(curr.width) as usize;
    let h = prev.height.min(curr.height) as usize;
    if w < 16 || h < 16 {
        return [0.0, 0.0];
    }

    let mut proj_x1 = vec![0.0f32; 64];
    let mut proj_x2 = vec![0.0f32; 64];
    let mut proj_y1 = vec![0.0f32; 32];
    let mut proj_y2 = vec![0.0f32; 32];

    for y in 0..h.min(32) {
        for x in 0..w.min(64) {
            let p1 = prev.pixel(x as u32, y as u32);
            let p2 = curr.pixel(x as u32, y as u32);
            let l1 = 0.2126 * (p1[0] as f32) + 0.7152 * (p1[1] as f32) + 0.0722 * (p1[2] as f32);
            let l2 = 0.2126 * (p2[0] as f32) + 0.7152 * (p2[1] as f32) + 0.0722 * (p2[2] as f32);
            proj_x1[x] += l1;
            proj_x2[x] += l2;
            proj_y1[y] += l1;
            proj_y2[y] += l2;
        }
    }

    let dx = phase_correlate_1d(&mut proj_x1, &mut proj_x2, 64) / 32.0;
    let dy = phase_correlate_1d(&mut proj_y1, &mut proj_y2, 32) / 16.0;
    [dx.clamp(-1.0, 1.0), dy.clamp(-1.0, 1.0)]
}

fn phase_correlate_1d(a: &mut [f32], b: &mut [f32], n: usize) -> f32 {
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n);
    let c2r = planner.plan_fft_inverse(n);

    let mut fa = r2c.make_output_vec();
    let mut fb = r2c.make_output_vec();
    let _ = r2c.process(a, &mut fa);
    let _ = r2c.process(b, &mut fb);

    let mut cross = r2c.make_output_vec();
    for i in 0..fa.len() {
        let prod = fa[i] * fb[i].conj();
        let norm = prod.norm().max(1e-6);
        cross[i] = prod / norm;
    }
    // Nyquist & DC imaginary parts must be 0 for c2r
    if let Some(first) = cross.first_mut() {
        first.im = 0.0;
    }
    if let Some(last) = cross.last_mut() {
        last.im = 0.0;
    }

    let mut corr = c2r.make_output_vec();
    let _ = c2r.process(&mut cross, &mut corr);

    let (best_idx, _) = corr
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    if best_idx > n / 2 {
        (best_idx as f32) - (n as f32)
    } else {
        best_idx as f32
    }
}

fn extract_neural_features(frame: &RgbaFrame, model_path: Option<&Path>) -> Vec<f32> {
    if let Some(path) = model_path
        && path.exists()
        && let Ok(v) = run_dinov2_ort(frame, path)
    {
        return v;
    }
    // Multi-scale spatial pyramid edge + color layout descriptor (24 dims)
    let mut feats = vec![0.0f32; 24];
    let w = frame.width.max(1);
    let h = frame.height.max(1);
    for y in 0..h {
        let qy = ((y * 2) / h).min(1) as usize;
        for x in 0..w {
            let qx = ((x * 2) / w).min(1) as usize;
            let px = frame.pixel(x, y);
            let q = (qy * 2 + qx) * 6;
            feats[q] += (px[0] as f32) / 255.0;
            feats[q + 1] += (px[1] as f32) / 255.0;
            feats[q + 2] += (px[2] as f32) / 255.0;
            if x + 1 < w && y + 1 < h {
                let px_r = frame.pixel(x + 1, y);
                let px_d = frame.pixel(x, y + 1);
                let gx = ((px_r[0] as f32) - (px[0] as f32)).abs() / 255.0;
                let gy = ((px_d[0] as f32) - (px[0] as f32)).abs() / 255.0;
                feats[q + 3] += gx;
                feats[q + 4] += gy;
                feats[q + 5] += (gx * gx + gy * gy).sqrt();
            }
        }
    }
    l2_normalize(&mut feats);
    feats
}

fn run_dinov2_ort(frame: &RgbaFrame, model_path: &Path) -> Result<Vec<f32>> {
    use ort::session::Session;
    use ort::value::TensorRef;

    let env = ort::init().build().map_err(|e| AeError::Model {
        model: "dinov2".into(),
        message: e.to_string(),
    })?;
    let mut session = Session::builder(&env)
        .map_err(|e| AeError::Model {
            model: "dinov2".into(),
            message: e.to_string(),
        })?
        .commit_from_file(model_path)
        .map_err(|e| AeError::Model {
            model: "dinov2".into(),
            message: e.to_string(),
        })?;

    let resized = stretch_frame_lanczos(frame, Resolution::new(224, 224))?;
    let mut input = Array4::<f32>::zeros((1, 3, 224, 224));
    for y in 0..224usize {
        for x in 0..224usize {
            let px = resized.pixel(x as u32, y as u32);
            input[[0, 0, y, x]] = ((px[0] as f32) / 255.0 - 0.485) / 0.229;
            input[[0, 1, y, x]] = ((px[1] as f32) / 255.0 - 0.456) / 0.224;
            input[[0, 2, y, x]] = ((px[2] as f32) / 255.0 - 0.406) / 0.225;
        }
    }

    let tensor = TensorRef::from_array_view(&input).map_err(|e| AeError::Model {
        model: "dinov2".into(),
        message: e.to_string(),
    })?;
    let outputs = session.run(ort::inputs![tensor]).map_err(|e| AeError::Model {
        model: "dinov2".into(),
        message: e.to_string(),
    })?;

    if let Some((_, val)) = outputs.iter().next()
        && let Ok((_, slice)) = val.try_extract_tensor::<f32>()
    {
        let mut v: Vec<f32> = slice.iter().take(64).copied().collect();
        l2_normalize(&mut v);
        return Ok(v);
    }

    Err(AeError::Model {
        model: "dinov2".into(),
        message: "empty embedding output".into(),
    })
}

pub fn l2_normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm > 1e-6 {
        for x in v {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ae_core::ShotBoundaryKind;

    #[test]
    fn test_similar_frames_score_higher_than_dissimilar() {
        let f_red_1 = RgbaFrame::solid(64, 36, [220, 30, 30, 255], Rational::from_secs(0));
        let f_red_2 = RgbaFrame::solid(64, 36, [215, 35, 32, 255], Rational::from_secs(1));
        let f_blue = RgbaFrame::solid(64, 36, [25, 40, 230, 255], Rational::from_secs(2));

        let shots = vec![
            Shot {
                id: 0,
                start: Rational::from_secs(0),
                end: Rational::from_millis(500),
                usable_start: Rational::from_secs(0),
                usable_end: Rational::from_millis(500),
                cut_confidence: 1.0,
                boundary_kind: ShotBoundaryKind::Initial,
                mean_luma: 0.3,
                enabled: true,
                locked: false,
            },
            Shot {
                id: 1,
                start: Rational::from_millis(500),
                end: Rational::from_millis(1500),
                usable_start: Rational::from_millis(500),
                usable_end: Rational::from_millis(1500),
                cut_confidence: 1.0,
                boundary_kind: ShotBoundaryKind::HardCut,
                mean_luma: 0.3,
                enabled: true,
                locked: false,
            },
            Shot {
                id: 2,
                start: Rational::from_millis(1500),
                end: Rational::from_millis(2500),
                usable_start: Rational::from_millis(1500),
                usable_end: Rational::from_millis(2500),
                cut_confidence: 1.0,
                boundary_kind: ShotBoundaryKind::HardCut,
                mean_luma: 0.3,
                enabled: true,
                locked: false,
            },
        ];

        let bank = build_descriptor_bank_from_frames(
            &[f_red_1, f_red_2, f_blue],
            &shots,
            &SimilarityConfig::default(),
        )
        .unwrap();

        let d0 = &bank.by_shot[0][0];
        let d1 = &bank.by_shot[1][0];
        let d2 = &bank.by_shot[2][0];

        let sim_red_red = bank.similarity(d0, d1);
        let sim_red_blue = bank.similarity(d0, d2);
        let ssim_red_red = bank.ms_ssim_similarity(d0, d1);
        let ssim_red_blue = bank.ms_ssim_similarity(d0, d2);

        assert!(sim_red_red > 0.90, "sim_red_red={sim_red_red}");
        assert!(sim_red_red > sim_red_blue + 0.25);
        assert!(ssim_red_red > ssim_red_blue);
    }
}
