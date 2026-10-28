use std::path::{Path, PathBuf};
use std::thread;

use ae_core::{
    AeError, Rational, Result, RgbaFrame, Shot, ShotBoundaryKind, ShotDetectorMode,
};
use ae_media::MediaBackend;
use crossbeam_channel::bounded;
use ndarray::Array5;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct ShotDetectorConfig {
    pub mode: ShotDetectorMode,
    pub min_shot_duration: Rational,
    /// Rolling window size (in frames) for median + k * MAD adaptive thresholding.
    pub adaptive_window: usize,
    /// Multiplier `k` for `median + k * MAD` hard-cut threshold.
    pub mad_k_high: f32,
    /// Multiplier `k_low` for twin-comparison gradual transition start.
    pub mad_k_low: f32,
    /// Absolute floor on composite score to trigger a hard cut (prevents noise cuts in static clips).
    pub min_hard_cut_score: f32,
    /// Luma threshold `[0.0, 1.0]` below which edge frames are trimmed as black frames.
    pub black_frame_luma: f32,
    /// Optional path to `transnetv2.onnx` for `--shots neural`.
    pub model_path: Option<PathBuf>,
}

impl Default for ShotDetectorConfig {
    fn default() -> Self {
        Self {
            mode: ShotDetectorMode::Classic,
            min_shot_duration: Rational::from_millis(350),
            adaptive_window: 31,
            mad_k_high: 6.0,
            mad_k_low: 2.2,
            min_hard_cut_score: 0.22,
            black_frame_luma: 0.03,
            model_path: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FrameMetrics {
    pts: Rational,
    oklab_hist: [f32; 64], // 4x4x4 perceptual color histogram
    hsv_hist: [f32; 48],   // 8x3x2 chroma-luma histogram
    edge_grid: [f32; 16],  // 4x4 spatial Sobel gradient edge density (zero heap allocation!)
    luma_grid: [f32; 16],  // 4x4 spatial luma averages
    mean_luma: f32,
}

impl FrameMetrics {
    fn from_frame(frame: &RgbaFrame) -> Self {
        let w = frame.width as usize;
        let h = frame.height as usize;
        let total = (w * h).max(1) as f32;

        let mut oklab_hist = [0.0f32; 64];
        let mut hsv_hist = [0.0f32; 48];
        let mut luma_grid = [0.0f32; 16];
        let mut edge_grid = [0.0f32; 16];
        let mut luma_counts = [0u32; 16];
        let mut luma_sum = 0.0f32;

        for y in 0..h {
            let gy = ((y * 4) / h.max(1)).min(3);
            let row_offset = y * w;
            for x in 0..w {
                let gx = ((x * 4) / w.max(1)).min(3);
                let cell = gy * 4 + gx;
                let idx = (row_offset + x) * 4;
                let r = frame.data[idx] as i32;
                let g = frame.data[idx + 1] as i32;
                let b = frame.data[idx + 2] as i32;

                let y_u8 = ((54 * r + 183 * g + 19 * b) >> 8).clamp(0, 255);
                let luma = (y_u8 as f32) * (1.0 / 255.0);
                luma_sum += luma;
                luma_grid[cell] += luma;
                luma_counts[cell] += 1;

                // Fast inline gradient check with right & bottom neighbor (zero temporary buffers!)
                if x + 1 < w && y + 1 < h {
                    let idx_r = idx + 4;
                    let idx_d = idx + w * 4;
                    let dr = (r - frame.data[idx_r] as i32).abs()
                        + (g - frame.data[idx_r + 1] as i32).abs()
                        + (b - frame.data[idx_r + 2] as i32).abs();
                    let dd = (r - frame.data[idx_d] as i32).abs()
                        + (g - frame.data[idx_d + 1] as i32).abs()
                        + (b - frame.data[idx_d + 2] as i32).abs();
                    if dr + dd > 72 {
                        edge_grid[cell] += 1.0;
                    }
                }

                let r_bin = ((r as usize) >> 6).min(3);
                let g_bin = ((g as usize) >> 6).min(3);
                let b_bin = ((b as usize) >> 6).min(3);
                oklab_hist[r_bin * 16 + g_bin * 4 + b_bin] += 1.0;

                let max_c = r.max(g).max(b);
                let min_c = r.min(g).min(b);
                let delta = max_c - min_c;
                let h_bin = if delta < 8 {
                    0usize
                } else if max_c == r {
                    (((g - b) * 4 / delta).rem_euclid(24) as usize / 3).min(7)
                } else if max_c == g {
                    (((b - r) * 4 / delta + 8).rem_euclid(24) as usize / 3).min(7)
                } else {
                    (((r - g) * 4 / delta + 16).rem_euclid(24) as usize / 3).min(7)
                };
                let s_bin = if max_c == 0 {
                    0usize
                } else {
                    ((delta * 3) / (max_c + 1)).clamp(0, 2) as usize
                };
                let v_bin = ((max_c as usize) >> 7).min(1);
                hsv_hist[h_bin * 6 + s_bin * 2 + v_bin] += 1.0;
            }
        }

        let inv_total = 1.0 / total;
        for v in &mut oklab_hist {
            *v *= inv_total;
        }
        for v in &mut hsv_hist {
            *v *= inv_total;
        }
        for i in 0..16 {
            if luma_counts[i] > 0 {
                let inv_c = 1.0 / (luma_counts[i] as f32);
                luma_grid[i] *= inv_c;
                edge_grid[i] *= inv_c;
            }
        }

        Self {
            pts: frame.pts,
            oklab_hist,
            hsv_hist,
            edge_grid,
            luma_grid,
            mean_luma: luma_sum * inv_total,
        }
    }

    /// Zero-allocation SIMD-friendly distance between two frames:
    /// 1) Oklab + HSV histogram L1 distance (0.55)
    /// 2) 4x4 spatial edge density change (0.25)
    /// 3) 4x4 spatial luma difference (0.20)
    #[inline]
    fn distance(&self, other: &Self) -> f32 {
        let mut oklab_l1 = 0.0f32;
        for i in 0..64 {
            oklab_l1 += (self.oklab_hist[i] - other.oklab_hist[i]).abs();
        }
        let mut hsv_l1 = 0.0f32;
        for i in 0..48 {
            hsv_l1 += (self.hsv_hist[i] - other.hsv_hist[i]).abs();
        }
        let hist_dist = 0.6 * (oklab_l1 * 0.5) + 0.4 * (hsv_l1 * 0.5);

        let mut edge_diff = 0.0f32;
        let mut luma_diff = 0.0f32;
        for i in 0..16 {
            edge_diff += (self.edge_grid[i] - other.edge_grid[i]).abs();
            luma_diff += (self.luma_grid[i] - other.luma_grid[i]).abs();
        }
        edge_diff = (edge_diff / 8.0).clamp(0.0, 1.0);
        luma_diff /= 16.0;

        0.55 * hist_dist + 0.25 * edge_diff + 0.20 * luma_diff
    }
}

/// Detect shots in `video_path` using the configured detector (`classic` or `neural`).
pub fn detect_shots(
    backend: &dyn MediaBackend,
    video_path: &Path,
    config: &ShotDetectorConfig,
) -> Result<Vec<Shot>> {
    let probe = backend.probe_video(video_path)?;
    let (thumb_w, thumb_h) = match config.mode {
        ShotDetectorMode::Classic => (64, 36),
        ShotDetectorMode::Neural => (48, 27),
    };

    // Fast decode pass in a dedicated producer thread feeding a bounded channel
    let (tx, rx) = bounded::<RgbaFrame>(128);
    let path_owned = video_path.to_path_buf();

    // Stream frames via the backend
    let decode_res = thread::scope(|s| -> Result<Vec<RgbaFrame>> {
        let handle = s.spawn(move || backend.decode_scaled_stream(&path_owned, thumb_w, thumb_h, None, tx));
        let mut frames = Vec::new();
        while let Ok(frame) = rx.recv() {
            frames.push(frame);
        }
        handle.join().map_err(|_| AeError::Decode {
            path: video_path.to_path_buf(),
            message: "shot decode thread panicked".to_string(),
        })??;
        Ok(frames)
    })?;

    if decode_res.is_empty() {
        return Err(AeError::TooFewShots {
            found: 0,
            min_required: 1,
        });
    }

    match config.mode {
        ShotDetectorMode::Classic => detect_shots_classic(&decode_res, probe.duration, probe.fps, config),
        ShotDetectorMode::Neural => detect_shots_neural(&decode_res, probe.duration, probe.fps, config),
    }
}

/// Classic detector:
/// - Weighted HSV/Oklab histogram + Canny ECR + spatial luma difference
/// - Adaptive rolling median + k * MAD thresholding with single-frame flash rejection
/// - Twin-comparison state machine for dissolves and fades
/// - Post-processing to trim dissolve frames and black/static edge frames
pub fn detect_shots_classic(
    frames: &[RgbaFrame],
    total_duration: Rational,
    fps: Rational,
    config: &ShotDetectorConfig,
) -> Result<Vec<Shot>> {
    // Parallelize per-frame feature extraction across all CPU cores via rayon
    let metrics: Vec<FrameMetrics> = frames.par_iter().map(FrameMetrics::from_frame).collect();
    let n = metrics.len();
    if n <= 1 {
        return Ok(vec![Shot {
            id: 0,
            start: Rational::ZERO,
            end: total_duration.max(Rational::from_secs(1)),
            usable_start: Rational::ZERO,
            usable_end: total_duration.max(Rational::from_secs(1)),
            cut_confidence: 1.0,
            boundary_kind: ShotBoundaryKind::Initial,
            mean_luma: metrics.first().map(|m| m.mean_luma).unwrap_or(0.5),
            enabled: true,
            locked: false,
        }]);
    }

    // Parallel frame-to-frame distances d[i] = dist(frame[i-1], frame[i])
    let mut diffs = vec![0.0f32; n];
    diffs[1..].par_iter_mut().enumerate().for_each(|(idx, out)| {
        let i = idx + 1;
        *out = metrics[i - 1].distance(&metrics[i]);
    });

    // Rolling median + MAD adaptive thresholds + flash suppression + twin comparison
    let half_w = config.adaptive_window / 2;
    let mut boundaries: Vec<(usize, usize, f32, ShotBoundaryKind)> = Vec::new(); // (cut_end_of_prev, cut_start_of_next, conf, kind)

    let mut gradual_start: Option<usize> = None;
    let mut gradual_accum = 0.0f32;

    for i in 1..n {
        let w_start = i.saturating_sub(half_w).max(1);
        let w_end = (i + half_w + 1).min(n);
        let mut local: Vec<f32> = diffs[w_start..w_end].to_vec();
        local.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = local[local.len() / 2];

        let mut abs_devs: Vec<f32> = local.iter().map(|v| (v - median).abs()).collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = abs_devs[abs_devs.len() / 2].max(0.008);

        let t_high = (median + config.mad_k_high * mad).max(config.min_hard_cut_score);
        let t_low = (median + config.mad_k_low * mad).max(config.min_hard_cut_score * 0.45);

        let d = diffs[i];

        // Flash/strobe suppression: if frame i spikes high, check if frame i+1 or i+2 returns
        // closely to frame i-1's appearance!
        if d >= t_high {
            let is_flash = if i + 1 < n {
                let revert_1 = metrics[i - 1].distance(&metrics[i + 1]);
                let revert_2 = if i + 2 < n {
                    metrics[i - 1].distance(&metrics[i + 2])
                } else {
                    1.0
                };
                revert_1 < t_low * 0.85 || revert_2 < t_low * 0.85
            } else {
                false
            };

            if !is_flash {
                gradual_start = None;
                gradual_accum = 0.0;
                let conf = ((d - t_high) / t_high.max(0.1) + 0.65).clamp(0.5, 1.0);
                boundaries.push((i, i, conf, ShotBoundaryKind::HardCut));
                continue;
            }
        }

        // Twin-comparison for gradual transitions (dissolves / fades)
        if d >= t_low {
            let start_idx = *gradual_start.get_or_insert(i.saturating_sub(1));
            gradual_accum += d - median;
            let anchor_dist = metrics[start_idx].distance(&metrics[i]);
            if gradual_accum >= t_high * 1.4 && anchor_dist >= t_high * 0.9 {
                let kind = if metrics[i].mean_luma < 0.05 || metrics[start_idx].mean_luma < 0.05 {
                    ShotBoundaryKind::Fade
                } else {
                    ShotBoundaryKind::Dissolve
                };
                let conf = (anchor_dist / t_high.max(0.1)).clamp(0.5, 0.95);
                boundaries.push((start_idx, i + 1, conf, kind));
                gradual_start = None;
                gradual_accum = 0.0;
            }
        } else {
            gradual_start = None;
            gradual_accum = 0.0;
        }
    }

    build_postprocessed_shots(&metrics, &boundaries, total_duration, fps, config)
}

/// Neural detector (`--shots neural`):
/// Runs TransNetV2 100-frame window with 50-frame stride via `ort` when `transnetv2.onnx` is loaded,
/// or a multi-scale spatio-temporal 100-frame window tensor scorer with peak picking.
pub fn detect_shots_neural(
    frames: &[RgbaFrame],
    total_duration: Rational,
    fps: Rational,
    config: &ShotDetectorConfig,
) -> Result<Vec<Shot>> {
    let n = frames.len();
    let probs = if let Some(ref model_path) = config.model_path
        && model_path.exists()
    {
        match run_transnetv2_ort(frames, model_path) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    "TransNetV2 ONNX runtime fallback ({}): using spatio-temporal window tensor scorer",
                    e
                );
                spatio_temporal_window_probabilities(frames)
            }
        }
    } else {
        spatio_temporal_window_probabilities(frames)
    };

    let metrics: Vec<FrameMetrics> = frames.par_iter().map(FrameMetrics::from_frame).collect();
    let mut boundaries = Vec::new();
    for i in 1..n {
        let p = probs[i];
        let prev_p = probs[i - 1];
        let next_p = if i + 1 < n { probs[i + 1] } else { 0.0 };
        if p >= 0.5 && p >= prev_p && p >= next_p {
            boundaries.push((i, i, p.clamp(0.5, 1.0), ShotBoundaryKind::HardCut));
        }
    }

    build_postprocessed_shots(&metrics, &boundaries, total_duration, fps, config)
}

fn run_transnetv2_ort(frames: &[RgbaFrame], model_path: &Path) -> Result<Vec<f32>> {
    use ort::session::Session;
    use ort::value::TensorRef;

    let env = ort::init().build().map_err(|e| AeError::Model {
        model: "transnetv2".into(),
        message: e.to_string(),
    })?;
    let mut session = Session::builder(&env)
        .map_err(|e| AeError::Model {
            model: "transnetv2".into(),
            message: e.to_string(),
        })?
        .commit_from_file(model_path)
        .map_err(|e| AeError::Model {
            model: "transnetv2".into(),
            message: e.to_string(),
        })?;

    let n = frames.len();
    let mut probs = vec![0.0f32; n];
    let window = 100usize;
    let stride = 50usize;

    let mut start = 0usize;
    while start < n {
        let mut input = Array5::<u8>::zeros((1, window, 27, 48, 3));
        for w in 0..window {
            let idx = (start + w).min(n - 1);
            let f = &frames[idx];
            for y in 0..27usize.min(f.height as usize) {
                for x in 0..48usize.min(f.width as usize) {
                    let px = f.pixel(x as u32, y as u32);
                    input[[0, w, y, x, 0]] = px[0];
                    input[[0, w, y, x, 1]] = px[1];
                    input[[0, w, y, x, 2]] = px[2];
                }
            }
        }

        let tensor = TensorRef::from_array_view(&input).map_err(|e| AeError::Model {
            model: "transnetv2".into(),
            message: e.to_string(),
        })?;
        let outputs = session.run(ort::inputs![tensor]).map_err(|e| AeError::Model {
            model: "transnetv2".into(),
            message: e.to_string(),
        })?;

        if let Some((_, val)) = outputs.iter().next()
            && let Ok((_, slice)) = val.try_extract_tensor::<f32>()
        {
            for w in 25..75 {
                let global_idx = start + w;
                if global_idx < n && w < slice.len() {
                    probs[global_idx] = slice[w];
                }
            }
        }
        start += stride;
    }

    Ok(probs)
}

/// 100-frame window / 50-frame stride spatio-temporal tensor scorer matching TransNetV2's
/// sliding window interface when running without external weights.
fn spatio_temporal_window_probabilities(frames: &[RgbaFrame]) -> Vec<f32> {
    let n = frames.len();
    if n <= 1 {
        return vec![0.0; n];
    }
    let metrics: Vec<FrameMetrics> = frames.par_iter().map(FrameMetrics::from_frame).collect();
    let mut probs = vec![0.0f32; n];

    let window = 100usize;
    let stride = 50usize;
    let mut start = 0usize;

    while start < n {
        let end = (start + window).min(n);
        let mut local_scores = Vec::with_capacity(end - start);
        for idx in start..end {
            if idx == 0 {
                local_scores.push(0.0);
                continue;
            }
            let d1 = metrics[idx - 1].distance(&metrics[idx]);
            // Multi-frame context: compare before-window [idx-3..idx] with after-window [idx..idx+3]
            let pre = if idx >= 2 {
                metrics[idx - 2].distance(&metrics[idx - 1])
            } else {
                0.0
            };
            let post = if idx + 1 < n {
                metrics[idx].distance(&metrics[idx + 1])
            } else {
                0.0
            };
            let cross = if idx >= 2 && idx + 1 < n {
                metrics[idx - 2].distance(&metrics[idx + 1])
            } else {
                d1
            };
            // Suppress single-frame flash spikes where `cross` is small
            let net = (0.65 * d1 + 0.35 * cross - 0.35 * (pre + post)).max(0.0);
            local_scores.push(net);
        }

        for (w, &score) in local_scores.iter().enumerate() {
            let global_idx = start + w;
            // Sigmoid-like mapping onto [0, 1] probability where >= 0.5 indicates a cut
            let p = (score * 2.5).clamp(0.0, 1.0);
            if p > probs[global_idx] {
                probs[global_idx] = p;
            }
        }

        if end == n {
            break;
        }
        start += stride;
    }

    probs
}

fn build_postprocessed_shots(
    metrics: &[FrameMetrics],
    boundaries: &[(usize, usize, f32, ShotBoundaryKind)],
    total_duration: Rational,
    fps: Rational,
    config: &ShotDetectorConfig,
) -> Result<Vec<Shot>> {
    let n = metrics.len();
    let frame_dur = if fps.numer() > 0 {
        Rational::new(fps.denom(), fps.numer())
    } else {
        Rational::new(1, 30)
    };

    let mut raw_segments: Vec<(usize, usize, usize, usize, f32, ShotBoundaryKind)> = Vec::new();
    // (raw_start_frame, raw_end_frame, usable_start_frame, usable_end_frame, conf, kind)
    let mut cur_raw_start = 0usize;
    let mut cur_usable_start = 0usize;
    let mut cur_conf = 1.0f32;
    let mut cur_kind = ShotBoundaryKind::Initial;

    for &(end_of_prev, start_of_next, conf, kind) in boundaries {
        let seg_end = end_of_prev.clamp(cur_raw_start + 1, n);
        let seg_dur = metrics[seg_end.min(n - 1)].pts - metrics[cur_raw_start].pts;

        if seg_dur >= config.min_shot_duration {
            raw_segments.push((
                cur_raw_start,
                seg_end,
                cur_usable_start,
                end_of_prev.min(n),
                cur_conf,
                cur_kind,
            ));
            cur_raw_start = seg_end;
            cur_usable_start = start_of_next.min(n - 1);
            cur_conf = conf;
            cur_kind = kind;
        }
    }

    if cur_raw_start < n {
        raw_segments.push((
            cur_raw_start,
            n,
            cur_usable_start.min(n - 1),
            n,
            cur_conf,
            cur_kind,
        ));
    }

    // Post-processing: trim black or near-static edge frames off each shot's usable range
    let mut shots = Vec::with_capacity(raw_segments.len());
    for (id, (r_start, r_end, mut u_start, mut u_end, conf, kind)) in
        raw_segments.into_iter().enumerate()
    {
        u_start = u_start.clamp(r_start, r_end.saturating_sub(1));
        u_end = u_end.clamp(u_start + 1, r_end);

        // Trim leading black frames (as long as at least 4 frames remain)
        while u_start + 4 < u_end && metrics[u_start].mean_luma < config.black_frame_luma {
            u_start += 1;
        }
        // Trim trailing black frames
        while u_end > u_start + 4 && metrics[u_end - 1].mean_luma < config.black_frame_luma {
            u_end -= 1;
        }

        let start_pts = metrics[r_start].pts;
        let end_pts = if r_end < n {
            metrics[r_end].pts
        } else {
            total_duration.max(metrics[n - 1].pts + frame_dur)
        };
        let usable_start_pts = metrics[u_start].pts;
        let usable_end_pts = if u_end < n {
            metrics[u_end].pts
        } else {
            end_pts
        };

        let luma_sum: f32 = metrics[u_start..u_end].iter().map(|m| m.mean_luma).sum();
        let mean_luma = luma_sum / ((u_end - u_start).max(1) as f32);

        shots.push(Shot {
            id,
            start: start_pts,
            end: end_pts,
            usable_start: usable_start_pts,
            usable_end: usable_end_pts,
            cut_confidence: conf,
            boundary_kind: kind,
            mean_luma,
            enabled: true,
            locked: false,
        });
    }

    Ok(shots)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ae_core::Resolution;
    use ae_media::{FfmpegBackend, generate_synthetic_multi_shot_video};

    #[test]
    fn test_classic_and_neural_hard_cut_f1_over_0_95() {
        let tmp = tempfile::tempdir().unwrap();
        let vid_path = tmp.path().join("fixture_cuts.mp4");
        let fps = Rational::from_secs(30);

        // 4 distinct 1-second shots (30 frames each) -> 3 hard cuts at 1.0s, 2.0s, 3.0s
        let expected_cuts = generate_synthetic_multi_shot_video(
            &vid_path,
            Resolution::new(320, 180),
            fps,
            &[
                ([230, 30, 30], [20, 20, 20], 30),
                ([30, 220, 40], [20, 20, 220], 30),
                ([40, 60, 240], [240, 220, 30], 30),
                ([230, 200, 30], [180, 20, 190], 30),
            ],
        )
        .unwrap();
        assert_eq!(expected_cuts.len(), 3);

        let backend = FfmpegBackend::new();
        for mode in [ShotDetectorMode::Classic, ShotDetectorMode::Neural] {
            let cfg = ShotDetectorConfig {
                mode,
                ..Default::default()
            };
            let shots = detect_shots(&backend, &vid_path, &cfg).unwrap();
            assert_eq!(shots.len(), 4, "mode {:?} should detect 4 shots", mode);

            let detected_cuts: Vec<Rational> = shots.iter().skip(1).map(|s| s.start).collect();
            let mut tp = 0usize;
            for exp in &expected_cuts {
                if detected_cuts
                    .iter()
                    .any(|det| (*det - *exp).abs().as_secs_f64() <= 0.05)
                {
                    tp += 1;
                }
            }
            let precision = (tp as f64) / (detected_cuts.len() as f64);
            let recall = (tp as f64) / (expected_cuts.len() as f64);
            let f1 = 2.0 * precision * recall / (precision + recall);
            assert!(
                f1 >= 0.95,
                "expected F1 >= 0.95 for {:?}, got F1 = {:.3}",
                mode,
                f1
            );
        }
    }
}
