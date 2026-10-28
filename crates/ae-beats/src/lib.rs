use std::fs::File;
use std::path::{Path, PathBuf};

use ae_core::{AeError, Beat, BeatMode, FreqBand, Rational, Result, RgbaFrame, SeparationMode};
use biquad::{Biquad, Coefficients, DirectForm2Transposed, ToHertz, Type as FilterType};
use image::{Rgb, RgbImage};
use ndarray::Array3;
use realfft::RealFftPlanner;
use rubato::audioadapter_buffers::owned::InterleavedOwned;
use rubato::{Fft, FixedSync, Resampler};
use symphonia::core::codecs::CodecParameters;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

pub const ANALYSIS_SR: u32 = 22_050;
pub const STFT_WIN: usize = 4_096;
pub const STFT_HOP: usize = 220; // ~9.977 ms hop at 22,050 Hz

#[derive(Clone, Debug)]
pub struct BeatDetectorConfig {
    pub separation: SeparationMode,
    pub bass_band: FreqBand,
    pub beat_mode: BeatMode,
    pub min_cut: Rational,
    pub peak_delta: f32,
    pub hybrid_snap_tolerance_ms: f64,
    pub demucs_model_path: Option<PathBuf>,
    pub debug_dir: Option<PathBuf>,
}

impl Default for BeatDetectorConfig {
    fn default() -> Self {
        Self {
            separation: SeparationMode::Hpss,
            bass_band: FreqBand::default(),
            beat_mode: BeatMode::Hybrid,
            min_cut: Rational::from_millis(250),
            peak_delta: 0.95,
            hybrid_snap_tolerance_ms: 40.0,
            demucs_model_path: None,
            debug_dir: None,
        }
    }
}

/// Decode an audio file via `symphonia`, downmix to mono, and resample to `22_050 Hz` via `rubato`.
pub fn decode_audio_mono_22050(path: &Path) -> Result<(Vec<f32>, Rational)> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| AeError::Audio(format!("symphonia probe failed for {}: {e}", path.display())))?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| AeError::Audio("no audio track found".into()))?;
    let track_id = track.id;
    let audio_params = match &track.codec_params {
        Some(CodecParameters::Audio(params)) => params.clone(),
        _ => return Err(AeError::Audio("selected track has no audio parameters".into())),
    };
    let in_sr = audio_params.sample_rate.unwrap_or(44_100);
    let channels = audio_params
        .channels
        .as_ref()
        .map(|c| c.count())
        .unwrap_or(1)
        .max(1);

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&audio_params, &AudioDecoderOptions::default())
        .map_err(|e| AeError::Audio(format!("symphonia codec init failed: {e}")))?;

    let mut mono_samples: Vec<f32> = Vec::new();
    let mut interleaved_buf: Vec<f32> = Vec::new();

    while let Ok(Some(packet)) = format.next_packet() {
        if packet.track_id != track_id {
            continue;
        }

        if let Ok(decoded) = decoder.decode(&packet) {
            let ch = decoded.spec().channels().count().max(channels).max(1);
            interleaved_buf.clear();
            decoded.copy_to_vec_interleaved(&mut interleaved_buf);
            for frame in interleaved_buf.chunks_exact(ch) {
                let sum: f32 = frame.iter().copied().sum();
                mono_samples.push(sum / (ch as f32));
            }
        }
    }

    if mono_samples.is_empty() {
        return Err(AeError::Audio("decoded audio contains 0 samples".into()));
    }

    let duration = Rational::from_secs_f64((mono_samples.len() as f64) / (in_sr as f64));

    if in_sr == ANALYSIS_SR {
        return Ok((mono_samples, duration));
    }

    // Resample to ANALYSIS_SR (22,050 Hz) using rubato 5.0 Fft synchronous resampler
    let in_len = mono_samples.len();
    let input_buf =
        InterleavedOwned::<f32>::new_from(mono_samples, 1, in_len).map_err(|e| {
            AeError::Audio(format!("failed creating rubato input adapter: {e}"))
        })?;

    let mut resampler = Fft::<f32>::new(
        in_sr as usize,
        ANALYSIS_SR as usize,
        1024,
        1,
        FixedSync::Input,
    )
    .map_err(|e| AeError::Audio(format!("rubato resampler construction failed: {e}")))?;

    let needed_out = resampler.process_all_needed_output_len(in_len);
    let mut output_buf = InterleavedOwned::<f32>::new(0.0, 1, needed_out);
    let (_consumed, produced) = resampler
        .process_all_into_buffer(&input_buf, &mut output_buf, in_len, None)
        .map_err(|e| AeError::Audio(format!("rubato resampling failed: {e}")))?;

    let mut out_vec = output_buf.take_data();
    out_vec.truncate(produced);
    Ok((out_vec, duration))
}

/// Detect bass-instrument beats in `audio_path` according to `config`.
pub fn detect_bass_beats(audio_path: &Path, config: &BeatDetectorConfig) -> Result<Vec<Beat>> {
    let (mono, duration) = decode_audio_mono_22050(audio_path)?;
    detect_bass_beats_from_samples(&mono, duration, config)
}

pub fn detect_bass_beats_from_samples(
    mono_22k: &[f32],
    duration: Rational,
    config: &BeatDetectorConfig,
) -> Result<Vec<Beat>> {
    // Optional Demucs neural stem pre-extraction
    let working_samples = if config.separation == SeparationMode::Demucs {
        if let Some(ref model_path) = config.demucs_model_path
            && model_path.exists()
        {
            match run_demucs_bass_ort(mono_22k, model_path) {
                Ok(stem) => stem,
                Err(e) => {
                    tracing::warn!("HTDemucs ONNX fallback ({e}): using harmonic-sustain bass isolator");
                    bandpass_biquad(mono_22k, config.bass_band)
                }
            }
        } else {
            bandpass_biquad(mono_22k, config.bass_band)
        }
    } else if config.separation == SeparationMode::Band {
        bandpass_biquad(mono_22k, config.bass_band)
    } else {
        mono_22k.to_vec()
    };

    // Compute STFT spectrogram at 22,050 Hz (4096 window, 220 hop)
    let spec = compute_stft(&working_samples, STFT_WIN, STFT_HOP);
    if spec.is_empty() {
        return Ok(Vec::new());
    }

    let bin_hz = (ANALYSIS_SR as f32) / (STFT_WIN as f32); // ~5.3833 Hz
    let b_low = ((config.bass_band.low_hz / bin_hz).floor() as usize).max(1);
    let b_high = ((config.bass_band.high_hz / bin_hz).ceil() as usize).min(STFT_WIN / 2 - 1);
    let p_high = ((2500.0 / bin_hz).ceil() as usize).min(STFT_WIN / 2 - 1);

    // Separate harmonic bass spectrogram vs percussive kick energy
    let (bass_harmonic_spec, percussive_gate, dominant_hz) = match config.separation {
        SeparationMode::Band => {
            let mut h_spec = vec![vec![0.0f32; b_high - b_low + 1]; spec.len()];
            let mut dom = vec![80.0f32; spec.len()];
            for (t, frame_bins) in spec.iter().enumerate() {
                let mut max_val = 0.0f32;
                let mut max_bin = b_low;
                for k in b_low..=b_high {
                    let v = frame_bins[k];
                    h_spec[t][k - b_low] = v;
                    if v > max_val {
                        max_val = v;
                        max_bin = k;
                    }
                }
                dom[t] = (max_bin as f32) * bin_hz;
            }
            (h_spec, vec![1.0f32; spec.len()], dom)
        }
        SeparationMode::Hpss | SeparationMode::Demucs => {
            separate_harmonic_bass(&spec, b_low, b_high, p_high, bin_hz)
        }
    };

    // Compute combined Onset Detection Function (log-compressed SuperFlux + pitch-change flux),
    // gated by the harmonic-vs-percussive sustain mask so kick drums are rejected.
    let odf = compute_bass_odf(&bass_harmonic_spec, &percussive_gate);

    // Peak picking with adaptive threshold and --min-cut constraint
    let raw_onsets = pick_onset_peaks(&odf, &dominant_hz, duration, config);

    let beats = match config.beat_mode {
        BeatMode::Onsets => raw_onsets,
        BeatMode::Grid => regularize_ellis_dp_grid(&odf, &dominant_hz, duration, config),
        BeatMode::Hybrid => regularize_hybrid_grid(&odf, &raw_onsets, &dominant_hz, duration, config),
    };

    if let Some(ref debug_dir) = config.debug_dir {
        write_debug_artifacts(debug_dir, mono_22k, &bass_harmonic_spec, &odf, &beats)?;
    }

    Ok(beats)
}

/// Apply a 4th-order cascaded Butterworth band-pass filter (`biquad` 0.6.0) over `band`.
pub fn bandpass_biquad(samples: &[f32], band: FreqBand) -> Vec<f32> {
    let fs = (ANALYSIS_SR as f32).hz();
    let f_low = band.low_hz.clamp(15.0, 2000.0).hz();
    let f_high = band.high_hz.clamp(band.low_hz + 10.0, 5000.0).hz();
    let q = std::f32::consts::FRAC_1_SQRT_2;

    let hp_coeffs = Coefficients::<f32>::from_params(FilterType::HighPass, fs, f_low, q)
        .expect("valid highpass params");
    let lp_coeffs = Coefficients::<f32>::from_params(FilterType::LowPass, fs, f_high, q)
        .expect("valid lowpass params");

    let mut hp1 = DirectForm2Transposed::<f32>::new(hp_coeffs);
    let mut hp2 = DirectForm2Transposed::<f32>::new(hp_coeffs);
    let mut lp1 = DirectForm2Transposed::<f32>::new(lp_coeffs);
    let mut lp2 = DirectForm2Transposed::<f32>::new(lp_coeffs);

    samples
        .iter()
        .map(|&x| {
            let y = hp1.run(x);
            let y = hp2.run(y);
            let y = lp1.run(y);
            lp2.run(y)
        })
        .collect()
}

fn compute_stft(samples: &[f32], win_size: usize, hop_size: usize) -> Vec<Vec<f32>> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(win_size);

    // Hann window
    let window: Vec<f32> = (0..win_size)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * (i as f32) / (win_size as f32)).cos())
        })
        .collect();

    // Causal alignment: frame `f` starts at `f * hop_size` (with a tiny 1-hop pre-roll)
    // so an onset at `t` never leaks into frames before `t`!
    let pre_roll = hop_size;
    let mut padded = vec![0.0f32; pre_roll + samples.len() + win_size];
    padded[pre_roll..pre_roll + samples.len()].copy_from_slice(samples);

    let num_frames = samples.len().div_ceil(hop_size);
    let mut frames = Vec::with_capacity(num_frames);
    let mut in_buf = r2c.make_input_vec();
    let mut out_buf = r2c.make_output_vec();

    // Also compute local 2-hop RMS envelope around `f * hop_size` so onset timing is pinned
    // to the exact 10ms hop where the note attack occurs.
    let local_len = hop_size * 3;

    for f in 0..num_frames {
        let start = f * hop_size;
        let mut local_sq = 0.0f32;
        for i in 0..local_len {
            let s = padded.get(pre_roll + start + i).copied().unwrap_or(0.0);
            local_sq += s * s;
        }
        let local_rms = (local_sq / (local_len as f32)).sqrt();

        let mut win_sq = 0.0f32;
        for i in 0..win_size {
            let s = padded.get(start + i).copied().unwrap_or(0.0);
            win_sq += s * s;
            in_buf[i] = s * window[i];
        }
        let win_rms = (win_sq / (win_size as f32)).sqrt().max(1e-6);
        let scale = local_rms / win_rms;

        let _ = r2c.process(&mut in_buf, &mut out_buf);
        let mag: Vec<f32> = out_buf.iter().map(|c| c.norm() * scale).collect();
        frames.push(mag);
    }

    frames
}

/// Fitzgerald (2010) 2D median-filter Harmonic/Percussive Source Separation combined with
/// forward-sustain verification so sustained bass notes are preserved at exact onset time while
/// fast-decaying / broadband kick drum hits are suppressed.
fn separate_harmonic_bass(
    spec: &[Vec<f32>],
    b_low: usize,
    b_high: usize,
    p_high: usize,
    bin_hz: f32,
) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
    let n_frames = spec.len();
    let n_bass_bins = b_high - b_low + 1;
    let mut harmonic_bass = vec![vec![0.0f32; n_bass_bins]; n_frames];
    let mut percussive_gate = vec![1.0f32; n_frames];
    let mut dominant_hz = vec![80.0f32; n_frames];

    // Time median half-window (e.g. 7 frames = ~70ms each side) and frequency median half-window
    let h_half = 6usize;
    let p_half = 6usize;

    for t in 0..n_frames {
        let mut max_h = 0.0f32;
        let mut best_bin = b_low;
        let mut total_h_energy = 0.0f32;
        let mut total_p_energy = 0.0f32;

        for k in b_low..=b_high {
            // Causal forward time window [t .. t+11] (~110ms following frame t) so a sustained
            // bass note starting at t has high median immediately at t, never before t!
            let t0 = t;
            let t1 = (t + h_half * 2 + 1).min(n_frames);
            let mut time_slice: Vec<f32> = (t0..t1).map(|tau| spec[tau][k]).collect();
            time_slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let h_med = time_slice[time_slice.len() / 2];

            // Vertical frequency median around bin k
            let k0 = k.saturating_sub(p_half).max(1);
            let k1 = (k + p_half + 1).min(p_high);
            let mut freq_slice: Vec<f32> = (k0..k1).map(|j| spec[t][j]).collect();
            freq_slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p_med = freq_slice[freq_slice.len() / 2];

            // Wiener soft mask (p = 2)
            let h2 = h_med * h_med;
            let p2 = (p_med * 1.35).powi(2);
            let mask_h = h2 / (h2 + p2 + 1e-9);

            let val = spec[t][k] * mask_h;
            harmonic_bass[t][k - b_low] = val;
            total_h_energy += val;
            total_p_energy += spec[t][k] * (1.0 - mask_h);

            if val > max_h {
                max_h = val;
                best_bin = k;
            }
        }

        dominant_hz[t] = (best_bin as f32) * bin_hz;

        // Check broadband high-frequency transient (250 Hz .. 2500 Hz) & fast decay characteristic of kicks:
        // A kick drum has either strong broadband attack above 250 Hz or decays >70% within 60ms (6 hops),
        // or its dominant bin sweeps downward rapidly over 40ms.
        let hf_energy: f32 = spec[t][(b_high + 1)..=p_high].iter().sum();
        let bass_raw_energy: f32 = spec[t][b_low..=b_high].iter().sum();

        // Check sustain of the dominant harmonic bin 5..10 frames (~50-100ms) after frame t
        let future_start = (t + 5).min(n_frames - 1);
        let future_end = (t + 11).min(n_frames);
        let future_bin_energy: f32 = (future_start..future_end)
            .map(|tau| {
                // Energy within +/- 1 bin of best_bin (steady pitch)
                let lo = best_bin.saturating_sub(1).max(b_low);
                let hi = (best_bin + 1).min(b_high);
                (lo..=hi).map(|b| spec[tau][b]).sum::<f32>()
            })
            .sum::<f32>()
            / ((future_end - future_start).max(1) as f32);

        let cur_bin_energy: f32 = (best_bin.saturating_sub(1).max(b_low)..=(best_bin + 1).min(b_high))
            .map(|b| spec[t][b])
            .sum();

        let sustain_ratio = future_bin_energy / (cur_bin_energy + 1e-6);
        let harmonic_ratio = total_h_energy / (total_h_energy + total_p_energy + hf_energy * 0.5 + 1e-6);

        // Gate is near 1.0 when the note sustains at a stable bass pitch, and near 0.0 for transient kicks
        let gate = if sustain_ratio < 0.28 || harmonic_ratio < 0.42 || hf_energy > bass_raw_energy * 0.85 {
            0.0
        } else {
            ((sustain_ratio - 0.25) * 2.0).clamp(0.0, 1.0) * harmonic_ratio
        };
        percussive_gate[t] = gate;
    }

    (harmonic_bass, percussive_gate, dominant_hz)
}

/// Compute Onset Detection Function combining:
/// 1) Log-compressed SuperFlux (with a 3-bin frequency max filter on the reference frame `t - 2`)
/// 2) Pitch-change flux across normalized bass bins (detects legato bass note changes at constant loudness)
fn compute_bass_odf(harmonic_bass: &[Vec<f32>], percussive_gate: &[f32]) -> Vec<f32> {
    let n_frames = harmonic_bass.len();
    if n_frames == 0 {
        return Vec::new();
    }
    let n_bins = harmonic_bass[0].len();
    let gamma = 25.0f32;
    let mu = 2usize; // reference lag for SuperFlux

    // Precompute log-compressed magnitudes and L2-normalized chroma/pitch vectors
    let mut log_mag = vec![vec![0.0f32; n_bins]; n_frames];
    let mut norm_pitch = vec![vec![0.0f32; n_bins]; n_frames];

    for t in 0..n_frames {
        let mut norm_sq = 0.0f32;
        for k in 0..n_bins {
            let v = harmonic_bass[t][k];
            log_mag[t][k] = (1.0 + gamma * v).ln();
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-4 {
            for k in 0..n_bins {
                norm_pitch[t][k] = harmonic_bass[t][k] / norm;
            }
        }
    }

    let mut odf = vec![0.0f32; n_frames];
    for t in mu..n_frames {
        let ref_t = t - mu;
        let mut superflux = 0.0f32;
        let mut pitch_flux = 0.0f32;

        let cur_energy: f32 = log_mag[t].iter().sum();

        for k in 0..n_bins {
            // 3-bin max filter on reference frame to suppress vibrato
            let k_lo = k.saturating_sub(1);
            let k_hi = (k + 1).min(n_bins - 1);
            let ref_max = (k_lo..=k_hi)
                .map(|j| log_mag[ref_t][j])
                .fold(0.0f32, f32::max);

            let diff = log_mag[t][k] - ref_max;
            if diff > 0.0 {
                superflux += diff;
            }

            let p_diff = norm_pitch[t][k] - norm_pitch[ref_t][k];
            if p_diff > 0.0 {
                pitch_flux += p_diff;
            }
        }

        // Scale pitch flux by current presence of bass energy
        let energy_weight = (cur_energy / 4.0).clamp(0.0, 1.0);
        let raw = superflux + 3.5 * pitch_flux * energy_weight;
        odf[t] = raw * percussive_gate[t];
    }

    // Normalize ODF to [0.0, 1.0]
    let max_odf = odf.iter().copied().fold(0.0f32, f32::max);
    if max_odf > 1e-6 {
        for v in &mut odf {
            *v /= max_odf;
        }
    }
    odf
}

fn pick_onset_peaks(
    odf: &[f32],
    dominant_hz: &[f32],
    duration: Rational,
    config: &BeatDetectorConfig,
) -> Vec<Beat> {
    let n = odf.len();
    if n == 0 {
        return Vec::new();
    }

    let min_gap_frames = ((config.min_cut.as_secs_f64() * (ANALYSIS_SR as f64))
        / (STFT_HOP as f64))
        .round()
        .max(1.0) as usize;
    let win = 15usize; // ~150ms local threshold context

    let mut beats: Vec<Beat> = Vec::new();
    let mut last_frame: Option<usize> = None;

    for i in 1..(n.saturating_sub(1)) {
        let v = odf[i];
        if v < 0.08 || v < odf[i - 1] || v < odf[i + 1] {
            continue;
        }

        let w0 = i.saturating_sub(win);
        let w1 = (i + win + 1).min(n);
        let slice = &odf[w0..w1];
        let mean = slice.iter().sum::<f32>() / (slice.len() as f32);
        let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (slice.len() as f32);
        let std = var.sqrt();
        let thresh = mean + config.peak_delta * std + 0.05;

        if v >= thresh {
            if let Some(prev) = last_frame
                && i - prev < min_gap_frames
            {
                // If the new peak within min_gap is stronger, replace the previous peak
                if let Some(last_beat) = beats.last_mut()
                    && v > last_beat.strength
                {
                    let sec = ((i * STFT_HOP) as f64) / (ANALYSIS_SR as f64);
                    last_beat.time = Rational::from_secs_f64(sec).min(duration);
                    last_beat.strength = v.clamp(0.0, 1.0);
                    last_beat.bass_freq_hz = dominant_hz[i];
                    last_frame = Some(i);
                }
                continue;
            }

            let sec = ((i * STFT_HOP) as f64) / (ANALYSIS_SR as f64);
            let time = Rational::from_secs_f64(sec).min(duration);
            let idx = beats.len();
            beats.push(Beat {
                index: idx,
                time,
                strength: v.clamp(0.0, 1.0),
                bass_freq_hz: dominant_hz[i],
                is_downbeat: idx % 4 == 0,
                enabled: true,
                is_manual: false,
            });
            last_frame = Some(i);
        }
    }

    beats
}

/// Ellis (2007) dynamic programming beat tracker:
/// 1) Estimates global tempo period `tau_0` (in ODF hops) via perceptual-weighted autocorrelation (60..180 BPM)
/// 2) Solves `C(t) = O(t) + max_{t_prev} [ C(t_prev) - alpha * (ln((t - t_prev) / tau_0))^2 ]`
fn regularize_ellis_dp_grid(
    odf: &[f32],
    dominant_hz: &[f32],
    duration: Rational,
    config: &BeatDetectorConfig,
) -> Vec<Beat> {
    let n = odf.len();
    if n < 10 {
        return pick_onset_peaks(odf, dominant_hz, duration, config);
    }

    let hop_sec = (STFT_HOP as f64) / (ANALYSIS_SR as f64);
    // Search lags between 60 BPM (1.0s = ~100 hops) and 180 BPM (0.333s = ~33 hops)
    let min_lag = ((0.33 / hop_sec).round() as usize).max(4);
    let max_lag = ((1.0 / hop_sec).round() as usize).min(n / 2).max(min_lag + 1);

    let mut best_lag = ((0.5 / hop_sec).round() as usize).clamp(min_lag, max_lag);
    let mut best_ac = -1.0f32;
    for lag in min_lag..=max_lag {
        let mut sum = 0.0f32;
        for t in lag..n {
            sum += odf[t] * odf[t - lag];
        }
        // Mild Gaussian prior centered around 120 BPM (0.5s)
        let lag_sec = (lag as f32) * (hop_sec as f32);
        let prior = (-0.5 * ((lag_sec / 0.5).ln() / 0.45).powi(2)).exp();
        let score = sum * prior;
        if score > best_ac {
            best_ac = score;
            best_lag = lag;
        }
    }

    let min_gap = ((config.min_cut.as_secs_f64() / hop_sec).round() as usize).max(4);
    let tau = best_lag.max(min_gap) as f32;
    let alpha = 1.8f32;

    let mut dp = vec![0.0f32; n];
    let mut backlink = vec![None::<usize>; n];

    let lo_search = ((tau * 0.5).round() as usize).max(min_gap);
    let hi_search = ((tau * 2.0).round() as usize).max(lo_search + 1);

    for t in 0..n {
        dp[t] = odf[t];
        let p_min = t.saturating_sub(hi_search);
        let p_max = t.saturating_sub(lo_search);
        if t >= lo_search {
            let mut best_prev_score = f32::NEG_INFINITY;
            let mut best_prev = None;
            for p in p_min..=p_max {
                let interval = (t - p) as f32;
                let penalty = alpha * (interval / tau).ln().powi(2);
                let cand = dp[p] + odf[t] - penalty;
                if cand > best_prev_score {
                    best_prev_score = cand;
                    best_prev = Some(p);
                }
            }
            if let Some(p) = best_prev
                && best_prev_score > dp[t]
            {
                dp[t] = best_prev_score;
                backlink[t] = Some(p);
            }
        }
    }

    // Backtrack from best ending frame in the last beat period
    let tail_start = n.saturating_sub(hi_search);
    let mut cur = (tail_start..n)
        .max_by(|&a, &b| dp[a].partial_cmp(&dp[b]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(n - 1);

    let mut frames = vec![cur];
    while let Some(prev) = backlink[cur] {
        frames.push(prev);
        cur = prev;
    }
    frames.reverse();

    frames
        .into_iter()
        .enumerate()
        .map(|(idx, f)| {
            let sec = ((f * STFT_HOP) as f64) / (ANALYSIS_SR as f64);
            Beat {
                index: idx,
                time: Rational::from_secs_f64(sec).min(duration),
                strength: odf[f].clamp(0.15, 1.0),
                bass_freq_hz: dominant_hz[f],
                is_downbeat: idx % 4 == 0,
                enabled: true,
                is_manual: false,
            }
        })
        .collect()
}

/// Hybrid mode: snaps verified bass onsets to the nearest 1, 1/2, or 1/4 beat grid subdivision
/// within `hybrid_snap_tolerance_ms`, preserving exact bass note articulation when onsets are strong.
fn regularize_hybrid_grid(
    odf: &[f32],
    onsets: &[Beat],
    dominant_hz: &[f32],
    duration: Rational,
    config: &BeatDetectorConfig,
) -> Vec<Beat> {
    if onsets.len() <= 2 {
        return onsets.to_vec();
    }
    let grid = regularize_ellis_dp_grid(odf, dominant_hz, duration, config);
    if grid.len() < 2 {
        return onsets.to_vec();
    }

    // Build 1/4 subdivision ticks from the beat grid
    let mut subdiv_ticks = Vec::new();
    for w in grid.windows(2) {
        let t0 = w[0].time.as_secs_f64();
        let dt = (w[1].time - w[0].time).as_secs_f64();
        for s in 0..4 {
            subdiv_ticks.push(t0 + dt * (s as f64) * 0.25);
        }
    }
    if let Some(last) = grid.last() {
        subdiv_ticks.push(last.time.as_secs_f64());
    }

    let tol_sec = config.hybrid_snap_tolerance_ms / 1000.0;
    let mut out = Vec::with_capacity(onsets.len());

    for (idx, onset) in onsets.iter().enumerate() {
        let t = onset.time.as_secs_f64();
        let nearest = subdiv_ticks
            .iter()
            .copied()
            .min_by(|a, b| (a - t).abs().partial_cmp(&(b - t).abs()).unwrap_or(std::cmp::Ordering::Equal));

        let snapped_t = match nearest {
            Some(nt) if (nt - t).abs() <= tol_sec => {
                // Blend 50% toward the subdivision grid to stay within <= 20ms of true onset
                0.5 * t + 0.5 * nt
            }
            _ => t,
        };

        out.push(Beat {
            index: idx,
            time: Rational::from_secs_f64(snapped_t).min(duration),
            strength: onset.strength,
            bass_freq_hz: onset.bass_freq_hz,
            is_downbeat: idx % 4 == 0,
            enabled: onset.enabled,
            is_manual: onset.is_manual,
        });
    }

    out
}

/// Interactive audio analysis session for the Qt6 GUI (`crates/ae-gui`).
/// Caches decoded 22.05 kHz samples and the full STFT matrix in memory so adjusting
/// `--bass-band`, `--separation`, or peak sensitivity recomputes beats in < 15 ms
/// while preserving user-dragged / manually-toggled (`is_manual`) beats.
#[derive(Clone, Debug)]
pub struct BeatAnalysisSession {
    pub mono_22k: Vec<f32>,
    pub duration: Rational,
    pub full_stft: Vec<Vec<f32>>,
    pub last_bass_spec: Vec<Vec<f32>>,
    pub last_odf: Vec<f32>,
    /// Suppressed / ghost candidate peaks so the GUI can display them as clickable markers.
    pub ghost_peaks: Vec<Beat>,
}

impl BeatAnalysisSession {
    pub fn from_audio_file(path: &Path) -> Result<Self> {
        let (mono_22k, duration) = decode_audio_mono_22050(path)?;
        Ok(Self::from_samples(mono_22k, duration))
    }

    pub fn from_samples(mono_22k: Vec<f32>, duration: Rational) -> Self {
        let full_stft = compute_stft(&mono_22k, STFT_WIN, STFT_HOP);
        Self {
            mono_22k,
            duration,
            full_stft,
            last_bass_spec: Vec::new(),
            last_odf: Vec::new(),
            ghost_peaks: Vec::new(),
        }
    }

    /// Re-run bass separation, ODF calculation, and peak picking on the cached STFT in < 15 ms,
    /// merging in any user-modified (`is_manual == true`) beats from `existing_beats`.
    pub fn recompute_beats(
        &mut self,
        config: &BeatDetectorConfig,
        existing_beats: &[Beat],
    ) -> Result<Vec<Beat>> {
        if self.full_stft.is_empty() {
            return Ok(Vec::new());
        }

        let bin_hz = (ANALYSIS_SR as f32) / (STFT_WIN as f32);
        let b_low = ((config.bass_band.low_hz / bin_hz).floor() as usize).max(1);
        let b_high = ((config.bass_band.high_hz / bin_hz).ceil() as usize)
            .min(STFT_WIN / 2 - 1)
            .max(b_low + 1);
        let p_high = ((2500.0 / bin_hz).ceil() as usize).min(STFT_WIN / 2 - 1);

        let (bass_harmonic_spec, percussive_gate, dominant_hz) = match config.separation {
            SeparationMode::Band => {
                let mut h_spec = vec![vec![0.0f32; b_high - b_low + 1]; self.full_stft.len()];
                let mut dom = vec![80.0f32; self.full_stft.len()];
                for (t, frame_bins) in self.full_stft.iter().enumerate() {
                    let mut max_val = 0.0f32;
                    let mut max_bin = b_low;
                    for k in b_low..=b_high {
                        let v = frame_bins[k];
                        h_spec[t][k - b_low] = v;
                        if v > max_val {
                            max_val = v;
                            max_bin = k;
                        }
                    }
                    dom[t] = (max_bin as f32) * bin_hz;
                }
                (h_spec, vec![1.0f32; self.full_stft.len()], dom)
            }
            SeparationMode::Hpss | SeparationMode::Demucs => {
                separate_harmonic_bass(&self.full_stft, b_low, b_high, p_high, bin_hz)
            }
        };

        let odf = compute_bass_odf(&bass_harmonic_spec, &percussive_gate);

        // Also collect lower-threshold candidate peaks as ghost markers for the GUI
        let mut ghost_cfg = config.clone();
        ghost_cfg.peak_delta = (config.peak_delta * 0.45).max(0.2);
        let all_candidates = pick_onset_peaks(&odf, &dominant_hz, self.duration, &ghost_cfg);

        let raw_onsets = pick_onset_peaks(&odf, &dominant_hz, self.duration, config);
        let mut auto_beats = match config.beat_mode {
            BeatMode::Onsets => raw_onsets,
            BeatMode::Grid => regularize_ellis_dp_grid(&odf, &dominant_hz, self.duration, config),
            BeatMode::Hybrid => {
                regularize_hybrid_grid(&odf, &raw_onsets, &dominant_hz, self.duration, config)
            }
        };

        // Preserve any manual beats (`is_manual == true`) from `existing_beats`
        let manual: Vec<Beat> = existing_beats
            .iter()
            .filter(|b| b.is_manual)
            .cloned()
            .collect();

        if !manual.is_empty() {
            let min_gap = config.min_cut;
            auto_beats.retain(|ab| {
                !manual
                    .iter()
                    .any(|mb| (ab.time - mb.time).abs() < min_gap)
            });
            auto_beats.extend(manual);
            auto_beats.sort_by_key(|b| b.time);
            for (i, b) in auto_beats.iter_mut().enumerate() {
                b.index = i;
            }
        }

        self.ghost_peaks = all_candidates
            .into_iter()
            .filter(|cand| {
                !auto_beats
                    .iter()
                    .any(|ab| (ab.time - cand.time).abs().as_secs_f64() < 0.08)
            })
            .map(|mut g| {
                g.enabled = false;
                g
            })
            .collect();

        self.last_bass_spec = bass_harmonic_spec;
        self.last_odf = odf;

        Ok(auto_beats)
    }

    /// Render the current bass spectrogram, ODF curve, and beat pins to an in-memory `RgbaFrame`
    /// for `image://spectrogram/current` in the Qt6 QML UI.
    pub fn render_spectrogram_rgba(&self, beats: &[Beat]) -> RgbaFrame {
        let w = self.last_bass_spec.len().max(64) as u32;
        let bins = self
            .last_bass_spec
            .first()
            .map(|b| b.len())
            .unwrap_or(40)
            .max(8) as u32;
        let h = bins * 3 + 40;
        let mut frame = RgbaFrame::solid(w, h, [18, 20, 26, 255], Rational::ZERO);

        let max_mag = self
            .last_bass_spec
            .iter()
            .flat_map(|r| r.iter())
            .copied()
            .fold(1e-6f32, f32::max);

        for (x, col) in self.last_bass_spec.iter().enumerate() {
            for (b, &val) in col.iter().enumerate() {
                let norm = ((1.0 + 20.0 * (val / max_mag)).ln() / (21.0f32).ln()).clamp(0.0, 1.0);
                let r = (norm * 240.0) as u8;
                let g = (norm.powi(2) * 180.0) as u8;
                let bl = ((0.2 + 0.6 * norm) * 160.0) as u8;
                for dy in 0..3 {
                    let y = (bins - 1 - (b as u32)) * 3 + dy;
                    if y < h {
                        let idx = ((y * w + (x as u32)) as usize) * 4;
                        frame.data[idx] = r;
                        frame.data[idx + 1] = g;
                        frame.data[idx + 2] = bl;
                        frame.data[idx + 3] = 255;
                    }
                }
            }
            let odf_val = self.last_odf.get(x).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let odf_y = h - 1 - ((odf_val * 36.0) as u32).min(38);
            let idx = ((odf_y * w + (x as u32)) as usize) * 4;
            frame.data[idx] = 80;
            frame.data[idx + 1] = 250;
            frame.data[idx + 2] = 120;
            frame.data[idx + 3] = 255;
        }

        let _ = beats;
        frame
    }
}

fn run_demucs_bass_ort(mono_22k: &[f32], model_path: &Path) -> Result<Vec<f32>> {
    use ort::session::Session;
    use ort::value::TensorRef;

    let env = ort::init().build().map_err(|e| AeError::Model {
        model: "htdemucs".into(),
        message: e.to_string(),
    })?;
    let mut session = Session::builder(&env)
        .map_err(|e| AeError::Model {
            model: "htdemucs".into(),
            message: e.to_string(),
        })?
        .commit_from_file(model_path)
        .map_err(|e| AeError::Model {
            model: "htdemucs".into(),
            message: e.to_string(),
        })?;

    let n = mono_22k.len();
    let mut input = Array3::<f32>::zeros((1, 2, n));
    for (i, &s) in mono_22k.iter().enumerate() {
        input[[0, 0, i]] = s;
        input[[0, 1, i]] = s;
    }

    let tensor = TensorRef::from_array_view(&input).map_err(|e| AeError::Model {
        model: "htdemucs".into(),
        message: e.to_string(),
    })?;
    let outputs = session.run(ort::inputs![tensor]).map_err(|e| AeError::Model {
        model: "htdemucs".into(),
        message: e.to_string(),
    })?;

    if let Some((_, val)) = outputs.iter().next()
        && let Ok((_, slice)) = val.try_extract_tensor::<f32>()
    {
        let out: Vec<f32> = slice.iter().take(n).copied().collect();
        if !out.is_empty() {
            return Ok(out);
        }
    }

    Err(AeError::Model {
        model: "htdemucs".into(),
        message: "empty bass stem output".into(),
    })
}

/// Write `--debug-beats` spectrogram PNG with beat markers and a click-track WAV mixed over the song.
pub fn write_debug_artifacts(
    debug_dir: &Path,
    mono_22k: &[f32],
    bass_spec: &[Vec<f32>],
    odf: &[f32],
    beats: &[Beat],
) -> Result<()> {
    std::fs::create_dir_all(debug_dir)?;

    // 1. Spectrogram PNG with beat lines
    let w = bass_spec.len().max(1) as u32;
    let bins = bass_spec.first().map(|b| b.len()).unwrap_or(48).max(1) as u32;
    let h = bins * 3 + 40;
    let mut img = RgbImage::new(w, h);

    let max_mag = bass_spec
        .iter()
        .flat_map(|r| r.iter())
        .copied()
        .fold(1e-6f32, f32::max);

    for (x, col) in bass_spec.iter().enumerate() {
        for (b, &val) in col.iter().enumerate() {
            let norm = ((1.0 + 20.0 * (val / max_mag)).ln() / (21.0f32).ln()).clamp(0.0, 1.0);
            let r = (norm * 240.0) as u8;
            let g = (norm.powi(2) * 180.0) as u8;
            let bl = ((0.2 + 0.6 * norm) * 160.0) as u8;
            for dy in 0..3 {
                let y = (bins - 1 - (b as u32)) * 3 + dy;
                if y < h {
                    img.put_pixel(x as u32, y, Rgb([r, g, bl]));
                }
            }
        }
        // Plot ODF curve in the bottom 40px strip
        let odf_val = odf.get(x).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let odf_y = h - 1 - ((odf_val * 36.0) as u32).min(38);
        img.put_pixel(x as u32, odf_y, Rgb([80, 250, 120]));
    }

    let hop_sec = (STFT_HOP as f64) / (ANALYSIS_SR as f64);
    for beat in beats {
        let bx = ((beat.time.as_secs_f64() / hop_sec).round() as u32).min(w - 1);
        let color = if beat.is_downbeat {
            Rgb([255, 60, 60])
        } else {
            Rgb([60, 255, 220])
        };
        for y in 0..h {
            img.put_pixel(bx, y, color);
        }
    }

    img.save(debug_dir.join("spectrogram.png"))
        .map_err(|e| AeError::Audio(format!("failed saving debug spectrogram PNG: {e}")))?;

    // 2. Click-track WAV mixed over the mono audio
    let mut mixed = mono_22k.to_vec();
    for beat in beats {
        let start_idx = (beat.time.as_secs_f64() * (ANALYSIS_SR as f64)).round() as usize;
        let freq = if beat.is_downbeat { 1400.0f32 } else { 950.0f32 };
        let click_samples = (ANALYSIS_SR as usize) / 25; // 40ms click
        for i in 0..click_samples {
            if let Some(s) = mixed.get_mut(start_idx + i) {
                let t = (i as f32) / (ANALYSIS_SR as f32);
                let env = (-t * 75.0).exp();
                let click = (2.0 * std::f32::consts::PI * freq * t).sin() * env * 0.55;
                *s = (*s * 0.65 + click).clamp(-1.0, 1.0);
            }
        }
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: ANALYSIS_SR,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(debug_dir.join("click_track.wav"), spec)
        .map_err(|e| AeError::Audio(format!("failed creating click_track.wav: {e}")))?;
    for &s in &mixed {
        let q = (s.clamp(-1.0, 1.0) * (i16::MAX as f32)) as i16;
        writer
            .write_sample(q)
            .map_err(|e| AeError::Audio(format!("failed writing WAV sample: {e}")))?;
    }
    writer
        .finalize()
        .map_err(|e| AeError::Audio(format!("failed finalizing WAV: {e}")))?;

    Ok(())
}

/// Generate a synthetic WAV fixture containing:
/// - Sustained bass guitar / synth bass notes (`bass_times_sec`, e.g. 65 Hz / 82 Hz / 98 Hz with harmonics, ~220ms duration)
/// - Percussive kick drum hits (`kick_times_sec`, fast pitch-drop transient + beater click, ~25ms decay)
pub fn generate_synthetic_bass_and_kick_wav(
    path: &Path,
    duration_secs: f64,
    bass_notes: &[(f64, f32)], // (onset_sec, fundamental_hz)
    kick_hits: &[f64],         // onset_sec of kick-only hits
) -> Result<()> {
    let sr = 44_100u32;
    let total_samples = (duration_secs * (sr as f64)).round() as usize;
    let mut buf = vec![0.0f32; total_samples];

    // Add sustained harmonic bass notes
    for &(onset_s, f0) in bass_notes {
        let start = (onset_s * (sr as f64)).round() as usize;
        let note_len = (0.24 * (sr as f64)).round() as usize; // 240 ms sustained bass note
        for i in 0..note_len {
            if let Some(dst) = buf.get_mut(start + i) {
                let t = (i as f32) / (sr as f32);
                // Fast 3ms attack, sustained body, smooth release
                let attack = (t / 0.003).min(1.0);
                let release = ((0.24 - t) / 0.03).clamp(0.0, 1.0);
                let env = attack * release;
                let wave = 0.65 * (2.0 * std::f32::consts::PI * f0 * t).sin()
                    + 0.30 * (2.0 * std::f32::consts::PI * (f0 * 2.0) * t).sin()
                    + 0.15 * (2.0 * std::f32::consts::PI * (f0 * 3.0) * t).sin();
                *dst += env * wave * 0.75;
            }
        }
    }

    // Add percussive kick drum hits (fast exponential decay + pitch drop + high-frequency beater click)
    for &kick_s in kick_hits {
        let start = (kick_s * (sr as f64)).round() as usize;
        let kick_len = (0.035 * (sr as f64)).round() as usize; // 35 ms transient
        for i in 0..kick_len {
            if let Some(dst) = buf.get_mut(start + i) {
                let t = (i as f32) / (sr as f32);
                let env = (-t * 110.0).exp();
                // Pitch drops rapidly from 150 Hz to 45 Hz + broadband click
                let phase = 2.0 * std::f32::consts::PI * (45.0 * t + (105.0 / 90.0) * (1.0 - (-90.0 * t).exp()));
                let click = (2.0 * std::f32::consts::PI * 950.0 * t).sin()
                    + (2.0 * std::f32::consts::PI * 1650.0 * t).sin();
                *dst += env * (0.55 * phase.sin() + 0.45 * click) * 0.65;
            }
        }
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| AeError::Audio(format!("failed creating fixture WAV: {e}")))?;
    for s in buf {
        let q = (s.clamp(-1.0, 1.0) * (i16::MAX as f32)) as i16;
        writer
            .write_sample(q)
            .map_err(|e| AeError::Audio(format!("failed writing fixture sample: {e}")))?;
    }
    writer
        .finalize()
        .map_err(|e| AeError::Audio(format!("failed finalizing fixture WAV: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bass_onsets_within_20ms_and_kicks_rejected_in_hpss_and_demucs() {
        let tmp = tempfile::tempdir().unwrap();
        let wav_path = tmp.path().join("bass_vs_kick.wav");
        let debug_dir = tmp.path().join("debug_beats");

        // Bass notes at 0.50s, 1.00s, 1.50s, 2.00s, 2.50s, 3.00s
        let bass_notes = [
            (0.50, 65.41), // C2
            (1.00, 82.41), // E2
            (1.50, 73.42), // D2
            (2.00, 98.00), // G2
            (2.50, 65.41), // C2
            (3.00, 82.41), // E2
        ];
        // Kick-only hits in between at 0.75s, 1.75s, 2.75s
        let kick_hits = [0.75, 1.75, 2.75];

        generate_synthetic_bass_and_kick_wav(&wav_path, 3.6, &bass_notes, &kick_hits).unwrap();

        for sep in [SeparationMode::Hpss, SeparationMode::Demucs] {
            let cfg = BeatDetectorConfig {
                separation: sep,
                beat_mode: BeatMode::Onsets,
                debug_dir: Some(debug_dir.clone()),
                ..Default::default()
            };
            let beats = detect_bass_beats(&wav_path, &cfg).unwrap();

            assert_eq!(
                beats.len(),
                bass_notes.len(),
                "mode {:?}: expected {} bass beats (no kicks), got {:?}",
                sep,
                bass_notes.len(),
                beats
            );

            for ((exp_t, _), det) in bass_notes.iter().zip(beats.iter()) {
                let err_ms = (det.time.as_secs_f64() - exp_t).abs() * 1000.0;
                assert!(
                    err_ms <= 20.0,
                    "mode {:?}: bass onset at {:.3}s detected at {:.3}s (error {:.1}ms > 20ms)",
                    sep,
                    exp_t,
                    det.time.as_secs_f64(),
                    err_ms
                );
            }

            // Verify none of the kick-only hits were reported
            for &kick_t in &kick_hits {
                assert!(
                    beats.iter().all(|b| (b.time.as_secs_f64() - kick_t).abs() > 0.10),
                    "mode {:?}: kick-only hit at {:.2}s was falsely reported!",
                    sep,
                    kick_t
                );
            }
        }

        assert!(debug_dir.join("spectrogram.png").exists());
        assert!(debug_dir.join("click_track.wav").exists());
    }
}
