use super::*;

type Boundary = (usize, usize, f32, ShotBoundaryKind);

/// Incremental classic detector. Set `finalize` only after the last frame has been pushed.
pub struct ClassicShotDetector {
    pub(super) metrics: Vec<FrameMetrics>,
    pub(super) diffs: Vec<f32>,
    config: ShotDetectorConfig,
    next: usize,
    gradual_start: Option<usize>,
    gradual_accum: f32,
    scratch: Vec<f32>,
    boundaries: Vec<Boundary>,
    consumed_boundaries: usize,
    shots: Vec<Shot>,
    raw_start: usize,
    usable_start: usize,
    confidence: f32,
    kind: ShotBoundaryKind,
    tail_start: usize,
    tail_end: usize,
    tail_sum: f32,
    pub(super) last_nonblack: Option<usize>,
    finalized: bool,
}

impl ClassicShotDetector {
    pub fn new(config: ShotDetectorConfig) -> Self {
        Self {
            metrics: Vec::new(),
            diffs: Vec::new(),
            config,
            next: 1,
            gradual_start: None,
            gradual_accum: 0.0,
            scratch: Vec::new(),
            boundaries: Vec::new(),
            consumed_boundaries: 0,
            shots: Vec::new(),
            raw_start: 0,
            usable_start: 0,
            confidence: 1.0,
            kind: ShotBoundaryKind::Initial,
            tail_start: 0,
            tail_end: 0,
            tail_sum: 0.0,
            last_nonblack: None,
            finalized: false,
        }
    }

    pub fn push(&mut self, frame: &RgbaFrame) {
        if self.finalized {
            // A finalized tail used truncated lookahead; recompute it if more frames arrive.
            self.next = 1;
            self.gradual_start = None;
            self.gradual_accum = 0.0;
            self.boundaries.clear();
            self.consumed_boundaries = 0;
            self.shots.clear();
            self.raw_start = 0;
            self.usable_start = 0;
            self.confidence = 1.0;
            self.kind = ShotBoundaryKind::Initial;
            self.tail_start = 0;
            self.tail_end = 0;
            self.tail_sum = 0.0;
            self.finalized = false;
        }
        let metric = FrameMetrics::from_frame(frame);
        self.diffs.push(
            self.metrics
                .last()
                .map_or(0.0, |prev| prev.distance(&metric)),
        );
        if metric.mean_luma >= self.config.black_frame_luma {
            self.last_nonblack = Some(self.metrics.len());
        }
        self.metrics.push(metric);
    }

    pub fn detect(
        &mut self,
        total_duration: Rational,
        fps: Rational,
        finalize: bool,
    ) -> Result<Vec<Shot>> {
        let n = self.metrics.len();
        if n <= 1 {
            return Ok(vec![Shot {
                id: 0,
                start: Rational::ZERO,
                end: total_duration.max(Rational::from_secs(1)),
                usable_start: Rational::ZERO,
                usable_end: total_duration.max(Rational::from_secs(1)),
                cut_confidence: 1.0,
                boundary_kind: ShotBoundaryKind::Initial,
                mean_luma: self.metrics.first().map(|m| m.mean_luma).unwrap_or(0.5),
                enabled: true,
                locked: false,
            }]);
        }
        let lookahead = (self.config.adaptive_window / 2).max(2);
        let end = if finalize {
            n
        } else {
            n.saturating_sub(lookahead)
        };
        while self.next < end {
            self.process_frame(self.next);
            self.next += 1;
        }
        self.finalized |= finalize;
        self.update_shots(total_duration, fps)
    }

    fn process_frame(&mut self, i: usize) {
        let n = self.metrics.len();
        let half_w = self.config.adaptive_window / 2;
        let start = i.saturating_sub(half_w).max(1);
        let end = i.saturating_add(half_w).saturating_add(1).min(n);
        self.scratch.clear();
        self.scratch.extend_from_slice(&self.diffs[start..end]);
        let mid = self.scratch.len() / 2;
        let median = *self
            .scratch
            .select_nth_unstable_by(mid, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .1;
        for value in &mut self.scratch {
            *value = (*value - median).abs();
        }
        let mad = (*self
            .scratch
            .select_nth_unstable_by(mid, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .1)
            .max(0.008);
        let t_high = (median + self.config.mad_k_high * mad).max(self.config.min_hard_cut_score);
        let t_low =
            (median + self.config.mad_k_low * mad).max(self.config.min_hard_cut_score * 0.45);
        let d = self.diffs[i];
        if d >= t_high {
            let is_flash = if i + 1 < n {
                let revert_1 = self.metrics[i - 1].distance(&self.metrics[i + 1]);
                let revert_2 = if i + 2 < n {
                    self.metrics[i - 1].distance(&self.metrics[i + 2])
                } else {
                    1.0
                };
                revert_1 < t_low * 0.85 || revert_2 < t_low * 0.85
            } else {
                false
            };
            if !is_flash {
                self.gradual_start = None;
                self.gradual_accum = 0.0;
                let conf = ((d - t_high) / t_high.max(0.1) + 0.65).clamp(0.5, 1.0);
                self.boundaries
                    .push((i, i, conf, ShotBoundaryKind::HardCut));
                return;
            }
        }
        if d >= t_low {
            let start_idx = *self.gradual_start.get_or_insert(i.saturating_sub(1));
            self.gradual_accum += d - median;
            let anchor_dist = self.metrics[start_idx].distance(&self.metrics[i]);
            if self.gradual_accum >= t_high * 1.4 && anchor_dist >= t_high * 0.9 {
                let kind = if self.metrics[i].mean_luma < 0.05
                    || self.metrics[start_idx].mean_luma < 0.05
                {
                    ShotBoundaryKind::Fade
                } else {
                    ShotBoundaryKind::Dissolve
                };
                let conf = (anchor_dist / t_high.max(0.1)).clamp(0.5, 0.95);
                self.boundaries.push((start_idx, i + 1, conf, kind));
                self.gradual_start = None;
                self.gradual_accum = 0.0;
            }
        } else {
            self.gradual_start = None;
            self.gradual_accum = 0.0;
        }
    }

    fn update_shots(&mut self, duration: Rational, fps: Rational) -> Result<Vec<Shot>> {
        let n = self.metrics.len();
        let frame_dur = if fps.numer() > 0 {
            Rational::new(fps.denom(), fps.numer())
        } else {
            Rational::new(1, 30)
        };
        while self.consumed_boundaries < self.boundaries.len() {
            let (end, next, conf, kind) = self.boundaries[self.consumed_boundaries];
            self.consumed_boundaries += 1;
            let seg_end = end.clamp(self.raw_start + 1, n);
            let seg_dur = self.metrics[seg_end.min(n - 1)].pts - self.metrics[self.raw_start].pts;
            if seg_dur >= self.config.min_shot_duration {
                // The open segment may have been trimmed or extended since the last snapshot.
                self.tail_end = 0;
                self.tail_sum = 0.0;
                let shot = self.make_shot(seg_end, end.min(n), duration, frame_dur);
                self.shots.push(shot);
                self.raw_start = seg_end;
                self.usable_start = next.min(n - 1);
                self.confidence = conf;
                self.kind = kind;
                self.tail_end = 0;
                self.tail_sum = 0.0;
            }
        }
        let mut shots = self.shots.clone();
        if self.raw_start < n {
            shots.push(self.make_shot(n, n, duration, frame_dur));
        }
        Ok(shots)
    }

    fn make_shot(
        &mut self,
        end: usize,
        usable_end: usize,
        duration: Rational,
        frame_dur: Rational,
    ) -> Shot {
        let n = self.metrics.len();
        let mut start = self
            .usable_start
            .clamp(self.raw_start, end.saturating_sub(1));
        if self.tail_end != 0 {
            start = start.max(self.tail_start);
        }
        let mut u_end = usable_end.clamp(start + 1, end);
        while start + 4 < u_end && self.metrics[start].mean_luma < self.config.black_frame_luma {
            start += 1;
        }
        // Completed shots are computed once; the open shot uses the cached last non-black frame.
        if end == n && usable_end == n {
            u_end = self
                .last_nonblack
                .map_or(start + 4, |last| (last + 1).max(start + 4))
                .min(n);
        } else {
            while u_end > start + 4
                && self.metrics[u_end - 1].mean_luma < self.config.black_frame_luma
            {
                u_end -= 1;
            }
        }
        if self.tail_start != start || self.tail_end < start || self.tail_end > u_end {
            self.tail_start = start;
            self.tail_end = start;
            self.tail_sum = 0.0;
        }
        while self.tail_end < u_end {
            self.tail_sum += self.metrics[self.tail_end].mean_luma;
            self.tail_end += 1;
        }
        let end_pts = if end < n {
            self.metrics[end].pts
        } else {
            duration.max(self.metrics[n - 1].pts + frame_dur)
        };
        Shot {
            id: self.shots.len(),
            start: self.metrics[self.raw_start].pts,
            end: end_pts,
            usable_start: self.metrics[start].pts,
            usable_end: if u_end < n {
                self.metrics[u_end].pts
            } else {
                end_pts
            },
            cut_confidence: self.confidence,
            boundary_kind: self.kind,
            mean_luma: self.tail_sum / ((u_end - start).max(1) as f32),
            enabled: true,
            locked: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn original(
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
        diffs[1..]
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, out)| {
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
                    let kind = if metrics[i].mean_luma < 0.05 || metrics[start_idx].mean_luma < 0.05
                    {
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

    #[test]
    fn matches_original_for_streams_and_windows() {
        let fps = Rational::from_secs(30);
        let mut frames = Vec::new();
        for i in 0..160 {
            let color = match i {
                0..=7 | 144..=159 => [0, 0, 0, 255],
                8..=24 => [190, 40, 30, 255],
                25 => [255, 255, 255, 255],
                26..=44 => [190, 40, 30, 255],
                45..=64 => [
                    (190 - (i - 45) * 7) as u8,
                    (40 + (i - 45) * 7) as u8,
                    30,
                    255,
                ],
                65..=99 => [30, 190, 30, 255],
                100..=119 => [30, (190 - (i - 100) * 9) as u8, 30, 255],
                _ => [35, 30, 180, 255],
            };
            frames.push(RgbaFrame::solid(
                8,
                8,
                color,
                Rational::from_frame_index(i, fps),
            ));
        }
        for &len in &[0, 1, 2, 3, 4, 9, 35, 70, 120, 160] {
            for &window in &[0, 1, 2, 5, 30, 31, 180] {
                let config = ShotDetectorConfig {
                    adaptive_window: window,
                    min_shot_duration: Rational::ZERO,
                    ..Default::default()
                };
                let duration = Rational::from_frame_index(len as i64, fps);
                let expected = original(&frames[..len], duration, fps, &config).unwrap();
                assert_eq!(
                    detect_shots_classic(&frames[..len], duration, fps, &config).unwrap(),
                    expected,
                    "batch len={len} window={window}"
                );
                for &chunk in &[1, 3, 17, 200] {
                    let mut detector = ClassicShotDetector::new(config.clone());
                    for part in frames[..len].chunks(chunk) {
                        for frame in part {
                            detector.push(frame);
                        }
                        let before = detector.next;
                        detector.detect(duration, fps, false).unwrap();
                        let expected_next = detector
                            .metrics
                            .len()
                            .saturating_sub((window / 2).max(2))
                            .max(1);
                        assert_eq!(detector.next, expected_next);
                        detector.detect(duration, fps, false).unwrap();
                        assert_eq!(detector.next, expected_next.max(before));
                    }
                    assert_eq!(
                        detector.detect(duration, fps, true).unwrap(),
                        expected,
                        "len={len} window={window} chunk={chunk}"
                    );
                    assert_eq!(detector.detect(duration, fps, true).unwrap(), expected);
                }
            }
        }
    }
    #[test]
    fn fade_and_flash_match_original() {
        let fps = Rational::from_secs(30);
        let mut frames = Vec::new();
        for i in 0..70 {
            let white = match i {
                0..=9 => 0,
                10..=19 => (i - 9) * 10,
                40 => 0,
                60..=69 => 0,
                _ => 100,
            };
            let mut frame =
                RgbaFrame::solid(100, 1, [0, 0, 0, 255], Rational::from_frame_index(i, fps));
            for pixel in frame.data.chunks_exact_mut(4).take(white as usize) {
                pixel.copy_from_slice(&[255, 255, 255, 255]);
            }
            frames.push(frame);
        }
        let config = ShotDetectorConfig {
            min_shot_duration: Rational::ZERO,
            min_hard_cut_score: 0.08,
            ..Default::default()
        };
        let duration = Rational::from_frame_index(70, fps);
        let expected = original(&frames, duration, fps, &config).unwrap();
        assert!(
            expected
                .iter()
                .any(|s| s.boundary_kind == ShotBoundaryKind::Fade),
            "{expected:?}"
        );
        assert!(!expected.iter().any(|s| s.start == frames[40].pts));
        let mut detector = ClassicShotDetector::new(config);
        for frame in &frames {
            detector.push(frame);
            detector.detect(duration, fps, false).unwrap();
        }
        assert_eq!(detector.detect(duration, fps, true).unwrap(), expected);
    }
    #[test]
    fn nonblack_open_segment_closes_without_double_counting() {
        let fps = Rational::from_secs(30);
        let frames: Vec<_> = (0..80)
            .map(|i| {
                let color = if i < 35 {
                    [200, 20, 20, 255]
                } else {
                    [20, 200, 20, 255]
                };
                RgbaFrame::solid(8, 8, color, Rational::from_frame_index(i, fps))
            })
            .collect();
        let config = ShotDetectorConfig::default();
        let duration = Rational::from_frame_index(80, fps);
        let expected = original(&frames, duration, fps, &config).unwrap();
        assert!(expected.len() > 1);
        let mut detector = ClassicShotDetector::new(config);
        for frame in &frames {
            detector.push(frame);
            detector.detect(duration, fps, false).unwrap();
        }
        assert_eq!(detector.detect(duration, fps, true).unwrap(), expected);
    }
    #[test]
    fn dissolve_matches_original() {
        let fps = Rational::from_secs(30);
        let frames: Vec<_> = (0..60)
            .map(|i| {
                let green = if i < 10 {
                    0
                } else if i < 20 {
                    (i - 9) * 10
                } else {
                    100
                };
                let mut frame = RgbaFrame::solid(
                    100,
                    1,
                    [220, 20, 20, 255],
                    Rational::from_frame_index(i, fps),
                );
                for pixel in frame.data.chunks_exact_mut(4).take(green as usize) {
                    pixel.copy_from_slice(&[20, 220, 20, 255]);
                }
                frame
            })
            .collect();
        let config = ShotDetectorConfig {
            min_shot_duration: Rational::ZERO,
            min_hard_cut_score: 0.06,
            ..Default::default()
        };
        let duration = Rational::from_frame_index(60, fps);
        let expected = original(&frames, duration, fps, &config).unwrap();
        assert!(
            expected
                .iter()
                .any(|s| s.boundary_kind == ShotBoundaryKind::Dissolve),
            "{expected:?}"
        );
        let mut detector = ClassicShotDetector::new(config);
        for frame in &frames {
            detector.push(frame);
            detector.detect(duration, fps, false).unwrap();
        }
        assert_eq!(detector.detect(duration, fps, true).unwrap(), expected);
    }
}
