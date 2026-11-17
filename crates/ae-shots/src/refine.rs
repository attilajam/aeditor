use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    thread,
};

use ae_core::{AeError, Rational, Resolution, Result, RgbaFrame, Shot, ShotBoundaryKind};
use ae_media::MediaBackend;

use super::FrameMetrics;

fn window(previous: &Shot, next: &Shot, sample: Rational, fps: Rational) -> (i64, i64) {
    let start = (next.start - sample)
        .max(previous.start)
        .to_floor_frame(fps);
    let end = (next.start + sample).min(next.end).to_floor_frame(fps) + 1;
    (start, end)
}

fn decode(
    backend: &dyn MediaBackend,
    path: &Path,
    fps: Rational,
    (start, end): (i64, i64),
) -> Result<Vec<FrameMetrics>> {
    let frames = backend.decode_segment(
        path,
        Rational::from_frame_index(start, fps),
        Rational::from_frame_index(end - start, fps),
        Resolution::new(64, 36),
        fps,
        1.0,
        None,
    )?;
    Ok(frames.iter().map(FrameMetrics::from_frame).collect())
}

fn apply_boundary(previous: &mut Shot, next: &mut Shot, metrics: &[FrameMetrics]) {
    let best = metrics
        .windows(2)
        .filter(|pair| pair[1].pts > previous.usable_start && pair[1].pts < next.usable_end)
        .max_by(|a, b| a[0].distance(&a[1]).total_cmp(&b[0].distance(&b[1])));
    if let Some(pair) = best {
        let boundary = pair[1].pts;
        previous.usable_end = if previous.usable_end == previous.end {
            boundary
        } else {
            previous.usable_end.min(boundary)
        };
        next.usable_start = if next.usable_start == next.start {
            boundary
        } else {
            next.usable_start.max(boundary)
        };
        previous.end = boundary;
        next.start = boundary;
    }
}

/// Refine cuts using native-rate 64x36 RGBA frames captured during ingest.
pub fn refine_shot_boundaries_from_frame_file(
    video_path: &Path,
    frame_file: &Path,
    shots: &mut [Shot],
    analysis_fps: Rational,
    source_fps: Rational,
    progress: &mut dyn FnMut(f32),
) -> Result<()> {
    const FRAME_BYTES: u64 = 64 * 36 * 4;
    if analysis_fps.numer() <= 0
        || source_fps.numer() <= 0
        || analysis_fps.denom() <= 0
        || source_fps.denom() <= 0
    {
        return Err(AeError::Config(
            "shot refinement requires positive frame rates".into(),
        ));
    }
    let error = |message: String| AeError::Decode {
        path: video_path.to_path_buf(),
        message: format!("frame file '{}': {message}", frame_file.display()),
    };
    let mut file = File::open(frame_file).map_err(|e| error(e.to_string()))?;
    let size = file.metadata().map_err(|e| error(e.to_string()))?.len();
    if size == 0 || size % FRAME_BYTES != 0 {
        return Err(error("empty or incomplete RGBA frames".into()));
    }
    let count = i64::try_from(size / FRAME_BYTES).map_err(|_| error("too many frames".into()))?;
    // The final proxy sample can extend past the last native frame by one analysis interval.
    let tail_frames = (source_fps.as_secs_f64() / analysis_fps.as_secs_f64()).ceil() as i64;
    let limit = count.saturating_add(tail_frames);
    for shot in shots.iter() {
        if shot.start < Rational::ZERO
            || shot.end < shot.start
            || shot.end.to_floor_frame(source_fps) > limit
            || shot.usable_start < shot.start
            || shot.usable_end > shot.end
            || shot.usable_end < shot.usable_start
        {
            return Err(error("shot range is outside the frame file".into()));
        }
    }
    if analysis_fps >= source_fps {
        progress(1.0);
        return Ok(());
    }
    let sample = Rational::new(analysis_fps.denom(), analysis_fps.numer());
    let total = shots
        .iter()
        .skip(1)
        .filter(|s| s.boundary_kind == ShotBoundaryKind::HardCut)
        .count();
    let mut completed = 0;
    progress(0.0);
    let mut frame = RgbaFrame::new(64, 36, Rational::ZERO);
    for i in 1..shots.len() {
        let (before, after) = shots.split_at_mut(i);
        let previous = &mut before[i - 1];
        let next = &mut after[0];
        if next.boundary_kind != ShotBoundaryKind::HardCut {
            continue;
        }
        let (start, end) = window(previous, next, sample, source_fps);
        if start < 0 || end < start || end > limit.saturating_add(1) {
            return Err(error("boundary window is outside the frame file".into()));
        }
        let first = start.min(count - 1);
        file.seek(SeekFrom::Start(first as u64 * FRAME_BYTES))
            .map_err(|e| error(e.to_string()))?;
        let mut metrics = Vec::new();
        for index in start..end {
            if index < count || index == start {
                file.read_exact(&mut frame.data)
                    .map_err(|e| error(e.to_string()))?;
            }
            frame.pts = Rational::from_frame_index(index, source_fps);
            metrics.push(FrameMetrics::from_frame(&frame));
        }
        apply_boundary(previous, next, &metrics);
        completed += 1;
        progress(completed as f32 / total as f32);
    }
    progress(1.0);
    Ok(())
}

/// Refine sampled hard cuts to source-frame boundaries before their ranges are used for editing.
pub fn refine_shot_boundaries(
    backend: &dyn MediaBackend,
    video_path: &Path,
    shots: &mut [Shot],
    analysis_fps: Rational,
    source_fps: Rational,
) -> Result<()> {
    refine_shot_boundaries_with_progress(
        backend,
        video_path,
        shots,
        analysis_fps,
        source_fps,
        &mut |_| {},
    )
}

/// Refine hard cuts while reporting completed work as a fraction from zero to one.
pub fn refine_shot_boundaries_with_progress(
    backend: &dyn MediaBackend,
    video_path: &Path,
    shots: &mut [Shot],
    analysis_fps: Rational,
    source_fps: Rational,
    progress: &mut dyn FnMut(f32),
) -> Result<()> {
    if analysis_fps.numer() <= 0 || source_fps.numer() <= 0 {
        return Err(AeError::Config(
            "shot refinement requires positive frame rates".into(),
        ));
    }
    if analysis_fps >= source_fps {
        progress(1.0);
        return Ok(());
    }

    let sample = Rational::new(analysis_fps.denom(), analysis_fps.numer());
    let indices: Vec<_> = (1..shots.len())
        .filter(|&i| shots[i].boundary_kind == ShotBoundaryKind::HardCut)
        .collect();
    let duration = shots.last().map_or(Rational::ZERO, |shot| shot.end);
    // shortcut: this density threshold can over-decode unusual footage; tune with real workloads.
    let dense = indices.len() >= 32
        && duration > Rational::ZERO
        && duration.as_secs_f64() / (indices.len() as f64) <= 10.0;
    let streamed = if dense {
        let mut ranges: Vec<_> = indices
            .iter()
            .map(|&i| window(&shots[i - 1], &shots[i], sample, source_fps))
            .collect();
        ranges.sort_unstable();
        let mut merged: Vec<(i64, i64)> = Vec::new();
        for (start, end) in ranges {
            if let Some(last) = merged.last_mut().filter(|last| start <= last.1) {
                last.1 = last.1.max(end);
            } else {
                merged.push((start, end));
            }
        }
        let mut metrics = Vec::new();
        let mut range_index = 0;
        let mut completed = 0i64;
        let total = duration.to_nearest_frame(source_fps).max(1);
        progress(0.0);
        backend.decode_segment_stream(
            video_path,
            Rational::ZERO,
            duration,
            Resolution::new(64, 36),
            source_fps,
            1.0,
            None,
            &mut |frame| {
                while range_index < merged.len() && completed >= merged[range_index].1 {
                    range_index += 1;
                }
                if range_index < merged.len() && completed >= merged[range_index].0 {
                    metrics.push((completed, FrameMetrics::from_frame(&frame)));
                }
                completed += 1;
                if completed % 128 == 0 {
                    progress((completed as f64 / total as f64).min(1.0) as f32);
                }
                Ok(())
            },
        )?;
        Some(metrics)
    } else {
        progress(0.0);
        None
    };
    let mut completed_cuts = 0;
    for batch in indices.chunks(4) {
        let jobs: Vec<_> = batch
            .iter()
            .map(|&i| (i, window(&shots[i - 1], &shots[i], sample, source_fps)))
            .collect();
        let results = if let Some(metrics) = &streamed {
            jobs.iter()
                .map(|&(_, (start, end))| {
                    let first = metrics.partition_point(|(frame, _)| *frame < start);
                    let last = metrics.partition_point(|(frame, _)| *frame < end);
                    Ok(metrics[first..last]
                        .iter()
                        .map(|(_, metric)| *metric)
                        .collect())
                })
                .collect()
        } else {
            thread::scope(|scope| {
                let handles: Vec<_> = jobs
                    .iter()
                    .map(|&(_, range)| {
                        scope.spawn(move || decode(backend, video_path, source_fps, range))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| {
                        handle.join().unwrap_or_else(|_| {
                            Err(AeError::Decode {
                                path: video_path.to_path_buf(),
                                message: "shot refinement thread panicked".into(),
                            })
                        })
                    })
                    .collect::<Vec<_>>()
            })
        };
        for ((i, original_range), result) in jobs.into_iter().zip(results) {
            let (before, after) = shots.split_at_mut(i);
            let previous = &mut before[i - 1];
            let next = &mut after[0];
            let current_range = window(previous, next, sample, source_fps);
            // An earlier refined cut can change this window; its speculative result is then stale.
            let metrics = if current_range == original_range {
                result?
            } else {
                decode(backend, video_path, source_fps, current_range)?
            };
            apply_boundary(previous, next, &metrics);
            completed_cuts += 1;
            if !dense {
                progress(completed_cuts as f32 / indices.len() as f32);
            }
        }
    }
    progress(1.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ae_core::{RgbaFrame, VideoCodec};
    use ae_media::{AudioProbeInfo, VideoProbeInfo, VideoWriter, VideoWriterConfig};
    use crossbeam_channel::Sender;
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    #[derive(Default)]
    struct Fake {
        active: AtomicUsize,
        peak: AtomicUsize,
        calls: AtomicUsize,
        stream_calls: AtomicUsize,
        fail: bool,
    }

    impl MediaBackend for Fake {
        fn probe_video(&self, _: &Path) -> Result<VideoProbeInfo> {
            unreachable!()
        }
        fn probe_audio(&self, _: &Path) -> Result<AudioProbeInfo> {
            unreachable!()
        }
        fn decode_scaled_stream(
            &self,
            _: &Path,
            _: u32,
            _: u32,
            _: Option<Rational>,
            _: Sender<RgbaFrame>,
        ) -> Result<()> {
            unreachable!()
        }
        fn decode_segment(
            &self,
            _: &Path,
            start: Rational,
            duration: Rational,
            _: Resolution,
            fps: Rational,
            _: f64,
            _: Option<Rational>,
        ) -> Result<Vec<RgbaFrame>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(active, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(10));
            self.active.fetch_sub(1, Ordering::SeqCst);
            if self.fail {
                return Err(AeError::Config("decode failed".into()));
            }
            let first = start.to_nearest_frame(fps);
            let count = duration.to_nearest_frame(fps);
            Ok((first..first + count)
                .map(|i| {
                    let pts = Rational::from_frame_index(i, fps);
                    let color = if (i / 3) % 2 == 0 { 0 } else { 255 };
                    RgbaFrame::solid(4, 4, [color, color, color, 255], pts)
                })
                .collect())
        }
        fn decode_segment_stream(
            &self,
            _: &Path,
            start: Rational,
            duration: Rational,
            _: Resolution,
            fps: Rational,
            _: f64,
            _: Option<Rational>,
            emit: &mut dyn FnMut(RgbaFrame) -> Result<()>,
        ) -> Result<()> {
            self.stream_calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return Err(AeError::Config("decode failed".into()));
            }
            for i in start.to_nearest_frame(fps)..(start + duration).to_nearest_frame(fps) {
                let color = if (i / 3) % 2 == 0 { 0 } else { 255 };
                emit(RgbaFrame::solid(
                    4,
                    4,
                    [color, color, color, 255],
                    Rational::from_frame_index(i, fps),
                ))?;
            }
            Ok(())
        }
        fn run_scdet_baseline(&self, _: &Path, _: f32) -> Result<Vec<Rational>> {
            unreachable!()
        }
        fn create_writer(&self, _: VideoWriterConfig) -> Result<Box<dyn VideoWriter>> {
            unreachable!()
        }
        fn passthrough_remux(
            &self,
            _: &Path,
            _: &Path,
            _: &Path,
            _: Option<Resolution>,
            _: VideoCodec,
        ) -> Result<()> {
            unreachable!()
        }
    }

    fn shots(times: &[i64]) -> Vec<Shot> {
        times
            .windows(2)
            .enumerate()
            .map(|(id, t)| Shot {
                id,
                start: Rational::from_millis(t[0]),
                end: Rational::from_millis(t[1]),
                usable_start: Rational::from_millis(t[0]),
                usable_end: Rational::from_millis(t[1]),
                cut_confidence: 1.0,
                boundary_kind: if id == 0 {
                    ShotBoundaryKind::Initial
                } else {
                    ShotBoundaryKind::HardCut
                },
                mean_luma: 0.5,
                enabled: true,
                locked: false,
            })
            .collect()
    }

    fn sequential(
        backend: &dyn MediaBackend,
        video_path: &Path,
        shots: &mut [Shot],
        analysis_fps: Rational,
        source_fps: Rational,
    ) -> Result<()> {
        if analysis_fps.numer() <= 0 || source_fps.numer() <= 0 {
            return Err(AeError::Config(
                "shot refinement requires positive frame rates".into(),
            ));
        }
        if analysis_fps >= source_fps {
            return Ok(());
        }

        let sample_duration = Rational::new(analysis_fps.denom(), analysis_fps.numer());
        for i in 1..shots.len() {
            let (before, after) = shots.split_at_mut(i);
            let previous = &mut before[i - 1];
            let next = &mut after[0];
            if next.boundary_kind != ShotBoundaryKind::HardCut {
                continue;
            }

            let start_frame = (next.start - sample_duration)
                .max(previous.start)
                .to_floor_frame(source_fps);
            let end_frame = (next.start + sample_duration)
                .min(next.end)
                .to_floor_frame(source_fps)
                + 1;
            let frames = backend.decode_segment(
                video_path,
                Rational::from_frame_index(start_frame, source_fps),
                Rational::from_frame_index(end_frame - start_frame, source_fps),
                Resolution::new(64, 36),
                source_fps,
                1.0,
                None,
            )?;
            let metrics: Vec<_> = frames.iter().map(FrameMetrics::from_frame).collect();
            let best = metrics
                .windows(2)
                .filter(|pair| pair[1].pts > previous.usable_start && pair[1].pts < next.usable_end)
                .max_by(|a, b| a[0].distance(&a[1]).total_cmp(&b[0].distance(&b[1])));
            if let Some(pair) = best {
                let boundary = pair[1].pts;
                previous.usable_end = if previous.usable_end == previous.end {
                    boundary
                } else {
                    previous.usable_end.min(boundary)
                };
                next.usable_start = if next.usable_start == next.start {
                    boundary
                } else {
                    next.usable_start.max(boundary)
                };
                previous.end = boundary;
                next.start = boundary;
            }
        }
        Ok(())
    }

    fn raw_frames(count: usize) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        for i in 0..count {
            let color = if (i / 3) % 2 == 0 { 0 } else { 255 };
            let frame = RgbaFrame::solid(64, 36, [color, color, color, 255], Rational::ZERO);
            file.write_all(&frame.data).unwrap();
        }
        file
    }

    #[test]
    fn frame_file_matches_sequential_and_dense() {
        for fps in [Rational::from_secs(30), Rational::new(30000, 1001)] {
            for interval in [100, 1000] {
                let times: Vec<_> = (0..=40).map(|i| i * interval).collect();
                let mut actual = shots(&times);
                actual[0].usable_start = Rational::from_millis(20);
                let last = actual.len() - 1;
                actual[last].usable_end = actual[last].end - Rational::from_millis(20);
                let mut expected = actual.clone();
                let mut dense = actual.clone();
                let analysis = Rational::from_secs(5);
                let count = actual[last].end.to_nearest_frame(fps) as usize + 2;
                let file = raw_frames(count);
                sequential(
                    &Fake::default(),
                    Path::new("test"),
                    &mut expected,
                    analysis,
                    fps,
                )
                .unwrap();
                refine_shot_boundaries_with_progress(
                    &Fake::default(),
                    Path::new("test"),
                    &mut dense,
                    analysis,
                    fps,
                    &mut |_| {},
                )
                .unwrap();
                let mut updates = Vec::new();
                refine_shot_boundaries_from_frame_file(
                    Path::new("test"),
                    file.path(),
                    &mut actual,
                    analysis,
                    fps,
                    &mut |p| updates.push(p),
                )
                .unwrap();
                assert_eq!(actual, expected);
                assert_eq!(actual, dense);
                assert_eq!(updates.first(), Some(&0.0));
                assert_eq!(updates.last(), Some(&1.0));
                assert!(updates.windows(2).all(|p| p[0] <= p[1] && p[1] <= 1.0));
            }
        }
    }

    #[test]
    fn frame_file_rejects_bad_inputs_and_pads_final_frame() {
        use std::io::Write;
        let fps = Rational::from_secs(30);
        let analysis = Rational::from_secs(5);
        let path = Path::new("test");
        let mut empty = tempfile::NamedTempFile::new().unwrap();
        let mut items = shots(&[0, 500, 1000]);
        let original = items.clone();
        assert!(
            refine_shot_boundaries_from_frame_file(
                path,
                empty.path(),
                &mut items,
                analysis,
                fps,
                &mut |_| {}
            )
            .is_err()
        );
        empty.write_all(&[1, 2, 3]).unwrap();
        assert!(
            refine_shot_boundaries_from_frame_file(
                path,
                empty.path(),
                &mut items,
                analysis,
                fps,
                &mut |_| {}
            )
            .is_err()
        );
        let short = raw_frames(10);
        assert!(
            refine_shot_boundaries_from_frame_file(
                path,
                short.path(),
                &mut items,
                analysis,
                fps,
                &mut |_| {}
            )
            .is_err()
        );
        assert!(
            refine_shot_boundaries_from_frame_file(
                path,
                short.path(),
                &mut items,
                Rational::ZERO,
                fps,
                &mut |_| {}
            )
            .is_err()
        );
        assert_eq!(items, original);
        let file = raw_frames(30);
        let mut edge = shots(&[0, 900, 1000]);
        refine_shot_boundaries_from_frame_file(
            path,
            file.path(),
            &mut edge,
            analysis,
            fps,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(edge[0].end, edge[1].start);
        let mut padded = shots(&[0, 900, 1200]);
        refine_shot_boundaries_from_frame_file(
            path,
            file.path(),
            &mut padded,
            analysis,
            fps,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(padded.last().unwrap().end, Rational::from_millis(1200));
    }

    #[test]
    fn concurrent_and_matches_sequential() {
        for fps in [Rational::from_secs(30), Rational::new(30000, 1001)] {
            for times in [
                &[0, 1000, 2000, 3000, 4000, 5000, 6000][..],
                &[0, 100, 180, 270, 350, 1000][..],
            ] {
                let mut actual = shots(times);
                actual[0].usable_start = Rational::from_millis(20);
                let last = actual.len() - 1;
                actual[last].usable_end = actual[last].end - Rational::from_millis(20);
                let mut expected = actual.clone();
                let backend = Fake::default();
                let analysis = Rational::from_secs(5);
                sequential(
                    &Fake::default(),
                    Path::new("test"),
                    &mut expected,
                    analysis,
                    fps,
                )
                .unwrap();
                let mut updates = Vec::new();
                refine_shot_boundaries_with_progress(
                    &backend,
                    Path::new("test"),
                    &mut actual,
                    analysis,
                    fps,
                    &mut |p| updates.push(p),
                )
                .unwrap();
                assert_eq!(updates.first(), Some(&0.0));
                assert_eq!(updates.last(), Some(&1.0));
                assert!(updates.windows(2).all(|p| p[0] <= p[1] && p[1] <= 1.0));
                assert_eq!(backend.stream_calls.load(Ordering::SeqCst), 0);
                assert_eq!(actual, expected);
                assert!((2..=4).contains(&backend.peak.load(Ordering::SeqCst)));
                if times[1] == 100 {
                    assert!(backend.calls.load(Ordering::SeqCst) > actual.len() - 1);
                }
                for pair in actual.windows(2) {
                    assert_eq!(pair[0].end, pair[1].start);
                    assert!(pair[0].usable_start <= pair[0].usable_end);
                    assert!(pair[0].usable_end <= pair[0].end);
                    assert!(pair[1].usable_start >= pair[1].start);
                }
            }
        }
    }

    #[test]
    fn dense_stream_matches_sequential_and_reports_progress() {
        for fps in [Rational::from_secs(30), Rational::new(30000, 1001)] {
            for interval in [100, 1000] {
                let times: Vec<_> = (0..=40).map(|i| i * interval).collect();
                let mut actual = shots(&times);
                actual[0].usable_start = Rational::from_millis(20);
                let last = actual.len() - 1;
                actual[last].usable_end = actual[last].end - Rational::from_millis(20);
                let mut expected = actual.clone();
                let analysis = Rational::from_secs(5);
                sequential(
                    &Fake::default(),
                    Path::new("test"),
                    &mut expected,
                    analysis,
                    fps,
                )
                .unwrap();
                let backend = Fake::default();
                let mut updates = Vec::new();
                refine_shot_boundaries_with_progress(
                    &backend,
                    Path::new("test"),
                    &mut actual,
                    analysis,
                    fps,
                    &mut |p| updates.push(p),
                )
                .unwrap();
                assert_eq!(actual, expected);
                assert_eq!(backend.stream_calls.load(Ordering::SeqCst), 1);
                if interval == 1000 {
                    assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
                }
                assert_eq!(updates.first(), Some(&0.0));
                assert_eq!(updates.last(), Some(&1.0));
                assert!(updates.windows(2).all(|p| p[0] <= p[1] && p[1] <= 1.0));
                if interval == 1000 {
                    assert!(updates.len() > 2);
                }
            }
        }
    }

    #[test]
    fn dense_stream_propagates_errors() {
        let times: Vec<_> = (0..=40).map(|i| i * 1000).collect();
        let mut items = shots(&times);
        let original = items.clone();
        let backend = Fake {
            fail: true,
            ..Fake::default()
        };
        let mut updates = Vec::new();
        assert!(
            refine_shot_boundaries_with_progress(
                &backend,
                Path::new("test"),
                &mut items,
                Rational::from_secs(5),
                Rational::from_secs(30),
                &mut |p| updates.push(p)
            )
            .is_err()
        );
        assert_eq!(items, original);
        assert_eq!(updates, [0.0]);
        assert_eq!(backend.stream_calls.load(Ordering::SeqCst), 1);
        assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn validates_rates_skips_full_rate_and_propagates_errors() {
        let backend = Fake {
            fail: true,
            ..Fake::default()
        };
        let mut items = shots(&[0, 1000, 2000, 3000]);
        let original = items.clone();
        let path = Path::new("test");
        let fps = Rational::from_secs(30);
        assert!(refine_shot_boundaries(&backend, path, &mut items, Rational::ZERO, fps).is_err());
        assert!(refine_shot_boundaries(&backend, path, &mut items, fps, Rational::ZERO).is_err());
        refine_shot_boundaries(&backend, path, &mut items, fps, fps).unwrap();
        assert_eq!(items, original);
        assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
        assert!(
            matches!(refine_shot_boundaries(&backend, path, &mut items, Rational::from_secs(5), fps), Err(AeError::Config(message)) if message == "decode failed")
        );
        assert_eq!(items, original);
    }
}
