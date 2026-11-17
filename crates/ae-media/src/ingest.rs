use super::*;

pub(super) fn first_frame(seconds: i64, fps: Rational) -> i64 {
    (Rational::from_secs(seconds).0 * fps.0).ceil().to_integer()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(path: &Path, fps: &str, offset: bool, variable: bool) {
        let mut command = Command::new("ffmpeg");
        command.args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc2=size=96x54:rate={fps}"),
            "-frames:v",
            "2031",
            "-vf",
            if variable {
                "setsar=2/1,setpts=PTS+if(gte(N\\,300)\\,15/TB\\,0)"
            } else {
                "setsar=2/1"
            },
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-g",
            "250",
            "-bf",
            "3",
        ]);
        if offset {
            command.args(["-output_ts_offset", "3.25"]);
        }
        let output = command.arg("-y").arg(path).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn parallel_scan_matches_sequential_pixels_timestamps_and_native_file() {
        let dir = tempfile::tempdir().unwrap();
        let backend = FfmpegBackend::new();
        for (rate, offset, variable) in [
            ("30", false, false),
            ("30000/1001", false, false),
            ("30000/1001", true, false),
            ("30", false, true),
        ] {
            let path = dir.path().join("clip.mp4");
            fixture(&path, rate, offset, variable);
            let before_file = dir.path().join("before.rgba");
            let after_file = dir.path().join("after.rgba");
            let fps = Rational::from_secs(7);
            let (tx, rx) = crossbeam_channel::unbounded();
            backend
                .decode_scaled_stream_with_analysis(&path, 80, 46, Some(fps), &before_file, tx)
                .unwrap();
            let before: Vec<_> = rx.into_iter().collect();
            let (tx, rx) = crossbeam_channel::unbounded();
            backend
                .decode_scaled_stream_parallel(&path, 80, 46, fps, Some(&after_file), tx)
                .unwrap();
            let after: Vec<_> = rx.into_iter().collect();
            assert_eq!(
                before.len(),
                after.len(),
                "proxy count: {rate} offset={offset}"
            );
            for (i, (a, b)) in before.iter().zip(&after).enumerate() {
                assert_eq!(a.pts, b.pts, "PTS {i}: {rate} offset={offset}");
                assert!(a.data == b.data, "pixels {i}: {rate} offset={offset}");
            }
            let a = std::fs::read(&before_file).unwrap();
            let b = std::fs::read(&after_file).unwrap();
            assert_eq!(a.len(), b.len(), "native count: {rate} offset={offset}");
            for (i, (a, b)) in a.chunks_exact(9216).zip(b.chunks_exact(9216)).enumerate() {
                assert!(a == b, "native pixels {i}: {rate} offset={offset}");
            }
            let (tx, rx) = crossbeam_channel::unbounded();
            backend
                .decode_scaled_stream_parallel(&path, 80, 46, fps, None, tx)
                .unwrap();
            let proxy_only: Vec<_> = rx.into_iter().collect();
            assert_eq!(before.len(), proxy_only.len());
            assert!(
                before
                    .iter()
                    .zip(proxy_only)
                    .all(|(a, b)| a.pts == b.pts && a.data == b.data)
            );
        }
        let (tx, rx) = crossbeam_channel::bounded(1);
        drop(rx);
        assert!(
            backend
                .decode_scaled_stream_parallel(
                    &dir.path().join("clip.mp4"),
                    80,
                    46,
                    Rational::from_secs(7),
                    None,
                    tx
                )
                .is_err()
        );
    }

    #[test]
    fn keyframe_preview_preserves_sparse_timestamps_and_handles_failures() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clip.mp4");
        fixture(&path, "30", false, false);
        let backend = FfmpegBackend::new();
        let frames = backend.decode_keyframes(&path, 80, 46).unwrap();
        assert!(frames.len() > 2 && frames.len() < 20);
        assert_eq!(frames[0].pts, Rational::ZERO);
        assert!(
            frames
                .windows(2)
                .all(|p| p[1].pts - p[0].pts > Rational::ONE)
        );
        assert!(
            frames
                .iter()
                .all(|f| f.width == 80 && f.height == 46 && f.data.len() == 80 * 46 * 4)
        );
        assert!(
            backend
                .decode_keyframes(&dir.path().join("missing.mp4"), 80, 46)
                .is_err()
        );
        assert!(backend.decode_keyframes(&path, 0, 46).is_err());
        assert_eq!(first_frame(17, Rational::new(30000, 1001)), 510);
    }
}

impl FfmpegBackend {
    /// Decode a small, timestamped keyframe reel for provisional scene previews.
    pub fn decode_keyframes(&self, path: &Path, width: u32, height: u32) -> Result<Vec<RgbaFrame>> {
        let output = Command::new(&self.ffprobe_bin)
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_packets",
                "-show_entries",
                "packet=pts_time,flags",
                "-of",
                "json",
            ])
            .arg(path)
            .output()?;
        if !output.status.success() {
            return Err(AeError::Decode {
                path: path.into(),
                message: "could not read keyframe index".into(),
            });
        }
        let index: serde_json::Value = serde_json::from_slice(&output.stdout)?;
        let packets = index["packets"].as_array().ok_or_else(|| AeError::Decode {
            path: path.into(),
            message: "missing packet index".into(),
        })?;
        let time = |p: &serde_json::Value| {
            p["pts_time"]
                .as_str()?
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .map(Rational::from_secs_f64)
        };
        let origin = packets
            .iter()
            .filter_map(time)
            .min()
            .unwrap_or(Rational::ZERO);
        let mut times: Vec<_> = packets
            .iter()
            .filter(|p| p["flags"].as_str().is_some_and(|f| f.contains('K')))
            .filter_map(time)
            .map(|pts| pts - origin)
            .collect();
        times.sort_unstable();
        if times.is_empty() {
            return Ok(Vec::new());
        }
        // shortcut: skip oversized/intra-only indexes; use the accurate scan instead of a second full reel.
        if times.len() > 2048 {
            return Ok(Vec::new());
        }
        let (tx, rx) = crossbeam_channel::unbounded();
        self.decode_scaled_stream_inner(
            path,
            width,
            height,
            None,
            tx,
            None,
            StreamSelection::Keyframes(&times),
        )?;
        Ok(rx.into_iter().collect())
    }

    /// Scan at most four sections concurrently, retaining the original global frame grids.
    pub fn decode_scaled_stream_parallel(
        &self,
        path: &Path,
        width: u32,
        height: u32,
        sample_fps: Rational,
        analysis_path: Option<&Path>,
        tx: Sender<RgbaFrame>,
    ) -> Result<()> {
        let info = self.probe_video(path)?;
        let workers = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(4);
        if info.duration < Rational::from_secs(60) || workers == 1 {
            return self.decode_scaled_stream_inner(
                path,
                width,
                height,
                Some(sample_fps),
                tx,
                analysis_path,
                StreamSelection::All,
            );
        }
        let scratch = tempfile::tempdir()?;
        std::thread::scope(|scope| {
            let mut receivers = Vec::new();
            let mut handles = Vec::new();
            for i in 0..workers {
                let start =
                    (info.duration.as_secs_f64() * i as f64 / workers as f64).floor() as i64;
                let end = (i + 1 < workers).then(|| {
                    (info.duration.as_secs_f64() * (i + 1) as f64 / workers as f64).floor() as i64
                });
                let analysis = analysis_path.map(|_| scratch.path().join(format!("{i}.rgba")));
                // Queues own the same buffers as the final proxy reel; draining moves them without copies.
                let (part_tx, part_rx) = crossbeam_channel::unbounded();
                receivers.push(part_rx);
                handles.push(scope.spawn(move || {
                    self.decode_scaled_stream_inner(
                        path,
                        width,
                        height,
                        Some(sample_fps),
                        part_tx,
                        analysis.as_deref(),
                        StreamSelection::Range { start, end },
                    )
                }));
            }
            for rx in &receivers {
                for frame in rx {
                    if tx.send(frame).is_err() {
                        drop(receivers);
                        return Err(AeError::Decode {
                            path: path.into(),
                            message: "frame receiver disconnected".into(),
                        });
                    }
                }
            }
            for handle in handles {
                handle.join().map_err(|_| AeError::Decode {
                    path: path.into(),
                    message: "parallel scan worker panicked".into(),
                })??;
            }
            if let Some(analysis_path) = analysis_path {
                let mut output = std::fs::File::create(analysis_path)?;
                for i in 0..workers {
                    let mut input = std::fs::File::open(scratch.path().join(format!("{i}.rgba")))?;
                    std::io::copy(&mut input, &mut output)?;
                }
            }
            Ok(())
        })
    }
}
