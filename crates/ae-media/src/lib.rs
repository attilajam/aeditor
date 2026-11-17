use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};

use ae_core::{AeError, Rational, Resolution, Result, RgbaFrame, VideoCodec};
use crossbeam_channel::Sender;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer, images::Image};
use serde::{Deserialize, Serialize};

mod ingest;

#[derive(Clone, Copy)]
enum StreamSelection<'a> {
    All,
    Range { start: i64, end: Option<i64> },
    Keyframes(&'a [Rational]),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VideoProbeInfo {
    pub width: u32,
    pub height: u32,
    pub sar_num: u32,
    pub sar_den: u32,
    pub rotation_degrees: i32,
    pub fps: Rational,
    pub duration: Rational,
    pub codec_name: String,
}

impl VideoProbeInfo {
    /// Effective display dimensions after applying rotation side-data and sample aspect ratio (SAR),
    /// prior to stretching to the final output resolution.
    pub fn oriented_dimensions(&self) -> (u32, u32) {
        let sar = if self.sar_num > 0 && self.sar_den > 0 {
            (self.sar_num as f64) / (self.sar_den as f64)
        } else {
            1.0
        };
        let sar_w = ((self.width as f64) * sar).round().max(2.0) as u32;
        let sar_h = self.height.max(2);
        match self.rotation_degrees.rem_euclid(360) {
            90 | 270 => (sar_h, sar_w),
            _ => (sar_w, sar_h),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioProbeInfo {
    pub sample_rate: u32,
    pub channels: u32,
    pub duration: Rational,
    pub codec_name: String,
}

/// Denoise before sharpening to avoid amplifying grain; shared by preview and export.
pub const LIGHT_CLEANUP_FILTER: &str = "hqdn3d=2.4:1.8:3.6:2.7,unsharp=5:5:0.85:5:5:0,eq=contrast=1.06:saturation=1.08:gamma=1.015:gamma_weight=0.7";

#[derive(Clone, Debug)]
pub struct VideoWriterConfig {
    pub output_path: PathBuf,
    pub audio_path: Option<PathBuf>,
    pub resolution: Resolution,
    pub fps: Rational,
    pub codec: VideoCodec,
    pub crf: u32,
    pub preset: String,
    pub total_duration: Option<Rational>,
    pub audio_fade_out_secs: f64,
    pub light_cleanup: bool,
}

impl Default for VideoWriterConfig {
    fn default() -> Self {
        Self {
            output_path: PathBuf::from("out.mp4"),
            audio_path: None,
            resolution: Resolution::P1080,
            fps: Rational::from_secs(30),
            codec: VideoCodec::X264,
            crf: 18,
            preset: "medium".to_string(),
            total_duration: None,
            audio_fade_out_secs: 0.35,
            light_cleanup: false,
        }
    }
}

pub trait VideoWriter: Send {
    fn write_frame(&mut self, frame: &RgbaFrame) -> Result<()>;
    fn finish(self: Box<Self>) -> Result<()>;
}

/// Abstraction over the underlying FFmpeg implementation (FFmpeg 9 CLI/pipe/sidecar or FFI).
pub trait MediaBackend: Send + Sync {
    fn probe_video(&self, path: &Path) -> Result<VideoProbeInfo>;
    fn probe_audio(&self, path: &Path) -> Result<AudioProbeInfo>;

    /// Decode the entire video into small RGB/RGBA frames (e.g. 64x36 or 48x27) in a single
    /// swscale pass (`SWS_LANCZOS | SWS_ACCURATE_RND`) and send them to `tx`.
    fn decode_scaled_stream(
        &self,
        path: &Path,
        width: u32,
        height: u32,
        sample_fps: Option<Rational>,
        tx: Sender<RgbaFrame>,
    ) -> Result<()>;

    /// Seek to the keyframe before `start`, decode forward to exact PTS, apply rotation & SAR,
    /// and stretch-to-fill `target_res` at output rate `fps`. When `source_end` is set,
    /// exclude frames at or after it and hold the last valid frame to fill `duration`.
    fn decode_segment(
        &self,
        path: &Path,
        start: Rational,
        duration: Rational,
        target_res: Resolution,
        fps: Rational,
        speed: f64,
        source_end: Option<Rational>,
    ) -> Result<Vec<RgbaFrame>>;

    /// Decode a segment and emit frames in presentation order.
    fn decode_segment_stream(
        &self,
        path: &Path,
        start: Rational,
        duration: Rational,
        target_res: Resolution,
        fps: Rational,
        speed: f64,
        source_end: Option<Rational>,
        emit: &mut dyn FnMut(RgbaFrame) -> Result<()>,
    ) -> Result<()> {
        for frame in
            self.decode_segment(path, start, duration, target_res, fps, speed, source_end)?
        {
            emit(frame)?;
        }
        Ok(())
    }

    /// Run FFmpeg's `scdet` filter graph for baseline shot-cut cross-checking in tests.
    fn run_scdet_baseline(&self, path: &Path, threshold: f32) -> Result<Vec<Rational>>;

    /// Create a frame-accurate video writer that muxes the trimmed song and sets BT.709 + SAR 1:1.
    fn create_writer(&self, config: VideoWriterConfig) -> Result<Box<dyn VideoWriter>>;

    /// M0 passthrough: re-encode video and mux audio untouched, verifying A/V sync.
    fn passthrough_remux(
        &self,
        video_path: &Path,
        audio_path: &Path,
        output_path: &Path,
        resolution: Option<Resolution>,
        codec: VideoCodec,
    ) -> Result<()>;
}

/// Stretch an in-memory `RgbaFrame` to `target` resolution using SIMD Lanczos3 (`fast_image_resize`),
/// scaling X and Y independently (stretch-to-fill, no crop or letterbox).
pub fn stretch_frame_lanczos(frame: &RgbaFrame, target: Resolution) -> Result<RgbaFrame> {
    if frame.width == target.width && frame.height == target.height {
        return Ok(frame.clone());
    }
    let mut src_buf = frame.data.clone();
    let src_image = Image::from_slice_u8(frame.width, frame.height, &mut src_buf, PixelType::U8x4)
        .map_err(|e| AeError::Decode {
            path: PathBuf::from("<memory>"),
            message: format!("fast_image_resize source init error: {e}"),
        })?;

    let mut dst_image = Image::new(target.width, target.height, PixelType::U8x4);
    let mut resizer = Resizer::new();
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
    resizer
        .resize(&src_image, &mut dst_image, &options)
        .map_err(|e| AeError::Decode {
            path: PathBuf::from("<memory>"),
            message: format!("fast_image_resize Lanczos3 error: {e}"),
        })?;

    Ok(RgbaFrame {
        width: target.width,
        height: target.height,
        data: dst_image.into_vec(),
        pts: frame.pts,
    })
}

/// Production FFmpeg 9 backend using `ffprobe` and `ffmpeg` raw-frame pipes with `swscale`
/// (`lanczos+accurate_rnd`), automatic rotation normalization, SAR 1:1, and BT.709 tagging.
#[derive(Clone, Debug, Default)]
pub struct FfmpegBackend {
    pub ffmpeg_bin: String,
    pub ffprobe_bin: String,
}

impl FfmpegBackend {
    pub fn new() -> Self {
        Self {
            ffmpeg_bin: "ffmpeg".to_string(),
            ffprobe_bin: "ffprobe".to_string(),
        }
    }

    /// Decode proxy frames and write native-rate 64x36 RGBA analysis frames in one pass.
    /// Analysis frame `n` corresponds to source time `n /` the probed frame rate.
    /// The analysis path is overwritten and may contain partial output on failure.
    pub fn decode_scaled_stream_with_analysis(
        &self,
        path: &Path,
        width: u32,
        height: u32,
        sample_fps: Option<Rational>,
        analysis_path: &Path,
        tx: Sender<RgbaFrame>,
    ) -> Result<()> {
        self.decode_scaled_stream_inner(
            path,
            width,
            height,
            sample_fps,
            tx,
            Some(analysis_path),
            StreamSelection::All,
        )
    }

    fn decode_scaled_stream_inner(
        &self,
        path: &Path,
        width: u32,
        height: u32,
        sample_fps: Option<Rational>,
        tx: Sender<RgbaFrame>,
        analysis_path: Option<&Path>,
        selection: StreamSelection<'_>,
    ) -> Result<()> {
        let info = self.probe_video(path)?;
        let effective_fps = sample_fps.unwrap_or(info.fps);
        if width == 0
            || height == 0
            || effective_fps <= Rational::ZERO
            || info.fps <= Rational::ZERO
        {
            return Err(AeError::Config(
                "frame decoding requires positive dimensions and frame rates".into(),
            ));
        }
        let first_frame = match selection {
            StreamSelection::Range { start, .. } => ingest::first_frame(start, effective_fps),
            _ => 0,
        };
        let trim = |fps: Rational| match selection {
            StreamSelection::Range { start, end } => {
                let mut filter = format!(",trim=start_pts={}", ingest::first_frame(start, fps));
                if let Some(end) = end {
                    filter.push_str(&format!(":end_pts={}", ingest::first_frame(end, fps)));
                }
                filter
            }
            _ => String::new(),
        };

        // Normalize SAR first, then stretch to target thumbnail dimensions in one fast filter pass
        let vf = if let Some(fps) = sample_fps {
            format!(
                "fps={}/{},scale=iw*sar:ih:flags=fast_bilinear,scale={}:{}:flags=fast_bilinear,setsar=1:1",
                fps.numer(),
                fps.denom(),
                width,
                height
            )
        } else {
            format!(
                "scale=iw*sar:ih:flags=fast_bilinear,scale={}:{}:flags=fast_bilinear,setsar=1:1",
                width, height
            )
        };
        let vf = format!("{vf}{}", trim(effective_fps));
        let fps_mode = if matches!(selection, StreamSelection::All) {
            "auto"
        } else {
            "passthrough"
        };

        let mut command = Command::new(&self.ffmpeg_bin);
        command.args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "auto",
            "-threads",
            "4",
            "-filter_complex_threads",
            "1",
        ]);
        match selection {
            StreamSelection::Range { start, .. } => {
                let seek = start.saturating_sub(1).max(0);
                // Retain the preceding keyframe through long VFR gaps so fps can hold the right image.
                command.args([
                    "-copyts",
                    "-start_at_zero",
                    "-noaccurate_seek",
                    "-ss",
                    &seek.to_string(),
                ]);
            }
            StreamSelection::Keyframes(_) => {
                command.args(["-skip_frame", "nokey"]);
            }
            StreamSelection::All => {}
        }
        command.arg("-i").arg(path);
        if let Some(analysis_path) = analysis_path {
            let graph = format!(
                "[0:v:0]split=2[proxy_in][analysis_in];[proxy_in]{vf}[proxy];[analysis_in]fps={}/{},scale=iw*sar:ih,scale=64:36:flags=lanczos+accurate_rnd,setsar=1:1{}[analysis]",
                info.fps.numer(),
                info.fps.denom(),
                trim(info.fps)
            );
            command
                .args([
                    "-filter_complex",
                    &graph,
                    "-map",
                    "[proxy]",
                    "-pix_fmt",
                    "rgba",
                    "-fps_mode",
                    fps_mode,
                    "-f",
                    "rawvideo",
                    "pipe:1",
                    "-map",
                    "[analysis]",
                    "-pix_fmt",
                    "rgba",
                    "-fps_mode",
                    fps_mode,
                    "-f",
                    "rawvideo",
                    "-y",
                ])
                .arg(analysis_path);
        } else {
            command.args([
                "-vf",
                &vf,
                "-pix_fmt",
                "rgba",
                "-fps_mode",
                fps_mode,
                "-f",
                "rawvideo",
                "pipe:1",
            ]);
        }
        let mut child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| AeError::Decode {
                path: path.to_path_buf(),
                message: format!("failed to spawn ffmpeg decode stream: {e}"),
            })?;

        let mut stderr = child.stderr.take().expect("stderr piped");
        let stderr_reader = std::thread::spawn(move || {
            let mut captured = Vec::new();
            let mut buf = [0; 4096];
            loop {
                match stderr.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => captured.extend_from_slice(
                        &buf[..n.min(65536usize.saturating_sub(captured.len()))],
                    ),
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
            captured
        });
        let mut stdout = child.stdout.take().expect("stdout piped");
        let frame_bytes = (width as usize) * (height as usize) * 4;
        let mut frame_idx = 0i64;

        let mut failure = None;
        loop {
            let mut buf = vec![0u8; frame_bytes];
            match read_exact_or_eof(&mut stdout, &mut buf) {
                Ok(true) => {
                    let pts = match selection {
                        StreamSelection::Keyframes(times) => match times.get(frame_idx as usize) {
                            Some(&pts) => pts,
                            None => {
                                failure = Some("decoder did not honor keyframe selection".into());
                                break;
                            }
                        },
                        _ => Rational::from_frame_index(first_frame + frame_idx, effective_fps),
                    };
                    frame_idx += 1;
                    let frame = RgbaFrame {
                        width,
                        height,
                        data: buf,
                        pts,
                    };
                    if tx.send(frame).is_err() {
                        failure = Some("frame receiver disconnected".to_string());
                        break;
                    }
                }
                Ok(false) => break,
                Err(e) => {
                    failure = Some(format!("error reading rawvideo pipe: {e}"));
                    break;
                }
            }
        }

        if failure.is_some() {
            let _ = child.kill();
        }
        drop(stdout);
        let status = child.wait();
        let stderr = stderr_reader.join().unwrap_or_default();
        if let Some(message) = failure {
            return Err(AeError::Decode {
                path: path.to_path_buf(),
                message,
            });
        }
        let status = status.map_err(|e| AeError::Decode {
            path: path.to_path_buf(),
            message: format!("failed to wait for ffmpeg: {e}"),
        })?;
        if !status.success() {
            return Err(AeError::Decode {
                path: path.to_path_buf(),
                message: format!(
                    "ffmpeg decode failed: {}",
                    String::from_utf8_lossy(&stderr).trim()
                ),
            });
        }
        if let StreamSelection::Keyframes(times) = selection {
            if frame_idx as usize != times.len() {
                return Err(AeError::Decode {
                    path: path.to_path_buf(),
                    message: "keyframe timestamps do not match decoded frames".into(),
                });
            }
        }
        Ok(())
    }
}

impl MediaBackend for FfmpegBackend {
    fn probe_video(&self, path: &Path) -> Result<VideoProbeInfo> {
        if !path.exists() {
            return Err(AeError::Decode {
                path: path.to_path_buf(),
                message: "video file does not exist".to_string(),
            });
        }
        let out = Command::new(&self.ffprobe_bin)
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,sample_aspect_ratio,r_frame_rate,avg_frame_rate,duration,codec_name:stream_side_data=rotation:format=duration",
                "-of",
                "json",
            ])
            .arg(path)
            .output()
            .map_err(|e| AeError::Decode {
                path: path.to_path_buf(),
                message: format!("failed to run ffprobe: {e}"),
            })?;

        if !out.status.success() {
            return Err(AeError::Decode {
                path: path.to_path_buf(),
                message: String::from_utf8_lossy(&out.stderr).trim().to_string(),
            });
        }

        let v: serde_json::Value =
            serde_json::from_slice(&out.stdout).map_err(|e| AeError::Decode {
                path: path.to_path_buf(),
                message: format!("invalid ffprobe JSON: {e}"),
            })?;

        let stream = v
            .get("streams")
            .and_then(|s| s.as_array())
            .and_then(|a| a.first())
            .ok_or_else(|| AeError::Decode {
                path: path.to_path_buf(),
                message: "no video stream found in file".to_string(),
            })?;

        let width = stream.get("width").and_then(|x| x.as_u64()).unwrap_or(1920) as u32;
        let height = stream
            .get("height")
            .and_then(|x| x.as_u64())
            .unwrap_or(1080) as u32;
        let codec_name = stream
            .get("codec_name")
            .and_then(|x| x.as_str())
            .unwrap_or("unknown")
            .to_string();

        let (sar_num, sar_den) = stream
            .get("sample_aspect_ratio")
            .and_then(|x| x.as_str())
            .and_then(|s| s.split_once(':'))
            .and_then(|(n, d)| Some((n.parse::<u32>().ok()?, d.parse::<u32>().ok()?)))
            .filter(|&(n, d)| n > 0 && d > 0)
            .unwrap_or((1, 1));

        let rotation_degrees = stream
            .get("side_data_list")
            .and_then(|l| l.as_array())
            .and_then(|arr| {
                arr.iter()
                    .find_map(|item| item.get("rotation").and_then(|r| r.as_i64()))
            })
            .unwrap_or(0) as i32;

        let fps_str = stream
            .get("avg_frame_rate")
            .and_then(|x| x.as_str())
            .filter(|s| *s != "0/0")
            .or_else(|| stream.get("r_frame_rate").and_then(|x| x.as_str()))
            .unwrap_or("30/1");
        let fps = Rational::parse_duration(fps_str).unwrap_or_else(|_| Rational::from_secs(30));

        let duration_secs = stream
            .get("duration")
            .and_then(|x| x.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .or_else(|| {
                v.get("format")
                    .and_then(|f| f.get("duration"))
                    .and_then(|x| x.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
            })
            .unwrap_or(0.0);

        Ok(VideoProbeInfo {
            width,
            height,
            sar_num,
            sar_den,
            rotation_degrees,
            fps,
            duration: Rational::from_secs_f64(duration_secs),
            codec_name,
        })
    }

    fn probe_audio(&self, path: &Path) -> Result<AudioProbeInfo> {
        if !path.exists() {
            return Err(AeError::Decode {
                path: path.to_path_buf(),
                message: "audio file does not exist".to_string(),
            });
        }
        let out = Command::new(&self.ffprobe_bin)
            .args([
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels,duration,codec_name:format=duration",
                "-of",
                "json",
            ])
            .arg(path)
            .output()
            .map_err(|e| AeError::Decode {
                path: path.to_path_buf(),
                message: format!("failed to run ffprobe: {e}"),
            })?;

        if !out.status.success() {
            return Err(AeError::Decode {
                path: path.to_path_buf(),
                message: String::from_utf8_lossy(&out.stderr).trim().to_string(),
            });
        }

        let v: serde_json::Value =
            serde_json::from_slice(&out.stdout).map_err(|e| AeError::Decode {
                path: path.to_path_buf(),
                message: format!("invalid ffprobe JSON: {e}"),
            })?;

        let stream = v
            .get("streams")
            .and_then(|s| s.as_array())
            .and_then(|a| a.first())
            .ok_or_else(|| AeError::Decode {
                path: path.to_path_buf(),
                message: "no audio stream found in file".to_string(),
            })?;

        let sample_rate = stream
            .get("sample_rate")
            .and_then(|x| x.as_str())
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(44100);
        let channels = stream.get("channels").and_then(|x| x.as_u64()).unwrap_or(2) as u32;
        let codec_name = stream
            .get("codec_name")
            .and_then(|x| x.as_str())
            .unwrap_or("unknown")
            .to_string();

        let duration_secs = stream
            .get("duration")
            .and_then(|x| x.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .or_else(|| {
                v.get("format")
                    .and_then(|f| f.get("duration"))
                    .and_then(|x| x.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
            })
            .unwrap_or(0.0);

        Ok(AudioProbeInfo {
            sample_rate,
            channels,
            duration: Rational::from_secs_f64(duration_secs),
            codec_name,
        })
    }

    fn decode_scaled_stream(
        &self,
        path: &Path,
        width: u32,
        height: u32,
        sample_fps: Option<Rational>,
        tx: Sender<RgbaFrame>,
    ) -> Result<()> {
        self.decode_scaled_stream_inner(
            path,
            width,
            height,
            sample_fps,
            tx,
            None,
            StreamSelection::All,
        )
    }

    fn decode_segment(
        &self,
        path: &Path,
        start: Rational,
        duration: Rational,
        target_res: Resolution,
        fps: Rational,
        speed: f64,
        source_end: Option<Rational>,
    ) -> Result<Vec<RgbaFrame>> {
        let mut frames = Vec::new();
        self.decode_segment_stream(
            path,
            start,
            duration,
            target_res,
            fps,
            speed,
            source_end,
            &mut |frame| {
                frames.push(frame);
                Ok(())
            },
        )?;
        Ok(frames)
    }

    fn decode_segment_stream(
        &self,
        path: &Path,
        start: Rational,
        duration: Rational,
        target_res: Resolution,
        fps: Rational,
        speed: f64,
        source_end: Option<Rational>,
        emit: &mut dyn FnMut(RgbaFrame) -> Result<()>,
    ) -> Result<()> {
        if source_end.is_some_and(|end| end <= start) {
            return Err(AeError::Config(
                "source end must follow source start".into(),
            ));
        }
        if fps <= Rational::ZERO || !speed.is_finite() || speed <= 0.0 {
            return Err(AeError::Config(
                "segment frame rate and speed must be positive".into(),
            ));
        }
        let target_frames = ((duration.as_secs_f64() * fps.as_secs_f64()).round() as usize).max(1);
        let mut src_duration = duration.as_secs_f64() * speed.max(0.1);
        if let Some(end) = source_end {
            src_duration = src_duration.min((end - start).as_secs_f64());
        }

        // Apply rotation (handled automatically by ffmpeg auto-rotate), normalize SAR,
        // retime if speed != 1.0, resample to output fps, and stretch X/Y independently with Lanczos
        let setpts = if (speed - 1.0).abs() > 1e-3 {
            format!("setpts=PTS/{:.6},", speed)
        } else {
            String::new()
        };
        // Trim before resampling so transition handles cannot include the following scene.
        let trim = source_end
            .map(|end| {
                // Leave one microsecond for the input seek's decimal rounding.
                format!(
                    "trim=end={:.6},",
                    ((end - start).as_secs_f64() - 0.000001).max(0.0)
                )
            })
            .unwrap_or_default();
        let vf = format!(
            "{trim}{setpts}fps={}/{},scale=iw*sar:ih,scale={}:{}:flags=lanczos+accurate_rnd,setsar=1:1",
            fps.numer(),
            fps.denom(),
            target_res.width,
            target_res.height
        );

        let mut child = Command::new(&self.ffmpeg_bin)
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-filter_threads",
                "1",
                "-hwaccel",
                "auto",
                "-threads",
                "2",
                "-ss",
                &format!("{:.6}", start.as_secs_f64().max(0.0)),
                "-t",
                &format!("{:.6}", (src_duration + 0.15).max(0.05)),
                "-i",
            ])
            .arg(path)
            .args([
                "-vf",
                &vf,
                "-frames:v",
                &target_frames.to_string(),
                "-pix_fmt",
                "rgba",
                "-f",
                "rawvideo",
                "pipe:1",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| AeError::Decode {
                path: path.to_path_buf(),
                message: format!("failed to spawn ffmpeg segment decoder: {e}"),
            })?;

        let mut stdout = child.stdout.take().expect("stdout piped");
        let mut stderr = child.stderr.take().expect("stderr piped");
        let stderr_reader = std::thread::spawn(move || {
            let mut captured = Vec::new();
            let mut buf = [0; 4096];
            loop {
                match stderr.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => captured.extend_from_slice(
                        &buf[..n.min(65536usize.saturating_sub(captured.len()))],
                    ),
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
            captured
        });
        let frame_bytes = (target_res.width as usize) * (target_res.height as usize) * 4;
        let mut last = None;
        let mut count = 0;
        let mut failure = None;

        for idx in 0..target_frames {
            let mut buf = vec![0u8; frame_bytes];
            match read_exact_or_eof(&mut stdout, &mut buf) {
                Ok(true) => {
                    let pts = start + Rational::from_frame_index(idx as i64, fps);
                    let frame = RgbaFrame {
                        width: target_res.width,
                        height: target_res.height,
                        data: buf,
                        pts,
                    };
                    last = Some(frame.clone());
                    if let Err(e) = emit(frame) {
                        failure = Some(e);
                        break;
                    }
                    count += 1;
                }
                Ok(false) => break,
                Err(e) => {
                    failure = Some(AeError::Decode {
                        path: path.to_path_buf(),
                        message: format!("segment decode pipe read failed: {e}"),
                    });
                    break;
                }
            }
        }
        if failure.is_some() {
            let _ = child.kill();
        }
        drop(stdout);
        let status = child.wait();
        let stderr = stderr_reader.join().unwrap_or_default();
        if let Some(error) = failure {
            return Err(error);
        }
        let status = status.map_err(|e| AeError::Decode {
            path: path.to_path_buf(),
            message: format!("failed waiting for ffmpeg segment decoder: {e}"),
        })?;
        if !status.success() {
            return Err(AeError::Decode {
                path: path.to_path_buf(),
                message: format!(
                    "ffmpeg segment decoder failed ({status}): {}",
                    String::from_utf8_lossy(&stderr).trim()
                ),
            });
        }

        if source_end.is_some() && last.is_none() {
            return Err(AeError::Decode {
                path: path.to_path_buf(),
                message: "no frames in the selected source range".into(),
            });
        }

        // Pad by repeating last frame if we hit EOF slightly early
        while count < target_frames {
            let pts = start + Rational::from_frame_index(count as i64, fps);
            let padded = if let Some(last) = &last {
                let mut padded: RgbaFrame = last.clone();
                padded.pts = pts;
                padded
            } else {
                RgbaFrame::new(target_res.width, target_res.height, pts)
            };
            emit(padded)?;
            count += 1;
        }

        Ok(())
    }

    fn run_scdet_baseline(&self, path: &Path, threshold: f32) -> Result<Vec<Rational>> {
        let vf = format!("scdet=threshold={:.2}:sc_pass=1", threshold);
        let out = Command::new(&self.ffmpeg_bin)
            .args(["-hide_banner", "-nostats", "-i"])
            .arg(path)
            .args(["-vf", &vf, "-f", "null", "-"])
            .stderr(Stdio::piped())
            .stdout(Stdio::null())
            .output()
            .map_err(|e| AeError::Decode {
                path: path.to_path_buf(),
                message: format!("failed to run ffmpeg scdet filter: {e}"),
            })?;

        let stderr = String::from_utf8_lossy(&out.stderr);
        let mut cuts = Vec::new();
        for line in stderr.lines() {
            if let Some(pos) = line.find("lavfi.scd.time:") {
                let rest = &line[pos + "lavfi.scd.time:".len()..];
                let num_str: String = rest
                    .trim()
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '.')
                    .collect();
                if let Ok(sec) = num_str.parse::<f64>() {
                    cuts.push(Rational::from_secs_f64(sec));
                }
            }
        }
        Ok(cuts)
    }

    fn create_writer(&self, config: VideoWriterConfig) -> Result<Box<dyn VideoWriter>> {
        if let Some(parent) = config.output_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        let mut cmd = Command::new(&self.ffmpeg_bin);
        cmd.args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &format!("{}x{}", config.resolution.width, config.resolution.height),
            "-r",
            &format!("{}/{}", config.fps.numer(), config.fps.denom()),
            "-i",
            "pipe:0",
        ]);

        if let Some(ref audio) = config.audio_path {
            cmd.arg("-i").arg(audio);
        }

        // Video filter sets SAR 1:1 explicitly so players never re-letterbox stretched output
        let vf = if config.light_cleanup {
            format!("setsar=1:1,format=yuv420p,{LIGHT_CLEANUP_FILTER}")
        } else {
            "setsar=1:1,format=yuv420p".into()
        };
        cmd.args(["-vf", &vf]);
        cmd.args(["-c:v", config.codec.ffmpeg_encoder_name()]);

        match config.codec {
            VideoCodec::X264 | VideoCodec::X265 => {
                cmd.args(["-crf", &config.crf.to_string(), "-preset", &config.preset]);
            }
            VideoCodec::Av1 => {
                cmd.args(["-crf", &config.crf.to_string(), "-preset", "8"]);
            }
            VideoCodec::Nvenc | VideoCodec::Vaapi => {
                cmd.args(["-qp", &config.crf.to_string()]);
            }
        }

        // Standard BT.709 color tags
        cmd.args([
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
        ]);

        if config.audio_path.is_some() {
            cmd.args([
                "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k",
            ]);
            if let Some(dur) = config.total_duration {
                let dur_s = dur.as_secs_f64();
                if config.audio_fade_out_secs > 0.0 && dur_s > config.audio_fade_out_secs * 2.0 {
                    let fade_start = (dur_s - config.audio_fade_out_secs).max(0.0);
                    cmd.args([
                        "-af",
                        &format!(
                            "afade=t=out:st={:.4}:d={:.4}",
                            fade_start, config.audio_fade_out_secs
                        ),
                    ]);
                }
                cmd.args(["-t", &format!("{:.6}", dur_s)]);
            } else {
                cmd.arg("-shortest");
            }
        }

        cmd.args(["-movflags", "+faststart"]);
        cmd.arg(&config.output_path);

        cmd.stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| AeError::Encode {
            path: config.output_path.clone(),
            message: format!("failed to spawn ffmpeg encoder: {e}"),
        })?;

        let stdin = child.stdin.take().expect("stdin piped");
        Ok(Box::new(FfmpegPipeWriter {
            child,
            stdin: Some(stdin),
            config,
        }))
    }

    fn passthrough_remux(
        &self,
        video_path: &Path,
        audio_path: &Path,
        output_path: &Path,
        resolution: Option<Resolution>,
        codec: VideoCodec,
    ) -> Result<()> {
        let v_info = self.probe_video(video_path)?;
        let a_info = self.probe_audio(audio_path)?;
        let out_res = resolution.unwrap_or_else(|| {
            let (w, h) = v_info.oriented_dimensions();
            Resolution::new(w, h)
        });
        let target_dur = v_info.duration.min(a_info.duration);

        let vf = format!(
            "scale=iw*sar:ih,scale={}:{}:flags=lanczos+accurate_rnd,setsar=1:1,format=yuv420p",
            out_res.width, out_res.height
        );

        let status = Command::new(&self.ffmpeg_bin)
            .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
            .arg(video_path)
            .arg("-i")
            .arg(audio_path)
            .args([
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-vf",
                &vf,
                "-c:v",
                codec.ffmpeg_encoder_name(),
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-t",
                &format!("{:.6}", target_dur.as_secs_f64()),
                "-movflags",
                "+faststart",
            ])
            .arg(output_path)
            .status()
            .map_err(|e| AeError::Encode {
                path: output_path.to_path_buf(),
                message: format!("passthrough ffmpeg invocation failed: {e}"),
            })?;

        if !status.success() {
            return Err(AeError::Encode {
                path: output_path.to_path_buf(),
                message: "ffmpeg passthrough exited with non-zero status".to_string(),
            });
        }
        Ok(())
    }
}

struct FfmpegPipeWriter {
    child: Child,
    stdin: Option<ChildStdin>,
    config: VideoWriterConfig,
}

impl VideoWriter for FfmpegPipeWriter {
    fn write_frame(&mut self, frame: &RgbaFrame) -> Result<()> {
        let expected =
            (self.config.resolution.width as usize) * (self.config.resolution.height as usize) * 4;
        let resized;
        let data = if frame.data.len() == expected {
            &frame.data
        } else {
            resized = stretch_frame_lanczos(frame, self.config.resolution)?;
            &resized.data
        };

        if let Some(ref mut stdin) = self.stdin {
            stdin.write_all(data).map_err(|e| AeError::Encode {
                path: self.config.output_path.clone(),
                message: format!("failed writing frame to encoder stdin: {e}"),
            })?;
        }
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<()> {
        drop(self.stdin.take());
        let output = self.child.wait_with_output().map_err(|e| AeError::Encode {
            path: self.config.output_path.clone(),
            message: format!("failed waiting on ffmpeg encoder: {e}"),
        })?;
        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            return Err(AeError::Encode {
                path: self.config.output_path.clone(),
                message: format!("ffmpeg encoder failed: {}", err.trim()),
            });
        }
        Ok(())
    }
}

fn read_exact_or_eof<R: Read>(reader: &mut R, mut buf: &mut [u8]) -> std::io::Result<bool> {
    let total = buf.len();
    let mut read_bytes = 0;
    while !buf.is_empty() {
        match reader.read(buf) {
            Ok(0) => {
                return if read_bytes == 0 {
                    Ok(false)
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "incomplete raw video frame",
                    ))
                };
            }
            Ok(n) => {
                read_bytes += n;
                buf = &mut buf[n..];
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(read_bytes == total)
}

/// Generate a synthetic multi-shot test video with distinct visual patterns and known hard cut timestamps.
/// Useful for automated integration tests and fixture generation.
pub fn generate_synthetic_multi_shot_video(
    output_path: &Path,
    resolution: Resolution,
    fps: Rational,
    shot_specs: &[([u8; 3], [u8; 3], usize)], // (primary_rgb, accent_rgb, num_frames)
) -> Result<Vec<Rational>> {
    let backend = FfmpegBackend::new();
    let mut writer = backend.create_writer(VideoWriterConfig {
        output_path: output_path.to_path_buf(),
        audio_path: None,
        resolution,
        fps,
        codec: VideoCodec::X264,
        crf: 16,
        preset: "ultrafast".to_string(),
        total_duration: None,
        audio_fade_out_secs: 0.0,
        light_cleanup: false,
    })?;

    let mut cut_times = Vec::new();
    let mut global_frame = 0i64;

    for (shot_idx, &(rgb1, rgb2, num_frames)) in shot_specs.iter().enumerate() {
        if shot_idx > 0 {
            cut_times.push(Rational::from_frame_index(global_frame, fps));
        }
        for local_f in 0..num_frames {
            let pts = Rational::from_frame_index(global_frame, fps);
            let mut frame = RgbaFrame::new(resolution.width, resolution.height, pts);
            let w = resolution.width as usize;
            let h = resolution.height as usize;
            let shift = (local_f * 2) % w.max(1);

            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) * 4;
                    let pattern = match shot_idx % 4 {
                        0 => ((x + shift) / 16 + y / 16) % 2 == 0,
                        1 => {
                            let dx = (x as isize) - ((w / 2) as isize);
                            let dy = (y as isize) - ((h / 2) as isize);
                            (dx * dx + dy * dy) < ((h.min(w) / 3) as isize).pow(2)
                        }
                        2 => ((x + shift) % 24) < 12,
                        _ => ((y + shift) % 24) < 12,
                    };
                    let c = if pattern { rgb1 } else { rgb2 };
                    frame.data[idx] = c[0];
                    frame.data[idx + 1] = c[1];
                    frame.data[idx + 2] = c[2];
                    frame.data[idx + 3] = 255;
                }
            }
            writer.write_frame(&frame)?;
            global_frame += 1;
        }
    }

    writer.finish()?;
    Ok(cut_times)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleanup_boosts_edges_and_color_without_clipping_skin_midtones() {
        let tmp = tempfile::tempdir().unwrap();
        let backend = FfmpegBackend::new();
        let resolution = Resolution::new(64, 64);
        let fps = Rational::from_secs(30);
        let mut frame = RgbaFrame::solid(64, 64, [90, 90, 90, 255], Rational::ZERO);
        for y in 0..64 {
            for x in 32..64 {
                let offset = (y * 64 + x) * 4;
                frame.data[offset..offset + 3].fill(160);
            }
        }
        for y in 40..64 {
            for x in 0..64 {
                let offset = (y * 64 + x) * 4;
                frame.data[offset..offset + 3].copy_from_slice(&[180, 135, 115]);
            }
        }
        let mut results = Vec::new();
        for light_cleanup in [false, true] {
            let path = tmp.path().join(format!("cleanup-{light_cleanup}.mp4"));
            let mut writer = backend
                .create_writer(VideoWriterConfig {
                    output_path: path.clone(),
                    resolution,
                    fps,
                    crf: 0,
                    light_cleanup,
                    ..Default::default()
                })
                .unwrap();
            writer.write_frame(&frame).unwrap();
            writer.finish().unwrap();
            results.push(
                backend
                    .decode_segment(
                        &path,
                        Rational::ZERO,
                        Rational::from_frame_index(1, fps),
                        resolution,
                        fps,
                        1.0,
                        None,
                    )
                    .unwrap()
                    .remove(0),
            );
        }
        let edge =
            |frame: &RgbaFrame| frame.pixel(32, 16)[0] as i16 - frame.pixel(31, 16)[0] as i16;
        assert!(
            edge(&results[1]) > edge(&results[0]) + 15,
            "edges should look crisper"
        );
        let before = results[0].pixel(16, 52);
        let after = results[1].pixel(16, 52);
        assert!(
            after[0] - after[2] > before[0] - before[2],
            "color should be richer"
        );
        for c in 0..3 {
            assert!(after[c] > 10 && after[c] < 245, "skin midtones clipped");
            assert!(
                after[c].abs_diff(before[c]) <= 12,
                "skin color shifted too far"
            );
        }
    }

    #[test]
    fn test_stretch_frame_lanczos_non_uniform() {
        let src = RgbaFrame::solid(320, 180, [200, 100, 50, 255], Rational::ZERO);
        let dst = stretch_frame_lanczos(&src, Resolution::new(180, 320)).unwrap();
        assert_eq!(dst.width, 180);
        assert_eq!(dst.height, 320);
        assert_eq!(dst.pixel(90, 160), [200, 100, 50, 255]);
    }

    #[test]
    fn test_synthetic_video_probe_and_passthrough() {
        let tmp = tempfile::tempdir().unwrap();
        let vid_path = tmp.path().join("test_shots.mp4");
        let fps = Rational::from_secs(30);
        let cuts = generate_synthetic_multi_shot_video(
            &vid_path,
            Resolution::new(320, 180),
            fps,
            &[
                ([220, 40, 40], [30, 10, 10], 15),
                ([40, 220, 60], [10, 30, 10], 15),
            ],
        )
        .unwrap();
        assert_eq!(cuts.len(), 1);

        let backend = FfmpegBackend::new();
        let info = backend.probe_video(&vid_path).unwrap();
        assert_eq!(info.width, 320);
        assert_eq!(info.height, 180);
        assert!((info.duration.as_secs_f64() - 1.0).abs() < 0.15);
    }

    #[test]
    fn analysis_stream_matches_native_segment_and_proxy() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("fractional.mp4");
        let analysis = tmp.path().join("analysis.rgba");
        let fps = Rational::new(30000, 1001);
        generate_synthetic_multi_shot_video(
            &path,
            Resolution::new(160, 90),
            fps,
            &[
                ([220, 40, 40], [30, 10, 10], 30),
                ([40, 220, 60], [10, 30, 10], 30),
            ],
        )
        .unwrap();
        let backend = FfmpegBackend::new();
        assert_eq!(backend.probe_video(&path).unwrap().fps, fps);
        let sample_fps = Some(Rational::from_secs(7));
        let (tx, rx) = crossbeam_channel::unbounded();
        backend
            .decode_scaled_stream(&path, 80, 46, sample_fps, tx)
            .unwrap();
        let proxy: Vec<_> = rx.iter().collect();
        let (tx, rx) = crossbeam_channel::unbounded();
        backend
            .decode_scaled_stream_with_analysis(&path, 80, 46, sample_fps, &analysis, tx)
            .unwrap();
        let combined: Vec<_> = rx.iter().collect();
        assert_eq!(proxy.len(), combined.len());
        for (a, b) in proxy.iter().zip(&combined) {
            assert_eq!(
                (a.width, a.height, a.pts, &a.data),
                (b.width, b.height, b.pts, &b.data)
            );
        }
        let native = backend
            .decode_segment(
                &path,
                Rational::ZERO,
                Rational::from_frame_index(60, fps),
                Resolution::new(64, 36),
                fps,
                1.0,
                None,
            )
            .unwrap();
        let raw = std::fs::read(&analysis).unwrap();
        assert_eq!(raw.len(), native.len() * 64 * 36 * 4);
        assert_eq!(native.len(), 60);
        for (i, (bytes, frame)) in raw.chunks_exact(64 * 36 * 4).zip(&native).enumerate() {
            assert_eq!((frame.width, frame.height), (64, 36));
            assert_eq!(frame.pts, Rational::from_frame_index(i as i64, fps));
            assert_eq!(bytes, frame.data);
        }
        let (tx, _rx) = crossbeam_channel::unbounded();
        assert!(
            backend
                .decode_scaled_stream_with_analysis(
                    &path,
                    80,
                    46,
                    sample_fps,
                    &tmp.path().join("missing/out"),
                    tx,
                )
                .is_err()
        );
        let (tx, rx) = crossbeam_channel::bounded(1);
        drop(rx);
        assert!(
            backend
                .decode_scaled_stream_with_analysis(&path, 80, 46, sample_fps, &analysis, tx)
                .is_err()
        );
    }

    #[test]
    fn segment_stream_matches_collector_and_pads_last_frame() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("segment.mp4");
        let fps = Rational::from_secs(10);
        let res = Resolution::new(32, 32);
        generate_synthetic_multi_shot_video(
            &path,
            res,
            fps,
            &[
                ([220, 40, 40], [30, 10, 10], 5),
                ([40, 220, 60], [10, 30, 10], 5),
            ],
        )
        .unwrap();
        let backend = FfmpegBackend::new();
        let start = Rational::from_secs_f64(0.1);
        let duration = Rational::from_secs(1);
        let end = Some(Rational::from_secs_f64(0.5));
        let expected = backend
            .decode_segment(&path, start, duration, res, fps, 1.0, end)
            .unwrap();
        let mut actual = Vec::new();
        let object: &dyn MediaBackend = &backend;
        object
            .decode_segment_stream(&path, start, duration, res, fps, 1.0, end, &mut |frame| {
                actual.push(frame);
                Ok(())
            })
            .unwrap();
        assert_eq!(actual.len(), 10);
        for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(a.data, b.data);
            assert_eq!(a.pts, start + Rational::from_frame_index(i as i64, fps));
            assert_eq!(a.pts, b.pts);
        }
        assert_eq!(actual[3].data, actual[9].data);

        let mut count = 0;
        let error = backend
            .decode_segment_stream(&path, start, duration, res, fps, 1.0, end, &mut |_| {
                count += 1;
                Err(AeError::Config("stop".into()))
            })
            .unwrap_err();
        assert_eq!(count, 1);
        assert!(matches!(error, AeError::Config(message) if message == "stop"));
        assert!(
            backend
                .decode_segment(
                    &tmp.path().join("missing.mp4"),
                    start,
                    duration,
                    res,
                    fps,
                    1.0,
                    None
                )
                .is_err()
        );
    }
}
