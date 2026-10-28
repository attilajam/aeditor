use std::path::PathBuf;

use ae_beats::{BeatDetectorConfig, detect_bass_beats};
use ae_core::{
    Beat, BeatMode, CacheManager, FreqBand, ModelManifest, Rational, Resolution, SeparationMode,
    SequenceOrder, Shot, ShotDetectorMode, SimilarityMode, SpeedRange, Timeline, VideoCodec,
};
use ae_media::{FfmpegBackend, MediaBackend};
use ae_render::{RenderConfig, render_timeline};
use ae_sequencer::{SequencerConfig, sequence_timeline};
use ae_shots::{ShotDetectorConfig, detect_shots};
use ae_similarity::{ShotDescriptorBank, SimilarityConfig, extract_shot_descriptors};
use ae_transitions::{StyleTomlConfig, list_registered_transitions};
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Parser, Debug)]
#[command(
    name = "auto-editor",
    version,
    about = "Automatic beat-synced match-cut video editor in Rust"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Input source video clip
    #[arg(short = 'v', long = "video")]
    pub video: Option<PathBuf>,

    /// Input audio / song file
    #[arg(short = 'a', long = "audio")]
    pub audio: Option<PathBuf>,

    /// Output rendered video file
    #[arg(short = 'o', long = "output", default_value = "out.mp4")]
    pub output: PathBuf,

    /// Target output resolution (stretched to fill: e.g. 1080x1920, 720p, 1080p, 4k, vertical)
    #[arg(short = 'r', long = "resolution")]
    pub resolution: Option<Resolution>,

    /// Shot detection mode (classic | neural)
    #[arg(long = "shots", default_value = "classic")]
    pub shots: ShotDetectorMode,

    /// Bass isolation mode (band | hpss | demucs)
    #[arg(long = "separation", default_value = "hpss")]
    pub separation: SeparationMode,

    /// Bass frequency band in Hz (e.g. 30-250)
    #[arg(long = "bass-band", default_value = "30-250")]
    pub bass_band: FreqBand,

    /// Beat tracking regularization mode (onsets | grid | hybrid)
    #[arg(long = "beat-mode", default_value = "hybrid")]
    pub beat_mode: BeatMode,

    /// Minimum time between cuts (e.g. 0.25s, 250ms)
    #[arg(long = "min-cut", default_value = "0.25s")]
    pub min_cut: Rational,

    /// Visual similarity descriptor mode (fast | neural)
    #[arg(long = "similarity", default_value = "fast")]
    pub similarity: SimilarityMode,

    /// Shot sequencing order (similarity | chronological | mixed)
    #[arg(long = "order", default_value = "similarity")]
    pub order: SequenceOrder,

    /// Allowed playback speed range to fit shots to beat slots (e.g. 0.8-1.25)
    #[arg(long = "allow-speed", default_value = "0.8-1.25")]
    pub allow_speed: SpeedRange,

    /// Optional TOML transition policy config
    #[arg(long = "transitions")]
    pub transitions: Option<PathBuf>,

    /// Directory containing custom *.wgsl transition shaders
    #[arg(long = "shaders-dir", default_value = "transitions")]
    pub shaders_dir: PathBuf,

    /// Enable wgpu GPU transition backend
    #[arg(long = "gpu")]
    pub gpu: bool,

    /// Output frame rate (defaults to input video frame rate)
    #[arg(long = "fps")]
    pub fps: Option<Rational>,

    /// Output video encoder codec (x264 | x265 | av1 | nvenc | vaapi)
    #[arg(long = "codec", default_value = "x264")]
    pub codec: VideoCodec,

    /// Deterministic RNG seed for beam search tie-breaking
    #[arg(long = "seed", default_value_t = 42)]
    pub seed: u64,

    /// Write detected shots to a JSON file
    #[arg(long = "dump-shots")]
    pub dump_shots: Option<PathBuf>,

    /// Write bass spectrogram PNG and click-track WAV to directory
    #[arg(long = "debug-beats")]
    pub debug_beats: Option<PathBuf>,

    /// Write computed edit Timeline to a JSON file
    #[arg(long = "dump-timeline")]
    pub dump_timeline: Option<PathBuf>,

    /// Render directly from an existing Timeline JSON file without re-running analysis
    #[arg(long = "from-timeline")]
    pub from_timeline: Option<PathBuf>,

    /// Directory containing ONNX models and manifest.toml
    #[arg(long = "models-dir", default_value = "models")]
    pub models_dir: PathBuf,

    /// Analyze and generate timeline only, skipping video rendering
    #[arg(long = "dry-run")]
    pub dry_run: bool,

    /// Disable reading/writing `.ae-cache`
    #[arg(long = "no-cache")]
    pub no_cache: bool,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// M0 Passthrough: re-encode video with muxed audio and optional stretched resolution
    Passthrough {
        #[arg(short = 'v', long = "video")]
        video: PathBuf,
        #[arg(short = 'a', long = "audio")]
        audio: PathBuf,
        #[arg(short = 'o', long = "output", default_value = "passthrough.mp4")]
        output: PathBuf,
        #[arg(short = 'r', long = "resolution")]
        resolution: Option<Resolution>,
        #[arg(long = "codec", default_value = "x264")]
        codec: VideoCodec,
    },
    /// List all registered CPU transitions and discovered WGSL GPU shaders
    ListTransitions,
    /// Inspect model manifest entries in models/manifest.toml
    Models {
        #[arg(long = "manifest", default_value = "models/manifest.toml")]
        manifest: PathBuf,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();
    run_cli(cli)
}

pub fn run_cli(cli: Cli) -> Result<()> {
    let backend = FfmpegBackend::new();

    if let Some(cmd) = cli.command {
        match cmd {
            Commands::Passthrough {
                video,
                audio,
                output,
                resolution,
                codec,
            } => {
                backend
                    .passthrough_remux(&video, &audio, &output, resolution, codec)
                    .context("passthrough remux failed")?;
                println!("Passthrough complete -> {}", output.display());
                return Ok(());
            }
            Commands::ListTransitions => {
                println!("Registered CPU transitions:");
                for (id, desc) in list_registered_transitions() {
                    println!("  {id:<14} - {desc}");
                }
                return Ok(());
            }
            Commands::Models { manifest } => {
                let m = ModelManifest::load(&manifest)
                    .with_context(|| format!("failed loading {}", manifest.display()))?;
                for (key, entry) in &m.models {
                    println!(
                        "{key:<14} | {:<24} | {:<38} | license: {:<10} | {}",
                        entry.filename, entry.name, entry.license, entry.description
                    );
                }
                return Ok(());
            }
        }
    }

    // If `--from-timeline` is provided, skip analysis and render straight from the JSON timeline
    if let Some(ref tl_path) = cli.from_timeline {
        let timeline = Timeline::load_json(tl_path)
            .with_context(|| format!("failed loading timeline from {}", tl_path.display()))?;

        if !cli.dry_run {
            let render_cfg = RenderConfig {
                video_path: cli
                    .video
                    .clone()
                    .unwrap_or_else(|| PathBuf::from(&timeline.source_video)),
                audio_path: cli.audio.clone().or_else(|| {
                    if timeline.source_audio.is_empty() {
                        None
                    } else {
                        Some(PathBuf::from(&timeline.source_audio))
                    }
                }),
                output_path: cli.output.clone(),
                resolution: cli.resolution,
                fps: cli.fps,
                codec: cli.codec,
                crf: 18,
                preset: "fast".to_string(),
                use_gpu: cli.gpu,
                shaders_dir: Some(cli.shaders_dir.clone()),
                cache_capacity: 16,
                progress_cb: None,
            };
            render_timeline(&backend, &timeline, &render_cfg)
                .context("failed rendering from timeline")?;
            println!("Rendered timeline -> {}", cli.output.display());
        }
        return Ok(());
    }

    let video_path = cli
        .video
        .as_ref()
        .context("missing --video <PATH> (or use --from-timeline <TIMELINE.json>)")?;
    let audio_path = cli
        .audio
        .as_ref()
        .context("missing --audio <PATH> (or use --from-timeline <TIMELINE.json>)")?;

    let cache = CacheManager::new(".ae-cache", !cli.no_cache);
    let v_probe = backend.probe_video(video_path)?;
    let target_res = cli.resolution.unwrap_or_else(|| {
        let (w, h) = v_probe.oriented_dimensions();
        Resolution::new(w, h)
    });
    let target_fps = cli.fps.unwrap_or(v_probe.fps);

    let pb = ProgressBar::new(4);
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {bar:32.cyan/blue} {pos}/{len} {msg}")
            .unwrap(),
    );

    // Step 1: Shot detection (M1)
    pb.set_message("Detecting shots in video...");
    let shot_cfg = ShotDetectorConfig {
        mode: cli.shots,
        model_path: Some(cli.models_dir.join("transnetv2.onnx")),
        ..Default::default()
    };
    let shot_cache_key = cache.file_cache_key(
        video_path,
        "shots_v1",
        &format!("{:?}_{:?}", shot_cfg.mode, shot_cfg.min_shot_duration),
    )?;
    let shots: Vec<Shot> = if let Some(cached) = cache.get(&shot_cache_key) {
        cached
    } else {
        let detected = detect_shots(&backend, video_path, &shot_cfg)?;
        cache.put(&shot_cache_key, &detected)?;
        detected
    };

    if let Some(ref dump_path) = cli.dump_shots {
        let json = serde_json::to_string_pretty(&shots)?;
        std::fs::write(dump_path, json)?;
    }
    pb.inc(1);

    // Step 2: Bass-instrument beat detection (M2)
    pb.set_message("Isolating bass instrument and detecting beats...");
    let beat_cfg = BeatDetectorConfig {
        separation: cli.separation,
        bass_band: cli.bass_band,
        beat_mode: cli.beat_mode,
        min_cut: cli.min_cut,
        demucs_model_path: Some(cli.models_dir.join("htdemucs.onnx")),
        debug_dir: cli.debug_beats.clone(),
        ..Default::default()
    };
    let beat_cache_key = cache.file_cache_key(
        audio_path,
        "beats_v1",
        &format!(
            "{:?}_{}_{:?}_{:?}",
            beat_cfg.separation, beat_cfg.bass_band, beat_cfg.beat_mode, beat_cfg.min_cut
        ),
    )?;
    let beats: Vec<Beat> = if cli.debug_beats.is_none()
        && let Some(cached) = cache.get(&beat_cache_key)
    {
        cached
    } else {
        let detected = detect_bass_beats(audio_path, &beat_cfg)?;
        cache.put(&beat_cache_key, &detected)?;
        detected
    };
    pb.inc(1);

    // Step 3: Visual descriptors & match-cut sequencing (M3)
    pb.set_message("Extracting visual descriptors and sequencing match cuts...");
    let sim_cfg = SimilarityConfig {
        mode: cli.similarity,
        model_path: Some(cli.models_dir.join("dinov2_vits14.onnx")),
        ..Default::default()
    };
    let sim_cache_key = cache.file_cache_key(
        video_path,
        "descriptors_v1",
        &format!("{:?}_{}", sim_cfg.mode, shots.len()),
    )?;
    let bank: ShotDescriptorBank = if let Some(cached) = cache.get(&sim_cache_key) {
        cached
    } else {
        let extracted = extract_shot_descriptors(&backend, video_path, &shots, &sim_cfg)?;
        cache.put(&sim_cache_key, &extracted)?;
        extracted
    };

    let policy = if let Some(ref style_path) = cli.transitions {
        StyleTomlConfig::load(style_path)?
    } else {
        StyleTomlConfig::default()
    };

    let seq_cfg = SequencerConfig {
        order: cli.order,
        allow_speed: cli.allow_speed,
        seed: cli.seed,
        resolution: target_res,
        fps: target_fps,
        source_video: video_path.to_string_lossy().to_string(),
        source_audio: audio_path.to_string_lossy().to_string(),
        ..Default::default()
    };

    let timeline = sequence_timeline(&shots, &beats, &bank, &policy, &seq_cfg)?;
    if timeline.slots.is_empty() {
        bail!("generated timeline has 0 slots");
    }

    if let Some(ref dump_tl) = cli.dump_timeline {
        timeline.save_json(dump_tl)?;
    }
    pb.inc(1);

    // Step 4: Render final stretched video with modular transitions (M4 + M5)
    if cli.dry_run {
        pb.finish_with_message(format!(
            "Dry-run complete ({} shots, {} bass beats, {} slots, mean match similarity {:.3})",
            shots.len(),
            beats.len(),
            timeline.slots.len(),
            timeline.mean_boundary_similarity
        ));
        return Ok(());
    }

    pb.set_message(format!("Rendering {}...", target_res));
    let render_cfg = RenderConfig {
        video_path: video_path.clone(),
        audio_path: Some(audio_path.clone()),
        output_path: cli.output.clone(),
        resolution: Some(target_res),
        fps: Some(target_fps),
        codec: cli.codec,
        crf: 18,
        preset: "fast".to_string(),
        use_gpu: cli.gpu,
        shaders_dir: Some(cli.shaders_dir.clone()),
        cache_capacity: 16,
        progress_cb: None,
    };

    render_timeline(&backend, &timeline, &render_cfg)?;
    pb.inc(1);
    pb.finish_with_message(format!(
        "Done! Rendered {} ({} slots, mean similarity {:.3}) -> {}",
        target_res,
        timeline.slots.len(),
        timeline.mean_boundary_similarity,
        cli.output.display()
    ));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ae_beats::generate_synthetic_bass_and_kick_wav;
    use ae_media::generate_synthetic_multi_shot_video;

    #[test]
    fn test_end_to_end_auto_editor_pipeline() {
        let tmp = tempfile::tempdir().unwrap();
        let video_path = tmp.path().join("input_clip.mp4");
        let audio_path = tmp.path().join("input_song.wav");
        let shots_json = tmp.path().join("shots.json");
        let timeline_json = tmp.path().join("timeline.json");
        let out_video = tmp.path().join("final_edited.mp4");
        let out_from_tl = tmp.path().join("from_timeline.mp4");

        let fps = Rational::from_secs(30);
        // Create a 4-shot video (each 1.0s = 30 frames) with visually matching pairs
        generate_synthetic_multi_shot_video(
            &video_path,
            Resolution::new(320, 180),
            fps,
            &[
                ([230, 35, 35], [40, 10, 10], 30),
                ([35, 220, 45], [10, 40, 10], 30),
                ([225, 40, 40], [45, 15, 15], 30),
                ([40, 215, 50], [15, 45, 15], 30),
            ],
        )
        .unwrap();

        // Create a 2.5s song with bass notes at 0.5s, 1.0s, 1.5s, 2.0s and a kick at 1.25s
        generate_synthetic_bass_and_kick_wav(
            &audio_path,
            2.5,
            &[(0.5, 65.41), (1.0, 82.41), (1.5, 73.42), (2.0, 98.0)],
            &[1.25],
        )
        .unwrap();

        let cli = Cli {
            command: None,
            video: Some(video_path.clone()),
            audio: Some(audio_path.clone()),
            output: out_video.clone(),
            resolution: Some(Resolution::new(240, 400)), // Stretched to fill custom vertical resolution
            shots: ShotDetectorMode::Classic,
            separation: SeparationMode::Hpss,
            bass_band: FreqBand::default(),
            beat_mode: BeatMode::Hybrid,
            min_cut: Rational::from_millis(250),
            similarity: SimilarityMode::Fast,
            order: SequenceOrder::Similarity,
            allow_speed: SpeedRange { min: 0.5, max: 1.5 },
            transitions: None,
            shaders_dir: PathBuf::from("../../transitions"),
            gpu: false,
            fps: Some(fps),
            codec: VideoCodec::X264,
            seed: 42,
            dump_shots: Some(shots_json.clone()),
            debug_beats: None,
            dump_timeline: Some(timeline_json.clone()),
            from_timeline: None,
            models_dir: PathBuf::from("../../models"),
            dry_run: false,
            no_cache: true,
        };

        run_cli(cli).unwrap();

        assert!(shots_json.exists());
        assert!(timeline_json.exists());
        assert!(out_video.exists());

        let backend = FfmpegBackend::new();
        let probe = backend.probe_video(&out_video).unwrap();
        assert_eq!(probe.width, 240);
        assert_eq!(probe.height, 400);
        assert_eq!(probe.sar_num, 1);
        assert_eq!(probe.sar_den, 1);

        let a_probe = backend.probe_audio(&out_video).unwrap();
        assert!(
            (probe.duration.as_secs_f64() - a_probe.duration.as_secs_f64()).abs() < 0.10,
            "video ({:.3}s) and audio ({:.3}s) must stay in sync",
            probe.duration.as_secs_f64(),
            a_probe.duration.as_secs_f64()
        );

        // Also verify rendering directly from the saved timeline (--from-timeline)
        let cli_from_tl = Cli {
            command: None,
            video: None,
            audio: None,
            output: out_from_tl.clone(),
            resolution: Some(Resolution::new(240, 400)),
            shots: ShotDetectorMode::Classic,
            separation: SeparationMode::Hpss,
            bass_band: FreqBand::default(),
            beat_mode: BeatMode::Hybrid,
            min_cut: Rational::from_millis(250),
            similarity: SimilarityMode::Fast,
            order: SequenceOrder::Similarity,
            allow_speed: SpeedRange { min: 0.5, max: 1.5 },
            transitions: None,
            shaders_dir: PathBuf::from("../../transitions"),
            gpu: false,
            fps: Some(fps),
            codec: VideoCodec::X264,
            seed: 42,
            dump_shots: None,
            debug_beats: None,
            dump_timeline: None,
            from_timeline: Some(timeline_json),
            models_dir: PathBuf::from("../../models"),
            dry_run: false,
            no_cache: true,
        };
        run_cli(cli_from_tl).unwrap();
        assert!(out_from_tl.exists());
    }
}
