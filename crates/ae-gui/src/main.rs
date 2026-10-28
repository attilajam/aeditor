use std::path::PathBuf;

use ae_gui::{AutoEditorEguiApp, ProjectController};
use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "auto-editor-gui",
    version,
    about = "Pure-Rust egui studio interface for auto-editor-rs"
)]
struct GuiArgs {
    /// Optional initial video clip to load on startup
    #[arg(short = 'v', long = "video")]
    video: Option<PathBuf>,

    /// Optional initial audio track to load on startup
    #[arg(short = 'a', long = "audio")]
    audio: Option<PathBuf>,

    /// Target output resolution (e.g. 1080x1920, vertical, 1080p)
    #[arg(short = 'r', long = "resolution", default_value = "vertical")]
    resolution: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Reserve 2 CPU cores for the egui UI thread and display compositor so background
    // video/audio analysis never starves UI frame rendering.
    let avail_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(avail_cores.saturating_sub(2).max(1))
        .build_global();

    let args = GuiArgs::parse();
    let mut ctrl = ProjectController::new();
    let _ = ctrl.set_resolution_str(&args.resolution);

    if let Some(ref v) = args.video {
        let _ = ctrl.import_video(v);
    }
    if let Some(ref a) = args.audio {
        let _ = ctrl.import_audio(a);
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("auto-editor-rs — egui Beat-Synced Match-Cut Studio")
            .with_inner_size([1920.0, 1140.0])
            .with_min_inner_size([1200.0, 760.0]),
        ..Default::default()
    };

    eframe::run_native(
        "auto-editor-rs",
        native_options,
        Box::new(|_cc| Ok(Box::new(AutoEditorEguiApp::new(ctrl)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;

    Ok(())
}
