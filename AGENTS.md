# Working in aeditor

Rust 2024 workspace for a beat-synced video editor. Prefer the smallest complete
change. Reuse existing types and helpers; add no dependency for a few lines.

## Find the right code

Paths below are under `crates/`. Most library tests live beside the code in
`#[cfg(test)]` modules; CLI tests are in `ae-cli/src/main.rs`.

| Crate | Owns / start here |
| --- | --- |
| `ae-core` | Shared types (`src/types.rs`), rational time, frames, errors, cache, model manifest |
| `ae-media` | FFmpeg backend, probe/decode/encode (`src/lib.rs`), ingest helpers (`src/ingest.rs`) |
| `ae-beats` | Audio decode, bass/beat detection, `BeatAnalysisSession` (`src/lib.rs`) |
| `ae-shots` | Shot detection (`src/lib.rs`), streaming (`src/incremental.rs`), boundaries (`src/refine.rs`) |
| `ae-similarity` | `FrameDescriptor`, position/motion scoring (`src/lib.rs`) |
| `ae-sequencer` | Timeline sequencing, alternative-shot ranking (`src/lib.rs`) |
| `ae-transitions` | CPU transitions, registry, easing, style policy (`src/lib.rs`) |
| `ae-transitions-gpu` | GPU engine (`src/lib.rs`); shaders in root `transitions/*.wgsl` |
| `ae-render` | Timeline/draft rendering (`src/lib.rs`), enhancement (`src/enhance.rs`), crop/face tracking (`src/reframe.rs`) |
| `ae-cli` | Arguments and pipeline (`src/main.rs`); binary `auto-editor` |
| `ae-gui` | egui UI (`src/app.rs`), orchestration (`src/project_controller.rs`), preview cache (`src/image_provider.rs`), beat/shot/timeline models |

Shared library APIs are re-exported in each `src/lib.rs`. CLI and GUI assemble
the pipeline; GUI behavior often spans `app.rs`, the controller, and a model.
Model metadata: `models/manifest.toml`; transition style fixture:
`tests/fixtures/style.toml`. Read crate manifests when checking dependencies.

## Keep exploration and output small

- Start with `git status --short`; preserve existing edits. Read the touched code
  and find callers with `rg -n 'symbol' crates/` before changing shared behavior.
- Use this map, scoped `rg`, and bounded file reads. Avoid dumping the repo,
  lockfile, generated files, or entire large modules into context.
- Inspect before compiling. Batch independent reads; run Cargo commands
  sequentially so they do not contend for the build lock.
- Reuse `target/` and `.ae-cache/`. Do not clean caches, change toolchains,
  update dependencies, or vary build flags as routine troubleshooting.

## Validate the change, once

Run from the repo root. The aliases use the unoptimized `agent` profile and
`--locked`; they suppress build chatter, preserve diagnostics and exit status,
and do not suppress tests. Always select the affected package(s) with `-p`.

```sh
# Behavior change: run a relevant regression, then other affected tests as needed.
cargo test-fast -p ae-core rational_time_and_frames_roundtrip
cargo test-fast -p ae-core

# Compile validation when tests do not cover the edited targets.
cargo check-fast -p ae-gui

# Format only edited Rust files; --config skip_children avoids sibling modules.
rustfmt --edition 2024 --config skip_children=true --check path/to/edited.rs
```

- For behavior changes, add or update a small regression for the bug or edge
  case. Check that a test filter actually ran tests; zero tests is not a pass.
- A test run compiles its targets: do not precede it with a redundant build/check.
  Re-run after further changes, failures, or a specific unresolved concern.
- For shared API/schema changes, check affected consumers, not just the defining
  crate. Add multiple `-p` flags; use `cargo check-fast --workspace` for changes
  with workspace-wide impact. Inspect callers even for compatible API changes.
- Docs/comments-only edits need no Cargo. For build/config edits, validate the
  changed configuration and a representative command. Formatting is not a
  substitute for compilation or behavioral tests on Rust changes.
- The first `agent` build has a separate cache in `target/agent/`. Stay on this
  profile for logic checks; use normal `cargo run -p ae-gui` or
  `cargo run -p ae-cli -- --help` for optimized dev execution. Use the normal dev
  or release profile for timing-sensitive tests and performance measurements;
  unoptimized media/DSP loops can be slower at runtime.
- Media/integration tests may require `ffmpeg` and `ffprobe` on `PATH`; ML builds
  can download ONNX Runtime. Some model-download tests are explicitly ignored.
  Run the relevant real-model/GPU/media path when changing it; a fallback-only
  pass does not verify hardware or inference. Do not globally enable ignored tests
  or download models for unrelated changes.
- Stop after sufficient scoped validation. Report changes, checks, and untested
  behavior briefly; never claim a check passed if it was skipped or blocked.
