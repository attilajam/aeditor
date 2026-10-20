# Autoedit and scene-splitter

This project includes two local video tools. `autoedit` builds a complete
music-driven edit from source footage. `scene-split` remains available for
splitting one video at detected scene cuts.

## Install

Install FFmpeg with `ffmpeg`, `ffprobe`, H.264 encoding (`libx264`) and AAC
encoding available on `PATH`. Then sync the locked Python dependencies and
install the command as a `uv` tool:

```sh
uv sync --locked
uv tool install --force --constraints requirements.lock .
```

Python 3.13 or newer is required. For development, `uv run autoedit ...` uses
the exact tested dependencies in `uv.lock`. A hashed `requirements.lock` is
also supplied for pip: `python -m pip install -r requirements.lock`, then
`python -m pip install --no-deps .` in a virtual environment.

The analysis and planning run locally with Python, NumPy, librosa, OpenCV and
FFmpeg. No model download, hosted service or API key is used.

`--force` refreshes an earlier installation of this same project; the existing
`scene-split` command is preserved alongside `autoedit`.

## Create an edit

The default command analyzes up to 30 seconds of music, finds beat-aligned cut
times, chooses source intervals with visual and motion continuity, and renders
an H.264 MP4 with AAC audio:

```sh
autoedit create --audio music.mp3 --clips ./clips --output edit.mp4
```

For the files in this workspace, the same command is:

```sh
autoedit create \
  --audio /home/attila/Documents/edit/tiktok_niko.cc116_7695074868118785294_audio.mp3 \
  --clips /home/attila/Documents/edit/clips \
  --output /home/attila/Documents/edit/edit.mp4
```

The default output follows the input clips' display aspect ratio and native
resolution at 30 fps, with medium pacing, playback speeds from 0.85× to 1.15×,
and each source clip used at most once. With mixed sources, the most common
display aspect ratio wins, then the most common resolution within that group;
ties favor larger source dimensions. Rotation metadata is included in this
decision. Odd dimensions round down to even values for H.264/yuv420p. A speed
above 1 plays the selected source interval faster; the consumed source duration
is output duration multiplied by speed. The renderer uses source presentation
timestamps, including variable frame rate (VFR) timing, to select the latest
available source frame at each output frame time. It creates a constant frame
rate output. Mixed aspect ratios use `contain` fitting unless requested otherwise.
Slow motion can repeat native frames through normal frame-rate conversion. The
renderer does not add freeze frames or black filler when a source interval
cannot supply its planned frames.

Increase response to weaker onsets and use more frequent beat cuts:

```sh
autoedit create --audio music.mp3 --clips ./clips --output edit.mp4 \
  --beat-sensitivity 2 --pacing fast
```

`--beat-sensitivity` ranges from 0.5 to 3 (default 1). Higher settings use a
finer audio hop, a lower onset peak threshold and a denser, more flexible beat
grid. Beats are tracked with madmom's RNN + DBN (threshold and transition
lambda scale with sensitivity); librosa remains the fallback if madmom cannot
be loaded, and still supplies the dense onset list.
They also divide the default pacing duration targets by the square root of
sensitivity. Explicit `min_shot`/`max_shot` settings remain absolute preferences.
Detected beats still determine normal-mode internal cut positions; the tool
does not invent additional beats by subdividing the grid.

Quiet sections now hold their shot even at high beat sensitivity. A separate
energy gate smooths RMS over the preceding 0.4 seconds and compares it with a
robust foreground level (the 80th percentile over the soundtrack interval).
Cuts are enabled above 60% of that level; hysteresis and a 0.3-second release
prevent gaps between drum hits from repeatedly opening and closing the gate.
Trailing smoothing avoids spreading a later drop across a long centered
window. The underlying RMS uses a short, approximately 93 ms centered window.
Detected beats are preserved, and active sections keep the selected sensitivity
and pacing.

Tune the gate with `--quiet-threshold 0.6`, or disable it with
`--quiet-threshold 0` to recover the previous cut schedule. Quiet holds may
exceed pacing targets. If available continuous footage cannot fill a hold,
the planner adds only the necessary extra cuts and records that relaxation;
it never hides infeasibility with frozen frames or repeated clips. Protected
intervals, suppressed opportunities and any forced cuts appear in the plan,
HTML report and validation JSON.

Source-cut detection also checks abrupt spatial pixel changes relative to the
clip's usual motion. This catches cuts between shots with similar palettes;
the planner selects continuous source regions for quiet holds. Existing frame
caches are migrated to the updated scene detector without decoding them again.

Pass an explicit resolution to stretch every shot to fill the entire output:

```sh
autoedit create --audio music.mp3 --clips ./clips --output edit.mp4 \
  --resolution 1080x1920
```

`--resolution WIDTHxHEIGHT` changes the image proportions when the requested
aspect ratio differs from a source. It overrides `--fit` and YAML canvas/fit
settings with `stretch`. The legacy `--size WIDTHxHEIGHT` option still sets a
canvas and honors `--fit contain|cover|stretch`; it cannot be combined with
`--resolution`. Both sizing modes use the same composition during analysis,
boundary refinement, rendering and validation.

The regenerated supplied-footage edit uses `--beat-sensitivity 2 --pacing fast`
at the sources' 1920×1080 resolution, with no clip reuse. Its opening holds until
the first active-section beat at 7.77 seconds. See
[QUIET_SECTIONS_VALIDATION.md](QUIET_SECTIONS_VALIDATION.md) for its exact
command, output validation and regression test results. The earlier sensitivity
comparison remains in [NATIVE_SENSITIVE_VALIDATION.md](NATIVE_SENSITIVE_VALIDATION.md).

### Interactive beat review

Open the local web editor for an existing edit:

```sh
autoedit review --plan edit.plan.json
```

The command opens your browser at `http://127.0.0.1:8765`. Keep the command
running while you edit; Ctrl+C stops the server. It uses local files only, with
no account, API key, Node installation or runtime internet connection. The
bundled wavesurfer.js waveform library is licensed under BSD-3-Clause; its
version and verified package hash are in `src/autoedit/web/vendor/provenance.json`.

The existing cut choices are selected, all other detected beats remain visible,
and quiet sections are shaded. Click a waveform marker to keep or skip it, drag
to adjust timing, or select it in the cut list and nudge by exact output frames.
Shift-click selects multiple markers for group adjustments. Space plays/pauses;
the loop control auditions the selected cut. Onset snapping is optional. Add a
cut at the playhead when the detector missed an event. Undo/redo and reset are
available. No-cut ranges protect a chosen span; their start is inclusive and
their end exclusive.

Changes save automatically to `edit.review.json`, separately from the original
plan and video. Opening the same plan again resumes the draft. Concurrent edits
from different tabs are detected instead of silently overwriting each other.
The waveform uses the decoded PCM soundtrack interval, including `audio_start`,
so MP3 seeking does not shift its peaks. Preview video and audio share one media
clock; a PCM audio player supplies audition when no video has been rendered yet.

Choose **Update preview** to rerun visual/motion source selection and render a
preview with a maximum side of 640 pixels. **Export video** renders at the
original planned output resolution, alongside a replayable plan, decoded-media
validation and an HTML transition report. The preview is labelled out of date
until it matches the current saved choices. Editing and audio audition remain
available while a background job runs.

Your frame boundaries and no-cut ranges are fixed during replanning. Speed
limits and the original reuse setting remain in effect (`forbid` for current
default edits). If a held section cannot fit a continuous source region, the
job explains which slot is infeasible; it does not insert extra cuts, freezes
or source-cut crossings. Manually moved cuts are recorded distinctly from the
original detected beats and validated against the saved user timing.

```sh
# Choose another port or a separately named draft.
autoedit review --plan edit.plan.json --port 8766 --session alternate.review.json

# Show the URL without automatically opening a browser.
autoedit review --plan edit.plan.json --no-browser --port 0

# Use an existing preview video stored at another path.
autoedit review --plan edit.plan.json --video /path/to/existing-edit.mp4
```

Temporary previews, waveform data and cached selections live under
`.autoedit-review/` next to the draft. Deleting that directory removes previews;
the JSON draft retains your timing choices. Source and plan fingerprints are
checked on load. If you regenerate the base plan, start a new draft with
`--session new.review.json`.
See [WEB_REVIEW_VALIDATION.md](WEB_REVIEW_VALIDATION.md) for tested commands,
browser checks and an actual export made with manually reviewed beat timing.

Set the music interval explicitly when you want a longer edit or a later part
of the soundtrack:

```sh
autoedit create --audio music.mp3 --clips ./clips --output edit.mp4 \
  --audio-start 12 --duration 45 --pacing fast --reuse forbid --seed 7
```

`--audio-start` and `--duration` are seconds. If duration is omitted, analysis
uses up to the first 30 seconds after `audio-start`; a shorter file is analyzed
to its end. The selected soundtrack interval is trimmed to the output duration.

## Commands and options

`autoedit create` accepts `--audio`, `--clips`, and `--output` (all required),
plus:

| Option | Default | Meaning |
| --- | --- | --- |
| `--config PATH` | none | Read settings from a YAML file. |
| `--duration SECONDS` | up to 30 seconds | Requested soundtrack duration. |
| `--audio-start SECONDS` | `0` | Start time in the source soundtrack. |
| `--pacing slow\|medium\|fast` | `medium` | Preferred shot-length range: 2–4, 1–2, or 0.5–1 seconds. |
| `--beat-sensitivity N` | `1` | 0.5–3; higher values retain weaker onsets and favor more frequent cuts. |
| `--quiet-threshold N` | `0.6` | Hold shots below this fraction of foreground RMS; 0 disables the gate. |
| `--resolution WIDTHxHEIGHT` | input dimensions | Stretch footage to exactly fill this resolution; even dimensions >=16. |
| `--size WIDTHxHEIGHT` | `auto` | Legacy canvas dimensions honoring `--fit`; automatic mode follows the input clips. |
| `--fps N` | `30` | Output frame rate, from 1 to 120 fps. |
| `--fit contain\|cover\|stretch` | `contain` | Pad, centrally crop, or stretch; `--resolution` always stretches. |
| `--speed-min N` | `0.85` | Minimum source playback speed. |
| `--speed-max N` | `1.15` | Maximum source playback speed. |
| `--reuse allow\|forbid` | `forbid` | Use each source clip at most once; reuse requires explicit `allow`. |
| `--seed N` | `0` | Seed candidate ordering and the bounded search. |
| `--cache-dir PATH` | `.autoedit-cache` | Store reusable analysis keyed by source content and settings. |
| `--overwrite` | off | Replace the output and its generated sidecar files. |

CLI options override values from `--config`; values omitted by both use the
defaults above. See [config.example.yaml](config.example.yaml) for every
supported setting, including analysis and search controls.

Other commands:

```sh
# Analyze inputs without rendering; --output names the analysis JSON.
autoedit analyze --audio music.mp3 --clips ./clips --output analysis.json

# Render a saved plan again after verifying its recorded source identities.
autoedit render --plan edit.plan.json --output replay.mp4

# Validate a rendered video against its saved plan.
autoedit validate --plan edit.plan.json --video edit.mp4

# Print the installed version.
autoedit --version
```

`analyze` accepts the same input and tuning options as `create`, but does not
write video. `render` needs only a plan and destination; it verifies the audio
and source files against their recorded content fingerprints before use.
`validate` decodes the rendered video and checks frame count, timing, selected
beat alignment when beats were available, and audio alignment.

## Planning, outputs and limits

The planner first builds feasible candidate intervals and searches complete
edits with a bounded beam. It then refines selected trims against native source
frames and searches again. Audio and video analysis results are cached using
content fingerprints and relevant settings. The fixed seed makes planning
choices reproducible for the same inputs and configuration.
Plans store absolute source paths; moving the source files requires a new
`create` run.

Beat reliability is a heuristic based on activity, beat support, beat count
and interval regularity; it is not a calibrated probability. Weak rhythms use
usable onsets, while silence or audio without usable rhythmic events uses
regular boundaries. The tool detects beats and onsets, not musical downbeats.
The plan records the chosen method and fallback warnings.

Creating `edit.mp4` also writes:

* `edit.plan.json` — source identities, configuration, chosen boundaries,
  segments, search details and warnings.
* `edit.validation.json` — decoded-frame, timing and audio validation results.
* `edit.report.html` — a visual inspection report with the edit timeline and
  quality measurements.

The validator also writes a `.validation.json` file next to a video passed to
`autoedit validate`. Existing destinations are protected; use `--overwrite` to
replace them. A failed render may leave an `.error.json` diagnostic file.

Visual matching favors the subject's contours, pose, scale and position in the
output frame, even when the palette or setting changes. It uses local features
and does not recognize people or detect skeletons. Scene boundaries are estimated from sampled frames, so cuts
between samples may be missed or placed approximately. Search is bounded by
candidate and beam limits. Hard cuts are used. Static or consistently soft
footage can provide weak motion or sharpness evidence, and the report exposes
measured warnings rather than promising perceptual quality.

The main descriptor pools unsigned gradient orientations into **8×8 and 16×16
screen grids**, with eight interpolated angle bins per cell. Light smoothing
suppresses pixel noise, and clipping unusually dense cells reduces the effect
of small textured patches. Square-root, unit-mass histograms are compared by
Hellinger distance, weighted 35% coarse and 65% fine. A broad Gaussian center
prior favors a central subject while retaining every cell. `edge_center_bias`
in YAML controls its strength (default 0.65; 0 gives equal screen weighting).
The descriptor retains screen coordinates: it does not recenter, rescale or
mirror subjects to manufacture a match. Reversed light/dark contrast preserves
unsigned edge directions. All component distances lie in [0,1].
The pyramid is stored as packed 16-bit bins in JSON, keeping analysis memory
bounded by compact descriptors rather than thousands of Python floats per frame.

Default visual/motion weights are **edge 0.75, color 0.03, structure 0.07,
motion 0.15**. Color uses four quadrant HSV histograms (half their L1 distance);
structure uses 12×12 luminance maps (mean absolute difference). Both are weak
secondary cues. Set `weights` in YAML to tune these terms. Optical flow is
measured within each shot on textured pixels, expressed in image diagonals per
second, and multiplied by playback speed before comparison. Motion distance
combines angular and saturating magnitude differences; unreliable flow blends
toward a neutral 0.5. Configurable weights account for different useful ranges.
Quality uses the visible source area, excluding deliberate contain padding;
black or almost uniform footage receives a strong additional penalty. Blur
is measured relative to each clip's median sharpness.

This is an approximate pose cue: intricate backgrounds, occlusion, similar
contours and very small subjects can still produce incorrect matches. It works
best when a prominent subject occupies much of the frame. It does not segment
the foreground or identify joints. The inspection report shows actual decoded
boundary pairs and records the descriptor version, weights and measured costs.
Older frame caches upgrade automatically without redecoding footage. Replaying
an existing plan preserves its selected timeline; reviewing an older plan runs
the new matcher and upgrades the former automatic weights, recording a warning.
Explicit custom weights are retained.

Scene changes compare content color, structure and edges at the configured
sample rate and reserve the interval around a detected change. Candidate trim
points span usable regions throughout each clip. Balanced pruning limits how
many candidates one clip can occupy. Beam states retain selected intervals
and whole-edit source usage; `forbid` means each source clip can appear once.
Native-frame refinement adjusts source trims while retaining slot length and
speed limits. It never shifts output cuts to improve a match. Intermediates
use lossless H.264; final assembly assigns absolute frame timestamps before
one lossy H.264 encode, avoiding container-duration rounding drift.

## Synthetic fixtures and matching benchmark

Generate deterministic short audio and video fixtures, including known
transient times, silence, weak rhythm, VFR, rotated metadata, portrait and
landscape sources, similar and mismatched compositions, motion, black, short,
and corrupt clips:

```sh
.venv/bin/python scripts/generate_test_media.py --output /tmp/autoedit-fixtures
```

Run the controlled matching benchmark, which compares the planner with a
seeded feasible random baseline and writes plans, rendered videos and metrics:

```sh
.venv/bin/python scripts/benchmark_matching.py --output benchmark-output
```

Compare spatial matching with the previous matcher and a seeded random
selection on generated subject poses, independently measured from rendered
frames:

```sh
uv run --locked python scripts/benchmark_spatial_matching.py --output examples/spatial-matching
```

See [SPATIAL_MATCHING_VALIDATION.md](SPATIAL_MATCHING_VALIDATION.md) for measured
results, the real-footage example and exact validation commands.

The automated tests exercise known synthetic audio and decode generated video
frames with FFmpeg. See [VALIDATION.md](VALIDATION.md) for the recorded test and
benchmark results.

## `scene-split`

Split a video into MP4 clips at automatically detected scene cuts. It uses
[PySceneDetect](https://www.scenedetect.com/) for detection and FFmpeg to write
the clips.

```sh
scene-split "path/to/video.mov"
```

Clips are written to `path/to/video_clips/` by default. Choose another output
directory with `-o`:

```sh
scene-split "path/to/video.mp4" -o ./clips
```

Adjust cut sensitivity, minimum clip length, and encoding quality as needed:

```sh
scene-split video.mp4 --threshold 22 --min-scene-len 1s --crf 18
```

Lower thresholds detect more cuts. The default detector threshold is 27. The
default encoder uses H.264 with CRF 20 and the `veryfast` preset.
