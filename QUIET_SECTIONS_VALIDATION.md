# Quiet-section edit validation

Version 0.2.1, tested on CPU on 2026-10-18. Beat sensitivity remains 2. A separate
relative-energy gate prevents the quiet introduction from generating cuts.

## Commands executed

From the repository root:

```sh
uv run --locked autoedit create \
  --audio tiktok_niko.cc116_7695074868118785294_audio.mp3 \
  --clips ./clips --output edit.mp4 \
  --beat-sensitivity 2 --pacing fast --reuse forbid --overwrite \
  > examples/quiet-gate-create.log 2>&1
uv run --locked pytest -q > examples/quiet-gate-tests.log 2>&1
uv tool install --force --constraints requirements.lock . \
  > examples/quiet-gate-install.log 2>&1
autoedit --version
autoedit validate --plan edit.plan.json --video edit.mp4 \
  > examples/quiet-gate-validate.log 2>&1
uv run --locked python examples/quiet-gate-check.py \
  > examples/quiet-gate-check.log 2>&1
uv pip check
uv build
```

The global command reports `autoedit 0.2.1`. The project retains the
`scene-splitter` distribution name and its existing `scene-split` command.

## Actual supplied-media results

- The quiet interval is 0–7.749 seconds. The first edit cut is frame 233,
  **7.767 seconds**, on the first eligible detected beat.
- Fourteen former intro cut opportunities are suppressed. There are no forced
  quiet-section cuts or unrecorded quiet-section transitions.
- The 41 detected beats and 112 detected onsets are unchanged from the previous
  sensitivity-2 analysis. Every cut after the drop occurs at exactly the same
  frame as in the previous 42-shot edit.
- The result has 28 shots from 28 different clips. Selected source content
  fingerprints are also all distinct. Native 1920×1080 (16:9) output is retained.
- Full decode passes: 625 frames, 30 fps, 20.833333333 seconds,
  H.264/yuv420p with one AAC soundtrack stream.
- Maximum cut quantization error relative to the selected detected beat:
  **15.918 ms**, below one 33.333 ms output frame.
- Soundtrack comparison: correlation **0.98816989**, measured lag **0 seconds**.
- All rendered entry/exit source-frame comparisons pass. Worst normalized RGB
  pixel MAE is **0.00214745**.

The opening uses `light_yagami_scene_012.mp4`, source interval
1.133333333–8.9 seconds, at normal speed. The decoded contact sheet was visually
inspected: it maintains one composition throughout the introduction and changes
at frame 233. Maximum adjacent-frame RGB MAE inside the entire rendered opening
is **0.00716616**; there is no large hidden source cut in that interval.

During inspection, a previous opening candidate was found to contain a hard cut
between shots with similar palettes. The updated scene detector combines the
existing composition descriptors with abrupt spatial pixel changes relative to
the clip's typical motion. It detects that source discontinuity at
2.833333333–3 seconds, rejects crossing candidates, and invalidates motion
estimates across the cut. Cached analysis frames are reused when migrating scene
metadata to detector version 2.

## Automated tests

Actual full-suite result:

```text
........................................................                 [100%]
56 passed in 61.58s (0:01:01)
```

The eleven added regressions cover quiet intros, quiet breakdowns, relative
energy scaling, isolated quiet transients, explicit footage-limited relaxation,
gate disablement, real librosa analysis, real CLI rendering with a held intro
and distinct sources, same-palette internal cuts, continuous footage, and cache
migration without video re-decoding. Existing timing, visual/motion search,
render/replay, native aspect, stretched resolution and no-reuse tests still pass.

## Gate controls and limits

The gate uses a trailing 0.4-second power average of approximately 93 ms centered
librosa RMS measurements. It enables cuts at 60% of the interval's 80th-percentile
RMS reference and releases below 75% of that threshold for 0.3 seconds.
This is a relative-energy editing heuristic, not musical drop classification.
Quiet but steady music retains cuts because the threshold scales to the track.

```sh
# More protection for quieter sections; detection sensitivity stays at 2.
autoedit create --audio music.mp3 --clips ./clips --output edit.mp4 \
  --beat-sensitivity 2 --pacing fast --quiet-threshold 0.7

# Disable energy gating and restore the previous cut eligibility.
autoedit create --audio music.mp3 --clips ./clips --output edit.mp4 \
  --beat-sensitivity 2 --pacing fast --quiet-threshold 0
```

`quiet_window` and `quiet_release` are also configurable in YAML. Quiet holds can
exceed pacing duration targets. When continuous source regions cannot fill a
hold within allowed speeds, necessary extra cuts are explicitly recorded as
relaxations. This example needs no such extra cuts. Scene detection is sampled
and heuristic; very brief or subtle source cuts can still be missed. Audio
timing was verified by waveform comparison; subjective listening was unavailable.

Only two plan warnings remain: the fractional soundtrack tail is floored to the
frame grid, and the pacing maximum is relaxed for the protected quiet hold.

## Artifacts

[Current edit](edit.mp4) · [Reproducible plan](edit.plan.json) ·
[HTML transition report](edit.report.html) · [Validation JSON](edit.validation.json) ·
[Opening contact sheet](examples/quiet-gate-opening.png) ·
[Measurements](examples/quiet-gate-summary.json) ·
[Previous edit](examples/before-quiet-gate/edit.mp4)
