# Native-aspect, sensitive edit validation

Historical validation for version 0.2.0. This 42-shot edit is preserved under
`examples/before-quiet-gate/`. The earlier portrait edit remains at
`examples/real.mp4`. The current `edit.mp4` holds the quiet introduction; see
[QUIET_SECTIONS_VALIDATION.md](QUIET_SECTIONS_VALIDATION.md).

## Command used

```sh
uv sync --locked
.venv/bin/autoedit create \
  --audio tiktok_niko.cc116_7695074868118785294_audio.mp3 \
  --clips ./clips --output edit.mp4 \
  --beat-sensitivity 2 --pacing fast --reuse forbid --overwrite
uv tool install --force --constraints requirements.lock .
autoedit validate --plan edit.plan.json --video edit.mp4
uv run --locked pytest -q
```

## Actual results

- All 136 input clips are 1920×1080 (16:9); output follows that display aspect and native resolution without the previous portrait canvas/padding.
- 42 shots from 42 distinct source clips, compared with 14 shots in the earlier edit. Selected source SHA-256 identities are also all distinct; no clip is reused.
- Sensitivity 2 detected 41 beats and 112 onsets. Effective audio hop is 362 samples; onset peak delta is 0.035; beat tracker tightness is 70.711. All effective controls are stored in the plan and included in cache keys.
- Full decode passed: 625 video frames, duration 20.833333333s, 1920×1080 at 30 fps, H.264/yuv420p and AAC soundtrack.
- Maximum internal-cut error relative to selected detected beats: 16.599 ms (one frame is 33.333 ms).
- Soundtrack correlation: 0.98816989; measured lag: 0.000000s.
- All rendered boundary frames passed comparison with the selected source regions. Maximum normalized pixel MAE: 0.002153.

Actual automated-test result:

```text
.............................................                            [100%]
45 passed in 39.58s
```

The additional regression coverage includes real rendered native-size inference, unique-source enforcement, forbidden insufficient footage, an explicit stretched portrait output from a landscape source, matching descriptors on the stretched canvas, resolution/config precedence, invalid and conflicting resolution arguments, and high-sensitivity detection of weaker transients. Silence still produces an explicitly recorded fallback with no invented beats. Legacy saved plans retain their recorded dimensions, fit and reuse policy.

## Updated CLI behavior

Output dimensions default to the most common source display aspect ratio and most frequent native resolution within that aspect group. Rotation metadata is respected; ties favor larger native dimensions. Each clip may appear at most once by default. Explicit `--reuse allow` remains available for cases where repetition is wanted.

```sh
# Native aspect; more responsive beat detection and faster cuts.
autoedit create --audio music.mp3 --clips ./clips --output edit.mp4 \
  --beat-sensitivity 2 --pacing fast

# Stretch footage to completely fill an explicit output resolution.
autoedit create --audio music.mp3 --clips ./clips --output portrait.mp4 \
  --beat-sensitivity 2 --pacing fast --resolution 1080x1920
```

`--resolution` always uses stretch, including when a YAML config or `--fit` says otherwise. It is mutually exclusive with the legacy `--size` canvas option. Analysis, refinement, render and validation use the same fit. Higher sensitivity also shortens default pacing targets by its square root; explicit shot-duration overrides are retained.

## Artifacts and limits

[Edit](examples/before-quiet-gate/edit.mp4) · [Plan](examples/before-quiet-gate/edit.plan.json) · [HTML inspection report](examples/before-quiet-gate/edit.report.html) · [Validation JSON](examples/before-quiet-gate/edit.validation.json) · [Representative transition pairs](examples/native-sensitive.contact.png) · [Machine-readable summary](examples/native-sensitive-summary.json)

Planning retains hard cuts on the detected beat grid and a bounded visual/motion search. It does not infer semantic story order or musical downbeats. Audio timing was checked by waveform comparison; subjective listening remains unavailable in this runtime. Any relaxed duration preferences are recorded below and in the plan.

Warnings:
- The soundtrack ended between output frames; duration was floored to avoid extending beyond available music.
- Detected event spacing could not satisfy configured shot limits; limits were relaxed to preserve the selected rhythm and duration.
- transition 37 has a weak measured match (0.344); no perceptual quality guarantee
