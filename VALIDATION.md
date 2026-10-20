# Validation record

The original validation below describes version 0.1.0 and its portrait defaults.
Version 0.2.0 uses source-sized output, optional stretching, sensitivity control
and no source reuse by default; its validation is recorded in
[NATIVE_SENSITIVE_VALIDATION.md](NATIVE_SENSITIVE_VALIDATION.md). The original
example remains at `examples/real.mp4`; the root `edit.mp4` is the newer edit.
Version 0.2.1 adds quiet-section holds; current output and results are recorded
in [QUIET_SECTIONS_VALIDATION.md](QUIET_SECTIONS_VALIDATION.md).

Tested locally on 2026-10-18 with CPU execution. Luna implemented and tested the synthetic media generator, audio analysis, visual/motion features, planner tests, and end-to-end tests; the primary agent integrated the pipeline and independently ran the complete checks.

## Installation and dependency check

Python 3.13.15; librosa 1.0.0; NumPy 2.5.4; OpenCV distribution 5.0.0.93; SciPy 1.18.1; PyYAML 6.0.3. FFmpeg/ffprobe n9.0.1. The installed librosa 1.0.0 docstrings and signatures for `beat_track` and `onset_strength` were inspected; the implementation uses their keyword interfaces. No models or services are used.

Commands actually executed from the repository root:

```sh
uv sync --locked
uv pip check
uv export --locked --format requirements-txt --all-groups --no-emit-project --output-file requirements.lock
uv build
uv venv /tmp/autoedit-install-check --python 3.13
uv pip sync --python /tmp/autoedit-install-check/bin/python requirements.lock
uv pip install --python /tmp/autoedit-install-check/bin/python --no-deps dist/scene_splitter-0.1.0-py3-none-any.whl
/tmp/autoedit-install-check/bin/autoedit --version
uv tool install --force --constraints requirements.lock .
autoedit --version
```

Dependency check: all installed packages compatible. Wheel and source distribution built successfully. Fresh isolated wheel installation and the refreshed global tool installation report `autoedit 0.1.0`. The package retains its existing `scene-splitter` distribution name and `scene-split` entry point.

```sh
/tmp/autoedit-install-check/bin/autoedit create \
  --audio 'examples/synthetic media/audio fixtures/rhythmic 120 bpm.wav' \
  --clips 'examples/synthetic media/video fixtures' --output examples/installed.mp4 \
  --size 160x240 --fps 29 --duration 2 --audio-start 0.375 --seed 17 \
  --cache-dir /tmp/autoedit-install-cache
```

This installed CLI produced and validated 58 frames at 29 fps, with soundtrack correlation 0.999996 and zero measured lag.

## Automated tests

```sh
uv run --locked pytest -q
```

Actual result:

```text
..................................                                       [100%]
34 passed in 26.84s
```

The tests include known audio transients (75 ms detector tolerance), silence, weak rhythm, interval offsets, frame quantization, energy-dependent pacing, real internal scene changes, fit handling, cache invalidation, spatial composition, extracted flow direction, motion changing planner selection, black-frame penalties, feasible speed/duration/PTS mapping, repetition and overlap costs, beam search beating a greedy trap, and deterministic complete refined planning.

Real subprocess/render tests cover 24/30 fps sources, variable frame timestamps, metadata rotation, portrait/landscape frames, source audio/no audio, playback speeds 0.85/1.15/0.95/1.10, spaces in paths, corrupt footage, one usable clip, short footage, empty input, forbidden reuse, missing/changed sources, excessive audio duration, and overwrite protection. Independent decoded-frame comparisons use native frame neighborhoods and visible frame identifiers; replay must produce identical decoded frames. Short-clip renders verify that assembly has no accumulated container-duration drift.

## Synthetic full pipeline

```sh
.venv/bin/python scripts/generate_test_media.py --output 'examples/synthetic media' --duration 6
.venv/bin/autoedit create \
  --audio 'examples/synthetic media/audio fixtures/rhythmic 120 bpm.wav' \
  --clips 'examples/synthetic media/video fixtures' --output examples/synthetic.mp4 \
  --size 160x240 --fps 24 --duration 5 --audio-start 0.375 --seed 17 --overwrite
.venv/bin/autoedit validate --plan examples/synthetic.plan.json --video examples/synthetic.mp4
```

Result: 120/120 frames fully decoded; H.264/yuv420p, AAC; soundtrack correlation 0.99999489, lag 0.000000s. Maximum selected-beat quantization error 15.113 ms (one output frame is 41.667 ms). All planned source-boundary comparisons passed. Corrupt input was skipped with its diagnostic reason.

Artifacts: [video](examples/synthetic.mp4), [plan](examples/synthetic.plan.json), [HTML report](examples/synthetic.report.html), [validation](examples/synthetic.validation.json).

## Controlled matching measurement

```sh
.venv/bin/python scripts/benchmark_matching.py --output examples/matching
.venv/bin/autoedit validate --plan examples/matching/enabled.plan.json --video examples/matching/enabled.mp4
.venv/bin/autoedit validate --plan examples/matching/random.plan.json --video examples/matching/random.mp4
```

The five generated sources comprise three related compositions and two deliberately mismatched compositions. Both methods use the same detected-beat schedule, speed limits, feasible candidate pools and whole-edit no-reuse constraint. The baseline makes seeded random feasible choices. Both paths are refined and rendered. Independent full-resolution RGB pixel MAE and Sobel-edge MAE are computed from decoded outgoing/incoming frames, without using planner descriptors or reported transition costs.

| Measured boundary difference (lower is better) | Matching | Seeded random |
| --- | ---: | ---: |
| Mean RGB MAE, normalized 0–1 | 0.10226829 | 0.37403944 |
| Mean edge MAE | 0.00982878 | 0.02171192 |

RGB boundary error was **72.66% lower** with matching. The fixture has four shots and three internal cuts, at output frames [31, 66, 103]. Mean error to the independently known synthetic transients was 27.778 ms; worst error was 41.667 ms. Maximum error to the selected detected beats was 19.195 ms. Both outputs fully decode to 144 frames. The matching edit retains one visibly mismatched transition because four distinct clips must be drawn from only three matching sources. This establishes an improvement on this controlled fixture, not universal perceptual quality.

Artifacts: [measurements](examples/matching/results.json), [matching video](examples/matching/enabled.mp4), [baseline video](examples/matching/random.mp4), [matching pairs](examples/matching/enabled.contact.png), [baseline pairs](examples/matching/random.contact.png), [plans](examples/matching/selected.plans).

## Supplied footage, exact default settings

```sh
.venv/bin/autoedit create \
  --audio tiktok_niko.cc116_7695074868118785294_audio.mp3 \
  --clips ./clips --output examples/real.mp4
.venv/bin/autoedit validate --plan examples/real.plan.json --video examples/real.mp4
autoedit create --audio tiktok_niko.cc116_7695074868118785294_audio.mp3 --clips ./clips --output edit.mp4
autoedit validate --plan edit.plan.json --video edit.mp4
```

The original refreshed global `autoedit` command completed this invocation. That original timeline is preserved in `examples/real.mp4` and its adjacent plan/report; the root `edit.mp4` has since been replaced by the native-aspect, more sensitive edit documented above.

All 136 input clips were analyzed. The default command chose 14 shots and produced 625 frames at 1080×1920/30 fps, duration 20.833333333s, H.264/yuv420p with one AAC soundtrack. No rhythm fallback was needed. Maximum cut-to-selected-beat error was **16.327 ms**, under one frame (33.333 ms). Soundtrack correlation was **0.98816989** with **zero measured lag**. Maximum normalized boundary-frame error against the planned native source was 0.008323. Full decode, dimensions, codecs, frame counts, PTS, and source-boundary checks all passed.

Artifacts: [finished video](examples/real.mp4), [plan](examples/real.plan.json), [HTML inspection report](examples/real.report.html), [validation](examples/real.validation.json), [pairs 1–6](examples/real.contact.png), [pairs 7–13](examples/real.remaining-contact.png).

## Review and observed limits

All 13 real transition pairs and both controlled contact sheets were visually reviewed. Many real transitions align similar head positions, poses and cool palettes; some backgrounds/framing still change abruptly. No unintended all-black joins were visible. Landscape footage has large intentional bars under the requested default portrait `contain` fit. All selected real playback speeds were 1×; the mixed-speed integration test verifies actual nonunit rendering. The final soundtrack ended between frames, so the default interval was floored to the last complete frame and that warning is saved in the plan.

Audio timing was checked objectively against decoded soundtrack waveforms and known synthetic events. **Subjective listening is unavailable in this runtime and has not been claimed.** The playable MP4s are supplied for audition. This is the remaining unperformed subjective review step.

The planner uses low-resolution nonsemantic descriptors, approximate sampled scene detection and bounded beam search; it can miss a better global path or a fast internal cut between samples. Motion on texture-poor footage is treated as uncertain. It uses hard cuts, fixed-speed segments and constant output frame rate; it does not implement downbeats, semantic story selection, crossfades, motion interpolation, GPU acceleration or arbitrary musical phrasing. Plans reproduce the timeline and verify source content; byte-identical encoding across different FFmpeg builds is not promised.
