# Interactive beat review validation

Version 0.3.0, tested locally on CPU on 2026-10-18.

## Start the editor

```sh
uv sync --locked
uv tool install --force --constraints requirements.lock .
autoedit review --plan edit.plan.json
```

This opens `http://127.0.0.1:8765`. Your timing choices save to
`edit.review.json`; the original plan and edit are preserved. The editor resumes
the draft on the next launch. A different port or draft can be selected with
`--port` or `--session`; `--no-browser` prints the URL without opening it.

## Automated checks

Commands executed:

```sh
uv run --locked pytest -q > examples/review-all-tests.log 2>&1
node --check src/autoedit/web/app.js
uv pip check
uv build > examples/review-build.log 2>&1
uv venv /tmp/autoedit-review-install --python 3.13
uv pip sync --python /tmp/autoedit-review-install/bin/python requirements.lock
uv pip install --python /tmp/autoedit-review-install/bin/python --no-deps \
  dist/scene_splitter-0.3.0-py3-none-any.whl
```

Actual full-suite result:

```text
........................................................................ [100%]
72 passed in 62.87s (0:01:02)
```

All 36 installed packages are compatible. The wheel and source distribution
build successfully. The wheel includes the HTML, CSS, JavaScript, bundled
wavesurfer.js 7.12.12, license and verified package integrity metadata. Node was
used only to check JavaScript syntax; running the installed editor requires no
Node installation, package manager, CDN, API key or model download.

Sixteen new tests cover the review server and fixed-cut planner. They include
HTTP media ranges, saved/reloaded drafts, optimistic revision checks, reset,
malformed markers, source fingerprints, local-request restrictions, exact manual
timing provenance, forged detected events, strict no-cut ranges, infeasible
no-reuse schedules, rendered preview/export, overwrite behavior, job failures
and one-job-at-a-time enforcement. Real media jobs finish before fixture cleanup.
Existing audio, visual/motion search, rendering and replay tests also pass.

## Real browser workflow

The T3 collaborative browser was used against the actual local server, with
native clicks, keyboard presses, dragging, form submission and playback. A copy
of the supplied-footage plan keeps the original edit intact:

```sh
uv run --locked autoedit review \
  --plan examples/web-review/example.plan.json --video edit.mp4 \
  --port 8766 --no-browser
```

Verified interactions:

1. Load all 41 detected beats with the automatic selected cuts and quiet intro.
2. Select the cut at frame 247 and press Right twice: it moves to frame 249.
3. Disable the cut at frame 262; protect frames `[300, 330)` from cuts.
4. Undo/redo the range operation, save automatically and reload the page.
5. Click waveform markers to keep/skip them; drag a cut, then undo its move.
6. Update the preview; the out-of-date label clears only when its saved revision
   matches. The real preview is 640×360 with the full 625-frame timeline.
7. Export through the filename/replacement dialog. Download links for video,
   plan, validation and HTML report remain available after page reload.
8. Attempt a second export without replacement permission: a clear existing-file
   error appears and the completed output is preserved.
9. Play, pause and loop the selected cut. The loop wraps from approximately
   9.19 seconds back to 7.31 seconds around the selected 8.3-second cut.
10. Check the layout at 390-pixel width: no document-level horizontal overflow.

Playback uses the preview video as one audio/video/waveform clock. The unused
PCM audio player stays paused. Sampled waveform time readout differed from the
video clock by at most 12.1 ms in the browser check; this is an observation on
this machine, not a cross-browser latency guarantee. An independently installed
wheel was also opened in the browser without a preview video: PCM waveform
playback, pause, all markers and the unrendered-preview state worked.

## Actual revised export

```sh
uv run --locked autoedit validate \
  --plan examples/web-review/example.reviewed.plan.json \
  --video examples/web-review/example.reviewed.mp4
```

Actual result: `Validation passed: 625 decoded frames; beat error 15.92 ms`.

- 1920×1080, 30 fps, 20.833333333 seconds, H.264/yuv420p with AAC soundtrack.
- 25 shots, 25 distinct source clips; the original quiet intro still holds until
  frame 233 (7.766666667 seconds).
- Frame 247 moves to 249 (8.3 seconds): a 66.667 ms adjustment to the previous
  output boundary. The original detected time, 8.241451247 seconds, is preserved
  separately and the adjusted cut is explicitly classified as user timing.
- Cuts at frames 262, 305 and 320 are absent; no internal cut lies in the strict
  `[300, 330)` range. No automatic cuts were inserted to replace them.
- Maximum untouched detected-beat quantization error: 15.918 ms.
- All rendered source-boundary comparisons pass; maximum normalized RGB MAE:
  0.00214846.
- Soundtrack correlation: 0.98816989; measured lag: 0 seconds.
- The copied base plan remains byte-identical to the original `edit.plan.json`.

[Revised video](examples/web-review/example.reviewed.mp4) ·
[Reproducible plan](examples/web-review/example.reviewed.plan.json) ·
[HTML transition report](examples/web-review/example.reviewed.report.html) ·
[Validation JSON](examples/web-review/example.reviewed.validation.json) ·
[Saved timing draft](examples/web-review/example.review.json) ·
[Editor screenshot](examples/web-review/editor.png)

## Design constraints

This editor controls cut timing. It keeps the soundtrack interval and output
frame count fixed; clip selection and source trims remain automatic. A no-cut
range can make a schedule infeasible if no continuous clip is long enough at
the allowed speeds. Such jobs fail clearly rather than adding hidden cuts or
filler. The original reuse policy remains in effect; current defaults forbid
clip reuse.

Revised video previews are rendered on request. CPU planning and rendering can
take tens of seconds for many sources; unchanged selection results are cached
and export reuses the selected full-resolution plan. One render runs per editor
process at a time. Timing edits can continue during a render, with older previews
labelled out of date. Browser media seeking is intended for audition; exact cut
frames are established by the rendered output and decoder validation.

Use `--video` only for an existing video rendered from the selected plan. The
server binds to localhost and serves this session's assets and registered media.
It is a local workstation tool, with no remote multi-user hosting support.
