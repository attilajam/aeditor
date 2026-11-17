# Animated character tagging and filtering

Status: candidate implementation added. Scoped regression tests and a selected
real-model/media round trip have passed. The labeled, representative-footage
accuracy checkpoint below remains outstanding; no general accuracy claim is made.

## Outcome and scope

On importing a video, analyze its scenes, discover recurring animated characters,
and show a portrait for each character. Clicking a portrait cycles through scenes
containing that character, scenes containing only that character, and scenes
without that character. A fourth click clears that portrait's filter.

Confirmed requirements:

- First release recognizes animated characters. Real people come later.
- Character filters affect both the scene browser and scenes eligible for an edit.
- Portraits come from the imported video; character names are optional user labels.

Launch scope includes automatic grouping, manual correction, local persistence,
progress/error states, and consistent filtering throughout timeline operations.
One imported video is the identity scope, matching the current controller.

Defer automatic fictional-character names, cross-video identity libraries, cloud
recognition, model training, character-following camera changes, and a CLI tagging
workflow. None is necessary for portrait-based selection.

## Current implementation and necessary changes

- `ae-gui/src/project_controller.rs`: `detect_characters()` samples three proxy
  frames per usable scene and stores a boolean in `character_detections`, keyed
  by usable time range. It cannot distinguish identities or count characters.
- `ae-render/src/reframe.rs`: `CharacterDetector` decodes the existing YOLOX model
  and returns one preferred center through `FaceDetector`. Recognition needs all
  character rectangles, with duplicates removed, and explicit inference errors.
- `character_filtered_shots()`, `snapshot()`, and `swap_slot_shot()` enforce the
  existing character-present rule. Alternative ranking also uses that rule.
  These must share the new predicate so browser and editing cannot disagree.
- `ae-gui/src/app.rs`: the media browser already has filters, asynchronous
  controller actions, snapshot sequence numbers, and texture caching to extend.
- `ShotPoolModel` supports trim, split, and merge; splitting/merging renumbers
  shot IDs. `TimelineModel` and the descriptor bank index by those IDs. Character
  evidence must therefore be anchored to source timestamps, not mutable shot IDs.
- `CacheManager` already provides source fingerprints and atomic JSON writes.
  The scene-analysis cache should remain independently reusable.
- Ingest proxies are only 240x136 or 320x180. Enlarging them to 640 pixels does
  not recover identity detail. Recognition needs a suitable source decode.
- `ae-similarity` has an ONNX embedding path, but it opens a session per call,
  takes only 64 output values, and can silently fall back to color/edge features.
  Those behaviors are unsuitable for identity recognition.

Paths in this section are under `crates/`. Preserve the existing working-tree
edits and inspect each function's callers again before implementation.

## Filter contract

The unit is the scene's current usable source interval, `[usable_start,
usable_end)`. A character appearing anywhere in that interval counts as present;
characters need not appear simultaneously. Do not silently trim scenes to make
them satisfy a filter. Rendering a shorter portion of an eligible scene does not
change the scene-level contract.

Each portrait has four states:

| State | Rule for character A |
| --- | --- |
| Neutral | No restriction from A |
| With | A has confirmed presence; other characters may appear |
| Only | A has confirmed presence, no other character is present, and there are no unresolved character detections |
| Without | A has no confirmed or plausible presence, and analysis of the usable interval is complete |

Cycle: `Neutral -> With -> Only -> Without -> Neutral`. Add a visible Clear
filters action and text labels; do not communicate modes through color alone.

Allow multiple active portraits. Combine their rules with AND: With A + With B
means both appear; With A + Without B means A appears and B does not. Only A +
Only B intentionally has no matches. Explain conflicting selections in the
empty state; never silently relax them or change the meaning of Only.

Examples for fully analyzed scenes:

| Scene contents | With A | Only A | Without A |
| --- | --- | --- | --- |
| A | yes | yes | no |
| A and B | yes | no | no |
| B | no | no | yes |
| No detected characters | no | no | yes |
| A plus an unresolved character | yes | no | no |
| Pending or failed analysis | no | no | no |

For Without A, an unresolved detection that cannot confidently exclude A blocks
the match. Only requires full interval coverage too. Here, complete means that
the configured sampling pass finished successfully, not that every source frame
was inspected or absence was proven. Surface this limit in the filter help.

With no active portrait filters, preserve today's default of selecting scenes
with detected characters. With active filters, use their result instead of
applying the old character-present gate first. This allows Without A to include
an analyzed background-only scene. Pending scenes remain visible with an
Analyzing label in the neutral browser, but are not auto-sequenced yet.

Character filtering must not overwrite `Shot.enabled`. Generation eligibility
is `manual_enabled && character_match`. Matching scenes remain visible when
manually disabled so users can re-enable them. Text search and sort remain
browser-only controls; character filters also affect sequencing.

## First milestone: prove recognition on actual animation

Do this before committing the rest of the UI to a model or promising accuracy.

1. Prepare a small labeled set of representative clips: clean close-ups, multiple
   characters, lookalikes, profiles, backs, costume changes, action blur, fades,
   occlusion, crowds, background-only scenes, and brief entrances. Include at
   least three intended animation styles and roughly 100 usable scene intervals.
   Use redistributable fixtures in the repo; keep restricted footage local.
2. Test the existing full-character detector with all-box decoding and NMS.
   Check whether it finds enough characters for the intended animation styles,
   rather than assuming anime detector results transfer to every cartoon.
3. Evaluate CCIP as the first identity candidate. It targets single-character
   images, and ONNX feature and distance models are published. Feed isolated
   detections rather than complete multi-character frames. The reference code
   defines preprocessing and a model-specific distance; do not substitute an
   arbitrary cosine threshold. [CCIP model card](https://huggingface.co/deepghs/ccip),
   [reference implementation](https://github.com/deepghs/imgutils/blob/main/imgutils/metrics/ccip.py).
4. Start with `ccip-caformer-24-randaug-pruned`; compare the smaller
   `ccip-caformer-6-randaug-pruned_fp32` only if runtime warrants it. Measure the
   actual target Mac with the existing `ort` runtime. Choose one variant for
   launch, not a model selector. [Published ONNX variants](https://huggingface.co/deepghs/ccip_onnx).
5. Verify exact artifact revisions, weight licenses and redistribution terms,
   input/output contracts, and checksums. CCIP is marked OpenRAIL; do not label
   it Apache/MIT or assume the hosting label resolves all distribution terms.
   The current detector is a single-class character detector, so identity must
   come from another stage. [Detector documentation](https://github.com/ksasao/anime-character-detector#model-information).
6. Use held-out scenes to choose detection and identity thresholds. Record false
   merges, split identities, missed appearances, unresolved rate, filter
   precision/recall, runtime, memory, and correction effort. Published image
   benchmarks are not editor acceptance results.

Proposed release gates: at least 95% precision for accepted identity assignments
and each filter mode, at least 90% recall for With on labeled visible appearances
lasting 0.5 seconds or more, and at least 80% scene coverage without manual
review. Score Only and Without against whole-interval ground truth, including
short appearances; report their recall as well so excluding everything cannot
pass. These are targets to test, not claims of achieved accuracy. Report results
per animation style, not only in aggregate.

If these gates fail, first adjust source resolution, crop isolation, and
thresholds. DINOv2 is an existing-manifest baseline worth comparing, but its
general visual features do not establish character identity accuracy.
[DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md).
If neither candidate works, record the failure and revise the model choice or
supported styles explicitly; do not ship scene-color similarity as recognition.

## Implementation structure

Use the existing crates and dependencies; no new crate, vector database,
clustering framework, Python runtime, or service is needed for the application.

Expected file scope:

- `crates/ae-render/src/reframe.rs` and `src/lib.rs`: all-box API and exports.
- `crates/ae-similarity/src/characters.rs` and `src/lib.rs`: identity inference.
- `crates/ae-gui/src/character_model.rs`, `src/lib.rs`,
  `src/project_controller.rs`, `src/shot_pool_model.rs`, `src/image_provider.rs`,
  and `src/app.rs`: state, integration, portraits, and controls.
- `crates/ae-gui/src/timeline_model.rs` only if required for filter/lock revision
  handling; preserve the existing sequencer API and descriptor indexing.
- `models/manifest.toml` and a short model-contract document beside it: selected
  artifacts, real checksums, licenses, preprocessing, and calibration results.
- Beside-code tests plus small labeled/golden fixtures under
  `tests/fixtures/characters/`; keep real-model runs separately selectable.
- `.gitignore`: exclude `.ae-project/` local correction data and generated
  portraits. Check existing media/model ignore rules before adding fixtures.

No workspace dependency or lockfile change is expected. Inspect consumers of new
exports and update any affected existing tests/fixtures as part of each step.

### All-character detection in `ae-render`

Add an analysis API alongside the current tracking API, initially in
`reframe.rs`, exporting it through `ae-render/src/lib.rs`:

- Return all retained rectangles and scores in input-frame coordinates as a
  `Result<Vec<CharacterDetection>>`.
- Reuse YOLOX preprocessing, ONNX environment/session setup, grid decoding, and
  coordinate mapping. Preserve width/height instead of discarding them.
- Apply score filtering and deterministic IoU NMS; clip to actual image bounds,
  reject padding-only/degenerate/non-finite boxes, and retain low-quality
  detections as unresolved evidence rather than falsely calling a scene empty.
- Share raw decoding with the current single-subject adapter. Preserve its
  existing continuity selection and fallback behavior; do not make framing
  identity-aware as part of this feature.
- The analysis API must return failures. The renderer's centered-crop fallback
  cannot stand in for successful character analysis.

### Identity inference in `ae-similarity`

Add a focused `characters.rs` module, re-exporting its concrete encoder through
`lib.rs`. It accepts character crops, not `ae-render` types, so no new crate
dependency is needed.

- Keep feature and distance sessions alive for the analysis job; reuse
  `ae_core::onnx_environment()`.
- Reproduce the selected model's preprocessing and full feature output. For the
  initial CCIP candidate, verify the reference RGB normalization, 384-pixel
  input, output dimensions, and metric-model contract against pinned artifacts.
- Validate tensor dimensions, finite values, and metric outputs. Return errors
  rather than falling back to the existing layout descriptor.
- Process bounded crop batches and distance comparisons. Do not construct an
  all-observations squared distance matrix.
- Keep character embeddings separate from `FrameDescriptor.combined_embedding`;
  the current match-cut scoring and its cache format should stay independent.

### Character state in `ae-gui`

Add `character_model.rs` for analysis results, identity grouping, manual edits,
and the pure filter predicate. Use existing `Rational`, frame, and error types.
Keep these feature-specific types out of `Shot` and `ae-core` unless another
consumer actually needs them.

Persist enough evidence to re-project onto changed scenes:

- A source fingerprint, schema/model/preprocessing versions, sampling policy,
  and successfully scanned time coverage.
- Observations with stable IDs, source PTS, source rectangle, detector quality,
  identity feature, confirmed assignment or plausible candidate identities.
- Character records with stable local IDs, optional label, exemplar observation
  IDs, portrait reference/revision, and review state.
- Manual changes separately from inferred assignments. User reassignment wins
  over later grouping; provisional automatic results must not overwrite it.
- Per-scene derived tags: present IDs, possible IDs/unresolved evidence, and
  pending/complete/failed status. These can be rebuilt from source evidence.

Do not put embeddings or full images in every `GuiStateSnapshot`. Send compact
portrait/tag/count/filter state and a result revision. Add fields to
`ShotViewItem` only where the UI needs them; use serde defaults for compatible
new snapshot fields and update constructors/test fixtures.

## Analysis pipeline and grouping

1. After verified scene boundaries are available, load character results for the
   matching source/models/configuration, or start the analysis worker.
2. Decode sequentially at the source aspect ratio, initially 640 pixels on the
   longest edge and 6 fps. Include a representative sample inside every short
   scene even if the regular grid misses it. Reuse existing proxy frames only
   when their resolution/aspect/PTS meet the recognition requirements. Use
   bounded `MediaBackend` streaming or short chunks, not a whole-video frame
   vector or a separate FFmpeg process for each detection.
3. Detect every retained character and produce tightly bounded crops. Reject
   badly mixed/occluded crops from confident identity matching, while retaining
   their unresolved-presence evidence. Crop from a higher-resolution source
   sample when a promising detection is too small for identity extraction.
4. Match clean observations against a bounded set of actual exemplars per
   character. Use the model's distance and a calibrated acceptance threshold
   plus separation from the next-best identity. Avoid averaging embeddings
   unless validated for the chosen metric. Ambiguous observations remain
   unresolved; do not force the nearest identity.
5. Seed a new provisional identity from a good unmatched crop. Require a second
   consistent observation before treating it as an established portrait group;
   retain single appearances in a review group so brief characters are not
   dropped. Use timestamps to avoid counting duplicate samples as support.
6. Maintain several diverse exemplars per identity, initially capped at eight.
   Use deterministic source-time ordering and tie-breaking. Prefer extra groups
   over false merges. A match to one exemplar must not cause an uncontrolled
   chain merging visually different identities. Co-occurring detections are
   useful ambiguity evidence, but reflections or repeated drawings mean they
   are not an unconditional different-identity rule.
7. Revisit ambiguous observations after the first pass establishes exemplars.
   Re-sample unresolved intervals more densely, up to source-frame cadence for
   short intervals. Record any remaining gaps instead of marking them complete.
8. Aggregate all evidence within each usable interval; generate one portrait
   from a sharp, sufficiently large, isolated representative crop. Keep the
   portrait stable until a materially better exemplar or user choice replaces it.

Six fps is a starting performance policy, not an absence guarantee. Do not cap
each long scene at three frames. Measure brief entrances and false empty scenes
in the first milestone, and raise baseline sampling or narrow supported footage
if necessary. The production constants and unresolved thresholds follow those
measurements. Any remaining sampling shortcut gets the repository's `shortcut:`
comment explaining its limit and upgrade condition.

## Cache, corrections, and scene edits

- Store replaceable machine analysis using `CacheManager` under a separate
  namespace such as `gui-character-analysis-v1`. Its key includes the source
  fingerprint, actual model checksums, preprocessing/grouping/sampling versions,
  and decode geometry. Keep the existing shot/descriptor cache independent.
- Validate loaded ranges, IDs, references, rectangles, feature dimensions, and
  finite values. A corrupt/stale machine cache is a miss; never a successful
  empty analysis. Publish atomic results and reuse completed work on retry.
- Store manual corrections in a small source-specific sidecar under
  `.ae-project/character-edits/`, separate from disposable `.ae-cache/`, reusing
  `CacheManager`'s atomic writer with that root. This adds no general project
  save system. Persist stable identity/exemplar anchors and correction evidence
  so clearing the machine cache does not silently erase labels and merges.
- If a source or model change makes an override ambiguous, retain it and flag
  it for review. Do not apply it to unrelated observations or discard it.
  Corrupt/unwritable correction data gets a visible error and is not overwritten
  automatically. Filter selections can remain session-local for this release.
- Trimming rebuilds tags from observations in the new usable interval. Splitting
  reassigns evidence by PTS and retains character IDs; merging unions evidence
  and scan coverage. A newly exposed or inadequately sampled range is pending.
- Manual presence/removal corrections are anchored to an explicit source-time
  interval. If a split/trim makes the original assertion insufficient, keep the
  saved correction but require review of the affected child interval rather
  than blindly copying an A-present tag into both children.
- Preserve the existing shot-ID/descriptor-bank invariant. Regenerate derived
  mappings when shots are renumbered; never use a filtered/reindexed shot vector
  as the source of truth.
- Source paths participate in today's fingerprint. Moving a video may require
  explicit correction reattachment; automatic cross-file migration is deferred.

## Controller and timeline behavior

Replace the boolean-specific policy at every entry point with one shared
predicate. Return an unrenumbered clone of all shots with eligibility reflected
in the clone's `enabled` values for the existing sequencer API.

Required integration points:

- Import completion and progressive snapshots.
- `resequence()` and its callers, including beat/configuration changes.
- `rank_slot_alternatives()` and manual `swap_slot_shot()`/drag-and-drop.
- `snapshot()` and `browser_items()` so the browser and candidate pool agree.
- `split_shot()` and any current/future controller trim/merge route.
- Quick Render, preview invalidation, and export of the current timeline.

On a filter change, update the browser immediately and resequence once using
the established controller action path. Coalesce rapid clicks by revision so
intermediate states do not queue repeated expensive resequencing. Preserve
manual enable flags and valid locked slots. Incompatible locked slots are
replaced according to the existing resequence policy, with a visible count of
affected locks; locks must not bypass the character restriction.

When a filter change is pending, invalidate the old draft and enhanced preview;
prevent rendering/exporting an old timeline under the new filter state. Publish
the new timeline and its revision together, then request its preview. Handle a
failed resequence visibly rather than ignoring its `Result`. With zero eligible
scenes, clear the old timeline/draft and show a useful empty state. Never fall
back to excluded scenes. If there are too few candidates and reuse is off,
preserve the existing shortage behavior and explain it without enabling reuse.

Character filters do not alter transitions, render framing, or cut-similarity
scoring. Scene-level eligibility remains valid when a slot slips inside the
same usable interval. Validate boundary changes before reusing its tags.

## GUI and correction workflow

Add a horizontally scrollable Characters row above the Media scene grid. Each
portrait shows its optional label, matching-scene count, and current mode.
Show filter badges in text: With, Only, Without. Supply keyboard focus,
Space/Enter activation, tooltips, and readable contrast at the existing HiDPI
scales. Keep audio browsing independent of character selection.

Use the current RGBA cache/egui texture pattern for portraits, keyed by source
generation, character ID, and portrait revision. Clear obsolete textures on
source replacement and reuse the upload budget; avoid decoding/cropping in an
egui paint callback. Counts use the complete scene pool before current character
filters, so selections do not make the other portraits disappear.

Add a compact review panel reachable from a portrait or scene's character tags:

- Rename a group or choose its portrait.
- Merge two groups that are the same character, retaining a canonical ID and
  remapping filters, assignments, and overrides atomically.
- Move selected appearances to an existing/new group to fix a false merge.
- Add/remove a character tag for a specified scene interval and review unresolved
  appearances. A manual exclusion of A alone does not certify that no other
  unknown characters exist; Only needs a reviewed full character list.

Persist a correction successfully before replacing the accepted state, or show
that it is unsaved. Correcting tags must not rerun detection for the entire
video; recompute affected grouping/tags and resequence once.

Provide distinct messages for analyzing, ready with unresolved scenes, model
failure, no characters found, and no filter matches. The active filter summary
must remain visible when the grid is empty, with Clear filters and Review
unresolved actions where applicable.

## Background work and failure behavior

Reuse threads, channels, progress tracking, and snapshot sequence numbers.
Long inference runs outside the GUI thread and outside a long-held controller
or media-cache lock. Copy only a bounded input chunk under the lock, process it,
and commit results only if source generation and analysis revision still match.
Check cancellation between decode/inference batches. A new source invalidates
old results and portrait textures; an old worker cannot modify the new source.

Download only selected artifacts, using pinned manifest URLs/checksums and the
existing temporary-file/verify/rename pattern. Include both CCIP feature and
distance artifacts if required. An interrupted/invalid download must never
replace a verified model. Make detection/grouping a local operation after model
download; no footage or portraits are sent to a service.

Failed/incomplete analysis is unknown, not absence. Preserve already usable
results and expose retry. New identity filters cannot use unsupported results;
the neutral flow may retain the existing successful character-presence analysis.
Do not repeatedly download/retry a failed model during every resequence.
Cache-write failure may leave analysis usable in memory with a warning; manual
correction persistence failure must be separately visible.

## Delivery order and validation

1. **Model and sampling checkpoint.** Produce the representative labels,
   supported-style decision, pinned-artifact contract, accuracy results, and
   target-machine timings. Resolve model suitability/distribution before
   declaring the automatic-recognition approach accepted.
2. **Detection and identity backend.** Add all-box detection/NMS and persistent
   encoder sessions. Test with pure tensors and crop/embedding golden fixtures,
   then run the actual selected ONNX models. Keep tracking behavior compatible.
3. **Character model, cache, and predicate.** Implement deterministic grouping,
   uncertainty, evidence projection, correction storage, and the four-state
   predicate. Use small synthetic observations to test state transitions before
   wiring the UI.
4. **Controller consistency.** Connect import, eligibility, alternatives,
   assignment, locks, edit invalidation, scene edits, and cancellation. Use
   labeled/stub observations so most regressions need no model downloads.
5. **Portrait GUI and review.** Add the row, mode labels, empty states, correction
   workflow, and texture lifecycle. Verify keyboard interaction and HiDPI.
6. **Real-video acceptance.** Run the labeled footage end to end, correct some
   identities, reimport, split/trim scenes, generate an edit, and render it.
   Review generated source intervals against the labels for every filter mode.
   Confirm failures and stale results cannot produce a misleading export.

Required focused regressions:

- NMS removes duplicate boxes but retains two distinct characters; geometry
  survives letterboxing, portrait sources, image edges, and padding.
- Invalid model/tensor outputs are errors, not empty detections.
- Recognition preprocessing and metric outputs agree with a pinned reference
  within a documented tolerance; repeated frames reuse sessions.
- Same identity across different poses is grouped; lookalikes and ambiguous
  margins remain separate/unresolved. Results are deterministic.
- The filter truth table, click cycle, AND combinations, empty backgrounds,
  unknown/failed results, manual disabled scenes, and neutral default all hold.
- A second character appearing later in a scene prevents Only A; a possible A
  blocks Without A. Source-frame boundaries never include adjacent-scene tags.
- Cache hits skip inference; changed model/policy/source invalidates analysis;
  corrupt cache recovers; corrupt corrections remain recoverable and visible.
- Split/trim/merge project observations correctly despite renumbered shot IDs.
  Labels, merges, reassignments, and interval corrections survive a reimport.
- Alternatives, drag-and-drop, locked slots, browser visibility, and automatic
  selection use the same predicate. Rapid changes publish only the latest state.
- Source replacement/cancellation rejects stale results. Portrait uploads remain
  bounded. Zero matches clears stale preview/export state.

Use repository aliases sequentially, with actual regression names introduced by
the implementation. Check that every targeted filter runs at least one test.
After focused regressions, run the affected-package tests once:

```sh
cargo test-fast -p ae-render -p ae-similarity -p ae-gui
```

If tests do not compile a changed consumer target, run the relevant check:

```sh
cargo check-fast -p ae-cli -p ae-gui
```

Format only edited Rust files using `rustfmt --edition 2024 --config
skip_children=true --check <edited-file>`. Add workspace checks only if the
actual shared API/schema scope requires them. Select individual real-model tests
explicitly; do not globally enable ignored tests. Use normal dev/release for
performance measurement, not the unoptimized agent profile. Record hardware,
clip length, sampling/resolution, model variant, cold/warm runtime, inference
counts, peak memory, and UI responsiveness. No real-time speed promise until
those measurements exist.

The feature is done when the recognition checkpoint passes for declared
supported footage, every click mode controls browser and edit eligibility,
corrections persist, scene edits retain correct evidence, and the real-model
rendered acceptance path passes. A stub-only test pass is insufficient.

## Later: real people

Add human face detection, alignment, and identity embeddings as a separate
analysis path once animation is working. Reuse the portrait/filter/correction
model and UI without assuming animation embeddings recognize people. Define
what back-facing or fully occluded people mean for Only/Without and benchmark
that footage before offering the same guarantees. No human-recognition models
or controls are required for this release.
