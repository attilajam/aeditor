# CCIP character recognition models

The artifacts come from [deepghs/ccip_onnx](https://huggingface.co/deepghs/ccip_onnx/tree/eb2acdd29af1703388d3d0c04221add322bc9110/ccip-caformer-24-randaug-pruned), revision `eb2acdd29af1703388d3d0c04221add322bc9110`:

| Cache filename | Upstream file | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `anime_identity_features.onnx` | `model_feat.onnx` | 150248245 | `4ea118d16496274f4f6e08d3afc768cc592389e8f7f32f8732ce2215c228ac5f` |
| `anime_identity_metric.onnx` | `model_metrics.onnx` | 1649 | `7e4646fdfbe369ad485b227ec47ac4092c98b8deef3af44e19fdbd8ba7bc25c1` |

The [upstream implementation](https://raw.githubusercontent.com/deepghs/imgutils/main/imgutils/metrics/ccip.py) expects images containing a single anime character. Convert to RGB, resize directly to 384 × 384 with PIL bilinear interpolation, transpose to CHW, divide by 255, and normalize each channel with mean `(0.48145466, 0.4578275, 0.40821073)` and standard deviation `(0.26862954, 0.26130258, 0.27577711)`. Stack as float32 NCHW. The feature model takes `input` and returns `output`, a float32 `[N, 768]` feature array.

Stack features as float32 `[N, 768]` for the metric model's `input`; its `output` is an `[N, N]` matrix of pairwise differences. Smaller means more similar. Upstream classifies pairs as the same when their difference is at most `0.17847511429108218`, the published default threshold. Treat this as an uncalibrated starting value for this application; it does not establish video accuracy.

The [model repository](https://huggingface.co/deepghs/ccip_onnx) labels its license `openrail`. That metadata does not identify the exact license text or establish licensing clearance; applicable terms and permissions remain to be confirmed with the model authors before distribution or use.

## Editor behavior

Video import starts local character analysis after scene detection. The Characters
row shows portraits and scene counts. Click a portrait or its button to cycle
With → Only → Without → All (clear that portrait's filter). Multiple portraits
combine with AND. With no active filter, the existing character-present scene
policy applies; Without can include analyzed background scenes.

Filters apply to both the scene browser and edit generation. Only and Without
exclude unresolved or incompletely sampled scenes. Sampling starts at six frames
per second at a 640-pixel longest edge, with an additional sample for short scenes
missed by the grid. This cannot establish frame-perfect absence. The published
distance threshold and identity margin have not been calibrated on editor footage.

Review appearances to move a detection between groups, create a new group, ignore
a false detection, or choose a portrait. Portrait buttons also offer rename and
merge menus. Right-click a scene to confirm its complete character list. Machine
results live in `.ae-cache/`; source-specific corrections live separately in
`.ae-project/character-edits/`. Corrupt or incompatible corrections are preserved
and reported, rather than overwritten. Both locations are local and ignored by Git.

## Verification

The pinned models have passed actual feature/distance inference, not only shape
stubs. A selected media test used the detector project's example artwork, found
characters, grouped appearances, generated portraits, applied selection, retained
a saved label, and rendered the resulting edit. The example is cached locally,
not committed. This verifies integration, not recognition accuracy across styles.

To prepare the optional fixture from the repository root:

```sh
mkdir -p .ae-cache/character-fixtures
curl --fail --location https://raw.githubusercontent.com/ksasao/anime-character-detector/main/result.jpg --output .ae-cache/character-fixtures/anime-example.jpg
```

Run selected real-model checks individually; do not enable every ignored test:

```sh
cargo test-fast -p ae-similarity character_encoder_real_models -- --ignored
cargo test-fast -p ae-gui character_identity_real_media_roundtrip -- --ignored
```

Labeled scenes from representative user footage are still needed for the plan's
accuracy and correction-effort gates. Real-person recognition is deferred.
