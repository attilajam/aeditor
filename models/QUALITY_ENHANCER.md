# Quality enhancer

The optional enhancer uses Philip Hofmann's **1xgaterv3_r_restore**, a trained
GaterV3 model for noise, compression artifacts and mild blur. It restores RGB
pixels at 1× resolution; alpha, frame size and timestamps are preserved.

Model and ONNX conversion: [original release](https://github.com/Phhofm/models/releases/tag/1xgaterv3_r_restore).
Weights: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), Philip Hofmann.
Architecture: [GaterV3](https://github.com/umzi2/GaterV3), MIT.
The FP32 ONNX weights are used unmodified, with overlapping tiles during inference.
The sharper preset blends 75% restored pixels with 25% source pixels, then adds
luminance sharpening at strength 0.6 with a 3×3 Gaussian kernel. A two-level noise
threshold and eight-level correction cap limit grain amplification and halos.
Alpha, timestamps and resolution remain unchanged. This preset is used for
AI-enhanced export.

First use downloads approximately 86 MB into ONNX Runtime's local model cache;
subsequent uses work offline. Processing happens locally. Build-time ONNX Runtime
binaries are supplied by the existing `ort` dependency. macOS uses Core ML
acceleration with fixed tile shapes; other platforms use CPU inference.
Enhancement adds render time and may alter intentional grain or fine texture.

Settings → Output offers Off (default), Light cleanup (fast), or AI restoration
for export. Light cleanup runs the same FFmpeg denoising, sharpening and color
preset as the preview, inside the existing encoder pass at the selected output
resolution, frame rate, codec and quality settings. AI restoration remains the
slower neural option. Settings → Preview applies
stronger FFmpeg denoising (`hqdn3d=2.4:1.8:3.6:2.7`), luma sharpening
(`unsharp=5:5:0.85:5:5:0`) and a restrained contrast/color boost
(`eq=contrast=1.06:saturation=1.08:gamma=1.015:gamma_weight=0.7`) to the
completed edit in one video pass. Denoising precedes sharpening to limit grain
amplification; chroma sharpening stays off to limit colored edge halos.
This preset improves perceived clarity rather than recovering missing detail.
It may soften intentional grain or exaggerate edges on already sharpened
footage. Preview cleanup does not run the AI model or download weights, and
does not represent the AI export result. The export selection is independent.

Preview retains separate original and filtered videos, copying audio without
re-encoding. Playback and seeking decode these cached videos. Hide/show reuses
the completed filtered video. Changing the edit rebuilds the original preview
and filters that whole video again; the original draft renderer still reuses
unchanged clip segments. Filtered previews use a separate cache key from older
AI-enhanced previews.
Drag the comparison divider or use its slider with the keyboard. Preview uses
the existing half-size draft resolution with CRF 18 to preserve fine detail;
export uses the selected output resolution and compression setting.
