//! Opt-in subject locking for preview and final rendering.
//!
//! A small fixed crop reserves pan room even at the source aspect ratio. The default locks
//! the detected center directly and holds it through dropouts. Character models
//! track full bodies without requiring a visible face; YuNet remains available
//! for human faces. Crop and resize preserve the output aspect ratio.
//!
//! Without a usable face model every frame falls back to a fixed centered crop,
//! which still preserves aspect ratio. The default render path is untouched.

use std::path::{Path, PathBuf};

use ae_core::{AeError, Resolution, Result, RgbaFrame};
use ae_media::stretch_frame_lanczos;

/// Subject-locking behavior. All fractions are relative to the crop size.
#[derive(Clone, Debug)]
pub struct ReframeConfig {
    pub enabled: bool,
    /// Optional character or YuNet ONNX model. Missing file => centered CLI fallback.
    pub model_path: Option<PathBuf>,
    /// EMA blend factor toward the detected target per frame (0..1].
    pub ema: f32,
    /// Movements smaller than this fraction of the crop size are ignored.
    pub dead_zone_frac: f32,
    /// Maximum pan per frame as a fraction of the crop size; zero is uncapped.
    pub max_pan_frac: f32,
    /// Frames to hold the last position while detection drops out.
    pub dropout_hold_frames: usize,
    /// Minimum detector score to accept a face.
    pub min_score: f32,
}

impl Default for ReframeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_path: None,
            ema: 1.0,
            dead_zone_frac: 0.0,
            max_pan_frac: 0.0,
            dropout_hold_frames: usize::MAX,
            min_score: 0.5,
        }
    }
}

/// Integer crop rectangle in source-frame pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Tracked subject center in source-frame pixels.
#[derive(Clone, Copy, Debug)]
pub struct FaceBox {
    pub cx: f32,
    pub cy: f32,
    pub score: f32,
}

pub trait FaceDetector {
    fn detect(&mut self, frame: &RgbaFrame) -> Option<FaceBox>;
    fn reset(&mut self) {}
    fn finish_shot(&self) -> Result<()> {
        Ok(())
    }
}

/// Fallback used when no model is available: always yields the centered crop.
#[derive(Clone, Debug, Default)]
pub struct CenterFallback;

impl FaceDetector for CenterFallback {
    fn detect(&mut self, _frame: &RgbaFrame) -> Option<FaceBox> {
        None
    }
}

/// Real YuNet face detector over the existing `ort` dependency, matching the
/// OpenCV `FaceDetectorYN` postprocessing used with the
/// `face_detection_yunet_2023mar` export (see `models/manifest.toml`).
///
/// - Input is the `input` blob: 640x640 BGR float in raw 0-255 pixel scale,
///   fit without distortion and zero-padded on the right/bottom.
/// - Per stride 8/16/32 the model emits named heads: box offsets (`bbox_*`),
///   face confidence (`cls_*`) and objectness (`obj_*`). Score per cell is
///   `sqrt(clamp(cls) * clamp(obj))`; boxes decode as
///   `cx = (c + dx) * s`, `w = exp(dw) * s` (likewise y/h), i.e. the prior is
///   the cell itself with size = stride. Landmark heads (`kps_*`) are ignored:
///   reframing only needs the face center.
/// - The highest-scoring cell above threshold wins (no NMS: a single crop
///   target is all the reframer consumes).
///
/// Any load/inference/shape surprise degrades sticky to the centered fallback
/// instead of failing the render.
pub struct YunetDetector {
    model_path: PathBuf,
    min_score: f32,
    session: Option<ort::session::Session>,
    failure: Option<String>,
}

/// YuNet working resolution. The 2023mar export fixes `input` at 640x640, so
/// stride grids are exactly 80/40/20 cells per side.
pub const YUNET_INPUT: usize = 640;
/// Strides decoded, finest first (mirrors the OpenCV head order).
pub const YUNET_STRIDES: [usize; 3] = [8, 16, 32];

/// Owned copy of one f32 model output, decoupled from the ort session borrow
/// so postprocessing stays pure and unit-testable.
#[derive(Clone, Debug)]
pub struct NamedHead {
    pub name: String,
    pub shape: Vec<i64>,
    pub values: Vec<f32>,
}

/// Decoded face in working-image pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YunetFace {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
}

/// Accept `[1, C, K]` or `[C, K]` head layouts; returns `(cells, channels)`.
fn norm_head_shape(shape: &[i64]) -> Option<(usize, usize)> {
    match shape {
        [1, c, k] => Some((*c as usize, *k as usize)),
        [c, k] => Some((*c as usize, *k as usize)),
        _ => None,
    }
}

/// First head whose name contains any of `keys` and whose cell count matches.
/// Cell-count matching (rather than parsing stride suffixes) accepts both the
/// `bbox_8`-style names of the 2023mar export and plain `loc`/`conf`/`iou`
/// names from other YuNet exports.
fn head_for<'h>(heads: &'h [NamedHead], keys: &[&str], cells: usize) -> Option<&'h NamedHead> {
    heads.iter().find(|h| {
        let n = h.name.to_lowercase();
        keys.iter().any(|k| n.contains(k))
            && norm_head_shape(&h.shape).map_or(false, |(c, _)| c == cells)
    })
}

fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// Pure YuNet postprocessing over already-fetched heads. `in_size` is the
/// square working-image side (640 for the pinned export).
pub fn decode_yunet(heads: &[NamedHead], in_size: usize, min_score: f32) -> Option<YunetFace> {
    const BOX_KEYS: &[&str] = &["bbox", "loc"];
    const CLS_KEYS: &[&str] = &["cls", "conf"];
    const OBJ_KEYS: &[&str] = &["obj", "iou"];
    let mut best: Option<YunetFace> = None;
    for s in YUNET_STRIDES {
        if in_size % s != 0 {
            continue;
        }
        let grid = in_size / s;
        let cells = grid * grid;
        let (Some(bh), Some(ch), Some(oh)) = (
            head_for(heads, BOX_KEYS, cells),
            head_for(heads, CLS_KEYS, cells),
            head_for(heads, OBJ_KEYS, cells),
        ) else {
            continue;
        };
        let (Some((bc, bk)), Some((cc, ck)), Some((oc, ok))) = (
            norm_head_shape(&bh.shape),
            norm_head_shape(&ch.shape),
            norm_head_shape(&oh.shape),
        ) else {
            continue;
        };
        if bc != cells || bk != 4 || cc != cells || ck != 1 || oc != cells || ok != 1 {
            continue;
        }
        if bh.values.len() < cells * 4 || ch.values.len() < cells || oh.values.len() < cells {
            continue;
        }
        let s = s as f32;
        for r in 0..grid {
            for c in 0..grid {
                let i = r * grid + c;
                let score = (clamp01(ch.values[i]) * clamp01(oh.values[i])).sqrt();
                if !score.is_finite() || score < min_score {
                    continue;
                }
                let (dx, dy, dw, dh) = (
                    bh.values[4 * i],
                    bh.values[4 * i + 1],
                    bh.values[4 * i + 2],
                    bh.values[4 * i + 3],
                );
                if ![dx, dy, dw, dh].iter().all(|v| v.is_finite()) {
                    continue;
                }
                // Exact OpenCV FaceDetectorYN decode: the cell is its own prior.
                let w = dw.exp() * s;
                let h = dh.exp() * s;
                if !(w.is_finite() && h.is_finite()) || w <= 0.0 || h <= 0.0 {
                    continue;
                }
                let face = YunetFace {
                    cx: ((c as f32 + dx) * s).clamp(0.0, in_size as f32),
                    cy: ((r as f32 + dy) * s).clamp(0.0, in_size as f32),
                    w,
                    h,
                    score,
                };
                if best.map_or(true, |b| face.score > b.score) {
                    best = Some(face);
                }
            }
        }
    }
    best
}

/// Fit an RGBA frame inside a square BGR tensor, padding right/bottom with zero.
/// Preserve face proportions so wide/portrait clips remain detectable.
pub fn yunet_bgr_input(frame: &RgbaFrame, size: usize) -> Vec<f32> {
    letterbox_input(frame, size, 0.0, false)
}

fn letterbox_input(frame: &RgbaFrame, size: usize, padding: f32, rgb: bool) -> Vec<f32> {
    let (sw, sh) = (frame.width as usize, frame.height as usize);
    let mut out = vec![padding; 3 * size * size];
    if sw == 0 || sh == 0 || size == 0 {
        return out;
    }
    let longest = sw.max(sh);
    let width = (sw * size / longest).max(1);
    let height = (sh * size / longest).max(1);
    for y in 0..height {
        let sy = (y * sh / height).min(sh - 1);
        for x in 0..width {
            let sx = (x * sw / width).min(sw - 1);
            let px = &frame.data[(sy * sw + sx) * 4..(sy * sw + sx) * 4 + 3];
            out[y * size + x] = px[if rgb { 0 } else { 2 }] as f32;
            out[size * size + y * size + x] = px[1] as f32;
            out[2 * size * size + y * size + x] = px[if rgb { 2 } else { 0 }] as f32;
        }
    }
    out
}

impl YunetDetector {
    pub fn new(model_path: &Path, min_score: f32) -> Self {
        Self {
            model_path: model_path.to_path_buf(),
            min_score,
            session: None,
            failure: None,
        }
    }

    fn ensure_session(&mut self) -> Option<&mut ort::session::Session> {
        if self.failure.is_some() {
            return None;
        }
        if self.session.is_none() {
            let loaded = (|| -> Result<ort::session::Session> {
                let env = ae_core::onnx_environment()?;
                ort::session::Session::builder(env)
                    .map_err(|e| AeError::Model {
                        model: self.model_path.display().to_string(),
                        message: e.to_string(),
                    })?
                    .commit_from_file(&self.model_path)
                    .map_err(|e| AeError::Model {
                        model: self.model_path.display().to_string(),
                        message: e.to_string(),
                    })
            })();
            match loaded {
                Ok(s) => self.session = Some(s),
                Err(e) => {
                    tracing::warn!("subject detector unavailable, using centered crop: {e}");
                    self.failure = Some(e.to_string());
                    return None;
                }
            }
        }
        self.session.as_mut()
    }
}

impl YunetDetector {
    fn infer(&mut self, input: Vec<f32>) -> Option<Vec<NamedHead>> {
        let tensor = match ort::value::TensorRef::from_array_view((
            [1usize, 3, YUNET_INPUT, YUNET_INPUT],
            input.as_slice(),
        )) {
            Ok(tensor) => tensor,
            Err(error) => {
                self.failure = Some(format!("Could not create model input: {error}"));
                return None;
            }
        };
        let session = self.ensure_session()?;
        let heads = session.run(ort::inputs![tensor]).and_then(|outputs| {
            outputs
                .iter()
                .map(|(name, value)| {
                    let (shape, values) = value.try_extract_tensor::<f32>()?;
                    Ok(NamedHead {
                        name: name.to_string(),
                        shape: shape.to_vec(),
                        values: values.to_vec(),
                    })
                })
                .collect()
        });
        match heads {
            Ok(heads) => Some(heads),
            Err(error) => {
                tracing::warn!("subject detector inference failed: {error}");
                self.failure = Some(format!("{}: {error}", self.model_path.display()));
                None
            }
        }
    }
}

impl FaceDetector for YunetDetector {
    fn detect(&mut self, frame: &RgbaFrame) -> Option<FaceBox> {
        if frame.width == 0 || frame.height == 0 {
            return None;
        }
        let heads = self.infer(yunet_bgr_input(frame, YUNET_INPUT))?;
        let hit = decode_yunet(&heads, YUNET_INPUT, self.min_score)?;
        let scale = YUNET_INPUT as f32 / frame.width.max(frame.height) as f32;
        Some(FaceBox {
            cx: (hit.cx / scale).clamp(0.0, frame.width as f32),
            cy: (hit.cy / scale).clamp(0.0, frame.height as f32),
            score: hit.score,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharacterDetection {
    pub rect: CropRect,
    pub score: f32,
}

pub struct CharacterAnalysisDetector {
    model: YunetDetector,
    nms_iou: f32,
}

impl CharacterAnalysisDetector {
    pub fn new(model_path: &Path, min_score: f32, nms_iou: f32) -> Self {
        Self {
            model: YunetDetector::new(model_path, min_score),
            nms_iou,
        }
    }

    pub fn detect_all(&mut self, frame: &RgbaFrame) -> Result<Vec<CharacterDetection>> {
        let model_name = self.model.model_path.display().to_string();
        let error = |message: String| AeError::Model {
            model: model_name.clone(),
            message,
        };
        if !(0.0..=1.0).contains(&self.model.min_score) || !(0.0..=1.0).contains(&self.nms_iou) {
            return Err(error("Invalid detection thresholds".into()));
        }
        if frame.width == 0
            || frame.height == 0
            || (frame.width as usize)
                .checked_mul(frame.height as usize)
                .and_then(|n| n.checked_mul(4))
                != Some(frame.data.len())
        {
            return Err(error("Invalid RGBA frame".into()));
        }
        let input = letterbox_input(frame, YUNET_INPUT, 114.0, true);
        let heads = self.model.infer(input).ok_or_else(|| AeError::Model {
            model: self.model.model_path.display().to_string(),
            message: self
                .model
                .failure
                .clone()
                .unwrap_or_else(|| "Inference failed".into()),
        })?;
        let head = heads
            .first()
            .filter(|h| h.shape == [1, 8400, 6] && h.values.len() == 8400 * 6)
            .ok_or_else(|| error("Unexpected character model output shape".into()))?;
        if !head.values.iter().all(|value| value.is_finite()) {
            return Err(error("Non-finite character model output".into()));
        }
        Ok(character_rectangles(
            head,
            frame.width,
            frame.height,
            self.model.min_score,
            self.nms_iou,
        ))
    }
}

fn character_rectangles(
    head: &NamedHead,
    width: u32,
    height: u32,
    min_score: f32,
    nms_iou: f32,
) -> Vec<CharacterDetection> {
    let mut hits = Vec::new();
    let scale = YUNET_INPUT as f32 / width.max(height) as f32;
    for hit in decode_character_boxes(head, min_score) {
        let x1 = ((hit.cx - hit.w / 2.0) / scale)
            .clamp(0.0, width as f32)
            .floor() as u32;
        let y1 = ((hit.cy - hit.h / 2.0) / scale)
            .clamp(0.0, height as f32)
            .floor() as u32;
        let x2 = ((hit.cx + hit.w / 2.0) / scale)
            .clamp(0.0, width as f32)
            .ceil() as u32;
        let y2 = ((hit.cy + hit.h / 2.0) / scale)
            .clamp(0.0, height as f32)
            .ceil() as u32;
        if x2 > x1 && y2 > y1 {
            hits.push(CharacterDetection {
                rect: CropRect {
                    x: x1,
                    y: y1,
                    w: x2 - x1,
                    h: y2 - y1,
                },
                score: hit.score,
            });
        }
    }
    hits.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut retained: Vec<CharacterDetection> = Vec::new();
    for hit in hits {
        if retained.iter().any(|other| {
            let a = hit.rect;
            let b = other.rect;
            let intersection = (a.x + a.w).min(b.x + b.w).saturating_sub(a.x.max(b.x)) as f64
                * (a.y + a.h).min(b.y + b.h).saturating_sub(a.y.max(b.y)) as f64;
            let union = a.w as f64 * a.h as f64 + b.w as f64 * b.h as f64 - intersection;
            intersection / union > nms_iou as f64
        }) {
            continue;
        }
        retained.push(hit);
    }
    retained
}

struct CharacterDetector {
    model: YunetDetector,
    previous: Option<FaceBox>,
}

// YOLOX-s character model: RGB 0..255, right/bottom padding 114, raw [1,8400,6]
// output; decode grid offsets at strides 8/16/32 and score objectness * class.
fn decode_character(
    head: &NamedHead,
    min_score: f32,
    previous: Option<FaceBox>,
) -> Option<FaceBox> {
    let mut best: Option<(f32, FaceBox)> = None;
    for hit in decode_character_boxes(head, min_score) {
        let (cx, cy, score) = (hit.cx, hit.cy, hit.score);
        if !(0.0..=YUNET_INPUT as f32).contains(&cx) || !(0.0..=YUNET_INPUT as f32).contains(&cy) {
            continue;
        }
        let rank = previous.map_or(score, |p| {
            score
                / (1.0
                    + 16.0 * ((cx - p.cx).powi(2) + (cy - p.cy).powi(2))
                        / (YUNET_INPUT * YUNET_INPUT) as f32)
        });
        if best.is_none_or(|(ranked, _)| rank > ranked) {
            best = Some((rank, FaceBox { cx, cy, score }));
        }
    }
    best.map(|(_, hit)| hit)
}

fn decode_character_boxes(head: &NamedHead, min_score: f32) -> Vec<YunetFace> {
    if head.shape != [1, 8400, 6] || head.values.len() != 8400 * 6 {
        return Vec::new();
    }
    let mut hits = Vec::new();
    let mut offset = 0;
    for stride in YUNET_STRIDES {
        let grid = YUNET_INPUT / stride;
        for i in 0..grid * grid {
            let row = &head.values[(offset + i) * 6..(offset + i + 1) * 6];
            if !row.iter().all(|v| v.is_finite())
                || !(0.0..=1.0).contains(&row[4])
                || !(0.0..=1.0).contains(&row[5])
            {
                continue;
            }
            let score = row[4] * row[5];
            let w = row[2].exp() * stride as f32;
            let h = row[3].exp() * stride as f32;
            let cx = (row[0] + (i % grid) as f32) * stride as f32;
            let cy = (row[1] + (i / grid) as f32) * stride as f32;
            if score < min_score
                || !w.is_finite()
                || !h.is_finite()
                || w <= 0.0
                || h <= 0.0
                || !cx.is_finite()
                || !cy.is_finite()
            {
                continue;
            }
            hits.push(YunetFace {
                cx,
                cy,
                w,
                h,
                score,
            });
        }
        offset += grid * grid;
    }
    hits
}

impl FaceDetector for CharacterDetector {
    fn detect(&mut self, frame: &RgbaFrame) -> Option<FaceBox> {
        if frame.width == 0 || frame.height == 0 {
            return None;
        }
        let heads = self
            .model
            .infer(letterbox_input(frame, YUNET_INPUT, 114.0, true))?;
        let Some(head) = heads
            .first()
            .filter(|head| head.shape == [1, 8400, 6] && head.values.len() == 8400 * 6)
        else {
            let message = format!(
                "Unexpected character model output: expected [1, 8400, 6], received {:?}",
                heads
                    .iter()
                    .map(|head| (&head.name, &head.shape))
                    .collect::<Vec<_>>()
            );
            tracing::warn!("{message}");
            self.model.failure = Some(message);
            return None;
        };
        let hit = decode_character(head, self.model.min_score, self.previous)?;
        let scale = YUNET_INPUT as f32 / frame.width.max(frame.height) as f32;
        if hit.cx >= frame.width as f32 * scale || hit.cy >= frame.height as f32 * scale {
            return None;
        }
        self.previous = Some(hit);
        Some(FaceBox {
            cx: hit.cx / scale,
            cy: hit.cy / scale,
            score: hit.score,
        })
    }

    fn reset(&mut self) {
        self.previous = None;
    }

    fn finish_shot(&self) -> Result<()> {
        if let Some(message) = &self.model.failure {
            return Err(AeError::Model {
                model: "animated character tracker".into(),
                message: message.clone(),
            });
        }
        Ok(())
    }
}

/// Cover-fit decode size: smallest resolution with the source aspect that still
/// covers `target`, so decoding (which stretches) introduces no distortion.
pub fn cover_decode_res(src_w: u32, src_h: u32, target: Resolution) -> Resolution {
    let (sw, sh) = (src_w.max(1) as f64, src_h.max(1) as f64);
    let scale = (target.width as f64 / sw).max(target.height as f64 / sh);
    Resolution::new(
        (sw * scale).round().clamp(1.0, 16384.0) as u32,
        (sh * scale).round().clamp(1.0, 16384.0) as u32,
    )
}

/// Per-shot reframing state. Create one per timeline slot and feed frames in
/// presentation order; it tracks the smoothed crop-center target.
pub struct Reframer {
    cfg: ReframeConfig,
    src_w: u32,
    src_h: u32,
    crop_w: u32,
    crop_h: u32,
    cx: f32,
    cy: f32,
    misses: usize,
    initialized: bool,
}

impl Reframer {
    /// `src_w`/`src_h` are the decoded frame size; the crop keeps `target`'s
    /// aspect with a fixed 10% crop margin so matching source/output aspects can pan.
    pub fn new(cfg: &ReframeConfig, src_w: u32, src_h: u32, target: Resolution) -> Self {
        let (sw, sh) = (src_w.max(1), src_h.max(1));
        let target_aspect = target.width as f64 / target.height.max(1) as f64;
        let src_aspect = sw as f64 / sh as f64;
        let (crop_w, crop_h) = if src_aspect > target_aspect {
            let h = sh;
            let w = ((h as f64 * target_aspect).round() as u32).clamp(1, sw);
            (w, h)
        } else {
            let w = sw;
            let h = ((w as f64 / target_aspect).round() as u32).clamp(1, sh);
            (w, h)
        };
        Self {
            cfg: cfg.clone(),
            src_w: sw,
            src_h: sh,
            crop_w: ((crop_w as f64 * 0.9).round() as u32).max(1),
            crop_h: ((crop_h as f64 * 0.9).round() as u32).max(1),
            cx: sw as f32 * 0.5,
            cy: sh as f32 * 0.5,
            misses: 0,
            initialized: false,
        }
    }

    /// Reset at each shot boundary back to the centered crop.
    pub fn reset(&mut self) {
        self.cx = self.src_w as f32 * 0.5;
        self.cy = self.src_h as f32 * 0.5;
        self.misses = 0;
        self.initialized = false;
    }

    /// Advance one frame toward the detected face center (`None` = dropout)
    /// and return the clamped crop rectangle.
    pub fn update(&mut self, face: Option<FaceBox>) -> CropRect {
        let (mut tx, mut ty) = (self.src_w as f32 * 0.5, self.src_h as f32 * 0.5);
        match face {
            Some(f) if f.score >= self.cfg.min_score => {
                self.misses = 0;
                tx = f.cx;
                ty = f.cy;
            }
            _ => {
                self.misses = self.misses.saturating_add(1);
                if self.misses <= self.cfg.dropout_hold_frames.max(1) && self.initialized {
                    // Hold the last smoothed position through brief dropouts.
                    tx = self.cx;
                    ty = self.cy;
                }
                // Otherwise drift back to the centered fallback.
            }
        }

        if !self.initialized {
            // First frame snaps (clamped) so reframes never start off-screen.
            self.initialized = true;
            if face.is_some() {
                self.cx = tx;
                self.cy = ty;
            }
        } else {
            let dx = tx - self.cx;
            let dy = ty - self.cy;
            let dead = self.cfg.dead_zone_frac.max(0.0) * self.crop_w as f32;
            let max_pan = self.cfg.max_pan_frac.max(0.0) * self.crop_w as f32;
            let dist = dx.hypot(dy);
            if dist > dead {
                let alpha = self.cfg.ema.clamp(0.0, 1.0);
                let mut step = dist * alpha;
                if max_pan > 0.0 {
                    step = step.min(max_pan);
                }
                self.cx += dx / dist * step;
                self.cy += dy / dist * step;
            }
        }
        self.rect()
    }

    pub fn rect(&self) -> CropRect {
        // Keep the framing wide: reaching an edge must never trigger a tighter zoom.
        clamp_crop(
            self.cx,
            self.cy,
            self.crop_w,
            self.crop_h,
            self.src_w,
            self.src_h,
        )
    }
}

/// Clamp a center-based crop window fully inside the source frame.
pub fn clamp_crop(cx: f32, cy: f32, w: u32, h: u32, src_w: u32, src_h: u32) -> CropRect {
    let w = w.min(src_w.max(1));
    let h = h.min(src_h.max(1));
    let x = (cx - w as f32 * 0.5).round().clamp(0.0, (src_w - w) as f32) as u32;
    let y = (cy - h as f32 * 0.5).round().clamp(0.0, (src_h - h) as f32) as u32;
    CropRect { x, y, w, h }
}

/// Crop `frame` to `rect` (pts preserved).
pub fn crop_frame(frame: &RgbaFrame, rect: CropRect) -> RgbaFrame {
    let mut out = RgbaFrame::new(rect.w, rect.h, frame.pts);
    for y in 0..rect.h as usize {
        let sy = (rect.y as usize + y).min(frame.height as usize - 1);
        for x in 0..rect.w as usize {
            let sx = (rect.x as usize + x).min(frame.width as usize - 1);
            let src = &frame.data[(sy * frame.width as usize + sx) * 4..][..4];
            out.data[(y * rect.w as usize + x) * 4..][..4].copy_from_slice(src);
        }
    }
    out
}

/// Crop then resize to `target`. The crop already matches the target aspect,
/// so unlike the default stretch path this never distorts the image.
pub fn reframe_frame(frame: &RgbaFrame, rect: CropRect, target: Resolution) -> Result<RgbaFrame> {
    let cropped = crop_frame(frame, rect);
    stretch_frame_lanczos(&cropped, target)
}

/// Open the GUI's full-character model (`anime_character.onnx`), or YuNet
/// for other model paths. A missing file retains the centered CLI fallback.
pub fn open_detector(cfg: &ReframeConfig) -> Box<dyn FaceDetector> {
    if let Some(p) = &cfg.model_path
        && Path::new(p).exists()
    {
        let model = YunetDetector::new(p, cfg.min_score);
        if p.file_name()
            .is_some_and(|name| name == "anime_character.onnx")
        {
            return Box::new(CharacterDetector {
                model,
                previous: None,
            });
        }
        return Box::new(model);
    }
    if cfg.model_path.is_some() {
        tracing::warn!("face model not found, using centered crop fallback");
    }
    Box::new(CenterFallback)
}

/// Detect, advance the per-shot smoother, and crop+resize one frame. Shared by
/// the final renderer and the incremental preview so both reframe identically.
pub fn reframe_detected_frame(
    detector: &mut dyn FaceDetector,
    reframer: &mut Reframer,
    frame: &RgbaFrame,
    target: Resolution,
) -> Result<RgbaFrame> {
    if !reframer.initialized {
        detector.reset();
    }
    let face = detector.detect(frame);
    let rect = reframer.update(face);
    reframe_frame(frame, rect, target)
}

/// Stable cache key for the reframe configuration. `"reframe-off"` when
/// disabled; otherwise the model file's content hash, or `"reframe-on:v5:no-model"`
/// when absent (which still reframes via the centered fallback). Any toggle or
/// model change alters the key so incremental caches never reuse stale segments.
pub fn reframe_fingerprint(enabled: bool, model_path: Option<&Path>) -> String {
    if !enabled {
        return "reframe-off".to_string();
    }
    match model_path.map(std::fs::read) {
        Some(Ok(bytes)) => {
            use sha2::Digest;
            format!(
                "reframe-on:v5:{}",
                hex::encode(sha2::Sha256::digest(&bytes))
            )
        }
        _ => "reframe-on:v5:no-model".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ae_core::Rational;

    fn cfg() -> ReframeConfig {
        ReframeConfig {
            enabled: true,
            ema: 0.5,
            dead_zone_frac: 0.01,
            max_pan_frac: 1.0, // uncapped unless a test lowers it
            dropout_hold_frames: 3,
            min_score: 0.5,
            ..Default::default()
        }
    }

    #[test]
    fn yunet_input_preserves_aspect_and_pads_instead_of_stretching_faces() {
        for (width, height) in [(4, 2), (2, 4)] {
            let frame = RgbaFrame::solid(width, height, [10, 20, 30, 255], Rational::ZERO);
            let input = yunet_bgr_input(&frame, 4);
            for y in 0..4 {
                for x in 0..4 {
                    for (channel, value) in [30.0, 20.0, 10.0].iter().enumerate() {
                        let expected = if x < width as usize && y < height as usize {
                            *value
                        } else {
                            0.0
                        };
                        assert_eq!(input[channel * 16 + y * 4 + x], expected);
                    }
                }
            }
        }
    }

    fn face(cx: f32, cy: f32) -> Option<FaceBox> {
        Some(FaceBox { cx, cy, score: 0.9 })
    }

    #[test]
    fn tracking_keeps_a_small_constant_crop_at_matching_aspect_and_near_edges() {
        let mut rf = Reframer::new(
            &ReframeConfig::default(),
            640,
            480,
            Resolution::new(640, 480),
        );
        let first = rf.update(face(320.0, 240.0));
        assert_eq!((first.w, first.h), (576, 432));
        let moved = rf.update(face(332.0, 252.0));
        assert_eq!((moved.x - first.x, moved.y - first.y), (12, 12));
        assert_eq!((moved.x + moved.w / 2, moved.y + moved.h / 2), (332, 252));
        for (cx, cy) in [(304.0, 224.0), (600.0, 420.0), (20.0, 30.0), (0.0, 480.0)] {
            let rect = rf.update(face(cx, cy));
            assert_eq!(
                (rect.w, rect.h),
                (576, 432),
                "tracking must not zoom tighter near edges"
            );
            assert_eq!(rect, clamp_crop(cx, cy, 576, 432, 640, 480));
            assert!(rect.x + rect.w <= 640 && rect.y + rect.h <= 480);
        }
        let held = rf.rect();
        for _ in 0..60 {
            assert_eq!(rf.update(None), held, "occlusion must preserve framing");
        }
    }

    #[test]
    #[ignore = "requires the locally cached anime_character.onnx model"]
    fn character_model_runs_multiple_sessions_with_one_environment() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/anime_character.onnx");
        assert!(
            path.is_file(),
            "cache the character model before running this test"
        );
        let environment = ae_core::onnx_environment().unwrap();
        for _ in 0..2 {
            let mut model = YunetDetector::new(&path, 0.5);
            let heads = model
                .infer(vec![114.0; 3 * YUNET_INPUT * YUNET_INPUT])
                .unwrap_or_else(|| panic!("character model failed: {:?}", model.failure));
            assert_eq!(heads[0].shape, [1, 8400, 6]);
            assert!(std::ptr::eq(
                environment,
                ae_core::onnx_environment().unwrap()
            ));
        }
    }

    #[test]
    fn character_tracking_allows_empty_shots_but_reports_model_failures() {
        let mut detector = CharacterDetector {
            model: YunetDetector::new(Path::new("unused.onnx"), 0.5),
            previous: None,
        };
        assert!(detector.finish_shot().is_ok());
        detector.previous = face(200.0, 240.0);
        assert!(detector.finish_shot().is_ok());
        detector.reset();
        assert!(detector.previous.is_none());
        assert!(detector.finish_shot().is_ok());
        detector.model.failure = Some("specific inference failure".into());
        assert!(
            detector
                .finish_shot()
                .unwrap_err()
                .to_string()
                .contains("specific inference failure")
        );
    }

    #[test]
    fn character_input_is_rgb_with_yolox_padding() {
        let frame = RgbaFrame::solid(2, 1, [10, 20, 30, 255], Rational::ZERO);
        let input = letterbox_input(&frame, 2, 114.0, true);
        assert_eq!(
            input,
            [
                10., 10., 114., 114., 20., 20., 114., 114., 30., 30., 114., 114.
            ]
        );
    }

    #[test]
    fn all_character_boxes_suppress_duplicates_and_clip() {
        let mut head = NamedHead {
            name: "output".into(),
            shape: vec![1, 8400, 6],
            values: vec![0.0; 8400 * 6],
        };
        let a = (10 * 80 + 10) * 6;
        let b = (10 * 80 + 11) * 6;
        let c = (10 * 80 + 50) * 6;
        head.values[a..a + 6].copy_from_slice(&[0., 0., 3., 3., 0.9, 1.]);
        head.values[b..b + 6].copy_from_slice(&[-1., 0., 3., 3., 0.8, 1.]);
        head.values[c..c + 6].copy_from_slice(&[0., 0., 3., 3., 0.7, 1.]);
        let hits = character_rectangles(&head, 640, 320, 0.5, 0.5);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].rect.x, 0);
    }

    #[test]
    fn character_analysis_rejects_invalid_inputs() {
        let mut detector = CharacterAnalysisDetector::new(Path::new("missing"), f32::NAN, 0.5);
        let frame = RgbaFrame::new(0, 0, ae_core::Rational::new(0, 1));
        assert!(detector.detect_all(&frame).is_err());
        detector.model.min_score = 0.5;
        assert!(detector.detect_all(&frame).is_err());
        let frame = RgbaFrame::solid(2, 2, [0; 4], ae_core::Rational::ZERO);
        assert!(
            detector.detect_all(&frame).is_err(),
            "missing model must not mean no character"
        );
    }

    #[test]
    fn character_boxes_reject_padding_and_invalid_rows() {
        let mut head = NamedHead {
            name: "output".into(),
            shape: vec![1, 8400, 6],
            values: vec![0.0; 8400 * 6],
        };
        let offset = (70 * 80 + 10) * 6;
        head.values[offset..offset + 6].copy_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.9, 1.0]);
        assert!(character_rectangles(&head, 640, 320, 0.5, 0.5).is_empty());
        assert!(!character_rectangles(&head, 320, 640, 0.5, 0.5).is_empty());
        head.values[offset] = f32::NAN;
        assert!(decode_character_boxes(&head, 0.5).is_empty());
        head.shape = vec![1, 1, 6];
        assert!(decode_character_boxes(&head, 0.5).is_empty());
    }

    #[test]
    fn character_boxes_need_no_face_and_keep_the_existing_subject() {
        let mut head = NamedHead {
            name: "output".into(),
            shape: vec![1, 8400, 6],
            values: vec![0.0; 8400 * 6],
        };
        // Full-body boxes only: the detector does not require facial landmarks.
        let left = (35 * 80 + 25) * 6;
        let right = (35 * 80 + 60) * 6;
        head.values[left..left + 6].copy_from_slice(&[0., 0., 2., 3., 0.95, 0.95]);
        head.values[right..right + 6].copy_from_slice(&[0., 0., 2., 3., 0.9, 0.9]);
        let first = decode_character(&head, 0.5, None).unwrap();
        assert_eq!((first.cx, first.cy), (200., 280.));
        head.values[left + 4] = 0.7;
        head.values[right + 4] = 1.0;
        head.values[right + 5] = 1.0;
        let next = decode_character(&head, 0.5, Some(first)).unwrap();
        assert_eq!(
            next.cx, first.cx,
            "confidence changes must not switch characters"
        );
        head.values[left + 4] = f32::NAN;
        head.values[right + 4] = -1.0;
        assert!(decode_character(&head, 0.5, Some(first)).is_none());
        let coarse = (6400 + 10 * 40 + 12) * 6;
        head.values[coarse..coarse + 6].copy_from_slice(&[0.25, 0.5, 2., 3., 0.95, 0.95]);
        let hit = decode_character(&head, 0.5, None).unwrap();
        assert_eq!((hit.cx, hit.cy), (196., 168.));
        head.values[coarse + 2] = -1000.0;
        assert!(decode_character(&head, 0.5, None).is_none());
        head.shape[2] = 5;
        assert!(decode_character(&head, 0.5, None).is_none());
    }

    #[test]
    fn crop_clamped_to_source_bounds() {
        let r = clamp_crop(-50.0, 10_000.0, 200, 100, 640, 480);
        assert_eq!(
            r,
            CropRect {
                x: 0,
                y: 380,
                w: 200,
                h: 100
            }
        );
        // Crop larger than the source collapses to the full frame.
        let r = clamp_crop(320.0, 240.0, 9999, 9999, 640, 480);
        assert_eq!(
            r,
            CropRect {
                x: 0,
                y: 0,
                w: 640,
                h: 480
            }
        );
    }

    #[test]
    fn ema_moves_partway_and_dead_zone_holds() {
        let mut rf = Reframer::new(&cfg(), 640, 480, Resolution::new(320, 240));
        // Matching aspect still reserves zoom margin for tracking.
        let r0 = rf.update(face(320.0, 240.0));
        assert_eq!((r0.x, r0.y), (32, 24));
        // Portrait target => narrow crop that can actually pan.
        let mut rf = Reframer::new(&cfg(), 640, 480, Resolution::new(240, 320));
        assert_eq!((rf.crop_w, rf.crop_h), (324, 432));
        rf.update(face(320.0, 240.0));
        assert_eq!(rf.rect().x, 158);
        // Small move inside the dead zone (0.01*324 = 3.24px) holds position.
        rf.update(face(322.0, 240.0));
        assert_eq!(rf.rect().x, 158);
        // Large move advances only partway (EMA alpha 0.5, uncapped pan).
        let before = rf.cx;
        rf.update(face(500.0, 240.0));
        let expected = before + (500.0 - before) * 0.5;
        assert!(
            (rf.cx - expected).abs() < 1e-4,
            "cx={} want {expected}",
            rf.cx
        );
    }

    #[test]
    fn max_pan_speed_limits_jump() {
        let mut c = cfg();
        c.max_pan_frac = 0.05; // 5% of 324px crop = 16.2px/frame
        c.ema = 1.0;
        let mut rf = Reframer::new(&c, 640, 480, Resolution::new(240, 320));
        rf.update(face(320.0, 240.0));
        let before = rf.cx;
        rf.update(face(600.0, 240.0));
        assert!((rf.cx - before - 16.2).abs() < 1e-4, "cx={}", rf.cx);
    }

    #[test]
    fn dropout_hold_then_fallback_to_center() {
        let mut rf = Reframer::new(&cfg(), 640, 480, Resolution::new(240, 320));
        rf.update(face(500.0, 240.0));
        let held = rf.cx;
        // Within the 3-frame hold the position freezes.
        for _ in 0..3 {
            rf.update(None);
            assert!((rf.cx - held).abs() < 1e-6);
        }
        // After the hold it drifts back toward center (320).
        rf.update(None);
        assert!(rf.cx < held && rf.cx > 320.0, "cx={}", rf.cx);
    }

    #[test]
    fn reset_returns_to_center() {
        let mut rf = Reframer::new(&cfg(), 640, 480, Resolution::new(240, 320));
        rf.update(face(600.0, 400.0));
        assert_ne!((rf.cx, rf.cy), (320.0, 240.0));
        rf.reset();
        assert_eq!(
            rf.rect(),
            CropRect {
                x: 158,
                y: 24,
                w: 324,
                h: 432
            }
        );
    }

    #[test]
    fn low_score_detections_are_ignored() {
        let mut rf = Reframer::new(&cfg(), 640, 480, Resolution::new(240, 320));
        rf.update(face(320.0, 240.0));
        let r = rf.update(Some(FaceBox {
            cx: 600.0,
            cy: 400.0,
            score: 0.1,
        }));
        assert_eq!(r.x, 158, "below-threshold face must not move the crop");
    }

    #[test]
    fn cover_decode_preserves_aspect_and_covers_target() {
        // 16:9 source -> 9:16 target: height matches, width over-covers.
        let d = cover_decode_res(1280, 720, Resolution::new(720, 1280));
        assert_eq!(d.height, 1280);
        assert!(d.width >= 720);
        assert!((d.width as f64 / d.height as f64 - 1280.0 / 720.0).abs() < 2.0 / 1280.0);
        // Same aspect => identical size.
        let d = cover_decode_res(640, 480, Resolution::new(320, 240));
        assert_eq!((d.width, d.height), (320, 240));
    }

    #[test]
    fn crop_then_resize_keeps_aspect_without_stretch() {
        // Wide gradient frame reframed to portrait: left/right columns must differ.
        let mut data = vec![0u8; 640 * 480 * 4];
        for y in 0..480 {
            for x in 0..640 {
                let i = (y * 640 + x) * 4;
                data[i] = (x * 255 / 639) as u8;
                data[i + 1] = 40;
                data[i + 2] = 200;
                data[i + 3] = 255;
            }
        }
        let frame = RgbaFrame::from_rgba(640, 480, data, Rational::from_millis(7));
        let target = Resolution::new(240, 320);
        let mut rf = Reframer::new(&cfg(), 640, 480, target);
        let rect = rf.update(face(200.0, 240.0)); // look left
        assert_eq!((rect.w, rect.h), (324, 432));
        let out = reframe_frame(&frame, rect, target).unwrap();
        assert_eq!((out.width, out.height), (240, 320));
        assert_eq!(out.pts, frame.pts);
        // Looking left: left edge dark, right edge of the crop mid-grey.
        assert!(out.pixel(5, 160)[0] < 60, "left edge should be dark");
        assert!(
            out.pixel(234, 160)[0] > 100,
            "crop must follow the face side"
        );
    }

    fn yunet_heads_for_stride(
        stride: usize,
        in_size: usize,
        fill: impl Fn(usize, usize) -> (f32, f32, [f32; 4]),
    ) -> Vec<NamedHead> {
        let grid = in_size / stride;
        let cells = grid * grid;
        let mut cls = vec![0.0f32; cells];
        let mut obj = vec![0.0f32; cells];
        let mut bbox = vec![0.0f32; cells * 4];
        for r in 0..grid {
            for c in 0..grid {
                let (cs, os, b) = fill(c, r);
                cls[r * grid + c] = cs;
                obj[r * grid + c] = os;
                bbox[(r * grid + c) * 4..(r * grid + c) * 4 + 4].copy_from_slice(&b);
            }
        }
        vec![
            NamedHead {
                name: format!("bbox_{stride}"),
                shape: vec![1, cells as i64, 4],
                values: bbox,
            },
            NamedHead {
                name: format!("cls_{stride}"),
                shape: vec![1, cells as i64, 1],
                values: cls,
            },
            NamedHead {
                name: format!("obj_{stride}"),
                shape: vec![1, cells as i64, 1],
                values: obj,
            },
        ]
    }

    #[test]
    fn yunet_input_is_bgr_raw_float() {
        // 2x1 frame: left pixel R=10,G=20,B=30; right pixel R=40,G=50,B=60.
        // Fit to 2x2 preserves the row and pads below it.
        let data = vec![10u8, 20, 30, 255, 40, 50, 60, 255];
        let frame = RgbaFrame::from_rgba(2, 1, data, Rational::ZERO);
        let v = yunet_bgr_input(&frame, 2);
        assert_eq!(v.len(), 3 * 2 * 2);
        assert_eq!(&v[0..4], &[30.0, 60.0, 0.0, 0.0]); // B plane, raw scale
        assert_eq!(&v[4..8], &[20.0, 50.0, 0.0, 0.0]); // G plane
        assert_eq!(&v[8..12], &[10.0, 40.0, 0.0, 0.0]); // R plane
    }

    #[test]
    fn yunet_decode_matches_opencv_math() {
        // Stride-8 cell (c=5, r=6), in_size=64 => grid 8x8.
        // cls=0.81, obj=0.64 => score = sqrt(0.5184) = 0.72.
        // dx=0.5, dy=-0.25 => cx=(5.5)*8=44, cy=(5.75)*8=46.
        // dw=ln3, dh=-ln2 => w=3*8=24, h=0.5*8=4.
        let heads = yunet_heads_for_stride(8, 64, |c, r| {
            if (c, r) == (5, 6) {
                (0.81, 0.64, [0.5, -0.25, 3.0f32.ln(), -(2.0f32.ln())])
            } else {
                (0.0, 0.0, [0.0; 4])
            }
        });
        let hit = decode_yunet(&heads, 64, 0.5).expect("face");
        assert!((hit.score - 0.72).abs() < 1e-5, "score={}", hit.score);
        assert!((hit.cx - 44.0).abs() < 1e-4, "cx={}", hit.cx);
        assert!((hit.cy - 46.0).abs() < 1e-4, "cy={}", hit.cy);
        assert!((hit.w - 24.0).abs() < 1e-4, "w={}", hit.w);
        assert!((hit.h - 4.0).abs() < 1e-4, "h={}", hit.h);
    }

    #[test]
    fn yunet_score_clamps_and_thresholds() {
        // cls > 1 clamps to 1: sqrt(1.0 * 0.5) ~= 0.7071.
        let heads = yunet_heads_for_stride(32, 64, |_, _| (2.0, 0.5, [0.0; 4]));
        let hit = decode_yunet(&heads, 64, 0.5).expect("face");
        assert!(
            (hit.score - 0.5f32.sqrt()).abs() < 1e-5,
            "score={}",
            hit.score
        );
        // Negative obj clamps to 0 => score 0 => rejected by any positive threshold.
        let heads = yunet_heads_for_stride(32, 64, |_, _| (0.9, -0.5, [0.0; 4]));
        assert!(decode_yunet(&heads, 64, 0.1).is_none());
        // Below-threshold scores are rejected even when finite.
        let heads = yunet_heads_for_stride(32, 64, |_, _| (0.36, 0.25, [0.0; 4]));
        assert!(decode_yunet(&heads, 64, 0.5).is_none()); // sqrt(0.09)=0.3 < 0.5
        assert!(decode_yunet(&heads, 64, 0.3).is_some());
        // NaN offsets never produce a face.
        let heads = yunet_heads_for_stride(32, 64, |_, _| (0.9, 0.9, [f32::NAN, 0.0, 0.0, 0.0]));
        assert!(decode_yunet(&heads, 64, 0.1).is_none());
    }

    #[test]
    fn yunet_matches_heads_by_cell_count_not_suffix() {
        // Plain loc/conf/iou names with stride-16 cell counts decode at s=16.
        let grid = 4usize; // in_size=64, stride 16
        let cells = grid * grid;
        let mut cls = vec![0.0f32; cells];
        let mut obj = vec![0.0f32; cells];
        let loc = vec![0.0f32; cells * 4];
        cls[0] = 0.9;
        obj[0] = 0.9;
        let heads = vec![
            NamedHead {
                name: "loc".into(),
                shape: vec![cells as i64, 4],
                values: loc,
            },
            NamedHead {
                name: "conf".into(),
                shape: vec![1, cells as i64, 1],
                values: cls,
            },
            NamedHead {
                name: "iou".into(),
                shape: vec![1, cells as i64, 1],
                values: obj,
            },
        ];
        let hit = decode_yunet(&heads, 64, 0.5).expect("face");
        // Cell (0,0): cx = 0*16 = 0, w = exp(0)*16 = 16.
        assert!((hit.cx - 0.0).abs() < 1e-5);
        assert!((hit.w - 16.0).abs() < 1e-5);
        assert!((hit.score - 0.9).abs() < 1e-5);
    }

    #[test]
    fn yunet_argmax_wins_and_center_clamps_to_bounds() {
        // Two hot cells across strides; the hotter one wins.
        let mut heads = yunet_heads_for_stride(8, 64, |c, r| {
            if (c, r) == (0, 0) {
                (0.7, 0.7, [-10.0, 0.0, 0.0, 0.0]) // cx = -80 -> clamped to 0
            } else {
                (0.0, 0.0, [0.0; 4])
            }
        });
        heads.extend(yunet_heads_for_stride(32, 64, |c, r| {
            if (c, r) == (1, 1) {
                (0.95, 0.95, [0.0; 4]) // score 0.95 > 0.7
            } else {
                (0.0, 0.0, [0.0; 4])
            }
        }));
        let hit = decode_yunet(&heads, 64, 0.5).expect("face");
        assert!((hit.score - 0.95).abs() < 1e-5, "argmax must win");
        assert!((hit.cx - 32.0).abs() < 1e-5, "stride-32 cell (1,1): cx=32");

        let heads = yunet_heads_for_stride(8, 64, |c, r| {
            if (c, r) == (0, 0) {
                (0.7, 0.7, [-10.0, 0.0, 0.0, 0.0])
            } else {
                (0.0, 0.0, [0.0; 4])
            }
        });
        let hit = decode_yunet(&heads, 64, 0.5).expect("face");
        assert_eq!(hit.cx, 0.0, "negative center clamps to 0");
    }

    #[test]
    fn yunet_shape_mismatch_returns_none() {
        // Box head with wrong channel count or missing heads => no face.
        let bad = vec![
            NamedHead {
                name: "bbox_8".into(),
                shape: vec![1, 6400, 3],
                values: vec![0.0; 6400 * 3],
            },
            NamedHead {
                name: "cls_8".into(),
                shape: vec![1, 6400, 1],
                values: vec![1.0; 6400],
            },
            NamedHead {
                name: "obj_8".into(),
                shape: vec![1, 6400, 1],
                values: vec![1.0; 6400],
            },
        ];
        assert!(decode_yunet(&bad, 640, 0.5).is_none());
        assert!(decode_yunet(&[], 640, 0.5).is_none());
    }

    #[test]
    fn missing_model_opens_centered_fallback() {
        let c = ReframeConfig {
            model_path: Some(PathBuf::from("/nonexistent/yunet.onnx")),
            ..cfg()
        };
        let mut d = open_detector(&c);
        let frame = RgbaFrame::solid(64, 64, [9, 9, 9, 255], Rational::ZERO);
        assert!(d.detect(&frame).is_none());
    }

    #[test]
    fn fingerprint_distinguishes_toggle_and_model_bytes() {
        // Disabled is a stable constant regardless of any path.
        assert_eq!(reframe_fingerprint(false, None), "reframe-off");
        assert_eq!(
            reframe_fingerprint(false, Some(Path::new("models/yunet.onnx"))),
            "reframe-off"
        );
        // Enabled without a readable model is distinct from disabled.
        let missing = reframe_fingerprint(true, Some(Path::new("/nonexistent/yunet.onnx")));
        assert_eq!(missing, "reframe-on:v5:no-model");
        assert_ne!(missing, reframe_fingerprint(false, None));
        // Enabled with a model hashes content: different bytes => different key.
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.onnx");
        let b = dir.path().join("b.onnx");
        std::fs::write(&a, b"fake-model-v1").unwrap();
        std::fs::write(&b, b"fake-model-v2").unwrap();
        let fa = reframe_fingerprint(true, Some(&a));
        let fb = reframe_fingerprint(true, Some(&b));
        assert!(fa.starts_with("reframe-on:"), "{fa}");
        assert_ne!(fa, fb, "model change must invalidate preview caches");
        assert_eq!(
            fa,
            reframe_fingerprint(true, Some(&a)),
            "same bytes => same key"
        );
    }

    #[test]
    fn shared_frame_helper_matches_manual_detect_update_reframe() {
        // A stub detector returning a fixed face must drive the helper exactly
        // like the manual detect/update/reframe sequence used before.
        struct Fixed(Option<FaceBox>);
        impl FaceDetector for Fixed {
            fn detect(&mut self, _frame: &RgbaFrame) -> Option<FaceBox> {
                self.0
            }
        }
        let frame = RgbaFrame::solid(640, 480, [11, 22, 33, 255], Rational::ZERO);
        let target = Resolution::new(240, 320);
        let face = Some(FaceBox {
            cx: 100.0,
            cy: 240.0,
            score: 0.9,
        });

        let mut r1 = Reframer::new(&cfg(), 640, 480, target);
        let f1 = face;
        let rect = r1.update(f1);
        let want = reframe_frame(&frame, rect, target).unwrap();

        let mut det = Fixed(face);
        let mut r2 = Reframer::new(&cfg(), 640, 480, target);
        let got = reframe_detected_frame(&mut det, &mut r2, &frame, target).unwrap();
        assert_eq!(got, want);
        assert_eq!((r1.cx, r1.cy), (r2.cx, r2.cy));
    }
}
