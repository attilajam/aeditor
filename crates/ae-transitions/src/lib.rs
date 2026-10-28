use std::path::Path;

use ae_core::{
    AeError, FrameMut, FrameRef, LinearRgbaFrame, Rational, Result, TransitionSpec,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use wide::f32x4;

#[derive(Clone, Debug)]
pub struct TransitionCtx {
    pub width: u32,
    pub height: u32,
    pub fps: Rational,
    pub duration: Rational,
    pub easing: String,
    pub beat_strength: f32,
}

impl Default for TransitionCtx {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            fps: Rational::from_secs(30),
            duration: Rational::from_millis(200),
            easing: "cubic_in_out".to_string(),
            beat_strength: 0.8,
        }
    }
}

pub trait Transition: Send + Sync {
    fn id(&self) -> &'static str;

    /// Frames needed before and after the cut point (the sequencer reserves these).
    fn handles(&self, ctx: &TransitionCtx) -> (u32, u32) {
        if self.id() == "cut" {
            return (0, 0);
        }
        let total = ((ctx.duration.as_secs_f64() * ctx.fps.as_secs_f64()).round() as u32).max(2);
        let pre = total / 2;
        let post = total - pre;
        (pre, post)
    }

    fn render(
        &self,
        ctx: &TransitionCtx,
        from: FrameRef<'_>,
        to: FrameRef<'_>,
        progress: f32,
        out: FrameMut<'_>,
    );
}

pub struct TransitionFactory {
    pub id: &'static str,
    pub description: &'static str,
    pub build: fn(&toml::Value) -> Result<Box<dyn Transition>>,
}

inventory::collect!(TransitionFactory);

/// Apply named easing function (`simple-easing` 1.1.1) while preserving exact endpoints `0.0` and `1.0`.
#[inline]
pub fn apply_easing(name: &str, t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    match name.trim().to_ascii_lowercase().as_str() {
        "linear" => simple_easing::linear(t),
        "quad_in" => simple_easing::quad_in(t),
        "quad_out" => simple_easing::quad_out(t),
        "quad_in_out" => simple_easing::quad_in_out(t),
        "cubic_in" => simple_easing::cubic_in(t),
        "cubic_out" => simple_easing::cubic_out(t),
        "cubic_in_out" | "smooth" => simple_easing::cubic_in_out(t),
        "expo_in" => simple_easing::expo_in(t),
        "expo_out" => simple_easing::expo_out(t),
        "expo_in_out" => simple_easing::expo_in_out(t),
        "sine_in_out" => simple_easing::sine_in_out(t),
        "back_out" => simple_easing::back_out(t).clamp(0.0, 1.0),
        _ => simple_easing::cubic_in_out(t),
    }
}

/// Look up and instantiate a registered transition by ID.
pub fn build_transition(id: &str, params: &toml::Value) -> Result<Box<dyn Transition>> {
    for factory in inventory::iter::<TransitionFactory> {
        if factory.id.eq_ignore_ascii_case(id) {
            return (factory.build)(params);
        }
    }
    Err(AeError::Transition {
        id: id.to_string(),
        message: format!(
            "unknown transition '{id}'. Available built-ins: {}",
            list_registered_transitions()
                .into_iter()
                .map(|(k, _)| k)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    })
}

pub fn list_registered_transitions() -> Vec<(&'static str, &'static str)> {
    let mut v: Vec<(&'static str, &'static str)> = inventory::iter::<TransitionFactory>
        .into_iter()
        .map(|f| (f.id, f.description))
        .collect();
    v.sort_by_key(|(id, _)| *id);
    v
}

// ============================================================================
// 1. Hard Cut ("cut")
// ============================================================================
pub struct HardCut;

impl HardCut {
    pub fn from_params(_: &toml::Value) -> Result<Box<dyn Transition>> {
        Ok(Box::new(Self))
    }
}

impl Transition for HardCut {
    fn id(&self) -> &'static str {
        "cut"
    }

    fn handles(&self, _ctx: &TransitionCtx) -> (u32, u32) {
        (0, 0)
    }

    fn render(
        &self,
        _ctx: &TransitionCtx,
        from: FrameRef<'_>,
        to: FrameRef<'_>,
        progress: f32,
        mut out: FrameMut<'_>,
    ) {
        if progress < 0.5 {
            out.copy_from_ref(from);
        } else {
            out.copy_from_ref(to);
        }
    }
}

inventory::submit! {
    TransitionFactory {
        id: "cut",
        description: "Instantaneous frame-accurate hard cut (match cut)",
        build: HardCut::from_params,
    }
}

// ============================================================================
// 2. Linear-Light Crossfade ("crossfade")
// ============================================================================
pub struct Crossfade;

impl Crossfade {
    pub fn from_params(_: &toml::Value) -> Result<Box<dyn Transition>> {
        Ok(Box::new(Self))
    }
}

impl Transition for Crossfade {
    fn id(&self) -> &'static str {
        "crossfade"
    }

    fn render(
        &self,
        ctx: &TransitionCtx,
        from: FrameRef<'_>,
        to: FrameRef<'_>,
        progress: f32,
        mut out: FrameMut<'_>,
    ) {
        if progress <= 0.0 {
            out.copy_from_ref(from);
            return;
        }
        if progress >= 1.0 {
            out.copy_from_ref(to);
            return;
        }
        let alpha = apply_easing(&ctx.easing, progress);
        let inv_a = f32x4::splat(1.0 - alpha);
        let a_vec = f32x4::splat(alpha);

        let lin_from = from.to_linear();
        let lin_to = to.to_linear();
        let mut lin_out = LinearRgbaFrame::new(from.width, from.height, from.pts);

        lin_out
            .data
            .par_iter_mut()
            .zip(lin_from.data.par_iter().zip(lin_to.data.par_iter()))
            .for_each(|(dst, (&p0, &p1))| {
                let v0 = f32x4::from(p0);
                let v1 = f32x4::from(p1);
                *dst = (v0 * inv_a + v1 * a_vec).to_array();
            });

        out.write_linear(&lin_out);
    }
}

inventory::submit! {
    TransitionFactory {
        id: "crossfade",
        description: "Linear-light BT.709 premultiplied RGBA crossfade",
        build: Crossfade::from_params,
    }
}

// ============================================================================
// 3. Dip to Black / White ("dip")
// ============================================================================
pub struct DipTransition {
    pub color_linear: [f32; 4],
}

impl DipTransition {
    pub fn from_params(params: &toml::Value) -> Result<Box<dyn Transition>> {
        let mode = params
            .get("color")
            .and_then(|v| v.as_str())
            .unwrap_or("black");
        let color_linear = match mode {
            "white" => [1.0, 1.0, 1.0, 1.0],
            _ => [0.0, 0.0, 0.0, 1.0],
        };
        Ok(Box::new(Self { color_linear }))
    }
}

impl Transition for DipTransition {
    fn id(&self) -> &'static str {
        "dip"
    }

    fn render(
        &self,
        ctx: &TransitionCtx,
        from: FrameRef<'_>,
        to: FrameRef<'_>,
        progress: f32,
        mut out: FrameMut<'_>,
    ) {
        if progress <= 0.0 {
            out.copy_from_ref(from);
            return;
        }
        if progress >= 1.0 {
            out.copy_from_ref(to);
            return;
        }
        let t = apply_easing(&ctx.easing, progress);
        let c = self.color_linear;
        let mut lin_out = LinearRgbaFrame::new(from.width, from.height, from.pts);

        if t < 0.5 {
            let k = t * 2.0;
            let lin_from = from.to_linear();
            lin_out
                .data
                .par_iter_mut()
                .zip(lin_from.data.par_iter())
                .for_each(|(dst, &p)| {
                    for ch in 0..4 {
                        dst[ch] = p[ch] * (1.0 - k) + c[ch] * k;
                    }
                });
        } else {
            let k = (t - 0.5) * 2.0;
            let lin_to = to.to_linear();
            lin_out
                .data
                .par_iter_mut()
                .zip(lin_to.data.par_iter())
                .for_each(|(dst, &p)| {
                    for ch in 0..4 {
                        dst[ch] = c[ch] * (1.0 - k) + p[ch] * k;
                    }
                });
        }

        out.write_linear(&lin_out);
    }
}

inventory::submit! {
    TransitionFactory {
        id: "dip",
        description: "Dip to black or white at the cut midpoint",
        build: DipTransition::from_params,
    }
}

// ============================================================================
// 4. Push / Slide ("push")
// ============================================================================
#[derive(Clone, Copy, Debug)]
pub enum PushDirection {
    Left,
    Right,
    Up,
    Down,
}

pub struct PushTransition {
    pub direction: PushDirection,
}

impl PushTransition {
    pub fn from_params(params: &toml::Value) -> Result<Box<dyn Transition>> {
        let dir = match params
            .get("direction")
            .and_then(|v| v.as_str())
            .unwrap_or("left")
        {
            "right" => PushDirection::Right,
            "up" => PushDirection::Up,
            "down" => PushDirection::Down,
            _ => PushDirection::Left,
        };
        Ok(Box::new(Self { direction: dir }))
    }
}

impl Transition for PushTransition {
    fn id(&self) -> &'static str {
        "push"
    }

    fn render(
        &self,
        ctx: &TransitionCtx,
        from: FrameRef<'_>,
        to: FrameRef<'_>,
        progress: f32,
        mut out: FrameMut<'_>,
    ) {
        if progress <= 0.0 {
            out.copy_from_ref(from);
            return;
        }
        if progress >= 1.0 {
            out.copy_from_ref(to);
            return;
        }
        let t = apply_easing(&ctx.easing, progress);
        let w = from.width;
        let h = from.height;
        let lin_from = from.to_linear();
        let lin_to = to.to_linear();
        let mut lin_out = LinearRgbaFrame::new(w, h, from.pts);
        let dir = self.direction;

        lin_out
            .data
            .par_chunks_exact_mut(w as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let v = (y as f32) / ((h.max(2) - 1) as f32);
                for (x, dst) in row.iter_mut().enumerate() {
                    let u = (x as f32) / ((w.max(2) - 1) as f32);
                    *dst = match dir {
                        PushDirection::Left => {
                            if u + t < 1.0 {
                                lin_from.sample_bilinear(u + t, v)
                            } else {
                                lin_to.sample_bilinear(u + t - 1.0, v)
                            }
                        }
                        PushDirection::Right => {
                            if u >= t {
                                lin_from.sample_bilinear(u - t, v)
                            } else {
                                lin_to.sample_bilinear(u - t + 1.0, v)
                            }
                        }
                        PushDirection::Up => {
                            if v + t < 1.0 {
                                lin_from.sample_bilinear(u, v + t)
                            } else {
                                lin_to.sample_bilinear(u, v + t - 1.0)
                            }
                        }
                        PushDirection::Down => {
                            if v >= t {
                                lin_from.sample_bilinear(u, v - t)
                            } else {
                                lin_to.sample_bilinear(u, v - t + 1.0)
                            }
                        }
                    };
                }
            });

        out.write_linear(&lin_out);
    }
}

inventory::submit! {
    TransitionFactory {
        id: "push",
        description: "Directional slide/push transition (left, right, up, down)",
        build: PushTransition::from_params,
    }
}

// ============================================================================
// 5. Zoom Punch ("zoom_punch")
// ============================================================================
pub struct ZoomPunch {
    pub strength: f32,
}

impl ZoomPunch {
    pub fn from_params(params: &toml::Value) -> Result<Box<dyn Transition>> {
        let strength = params
            .get("strength")
            .and_then(|v| v.as_float())
            .unwrap_or(0.35) as f32;
        Ok(Box::new(Self { strength }))
    }
}

impl Transition for ZoomPunch {
    fn id(&self) -> &'static str {
        "zoom_punch"
    }

    fn render(
        &self,
        ctx: &TransitionCtx,
        from: FrameRef<'_>,
        to: FrameRef<'_>,
        progress: f32,
        mut out: FrameMut<'_>,
    ) {
        if progress <= 0.0 {
            out.copy_from_ref(from);
            return;
        }
        if progress >= 1.0 {
            out.copy_from_ref(to);
            return;
        }
        let t = apply_easing(&ctx.easing, progress);
        let pulse = (t * std::f32::consts::PI).sin();
        let zoom = 1.0 + self.strength * (0.5 + 0.5 * ctx.beat_strength) * pulse;
        let inv_zoom = 1.0 / zoom;

        let w = from.width;
        let h = from.height;
        let lin_from = from.to_linear();
        let lin_to = to.to_linear();
        let mut lin_out = LinearRgbaFrame::new(w, h, from.pts);

        lin_out
            .data
            .par_chunks_exact_mut(w as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let v = (y as f32) / ((h.max(2) - 1) as f32);
                let zv = 0.5 + (v - 0.5) * inv_zoom;
                for (x, dst) in row.iter_mut().enumerate() {
                    let u = (x as f32) / ((w.max(2) - 1) as f32);
                    let zu = 0.5 + (u - 0.5) * inv_zoom;
                    let p0 = lin_from.sample_bilinear(zu, zv);
                    let p1 = lin_to.sample_bilinear(zu, zv);
                    for c in 0..4 {
                        dst[c] = p0[c] * (1.0 - t) + p1[c] * t;
                    }
                }
            });

        out.write_linear(&lin_out);
    }
}

inventory::submit! {
    TransitionFactory {
        id: "zoom_punch",
        description: "Scale pulse zoom punch on the beat",
        build: ZoomPunch::from_params,
    }
}

// ============================================================================
// 6. Whip Pan with Directional Blur ("whip_pan")
// ============================================================================
pub struct WhipPan {
    pub blur_samples: usize,
}

impl WhipPan {
    pub fn from_params(params: &toml::Value) -> Result<Box<dyn Transition>> {
        let blur_samples = params
            .get("samples")
            .and_then(|v| v.as_integer())
            .unwrap_or(7)
            .clamp(3, 15) as usize;
        Ok(Box::new(Self { blur_samples }))
    }
}

impl Transition for WhipPan {
    fn id(&self) -> &'static str {
        "whip_pan"
    }

    fn render(
        &self,
        ctx: &TransitionCtx,
        from: FrameRef<'_>,
        to: FrameRef<'_>,
        progress: f32,
        mut out: FrameMut<'_>,
    ) {
        if progress <= 0.0 {
            out.copy_from_ref(from);
            return;
        }
        if progress >= 1.0 {
            out.copy_from_ref(to);
            return;
        }
        let t = apply_easing(&ctx.easing, progress);
        let blur_radius = 0.12 * (t * std::f32::consts::PI).sin();
        let taps = self.blur_samples;
        let inv_taps = 1.0 / (taps as f32);

        let w = from.width;
        let h = from.height;
        let lin_from = from.to_linear();
        let lin_to = to.to_linear();
        let mut lin_out = LinearRgbaFrame::new(w, h, from.pts);

        lin_out
            .data
            .par_chunks_exact_mut(w as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let v = (y as f32) / ((h.max(2) - 1) as f32);
                for (x, dst) in row.iter_mut().enumerate() {
                    let u = (x as f32) / ((w.max(2) - 1) as f32);
                    let mut acc = [0.0f32; 4];
                    for s in 0..taps {
                        let offset =
                            (((s as f32) / ((taps - 1) as f32)) - 0.5) * blur_radius;
                        let su = (u + t + offset).rem_euclid(2.0);
                        let sample = if su < 1.0 {
                            lin_from.sample_bilinear(su, v)
                        } else {
                            lin_to.sample_bilinear(su - 1.0, v)
                        };
                        for c in 0..4 {
                            acc[c] += sample[c] * inv_taps;
                        }
                    }
                    *dst = acc;
                }
            });

        out.write_linear(&lin_out);
    }
}

inventory::submit! {
    TransitionFactory {
        id: "whip_pan",
        description: "Fast horizontal whip pan with multi-tap directional motion blur",
        build: WhipPan::from_params,
    }
}

// ============================================================================
// 7. Luma Wipe ("luma_wipe")
// ============================================================================
pub struct LumaWipe {
    pub softness: f32,
}

impl LumaWipe {
    pub fn from_params(params: &toml::Value) -> Result<Box<dyn Transition>> {
        let softness = params
            .get("softness")
            .and_then(|v| v.as_float())
            .unwrap_or(0.15) as f32;
        Ok(Box::new(Self {
            softness: softness.clamp(0.02, 0.5),
        }))
    }
}

impl Transition for LumaWipe {
    fn id(&self) -> &'static str {
        "luma_wipe"
    }

    fn render(
        &self,
        ctx: &TransitionCtx,
        from: FrameRef<'_>,
        to: FrameRef<'_>,
        progress: f32,
        mut out: FrameMut<'_>,
    ) {
        if progress <= 0.0 {
            out.copy_from_ref(from);
            return;
        }
        if progress >= 1.0 {
            out.copy_from_ref(to);
            return;
        }
        let t = apply_easing(&ctx.easing, progress);
        let soft = self.softness;

        let lin_from = from.to_linear();
        let lin_to = to.to_linear();
        let mut lin_out = LinearRgbaFrame::new(from.width, from.height, from.pts);

        lin_out
            .data
            .par_iter_mut()
            .zip(lin_from.data.par_iter().zip(lin_to.data.par_iter()))
            .for_each(|(dst, (&p0, &p1))| {
                let luma = (0.2126 * p0[0] + 0.7152 * p0[1] + 0.0722 * p0[2]).clamp(0.0, 1.0);
                let threshold = t * (1.0 + soft * 2.0) - soft;
                let k = ((threshold - luma + soft) / (2.0 * soft)).clamp(0.0, 1.0);
                for c in 0..4 {
                    dst[c] = p0[c] * (1.0 - k) + p1[c] * k;
                }
            });

        out.write_linear(&lin_out);
    }
}

inventory::submit! {
    TransitionFactory {
        id: "luma_wipe",
        description: "Soft-edge luminance-guided wipe between outgoing and incoming shots",
        build: LumaWipe::from_params,
    }
}

// ============================================================================
// 8. RGB-Split Glitch ("glitch")
// ============================================================================
pub struct GlitchTransition {
    pub intensity: f32,
}

impl GlitchTransition {
    pub fn from_params(params: &toml::Value) -> Result<Box<dyn Transition>> {
        let intensity = params
            .get("intensity")
            .and_then(|v| v.as_float())
            .unwrap_or(0.06) as f32;
        Ok(Box::new(Self { intensity }))
    }
}

impl Transition for GlitchTransition {
    fn id(&self) -> &'static str {
        "glitch"
    }

    fn render(
        &self,
        ctx: &TransitionCtx,
        from: FrameRef<'_>,
        to: FrameRef<'_>,
        progress: f32,
        mut out: FrameMut<'_>,
    ) {
        if progress <= 0.0 {
            out.copy_from_ref(from);
            return;
        }
        if progress >= 1.0 {
            out.copy_from_ref(to);
            return;
        }
        let t = apply_easing(&ctx.easing, progress);
        let env = (t * std::f32::consts::PI).sin();
        let shift = self.intensity * env;

        let w = from.width;
        let h = from.height;
        let lin_from = from.to_linear();
        let lin_to = to.to_linear();
        let mut lin_out = LinearRgbaFrame::new(w, h, from.pts);

        lin_out
            .data
            .par_chunks_exact_mut(w as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let v = (y as f32) / ((h.max(2) - 1) as f32);
                let band_jitter = (((y / 8) as f32 * 12.9898).sin() * 43758.547).fract() - 0.5;
                let slice_offset = band_jitter * shift * 1.5;

                for (x, dst) in row.iter_mut().enumerate() {
                    let u = (x as f32) / ((w.max(2) - 1) as f32) + slice_offset;
                    let r0 = lin_from.sample_bilinear(u + shift, v)[0];
                    let g0 = lin_from.sample_bilinear(u, v)[1];
                    let b0 = lin_from.sample_bilinear(u - shift, v)[2];

                    let r1 = lin_to.sample_bilinear(u - shift, v)[0];
                    let g1 = lin_to.sample_bilinear(u, v)[1];
                    let b1 = lin_to.sample_bilinear(u + shift, v)[2];

                    dst[0] = r0 * (1.0 - t) + r1 * t;
                    dst[1] = g0 * (1.0 - t) + g1 * t;
                    dst[2] = b0 * (1.0 - t) + b1 * t;
                    dst[3] = 1.0;
                }
            });

        out.write_linear(&lin_out);
    }
}

inventory::submit! {
    TransitionFactory {
        id: "glitch",
        description: "RGB chromatic aberration split with scanline slice displacement",
        build: GlitchTransition::from_params,
    }
}

// ============================================================================
// 9. Flash ("flash")
// ============================================================================
pub struct FlashTransition {
    pub exposure: f32,
}

impl FlashTransition {
    pub fn from_params(params: &toml::Value) -> Result<Box<dyn Transition>> {
        let exposure = params
            .get("exposure")
            .and_then(|v| v.as_float())
            .unwrap_or(0.85) as f32;
        Ok(Box::new(Self { exposure }))
    }
}

impl Transition for FlashTransition {
    fn id(&self) -> &'static str {
        "flash"
    }

    fn render(
        &self,
        ctx: &TransitionCtx,
        from: FrameRef<'_>,
        to: FrameRef<'_>,
        progress: f32,
        mut out: FrameMut<'_>,
    ) {
        if progress <= 0.0 {
            out.copy_from_ref(from);
            return;
        }
        if progress >= 1.0 {
            out.copy_from_ref(to);
            return;
        }
        let t = apply_easing(&ctx.easing, progress);
        let bloom = (t * std::f32::consts::PI).sin().powi(2) * self.exposure;

        let lin_from = from.to_linear();
        let lin_to = to.to_linear();
        let mut lin_out = LinearRgbaFrame::new(from.width, from.height, from.pts);

        lin_out
            .data
            .par_iter_mut()
            .zip(lin_from.data.par_iter().zip(lin_to.data.par_iter()))
            .for_each(|(dst, (&p0, &p1))| {
                for c in 0..3 {
                    let base = p0[c] * (1.0 - t) + p1[c] * t;
                    dst[c] = (base + bloom).clamp(0.0, 1.0);
                }
                dst[3] = 1.0;
            });

        out.write_linear(&lin_out);
    }
}

inventory::submit! {
    TransitionFactory {
        id: "flash",
        description: "Additive linear-light exposure bloom on the beat",
        build: FlashTransition::from_params,
    }
}

// ============================================================================
// Transition Policy (Automatic Match-Cut vs Stylized Cut Selection + TOML Config)
// ============================================================================
pub trait TransitionPolicy: Send + Sync {
    fn select(
        &self,
        slot_index: usize,
        boundary_similarity: f32,
        beat_strength: f32,
        is_downbeat: bool,
        slot_duration: Rational,
    ) -> TransitionSpec;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StyleTomlConfig {
    #[serde(default = "default_match_cut_thresh")]
    pub match_cut_threshold: f32,
    #[serde(default = "default_strong_beat_thresh")]
    pub strong_beat_threshold: f32,
    #[serde(default = "default_duration_ms")]
    pub default_duration_ms: i64,
    #[serde(default = "default_easing_str")]
    pub default_easing: String,
    #[serde(default = "default_strong_transition")]
    pub strong_beat_transition: String,
    #[serde(default = "default_low_sim_transition")]
    pub low_similarity_transition: String,
}

fn default_match_cut_thresh() -> f32 {
    0.0
}
fn default_strong_beat_thresh() -> f32 {
    1.01
}
fn default_duration_ms() -> i64 {
    0
}
fn default_easing_str() -> String {
    "linear".to_string()
}
fn default_strong_transition() -> String {
    "cut".to_string()
}
fn default_low_sim_transition() -> String {
    "cut".to_string()
}

impl Default for StyleTomlConfig {
    fn default() -> Self {
        Self {
            match_cut_threshold: default_match_cut_thresh(),
            strong_beat_threshold: default_strong_beat_thresh(),
            default_duration_ms: default_duration_ms(),
            default_easing: default_easing_str(),
            strong_beat_transition: default_strong_transition(),
            low_similarity_transition: default_low_sim_transition(),
        }
    }
}

impl StyleTomlConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let cfg: Self = toml::from_str(&text)?;
        Ok(cfg)
    }
}

impl TransitionPolicy for StyleTomlConfig {
    fn select(
        &self,
        slot_index: usize,
        boundary_similarity: f32,
        beat_strength: f32,
        is_downbeat: bool,
        slot_duration: Rational,
    ) -> TransitionSpec {
        if slot_index == 0 {
            return TransitionSpec::default();
        }
        // When boundary similarity is high, use a clean hard cut to showcase the visual match cut!
        if boundary_similarity >= self.match_cut_threshold && !is_downbeat {
            return TransitionSpec::default();
        }

        let max_trans_dur = Rational::from_secs_f64(slot_duration.as_secs_f64() * 0.35);
        let dur = Rational::from_millis(self.default_duration_ms).min(max_trans_dur);

        let id = if beat_strength >= self.strong_beat_threshold || is_downbeat {
            self.strong_beat_transition.clone()
        } else {
            self.low_similarity_transition.clone()
        };

        if id == "cut" || dur <= Rational::ZERO {
            return TransitionSpec::default();
        }

        TransitionSpec {
            id,
            duration: dur,
            easing: self.default_easing.clone(),
            params: toml::Value::Table(toml::map::Map::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ae_core::RgbaFrame;

    #[test]
    fn property_all_transitions_preserve_endpoints_and_midpoint_is_valid() {
        let from = RgbaFrame::solid(64, 36, [220, 40, 60, 255], Rational::ZERO);
        let to = RgbaFrame::solid(64, 36, [40, 180, 230, 255], Rational::ONE);
        let ctx = TransitionCtx {
            width: 64,
            height: 36,
            ..Default::default()
        };
        let empty_params = toml::Value::Table(toml::map::Map::new());

        let registered = list_registered_transitions();
        assert!(
            registered.len() >= 9,
            "expected all 9 built-in transitions registered, found {}",
            registered.len()
        );

        for (id, _) in registered {
            let tr = build_transition(id, &empty_params).unwrap();

            // Progress = 0.0 -> exact `from` frame
            let mut out0 = RgbaFrame::new(64, 36, Rational::ZERO);
            tr.render(&ctx, from.as_ref(), to.as_ref(), 0.0, out0.as_mut());
            assert_eq!(
                out0.data, from.data,
                "transition '{id}' failed endpoint progress=0.0"
            );

            // Progress = 1.0 -> exact `to` frame
            let mut out1 = RgbaFrame::new(64, 36, Rational::ONE);
            tr.render(&ctx, from.as_ref(), to.as_ref(), 1.0, out1.as_mut());
            assert_eq!(
                out1.data, to.data,
                "transition '{id}' failed endpoint progress=1.0"
            );

            // Progress = 0.5 -> valid non-empty intermediate frame
            let mut out_mid = RgbaFrame::new(64, 36, Rational::new(1, 2));
            tr.render(&ctx, from.as_ref(), to.as_ref(), 0.5, out_mid.as_mut());
            assert!(out_mid.mean_luma() >= 0.0 && out_mid.mean_luma() <= 1.0);
        }
    }
}
