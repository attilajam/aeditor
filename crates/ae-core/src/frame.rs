use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::time::Rational;

/// 8-bit per channel RGBA frame in BT.709 / sRGB transfer space.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RgbaFrame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub pts: Rational,
}

impl RgbaFrame {
    pub fn new(width: u32, height: u32, pts: Rational) -> Self {
        let len = (width as usize) * (height as usize) * 4;
        Self {
            width,
            height,
            data: vec![0u8; len],
            pts,
        }
    }

    pub fn from_rgba(width: u32, height: u32, data: Vec<u8>, pts: Rational) -> Self {
        assert_eq!(
            data.len(),
            (width as usize) * (height as usize) * 4,
            "RgbaFrame buffer size mismatch"
        );
        Self {
            width,
            height,
            data,
            pts,
        }
    }

    pub fn solid(width: u32, height: u32, rgba: [u8; 4], pts: Rational) -> Self {
        let pixels = (width as usize) * (height as usize);
        let mut data = Vec::with_capacity(pixels * 4);
        for _ in 0..pixels {
            data.extend_from_slice(&rgba);
        }
        Self {
            width,
            height,
            data,
            pts,
        }
    }

    #[inline]
    pub fn as_ref(&self) -> FrameRef<'_> {
        FrameRef {
            width: self.width,
            height: self.height,
            data: &self.data,
            pts: self.pts,
        }
    }

    #[inline]
    pub fn as_mut(&mut self) -> FrameMut<'_> {
        FrameMut {
            width: self.width,
            height: self.height,
            data: &mut self.data,
            pts: self.pts,
        }
    }

    #[inline]
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let idx = ((y * self.width + x) as usize) * 4;
        [
            self.data[idx],
            self.data[idx + 1],
            self.data[idx + 2],
            self.data[idx + 3],
        ]
    }

    /// Mean luma in [0.0, 1.0] (BT.709 coefficients).
    pub fn mean_luma(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0f64;
        let pixels = (self.width as usize) * (self.height as usize);
        for px in self.data.as_chunks::<4>().0 {
            let y = 0.2126 * (px[0] as f64) + 0.7152 * (px[1] as f64) + 0.0722 * (px[2] as f64);
            sum += y / 255.0;
        }
        (sum / (pixels as f64)) as f32
    }

    /// Convert from BT.709 / sRGB u8 to linear-light premultiplied RGBA f32.
    pub fn to_linear(&self) -> LinearRgbaFrame {
        let lut = bt709_u8_to_linear_lut();
        let pixels = (self.width as usize) * (self.height as usize);
        let mut out = vec![[0.0f32; 4]; pixels];
        out.par_iter_mut()
            .zip(self.data.par_chunks_exact(4))
            .for_each(|(dst, src)| {
                let a = (src[3] as f32) / 255.0;
                dst[0] = lut[src[0] as usize] * a;
                dst[1] = lut[src[1] as usize] * a;
                dst[2] = lut[src[2] as usize] * a;
                dst[3] = a;
            });
        LinearRgbaFrame {
            width: self.width,
            height: self.height,
            data: out,
            pts: self.pts,
        }
    }

    /// Compute mean CIE76 / Oklab perceptual ΔE (scaled to 0..100) between two frames.
    pub fn mean_delta_e(&self, other: &Self) -> f32 {
        assert_eq!(self.width, other.width);
        assert_eq!(self.height, other.height);
        let pixels = (self.width as usize) * (self.height as usize);
        if pixels == 0 {
            return 0.0;
        }
        let sum: f64 = self
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .zip(other.data.as_chunks::<4>().0.iter())
            .map(|(a, b)| {
                let dr = (a[0] as f64 - b[0] as f64) / 255.0;
                let dg = (a[1] as f64 - b[1] as f64) / 255.0;
                let db = (a[2] as f64 - b[2] as f64) / 255.0;
                (0.2126 * dr * dr + 0.7152 * dg * dg + 0.0722 * db * db).sqrt() * 100.0
            })
            .sum();
        (sum / (pixels as f64)) as f32
    }
}

/// Read-only view of an 8-bit RGBA frame.
#[derive(Clone, Copy, Debug)]
pub struct FrameRef<'a> {
    pub width: u32,
    pub height: u32,
    pub data: &'a [u8],
    pub pts: Rational,
}

impl<'a> FrameRef<'a> {
    pub fn to_owned_frame(self) -> RgbaFrame {
        RgbaFrame {
            width: self.width,
            height: self.height,
            data: self.data.to_vec(),
            pts: self.pts,
        }
    }

    pub fn to_linear(self) -> LinearRgbaFrame {
        let lut = bt709_u8_to_linear_lut();
        let pixels = (self.width as usize) * (self.height as usize);
        let mut out = vec![[0.0f32; 4]; pixels];
        out.par_iter_mut()
            .zip(self.data.par_chunks_exact(4))
            .for_each(|(dst, src)| {
                let a = (src[3] as f32) / 255.0;
                dst[0] = lut[src[0] as usize] * a;
                dst[1] = lut[src[1] as usize] * a;
                dst[2] = lut[src[2] as usize] * a;
                dst[3] = a;
            });
        LinearRgbaFrame {
            width: self.width,
            height: self.height,
            data: out,
            pts: self.pts,
        }
    }
}

/// Mutable view of an 8-bit RGBA frame.
#[derive(Debug)]
pub struct FrameMut<'a> {
    pub width: u32,
    pub height: u32,
    pub data: &'a mut [u8],
    pub pts: Rational,
}

impl<'a> FrameMut<'a> {
    pub fn copy_from_ref(&mut self, src: FrameRef<'_>) {
        assert_eq!(self.width, src.width);
        assert_eq!(self.height, src.height);
        self.data.copy_from_slice(src.data);
    }

    pub fn write_linear(&mut self, linear: &LinearRgbaFrame) {
        assert_eq!(self.width, linear.width);
        assert_eq!(self.height, linear.height);
        self.data
            .par_chunks_exact_mut(4)
            .zip(linear.data.par_iter())
            .for_each(|(dst, src)| {
                let a = src[3].clamp(0.0, 1.0);
                let inv_a = if a > 1e-6 { 1.0 / a } else { 0.0 };
                dst[0] = linear_to_bt709_u8(src[0] * inv_a);
                dst[1] = linear_to_bt709_u8(src[1] * inv_a);
                dst[2] = linear_to_bt709_u8(src[2] * inv_a);
                dst[3] = (a * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            });
    }
}

/// Linear-light premultiplied RGBA f32 frame for physically accurate compositing & transitions.
#[derive(Clone, Debug)]
pub struct LinearRgbaFrame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<[f32; 4]>,
    pub pts: Rational,
}

impl LinearRgbaFrame {
    pub fn new(width: u32, height: u32, pts: Rational) -> Self {
        Self {
            width,
            height,
            data: vec![[0.0, 0.0, 0.0, 1.0]; (width as usize) * (height as usize)],
            pts,
        }
    }

    #[inline]
    pub fn sample_bilinear(&self, u: f32, v: f32) -> [f32; 4] {
        let w = self.width as f32;
        let h = self.height as f32;
        let x = (u.clamp(0.0, 1.0) * (w - 1.0)).clamp(0.0, w - 1.0);
        let y = (v.clamp(0.0, 1.0) * (h - 1.0)).clamp(0.0, h - 1.0);
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = x - (x0 as f32);
        let fy = y - (y0 as f32);

        let p00 = self.data[(y0 * self.width + x0) as usize];
        let p10 = self.data[(y0 * self.width + x1) as usize];
        let p01 = self.data[(y1 * self.width + x0) as usize];
        let p11 = self.data[(y1 * self.width + x1) as usize];

        let mut out = [0.0f32; 4];
        for c in 0..4 {
            let top = p00[c] * (1.0 - fx) + p10[c] * fx;
            let bot = p01[c] * (1.0 - fx) + p11[c] * fx;
            out[c] = top * (1.0 - fy) + bot * fy;
        }
        out
    }

    pub fn to_rgba_u8(&self) -> RgbaFrame {
        let mut out = RgbaFrame::new(self.width, self.height, self.pts);
        out.as_mut().write_linear(self);
        out
    }
}

/// BT.709 / sRGB electro-optical transfer function (EOTF) to linear light [0.0, 1.0].
#[inline]
pub fn bt709_u8_to_linear(v: u8) -> f32 {
    let x = (v as f32) / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
pub fn bt709_u8_to_linear_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        lut[i] = bt709_u8_to_linear(i as u8);
        i += 1;
    }
    lut
}

/// Linear light [0.0, 1.0] to BT.709 / sRGB u8 OETF.
#[inline]
pub fn linear_to_bt709_u8(linear: f32) -> u8 {
    let x = linear.clamp(0.0, 1.0);
    let encoded = if x <= 0.0031308 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}
