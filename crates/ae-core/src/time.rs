use std::fmt;
use std::ops::{Add, Div, Mul, Sub};
use std::str::FromStr;

use num_rational::Rational64;
use serde::{Deserialize, Serialize};

use crate::error::{AeError, Result};

/// Exact rational timestamp in seconds (PTS), backed by `Rational64`.
/// Never uses frame indices as the source of truth so VFR footage stays accurate.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub struct Rational(pub Rational64);

impl Rational {
    pub const ZERO: Self = Self(Rational64::new_raw(0, 1));
    pub const ONE: Self = Self(Rational64::new_raw(1, 1));
    pub const MICRO_DEN: i64 = 1_000_000;

    #[inline]
    pub fn new(num: i64, den: i64) -> Self {
        assert!(den != 0, "Rational denominator cannot be zero");
        Self(Rational64::new(num, den))
    }

    #[inline]
    pub fn from_micros(micros: i64) -> Self {
        Self(Rational64::new(micros, Self::MICRO_DEN))
    }

    #[inline]
    pub fn from_millis(millis: i64) -> Self {
        Self(Rational64::new(millis, 1_000))
    }

    #[inline]
    pub fn from_secs(secs: i64) -> Self {
        Self(Rational64::from_integer(secs))
    }

    #[inline]
    pub fn from_secs_f64(secs: f64) -> Self {
        if !secs.is_finite() {
            return Self::ZERO;
        }
        let micros = (secs * (Self::MICRO_DEN as f64)).round() as i64;
        Self::from_micros(micros)
    }

    #[inline]
    pub fn as_secs_f64(self) -> f64 {
        (*self.0.numer() as f64) / (*self.0.denom() as f64)
    }

    #[inline]
    pub fn as_secs_f32(self) -> f32 {
        self.as_secs_f64() as f32
    }

    #[inline]
    pub fn numer(self) -> i64 {
        *self.0.numer()
    }

    #[inline]
    pub fn denom(self) -> i64 {
        *self.0.denom()
    }

    #[inline]
    pub fn is_negative(self) -> bool {
        *self.0.numer() < 0
    }

    #[inline]
    pub fn abs(self) -> Self {
        if self.is_negative() {
            Self(-self.0)
        } else {
            self
        }
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        if self >= other { self } else { other }
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        if self <= other { self } else { other }
    }

    /// Convert a frame index at a given frame rate (frames per second) to timestamp.
    #[inline]
    pub fn from_frame_index(frame_idx: i64, fps: Self) -> Self {
        if fps.numer() <= 0 {
            return Self::ZERO;
        }
        Self(Rational64::from_integer(frame_idx) / fps.0)
    }

    /// Nearest frame index for a given FPS.
    #[inline]
    pub fn to_nearest_frame(self, fps: Self) -> i64 {
        let exact = self.0 * fps.0;
        let num = *exact.numer();
        let den = *exact.denom();
        if num >= 0 {
            (num + den / 2) / den
        } else {
            (num - den / 2) / den
        }
    }

    /// Floor frame index for a given FPS.
    #[inline]
    pub fn to_floor_frame(self, fps: Self) -> i64 {
        let exact = self.0 * fps.0;
        exact.floor().to_integer()
    }

    /// Parse duration string like `"0.25s"`, `"250ms"`, `"1.5"`, or `"1001/30000"`.
    pub fn parse_duration(s: &str) -> Result<Self> {
        let s = s.trim();
        if let Some(ms) = s.strip_suffix("ms") {
            let val: f64 = ms
                .trim()
                .parse()
                .map_err(|_| AeError::Config(format!("invalid ms duration: {s}")))?;
            Ok(Self::from_secs_f64(val / 1000.0))
        } else if let Some(sec) = s.strip_suffix('s') {
            let val: f64 = sec
                .trim()
                .parse()
                .map_err(|_| AeError::Config(format!("invalid seconds duration: {s}")))?;
            Ok(Self::from_secs_f64(val))
        } else if let Some((n, d)) = s.split_once('/') {
            let num: i64 = n
                .trim()
                .parse()
                .map_err(|_| AeError::Config(format!("invalid rational numerator: {s}")))?;
            let den: i64 = d
                .trim()
                .parse()
                .map_err(|_| AeError::Config(format!("invalid rational denominator: {s}")))?;
            if den == 0 {
                return Err(AeError::Config("zero denominator in rational".into()));
            }
            Ok(Self::new(num, den))
        } else {
            let val: f64 = s
                .parse()
                .map_err(|_| AeError::Config(format!("invalid duration: {s}")))?;
            Ok(Self::from_secs_f64(val))
        }
    }
}

impl FromStr for Rational {
    type Err = AeError;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse_duration(s)
    }
}

impl fmt::Debug for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Rational({}/{}, {:.4}s)",
            self.numer(),
            self.denom(),
            self.as_secs_f64()
        )
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}s", self.as_secs_f64())
    }
}

impl Add for Rational {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        // Reduce denominator if needed to avoid i64 overflow on long timelines
        if self.denom().saturating_mul(rhs.denom()) > 10_000_000_000 {
            Self::from_secs_f64(self.as_secs_f64() + rhs.as_secs_f64())
        } else {
            Self(self.0 + rhs.0)
        }
    }
}

impl Sub for Rational {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        if self.denom().saturating_mul(rhs.denom()) > 10_000_000_000 {
            Self::from_secs_f64(self.as_secs_f64() - rhs.as_secs_f64())
        } else {
            Self(self.0 - rhs.0)
        }
    }
}

impl Mul<i64> for Rational {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: i64) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl Div<i64> for Rational {
    type Output = Self;
    #[inline]
    fn div(self, rhs: i64) -> Self::Output {
        Self(self.0 / rhs)
    }
}
