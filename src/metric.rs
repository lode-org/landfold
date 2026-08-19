//! High-dimensional dissimilarities.
//!
//! Euclidean and minimum-image torus metrics are standard. The n-sphere
//! geodesic is the great-circle distance after the usual polar embedding.
//! `-log(a·b)` is the SOAP-style kernel distance of Bartok, Kondor and
//! Csanyi, *Phys. Rev. B* **87**, 184115 (2013),
//! <https://doi.org/10.1103/PhysRevB.87.184115>.

use crate::error::{LandfoldError, Result};

pub trait Metric: Send + Sync {
    fn dim(&self) -> Option<usize> {
        None
    }

    fn dist(&self, a: &[f64], b: &[f64]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(LandfoldError::MetricSize {
                left: a.len(),
                right: b.len(),
            });
        }
        if let Some(expected) = self.dim() {
            if a.len() != expected {
                return Err(LandfoldError::MetricSize {
                    left: a.len(),
                    right: expected,
                });
            }
        }
        Ok(self.dist_unchecked(a, b))
    }

    fn dist_unchecked(&self, a: &[f64], b: &[f64]) -> f64;

    /// True for plain Euclidean so embed can use the Gram GEMM.
    fn is_euclid(&self) -> bool {
        false
    }

    /// `c = a - b` in the metric's tangent sense (PBC-wrapped when needed).
    fn diff(&self, a: &[f64], b: &[f64], c: &mut [f64]) {
        for i in 0..a.len() {
            c[i] = a[i] - b[i];
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Euclid;

impl Metric for Euclid {
    fn is_euclid(&self) -> bool {
        true
    }

    fn dist_unchecked(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut acc = 0.0;
        for i in 0..a.len() {
            let d = b[i] - a[i];
            acc += d * d;
        }
        acc.sqrt()
    }
}

/// Hypertoroidal (minimum-image) Euclidean metric. `periods[i]` is the
/// period of coordinate `i`.
#[derive(Clone, Debug)]
pub struct Periodic {
    pub periods: Vec<f64>,
}

impl Periodic {
    pub fn new(periods: Vec<f64>) -> Result<Self> {
        validate_periods(&periods)?;
        Ok(Self { periods })
    }

    pub fn isotropic(dim: usize, period: f64) -> Result<Self> {
        Self::new(vec![period; dim])
    }
}

impl Metric for Periodic {
    fn dim(&self) -> Option<usize> {
        Some(self.periods.len())
    }

    fn dist_unchecked(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut acc = 0.0;
        for i in 0..a.len() {
            let mut dx = b[i] - a[i];
            dx /= self.periods[i];
            dx -= dx.round();
            dx *= self.periods[i];
            acc += dx * dx;
        }
        acc.sqrt()
    }

    fn diff(&self, a: &[f64], b: &[f64], c: &mut [f64]) {
        for i in 0..a.len() {
            let mut dx = b[i] - a[i];
            dx /= self.periods[i];
            dx -= dx.round();
            dx *= self.periods[i];
            c[i] = dx;
        }
    }
}

/// Geodesic distance on an n-sphere.
#[derive(Clone, Debug)]
pub struct Sphere {
    pub periods: Vec<f64>,
}

impl Sphere {
    pub fn new(periods: Vec<f64>) -> Result<Self> {
        validate_periods(&periods)?;
        Ok(Self { periods })
    }
}

fn validate_periods(periods: &[f64]) -> Result<()> {
    if periods.is_empty() || periods.iter().any(|&p| !p.is_finite() || p <= 0.0) {
        return Err(LandfoldError::Msg(
            "metric periods must be finite and > 0".into(),
        ));
    }
    Ok(())
}

impl Metric for Sphere {
    fn dim(&self) -> Option<usize> {
        Some(self.periods.len())
    }

    fn dist_unchecked(&self, a: &[f64], b: &[f64]) -> f64 {
        let n = a.len();
        let twopi = std::f64::consts::TAU;
        let mut xs = 1.0;
        let mut ys = 1.0;
        let mut xy = 0.0;
        for i in 0..n {
            let xi = xs * (a[i] * twopi / self.periods[i]).cos();
            let yi = ys * (b[i] * twopi / self.periods[i]).cos();
            xs *= (a[i] * twopi / self.periods[i]).sin();
            ys *= (b[i] * twopi / self.periods[i]).sin();
            xy += xi * yi;
        }
        xy += xs * ys;
        if xy >= 1.0 {
            0.0
        } else if xy <= -1.0 {
            std::f64::consts::PI
        } else {
            xy.acos()
        }
    }

    fn diff(&self, a: &[f64], b: &[f64], c: &mut [f64]) {
        let n = a.len();
        for i in 0..n.saturating_sub(1) {
            c[i] = b[i] - a[i];
        }
        if n > 0 {
            let p = self.periods[n - 1];
            let mut dx = b[n - 1] - a[n - 1];
            dx /= p;
            dx -= dx.round();
            dx *= p;
            c[n - 1] = dx;
        }
    }
}

/// `d(a,b) = -log(a·b)`. Used for SOAP-like unit-sphere descriptors.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dot;

impl Metric for Dot {
    fn dist_unchecked(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut acc = 0.0;
        for i in 0..a.len() {
            acc += b[i] * a[i];
        }
        if acc <= 0.0 { f64::INFINITY } else { -acc.ln() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn euclid_3_4_5() {
        let m = Euclid;
        assert_relative_eq!(
            m.dist(&[0.0, 0.0], &[3.0, 4.0]).unwrap(),
            5.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn pbc_wraps() {
        let m = Periodic::isotropic(1, 1.0).unwrap();
        assert_relative_eq!(m.dist(&[0.05], &[0.95]).unwrap(), 0.1, epsilon = 1e-14);
    }

    #[test]
    fn sphere_identical_is_zero() {
        let m = Sphere::new(vec![1.0, 1.0]).unwrap();
        assert_relative_eq!(
            m.dist(&[0.1, 0.2], &[0.1, 0.2]).unwrap(),
            0.0,
            epsilon = 1e-14
        );
    }

    #[test]
    fn rejects_invalid_periods() {
        assert!(Periodic::new(vec![0.0]).is_err());
        assert!(Periodic::new(vec![f64::NAN]).is_err());
        assert!(Sphere::new(Vec::new()).is_err());
        assert!(Sphere::new(vec![-1.0]).is_err());
    }

    #[test]
    fn rejects_metric_dimension_mismatches() {
        let periodic = Periodic::isotropic(1, 1.0).unwrap();
        assert!(periodic.dist(&[], &[]).is_err());
        assert!(periodic.dist(&[0.0, 0.0], &[0.0, 0.0]).is_err());
        let sphere = Sphere::new(vec![1.0, 1.0]).unwrap();
        assert!(sphere.dist(&[0.0], &[0.0]).is_err());
        assert!(sphere.dist(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn dot_of_ones() {
        let m = Dot;
        let a = [1.0_f64 / 2.0_f64.sqrt(); 2];
        assert_relative_eq!(m.dist(&a, &a).unwrap(), 0.0, epsilon = 1e-14);
    }
}
