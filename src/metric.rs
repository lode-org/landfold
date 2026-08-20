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
        match self.dim() {
            Some(expected) if a.len() != expected => {
                return Err(LandfoldError::MetricSize {
                    left: a.len(),
                    right: expected,
                });
            }
            _ => {}
        }
        let distance = self.dist_unchecked(a, b);
        validate_distance(distance)?;
        Ok(distance)
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

pub(crate) fn validate_distance(distance: f64) -> Result<()> {
    if !distance.is_finite() || distance < 0.0 {
        return Err(LandfoldError::Msg(
            "metric distance must be finite and nonnegative".into(),
        ));
    }
    Ok(())
}

pub(crate) fn stable_euclid<'a, 'b>(
    a: impl Iterator<Item = &'a f64>,
    b: impl Iterator<Item = &'b f64>,
) -> f64 {
    let mut scale = 0.0;
    let mut sum = 0.0;
    for (&ai, &bi) in a.zip(b) {
        let delta = (bi - ai).abs();
        if !delta.is_finite() {
            return f64::INFINITY;
        }
        if delta > scale {
            let ratio = if scale == 0.0 { 0.0 } else { scale / delta };
            sum = sum * ratio * ratio + 1.0;
            scale = delta;
        } else if scale > 0.0 {
            let ratio = delta / scale;
            sum += ratio * ratio;
        }
    }
    scale * sum.sqrt()
}

fn periodic_delta(a: f64, b: f64, period: f64) -> f64 {
    let a_mod = a.rem_euclid(period);
    let b_mod = b.rem_euclid(period);
    let mut delta = b_mod - a_mod;
    let half_period = 0.5 * period;
    if delta > half_period {
        delta -= period;
    } else if delta < -half_period {
        delta += period;
    }
    delta
}

fn periodic_phase(value: f64, period: f64) -> f64 {
    value.rem_euclid(period) / period * std::f64::consts::TAU
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Euclid;

impl Metric for Euclid {
    fn is_euclid(&self) -> bool {
        true
    }

    fn dist_unchecked(&self, a: &[f64], b: &[f64]) -> f64 {
        stable_euclid(a.iter(), b.iter())
    }
}

/// Euclidean distance stretched along one unit axis `u`:
/// `D^2 = ||x-y||^2 + alpha (u·(x-y))^2`.
///
/// Used to emphasise a named structural contrast (fcc vs ico in the
/// coordination-count space) before the Ceriotti transfer.
#[derive(Clone, Debug)]
pub struct Stretch {
    axis: Vec<f64>,
    alpha: f64,
}

impl Stretch {
    pub fn new(axis: Vec<f64>, alpha: f64) -> Result<Self> {
        if axis.is_empty() || axis.iter().any(|v| !v.is_finite()) {
            return Err(LandfoldError::Msg(
                "stretch axis must be finite and nonempty".into(),
            ));
        }
        if !alpha.is_finite() || alpha < 0.0 {
            return Err(LandfoldError::Msg(
                "stretch alpha must be finite and nonnegative".into(),
            ));
        }
        let n2: f64 = axis.iter().map(|v| v * v).sum();
        if !(n2 > 0.0 && n2.is_finite()) {
            return Err(LandfoldError::Msg(
                "stretch axis must have positive finite length".into(),
            ));
        }
        let n = n2.sqrt();
        Ok(Self {
            axis: axis.into_iter().map(|v| v / n).collect(),
            alpha,
        })
    }

    pub fn from_refs(a: &[f64], b: &[f64], alpha: f64) -> Result<Self> {
        if a.len() != b.len() {
            return Err(LandfoldError::MetricSize {
                left: a.len(),
                right: b.len(),
            });
        }
        let axis: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
        Self::new(axis, alpha)
    }
}

impl Metric for Stretch {
    fn dim(&self) -> Option<usize> {
        Some(self.axis.len())
    }

    fn dist_unchecked(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut eu2 = 0.0;
        let mut proj = 0.0;
        for i in 0..a.len() {
            let d = a[i] - b[i];
            eu2 += d * d;
            proj += d * self.axis[i];
        }
        let d2 = eu2 + self.alpha * proj * proj;
        if !d2.is_finite() || d2 < 0.0 {
            return f64::INFINITY;
        }
        d2.sqrt()
    }
}

/// Pooled within-class Mahalanobis metric.
///
/// Points are labelled by the nearer of two named references. Distances
/// are `sqrt((x-y)^T (S_w + ridge I)^{-1} (x-y))`, so the thermal
/// directions inside each class are down-weighted and the fcc–ico
/// contrast is the long axis.
#[derive(Clone, Debug)]
pub struct Fisher {
    dim: usize,
    prec: Vec<f64>,
}

impl Fisher {
    pub fn from_refs(points: &[Vec<f64>], ref_a: &[f64], ref_b: &[f64], ridge: f64) -> Result<Self> {
        let d = ref_a.len();
        if d == 0 || ref_b.len() != d {
            return Err(LandfoldError::MetricSize {
                left: d,
                right: ref_b.len(),
            });
        }
        if !ridge.is_finite() || ridge < 0.0 {
            return Err(LandfoldError::Msg(
                "fisher ridge must be finite and nonnegative".into(),
            ));
        }
        if points
            .iter()
            .any(|p| p.len() != d || p.iter().any(|v| !v.is_finite()))
            || ref_a.iter().chain(ref_b).any(|v| !v.is_finite())
        {
            return Err(LandfoldError::Msg(
                "fisher points and references must be finite and of one dimension".into(),
            ));
        }
        let mut a: Vec<&[f64]> = Vec::new();
        let mut b: Vec<&[f64]> = Vec::new();
        for p in points {
            let da: f64 = p.iter().zip(ref_a).map(|(x, y)| (x - y) * (x - y)).sum();
            let db: f64 = p.iter().zip(ref_b).map(|(x, y)| (x - y) * (x - y)).sum();
            if da <= db {
                a.push(p);
            } else {
                b.push(p);
            }
        }
        if a.len() < 2 || b.len() < 2 {
            return Err(LandfoldError::Msg(
                "fisher needs at least two points on each side of the references".into(),
            ));
        }
        let mean = |cls: &[&[f64]]| -> Vec<f64> {
            let n = cls.len() as f64;
            let mut m = vec![0.0; d];
            for p in cls {
                for k in 0..d {
                    m[k] += p[k] / n;
                }
            }
            m
        };
        let ma = mean(&a);
        let mb = mean(&b);
        let mut sw = nalgebra::DMatrix::<f64>::zeros(d, d);
        for (cls, mu) in [(&a, &ma), (&b, &mb)] {
            for p in cls {
                for i in 0..d {
                    let di = p[i] - mu[i];
                    for j in 0..d {
                        sw[(i, j)] += di * (p[j] - mu[j]);
                    }
                }
            }
        }
        let denom = (a.len() + b.len() - 2) as f64;
        if denom <= 0.0 {
            return Err(LandfoldError::Msg("fisher class counts underflowed".into()));
        }
        sw /= denom;
        for i in 0..d {
            sw[(i, i)] += ridge;
        }
        let prec = sw.try_inverse().ok_or_else(|| {
            LandfoldError::Msg("fisher pooled covariance is singular".into())
        })?;
        if prec.iter().any(|v| !v.is_finite()) {
            return Err(LandfoldError::Msg(
                "fisher precision is non-finite".into(),
            ));
        }
        let mut packed = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                packed[i * d + j] = prec[(i, j)];
            }
        }
        Ok(Self {
            dim: d,
            prec: packed,
        })
    }
}

impl Metric for Fisher {
    fn dim(&self) -> Option<usize> {
        Some(self.dim)
    }

    fn dist_unchecked(&self, a: &[f64], b: &[f64]) -> f64 {
        let d = self.dim;
        let mut diff = vec![0.0; d];
        for i in 0..d {
            diff[i] = a[i] - b[i];
        }
        let mut acc = 0.0;
        for i in 0..d {
            let mut s = 0.0;
            for j in 0..d {
                s += self.prec[i * d + j] * diff[j];
            }
            acc += diff[i] * s;
        }
        if !acc.is_finite() || acc < 0.0 {
            return f64::INFINITY;
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
            let dx = periodic_delta(a[i], b[i], self.periods[i]);
            acc += dx * dx;
        }
        acc.sqrt()
    }

    fn diff(&self, a: &[f64], b: &[f64], c: &mut [f64]) {
        for i in 0..a.len() {
            c[i] = periodic_delta(a[i], b[i], self.periods[i]);
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
        let mut xs = 1.0;
        let mut ys = 1.0;
        let mut xy = 0.0;
        for i in 0..n {
            let a_phase = periodic_phase(a[i], self.periods[i]);
            let b_phase = periodic_phase(b[i], self.periods[i]);
            let xi = xs * a_phase.cos();
            let yi = ys * b_phase.cos();
            xs *= a_phase.sin();
            ys *= b_phase.sin();
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

/// L1 (Manhattan) distance. Packing-family DECAF histograms compare
/// this way: same family iff the L1 is at most the packing merge.
#[derive(Clone, Copy, Debug, Default)]
pub struct L1;

impl Metric for L1 {
    fn dist(&self, a: &[f64], b: &[f64]) -> Result<f64> {
        let distance = self.dist_unchecked(a, b);
        validate_distance(distance)?;
        Ok(distance)
    }

    fn dist_unchecked(&self, a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().max(b.len());
        let mut acc = 0.0;
        for i in 0..n {
            let left = if i < a.len() { a[i] } else { 0.0 };
            let right = if i < b.len() { b[i] } else { 0.0 };
            acc += (left - right).abs();
        }
        acc
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
    fn l1_pads_the_shorter_histogram() {
        let m = L1;
        assert_relative_eq!(m.dist(&[0.5, 0.5], &[1.0]).unwrap(), 1.0, epsilon = 1e-15);
    }

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
    fn stretch_increases_distance_along_the_axis() {
        let m = Stretch::from_refs(&[0.0, 0.0], &[1.0, 0.0], 3.0).unwrap();
        let along = m.dist(&[0.0, 0.0], &[1.0, 0.0]).unwrap();
        let across = m.dist(&[0.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!(along > 1.9);
        assert_relative_eq!(across, 1.0, epsilon = 1e-14);
        assert!(Stretch::new(vec![0.0, 0.0], 1.0).is_err());
    }

    #[test]
    fn fisher_shrinks_the_long_in_class_direction() {
        let mut pts = Vec::new();
        for i in 0..6 {
            pts.push(vec![i as f64 * 0.1, 0.0]);
            pts.push(vec![10.0 + i as f64 * 0.1, 0.0]);
        }
        let m = Fisher::from_refs(&pts, &[0.2, 0.0], &[10.2, 0.0], 1e-3).unwrap();
        let along_class = m.dist(&[0.0, 0.0], &[0.5, 0.0]).unwrap();
        let between = m.dist(&[0.2, 0.0], &[10.2, 0.0]).unwrap();
        assert!(between > 5.0 * along_class);
        assert!(Fisher::from_refs(&pts[..2], &[0.0, 0.0], &[10.0, 0.0], 1e-3).is_err());
    }

    #[test]
    fn euclid_accepts_large_finite_separations() {
        let m = Euclid;
        assert_relative_eq!(
            m.dist(&[1.0e200], &[-1.0e200]).unwrap(),
            2.0e200,
            max_relative = 1e-14
        );
    }

    #[test]
    fn pbc_wraps() {
        let m = Periodic::isotropic(1, 1.0).unwrap();
        assert_relative_eq!(m.dist(&[0.05], &[0.95]).unwrap(), 0.1, epsilon = 1e-14);
    }

    #[test]
    fn pbc_accepts_finite_extreme_coordinates() {
        let m = Periodic::isotropic(1, 3.0).unwrap();
        let distance = m.dist(&[f64::MAX], &[-f64::MAX]).unwrap();
        assert!(distance.is_finite());
        let mut diff = [0.0];
        m.diff(&[f64::MAX], &[-f64::MAX], &mut diff);
        assert!(diff[0].is_finite());
        assert!(diff[0].abs() <= 1.5);
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
    fn sphere_accepts_finite_coordinates_with_small_periods() {
        let m = Sphere::new(vec![1.0e-300, 1.0e-300]).unwrap();
        let distance = m.dist(&[f64::MAX, f64::MAX], &[-f64::MAX, -f64::MAX]);
        assert!(distance.is_ok_and(|value| value.is_finite()));
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

    struct InvalidMetric {
        distance: f64,
    }

    impl Metric for InvalidMetric {
        fn dist_unchecked(&self, _a: &[f64], _b: &[f64]) -> f64 {
            self.distance
        }
    }

    #[test]
    fn rejects_invalid_metric_distances() {
        for distance in [f64::NAN, f64::INFINITY, -1.0] {
            assert!(InvalidMetric { distance }.dist(&[0.0], &[1.0]).is_err());
        }
    }
}
