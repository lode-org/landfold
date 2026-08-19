//! χ as an eindir `Objective` + `Gradient`.
//!
//! The kernel stays in [`crate::stress::Stress`]. This type is the typed
//! `S -> R` handle that eindir consumers (and the xtsci-optimize port)
//! already know how to take. Solvers must not grow a second oracle shape.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

use crate::stress::{Stress, StressEval};

/// Wide box used when χ is treated as unconstrained.
pub const UNBOUNDED: f64 = 1e12;

/// Packed low-D χ: dimension `n * d`, analytic ∇χ from [`Stress::eval`].
pub struct ChiObjective<'a> {
    stress: &'a Stress,
    d: usize,
    bounds: Bounds<f64>,
}

impl<'a> ChiObjective<'a> {
    /// Unconstrained χ on packed coordinates of length `stress.n * d`.
    pub fn new(stress: &'a Stress, d: usize) -> Self {
        Self::with_box(stress, d, -UNBOUNDED, UNBOUNDED)
    }

    /// Axis-aligned box on every packed coordinate (HiGHS-style clip).
    pub fn with_box(stress: &'a Stress, d: usize, lo: f64, hi: f64) -> Self {
        let dim = stress.n * d;
        Self {
            stress,
            d,
            bounds: Bounds::new(Array1::from_elem(dim, lo), Array1::from_elem(dim, hi), 0.0),
        }
    }

    /// Embedding dimension baked into this objective.
    pub fn embedding_dim(&self) -> usize {
        self.d
    }

    /// Fused χ and ∇χ. Prefer this over separate `eval` + `grad`.
    pub fn eval_full(&self, x: ArrayView1<f64>) -> StressEval {
        self.stress.eval(x, self.d)
    }
}

impl Objective<f64> for ChiObjective<'_> {
    fn dim(&self) -> usize {
        self.stress.n * self.d
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.eval_full(x).value
    }
}

impl Gradient<f64> for ChiObjective<'_> {
    fn dim(&self) -> usize {
        self.stress.n * self.d
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.eval_full(x).grad
    }
}

impl DifferentiableObjective<f64> for ChiObjective<'_> {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let ev = self.eval_full(x);
        (ev.value, ev.grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pairwise::pairwise_euclid;
    use crate::transfer::Transfer;
    use approx::assert_relative_eq;
    use ndarray::{Array, array};

    fn toy() -> (Stress, Array1<f64>) {
        let pts = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let hd = pairwise_euclid(pts.view()).unwrap();
        let fhd = hd.clone();
        let s = Stress::new(hd, fhd, Transfer::identity(), 0.0, None, None);
        let coords = Array::from_iter(pts.iter().copied());
        (s, coords)
    }

    #[test]
    fn fused_matches_stress_eval() {
        let (s, coords) = toy();
        let obj = ChiObjective::new(&s, 2);
        assert_eq!(Objective::dim(&obj), 6);
        let ev = s.eval(coords.view(), 2);
        let (v, g) = obj.value_and_gradient(coords.view());
        assert_relative_eq!(v, ev.value, epsilon = 1e-14);
        for k in 0..6 {
            assert_relative_eq!(g[k], ev.grad[k], epsilon = 1e-14);
        }
    }

    #[test]
    fn box_bounds_clip_domain() {
        let (s, _) = toy();
        let obj = ChiObjective::with_box(&s, 2, -0.3, 0.3);
        let x = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(obj.bounds().contains(x.view()));
        let out = array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(!obj.bounds().contains(out.view()));
        let clipped = obj.bounds().clip(out.view());
        assert_relative_eq!(clipped[0], 0.3, epsilon = 1e-15);
    }
}
