//! Bound-constrained L-BFGS quadratic model on χ, solved by HiGHS.
//!
//! HiGHS is the LP / MIP / convex-QP solver of Huangfu and Hall,
//! *Math. Prog. Comp.* **10**, 119 (2018),
//! <https://doi.org/10.1007/s12532-017-0130-5>. Each step minimises
//! `g·p + (1/2) p^T B p` where `B` is the compact L-BFGS Hessian
//! (Nocedal-Wright 7.19), subject to an L_inf trust region, optional
//! box bounds on the coordinates, and optional centering
//! `sum_i p_{i,h} = 0`. Extra arm: the published default remains
//! unconstrained Polak-Ribiere CG.

use ndarray::ArrayView1;
use xtsci_optimize::{HighsStep, Lbfgs};

use crate::cg::{CgReport, validate_packed_init};
use crate::error::{LandfoldError, Result};
use crate::stress::Stress;

#[derive(Clone, Debug)]
pub struct HighsOpts {
    pub maxiter: usize,
    /// L_inf trust radius on the step.
    pub trust: f64,
    /// Optional box lower bound on every coordinate.
    pub lo: Option<f64>,
    /// Optional box upper bound on every coordinate.
    pub hi: Option<f64>,
    /// Keep the centre of mass (one equality per low-D axis).
    pub center: bool,
}

impl Default for HighsOpts {
    fn default() -> Self {
        Self {
            maxiter: 40,
            trust: 0.5,
            lo: None,
            hi: None,
            center: true,
        }
    }
}

impl HighsOpts {
    fn validate(&self) -> Result<()> {
        if !self.trust.is_finite() || self.trust <= 0.0 {
            return Err(LandfoldError::Msg(
                "HiGHS trust radius must be finite and > 0".into(),
            ));
        }
        if self
            .lo
            .into_iter()
            .chain(self.hi)
            .any(|value| !value.is_finite())
        {
            return Err(LandfoldError::Msg("HiGHS bounds must be finite".into()));
        }
        if let (Some(lo), Some(hi)) = (self.lo, self.hi)
            && lo > hi
        {
            return Err(LandfoldError::Msg(
                "HiGHS lower bound must not exceed upper bound".into(),
            ));
        }
        Ok(())
    }

    fn step(&self, d: usize, n_atoms: usize, trust: f64) -> HighsStep {
        let mut equalities = Vec::new();
        if self.center {
            for h in 0..d {
                let coeffs: Vec<_> = (0..n_atoms).map(|i| (i * d + h, 1.0)).collect();
                equalities.push((coeffs, 0.0));
            }
        }
        HighsStep {
            trust: Some(trust),
            lo: self.lo,
            hi: self.hi,
            equalities,
        }
    }
}

/// Sequential L-BFGS-QP steps on packed χ. Each QP is solved by HiGHS.
pub fn minimize_highs(
    stress: &Stress,
    init: ArrayView1<f64>,
    d: usize,
    opts: &HighsOpts,
) -> Result<CgReport> {
    validate_packed_init(init, stress.n, d)?;
    opts.validate()?;
    let mut pos = init.to_owned();
    for v in pos.iter_mut() {
        if let Some(b) = opts.lo {
            *v = v.max(b);
        }
        if let Some(b) = opts.hi {
            *v = v.min(b);
        }
    }
    let mut ev = stress.try_eval(pos.view(), d)?;
    let mut steps = 0;
    let trust0 = opts.trust.max(1e-8);
    let mut trust = trust0;
    let mut lbfgs = Lbfgs::with_capacity(8);
    let n_atoms = pos.len() / d;
    for _ in 0..opts.maxiter {
        let gnorm: f64 = ev.grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gnorm < 1e-8 {
            break;
        }
        lbfgs.highs = Some(opts.step(d, n_atoms, trust));
        let step = lbfgs
            .highs_step(pos.view(), ev.grad.view())
            .map_err(|e| LandfoldError::Optimize(format!("HiGHS {e}")))?;
        let mut t = 1.0;
        let mut accepted = false;
        for _ in 0..8 {
            let mut trial = pos.clone();
            for i in 0..trial.len() {
                trial[i] += t * step[i];
            }
            if let Some(b) = opts.lo {
                for v in trial.iter_mut() {
                    *v = v.max(b);
                }
            }
            if let Some(b) = opts.hi {
                for v in trial.iter_mut() {
                    *v = v.min(b);
                }
            }
            let ev1 = stress.try_eval(trial.view(), d)?;
            if ev1.value < ev.value {
                let s = &trial - &pos;
                let y = &ev1.grad - &ev.grad;
                lbfgs.record(s, y);
                pos = trial;
                ev = ev1;
                accepted = true;
                steps += 1;
                break;
            }
            t *= 0.5;
        }
        if accepted {
            trust = (trust * 1.2).min(trust0 * 4.0);
        } else {
            trust *= 0.5;
            if trust < 1e-10 {
                break;
            }
        }
    }
    Ok(CgReport {
        value: ev.value,
        coords: pos,
        steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pairwise::{apply_transfer, pairwise_euclid};
    use crate::transfer::Transfer;
    use ndarray::{Array, array};

    #[test]
    fn highs_lowers_or_matches_init() {
        let pts = array![[0.0, 0.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]];
        let hd = pairwise_euclid(pts.view()).unwrap();
        let t = Transfer::xsigmoid(1.0, 4.0, 2.0).unwrap();
        let mut fhd = hd.clone();
        apply_transfer(&mut fhd, &t).unwrap();
        let s = Stress::new(hd, fhd, t, 0.0, None, None).unwrap();
        let init = Array::from_iter([0.0, 0.0, 0.2, 0.1, -0.1, 0.3, 0.4, -0.2]);
        let ev0 = s.eval(init.view(), 2);
        let ho = HighsOpts {
            maxiter: 20,
            lo: Some(-2.0),
            hi: Some(2.0),
            ..HighsOpts::default()
        };
        let rep = minimize_highs(&s, init.view(), 2, &ho).unwrap();
        assert!(rep.value <= ev0.value + 1e-12);
        for v in rep.coords.iter() {
            assert!(*v >= -2.0 - 1e-9 && *v <= 2.0 + 1e-9);
        }
    }

    #[test]
    fn highs_clips_infeasible_start_into_the_box() {
        let pts = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let hd = pairwise_euclid(pts.view()).unwrap();
        let t = Transfer::identity();
        let mut fhd = hd.clone();
        apply_transfer(&mut fhd, &t).unwrap();
        let s = Stress::new(hd, fhd, t, 0.0, None, None).unwrap();
        let init = Array::from_iter([5.0, -5.0, 4.0, 4.0, -3.0, 3.0, 2.0, -2.0]);
        let ho = HighsOpts {
            maxiter: 15,
            lo: Some(-0.3),
            hi: Some(0.3),
            ..HighsOpts::default()
        };
        let rep = minimize_highs(&s, init.view(), 2, &ho).unwrap();
        for v in rep.coords.iter() {
            assert!(
                *v >= -0.3 - 1e-9 && *v <= 0.3 + 1e-9,
                "coord {v} left the box"
            );
        }
    }

    #[test]
    fn rejects_invalid_options_and_transfer_overflow() {
        let hd = array![[0.0, 1.0], [1.0, 0.0]];
        let stress = Stress::new(hd.clone(), hd, Transfer::identity(), 1.0, None, None).unwrap();
        assert!(
            minimize_highs(
                &stress,
                array![0.0, 0.0].view(),
                1,
                &HighsOpts {
                    trust: f64::NAN,
                    ..HighsOpts::default()
                }
            )
            .is_err()
        );

        let mut stress = stress;
        stress.tfun_ld = Transfer::xsigmoid(1.0, 8.0, 1.0).unwrap();
        assert!(
            minimize_highs(
                &stress,
                array![0.0, 1.0e154].view(),
                1,
                &HighsOpts::default()
            )
            .is_err()
        );
    }
}
