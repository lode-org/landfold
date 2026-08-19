//! Bound-constrained sequential linear program on χ, solved by HiGHS.
//!
//! HiGHS is the LP / MIP / convex-QP solver of Huangfu and Hall,
//! *Math. Prog. Comp.* **10**, 119 (2018),
//! <https://doi.org/10.1007/s12532-017-0130-5>. Each step minimises
//! `g·p` subject to an L_inf trust region, optional box bounds on the
//! coordinates, and optional centering `sum_i p_{i,h} = 0`. Extra arm:
//! the published default remains unconstrained Polak-Ribiere CG.

use highs::{RowProblem, Sense};
use ndarray::{Array1, ArrayView1};

use crate::cg::CgReport;
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

/// Sequential LP steps on packed χ. Each LP is solved by HiGHS.
pub fn minimize_highs(
    stress: &Stress,
    init: ArrayView1<f64>,
    d: usize,
    opts: &HighsOpts,
) -> Result<CgReport> {
    let mut pos = init.to_owned();
    for v in pos.iter_mut() {
        if let Some(b) = opts.lo {
            *v = v.max(b);
        }
        if let Some(b) = opts.hi {
            *v = v.min(b);
        }
    }
    let mut ev = stress.eval(pos.view(), d);
    let mut steps = 0;
    let trust0 = opts.trust.max(1e-8);
    let mut trust = trust0;
    for _ in 0..opts.maxiter {
        let gnorm: f64 = ev.grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gnorm < 1e-8 {
            break;
        }
        let step = slp_step(&pos, ev.grad.view(), d, trust, opts)?;
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
            let ev1 = stress.eval(trial.view(), d);
            if ev1.value < ev.value {
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

fn slp_step(
    x: &Array1<f64>,
    g: ArrayView1<f64>,
    d: usize,
    trust: f64,
    opts: &HighsOpts,
) -> Result<Array1<f64>> {
    let nv = x.len();
    let n = nv / d;
    let mut pb = RowProblem::default();
    let mut cols = Vec::with_capacity(nv);
    for k in 0..nv {
        let mut lo = -trust;
        let mut hi = trust;
        if let Some(b) = opts.lo {
            lo = lo.max(b - x[k]);
        }
        if let Some(b) = opts.hi {
            hi = hi.min(b - x[k]);
        }
        if lo > hi {
            lo = hi;
        }
        cols.push(pb.add_column(g[k], lo..=hi));
    }
    if opts.center {
        for h in 0..d {
            let row: Vec<_> = (0..n).map(|i| (cols[i * d + h], 1.0)).collect();
            pb.add_row(0.0..=0.0, &row);
        }
    }
    let mut model = pb.optimise(Sense::Minimise);
    model.make_quiet();
    let solved = model
        .try_solve()
        .map_err(|e| LandfoldError::Optimize(format!("HiGHS {e:?}")))?;
    let sol = solved.get_solution();
    let p = sol.columns();
    if p.len() != nv {
        return Err(LandfoldError::Optimize(
            "HiGHS returned the wrong column count".into(),
        ));
    }
    Ok(Array1::from(p.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pairwise::{apply_transfer, pairwise_euclid};
    use crate::transfer::Transfer;
    use ndarray::{array, Array};

    #[test]
    fn highs_lowers_or_matches_init() {
        let pts = array![[0.0, 0.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]];
        let hd = pairwise_euclid(pts.view()).unwrap();
        let t = Transfer::xsigmoid(1.0, 4.0, 2.0).unwrap();
        let mut fhd = hd.clone();
        apply_transfer(&mut fhd, &t).unwrap();
        let s = Stress::new(hd, fhd, t, 0.0, None, None);
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
        let s = Stress::new(hd, fhd, t, 0.0, None, None);
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
}
