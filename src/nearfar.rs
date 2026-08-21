//! Split isometric embedding: near-to-near, far-to-far, Riesz uniform.
//!
//! Ceriotti's sigmoid `F` saturates: every large `D` maps to 1, so far
//! pairs are indistinguishable in χ (PNAS 2011). The published path
//! uses `imix = 0` and therefore throws the tail away. This arm keeps
//! two identity bands and a Riesz energy:
//!
//! `L = Σ_{D≤σ} (d-D)² + λ Σ_{D≥τ} (d-D)² + μ Σ_{i<j} 1/(d²+ε)`
//!
//! The first sum is a local isometry (Kruskal 1964). The second is a
//! far isometry on the tail Ceriotti saturates. The third is the Riesz
//! `s=2` energy whose minimisers on a compact set are Fekete-type
//! configurations (Saff and Kuijlaars, *Math. Intelligencer* **19**,
//! 5 (1997)). No rank-CDF flatten: that map is not an isometry.

use ndarray::{Array2, ArrayView2};

use crate::error::{LandfoldError, Result};
use crate::mds::classical_mds;
use crate::metric::Metric;
use crate::pairwise::pairwise;
use crate::scale::suggest_scale;

#[derive(Clone, Debug)]
pub struct NearFarOpts {
    pub lowdim: usize,
    pub near: Option<f64>,
    pub far: Option<f64>,
    pub far_weight: f64,
    pub riesz: f64,
    pub steps: usize,
    pub lr: f64,
}

impl Default for NearFarOpts {
    fn default() -> Self {
        Self {
            lowdim: 2,
            near: None,
            far: None,
            far_weight: 1.0,
            riesz: 0.05,
            steps: 400,
            lr: 0.05,
        }
    }
}

/// Identity match on the near and far bands of `D`, plus Riesz `s=2`.
pub fn nearfar_embed(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    opts: &NearFarOpts,
) -> Result<(Array2<f64>, f64, f64)> {
    let n = points.nrows();
    if n < 3 {
        return Err(LandfoldError::Msg("near-far embed needs at least three points".into()));
    }
    if opts.lowdim == 0 || opts.lowdim > n {
        return Err(LandfoldError::LowDim {
            low: opts.lowdim,
            high: n,
        });
    }
    if !(opts.far_weight >= 0.0 && opts.riesz >= 0.0 && opts.lr > 0.0)
        || !opts.far_weight.is_finite()
        || !opts.riesz.is_finite()
        || !opts.lr.is_finite()
    {
        return Err(LandfoldError::Msg(
            "near-far weights and step must be finite and nonnegative".into(),
        ));
    }
    let hd = pairwise(points, metric)?;
    let mut ds = Vec::new();
    for i in 0..n {
        for j in 0..i {
            let d = hd[(i, j)];
            if d.is_finite() && d > 0.0 {
                ds.push(d);
            }
        }
    }
    let scale = suggest_scale(&ds)?;
    let sigma = opts.near.unwrap_or(scale.q25);
    let tau = opts.far.unwrap_or(scale.q75);
    if !(sigma > 0.0 && tau > sigma) {
        return Err(LandfoldError::Msg(
            "near cut must be positive and strictly below the far cut".into(),
        ));
    }
    let (mut y, _) = classical_mds(hd.view(), opts.lowdim)?;
    let mut m1 = Array2::<f64>::zeros((n, opts.lowdim));
    let mut m2 = Array2::<f64>::zeros((n, opts.lowdim));
    const BETA1: f64 = 0.9;
    const BETA2: f64 = 0.999;
    const EPS: f64 = 1e-8;
    const SOFT: f64 = 1e-8;
    for step in 0..opts.steps {
        let mut grad = Array2::<f64>::zeros((n, opts.lowdim));
        for i in 0..n {
            for j in 0..i {
                let dij = {
                    let mut s = 0.0;
                    for h in 0..opts.lowdim {
                        let e = y[(i, h)] - y[(j, h)];
                        s += e * e;
                    }
                    s.sqrt()
                };
                let d = dij.max(SOFT);
                let mut coeff = 0.0;
                let hdij = hd[(i, j)];
                if hdij <= sigma {
                    coeff += 2.0 * (d - hdij) / d;
                }
                if hdij >= tau {
                    coeff += opts.far_weight * 2.0 * (d - hdij) / d;
                }
                if opts.riesz > 0.0 {
                    let d2e = d * d + SOFT;
                    coeff -= opts.riesz * 2.0 / (d2e * d2e);
                }
                if !coeff.is_finite() {
                    continue;
                }
                for h in 0..opts.lowdim {
                    let e = y[(i, h)] - y[(j, h)];
                    grad[(i, h)] += coeff * e;
                    grad[(j, h)] -= coeff * e;
                }
            }
        }
        let t = (step + 1) as f64;
        let bc1 = 1.0 - BETA1.powf(t);
        let bc2 = 1.0 - BETA2.powf(t);
        for i in 0..n {
            for h in 0..opts.lowdim {
                let g = grad[(i, h)];
                m1[(i, h)] = BETA1 * m1[(i, h)] + (1.0 - BETA1) * g;
                m2[(i, h)] = BETA2 * m2[(i, h)] + (1.0 - BETA2) * g * g;
                let mh = m1[(i, h)] / bc1;
                let vh = m2[(i, h)] / bc2;
                y[(i, h)] -= opts.lr * mh / (vh.sqrt() + EPS);
                if !y[(i, h)].is_finite() {
                    return Err(LandfoldError::Msg(
                        "near-far coordinate is not finite".into(),
                    ));
                }
            }
        }
    }
    Ok((y, sigma, tau))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::Euclid;
    use ndarray::Array2;

    #[test]
    fn far_band_keeps_two_blobs_apart() {
        let mut pts = Array2::<f64>::zeros((16, 4));
        for i in 0..8 {
            pts[(i, 0)] = 0.02 * i as f64;
            pts[(i + 8, 0)] = 8.0 + 0.02 * i as f64;
        }
        let (y, sigma, tau) = nearfar_embed(
            pts.view(),
            &Euclid,
            &NearFarOpts {
                steps: 120,
                riesz: 0.01,
                ..NearFarOpts::default()
            },
        )
        .unwrap();
        assert!(sigma < tau);
        let mut ca = [0.0, 0.0];
        let mut cb = [0.0, 0.0];
        for i in 0..8 {
            ca[0] += y[(i, 0)];
            ca[1] += y[(i, 1)];
            cb[0] += y[(i + 8, 0)];
            cb[1] += y[(i + 8, 1)];
        }
        let gap = ((ca[0] - cb[0]).hypot(ca[1] - cb[1])) / 8.0;
        assert!(gap > 1.0, "near-far gap {gap}");
    }
}
