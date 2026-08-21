//! Three-band embed: identity near, Ceriotti χ mid, identity far, Riesz.
//!
//! The published χ path (`imix = 0`) puts every pair through a
//! saturating `F`. Lean `sat_far_cannot_tell`: once `F(D) = 1`, two
//! different large `D` give the same residual. The near-far arm kept
//! identity on the tail and dropped the mid band, which is the scale
//! the two TSE lobes live on (Appendix A knee). This loss keeps all
//! three:
//!
//! `L = (1/N_≤σ) Σ_{D≤σ} (d-D)²
//!    + (λ/N_mid) Σ_{σ<D<τ} (F(D)-f(d))²
//!    + (ν/N_≥τ) Σ_{D≥τ} (d-D)²
//!    + (μ/N_≤σ) Σ_{D≤σ} 1/(d²+ε)`
//!
//! Near is Kruskal local isometry. Mid is Ceriotti χ on the
//! transfer-sensitive window. Far is an isometry of the diameter
//! (Lean `id_stress_separates`). Riesz `s=2` spaces *inside* the
//! near band only (Saff-Kuijlaars). A global Riesz sum sphericalizes
//! the map. No rank-CDF flatten.

use ndarray::{Array2, ArrayView2};

use crate::error::{LandfoldError, Result};
use crate::mds::classical_mds;
use crate::metric::Metric;
use crate::pairwise::pairwise;
use crate::scale::suggest_scale;
use crate::transfer::Transfer;

#[derive(Clone, Debug)]
pub struct BandOpts {
    pub lowdim: usize,
    pub near: Option<f64>,
    pub far: Option<f64>,
    pub tfun_hd: Transfer,
    pub tfun_ld: Transfer,
    pub mid_weight: f64,
    pub far_weight: f64,
    pub riesz: f64,
    pub steps: usize,
    pub lr: f64,
}

impl Default for BandOpts {
    fn default() -> Self {
        Self {
            lowdim: 2,
            near: None,
            far: None,
            tfun_hd: Transfer::identity(),
            tfun_ld: Transfer::identity(),
            mid_weight: 1.0,
            far_weight: 1.0,
            riesz: 0.05,
            steps: 500,
            lr: 0.08,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BandReport {
    pub sigma: f64,
    pub tau: f64,
    pub n_near: usize,
    pub n_mid: usize,
    pub n_far: usize,
}

fn resolve_mid(t: &Transfer, knee: f64, high: bool) -> Result<Transfer> {
    if t.mode() != crate::transfer::TransferMode::Identity {
        return Ok(t.clone());
    }
    if high {
        Transfer::xsigmoid(knee, 8.0, 1.0)
    } else {
        Transfer::xsigmoid(knee, 2.0, 2.0)
    }
}

/// Identity near, Ceriotti χ mid, identity far, Riesz `s=2`.
pub fn bands_embed(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    opts: &BandOpts,
) -> Result<(Array2<f64>, BandReport)> {
    let n = points.nrows();
    if n < 3 {
        return Err(LandfoldError::Msg(
            "three-band embed needs at least three points".into(),
        ));
    }
    if opts.lowdim == 0 || opts.lowdim > n {
        return Err(LandfoldError::LowDim {
            low: opts.lowdim,
            high: n,
        });
    }
    if !(opts.mid_weight >= 0.0 && opts.far_weight >= 0.0 && opts.riesz >= 0.0 && opts.lr > 0.0)
        || !opts.mid_weight.is_finite()
        || !opts.far_weight.is_finite()
        || !opts.riesz.is_finite()
        || !opts.lr.is_finite()
    {
        return Err(LandfoldError::Msg(
            "three-band weights and step must be finite and nonnegative".into(),
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
    let t_hd = resolve_mid(&opts.tfun_hd, scale.knee, true)?;
    let t_ld = resolve_mid(&opts.tfun_ld, scale.knee, false)?;
    let mut fhd = Array2::<f64>::zeros((n, n));
    let mut n_near = 0usize;
    let mut n_mid = 0usize;
    let mut n_far = 0usize;
    for i in 0..n {
        for j in 0..i {
            let d = hd[(i, j)];
            if d <= sigma {
                n_near += 1;
            } else if d >= tau {
                n_far += 1;
            } else {
                n_mid += 1;
                fhd[(i, j)] = t_hd.f(d);
                fhd[(j, i)] = fhd[(i, j)];
            }
        }
    }
    if n_near + n_mid + n_far == 0 {
        return Err(LandfoldError::Msg("three-band found no pairs".into()));
    }
    let w_near = if n_near > 0 { 1.0 / n_near as f64 } else { 0.0 };
    let w_mid = if n_mid > 0 {
        opts.mid_weight / n_mid as f64
    } else {
        0.0
    };
    let w_far = if n_far > 0 {
        opts.far_weight / n_far as f64
    } else {
        0.0
    };
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
                let mut s = 0.0;
                for h in 0..opts.lowdim {
                    let e = y[(i, h)] - y[(j, h)];
                    s += e * e;
                }
                let d = s.sqrt().max(SOFT);
                let hdij = hd[(i, j)];
                let mut coeff = 0.0;
                if hdij <= sigma {
                    coeff += w_near * 2.0 * (d - hdij) / d;
                    if opts.riesz > 0.0 {
                        let d2e = d * d + SOFT;
                        coeff -= w_near * opts.riesz * 2.0 / (d2e * d2e);
                    }
                } else if hdij >= tau {
                    coeff += w_far * 2.0 * (d - hdij) / d;
                } else {
                    let (fld, dfld) = t_ld.fdf(d);
                    coeff += -w_mid * 2.0 * (fhd[(i, j)] - fld) * dfld / d;
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
                        "three-band coordinate is not finite".into(),
                    ));
                }
            }
        }
    }
    Ok((
        y,
        BandReport {
            sigma,
            tau,
            n_near,
            n_mid,
            n_far,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::Euclid;
    use ndarray::Array2;

    #[test]
    fn far_identity_keeps_two_blobs_at_hd_gap() {
        let mut pts = Array2::<f64>::zeros((16, 4));
        for i in 0..8 {
            pts[(i, 0)] = 0.02 * i as f64;
            pts[(i + 8, 0)] = 8.0 + 0.02 * i as f64;
        }
        let (y, rep) = bands_embed(
            pts.view(),
            &Euclid,
            &BandOpts {
                steps: 150,
                riesz: 0.01,
                ..BandOpts::default()
            },
        )
        .unwrap();
        assert!(rep.sigma < rep.tau);
        assert!(rep.n_far > 0);
        assert!(rep.n_near > 0);
        let mut ca = [0.0, 0.0];
        let mut cb = [0.0, 0.0];
        for i in 0..8 {
            ca[0] += y[(i, 0)];
            ca[1] += y[(i, 1)];
            cb[0] += y[(i + 8, 0)];
            cb[1] += y[(i + 8, 1)];
        }
        let gap = ((ca[0] - cb[0]).hypot(ca[1] - cb[1])) / 8.0;
        assert!(gap > 4.0, "three-band far gap {gap}");
    }

    #[test]
    fn mid_band_sees_the_in_between_scale() {
        let mut pts = Array2::<f64>::zeros((12, 2));
        for i in 0..12 {
            pts[(i, 0)] = 0.6 * i as f64;
        }
        let (_, rep) = bands_embed(
            pts.view(),
            &Euclid,
            &BandOpts {
                steps: 40,
                riesz: 0.0,
                ..BandOpts::default()
            },
        )
        .unwrap();
        assert!(rep.n_mid > 0, "chain must populate the mid band");
    }
}
