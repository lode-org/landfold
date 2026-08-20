//! Gap-split embedding.
//!
//! Take the slow mode `ψ` of a locally scaled diffusion operator
//! (Coifman and Lafon, *Appl. Comput. Harmon. Anal.* **21**, 5 (2006);
//! Rohrdanz, Zheng, Maggioni, Clementi, *J. Chem. Phys.* **134**, 124116
//! (2011)). Kill Ceriotti pair weights across a cut of `ψ`. The Lean
//! lemma `chi_decouples` says χ then splits into independent block
//! stresses; `display_separates` says the map `(ψ, s)` still separates
//! those blocks. This is the dual of stretch: stretch inserts a named
//! axis, gap-split removes the axis the sample already has.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{LandfoldError, Result};
use crate::iter::{embed, IterOpts};
use crate::metric::Metric;
use crate::phate::slow_mode;

#[derive(Clone, Debug)]
pub struct GapReport {
    pub tau: f64,
    pub n_kept: usize,
    pub n_pairs: usize,
}

/// Hard cut: `w_ij = 1` iff `|ψ_i - ψ_j| ≤ τ`.
pub fn gap_pair_weights(psi: ArrayView1<f64>, tau: f64) -> Result<Array2<f64>> {
    if !(tau >= 0.0) || !tau.is_finite() {
        return Err(LandfoldError::Msg("gap tau must be finite and nonnegative".into()));
    }
    let n = psi.len();
    if n == 0 {
        return Err(LandfoldError::Empty);
    }
    if psi.iter().any(|v| !v.is_finite()) {
        return Err(LandfoldError::Msg("slow mode must be finite".into()));
    }
    let mut w = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..i {
            let d = (psi[i] - psi[j]).abs();
            let wij = if d <= tau { 1.0 } else { 0.0 };
            w[(i, j)] = wij;
            w[(j, i)] = wij;
        }
    }
    Ok(w)
}

/// Default cut: 35% of the `ψ` range. Empty range is an error.
pub fn suggest_tau(psi: ArrayView1<f64>) -> Result<f64> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in psi {
        if !v.is_finite() {
            return Err(LandfoldError::Msg("slow mode must be finite".into()));
        }
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let span = hi - lo;
    if !(span > 0.0) {
        return Err(LandfoldError::Msg("slow mode is constant".into()));
    }
    Ok(0.35 * span)
}

/// Embed as `(ψ, s)`: `ψ` is the slow mode, `s` is 1-D Ceriotti χ on
/// pairs that the slow mode does not already separate.
pub fn gap_split_embed(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    opts: &IterOpts,
    knn: usize,
    decay: f64,
    tau: Option<f64>,
) -> Result<(Array2<f64>, GapReport)> {
    let n = points.nrows();
    let psi = slow_mode(points, metric, knn, decay)?;
    let tau = match tau {
        Some(t) => t,
        None => suggest_tau(psi.view())?,
    };
    let w = gap_pair_weights(psi.view(), tau)?;
    let mut n_kept = 0usize;
    let mut n_pairs = 0usize;
    for i in 0..n {
        for j in 0..i {
            n_pairs += 1;
            if w[(i, j)] > 0.0 {
                n_kept += 1;
            }
        }
    }
    if n_kept == 0 {
        return Err(LandfoldError::Msg(
            "gap-split kept no pairs; increase tau".into(),
        ));
    }
    let mut chi_opts = opts.clone();
    chi_opts.lowdim = 1;
    chi_opts.pair_weights = Some(w);
    chi_opts.midweight = false;
    let (emb, _) = embed(points, metric, &chi_opts, None, None, None)?;
    let mut coords = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        coords[(i, 0)] = psi[i];
        coords[(i, 1)] = emb.low[(i, 0)];
    }
    Ok((
        coords,
        GapReport {
            tau,
            n_kept,
            n_pairs,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::Euclid;
    use crate::transfer::Transfer;

    #[test]
    fn cut_kills_cross_pairs() {
        let psi = Array1::from(vec![-1.0, -0.9, 1.0, 1.1]);
        let w = gap_pair_weights(psi.view(), 0.3).unwrap();
        assert_eq!(w[(0, 1)], 1.0);
        assert_eq!(w[(2, 3)], 1.0);
        assert_eq!(w[(0, 2)], 0.0);
        assert_eq!(w[(1, 3)], 0.0);
    }

    #[test]
    fn two_blobs_get_a_slow_axis() {
        let mut pts = Array2::<f64>::zeros((16, 4));
        for i in 0..8 {
            pts[(i, 0)] = 0.02 * i as f64;
            pts[(i + 8, 0)] = 5.0 + 0.02 * i as f64;
        }
        let opts = IterOpts {
            tfun_hd: Transfer::xsigmoid(2.0, 4.0, 3.0).unwrap(),
            tfun_ld: Transfer::xsigmoid(2.0, 2.0, 3.0).unwrap(),
            ..IterOpts::default()
        };
        let (y, rep) = gap_split_embed(pts.view(), &Euclid, &opts, 3, 8.0, None).unwrap();
        assert!(rep.n_kept > 0);
        assert_eq!(y.nrows(), 16);
        let mut ca = 0.0;
        let mut cb = 0.0;
        for i in 0..8 {
            ca += y[(i, 0)];
            cb += y[(i + 8, 0)];
        }
        assert!((ca / 8.0 - cb / 8.0).abs() > 1e-6);
    }
}
