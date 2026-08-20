//! Length-scale suggestion from a high-D pairwise histogram.
//!
//! Ceriotti, Tribello and Parrinello, *J. Chem. Theory Comput.* **9**,
//! 1521 (2013), Appendix A: \(\sigma\) is the mid-scale between short
//! thermal (Gaussian-in-\(D\)) distances and long uniform-in-\(D\)
//! distances. The knee of the empirical CDF is that mid-scale.

use ndarray::{ArrayView1, ArrayView2};

use crate::error::{LandfoldError, Result};
use crate::transfer::Transfer;

/// Quantiles and the CDF-knee suggestion for `--fun-hd` / `--fun-ld`.
#[derive(Clone, Copy, Debug)]
pub struct ScaleReport {
    pub n_pairs: usize,
    pub q25: f64,
    pub q50: f64,
    pub q75: f64,
    pub knee: f64,
}

/// Finite off-diagonal pairwise distances.
pub fn suggest_scale(distances: &[f64]) -> Result<ScaleReport> {
    let mut xs: Vec<f64> = distances
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .collect();
    if xs.len() < 3 {
        return Err(LandfoldError::Msg(
            "scale suggestion needs at least three positive pairwise distances".into(),
        ));
    }
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    let q25 = xs[(n * 25) / 100];
    let q50 = xs[n / 2];
    let q75 = xs[(n * 75) / 100];
    let knee = cdf_knee(&xs);
    Ok(ScaleReport {
        n_pairs: n,
        q25,
        q50,
        q75,
        knee,
    })
}

/// Maximum deviation of the empirical CDF from the chord of its endpoints.
fn cdf_knee(sorted: &[f64]) -> f64 {
    let first = sorted[0];
    let last = *sorted.last().unwrap();
    let span = last - first;
    if !(span > 0.0) {
        return first;
    }
    let n = sorted.len() as f64;
    let mut best_i = 0;
    let mut best_dev = -1.0;
    for (i, &x) in sorted.iter().enumerate() {
        let cdf = (i as f64 + 1.0) / n;
        let chord = (x - first) / span;
        let dev = (cdf - chord).abs();
        if dev > best_dev {
            best_dev = dev;
            best_i = i;
        }
    }
    sorted[best_i]
}

/// Posterior for the stretch weight.
///
/// Pair labels (same named class / different) are a Bernoulli with
/// success probability \(F(D(\alpha))\), the Ceriotti transfer already
/// used in χ. A flat prior on \(\alpha\ge 0\) gives a one-dimensional
/// posterior. `alpha` is the MAP; `alpha_lo` / `alpha_hi` are the 16th
/// and 84th percentiles. The plug-in median of \((\sigma^2-D_0^2)/p^2\)
/// is `alpha_plugin` and is biased high: it drops every between-class
/// pair that already sits past \(\sigma\).
#[derive(Clone, Copy, Debug)]
pub struct StretchReport {
    pub alpha: f64,
    pub alpha_lo: f64,
    pub alpha_hi: f64,
    pub alpha_plugin: f64,
    pub n_between: usize,
    pub n_within: usize,
    pub n_used: usize,
    pub median_within: f64,
    pub median_between: f64,
    pub sigma: f64,
}

pub fn suggest_alpha(
    points: ArrayView2<f64>,
    ref_a: ArrayView1<f64>,
    ref_b: ArrayView1<f64>,
    tfun: &Transfer,
) -> Result<StretchReport> {
    let sigma = tfun.xsigmoid_params().map(|(s, _, _)| s).unwrap_or(1.0);
    suggest_alpha_sigma(points, ref_a, ref_b, sigma, Some(tfun))
}

fn suggest_alpha_sigma(
    points: ArrayView2<f64>,
    ref_a: ArrayView1<f64>,
    ref_b: ArrayView1<f64>,
    sigma: f64,
    tfun: Option<&Transfer>,
) -> Result<StretchReport> {
    let n = points.nrows();
    let d = points.ncols();
    if n < 2 || d == 0 {
        return Err(LandfoldError::Empty);
    }
    if ref_a.len() != d || ref_b.len() != d {
        return Err(LandfoldError::MetricSize {
            left: d,
            right: ref_a.len(),
        });
    }
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(LandfoldError::Msg(
            "stretch suggestion needs a finite sigma > 0".into(),
        ));
    }
    if points.iter().chain(ref_a.iter()).chain(ref_b.iter()).any(|v| !v.is_finite())
    {
        return Err(LandfoldError::Msg(
            "stretch suggestion needs finite coordinates".into(),
        ));
    }
    let mut axis: Vec<f64> = (0..d).map(|k| ref_a[k] - ref_b[k]).collect();
    let n2: f64 = axis.iter().map(|v| v * v).sum();
    if !(n2 > 0.0 && n2.is_finite()) {
        return Err(LandfoldError::Msg(
            "stretch references must differ".into(),
        ));
    }
    let inv = 1.0 / n2.sqrt();
    for v in &mut axis {
        *v *= inv;
    }
    let mut lab = vec![false; n];
    for i in 0..n {
        let mut da = 0.0;
        let mut db = 0.0;
        for k in 0..d {
            let xa = points[(i, k)] - ref_a[k];
            let xb = points[(i, k)] - ref_b[k];
            da += xa * xa;
            db += xb * xb;
        }
        lab[i] = da <= db;
    }
    if lab.iter().all(|&t| t) || lab.iter().all(|&t| !t) {
        return Err(LandfoldError::Msg(
            "stretch suggestion needs points on both sides of the references".into(),
        ));
    }
    let mut within = Vec::new();
    let mut between = Vec::new();
    let mut alphas = Vec::new();
    let mut pair_w: Vec<(f64, f64)> = Vec::new();
    let mut pair_b: Vec<(f64, f64)> = Vec::new();
    let sig2 = sigma * sigma;
    for i in 0..n {
        for j in 0..i {
            let mut d0s = 0.0;
            let mut proj = 0.0;
            for k in 0..d {
                let delta = points[(i, k)] - points[(j, k)];
                d0s += delta * delta;
                proj += delta * axis[k];
            }
            let d0 = d0s.sqrt();
            if !d0.is_finite() {
                continue;
            }
            let p2 = proj * proj;
            if lab[i] == lab[j] {
                within.push(d0);
                if p2 > 0.0 {
                    pair_w.push((d0s, p2));
                }
            } else {
                between.push(d0);
                if p2 > 0.0 {
                    pair_b.push((d0s, p2));
                }
                if p2 > 0.0 && d0s < sig2 {
                    let a = (sig2 - d0s) / p2;
                    if a.is_finite() && a >= 0.0 {
                        alphas.push(a);
                    }
                }
            }
        }
    }
    if between.is_empty() {
        return Err(LandfoldError::Msg(
            "stretch suggestion found no between-class pairs".into(),
        ));
    }
    within.sort_by(f64::total_cmp);
    between.sort_by(f64::total_cmp);
    alphas.sort_by(f64::total_cmp);
    let alpha_plugin = if alphas.is_empty() {
        0.0
    } else {
        alphas[alphas.len() / 2]
    };
    let (alpha, alpha_lo, alpha_hi) = if let Some(tf) = tfun {
        map_alpha(&pair_w, &pair_b, tf)?
    } else {
        (alpha_plugin, alpha_plugin, alpha_plugin)
    };
    Ok(StretchReport {
        alpha,
        alpha_lo,
        alpha_hi,
        alpha_plugin,
        n_between: between.len(),
        n_within: within.len(),
        n_used: alphas.len(),
        median_within: median(&within),
        median_between: median(&between),
        sigma,
    })
}

/// MAP and 16/84 posterior percentiles of α under a flat prior on α≥0
/// and Bernoulli pair labels with success probability F(D(α)).
fn map_alpha(
    pair_w: &[(f64, f64)],
    pair_b: &[(f64, f64)],
    tfun: &Transfer,
) -> Result<(f64, f64, f64)> {
    const NGRID: usize = 201;
    const AMAX: f64 = 20.0;
    let mut nll = vec![0.0; NGRID];
    let mut best = 0usize;
    let mut best_v = f64::INFINITY;
    for g in 0..NGRID {
        let alpha = AMAX * (g as f64) / ((NGRID - 1) as f64);
        let mut v = 0.0;
        for &(d0s, p2) in pair_b {
            let d = (d0s + alpha * p2).sqrt();
            let f = tfun.f(d).clamp(1e-12, 1.0 - 1e-12);
            v -= f.ln();
        }
        for &(d0s, p2) in pair_w {
            let d = (d0s + alpha * p2).sqrt();
            let f = tfun.f(d).clamp(1e-12, 1.0 - 1e-12);
            v -= (1.0 - f).ln();
        }
        if !v.is_finite() {
            v = f64::INFINITY;
        }
        nll[g] = v;
        if v < best_v {
            best_v = v;
            best = g;
        }
    }
    if !best_v.is_finite() {
        return Err(LandfoldError::Msg(
            "stretch posterior is non-finite".into(),
        ));
    }
    let mut post = vec![0.0; NGRID];
    let mut z = 0.0;
    for g in 0..NGRID {
        let w = (-(nll[g] - best_v)).exp();
        post[g] = w;
        z += w;
    }
    if !(z > 0.0 && z.is_finite()) {
        return Err(LandfoldError::Msg(
            "stretch posterior could not be normalised".into(),
        ));
    }
    for w in &mut post {
        *w /= z;
    }
    let mut cdf = 0.0;
    let mut lo = 0.0;
    let mut hi = AMAX;
    let mut seen_lo = false;
    for g in 0..NGRID {
        let alpha = AMAX * (g as f64) / ((NGRID - 1) as f64);
        cdf += post[g];
        if !seen_lo && cdf >= 0.16 {
            lo = alpha;
            seen_lo = true;
        }
        if cdf >= 0.84 {
            hi = alpha;
            break;
        }
    }
    let map = AMAX * (best as f64) / ((NGRID - 1) as f64);
    Ok((map, lo, hi))
}

fn median(sorted: &[f64]) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[sorted.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn knee_sits_between_a_tight_cluster_and_a_far_tail() {
        let mut d = Vec::new();
        for _ in 0..80 {
            d.push(1.0);
        }
        for _ in 0..20 {
            d.push(10.0);
        }
        let s = suggest_scale(&d).unwrap();
        assert!(s.knee >= 1.0 && s.knee <= 10.0);
        assert_relative_eq!(s.q50, 1.0, epsilon = 1e-12);
        assert_eq!(s.n_pairs, 100);
    }

    #[test]
    fn rejects_empty_and_nonpositive() {
        assert!(suggest_scale(&[]).is_err());
        assert!(suggest_scale(&[0.0, 0.0]).is_err());
    }

    #[test]
    fn map_alpha_is_positive_when_the_sigmoid_cannot_tell_the_classes_apart() {
        use crate::transfer::Transfer;
        use ndarray::array;
        let pts = array![[0.0, 0.0], [0.1, 0.0], [3.0, 0.0], [3.1, 0.0]];
        let tf = Transfer::xsigmoid(5.0, 8.0, 1.0).unwrap();
        let s = suggest_alpha(pts.view(), array![0.0, 0.0].view(), array![3.0, 0.0].view(), &tf)
            .unwrap();
        assert!(s.alpha > 0.0);
        assert!(s.alpha_lo <= s.alpha && s.alpha <= s.alpha_hi);
        assert!(s.n_between >= 1);
    }

    #[test]
    fn map_alpha_is_near_zero_when_classes_are_already_split_by_f() {
        use crate::transfer::Transfer;
        use ndarray::array;
        let pts = array![[0.0, 0.0], [0.1, 0.0], [20.0, 0.0], [20.1, 0.0]];
        let tf = Transfer::xsigmoid(5.0, 8.0, 1.0).unwrap();
        let s = suggest_alpha(
            pts.view(),
            array![0.0, 0.0].view(),
            array![20.0, 0.0].view(),
            &tf,
        )
        .unwrap();
        assert!(s.alpha <= 1.0);
        assert!(s.median_between > 5.0);
    }
}
