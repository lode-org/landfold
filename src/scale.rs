//! Length-scale suggestion from a high-D pairwise histogram.
//!
//! Ceriotti, Tribello and Parrinello, *J. Chem. Theory Comput.* **9**,
//! 1521 (2013), Appendix A: \(\sigma\) is the mid-scale between short
//! thermal (Gaussian-in-\(D\)) distances and long uniform-in-\(D\)
//! distances. The knee of the empirical CDF is that mid-scale.

use crate::error::{LandfoldError, Result};

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
}
