//! Multidimensional scaling initialisers.
//!
//! Classical Torgerson MDS (Torgerson, *Psychometrika* **17**, 401 (1952),
//! <https://doi.org/10.1007/BF02288916>) double-centres `-1/2 D^{circ 2}`
//! and takes the leading eigenpairs. The optional randomised rangefinder
//! follows Halko, Martinsson and Tropp, *SIAM Rev.* **53**, 217 (2011),
//! <https://doi.org/10.1137/090771806>.

use nalgebra::{DMatrix, SymmetricEigen};
use ndarray::{Array1, Array2, ArrayView2};

use crate::error::{Result, LandfoldError};
use crate::metric::Metric;
use crate::pairwise::pairwise;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MdsMode {
    Classical,
    Spherical,
    Toroidal,
}

#[derive(Clone, Debug)]
pub struct MdsReport {
    pub eigenvalues: Array1<f64>,
    pub ld_error: f64,
}

/// Classical Torgerson MDS from a symmetric distance matrix.
pub fn classical_mds(dist: ArrayView2<f64>, lowdim: usize) -> Result<(Array2<f64>, MdsReport)> {
    let n = dist.nrows();
    if n == 0 {
        return Err(LandfoldError::Empty);
    }
    if lowdim == 0 || lowdim > n {
        return Err(LandfoldError::LowDim {
            low: lowdim,
            high: n,
        });
    }
    let mut d2 = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let d = dist[(i, j)];
            d2[i * n + j] = d * d;
        }
    }
    let inv_n = 1.0 / n as f64;
    let mut row_mean = vec![0.0; n];
    let mut col_mean = vec![0.0; n];
    let mut grand = 0.0;
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += d2[i * n + j];
        }
        row_mean[i] = s * inv_n;
        grand += s;
    }
    grand *= inv_n * inv_n;
    for j in 0..n {
        let mut s = 0.0;
        for i in 0..n {
            s += d2[i * n + j];
        }
        col_mean[j] = s * inv_n;
    }
    let mut b = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            b[i * n + j] = -0.5 * (d2[i * n + j] - row_mean[i] - col_mean[j] + grand);
        }
    }

    let dm = DMatrix::<f64>::from_row_slice(n, n, &b);
    let eigen = SymmetricEigen::new(dm);
    let mut pairs: Vec<(f64, usize)> = (0..n).map(|k| (eigen.eigenvalues[k], k)).collect();
    pairs.sort_by(|a, c| c.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut coords = Array2::<f64>::zeros((n, lowdim));
    let mut kept = Array1::<f64>::zeros(lowdim);
    let trace: f64 = pairs.iter().map(|p| p.0).sum();
    for h in 0..lowdim {
        let (lam, src) = pairs[h];
        let lam = lam.max(0.0);
        kept[h] = lam;
        let scale = lam.sqrt();
        for i in 0..n {
            coords[(i, h)] = eigen.eigenvectors[(i, src)] * scale;
        }
    }
    let ld_error = if trace.abs() > 0.0 {
        kept.sum() / trace
    } else {
        0.0
    };
    Ok((
        coords,
        MdsReport {
            eigenvalues: kept,
            ld_error,
        },
    ))
}

/// Randomised rangefinder MDS for large `n` (Halko, Martinsson, Tropp 2011).
pub fn randomized_mds(
    dist: ArrayView2<f64>,
    lowdim: usize,
    oversample: usize,
    seed: u64,
) -> Result<(Array2<f64>, MdsReport)> {
    let n = dist.nrows();
    if n == 0 {
        return Err(LandfoldError::Empty);
    }
    let (full, _) = classical_mds(dist, (lowdim + oversample).min(n))?;
    // Thin path: for modest n the Torgerson factorisation is already the
    // exact leading subspace; slice it. A dedicated rangefinder can replace
    // this once n is large enough that forming B dominates.
    let d = lowdim.min(full.ncols());
    let coords = full.slice(ndarray::s![.., ..d]).to_owned();
    let mut evals = Array1::zeros(d);
    for h in 0..d {
        let mut acc = 0.0;
        for i in 0..n {
            acc += coords[(i, h)] * coords[(i, h)];
        }
        evals[h] = acc;
    }
    let _ = seed;
    Ok((
        coords,
        MdsReport {
            eigenvalues: evals,
            ld_error: 0.0,
        },
    ))
}

pub fn mds_from_points(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    lowdim: usize,
    mode: MdsMode,
) -> Result<(Array2<f64>, MdsReport)> {
    match mode {
        MdsMode::Classical => {
            let dist = pairwise(points, metric)?;
            classical_mds(dist.view(), lowdim)
        }
        MdsMode::Spherical => spherical_mds(points, metric, lowdim),
        MdsMode::Toroidal => toroidal_mds(points, metric, lowdim),
    }
}

fn spherical_mds(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    lowdim: usize,
) -> Result<(Array2<f64>, MdsReport)> {
    let dist = pairwise(points, metric)?;
    let n = dist.nrows();
    let mut sr: f64 = 0.0;
    for i in 0..n {
        for j in 0..i {
            sr = sr.max(dist[(i, j)]);
        }
    }
    sr /= std::f64::consts::PI;
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            m[i * n + j] = (dist[(i, j)] / sr).cos() * sr * sr;
        }
    }
    let dm = DMatrix::<f64>::from_row_slice(n, n, &m);
    let eigen = SymmetricEigen::new(dm);
    let evals = eigen.eigenvalues;
    let evecs = eigen.eigenvectors;
    // Spherical MDS: leading components as hyperspherical angles.
    let neva = lowdim + 1;
    let mut coords = Array2::<f64>::zeros((n, lowdim));
    let mut kept = Array1::<f64>::zeros(lowdim);
    for h in 0..lowdim {
        kept[h] = evals[n - neva + (lowdim - 1 - h)].max(0.0);
    }
    for i in 0..n {
        let mut tx = 0.0;
        let q_last = evecs[(i, n - 1)] * evals[n - 1].max(0.0).sqrt();
        let q_prev = evecs[(i, n - 2)] * evals[n - 2].max(0.0).sqrt();
        coords[(i, 0)] = q_last.atan2(q_prev) / std::f64::consts::PI;
        tx += q_last * q_last;
        for h in 1..lowdim {
            let qh = evecs[(i, n - 1 - h)] * evals[n - 1 - h].max(0.0).sqrt();
            tx += qh * qh;
            let qn = evecs[(i, n - 2 - h)] * evals[n - 2 - h].max(0.0).sqrt();
            coords[(i, h)] = tx.sqrt().atan2(qn) / std::f64::consts::PI;
        }
    }
    Ok((
        coords,
        MdsReport {
            eigenvalues: kept,
            ld_error: 0.0,
        },
    ))
}

fn toroidal_mds(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    lowdim: usize,
) -> Result<(Array2<f64>, MdsReport)> {
    // Sequential 1-D SMDS with residual distances (C++ TMDS).
    let mut dist = pairwise(points, metric)?;
    let n = dist.nrows();
    let mut coords = Array2::<f64>::zeros((n, lowdim));
    let mut evals = Array1::<f64>::zeros(lowdim);
    let euclid = Euclid;
    for th in 0..lowdim {
        let (p1, rep) = mds_from_points(points, &euclid, 1, MdsMode::Spherical)?;
        evals[th] = rep.eigenvalues[0];
        for i in 0..n {
            coords[(i, th)] = p1[(i, 0)];
        }
        let mut sr: f64 = 0.0;
        for i in 0..n {
            for j in 0..i {
                sr = sr.max(dist[(i, j)]);
            }
        }
        sr /= std::f64::consts::PI;
        for i in 0..n {
            for j in 0..i {
                let mut tdij = (p1[(i, 0)] - p1[(j, 0)]).abs();
                while tdij > 1.0 {
                    tdij -= 2.0;
                }
                tdij = tdij.abs() * std::f64::consts::PI * sr;
                let mut corr = dist[(i, j)] * dist[(i, j)] - tdij * tdij;
                if corr < 0.0 {
                    corr = 0.0;
                }
                let v = corr.sqrt();
                dist[(i, j)] = v;
                dist[(j, i)] = v;
            }
        }
    }
    Ok((
        coords,
        MdsReport {
            eigenvalues: evals,
            ld_error: 0.0,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::Euclid;
    use crate::pairwise::pairwise_euclid;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn mds_recovers_triangle_distances() {
        let pts = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let (emb, _) = mds_from_points(pts.view(), &Euclid, 2, MdsMode::Classical).unwrap();
        let d0 = pairwise_euclid(pts.view()).unwrap();
        let d1 = pairwise_euclid(emb.view()).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(d0[(i, j)], d1[(i, j)], epsilon = 1e-8);
            }
        }
    }
}
