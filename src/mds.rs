//! Multidimensional scaling initialisers.
//!
//! Classical Torgerson MDS (Torgerson, *Psychometrika* **17**, 401 (1952),
//! <https://doi.org/10.1007/BF02288916>) double-centres `-1/2 D^{circ 2}`
//! and takes the leading eigenpairs. The optional randomised rangefinder
//! follows Halko, Martinsson and Tropp, *SIAM Rev.* **53**, 217 (2011),
//! <https://doi.org/10.1137/090771806>.

use nalgebra::{DMatrix, QR, SymmetricEigen};
use ndarray::{Array1, Array2, ArrayView2};

use crate::error::{LandfoldError, Result};
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

/// Double-centred Gram `B = -1/2 H D^{circ 2} H` (Torgerson 1952).
fn checked_add(left: f64, right: f64, what: &'static str) -> Result<f64> {
    let value = left + right;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(LandfoldError::Msg(what.into()))
    }
}

fn torgerson_b(dist: ArrayView2<f64>) -> Result<(usize, Vec<f64>)> {
    let n = dist.nrows();
    if n == 0 {
        return Err(LandfoldError::Empty);
    }
    if dist.ncols() != n {
        return Err(LandfoldError::Shape("MDS distance matrix must be square"));
    }
    if dist.iter().any(|&value| !value.is_finite() || value < 0.0) {
        return Err(LandfoldError::Msg(
            "MDS distance matrix must be finite and nonnegative".into(),
        ));
    }
    let matrix_len = n
        .checked_mul(n)
        .ok_or(LandfoldError::Msg("MDS matrix dimension overflowed".into()))?;
    let mut d2 = vec![0.0; matrix_len];
    for i in 0..n {
        for j in 0..n {
            let d = dist[(i, j)];
            let squared = d * d;
            if !squared.is_finite() {
                return Err(LandfoldError::Msg(
                    "MDS distance squaring overflowed".into(),
                ));
            }
            d2[i * n + j] = squared;
        }
    }
    let inv_n = 1.0 / n as f64;
    let mut row_mean = vec![0.0; n];
    let mut col_mean = vec![0.0; n];
    let mut grand = 0.0;
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s = checked_add(
                s,
                d2[i * n + j],
                "MDS row-centering accumulation overflowed",
            )?;
        }
        row_mean[i] = s * inv_n;
        grand = checked_add(
            grand,
            s,
            "MDS grand-centering accumulation overflowed",
        )?;
    }
    grand *= inv_n * inv_n;
    if !grand.is_finite() {
        return Err(LandfoldError::Msg(
            "MDS grand-centering accumulation overflowed".into(),
        ));
    }
    for j in 0..n {
        let mut s = 0.0;
        for i in 0..n {
            s = checked_add(
                s,
                d2[i * n + j],
                "MDS column-centering accumulation overflowed",
            )?;
        }
        col_mean[j] = s * inv_n;
    }
    let mut b = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let value = -0.5 * (d2[i * n + j] - row_mean[i] - col_mean[j] + grand);
            if !value.is_finite() {
                return Err(LandfoldError::Msg(
                    "MDS centered Gram matrix overflowed".into(),
                ));
            }
            b[i * n + j] = value;
        }
    }
    Ok((n, b))
}

fn coords_from_eigen(
    n: usize,
    lowdim: usize,
    evals: &[f64],
    evecs: &DMatrix<f64>,
) -> Result<(Array2<f64>, Array1<f64>, f64)> {
    if evals.len() != n || evecs.nrows() != n || evecs.ncols() != n {
        return Err(LandfoldError::Shape("MDS eigensystem dimensions"));
    }
    if evals.iter().any(|value| !value.is_finite()) {
        return Err(LandfoldError::Msg("MDS eigenvalues must be finite".into()));
    }
    let mut pairs: Vec<(f64, usize)> = (0..n).map(|k| (evals[k], k)).collect();
    pairs.sort_by(|a, c| c.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut coords = Array2::<f64>::zeros((n, lowdim));
    let mut kept = Array1::<f64>::zeros(lowdim);
    let trace = pairs.iter().try_fold(0.0, |trace, pair| {
        let value = trace + pair.0;
        value
            .is_finite()
            .then_some(value)
            .ok_or_else(|| LandfoldError::Msg("MDS spectral trace overflowed".into()))
    })?;
    for h in 0..lowdim {
        let (lam, src) = pairs[h];
        let lam = lam.max(0.0);
        kept[h] = lam;
        let scale = lam.sqrt();
        for i in 0..n {
            let value = evecs[(i, src)] * scale;
            if !value.is_finite() {
                return Err(LandfoldError::Msg(
                    "MDS coordinate reconstruction overflowed".into(),
                ));
            }
            coords[(i, h)] = value;
        }
    }
    let ld_error = if trace.abs() > 0.0 {
        kept.sum() / trace
    } else {
        0.0
    };
    if !ld_error.is_finite() {
        return Err(LandfoldError::Msg(
            "MDS explained variance is not finite".into(),
        ));
    }
    Ok((coords, kept, ld_error))
}

/// Classical Torgerson MDS from a symmetric distance matrix.
///
/// Eigenvector signs follow the eigensolver, so the coordinate matrix is
/// not a C++ golden. Compare `pairwise` of the embedding instead: those
/// distances are the invariant Torgerson pairwise dumps in `oracle/oracle.cpp`.
pub fn classical_mds(dist: ArrayView2<f64>, lowdim: usize) -> Result<(Array2<f64>, MdsReport)> {
    let (n, b) = torgerson_b(dist)?;
    if lowdim == 0 || lowdim > n {
        return Err(LandfoldError::LowDim {
            low: lowdim,
            high: n,
        });
    }
    let dm = DMatrix::<f64>::from_row_slice(n, n, &b);
    let eigen = SymmetricEigen::new(dm);
    let evals: Vec<f64> = (0..n).map(|k| eigen.eigenvalues[k]).collect();
    let (coords, kept, ld_error) = coords_from_eigen(n, lowdim, &evals, &eigen.eigenvectors)?;
    Ok((
        coords,
        MdsReport {
            eigenvalues: kept,
            ld_error,
        },
    ))
}

/// Randomised rangefinder MDS (Halko, Martinsson, Tropp, *SIAM Rev.* 2011).
///
/// Draw `Omega`, form `Y = B Omega`, thin QR `Y = QR`, then the small
/// eigenproblem on `Q^T B Q`. `seed` is the SplitMix64 start.
pub fn randomized_mds(
    dist: ArrayView2<f64>,
    lowdim: usize,
    oversample: usize,
    seed: u64,
) -> Result<(Array2<f64>, MdsReport)> {
    let (n, b) = torgerson_b(dist)?;
    if lowdim == 0 || lowdim > n {
        return Err(LandfoldError::LowDim {
            low: lowdim,
            high: n,
        });
    }
    let requested = lowdim
        .checked_add(oversample.max(2))
        .ok_or(LandfoldError::Msg(
            "MDS randomized dimension overflowed".into(),
        ))?;
    let ell = requested.min(n);
    let bm = DMatrix::<f64>::from_row_slice(n, n, &b);
    let mut omega = DMatrix::<f64>::zeros(n, ell);
    let mut state = seed | 1;
    for i in 0..n {
        for j in 0..ell {
            // Box-Muller
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let u1 = ((z ^ (z >> 31)) as f64 / u64::MAX as f64).clamp(1e-12, 1.0);
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z2 = state;
            z2 = (z2 ^ (z2 >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z2 = (z2 ^ (z2 >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let u2 = (z2 ^ (z2 >> 31)) as f64 / u64::MAX as f64;
            omega[(i, j)] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
    }
    let y = &bm * &omega;
    let qr = QR::new(y);
    let q = qr.q();
    let small = q.transpose() * &bm * &q;
    let eigen = SymmetricEigen::new(small);
    if eigen.eigenvalues.iter().any(|value| !value.is_finite())
        || eigen.eigenvectors.iter().any(|value| !value.is_finite())
    {
        return Err(LandfoldError::Msg(
            "MDS randomized eigensystem is not finite".into(),
        ));
    }
    let mut pairs: Vec<(f64, usize)> = (0..ell).map(|k| (eigen.eigenvalues[k], k)).collect();
    pairs.sort_by(|a, c| c.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let d = lowdim.min(ell);
    let mut coords = Array2::<f64>::zeros((n, d));
    let mut kept = Array1::<f64>::zeros(d);
    for h in 0..d {
        let (lam, src) = pairs[h];
        let lam = lam.max(0.0);
        kept[h] = lam;
        let scale = lam.sqrt();
        for i in 0..n {
            let mut acc = 0.0;
            for r in 0..ell {
                let term = q[(i, r)] * eigen.eigenvectors[(r, src)];
                acc += term;
                if !acc.is_finite() {
                    return Err(LandfoldError::Msg(
                        "MDS eigenvector accumulation overflowed".into(),
                    ));
                }
            }
            let value = acc * scale;
            if !value.is_finite() {
                return Err(LandfoldError::Msg(
                    "MDS randomized coordinate reconstruction overflowed".into(),
                ));
            }
            coords[(i, h)] = value;
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

fn spherical_from_dist(dist: ArrayView2<f64>, lowdim: usize) -> Result<(Array2<f64>, MdsReport)> {
    let n = dist.nrows();
    if n < 2 {
        return Err(LandfoldError::Empty);
    }
    if lowdim == 0 || lowdim >= n {
        return Err(LandfoldError::LowDim {
            low: lowdim,
            high: n.saturating_sub(1),
        });
    }
    let mut sr: f64 = 0.0;
    for i in 0..n {
        for j in 0..i {
            sr = sr.max(dist[(i, j)]);
        }
    }
    if sr <= 0.0 {
        sr = 1.0;
    }
    sr /= std::f64::consts::PI;
    let sr2 = sr * sr;
    if !sr2.is_finite() {
        return Err(LandfoldError::Msg(
            "spherical MDS scale overflowed".into(),
        ));
    }
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let value = (dist[(i, j)] / sr).cos() * sr2;
            if !value.is_finite() {
                return Err(LandfoldError::Msg(
                    "spherical MDS matrix overflowed".into(),
                ));
            }
            m[i * n + j] = value;
        }
    }
    let dm = DMatrix::<f64>::from_row_slice(n, n, &m);
    let eigen = SymmetricEigen::new(dm);
    let mut pairs: Vec<(f64, usize)> = (0..n).map(|k| (eigen.eigenvalues[k], k)).collect();
    pairs.sort_by(|a, c| c.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    // Hyperspherical angles from the leading (lowdim+1) components.
    let mut q = vec![0.0; n * (lowdim + 1)];
    let mut kept = Array1::<f64>::zeros(lowdim);
    for h in 0..=lowdim {
        let (lam, src) = pairs[h];
        let lam = lam.max(0.0);
        if h < lowdim {
            kept[h] = lam;
        }
        let scale = lam.sqrt();
        for i in 0..n {
            q[i * (lowdim + 1) + h] = eigen.eigenvectors[(i, src)] * scale;
        }
    }
    let mut coords = Array2::<f64>::zeros((n, lowdim));
    let dim = lowdim + 1;
    for i in 0..n {
        let mut tx = 0.0;
        let q0 = q[i * dim];
        let q1 = q[i * dim + 1];
        coords[(i, 0)] = q1.atan2(q0) / std::f64::consts::PI;
        tx += q1 * q1;
        for h in 1..lowdim {
            let qh = q[i * dim + h];
            tx += qh * qh;
            let qn = q[i * dim + h + 1];
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

fn spherical_mds(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    lowdim: usize,
) -> Result<(Array2<f64>, MdsReport)> {
    let dist = pairwise(points, metric)?;
    spherical_from_dist(dist.view(), lowdim)
}

fn toroidal_mds(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    lowdim: usize,
) -> Result<(Array2<f64>, MdsReport)> {
    // Sequential 1-D spherical MDS of the *current residual* distances.
    let mut dist = pairwise(points, metric)?;
    let n = dist.nrows();
    let mut coords = Array2::<f64>::zeros((n, lowdim));
    let mut evals = Array1::<f64>::zeros(lowdim);
    for th in 0..lowdim {
        let (p1, rep) = spherical_from_dist(dist.view(), 1)?;
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
        if sr <= 0.0 {
            sr = 1.0;
        }
        sr /= std::f64::consts::PI;
        if !sr.is_finite() {
            return Err(LandfoldError::Msg(
                "toroidal MDS scale overflowed".into(),
            ));
        }
        for i in 0..n {
            for j in 0..i {
                let mut tdij = (p1[(i, 0)] - p1[(j, 0)]).abs();
                while tdij > 1.0 {
                    tdij -= 2.0;
                }
                tdij = tdij.abs() * std::f64::consts::PI * sr;
                let base_sq = dist[(i, j)] * dist[(i, j)];
                let torus_sq = tdij * tdij;
                if !base_sq.is_finite() || !torus_sq.is_finite() {
                    return Err(LandfoldError::Msg(
                        "toroidal MDS residual squaring overflowed".into(),
                    ));
                }
                let mut corr = base_sq - torus_sq;
                if !corr.is_finite() {
                    return Err(LandfoldError::Msg(
                        "toroidal MDS residual overflowed".into(),
                    ));
                }
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

    #[test]
    fn randomized_mds_recovers_triangle() {
        let pts = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let dist = pairwise_euclid(pts.view()).unwrap();
        let (emb, _) = randomized_mds(dist.view(), 2, 2, 7).unwrap();
        let d0 = pairwise_euclid(pts.view()).unwrap();
        let d1 = pairwise_euclid(emb.view()).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(d0[(i, j)], d1[(i, j)], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn rejects_malformed_distance_matrices() {
        assert!(classical_mds(array![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]].view(), 1).is_err());
        assert!(classical_mds(array![[0.0, f64::NAN], [f64::NAN, 0.0]].view(), 1).is_err());
        assert!(randomized_mds(array![[0.0, f64::INFINITY], [1.0, 0.0]].view(), 1, 2, 0).is_err());
        assert!(classical_mds(array![[0.0, -1.0], [-1.0, 0.0]].view(), 1).is_err());
        assert!(randomized_mds(array![[0.0, 1.0e200], [1.0e200, 0.0]].view(), 1, 2, 0).is_err());
    }

    #[test]
    fn rejects_mds_centering_overflow() {
        let dist = array![
            [0.0, 1.0e154, 1.0e154],
            [1.0e154, 0.0, 1.0e154],
            [1.0e154, 1.0e154, 0.0],
        ];
        assert!(classical_mds(dist.view(), 1).is_err());
        assert!(randomized_mds(dist.view(), 1, 2, 0).is_err());
    }

    #[test]
    fn rejects_mds_spectral_overflow_and_dimensions() {
        let eigenvectors = DMatrix::<f64>::identity(2, 2);
        assert!(coords_from_eigen(2, 1, &[f64::NAN, 1.0], &eigenvectors).is_err());
        assert!(coords_from_eigen(2, 1, &[1.0, 1.0], &DMatrix::identity(1, 1)).is_err());
        let dist = array![[0.0, 1.0], [1.0, 0.0]];
        assert!(randomized_mds(dist.view(), 1, usize::MAX, 0).is_err());
    }

    #[test]
    fn rejects_spherical_and_toroidal_overflow() {
        struct HugeMetric;

        impl Metric for HugeMetric {
            fn dist_unchecked(&self, a: &[f64], b: &[f64]) -> f64 {
                if a == b { 0.0 } else { 1.0e200 }
            }
        }

        let points = array![[0.0], [1.0], [2.0]];
        let metric = HugeMetric;
        assert!(mds_from_points(points.view(), &metric, 1, MdsMode::Spherical).is_err());
        assert!(mds_from_points(points.view(), &metric, 1, MdsMode::Toroidal).is_err());
    }
}
