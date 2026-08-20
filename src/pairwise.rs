//! Pairwise distance matrices.
//!
//! Euclid uses a Gram GEMM (`||xi-xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi·xj`),
//! the standard tensor contraction for Euclidean distances. Other metrics
//! walk the upper triangle, in parallel when the `parallel` feature is on.

use ndarray::{Array1, Array2, ArrayView2};

use crate::error::Result;
use crate::metric::{Metric, stable_euclid, validate_distance};

fn checked_product(left: usize, right: usize, what: &'static str) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| {
        crate::error::LandfoldError::Msg(format!("{what} dimension product overflowed"))
    })
}

fn validate_metric_dim(points: ArrayView2<f64>, metric: &dyn Metric) -> Result<()> {
    match metric.dim() {
        Some(expected) if points.ncols() != expected => {
            Err(crate::error::LandfoldError::MetricSize {
                left: points.ncols(),
                right: expected,
            })
        }
        _ => Ok(()),
    }
}

fn validate_finite_points(points: ArrayView2<f64>) -> Result<()> {
    if points.iter().any(|&value| !value.is_finite()) {
        return Err(crate::error::LandfoldError::Msg(
            "point coordinates must be finite".into(),
        ));
    }
    Ok(())
}

fn gemm_euclid(norm_i: f64, norm_j: f64, gram: f64) -> Option<f64> {
    let squared = norm_i + norm_j - 2.0 * gram;
    let scale = norm_i.abs() + norm_j.abs() + 2.0 * gram.abs();
    if !squared.is_finite()
        || squared < 0.0
        || !scale.is_finite()
        || squared <= 64.0 * f64::EPSILON * scale.max(1.0)
    {
        return None;
    }
    Some(squared.sqrt())
}

/// Symmetric `n x n` distance matrix. Diagonal is zero.
pub fn pairwise(points: ArrayView2<f64>, metric: &dyn Metric) -> Result<Array2<f64>> {
    let n = points.nrows();
    let d = points.ncols();
    if n == 0 {
        return Err(crate::error::LandfoldError::Empty);
    }
    let matrix_len = checked_product(n, n, "pairwise matrix")?;
    validate_finite_points(points)?;
    validate_metric_dim(points, metric)?;
    #[cfg(feature = "parallel")]
    let packed_len = checked_product(n, d, "pairwise point")?;
    let mut out = Array2::<f64>::zeros((n, n));
    debug_assert_eq!(matrix_len, out.len());

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let packed: Vec<f64> = points.iter().copied().collect();
        debug_assert_eq!(packed_len, packed.len());
        let rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| -> Result<Vec<f64>> {
                let a = &packed[i * d..(i + 1) * d];
                let mut row = vec![0.0; i];
                for j in 0..i {
                    let b = &packed[j * d..(j + 1) * d];
                    let distance = metric.dist_unchecked(a, b);
                    validate_distance(distance)?;
                    row[j] = distance;
                }
                Ok(row)
            })
            .collect::<Result<Vec<_>>>()?;
        for i in 0..n {
            for j in 0..i {
                let v = rows[i][j];
                out[(i, j)] = v;
                out[(j, i)] = v;
            }
        }
        Ok(out)
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut ai = vec![0.0; d];
        let mut bj = vec![0.0; d];
        for i in 0..n {
            for (k, &v) in points.row(i).iter().enumerate() {
                ai[k] = v;
            }
            for j in 0..i {
                for (k, &v) in points.row(j).iter().enumerate() {
                    bj[k] = v;
                }
                let v = metric.dist_unchecked(&ai, &bj);
                validate_distance(v)?;
                out[(i, j)] = v;
                out[(j, i)] = v;
            }
        }
        Ok(out)
    }
}

/// Fast Euclidean pairwise via GEMM. Preferred over [`pairwise`] for Euclid.
pub fn pairwise_euclid(points: ArrayView2<f64>) -> Result<Array2<f64>> {
    let n = points.nrows();
    if n == 0 {
        return Err(crate::error::LandfoldError::Empty);
    }
    let matrix_len = checked_product(n, n, "Euclidean pairwise matrix")?;
    validate_finite_points(points)?;
    #[cfg(feature = "parallel")]
    let packed_len = checked_product(n, points.ncols(), "Euclidean pairwise point")?;
    let norms: Array1<f64> = points
        .rows()
        .into_iter()
        .map(|r| r.iter().map(|x| x * x).sum())
        .collect();
    let gram = points.dot(&points.t());
    let gemm_valid =
        norms.iter().all(|&value| value.is_finite()) && gram.iter().all(|&value| value.is_finite());
    let mut out = Array2::<f64>::zeros((n, n));
    debug_assert_eq!(matrix_len, out.len());
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let packed: Vec<f64> = points.iter().copied().collect();
        debug_assert_eq!(packed_len, packed.len());
        let gram_s = gram.as_slice().expect("gram contiguous");
        let norms_s = norms.as_slice().expect("norms contiguous");
        let rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| -> Result<Vec<f64>> {
                let mut row = vec![0.0; i];
                for j in 0..i {
                    let a = &packed[i * points.ncols()..(i + 1) * points.ncols()];
                    let b = &packed[j * points.ncols()..(j + 1) * points.ncols()];
                    let distance = if gemm_valid {
                        gemm_euclid(norms_s[i], norms_s[j], gram_s[i * n + j])
                            .unwrap_or_else(|| stable_euclid(a.iter(), b.iter()))
                    } else {
                        stable_euclid(a.iter(), b.iter())
                    };
                    validate_distance(distance)?;
                    row[j] = distance;
                }
                Ok(row)
            })
            .collect::<Result<Vec<_>>>()?;
        for i in 0..n {
            for j in 0..i {
                let v = rows[i][j];
                out[(i, j)] = v;
                out[(j, i)] = v;
            }
        }
        Ok(out)
    }
    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..n {
            for j in 0..i {
                let a = points.row(i);
                let b = points.row(j);
                let v = if gemm_valid {
                    gemm_euclid(norms[i], norms[j], gram[(i, j)])
                        .unwrap_or_else(|| stable_euclid(a.iter(), b.iter()))
                } else {
                    stable_euclid(a.iter(), b.iter())
                };
                validate_distance(v)?;
                out[(i, j)] = v;
                out[(j, i)] = v;
            }
        }
        Ok(out)
    }
}

/// Apply a transfer function to a precomputed distance matrix transactionally.
pub fn apply_transfer(
    dist: &mut Array2<f64>,
    t: &crate::transfer::Transfer,
) -> crate::error::Result<()> {
    let n = dist.nrows();
    if dist.ncols() != n {
        return Err(crate::error::LandfoldError::Shape(
            "distance matrix must be square",
        ));
    }
    if dist.iter().any(|&value| !value.is_finite() || value < 0.0) {
        return Err(crate::error::LandfoldError::Msg(
            "distance matrix must be finite and nonnegative".into(),
        ));
    }
    let matrix_len = checked_product(n, n, "transformed distance matrix")?;
    let mut transformed = Array2::<f64>::zeros((n, n));
    debug_assert_eq!(matrix_len, transformed.len());
    let diagonal = t.try_fdf(0.0)?.0;
    for i in 0..n {
        transformed[(i, i)] = diagonal;
        for j in 0..i {
            let v = t.try_fdf(dist[(i, j)])?.0;
            transformed[(i, j)] = v;
            transformed[(j, i)] = v;
        }
    }
    *dist = transformed;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{Euclid, Periodic};
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn rejects_dimension_product_overflow() {
        assert!(checked_product(usize::MAX, 2, "test").is_err());
        assert_eq!(checked_product(usize::MAX, 1, "test").unwrap(), usize::MAX);
    }

    #[test]
    fn euclid_gemm_matches_metric() {
        let pts = array![[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]];
        let a = pairwise_euclid(pts.view()).unwrap();
        let b = pairwise(pts.view(), &Euclid).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(a[(i, j)], b[(i, j)], epsilon = 1e-14);
            }
        }
        assert_relative_eq!(a[(0, 1)], 5.0, epsilon = 1e-15);
    }

    #[test]
    fn large_translated_points_keep_small_euclidean_separations() {
        let points = array![[1.0e16, 0.0], [1.0e16 + 2.0, 0.0]];
        let distances = pairwise_euclid(points.view()).unwrap();
        assert_relative_eq!(distances[(0, 1)], 2.0, epsilon = 1e-14);
    }

    #[test]
    fn rejects_metric_dimension_mismatch_before_pairwise_indexing() {
        let points = array![[0.0, 0.0], [1.0, 0.0]];
        let metric = Periodic::isotropic(1, 1.0).unwrap();
        assert!(pairwise(points.view(), &metric).is_err());
    }

    #[test]
    fn rejects_nonfinite_points() {
        let points = array![[0.0, f64::NAN], [1.0, 0.0]];
        assert!(pairwise(points.view(), &Euclid).is_err());
        assert!(pairwise_euclid(points.view()).is_err());
    }

    struct InvalidMetric;

    impl Metric for InvalidMetric {
        fn dist_unchecked(&self, _a: &[f64], _b: &[f64]) -> f64 {
            f64::NAN
        }
    }

    #[test]
    fn rejects_invalid_metric_distances() {
        let points = array![[0.0], [1.0]];
        assert!(pairwise(points.view(), &InvalidMetric).is_err());
    }

    #[test]
    fn handles_finite_distances_beyond_gram_range() {
        let points = array![[1.0e200, 0.0], [0.0, 0.0]];
        let distances = pairwise_euclid(points.view()).unwrap();
        assert_eq!(distances[(0, 1)], 1.0e200);
    }

    #[test]
    fn accepts_non_contiguous_point_rows() {
        let points = array![[0.0, 3.0], [0.0, 4.0], [9.0, 9.0]];
        let strided = points.view().reversed_axes();
        let distances = pairwise_euclid(strided).unwrap();
        assert_relative_eq!(distances[(0, 1)], 5.0, epsilon = 1e-14);
    }

    #[test]
    fn transfer_application_is_checked_and_transactional() {
        let transfer = crate::transfer::Transfer::identity();
        let mut nonsquare = Array2::<f64>::zeros((2, 1));
        assert!(apply_transfer(&mut nonsquare, &transfer).is_err());

        let mut invalid = array![[0.0, f64::NAN], [f64::NAN, 0.0]];
        let original = invalid.clone();
        assert!(apply_transfer(&mut invalid, &transfer).is_err());
        assert!(
            invalid
                .iter()
                .zip(original.iter())
                .all(|(left, right)| left.to_bits() == right.to_bits())
        );

        let mut distances = array![[7.0, 2.0], [2.0, 9.0]];
        apply_transfer(&mut distances, &transfer).unwrap();
        assert_eq!(distances, array![[0.0, 2.0], [2.0, 0.0]]);
    }
}
