//! Pairwise distance matrices.
//!
//! Euclid uses a Gram GEMM (`||xi-xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi·xj`),
//! the standard tensor contraction for Euclidean distances. Other metrics
//! walk the upper triangle, in parallel when the `parallel` feature is on.

use ndarray::{Array1, Array2, ArrayView2};

use crate::error::Result;
use crate::metric::Metric;

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

/// Symmetric `n x n` distance matrix. Diagonal is zero.
pub fn pairwise(points: ArrayView2<f64>, metric: &dyn Metric) -> Result<Array2<f64>> {
    let n = points.nrows();
    let d = points.ncols();
    if n == 0 {
        return Err(crate::error::LandfoldError::Empty);
    }
    validate_metric_dim(points, metric)?;
    let mut out = Array2::<f64>::zeros((n, n));

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let packed: Vec<f64> = points.iter().copied().collect();
        let rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let a = &packed[i * d..(i + 1) * d];
                let mut row = vec![0.0; i];
                for j in 0..i {
                    let b = &packed[j * d..(j + 1) * d];
                    row[j] = metric.dist_unchecked(a, b);
                }
                row
            })
            .collect();
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
    let norms: Array1<f64> = points
        .rows()
        .into_iter()
        .map(|r| r.iter().map(|x| x * x).sum())
        .collect();
    let gram = points.dot(&points.t());
    let mut out = Array2::<f64>::zeros((n, n));
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let gram_s = gram.as_slice().expect("gram contiguous");
        let norms_s = norms.as_slice().expect("norms contiguous");
        let rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row = vec![0.0; i];
                for j in 0..i {
                    row[j] = (norms_s[i] + norms_s[j] - 2.0 * gram_s[i * n + j])
                        .max(0.0)
                        .sqrt();
                }
                row
            })
            .collect();
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
                let v = (norms[i] + norms[j] - 2.0 * gram[(i, j)]).max(0.0).sqrt();
                out[(i, j)] = v;
                out[(j, i)] = v;
            }
        }
        Ok(out)
    }
}

/// Apply a transfer function to a precomputed distance matrix (in place).
pub fn apply_transfer(dist: &mut Array2<f64>, t: &crate::transfer::Transfer) {
    let n = dist.nrows();
    for i in 0..n {
        dist[(i, i)] = t.f(0.0);
        for j in 0..i {
            let v = t.f(dist[(i, j)]);
            dist[(i, j)] = v;
            dist[(j, i)] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{Euclid, Periodic};
    use approx::assert_relative_eq;
    use ndarray::array;

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
    fn rejects_metric_dimension_mismatch_before_pairwise_indexing() {
        let points = array![[0.0, 0.0], [1.0, 0.0]];
        let metric = Periodic::isotropic(1, 1.0).unwrap();
        assert!(pairwise(points.view(), &metric).is_err());
    }
}
