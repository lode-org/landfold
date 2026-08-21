//! Exact projection onto a named high-D contrast, residual PCs.
//!
//! Given basin refs `a` and `b`,
//!
//! `s₁(x) = (x-a)·u / ‖b-a‖`, `u = (b-a)/‖b-a‖`.
//!
//! Then `s₁(a) = 0`, `s₁(b) = 1`, and
//! `s₁(x)-s₁(y) = (x-y)·u / ‖b-a‖`: the first axis is an isometry
//! of the one-dimensional subspace spanned by the contrast. Residual
//! coordinates are the leading principal components of
//! `r = (x-a) - ((x-a)·u) u`. This is closed form; it is not χ.

use nalgebra::{DMatrix, SymmetricEigen};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{LandfoldError, Result};

#[derive(Clone, Debug)]
pub struct AxisModel {
    pub origin: Array1<f64>,
    pub unit: Array1<f64>,
    pub gap: f64,
    pub residual: Array2<f64>,
}

impl AxisModel {
    pub fn dim(&self) -> usize {
        self.origin.len()
    }

    pub fn lowdim(&self) -> usize {
        1 + self.residual.nrows()
    }
}

fn dot(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn unit_gap(a: ArrayView1<f64>, b: ArrayView1<f64>) -> Result<(Array1<f64>, f64)> {
    if a.len() != b.len() || a.is_empty() {
        return Err(LandfoldError::Shape(
            "axis refs must share a positive dimension",
        ));
    }
    let mut gap2 = 0.0;
    for k in 0..a.len() {
        let e = b[k] - a[k];
        gap2 += e * e;
    }
    if !(gap2 > 0.0) || !gap2.is_finite() {
        return Err(LandfoldError::Msg("axis refs must differ".into()));
    }
    let gap = gap2.sqrt();
    let mut u = Array1::<f64>::zeros(a.len());
    for k in 0..a.len() {
        u[k] = (b[k] - a[k]) / gap;
    }
    Ok((u, gap))
}

/// `ξ(1-ξ)`: minima at the two refs, barrier at the midpoint.
pub fn double_well(xi: f64) -> f64 {
    xi * (1.0 - xi)
}

/// Fit the contrast axis and residual PCs on `points`.
pub fn axis_fit(
    points: ArrayView2<f64>,
    ref_a: ArrayView1<f64>,
    ref_b: ArrayView1<f64>,
    lowdim: usize,
) -> Result<AxisModel> {
    let n = points.nrows();
    let d = points.ncols();
    if n == 0 {
        return Err(LandfoldError::Empty);
    }
    if ref_a.len() != d || ref_b.len() != d {
        return Err(LandfoldError::Shape(
            "axis refs must match descriptor dimension",
        ));
    }
    if lowdim == 0 || lowdim > d.max(1) {
        return Err(LandfoldError::LowDim {
            low: lowdim,
            high: d.max(1),
        });
    }
    if points.iter().any(|v| !v.is_finite())
        || ref_a.iter().any(|v| !v.is_finite())
        || ref_b.iter().any(|v| !v.is_finite())
    {
        return Err(LandfoldError::Msg("axis inputs must be finite".into()));
    }
    let (unit, gap) = unit_gap(ref_a, ref_b)?;
    let n_res = lowdim - 1;
    let mut residual = Array2::<f64>::zeros((n_res, d));
    if n_res > 0 {
        let mut r = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            let t = dot(points.row(i), unit.view()) - dot(ref_a, unit.view());
            for k in 0..d {
                r[(i, k)] = points[(i, k)] - ref_a[k] - t * unit[k];
            }
        }
        let mut mean = vec![0.0; d];
        for i in 0..n {
            for k in 0..d {
                mean[k] += r[(i, k)];
            }
        }
        let inv = 1.0 / n as f64;
        for k in 0..d {
            mean[k] *= inv;
        }
        for i in 0..n {
            for k in 0..d {
                r[(i, k)] -= mean[k];
            }
        }
        let mut cov = DMatrix::<f64>::zeros(d, d);
        for i in 0..n {
            for p in 0..d {
                for q in 0..=p {
                    let v = r[(i, p)] * r[(i, q)];
                    cov[(p, q)] += v;
                    if p != q {
                        cov[(q, p)] += v;
                    }
                }
            }
        }
        let eigen = SymmetricEigen::new(cov);
        let mut pairs: Vec<(f64, usize)> = (0..d).map(|k| (eigen.eigenvalues[k], k)).collect();
        pairs.sort_by(|a, c| c.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for j in 0..n_res {
            let (ev, idx) = pairs[j];
            if !(ev > 1e-14) || !ev.is_finite() {
                continue;
            }
            let mut flip = 1.0;
            for k in 0..d {
                let c = eigen.eigenvectors[(k, idx)];
                if c.abs() > 1e-12 {
                    if c < 0.0 {
                        flip = -1.0;
                    }
                    break;
                }
            }
            for k in 0..d {
                residual[(j, k)] = flip * eigen.eigenvectors[(k, idx)];
            }
        }
    }
    Ok(AxisModel {
        origin: ref_a.to_owned(),
        unit,
        gap,
        residual,
    })
}

/// Apply a fitted axis model. `s₁` is the exact contrast coordinate.
pub fn axis_project(model: &AxisModel, points: ArrayView2<f64>) -> Result<Array2<f64>> {
    let n = points.nrows();
    let d = points.ncols();
    if n == 0 {
        return Err(LandfoldError::Empty);
    }
    if d != model.dim() {
        return Err(LandfoldError::Shape("axis project dimension mismatch"));
    }
    let low = model.lowdim();
    let mut out = Array2::<f64>::zeros((n, low));
    for i in 0..n {
        let row = points.row(i);
        let mut t = 0.0;
        for k in 0..d {
            t += (row[k] - model.origin[k]) * model.unit[k];
        }
        out[(i, 0)] = t / model.gap;
        if low > 1 {
            for k in 0..d {
                let rk = row[k] - model.origin[k] - t * model.unit[k];
                for j in 0..model.residual.nrows() {
                    out[(i, 1 + j)] += rk * model.residual[(j, k)];
                }
            }
        }
        for j in 0..low {
            if !out[(i, j)].is_finite() {
                return Err(LandfoldError::Msg("axis coordinate is not finite".into()));
            }
        }
    }
    Ok(out)
}

/// Fit and project in one pass.
pub fn axis_embed(
    points: ArrayView2<f64>,
    ref_a: ArrayView1<f64>,
    ref_b: ArrayView1<f64>,
    lowdim: usize,
) -> Result<(Array2<f64>, AxisModel)> {
    let model = axis_fit(points, ref_a, ref_b, lowdim)?;
    let coords = axis_project(&model, points)?;
    Ok((coords, model))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn refs_land_at_zero_and_one() {
        let pts = array![[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 1.0, 0.0]];
        let (y, model) = axis_embed(
            pts.view(),
            array![0.0, 0.0, 0.0].view(),
            array![4.0, 0.0, 0.0].view(),
            2,
        )
        .unwrap();
        assert!((y[(0, 0)] - 0.0).abs() < 1e-12);
        assert!((y[(1, 0)] - 1.0).abs() < 1e-12);
        assert!((model.gap - 4.0).abs() < 1e-12);
        assert!(y[(0, 1)].abs() < 1e-12);
        assert!(y[(1, 1)].abs() < 1e-12);
        assert!(y[(2, 1)].abs() > 0.5);
    }

    #[test]
    fn contrast_difference_is_the_hd_dot() {
        let a = array![1.0, 2.0, 0.0];
        let b = array![4.0, 6.0, 0.0];
        let pts = array![[1.0, 2.0, 0.0], [2.5, 4.0, 3.0], [4.0, 6.0, 0.0]];
        let (y, model) = axis_embed(pts.view(), a.view(), b.view(), 1).unwrap();
        let dx0 = pts[(2, 0)] - pts[(1, 0)];
        let dx1 = pts[(2, 1)] - pts[(1, 1)];
        let dx2 = pts[(2, 2)] - pts[(1, 2)];
        let hd = (dx0 * model.unit[0] + dx1 * model.unit[1] + dx2 * model.unit[2]) / model.gap;
        assert!((y[(2, 0)] - y[(1, 0)] - hd).abs() < 1e-12);
    }

    #[test]
    fn residual_is_orthogonal_to_the_axis() {
        let pts = array![[0.0, 0.0], [1.0, 0.0], [0.4, 0.7], [0.8, -0.3]];
        let model = axis_fit(
            pts.view(),
            array![0.0, 0.0].view(),
            array![1.0, 0.0].view(),
            2,
        )
        .unwrap();
        assert!(dot(model.unit.view(), model.residual.row(0)).abs() < 1e-12);
    }

    #[test]
    fn two_clouds_split_on_s1() {
        let mut pts = Array2::<f64>::zeros((16, 3));
        for i in 0..8 {
            pts[(i, 0)] = 0.02 * i as f64;
            pts[(i + 8, 0)] = 6.0 + 0.02 * i as f64;
            pts[(i, 1)] = 0.01 * (i as f64 - 3.5);
            pts[(i + 8, 1)] = 0.01 * (i as f64 - 3.5);
        }
        let (y, _) = axis_embed(
            pts.view(),
            array![0.0, 0.0, 0.0].view(),
            array![6.0, 0.0, 0.0].view(),
            2,
        )
        .unwrap();
        let ma: f64 = (0..8).map(|i| y[(i, 0)]).sum::<f64>() / 8.0;
        let mb: f64 = (8..16).map(|i| y[(i, 0)]).sum::<f64>() / 8.0;
        assert!(ma < 0.1, "fcc-side mean {ma}");
        assert!(mb > 0.9, "ico-side mean {mb}");
    }

    #[test]
    fn project_matches_embed() {
        let pts = array![[0.0, 1.0], [2.0, 1.0], [1.0, 3.0], [1.5, 0.2]];
        let a = array![0.0, 1.0];
        let b = array![2.0, 1.0];
        let (y, model) = axis_embed(pts.view(), a.view(), b.view(), 2).unwrap();
        let z = axis_project(&model, pts.view()).unwrap();
        for i in 0..4 {
            assert!((y[(i, 0)] - z[(i, 0)]).abs() < 1e-14);
            assert!((y[(i, 1)] - z[(i, 1)]).abs() < 1e-14);
        }
    }

    #[test]
    fn double_well_minima_at_the_refs() {
        assert!((double_well(0.0) - 0.0).abs() < 1e-15);
        assert!((double_well(1.0) - 0.0).abs() < 1e-15);
        assert!((double_well(0.5) - 0.25).abs() < 1e-15);
        assert!(double_well(0.5) > double_well(0.1));
        assert!(double_well(0.5) > double_well(0.9));
    }
}
