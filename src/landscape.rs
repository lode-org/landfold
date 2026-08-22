//! Landscape chi: Laplacian eigenmaps of the energy-weighted minima graph.
//!
//! Sketch-map of a packing histogram does not encode barriers, so occupancy
//! of inherent structures cannot show Wales basins of attraction. Build the
//! k-NN graph in the high-D metric, weight each edge by
//! `exp(-(|E_i-E_j| + lambda * D_ij) / T)`, and take the first two nontrivial
//! eigenvectors of the symmetric normalized Laplacian (Belkin and Niyogi,
//! *Neural Comput.* **15**, 1373 (2003)).

use nalgebra::{DMatrix, SymmetricEigen};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{LandfoldError, Result};
use crate::metric::Metric;

#[derive(Clone, Debug)]
pub struct LandscapeOpts {
    pub knn: usize,
    pub temperature: f64,
    pub lambda: f64,
    pub lowdim: usize,
}

impl Default for LandscapeOpts {
    fn default() -> Self {
        Self {
            knn: 12,
            temperature: 0.5,
            lambda: 2.0,
            lowdim: 2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LandscapeReport {
    pub eigenvalues: Array1<f64>,
    pub n_edges: usize,
}

/// Symmetric normalized Laplacian eigenmaps of the energy-weighted k-NN graph.
pub fn landscape_embed(
    points: ArrayView2<f64>,
    energy: ArrayView1<f64>,
    metric: &dyn Metric,
    opts: &LandscapeOpts,
) -> Result<(Array2<f64>, LandscapeReport)> {
    let n = points.nrows();
    if n == 0 {
        return Err(LandfoldError::Empty);
    }
    if energy.len() != n {
        return Err(LandfoldError::Shape(
            "landscape energy length must match the number of points",
        ));
    }
    if opts.lowdim == 0 || opts.lowdim >= n {
        return Err(LandfoldError::LowDim {
            low: opts.lowdim,
            high: n,
        });
    }
    if opts.knn == 0 || opts.knn >= n {
        return Err(LandfoldError::Msg(format!(
            "landscape knn must be in 1..n-1 (got knn={}, n={})",
            opts.knn, n
        )));
    }
    if !(opts.temperature > 0.0) || !opts.temperature.is_finite() {
        return Err(LandfoldError::Msg(
            "landscape temperature must be positive".into(),
        ));
    }
    if !opts.lambda.is_finite() || opts.lambda < 0.0 {
        return Err(LandfoldError::Msg(
            "landscape lambda must be finite and nonnegative".into(),
        ));
    }

    let mut affinity = vec![0.0; n * n];
    let mut n_edges = 0usize;
    for i in 0..n {
        let mut neigh: Vec<(f64, usize)> = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i == j {
                continue;
            }
            let a = points.row(i);
            let b = points.row(j);
            let a = a.as_slice().ok_or(LandfoldError::Empty)?;
            let b = b.as_slice().ok_or(LandfoldError::Empty)?;
            let d = metric.dist(a, b)?;
            if !d.is_finite() || d < 0.0 {
                return Err(LandfoldError::Msg(
                    "landscape metric produced a non-finite distance".into(),
                ));
            }
            neigh.push((d, j));
        }
        neigh.sort_by(|a, b| a.0.total_cmp(&b.0));
        for &(d, j) in neigh.iter().take(opts.knn) {
            let de = (energy[i] - energy[j]).abs();
            let cost = de + opts.lambda * d;
            let w = (-cost / opts.temperature).exp();
            let a = i * n + j;
            let b = j * n + i;
            if affinity[a] == 0.0 && affinity[b] == 0.0 {
                n_edges += 1;
            }
            if w > affinity[a] {
                affinity[a] = w;
                affinity[b] = w;
            }
        }
    }

    let mut deg = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += affinity[i * n + j];
        }
        deg[i] = s.max(1e-300);
    }
    let mut s = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        let di = deg[i].sqrt();
        for j in 0..n {
            let dj = deg[j].sqrt();
            s[(i, j)] = affinity[i * n + j] / (di * dj);
        }
    }
    let ev = SymmetricEigen::new(s);
    // nalgebra returns eigenvalues in increasing order
    let mut pairs: Vec<(f64, usize)> = ev
        .eigenvalues
        .iter()
        .copied()
        .enumerate()
        .map(|(k, lam)| (lam, k))
        .collect();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    // skip the constant mode (largest eigenvalue ~ 1)
    let mut coords = Array2::<f64>::zeros((n, opts.lowdim));
    let mut evals = Array1::<f64>::zeros(opts.lowdim + 1);
    for (slot, (lam, idx)) in pairs.iter().take(opts.lowdim + 1).enumerate() {
        evals[slot] = *lam;
        if slot == 0 {
            continue;
        }
        let col = slot - 1;
        for i in 0..n {
            coords[(i, col)] = ev.eigenvectors[(i, *idx)] / deg[i].sqrt();
        }
    }
    Ok((
        coords,
        LandscapeReport {
            eigenvalues: evals,
            n_edges,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::Euclid;
    use ndarray::array;

    #[test]
    fn two_wells_separate() {
        let pts = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 0.0],
            [5.1, 0.0],
            [5.0, 0.1],
        ];
        let e = array![-2.0, -1.8, -1.7, -1.9, -1.7, -1.6];
        let (xy, rep) = landscape_embed(pts.view(), e.view(), &Euclid, &LandscapeOpts::default())
            .expect("embed");
        assert_eq!(xy.nrows(), 6);
        assert_eq!(xy.ncols(), 2);
        assert!(rep.n_edges > 0);
        let left = (xy.row(0).to_owned() + xy.row(1).to_owned() + xy.row(2).to_owned()) / 3.0;
        let right = (xy.row(3).to_owned() + xy.row(4).to_owned() + xy.row(5).to_owned()) / 3.0;
        let sep = ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2)).sqrt();
        let d_in = ((xy[(0, 0)] - xy[(1, 0)]).powi(2) + (xy[(0, 1)] - xy[(1, 1)]).powi(2)).sqrt();
        assert!(sep > 2.0 * d_in, "sep={sep} din={d_in}");
    }
}
