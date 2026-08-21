//! PaCMAP: near-to-near, mid-near, far-to-far.
//!
//! Wang, Huang, Rudin and Shaposhnik, *J. Mach. Learn. Res.* **22**,
//! 1 (2021), <https://www.jmlr.org/papers/v22/20-1061.html>.
//! Neighbor pairs attract as `d^2/(10+d^2)`, mid-near pairs as
//! `d^2/(10000+d^2)`, far pairs repel as `1/(1+d^2)`. That is the
//! split loss: local k-NN plus a far-pair term. Rank-uniform maps
//! each low-D axis through its empirical CDF so the cloud is uniform
//! on the square (axis ranks are preserved).

use ndarray::{Array2, ArrayView2};

use crate::error::{LandfoldError, Result};
use crate::mds::classical_mds;
use crate::metric::Metric;
use crate::pairwise::pairwise;

#[derive(Clone, Debug)]
pub struct PacmapOpts {
    pub n_neighbors: usize,
    pub mn_ratio: f64,
    pub fp_ratio: f64,
    pub steps: usize,
    pub lr: f64,
    pub lowdim: usize,
    pub uniform: bool,
}

impl Default for PacmapOpts {
    fn default() -> Self {
        Self {
            n_neighbors: 10,
            mn_ratio: 0.5,
            fp_ratio: 2.0,
            steps: 200,
            lr: 1.0,
            lowdim: 2,
            uniform: true,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum PairKind {
    Near,
    Mid,
    Far,
}

struct Pair {
    i: usize,
    j: usize,
    kind: PairKind,
}

/// Embed `points` under `metric` with the PaCMAP graph and loss.
pub fn pacmap_embed(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    opts: &PacmapOpts,
) -> Result<Array2<f64>> {
    let n = points.nrows();
    if n < 4 {
        return Err(LandfoldError::Msg("PaCMAP needs at least four points".into()));
    }
    if opts.lowdim == 0 || opts.lowdim > n {
        return Err(LandfoldError::LowDim {
            low: opts.lowdim,
            high: n,
        });
    }
    let k = opts.n_neighbors.min(n - 1).max(1);
    let n_mid = ((k as f64) * opts.mn_ratio).round() as usize;
    let n_far = ((k as f64) * opts.fp_ratio).round() as usize;
    let n_mid = n_mid.max(1);
    let n_far = n_far.max(1);
    let dist = pairwise(points, metric)?;
    let pairs = build_pairs(dist.view(), k, n_mid, n_far)?;
    let (mut y, _) = classical_mds(dist.view(), opts.lowdim)?;
    let mut grad = Array2::<f64>::zeros((n, opts.lowdim));
    for step in 0..opts.steps {
        grad.fill(0.0);
        for p in &pairs {
            let mut d2 = 0.0;
            let mut delta = vec![0.0; opts.lowdim];
            for h in 0..opts.lowdim {
                delta[h] = y[(p.i, h)] - y[(p.j, h)];
                d2 += delta[h] * delta[h];
            }
            // Paper: dtilde = ||yi-yj||^2 + 1. Attractive L = dtilde/(C+dtilde)
            // has grad 2C u / (C+dtilde)^2. Far L = 1/(1+dtilde) has
            // grad -2 u / (1+dtilde)^2.
            let dt = d2 + 1.0;
            let scale = match p.kind {
                PairKind::Near => 20.0 / (10.0 + dt).powi(2),
                PairKind::Mid => 20000.0 / (10000.0 + dt).powi(2),
                PairKind::Far => -2.0 / (1.0 + dt).powi(2),
            };
            if !scale.is_finite() {
                continue;
            }
            for h in 0..opts.lowdim {
                grad[(p.i, h)] += scale * delta[h];
                grad[(p.j, h)] -= scale * delta[h];
            }
        }
        let t = opts.lr / (1.0 + 0.01 * step as f64);
        for i in 0..n {
            for h in 0..opts.lowdim {
                y[(i, h)] -= t * grad[(i, h)];
                if !y[(i, h)].is_finite() {
                    return Err(LandfoldError::Msg("PaCMAP coordinate is not finite".into()));
                }
            }
        }
    }
    if opts.uniform {
        rank_uniform(&mut y);
    }
    Ok(y)
}

/// Empirical-CDF map of each column onto `{0, 1/(n-1), ..., 1}`.
pub fn rank_uniform(y: &mut Array2<f64>) {
    let n = y.nrows();
    if n < 2 {
        return;
    }
    let dim = y.ncols();
    for h in 0..dim {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| {
            y[(i, h)]
                .partial_cmp(&y[(j, h)])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let den = (n - 1) as f64;
        for (rank, &i) in idx.iter().enumerate() {
            y[(i, h)] = rank as f64 / den;
        }
    }
}

/// Place queries as a k-NN average of landmark low-D coordinates.
pub fn knn_project(
    landmarks: ArrayView2<f64>,
    landmark_ld: ArrayView2<f64>,
    queries: ArrayView2<f64>,
    metric: &dyn Metric,
    k: usize,
) -> Result<Array2<f64>> {
    if landmarks.nrows() != landmark_ld.nrows() {
        return Err(LandfoldError::Shape(
            "knn project: landmark HD/LD row counts differ",
        ));
    }
    if landmarks.ncols() != queries.ncols() {
        return Err(LandfoldError::Shape(
            "knn project: query dimension mismatch",
        ));
    }
    let n = landmarks.nrows();
    let kk = k.min(n).max(1);
    let dim = landmark_ld.ncols();
    let m = queries.nrows();
    let mut out = Array2::<f64>::zeros((m, dim));
    for q in 0..m {
        let qs = queries.row(q);
        let qsl = qs.as_slice().ok_or(LandfoldError::Shape(
            "knn project: query row is not contiguous",
        ))?;
        let mut nb: Vec<(f64, usize)> = Vec::with_capacity(n);
        for i in 0..n {
            let ls = landmarks.row(i);
            let lsl = ls.as_slice().ok_or(LandfoldError::Shape(
                "knn project: landmark row is not contiguous",
            ))?;
            nb.push((metric.dist(qsl, lsl)?, i));
        }
        nb.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut wsum = 0.0;
        let mut acc = vec![0.0; dim];
        for &(d, i) in nb.iter().take(kk) {
            let w = 1.0 / (d + 1e-8);
            wsum += w;
            for h in 0..dim {
                acc[h] += w * landmark_ld[(i, h)];
            }
        }
        if !(wsum > 0.0) {
            return Err(LandfoldError::Msg("knn project: zero weight".into()));
        }
        for h in 0..dim {
            out[(q, h)] = acc[h] / wsum;
            if !out[(q, h)].is_finite() {
                return Err(LandfoldError::Msg("knn project: non-finite".into()));
            }
        }
    }
    Ok(out)
}

fn build_pairs(
    dist: ArrayView2<f64>,
    k: usize,
    n_mid: usize,
    n_far: usize,
) -> Result<Vec<Pair>> {
    let n = dist.nrows();
    let mut pairs = Vec::new();
    let mut seen = vec![false; n * n];
    let mut push = |i: usize, j: usize, kind: PairKind, seen: &mut [bool]| {
        if i == j {
            return;
        }
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        let key = a * n + b;
        if seen[key] {
            return;
        }
        seen[key] = true;
        pairs.push(Pair { i: a, j: b, kind });
    };
    for i in 0..n {
        let mut order: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (dist[(i, j)], j))
            .collect();
        order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for &(d, j) in order.iter().take(k) {
            if !d.is_finite() {
                return Err(LandfoldError::Msg("PaCMAP neighbor distance is not finite".into()));
            }
            push(i, j, PairKind::Near, &mut seen);
        }
        let mid_start = k.min(order.len());
        let mid_end = (5 * k).min(order.len());
        if mid_end > mid_start {
            let step = ((mid_end - mid_start) / n_mid.max(1)).max(1);
            let mut taken = 0;
            let mut t = mid_start;
            while t < mid_end && taken < n_mid {
                push(i, order[t].1, PairKind::Mid, &mut seen);
                taken += 1;
                t += step;
            }
        }
        let far_start = (n / 2).min(order.len());
        if order.len() > far_start {
            let step = ((order.len() - far_start) / n_far.max(1)).max(1);
            let mut taken = 0;
            let mut t = far_start;
            while t < order.len() && taken < n_far {
                push(i, order[t].1, PairKind::Far, &mut seen);
                taken += 1;
                t += step;
            }
        }
    }
    if pairs.is_empty() {
        return Err(LandfoldError::Msg("PaCMAP built no pairs".into()));
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::Euclid;
    use ndarray::Array2;

    #[test]
    fn two_blobs_stay_apart() {
        let mut pts = Array2::<f64>::zeros((16, 4));
        for i in 0..8 {
            pts[(i, 0)] = 0.02 * i as f64;
            pts[(i + 8, 0)] = 6.0 + 0.02 * i as f64;
        }
        let y = pacmap_embed(
            pts.view(),
            &Euclid,
            &PacmapOpts {
                n_neighbors: 3,
                steps: 80,
                uniform: false,
                ..PacmapOpts::default()
            },
        )
        .unwrap();
        let mut ca = [0.0, 0.0];
        let mut cb = [0.0, 0.0];
        for i in 0..8 {
            ca[0] += y[(i, 0)];
            ca[1] += y[(i, 1)];
            cb[0] += y[(i + 8, 0)];
            cb[1] += y[(i + 8, 1)];
        }
        let gap = ((ca[0] - cb[0]).hypot(ca[1] - cb[1])) / 8.0;
        assert!(gap > 0.5, "PaCMAP gap {gap}");
    }

    #[test]
    fn rank_uniform_fills_the_unit_interval() {
        let mut y = Array2::from_shape_vec((5, 1), vec![3.0, -1.0, 0.0, 9.0, 2.0]).unwrap();
        rank_uniform(&mut y);
        let mut col: Vec<f64> = (0..5).map(|i| y[(i, 0)]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (i, v) in col.iter().enumerate() {
            assert!((v - i as f64 / 4.0).abs() < 1e-12);
        }
    }
}
