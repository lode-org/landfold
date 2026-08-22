//! Landscape chi: Laplacian eigenmaps of the energy-weighted minima graph.
//!
//! Sketch-map of a packing histogram does not encode barriers, so occupancy
//! of inherent structures cannot show Wales basins of attraction. Build the
//! k-NN graph in the high-D metric, weight each edge by
//! `exp(-(|E_i-E_j| + lambda * D_ij) / T)`, and take the first two nontrivial
//! eigenvectors of the symmetric normalized Laplacian (Belkin and Niyogi,
//! *Neural Comput.* **15**, 1373 (2003)).
//!
//! Do not flood the graph with a median L1 cut. On the Elja LJ38 book that
//! cut takes half of all pairs, ico becomes a degree-200 hub, and an
//! undirected committor dumps 399/400 families to ico. k-NN alone keeps the
//! steepest-descent split (GM 16, ico 52, other 332). The GM funnel is
//! small. Leftover-well occupancy cannot make it the deepest well.

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

fn knn_affinity(
    points: ArrayView2<f64>,
    energy: ArrayView1<f64>,
    metric: &dyn Metric,
    opts: &LandscapeOpts,
) -> Result<(Vec<f64>, usize)> {
    let n = points.nrows();
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
    Ok((affinity, n_edges))
}

/// Directed Metropolis committor on the same k-NN graph: q(src)=0, q(sink)=1.
pub fn landscape_committor(
    points: ArrayView2<f64>,
    energy: ArrayView1<f64>,
    metric: &dyn Metric,
    opts: &LandscapeOpts,
    src: usize,
    sink: usize,
) -> Result<Array1<f64>> {
    let n = points.nrows();
    if src >= n || sink >= n || src == sink {
        return Err(LandfoldError::Msg(
            "landscape committor src and sink must be distinct indices".into(),
        ));
    }
    let (aff, _) = knn_affinity(points, energy, metric, opts)?;
    // Metropolis: keep undirected support, direct the walk by energy
    let mut p = vec![0.0; n * n];
    for i in 0..n {
        let mut row = 0.0;
        for j in 0..n {
            if aff[i * n + j] == 0.0 {
                continue;
            }
            let de = (energy[j] - energy[i]).max(0.0);
            let w = (-de / opts.temperature).exp();
            p[i * n + j] = w;
            row += w;
        }
        if row > 0.0 {
            for j in 0..n {
                p[i * n + j] /= row;
            }
        }
    }
    let trans: Vec<usize> = (0..n).filter(|&i| i != src && i != sink).collect();
    let m = trans.len();
    let mut a = DMatrix::<f64>::zeros(m, m);
    let mut rhs = nalgebra::DVector::<f64>::zeros(m);
    for (ii, &i) in trans.iter().enumerate() {
        a[(ii, ii)] = 1.0;
        for (jj, &j) in trans.iter().enumerate() {
            a[(ii, jj)] -= p[i * n + j];
        }
        rhs[ii] = p[i * n + sink];
    }
    let q_t = a.lu().solve(&rhs).ok_or_else(|| {
        LandfoldError::Msg("landscape committor linear system is singular".into())
    })?;
    let mut q = Array1::<f64>::zeros(n);
    q[sink] = 1.0;
    for (ii, &i) in trans.iter().enumerate() {
        q[i] = q_t[ii].clamp(0.0, 1.0);
    }
    Ok(q)
}

/// Coordinates (q, (E-Emin)/(Emax-Emin)) so the two funnels are the axes.
pub fn landscape_qe(
    points: ArrayView2<f64>,
    energy: ArrayView1<f64>,
    metric: &dyn Metric,
    opts: &LandscapeOpts,
    src: usize,
    sink: usize,
) -> Result<(Array2<f64>, Array1<f64>)> {
    let q = landscape_committor(points, energy, metric, opts, src, sink)?;
    let emin = energy.iter().copied().fold(f64::INFINITY, f64::min);
    let emax = energy.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (emax - emin).max(1e-12);
    let n = points.nrows();
    let mut xy = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        xy[(i, 0)] = q[i];
        xy[(i, 1)] = (energy[i] - emin) / span;
    }
    Ok((xy, q))
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

    let (affinity, n_edges) = knn_affinity(points, energy, metric, opts)?;

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

/// Steepest-descent attractors, MDS of the barrier ultrametric, GM at the origin.
pub fn landscape_attractors(
    points: ArrayView2<f64>,
    energy: ArrayView1<f64>,
    metric: &dyn Metric,
    opts: &LandscapeOpts,
    src: usize,
    residual: f64,
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
    if src >= n {
        return Err(LandfoldError::Msg("landscape attractor src out of range".into()));
    }
    if opts.knn == 0 || opts.knn >= n {
        return Err(LandfoldError::Msg(format!(
            "landscape knn must be in 1..n-1 (got knn={}, n={})",
            opts.knn, n
        )));
    }
    let dim = points.ncols();
    let mut edges: Vec<(f64, usize, usize)> = Vec::new();
    let mut seen = vec![false; n * n];
    for i in 0..n {
        let a = points.row(i);
        let a = a.as_slice().ok_or(LandfoldError::Empty)?;
        let mut neigh: Vec<(f64, usize)> = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i == j {
                continue;
            }
            let b = points.row(j);
            let b = b.as_slice().ok_or(LandfoldError::Empty)?;
            neigh.push((metric.dist(a, b)?, j));
        }
        neigh.sort_by(|x, y| x.0.total_cmp(&y.0));
        for &(d, j) in neigh.iter().take(opts.knn) {
            let lo = i.min(j);
            let hi = i.max(j);
            let key = lo * n + hi;
            if seen[key] {
                continue;
            }
            seen[key] = true;
            let h = energy[i].max(energy[j]) + opts.lambda * d;
            edges.push((h, i, j));
        }
    }
    edges.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut parent: Vec<usize> = (0..n).collect();
    let mut best = energy.to_vec();
    for &(_, i, j) in &edges {
        if energy[j] < best[i] {
            best[i] = energy[j];
            parent[i] = j;
        }
        if energy[i] < best[j] {
            best[j] = energy[i];
            parent[j] = i;
        }
    }
    let mut attract = vec![0usize; n];
    for i in 0..n {
        let mut x = i;
        let mut guard = 0;
        while parent[x] != x && guard < n {
            x = parent[x];
            guard += 1;
        }
        attract[i] = x;
    }
    let mut roots: Vec<usize> = attract.clone();
    roots.sort_unstable();
    roots.dedup();
    let na = roots.len();
    let mut amap = vec![n; n];
    for (k, &r) in roots.iter().enumerate() {
        amap[r] = k;
    }

    let mut uf: Vec<usize> = (0..n).collect();
    fn find(uf: &mut [usize], mut x: usize) -> usize {
        while uf[x] != x {
            uf[x] = uf[uf[x]];
            x = uf[x];
        }
        x
    }
    let mut merge = vec![f64::NAN; na * na];
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    for &(h, i, j) in &edges {
        let ri = find(&mut uf, i);
        let rj = find(&mut uf, j);
        if ri == rj {
            continue;
        }
        for &a in &members[ri] {
            if amap[a] < n {
                for &b in &members[rj] {
                    if amap[b] < n {
                        merge[amap[a] * na + amap[b]] = h;
                        merge[amap[b] * na + amap[a]] = h;
                    }
                }
            }
        }
        uf[rj] = ri;
        let mut right = std::mem::take(&mut members[rj]);
        members[ri].append(&mut right);
    }
    let mut dist = Array2::<f64>::zeros((na, na));
    let mut dmax = 0.0;
    for a in 0..na {
        for b in 0..na {
            if a == b {
                continue;
            }
            let mut h = merge[a * na + b];
            if !h.is_finite() {
                h = dmax;
            }
            dmax = dmax.max(h);
            let emin = energy[roots[a]].min(energy[roots[b]]);
            dist[(a, b)] = (h - emin).max(0.0);
        }
    }
    for a in 0..na {
        for b in 0..na {
            if a != b && merge[a * na + b].is_nan() {
                dist[(a, b)] = (dmax + 1.0 - energy[roots[a]].min(energy[roots[b]])).max(0.0);
            }
        }
    }
    let (xy_a, _) = crate::mds::classical_mds(dist.view(), 2)?;
    let gm_k = amap[attract[src]];
    if gm_k >= n {
        return Err(LandfoldError::Msg("src is not an attractor root".into()));
    }
    let mut xy_a = xy_a;
    let ox = xy_a[(gm_k, 0)];
    let oy = xy_a[(gm_k, 1)];
    for k in 0..na {
        xy_a[(k, 0)] -= ox;
        xy_a[(k, 1)] -= oy;
    }
    let sink = if src == 0 && n > 1 { 1 } else { src.saturating_sub(1) };
    let ico_k = amap[attract[sink]];
    if ico_k < n {
        let vx = xy_a[(ico_k, 0)];
        let vy = xy_a[(ico_k, 1)];
        let nrm = (vx * vx + vy * vy).sqrt();
        if nrm > 1e-15 {
            let c = vx / nrm;
            let s = -vy / nrm;
            for k in 0..na {
                let x = xy_a[(k, 0)];
                let y = xy_a[(k, 1)];
                xy_a[(k, 0)] = c * x - s * y;
                xy_a[(k, 1)] = s * x + c * y;
            }
        }
    }

    let mut coords = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let k = amap[attract[i]];
        coords[(i, 0)] = xy_a[(k, 0)];
        coords[(i, 1)] = xy_a[(k, 1)];
    }
    if residual > 0.0 {
        for &root in &roots {
            let idx: Vec<usize> = (0..n).filter(|&i| attract[i] == root).collect();
            if idx.len() < 2 {
                continue;
            }
            let mut mean = vec![0.0; dim];
            for &i in &idx {
                for d in 0..dim {
                    mean[d] += points[(i, d)];
                }
            }
            let inv = 1.0 / idx.len() as f64;
            for m in &mut mean {
                *m *= inv;
            }
            let mut xc = DMatrix::<f64>::zeros(idx.len(), dim);
            for (r, &i) in idx.iter().enumerate() {
                for d in 0..dim {
                    xc[(r, d)] = points[(i, d)] - mean[d];
                }
            }
            let svd = nalgebra::SVD::new(xc, true, false);
            if let (Some(u), Some(s)) = (svd.u, Some(svd.singular_values)) {
                let kmax = 2.min(s.len()).min(u.ncols());
                for (r, &i) in idx.iter().enumerate() {
                    for k in 0..kmax {
                        coords[(i, k)] += residual * u[(r, k)] * s[k];
                    }
                }
            }
        }
    }
    let mut ev = Array1::<f64>::zeros(2);
    ev[0] = na as f64;
    ev[1] = edges.len() as f64;
    Ok((
        coords,
        LandscapeReport {
            eigenvalues: ev,
            n_edges: edges.len(),
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
        let opts = LandscapeOpts {
            knn: 2,
            ..LandscapeOpts::default()
        };
        let (xy, rep) = landscape_embed(pts.view(), e.view(), &Euclid, &opts).expect("embed");
        assert_eq!(xy.nrows(), 6);
        assert_eq!(xy.ncols(), 2);
        assert!(rep.n_edges > 0);
        let left = (xy.row(0).to_owned() + xy.row(1).to_owned() + xy.row(2).to_owned()) / 3.0;
        let right = (xy.row(3).to_owned() + xy.row(4).to_owned() + xy.row(5).to_owned()) / 3.0;
        let sep = ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2)).sqrt();
        let d_in = ((xy[(0, 0)] - xy[(1, 0)]).powi(2) + (xy[(0, 1)] - xy[(1, 1)]).powi(2)).sqrt();
        assert!(sep > 2.0 * d_in, "sep={sep} din={d_in}");
    }

    #[test]
    fn committor_splits_wells() {
        let pts = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 0.0],
            [5.1, 0.0],
            [5.0, 0.1],
        ];
        let e = array![-2.0, -1.8, -1.7, -1.9, -1.7, -1.6];
        let opts = LandscapeOpts {
            knn: 2,
            temperature: 0.2,
            ..LandscapeOpts::default()
        };
        let q = landscape_committor(pts.view(), e.view(), &Euclid, &opts, 0, 3).expect("q");
        assert!((q[0] - 0.0).abs() < 1e-12);
        assert!((q[3] - 1.0).abs() < 1e-12);
        assert!(q[1] < 0.5, "left well q={}", q[1]);
        assert!(q[4] > 0.5, "right well q={}", q[4]);
    }

    #[test]
    fn attractors_put_gm_at_origin() {
        let pts = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 0.0],
            [5.1, 0.0],
            [5.0, 0.1],
        ];
        let e = array![-2.0, -1.8, -1.7, -1.9, -1.7, -1.6];
        let opts = LandscapeOpts {
            knn: 2,
            ..LandscapeOpts::default()
        };
        let (xy, _) =
            landscape_attractors(pts.view(), e.view(), &Euclid, &opts, 0, 0.0).expect("attr");
        let r0 = (xy[(0, 0)] * xy[(0, 0)] + xy[(0, 1)] * xy[(0, 1)]).sqrt();
        assert!(r0 < 0.2, "GM not at origin: {:?}", xy.row(0));
        let sep = ((xy[(0, 0)] - xy[(3, 0)]).powi(2) + (xy[(0, 1)] - xy[(3, 1)]).powi(2)).sqrt();
        assert!(sep > 0.5, "funnels stacked sep={sep}");
    }
}
