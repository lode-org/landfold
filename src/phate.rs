//! PHATE: Potential of Heat-diffusion for Affinity-based Trajectory Embedding.
//!
//! Moon, van Dijk, Wang, Gigante, Burkhardt, Chen, Hirn, Wolf, Krishnaswamy,
//! *Nat. Biotechnol.* **37**, 1482 (2019),
//! <https://doi.org/10.1038/s41587-019-0336-3>.
//! The kernel is the locally scaled diffusion-map affinity of Rohrdanz,
//! Zheng, Maggioni and Clementi, *J. Chem. Phys.* **134**, 124116 (2011),
//! <https://doi.org/10.1063/1.3569857> (k-NN bandwidth, alpha-decay).
//! The diffusion operator is Coifman and Lafon, *Appl. Comput. Harmon. Anal.*
//! **21**, 5 (2006), <https://doi.org/10.1016/j.acha.2006.04.006>. PHATE
//! replaces the diffusion-map eigenplot with the information potential
//! distance and a Torgerson MDS of that distance.

use nalgebra::{DMatrix, SymmetricEigen};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{LandfoldError, Result};
use crate::mds::classical_mds;
use crate::metric::Metric;
use crate::pairwise::pairwise;

const POT_FLOOR: f64 = 1e-12;
const EXP_CLIP: f64 = 60.0;

#[derive(Clone, Debug)]
pub struct PhateOpts {
    pub knn: usize,
    pub decay: f64,
    pub t: Option<usize>,
    pub t_max: usize,
    pub gamma: f64,
    pub lowdim: usize,
}

impl Default for PhateOpts {
    fn default() -> Self {
        Self {
            knn: 5,
            decay: 40.0,
            t: None,
            t_max: 100,
            gamma: 1.0,
            lowdim: 2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PhateModel {
    pub knn: usize,
    pub decay: f64,
    pub gamma: f64,
    pub t: usize,
    pub bandwidths: Array1<f64>,
    pub potential: Array2<f64>,
    pub p_tm1: Array2<f64>,
    pub coords: Array2<f64>,
}

impl PhateOpts {
    fn validate(&self, n: usize) -> Result<()> {
        if n == 0 {
            return Err(LandfoldError::Empty);
        }
        if self.lowdim == 0 || self.lowdim > n {
            return Err(LandfoldError::LowDim {
                low: self.lowdim,
                high: n,
            });
        }
        if self.knn == 0 || self.knn >= n {
            return Err(LandfoldError::Msg(format!(
                "PHATE knn must be in 1..n-1 (got knn={}, n={})",
                self.knn, n
            )));
        }
        if !(self.decay > 0.0) || !self.decay.is_finite() {
            return Err(LandfoldError::Msg("PHATE decay must be positive".into()));
        }
        if !self.gamma.is_finite() {
            return Err(LandfoldError::Msg("PHATE gamma must be finite".into()));
        }
        if self.t_max == 0 {
            return Err(LandfoldError::Msg("PHATE t_max must be positive".into()));
        }
        Ok(())
    }
}

/// Fit PHATE on `points` and return the low-D coordinates plus the model
/// needed to place new rows (Nyström potential, then metric MDS).
pub fn phate_embed(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    opts: &PhateOpts,
) -> Result<(Array2<f64>, PhateModel)> {
    let n = points.nrows();
    opts.validate(n)?;
    let dist = pairwise(points, metric)?;
    let bandwidths = knn_bandwidth(dist.view(), opts.knn)?;
    let kernel = alpha_kernel(dist.view(), bandwidths.view(), opts.decay)?;
    let (p_t, p_tm1, t) = diffuse(&kernel, opts.t, opts.t_max)?;
    let potential = potential_from_pt(&p_t, opts.gamma)?;
    let pot_dist = pairwise_rows(potential.view())?;
    let (coords, _) = classical_mds(pot_dist.view(), opts.lowdim)?;
    let model = PhateModel {
        knn: opts.knn,
        decay: opts.decay,
        gamma: opts.gamma,
        t,
        bandwidths,
        potential,
        p_tm1,
        coords: coords.clone(),
    };
    Ok((coords, model))
}

/// Place `new_points` on a fitted PHATE map. `landmark_ld` is the stored
/// embedding of the landmarks (sign/rotation of a fresh MDS are discarded).
/// New rows are a Nyström average of those coordinates (Lafon, Keller,
/// Coifman, *IEEE Trans. Pattern Anal. Mach. Intell.* **28**, 1784 (2006)).
pub fn phate_project(
    landmarks: ArrayView2<f64>,
    landmark_ld: ArrayView2<f64>,
    new_points: ArrayView2<f64>,
    metric: &dyn Metric,
    opts: &PhateOpts,
) -> Result<Array2<f64>> {
    if landmarks.nrows() != landmark_ld.nrows() {
        return Err(LandfoldError::Shape(
            "PHATE project: landmark HD and LD row counts differ",
        ));
    }
    if landmark_ld.ncols() != opts.lowdim {
        return Err(LandfoldError::Shape(
            "PHATE project: landmark LD width is not lowdim",
        ));
    }
    if landmarks.ncols() != new_points.ncols() {
        return Err(LandfoldError::Shape(
            "PHATE project: new points do not match landmark dimension",
        ));
    }
    let (_, model) = phate_embed(landmarks, metric, opts)?;
    let m = new_points.nrows();
    let dim = opts.lowdim;
    let mut out = Array2::<f64>::zeros((m, dim));
    for i in 0..m {
        let row = new_points.row(i);
        let y = place_one(row, landmarks, landmark_ld, &model, metric, dim)?;
        for h in 0..dim {
            out[(i, h)] = y[h];
        }
    }
    Ok(out)
}

fn knn_bandwidth(dist: ArrayView2<f64>, knn: usize) -> Result<Array1<f64>> {
    let n = dist.nrows();
    let mut bw = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut row: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| dist[(i, j)]).collect();
        row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let eps = row[knn - 1];
        if !eps.is_finite() || eps <= 0.0 {
            return Err(LandfoldError::Msg(format!(
                "PHATE k-NN bandwidth at row {i} is not positive"
            )));
        }
        bw[i] = eps;
    }
    Ok(bw)
}

fn alpha_kernel(dist: ArrayView2<f64>, bw: ArrayView1<f64>, decay: f64) -> Result<Array2<f64>> {
    let n = dist.nrows();
    let mut k = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i == j {
                k[(i, j)] = 1.0;
                continue;
            }
            let r = dist[(i, j)] / bw[i];
            let e = r.powf(decay);
            k[(i, j)] = if e >= EXP_CLIP { 0.0 } else { (-e).exp() };
        }
    }
    for i in 0..n {
        for j in 0..i {
            let s = 0.5 * (k[(i, j)] + k[(j, i)]);
            k[(i, j)] = s;
            k[(j, i)] = s;
        }
    }
    Ok(k)
}

/// Symmetric kernel -> P^t and P^{t-1} via the normalized Laplacian trick.
fn diffuse(
    kernel: &Array2<f64>,
    t_fixed: Option<usize>,
    t_max: usize,
) -> Result<(Array2<f64>, Array2<f64>, usize)> {
    let n = kernel.nrows();
    let mut deg = vec![0.0; n];
    for i in 0..n {
        let s: f64 = (0..n).map(|j| kernel[(i, j)]).sum();
        if !(s > 0.0) || !s.is_finite() {
            return Err(LandfoldError::Msg(format!(
                "PHATE degree at row {i} is not positive"
            )));
        }
        deg[i] = s;
    }
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        let si = deg[i].sqrt();
        for j in 0..n {
            m[i * n + j] = kernel[(i, j)] / (si * deg[j].sqrt());
        }
    }
    let dm = DMatrix::<f64>::from_row_slice(n, n, &m);
    let eigen = SymmetricEigen::new(dm);
    let mut evals: Vec<f64> = (0..n).map(|k| eigen.eigenvalues[k]).collect();
    for e in &mut evals {
        if !e.is_finite() {
            return Err(LandfoldError::Msg(
                "PHATE diffusion eigenvalues must be finite".into(),
            ));
        }
        if *e < 0.0 {
            *e = 0.0;
        }
        if *e > 1.0 {
            *e = 1.0;
        }
    }
    let t = match t_fixed {
        Some(t) if t >= 1 => t,
        Some(_) => {
            return Err(LandfoldError::Msg("PHATE t must be at least 1".into()));
        }
        None => auto_t(&evals, t_max),
    };
    let p_t = assemble_p(&eigen.eigenvectors, &evals, &deg, t)?;
    let p_tm1 = if t == 1 {
        Array2::<f64>::from_shape_fn((n, n), |(i, j)| if i == j { 1.0 } else { 0.0 })
    } else {
        assemble_p(&eigen.eigenvectors, &evals, &deg, t - 1)?
    };
    Ok((p_t, p_tm1, t))
}

/// Second eigenfunction of the locally scaled walk (Rohrdanz / Coifman).
/// This is the data-driven slow coordinate `ψ` in the gap-split theorem.
pub fn slow_mode(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    knn: usize,
    decay: f64,
) -> Result<Array1<f64>> {
    let n = points.nrows();
    if n < 3 {
        return Err(LandfoldError::Msg(
            "slow mode needs at least three points".into(),
        ));
    }
    if knn == 0 || knn >= n {
        return Err(LandfoldError::Msg(format!(
            "slow-mode knn must be in 1..n-1 (got knn={}, n={})",
            knn, n
        )));
    }
    let dist = pairwise(points, metric)?;
    let bandwidths = knn_bandwidth(dist.view(), knn)?;
    let kernel = alpha_kernel(dist.view(), bandwidths.view(), decay)?;
    let mut deg = vec![0.0; n];
    for i in 0..n {
        let s: f64 = (0..n).map(|j| kernel[(i, j)]).sum();
        if !(s > 0.0) || !s.is_finite() {
            return Err(LandfoldError::Msg(format!(
                "slow-mode degree at row {i} is not positive"
            )));
        }
        deg[i] = s;
    }
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        let si = deg[i].sqrt();
        for j in 0..n {
            m[i * n + j] = kernel[(i, j)] / (si * deg[j].sqrt());
        }
    }
    let dm = DMatrix::<f64>::from_row_slice(n, n, &m);
    let eigen = SymmetricEigen::new(dm);
    let mut order: Vec<(f64, usize)> = (0..n)
        .map(|k| (eigen.eigenvalues[k], k))
        .collect();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let src = order[1].1;
    let mut psi = Array1::<f64>::zeros(n);
    for i in 0..n {
        psi[i] = eigen.eigenvectors[(i, src)];
        if !psi[i].is_finite() {
            return Err(LandfoldError::Msg(
                "slow-mode coordinate is not finite".into(),
            ));
        }
    }
    Ok(psi)
}

fn assemble_p(
    evecs: &DMatrix<f64>,
    evals: &[f64],
    deg: &[f64],
    t: usize,
) -> Result<Array2<f64>> {
    let n = evals.len();
    let mut out = Array2::<f64>::zeros((n, n));
    let tf = t as i32;
    for i in 0..n {
        let si = deg[i].sqrt();
        for j in 0..n {
            let sj = deg[j].sqrt();
            let mut acc = 0.0;
            for k in 0..n {
                let lam = evals[k].powi(tf);
                acc += evecs[(i, k)] * lam * evecs[(j, k)];
            }
            let val = acc * (sj / si);
            if !val.is_finite() {
                return Err(LandfoldError::Msg(
                    "PHATE P^t entry is not finite".into(),
                ));
            }
            out[(i, j)] = val.max(0.0);
        }
    }
    Ok(out)
}

fn auto_t(evals: &[f64], t_max: usize) -> usize {
    let mut ev: Vec<f64> = evals.iter().map(|e| e.abs()).collect();
    ev.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut entropy = Vec::with_capacity(t_max);
    let mut powered = ev.clone();
    for _ in 0..t_max {
        let z: f64 = powered.iter().sum();
        if !(z > 0.0) {
            entropy.push(0.0);
        } else {
            let mut h = 0.0;
            for &p in &powered {
                let q = p / z + f64::EPSILON;
                h -= q * q.ln();
            }
            entropy.push(h);
        }
        for (p, e) in powered.iter_mut().zip(ev.iter()) {
            *p *= *e;
        }
    }
    find_knee(&entropy).max(1)
}

/// Two-line residual knee (Moon et al. official `phate.vne.find_knee_point`).
/// Returns a 1-based diffusion time.
fn find_knee(y: &[f64]) -> usize {
    let n = y.len();
    if n < 3 {
        return n.max(1);
    }
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut best_i = 1;
    let mut best_e = f64::INFINITY;
    for br in 1..n - 1 {
        let (ml, bl) = match fit_line(&x[..=br], &y[..=br]) {
            Some(ab) => ab,
            None => continue,
        };
        let (mr, brt) = match fit_line(&x[br..], &y[br..]) {
            Some(ab) => ab,
            None => continue,
        };
        let mut e = 0.0;
        for i in 0..=br {
            e += (ml * x[i] + bl - y[i]).abs();
        }
        for i in br..n {
            e += (mr * x[i] + brt - y[i]).abs();
        }
        if e < best_e {
            best_e = e;
            best_i = br;
        }
    }
    // y[0] is t = 1; official code returns the 0-based index as t.
    (best_i + 1).max(1)
}

fn fit_line(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let n = x.len() as f64;
    if x.len() < 2 {
        return None;
    }
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        sx += xi;
        sy += yi;
        sxx += xi * xi;
        sxy += xi * yi;
    }
    let det = n * sxx - sx * sx;
    if det.abs() < 1e-18 {
        return None;
    }
    let m = (n * sxy - sx * sy) / det;
    let b = (sxx * sy - sx * sxy) / det;
    Some((m, b))
}

fn potential_from_pt(p_t: &Array2<f64>, gamma: f64) -> Result<Array2<f64>> {
    let n = p_t.nrows();
    let mut u = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let p = p_t[(i, j)].max(POT_FLOOR);
            let val = if (gamma - 1.0).abs() < 1e-12 {
                -p.ln()
            } else if gamma.abs() < 1e-12 {
                p.sqrt()
            } else {
                p.powf(gamma)
            };
            if !val.is_finite() {
                return Err(LandfoldError::Msg(
                    "PHATE potential entry is not finite".into(),
                ));
            }
            u[(i, j)] = val;
        }
    }
    Ok(u)
}

fn pairwise_rows(u: ArrayView2<f64>) -> Result<Array2<f64>> {
    let n = u.nrows();
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..i {
            let mut s = 0.0;
            for k in 0..u.ncols() {
                let e = u[(i, k)] - u[(j, k)];
                s += e * e;
            }
            let v = s.sqrt();
            if !v.is_finite() {
                return Err(LandfoldError::Msg(
                    "PHATE potential distance is not finite".into(),
                ));
            }
            d[(i, j)] = v;
            d[(j, i)] = v;
        }
    }
    Ok(d)
}

fn place_one(
    z: ArrayView1<f64>,
    landmarks: ArrayView2<f64>,
    landmark_ld: ArrayView2<f64>,
    model: &PhateModel,
    metric: &dyn Metric,
    dim: usize,
) -> Result<Vec<f64>> {
    let n = landmarks.nrows();
    let zsl = z.as_slice().ok_or(LandfoldError::Shape(
        "PHATE project: new row is not contiguous",
    ))?;
    let mut drow = vec![0.0; n];
    for j in 0..n {
        let lj = landmarks.row(j);
        let ljs = lj.as_slice().ok_or(LandfoldError::Shape(
            "PHATE project: landmark row is not contiguous",
        ))?;
        drow[j] = metric.dist(zsl, ljs)?;
    }
    let mut pair: Vec<(f64, usize)> = drow.iter().copied().zip(0..n).collect();
    pair.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let kth = model.knn.min(n - 1).max(1);
    let eps = pair[kth - 1].0.max(1e-12);
    let mut aff = vec![0.0; n];
    let mut zsum = 0.0;
    for j in 0..n {
        let e = (drow[j] / eps).powf(model.decay);
        let a = if e >= EXP_CLIP { 0.0 } else { (-e).exp() };
        aff[j] = a;
        zsum += a;
    }
    if !(zsum > 0.0) {
        return Err(LandfoldError::Msg(
            "PHATE project: new point has zero affinity".into(),
        ));
    }
    for a in &mut aff {
        *a /= zsum;
    }
    // Landmark coordinates already carry P^t through the potential MDS.
    // A second P^{t-1} on the query washes the map out to a line.
    let mut y = vec![0.0; dim];
    for j in 0..n {
        let w = aff[j];
        for h in 0..dim {
            y[h] += w * landmark_ld[(j, h)];
        }
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(LandfoldError::Msg(
            "PHATE OOS placement is not finite".into(),
        ));
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::Euclid;
    use ndarray::Array2;

    fn two_blobs(n_each: usize, dim: usize, sep: f64) -> Array2<f64> {
        let mut p = Array2::<f64>::zeros((2 * n_each, dim));
        for i in 0..n_each {
            p[(i, 0)] = 0.02 * i as f64;
            p[(i + n_each, 0)] = sep + 0.02 * i as f64;
            if dim > 1 {
                p[(i + n_each, 1)] = 0.15;
            }
        }
        p
    }

    #[test]
    fn two_blobs_split() {
        let pts = two_blobs(12, 8, 6.0);
        let opts = PhateOpts {
            knn: 4,
            decay: 10.0,
            t: Some(4),
            ..PhateOpts::default()
        };
        let (y, model) = phate_embed(pts.view(), &Euclid, &opts).unwrap();
        assert_eq!(model.t, 4);
        let mut ca = [0.0, 0.0];
        let mut cb = [0.0, 0.0];
        for i in 0..12 {
            ca[0] += y[(i, 0)];
            ca[1] += y[(i, 1)];
            cb[0] += y[(i + 12, 0)];
            cb[1] += y[(i + 12, 1)];
        }
        for c in [&mut ca, &mut cb] {
            c[0] /= 12.0;
            c[1] /= 12.0;
        }
        let gap = ((ca[0] - cb[0]).hypot(ca[1] - cb[1])).abs();
        let mut intra = 0.0;
        let mut n_in = 0;
        for i in 0..12 {
            for j in (i + 1)..12 {
                intra += (y[(i, 0)] - y[(j, 0)]).hypot(y[(i, 1)] - y[(j, 1)]);
                intra += (y[(i + 12, 0)] - y[(j + 12, 0)]).hypot(y[(i + 12, 1)] - y[(j + 12, 1)]);
                n_in += 2;
            }
        }
        intra /= n_in as f64;
        assert!(
            gap > 2.0 * intra,
            "PHATE gap {gap} must beat twice intra {intra}"
        );
    }

    #[test]
    fn project_lands_near_home_blob() {
        let pts = two_blobs(10, 6, 5.0);
        let opts = PhateOpts {
            knn: 3,
            decay: 8.0,
            t: Some(3),
            ..PhateOpts::default()
        };
        let (y, _) = phate_embed(pts.view(), &Euclid, &opts).unwrap();
        let mut q = Array2::<f64>::zeros((2, 6));
        q[(0, 0)] = 0.05;
        q[(1, 0)] = 5.05;
        q[(1, 1)] = 0.15;
        let proj = phate_project(pts.view(), y.view(), q.view(), &Euclid, &opts).unwrap();
        let mut ca = [0.0, 0.0];
        let mut cb = [0.0, 0.0];
        for i in 0..10 {
            ca[0] += y[(i, 0)];
            ca[1] += y[(i, 1)];
            cb[0] += y[(i + 10, 0)];
            cb[1] += y[(i + 10, 1)];
        }
        for c in [&mut ca, &mut cb] {
            c[0] /= 10.0;
            c[1] /= 10.0;
        }
        let d0a = (proj[(0, 0)] - ca[0]).hypot(proj[(0, 1)] - ca[1]);
        let d0b = (proj[(0, 0)] - cb[0]).hypot(proj[(0, 1)] - cb[1]);
        let d1a = (proj[(1, 0)] - ca[0]).hypot(proj[(1, 1)] - ca[1]);
        let d1b = (proj[(1, 0)] - cb[0]).hypot(proj[(1, 1)] - cb[1]);
        assert!(d0a < d0b, "query in blob A must land nearer A ({d0a} vs {d0b})");
        assert!(d1b < d1a, "query in blob B must land nearer B ({d1b} vs {d1a})");
        let xmin = y.column(0).iter().copied().fold(f64::INFINITY, f64::min);
        let xmax = y.column(0).iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ymin = y.column(1).iter().copied().fold(f64::INFINITY, f64::min);
        let ymax = y.column(1).iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for i in 0..2 {
            assert!(proj[(i, 0)] >= xmin - 1e-6 && proj[(i, 0)] <= xmax + 1e-6);
            assert!(proj[(i, 1)] >= ymin - 1e-6 && proj[(i, 1)] <= ymax + 1e-6);
        }
    }

    #[test]
    fn knee_on_exponential_is_interior() {
        let y: Vec<f64> = (0..20).map(|i| (-(i as f64) / 10.0).exp()).collect();
        let k = find_knee(&y);
        assert!(k >= 2 && k <= 18, "knee {k} should sit inside the decay");
    }
}
