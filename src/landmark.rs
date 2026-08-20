//! Farthest-point landmarks.
//!
//! Gonzalez's 2-approximate k-center traversal (Gonzalez, *Theor. Comput.
//! Sci.* **38**, 293 (1985), <https://doi.org/10.1016/0304-3975(85)90224-5>),
//! optionally reweighted.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashSet;

use crate::error::{LandfoldError, Result};
use crate::metric::Metric;

/// C++ `dimlandmark -mode`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LandmarkMode {
    /// Equally spaced along the input order.
    Stride,
    /// Uniform random (optionally unique).
    Random,
    /// Gonzalez farthest-point (C++ default).
    MinMax,
    /// Mix of random and farthest-point, `gamma -> 0` random, large `gamma` minmax.
    Resample { gamma: f64 },
    /// Minmax cover, Voronoi masses, then sample \(P^\gamma\).
    Staged { gamma: f64 },
}

impl LandmarkMode {
    pub fn from_cli(spec: &str, gamma: f64) -> Result<Self> {
        match spec.trim().to_ascii_lowercase().as_str() {
            "stride" => Ok(Self::Stride),
            "random" => Ok(Self::Random),
            "minmax" | "fps" => Ok(Self::MinMax),
            "resample" => {
                if !gamma.is_finite() || gamma <= 0.0 {
                    return Err(LandfoldError::Msg(
                        "resample gamma must be finite and > 0".into(),
                    ));
                }
                Ok(Self::Resample { gamma })
            }
            "staged" => {
                if !gamma.is_finite() || gamma <= 0.0 {
                    return Err(LandfoldError::Msg(
                        "staged gamma must be finite and > 0".into(),
                    ));
                }
                Ok(Self::Staged { gamma })
            }
            _ => Err(LandfoldError::Parse(format!(
                "landmark mode `{spec}` (stride|random|minmax|resample|staged)"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Landmarks {
    pub index: Vec<usize>,
    pub points: Array2<f64>,
    pub weights: Array1<f64>,
}

fn validate_landmark_inputs(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    weights: Option<ArrayView1<f64>>,
) -> Result<()> {
    if points.iter().any(|&value| !value.is_finite()) {
        return Err(LandfoldError::Msg(
            "landmark coordinates must be finite".into(),
        ));
    }
    if let Some(expected) = metric.dim()
        && points.ncols() != expected
    {
        return Err(LandfoldError::MetricSize {
            left: points.ncols(),
            right: expected,
        });
    }
    if let Some(w) = weights {
        if w.len() != points.nrows() {
            return Err(LandfoldError::Shape("landmark weight length"));
        }
        if w.iter().any(|&value| !value.is_finite() || value < 0.0) {
            return Err(LandfoldError::Msg(
                "landmark weights must be finite and nonnegative".into(),
            ));
        }
    }
    Ok(())
}

pub fn farthest_point(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    k: usize,
    weights: Option<ArrayView1<f64>>,
    seed: usize,
) -> Result<Landmarks> {
    let n = points.nrows();
    if k == 0 || k > n {
        return Err(LandfoldError::LowDim { low: k, high: n });
    }
    validate_landmark_inputs(points, metric, weights)?;
    let mut min_d = vec![f64::INFINITY; n];
    let mut selected = vec![false; n];
    let mut chosen = Vec::with_capacity(k);
    let start = seed % n;
    chosen.push(start);
    selected[start] = true;
    update_min_d(points, metric, start, &mut min_d)?;
    farthest_from_chosen(
        points,
        metric,
        k,
        weights,
        &mut chosen,
        &mut selected,
        &mut min_d,
    )?;
    pack_landmarks(points, k, weights, &chosen)
}

/// Gonzalez farthest-point that pins the leading `ifirst` rows, then
/// fills the rest. Used so a figure cannot drop a named reference.
pub fn farthest_point_ifirst(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    k: usize,
    weights: Option<ArrayView1<f64>>,
    ifirst: usize,
) -> Result<Landmarks> {
    let n = points.nrows();
    if k == 0 || k > n {
        return Err(LandfoldError::LowDim { low: k, high: n });
    }
    if ifirst == 0 {
        return farthest_point(points, metric, k, weights, 0);
    }
    validate_landmark_inputs(points, metric, weights)?;
    let pin = ifirst.min(k).min(n);
    let mut min_d = vec![f64::INFINITY; n];
    let mut selected = vec![false; n];
    let mut chosen = Vec::with_capacity(k);
    for (i, selected_i) in selected.iter_mut().enumerate().take(pin) {
        chosen.push(i);
        *selected_i = true;
        update_min_d(points, metric, i, &mut min_d)?;
    }
    farthest_from_chosen(
        points,
        metric,
        k,
        weights,
        &mut chosen,
        &mut selected,
        &mut min_d,
    )?;
    pack_landmarks(points, k, weights, &chosen)
}

/// C++ `dimlandmark` selection. `MinMax` with `ifirst == 0` is Gonzalez.
pub fn select_landmarks(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    k: usize,
    weights: Option<ArrayView1<f64>>,
    seed: usize,
    ifirst: Option<usize>,
    unique: bool,
    mode: LandmarkMode,
) -> Result<Landmarks> {
    let n = points.nrows();
    if k == 0 || k > n {
        return Err(LandfoldError::LowDim { low: k, high: n });
    }
    validate_landmark_inputs(points, metric, weights)?;
    match mode {
        LandmarkMode::MinMax => match ifirst {
            Some(pin) if pin > 0 => farthest_point_ifirst(points, metric, k, weights, pin),
            _ => farthest_point(points, metric, k, weights, seed),
        },
        LandmarkMode::Stride => landmarks_stride(points, k, weights),
        LandmarkMode::Random => landmarks_random(points, k, weights, seed, unique),
        LandmarkMode::Resample { gamma } => {
            landmarks_resample(points, metric, k, weights, seed, unique, gamma, ifirst)
        }
        LandmarkMode::Staged { gamma } => {
            landmarks_staged(points, metric, k, weights, seed, unique, gamma)
        }
    }
}

fn landmarks_stride(
    points: ArrayView2<f64>,
    k: usize,
    weights: Option<ArrayView1<f64>>,
) -> Result<Landmarks> {
    let n = points.nrows();
    let stride = (n / k).max(1);
    let mut chosen = Vec::with_capacity(k);
    let mut seen = HashSet::new();
    for i in 0..k {
        let mut idx = (i * stride).min(n - 1);
        while seen.contains(&idx) {
            idx = (idx + 1) % n;
        }
        seen.insert(idx);
        chosen.push(idx);
    }
    pack_landmarks(points, k, weights, &chosen)
}

fn landmarks_random(
    points: ArrayView2<f64>,
    k: usize,
    weights: Option<ArrayView1<f64>>,
    seed: usize,
    unique: bool,
) -> Result<Landmarks> {
    let n = points.nrows();
    if unique && k > n {
        return Err(LandfoldError::LowDim { low: k, high: n });
    }
    let mut rng = Lcg::new(seed);
    let mut chosen = Vec::with_capacity(k);
    let mut seen = HashSet::new();
    let mut guard = 0usize;
    while chosen.len() < k {
        let idx = rng.below(n);
        if unique && !seen.insert(idx) {
            guard += 1;
            if guard > n.saturating_mul(32).max(32) {
                return Err(LandfoldError::Msg(
                    "unique random landmarks failed to fill k".into(),
                ));
            }
            continue;
        }
        chosen.push(idx);
    }
    pack_landmarks(points, k, weights, &chosen)
}

fn landmarks_resample(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    k: usize,
    weights: Option<ArrayView1<f64>>,
    seed: usize,
    unique: bool,
    gamma: f64,
    ifirst: Option<usize>,
) -> Result<Landmarks> {
    let n = points.nrows();
    let mut rng = Lcg::new(seed);
    let start = match ifirst {
        Some(i) if i < n => i,
        _ => rng.below(n),
    };
    let mut chosen = vec![start];
    let mut seen = HashSet::from([start]);
    let mut mdlist = vec![0.0; n];
    update_inv_pow(points, metric, start, gamma, &mut mdlist)?;
    let mut guard = 0usize;
    while chosen.len() < k {
        let mut tot = 0.0;
        for (j, acc) in mdlist.iter().enumerate() {
            let w = weights.map(|ww| ww[j]).unwrap_or(1.0);
            let term = w * acc.powf(-gamma);
            if !term.is_finite() {
                return Err(LandfoldError::Msg(
                    "resample weight overflowed".into(),
                ));
            }
            tot += term;
        }
        if !(tot > 0.0 && tot.is_finite()) {
            return Err(LandfoldError::Msg("resample has no positive mass".into()));
        }
        let mut sel = rng.unit() * tot;
        let mut pick = 0usize;
        for j in 0..n {
            let w = weights.map(|ww| ww[j]).unwrap_or(1.0);
            sel -= w * mdlist[j].powf(-gamma);
            if sel < 0.0 {
                pick = j;
                break;
            }
            pick = j;
        }
        if unique && !seen.insert(pick) {
            guard += 1;
            if guard > n.saturating_mul(32).max(32) {
                return Err(LandfoldError::Msg(
                    "unique resample landmarks failed to fill k".into(),
                ));
            }
            continue;
        }
        chosen.push(pick);
        seen.insert(pick);
        update_inv_pow(points, metric, pick, gamma, &mut mdlist)?;
    }
    pack_landmarks(points, k, weights, &chosen)
}

fn landmarks_staged(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    k: usize,
    weights: Option<ArrayView1<f64>>,
    seed: usize,
    unique: bool,
    gamma: f64,
) -> Result<Landmarks> {
    let n = points.nrows();
    let cover = ((k as f64) * (n as f64)).sqrt().ceil() as usize;
    let cover = cover.clamp(k, n);
    let cover_lm = farthest_point(points, metric, cover, weights, seed)?;
    let masses = voronoi_weights(points, &cover_lm, metric, weights, gamma)?;
    let mut rng = Lcg::new(seed.wrapping_add(1));
    let mut cells: Vec<Vec<usize>> = vec![Vec::new(); cover];
    let d = points.ncols();
    let mut a = vec![0.0; d];
    let mut b = vec![0.0; d];
    for j in 0..n {
        for h in 0..d {
            b[h] = points[(j, h)];
        }
        let mut best_i = 0usize;
        let mut best_d = f64::INFINITY;
        for i in 0..cover {
            for (h, value) in a.iter_mut().enumerate().take(d) {
                *value = cover_lm.points[(i, h)];
            }
            let dist = metric.dist(&a, &b)?;
            if dist < best_d {
                best_d = dist;
                best_i = i;
            }
        }
        cells[best_i].push(j);
    }
    let mut tot = 0.0;
    for &w in masses.iter() {
        tot += w;
        if !tot.is_finite() {
            return Err(LandfoldError::Msg("staged mass overflowed".into()));
        }
    }
    if !(tot > 0.0) {
        return Err(LandfoldError::Msg("staged has no positive Voronoi mass".into()));
    }
    let mut chosen = Vec::with_capacity(k);
    let mut seen = HashSet::new();
    let mut guard = 0usize;
    while chosen.len() < k {
        let mut sel = rng.unit() * tot;
        let mut cell = 0usize;
        for i in 0..cover {
            sel -= masses[i];
            if sel < 0.0 {
                cell = i;
                break;
            }
            cell = i;
        }
        if cells[cell].is_empty() {
            guard += 1;
            if guard > n.saturating_mul(32).max(32) {
                return Err(LandfoldError::Msg("staged failed to fill k".into()));
            }
            continue;
        }
        let pick = cells[cell][rng.below(cells[cell].len())];
        if unique && !seen.insert(pick) {
            guard += 1;
            if guard > n.saturating_mul(32).max(32) {
                return Err(LandfoldError::Msg("unique staged failed to fill k".into()));
            }
            continue;
        }
        chosen.push(pick);
        seen.insert(pick);
    }
    pack_landmarks(points, k, weights, &chosen)
}

fn update_inv_pow(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    src: usize,
    gamma: f64,
    mdlist: &mut [f64],
) -> Result<()> {
    let d = points.ncols();
    let mut a = vec![0.0; d];
    let mut b = vec![0.0; d];
    for h in 0..d {
        a[h] = points[(src, h)];
    }
    for i in 0..points.nrows() {
        for h in 0..d {
            b[h] = points[(i, h)];
        }
        let dist = metric.dist(&a, &b)?;
        let term = if dist == 0.0 {
            1.0e200
        } else {
            dist.powf(-gamma)
        };
        if !term.is_finite() {
            return Err(LandfoldError::Msg(
                "resample inverse-distance overflowed".into(),
            ));
        }
        mdlist[i] += term;
        if !mdlist[i].is_finite() {
            return Err(LandfoldError::Msg("resample accumulator overflowed".into()));
        }
    }
    Ok(())
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: usize) -> Self {
        Self {
            state: (seed as u64).wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.state
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.unit() * n as f64).floor() as usize % n
    }
}

fn farthest_from_chosen(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    k: usize,
    weights: Option<ArrayView1<f64>>,
    chosen: &mut Vec<usize>,
    selected: &mut [bool],
    min_d: &mut [f64],
) -> Result<()> {
    let n = points.nrows();
    while chosen.len() < k {
        let mut best = None;
        let mut best_s = f64::NEG_INFINITY;
        for i in 0..n {
            if selected[i] {
                continue;
            }
            let w = weights.map(|ww| ww[i]).unwrap_or(1.0);
            let score = min_d[i] * w;
            if !score.is_finite() {
                return Err(LandfoldError::Msg(
                    "weighted landmark score overflowed".into(),
                ));
            }
            if score > best_s {
                best_s = score;
                best = Some(i);
            }
        }
        let best = best.ok_or_else(|| {
            LandfoldError::Msg("landmark scores contain no selectable finite value".into())
        })?;
        chosen.push(best);
        selected[best] = true;
        update_min_d(points, metric, best, &mut *min_d)?;
    }
    Ok(())
}

fn pack_landmarks(
    points: ArrayView2<f64>,
    k: usize,
    weights: Option<ArrayView1<f64>>,
    chosen: &[usize],
) -> Result<Landmarks> {
    let d = points.ncols();
    let mut lp = Array2::<f64>::zeros((k, d));
    let mut lw = Array1::<f64>::zeros(k);
    for (t, &i) in chosen.iter().enumerate() {
        for h in 0..d {
            lp[(t, h)] = points[(i, h)];
        }
        lw[t] = weights.map(|ww| ww[i]).unwrap_or(1.0);
    }
    Ok(Landmarks {
        index: chosen.to_vec(),
        points: lp,
        weights: lw,
    })
}

/// Voronoi mass of each landmark: sum of source weights of the nearest
/// points, then `w_i^wgamma` and renormalise (Ceriotti `-w` / `-wgamma`).
pub fn voronoi_weights(
    points: ArrayView2<f64>,
    landmarks: &Landmarks,
    metric: &dyn Metric,
    src_weights: Option<ArrayView1<f64>>,
    wgamma: f64,
) -> Result<Array1<f64>> {
    let n = points.nrows();
    let k = landmarks.index.len();
    if k == 0 {
        return Err(LandfoldError::Empty);
    }
    if !wgamma.is_finite() {
        return Err(LandfoldError::Msg("wgamma must be finite".into()));
    }
    if points.iter().any(|&value| !value.is_finite())
        || landmarks.points.iter().any(|&value| !value.is_finite())
    {
        return Err(LandfoldError::Msg(
            "landmark coordinates must be finite".into(),
        ));
    }
    if landmarks.index.iter().any(|&i| i >= n) {
        return Err(LandfoldError::Shape("landmark index out of bounds"));
    }
    let unique_indices: HashSet<usize> = landmarks.index.iter().copied().collect();
    if unique_indices.len() != k {
        return Err(LandfoldError::Msg(
            "landmark indices must be distinct".into(),
        ));
    }
    if src_weights.is_some_and(|w| w.len() != n) {
        return Err(LandfoldError::Shape("source weight length"));
    }
    if src_weights.is_some_and(|w| w.iter().any(|&value| !value.is_finite() || value < 0.0)) {
        return Err(LandfoldError::Msg(
            "source weights must be finite and nonnegative".into(),
        ));
    }
    let d = points.ncols();
    if landmarks.points.nrows() != k || landmarks.points.ncols() != d {
        return Err(LandfoldError::Shape("landmark dim"));
    }
    if landmarks.weights.len() != k {
        return Err(LandfoldError::Shape("landmark weight length"));
    }
    if landmarks
        .weights
        .iter()
        .any(|&value| !value.is_finite() || value < 0.0)
    {
        return Err(LandfoldError::Msg(
            "landmark weights must be finite and nonnegative".into(),
        ));
    }
    if let Some(expected) = metric.dim()
        && d != expected
    {
        return Err(LandfoldError::MetricSize {
            left: d,
            right: expected,
        });
    }
    let mut acc = vec![0.0; k];
    let mut a = vec![0.0; d];
    let mut b = vec![0.0; d];
    for j in 0..n {
        for h in 0..d {
            b[h] = points[(j, h)];
        }
        let mut best_i = 0usize;
        let mut best_d = f64::INFINITY;
        for i in 0..k {
            for (h, value) in a.iter_mut().enumerate().take(d) {
                *value = landmarks.points[(i, h)];
            }
            let dist = metric.dist(&a, &b)?;
            if dist < best_d {
                best_d = dist;
                best_i = i;
            }
        }
        acc[best_i] += src_weights.map(|w| w[j]).unwrap_or(1.0);
        if !acc[best_i].is_finite() {
            return Err(LandfoldError::Msg(
                "Voronoi weight accumulation overflowed".into(),
            ));
        }
    }
    let mut tw = 0.0;
    for w in &mut acc {
        *w = if *w > 0.0 { w.powf(wgamma) } else { 0.0 };
        if !w.is_finite() {
            return Err(LandfoldError::Msg(
                "Voronoi weight exponentiation overflowed".into(),
            ));
        }
        tw += *w;
        if !tw.is_finite() {
            return Err(LandfoldError::Msg(
                "Voronoi weight normalization overflowed".into(),
            ));
        }
    }
    if tw <= 0.0 {
        tw = 1.0;
    }
    let normalized = Array1::from_iter(acc.into_iter().map(|w| w / tw));
    if normalized.iter().any(|&value| !value.is_finite()) {
        return Err(LandfoldError::Msg(
            "Voronoi weights are non-finite after normalization".into(),
        ));
    }
    Ok(normalized)
}

impl Landmarks {
    pub fn assign_voronoi(
        &mut self,
        points: ArrayView2<f64>,
        metric: &dyn Metric,
        src_weights: Option<ArrayView1<f64>>,
        wgamma: f64,
    ) -> Result<()> {
        self.weights = voronoi_weights(points, self, metric, src_weights, wgamma)?;
        Ok(())
    }
}

fn update_min_d(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    src: usize,
    min_d: &mut [f64],
) -> Result<()> {
    let d = points.ncols();
    let mut a = vec![0.0; d];
    let mut b = vec![0.0; d];
    for h in 0..d {
        a[h] = points[(src, h)];
    }
    for i in 0..points.nrows() {
        for h in 0..d {
            b[h] = points[(i, h)];
        }
        let dist = metric.dist(&a, &b)?;
        if dist < min_d[i] {
            min_d[i] = dist;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::Euclid;
    use ndarray::array;

    #[test]
    fn picks_k_distinct() {
        let pts = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]];
        let lm = farthest_point(pts.view(), &Euclid, 3, None, 0).unwrap();
        assert_eq!(lm.index.len(), 3);
        let mut s = lm.index.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn picks_distinct_landmarks_when_all_scores_tie() {
        let pts = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let lm = farthest_point(pts.view(), &Euclid, 3, None, 0).unwrap();
        let mut indices = lm.index;
        indices.sort();
        indices.dedup();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn rejects_invalid_landmark_weights() {
        let pts = array![[0.0], [1.0], [2.0]];
        assert!(farthest_point(pts.view(), &Euclid, 2, Some(array![1.0, 1.0].view()), 0).is_err());
        assert!(
            farthest_point(
                pts.view(),
                &Euclid,
                2,
                Some(array![1.0, -1.0, 1.0].view()),
                0
            )
            .is_err()
        );
        assert!(
            farthest_point(
                pts.view(),
                &Euclid,
                2,
                Some(array![1.0, f64::NAN, 1.0].view()),
                0
            )
            .is_err()
        );
    }

    #[test]
    fn pinned_landmarks_validate_metric_and_weight_shapes() {
        let pts = array![[0.0], [1.0], [2.0]];
        assert!(
            farthest_point_ifirst(
                pts.view(),
                &crate::metric::Periodic::isotropic(2, 1.0).unwrap(),
                2,
                None,
                1,
            )
            .is_err()
        );
        assert!(
            farthest_point_ifirst(pts.view(), &Euclid, 2, Some(array![1.0].view()), 1,).is_err()
        );
    }

    #[test]
    fn voronoi_weights_reject_malformed_landmark_state_and_overflow() {
        let points = array![[0.0], [1.0]];
        let duplicate = Landmarks {
            index: vec![0, 0],
            points: array![[0.0], [1.0]],
            weights: array![1.0, 1.0],
        };
        assert!(voronoi_weights(points.view(), &duplicate, &Euclid, None, 1.0).is_err());

        let valid = Landmarks {
            index: vec![0],
            points: array![[0.0]],
            weights: array![1.0],
        };
        assert!(
            voronoi_weights(
                points.view(),
                &valid,
                &Euclid,
                Some(array![f64::MAX, f64::MAX].view()),
                1.0,
            )
            .is_err()
        );
    }

    #[test]
    fn farthest_point_rejects_overflowing_weighted_scores() {
        let points = array![[0.0], [f64::MAX]];
        assert!(
            farthest_point(
                points.view(),
                &crate::metric::L1,
                2,
                Some(array![1.0, f64::MAX].view()),
                0,
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_invalid_voronoi_inputs() {
        let pts = array![[0.0], [1.0]];
        let lm = farthest_point(pts.view(), &Euclid, 1, None, 0).unwrap();
        assert!(
            voronoi_weights(
                pts.view(),
                &lm,
                &Euclid,
                Some(array![1.0, f64::NAN].view()),
                1.0,
            )
            .is_err()
        );
        assert!(voronoi_weights(pts.view(), &lm, &Euclid, None, f64::NAN).is_err());
    }

    #[test]
    fn square_plus_centre_is_0_3_1() {
        let pts = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let lm = farthest_point(pts.view(), &Euclid, 3, None, 0).unwrap();
        assert_eq!(lm.index, vec![0, 3, 1]);
    }

    #[test]
    fn voronoi_mass_on_a_line() {
        let pts = array![[0.0], [0.1], [0.2], [10.0]];
        let lm = farthest_point(pts.view(), &Euclid, 2, None, 0).unwrap();
        // seed 0 then farthest = 10 -> indices 0, 3
        assert_eq!(lm.index, vec![0, 3]);
        let mut lm = lm;
        lm.assign_voronoi(pts.view(), &Euclid, None, 1.0).unwrap();
        assert!((lm.weights[0] - 0.75).abs() < 1e-12);
        assert!((lm.weights[1] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn stride_picks_evenly_spaced_rows() {
        let pts = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
        let lm = select_landmarks(
            pts.view(),
            &Euclid,
            3,
            None,
            0,
            None,
            true,
            LandmarkMode::Stride,
        )
        .unwrap();
        assert_eq!(lm.index, vec![0, 2, 4]);
    }

    #[test]
    fn random_unique_returns_k_distinct() {
        let pts = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let lm = select_landmarks(
            pts.view(),
            &Euclid,
            4,
            None,
            7,
            None,
            true,
            LandmarkMode::Random,
        )
        .unwrap();
        let mut s = lm.index.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 4);
    }

    #[test]
    fn mode_from_cli_rejects_unknown() {
        assert!(LandmarkMode::from_cli("voronoi", 1.0).is_err());
        assert!(LandmarkMode::from_cli("resample", 0.0).is_err());
        assert_eq!(
            LandmarkMode::from_cli("minmax", 1.0).unwrap(),
            LandmarkMode::MinMax
        );
    }
}
