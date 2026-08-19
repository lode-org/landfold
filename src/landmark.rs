//! Farthest-point landmarks.
//!
//! Gonzalez's 2-approximate k-center traversal (Gonzalez, *Theor. Comput.
//! Sci.* **38**, 293 (1985), <https://doi.org/10.1016/0304-3975(85)90224-5>),
//! optionally reweighted.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{LandfoldError, Result};
use crate::metric::Metric;

#[derive(Clone, Debug)]
pub struct Landmarks {
    pub index: Vec<usize>,
    pub points: Array2<f64>,
    pub weights: Array1<f64>,
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
        if w.len() != n {
            return Err(LandfoldError::Shape("landmark weight length"));
        }
        if w.iter().any(|&value| !value.is_finite() || value < 0.0) {
            return Err(LandfoldError::Msg(
                "landmark weights must be finite and nonnegative".into(),
            ));
        }
    }
    let d = points.ncols();
    let mut min_d = vec![f64::INFINITY; n];
    let mut selected = vec![false; n];
    let mut chosen = Vec::with_capacity(k);
    let start = seed % n;
    chosen.push(start);
    selected[start] = true;
    update_min_d(points, metric, start, &mut min_d);
    while chosen.len() < k {
        let mut best = None;
        let mut best_s = f64::NEG_INFINITY;
        for i in 0..n {
            if selected[i] {
                continue;
            }
            let w = weights.map(|ww| ww[i]).unwrap_or(1.0);
            let score = min_d[i] * w;
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
        update_min_d(points, metric, best, &mut min_d);
    }
    let mut lp = Array2::<f64>::zeros((k, d));
    let mut lw = Array1::<f64>::zeros(k);
    for (t, &i) in chosen.iter().enumerate() {
        for h in 0..d {
            lp[(t, h)] = points[(i, h)];
        }
        lw[t] = weights.map(|ww| ww[i]).unwrap_or(1.0);
    }
    Ok(Landmarks {
        index: chosen,
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
            let dist = metric.dist_unchecked(&a, &b);
            if dist < best_d {
                best_d = dist;
                best_i = i;
            }
        }
        acc[best_i] += src_weights.map(|w| w[j]).unwrap_or(1.0);
    }
    let mut tw = 0.0;
    for w in &mut acc {
        *w = if *w > 0.0 { w.powf(wgamma) } else { 0.0 };
        tw += *w;
    }
    if tw <= 0.0 {
        tw = 1.0;
    }
    Ok(Array1::from_iter(acc.into_iter().map(|w| w / tw)))
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

fn update_min_d(points: ArrayView2<f64>, metric: &dyn Metric, src: usize, min_d: &mut [f64]) {
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
        let dist = metric.dist_unchecked(&a, &b);
        if dist < min_d[i] {
            min_d[i] = dist;
        }
    }
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
}
