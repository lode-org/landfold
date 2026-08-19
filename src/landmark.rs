//! Farthest-point landmarks.
//!
//! Gonzalez's 2-approximate k-center traversal (Gonzalez, *Theor. Comput.
//! Sci.* **38**, 293 (1985), <https://doi.org/10.1016/0304-3975(85)90224-5>),
//! optionally reweighted.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{Result, LandfoldError};
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
    let d = points.ncols();
    let mut min_d = vec![f64::INFINITY; n];
    let mut chosen = Vec::with_capacity(k);
    let start = seed % n;
    chosen.push(start);
    update_min_d(points, metric, start, &mut min_d);
    while chosen.len() < k {
        let mut best = 0usize;
        let mut best_s = -1.0;
        for i in 0..n {
            let w = weights.map(|ww| ww[i]).unwrap_or(1.0);
            let score = min_d[i] * w;
            if score > best_s {
                best_s = score;
                best = i;
            }
        }
        chosen.push(best);
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
}
