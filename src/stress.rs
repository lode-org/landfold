//! Stress χ on transformed pairwise distances.
//!
//! χ = (1/Z) Σ_{i<j} w_ij [ (1-imix) (F_HD(D_ij) - F_LD(d_ij))^2
//!                         + imix     (D_ij - d_ij)^2 ]
//!
//! This is the objective of Ceriotti, Tribello and Parrinello,
//! *Proc. Natl. Acad. Sci. U.S.A.* **108**, 13023 (2011),
//! <https://doi.org/10.1073/pnas.1108486108>. The gradient always
//! divides by `d_ij` so it is the true derivative of χ.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::metric::{Euclid, Metric};
use crate::transfer::Transfer;

pub(crate) const OVERLAP: f64 = 1e-100;

#[derive(Clone, Debug)]
pub struct Stress {
    pub n: usize,
    pub d: usize,
    pub imix: f64,
    pub tfun_ld: Transfer,
    pub hd: Array2<f64>,
    pub fhd: Array2<f64>,
    pub weights: Option<Array2<f64>>,
}

#[derive(Clone, Debug)]
pub struct StressEval {
    pub value: f64,
    pub grad: Array1<f64>,
}

impl Stress {
    pub fn new(
        hd: Array2<f64>,
        fhd: Array2<f64>,
        tfun_ld: Transfer,
        imix: f64,
        weights: Option<Array1<f64>>,
        pair_weights: Option<Array2<f64>>,
    ) -> Self {
        let n = hd.nrows();
        let mut wmat = pair_weights;
        if let Some(ref w) = weights {
            let mut m = wmat.unwrap_or_else(|| Array2::ones((n, n)));
            for i in 0..n {
                for j in 0..n {
                    m[(i, j)] *= w[i] * w[j];
                }
            }
            wmat = Some(m);
        }
        Self {
            n,
            d: 0,
            imix,
            tfun_ld,
            hd,
            fhd,
            weights: wmat,
        }
    }

    /// Evaluate χ and ∇χ on packed low-D coordinates (`n * d`).
    pub fn eval(&self, coords: ArrayView1<f64>, d: usize) -> StressEval {
        let n = self.n;
        debug_assert_eq!(coords.len(), n * d);
        let mut pval = 0.0;
        let mut pgrad = vec![0.0; n * d];
        let mut tw = 0.0;
        let metric = Euclid;
        let imix = self.imix;
        let omix = 1.0 - imix;

        for i in 0..n {
            let xi = &coords.as_slice().unwrap()[i * d..(i + 1) * d];
            for j in 0..i {
                let xj = &coords.as_slice().unwrap()[j * d..(j + 1) * d];
                let ld = metric.dist_unchecked(xi, xj);
                let (fld, dfld) = self.tfun_ld.fdf(ld);
                let wij = self.weights.as_ref().map(|w| w[(i, j)]).unwrap_or(1.0);
                tw += wij;
                let df = self.fhd[(i, j)] - fld;
                let dd = self.hd[(i, j)] - ld;
                pval += (df * df * omix + imix * dd * dd) * wij;
                let dld = if ld < OVERLAP { OVERLAP } else { ld };
                let gij = (df * dfld * omix + imix * dd) / dld * wij;
                for h in 0..d {
                    let delta = xi[h] - xj[h];
                    pgrad[i * d + h] += gij * delta;
                    pgrad[j * d + h] -= gij * delta;
                }
            }
        }
        if tw <= 0.0 {
            tw = 1.0;
        }
        pval /= tw;
        for g in &mut pgrad {
            *g *= -2.0 / tw;
        }
        StressEval {
            value: pval,
            grad: Array1::from(pgrad),
        }
    }

    /// Pointwise χ for out-of-sample / grid projection of point `skip`.
    pub fn chi1(
        &self,
        x: ArrayView1<f64>,
        landmarks: ArrayView2<f64>,
        skip: usize,
        point_w: ArrayView1<f64>,
    ) -> (f64, Array1<f64>) {
        let d = x.len();
        let n = landmarks.nrows();
        let mut vv = 0.0;
        let mut vg = Array1::<f64>::zeros(d);
        let mut tw = 0.0;
        let omix = 1.0 - self.imix;
        for i in 0..n {
            if i == skip {
                continue;
            }
            let mut v1 = vec![0.0; d];
            let mut ld2 = 0.0;
            for h in 0..d {
                v1[h] = landmarks[(i, h)] - x[h];
                ld2 += v1[h] * v1[h];
            }
            let ld = ld2.sqrt();
            if ld <= 0.0 {
                continue;
            }
            let (lfd, ldfd) = self.tfun_ld.fdf(ld);
            let w = if point_w.len() == n { point_w[i] } else { 1.0 };
            let diff = self.fhd[(skip, i)] - lfd;
            let dd = self.hd[(skip, i)] - ld;
            vv += (diff * diff * omix + self.imix * dd * dd) * w;
            let scale = 2.0 * (diff * ldfd * omix + self.imix * dd) / ld * w;
            for h in 0..d {
                vg[h] += v1[h] * scale;
            }
            tw += w;
        }
        if tw <= 0.0 {
            tw = 1.0;
        }
        vv /= tw;
        vg.mapv_inplace(|g| g / tw);
        (vv, vg)
    }

    /// Stress of a new low-D point against all landmarks.
    /// `hd_row[i]` / `fhd_row[i]` are distances from the query to landmark `i`.
    pub fn chi_query(
        &self,
        x: ArrayView1<f64>,
        landmarks: ArrayView2<f64>,
        hd_row: ArrayView1<f64>,
        fhd_row: ArrayView1<f64>,
        point_w: ArrayView1<f64>,
    ) -> (f64, Array1<f64>) {
        let d = x.len();
        let n = landmarks.nrows();
        let mut vv = 0.0;
        let mut vg = Array1::<f64>::zeros(d);
        let mut tw = 0.0;
        let omix = 1.0 - self.imix;
        for i in 0..n {
            let mut v1 = vec![0.0; d];
            let mut ld2 = 0.0;
            for h in 0..d {
                v1[h] = landmarks[(i, h)] - x[h];
                ld2 += v1[h] * v1[h];
            }
            let ld = ld2.sqrt();
            if ld <= 0.0 {
                continue;
            }
            let (lfd, ldfd) = self.tfun_ld.fdf(ld);
            let w = if point_w.len() == n { point_w[i] } else { 1.0 };
            let diff = fhd_row[i] - lfd;
            let dd = hd_row[i] - ld;
            vv += (diff * diff * omix + self.imix * dd * dd) * w;
            let scale = 2.0 * (diff * ldfd * omix + self.imix * dd) / ld * w;
            for h in 0..d {
                vg[h] += v1[h] * scale;
            }
            tw += w;
        }
        if tw <= 0.0 {
            tw = 1.0;
        }
        vv /= tw;
        vg.mapv_inplace(|g| g / tw);
        (vv, vg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pairwise::pairwise_euclid;
    use crate::transfer::Transfer;
    use approx::assert_relative_eq;
    use ndarray::{array, Array};

    #[test]
    fn identity_stress_zero_on_isometry() {
        let pts = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let hd = pairwise_euclid(pts.view()).unwrap();
        let fhd = hd.clone();
        let s = Stress::new(hd, fhd, Transfer::identity(), 0.0, None, None);
        let coords = Array::from_iter(pts.iter().copied());
        let ev = s.eval(coords.view(), 2);
        assert_relative_eq!(ev.value, 0.0, epsilon = 1e-14);
        for g in ev.grad.iter() {
            assert_relative_eq!(*g, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let hd = array![
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
            [2.0, 1.5, 0.0]
        ];
        let t = Transfer::xsigmoid(1.0, 4.0, 3.0).unwrap();
        let mut fhd = hd.clone();
        crate::pairwise::apply_transfer(&mut fhd, &t);
        let s = Stress::new(hd, fhd, t, 0.1, None, None);
        let coords = array![0.0, 0.0, 0.8, 0.1, -0.2, 0.7];
        let ev = s.eval(coords.view(), 2);
        let h = 1e-7;
        for k in 0..6 {
            let mut up = coords.clone();
            up[k] += h;
            let mut dn = coords.clone();
            dn[k] -= h;
            let fd = (s.eval(up.view(), 2).value - s.eval(dn.view(), 2).value) / (2.0 * h);
            assert_relative_eq!(ev.grad[k], fd, epsilon = 1e-6);
        }
    }
}
