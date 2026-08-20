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

pub(crate) fn validate_weights(
    weights: Option<ArrayView1<'_, f64>>,
    n: usize,
) -> crate::error::Result<()> {
    if let Some(w) = weights {
        if w.len() != n {
            return Err(crate::error::LandfoldError::Shape("stress weight length"));
        }
        if w.iter().any(|&value| !value.is_finite() || value < 0.0) {
            return Err(crate::error::LandfoldError::Msg(
                "stress weights must be finite and nonnegative".into(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_distance_matrix(
    matrix: &Array2<f64>,
    label: &str,
) -> crate::error::Result<()> {
    let n = matrix.nrows();
    for i in 0..n {
        if matrix[(i, i)] != 0.0 {
            return Err(crate::error::LandfoldError::Msg(format!(
                "{label} distance matrix diagonal must be zero"
            )));
        }
        for j in 0..i {
            let left = matrix[(i, j)];
            let right = matrix[(j, i)];
            let scale = left.abs().max(right.abs()).max(1.0);
            if (left - right).abs() > 64.0 * f64::EPSILON * scale {
                return Err(crate::error::LandfoldError::Msg(format!(
                    "{label} distance matrix must be symmetric"
                )));
            }
        }
    }
    Ok(())
}

fn validate_pair_weight_mass(weights: &Array2<f64>) -> crate::error::Result<()> {
    let mut total = 0.0;
    for i in 0..weights.nrows() {
        for j in 0..i {
            total += weights[(i, j)];
            if !total.is_finite() {
                return Err(crate::error::LandfoldError::Msg(
                    "stress pair-weight mass overflowed".into(),
                ));
            }
        }
    }
    Ok(())
}

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

const OPTIMIZER_PENALTY: f64 = f64::MAX / 4.0;

struct PairData<'a> {
    n: usize,
    d: usize,
    coords: &'a [f64],
    hd: &'a [f64],
    fhd: &'a [f64],
    weights: Option<&'a [f64]>,
    omix: f64,
}

struct PairAccum<'a> {
    value: &'a mut f64,
    weight: &'a mut f64,
    grad: &'a mut [f64],
}

pub(crate) fn validate_imix(imix: f64) -> crate::error::Result<()> {
    if !imix.is_finite() || !(0.0..=1.0).contains(&imix) {
        return Err(crate::error::LandfoldError::Msg(
            "imix must be finite and in 0..=1".into(),
        ));
    }
    Ok(())
}

impl Stress {
    fn validate_state(&self) -> crate::error::Result<()> {
        let n = self.n;
        if self.hd.nrows() != n
            || self.hd.ncols() != n
            || self.fhd.nrows() != n
            || self.fhd.ncols() != n
            || self
                .weights
                .as_ref()
                .is_some_and(|weights| weights.nrows() != n || weights.ncols() != n)
        {
            return Err(crate::error::LandfoldError::Shape(
                "stress state matrices must be square and match n",
            ));
        }
        if self.hd.as_slice().is_none()
            || self.fhd.as_slice().is_none()
            || self
                .weights
                .as_ref()
                .is_some_and(|weights| weights.as_slice().is_none())
        {
            return Err(crate::error::LandfoldError::Shape(
                "stress state matrices must be contiguous",
            ));
        }
        if self
            .hd
            .iter()
            .chain(self.fhd.iter())
            .any(|&value| !value.is_finite() || value < 0.0)
            || self.weights.as_ref().is_some_and(|weights| {
                weights
                    .iter()
                    .any(|&value| !value.is_finite() || value < 0.0)
            })
        {
            return Err(crate::error::LandfoldError::Msg(
                "stress distances and weights must be finite and nonnegative".into(),
            ));
        }
        if let Some(weights) = self.weights.as_ref() {
            validate_pair_weight_mass(weights)?;
        }
        validate_distance_matrix(&self.hd, "high-D")?;
        validate_distance_matrix(&self.fhd, "transformed high-D")?;
        validate_imix(self.imix)
    }

    pub fn try_new(
        hd: Array2<f64>,
        fhd: Array2<f64>,
        tfun_ld: Transfer,
        imix: f64,
        weights: Option<Array1<f64>>,
        pair_weights: Option<Array2<f64>>,
    ) -> crate::error::Result<Self> {
        validate_imix(imix)?;
        let n = hd.nrows();
        if hd.ncols() != n || fhd.raw_dim() != hd.raw_dim() {
            return Err(crate::error::LandfoldError::Shape(
                "stress distance matrices must be square and matching",
            ));
        }
        if hd
            .iter()
            .chain(fhd.iter())
            .any(|&value| !value.is_finite() || value < 0.0)
        {
            return Err(crate::error::LandfoldError::Msg(
                "stress distance matrices must be finite and nonnegative".into(),
            ));
        }
        validate_distance_matrix(&hd, "high-D")?;
        validate_distance_matrix(&fhd, "transformed high-D")?;
        validate_weights(weights.as_ref().map(|w| w.view()), n)?;
        if pair_weights
            .as_ref()
            .is_some_and(|w| w.raw_dim() != hd.raw_dim())
        {
            return Err(crate::error::LandfoldError::Shape(
                "stress pair weight shape",
            ));
        }
        let stress = Self::from_parts(
            hd,
            fhd,
            tfun_ld,
            imix,
            weights,
            pair_weights,
        );
        if stress.weights.as_ref().is_some_and(|weights| {
            weights
                .iter()
                .any(|&value| !value.is_finite() || value < 0.0)
        }) {
            return Err(crate::error::LandfoldError::Msg(
                "stress assembled pair weights must be finite and nonnegative".into(),
            ));
        }
        if let Some(weights) = stress.weights.as_ref() {
            validate_pair_weight_mass(weights)?;
        }
        Ok(stress)
    }

    pub fn new(
        hd: Array2<f64>,
        fhd: Array2<f64>,
        tfun_ld: Transfer,
        imix: f64,
        weights: Option<Array1<f64>>,
        pair_weights: Option<Array2<f64>>,
    ) -> crate::error::Result<Self> {
        Self::try_new(hd, fhd, tfun_ld, imix, weights, pair_weights)
    }

    /// Pair weights `F(D)(1-F(D))`. Only mid-scale pairs contribute to χ.
    pub fn midscale_pair_weights(fhd: ArrayView2<f64>) -> crate::error::Result<Array2<f64>> {
        let n = fhd.nrows();
        if fhd.ncols() != n {
            return Err(crate::error::LandfoldError::Shape(
                "midscale weights need a square F(D) matrix",
            ));
        }
        let mut w = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..i {
                let f = fhd[(i, j)];
                if !f.is_finite() || f < 0.0 {
                    return Err(crate::error::LandfoldError::Msg(
                        "midscale F(D) must be finite and nonnegative".into(),
                    ));
                }
                let wij = f * (1.0 - f).max(0.0);
                if !wij.is_finite() {
                    return Err(crate::error::LandfoldError::Msg(
                        "midscale pair weight overflowed".into(),
                    ));
                }
                w[(i, j)] = wij;
                w[(j, i)] = wij;
            }
        }
        validate_pair_weight_mass(&w)?;
        Ok(w)
    }

    fn from_parts(
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
        let Some(expected_len) = self.n.checked_mul(d) else {
            return StressEval {
                value: OPTIMIZER_PENALTY,
                grad: Array1::zeros(coords.len()),
            };
        };
        if d == 0
            || coords.len() != expected_len
            || coords.as_slice().is_none()
            || self.hd.as_slice().is_none()
            || self.fhd.as_slice().is_none()
            || self
                .weights
                .as_ref()
                .is_some_and(|weights| weights.as_slice().is_none())
        {
            return StressEval {
                value: OPTIMIZER_PENALTY,
                grad: Array1::zeros(coords.len()),
            };
        }
        #[cfg(feature = "parallel")]
        {
            self.eval_parallel(coords, d)
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.eval_serial(coords, d)
        }
    }

    /// Checked evaluation for external callers with fallible input handling.
    pub fn try_eval(&self, coords: ArrayView1<f64>, d: usize) -> crate::error::Result<StressEval> {
        self.validate_state()?;
        if d == 0 {
            return Err(crate::error::LandfoldError::LowDim {
                low: 0,
                high: self.n,
            });
        }
        let coordinate_len = self.n.checked_mul(d).ok_or_else(|| {
            crate::error::LandfoldError::Msg(
                "stress coordinate dimension product overflowed".into(),
            )
        })?;
        if coords.len() != coordinate_len {
            return Err(crate::error::LandfoldError::Shape(
                "stress coordinates must have length n * d",
            ));
        }
        if coords.as_slice().is_none() {
            return Err(crate::error::LandfoldError::Shape(
                "stress coordinates must be contiguous",
            ));
        }
        if coords.iter().any(|&value| !value.is_finite()) {
            return Err(crate::error::LandfoldError::Msg(
                "stress coordinates must be finite".into(),
            ));
        }
        let packed = coords.as_slice().expect("contiguous coordinates");
        for i in 0..self.n {
            let xi = &packed[i * d..(i + 1) * d];
            for j in 0..i {
                let xj = &packed[j * d..(j + 1) * d];
                let ld = Euclid.dist(xi, xj)?;
                self.tfun_ld.try_fdf(ld)?;
            }
        }
        let evaluation = self.eval(coords, d);
        if !evaluation.value.is_finite()
            || evaluation
                .grad
                .iter()
                .any(|&component| !component.is_finite())
        {
            return Err(crate::error::LandfoldError::Msg(
                "stress evaluation is non-finite".into(),
            ));
        }
        Ok(evaluation)
    }

    /// Same checks as [`Self::try_eval`], serial kernel only.
    ///
    /// Used when a HiGHS OpenMP pool must not meet Rayon.
    #[allow(dead_code)]
    pub(crate) fn try_eval_serial(
        &self,
        coords: ArrayView1<f64>,
        d: usize,
    ) -> crate::error::Result<StressEval> {
        self.validate_state()?;
        if d == 0 {
            return Err(crate::error::LandfoldError::LowDim {
                low: 0,
                high: self.n,
            });
        }
        let coordinate_len = self.n.checked_mul(d).ok_or_else(|| {
            crate::error::LandfoldError::Msg(
                "stress coordinate dimension product overflowed".into(),
            )
        })?;
        if coords.len() != coordinate_len {
            return Err(crate::error::LandfoldError::Shape(
                "stress coordinates must have length n * d",
            ));
        }
        if coords.as_slice().is_none() {
            return Err(crate::error::LandfoldError::Shape(
                "stress coordinates must be contiguous",
            ));
        }
        if coords.iter().any(|&value| !value.is_finite()) {
            return Err(crate::error::LandfoldError::Msg(
                "stress coordinates must be finite".into(),
            ));
        }
        let packed = coords.as_slice().expect("contiguous coordinates");
        for i in 0..self.n {
            let xi = &packed[i * d..(i + 1) * d];
            for j in 0..i {
                let xj = &packed[j * d..(j + 1) * d];
                let ld = Euclid.dist(xi, xj)?;
                self.tfun_ld.try_fdf(ld)?;
            }
        }
        let evaluation = self.eval_serial(coords, d);
        if !evaluation.value.is_finite()
            || evaluation
                .grad
                .iter()
                .any(|&component| !component.is_finite())
        {
            return Err(crate::error::LandfoldError::Msg(
                "stress evaluation is non-finite".into(),
            ));
        }
        Ok(evaluation)
    }

    /// Infallible objective boundary for optimizer traits that cannot return errors.
    pub(crate) fn eval_for_optimizer(&self, coords: ArrayView1<f64>, d: usize) -> StressEval {
        match self.try_eval(coords, d) {
            Ok(evaluation) => evaluation,
            Err(_) => StressEval {
                value: OPTIMIZER_PENALTY,
                grad: Array1::zeros(coords.len()),
            },
        }
    }

    fn pair_kernel(&self, i: usize, data: &PairData<'_>, acc: &mut PairAccum<'_>) {
        let xi = &data.coords[i * data.d..(i + 1) * data.d];
        for j in 0..i {
            let xj = &data.coords[j * data.d..(j + 1) * data.d];
            let ld = Euclid.dist_unchecked(xi, xj);
            let (fld, dfld) = self.tfun_ld.fdf(ld);
            let wij = data.weights.map(|w| w[i * data.n + j]).unwrap_or(1.0);
            *acc.weight += wij;
            let df = data.fhd[i * data.n + j] - fld;
            let dd = data.hd[i * data.n + j] - ld;
            *acc.value += (df * df * data.omix + self.imix * dd * dd) * wij;
            let dld = if ld < OVERLAP { OVERLAP } else { ld };
            let gij = (df * dfld * data.omix + self.imix * dd) / dld * wij;
            if data.d == 2 {
                let dx = xi[0] - xj[0];
                let dy = xi[1] - xj[1];
                acc.grad[i * 2] += gij * dx;
                acc.grad[i * 2 + 1] += gij * dy;
                acc.grad[j * 2] -= gij * dx;
                acc.grad[j * 2 + 1] -= gij * dy;
            } else {
                for h in 0..data.d {
                    let delta = xi[h] - xj[h];
                    acc.grad[i * data.d + h] += gij * delta;
                    acc.grad[j * data.d + h] -= gij * delta;
                }
            }
        }
    }

    fn finish(mut pval: f64, mut tw: f64, mut pgrad: Vec<f64>) -> StressEval {
        if tw <= 0.0 {
            tw = 1.0;
        }
        pval /= tw;
        let scale = -2.0 / tw;
        for g in &mut pgrad {
            *g *= scale;
        }
        StressEval {
            value: pval,
            grad: Array1::from(pgrad),
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn eval_serial(&self, coords: ArrayView1<f64>, d: usize) -> StressEval {
        let n = self.n;
        debug_assert_eq!(coords.len(), n * d);
        let coords = coords.as_slice().expect("packed coords contiguous");
        let hd = self.hd.as_slice().expect("hd contiguous");
        let fhd = self.fhd.as_slice().expect("fhd contiguous");
        let weights = self
            .weights
            .as_ref()
            .map(|w| w.as_slice().expect("w contiguous"));
        let omix = 1.0 - self.imix;
        let mut pval = 0.0;
        let mut tw = 0.0;
        let mut pgrad = vec![0.0; n * d];
        let data = PairData {
            n,
            d,
            coords,
            hd,
            fhd,
            weights,
            omix,
        };
        let mut acc = PairAccum {
            value: &mut pval,
            weight: &mut tw,
            grad: &mut pgrad,
        };
        for i in 0..n {
            self.pair_kernel(i, &data, &mut acc);
        }
        Self::finish(pval, tw, pgrad)
    }

    #[cfg(feature = "parallel")]
    fn eval_parallel(&self, coords: ArrayView1<f64>, d: usize) -> StressEval {
        use rayon::prelude::*;
        let n = self.n;
        debug_assert_eq!(coords.len(), n * d);
        let coords = coords.as_slice().expect("packed coords contiguous");
        let hd = self.hd.as_slice().expect("hd contiguous");
        let fhd = self.fhd.as_slice().expect("fhd contiguous");
        let weights = self
            .weights
            .as_ref()
            .map(|w| w.as_slice().expect("w contiguous"));
        let omix = 1.0 - self.imix;
        let (pval, tw, pgrad) = (0..n)
            .into_par_iter()
            .fold(
                || (0.0, 0.0, vec![0.0; n * d]),
                |mut acc, i| {
                    let data = PairData {
                        n,
                        d,
                        coords,
                        hd,
                        fhd,
                        weights,
                        omix,
                    };
                    let mut pair_acc = PairAccum {
                        value: &mut acc.0,
                        weight: &mut acc.1,
                        grad: &mut acc.2,
                    };
                    self.pair_kernel(i, &data, &mut pair_acc);
                    acc
                },
            )
            .reduce(
                || (0.0, 0.0, vec![0.0; n * d]),
                |mut a, b| {
                    a.0 += b.0;
                    a.1 += b.1;
                    for (x, y) in a.2.iter_mut().zip(b.2) {
                        *x += y;
                    }
                    a
                },
            );
        Self::finish(pval, tw, pgrad)
    }

    /// Pointwise χ for out-of-sample / grid projection of point `skip`.
    pub fn chi1(
        &self,
        x: ArrayView1<f64>,
        landmarks: ArrayView2<f64>,
        skip: usize,
        point_w: ArrayView1<f64>,
    ) -> (f64, Array1<f64>) {
        match self.chi1_checked(x, landmarks, skip, point_w) {
            Ok(result) => result,
            Err(_) => (OPTIMIZER_PENALTY, Array1::zeros(x.len())),
        }
    }

    /// Checked pointwise χ for out-of-sample / grid projection of `skip`.
    pub fn chi1_checked(
        &self,
        x: ArrayView1<f64>,
        landmarks: ArrayView2<f64>,
        skip: usize,
        point_w: ArrayView1<f64>,
    ) -> crate::error::Result<(f64, Array1<f64>)> {
        let d = x.len();
        let n = self.n;
        if skip >= n || landmarks.nrows() != n || landmarks.ncols() != d || d == 0 {
            return Err(crate::error::LandfoldError::Shape(
                "chi1 arrays have incompatible shapes",
            ));
        }
        if !point_w.is_empty() && point_w.len() != n {
            return Err(crate::error::LandfoldError::Shape("chi1 weight length"));
        }
        self.validate_state()?;
        if x.iter()
            .chain(landmarks.iter())
            .any(|&value| !value.is_finite())
            || point_w
                .iter()
                .any(|&value| !value.is_finite() || value < 0.0)
        {
            return Err(crate::error::LandfoldError::Msg(
                "chi1 coordinates and weights must be finite and nonnegative where applicable"
                    .into(),
            ));
        }
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
            let (lfd, ldfd) = self.tfun_ld.try_fdf(ld)?;
            let w = if point_w.len() == n { point_w[i] } else { 1.0 };
            let diff = self.fhd[(skip, i)] - lfd;
            let dd = self.hd[(skip, i)] - ld;
            vv += (diff * diff * omix + self.imix * dd * dd) * w;
            tw += w;
            if ld <= 0.0 {
                continue;
            }
            let scale = 2.0 * (diff * ldfd * omix + self.imix * dd) / ld * w;
            for h in 0..d {
                vg[h] += v1[h] * scale;
            }
        }
        if tw <= 0.0 {
            tw = 1.0;
        }
        vv /= tw;
        vg.mapv_inplace(|g| g / tw);
        if !vv.is_finite() || vg.iter().any(|&value| !value.is_finite()) {
            return Err(crate::error::LandfoldError::Msg(
                "chi1 evaluation is non-finite".into(),
            ));
        }
        Ok((vv, vg))
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
        query_chi(
            x,
            landmarks,
            hd_row,
            fhd_row,
            &self.tfun_ld,
            self.imix,
            point_w,
        )
    }
}

/// One-point χ of a low-D query against landmarks (JCTC 2013, Eq. 4).
///
/// Matches `compute_chi1` in the C++ reference (no skip index: the query is
/// not one of the landmarks).
pub fn query_chi(
    x: ArrayView1<f64>,
    landmarks: ArrayView2<f64>,
    hd_row: ArrayView1<f64>,
    fhd_row: ArrayView1<f64>,
    tfun_ld: &Transfer,
    imix: f64,
    point_w: ArrayView1<f64>,
) -> (f64, Array1<f64>) {
    match query_chi_checked(x, landmarks, hd_row, fhd_row, tfun_ld, imix, point_w) {
        Ok(result) => result,
        Err(_) => (OPTIMIZER_PENALTY, Array1::zeros(x.len())),
    }
}

/// Checked one-point χ of a low-D query against landmarks.
pub fn query_chi_checked(
    x: ArrayView1<f64>,
    landmarks: ArrayView2<f64>,
    hd_row: ArrayView1<f64>,
    fhd_row: ArrayView1<f64>,
    tfun_ld: &Transfer,
    imix: f64,
    point_w: ArrayView1<f64>,
) -> crate::error::Result<(f64, Array1<f64>)> {
    let d = x.len();
    let n = landmarks.nrows();
    if d == 0 || landmarks.ncols() != d || hd_row.len() != n || fhd_row.len() != n {
        return Err(crate::error::LandfoldError::Shape(
            "query stress arrays have incompatible shapes",
        ));
    }
    if !point_w.is_empty() && point_w.len() != n {
        return Err(crate::error::LandfoldError::Shape(
            "query stress weight length",
        ));
    }
    validate_imix(imix)?;
    if x.iter()
        .chain(landmarks.iter())
        .any(|&value| !value.is_finite())
        || hd_row
            .iter()
            .chain(fhd_row.iter())
            .any(|&value| !value.is_finite() || value < 0.0)
        || point_w
            .iter()
            .any(|&value| !value.is_finite() || value < 0.0)
    {
        return Err(crate::error::LandfoldError::Msg(
            "query stress inputs must be finite and nonnegative where applicable".into(),
        ));
    }
    let mut vv = 0.0;
    let mut vg = Array1::<f64>::zeros(d);
    let mut tw = 0.0;
    let omix = 1.0 - imix;
    for i in 0..n {
        let mut ld2 = 0.0;
        for h in 0..d {
            let delta = landmarks[(i, h)] - x[h];
            ld2 += delta * delta;
        }
        let ld = ld2.sqrt();
        let (lfd, ldfd) = tfun_ld.try_fdf(ld)?;
        let w = if point_w.len() == n { point_w[i] } else { 1.0 };
        let diff = fhd_row[i] - lfd;
        let dd = hd_row[i] - ld;
        vv += (diff * diff * omix + imix * dd * dd) * w;
        tw += w;
        if ld <= 0.0 {
            continue;
        }
        let scale = 2.0 * (diff * ldfd * omix + imix * dd) / ld * w;
        for h in 0..d {
            vg[h] += (landmarks[(i, h)] - x[h]) * scale;
        }
    }
    if tw <= 0.0 {
        tw = 1.0;
    }
    vv /= tw;
    vg.mapv_inplace(|g| g / tw);
    if !vv.is_finite() || vg.iter().any(|&value| !value.is_finite()) {
        return Err(crate::error::LandfoldError::Msg(
            "query stress evaluation is non-finite".into(),
        ));
    }
    Ok((vv, vg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pairwise::pairwise_euclid;
    use crate::transfer::Transfer;
    use approx::assert_relative_eq;
    use ndarray::{Array, array};

    #[test]
    fn identity_stress_zero_on_isometry() {
        let pts = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let hd = pairwise_euclid(pts.view()).unwrap();
        let fhd = hd.clone();
        let s = Stress::new(hd, fhd, Transfer::identity(), 0.0, None, None).unwrap();
        let coords = Array::from_iter(pts.iter().copied());
        let ev = s.eval(coords.view(), 2);
        assert_relative_eq!(ev.value, 0.0, epsilon = 1e-14);
        for g in ev.grad.iter() {
            assert_relative_eq!(*g, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn midscale_weights_peak_at_half() {
        let f = array![[0.0, 0.5, 1.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let w = Stress::midscale_pair_weights(f.view()).unwrap();
        assert!((w[(0, 1)] - 0.25).abs() < 1e-15);
        assert!(w[(0, 2)].abs() < 1e-15);
        assert_eq!(w[(0, 0)], 0.0);
    }

    #[test]
    fn try_new_rejects_incompatible_shapes() {
        let square = Array2::zeros((2, 2));
        assert!(
            Stress::try_new(
                square.clone(),
                Array2::zeros((2, 3)),
                Transfer::identity(),
                0.0,
                None,
                None,
            )
            .is_err()
        );
        assert!(
            Stress::try_new(
                square.clone(),
                square.clone(),
                Transfer::identity(),
                0.0,
                Some(array![1.0]),
                None,
            )
            .is_err()
        );
        assert!(
            Stress::try_new(
                square.clone(),
                square,
                Transfer::identity(),
                0.0,
                None,
                Some(Array2::zeros((3, 3))),
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_invalid_assembled_pair_weights() {
        let hd = Array2::zeros((2, 2));
        assert!(Stress::new(
            hd.clone(),
            hd.clone(),
            Transfer::identity(),
            0.0,
            Some(array![f64::MAX, f64::MAX]),
            None,
        )
        .is_err());
        assert!(Stress::new(
            hd.clone(),
            hd,
            Transfer::identity(),
            0.0,
            None,
            Some(Array2::from_elem((2, 2), f64::NAN)),
        )
        .is_err());
    }

    #[test]
    fn rejects_overflowing_pair_weight_mass() {
        let hd = Array2::zeros((3, 3));
        let pair_weights = Array2::from_elem((3, 3), f64::MAX);
        assert!(Stress::new(
            hd.clone(),
            hd,
            Transfer::identity(),
            0.0,
            None,
            Some(pair_weights),
        )
        .is_err());
    }

    #[test]
    fn eval_penalizes_non_contiguous_public_distance_state() {
        use ndarray::ShapeBuilder;

        let hd = Array2::from_shape_vec((2, 2).strides((1, 2)), vec![0.0, 2.0, 2.0, 0.0])
            .unwrap();
        let stress = Stress::new(
            hd.clone(),
            hd,
            Transfer::identity(),
            0.0,
            None,
            None,
        )
        .unwrap();
        let evaluation = stress.eval(array![0.0, 0.0, 1.0, 0.0].view(), 2);
        assert_eq!(evaluation.value, OPTIMIZER_PENALTY);
        assert_eq!(evaluation.grad.len(), 4);
        assert!(stress
            .try_eval(array![0.0, 0.0, 1.0, 0.0].view(), 2)
            .is_err());
    }

    #[test]
    fn try_new_rejects_invalid_imix() {
        let square = Array2::zeros((2, 2));
        for imix in [-0.1, 1.1, f64::NAN] {
            assert!(
                Stress::try_new(
                    square.clone(),
                    square.clone(),
                    Transfer::identity(),
                    imix,
                    None,
                    None,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn try_new_rejects_invalid_weights() {
        let square = Array2::zeros((2, 2));
        for weights in [array![1.0, -1.0], array![1.0, f64::NAN]] {
            assert!(
                Stress::try_new(
                    square.clone(),
                    square.clone(),
                    Transfer::identity(),
                    0.0,
                    Some(weights),
                    None,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn try_new_rejects_nonfinite_distances() {
        let hd = array![[0.0, f64::NAN], [f64::NAN, 0.0]];
        assert!(Stress::try_new(hd.clone(), hd, Transfer::identity(), 0.0, None, None).is_err());
    }

    #[test]
    fn new_rejects_invalid_state() {
        let hd = array![[0.0, -1.0], [-1.0, 0.0]];
        assert!(Stress::new(hd.clone(), hd, Transfer::identity(), 0.0, None, None).is_err());
    }

    #[test]
    fn rejects_nonsymmetric_or_nonzero_diagonal_distances() {
        let asymmetric = array![[0.0, 1.0], [2.0, 0.0]];
        assert!(Stress::new(
            asymmetric.clone(),
            asymmetric,
            Transfer::identity(),
            0.0,
            None,
            None,
        )
        .is_err());
        let diagonal = array![[1.0, 0.0], [0.0, 0.0]];
        assert!(Stress::new(
            diagonal.clone(),
            diagonal,
            Transfer::identity(),
            0.0,
            None,
            None,
        )
        .is_err());
    }

    #[test]
    fn coincident_query_keeps_its_objective_value() {
        let stress = Stress::new(
            array![[0.0, 1.0], [1.0, 0.0]],
            array![[0.0, 1.0], [1.0, 0.0]],
            Transfer::identity(),
            0.0,
            None,
            None,
        )
        .unwrap();
        let x = array![0.0, 0.0];
        let landmarks = array![[0.0, 0.0]];
        let (value, grad) = query_chi(
            x.view(),
            landmarks.view(),
            array![1.0].view(),
            array![1.0].view(),
            &stress.tfun_ld,
            stress.imix,
            array![1.0].view(),
        );
        assert_relative_eq!(value, 1.0, epsilon = 1e-14);
        assert_relative_eq!(grad[0], 0.0, epsilon = 1e-14);
        assert_relative_eq!(grad[1], 0.0, epsilon = 1e-14);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_matches_serial() {
        let hd = array![[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]];
        let t = Transfer::xsigmoid(1.0, 4.0, 3.0).unwrap();
        let mut fhd = hd.clone();
        crate::pairwise::apply_transfer(&mut fhd, &t).unwrap();
        let s = Stress::new(hd, fhd, t, 0.1, None, None).unwrap();
        let coords = array![0.0, 0.0, 0.8, 0.1, -0.2, 0.7];
        let a = s.eval_serial(coords.view(), 2);
        let b = s.eval(coords.view(), 2);
        assert_relative_eq!(a.value, b.value, epsilon = 1e-14);
        for k in 0..6 {
            assert_relative_eq!(a.grad[k], b.grad[k], epsilon = 1e-14);
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let hd = array![[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]];
        let t = Transfer::xsigmoid(1.0, 4.0, 3.0).unwrap();
        let mut fhd = hd.clone();
        crate::pairwise::apply_transfer(&mut fhd, &t).unwrap();
        let s = Stress::new(hd, fhd, t, 0.1, None, None).unwrap();
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

    #[test]
    fn checked_eval_rejects_malformed_coordinates() {
        let hd = array![[0.0, 1.0], [1.0, 0.0]];
        let stress = Stress::new(hd.clone(), hd, Transfer::identity(), 0.0, None, None).unwrap();
        assert!(stress.try_eval(array![0.0].view(), 1).is_err());
        assert!(stress.try_eval(array![0.0, f64::NAN].view(), 1).is_err());
        assert!(stress.try_eval(array![0.0, 0.0].view(), 0).is_err());
        assert!(stress.try_eval(array![].view(), usize::MAX).is_err());
        assert!(stress.try_eval(array![0.0, 0.0].view(), 1).is_ok());
        assert_eq!(stress.eval(array![0.0].view(), 1).value, OPTIMIZER_PENALTY);
        assert_eq!(
            stress.eval(array![].view(), usize::MAX).value,
            OPTIMIZER_PENALTY
        );
    }

    #[test]
    fn checked_eval_rejects_mutated_stress_state() {
        let hd = array![[0.0, 1.0], [1.0, 0.0]];
        let mut stress =
            Stress::new(hd.clone(), hd, Transfer::identity(), 0.0, None, None).unwrap();
        stress.fhd = Array2::zeros((1, 1));
        assert!(stress.try_eval(array![0.0, 0.0].view(), 1).is_err());

        let hd = array![[0.0, 1.0], [1.0, 0.0]];
        let mut stress =
            Stress::new(hd.clone(), hd, Transfer::identity(), 0.0, None, None).unwrap();
        stress.n = 3;
        assert!(stress.try_eval(array![0.0, 0.0].view(), 1).is_err());
    }

    #[test]
    fn checked_eval_rejects_transfer_overflow() {
        let hd = array![[0.0, 1.0], [1.0, 0.0]];
        let mut stress =
            Stress::new(hd.clone(), hd, Transfer::identity(), 0.0, None, None).unwrap();
        stress.tfun_ld = Transfer::xsigmoid(1.0, 8.0, 1.0).unwrap();
        assert!(stress.try_eval(array![0.0, 1.0e154].view(), 1).is_err());
    }

    #[test]
    fn checked_query_rejects_transfer_overflow() {
        let transfer = Transfer::xsigmoid(1.0, 8.0, 1.0).unwrap();
        let x = array![0.0, 1.0e154];
        let landmarks = array![[0.0, 0.0]];
        let result = query_chi_checked(
            x.view(),
            landmarks.view(),
            array![1.0].view(),
            array![1.0].view(),
            &transfer,
            0.0,
            array![1.0].view(),
        );
        assert!(result.is_err());
        let (value, grad) = query_chi(
            x.view(),
            landmarks.view(),
            array![1.0].view(),
            array![1.0].view(),
            &transfer,
            0.0,
            array![1.0].view(),
        );
        assert_eq!(value, OPTIMIZER_PENALTY);
        assert!(grad.iter().all(|component| *component == 0.0));
    }

    #[test]
    fn checked_chi1_rejects_invalid_inputs_and_transfer_overflow() {
        let hd = array![[0.0, 1.0], [1.0, 0.0]];
        let mut stress =
            Stress::new(hd.clone(), hd, Transfer::identity(), 0.0, None, None).unwrap();
        let x = array![0.0, 1.0e154];
        let landmarks = array![[0.0, 0.0], [1.0, 0.0]];
        assert!(
            stress
                .chi1_checked(x.view(), landmarks.view(), 2, array![].view())
                .is_err()
        );

        stress.tfun_ld = Transfer::xsigmoid(1.0, 8.0, 1.0).unwrap();
        assert!(
            stress
                .chi1_checked(x.view(), landmarks.view(), 0, array![1.0, 1.0].view())
                .is_err()
        );
        let (value, grad) = stress.chi1(x.view(), landmarks.view(), 0, array![1.0, 1.0].view());
        assert_eq!(value, OPTIMIZER_PENALTY);
        assert!(grad.iter().all(|component| *component == 0.0));
    }
}
