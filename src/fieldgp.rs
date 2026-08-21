//! MAP inverse-multiquadric GP of a readable field on the χ plane.
//!
//! MethodsX (Goswami, *MethodsX* 2026, 10.1016/j.mex.2026.103851): the
//! field is not occupancy invert. On a landfold map it is the basin
//! coordinate `ξ = d(x,a)/(d(x,a)+d(x,b))` of the high-D descriptors.
//! Hyperparameters are type-II maximum a posteriori under a Gaussian
//! prior on `log ℓ`, the same MAP-NLL path as ChemGP (`train.rs`).
//! Posterior variance is the reliability.

use nalgebra::{Cholesky, DMatrix, DVector};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{LandfoldError, Result};

/// Inverse multiquadric: `k = σ_f² / sqrt(1 + r²/ℓ²)`.
pub fn imq(r2: f64, sigma_f2: f64, ell2: f64) -> f64 {
    sigma_f2 / (1.0 + r2 / ell2).sqrt()
}

pub fn basin_coordinate(
    descriptors: ArrayView2<f64>,
    ref_a: ArrayView1<f64>,
    ref_b: ArrayView1<f64>,
) -> Result<Array1<f64>> {
    let n = descriptors.nrows();
    let d = descriptors.ncols();
    if n == 0 {
        return Err(LandfoldError::Empty);
    }
    if ref_a.len() != d || ref_b.len() != d {
        return Err(LandfoldError::Shape(
            "basin refs must match descriptor dimension",
        ));
    }
    let mut gap2 = 0.0;
    for k in 0..d {
        let e = ref_a[k] - ref_b[k];
        gap2 += e * e;
    }
    if !(gap2 > 0.0) {
        return Err(LandfoldError::Msg("basin refs must differ".into()));
    }
    let mut xi = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut da = 0.0;
        let mut db = 0.0;
        for k in 0..d {
            let xa = descriptors[(i, k)] - ref_a[k];
            let xb = descriptors[(i, k)] - ref_b[k];
            da += xa * xa;
            db += xb * xb;
        }
        let denom = da.sqrt() + db.sqrt();
        if !(denom > 0.0) {
            return Err(LandfoldError::Msg(
                "descriptor coincides with both basin refs".into(),
            ));
        }
        xi[i] = da.sqrt() / denom;
    }
    Ok(xi)
}

#[derive(Clone, Debug)]
pub struct FieldGp {
    pub sigma_f2: f64,
    pub ell: f64,
    pub noise: f64,
    pub nll: f64,
}

#[derive(Clone, Debug)]
pub struct FieldPredict {
    pub mean: Array1<f64>,
    pub var: Array1<f64>,
}

fn kernel_matrix(x: ArrayView2<f64>, sigma_f2: f64, ell: f64, noise: f64) -> Result<DMatrix<f64>> {
    let n = x.nrows();
    let ell2 = ell * ell;
    let mut k = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut r2 = 0.0;
            for h in 0..x.ncols() {
                let e = x[(i, h)] - x[(j, h)];
                r2 += e * e;
            }
            let mut kij = imq(r2, sigma_f2, ell2);
            if i == j {
                kij += noise;
            }
            if !kij.is_finite() {
                return Err(LandfoldError::Msg("IMQ kernel is not finite".into()));
            }
            k[(i, j)] = kij;
            k[(j, i)] = kij;
        }
    }
    Ok(k)
}

fn nll_of(x: ArrayView2<f64>, y: ArrayView1<f64>, sigma_f2: f64, ell: f64, noise: f64) -> Result<f64> {
    let n = x.nrows();
    let k = kernel_matrix(x, sigma_f2, ell, noise)?;
    let chol = Cholesky::new(k).ok_or_else(|| {
        LandfoldError::Msg("IMQ covariance is not positive definite".into())
    })?;
    let yv = DVector::from_iterator(n, y.iter().copied());
    let alpha = chol.solve(&yv);
    let quad = yv.dot(&alpha);
    let mut logdet = 0.0;
    for i in 0..n {
        let lii = chol.l()[(i, i)];
        if !(lii > 0.0) {
            return Err(LandfoldError::Msg("IMQ Cholesky diagonal is not positive".into()));
        }
        logdet += lii.ln();
    }
    logdet *= 2.0;
    Ok(0.5 * quad + 0.5 * logdet + 0.5 * (n as f64) * (2.0 * std::f64::consts::PI).ln())
}

/// Type-II MAP: grid on `log ℓ` with a Gaussian prior, `σ_f² = Var(y)`,
/// `σ_n² = 10⁻³ Var(y)`. ChemGP keeps noise fixed and MAP-fits the
/// lengthscale the same way.
pub fn fit_imq_map(x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<FieldGp> {
    let n = x.nrows();
    if n == 0 || n != y.len() {
        return Err(LandfoldError::Shape("field GP x/y row counts differ"));
    }
    if x.ncols() == 0 {
        return Err(LandfoldError::Shape("field GP needs positive dimension"));
    }
    let mean: f64 = y.iter().sum::<f64>() / n as f64;
    let mut var = 0.0;
    for &v in y {
        let e = v - mean;
        var += e * e;
    }
    var /= n as f64;
    if !(var > 0.0) {
        return Err(LandfoldError::Msg("field is constant".into()));
    }
    let sigma_f2 = var;
    let noise = 1e-3 * var;
    let mut span: f64 = 0.0;
    for h in 0..x.ncols() {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for i in 0..n {
            lo = lo.min(x[(i, h)]);
            hi = hi.max(x[(i, h)]);
        }
        span = span.max(hi - lo);
    }
    if !(span > 0.0) {
        return Err(LandfoldError::Msg("field GP inputs are coincident".into()));
    }
    let mu = span.ln() - 1.0;
    let tau = 1.0;
    let mut best = FieldGp {
        sigma_f2,
        ell: span,
        noise,
        nll: f64::INFINITY,
    };
    for k in 0..25 {
        let log_ell = mu - 2.0 + 4.0 * (k as f64) / 24.0;
        let ell = log_ell.exp();
        let nll = nll_of(x, y, sigma_f2, ell, noise)?;
        let map = nll + 0.5 * ((log_ell - mu) / tau).powi(2);
        if map < best.nll {
            best = FieldGp {
                sigma_f2,
                ell,
                noise,
                nll: map,
            };
        }
    }
    if !best.nll.is_finite() {
        return Err(LandfoldError::Msg("IMQ MAP nll is not finite".into()));
    }
    Ok(best)
}

pub fn predict_imq(
    model: &FieldGp,
    x_train: ArrayView2<f64>,
    y: ArrayView1<f64>,
    x_star: ArrayView2<f64>,
) -> Result<FieldPredict> {
    if x_train.nrows() != y.len() || x_train.ncols() != x_star.ncols() {
        return Err(LandfoldError::Shape("field GP predict shape"));
    }
    let n = x_train.nrows();
    let m = x_star.nrows();
    let k = kernel_matrix(x_train, model.sigma_f2, model.ell, model.noise)?;
    let chol = Cholesky::new(k).ok_or_else(|| {
        LandfoldError::Msg("IMQ covariance is not positive definite".into())
    })?;
    let yv = DVector::from_iterator(n, y.iter().copied());
    let alpha = chol.solve(&yv);
    let ell2 = model.ell * model.ell;
    let mut mean = Array1::<f64>::zeros(m);
    let mut var = Array1::<f64>::zeros(m);
    for s in 0..m {
        let mut ks = DVector::<f64>::zeros(n);
        for i in 0..n {
            let mut r2 = 0.0;
            for h in 0..x_train.ncols() {
                let e = x_train[(i, h)] - x_star[(s, h)];
                r2 += e * e;
            }
            ks[i] = imq(r2, model.sigma_f2, ell2);
        }
        mean[s] = ks.dot(&alpha);
        let v = chol.solve(&ks);
        var[s] = (model.sigma_f2 - ks.dot(&v)).max(0.0);
        if !mean[s].is_finite() || !var[s].is_finite() {
            return Err(LandfoldError::Msg("IMQ prediction is not finite".into()));
        }
    }
    Ok(FieldPredict { mean, var })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn basin_ends() {
        let desc = array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]];
        let xi = basin_coordinate(desc.view(), array![0.0, 0.0].view(), array![1.0, 0.0].view())
            .unwrap();
        assert!((xi[0] - 0.0).abs() < 1e-12);
        assert!((xi[1] - 1.0).abs() < 1e-12);
        assert!((xi[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn imq_interpolates_two_wells() {
        let x = array![[0.0, 0.0], [2.0, 0.0]];
        let y = array![0.0, 1.0];
        let gp = fit_imq_map(x.view(), y.view()).unwrap();
        let pred = predict_imq(&gp, x.view(), y.view(), x.view()).unwrap();
        assert!((pred.mean[0] - 0.0).abs() < 0.05);
        assert!((pred.mean[1] - 1.0).abs() < 0.05);
        assert!(pred.var[0] < 0.05);
    }
}
