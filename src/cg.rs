//! Polak-Ribiere conjugate gradient with Brent line search, via xtsci-optimize.
//!
//! Direction and β are Nocedal and Wright algorithm 5.4 / 5.44 (Polak and
//! Ribiere, *Rev. Fr. Inform. Rech. Opér.* **16**, 35 (1969)). Line search
//! is Brent (*Algorithms for Minimization without Derivatives*, 1973).

use ndarray::{Array1, ArrayView1};
use xtsci_optimize::{Control, LineSearch, Method, Oracle};

use eindir_core::DifferentiableObjective;

use crate::chi_obj::ChiObjective;
use crate::error::{LandfoldError, Result};
use crate::stress::Stress;

#[derive(Clone, Debug)]
pub struct CgOpts {
    pub maxiter: usize,
    pub tol: f64,
    pub ls_maxiter: usize,
    pub ls_tol: f64,
    pub istep: f64,
}

impl Default for CgOpts {
    fn default() -> Self {
        Self {
            maxiter: 100,
            tol: 1e-5,
            ls_maxiter: 5,
            ls_tol: 5e-10,
            istep: 1.0,
        }
    }
}

impl CgOpts {
    pub(crate) fn validate(&self) -> Result<()> {
        if !self.tol.is_finite() || self.tol < 0.0 {
            return Err(LandfoldError::Msg(
                "CG tolerance must be finite and nonnegative".into(),
            ));
        }
        if !self.ls_tol.is_finite() || self.ls_tol <= 0.0 {
            return Err(LandfoldError::Msg(
                "CG line-search tolerance must be finite and > 0".into(),
            ));
        }
        if !self.istep.is_finite() || self.istep <= 0.0 {
            return Err(LandfoldError::Msg(
                "CG initial step must be finite and > 0".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct CgReport {
    pub value: f64,
    pub coords: Array1<f64>,
    pub steps: usize,
}

fn control(opts: &CgOpts) -> (Control, LineSearch) {
    (
        Control {
            maxiter: opts.maxiter,
            gtol: opts.tol,
            istep: opts.istep,
            maxmove: None,
        },
        LineSearch::Brent {
            maxiter: opts.ls_maxiter.max(20),
            tol: opts.ls_tol,
        },
    )
}

fn to_report(r: xtsci_optimize::Report) -> CgReport {
    CgReport {
        value: r.value,
        coords: r.coords,
        steps: r.steps,
    }
}

/// Full-pair χ on packed coordinates. `Solver::Standard` uses this path.
pub fn minimize(
    stress: &Stress,
    init: ArrayView1<f64>,
    d: usize,
    opts: &CgOpts,
) -> Result<CgReport> {
    validate_packed_init(init, stress.n, d)?;
    let obj = ChiObjective::new(stress, d);
    minimize_diff(&obj, init, opts)
}

/// Polak-Ribiere + Brent on any eindir differentiable objective.
pub fn minimize_diff<O>(obj: &O, init: ArrayView1<f64>, opts: &CgOpts) -> Result<CgReport>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    minimize_with(obj, init, opts, Method::polak_ribiere())
}

/// Any xtsci-optimize [`Method`] on χ. landfold extra arms (L-BFGS, BFGS, ...) use this.
pub fn minimize_with<O>(
    obj: &O,
    init: ArrayView1<f64>,
    opts: &CgOpts,
    method: Method,
) -> Result<CgReport>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    opts.validate()?;
    if init.iter().any(|value| !value.is_finite()) {
        return Err(LandfoldError::Msg(
            "optimizer coordinates must be finite".into(),
        ));
    }
    let (ctrl, ls) = control(opts);
    xtsci_optimize::minimize_method(obj, init.to_owned(), &ctrl, method, ls)
        .map(to_report)
        .map_err(|e| LandfoldError::Optimize(e.to_string()))
}

pub(crate) fn validate_packed_init(init: ArrayView1<'_, f64>, n: usize, d: usize) -> Result<()> {
    if d == 0 || init.len() != n.saturating_mul(d) {
        return Err(LandfoldError::Shape("optimizer coordinates shape"));
    }
    if init.iter().any(|value| !value.is_finite()) {
        return Err(LandfoldError::Msg(
            "optimizer coordinates must be finite".into(),
        ));
    }
    Ok(())
}

/// Packed χ through xtsci-optimize. `Solver::Xtsci` uses this path.
pub fn minimize_xtsci(
    stress: &Stress,
    init: ArrayView1<f64>,
    d: usize,
    opts: &CgOpts,
    method: Method,
) -> Result<CgReport> {
    validate_packed_init(init, stress.n, d)?;
    let obj = ChiObjective::new(stress, d);
    minimize_with(&obj, init, opts, method)
}

/// Polak-Ribiere + Brent on any scalar `f` with analytic gradient.
///
/// Out-of-sample projection uses this on the one-point χ of Ceriotti,
/// Tribello and Parrinello, *J. Chem. Theory Comput.* **9**, 1521 (2013),
/// <https://doi.org/10.1021/ct3010563>.
pub fn minimize_oracle<F>(oracle: F, init: ArrayView1<f64>, opts: &CgOpts) -> Result<CgReport>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
{
    try_minimize_oracle(oracle, init, opts)
}

/// Checked Polak-Ribiere + Brent minimization for projection and other
/// callers that must distinguish an optimizer failure from a valid report.
pub fn try_minimize_oracle<F>(oracle: F, init: ArrayView1<f64>, opts: &CgOpts) -> Result<CgReport>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
{
    let obj = Oracle::unbounded(init.len(), oracle);
    minimize_diff(&obj, init, opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn rejects_invalid_options() {
        assert!(
            CgOpts {
                tol: f64::NAN,
                ..CgOpts::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            CgOpts {
                istep: 0.0,
                ..CgOpts::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn oracle_reports_failures_instead_of_sentinel_reports() {
        let report = minimize_oracle(
            |_x| (0.0, array![0.0]),
            array![0.0].view(),
            &CgOpts {
                tol: f64::NAN,
                ..CgOpts::default()
            },
        );
        assert!(report.is_err());
    }
}
