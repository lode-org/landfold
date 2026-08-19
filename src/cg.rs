//! Polak-Ribiere conjugate gradient with Brent line search, via quench-core.
//!
//! Direction and β are Nocedal and Wright algorithm 5.4 / 5.44 (Polak and
//! Ribiere, *Rev. Fr. Inform. Rech. Opér.* **16**, 35 (1969)). Line search
//! is Brent (*Algorithms for Minimization without Derivatives*, 1973).

use ndarray::{Array1, ArrayView1};
use quench_core::{Control, LineSearch, Method, Oracle};

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

fn to_report(r: quench_core::Report) -> CgReport {
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

/// Any quench [`Method`] on χ. landfold extra arms (L-BFGS, BFGS, ...) use this.
pub fn minimize_with<O>(
    obj: &O,
    init: ArrayView1<f64>,
    opts: &CgOpts,
    method: Method,
) -> Result<CgReport>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let (ctrl, ls) = control(opts);
    quench_core::minimize_method(obj, init.to_owned(), &ctrl, method, ls)
        .map(to_report)
        .map_err(|e| LandfoldError::Optimize(e.to_string()))
}

/// Packed χ through a quench method. `Solver::Quench` uses this path.
pub fn minimize_quench(
    stress: &Stress,
    init: ArrayView1<f64>,
    d: usize,
    opts: &CgOpts,
    method: Method,
) -> Result<CgReport> {
    let obj = ChiObjective::new(stress, d);
    minimize_with(&obj, init, opts, method)
}

/// Polak-Ribiere + Brent on any scalar `f` with analytic gradient.
///
/// Out-of-sample projection uses this on the one-point χ of Ceriotti,
/// Tribello and Parrinello, *J. Chem. Theory Comput.* **9**, 1521 (2013),
/// <https://doi.org/10.1021/ct3010563>.
pub fn minimize_oracle<F>(oracle: F, init: ArrayView1<f64>, opts: &CgOpts) -> CgReport
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
{
    try_minimize_oracle(oracle, init, opts).unwrap_or_else(|_| CgReport {
        value: f64::INFINITY,
        coords: init.to_owned(),
        steps: 0,
    })
}

/// Checked Polak-Ribiere + Brent minimization for projection and other
/// callers that must distinguish an optimizer failure from a valid report.
pub fn try_minimize_oracle<F>(
    oracle: F,
    init: ArrayView1<f64>,
    opts: &CgOpts,
) -> Result<CgReport>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
{
    let obj = Oracle::unbounded(init.len(), oracle);
    minimize_diff(&obj, init, opts)
}
