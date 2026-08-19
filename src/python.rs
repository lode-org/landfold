//! Optional CPython module (`--features python`).
//!
//! pyo3/numpy track dlpk 0.4.1's optional pyo3 0.29 so `--features python`
//! has one pyo3 major. dlpk's `pyo3` feature stays off: DLPack export is
//! `array::to_dlpack` and does not add a second pyo3-ffi `links = "python"` edge.

use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{
    embed_points, farthest_point, fes_from_points, project_one, Euclid, IterOpts, ProjOpts,
    Transfer,
};
use ndarray::Array2;

fn copy_f64_2d(points: PyReadonlyArray2<'_, f64>) -> Array2<f64> {
    points.as_array().to_owned()
}

fn to_pyarray2<'py>(py: Python<'py>, mat: Array2<f64>) -> Bound<'py, PyArray2<f64>> {
    PyArray2::from_owned_array(py, mat)
}

type PyFesResult<'py> = PyResult<(Vec<f64>, Vec<f64>, Bound<'py, PyArray2<f64>>)>;

#[pyfunction]
#[pyo3(signature = (points, lowdim=2, fun_hd="identity", fun_ld="identity", imix=0.0, steps=100))]
fn embed_euclid<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    lowdim: usize,
    fun_hd: &str,
    fun_ld: &str,
    imix: f64,
    steps: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let pts = copy_f64_2d(points);
    let mut opts = IterOpts {
        lowdim,
        imix,
        ..IterOpts::default()
    };
    opts.cg.maxiter = steps;
    opts.tfun_hd = Transfer::from_cli(fun_hd).map_err(|e| PyValueError::new_err(e.to_string()))?;
    opts.tfun_ld = Transfer::from_cli(fun_ld).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let emb = embed_points(pts.view(), &Euclid, &opts)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(to_pyarray2(py, emb.low))
}

#[pyfunction]
#[pyo3(signature = (high, low, query, fun_hd="identity", fun_ld="identity", imix=0.0, gridw=1.0, grid_coarse=21, grid_fine=201, refine=0))]
#[allow(clippy::too_many_arguments)]
fn project_euclid<'py>(
    py: Python<'py>,
    high: PyReadonlyArray2<'py, f64>,
    low: PyReadonlyArray2<'py, f64>,
    query: PyReadonlyArray2<'py, f64>,
    fun_hd: &str,
    fun_ld: &str,
    imix: f64,
    gridw: f64,
    grid_coarse: usize,
    grid_fine: usize,
    refine: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let high = copy_f64_2d(high);
    let low = copy_f64_2d(low);
    let query = copy_f64_2d(query);
    let emb = crate::Embedding::from_landmarks(
        high,
        low,
        &Euclid,
        Transfer::from_cli(fun_hd).map_err(|e| PyValueError::new_err(e.to_string()))?,
        Transfer::from_cli(fun_ld).map_err(|e| PyValueError::new_err(e.to_string()))?,
        imix,
        None,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let opts = ProjOpts {
        gridw,
        grid_coarse,
        grid_fine,
        cg_steps: refine,
    };
    let nq = query.nrows();
    let d = emb.low.ncols();
    let mut out = Array2::<f64>::zeros((nq, d));
    for i in 0..nq {
        let p = project_one(&emb, query.row(i), &Euclid, &opts)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        for h in 0..d {
            out[(i, h)] = p[h];
        }
    }
    Ok(to_pyarray2(py, out))
}

#[pyfunction]
#[pyo3(signature = (points, k, seed=0))]
fn farthest_euclid<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    k: usize,
    seed: usize,
) -> PyResult<(Vec<usize>, Bound<'py, PyArray2<f64>>)> {
    let pts = copy_f64_2d(points);
    let lm = farthest_point(pts.view(), &Euclid, k, None, seed)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((lm.index, to_pyarray2(py, lm.points)))
}

#[pyfunction]
#[pyo3(signature = (xy, nx=80, ny=80, kt=1.0, pad=0.05))]
fn fes_xy<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray2<'py, f64>,
    nx: usize,
    ny: usize,
    kt: f64,
    pad: f64,
) -> PyFesResult<'py> {
    let pts = copy_f64_2d(xy);
    let fes = fes_from_points(pts.view(), nx, ny, kt, pad, None)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut f = Array2::<f64>::from_elem((fes.f.nrows(), fes.f.ncols()), f64::NAN);
    for iy in 0..fes.f.nrows() {
        for ix in 0..fes.f.ncols() {
            if let Some(v) = fes.f[(iy, ix)] {
                f[(iy, ix)] = v;
            }
        }
    }
    Ok((
        fes.x_centers.iter().copied().collect(),
        fes.y_centers.iter().copied().collect(),
        to_pyarray2(py, f),
    ))
}

#[pymodule]
fn landfold(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(embed_euclid, m)?)?;
    m.add_function(wrap_pyfunction!(project_euclid, m)?)?;
    m.add_function(wrap_pyfunction!(farthest_euclid, m)?)?;
    m.add_function(wrap_pyfunction!(fes_xy, m)?)?;
    m.add("version", crate::VERSION)?;
    Ok(())
}
