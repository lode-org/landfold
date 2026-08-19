//! Optional CPython module (`--features python`).
//!
//! pyo3/numpy stay on 0.26 so they share one major with dlpk 0.1.5.
//! dlpk's `pyo3` feature is left off: the crate exports DLPack from Rust
//! and does not need a second pyo3-ffi `links = "python"` edge.

use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{
    Euclid, IterOpts, ProjOpts, Transfer, embed_points, farthest_point, fes_from_points,
    project_one,
};
use ndarray::Array2;

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
    // numpy 0.26 ships ndarray 0.16; landfold is on 0.17. Copy through slices.
    let shape = points.shape();
    let (n, d) = (shape[0], shape[1]);
    let sl = points
        .as_slice()
        .map_err(|_| PyValueError::new_err("points must be a contiguous C-order float64 array"))?;
    let mut pts = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for h in 0..d {
            pts[(i, h)] = sl[i * d + h];
        }
    }
    let mut opts = IterOpts::default();
    opts.lowdim = lowdim;
    opts.imix = imix;
    opts.cg.maxiter = steps;
    opts.tfun_hd = Transfer::from_cli(fun_hd).map_err(|e| PyValueError::new_err(e.to_string()))?;
    opts.tfun_ld = Transfer::from_cli(fun_ld).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let emb = embed_points(pts.view(), &Euclid, &opts)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut rows = Vec::with_capacity(emb.low.nrows());
    for i in 0..emb.low.nrows() {
        let mut row = Vec::with_capacity(emb.low.ncols());
        for h in 0..emb.low.ncols() {
            row.push(emb.low[(i, h)]);
        }
        rows.push(row);
    }
    PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn copy_f64_2d(points: PyReadonlyArray2<'_, f64>) -> PyResult<Array2<f64>> {
    let shape = points.shape();
    let (n, d) = (shape[0], shape[1]);
    let sl = points
        .as_slice()
        .map_err(|_| PyValueError::new_err("points must be a contiguous C-order float64 array"))?;
    let mut pts = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for h in 0..d {
            pts[(i, h)] = sl[i * d + h];
        }
    }
    Ok(pts)
}

#[pyfunction]
#[pyo3(signature = (high, low, query, fun_hd="identity", fun_ld="identity", imix=0.0, gridw=1.0, grid_coarse=21, grid_fine=201, refine=0))]
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
    let high = copy_f64_2d(high)?;
    let low = copy_f64_2d(low)?;
    let query = copy_f64_2d(query)?;
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
    let mut rows = Vec::with_capacity(nq);
    for i in 0..nq {
        let p = project_one(&emb, query.row(i), &Euclid, &opts)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut row = Vec::with_capacity(d);
        for h in 0..d {
            row.push(p[h]);
        }
        rows.push(row);
    }
    PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (points, k, seed=0))]
fn farthest_euclid<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    k: usize,
    seed: usize,
) -> PyResult<(Vec<usize>, Bound<'py, PyArray2<f64>>)> {
    let pts = copy_f64_2d(points)?;
    let lm = farthest_point(pts.view(), &Euclid, k, None, seed)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut rows = Vec::with_capacity(lm.points.nrows());
    for i in 0..lm.points.nrows() {
        let mut row = Vec::with_capacity(lm.points.ncols());
        for h in 0..lm.points.ncols() {
            row.push(lm.points[(i, h)]);
        }
        rows.push(row);
    }
    let arr = PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((lm.index, arr))
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
) -> PyResult<(Vec<f64>, Vec<f64>, Bound<'py, PyArray2<f64>>)> {
    let pts = copy_f64_2d(xy)?;
    let fes = fes_from_points(pts.view(), nx, ny, kt, pad, None)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut rows = Vec::with_capacity(fes.f.nrows());
    for iy in 0..fes.f.nrows() {
        let mut row = Vec::with_capacity(fes.f.ncols());
        for ix in 0..fes.f.ncols() {
            row.push(fes.f[(iy, ix)].unwrap_or(f64::NAN));
        }
        rows.push(row);
    }
    let arr = PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        fes.x_centers.iter().copied().collect(),
        fes.y_centers.iter().copied().collect(),
        arr,
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
