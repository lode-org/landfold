//! Optional CPython module (`--features python`).
//!
//! pyo3/numpy stay on 0.26 so they share one major with dlpk 0.1.5.
//! dlpk's `pyo3` feature is left off: the crate exports DLPack from Rust
//! and does not need a second pyo3-ffi `links = "python"` edge.

use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{embed_points, Euclid, IterOpts, Transfer};
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
    let sl = points.as_slice().map_err(|_| {
        PyValueError::new_err("points must be a contiguous C-order float64 array")
    })?;
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

#[pymodule]
fn landfold(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(embed_euclid, m)?)?;
    m.add("version", crate::VERSION)?;
    Ok(())
}
