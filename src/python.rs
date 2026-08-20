//! Optional CPython module (`--features python`).
//!
//! pyo3/numpy track dlpk 0.4.1's optional pyo3 0.29 so `--features python`
//! has one pyo3 major. dlpk's `pyo3` feature stays off: DLPack export is
//! `array::to_dlpack` and does not add a second pyo3-ffi `links = "python"` edge.

use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::provenance::{PROVENANCE_SCHEMA, Provenance};
use crate::{
    Euclid, IterOpts, ProjOpts, Transfer, embed_points, farthest_point, fes_from_points,
    project_one,
};
use ndarray::Array2;

fn copy_f64_2d(points: PyReadonlyArray2<'_, f64>) -> Array2<f64> {
    points.as_array().to_owned()
}

fn to_pyarray2<'py>(py: Python<'py>, mat: Array2<f64>) -> Bound<'py, PyArray2<f64>> {
    PyArray2::from_owned_array(py, mat)
}

fn validate_provenance_schema(schema: &str) -> Result<(), &'static str> {
    if schema == PROVENANCE_SCHEMA {
        Ok(())
    } else {
        Err("metadata.provenance.schema has an incompatible version")
    }
}

#[allow(dead_code)]
fn validated_metadata<'py>(metadata: Option<Bound<'py, PyDict>>) -> PyResult<Bound<'py, PyDict>> {
    let metadata = metadata.ok_or_else(|| {
        PyValueError::new_err("metadata with provenance is required for result artifacts")
    })?;
    let provenance = metadata
        .get_item("provenance")?
        .ok_or_else(|| {
            PyValueError::new_err("metadata.provenance is required for result artifacts")
        })?
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("metadata.provenance must be a dictionary"))?
        .to_owned();
    let schema = provenance
        .get_item("schema")?
        .ok_or_else(|| PyValueError::new_err("metadata.provenance.schema is required"))?
        .extract::<String>()
        .map_err(|_| PyValueError::new_err("metadata.provenance.schema must be a string"))?;
    validate_provenance_schema(&schema).map_err(PyValueError::new_err)?;
    let string = |key: &str| -> PyResult<String> {
        provenance
            .get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("metadata.provenance.{key} is required")))?
            .extract()
            .map_err(|_| PyValueError::new_err(format!("metadata.provenance.{key} must be a string")))
    };
    let optional_string = |key: &str| -> PyResult<Option<String>> {
        provenance
            .get_item(key)?
            .map(|value| {
                value.extract().map_err(|_| {
                    PyValueError::new_err(format!("metadata.provenance.{key} must be a string"))
                })
            })
            .transpose()
    };
    let integer = |key: &str| -> PyResult<u16> {
        provenance
            .get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("metadata.provenance.{key} is required")))?
            .extract()
            .map_err(|_| PyValueError::new_err(format!("metadata.provenance.{key} must be an integer")))
    };
    let layout: u32 = provenance
        .get_item("abi_layout_revision")?
        .ok_or_else(|| PyValueError::new_err("metadata.provenance.abi_layout_revision is required"))?
        .extract()
        .map_err(|_| PyValueError::new_err("metadata.provenance.abi_layout_revision must be an integer"))?;
    let engine_id = string("engine_id")?;
    let run_id = string("run_id")?;
    let input_digest = string("input_digest")?;
    let protocol_family = string("protocol_family")?;
    let protocol_major = integer("protocol_major")?;
    let protocol_minor = integer("protocol_minor")?;
    let dlpack_major = integer("dlpack_major")?;
    let dlpack_minor = integer("dlpack_minor")?;
    let record = match optional_string("eindir_revision")? {
        Some(revision) => Provenance::new_with_eindir_revision(
            run_id,
            input_digest,
            engine_id,
            protocol_family,
            protocol_major,
            protocol_minor,
            layout,
            dlpack_major,
            dlpack_minor,
            revision,
        ),
        None => Provenance::new(
            run_id,
            input_digest,
            engine_id,
            protocol_family,
            protocol_major,
            protocol_minor,
            layout,
            dlpack_major,
            dlpack_minor,
        ),
    };
    record.map_err(PyValueError::new_err)?;
    Ok(metadata)
}

fn embedding_options(
    points: Array2<f64>,
    lowdim: usize,
    fun_hd: &str,
    fun_ld: &str,
    imix: f64,
    steps: usize,
) -> crate::Result<crate::Embedding> {
    let mut opts = IterOpts {
        lowdim,
        imix,
        ..IterOpts::default()
    };
    opts.cg.maxiter = steps;
    opts.tfun_hd = Transfer::from_cli(fun_hd)?;
    opts.tfun_ld = Transfer::from_cli(fun_ld)?;
    embed_points(points.view(), &Euclid, &opts)
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
    let emb = embedding_options(copy_f64_2d(points), lowdim, fun_hd, fun_ld, imix, steps)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(to_pyarray2(py, emb.low))
}

/// Return an embedding together with the metadata needed by result consumers.
#[pyfunction]
#[pyo3(signature = (points, lowdim=2, fun_hd="identity", fun_ld="identity", imix=0.0, steps=100, metadata=None))]
#[allow(clippy::too_many_arguments)]
fn embed_euclid_result<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    lowdim: usize,
    fun_hd: &str,
    fun_ld: &str,
    imix: f64,
    steps: usize,
    metadata: Option<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let metadata = validated_metadata(metadata)?;
    let emb = embedding_options(copy_f64_2d(points), lowdim, fun_hd, fun_ld, imix, steps)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let highdim = emb.high.ncols();
    let lowdim = emb.low.ncols();
    let n_points = emb.high.nrows();
    let result = PyDict::new(py);
    result.set_item("schema", crate::artifact::EMBEDDING_SCHEMA)?;
    result.set_item("coordinates", to_pyarray2(py, emb.low))?;
    result.set_item("stress", emb.stress)?;
    result.set_item("n_points", n_points)?;
    result.set_item("highdim", highdim)?;
    result.set_item("lowdim", lowdim)?;
    result.set_item("fun_hd", fun_hd)?;
    result.set_item("fun_ld", fun_ld)?;
    result.set_item("imix", imix)?;
    result.set_item("metadata", metadata)?;
    Ok(result)
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

/// Return projected coordinates together with χ and nearest-landmark data.
#[pyfunction]
#[pyo3(signature = (high, low, query, fun_hd="identity", fun_ld="identity", imix=0.0, gridw=1.0, grid_coarse=21, grid_fine=201, refine=0, metadata=None))]
#[allow(clippy::too_many_arguments)]
fn project_euclid_result<'py>(
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
    metadata: Option<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let metadata = validated_metadata(metadata)?;
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
    let reports = crate::project_many_report(&emb, query.view(), &Euclid, &opts)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dim = emb.low.ncols();
    let mut coordinates = Array2::<f64>::zeros((reports.len(), dim));
    let mut chi = Vec::with_capacity(reports.len());
    let mut nearest = Vec::with_capacity(reports.len());
    let mut nearest_idx = Vec::with_capacity(reports.len());
    for (row, report) in reports.iter().enumerate() {
        for column in 0..dim {
            coordinates[(row, column)] = report.coords[column];
        }
        chi.push(report.chi);
        nearest.push(report.nearest);
        nearest_idx.push(report.nearest_idx);
    }
    let result = PyDict::new(py);
    result.set_item("schema", crate::artifact::PROJECTION_SCHEMA)?;
    result.set_item("coordinates", to_pyarray2(py, coordinates))?;
    result.set_item("chi", chi)?;
    result.set_item("nearest_distance", nearest)?;
    result.set_item("nearest_index", nearest_idx)?;
    result.set_item("fun_hd", fun_hd)?;
    result.set_item("fun_ld", fun_ld)?;
    result.set_item("imix", imix)?;
    result.set_item("metadata", metadata)?;
    Ok(result)
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

/// Return a FES with density and grid metadata for plotting adapters.
#[pyfunction]
#[pyo3(signature = (xy, nx=80, ny=80, kt=1.0, pad=0.05, metadata=None))]
fn fes_xy_result<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray2<'py, f64>,
    nx: usize,
    ny: usize,
    kt: f64,
    pad: f64,
    metadata: Option<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let metadata = validated_metadata(metadata)?;
    let pts = copy_f64_2d(xy);
    let fes = fes_from_points(pts.view(), nx, ny, kt, pad, None)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let f = Array2::from_shape_fn(fes.f.raw_dim(), |(iy, ix)| {
        fes.f[(iy, ix)].unwrap_or(f64::NAN)
    });
    let result = PyDict::new(py);
    result.set_item("schema", crate::artifact::FES_SCHEMA)?;
    result.set_item("x", fes.x_centers.to_vec())?;
    result.set_item("y", fes.y_centers.to_vec())?;
    result.set_item("free_energy", to_pyarray2(py, f))?;
    result.set_item("density", to_pyarray2(py, fes.rho))?;
    result.set_item("kt", kt)?;
    result.set_item("metadata", metadata)?;
    Ok(result)
}

#[pymodule]
fn landfold(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(embed_euclid, m)?)?;
    m.add_function(wrap_pyfunction!(embed_euclid_result, m)?)?;
    m.add_function(wrap_pyfunction!(project_euclid, m)?)?;
    m.add_function(wrap_pyfunction!(project_euclid_result, m)?)?;
    m.add_function(wrap_pyfunction!(farthest_euclid, m)?)?;
    m.add_function(wrap_pyfunction!(fes_xy, m)?)?;
    m.add_function(wrap_pyfunction!(fes_xy_result, m)?)?;
    m.add("version", crate::VERSION)?;
    m.add("eindir_revision", crate::EINDIR_REVISION)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_schema_is_required_and_versioned() {
        assert!(validate_provenance_schema(PROVENANCE_SCHEMA).is_ok());
        assert!(validate_provenance_schema("landfold.provenance.v2").is_err());
        assert!(validate_provenance_schema("").is_err());
    }
}
