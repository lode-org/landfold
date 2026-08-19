//! DLPack export of embeddings and distance matrices (readcon-core contract).
//!
//! dlpk is the rust-side tensor waist. The optional `dlpk/pyo3` feature stays
//! off so this crate does not pull a second `pyo3-ffi` `links = "python"` edge.

use dlpk::DLPackTensor;
use ndarray::Array2;

use crate::error::{LandfoldError, Result};

/// Owned DLPack export of an `n x k` f64 matrix.
///
/// The tensor takes ownership of a clone so the capsule outlives the Rust
/// stack frame. Same `TryFrom<ndarray::Array<T, D>>` path as readcon-core.
pub fn to_dlpack(mat: &Array2<f64>) -> Result<DLPackTensor> {
    DLPackTensor::try_from(mat.to_owned()).map_err(|e| LandfoldError::Dlpack(e.to_string()))
}
