//! DLPack export of embeddings and distance matrices (readcon-core contract).

use dlpk::DLPackTensor;
use ndarray::{Array2, ArrayD, IxDyn};

use crate::error::{LandfoldError, Result};

/// Zero-copy-capable DLPack export of an `n x k` f64 matrix.
///
/// The tensor takes ownership of a row-major clone so the capsule outlives
/// the Rust stack frame. Same `TryFrom<ArrayD<T>>` path as readcon-core.
pub fn to_dlpack(mat: &Array2<f64>) -> Result<DLPackTensor> {
    let owned: ArrayD<f64> = ArrayD::from_shape_vec(
        IxDyn(&[mat.nrows(), mat.ncols()]),
        mat.iter().copied().collect(),
    )
    .map_err(|e| LandfoldError::Dlpack(e.to_string()))?;
    DLPackTensor::try_from(owned).map_err(|e| LandfoldError::Dlpack(format!("{e:?}")))
}
