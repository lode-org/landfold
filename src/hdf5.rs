//! Adapter for flattened trajectory datasets stored in HDF5.
//!
//! The adapter consumes the interchange layout used by chemparseplot:
//! `/path/images` contains one flattened Cartesian coordinate row per frame.
//! HDF5 remains optional so users that use CON or chemfiles do not link the
//! native HDF5 library.

use std::path::Path;

use ndarray::Array2;

use crate::trajectory::FrameBatch;
use crate::{LandfoldError, Result};

const IMAGES_DATASET: &str = "/path/images";

/// Read chemparseplot-compatible flattened images into a validated batch.
pub fn read_hdf5_batch(path: &Path) -> Result<FrameBatch> {
    let file = hdf5::File::open(path).map_err(|error| LandfoldError::Parse(error.to_string()))?;
    let dataset = file
        .dataset(IMAGES_DATASET)
        .map_err(|error| LandfoldError::Parse(error.to_string()))?;
    let shape = dataset.shape();
    if shape.len() != 2 {
        return Err(LandfoldError::Shape(
            "HDF5 /path/images must be a two-dimensional dataset",
        ));
    }
    if shape[1] == 0 || !shape[1].is_multiple_of(3) {
        return Err(LandfoldError::Shape(
            "HDF5 /path/images width must be a nonzero multiple of three",
        ));
    }
    let values = dataset
        .read_raw::<f64>()
        .map_err(|error| LandfoldError::Parse(error.to_string()))?;
    let points = Array2::from_shape_vec((shape[0], shape[1]), values)
        .map_err(|_| LandfoldError::Shape("HDF5 /path/images data shape"))?;
    let n_atoms = shape[1] / 3;
    let atom_ids = read_ids(&file, "/metadata/atom_ids", n_atoms)?
        .unwrap_or_else(|| (0..shape[1] as u64 / 3).collect());
    let frame_ids = read_ids(&file, "/path/frame_ids", shape[0])?
        .unwrap_or_else(|| (0..shape[0] as u64).collect());
    let mut batch = FrameBatch::from_flattened_points(points, atom_ids, frame_ids, None)?;
    batch.atomic_numbers = read_atomic_numbers(&file, "/metadata/atomic_numbers", n_atoms)?;
    if file.link_exists("/metadata/cell") {
        let dataset = file
            .dataset("/metadata/cell")
            .map_err(|error| LandfoldError::Parse(error.to_string()))?;
        if dataset.shape() != [3, 3] {
            return Err(LandfoldError::Shape("HDF5 cell must be a 3x3 dataset"));
        }
        let values = dataset
            .read_raw::<f64>()
            .map_err(|error| LandfoldError::Parse(error.to_string()))?;
        batch.cell = Some(
            Array2::from_shape_vec((3, 3), values)
                .map_err(|_| LandfoldError::Shape("HDF5 cell data shape"))?,
        );
    }
    batch.validate()?;
    Ok(batch)
}

fn read_atomic_numbers(
    file: &hdf5::File,
    path: &str,
    expected: usize,
) -> Result<Option<Vec<u64>>> {
    if !file.link_exists(path) {
        return Ok(None);
    }
    let dataset = file
        .dataset(path)
        .map_err(|error| LandfoldError::Parse(error.to_string()))?;
    if dataset.ndim() != 1 || dataset.shape()[0] != expected {
        return Err(LandfoldError::Shape(
            "HDF5 atomic number dataset shape",
        ));
    }
    let values = dataset
        .read_raw::<i64>()
        .map_err(|error| LandfoldError::Parse(error.to_string()))?;
    values
        .into_iter()
        .map(|number| {
            u64::try_from(number)
                .ok()
                .filter(|&number| (1..=118).contains(&number))
                .ok_or_else(|| {
                    LandfoldError::Msg(
                        "HDF5 atomic numbers must be in the range 1..=118".into(),
                    )
                })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn read_ids(file: &hdf5::File, path: &str, expected: usize) -> Result<Option<Vec<u64>>> {
    if !file.link_exists(path) {
        return Ok(None);
    }
    let dataset = file
        .dataset(path)
        .map_err(|error| LandfoldError::Parse(error.to_string()))?;
    if dataset.ndim() != 1 || dataset.shape()[0] != expected {
        return Err(LandfoldError::Shape(
            "HDF5 trajectory identity dataset shape",
        ));
    }
    let ids = dataset
        .read_raw::<u64>()
        .map_err(|error| LandfoldError::Parse(error.to_string()))?;
    Ok(Some(ids))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reads_images_and_explicit_identity() {
        let path = std::env::temp_dir().join(format!("landfold-hdf5-{}.h5", std::process::id()));
        let file = hdf5::File::create(&path).expect("create HDF5 fixture");
        let path_group = file.create_group("path").expect("create path group");
        path_group
            .new_dataset::<f64>()
            .shape((2, 6))
            .create("images")
            .expect("create images")
            .write_raw(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
            .expect("write images");
        path_group
            .new_dataset::<i64>()
            .shape(2)
            .create("frame_ids")
            .expect("create frame IDs")
            .write_raw(&[41_u64, 42])
            .expect("write frame IDs");
        let metadata_group = file
            .create_group("metadata")
            .expect("create metadata group");
        metadata_group
            .new_dataset::<u64>()
            .shape(2)
            .create("atom_ids")
            .expect("create atom IDs")
            .write_raw(&[7_u64, 8])
            .expect("write atom IDs");
        metadata_group
            .new_dataset::<u64>()
            .shape(2)
            .create("atomic_numbers")
            .expect("create atomic numbers")
            .write_raw(&[6_i64, 1])
            .expect("write atomic numbers");
        metadata_group
            .new_dataset::<f64>()
            .shape((3, 3))
            .create("cell")
            .expect("create cell")
            .write_raw(&[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0])
            .expect("write cell");
        drop(file);

        let batch = read_hdf5_batch(&path).expect("read HDF5 fixture");
        std::fs::remove_file(path).expect("remove HDF5 fixture");
        assert_eq!(batch.frame_ids, vec![41, 42]);
        assert_eq!(batch.atom_ids, vec![7, 8]);
        assert_eq!(batch.atomic_numbers, Some(vec![6, 1]));
        assert_eq!(batch.cell.as_ref().expect("cell").dim(), (3, 3));
        assert_eq!(
            batch.frame(1).expect("second frame").row(0).to_vec(),
            vec![6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn defaults_identity_and_rejects_nonfinite_images() {
        let path =
            std::env::temp_dir().join(format!("landfold-hdf5-invalid-{}.h5", std::process::id()));
        let file = hdf5::File::create(&path).expect("create HDF5 fixture");
        let path_group = file.create_group("path").expect("create path group");
        path_group
            .new_dataset::<f64>()
            .shape((1, 3))
            .create("images")
            .expect("create images")
            .write_raw(&[0.0, f64::NAN, 1.0])
            .expect("write images");
        drop(file);

        let error = read_hdf5_batch(&path).expect_err("non-finite images must be rejected");
        std::fs::remove_file(path).expect("remove HDF5 fixture");
        assert!(error.to_string().contains("finite"));
    }

    #[test]
    fn reads_float32_images_into_f64_coordinates() {
        let path =
            std::env::temp_dir().join(format!("landfold-hdf5-f32-{}.h5", std::process::id()));
        let file = hdf5::File::create(&path).expect("create HDF5 fixture");
        let path_group = file.create_group("path").expect("create path group");
        path_group
            .new_dataset::<f32>()
            .shape((1, 3))
            .create("images")
            .expect("create float32 images")
            .write_raw(&[1.25_f32, 2.5, 3.75])
            .expect("write float32 images");
        drop(file);

        let batch = read_hdf5_batch(&path).expect("read float32 HDF5 fixture");
        std::fs::remove_file(path).expect("remove HDF5 fixture");
        assert_eq!(batch.frame(0).expect("first frame").row(0).to_vec(), [1.25, 2.5, 3.75]);
    }
}
