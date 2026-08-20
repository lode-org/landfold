//! Adapter for flattened trajectory datasets stored in HDF5.
//!
//! The adapter consumes the interchange layout used by chemparseplot:
//! `/path/images` contains one flattened Cartesian coordinate row per frame.
//! HDF5 remains optional so users that use CON or chemfiles do not link the
//! native HDF5 library.

use std::{collections::BTreeMap, path::Path};

use hdf5::types::{VarLenAscii, VarLenUnicode};
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
    let length_unit = read_length_unit(&file, "/metadata/length_unit")?;
    let mut batch = FrameBatch::from_flattened_points(points, atom_ids, frame_ids, length_unit)?;
    batch.atomic_numbers = read_atomic_numbers(&file, "/metadata/atomic_numbers", n_atoms)?;
    batch.atom_symbols = read_atom_symbols(&file, "/metadata/atom_symbols", n_atoms)?;
    batch.gradients = read_frame_gradients(&file, "/path/gradients", shape[0], shape[1])?;
    if file.link_exists("/metadata/cell") {
        let dataset = file
            .dataset("/metadata/cell")
            .map_err(|error| LandfoldError::Parse(error.to_string()))?;
        if dataset.shape() != [3, 3] && dataset.shape() != [9] {
            return Err(LandfoldError::Shape(
                "HDF5 cell must be a 3x3 or flattened 9-element dataset",
            ));
        }
        let values = dataset
            .read_raw::<f64>()
            .map_err(|error| LandfoldError::Parse(error.to_string()))?;
        batch.cell = Some(
            Array2::from_shape_vec((3, 3), values)
                .map_err(|_| LandfoldError::Shape("HDF5 cell data shape"))?,
        );
    }
    let path_observables = [
        ("energies", "/path/energies"),
        ("f_para", "/path/f_para"),
        ("rxn_coord", "/path/rxn_coord"),
    ];
    let mut metadata = vec![BTreeMap::new(); shape[0]];
    for (key, path) in path_observables {
        if let Some(values) = read_frame_scalars(&file, path, shape[0])? {
            for (frame_metadata, value) in metadata.iter_mut().zip(values) {
                frame_metadata.insert(key.into(), serde_json::json!(value));
            }
        }
    }
    if metadata
        .iter()
        .any(|frame_metadata| !frame_metadata.is_empty())
    {
        batch.metadata = metadata;
    }
    batch.validate()?;
    Ok(batch)
}

fn read_length_unit(file: &hdf5::File, path: &str) -> Result<Option<String>> {
    if !file.link_exists(path) {
        return Ok(None);
    }
    let dataset = file
        .dataset(path)
        .map_err(|error| LandfoldError::Parse(error.to_string()))?;
    if dataset.ndim() != 0 {
        return Err(LandfoldError::Shape("HDF5 length unit must be scalar"));
    }
    let value = dataset
        .read_scalar::<VarLenUnicode>()
        .map_err(|error| LandfoldError::Parse(error.to_string()))?;
    let unit = value.as_str().trim();
    if unit.is_empty() {
        return Err(LandfoldError::Msg(
            "HDF5 length unit must not be empty".into(),
        ));
    }
    Ok(Some(unit.to_owned()))
}

fn read_frame_scalars(file: &hdf5::File, path: &str, expected: usize) -> Result<Option<Vec<f64>>> {
    if !file.link_exists(path) {
        return Ok(None);
    }
    let dataset = file
        .dataset(path)
        .map_err(|error| LandfoldError::Parse(error.to_string()))?;
    if dataset.ndim() != 1 || dataset.shape()[0] != expected {
        return Err(LandfoldError::Shape("HDF5 path observable dataset shape"));
    }
    let values = dataset
        .read_raw::<f64>()
        .map_err(|error| LandfoldError::Parse(error.to_string()))?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(LandfoldError::Msg(
            "HDF5 path observables must be finite".into(),
        ));
    }
    Ok(Some(values))
}

fn read_frame_gradients(
    file: &hdf5::File,
    path: &str,
    expected_frames: usize,
    expected_width: usize,
) -> Result<Option<Array2<f64>>> {
    if !file.link_exists(path) {
        return Ok(None);
    }
    let dataset = file
        .dataset(path)
        .map_err(|error| LandfoldError::Parse(error.to_string()))?;
    if dataset.shape() != [expected_frames, expected_width] {
        return Err(LandfoldError::Shape("HDF5 path gradients dataset shape"));
    }
    let values = dataset
        .read_raw::<f64>()
        .map_err(|error| LandfoldError::Parse(error.to_string()))?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(LandfoldError::Msg(
            "HDF5 path gradients must be finite".into(),
        ));
    }
    Array2::from_shape_vec((expected_frames, expected_width), values)
        .map(Some)
        .map_err(|_| LandfoldError::Shape("HDF5 path gradients data shape"))
}

fn read_atomic_numbers(file: &hdf5::File, path: &str, expected: usize) -> Result<Option<Vec<u64>>> {
    if !file.link_exists(path) {
        return Ok(None);
    }
    let dataset = file
        .dataset(path)
        .map_err(|error| LandfoldError::Parse(error.to_string()))?;
    if dataset.ndim() != 1 || dataset.shape()[0] != expected {
        return Err(LandfoldError::Shape("HDF5 atomic number dataset shape"));
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
                    LandfoldError::Msg("HDF5 atomic numbers must be in the range 1..=118".into())
                })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn read_atom_symbols(
    file: &hdf5::File,
    path: &str,
    expected: usize,
) -> Result<Option<Vec<String>>> {
    if !file.link_exists(path) {
        return Ok(None);
    }
    let dataset = file
        .dataset(path)
        .map_err(|error| LandfoldError::Parse(error.to_string()))?;
    if dataset.ndim() != 1 || dataset.shape()[0] != expected {
        return Err(LandfoldError::Shape("HDF5 atom symbol dataset shape"));
    }
    let symbols = if let Ok(values) = dataset.read_raw::<VarLenUnicode>() {
        values
            .into_iter()
            .map(|symbol| symbol.as_str().to_owned())
            .collect::<Vec<_>>()
    } else {
        dataset
            .read_raw::<VarLenAscii>()
            .map_err(|error| LandfoldError::Parse(error.to_string()))?
            .into_iter()
            .map(|symbol| symbol.as_str().to_owned())
            .collect::<Vec<_>>()
    };
    if symbols.iter().any(|symbol| symbol.trim().is_empty()) {
        return Err(LandfoldError::Msg(
            "HDF5 atom symbols must be nonempty".into(),
        ));
    }
    Ok(Some(symbols))
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
    use std::str::FromStr;
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
            .new_dataset::<f64>()
            .shape((2, 6))
            .create("gradients")
            .expect("create gradients")
            .write_raw(&[0.0; 12])
            .expect("write gradients");
        path_group
            .new_dataset::<i64>()
            .shape(2)
            .create("frame_ids")
            .expect("create frame IDs")
            .write_raw(&[41_i64, 42])
            .expect("write frame IDs");
        for (name, values) in [
            ("energies", [1.0, 2.0]),
            ("f_para", [0.1, 0.2]),
            ("rxn_coord", [0.0, 1.0]),
        ] {
            path_group
                .new_dataset::<f64>()
                .shape(2)
                .create(name)
                .expect("create path observable")
                .write_raw(&values)
                .expect("write path observable");
        }
        let metadata_group = file
            .create_group("metadata")
            .expect("create metadata group");
        metadata_group
            .new_dataset::<VarLenUnicode>()
            .create("length_unit")
            .expect("create length unit")
            .write_scalar(&VarLenUnicode::from_str("angstrom").expect("valid length unit"))
            .expect("write length unit");
        metadata_group
            .new_dataset::<u64>()
            .shape(2)
            .create("atom_ids")
            .expect("create atom IDs")
            .write_raw(&[7_u64, 8])
            .expect("write atom IDs");
        metadata_group
            .new_dataset::<i64>()
            .shape(2)
            .create("atomic_numbers")
            .expect("create atomic numbers")
            .write_raw(&[6_i64, 1])
            .expect("write atomic numbers");
        metadata_group
            .new_dataset::<VarLenUnicode>()
            .shape(2)
            .create("atom_symbols")
            .expect("create atom symbols")
            .write_raw(&[
                VarLenUnicode::from_str("C").expect("valid symbol"),
                VarLenUnicode::from_str("H").expect("valid symbol"),
            ])
            .expect("write atom symbols");
        metadata_group
            .new_dataset::<f64>()
            .shape(9)
            .create("cell")
            .expect("create cell")
            .write_raw(&[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0])
            .expect("write cell");
        drop(file);

        let batch = read_hdf5_batch(&path).expect("read HDF5 fixture");
        std::fs::remove_file(path).expect("remove HDF5 fixture");
        assert_eq!(batch.frame_ids, vec![41, 42]);
        assert_eq!(batch.atom_ids, vec![7, 8]);
        assert_eq!(batch.length_unit.as_deref(), Some("angstrom"));
        assert_eq!(batch.atomic_numbers, Some(vec![6, 1]));
        assert_eq!(batch.atom_symbols, Some(vec!["C".into(), "H".into()]));
        assert_eq!(batch.gradients.as_ref().expect("gradients").dim(), (2, 6));
        assert_eq!(batch.cell.as_ref().expect("cell").dim(), (3, 3));
        assert_eq!(
            batch.frame_metadata(1).and_then(|m| m.get("energies")),
            Some(&serde_json::json!(2.0))
        );
        assert_eq!(
            batch.frame_metadata(0).and_then(|m| m.get("f_para")),
            Some(&serde_json::json!(0.1))
        );
        assert_eq!(
            batch.frame_metadata(1).and_then(|m| m.get("rxn_coord")),
            Some(&serde_json::json!(1.0))
        );
        assert_eq!(
            batch.frame(1).expect("second frame").row(0).to_vec(),
            vec![6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn reads_ascii_atom_symbols() {
        let path =
            std::env::temp_dir().join(format!("landfold-hdf5-ascii-{}.h5", std::process::id()));
        let file = hdf5::File::create(&path).expect("create HDF5 fixture");
        file.create_group("path")
            .expect("create path group")
            .new_dataset::<f64>()
            .shape((1, 6))
            .create("images")
            .expect("create images")
            .write_raw(&[0.0; 6])
            .expect("write images");
        file.create_group("metadata")
            .expect("create metadata group")
            .new_dataset::<VarLenAscii>()
            .shape(2)
            .create("atom_symbols")
            .expect("create atom symbols")
            .write_raw(&[
                VarLenAscii::from_ascii("C").expect("valid symbol"),
                VarLenAscii::from_ascii("H").expect("valid symbol"),
            ])
            .expect("write atom symbols");
        drop(file);

        let batch = read_hdf5_batch(&path).expect("read ASCII HDF5 fixture");
        std::fs::remove_file(path).expect("remove HDF5 fixture");
        assert_eq!(batch.atom_symbols, Some(vec!["C".into(), "H".into()]));
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
        assert_eq!(
            batch.frame(0).expect("first frame").row(0).to_vec(),
            [1.25, 2.5, 3.75]
        );
    }

    #[test]
    fn rejects_nonfinite_path_observables() {
        let path = std::env::temp_dir().join(format!(
            "landfold-hdf5-nonfinite-observable-{}.h5",
            std::process::id()
        ));
        let file = hdf5::File::create(&path).expect("create HDF5 fixture");
        let path_group = file.create_group("path").expect("create path group");
        path_group
            .new_dataset::<f64>()
            .shape((1, 3))
            .create("images")
            .expect("create images")
            .write_raw(&[0.0, 1.0, 2.0])
            .expect("write images");
        path_group
            .new_dataset::<f64>()
            .shape(1)
            .create("energies")
            .expect("create energies")
            .write_raw(&[f64::NAN])
            .expect("write energies");
        drop(file);

        let error = read_hdf5_batch(&path).expect_err("non-finite observable must be rejected");
        std::fs::remove_file(path).expect("remove HDF5 fixture");
        assert!(error.to_string().contains("observables must be finite"));
    }

    #[test]
    fn rejects_nonfinite_path_gradients() {
        let path = std::env::temp_dir().join(format!(
            "landfold-hdf5-nonfinite-gradients-{}.h5",
            std::process::id()
        ));
        let file = hdf5::File::create(&path).expect("create HDF5 fixture");
        let path_group = file.create_group("path").expect("create path group");
        path_group
            .new_dataset::<f64>()
            .shape((1, 3))
            .create("images")
            .expect("create images")
            .write_raw(&[0.0, 1.0, 2.0])
            .expect("write images");
        path_group
            .new_dataset::<f64>()
            .shape((1, 3))
            .create("gradients")
            .expect("create gradients")
            .write_raw(&[0.0, f64::INFINITY, 0.0])
            .expect("write gradients");
        drop(file);

        let error = read_hdf5_batch(&path).expect_err("non-finite gradients must be rejected");
        std::fs::remove_file(path).expect("remove HDF5 fixture");
        assert!(error.to_string().contains("path gradients must be finite"));
    }

    #[test]
    fn rejects_invalid_atomic_numbers() {
        let path = std::env::temp_dir().join(format!(
            "landfold-hdf5-invalid-atomic-number-{}.h5",
            std::process::id()
        ));
        let file = hdf5::File::create(&path).expect("create HDF5 fixture");
        let path_group = file.create_group("path").expect("create path group");
        path_group
            .new_dataset::<f64>()
            .shape((1, 3))
            .create("images")
            .expect("create images")
            .write_raw(&[0.0, 1.0, 2.0])
            .expect("write images");
        let metadata_group = file
            .create_group("metadata")
            .expect("create metadata group");
        metadata_group
            .new_dataset::<i64>()
            .shape(1)
            .create("atomic_numbers")
            .expect("create atomic numbers")
            .write_raw(&[0_i64])
            .expect("write atomic numbers");
        drop(file);

        let error = read_hdf5_batch(&path).expect_err("invalid atomic number must be rejected");
        std::fs::remove_file(path).expect("remove HDF5 fixture");
        assert!(error.to_string().contains("atomic numbers"));
    }
}
