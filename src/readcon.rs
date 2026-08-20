//! Adapters from readcon CON frames to landfold point arrays.
//!
//! The readcon feature keeps the complete [`ConFrame`] available to callers.
//! Coordinate conversion is explicit, so headers, units, atom IDs, and
//! optional sections are not silently discarded at the ingestion boundary.

use std::path::Path;

use ndarray::Array2;
use readcon_core::types::ConFrame;

use crate::trajectory::FrameBatch;
use crate::{LandfoldError, Result};

/// Read all frames from a canonical readcon CON or CONVEL file.
pub fn read_con_frames(path: &Path) -> Result<Vec<ConFrame>> {
    readcon_core::iterators::read_all_frames(path)
        .map_err(|error| LandfoldError::Parse(error.to_string()))
}

/// Convert one readcon frame to an `(n_atoms, 3)` array in frame atom order.
pub fn frame_positions(frame: &ConFrame) -> Result<Array2<f64>> {
    let n_atoms = frame.positions.nrows();
    if frame.atom_ids.len() != n_atoms {
        return Err(LandfoldError::Shape(
            "readcon positions and atom IDs have different lengths",
        ));
    }
    let mut values = Vec::with_capacity(n_atoms * 3);
    for atom in 0..n_atoms {
        let position = frame.positions.as_f64_row(atom);
        if !position.iter().all(|value| value.is_finite()) {
            return Err(LandfoldError::Msg(format!(
                "readcon frame contains non-finite coordinates at atom {atom}"
            )));
        }
        values.extend_from_slice(&position);
    }
    Array2::from_shape_vec((n_atoms, 3), values)
        .map_err(|_| LandfoldError::Shape("readcon coordinates must have three columns"))
}

/// Convert frames while requiring stable atom identity and atom count.
pub fn frames_positions(frames: &[ConFrame]) -> Result<Vec<Array2<f64>>> {
    let Some(first) = frames.first() else {
        return Ok(Vec::new());
    };
    let expected_ids = first
        .atom_ids
        .as_slice()
        .ok_or(LandfoldError::Shape("readcon atom IDs must be contiguous"))?
        .to_vec();
    let expected_count = expected_ids.len();
    let mut positions = Vec::with_capacity(frames.len());
    for (frame_index, frame) in frames.iter().enumerate() {
        if frame.atom_ids.len() != expected_count {
            return Err(LandfoldError::Shape(
                "readcon frames have different atom counts",
            ));
        }
        let atom_ids = frame
            .atom_ids
            .as_slice()
            .ok_or(LandfoldError::Shape("readcon atom IDs must be contiguous"))?;
        if atom_ids != expected_ids.as_slice() {
            return Err(LandfoldError::Msg(format!(
                "readcon frame {frame_index} has different atom IDs"
            )));
        }
        positions.push(frame_positions(frame)?);
    }
    Ok(positions)
}

/// Read a CON/CONVEL file and return validated coordinate arrays.
pub fn read_con_positions(path: &Path) -> Result<Vec<Array2<f64>>> {
    let frames = read_con_frames(path)?;
    frames_positions(&frames)
}

/// Read a CON/CONVEL file into the format-neutral trajectory contract.
pub fn read_con_batch(path: &Path) -> Result<FrameBatch> {
    let frames = read_con_frames(path)?;
    frames_batch(&frames)
}

/// Convert readcon frames into a validated, format-neutral trajectory batch.
pub fn frames_batch(frames: &[ConFrame]) -> Result<FrameBatch> {
    let positions = frames_positions(frames)?;
    let Some(first) = frames.first() else {
        return FrameBatch::new(Vec::new(), Vec::new(), Vec::new(), None);
    };
    let atom_ids = first
        .atom_ids
        .as_slice()
        .ok_or(LandfoldError::Shape("readcon atom IDs must be contiguous"))?
        .to_vec();
    if first.atom_data.len() != atom_ids.len() {
        return Err(LandfoldError::Shape("readcon atom symbols and IDs"));
    }
    let atom_symbols = first
        .atom_data
        .iter()
        .map(|atom| atom.symbol.to_string())
        .collect::<Vec<_>>();
    for (frame_index, frame) in frames.iter().enumerate() {
        let symbols = frame
            .atom_data
            .iter()
            .map(|atom| atom.symbol.to_string())
            .collect::<Vec<_>>();
        if symbols != atom_symbols {
            return Err(LandfoldError::Msg(format!(
                "readcon frame {frame_index} has different atom symbols"
            )));
        }
    }
    let explicit_frame_ids = frames
        .iter()
        .map(|frame| frame.header.frame_index())
        .collect::<Vec<_>>();
    let has_explicit_frame_ids = explicit_frame_ids.iter().any(Option::is_some);
    if has_explicit_frame_ids && explicit_frame_ids.iter().any(Option::is_none) {
        return Err(LandfoldError::Msg(
            "readcon frames mix explicit and implicit frame IDs".into(),
        ));
    }
    let frame_ids = if has_explicit_frame_ids {
        explicit_frame_ids.into_iter().flatten().collect()
    } else {
        (0..frames.len() as u64).collect()
    };
    let length_unit = first.header.length_unit().map(str::to_owned);
    if frames
        .iter()
        .skip(1)
        .map(|frame| frame.header.length_unit())
        .any(|unit| unit != length_unit.as_deref())
    {
        return Err(LandfoldError::Msg(
            "readcon frames have inconsistent length units".into(),
        ));
    }
    let metadata = frames
        .iter()
        .map(|frame| frame.header.metadata.clone())
        .collect();
    let mut batch =
        FrameBatch::new_with_metadata(positions, atom_ids, frame_ids, length_unit, metadata)?;
    batch.atom_symbols = Some(atom_symbols);
    batch.validate()?;
    Ok(batch)
}

/// Read a chemfiles-supported trajectory through readcon's canonical frame
/// conversion layer.
#[cfg(feature = "readcon-chemfiles")]
pub fn read_trajectory_frames(path: &Path) -> Result<Vec<ConFrame>> {
    readcon_core::chemfiles_import::con_frames_from_trajectory_path(path)
        .map_err(|error| LandfoldError::Parse(error.to_string()))
}

/// Read a chemfiles-supported trajectory and return validated coordinates.
#[cfg(feature = "readcon-chemfiles")]
pub fn read_trajectory_positions(path: &Path) -> Result<Vec<Array2<f64>>> {
    let frames = read_trajectory_frames(path)?;
    frames_positions(&frames)
}

/// Read a Chemfiles-supported trajectory into the format-neutral batch.
#[cfg(feature = "readcon-chemfiles")]
pub fn read_trajectory_batch(path: &Path) -> Result<FrameBatch> {
    let frames = read_trajectory_frames(path)?;
    frames_batch(&frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use readcon_core::types::ConFrameBuilder;

    fn frame(x: f64) -> ConFrame {
        let mut builder = ConFrameBuilder::new([10.0; 3], [90.0; 3]);
        builder.add_atom("H", x, 1.0, 2.0, [false; 3], 0, 1.0);
        builder.add_atom("H", x + 1.0, 1.0, 2.0, [false; 3], 1, 1.0);
        builder.build()
    }

    #[test]
    fn converts_positions_and_preserves_frame_order() {
        let frames = vec![frame(0.0), frame(2.0)];
        let positions = frames_positions(&frames).expect("valid frame positions");
        assert_eq!(positions[0][[0, 0]], 0.0);
        assert_eq!(positions[1][[0, 0]], 2.0);
    }

    #[test]
    fn rejects_changed_atom_identity() {
        let first = frame(0.0);
        let mut second = frame(1.0);
        second.atom_ids[0] += 1;
        let error = frames_positions(&[first, second]).expect_err("identity must be stable");
        assert!(error.to_string().contains("different atom IDs"));
    }

    #[test]
    fn rejects_changed_atom_symbols_in_a_batch() {
        let first = frame(0.0);
        let mut second = frame(1.0);
        second.atom_data[0].symbol = "O".into();
        let error = frames_batch(&[first, second]).expect_err("symbols must be stable");
        assert!(error.to_string().contains("different atom symbols"));
    }

    #[test]
    fn reads_positions_from_a_con_file() {
        use readcon_core::writer::ConFrameWriter;

        let path =
            std::env::temp_dir().join(format!("landfold-readcon-{}.con", std::process::id()));
        let mut writer = ConFrameWriter::from_path(&path).expect("create CON fixture");
        writer.write_frame(&frame(3.0)).expect("write CON fixture");
        drop(writer);

        let positions = read_con_positions(&path).expect("read CON fixture");
        let batch = read_con_batch(&path).expect("read CON batch");
        std::fs::remove_file(path).expect("remove CON fixture");
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0][[0, 0]], 3.0);
        assert_eq!(positions[0][[1, 0]], 4.0);
        assert_eq!(batch.frame_ids, vec![0]);
        assert_eq!(batch.atom_ids, vec![0, 1]);
        assert_eq!(
            batch
                .atom_symbols
                .as_ref()
                .map(|symbols| symbols.iter().map(String::as_str).collect::<Vec<_>>()),
            Some(vec!["H", "H"])
        );
    }

    #[test]
    fn converts_frames_into_the_format_neutral_batch() {
        let mut first = frame(0.0);
        first.header.set_frame_index(41);
        let mut second = frame(2.0);
        second.header.set_frame_index(42);
        let batch = frames_batch(&[first, second]).expect("convert CON frames");
        assert_eq!(batch.frame_ids, vec![41, 42]);
        assert_eq!(batch.atom_ids, vec![0, 1]);
        assert_eq!(batch.n_frames(), 2);
        assert_eq!(batch.length_unit.as_deref(), Some("angstrom"));
        assert_eq!(batch.metadata.len(), 2);
    }

    #[test]
    fn preserves_json_metadata_in_the_format_neutral_batch() {
        use std::collections::BTreeMap;

        let mut first = frame(0.0);
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "units".into(),
            serde_json::json!({"length": "angstrom"}),
        );
        metadata.insert("energy".into(), serde_json::json!(-1.25));
        metadata.insert("generator".into(), serde_json::json!("eon"));
        first.header.metadata = metadata;

        let batch = frames_batch(&[first, frame(2.0)]).expect("convert CON frames");
        assert_eq!(
            batch.frame_metadata(0).and_then(|m| m.get("energy")),
            Some(&serde_json::json!(-1.25))
        );
        assert_eq!(
            batch.frame_metadata(0).and_then(|m| m.get("generator")),
            Some(&serde_json::json!("eon"))
        );
        assert!(batch.frame_metadata(1).is_some());
    }

    #[test]
    fn rejects_mixed_explicit_and_implicit_frame_ids() {
        let mut first = frame(0.0);
        first.header.set_frame_index(41);
        let error =
            frames_batch(&[first, frame(2.0)]).expect_err("mixed frame identity must be rejected");
        assert!(error.to_string().contains("mix explicit and implicit"));
    }

    #[cfg(feature = "readcon-chemfiles")]
    #[test]
    fn reads_positions_from_a_chemfiles_xyz_trajectory() {
        let path = std::env::temp_dir().join(format!(
            "landfold-readcon-chemfiles-{}.xyz",
            std::process::id()
        ));
        let xyz = "2\nframe 0\nH 0 0 0\nH 1 0 0\n2\nframe 1\nH 2 0 0\nH 3 0 0\n";
        std::fs::write(&path, xyz).expect("write XYZ fixture");

        let positions = read_trajectory_positions(&path).expect("read XYZ fixture");
        let batch = read_trajectory_batch(&path).expect("read XYZ batch");
        std::fs::remove_file(path).expect("remove XYZ fixture");
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0][[0, 0]], 0.0);
        assert_eq!(positions[1][[0, 0]], 2.0);
        assert_eq!(batch.frame_ids, vec![0, 1]);
        assert_eq!(batch.atom_ids, vec![0, 1]);
        assert_eq!(batch.length_unit.as_deref(), Some("angstrom"));
    }
}
