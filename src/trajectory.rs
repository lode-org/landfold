//! Validated, format-neutral batches of coordinate frames.

use ndarray::{Array2, ArrayView2};
use std::collections::HashSet;

use crate::{LandfoldError, Result};

/// Coordinate frames and the identity needed to interpret them consistently.
///
/// Format adapters populate this type; embedding and analysis code consumes the
/// ordered coordinate frames without depending on a file format.
#[derive(Clone, Debug)]
pub struct FrameBatch {
    pub frames: Vec<Array2<f64>>,
    pub atom_ids: Vec<u64>,
    pub frame_ids: Vec<u64>,
    pub length_unit: Option<String>,
}

impl FrameBatch {
    /// Build a batch and validate frame shape, finiteness, and atom identity.
    pub fn new(
        frames: Vec<Array2<f64>>,
        atom_ids: Vec<u64>,
        frame_ids: Vec<u64>,
        length_unit: Option<String>,
    ) -> Result<Self> {
        if length_unit.as_deref().is_some_and(str::is_empty) {
            return Err(LandfoldError::Msg(
                "trajectory length unit must not be empty".into(),
            ));
        }
        if frames.is_empty() {
            if !atom_ids.is_empty() || !frame_ids.is_empty() {
                return Err(LandfoldError::Shape(
                    "empty trajectory batches cannot have IDs",
                ));
            }
            return Ok(Self {
                frames,
                atom_ids,
                frame_ids,
                length_unit,
            });
        }
        let shape = frames[0].dim();
        if shape.1 != 3 || shape.0 != atom_ids.len() {
            return Err(LandfoldError::Shape(
                "trajectory frames must be (n_atoms, 3) and match atom IDs",
            ));
        }
        if frame_ids.len() != frames.len() {
            return Err(LandfoldError::Shape("trajectory frame ID length"));
        }
        if frames
            .iter()
            .any(|frame| frame.dim() != shape || frame.iter().any(|&value| !value.is_finite()))
        {
            return Err(LandfoldError::Msg(
                "trajectory frames must have a stable shape and finite coordinates".into(),
            ));
        }
        let unique_ids: HashSet<u64> = atom_ids.iter().copied().collect();
        if unique_ids.len() != atom_ids.len() {
            return Err(LandfoldError::Msg(
                "trajectory atom IDs must be unique".into(),
            ));
        }
        Ok(Self {
            frames,
            atom_ids,
            frame_ids,
            length_unit,
        })
    }

    /// Construct a batch with implicit atom and frame IDs.
    pub fn from_positions(frames: Vec<Array2<f64>>) -> Result<Self> {
        let n_atoms = frames.first().map_or(0, Array2::nrows);
        let frame_ids = (0..frames.len() as u64).collect();
        Self::new(frames, (0..n_atoms as u64).collect(), frame_ids, None)
    }

    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }

    pub fn n_atoms(&self) -> usize {
        self.atom_ids.len()
    }

    pub fn frame(&self, index: usize) -> Option<ArrayView2<'_, f64>> {
        self.frames.get(index).map(|frame| frame.view())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn validates_shape_and_identity_once_for_a_batch() {
        let batch = FrameBatch::new(
            vec![array![[0.0, 0.0, 0.0]], array![[1.0, 0.0, 0.0]]],
            vec![7],
            vec![11, 12],
            Some("angstrom".into()),
        )
        .unwrap();
        assert_eq!(batch.n_frames(), 2);
        assert_eq!(batch.n_atoms(), 1);
        assert_eq!(batch.frame_ids, vec![11, 12]);
        assert_eq!(batch.length_unit.as_deref(), Some("angstrom"));
    }

    #[test]
    fn rejects_mismatched_frame_shape_and_ids() {
        assert!(FrameBatch::new(Vec::new(), vec![0], Vec::new(), None).is_err());
        assert!(
            FrameBatch::new(
                vec![array![[0.0, 0.0, 0.0]], array![[1.0, 0.0]]],
                vec![0],
                vec![0, 1],
                None,
            )
            .is_err()
        );
        assert!(
            FrameBatch::new(vec![array![[0.0, 0.0, 0.0]]], vec![0, 1], vec![0], None,).is_err()
        );
    }
}
