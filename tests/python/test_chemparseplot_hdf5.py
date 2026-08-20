"""Smoke-test the ChemGP HDF5 to Landfold result bridge."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("chemparseplot")
pytest.importorskip("landfold")

EXAMPLE = Path(__file__).parents[2] / "examples" / "chemparseplot_hdf5.py"
SPEC = importlib.util.spec_from_file_location("chemparseplot_hdf5", EXAMPLE)
assert SPEC is not None and SPEC.loader is not None
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


def test_chemgp_hdf5_path_preserves_shape_and_provenance(tmp_path: Path) -> None:
    path = tmp_path / "neb_result.h5"
    images = np.arange(18, dtype=np.float64).reshape(3, 6)
    with h5py.File(path, "w") as handle:
        path_group = handle.create_group("path")
        path_group.create_dataset("images", data=images)
        path_group.create_dataset("energies", data=np.array([0.0, 1.0, 0.5]))
        path_group.create_dataset("gradients", data=np.zeros_like(images))
        path_group.create_dataset("f_para", data=np.zeros(3))
        path_group.create_dataset("rxn_coord", data=np.arange(3, dtype=np.float64))
        metadata_group = handle.create_group("metadata")
        metadata_group.create_dataset("atomic_numbers", data=np.array([1, 8]))
        metadata_group.create_dataset("length_unit", data=np.bytes_("angstrom"))

    result = BRIDGE.embed_hdf5(path)

    assert result["schema"] == "landfold.embedding.v1"
    assert np.asarray(result["coordinates"]).shape == (3, 2)
    assert result["n_points"] == 3
    assert result["highdim"] == 6
    metadata = result["metadata"]
    assert metadata["frame_indices"] == [0, 1, 2]
    assert metadata["atom_ids"] == [0, 1]
    assert metadata["length_unit"] == "angstrom"
    assert metadata["n_atoms"] == 2
    assert metadata["source_metadata"]["atomic_numbers"] == [1, 8]
    assert metadata["provenance"]["schema"] == "landfold.provenance.v1"
    assert metadata["provenance"]["input_digest"].startswith("sha256:")
    assert len(metadata["provenance"]["eindir_revision"]) == 40
