"""Smoke-test the ChemGP HDF5 to Landfold result bridge."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
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
    assert metadata["path_observables"] == {
        "energies": [0.0, 1.0, 0.5],
        "f_para": [0.0, 0.0, 0.0],
        "rxn_coord": [0.0, 1.0, 2.0],
    }
    assert metadata["path_gradients"] == np.zeros((3, 6)).tolist()
    assert metadata["provenance"]["schema"] == "landfold.provenance.v1"
    assert metadata["provenance"]["input_digest"].startswith("sha256:")
    assert len(metadata["provenance"]["eindir_revision"]) == 40


def test_python_metadata_normalizes_nested_numpy_values() -> None:
    value = {
        "unit": b"angstrom",
        "scalar": np.float64(1.5),
        "array": np.array([np.int64(2), b"ok"], dtype=object),
        "nested": {"flag": np.bool_(True)},
        "tuple": (np.float64(3.0),),
    }

    assert BRIDGE._python_metadata(value) == {
        "unit": "angstrom",
        "scalar": 1.5,
        "array": [2, "ok"],
        "nested": {"flag": True},
        "tuple": [3.0],
    }


@pytest.mark.parametrize(
    "values, message",
    [
        (np.array([0.0]), "one value per image"),
        (np.array([0.0, np.nan]), "must be finite"),
    ],
)
def test_path_observables_are_finite_and_image_aligned(
    values: np.ndarray, message: str
) -> None:
    result = SimpleNamespace(
        path=SimpleNamespace(energies=values, f_para=np.zeros(2), rxn_coord=np.zeros(2))
    )

    with pytest.raises(ValueError, match=message):
        BRIDGE._path_observables(result, 2)


@pytest.mark.parametrize(
    "gradients, message",
    [
        (np.zeros((1, 6)), "match image shape"),
        (np.full((2, 6), np.nan), "must be finite"),
    ],
)
def test_path_gradients_are_finite_and_shape_aligned(
    gradients: np.ndarray, message: str
) -> None:
    result = SimpleNamespace(path=SimpleNamespace(gradients=gradients))

    with pytest.raises(ValueError, match=message):
        BRIDGE._path_gradients(result, 2, 6)
