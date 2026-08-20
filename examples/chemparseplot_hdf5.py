"""Embed a ChemGP NEB HDF5 path with landfold.

The HDF5 reader remains owned by chemparseplot. Landfold receives one
flattened coordinate vector per image and returns a versioned result mapping.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from chemparseplot.parse.trajectory.hdf5 import load_neb_result

import landfold


def _python_metadata(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, Mapping):
        return {
            str(_python_metadata(key)): _python_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, np.ndarray):
        return _python_metadata(value.tolist())
    if isinstance(value, np.generic):
        return _python_metadata(value.item())
    if isinstance(value, (list, tuple)):
        return [_python_metadata(item) for item in value]
    return value


def _path_observables(result: object, n_images: int) -> dict[str, list[float]]:
    path = result.path
    observables: dict[str, list[float]] = {}
    for name in ("energies", "f_para", "rxn_coord"):
        values = np.asarray(getattr(path, name), dtype=np.float64)
        if values.ndim != 1 or values.shape[0] != n_images:
            raise ValueError(f"ChemGP path {name} must have one value per image")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"ChemGP path {name} must be finite")
        observables[name] = values.tolist()
    return observables


def _path_gradients(result: object, n_images: int, width: int) -> list[list[float]]:
    gradients = np.asarray(result.path.gradients, dtype=np.float64)
    if gradients.shape != (n_images, width):
        raise ValueError("ChemGP path gradients must match image shape")
    if not np.all(np.isfinite(gradients)):
        raise ValueError("ChemGP path gradients must be finite")
    return gradients.tolist()


def embed_hdf5(path: Path, *, lowdim: int = 2) -> dict:
    result = load_neb_result(str(path))
    images = np.asarray(result.path.images, dtype=np.float64)
    if images.ndim != 2 or images.shape[1] == 0 or images.shape[1] % 3:
        raise ValueError("ChemGP HDF5 path images must have shape (n_images, 3*n_atoms)")

    source_metadata = {
        key: _python_metadata(value)
        for key, value in result.get("metadata", {}).items()
    }
    frame_ids = list(range(images.shape[0]))
    atom_ids = list(range(images.shape[1] // 3))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {
        "schema": "landfold.provenance.v1",
        "run_id": f"chemparseplot:{digest[:16]}",
        "input_digest": f"sha256:{digest}",
        "engine_id": "chemparseplot",
        "protocol_family": "chemparseplot.trajectory",
        "protocol_major": 1,
        "protocol_minor": 0,
        "abi_layout_revision": 1,
        "dlpack_major": 1,
        "dlpack_minor": 0,
    }
    eindir_revision = getattr(landfold, "eindir_revision", None)
    if isinstance(eindir_revision, str) and len(eindir_revision) == 40:
        provenance["eindir_revision"] = eindir_revision
    metadata = {
        "source_format": "ChemGP HDF5 NEB",
        "source_path": str(path),
        "frame_indices": frame_ids,
        "frame_ids": frame_ids,
        "atom_ids": atom_ids,
        "length_unit": source_metadata.get("length_unit"),
        "n_atoms": images.shape[1] // 3,
        "source_metadata": source_metadata,
        "path_observables": _path_observables(result, images.shape[0]),
        "path_gradients": _path_gradients(result, images.shape[0], images.shape[1]),
        "provenance": provenance,
    }
    return landfold.embed_euclid_result(
        images,
        lowdim=lowdim,
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("--lowdim", type=int, default=2)
    args = parser.parse_args()
    result = embed_hdf5(args.hdf5, lowdim=args.lowdim)
    coordinates = np.asarray(result["coordinates"])
    print(f"schema={result['schema']}")
    print(f"stress={result['stress']:.8g}")
    np.savetxt("landfold-embedding.dat", coordinates)


if __name__ == "__main__":
    main()
