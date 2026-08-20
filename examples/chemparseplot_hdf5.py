"""Embed a ChemGP NEB HDF5 path with landfold.

The HDF5 reader remains owned by chemparseplot. Landfold receives one
flattened coordinate vector per image and returns a versioned result mapping.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from chemparseplot.parse.trajectory.hdf5 import load_neb_result

import landfold


def _python_metadata(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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
    metadata = {
        "source_format": "ChemGP HDF5 NEB",
        "source_path": str(path),
        "frame_indices": frame_ids,
        "frame_ids": frame_ids,
        "atom_ids": atom_ids,
        "length_unit": source_metadata.get("length_unit"),
        "n_atoms": images.shape[1] // 3,
        "source_metadata": source_metadata,
        "provenance": {
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
        },
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
