#!/usr/bin/env python3
"""Regenerate DECAF golden fixtures with Lai's reference package."""

import argparse
from pathlib import Path
import subprocess
import tomllib

import decaf
import numpy as np
from ase import io as aseio


REFERENCE_COMMIT = "dd8b2c5bdf224cc583855133d4c63a985763294f"
REFERENCE_VERSION = "1.0.6"
PD_STRUCTURE = "Pd_211.con"
PD_NEAREST_NEIGHBOUR = 2.75
PD_SOAP = {
    "SOAP_L_nmax": 4,
    "SOAP_L_lmax": 3,
    "SOAP_S_nmax": 8,
    "SOAP_S_lmax": 4,
    "SOAP_S_cut": np.sqrt(1 / 2) + 0.5,
    "SOAP_L_cut": np.sqrt(3) / 2 + 1,
    "periodic": True,
}
PD_SOAP["SOAP_S_sigma"] = PD_SOAP["SOAP_S_cut"] / 8
PD_SOAP["SOAP_L_sigma"] = PD_SOAP["SOAP_L_cut"] / 8


def labels_from_model(model, n_atoms):
    labels = np.full(n_atoms, -1, dtype=int)
    for class_id, atom_ids in enumerate(model.f_gID[:-1]):
        labels[atom_ids] = class_id
    return labels


def generate_pah(root, output):
    names = ["benzene", "naphthalene", "anthracene", "tetracene", "phenanthren", "graphene"]
    structures = [
        aseio.read(root / "examples" / "Structures" / f"PAH_{name}.con") for name in names
    ]
    structures[-1].pbc = [True, True, False]
    short_cutoff = (1 + np.sqrt(3)) / 2
    long_cutoff = (2 + 3) / 2
    descriptors = np.concatenate(
        [
            decaf.get_SOAP(
                structure,
                1.42,
                species=[6, 1],
                SOAP_S_cut=short_cutoff,
                SOAP_S_sigma=short_cutoff / 8,
                SOAP_L_cut=long_cutoff,
                SOAP_L_sigma=long_cutoff / 8,
                periodic=(True in structure.pbc),
            )
            for structure in structures
        ]
    )
    model = decaf.embed_cluster(
        descriptors,
        embed_str="MDS",
        max_training_size=10000,
        cluster_str="MSC",
        text_out=True,
    )
    model.workflow(
        dim=2,
        bandwidth=0.0241,
        bandwidth_estimate="HaarWavelet",
        reorder_mode_x=False,
    )
    distances = np.linalg.norm(
        descriptors[:, np.newaxis, :] - descriptors[np.newaxis, :, :], axis=2
    )
    np.savetxt(output / "decaf_pah_distances.txt", distances, delimiter="\t", fmt="%.17g")
    np.savetxt(
        output / "decaf_pah_labels.txt",
        labels_from_model(model, descriptors.shape[0]),
        fmt="%d",
    )


def generate_pd211(root, output):
    """Regenerate the fhi-decaf 1.0.6 Pd(211) fixture with its published SOAP."""
    structure = aseio.read(root / "examples" / "Structures" / PD_STRUCTURE)
    structure.pbc = [True, True, False]
    descriptors = decaf.get_SOAP(
        structure,
        PD_NEAREST_NEIGHBOUR,
        **PD_SOAP,
    )
    model = decaf.embed_cluster(
        descriptors,
        embed_str="MDS",
        cluster_str="MSC",
        text_out=True,
    )
    model.workflow(
        dim=0,
        bandwidth_estimate="Gaussian",
        sigma=0.0092,
        reorder_mode_x=True,
    )
    distances = np.linalg.norm(
        descriptors[:, np.newaxis, :] - descriptors[np.newaxis, :, :], axis=2
    )
    np.savetxt(output / "decaf_pd211_distances.txt", distances, delimiter="\t", fmt="%.17g")
    np.savetxt(output / "decaf_pd211_embedded.txt", model.t_embedded, fmt="%.17g")
    np.savetxt(
        output / "decaf_pd211_labels.txt",
        labels_from_model(model, descriptors.shape[0]),
        fmt="%d",
    )


def generate_out_of_sample(output):
    training = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.8, 0.0, 0.6],
            [-1.0, 0.0, 0.0],
            [-0.8, -0.6, 0.0],
            [-0.8, 0.0, -0.6],
            [0.0, 1.0, 0.0],
        ]
    )
    queries = np.array([[0.6, 0.8, 0.0], [0.6, 0.0, 0.8]])
    model = decaf.embed_cluster(training, embed_str="MDS", cluster_str="MSC")
    model.get_cMDS(dim=2)
    projected = model.embed_cMDS(queries)
    distances = np.linalg.norm(
        model.t_embedded[:, np.newaxis, :] - projected[np.newaxis, :, :], axis=2
    )
    np.savetxt(output / "decaf_oos_reference_distances.txt", distances, fmt="%.17g")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    commit = subprocess.check_output(
        ["git", "-C", str(args.reference_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != REFERENCE_COMMIT:
        raise SystemExit(f"expected reference commit {REFERENCE_COMMIT}, got {commit}")
    metadata = tomllib.loads((args.reference_root / "pyproject.toml").read_text())
    source_version = metadata["project"]["version"]
    if source_version != REFERENCE_VERSION:
        raise SystemExit(f"expected fhi-decaf {REFERENCE_VERSION}, got {source_version}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_pah(args.reference_root, args.output_dir)
    generate_pd211(args.reference_root, args.output_dir)
    generate_out_of_sample(args.output_dir)


if __name__ == "__main__":
    main()
