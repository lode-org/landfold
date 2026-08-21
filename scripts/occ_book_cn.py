#!/usr/bin/env python3
"""n4..n13 coordination histograms from an energy-leading .min dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def switching(d: np.ndarray, r0: float = 1.5, r1: float = 1.25) -> np.ndarray:
    out = np.zeros_like(d)
    out[d < r1] = 1.0
    mid = (d > r1) & (d < r0)
    y = (d[mid] - r1) / (r0 - r1)
    out[mid] = (y - 1.0) ** 2 * (2.0 * y + 1.0)
    return out


def nc_vector(pos: np.ndarray, sig: float = 0.5) -> np.ndarray:
    n = pos.shape[0]
    ci = np.zeros(n)
    for i in range(n):
        d = np.linalg.norm(pos - pos[i], axis=1)
        d[i] = 1e9
        ci[i] = float(switching(d).sum())
    vec = np.zeros(10)
    for k, c in enumerate(range(4, 14)):
        vec[k] = float(np.exp(-0.5 * ((ci - c) / sig) ** 2).sum())
    return vec


def load_min(path: Path, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    energies = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        nums = [float(x) for x in line.split()]
        energy, coords = nums[0], nums[1:]
        if len(coords) != n_atoms * 3:
            raise SystemExit(f"{path}: expected {n_atoms * 3} coords, got {len(coords)}")
        energies.append(energy)
        rows.append(np.asarray(coords).reshape(n_atoms, 3))
    return np.asarray(energies), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("minfile")
    ap.add_argument("-n", "--n-atoms", type=int, required=True)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    src = Path(args.minfile)
    energies, frames = load_min(src, args.n_atoms)
    desc = np.vstack([nc_vector(p) for p in frames])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, desc, fmt="%.8e")
    np.savetxt(out.with_suffix(".energy"), energies, fmt="%.10e")
    meta = {
        "n": int(desc.shape[0]),
        "n_atoms": args.n_atoms,
        "e_min": float(energies.min()) if len(energies) else None,
        "e_max": float(energies.max()) if len(energies) else None,
        "cv": str(out),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
