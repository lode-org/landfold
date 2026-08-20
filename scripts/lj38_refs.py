#!/usr/bin/env python3
"""n4..n13 of the fcc and ico xyz (course coordination counts)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "examples" / "cosmo-lj38"


def load_xyz(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    pos = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        pos.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
    return np.asarray(pos)


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


def main() -> None:
    fcc = nc_vector(load_xyz(ROOT / "lj38_fcc.xyz"))
    ico = nc_vector(load_xyz(ROOT / "lj38_ico.xyz"))
    out = ROOT / "out" / "lj38_refs.cv"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, np.vstack([fcc, ico]), fmt="%.8e")
    print("fcc", fcc)
    print("ico", ico)
    print("wrote", out)


if __name__ == "__main__":
    main()
