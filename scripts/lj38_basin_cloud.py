#!/usr/bin/env python3
"""Rattle fcc and ico into local HD clouds (course n4..n13).

ts.all is a TSE. The two minima are not in it. This writes HD rows
that sit on the two crystals so an occupancy map can show basins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lj38_refs import ROOT, load_xyz, nc_vector


def rattle(pos: np.ndarray, n: int, amp: float, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((n, 10))
    for i in range(n):
        out[i] = nc_vector(pos + rng.normal(0.0, amp, size=pos.shape))
    return out


def main() -> None:
    rng = np.random.default_rng(1)
    fcc = load_xyz(ROOT / "lj38_fcc.xyz")
    ico = load_xyz(ROOT / "lj38_ico.xyz")
    n = 200
    amp = 0.04
    cloud = np.vstack(
        [
            rattle(fcc, n, amp, rng),
            rattle(ico, n, amp, rng),
            nc_vector(fcc)[None, :],
            nc_vector(ico)[None, :],
        ]
    )
    dest = ROOT / "out" / "lj38_basins.cv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(dest, cloud, fmt="%.8e")
    print("wrote", dest, "rows", cloud.shape[0], "amp", amp)


if __name__ == "__main__":
    main()
