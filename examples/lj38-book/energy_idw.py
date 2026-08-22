#!/usr/bin/env python3
"""Compact-support IDW of E-E_GM on a landfold plane.

The field is quenched energy, not occupancy invert. Kernel and support
cut come from scripts/occ_book_idw_fill.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import occ_book_idw_fill as idw


def load_xy(path: Path) -> np.ndarray:
    xy = np.loadtxt(path)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise SystemExit(f"{path}: expected (*, >=2), got {xy.shape}")
    return np.asarray(xy[:, :2], dtype=float)


def load_energy(path: Path) -> np.ndarray:
    energy = np.loadtxt(path)
    return np.asarray(energy, dtype=float).reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xy", required=True, type=Path, help="landfold .proj / .ld")
    ap.add_argument("--energy", required=True, type=Path, help="one energy per row")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gm-idx", type=int, default=None)
    ap.add_argument("--ico-idx", type=int, default=None)
    ap.add_argument("--title", default=r"energy IDW  $E-E_{\mathrm{GM}}$")
    args = ap.parse_args()

    xy = load_xy(args.xy)
    energy = load_energy(args.energy)
    if energy.shape[0] != xy.shape[0]:
        raise SystemExit(f"energy {energy.shape} vs xy {xy.shape}")
    gm = int(np.argmin(energy)) if args.gm_idx is None else args.gm_idx
    z = energy - float(energy[gm])
    k = min(6, xy.shape[0])
    gx, gy, field = idw.fill(xy, z, k=k)

    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor="white")
    mesh = ax.pcolormesh(
        gx, gy, np.clip(field, 0, 5.0), cmap=idw.PES, shading="auto", vmin=0, vmax=5.0
    )
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=np.clip(z, 0, 5.0),
        s=7,
        cmap=idw.PES,
        vmin=0,
        vmax=5.0,
        edgecolors="none",
        alpha=0.55,
        zorder=20,
    )
    ax.scatter(
        xy[gm, 0],
        xy[gm, 1],
        s=170,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.7,
        zorder=50,
        label=rf"GM ${float(energy[gm]):.3f}$",
    )
    if args.ico_idx is not None:
        ico = args.ico_idx
        ax.scatter(
            xy[ico, 0],
            xy[ico, 1],
            s=95,
            marker="D",
            c="k",
            edgecolors="white",
            linewidths=0.7,
            zorder=50,
            label=rf"ico ${float(energy[ico]):.3f}$",
        )
    ax.legend(loc="best", fontsize=8, frameon=True, fancybox=False)
    ax.set_xlabel(r"$\chi_1$")
    ax.set_ylabel(r"$\chi_2$")
    ax.set_title(args.title)
    fig.colorbar(mesh, ax=ax, label=r"$E-E_\mathrm{GM}$")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, facecolor="white")
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
