#!/usr/bin/env python3
"""Occupancy-book native Torgerson map plus every landfold method."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_xy(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        rows.append([float(parts[0]), float(parts[1])])
    return np.asarray(rows)


def load_book(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data


def scatter(ax, xy, c, title, cmap="viridis"):
    if c is None:
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=10, c="#2255aa", linewidths=0, alpha=0.85)
    else:
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=10, c=c, cmap=cmap, linewidths=0, alpha=0.85)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--energy")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--label", default="occ")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    book = load_book(Path(args.book))
    energy = None
    if args.energy and Path(args.energy).is_file():
        energy = np.loadtxt(args.energy)
    work = Path(args.dir)
    label = args.label

    methods = [
        ("ceriotti.proj", "Ceriotti chi"),
        ("asinh.proj", "asinh"),
        ("phate.proj", "PHATE"),
        ("pacmap.proj", "PaCMAP"),
        ("nearfar.proj", "near-far"),
        ("bands.proj", "three-band"),
        ("gapsplit.proj", "gap-split"),
        ("axis.ld", "fcc-ico axis"),
    ]

    panels = [("native", None)]
    for suffix, title in methods:
        p = work / f"{label}_{suffix}"
        if not (p.is_file() and p.stat().st_size > 0) and suffix.endswith(".proj"):
            p = work / f"{label}_{suffix.replace('.proj', '.ld')}"
        if p.is_file() and p.stat().st_size > 0:
            panels.append((title, p))

    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.atleast_2d(axes)
    for i, (title, path) in enumerate(panels):
        ax = axes[i // ncols, i % ncols]
        if path is None:
            xy = np.column_stack([book["x"], book["y"]])
            c = book["wells"]
            scatter(ax, xy, c, f"native Torgerson (wells={int(book['wells'].sum())})", cmap="magma")
        else:
            xy = load_xy(path)
            c = None
            if energy is not None and len(energy) == len(xy):
                c = energy
            scatter(ax, xy, c, f"{title} n={len(xy)}")
    for j in range(len(panels), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle(f"Elja occupancy book {label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
