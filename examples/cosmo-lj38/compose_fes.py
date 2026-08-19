#!/usr/bin/env python3
"""LJ38 teaching figure: FES from landfold plus xyzrender frames and CN bars."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"

CMAP = LinearSegmentedColormap.from_list(
    "fes",
    ["#e65014", "#5a145a", "#281e8c", "#326ed2", "#dcebfa"],
)


def load_fes(path: Path):
    xs, ys, fs = [], [], []
    with path.open() as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            row = raw.split()
            if len(row) < 3 or row[2] == "nan":
                continue
            xs.append(float(row[0]))
            ys.append(float(row[1]))
            fs.append(float(row[2]))
    x = np.asarray(xs)
    y = np.asarray(ys)
    f = np.asarray(fs)
    ux = np.unique(np.round(x, 8))
    uy = np.unique(np.round(y, 8))
    grid = np.full((len(uy), len(ux)), np.nan)
    ix = {v: i for i, v in enumerate(ux)}
    iy = {v: i for i, v in enumerate(uy)}
    for xi, yi, fi in zip(x, y, f):
        grid[iy[round(yi, 8)], ix[round(xi, 8)]] = fi
    return ux, uy, grid


def load_xy(path: Path):
    pts = np.loadtxt(path)
    return np.atleast_2d(pts)[:, :2]


def cn_hist(xyz: Path, cutoff: float = 1.5):
    lines = xyz.read_text().splitlines()
    n = int(lines[0].strip())
    pos = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        pos.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
    p = np.asarray(pos)
    cn = np.zeros(n, dtype=int)
    for i in range(n):
        d = np.linalg.norm(p - p[i], axis=1)
        cn[i] = int(np.sum((d > 1e-8) & (d < cutoff)))
    bins = np.arange(3, 14)
    counts, _ = np.histogram(cn, bins=bins)
    return bins[:-1], counts


def main():
    fes = OUT / "fes.csv"
    if not fes.exists():
        raise SystemExit("run ./run.sh first")
    ux, uy, grid = load_fes(fes)

    fig = plt.figure(figsize=(11.2, 8.4))
    gs = GridSpec(3, 4, height_ratios=[3.2, 1.6, 1.1], hspace=0.35, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    mesh = ax.contourf(ux, uy, grid, levels=18, cmap=CMAP)
    ax.contour(ux, uy, grid, levels=10, colors="k", linewidths=0.25, alpha=0.4)
    if (OUT / "ts.proj").exists():
        xy = load_xy(OUT / "ts.proj")
        ax.scatter(xy[:, 0], xy[:, 1], s=3, c="0.85", alpha=0.25, linewidths=0)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(r"$F/\epsilon$")
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    ax.set_title(
        r"LJ$_{38}$ TSE map from COSMO teaching ts.all  "
        r"($kT^*=0.168$, $\sigma,a,b=5,8,1$ / $5,2,2$)"
    )

    panels = [
        ("lj38_fcc.png", "lj38_fcc.xyz", "fcc GM"),
        ("lj38_ico.png", "lj38_ico.xyz", "ico"),
        ("lj38.17.png", "lj38.17.xyz", "meta n6~17"),
        ("lj38.19.png", "lj38.19.xyz", "meta n6~19"),
    ]
    for i, (png_name, xyz_name, lab) in enumerate(panels):
        axm = fig.add_subplot(gs[1, i])
        axh = fig.add_subplot(gs[2, i])
        png = OUT / png_name
        if png.exists():
            axm.imshow(Image.open(png))
        axm.set_axis_off()
        axm.set_title(lab, fontsize=9)
        xyz = ROOT / xyz_name
        if xyz.exists():
            k, c = cn_hist(xyz)
            axh.bar(k, c, color="#1b4f72")
            axh.set_xticks(k)
        axh.set_xlabel("CN")
        if i == 0:
            axh.set_ylabel("atoms")

    fig.savefig(OUT / "lj38_fes.png", dpi=150)
    fig.savefig(OUT / "lj38_fes.svg")
    print("wrote", OUT / "lj38_fes.png")


if __name__ == "__main__":
    main()
