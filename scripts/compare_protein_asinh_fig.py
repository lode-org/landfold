#!/usr/bin/env python3
"""Published Ceriotti protein map vs asinh on the same 1000 landmarks.

Ardevol et al. JCTC 2015 hairpin, 30-D periodic CVs, fun 6,8,8 / 6,2,8.
Published far Spearman(D,d) is ~0.18: this is the system where saturating
F actually loses far order. TSE LJ38 does not (far Pearson 0.95).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SKMAP = Path.home() / "Git/Github/HaoZeke/sketchmap/examples/protein"
OUT = Path(__file__).resolve().parents[1] / "docs" / "ceriotti-figs"
ASINH = Path("/tmp/landfold-asinh/protein_asinh.ld")
PERIOD = 2.0 * np.pi
CMAP = LinearSegmentedColormap.from_list(
    "fes",
    [
        (0.00, "#e65014"),
        (0.18, "#7a1460"),
        (0.40, "#2c1a80"),
        (0.62, "#2f4ec4"),
        (0.82, "#6aa4e0"),
        (1.00, "#e8f2fb"),
    ],
)


def pair_period(x, period=PERIOD):
    n = len(x)
    D = np.empty(n * (n - 1) // 2)
    k = 0
    for i in range(n):
        dx = np.abs(x[i + 1 :] - x[i])
        dx = np.minimum(dx, period - dx)
        D[k : k + len(dx)] = np.sqrt((dx * dx).sum(1))
        k += len(dx)
    return D


def pair_euclid(x):
    n = len(x)
    d = np.empty(n * (n - 1) // 2)
    k = 0
    for i in range(n):
        dh = np.linalg.norm(x[i + 1 :] - x[i], axis=1)
        d[k : k + len(dh)] = dh
        k += len(dh)
    return d


def scores(D, d):
    q25, q75 = np.quantile(D, [0.25, 0.75])
    far = D >= q75
    near = D <= q25
    pear_far = float(np.corrcoef(D[far], d[far])[0, 1])
    spear_far = float(
        np.corrcoef(np.argsort(np.argsort(D[far])), np.argsort(np.argsort(d[far])))[0, 1]
    )
    rm_near = float(np.sqrt(np.mean((d[near] - D[near]) ** 2)))
    return pear_far, spear_far, rm_near


def pd_hist(ax, D, d, title):
    mx = max(float(np.quantile(D, 0.99)), float(np.quantile(d, 0.99)))
    H, xe, ye = np.histogram2d(D, d, bins=80, range=[[0, mx], [0, mx]])
    H = H.T
    H = H / H.max() if H.max() > 0 else H
    mesh = ax.pcolormesh(xe, ye, H, cmap=CMAP, shading="auto", vmin=0, vmax=1)
    ax.plot([0, mx], [0, mx], color="#111111", lw=0.6, ls="--")
    pear, spear, rm = scores(D, d)
    ax.set_title(title)
    ax.set_xlabel(r"$D$")
    ax.set_ylabel(r"$d$")
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.04,
        0.96,
        f"far Pearson {pear:.3f}\nfar Spearman {spear:.3f}\nnear RMSE {rm:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(fc="white", ec="none", alpha=0.8),
    )
    return mesh


def main() -> None:
    hd = np.loadtxt(SKMAP / "lm4.30cv.w01.1")[:, :30]
    pub = np.loadtxt(SKMAP / "lm4.30cv.w01.1.proj")[:, :2]
    D = pair_period(hd)
    d0 = pair_euclid(pub)
    print("Ceriotti protein", scores(D, d0))
    if ASINH.is_file():
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), dpi=160, facecolor="white")
        pd_hist(axes[0], D, d0, r"Ceriotti $6,8,8$ $P(D,d)$")
        ash = np.loadtxt(ASINH)[:, :2]
        d1 = pair_euclid(ash)
        print("asinh protein", scores(D, d1))
        pd_hist(axes[1], D, d1, r"asinh $P(D,d)$")
    else:
        fig, ax = plt.subplots(figsize=(5.4, 4.8), dpi=160, facecolor="white")
        pd_hist(ax, D, d0, r"Ceriotti hairpin $P(D,d)$ (far Spearman 0.18)")
        print("need", ASINH)
    dest = OUT / "protein_asinh_vs_ceriotti.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=160, facecolor="white")
    print("wrote", dest)


if __name__ == "__main__":
    main()
