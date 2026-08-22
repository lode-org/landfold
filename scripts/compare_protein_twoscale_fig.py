#!/usr/bin/env python3
"""Hairpin P(D,d): published Ceriotti 6,8,8 vs two-scale / midweight.

Ardevol et al. JCTC 2015, 30-D periodic CVs. Ceriotti far Spearman is
~0.181. Writes docs/ceriotti-figs/protein_twoscale_pd.png.
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
WORK = Path("/tmp/landfold-twoscale")
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

PANELS = (
    ("pub", None, r"Ceriotti $6,8,8$"),
    ("asinh", ASINH, r"asinh $\sigma=6$"),
    ("asinh_mw", WORK / "protein_asinh_mw.ld", r"asinh + midweight"),
    ("asinh_cer", WORK / "protein_asinh_cer.ld", r"HD asinh / LD $6,2,8$"),
    ("asinh_cer_mw", WORK / "protein_asinh_cer_mw.ld", r"HD asinh / LD $6,2,8$ + mw"),
    ("ts", WORK / "protein_ts.ld", r"$ts,6,8,8$"),
    ("ts_mw", WORK / "protein_ts_mw.ld", r"$ts,6,8,8$ + midweight"),
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
    return mesh, (pear, spear, rm)


def main() -> None:
    hd = np.loadtxt(SKMAP / "lm4.30cv.w01.1")[:, :30]
    pub = np.loadtxt(SKMAP / "lm4.30cv.w01.1.proj")[:, :2]
    D = pair_period(hd)
    rows = []
    for key, path, title in PANELS:
        if key == "pub":
            xy = pub
        elif path is not None and path.is_file():
            xy = np.loadtxt(path)[:, :2]
        else:
            print("missing", key, path)
            continue
        d = pair_euclid(xy)
        s = scores(D, d)
        print(key, "pear_far spear_far rm_near", s)
        rows.append((key, title, D, d, s))
    if not rows:
        raise SystemExit("no embeddings")
    n = len(rows)
    ncols = 2 if n <= 2 else (3 if n <= 6 else 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.4 * ncols, 4.2 * nrows), dpi=160, facecolor="white"
    )
    axes = np.atleast_1d(axes).ravel()
    for i, (key, title, Dv, dv, _s) in enumerate(rows):
        pd_hist(axes[i], Dv, dv, title)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    dest = OUT / "protein_twoscale_pd.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=160, facecolor="white")
    print("wrote", dest)


if __name__ == "__main__":
    main()
