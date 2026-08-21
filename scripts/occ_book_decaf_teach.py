#!/usr/bin/env python3
"""Ceriotti-class occupancy FES of the Elja LJ38 DECAF book.

The plane is Euclidean PCA of the 23-D leftover-well class histograms
(400 families). Weight is leftover-well count. Pins are journal
minima whose DECAF histogram matches fcc, ico, leftover packing, and
a liquid-like member. The figure states that the occupancy well is
not the crystals.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "cosmo-lj38"
OUT = ROOT / "docs" / "ceriotti-figs"
HIST = Path("/tmp/occ-book/lj38_decaf.hist")
CV = Path("/tmp/occ-book/lj38.cv")
RENDER = Path("/tmp/occ-book/render")

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)

# 9-D DECAF codebook rows (fcc, ico, leftover packing, liquid), padded
# into the 23-D book by L1 match. fcc and ico match families 0 and 1
# at L1 = 0. leftover and liquid are nearest book rows.
REF = np.array(
    [
        [0.631578947, 0.157894737, 0.210526316, 0, 0, 0, 0, 0, 0],
        [0.421052632, 0, 0, 0.394736842, 0.131578947, 0.026315789, 0.026315789, 0, 0],
        [0.131578947, 0.078947368, 0.131578947, 0.263157895, 0.078947368, 0, 0, 0.289473684, 0.026315789],
        [0.263157895, 0, 0.052631579, 0.263157895, 0.105263158, 0.052631579, 0, 0.236842105, 0.026315789],
    ]
)
NAMES = ("fcc", "ico", "left", "liquid")
LABELS = {
    "fcc": "fcc",
    "ico": "ico",
    "left": "other packing",
    "liquid": "liquid",
}
ENERGIES = {"fcc": -173.93, "ico": -173.25, "left": -170.67, "liquid": -168.05}
SIDX = {"fcc": 0, "ico": 40, "left": 1393, "liquid": 3674}
# figure-fraction corners: fcc left, liquid into the well, ico top, leftover right
POS = {
    "fcc": (0.13, 0.34),
    "liquid": (0.13, 0.76),
    "ico": (0.85, 0.76),
    "left": (0.85, 0.34),
}


def _cf():
    spec = importlib.util.spec_from_file_location("compose_fes", EX / "compose_fes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_book(path: Path):
    rows, wells, fams = [], [], []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        fams.append(int(parts[0]))
        wells.append(float(parts[1]))
        rows.append([float(x) for x in parts[2:]])
    return np.asarray(rows), np.asarray(wells), np.asarray(fams)


def match_refs(hist: np.ndarray) -> dict[str, int]:
    pad = np.zeros((4, hist.shape[1]))
    pad[:, : REF.shape[1]] = REF
    out = {}
    for i, name in enumerate(NAMES):
        out[name] = int(np.abs(hist - pad[i]).sum(1).argmin())
    return out


def pca2(hist: np.ndarray) -> np.ndarray:
    x = hist - hist.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def kde_fes(xy: np.ndarray, weights: np.ndarray, cf, ngrid: int = 160, sigma: float = 4.0):
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= 0.12 * dx
    xmax += 0.12 * dx
    ymin -= 0.12 * dy
    ymax += 0.12 * dy
    counts, xe, ye = np.histogram2d(
        x, y, bins=ngrid, range=[[xmin, xmax], [ymin, ymax]], weights=weights
    )
    rho = cf._blur2d(counts.T, sigma=sigma)
    gx = 0.5 * (xe[:-1] + xe[1:])
    gy = 0.5 * (ye[:-1] + ye[1:])
    rmax = float(rho.max())
    mask = cf._fill_mask_holes(rho > 0.004 * rmax)
    fes = np.full_like(rho, np.nan)
    on = mask & (rho > 0)
    fes[on] = -np.log(np.clip(rho[on] / rmax, 1e-12, 1))
    return gx, gy, np.clip(fes, 0, 2)


def load_frame(png: Path) -> Image.Image:
    arr = np.asarray(Image.open(png).convert("RGBA")).copy()
    lum = arr[:, :, :3].max(2)
    arr[lum < 40, 3] = 0
    ys, xs = np.where(arr[:, :, 3] > 10)
    if ys.size:
        pad = 8
        arr = arr[
            max(ys.min() - pad, 0) : ys.max() + pad + 1,
            max(xs.min() - pad, 0) : xs.max() + pad + 1,
        ]
    return Image.fromarray(arr)


def main() -> None:
    cf = _cf()
    hist, wells, fams = load_book(HIST)
    row_of = match_refs(hist)
    pc = pca2(hist)
    cv = np.loadtxt(CV)
    gx, gy, fes = kde_fes(pc, wells, cf)

    fig = plt.figure(figsize=(13.0, 9.8), facecolor="white")
    ax = fig.add_axes([0.22, 0.20, 0.56, 0.64])
    ax.set_facecolor("white")
    mesh = ax.contourf(gx, gy, fes, levels=np.linspace(0, 2, 21), cmap=PES, extend="max")
    ax.contour(
        gx,
        gy,
        np.where(np.isfinite(fes), fes, np.nan),
        levels=np.linspace(0.15, 1.85, 10),
        colors="#1a1a2e",
        linewidths=0.35,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes([0.30, 0.10, 0.40, 0.026])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label(r"$F/kT$  leftover-well occupancy on DECAF PCA", fontsize=11)
    cb.set_ticks([0, 0.5, 1, 1.5, 2])

    for name, (fx, fy) in POS.items():
        j = row_of[name]
        tip = pc[j]
        img = load_frame(RENDER / f"lj38_{name}.png")
        fig.add_artist(
            AnnotationBbox(
                OffsetImage(img, zoom=0.095),
                (fx, fy),
                xycoords="figure fraction",
                frameon=False,
                box_alignment=(0.5, 0.5),
            )
        )
        ax.annotate(
            "",
            xy=tip,
            xycoords="data",
            xytext=(fx, fy),
            textcoords="figure fraction",
            arrowprops=dict(arrowstyle="-", color="k", lw=0.8),
        )
        fval = cf.fes_at(gx, gy, fes, tip)
        hax = fig.add_axes([fx - 0.09, fy - 0.22, 0.18, 0.075])
        bins = np.arange(4, 14)
        hax.bar(bins, cv[SIDX[name]], color="k", width=0.7)
        hax.set_xlim(3.5, 13.5)
        hax.set_xticks(bins)
        hax.set_xticklabels([str(int(b)) for b in bins])
        hax.tick_params(labelsize=6, length=2)
        hax.set_yticks([])
        hax.set_title(
            f"{LABELS[name]}  {ENERGIES[name]:.1f}  F={fval:.2f}",
            fontsize=8,
            pad=2,
        )
        for spine in hax.spines.values():
            spine.set_linewidth(0.45)
        print(
            name,
            "fam",
            int(fams[j]),
            "wells",
            float(wells[j]),
            "pc",
            tip,
            "F",
            fval,
        )

    dest = OUT / "elja_occ_lj38_decaf_teach.png"
    fig.savefig(dest, dpi=170, facecolor="white")
    print("wrote", dest)


if __name__ == "__main__":
    main()
