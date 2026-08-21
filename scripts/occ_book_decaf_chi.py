#!/usr/bin/env python3
"""Occupancy FES of landfold asinh-chi of the Elja DECAF book.

The plane is chi (stress of F(L1) vs f(d)), not PCA and not Torgerson.
Pins are the same journal minima as occ_book_decaf_teach.py.
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
XY = Path("/tmp/occ-book/lj38_decaf_asinh.ld")
CV = Path("/tmp/occ-book/lj38.cv")
RENDER = Path("/tmp/occ-book/render")

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)

# book rows: fcc fam0 = Wales GM, ico fam1 = second minimum
ROW = {"fcc": 0, "ico": 1, "left": 147, "liquid": 36}
LABELS = {
    "fcc": "GM fcc",
    "ico": "ico",
    "left": "other packing",
    "liquid": "liquid",
}
# Wales / Doye catalogued energies (also the journal exact values)
ENERGIES = {"fcc": -173.928427, "ico": -173.252378, "left": -170.67, "liquid": -168.05}
SIDX = {"fcc": 0, "ico": 40, "left": 1393, "liquid": 3674}
# asinh chi: fcc right, ico top, leftover left, liquid into the well
POS = {
    "fcc": (0.85, 0.34),
    "ico": (0.85, 0.76),
    "left": (0.13, 0.34),
    "liquid": (0.13, 0.76),
}


def _cf():
    spec = importlib.util.spec_from_file_location("compose_fes", EX / "compose_fes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_xy(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        rows.append([float(p[0]), float(p[1])])
    return np.asarray(rows)


def load_wells(path: Path) -> np.ndarray:
    wells = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        wells.append(float(line.split()[1]))
    return np.asarray(wells)


def kde_fes(xy, weights, cf, ngrid=160, sigma=4.0):
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
    xy = load_xy(XY)
    wells = load_wells(HIST)
    cv = np.loadtxt(CV)
    gx, gy, fes = kde_fes(xy, wells, cf)

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
    cb.set_label(r"$F/kT$  leftover-well occupancy on asinh $\chi$ of DECAF $L^1$", fontsize=11)
    cb.set_ticks([0, 0.5, 1, 1.5, 2])

    # known catalogued minima sit on the map, not only in the corner frames
    for name, marker, color in (("fcc", "*", "#111111"), ("ico", "D", "#111111")):
        tip = xy[ROW[name]]
        ax.scatter(
            tip[0],
            tip[1],
            s=90 if name == "fcc" else 55,
            marker=marker,
            c=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=6,
        )
        ax.annotate(
            f"{LABELS[name]}  {ENERGIES[name]:.3f}",
            tip,
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color="#111111",
            zorder=7,
        )

    for name, (fx, fy) in POS.items():
        j = ROW[name]
        tip = xy[j]
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
            f"{LABELS[name]}  {ENERGIES[name]:.3f}  F={fval:.2f}",
            fontsize=8,
            pad=2,
        )
        for spine in hax.spines.values():
            spine.set_linewidth(0.45)
        print(name, "row", j, "xy", tip, "F", fval, "wells", wells[j])

    dest = OUT / "elja_occ_lj38_decaf_chi.png"
    fig.savefig(dest, dpi=170, facecolor="white")
    print("wrote", dest)
    write_cmp(xy, wells, cf)


def write_cmp(xy_asinh: np.ndarray, wells: np.ndarray, cf) -> None:
    """Ceriotti vs asinh, with the two known minima marked."""
    cer = load_xy(Path("/tmp/occ-book/lj38_decaf_cer.ld"))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), facecolor="white")
    for ax, xy, title in (
        (axes[0], cer, r"Ceriotti $0.5,8,8$  $\chi$  stress $0.025$"),
        (axes[1], xy_asinh, r"asinh $\sigma=0.5$  $\chi$  stress $0.007$"),
    ):
        gx, gy, fes = kde_fes(xy, wells, cf)
        ax.contourf(gx, gy, fes, levels=np.linspace(0, 2, 21), cmap=PES, extend="max")
        ax.contour(
            gx,
            gy,
            np.where(np.isfinite(fes), fes, np.nan),
            levels=np.linspace(0.15, 1.85, 8),
            colors="#1a1a2e",
            linewidths=0.3,
        )
        marks = (
            ("fcc", ROW["fcc"], "*", 110, r"GM $-173.928$"),
            ("ico", ROW["ico"], "D", 70, r"ico $-173.252$"),
            ("occ", 19, "o", 50, r"occ $321$ wells"),
        )
        for _name, j, marker, size, lab in marks:
            ax.scatter(
                xy[j, 0],
                xy[j, 1],
                s=size,
                marker=marker,
                c="k",
                edgecolors="white",
                linewidths=0.6,
                zorder=6,
            )
            ax.annotate(lab, xy[j], textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    fig.tight_layout()
    dest = OUT / "elja_occ_lj38_decaf_chi_cmp.png"
    fig.savefig(dest, dpi=160, facecolor="white")
    print("wrote", dest)


if __name__ == "__main__":
    main()
