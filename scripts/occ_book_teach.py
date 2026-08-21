#!/usr/bin/env python3
"""Annotated occupancy FES of the Elja LJ38 book.

The map is asinh sketch-map of n4..n13 (unbounded F). Color is
occupancy F/eps. Crystal frames are xyzrender of the journal
minima, not the course pngs.
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
PROJ = Path("/tmp/landfold-occ-from-terra/landfold-occ-book/lj38_asinh.proj")
if not PROJ.is_file():
    PROJ = Path("/tmp/landfold-occ-book/lj38_asinh.proj")
CSV = Path("/tmp/occ-book/lj38_asinh_fes.csv")
RENDER = Path("/tmp/occ-book/render")

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)


def load_xy(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        rows.append([float(p[0]), float(p[1])])
    return np.asarray(rows)


def write_fes(xy: np.ndarray, dest: Path, ngrid: int = 160, kt: float = 0.168):
    spec = importlib.util.spec_from_file_location("compose_fes", EX / "compose_fes.py")
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= 0.10 * dx
    xmax += 0.10 * dx
    ymin -= 0.10 * dy
    ymax += 0.10 * dy
    counts, xedges, yedges = np.histogram2d(
        x, y, bins=ngrid, range=[[xmin, xmax], [ymin, ymax]]
    )
    rho = cf._blur2d(counts.T, sigma=5.0)
    gx = 0.5 * (xedges[:-1] + xedges[1:])
    gy = 0.5 * (yedges[:-1] + yedges[1:])
    rmax = float(rho.max())
    mask = cf._fill_mask_holes(rho > 0.004 * rmax)
    fes = np.full_like(rho, np.nan)
    on = mask & (rho > 0.0)
    fes[on] = -kt * np.log(np.clip(rho[on] / rmax, 1e-12, 1.0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as w:
        w.write("# x y F rho\n")
        for iy, yv in enumerate(gy):
            for ix, xv in enumerate(gx):
                fv = fes[iy, ix]
                fs = "nan" if not np.isfinite(fv) else f"{fv:.10e}"
                w.write(f"{xv:.8f} {yv:.8f} {fs} {rho[iy, ix]:.10e}\n")
            w.write("\n")


def load_fes(path: Path):
    raw = np.loadtxt(path, comments="#")
    xs = np.unique(np.round(raw[:, 0], 12))
    # first-seen unique
    def uniq(col):
        _, idx = np.unique(np.round(raw[:, col], 12), return_index=True)
        return raw[np.sort(idx), col]

    x = uniq(0)
    y = uniq(1)
    nx, ny = x.size, y.size
    fes = raw[:, 2].reshape(ny, nx)
    return x, y, fes


def frame(png: Path) -> Image.Image:
    img = Image.open(png).convert("RGBA")
    arr = np.asarray(img).copy()
    ink = arr[:, :, :3].astype(np.int16)
    arr[ink.max(axis=2) < 28, 3] = 0
    return Image.fromarray(arr)


def main() -> None:
    xy = load_xy(PROJ)
    e = np.loadtxt("/tmp/occ-book/lj38.energy")
    if not CSV.is_file():
        write_fes(xy, CSV)
    gx, gy, fes = load_fes(CSV)
    fcc = xy[0]
    ico = xy[40]
    left = xy[1393]
    liquid = xy[2961]

    fig = plt.figure(figsize=(11.4, 7.6), facecolor="white")
    ax = fig.add_axes([0.22, 0.16, 0.56, 0.70])
    levels = np.linspace(0.0, 2.0, 21)
    mesh = ax.contourf(gx, gy, fes, levels=levels, cmap=PES, extend="max")
    finite = np.isfinite(fes)
    ax.contour(
        gx,
        gy,
        np.where(finite, fes, np.nan),
        levels=np.linspace(0.15, 1.85, 10),
        colors="#1a1a2e",
        linewidths=0.35,
    )
    ax.set_xlabel(r"$s_1$  (asinh $F$ of n4..n13)")
    ax.set_ylabel(r"$s_2$")
    ax.set_title("Elja leftover wells, asinh occupancy FES")
    ax.set_aspect("equal", adjustable="box")

    ax.annotate(
        "liquid / ico-like\nmost of the 4042 minima",
        xy=liquid,
        xytext=(2.0, 11.2),
        ha="center",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3),
    )
    ax.annotate(
        "second occupied\npacking family",
        xy=left,
        xytext=(-13.5, 8.8),
        ha="center",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3),
    )
    ax.annotate(
        "fcc ground state\nE = -173.93\nrare (41 of 4042)",
        xy=fcc,
        xytext=(5.5, -13.8),
        ha="center",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3),
    )
    ax.scatter([fcc[0]], [fcc[1]], s=40, marker="*", c="k", zorder=5)
    ax.scatter([ico[0]], [ico[1]], s=28, marker="o", facecolors="none", edgecolors="k", zorder=5)
    ax.annotate(
        "ico\nE = -173.25",
        xy=ico,
        xytext=(16.2, 4.0),
        fontsize=8,
        ha="left",
        arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3),
    )

    cax = fig.add_axes([0.28, 0.07, 0.44, 0.022])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label(r"$F/\varepsilon = -kT\ln(\rho/\rho_{\max})$   dark = the search sat here")
    cb.set_ticks([0, 0.5, 1.0, 1.5, 2.0])

    insets = [
        (RENDER / "lj38_fcc.png", fcc, (0.88, 0.22)),
        (RENDER / "lj38_ico.png", ico, (0.88, 0.78)),
        (RENDER / "lj38_left.png", left, (0.10, 0.78)),
        (RENDER / "lj38_liquid.png", liquid, (0.10, 0.22)),
    ]
    for png, tip, (fx, fy) in insets:
        if not png.is_file():
            continue
        img = frame(png)
        fig.add_artist(
            AnnotationBbox(
                OffsetImage(img, zoom=0.16),
                (fx, fy),
                xycoords=fig.transFigure,
                frameon=False,
                box_alignment=(0.5, 0.5),
            )
        )
        ax.annotate(
            "",
            xy=tip,
            xycoords=ax.transData,
            xytext=(fx, fy),
            textcoords=fig.transFigure,
            arrowprops=dict(arrowstyle="-", color="k", lw=0.7),
        )

    fig.text(
        0.50,
        0.015,
        "asinh F on n4..n13. Color is occupancy, not energy. "
        "Drawings are xyzrender of the journal minima at those points.",
        ha="center",
        fontsize=8,
    )
    dest = OUT / "elja_occ_lj38_teach.png"
    fig.savefig(dest, dpi=170, facecolor="white")
    print("wrote", dest, "fcc", fcc, "ico", ico, "E fcc/ico", e[0], e[40])


if __name__ == "__main__":
    main()
