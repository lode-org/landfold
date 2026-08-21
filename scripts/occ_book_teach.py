#!/usr/bin/env python3
"""Annotated occupancy FES of the Elja LJ38 book.

The map is Ceriotti chi of n4..n13. Color is occupancy
F/eps = -kT ln(rho/rhomax). Callouts name the wells.
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
CSV = Path("/tmp/occ-book/lj38_ceriotti_fes.csv")
PROJ = Path("/tmp/landfold-occ-from-terra/landfold-occ-book/lj38_ceriotti.proj")
if not PROJ.is_file():
    PROJ = Path("/tmp/landfold-occ-book/lj38_ceriotti.proj")

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
    gx, gy, fes = load_fes(CSV)
    xy = load_xy(PROJ)
    e = np.loadtxt("/tmp/occ-book/lj38.energy")
    fcc = xy[0]
    ico = xy[40]

    fig = plt.figure(figsize=(11.2, 7.4), facecolor="white")
    ax = fig.add_axes([0.28, 0.16, 0.46, 0.72])
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
    ax.set_xlabel(r"$s_1$  (Ceriotti $\chi$ of n4..n13)")
    ax.set_ylabel(r"$s_2$")
    ax.set_title("Elja leftover wells, occupancy FES")
    ax.set_aspect("equal", adjustable="box")

    # Name the wells on the map.
    ax.annotate(
        "liquid / ico-like\nmost of the 4042 minima\nlive here",
        xy=(3.0, 2.0),
        xytext=(3.0, 11.5),
        ha="center",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3),
    )
    ax.annotate(
        "second occupied\npacking family",
        xy=(-7.2, 0.0),
        xytext=(-14.5, 8.5),
        ha="center",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3),
    )
    ax.annotate(
        "fcc ground state\nE = -173.93\nrare (41 of 4042)",
        xy=fcc,
        xytext=(6.0, -14.5),
        ha="center",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3),
    )
    ax.scatter([fcc[0]], [fcc[1]], s=40, marker="*", c="k", zorder=5)
    ax.scatter([ico[0]], [ico[1]], s=28, marker="o", facecolors="none", edgecolors="k", zorder=5)
    ax.annotate("ico\nE = -173.25", xy=ico, xytext=(16.5, 3.5), fontsize=8, ha="left",
                arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))

    cax = fig.add_axes([0.30, 0.06, 0.42, 0.025])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label(r"$F/\varepsilon = -kT\ln(\rho/\rho_{\max})$   dark = the search sat here")
    cb.set_ticks([0, 0.5, 1.0, 1.5, 2.0])

    # Structure insets.
    insets = [
        (EX / "out" / "lj38_fcc.png", fcc, (0.86, 0.22), "fcc"),
        (EX / "out" / "lj38_ico.png", ico, (0.86, 0.72), "ico"),
    ]
    for png, tip, (fx, fy), name in insets:
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
        0.03,
        0.50,
        "How to read this\n"
        "\n"
        "Axes: sketch-map of\n"
        "coordination counts\n"
        "(n4..n13), not raw xyz.\n"
        "\n"
        "Color: occupancy of the\n"
        "Elja leftover wells,\n"
        "not the potential energy.\n"
        "A deep energy well that\n"
        "was barely visited is\n"
        "pale or a small pocket.\n"
        "\n"
        "fcc is the energy GS\n"
        "and almost empty here.\n"
        "ico sits on the rim of\n"
        "the liquid cloud.\n"
        "\n"
        "The TSE (ts.all) is the\n"
        "ridge between wells.\n"
        "It cannot draw this.",
        fontsize=8,
        va="center",
        family="sans-serif",
        linespacing=1.35,
    )
    dest = OUT / "elja_occ_lj38_teach.png"
    fig.savefig(dest, dpi=170, facecolor="white")
    print("wrote", dest, "fcc", fcc, "ico", ico, "E fcc/ico", e[0], e[40])


if __name__ == "__main__":
    main()
