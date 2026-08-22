#!/usr/bin/env python3
"""Inherent-structure occupancy FES of the Elja LJ38 book.

Ceriotti JCTC Fig. 5 is occupancy of many frames on chi. The 400-family
Boltzmann map at T*=0.18 is 95% one family and collapses to dots. This
panel is all 4042 quenched minima on asinh vs published Ceriotti chi,
same KDE occupancy invert as the teaching FES. The Wales GM and ico
are marked on the surface.
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
SRC = Path("/tmp/landfold-occ-from-terra/landfold-occ-book")
ENERGY = Path("/tmp/occ-book/lj38.energy")
CV = Path("/tmp/occ-book/lj38.cv")
RENDER = Path("/tmp/occ-book/render")

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)
GM_E = -173.928427
ICO_E = -173.252378
KT = 0.168  # matches the teaching FES (F/eps)


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


def kde_fes(xy, cf, ngrid=160, sigma=5.0, kt: float = KT):
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= 0.10 * dx
    xmax += 0.10 * dx
    ymin -= 0.10 * dy
    ymax += 0.10 * dy
    counts, xe, ye = np.histogram2d(x, y, bins=ngrid, range=[[xmin, xmax], [ymin, ymax]])
    rho = cf._blur2d(counts.T, sigma=sigma)
    gx = 0.5 * (xe[:-1] + xe[1:])
    gy = 0.5 * (ye[:-1] + ye[1:])
    rmax = float(rho.max())
    mask = cf._fill_mask_holes(rho > 0.004 * rmax)
    fes = np.full_like(rho, np.nan)
    on = mask & (rho > 0)
    fes[on] = -kt * np.log(np.clip(rho[on] / rmax, 1e-12, 1))
    return gx, gy, np.clip(fes, 0, 2)


def paint(ax, gx, gy, fes):
    mesh = ax.contourf(gx, gy, fes, levels=np.linspace(0, 2, 21), cmap=PES, extend="max")
    ax.contour(
        gx,
        gy,
        np.where(np.isfinite(fes), fes, np.nan),
        levels=np.linspace(0.15, 1.85, 12),
        colors="#1a1a2e",
        linewidths=0.35,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mesh


def mark(ax, gm, ico):
    h1 = ax.scatter(
        gm[0], gm[1], s=120, marker="*", c="k", edgecolors="white", linewidths=0.6, zorder=6,
        label=rf"GM ${GM_E:.3f}$",
    )
    h2 = ax.scatter(
        ico[0], ico[1], s=70, marker="D", c="k", edgecolors="white", linewidths=0.6, zorder=6,
        label=rf"ico ${ICO_E:.3f}$",
    )
    ax.legend(handles=[h1, h2], loc="best", fontsize=8, frameon=True, fancybox=False)


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


def plot_cmp(cf, e) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), facecolor="white")
    mesh = None
    for ax, fname, title in (
        (axes[0], "lj38_ceriotti.proj", r"Ceriotti $5,8,8$  $\chi$"),
        (axes[1], "lj38_asinh.proj", r"asinh $\sigma=5$  $\chi$"),
    ):
        xy = load_xy(SRC / fname)
        gx, gy, fes = kde_fes(xy, cf)
        mesh = paint(ax, gx, gy, fes)
        gm = xy[int(np.argmin(e))]
        ico = xy[int(np.argmin(np.abs(e - ICO_E)))]
        mark(ax, gm, ico)
        ax.set_title(title, fontsize=11)
        print(
            title,
            "F(GM)",
            cf.fes_at(gx, gy, fes, gm),
            "F(ico)",
            cf.fes_at(gx, gy, fes, ico),
        )
    cax = fig.add_axes([0.28, 0.08, 0.44, 0.03])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label(r"$F/\varepsilon$  occupancy of 4042 inherent structures")
    cb.set_ticks([0, 0.5, 1, 1.5, 2])
    dest = OUT / "elja_occ_lj38_chi_surface.png"
    fig.savefig(dest, dpi=170, facecolor="white", bbox_inches="tight")
    print("wrote", dest)


def plot_twoscale(cf, e) -> None:
    """Ceriotti vs asinh vs HD-asinh/LD-Ceriotti occupancy of 4042 minima."""
    panels = (
        ("lj38_ceriotti.proj", r"Ceriotti $5,8,1$ / $5,2,2$"),
        ("lj38_asinh.proj", r"asinh $\sigma=5$"),
        ("lj38_asinh_cer.proj", r"HD asinh / LD $5,2,2$"),
        ("lj38_asinh_cer_mw.proj", r"HD asinh / LD $5,2,2$ + mw"),
    )
    have = [(f, t) for f, t in panels if (SRC / f).is_file() and (SRC / f).stat().st_size > 100]
    if not have:
        print("no twoscale occupancy maps")
        return
    fig, axes = plt.subplots(1, len(have), figsize=(5.2 * len(have), 4.8), facecolor="white")
    axes = np.atleast_1d(axes)
    mesh = None
    for ax, (fname, title) in zip(axes, have):
        xy = load_xy(SRC / fname)
        if len(xy) != len(e):
            print("row mismatch", fname, len(xy), len(e))
            ax.axis("off")
            continue
        gx, gy, fes = kde_fes(xy, cf)
        mesh = paint(ax, gx, gy, fes)
        gm = xy[int(np.argmin(e))]
        ico = xy[int(np.argmin(np.abs(e - ICO_E)))]
        mark(ax, gm, ico)
        ax.set_title(title, fontsize=11)
        print(
            title,
            "F(GM)",
            cf.fes_at(gx, gy, fes, gm),
            "F(ico)",
            cf.fes_at(gx, gy, fes, ico),
        )
    if mesh is not None:
        fig.subplots_adjust(bottom=0.18, wspace=0.08)
        cax = fig.add_axes([0.28, 0.08, 0.44, 0.03])
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cb.set_label(r"$F/\varepsilon$  occupancy of 4042 inherent structures")
        cb.set_ticks([0, 0.5, 1, 1.5, 2])
    dest = OUT / "elja_occ_lj38_twoscale_fes.png"
    fig.savefig(dest, dpi=170, facecolor="white", bbox_inches="tight")
    print("wrote", dest)


def plot_teach(cf, e) -> None:
    xy = load_xy(SRC / "lj38_asinh.proj")
    cv = np.loadtxt(CV)
    gx, gy, fes = kde_fes(xy, cf)
    picks = {"fcc": 0, "ico": 40, "left": 1393}
    mask = (e > -172.5) & (xy[:, 0] > -5.0) & (xy[:, 0] < 8.0)
    picks["liquid"] = int(np.argmax(np.where(mask, xy[:, 1], -1e9)))
    labels = {"fcc": "GM fcc", "ico": "ico", "left": "other packing", "liquid": "liquid"}
    pos = {
        "liquid": (0.13, 0.76),
        "left": (0.13, 0.34),
        "ico": (0.85, 0.76),
        "fcc": (0.85, 0.34),
    }
    fig = plt.figure(figsize=(13.0, 9.8), facecolor="white")
    ax = fig.add_axes([0.22, 0.20, 0.56, 0.64])
    ax.set_facecolor("white")
    mesh = paint(ax, gx, gy, fes)
    cax = fig.add_axes([0.30, 0.10, 0.40, 0.026])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label(r"$F/\varepsilon$  occupancy of 4042 inherent structures", fontsize=11)
    cb.set_ticks([0, 0.5, 1, 1.5, 2])
    for name, (fx, fy) in pos.items():
        index = picks[name]
        tip = xy[index]
        png = RENDER / f"lj38_{name}.png"
        if not png.is_file():
            continue
        fig.add_artist(
            AnnotationBbox(
                OffsetImage(load_frame(png), zoom=0.095),
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
        hax.bar(bins, cv[index], color="k", width=0.7)
        hax.set_xlim(3.5, 13.5)
        hax.set_xticks(bins)
        hax.set_xticklabels([str(int(b)) for b in bins])
        hax.tick_params(labelsize=6, length=2)
        hax.set_yticks([])
        hax.set_title(f"{labels[name]}  {e[index]:.3f}  F={fval:.2f}", fontsize=8, pad=2)
        for spine in hax.spines.values():
            spine.set_linewidth(0.45)
        print(name, "idx", index, "E", float(e[index]), "F", fval)
    dest = OUT / "elja_occ_lj38_teach.png"
    fig.savefig(dest, dpi=170, facecolor="white")
    print("wrote", dest)


def main() -> None:
    cf = _cf()
    e = np.loadtxt(ENERGY)
    plot_cmp(cf, e)
    plot_twoscale(cf, e)
    plot_teach(cf, e)


if __name__ == "__main__":
    main()
