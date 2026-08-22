#!/usr/bin/env python3
"""4042-min OOS occupancy on Ceriotti vs asinh DECAF chi.

The saturating Switch Torgerson of the same L1 book is a line (lambda2=0).
Full chi of asinh(L1) is 2-D; OOS of every quenched minimum fills it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "cosmo-lj38"
OUT = ROOT / "docs" / "ceriotti-figs"
ASINH = Path("/tmp/occ-book/lj38_decaf_oos.proj")
CER = Path("/tmp/occ-book/lj38_decaf_oos_cer.proj")
ENERGY = Path("/tmp/occ-book/lj38.energy")
BOOK = Path("/tmp/occ-book/lj38_decaf_book.json")

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)
GM_E = -173.928427
ICO_E = -173.252378
KT = 0.168


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


def kde_fes(xy, cf, ngrid=160, sigma=5.0):
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
    fes[on] = -KT * np.log(np.clip(rho[on] / rmax, 1e-12, 1))
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


def main() -> None:
    cf = _cf()
    e = np.loadtxt(ENERGY)
    ig = int(np.argmin(e))
    ii = int(np.argmin(np.abs(e - ICO_E)))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), facecolor="white")
    mesh = None
    for ax, path, title in (
        (axes[0], CER, r"Ceriotti $0.5,8,8$  $\chi$  4042 OOS"),
        (axes[1], ASINH, r"asinh $\sigma=0.5$  $\chi$  4042 OOS"),
    ):
        xy = load_xy(path)
        gx, gy, fes = kde_fes(xy, cf)
        mesh = paint(ax, gx, gy, fes)
        gm, ico = xy[ig], xy[ii]
        ax.scatter(gm[0], gm[1], s=110, marker="*", c="k", edgecolors="white", linewidths=0.6, zorder=6, label=rf"GM ${GM_E:.3f}$")
        ax.scatter(ico[0], ico[1], s=70, marker="D", c="k", edgecolors="white", linewidths=0.6, zorder=6, label=rf"ico ${ICO_E:.3f}$")
        ax.legend(loc="best", fontsize=8, frameon=True, fancybox=False)
        ax.set_title(title, fontsize=11)
        print(title, "F(GM)", cf.fes_at(gx, gy, fes, gm), "F(ico)", cf.fes_at(gx, gy, fes, ico))

    if BOOK.is_file():
        pts = json.loads(BOOK.read_text())["points"]
        xs = np.array([p["x"] for p in pts])
        ys = np.array([p["y"] for p in pts])
        inset = fig.add_axes([0.42, 0.12, 0.16, 0.16])
        inset.scatter(xs, ys, s=2, c="k", linewidths=0)
        inset.set_title(r"switch Torgerson  $\lambda_2=0$", fontsize=7, pad=2)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_linewidth(0.4)

    cax = fig.add_axes([0.28, 0.06, 0.44, 0.03])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label(r"$F/\varepsilon$  occupancy of 4042 inherent structures")
    dest = OUT / "elja_occ_lj38_decaf_oos_cmp.png"
    fig.savefig(dest, dpi=170, facecolor="white", bbox_inches="tight")
    print("wrote", dest)


if __name__ == "__main__":
    main()
