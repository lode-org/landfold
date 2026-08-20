#!/usr/bin/env python3
"""JCTC-class panel of the TSE committor on the same body as compose_fes.

Same filled body, same cluster insets, same leader lines. The colour is
the binned mean of ts.all column 2 (p_B), not occupancy invert and not
a raw scatter / unbounded GP.
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

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("compose_fes", ROOT / "compose_fes.py")
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)

# Reactant (blue) to product (orange). Same family as the FES panel,
# opposite sense: high p_B is the hot colour.
PB_CMAP = LinearSegmentedColormap.from_list(
    "pb",
    [
        (0.00, "#2f4ec4"),
        (0.25, "#6aa4e0"),
        (0.50, "#e8e4ee"),
        (0.75, "#c43b6a"),
        (1.00, "#e65014"),
    ],
)


def mean_field(xy: np.ndarray, values: np.ndarray, gx, gy, mask, sigma: float = 7.0):
    xmin = float(gx[0] - 0.5 * (gx[1] - gx[0]))
    xmax = float(gx[-1] + 0.5 * (gx[1] - gx[0]))
    ymin = float(gy[0] - 0.5 * (gy[1] - gy[0]))
    ymax = float(gy[-1] + 0.5 * (gy[1] - gy[0]))
    ngrid = gx.size
    num, xedges, yedges = np.histogram2d(
        xy[:, 0],
        xy[:, 1],
        bins=ngrid,
        range=[[xmin, xmax], [ymin, ymax]],
        weights=values,
    )
    den, _, _ = np.histogram2d(
        xy[:, 0],
        xy[:, 1],
        bins=ngrid,
        range=[[xmin, xmax], [ymin, ymax]],
    )
    num = cf._blur2d(num.T, sigma=sigma)
    den = cf._blur2d(den.T, sigma=sigma)
    field = np.full_like(num, np.nan)
    on = mask & (den > 1e-12)
    field[on] = np.clip(num[on] / den[on], 0.0, 1.0)
    return field


def main() -> None:
    proj = cf.OUT / "ts.proj"
    if not proj.exists():
        raise SystemExit("run ./run.sh first")
    xy = cf.load_xy(proj)
    ts = np.loadtxt(cf.ROOT / "ts.all")
    if ts.shape[0] != xy.shape[0]:
        raise SystemExit("ts.all and ts.proj row counts differ")
    pb = ts[:, 1]
    tscv = ts[:, 2:12]
    gx, gy, fes = cf.kde_fes(xy)
    mask = np.isfinite(fes)
    field = mean_field(xy, pb, gx, gy, mask)

    fig = plt.figure(figsize=(13.0, 9.8), facecolor="white")
    ax = fig.add_axes([0.22, 0.20, 0.56, 0.64])
    ax.set_facecolor("white")
    mesh = ax.contourf(
        gx, gy, field, levels=np.linspace(0.0, 1.0, 21), cmap=PB_CMAP, extend="neither"
    )
    ax.contour(
        gx,
        gy,
        field,
        levels=np.linspace(0.08, 0.92, 11),
        colors="#1a1a2e",
        linewidths=0.35,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    cax = fig.add_axes([0.26, 0.09, 0.28, 0.028])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label(r"$p_B$", fontsize=12)
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    panels = [
        ("lj38_ico.png", "lj38_ico.xyz", (0.84, 0.78)),
        ("lj38_fcc.png", "lj38_fcc.xyz", (0.84, 0.34)),
        ("lj38.19.png", "lj38.19.xyz", (0.14, 0.78)),
        ("lj38.17.png", "lj38.17.xyz", (0.14, 0.34)),
    ]
    for png_name, xyz_name, (fx, fy) in panels:
        png = cf.OUT / png_name
        xyz = cf.ROOT / xyz_name
        if not png.exists() or not xyz.exists():
            continue
        tip = cf.match_tip(xy, tscv, cf.cn_vector(xyz))
        img = cf.load_frame(png)
        im = OffsetImage(img, zoom=0.155)
        fig.add_artist(
            AnnotationBbox(
                im,
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
            arrowprops=dict(arrowstyle="-", color="k", lw=0.8),
        )
        pval = cf.fes_at(gx, gy, field, tip)
        k, c = cf.cn_hist(xyz)
        hax = fig.add_axes([fx - 0.08, fy - 0.24, 0.16, 0.08])
        hax.bar(k, c, color="k", width=0.7)
        hax.set_xlim(2.5, 12.5)
        hax.set_xticks(cf.CN_BINS)
        hax.set_xticklabels([str(int(v)) for v in cf.CN_BINS])
        hax.tick_params(labelsize=6, length=2)
        hax.set_yticks([])
        hax.set_title(f"$p_B$ = {pval:.2f}", fontsize=9, pad=2)
        for spine in hax.spines.values():
            spine.set_linewidth(0.45)

    dest = cf.OUT / "lj38_committor.png"
    figs = cf.ROOT.parents[1] / "docs" / "ceriotti-figs" / "lj38_committor_panel.png"
    figs.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=170, facecolor="white")
    fig.savefig(figs, dpi=170, facecolor="white")
    print("wrote", dest)
    print("wrote", figs)


if __name__ == "__main__":
    main()
