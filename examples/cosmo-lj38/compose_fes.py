#!/usr/bin/env python3
"""
Compose a FES-in-the-middle figure: filled contours, leader lines to
xyzrender frames, integer CN 3-12 bars, F labels. Layout matches the
Ceriotti cluster-FES panel class, not a 1+4 matplotlib gallery.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
KT = 0.168

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


def load_xy(path: Path) -> np.ndarray:
    pts = np.loadtxt(path)
    return np.atleast_2d(pts)[:, :2]


def kde_fes(xy: np.ndarray, ngrid: int = 220, pad: float = 0.18):
    """Gaussian KDE on the embedded points -> F = -kT ln(rho/rhomax)."""
    x = xy[:, 0]
    y = xy[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= pad * dx
    xmax += pad * dx
    ymin -= pad * dy
    ymax += pad * dy
    # Scott bandwidth, then inflate so the map is a filled body.
    n = xy.shape[0]
    hx = 1.15 * x.std(ddof=1) * n ** (-1.0 / 6.0)
    hy = 1.15 * y.std(ddof=1) * n ** (-1.0 / 6.0)
    hx = max(hx, 0.25)
    hy = max(hy, 0.25)
    gx = np.linspace(xmin, xmax, ngrid)
    gy = np.linspace(ymin, ymax, ngrid)
    XX, YY = np.meshgrid(gx, gy)
    rho = np.zeros_like(XX)
    inv = 1.0 / (2.0 * np.pi * hx * hy * n)
    # subsample so the grid fill stays cheap
    if n > 600:
        pick = np.random.default_rng(0).choice(n, 600, replace=False)
        sample = xy[pick]
    else:
        sample = xy
    inv = 1.0 / (2.0 * np.pi * hx * hy * len(sample))
    for px, py in sample:
        rho += np.exp(-0.5 * ((XX - px) / hx) ** 2 - 0.5 * ((YY - py) / hy) ** 2)
    rho *= inv
    rmax = rho.max()
    fes = np.full_like(rho, np.nan)
    mask = rho > 0.02 * rmax
    fes[mask] = -KT * np.log(rho[mask] / rmax)
    fes = np.clip(fes, 0.0, 2.0)
    return gx, gy, fes


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


def basin_f(xy: np.ndarray, gx, gy, fes, pt):
    # nearest projected point, then F at that KDE cell
    d = np.hypot(xy[:, 0] - pt[0], xy[:, 1] - pt[1])
    j = int(np.argmin(d))
    ix = int(np.argmin(np.abs(gx - xy[j, 0])))
    iy = int(np.argmin(np.abs(gy - xy[j, 1])))
    v = fes[iy, ix]
    return 0.0 if np.isnan(v) else float(v)


def main():
    proj = OUT / "ts.proj"
    if not proj.exists():
        raise SystemExit("run ./run.sh first")
    xy = load_xy(proj)
    gx, gy, fes = kde_fes(xy)

    # k-means-ish 4 basins by greedy farthest then assign
    rng = np.random.default_rng(1)
    centers = [xy[rng.integers(0, len(xy))]]
    for _ in range(3):
        dmin = np.min([np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]) for c in centers], axis=0)
        centers.append(xy[int(np.argmax(dmin))])
    centers = np.asarray(centers)
    assign = np.argmin(
        [np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]) for c in centers], axis=0
    )
    basin_xy = []
    for k in range(4):
        pts = xy[assign == k]
        basin_xy.append(pts.mean(axis=0) if len(pts) else centers[k])

    fig = plt.figure(figsize=(12.4, 9.6), facecolor="white")
    ax = fig.add_axes([0.22, 0.22, 0.56, 0.62])
    ax.set_facecolor("white")
    mesh = ax.contourf(gx, gy, fes, levels=np.linspace(0, 2, 21), cmap=CMAP, extend="max")
    ax.contour(
        gx, gy, fes, levels=np.linspace(0.1, 1.8, 12), colors="#1a1a2e", linewidths=0.35
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    cax = fig.add_axes([0.62, 0.08, 0.28, 0.028])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label(r"$F/\epsilon$", fontsize=12)
    cb.set_ticks([0, 0.5, 1.0, 1.5, 2.0])

    panels = [
        ("lj38_fcc.png", "lj38_fcc.xyz", basin_xy[0], (0.04, 0.70)),
        ("lj38_ico.png", "lj38_ico.xyz", basin_xy[1], (0.80, 0.70)),
        ("lj38.17.png", "lj38.17.xyz", basin_xy[2], (0.04, 0.18)),
        ("lj38.19.png", "lj38.19.xyz", basin_xy[3], (0.80, 0.18)),
        ("lj38.png", "lj38.xyz", basin_xy[0] * 0.5 + basin_xy[1] * 0.5, (0.40, 0.04)),
    ]
    for png_name, xyz_name, tip, (fx, fy) in panels:
        png = OUT / png_name
        xyz = ROOT / xyz_name
        if not png.exists():
            continue
        img = Image.open(png)
        im = OffsetImage(img, zoom=0.13)
        ab = AnnotationBbox(
            im,
            (fx, fy),
            xycoords=fig.transFigure,
            frameon=False,
            box_alignment=(0.5, 0.5),
        )
        fig.add_artist(ab)
        # leader from inset toward basin
        ax.annotate(
            "",
            xy=tip,
            xycoords=ax.transData,
            xytext=(fx, fy),
            textcoords=fig.transFigure,
            arrowprops=dict(arrowstyle="-", color="k", lw=0.8),
        )
        fval = basin_f(xy, gx, gy, fes, tip)
        fig.text(fx, fy - 0.11, f"F = {fval:.2f}", ha="center", fontsize=8)
        if xyz.exists():
            k, c = cn_hist(xyz)
            # tiny histogram just under the F label
            hx = fx - 0.07
            hy = fy - 0.20
            hax = fig.add_axes([hx, hy, 0.14, 0.07])
            hax.bar(k, c, color="k", width=0.7)
            hax.set_xlim(2.5, 12.5)
            hax.set_xticks(k)
            hax.tick_params(labelsize=5, length=1)
            hax.set_yticks([])
            for s in hax.spines.values():
                s.set_linewidth(0.4)

    fig.savefig(OUT / "lj38_fes.png", dpi=160, facecolor="white")
    fig.savefig(OUT / "lj38_fes.svg", facecolor="white")
    print("wrote", OUT / "lj38_fes.png")


if __name__ == "__main__":
    main()
