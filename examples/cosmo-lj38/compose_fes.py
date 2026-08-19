#!/usr/bin/env python3
"""
Compose the LJ38 teaching figure in the JCTC 2013 panel class
(Ceriotti, Tribello, Parrinello, J. Chem. Theory Comput. 9, 1521 (2013),
https://doi.org/10.1021/ct3010563): filled F/eps blob, leader lines to
xyzrender frames, F labels, integer CN 3-12 bars.

The public teaching zip and lab-cosmo/sampling-tutorial ship ts.all
(exercise 5.5 TSE) and not the long out.all MD used for the paper
landscape. This figure is a Gaussian KDE of the TSE embedding, laid
out as that panel. It is not a 1+4 matplotlib gallery and not a
white-field scatter of TSE points.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
KT = 0.168
CN_BINS = np.arange(3, 13)

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


def load_ts_cv(path: Path) -> np.ndarray:
    raw = np.loadtxt(path)
    return np.atleast_2d(raw)[:, 2:12]


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes so the FES is one connected body."""
    ny, nx = mask.shape
    reach = np.zeros_like(mask, dtype=bool)
    stack = []
    for i in range(ny):
        if not mask[i, 0]:
            stack.append((i, 0))
        if not mask[i, nx - 1]:
            stack.append((i, nx - 1))
    for j in range(nx):
        if not mask[0, j]:
            stack.append((0, j))
        if not mask[ny - 1, j]:
            stack.append((ny - 1, j))
    while stack:
        i, j = stack.pop()
        if i < 0 or j < 0 or i >= ny or j >= nx or reach[i, j] or mask[i, j]:
            continue
        reach[i, j] = True
        stack.extend(((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)))
    return mask | (~mask & ~reach)


def _keep_largest(mask: np.ndarray) -> np.ndarray:
    """Drop detached islands so the map is one body."""
    ny, nx = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = None
    best_n = 0
    for i0 in range(ny):
        for j0 in range(nx):
            if not mask[i0, j0] or seen[i0, j0]:
                continue
            stack = [(i0, j0)]
            cells = []
            while stack:
                i, j = stack.pop()
                if i < 0 or j < 0 or i >= ny or j >= nx:
                    continue
                if seen[i, j] or not mask[i, j]:
                    continue
                seen[i, j] = True
                cells.append((i, j))
                stack.extend(((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)))
            if len(cells) > best_n:
                best_n = len(cells)
                best = cells
    out = np.zeros_like(mask)
    if best:
        for i, j in best:
            out[i, j] = True
    return out


def _gauss1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _blur2d(z: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(int(np.ceil(3.0 * sigma)), 1)
    k = _gauss1d(sigma, radius)
    pad = np.pad(z, ((0, 0), (radius, radius)), mode="constant")
    tmp = np.empty_like(z)
    for i in range(z.shape[0]):
        tmp[i] = np.convolve(pad[i], k, mode="valid")
    pad = np.pad(tmp, ((radius, radius), (0, 0)), mode="constant")
    out = np.empty_like(z)
    for j in range(z.shape[1]):
        out[:, j] = np.convolve(pad[:, j], k, mode="valid")
    return out


def kde_fes(xy: np.ndarray, ngrid: int = 240, pad: float = 0.10):
    """Histogram plus Gaussian blur on every embedded point.

    Equivalent to a binned KDE: F = -kT ln(rho/rhomax) on a filled body.
    """
    x = xy[:, 0]
    y = xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= pad * dx
    xmax += pad * dx
    ymin -= pad * dy
    ymax += pad * dy
    counts, xedges, yedges = np.histogram2d(
        x, y, bins=ngrid, range=[[xmin, xmax], [ymin, ymax]]
    )
    # histogram2d is (nx, ny); contourf wants (ny, nx)
    rho = _blur2d(counts.T, sigma=7.0)
    gx = 0.5 * (xedges[:-1] + xedges[1:])
    gy = 0.5 * (yedges[:-1] + yedges[1:])
    rmax = float(rho.max())
    if rmax <= 0.0:
        raise SystemExit("empty projection")
    mask = rho > 0.006 * rmax
    mask = _keep_largest(_fill_mask_holes(mask))
    fes = np.full_like(rho, np.nan)
    on = mask & (rho > 0.0)
    fes[on] = -KT * np.log(np.clip(rho[on] / rmax, 1e-12, 1.0))
    hole = mask & ~on
    fes[hole] = 2.0
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
    counts, _ = np.histogram(cn, bins=np.arange(3, 14))
    return CN_BINS, counts


def cn_vector(xyz: Path) -> np.ndarray:
    """Integer n4..n13 counts, aligned with ts.all columns 3-12."""
    _bins, counts = cn_hist(xyz)
    vec = np.zeros(10)
    # counts[0] is CN=3; n4 is counts[1]
    vec[:9] = counts[1:]
    return vec


def match_tip(xy: np.ndarray, tscv: np.ndarray, vec: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(tscv - vec[None, :], axis=1)
    take = np.argpartition(dist, 20)[:20]
    return xy[take].mean(axis=0)


def fes_at(gx, gy, fes, pt) -> float:
    ix = int(np.argmin(np.abs(gx - pt[0])))
    iy = int(np.argmin(np.abs(gy - pt[1])))
    v = fes[iy, ix]
    if np.isnan(v):
        # nearest finite cell
        yy, xx = np.where(np.isfinite(fes))
        if len(xx) == 0:
            return 0.0
        j = int(np.argmin((gx[xx] - pt[0]) ** 2 + (gy[yy] - pt[1]) ** 2))
        v = fes[yy[j], xx[j]]
    return float(v)


def load_frame(png: Path) -> Image.Image:
    img = Image.open(png).convert("RGBA")
    arr = np.asarray(img).copy()
    ink = arr[:, :, :3].astype(np.int16)
    dark = ink.max(axis=2) < 28
    arr[dark, 3] = 0
    return Image.fromarray(arr)


def main():
    proj = OUT / "ts.proj"
    if not proj.exists():
        raise SystemExit("run ./run.sh first")
    xy = load_xy(proj)
    gx, gy, fes = kde_fes(xy)
    tscv = load_ts_cv(ROOT / "ts.all")
    if tscv.shape[0] != xy.shape[0]:
        raise SystemExit("ts.all and ts.proj row counts differ")

    fig = plt.figure(figsize=(13.0, 9.8), facecolor="white")
    ax = fig.add_axes([0.22, 0.20, 0.56, 0.64])
    ax.set_facecolor("white")
    mesh = ax.contourf(
        gx, gy, fes, levels=np.linspace(0, 2, 21), cmap=CMAP, extend="max"
    )
    ax.contour(
        gx,
        gy,
        fes,
        levels=np.linspace(0.15, 1.85, 12),
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
    cb.set_label(r"$F/\epsilon$", fontsize=12)
    cb.set_ticks([0, 0.5, 1.0, 1.5, 2.0])

    # Four teaching snapshots. Positions leave room for CN 3-12 bars.
    panels = [
        ("lj38_ico.png", "lj38_ico.xyz", (0.84, 0.78)),
        ("lj38_fcc.png", "lj38_fcc.xyz", (0.84, 0.34)),
        ("lj38.19.png", "lj38.19.xyz", (0.14, 0.78)),
        ("lj38.17.png", "lj38.17.xyz", (0.14, 0.34)),
    ]
    for png_name, xyz_name, (fx, fy) in panels:
        png = OUT / png_name
        xyz = ROOT / xyz_name
        if not png.exists() or not xyz.exists():
            continue
        tip = match_tip(xy, tscv, cn_vector(xyz))
        img = load_frame(png)
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
        fval = fes_at(gx, gy, fes, tip)
        k, c = cn_hist(xyz)
        hax = fig.add_axes([fx - 0.08, fy - 0.24, 0.16, 0.08])
        hax.bar(k, c, color="k", width=0.7)
        hax.set_xlim(2.5, 12.5)
        hax.set_xticks(CN_BINS)
        hax.set_xticklabels([str(int(v)) for v in CN_BINS])
        hax.tick_params(labelsize=6, length=2)
        hax.set_yticks([])
        hax.set_title(f"F = {fval:.2f}", fontsize=9, pad=2)
        for spine in hax.spines.values():
            spine.set_linewidth(0.45)

    ticks = [int(v) for v in CN_BINS]
    if ticks != list(range(3, 13)):
        raise SystemExit(f"CN ticks must be 3-12, got {ticks}")
    finite = int(np.isfinite(fes).sum())
    fig.savefig(OUT / "lj38_fes.png", dpi=170, facecolor="white")
    fig.savefig(OUT / "lj38_fes.svg", facecolor="white")
    fig.savefig(ROOT / "lj38_fes.png", dpi=170, facecolor="white")
    print("wrote", ROOT / "lj38_fes.png")
    print(f"fes finite cells {finite}/{fes.size}; cn ticks {ticks[0]}-{ticks[-1]}")


if __name__ == "__main__":
    main()
