#!/usr/bin/env python3
"""Energy-filled MDS of n4..n13 histograms computed from 4042 xyz."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ceriotti-figs"
BOOK = Path("/tmp/occ-book")
CAND = BOOK / "cand-cnxyz"
HIST = BOOK / "lj38_cnxyz.hist"
ENERGY = BOOK / "lj38_cnxyz.energy"
GM_E = -173.928427
ICO_E = -173.252378
PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)


def pairwise_l2(x: np.ndarray) -> np.ndarray:
    gram = x @ x.T
    nrm = np.einsum("ij,ij->i", x, x)
    d2 = nrm[:, None] + nrm[None, :] - 2.0 * gram
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def torgerson(d: np.ndarray, dim: int = 2) -> np.ndarray:
    n = d.shape[0]
    d2 = d * d
    j = np.eye(n) - np.full((n, n), 1.0 / n)
    b = -0.5 * j @ d2 @ j
    evals, evecs = np.linalg.eigh(b)
    order = np.argsort(evals)[::-1]
    ev = np.clip(evals[order[:dim]], 0.0, None)
    return evecs[:, order[:dim]] * np.sqrt(ev)


def knn(ref: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    chunk = 512
    dists = np.empty((query.shape[0], k), dtype=np.float64)
    idxs = np.empty((query.shape[0], k), dtype=np.int64)
    for start in range(0, query.shape[0], chunk):
        stop = min(start + chunk, query.shape[0])
        d2 = ((query[start:stop, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
        part = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        take = np.take_along_axis(d2, part, axis=1)
        order = np.argsort(take, axis=1)
        idxs[start:stop] = np.take_along_axis(part, order, axis=1)
        dists[start:stop] = np.sqrt(np.take_along_axis(take, order, axis=1))
    return dists, idxs


def energy_fill(xy: np.ndarray, energy: np.ndarray, ngrid: int = 180, k: int = 8):
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= 0.12 * dx
    xmax += 0.12 * dx
    ymin -= 0.12 * dy
    ymax += 0.12 * dy
    gx = np.linspace(xmin, xmax, ngrid)
    gy = np.linspace(ymin, ymax, ngrid)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    dist, idx = knn(xy, grid, k=k)
    w = 1.0 / np.clip(dist, 1e-9, None) ** 2
    w /= w.sum(axis=1, keepdims=True)
    field = (w * energy[idx]).sum(axis=1).reshape(ngrid, ngrid)
    nn = dist[:, 0].reshape(ngrid, ngrid)
    cutoff = 2.5 * float(np.median(knn(xy, xy, k=2)[0][:, 1]))
    field = np.where(nn < cutoff, field, np.nan)
    return gx, gy, field


def mark(ax, gm_xy, ico_xy):
    ax.scatter(gm_xy[0], gm_xy[1], s=140, marker="*", c="k", edgecolors="white", linewidths=0.6, zorder=50, label=rf"GM ${GM_E:.3f}$")
    ax.scatter(ico_xy[0], ico_xy[1], s=80, marker="D", c="k", edgecolors="white", linewidths=0.6, zorder=50, label=rf"ico ${ICO_E:.3f}$")
    ax.legend(loc="best", fontsize=8, frameon=True, fancybox=False, framealpha=1.0, facecolor="white", edgecolor="k")


def draw(name: str, xy: np.ndarray, z: np.ndarray, gm: int, ico: int) -> None:
    gx, gy, field = energy_fill(xy, z)
    vmax = max(float(np.nanpercentile(field, 92)), 2.0)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    levels = np.linspace(0.0, vmax, 21)
    mesh = ax.contourf(gx, gy, field, levels=levels, cmap=PES, extend="max")
    ax.contour(gx, gy, np.where(np.isfinite(field), field, np.nan), levels=np.linspace(0.15 * vmax, 0.95 * vmax, 10), colors="k", linewidths=0.25)
    ax.scatter(xy[:, 0], xy[:, 1], c=z, s=6, cmap=PES, vmin=0, vmax=vmax, edgecolors="none", alpha=0.55, zorder=20)
    mark(ax, xy[gm], xy[ico])
    ax.set_xlabel(r"MDS$_1$")
    ax.set_ylabel(r"MDS$_2$")
    ax.set_title(f"LJ38 n4..n13 from xyz  {name}  energy IDW")
    fig.colorbar(mesh, ax=ax, label=r"$E-E_\mathrm{GM}$")
    fig.tight_layout()
    fig.savefig(OUT / f"elja_occ_lj38_{name}.png", dpi=180)
    fig.savefig(CAND / f"{name}.png", dpi=180)
    plt.close(fig)
    fgm = field[int(np.clip(np.searchsorted(gy, xy[gm, 1]) - 1, 0, field.shape[0] - 1)), int(np.clip(np.searchsorted(gx, xy[gm, 0]) - 1, 0, field.shape[1] - 1))]
    fico = field[int(np.clip(np.searchsorted(gy, xy[ico, 1]) - 1, 0, field.shape[0] - 1)), int(np.clip(np.searchsorted(gx, xy[ico, 0]) - 1, 0, field.shape[1] - 1))]
    sep = float(np.linalg.norm(xy[gm] - xy[ico]) / max(np.linalg.norm(xy.max(0) - xy.min(0)), 1e-9))
    print(f"{name} F(GM)={fgm:.4f} F(ico)={fico:.4f} sep_norm={sep:.4f}")


def main() -> None:
    CAND.mkdir(parents=True, exist_ok=True)
    hist = np.loadtxt(HIST)
    energy = np.loadtxt(ENERGY)
    gm = int(np.argmin(energy))
    ico = int(np.argmin(np.abs(energy - ICO_E)))
    z = energy - float(energy[gm])
    d = pairwise_l2(hist)
    xy = torgerson(d)
    med = float(np.median(d[d > 0]))
    xy_a = torgerson(np.arcsinh(d / med) / (2.0 * np.arcsinh(1.0)))
    np.savez(CAND / "xy.npz", raw=xy, asinh=xy_a, energy=energy, gm=gm, ico=ico)
    draw("cnxyz", xy, z, gm, ico)
    draw("cnxyz_asinh", xy_a, z, gm, ico)
    # structure-committor analogue: L2 to GM vs L2 to ico in hist space
    d_gm = np.linalg.norm(hist - hist[gm], axis=1)
    d_ico = np.linalg.norm(hist - hist[ico], axis=1)
    xy_c = np.column_stack([d_gm, d_ico])
    draw("cnxyz_committor", xy_c, z, gm, ico)


if __name__ == "__main__":
    main()
