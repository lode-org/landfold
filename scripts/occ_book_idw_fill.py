#!/usr/bin/env python3
"""IDW energy fill of structure planes. NN/IDW, not leftover occupancy, not IMQ."""

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
ENERGY = BOOK / "lj38.energy"
GM_E = -173.928427
ICO_E = -173.252378
PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)

JOBS = [
    (BOOK / "cand-cm" / "asinh.xy", "elja_occ_lj38_cm_idw.png", "Coulomb-eig asinh  energy IDW"),
    (BOOK / "cand-cm" / "euclid.xy", "elja_occ_lj38_cm_euclid_idw.png", "Coulomb-eig Euclidean  energy IDW"),
    (BOOK / "cand-geostruc" / "asinh.xy", "elja_occ_lj38_geostruc_idw.png", "structure geodesic asinh  energy IDW"),
    (BOOK / "cand-dpair" / "pair703_asinh.xy", "elja_occ_lj38_dpair_idw.png", "pair-distance asinh  energy IDW"),
    (BOOK / "cand-dpair" / "pair703.xy", "elja_occ_lj38_dpair_raw_idw.png", "pair-distance MDS  energy IDW"),
    (BOOK / "cand-lfstruc" / "lfstruc.xy", "elja_occ_lj38_lfstruc_idw.png", "landfold structure  energy IDW"),
]


def knn(ref, query, k):
    chunk = 400
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


def fill(xy, z, ngrid=150, k=6):
    xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
    ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= 0.08 * dx
    xmax += 0.08 * dx
    ymin -= 0.08 * dy
    ymax += 0.08 * dy
    gx = np.linspace(xmin, xmax, ngrid)
    gy = np.linspace(ymin, ymax, ngrid)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    dist, idx = knn(xy, grid, k)
    w = 1.0 / np.clip(dist, 1e-9, None) ** 2
    w /= w.sum(axis=1, keepdims=True)
    field = (w * z[idx]).sum(axis=1).reshape(ngrid, ngrid)
    nn = dist[:, 0].reshape(ngrid, ngrid)
    nn_data = knn(xy, xy, k=2)[0][:, 1]
    pos = nn_data[nn_data > 1e-12]
    span = max(xmax - xmin, ymax - ymin)
    cutoff = max(4.0 * float(np.median(pos)) if pos.size else 0.08 * span, 0.07 * span)
    field = np.where(nn < cutoff, field, np.nan)
    return gx, gy, field


def sample(gx, gy, field, pt):
    if not np.isfinite(field).any():
        return float("nan")
    ix = int(np.clip(np.searchsorted(gx, pt[0]) - 1, 0, field.shape[1] - 1))
    iy = int(np.clip(np.searchsorted(gy, pt[1]) - 1, 0, field.shape[0] - 1))
    return float(field[iy, ix])


def draw(xy, z, gm, ico, path: Path, title: str) -> None:
    gx, gy, field = fill(xy, z)
    finite = field[np.isfinite(field)]
    vmax = 5.0
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    mesh = ax.pcolormesh(gx, gy, np.clip(field, 0, vmax), cmap=PES, shading="auto", vmin=0, vmax=vmax)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=np.clip(z, 0, vmax),
        s=7,
        cmap=PES,
        vmin=0,
        vmax=vmax,
        edgecolors="none",
        alpha=0.55,
        zorder=20,
    )
    ax.scatter(xy[gm, 0], xy[gm, 1], s=170, marker="*", c="k", edgecolors="white", linewidths=0.7, zorder=50, label=rf"GM ${GM_E:.3f}$")
    ax.scatter(xy[ico, 0], xy[ico, 1], s=95, marker="D", c="k", edgecolors="white", linewidths=0.7, zorder=50, label=rf"ico ${ICO_E:.3f}$")
    ax.legend(loc="best", fontsize=8, frameon=True, fancybox=False, framealpha=1.0, facecolor="white", edgecolor="k")
    ax.set_xlabel(r"$\chi_1$")
    ax.set_ylabel(r"$\chi_2$")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(mesh, ax=ax, label=r"$E-E_\mathrm{GM}$")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    fgm = sample(gx, gy, field, xy[gm])
    fico = sample(gx, gy, field, xy[ico])
    sep = float(np.linalg.norm(xy[gm] - xy[ico]) / max(np.linalg.norm(xy.max(0) - xy.min(0)), 1e-9))
    print(f"{path.name} F(GM)={fgm:.4f} F(ico)={fico:.4f} sep_norm={sep:.4f} nfin={finite.size}")


def main() -> None:
    energy = np.loadtxt(ENERGY)
    gm = int(np.argmin(energy))
    ico = int(np.argmin(np.abs(energy - ICO_E)))
    z = energy - float(energy[gm])
    for src, name, title in JOBS:
        if not src.exists():
            print("missing", src)
            continue
        xy = np.loadtxt(src)
        draw(xy, z, gm, ico, OUT / name, title)
        draw(xy, z, gm, ico, src.parent / name, title)


if __name__ == "__main__":
    main()
