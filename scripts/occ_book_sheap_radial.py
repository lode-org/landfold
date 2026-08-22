#!/usr/bin/env python3
"""SHEAP-style radial energy layout of an existing structure plane.

Translate so GM is at the origin. Scale each radius by a monotone
function of E-E_GM so high-energy liquid is pushed out and both
low-energy motifs sit near the centre. Field is IDW energy.
"""

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
CAND = BOOK / "cand-sheap"
GM_E = -173.928427
ICO_E = -173.252378
PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)

SOURCES = [
    (BOOK / "cand-geostruc" / "asinh.xy", "elja_occ_lj38_sheap_geo.png", "structure geodesic  SHEAP radial  energy IDW"),
    (BOOK / "cand-cm" / "asinh.xy", "elja_occ_lj38_sheap_cm.png", "Coulomb-eig  SHEAP radial  energy IDW"),
    (BOOK / "cand-dpair" / "pair703_asinh.xy", "elja_occ_lj38_sheap_dpair.png", "pair-distance  SHEAP radial  energy IDW"),
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


def sheap_radial(xy: np.ndarray, z: np.ndarray, gm: int, floor: float = 0.12) -> np.ndarray:
    out = xy - xy[gm]
    zmax = float(np.percentile(z, 95))
    scale = floor + (1.0 - floor) * np.clip(z / max(zmax, 1e-9), 0.0, 1.5)
    return out * scale[:, None]


def draw(xy, z, gm, ico, path: Path, title: str) -> None:
    gx, gy, field = fill(xy, z)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    vmax = 5.0
    mesh = ax.pcolormesh(gx, gy, np.clip(field, 0, vmax), cmap=PES, shading="auto", vmin=0, vmax=vmax)
    ax.scatter(xy[:, 0], xy[:, 1], c=np.clip(z, 0, vmax), s=7, cmap=PES, vmin=0, vmax=vmax, edgecolors="none", alpha=0.55, zorder=20)
    ax.scatter(xy[gm, 0], xy[gm, 1], s=170, marker="*", c="k", edgecolors="white", linewidths=0.7, zorder=50, label=rf"GM ${GM_E:.3f}$")
    ax.scatter(xy[ico, 0], xy[ico, 1], s=95, marker="D", c="k", edgecolors="white", linewidths=0.7, zorder=50, label=rf"ico ${ICO_E:.3f}$")
    ax.legend(loc="best", fontsize=8, frameon=True, fancybox=False, framealpha=1.0, facecolor="white", edgecolor="k")
    ax.set_xlabel(r"SHEAP$_1$")
    ax.set_ylabel(r"SHEAP$_2$")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(mesh, ax=ax, label=r"$E-E_\mathrm{GM}$")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    ix = int(np.clip(np.searchsorted(gx, xy[gm, 0]) - 1, 0, field.shape[1] - 1))
    iy = int(np.clip(np.searchsorted(gy, xy[gm, 1]) - 1, 0, field.shape[0] - 1))
    jx = int(np.clip(np.searchsorted(gx, xy[ico, 0]) - 1, 0, field.shape[1] - 1))
    jy = int(np.clip(np.searchsorted(gy, xy[ico, 1]) - 1, 0, field.shape[0] - 1))
    print(f"{path.name} F(GM)={field[iy, ix]:.4f} F(ico)={field[jy, jx]:.4f} r_ico={np.linalg.norm(xy[ico]):.4f}")


def main() -> None:
    CAND.mkdir(parents=True, exist_ok=True)
    energy = np.loadtxt(ENERGY)
    gm = int(np.argmin(energy))
    ico = int(np.argmin(np.abs(energy - ICO_E)))
    z = energy - float(energy[gm])
    for src, name, title in SOURCES:
        if not src.exists():
            print("missing", src)
            continue
        xy = sheap_radial(np.loadtxt(src), z, gm)
        np.savetxt(CAND / (src.stem + "_sheap.xy"), xy)
        draw(xy, z, gm, ico, OUT / name, title)
        draw(xy, z, gm, ico, CAND / name, title)


if __name__ == "__main__":
    main()
