#!/usr/bin/env python3
"""NaN-safe energy-filled plots of Q4/Q6 and the structure-committor plane."""

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
QNPZ = BOOK / "cand-q6" / "q4q6.npz"
CNPZ = BOOK / "cand-hung" / "struct_committor.npz"
GM_E = -173.928427
ICO_E = -173.252378
PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)


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


def fill(xy, z, ngrid=160, k=8):
    pad = 0.10
    xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
    ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= pad * dx
    xmax += pad * dx
    ymin -= pad * dy
    ymax += pad * dy
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
    cutoff = 3.0 * float(np.median(pos)) if pos.size else 0.08 * span
    cutoff = max(cutoff, 0.06 * span)
    mask = nn < cutoff
    field = np.where(mask, field, np.nan)
    return gx, gy, field


def draw(path, xy, z, gm, ico, xlabel, ylabel, title):
    gx, gy, field = fill(xy, z)
    finite = field[np.isfinite(field)]
    vmax = max(float(np.percentile(finite, 90)) if finite.size else 2.0, 1.5)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    mesh = ax.pcolormesh(gx, gy, np.clip(field, 0, vmax), cmap=PES, shading="auto", vmin=0, vmax=vmax)
    ax.scatter(xy[:, 0], xy[:, 1], c=np.clip(z, 0, vmax), s=8, cmap=PES, vmin=0, vmax=vmax, edgecolors="none", alpha=0.65, zorder=20)
    ax.scatter(xy[gm, 0], xy[gm, 1], s=170, marker="*", c="k", edgecolors="white", linewidths=0.7, zorder=50, label=rf"GM ${GM_E:.3f}$")
    ax.scatter(xy[ico, 0], xy[ico, 1], s=95, marker="D", c="k", edgecolors="white", linewidths=0.7, zorder=50, label=rf"ico ${ICO_E:.3f}$")
    ax.legend(loc="best", fontsize=8, frameon=True, fancybox=False, framealpha=1.0, facecolor="white", edgecolor="k")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label=r"$E-E_\mathrm{GM}$")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    # sample field at GM / ico
    ix = int(np.clip(np.searchsorted(gx, xy[gm, 0]) - 1, 0, field.shape[1] - 1))
    iy = int(np.clip(np.searchsorted(gy, xy[gm, 1]) - 1, 0, field.shape[0] - 1))
    jx = int(np.clip(np.searchsorted(gx, xy[ico, 0]) - 1, 0, field.shape[1] - 1))
    jy = int(np.clip(np.searchsorted(gy, xy[ico, 1]) - 1, 0, field.shape[0] - 1))
    print(f"{path.name} F(GM)={field[iy, ix]:.4f} F(ico)={field[jy, jx]:.4f} n_finite={finite.size}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    q = np.load(QNPZ)
    energy, q4, q6 = q["energy"], q["q4"], q["q6"]
    gm, ico = int(q["gm"]), int(q["ico"])
    z = energy - float(energy[gm])
    print(f"Q6 GM={q6[gm]:.4f} ico={q6[ico]:.4f} Q4 GM={q4[gm]:.4f} ico={q4[ico]:.4f}")
    draw(
        OUT / "elja_occ_lj38_q6q4.png",
        np.column_stack([q4, q6]),
        z,
        gm,
        ico,
        r"$Q_4$",
        r"$Q_6$",
        r"LJ38 inherent structures  $Q_4$–$Q_6$  energy",
    )
    draw(
        OUT / "elja_occ_lj38_q6E.png",
        np.column_stack([q6, z]),
        z,
        gm,
        ico,
        r"$Q_6$",
        r"$E-E_\mathrm{GM}$",
        r"LJ38 inherent structures  $Q_6$ vs energy",
    )
    if CNPZ.exists():
        c = np.load(CNPZ)
        xy = c["xy"]
        gm2, ico2 = int(c["gm"]), int(c["ico"])
        z2 = c["energy"] - float(c["energy"][gm2])
        draw(
            OUT / "elja_occ_lj38_struct_committor.png",
            xy,
            z2,
            gm2,
            ico2,
            r"$d(\mathrm{fp},\mathrm{GM})$",
            r"$d(\mathrm{fp},\mathrm{ico})$",
            "LJ38 structure committor  energy",
        )


if __name__ == "__main__":
    main()
