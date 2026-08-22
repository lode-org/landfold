#!/usr/bin/env python3
"""Compact-support kNN IDW of E-E_GM on the 703-vector asinh MDS plane.

Re-paints pair703_asinh.xy with the same IDW kernel as occ_book_soap_idw.py.
Interpolating IMQ on this plane invents many fake wells. If the GM teal
well (F=0) and ico blue well (F=0.676) survive IDW, those points are
low E; if they vanish, IMQ invented them. No occupancy invert.
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
XY = BOOK / "cand-dpair" / "pair703_asinh.xy"
ENERGY = BOOK / "lj38.energy"
FIG = OUT / "elja_occ_lj38_dpair_idw.png"
CAND = BOOK / "cand-dpair"
KS = (1, 4, 8)
GM_E = -173.928427
ICO_E = -173.252378
GM_IDX = 0
ICO_IDX = 40
# Teal / blue on ruhi_pes. True ico-GM is 0.676.
GM_WELL = 0.55
ICO_WELL = 1.50
VMAX = 5.0
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


def ring_barrier(xy, gx, gy, field, idx: int, r_in: float, r_out: float) -> float:
    c = xy[idx]
    xx, yy = np.meshgrid(gx, gy)
    r = np.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2)
    ring = field[(r >= r_in) & (r <= r_out)]
    ring = ring[np.isfinite(ring)]
    if ring.size == 0:
        return float("nan")
    return float(np.mean(ring) - sample(gx, gy, field, c))


def paint(ax, xy, z, gm, ico, gx, gy, field, title: str, legend: bool) -> object:
    mesh = ax.pcolormesh(
        gx, gy, np.clip(field, 0, VMAX), cmap=PES, shading="auto", vmin=0, vmax=VMAX
    )
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=np.clip(z, 0, VMAX),
        s=7,
        cmap=PES,
        vmin=0,
        vmax=VMAX,
        edgecolors="none",
        alpha=0.55,
        zorder=20,
    )
    ax.scatter(
        xy[gm, 0],
        xy[gm, 1],
        s=170,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.7,
        zorder=50,
        label=rf"GM ${GM_E:.3f}$",
    )
    ax.scatter(
        xy[ico, 0],
        xy[ico, 1],
        s=95,
        marker="D",
        c="k",
        edgecolors="white",
        linewidths=0.7,
        zorder=50,
        label=rf"ico ${ICO_E:.3f}$",
    )
    if legend:
        ax.legend(
            loc="best",
            fontsize=8,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            facecolor="white",
            edgecolor="k",
        )
    ax.set_xlabel(r"$\chi_1$")
    ax.set_ylabel(r"$\chi_2$")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    return mesh


def main() -> None:
    if not XY.exists():
        raise SystemExit(f"missing {XY}")
    energy = np.loadtxt(ENERGY)
    xy = np.loadtxt(XY)
    if xy.shape[0] != energy.shape[0]:
        raise SystemExit(f"row mismatch xy={xy.shape[0]} energy={energy.shape[0]}")
    gm = GM_IDX
    ico = ICO_IDX
    if abs(float(energy[gm]) - GM_E) > 1e-3 or abs(float(energy[ico]) - ICO_E) > 1e-3:
        raise SystemExit(f"index check failed E[{gm}]={energy[gm]} E[{ico}]={energy[ico]}")
    z = energy - float(energy[gm])
    nn_dist, nn_idx = knn(xy, xy[gm : gm + 1], k=2)
    nn_e = float(z[int(nn_idx[0, 1])])
    print(
        f"NN_E(GM)={nn_e:.6f}  nn_idx={int(nn_idx[0, 1])}  "
        f"nn_dist={float(nn_dist[0, 1]):.3e}  z[GM]={float(z[gm]):.6f}  z[ico]={float(z[ico]):.6f}"
    )

    fig, axes = plt.subplots(1, len(KS), figsize=(16.8, 5.4))
    mesh = None
    for ax, k in zip(axes, KS):
        gx, gy, field = fill(xy, z, k=k)
        fgm = sample(gx, gy, field, xy[gm])
        fico = sample(gx, gy, field, xy[ico])
        diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
        bar_gm = ring_barrier(xy, gx, gy, field, gm, 0.06 * diam, 0.16 * diam)
        bar_ico = ring_barrier(xy, gx, gy, field, ico, 0.06 * diam, 0.16 * diam)
        gm_ok = bool(np.isfinite(fgm) and fgm <= GM_WELL)
        ico_ok = bool(np.isfinite(fico) and fico <= ICO_WELL)
        survive = gm_ok and ico_ok
        print(f"k={k} F(GM)={fgm:.4f} F(ico)={fico:.4f}")
        print(
            f"k={k} NN_E(GM)={nn_e:.4f} "
            f"bar_GM={bar_gm:.4f} bar_ico={bar_ico:.4f} "
            f"gm_well={'yes' if gm_ok else 'no'} ico_well={'yes' if ico_ok else 'no'} "
            f"survive={'yes' if survive else 'no'}"
        )
        mesh = paint(
            ax,
            xy,
            z,
            gm,
            ico,
            gx,
            gy,
            field,
            title=rf"$k={k}$",
            legend=(k == KS[-1]),
        )
    fig.suptitle(
        r"pair-distance 703 asinh MDS   compact-support kNN IDW of $E-E_{\mathrm{GM}}$",
        fontsize=12,
    )
    fig.colorbar(mesh, ax=list(axes), fraction=0.02, pad=0.02, label=r"$E-E_\mathrm{GM}$")
    fig.savefig(FIG, dpi=180, bbox_inches="tight")
    CAND.mkdir(parents=True, exist_ok=True)
    fig.savefig(CAND / FIG.name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
