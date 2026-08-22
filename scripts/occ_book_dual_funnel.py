#!/usr/bin/env python3
"""Two-funnel SHEAP of a structure plane.

Points nearer GM (structure fingerprint) sit in a funnel whose origin is
the GM. Points nearer ico sit in a second funnel. Radius inside each
funnel scales with E-E_GM so the motif is at the centre of its basin
and liquid is the mouth. Field is IDW energy.
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
CAND = BOOK / "cand-sheap"
ENERGY = BOOK / "lj38.energy"
HD = BOOK / "cand-dpair" / "hd703.npy"
XY = BOOK / "cand-geostruc" / "asinh.xy"
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


def fill(xy, z, ngrid=160, k=6):
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


def dual_funnel(xy, hd, z, gm, ico, sep=1.15, floor=0.10):
    d_gm = np.linalg.norm(xy - xy[gm], axis=1)
    d_ico = np.linalg.norm(xy - xy[ico], axis=1)
    nearer_gm = d_gm <= d_ico
    zmax = float(np.percentile(z, 95))
    scale = floor + (1.0 - floor) * np.clip(z / max(zmax, 1e-9), 0.0, 1.6)
    out = np.empty_like(xy)
    left = np.array([-sep, 0.0])
    right = np.array([sep, 0.0])
    out[nearer_gm] = (xy[nearer_gm] - xy[gm]) * scale[nearer_gm, None] + left
    out[~nearer_gm] = (xy[~nearer_gm] - xy[ico]) * scale[~nearer_gm, None] + right
    return out, nearer_gm


def main() -> None:
    CAND.mkdir(parents=True, exist_ok=True)
    energy = np.loadtxt(ENERGY)
    gm = int(np.argmin(energy))
    ico = int(np.argmin(np.abs(energy - ICO_E)))
    z = energy - float(energy[gm])
    xy0 = np.loadtxt(XY)
    hd = np.load(HD)
    xy, nearer_gm = dual_funnel(xy0, hd, z, gm, ico)
    np.savetxt(CAND / "dual_funnel.xy", xy)
    print(f"n_GM_funnel={int(nearer_gm.sum())} n_ico_funnel={int((~nearer_gm).sum())}")
    gx, gy, field = fill(xy, z)
    vmax = 5.0
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    mesh = ax.pcolormesh(gx, gy, np.clip(field, 0, vmax), cmap=PES, shading="auto", vmin=0, vmax=vmax)
    ax.scatter(xy[:, 0], xy[:, 1], c=np.clip(z, 0, vmax), s=7, cmap=PES, vmin=0, vmax=vmax, edgecolors="none", alpha=0.55, zorder=20)
    ax.scatter(xy[gm, 0], xy[gm, 1], s=180, marker="*", c="k", edgecolors="white", linewidths=0.7, zorder=50, label=rf"GM ${GM_E:.3f}$")
    ax.scatter(xy[ico, 0], xy[ico, 1], s=100, marker="D", c="k", edgecolors="white", linewidths=0.7, zorder=50, label=rf"ico ${ICO_E:.3f}$")
    ax.legend(loc="upper right", fontsize=8, frameon=True, fancybox=False, framealpha=1.0, facecolor="white", edgecolor="k")
    ax.set_xlabel(r"dual-funnel $_1$")
    ax.set_ylabel(r"dual-funnel $_2$")
    ax.set_title("LJ38 structure  dual-funnel SHEAP  energy IDW")
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(mesh, ax=ax, label=r"$E-E_\mathrm{GM}$")
    fig.tight_layout()
    fig.savefig(OUT / "elja_occ_lj38_dual_funnel.png", dpi=180)
    fig.savefig(CAND / "dual_funnel.png", dpi=180)
    plt.close(fig)
    ix = int(np.clip(np.searchsorted(gx, xy[gm, 0]) - 1, 0, field.shape[1] - 1))
    iy = int(np.clip(np.searchsorted(gy, xy[gm, 1]) - 1, 0, field.shape[0] - 1))
    jx = int(np.clip(np.searchsorted(gx, xy[ico, 0]) - 1, 0, field.shape[1] - 1))
    jy = int(np.clip(np.searchsorted(gy, xy[ico, 1]) - 1, 0, field.shape[0] - 1))
    print(f"F(GM)={field[iy, ix]:.4f} F(ico)={field[jy, jx]:.4f}")
    print(f"wrote {OUT / 'elja_occ_lj38_dual_funnel.png'}")


if __name__ == "__main__":
    main()
