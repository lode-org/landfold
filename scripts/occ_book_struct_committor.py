#!/usr/bin/env python3
"""Structure-committor plane: distance-to-GM vs distance-to-ico.

HD is the 703-vector sorted internuclear list (permutation invariant).
The plane is already two basins if the motifs differ. Field is E-E_GM.
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
CAND = BOOK / "cand-hung"
HD = BOOK / "cand-dpair" / "hd703.npy"
ENERGY = BOOK / "cand-dpair" / "energy.npy"
ENERGY_FALLBACK = BOOK / "lj38.energy"
GM_E = -173.928427
ICO_E = -173.252378
PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)


def knn(ref, query, k):
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


def energy_fill(xy, energy, ngrid=200, k=8):
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= 0.10 * dx
    xmax += 0.10 * dx
    ymin -= 0.10 * dy
    ymax += 0.10 * dy
    gx = np.linspace(xmin, xmax, ngrid)
    gy = np.linspace(ymin, ymax, ngrid)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    dist, idx = knn(xy, grid, k)
    w = 1.0 / np.clip(dist, 1e-9, None) ** 2
    w /= w.sum(axis=1, keepdims=True)
    field = (w * energy[idx]).sum(axis=1).reshape(ngrid, ngrid)
    nn = dist[:, 0].reshape(ngrid, ngrid)
    cutoff = 2.2 * float(np.median(knn(xy, xy, k=2)[0][:, 1]))
    field = np.where(nn < cutoff, field, np.nan)
    return gx, gy, field


def main() -> None:
    CAND.mkdir(parents=True, exist_ok=True)
    hd = np.load(HD)
    energy = np.load(ENERGY) if ENERGY.exists() else np.loadtxt(ENERGY_FALLBACK)
    gm = int(np.argmin(energy))
    ico = int(np.argmin(np.abs(energy - ICO_E)))
    z = energy - float(energy[gm])
    d_gm = np.linalg.norm(hd - hd[gm], axis=1)
    d_ico = np.linalg.norm(hd - hd[ico], axis=1)
    xy = np.column_stack([d_gm, d_ico])
    print(f"n={len(energy)} GM {gm} E={energy[gm]:.6f} ico {ico} E={energy[ico]:.6f}")
    print(f"d(GM,ico)={d_gm[ico]:.4f}  d_gm median={np.median(d_gm):.4f} d_ico median={np.median(d_ico):.4f}")
    np.savez(CAND / "struct_committor.npz", xy=xy, energy=energy, gm=gm, ico=ico, d_gm=d_gm, d_ico=d_ico)

    gx, gy, field = energy_fill(xy, z)
    vmax = max(float(np.nanpercentile(field, 90)), 1.5)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    levels = np.linspace(0.0, vmax, 21)
    mesh = ax.contourf(gx, gy, field, levels=levels, cmap=PES, extend="max")
    ax.contour(
        gx,
        gy,
        np.where(np.isfinite(field), field, np.nan),
        levels=np.linspace(0.12 * vmax, 0.95 * vmax, 10),
        colors="k",
        linewidths=0.25,
    )
    ax.scatter(xy[:, 0], xy[:, 1], c=z, s=7, cmap=PES, vmin=0, vmax=vmax, edgecolors="none", alpha=0.6, zorder=20)
    ax.scatter(xy[gm, 0], xy[gm, 1], s=160, marker="*", c="k", edgecolors="white", linewidths=0.7, zorder=50, label=rf"GM ${GM_E:.3f}$")
    ax.scatter(xy[ico, 0], xy[ico, 1], s=90, marker="D", c="k", edgecolors="white", linewidths=0.7, zorder=50, label=rf"ico ${ICO_E:.3f}$")
    ax.legend(loc="upper right", fontsize=8, frameon=True, fancybox=False, framealpha=1.0, facecolor="white", edgecolor="k")
    ax.set_xlabel(r"$d(\chi,\mathrm{GM})$  pair-distance fingerprint")
    ax.set_ylabel(r"$d(\chi,\mathrm{ico})$")
    ax.set_title("LJ38 structure committor  energy IDW")
    fig.colorbar(mesh, ax=ax, label=r"$E-E_\mathrm{GM}$")
    fig.tight_layout()
    fig.savefig(OUT / "elja_occ_lj38_struct_committor.png", dpi=180)
    fig.savefig(CAND / "struct_committor.png", dpi=180)
    plt.close(fig)

    # also Q6-free two-axis: assign basin by nearer prototype
    nearer_gm = d_gm < d_ico
    print(f"nearer GM {int(nearer_gm.sum())} nearer ico {int((~nearer_gm).sum())}")
    print(f"mean E nearer GM {z[nearer_gm].mean():.3f} nearer ico {z[~nearer_gm].mean():.3f}")
    print(f"wrote {OUT / 'elja_occ_lj38_struct_committor.png'}")


if __name__ == "__main__":
    main()
