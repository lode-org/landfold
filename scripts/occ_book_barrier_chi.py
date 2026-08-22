#!/usr/bin/env python3
"""asinh chi of the Wales superbasin barrier ultrametric.

HD distance is the Kruskal merge height on DECAF L1 neighbours
(barrier = max E + 2 L1), minus min(E_i, E_j). Same funnel => small D.
The two crystal funnels are far. Occupancy KDE on that plane cannot
put the GM in the middle of the ico well.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ceriotti-figs"
EX = ROOT / "examples" / "cosmo-lj38"
HIST = Path("/tmp/occ-book/lj38_decaf_e.hist")
DIST = Path("/tmp/occ-book/lj38_barrier.dist")

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)
L1_CUT = 0.55
L1_BARRIER = 2.0
GM_E = -173.928427
ICO_E = -173.252378
KT = 0.168


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True


def load_book(path: Path):
    wells, emin, rows = [], [], []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        wells.append(float(p[1]))
        emin.append(float(p[2]))
        rows.append([float(x) for x in p[3:]])
    return np.asarray(rows), np.asarray(emin), np.asarray(wells)


def barrier_matrix(hist: np.ndarray, energy: np.ndarray) -> np.ndarray:
    n = len(energy)
    edges = []
    for i in range(n):
        drow = np.abs(hist - hist[i]).sum(1)
        for j in range(i + 1, n):
            if drow[j] <= L1_CUT:
                h = float(max(energy[i], energy[j]) + L1_BARRIER * drow[j])
                edges.append((h, i, j))
    edges.sort()
    uf = UnionFind(n)
    members = {i: [i] for i in range(n)}
    merge = np.full((n, n), np.nan)
    for h, i, j in edges:
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        left, right = members[ri], members[rj]
        for a in left:
            for b in right:
                merge[a, b] = merge[b, a] = h
        uf.union(i, j)
        root = uf.find(i)
        members[root] = left + right
        if root != ri:
            members.pop(ri, None)
        if root != rj:
            members.pop(rj, None)
    np.fill_diagonal(merge, 0.0)
    # leftover disconnected pairs: diameter
    if np.isnan(merge).any():
        cap = float(np.nanmax(merge)) + 1.0
        merge = np.where(np.isnan(merge), cap, merge)
    excess = merge - np.minimum.outer(energy, energy)
    excess = np.clip(excess, 0.0, None)
    np.fill_diagonal(excess, 0.0)
    return excess


def torgerson(dist: np.ndarray, dim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    n = dist.shape[0]
    d2 = dist * dist
    h = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * h @ d2 @ h
    w, v = np.linalg.eigh(b)
    idx = np.argsort(w)[::-1]
    w, v = w[idx], v[:, idx]
    lam = np.clip(w[:dim], 0.0, None)
    xy = v[:, :dim] * np.sqrt(lam)
    return xy, w[:4]


def _cf():
    spec = importlib.util.spec_from_file_location("compose_fes", EX / "compose_fes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def kde_fes(xy, weights, cf, ngrid=160, sigma=4.0):
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= 0.12 * dx
    xmax += 0.12 * dx
    ymin -= 0.12 * dy
    ymax += 0.12 * dy
    counts, xe, ye = np.histogram2d(
        x, y, bins=ngrid, range=[[xmin, xmax], [ymin, ymax]], weights=weights
    )
    rho = cf._blur2d(counts.T, sigma=sigma)
    gx = 0.5 * (xe[:-1] + xe[1:])
    gy = 0.5 * (ye[:-1] + ye[1:])
    rmax = float(rho.max())
    mask = cf._fill_mask_holes(rho > 0.004 * rmax)
    fes = np.full_like(rho, np.nan)
    on = mask & (rho > 0)
    fes[on] = -KT * np.log(np.clip(rho[on] / rmax, 1e-12, 1))
    return gx, gy, np.clip(fes, 0, 2)


def main() -> None:
    hist, energy, wells = load_book(HIST)
    dist = barrier_matrix(hist, energy)
    print("D(GM,ico)", dist[0, 1], "median D", float(np.median(dist[np.triu_indices(len(dist), 1)])))
    # asinh of D at sigma = median
    sig = float(np.median(dist[dist > 0]))
    sig = max(sig, 1e-6)
    fdist = np.arcsinh(dist / sig) / (2.0 * np.arcsinh(1.0))
    np.fill_diagonal(fdist, 0.0)
    xy, ev = torgerson(fdist, 2)
    print("Torgerson ev", ev, "y/x", xy.std(0)[1] / max(xy.std(0)[0], 1e-12))
    np.savetxt(DIST, dist, fmt="%.8e")
    print("wrote", DIST)

    cf = _cf()
    # occupancy: leftover wells; energy: champion E
    gx, gy, fes = kde_fes(xy, wells, cf)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), facecolor="white")
    mesh = axes[0].contourf(gx, gy, fes, levels=np.linspace(0, 2, 21), cmap=PES, extend="max")
    axes[0].contour(
        gx, gy, np.where(np.isfinite(fes), fes, np.nan),
        levels=np.linspace(0.15, 1.85, 10), colors="#1a1a2e", linewidths=0.35,
    )
    axes[0].scatter(xy[0, 0], xy[0, 1], s=120, marker="*", c="k", edgecolors="white", linewidths=0.6, zorder=6, label=rf"GM ${GM_E:.3f}$")
    axes[0].scatter(xy[1, 0], xy[1, 1], s=70, marker="D", c="k", edgecolors="white", linewidths=0.6, zorder=6, label=rf"ico ${ICO_E:.3f}$")
    axes[0].legend(fontsize=8, frameon=True, fancybox=False)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title(r"barrier-asinh $\chi$  leftover-well occupancy")
    for sp in axes[0].spines.values():
        sp.set_visible(False)
    fig.colorbar(mesh, ax=axes[0], fraction=0.046, pad=0.03).set_label(r"$F/\varepsilon$")

    rel = np.clip(energy - energy.min(), 0, 8)
    sc = axes[1].scatter(xy[:, 0], xy[:, 1], c=rel, s=22, cmap=PES, vmin=0, vmax=6, linewidths=0)
    axes[1].scatter(xy[0, 0], xy[0, 1], s=120, marker="*", c="k", edgecolors="white", linewidths=0.6, zorder=6)
    axes[1].scatter(xy[1, 0], xy[1, 1], s=70, marker="D", c="k", edgecolors="white", linewidths=0.6, zorder=6)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title(r"barrier-asinh $\chi$  champion $E-E_{\mathrm{GM}}$")
    for sp in axes[1].spines.values():
        sp.set_visible(False)
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.03).set_label(r"$E-E_{\mathrm{GM}}/\varepsilon$")
    dest = OUT / "elja_occ_lj38_barrier_chi.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=170, facecolor="white")
    print("wrote", dest, "GM", xy[0], "ico", xy[1], "sep", float(np.linalg.norm(xy[0] - xy[1])))
    print("F(GM)", cf.fes_at(gx, gy, fes, xy[0]), "F(ico)", cf.fes_at(gx, gy, fes, xy[1]))


if __name__ == "__main__":
    main()
