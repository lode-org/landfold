#!/usr/bin/env python3
"""Attractor-MDS + CN residual layout for the Elja LJ38 packing book.

Steepest-descent attractors on the undirected kNN=12 L1 graph.
Classical MDS of the barrier ultrametric among attractors only,
GM attractor at the origin. Each family sits at its attractor plus
a 0.15 * centered CN-histogram PCA-2D residual inside the basin.

This is a 2-D map of basins of attraction, not a disconnectivity tree.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "cosmo-lj38"
OUT_DOCS = ROOT / "docs" / "ceriotti-figs"
HIST = Path("/tmp/occ-book/lj38_decaf_e.hist")
DEST = Path("/tmp/occ-book/cand-attract")

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)
KNN = 12
L1_BARRIER = 2.0
RES_SCALE = 0.15
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


def knn_edges(hist: np.ndarray, energy: np.ndarray, knn: int = KNN):
    """Undirected kNN in packing L1. Height = max E + 2 L1."""
    n = len(energy)
    edges = []
    seen = set()
    for i in range(n):
        drow = np.abs(hist - hist[i]).sum(1)
        order = np.argsort(drow)
        picked = 0
        for j in order:
            j = int(j)
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b))
                height = float(max(energy[i], energy[j]) + L1_BARRIER * drow[j])
                edges.append((height, i, j, float(drow[j])))
            picked += 1
            if picked >= knn:
                break
    return edges


def steepest_basins(n: int, energy: np.ndarray, edges):
    """Each packing drains to the lowest-energy neighbour; compress."""
    parent = np.arange(n)
    best = energy.copy()
    for _h, i, j, _d in edges:
        if energy[j] < best[i] - 1e-12:
            best[i] = energy[j]
            parent[i] = j
        if energy[i] < best[j] - 1e-12:
            best[j] = energy[i]
            parent[j] = i
    attract = np.empty(n, dtype=int)
    for i in range(n):
        x = i
        seen = set()
        while parent[x] != x and x not in seen:
            seen.add(x)
            x = int(parent[x])
        attract[i] = x
    return attract


def attractor_ultrametric(energy: np.ndarray, edges, attractors: np.ndarray):
    """Kruskal merge height among attractors, then D = clip(h - min E, 0)."""
    na = len(attractors)
    amap = {int(a): k for k, a in enumerate(attractors)}
    n = len(energy)
    uf = UnionFind(n)
    members = {i: [i] for i in range(n)}
    merge = np.full((na, na), np.nan)
    for h, i, j, _d in sorted(edges, key=lambda t: t[0]):
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        left, right = members[ri], members[rj]
        la = [amap[x] for x in left if x in amap]
        ra = [amap[x] for x in right if x in amap]
        for a in la:
            for b in ra:
                merge[a, b] = merge[b, a] = h
        uf.union(i, j)
        root = uf.find(i)
        members[root] = left + right
        if root != ri:
            members.pop(ri, None)
        if root != rj:
            members.pop(rj, None)
    np.fill_diagonal(merge, 0.0)
    if np.isnan(merge).any():
        cap = float(np.nanmax(merge)) + 1.0
        merge = np.where(np.isnan(merge), cap, merge)
    e_attr = energy[attractors]
    dist = np.clip(merge - np.minimum.outer(e_attr, e_attr), 0.0, None)
    np.fill_diagonal(dist, 0.0)
    return dist


def torgerson(dist: np.ndarray, dim: int = 2):
    n = dist.shape[0]
    d2 = dist * dist
    h = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * h @ d2 @ h
    w, v = np.linalg.eigh(b)
    idx = np.argsort(w)[::-1]
    w, v = w[idx], v[:, idx]
    lam = np.clip(w[:dim], 0.0, None)
    xy = v[:, :dim] * np.sqrt(lam)
    return xy, w[:6]


def place_gm_origin(xy_attr: np.ndarray, gm_k: int, ico_k: int) -> np.ndarray:
    """Translate GM attractor to the origin; rotate ico onto +x."""
    out = xy_attr - xy_attr[gm_k]
    vec = out[ico_k]
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-15:
        return out
    ang = np.arctan2(vec[1], vec[0])
    c, s = np.cos(-ang), np.sin(-ang)
    rot = np.array([[c, -s], [s, c]])
    return out @ rot.T


def basin_residual(hist: np.ndarray, attract: np.ndarray, attractors: np.ndarray):
    """Centered CN-histogram PCA-2D inside each basin, scaled by RES_SCALE."""
    n = len(attract)
    residual = np.zeros((n, 2))
    for a in attractors:
        idx = np.where(attract == a)[0]
        if idx.size <= 1:
            continue
        xc = hist[idx] - hist[idx].mean(0)
        u, s, _vt = np.linalg.svd(xc, full_matrices=False)
        k = min(2, u.shape[1], s.size)
        pca = np.zeros((idx.size, 2))
        pca[:, :k] = u[:, :k] * s[:k]
        residual[idx] = RES_SCALE * pca
    return residual


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
    rmax = float(rho.max()) if float(rho.max()) > 0 else 1.0
    mask = cf._fill_mask_holes(rho > 0.004 * rmax)
    fes = np.full_like(rho, np.nan)
    on = mask & (rho > 0)
    fes[on] = -KT * np.log(np.clip(rho[on] / rmax, 1e-12, 1))
    return gx, gy, np.clip(fes, 0, 2)


def panel_row(ax0, ax1, xy, gx, gy, fes, energy):
    mesh = ax0.contourf(
        gx, gy, fes, levels=np.linspace(0, 2, 21), cmap=PES, extend="max"
    )
    ax0.contour(
        gx,
        gy,
        np.where(np.isfinite(fes), fes, np.nan),
        levels=np.linspace(0.15, 1.85, 10),
        colors="#1a1a2e",
        linewidths=0.35,
    )
    ax0.scatter(
        xy[0, 0],
        xy[0, 1],
        s=120,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=6,
        label=rf"GM ${GM_E:.3f}$",
    )
    ax0.scatter(
        xy[1, 0],
        xy[1, 1],
        s=70,
        marker="D",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=6,
        label=rf"ico ${ICO_E:.3f}$",
    )
    ax0.legend(fontsize=7, frameon=True, fancybox=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("attractor-MDS  occupancy")
    for sp in ax0.spines.values():
        sp.set_visible(False)
    rel = np.clip(energy - energy.min(), 0, 8)
    sc = ax1.scatter(xy[:, 0], xy[:, 1], c=rel, s=18, cmap=PES, vmin=0, vmax=6, linewidths=0)
    ax1.scatter(
        xy[0, 0],
        xy[0, 1],
        s=120,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=6,
    )
    ax1.scatter(
        xy[1, 0],
        xy[1, 1],
        s=70,
        marker="D",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=6,
    )
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(r"attractor-MDS  $E-E_{\mathrm{GM}}$")
    for sp in ax1.spines.values():
        sp.set_visible(False)
    return mesh, sc


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    hist, energy, wells = load_book(HIST)
    n = len(energy)
    edges = knn_edges(hist, energy, knn=KNN)
    attract = steepest_basins(n, energy, edges)
    attractors = np.unique(attract)
    n_attractors = int(attractors.size)
    n_gm = int((attract == attract[0]).sum())
    n_ico = int((attract == attract[1]).sum())
    gm_k = int(np.where(attractors == attract[0])[0][0])
    ico_k = int(np.where(attractors == attract[1])[0][0])

    dist = attractor_ultrametric(energy, edges, attractors)
    xy_attr, ev = torgerson(dist, 2)
    xy_attr = place_gm_origin(xy_attr, gm_k, ico_k)

    residual = basin_residual(hist, attract, attractors)
    attr_of = np.searchsorted(attractors, attract)
    xy = xy_attr[attr_of] + residual

    sep = float(np.linalg.norm(xy[0] - xy[1]))
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    sep_norm = sep / max(diam, 1e-12)

    leftover = (attract != attract[0]) & (attract != attract[1])

    cf = _cf()
    gx, gy, fes = kde_fes(xy, wells, cf)

    np.savetxt(DEST / "attract.xy", xy, fmt="%.8e")
    np.savetxt(DEST / "attractor.xy", xy_attr, fmt="%.8e")
    np.savetxt(DEST / "residual.xy", residual, fmt="%.8e")
    np.savetxt(DEST / "attract.idx", attract, fmt="%d")
    np.savetxt(DEST / "D_attr.dist", dist, fmt="%.8e")
    payload = {
        "n": n,
        "n_edges": len(edges),
        "n_attractors": n_attractors,
        "n_GM": n_gm,
        "n_ico": n_ico,
        "sep": sep,
        "diam": diam,
        "sep_norm": sep_norm,
        "D_gm_ico": float(dist[gm_k, ico_k]),
        "mds_eigs": [float(x) for x in ev],
        "attractors": [int(a) for a in attractors],
        "basin_size": {str(int(a)): int((attract == a).sum()) for a in attractors},
        "basin_wells": {
            str(int(a)): float(wells[attract == a].sum()) for a in attractors
        },
        "leftover_families": int(leftover.sum()),
        "leftover_wells": float(wells[leftover].sum()),
        "gm": [float(xy[0, 0]), float(xy[0, 1])],
        "ico": [float(xy[1, 0]), float(xy[1, 1])],
        "knn": KNN,
        "res_scale": RES_SCALE,
    }
    (DEST / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), facecolor="white")
    mesh, sc = panel_row(axes[0], axes[1], xy, gx, gy, fes, energy)
    fig.colorbar(mesh, ax=axes[0], fraction=0.046, pad=0.03).set_label(r"$F/\varepsilon$")
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.03).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    fig.tight_layout()
    png_local = DEST / "elja_occ_lj38_cand_attract.png"
    fig.savefig(png_local, dpi=170, facecolor="white")
    png_docs = OUT_DOCS / "elja_occ_lj38_cand_attract.png"
    fig.savefig(png_docs, dpi=170, facecolor="white")
    plt.close(fig)

    print("n_attractors", n_attractors)
    print("n_GM", n_gm)
    print("n_ico", n_ico)
    print("sep_norm", f"{sep_norm:.6f}")
    print("wrote", png_local)
    print("wrote", png_docs)
    print("wrote", DEST / "scores.json")


if __name__ == "__main__":
    main()
