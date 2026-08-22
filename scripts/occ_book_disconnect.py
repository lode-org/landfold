#!/usr/bin/env python3
"""Wales-style superbasin disconnectivity of the Elja LJ38 packing book.

No PATHSAMPLE TS file. Edges are DECAF L1 neighbours; the barrier
surrogate is max(E_i, E_j), a lower bound. Superbasins are the
union-find components at each energy, the same analysis disconnectionDPS
runs on a TS database (DELTA / FIRST / LEVELS).

This is the basin-of-attraction graph. It is not a sketch-map figure.
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
HIST = Path("/tmp/occ-book/lj38_decaf_e.hist")
ASINH = Path("/tmp/occ-book/lj38_decaf_asinh.ld")
OOS = Path("/tmp/occ-book/lj38_decaf_oos.proj")
ENERGY = Path("/tmp/occ-book/lj38.energy")

GM_E = -173.928427
ICO_E = -173.252378
L1_CUT = 0.55
# extra height per unit L1 so far packing families do not merge at max(E)
L1_BARRIER = 2.0


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> tuple[int, int] | None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.size[ra] += self.size[rb]
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return ra, rb


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


def load_xy(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        rows.append([float(p[0]), float(p[1])])
    return np.asarray(rows)


def knn_edges(hist: np.ndarray, cut: float) -> list[tuple[float, int, int]]:
    n = len(hist)
    edges = []
    for i in range(n):
        d = np.abs(hist - hist[i]).sum(1)
        for j in range(i + 1, n):
            if d[j] <= cut:
                edges.append((float(d[j]), i, j))
    return edges


def superbasin_merges(energy: np.ndarray, edges: list[tuple[float, int, int]]):
    """Kruskal on neighbour edges, height = max(E_i, E_j)."""
    n = len(energy)
    ranked = []
    for dist, i, j in edges:
        ranked.append((float(max(energy[i], energy[j]) + L1_BARRIER * dist), i, j))
    ranked.sort()
    uf = UnionFind(n)
    merges = []
    for height, i, j in ranked:
        out = uf.union(i, j)
        if out is None:
            continue
        merges.append((height, i, j, uf.find(i), uf.size[uf.find(i)]))
        if uf.size[uf.find(0)] == n:
            break
    return merges, uf


def components_at(energy: np.ndarray, merges, level: float) -> np.ndarray:
    n = len(energy)
    uf = UnionFind(n)
    for height, i, j, _root, _sz in merges:
        if height > level:
            break
        uf.union(i, j)
    return np.array([uf.find(i) for i in range(n)])


def leaf_order(n: int, merges) -> list[int]:
    """Contiguous order: each merge concatenates the two blocks."""
    blocks = {i: [i] for i in range(n)}
    parent = {i: i for i in range(n)}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _h, i, j, _r, _s in merges:
        ri, rj = root(i), root(j)
        if ri == rj:
            continue
        blocks[ri] = blocks[ri] + blocks[rj]
        parent[rj] = ri
    r0 = root(0)
    seen = set(blocks[r0])
    rest = [k for k in range(n) if k not in seen]
    return blocks[r0] + rest


def plot_disconnect(energy, merges, order, dest: Path, e_fcc_ico: float) -> None:
    pos = {idx: k for k, idx in enumerate(order)}
    fig, ax = plt.subplots(figsize=(8.4, 6.2), facecolor="white")
    # verticals: each min up to first merge
    first_merge = {}
    for h, i, j, _r, _s in merges:
        for node in (i, j):
            first_merge.setdefault(node, h)
    for idx, e in enumerate(energy):
        top = first_merge.get(idx, float(energy.max()) + 0.5)
        color = "#0b6e4f" if idx == 0 else ("#1d4e89" if idx == 1 else "#888888")
        lw = 1.6 if idx in (0, 1) else 0.45
        ax.plot([pos[idx], pos[idx]], [e, top], color=color, lw=lw, zorder=3 if idx < 2 else 1)
    # horizontals at each merge, spanning the two blocks then known
    uf = UnionFind(len(energy))
    members = {i: {i} for i in range(len(energy))}
    for h, i, j, _r, _s in merges:
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        left = min(pos[k] for k in members[ri] | members[rj])
        right = max(pos[k] for k in members[ri] | members[rj])
        ax.plot([left, right], [h, h], color="#444444", lw=0.5, zorder=2)
        uf.union(i, j)
        rr = uf.find(i)
        other = rj if rr == ri else ri
        members[rr] = members[ri] | members[rj]
        members.pop(other, None)
    ax.axhline(e_fcc_ico, color="#b42318", lw=0.8, ls="--")
    ax.scatter([pos[0]], [energy[0]], s=40, marker="*", c="#0b6e4f", zorder=5)
    ax.scatter([pos[1]], [energy[1]], s=28, marker="D", c="#1d4e89", zorder=5)
    ax.annotate("GM", (pos[0], energy[0]), textcoords="offset points", xytext=(6, 8), fontsize=8)
    ax.annotate("ico", (pos[1], energy[1]), textcoords="offset points", xytext=(6, 8), fontsize=8)
    ax.set_ylabel(r"$E/\varepsilon$")
    ax.set_xticks([])
    ax.set_xlim(-2, len(energy) + 1)
    # Wales: energy increases upward, GM at the bottom
    # bottom of the landscape only: the two crystal stems and the join
    ax.set_ylim(float(energy.min()) - 0.3, (e_fcc_ico if e_fcc_ico else -170.0) + 1.2)
    ax.set_title("packing superbasins  (L1 neighbours, barrier = max $E$ + 2 L1)")
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(dest, dpi=170, facecolor="white")
    print("wrote", dest)


def plot_chi_basins(xy, labels, dest: Path, title: str) -> None:
    pes = LinearSegmentedColormap.from_list("ruhi", ["#004D40", "#1E88E5", "#D81B60"], N=8)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor="white")
    colors = np.array(["#bbbbbb"] * len(xy), dtype=object)
    colors[labels == 0] = "#0b6e4f"
    colors[labels == 1] = "#1d4e89"
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=18, linewidths=0, zorder=2)
    ax.scatter(xy[0, 0], xy[0, 1], s=120, marker="*", c="#0b6e4f", edgecolors="white", linewidths=0.6, zorder=5, label="GM superbasin")
    ax.scatter(xy[1, 0], xy[1, 1], s=70, marker="D", c="#1d4e89", edgecolors="white", linewidths=0.6, zorder=5, label="ico superbasin")
    ax.legend(fontsize=8, frameon=True, fancybox=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(dest, dpi=170, facecolor="white")
    print("wrote", dest)
    _ = pes


def main() -> None:
    hist, energy, wells = load_book(HIST)
    edges = knn_edges(hist, L1_CUT)
    print("neighbour edges", len(edges), "cut", L1_CUT)
    merges, _uf = superbasin_merges(energy, edges)
    print("merges", len(merges))
    # energy where GM and ico first share a component
    e_join = None
    uf = UnionFind(len(energy))
    for h, i, j, _r, _s in merges:
        uf.union(i, j)
        if uf.find(0) == uf.find(1):
            e_join = h
            break
    print("GM-ico join energy", e_join)
    order = leaf_order(len(energy), merges)
    # put the two crystal superbasins first so the stems are visible
    if e_join is not None:
        pre = components_at(energy, merges, e_join - 1e-6)
        gm_set = {i for i, c in enumerate(pre) if c == pre[0]}
        ico_set = {i for i, c in enumerate(pre) if c == pre[1]}
        order = (
            [i for i in order if i in gm_set]
            + [i for i in order if i in ico_set]
            + [i for i in order if i not in gm_set and i not in ico_set]
        )
    plot_disconnect(energy, merges, order, OUT / "elja_occ_lj38_disconnect.png", e_join or 0.0)

    # superbasins just below the join: GM component vs ico component
    if e_join is None:
        level = float(np.median(energy))
    else:
        level = e_join - 1e-6
    comp = components_at(energy, merges, level)
    tags = np.full(len(energy), -1, dtype=int)
    tags[comp == comp[0]] = 0
    tags[comp == comp[1]] = 1
    print(
        "at E",
        level,
        "GM basin",
        int((tags == 0).sum()),
        "ico basin",
        int((tags == 1).sum()),
        "other",
        int((tags < 0).sum()),
    )
    xy = load_xy(ASINH)
    plot_chi_basins(
        xy,
        tags,
        OUT / "elja_occ_lj38_superbasin_chi.png",
        rf"asinh $\chi$  superbasins below $E={level:.2f}$",
    )


if __name__ == "__main__":
    main()
