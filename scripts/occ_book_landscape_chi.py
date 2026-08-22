#!/usr/bin/env python3
"""Landscape chi of the Elja LJ38 packing book.

Ceriotti chi of CN histograms does not encode barriers, so occupancy
of inherent structures cannot show Wales basins of attraction.
asinh of the barrier ultrametric collapses the two funnels (GM-ico
sep 0.04). This script embeds the DECAF L1 + energy graph itself:

  raw MDS of the barrier excess
  Isomap of geodesic barrier heights
  Laplacian eigenmaps / diffusion / commute-time
  committor and Fiedler polar maps with r = E - E_GM

Defeat bar (all must hold): GM-ico sep / diameter > 0.25, two FES
wells, GM inside the deeper well, GM not on the convex hull.
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
OUT = ROOT / "docs" / "ceriotti-figs"
EX = ROOT / "examples" / "cosmo-lj38"
HIST = Path("/tmp/occ-book/lj38_decaf_e.hist")
SCORE = Path("/tmp/occ-book/landscape_scores.json")
COORD_DIR = Path("/tmp/occ-book/landscape")

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
SEP_BAR = 0.25


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

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.size[ra] += self.size[rb]
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


def neighbour_edges(hist: np.ndarray, energy: np.ndarray, knn: int = 12):
    """kNN in packing L1, plus every pair with L1 <= cut.

    Edge cost is the positive local barrier, never the signed height:
    undirected Floyd-Warshall on max(E) ~ -170 walks into -inf.
    """
    n = len(energy)
    edges = []
    seen = set()
    l1 = np.zeros((n, n))
    for i in range(n):
        drow = np.abs(hist - hist[i]).sum(1)
        l1[i] = drow
        order = np.argsort(drow)
        picked = []
        for j in order:
            j = int(j)
            if j == i:
                continue
            if drow[j] <= L1_CUT or len(picked) < knn:
                picked.append(j)
            if drow[j] > L1_CUT and len(picked) >= knn:
                break
        for j in picked:
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            cost = float(abs(energy[i] - energy[j]) + L1_BARRIER * drow[j])
            height = float(max(energy[i], energy[j]) + L1_BARRIER * drow[j])
            edges.append((height, i, j, float(drow[j]), cost))
    return edges, l1


def barrier_excess(n: int, energy: np.ndarray, edges):
    ranked = sorted(edges, key=lambda t: t[0])
    uf = UnionFind(n)
    members = {i: [i] for i in range(n)}
    merge = np.full((n, n), np.nan)
    join_gm_ico = None
    for h, i, j, _d, _c in ranked:
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        left, right = members[ri], members[rj]
        for a in left:
            for b in right:
                merge[a, b] = merge[b, a] = h
        if join_gm_ico is None and uf.find(0) != uf.find(1) and (
            (uf.find(i) == uf.find(0) and uf.find(j) == uf.find(1))
            or (uf.find(i) == uf.find(1) and uf.find(j) == uf.find(0))
        ):
            join_gm_ico = h
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
    excess = np.clip(merge - np.minimum.outer(energy, energy), 0.0, None)
    np.fill_diagonal(excess, 0.0)
    labels = superbasin_labels(n, ranked, join_gm_ico)
    return excess, join_gm_ico, labels


def superbasin_labels(n: int, ranked, join):
    uf = UnionFind(n)
    cut = join - 1e-9 if join is not None else np.inf
    for h, i, j, _d, _c in ranked:
        if h >= cut:
            break
        uf.union(i, j)
    roots = np.array([uf.find(i) for i in range(n)])
    return roots


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


def floyd(n: int, edges):
    g = np.full((n, n), np.inf)
    np.fill_diagonal(g, 0.0)
    for _h, i, j, _d, cost in edges:
        w = max(float(cost), 0.0)
        if w < g[i, j]:
            g[i, j] = g[j, i] = w
    for k in range(n):
        g = np.minimum(g, g[:, k : k + 1] + g[k : k + 1, :])
    if not np.isfinite(g).all():
        finite = g[np.isfinite(g) & (g > 0)]
        cap = float(finite.max()) * 2.0 if finite.size else 1.0
        g = np.where(np.isfinite(g), g, cap)
        np.fill_diagonal(g, 0.0)
    return g


def adjacency(n: int, energy: np.ndarray, edges, mode="boltz"):
    a = np.zeros((n, n))
    emin = float(energy.min())
    for h, i, j, d, cost in edges:
        if mode == "boltz":
            w = float(np.exp(-(h - emin) / KT))
        elif mode == "excess":
            w = float(np.exp(-cost / KT))
        else:
            w = 1.0 / max(d, 1e-6)
        a[i, j] = a[j, i] = max(w, 1e-300)
    return a


def metropolis(n: int, energy: np.ndarray, edges, tstar: float):
    """Directed Metropolis walk. Low T approximates steepest descent."""
    a = np.zeros((n, n))
    for _h, i, j, _d, _c in edges:
        de = float(energy[j] - energy[i])
        a[i, j] = max(float(np.exp(-max(de, 0.0) / tstar)), 1e-300)
        a[j, i] = max(float(np.exp(-max(-de, 0.0) / tstar)), 1e-300)
    return a


def steepest_basins(n: int, energy: np.ndarray, edges):
    """Each packing drains to the lowest-energy neighbour; compress."""
    parent = np.arange(n)
    best = energy.copy()
    for _h, i, j, _d, _c in edges:
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


def eigenmaps(a: np.ndarray):
    deg = a.sum(1)
    deg = np.clip(deg, 1e-300, None)
    dinvsqrt = 1.0 / np.sqrt(deg)
    s = (dinvsqrt[:, None] * a) * dinvsqrt[None, :]
    w, v = np.linalg.eigh(s)
    idx = np.argsort(w)[::-1]
    w, v = w[idx], v[:, idx]
    # skip the constant mode
    xy = v[:, 1:3] * dinvsqrt[:, None]
    return xy, w[:6]


def diffusion(a: np.ndarray, t: int = 8):
    deg = np.clip(a.sum(1), 1e-300, None)
    p = a / deg[:, None]
    w, v = np.linalg.eig(p)
    w, v = np.real(w), np.real(v)
    idx = np.argsort(w)[::-1]
    w, v = w[idx], v[:, idx]
    xy = v[:, 1:3] * (w[1:3] ** t)
    return xy, w[:6]


def commute_time(a: np.ndarray):
    n = a.shape[0]
    deg = a.sum(1)
    lap = np.diag(deg) - a
    w, v = np.linalg.eigh(lap)
    # pinv: invert positive eigenvalues
    inv = np.zeros_like(w)
    pos = w > 1e-10 * max(float(w.max()), 1.0)
    inv[pos] = 1.0 / w[pos]
    lplus = (v * inv) @ v.T
    vol = float(deg.sum())
    diag = np.diag(lplus)
    ct = vol * (diag[:, None] + diag[None, :] - 2.0 * lplus)
    ct = np.clip(ct, 0.0, None)
    np.fill_diagonal(ct, 0.0)
    xy, ev = torgerson(np.sqrt(ct), 2)
    return xy, ev, ct


def committor(a: np.ndarray, src=0, sink=1):
    n = a.shape[0]
    deg = np.clip(a.sum(1), 1e-300, None)
    p = a / deg[:, None]
    mask = np.ones(n, dtype=bool)
    mask[src] = False
    mask[sink] = False
    tix = np.where(mask)[0]
    q = np.zeros(n)
    q[sink] = 1.0
    if tix.size == 0:
        return q
    i_minus_p = np.eye(tix.size) - p[np.ix_(tix, tix)]
    rhs = p[tix, sink].copy()
    try:
        q[tix] = np.linalg.solve(i_minus_p, rhs)
    except np.linalg.LinAlgError:
        q[tix] = np.linalg.lstsq(i_minus_p, rhs, rcond=None)[0]
    return np.clip(q, 0.0, 1.0)


def polar(radius: np.ndarray, angle: np.ndarray):
    ang = np.clip(angle, 0.0, 1.0) * np.pi
    r = np.clip(radius, 0.0, None)
    return np.column_stack((r * np.cos(ang), r * np.sin(ang)))


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


def local_minima(fes: np.ndarray):
    out = []
    ny, nx = fes.shape
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            v = fes[i, j]
            if not np.isfinite(v):
                continue
            nb = fes[i - 1 : i + 2, j - 1 : j + 2]
            if np.nanmin(nb) >= v - 1e-12 and v < 0.35:
                out.append((v, i, j))
    out.sort()
    kept = []
    for v, i, j in out:
        if all(abs(i - ii) + abs(j - jj) > 8 for _, ii, jj in kept):
            kept.append((v, i, j))
    return kept


def on_hull(xy: np.ndarray, idx: int = 0, tol: float = 0.04) -> bool:
    pts = xy - xy.mean(0)
    # cheap rim test: near the axis-aligned bbox
    lo, hi = pts.min(0), pts.max(0)
    p = pts[idx]
    span = np.clip(hi - lo, 1e-12, None)
    t = float(np.min(np.minimum(p - lo, hi - p) / span))
    return bool(t < tol)


def score_map(name, xy, wells, energy, labels, cf):
    gm, ico = xy[0], xy[1]
    sep = float(np.linalg.norm(gm - ico))
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    sep_norm = sep / max(diam, 1e-12)
    gx, gy, fes = kde_fes(xy, wells, cf)
    f_gm = cf.fes_at(gx, gy, fes, gm)
    f_ico = cf.fes_at(gx, gy, fes, ico)
    mins = local_minima(fes)
    two_wells = len(mins) >= 2
    # GM in deeper well: F(GM) finite and among the lowest third of F at points
    f_pts = []
    for p in xy:
        f_pts.append(cf.fes_at(gx, gy, fes, p))
    f_pts = np.asarray(f_pts, dtype=float)
    finite = np.isfinite(f_pts)
    deeper = False
    if np.isfinite(f_gm) and finite.any():
        deeper = f_gm <= np.nanpercentile(f_pts[finite], 25)
    rim = on_hull(xy, 0)
    # silhouette of the two steepest-descent basins of GM and ico
    lab_gm, lab_ico = labels[0], labels[1]
    sil = 0.0
    if lab_gm != lab_ico:
        a_mask = labels == lab_gm
        b_mask = labels == lab_ico
        if a_mask.sum() > 1 and b_mask.sum() > 1:
            da = np.linalg.norm(xy[a_mask] - xy[a_mask].mean(0), axis=1).mean()
            db = np.linalg.norm(xy[b_mask] - xy[b_mask].mean(0), axis=1).mean()
            between = float(np.linalg.norm(xy[a_mask].mean(0) - xy[b_mask].mean(0)))
            sil = between / max(da + db, 1e-12)
    defeat = (
        sep_norm > SEP_BAR
        and two_wells
        and deeper
        and not rim
        and sil > 0.8
    )
    return {
        "name": name,
        "sep": sep,
        "sep_norm": sep_norm,
        "diam": diam,
        "F_GM": None if not np.isfinite(f_gm) else float(f_gm),
        "F_ico": None if not np.isfinite(f_ico) else float(f_ico),
        "n_wells": len(mins),
        "two_wells": two_wells,
        "gm_deeper": bool(deeper),
        "gm_on_rim": bool(rim),
        "silhouette": sil,
        "defeat": bool(defeat),
        "gm": [float(gm[0]), float(gm[1])],
        "ico": [float(ico[0]), float(ico[1])],
    }, (gx, gy, fes)


def panel_row(ax0, ax1, xy, gx, gy, fes, energy, title):
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
    ax0.set_title(title + "  occupancy")
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
    ax1.set_title(title + r"  $E-E_{\mathrm{GM}}$")
    for sp in ax1.spines.values():
        sp.set_visible(False)
    return mesh, sc


def save_pair(name, xy, gx, gy, fes, energy):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), facecolor="white")
    mesh, sc = panel_row(axes[0], axes[1], xy, gx, gy, fes, energy, name)
    fig.colorbar(mesh, ax=axes[0], fraction=0.046, pad=0.03).set_label(r"$F/\varepsilon$")
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.03).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    dest = OUT / f"elja_occ_lj38_{name}.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", dest)


def main() -> None:
    hist, energy, wells = load_book(HIST)
    n = len(energy)
    print("n", n, "E[0]", energy[0], "E[1]", energy[1])
    edges, _l1 = neighbour_edges(hist, energy, knn=12)
    print("edges", len(edges))
    excess, join, labels = barrier_excess(n, energy, edges)
    attract = steepest_basins(n, energy, edges)
    print("join", join, "D(GM,ico)", excess[0, 1], "n_kruskal", len(np.unique(labels)))
    print("superbasin sizes", {int(k): int((labels == k).sum()) for k in (labels[0], labels[1])})
    print(
        "steepest attractors",
        len(np.unique(attract)),
        "n_GM",
        int((attract == attract[0]).sum()),
        "n_ico",
        int((attract == attract[1]).sum()),
    )

    a_boltz = adjacency(n, energy, edges, "boltz")
    a_ex = adjacency(n, energy, edges, "excess")
    a_metro = metropolis(n, energy, edges, tstar=0.05)
    geo = floyd(n, edges)
    print("geo(GM,ico)", geo[0, 1], "geo median", float(np.median(geo[geo > 0])))

    q = committor(a_metro, 0, 1)
    q_hot = committor(a_ex, 0, 1)
    print(
        "q_cold median/std",
        float(np.median(q)),
        float(q.std()),
        "q_hot median",
        float(np.median(q_hot)),
    )

    maps = {}
    maps["rawmds"] = torgerson(excess, 2)[0]
    maps["isomap"] = torgerson(geo, 2)[0]
    maps["laplace"] = eigenmaps(a_boltz)[0]
    maps["diffuse"] = diffusion(a_boltz, t=8)[0]
    maps["commute"] = commute_time(a_boltz)[0]
    maps["qe"] = np.column_stack((q, energy - energy.min()))
    r_e = energy - energy.min()
    maps["polarq"] = polar(r_e, q)
    ang = np.clip(q, 0.0, 1.0).copy()
    ang[attract == attract[0]] = 0.0
    ang[attract == attract[1]] = 1.0
    maps["basinpol"] = polar(r_e, ang)
    fied = eigenmaps(a_boltz)[0][:, 0]
    fied_n = (fied - fied.min()) / max(float(fied.max() - fied.min()), 1e-12)
    # flip so GM is on the left / low angle
    if fied_n[0] > fied_n[1]:
        fied_n = 1.0 - fied_n
    maps["polarfi"] = polar(r_e, fied_n)
    # hybrid: blend asinh CN L1 with geodesic, then MDS
    l1 = np.zeros((n, n))
    for i in range(n):
        l1[i] = np.abs(hist - hist[i]).sum(1)
    sig_l = float(np.median(l1[l1 > 0]))
    sig_g = float(np.median(geo[geo > 0]))
    hyb = 0.35 * np.arcsinh(l1 / max(sig_l, 1e-9)) + 0.65 * np.arcsinh(geo / max(sig_g, 1e-9))
    np.fill_diagonal(hyb, 0.0)
    maps["hybrid"] = torgerson(hyb, 2)[0]
    # energy-centered: shift so GM is origin of laplace then scale by r_e of each basin
    lap = maps["laplace"].copy()
    if np.linalg.norm(lap[1] - lap[0]) > 0:
        axis = lap[1] - lap[0]
        axis = axis / np.linalg.norm(axis)
        orth = np.array([-axis[1], axis[0]])
        centered = lap - lap[0]
        maps["gmcenter"] = np.column_stack(
            (centered @ axis, centered @ orth)
        )

    COORD_DIR.mkdir(parents=True, exist_ok=True)
    cf = _cf()
    scores = []
    winners = []
    for name, xy in maps.items():
        np.savetxt(COORD_DIR / f"{name}.xy", xy, fmt="%.8e")
        rec, (gx, gy, fes) = score_map(name, xy, wells, energy, attract, cf)
        scores.append(rec)
        print(
            f"{name:10s} sep={rec['sep_norm']:.3f} wells={rec['n_wells']} "
            f"Fgm={rec['F_GM']} Fico={rec['F_ico']} sil={rec['silhouette']:.3f} "
            f"rim={rec['gm_on_rim']} deeper={rec['gm_deeper']} defeat={rec['defeat']}"
        )
        save_pair(name, xy, gx, gy, fes, energy)
        if rec["sep_norm"] > SEP_BAR:
            winners.append(name)

    # comparison strip of occupancy only
    names = list(maps.keys())
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.6, 3.6 * nrow), facecolor="white")
    axes = np.atleast_2d(axes)
    for k, name in enumerate(names):
        ax = axes[k // ncol, k % ncol]
        xy = maps[name]
        rec, (gx, gy, fes) = score_map(name, xy, wells, energy, attract, cf)
        ax.contourf(gx, gy, fes, levels=np.linspace(0, 2, 21), cmap=PES, extend="max")
        ax.scatter(xy[0, 0], xy[0, 1], s=80, marker="*", c="k", edgecolors="white", zorder=6)
        ax.scatter(xy[1, 0], xy[1, 1], s=50, marker="D", c="k", edgecolors="white", zorder=6)
        mark = "WIN" if rec["defeat"] else ("sep" if rec["sep_norm"] > SEP_BAR else "fail")
        ax.set_title(f"{name}  {mark}  sep={rec['sep_norm']:.2f} sil={rec['silhouette']:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for k in range(len(names), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    dest = OUT / "elja_occ_lj38_landscape_cmp.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", dest)

    payload = {
        "join": join,
        "D_gm_ico": float(excess[0, 1]),
        "geo_gm_ico": float(geo[0, 1]),
        "committor_median": float(np.median(q)),
        "scores": scores,
        "winners": winners,
        "any_defeat": any(s["defeat"] for s in scores),
    }
    SCORE.write_text(json.dumps(payload, indent=2))
    print("wrote", SCORE)
    print("winners", winners, "any_defeat", payload["any_defeat"])


if __name__ == "__main__":
    main()
