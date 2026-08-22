#!/usr/bin/env python3
"""Landscape geodesic on structure kNN of the 4042 LJ38 inherent structures.

Graph lives in the xyz fingerprint, not packing-CN L1. Each minimum is
the sorted list of the 38-choose-2 pair distances (rotation, translation
and permutation invariant). kNN=12 among distinct fingerprints; exact
permutational copies (d_struct ~ 1e-7) stay linked to each other so they
share a geodesic location.

    cost_ij = |E_i - E_j| + 2 d_struct(i, j)

All-pairs Dijkstra of those costs, then classical Torgerson MDS of the
landfold asinh of the geodesic. The painted field is E - E_GM (lower envelope and
local IDW), never leftover-well occupancy invert.

GM is index 0, ico is index 40 of lj38_0013.min.
"""

from __future__ import annotations

import json
import sys
import time
from heapq import heappop, heappush
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import occ_book_landscape_chi as lc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ceriotti-figs"
DEST = Path("/tmp/occ-book/cand-geostruc")
MINFILE = Path("/tmp/occ-book/lj38_0013.min")
ENERGY = Path("/tmp/occ-book/lj38.energy")
DPAIR = Path("/tmp/occ-book/lj38_dpair.dist")
KABSCH_CANDS = (
    Path("/tmp/occ-book/lj38_kabsch.dist"),
    Path("/tmp/occ-book/lj38.kabsch"),
    Path("/tmp/occ-book/kabsch.dist"),
)

KNN = 12
LAM = 2.0
COPY_EPS = 1e-4
GM = 0
ICO = 40
ASINH_NORM = 2.0 * float(np.arcsinh(1.0))
SEP_BAR = 0.25
EMAX = 5.0
GEO_NPY = DEST / "geo.npy"

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)


def _now() -> float:
    return time.perf_counter()


def load_min(path: Path):
    energies = []
    coords = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        p = np.fromstring(line, sep=" ")
        energies.append(float(p[0]))
        xyz = np.asarray(p[1:], dtype=float).reshape(-1, 3)
        coords.append(xyz)
    return np.asarray(energies, dtype=float), coords


def sorted_pair_fp(xyz: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    iu = np.triu_indices(xyz.shape[0], k=1)
    return np.sort(d[iu])


def fingerprints(coords) -> np.ndarray:
    return np.vstack([sorted_pair_fp(xyz) for xyz in coords])


def load_square(path: Path) -> np.ndarray:
    with path.open("rb") as fh:
        magic = fh.read(6)
        fh.seek(0)
        if magic == b"\x93NUMPY":
            return np.load(fh)
    return np.loadtxt(path)


def save_square(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        np.save(fh, np.asarray(arr, dtype=np.float64))


def pair_l2(fps: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial.distance import pdist, squareform

        return squareform(pdist(fps, metric="euclidean"))
    except ImportError:
        n = fps.shape[0]
        out = np.zeros((n, n), dtype=np.float64)
        chunk = 64
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            diff = fps[i0:i1, None, :] - fps[None, :, :]
            out[i0:i1] = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        np.fill_diagonal(out, 0.0)
        return out


def structure_d(fps: np.ndarray) -> tuple[np.ndarray, str]:
    for cand in KABSCH_CANDS:
        if cand.is_file():
            d = load_square(cand)
            if d.shape[0] != len(fps):
                raise SystemExit("kabsch %s shape %s != %d" % (cand, d.shape, len(fps)))
            print("reuse kabsch", cand, d.shape)
            return np.asarray(d, dtype=np.float64), "kabsch:%s" % cand
    if DPAIR.is_file():
        d = load_square(DPAIR)
        if d.shape[0] == len(fps):
            print("reuse", DPAIR, d.shape)
            return np.asarray(d, dtype=np.float64), "dpair:%s" % DPAIR
        print("ignore stale", DPAIR, "shape", d.shape)
    t0 = _now()
    d = pair_l2(fps)
    np.fill_diagonal(d, 0.0)
    save_square(DPAIR, d)
    print("wrote", DPAIR, "pair_l2", "%.1fs" % (_now() - t0))
    return d, "sorted-pair-L2"


def knn_edges(dstruct: np.ndarray, energy: np.ndarray, knn: int = KNN, lam: float = LAM):
    """Symmetrized kNN among distinct fingerprints. Copies stay linked.

    A neighbour with d_struct <= COPY_EPS is a permutational copy and
    does not consume a kNN slot. Each site still gets `knn` strictly
    positive-distance neighbours so a 40-copy GM clique is not isolated.
    """
    n = len(energy)
    edges = []
    seen = set()
    n_copy = 0
    n_distinct = 0

    def add(i: int, j: int, ds: float) -> None:
        a, b = (i, j) if i < j else (j, i)
        if a == b or (a, b) in seen:
            return
        seen.add((a, b))
        cost = float(abs(energy[i] - energy[j]) + lam * ds)
        if cost < 0.0:
            raise RuntimeError("negative edge %d-%d: %s" % (a, b, cost))
        edges.append((cost, i, j, float(ds), cost))

    for i in range(n):
        drow = dstruct[i]
        order = np.argsort(drow, kind="mergesort")
        got = 0
        for j in order:
            j = int(j)
            if j == i:
                continue
            ds = float(drow[j])
            if ds <= COPY_EPS:
                add(i, j, ds)
                n_copy += 1
                continue
            add(i, j, ds)
            n_distinct += 1
            got += 1
            if got >= knn:
                break
    return edges, n_copy, n_distinct


def n_components(n: int, edges) -> tuple[int, np.ndarray]:
    uf = lc.UnionFind(n)
    for _c, i, j, _d, _w in edges:
        uf.union(i, j)
    labs = np.array([uf.find(i) for i in range(n)], dtype=int)
    return int(len(np.unique(labs))), labs


def stitch_components(dstruct: np.ndarray, energy: np.ndarray, edges, labs, lam: float = LAM):
    """Nearest structure edge between components until the graph is one piece."""
    roots = np.unique(labs)
    if len(roots) <= 1:
        return edges
    members = {int(r): np.where(labs == r)[0] for r in roots}
    added = []
    uf = lc.UnionFind(len(energy))
    remap = {int(r): k for k, r in enumerate(roots)}
    for _c, i, j, _d, _w in edges:
        uf.union(remap[int(labs[i])], remap[int(labs[j])])
    # Kruskal on component-pair nearest structure distances
    pairs = []
    rlist = [int(r) for r in roots]
    for a in range(len(rlist)):
        ia = members[rlist[a]]
        for b in range(a + 1, len(rlist)):
            ib = members[rlist[b]]
            block = dstruct[np.ix_(ia, ib)]
            t = int(np.argmin(block))
            ia_, ib_ = divmod(t, block.shape[1])
            i, j = int(ia[ia_]), int(ib[ib_])
            pairs.append((float(block[ia_, ib_]), i, j, remap[rlist[a]], remap[rlist[b]]))
    pairs.sort()
    seen = {(min(e[1], e[2]), max(e[1], e[2])) for e in edges}
    for ds, i, j, ca, cb in pairs:
        if uf.find(ca) == uf.find(cb):
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            uf.union(ca, cb)
            continue
        seen.add((a, b))
        cost = float(abs(energy[i] - energy[j]) + lam * ds)
        rec = (cost, i, j, float(ds), cost)
        edges.append(rec)
        added.append(rec)
        uf.union(ca, cb)
        if all(uf.find(0) == uf.find(k) for k in range(len(roots))):
            break
    print("stitched", len(added), "inter-component structure edges")
    return edges


def dijkstra_all(n: int, edges) -> np.ndarray:
    rows, cols, data = [], [], []
    for _h, i, j, _d, cost in edges:
        w = float(cost)
        rows.extend((i, j))
        cols.extend((j, i))
        data.extend((w, w))
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

        csr = csr_matrix((data, (rows, cols)), shape=(n, n))
        geo = dijkstra(csr, directed=False)
    except ImportError:
        adj = [[] for _ in range(n)]
        for i, j, w in zip(rows, cols, data):
            adj[i].append((j, w))
        geo = np.full((n, n), np.inf)
        for s in range(n):
            dist = np.full(n, np.inf)
            dist[s] = 0.0
            heap = [(0.0, s)]
            while heap:
                du, u = heappop(heap)
                if du > dist[u]:
                    continue
                for v, w in adj[u]:
                    nd = du + w
                    if nd < dist[v]:
                        dist[v] = nd
                        heappush(heap, (nd, v))
            geo[s] = dist
    if not np.isfinite(geo).all():
        finite = geo[np.isfinite(geo) & (geo > 0)]
        cap = float(finite.max()) * 2.0 if finite.size else 1.0
        n_inf = int((~np.isfinite(geo)).sum())
        print("WARN cap", n_inf, "non-finite geodesic entries at", cap)
        geo = np.where(np.isfinite(geo), geo, cap)
        np.fill_diagonal(geo, 0.0)
    if (geo < 0).any():
        raise RuntimeError("geodesic has a negative entry")
    np.fill_diagonal(geo, 0.0)
    return geo


def asinh_med(dist: np.ndarray) -> tuple[np.ndarray, float]:
    pos = dist[dist > 0]
    sig = float(np.median(pos)) if pos.size else 1.0
    sig = max(sig, 1e-9)
    out = np.arcsinh(dist / sig)
    np.fill_diagonal(out, 0.0)
    return out, sig


def landfold_asinh(dist: np.ndarray) -> tuple[np.ndarray, float]:
    out, sig = asinh_med(dist)
    out = out / ASINH_NORM
    np.fill_diagonal(out, 0.0)
    return out, sig


def orient(xy: np.ndarray, i: int = GM, j: int = ICO) -> np.ndarray:
    out = xy.copy()
    if out[i, 0] > out[j, 0]:
        out[:, 0] *= -1.0
    if out[i, 1] > 0.0:
        out[:, 1] *= -1.0
    return out


def unique_reps(dstruct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = dstruct.shape[0]
    uf = lc.UnionFind(n)
    for i in range(n):
        row = dstruct[i]
        for j in np.flatnonzero(row[i + 1 :] <= COPY_EPS):
            uf.union(i, i + 1 + int(j))
    labs = np.array([uf.find(i) for i in range(n)], dtype=int)
    reps = []
    seen = set()
    for i, r in enumerate(labs):
        if r not in seen:
            seen.add(int(r))
            reps.append(i)
    return np.asarray(reps, dtype=int), labs


def broadcast(xy_rep: np.ndarray, reps: np.ndarray, labs: np.ndarray) -> np.ndarray:
    root_of = {int(labs[r]): k for k, r in enumerate(reps)}
    xy = np.empty((len(labs), 2), dtype=float)
    for i, r in enumerate(labs):
        xy[i] = xy_rep[root_of[int(r)]]
    return xy


def smacof(dist: np.ndarray, dim: int = 2, max_iter: int = 80):
    from sklearn.manifold import MDS

    model = MDS(
        n_components=dim,
        metric=True,
        dissimilarity="precomputed",
        n_init=1,
        max_iter=max_iter,
        n_jobs=1,
        normalized_stress="auto",
        init="classical_mds",
    )
    xy = np.asarray(model.fit_transform(dist), dtype=float)
    return xy, float(getattr(model, "stress_", np.nan))


def _extent(xy: np.ndarray, pad: float):
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    return (
        xmin - pad * dx,
        xmax + pad * dx,
        ymin - pad * dy,
        ymax + pad * dy,
        dx,
        dy,
    )


def envelope(xy: np.ndarray, energy: np.ndarray, ngrid: int = 180, sigma: float = 1.6, pad: float = 0.12):
    """Blurred per-cell minimum of E - E_GM. Isolated tips keep their wells."""
    from scipy.ndimage import gaussian_filter

    xmin, xmax, ymin, ymax, _dx, _dy = _extent(xy, pad)
    xe = np.linspace(xmin, xmax, ngrid + 1)
    ye = np.linspace(ymin, ymax, ngrid + 1)
    ix = np.clip(np.digitize(xy[:, 0], xe) - 1, 0, ngrid - 1)
    iy = np.clip(np.digitize(xy[:, 1], ye) - 1, 0, ngrid - 1)
    grid = np.full((ngrid, ngrid), np.inf)
    for i, j, e in zip(iy, ix, energy):
        if e < grid[i, j]:
            grid[i, j] = e
    cap = float(np.nanmax(energy)) + 4.0
    filled = np.where(np.isfinite(grid), grid, cap)
    blur = gaussian_filter(filled, sigma=sigma, mode="nearest")
    body = gaussian_filter(np.isfinite(grid).astype(float), sigma=sigma) > 0.035
    rel = np.where(body, np.clip(blur - float(energy.min()), 0.0, EMAX), np.nan)
    gx = 0.5 * (xe[:-1] + xe[1:])
    gy = 0.5 * (ye[:-1] + ye[1:])
    return gx, gy, rel


def knn_idw(xy: np.ndarray, values: np.ndarray, ngrid: int = 180, k: int = 8, power: float = 3.0, pad: float = 0.12):
    from scipy.spatial import cKDTree

    xmin, xmax, ymin, ymax, dx, dy = _extent(xy, pad)
    gx = np.linspace(xmin, xmax, ngrid)
    gy = np.linspace(ymin, ymax, ngrid)
    xx, yy = np.meshgrid(gx, gy)
    key = np.round(xy, 8)
    _, uniq = np.unique(key, axis=0, return_index=True)
    pts = xy[uniq]
    val = values[uniq]
    tree = cKDTree(pts)
    q = np.column_stack([xx.ravel(), yy.ravel()])
    kk = min(k, len(pts))
    dist, idx = tree.query(q, k=kk)
    if kk == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    w = 1.0 / np.clip(dist, 1e-12, None) ** power
    field = ((w * val[idx]).sum(1) / np.clip(w.sum(1), 1e-18, None)).reshape(xx.shape)
    span = max(dx, dy)
    body = dist[:, 0].reshape(xx.shape) < 0.07 * span
    field = np.where(body, np.clip(field, 0.0, EMAX), np.nan)
    return gx, gy, field


def imq_field(xy: np.ndarray, values: np.ndarray, ngrid: int = 180, ell: float = 0.07, pad: float = 0.10):
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= pad * dx
    xmax += pad * dx
    ymin -= pad * dy
    ymax += pad * dy
    gx = np.linspace(xmin, xmax, ngrid)
    gy = np.linspace(ymin, ymax, ngrid)
    xx, yy = np.meshgrid(gx, gy)
    # collapse exact coordinate copies so a 183-fold liquid does not own the kernel
    key = np.round(xy, 8)
    _, uniq = np.unique(key, axis=0, return_index=True)
    pts = xy[uniq]
    val = values[uniq]
    ell_abs = ell * max(dx, dy)
    ell2 = ell_abs * ell_abs
    field = np.empty((ngrid, ngrid), dtype=float)
    step = 30
    for i0 in range(0, ngrid, step):
        i1 = min(i0 + step, ngrid)
        xs = xx[i0:i1].ravel()
        ys = yy[i0:i1].ravel()
        dx_ = xs[:, None] - pts[:, 0][None, :]
        dy_ = ys[:, None] - pts[:, 1][None, :]
        k = 1.0 / np.sqrt(1.0 + (dx_ * dx_ + dy_ * dy_) / ell2)
        num = k @ val
        den = np.clip(k.sum(1), 1e-12, None)
        field[i0:i1] = (num / den).reshape((i1 - i0, ngrid))
    return gx, gy, field, ell_abs


def idw_field(xy: np.ndarray, values: np.ndarray, ngrid: int = 180, power: float = 2.0, pad: float = 0.10):
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= pad * dx
    xmax += pad * dx
    ymin -= pad * dy
    ymax += pad * dy
    gx = np.linspace(xmin, xmax, ngrid)
    gy = np.linspace(ymin, ymax, ngrid)
    xx, yy = np.meshgrid(gx, gy)
    key = np.round(xy, 8)
    _, uniq = np.unique(key, axis=0, return_index=True)
    pts = xy[uniq]
    val = values[uniq]
    field = np.empty((ngrid, ngrid), dtype=float)
    step = 30
    for i0 in range(0, ngrid, step):
        i1 = min(i0 + step, ngrid)
        xs = xx[i0:i1].ravel()
        ys = yy[i0:i1].ravel()
        dx_ = xs[:, None] - pts[:, 0][None, :]
        dy_ = ys[:, None] - pts[:, 1][None, :]
        r2 = dx_ * dx_ + dy_ * dy_
        w = 1.0 / np.clip(r2 ** (power / 2.0), 1e-18, None)
        field[i0:i1] = ((w @ val) / np.clip(w.sum(1), 1e-18, None)).reshape((i1 - i0, ngrid))
    return gx, gy, field


def field_at(gx, gy, zz, p) -> float:
    j = int(np.argmin(np.abs(gx - p[0])))
    i = int(np.argmin(np.abs(gy - p[1])))
    return float(zz[i, j])


def local_minima(field: np.ndarray, vmax: float = 2.5, sep: int = 8):
    out = []
    ny, nx = field.shape
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            v = field[i, j]
            if not np.isfinite(v) or v > vmax:
                continue
            nb = field[i - 1 : i + 2, j - 1 : j + 2]
            if np.nanmin(nb) >= v - 1e-12:
                out.append((float(v), i, j))
    out.sort()
    kept = []
    for v, i, j in out:
        if all(abs(i - ii) + abs(j - jj) > sep for _, ii, jj in kept):
            kept.append((v, i, j))
    return kept


def downhill(field: np.ndarray, i: int, j: int):
    ny, nx = field.shape
    seen = set()
    while True:
        if (i, j) in seen:
            break
        seen.add((i, j))
        best = (field[i, j], i, j)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i + di, j + dj
                if 0 <= ii < ny and 0 <= jj < nx and np.isfinite(field[ii, jj]):
                    if field[ii, jj] < best[0] - 1e-15:
                        best = (field[ii, jj], ii, jj)
        if (best[1], best[2]) == (i, j):
            break
        i, j = best[1], best[2]
    return i, j, float(field[i, j])


def grid_index(gx, gy, p):
    j = int(np.argmin(np.abs(gx - p[0])))
    i = int(np.argmin(np.abs(gy - p[1])))
    return i, j


def score_energy(name, xy, energy, gx, gy, field):
    gm, ico = xy[GM], xy[ICO]
    sep = float(np.linalg.norm(gm - ico))
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    sep_norm = sep / max(diam, 1e-12)
    e_gm = field_at(gx, gy, field, gm)
    e_ico = field_at(gx, gy, field, ico)
    mins = local_minima(field)
    two_wells = len(mins) >= 2
    ig, jg = grid_index(gx, gy, gm)
    ii, ji = grid_index(gx, gy, ico)
    ag_i, ag_j, ag_v = downhill(field, ig, jg)
    ai_i, ai_j, ai_v = downhill(field, ii, ji)
    same_attractor = abs(ag_i - ai_i) + abs(ag_j - ai_j) <= 4
    two_basins = two_wells and not same_attractor
    # GM sits in a well when its downhill end is next to it and the field is deep
    hop = abs(ag_i - ig) + abs(ag_j - jg)
    finite = field[np.isfinite(field)]
    deep = bool(np.isfinite(e_gm) and finite.size and e_gm <= np.percentile(finite, 25))
    near_min = False
    if mins:
        gi, gj = ig, jg
        near_min = any(abs(i - gi) + abs(j - gj) <= 12 for _, i, j in mins)
    gm_in_well = (hop <= 12 or near_min) and deep
    rim = lc.on_hull(xy, GM)
    rec = {
        "name": name,
        "sep": sep,
        "sep_norm": sep_norm,
        "diam": diam,
        "E_GM": e_gm,
        "E_ico": e_ico,
        "n_wells": len(mins),
        "two_wells": two_wells,
        "two_basins": bool(two_basins),
        "gm_in_well": bool(gm_in_well),
        "gm_deeper": bool(np.isfinite(e_gm) and np.isfinite(e_ico) and e_gm < e_ico - 1e-6),
        "gm_on_rim": bool(rim),
        "gm_hop": int(hop),
        "well_vals": [float(v) for v, _, _ in mins[:6]],
        "gm": [float(gm[0]), float(gm[1])],
        "ico": [float(ico[0]), float(ico[1])],
        "pass": bool(sep_norm > SEP_BAR and two_basins and gm_in_well),
    }
    return rec


def save_pair(xy, gx, gy, field, energy, dest: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), facecolor="white")
    zz = np.clip(field, 0.0, EMAX)
    mesh = axes[0].contourf(
        gx, gy, zz, levels=np.linspace(0, EMAX, 21), cmap=PES, extend="max"
    )
    axes[0].contour(
        gx,
        gy,
        np.where(np.isfinite(zz), zz, np.nan),
        levels=np.linspace(0.2, EMAX - 0.3, 12),
        colors="#1a1a2e",
        linewidths=0.35,
    )
    rel = np.clip(energy - energy.min(), 0.0, EMAX)
    sc = axes[1].scatter(
        xy[:, 0], xy[:, 1], c=rel, s=8, cmap=PES, vmin=0, vmax=EMAX, linewidths=0, alpha=0.85
    )
    for ax in axes:
        ax.scatter(
            xy[GM, 0],
            xy[GM, 1],
            s=130,
            marker="*",
            c="k",
            edgecolors="white",
            linewidths=0.6,
            zorder=6,
            label=rf"GM ${lc.GM_E:.3f}$",
        )
        ax.scatter(
            xy[ICO, 0],
            xy[ICO, 1],
            s=75,
            marker="D",
            c="k",
            edgecolors="white",
            linewidths=0.6,
            zorder=6,
            label=rf"ico ${lc.ICO_E:.3f}$",
        )
        ax.legend(fontsize=7, frameon=True, fancybox=False, loc="best")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_title(title + r"  IDW $E-E_{\mathrm{GM}}$")
    axes[1].set_title(title + r"  $E-E_{\mathrm{GM}}$")
    fig.colorbar(mesh, ax=axes[0], fraction=0.046, pad=0.03).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.03).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", dest)


def write_xy(path: Path, xy: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, xy, fmt="%.8e")


def main() -> None:
    t_all = _now()
    energy, coords = load_min(MINFILE)
    n = len(energy)
    if ENERGY.is_file():
        e2 = np.loadtxt(ENERGY)
        if len(e2) == n and float(np.max(np.abs(e2 - energy))) > 1e-6:
            print("WARN energy file disagrees with .min; using .min")
    if abs(float(energy[GM]) - lc.GM_E) > 1e-4:
        raise SystemExit("index 0 is not GM: %s" % energy[GM])
    if abs(float(energy[ICO]) - lc.ICO_E) > 1e-4:
        raise SystemExit("index 40 is not ico: %s" % energy[ICO])
    print("n", n, "natoms", coords[0].shape[0], "E[GM]", float(energy[GM]), "E[ico]", float(energy[ICO]))

    t0 = _now()
    fps = fingerprints(coords)
    print("fp", fps.shape, "t", "%.2fs" % (_now() - t0))

    dstruct, src = structure_d(fps)
    pos = dstruct[np.triu_indices(n, 1)]
    print(
        "d_struct src",
        src,
        "min",
        float(pos.min()),
        "med",
        float(np.median(pos)),
        "max",
        float(pos.max()),
        "GM-ico",
        float(dstruct[GM, ICO]),
        "GM-copy",
        float(dstruct[GM, 1]),
    )

    n_edges = 0
    comps = 1
    if GEO_NPY.is_file():
        geo = np.load(GEO_NPY).astype(np.float64)
        if geo.shape != (n, n):
            raise SystemExit("stale geo %s" % (geo.shape,))
        print("reuse", GEO_NPY, geo.shape, "geo(GM,ico)", float(geo[GM, ICO]))
    else:
        edges, n_copy, n_pick = knn_edges(dstruct, energy, knn=KNN, lam=LAM)
        comps, labs = n_components(n, edges)
        print(
            "kNN",
            KNN,
            "edges",
            len(edges),
            "components",
            comps,
            "copy_links",
            n_copy,
            "distinct_picks",
            n_pick,
            "cost_min",
            float(min(e[4] for e in edges)),
            "cost_med",
            float(np.median([e[4] for e in edges])),
        )
        if comps > 1:
            edges = stitch_components(dstruct, energy, edges, labs, lam=LAM)
            comps, labs = n_components(n, edges)
            print("after stitch components", comps, "edges", len(edges))
        t0 = _now()
        geo = dijkstra_all(n, edges)
        n_edges = len(edges)
        print(
            "dijkstra",
            "%.1fs" % (_now() - t0),
            "geo(GM,ico)",
            float(geo[GM, ICO]),
            "geo_med",
            float(np.median(geo[geo > 0])),
            "geo_max",
            float(geo.max()),
            "finite",
            bool(np.isfinite(geo).all()),
        )

    reps, copy_labs = unique_reps(dstruct)
    print("n_unique", len(reps), "GM_rep", int(GM in set(reps)), "ico_rep", int(ICO in set(reps)))

    f_geo, sig_geo = landfold_asinh(geo)
    ash_geo, _ = asinh_med(geo)
    print("sig_geo", sig_geo, "Fgeo(GM,ico)", float(f_geo[GM, ICO]), "asinh_norm", ASINH_NORM)

    maps = {}
    evs = {}
    extra = {}

    xy, ev = lc.torgerson(f_geo, 2)
    maps["asinh"] = orient(xy)
    evs["asinh"] = ev

    xy, ev = lc.torgerson(geo, 2)
    maps["raw"] = orient(xy)
    evs["raw"] = ev

    xy, ev = lc.torgerson(ash_geo, 2)
    maps["asinh_med"] = orient(xy)
    evs["asinh_med"] = ev

    geo_u = geo[np.ix_(reps, reps)]
    f_u, _ = landfold_asinh(geo_u)
    xy_u, ev_u = lc.torgerson(f_u, 2)
    maps["asinh_unique"] = orient(broadcast(xy_u, reps, copy_labs))
    evs["asinh_unique"] = ev_u

    smacof_xy = DEST / "asinh_smacof.xy"
    if smacof_xy.is_file():
        maps["asinh_smacof"] = np.loadtxt(smacof_xy)
        print("reuse", smacof_xy)
    else:
        try:
            xy_s, stress = smacof(f_u, max_iter=80)
            maps["asinh_smacof"] = orient(broadcast(xy_s, reps, copy_labs))
            extra["smacof_stress"] = stress
            print("smacof stress", stress)
        except Exception as exc:
            print("skip smacof", type(exc).__name__, exc)

    DEST.mkdir(parents=True, exist_ok=True)
    write_xy(DEST / "geo_asinh.xy", maps["asinh"])
    np.save(DEST / "geo.npy", geo.astype(np.float32))
    np.save(DEST / "fp.npy", fps.astype(np.float32))

    rel = np.clip(energy - float(energy.min()), 0.0, None)
    scores = []
    fields = {}
    for name, xy in maps.items():
        gx, gy, field = knn_idw(xy, rel, ngrid=180, k=8, power=3.0)
        rec = score_energy(name, xy, energy, gx, gy, field)
        rec["fill"] = "knn_idw"
        scores.append(rec)
        fields[name] = (gx, gy, field)
        print(
            f"{name:16s} sep_norm={rec['sep_norm']:.6f}  sep={rec['sep']:.6f}  "
            f"diam={rec['diam']:.6f}  wells={rec['n_wells']}  "
            f"Egm={rec['E_GM']:.3f}  Eico={rec['E_ico']:.3f}  "
            f"basins={rec['two_basins']}  gm_well={rec['gm_in_well']}  "
            f"rim={rec['gm_on_rim']}  hop={rec['gm_hop']}  pass={rec['pass']}  "
            f"wellsE={rec['well_vals']}"
        )
        save_pair(xy, gx, gy, field, energy, DEST / f"{name}.png", name)
        write_xy(DEST / f"{name}.xy", xy)

    # prefer a passing asinh map; otherwise the largest sep_norm among asinh* that has GM in a well
    def rank(rec):
        return (
            int(rec["pass"]),
            int(rec["gm_in_well"]),
            int(rec["two_basins"]),
            rec["sep_norm"],
        )

    ranked = sorted(
        [r for r in scores if r["name"] in maps],
        key=rank,
        reverse=True,
    )
    best = ranked[0]
    asinh_ok = [r for r in ranked if r["name"].startswith("asinh")]
    if asinh_ok:
        best = asinh_ok[0]
    winner = best["name"]
    print("winner", winner, "sep_norm", best["sep_norm"], "gm_in_well", best["gm_in_well"], "two_basins", best["two_basins"])

    gx, gy, field = fields[winner]
    save_pair(
        maps[winner],
        gx,
        gy,
        field,
        energy,
        DEST / "elja_occ_lj38_geostruc.png",
        r"structure kNN geodesic",
    )
    save_pair(
        maps[winner],
        gx,
        gy,
        field,
        energy,
        OUT / "elja_occ_lj38_geostruc.png",
        r"structure kNN geodesic",
    )
    write_xy(DEST / "geostruc.xy", maps[winner])

    payload = {
        "n": n,
        "n_unique": int(len(reps)),
        "knn": KNN,
        "lambda": LAM,
        "cost": "|Ei-Ej|+2 d_struct",
        "d_struct": src,
        "d_struct_gm_ico": float(dstruct[GM, ICO]),
        "n_edges": n_edges,
        "n_components": comps,
        "geo_gm_ico": float(geo[GM, ICO]),
        "sig_geo": sig_geo,
        "asinh_norm": ASINH_NORM,
        "gm": GM,
        "ico": ICO,
        "winner": winner,
        "scores": scores,
        "ev": {k: [float(x) for x in v] for k, v in evs.items()},
        "extra": extra,
        "seconds": float(_now() - t_all),
    }
    (DEST / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", DEST / "scores.json")
    print("--- verdict ---")
    print(
        f"sep_norm GM-ico {best['sep_norm']:.6f}  wells={best['n_wells']}  "
        f"two_basins={best['two_basins']}  gm_in_well={best['gm_in_well']}  "
        f"pass={best['pass']}"
    )
    print("paths", DEST, OUT / "elja_occ_lj38_geostruc.png")


if __name__ == "__main__":
    main()
