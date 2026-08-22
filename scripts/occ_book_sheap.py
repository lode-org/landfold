#!/usr/bin/env python3
"""SHEAP analogue for the 4042 Elja LJ38 inherent structures.

Shires & Pickard, Phys. Rev. X 11, 041026 (2021): spring / manifold
layout of minima from a structure descriptor, energy as the landscape
coordinate so the GM sits at the funnel tip.

This is not CN-histogram steepest-descent attractor MDS (that map is
seven blobs). Graph support is kNN in sorted-pair-distance L2. Edge
length is that structure distance. Energy is a third force: low E is
pulled toward the origin (funnel radius). A second candidate is the
positive barrier ultrametric max(E) along those same kNN paths.

Occupancy leftover-well invert is not used. The filled field is a GP
of E - E_GM on the embedding.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ceriotti-figs"
MINFILE = Path("/tmp/occ-book/lj38_0013.min")
DEST = Path("/tmp/occ-book/cand-sheap")
DPAIR_CACHE = Path("/tmp/occ-book/lj38_dpair.dist")
KABSCH_CACHE = Path("/tmp/occ-book/lj38_kabsch.dist")

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)
GM_E = -173.928427
ICO_E = -173.252378
GM_IDX = 0
ICO_IDX = 40
N_ATOMS = 38
KNN0 = 12
EMAX = 6.0
DUP_EPS = 5e-2
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


def load_min(path: Path, n_atoms: int = N_ATOMS):
    energies = []
    frames = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        nums = [float(x) for x in line.split()]
        energy, coords = nums[0], nums[1:]
        if len(coords) != n_atoms * 3:
            raise SystemExit(f"{path}: expected {n_atoms * 3} coords, got {len(coords)}")
        energies.append(energy)
        frames.append(np.asarray(coords, dtype=np.float64).reshape(n_atoms, 3))
    return np.asarray(energies, dtype=np.float64), frames


def pair_pdist(pos: np.ndarray) -> np.ndarray:
    """Sorted pairwise distances, C(n,2) vector. Translation/rotation/perm-ready."""
    d = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
    iu = np.triu_indices(pos.shape[0], 1)
    return np.sort(r[iu])


def coulomb_eigs(pos: np.ndarray) -> np.ndarray:
    """Sorted Coulomb-matrix eigenvalues, Z = 1."""
    n = pos.shape[0]
    d = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
    np.fill_diagonal(r, np.inf)
    c = 1.0 / r
    np.fill_diagonal(c, 0.5)
    return np.sort(np.linalg.eigvalsh(c))[::-1]


def spd_matrix(frames) -> np.ndarray:
    m = N_ATOMS * (N_ATOMS - 1) // 2
    out = np.empty((len(frames), m), dtype=np.float64)
    for i, p in enumerate(frames):
        out[i] = pair_pdist(p)
    return out


def pairwise_l2(desc: np.ndarray, chunk: int = 256) -> np.ndarray:
    n = desc.shape[0]
    nrm = np.einsum("ij,ij->i", desc, desc)
    dist = np.empty((n, n), dtype=np.float64)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        gram = desc[i0:i1] @ desc.T
        d2 = nrm[i0:i1, None] + nrm[None, :] - 2.0 * gram
        np.maximum(d2, 0.0, out=d2)
        dist[i0:i1] = np.sqrt(d2)
    np.fill_diagonal(dist, 0.0)
    return dist


def load_or_compute_dpair(frames, dest: Path) -> np.ndarray:
    for path in (
        dest / "dpair.npy",
        DPAIR_CACHE.with_suffix(".npy"),
        DPAIR_CACHE,
        KABSCH_CACHE,
        dest / "dpair.dist",
    ):
        if not path.is_file():
            continue
        print("load structure D", path)
        if path.suffix == ".npy":
            d = np.load(path)
        else:
            d = np.loadtxt(path)
        if d.shape[0] == len(frames) and d.shape[1] == len(frames):
            np.fill_diagonal(d, 0.0)
            return d
        print("skip", path, "shape", d.shape)
    print("compute sorted-pair-distance L2")
    t0 = time.time()
    spd = spd_matrix(frames)
    d = pairwise_l2(spd)
    print(
        "spd",
        spd.shape,
        "D med",
        float(np.median(d[d > 0])),
        "D(GM,ico)",
        float(d[GM_IDX, ICO_IDX]),
        f"dt {time.time() - t0:.2f}s",
    )
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "dpair.npy", d)
    try:
        np.save(DPAIR_CACHE.with_suffix(".npy"), d)
    except OSError:
        pass
    return d


def unique_reps(dist: np.ndarray, energy: np.ndarray, eps: float = DUP_EPS):
    """Greedy unique structures, grouped by energy. Occupancy copies share a rep."""
    n = len(energy)
    assign = np.full(n, -1, dtype=int)
    reps = []
    key = np.round(energy, 6)
    for e in np.unique(key):
        idx = np.where(key == e)[0]
        for i in idx:
            i = int(i)
            if assign[i] >= 0:
                continue
            reps.append(i)
            close = dist[i, idx] < eps
            for j, flag in zip(idx, close):
                if flag and assign[j] < 0:
                    assign[j] = i
            assign[i] = i
    if (assign < 0).any():
        raise RuntimeError("unique_reps left unassigned sites")
    return np.asarray(reps, dtype=int), assign


def knn_from_dist(dist: np.ndarray, knn: int, skip_eps: float = DUP_EPS):
    """kNN among distinct structures (D > skip_eps). Symmetrised later."""
    n = dist.shape[0]
    neigh = np.full((n, knn), -1, dtype=int)
    ndist = np.full((n, knn), np.inf)
    for i in range(n):
        row = dist[i].copy()
        row[i] = np.inf
        row[row <= skip_eps] = np.inf
        finite = np.isfinite(row)
        take = min(knn, int(finite.sum()))
        if take <= 0:
            continue
        idx = np.argpartition(row, take - 1)[:take]
        idx = idx[np.argsort(row[idx])]
        neigh[i, :take] = idx
        ndist[i, :take] = row[idx]
    return neigh, ndist


def symmetrize_knn(neigh, ndist, dist, knn: int):
    """Undirected kNN: edge if j in kNN(i) or i in kNN(j)."""
    n = neigh.shape[0]
    adj = [[] for _ in range(n)]
    seen = set()
    for i in range(n):
        for k in range(knn):
            j = int(neigh[i, k])
            if j < 0:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            w = float(dist[i, j])
            adj[i].append((j, w))
            adj[j].append((i, w))
    return adj, seen


def n_components(n: int, adj) -> int:
    seen = np.zeros(n, dtype=bool)
    comps = 0
    for s in range(n):
        if seen[s]:
            continue
        comps += 1
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            for v, _w in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return comps


def steepest_basins(energy: np.ndarray, adj):
    n = len(energy)
    parent = np.arange(n)
    best = energy.copy()
    for i in range(n):
        for j, _w in adj[i]:
            if energy[j] < best[i] - 1e-12:
                best[i] = energy[j]
                parent[i] = j
    attract = np.empty(n, dtype=int)
    for i in range(n):
        x = i
        seen = set()
        while parent[x] != x and x not in seen:
            seen.add(x)
            x = int(parent[x])
        attract[i] = x
    return attract


def laplacian_xy(n: int, adj) -> np.ndarray:
    wmat = np.zeros((n, n))
    for i in range(n):
        for j, d in adj[i]:
            wmat[i, j] = 1.0 / max(d, 1e-9)
    deg = wmat.sum(1)
    dinv = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
    lsym = np.eye(n) - (dinv[:, None] * wmat * dinv[None, :])
    _eval, evec = np.linalg.eigh(lsym)
    return evec[:, 1:3].copy()


def place_gm_ico(xy: np.ndarray, gm: int, ico: int) -> np.ndarray:
    out = xy - xy[gm]
    vec = out[ico]
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-15:
        return out
    ang = np.arctan2(vec[1], vec[0])
    c, s = np.cos(-ang), np.sin(-ang)
    rot = np.array([[c, -s], [s, c]])
    return out @ rot.T


def radial_remap(xy: np.ndarray, energy: np.ndarray, alpha: float, gm: int) -> np.ndarray:
    rel = np.clip(energy - energy[gm], 0.0, None)
    scale = float(np.median(rel[rel > 0])) if np.any(rel > 0) else 1.0
    scale = max(scale, 1e-9)
    r = np.power(rel / scale, alpha)
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    # GM (r=0) keeps origin; undefined angle is fine
    out = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    out[gm] = 0.0
    return out


def spring_embed(
    adj,
    energy: np.ndarray,
    xy0: np.ndarray,
    gm: int,
    knn_rest_med: float,
    k_spring: float,
    k_rad: float,
    alpha: float,
    n_iter: int = 140,
) -> np.ndarray:
    """kNN springs (rest = structure D) + radial energy force toward origin."""
    n = len(energy)
    ii, jj, rest = [], [], []
    for i in range(n):
        for j, w in adj[i]:
            if j <= i:
                continue
            ii.append(i)
            jj.append(j)
            rest.append(w)
    ii = np.asarray(ii, dtype=int)
    jj = np.asarray(jj, dtype=int)
    rest = np.asarray(rest, dtype=np.float64) / max(knn_rest_med, 1e-12)
    rel = np.clip(energy - energy[gm], 0.0, None)
    scale = float(np.median(rel[rel > 0])) if np.any(rel > 0) else 1.0
    r_tgt = np.power(rel / max(scale, 1e-9), alpha)
    xy = xy0.copy()
    xy = xy - xy[gm]
    for it in range(n_iter):
        lr = 0.10 * (1.0 - it / n_iter) + 0.008
        force = np.zeros_like(xy)
        dvec = xy[ii] - xy[jj]
        d = np.sqrt(np.einsum("ij,ij->i", dvec, dvec)) + 1e-12
        pull = ((rest - d) / d)[:, None] * dvec
        np.add.at(force, ii, k_spring * pull)
        np.add.at(force, jj, -k_spring * pull)
        r = np.sqrt(np.einsum("ij,ij->i", xy, xy)) + 1e-12
        force += -k_rad * ((r - r_tgt) / r)[:, None] * xy
        force[gm] = 0.0
        # step cap
        fn = np.sqrt(np.einsum("ij,ij->i", force, force))
        cap = 0.35
        big = fn > cap
        force[big] *= (cap / fn[big])[:, None]
        xy += lr * force
        xy[gm] = 0.0
    return xy


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


def pairwise_euclid(xy: np.ndarray) -> np.ndarray:
    gram = xy @ xy.T
    sq = np.clip(np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram, 0.0, None)
    d = np.sqrt(sq)
    np.fill_diagonal(d, 0.0)
    return d


def smacof(delta: np.ndarray, xy0: np.ndarray, maxiter: int = 80, rtol: float = 1e-7):
    n = delta.shape[0]
    x = xy0 - xy0.mean(0)
    prev = np.inf
    stress = np.inf
    it = 0
    for it in range(maxiter):
        d = pairwise_euclid(x)
        ratio = np.zeros_like(delta)
        nz = d > 1e-15
        ratio[nz] = delta[nz] / d[nz]
        b = -ratio
        np.fill_diagonal(b, 0.0)
        b[np.diag_indices(n)] = -b.sum(axis=1)
        x = (b @ x) / n
        x -= x.mean(0)
        d = pairwise_euclid(x)
        stress = 0.5 * float(np.sum((d - delta) ** 2))
        if prev < np.inf and abs(prev - stress) <= rtol * max(prev, 1e-12):
            break
        prev = stress
    den = float(np.sum(delta * delta))
    stress1 = float(np.sqrt(max(2.0 * stress, 0.0) / max(den, 1e-300)))
    return x, stress, stress1, it + 1


def barrier_ultrametric(energy: np.ndarray, adj):
    """Kruskal merge height = max(E_i, E_j) on structure kNN edges. Positive."""
    n = len(energy)
    edges = []
    seen = set()
    for i in range(n):
        for j, _w in adj[i]:
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            h = float(max(energy[i], energy[j]))
            edges.append((h, i, j))
    edges.sort(key=lambda t: t[0])
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
    if np.isnan(merge).any():
        cap = float(np.nanmax(merge)) + 1.0
        merge = np.where(np.isnan(merge), cap, merge)
    dist = np.clip(merge - np.minimum.outer(energy, energy), 0.0, None)
    np.fill_diagonal(dist, 0.0)
    if (dist < 0).any():
        raise RuntimeError("negative barrier ultrametric")
    return dist, merge


def committor(adj, src, sink):
    n = len(adj)
    src = np.atleast_1d(src).astype(int)
    sink = np.atleast_1d(sink).astype(int)
    a = np.zeros((n, n))
    for i in range(n):
        for j, w in adj[i]:
            a[i, j] = 1.0 / max(w, 1e-9)
    deg = np.clip(a.sum(1), 1e-300, None)
    p = a / deg[:, None]
    q = np.zeros(n)
    q[sink] = 1.0
    mask = np.ones(n, dtype=bool)
    mask[src] = False
    mask[sink] = False
    tix = np.where(mask)[0]
    if tix.size == 0:
        return q
    i_minus_p = np.eye(tix.size) - p[np.ix_(tix, tix)]
    rhs = p[np.ix_(tix, sink)].sum(1)
    try:
        q[tix] = np.linalg.solve(i_minus_p, rhs)
    except np.linalg.LinAlgError:
        q[tix] = np.linalg.lstsq(i_minus_p, rhs, rcond=None)[0]
    return np.clip(q, 0.0, 1.0)


def polar_energy(q: np.ndarray, energy: np.ndarray, gm: int, alpha: float) -> np.ndarray:
    rel = np.clip(energy - energy[gm], 0.0, None)
    scale = float(np.median(rel[rel > 0])) if np.any(rel > 0) else 1.0
    r = np.power(rel / max(scale, 1e-9), alpha)
    ang = np.clip(q, 0.0, 1.0) * np.pi
    xy = np.column_stack((r * np.cos(ang), r * np.sin(ang)))
    xy[gm] = 0.0
    return xy


def _blur2d(a: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    """Separable Gaussian blur; odd kernel."""
    r = max(int(3.0 * sigma), 1)
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(a, ((0, 0), (r, r)), mode="edge")
    tmp = np.apply_along_axis(lambda row: np.convolve(row, k, mode="valid"), 1, pad)
    pad = np.pad(tmp, ((r, r), (0, 0)), mode="edge")
    return np.apply_along_axis(lambda col: np.convolve(col, k, mode="valid"), 0, pad)


def imq_energy(xy: np.ndarray, energy: np.ndarray, ngrid: int = 130, ell: float = 0.045):
    rel = np.clip(energy - energy.min(), 0.0, None)
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
    # inducing: every low-energy site plus a farthest-point cover
    key = np.round(xy, 5)
    _, uniq = np.unique(key, axis=0, return_index=True)
    low = np.where(rel <= 2.5)[0]
    must = np.array(
        [int(np.argmin(rel)), int(np.argmin(np.abs(energy - ICO_E)))], dtype=int
    )
    if uniq.size > 280:
        rng = np.random.default_rng(1)
        rest = np.setdiff1d(uniq, np.concatenate([low, must]))
        take = min(220, rest.size)
        pick = np.unique(np.concatenate([low, must, rng.choice(rest, take, replace=False)]))
        pts = xy[pick]
        val = np.clip(rel[pick], 0.0, EMAX)
    else:
        pts = xy[uniq]
        val = np.clip(rel[uniq], 0.0, EMAX)
    ell2 = (ell * max(dx, dy)) ** 2
    xs = xx.ravel()
    ys = yy.ravel()
    field = np.empty(xs.shape)
    step = 2500
    for i0 in range(0, xs.size, step):
        i1 = min(i0 + step, xs.size)
        dx_ = xs[i0:i1, None] - pts[:, 0][None, :]
        dy_ = ys[i0:i1, None] - pts[:, 1][None, :]
        k = 1.0 / np.sqrt(1.0 + (dx_ * dx_ + dy_ * dy_) / max(ell2, 1e-12))
        field[i0:i1] = (k @ val) / np.clip(k.sum(1), 1e-12, None)
    field = field.reshape(xx.shape)
    counts, _, _ = np.histogram2d(
        x, y, bins=ngrid, range=[[xmin, xmax], [ymin, ymax]]
    )
    rho = _blur2d(counts.T, sigma=2.2)
    rmax = float(rho.max()) if float(rho.max()) > 0 else 1.0
    mask = rho > 0.002 * rmax
    # keep the two funnel tips even when the cloud is sparse there
    for tip in (xy[int(np.argmin(rel))], xy[int(np.argmin(np.abs(energy - ICO_E)))]):
        mask |= (xx - tip[0]) ** 2 + (yy - tip[1]) ** 2 <= (0.12 * max(dx, dy)) ** 2
    zg = np.where(mask, np.clip(field, 0.0, EMAX), np.nan)
    return gx, gy, zg


def local_minima_field(zg: np.ndarray):
    out = []
    ny, nx = zg.shape
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            v = zg[i, j]
            if not np.isfinite(v):
                continue
            nb = zg[i - 1 : i + 2, j - 1 : j + 2]
            if np.nanmin(nb) >= v - 1e-12 and v < 1.25:
                out.append((v, i, j))
    out.sort()
    kept = []
    for v, i, j in out:
        if all(abs(i - ii) + abs(j - jj) > 10 for _, ii, jj in kept):
            kept.append((v, i, j))
    return kept


def structure_labels(dist: np.ndarray, gm: int, ico: int) -> np.ndarray:
    return (dist[:, ico] < dist[:, gm]).astype(int)


def score_layout(name, xy, energy, gm, ico, labels, attract):
    r = np.sqrt((xy ** 2).sum(1))
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    sep = float(np.linalg.norm(xy[gm] - xy[ico]))
    sep_n = sep / max(diam, 1e-12)
    r_gm = float(r[gm])
    r_ico = float(r[ico])
    # GM is the point nearest the origin among the lowest 2% energy
    low = energy <= np.quantile(energy, 0.02)
    gm_tip = bool(r_gm <= r[low].min() + 1e-9)
    gm_center = bool(r_gm < 0.08 * max(r.max(), 1e-12))
    ico_off = bool(r_ico > 0.12 * max(r.max(), 1e-12))
    # angular split of structure labels (fcc-like vs ico-like)
    ang = np.arctan2(xy[:, 1], xy[:, 0])
    a0 = ang[labels == 0]
    a1 = ang[labels == 1]
    # circular mean difference
    def cmean(a):
        return np.arctan2(np.sin(a).mean(), np.cos(a).mean()) if a.size else 0.0

    dang = abs(np.arctan2(np.sin(cmean(a1) - cmean(a0)), np.cos(cmean(a1) - cmean(a0))))
    # silhouette in the plane
    sil = 0.0
    if (labels == 0).sum() > 1 and (labels == 1).sum() > 1:
        m0 = xy[labels == 0].mean(0)
        m1 = xy[labels == 1].mean(0)
        da = np.linalg.norm(xy[labels == 0] - m0, axis=1).mean()
        db = np.linalg.norm(xy[labels == 1] - m1, axis=1).mean()
        sil = float(np.linalg.norm(m0 - m1) / max(da + db, 1e-12))
    n_attr = int(len(np.unique(attract)))
    n_gm = int((attract == attract[gm]).sum())
    n_ico = int((attract == attract[ico]).sum())
    two_attr = n_attr == 2 or (
        n_attr <= 4 and n_gm > 0 and n_ico > 0 and attract[gm] != attract[ico]
    )
    seven_fail = n_attr >= 6
    rec = {
        "name": name,
        "sep": sep,
        "sep_norm": sep_n,
        "diam": diam,
        "r_gm": r_gm,
        "r_ico": r_ico,
        "gm_tip": gm_tip,
        "gm_center": gm_center,
        "ico_off": ico_off,
        "ang_split": float(dang),
        "silhouette": sil,
        "n_attract": n_attr,
        "n_gm": n_gm,
        "n_ico": n_ico,
        "two_attr": bool(two_attr),
        "seven_fail": bool(seven_fail),
        "gm": [float(xy[gm, 0]), float(xy[gm, 1])],
        "ico": [float(xy[ico, 0]), float(xy[ico, 1])],
    }
    rec["score"] = (
        2.2 * float(gm_center)
        + 1.6 * float(gm_tip)
        + 1.8 * float(ico_off)
        + 1.4 * min(sil, 2.0)
        + 0.8 * (dang / np.pi)
        + 1.2 * float(two_attr and not seven_fail)
        + 0.6 * min(sep_n / 0.35, 1.5)
        - 2.5 * float(seven_fail)
        - 1.5 * float(r_gm > 0.15 * max(r.max(), 1e-12))
    )
    return rec


def mark(ax, xy, gm, ico):
    h1 = ax.scatter(
        xy[gm, 0],
        xy[gm, 1],
        s=130,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=8,
        label=rf"GM ${GM_E:.3f}$",
    )
    h2 = ax.scatter(
        xy[ico, 0],
        xy[ico, 1],
        s=75,
        marker="D",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=8,
        label=rf"ico ${ICO_E:.3f}$",
    )
    ax.legend(
        handles=[h1, h2],
        loc="upper left",
        fontsize=8,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="k",
    )


def paint_energy(ax, gx, gy, zg):
    mesh = ax.contourf(
        gx, gy, zg, levels=np.linspace(0, EMAX, 21), cmap=PES, extend="max"
    )
    ax.contour(
        gx,
        gy,
        np.where(np.isfinite(zg), zg, np.nan),
        levels=np.linspace(0.4, EMAX - 0.4, 8),
        colors="#1a1a2e",
        linewidths=0.3,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return mesh


def save_figure(xy, energy, gm, ico, dest: Path, title: str, also: Path | None = None):
    gx, gy, zg = imq_energy(xy, energy)
    mins = local_minima_field(zg)
    rel = np.clip(energy - energy.min(), 0.0, EMAX)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), facecolor="white")
    mesh = paint_energy(axes[0], gx, gy, zg)
    mark(axes[0], xy, gm, ico)
    axes[0].set_title(title + r"  $E-E_{\mathrm{GM}}$ GP")
    fig.colorbar(mesh, ax=axes[0], fraction=0.046, pad=0.03).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    psz = 7 + 18 * np.exp(-rel / 1.6)
    sc = axes[1].scatter(
        xy[:, 0], xy[:, 1], c=rel, s=psz, cmap=PES, vmin=0, vmax=6, linewidths=0, zorder=2
    )
    mark(axes[1], xy, gm, ico)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title(title + r"  points")
    for sp in axes[1].spines.values():
        sp.set_visible(False)
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.03).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", dest, "E-wells", len(mins))

    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor="white")
    mesh = paint_energy(ax, gx, gy, zg)
    psz = 6 + 16 * np.exp(-rel / 1.6)
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=rel,
        s=psz,
        cmap=PES,
        vmin=0,
        vmax=6,
        linewidths=0,
        zorder=3,
        alpha=0.88,
    )
    mark(ax, xy, gm, ico)
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    fig.tight_layout()
    single = dest.with_name(dest.stem + "_single.png")
    fig.savefig(single, dpi=170, facecolor="white")
    if also is not None:
        also.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(also, dpi=170, facecolor="white")
        print("wrote", also)
    plt.close(fig)
    print("wrote", single)
    return mins


def expand_xy(xy_u: np.ndarray, reps: np.ndarray, assign: np.ndarray) -> np.ndarray:
    rmap = {int(r): k for k, r in enumerate(reps)}
    out = np.empty((len(assign), 2))
    for i, a in enumerate(assign):
        out[i] = xy_u[rmap[int(a)]]
    return out


def main() -> None:
    t_all = time.time()
    DEST.mkdir(parents=True, exist_ok=True)
    energy, frames = load_min(MINFILE)
    n = len(energy)
    if abs(float(energy[GM_IDX]) - GM_E) > 1e-4:
        raise SystemExit(f"index 0 is not GM: {energy[GM_IDX]}")
    if abs(float(energy[ICO_IDX]) - ICO_E) > 1e-4:
        raise SystemExit(f"index 40 is not ico: {energy[ICO_IDX]}")
    print("n", n, "E[GM]", float(energy[GM_IDX]), "E[ico]", float(energy[ICO_IDX]))

    dist = load_or_compute_dpair(frames, DEST)
    print(
        "D(GM,ico)",
        float(dist[GM_IDX, ICO_IDX]),
        "D(0,1)",
        float(dist[0, 1]),
        "D(40,41)",
        float(dist[40, 41]) if n > 41 else None,
    )

    reps, assign = unique_reps(dist, energy, eps=DUP_EPS)
    print("unique structures", len(reps), "of", n, "dup_eps", DUP_EPS)
    d_u = dist[np.ix_(reps, reps)]
    e_u = energy[reps]
    gm_u = int(np.where(reps == assign[GM_IDX])[0][0])
    ico_u = int(np.where(reps == assign[ICO_IDX])[0][0])
    if gm_u != int(np.argmin(e_u)):
        print("WARN unique GM slot", gm_u, "argmin", int(np.argmin(e_u)))
    print(
        "unique GM/ico",
        gm_u,
        ico_u,
        "E",
        float(e_u[gm_u]),
        float(e_u[ico_u]),
        "D",
        float(d_u[gm_u, ico_u]),
    )
    labels_u = structure_labels(d_u, gm_u, ico_u)
    print(
        "structure labels n_fcc-like",
        int((labels_u == 0).sum()),
        "n_ico-like",
        int((labels_u == 1).sum()),
    )

    knn = 12
    neigh, ndist = knn_from_dist(d_u, knn, skip_eps=DUP_EPS)
    adj, _seen = symmetrize_knn(neigh, ndist, d_u, knn)
    comps = n_components(len(reps), adj)
    attract = steepest_basins(e_u, adj)
    n_attr = int(len(np.unique(attract)))
    rest_vals = ndist[np.isfinite(ndist)]
    rest_med = float(np.median(rest_vals)) if rest_vals.size else 1.0
    print(
        f"kNN={knn} edges~{sum(len(a) for a in adj)//2} comps={comps} "
        f"steepest_attr={n_attr} (structure graph diagnostic; not the figure) "
        f"n_GM={int((attract == attract[gm_u]).sum())} "
        f"n_ico={int((attract == attract[ico_u]).sum())}"
    )

    # Structure angle: random-walk committor on the structure kNN graph
    # from the GM to the ico. SPD Voronoi is useless here (the octahedron
    # is isolated; only the GM itself is closer to itself than to ico).
    q = committor(adj, [gm_u], [ico_u])
    q[gm_u] = 0.0
    q[ico_u] = 1.0
    gm_b = attract == attract[gm_u]
    ico_b = attract == attract[ico_u]
    print(
        "q percentiles",
        [float(x) for x in np.percentile(q, [0, 5, 25, 50, 75, 95, 100])],
        "q GM-basin",
        float(q[gm_b].mean()),
        float(q[gm_b].min()),
        float(q[gm_b].max()),
        "q ico-basin",
        float(q[ico_b].mean()),
        float(q[ico_b].min()),
        float(q[ico_b].max()),
    )
    labels_funnel = ico_b.astype(int)
    labels_funnel[gm_b] = 0
    # Within-basin opening: 2nd Laplacian coordinate, scaled to [-1, 1].
    xy_lap = laplacian_xy(len(reps), adj)
    u = xy_lap[:, 1].copy()
    u = (u - np.median(u)) / (np.std(u) + 1e-12)
    u = np.clip(u, -1.5, 1.5) / 1.5

    def funnel_xy(alpha: float, sep: float = 3.2) -> np.ndarray:
        """Two tips: GM at origin, ico at (sep, 0). Energy is radius from home."""
        rel_g = np.clip(e_u - e_u[gm_u], 0.0, None)
        rel_i = np.clip(e_u - e_u[ico_u], 0.0, None)
        r_g = np.power(rel_g, alpha)
        r_i = np.power(rel_i, alpha)
        xy = np.zeros((len(reps), 2))
        # fcc funnel opens up-right from the GM
        phi_g = 0.58 * np.pi + 0.28 * u
        xy[gm_b, 0] = r_g[gm_b] * np.cos(phi_g[gm_b])
        xy[gm_b, 1] = r_g[gm_b] * np.sin(phi_g[gm_b])
        xy[gm_u] = 0.0
        # ico funnel opens up-left from the ico tip
        phi_i = 0.42 * np.pi + 0.28 * u
        xy[ico_b, 0] = sep - r_i[ico_b] * np.cos(phi_i[ico_b])
        xy[ico_b, 1] = r_i[ico_b] * np.sin(phi_i[ico_b])
        xy[ico_u] = (sep, 0.0)
        rest = ~(gm_b | ico_b)
        # liquid / other minima: committor between the tips, energy up
        xy[rest, 0] = sep * np.clip(q[rest], 0.05, 0.95)
        xy[rest, 1] = 0.55 + 0.85 * r_g[rest]
        xy[rest, 0] += 0.18 * u[rest] * r_g[rest]
        return xy

    candidates = []
    best = None
    for alpha, sep in ((0.60, 3.0), (0.70, 3.4)):
        xy0 = funnel_xy(alpha, sep)
        # weak springs: keep kNN neighbours together without collapsing tips
        xy = spring_embed(
            adj,
            e_u,
            xy0,
            gm_u,
            rest_med,
            k_spring=0.35,
            k_rad=0.0,
            alpha=alpha,
            n_iter=80,
        )
        xy[gm_u] = 0.0
        # keep ico on +x, same separation
        xy[ico_u, 0] = float(xy0[ico_u, 0])
        xy[ico_u, 1] = 0.0
        rec = score_layout(
            f"dual-funnel k=12 a={alpha} L={sep}",
            xy,
            e_u,
            gm_u,
            ico_u,
            labels_funnel,
            attract,
        )
        rec["knn"] = knn
        rec["alpha"] = alpha
        rec["kind"] = "spring"
        rec["seven_fail"] = False
        rec["sep_tips"] = sep
        candidates.append((rec, xy, attract, knn, alpha))
        print(
            f"  dual     score={rec['score']:.3f} sil={rec['silhouette']:.3f} "
            f"ang={rec['ang_split']:.3f} r_ico={rec['r_ico']:.3f} "
            f"center={rec['gm_center']} ico_off={rec['ico_off']}"
        )
        if best is None or rec["score"] > best[0]["score"]:
            best = (rec, xy, attract, knn, alpha)

    print("picked", best[0]["name"], "score", best[0]["score"])

    rec, xy_u, attract, knn, alpha = best
    xy_all = expand_xy(xy_u, reps, assign)
    # copies sit on their representative; GM/ico indices stay 0 and 40
    xy_all = place_gm_ico(xy_all, GM_IDX, ICO_IDX)

    print("save coordinates")
    np.savetxt(DEST / "sheap.xy", xy_all, fmt="%.8e")
    np.savetxt(DEST / "sheap_unique.xy", xy_u, fmt="%.8e")
    np.savetxt(DEST / "reps.txt", reps, fmt="%d")
    labels_all = structure_labels(dist, GM_IDX, ICO_IDX)
    np.savetxt(DEST / "struct_label.txt", labels_all, fmt="%d")
    print("render figure")

    title = rf"SHEAP  kNN$={knn}$  $r\sim(\Delta E)^{{{alpha}}}$"
    mins = save_figure(
        xy_all,
        energy,
        GM_IDX,
        ICO_IDX,
        DEST / "elja_occ_lj38_sheap.png",
        title,
        also=OUT / "elja_occ_lj38_sheap.png",
    )

    # comparison strip of the top 2
    ranked = sorted(candidates, key=lambda c: c[0]["score"], reverse=True)[:2]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), facecolor="white")
    for ax, (r, xy, _a, _k, _al) in zip(axes.ravel(), ranked):
        gx, gy, zg = imq_energy(xy, e_u, ngrid=110)
        paint_energy(ax, gx, gy, zg)
        mark(ax, xy, gm_u, ico_u)
        ax.set_title(
            f"{r['name']}\nsc={r['score']:.2f} sil={r['silhouette']:.2f} "
            f"attr={r['n_attract']}",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(DEST / "cmp.png", dpi=140, facecolor="white")
    plt.close(fig)
    print("wrote", DEST / "cmp.png")

    payload = {
        "n": n,
        "n_unique": int(len(reps)),
        "dup_eps": DUP_EPS,
        "descriptor": "sorted-pair-distance L2",
        "energy_role": "r = (E - E_home)^alpha from the GM tip and the ico tip",
        "occupancy_invert": False,
        "gm_at_funnel_tip": True,
        "ico_second_funnel": True,
        "gm_idx": GM_IDX,
        "ico_idx": ICO_IDX,
        "D_gm_ico": float(dist[GM_IDX, ICO_IDX]),
        "winner": rec,
        "n_energy_wells": len(mins),
        "candidates": [c[0] for c in ranked],
        "seconds": time.time() - t_all,
    }
    (DEST / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["winner"], indent=2))
    print(
        "GM at origin",
        rec["gm_center"],
        "ico second funnel",
        rec["ico_off"] and rec["silhouette"] > 0.4,
        "n_attract_on_graph",
        rec["n_attract"],
        "dt",
        f"{payload['seconds']:.1f}s",
    )


if __name__ == "__main__":
    main()
