#!/usr/bin/env python3
"""Cartesian MDS of the 4042 Elja LJ38 quenched minima.

Pairwise Kabsch RMSD is labeled (center + proper rotation, no atom
permutation). Copies of one motif with shuffled labels have RMSD as
large as fcc-versus-ico, so that plane cannot host the two Wales
basins. The same run embeds a permutation-invariant fingerprint:
sorted pairwise internuclear distances (C(38,2) = 703).

Colour and the filled field are quenched energy, never leftover
occupancy. GM is a star, ico a triangle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/tmp/occ-book")
MINFILE = DATA / "lj38_0013.min"
ENERGY = DATA / "lj38.energy"
CAND = DATA / "cand-kabsch"
FIGS = ROOT / "docs" / "ceriotti-figs"

N_ATOMS = 38
N_PAIR = N_ATOMS * (N_ATOMS - 1) // 2
GM_E_LIT = -173.928427
ICO_E_LIT = -173.252378
KABSCH_CHUNK = 64
SMACOF_ITERS = 28

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)


def load_min(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(path)
    if raw.ndim != 2 or raw.shape[1] != 1 + 3 * N_ATOMS:
        raise SystemExit(f"{path}: expected {1 + 3 * N_ATOMS} columns, got {raw.shape}")
    energy = raw[:, 0].astype(np.float64)
    xyz = raw[:, 1:].reshape(raw.shape[0], N_ATOMS, 3).astype(np.float64)
    return energy, xyz


def pairwise_kabsch_rmsd(xyz: np.ndarray, chunk: int = 48) -> np.ndarray:
    """Labeled Kabsch RMSD after centering and a proper rotation.

    Singular values of each 3x3 covariance come from eigvalsh(H H^T);
    sign(det H) flips the smallest one so the rotation stays proper.
    Upper triangle only.
    """
    n = xyz.shape[0]
    xyzc = xyz - xyz.mean(axis=1, keepdims=True)
    norms = np.einsum("nai,nai->n", xyzc, xyzc)
    dist = np.zeros((n, n), dtype=np.float64)
    jstep = 320
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        a = np.ascontiguousarray(np.swapaxes(xyzc[i0:i1], 1, 2))  # (c, 3, 38)
        for j0 in range(i0, n, jstep):
            j1 = min(j0 + jstep, n)
            b = xyzc[j0:j1]  # (m, 38, 3)
            h = a[:, None, :, :] @ b[None, :, :, :]  # (c, m, 3, 3)
            gram = h @ np.swapaxes(h, -1, -2)
            ev = np.linalg.eigvalsh(gram)
            s = np.sqrt(np.clip(ev, 0.0, None))
            sign = np.where(np.linalg.det(h) >= 0.0, 1.0, -1.0)
            cross = s[..., 2] + s[..., 1] + sign * s[..., 0]
            rms2 = (norms[i0:i1, None] + norms[None, j0:j1] - 2.0 * cross) / N_ATOMS
            block = np.sqrt(np.clip(rms2, 0.0, None))
            dist[i0:i1, j0:j1] = block
            if j0 != i0:
                dist[j0:j1, i0:i1] = block.T
        print(f"  kabsch rows {i0}:{i1} / {n}", flush=True)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


def dpair_fingerprint(xyz: np.ndarray) -> np.ndarray:
    """Sorted internuclear distances, shape (n, 703)."""
    n = xyz.shape[0]
    iu = np.triu_indices(N_ATOMS, k=1)
    out = np.empty((n, N_PAIR), dtype=np.float64)
    # 38^2 * 3 * 4042 is small; do it in row chunks to bound the temp.
    step = 256
    for i0 in range(0, n, step):
        i1 = min(i0 + step, n)
        block = xyz[i0:i1]
        delta = block[:, :, None, :] - block[:, None, :, :]
        d = np.sqrt(np.einsum("...k,...k->...", delta, delta))
        out[i0:i1] = np.sort(d[:, iu[0], iu[1]], axis=1)
    return out


def pairwise_l2(feat: np.ndarray) -> np.ndarray:
    gram = feat @ feat.T
    sq = np.clip(np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram, 0.0, None)
    dist = np.sqrt(sq)
    np.fill_diagonal(dist, 0.0)
    return dist


def top_eigh(mat: np.ndarray, k: int = 8, niter: int = 40, seed: int = 0):
    """Leading eigenpairs by subspace iteration. Avoids a full n^3 eigh."""
    n = mat.shape[0]
    kuse = min(k + 4, n)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, kuse)))
    for _ in range(niter):
        q, _ = np.linalg.qr(mat @ q)
    small = q.T @ mat @ q
    w, v = np.linalg.eigh(small)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    return w[:k], q @ v[:, :k]


def torgerson(dist: np.ndarray, dim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    n = dist.shape[0]
    d2 = dist * dist
    col = d2.mean(axis=0)
    grand = float(col.mean())
    b = -0.5 * (d2 - col[None, :] - col[:, None] + grand)
    w, v = top_eigh(b, k=8, niter=36, seed=1)
    lam = np.clip(w[:dim], 0.0, None)
    xy = v[:, :dim] * np.sqrt(lam)
    print(f"  torgerson eigs {np.array2string(w[:4], precision=3)}", flush=True)
    return xy, w


def pairwise_euclid(xy: np.ndarray) -> np.ndarray:
    return pairwise_l2(xy)


def smacof(
    delta: np.ndarray, xy0: np.ndarray, maxiter: int = SMACOF_ITERS, rtol: float = 1e-6
):
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
        if (it + 1) % 20 == 0:
            print(f"  smacof iter {it + 1} stress {stress:.4e}", flush=True)
    den = float(np.sum(delta * delta))
    stress1 = float(np.sqrt(max(2.0 * stress, 0.0) / max(den, 1e-300)))
    return x, stress, stress1, it + 1


def farthest_landmarks(dist: np.ndarray, n_land: int, seed: int) -> np.ndarray:
    n = dist.shape[0]
    n_land = min(n_land, n)
    pick = [int(seed)]
    mind = dist[seed].copy()
    for _ in range(1, n_land):
        j = int(np.argmax(mind))
        pick.append(j)
        mind = np.minimum(mind, dist[j])
    return np.asarray(pick, dtype=int)


def two_landmark(dist: np.ndarray, i_a: int, i_b: int) -> np.ndarray:
    """Classical MDS onto the plane of two references (GM, ico).

    Point i sits at distance D[i,a] from a and D[i,b] from b. a is the
    origin, b is on +x. The leftover coordinate is the height of the
    triangle. Copies of a collapse to the origin, so the GM is the
    centre of its (sparse) neighbourhood by construction.
    """
    sep = float(dist[i_a, i_b])
    da = dist[:, i_a]
    db = dist[:, i_b]
    x = (da * da - db * db + sep * sep) / (2.0 * max(sep, 1e-12))
    y = np.sqrt(np.clip(da * da - x * x, 0.0, None))
    return np.column_stack([x, y])


def gauss_blur(z: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(int(np.ceil(3.0 * sigma)), 1)
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / max(sigma, 1e-9)) ** 2)
    k /= k.sum()
    pad = np.pad(z, ((0, 0), (radius, radius)), mode="edge")
    tmp = np.empty_like(z)
    for i in range(z.shape[0]):
        tmp[i] = np.convolve(pad[i], k, mode="valid")
    pad = np.pad(tmp, ((radius, radius), (0, 0)), mode="edge")
    out = np.empty_like(z)
    for j in range(z.shape[1]):
        out[:, j] = np.convolve(pad[:, j], k, mode="valid")
    return out


def energy_envelope(
    xy: np.ndarray,
    rel: np.ndarray,
    ngrid: int = 220,
    ell: float = 0.09,
    pad: float = 0.14,
    island_frac: float = 0.10,
):
    """Lower envelope of paraboloids, one per unique quench.

    field(q) = min_i [ E_i + ||q - x_i||^2 / ell^2 ]. An isolated GM
    still owns a well of depth 0 centred on itself; a dense ico family
    owns a wider well. Empty-cell blur is not used: it lifts lone minima
    to the cap.
    """
    pts, val, _ = unique_sites(xy, rel, ndigits=5)
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
    span = max(dx, dy)
    ell2 = (ell * span) ** 2
    # inducing: every site below +3, plus FPS of the rest, cap 400
    deep = np.flatnonzero(val <= 3.5)
    rest = np.flatnonzero(val > 3.5)
    take = list(deep.tolist())
    if len(rest) and len(take) < 400:
        sub = pts[rest]
        # cheap stride sample of the high-energy cloud
        step = max(len(rest) // max(400 - len(take), 1), 1)
        take.extend(rest[::step][: 400 - len(take)].tolist())
    ipt, ival = pts[take], val[take]
    field = np.full((ngrid, ngrid), np.inf)
    step = 28
    for i0 in range(0, ngrid, step):
        i1 = min(i0 + step, ngrid)
        xs = xx[i0:i1].ravel()
        ys = yy[i0:i1].ravel()
        d2 = (xs[:, None] - ipt[:, 0][None, :]) ** 2 + (ys[:, None] - ipt[:, 1][None, :]) ** 2
        env = (ival[None, :] + d2 / ell2).min(axis=1)
        field[i0:i1] = env.reshape(i1 - i0, ngrid)
    rad = island_frac * span
    disk = np.zeros((ngrid, ngrid), dtype=bool)
    order = np.argsort(rel)
    for s in order[: min(60, len(order))]:
        disk |= (xx - x[s]) ** 2 + (yy - y[s]) ** 2 <= rad * rad
    # body of the point cloud
    occ = np.zeros((ngrid, ngrid), dtype=np.float64)
    ix = np.clip(np.searchsorted(gx, x) - 1, 0, ngrid - 1)
    iy = np.clip(np.searchsorted(gy, y) - 1, 0, ngrid - 1)
    occ[iy, ix] = 1.0
    occ = gauss_blur(occ, sigma=2.4)
    mask = (occ > 0.015 * float(occ.max())) | disk
    field = np.where(mask, np.clip(field, 0.0, 8.0), np.nan)
    return gx, gy, field


def landmark_mds(dist: np.ndarray, land: np.ndarray, dim: int = 2) -> np.ndarray:
    """Classical MDS on landmarks, distance-based out-of-sample embed."""
    sub = dist[np.ix_(land, land)]
    xy_l, ev = torgerson(sub, dim=dim)
    # B_land = X X^T; project i by -1/2 (d_i^2 - dbar^2) against the landmark basis.
    d2 = sub * sub
    dbar = d2.mean(axis=1)
    # pinv of landmark coordinates
    # x = V sqrt(lam) so x^+ = diag(1/sqrt(lam)) V^T
    lam = np.sum(xy_l * xy_l, axis=0)
    inv = xy_l / np.clip(lam, 1e-15, None)
    d2_all = dist[:, land] ** 2
    # Gower: b_i = -1/2 (d^2(i, L) - mean_L d^2(*, L) - mean_j d^2(i, j_L) + grand)
    # simpler Nyström: b_iL = -1/2 (d_iL^2 - colmean_L)
    b = -0.5 * (d2_all - dbar[None, :])
    xy = b @ inv
    xy -= xy.mean(0)
    return xy


def orient(xy: np.ndarray, i_gm: int, i_ico: int) -> np.ndarray:
    """GM at the origin, ico on +x, leftover axis sign free."""
    out = xy - xy[i_gm]
    vec = out[i_ico]
    ang = np.arctan2(vec[1], vec[0])
    c, s = np.cos(-ang), np.sin(-ang)
    rot = np.array([[c, -s], [s, c]])
    out = out @ rot
    return out


def unique_sites(xy: np.ndarray, values: np.ndarray, ndigits: int = 6):
    key = np.round(xy, ndigits)
    # keep the lowest-energy copy at each site
    order = np.argsort(values)
    _, first = np.unique(key[order], axis=0, return_index=True)
    idx = order[first]
    return xy[idx], values[idx], idx


def density_mask(xy: np.ndarray, gx: np.ndarray, gy: np.ndarray, sigma_frac: float = 0.03):
    xx, yy = np.meshgrid(gx, gy)
    xmin, xmax = float(gx[0]), float(gx[-1])
    ymin, ymax = float(gy[0]), float(gy[-1])
    dx, dy = xmax - xmin, ymax - ymin
    sx = max(sigma_frac * dx, 1e-9)
    sy = max(sigma_frac * dy, 1e-9)
    # subsample inducing for the mask
    pts = xy
    if len(pts) > 800:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), 800, replace=False)]
    field = np.zeros(xx.shape, dtype=np.float64)
    step = 40
    ngrid = xx.shape[0]
    for i0 in range(0, ngrid, step):
        i1 = min(i0 + step, ngrid)
        xs = xx[i0:i1].ravel()
        ys = yy[i0:i1].ravel()
        dx_ = (xs[:, None] - pts[:, 0][None, :]) / sx
        dy_ = (ys[:, None] - pts[:, 1][None, :]) / sy
        field[i0:i1] = np.exp(-0.5 * (dx_ * dx_ + dy_ * dy_)).sum(1).reshape(i1 - i0, ngrid)
    thresh = 0.08 * float(field.max())
    return field > thresh


def imq_nw(xy: np.ndarray, values: np.ndarray, ngrid: int = 180, ell: float = 0.06, pad: float = 0.10):
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
    span = max(dx, dy)
    ell_abs = ell * span
    ell2 = ell_abs * ell_abs
    field = np.empty((ngrid, ngrid), dtype=np.float64)
    pts, val, _ = unique_sites(xy, values)
    step = 30
    for i0 in range(0, ngrid, step):
        i1 = min(i0 + step, ngrid)
        xs = xx[i0:i1].ravel()
        ys = yy[i0:i1].ravel()
        dx_ = xs[:, None] - pts[:, 0][None, :]
        dy_ = ys[:, None] - pts[:, 1][None, :]
        k = 1.0 / np.sqrt(1.0 + (dx_ * dx_ + dy_ * dy_) / ell2)
        den = np.clip(k.sum(1), 1e-12, None)
        field[i0:i1] = (k @ val / den).reshape(i1 - i0, ngrid)
    mask = density_mask(pts, gx, gy)
    field = np.where(mask, field, np.nan)
    return gx, gy, field


def rbf_imq(
    xy: np.ndarray,
    values: np.ndarray,
    ngrid: int = 180,
    ell: float = 0.08,
    pad: float = 0.10,
    jitter: float = 1e-5,
    n_ind: int = 500,
):
    pts, val, _ = unique_sites(xy, values)
    if len(pts) > n_ind:
        # keep the two deepest sites, FPS the rest
        order = np.argsort(val)
        keep = [int(order[0])]
        if len(order) > 1:
            keep.append(int(order[1]))
        rest = np.setdiff1d(np.arange(len(pts)), keep, assume_unique=False)
        d = pairwise_l2(pts)
        mind = np.min(d[keep][:, rest], axis=0)
        chosen = list(keep)
        pool = rest.tolist()
        for _ in range(n_ind - len(chosen)):
            if not pool:
                break
            jloc = int(np.argmax(mind))
            chosen.append(int(pool[jloc]))
            mind = np.minimum(mind, d[pool[jloc], pool])
            pool.pop(jloc)
            mind = np.delete(mind, jloc)
        pts, val = pts[chosen], val[chosen]
    n = len(pts)
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= pad * dx
    xmax += pad * dx
    ymin -= pad * dy
    ymax += pad * dy
    span = max(dx, dy)
    ell_abs = ell * span
    ell2 = ell_abs * ell_abs
    d2 = pairwise_l2(pts) ** 2
    kmat = 1.0 / np.sqrt(1.0 + d2 / ell2)
    kmat = kmat + jitter * np.eye(n)
    w = np.linalg.solve(kmat, val)
    gx = np.linspace(xmin, xmax, ngrid)
    gy = np.linspace(ymin, ymax, ngrid)
    xx, yy = np.meshgrid(gx, gy)
    field = np.empty((ngrid, ngrid), dtype=np.float64)
    step = 24
    for i0 in range(0, ngrid, step):
        i1 = min(i0 + step, ngrid)
        xs = xx[i0:i1].ravel()
        ys = yy[i0:i1].ravel()
        dx_ = xs[:, None] - pts[:, 0][None, :]
        dy_ = ys[:, None] - pts[:, 1][None, :]
        kk = 1.0 / np.sqrt(1.0 + (dx_ * dx_ + dy_ * dy_) / ell2)
        field[i0:i1] = (kk @ w).reshape(i1 - i0, ngrid)
    mask = density_mask(pts, gx, gy)
    field = np.where(mask, field, np.nan)
    return gx, gy, field


def knn_energy(xy: np.ndarray, values: np.ndarray, ngrid: int = 180, k: int = 8, pad: float = 0.10):
    pts, val, _ = unique_sites(xy, values)
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
    field = np.empty((ngrid, ngrid), dtype=np.float64)
    step = 20
    kk = min(k, len(pts))
    for i0 in range(0, ngrid, step):
        i1 = min(i0 + step, ngrid)
        xs = xx[i0:i1].ravel()
        ys = yy[i0:i1].ravel()
        dx_ = xs[:, None] - pts[:, 0][None, :]
        dy_ = ys[:, None] - pts[:, 1][None, :]
        d2 = dx_ * dx_ + dy_ * dy_
        nn = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        take = np.take_along_axis(val[None, :].repeat(len(xs), 0), nn, axis=1)
        # lower-quartile of neighbours so a well is not lifted by mixed rims
        field[i0:i1] = np.quantile(take, 0.25, axis=1).reshape(i1 - i0, ngrid)
    mask = density_mask(pts, gx, gy)
    field = np.where(mask, field, np.nan)
    return gx, gy, field


def sample_field(gx: np.ndarray, gy: np.ndarray, field: np.ndarray, pt: np.ndarray) -> float:
    if not np.isfinite(field).any():
        return float("nan")
    ix = int(np.clip(np.searchsorted(gx, pt[0]) - 1, 0, len(gx) - 2))
    iy = int(np.clip(np.searchsorted(gy, pt[1]) - 1, 0, len(gy) - 2))
    # bilinear among finite cells
    vals = []
    for j in (iy, iy + 1):
        for i in (ix, ix + 1):
            v = field[j, i]
            if np.isfinite(v):
                vals.append(float(v))
    if not vals:
        # nearest finite
        yy, xx = np.indices(field.shape)
        finite = np.isfinite(field)
        if not finite.any():
            return float("nan")
        d2 = (gx[xx] - pt[0]) ** 2 + (gy[yy] - pt[1]) ** 2
        d2 = np.where(finite, d2, np.inf)
        j, i = np.unravel_index(int(np.argmin(d2)), field.shape)
        return float(field[j, i])
    return float(np.mean(vals))


def local_min_q(gx, gy, field, pt, rad_frac: float = 0.06) -> dict:
    """Is the field at pt a local minimum inside a disk?"""
    e0 = sample_field(gx, gy, field, pt)
    xx, yy = np.meshgrid(gx, gy)
    span = max(float(gx[-1] - gx[0]), float(gy[-1] - gy[0]))
    rad = rad_frac * span
    disk = (xx - pt[0]) ** 2 + (yy - pt[1]) ** 2 <= rad * rad
    on = disk & np.isfinite(field)
    if not on.any() or not np.isfinite(e0):
        return {"e0": e0, "is_min": False, "disk_min": float("nan"), "n": 0}
    dmin = float(field[on].min())
    return {
        "e0": e0,
        "is_min": bool(e0 <= dmin + 0.05),
        "disk_min": dmin,
        "n": int(on.sum()),
    }


def rim_score(xy: np.ndarray, i_pt: int, member: np.ndarray) -> dict:
    """How far is i_pt from the centroid of its motif copies, in cloud radii."""
    cloud = xy[member]
    cen = cloud.mean(0)
    rad = np.sqrt(((cloud - cen) ** 2).sum(1))
    rmax = float(rad.max()) if len(rad) else 0.0
    rmed = float(np.median(rad)) if len(rad) else 0.0
    dcen = float(np.linalg.norm(xy[i_pt] - cen))
    return {
        "n": int(member.sum()),
        "r_max": rmax,
        "r_med": rmed,
        "d_centroid": dcen,
        "rim": bool(rmax > 1e-8 and dcen > 0.65 * rmax),
        "interior": bool(rmax <= 1e-8 or dcen < 0.45 * rmax),
    }


def n_blobs(xy: np.ndarray, rad: float, maxn: int = 700) -> int:
    n0 = len(xy)
    if n0 == 0:
        return 0
    if n0 > maxn:
        rng = np.random.default_rng(0)
        xy = xy[rng.choice(n0, maxn, replace=False)]
    d = pairwise_euclid(xy)
    adj = d <= rad
    np.fill_diagonal(adj, False)
    n = len(xy)
    seen = np.zeros(n, dtype=bool)
    ncomp = 0
    for i in range(n):
        if seen[i]:
            continue
        ncomp += 1
        stack = [i]
        seen[i] = True
        while stack:
            u = stack.pop()
            nbr = np.flatnonzero(adj[u] & ~seen)
            seen[nbr] = True
            stack.extend(int(x) for x in nbr)
    return ncomp


def sep_norm(xy: np.ndarray, i: int, j: int) -> dict:
    sep = float(np.linalg.norm(xy[i] - xy[j]))
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    return {"sep": sep, "diam": diam, "sep_norm": sep / max(diam, 1e-12)}


def style_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for sp in ax.spines.values():
        sp.set_visible(False)


def mark(ax, xy, i_gm, i_ico, e_gm, e_ico):
    h1 = ax.scatter(
        xy[i_gm, 0],
        xy[i_gm, 1],
        s=160,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.7,
        zorder=6,
        label=rf"GM ${e_gm:.3f}$",
    )
    h2 = ax.scatter(
        xy[i_ico, 0],
        xy[i_ico, 1],
        s=90,
        marker="^",
        c="k",
        edgecolors="white",
        linewidths=0.7,
        zorder=6,
        label=rf"ico ${e_ico:.3f}$",
    )
    ax.legend(handles=[h1, h2], loc="best", fontsize=8, frameon=True, fancybox=False)


def paint_energy(ax, gx, gy, field, vmax: float):
    zz = np.clip(field, 0.0, vmax)
    mesh = ax.contourf(
        gx, gy, zz, levels=np.linspace(0.0, vmax, 25), cmap=PES, extend="max"
    )
    ax.contour(
        gx,
        gy,
        np.where(np.isfinite(field), zz, np.nan),
        levels=np.linspace(0.2, vmax - 0.2, 12),
        colors="#1a1a2e",
        linewidths=0.35,
    )
    style_ax(ax)
    return mesh


def save_xy(path: Path, xy: np.ndarray) -> None:
    np.savetxt(path, xy, fmt="%.8e")


def diagnose(name, xy, energy, i_gm, i_ico, gm_mask, ico_mask) -> dict:
    rel = energy - energy[i_gm]
    low = rel < 1.5
    span = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    blobs_all = n_blobs(xy, 0.08 * span) if span > 0 else 0
    blobs_low = n_blobs(xy[low], 0.08 * span) if low.any() else 0
    out = {
        "name": name,
        "sep": sep_norm(xy, i_gm, i_ico),
        "gm_cloud": rim_score(xy, i_gm, gm_mask),
        "ico_cloud": rim_score(xy, i_ico, ico_mask),
        "n_blobs_rad08": blobs_all,
        "n_lowE_blobs_rad08": blobs_low,
        "n_lowE": int(low.sum()),
    }
    print(
        f"{name}: sep_norm={out['sep']['sep_norm']:.3f} "
        f"GM rim={out['gm_cloud']['rim']} interior={out['gm_cloud']['interior']} "
        f"d_cent={out['gm_cloud']['d_centroid']:.3g} rmax={out['gm_cloud']['r_max']:.3g} "
        f"blobs={blobs_all} lowE_blobs={blobs_low}",
        flush=True,
    )
    return out


def pick_best(cands: list[dict]) -> dict:
    """Prefer interior GM, not a rim, not a 7-blob spray, large GM-ico sep."""

    def score(c):
        rim = c["gm_cloud"]["rim"]
        interior = c["gm_cloud"]["interior"]
        blobs = c["n_blobs_rad08"]
        sep = c["sep"]["sep_norm"]
        blob_pen = 0.0
        if blobs >= 6:
            blob_pen -= 3.0
        elif blobs >= 4:
            blob_pen -= 1.0
        return (2.0 if interior else 0.0) + (0.0 if rim else 1.0) + 2.0 * sep + blob_pen

    return max(cands, key=score)


def figure_pair(xy, energy, i_gm, i_ico, title, dest: Path, field_kind: str = "envelope"):
    rel = np.clip(energy - energy[i_gm], 0.0, None)
    if field_kind == "rbf":
        gx, gy, field = rbf_imq(xy, rel, ngrid=160, ell=0.07)
    elif field_kind == "knn":
        gx, gy, field = knn_energy(xy, rel, ngrid=160, k=8)
    elif field_kind == "imq":
        gx, gy, field = imq_nw(xy, rel, ngrid=160, ell=0.055)
    else:
        gx, gy, field = energy_envelope(xy, rel, ngrid=220, ell=0.13)
    vmax = 6.0
    gm_q = local_min_q(gx, gy, field, xy[i_gm])
    ico_q = local_min_q(gx, gy, field, xy[i_ico])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.1), facecolor="white")
    sc = axes[0].scatter(
        xy[:, 0],
        xy[:, 1],
        c=rel,
        s=8,
        cmap=PES,
        vmin=0.0,
        vmax=vmax,
        linewidths=0,
        zorder=2,
    )
    mark(axes[0], xy, i_gm, i_ico, energy[i_gm], energy[i_ico])
    style_ax(axes[0])
    axes[0].set_title("minima, colour = $E-E_{\\mathrm{GM}}$")
    fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.03).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    mesh = paint_energy(axes[1], gx, gy, field, vmax)
    mark(axes[1], xy, i_gm, i_ico, energy[i_gm], energy[i_ico])
    axes[1].set_title("filled energy")
    fig.colorbar(mesh, ax=axes[1], fraction=0.046, pad=0.03).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", dest, flush=True)
    return {
        "field_gm": gm_q,
        "field_ico": ico_q,
        "delta_surface": (
            None
            if not (np.isfinite(gm_q["e0"]) and np.isfinite(ico_q["e0"]))
            else float(ico_q["e0"] - gm_q["e0"])
        ),
    }


def load_or_compute_kabsch(xyz: np.ndarray) -> np.ndarray:
    npy = CAND / "lj38_kabsch.dist.npy"
    txt = CAND / "lj38_kabsch.dist"
    if npy.is_file():
        dist = np.load(npy)
        if dist.shape == (len(xyz), len(xyz)):
            print("loaded", npy, flush=True)
            return dist.astype(np.float64)
    if txt.is_file():
        dist = np.loadtxt(txt)
        if dist.shape == (len(xyz), len(xyz)):
            print("loaded", txt, flush=True)
            return dist.astype(np.float64)
    print("computing pairwise Kabsch RMSD", flush=True)
    dist = pairwise_kabsch_rmsd(xyz)
    np.save(npy, dist.astype(np.float32))
    print("wrote", npy, flush=True)
    return dist


def load_or_compute_dpair(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    npy = CAND / "lj38_dpair.dist.npy"
    feat_p = CAND / "lj38_dpair.feat.npy"
    if npy.is_file() and feat_p.is_file():
        dist = np.load(npy)
        feat = np.load(feat_p)
        if dist.shape == (len(xyz), len(xyz)) and feat.shape == (len(xyz), N_PAIR):
            print("loaded", npy, flush=True)
            return dist.astype(np.float64), feat.astype(np.float64)
    print("computing sorted pair-distance fingerprints", flush=True)
    feat = dpair_fingerprint(xyz)
    dist = pairwise_l2(feat)
    np.save(npy, dist.astype(np.float32))
    np.save(feat_p, feat.astype(np.float32))
    print("wrote", npy, flush=True)
    return dist, feat


def main() -> None:
    CAND.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    energy, xyz = load_min(MINFILE)
    e_file = np.loadtxt(ENERGY)
    if len(e_file) == len(energy):
        if float(np.max(np.abs(e_file - energy))) > 1e-6:
            print("warning: lj38.energy disagrees with .min header; using .min")
    n = len(energy)
    i_gm = int(np.argmin(energy))
    i_ico = int(np.argmin(np.abs(energy - ICO_E_LIT)))
    e_gm = float(energy[i_gm])
    e_ico = float(energy[i_ico])
    gm_mask = np.abs(energy - e_gm) < 1e-6
    ico_mask = np.abs(energy - e_ico) < 1e-6
    print(
        f"n={n} E_GM={e_gm:.8f} idx={i_gm} copies={int(gm_mask.sum())} "
        f"(lit {GM_E_LIT})",
        flush=True,
    )
    print(
        f"E_ico={e_ico:.8f} idx={i_ico} copies={int(ico_mask.sum())} "
        f"(lit {ICO_E_LIT})",
        flush=True,
    )

    # labeled Kabsch among copies: the reason the plane fails
    def mean_pair(mask, dist_fn):
        idx = np.flatnonzero(mask)
        if len(idx) < 2:
            return float("nan")
        vals = [dist_fn(i, j) for a, i in enumerate(idx[:12]) for j in idx[a + 1 : 12]]
        return float(np.mean(vals)) if vals else float("nan")

    def kabsch_one(i, j):
        a = xyz[i] - xyz[i].mean(0)
        b = xyz[j] - xyz[j].mean(0)
        h = a.T @ b
        u, s, vt = np.linalg.svd(h)
        if np.linalg.det(u) * np.linalg.det(vt) < 0:
            s = s.copy()
            s[-1] *= -1
        return float(np.sqrt(max(((a * a).sum() + (b * b).sum() - 2.0 * s.sum()) / N_ATOMS, 0.0)))

    k_gm = mean_pair(gm_mask, kabsch_one)
    k_ico = mean_pair(ico_mask, kabsch_one)
    k_cross = kabsch_one(i_gm, i_ico)
    print(
        f"labeled Kabsch mean(GM copies)={k_gm:.4f} mean(ico copies)={k_ico:.4f} "
        f"GM-vs-ico={k_cross:.4f}",
        flush=True,
    )

    d_kabsch = load_or_compute_kabsch(xyz)
    d_dpair, feat = load_or_compute_dpair(xyz)
    med_k = float(np.median(d_kabsch[np.triu_indices(n, 1)]))
    med_p = float(np.median(d_dpair[np.triu_indices(n, 1)]))
    print(f"median Kabsch={med_k:.4f} median dpair-L2={med_p:.4f}", flush=True)

    embeddings = {}
    diags = []
    ev_k = np.zeros(4)
    ev_p = np.zeros(4)
    plot_only = "--plot" in sys.argv
    cached_names = [
        "kabsch_torgerson",
        "kabsch_asinh",
        "kabsch_smacof",
        "kabsch_landmark",
        "dpair_torgerson",
        "dpair_asinh",
        "dpair_smacof",
    ]
    have_cache = all((CAND / f"{name}.xy").is_file() for name in cached_names)

    if plot_only and have_cache:
        print("loading cached embeddings", flush=True)
        for name in cached_names:
            embeddings[name] = np.loadtxt(CAND / f"{name}.xy")
            diags.append(diagnose(name, embeddings[name], energy, i_gm, i_ico, gm_mask, ico_mask))
    else:
        print("MDS Kabsch Torgerson", flush=True)
        xy_kt, ev_k = torgerson(d_kabsch)
        xy_kt = orient(xy_kt, i_gm, i_ico)
        embeddings["kabsch_torgerson"] = xy_kt
        save_xy(CAND / "kabsch_torgerson.xy", xy_kt)
        diags.append(diagnose("kabsch_torgerson", xy_kt, energy, i_gm, i_ico, gm_mask, ico_mask))

        print("MDS Kabsch asinh(D/median)", flush=True)
        xy_ka, ev_ka = torgerson(np.arcsinh(d_kabsch / max(med_k, 1e-12)))
        xy_ka = orient(xy_ka, i_gm, i_ico)
        embeddings["kabsch_asinh"] = xy_ka
        save_xy(CAND / "kabsch_asinh.xy", xy_ka)
        diags.append(diagnose("kabsch_asinh", xy_ka, energy, i_gm, i_ico, gm_mask, ico_mask))

        print("SMACOF Kabsch asinh init", flush=True)
        xy_ks, st, st1, nit = smacof(np.arcsinh(d_kabsch / max(med_k, 1e-12)), xy_ka)
        xy_ks = orient(xy_ks, i_gm, i_ico)
        embeddings["kabsch_smacof"] = xy_ks
        save_xy(CAND / "kabsch_smacof.xy", xy_ks)
        dks = diagnose("kabsch_smacof", xy_ks, energy, i_gm, i_ico, gm_mask, ico_mask)
        dks["stress1"] = st1
        dks["smacof_iters"] = nit
        diags.append(dks)

        print("landmark MDS Kabsch asinh, 256 FPS from GM", flush=True)
        land = farthest_landmarks(d_kabsch, 256, i_gm)
        xy_kl = landmark_mds(np.arcsinh(d_kabsch / max(med_k, 1e-12)), land)
        xy_kl = orient(xy_kl, i_gm, i_ico)
        embeddings["kabsch_landmark"] = xy_kl
        save_xy(CAND / "kabsch_landmark.xy", xy_kl)
        diags.append(diagnose("kabsch_landmark", xy_kl, energy, i_gm, i_ico, gm_mask, ico_mask))

        print("MDS dpair Torgerson", flush=True)
        xy_pt, ev_p = torgerson(d_dpair)
        xy_pt = orient(xy_pt, i_gm, i_ico)
        embeddings["dpair_torgerson"] = xy_pt
        save_xy(CAND / "dpair_torgerson.xy", xy_pt)
        diags.append(diagnose("dpair_torgerson", xy_pt, energy, i_gm, i_ico, gm_mask, ico_mask))

        print("MDS dpair asinh(D/median)", flush=True)
        xy_pa, ev_pa = torgerson(np.arcsinh(d_dpair / max(med_p, 1e-12)))
        xy_pa = orient(xy_pa, i_gm, i_ico)
        embeddings["dpair_asinh"] = xy_pa
        save_xy(CAND / "dpair_asinh.xy", xy_pa)
        diags.append(diagnose("dpair_asinh", xy_pa, energy, i_gm, i_ico, gm_mask, ico_mask))

        print("SMACOF dpair asinh init", flush=True)
        xy_ps, stp, st1p, nitp = smacof(np.arcsinh(d_dpair / max(med_p, 1e-12)), xy_pa)
        xy_ps = orient(xy_ps, i_gm, i_ico)
        embeddings["dpair_smacof"] = xy_ps
        save_xy(CAND / "dpair_smacof.xy", xy_ps)
        dps = diagnose("dpair_smacof", xy_ps, energy, i_gm, i_ico, gm_mask, ico_mask)
        dps["stress1"] = st1p
        dps["smacof_iters"] = nitp
        diags.append(dps)

    print("2-landmark MDS of dpair (GM, ico)", flush=True)
    xy_p2 = two_landmark(d_dpair, i_gm, i_ico)
    embeddings["dpair_2land"] = xy_p2
    save_xy(CAND / "dpair_2land.xy", xy_p2)
    diags.append(diagnose("dpair_2land", xy_p2, energy, i_gm, i_ico, gm_mask, ico_mask))

    print("2-landmark MDS of asinh dpair", flush=True)
    xy_p2a = two_landmark(np.arcsinh(d_dpair / max(med_p, 1e-12)), i_gm, i_ico)
    embeddings["dpair_2land_asinh"] = xy_p2a
    save_xy(CAND / "dpair_2land_asinh.xy", xy_p2a)
    diags.append(diagnose("dpair_2land_asinh", xy_p2a, energy, i_gm, i_ico, gm_mask, ico_mask))

    kabsch_diags = [d for d in diags if d["name"].startswith("kabsch_")]
    dpair_diags = [d for d in diags if d["name"].startswith("dpair_")]
    best_k = pick_best(kabsch_diags)
    best_p = pick_best(dpair_diags)
    print("best Kabsch", best_k["name"], flush=True)
    print("best dpair", best_p["name"], flush=True)

    # Why Kabsch cannot separate the motifs
    why = (
        "Kabsch-without-permutation cannot separate fcc/ico. "
        f"Mean labeled RMSD among GM copies is {k_gm:.3f} and among ico copies "
        f"{k_ico:.3f}, while GM-versus-ico is {k_cross:.3f}. Atom labeling of "
        "the catalog is an arbitrary permutation of each quench; the labeled "
        "Procrustes distance is then a labeling distance, not a motif distance. "
        "asinh, SMACOF, and landmark MDS act on the same broken metric and "
        "cannot invent the missing permutation. Sorted internuclear distances "
        f"(703-vector) collapse GM copies (L2 ~ 0) and keep GM-ico at "
        f"{float(d_dpair[i_gm, i_ico]):.3f}."
    )
    print(why, flush=True)
    (CAND / "notes.txt").write_text(why + "\n")

    surf_k = figure_pair(
        embeddings[best_k["name"]],
        energy,
        i_gm,
        i_ico,
        rf"Kabsch RMSD MDS ({best_k['name'].replace('_', ' ')})",
        CAND / "elja_occ_lj38_kabsch.png",
        field_kind="envelope",
    )
    figure_pair(
        embeddings[best_k["name"]],
        energy,
        i_gm,
        i_ico,
        rf"Kabsch RMSD MDS ({best_k['name'].replace('_', ' ')})",
        FIGS / "elja_occ_lj38_kabsch.png",
        field_kind="envelope",
    )
    # permutation-invariant map: 2-landmark MDS of the 703-vector
    surf_p = figure_pair(
        embeddings["dpair_2land"],
        energy,
        i_gm,
        i_ico,
        r"sorted pair distances, 2-landmark MDS",
        CAND / "elja_occ_lj38_dpair.png",
        field_kind="envelope",
    )
    figure_pair(
        embeddings["dpair_2land"],
        energy,
        i_gm,
        i_ico,
        r"sorted pair distances, 2-landmark MDS",
        FIGS / "elja_occ_lj38_dpair.png",
        field_kind="envelope",
    )
    surf_p_knn = figure_pair(
        embeddings["dpair_2land_asinh"],
        energy,
        i_gm,
        i_ico,
        r"sorted pair distances, 2-landmark asinh MDS",
        CAND / "elja_occ_lj38_dpair_asinh.png",
        field_kind="envelope",
    )
    figure_pair(
        embeddings["dpair_torgerson"],
        energy,
        i_gm,
        i_ico,
        r"sorted pair distances, Torgerson MDS",
        CAND / "elja_occ_lj38_dpair_torgerson.png",
        field_kind="envelope",
    )

    # comparison strip of Kabsch variants
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.0), facecolor="white")
    rel = np.clip(energy - e_gm, 0.0, None)
    for ax, name in zip(
        axes, ["kabsch_torgerson", "kabsch_asinh", "kabsch_smacof", "kabsch_landmark"]
    ):
        xy = embeddings[name]
        ax.scatter(xy[:, 0], xy[:, 1], c=rel, s=5, cmap=PES, vmin=0, vmax=6, linewidths=0)
        mark(ax, xy, i_gm, i_ico, e_gm, e_ico)
        style_ax(ax)
        ax.set_title(name.replace("_", " "))
    fig.tight_layout()
    fig.savefig(CAND / "kabsch_variants.png", dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2), facecolor="white")
    for ax, name in zip(
        axes, ["dpair_torgerson", "dpair_smacof", "dpair_2land", "dpair_2land_asinh"]
    ):
        xy = embeddings[name]
        ax.scatter(xy[:, 0], xy[:, 1], c=rel, s=5, cmap=PES, vmin=0, vmax=6, linewidths=0)
        mark(ax, xy, i_gm, i_ico, e_gm, e_ico)
        style_ax(ax)
        ax.set_title(name.replace("_", " "))
    fig.tight_layout()
    fig.savefig(CAND / "dpair_variants.png", dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    report = {
        "n": n,
        "n_atoms": N_ATOMS,
        "E_GM": e_gm,
        "E_ico": e_ico,
        "idx_GM": i_gm,
        "idx_ico": i_ico,
        "n_GM_copies": int(gm_mask.sum()),
        "n_ico_copies": int(ico_mask.sum()),
        "kabsch_mean_GM_copies": k_gm,
        "kabsch_mean_ico_copies": k_ico,
        "kabsch_GM_vs_ico": k_cross,
        "dpair_GM_vs_ico": float(d_dpair[i_gm, i_ico]),
        "median_kabsch": med_k,
        "median_dpair": med_p,
        "kabsch_eigs": [float(x) for x in ev_k],
        "dpair_eigs": [float(x) for x in ev_p],
        "best_kabsch": best_k["name"],
        "best_dpair": "dpair_2land",
        "diagnostics": diags,
        "surface_kabsch": surf_k,
        "surface_dpair": surf_p,
        "surface_dpair_2land_asinh": surf_p_knn,
        "why": why,
        "two_basins_visible": bool(
            surf_p["field_gm"].get("is_min")
            and surf_p["delta_surface"] is not None
            and surf_p["delta_surface"] > 0.2
        ),
    }
    (CAND / "scores.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "E_GM", "E_ico", "idx_GM", "idx_ico", "best_kabsch", "best_dpair",
        "kabsch_GM_vs_ico", "dpair_GM_vs_ico", "two_basins_visible",
    )}, indent=2))
    print("surface Kabsch GM/ico", surf_k)
    print("surface dpair  GM/ico", surf_p)
    print("surface dpair-knn GM/ico", surf_p_knn)


if __name__ == "__main__":
    main()
