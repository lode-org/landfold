#!/usr/bin/env python3
"""SOAP-like Gaussian neighbor-density map of the Elja LJ38 book.

Each inherent structure is a permutation-invariant fingerprint:
radial Gaussian neighbor density on r in [0.8, 2.8] (n_r=16, sigma=0.15)
plus a 3-body neighbor-angle histogram (n_a=8), averaged over the 38
atoms. Classical MDS of L2 (PCA of the feature matrix) and asinh-L2
Torgerson sit in 2D. The filled field is E-E_GM (IMQ / Gaussian NW /
nearest-neighbour), not occupancy invert.

If the mean fingerprint does not split the Wales funnels, angular
resolution is raised and per-atom fingerprints are concatenated after
sorting atoms by coordination.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ceriotti-figs"
MINFILE = Path("/tmp/occ-book/lj38_0013.min")
DEST = Path("/tmp/occ-book/cand-soap")
FIG = OUT / "elja_occ_lj38_soap.png"

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
R_MIN = 0.8
R_MAX = 2.8
N_R = 16
SIG_R = 0.15
N_A = 8
CN_CUT = 1.5
ASINH_NORM = 2.0 * float(np.arcsinh(1.0))
EMAX = 6.0
SEP_BAR = 0.20
N_INDUCING = 360
NGRID = 180


def load_min(path: Path, n_atoms: int = N_ATOMS) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(path)
    expect = 1 + 3 * n_atoms
    if raw.ndim != 2 or raw.shape[1] != expect:
        raise SystemExit(f"{path}: expected (*, {expect}), got {raw.shape}")
    return raw[:, 0].astype(float), raw[:, 1:].reshape(-1, n_atoms, 3)


def pairwise_d(pos: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d


def radial_hist(d: np.ndarray, n_r: int = N_R, sigma: float = SIG_R) -> np.ndarray:
    centers = np.linspace(R_MIN, R_MAX, n_r)
    g = np.exp(-0.5 * ((d[..., None] - centers) / sigma) ** 2)
    g[d > (R_MAX + 4.0 * sigma)] = 0.0
    return g.sum(axis=1)


def angle_hist(pos: np.ndarray, d: np.ndarray, n_a: int) -> np.ndarray:
    n = pos.shape[0]
    if n_a <= 0:
        return np.zeros((n, 0))
    centers = np.linspace(0.0, np.pi, n_a)
    sig = (np.pi / max(n_a - 1, 1)) * 0.65
    out = np.zeros((n, n_a))
    for i in range(n):
        js = np.flatnonzero((d[i] >= R_MIN) & (d[i] <= R_MAX))
        if js.size < 2:
            continue
        vecs = pos[js] - pos[i]
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        c = np.clip(vecs @ vecs.T, -1.0, 1.0)
        ii, jj = np.triu_indices(js.size, k=1)
        ang = np.arccos(c[ii, jj])
        out[i] = np.exp(-0.5 * ((ang[:, None] - centers) / sig) ** 2).sum(0)
    return out


def atom_fps(pos: np.ndarray, n_r: int, n_a: int) -> tuple[np.ndarray, np.ndarray]:
    d = pairwise_d(pos)
    rad = radial_hist(d, n_r=n_r)
    ang = angle_hist(pos, d, n_a=n_a) if n_a > 0 else np.zeros((pos.shape[0], 0))
    cn = np.sum(np.isfinite(d) & (d < CN_CUT), axis=1).astype(float)
    return np.concatenate([rad, ang], axis=1), cn


def mean_fp(pos: np.ndarray, n_r: int, n_a: int) -> np.ndarray:
    fp, _cn = atom_fps(pos, n_r, n_a)
    return fp.mean(axis=0)


def meanstd_fp(pos: np.ndarray, n_r: int, n_a: int) -> np.ndarray:
    fp, _cn = atom_fps(pos, n_r, n_a)
    return np.concatenate([fp.mean(axis=0), fp.std(axis=0)])


def sorted_fp(pos: np.ndarray, n_r: int, n_a: int) -> np.ndarray:
    fp, cn = atom_fps(pos, n_r, n_a)
    order = np.argsort(-cn, kind="mergesort")
    return fp[order].ravel()


def reduce_fp(fp: np.ndarray, cn: np.ndarray, kind: str) -> np.ndarray:
    if kind == "mean":
        return fp.mean(axis=0)
    if kind == "meanstd":
        return np.concatenate([fp.mean(axis=0), fp.std(axis=0)])
    if kind == "sorted":
        return fp[np.argsort(-cn, kind="mergesort")].ravel()
    raise ValueError(kind)


def build_fps(frames: np.ndarray, kind: str, n_r: int, n_a: int) -> np.ndarray:
    rows = []
    for i in range(frames.shape[0]):
        fp, cn = atom_fps(frames[i], n_r, n_a)
        rows.append(reduce_fp(fp, cn, kind))
    return np.vstack(rows)


def _family_chunk(frames: np.ndarray, n_r: int, n_a: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = frames.shape[0]
    mean_rows = np.empty((n, n_r + n_a))
    std_rows = np.empty((n, 2 * (n_r + n_a)))
    sort_rows = np.empty((n, N_ATOMS * (n_r + n_a)))
    for i in range(n):
        fp, cn = atom_fps(frames[i], n_r, n_a)
        mean_rows[i] = fp.mean(axis=0)
        std_rows[i] = np.concatenate([fp.mean(axis=0), fp.std(axis=0)])
        sort_rows[i] = fp[np.argsort(-cn, kind="mergesort")].ravel()
    return mean_rows, std_rows, sort_rows


def chunk_path(n_r: int, n_a: int, i: int) -> Path:
    return DEST / f"fp_chunk_{i:02d}_r{n_r}a{n_a}.npz"


def run_fp_chunk(chunk: int, chunks: int, n_r: int, n_a: int) -> Path:
    DEST.mkdir(parents=True, exist_ok=True)
    energy, frames = load_min(MINFILE)
    n = frames.shape[0]
    lo = n * chunk // chunks
    hi = n * (chunk + 1) // chunks
    print(f"chunk {chunk}/{chunks} frames {lo}:{hi}", flush=True)
    mean, std, sort = _family_chunk(frames[lo:hi], n_r, n_a)
    dest = chunk_path(n_r, n_a, chunk)
    np.savez_compressed(dest, mean=mean, meanstd=std, sorted=sort, lo=lo, hi=hi)
    print("wrote", dest, mean.shape, flush=True)
    return dest


def build_family(frames: np.ndarray, n_r: int, n_a: int, workers: int = 10) -> dict[str, np.ndarray]:
    """Spawn this script as workers so the angle loop is multi-process."""
    n = frames.shape[0]
    workers = max(1, min(workers, n))
    script = Path(__file__).resolve()
    procs = []
    for i in range(workers):
        dest = chunk_path(n_r, n_a, i)
        if dest.is_file() and dest.stat().st_size > 1000:
            continue
        cmd = [
            sys.executable,
            str(script),
            "--fp-chunk",
            str(i),
            "--fp-chunks",
            str(workers),
            "--n-r",
            str(n_r),
            "--n-a",
            str(n_a),
        ]
        procs.append(subprocess.Popen(cmd))
    rc = 0
    for p in procs:
        rc = p.wait() or rc
    if rc:
        raise SystemExit(f"fingerprint worker failed rc={rc}")
    mean_p, std_p, sort_p = [], [], []
    for i in range(workers):
        z = np.load(chunk_path(n_r, n_a, i))
        mean_p.append(z["mean"])
        std_p.append(z["meanstd"])
        sort_p.append(z["sorted"])
    out = {
        "mean": np.vstack(mean_p),
        "meanstd": np.vstack(std_p),
        "sorted": np.vstack(sort_p),
    }
    if out["mean"].shape[0] != n:
        raise SystemExit(f"fingerprint row mismatch {out['mean'].shape[0]} vs {n}")
    return out


def pca2(x: np.ndarray, zscore: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if zscore:
        x = (x - x.mean(0)) / np.clip(x.std(0), 1e-9, None)
    xc = x - x.mean(0)
    # economy SVD; classical MDS of Euclidean distances
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    ev = (s[:6] ** 2) / max(x.shape[0] - 1, 1)
    return xc @ vt[:2].T, ev


def pairwise_l2(x: np.ndarray) -> np.ndarray:
    nrm = np.einsum("ij,ij->i", x, x)
    d2 = nrm[:, None] + nrm[None, :] - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2, dtype=float)


def torgerson(dist: np.ndarray, dim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    n = dist.shape[0]
    d2 = dist * dist
    h = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * h @ d2 @ h
    w, v = np.linalg.eigh(b)
    idx = np.argsort(w)[::-1]
    w, v = w[idx], v[:, idx]
    lam = np.clip(w[:dim], 0.0, None)
    return v[:, :dim] * np.sqrt(lam), w[:6]


def asinh_l2_mds(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    dist = pairwise_l2(x)
    pos = dist[dist > 0]
    sig = float(np.median(pos)) if pos.size else 1.0
    sig = max(sig, 1e-9)
    ash = np.arcsinh(dist / sig) / ASINH_NORM
    np.fill_diagonal(ash, 0.0)
    xy, ev = torgerson(ash, 2)
    return xy, ev, sig


def orient(xy: np.ndarray, i: int = GM_IDX, j: int = ICO_IDX) -> np.ndarray:
    out = xy.copy()
    if out[i, 0] > out[j, 0]:
        out[:, 0] *= -1.0
    if out[j, 1] < out[i, 1]:
        out[:, 1] *= -1.0
    return out


def unit_xy(xy: np.ndarray) -> np.ndarray:
    lo = xy.min(0)
    span = np.clip(xy.max(0) - lo, 1e-12, None)
    return (xy - lo) / span


def farthest(xy: np.ndarray, k: int, must) -> np.ndarray:
    n = len(xy)
    chosen = [int(i) for i in must]
    dmin = np.full(n, np.inf)
    for i in chosen:
        dmin = np.minimum(dmin, np.linalg.norm(xy - xy[i], axis=1))
    k = min(max(k, len(chosen)), n)
    while len(chosen) < k:
        j = int(np.argmax(dmin))
        chosen.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(xy - xy[j], axis=1))
    return np.unique(np.asarray(chosen, dtype=int))


def grid_box(xy: np.ndarray, ngrid: int, pad: float):
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
    return gx, gy, xx, yy


def support(xy: np.ndarray, xx: np.ndarray, yy: np.ndarray, pts: np.ndarray, cut: float):
    field = np.empty(xx.shape, dtype=bool)
    ngrid = xx.shape[1]
    for i0 in range(0, xx.shape[0], 40):
        i1 = min(i0 + 40, xx.shape[0])
        xs = xx[i0:i1].ravel()
        ys = yy[i0:i1].ravel()
        dx_ = xs[:, None] - pts[:, 0][None, :]
        dy_ = ys[:, None] - pts[:, 1][None, :]
        d2 = dx_ * dx_ + dy_ * dy_
        field[i0:i1] = (d2.min(1) <= cut * cut).reshape((i1 - i0, ngrid))
    return field


def nw_field(xy, values, pts, val, kernel: str, ell: float, ngrid=NGRID, pad=0.08):
    gx, gy, xx, yy = grid_box(xy, ngrid, pad)
    field = np.empty(xx.shape)
    ell2 = ell * ell
    for i0 in range(0, ngrid, 30):
        i1 = min(i0 + 30, ngrid)
        xs = xx[i0:i1].ravel()
        ys = yy[i0:i1].ravel()
        dx_ = xs[:, None] - pts[:, 0][None, :]
        dy_ = ys[:, None] - pts[:, 1][None, :]
        r2 = dx_ * dx_ + dy_ * dy_
        if kernel == "imq":
            k = 1.0 / np.sqrt(1.0 + r2 / ell2)
        else:
            k = np.exp(-0.5 * r2 / ell2)
        num = k @ val
        den = np.clip(k.sum(1), 1e-12, None)
        field[i0:i1] = (num / den).reshape((i1 - i0, ngrid))
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    mask = support(xy, xx, yy, pts, cut=0.10 * max(diam, 1e-9))
    field = np.where(mask, np.clip(field, 0.0, EMAX), np.nan)
    return gx, gy, field


def nn_field(xy, values, ngrid=NGRID, pad=0.08, k: int = 4):
    gx, gy, xx, yy = grid_box(xy, ngrid, pad)
    field = np.empty(xx.shape)
    for i0 in range(0, ngrid, 30):
        i1 = min(i0 + 30, ngrid)
        xs = xx[i0:i1].ravel()
        ys = yy[i0:i1].ravel()
        dx_ = xs[:, None] - xy[:, 0][None, :]
        dy_ = ys[:, None] - xy[:, 1][None, :]
        d2 = dx_ * dx_ + dy_ * dy_
        if k <= 1:
            nn = d2.argmin(1)
            pred = values[nn]
        else:
            part = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
            rows = np.arange(d2.shape[0])[:, None]
            dd = d2[rows, part]
            w = 1.0 / np.clip(dd, 1e-12, None)
            pred = (w * values[part]).sum(1) / w.sum(1)
        field[i0:i1] = pred.reshape((i1 - i0, ngrid))
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    mask = support(xy, xx, yy, xy, cut=0.10 * max(diam, 1e-9))
    field = np.where(mask, np.clip(field, 0.0, EMAX), np.nan)
    return gx, gy, field


def field_at(gx, gy, zz, pt) -> float:
    i = int(np.argmin(np.abs(gy - pt[1])))
    j = int(np.argmin(np.abs(gx - pt[0])))
    return float(zz[i, j])


def local_minima(field: np.ndarray, vmax: float = 2.2, sep: int = 8):
    out = []
    ny, nx = field.shape
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            v = field[i, j]
            if not np.isfinite(v) or v > vmax:
                continue
            nb = field[i - 1 : i + 2, j - 1 : j + 2]
            if np.nanmin(nb) >= v - 1e-12:
                out.append((v, i, j))
    out.sort()
    kept = []
    for v, i, j in out:
        if all(abs(i - ii) + abs(j - jj) > sep for _, ii, jj in kept):
            kept.append((v, i, j))
    return kept


def on_hull(xy: np.ndarray, idx: int, tol: float = 0.04) -> bool:
    pts = xy - xy.mean(0)
    lo, hi = pts.min(0), pts.max(0)
    p = pts[idx]
    span = np.clip(hi - lo, 1e-12, None)
    t = float(np.min(np.minimum(p - lo, hi - p) / span))
    return bool(t < tol)


def ring_barrier(xy, gx, gy, field, idx: int, r_in: float, r_out: float) -> float:
    """Mean field on an annulus minus field at the site. Positive = a well."""
    c = xy[idx]
    xx, yy = np.meshgrid(gx, gy)
    r = np.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2)
    ring = field[(r >= r_in) & (r <= r_out)]
    ring = ring[np.isfinite(ring)]
    if ring.size == 0:
        return 0.0
    return float(np.mean(ring) - field_at(gx, gy, field, c))


def nearest_min(mins, gx, gy, pt):
    if not mins:
        return None, np.inf
    best = None
    best_d = np.inf
    for k, (v, i, j) in enumerate(mins):
        q = np.array([gx[j], gy[i]])
        d = float(np.linalg.norm(q - pt))
        if d < best_d:
            best_d = d
            best = k
    return best, best_d


def score_energy(name: str, xy: np.ndarray, gx, gy, field) -> dict:
    gm, ico = xy[GM_IDX], xy[ICO_IDX]
    sep = float(np.linalg.norm(gm - ico))
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    sep_norm = sep / max(diam, 1e-12)
    e_gm = field_at(gx, gy, field, gm)
    e_ico = field_at(gx, gy, field, ico)
    mins = local_minima(field)
    ig, dg = nearest_min(mins, gx, gy, gm)
    ii, di = nearest_min(mins, gx, gy, ico)
    finite = field[np.isfinite(field)]
    p15 = float(np.nanpercentile(finite, 15)) if finite.size else np.nan
    well_gm = bool(np.isfinite(e_gm) and e_gm <= max(p15, 0.55))
    barrier = ring_barrier(xy, gx, gy, field, GM_IDX, 0.06 * diam, 0.16 * diam)
    two = bool(len(mins) >= 2 and ig is not None and ii is not None and ig != ii)
    if not two and sep_norm > SEP_BAR and np.isfinite(e_gm) and np.isfinite(e_ico):
        # split along the GM-ico axis of low-energy sites
        two = bool(abs(e_gm - e_ico) < 2.5 and sep_norm > SEP_BAR and well_gm)
    rim = on_hull(xy, GM_IDX)
    win = bool(two and well_gm and sep_norm > SEP_BAR and barrier > 0.08 and e_gm <= 1.0)
    return {
        "name": name,
        "sep": sep,
        "sep_norm": sep_norm,
        "diam": diam,
        "E_GM": None if not np.isfinite(e_gm) else float(e_gm),
        "E_ico": None if not np.isfinite(e_ico) else float(e_ico),
        "n_wells": len(mins),
        "two_basins": two,
        "gm_in_well": well_gm,
        "gm_barrier": barrier,
        "gm_on_rim": bool(rim),
        "well_gm": None if ig is None else int(ig),
        "well_ico": None if ii is None else int(ii),
        "d_well_gm": float(dg),
        "d_well_ico": float(di),
        "win": win,
        "gm": [float(gm[0]), float(gm[1])],
        "ico": [float(ico[0]), float(ico[1])],
    }


def mark(ax, xy):
    ax.scatter(
        xy[GM_IDX, 0],
        xy[GM_IDX, 1],
        s=140,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.7,
        zorder=6,
        label=rf"GM ${GM_E:.3f}$",
    )
    ax.scatter(
        xy[ICO_IDX, 0],
        xy[ICO_IDX, 1],
        s=80,
        marker="D",
        c="k",
        edgecolors="white",
        linewidths=0.7,
        zorder=6,
        label=rf"ico ${ICO_E:.3f}$",
    )
    ax.legend(fontsize=8, frameon=True, fancybox=False, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def paint_energy(ax, gx, gy, field, title: str):
    mesh = ax.contourf(
        gx, gy, field, levels=np.linspace(0, EMAX, 25), cmap=PES, extend="max"
    )
    ax.contour(
        gx,
        gy,
        np.where(np.isfinite(field), field, np.nan),
        levels=np.linspace(0.25, EMAX - 0.5, 12),
        colors="#1a1a2e",
        linewidths=0.35,
    )
    ax.set_title(title, fontsize=11)
    return mesh


def save_map(xy, gx, gy, field, dest: Path, title: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.6), facecolor="white")
    mesh = paint_energy(ax, gx, gy, field, title)
    mark(ax, xy)
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    fig.tight_layout()
    fig.savefig(dest, dpi=190, facecolor="white")
    plt.close(fig)
    print("wrote", dest)


def save_scatter(xy, energy, dest: Path, title: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.4), facecolor="white")
    rel = np.clip(energy - energy.min(), 0.0, EMAX)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=rel, s=8, cmap=PES, vmin=0, vmax=EMAX, linewidths=0)
    mark(ax, xy)
    ax.set_title(title, fontsize=11)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    fig.tight_layout()
    fig.savefig(dest, dpi=150, facecolor="white")
    plt.close(fig)


def blur2d(z: np.ndarray, sigma: float = 2.4) -> np.ndarray:
    rad = int(np.ceil(3.0 * sigma))
    t = np.arange(-rad, rad + 1, dtype=float)
    k = np.exp(-0.5 * (t / max(sigma, 1e-9)) ** 2)
    k /= k.sum()
    p = np.pad(z, rad, mode="edge")
    tmp = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, p)
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, tmp)


def cellmin_field(xy, values, ngrid=NGRID, pad=0.08, sigma: float = 2.6):
    gx, gy, xx, yy = grid_box(xy, ngrid, pad)
    xmin, xmax = float(gx[0]), float(gx[-1])
    ymin, ymax = float(gy[0]), float(gy[-1])
    ix = np.clip(((xy[:, 0] - xmin) / (xmax - xmin) * (ngrid - 1)).astype(int), 0, ngrid - 1)
    iy = np.clip(((xy[:, 1] - ymin) / (ymax - ymin) * (ngrid - 1)).astype(int), 0, ngrid - 1)
    grid = np.full((ngrid, ngrid), np.inf)
    for i, j, v in zip(iy, ix, values):
        if v < grid[i, j]:
            grid[i, j] = v
    obs = np.isfinite(grid)
    num = blur2d(np.where(obs, grid, 0.0), sigma=sigma)
    den = blur2d(obs.astype(float), sigma=sigma)
    blur = num / np.clip(den, 1e-12, None)
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    mask = support(xy, xx, yy, xy, cut=0.10 * max(diam, 1e-9)) & (den > 0.06)
    field = np.where(mask, np.clip(blur, 0.0, EMAX), np.nan)
    return gx, gy, field


def inducing(xy: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Low-energy sites plus farthest fill. Farthest-only picks the hot rim."""
    must = [GM_IDX, ICO_IDX]
    low = np.flatnonzero(rel <= 1.5)
    extra = [int(i) for i in low if int(i) not in must]
    seed = must + extra
    k = min(max(N_INDUCING, len(seed)), len(xy))
    return farthest(xy, k, seed)


def fill_all(xy: np.ndarray, rel: np.ndarray):
    idx = inducing(xy, rel)
    pts = xy[idx]
    val = rel[idx]
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    ell = 0.045 * max(diam, 1e-9)
    out = {}
    out["imq"] = nw_field(xy, rel, pts, val, "imq", ell)
    out["rbf"] = nw_field(xy, rel, pts, val, "rbf", ell * 0.80)
    out["nn"] = nn_field(xy, rel, ngrid=140, k=3)
    out["cellmin"] = cellmin_field(xy, rel)
    return out, ell, idx


def describe_split(fp: np.ndarray) -> dict:
    d = float(np.linalg.norm(fp[GM_IDX] - fp[ICO_IDX]))
    # median pairwise of a 200-point subsample
    rng = np.random.default_rng(0)
    take = rng.choice(len(fp), size=min(200, len(fp)), replace=False)
    sub = fp[take]
    med = float(np.median(pairwise_l2(sub)[np.triu_indices(len(sub), 1)]))
    return {"l2_gm_ico": d, "l2_med_sub": med, "l2_ratio": d / max(med, 1e-12)}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    energy, frames = load_min(MINFILE)
    if abs(float(energy[GM_IDX]) - GM_E) > 1e-3 or abs(float(energy[ICO_IDX]) - ICO_E) > 1e-3:
        raise SystemExit(f"index check failed E[0]={energy[GM_IDX]} E[40]={energy[ICO_IDX]}")
    rel = np.clip(energy - float(energy.min()), 0.0, None)
    print("n", len(energy), "E[GM]", float(energy[GM_IDX]), "E[ico]", float(energy[ICO_IDX]))

    # cheapest-first. Column-zscore recovers the second SOAP axis.
    # Sorted concat and extra angular bins only if the mean family fails.
    variants = [
        ("mean_r16a8_z", "mean", 16, 8, "pca_z"),
        ("meanstd_r16a8_z", "meanstd", 16, 8, "pca_z"),
        ("meanstd_r16a8", "meanstd", 16, 8, "pca"),
        ("mean_r16a8", "mean", 16, 8, "pca"),
        ("sorted_r16a8_z", "sorted", 16, 8, "pca_z"),
        ("sorted_r16a8", "sorted", 16, 8, "pca"),
        ("mean_r16a16_z", "mean", 16, 16, "pca_z"),
    ]

    family: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    scores = []
    maps = {}
    fields = {}
    winner = None

    for name, kind, n_r, n_a, embed in variants:
        fkey = (n_r, n_a)
        if fkey not in family:
            npy = {kn: DEST / f"fp_{kn}_r{n_r}a{n_a}.npy" for kn in ("mean", "meanstd", "sorted")}
            if all(p.is_file() and p.stat().st_size > 1000 for p in npy.values()):
                print(f"fingerprint family n_r={n_r} n_a={n_a} cache")
                family[fkey] = {kn: np.load(p) for kn, p in npy.items()}
            else:
                print(f"fingerprint family n_r={n_r} n_a={n_a} ...", flush=True)
                family[fkey] = build_family(frames, n_r, n_a)
                for kn, mat in family[fkey].items():
                    np.save(npy[kn], mat)
                    np.savetxt(DEST / f"fp_{kn}_r{n_r}a{n_a}.txt", mat, fmt="%.8e")
            for kn, mat in family[fkey].items():
                print(" ", kn, mat.shape, describe_split(mat))
        fp = family[fkey][kind]
        if embed == "pca":
            xy, ev = pca2(fp, zscore=False)
            extra = {"ev": [float(x) for x in ev], "sig": None}
        elif embed == "pca_z":
            xy, ev = pca2(fp, zscore=True)
            extra = {"ev": [float(x) for x in ev], "sig": None}
        else:
            xy, ev, sig = asinh_l2_mds(fp)
            extra = {"ev": [float(x) for x in ev], "sig": sig}
        xy = unit_xy(orient(xy))
        maps[name] = xy
        np.savetxt(DEST / f"{name}.xy", xy, fmt="%.8e")
        save_scatter(xy, energy, DEST / f"{name}_scatter.png", name)
        filled, ell, ind = fill_all(xy, rel)
        print(f"  embed {embed} ev={extra['ev'][:3]} ell={ell:.4f} inducing={len(ind)}")
        for fill_name, (gx, gy, field) in filled.items():
            tag = f"{name}_{fill_name}"
            rec = score_energy(tag, xy, gx, gy, field)
            rec.update(
                {
                    "kind": kind,
                    "n_r": n_r,
                    "n_a": n_a,
                    "embed": embed,
                    "fill": fill_name,
                    "hd": int(fp.shape[1]),
                    "ell": ell,
                    **describe_split(fp),
                    **{f"ev{i}": extra["ev"][i] if i < len(extra["ev"]) else None for i in range(3)},
                }
            )
            scores.append(rec)
            fields[tag] = (xy, gx, gy, field)
            print(
                f"  {tag:28s} sep={rec['sep_norm']:.3f} wells={rec['n_wells']} "
                f"Egm={rec['E_GM']} Eico={rec['E_ico']} two={rec['two_basins']} "
                f"well={rec['gm_in_well']} bar={rec['gm_barrier']:.3f} "
                f"rim={rec['gm_on_rim']} win={rec['win']}"
            )
            save_map(xy, gx, gy, field, DEST / f"{tag}.png", tag.replace("_", " "))
            if rec["win"] and winner is None:
                winner = tag
        # stop early once a mean or meanstd variant already wins
        if winner is not None and kind in ("mean", "meanstd"):
            break
        if winner is not None and kind == "sorted":
            break

    if winner is None:
        # pick the strongest two-basin / GM-well candidate
        def rank(r):
            return (
                int(r["two_basins"]),
                int(r["gm_in_well"]),
                int(not r["gm_on_rim"]),
                r["sep_norm"],
                r["gm_barrier"],
                -abs(r["E_GM"] if r["E_GM"] is not None else 9.0),
            )

        ranked = sorted(scores, key=rank, reverse=True)
        winner = ranked[0]["name"] if ranked else None
        print("no strict win; taking", winner)

    payload = {
        "n": int(len(energy)),
        "gm_idx": GM_IDX,
        "ico_idx": ICO_IDX,
        "winner": winner,
        "scores": scores,
    }
    (DEST / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")

    if winner is None:
        raise SystemExit("no SOAP map produced")
    xy, gx, gy, field = fields[winner]
    rec = next(s for s in scores if s["name"] == winner)
    title = r"SOAP-like neighbor density   $E-E_{\mathrm{GM}}$"
    save_map(xy, gx, gy, field, FIG, title)
    save_map(xy, gx, gy, field, DEST / "elja_occ_lj38_soap.png", title)
    print("winner", winner, json.dumps({k: rec[k] for k in (
        "sep_norm", "n_wells", "two_basins", "gm_in_well", "gm_barrier",
        "gm_on_rim", "E_GM", "E_ico", "win",
    )}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-chunk", type=int, default=None)
    ap.add_argument("--fp-chunks", type=int, default=10)
    ap.add_argument("--n-r", type=int, default=N_R)
    ap.add_argument("--n-a", type=int, default=N_A)
    args = ap.parse_args()
    if args.fp_chunk is not None:
        run_fp_chunk(args.fp_chunk, args.fp_chunks, args.n_r, args.n_a)
    else:
        main()
