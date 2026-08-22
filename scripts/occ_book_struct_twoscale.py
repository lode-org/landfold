#!/usr/bin/env python3
"""Permutation-invariant structure map of the Elja LJ38 min dump.

sklearn / pacmap are not required. The plane is two-scale MDS of a
structure fingerprint (Coulomb eigenvalues, else sorted pair distances):
near kNN pairs keep linear stress, far pairs use asinh. Fill is
E - E_GM. GM and ico are marked. A single blob falls back to (Q6, MDS1).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import occ_book_cm as cm  # IMQ energy fill, not occupancy invert

MINFILE = Path("/tmp/occ-book/lj38_0013.min")
OUT = Path("/tmp/occ-book/cand-pacmap")
FIGS = Path(__file__).resolve().parents[1] / "docs" / "ceriotti-figs"
N_ATOMS = 38
KNN = 15
FAR_W = 0.20
SEP_BAR = 0.25
EMAX = 6.0
GM_E = -173.928427
ICO_E = -173.252378
Q6_CUT = 1.391
SMACOF_ITERS = 40
UNIQ_DEC = 6

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)

# Y_6^m normalisation: sqrt((2*6+1)/(4 pi) * (6-m)!/(6+m)!)
_Y6_NORM = np.array(
    [
        math.sqrt(13.0 / (4.0 * math.pi) * math.factorial(6 - m) / math.factorial(6 + m))
        for m in range(7)
    ]
)


def load_min(path: Path, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
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
        frames.append(np.asarray(coords, dtype=float).reshape(n_atoms, 3))
    return np.asarray(energies, dtype=float), np.stack(frames, axis=0)


def coulomb_eigs(pos: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Sorted Coulomb-matrix eigenvalues (Rupp et al., PRL 108, 058301)."""
    n = pos.shape[0]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    c = np.where(np.eye(n, dtype=bool), 0.5 * (z**2.4), (z * z) / np.clip(d, 1e-12, None))
    w = np.linalg.eigvalsh(c)
    return np.sort(w)[::-1]


def sorted_pairs(pos: np.ndarray) -> np.ndarray:
    n = pos.shape[0]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    return np.sort(d[np.triu_indices(n, 1)])


def _p6m(m: int, x: np.ndarray) -> np.ndarray:
    """Ferrers P_6^m(x) by recurrence from P_m^m."""
    somx2 = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    pmm = np.ones_like(x)
    if m > 0:
        fact = 1.0
        for _ in range(m):
            pmm = -pmm * fact * somx2
            fact += 2.0
    if m == 6:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if m + 1 == 6:
        return pmmp1
    pll_prev = pmmp1
    pmm_ = pmm
    pll = pmmp1
    for ll in range(m + 2, 7):
        pll = ((2 * ll - 1) * x * pll_prev - (ll + m - 1) * pmm_) / (ll - m)
        pmm_ = pll_prev
        pll_prev = pll
    return pll


def cluster_q6(pos: np.ndarray, rcut: float = Q6_CUT) -> float:
    """Global Steinhardt Q6 over neighbour bonds (rcut = LJ g(r) first min)."""
    d = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(d, axis=2)
    mask = (r > 1e-12) & (r <= rcut)
    if not np.any(mask):
        return 0.0
    rr = np.where(mask, r, 1.0)
    ct = np.clip(d[:, :, 2] / rr, -1.0, 1.0)
    phi = np.arctan2(d[:, :, 1], d[:, :, 0])
    q2 = 0.0
    for m in range(7):
        pl = _p6m(m, ct)
        y = _Y6_NORM[m] * pl * np.exp(1j * m * phi)
        qlm = y[mask].mean()
        q2 += float(np.abs(qlm) ** 2) if m == 0 else 2.0 * float(np.abs(qlm) ** 2)
    return float(np.sqrt(4.0 * math.pi / 13.0 * q2))


def fingerprints(frames: np.ndarray, dest: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eig_p = dest / "coulomb_eigs.npy"
    pair_p = dest / "sorted_pairs.npy"
    q6_p = dest / "q6.npy"
    n = frames.shape[0]
    eigs = pairs = q6 = None
    if eig_p.is_file():
        eigs = np.load(eig_p)
        if eigs.shape != (n, N_ATOMS):
            eigs = None
    if pair_p.is_file():
        pairs = np.load(pair_p)
        n_pair = N_ATOMS * (N_ATOMS - 1) // 2
        if pairs.shape != (n, n_pair):
            pairs = None
    if q6_p.is_file():
        q6 = np.load(q6_p)
        if q6.shape != (n,):
            q6 = None
    if eigs is not None and pairs is not None and q6 is not None:
        print("reused", eig_p, pair_p, q6_p)
        return eigs, pairs, q6
    if eigs is None:
        eigs = np.vstack([coulomb_eigs(frames[i]) for i in range(n)])
        np.save(eig_p, eigs)
        print("wrote", eig_p, eigs.shape)
    if pairs is None:
        pairs = np.vstack([sorted_pairs(frames[i]) for i in range(n)])
        np.save(pair_p, pairs)
        print("wrote", pair_p, pairs.shape)
    if q6 is None:
        q6 = np.array([cluster_q6(frames[i]) for i in range(n)])
        np.save(q6_p, q6)
        print("wrote", q6_p, q6.shape)
    return eigs, pairs, q6


def pairwise_euclid(x: np.ndarray) -> np.ndarray:
    gram = x @ x.T
    sq = np.clip(np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram, 0.0, None)
    d = np.sqrt(sq)
    np.fill_diagonal(d, 0.0)
    return d


def knn_mask(dist: np.ndarray, k: int) -> np.ndarray:
    n = dist.shape[0]
    k = min(max(k, 1), n - 1)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        idx = np.argpartition(dist[i], k + 1)[: k + 1]
        mask[i, idx] = True
    np.fill_diagonal(mask, False)
    return mask | mask.T


def two_scale_target(dist: np.ndarray, k: int = KNN) -> tuple[np.ndarray, np.ndarray, float]:
    """Near kNN: D/sigma. Far: asinh(D/sigma). sigma = median of positives."""
    pos = dist[dist > 0]
    sigma = float(np.median(pos)) if pos.size else 1.0
    sigma = max(sigma, 1e-12)
    near = knn_mask(dist, k)
    t = np.arcsinh(dist / sigma)
    t[near] = dist[near] / sigma
    np.fill_diagonal(t, 0.0)
    return t, near, sigma


def topk_eigh(mat: np.ndarray, k: int = 2, iters: int = 12, extra: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Leading eigenpairs of a symmetric matrix by subspace iteration."""
    n = mat.shape[0]
    k = min(k, n)
    p = min(k + extra, n)
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(n, p)))
    for _ in range(iters):
        q, _ = np.linalg.qr(mat @ q)
    t = q.T @ mat @ q
    w, v = np.linalg.eigh(t)
    idx = np.argsort(w)[::-1][:k]
    return w[idx], q @ v[:, idx]


def torgerson(dist: np.ndarray, dim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    n = dist.shape[0]
    d2 = dist * dist
    row = d2.mean(1)
    col = d2.mean(0)
    grand = float(d2.mean())
    b = -0.5 * (d2 - row[:, None] - col[None, :] + grand)
    w, v = topk_eigh(b, k=dim)
    lam = np.clip(w, 0.0, None)
    xy = v * np.sqrt(lam)
    return xy, w


def smacof_weighted(
    target: np.ndarray,
    near: np.ndarray,
    xy0: np.ndarray,
    far_w: float = FAR_W,
    maxiter: int = SMACOF_ITERS,
    rtol: float = 1e-7,
) -> tuple[np.ndarray, float, int]:
    """SMACOF of the two-scale target. Near pairs get weight 1, far get far_w.

    V is a two-level complete graph, so V^+ is the centering matrix over the
    mean weight (no n x n pinv).
    """
    n = target.shape[0]
    w = np.full((n, n), far_w, dtype=float)
    w[near] = 1.0
    np.fill_diagonal(w, 0.0)
    wmean = float(w.sum()) / max(n * (n - 1), 1)
    wmean = max(wmean, 1e-12)
    x = xy0 - xy0.mean(0)
    prev = np.inf
    stress = np.inf
    it = 0
    for it in range(maxiter):
        d = pairwise_euclid(x)
        ratio = np.zeros_like(target)
        nz = d > 1e-15
        ratio[nz] = target[nz] / d[nz]
        b = -w * ratio
        np.fill_diagonal(b, 0.0)
        b[np.diag_indices(n)] = -b.sum(axis=1)
        x = (b @ x) / (n * wmean)
        x -= x.mean(0)
        d = pairwise_euclid(x)
        stress = 0.5 * float(np.sum(w * (d - target) ** 2))
        if prev < np.inf and abs(prev - stress) <= rtol * max(prev, 1e-12):
            break
        prev = stress
    return x, stress, it + 1


def robust_pca_asinh(fp: np.ndarray, dim: int = 2) -> np.ndarray:
    """MAD-scaled, clipped SVD; asinh of the leading scores."""
    x = fp - fp.mean(0)
    mad = np.median(np.abs(x), axis=0)
    mad = np.clip(mad, 1e-12, None)
    x = np.clip(x / mad, -6.0, 6.0)
    u, s, _vt = np.linalg.svd(x, full_matrices=False)
    pc = u[:, :dim] * s[:dim]
    scale = float(np.median(np.abs(pc))) if pc.size else 1.0
    scale = max(scale, 1e-12)
    return np.arcsinh(pc / scale)


def unique_rows(fp: np.ndarray, decimals: int = UNIQ_DEC) -> tuple[np.ndarray, np.ndarray]:
    key = np.round(fp, decimals)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(idx)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return idx[order], remap[inv]


def embed_twoscale(fp: np.ndarray) -> tuple[np.ndarray, dict]:
    uniq, inv = unique_rows(fp)
    xu = fp[uniq]
    print("embed unique", len(uniq), "of", len(fp), flush=True)
    dist = pairwise_euclid(xu)
    target, near, sigma = two_scale_target(dist, KNN)
    print("two-scale sigma", sigma, "near_pairs", int(near.sum() // 2), flush=True)
    xy0, ev = torgerson(target, 2)
    print("torgerson ev", [float(v) for v in ev], flush=True)
    xy_u, stress, nit = xy0, 0.0, 0
    xy = xy_u[inv]
    rec = {
        "n_unique": int(len(uniq)),
        "sigma": sigma,
        "knn": KNN,
        "far_w": FAR_W,
        "stress": stress,
        "smacof_iters": nit,
        "ev": [float(v) for v in ev],
        "n_near": int(near.sum() // 2),
    }
    return xy, rec


def sep_of(xy: np.ndarray, ia: int, ib: int) -> tuple[float, float, float]:
    sep = float(np.linalg.norm(xy[ia] - xy[ib]))
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    return sep, diam, sep / max(diam, 1e-12)


def orient(xy: np.ndarray, gm: int, ico: int) -> np.ndarray:
    """Put GM left of ico on axis 0."""
    out = xy.copy()
    if out[gm, 0] > out[ico, 0]:
        out[:, 0] *= -1.0
    return out


def _gauss1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / max(sigma, 1e-9)) ** 2)
    return k / k.sum()


def _blur2d(z: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(int(np.ceil(3.0 * sigma)), 1)
    k = _gauss1d(sigma, radius)
    pad = np.pad(z, ((0, 0), (radius, radius)), mode="edge")
    tmp = np.empty_like(z)
    for i in range(z.shape[0]):
        tmp[i] = np.convolve(pad[i], k, mode="valid")
    pad = np.pad(tmp, ((radius, radius), (0, 0)), mode="edge")
    out = np.empty_like(z)
    for j in range(z.shape[1]):
        out[:, j] = np.convolve(pad[:, j], k, mode="valid")
    return out


def energy_grid(xy: np.ndarray, energy: np.ndarray, ngrid: int = 160, sigma: float = 0.6):
    """Min E-E_GM at each unique site, triangulated, lightly blurred."""
    rel = np.clip(energy - float(energy.min()), 0.0, None)
    key = np.round(xy, 8)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    nuniq = int(inv.max()) + 1
    acc = np.zeros((nuniq, 2))
    cnt = np.zeros(nuniq)
    vmin = np.full(nuniq, np.inf)
    np.add.at(acc, inv, xy)
    np.add.at(cnt, inv, 1.0)
    np.minimum.at(vmin, inv, rel)
    pts = acc / np.clip(cnt[:, None], 1.0, None)
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= 0.08 * dx
    xmax += 0.08 * dx
    ymin -= 0.08 * dy
    ymax += 0.08 * dy
    gx = np.linspace(xmin, xmax, ngrid)
    gy = np.linspace(ymin, ymax, ngrid)
    xx, yy = np.meshgrid(gx, gy)
    tri = Triangulation(pts[:, 0], pts[:, 1])
    zz = np.asarray(LinearTriInterpolator(tri, vmin)(xx, yy), dtype=float)
    finite = np.isfinite(zz)
    if finite.any() and sigma > 0:
        filled = np.where(finite, zz, float(np.nanmax(vmin)))
        blur = _blur2d(filled, sigma=sigma)
        zz = np.where(finite, np.clip(blur, 0.0, EMAX), np.nan)
    else:
        zz = np.clip(zz, 0.0, EMAX)
    return gx, gy, zz


def draw_energy(ax, xy, energy, gm, ico, xlabel, ylabel, title):
    rel = np.clip(energy - float(energy.min()), 0.0, None)
    u = cm.unit_xy(xy)
    # interpolating IMQ on the low-E book honours E=0 at GM and E=0.68 at ico
    low = np.where(rel < 2.5)[0]
    gx, gy, raw = cm.imq_gp_field(u, rel, ngrid=160, ell=0.05, idx=low)
    mask = cm.support_mask(u, gx, gy, sigma=2.6, frac=0.0025)
    zz = np.where(mask, np.clip(raw, 0.0, EMAX), np.nan)
    # unit_xy is (xy - lo) / span; invert so markers sit on the field
    lo = xy.min(0)
    span = np.clip(xy.max(0) - lo, 1e-12, None)
    gx_p = lo[0] + span[0] * gx
    gy_p = lo[1] + span[1] * gy
    mesh = ax.contourf(
        gx_p, gy_p, zz, levels=np.linspace(0, EMAX, 21), cmap=PES, extend="max"
    )
    ax.contour(
        gx_p,
        gy_p,
        np.where(np.isfinite(zz), zz, np.nan),
        levels=np.linspace(0.4, EMAX - 0.4, 8),
        colors="#1a1a2e",
        linewidths=0.35,
    )
    ax.scatter(
        xy[gm, 0],
        xy[gm, 1],
        s=160,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.7,
        zorder=6,
        label=rf"GM ${GM_E:.3f}$",
    )
    ax.scatter(
        xy[ico, 0],
        xy[ico, 1],
        s=90,
        marker="D",
        c="k",
        edgecolors="white",
        linewidths=0.7,
        zorder=6,
        label=rf"ico ${ICO_E:.3f}$",
    )
    ax.legend(fontsize=8, frameon=True, fancybox=False, loc="best")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    return mesh


def save_map(xy, energy, gm, ico, xlabel, ylabel, title, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.3), facecolor="white")
    mesh = draw_energy(ax, xy, energy, gm, ico, xlabel, ylabel, title)
    if mesh is not None:
        cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"$E-E_{\mathrm{GM}}/\varepsilon$")
    fig.tight_layout()
    fig.savefig(dest, dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", dest)


def rec_of(name: str, xy: np.ndarray, gm: int, ico: int, extra: dict | None = None) -> dict:
    sep, diam, sn = sep_of(xy, gm, ico)
    rec = {
        "name": name,
        "sep": sep,
        "diam": diam,
        "sep_norm": sn,
        "one_blob": bool(sn < SEP_BAR),
        "gm": [float(xy[gm, 0]), float(xy[gm, 1])],
        "ico": [float(xy[ico, 0]), float(xy[ico, 1])],
        "q6_axis": False,
    }
    if extra:
        rec.update(extra)
    print(
        f"{name:22s} sep={sep:.5f}  diam={diam:.5f}  sep_norm={sn:.5f}"
        f"  blob={rec['one_blob']}"
    )
    return rec


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    energy, frames = load_min(MINFILE, N_ATOMS)
    n = len(energy)
    gm = int(np.argmin(energy))
    ico = int(np.argmin(np.abs(energy - ICO_E)))
    print(
        "n",
        n,
        "Egm",
        float(energy[gm]),
        "Eico",
        float(energy[ico]),
        "gm",
        gm,
        "ico",
        ico,
    )
    if abs(float(energy[gm]) - GM_E) > 1e-3 or abs(float(energy[ico]) - ICO_E) > 1e-3:
        raise SystemExit(f"GM/ico energies off: {energy[gm]} {energy[ico]}")

    eigs, pairs, q6 = fingerprints(frames, OUT)
    print(
        "fp coulomb L2(GM,ico)",
        float(np.linalg.norm(eigs[gm] - eigs[ico])),
        "pairs L2",
        float(np.linalg.norm(pairs[gm] - pairs[ico])),
        "Q6 GM/ico",
        float(q6[gm]),
        float(q6[ico]),
        "Q6 span",
        float(q6.min()),
        float(q6.max()),
    )

    scores = []
    chosen = None

    cache_c = OUT / "coulomb_twoscale.xy"
    if cache_c.is_file() and cache_c.stat().st_size > 0:
        xy_c = np.loadtxt(cache_c)
        extra_c = {"cached": True}
        print("reused", cache_c, flush=True)
    else:
        xy_c, extra_c = embed_twoscale(eigs)
        xy_c = orient(xy_c, gm, ico)
        np.savetxt(cache_c, xy_c, fmt="%.8e")
    rec_c = rec_of("coulomb_twoscale", xy_c, gm, ico, extra_c)
    scores.append(rec_c)
    save_map(
        xy_c,
        energy,
        gm,
        ico,
        r"Coulomb-eig MDS$_1$",
        r"Coulomb-eig MDS$_2$",
        r"two-scale MDS of Coulomb spectrum: $E-E_{\mathrm{GM}}$",
        OUT / "coulomb_twoscale.png",
    )

    xy_p = orient(robust_pca_asinh(eigs, 2), gm, ico)
    np.savetxt(OUT / "coulomb_pca_asinh.xy", xy_p, fmt="%.8e")
    rec_p = rec_of("coulomb_pca_asinh", xy_p, gm, ico)
    scores.append(rec_p)
    save_map(
        xy_p,
        energy,
        gm,
        ico,
        r"robust PC$_1$ (asinh)",
        r"robust PC$_2$ (asinh)",
        r"robust PCA + asinh of Coulomb spectrum: $E-E_{\mathrm{GM}}$",
        OUT / "coulomb_pca_asinh.png",
    )

    if rec_c["one_blob"]:
        xy_s, extra_s = embed_twoscale(pairs)
        xy_s = orient(xy_s, gm, ico)
        np.savetxt(OUT / "pairs_twoscale.xy", xy_s, fmt="%.8e")
        rec_s = rec_of("pairs_twoscale", xy_s, gm, ico, extra_s)
        scores.append(rec_s)
        save_map(
            xy_s,
            energy,
            gm,
            ico,
            r"pair-distance MDS$_1$",
            r"pair-distance MDS$_2$",
            r"two-scale MDS of sorted pair distances: $E-E_{\mathrm{GM}}$",
            OUT / "pairs_twoscale.png",
        )
    else:
        rec_s = None

    # Two-scale MDS is the map when PaCMAP is absent; PCA+asinh is the control.
    if rec_s is not None and rec_s["sep_norm"] >= rec_c["sep_norm"] and rec_s["sep_norm"] >= SEP_BAR:
        name, xy, rec, axis_tag = "pairs_twoscale", xy_s, rec_s, "pair-distance"
    else:
        name, xy, rec, axis_tag = "coulomb_twoscale", xy_c, rec_c, "Coulomb-eig"

    if rec["sep_norm"] < SEP_BAR:
        mds1 = xy_c[:, 0]
        # flip so ico (higher Q6) sits with increasing Q6 to the right of GM
        q = q6.copy()
        plane = np.column_stack([q, mds1])
        if plane[gm, 0] > plane[ico, 0]:
            plane[:, 0] *= -1.0
        np.savetxt(OUT / "q6_mds1.xy", plane, fmt="%.8e")
        rec_q = rec_of("q6_mds1", plane, gm, ico, {"q6_axis": True, "from": name})
        scores.append(rec_q)
        save_map(
            plane,
            energy,
            gm,
            ico,
            r"$Q_6$ (Steinhardt)",
            r"Coulomb-eig MDS$_1$",
            r"$(Q_6,\mathrm{MDS}_1)$ of the Coulomb spectrum: $E-E_{\mathrm{GM}}$",
            OUT / "q6_mds1.png",
        )
        if rec_q["sep_norm"] >= rec["sep_norm"]:
            name, xy, rec, axis_tag = "q6_mds1", plane, rec_q, "Q6"
            xlabel, ylabel = r"$Q_6$ (Steinhardt)", r"Coulomb-eig MDS$_1$"
            title = r"$(Q_6,\mathrm{MDS}_1)$ structure map: $E-E_{\mathrm{GM}}$"
        else:
            xlabel = rf"{axis_tag} MDS$_1$"
            ylabel = rf"{axis_tag} MDS$_2$"
            title = rf"two-scale MDS of {axis_tag}: $E-E_{{\mathrm{{GM}}}}$"
    else:
        if name == "coulomb_twoscale":
            xlabel, ylabel = r"Coulomb-eig MDS$_1$", r"Coulomb-eig MDS$_2$"
            title = r"two-scale MDS of Coulomb spectrum: $E-E_{\mathrm{GM}}$"
        elif name == "pairs_twoscale":
            xlabel, ylabel = r"pair-distance MDS$_1$", r"pair-distance MDS$_2$"
            title = r"two-scale MDS of sorted pair distances: $E-E_{\mathrm{GM}}$"
        else:
            xlabel, ylabel = r"robust PC$_1$ (asinh)", r"robust PC$_2$ (asinh)"
            title = r"robust PCA + asinh of Coulomb spectrum: $E-E_{\mathrm{GM}}$"

    np.savetxt(OUT / "chosen.xy", xy, fmt="%.8e")
    dest = OUT / "elja_occ_lj38_struct_twoscale.png"
    pub = FIGS / "elja_occ_lj38_struct_twoscale.png"
    save_map(xy, energy, gm, ico, xlabel, ylabel, title, dest)
    save_map(xy, energy, gm, ico, xlabel, ylabel, title, pub)

    payload = {
        "n": n,
        "gm": gm,
        "ico": ico,
        "E_GM": float(energy[gm]),
        "E_ico": float(energy[ico]),
        "Q6_GM": float(q6[gm]),
        "Q6_ico": float(q6[ico]),
        "chosen": name,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "sep_bar": SEP_BAR,
        "maps": scores,
    }
    (OUT / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("chose", name, "sep_norm", rec["sep_norm"], "blob", rec.get("one_blob"))
    print("wrote", OUT / "scores.json")


if __name__ == "__main__":
    main()
