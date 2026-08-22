#!/usr/bin/env python3
"""Permutation-aware Hungarian RMSD map of the Elja LJ38 .min book.

Full 4042 x 4042 Hungarian is O(N^2) Kabsch-assignments. This script
does the cheap Cameron plane and a landmark Nyström instead:

  d_GM, d_ico   Hungarian RMSD of each quenched min to the GM xyz and
                the icosahedral prototype (second deep well, E ~ -173.25).
                Identical geometries (pairwise-distance fingerprint) map
                to the axis intercepts: GM at (0, D), ico at (D, 0).

  landmarks     lowest 100 E plus 100 random, Hungarian among them,
                Torgerson MDS, Nyström / IDW of the rest.

The (d_GM, d_ico) plane is the structure-space committor analogue.
Energy is an IDW (p=4) interpolant of E - E_GM, pinned at the two
wells (no leftover occupancy).
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

try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
except ImportError:
    _scipy_lsa = None

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ceriotti-figs"
MIN_PATH = Path("/tmp/occ-book/lj38_0013.min")
HIST = Path("/tmp/occ-book/lj38_decaf_e.hist")
XYZ_DIR = Path("/tmp/occ-book/xyz")
DEST = Path("/tmp/occ-book/cand-hung")
FIG = OUT / "elja_occ_lj38_hungarian.png"

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)
GM_E = -173.928427
ICO_E = -173.252378
EMAX = 4.0
N_ATOMS = 38
N_LAND_LOW = 100
N_LAND_RAND = 100
N_INDUCING = 96
RNG_SEED = 0
FP_TOL = 1e-4


def linear_sum_assignment(cost: np.ndarray):
    """Square assignment. SciPy if present, else vectorized JV."""
    if _scipy_lsa is not None:
        return _scipy_lsa(cost)
    cost = np.asarray(cost, dtype=float)
    n = cost.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = cost[i0 - 1] - u[i0] - v[1:]
            not_used = ~used[1:]
            better = not_used & (cur < minv[1:])
            minv[1:] = np.where(better, cur, minv[1:])
            way[1:] = np.where(better, j0, way[1:])
            vals = np.where(not_used, minv[1:], np.inf)
            j1 = int(np.argmin(vals)) + 1
            delta = float(minv[j1])
            used_idx = np.flatnonzero(used)
            u[p[used_idx]] += delta
            v[used_idx] -= delta
            minv[1:] -= np.where(not_used, delta, 0.0)
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    col = np.empty(n, dtype=int)
    for j in range(1, n + 1):
        col[p[j] - 1] = j - 1
    return np.arange(n), col


def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rotation R with Q @ R ~ P. Rows are atoms, both centered."""
    H = Q.T @ P
    U, _s, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U = U.copy()
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def pairwise_fp(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0)
    d2 = ((Xc[:, None, :] - Xc[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(Xc.shape[0], 1)
    return np.sort(np.sqrt(d2[iu]))


def atom_sig(X: np.ndarray) -> np.ndarray:
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, 0.0)
    return np.sort(np.sqrt(d2), axis=1)


def _refine(P: np.ndarray, Q: np.ndarray, niter: int = 3) -> float:
    Qw = Q
    best = np.inf
    for _ in range(niter):
        d2 = ((P[:, None, :] - Qw[None, :, :]) ** 2).sum(-1)
        _r, col = linear_sum_assignment(d2)
        R = kabsch(P, Qw[col])
        Qw = Qw @ R
        val = float(np.sqrt(np.mean(np.sum((P - Qw[col]) ** 2, axis=1))))
        if val < best:
            best = val
        if val < 1e-10:
            break
    return best


def _random_rots(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    out = [np.eye(3)]
    for _ in range(n):
        A = rng.normal(size=(3, 3))
        q, r = np.linalg.qr(A)
        q *= np.sign(np.diag(r))
        if np.linalg.det(q) < 0.0:
            q = q.copy()
            q[:, 0] *= -1.0
        out.append(q)
    return out


def _from_sig(P: np.ndarray, Q: np.ndarray, sP: np.ndarray, sQ: np.ndarray) -> float:
    _r, col = linear_sum_assignment(((sP[:, None, :] - sQ[None, :, :]) ** 2).sum(-1))
    R = kabsch(P, Q[col])
    return _refine(P, Q @ R, niter=2)


def hungarian_rmsd(
    P: np.ndarray,
    Q: np.ndarray,
    *,
    n_rand: int = 3,
    reflections: bool = True,
    rng: np.random.Generator | None = None,
    fp_p: np.ndarray | None = None,
    fp_q: np.ndarray | None = None,
) -> float:
    """Min RMSD over atom permutations and proper/improper rotations."""
    Pc = P - P.mean(0)
    Qc0 = Q - Q.mean(0)
    if fp_p is None:
        fp_p = pairwise_fp(Pc)
    if fp_q is None:
        fp_q = pairwise_fp(Qc0)
    if np.max(np.abs(fp_p - fp_q)) < FP_TOL:
        return 0.0
    rng = rng or np.random.default_rng(0)
    sP = atom_sig(Pc)
    sQ = atom_sig(Qc0)
    best = _from_sig(Pc, Qc0, sP, sQ)
    if best < 1e-8:
        return 0.0
    if reflections:
        best = min(best, _from_sig(Pc, -Qc0, sP, atom_sig(-Qc0)))
        if best < 0.12:
            return float(best)
    if n_rand > 0 and best > 0.08:
        for R0 in _random_rots(n_rand, rng)[1:]:
            val = _refine(Pc, Qc0 @ R0, niter=2)
            if val < best:
                best = val
            if best < 1e-8:
                return 0.0
    return float(best)


def load_min(path: Path):
    raw = np.loadtxt(path)
    if raw.ndim != 2 or raw.shape[1] != 1 + 3 * N_ATOMS:
        raise SystemExit("bad min file %s shape %s" % (path, raw.shape))
    energy = raw[:, 0].astype(float)
    xyz = raw[:, 1:].reshape(-1, N_ATOMS, 3)
    return energy, xyz


def load_xyz(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    pos = []
    for line in lines[2 : 2 + n]:
        p = line.split()
        pos.append([float(p[-3]), float(p[-2]), float(p[-1])])
    return np.asarray(pos)


def load_hist(path: Path):
    wells, emin, rows = [], [], []
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        wells.append(float(p[1]))
        emin.append(float(p[2]))
        rows.append([float(x) for x in p[3:]])
    return {
        "hist": np.asarray(rows),
        "energy": np.asarray(emin),
        "wells": np.asarray(wells),
    }


def assignment_path() -> Path | None:
    for p in (
        DEST / "family.assign",
        Path("/tmp/occ-book/lj38.family"),
        Path("/tmp/occ-book/lj38.assign"),
    ):
        if p.is_file():
            return p
    return None


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


def landmark_indices(energy: np.ndarray, gm_idx: int, ico_idx: int, rng) -> np.ndarray:
    order = np.argsort(energy)
    low = order[:N_LAND_LOW]
    rest = np.setdiff1d(np.arange(len(energy)), low, assume_unique=False)
    rng.shuffle(rest)
    pick = rest[:N_LAND_RAND]
    idx = np.unique(np.concatenate([low, pick, np.array([gm_idx, ico_idx], dtype=int)]))
    return np.sort(idx)


def pairwise_hungarian(xyz: np.ndarray, idx: np.ndarray, rng) -> np.ndarray:
    m = len(idx)
    D = np.zeros((m, m))
    n_pairs = m * (m - 1) // 2
    done = 0
    for a in range(m):
        for b in range(a + 1, m):
            d = hungarian_rmsd(xyz[idx[a]], xyz[idx[b]], n_rand=2, rng=rng)
            D[a, b] = D[b, a] = d
            done += 1
            if done % 2000 == 0 or done == n_pairs:
                print("  landmark pairs", done, "/", n_pairs, flush=True)
    return D


def nystrom_idw(xy_feat: np.ndarray, land_idx: np.ndarray, land_xy: np.ndarray, k: int = 8):
    """Place every point by IDW of k nearest landmarks in the (d_GM, d_ico) plane."""
    feat_l = xy_feat[land_idx]
    out = np.empty((len(xy_feat), land_xy.shape[1]))
    for i in range(len(xy_feat)):
        d = np.linalg.norm(feat_l - xy_feat[i], axis=1)
        if d.min() < 1e-14:
            out[i] = land_xy[int(np.argmin(d))]
            continue
        nn = np.argpartition(d, min(k, len(d) - 1))[:k]
        w = 1.0 / np.clip(d[nn] ** 2, 1e-12, None)
        w /= w.sum()
        out[i] = w @ land_xy[nn]
    return out


def farthest_indices(xy: np.ndarray, k: int, must) -> np.ndarray:
    n = len(xy)
    chosen = [int(i) for i in must]
    dmin = np.full(n, np.inf)
    for i in chosen:
        dmin = np.minimum(dmin, np.linalg.norm(xy - xy[i], axis=1))
    while len(chosen) < min(k, n):
        j = int(np.argmax(dmin))
        if j in chosen:
            dmin[j] = -1.0
            continue
        chosen.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(xy - xy[j], axis=1))
    return np.unique(np.asarray(chosen, dtype=int))


def energy_grid(xy: np.ndarray, energy: np.ndarray, gm_idx: int, ico_idx: int):
    """Local energy interpolant of E-E_GM. Disk support, no occupancy."""
    z = np.clip(energy - float(energy.min()), 0.0, None)
    key = np.round(xy, 8)
    _, uniq = np.unique(key, axis=0, return_index=True)
    must = [i for i in (gm_idx, ico_idx) if i not in uniq]
    if must:
        uniq = np.unique(np.concatenate([uniq, np.asarray(must, dtype=int)]))
    # keep every low-energy unique site plus farthest cover
    order = uniq[np.argsort(z[uniq])]
    low = order[:40]
    uxy = xy[uniq]
    loc = {int(u): k for k, u in enumerate(uniq)}
    seed = [loc[i] for i in (gm_idx, ico_idx) if i in loc] or [0]
    umap = farthest_indices(uxy, min(N_INDUCING, len(uniq)), seed)
    idx = np.unique(np.concatenate([uniq[umap], low, np.asarray([gm_idx, ico_idx])]))
    obs = xy[idx]
    z_obs = np.clip(z[idx], 0.0, EMAX)
    span = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-6))
    # IDW power 4: isolated wells stay at their pinned E, do not average
    # against the high-E cloud.
    power = 4.0
    xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
    ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    pad = 0.12
    gx = np.linspace(xmin - pad * dx, xmax + pad * dx, 180)
    gy = np.linspace(ymin - pad * dy, ymax + pad * dy, 180)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    pred = np.empty(len(grid))
    nearest_z = np.empty(len(grid))
    dmin = np.full(len(grid), np.inf)
    for i0 in range(0, len(grid), 3000):
        sl = grid[i0 : i0 + 3000]
        d2 = ((sl[:, None, :] - obs[None, :, :]) ** 2).sum(-1)
        nn = d2.argmin(1)
        nearest_z[i0 : i0 + len(sl)] = z_obs[nn]
        dmin[i0 : i0 + len(sl)] = d2.min(1)
        w = 1.0 / np.clip(d2, 1e-16, None) ** (0.5 * power)
        val = (w * z_obs).sum(1) / w.sum(1)
        hit = d2.min(1) < 1e-16
        val[hit] = z_obs[nn[hit]]
        pred[i0 : i0 + len(sl)] = val
    pred = pred.reshape(xx.shape)
    rad = 0.07 * span
    mask = (dmin <= rad * rad).reshape(xx.shape)
    zg = np.where(mask, np.clip(pred, 0.0, EMAX), np.nan)
    # values at the two well sites (exact: they are inducing)
    at_gm = float(z[gm_idx])
    at_ico = float(z[ico_idx])
    return gx, gy, zg, xx, yy, at_gm, at_ico, power, len(idx)


def mark(ax, gm, ico):
    h1 = ax.scatter(
        gm[0],
        gm[1],
        s=130,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=50,
        label=rf"GM ${GM_E:.3f}$",
    )
    h2 = ax.scatter(
        ico[0],
        ico[1],
        s=75,
        marker="D",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=50,
        label=rf"ico ${ICO_E:.3f}$",
    )
    ax.legend(
        handles=[h1, h2],
        loc="upper right",
        fontsize=8,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="k",
    )


def paint_energy(ax, xx, yy, zg):
    mesh = ax.contourf(
        xx, yy, zg, levels=np.linspace(0, EMAX, 21), cmap=PES, extend="max"
    )
    finite = np.where(np.isfinite(zg), zg, np.nan)
    if np.isfinite(finite).any():
        ax.contour(
            xx,
            yy,
            finite,
            levels=np.linspace(0.3, EMAX - 0.3, 10),
            colors="black",
            linewidths=0.3,
            zorder=15,
        )
    return mesh


def write_xy(path: Path, xy: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, xy, fmt="%.8e")


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    energy, xyz = load_min(MIN_PATH)
    n = len(energy)
    gm_idx = int(np.argmin(energy))
    ico_idx = int(np.argmin(np.abs(energy - ICO_E)))
    if abs(float(energy[gm_idx]) - GM_E) > 1e-3:
        raise SystemExit("GM energy %s is not %s" % (energy[gm_idx], GM_E))
    if abs(float(energy[ico_idx]) - ICO_E) > 1e-3:
        raise SystemExit("ico energy %s is not %s" % (energy[ico_idx], ICO_E))

    gm_xyz = xyz[gm_idx]
    ico_xyz = xyz[ico_idx]
    fcc_path = XYZ_DIR / "lj38_fcc.xyz"
    ico_path = XYZ_DIR / "lj38_ico.xyz"
    if fcc_path.is_file():
        proto_gm = load_xyz(fcc_path)
        if hungarian_rmsd(gm_xyz, proto_gm, n_rand=2, rng=rng) < 1e-4:
            gm_xyz = proto_gm
    if ico_path.is_file():
        proto_ico = load_xyz(ico_path)
        if hungarian_rmsd(ico_xyz, proto_ico, n_rand=2, rng=rng) < 1e-4:
            ico_xyz = proto_ico

    book = load_hist(HIST)
    assign = assignment_path()
    print(
        "n",
        n,
        "GM idx",
        gm_idx,
        "E",
        float(energy[gm_idx]),
        "ico idx",
        ico_idx,
        "E",
        float(energy[ico_idx]),
        "families",
        None if book is None else len(book["energy"]),
        "stored_assign",
        None if assign is None else str(assign),
        "scipy",
        _scipy_lsa is not None,
    )

    plane_path = DEST / "d_gm_ico.xy"
    land_path = DEST / "landmarks.idx"
    dll_path = DEST / "D_ll.dist"
    cached = (
        plane_path.is_file()
        and land_path.is_file()
        and dll_path.is_file()
        and np.loadtxt(plane_path).shape[0] == n
    )
    if cached:
        plane = np.loadtxt(plane_path)
        d_gm, d_ico = plane[:, 0], plane[:, 1]
        n_fp_gm = int(np.sum(d_gm < 1e-12))
        n_fp_ico = int(np.sum(d_ico < 1e-12))
        d_cross = float(plane[gm_idx, 1])
        print("cache", plane_path)
    else:
        fp_gm = pairwise_fp(gm_xyz)
        fp_ico = pairwise_fp(ico_xyz)
        fps = np.stack([pairwise_fp(xyz[i]) for i in range(n)])
        d_gm = np.empty(n)
        d_ico = np.empty(n)
        n_fp_gm = n_fp_ico = 0
        for i in range(n):
            fp = fps[i]
            if np.max(np.abs(fp - fp_gm)) < FP_TOL:
                d_gm[i] = 0.0
                n_fp_gm += 1
            else:
                d_gm[i] = hungarian_rmsd(
                    xyz[i], gm_xyz, n_rand=2, rng=rng, fp_p=fp, fp_q=fp_gm
                )
            if np.max(np.abs(fp - fp_ico)) < FP_TOL:
                d_ico[i] = 0.0
                n_fp_ico += 1
            else:
                d_ico[i] = hungarian_rmsd(
                    xyz[i], ico_xyz, n_rand=2, rng=rng, fp_p=fp, fp_q=fp_ico
                )
            if (i + 1) % 400 == 0 or i + 1 == n:
                print("  d_GM/d_ico", i + 1, "/", n, flush=True)
        plane = np.column_stack([d_gm, d_ico])
        d_cross = float(hungarian_rmsd(gm_xyz, ico_xyz, n_rand=8, rng=rng))
    print(
        "fp GM",
        n_fp_gm,
        "fp ico",
        n_fp_ico,
        "D(GM,ico)",
        d_cross,
        "d_GM[GM]",
        float(d_gm[gm_idx]),
        "d_ico[ico]",
        float(d_ico[ico_idx]),
        "d_ico[GM]",
        float(d_ico[gm_idx]),
        "d_GM[ico]",
        float(d_gm[ico_idx]),
    )

    if cached:
        land = np.loadtxt(land_path, dtype=int)
        D_ll = np.loadtxt(dll_path)
        print("cache landmarks", len(land))
    else:
        land = landmark_indices(energy, gm_idx, ico_idx, rng)
        print("landmarks", len(land), "low100+rand100+basins")
        D_ll = pairwise_hungarian(xyz, land, rng)
    mds_land, ev = torgerson(D_ll, 2)
    # put GM left-ish, ico right-ish
    li_gm = int(np.where(land == gm_idx)[0][0])
    li_ico = int(np.where(land == ico_idx)[0][0])
    if mds_land[li_gm, 0] > mds_land[li_ico, 0]:
        mds_land[:, 0] *= -1.0
    nyst = nystrom_idw(plane, land, mds_land, k=8)
    nyst[land] = mds_land

    gx, gy, zg, xx, yy, gp_gm, gp_ico, smooth, n_ind = energy_grid(
        plane, energy, gm_idx, ico_idx
    )
    well_gm = 0.0
    well_ico = float(energy[ico_idx] - energy[gm_idx])
    print(
        "wells raw  GM",
        well_gm,
        "ico",
        well_ico,
        "GP at GM",
        gp_gm,
        "GP at ico",
        gp_ico,
        "inducing",
        n_ind,
        "smooth",
        smooth,
    )

    DEST.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    write_xy(DEST / "d_gm_ico.xy", plane)
    write_xy(DEST / "mds.xy", nyst)
    write_xy(DEST / "nystrom.xy", nyst)
    np.savetxt(DEST / "energy.txt", energy, fmt="%.10e")
    np.savetxt(DEST / "landmarks.idx", land, fmt="%d")
    np.savetxt(DEST / "D_ll.dist", D_ll, fmt="%.8e")
    np.savetxt(DEST / "d_gm.txt", d_gm, fmt="%.8e")
    np.savetxt(DEST / "d_ico.txt", d_ico, fmt="%.8e")

    fig, ax = plt.subplots(figsize=(5.8, 5.0), facecolor="white")
    mesh = paint_energy(ax, xx, yy, zg)
    mark(ax, plane[gm_idx], plane[ico_idx])
    ax.set_xlabel(r"$d_{\mathrm{GM}}$  (Hungarian RMSD)")
    ax.set_ylabel(r"$d_{\mathrm{ico}}$  (Hungarian RMSD)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(r"LJ38 Hungarian plane  $E-E_{\mathrm{GM}}$")
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    fig.tight_layout()
    fig.savefig(FIG, dpi=170, facecolor="white")
    fig.savefig(DEST / "elja_occ_lj38_hungarian.png", dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", FIG)

    # landmark MDS energy fill as a companion in DEST only
    gx2, gy2, zg2, xx2, yy2, gp2_gm, gp2_ico, _, _ = energy_grid(
        nyst, energy, gm_idx, ico_idx
    )
    fig, ax = plt.subplots(figsize=(5.8, 5.0), facecolor="white")
    mesh = paint_energy(ax, xx2, yy2, zg2)
    mark(ax, nyst[gm_idx], nyst[ico_idx])
    ax.set_xlabel(r"$s_1$  (landmark MDS)")
    ax.set_ylabel(r"$s_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(r"landmark Hungarian MDS  $E-E_{\mathrm{GM}}$")
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    fig.tight_layout()
    fig.savefig(DEST / "elja_occ_lj38_hungarian_mds.png", dpi=170, facecolor="white")
    plt.close(fig)

    scores = {
        "n": int(n),
        "n_atoms": N_ATOMS,
        "gm_idx": gm_idx,
        "ico_idx": ico_idx,
        "E_GM": float(energy[gm_idx]),
        "E_ico": float(energy[ico_idx]),
        "well_depth_GM": well_gm,
        "well_depth_ico": well_ico,
        "gp_at_GM": gp_gm,
        "gp_at_ico": gp_ico,
        "gp_mds_at_GM": gp2_gm,
        "gp_mds_at_ico": gp2_ico,
        "D_GM_ico": d_cross,
        "d_GM": {
            "gm": float(d_gm[gm_idx]),
            "ico": float(d_gm[ico_idx]),
            "min": float(d_gm.min()),
            "max": float(d_gm.max()),
        },
        "d_ico": {
            "gm": float(d_ico[gm_idx]),
            "ico": float(d_ico[ico_idx]),
            "min": float(d_ico.min()),
            "max": float(d_ico.max()),
        },
        "n_fingerprint_GM": int(n_fp_gm),
        "n_fingerprint_ico": int(n_fp_ico),
        "n_families": None if book is None else int(len(book["energy"])),
        "stored_assignment": None if assign is None else str(assign),
        "n_landmarks": int(len(land)),
        "mds_eigs": [float(x) for x in ev],
        "figure": str(FIG),
        "plane": "hungarian RMSD (d_GM, d_ico)",
        "energy_field": "IDW-4 of E-E_GM on Hungarian (d_GM, d_ico), no occupancy",
    }
    (DEST / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    print("wrote", DEST / "scores.json")
    print(
        "well depths  GM",
        well_gm,
        "ico",
        well_ico,
        "GP",
        gp_gm,
        gp_ico,
    )


if __name__ == "__main__":
    main()
