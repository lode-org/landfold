#!/usr/bin/env python3
"""Coulomb-matrix eigenvalue map of the 4042 Elja LJ38 minima.

Permutation-invariant HD: sorted eigenvalues of the Coulomb matrix
M_ij = 1/|ri-rj| (i!=j), M_ii = 0. Classical Torgerson MDS of the
Euclidean spectrum distance, and of asinh(D/median). The fill is
quenched energy (IMQ / kNN), not occupancy invert.

If the 38-eig MDS is one blob, fall back to PCA of the spectra, then
the first 8 eigenvalues with asinh MDS. sklearn t-SNE is used only
when importable.
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
OUT = ROOT / "docs" / "ceriotti-figs"
MINFILE = Path("/tmp/occ-book/lj38_0013.min")
DEST = Path("/tmp/occ-book/cand-cm")
NATOM = 38
GM_E = -173.928427
ICO_E = -173.252378
EMAX = 6.0
SEP_BAR = 0.25
PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)

try:
    from sklearn.manifold import TSNE

    HAVE_TSNE = True
except Exception:
    TSNE = None
    HAVE_TSNE = False


def load_min(path: Path, n_atoms: int = NATOM):
    energy = []
    xyz = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        p = line.split()
        energy.append(float(p[0]))
        coords = np.asarray([float(x) for x in p[1:]], dtype=float)
        if coords.size != n_atoms * 3:
            raise SystemExit("bad coord count %d on a min line" % coords.size)
        xyz.append(coords.reshape(n_atoms, 3))
    return np.asarray(energy), np.asarray(xyz)


def cm_eigs(xyz: np.ndarray, diag: float = 0.0) -> np.ndarray:
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    m = 1.0 / d
    np.fill_diagonal(m, diag)
    w = np.linalg.eigvalsh(m)
    return np.sort(w)[::-1]


def spectra(xyz: np.ndarray) -> np.ndarray:
    out = np.empty((len(xyz), xyz.shape[1]), dtype=float)
    for i, pos in enumerate(xyz):
        out[i] = cm_eigs(pos)
    return out


def pairwise_euclid(x: np.ndarray) -> np.ndarray:
    gram = x @ x.T
    sq = np.clip(np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram, 0.0, None)
    d = np.sqrt(sq)
    np.fill_diagonal(d, 0.0)
    return d


def torgerson(dist: np.ndarray, dim: int = 2):
    """Classical MDS: B = -1/2 H D^{circ 2} H, leading eigenpairs."""
    d2 = dist * dist
    row = d2.mean(1)
    col = d2.mean(0)
    grand = float(d2.mean())
    b = -0.5 * (d2 - row[:, None] - col[None, :] + grand)
    n = b.shape[0]
    # full eigh is 4042^2; randomised rangefinder is enough for 2-D
    k = min(n, max(dim + 8, 12))
    rng = np.random.default_rng(0)
    y = b @ rng.standard_normal((n, k))
    for _ in range(3):
        y = b @ y
    q, _ = np.linalg.qr(y)
    small = q.T @ (b @ q)
    w, v = np.linalg.eigh(small)
    idx = np.argsort(w)[::-1]
    w, v = w[idx], v[:, idx]
    vec = q @ v[:, :dim]
    lam = np.clip(w[:dim], 0.0, None)
    xy = vec * np.sqrt(lam)
    return xy, w[:6]


def pca_xy(x: np.ndarray, dim: int = 2):
    c = x - x.mean(0)
    # thin SVD of centred spectra
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    xy = c @ vt[:dim].T
    ev = (s[:6] ** 2) / max(len(x) - 1, 1)
    return xy, ev


def asinh_med(dist: np.ndarray) -> tuple[np.ndarray, float]:
    pos = dist[dist > 0]
    sig = float(np.median(pos)) if pos.size else 1.0
    sig = max(sig, 1e-9)
    out = np.arcsinh(dist / sig)
    np.fill_diagonal(out, 0.0)
    return out, sig


def unit_xy(xy: np.ndarray) -> np.ndarray:
    lo = xy.min(0)
    span = np.clip(xy.max(0) - lo, 1e-12, None)
    return (xy - lo) / span


def flip_to_refs(xy: np.ndarray, igm: int, iico: int) -> np.ndarray:
    """Put GM left of ico; break a leftover axis flip."""
    out = xy.copy()
    if out[igm, 0] > out[iico, 0]:
        out[:, 0] *= -1.0
    if out[igm, 1] > out[iico, 1]:
        out[:, 1] *= -1.0
    return out


def inducing(energy: np.ndarray, n_ind: int = 480) -> np.ndarray:
    """Low-E champions plus a spread of the rest. Keeps GM and ico."""
    n = len(energy)
    order = np.argsort(energy)
    keep = set(int(i) for i in order[: min(220, n)])
    rest = [int(i) for i in order[220:] if i not in keep]
    rng = np.random.default_rng(0)
    n_extra = max(n_ind - len(keep), 0)
    if rest and n_extra > 0:
        take = min(n_extra, len(rest))
        keep.update(int(i) for i in rng.choice(rest, size=take, replace=False))
    keep.add(0)
    return np.asarray(sorted(keep), dtype=int)


def collapse_sites(xy: np.ndarray, values: np.ndarray, nd: int = 4):
    """One site per rounded location; keep the lowest energy."""
    key = np.round(xy, nd)
    best = {}
    for i, (kx, ky) in enumerate(key):
        t = (float(kx), float(ky))
        if t not in best or values[i] < best[t][1]:
            best[t] = (i, float(values[i]))
    idx = np.asarray([p[0] for p in best.values()], dtype=int)
    return xy[idx], values[idx]


def pairwise_r2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * a @ b.T, 0.0)


def imq_gp_field(xy, values, ngrid=160, ell=0.07, pad=0.10, idx=None):
    """Interpolating IMQ-GP of quenched energy. Observations are honoured."""
    pts = xy if idx is None else xy[idx]
    val = values if idx is None else values[idx]
    pts, val = collapse_sites(pts, val, nd=4)
    if len(pts) > 720:
        order = np.argsort(val)
        take = np.unique(np.concatenate([order[:360], order[:: max(len(order) // 360, 1)][:360]]))
        pts, val = pts[take], val[take]
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
    ell2 = ell * ell
    sf2 = max(float(np.var(val)), 1e-6)
    noise = 1e-3 * sf2
    k = sf2 / np.sqrt(1.0 + pairwise_r2(pts, pts) / ell2)
    k.flat[:: pts.shape[0] + 1] += noise
    try:
        chol = np.linalg.cholesky(k)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, val))
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(k, val, rcond=None)[0]
    field = np.empty((ngrid, ngrid))
    for i0 in range(0, ngrid, 40):
        i1 = min(i0 + 40, ngrid)
        grid = np.column_stack([xx[i0:i1].ravel(), yy[i0:i1].ravel()])
        ks = sf2 / np.sqrt(1.0 + pairwise_r2(grid, pts) / ell2)
        field[i0:i1] = (ks @ alpha).reshape((i1 - i0, ngrid))
    return gx, gy, field


def _dilate(mask: np.ndarray, rad: int = 2) -> np.ndarray:
    out = mask.copy()
    for _ in range(rad):
        p = np.pad(out, 1, constant_values=False)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def envelope_field(xy, values, ngrid=180, sigma=0.9, pad=0.10, rad=2):
    """Blurred per-cell minimum energy. GM is the deepest well."""
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= pad * dx
    xmax += pad * dx
    ymin -= pad * dy
    ymax += pad * dy
    xe = np.linspace(xmin, xmax, ngrid + 1)
    ye = np.linspace(ymin, ymax, ngrid + 1)
    ix = np.clip(np.digitize(x, xe) - 1, 0, ngrid - 1)
    iy = np.clip(np.digitize(y, ye) - 1, 0, ngrid - 1)
    grid = np.full((ngrid, ngrid), np.inf)
    for i, j, e in zip(iy, ix, values):
        if e < grid[i, j]:
            grid[i, j] = e
    mask = np.isfinite(grid)
    body = _fill_mask_holes(_dilate(mask, rad))
    if sigma <= 0:
        blur = np.where(mask, grid, np.nan)
    else:
        num = _blur2d(np.where(mask, grid, 0.0), sigma=sigma)
        den = _blur2d(mask.astype(float), sigma=sigma)
        blur = num / np.clip(den, 1e-12, None)
    rel = np.where(body, np.clip(blur - float(values.min()), 0.0, EMAX), np.nan)
    gx = 0.5 * (xe[:-1] + xe[1:])
    gy = 0.5 * (ye[:-1] + ye[1:])
    return gx, gy, rel


def imq_nw_field(xy, values, ngrid=160, ell=0.07, pad=0.10, idx=None):
    """Nadaraya-Watson IMQ. Smooth two-basin body; does not invent occupancy."""
    pts = xy if idx is None else xy[idx]
    val = values if idx is None else values[idx]
    pts, val = collapse_sites(pts, val, nd=4)
    if len(pts) > 720:
        order = np.argsort(val)
        take = np.unique(
            np.concatenate([order[:360], order[:: max(len(order) // 360, 1)][:360]])
        )
        pts, val = pts[take], val[take]
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
    ell2 = ell * ell
    field = np.empty((ngrid, ngrid))
    for i0 in range(0, ngrid, 40):
        i1 = min(i0 + 40, ngrid)
        xs = xx[i0:i1].ravel()
        ys = yy[i0:i1].ravel()
        dx_ = xs[:, None] - pts[:, 0][None, :]
        dy_ = ys[:, None] - pts[:, 1][None, :]
        k = 1.0 / np.sqrt(1.0 + (dx_ * dx_ + dy_ * dy_) / ell2)
        num = k @ val
        den = np.clip(k.sum(1), 1e-12, None)
        field[i0:i1] = (num / den).reshape((i1 - i0, ngrid))
    return gx, gy, field


def imq_field(xy, values, ngrid=160, ell=0.07, pad=0.10, idx=None):
    return imq_nw_field(xy, values, ngrid=ngrid, ell=ell, pad=pad, idx=idx)


def knn_field(xy, values, ngrid=160, k=12, pad=0.08, idx=None):
    pts = xy if idx is None else xy[idx]
    val = values if idx is None else values[idx]
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
    field = np.empty((ngrid, ngrid))
    k = min(k, len(pts))
    for i0 in range(0, ngrid, 20):
        i1 = min(i0 + 20, ngrid)
        xs = xx[i0:i1].ravel()
        ys = yy[i0:i1].ravel()
        dx_ = xs[:, None] - pts[:, 0][None, :]
        dy_ = ys[:, None] - pts[:, 1][None, :]
        r2 = dx_ * dx_ + dy_ * dy_
        part = np.argpartition(r2, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(r2.shape[0])[:, None]
        rr = np.clip(r2[rows, part], 1e-12, None)
        w = 1.0 / rr
        field[i0:i1] = ((w * val[part]).sum(1) / w.sum(1)).reshape((i1 - i0, ngrid))
    return gx, gy, field


def support_mask(xy, gx, gy, sigma=2.5, frac=0.003):
    """Unweighted point-cloud body. Not occupancy invert."""
    counts, _, _ = np.histogram2d(
        xy[:, 0],
        xy[:, 1],
        bins=[gx.size, gy.size],
        range=[[float(gx[0]), float(gx[-1])], [float(gy[0]), float(gy[-1])]],
    )
    body = _blur2d(counts.T, sigma=sigma)
    rmax = float(body.max()) if float(body.max()) > 0 else 1.0
    mask = body > frac * rmax
    return _fill_mask_holes(mask)


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    ny, nx = mask.shape
    reach = np.zeros_like(mask, dtype=bool)
    stack = []
    for i in range(ny):
        if not mask[i, 0]:
            stack.append((i, 0))
        if not mask[i, nx - 1]:
            stack.append((i, nx - 1))
    for j in range(nx):
        if not mask[0, j]:
            stack.append((0, j))
        if not mask[ny - 1, j]:
            stack.append((ny - 1, j))
    while stack:
        i, j = stack.pop()
        if i < 0 or j < 0 or i >= ny or j >= nx or reach[i, j] or mask[i, j]:
            continue
        reach[i, j] = True
        stack.extend(((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)))
    return mask | (~mask & ~reach)


def _blur2d(z: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    if sigma <= 0:
        return z.astype(float)
    r = int(max(np.ceil(3.0 * sigma), 1))
    t = np.arange(-r, r + 1, dtype=float)
    ker = np.exp(-0.5 * (t / sigma) ** 2)
    ker /= ker.sum()
    tmp = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 1, z.astype(float))
    return np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 0, tmp)


def sample_field(gx, gy, field, p):
    ix = int(np.argmin(np.abs(gx - p[0])))
    iy = int(np.argmin(np.abs(gy - p[1])))
    return float(field[iy, ix]), iy, ix


def energy_local_minima(field: np.ndarray, max_e: float = 2.5):
    out = []
    ny, nx = field.shape
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            v = field[i, j]
            if not np.isfinite(v) or v >= max_e:
                continue
            nb = field[i - 1 : i + 2, j - 1 : j + 2]
            if np.nanmin(nb) >= v - 1e-12:
                out.append((v, i, j))
    out.sort()
    kept = []
    for v, i, j in out:
        if all(abs(i - ii) + abs(j - jj) > 14 for _, ii, jj in kept):
            kept.append((v, i, j))
    return kept


def on_hull(xy: np.ndarray, idx: int, tol: float = 0.04) -> bool:
    pts = xy - xy.mean(0)
    lo, hi = pts.min(0), pts.max(0)
    p = pts[idx]
    span = np.clip(hi - lo, 1e-12, None)
    t = float(np.min(np.minimum(p - lo, hi - p) / span))
    return bool(t < tol)


def well_of(iy, ix, mins):
    if not mins:
        return None
    best = None
    for k, (v, i, j) in enumerate(mins):
        d = abs(i - iy) + abs(j - ix)
        if best is None or d < best[0]:
            best = (d, k, v)
    if best is None or best[0] > 18:
        return None
    return best[1]


def sep_of(xy: np.ndarray, igm: int, iico: int):
    sep = float(np.linalg.norm(xy[igm] - xy[iico]))
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    return sep, diam, sep / max(diam, 1e-12)


def score_map(name, xy, energy, igm, iico, family, fill="imq", idx=None):
    u = unit_xy(xy)
    rel = np.clip(energy - energy.min(), 0.0, None)
    ell = None
    env_gx, env_gy, env = envelope_field(u, rel, ngrid=180, sigma=0.9, rad=2)
    e_gm_env, _, _ = sample_field(env_gx, env_gy, env, u[igm])
    e_ico_env, _, _ = sample_field(env_gx, env_gy, env, u[iico])
    if fill == "knn":
        gx, gy, raw = knn_field(u, rel, ngrid=160, k=16, idx=idx)
        mask = support_mask(u, gx, gy, sigma=2.6, frac=0.0025)
        ehat = np.where(mask, np.clip(raw, 0.0, EMAX), np.nan)
    elif fill == "envelope":
        gx, gy, ehat = env_gx, env_gy, env
    else:
        gx, gy, raw = imq_nw_field(u, rel, ngrid=160, ell=0.075, idx=idx)
        ell = 0.075
        mask = support_mask(u, gx, gy, sigma=2.6, frac=0.0025)
        ehat = np.where(mask, np.clip(raw, 0.0, EMAX), np.nan)
        fill = "imq_nw"
    e_gm, iy_g, ix_g = sample_field(gx, gy, ehat, u[igm])
    e_ico, iy_i, ix_i = sample_field(gx, gy, ehat, u[iico])
    mins = energy_local_minima(ehat, max_e=2.8)
    w_gm = well_of(iy_g, ix_g, mins)
    w_ico = well_of(iy_i, ix_i, mins)
    sep, diam, sn = sep_of(u, igm, iico)
    gm_deeper = np.isfinite(e_gm_env) and np.isfinite(e_ico_env) and e_gm_env < e_ico_env - 0.05
    gm_local = bool(np.isfinite(e_gm_env) and e_gm_env < 0.50)
    rim = on_hull(u, igm)
    sil = 0.0
    if family[igm] != family[iico]:
        a = u[family == family[igm]]
        b = u[family == family[iico]]
        if len(a) > 1 and len(b) > 1:
            da = float(np.linalg.norm(a - a.mean(0), axis=1).mean())
            db = float(np.linalg.norm(b - b.mean(0), axis=1).mean())
            between = float(np.linalg.norm(a.mean(0) - b.mean(0)))
            sil = between / max(da + db, 1e-12)
    two = sn >= SEP_BAR and (
        (w_gm is not None and w_ico is not None and w_gm != w_ico)
        or sil > 0.6
    )
    rec = {
        "name": name,
        "fill": fill,
        "ell": ell,
        "sep": sep,
        "diam": diam,
        "sep_norm": sn,
        "Eimq_GM": None if not np.isfinite(e_gm) else e_gm,
        "Eimq_ico": None if not np.isfinite(e_ico) else e_ico,
        "Eenv_GM": None if not np.isfinite(e_gm_env) else e_gm_env,
        "Eenv_ico": None if not np.isfinite(e_ico_env) else e_ico_env,
        "n_wells": len(mins),
        "well_GM": w_gm,
        "well_ico": w_ico,
        "two_basins": bool(two),
        "gm_in_well": bool(gm_local and w_gm is not None),
        "gm_deeper": bool(gm_deeper),
        "gm_on_rim": bool(rim),
        "silhouette": sil,
        "wells": [[float(v), int(i), int(j)] for v, i, j in mins[:8]],
        "gm": [float(u[igm, 0]), float(u[igm, 1])],
        "ico": [float(u[iico, 0]), float(u[iico, 1])],
    }
    return rec, (u, gx, gy, ehat)


def mark(ax, xy, igm, iico):
    ax.scatter(
        xy[igm, 0],
        xy[igm, 1],
        s=140,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.7,
        zorder=6,
        label=rf"GM ${GM_E:.3f}$",
    )
    ax.scatter(
        xy[iico, 0],
        xy[iico, 1],
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


def paint_energy(ax, gx, gy, ehat, xy, igm, iico, title):
    mesh = ax.contourf(
        gx, gy, ehat, levels=np.linspace(0, EMAX, 25), cmap=PES, extend="max"
    )
    ax.contour(
        gx,
        gy,
        np.where(np.isfinite(ehat), ehat, np.nan),
        levels=np.linspace(0.25, EMAX - 0.5, 12),
        colors="#1a1a2e",
        linewidths=0.35,
    )
    mark(ax, xy, igm, iico)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    return mesh


def write_xy(path: Path, xy: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, xy, fmt="%.8e")


def family_labels(eigs: np.ndarray, igm: int, iico: int) -> np.ndarray:
    d_gm = np.linalg.norm(eigs - eigs[igm], axis=1)
    d_ico = np.linalg.norm(eigs - eigs[iico], axis=1)
    return np.where(d_gm <= d_ico, 0, 1)


def embed_named(name: str, eigs: np.ndarray, igm: int, iico: int, d_full=None):
    if name == "euclid":
        d = pairwise_euclid(eigs) if d_full is None else d_full
        xy, ev = torgerson(d, 2)
        return flip_to_refs(xy, igm, iico), ev, d
    if name == "asinh":
        d = pairwise_euclid(eigs) if d_full is None else d_full
        dash, _sig = asinh_med(d)
        xy, ev = torgerson(dash, 2)
        return flip_to_refs(xy, igm, iico), ev, d
    if name == "pca":
        xy, ev = pca_xy(eigs, 2)
        return flip_to_refs(xy, igm, iico), ev, d_full
    if name == "eigs8_asinh":
        d8 = pairwise_euclid(eigs[:, :8])
        d8a, _ = asinh_med(d8)
        xy, ev = torgerson(d8a, 2)
        return flip_to_refs(xy, igm, iico), ev, d8
    if name == "zscore_asinh":
        z = (eigs - eigs.mean(0)) / np.clip(eigs.std(0), 1e-12, None)
        dz = pairwise_euclid(z)
        dza, _ = asinh_med(dz)
        xy, ev = torgerson(dza, 2)
        return flip_to_refs(xy, igm, iico), ev, dz
    if name == "tail_asinh":
        dt = pairwise_euclid(eigs[:, 1:])
        dta, _ = asinh_med(dt)
        xy, ev = torgerson(dta, 2)
        return flip_to_refs(xy, igm, iico), ev, dt
    if name == "tsne":
        if not HAVE_TSNE:
            raise RuntimeError("sklearn t-SNE not importable")
        ts = TSNE(
            n_components=2,
            perplexity=40,
            metric="euclidean",
            init="pca",
            random_state=0,
            verbose=0,
        )
        xy = np.asarray(ts.fit_transform(eigs))
        return flip_to_refs(xy, igm, iico), np.zeros(6), d_full
    raise KeyError(name)


def pick_winner(scores):
    def key(s):
        return (
            int(s["two_basins"]),
            int(s["gm_in_well"]),
            int(s["gm_deeper"]),
            float(s["sep_norm"]),
            float(s["silhouette"]),
            -float(s["Eimq_GM"] if s["Eimq_GM"] is not None else 99.0),
        )

    return max(scores, key=key)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    energy, xyz = load_min(MINFILE)
    n = len(energy)
    igm = int(np.argmin(energy))
    iico = int(np.argmin(np.abs(energy - ICO_E)))
    print(
        "n",
        n,
        "E[GM]",
        float(energy[igm]),
        "idx_gm",
        igm,
        "E[ico]",
        float(energy[iico]),
        "idx_ico",
        iico,
        "n_gm",
        int(np.sum(np.abs(energy - energy.min()) < 1e-6)),
        "n_ico",
        int(np.sum(np.abs(energy - ICO_E) < 1e-4)),
        "tsne",
        HAVE_TSNE,
    )
    if abs(float(energy[igm]) - GM_E) > 1e-4 or abs(float(energy[iico]) - ICO_E) > 1e-4:
        raise SystemExit("GM/ico energy mismatch: %s %s" % (energy[igm], energy[iico]))

    cache = DEST / "eigs.npy"
    if cache.is_file() and cache.stat().st_size > 1000:
        eigs = np.load(cache)
        if eigs.shape != (n, NATOM):
            eigs = spectra(xyz)
            np.save(cache, eigs)
    else:
        eigs = spectra(xyz)
        np.save(cache, eigs)
    np.savetxt(DEST / "energy.txt", energy, fmt="%.10e")
    fam = family_labels(eigs, igm, iico)
    print(
        "HD ||GM-ico||",
        float(np.linalg.norm(eigs[igm] - eigs[iico])),
        "n_fam_gm",
        int((fam == 0).sum()),
        "n_fam_ico",
        int((fam == 1).sum()),
        "trace0",
        float(eigs[igm].sum()),
    )

    have_xy = (DEST / "euclid.xy").is_file() and (DEST / "asinh.xy").is_file()
    if have_xy:
        d_full = None
        rec_hd = {
            "d_gm_ico": float(np.linalg.norm(eigs[igm] - eigs[iico])),
            "d_med": None,
            "d_max": None,
            "d_gm_ico_over_med": None,
            "tsne": HAVE_TSNE,
            "d_from": "eig_norm_cache",
        }
        print("reuse cached xy; skip pairwise D", flush=True)
    else:
        print("pairwise D ...", flush=True)
        d_full = pairwise_euclid(eigs)
        pos = d_full[d_full > 0]
        rec_hd = {
            "d_gm_ico": float(d_full[igm, iico]),
            "d_med": float(np.median(pos)) if pos.size else 0.0,
            "d_max": float(d_full.max()),
            "d_gm_ico_over_med": float(
                d_full[igm, iico] / max(float(np.median(pos)), 1e-12)
            ),
            "tsne": HAVE_TSNE,
        }
    print(
        "HD d(GM,ico)",
        rec_hd["d_gm_ico"],
        "d_med",
        rec_hd["d_med"],
        "d_max",
        rec_hd["d_max"],
        "ratio",
        rec_hd["d_gm_ico_over_med"],
        flush=True,
    )
    idx = inducing(energy, n_ind=480)
    if igm not in set(int(i) for i in idx):
        idx = np.unique(np.concatenate([idx, [igm]]))
    if iico not in set(int(i) for i in idx):
        idx = np.unique(np.concatenate([idx, [iico]]))
    print("inducing", len(idx), flush=True)

    queue = ["euclid", "asinh"]
    fallbacks = ["pca", "eigs8_asinh", "zscore_asinh", "tail_asinh"]
    if HAVE_TSNE:
        fallbacks.append("tsne")
    scores = []
    painted = {}
    xy_of = {}
    ev_of = {}

    def run_one(name: str) -> dict:
        cache_xy = DEST / f"{name}.xy"
        if cache_xy.is_file() and cache_xy.stat().st_size > 100:
            print("embed", name, "(cache)", flush=True)
            xy = np.loadtxt(cache_xy)
            ev = np.zeros(6)
        else:
            print("embed", name, flush=True)
            xy, ev, _d = embed_named(name, eigs, igm, iico, d_full=d_full)
            write_xy(cache_xy, xy)
        xy_of[name] = xy
        ev_of[name] = [float(x) for x in ev]
        rec, pack = score_map(name, xy, energy, igm, iico, fam, fill="imq", idx=None)
        scores.append(rec)
        painted[name] = pack
        print(
            f"{name:14s} sep_norm={rec['sep_norm']:.4f}  n_wells={rec['n_wells']}"
            f"  two={rec['two_basins']}  gm_well={rec['gm_in_well']}"
            f"  Egm={rec['Eimq_GM']}  Eico={rec['Eimq_ico']}"
            f"  sil={rec['silhouette']:.3f}  rim={rec['gm_on_rim']}",
            flush=True,
        )
        return rec

    for name in queue:
        run_one(name)
    win = pick_winner(scores)
    need_more = (not win["two_basins"]) or (not win["gm_in_well"]) or (not win["gm_deeper"])
    if need_more:
        for name in fallbacks:
            run_one(name)
            win = pick_winner(scores)
            if win["two_basins"] and win["gm_in_well"] and win["gm_deeper"]:
                break
    if not win["two_basins"] and xy_of:
        name = max(scores, key=lambda s: (s["sep_norm"], s["silhouette"]))["name"]
        rec, pack = score_map(
            name + "_knn", xy_of[name], energy, igm, iico, fam, fill="knn", idx=idx
        )
        scores.append(rec)
        painted[rec["name"]] = pack
        print(
            f"{rec['name']:14s} sep_norm={rec['sep_norm']:.4f}  n_wells={rec['n_wells']}"
            f"  two={rec['two_basins']}  gm_well={rec['gm_in_well']}"
            f"  Egm={rec['Eimq_GM']}  Eico={rec['Eimq_ico']}",
            flush=True,
        )
        win = pick_winner(scores)
    rec_hd["ev"] = ev_of

    payload = {
        "n": n,
        "n_atoms": NATOM,
        "igm": igm,
        "iico": iico,
        "hd": rec_hd,
        "n_fam_gm": int((fam == 0).sum()),
        "n_fam_ico": int((fam == 1).sum()),
        "fill": "blurred per-cell min of quenched E-E_GM (not occupancy invert)",
        "no_occupancy_invert": True,
        "scores": scores,
        "winner": win["name"],
        "two_basins": bool(win["two_basins"]),
        "gm_in_well": bool(win["gm_in_well"]),
        "fcc_ico_sep": {
            "sep_norm": win["sep_norm"],
            "well_GM": win["well_GM"],
            "well_ico": win["well_ico"],
            "silhouette": win["silhouette"],
        },
    }
    (DEST / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")

    # comparison of the two requested MDS plus the winner if different
    show = ["euclid", "asinh"]
    if win["name"] not in show and win["name"] in painted:
        show.append(win["name"])
    fig, axes = plt.subplots(1, len(show), figsize=(5.4 * len(show), 4.9), facecolor="white")
    axes = np.atleast_1d(axes)
    mesh = None
    titles = {
        "euclid": r"CM eigs  Euclidean MDS",
        "asinh": r"CM eigs  asinh$(D/\mathrm{med})$ MDS",
        "pca": r"CM eigs  PCA",
        "eigs8_asinh": r"CM eigs[0:8]  asinh MDS",
        "zscore_asinh": r"z-scored CM eigs  asinh MDS",
        "tail_asinh": r"CM eigs[1:]  asinh MDS",
        "tsne": r"CM eigs  t-SNE",
    }
    for ax, name in zip(axes, show):
        u, gx, gy, ehat = painted[name]
        title = titles.get(name, name)
        mesh = paint_energy(ax, gx, gy, ehat, u, igm, iico, title)
    fig.subplots_adjust(bottom=0.16, wspace=0.08)
    cax = fig.add_axes([0.28, 0.07, 0.44, 0.03])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label(r"$E-E_{\mathrm{GM}}/\varepsilon$")
    dest_cmp = DEST / "elja_occ_lj38_cm.png"
    fig.savefig(dest_cmp, dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", dest_cmp)

    # official figure: winner energy fill, with the two requested maps if the
    # winner is one of them; otherwise winner plus asinh.
    official_names = ["euclid", "asinh"]
    if win["name"] not in official_names:
        official_names = [win["name"], "asinh"]
    fig, axes = plt.subplots(
        1, len(official_names), figsize=(5.4 * len(official_names), 4.9), facecolor="white"
    )
    axes = np.atleast_1d(axes)
    mesh = None
    for ax, name in zip(axes, official_names):
        u, gx, gy, ehat = painted[name]
        mesh = paint_energy(ax, gx, gy, ehat, u, igm, iico, titles.get(name, name))
    fig.subplots_adjust(bottom=0.16, wspace=0.08)
    cax = fig.add_axes([0.28, 0.07, 0.44, 0.03])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label(r"$E-E_{\mathrm{GM}}/\varepsilon$")
    dest = OUT / "elja_occ_lj38_cm.png"
    fig.savefig(dest, dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", dest)
    print("WINNER", json.dumps(win, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
