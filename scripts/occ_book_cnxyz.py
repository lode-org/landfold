#!/usr/bin/env python3
"""Hard n4..n13 and Steinhardt (Q6, Q4) of the 4042 LJ38 minima.

Recompute coordination histograms from the energy-leading .min dump.
Do not read lj38.cv. Embed asinh W1 of n4..n13 by classical MDS, and
fill the published (Q6, Q4) plane with quenched energy, not occupancy.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ceriotti-figs"
BOOK = Path("/tmp/occ-book")
MINFILE = BOOK / "lj38_0013.min"
CAND = BOOK / "cand-cnxyz"
N_ATOMS = 38
GM_E = -173.928427
ICO_E = -173.252378
RCUTS = (1.20, 1.35)
CN_LO, CN_HI = 4, 13
EMAX = 6.0
PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)
ASINH_NORM = 2.0 * float(np.arcsinh(1.0))


def load_min(path: Path) -> tuple[np.ndarray, np.ndarray]:
    energies = []
    frames = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        nums = [float(x) for x in line.split()]
        if len(nums) != 1 + N_ATOMS * 3:
            raise SystemExit(f"{path}: expected {1 + N_ATOMS * 3} fields, got {len(nums)}")
        energies.append(nums[0])
        frames.append(np.asarray(nums[1:], dtype=np.float64).reshape(N_ATOMS, 3))
    return np.asarray(energies), np.stack(frames, axis=0)


def _ylm4(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Y_4m, m = -4..4. Shape (9, n). Physics / Condon-Shortley."""
    c = np.cos(theta)
    s = np.sin(theta)
    c2 = c * c
    s2 = s * s
    e = np.exp(1j * phi)
    e2 = e * e
    e3 = e2 * e
    e4 = e2 * e2
    y0 = (3.0 / 16.0) * np.sqrt(1.0 / np.pi) * (35.0 * c2 * c2 - 30.0 * c2 + 3.0)
    y1 = -(3.0 / 8.0) * np.sqrt(5.0 / np.pi) * s * c * (7.0 * c2 - 3.0) * e
    y2 = (3.0 / 8.0) * np.sqrt(5.0 / (2.0 * np.pi)) * s2 * (7.0 * c2 - 1.0) * e2
    y3 = -(3.0 / 8.0) * np.sqrt(35.0 / np.pi) * s2 * s * c * e3
    y4 = (3.0 / 16.0) * np.sqrt(35.0 / (2.0 * np.pi)) * s2 * s2 * e4
    return np.stack(
        [np.conj(y4), -np.conj(y3), np.conj(y2), -np.conj(y1), y0, y1, y2, y3, y4]
    )


def _ylm6(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Y_6m, m = -6..6. Shape (13, n). Physics / Condon-Shortley."""
    c = np.cos(theta)
    s = np.sin(theta)
    c2 = c * c
    c4 = c2 * c2
    s2 = s * s
    s4 = s2 * s2
    e = np.exp(1j * phi)
    e2 = e * e
    e3 = e2 * e
    e4 = e2 * e2
    e5 = e4 * e
    e6 = e3 * e3
    y0 = (1.0 / 32.0) * np.sqrt(13.0 / np.pi) * (231.0 * c4 * c2 - 315.0 * c4 + 105.0 * c2 - 5.0)
    y1 = (
        -(1.0 / 16.0)
        * np.sqrt(273.0 / (2.0 * np.pi))
        * s
        * c
        * (33.0 * c4 - 30.0 * c2 + 5.0)
        * e
    )
    y2 = (1.0 / 64.0) * np.sqrt(1365.0 / np.pi) * s2 * (33.0 * c4 - 18.0 * c2 + 1.0) * e2
    y3 = -(1.0 / 32.0) * np.sqrt(1365.0 / np.pi) * s2 * s * c * (11.0 * c2 - 3.0) * e3
    y4 = (3.0 / 32.0) * np.sqrt(91.0 / (2.0 * np.pi)) * s4 * (11.0 * c2 - 1.0) * e4
    y5 = -(3.0 / 32.0) * np.sqrt(1001.0 / np.pi) * s4 * s * c * e5
    y6 = (1.0 / 64.0) * np.sqrt(3003.0 / np.pi) * s4 * s2 * e6
    return np.stack(
        [
            np.conj(y6),
            -np.conj(y5),
            np.conj(y4),
            -np.conj(y3),
            np.conj(y2),
            -np.conj(y1),
            y0,
            y1,
            y2,
            y3,
            y4,
            y5,
            y6,
        ]
    )


def _angles(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nrm = np.linalg.norm(vec, axis=1)
    ok = nrm > 1e-15
    u = np.zeros_like(vec)
    u[ok] = vec[ok] / nrm[ok, None]
    theta = np.arccos(np.clip(u[:, 2], -1.0, 1.0))
    phi = np.arctan2(u[:, 1], u[:, 0])
    return theta, phi


def _ql_from_ylm(ylm: np.ndarray) -> float:
    """ylm is (2l+1, n_bonds). Global Q_l from mean Q_lm."""
    if ylm.size == 0 or ylm.shape[1] == 0:
        return float("nan")
    qlm = ylm.mean(axis=1)
    ell = (ylm.shape[0] - 1) // 2
    return float(np.sqrt(4.0 * np.pi / (2 * ell + 1) * np.real(np.vdot(qlm, qlm))))


def _ql_local_mean(pos: np.ndarray, dmat: np.ndarray, rcut: float, yfn) -> float:
    n = pos.shape[0]
    vals = []
    for i in range(n):
        nbr = np.where((dmat[i] < rcut) & (dmat[i] > 0.0))[0]
        if nbr.size == 0:
            continue
        theta, phi = _angles(pos[nbr] - pos[i])
        qlm = yfn(theta, phi).mean(axis=1)
        ell = (qlm.size - 1) // 2
        vals.append(np.sqrt(4.0 * np.pi / (2 * ell + 1) * np.real(np.vdot(qlm, qlm))))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def descriptors(pos: np.ndarray, rcut: float) -> dict[str, np.ndarray | float]:
    dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(dmat, 0.0)
    neigh = (dmat > 0.0) & (dmat < rcut)
    cn = neigh.sum(axis=1).astype(np.int32)
    hist = np.zeros(CN_HI - CN_LO + 1, dtype=np.float64)
    for k, c in enumerate(range(CN_LO, CN_HI + 1)):
        hist[k] = float(np.sum(cn == c))
    ii, jj = np.where(np.triu(neigh, 1))
    if ii.size:
        theta, phi = _angles(pos[jj] - pos[ii])
        q4 = _ql_from_ylm(_ylm4(theta, phi))
        q6 = _ql_from_ylm(_ylm6(theta, phi))
    else:
        q4 = q6 = float("nan")
    q4_loc = _ql_local_mean(pos, dmat, rcut, _ylm4)
    q6_loc = _ql_local_mean(pos, dmat, rcut, _ylm6)
    return {
        "hist": hist,
        "cn": cn,
        "q4": q4,
        "q6": q6,
        "q4_loc": q4_loc,
        "q6_loc": q6_loc,
        "n_bonds": int(ii.size),
        "cn_mean": float(cn.mean()),
        "below": int(np.sum(cn < CN_LO)),
        "above": int(np.sum(cn > CN_HI)),
    }


def pairwise_l1(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    d = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        d[i] = np.abs(x - x[i]).sum(axis=1)
    np.fill_diagonal(d, 0.0)
    return d


def w1_matrix(hist: np.ndarray) -> np.ndarray:
    mass = hist.sum(axis=1, keepdims=True)
    mass = np.clip(mass, 1e-15, None)
    cdf = np.cumsum(hist / mass, axis=1)
    return pairwise_l1(cdf)


def torgerson(dist: np.ndarray, dim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    n = dist.shape[0]
    d2 = dist * dist
    h = np.eye(n) - np.full((n, n), 1.0 / n)
    b = -0.5 * h @ d2 @ h
    w, v = np.linalg.eigh(b)
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    lam = np.clip(w[:dim], 0.0, None)
    return v[:, :dim] * np.sqrt(lam), w[:6]


def asinh_d(d: np.ndarray) -> np.ndarray:
    pos = d[d > 0]
    sigma = float(np.median(pos)) if pos.size else 1.0
    out = np.arcsinh(d / max(sigma, 1e-15)) / ASINH_NORM
    np.fill_diagonal(out, 0.0)
    return out, sigma


def orient(xy: np.ndarray, gm: int, ico: int) -> np.ndarray:
    out = xy - xy[gm]
    vec = out[ico]
    if np.linalg.norm(vec) < 1e-15:
        return out
    axis = vec / np.linalg.norm(vec)
    rot = np.array([[axis[0], axis[1]], [-axis[1], axis[0]]])
    return out @ rot.T


def energy_envelope(xy: np.ndarray, rel: np.ndarray, ngrid: int = 160, pad: float = 0.12):
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
    for i, j, e in zip(iy, ix, rel):
        if e < grid[i, j]:
            grid[i, j] = e
    occ = np.isfinite(grid)
    field = np.where(occ, grid, np.nan)
    gx = 0.5 * (xe[:-1] + xe[1:])
    gy = 0.5 * (ye[:-1] + ye[1:])
    return gx, gy, field, occ


def unique_sites(xy: np.ndarray, rel: np.ndarray, ndigits: int = 5):
    """Collapse permutational copies; keep the lowest energy at each site."""
    key = np.round(xy, ndigits)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    n_u = int(inv.max()) + 1
    uxy = np.zeros((n_u, 2))
    uel = np.full(n_u, np.inf)
    for i, u in enumerate(inv):
        if rel[i] < uel[u]:
            uel[u] = rel[i]
            uxy[u] = xy[i]
    return uxy, uel


def energy_idw(xy: np.ndarray, rel: np.ndarray, ngrid: int = 180, k: int = 6, pad: float = 0.10):
    """Inverse-distance energy field on unique sites."""
    uxy, uel = unique_sites(xy, rel)
    k = min(k, len(uxy))
    x, y = uxy[:, 0], uxy[:, 1]
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
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    field = np.empty(grid.shape[0], dtype=np.float64)
    nn = np.empty(grid.shape[0], dtype=np.float64)
    chunk = 2048
    for s in range(0, grid.shape[0], chunk):
        pts = grid[s : s + chunk]
        d2 = ((pts[:, None, :] - uxy[None, :, :]) ** 2).sum(axis=2)
        take = np.argmin(d2, axis=1)
        field[s : s + chunk] = uel[take]
        nn[s : s + chunk] = np.sqrt(d2[np.arange(pts.shape[0]), take])
    field = field.reshape(ngrid, ngrid)
    nn = nn.reshape(ngrid, ngrid)
    # local spacing of unique sites; keep the sparse fcc arm
    if len(uxy) > 1:
        nn_site = []
        for i in range(0, len(uxy), 256):
            block = uxy[i : i + 256]
            d2 = ((block[:, None, :] - uxy[None, :, :]) ** 2).sum(axis=2)
            np.fill_diagonal(d2[: block.shape[0], i : i + block.shape[0]], np.inf)
            nn_site.append(np.sqrt(np.clip(d2.min(axis=1), 0.0, None)))
        med = float(np.median(np.concatenate(nn_site)))
    else:
        med = 0.05 * max(dx, dy)
    cutoff = max(8.0 * med, 0.10 * max(dx, dy))
    field = np.where(nn < cutoff, np.clip(field, 0.0, None), np.nan)
    return gx, gy, field


class _UF:
    def __init__(self, n: int):
        self.p = np.arange(n)

    def find(self, i: int) -> int:
        p = self.p
        while p[i] != i:
            p[i] = p[p[i]]
            i = int(p[i])
        return i

    def union(self, i: int, j: int) -> None:
        a, b = self.find(i), self.find(j)
        if a != b:
            self.p[b] = a


def knn_saddle(xy: np.ndarray, rel: np.ndarray, gm: int, ico: int, knn: int = 12):
    """Lowest max-energy join of GM and ico on unique-site kNN in the plane."""
    uxy, uel = unique_sites(xy, rel)
    # map GM / ico onto unique sites
    gm_u = int(np.argmin(np.linalg.norm(uxy - xy[gm], axis=1)))
    ico_u = int(np.argmin(np.linalg.norm(uxy - xy[ico], axis=1)))
    n = len(uel)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    knn = min(max(knn, 8), n - 1)
    dmat = np.linalg.norm(uxy[:, None, :] - uxy[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    nn = dmat.min(axis=1)
    span = float(np.linalg.norm(uxy.max(0) - uxy.min(0)))
    med = float(np.median(nn[np.isfinite(nn)])) if np.isfinite(nn).any() else 0.0
    rad = max(8.0 * med, 0.10 * span)
    edges = []
    seen = set()
    for i in range(n):
        order = np.argsort(dmat[i])
        picked = set(int(j) for j in order[:knn])
        if rad > 0:
            picked.update(int(j) for j in np.where(dmat[i] <= rad)[0])
        for j in picked:
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            edges.append((float(max(uel[a], uel[b])), a, b))
    edges.sort()
    uf = _UF(n)
    join = None
    for h, i, j in edges:
        if uf.find(i) == uf.find(j):
            continue
        uf.union(i, j)
        if join is None and uf.find(gm_u) == uf.find(ico_u):
            join = h
            break
    if join is None:
        return float("inf"), float("inf"), float("inf")
    return float(join - uel[gm_u]), float(join - uel[ico_u]), float(join)


def sample_field(gx, gy, field, pt) -> float:
    if not np.isfinite(field).any():
        return float("nan")
    ix = int(np.clip(np.searchsorted(gx, pt[0]) - 1, 0, field.shape[1] - 1))
    iy = int(np.clip(np.searchsorted(gy, pt[1]) - 1, 0, field.shape[0] - 1))
    v = field[iy, ix]
    if np.isfinite(v):
        return float(v)
    yy, xx = np.where(np.isfinite(field))
    if xx.size == 0:
        return float("nan")
    j = int(np.argmin((gx[xx] - pt[0]) ** 2 + (gy[yy] - pt[1]) ** 2))
    return float(field[yy[j], xx[j]])


def _cell(gx, gy, field, pt) -> tuple[int, int]:
    ix = int(np.clip(np.searchsorted(gx, pt[0]) - 1, 0, field.shape[1] - 1))
    iy = int(np.clip(np.searchsorted(gy, pt[1]) - 1, 0, field.shape[0] - 1))
    if not np.isfinite(field[iy, ix]):
        yy, xx = np.where(np.isfinite(field))
        if xx.size == 0:
            return iy, ix
        j = int(np.argmin((gx[xx] - pt[0]) ** 2 + (gy[yy] - pt[1]) ** 2))
        return int(yy[j]), int(xx[j])
    return iy, ix


def _reachable(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    if not mask[start] or not mask[goal]:
        return False
    if start == goal:
        return True
    ny, nx = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    stack = [start]
    seen[start] = True
    while stack:
        i, j = stack.pop()
        if (i, j) == goal:
            return True
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and mask[a, b] and not seen[a, b]:
                seen[a, b] = True
                stack.append((a, b))
    return False


def disconnect_saddle(gx, gy, field, a, b) -> float:
    """Lowest level t where the energy envelope connects a to b."""
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return float("nan")
    sa = _cell(gx, gy, field, a)
    sb = _cell(gx, gy, field, b)
    if sa == sb:
        return float(field[sa])
    lo = float(max(field[sa], field[sb]))
    hi = float(finite.max())
    if not _reachable(np.isfinite(field) & (field <= hi + 1e-12), sa, sb):
        return float("inf")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _reachable(np.isfinite(field) & (field <= mid), sa, sb):
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-4:
            break
    return float(hi)


def local_minima(field: np.ndarray) -> list[tuple[float, int, int]]:
    out = []
    ny, nx = field.shape
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            v = field[i, j]
            if not np.isfinite(v):
                continue
            nb = field[i - 1 : i + 2, j - 1 : j + 2]
            if np.nanmin(nb) >= v - 1e-12:
                out.append((float(v), i, j))
    out.sort()
    kept = []
    for rec in out:
        if all(abs(rec[1] - i) + abs(rec[2] - j) > 4 for _, i, j in kept):
            kept.append(rec)
    return kept


def mark(ax, gm_xy, ico_xy):
    h1 = ax.scatter(
        gm_xy[0],
        gm_xy[1],
        s=140,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=50,
        label=rf"GM ${GM_E:.3f}$",
    )
    h2 = ax.scatter(
        ico_xy[0],
        ico_xy[1],
        s=80,
        marker="D",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=50,
        label=rf"ico ${ICO_E:.3f}$",
    )
    ax.legend(
        handles=[h1, h2],
        loc="best",
        fontsize=8,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="k",
    )


def paint(ax, gx, gy, field, xy, rel, gm, ico, xlabel, ylabel, title):
    levels = np.linspace(0.0, EMAX, 21)
    mesh = ax.contourf(gx, gy, np.clip(field, 0, EMAX), levels=levels, cmap=PES, extend="max")
    finite = np.where(np.isfinite(field), field, np.nan)
    ax.contour(
        gx,
        gy,
        finite,
        levels=np.linspace(0.4, EMAX - 0.4, 8),
        colors="#1a1a2e",
        linewidths=0.3,
    )
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=np.clip(rel, 0, EMAX),
        s=6,
        cmap=PES,
        vmin=0,
        vmax=EMAX,
        edgecolors="none",
        alpha=0.45,
        zorder=20,
    )
    mark(ax, xy[gm], xy[ico])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    return mesh


def score_plane(name, xy, rel, gm, ico) -> dict:
    gx, gy, field = energy_idw(xy, rel)
    e_gm = sample_field(gx, gy, field, xy[gm])
    e_ico = sample_field(gx, gy, field, xy[ico])
    flood = disconnect_saddle(gx, gy, field, xy[gm], xy[ico])
    well_gm, well_ico, join = knn_saddle(xy, rel, gm, ico)
    mins = local_minima(field)
    sep = float(np.linalg.norm(xy[gm] - xy[ico]))
    span = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    sep_norm = sep / max(span, 1e-12)
    two_wells = bool(np.isfinite(join) and join > rel[ico] + 0.05 and sep_norm > 0.05)
    gm_deeper = bool(np.isfinite(e_gm) and np.isfinite(e_ico) and e_gm < e_ico - 1e-6)
    return {
        "name": name,
        "sep": sep,
        "sep_norm": sep_norm,
        "Eenv_GM": None if not np.isfinite(e_gm) else float(e_gm),
        "Eenv_ico": None if not np.isfinite(e_ico) else float(e_ico),
        "saddle": None if not np.isfinite(join) else float(join),
        "saddle_flood": None if not np.isfinite(flood) else float(flood),
        "well_GM": None if not np.isfinite(well_gm) else float(well_gm),
        "well_ico": None if not np.isfinite(well_ico) else float(well_ico),
        "n_local_min": len(mins),
        "two_wells": two_wells,
        "gm_deeper": gm_deeper,
        "gm": [float(xy[gm, 0]), float(xy[gm, 1])],
        "ico": [float(xy[ico, 0]), float(xy[ico, 1])],
    }, (gx, gy, field)


def hist_label(hist: np.ndarray) -> str:
    parts = []
    for k, c in enumerate(range(CN_LO, CN_HI + 1)):
        if hist[k] > 0:
            parts.append(f"n{c}={int(hist[k])}")
    return " ".join(parts) if parts else "empty"


def compute_all(xyz: np.ndarray, rcut: float):
    n = xyz.shape[0]
    hist = np.zeros((n, CN_HI - CN_LO + 1), dtype=np.float64)
    q4 = np.empty(n)
    q6 = np.empty(n)
    q4_loc = np.empty(n)
    q6_loc = np.empty(n)
    n_bonds = np.empty(n, dtype=np.int32)
    cn_mean = np.empty(n)
    for i in range(n):
        rec = descriptors(xyz[i], rcut)
        hist[i] = rec["hist"]
        q4[i] = rec["q4"]
        q6[i] = rec["q6"]
        q4_loc[i] = rec["q4_loc"]
        q6_loc[i] = rec["q6_loc"]
        n_bonds[i] = rec["n_bonds"]
        cn_mean[i] = rec["cn_mean"]
        if i % 400 == 0:
            print(f"  r={rcut:.2f} {i}/{n}", flush=True)
    return {
        "hist": hist,
        "q4": q4,
        "q6": q6,
        "q4_loc": q4_loc,
        "q6_loc": q6_loc,
        "n_bonds": n_bonds,
        "cn_mean": cn_mean,
    }


def main() -> None:
    CAND.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    energy, xyz = load_min(MINFILE)
    gm = int(np.argmin(energy))
    ico = int(np.argmin(np.abs(energy - ICO_E)))
    rel = np.clip(energy - float(energy[gm]), 0.0, None)
    print(
        f"n={len(energy)} GM {gm} E={energy[gm]:.8f} "
        f"ico {ico} E={energy[ico]:.8f} dE={energy[ico] - energy[gm]:.6f}"
    )

    scores = []
    planes = {}
    recs = {}
    for rcut in RCUTS:
        tag = f"{rcut:.2f}".replace(".", "p")
        qpath = CAND / f"q4q6_r{tag}.txt"
        hpath = CAND / f"n4n13_r{tag}.cv"
        if qpath.is_file() and hpath.is_file() and qpath.stat().st_size > 100:
            print(f"load cached descriptors r_cut={rcut}", flush=True)
            qtab = np.loadtxt(qpath)
            pack = {
                "hist": np.loadtxt(hpath),
                "q4": qtab[:, 0],
                "q6": qtab[:, 1],
                "q4_loc": qtab[:, 2],
                "q6_loc": qtab[:, 3],
                "cn_mean": qtab[:, 4],
            }
        else:
            print(f"descriptors r_cut={rcut}", flush=True)
            pack = compute_all(xyz, rcut)
            np.savetxt(CAND / f"n4n13_r{tag}.cv", pack["hist"], fmt="%.8e")
            np.savetxt(
                CAND / f"q4q6_r{tag}.txt",
                np.column_stack(
                    [pack["q4"], pack["q6"], pack["q4_loc"], pack["q6_loc"], pack["cn_mean"]]
                ),
                fmt="%.10e",
                header="Q4_global Q6_global q4_localmean q6_localmean cn_mean",
            )
        recs[rcut] = pack
        print(
            f"  GM hist {hist_label(pack['hist'][gm])}  "
            f"Q4={pack['q4'][gm]:.5f} Q6={pack['q6'][gm]:.5f}  "
            f"q4loc={pack['q4_loc'][gm]:.5f} q6loc={pack['q6_loc'][gm]:.5f}"
        )
        print(
            f"  ico hist {hist_label(pack['hist'][ico])}  "
            f"Q4={pack['q4'][ico]:.5f} Q6={pack['q6'][ico]:.5f}  "
            f"q4loc={pack['q4_loc'][ico]:.5f} q6loc={pack['q6_loc'][ico]:.5f}"
        )

        qplane = np.column_stack([pack["q6"], pack["q4"]])
        q_sc, (gx, gy, field) = score_plane(f"q6q4_r{tag}", qplane, rel, gm, ico)
        q_sc.update(
            {
                "rcut": rcut,
                "kind": "q6q4",
                "Q4_GM": float(pack["q4"][gm]),
                "Q6_GM": float(pack["q6"][gm]),
                "Q4_ico": float(pack["q4"][ico]),
                "Q6_ico": float(pack["q6"][ico]),
                "q4loc_GM": float(pack["q4_loc"][gm]),
                "q6loc_GM": float(pack["q6_loc"][gm]),
                "q4loc_ico": float(pack["q4_loc"][ico]),
                "q6loc_ico": float(pack["q6_loc"][ico]),
                "hist_GM": pack["hist"][gm].tolist(),
                "hist_ico": pack["hist"][ico].tolist(),
            }
        )
        scores.append(q_sc)
        planes[f"q6q4_r{tag}"] = (qplane, gx, gy, field, rf"$Q_6$, $Q_4$  $r_c={rcut}$")

        w1 = w1_matrix(pack["hist"])
        d_asinh, sigma = asinh_d(w1)
        xy_mds, ev = torgerson(d_asinh)
        xy_mds = orient(xy_mds, gm, ico)
        np.savetxt(CAND / f"mds_asinh_r{tag}.xy", xy_mds, fmt="%.8e")
        m_sc, (gx2, gy2, field2) = score_plane(f"mds_asinh_r{tag}", xy_mds, rel, gm, ico)
        m_sc.update(
            {
                "rcut": rcut,
                "kind": "mds_asinh_w1",
                "w1_gm_ico": float(w1[gm, ico]),
                "sigma_w1": sigma,
                "ev": [float(x) for x in ev],
            }
        )
        scores.append(m_sc)
        planes[f"mds_asinh_r{tag}"] = (
            xy_mds,
            gx2,
            gy2,
            field2,
            rf"asinh W1 $n_4\ldots n_{{13}}$  $r_c={rcut}$",
        )
        print(
            f"  Q plane sep_norm={q_sc['sep_norm']:.4f} "
            f"Egm={q_sc['Eenv_GM']} Eico={q_sc['Eenv_ico']} "
            f"saddle={q_sc['saddle']} wellGM={q_sc['well_GM']} wellico={q_sc['well_ico']}"
        )
        print(
            f"  MDS   sep_norm={m_sc['sep_norm']:.4f} "
            f"Egm={m_sc['Eenv_GM']} Eico={m_sc['Eenv_ico']} "
            f"saddle={m_sc['saddle']} W1(GM,ico)={w1[gm, ico]:.4f}"
        )

    order = ["q6q4_r1p20", "q6q4_r1p35", "mds_asinh_r1p20", "mds_asinh_r1p35"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.6), facecolor="white")
    mesh = None
    for ax, key in zip(axes.ravel(), order):
        xy, gx, gy, field, title = planes[key]
        xlab = r"$Q_6$" if key.startswith("q6q4") else r"MDS$_1$"
        ylab = r"$Q_4$" if key.startswith("q6q4") else r"MDS$_2$"
        mesh = paint(ax, gx, gy, field, xy, rel, gm, ico, xlab, ylab, title)
        if key.startswith("q6q4"):
            ax.set_aspect("auto")
        else:
            ax.set_aspect("equal", adjustable="datalim")
    fig.subplots_adjust(right=0.90, wspace=0.18, hspace=0.22)
    cax = fig.add_axes([0.92, 0.18, 0.018, 0.64])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label(r"$E-E_{\mathrm{GM}}/\varepsilon$")
    dest = OUT / "elja_occ_lj38_cnxyz.png"
    fig.savefig(dest, dpi=170, facecolor="white")
    fig.savefig(CAND / "elja_occ_lj38_cnxyz.png", dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", dest)

    # dedicated (Q6, Q4) hero at the cutoff that splits more
    q_scores = [s for s in scores if s["kind"] == "q6q4"]
    best = max(q_scores, key=lambda s: (s["sep_norm"], s["two_wells"], -abs((s["Eenv_GM"] or 9))))
    key = best["name"]
    xy, gx, gy, field, title = planes[key]
    fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor="white")
    mesh = paint(ax, gx, gy, field, xy, rel, gm, ico, r"$Q_6$", r"$Q_4$", title + r": $E-E_{\mathrm{GM}}$")
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03).set_label(r"$E-E_{\mathrm{GM}}/\varepsilon$")
    fig.tight_layout()
    fig.savefig(CAND / "q6q4_energy.png", dpi=170, facecolor="white")
    plt.close(fig)

    payload = {
        "n": int(len(energy)),
        "minfile": str(MINFILE),
        "source": "recomputed from lj38_0013.min, not lj38.cv",
        "field": "E-E_GM nearest unique site (lower envelope), not occupancy",
        "gm": gm,
        "ico": ico,
        "E_GM": float(energy[gm]),
        "E_ico": float(energy[ico]),
        "dE_ico": float(energy[ico] - energy[gm]),
        "rcuts": list(RCUTS),
        "hist_bins": list(range(CN_LO, CN_HI + 1)),
        "scores": scores,
        "best_q_plane": best["name"],
    }
    (CAND / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        f"GM idx {gm} E {energy[gm]:.8f}",
        f"ico idx {ico} E {energy[ico]:.8f} dE {energy[ico] - energy[gm]:.6f}",
    ]
    for s in scores:
        if s["kind"] == "q6q4":
            lines.append(
                f"{s['name']} Q4/Q6 GM {s['Q4_GM']:.5f}/{s['Q6_GM']:.5f} "
                f"ico {s['Q4_ico']:.5f}/{s['Q6_ico']:.5f} "
                f"wellGM {s['well_GM']} wellico {s['well_ico']} "
                f"saddle {s['saddle']} sep {s['sep_norm']:.4f}"
            )
        else:
            lines.append(
                f"{s['name']} W1 {s.get('w1_gm_ico')} wellGM {s['well_GM']} "
                f"wellico {s['well_ico']} saddle {s['saddle']} sep {s['sep_norm']:.4f}"
            )
    (CAND / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
