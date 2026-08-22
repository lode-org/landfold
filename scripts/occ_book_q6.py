#!/usr/bin/env python3
"""Steinhardt Q4/Q6 of the Elja LJ38 book (4042 inherent structures).

The published Wales/Doye basin plane is the *global* bond-order pair
(Q4, Q6): fcc truncated octahedron at high Q, incomplete Mackay ico
at low Q. Per-atom q_l averaged over the cluster (the local mean) is
also computed; textbook bulk values (ico q6 ~ 0.66, fcc ~ 0.57) are
the 12-neighbour crystal numbers and are *measured* here, not assumed.

Neighbour cutoff 1.2 sigma (first LJ shell). If that cutoff fails to
split the two lowest-E motifs, r_cut is scanned in {1.15, 1.20, 1.25,
1.35} and the split-maximising value is kept.

Energy fill is IMQ (MethodsX inverse-multiquadric), not occupancy
invert. Markers: white star = GM, white triangle = ico.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

MINFILE = Path("/tmp/occ-book/lj38_0013.min")
OUT = Path("/tmp/occ-book/cand-q6")
FIGDIR = Path(__file__).resolve().parents[1] / "docs" / "ceriotti-figs"
N_ATOMS = 38
GM_E = -173.928427
ICO_E = -173.252378
CUTS = (1.15, 1.20, 1.25, 1.35)
DEFAULT_RC = 1.20
MIN_NB = 4


def _P4(x: np.ndarray) -> np.ndarray:
    x2 = x * x
    return (35.0 * x2 * x2 - 30.0 * x2 + 3.0) / 8.0


def _P6(x: np.ndarray) -> np.ndarray:
    x2 = x * x
    x4 = x2 * x2
    return (231.0 * x2 * x4 - 315.0 * x4 + 105.0 * x2 - 5.0) / 16.0


def _ql_from_u(u: np.ndarray) -> tuple[float, float]:
    """Rotationally invariant q4, q6 of a set of unit bond vectors.

    Spherical-harmonic addition theorem:
        q_l^2 = <P_l(n_j · n_k)>_{jk}  ==  (4π/(2l+1)) sum_m |<Y_lm>|^2
    so this is the Y_lm definition with no complex arithmetic.
    """
    if u.shape[0] == 0:
        return float("nan"), float("nan")
    c = np.clip(u @ u.T, -1.0, 1.0)
    q4 = float(np.sqrt(max(float(_P4(c).mean()), 0.0)))
    q6 = float(np.sqrt(max(float(_P6(c).mean()), 0.0)))
    return q4, q6


def load_min(path: Path, n_atoms: int) -> tuple[np.ndarray, list[np.ndarray]]:
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


def steinhardt(pos: np.ndarray, r_cut: float) -> dict[str, float]:
    """Per-atom q_l and global Q_l for l = 4, 6. Vectorised over atoms."""
    n = pos.shape[0]
    dvec = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(dvec, axis=2)
    np.fill_diagonal(dist, np.inf)
    neigh = dist < r_cut
    nbs = neigh.sum(axis=1)
    out = {
        "mean_nb": float(nbs.mean()) if n else 0.0,
        "min_nb": float(nbs.min()) if n else 0.0,
        "max_nb": float(nbs.max()) if n else 0.0,
        "n_core12": int((nbs >= 12).sum()),
        "n_starved": int((nbs < MIN_NB).sum()),
    }
    max_nb = int(nbs.max()) if n else 0
    if max_nb == 0:
        for key in ("q4_local", "q6_local", "Q4_global", "Q6_global", "q4_core", "q6_core"):
            out[key] = float("nan")
        return out
    idx = np.full((n, max_nb), -1, dtype=np.intp)
    for i in range(n):
        js = np.flatnonzero(neigh[i])
        idx[i, : js.size] = js
    js = np.clip(idx, 0, n - 1)
    take_i = np.arange(n)[:, None]
    vec = dvec[take_i, js]
    rij = dist[take_i, js]
    valid = idx >= 0
    u = np.zeros_like(vec)
    u[valid] = vec[valid] / rij[valid][:, None]
    gram = np.einsum("akc,amc->akm", u, u)
    pair = valid[:, :, None] & valid[:, None, :]
    gram = np.clip(gram, -1.0, 1.0)
    n_pair = pair.sum(axis=(1, 2)).astype(np.float64)
    n_pair = np.clip(n_pair, 1.0, None)
    q4_i = np.sqrt(np.clip((_P4(gram) * pair).sum(axis=(1, 2)) / n_pair, 0.0, None))
    q6_i = np.sqrt(np.clip((_P6(gram) * pair).sum(axis=(1, 2)) / n_pair, 0.0, None))
    q4_i = np.where(nbs > 0, q4_i, np.nan)
    q6_i = np.where(nbs > 0, q6_i, np.nan)
    good = np.isfinite(q4_i)
    core = nbs >= 12
    out["q4_local"] = float(q4_i[good].mean()) if good.any() else float("nan")
    out["q6_local"] = float(q6_i[good].mean()) if good.any() else float("nan")
    out["q4_core"] = float(q4_i[core].mean()) if core.any() else float("nan")
    out["q6_core"] = float(q6_i[core].mean()) if core.any() else float("nan")
    ub = u[valid]
    out["Q4_global"], out["Q6_global"] = _ql_from_u(ub)
    return out


def compute_book(frames: list[np.ndarray], r_cut: float) -> dict[str, np.ndarray]:
    recs = []
    n = len(frames)
    for k, pos in enumerate(frames):
        recs.append(steinhardt(pos, r_cut))
        if (k + 1) % 500 == 0 or k + 1 == n:
            print(f"  r_cut={r_cut:.2f}  {k + 1}/{n}", flush=True)
    keys = recs[0].keys()
    return {key: np.asarray([r[key] for r in recs], dtype=np.float64) for key in keys}


def motif_ids(energy: np.ndarray) -> tuple[int, int]:
    gm = int(np.argmin(energy))
    ico = int(np.argmin(np.abs(energy - ICO_E)))
    return gm, ico


def split_score(book: dict[str, np.ndarray], gm: int, ico: int, kind: str) -> dict[str, float]:
    if kind == "local":
        q4, q6 = book["q4_local"], book["q6_local"]
    else:
        q4, q6 = book["Q4_global"], book["Q6_global"]
    d4 = float(q4[gm] - q4[ico])
    d6 = float(q6[gm] - q6[ico])
    sep = float(math.hypot(d4, d6))
    # overlap: fraction of all minima whose (Q4,Q6) sits in both
    # 1-sigma balls around the two motifs (using the motif copies)
    e = None
    return {
        "dQ4": d4,
        "dQ6": d6,
        "sep": sep,
        "Q4_gm": float(q4[gm]),
        "Q6_gm": float(q6[gm]),
        "Q4_ico": float(q4[ico]),
        "Q6_ico": float(q6[ico]),
    }


def overlap_1d(q6: np.ndarray, energy: np.ndarray, gm: int, ico: int) -> dict[str, float]:
    """How much the low-E copies of each motif overlap in Q6."""
    gm_mask = np.abs(energy - energy[gm]) < 1e-6
    ico_mask = np.abs(energy - energy[ico]) < 1e-4
    qg, qi = q6[gm_mask], q6[ico_mask]
    lo = max(float(qg.min()), float(qi.min()))
    hi = min(float(qg.max()), float(qi.max()))
    return {
        "n_gm": int(gm_mask.sum()),
        "n_ico": int(ico_mask.sum()),
        "q6_gm_span": [float(qg.min()), float(qg.max())],
        "q6_ico_span": [float(qi.min()), float(qi.max())],
        "q6_range_overlap": max(0.0, hi - lo),
        "midpoint_gap": float(abs(qg.mean() - qi.mean())),
    }


def pick_plane(
    scans: dict[float, dict[str, np.ndarray]], energy: np.ndarray, gm: int, ico: int
) -> tuple[float, str, dict[str, float]]:
    """Largest GM-ico separation among cutoffs that are not starved."""
    best = None
    for rc, book in scans.items():
        starved = float(book["n_starved"].mean())
        if starved > 4.0:
            continue
        for kind in ("global", "local"):
            sc = split_score(book, gm, ico, kind)
            sc["r_cut"] = rc
            sc["kind"] = kind
            sc["mean_nb_gm"] = float(book["mean_nb"][gm])
            sc["mean_nb_ico"] = float(book["mean_nb"][ico])
            if best is None or sc["sep"] > best["sep"]:
                best = sc
    if best is None:
        raise SystemExit("every cutoff starved of neighbours")
    return float(best["r_cut"]), str(best["kind"]), best


def _unit_grid(xy: np.ndarray, ngrid: int, pad: float):
    lo = xy.min(0)
    span = np.clip(xy.max(0) - lo, 1e-12, None)
    uv = (xy - lo) / span
    umin, vmin = float(uv[:, 0].min()), float(uv[:, 1].min())
    umax, vmax = float(uv[:, 0].max()), float(uv[:, 1].max())
    du, dv = max(umax - umin, 1e-6), max(vmax - vmin, 1e-6)
    umin -= pad * du
    umax += pad * du
    vmin -= pad * dv
    vmax += pad * dv
    gu = np.linspace(umin, umax, ngrid)
    gv = np.linspace(vmin, vmax, ngrid)
    gx = lo[0] + gu * span[0]
    gy = lo[1] + gv * span[1]
    return lo, span, uv, gu, gv, gx, gy


def _blur(z: np.ndarray, passes: int = 2) -> np.ndarray:
    k = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    k /= k.sum()
    out = z.copy()
    for _ in range(passes):
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, out)
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)
    return out


def energy_field(xy: np.ndarray, values: np.ndarray, ngrid: int = 220, pad: float = 0.10, cutoff: float = 0.055, power: float = 4.0):
    """Compact-support IDW of energy, seeded by the per-cell lower envelope.

    IMQ with infinite support averages 4000 liquid-like minima over the
    whole plane and buries both wells. A hard cutoff plus a high IDW
    power keeps the GM and ico floors local. Empty space stays NaN.
    """
    lo, span, uv, gu, gv, gx, gy = _unit_grid(xy, ngrid, pad)
    uu, vv = np.meshgrid(gu, gv)
    # density mask from a compact Gaussian histogram
    counts, _, _ = np.histogram2d(
        uv[:, 0], uv[:, 1], bins=ngrid, range=[[gu[0], gu[-1]], [gv[0], gv[-1]]]
    )
    dens = _blur(counts.T, passes=3)
    dmax = float(dens.max()) if float(dens.max()) > 0 else 1.0
    mask = dens > 0.012 * dmax

    # per-cell minimum (the inherent-structure envelope)
    xe = np.linspace(gu[0], gu[-1], ngrid + 1)
    ye = np.linspace(gv[0], gv[-1], ngrid + 1)
    ix = np.clip(np.digitize(uv[:, 0], xe) - 1, 0, ngrid - 1)
    iy = np.clip(np.digitize(uv[:, 1], ye) - 1, 0, ngrid - 1)
    grid = np.full((ngrid, ngrid), np.inf)
    for i, j, e in zip(iy, ix, values):
        if e < grid[i, j]:
            grid[i, j] = e
    occupied = np.isfinite(grid)
    # IDW fill of masked empty cells from occupied cell centres
    cy, cx = np.where(occupied)
    src = np.column_stack([gu[cx], gv[cy]])
    sval = grid[occupied]
    field = np.full((ngrid, ngrid), np.nan)
    field[occupied] = sval
    need = mask & ~occupied
    qy, qx = np.where(need)
    if qy.size and src.shape[0]:
        qs = np.column_stack([gu[qx], gv[qy]])
        batch = 500
        filled = np.empty(qy.size)
        for i0 in range(0, qy.size, batch):
            i1 = min(i0 + batch, qy.size)
            r2 = (qs[i0:i1, 0:1] - src[None, :, 0]) ** 2 + (qs[i0:i1, 1:2] - src[None, :, 1]) ** 2
            r2 = r2.reshape(i1 - i0, src.shape[0])
            near = r2 <= cutoff * cutoff
            w = np.where(near, 1.0 / np.power(r2 + 1e-12, 0.5 * power), 0.0)
            den = w.sum(1)
            num = w @ sval
            filled[i0:i1] = np.where(den > 0, num / np.clip(den, 1e-12, None), np.nan)
        field[qy, qx] = filled
    # light blur of finite cells only
    finite = np.isfinite(field)
    fill = np.where(finite, field, 0.0)
    wgt = finite.astype(np.float64)
    fill_b = _blur(fill, passes=2)
    wgt_b = _blur(wgt, passes=2)
    field = np.where(mask & (wgt_b > 0.15), fill_b / np.clip(wgt_b, 1e-12, None), np.nan)
    return gx, gy, field, dens


def field_at(gx, gy, field, pt) -> float:
    i = int(np.argmin(np.abs(gy - pt[1])))
    j = int(np.argmin(np.abs(gx - pt[0])))
    return float(field[i, j])


def well_depth(gx, gy, field, pt, rad_frac: float = 0.08) -> float:
    """Deepest finite IMQ value inside a fraction of the plane span."""
    xx, yy = np.meshgrid(gx, gy)
    span = max(float(gx.max() - gx.min()), float(gy.max() - gy.min()), 1e-12)
    rad = rad_frac * span
    near = (xx - pt[0]) ** 2 + (yy - pt[1]) ** 2 <= rad * rad
    chunk = field[near]
    chunk = chunk[np.isfinite(chunk)]
    return float(chunk.min()) if chunk.size else float("nan")


def saddle_along(gx, gy, field, a, b, n: int = 80) -> float:
    ts = np.linspace(0.0, 1.0, n)
    vals = []
    for t in ts:
        p = (1.0 - t) * a + t * b
        vals.append(field_at(gx, gy, field, p))
    arr = np.asarray(vals, dtype=np.float64)
    if not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmax(arr))


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "mathtext.fontset": "dejavusans",
        }
    )


def mark(ax, gm_xy, ico_xy, gm_e, ico_e) -> None:
    h1 = ax.scatter(
        gm_xy[0],
        gm_xy[1],
        s=220,
        marker="*",
        c="white",
        edgecolors="black",
        linewidths=0.9,
        zorder=7,
        label=rf"GM  ${gm_e:.3f}$",
    )
    h2 = ax.scatter(
        ico_xy[0],
        ico_xy[1],
        s=110,
        marker="^",
        c="white",
        edgecolors="black",
        linewidths=0.9,
        zorder=7,
        label=rf"ico  ${ico_e:.3f}$",
    )
    ax.legend(handles=[h1, h2], loc="best", frameon=True, fancybox=False, framealpha=0.92)


def plot_q6q4(xy, rel, gm, ico, energy, kind, r_cut, dest: Path, extra: Path | None) -> dict:
    print("IDW/envelope fill ...", flush=True)
    gx, gy, ehat, _dens = energy_field(xy, rel, ngrid=220, pad=0.10, cutoff=0.05, power=4.0)
    print("fill done", flush=True)
    e_clip = 6.0
    painted = np.clip(ehat, 0.0, e_clip)
    gm_xy, ico_xy = xy[gm], xy[ico]
    e_gm = field_at(gx, gy, painted, gm_xy)
    e_ico = field_at(gx, gy, painted, ico_xy)
    d_gm = well_depth(gx, gy, painted, gm_xy, rad_frac=0.05)
    d_ico = well_depth(gx, gy, painted, ico_xy, rad_frac=0.05)
    sad = saddle_along(gx, gy, painted, gm_xy, ico_xy)

    fig, ax = plt.subplots(figsize=(6.6, 5.7), facecolor="white")
    levels = np.linspace(0.0, e_clip, 25)
    mesh = ax.contourf(gx, gy, painted, levels=levels, cmap="cividis", extend="max")
    ax.contour(
        gx,
        gy,
        np.where(np.isfinite(painted), painted, np.nan),
        levels=np.linspace(0.35, e_clip - 0.35, 12),
        colors="0.12",
        linewidths=0.35,
        alpha=0.80,
    )
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=6,
        c=np.clip(rel, 0, e_clip),
        cmap="cividis",
        vmin=0,
        vmax=e_clip,
        linewidths=0,
        alpha=0.35,
        zorder=3,
    )
    mark(ax, gm_xy, ico_xy, energy[gm], energy[ico])
    xlab = r"$\langle q_4\rangle$" if kind == "local" else r"$Q_4$"
    ylab = r"$\langle q_6\rangle$" if kind == "local" else r"$Q_6$"
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(rf"LJ38  Steinhardt {kind}  $r_{{\mathrm{{cut}}}}={r_cut:.2f}\,\sigma$")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    for sp in ax.spines.values():
        sp.set_color("0.25")
    cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$E-E_{\mathrm{GM}}/\varepsilon$")
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=200, facecolor="white")
    if extra is not None:
        extra.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(extra, dpi=200, facecolor="white")
    plt.close(fig)
    return {
        "Eimq_GM": e_gm,
        "Eimq_ico": e_ico,
        "well_GM": d_gm,
        "well_ico": d_ico,
        "saddle_GM_ico": sad,
        "well_depth_GM": float(sad - d_gm) if np.isfinite(sad) and np.isfinite(d_gm) else float("nan"),
        "well_depth_ico": float(sad - d_ico) if np.isfinite(sad) and np.isfinite(d_ico) else float("nan"),
    }


def plot_q6e(q6, energy, gm, ico, kind, r_cut, dest: Path, extra: Path | None) -> None:
    rel = energy - energy.min()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), facecolor="white")

    ax = axes[0]
    sc = ax.scatter(
        q6,
        energy,
        s=8,
        c=rel,
        cmap="cividis",
        vmin=0,
        vmax=8,
        linewidths=0,
        alpha=0.55,
        zorder=3,
    )
    mark(ax, (q6[gm], energy[gm]), (q6[ico], energy[ico]), energy[gm], energy[ico])
    ax.set_xlabel(r"$\langle q_6\rangle$" if kind == "local" else r"$Q_6$")
    ax.set_ylabel(r"$E/\varepsilon$")
    ax.set_title("scatter")
    ax.set_ylim(energy.min() - 0.4, min(energy.max(), energy.min() + 14.0))
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$E-E_{\mathrm{GM}}/\varepsilon$")

    ax = axes[1]
    qmin, qmax = float(q6.min()), float(q6.max())
    emin, emax = float(energy.min()), float(energy.max())
    pad_q = 0.08 * max(qmax - qmin, 1e-6)
    pad_e = 0.08 * max(emax - emin, 1e-6)
    counts, xe, ye = np.histogram2d(
        q6,
        energy,
        bins=90,
        range=[[qmin - pad_q, qmax + pad_q], [emin - pad_e, emax + pad_e]],
    )
    # blur with a small Gaussian via FFT-free box-ish convolution
    k = np.array([1, 4, 6, 4, 1], dtype=np.float64)
    k /= k.sum()
    blur = counts.astype(np.float64)
    for _ in range(3):
        blur = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, blur)
        blur = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, blur)
    gx = 0.5 * (xe[:-1] + xe[1:])
    gy = 0.5 * (ye[:-1] + ye[1:])
    rho = blur.T
    rmax = float(rho.max()) if float(rho.max()) > 0 else 1.0
    dens = np.where(rho > 0.004 * rmax, rho / rmax, np.nan)
    mesh = ax.contourf(gx, gy, dens, levels=np.linspace(0.0, 1.0, 21), cmap="cividis")
    ax.contour(
        gx,
        gy,
        np.where(np.isfinite(dens), dens, np.nan),
        levels=np.linspace(0.12, 0.92, 8),
        colors="0.15",
        linewidths=0.35,
        alpha=0.7,
    )
    mark(ax, (q6[gm], energy[gm]), (q6[ico], energy[ico]), energy[gm], energy[ico])
    ax.set_xlabel(r"$\langle q_6\rangle$" if kind == "local" else r"$Q_6$")
    ax.set_ylabel(r"$E/\varepsilon$")
    ax.set_title("density")
    ax.set_ylim(energy.min() - 0.4, min(energy.max(), energy.min() + 14.0))
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03).set_label("normalised density")
    fig.suptitle(
        rf"LJ38  $Q_6$ vs $E$  ({kind}, $r_{{\mathrm{{cut}}}}={r_cut:.2f}\,\sigma$)",
        y=1.01,
    )
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=200, facecolor="white", bbox_inches="tight")
    if extra is not None:
        extra.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(extra, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    style()
    energy, frames = load_min(MINFILE, N_ATOMS)
    if len(frames) != 4042:
        print(f"warning: expected 4042 minima, got {len(frames)}")
    gm, ico = motif_ids(energy)
    print(
        f"n={len(frames)}  GM idx={gm} E={energy[gm]:.8f}  "
        f"ico idx={ico} E={energy[ico]:.8f}  dE={energy[ico] - energy[gm]:.6f}"
    )

    print("\n=== cutoff scan on GM and ico only ===", flush=True)
    motif_scan: dict[float, dict[str, dict[str, float]]] = {}
    for rc in CUTS:
        gm_rec = steinhardt(frames[gm], rc)
        ico_rec = steinhardt(frames[ico], rc)
        motif_scan[rc] = {"gm": gm_rec, "ico": ico_rec}
        d4g = gm_rec["Q4_global"] - ico_rec["Q4_global"]
        d6g = gm_rec["Q6_global"] - ico_rec["Q6_global"]
        d4l = gm_rec["q4_local"] - ico_rec["q4_local"]
        d6l = gm_rec["q6_local"] - ico_rec["q6_local"]
        print(
            f"r_cut={rc:.2f}  nb GM/ico={gm_rec['mean_nb']:.2f}/{ico_rec['mean_nb']:.2f}  "
            f"starved GM/ico={gm_rec['n_starved']}/{ico_rec['n_starved']}"
        )
        print(
            f"  local  q4 GM/ico={gm_rec['q4_local']:.4f}/{ico_rec['q4_local']:.4f}  "
            f"q6 GM/ico={gm_rec['q6_local']:.4f}/{ico_rec['q6_local']:.4f}  "
            f"sep={math.hypot(d4l, d6l):.4f}"
        )
        print(
            f"  global Q4 GM/ico={gm_rec['Q4_global']:.4f}/{ico_rec['Q4_global']:.4f}  "
            f"Q6 GM/ico={gm_rec['Q6_global']:.4f}/{ico_rec['Q6_global']:.4f}  "
            f"sep={math.hypot(d4g, d6g):.4f}"
        )
        print(
            f"  core12 q4 GM/ico={gm_rec['q4_core']:.4f}/{ico_rec['q4_core']:.4f}  "
            f"q6 GM/ico={gm_rec['q6_core']:.4f}/{ico_rec['q6_core']:.4f}"
        )

    def _motif_sep(rc: float, kind: str) -> float:
        a, b = motif_scan[rc]["gm"], motif_scan[rc]["ico"]
        if kind == "local":
            return math.hypot(a["q4_local"] - b["q4_local"], a["q6_local"] - b["q6_local"])
        return math.hypot(a["Q4_global"] - b["Q4_global"], a["Q6_global"] - b["Q6_global"])

    r_cut = DEFAULT_RC
    default = motif_scan[DEFAULT_RC]
    starved = default["gm"]["n_starved"] > 0 or default["ico"]["n_starved"] > 0
    sep_g = _motif_sep(DEFAULT_RC, "global")
    sep_l = _motif_sep(DEFAULT_RC, "local")
    if starved or max(sep_g, sep_l) < 0.03:
        r_cut = max(CUTS, key=lambda rc: max(_motif_sep(rc, "global"), _motif_sep(rc, "local")))
        print(f"default cutoff failed split/neighbours; using r_cut={r_cut:.2f}")
    kind = "global" if _motif_sep(r_cut, "global") >= _motif_sep(r_cut, "local") else "local"
    print(f"\nfull book at r_cut={r_cut:.2f}  kind={kind}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"book_rc{r_cut:.2f}.npz"
    if cache.exists():
        loaded = np.load(cache)
        book = {k: loaded[k] for k in loaded.files}
        print(f"loaded cache {cache} n={book['q6_local'].size}", flush=True)
    else:
        book = compute_book(frames, r_cut)
        np.savez(cache, **book)
        print(f"wrote {cache}", flush=True)
    picked = split_score(book, gm, ico, kind)
    picked["r_cut"] = r_cut
    picked["kind"] = kind
    print(f"\nchosen r_cut={r_cut:.2f}  kind={kind}  sep={picked['sep']:.4f}")
    print(
        "VERIFY  (do not assume ico~0.66 / fcc~0.57):\n"
        f"  mean local q6  GM(fcc)={book['q6_local'][gm]:.4f}  ico={book['q6_local'][ico]:.4f}\n"
        f"  core-12  q6    GM(fcc)={book['q6_core'][gm]:.4f}  ico={book['q6_core'][ico]:.4f}\n"
        f"  global   Q6    GM(fcc)={book['Q6_global'][gm]:.4f}  ico={book['Q6_global'][ico]:.4f}"
    )
    ov = overlap_1d(
        book["q6_local"] if kind == "local" else book["Q6_global"], energy, gm, ico
    )
    print("Q6 overlap of motif copies:", json.dumps(ov))

    if kind == "local":
        q4, q6 = book["q4_local"], book["q6_local"]
    else:
        q4, q6 = book["Q4_global"], book["Q6_global"]
    xy = np.column_stack([q4, q6])
    rel = energy - energy.min()

    OUT.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT / "q4q6.xy", xy, fmt="%.10e")
    np.savetxt(OUT / "q6E.xy", np.column_stack([q6, energy]), fmt="%.10e")
    np.savetxt(OUT / "energy.txt", energy, fmt="%.10e")

    fig_q6q4 = OUT / "elja_occ_lj38_q6q4.png"
    fig_q6e = OUT / "elja_occ_lj38_q6E.png"
    wells = plot_q6q4(
        xy, rel, gm, ico, energy, kind, r_cut, fig_q6q4, FIGDIR / "elja_occ_lj38_q6q4.png"
    )
    plot_q6e(q6, energy, gm, ico, kind, r_cut, fig_q6e, FIGDIR / "elja_occ_lj38_q6E.png")
    print("wrote", fig_q6q4)
    print("wrote", fig_q6e)
    print("wrote", FIGDIR / "elja_occ_lj38_q6q4.png")
    print("wrote", FIGDIR / "elja_occ_lj38_q6E.png")
    print(
        "wells  fill@GM={:.4f} fill@ico={:.4f}  "
        "localmin GM/ico={:.4f}/{:.4f}  saddle={:.4f}  "
        "depth GM/ico={:.4f}/{:.4f}".format(
            wells["Eimq_GM"],
            wells["Eimq_ico"],
            wells["well_GM"],
            wells["well_ico"],
            wells["saddle_GM_ico"],
            wells["well_depth_GM"],
            wells["well_depth_ico"],
        )
    )

    scores = {
        "n": int(len(frames)),
        "n_atoms": N_ATOMS,
        "r_cut": r_cut,
        "kind": kind,
        "gm_idx": gm,
        "ico_idx": ico,
        "E_GM": float(energy[gm]),
        "E_ico": float(energy[ico]),
        "dE_ico_minus_GM": float(energy[ico] - energy[gm]),
        "Q4_GM": float(q4[gm]),
        "Q6_GM": float(q6[gm]),
        "Q4_ico": float(q4[ico]),
        "Q6_ico": float(q6[ico]),
        "q6_local_GM": float(book["q6_local"][gm]),
        "q6_local_ico": float(book["q6_local"][ico]),
        "q6_core_GM": float(book["q6_core"][gm]),
        "q6_core_ico": float(book["q6_core"][ico]),
        "Q6_global_GM": float(book["Q6_global"][gm]),
        "Q6_global_ico": float(book["Q6_global"][ico]),
        "Q4_global_GM": float(book["Q4_global"][gm]),
        "Q4_global_ico": float(book["Q4_global"][ico]),
        "sep": float(picked["sep"]),
        "mean_nb_GM": float(book["mean_nb"][gm]),
        "mean_nb_ico": float(book["mean_nb"][ico]),
        "overlap": ov,
        "wells": wells,
        "note": (
            "Textbook ico q6~0.66 is the 13-atom icosahedron centre. "
            "LJ38 ico is an incomplete Mackay; its global Q6 is *lower* than fcc. "
            "Doye/Miller/Wales use global Q4 (fcc~0.186, ico~0.015)."
        ),
    }
    (OUT / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
