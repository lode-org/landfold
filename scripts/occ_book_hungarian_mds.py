#!/usr/bin/env python3
"""Energy IDW of the landmark MDS of Hungarian RMSD.

The labelled (d_GM, d_ico) plane is elja_occ_lj38_hungarian.png: GM is
a satellite by construction. This script fills the unsupervised
landmark MDS of the landmark-landmark Hungarian matrix D_ll.

mds.xy is reused when present. Otherwise Torgerson of D_ll.dist plus
Nyström from (d_GM, d_ico) if that plane exists.

The painted field is E - E_GM (IDW, no leftover occupancy). Candidate
fills are scored for two basins, GM in a well, and connectivity.
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

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import occ_book_hungarian as hung

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ceriotti-figs"
DEST = Path("/tmp/occ-book/cand-hung")
FIG = OUT / "elja_occ_lj38_hungarian_mds.png"
ENERGY = Path("/tmp/occ-book/lj38.energy")

PES = hung.PES
GM_E = hung.GM_E
ICO_E = hung.ICO_E
EMAX = 4.0
GM = 0
ICO = 40
SEP_BAR = 0.25


def load_energy() -> np.ndarray:
    if (DEST / "energy.txt").is_file():
        return np.loadtxt(DEST / "energy.txt")
    if ENERGY.is_file():
        return np.loadtxt(ENERGY)
    raise SystemExit("no energy.txt / lj38.energy")


def torgerson(dist: np.ndarray, dim: int = 2):
    return hung.torgerson(dist, dim)


def nystrom_idw(xy_feat, land_idx, land_xy, k=8):
    return hung.nystrom_idw(xy_feat, land_idx, land_xy, k=k)


def unique_lowest(xy: np.ndarray, z: np.ndarray):
    keep = np.isfinite(xy).all(1)
    xy, z = xy[keep], z[keep]
    key = np.round(xy, 8)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    vals = np.full(len(first), np.inf)
    np.minimum.at(vals, inv, z)
    return xy[first].copy(), vals


def collapse_copies(xy: np.ndarray, z: np.ndarray, d_gm: np.ndarray, d_ico: np.ndarray):
    """Park fingerprint-identical GM / ico copies on the basin sites."""
    out = xy.copy()
    zz = z.copy()
    out[d_gm < 1e-8] = xy[GM]
    zz[d_gm < 1e-8] = 0.0
    out[d_ico < 1e-8] = xy[ICO]
    zz[d_ico < 1e-8] = float(z[ICO])
    return out, zz


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


def inducing(xy: np.ndarray, z: np.ndarray, n_ind: int = 110, n_low: int = 50):
    pts, val = unique_lowest(xy, z)
    order = np.argsort(val)
    low = order[: min(n_low, len(order))]
    loc_gm = int(np.argmin(np.linalg.norm(pts - xy[GM], axis=1)))
    loc_ico = int(np.argmin(np.linalg.norm(pts - xy[ICO], axis=1)))
    cover = farthest_indices(pts, min(n_ind, len(pts)), [loc_gm, loc_ico])
    idx = np.unique(np.concatenate([cover, low, np.array([loc_gm, loc_ico])]))
    return pts[idx], np.clip(val[idx], 0.0, EMAX)


def knn(ref, query, k):
    k = min(int(k), len(ref))
    chunk = 350
    dists = np.empty((query.shape[0], k), dtype=np.float64)
    idxs = np.empty((query.shape[0], k), dtype=np.int64)
    for start in range(0, query.shape[0], chunk):
        stop = min(start + chunk, query.shape[0])
        d2 = ((query[start:stop, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
        part = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        take = np.take_along_axis(d2, part, axis=1)
        order = np.argsort(take, axis=1)
        idxs[start:stop] = np.take_along_axis(part, order, axis=1)
        dists[start:stop] = np.sqrt(np.take_along_axis(take, order, axis=1))
    return dists, idxs


def energy_idw(
    xy,
    z,
    ngrid=190,
    k=6,
    power=4.0,
    pad=0.10,
    cutoff=None,
    pin=None,
    n_ind: int = 110,
    n_low: int = 50,
):
    """Inducing-point IDW of E-E_GM. Disk support, wells pinned."""
    finite = np.isfinite(xy).all(1)
    xy_f, z_f = xy[finite], z[finite]
    pts, val = inducing(xy_f, z_f, n_ind=n_ind, n_low=n_low)
    xmin, xmax = float(xy_f[:, 0].min()), float(xy_f[:, 0].max())
    ymin, ymax = float(xy_f[:, 1].min()), float(xy_f[:, 1].max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= pad * dx
    xmax += pad * dx
    ymin -= pad * dy
    ymax += pad * dy
    gx = np.linspace(xmin, xmax, ngrid)
    gy = np.linspace(ymin, ymax, ngrid)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    # support is distance to any data site, not just inducing
    data_pts, _ = unique_lowest(xy_f, z_f)
    d_data, _ = knn(data_pts, grid, k=1)
    d_ind, idx = knn(pts, grid, k=min(k, len(pts)))
    hit = d_ind[:, 0] < 1e-14
    w = 1.0 / np.clip(d_ind, 1e-12, None) ** power
    w /= w.sum(axis=1, keepdims=True)
    field = (w * val[idx]).sum(axis=1)
    field[hit] = val[idx[hit, 0]]
    field = field.reshape(ngrid, ngrid)
    nn = d_data[:, 0].reshape(ngrid, ngrid)
    span = max(dx, dy)
    if cutoff is None:
        nn_data = knn(data_pts, data_pts, k=2)[0][:, 1]
        pos = nn_data[nn_data > 1e-12]
        cutoff = max(5.0 * float(np.median(pos)) if pos.size else 0.09 * span, 0.09 * span)
        # stitch ico to the main body if the gap is only one island
        cutoff = min(max(cutoff, 0.10 * span), 0.13 * span)
    field = np.where(nn < cutoff, np.clip(field, 0.0, EMAX), np.nan)
    if pin:
        for pt, ev in pin:
            r2 = (xx - pt[0]) ** 2 + (yy - pt[1]) ** 2
            near = r2 <= (0.018 * span) ** 2
            field = np.where(near, float(ev), field)
    return gx, gy, field, float(cutoff)


def field_at(gx, gy, zz, p) -> float:
    j = int(np.argmin(np.abs(gx - p[0])))
    i = int(np.argmin(np.abs(gy - p[1])))
    return float(zz[i, j])


def local_minima(field: np.ndarray, vmax: float = 2.2, sep: int = 10):
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


def downhill(field, i, j):
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


def mask_components(mask: np.ndarray) -> int:
    ny, nx = mask.shape
    parent = {}

    def key(i, j):
        return i * nx + j

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    cells = []
    for i in range(ny):
        for j in range(nx):
            if mask[i, j]:
                k = key(i, j)
                parent[k] = k
                cells.append((i, j, k))
    for i, j, k in cells:
        if j + 1 < nx and mask[i, j + 1]:
            union(k, key(i, j + 1))
        if i + 1 < ny and mask[i + 1, j]:
            union(k, key(i + 1, j))
    if not cells:
        return 0
    return len({find(k) for _, _, k in cells})


def score_fill(name, xy, energy, gx, gy, field):
    gm, ico = xy[GM], xy[ICO]
    sep = float(np.linalg.norm(gm - ico))
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    sep_norm = sep / max(diam, 1e-12)
    e_gm = field_at(gx, gy, field, gm)
    e_ico = field_at(gx, gy, field, ico)
    mins = local_minima(field)
    ig = int(np.argmin(np.abs(gy - gm[1])))
    jg = int(np.argmin(np.abs(gx - gm[0])))
    ii = int(np.argmin(np.abs(gy - ico[1])))
    ji = int(np.argmin(np.abs(gx - ico[0])))
    ag_i, ag_j, ag_v = downhill(field, ig, jg)
    ai_i, ai_j, ai_v = downhill(field, ii, ji)
    same = abs(ag_i - ai_i) + abs(ag_j - ai_j) <= 4
    two_wells = len(mins) >= 2
    two_basins = two_wells and not same
    hop = abs(ag_i - ig) + abs(ag_j - jg)
    finite = field[np.isfinite(field)]
    deep = bool(np.isfinite(e_gm) and finite.size and e_gm <= np.percentile(finite, 25))
    near_min = any(abs(i - ig) + abs(j - jg) <= 12 for _, i, j in mins)
    gm_in_well = (hop <= 12 or near_min) and deep
    ncomp = mask_components(np.isfinite(field))
    rec = {
        "name": name,
        "sep": sep,
        "sep_norm": sep_norm,
        "diam": diam,
        "E_GM": e_gm,
        "E_ico": e_ico,
        "n_wells": len(mins),
        "two_wells": bool(two_wells),
        "two_basins": bool(two_basins),
        "gm_in_well": bool(gm_in_well),
        "gm_deeper": bool(np.isfinite(e_gm) and np.isfinite(e_ico) and e_gm < e_ico - 1e-6),
        "gm_hop": int(hop),
        "n_components": int(ncomp),
        "connected": bool(ncomp == 1),
        "well_vals": [float(v) for v, _, _ in mins[:8]],
        "n_finite": int(np.isfinite(field).sum()),
        "pass": bool(sep_norm > SEP_BAR and two_basins and gm_in_well),
    }
    return rec


def rank(rec):
    # two basins + GM in a well first. Prefer 1-2 components (ico joined
    # to the cloud, GM may stay a satellite). Reject empty-space bridges
    # that mint dozens of wells.
    e_gm_ok = rec["E_GM"] < 0.25
    e_ico_ok = 0.2 < rec["E_ico"] < 1.6
    ncomp = rec["n_components"]
    compact = 1 if ncomp in (1, 2) else 0
    return (
        int(rec["pass"]),
        int(rec["gm_in_well"]),
        int(rec["two_basins"]),
        int(rec["gm_deeper"]),
        int(e_gm_ok),
        int(e_ico_ok),
        compact,
        -ncomp,
        -abs(rec["n_wells"] - 2),
        rec["sep_norm"],
    )


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


def paint(ax, gx, gy, field):
    xx, yy = np.meshgrid(gx, gy)
    mesh = ax.contourf(
        xx, yy, field, levels=np.linspace(0, EMAX, 21), cmap=PES, extend="max"
    )
    finite = np.where(np.isfinite(field), field, np.nan)
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


def save_fig(xy, gx, gy, field, dest: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.0), facecolor="white")
    mesh = paint(ax, gx, gy, field)
    mark(ax, xy[GM], xy[ICO])
    ax.set_xlabel(r"$s_1$  (landmark MDS)")
    ax.set_ylabel(r"$s_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=170, facecolor="white")
    plt.close(fig)


def unique_landmark_cliques(D: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Representatives of D=0 cliques (permutational copies)."""
    n = D.shape[0]
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        hits = np.flatnonzero(D[i, i + 1 :] < eps)
        for j in hits:
            ri, rj = find(i), find(i + 1 + int(j))
            if ri != rj:
                parent[rj] = ri
    labs = np.array([find(i) for i in range(n)], dtype=int)
    reps, seen = [], set()
    for i, r in enumerate(labs):
        if int(r) not in seen:
            seen.add(int(r))
            reps.append(i)
    return np.asarray(reps, dtype=int), labs


def orient_mds(xy: np.ndarray, i_gm: int, i_ico: int) -> np.ndarray:
    out = xy.copy()
    if out[i_gm, 0] > out[i_ico, 0]:
        out[:, 0] *= -1.0
    if out[i_gm, 1] > 0.0:
        out[:, 1] *= -1.0
    return out


def mds_from_dll(energy: np.ndarray, unique: bool = True):
    """Torgerson of D_ll, optionally after collapsing D=0 copies."""
    n = len(energy)
    gm_idx = int(np.argmin(energy))
    ico_idx = int(np.argmin(np.abs(energy - ICO_E)))
    land_path = DEST / "landmarks.idx"
    dll_path = DEST / "D_ll.dist"
    plane_path = DEST / "d_gm_ico.xy"
    if not (dll_path.is_file() and land_path.is_file()):
        raise SystemExit("need D_ll.dist + landmarks.idx in %s" % DEST)
    land = np.loadtxt(land_path, dtype=int)
    D_ll = np.loadtxt(dll_path)
    if D_ll.shape[0] != len(land):
        raise SystemExit("D_ll %s vs landmarks %s" % (D_ll.shape, land.shape))
    if unique:
        reps, labs = unique_landmark_cliques(D_ll)
        Du = D_ll[np.ix_(reps, reps)]
        xy_u, ev_w = torgerson(Du, 2)
        li_gm = int(np.where(land[reps] == gm_idx)[0][0])
        li_ico = int(np.where(land[reps] == ico_idx)[0][0])
        xy_u = orient_mds(xy_u, li_gm, li_ico)
        xy_l = np.empty((len(land), 2))
        root = {int(labs[r]): k for k, r in enumerate(reps)}
        for i, r in enumerate(labs):
            xy_l[i] = xy_u[root[int(r)]]
        print("unique landmarks", len(reps), "of", len(land), "eigs", [float(x) for x in ev_w])
    else:
        xy_l, ev_w = torgerson(D_ll, 2)
        li_gm = int(np.where(land == gm_idx)[0][0])
        li_ico = int(np.where(land == ico_idx)[0][0])
        xy_l = orient_mds(xy_l, li_gm, li_ico)
        print("full landmarks", len(land), "eigs", [float(x) for x in ev_w])
    if plane_path.is_file():
        plane = np.loadtxt(plane_path)
        xy = nystrom_idw(plane, land, xy_l, k=8)
        xy[land] = xy_l
    else:
        xy = np.full((n, 2), np.nan)
        xy[land] = xy_l
    return xy, [float(x) for x in ev_w], land


def load_or_compute_mds(energy: np.ndarray) -> tuple[np.ndarray, list, np.ndarray]:
    n = len(energy)
    unique_path = DEST / "mds_unique.xy"
    if unique_path.is_file():
        xy = np.loadtxt(unique_path)
        if xy.shape[0] == n:
            land = (
                np.loadtxt(DEST / "landmarks.idx", dtype=int)
                if (DEST / "landmarks.idx").is_file()
                else np.array([], dtype=int)
            )
            print("cache", unique_path, xy.shape)
            return xy, [], land
    if (DEST / "D_ll.dist").is_file() and (DEST / "landmarks.idx").is_file():
        xy, ev, land = mds_from_dll(energy, unique=True)
        DEST.mkdir(parents=True, exist_ok=True)
        np.savetxt(unique_path, xy, fmt="%.8e")
        if not (DEST / "mds.xy").is_file():
            np.savetxt(DEST / "mds.xy", xy, fmt="%.8e")
        return xy, ev, land
    mds_path = DEST / "mds.xy"
    if mds_path.is_file():
        xy = np.loadtxt(mds_path)
        if xy.shape[0] == n:
            land = (
                np.loadtxt(DEST / "landmarks.idx", dtype=int)
                if (DEST / "landmarks.idx").is_file()
                else np.array([], dtype=int)
            )
            print("cache", mds_path, xy.shape)
            return xy, [], land
    raise SystemExit("need mds.xy or D_ll.dist + landmarks.idx in %s" % DEST)


def main() -> None:
    energy = load_energy()
    gm_idx = int(np.argmin(energy))
    ico_idx = int(np.argmin(np.abs(energy - ICO_E)))
    if gm_idx != GM or ico_idx != ICO:
        raise SystemExit("expected GM=0 ico=40, got %s %s" % (gm_idx, ico_idx))
    z = np.clip(energy - float(energy[gm_idx]), 0.0, None)
    xy, ev, land = load_or_compute_mds(energy)
    if xy.shape[0] != len(energy):
        raise SystemExit("xy %s vs energy %s" % (xy.shape, energy.shape))
    d_gm_path, d_ico_path = DEST / "d_gm.txt", DEST / "d_ico.txt"
    if d_gm_path.is_file() and d_ico_path.is_file():
        d_gm = np.loadtxt(d_gm_path)
        d_ico = np.loadtxt(d_ico_path)
        xy_fill, z_fill = collapse_copies(xy, z, d_gm, d_ico)
    else:
        xy_fill, z_fill = xy, z

    span = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-6))
    pin = ((xy[gm_idx], 0.0), (xy[ico_idx], float(z[ico_idx])))
    print(
        "n",
        len(energy),
        "span",
        span,
        "sep",
        float(np.linalg.norm(xy[gm_idx] - xy[ico_idx])),
        "E_ico",
        float(z[ico_idx]),
        "n_land",
        0 if land is None else len(land),
    )

    nn_pts, _ = unique_lowest(xy_fill, z_fill)
    nn = knn(nn_pts, nn_pts, k=2)[0][:, 1]
    med_nn = float(np.median(nn[nn > 1e-12]))
    print("median nn", med_nn, "0.07*span", 0.07 * span)

    trials = [
        ("ind_k6_p4_auto", dict(k=6, power=4.0, cutoff=None, n_ind=110, n_low=50)),
        ("ind_k6_p4_s09", dict(k=6, power=4.0, cutoff=0.09 * span, n_ind=110, n_low=50)),
        ("ind_k6_p4_s11", dict(k=6, power=4.0, cutoff=0.11 * span, n_ind=110, n_low=50)),
        ("ind_k6_p4_s13", dict(k=6, power=4.0, cutoff=0.13 * span, n_ind=120, n_low=60)),
        ("ind_k8_p4_s11", dict(k=8, power=4.0, cutoff=0.11 * span, n_ind=96, n_low=40)),
        ("ind_k6_p4_s11_dense", dict(k=6, power=4.0, cutoff=0.11 * span, n_ind=160, n_low=80)),
        ("ind_k6_p3_s11", dict(k=6, power=3.0, cutoff=0.11 * span, n_ind=110, n_low=50)),
    ]

    scores = []
    fields = {}
    for name, kw in trials:
        gx, gy, field, cut = energy_idw(xy_fill, z_fill, pin=pin, **kw)
        rec = score_fill(name, xy, energy, gx, gy, field)
        rec["cutoff"] = cut
        rec["k"] = kw["k"]
        rec["power"] = kw["power"]
        scores.append(rec)
        fields[name] = (gx, gy, field)
        print(
            f"{name:22s} cut={cut:.4f} wells={rec['n_wells']} "
            f"basins={rec['two_basins']} gm_well={rec['gm_in_well']} "
            f"comp={rec['n_components']} Egm={rec['E_GM']:.3f} "
            f"Eico={rec['E_ico']:.3f} pass={rec['pass']} "
            f"wellsE={rec['well_vals']}"
        )
        save_fig(
            xy,
            gx,
            gy,
            field,
            DEST / f"mds_{name}.png",
            r"landmark Hungarian MDS  $E-E_{\mathrm{GM}}$",
        )

    ranked = sorted(scores, key=rank, reverse=True)
    best = ranked[0]
    winner = best["name"]
    print("winner", winner, json.dumps({k: best[k] for k in (
        "pass", "gm_in_well", "two_basins", "connected", "n_components",
        "n_wells", "E_GM", "E_ico", "cutoff",
    )}))

    gx, gy, field = fields[winner]
    title = r"landmark Hungarian MDS  $E-E_{\mathrm{GM}}$"
    save_fig(xy, gx, gy, field, FIG, title)
    save_fig(xy, gx, gy, field, DEST / "elja_occ_lj38_hungarian_mds.png", title)
    print("wrote", FIG)
    print("wrote", DEST / "elja_occ_lj38_hungarian_mds.png")

    payload = {
        "n": int(len(energy)),
        "gm_idx": gm_idx,
        "ico_idx": ico_idx,
        "E_GM": float(energy[gm_idx]),
        "E_ico": float(energy[ico_idx]),
        "well_depth_GM": 0.0,
        "well_depth_ico": float(z[ico_idx]),
        "n_landmarks": 0 if land is None else int(len(land)),
        "mds_eigs": ev,
        "winner": best,
        "scores": scores,
        "figure": str(FIG),
        "plane": "landmark Torgerson MDS of Hungarian D_ll",
        "energy_field": "kNN IDW of E-E_GM, no occupancy",
        "occupancy_invert": False,
    }
    (DEST / "mds_scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", DEST / "mds_scores.json")


if __name__ == "__main__":
    main()
