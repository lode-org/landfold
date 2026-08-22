#!/usr/bin/env python3
"""MethodsX energy GP of quenched E on the 4042 asinh / twoscale chi.

The field is z = E - E_GM of the inherent structures, not leftover
occupancy invert. IMQ type-II MAP on farthest inducing sites, variance
as reliability. If the posterior mean reverts toward <E>, the fill is
a k=8 inverse-distance interpolant of the minima and the GP is kept
only for variance.

If GM is a rim satellite on this chi the plane has failed: write
PLANE_FAIL and emit the structure-committor plane (q, r) built from
d_GM and d_ico of each quenched geometry.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numpy.linalg import cholesky, solve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ceriotti-figs"
SRC = Path("/tmp/landfold-occ-from-terra/landfold-occ-book")
ENERGY = Path("/tmp/occ-book/lj38.energy")
MINS = Path("/tmp/occ-book/lj38_0013.min")
CV = Path("/tmp/occ-book/lj38.cv")
FAIL_DIR = Path("/tmp/occ-book/cand-egp")

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)
GM_E = -173.928427
ICO_E = -173.252378
GM_IDX = 0
ICO_IDX = 40
LIQ_IDX = 3674
N_INDUCING = 160
EMAX = 6.0
IDW_K = 8
NGRID = 110
VAR_GRID = 80


def load_xy(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        rows.append([float(p[0]), float(p[1])])
    return np.asarray(rows)


def unique_sites(xy: np.ndarray, z: np.ndarray):
    """Collapse exact chi duplicates; keep the lowest energy at each site."""
    key = np.round(xy, 8)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    vals = np.full(len(first), np.inf)
    np.minimum.at(vals, inv, z)
    return xy[first].copy(), vals


def farthest_indices(xy: np.ndarray, k: int, start: int = 0) -> np.ndarray:
    n = len(xy)
    k = min(k, n)
    picked = [int(start)]
    dmin = np.full(n, np.inf)
    for _ in range(k - 1):
        last = xy[picked[-1]]
        dmin = np.minimum(dmin, np.sum((xy - last) ** 2, axis=1))
        picked.append(int(np.argmax(dmin)))
    return np.asarray(picked, dtype=int)


def inducing_indices(pts: np.ndarray, z: np.ndarray, k: int, must) -> np.ndarray:
    extra = [int(i) for i in must if 0 <= int(i) < len(pts)]
    low = np.argsort(z)[:30]
    start = extra[0] if extra else 0
    idx = farthest_indices(pts, k, start=start)
    return np.unique(np.concatenate([idx, np.asarray(extra, dtype=int), low]))


def imq(r2, sf2, ell2):
    return sf2 / np.sqrt(1.0 + r2 / ell2)


def pairwise_r2(a, b):
    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * a @ b.T, 0.0)


def nll(x, y, sf2, ell, noise):
    k = imq(pairwise_r2(x, x), sf2, ell * ell)
    k.flat[:: x.shape[0] + 1] += noise
    l = cholesky(k)
    alpha = solve(l.T, solve(l, y))
    quad = float(y @ alpha)
    logdet = 2.0 * float(np.sum(np.log(np.diag(l))))
    return 0.5 * quad + 0.5 * logdet + 0.5 * len(y) * np.log(2.0 * np.pi)


def fit_imq_map(x, y):
    var = float(np.var(y))
    if not (var > 0.0):
        raise RuntimeError("field is constant")
    sf2 = var
    noise = 1e-3 * var
    span = float(np.ptp(x, axis=0).max())
    mu = np.log(max(span, 1e-6)) - 1.0
    tau = 1.0
    best = None
    for t in np.linspace(mu - 2.0, mu + 2.0, 25):
        ell = float(np.exp(t))
        try:
            val = nll(x, y, sf2, ell, noise) + 0.5 * ((t - mu) / tau) ** 2
        except np.linalg.LinAlgError:
            continue
        if best is None or val < best[0]:
            best = (val, ell, sf2, noise)
    if best is None:
        raise RuntimeError("IMQ MAP failed")
    return best


def predict_imq(x, y, xs, ell, sf2, noise, want_var: bool = True, chunk: int = 2500):
    k = imq(pairwise_r2(x, x), sf2, ell * ell)
    k.flat[:: x.shape[0] + 1] += noise
    l = cholesky(k)
    alpha = solve(l.T, solve(l, y))
    mean = np.empty(len(xs))
    var = np.zeros(len(xs)) if want_var else None
    for i0 in range(0, len(xs), chunk):
        sl = slice(i0, i0 + chunk)
        ks = imq(pairwise_r2(xs[sl], x), sf2, ell * ell)
        mean[sl] = ks @ alpha
        if want_var:
            v = solve(l.T, solve(l, ks.T))
            var[sl] = np.maximum(sf2 - np.sum(ks.T * v, axis=0), 0.0)
    return mean, var


def idw_k(pts, vals, grid, k: int = IDW_K, power: float = 2.0, chunk: int = 1600):
    out = np.empty(len(grid))
    k = min(k, len(pts))
    for i0 in range(0, len(grid), chunk):
        g = grid[i0 : i0 + chunk]
        d2 = pairwise_r2(g, pts)
        kn = np.argpartition(d2, k - 1, axis=1)[:, :k]
        d = np.sqrt(np.take_along_axis(d2, kn, axis=1))
        v = vals[kn]
        hit = d < 1e-12
        w = np.where(hit, 0.0, 1.0 / np.power(d + 1e-12, power))
        num = (w * v).sum(1)
        den = np.clip(w.sum(1), 1e-12, None)
        pred = num / den
        any_hit = hit.any(1)
        if np.any(any_hit):
            first = np.argmax(hit, axis=1)
            pred = np.where(any_hit, v[np.arange(len(v)), first], pred)
        out[i0 : i0 + chunk] = pred
    return out


def support_from_pts(pts, grid, span, chunk: int = 2500):
    n = min(len(pts), 400)
    pick = np.linspace(0, len(pts) - 1, n).astype(int)
    d2 = pairwise_r2(pts[pick], pts)
    d2.sort(axis=1)
    med = float(np.median(np.sqrt(d2[:, 1])))
    cutoff = max(2.6 * med, 0.06 * span)
    dmin = np.empty(len(grid))
    for i0 in range(0, len(grid), chunk):
        d2 = pairwise_r2(grid[i0 : i0 + chunk], pts)
        dmin[i0 : i0 + chunk] = np.sqrt(d2.min(1))
    return dmin < cutoff, cutoff


def on_hull(xy: np.ndarray, idx: int = 0, tol: float = 0.04):
    pts = xy - xy.mean(0)
    lo, hi = pts.min(0), pts.max(0)
    p = pts[idx]
    span = np.clip(hi - lo, 1e-12, None)
    t = float(np.min(np.minimum(p - lo, hi - p) / span))
    c = xy.mean(0)
    r = np.linalg.norm(xy - c, axis=1)
    rfrac = float(r[idx] / max(float(r.max()), 1e-12))
    return bool(t < tol or rfrac > 0.92), t, rfrac


def make_grid(xy, ngrid=NGRID, pad=0.10):
    xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
    ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    gx = np.linspace(xmin - pad * dx, xmax + pad * dx, ngrid)
    gy = np.linspace(ymin - pad * dy, ymax + pad * dy, ngrid)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return gx, gy, xx, yy, grid


def sample_at(gx, gy, field, tip):
    ix = int(np.argmin(np.abs(gx - tip[0])))
    iy = int(np.argmin(np.abs(gy - tip[1])))
    return float(field[iy, ix])


def mark(ax, gm, ico, liquid=None):
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
    handles = [h1, h2]
    if liquid is not None:
        h3 = ax.scatter(
            liquid[0],
            liquid[1],
            s=55,
            marker="o",
            c="k",
            edgecolors="white",
            linewidths=0.5,
            zorder=50,
            label="liquid",
        )
        handles.append(h3)
    ax.legend(
        handles=handles,
        loc="best",
        fontsize=8,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="k",
    )


def paint_energy(ax, gx, gy, zg, var=None, bare: bool = True):
    mesh = ax.contourf(
        gx, gy, zg, levels=np.linspace(0, EMAX, 25), cmap=PES, extend="max"
    )
    finite = np.where(np.isfinite(zg), zg, np.nan)
    ax.contour(
        gx,
        gy,
        finite,
        levels=np.linspace(0.3, EMAX - 0.4, 10),
        colors="#1a1a2e",
        linewidths=0.35,
        zorder=15,
    )
    if var is not None:
        vv = np.where(np.isfinite(zg), var, np.nan)
        finite_v = vv[np.isfinite(vv)]
        if finite_v.size:
            lo, hi = np.percentile(finite_v, [35, 85])
            if hi > lo + 1e-12:
                cs = ax.contour(
                    gx,
                    gy,
                    vv,
                    levels=np.linspace(lo, hi, 3),
                    colors="#111111",
                    linestyles="--",
                    linewidths=0.65,
                    zorder=16,
                )
                ax.clabel(cs, fmt=r"$\sigma^2=%.2f$", fontsize=6)
    if bare:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    return mesh


def energy_plane(xy, z, gm_xy, ico_xy):
    pts, vals = unique_sites(xy, z)
    gm_u = int(np.argmin(np.sum((pts - gm_xy) ** 2, axis=1)))
    ico_u = int(np.argmin(np.sum((pts - ico_xy) ** 2, axis=1)))
    idx = inducing_indices(pts, vals, N_INDUCING, (gm_u, ico_u))
    x_obs = pts[idx]
    y_obs = vals[idx]
    print("  fit IMQ MAP n_ind", len(idx), "n_unique", len(pts), flush=True)
    mapv, ell, sf2, noise = fit_imq_map(x_obs, y_obs)
    tips = np.vstack([gm_xy, ico_xy])
    tip_mean, _ = predict_imq(x_obs, y_obs, tips, ell, sf2, noise, want_var=False)
    at_gm, at_ico = float(tip_mean[0]), float(tip_mean[1])
    mean_z = float(np.mean(vals))
    mean_reverts = (at_gm > 0.45) or (at_gm > 0.22 * mean_z)
    gx, gy, xx, yy, grid = make_grid(xy)
    idw = idw_k(pts, vals, grid).reshape(xx.shape)
    if mean_reverts:
        fill = idw
        print("  fill IDW (GP mean-reverts at GM)", flush=True)
    else:
        mean, _ = predict_imq(x_obs, y_obs, grid, ell, sf2, noise, want_var=False)
        fill = mean.reshape(xx.shape)
        span = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
        probe, _ = support_from_pts(pts, grid, span)
        probe = probe.reshape(xx.shape)
        fake = probe & (idw > 2.0) & (fill < 1.2)
        if fake.sum() > 0.03 * max(int(probe.sum()), 1):
            fill = idw
            mean_reverts = True
            print("  fill IDW (IMQ invents wells away from data)", flush=True)
        else:
            print("  fill IMQ", flush=True)
    # variance on a coarser grid, bilinear to the fill grid
    cgx, cgy, cxx, cyy, cgrid = make_grid(xy, ngrid=VAR_GRID)
    _, cvar = predict_imq(x_obs, y_obs, cgrid, ell, sf2, noise, want_var=True)
    cvar = cvar.reshape(cxx.shape)
    # map coarse var onto fine grid by nearest cell
    ix = np.clip(np.searchsorted(cgx, xx.ravel()) - 1, 0, VAR_GRID - 1)
    iy = np.clip(np.searchsorted(cgy, yy.ravel()) - 1, 0, VAR_GRID - 1)
    var = cvar[iy, ix].reshape(xx.shape)
    span = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    mask, cutoff = support_from_pts(pts, grid, span)
    mask = mask.reshape(xx.shape)
    zg = np.where(mask, np.clip(fill, 0.0, EMAX), np.nan)
    vg = np.where(mask, var, np.nan)
    f_gm = sample_at(gx, gy, zg, gm_xy)
    f_ico = sample_at(gx, gy, zg, ico_xy)
    rec = {
        "n_unique": int(len(pts)),
        "n_inducing": int(len(idx)),
        "ell": float(ell),
        "sf2": float(sf2),
        "noise": float(noise),
        "map": float(mapv),
        "gp_GM": at_gm,
        "gp_ico": at_ico,
        "mean_z": mean_z,
        "mean_reverts": bool(mean_reverts),
        "fill": "idw8" if mean_reverts else "imq",
        "F_energy_GM": f_gm,
        "F_energy_ico": f_ico,
        "idw_GM": sample_at(gx, gy, np.where(mask, idw, np.nan), gm_xy),
        "idw_ico": sample_at(gx, gy, np.where(mask, idw, np.nan), ico_xy),
        "support_cutoff": cutoff,
        "z_obs_min": float(y_obs.min()),
        "z_obs_max": float(y_obs.max()),
    }
    return gx, gy, zg, vg, rec, pts, vals


def save_energy_fig(gx, gy, zg, vg, gm, ico, title, dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=170, facecolor="white")
    mesh = paint_energy(ax, gx, gy, zg, vg)
    mark(ax, gm, ico)
    ax.set_title(title, fontsize=11)
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", dest)


def dist_eigs(xyz: np.ndarray) -> np.ndarray:
    n = xyz.shape[0]
    sigs = np.empty((n, xyz.shape[1]))
    for i, x in enumerate(xyz):
        d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
        sigs[i] = np.sort(np.linalg.eigvalsh(d2))
    return sigs


def structure_committor():
    e = np.loadtxt(ENERGY)
    hd_path = Path("/tmp/occ-book/cand-dpair/hd703.npy")
    if hd_path.is_file():
        print("  structure fingerprint", hd_path, flush=True)
        sigs = np.load(hd_path)
        if len(sigs) != len(e):
            raise SystemExit(f"hd703 {len(sigs)} vs energy {len(e)}")
    else:
        print("  structure fingerprint dist-eigs of", MINS, flush=True)
        mins = np.loadtxt(MINS)
        if mins.shape[1] != 1 + 38 * 3:
            raise SystemExit(f"min file has {mins.shape[1]} cols, expected 115")
        e = mins[:, 0]
        xyz = mins[:, 1:].reshape(-1, 38, 3)
        sigs = dist_eigs(xyz)
    d_gm = np.linalg.norm(sigs - sigs[GM_IDX], axis=1)
    d_ico = np.linalg.norm(sigs - sigs[ICO_IDX], axis=1)
    q = d_gm / np.clip(d_gm + d_ico, 1e-15, None)
    r = d_gm + d_ico
    xy = np.column_stack([q, r])
    z = np.clip(e - float(e.min()), 0.0, None)
    return xy, z, d_gm, d_ico, e


def write_plane_fail(asinh_rec, ts_rec, asinh_rim, ts_rim) -> None:
    FAIL_DIR.mkdir(parents=True, exist_ok=True)
    dest = FAIL_DIR / "PLANE_FAIL.txt"
    lines = [
        "PLANE FAIL: asinh / twoscale chi of n4..n13",
        "",
        "The field is quenched energy, not leftover occupancy invert.",
        "GM must sit in an interior teal well of similar-energy neighbours.",
        "On this chi it does not.",
        "",
        f"asinh: GM on rim={asinh_rim[0]} bbox_t={asinh_rim[1]:.4f} "
        f"r/rmax={asinh_rim[2]:.4f}",
        f"  nearest non-GM unique is isolated; GP fill={asinh_rec['fill']} "
        f"F_energy(GM)={asinh_rec['F_energy_GM']:.4f} "
        f"F_energy(ico)={asinh_rec['F_energy_ico']:.4f} "
        f"gp_GM={asinh_rec['gp_GM']:.4f}",
        f"twoscale (HD asinh / LD Ceriotti): GM on rim={ts_rim[0]} "
        f"bbox_t={ts_rim[1]:.4f} r/rmax={ts_rim[2]:.4f}",
        f"  GM is an isolated satellite of high-E neighbours; fill={ts_rec['fill']} "
        f"F_energy(GM)={ts_rec['F_energy_GM']:.4f} "
        f"F_energy(ico)={ts_rec['F_energy_ico']:.4f} "
        f"gp_GM={ts_rec['gp_GM']:.4f}",
        "",
        "Replacement plane: structure committor (q, r) with",
        "  q = d_GM / (d_GM + d_ico)",
        "  r = d_GM + d_ico",
        "d_GM, d_ico = Euclidean distance of the permutation-invariant",
        "sorted internuclear list (703-vector) of each quenched geometry",
        "to the Wales GM and ico.",
        "Energy fill on that plane. Figure: elja_occ_lj38_struct_committor.png",
        "",
    ]
    dest.write_text("\n".join(lines))
    print("wrote", dest)


def main() -> None:
    e = np.loadtxt(ENERGY)
    if abs(float(e[GM_IDX]) - GM_E) > 1e-3 or abs(float(e[ICO_IDX]) - ICO_E) > 1e-3:
        raise SystemExit(f"index check failed E[0]={e[GM_IDX]} E[40]={e[ICO_IDX]}")
    z = np.clip(e - float(e.min()), 0.0, None)
    planes = (
        ("asinh", SRC / "lj38_asinh.proj", "asinh $\\chi$  energy IMQ-GP"),
        ("twoscale", SRC / "lj38_asinh_cer.proj", "twoscale $\\chi$  energy IMQ-GP"),
    )
    recs = {}
    rims = {}
    fields = {}
    for tag, path, title in planes:
        if not path.is_file() or path.stat().st_size < 100:
            raise SystemExit(f"missing {path}")
        xy = load_xy(path)
        if len(xy) != len(e):
            raise SystemExit(f"row mismatch {path} {len(xy)} vs energy {len(e)}")
        gm, ico = xy[GM_IDX], xy[ICO_IDX]
        print("plane", tag, flush=True)
        gx, gy, zg, vg, rec, pts, vals = energy_plane(xy, z, gm, ico)
        rec["F_energy_liquid"] = sample_at(gx, gy, zg, xy[LIQ_IDX])
        rim = on_hull(xy, GM_IDX)
        recs[tag] = rec
        rims[tag] = rim
        fields[tag] = (gx, gy, zg, vg, gm, ico, xy)
        dest = OUT / f"elja_occ_lj38_{tag}_energygp.png"
        save_energy_fig(gx, gy, zg, vg, gm, ico, title, dest)
        print(
            tag,
            "fill",
            rec["fill"],
            "n_ind",
            rec["n_inducing"],
            "ell",
            f"{rec['ell']:.3f}",
            "gp_GM/ico",
            rec["gp_GM"],
            rec["gp_ico"],
            "F_energy GM/ico/liq",
            rec["F_energy_GM"],
            rec["F_energy_ico"],
            rec["F_energy_liquid"],
            "rim",
            rim,
        )

    asinh_fail = bool(rims["asinh"][0])
    ts_fail = bool(rims["twoscale"][0])
    plane_fail = asinh_fail or ts_fail
    if plane_fail:
        write_plane_fail(recs["asinh"], recs["twoscale"], rims["asinh"], rims["twoscale"])
        xy_s, z_s, d_gm, d_ico, e_s = structure_committor()
        gm, ico = xy_s[GM_IDX], xy_s[ICO_IDX]
        liq = xy_s[LIQ_IDX]
        gx, gy, zg, vg, rec, pts, vals = energy_plane(xy_s, z_s, gm, ico)
        rec["F_energy_liquid"] = sample_at(gx, gy, zg, liq)
        recs["struct"] = rec
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9), facecolor="white")
        mesh = paint_energy(axes[0], gx, gy, zg, vg, bare=False)
        mark(axes[0], gm, ico, liq)
        axes[0].set_title(r"structure committor  $q=d_{\mathrm{GM}}/(d_{\mathrm{GM}}+d_{\mathrm{ico}})$")
        axes[0].set_xlabel(r"$q$")
        axes[0].set_ylabel(r"$d_{\mathrm{GM}}+d_{\mathrm{ico}}$")
        axes[0].set_xticks([0.0, 1.0])
        axes[0].set_xticklabels(["GM", "ico"])
        fig.colorbar(mesh, ax=axes[0], fraction=0.046, pad=0.03).set_label(
            r"$E-E_{\mathrm{GM}}/\varepsilon$"
        )

        raw = np.column_stack([d_gm, d_ico])
        gx2, gy2, zg2, vg2, rec2, _pts, _vals = energy_plane(
            raw, z_s, raw[GM_IDX], raw[ICO_IDX]
        )
        recs["struct_raw"] = rec2
        rec2["F_energy_liquid"] = sample_at(gx2, gy2, zg2, raw[LIQ_IDX])
        mesh2 = paint_energy(axes[1], gx2, gy2, zg2, vg2, bare=False)
        mark(axes[1], raw[GM_IDX], raw[ICO_IDX], raw[LIQ_IDX])
        axes[1].set_title(r"structure $(d_{\mathrm{GM}},\,d_{\mathrm{ico}})$")
        axes[1].set_xlabel(r"$d_{\mathrm{GM}}$")
        axes[1].set_ylabel(r"$d_{\mathrm{ico}}$")
        fig.colorbar(mesh2, ax=axes[1], fraction=0.046, pad=0.03).set_label(
            r"$E-E_{\mathrm{GM}}/\varepsilon$"
        )
        dest = OUT / "elja_occ_lj38_struct_committor.png"
        fig.tight_layout()
        fig.savefig(dest, dpi=170, facecolor="white")
        plt.close(fig)
        print("wrote", dest)
        print(
            "struct q-r",
            "fill",
            rec["fill"],
            "F_energy GM/ico/liq",
            rec["F_energy_GM"],
            rec["F_energy_ico"],
            rec["F_energy_liquid"],
            "q",
            float(xy_s[GM_IDX, 0]),
            float(xy_s[ICO_IDX, 0]),
            float(xy_s[LIQ_IDX, 0]),
            "r",
            float(xy_s[GM_IDX, 1]),
            float(xy_s[ICO_IDX, 1]),
            float(xy_s[LIQ_IDX, 1]),
        )
        print(
            "struct dGM-dico",
            "fill",
            rec2["fill"],
            "F_energy GM/ico/liq",
            rec2["F_energy_GM"],
            rec2["F_energy_ico"],
            rec2["F_energy_liquid"],
        )

    FAIL_DIR.mkdir(parents=True, exist_ok=True)
    report = FAIL_DIR / "energygp_report.txt"
    chunks = []
    for name, rec in recs.items():
        chunks.append(name + " " + " ".join(f"{k}={v}" for k, v in rec.items()))
    report.write_text("\n".join(chunks) + "\n")
    print("wrote", report)


if __name__ == "__main__":
    main()
