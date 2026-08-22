#!/usr/bin/env python3
"""SHEAP radial energy layout of the SOAP plane.

SOAP 2D coords (mean_r16a8_z) keep the structure angle. Radius is a
monotone function of E-E_GM so the GM sits at the origin of a funnel,
ico is a second well along its SOAP ray, and liquid is pushed out.

A second map assigns each point to the nearer of GM and ico in the
SOAP plane and opens two funnels with origins at (-s, 0) and (s, 0).

The filled field is IDW of E-E_GM. Occupancy invert is not used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ceriotti-figs"
BOOK = Path("/tmp/occ-book")
ENERGY = BOOK / "lj38.energy"
SOAP_XY = BOOK / "cand-soap" / "mean_r16a8_z.xy"
DEST = BOOK / "cand-soap"
GM_E = -173.928427
ICO_E = -173.252378
PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)


def knn(ref, query, k):
    chunk = 800
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


def unique_cloud(xy, z):
    """Collapse exact SOAP/SHEAP copies so IDW is not 24x-weighted at GM."""
    key = np.round(xy, 8)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    n = int(inv.max()) + 1
    acc = np.zeros((n, 2), dtype=np.float64)
    ez = np.zeros(n, dtype=np.float64)
    cnt = np.zeros(n, dtype=np.float64)
    np.add.at(acc, inv, xy)
    np.add.at(ez, inv, z)
    np.add.at(cnt, inv, 1.0)
    return acc / cnt[:, None], ez / cnt


def fill(xy, z, ngrid=150, k=6):
    pts, val = unique_cloud(xy, z)
    xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
    ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= 0.08 * dx
    xmax += 0.08 * dx
    ymin -= 0.08 * dy
    ymax += 0.08 * dy
    gx = np.linspace(xmin, xmax, ngrid)
    gy = np.linspace(ymin, ymax, ngrid)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    dist, idx = knn(pts, grid, k)
    w = 1.0 / np.clip(dist, 1e-9, None) ** 2
    w /= w.sum(axis=1, keepdims=True)
    field = (w * val[idx]).sum(axis=1).reshape(ngrid, ngrid)
    nn = dist[:, 0].reshape(ngrid, ngrid)
    nn_data = knn(pts, pts, k=2)[0][:, 1]
    pos = nn_data[nn_data > 1e-12]
    span = max(xmax - xmin, ymax - ymin)
    cutoff = max(4.0 * float(np.median(pos)) if pos.size else 0.08 * span, 0.07 * span)
    field = np.where(nn < cutoff, field, np.nan)
    return gx, gy, field


def sample(gx, gy, field, pt):
    if not np.isfinite(field).any():
        return float("nan")
    ix = int(np.clip(np.searchsorted(gx, pt[0]) - 1, 0, field.shape[1] - 1))
    iy = int(np.clip(np.searchsorted(gy, pt[1]) - 1, 0, field.shape[0] - 1))
    return float(field[iy, ix])


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


def nearest_min(mins, gx, gy, pt):
    if not mins:
        return None, np.inf
    best = None
    best_d = np.inf
    for k, (_v, i, j) in enumerate(mins):
        q = np.array([gx[j], gy[i]])
        d = float(np.linalg.norm(q - pt))
        if d < best_d:
            best_d = d
            best = k
    return best, best_d


def ring_barrier(xy, gx, gy, field, idx: int, r_in: float, r_out: float) -> float:
    c = xy[idx]
    xx, yy = np.meshgrid(gx, gy)
    r = np.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2)
    ring = field[(r >= r_in) & (r <= r_out)]
    ring = ring[np.isfinite(ring)]
    if ring.size == 0:
        return 0.0
    return float(np.mean(ring) - sample(gx, gy, field, c))


def energy_scale(z: np.ndarray, floor: float, alpha: float, clip: float = 1.6) -> np.ndarray:
    zmax = float(np.percentile(z, 95))
    zn = np.clip(z / max(zmax, 1e-9), 0.0, clip)
    return floor + (1.0 - floor) * np.power(zn, alpha)


def unit_hat(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.linalg.norm(vec, axis=1)
    hat = np.zeros_like(vec)
    mask = r > 1e-12
    hat[mask] = vec[mask] / r[mask, None]
    return hat, r


def rotate_hats(hat: np.ndarray, mask: np.ndarray, target: float = 0.5 * np.pi, weights=None) -> np.ndarray:
    """Rotate the masked unit vectors so their circular mean sits at target."""
    ang = np.arctan2(hat[mask, 1], hat[mask, 0])
    if ang.size == 0:
        return hat
    if weights is None:
        w = np.ones(ang.shape)
    else:
        w = np.clip(np.asarray(weights, dtype=float), 1e-9, None)
    w = w / w.sum()
    mid = float(np.arctan2(np.sum(w * np.sin(ang)), np.sum(w * np.cos(ang))))
    d = target - mid
    ca, sa = np.cos(d), np.sin(d)
    rot = np.array([[ca, -sa], [sa, ca]])
    out = hat.copy()
    out[mask] = hat[mask] @ rot.T
    return out


def sheap_radial(xy: np.ndarray, z: np.ndarray, gm: int, floor: float = 0.08, alpha: float = 0.80) -> np.ndarray:
    """SOAP angle from GM; radius is a monotone function of E-E_GM."""
    hat, r = unit_hat(xy - xy[gm])
    rr = np.power(np.clip(z, 0.0, None), alpha)
    rr = (1.0 - floor) * rr + floor * r
    rr[r <= 1e-12] = 0.0
    out = hat * rr[:, None]
    # open the funnel upward so GM sits at the basin tip
    return rotate_hats(out, r > 1e-12, target=0.5 * np.pi, weights=z[r > 1e-12])


def sheap_scale(xy: np.ndarray, z: np.ndarray, gm: int, floor: float = 0.12, alpha: float = 1.0) -> np.ndarray:
    """Translate GM to the origin and stretch each SOAP radius by energy."""
    return (xy - xy[gm]) * energy_scale(z, floor, alpha)[:, None]


def dual_funnel(xy, z, gm, ico, sep=1.55, floor=0.06, alpha=0.55):
    """Voronoi on the SOAP plane; each motif is the origin of an energy funnel."""
    d_gm = np.linalg.norm(xy - xy[gm], axis=1)
    d_ico = np.linalg.norm(xy - xy[ico], axis=1)
    nearer_gm = d_gm <= d_ico
    hat_g, r_g = unit_hat(xy - xy[gm])
    hat_i, r_i = unit_hat(xy - xy[ico])
    hat_g = rotate_hats(hat_g, nearer_gm, weights=np.clip(z[nearer_gm], 0.0, None) + 0.05)
    hat_i = rotate_hats(hat_i, ~nearer_gm, weights=np.clip(z[~nearer_gm], 0.0, None) + 0.05)
    zg = np.clip(z, 0.0, None)
    zi = np.clip(z - float(z[ico]), 0.0, None)
    zmax_g = max(float(np.percentile(zg[nearer_gm], 95)), 1e-9)
    zmax_i = max(float(np.percentile(zi[~nearer_gm], 95)), 1e-9)
    med_g = max(float(np.median(r_g[nearer_gm & (r_g > 1e-12)])), 1e-9)
    med_i = max(float(np.median(r_i[(~nearer_gm) & (r_i > 1e-12)])), 1e-9)
    rr_g = (1.0 - floor) * np.power(zg / zmax_g, alpha) + floor * (r_g / med_g)
    rr_i = (1.0 - floor) * np.power(zi / zmax_i, alpha) + floor * (r_i / med_i)
    rr_g[r_g <= 1e-12] = 0.0
    rr_i[r_i <= 1e-12] = 0.0
    out = np.empty_like(xy)
    out[nearer_gm] = hat_g[nearer_gm] * rr_g[nearer_gm, None] + np.array([-sep, 0.0])
    out[~nearer_gm] = hat_i[~nearer_gm] * rr_i[~nearer_gm, None] + np.array([sep, 0.0])
    return out, nearer_gm


def score_layout(xy, z, gm, ico, gx, gy, field) -> dict:
    fgm = sample(gx, gy, field, xy[gm])
    fico = sample(gx, gy, field, xy[ico])
    mins = local_minima(field)
    ig, dg = nearest_min(mins, gx, gy, xy[gm])
    ii, di = nearest_min(mins, gx, gy, xy[ico])
    finite = field[np.isfinite(field)]
    p15 = float(np.nanpercentile(finite, 15)) if finite.size else np.nan
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    sep = float(np.linalg.norm(xy[gm] - xy[ico]))
    sep_norm = sep / max(diam, 1e-12)
    bar_gm = ring_barrier(xy, gx, gy, field, gm, 0.06 * diam, 0.18 * diam)
    bar_ico = ring_barrier(xy, gx, gy, field, ico, 0.04 * diam, 0.14 * diam)
    two = bool(ig is not None and ii is not None and ig != ii)
    gm_well = bool(np.isfinite(fgm) and fgm <= max(p15, 0.55) and bar_gm > 0.05)
    ico_well = bool(np.isfinite(fico) and fico < 1.6 and bar_ico > 0.02)
    # GM should sit near the centre of its own basin, not on the hull
    lo, hi = xy.min(0), xy.max(0)
    span = np.clip(hi - lo, 1e-12, None)
    t = float(np.min(np.minimum(xy[gm] - lo, hi - xy[gm]) / span))
    gm_center = bool(t > 0.18)
    score = 0.0
    score += 12.0 if gm_well else -8.0
    score += 10.0 if ico_well else -6.0
    score += 12.0 if two else -10.0
    score += 6.0 if gm_center else -8.0
    score += 5.0 * float(np.clip(bar_gm, -0.5, 2.5))
    score += 3.0 * float(np.clip(bar_ico, -0.5, 2.0))
    score += 4.0 * float(np.clip(1.0 - abs(sep_norm - 0.35) / 0.35, 0.0, 1.0))
    if np.isfinite(fgm):
        score += 4.0 * float(np.clip(1.0 - fgm, -1.0, 1.0))
    if np.isfinite(fico):
        score += 2.0 * float(np.clip(1.2 - fico, -1.0, 1.0))
    return {
        "F_GM": fgm,
        "F_ico": fico,
        "n_wells": len(mins),
        "two": two,
        "d_well_gm": float(dg),
        "d_well_ico": float(di),
        "bar_gm": bar_gm,
        "bar_ico": bar_ico,
        "sep_norm": sep_norm,
        "r_ico": sep,
        "gm_center": gm_center,
        "gm_well": gm_well,
        "ico_well": ico_well,
        "score": score,
    }


def draw(xy, z, gm, ico, path: Path, title: str, xlabel: str, ylabel: str) -> dict:
    gx, gy, field = fill(xy, z)
    rec = score_layout(xy, z, gm, ico, gx, gy, field)
    vmax = 5.0
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    mesh = ax.pcolormesh(gx, gy, np.clip(field, 0, vmax), cmap=PES, shading="auto", vmin=0, vmax=vmax)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=np.clip(z, 0, vmax),
        s=7,
        cmap=PES,
        vmin=0,
        vmax=vmax,
        edgecolors="none",
        alpha=0.55,
        zorder=20,
    )
    ax.scatter(
        xy[gm, 0],
        xy[gm, 1],
        s=170,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.7,
        zorder=50,
        label=rf"GM ${GM_E:.3f}$",
    )
    ax.scatter(
        xy[ico, 0],
        xy[ico, 1],
        s=95,
        marker="D",
        c="k",
        edgecolors="white",
        linewidths=0.7,
        zorder=50,
        label=rf"ico ${ICO_E:.3f}$",
    )
    ax.legend(loc="best", fontsize=8, frameon=True, fancybox=False, framealpha=1.0, facecolor="white", edgecolor="k")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(mesh, ax=ax, label=r"$E-E_\mathrm{GM}$")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(
        f"{path.name} score={rec['score']:.2f} F(GM)={rec['F_GM']:.4f} F(ico)={rec['F_ico']:.4f} "
        f"wells={rec['n_wells']} two={rec['two']} gm_well={rec['gm_well']} ico_well={rec['ico_well']} "
        f"gm_center={rec['gm_center']} bar_gm={rec['bar_gm']:.3f} bar_ico={rec['bar_ico']:.3f} "
        f"r_ico={rec['r_ico']:.4f} sep_norm={rec['sep_norm']:.3f}"
    )
    return rec


def search_sheap(xy0, z, gm, ico) -> tuple[np.ndarray, dict]:
    best = None
    trials = []
    for floor in (0.00, 0.08, 0.16):
        for alpha in (0.50, 0.65, 0.80):
            trials.append(("ray", floor, alpha))
    for floor in (0.20, 0.35, 0.50):
        for alpha in (0.70, 1.00):
            trials.append(("scale", floor, alpha))
    for kind, floor, alpha in trials:
        if kind == "ray":
            xy = sheap_radial(xy0, z, gm, floor=floor, alpha=alpha)
        else:
            xy = sheap_scale(xy0, z, gm, floor=floor, alpha=alpha)
        gx, gy, field = fill(xy, z, ngrid=110)
        rec = score_layout(xy, z, gm, ico, gx, gy, field)
        rec["kind"] = kind
        rec["floor"] = floor
        rec["alpha"] = alpha
        print(
            f"  {kind:5s} f={floor:.2f} a={alpha:.2f} score={rec['score']:6.2f} "
            f"Fgm={rec['F_GM']:.3f} Fico={rec['F_ico']:.3f} two={rec['two']} "
            f"gw={rec['gm_well']} iw={rec['ico_well']} ctr={rec['gm_center']} "
            f"bg={rec['bar_gm']:.2f} bi={rec['bar_ico']:.2f} r={rec['r_ico']:.3f}"
        )
        if best is None or rec["score"] > best[1]["score"]:
            best = (xy, rec)
    return best


def search_dual(xy0, z, gm, ico) -> tuple[np.ndarray, np.ndarray, dict]:
    best = None
    for sep in (1.35, 1.55, 1.80):
        for floor in (0.00, 0.08):
            for alpha in (0.45, 0.60):
                xy, nearer = dual_funnel(xy0, z, gm, ico, sep=sep, floor=floor, alpha=alpha)
                gx, gy, field = fill(xy, z, ngrid=110)
                rec = score_layout(xy, z, gm, ico, gx, gy, field)
                rec["sep"] = sep
                rec["floor"] = floor
                rec["alpha"] = alpha
                rec["n_gm"] = int(nearer.sum())
                rec["n_ico"] = int((~nearer).sum())
                print(
                    f"  dual s={sep:.2f} f={floor:.2f} a={alpha:.2f} score={rec['score']:6.2f} "
                    f"Fgm={rec['F_GM']:.3f} Fico={rec['F_ico']:.3f} two={rec['two']} "
                    f"gw={rec['gm_well']} iw={rec['ico_well']} ctr={rec['gm_center']} "
                    f"bg={rec['bar_gm']:.2f} bi={rec['bar_ico']:.2f}"
                )
                if best is None or rec["score"] > best[2]["score"]:
                    best = (xy, nearer, rec)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true", help="grid-search layout parameters")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    energy = np.loadtxt(ENERGY)
    xy0 = np.loadtxt(SOAP_XY)
    if xy0.shape[0] != energy.shape[0]:
        raise SystemExit(f"length mismatch: soap {xy0.shape} energy {energy.shape}")
    gm = int(np.argmin(energy))
    ico = int(np.argmin(np.abs(energy - ICO_E)))
    z = energy - float(energy[gm])
    print(f"n={len(z)} gm={gm} E={energy[gm]:.6f} ico={ico} E={energy[ico]:.6f} soap={SOAP_XY}")

    if args.search:
        print("search SHEAP radial")
        xy, rec = search_sheap(xy0, z, gm, ico)
        print("picked sheap", rec)
        print("search dual-funnel")
        xy_d, nearer, rec_d = search_dual(xy0, z, gm, ico)
        print("picked dual", rec_d)
    else:
        # Baked: SOAP-ray SHEAP (ico is a second well) and energy-radius dual funnels.
        xy = sheap_radial(xy0, z, gm, floor=0.08, alpha=0.80)
        xy_d, nearer = dual_funnel(xy0, z, gm, ico, sep=1.55, floor=0.06, alpha=0.55)
        rec_d = {"n_gm": int(nearer.sum()), "n_ico": int((~nearer).sum())}

    np.savetxt(DEST / "soap_sheap.xy", xy, fmt="%.8e")
    np.savetxt(DEST / "soap_dual.xy", xy_d, fmt="%.8e")
    print(f"n_GM_funnel={rec_d.get('n_gm', int((np.linalg.norm(xy0 - xy0[gm], axis=1) <= np.linalg.norm(xy0 - xy0[ico], axis=1)).sum()))} "
          f"n_ico_funnel={rec_d.get('n_ico', -1)}")

    draw(
        xy,
        z,
        gm,
        ico,
        OUT / "elja_occ_lj38_soap_sheap.png",
        r"SOAP  SHEAP radial  energy IDW",
        r"SHEAP$_1$",
        r"SHEAP$_2$",
    )
    draw(
        xy,
        z,
        gm,
        ico,
        DEST / "elja_occ_lj38_soap_sheap.png",
        r"SOAP  SHEAP radial  energy IDW",
        r"SHEAP$_1$",
        r"SHEAP$_2$",
    )
    draw(
        xy_d,
        z,
        gm,
        ico,
        OUT / "elja_occ_lj38_soap_dual.png",
        r"SOAP  dual-funnel SHEAP  energy IDW",
        r"dual-funnel $_1$",
        r"dual-funnel $_2$",
    )
    draw(
        xy_d,
        z,
        gm,
        ico,
        DEST / "elja_occ_lj38_soap_dual.png",
        r"SOAP  dual-funnel SHEAP  energy IDW",
        r"dual-funnel $_1$",
        r"dual-funnel $_2$",
    )


if __name__ == "__main__":
    main()
