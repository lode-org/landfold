#!/usr/bin/env python3
"""Soft-committor dual funnel of the structure-geodesic plane.

Hard assignment (nearer-GM vs nearer-ico) splits the map into two
islands and a white gap. The soft committor

    q = d_GM / (d_GM + d_ico)

is evaluated in cand-geostruc/asinh.xy. After rotating that plane so
the GM-ico chord lies on +x, the layout is

    x = (2q - 1) * (1 + α (E - E_GM) / Δ)
    y = residual_perp * (1 + α (E - E_GM) / Δ)

with Δ = E_ico - E_GM. α is chosen so both wells stay visible and the
barrier stays a filled, connected body. Field is IDW of E - E_GM.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ceriotti-figs"
BOOK = Path("/tmp/occ-book")
CAND = BOOK / "cand-soft-funnel"
ENERGY = BOOK / "lj38.energy"
XY = BOOK / "cand-geostruc" / "asinh.xy"
GM_E = -173.928427
ICO_E = -173.252378
DELTA = ICO_E - GM_E
# Stretch that keeps GM teal and ico in a second well; larger α
# parks ico on the inner shoulder of its basin.
ALPHA = 0.15
ALPHAS = (0.00, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.70)
PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)


def knn(ref, query, k):
    chunk = 400
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


def fill(xy, z, ngrid=160, k=6):
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
    dist, idx = knn(xy, grid, k)
    w = 1.0 / np.clip(dist, 1e-9, None) ** 2
    w /= w.sum(axis=1, keepdims=True)
    field = (w * z[idx]).sum(axis=1).reshape(ngrid, ngrid)
    nn = dist[:, 0].reshape(ngrid, ngrid)
    nn_data = knn(xy, xy, k=2)[0][:, 1]
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


def rotate_gm_ico(xy, gm, ico):
    """Translate GM to the origin and rotate the GM-ico chord onto +x."""
    out = xy - xy[gm]
    vec = out[ico]
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-15:
        return out
    ang = np.arctan2(vec[1], vec[0])
    c, s = np.cos(-ang), np.sin(-ang)
    rot = np.array([[c, -s], [s, c]])
    return out @ rot.T


def soft_funnel(xy0, z, gm, ico, alpha, delta=DELTA):
    d_gm = np.linalg.norm(xy0 - xy0[gm], axis=1)
    d_ico = np.linalg.norm(xy0 - xy0[ico], axis=1)
    q = d_gm / np.clip(d_gm + d_ico, 1e-15, None)
    rotated = rotate_gm_ico(xy0, gm, ico)
    residual = rotated[:, 1]
    scale = 1.0 + alpha * np.clip(z, 0.0, None) / max(delta, 1e-9)
    xy = np.empty_like(xy0)
    xy[:, 0] = (2.0 * q - 1.0) * scale
    xy[:, 1] = residual * scale
    return xy, q, residual, scale


def components(field):
    finite = np.isfinite(field)
    h, w = finite.shape
    lab = np.zeros((h, w), dtype=np.int32)
    n = 0
    for i in range(h):
        for j in range(w):
            if not finite[i, j] or lab[i, j]:
                continue
            n += 1
            stack = [(i, j)]
            lab[i, j] = n
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and finite[ny, nx] and lab[ny, nx] == 0:
                        lab[ny, nx] = n
                        stack.append((ny, nx))
    return lab, n


def score_layout(xy, z, gm, ico, gx, gy, field):
    finite = np.isfinite(field)
    fgm = sample(gx, gy, field, xy[gm])
    fico = sample(gx, gy, field, xy[ico])
    lab, nlab = components(field)
    ix = int(np.clip(np.searchsorted(gx, xy[gm, 0]) - 1, 0, field.shape[1] - 1))
    iy = int(np.clip(np.searchsorted(gy, xy[gm, 1]) - 1, 0, field.shape[0] - 1))
    jx = int(np.clip(np.searchsorted(gx, xy[ico, 0]) - 1, 0, field.shape[1] - 1))
    jy = int(np.clip(np.searchsorted(gy, xy[ico, 1]) - 1, 0, field.shape[0] - 1))
    same = bool(lab[iy, ix] > 0 and lab[iy, ix] == lab[jy, jx])
    x0, x1 = sorted((float(xy[gm, 0]), float(xy[ico, 0])))
    cols = (gx >= x0) & (gx <= x1)
    if np.any(cols):
        gap_frac = 1.0 - float(finite[:, cols].any(axis=0).mean())
    else:
        gap_frac = 1.0
    mid = (xy[:, 0] >= x0) & (xy[:, 0] <= x1)
    mid_z = float(np.median(z[mid])) if np.any(mid) else float("nan")
    # Field on the segment between the motifs: a filled barrier is high.
    nseg = 24
    seg = np.linspace(0.0, 1.0, nseg)
    line = xy[gm] + (xy[ico] - xy[gm])[None, :] * seg[:, None]
    fseg = np.array([sample(gx, gy, field, p) for p in line])
    n_mid_finite = int(np.isfinite(fseg).sum())
    fmid = float(np.nanmedian(fseg)) if n_mid_finite else float("nan")
    score = 0.0
    if same:
        score += 6.0
    score -= 12.0 * gap_frac
    if np.isfinite(fgm) and fgm < 0.35:
        score += 2.0
    if np.isfinite(fico) and fico < 1.40:
        score += 2.0
    if np.isfinite(fmid) and np.isfinite(fgm) and np.isfinite(fico):
        if fmid > max(fgm, fico) + 0.4:
            score += 2.0
    score += 0.5 * (n_mid_finite / nseg)
    nfin = int(finite.sum())
    return {
        "F_GM": fgm,
        "F_ico": fico,
        "F_mid": fmid,
        "same_cc": same,
        "n_cc": int(nlab),
        "gap_frac": gap_frac,
        "nfin": nfin,
        "mid_z": mid_z,
        "n_mid_finite": n_mid_finite,
        "score": score,
        "sep": float(np.linalg.norm(xy[gm] - xy[ico])),
    }


def draw(ax, xy, z, gm, ico, gx, gy, field, title, vmax=5.0):
    mesh = ax.pcolormesh(
        gx, gy, np.clip(field, 0, vmax), cmap=PES, shading="auto", vmin=0, vmax=vmax
    )
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
        s=180,
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
        s=100,
        marker="D",
        c="k",
        edgecolors="white",
        linewidths=0.7,
        zorder=50,
        label=rf"ico ${ICO_E:.3f}$",
    )
    ax.legend(
        loc="upper right",
        fontsize=8,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="k",
    )
    ax.set_xlabel(r"soft-funnel $_1$")
    ax.set_ylabel(r"soft-funnel $_2$")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    return mesh


def main() -> None:
    CAND.mkdir(parents=True, exist_ok=True)
    energy = np.loadtxt(ENERGY)
    gm = int(np.argmin(energy))
    ico = int(np.argmin(np.abs(energy - ICO_E)))
    z = np.clip(energy - float(energy[gm]), 0.0, None)
    xy0 = np.loadtxt(XY)
    if len(xy0) != len(energy):
        raise SystemExit(f"row mismatch {XY} {len(xy0)} vs energy {len(energy)}")

    rows = []
    fields = {}
    for alpha in ALPHAS:
        xy, q, residual, scale = soft_funnel(xy0, z, gm, ico, alpha)
        gx, gy, field = fill(xy, z)
        rec = score_layout(xy, z, gm, ico, gx, gy, field)
        rec["alpha"] = float(alpha)
        rows.append((rec, xy, gx, gy, field, q, residual, scale))
        fields[alpha] = (xy, gx, gy, field, rec)
        print(
            f"alpha={alpha:.2f} score={rec['score']:.2f} same={rec['same_cc']} "
            f"gap={rec['gap_frac']:.3f} n_cc={rec['n_cc']} "
            f"F(GM)={rec['F_GM']:.3f} F(ico)={rec['F_ico']:.3f} "
            f"F(mid)={rec['F_mid']:.3f} sep={rec['sep']:.3f} nfin={rec['nfin']}"
        )

    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.2))
    for ax, (rec, xy, gx, gy, field, _q, _r, _s) in zip(axes.ravel(), rows):
        draw(
            ax,
            xy,
            z,
            gm,
            ico,
            gx,
            gy,
            field,
            rf"$\alpha={rec['alpha']:.2f}$  cc={rec['n_cc']}  gap={rec['gap_frac']:.2f}",
        )
        ax.legend_.remove()
    fig.tight_layout()
    fig.savefig(CAND / "alpha_sweep.png", dpi=140)
    plt.close(fig)

    # Prefer the prescribed ALPHA if it is connected and gap-free; else best score.
    picked = None
    for rec, xy, gx, gy, field, q, residual, scale in rows:
        if abs(rec["alpha"] - ALPHA) < 1e-12:
            picked = (rec, xy, gx, gy, field, q, residual, scale)
            break
    if picked is None or (not picked[0]["same_cc"]) or picked[0]["gap_frac"] > 0.05:
        ranked = sorted(rows, key=lambda t: t[0]["score"], reverse=True)
        picked = ranked[0]
    rec, xy, gx, gy, field, q, residual, scale = picked
    alpha = float(rec["alpha"])

    np.savetxt(CAND / "soft_funnel.xy", xy)
    np.savetxt(CAND / "q.txt", q)
    np.savetxt(CAND / "residual.txt", residual)
    (CAND / "alpha.txt").write_text(f"{alpha:.6f}\n")

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    mesh = draw(
        ax,
        xy,
        z,
        gm,
        ico,
        gx,
        gy,
        field,
        rf"LJ38 structure  soft-committor funnel  $\alpha={alpha:.2f}$  energy IDW",
    )
    fig.colorbar(mesh, ax=ax, label=r"$E-E_\mathrm{GM}$")
    fig.tight_layout()
    dest = OUT / "elja_occ_lj38_soft_funnel.png"
    fig.savefig(dest, dpi=180)
    fig.savefig(CAND / "soft_funnel.png", dpi=180)
    plt.close(fig)

    print(
        f"picked alpha={alpha:.2f} F(GM)={rec['F_GM']:.4f} F(ico)={rec['F_ico']:.4f} "
        f"same_cc={rec['same_cc']} gap={rec['gap_frac']:.4f} n_cc={rec['n_cc']}"
    )
    print(f"q(GM)={q[gm]:.4f} q(ico)={q[ico]:.4f} Δ={DELTA:.6f}")
    print(f"scale GM/ico/p95={scale[gm]:.3f}/{scale[ico]:.3f}/{np.percentile(scale, 95):.3f}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
