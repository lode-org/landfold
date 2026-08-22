#!/usr/bin/env python3
"""SOAP 24-D concatenated with the 40-bin pair-distance histogram.

z-scored mean SOAP (n_r=16, n_a=8) is the connected body. The extra
block is the reduced pair-distance fingerprint (40-bin histogram of
the 703 sorted internuclear distances). That block is z-scored on
its own and scaled by a SOAP:hist weight so it can pull the Wales
funnels apart without drowning the SOAP metric or parking ico on a
ridge.

asinh-L2 Torgerson is the house HD transfer. The painted field is
E-E_GM IDW, never occupancy invert.
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
SOAP_FP = BOOK / "cand-soap" / "fp_mean_r16a8.npy"
SOAP_TXT = BOOK / "cand-soap" / "fp_mean_r16a8.txt"
HD40 = BOOK / "cand-dpair" / "hd40.npy"
HD40_HIST = BOOK / "cand-dpair" / "dpair40.hist"
ENERGY = BOOK / "lj38.energy"
DEST = BOOK / "cand-soap-dpair"
FIG = OUT / "elja_occ_lj38_soap_dpair.png"

GM_E = -173.928427
ICO_E = -173.252378
GM_IDX = 0
ICO_IDX = 40
ICO_TRUE = ICO_E - GM_E
SOAP_SEP = 0.53
# SOAP:hist after per-block z-score and RMS-per-dimension scale.
# (4,1) gives SOAP four times the block RMS of the 40-bin hist.
RATIOS = (
    (1, 0),
    (8, 1),
    (4, 1),
    (3, 1),
    (2, 1),
    (3, 2),
    (1, 1),
    (2, 3),
    (1, 2),
    (1, 4),
)
ASINH_NORM = 2.0 * float(np.arcsinh(1.0))
EMAX = 5.0
PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)


def load_soap() -> np.ndarray:
    if SOAP_FP.is_file():
        soap = np.load(SOAP_FP)
    elif SOAP_TXT.is_file():
        soap = np.loadtxt(SOAP_TXT)
    else:
        raise SystemExit(f"need {SOAP_FP} or {SOAP_TXT}")
    if soap.ndim != 2 or soap.shape[1] != 24:
        raise SystemExit(f"{SOAP_FP}: expected (*, 24), got {soap.shape}")
    return np.asarray(soap, dtype=float)


def load_hist() -> np.ndarray:
    if HD40.is_file():
        hist = np.load(HD40)
    elif HD40_HIST.is_file():
        hist = np.loadtxt(HD40_HIST)
    else:
        raise SystemExit(f"need {HD40} or {HD40_HIST}")
    if hist.ndim != 2 or hist.shape[1] != 40:
        raise SystemExit(f"pair-hist: expected (*, 40), got {hist.shape}")
    return np.asarray(hist, dtype=float)


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / np.clip(x.std(0), 1e-9, None)


def block_scale(x: np.ndarray) -> np.ndarray:
    """Unit-column block with RMS 1 so SOAP:hist ratios are comparable."""
    return x / np.sqrt(float(x.shape[1]))


def pairwise_l2(x: np.ndarray) -> np.ndarray:
    nrm = np.einsum("ij,ij->i", x, x)
    d2 = nrm[:, None] + nrm[None, :] - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2, dtype=float)


def torgerson(dist: np.ndarray, dim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Top-`dim` Torgerson via double-centering + randomized SVD."""
    d2 = dist * dist
    row = d2.mean(1, keepdims=True)
    col = d2.mean(0, keepdims=True)
    b = -0.5 * (d2 - row - col + d2.mean())
    n = b.shape[0]
    p = min(n, dim + 8)
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(b @ rng.standard_normal((n, p)))
    for _ in range(2):
        q, _ = np.linalg.qr(b @ q)
    u, s, _ = np.linalg.svd(b @ q, full_matrices=False)
    return u[:, :dim] * np.sqrt(np.clip(s[:dim], 0.0, None)), s[:6]


def asinh_l2_mds(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    dist = pairwise_l2(x)
    pos = dist[dist > 0]
    sig = float(np.median(pos)) if pos.size else 1.0
    sig = max(sig, 1e-9)
    ash = np.arcsinh(dist / sig) / ASINH_NORM
    np.fill_diagonal(ash, 0.0)
    xy, ev = torgerson(ash, 2)
    return xy, ev, sig


def orient(xy: np.ndarray, i: int, j: int) -> np.ndarray:
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


def energy_idw(xy, z, ngrid=150, k=6):
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
    return gx, gy, field, cutoff


def sample(gx, gy, field, pt) -> float:
    if not np.isfinite(field).any():
        return float("nan")
    ix = int(np.clip(np.searchsorted(gx, pt[0]) - 1, 0, field.shape[1] - 1))
    iy = int(np.clip(np.searchsorted(gy, pt[1]) - 1, 0, field.shape[0] - 1))
    return float(field[iy, ix])


def ring_barrier(xy, gx, gy, field, idx: int, r_in: float, r_out: float) -> float:
    c = xy[idx]
    xx, yy = np.meshgrid(gx, gy)
    r = np.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2)
    ring = field[(r >= r_in) & (r <= r_out)]
    ring = ring[np.isfinite(ring)]
    if ring.size == 0:
        return float("nan")
    return float(np.mean(ring) - sample(gx, gy, field, c))


def n_components(mask: np.ndarray) -> tuple[int, float]:
    """4-connected components of a boolean mask. Returns (n, largest fraction)."""
    ny, nx = mask.shape
    labels = np.zeros((ny, nx), dtype=np.int32)
    n = 0
    sizes = []
    for i in range(ny):
        for j in range(nx):
            if not mask[i, j] or labels[i, j]:
                continue
            n += 1
            stack = [(i, j)]
            labels[i, j] = n
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < ny and 0 <= xx < nx and mask[yy, xx] and not labels[yy, xx]:
                        labels[yy, xx] = n
                        stack.append((yy, xx))
            sizes.append(size)
    tot = int(mask.sum())
    frac = (max(sizes) / tot) if sizes and tot else 0.0
    return n, float(frac)


def local_min_at(gx, gy, field, pt, rad: int = 3) -> bool:
    iy = int(np.clip(np.searchsorted(gy, pt[1]) - 1, 0, field.shape[0] - 1))
    ix = int(np.clip(np.searchsorted(gx, pt[0]) - 1, 0, field.shape[1] - 1))
    v = field[iy, ix]
    if not np.isfinite(v):
        return False
    i0, i1 = max(iy - rad, 0), min(iy + rad + 1, field.shape[0])
    j0, j1 = max(ix - rad, 0), min(ix + rad + 1, field.shape[1])
    nb = field[i0:i1, j0:j1]
    finite = nb[np.isfinite(nb)]
    if finite.size == 0:
        return False
    return bool(v <= float(np.min(finite)) + 1e-9)


def concat_hd(soap_z: np.ndarray, hist_z: np.ndarray, ws: float, wh: float) -> np.ndarray:
    soap_b = block_scale(soap_z) * float(ws)
    if wh <= 0.0:
        return soap_b
    hist_b = block_scale(hist_z) * float(wh)
    return np.concatenate([soap_b, hist_b], axis=1)


def score(name: str, xy: np.ndarray, gx, gy, field, gm: int, ico: int) -> dict:
    sep = float(np.linalg.norm(xy[gm] - xy[ico]))
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    sep_norm = sep / max(diam, 1e-12)
    fgm = sample(gx, gy, field, xy[gm])
    fico = sample(gx, gy, field, xy[ico])
    finite = np.isfinite(field)
    ncomp, big = n_components(finite)
    bar_gm = ring_barrier(xy, gx, gy, field, gm, 0.06 * diam, 0.16 * diam)
    bar_ico = ring_barrier(xy, gx, gy, field, ico, 0.06 * diam, 0.16 * diam)
    ico_well = bool(
        np.isfinite(bar_ico)
        and bar_ico > 0.05
        and local_min_at(gx, gy, field, xy[ico])
    )
    connected = bool(ncomp == 1 or big >= 0.95)
    gm_ok = bool(np.isfinite(fgm) and abs(fgm) <= 0.05)
    ico_ok = bool(np.isfinite(fico) and abs(fico - ICO_TRUE) <= 0.08)
    sep_ok = bool(sep_norm > SOAP_SEP)
    return {
        "name": name,
        "sep": sep,
        "sep_norm": sep_norm,
        "diam": diam,
        "F_GM": None if not np.isfinite(fgm) else float(fgm),
        "F_ico": None if not np.isfinite(fico) else float(fico),
        "nfin": int(finite.sum()),
        "ncomp": int(ncomp),
        "largest_frac": big,
        "connected": connected,
        "gm_barrier": None if not np.isfinite(bar_gm) else float(bar_gm),
        "ico_barrier": None if not np.isfinite(bar_ico) else float(bar_ico),
        "ico_well": ico_well,
        "gm_floor": gm_ok,
        "ico_floor": ico_ok,
        "sep_ok": sep_ok,
        "pass": bool(connected and gm_ok and ico_ok and sep_ok and ico_well),
        "gm": [float(xy[gm, 0]), float(xy[gm, 1])],
        "ico": [float(xy[ico, 0]), float(xy[ico, 1])],
    }


def draw(xy, z, gx, gy, field, gm, ico, dest: Path, title: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor="white")
    mesh = ax.pcolormesh(
        gx, gy, np.clip(field, 0, EMAX), cmap=PES, shading="auto", vmin=0, vmax=EMAX
    )
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=np.clip(z, 0, EMAX),
        s=7,
        cmap=PES,
        vmin=0,
        vmax=EMAX,
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
    ax.legend(
        loc="best",
        fontsize=8,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="k",
    )
    ax.set_xlabel(r"MDS$_1$")
    ax.set_ylabel(r"MDS$_2$")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(mesh, ax=ax, label=r"$E-E_\mathrm{GM}$")
    fig.tight_layout()
    fig.savefig(dest, dpi=180, facecolor="white")
    plt.close(fig)
    print("wrote", dest)


def ratio_name(ws: float, wh: float) -> str:
    if wh <= 0.0:
        return "soap_only"
    if abs(ws - round(ws)) < 1e-9 and abs(wh - round(wh)) < 1e-9:
        return f"s{int(round(ws))}h{int(round(wh))}"
    return f"s{ws:g}h{wh:g}"


def rank_key(rec: dict) -> tuple:
    fgm = rec["F_GM"] if rec["F_GM"] is not None else 9.0
    fico = rec["F_ico"] if rec["F_ico"] is not None else 9.0
    bar = rec["ico_barrier"] if rec["ico_barrier"] is not None else -9.0
    return (
        int(rec["pass"]),
        int(rec["connected"]),
        int(rec["gm_floor"]),
        int(rec["ico_floor"]),
        int(rec["ico_well"]),
        int(rec["sep_ok"]),
        rec["sep_norm"],
        -abs(fgm),
        -abs(fico - ICO_TRUE),
        bar,
    )


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    soap = load_soap()
    hist = load_hist()
    energy = np.loadtxt(ENERGY)
    if soap.shape[0] != energy.shape[0] or hist.shape[0] != energy.shape[0]:
        raise SystemExit(
            f"row mismatch SOAP {soap.shape} hist {hist.shape} energy {energy.shape}"
        )
    gm, ico = GM_IDX, ICO_IDX
    if abs(float(energy[gm]) - GM_E) > 1e-3 or abs(float(energy[ico]) - ICO_E) > 1e-3:
        raise SystemExit(f"index check failed E[gm]={energy[gm]} E[ico]={energy[ico]}")
    z = energy - float(energy[gm])
    soap_z = zscore(soap)
    hist_z = zscore(hist)
    print(
        f"n={len(energy)} SOAP {soap.shape} hist {hist.shape}  "
        f"SOAP_z L2 GM-ico={np.linalg.norm(soap_z[gm] - soap_z[ico]):.4f}  "
        f"hist_z L2 GM-ico={np.linalg.norm(hist_z[gm] - hist_z[ico]):.4f}  "
        f"true ico-GM={ICO_TRUE:.6f}"
    )

    scores = []
    maps = {}
    for ws, wh in RATIOS:
        hd = concat_hd(soap_z, hist_z, ws, wh)
        hd_split = float(np.linalg.norm(hd[gm] - hd[ico]))
        name = ratio_name(ws, wh)
        print(f"{name}: HD {hd.shape} L2 GM-ico={hd_split:.4f}", flush=True)
        xyp = DEST / f"{name}.xy"
        ev = np.zeros(6)
        sig = None
        if xyp.is_file() and xyp.stat().st_size > 100:
            xy = np.loadtxt(xyp)
            print(f"{name}: reuse {xyp}", flush=True)
        else:
            xy, ev, sig = asinh_l2_mds(hd)
            xy = unit_xy(orient(xy, gm, ico))
            np.savetxt(xyp, xy, fmt="%.8e")
        maps[name] = xy
        gx, gy, field, cutoff = energy_idw(xy, z)
        rec = score(name, xy, gx, gy, field, gm, ico)
        rec.update(
            {
                "soap_w": float(ws),
                "hist_w": float(wh),
                "embed": "asinh",
                "hd": int(hd.shape[1]),
                "hd_l2_gm_ico": hd_split,
                "sig": sig,
                "cutoff": cutoff,
                **{f"ev{i}": float(ev[i]) if i < len(ev) else None for i in range(3)},
            }
        )
        scores.append(rec)
        print(
            f"{name:12s} sep_norm={rec['sep_norm']:.4f}  "
            f"F(GM)={rec['F_GM']:.4f}  F(ico)={rec['F_ico']:.4f}  "
            f"ncomp={rec['ncomp']} big={rec['largest_frac']:.3f}  "
            f"conn={rec['connected']} ico_well={rec['ico_well']}  "
            f"bar_ico={rec['ico_barrier']:.3f} pass={rec['pass']}",
            flush=True,
        )
        title = (
            r"SOAP only  asinh MDS  energy IDW"
            if wh <= 0.0
            else rf"SOAP 24-D + pair-hist$_{{40}}$  ${int(ws)}:{int(wh)}$  asinh MDS  energy IDW"
        )
        png = DEST / f"{name}.png"
        if not png.is_file():
            draw(xy, z, gx, gy, field, gm, ico, png, title)

    ranked = sorted(scores, key=rank_key, reverse=True)
    passed = [r for r in ranked if r["pass"] and r["hist_w"] > 0.0]
    if passed:
        winner = passed[0]["name"]
    else:
        # still prefer a weighted asinh map over SOAP-only
        weighted = [r for r in ranked if r["hist_w"] > 0.0]
        winner = (weighted[0]["name"] if weighted else ranked[0]["name"])
        print("no strict pass; taking", winner)
    rec = next(s for s in scores if s["name"] == winner)
    xy = maps[winner]
    gx, gy, field, _cut = energy_idw(xy, z)
    if rec["hist_w"] <= 0.0:
        title = r"SOAP only  asinh MDS  energy IDW"
    else:
        title = (
            rf"SOAP 24-D + pair-hist$_{{40}}$  "
            rf"${int(rec['soap_w'])}:{int(rec['hist_w'])}$  "
            rf"asinh MDS  energy IDW"
        )
    draw(xy, z, gx, gy, field, gm, ico, FIG, title)
    draw(xy, z, gx, gy, field, gm, ico, DEST / "elja_occ_lj38_soap_dpair.png", title)
    payload = {
        "n": int(len(energy)),
        "gm_idx": gm,
        "ico_idx": ico,
        "winner": winner,
        "SOAP_sep_bar": SOAP_SEP,
        "ico_true": ICO_TRUE,
        "note": "energy IDW of SOAP+pair-hist asinh MDS; no occupancy invert",
        "scores": scores,
    }
    (DEST / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"winner {winner}  sep_norm={rec['sep_norm']:.4f}  "
        f"F(GM)={rec['F_GM']:.4f}  F(ico)={rec['F_ico']:.4f}  "
        f"connected={rec['connected']} ico_well={rec['ico_well']} pass={rec['pass']}"
    )


if __name__ == "__main__":
    main()
