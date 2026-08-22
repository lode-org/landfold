#!/usr/bin/env python3
"""SOAP 24-D concatenated with (Q4, Q6), then MDS / asinh MDS.

z-scored mean SOAP (n_r=16, n_a=8) is the connected body. The extra
two coordinates are the Wales/Doye global bond-order pair: fcc sits
at high Q6, incomplete Mackay ico at low global Q6. The Q block is
z-scored on its own and scaled by a weight in {1, 2, 4} so it can
pull the funnels apart without drowning the SOAP metric.

Classical MDS of L2 is PCA of the concatenated matrix. asinh-L2
Torgerson is the house HD transfer. The painted field is E-E_GM
IDW, never occupancy invert.
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
QNPZ = BOOK / "cand-q6" / "q4q6.npz"
QXY = BOOK / "cand-q6" / "q4q6.xy"
ENERGY = BOOK / "lj38.energy"
DEST = BOOK / "cand-soap-q6"
FIG = OUT / "elja_occ_lj38_soap_q6.png"

GM_E = -173.928427
ICO_E = -173.252378
GM_IDX = 0
ICO_IDX = 40
WEIGHTS = (1.0, 2.0, 4.0)
ASINH_NORM = 2.0 * float(np.arcsinh(1.0))
EMAX = 5.0
PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)


def load_q() -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    if QNPZ.is_file():
        z = np.load(QNPZ)
        energy = np.asarray(z["energy"], dtype=float)
        q4 = np.asarray(z["q4"], dtype=float)
        q6 = np.asarray(z["q6"], dtype=float)
        gm = int(z["gm"]) if "gm" in z.files else int(np.argmin(energy))
        ico = int(z["ico"]) if "ico" in z.files else int(np.argmin(np.abs(energy - ICO_E)))
        return energy, q4, q6, gm, ico
    if not QXY.is_file():
        raise SystemExit(f"need {QNPZ} or {QXY}")
    xy = np.loadtxt(QXY)
    if ENERGY.is_file():
        energy = np.loadtxt(ENERGY)
    else:
        raise SystemExit(f"need {ENERGY} when only q4q6.xy is present")
    gm = int(np.argmin(energy))
    ico = int(np.argmin(np.abs(energy - ICO_E)))
    return energy, xy[:, 0], xy[:, 1], gm, ico


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / np.clip(x.std(0), 1e-9, None)


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


def pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xc = x - x.mean(0)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    ev = (s[:6] ** 2) / max(x.shape[0] - 1, 1)
    return xc @ vt[:2].T, ev


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
    return gx, gy, field


def sample(gx, gy, field, pt) -> float:
    if not np.isfinite(field).any():
        return float("nan")
    ix = int(np.clip(np.searchsorted(gx, pt[0]) - 1, 0, field.shape[1] - 1))
    iy = int(np.clip(np.searchsorted(gy, pt[1]) - 1, 0, field.shape[0] - 1))
    return float(field[iy, ix])


def score(name: str, xy: np.ndarray, gx, gy, field, gm: int, ico: int) -> dict:
    sep = float(np.linalg.norm(xy[gm] - xy[ico]))
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    sep_norm = sep / max(diam, 1e-12)
    fgm = sample(gx, gy, field, xy[gm])
    fico = sample(gx, gy, field, xy[ico])
    finite = field[np.isfinite(field)]
    return {
        "name": name,
        "sep": sep,
        "sep_norm": sep_norm,
        "diam": diam,
        "F_GM": None if not np.isfinite(fgm) else float(fgm),
        "F_ico": None if not np.isfinite(fico) else float(fico),
        "nfin": int(finite.size),
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


def rank_key(rec: dict) -> tuple:
    fgm = rec["F_GM"] if rec["F_GM"] is not None else 9.0
    fico = rec["F_ico"] if rec["F_ico"] is not None else 9.0
    # GM floor near 0, ico near its true 0.676, then largest GM-ico split
    return (
        int(fgm <= 0.35),
        int(abs(fico - (ICO_E - GM_E)) < 0.45),
        rec["sep_norm"],
        -abs(fgm),
        -abs(fico - (ICO_E - GM_E)),
    )


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not SOAP_FP.is_file():
        raise SystemExit(f"need {SOAP_FP}")
    soap = np.load(SOAP_FP)
    if soap.ndim != 2 or soap.shape[1] != 24:
        raise SystemExit(f"{SOAP_FP}: expected (*, 24), got {soap.shape}")
    energy, q4, q6, gm, ico = load_q()
    if soap.shape[0] != energy.shape[0]:
        raise SystemExit(f"row mismatch SOAP {soap.shape[0]} vs Q {energy.shape[0]}")
    if abs(float(energy[gm]) - GM_E) > 1e-3 or abs(float(energy[ico]) - ICO_E) > 1e-3:
        raise SystemExit(f"index check failed E[gm]={energy[gm]} E[ico]={energy[ico]}")
    z = energy - float(energy[gm])
    soap_z = zscore(soap)
    q = np.column_stack([q4, q6])
    q_z = zscore(q)
    print(
        f"n={len(energy)} SOAP {soap.shape}  "
        f"Q6 GM/ico={q6[gm]:.4f}/{q6[ico]:.4f}  "
        f"Q4 GM/ico={q4[gm]:.4f}/{q4[ico]:.4f}"
    )
    print(
        f"HD L2 GM-ico  SOAP_z={np.linalg.norm(soap_z[gm] - soap_z[ico]):.4f}  "
        f"Q_z={np.linalg.norm(q_z[gm] - q_z[ico]):.4f}"
    )

    scores = []
    maps = {}
    for w in WEIGHTS:
        hd = np.concatenate([soap_z, w * q_z], axis=1)
        hd_split = float(np.linalg.norm(hd[gm] - hd[ico]))
        for embed in ("mds", "asinh"):
            if embed == "mds":
                xy, ev = pca2(hd)
                extra = {"ev": [float(x) for x in ev], "sig": None}
            else:
                xy, ev, sig = asinh_l2_mds(hd)
                extra = {"ev": [float(x) for x in ev], "sig": sig}
            xy = unit_xy(orient(xy, gm, ico))
            name = f"w{int(w)}_{embed}"
            maps[name] = xy
            np.savetxt(DEST / f"{name}.xy", xy, fmt="%.8e")
            gx, gy, field = energy_idw(xy, z)
            rec = score(name, xy, gx, gy, field, gm, ico)
            rec.update(
                {
                    "weight": float(w),
                    "embed": embed,
                    "hd": int(hd.shape[1]),
                    "hd_l2_gm_ico": hd_split,
                    "sig": extra["sig"],
                    **{f"ev{i}": extra["ev"][i] if i < len(extra["ev"]) else None for i in range(3)},
                }
            )
            scores.append(rec)
            print(
                f"{name:14s} sep_norm={rec['sep_norm']:.4f}  "
                f"F(GM)={rec['F_GM']:.4f}  F(ico)={rec['F_ico']:.4f}  "
                f"hd_l2={hd_split:.3f}  nfin={rec['nfin']}"
            )
            title = rf"SOAP 24-D + $w={int(w)}\,(Q_4,Q_6)$  {embed}  energy IDW"
            draw(xy, z, gx, gy, field, gm, ico, DEST / f"{name}.png", title)

    ranked = sorted(scores, key=rank_key, reverse=True)
    # house figure is asinh unless MDS is the only map with a GM well
    asinh_ok = [r for r in ranked if r["embed"] == "asinh"]
    winner = asinh_ok[0]["name"] if asinh_ok else ranked[0]["name"]
    rec = next(s for s in scores if s["name"] == winner)
    xy = maps[winner]
    gx, gy, field = energy_idw(xy, z)
    title = (
        rf"SOAP 24-D + $w={int(rec['weight'])}\,(Q_4,Q_6)$  "
        rf"{rec['embed']} MDS  energy IDW"
    )
    draw(xy, z, gx, gy, field, gm, ico, FIG, title)
    draw(xy, z, gx, gy, field, gm, ico, DEST / "elja_occ_lj38_soap_q6.png", title)
    payload = {
        "n": int(len(energy)),
        "gm_idx": gm,
        "ico_idx": ico,
        "winner": winner,
        "Q6_GM": float(q6[gm]),
        "Q6_ico": float(q6[ico]),
        "Q4_GM": float(q4[gm]),
        "Q4_ico": float(q4[ico]),
        "scores": scores,
    }
    (DEST / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"winner {winner}  sep_norm={rec['sep_norm']:.4f}  "
        f"F(GM)={rec['F_GM']:.4f}  F(ico)={rec['F_ico']:.4f}"
    )


if __name__ == "__main__":
    main()
