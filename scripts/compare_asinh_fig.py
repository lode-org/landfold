#!/usr/bin/env python3
"""Ceriotti sigmoid vs asinh transfer on the published landmark path.

Same 200 landmarks, same project grid. Quality is the PNAS joint
P(D,d), then TSE occupancy and xi. asinh is not a polish of chi.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "cosmo-lj38"
OUT = ROOT / "docs" / "ceriotti-figs"
spec = importlib.util.spec_from_file_location("compose_fes", EX / "compose_fes.py")
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)

ASINH = Path("/tmp/ts_asinh.proj")
BASE = Path("/tmp/landfold-cmp-base.proj")


def basin_xi(desc, a, b):
    da = np.linalg.norm(desc - a, axis=1)
    db = np.linalg.norm(desc - b, axis=1)
    return da / (da + db)


def pair_d(x):
    n = len(x)
    out = np.empty(n * (n - 1) // 2)
    k = 0
    for i in range(n):
        d = np.linalg.norm(x[i + 1 :] - x[i], axis=1)
        out[k : k + len(d)] = d
        k += len(d)
    return out


def pd_hist(ax, D, d, title):
    mx = max(float(np.quantile(D, 0.99)), float(np.quantile(d, 0.99)))
    H, xe, ye = np.histogram2d(D, d, bins=80, range=[[0, mx], [0, mx]])
    H = H.T
    H = H / H.max() if H.max() > 0 else H
    mesh = ax.pcolormesh(xe, ye, H, cmap=cf.CMAP, shading="auto", vmin=0, vmax=1)
    ax.plot([0, mx], [0, mx], color="#111111", lw=0.6, ls="--")
    ax.set_title(title)
    ax.set_xlabel(r"$D$")
    ax.set_ylabel(r"$d$")
    ax.set_aspect("equal", adjustable="box")
    q25, q75 = np.quantile(D, [0.25, 0.75])
    far = D >= q75
    near = D <= q25
    sp_far = float(np.corrcoef(D[far], d[far])[0, 1]) if far.any() else 0.0
    rm_near = float(np.sqrt(np.mean((d[near] - D[near]) ** 2)))
    ax.text(
        0.04,
        0.96,
        f"far corr {sp_far:.3f}\nnear RMSE {rm_near:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(fc="white", ec="none", alpha=0.8),
    )
    return mesh


def occ(ax, xy, title, fa, ia):
    gx, gy, fes = cf.kde_fes(xy)
    mesh = ax.contourf(
        gx, gy, fes, levels=np.linspace(0, 2, 21), cmap=cf.CMAP, extend="max"
    )
    ax.contour(
        gx, gy, fes, levels=np.linspace(0.15, 1.85, 12), colors="#1a1a2e", linewidths=0.35
    )
    ax.scatter(*fa, s=80, marker="*", c="#f4d35e", edgecolors="k", zorder=5, label="TSE tip fcc")
    ax.scatter(*ia, s=80, marker="*", c="#e63946", edgecolors="k", zorder=5, label="TSE tip ico")
    ax.set_title(title)
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, loc="lower left")
    return mesh


def xi_scatter(ax, xy, xi, title, fa, ia):
    lo, hi = np.quantile(xi, [0.05, 0.95])
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=xi, s=6, cmap=cf.CMAP, vmin=lo, vmax=hi, linewidths=0)
    ax.scatter(*fa, s=80, marker="*", c="#f4d35e", edgecolors="k", zorder=5)
    ax.scatter(*ia, s=80, marker="*", c="#111111", zorder=5)
    ax.set_title(title)
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    ax.set_aspect("equal", adjustable="box")
    return sc


def main() -> None:
    if not ASINH.is_file():
        print("need", ASINH)
        raise SystemExit(2)
    base = np.loadtxt(BASE if BASE.is_file() else EX / "out" / "ts.proj")[:, :2]
    ash = np.loadtxt(ASINH)[:, :2]
    hd = np.loadtxt(EX / "ts.cv")
    D = pair_d(hd)
    d0 = pair_d(base)
    d1 = pair_d(ash)
    ts = np.loadtxt(EX / "ts.all")[:, 2:12]
    refs = np.loadtxt(EX / "out" / "lj38_refs.cv")
    xi = basin_xi(ts, refs[0], refs[1])
    tscv = cf.load_ts_cv(EX / "ts.all")
    fcc0 = cf.match_tip(base, tscv, cf.cn_vector(EX / "lj38_fcc.xyz"))
    ico0 = cf.match_tip(base, tscv, cf.cn_vector(EX / "lj38_ico.xyz"))
    fcc1 = cf.match_tip(ash, tscv, cf.cn_vector(EX / "lj38_fcc.xyz"))
    ico1 = cf.match_tip(ash, tscv, cf.cn_vector(EX / "lj38_ico.xyz"))
    print("Ceriotti tip", float(np.linalg.norm(fcc0 - ico0)))
    print("asinh tip", float(np.linalg.norm(fcc1 - ico1)))
    print(
        "corr any-xi Ceriotti",
        float(max(abs(np.corrcoef(base[:, 0], xi)[0, 1]), abs(np.corrcoef(base[:, 1], xi)[0, 1]))),
    )
    print(
        "corr any-xi asinh",
        float(max(abs(np.corrcoef(ash[:, 0], xi)[0, 1]), abs(np.corrcoef(ash[:, 1], xi)[0, 1]))),
    )
    fig, axes = plt.subplots(3, 2, figsize=(10.4, 13.6), dpi=150, facecolor="white")
    pd_hist(axes[0, 0], D, d0, r"Ceriotti $P(D,d)$")
    pd_hist(axes[0, 1], D, d1, r"asinh $P(D,d)$")
    occ(axes[1, 0], base, r"Ceriotti occupancy", fcc0, ico0)
    occ(axes[1, 1], ash, r"asinh occupancy", fcc1, ico1)
    xi_scatter(axes[2, 0], base, xi, r"Ceriotti $\xi$", fcc0, ico0)
    sc = xi_scatter(axes[2, 1], ash, xi, r"asinh $\xi$", fcc1, ico1)
    fig.colorbar(sc, ax=axes[2, :], fraction=0.03, pad=0.02).set_label(r"$\xi$")
    dest = OUT / "lj38_asinh_vs_ceriotti.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=150, facecolor="white")
    print("wrote", dest)


if __name__ == "__main__":
    main()
