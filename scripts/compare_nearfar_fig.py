#!/usr/bin/env python3
"""Ceriotti chi vs split-isometry (identity near + identity far + Riesz)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "cosmo-lj38"
spec = importlib.util.spec_from_file_location("compose_fes", EX / "compose_fes.py")
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)


def basin_xi(desc, a, b):
    da = np.linalg.norm(desc - a, axis=1)
    db = np.linalg.norm(desc - b, axis=1)
    return da / (da + db)


def occ(ax, xy, title, fa, ia):
    gx, gy, fes = cf.kde_fes(xy)
    mesh = ax.contourf(
        gx, gy, fes, levels=np.linspace(0, 2, 21), cmap=cf.CMAP, extend="max"
    )
    ax.contour(
        gx, gy, fes, levels=np.linspace(0.15, 1.85, 12), colors="#1a1a2e", linewidths=0.35
    )
    ax.scatter(*fa, s=80, marker="*", c="#f4d35e", edgecolors="k", zorder=5, label="fcc")
    ax.scatter(*ia, s=80, marker="*", c="#e63946", edgecolors="k", zorder=5, label="ico")
    ax.set_title(title)
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="lower left")
    return mesh


def xi_scatter(ax, xy, xi, title, fa, ia):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=xi, s=6, cmap=cf.CMAP, vmin=0, vmax=1, linewidths=0)
    ax.scatter(*fa, s=80, marker="*", c="#f4d35e", edgecolors="k", zorder=5, label="fcc")
    ax.scatter(*ia, s=80, marker="*", c="#111111", zorder=5, label="ico")
    ax.set_title(title)
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="lower left")
    return sc


def main() -> None:
    base = np.loadtxt("/tmp/landfold-cmp-base.proj")
    nf = np.loadtxt("/tmp/ts_nearfar.proj")
    refs = np.loadtxt("/tmp/refs_nearfar.ld")
    ts = np.loadtxt(EX / "ts.all")[:, 2:12]
    refs_hd = np.loadtxt(EX / "out" / "lj38_refs.cv")
    xi = basin_xi(ts, refs_hd[0], refs_hd[1])
    tscv = cf.load_ts_cv(EX / "ts.all")
    fcc0 = cf.match_tip(base, tscv, cf.cn_vector(EX / "lj38_fcc.xyz"))
    ico0 = cf.match_tip(base, tscv, cf.cn_vector(EX / "lj38_ico.xyz"))
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 9.2), dpi=160, facecolor="white")
    occ(axes[0, 0], base, r"Ceriotti $\chi$ occupancy", fcc0, ico0)
    occ(axes[0, 1], nf, r"near-far occupancy", refs[0], refs[1])
    xi_scatter(axes[1, 0], base, xi, r"Ceriotti $\chi$ coloured by $\xi$", fcc0, ico0)
    sc = xi_scatter(axes[1, 1], nf, xi, r"near-far coloured by $\xi$", refs[0], refs[1])
    fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02).set_label(r"$\xi$")
    dest = ROOT / "docs" / "ceriotti-figs" / "lj38_nearfar_vs_ceriotti.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160, facecolor="white")
    print("wrote", dest)


if __name__ == "__main__":
    main()
