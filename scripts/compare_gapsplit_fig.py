#!/usr/bin/env python3
"""Ceriotti chi occupancy vs gap-split (psi, s) on the same TSE landmarks."""

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


def _panel(ax, xy, title, fa, ia):
    gx, gy, fes = cf.kde_fes(xy)
    mesh = ax.contourf(
        gx, gy, fes, levels=np.linspace(0, 2, 21), cmap=cf.CMAP, extend="max"
    )
    ax.contour(
        gx, gy, fes, levels=np.linspace(0.15, 1.85, 12), colors="#1a1a2e", linewidths=0.35
    )
    ax.scatter(*fa, s=80, marker="*", c="#f4d35e", edgecolors="k", zorder=5, label="fcc")
    ax.scatter(*ia, s=80, marker="*", c="#e63946", edgecolors="k", zorder=5, label="ico")
    xs = [gx[0], gx[-1], fa[0], ia[0]]
    ys = [gy[0], gy[-1], fa[1], ia[1]]
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
    ax.set_title(title)
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="lower left", frameon=True)
    return mesh


def main() -> None:
    base = np.loadtxt("/tmp/landfold-cmp-base.proj")
    gs = np.loadtxt("/tmp/ts_gap.proj")
    refs = np.loadtxt("/tmp/refs_gap.ld")
    tscv = cf.load_ts_cv(EX / "ts.all")
    fcc0 = cf.match_tip(base, tscv, cf.cn_vector(EX / "lj38_fcc.xyz"))
    ico0 = cf.match_tip(base, tscv, cf.cn_vector(EX / "lj38_ico.xyz"))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=170, facecolor="white")
    _panel(axes[0], base, r"Ceriotti $\chi$", fcc0, ico0)
    mesh = _panel(axes[1], gs, r"gap-split $(\psi,s)$", refs[0], refs[1])
    fig.colorbar(mesh, ax=axes, fraction=0.03, pad=0.02).set_label(r"$F/\varepsilon$")
    dest = ROOT / "docs" / "ceriotti-figs" / "lj38_gapsplit_vs_ceriotti.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=170, facecolor="white")
    print("wrote", dest)


if __name__ == "__main__":
    main()
