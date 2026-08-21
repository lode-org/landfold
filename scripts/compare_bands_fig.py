#!/usr/bin/env python3
"""Ceriotti chi vs three-band embed on the same TSE.

Same 3451 ts.all frames. No rattled crystals. Occupancy and xi
colouring. Stars are nearest TSE tips, not the crystals.
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

PROJ = Path("/tmp/ts_bands.proj")
BASE = Path("/tmp/landfold-cmp-base.proj")


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
    ax.scatter(*fa, s=80, marker="*", c="#f4d35e", edgecolors="k", zorder=5, label="TSE tip fcc")
    ax.scatter(*ia, s=80, marker="*", c="#111111", zorder=5, label="TSE tip ico")
    ax.set_title(title)
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, loc="lower left")
    return sc


def main() -> None:
    if not PROJ.is_file():
        print("need", PROJ)
        print(
            "landfold embed -D 10 -d 2 --bands < examples/cosmo-lj38/ts.cv > /tmp/ts_bands.proj"
        )
        raise SystemExit(2)
    base = np.loadtxt(BASE if BASE.is_file() else EX / "out" / "ts.proj")[:, :2]
    bands = np.loadtxt(PROJ)[:, :2]
    ts = np.loadtxt(EX / "ts.all")[:, 2:12]
    refs = np.loadtxt(EX / "out" / "lj38_refs.cv")
    xi = basin_xi(ts, refs[0], refs[1])
    tscv = cf.load_ts_cv(EX / "ts.all")
    fcc0 = cf.match_tip(base, tscv, cf.cn_vector(EX / "lj38_fcc.xyz"))
    ico0 = cf.match_tip(base, tscv, cf.cn_vector(EX / "lj38_ico.xyz"))
    fcc1 = cf.match_tip(bands, tscv, cf.cn_vector(EX / "lj38_fcc.xyz"))
    ico1 = cf.match_tip(bands, tscv, cf.cn_vector(EX / "lj38_ico.xyz"))
    print("xi range", float(xi.min()), float(xi.max()))
    print("Ceriotti tip gap", float(np.linalg.norm(fcc0 - ico0)))
    print("bands tip gap", float(np.linalg.norm(fcc1 - ico1)))
    print("corr s1-xi Ceriotti", float(np.corrcoef(base[:, 0], xi)[0, 1]))
    print("corr s1-xi bands", float(np.corrcoef(bands[:, 0], xi)[0, 1]))
    print("corr any-xi Ceriotti", float(max(
        abs(np.corrcoef(base[:, 0], xi)[0, 1]),
        abs(np.corrcoef(base[:, 1], xi)[0, 1]),
    )))
    print("corr any-xi bands", float(max(
        abs(np.corrcoef(bands[:, 0], xi)[0, 1]),
        abs(np.corrcoef(bands[:, 1], xi)[0, 1]),
    )))
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 9.2), dpi=160, facecolor="white")
    occ(axes[0, 0], base, r"Ceriotti $\chi$ occupancy", fcc0, ico0)
    occ(axes[0, 1], bands, r"three-band occupancy", fcc1, ico1)
    xi_scatter(axes[1, 0], base, xi, r"Ceriotti $\chi$ coloured by $\xi$", fcc0, ico0)
    sc = xi_scatter(axes[1, 1], bands, xi, r"three-band coloured by $\xi$", fcc1, ico1)
    fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02).set_label(r"$\xi$")
    dest = OUT / "lj38_bands_vs_ceriotti.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160, facecolor="white")
    print("wrote", dest)


if __name__ == "__main__":
    main()
