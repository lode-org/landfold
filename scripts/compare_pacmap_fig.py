#!/usr/bin/env python3
"""Ceriotti occupancy vs PaCMAP plane with MethodsX xi (not occupancy).

Occupancy on a rank-uniform map is flat by construction. The readable
field is xi = d(fcc)/(d(fcc)+d(ico)) of n4..n13, MAP IMQ-GP.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import cholesky, solve

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "cosmo-lj38"
spec = importlib.util.spec_from_file_location("compose_fes", EX / "compose_fes.py")
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)


def basin_xi(desc, a, b):
    da = np.linalg.norm(desc - a, axis=1)
    db = np.linalg.norm(desc - b, axis=1)
    return da / (da + db)


def imq(r2, sf2, ell2):
    return sf2 / np.sqrt(1.0 + r2 / ell2)


def r2(a, b):
    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * a @ b.T, 0.0)


def fit_predict(x, y, grid):
    var = float(np.var(y))
    sf2 = max(var, 1e-6)
    noise = 1e-3 * sf2
    span = float(np.max(np.ptp(x, axis=0)))
    mu = np.log(max(span, 1e-6)) - 1.0
    best = None
    for t in np.linspace(mu - 2.0, mu + 2.0, 21):
        ell = float(np.exp(t))
        k = imq(r2(x, x), sf2, ell * ell)
        k.flat[:: x.shape[0] + 1] += noise
        try:
            l = cholesky(k)
        except np.linalg.LinAlgError:
            continue
        alpha = solve(l.T, solve(l, y))
        nll = 0.5 * float(y @ alpha) + float(np.sum(np.log(np.diag(l))))
        nll += 0.5 * ((t - mu) ** 2)
        if best is None or nll < best[0]:
            best = (nll, ell, alpha, sf2, noise)
    _nll, ell, alpha, sf2, noise = best
    ks = imq(r2(grid, x), sf2, ell * ell)
    mean = (ks @ alpha).reshape(-1)
    return mean, ell


def panel_occ(ax, xy, title, fa, ia):
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


def panel_xi(ax, xy, xi, title, fa, ia):
    rng = np.random.default_rng(1)
    pick = rng.choice(len(xy), size=min(280, len(xy)), replace=False)
    x = np.vstack([xy[pick], fa, ia])
    y = np.concatenate([xi[pick], [0.0, 1.0]])
    gx = np.linspace(xy[:, 0].min() - 0.5, xy[:, 0].max() + 0.5, 70)
    gy = np.linspace(xy[:, 1].min() - 0.5, xy[:, 1].max() + 0.5, 70)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    mean, _ell = fit_predict(x, y, grid)
    mean = mean.reshape(xx.shape)
    mesh = ax.contourf(gx, gy, mean, levels=np.linspace(0, 1, 21), cmap=cf.CMAP)
    ax.contour(gx, gy, mean, levels=np.linspace(0.1, 0.9, 9), colors="#1a1a2e", linewidths=0.35)
    ax.scatter(*fa, s=80, marker="*", c="#f4d35e", edgecolors="k", zorder=5, label="fcc")
    ax.scatter(*ia, s=80, marker="*", c="#e63946", edgecolors="k", zorder=5, label="ico")
    ax.set_title(title)
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="lower left")
    return mesh


def main() -> None:
    base = np.loadtxt("/tmp/landfold-cmp-base.proj")
    pc = np.loadtxt("/tmp/ts_pacmap.proj")
    refs = np.loadtxt("/tmp/refs_pacmap.ld")
    ts = np.loadtxt(EX / "ts.all")[:, 2:12]
    refs_hd = np.loadtxt(EX / "out" / "lj38_refs.cv")
    xi = basin_xi(ts, refs_hd[0], refs_hd[1])
    tscv = cf.load_ts_cv(EX / "ts.all")
    fcc0 = cf.match_tip(base, tscv, cf.cn_vector(EX / "lj38_fcc.xyz"))
    ico0 = cf.match_tip(base, tscv, cf.cn_vector(EX / "lj38_ico.xyz"))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=170, facecolor="white")
    panel_occ(axes[0], base, r"Ceriotti $\chi$ occupancy", fcc0, ico0)
    mesh = panel_xi(axes[1], pc, xi, r"PaCMAP $W_1$ + MAP IMQ $\xi$", refs[0], refs[1])
    fig.colorbar(mesh, ax=axes, fraction=0.03, pad=0.02).set_label(r"$\xi$")
    dest = ROOT / "docs" / "ceriotti-figs" / "lj38_pacmap_vs_ceriotti.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=170, facecolor="white")
    print("wrote", dest)


if __name__ == "__main__":
    main()
