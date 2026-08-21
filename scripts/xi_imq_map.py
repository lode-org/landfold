#!/usr/bin/env python3
"""MethodsX basin coordinate on the Ceriotti chi plane, MAP IMQ-GP.

xi = d(x,fcc)/(d(x,fcc)+d(x,ico)) of the n4..n13 counts
(chemparseplot.parse.representation.basin_coordinate). Occupancy invert
is rejected. Lengthscale is type-II MAP with a Gaussian prior on log ell,
the ChemGP train path. Variance is the reliability.
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
OUT = ROOT / "docs" / "ceriotti-figs"
spec = importlib.util.spec_from_file_location("compose_fes", EX / "compose_fes.py")
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)

CMAP = cf.CMAP


def basin_coordinate(desc, a, b):
    da = np.linalg.norm(desc - a, axis=1)
    db = np.linalg.norm(desc - b, axis=1)
    return da / (da + db)


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


def fit_map(x, y):
    var = float(np.var(y))
    sf2 = var
    noise = 1e-3 * var
    span = float(np.ptp(x, axis=0).max())
    mu = np.log(span) - 1.0
    tau = 1.0
    best = None
    for t in np.linspace(mu - 2.0, mu + 2.0, 25):
        ell = float(np.exp(t))
        val = nll(x, y, sf2, ell, noise) + 0.5 * ((t - mu) / tau) ** 2
        if best is None or val < best[0]:
            best = (val, ell, sf2, noise)
    return best


def predict(x, y, xs, ell, sf2, noise):
    k = imq(pairwise_r2(x, x), sf2, ell * ell)
    k.flat[:: x.shape[0] + 1] += noise
    l = cholesky(k)
    alpha = solve(l.T, solve(l, y))
    ks = imq(pairwise_r2(xs, x), sf2, ell * ell)
    mean = ks @ alpha
    v = solve(l.T, solve(l, ks.T))
    var = np.maximum(sf2 - np.sum(ks.T * v, axis=0), 0.0)
    return mean, var


def main() -> None:
    xy = np.loadtxt(EX / "out" / "ts.proj")[:, :2]
    ts = np.loadtxt(EX / "ts.all")[:, 2:12]
    refs = np.loadtxt(EX / "out" / "lj38_refs.cv")
    xi = basin_coordinate(ts, refs[0], refs[1])
    rng = np.random.default_rng(1)
    pick = rng.choice(len(xy), size=min(300, len(xy)), replace=False)
    x = xy[pick]
    y = xi[pick]
    lm = np.loadtxt(EX / "out" / "lj38.lm")[:, :10]
    ld = np.loadtxt(EX / "out" / "lj38.ld")[:, :2]
    ref_ld = []
    for ref, val in ((refs[0], 0.0), (refs[1], 1.0)):
        j = int(np.argmin(np.linalg.norm(lm - ref, axis=1)))
        ref_ld.append(ld[j])
        x = np.vstack([x, ld[j]])
        y = np.concatenate([y, [val]])
    fcc_xy, ico_xy = ref_ld
    _mapv, ell, sf2, noise = fit_map(x, y)
    print(f"MAP ell={ell:.3f} sf2={sf2:.4f} noise={noise:.5f} n={len(y)}")

    gx = np.linspace(xy[:, 0].min() - 1.5, xy[:, 0].max() + 1.5, 80)
    gy = np.linspace(xy[:, 1].min() - 1.5, xy[:, 1].max() + 1.5, 80)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    mean, var = predict(x, y, grid, ell, sf2, noise)
    mean = mean.reshape(xx.shape)
    var = var.reshape(xx.shape)

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=170, facecolor="white")
    mesh = ax.contourf(gx, gy, mean, levels=np.linspace(0.0, 1.0, 21), cmap=CMAP)
    ax.contour(gx, gy, mean, levels=np.linspace(0.1, 0.9, 9), colors="#1a1a2e", linewidths=0.35)
    cs = ax.contour(gx, gy, var, levels=[0.02, 0.08, 0.2], colors="#111111", linestyles="--", linewidths=0.7)
    ax.clabel(cs, fmt=r"$\sigma^2=%.2f$", fontsize=7)
    ax.scatter(*fcc_xy, s=80, marker="*", c="#f4d35e", edgecolors="k", zorder=5, label="fcc")
    ax.scatter(*ico_xy, s=80, marker="*", c="#e63946", edgecolors="k", zorder=5, label="ico")
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="lower left")
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label(r"$\xi$")
    dest = OUT / "lj38_xi_map.png"
    fig.savefig(dest, dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", dest)

    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=170, facecolor="white")
    mesh = ax.contourf(gx, gy, var, levels=20, cmap="magma")
    ax.scatter(*fcc_xy, s=80, marker="*", c="#f4d35e", edgecolors="k", zorder=5, label="fcc")
    ax.scatter(*ico_xy, s=80, marker="*", c="#e63946", edgecolors="k", zorder=5, label="ico")
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="lower left")
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label(r"$\sigma^2(\xi)$")
    dest2 = OUT / "lj38_xi_variance.png"
    fig.savefig(dest2, dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", dest2)


if __name__ == "__main__":
    main()
