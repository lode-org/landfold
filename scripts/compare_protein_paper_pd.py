#!/usr/bin/env python3
"""Paper figure: hairpin P(D,d) and native-D maps, Ceriotti vs asinh.

Same 1000 landmarks. Row 1 reuses compare_asinh_fig.pd_hist on one
shared axis so the sigmoid wall sits next to the asinh diagonal.
Row 2 is the landmark map coloured by periodic D to the native-like
landmark, one colour scale.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ceriotti-figs"
SKMAP = Path.home() / "Git/Github/HaoZeke/sketchmap/examples/protein"
ASINH = Path("/tmp/landfold-asinh/protein_asinh.ld")
PERIOD = 2.0 * np.pi

spec = importlib.util.spec_from_file_location(
    "compare_asinh_fig", ROOT / "scripts" / "compare_asinh_fig.py"
)
caf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(caf)


def pair_period(x, period=PERIOD):
    n = len(x)
    D = np.empty(n * (n - 1) // 2)
    k = 0
    for i in range(n):
        dx = np.abs(x[i + 1 :] - x[i])
        dx = np.minimum(dx, period - dx)
        D[k : k + len(dx)] = np.sqrt((dx * dx).sum(1))
        k += len(dx)
    return D


def pair_euclid(x):
    n = len(x)
    d = np.empty(n * (n - 1) // 2)
    k = 0
    for i in range(n):
        dh = np.linalg.norm(x[i + 1 :] - x[i], axis=1)
        d[k : k + len(dh)] = dh
        k += len(dh)
    return d


def scores(D, d):
    q25, q75 = np.quantile(D, [0.25, 0.75])
    far = D >= q75
    near = D <= q25
    pear_far = float(np.corrcoef(D[far], d[far])[0, 1])
    spear_far = float(
        np.corrcoef(np.argsort(np.argsort(D[far])), np.argsort(np.argsort(d[far])))[0, 1]
    )
    rm_near = float(np.sqrt(np.mean((d[near] - D[near]) ** 2)))
    return pear_far, spear_far, rm_near


def period_l2(a: np.ndarray, b: np.ndarray, period: float = PERIOD) -> np.ndarray:
    d = a - b
    d = d - period * np.round(d / period)
    return np.sqrt((d * d).sum(axis=1))


def main() -> None:
    if not ASINH.is_file():
        print("need", ASINH)
        raise SystemExit(2)
    hd = np.loadtxt(SKMAP / "lm4.30cv.w01.1")[:, :30]
    pub = np.loadtxt(SKMAP / "lm4.30cv.w01.1.proj")[:, :2]
    ash = np.loadtxt(ASINH)[:, :2]
    D = pair_period(hd)
    d0 = pair_euclid(pub)
    d1 = pair_euclid(ash)
    s0 = scores(D, d0)
    s1 = scores(D, d1)
    print("Ceriotti protein pear_far spear_far rm_near", s0)
    print("asinh protein pear_far spear_far rm_near", s1)

    def xsig(x, sigma, a, b):
        u = np.clip(np.asarray(x, dtype=float) / sigma, 0.0, None)
        return 1.0 - (1.0 + (2.0 ** (a / b) - 1.0) * np.power(u, a)) ** (-b / a)

    def asinh_f(x, sigma=6.0):
        return np.arcsinh(np.asarray(x, dtype=float) / sigma) / (2.0 * np.arcsinh(1.0))

    # Residual space of chi: F(D) vs f(d). Raw D vs d hides asinh because
    # HD D ~ 70 and asinh LD d ~ 12.
    FD0, fd0 = xsig(D, 6.0, 8.0, 8.0), xsig(d0, 6.0, 2.0, 8.0)
    FD1, fd1 = asinh_f(D), asinh_f(d1)
    native = hd[np.argmin(np.linalg.norm(pub, axis=1))]
    color = period_l2(hd, native)
    vlo, vhi = np.quantile(color, [0.02, 0.98])
    print("native color range", float(color.min()), float(color.max()), float(vlo), float(vhi))

    fig = plt.figure(figsize=(10.4, 10.0), dpi=170, facecolor="white")
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.05, 1.0],
        hspace=0.22,
        wspace=0.16,
        left=0.07,
        right=0.98,
        top=0.93,
        bottom=0.10,
    )
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    # Each panel uses its own range. A shared max of ~72 shrinks the
    # asinh cloud to a corner and hides the diagonal.
    caf.pd_hist(
        ax00,
        FD0,
        fd0,
        rf"Ceriotti $F(D),f(d)$  raw far Spearman {s0[1]:.3f}",
    )
    caf.pd_hist(
        ax01,
        FD1,
        fd1,
        rf"asinh $F(D),f(d)$  raw far Spearman {s1[1]:.3f}",
    )
    sc = None
    for ax, xy, title in (
        (ax10, pub, r"Ceriotti $6,8,8$ / $6,2,8$"),
        (ax11, ash, r"asinh $\sigma=6$"),
    ):
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=color,
            s=8,
            cmap="magma_r",
            linewidths=0,
            vmin=vlo,
            vmax=vhi,
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
    cax = fig.add_axes([0.28, 0.035, 0.44, 0.022])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label(r"periodic $D$ to the most native-like landmark", fontsize=9)
    fig.suptitle("beta-hairpin landmarks (Ardevol 2015)", fontsize=12, y=0.98)
    dest = OUT / "protein_paper_pd.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=170, facecolor="white")
    print("wrote", dest)


if __name__ == "__main__":
    main()
