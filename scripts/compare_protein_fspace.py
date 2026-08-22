#!/usr/bin/env python3
"""Hairpin P(F(D), f(d)): the transfer-function result.

Ceriotti's saturating F is 1 on the far tail (Lean sat_far_cannot_tell).
asinh keeps a rank on that tail. Mixed asinh-HD / Ceriotti-LD is the
empirical Spearman winner and is theoretically dirty (F_HD unbounded,
F_LD < 1). This figure is F-space only. Raw D vs d hides the asinh
diagonal because HD D ~ 70.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ceriotti-figs"
SKMAP = Path.home() / "Git/Github/HaoZeke/sketchmap/examples/protein"
ASINH = Path("/tmp/landfold-asinh/protein_asinh.ld")
MIXED = Path("/tmp/landfold-twoscale/protein_asinh_cer.ld")
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
    q75 = np.quantile(D, 0.75)
    far = D >= q75
    spear = float(
        np.corrcoef(np.argsort(np.argsort(D[far])), np.argsort(np.argsort(d[far])))[0, 1]
    )
    pear = float(np.corrcoef(D[far], d[far])[0, 1])
    return pear, spear


def xsig(x, sigma, a, b):
    u = np.clip(np.asarray(x, dtype=float) / sigma, 0.0, None)
    return 1.0 - (1.0 + (2.0 ** (a / b) - 1.0) * np.power(u, a)) ** (-b / a)


def asinh_f(x, sigma=6.0):
    return np.arcsinh(np.asarray(x, dtype=float) / sigma) / (2.0 * np.arcsinh(1.0))


def main() -> None:
    if not ASINH.is_file() or not MIXED.is_file():
        raise SystemExit(f"need {ASINH} and {MIXED}")
    hd = np.loadtxt(SKMAP / "lm4.30cv.w01.1")[:, :30]
    pub = np.loadtxt(SKMAP / "lm4.30cv.w01.1.proj")[:, :2]
    ash = np.loadtxt(ASINH)[:, :2]
    mix = np.loadtxt(MIXED)[:, :2]
    D = pair_period(hd)
    panels = (
        (
            xsig(D, 6.0, 8.0, 8.0),
            xsig(pair_euclid(pub), 6.0, 2.0, 8.0),
            scores(D, pair_euclid(pub)),
            r"Ceriotti $F(D),f(d)$",
        ),
        (
            asinh_f(D),
            asinh_f(pair_euclid(ash)),
            scores(D, pair_euclid(ash)),
            r"asinh $F(D),f(d)$",
        ),
        (
            asinh_f(D),
            xsig(pair_euclid(mix), 6.0, 2.0, 8.0),
            scores(D, pair_euclid(mix)),
            r"HD asinh / LD $6,2,8$",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), facecolor="white")
    for ax, (FD, fd, sc, title) in zip(axes, panels):
        print(title, "pear_far", sc[0], "spear_far", sc[1])
        caf.pd_hist(ax, FD, fd, rf"{title}" + "\n" + rf"far Spearman {sc[1]:.3f}")
    dest = OUT / "protein_fspace_pd.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=180, facecolor="white")
    print("wrote", dest)


if __name__ == "__main__":
    main()
