#!/usr/bin/env python3
"""ChemGP of the committor on the landfold plane.

ts.all column 2 is p_B from the course committor analysis. Occupancy
of the TSE is the wrong field. The MethodsX picture is p_B(s1, s2)
plus GP variance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "cosmo-lj38"
OUT = ROOT / "docs" / "ceriotti-figs"


def main() -> None:
    ts = np.loadtxt(EX / "ts.all")
    xy = np.loadtxt(EX / "out" / "ts.proj")
    if ts.shape[0] != xy.shape[0]:
        raise SystemExit(f"row mismatch ts.all {ts.shape[0]} vs ts.proj {xy.shape[0]}")
    pb = ts[:, 1]
    desc = ts[:, 2:12]
    s1, s2 = xy[:, 0], xy[:, 1]
    print(
        "pB",
        f"n={pb.size}",
        f"min={pb.min():.3f}",
        f"max={pb.max():.3f}",
        f"mean={pb.mean():.3f}",
        f"median={np.median(pb):.3f}",
        f"frac0={np.mean(pb == 0):.3f}",
        f"frac1={np.mean(pb == 1):.3f}",
    )
    left = s1 < 0.0
    right = s1 >= 0.0
    print(f"left <0  n={left.sum()}  <pB>={pb[left].mean():.3f}  med={np.median(pb[left]):.3f}")
    print(
        f"right>=0 n={right.sum()}  <pB>={pb[right].mean():.3f}  med={np.median(pb[right]):.3f}"
    )
    # ξ vs fcc/ico: take extreme n6 (col 2 of desc is n6-ish, index 2 of cv is n6)
    # refs: rows nearest the published structures if present, else min/max n6
    n6 = desc[:, 2]
    ref_a = desc[np.argmax(n6)]
    ref_b = desc[np.argmin(n6)]
    da = np.linalg.norm(desc - ref_a, axis=1)
    db = np.linalg.norm(desc - ref_b, axis=1)
    xi = da / (da + db)
    print(f"xi vs pB corr {np.corrcoef(xi, pb)[0, 1]:.3f}")
    print(f"s1 vs pB corr {np.corrcoef(s1, pb)[0, 1]:.3f}")
    print(f"s2 vs pB corr {np.corrcoef(s2, pb)[0, 1]:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=170, facecolor="white")
    sc = ax.scatter(s1, s2, c=pb, s=6, cmap="coolwarm", vmin=0.0, vmax=1.0, linewidths=0)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label(r"$p_B$")
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("LJ38 TSE, committor on the landfold plane")
    fig.tight_layout()
    fig.savefig(OUT / "lj38_committor_scatter.png")
    plt.close(fig)

    from chemparseplot.parse.representation import EnergyRepresentation
    from chemparseplot.plot.representation import plot_energy

    rng = np.random.default_rng(1)
    if pb.size > 400:
        pick = rng.choice(pb.size, size=400, replace=False)
    else:
        pick = np.arange(pb.size)
    # Committor is in [0, 1]. Fit logit, invert, clip. An unbounded
    # value-IMQ of p_B itself walks outside the interval.
    eps = 1e-3
    y_fit = np.log((np.clip(pb[pick], eps, 1.0 - eps)) / (1.0 - np.clip(pb[pick], eps, 1.0 - eps)))
    rep = EnergyRepresentation.from_mapping(
        {
            "schema": "chemparseplot.energy.v1",
            "x": s1[pick],
            "y": s2[pick],
            "energy": y_fit,
            "frame": "landfold",
            "xlabel": r"$s_1$",
            "ylabel": r"$s_2$",
            "metadata": {"field": "committor_logit"},
        }
    )
    fig = plot_energy(rep, method="rbf", clabel=r"$\mathrm{logit}\,p_B$", show_pts=True)
    fig.savefig(OUT / "lj38_committor_chemgp_logit.png")
    plt.close(fig)

    from scipy.interpolate import Rbf

    rbf = Rbf(s1[pick], s2[pick], y_fit, function="multiquadric", smooth=1.0)
    gx = np.linspace(s1.min() - 0.5, s1.max() + 0.5, 160)
    gy = np.linspace(s2.min() - 0.5, s2.max() + 0.5, 140)
    xx, yy = np.meshgrid(gx, gy)
    zz = 1.0 / (1.0 + np.exp(-np.clip(rbf(xx, yy), -20.0, 20.0)))
    fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=170, facecolor="white")
    cf = ax.contourf(xx, yy, zz, levels=16, cmap="coolwarm", vmin=0.0, vmax=1.0)
    ax.scatter(s1[pick], s2[pick], c="k", s=4, linewidths=0, alpha=0.35)
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04).set_label(r"$p_B$")
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("ChemGP of $p_B$ (logit IMQ, inverted)")
    fig.tight_layout()
    fig.savefig(OUT / "lj38_committor_chemgp.png")
    plt.close(fig)
    print("wrote", OUT / "lj38_committor_scatter.png")
    print("wrote", OUT / "lj38_committor_chemgp.png")


if __name__ == "__main__":
    main()
