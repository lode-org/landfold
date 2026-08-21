#!/usr/bin/env python3
"""Exact fcc-ico axis vs Ceriotti TSE occupancy.

ts.all is the transition-state ensemble. Occupancy of those 3451
frames cannot show the crystals: they are not in the file. This
figure puts rattled fcc and ico into the sample, embeds with the
closed-form contrast axis (s1(a)=0, s1(b)=1), and paints the
readable field E = s1(1-s1), minima at the two refs.
"""

from __future__ import annotations

import importlib.util
import os
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
rspec = importlib.util.spec_from_file_location("lj38_refs", ROOT / "scripts" / "lj38_refs.py")
refs_mod = importlib.util.module_from_spec(rspec)
rspec.loader.exec_module(refs_mod)

CLOUD = Path(os.environ.get("AXIS_CLOUD", "/tmp/lj38_axis_cloud.cv"))
META = Path(os.environ.get("AXIS_META", "/tmp/lj38_axis_meta.csv"))
PROJ = Path(os.environ.get("AXIS_PROJ", "/tmp/lj38_axis.proj"))
N_RATTLE = 220
RATTLE = 0.035


def lj_energy(pos: np.ndarray) -> float:
    n = pos.shape[0]
    e = 0.0
    for i in range(n):
        d = np.linalg.norm(pos[i + 1 :] - pos[i], axis=1)
        r6 = (1.0 / d) ** 6
        e += 4.0 * float(np.sum(r6 * r6 - r6))
    return e


def write_cloud() -> None:
    rng = np.random.default_rng(1)
    fcc_xyz = refs_mod.load_xyz(EX / "lj38_fcc.xyz")
    ico_xyz = refs_mod.load_xyz(EX / "lj38_ico.xyz")
    e_fcc = lj_energy(fcc_xyz)
    e_ico = lj_energy(ico_xyz)
    print(f"LJ fcc {e_fcc:.6f} header -173.928427")
    print(f"LJ ico {e_ico:.6f} header -173.252378")
    fcc_cn = []
    ico_cn = []
    e_fcc_r = []
    e_ico_r = []
    for _ in range(N_RATTLE):
        a = fcc_xyz + rng.normal(0.0, RATTLE, fcc_xyz.shape)
        b = ico_xyz + rng.normal(0.0, RATTLE, ico_xyz.shape)
        fcc_cn.append(refs_mod.nc_vector(a))
        ico_cn.append(refs_mod.nc_vector(b))
        e_fcc_r.append(lj_energy(a))
        e_ico_r.append(lj_energy(b))
    fcc_cn = np.asarray(fcc_cn)
    ico_cn = np.asarray(ico_cn)
    ts = np.loadtxt(EX / "ts.all")[:, 2:12]
    refs = np.loadtxt(EX / "out" / "lj38_refs.cv")
    cloud = np.vstack([refs, fcc_cn, ts, ico_cn])
    labels = np.concatenate(
        [
            np.array([0, 1], dtype=int),
            np.full(N_RATTLE, 0, dtype=int),
            np.full(len(ts), 2, dtype=int),
            np.full(N_RATTLE, 1, dtype=int),
        ]
    )
    energy = np.concatenate(
        [
            np.array([e_fcc, e_ico]),
            np.asarray(e_fcc_r),
            np.full(len(ts), np.nan),
            np.asarray(e_ico_r),
        ]
    )
    np.savetxt(CLOUD, cloud, fmt="%.8e")
    np.savetxt(
        META,
        np.column_stack([labels, energy]),
        fmt=["%d", "%.8e"],
        header="label energy  (0=fcc 1=ico 2=tse)",
        comments="# ",
    )
    print("wrote", CLOUD, "rows", len(cloud))
    print("wrote", META)


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
    ax.scatter(*fa, s=90, marker="*", c="#f4d35e", edgecolors="k", zorder=5, label="fcc")
    ax.scatter(*ia, s=90, marker="*", c="#e63946", edgecolors="k", zorder=5, label="ico")
    ax.set_title(title)
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="lower left")
    return mesh


def main() -> None:
    if not (CLOUD.is_file() and META.is_file()):
        write_cloud()
    else:
        print("reuse", CLOUD)
    if not PROJ.is_file():
        print("need", PROJ)
        print(
            "landfold embed -D 10 -d 2 --axis examples/cosmo-lj38/out/lj38_refs.cv"
            f" < {CLOUD} > {PROJ}"
        )
        raise SystemExit(2)
    xy = np.loadtxt(PROJ)
    meta = np.loadtxt(META)
    labels = meta[:, 0].astype(int)
    energy = meta[:, 1]
    cloud = np.loadtxt(CLOUD)
    refs_hd = np.loadtxt(EX / "out" / "lj38_refs.cv")
    xi = basin_xi(cloud, refs_hd[0], refs_hd[1])
    base = np.loadtxt("/tmp/landfold-cmp-base.proj") if Path("/tmp/landfold-cmp-base.proj").is_file() else np.loadtxt(EX / "out" / "ts.proj")[:, :2]
    tscv = cf.load_ts_cv(EX / "ts.all")
    fcc0 = cf.match_tip(base, tscv, cf.cn_vector(EX / "lj38_fcc.xyz"))
    ico0 = cf.match_tip(base, tscv, cf.cn_vector(EX / "lj38_ico.xyz"))

    fcc = labels == 0
    ico = labels == 1
    tse = labels == 2
    print(
        "s1 fcc",
        float(xy[fcc, 0].min()),
        float(xy[fcc, 0].max()),
        "mean",
        float(xy[fcc, 0].mean()),
    )
    print(
        "s1 ico",
        float(xy[ico, 0].min()),
        float(xy[ico, 0].max()),
        "mean",
        float(xy[ico, 0].mean()),
    )
    print(
        "s1 tse",
        float(xy[tse, 0].min()),
        float(xy[tse, 0].max()),
        "mean",
        float(xy[tse, 0].mean()),
    )
    print("refs ld", xy[0], xy[1])
    print("corr s1-xi", float(np.corrcoef(xy[:, 0], xi)[0, 1]))

    pad = 0.08
    gx = np.linspace(-0.08, 1.08, 160)
    gy = np.linspace(xy[:, 1].min() - pad, xy[:, 1].max() + pad, 160)
    xx, yy = np.meshgrid(gx, gy)
    mean = xx * (1.0 - xx)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 9.4), dpi=160, facecolor="white")
    occ(
        axes[0, 0],
        base,
        r"Ceriotti $\chi$ occupancy of ts.all",
        fcc0,
        ico0,
    )
    ax = axes[0, 1]
    ax.scatter(xy[tse, 0], xy[tse, 1], s=4, c="#b8b8c8", linewidths=0, label="TSE")
    ax.scatter(xy[fcc, 0], xy[fcc, 1], s=10, c="#f4d35e", linewidths=0, label="fcc")
    ax.scatter(xy[ico, 0], xy[ico, 1], s=10, c="#e63946", linewidths=0, label="ico")
    ax.scatter(*xy[0], s=110, marker="*", c="#f4d35e", edgecolors="k", zorder=5)
    ax.scatter(*xy[1], s=110, marker="*", c="#111111", edgecolors="w", zorder=5)
    ax.set_title(r"axis map (basins in the sample)")
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    ax.legend(fontsize=8, loc="best")

    ax = axes[1, 0]
    mesh = ax.contourf(gx, gy, mean, levels=np.linspace(0.0, 0.25, 21), cmap=cf.CMAP)
    ax.contour(gx, gy, mean, levels=np.linspace(0.02, 0.24, 12), colors="#1a1a2e", linewidths=0.35)
    ax.scatter(xy[tse, 0], xy[tse, 1], s=3, c="#111111", alpha=0.18, linewidths=0, zorder=2)
    ax.scatter(xy[fcc, 0], xy[fcc, 1], s=8, c="#f4d35e", linewidths=0, zorder=3)
    ax.scatter(xy[ico, 0], xy[ico, 1], s=8, c="#e63946", linewidths=0, zorder=3)
    ax.scatter(*xy[0], s=110, marker="*", c="#f4d35e", edgecolors="k", zorder=5, label="fcc")
    ax.scatter(*xy[1], s=110, marker="*", c="#e63946", edgecolors="k", zorder=5, label="ico")
    ax.set_title(r"$E=s_1(1-s_1)$ (minima at the refs)")
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label(r"$E$")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1, 1]
    known = np.isfinite(energy)
    sc = ax.scatter(
        xy[known, 0],
        xy[known, 1],
        c=energy[known],
        s=12,
        cmap=cf.CMAP,
        linewidths=0,
    )
    ax.scatter(xy[tse, 0], xy[tse, 1], s=3, c="#d0d0d8", linewidths=0, zorder=0)
    ax.scatter(*xy[0], s=110, marker="*", c="#f4d35e", edgecolors="k", zorder=5, label="fcc")
    ax.scatter(*xy[1], s=110, marker="*", c="#e63946", edgecolors="k", zorder=5, label="ico")
    ax.set_title(r"LJ $12$-$6$ of rattled crystals")
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label(r"$E_{\mathrm{LJ}}$")
    ax.legend(fontsize=8, loc="lower left")

    dest = OUT / "lj38_axis_vs_ceriotti.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=160, facecolor="white")
    print("wrote", dest)


if __name__ == "__main__":
    main()
