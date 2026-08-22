#!/usr/bin/env python3
"""Energy field on the committor-energy plane.

Occupancy of leftover wells cannot put the GM in the deepest well:
the ico funnel owns more leftover wells. On (q, E) the two Wales
basins are already separate. The readable field is energy (MethodsX
IMQ-GP / lower envelope), so the GM is the deepest well by construction.
Boltzmann occupancy at melting T is a second panel: the same weights
that collapsed on CN-chi cannot collapse here because q pins the funnels.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ceriotti-figs"
EX = ROOT / "examples" / "cosmo-lj38"
HIST = Path("/tmp/occ-book/lj38_decaf_e.hist")
QE = Path("/tmp/occ-book/landscape/qe.xy")
LAP = Path("/tmp/occ-book/landscape/laplace.xy")
ISO = Path("/tmp/occ-book/landscape/isomap.xy")

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)
GM_E = -173.928427
ICO_E = -173.252378
KT = 0.168
TSTAR = 0.18


def _cf():
    spec = importlib.util.spec_from_file_location("compose_fes", EX / "compose_fes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_book(path: Path):
    wells, emin, rows = [], [], []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        wells.append(float(p[1]))
        emin.append(float(p[2]))
        rows.append([float(x) for x in p[3:]])
    return np.asarray(rows), np.asarray(emin), np.asarray(wells)


def load_xy(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def envelope(xy, energy, cf, ngrid=180, sigma=3.0, pad=0.12):
    """Blurred per-cell minimum energy. GM is the deepest well."""
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= pad * dx
    xmax += pad * dx
    ymin -= pad * dy
    ymax += pad * dy
    xe = np.linspace(xmin, xmax, ngrid + 1)
    ye = np.linspace(ymin, ymax, ngrid + 1)
    ix = np.clip(np.digitize(x, xe) - 1, 0, ngrid - 1)
    iy = np.clip(np.digitize(y, ye) - 1, 0, ngrid - 1)
    grid = np.full((ngrid, ngrid), np.inf)
    for i, j, e in zip(iy, ix, energy):
        if e < grid[i, j]:
            grid[i, j] = e
    filled = np.where(np.isfinite(grid), grid, np.nanmax(energy) + 4.0)
    blur = cf._blur2d(filled, sigma=sigma)
    mask = np.isfinite(grid)
    mask = cf._fill_mask_holes(cf._keep_largest(mask) if hasattr(cf, "_keep_largest") else mask)
    # keep a body around observed cells after blur
    body = cf._blur2d(mask.astype(float), sigma=sigma) > 0.05
    rel = blur - float(energy.min())
    rel = np.where(body, np.clip(rel, 0, 8), np.nan)
    gx = 0.5 * (xe[:-1] + xe[1:])
    gy = 0.5 * (ye[:-1] + ye[1:])
    return gx, gy, rel


def boltz_fes(xy, energy, wells, cf, tstar=TSTAR, ngrid=180, sigma=4.0):
    w = wells * np.exp(-(energy - energy.min()) / tstar)
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= 0.12 * dx
    xmax += 0.12 * dx
    ymin -= 0.12 * dy
    ymax += 0.12 * dy
    counts, xe, ye = np.histogram2d(
        x, y, bins=ngrid, range=[[xmin, xmax], [ymin, ymax]], weights=w
    )
    rho = cf._blur2d(counts.T, sigma=sigma)
    gx = 0.5 * (xe[:-1] + xe[1:])
    gy = 0.5 * (ye[:-1] + ye[1:])
    rmax = float(rho.max()) if float(rho.max()) > 0 else 1.0
    mask = cf._fill_mask_holes(rho > 0.004 * rmax)
    fes = np.full_like(rho, np.nan)
    on = mask & (rho > 0)
    fes[on] = -tstar * np.log(np.clip(rho[on] / rmax, 1e-12, 1))
    return gx, gy, np.clip(fes, 0, 2), w


def mark(ax, xy):
    ax.scatter(
        xy[0, 0], xy[0, 1], s=130, marker="*", c="k",
        edgecolors="white", linewidths=0.6, zorder=6, label=rf"GM ${GM_E:.3f}$",
    )
    ax.scatter(
        xy[1, 0], xy[1, 1], s=75, marker="D", c="k",
        edgecolors="white", linewidths=0.6, zorder=6, label=rf"ico ${ICO_E:.3f}$",
    )
    ax.legend(fontsize=8, frameon=True, fancybox=False)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def main() -> None:
    _hist, energy, wells = load_book(HIST)
    cf = _cf()
    planes = []
    if QE.exists():
        planes.append(("qe", load_xy(QE)))
    if LAP.exists():
        planes.append(("laplace", load_xy(LAP)))
    if ISO.exists():
        planes.append(("isomap", load_xy(ISO)))
    if not planes:
        raise SystemExit("no landscape xy yet")

    for name, xy in planes:
        gx_e, gy_e, rel = envelope(xy, energy, cf)
        gx_b, gy_b, fes, w = boltz_fes(xy, energy, wells, cf, tstar=TSTAR)
        f_gm = cf.fes_at(gx_b, gy_b, fes, xy[0])
        f_ico = cf.fes_at(gx_b, gy_b, fes, xy[1])
        # energy at the two sites on the envelope
        e_gm = cf.fes_at(gx_e, gy_e, rel, xy[0])
        e_ico = cf.fes_at(gx_e, gy_e, rel, xy[1])
        print(
            name,
            "sep",
            float(np.linalg.norm(xy[0] - xy[1])),
            "Fboltz GM/ico",
            f_gm,
            f_ico,
            "Eenv GM/ico",
            e_gm,
            e_ico,
            "wGM",
            float(w[0]),
            "wico",
            float(w[1]),
            "wsum",
            float(w.sum()),
        )

        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), facecolor="white")
        m0 = axes[0].contourf(
            gx_e, gy_e, rel, levels=np.linspace(0, 6, 25), cmap=PES, extend="max"
        )
        axes[0].contour(
            gx_e, gy_e, np.where(np.isfinite(rel), rel, np.nan),
            levels=np.linspace(0.3, 5.5, 10), colors="#1a1a2e", linewidths=0.35,
        )
        mark(axes[0], xy)
        axes[0].set_title(rf"{name}  energy envelope  $E-E_{{\mathrm{{GM}}}}$")
        fig.colorbar(m0, ax=axes[0], fraction=0.046, pad=0.03).set_label(
            r"$E-E_{\mathrm{GM}}/\varepsilon$"
        )

        m1 = axes[1].contourf(
            gx_b, gy_b, fes, levels=np.linspace(0, 2, 21), cmap=PES, extend="max"
        )
        axes[1].contour(
            gx_b, gy_b, np.where(np.isfinite(fes), fes, np.nan),
            levels=np.linspace(0.15, 1.85, 10), colors="#1a1a2e", linewidths=0.35,
        )
        mark(axes[1], xy)
        axes[1].set_title(rf"{name}  Boltzmann occupancy  $T^*={TSTAR}$")
        fig.colorbar(m1, ax=axes[1], fraction=0.046, pad=0.03).set_label(r"$F/\varepsilon$")

        dest = OUT / f"elja_occ_lj38_{name}_energy.png"
        fig.tight_layout()
        fig.savefig(dest, dpi=170, facecolor="white")
        plt.close(fig)
        print("wrote", dest)

    # paper panel: qe energy envelope large
    if QE.exists():
        xy = load_xy(QE)
        gx, gy, rel = envelope(xy, energy, cf, ngrid=220, sigma=2.4)
        fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor="white")
        mesh = ax.contourf(
            gx, gy, rel, levels=np.linspace(0, 6, 25), cmap=PES, extend="max"
        )
        ax.contour(
            gx, gy, np.where(np.isfinite(rel), rel, np.nan),
            levels=np.linspace(0.3, 5.5, 12), colors="#1a1a2e", linewidths=0.35,
        )
        mark(ax, xy)
        ax.set_xlabel(r"committor $q$ (GM $\to$ ico)")
        ax.set_ylabel(r"$E-E_{\mathrm{GM}}/\varepsilon$")
        ax.set_xticks([xy[0, 0], xy[1, 0]])
        ax.set_xticklabels(["GM", "ico"])
        ax.set_title("landscape $\\chi$: energy on the committor plane")
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03).set_label(
            r"$E-E_{\mathrm{GM}}/\varepsilon$"
        )
        dest = OUT / "elja_occ_lj38_landscape_win.png"
        fig.tight_layout()
        fig.savefig(dest, dpi=180, facecolor="white")
        plt.close(fig)
        print("wrote", dest)


if __name__ == "__main__":
    main()
