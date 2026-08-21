#!/usr/bin/env python3
"""Energy on the landfold plane, not leftover-well occupancy.

Occupancy invert of leftover-well counts puts the Wales GM on a yellow
island. The field here is E - E_GM of the quenched minima (n4..n13
asinh chi) or of each packing-family champion (DECAF asinh chi).
The GM is the deep well.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import LinearTriInterpolator, Triangulation

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "cosmo-lj38"
OUT = ROOT / "docs" / "ceriotti-figs"
PROJ = Path("/tmp/landfold-occ-from-terra/landfold-occ-book/lj38_asinh.proj")
ENERGY = Path("/tmp/occ-book/lj38.energy")
CV = Path("/tmp/occ-book/lj38.cv")
DECAF_XY = Path("/tmp/occ-book/lj38_decaf_asinh.ld")
DECAF_CER = Path("/tmp/occ-book/lj38_decaf_cer.ld")
DECAF_E = Path("/tmp/occ-book/lj38_decaf_e.hist")

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)
EMAX = 6.0
GM_E = -173.928427
ICO_E = -173.252378


def _cf():
    spec = importlib.util.spec_from_file_location("compose_fes", EX / "compose_fes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_xy(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        rows.append([float(p[0]), float(p[1])])
    return np.asarray(rows)


def energy_grid(xy: np.ndarray, energy: np.ndarray, ngrid: int = 140):
    rel = np.clip(energy - GM_E, 0.0, None)
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= 0.10 * dx
    xmax += 0.10 * dx
    ymin -= 0.10 * dy
    ymax += 0.10 * dy
    gx = np.linspace(xmin, xmax, ngrid)
    gy = np.linspace(ymin, ymax, ngrid)
    xx, yy = np.meshgrid(gx, gy)
    # unique sites: exact CN duplicates collapse
    key = np.round(xy, 8)
    _, uniq = np.unique(key, axis=0, return_index=True)
    pts = xy[uniq]
    vals = rel[uniq]
    tri = Triangulation(pts[:, 0], pts[:, 1])
    interp = LinearTriInterpolator(tri, vals)
    zz = np.asarray(interp(xx, yy), dtype=float)
    zz = np.clip(zz, 0.0, EMAX)
    return gx, gy, zz


def draw(ax, gx, gy, zz, marks, title):
    mesh = ax.contourf(
        gx, gy, zz, levels=np.linspace(0, EMAX, 21), cmap=PES, extend="max"
    )
    ax.contour(
        gx,
        gy,
        np.where(np.isfinite(zz), zz, np.nan),
        levels=np.linspace(0.4, EMAX - 0.4, 8),
        colors="#1a1a2e",
        linewidths=0.3,
    )
    handles = []
    for lab, tip, marker, size in marks:
        h = ax.scatter(
            tip[0],
            tip[1],
            s=size,
            marker=marker,
            c="k",
            edgecolors="white",
            linewidths=0.6,
            zorder=6,
            label=lab,
        )
        handles.append(h)
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=True, fancybox=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    return mesh


def plot_minima_energy() -> None:
    xy = load_xy(PROJ)
    e = np.loadtxt(ENERGY)
    gx, gy, zz = energy_grid(xy, e)
    gm = xy[int(np.argmin(e))]
    ico = xy[int(np.argmin(np.abs(e - ICO_E)))]
    fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor="white")
    mesh = draw(
        ax,
        gx,
        gy,
        zz,
        (
            (rf"GM ${GM_E:.3f}$", gm, "*", 120),
            (rf"ico ${ICO_E:.3f}$", ico, "D", 70),
        ),
        r"asinh $\chi$ of $n_4\ldots n_{13}$: $E-E_{\mathrm{GM}}$",
    )
    cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$E-E_{\mathrm{GM}}/\varepsilon$")
    dest = OUT / "elja_occ_lj38_asinh_energy.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=170, facecolor="white")
    print("wrote", dest, "GM at", gm, "ico at", ico, "Egm", float(e.min()))


def plot_decaf_energy() -> None:
    if not DECAF_E.is_file():
        print("skip DECAF energy: no", DECAF_E)
        return
    emin = []
    for line in DECAF_E.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        emin.append(float(p[2]))
    emin = np.asarray(emin)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), facecolor="white")
    for ax, path, title in (
        (axes[0], DECAF_CER, r"Ceriotti $\chi$: $E-E_{\mathrm{GM}}$"),
        (axes[1], DECAF_XY, r"asinh $\chi$: $E-E_{\mathrm{GM}}$"),
    ):
        xy = load_xy(path)
        gx, gy, zz = energy_grid(xy, emin)
        gm = xy[0]
        ico = xy[1]
        mesh = draw(
            ax,
            gx,
            gy,
            zz,
            (
                (rf"GM ${GM_E:.3f}$", gm, "*", 110),
                (rf"ico ${ICO_E:.3f}$", ico, "D", 70),
            ),
            title,
        )
    fig.colorbar(mesh, ax=axes, fraction=0.03, pad=0.02, label=r"$E-E_{\mathrm{GM}}/\varepsilon$")
    dest = OUT / "elja_occ_lj38_decaf_energy.png"
    fig.savefig(dest, dpi=160, facecolor="white", bbox_inches="tight")
    print("wrote", dest, "fam0 E", emin[0], "fam1 E", emin[1], "min E", emin.min())


def main() -> None:
    plot_minima_energy()
    plot_decaf_energy()


if __name__ == "__main__":
    main()
