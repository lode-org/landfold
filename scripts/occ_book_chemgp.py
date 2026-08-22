#!/usr/bin/env python3
"""MethodsX energy GP + occupancy on asinh chi of 4042 inherent structures.

Occupancy invert of leftover wells is the wrong energy field. The GP
fits z = E - E_GM of the quenched minima on asinh n4..n13 chi. The
occupancy KDE is a separate panel (and contour overlay).
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
EX = ROOT / "examples" / "cosmo-lj38"
OUT = ROOT / "docs" / "ceriotti-figs"
PROJ = Path("/tmp/landfold-occ-from-terra/landfold-occ-book/lj38_asinh.proj")
ENERGY = Path("/tmp/occ-book/lj38.energy")

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)
GM_E = -173.928427
ICO_E = -173.252378
KT = 0.168
N_INDUCING = 80
GM_IDX = 0
ICO_IDX = 40


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


def kde_fes(xy, cf, ngrid=160, sigma=5.0):
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= 0.10 * dx
    xmax += 0.10 * dx
    ymin -= 0.10 * dy
    ymax += 0.10 * dy
    counts, xe, ye = np.histogram2d(x, y, bins=ngrid, range=[[xmin, xmax], [ymin, ymax]])
    rho = cf._blur2d(counts.T, sigma=sigma)
    gx = 0.5 * (xe[:-1] + xe[1:])
    gy = 0.5 * (ye[:-1] + ye[1:])
    rmax = float(rho.max())
    mask = cf._fill_mask_holes(rho > 0.004 * rmax)
    fes = np.full_like(rho, np.nan)
    on = mask & (rho > 0)
    fes[on] = -KT * np.log(np.clip(rho[on] / rmax, 1e-12, 1))
    return gx, gy, np.clip(fes, 0, 2)


def mark(ax, gm, ico):
    h1 = ax.scatter(
        gm[0],
        gm[1],
        s=120,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=50,
        label=rf"GM ${GM_E:.3f}$",
    )
    h2 = ax.scatter(
        ico[0],
        ico[1],
        s=70,
        marker="D",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=50,
        label=rf"ico ${ICO_E:.3f}$",
    )
    ax.legend(handles=[h1, h2], loc="best", fontsize=8, frameon=True, fancybox=False)


def paint_occ(ax, gx, gy, fes):
    mesh = ax.contourf(gx, gy, fes, levels=np.linspace(0, 2, 21), cmap=PES, extend="max")
    ax.contour(
        gx,
        gy,
        np.where(np.isfinite(fes), fes, np.nan),
        levels=np.linspace(0.15, 1.85, 12),
        colors="#1a1a2e",
        linewidths=0.35,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mesh


def occ_contours(ax, gx, gy, fes):
    ax.contour(
        gx,
        gy,
        np.where(np.isfinite(fes), fes, np.nan),
        levels=np.linspace(0.15, 1.85, 12),
        colors="#1a1a2e",
        linewidths=0.35,
        zorder=20,
    )


def inducing_indices(xy: np.ndarray, k: int, must) -> np.ndarray:
    from chemparseplot.plot.representer import farthest_indices

    idx = farthest_indices(xy, k)
    extra = np.asarray(list(must), dtype=int)
    return np.unique(np.concatenate([idx, extra]))


def energy_cloud(xy: np.ndarray, energy: np.ndarray):
    z = energy - float(energy.min())
    idx = inducing_indices(xy, N_INDUCING, (GM_IDX, ICO_IDX))
    return xy[idx, 0], xy[idx, 1], z[idx], z


def rbf_smooth_of(s1, s2) -> float:
    span = float(max(np.ptp(s1), np.ptp(s2)))
    return max(0.1 * span, 1e-3)


def draw_energy_gp(ax, s1, s2, z) -> None:
    from chemparseplot.plot.neb import SurfaceFitConfig, plot_landscape_surface
    from chemparseplot.plot.theme import RUHI_THEME, get_theme, setup_publication_theme

    setup_publication_theme(get_theme("ruhi"))
    plot_landscape_surface(
        ax,
        s1,
        s2,
        None,
        None,
        z,
        method="rbf",
        rbf_smooth=rbf_smooth_of(s1, s2),
        project_path=False,
        cmap=RUHI_THEME.cmap_landscape,
        show_pts=False,
        surface_fit=SurfaceFitConfig(auto_thin=False, max_surface_points=N_INDUCING + 2),
        n_inducing=N_INDUCING,
        variance_threshold=0.5,
    )


def energy_gp_figure(s1, s2, z, gx, gy, fes, gm, ico, dest: Path) -> None:
    from chemparseplot.parse.types import EnergyRepresentation
    from chemparseplot.plot.landfold import plot_fes
    from chemparseplot.plot.representation import plot_energy

    clabel = r"$E-E_{\mathrm{GM}}/\varepsilon$"
    fig = None
    err = None
    try:
        fig = plot_fes(
            cloud=(s1, s2, z),
            method="rbf",
            n_inducing=N_INDUCING,
            show_pts=False,
            clabel=clabel,
            xlabel=r"$s_1$",
            ylabel=r"$s_2$",
        )
    except Exception as exc:
        err = exc
        try:
            rep = EnergyRepresentation.from_mapping(
                {
                    "schema": "chemparseplot.energy.v1",
                    "x": s1,
                    "y": s2,
                    "energy": z,
                    "frame": "plane",
                    "xlabel": r"$s_1$",
                    "ylabel": r"$s_2$",
                    "metadata": {"field": "quenched_energy"},
                }
            )
            fig = plot_energy(
                rep,
                method="rbf",
                n_inducing=N_INDUCING,
                show_pts=False,
                xlabel=r"$s_1$",
                ylabel=r"$s_2$",
                clabel=clabel,
                figsize=(5.6, 4.8),
                dpi=170,
            )
            err = None
        except Exception as exc2:
            err = exc2
            fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=170, facecolor="white")
            draw_energy_gp(ax, s1, s2, z)
            filled = next(
                (c for c in ax.collections if getattr(c, "filled", False)),
                ax.collections[0] if ax.collections else None,
            )
            if filled is not None:
                fig.colorbar(filled, ax=ax, fraction=0.046, pad=0.04).set_label(clabel)
            ax.set_xlabel(r"$s_1$")
            ax.set_ylabel(r"$s_2$")
            ax.set_aspect("equal", adjustable="box")
            fig.tight_layout()
    ax = fig.axes[0]
    occ_contours(ax, gx, gy, fes)
    mark(ax, gm, ico)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", dest)
    if err is not None:
        print("plot_fes/plot_energy failed, used plot_landscape_surface:", type(err).__name__, err)


def two_panel(xy, e, s1, s2, z, gx, gy, fes, gm, ico, dest: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), facecolor="white")
    mesh = paint_occ(axes[0], gx, gy, fes)
    mark(axes[0], gm, ico)
    axes[0].set_title(r"occupancy of 4042 inherent structures")
    fig.colorbar(mesh, ax=axes[0], fraction=0.046, pad=0.03).set_label(r"$F/\varepsilon$")

    draw_energy_gp(axes[1], s1, s2, z)
    occ_contours(axes[1], gx, gy, fes)
    mark(axes[1], gm, ico)
    axes[1].set_title(r"$E-E_{\mathrm{GM}}$ RBF GP")
    filled = next(
        (c for c in axes[1].collections if getattr(c, "filled", False)),
        axes[1].collections[0] if axes[1].collections else None,
    )
    if filled is not None:
        fig.colorbar(filled, ax=axes[1], fraction=0.046, pad=0.03).set_label(
            r"$E-E_{\mathrm{GM}}/\varepsilon$"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", dest)


def main() -> None:
    cf = _cf()
    xy = load_xy(PROJ)
    e = np.loadtxt(ENERGY)
    if len(xy) != len(e):
        raise SystemExit(f"row mismatch proj {len(xy)} vs energy {len(e)}")
    gm = xy[GM_IDX]
    ico = xy[ICO_IDX]
    if abs(float(e[GM_IDX]) - GM_E) > 1e-3 or abs(float(e[ICO_IDX]) - ICO_E) > 1e-3:
        raise SystemExit(
            f"index check failed E[0]={e[GM_IDX]} E[40]={e[ICO_IDX]}"
        )
    gx, gy, fes = kde_fes(xy, cf)
    s1, s2, z, z_all = energy_cloud(xy, e)
    print(
        "n",
        len(e),
        "inducing",
        len(s1),
        "GM",
        gm,
        float(z_all[GM_IDX]),
        "ico",
        ico,
        float(z_all[ICO_IDX]),
        "F(GM)",
        cf.fes_at(gx, gy, fes, gm),
        "F(ico)",
        cf.fes_at(gx, gy, fes, ico),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    energy_gp_figure(
        s1, s2, z, gx, gy, fes, gm, ico, OUT / "elja_occ_lj38_asinh_chemgp.png"
    )
    two_panel(
        xy,
        e,
        s1,
        s2,
        z,
        gx,
        gy,
        fes,
        gm,
        ico,
        OUT / "elja_occ_lj38_asinh_occ_energy.png",
    )


if __name__ == "__main__":
    main()
