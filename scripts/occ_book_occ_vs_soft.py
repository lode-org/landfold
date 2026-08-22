#!/usr/bin/env python3
"""Two-panel: Ceriotti occupancy invert vs soft-committor energy funnel.

Left: leftover-well KDE occupancy invert of the 4042 Elja book on
published/asinh Ceriotti chi (lj38_asinh_cer.proj). F(GM)~0.32,
F(ico)~0.01: ico is the occupied well, GM is a leftover satellite.
Right: soft-committor funnel of the structure-geodesic plane
(occ_book_soft_funnel.py). Connected two wells, F(GM)=0, F(ico)~0.676.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EX = ROOT / "examples" / "cosmo-lj38"
OUT = ROOT / "docs" / "ceriotti-figs"
FIG = OUT / "elja_occ_lj38_occ_vs_soft.png"
BOOK = Path("/tmp/occ-book")
ENERGY = BOOK / "lj38.energy"
SOFT_XY = BOOK / "cand-soft-funnel" / "soft_funnel.xy"
GEO_XY = BOOK / "cand-geostruc" / "asinh.xy"
SRC_DIRS = (
    Path("/tmp/landfold-occ-from-terra/landfold-occ-book"),
    Path("/tmp/landfold-occ-book"),
)
OCC_NAME = "lj38_asinh_cer.proj"

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)
GM_E = -173.928427
ICO_E = -173.252378
KT = 0.168
FMAX_OCC = 2.0
EMAX = 5.0

sys.path.insert(0, str(SCRIPTS))
import occ_book_soft_funnel as soft  # noqa: E402


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


def find_occ() -> Path:
    for folder in SRC_DIRS:
        path = folder / OCC_NAME
        if path.is_file() and path.stat().st_size > 100:
            return path
    raise SystemExit(f"no occupancy projection {OCC_NAME}")


def load_soft_xy(energy: np.ndarray, gm: int, ico: int) -> np.ndarray:
    if SOFT_XY.is_file() and SOFT_XY.stat().st_size > 100:
        xy = np.loadtxt(SOFT_XY)
        if xy.shape[0] == energy.shape[0]:
            return xy
    if not GEO_XY.is_file():
        raise SystemExit(f"missing {SOFT_XY} and {GEO_XY}")
    xy0 = np.loadtxt(GEO_XY)
    if xy0.shape[0] != energy.shape[0]:
        raise SystemExit(f"row mismatch {GEO_XY} {len(xy0)} vs energy {len(energy)}")
    z = np.clip(energy - float(energy[gm]), 0.0, None)
    xy, _q, _r, _s = soft.soft_funnel(xy0, z, gm, ico, soft.ALPHA)
    return xy


def occ_fes(xy: np.ndarray, cf, ngrid: int = 160, sigma: float = 5.0):
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
    fes[on] = -KT * np.log(np.clip(rho[on] / rmax, 1e-12, 1.0))
    return gx, gy, np.clip(fes, 0.0, FMAX_OCC)


def strip_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def tighten(ax, gx, gy, field, pad: float = 0.03) -> None:
    yy, xx = np.where(np.isfinite(field))
    if xx.size == 0:
        return
    x0, x1 = float(gx[xx].min()), float(gx[xx].max())
    y0, y1 = float(gy[yy].min()), float(gy[yy].max())
    dx, dy = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    ax.set_xlim(x0 - pad * dx, x1 + pad * dx)
    ax.set_ylim(y0 - pad * dy, y1 + pad * dy)


def paint_occ(ax, gx, gy, field):
    mesh = ax.contourf(
        gx, gy, field, levels=np.linspace(0.0, FMAX_OCC, 21), cmap=PES, extend="neither"
    )
    ax.contour(
        gx,
        gy,
        np.where(np.isfinite(field), field, np.nan),
        levels=np.linspace(0.15, 1.85, 12),
        colors="#1a1a2e",
        linewidths=0.35,
    )
    strip_axes(ax)
    tighten(ax, gx, gy, field)
    return mesh


def paint_energy(ax, gx, gy, field):
    mesh = ax.pcolormesh(
        gx,
        gy,
        np.clip(field, 0.0, EMAX),
        cmap=PES,
        shading="auto",
        vmin=0.0,
        vmax=EMAX,
    )
    strip_axes(ax)
    tighten(ax, gx, gy, field)
    return mesh


def mark(ax, gm, ico):
    h1 = ax.scatter(
        gm[0],
        gm[1],
        s=140,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=8,
        label=rf"GM ${GM_E:.3f}$",
    )
    h2 = ax.scatter(
        ico[0],
        ico[1],
        s=80,
        marker="D",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=8,
        label=rf"ico ${ICO_E:.3f}$",
    )
    ax.legend(
        handles=[h1, h2],
        loc="center left",
        bbox_to_anchor=(0.13, 0.50),
        borderaxespad=0.0,
        fontsize=8,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="k",
    )


def f_title(head: str, fgm: float, fico: float) -> str:
    return rf"{head}" + "\n" + rf"$F(\mathrm{{GM}})={fgm:.3f}$  $F(\mathrm{{ico}})={fico:.3f}$"


def main() -> None:
    energy = np.loadtxt(ENERGY)
    gm = int(np.argmin(energy))
    ico = int(np.argmin(np.abs(energy - ICO_E)))
    if abs(float(energy[gm]) - GM_E) > 1e-3 or abs(float(energy[ico]) - ICO_E) > 1e-3:
        raise SystemExit(f"index check failed E[{gm}]={energy[gm]} E[{ico}]={energy[ico]}")
    z = np.clip(energy - float(energy[gm]), 0.0, None)
    cf = _cf()

    occ_path = find_occ()
    xy_occ = load_xy(occ_path)
    xy_soft = load_soft_xy(energy, gm, ico)
    for name, xy in (("occ", xy_occ), ("soft", xy_soft)):
        if xy.shape[0] != energy.shape[0]:
            raise SystemExit(f"{name} rows {xy.shape[0]} vs energy {energy.shape[0]}")

    gx_o, gy_o, fes = occ_fes(xy_occ, cf)
    fgm_o = cf.fes_at(gx_o, gy_o, fes, xy_occ[gm])
    fico_o = cf.fes_at(gx_o, gy_o, fes, xy_occ[ico])
    print(f"occ {occ_path.name} F(GM)={fgm_o:.4f} F(ico)={fico_o:.4f}")

    gx_s, gy_s, e_soft = soft.fill(xy_soft, z)
    fgm_s = soft.sample(gx_s, gy_s, e_soft, xy_soft[gm])
    fico_s = soft.sample(gx_s, gy_s, e_soft, xy_soft[ico])
    lab, nlab = soft.components(e_soft)
    ix = int(np.clip(np.searchsorted(gx_s, xy_soft[gm, 0]) - 1, 0, e_soft.shape[1] - 1))
    iy = int(np.clip(np.searchsorted(gy_s, xy_soft[gm, 1]) - 1, 0, e_soft.shape[0] - 1))
    jx = int(np.clip(np.searchsorted(gx_s, xy_soft[ico, 0]) - 1, 0, e_soft.shape[1] - 1))
    jy = int(np.clip(np.searchsorted(gy_s, xy_soft[ico, 1]) - 1, 0, e_soft.shape[0] - 1))
    same = bool(lab[iy, ix] > 0 and lab[iy, ix] == lab[jy, jx])
    print(
        f"soft-funnel F(GM)={fgm_s:.4f} F(ico)={fico_s:.4f} "
        f"n_cc={nlab} same_cc={same}"
    )
    if not same:
        raise SystemExit("soft-funnel wells are not connected")

    fig = plt.figure(figsize=(13.8, 5.15), facecolor="white", layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.72], wspace=0.08)
    ax_o = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1])

    mesh_o = paint_occ(ax_o, gx_o, gy_o, fes)
    mark(ax_o, xy_occ[gm], xy_occ[ico])
    ax_o.set_title(f_title(r"Ceriotti occupancy invert", fgm_o, fico_o), fontsize=11)
    cb_o = fig.colorbar(mesh_o, ax=ax_o, fraction=0.046, pad=0.03)
    cb_o.set_label(r"$F/\varepsilon$  occupancy invert")
    cb_o.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0])

    mesh_s = paint_energy(ax_s, gx_s, gy_s, e_soft)
    mark(ax_s, xy_soft[gm], xy_soft[ico])
    ax_s.set_title(f_title(r"soft-committor funnel", fgm_s, fico_s), fontsize=11)
    cb_s = fig.colorbar(mesh_s, ax=ax_s, fraction=0.046, pad=0.03)
    cb_s.set_label(r"$E-E_{\mathrm{GM}}/\varepsilon$  energy IDW")
    cb_s.set_ticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    cb_s.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=180, facecolor="white")
    plt.close(fig)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
