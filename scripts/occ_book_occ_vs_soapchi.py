#!/usr/bin/env python3
"""Two-panel: leftover occupancy invert vs landfold SOAP asinh chi energy.

Left: leftover-well occupancy invert of the 4042 Elja book on HD-asinh /
LD-Ceriotti chi (lj38_asinh_cer.proj). F(GM)=0.32, F(ico)=0.01.
Right: landfold SOAP asinh chi (cand-soap-chi/soap_chi.xy), compact-support
IDW of E-E_GM. F(GM)=0, F(ico)=0.676.

The right plane is the landfold embed, not python SOAP MDS.
Occupancy invert is not used on the energy panel.
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
OUT = ROOT / "docs" / "ceriotti-figs"
EX = ROOT / "examples" / "cosmo-lj38"
BOOK = Path("/tmp/occ-book")
ENERGY = BOOK / "lj38.energy"
CHI_XYS = (
    BOOK / "cand-soap-chi" / "soap_chi.xy",
    BOOK / "cand-soap-chi" / "soap.proj",
)
FIG = OUT / "elja_occ_lj38_occ_vs_soapchi.png"
SRC_DIRS = (
    Path("/tmp/landfold-occ-from-terra/landfold-occ-book"),
    Path("/tmp/landfold-occ-book"),
)
OCC_PROJS = (
    ("lj38_asinh_cer.proj", r"occupancy invert"),
    ("lj38_ceriotti.proj", r"occupancy invert"),
    ("lj38_asinh.proj", r"occupancy invert"),
)

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)
GM_E = -173.928427
ICO_E = -173.252378
GM_IDX = 0
ICO_IDX = 40
KT = 0.168
FMAX_OCC = 2.0
EMAX = 5.0

sys.path.insert(0, str(SCRIPTS))
import occ_book_idw_fill as idw


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


def find_occ() -> tuple[Path, str]:
    for folder in SRC_DIRS:
        for name, title in OCC_PROJS:
            path = folder / name
            if path.is_file() and path.stat().st_size > 100:
                return path, title
    raise SystemExit("no occupancy projection (asinh_cer / ceriotti / asinh)")


def find_chi() -> Path:
    for path in CHI_XYS:
        if path.is_file() and path.stat().st_size > 100:
            return path
    raise SystemExit("no landfold SOAP chi (cand-soap-chi/soap_chi.xy)")


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


def paint_occ(ax, gx, gy, field):
    mesh = ax.contourf(
        gx, gy, field, levels=np.linspace(0.0, FMAX_OCC, 21), cmap=PES, extend="max"
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
        loc="upper right",
        fontsize=8,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="k",
    )


def fmt_f(v: float, digits: int) -> str:
    if not np.isfinite(v):
        return "nan"
    if abs(v) < 0.5 * 10 ** (-digits):
        return "0"
    return f"{v:.{digits}f}"


def f_title(head: str, fgm: float, fico: float, digits: int) -> str:
    return (
        rf"{head}"
        + "\n"
        + rf"$F(\mathrm{{GM}})={fmt_f(fgm, digits)}$  "
        + rf"$F(\mathrm{{ico}})={fmt_f(fico, digits)}$"
    )


def main() -> None:
    energy = np.loadtxt(ENERGY)
    if abs(float(energy[GM_IDX]) - GM_E) > 1e-3 or abs(float(energy[ICO_IDX]) - ICO_E) > 1e-3:
        raise SystemExit(f"index check failed E[0]={energy[GM_IDX]} E[40]={energy[ICO_IDX]}")
    chi_path = find_chi()
    z = energy - float(energy[GM_IDX])
    cf = _cf()
    occ_path, occ_head = find_occ()
    xy_occ = load_xy(occ_path)
    xy_chi = load_xy(chi_path)
    for name, xy in (("occ", xy_occ), ("soap_chi", xy_chi)):
        if xy.shape[0] != energy.shape[0]:
            raise SystemExit(f"{name} rows {xy.shape[0]} vs energy {energy.shape[0]}")

    gx_o, gy_o, fes = occ_fes(xy_occ, cf)
    fgm_o = cf.fes_at(gx_o, gy_o, fes, xy_occ[GM_IDX])
    fico_o = cf.fes_at(gx_o, gy_o, fes, xy_occ[ICO_IDX])
    print(f"occ {occ_path.name} F(GM)={fgm_o:.4f} F(ico)={fico_o:.4f}")

    gx_c, gy_c, e_chi = idw.fill(xy_chi, z)
    fgm_c = idw.sample(gx_c, gy_c, e_chi, xy_chi[GM_IDX])
    fico_c = idw.sample(gx_c, gy_c, e_chi, xy_chi[ICO_IDX])
    print(f"SOAP chi energy IDW {chi_path.name} F(GM)={fgm_c:.4f} F(ico)={fico_c:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.15), facecolor="white")
    mesh_o = paint_occ(axes[0], gx_o, gy_o, fes)
    mark(axes[0], xy_occ[GM_IDX], xy_occ[ICO_IDX])
    axes[0].set_title(f_title(occ_head, fgm_o, fico_o, digits=2), fontsize=11)
    cb_o = fig.colorbar(mesh_o, ax=axes[0], fraction=0.046, pad=0.03)
    cb_o.set_label(r"$F/\varepsilon$  occupancy invert")
    cb_o.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0])

    mesh_c = paint_energy(axes[1], gx_c, gy_c, e_chi)
    mark(axes[1], xy_chi[GM_IDX], xy_chi[ICO_IDX])
    axes[1].set_title(
        f_title(r"SOAP asinh $\chi$ energy", fgm_c, fico_c, digits=3), fontsize=11
    )
    cb_c = fig.colorbar(mesh_c, ax=axes[1], fraction=0.046, pad=0.03)
    cb_c.set_label(r"$E-E_{\mathrm{GM}}/\varepsilon$")
    cb_c.set_ticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=180, facecolor="white")
    plt.close(fig)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
