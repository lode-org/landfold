#!/usr/bin/env python3
"""Occupancy invert of published/asinh chi next to SOAP and SHEAP energy IDW.

Left: leftover-well occupancy invert of the 4042 Elja book on HD-asinh /
LD-Ceriotti chi (lj38_asinh_cer.proj, else ceriotti.proj). F in [0, 2].
Middle: SOAP mean_r16a8_z plane, field = compact-support IDW of E-E_GM.
Right: SHEAP dual-funnel plane, field = the same IDW of E-E_GM.

Occupancy invert is not used on the energy panels.
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
SOAP_XY = BOOK / "cand-soap" / "mean_r16a8_z.xy"
SHEAP_XY = BOOK / "cand-sheap" / "sheap.xy"
FIG = OUT / "elja_occ_lj38_defeat_panel.png"
SRC_DIRS = (
    Path("/tmp/landfold-occ-from-terra/landfold-occ-book"),
    Path("/tmp/landfold-occ-book"),
)
OCC_PROJS = (
    ("lj38_asinh_cer.proj", r"occupancy invert  published/asinh $\chi$"),
    ("lj38_ceriotti.proj", r"occupancy invert  Ceriotti $\chi$"),
    ("lj38_asinh.proj", r"occupancy invert  asinh $\chi$"),
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


def unique_cloud(xy: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse exact copies so IDW is not weighted by leftover-well occupancy."""
    key = np.round(xy, 8)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    n = int(inv.max()) + 1
    acc = np.zeros((n, 2), dtype=np.float64)
    ez = np.full(n, np.inf, dtype=np.float64)
    cnt = np.zeros(n, dtype=np.float64)
    np.add.at(acc, inv, xy)
    np.add.at(cnt, inv, 1.0)
    np.minimum.at(ez, inv, z)
    return acc / cnt[:, None], ez


def idw_field(xy: np.ndarray, z: np.ndarray, ngrid: int = 150, k: int = 6):
    pts, val = unique_cloud(xy, z)
    xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
    ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= 0.08 * dx
    xmax += 0.08 * dx
    ymin -= 0.08 * dy
    ymax += 0.08 * dy
    gx = np.linspace(xmin, xmax, ngrid)
    gy = np.linspace(ymin, ymax, ngrid)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    dist, idx = idw.knn(pts, grid, k)
    w = 1.0 / np.clip(dist, 1e-9, None) ** 2
    w /= w.sum(axis=1, keepdims=True)
    field = (w * val[idx]).sum(axis=1).reshape(ngrid, ngrid)
    nn = dist[:, 0].reshape(ngrid, ngrid)
    nn_data = idw.knn(pts, pts, k=2)[0][:, 1]
    pos = nn_data[nn_data > 1e-12]
    span = max(xmax - xmin, ymax - ymin)
    cutoff = max(4.0 * float(np.median(pos)) if pos.size else 0.08 * span, 0.07 * span)
    field = np.where(nn < cutoff, field, np.nan)
    return gx, gy, field


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


def sample(gx, gy, field, pt) -> float:
    if not np.isfinite(field).any():
        return float("nan")
    iy = int(np.argmin(np.abs(gy - pt[1])))
    ix = int(np.argmin(np.abs(gx - pt[0])))
    v = field[iy, ix]
    if np.isfinite(v):
        return float(v)
    yy, xx = np.where(np.isfinite(field))
    if xx.size == 0:
        return float("nan")
    j = int(np.argmin((gx[xx] - pt[0]) ** 2 + (gy[yy] - pt[1]) ** 2))
    return float(field[yy[j], xx[j]])


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


def f_title(head: str, fgm: float, fico: float) -> str:
    return rf"{head}" + "\n" + rf"$F(\mathrm{{GM}})={fgm:.2f}$  $F(\mathrm{{ico}})={fico:.2f}$"


def main() -> None:
    energy = np.loadtxt(ENERGY)
    if abs(float(energy[GM_IDX]) - GM_E) > 1e-3 or abs(float(energy[ICO_IDX]) - ICO_E) > 1e-3:
        raise SystemExit(f"index check failed E[0]={energy[GM_IDX]} E[40]={energy[ICO_IDX]}")
    if not SOAP_XY.is_file():
        raise SystemExit(f"missing {SOAP_XY}")
    if not SHEAP_XY.is_file():
        raise SystemExit(f"missing {SHEAP_XY}")
    z = energy - float(energy[GM_IDX])
    cf = _cf()
    occ_path, occ_head = find_occ()
    xy_occ = load_xy(occ_path)
    xy_soap = np.loadtxt(SOAP_XY)
    xy_sheap = np.loadtxt(SHEAP_XY)
    for name, xy in (("occ", xy_occ), ("soap", xy_soap), ("sheap", xy_sheap)):
        if xy.shape[0] != energy.shape[0]:
            raise SystemExit(f"{name} rows {xy.shape[0]} vs energy {energy.shape[0]}")

    gx_o, gy_o, fes = occ_fes(xy_occ, cf)
    fgm_o = cf.fes_at(gx_o, gy_o, fes, xy_occ[GM_IDX])
    fico_o = cf.fes_at(gx_o, gy_o, fes, xy_occ[ICO_IDX])
    print(f"occ {occ_path.name} F(GM)={fgm_o:.4f} F(ico)={fico_o:.4f}")

    gx_s, gy_s, e_soap = idw_field(xy_soap, z)
    fgm_s = sample(gx_s, gy_s, e_soap, xy_soap[GM_IDX])
    fico_s = sample(gx_s, gy_s, e_soap, xy_soap[ICO_IDX])
    print(f"SOAP energy IDW F(GM)={fgm_s:.4f} F(ico)={fico_s:.4f}")

    gx_h, gy_h, e_sheap = idw_field(xy_sheap, z)
    fgm_h = sample(gx_h, gy_h, e_sheap, xy_sheap[GM_IDX])
    fico_h = sample(gx_h, gy_h, e_sheap, xy_sheap[ICO_IDX])
    print(f"SHEAP energy IDW F(GM)={fgm_h:.4f} F(ico)={fico_h:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.15), facecolor="white")
    mesh_o = paint_occ(axes[0], gx_o, gy_o, fes)
    mark(axes[0], xy_occ[GM_IDX], xy_occ[ICO_IDX])
    axes[0].set_title(f_title(occ_head, fgm_o, fico_o), fontsize=11)
    cb_o = fig.colorbar(mesh_o, ax=axes[0], fraction=0.046, pad=0.03)
    cb_o.set_label(r"$F/\varepsilon$  occupancy invert")
    cb_o.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0])

    mesh_s = paint_energy(axes[1], gx_s, gy_s, e_soap)
    mark(axes[1], xy_soap[GM_IDX], xy_soap[ICO_IDX])
    axes[1].set_title(f_title(r"SOAP  $E-E_{\mathrm{GM}}$ IDW", fgm_s, fico_s), fontsize=11)
    cb_s = fig.colorbar(mesh_s, ax=axes[1], fraction=0.046, pad=0.03)
    cb_s.set_label(r"$E-E_{\mathrm{GM}}/\varepsilon$")
    cb_s.set_ticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    mesh_h = paint_energy(axes[2], gx_h, gy_h, e_sheap)
    mark(axes[2], xy_sheap[GM_IDX], xy_sheap[ICO_IDX])
    axes[2].set_title(f_title(r"SHEAP  $E-E_{\mathrm{GM}}$ IDW", fgm_h, fico_h), fontsize=11)
    cb_h = fig.colorbar(mesh_h, ax=axes[2], fraction=0.046, pad=0.03)
    cb_h.set_label(r"$E-E_{\mathrm{GM}}/\varepsilon$")
    cb_h.set_ticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=180, facecolor="white")
    plt.close(fig)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
