#!/usr/bin/env python3
"""Hairpin F-space P(D,d) plus LJ38 SOAP / geodesic energy basins.

Row 1 is the measured hairpin transfer-function result (Ceriotti / asinh /
mixed). The published sketch-map protein.cv is dihedral HD, not xyz, so
this figure does not invent a SOAP of the hairpin. Row 2 is E-E_GM on
the Elja LJ38 book: SOAP-like neighbor density and structure-geodesic
asinh IDW. Neither basin panel inverts occupancy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "docs" / "ceriotti-figs"
FSPACE = OUT / "protein_fspace_pd.png"
DEST = OUT / "elja_occ_lj38_metric_and_basins.png"
SKMAP = Path.home() / "Git/Github/HaoZeke/sketchmap/examples/protein"
ASINH = Path("/tmp/landfold-asinh/protein_asinh.ld")
MIXED = Path("/tmp/landfold-twoscale/protein_asinh_cer.ld")
BOOK = Path("/tmp/occ-book")
ENERGY = BOOK / "lj38.energy"
SOAP_XY = BOOK / "cand-soap" / "winner.xy"
GEO_XY = BOOK / "cand-geostruc" / "asinh.xy"
GM_E = -173.928427
ICO_E = -173.252378

sys.path.insert(0, str(SCRIPTS))
import occ_book_idw_fill as idw
import occ_book_soap as soap


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


caf = _load("compare_asinh_fig", SCRIPTS / "compare_asinh_fig.py")
pfs = _load("compare_protein_fspace", SCRIPTS / "compare_protein_fspace.py")


def _need(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit("need " + ", ".join(missing))


def confirm_fspace() -> None:
    if not FSPACE.is_file():
        raise SystemExit(f"missing {FSPACE}")
    st = FSPACE.stat()
    print(f"protein_fspace_pd.png exists bytes={st.st_size} mtime={st.st_mtime:.0f}")


def fspace_panels():
    hd = np.loadtxt(SKMAP / "lm4.30cv.w01.1")[:, :30]
    pub = np.loadtxt(SKMAP / "lm4.30cv.w01.1.proj")[:, :2]
    ash = np.loadtxt(ASINH)[:, :2]
    mix = np.loadtxt(MIXED)[:, :2]
    D = pfs.pair_period(hd)
    return (
        (
            pfs.xsig(D, 6.0, 8.0, 8.0),
            pfs.xsig(pfs.pair_euclid(pub), 6.0, 2.0, 8.0),
            pfs.scores(D, pfs.pair_euclid(pub)),
            r"Ceriotti $F(D),f(d)$",
        ),
        (
            pfs.asinh_f(D),
            pfs.asinh_f(pfs.pair_euclid(ash)),
            pfs.scores(D, pfs.pair_euclid(ash)),
            r"asinh $F(D),f(d)$",
        ),
        (
            pfs.asinh_f(D),
            pfs.xsig(pfs.pair_euclid(mix), 6.0, 2.0, 8.0),
            pfs.scores(D, pfs.pair_euclid(mix)),
            r"HD asinh / LD $6,2,8$",
        ),
    )


def soap_energy(xy: np.ndarray, rel: np.ndarray):
    idx = soap.inducing(xy, rel)
    pts = xy[idx]
    val = rel[idx]
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    ell = 0.045 * max(diam, 1e-9)
    gx, gy, field = soap.nw_field(xy, rel, pts, val, "rbf", ell * 0.80)
    return gx, gy, field, ell, len(idx)


def mark(ax, xy, gm: int, ico: int) -> None:
    ax.scatter(
        xy[gm, 0],
        xy[gm, 1],
        s=140,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.7,
        zorder=50,
        label=rf"GM ${GM_E:.3f}$",
    )
    ax.scatter(
        xy[ico, 0],
        xy[ico, 1],
        s=80,
        marker="D",
        c="k",
        edgecolors="white",
        linewidths=0.7,
        zorder=50,
        label=rf"ico ${ICO_E:.3f}$",
    )
    ax.legend(
        fontsize=8,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="k",
        loc="best",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def letter(ax, lab: str) -> None:
    ax.text(
        0.0,
        1.06,
        lab,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def main() -> None:
    confirm_fspace()
    _need(ASINH, MIXED, SKMAP / "lm4.30cv.w01.1", SKMAP / "lm4.30cv.w01.1.proj")
    _need(ENERGY, SOAP_XY, GEO_XY)

    panels = fspace_panels()
    for _fd, _fd_ld, sc, title in panels:
        print(title, "pear_far", sc[0], "spear_far", sc[1])

    energy = np.loadtxt(ENERGY)
    gm = int(np.argmin(energy))
    ico = int(np.argmin(np.abs(energy - ICO_E)))
    if abs(float(energy[gm]) - GM_E) > 1e-3 or abs(float(energy[ico]) - ICO_E) > 1e-3:
        raise SystemExit(f"index check failed E[{gm}]={energy[gm]} E[{ico}]={energy[ico]}")
    rel = np.clip(energy - float(energy[gm]), 0.0, None)
    soap_xy = np.loadtxt(SOAP_XY)
    geo_xy = np.loadtxt(GEO_XY)
    gx_s, gy_s, field_s, ell, n_ind = soap_energy(soap_xy, rel)
    gx_g, gy_g, field_g = idw.fill(geo_xy, rel)
    print(
        "SOAP RBF",
        "ell",
        f"{ell:.4f}",
        "inducing",
        n_ind,
        "E(GM)",
        soap.field_at(gx_s, gy_s, field_s, soap_xy[gm]),
        "E(ico)",
        soap.field_at(gx_s, gy_s, field_s, soap_xy[ico]),
    )
    print(
        "geodesic IDW",
        "E(GM)",
        idw.sample(gx_g, gy_g, field_g, geo_xy[gm]),
        "E(ico)",
        idw.sample(gx_g, gy_g, field_g, geo_xy[ico]),
    )

    fig = plt.figure(figsize=(13.4, 9.6), facecolor="white")
    outer = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.0, 1.18],
        hspace=0.24,
        left=0.045,
        right=0.985,
        top=0.93,
        bottom=0.04,
    )
    gs1 = outer[0].subgridspec(1, 3, wspace=0.16)
    gs2 = outer[1].subgridspec(1, 2, wspace=0.12)
    axes_f = [fig.add_subplot(gs1[0, i]) for i in range(3)]
    ax_soap = fig.add_subplot(gs2[0, 0])
    ax_geo = fig.add_subplot(gs2[0, 1])

    for ax, (fd, fd_ld, sc, title), lab in zip(axes_f, panels, "abc"):
        caf.pd_hist(ax, fd, fd_ld, rf"{title}" + "\n" + rf"far Spearman {sc[1]:.3f}")
        letter(ax, lab)

    mesh_s = soap.paint_energy(
        ax_soap, gx_s, gy_s, field_s, r"SOAP-like neighbor density   $E-E_{\mathrm{GM}}$"
    )
    mark(ax_soap, soap_xy, gm, ico)
    fig.colorbar(mesh_s, ax=ax_soap, fraction=0.046, pad=0.03).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    letter(ax_soap, "d")

    vmax = 5.0
    mesh_g = ax_geo.pcolormesh(
        gx_g,
        gy_g,
        np.clip(field_g, 0, vmax),
        cmap=idw.PES,
        shading="auto",
        vmin=0,
        vmax=vmax,
    )
    ax_geo.scatter(
        geo_xy[:, 0],
        geo_xy[:, 1],
        c=np.clip(rel, 0, vmax),
        s=5,
        cmap=idw.PES,
        vmin=0,
        vmax=vmax,
        edgecolors="none",
        alpha=0.50,
        zorder=20,
    )
    ax_geo.set_title("structure geodesic asinh  energy IDW")
    ax_geo.set_aspect("equal", adjustable="datalim")
    mark(ax_geo, geo_xy, gm, ico)
    fig.colorbar(mesh_g, ax=ax_geo, fraction=0.046, pad=0.03).set_label(
        r"$E-E_{\mathrm{GM}}$"
    )
    letter(ax_geo, "e")

    fig.savefig(DEST, dpi=180, facecolor="white")
    print("wrote", DEST)


if __name__ == "__main__":
    main()
