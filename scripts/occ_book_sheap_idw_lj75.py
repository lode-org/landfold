#!/usr/bin/env python3
"""Compact-support kNN IDW of E-E_GM on the LJ75 SHEAP dual-funnel plane.

Same kernel as occ_book_sheap_idw.py. Occupancy leftover-well invert
is not used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import occ_book_idw_fill as idw
import occ_book_sheap_idw as sheap_idw

OUT = ROOT / "docs" / "ceriotti-figs"
BOOK = Path("/tmp/occ-book")
CAND = BOOK / "cand-sheap-lj75"
XY_CANDIDATES = (CAND / "sheap.xy", CAND / "asinh_sheap.xy")
ENERGY = BOOK / "lj75.energy"
MOTIFS = CAND / "motifs.json"
FIG = OUT / "elja_occ_lj75_sheap_idw.png"
KS = (1, 4, 8)
GM_WELL = 0.55
MOTIF_WELL = 1.50
VMAX = 5.0


def load_xy() -> Path:
    for path in XY_CANDIDATES:
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise SystemExit(f"missing {' or '.join(str(p) for p in XY_CANDIDATES)}")


def paint(ax, xy, z, gm, motif, gm_e, motif_e, gx, gy, field, title: str, legend: bool):
    mesh = ax.pcolormesh(
        gx, gy, np.clip(field, 0, VMAX), cmap=idw.PES, shading="auto", vmin=0, vmax=VMAX
    )
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=np.clip(z, 0, VMAX),
        s=7,
        cmap=idw.PES,
        vmin=0,
        vmax=VMAX,
        edgecolors="none",
        alpha=0.55,
        zorder=20,
    )
    ax.scatter(
        xy[gm, 0],
        xy[gm, 1],
        s=170,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.7,
        zorder=50,
        label=rf"GM ${gm_e:.3f}$",
    )
    ax.scatter(
        xy[motif, 0],
        xy[motif, 1],
        s=95,
        marker="D",
        c="k",
        edgecolors="white",
        linewidths=0.7,
        zorder=50,
        label=rf"2nd ${motif_e:.3f}$",
    )
    if legend:
        ax.legend(
            loc="best",
            fontsize=8,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            facecolor="white",
            edgecolor="k",
        )
    ax.set_xlabel(r"$\chi_1$")
    ax.set_ylabel(r"$\chi_2$")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    return mesh


def main() -> None:
    src = load_xy()
    if not ENERGY.exists():
        raise SystemExit(f"missing {ENERGY}")
    energy = np.loadtxt(ENERGY)
    xy = np.loadtxt(src)
    if xy.shape[0] != energy.shape[0]:
        raise SystemExit(f"row mismatch xy={xy.shape[0]} energy={energy.shape[0]}")
    gm = int(np.argmin(energy))
    if MOTIFS.is_file():
        rec = json.loads(MOTIFS.read_text())
        motif = int(rec["second_idx"])
        if abs(float(energy[motif]) - float(rec["second_E"])) > 1e-3:
            raise SystemExit(
                f"motif check failed E[{motif}]={energy[motif]} vs {rec['second_E']}"
            )
    else:
        raise SystemExit(f"missing {MOTIFS}")
    if abs(float(energy[gm]) - float(energy.min())) > 1e-9:
        raise SystemExit(f"argmin mismatch E[{gm}]={energy[gm]}")
    gm_e = float(energy[gm])
    motif_e = float(energy[motif])
    true_dE = motif_e - gm_e
    z = energy - gm_e
    nn_dist, nn_idx = sheap_idw.knn(xy, xy[gm : gm + 1], k=2)
    nn_e = float(z[int(nn_idx[0, 1])])
    print(f"xy {src}")
    print(
        f"GM idx={gm} E={gm_e:.8f}  second motif idx={motif} E={motif_e:.8f}  "
        f"dE={true_dE:.6f}"
    )
    print(
        f"NN_E(GM)={nn_e:.6f}  nn_idx={int(nn_idx[0, 1])}  "
        f"nn_dist={float(nn_dist[0, 1]):.3e}  z[GM]={float(z[gm]):.6f}  "
        f"z[2nd]={float(z[motif]):.6f}"
    )

    fig, axes = plt.subplots(1, len(KS), figsize=(16.8, 5.4))
    mesh = None
    f_gm = float("nan")
    f_sec = float("nan")
    for ax, k in zip(axes, KS):
        gx, gy, field = sheap_idw.fill(xy, z, k=k)
        fgm = sheap_idw.sample(gx, gy, field, xy[gm])
        f2 = sheap_idw.sample(gx, gy, field, xy[motif])
        diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
        bar_gm = sheap_idw.ring_barrier(xy, gx, gy, field, gm, 0.06 * diam, 0.16 * diam)
        bar_2 = sheap_idw.ring_barrier(
            xy, gx, gy, field, motif, 0.06 * diam, 0.16 * diam
        )
        gm_ok = bool(np.isfinite(fgm) and fgm <= GM_WELL)
        mot_ok = bool(np.isfinite(f2) and f2 <= max(MOTIF_WELL, true_dE + 0.55))
        data_gm = bool(np.isfinite(fgm) and abs(fgm - 0.0) <= 0.05)
        data_2 = bool(np.isfinite(f2) and abs(f2 - true_dE) <= 0.05)
        survive = gm_ok and mot_ok
        print(
            f"k={k} F(GM)={fgm:.4f} F(second motif)={f2:.4f} NN_E(GM)={nn_e:.4f} "
            f"bar_GM={bar_gm:.4f} bar_2nd={bar_2:.4f} "
            f"gm_well={'yes' if gm_ok else 'no'} 2nd_well={'yes' if mot_ok else 'no'} "
            f"survive={'yes' if survive else 'no'} "
            f"at_0_and_dE={'yes' if (data_gm and data_2) else 'no'}"
        )
        if k == 4:
            f_gm, f_sec = fgm, f2
        mesh = paint(
            ax,
            xy,
            z,
            gm,
            motif,
            gm_e,
            motif_e,
            gx,
            gy,
            field,
            title=rf"$k={k}$",
            legend=(k == KS[-1]),
        )
    fig.suptitle(
        r"LJ75 SHEAP dual-funnel   compact-support kNN IDW of $E-E_{\mathrm{GM}}$",
        fontsize=12,
    )
    fig.colorbar(mesh, ax=list(axes), fraction=0.02, pad=0.02, label=r"$E-E_\mathrm{GM}$")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=180, bbox_inches="tight")
    CAND.mkdir(parents=True, exist_ok=True)
    fig.savefig(CAND / FIG.name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG)
    print(f"F(GM)={f_gm:.4f} F(second motif)={f_sec:.4f} n={energy.shape[0]}")


if __name__ == "__main__":
    main()
