#!/usr/bin/env python3
"""Three-panel SHEAP energy IDW of the Elja LJ38 / LJ75 / LJ98 books.

Re-paints sheap.xy with the same compact-support kNN IDW as
occ_book_sheap_idw.py so the panels share ruhi_pes, GM star,
second-motif diamond, and E-E_GM in [0, 5]. Occupancy leftover-well
invert is not used.

LJ75 book lowest is the ico at -396.282; the Marks deca at -397.492
is not in lj75_0013.min.
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
FIG = OUT / "elja_occ_lj_sheap_idw_n38_n75_n98.png"
K = 4
VMAX = 5.0
LJ38_GM_E = -173.928427
LJ38_ICO_E = -173.252378
LJ38_GM_IDX = 0
LJ38_ICO_IDX = 40
LJ75_BOOK_GM = -396.282
LJ75_MARKS_DECA = -397.492
CAPTION = (
    "LJ75 book lowest is the ico at "
    f"{LJ75_BOOK_GM:.3f}; the Marks deca at {LJ75_MARKS_DECA:.3f} "
    "is not in lj75_0013.min."
)

PANELS = (
    {
        "n": 38,
        "label": r"LJ38",
        "cand": BOOK / "cand-sheap",
        "energy": BOOK / "lj38.energy",
        "motifs": BOOK / "cand-sheap" / "motifs.json",
    },
    {
        "n": 75,
        "label": r"LJ75",
        "cand": BOOK / "cand-sheap-lj75",
        "energy": BOOK / "lj75.energy",
        "motifs": BOOK / "cand-sheap-lj75" / "motifs.json",
    },
    {
        "n": 98,
        "label": r"LJ98",
        "cand": BOOK / "cand-sheap-lj98",
        "energy": BOOK / "lj98.energy",
        "motifs": BOOK / "cand-sheap-lj98" / "motifs.json",
    },
)


def load_xy(cand: Path) -> Path:
    for path in (cand / "sheap.xy", cand / "asinh_sheap.xy"):
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise SystemExit(f"missing {cand / 'sheap.xy'} or {cand / 'asinh_sheap.xy'}")


def second_motif(n: int, energy: np.ndarray, motifs: Path) -> int:
    if motifs.is_file():
        rec = json.loads(motifs.read_text())
        motif = int(rec["second_idx"])
        if abs(float(energy[motif]) - float(rec["second_E"])) > 1e-3:
            raise SystemExit(
                f"N={n} motif check failed E[{motif}]={energy[motif]} vs {rec['second_E']}"
            )
        return motif
    if n != 38:
        raise SystemExit(f"missing {motifs}")
    motif = LJ38_ICO_IDX
    gm = LJ38_GM_IDX
    if abs(float(energy[gm]) - LJ38_GM_E) > 1e-3 or abs(float(energy[motif]) - LJ38_ICO_E) > 1e-3:
        raise SystemExit(f"N=38 index check failed E[{gm}]={energy[gm]} E[{motif}]={energy[motif]}")
    return motif


def load_panel(spec: dict) -> dict:
    src = load_xy(spec["cand"])
    if not spec["energy"].is_file():
        raise SystemExit(f"missing {spec['energy']}")
    energy = np.loadtxt(spec["energy"])
    xy = np.loadtxt(src)
    if xy.shape[0] != energy.shape[0]:
        raise SystemExit(f"N={spec['n']} row mismatch xy={xy.shape[0]} energy={energy.shape[0]}")
    gm = int(np.argmin(energy))
    if abs(float(energy[gm]) - float(energy.min())) > 1e-9:
        raise SystemExit(f"N={spec['n']} argmin mismatch E[{gm}]={energy[gm]}")
    motif = second_motif(spec["n"], energy, spec["motifs"])
    gm_e = float(energy[gm])
    motif_e = float(energy[motif])
    z = energy - gm_e
    return {
        "n": spec["n"],
        "label": spec["label"],
        "src": src,
        "xy": xy,
        "z": z,
        "gm": gm,
        "motif": motif,
        "gm_e": gm_e,
        "motif_e": motif_e,
        "true_dE": motif_e - gm_e,
        "count": int(energy.shape[0]),
    }


def paint(ax, rec: dict, gx, gy, field, legend: bool):
    xy, z = rec["xy"], rec["z"]
    gm, motif = rec["gm"], rec["motif"]
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
        label=rf"GM ${rec['gm_e']:.3f}$",
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
        label=rf"2nd ${rec['motif_e']:.3f}$",
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
    ax.set_title(rec["label"])
    ax.set_aspect("equal", adjustable="datalim")
    return mesh


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.7), facecolor="white")
    mesh = None
    for ax, spec in zip(axes, PANELS):
        rec = load_panel(spec)
        gx, gy, field = sheap_idw.fill(rec["xy"], rec["z"], k=K)
        fgm = sheap_idw.sample(gx, gy, field, rec["xy"][rec["gm"]])
        f2 = sheap_idw.sample(gx, gy, field, rec["xy"][rec["motif"]])
        print(
            f"N={rec['n']} xy {rec['src']} n={rec['count']} "
            f"GM idx={rec['gm']} E={rec['gm_e']:.8f}  "
            f"2nd idx={rec['motif']} E={rec['motif_e']:.8f}  "
            f"dE={rec['true_dE']:.6f}"
        )
        print(f"N={rec['n']} F(GM)={fgm:.4f} F(2nd)={f2:.4f}")
        if rec["n"] == 75:
            if abs(rec["gm_e"] - LJ75_BOOK_GM) > 5e-3:
                raise SystemExit(f"LJ75 book GM {rec['gm_e']} is not the ico at {LJ75_BOOK_GM}")
            energy75 = np.loadtxt(spec["energy"])
            if np.any(np.abs(energy75 - LJ75_MARKS_DECA) <= 1e-3):
                raise SystemExit("LJ75 Marks deca is present; caption would be false")
        mesh = paint(ax, rec, gx, gy, field, legend=True)
    fig.suptitle(
        rf"SHEAP dual-funnel   compact-support $k={K}$ IDW of $E-E_{{\mathrm{{GM}}}}$",
        fontsize=12,
    )
    fig.colorbar(mesh, ax=list(axes), fraction=0.02, pad=0.02, label=r"$E-E_\mathrm{GM}$")
    fig.text(0.5, 0.01, CAPTION, ha="center", va="bottom", fontsize=8)
    print("caption:", CAPTION)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
