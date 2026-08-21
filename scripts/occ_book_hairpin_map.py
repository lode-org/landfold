#!/usr/bin/env python3
"""PNAS-class hairpin landmark map: published Ceriotti vs asinh.

Quality number lives in P(D,d). This panel is the map a reader sees:
1000 published landmarks, same color (distance from the native-like
landmark), so the far-order claim is visible as spread vs collapse.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ceriotti-figs"
PUB = Path.home() / "Git/Github/HaoZeke/sketchmap/examples/protein"
ASINH = Path("/tmp/landfold-asinh/protein_asinh.ld")
HD = PUB / "lm4.30cv.w01.1"
PUB_LD = PUB / "lm4.30cv.w01.1.proj"


def load_xy(path: Path, cols=(0, 1)) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        rows.append([float(p[cols[0]]), float(p[cols[1]])])
    return np.asarray(rows)


def load_hd(path: Path, dim: int = 30) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = [float(x) for x in line.split()[:dim]]
        rows.append(p)
    return np.asarray(rows)


def period_l2(a: np.ndarray, b: np.ndarray, period: float = 2 * np.pi) -> np.ndarray:
    d = a - b
    d = d - period * np.round(d / period)
    return np.sqrt((d * d).sum(axis=1))


def main() -> None:
    hd = load_hd(HD)
    pub = load_xy(PUB_LD)
    ash = load_xy(ASINH)
    # native-like = smallest radius of gyration of dihedrals about 0
    # (beta-hairpin folded landmarks sit near the published origin cluster)
    native = hd[np.argmin(np.linalg.norm(pub, axis=1))]
    color = period_l2(hd, native)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), facecolor="white")
    for ax, xy, title in (
        (axes[0], pub, "Ceriotti 6,8,8 / 6,2,8"),
        (axes[1], ash, r"asinh $\sigma=6$"),
    ):
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=color,
            s=8,
            cmap="magma_r",
            linewidths=0,
            vmin=np.quantile(color, 0.02),
            vmax=np.quantile(color, 0.98),
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
    cax = fig.add_axes([0.30, 0.08, 0.40, 0.03])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label(r"periodic $D$ to the most native-like landmark", fontsize=9)
    fig.suptitle("beta-hairpin landmarks (Ardevol 2015)", fontsize=12, y=0.98)
    dest = OUT / "protein_asinh_map.png"
    fig.savefig(dest, dpi=170, facecolor="white", bbox_inches="tight")
    print("wrote", dest)
    print("color range", float(color.min()), float(color.max()))


if __name__ == "__main__":
    main()
