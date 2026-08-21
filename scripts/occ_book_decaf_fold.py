#!/usr/bin/env python3
"""Switch vs asinh Torgerson of the DECAF occupancy book.

Production map: Ceriotti switch on L1, then Torgerson (y ~ 0).
asinh keeps far packing L1 ordered. Color is leftover wells.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parents[1] / "docs" / "ceriotti-figs"
PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)


def load_book(path: Path):
    text = path.read_text().strip()
    # last JSON object if the file is JSONL
    last = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        last = json.loads(line)
    if last is None:
        last = json.loads(text)
    pts = last["points"]
    switch = np.array([[p["x"], p["y"]] for p in pts], float)
    asinh = np.array([[p["asinh_x"], p["asinh_y"]] for p in pts], float)
    wells = np.array([p["wells"] for p in pts], float)
    return last, switch, asinh, wells


def panel(ax, xy, wells, title, spectrum=None):
    sizes = 12 + 40 * np.sqrt(wells / max(wells.max(), 1.0))
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=wells, cmap="magma", linewidths=0)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    if spectrum is not None:
        l1, l2 = spectrum
        ax.text(
            0.04,
            0.96,
            f"λ2/λ1 = {l2 / max(l1, 1e-15):.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(fc="white", ec="none", alpha=0.85),
        )
    return sc


def main() -> None:
    src = Path("/tmp/occ-book/lj38_decaf_book.json")
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    meta, switch, asinh, wells = load_book(src)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), facecolor="white")
    panel(
        axes[0],
        switch,
        wells,
        "switch Torgerson (production)",
        (meta.get("switch_l1"), meta.get("switch_l2")),
    )
    sc = panel(
        axes[1],
        asinh,
        wells,
        "asinh Torgerson (far L1 ordered)",
        (meta.get("asinh_l1"), meta.get("asinh_l2")),
    )
    fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02, label="leftover wells")
    fig.suptitle("DECAF histogram book, L1 fold")
    dest = OUT / "elja_occ_lj38_decaf_fold.png"
    fig.savefig(dest, dpi=170, facecolor="white")
    print("wrote", dest)
    print(
        "n",
        len(wells),
        "switch λ",
        meta.get("switch_l1"),
        meta.get("switch_l2"),
        "asinh λ",
        meta.get("asinh_l1"),
        meta.get("asinh_l2"),
    )


if __name__ == "__main__":
    main()
