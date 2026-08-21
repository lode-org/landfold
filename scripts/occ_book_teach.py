#!/usr/bin/env python3
"""Ceriotti-class teaching FES of the Elja LJ38 book.

asinh chi of n4..n13. Occupancy F/eps. xyzrender --config paton of
the journal minima. Layout matches examples/cosmo-lj38/compose_fes.py:
no on-map text, four corner frames, CN bars, F labels.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "cosmo-lj38"
OUT = ROOT / "docs" / "ceriotti-figs"
PROJ = Path("/tmp/landfold-occ-from-terra/landfold-occ-book/lj38_asinh.proj")
if not PROJ.is_file():
    PROJ = Path("/tmp/landfold-occ-book/lj38_asinh.proj")
CSV = Path("/tmp/occ-book/lj38_asinh_fes.csv")
XYZ = Path("/tmp/occ-book/xyz")
RENDER = Path("/tmp/occ-book/render")
MINFILE = Path("/tmp/occ-book/lj38_0013.min")

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)

# fcc, ico, second family, liquid rim (top of the main well)
PICKS = {"fcc": 0, "ico": 40, "left": 1393, "liquid": None}


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


def write_xyz(min_rows: list[str], index: int, dest: Path, label: str) -> float:
    nums = [float(x) for x in min_rows[index].split()]
    energy, coords = nums[0], nums[1:]
    lines = ["38", f"E={energy:.6f} {label}"]
    for k in range(38):
        x, y, z = coords[3 * k : 3 * k + 3]
        lines.append(f"Ar {x:.8f} {y:.8f} {z:.8f}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")
    return energy


def xyzrender(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "xyzrender",
            "-t",
            "--orient",
            "--config",
            "paton",
            "--no-fog",
            "-S",
            "640",
            "--atom-stroke-width",
            "0.5",
            "--bond-outline-width",
            "0",
            "--mol-color",
            "#2a6f97",
            "-o",
            str(dest),
            str(src),
        ],
        check=True,
    )


def write_fes(xy: np.ndarray, dest: Path, ngrid: int = 160, kt: float = 0.168):
    cf = _cf()
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= 0.10 * dx
    xmax += 0.10 * dx
    ymin -= 0.10 * dy
    ymax += 0.10 * dy
    counts, xedges, yedges = np.histogram2d(
        x, y, bins=ngrid, range=[[xmin, xmax], [ymin, ymax]]
    )
    rho = cf._blur2d(counts.T, sigma=5.0)
    gx = 0.5 * (xedges[:-1] + xedges[1:])
    gy = 0.5 * (yedges[:-1] + yedges[1:])
    rmax = float(rho.max())
    mask = cf._fill_mask_holes(rho > 0.004 * rmax)
    fes = np.full_like(rho, np.nan)
    on = mask & (rho > 0.0)
    fes[on] = -kt * np.log(np.clip(rho[on] / rmax, 1e-12, 1.0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as w:
        w.write("# x y F rho\n")
        for iy, yv in enumerate(gy):
            for ix, xv in enumerate(gx):
                fv = fes[iy, ix]
                fs = "nan" if not np.isfinite(fv) else f"{fv:.10e}"
                w.write(f"{xv:.8f} {yv:.8f} {fs} {rho[iy, ix]:.10e}\n")
            w.write("\n")


def load_fes(path: Path):
    raw = np.loadtxt(path, comments="#")

    def uniq(col):
        _, idx = np.unique(np.round(raw[:, col], 12), return_index=True)
        return raw[np.sort(idx), col]

    x = uniq(0)
    y = uniq(1)
    fes = raw[:, 2].reshape(y.size, x.size)
    return x, y, fes


def load_frame(png: Path) -> Image.Image:
    """Drop the black canvas. Keep teal atoms."""
    arr = np.asarray(Image.open(png).convert("RGBA")).copy()
    lum = arr[:, :, :3].max(axis=2)
    arr[lum < 40, 3] = 0
    ys, xs = np.where(arr[:, :, 3] > 10)
    if ys.size:
        pad = 8
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad + 1, arr.shape[0])
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad + 1, arr.shape[1])
        arr = arr[y0:y1, x0:x1]
    return Image.fromarray(arr)


def main() -> None:
    cf = _cf()
    xy = load_xy(PROJ)
    e = np.loadtxt("/tmp/occ-book/lj38.energy")
    cv = np.loadtxt("/tmp/occ-book/lj38.cv")
    if not CSV.is_file():
        write_fes(xy, CSV)
    gx, gy, fes = load_fes(CSV)

    # Liquid inset: highest s2 inside the main well, so the leader is short.
    mask = (e > -172.5) & (xy[:, 0] > -5.0) & (xy[:, 0] < 8.0)
    liquid_i = int(np.argmax(np.where(mask, xy[:, 1], -1e9)))
    picks = {"fcc": 0, "ico": 40, "left": 1393, "liquid": liquid_i}

    min_rows = [ln for ln in MINFILE.read_text().splitlines() if ln.strip()]
    energies = {}
    for name, index in picks.items():
        src = XYZ / f"lj38_{name}.xyz"
        dest = RENDER / f"lj38_{name}.png"
        energies[name] = write_xyz(min_rows, index, src, name)
        if not dest.is_file():
            xyzrender(src, dest)

    fig = plt.figure(figsize=(13.0, 9.8), facecolor="white")
    ax = fig.add_axes([0.22, 0.20, 0.56, 0.64])
    ax.set_facecolor("white")
    mesh = ax.contourf(gx, gy, fes, levels=np.linspace(0, 2, 21), cmap=PES, extend="max")
    finite = np.isfinite(fes)
    ax.contour(
        gx,
        gy,
        np.where(finite, fes, np.nan),
        levels=np.linspace(0.15, 1.85, 12),
        colors="#1a1a2e",
        linewidths=0.35,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    cax = fig.add_axes([0.30, 0.10, 0.40, 0.026])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label(r"$F/\varepsilon$", fontsize=12)
    cb.set_ticks([0, 0.5, 1.0, 1.5, 2.0])

    panels = [
        ("liquid", picks["liquid"], (0.13, 0.76), "liquid"),
        ("left", picks["left"], (0.13, 0.34), "other packing"),
        ("ico", picks["ico"], (0.85, 0.76), "ico"),
        ("fcc", picks["fcc"], (0.85, 0.34), "fcc"),
    ]

    for name, index, (fx, fy), label in panels:
        png = RENDER / f"lj38_{name}.png"
        xyz = XYZ / f"lj38_{name}.xyz"
        if not png.is_file():
            continue
        tip = xy[index]
        img = load_frame(png)
        fig.add_artist(
            AnnotationBbox(
                OffsetImage(img, zoom=0.095),
                (fx, fy),
                xycoords=fig.transFigure,
                frameon=False,
                box_alignment=(0.5, 0.5),
            )
        )
        ax.annotate(
            "",
            xy=tip,
            xycoords=ax.transData,
            xytext=(fx, fy),
            textcoords=fig.transFigure,
            arrowprops=dict(arrowstyle="-", color="k", lw=0.8),
        )
        fval = cf.fes_at(gx, gy, fes, tip)
        desc = cv[index]
        hax = fig.add_axes([fx - 0.09, fy - 0.22, 0.18, 0.075])
        bins = np.arange(4, 14)
        hax.bar(bins, desc, color="k", width=0.7)
        hax.set_xlim(3.5, 13.5)
        hax.set_xticks(bins)
        hax.set_xticklabels([str(int(v)) for v in bins])
        hax.tick_params(labelsize=6, length=2)
        hax.set_yticks([])
        hax.set_title(f"{label}  {energies[name]:.1f}  F={fval:.2f}", fontsize=8, pad=2)
        for spine in hax.spines.values():
            spine.set_linewidth(0.45)

    dest = OUT / "elja_occ_lj38_teach.png"
    fig.savefig(dest, dpi=170, facecolor="white")
    print("wrote", dest)
    print("picks", picks, {k: float(energies[k]) for k in energies})


if __name__ == "__main__":
    main()
