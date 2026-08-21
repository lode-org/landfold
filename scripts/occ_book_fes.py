#!/usr/bin/env python3
"""Elja-book FES/PES through chemparseplot (grad_imq), not a homemade interpolant.

Occupancy: landfold invert F = -kT ln(rho/rhomax) as a regular grid, then
chemparseplot.plot_fes(..., method='grad_imq') — same stack as
docs/ceriotti-figs/lj38_chemgp_surface.png.

Energy: EnergyRepresentation of the quenched E on that plane, then
chemparseplot.plot_energy(..., method='grad_imq').
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CF = ROOT / "examples" / "cosmo-lj38" / "compose_fes.py"
OUT = ROOT / "docs" / "ceriotti-figs"
SRC = Path("/tmp/landfold-occ-from-terra/landfold-occ-book")
if not SRC.is_dir():
    SRC = Path("/tmp/landfold-occ-book")


def _cf():
    spec = importlib.util.spec_from_file_location("compose_fes", CF)
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


def write_fes_csv(xy: np.ndarray, dest: Path, weights=None, ngrid: int = 160, kt: float = 0.168):
    """Regular # x y F rho grid, iy-major, matching landfold fes --csv."""
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
        x, y, bins=ngrid, range=[[xmin, xmax], [ymin, ymax]], weights=weights
    )
    rho = cf._blur2d(counts.T, sigma=5.0)
    gx = 0.5 * (xedges[:-1] + xedges[1:])
    gy = 0.5 * (yedges[:-1] + yedges[1:])
    rmax = float(rho.max())
    mask = rho > 0.004 * rmax
    mask = cf._fill_mask_holes(mask)
    fes = np.full_like(rho, np.nan)
    on = mask & (rho > 0.0)
    fes[on] = -kt * np.log(np.clip(rho[on] / rmax, 1e-12, 1.0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as w:
        w.write("# x y F rho\n")
        for iy, yv in enumerate(gy):
            for ix, xv in enumerate(gx):
                fv = fes[iy, ix]
                rv = rho[iy, ix]
                fs = "nan" if not np.isfinite(fv) else f"{fv:.10e}"
                w.write(f"{xv:.8f} {yv:.8f} {fs} {rv:.10e}\n")
            w.write("\n")
    return dest


def main() -> None:
    from chemparseplot.parse.landfold import load_fes_csv
    from chemparseplot.parse.representation import EnergyRepresentation
    from chemparseplot.plot.landfold import plot_fes
    from chemparseplot.plot.neb import SurfaceFitConfig
    from chemparseplot.plot.representation import plot_energy

    energy = np.loadtxt("/tmp/occ-book/lj38.energy")
    fcc, ico = 0, 40
    OUT.mkdir(parents=True, exist_ok=True)
    fit = SurfaceFitConfig(auto_thin=True, max_surface_points=300)

    maps = [
        ("lj38_ceriotti.proj", "ceriotti", "Ceriotti"),
        ("lj38_asinh.proj", "asinh", "asinh"),
        ("lj38_pacmap.proj", "pacmap", "PaCMAP"),
    ]
    for fname, tag, title in maps:
        path = SRC / fname
        if not path.is_file() or path.stat().st_size < 100:
            print("missing", path)
            continue
        xy = load_xy(path)
        if len(xy) != len(energy):
            print("row mismatch", fname, len(xy), len(energy))
            continue
        tips = {
            "fcc": (float(xy[fcc, 0]), float(xy[fcc, 1])),
            "ico": (float(xy[ico, 0]), float(xy[ico, 1])),
        }
        csv = Path(f"/tmp/occ-book/lj38_{tag}_fes.csv")
        write_fes_csv(xy, csv)
        fes = load_fes_csv(csv, kt=0.168)
        fig = plot_fes(
            fes,
            on="density",
            floor=0.004,
            fmax=2.0,
            clabel=r"$F/\varepsilon$",
            method="grad_imq",
            points=tips,
            show_pts=False,
            surface_fit=fit,
        )
        dest = OUT / f"elja_occ_lj38_{tag}_fes.png"
        fig.savefig(dest, dpi=170, facecolor="white")
        print("wrote", dest)

        # Quenched energy on the same plane (MethodsX field).
        z = energy - energy.min()
        fig = plot_fes(
            cloud=(xy[:, 0], xy[:, 1], z),
            fmax=2.0,
            clabel=r"$E-E_{\min}$",
            method="grad_imq",
            points=tips,
            show_pts=False,
            surface_fit=fit,
        )
        dest = OUT / f"elja_occ_lj38_{tag}_pes.png"
        fig.savefig(dest, dpi=170, facecolor="white")
        print("wrote", dest)

        rng = np.random.default_rng(1)
        pick = rng.choice(len(energy), size=min(400, len(energy)), replace=False)
        pick = np.unique(np.concatenate([pick, [fcc, ico]]))
        rep = EnergyRepresentation.from_mapping(
            {
                "schema": "chemparseplot.energy.v1",
                "x": xy[pick, 0],
                "y": xy[pick, 1],
                "energy": z[pick],
                "frame": "landfold",
                "xlabel": r"$s_1$",
                "ylabel": r"$s_2$",
                "metadata": {"field": "quenched_energy", "embedder": tag},
            }
        )
        fig = plot_energy(
            rep,
            method="grad_imq",
            clabel=r"$E-E_{\min}$",
            show_pts=False,
            extra_points=tips,
            surface_fit=fit,
        )
        dest = OUT / f"elja_occ_lj38_{tag}_energy.png"
        fig.savefig(dest, dpi=170, facecolor="white")
        print("wrote", dest)


if __name__ == "__main__":
    main()
