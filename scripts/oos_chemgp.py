#!/usr/bin/env python3
"""OOS ChemGP of the occupancy invert — the two-basin landscape.

This is the figure class that already showed the wells
(docs/ceriotti-figs/lj38_chemgp_surface.png). It fits
``-kT ln(rho/rhomax)`` on the landfold FES grid with grad_imq, same
stack as ``rgpycrumbs landfold plot-fes``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "cosmo-lj38"
OUT = ROOT / "docs" / "ceriotti-figs"


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "compose_fes", EX / "compose_fes.py"
    )
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)

    from chemparseplot.parse.landfold import load_fes_csv
    from chemparseplot.plot.landfold import plot_fes
    from chemparseplot.plot.neb import SurfaceFitConfig

    csv = EX / "out" / "fes.csv"
    proj = EX / "out" / "ts.proj"
    if not csv.is_file() or not proj.is_file():
        raise SystemExit("need out/fes.csv and out/ts.proj from run.sh")
    xy = cf.load_xy(proj)
    tscv = cf.load_ts_cv(EX / "ts.all")
    points = {}
    for label, xyz_name in (("fcc", "lj38_fcc.xyz"), ("ico", "lj38_ico.xyz")):
        xyz = EX / xyz_name
        if xyz.is_file():
            points[label] = tuple(cf.match_tip(xy, tscv, cf.cn_vector(xyz)))
    fes = load_fes_csv(csv, kt=0.168)
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plot_fes(
        fes,
        on="density",
        floor=0.006,
        fmax=2.0,
        clabel=r"$F/\varepsilon$",
        method="grad_imq",
        points=points or None,
        show_pts=False,
        surface_fit=SurfaceFitConfig(auto_thin=True, max_surface_points=300),
    )
    dest = OUT / "lj38_oos_chemgp.png"
    fig.savefig(dest, dpi=170, facecolor="white")
    print("wrote", dest)
    if points:
        print("tips", points)


if __name__ == "__main__":
    main()
