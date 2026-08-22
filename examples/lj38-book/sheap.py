#!/usr/bin/env python3
"""SHEAP dual-funnel layout of a minima book, filled with energy IDW.

Calls scripts/occ_book_sheap.py for the structure-kNN dual funnel and
scripts/occ_book_sheap_idw.py for compact-support IDW of E-E_GM.
MIN and ENERGY may be flags or environment variables. Paths stay in
--dest / $OUT; this example does not write into a home directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import occ_book_sheap as sheap
import occ_book_sheap_idw as sheap_idw


def env_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else None


def load_energy(path: Path) -> np.ndarray:
    energy = np.loadtxt(path)
    return np.asarray(energy, dtype=float).reshape(-1)


def write_energy(path: Path, energy: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(f"# energy rows={energy.shape[0]}\n")
        np.savetxt(fh, energy.reshape(-1, 1), fmt="%.8e")
    print("wrote", path, energy.shape[0])


def seed_dpair(dest: Path, pair_hist: Path | None) -> None:
    """Reuse pair HD as the SHEAP structure distance if the table exists."""
    cache = dest / "dpair.npy"
    if cache.is_file():
        print("reuse", cache)
        return
    if pair_hist is None or not pair_hist.is_file():
        return
    hd = np.loadtxt(pair_hist)
    if hd.ndim != 2:
        return
    dist = sheap.pairwise_l2(hd)
    dest.mkdir(parents=True, exist_ok=True)
    np.save(cache, dist)
    print("wrote", cache, dist.shape)


def bind(minfile: Path, energy: Path, dest: Path, fig: Path, natoms: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    os.environ["MIN"] = str(minfile)
    os.environ["ENERGY"] = str(energy)
    os.environ["SHEAP_DEST"] = str(dest)
    os.environ["SHEAP_OUT"] = str(dest)
    os.environ["SHEAP_FIG"] = str(fig)
    os.environ.setdefault("NATOMS", str(natoms))
    sheap.MINFILE = minfile
    sheap.DEST = dest
    sheap.OUT = dest
    sheap.DPAIR_CACHE = dest / "lj38_dpair.dist"
    sheap.KABSCH_CACHE = dest / "lj38_kabsch.dist"
    sheap.N_ATOMS = natoms
    sheap_idw.BOOK = dest
    sheap_idw.CAND = dest
    sheap_idw.ENERGY = energy
    sheap_idw.OUT = dest
    sheap_idw.FIG = fig
    sheap_idw.XY_CANDIDATES = (dest / "sheap.xy", dest / "asinh_sheap.xy")


def resolve_ico(energy: np.ndarray) -> int | None:
    ico = int(np.argmin(np.abs(energy - sheap_idw.ICO_E)))
    if abs(float(energy[ico]) - sheap_idw.ICO_E) > 1e-3:
        return None
    return ico


def paint(xy: np.ndarray, energy: np.ndarray, fig_path: Path, title: str) -> None:
    gm = int(np.argmin(energy))
    ico = resolve_ico(energy)
    z = energy - float(energy[gm])
    k = min(6, xy.shape[0])
    gx, gy, field = sheap_idw.fill(xy, z, k=k)
    fgm = sheap_idw.sample(gx, gy, field, xy[gm])
    fico = (
        sheap_idw.sample(gx, gy, field, xy[ico]) if ico is not None else float("nan")
    )
    print(f"SHEAP energy IDW F(GM)={fgm:.4f} F(ico)={fico:.4f}")

    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor="white")
    mesh = ax.pcolormesh(
        gx,
        gy,
        np.clip(field, 0, sheap_idw.VMAX),
        cmap=sheap_idw.PES,
        shading="auto",
        vmin=0,
        vmax=sheap_idw.VMAX,
    )
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=np.clip(z, 0, sheap_idw.VMAX),
        s=7,
        cmap=sheap_idw.PES,
        vmin=0,
        vmax=sheap_idw.VMAX,
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
        label=rf"GM ${float(energy[gm]):.3f}$",
    )
    if ico is not None:
        ax.scatter(
            xy[ico, 0],
            xy[ico, 1],
            s=95,
            marker="D",
            c="k",
            edgecolors="white",
            linewidths=0.7,
            zorder=50,
            label=rf"ico ${float(energy[ico]):.3f}$",
        )
    ax.legend(loc="best", fontsize=8, frameon=True, fancybox=False)
    ax.set_xlabel(r"SHEAP$_1$")
    ax.set_ylabel(r"SHEAP$_2$")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(mesh, ax=ax, label=r"$E-E_\mathrm{GM}$")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=180, facecolor="white")
    plt.close(fig)
    print("wrote", fig_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--min",
        type=Path,
        default=env_path("MIN"),
        help="book: energy then 3N coordinates (env MIN)",
    )
    ap.add_argument(
        "--energy",
        type=Path,
        default=env_path("ENERGY"),
        help="one energy per row (env ENERGY); default extract from --min",
    )
    ap.add_argument("--out", type=Path, default=None, help="sheap_idw.png")
    ap.add_argument(
        "--dest",
        type=Path,
        default=env_path("SHEAP_DEST") or env_path("DEST") or env_path("OUT"),
        help="working directory for sheap.xy (env SHEAP_DEST / DEST / OUT)",
    )
    ap.add_argument("--xy", type=Path, default=None, help="existing sheap.xy; skip layout")
    ap.add_argument(
        "--pair-hist",
        type=Path,
        default=None,
        help="sorted-pair .hist used as the structure distance",
    )
    ap.add_argument(
        "--n-atoms",
        type=int,
        default=int(os.environ.get("NATOMS", "38")),
    )
    ap.add_argument(
        "--title",
        default=r"SHEAP dual-funnel  energy IDW  $E-E_{\mathrm{GM}}$",
    )
    args = ap.parse_args()
    if args.min is None:
        raise SystemExit("set --min or MIN to the .min book")
    minfile = args.min
    if not minfile.is_file():
        raise SystemExit(f"missing MIN={minfile}")

    dest = args.dest
    if dest is None:
        dest = args.out.parent if args.out is not None else Path.cwd()
    dest = dest.resolve()
    fig_path = args.out if args.out is not None else dest / "sheap_idw.png"
    fig_path = fig_path.resolve()

    if args.energy is not None and args.energy.is_file():
        energy_path = args.energy
        energy = load_energy(energy_path)
    else:
        energy, _frames = sheap.load_min(minfile, args.n_atoms)
        energy_path = dest / "book.energy"
        write_energy(energy_path, energy)

    pair_hist = args.pair_hist
    if pair_hist is None:
        candidate = dest / "pair.hist"
        pair_hist = candidate if candidate.is_file() else None

    bind(minfile, energy_path, dest, fig_path, args.n_atoms)
    if args.xy is None:
        seed_dpair(dest, pair_hist)
        sheap.main()
        xy_path = dest / "sheap.xy"
    else:
        xy_path = args.xy
        sheap_idw.XY_CANDIDATES = (xy_path,)
    if not xy_path.is_file():
        raise SystemExit(f"missing {xy_path}")

    xy = np.loadtxt(xy_path)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise SystemExit(f"{xy_path}: expected (*, >=2), got {xy.shape}")
    xy = np.asarray(xy[:, :2], dtype=float)
    energy = load_energy(energy_path)
    if energy.shape[0] != xy.shape[0]:
        raise SystemExit(f"energy {energy.shape} vs xy {xy.shape}")
    paint(xy, energy, fig_path, args.title)


if __name__ == "__main__":
    main()
