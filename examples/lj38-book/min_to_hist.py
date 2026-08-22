#!/usr/bin/env python3
"""Read a .min book (E + 3N per row) and write SOAP or sorted-pair HD as .hist.

SOAP mean (n_r=16, n_a=8) is the 24-D engine in scripts/occ_book_soap.py.
Sorted internuclear pairs are the 703-D LJ38 engine in scripts/occ_book_dpair.py.
sigma is the median positive pairwise L2 of the fingerprint table, the
scale used by landfold embed --fun-hd asinh,sigma.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def write_table(path: Path, arr: np.ndarray, comment: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        if comment:
            fh.write(f"# {comment}\n")
        np.savetxt(fh, arr, fmt="%.8e")
    print("wrote", path, arr.shape)


def median_l2_sigma(fp: np.ndarray) -> float:
    import occ_book_soap as soap

    dist = soap.pairwise_l2(fp)
    pos = dist[dist > 0]
    sigma = float(np.median(pos)) if pos.size else 1.0
    return max(sigma, 1e-9)


def farthest_on_d(dist: np.ndarray, nland: int, must) -> np.ndarray:
    n = dist.shape[0]
    chosen = []
    seen = set()
    for i in must:
        i = int(i)
        if 0 <= i < n and i not in seen:
            chosen.append(i)
            seen.add(i)
    dmin = np.full(n, np.inf)
    for i in chosen:
        dmin = np.minimum(dmin, dist[i])
    nland = min(max(nland, len(chosen)), n)
    while len(chosen) < nland:
        j = int(np.argmax(dmin))
        if j in seen:
            dmin[j] = -1.0
            continue
        chosen.append(j)
        seen.add(j)
        dmin = np.minimum(dmin, dist[j])
    return np.asarray(chosen, dtype=int)


def soap_table(minfile: Path, n_atoms: int, n_r: int, n_a: int, kind: str, dest: Path, workers: int):
    import occ_book_soap as soap

    soap.MINFILE = minfile
    soap.DEST = dest
    soap.N_ATOMS = n_atoms
    energy, frames = soap.load_min(minfile, n_atoms)
    if workers > 1:
        fp = soap.build_family(frames, n_r, n_a, workers=workers)[kind]
    else:
        fp = soap.build_fps(frames, kind, n_r, n_a)
    return energy, fp


def pair_table(minfile: Path, n_atoms: int):
    import occ_book_dpair as dpair

    energy, frames = dpair.load_min(minfile, n_atoms)
    hd = np.vstack([dpair.sorted_pairs(pos) for pos in frames])
    return energy, hd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min", required=True, type=Path, help="book: energy then 3N coordinates")
    ap.add_argument("--out", required=True, type=Path, help="landfold .hist")
    ap.add_argument("--kind", choices=("soap", "pair"), default="soap")
    ap.add_argument("--n-atoms", type=int, default=38)
    ap.add_argument("--energy", type=Path, default=None, help="one energy per row")
    ap.add_argument("--sigma", type=Path, default=None, help="median pairwise L2")
    ap.add_argument("--landmarks", type=Path, default=None, help="farthest-point landmark .hist")
    ap.add_argument("--land-idx", type=Path, default=None)
    ap.add_argument("--n-land", type=int, default=0, help="write --landmarks with this many rows")
    ap.add_argument("--pin", type=int, action="append", default=[], help="landmark index to pin")
    ap.add_argument("--dest", type=Path, default=None, help="SOAP worker cache")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-r", type=int, default=16)
    ap.add_argument("--n-a", type=int, default=8)
    ap.add_argument("--soap-kind", choices=("mean", "meanstd", "sorted"), default="mean")
    args = ap.parse_args()
    dest = args.dest or args.out.parent
    dest.mkdir(parents=True, exist_ok=True)

    if args.kind == "soap":
        energy, hd = soap_table(
            args.min, args.n_atoms, args.n_r, args.n_a, args.soap_kind, dest, args.workers
        )
        comment = (
            f"SOAP {args.soap_kind} n_r={args.n_r} n_a={args.n_a} "
            f"rows={hd.shape[0]} D={hd.shape[1]}"
        )
    else:
        energy, hd = pair_table(args.min, args.n_atoms)
        comment = f"sorted internuclear pairs descending rows={hd.shape[0]} D={hd.shape[1]}"

    write_table(args.out, hd, comment)
    if args.energy is not None:
        write_table(args.energy, energy.reshape(-1, 1), f"energy rows={energy.shape[0]}")

    sigma = median_l2_sigma(hd)
    if args.sigma is not None:
        args.sigma.write_text("%.10e\n" % sigma)
        print("wrote", args.sigma, "sigma", sigma)
    else:
        print("sigma", sigma)

    if args.n_land > 0:
        import occ_book_soap as soap

        dist = soap.pairwise_l2(hd)
        np.fill_diagonal(dist, 0.0)
        must = list(args.pin)
        if not must:
            must = [int(np.argmin(energy))]
        idx = farthest_on_d(dist, args.n_land, must)
        lm_path = args.landmarks if args.landmarks is not None else args.out.with_suffix(".lm")
        write_table(lm_path, hd[idx], f"landmarks n={len(idx)} of {hd.shape[0]}")
        idx_path = args.land_idx if args.land_idx is not None else lm_path.with_suffix(".idx")
        np.savetxt(idx_path, idx, fmt="%d")
        print("wrote", idx_path, "nland", len(idx))


if __name__ == "__main__":
    main()
