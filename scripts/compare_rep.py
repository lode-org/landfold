#!/usr/bin/env python3
"""Score two-well separation of a 2-D embedding against Ceriotti HD.

Reads landmark HD (+weight), two reference CV rows (fcc, ico), and one
or more low-D maps. Reports inter/intra centroid gap and 10-NN overlap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_table(path: Path, dim: int, weighted: bool) -> tuple[np.ndarray, np.ndarray | None]:
    raw = np.loadtxt(path)
    raw = np.atleast_2d(raw)
    need = dim + int(weighted)
    if raw.shape[1] < need:
        raise SystemExit(f"{path} has {raw.shape[1]} cols, need {need}")
    pts = raw[:, :dim]
    w = raw[:, dim] if weighted else None
    return pts, w


def nn10_overlap(hd: np.ndarray, ld: np.ndarray) -> float:
    n = hd.shape[0]
    k = min(10, n - 1)
    if k <= 0:
        return 0.0
    acc = 0.0
    for i in range(n):
        d_h = np.linalg.norm(hd - hd[i], axis=1)
        d_l = np.linalg.norm(ld - ld[i], axis=1)
        d_h[i] = np.inf
        d_l[i] = np.inf
        hset = set(np.argpartition(d_h, k)[:k].tolist())
        lset = set(np.argpartition(d_l, k)[:k].tolist())
        acc += len(hset & lset) / k
    return acc / n


def score(hd: np.ndarray, ld: np.ndarray, ref_a: np.ndarray, ref_b: np.ndarray) -> dict:
    da = np.linalg.norm(hd - ref_a, axis=1)
    db = np.linalg.norm(hd - ref_b, axis=1)
    lab = da <= db
    a = ld[lab]
    b = ld[~lab]
    out = {
        "n_a": int(lab.sum()),
        "n_b": int((~lab).sum()),
        "nn10": nn10_overlap(hd, ld),
    }
    if a.size == 0 or b.size == 0:
        out["gap"] = 0.0
        out["intra"] = 0.0
        out["ratio"] = 0.0
        return out
    ca = a.mean(axis=0)
    cb = b.mean(axis=0)
    gap = float(np.linalg.norm(ca - cb))
    intra = 0.5 * (
        float(np.mean(np.linalg.norm(a - ca, axis=1)))
        + float(np.mean(np.linalg.norm(b - cb, axis=1)))
    )
    out["gap"] = gap
    out["intra"] = intra
    out["ratio"] = gap / intra if intra > 1e-12 else 0.0
    ia = int(np.argmin(da))
    ib = int(np.argmin(db))
    out["ref_dist"] = float(np.linalg.norm(ld[ia] - ld[ib]))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hd", type=Path, required=True)
    p.add_argument("--refs", type=Path, required=True, help="two rows: fcc, ico CVs")
    p.add_argument("--dim", type=int, default=10)
    p.add_argument("-w", dest="weighted", action="store_true")
    p.add_argument("maps", nargs="+", type=Path)
    args = p.parse_args()
    hd, _ = load_table(args.hd, args.dim, args.weighted)
    refs = np.loadtxt(args.refs)
    refs = np.atleast_2d(refs)
    if refs.shape[0] < 2:
        raise SystemExit("refs need two rows")
    print(f"{'map':<24} {'nA':>4} {'nB':>4} {'gap':>8} {'intra':>8} {'ratio':>8} {'refd':>8} {'nn10':>8}")
    for path in args.maps:
        ld = np.loadtxt(path)
        ld = np.atleast_2d(ld)[:, :2]
        if ld.shape[0] != hd.shape[0]:
            raise SystemExit(f"{path} rows {ld.shape[0]} != HD {hd.shape[0]}")
        s = score(hd, ld, refs[0], refs[1])
        print(
            f"{path.name:<24} {s['n_a']:4d} {s['n_b']:4d} "
            f"{s['gap']:8.3f} {s['intra']:8.3f} {s['ratio']:8.3f} "
            f"{s.get('ref_dist', 0.0):8.3f} {s['nn10']:8.3f}"
        )


if __name__ == "__main__":
    main()
