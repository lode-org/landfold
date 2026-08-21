#!/usr/bin/env python3
"""Dump unique quenched Cartesian minima from a catalog-requests-v5.bin.

Walks length-prefixed Cap'n Proto frames and collects List(Float64) of
the expected coordinate length (38*3 / 75*3 / 98*3). Energy is the
lowest in-range scalar f64 in the same frame that is not a coordinate.
"""
import argparse
import json
import struct
import sys
from pathlib import Path


def list_ptr(word):
    if (word & 3) != 1:
        return None
    off = (word >> 2) & 0x3FFFFFFF
    if off >= 0x20000000:
        off -= 0x40000000
    elem_size = (word >> 32) & 7
    count = word >> 35
    if elem_size != 5 or count <= 0 or count > 4096:
        return None
    return off, int(count)


def lists_f64(payload, want):
    n_words = len(payload) // 8
    if n_words < 2:
        return []
    words = struct.unpack_from("<" + "Q" * n_words, payload)
    found = []
    seen_off = set()
    for i, w in enumerate(words):
        parsed = list_ptr(w)
        if parsed is None:
            continue
        off, count = parsed
        if count != want:
            continue
        start = i + 1 + off
        if start < 0 or start + count > n_words:
            continue
        if start in seen_off:
            continue
        seen_off.add(start)
        vals = struct.unpack_from("<" + "d" * count, payload, start * 8)
        if all(-40.0 < v < 40.0 for v in vals) and any(abs(v) > 1e-6 for v in vals):
            found.append((start * 8, vals))
    return found


def frame_energy(payload, coord_spans, lo, hi):
    n = len(payload) // 8
    blocked = set()
    for start, nbytes in coord_spans:
        for off in range(start, start + nbytes, 8):
            blocked.add(off)
    best = None
    for i in range(n):
        off = i * 8
        if off in blocked:
            continue
        (v,) = struct.unpack_from("<d", payload, off)
        if lo < v < hi:
            if best is None or v < best:
                best = v
    return best


def cluster_like(coords):
    n = len(coords) // 3
    if n < 4:
        return False
    xs = coords[0::3]
    ys = coords[1::3]
    zs = coords[2::3]
    cx = sum(xs) / n
    cy = sum(ys) / n
    cz = sum(zs) / n
    r2 = [((xs[i] - cx) ** 2 + (ys[i] - cy) ** 2 + (zs[i] - cz) ** 2) for i in range(n)]
    rms = (sum(r2) / n) ** 0.5
    return 1.2 < rms < 12.0


def extract(journal, n_coord, elo, ehi):
    data = journal.read_bytes()
    i = 0
    n = len(data)
    frames = 0
    hits = 0
    seen = {}
    while i + 8 <= n:
        (length,) = struct.unpack_from("<Q", data, i)
        i += 8
        if length == 0 or i + length > n:
            break
        payload = data[i : i + length]
        i += length
        frames += 1
        lists = lists_f64(payload, n_coord)
        if not lists:
            continue
        coords = None
        for start, vals in lists:
            if cluster_like(vals):
                coords = (start, vals)
                break
        if coords is None:
            continue
        spans = [(s, n_coord * 8) for s, _ in lists]
        energy = frame_energy(payload, spans, elo, ehi)
        if energy is None:
            continue
        hits += 1
        key = (round(energy, 5), tuple(round(c, 2) for c in coords[1][::6]))
        prev = seen.get(key)
        if prev is None or energy < prev[0]:
            seen[key] = (energy, coords[1])
    return frames, hits, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("journal")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-n", "--n-atoms", type=int, required=True)
    ap.add_argument("--elo", type=float, required=True)
    ap.add_argument("--ehi", type=float, required=True)
    args = ap.parse_args()
    journal = Path(args.journal)
    n_coord = args.n_atoms * 3
    frames, hits, seen = extract(journal, n_coord, args.elo, args.ehi)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(seen.values(), key=lambda r: r[0])
    with out.open("w") as w:
        for energy, coords in rows:
            w.write(f"{energy:.10e}")
            for c in coords:
                w.write(f" {c:.8e}")
            w.write("\n")
    meta = {
        "journal": str(journal),
        "bytes": journal.stat().st_size,
        "frames": frames,
        "candidate_hits": hits,
        "unique": len(rows),
        "n_atoms": args.n_atoms,
        "e_min": rows[0][0] if rows else None,
        "e_max": rows[-1][0] if rows else None,
        "out": str(out),
    }
    print(json.dumps(meta, indent=2))
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
