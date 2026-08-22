#!/usr/bin/env python3
"""Hybrid HD distance of the Elja LJ38 DECAF occupancy book.

Graph: symmetrized kNN=12 in packing L1. Edge cost is the positive
local barrier

    cost_ij = |E_i - E_j| + 2 L1_ij

Geodesic g = Floyd-Warshall of those costs.

Two candidate distances, then classical Torgerson MDS:

  hybrid     0.35 asinh(L1_CN / med L1) + 0.65 asinh(g / med g)
  geo_asinh  landfold asinh of g only:
             F(x) = asinh(x / med g) / (2 asinh 1)

Raw asinh (no median scale) of the same blend is the control: L1 is
O(1) and g is a path sum, so 0.35/0.65 is not a real mix until each
term is scaled.

sep_norm vs Ceriotti and asinh-CN chi coords when those .ld files
exist. Occupancy + energy panels; GM star, ico diamond.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import occ_book_landscape_chi as lc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ceriotti-figs"
HIST = Path("/tmp/occ-book/lj38_decaf_e.hist")
DEST = Path("/tmp/occ-book/cand-hybrid")
CER = Path("/tmp/occ-book/lj38_decaf_cer.ld")
ASH = Path("/tmp/occ-book/lj38_decaf_asinh.ld")
KNN = 12
W_L1 = 0.35
W_GEO = 0.65
GM = 0
ICO = 1
ASINH_NORM = 2.0 * float(np.arcsinh(1.0))


def knn_cost_edges(hist: np.ndarray, energy: np.ndarray, knn: int = KNN):
    """Symmetrized kNN in packing L1. Cost = |dE| + 2 L1."""
    n = len(energy)
    edges = []
    seen = set()
    l1 = np.zeros((n, n))
    for i in range(n):
        drow = np.abs(hist - hist[i]).sum(1)
        l1[i] = drow
        picked = 0
        for j in np.argsort(drow):
            j = int(j)
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b))
                cost = float(abs(energy[i] - energy[j]) + 2.0 * drow[j])
                if cost < 0.0:
                    raise RuntimeError("negative edge %d-%d: %s" % (a, b, cost))
                edges.append((cost, i, j, float(drow[j]), cost))
            picked += 1
            if picked >= knn:
                break
    np.fill_diagonal(l1, 0.0)
    return edges, l1


def n_components(n: int, edges) -> int:
    seen = np.zeros(n, dtype=bool)
    adj = [[] for _ in range(n)]
    for _h, i, j, _d, _c in edges:
        adj[i].append(j)
        adj[j].append(i)
    comps = 0
    for s in range(n):
        if seen[s]:
            continue
        comps += 1
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return comps


def load_xy(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        rows.append([float(p[0]), float(p[1])])
    return np.asarray(rows)


def sep_of(xy: np.ndarray) -> tuple[float, float, float]:
    sep = float(np.linalg.norm(xy[0] - xy[1]))
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    return sep, diam, sep / max(diam, 1e-12)


def asinh_med(dist: np.ndarray) -> tuple[np.ndarray, float]:
    pos = dist[dist > 0]
    sig = float(np.median(pos)) if pos.size else 1.0
    sig = max(sig, 1e-9)
    out = np.arcsinh(dist / sig)
    np.fill_diagonal(out, 0.0)
    return out, sig


def landfold_asinh(dist: np.ndarray) -> tuple[np.ndarray, float]:
    """F(x) = asinh(x/sigma) / (2 asinh 1), sigma = median of positives."""
    out, sig = asinh_med(dist)
    out = out / ASINH_NORM
    np.fill_diagonal(out, 0.0)
    return out, sig


def write_xy(path: Path, xy: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, xy, fmt="%.8e")


def save_pair(xy, gx, gy, fes, energy, dest: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), facecolor="white")
    mesh, sc = lc.panel_row(axes[0], axes[1], xy, gx, gy, fes, energy, title)
    fig.colorbar(mesh, ax=axes[0], fraction=0.046, pad=0.03).set_label(r"$F/\varepsilon$")
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.03).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", dest)


def baseline_sep(path: Path, name: str) -> dict | None:
    if not path.is_file():
        print("skip baseline", name, "no", path)
        return None
    xy = load_xy(path)
    sep, diam, sn = sep_of(xy)
    rec = {
        "name": name,
        "path": str(path),
        "n": int(len(xy)),
        "sep": sep,
        "diam": diam,
        "sep_norm": sn,
        "gm": [float(xy[0, 0]), float(xy[0, 1])],
        "ico": [float(xy[1, 0]), float(xy[1, 1])],
    }
    print(
        f"{name:16s} n={len(xy):4d}  sep={sep:.6f}  diam={diam:.6f}  "
        f"sep_norm={sn:.6f}"
    )
    return rec


def main() -> None:
    hist, energy, wells = lc.load_book(HIST)
    n = len(energy)
    if abs(float(energy[GM]) - lc.GM_E) > 1e-4 or abs(float(energy[ICO]) - lc.ICO_E) > 1e-4:
        raise SystemExit(
            "family 0/1 are not GM/ico: %s %s" % (energy[GM], energy[ICO])
        )
    edges, l1 = knn_cost_edges(hist, energy, knn=KNN)
    comps = n_components(n, edges)
    costs = np.asarray([e[4] for e in edges])
    print(
        "n",
        n,
        "edges",
        len(edges),
        "components",
        comps,
        "kNN",
        KNN,
        "E[GM]",
        float(energy[GM]),
        "E[ico]",
        float(energy[ICO]),
        "L1(GM,ico)",
        float(l1[GM, ICO]),
        "cost_min",
        float(costs.min()),
        "cost_med",
        float(np.median(costs)),
    )
    if comps > 1:
        print("WARN graph has", comps, "components; Floyd will cap disconnected pairs")

    geo = lc.floyd(n, edges)
    geo_gm_ico = float(geo[GM, ICO])
    geo_med = float(np.median(geo[geo > 0]))
    print(
        "geo(GM,ico)",
        geo_gm_ico,
        "geo_med",
        geo_med,
        "geo_max",
        float(geo.max()),
        "finite",
        bool(np.isfinite(geo).all()),
    )

    attract = lc.steepest_basins(n, energy, edges)
    n_gm = int((attract == attract[GM]).sum())
    n_ico = int((attract == attract[ICO]).sum())
    print(
        "steepest attractors",
        int(len(np.unique(attract))),
        "n_GM",
        n_gm,
        "n_ico",
        n_ico,
    )

    ash_l1, sig_l1 = asinh_med(l1)
    ash_geo, sig_geo = asinh_med(geo)
    hyb = W_L1 * ash_l1 + W_GEO * ash_geo
    np.fill_diagonal(hyb, 0.0)

    hyb_raw = W_L1 * np.arcsinh(l1) + W_GEO * np.arcsinh(geo)
    np.fill_diagonal(hyb_raw, 0.0)

    f_geo, sig_lf = landfold_asinh(geo)
    f_l1, _ = landfold_asinh(l1)
    hyb_lf = W_L1 * f_l1 + W_GEO * f_geo
    np.fill_diagonal(hyb_lf, 0.0)

    print(
        "sig_L1",
        sig_l1,
        "sig_geo",
        sig_geo,
        "asinh_norm",
        ASINH_NORM,
        "hyb(GM,ico)",
        float(hyb[GM, ICO]),
        "hyb_raw(GM,ico)",
        float(hyb_raw[GM, ICO]),
        "Fgeo(GM,ico)",
        float(f_geo[GM, ICO]),
    )
    print(
        "term scales  asinh(L1)/asinh(g) med",
        float(np.median(ash_l1[ash_l1 > 0]) / max(np.median(ash_geo[ash_geo > 0]), 1e-12)),
        "raw asinh(L1)/asinh(g) med",
        float(
            np.median(np.arcsinh(l1)[l1 > 0])
            / max(np.median(np.arcsinh(geo)[geo > 0]), 1e-12)
        ),
    )

    maps = {}
    evs = {}
    xy_h, ev_h = lc.torgerson(hyb, 2)
    maps["hybrid"] = xy_h
    evs["hybrid"] = ev_h
    xy_r, ev_r = lc.torgerson(hyb_raw, 2)
    maps["hybrid_raw"] = xy_r
    evs["hybrid_raw"] = ev_r
    xy_g, ev_g = lc.torgerson(f_geo, 2)
    maps["geo_asinh"] = xy_g
    evs["geo_asinh"] = ev_g
    # landfold F on both terms is a global scale of hybrid; keep as a check
    xy_lf, ev_lf = lc.torgerson(hyb_lf, 2)
    maps["hybrid_lf"] = xy_lf
    evs["hybrid_lf"] = ev_lf
    print("Torgerson ev hybrid", ev_h)
    print("Torgerson ev hybrid_raw", ev_r)
    print("Torgerson ev geo_asinh", ev_g)
    print("Torgerson ev hybrid_lf", ev_lf)
    print(
        "hybrid vs hybrid_lf procrustes-free scale",
        float(np.linalg.norm(xy_h) / max(np.linalg.norm(xy_lf), 1e-12)),
    )

    DEST.mkdir(parents=True, exist_ok=True)
    np.savetxt(DEST / "l1.txt", l1, fmt="%.8e")
    np.savetxt(DEST / "geo.txt", geo, fmt="%.8e")
    np.savetxt(DEST / "hybrid.dist", hyb, fmt="%.8e")
    np.savetxt(DEST / "geo_asinh.dist", f_geo, fmt="%.8e")
    for name, xy in maps.items():
        write_xy(DEST / f"{name}.xy", xy)

    print("--- sep_norm baselines ---")
    baselines = []
    for path, name in ((CER, "ceriotti"), (ASH, "asinh_cn")):
        rec = baseline_sep(path, name)
        if rec is not None:
            baselines.append(rec)

    cf = lc._cf()
    scores = []
    titles = {
        "hybrid": r"hybrid $0.35\,\mathrm{asinh}\,L^1+0.65\,\mathrm{asinh}\,g$",
        "hybrid_raw": r"hybrid raw asinh",
        "geo_asinh": r"landfold asinh $g$",
        "hybrid_lf": r"hybrid landfold $F$",
    }
    print("--- candidates ---")
    for name, xy in maps.items():
        rec, (gx, gy, fes) = lc.score_map(name, xy, wells, energy, attract, cf)
        scores.append(rec)
        print(
            f"{name:12s} sep_norm={rec['sep_norm']:.6f}  sep={rec['sep']:.6f}  "
            f"diam={rec['diam']:.6f}  sil={rec['silhouette']:.3f}  "
            f"wells={rec['n_wells']}  Fgm={rec['F_GM']}  Fico={rec['F_ico']}  "
            f"rim={rec['gm_on_rim']}  deeper={rec['gm_deeper']}"
        )
        save_pair(xy, gx, gy, fes, energy, DEST / f"{name}.png", titles[name])
        if name == "hybrid":
            save_pair(
                xy,
                gx,
                gy,
                fes,
                energy,
                OUT / "elja_occ_lj38_cand_hybrid.png",
                titles[name],
            )

    names = ["hybrid", "geo_asinh"]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.8), facecolor="white")
    for k, name in enumerate(names):
        rec, (gx, gy, fes) = lc.score_map(name, maps[name], wells, energy, attract, cf)
        lc.panel_row(
            axes[k, 0],
            axes[k, 1],
            maps[name],
            gx,
            gy,
            fes,
            energy,
            f"{name} sep={rec['sep_norm']:.3f}",
        )
    fig.tight_layout()
    fig.savefig(DEST / "cmp.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", DEST / "cmp.png")

    payload = {
        "n": n,
        "knn": KNN,
        "weights": {"l1": W_L1, "geo": W_GEO},
        "cost": "|Ei-Ej|+2 L1",
        "n_edges": len(edges),
        "n_components": comps,
        "geo_gm_ico": geo_gm_ico,
        "l1_gm_ico": float(l1[GM, ICO]),
        "sig_l1": sig_l1,
        "sig_geo": sig_geo,
        "asinh_norm": ASINH_NORM,
        "hyb_gm_ico": float(hyb[GM, ICO]),
        "fgeo_gm_ico": float(f_geo[GM, ICO]),
        "n_gm": n_gm,
        "n_ico": n_ico,
        "n_attractors": int(len(np.unique(attract))),
        "baselines": baselines,
        "scores": scores,
        "ev": {k: [float(x) for x in v] for k, v in evs.items()},
        "hybrid": "0.35 asinh(L1/med)+0.65 asinh(g/med), Torgerson",
        "geo_asinh": "landfold asinh(g/med)/(2 asinh 1), Torgerson",
    }
    (DEST / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", DEST / "scores.json")

    print("--- sep_norm ---")
    print(f"{'method':16s} {'sep_norm':>10s} {'sep':>10s} {'diam':>10s}")
    for rec in baselines:
        print(
            f"{rec['name']:16s} {rec['sep_norm']:10.6f} {rec['sep']:10.6f} "
            f"{rec['diam']:10.6f}"
        )
    for rec in scores:
        print(
            f"{rec['name']:16s} {rec['sep_norm']:10.6f} {rec['sep']:10.6f} "
            f"{rec['diam']:10.6f}"
        )


if __name__ == "__main__":
    main()
