#!/usr/bin/env python3
"""SHEAP analogue for the Elja LJ98 inherent structures.

Same layout as occ_book_sheap.py: graph support is kNN=12 in
sorted-pair-distance L2, energy is the radial force so the GM sits
at a funnel tip. GM is argmin E. The second motif is the lowest-E
structure whose pair list is not a GM copy.

Occupancy leftover-well invert is not used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import occ_book_sheap as sh

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ceriotti-figs"
MINFILE = Path("/tmp/occ-book/lj98_0013.min")
ENERGY_FILE = Path("/tmp/occ-book/lj98.energy")
DEST = Path("/tmp/occ-book/cand-sheap-lj98")
N_ATOMS = 98
KNN = 12
DUP_EPS = 5e-2
COPY_EPS = 1e-6


def spd_matrix(frames) -> np.ndarray:
    m = N_ATOMS * (N_ATOMS - 1) // 2
    out = np.empty((len(frames), m), dtype=np.float64)
    for i, p in enumerate(frames):
        out[i] = sh.pair_pdist(p)
    return out


def find_second_motif(energy: np.ndarray, hd: np.ndarray):
    gm = int(np.argmin(energy))
    d_gm = np.linalg.norm(hd - hd[gm], axis=1)
    copies = d_gm < COPY_EPS
    rest = np.where(~copies)[0]
    if rest.size == 0:
        raise SystemExit("no second motif: every pair list is a GM copy")
    motif = int(rest[np.argmin(energy[rest])])
    return gm, motif, d_gm, copies


def funnel_xy(e_u, gm_u, ico_u, gm_b, ico_b, q, u, alpha: float, sep: float):
    rel_g = np.clip(e_u - e_u[gm_u], 0.0, None)
    rel_i = np.clip(e_u - e_u[ico_u], 0.0, None)
    r_g = np.power(rel_g, alpha)
    r_i = np.power(rel_i, alpha)
    xy = np.zeros((len(e_u), 2))
    phi_g = 0.58 * np.pi + 0.28 * u
    xy[gm_b, 0] = r_g[gm_b] * np.cos(phi_g[gm_b])
    xy[gm_b, 1] = r_g[gm_b] * np.sin(phi_g[gm_b])
    xy[gm_u] = 0.0
    phi_i = 0.42 * np.pi + 0.28 * u
    xy[ico_b, 0] = sep - r_i[ico_b] * np.cos(phi_i[ico_b])
    xy[ico_b, 1] = r_i[ico_b] * np.sin(phi_i[ico_b])
    xy[ico_u] = (sep, 0.0)
    rest = ~(gm_b | ico_b)
    xy[rest, 0] = sep * np.clip(q[rest], 0.05, 0.95)
    xy[rest, 1] = 0.55 + 0.85 * r_g[rest]
    xy[rest, 0] += 0.18 * u[rest] * r_g[rest]
    return xy


def main() -> None:
    t_all = time.time()
    DEST.mkdir(parents=True, exist_ok=True)
    sh.N_ATOMS = N_ATOMS
    sh.DEST = DEST
    energy, frames = sh.load_min(MINFILE, n_atoms=N_ATOMS)
    n = len(energy)
    book_e = np.loadtxt(ENERGY_FILE)
    if book_e.shape[0] != n:
        raise SystemExit(f"energy rows {book_e.shape[0]} != min rows {n}")
    if np.max(np.abs(book_e - energy)) > 1e-6:
        raise SystemExit("lj98.energy does not match first column of lj98_0013.min")

    print("compute sorted-pair-distance L2")
    t0 = time.time()
    hd = spd_matrix(frames)
    dist = sh.pairwise_l2(hd)
    print(
        "spd",
        hd.shape,
        "D med",
        float(np.median(dist[dist > 0])),
        f"dt {time.time() - t0:.2f}s",
    )
    np.save(DEST / "dpair.npy", dist)
    np.save(DEST / "hd.npy", hd)

    gm, motif, d_gm, copies = find_second_motif(energy, hd)
    gm_e = float(energy[gm])
    motif_e = float(energy[motif])
    print(
        "GM idx",
        gm,
        "E",
        gm_e,
        "identical-HD copies",
        int(copies.sum()),
    )
    print(
        "second motif idx",
        motif,
        "E",
        motif_e,
        "dE",
        motif_e - gm_e,
        "HD L2 to GM",
        float(d_gm[motif]),
    )
    sh.GM_IDX = gm
    sh.ICO_IDX = motif
    sh.GM_E = gm_e
    sh.ICO_E = motif_e

    print(
        "D(GM,2nd)",
        float(dist[gm, motif]),
        "D(0,1)",
        float(dist[0, 1]) if n > 1 else None,
    )

    reps, assign = sh.unique_reps(dist, energy, eps=DUP_EPS)
    print("unique structures", len(reps), "of", n, "dup_eps", DUP_EPS)
    d_u = dist[np.ix_(reps, reps)]
    e_u = energy[reps]
    gm_u = int(np.where(reps == assign[gm])[0][0])
    ico_u = int(np.where(reps == assign[motif])[0][0])
    if gm_u != int(np.argmin(e_u)):
        print("WARN unique GM slot", gm_u, "argmin", int(np.argmin(e_u)))
    print(
        "unique GM/2nd",
        gm_u,
        ico_u,
        "E",
        float(e_u[gm_u]),
        float(e_u[ico_u]),
        "D",
        float(d_u[gm_u, ico_u]),
    )
    labels_u = sh.structure_labels(d_u, gm_u, ico_u)
    print(
        "structure labels n_GM-like",
        int((labels_u == 0).sum()),
        "n_2nd-like",
        int((labels_u == 1).sum()),
    )

    neigh, ndist = sh.knn_from_dist(d_u, KNN, skip_eps=DUP_EPS)
    adj, _seen = sh.symmetrize_knn(neigh, ndist, d_u, KNN)
    comps = sh.n_components(len(reps), adj)
    attract = sh.steepest_basins(e_u, adj)
    n_attr = int(len(np.unique(attract)))
    rest_vals = ndist[np.isfinite(ndist)]
    rest_med = float(np.median(rest_vals)) if rest_vals.size else 1.0
    print(
        f"kNN={KNN} edges~{sum(len(a) for a in adj)//2} comps={comps} "
        f"steepest_attr={n_attr} (structure graph diagnostic; not the figure) "
        f"n_GM={int((attract == attract[gm_u]).sum())} "
        f"n_2nd={int((attract == attract[ico_u]).sum())}"
    )

    q = sh.committor(adj, [gm_u], [ico_u])
    q[gm_u] = 0.0
    q[ico_u] = 1.0
    gm_b = attract == attract[gm_u]
    ico_b = attract == attract[ico_u]
    if attract[gm_u] == attract[ico_u]:
        gm_b = labels_u == 0
        ico_b = labels_u == 1
        print("GM and 2nd share a steepest basin; split by structure Voronoi")
    print(
        "q percentiles",
        [float(x) for x in np.percentile(q, [0, 5, 25, 50, 75, 95, 100])],
        "q GM-basin",
        float(q[gm_b].mean()) if gm_b.any() else None,
        "q 2nd-basin",
        float(q[ico_b].mean()) if ico_b.any() else None,
    )
    labels_funnel = ico_b.astype(int)
    labels_funnel[gm_b] = 0
    xy_lap = sh.laplacian_xy(len(reps), adj)
    u = xy_lap[:, 1].copy()
    u = (u - np.median(u)) / (np.std(u) + 1e-12)
    u = np.clip(u, -1.5, 1.5) / 1.5

    candidates = []
    best = None
    for alpha, sep in ((0.60, 3.0), (0.70, 3.4)):
        xy0 = funnel_xy(e_u, gm_u, ico_u, gm_b, ico_b, q, u, alpha, sep)
        xy = sh.spring_embed(
            adj,
            e_u,
            xy0,
            gm_u,
            rest_med,
            k_spring=0.35,
            k_rad=0.0,
            alpha=alpha,
            n_iter=80,
        )
        xy[gm_u] = 0.0
        xy[ico_u, 0] = float(xy0[ico_u, 0])
        xy[ico_u, 1] = 0.0
        rec = sh.score_layout(
            f"dual-funnel k={KNN} a={alpha} L={sep}",
            xy,
            e_u,
            gm_u,
            ico_u,
            labels_funnel,
            attract,
        )
        rec["knn"] = KNN
        rec["alpha"] = alpha
        rec["kind"] = "spring"
        rec["seven_fail"] = False
        rec["sep_tips"] = sep
        candidates.append((rec, xy, attract, KNN, alpha))
        print(
            f"  dual     score={rec['score']:.3f} sil={rec['silhouette']:.3f} "
            f"ang={rec['ang_split']:.3f} r_ico={rec['r_ico']:.3f} "
            f"center={rec['gm_center']} ico_off={rec['ico_off']}"
        )
        if best is None or rec["score"] > best[0]["score"]:
            best = (rec, xy, attract, KNN, alpha)

    print("picked", best[0]["name"], "score", best[0]["score"])
    rec, xy_u, attract, knn, alpha = best
    xy_all = sh.expand_xy(xy_u, reps, assign)
    xy_all = sh.place_gm_ico(xy_all, gm, motif)

    print("save coordinates")
    np.savetxt(DEST / "sheap.xy", xy_all, fmt="%.8e")
    np.savetxt(DEST / "sheap_unique.xy", xy_u, fmt="%.8e")
    np.savetxt(DEST / "reps.txt", reps, fmt="%d")
    labels_all = sh.structure_labels(dist, gm, motif)
    np.savetxt(DEST / "struct_label.txt", labels_all, fmt="%d")
    (DEST / "motifs.json").write_text(
        json.dumps(
            {
                "gm_idx": gm,
                "gm_E": gm_e,
                "second_idx": motif,
                "second_E": motif_e,
                "dE": motif_e - gm_e,
                "hd_l2_gm_second": float(d_gm[motif]),
                "n_gm_copies": int(copies.sum()),
            },
            indent=2,
        )
        + "\n"
    )

    title = rf"LJ98 SHEAP  kNN$={knn}$  $r\sim(\Delta E)^{{{alpha}}}$"
    mins = sh.save_figure(
        xy_all,
        energy,
        gm,
        motif,
        DEST / "elja_occ_lj98_sheap.png",
        title,
        also=OUT / "elja_occ_lj98_sheap.png",
    )

    payload = {
        "n": n,
        "n_atoms": N_ATOMS,
        "n_unique": int(len(reps)),
        "dup_eps": DUP_EPS,
        "descriptor": "sorted-pair-distance L2",
        "energy_role": "r = (E - E_home)^alpha from the GM tip and the second motif tip",
        "occupancy_invert": False,
        "gm_at_funnel_tip": True,
        "second_funnel": True,
        "gm_idx": gm,
        "second_idx": motif,
        "E_GM": gm_e,
        "E_second": motif_e,
        "D_gm_second": float(dist[gm, motif]),
        "winner": rec,
        "n_energy_wells": len(mins),
        "candidates": [c[0] for c in candidates],
        "seconds": time.time() - t_all,
    }
    (DEST / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["winner"], indent=2))
    print(
        "second motif idx",
        motif,
        "E",
        motif_e,
        "GM at origin",
        rec["gm_center"],
        "dt",
        f"{payload['seconds']:.1f}s",
    )


if __name__ == "__main__":
    main()
