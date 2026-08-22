#!/usr/bin/env python3
"""Candidate commute-time map of the energy-weighted DECAF graph.

Height-Boltzmann commute (A = exp(-(max(E)+2 L1 - Emin)/kT)) isolates
the Wales GM. This candidate weights the kNN=12 packing graph by the
edge cost itself:

    cost_ij = |E_i - E_j| + 2 L1_ij
    A_ij    = exp(-cost_ij / kT)
    L       = D - A
    CT_ij   = vol (L+_ii + L+_jj - 2 L+_ij)

vol = 1^T D 1 = sum_i deg_i. L+ is the Moore-Penrose inverse: invert
every positive Laplacian eigenvalue and leave the kernel at 0.

Torgerson of sqrt(CT) recovers the spectral commute embedding
phi_{i,k} = sqrt(vol / lambda_k) u_{i,k} (k = 2, 3, ...), because

    L+_ii + L+_jj - 2 L+_ij = sum_{k: lambda_k > 0} (u_{ik}-u_{jk})^2 / lambda_k

so CT is already Euclidean in that space. Raw CT MDS is the control
that treats CT as a distance rather than a squared form.
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
DEST = Path("/tmp/occ-book/cand-commute")
KNN = 12
KT = lc.KT
GM = 0
ICO = 1


def knn_cost_edges(hist: np.ndarray, energy: np.ndarray, knn: int = KNN):
    """Symmetrized kNN in packing L1. Cost = |dE| + 2 L1."""
    n = len(energy)
    edges = []
    seen = set()
    l1 = np.zeros((n, n))
    for i in range(n):
        drow = np.abs(hist - hist[i]).sum(1)
        l1[i] = drow
        order = np.argsort(drow)
        picked = 0
        for j in order:
            j = int(j)
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b))
                cost = float(abs(energy[i] - energy[j]) + 2.0 * drow[j])
                edges.append((i, j, float(drow[j]), cost))
            picked += 1
            if picked >= knn:
                break
    return edges, l1


def adjacency_cost(n: int, edges) -> np.ndarray:
    a = np.zeros((n, n))
    for i, j, _d, cost in edges:
        w = float(np.exp(-cost / KT))
        a[i, j] = a[j, i] = max(w, 1e-300)
    return a


def n_components(a: np.ndarray) -> int:
    n = a.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = 0
    for s in range(n):
        if seen[s]:
            continue
        comps += 1
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(a[u] > 0):
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
    return comps


def commute_matrix(a: np.ndarray):
    """CT = vol (diag(L+) 1^T + 1 diag(L+)^T - 2 L+)."""
    n = a.shape[0]
    deg = a.sum(1)
    lap = np.diag(deg) - a
    w, v = np.linalg.eigh(lap)
    inv = np.zeros_like(w)
    pos = w > 1e-10 * max(float(w.max()), 1.0)
    inv[pos] = 1.0 / w[pos]
    lplus = (v * inv) @ v.T
    vol = float(deg.sum())
    diag = np.diag(lplus)
    ct = vol * (diag[:, None] + diag[None, :] - 2.0 * lplus)
    ct = np.clip(ct, 0.0, None)
    np.fill_diagonal(ct, 0.0)
    return ct, w, v, vol, int(pos.sum())


def spectral_embed(w: np.ndarray, v: np.ndarray, vol: float, dim: int = 2):
    """phi_{i,k} = sqrt(vol / lambda_k) u_{i,k} on the first dim positive modes."""
    order = np.argsort(w)
    w, v = w[order], v[:, order]
    take = []
    for k, lam in enumerate(w):
        if lam > 1e-10 * max(float(w.max()), 1.0):
            take.append(k)
        if len(take) == dim:
            break
    if len(take) < dim:
        raise RuntimeError("Laplacian has fewer than %d positive eigenvalues" % dim)
    lam = w[take]
    xy = v[:, take] * np.sqrt(vol / lam)
    return xy, lam


def procrustes(a: np.ndarray, b: np.ndarray) -> float:
    """Relative Frobenius residual after the best orthogonal map A R ~ B."""
    a0 = a - a.mean(0)
    b0 = b - b.mean(0)
    u, _s, vt = np.linalg.svd(a0.T @ b0)
    r = u @ vt
    aligned = a0 @ r
    num = float(np.linalg.norm(aligned - b0))
    den = float(np.linalg.norm(b0))
    return num / max(den, 1e-12)


def steepest_from_edges(n: int, energy: np.ndarray, edges):
    """Each family drains to its lowest-energy graph neighbour."""
    parent = np.arange(n)
    best = energy.copy()
    for i, j, _d, _c in edges:
        if energy[j] < best[i] - 1e-12:
            best[i] = energy[j]
            parent[i] = j
        if energy[i] < best[j] - 1e-12:
            best[j] = energy[i]
            parent[j] = i
    attract = np.empty(n, dtype=int)
    for i in range(n):
        x = i
        seen = set()
        while parent[x] != x and x not in seen:
            seen.add(x)
            x = int(parent[x])
        attract[i] = x
    return attract


def write_xy(path: Path, xy: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, xy, fmt="%.8e")


def save_pair(name: str, xy, gx, gy, fes, energy, dest: Path, title: str) -> None:
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


def main() -> None:
    hist, energy, wells = lc.load_book(HIST)
    n = len(energy)
    edges, _l1 = knn_cost_edges(hist, energy, knn=KNN)
    a = adjacency_cost(n, edges)
    comps = n_components(a)
    print(
        "n",
        n,
        "edges",
        len(edges),
        "components",
        comps,
        "E[GM]",
        energy[GM],
        "E[ico]",
        energy[ICO],
        "kT",
        KT,
        "kNN",
        KNN,
    )
    if comps > 1:
        print("WARN graph has", comps, "components; L+ kernel is larger than 1")

    ct, w, v, vol, npos = commute_matrix(a)
    ct_gm_ico = float(ct[GM, ICO])
    print("vol", vol, "n_pos_eigs", npos, "lambda2", float(np.sort(w)[1]))
    print("CT(GM,ico)", ct_gm_ico)

    xy_sqrt, ev_sqrt = lc.torgerson(np.sqrt(ct), 2)
    xy_raw, ev_raw = lc.torgerson(ct, 2)
    xy_spec, lam_spec = spectral_embed(w, v, vol, 2)
    proc = procrustes(xy_sqrt, xy_spec)
    print("Torgerson ev sqrt(CT)", ev_sqrt)
    print("Torgerson ev raw CT", ev_raw)
    print("spectral lambdas", lam_spec, "procrustes sqrtCT vs spectral", proc)

    attract = steepest_from_edges(n, energy, edges)
    print(
        "steepest n_GM",
        int((attract == attract[GM]).sum()),
        "n_ico",
        int((attract == attract[ICO]).sum()),
        "n_attract",
        len(np.unique(attract)),
    )

    DEST.mkdir(parents=True, exist_ok=True)
    write_xy(DEST / "sqrt_ct.xy", xy_sqrt)
    write_xy(DEST / "raw_ct.xy", xy_raw)
    write_xy(DEST / "spectral.xy", xy_spec)
    np.savetxt(DEST / "ct_row_gm.txt", ct[GM], fmt="%.8e")
    np.savetxt(DEST / "attract.lab", attract, fmt="%d")

    cf = lc._cf()
    scores = []
    for name, xy in (("sqrt_ct", xy_sqrt), ("raw_ct", xy_raw)):
        rec, (gx, gy, fes) = lc.score_map(name, xy, wells, energy, attract, cf)
        scores.append(rec)
        print(
            f"{name:8s} CT(GM,ico)={ct_gm_ico:.6e} sep_norm={rec['sep_norm']:.6f} "
            f"sil={rec['silhouette']:.6f} wells={rec['n_wells']} "
            f"Fgm={rec['F_GM']} Fico={rec['F_ico']} rim={rec['gm_on_rim']} "
            f"deeper={rec['gm_deeper']} defeat={rec['defeat']}"
        )
        save_pair(
            name,
            xy,
            gx,
            gy,
            fes,
            energy,
            DEST / f"{name}.png",
            "cand commute" if name == "sqrt_ct" else "raw CT MDS",
        )
        if name == "sqrt_ct":
            save_pair(
                name,
                xy,
                gx,
                gy,
                fes,
                energy,
                OUT / "elja_occ_lj38_cand_commute.png",
                "cand commute",
            )

    payload = {
        "n": n,
        "knn": KNN,
        "kt": KT,
        "cost": "|Ei-Ej|+2 L1",
        "affinity": "exp(-cost/kT)",
        "coords": "Torgerson of sqrt(CT); raw CT MDS control",
        "n_edges": len(edges),
        "n_components": comps,
        "vol": vol,
        "n_pos_eigs": npos,
        "lambda2": float(np.sort(w)[1]),
        "CT_GM_ico": ct_gm_ico,
        "procrustes_sqrt_vs_spectral": proc,
        "steepest_n_GM": int((attract == attract[GM]).sum()),
        "steepest_n_ico": int((attract == attract[ICO]).sum()),
        "scores": scores,
        "ev_sqrt": [float(x) for x in ev_sqrt],
        "ev_raw": [float(x) for x in ev_raw],
    }
    (DEST / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", DEST / "scores.json")
    print("---")
    print(f"CT(GM,ico)={ct_gm_ico:.6e}")
    for rec in scores:
        print(
            f"{rec['name']:8s} sep_norm={rec['sep_norm']:.6f} "
            f"silhouette={rec['silhouette']:.6f}"
        )


if __name__ == "__main__":
    main()
