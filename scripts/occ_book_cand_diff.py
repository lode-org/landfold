#!/usr/bin/env python3
"""Diffusion maps of the energy-weighted DECAF graph for LJ38.

Adjacency on undirected kNN=12 in DECAF L1:

    cost_ij = |E_i - E_j| + 2 L1_ij
    A_ij    = exp(-cost_ij / kT)

S = D^{-1/2} A D^{-1/2} shares the spectrum of the random-walk operator
P = D^{-1} A. Coordinates are Coifman-Lafon on the L_sym eigenvectors
(skip the constant mode):

    (mu_2^t u_2, mu_3^t u_3)

Do not multiply by D^{-1/2}: the GM is a pendant and that factor
sends it to infinity.

Sweep kT in {0.05, 0.168, 0.4} and t in {2, 8, 16}. Winner is the
largest sep_norm(GM, ico) * silhouette of steepest-descent basins
on the same kNN graph.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

LANDFOLD = Path(__file__).resolve().parents[1]
EX = LANDFOLD / "examples" / "cosmo-lj38"
HIST = Path("/tmp/occ-book/lj38_decaf_e.hist")
OUT = Path("/tmp/occ-book/cand-diff")
FIG = LANDFOLD / "docs" / "ceriotti-figs" / "elja_occ_lj38_cand_diff.png"

PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)
GM_E = -173.928427
ICO_E = -173.252378
KNN = 12
L1_COEF = 2.0
KT_BOOK = 0.168
KTS = (0.05, 0.168, 0.4)
TIMES = (2, 8, 16)
W_FLOOR = 1e-300
ALIVE = 1e-20


def load_book(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse `# family wells min_e h0...` occupancy book."""
    wells: list[float] = []
    emin: list[float] = []
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        wells.append(float(p[1]))
        emin.append(float(p[2]))
        rows.append([float(x) for x in p[3:]])
    return np.asarray(rows), np.asarray(emin), np.asarray(wells)


def l1_matrix(hist: np.ndarray) -> np.ndarray:
    """Pairwise L1 of CN histograms. n=400 is dense-cheap."""
    n = hist.shape[0]
    l1 = np.empty((n, n))
    for i in range(n):
        l1[i] = np.abs(hist - hist[i]).sum(1)
    np.fill_diagonal(l1, 0.0)
    return l1


def knn_pairs(l1: np.ndarray, knn: int = KNN) -> list[tuple[int, int, float]]:
    """Undirected kNN: edge if j is among i's knn or i among j's knn."""
    n = l1.shape[0]
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        order = np.argsort(l1[i])
        taken = 0
        for j in order:
            j = int(j)
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b))
                pairs.append((a, b, float(l1[a, b])))
            taken += 1
            if taken >= knn:
                break
    return pairs


def adjacency(
    n: int, energy: np.ndarray, pairs: list[tuple[int, int, float]], kt: float
) -> np.ndarray:
    """A_ij = exp(-(|Ei-Ej| + 2 L1_ij) / kT) on the kNN support."""
    a = np.zeros((n, n))
    for i, j, d in pairs:
        cost = abs(float(energy[i] - energy[j])) + L1_COEF * d
        w = float(np.exp(-cost / kt))
        a[i, j] = a[j, i] = max(w, W_FLOOR)
    return a


def n_alive(a: np.ndarray) -> int:
    return int(np.sum(np.triu(a, 1) > ALIVE))


def n_components(a: np.ndarray) -> int:
    """Union-find on edges with weight > ALIVE."""
    n = a.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ii, jj = np.where(np.triu(a, 1) > ALIVE)
    for i, j in zip(ii.tolist(), jj.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    return len({find(i) for i in range(n)})


def diffusion_spectrum(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """L_sym eigenvectors of S = D^{-1/2} A D^{-1/2}.

    mu_k = lambda_k(P). u_k are orthonormal. Do not form D^{-1/2} u:
    the GM pendant would dominate the diameter.
    """
    deg = np.clip(a.sum(1), W_FLOOR, None)
    dinvsqrt = 1.0 / np.sqrt(deg)
    s = (dinvsqrt[:, None] * a) * dinvsqrt[None, :]
    w, u = np.linalg.eigh(s)
    idx = np.argsort(w)[::-1]
    return w[idx], u[:, idx]


def embed(u: np.ndarray, mu: np.ndarray, t: int) -> np.ndarray:
    """(mu_2^t u_2, mu_3^t u_3)."""
    return u[:, 1:3] * np.power(mu[1:3], t)


def steepest_basins(
    n: int, energy: np.ndarray, pairs: list[tuple[int, int, float]]
) -> np.ndarray:
    """Each packing drains to its lowest-energy kNN neighbour."""
    parent = np.arange(n)
    best = energy.copy()
    for i, j, _d in pairs:
        if energy[j] < best[i] - 1e-12:
            best[i] = energy[j]
            parent[i] = j
        if energy[i] < best[j] - 1e-12:
            best[j] = energy[i]
            parent[j] = i
    attract = np.empty(n, dtype=int)
    for i in range(n):
        x = i
        seen: set[int] = set()
        while parent[x] != x and x not in seen:
            seen.add(x)
            x = int(parent[x])
        attract[i] = x
    return attract


def sep_norm(xy: np.ndarray, i: int = 0, j: int = 1) -> tuple[float, float, float]:
    sep = float(np.linalg.norm(xy[i] - xy[j]))
    diam = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    return sep / max(diam, 1e-12), sep, diam


def silhouette(xy: np.ndarray, labels: np.ndarray, i: int = 0, j: int = 1) -> float:
    """Centroid-ratio silhouette of the two SD basins holding GM and ico."""
    lab_a, lab_b = labels[i], labels[j]
    if lab_a == lab_b:
        return 0.0
    a_mask = labels == lab_a
    b_mask = labels == lab_b
    if a_mask.sum() <= 1 or b_mask.sum() <= 1:
        return 0.0
    da = np.linalg.norm(xy[a_mask] - xy[a_mask].mean(0), axis=1).mean()
    db = np.linalg.norm(xy[b_mask] - xy[b_mask].mean(0), axis=1).mean()
    between = float(np.linalg.norm(xy[a_mask].mean(0) - xy[b_mask].mean(0)))
    return float(between / max(da + db, 1e-12))


def orient(xy: np.ndarray) -> np.ndarray:
    """GM left of ico; second axis increases toward ico."""
    out = xy.copy()
    if out[0, 0] > out[1, 0]:
        out[:, 0] *= -1.0
    if out[1, 1] < out[0, 1]:
        out[:, 1] *= -1.0
    return out


def tag(kt: float, t: int) -> str:
    return f"kt{kt:g}_t{t:02d}"


def _cf():
    spec = importlib.util.spec_from_file_location("compose_fes", EX / "compose_fes.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def kde_fes(xy, weights, cf, ngrid=160, sigma=4.0, kt: float = KT_BOOK):
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xmin -= 0.12 * dx
    xmax += 0.12 * dx
    ymin -= 0.12 * dy
    ymax += 0.12 * dy
    counts, xe, ye = np.histogram2d(
        x, y, bins=ngrid, range=[[xmin, xmax], [ymin, ymax]], weights=weights
    )
    rho = cf._blur2d(counts.T, sigma=sigma)
    gx = 0.5 * (xe[:-1] + xe[1:])
    gy = 0.5 * (ye[:-1] + ye[1:])
    rmax = float(rho.max()) if float(rho.max()) > 0 else 1.0
    mask = cf._fill_mask_holes(rho > 0.004 * rmax)
    fes = np.full_like(rho, np.nan)
    on = mask & (rho > 0)
    fes[on] = -kt * np.log(np.clip(rho[on] / rmax, 1e-12, 1))
    return gx, gy, np.clip(fes, 0, 2)


def save_pair(name: str, xy, gx, gy, fes, energy, dest: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), facecolor="white")
    ax0, ax1 = axes
    mesh = ax0.contourf(
        gx, gy, fes, levels=np.linspace(0, 2, 21), cmap=PES, extend="max"
    )
    ax0.contour(
        gx,
        gy,
        np.where(np.isfinite(fes), fes, np.nan),
        levels=np.linspace(0.15, 1.85, 10),
        colors="#1a1a2e",
        linewidths=0.35,
    )
    ax0.scatter(
        xy[0, 0],
        xy[0, 1],
        s=120,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=6,
        label=rf"GM ${GM_E:.3f}$",
    )
    ax0.scatter(
        xy[1, 0],
        xy[1, 1],
        s=70,
        marker="D",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=6,
        label=rf"ico ${ICO_E:.3f}$",
    )
    ax0.legend(fontsize=7, frameon=True, fancybox=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title(name + "  occupancy")
    for sp in ax0.spines.values():
        sp.set_visible(False)
    rel = np.clip(energy - energy.min(), 0, 8)
    sc = ax1.scatter(xy[:, 0], xy[:, 1], c=rel, s=18, cmap=PES, vmin=0, vmax=6, linewidths=0)
    ax1.scatter(
        xy[0, 0],
        xy[0, 1],
        s=120,
        marker="*",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=6,
    )
    ax1.scatter(
        xy[1, 0],
        xy[1, 1],
        s=70,
        marker="D",
        c="k",
        edgecolors="white",
        linewidths=0.6,
        zorder=6,
    )
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(name + r"  $E-E_{\mathrm{GM}}$")
    for sp in ax1.spines.values():
        sp.set_visible(False)
    fig.colorbar(mesh, ax=ax0, fraction=0.046, pad=0.03).set_label(r"$F/\varepsilon$")
    fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.03).set_label(
        r"$E-E_{\mathrm{GM}}/\varepsilon$"
    )
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", dest)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hist, energy, wells = load_book(HIST)
    n = len(energy)
    print(
        f"n {n} E[0]={energy[0]:.9f} E[1]={energy[1]:.9f} "
        f"wells0={wells[0]:.0f} wells1={wells[1]:.0f}"
    )
    assert abs(energy[0] - GM_E) < 1e-4, energy[0]
    assert abs(energy[1] - ICO_E) < 1e-4, energy[1]

    l1 = l1_matrix(hist)
    pairs = knn_pairs(l1, KNN)
    print(f"undirected kNN={KNN} edges {len(pairs)} L1(GM,ico)={l1[0, 1]:.6f}")
    attract = steepest_basins(n, energy, pairs)
    n_attr = int(len(np.unique(attract)))
    n_gm = int((attract == attract[0]).sum())
    n_ico = int((attract == attract[1]).sum())
    same = bool(attract[0] == attract[1])
    print(
        f"SD attractors {n_attr} n_GM={n_gm} n_ico={n_ico} "
        f"same_basin={same} attract[0]={int(attract[0])} attract[1]={int(attract[1])}"
    )
    np.savetxt(OUT / "attract.lab", attract, fmt="%d")

    cf = _cf()
    recs = []
    maps: dict[str, np.ndarray] = {}
    surfaces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for kt in KTS:
        a = adjacency(n, energy, pairs, kt)
        mu, u = diffusion_spectrum(a)
        alive = n_alive(a)
        ncomp = n_components(a)
        print(
            f"kT={kt:g} alive={alive} ncomp={ncomp} "
            f"deg_gm={float(a[0].sum()):.3e} deg_ico={float(a[1].sum()):.3e} "
            f"mu={', '.join(f'{x:.6f}' for x in mu[:6])}"
        )
        for t in TIMES:
            name = tag(kt, t)
            xy = orient(embed(u, mu, t))
            maps[name] = xy
            np.savetxt(OUT / f"{name}.xy", xy, fmt="%.8e")
            sn, sep, diam = sep_norm(xy)
            sil = silhouette(xy, attract)
            rec = {
                "name": name,
                "kt": kt,
                "t": t,
                "sep_norm": sn,
                "sep": sep,
                "diam": diam,
                "silhouette": sil,
                "score": float(sn * sil),
                "mu": [float(x) for x in mu[:6]],
                "n_alive": alive,
                "n_components": ncomp,
                "deg_gm": float(a[0].sum()),
                "deg_ico": float(a[1].sum()),
                "A_gm_ico": float(a[0, 1]),
                "gm": [float(xy[0, 0]), float(xy[0, 1])],
                "ico": [float(xy[1, 0]), float(xy[1, 1])],
            }
            recs.append(rec)
            print(
                f"  {name:14s} sep_norm={sn:.6f} sil={sil:.6f} "
                f"score={rec['score']:.6f}"
            )

    recs.sort(key=lambda r: (r["score"], r["sep_norm"], r["silhouette"]), reverse=True)
    best = recs[0]
    print(
        f"winner {best['name']} score={best['score']:.6f} "
        f"sep_norm={best['sep_norm']:.6f} sil={best['silhouette']:.6f}"
    )
    np.savetxt(OUT / "best.xy", maps[best["name"]], fmt="%.8e")

    for rec in recs:
        name = rec["name"]
        xy = maps[name]
        gx, gy, fes = kde_fes(xy, wells, cf)
        surfaces[name] = (gx, gy, fes)
        save_pair(
            rf"diff $kT={rec['kt']:g}$, $t={rec['t']}$",
            xy,
            gx,
            gy,
            fes,
            energy,
            OUT / f"{name}.png",
        )

    fig, axes = plt.subplots(len(KTS), len(TIMES), figsize=(12.6, 11.2), facecolor="white")
    for i, kt in enumerate(KTS):
        for j, t in enumerate(TIMES):
            name = tag(kt, t)
            ax = axes[i, j]
            xy = maps[name]
            rec = next(r for r in recs if r["name"] == name)
            gx, gy, fes = surfaces[name]
            ax.contourf(gx, gy, fes, levels=np.linspace(0, 2, 21), cmap=PES, extend="max")
            ax.scatter(
                xy[0, 0],
                xy[0, 1],
                s=90,
                marker="*",
                c="k",
                edgecolors="white",
                linewidths=0.5,
                zorder=6,
            )
            ax.scatter(
                xy[1, 0],
                xy[1, 1],
                s=55,
                marker="D",
                c="k",
                edgecolors="white",
                linewidths=0.5,
                zorder=6,
            )
            mark = "BEST" if name == best["name"] else ""
            ax.set_title(
                rf"$kT={kt:g}$  $t={t}$  "
                rf"sep={rec['sep_norm']:.2f} sil={rec['silhouette']:.2f} {mark}",
                fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.tight_layout()
    dest = OUT / "grid.png"
    fig.savefig(dest, dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", dest)

    xy = maps[best["name"]]
    gx, gy, fes = surfaces[best["name"]]
    save_pair(
        rf"diff $kT={best['kt']:g}$, $t={best['t']}$",
        xy,
        gx,
        gy,
        fes,
        energy,
        FIG,
    )
    save_pair(
        rf"diff $kT={best['kt']:g}$, $t={best['t']}$",
        xy,
        gx,
        gy,
        fes,
        energy,
        OUT / "elja_occ_lj38_cand_diff.png",
    )

    payload = {
        "n": n,
        "knn": KNN,
        "l1_coef": L1_COEF,
        "l1_gm_ico": float(l1[0, 1]),
        "n_edges": len(pairs),
        "sd_attractors": n_attr,
        "n_gm_basin": n_gm,
        "n_ico_basin": n_ico,
        "same_sd_basin": same,
        "kts": list(KTS),
        "times": list(TIMES),
        "best": best["name"],
        "winner": best,
        "scores": recs,
        "formula": "A_ij = exp(-(|Ei-Ej| + 2 L1_ij) / kT) on undirected kNN=12",
        "embedding": "L_sym eigenvectors u_2, u_3 of S, coords (mu_k^t u_k)",
        "pick": "argmax sep_norm * silhouette",
    }
    (OUT / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", OUT / "scores.json")
    print("---")
    print(f"sep_norm     {best['sep_norm']:.8f}")
    print(f"silhouette   {best['silhouette']:.8f}")
    print(f"score        {best['score']:.8f}")
    print(f"chosen       {best['name']}")


if __name__ == "__main__":
    main()
