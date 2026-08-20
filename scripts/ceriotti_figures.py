#!/usr/bin/env python3
"""JCTC-class FES figures for the public Ceriotti sketch-map sets.

Same construction as examples/cosmo-lj38/compose_fes.py and
`landfold fes --blur --floor --fmax`: binned KDE, connected body,
filled contours, F clipped to the paper scale. A raw 80x80 pcolormesh
of unsmoothed counts is not the panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

CMAP = LinearSegmentedColormap.from_list(
    "fes",
    [
        (0.00, "#e65014"),
        (0.18, "#7a1460"),
        (0.40, "#2c1a80"),
        (0.62, "#2f4ec4"),
        (0.82, "#6aa4e0"),
        (1.00, "#e8f2fb"),
    ],
)


def load_xyw(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    pts = np.atleast_2d(np.loadtxt(path))
    xy = pts[:, :2]
    if pts.shape[1] >= 3:
        w = pts[:, -1]
        if np.all(np.isfinite(w)) and np.all(w >= 0.0) and w.max() > 0.0:
            if w.std() > 0.0 or pts.shape[1] == 3:
                return xy, w
    return xy, None


def load_xy(path: Path) -> np.ndarray:
    return load_xyw(path)[0]


def procrustes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Orthogonal Procrustes: map A onto B (centered)."""
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return ac @ r + b.mean(axis=0)


def fes_hist(xy: np.ndarray, nx: int, ny: int, kt: float):
    """Return a normalized histogram free-energy surface for compatibility."""
    if not np.isfinite(kt) or kt <= 0.0:
        raise ValueError("kt must be finite and > 0")
    if nx <= 0 or ny <= 0:
        raise ValueError("histogram dimensions must be > 0")
    if xy.ndim != 2 or xy.shape[1] < 2 or xy.shape[0] == 0:
        raise ValueError("xy must contain at least one two-dimensional point")
    x = xy[:, 0]
    y = xy[:, 1]
    pad = 0.05
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    dx, dy = xmax - xmin, ymax - ymin
    if dx == 0.0:
        dx = max(abs(xmin), 1.0)
        xmin -= 0.5 * dx
        xmax += 0.5 * dx
    if dy == 0.0:
        dy = max(abs(ymin), 1.0)
        ymin -= 0.5 * dy
        ymax += 0.5 * dy
    xmin -= pad * dx
    xmax += pad * dx
    ymin -= pad * dy
    ymax += pad * dy
    histogram, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=[nx, ny],
        range=[[xmin, xmax], [ymin, ymax]],
    )
    density = histogram.T
    if density.max() > 0.0:
        density = density / density.max()
    fes = np.full_like(density, np.nan, dtype=float)
    occupied = density > 0.0
    fes[occupied] = -kt * np.log(density[occupied])
    if np.isfinite(fes).any():
        fes -= np.nanmin(fes)
    return xedges, yedges, fes


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    ny, nx = mask.shape
    reach = np.zeros_like(mask, dtype=bool)
    stack = []
    for i in range(ny):
        if not mask[i, 0]:
            stack.append((i, 0))
        if not mask[i, nx - 1]:
            stack.append((i, nx - 1))
    for j in range(nx):
        if not mask[0, j]:
            stack.append((0, j))
        if not mask[ny - 1, j]:
            stack.append((ny - 1, j))
    while stack:
        i, j = stack.pop()
        if i < 0 or j < 0 or i >= ny or j >= nx or reach[i, j] or mask[i, j]:
            continue
        reach[i, j] = True
        stack.extend(((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)))
    return mask | (~mask & ~reach)


def _keep_largest(mask: np.ndarray) -> np.ndarray:
    ny, nx = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = None
    best_n = 0
    for i0 in range(ny):
        for j0 in range(nx):
            if not mask[i0, j0] or seen[i0, j0]:
                continue
            stack = [(i0, j0)]
            cells = []
            while stack:
                i, j = stack.pop()
                if i < 0 or j < 0 or i >= ny or j >= nx:
                    continue
                if seen[i, j] or not mask[i, j]:
                    continue
                seen[i, j] = True
                cells.append((i, j))
                stack.extend(((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)))
            if len(cells) > best_n:
                best_n = len(cells)
                best = cells
    out = np.zeros_like(mask)
    if best:
        for i, j in best:
            out[i, j] = True
    return out


def _gauss1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _blur2d(z: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(int(np.ceil(3.0 * sigma)), 1)
    k = _gauss1d(sigma, radius)
    pad = np.pad(z, ((0, 0), (radius, radius)), mode="constant")
    tmp = np.empty_like(z)
    for i in range(z.shape[0]):
        tmp[i] = np.convolve(pad[i], k, mode="valid")
    pad = np.pad(tmp, ((radius, radius), (0, 0)), mode="constant")
    out = np.empty_like(z)
    for j in range(z.shape[1]):
        out[:, j] = np.convolve(pad[:, j], k, mode="valid")
    return out


def kde_fes(
    xy: np.ndarray,
    kt: float,
    fmax: float,
    weights: np.ndarray | None = None,
    ngrid: int = 240,
    sigma: float = 7.0,
    floor: float = 0.006,
    pad: float = 0.10,
    limits: tuple[float, float, float, float] | None = None,
):
    """Histogram plus Gaussian blur. F = -kT ln(rho/rhomax) on one body."""
    if not np.isfinite(kt) or kt <= 0.0:
        raise ValueError("kt must be finite and > 0")
    if not np.isfinite(fmax) or fmax <= 0.0:
        raise ValueError("fmax must be finite and > 0")
    x = xy[:, 0]
    y = xy[:, 1]
    if limits is None:
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        dx = max(xmax - xmin, 1e-6)
        dy = max(ymax - ymin, 1e-6)
        xmin -= pad * dx
        xmax += pad * dx
        ymin -= pad * dy
        ymax += pad * dy
    else:
        xmin, xmax, ymin, ymax = limits
    counts, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=ngrid,
        range=[[xmin, xmax], [ymin, ymax]],
        weights=weights,
    )
    rho = _blur2d(counts.T, sigma=sigma)
    gx = 0.5 * (xedges[:-1] + xedges[1:])
    gy = 0.5 * (yedges[:-1] + yedges[1:])
    rmax = float(rho.max())
    if rmax <= 0.0:
        raise SystemExit("empty projection")
    mask = rho > floor * rmax
    mask = _keep_largest(_fill_mask_holes(mask))
    fes = np.full_like(rho, np.nan)
    on = mask & (rho > 0.0)
    fes[on] = -kt * np.log(np.clip(rho[on] / rmax, 1e-12, 1.0))
    hole = mask & ~on
    fes[hole] = fmax
    fes = np.clip(fes, 0.0, fmax)
    return gx, gy, fes


def draw_fes(ax, gx, gy, fes, fmax, title, xlabel=True, ylabel=True):
    levels = np.linspace(0.0, fmax, 21)
    mesh = ax.contourf(gx, gy, fes, levels=levels, cmap=CMAP, extend="max")
    ax.contour(
        gx,
        gy,
        fes,
        levels=np.linspace(0.08 * fmax, 0.92 * fmax, 12),
        colors="#1a1a2e",
        linewidths=0.35,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11)
    if xlabel:
        ax.set_xlabel(r"$s_1$")
    if ylabel:
        ax.set_ylabel(r"$s_2$")
    return mesh


def save_fes(path: Path, gx, gy, fes, fmax, title, cbar_label):
    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=160, facecolor="white")
    mesh = draw_fes(ax, gx, gy, fes, fmax, title)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lj38-proj", type=Path)
    p.add_argument("--protein-lm-pub", type=Path)
    p.add_argument("--protein-lm-ours", type=Path)
    p.add_argument("--protein-smap-pub", type=Path)
    p.add_argument("--protein-smap-ours", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.lj38_proj and args.lj38_proj.is_file():
        xy = load_xy(args.lj38_proj)
        gx, gy, fes = kde_fes(xy, kt=0.168, fmax=2.0, ngrid=240, sigma=7.0)
        save_fes(
            args.out / "lj38_oos_fes.png",
            gx,
            gy,
            fes,
            2.0,
            r"LJ38 TSE out-of-sample  ($kT^*=0.168$)",
            r"$F/\varepsilon$",
        )

    if args.protein_smap_pub and args.protein_smap_pub.is_file():
        pub, wpub = load_xyw(args.protein_smap_pub)
        gx, gy, fes = kde_fes(
            pub, kt=1.0, fmax=4.0, weights=wpub, ngrid=280, sigma=4.0, floor=0.004
        )
        save_fes(
            args.out / "protein_published_fes.png",
            gx,
            gy,
            fes,
            4.0,
            rf"published sketch-map  $n={len(pub)}$",
            r"$F/kT$",
        )
        if args.protein_lm_pub and args.protein_lm_ours:
            pub_lm = load_xy(args.protein_lm_pub)
            ours_lm = load_xy(args.protein_lm_ours)
            aligned = procrustes(ours_lm, pub_lm)
            rmsd = float(np.sqrt(np.mean(np.sum((aligned - pub_lm) ** 2, axis=1))))
            fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=160, facecolor="white")
            mesh = draw_fes(ax, gx, gy, fes, 4.0, "beta-hairpin landmarks on published FES")
            ax.scatter(
                pub_lm[:, 0],
                pub_lm[:, 1],
                s=8,
                c="white",
                edgecolors="#1a1a2e",
                linewidths=0.35,
                label="published C++",
                zorder=3,
            )
            ax.scatter(
                aligned[:, 0],
                aligned[:, 1],
                s=6,
                c="#e65014",
                edgecolors="none",
                alpha=0.85,
                label=rf"ours (Procrustes RMSD {rmsd:.2f})",
                zorder=4,
            )
            ax.legend(fontsize=8, loc="lower left", frameon=True)
            cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(r"$F/kT$")
            fig.tight_layout()
            fig.savefig(args.out / "protein_landmarks.png", dpi=170, facecolor="white")
            plt.close(fig)
            (args.out / "protein_landmarks_rmsd.txt").write_text(f"{rmsd:.6f}\n")
    elif args.protein_lm_pub and args.protein_lm_ours:
        pub = load_xy(args.protein_lm_pub)
        ours = load_xy(args.protein_lm_ours)
        aligned = procrustes(ours, pub)
        rmsd = float(np.sqrt(np.mean(np.sum((aligned - pub) ** 2, axis=1))))
        gx, gy, fes = kde_fes(pub, kt=1.0, fmax=4.0, ngrid=200, sigma=3.0)
        fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=160, facecolor="white")
        mesh = draw_fes(ax, gx, gy, fes, 4.0, "beta-hairpin landmarks")
        ax.scatter(pub[:, 0], pub[:, 1], s=8, c="white", edgecolors="#1a1a2e", linewidths=0.35)
        ax.scatter(aligned[:, 0], aligned[:, 1], s=6, c="#e65014", alpha=0.85)
        ax.set_title(rf"beta-hairpin landmarks  (Procrustes RMSD {rmsd:.2f})")
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label(r"$F/kT$")
        fig.tight_layout()
        fig.savefig(args.out / "protein_landmarks.png", dpi=170, facecolor="white")
        plt.close(fig)
        (args.out / "protein_landmarks_rmsd.txt").write_text(f"{rmsd:.6f}\n")

    if args.protein_smap_ours and args.protein_smap_ours.is_file():
        ours, wours = load_xyw(args.protein_smap_ours)
        gx, gy, fes = kde_fes(
            ours, kt=1.0, fmax=4.0, weights=wours, ngrid=280, sigma=4.0, floor=0.004
        )
        save_fes(
            args.out / "protein_oos_fes.png",
            gx,
            gy,
            fes,
            4.0,
            rf"landfold project  $n={len(ours)}$",
            r"$F/kT$",
        )

    if (
        args.protein_smap_pub
        and args.protein_smap_ours
        and args.protein_smap_pub.is_file()
        and args.protein_smap_ours.is_file()
    ):
        pub, wpub = load_xyw(args.protein_smap_pub)
        ours, wours = load_xyw(args.protein_smap_ours)
        lo = (
            min(pub[:, 0].min(), ours[:, 0].min()),
            max(pub[:, 0].max(), ours[:, 0].max()),
            min(pub[:, 1].min(), ours[:, 1].min()),
            max(pub[:, 1].max(), ours[:, 1].max()),
        )
        dx = lo[1] - lo[0]
        dy = lo[3] - lo[2]
        limits = (
            lo[0] - 0.08 * dx,
            lo[1] + 0.08 * dx,
            lo[2] - 0.08 * dy,
            lo[3] + 0.08 * dy,
        )
        g1 = kde_fes(
            pub,
            kt=1.0,
            fmax=4.0,
            weights=wpub,
            ngrid=280,
            sigma=4.0,
            floor=0.004,
            limits=limits,
        )
        g2 = kde_fes(
            ours,
            kt=1.0,
            fmax=4.0,
            weights=wours,
            ngrid=280,
            sigma=4.0,
            floor=0.004,
            limits=limits,
        )
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), dpi=160, facecolor="white")
        m0 = draw_fes(axes[0], *g1, 4.0, "published smap")
        draw_fes(axes[1], *g2, 4.0, "landfold out-of-sample", ylabel=False)
        fig.colorbar(m0, ax=axes, fraction=0.025, pad=0.02).set_label(r"$F/kT$")
        fig.savefig(args.out / "protein_oos_vs_published.png", dpi=170, facecolor="white")
        plt.close(fig)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
