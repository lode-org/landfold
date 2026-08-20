#!/usr/bin/env python3
"""Figures for the public Ceriotti sketch-map sets."""

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


def load_xy(path: Path) -> np.ndarray:
    pts = np.loadtxt(path)
    return np.atleast_2d(pts)[:, :2]


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
    h, xe, ye = np.histogram2d(x, y, bins=[nx, ny], range=[[xmin, xmax], [ymin, ymax]])
    dens = h.T
    dens = dens / dens.max() if dens.max() > 0 else dens
    fes = np.full_like(dens, np.nan, dtype=float)
    on = dens > 0
    fes[on] = -kt * np.log(dens[on])
    if np.isfinite(fes).any():
        fes -= np.nanmin(fes)
    return xe, ye, fes


def panel_fes(ax, xy, title, kt, vmax=3.0):
    xe, ye, fes = fes_hist(xy, 80, 80, kt)
    m = np.ma.masked_invalid(fes)
    im = ax.pcolormesh(xe, ye, m, cmap=CMAP, vmin=0.0, vmax=vmax, shading="auto")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("s1")
    ax.set_ylabel("s2")
    return im


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
        fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=140)
        im = panel_fes(ax, xy, "LJ38 TSE out-of-sample FES  (kT* = 0.168)", 0.168, vmax=2.0)
        fig.colorbar(im, ax=ax, label="F")
        fig.tight_layout()
        fig.savefig(args.out / "lj38_oos_fes.png")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=140)
        ax.scatter(xy[:, 0], xy[:, 1], s=4, c="#2c1a80", alpha=0.35, linewidths=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"LJ38 TSE projection  n={len(xy)}")
        ax.set_xlabel("s1")
        ax.set_ylabel("s2")
        fig.tight_layout()
        fig.savefig(args.out / "lj38_oos_scatter.png")
        plt.close(fig)

    if args.protein_lm_pub and args.protein_lm_ours:
        pub = load_xy(args.protein_lm_pub)
        ours = load_xy(args.protein_lm_ours)
        aligned = procrustes(ours, pub)
        rmsd = float(np.sqrt(np.mean(np.sum((aligned - pub) ** 2, axis=1))))
        fig, ax = plt.subplots(figsize=(5.4, 4.4), dpi=140)
        ax.scatter(pub[:, 0], pub[:, 1], s=10, c="#6aa4e0", label="published C++", alpha=0.85)
        ax.scatter(
            aligned[:, 0],
            aligned[:, 1],
            s=8,
            c="#e65014",
            label=f"ours (Procrustes RMSD {rmsd:.2f})",
            alpha=0.75,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=8)
        ax.set_title("beta-hairpin landmarks")
        ax.set_xlabel("s1")
        ax.set_ylabel("s2")
        fig.tight_layout()
        fig.savefig(args.out / "protein_landmarks.png")
        plt.close(fig)
        (args.out / "protein_landmarks_rmsd.txt").write_text(f"{rmsd:.6f}\n")

    if args.protein_smap_pub and args.protein_smap_pub.is_file():
        pub = load_xy(args.protein_smap_pub)
        fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=140)
        im = panel_fes(ax, pub, f"published sketch-map  n={len(pub)}", 1.0, vmax=4.0)
        fig.colorbar(im, ax=ax, label="F")
        fig.tight_layout()
        fig.savefig(args.out / "protein_published_fes.png")
        plt.close(fig)

    if args.protein_smap_ours and args.protein_smap_ours.is_file():
        ours = load_xy(args.protein_smap_ours)
        fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=140)
        im = panel_fes(ax, ours, f"landfold project  n={len(ours)}", 1.0, vmax=4.0)
        fig.colorbar(im, ax=ax, label="F")
        fig.tight_layout()
        fig.savefig(args.out / "protein_oos_fes.png")
        plt.close(fig)

    if (
        args.protein_smap_pub
        and args.protein_smap_ours
        and args.protein_smap_pub.is_file()
        and args.protein_smap_ours.is_file()
    ):
        pub = load_xy(args.protein_smap_pub)
        ours = load_xy(args.protein_smap_ours)
        n = min(len(pub), len(ours))
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), dpi=140)
        panel_fes(axes[0], pub[:n], "published smap", 1.0, vmax=4.0)
        panel_fes(axes[1], ours[:n], "landfold out-of-sample", 1.0, vmax=4.0)
        fig.tight_layout()
        fig.savefig(args.out / "protein_oos_vs_published.png")
        plt.close(fig)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
