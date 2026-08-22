#!/usr/bin/env python3
"""asinh chi of the 24-D SOAP fingerprint of 4042 LJ38 minima.

The mean SOAP (n_r=16, n_a=8) is written as a .hist landfold can
embed. sigma is the median pairwise L2 of those fingerprints.
landfold embed --fun-hd asinh,sigma folds a landmark block; project
places all 4042. The filled field is E-E_GM IDW, not occupancy invert.

If the remote landfold is unreachable, classical MDS of asinh(D/sigma)
of the same SOAP L2 is the fallback plane.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import occ_book_dpair as dp
import occ_book_idw_fill as idw

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "docs" / "ceriotti-figs"
BOOK = Path(os.environ.get("SOAP_CHI_BOOK", "/tmp/occ-book"))
FP_NPY = Path(os.environ.get("SOAP_CHI_FP", str(BOOK / "cand-soap" / "fp_mean_r16a8.npy")))
FP_TXT = BOOK / "cand-soap" / "fp_mean_r16a8.txt"
ENERGY = BOOK / "lj38.energy"
DEST = Path(os.environ.get("SOAP_CHI_DEST", str(BOOK / "cand-soap-chi")))
TERRA = os.environ.get("SOAP_CHI_TERRA", "rg.terra")
TERRA_WORK = os.environ.get("SOAP_CHI_TERRA_WORK", "/tmp/occ-book/cand-soap-chi")
N_LAND = int(os.environ.get("SOAP_CHI_NLAND", "200"))
GM_E = dp.GM_E
ICO_E = dp.ICO_E
GM_IDX = 0
ICO_IDX = 40
HD = 24
ASINH_NORM = 2.0 * float(np.arcsinh(1.0))
PES = LinearSegmentedColormap.from_list(
    "ruhi_pes",
    ["#004D40", "#1E88E5", "#D81B60", "#FF655D", "#F1DB4B"],
    N=256,
)
VMAX = 5.0


def load_fp() -> np.ndarray:
    if FP_NPY.is_file():
        fp = np.load(FP_NPY)
    elif FP_TXT.is_file():
        fp = np.loadtxt(FP_TXT)
    else:
        raise SystemExit(f"missing SOAP fingerprint {FP_NPY} or {FP_TXT}")
    if fp.ndim != 2 or fp.shape[1] != HD:
        raise SystemExit(f"expected (*, {HD}) SOAP, got {fp.shape}")
    if not np.isfinite(fp).all():
        raise SystemExit("SOAP fingerprint has non-finite values")
    return np.asarray(fp, dtype=np.float64)


def write_hist(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(f"# SOAP mean n_r=16 n_a=8 rows={arr.shape[0]} D={arr.shape[1]}\n")
        np.savetxt(fh, arr, fmt="%.8e")
    print("wrote", path, arr.shape)


def write_table(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr, fmt="%.8e")
    print("wrote", path, arr.shape)


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


def stress_asinh(xy: np.ndarray, dist: np.ndarray, sigma: float) -> float:
    """Kruskal-style chi^2 of landfold asinh F(D) vs F(d)."""
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, 0.0)
    fd = np.arcsinh(dist / sigma) / ASINH_NORM
    fd_ld = np.arcsinh(d / sigma) / ASINH_NORM
    np.fill_diagonal(fd, 0.0)
    np.fill_diagonal(fd_ld, 0.0)
    num = float(((fd - fd_ld) ** 2).sum())
    den = float((fd**2).sum())
    return num / den if den > 0 else float("nan")


def parse_stress(err_text: str) -> float | None:
    for line in err_text.splitlines():
        if line.lower().startswith("# stress"):
            try:
                return float(line.split()[-1])
            except ValueError:
                return None
    return None


def numpy_asinh_mds(dist: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    ash = np.arcsinh(dist / sigma) / ASINH_NORM
    np.fill_diagonal(ash, 0.0)
    return dp.torgerson(ash, 2)


def orient(xy: np.ndarray, gm: int, ico: int) -> np.ndarray:
    out = xy.copy()
    if out[gm, 0] > out[ico, 0]:
        out[:, 0] *= -1.0
    if out[ico, 1] < out[gm, 1]:
        out[:, 1] *= -1.0
    return out


def ring_barrier(xy, gx, gy, field, idx: int, r_in: float, r_out: float) -> float:
    c = xy[idx]
    xx, yy = np.meshgrid(gx, gy)
    r = np.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2)
    ring = field[(r >= r_in) & (r <= r_out)]
    ring = ring[np.isfinite(ring)]
    if ring.size == 0:
        return 0.0
    return float(np.mean(ring) - idw.sample(gx, gy, field, c))


def try_terra(work: Path, nland: int) -> dict | None:
    local_proj = work / "soap.proj"
    local_err = work / "soap.err"
    if local_proj.is_file() and local_proj.stat().st_size > 10:
        xy = np.loadtxt(local_proj)
        if xy.ndim == 2 and xy.shape[0] == 4042:
            err = local_err.read_text() if local_err.is_file() else ""
            print("reusing", local_proj, xy.shape)
            return {"xy": xy, "err": err, "stress": parse_stress(err), "engine": "landfold"}
    ping = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", TERRA, "true"],
        capture_output=True,
        text=True,
    )
    if ping.returncode != 0:
        print("rg.terra unreachable; numpy asinh MDS fallback")
        return None
    remote = TERRA_WORK
    mkdir = subprocess.run(
        ["ssh", TERRA, f"mkdir -p {remote}"], capture_output=True, text=True
    )
    if mkdir.returncode != 0:
        print("terra mkdir failed", mkdir.stderr)
        return None
    driver = SCRIPTS / "occ_book_soap_chi_terra.sh"
    for name in ("soap.hist", "soap.lm", "sigma.txt"):
        src = work / name
        scp = subprocess.run(
            ["scp", "-q", str(src), f"{TERRA}:{remote}/{name}"],
            capture_output=True,
            text=True,
        )
        if scp.returncode != 0:
            print("scp failed", name, scp.stderr)
            return None
    scp = subprocess.run(
        ["scp", "-q", str(driver), f"{TERRA}:{remote}/occ_book_soap_chi_terra.sh"],
        capture_output=True,
        text=True,
    )
    if scp.returncode != 0:
        print("scp driver failed", scp.stderr)
        return None
    lf = os.environ.get("TERRA_LANDFOLD") or os.environ.get("LANDFOLD")
    if not lf:
        print("set TERRA_LANDFOLD or LANDFOLD to the remote release binary")
        return None
    cmd = (
        f"chmod +x {remote}/occ_book_soap_chi_terra.sh; "
        f"LANDFOLD={lf} WORK={remote} "
        f"bash {remote}/occ_book_soap_chi_terra.sh"
    )
    print("terra embed+project nland", nland)
    run = subprocess.run(
        ["ssh", TERRA, cmd], capture_output=True, text=True, timeout=1800
    )
    print("terra rc", run.returncode)
    if run.stdout:
        print(run.stdout[-1600:])
    if run.stderr:
        print(run.stderr[-1600:])
    for name in ("soap.ld", "soap.err", "soap.proj", "soap.proj.err"):
        subprocess.run(
            ["scp", "-q", f"{TERRA}:{remote}/{name}", str(work / name)],
            capture_output=True,
        )
    proj = work / "soap.proj"
    if not proj.is_file() or proj.stat().st_size < 10:
        print("no terra projection")
        err = work / "soap.err"
        if err.is_file():
            print(err.read_text()[-800:])
        return None
    xy = np.loadtxt(proj)
    if xy.ndim != 2 or xy.shape[0] != 4042:
        print("bad proj shape", getattr(xy, "shape", None))
        return None
    err = (work / "soap.err").read_text() if (work / "soap.err").is_file() else ""
    return {"xy": xy, "err": err, "stress": parse_stress(err), "engine": "landfold"}


def draw(xy, z, gm, ico, path: Path, title: str, energy: np.ndarray):
    gx, gy, field = idw.fill(xy, z)
    finite = field[np.isfinite(field)]
    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor="white")
    mesh = ax.pcolormesh(
        gx, gy, np.clip(field, 0, VMAX), cmap=PES, shading="auto", vmin=0, vmax=VMAX
    )
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=np.clip(z, 0, VMAX),
        s=7,
        cmap=PES,
        vmin=0,
        vmax=VMAX,
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
    ax.legend(
        loc="best",
        fontsize=8,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="k",
    )
    ax.set_xlabel(r"$\chi_1$")
    ax.set_ylabel(r"$\chi_2$")
    ax.set_title(title)
    yy, xx = np.where(np.isfinite(field))
    if yy.size:
        ax.set_xlim(gx[xx.min()], gx[xx.max()])
        ax.set_ylim(gy[yy.min()], gy[yy.max()])
    fig.colorbar(mesh, ax=ax, label=r"$E-E_\mathrm{GM}$")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    print("wrote", path, "nfin", int(finite.size))
    return gx, gy, field


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    fp = load_fp()
    energy = np.loadtxt(ENERGY)
    if energy.shape[0] != fp.shape[0]:
        raise SystemExit(f"energy {energy.shape} vs SOAP {fp.shape}")
    if abs(float(energy[GM_IDX]) - GM_E) > 1e-3 or abs(float(energy[ICO_IDX]) - ICO_E) > 1e-3:
        raise SystemExit(f"index check failed E[0]={energy[GM_IDX]} E[40]={energy[ICO_IDX]}")
    gm, ico = GM_IDX, ICO_IDX
    print("n", len(energy), "hd", fp.shape[1], "E[GM]", float(energy[gm]), "E[ico]", float(energy[ico]))

    write_hist(DEST / "soap.hist", fp)

    cache_d = DEST / "soap.d.npy"
    if cache_d.is_file():
        dist = np.load(cache_d)
        print("loaded D cache", dist.shape)
    else:
        print("pairwise L2 of SOAP ...")
        dist = dp.pairwise_l2(fp)
        np.fill_diagonal(dist, 0.0)
        np.save(cache_d, dist)
    pos = dist[dist > 0]
    sigma = float(np.median(pos)) if pos.size else 1.0
    sigma = max(sigma, 1e-9)
    print("D(GM,ico)", float(dist[gm, ico]), "D med", sigma, "D max", float(dist.max()))
    (DEST / "sigma.txt").write_text("%.10e\n" % sigma)

    land_idx = farthest_on_d(dist, N_LAND, (gm, ico))
    if land_idx[0] != gm or ico not in set(land_idx.tolist()):
        raise SystemExit("landmarks must pin GM and ico")
    write_hist(DEST / "soap.lm", fp[land_idx])
    np.savetxt(DEST / "land.idx", land_idx, fmt="%d")
    print("landmarks", len(land_idx), "first", land_idx[:8], "ico at", int(np.where(land_idx == ico)[0][0]))

    xy_init, ev_init = numpy_asinh_mds(dist[np.ix_(land_idx, land_idx)], sigma)
    write_table(DEST / "init.xy", xy_init)
    print("init asinh MDS ev", ev_init)

    meta = {}
    terra = try_terra(DEST, len(land_idx))
    if terra is not None:
        xy = terra["xy"]
        engine = "landfold"
        meta = {
            "engine": engine,
            "stress": terra["stress"],
            "err": terra["err"].strip(),
        }
        print("landfold stress", terra["stress"])
    else:
        print("fallback: numpy asinh MDS of SOAP L2")
        xy, ev = numpy_asinh_mds(dist, sigma)
        engine = "numpy_asinh_mds"
        meta = {
            "engine": engine,
            "stress": stress_asinh(xy, dist, sigma),
            "ev": [float(x) for x in ev],
        }
        write_table(DEST / "fallback.xy", xy)
        print("fallback ev", ev, "stress", meta["stress"])

    xy = orient(xy, gm, ico)
    write_table(DEST / "soap_chi.xy", xy)
    z = energy - float(energy[gm])
    title = r"SOAP asinh $\chi$  energy IDW"
    if engine == "numpy_asinh_mds":
        title = r"SOAP asinh MDS  energy IDW"
    gx, gy, field = draw(xy, z, gm, ico, DEST / "elja_occ_lj38_soap_chi.png", title, energy)
    draw(xy, z, gm, ico, FIGS / "elja_occ_lj38_soap_chi.png", title, energy)

    rec = dp.score_energy("soap_chi", xy, gx, gy, field, energy, gm, ico)
    diam = rec["diam"]
    rec["engine"] = engine
    rec["stress_embed"] = meta.get("stress")
    rec["chi2_full"] = stress_asinh(xy, dist, sigma)
    rec["gm_barrier"] = ring_barrier(xy, gx, gy, field, gm, 0.06 * diam, 0.16 * diam)
    rec["sigma"] = sigma
    rec["n_land"] = int(len(land_idx))
    rec["hd"] = HD
    print(
        f"soap_chi sep_norm={rec['sep_norm']:.4f} wells={rec['n_wells']} "
        f"two={rec['two_basins']} gm_well={rec['gm_in_well']} "
        f"deeper={rec['gm_deeper']} rim={rec['gm_on_rim']} "
        f"bar={rec['gm_barrier']:.3f} Egm={rec['Efill_GM']} Eico={rec['Efill_ico']} "
        f"stress={rec['stress_embed']} chi2={rec['chi2_full']} verdict={rec['verdict']}"
    )

    payload = {
        "n": int(len(energy)),
        "hd": HD,
        "n_land": int(len(land_idx)),
        "gm_idx": int(gm),
        "ico_idx": int(ico),
        "E_GM": float(energy[gm]),
        "E_ico": float(energy[ico]),
        "D_gm_ico": float(dist[gm, ico]),
        "sigma": sigma,
        "asinh_norm": ASINH_NORM,
        "meta": meta,
        "score": rec,
        "note": "energy IDW of SOAP asinh chi; no occupancy leftover invert",
    }
    (DEST / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", DEST / "scores.json")
    print("FIG", FIGS / "elja_occ_lj38_soap_chi.png")
    print(
        "VERDICT",
        "engine",
        engine,
        "gm_in_well",
        rec["gm_in_well"],
        "gm_deeper",
        rec["gm_deeper"],
        "two_basins",
        rec["two_basins"],
        "rim",
        rec["gm_on_rim"],
        "stress",
        rec["stress_embed"],
        "chi2",
        rec["chi2_full"],
    )


if __name__ == "__main__":
    main()
