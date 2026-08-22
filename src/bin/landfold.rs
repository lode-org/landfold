//! landfold CLI: embed, project, landmarks, dist, mds, fes.
//!
//! `project` is a coarse-then-fine χ grid plus `--refine` steps
//! (Ceriotti, Tribello, Parrinello, *J. Chem. Theory Comput.* **9**,
//! 1521 (2013), <https://doi.org/10.1021/ct3010563>). `landmarks` is
//! Gonzalez farthest-point sampling. `fes` writes `F = -kT ln(rho/rhomax)`
//! as CSV/SVG and optional coordination histograms.

use std::io::{self, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use landfold::{
    axis_embed, axis_fit, axis_project, bands_embed, basin_coordinate, coordination_histogram,
    embed, embed_sigma_schedule, fit_imq_map, gap_split_embed, joint_pairwise_hist, knn_project,
    mds_from_points, nearfar_embed, pacmap_embed, pairwise, pairwise_euclid, phate_embed,
    phate_project, predict_imq, project_many_report, read_points, select_landmarks, suggest_alpha,
    suggest_scale, write_plumed, write_points, AnnealOpts, BandOpts, Dot, Embedding, Euclid,
    Fisher, FreeEnergy, Histogram2d, IterOpts, LandmarkMode, MdsMode, Metric, NearFarOpts,
    PacmapOpts, Periodic, PhateOpts, ProjOpts, ReplicaOpts, Solver, Sphere, StochOpts, Stretch,
    Transfer, Wasserstein1, FUN_SPEC_HELP, L1,
};

#[derive(Parser, Debug)]
#[command(
    name = "landfold",
    version,
    about = "Landscape And Nonlinear Distance Folding Onto Low Dimensions",
    after_help = "Python: cargo build --features python (pyo3/numpy 0.29; one major, dlpk pyo3 off)."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Fit a low-D embedding (standard CG, or --stoch for pair search)
    Embed {
        #[arg(short = 'D', default_value_t = 3)]
        high: usize,
        #[arg(short = 'd', default_value_t = 2)]
        low: usize,
        #[arg(long = "pi", default_value_t = 0.0)]
        period: f64,
        #[arg(long = "spi", default_value_t = 0.0)]
        sphere: f64,
        #[arg(short = 'w')]
        weighted: bool,
        #[arg(long)]
        dot: bool,
        #[arg(long)]
        l1: bool,
        #[arg(long)]
        center: bool,
        #[arg(long)]
        similarity: bool,
        #[arg(long = "fun-hd", default_value = "identity", help = FUN_SPEC_HELP)]
        fun_hd: String,
        #[arg(long = "fun-ld", default_value = "identity", help = FUN_SPEC_HELP)]
        fun_ld: String,
        #[arg(long = "imix", default_value_t = 0.0)]
        imix: f64,
        #[arg(long = "steps", default_value_t = 100)]
        steps: usize,
        /// First CG phase (`dimred -preopt`). Zero keeps `--steps`.
        #[arg(long = "preopt", default_value_t = 0)]
        preopt: usize,
        /// Pointwise global grid after preopt: gw,g1,g2 (`dimred -grid`)
        #[arg(long = "grid")]
        grid: Option<String>,
        /// CG steps after each successful grid move (`dimred -gopt`)
        #[arg(long = "gopt", default_value_t = 0)]
        gopt: usize,
        /// Write `# Error in fitting LD points:` on stdout (`dimred -v`)
        #[arg(short = 'v', long)]
        verbose: bool,
        /// PLUMED landmark dump (`dimred -plumed`)
        #[arg(long)]
        plumed: bool,
        /// Pair weights F(D)(1-F(D)): only mid-scale pairs drive χ
        #[arg(long)]
        midweight: bool,
        /// Classical MDS of F(D) rather than raw D
        #[arg(long = "init-f")]
        init_transformed: bool,
        /// Decreasing HD/LD σ, warm-started (last value is the published scale)
        #[arg(long = "continue-sigma")]
        continue_sigma: Option<String>,
        /// Stretch HD along `ref_a - ref_b` (two D-vectors, one per line)
        #[arg(long = "stretch")]
        stretch: Option<PathBuf>,
        /// Stretch weight. Default: send median between-class D to σ.
        #[arg(long = "alpha")]
        alpha: Option<f64>,
        /// Pooled within-class Mahalanobis using the two `--stretch` refs
        #[arg(long)]
        fisher: bool,
        #[arg(long = "init")]
        init: Option<PathBuf>,
        /// Random pair mini-batches instead of full-pair CG
        #[arg(long)]
        stoch: bool,
        #[arg(long = "batch", default_value_t = 64)]
        batch: usize,
        /// Simulated annealing then CG polish (extra arm)
        #[arg(long)]
        anneal: bool,
        /// Replica-exchange (parallel tempering) then CG polish
        #[arg(long)]
        replica: bool,
        /// L-BFGS two-loop with trust/center projection (needs --features highs)
        #[arg(long)]
        highs: bool,
        /// xtsci-optimize L-BFGS (extra arm; same ChiObjective)
        #[arg(long)]
        lbfgs: bool,
        /// Box bounds `lo,hi` for `--highs`
        #[arg(long = "box")]
        box_bounds: Option<String>,
        /// Per-site trust radius for `--highs` (default 0.5)
        #[arg(long = "trust")]
        trust: Option<f64>,
        /// PHATE (Moon et al., Nat. Biotechnol. 2019) instead of χ
        #[arg(long)]
        phate: bool,
        /// PHATE k-NN bandwidth (Moon default 5)
        #[arg(long = "phate-knn", default_value_t = 5)]
        phate_knn: usize,
        /// PHATE alpha-decay (Moon default 40)
        #[arg(long = "phate-decay", default_value_t = 40.0)]
        phate_decay: f64,
        /// PHATE diffusion time. Default: von Neumann entropy knee
        #[arg(long = "phate-t")]
        phate_t: Option<usize>,
        /// Gap-split: display (ψ, s) with χ only on pairs the slow mode
        /// does not already separate (Lean `chi_decouples`)
        #[arg(long)]
        gapsplit: bool,
        /// Gap cut on |Δψ|. Default: 0.35 of the ψ range
        #[arg(long = "gap-tau")]
        gap_tau: Option<f64>,
        /// PaCMAP: near-to-near, mid-near, far-to-far (Wang et al. 2021)
        #[arg(long)]
        pacmap: bool,
        /// 1-Wasserstein metric on histogram rows (n4..n13)
        #[arg(long = "w1")]
        w1: bool,
        /// Rank-uniform each low-D axis (empirical CDF)
        #[arg(long)]
        uniform: bool,
        /// Split isometry: identity on D≤σ and D≥τ, plus Riesz s=2
        #[arg(long)]
        nearfar: bool,
        #[arg(long = "near-cut")]
        near_cut: Option<f64>,
        #[arg(long = "far-cut")]
        far_cut: Option<f64>,
        #[arg(long = "riesz", default_value_t = 0.05)]
        riesz: f64,
        /// Identity near, Ceriotti χ mid, PaCMAP far repulsion, diameter pins
        #[arg(long)]
        bands: bool,
        /// Identity-match the K largest far pairs (diameter)
        #[arg(long = "pin-k", default_value_t = 256)]
        pin_k: usize,
        /// Steps with far weight zero so mid χ can form the lobes
        #[arg(long = "warm", default_value_t = 100)]
        warm: usize,
        /// Weight on PaCMAP far repulsion (three-band)
        #[arg(long = "far-weight", default_value_t = 1.0)]
        far_weight: f64,
        /// Exact projection onto the contrast of two reference rows
        #[arg(long = "axis")]
        axis: Option<PathBuf>,
    },
    /// Project new high-D rows into a fitted embedding (grid + local refine)
    Project {
        #[arg(short = 'D', default_value_t = 3)]
        high: usize,
        #[arg(short = 'd', default_value_t = 2)]
        low: usize,
        #[arg(long = "high-file")]
        high_file: PathBuf,
        #[arg(long = "low-file")]
        low_file: PathBuf,
        #[arg(long = "pi", default_value_t = 0.0)]
        period: f64,
        #[arg(long = "spi", default_value_t = 0.0)]
        sphere: f64,
        #[arg(short = 'w')]
        weighted: bool,
        #[arg(long)]
        dot: bool,
        #[arg(long)]
        similarity: bool,
        #[arg(long = "stretch")]
        stretch: Option<PathBuf>,
        #[arg(long = "alpha")]
        alpha: Option<f64>,
        #[arg(long)]
        fisher: bool,
        #[arg(long = "fun-hd", default_value = "identity", help = FUN_SPEC_HELP)]
        fun_hd: String,
        #[arg(long = "fun-ld", default_value = "identity", help = FUN_SPEC_HELP)]
        fun_ld: String,
        #[arg(long = "imix", default_value_t = 0.0)]
        imix: f64,
        /// Coarse then fine grid: half-width, coarse points, fine points
        #[arg(long = "grid", default_value = "1.0,21,201")]
        grid: String,
        /// Polak-Ribiere + Brent steps after the grid minimum
        #[arg(long = "refine", default_value_t = 0)]
        refine: usize,
        /// Also print χ and nearest-landmark HD distance
        #[arg(long)]
        print_error: bool,
        /// Path-like average of landmark LD coords (`dimproj -path`)
        #[arg(long = "path", default_value_t = -1.0)]
        path: f64,
        /// Softmax temperature on the grid (`dimproj -gt`). Zero keeps the min.
        #[arg(long = "gt", default_value_t = 0.0)]
        gtemp: f64,
        /// PHATE OOS (Nyström potential + metric MDS) instead of χ grid
        #[arg(long)]
        phate: bool,
        #[arg(long = "phate-knn", default_value_t = 5)]
        phate_knn: usize,
        #[arg(long = "phate-decay", default_value_t = 40.0)]
        phate_decay: f64,
        #[arg(long = "phate-t")]
        phate_t: Option<usize>,
        #[arg(long)]
        pacmap: bool,
        #[arg(long)]
        nearfar: bool,
        #[arg(long)]
        bands: bool,
        #[arg(long = "w1")]
        w1: bool,
        #[arg(long)]
        l1: bool,
        #[arg(long = "knn", default_value_t = 10)]
        knn: usize,
        /// Reuse `--axis` refs; residual PCs come from `--high-file`
        #[arg(long = "axis")]
        axis: Option<PathBuf>,
    },
    /// Farthest-point (Gonzalez) landmarks
    Landmarks {
        #[arg(short = 'D', default_value_t = 3)]
        high: usize,
        #[arg(short = 'n', default_value_t = 100)]
        nland: usize,
        #[arg(long = "pi", default_value_t = 0.0)]
        period: f64,
        #[arg(long = "spi", default_value_t = 0.0)]
        sphere: f64,
        #[arg(long)]
        dot: bool,
        #[arg(long)]
        l1: bool,
        #[arg(short = 'w')]
        weighted: bool,
        #[arg(long, default_value_t = 0)]
        seed: usize,
        /// Pin the leading N rows as landmarks before farthest-point.
        #[arg(long = "ifirst", default_value_t = 0)]
        ifirst: usize,
        /// Write `# indices ...` before the point table
        #[arg(long)]
        indices: bool,
        /// Replace FPS scores with Voronoi masses of the source points
        #[arg(long)]
        voronoi: bool,
        #[arg(long = "wgamma", default_value_t = 1.0)]
        wgamma: f64,
        /// C++ `dimlandmark -mode`: stride | random | minmax | resample | staged
        #[arg(long = "mode", default_value = "minmax")]
        mode: String,
        /// `resample` / `staged` temperature (`dimlandmark -gamma`)
        #[arg(long, default_value_t = 1.0)]
        gamma: f64,
        /// Refuse duplicate indices (`dimlandmark -unique`)
        #[arg(long)]
        unique: bool,
    },
    /// Pairwise distances, Appendix A σ, or C++ `dimdist` joint P(D,d).
    Dist {
        #[arg(short = 'D', default_value_t = 3)]
        high: usize,
        #[arg(short = 'd', default_value_t = 0)]
        low: usize,
        #[arg(long = "pi", default_value_t = 0.0)]
        period: f64,
        #[arg(long)]
        l1: bool,
        #[arg(long)]
        dot: bool,
        #[arg(short = 'w')]
        weighted: bool,
        /// Print \(\sigma\) suggestion instead of the distance matrix
        #[arg(long)]
        suggest: bool,
        /// Low-D coordinates for the joint P(D,d) histogram (`dimdist -p`)
        #[arg(long = "low-file", visible_alias = "p")]
        low_file: Option<PathBuf>,
        #[arg(long = "nbin", default_value_t = 100)]
        nbin: usize,
        #[arg(long)]
        gnuplot: bool,
        /// HD,LD histogram ceilings (default: observed maxima)
        #[arg(long = "maxd")]
        maxd: Option<String>,
    },
    /// Classical Torgerson MDS (Torgerson 1952). Default solver init.
    ///
    /// `--distances` dumps i<j-equivalent full pairwise of the embedding:
    /// those distances are the C++ Torgerson pairwise invariant (coords are unsigned).
    Mds {
        #[arg(short = 'D', default_value_t = 3)]
        high: usize,
        #[arg(short = 'd', default_value_t = 2)]
        low: usize,
        #[arg(long = "pi", default_value_t = 0.0)]
        period: f64,
        /// Dump pairwise distances of the embedding (sign-invariant)
        #[arg(long)]
        distances: bool,
    },
    /// 2-D F = -kT ln(rho/rhomax) plus optional coordination histogram
    Fes {
        #[arg(long)]
        input: Option<PathBuf>,
        #[arg(long, default_value_t = 80)]
        nx: usize,
        #[arg(long, default_value_t = 80)]
        ny: usize,
        #[arg(long = "kt", default_value_t = 1.0)]
        kt: f64,
        #[arg(long)]
        xmin: Option<f64>,
        #[arg(long)]
        xmax: Option<f64>,
        #[arg(long)]
        ymin: Option<f64>,
        #[arg(long)]
        ymax: Option<f64>,
        #[arg(long)]
        csv: Option<PathBuf>,
        #[arg(long)]
        svg: Option<PathBuf>,
        /// Cartesian frames (3-D rows) for a coordination-number histogram
        #[arg(long)]
        frames: Option<PathBuf>,
        #[arg(long = "cn-csv")]
        cn_csv: Option<PathBuf>,
        #[arg(long, default_value_t = 1.2)]
        cn_cutoff: f64,
        #[arg(long, default_value_t = 12)]
        cn_max: usize,
        /// Gaussian blur of the count field, in bins (0 = off)
        #[arg(long, default_value_t = 0.0)]
        blur: f64,
        /// Colour-scale ceiling for the SVG (JCTC 2013 panel uses 2)
        #[arg(long = "fmax", default_value_t = 2.0)]
        fmax: f64,
        /// Keep the largest connected body above this fraction of rho_max
        #[arg(long = "floor", default_value_t = 0.0)]
        floor: f64,
    },
    /// MAP IMQ-GP of the basin coordinate ξ on a landfold plane
    Field {
        #[arg(short = 'D', default_value_t = 10)]
        high: usize,
        #[arg(long = "low-file")]
        low_file: PathBuf,
        /// Two reference rows (fcc, ico) in the same HD
        #[arg(long = "refs")]
        refs: PathBuf,
        #[arg(short = 'w')]
        weighted: bool,
        #[arg(long, default_value_t = 60)]
        nx: usize,
        #[arg(long, default_value_t = 60)]
        ny: usize,
    },
}

fn main() -> landfold::Result<()> {
    match Cli::parse().cmd {
        Cmd::Embed {
            high,
            low,
            period,
            sphere,
            weighted,
            dot,
            l1,
            center,
            similarity,
            fun_hd,
            fun_ld,
            imix,
            steps,
            preopt,
            grid,
            gopt,
            verbose,
            plumed,
            midweight,
            init_transformed,
            continue_sigma,
            stretch,
            alpha,
            fisher,
            init,
            stoch,
            batch,
            anneal,
            replica,
            highs,
            lbfgs,
            box_bounds,
            trust,
            phate,
            phate_knn,
            phate_decay,
            phate_t,
            gapsplit,
            gap_tau,
            pacmap,
            w1,
            uniform,
            nearfar,
            near_cut,
            far_cut,
            riesz,
            bands,
            pin_k,
            warm,
            far_weight,
            axis,
        } => {
            let set = read_points(io::stdin().lock(), high, weighted)?;
            if bands {
                let metric: Box<dyn Metric> = if w1 {
                    Box::new(Wasserstein1)
                } else {
                    Box::new(Euclid)
                };
                let polish = init.is_some();
                let popts = BandOpts {
                    lowdim: low,
                    near: near_cut,
                    far: far_cut,
                    tfun_hd: Transfer::from_cli(&fun_hd)?,
                    tfun_ld: Transfer::from_cli(&fun_ld)?,
                    riesz: if polish && (riesz - 0.05).abs() < 1e-15 {
                        0.0
                    } else {
                        riesz
                    },
                    pin_k,
                    warm: if polish && warm == 100 { 0 } else { warm },
                    far_weight: if polish && (far_weight - 1.0).abs() < 1e-15 {
                        0.5
                    } else {
                        far_weight
                    },
                    steps: if polish && steps == 100 { 80 } else { steps },
                    ..BandOpts::default()
                };
                let init_pts = if let Some(p) = init {
                    Some(read_points(
                        std::io::BufReader::new(std::fs::File::open(p)?),
                        low,
                        false,
                    )?)
                } else {
                    None
                };
                let (coords, rep) = bands_embed(
                    set.points.view(),
                    metric.as_ref(),
                    &popts,
                    init_pts.as_ref().map(|p| p.points.view()),
                )?;
                write_points(&mut io::stdout().lock(), &coords, None)?;
                writeln!(
                    io::stderr(),
                    "# bands sigma {} tau {} near {} mid {} far {} pin {} warm {} riesz {}",
                    rep.sigma,
                    rep.tau,
                    rep.n_near,
                    rep.n_mid,
                    rep.n_far,
                    rep.n_pin,
                    warm,
                    riesz
                )?;
                return Ok(());
            }
            if let Some(path) = axis {
                let refs = read_points(
                    std::io::BufReader::new(std::fs::File::open(path)?),
                    high,
                    false,
                )?;
                if refs.points.nrows() != 2 {
                    return Err(landfold::LandfoldError::Parse(
                        "--axis needs exactly two reference rows".into(),
                    ));
                }
                let (coords, model) = axis_embed(
                    set.points.view(),
                    refs.points.row(0),
                    refs.points.row(1),
                    low,
                )?;
                write_points(&mut io::stdout().lock(), &coords, None)?;
                writeln!(io::stderr(), "# axis gap {} s1_span {}", model.gap, {
                    let mut lo = f64::INFINITY;
                    let mut hi = f64::NEG_INFINITY;
                    for i in 0..coords.nrows() {
                        lo = lo.min(coords[(i, 0)]);
                        hi = hi.max(coords[(i, 0)]);
                    }
                    hi - lo
                })?;
                return Ok(());
            }
            if nearfar {
                let metric: Box<dyn Metric> = if w1 {
                    Box::new(Wasserstein1)
                } else {
                    Box::new(Euclid)
                };
                let popts = NearFarOpts {
                    lowdim: low,
                    near: near_cut,
                    far: far_cut,
                    riesz,
                    ..NearFarOpts::default()
                };
                let (coords, sigma, tau) =
                    nearfar_embed(set.points.view(), metric.as_ref(), &popts)?;
                write_points(&mut io::stdout().lock(), &coords, None)?;
                writeln!(
                    io::stderr(),
                    "# near-far sigma {} tau {} riesz {}",
                    sigma,
                    tau,
                    riesz
                )?;
                return Ok(());
            }
            if pacmap {
                let metric: Box<dyn Metric> = if w1 {
                    Box::new(Wasserstein1)
                } else {
                    Box::new(Euclid)
                };
                let popts = PacmapOpts {
                    lowdim: low,
                    uniform,
                    ..PacmapOpts::default()
                };
                let coords = pacmap_embed(set.points.view(), metric.as_ref(), &popts)?;
                write_points(&mut io::stdout().lock(), &coords, None)?;
                writeln!(
                    io::stderr(),
                    "# PaCMAP neighbors {} uniform {} w1 {}",
                    popts.n_neighbors,
                    uniform,
                    w1
                )?;
                return Ok(());
            }
            if gapsplit {
                let mut opts = IterOpts {
                    lowdim: 1,
                    tfun_hd: Transfer::from_cli(&fun_hd)?,
                    tfun_ld: Transfer::from_cli(&fun_ld)?,
                    ..IterOpts::default()
                };
                opts.cg.maxiter = steps;
                opts.preopt = preopt;
                opts.init_transformed = init_transformed;
                let (coords, rep) = gap_split_embed(
                    set.points.view(),
                    &Euclid,
                    &opts,
                    phate_knn,
                    phate_decay,
                    gap_tau,
                )?;
                write_points(&mut io::stdout().lock(), &coords, None)?;
                writeln!(
                    io::stderr(),
                    "# gap-split tau {} kept {}/{} pairs",
                    rep.tau,
                    rep.n_kept,
                    rep.n_pairs
                )?;
                return Ok(());
            }
            if phate {
                let popts = PhateOpts {
                    knn: phate_knn,
                    decay: phate_decay,
                    t: phate_t,
                    lowdim: low,
                    ..PhateOpts::default()
                };
                let (coords, model) = phate_embed(set.points.view(), &Euclid, &popts)?;
                let mut out = io::stdout().lock();
                write_points(&mut out, &coords, None)?;
                writeln!(
                    io::stderr(),
                    "# PHATE t {} knn {} decay {} gamma {}",
                    model.t,
                    model.knn,
                    model.decay,
                    model.gamma
                )?;
                return Ok(());
            }
            let mut opts = IterOpts {
                lowdim: low,
                imix,
                tfun_hd: Transfer::from_cli(&fun_hd)?,
                tfun_ld: Transfer::from_cli(&fun_ld)?,
                center,
                solver: if highs {
                    #[cfg(feature = "highs")]
                    {
                        let mut ho = landfold::HighsOpts {
                            maxiter: steps,
                            adaptive_trust: trust.is_none(),
                            ..landfold::HighsOpts::default()
                        };
                        if let Some(t) = trust {
                            ho.trust = t;
                        }
                        if let Some(spec) = box_bounds {
                            let parts: Vec<f64> = spec
                                .split(',')
                                .map(|s| s.trim().parse::<f64>())
                                .collect::<std::result::Result<Vec<_>, _>>()
                                .map_err(|e| landfold::LandfoldError::Parse(e.to_string()))?;
                            if parts.len() != 2 {
                                return Err(landfold::LandfoldError::Parse(
                                    "--box needs lo,hi".into(),
                                ));
                            }
                            ho.lo = Some(parts[0]);
                            ho.hi = Some(parts[1]);
                        }
                        Solver::Highs(ho)
                    }
                    #[cfg(not(feature = "highs"))]
                    {
                        let _ = (box_bounds, trust);
                        return Err(landfold::LandfoldError::Msg(
                            "rebuild with --features highs for the HiGHS arm".into(),
                        ));
                    }
                } else if lbfgs {
                    Solver::Xtsci(xtsci_optimize::Method::lbfgs())
                } else if replica {
                    Solver::Replica(ReplicaOpts {
                        steps,
                        ..ReplicaOpts::default()
                    })
                } else if anneal {
                    Solver::Anneal(AnnealOpts {
                        steps,
                        ..AnnealOpts::default()
                    })
                } else if stoch {
                    Solver::Stochastic(StochOpts {
                        steps,
                        batch,
                        ..StochOpts::default()
                    })
                } else {
                    Solver::Standard
                },
                ..IterOpts::default()
            };
            opts.cg.maxiter = steps;
            opts.preopt = preopt;
            opts.gopt = gopt;
            opts.midweight = midweight;
            opts.init_transformed = init_transformed;
            if let Some(spec) = grid {
                opts.global = Some(ProjOpts::from_cli(&spec)?);
            }
            let init = if let Some(p) = init {
                Some(read_points(
                    std::io::BufReader::new(std::fs::File::open(p)?),
                    low,
                    false,
                )?)
            } else {
                None
            };
            let metric: Box<dyn Metric> = if let Some(path) = stretch {
                let refs = read_points(
                    std::io::BufReader::new(std::fs::File::open(path)?),
                    high,
                    false,
                )?;
                if refs.points.nrows() != 2 {
                    return Err(landfold::LandfoldError::Parse(
                        "--stretch needs exactly two reference rows".into(),
                    ));
                }
                let a: Vec<f64> = refs.points.row(0).iter().copied().collect();
                let b: Vec<f64> = refs.points.row(1).iter().copied().collect();
                if fisher {
                    let pts: Vec<Vec<f64>> = set
                        .points
                        .rows()
                        .into_iter()
                        .map(|r| r.iter().copied().collect())
                        .collect();
                    Box::new(Fisher::from_refs(&pts, &a, &b, 1e-3)?)
                } else {
                    let alpha = resolve_alpha(set.points.view(), &a, &b, &fun_hd, alpha)?;
                    Box::new(Stretch::from_refs(&a, &b, alpha)?)
                }
            } else if fisher {
                return Err(landfold::LandfoldError::Parse(
                    "--fisher needs --stretch refs".into(),
                ));
            } else if l1 {
                Box::new(L1)
            } else if dot {
                Box::new(landfold::Dot)
            } else if sphere != 0.0 {
                Box::new(Sphere::new(vec![sphere; high])?)
            } else if period != 0.0 {
                Box::new(Periodic::isotropic(high, period)?)
            } else {
                Box::new(Euclid)
            };
            let pre = if similarity {
                Some(set.points.view())
            } else {
                None
            };
            let (emb, _) = if let Some(spec) = continue_sigma {
                let sigmas: Vec<f64> = spec
                    .split(',')
                    .map(|s| s.trim().parse::<f64>())
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| landfold::LandfoldError::Parse(e.to_string()))?;
                embed_sigma_schedule(
                    set.points.view(),
                    metric.as_ref(),
                    &opts,
                    &sigmas,
                    init.as_ref().map(|p| p.points.view()),
                    set.weights.as_ref().map(|w| w.view()),
                    pre,
                )?
            } else {
                embed(
                    set.points.view(),
                    metric.as_ref(),
                    &opts,
                    init.as_ref().map(|p| p.points.view()),
                    set.weights.as_ref().map(|w| w.view()),
                    pre,
                )?
            };
            let mut out = io::stdout().lock();
            if verbose {
                writeln!(out, " # Error in fitting LD points: {}", emb.stress)?;
            }
            if plumed {
                write_plumed(&mut out, &emb.high, &emb.low, Some(&emb.weights))?;
            } else {
                write_points(&mut out, &emb.low, None)?;
            }
            writeln!(io::stderr(), "# stress {}", emb.stress)?;
            let _ = trust;
        }
        Cmd::Project {
            high,
            low,
            high_file,
            low_file,
            period,
            sphere,
            weighted,
            dot,
            similarity,
            stretch,
            alpha,
            fisher,
            fun_hd,
            fun_ld,
            imix,
            grid,
            refine,
            print_error,
            path,
            gtemp,
            phate,
            phate_knn,
            phate_decay,
            phate_t,
            pacmap,
            nearfar,
            bands,
            w1,
            l1,
            knn,
            axis,
        } => {
            let hi = read_points(
                std::io::BufReader::new(std::fs::File::open(&high_file)?),
                high,
                weighted,
            )?;
            if let Some(path) = axis {
                let refs = read_points(
                    std::io::BufReader::new(std::fs::File::open(path)?),
                    high,
                    false,
                )?;
                if refs.points.nrows() != 2 {
                    return Err(landfold::LandfoldError::Parse(
                        "--axis needs exactly two reference rows".into(),
                    ));
                }
                let model = axis_fit(
                    hi.points.view(),
                    refs.points.row(0),
                    refs.points.row(1),
                    low,
                )?;
                let query = read_points(io::stdin().lock(), high, false)?;
                let proj = axis_project(&model, query.points.view())?;
                write_points(&mut io::stdout().lock(), &proj, None)?;
                return Ok(());
            }
            let lo = read_points(
                std::io::BufReader::new(std::fs::File::open(&low_file)?),
                low,
                false,
            )?;
            if pacmap || nearfar || bands {
                let metric: Box<dyn Metric> = if l1 {
                    Box::new(L1)
                } else if w1 {
                    Box::new(Wasserstein1)
                } else {
                    Box::new(Euclid)
                };
                let query = read_points(io::stdin().lock(), high, false)?;
                let proj = knn_project(
                    hi.points.view(),
                    lo.points.view(),
                    query.points.view(),
                    metric.as_ref(),
                    knn,
                )?;
                write_points(&mut io::stdout().lock(), &proj, None)?;
                return Ok(());
            }
            if phate {
                let popts = PhateOpts {
                    knn: phate_knn,
                    decay: phate_decay,
                    t: phate_t,
                    lowdim: low,
                    ..PhateOpts::default()
                };
                let query = read_points(io::stdin().lock(), high, false)?;
                let proj = phate_project(
                    hi.points.view(),
                    lo.points.view(),
                    query.points.view(),
                    &Euclid,
                    &popts,
                )?;
                write_points(&mut io::stdout().lock(), &proj, None)?;
                return Ok(());
            }
            let t_hd = Transfer::from_cli(&fun_hd)?;
            let t_ld = Transfer::from_cli(&fun_ld)?;
            let metric: Box<dyn Metric> = if let Some(path) = stretch {
                let refs = read_points(
                    std::io::BufReader::new(std::fs::File::open(path)?),
                    high,
                    false,
                )?;
                if refs.points.nrows() != 2 {
                    return Err(landfold::LandfoldError::Parse(
                        "--stretch needs exactly two reference rows".into(),
                    ));
                }
                let a: Vec<f64> = refs.points.row(0).iter().copied().collect();
                let b: Vec<f64> = refs.points.row(1).iter().copied().collect();
                if fisher {
                    let pts: Vec<Vec<f64>> = hi
                        .points
                        .rows()
                        .into_iter()
                        .map(|r| r.iter().copied().collect())
                        .collect();
                    Box::new(Fisher::from_refs(&pts, &a, &b, 1e-3)?)
                } else {
                    let alpha = resolve_alpha(hi.points.view(), &a, &b, &fun_hd, alpha)?;
                    Box::new(Stretch::from_refs(&a, &b, alpha)?)
                }
            } else if fisher {
                return Err(landfold::LandfoldError::Parse(
                    "--fisher needs --stretch refs".into(),
                ));
            } else if l1 {
                Box::new(L1)
            } else if w1 {
                Box::new(Wasserstein1)
            } else if dot {
                Box::new(Dot)
            } else if sphere != 0.0 {
                Box::new(Sphere::new(vec![sphere; high])?)
            } else if period != 0.0 {
                Box::new(Periodic::isotropic(high, period)?)
            } else {
                Box::new(Euclid)
            };
            let emb = Embedding::from_landmarks(
                hi.points,
                lo.points,
                metric.as_ref(),
                t_hd,
                t_ld,
                imix,
                hi.weights,
            )?;
            let mut po = ProjOpts::from_cli(&grid)?;
            po.cg_steps = refine;
            po.similarity = similarity;
            po.path_lambda = path;
            po.gtemp = gtemp;
            let qdim = if similarity { emb.high.nrows() } else { high };
            let q = read_points(io::stdin().lock(), qdim, false)?;
            let reports = project_many_report(&emb, q.points.view(), metric.as_ref(), &po)?;
            let mut out = io::stdout().lock();
            for r in &reports {
                for h in 0..r.coords.len() {
                    if h > 0 {
                        write!(out, " ")?;
                    }
                    write!(out, "{:.12}", r.coords[h])?;
                }
                if print_error {
                    write!(out, " {:.12} {:.12}", r.chi, r.nearest)?;
                }
                writeln!(out)?;
            }
        }
        Cmd::Landmarks {
            high,
            nland,
            period,
            sphere,
            dot,
            l1,
            weighted,
            seed,
            ifirst,
            indices,
            voronoi,
            wgamma,
            mode,
            gamma,
            unique,
        } => {
            let set = read_points(io::stdin().lock(), high, weighted)?;
            let metric: Box<dyn Metric> = if l1 {
                Box::new(L1)
            } else if dot {
                Box::new(Dot)
            } else if sphere != 0.0 {
                Box::new(Sphere::new(vec![sphere; high])?)
            } else if period != 0.0 {
                Box::new(Periodic::isotropic(high, period)?)
            } else {
                Box::new(Euclid)
            };
            let mode = LandmarkMode::from_cli(&mode, gamma)?;
            let mut lm = select_landmarks(
                set.points.view(),
                metric.as_ref(),
                nland,
                set.weights.as_ref().map(|w| w.view()),
                seed,
                (ifirst > 0).then_some(ifirst),
                unique,
                mode,
            )?;
            if voronoi {
                lm.assign_voronoi(
                    set.points.view(),
                    metric.as_ref(),
                    set.weights.as_ref().map(|w| w.view()),
                    wgamma,
                )?;
            }
            let mut out = io::stdout().lock();
            if indices {
                write!(out, "# indices")?;
                for i in &lm.index {
                    write!(out, " {i}")?;
                }
                writeln!(out)?;
            }
            write_points(&mut out, &lm.points, Some(&lm.weights))?;
        }
        Cmd::Dist {
            high,
            low,
            period,
            l1,
            dot,
            weighted,
            suggest,
            low_file,
            nbin,
            gnuplot,
            maxd,
        } => {
            let set = read_points(io::stdin().lock(), high, weighted)?;
            let metric: Box<dyn Metric> = if l1 {
                Box::new(L1)
            } else if dot {
                Box::new(Dot)
            } else if period != 0.0 {
                Box::new(Periodic::isotropic(high, period)?)
            } else {
                Box::new(Euclid)
            };
            let d = if l1 || period != 0.0 || dot {
                pairwise(set.points.view(), metric.as_ref())?
            } else {
                pairwise_euclid(set.points.view())?
            };
            if let Some(p) = low_file {
                let ld_dim = if low > 0 { low } else { 2 };
                let lo = read_points(
                    std::io::BufReader::new(std::fs::File::open(p)?),
                    ld_dim,
                    false,
                )?;
                let ld = pairwise_euclid(lo.points.view())?;
                let (max_d, max_r) = match maxd.as_deref() {
                    None => (None, None),
                    Some(spec) => {
                        let parts: Vec<f64> = spec
                            .split(',')
                            .map(|s| s.trim().parse::<f64>())
                            .collect::<std::result::Result<Vec<_>, _>>()
                            .map_err(|e| landfold::LandfoldError::Parse(e.to_string()))?;
                        match parts.as_slice() {
                            [a] => (Some(*a), Some(*a)),
                            [a, b] => (Some(*a), Some(*b)),
                            _ => {
                                return Err(landfold::LandfoldError::Parse(
                                    "--maxd needs maxD or maxD,maxd".into(),
                                ));
                            }
                        }
                    }
                };
                let (hist, frac) = joint_pairwise_hist(
                    d.view(),
                    ld.view(),
                    nbin,
                    nbin,
                    max_d,
                    max_r,
                    set.weights.as_ref().map(|w| w.view()),
                )?;
                let mut out = io::stdout().lock();
                writeln!(out, "# Fraction outside: {frac}")?;
                hist.write_joint(&mut out, gnuplot)?;
            } else if suggest {
                let n = d.nrows();
                let mut pairs = Vec::with_capacity(n.saturating_mul(n.saturating_sub(1)) / 2);
                for i in 0..n {
                    for j in 0..i {
                        pairs.push(d[(i, j)]);
                    }
                }
                let s = suggest_scale(&pairs)?;
                let mut out = io::stdout().lock();
                writeln!(out, "# pairs {}", s.n_pairs)?;
                writeln!(out, "# q25 {:.8} q50 {:.8} q75 {:.8}", s.q25, s.q50, s.q75)?;
                writeln!(out, "# knee {:.8}", s.knee)?;
                writeln!(out, "ceriotti,{:.6},8,1", s.knee)?;
                writeln!(out, "imq,{:.6}", s.knee)?;
            } else {
                write_points(&mut io::stdout().lock(), &d, None)?;
            }
        }
        Cmd::Mds {
            high,
            low,
            period,
            distances,
        } => {
            let set = read_points(io::stdin().lock(), high, false)?;
            let metric: Box<dyn Metric> = if period != 0.0 {
                Box::new(Periodic::isotropic(high, period)?)
            } else {
                Box::new(Euclid)
            };
            let (emb, report) =
                mds_from_points(set.points.view(), metric.as_ref(), low, MdsMode::Classical)?;
            if distances {
                write_points(
                    &mut io::stdout().lock(),
                    &pairwise_euclid(emb.view())?,
                    None,
                )?;
            } else {
                write_points(&mut io::stdout().lock(), &emb, None)?;
            }
            writeln!(
                io::stderr(),
                "# mds ld_error {} evals {}",
                report.ld_error,
                report
                    .eigenvalues
                    .iter()
                    .map(|v| format!("{v}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            )?;
        }
        Cmd::Fes {
            input,
            nx,
            ny,
            kt,
            xmin,
            xmax,
            ymin,
            ymax,
            csv,
            svg,
            frames,
            cn_csv,
            cn_cutoff,
            cn_max,
            blur,
            fmax,
            floor,
        } => {
            let set = if let Some(p) = input {
                read_points(std::io::BufReader::new(std::fs::File::open(p)?), 2, false)?
            } else {
                read_points(io::stdin().lock(), 2, false)?
            };
            let xs = set.points.column(0);
            let ys = set.points.column(1);
            let (axmin, axmax) = minmax(xs.iter().copied());
            let (aymin, aymax) = minmax(ys.iter().copied());
            let px = 0.05 * (axmax - axmin).max(1e-6);
            let py = 0.05 * (aymax - aymin).max(1e-6);
            let xlo = xmin.unwrap_or(axmin - px);
            let xhi = xmax.unwrap_or(axmax + px);
            let ylo = ymin.unwrap_or(aymin - py);
            let yhi = ymax.unwrap_or(aymax + py);
            let mut h = Histogram2d::new(xlo, xhi, nx, ylo, yhi, ny)?;
            h.add_points(set.points.view(), set.weights.as_ref().map(|w| w.view()))?;
            let mut fes = if blur > 0.0 {
                FreeEnergy::from_histogram_blurred(&h, kt, blur)?
            } else {
                FreeEnergy::from_histogram(&h, kt)?
            };
            if floor > 0.0 {
                fes.connected_body(floor)?;
            }
            if let Some(p) = csv {
                fes.write_csv(&mut std::fs::File::create(p)?)?;
            } else {
                fes.write_csv(&mut io::stdout().lock())?;
            }
            if let Some(p) = svg {
                fes.write_svg_scaled(&mut std::fs::File::create(p)?, 800, 640, fmax)?;
            }
            if let Some(p) = frames {
                let fr = read_points(std::io::BufReader::new(std::fs::File::open(p)?), 3, false)?;
                let cn = coordination_histogram(fr.points.view(), cn_cutoff, cn_max)?;
                if let Some(out) = cn_csv {
                    cn.write_cn_csv(&mut std::fs::File::create(out)?)?;
                } else {
                    for i in 0..cn.counts.len() {
                        writeln!(io::stderr(), "# CN {} {}", i, cn.counts[i])?;
                    }
                }
            } else if cn_csv.is_some() {
                return Err(landfold::LandfoldError::Parse(
                    "--cn-csv needs --frames".into(),
                ));
            }
        }
        Cmd::Field {
            high,
            low_file,
            refs,
            weighted,
            nx,
            ny,
        } => {
            let hi = read_points(io::stdin().lock(), high, weighted)?;
            let lo = read_points(
                std::io::BufReader::new(std::fs::File::open(low_file)?),
                2,
                false,
            )?;
            let rf = read_points(
                std::io::BufReader::new(std::fs::File::open(refs)?),
                high,
                false,
            )?;
            if hi.points.nrows() != lo.points.nrows() {
                return Err(landfold::LandfoldError::Shape(
                    "field: HD rows must match the low-file",
                ));
            }
            if rf.points.nrows() != 2 {
                return Err(landfold::LandfoldError::Parse(
                    "--refs needs exactly two rows".into(),
                ));
            }
            let xi = basin_coordinate(hi.points.view(), rf.points.row(0), rf.points.row(1))?;
            let gp = fit_imq_map(lo.points.view(), xi.view())?;
            writeln!(
                io::stderr(),
                "# field MAP ell {} sf2 {} noise {} nll {}",
                gp.ell,
                gp.sigma_f2,
                gp.noise,
                gp.nll
            )?;
            let (xmin, xmax) = minmax(lo.points.column(0).iter().copied());
            let (ymin, ymax) = minmax(lo.points.column(1).iter().copied());
            let px = 0.08 * (xmax - xmin).max(1e-6);
            let py = 0.08 * (ymax - ymin).max(1e-6);
            let mut grid = ndarray::Array2::<f64>::zeros((nx * ny, 2));
            for iy in 0..ny {
                let yv = ymin - py + (ymax - ymin + 2.0 * py) * (iy as f64) / (ny - 1) as f64;
                for ix in 0..nx {
                    let xv = xmin - px + (xmax - xmin + 2.0 * px) * (ix as f64) / (nx - 1) as f64;
                    let r = iy * nx + ix;
                    grid[(r, 0)] = xv;
                    grid[(r, 1)] = yv;
                }
            }
            let pred = predict_imq(&gp, lo.points.view(), xi.view(), grid.view())?;
            let mut out = io::stdout().lock();
            for i in 0..grid.nrows() {
                writeln!(
                    out,
                    "{:.8} {:.8} {:.8} {:.8}",
                    grid[(i, 0)],
                    grid[(i, 1)],
                    pred.mean[i],
                    pred.var[i]
                )?;
            }
        }
    }
    Ok(())
}

fn resolve_alpha(
    points: ndarray::ArrayView2<f64>,
    a: &[f64],
    b: &[f64],
    fun_hd: &str,
    explicit: Option<f64>,
) -> landfold::Result<f64> {
    if let Some(v) = explicit {
        if !v.is_finite() || v < 0.0 {
            return Err(landfold::LandfoldError::Parse(
                "--alpha must be finite and nonnegative".into(),
            ));
        }
        return Ok(v);
    }
    let tfun = Transfer::from_cli(fun_hd)?;
    if tfun.xsigmoid_params().is_none() {
        return Err(landfold::LandfoldError::Parse(
            "--stretch without --alpha needs a Ceriotti --fun-hd".into(),
        ));
    }
    let ra = ndarray::Array1::from(a.to_vec());
    let rb = ndarray::Array1::from(b.to_vec());
    let s = suggest_alpha(points, ra.view(), rb.view(), &tfun)?;
    writeln!(
        io::stderr(),
        "# stretch MAP alpha {}  68% [{}, {}]  plugin {}  sigma {}  median_within {}  median_between {}",
        s.alpha,
        s.alpha_lo,
        s.alpha_hi,
        s.alpha_plugin,
        s.sigma,
        s.median_within,
        s.median_between
    )?;
    Ok(s.alpha)
}

fn minmax<I: Iterator<Item = f64>>(it: I) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for v in it {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    (lo, hi)
}
