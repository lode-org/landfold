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
    AnnealOpts, Dot, Embedding, Euclid, FUN_SPEC_HELP, FreeEnergy, Histogram2d, IterOpts, L1, Stretch,
    LandmarkMode, MdsMode, Metric, Periodic, ProjOpts, ReplicaOpts, Solver, Sphere, StochOpts,
    Transfer, coordination_histogram, embed, embed_sigma_schedule, joint_pairwise_hist,
    mds_from_points, pairwise,
    pairwise_euclid, project_many_report, read_points, select_landmarks, suggest_scale,
    write_plumed, write_points,
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
        #[arg(long = "alpha", default_value_t = 3.0)]
        alpha: f64,
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
            init,
            stoch,
            batch,
            anneal,
            replica,
            highs,
            lbfgs,
            box_bounds,
            trust,
        } => {
            let set = read_points(io::stdin().lock(), high, weighted)?;
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
                Box::new(Stretch::from_refs(&a, &b, alpha)?)
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
            fun_hd,
            fun_ld,
            imix,
            grid,
            refine,
            print_error,
            path,
            gtemp,
        } => {
            let hi = read_points(
                std::io::BufReader::new(std::fs::File::open(&high_file)?),
                high,
                weighted,
            )?;
            let lo = read_points(
                std::io::BufReader::new(std::fs::File::open(&low_file)?),
                low,
                false,
            )?;
            let t_hd = Transfer::from_cli(&fun_hd)?;
            let t_ld = Transfer::from_cli(&fun_ld)?;
            let metric: Box<dyn Metric> = if dot {
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
                writeln!(
                    out,
                    "# q25 {:.8} q50 {:.8} q75 {:.8}",
                    s.q25, s.q50, s.q75
                )?;
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
    }
    Ok(())
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
