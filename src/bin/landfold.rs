//! landfold CLI: embed, project, landmarks, dist, mds, fes.
//!
//! `project` is a coarse-then-fine χ grid plus `--refine` steps
//! (Ceriotti, Tribello, Parrinello, *J. Chem. Theory Comput.* **9**,
//! 1521 (2013), <https://doi.org/10.1021/ct3010563>). `landmarks` is
//! Gonzalez farthest-point sampling. `fes` writes `F = -ln(rho/rhomax)`
//! as CSV/SVG and optional coordination histograms.

use std::io::{self, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use landfold::{
    AnnealOpts, Dot, Embedding, Euclid, FreeEnergy, Histogram2d, IterOpts, L1, MdsMode, Metric,
    Periodic, ProjOpts, ReplicaOpts, Solver, Sphere, StochOpts, Transfer, coordination_histogram,
    embed, farthest_point, farthest_point_ifirst, mds_from_points, pairwise, pairwise_euclid,
    project_many_report, read_points, write_points,
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
        #[arg(long = "fun-hd", default_value = "identity")]
        fun_hd: String,
        #[arg(long = "fun-ld", default_value = "identity")]
        fun_ld: String,
        #[arg(long = "imix", default_value_t = 0.0)]
        imix: f64,
        #[arg(long = "steps", default_value_t = 100)]
        steps: usize,
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
        /// Bound-constrained HiGHS sequential LP (needs --features highs)
        #[arg(long)]
        highs: bool,
        /// xtsci-optimize L-BFGS (extra arm; same ChiObjective)
        #[arg(long)]
        lbfgs: bool,
        /// Box bounds `lo,hi` for `--highs`
        #[arg(long = "box")]
        box_bounds: Option<String>,
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
        #[arg(short = 'w')]
        weighted: bool,
        #[arg(long)]
        dot: bool,
        #[arg(long = "fun-hd", default_value = "identity")]
        fun_hd: String,
        #[arg(long = "fun-ld", default_value = "identity")]
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
    },
    /// Farthest-point (Gonzalez) landmarks
    Landmarks {
        #[arg(short = 'D', default_value_t = 3)]
        high: usize,
        #[arg(short = 'n', default_value_t = 100)]
        nland: usize,
        #[arg(long = "pi", default_value_t = 0.0)]
        period: f64,
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
    },
    /// Pairwise distance matrix
    Dist {
        #[arg(short = 'D', default_value_t = 3)]
        high: usize,
        #[arg(long = "pi", default_value_t = 0.0)]
        period: f64,
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
    /// 2-D F = -ln(rho/rhomax) plus optional coordination histogram
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
            init,
            stoch,
            batch,
            anneal,
            replica,
            highs,
            lbfgs,
            box_bounds,
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
                            ..landfold::HighsOpts::default()
                        };
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
                        let _ = box_bounds;
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
            let init = if let Some(p) = init {
                Some(read_points(
                    std::io::BufReader::new(std::fs::File::open(p)?),
                    low,
                    false,
                )?)
            } else {
                None
            };
            let metric: Box<dyn Metric> = if l1 {
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
            let (emb, _) = embed(
                set.points.view(),
                metric.as_ref(),
                &opts,
                init.as_ref().map(|p| p.points.view()),
                set.weights.as_ref().map(|w| w.view()),
                pre,
            )?;
            write_points(&mut io::stdout().lock(), &emb.low, None)?;
            writeln!(io::stderr(), "# stress {}", emb.stress)?;
        }
        Cmd::Project {
            high,
            low,
            high_file,
            low_file,
            period,
            weighted,
            dot,
            fun_hd,
            fun_ld,
            imix,
            grid,
            refine,
            print_error,
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
            let q = read_points(io::stdin().lock(), high, false)?;
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
            dot,
            l1,
            weighted,
            seed,
            ifirst,
            indices,
            voronoi,
            wgamma,
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
            let mut lm = if ifirst > 0 {
                farthest_point_ifirst(
                    set.points.view(),
                    metric.as_ref(),
                    nland,
                    set.weights.as_ref().map(|w| w.view()),
                    ifirst,
                )?
            } else {
                farthest_point(
                    set.points.view(),
                    metric.as_ref(),
                    nland,
                    set.weights.as_ref().map(|w| w.view()),
                    seed,
                )?
            };
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
        Cmd::Dist { high, period } => {
            let set = read_points(io::stdin().lock(), high, false)?;
            let d = if period == 0.0 {
                pairwise_euclid(set.points.view())?
            } else {
                pairwise(set.points.view(), &Periodic::isotropic(high, period)?)?
            };
            write_points(&mut io::stdout().lock(), &d, None)?;
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
