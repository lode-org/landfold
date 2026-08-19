//! landfold CLI: embed, project, landmarks, dist, fes.

use std::io::{self, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use ndarray::Array1;
use landfold::{
    apply_transfer, coordination_histogram, embed, farthest_point, pairwise, pairwise_euclid,
    project_many, read_points, write_points, Embedding, Euclid, FreeEnergy, Histogram2d, IterOpts,
    Metric, Periodic, ProjOpts, Solver, Sphere, StochOpts, Transfer,
};

#[derive(Parser, Debug)]
#[command(name = "landfold", about = "Sigmoid-distance nonlinear embedding")]
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
    },
    /// Project new high-D rows into a fitted embedding
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
        #[arg(long = "fun-hd", default_value = "identity")]
        fun_hd: String,
        #[arg(long = "fun-ld", default_value = "identity")]
        fun_ld: String,
        #[arg(long = "imix", default_value_t = 0.0)]
        imix: f64,
        #[arg(long = "grid", default_value = "1.0,21,201")]
        grid: String,
        #[arg(long = "refine", default_value_t = 0)]
        refine: usize,
    },
    /// Farthest-point landmarks
    Landmarks {
        #[arg(short = 'D', default_value_t = 3)]
        high: usize,
        #[arg(short = 'n', default_value_t = 100)]
        nland: usize,
        #[arg(long = "pi", default_value_t = 0.0)]
        period: f64,
        #[arg(short = 'w')]
        weighted: bool,
        #[arg(long, default_value_t = 0)]
        seed: usize,
    },
    /// Pairwise distance matrix
    Dist {
        #[arg(short = 'D', default_value_t = 3)]
        high: usize,
        #[arg(long = "pi", default_value_t = 0.0)]
        period: f64,
    },
    /// 2-D free-energy surface from embedded coords
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
        csv: Option<PathBuf>,
        #[arg(long)]
        svg: Option<PathBuf>,
        #[arg(long)]
        frames: Option<PathBuf>,
        #[arg(long, default_value_t = 1.2)]
        cn_cutoff: f64,
        #[arg(long, default_value_t = 12)]
        cn_max: usize,
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
            center,
            similarity,
            fun_hd,
            fun_ld,
            imix,
            steps,
            init,
            stoch,
            batch,
        } => {
            let set = read_points(io::stdin().lock(), high, weighted)?;
            let mut opts = IterOpts {
                lowdim: low,
                imix,
                tfun_hd: Transfer::from_cli(&fun_hd)?,
                tfun_ld: Transfer::from_cli(&fun_ld)?,
                center,
                solver: if stoch {
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
            let euclid = Euclid;
            let peri = Periodic::isotropic(high, period);
            let sph = Sphere::new(vec![sphere; high]);
            let metric: &dyn Metric = if dot {
                &landfold::Dot
            } else if sphere != 0.0 {
                &sph
            } else if period != 0.0 {
                &peri
            } else {
                &euclid
            };
            let pre = if similarity {
                Some(set.points.view())
            } else {
                None
            };
            let (emb, _) = embed(
                set.points.view(),
                metric,
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
            fun_hd,
            fun_ld,
            imix,
            grid,
            refine,
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
            if hi.points.nrows() != lo.points.nrows() {
                return Err(landfold::LandfoldError::Shape(
                    "landmark HD/LD count mismatch",
                ));
            }
            let t_hd = Transfer::from_cli(&fun_hd)?;
            let t_ld = Transfer::from_cli(&fun_ld)?;
            let euclid = Euclid;
            let peri = Periodic::isotropic(high, period);
            let metric: &dyn Metric = if period != 0.0 { &peri } else { &euclid };
            let hd = pairwise(hi.points.view(), metric)?;
            let mut fhd = hd.clone();
            apply_transfer(&mut fhd, &t_hd);
            let n = hi.points.nrows();
            let emb = Embedding {
                high: hi.points,
                low: lo.points,
                weights: hi.weights.unwrap_or_else(|| Array1::ones(n)),
                stress: 0.0,
                hd,
                fhd,
                tfun_hd: t_hd,
                tfun_ld: t_ld,
                imix,
            };
            let mut po = ProjOpts::from_cli(&grid)?;
            po.cg_steps = refine;
            let q = read_points(io::stdin().lock(), high, false)?;
            let proj = project_many(&emb, q.points.view(), metric, &po)?;
            write_points(&mut io::stdout().lock(), &proj, None)?;
        }
        Cmd::Landmarks {
            high,
            nland,
            period,
            weighted,
            seed,
        } => {
            let set = read_points(io::stdin().lock(), high, weighted)?;
            let euclid = Euclid;
            let peri = Periodic::isotropic(high, period);
            let metric: &dyn Metric = if period != 0.0 { &peri } else { &euclid };
            let lm = farthest_point(
                set.points.view(),
                metric,
                nland,
                set.weights.as_ref().map(|w| w.view()),
                seed,
            )?;
            write_points(&mut io::stdout().lock(), &lm.points, Some(&lm.weights))?;
        }
        Cmd::Dist { high, period } => {
            let set = read_points(io::stdin().lock(), high, false)?;
            let d = if period == 0.0 {
                pairwise_euclid(set.points.view())?
            } else {
                pairwise(set.points.view(), &Periodic::isotropic(high, period))?
            };
            write_points(&mut io::stdout().lock(), &d, None)?;
        }
        Cmd::Fes {
            input,
            nx,
            ny,
            kt,
            csv,
            svg,
            frames,
            cn_cutoff,
            cn_max,
        } => {
            let set = if let Some(p) = input {
                read_points(std::io::BufReader::new(std::fs::File::open(p)?), 2, false)?
            } else {
                read_points(io::stdin().lock(), 2, false)?
            };
            let xs = set.points.column(0);
            let ys = set.points.column(1);
            let (xmin, xmax) = minmax(xs.iter().copied());
            let (ymin, ymax) = minmax(ys.iter().copied());
            let px = 0.05 * (xmax - xmin).max(1e-6);
            let py = 0.05 * (ymax - ymin).max(1e-6);
            let mut h = Histogram2d::new(xmin - px, xmax + px, nx, ymin - py, ymax + py, ny)?;
            h.add_points(set.points.view(), set.weights.as_ref().map(|w| w.view()));
            let fes = FreeEnergy::from_histogram(&h, kt);
            if let Some(p) = csv {
                fes.write_csv(&mut std::fs::File::create(p)?)?;
            } else {
                fes.write_csv(&mut io::stdout().lock())?;
            }
            if let Some(p) = svg {
                fes.write_svg(&mut std::fs::File::create(p)?, 800, 640)?;
            }
            if let Some(p) = frames {
                let fr = read_points(std::io::BufReader::new(std::fs::File::open(p)?), 3, false)?;
                let cn = coordination_histogram(fr.points.view(), cn_cutoff, cn_max);
                for i in 0..cn.counts.len() {
                    writeln!(io::stderr(), "# CN {} {}", i, cn.counts[i])?;
                }
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
