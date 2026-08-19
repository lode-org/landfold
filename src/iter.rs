//! Embed: MDS init, then CG, pair search, or annealing.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::anneal::{minimize_anneal, AnnealOpts};
use crate::cg::{minimize, CgOpts, CgReport};
use crate::error::Result;
use crate::mds::classical_mds;
use crate::metric::Metric;
use crate::pairwise::{apply_transfer, pairwise, pairwise_euclid};
use crate::search::{minimize_stochastic, StochOpts};
use crate::stress::Stress;
use crate::transfer::Transfer;

/// Solver arm. `Standard` is the published full-pair CG path.
#[derive(Clone, Debug)]
pub enum Solver {
    Standard,
    Stochastic(StochOpts),
    Anneal(AnnealOpts),
}

#[derive(Clone, Debug)]
pub struct IterOpts {
    pub lowdim: usize,
    pub imix: f64,
    pub tfun_hd: Transfer,
    pub tfun_ld: Transfer,
    pub cg: CgOpts,
    pub solver: Solver,
    pub center: bool,
}

impl Default for IterOpts {
    fn default() -> Self {
        Self {
            lowdim: 2,
            imix: 0.0,
            tfun_hd: Transfer::identity(),
            tfun_ld: Transfer::identity(),
            cg: CgOpts::default(),
            solver: Solver::Standard,
            center: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Embedding {
    pub high: Array2<f64>,
    pub low: Array2<f64>,
    pub weights: Array1<f64>,
    pub stress: f64,
    pub hd: Array2<f64>,
    pub fhd: Array2<f64>,
    pub tfun_hd: Transfer,
    pub tfun_ld: Transfer,
    pub imix: f64,
}

impl Embedding {
    pub fn packed_low(&self) -> Array1<f64> {
        Array1::from_iter(self.low.iter().copied())
    }
}

pub fn embed(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    opts: &IterOpts,
    init: Option<ArrayView2<f64>>,
    weights: Option<ArrayView1<f64>>,
    precomputed_dist: Option<ArrayView2<f64>>,
) -> Result<(Embedding, CgReport)> {
    let n = points.nrows();
    let hd = match precomputed_dist {
        Some(d) => d.to_owned(),
        None if metric.is_euclid() => pairwise_euclid(points)?,
        None => pairwise(points, metric)?,
    };
    let mut fhd = hd.clone();
    apply_transfer(&mut fhd, &opts.tfun_hd);

    let mut low = if let Some(p) = init {
        if p.nrows() != n || p.ncols() != opts.lowdim {
            return Err(crate::error::LandfoldError::Shape(
                "init embedding shape must be n x lowdim",
            ));
        }
        p.to_owned()
    } else {
        classical_mds(hd.view(), opts.lowdim)?.0
    };
    if opts.center {
        center_in_place(&mut low, weights);
    }

    let w1 = weights.map(|w| w.to_owned());
    let stress = Stress::new(
        hd.clone(),
        fhd.clone(),
        opts.tfun_ld.clone(),
        opts.imix,
        w1.clone(),
        None,
    );
    let packed = Array1::from_iter(low.iter().copied());
    let report = match &opts.solver {
        Solver::Standard => minimize(&stress, packed.view(), opts.lowdim, &opts.cg)?,
        Solver::Stochastic(so) => minimize_stochastic(&stress, packed.view(), opts.lowdim, so),
        Solver::Anneal(ao) => minimize_anneal(&stress, packed.view(), opts.lowdim, ao, &opts.cg)?,
    };
    let mut out = Array2::<f64>::zeros((n, opts.lowdim));
    for i in 0..n {
        for h in 0..opts.lowdim {
            out[(i, h)] = report.coords[i * opts.lowdim + h];
        }
    }
    if opts.center {
        center_in_place(&mut out, weights);
    }
    let weights = w1.unwrap_or_else(|| Array1::ones(n));
    Ok((
        Embedding {
            high: points.to_owned(),
            low: out,
            weights,
            stress: report.value,
            hd,
            fhd,
            tfun_hd: opts.tfun_hd.clone(),
            tfun_ld: opts.tfun_ld.clone(),
            imix: opts.imix,
        },
        report,
    ))
}

pub fn embed_points(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    opts: &IterOpts,
) -> Result<Embedding> {
    Ok(embed(points, metric, opts, None, None, None)?.0)
}

fn center_in_place(low: &mut Array2<f64>, weights: Option<ArrayView1<f64>>) {
    let n = low.nrows();
    let d = low.ncols();
    let mut mass = 0.0;
    let mut com = vec![0.0; d];
    for i in 0..n {
        let w = weights.map(|ww| ww[i]).unwrap_or(1.0);
        mass += w;
        for h in 0..d {
            com[h] += w * low[(i, h)];
        }
    }
    if mass == 0.0 {
        return;
    }
    for h in 0..d {
        com[h] /= mass;
    }
    for i in 0..n {
        for h in 0..d {
            low[(i, h)] -= com[h];
        }
    }
}

/// Re-export so callers that only want classical MDS do not touch `mds`.
pub fn mds_init(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    lowdim: usize,
) -> Result<Array2<f64>> {
    let dist = pairwise(points, metric)?;
    Ok(classical_mds(dist.view(), lowdim)?.0)
}
