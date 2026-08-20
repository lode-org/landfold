//! Embed: MDS init, then CG, pair search, or annealing.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::anneal::{AnnealOpts, minimize_anneal};
use crate::cg::{CgOpts, CgReport, minimize};
use crate::error::Result;
use crate::mds::classical_mds;
use crate::metric::Metric;
use crate::pairwise::{apply_transfer, pairwise, pairwise_euclid};
use crate::replica::{ReplicaOpts, minimize_replica};
use crate::search::{StochOpts, minimize_stochastic};
use crate::stress::{Stress, validate_distance_matrix, validate_imix, validate_weights};
use crate::transfer::Transfer;

/// Solver arm. `Standard` is the published full-pair CG path.
#[derive(Clone, Debug)]
pub enum Solver {
    Standard,
    Stochastic(StochOpts),
    Anneal(AnnealOpts),
    Replica(ReplicaOpts),
    /// Bound-constrained sequential LP via HiGHS. Extra arm.
    #[cfg(feature = "highs")]
    Highs(crate::highs_slp::HighsOpts),
    /// xtsci-optimize (L-BFGS, BFGS, SR1, Adam, other NLCG). Extra arm.
    Xtsci(xtsci_optimize::Method),
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
    pub(crate) fn validate_state(&self) -> Result<()> {
        let n = self.high.nrows();
        if n == 0
            || self.low.nrows() != n
            || self.low.ncols() == 0
            || self.hd.nrows() != n
            || self.hd.ncols() != n
            || self.fhd.nrows() != n
            || self.fhd.ncols() != n
            || self.weights.len() != n
        {
            return Err(crate::error::LandfoldError::Shape(
                "invalid embedding array dimensions",
            ));
        }
        if self
            .high
            .iter()
            .chain(self.low.iter())
            .any(|&value| !value.is_finite())
            || self
                .hd
                .iter()
                .chain(self.fhd.iter())
                .any(|&value| !value.is_finite() || value < 0.0)
            || !self.stress.is_finite()
            || self.stress < 0.0
        {
            return Err(crate::error::LandfoldError::Msg(
                "embedding coordinates and distances must be finite; distances must be nonnegative"
                    .into(),
            ));
        }
        validate_weights(Some(self.weights.view()), n)?;
        validate_distance_matrix(&self.hd, "high-D")?;
        validate_distance_matrix(&self.fhd, "transformed high-D")?;
        validate_imix(self.imix)
    }

    pub fn packed_low(&self) -> Array1<f64> {
        Array1::from_iter(self.low.iter().copied())
    }

    /// Landmark table already fitted in low-D. Used by out-of-sample project.
    pub fn from_landmarks(
        high: Array2<f64>,
        low: Array2<f64>,
        metric: &dyn Metric,
        tfun_hd: Transfer,
        tfun_ld: Transfer,
        imix: f64,
        weights: Option<Array1<f64>>,
    ) -> Result<Self> {
        validate_imix(imix)?;
        if high.nrows() != low.nrows() {
            return Err(crate::error::LandfoldError::Shape(
                "landmark HD/LD count mismatch",
            ));
        }
        if high.nrows() == 0 {
            return Err(crate::error::LandfoldError::Empty);
        }
        if low.ncols() == 0 {
            return Err(crate::error::LandfoldError::LowDim {
                low: 0,
                high: high.nrows(),
            });
        }
        if low.iter().any(|&value| !value.is_finite()) {
            return Err(crate::error::LandfoldError::Msg(
                "landmark low-D coordinates must be finite".into(),
            ));
        }
        let n = high.nrows();
        validate_weights(weights.as_ref().map(|w| w.view()), n)?;
        let hd = if metric.is_euclid() {
            pairwise_euclid(high.view())?
        } else {
            pairwise(high.view(), metric)?
        };
        let mut fhd = hd.clone();
        apply_transfer(&mut fhd, &tfun_hd)?;
        Ok(Self {
            high,
            low,
            weights: weights.unwrap_or_else(|| Array1::ones(n)),
            stress: 0.0,
            hd,
            fhd,
            tfun_hd,
            tfun_ld,
            imix,
        })
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
    if opts.lowdim == 0 {
        return Err(crate::error::LandfoldError::LowDim { low: 0, high: n });
    }
    validate_weights(weights, n)?;
    let hd = match precomputed_dist {
        Some(d) => {
            if d.nrows() != n || d.ncols() != n {
                return Err(crate::error::LandfoldError::Shape(
                    "precomputed distance matrix must be n x n",
                ));
            }
            d.to_owned()
        }
        None if metric.is_euclid() => pairwise_euclid(points)?,
        None => pairwise(points, metric)?,
    };
    let mut fhd = hd.clone();
    apply_transfer(&mut fhd, &opts.tfun_hd)?;

    let mut low = if let Some(p) = init {
        if p.nrows() != n || p.ncols() != opts.lowdim {
            return Err(crate::error::LandfoldError::Shape(
                "init embedding shape must be n x lowdim",
            ));
        }
        if p.iter().any(|value| !value.is_finite()) {
            return Err(crate::error::LandfoldError::Msg(
                "init embedding coordinates must be finite".into(),
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
    let stress = Stress::try_new(
        hd.clone(),
        fhd.clone(),
        opts.tfun_ld.clone(),
        opts.imix,
        w1.clone(),
        None,
    )?;
    let packed = Array1::from_iter(low.iter().copied());
    let report = match &opts.solver {
        Solver::Standard => minimize(&stress, packed.view(), opts.lowdim, &opts.cg)?,
        Solver::Stochastic(so) => minimize_stochastic(&stress, packed.view(), opts.lowdim, so)?,
        Solver::Anneal(ao) => minimize_anneal(&stress, packed.view(), opts.lowdim, ao, &opts.cg)?,
        Solver::Replica(ro) => minimize_replica(&stress, packed.view(), opts.lowdim, ro, &opts.cg)?,
        #[cfg(feature = "highs")]
        Solver::Highs(ho) => {
            crate::highs_slp::minimize_highs(&stress, packed.view(), opts.lowdim, ho)?
        }
        Solver::Xtsci(method) => crate::cg::minimize_xtsci(
            &stress,
            packed.view(),
            opts.lowdim,
            &opts.cg,
            method.clone(),
        )?,
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
    for value in com.iter_mut().take(d) {
        *value /= mass;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::Euclid;
    use ndarray::array;

    #[test]
    fn rejects_invalid_embedding_weights_before_centering() {
        let points = array![[0.0, 0.0], [1.0, 0.0]];
        let opts = IterOpts::default();
        assert!(
            embed(
                points.view(),
                &Euclid,
                &opts,
                None,
                Some(array![1.0].view()),
                None,
            )
            .is_err()
        );
        assert!(
            embed(
                points.view(),
                &Euclid,
                &opts,
                None,
                Some(array![1.0, -1.0].view()),
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_nonfinite_custom_initial_coordinates() {
        let points = array![[0.0, 0.0], [1.0, 0.0]];
        let init = array![[0.0, 0.0], [f64::NAN, 1.0]];
        assert!(
            embed(
                points.view(),
                &Euclid,
                &IterOpts::default(),
                Some(init.view()),
                None,
                None,
            )
            .is_err()
        );
    }
}
