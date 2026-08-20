//! Embed: MDS init, then CG, pair search, or annealing.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::anneal::{AnnealOpts, minimize_anneal};
use crate::cg::{CgOpts, CgReport, minimize};
use crate::error::Result;
use crate::mds::classical_mds;
use crate::metric::Metric;
use crate::pairwise::{apply_transfer, pairwise, pairwise_euclid};
use crate::project::{scan_grid_1d, scan_grid_2d, ProjOpts};
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
    /// First CG phase (`dimred -preopt`). Zero uses `cg.maxiter` when there
    /// is no `-grid` phase (the single-loop shortcut).
    pub preopt: usize,
    /// CG steps after each successful pointwise move (`dimred -gopt`).
    pub gopt: usize,
    /// Pointwise global grid (`dimred -grid`). `None` skips that phase.
    pub global: Option<ProjOpts>,
    /// Pair weights `F(D)(1-F(D))` so only mid-scale pairs drive χ.
    pub midweight: bool,
    /// Explicit pair weights. When set, they replace `--midweight`.
    pub pair_weights: Option<Array2<f64>>,
    /// Classical MDS of `F(D)` rather than `D`.
    pub init_transformed: bool,
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
            preopt: 0,
            gopt: 0,
            global: None,
            midweight: false,
            pair_weights: None,
            init_transformed: false,
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
    } else if opts.init_transformed {
        classical_mds(fhd.view(), opts.lowdim)?.0
    } else {
        classical_mds(hd.view(), opts.lowdim)?.0
    };
    if opts.center {
        center_in_place(&mut low, weights)?;
    }

    let w1 = weights.map(|w| w.to_owned());
    let pair_w = if let Some(pw) = &opts.pair_weights {
        Some(pw.clone())
    } else if opts.midweight {
        Some(Stress::midscale_pair_weights(fhd.view())?)
    } else {
        None
    };
    let stress = Stress::try_new(
        hd.clone(),
        fhd.clone(),
        opts.tfun_ld.clone(),
        opts.imix,
        w1.clone(),
        pair_w,
    )?;
    let first_steps = if opts.preopt > 0 {
        opts.preopt
    } else if opts.global.is_none() {
        opts.cg.maxiter
    } else {
        0
    };
    let mut report = CgReport {
        coords: Array1::from_iter(low.iter().copied()),
        value: 0.0,
        steps: 0,
    };
    if first_steps > 0 {
        let mut first_cg = opts.cg.clone();
        first_cg.maxiter = first_steps;
        report = run_solver(&stress, report.coords.view(), opts, &first_cg)?;
        unpack_low(&report.coords, n, opts.lowdim, &mut low);
    }
    if let Some(grid) = &opts.global {
        report = pointwise_global(&stress, &mut low, w1.as_ref(), grid, opts)?;
    }
    let mut out = low;
    if opts.center {
        center_in_place(&mut out, weights)?;
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

/// Ceriotti χ at decreasing σ, warm-started. Large σ is almost MDS;
/// the last σ is the published scale.
pub fn embed_sigma_schedule(
    points: ArrayView2<f64>,
    metric: &dyn Metric,
    opts: &IterOpts,
    sigmas: &[f64],
    init: Option<ArrayView2<f64>>,
    weights: Option<ArrayView1<f64>>,
    precomputed_dist: Option<ArrayView2<f64>>,
) -> Result<(Embedding, CgReport)> {
    if sigmas.is_empty() {
        return Err(crate::error::LandfoldError::Msg(
            "sigma schedule needs at least one scale".into(),
        ));
    }
    if sigmas.iter().any(|s| !s.is_finite() || *s <= 0.0) {
        return Err(crate::error::LandfoldError::Msg(
            "sigma schedule entries must be finite and > 0".into(),
        ));
    }
    let (sigma0, a, b) = opts.tfun_hd.xsigmoid_params().ok_or_else(|| {
        crate::error::LandfoldError::Msg(
            "sigma schedule needs a Ceriotti xsigmoid --fun-hd".into(),
        )
    })?;
    let _ = sigma0;
    let ld_ab = opts.tfun_ld.xsigmoid_params();
    let mut current_init = init.map(|p| p.to_owned());
    let mut last = None;
    for (k, &sigma) in sigmas.iter().enumerate() {
        let mut stage = opts.clone();
        stage.tfun_hd = Transfer::xsigmoid(sigma, a, b)?;
        if let Some((_, la, lb)) = ld_ab {
            stage.tfun_ld = Transfer::xsigmoid(sigma, la, lb)?;
        }
        if k + 1 < sigmas.len() {
            stage.global = None;
            stage.preopt = 0;
        }
        let (emb, report) = embed(
            points,
            metric,
            &stage,
            current_init.as_ref().map(|p| p.view()),
            weights,
            precomputed_dist,
        )?;
        current_init = Some(emb.low.clone());
        last = Some((emb, report));
    }
    last.ok_or_else(|| crate::error::LandfoldError::Msg("empty sigma schedule".into()))
}

fn run_solver(
    stress: &Stress,
    packed: ArrayView1<f64>,
    opts: &IterOpts,
    cg: &CgOpts,
) -> Result<CgReport> {
    match &opts.solver {
        Solver::Standard => minimize(stress, packed, opts.lowdim, cg),
        Solver::Stochastic(so) => minimize_stochastic(stress, packed, opts.lowdim, so),
        Solver::Anneal(ao) => minimize_anneal(stress, packed, opts.lowdim, ao, cg),
        Solver::Replica(ro) => minimize_replica(stress, packed, opts.lowdim, ro, cg),
        #[cfg(feature = "highs")]
        Solver::Highs(ho) => crate::highs_slp::minimize_highs(stress, packed, opts.lowdim, ho),
        Solver::Xtsci(method) => {
            crate::cg::minimize_xtsci(stress, packed, opts.lowdim, cg, method.clone())
        }
    }
}

fn unpack_low(coords: &Array1<f64>, n: usize, d: usize, low: &mut Array2<f64>) {
    for i in 0..n {
        for h in 0..d {
            low[(i, h)] = coords[i * d + h];
        }
    }
}

fn pack_low(low: &Array2<f64>) -> Array1<f64> {
    Array1::from_iter(low.iter().copied())
}

/// C++ `dimred -grid` / `-gopt`: one-point χ grid, then full-pair CG
/// after each move that actually improves χ.
fn pointwise_global(
    stress: &Stress,
    low: &mut Array2<f64>,
    weights: Option<&Array1<f64>>,
    grid: &ProjOpts,
    opts: &IterOpts,
) -> Result<CgReport> {
    grid.validate()?;
    let n = low.nrows();
    let d = low.ncols();
    let ones = Array1::ones(n);
    let w = weights.unwrap_or(&ones);
    if opts.gopt > 0 {
        let mut gcg = opts.cg.clone();
        gcg.maxiter = opts.gopt;
        let report = run_solver(stress, pack_low(low).view(), opts, &gcg)?;
        unpack_low(&report.coords, n, d, low);
    }
    for ip in 0..n {
        let (best, best_f) = grid_min_chi1(stress, low.view(), ip, w.view(), grid)?;
        let current = low.row(ip).to_owned();
        let (init_f, _) = stress.chi1_checked(current.view(), low.view(), ip, w.view())?;
        if best_f < init_f {
            for h in 0..d {
                low[(ip, h)] = best[h];
            }
            if opts.gopt > 0 {
                let mut gcg = opts.cg.clone();
                gcg.maxiter = opts.gopt;
                let report = run_solver(stress, pack_low(low).view(), opts, &gcg)?;
                unpack_low(&report.coords, n, d, low);
            }
        }
    }
    let packed = pack_low(low);
    let ev = stress.eval(packed.view(), d);
    Ok(CgReport {
        coords: packed,
        value: ev.value,
        steps: opts.gopt,
    })
}

fn grid_min_chi1(
    stress: &Stress,
    low: ArrayView2<f64>,
    skip: usize,
    weights: ArrayView1<f64>,
    opts: &ProjOpts,
) -> Result<(Array1<f64>, f64)> {
    let d = low.ncols();
    let mut best = low.row(skip).to_owned();
    let (mut best_f, _) = stress.chi1_checked(best.view(), low, skip, weights)?;
    let eval = |q: ArrayView1<f64>| stress.chi1(q, low, skip, weights).0;
    if d == 2 && opts.grid_coarse >= 2 {
        let w = opts.gridw;
        let g1 = opts.grid_coarse;
        scan_grid_2d(-w, w, -w, w, g1, |q| {
            let f = eval(q.view());
            if f < best_f {
                best_f = f;
                best = q;
            }
        });
        let g2 = opts.grid_fine.max(2);
        let span = 2.0 * w / (g1 as f64);
        let cx = best[0];
        let cy = best[1];
        scan_grid_2d(cx - span, cx + span, cy - span, cy + span, g2, |q| {
            let f = eval(q.view());
            if f < best_f {
                best_f = f;
                best = q;
            }
        });
    } else if d == 1 && opts.grid_coarse >= 2 {
        let w = opts.gridw;
        let g1 = opts.grid_coarse;
        scan_grid_1d(-w, w, g1, |q| {
            let f = eval(q.view());
            if f < best_f {
                best_f = f;
                best = q;
            }
        });
        let g2 = opts.grid_fine.max(2);
        let span = 2.0 * w / (g1 as f64);
        let cx = best[0];
        scan_grid_1d(cx - span, cx + span, g2, |q| {
            let f = eval(q.view());
            if f < best_f {
                best_f = f;
                best = q;
            }
        });
    }
    Ok((best, best_f))
}

fn center_in_place(
    low: &mut Array2<f64>,
    weights: Option<ArrayView1<f64>>,
) -> crate::error::Result<()> {
    let n = low.nrows();
    let d = low.ncols();
    let mut mass = 0.0;
    let mut com = vec![0.0; d];
    for i in 0..n {
        let w = weights.map(|ww| ww[i]).unwrap_or(1.0);
        mass += w;
        if !mass.is_finite() {
            return Err(crate::error::LandfoldError::Msg(
                "embedding center-of-mass mass overflowed".into(),
            ));
        }
        for h in 0..d {
            com[h] += w * low[(i, h)];
            if !com[h].is_finite() {
                return Err(crate::error::LandfoldError::Msg(
                    "embedding center-of-mass coordinate overflowed".into(),
                ));
            }
        }
    }
    if mass == 0.0 {
        return Ok(());
    }
    for value in com.iter_mut().take(d) {
        *value /= mass;
        if !value.is_finite() {
            return Err(crate::error::LandfoldError::Msg(
                "embedding center-of-mass coordinate became non-finite".into(),
            ));
        }
    }
    for i in 0..n {
        for h in 0..d {
            low[(i, h)] -= com[h];
            if !low[(i, h)].is_finite() {
                return Err(crate::error::LandfoldError::Msg(
                    "centered embedding coordinate became non-finite".into(),
                ));
            }
        }
    }
    Ok(())
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

    #[test]
    fn rejects_overflowing_embedding_center() {
        let mut low = array![[f64::MAX], [f64::MAX]];
        assert!(center_in_place(&mut low, None).is_err());
    }

    #[test]
    fn preopt_then_grid_returns_finite_stress() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let opts = IterOpts {
            lowdim: 2,
            preopt: 4,
            gopt: 2,
            global: Some(crate::ProjOpts {
                gridw: 2.0,
                grid_coarse: 5,
                grid_fine: 9,
                cg_steps: 0,
                ..crate::ProjOpts::default()
            }),
            ..IterOpts::default()
        };
        let (emb, _) = embed(points.view(), &Euclid, &opts, None, None, None).unwrap();
        assert!(emb.stress.is_finite());
        assert_eq!(emb.low.nrows(), 4);
        assert_eq!(emb.low.ncols(), 2);
    }

    #[test]
    fn midweight_and_sigma_schedule_return_finite_maps() {
        let points = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [8.0, 0.0],
            [8.1, 0.0],
            [8.0, 0.1],
        ];
        let mut opts = IterOpts {
            lowdim: 2,
            tfun_hd: Transfer::xsigmoid(3.0, 4.0, 2.0).unwrap(),
            tfun_ld: Transfer::xsigmoid(3.0, 2.0, 2.0).unwrap(),
            midweight: true,
            init_transformed: true,
            ..IterOpts::default()
        };
        opts.cg.maxiter = 8;
        let (emb, _) = embed(points.view(), &Euclid, &opts, None, None, None).unwrap();
        assert!(emb.stress.is_finite());
        let (emb2, _) = embed_sigma_schedule(
            points.view(),
            &Euclid,
            &opts,
            &[6.0, 3.0],
            None,
            None,
            None,
        )
        .unwrap();
        assert!(emb2.stress.is_finite());
        assert_eq!(emb2.low.nrows(), 6);
    }
}
