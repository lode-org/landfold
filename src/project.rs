//! Out-of-sample placement of new high-D points.
//!
//! A new point is dropped onto the low-D map by minimising its one-point χ
//! against the landmarks (Ceriotti, Tribello, Parrinello, *J. Chem. Theory
//! Comput.* **9**, 1521 (2013), <https://doi.org/10.1021/ct3010563>):
//!
//! 1. coarse grid of χ on `[-gridw, gridw]^d` (`d = 1` or `2`);
//! 2. local fine grid of one coarse cell around that min;
//! 3. optional Polak-Ribiere + Brent refine (`cg_steps`).
//!
//! Higher-D embeddings seed from the nearest landmark, then refine.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::cg::{CgOpts, try_minimize_oracle};
use crate::error::{LandfoldError, Result};
use crate::iter::Embedding;
use crate::metric::Metric;
use crate::stress::{query_chi, query_chi_checked};

#[derive(Clone, Debug)]
pub struct ProjOpts {
    pub gridw: f64,
    pub grid_coarse: usize,
    pub grid_fine: usize,
    pub cg_steps: usize,
    /// Query row is already distances to the n landmarks (`dimproj -similarity`).
    pub similarity: bool,
    /// Path-like average of landmark LD coords (`dimproj -path lambda`).
    pub path_lambda: f64,
    /// Softmax temperature on the fine grid (`dimproj -gt`). Zero keeps the min.
    pub gtemp: f64,
}

impl Default for ProjOpts {
    fn default() -> Self {
        Self {
            gridw: 1.0,
            grid_coarse: 21,
            grid_fine: 201,
            cg_steps: 0,
            similarity: false,
            path_lambda: -1.0,
            gtemp: 0.0,
        }
    }
}

impl ProjOpts {
    pub(crate) fn validate(&self) -> Result<()> {
        if !self.gridw.is_finite() || self.gridw <= 0.0 {
            return Err(LandfoldError::Msg(
                "projection grid width must be finite and > 0".into(),
            ));
        }
        if self.grid_coarse == 0 || self.grid_fine == 0 {
            return Err(LandfoldError::Msg(
                "projection grid sizes must be > 0".into(),
            ));
        }
        if !self.gtemp.is_finite() || self.gtemp < 0.0 {
            return Err(LandfoldError::Msg(
                "projection gtemp must be finite and nonnegative".into(),
            ));
        }
        if !self.path_lambda.is_finite() {
            return Err(LandfoldError::Msg(
                "projection path lambda must be finite".into(),
            ));
        }
        Ok(())
    }

    pub fn from_cli(spec: &str) -> Result<Self> {
        let parts: Vec<f64> = spec
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<f64>()
                    .map_err(|e| LandfoldError::Parse(format!("grid `{spec}`: {e}")))
            })
            .collect::<Result<Vec<_>>>()?;
        match parts.as_slice() {
            [w, g1, g2] => {
                if !w.is_finite() || *w <= 0.0 {
                    return Err(LandfoldError::Parse("-grid width must be > 0".into()));
                }
                if !g1.is_finite() || *g1 < 1.0 || g1.fract() != 0.0 {
                    return Err(LandfoldError::Parse("-grid g1 must be >= 1".into()));
                }
                if !g2.is_finite() || *g2 < 1.0 || g2.fract() != 0.0 {
                    return Err(LandfoldError::Parse("-grid g2 must be >= 1".into()));
                }
                let usize_limit = usize::MAX as f64;
                if *g1 >= usize_limit || *g2 >= usize_limit {
                    return Err(LandfoldError::Parse(
                        "-grid counts exceed the platform usize range".into(),
                    ));
                }
                Ok(Self {
                    gridw: *w,
                    grid_coarse: *g1 as usize,
                    grid_fine: (*g2 as usize).max(1),
                    cg_steps: 0,
                    similarity: false,
                    path_lambda: -1.0,
                    gtemp: 0.0,
                })
            }
            _ => Err(LandfoldError::Parse("-grid needs gw,g1,g2".into())),
        }
    }
}

/// Low-D placement of one query plus the χ and nearest-landmark HD distance.
#[derive(Clone, Debug)]
pub struct ProjReport {
    pub coords: Array1<f64>,
    pub chi: f64,
    pub nearest: f64,
    pub nearest_idx: usize,
}

/// Project one high-D query onto the landmark embedding.
pub fn project_one(
    emb: &Embedding,
    query: ArrayView1<f64>,
    metric: &dyn Metric,
    opts: &ProjOpts,
) -> Result<Array1<f64>> {
    Ok(project_report(emb, query, metric, opts)?.coords)
}

/// Grid + local refine, then the one-point χ and nearest-landmark HD distance.
pub fn project_report(
    emb: &Embedding,
    query: ArrayView1<f64>,
    metric: &dyn Metric,
    opts: &ProjOpts,
) -> Result<ProjReport> {
    emb.validate_state()?;
    opts.validate()?;
    let n = emb.high.nrows();
    let d_hi = emb.high.ncols();
    if n == 0 {
        return Err(LandfoldError::Empty);
    }
    if opts.similarity {
        if query.len() != n {
            return Err(LandfoldError::MetricSize {
                left: query.len(),
                right: n,
            });
        }
    } else if query.len() != d_hi {
        return Err(LandfoldError::MetricSize {
            left: query.len(),
            right: d_hi,
        });
    }
    let qslice: Vec<f64> = query.iter().copied().collect();
    let mut hd_row = Array1::<f64>::zeros(n);
    let mut landmark = vec![0.0; d_hi];
    let mut nearest = f64::INFINITY;
    let mut nearest_idx = 0usize;
    for i in 0..n {
        let d = if opts.similarity {
            query[i]
        } else {
            for (h, value) in landmark.iter_mut().enumerate().take(d_hi) {
                *value = emb.high[(i, h)];
            }
            metric.dist(&qslice, &landmark)?
        };
        if !d.is_finite() || d < 0.0 {
            return Err(LandfoldError::Msg(
                "projection distances must be finite and nonnegative".into(),
            ));
        }
        hd_row[i] = d;
        if d < nearest {
            nearest = d;
            nearest_idx = i;
        }
    }
    let mut fhd_row = Array1::<f64>::zeros(n);
    for i in 0..n {
        fhd_row[i] = emb.tfun_hd.try_fdf(hd_row[i])?.0;
    }
    let d = emb.low.ncols();
    if opts.path_lambda > 0.0 {
        return path_average(emb, hd_row.view(), nearest, nearest_idx, opts.path_lambda);
    }
    let mut best = emb.low.row(nearest_idx).to_owned();
    if nearest <= 1e-14 {
        let (chi, _) = query_chi(
            best.view(),
            emb.low.view(),
            hd_row.view(),
            fhd_row.view(),
            &emb.tfun_ld,
            emb.imix,
            emb.weights.view(),
        );
        return Ok(ProjReport {
            coords: best,
            chi,
            nearest,
            nearest_idx,
        });
    }
    let eval = |x: ArrayView1<f64>| {
        query_chi(
            x,
            emb.low.view(),
            hd_row.view(),
            fhd_row.view(),
            &emb.tfun_ld,
            emb.imix,
            emb.weights.view(),
        )
    };
    // Nearest landmark (and every other landmark) is a candidate.
    // A coarse grid on [-w,w] can miss the cloud and must not discard it.
    let (mut best_f, _) = eval(best.view());
    for i in 0..n {
        if i == nearest_idx {
            continue;
        }
        let cand = emb.low.row(i).to_owned();
        let (f, _) = eval(cand.view());
        if f < best_f {
            best_f = f;
            best = cand;
        }
    }

    if d == 2 && opts.grid_coarse >= 2 {
        let w = opts.gridw;
        let g1 = opts.grid_coarse;
        scan_grid_2d(-w, w, -w, w, g1, |q| {
            let (f, _) = eval(q.view());
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
            let (f, _) = eval(q.view());
            if f < best_f {
                best_f = f;
                best = q;
            }
        });
    } else if d == 1 && opts.grid_coarse >= 2 {
        let w = opts.gridw;
        let g1 = opts.grid_coarse;
        scan_grid_1d(-w, w, g1, |q| {
            let (f, _) = eval(q.view());
            if f < best_f {
                best_f = f;
                best = q;
            }
        });
        let g2 = opts.grid_fine.max(2);
        let span = 2.0 * w / (g1 as f64);
        let cx = best[0];
        scan_grid_1d(cx - span, cx + span, g2, |q| {
            let (f, _) = eval(q.view());
            if f < best_f {
                best_f = f;
                best = q;
            }
        });
    }

    if opts.gtemp > 0.0 && d == 2 && opts.grid_coarse >= 2 {
        let w = opts.gridw;
        let g1 = opts.grid_coarse;
        let mut tw = 0.0;
        let mut acc = Array1::<f64>::zeros(d);
        let reff = best_f_after_grid(&eval, best.view());
        scan_grid_2d(-w, w, -w, w, g1, |q| {
            let f = eval(q.view()).0;
            let ww = ((reff - f) / opts.gtemp).exp();
            if ww.is_finite() {
                tw += ww;
                acc[0] += ww * q[0];
                acc[1] += ww * q[1];
            }
        });
        if tw > 0.0 && tw.is_finite() {
            best[0] = acc[0] / tw;
            best[1] = acc[1] / tw;
        }
    }

    if opts.cg_steps > 0 {
        let cgopts = CgOpts {
            maxiter: opts.cg_steps,
            ls_maxiter: 4,
            ls_tol: 1e-9,
            ..CgOpts::default()
        };
        let report = try_minimize_oracle(|x| eval(x), best.view(), &cgopts)?;
        best = report.coords;
    }

    let (chi, _) = query_chi_checked(
        best.view(),
        emb.low.view(),
        hd_row.view(),
        fhd_row.view(),
        &emb.tfun_ld,
        emb.imix,
        emb.weights.view(),
    )?;
    Ok(ProjReport {
        coords: best,
        chi,
        nearest,
        nearest_idx,
    })
}

pub fn project_many(
    emb: &Embedding,
    queries: ArrayView2<f64>,
    metric: &dyn Metric,
    opts: &ProjOpts,
) -> Result<Array2<f64>> {
    let reports = project_many_report(emb, queries, metric, opts)?;
    let d = emb.low.ncols();
    let mut out = Array2::<f64>::zeros((reports.len(), d));
    for (i, r) in reports.iter().enumerate() {
        for h in 0..d {
            out[(i, h)] = r.coords[h];
        }
    }
    Ok(out)
}

pub fn project_many_report(
    emb: &Embedding,
    queries: ArrayView2<f64>,
    metric: &dyn Metric,
    opts: &ProjOpts,
) -> Result<Vec<ProjReport>> {
    let nq = queries.nrows();
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..nq)
            .into_par_iter()
            .map(|i| project_report(emb, queries.row(i), metric, opts))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut out = Vec::with_capacity(nq);
        for i in 0..nq {
            out.push(project_report(emb, queries.row(i), metric, opts)?);
        }
        Ok(out)
    }
}

fn path_average(
    emb: &Embedding,
    hd_row: ArrayView1<f64>,
    nearest: f64,
    nearest_idx: usize,
    lambda: f64,
) -> Result<ProjReport> {
    let d = emb.low.ncols();
    let n = emb.low.nrows();
    let mut acc = Array1::<f64>::zeros(d);
    let mut tw = 0.0;
    for i in 0..n {
        let w = (-hd_row[i] / lambda).exp() * emb.weights[i];
        if !w.is_finite() {
            return Err(LandfoldError::Msg("path-average weight overflowed".into()));
        }
        tw += w;
        for h in 0..d {
            acc[h] += w * emb.low[(i, h)];
        }
    }
    if !(tw > 0.0 && tw.is_finite()) {
        return Err(LandfoldError::Msg("path-average has no positive mass".into()));
    }
    for h in 0..d {
        acc[h] /= tw;
        if !acc[h].is_finite() {
            return Err(LandfoldError::Msg(
                "path-average coordinate became non-finite".into(),
            ));
        }
    }
    Ok(ProjReport {
        coords: acc,
        chi: 0.0,
        nearest,
        nearest_idx,
    })
}

fn best_f_after_grid(
    eval: &impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>),
    best: ArrayView1<f64>,
) -> f64 {
    eval(best).0
}

pub(crate) fn scan_grid_2d(
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    n: usize,
    mut visit: impl FnMut(Array1<f64>),
) {
    let denom = (n.saturating_sub(1) as f64).max(1.0);
    for i in 0..n {
        let x = x0 + (x1 - x0) * (i as f64) / denom;
        for j in 0..n {
            let y = y0 + (y1 - y0) * (j as f64) / denom;
            visit(Array1::from(vec![x, y]));
        }
    }
}

pub(crate) fn scan_grid_1d(x0: f64, x1: f64, n: usize, mut visit: impl FnMut(Array1<f64>)) {
    let denom = (n.saturating_sub(1) as f64).max(1.0);
    for i in 0..n {
        let x = x0 + (x1 - x0) * (i as f64) / denom;
        visit(Array1::from(vec![x]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::Euclid;
    use crate::transfer::Transfer;
    use ndarray::array;

    #[test]
    fn from_cli_rejects_bad_spec() {
        assert!(ProjOpts::from_cli("1.0,21").is_err());
        assert!(ProjOpts::from_cli("0.0,21,201").is_err());
        assert!(ProjOpts::from_cli("1.0,1.5,201").is_err());
        assert!(ProjOpts::from_cli("1.0,21,0").is_err());
        assert!(ProjOpts::from_cli("1.0,NaN,201").is_err());
        assert!(ProjOpts::from_cli("1.0,18446744073709551616,2").is_err());
        let p = ProjOpts::from_cli("2.5,11,41").unwrap();
        assert_eq!(p.grid_coarse, 11);
        assert!((p.gridw - 2.5).abs() < 1e-15);
    }

    #[test]
    fn rejects_invalid_direct_projection_options() {
        let opts = ProjOpts {
            gridw: f64::NAN,
            ..ProjOpts::default()
        };
        assert!(opts.validate().is_err());
        let opts = ProjOpts {
            grid_coarse: 0,
            ..ProjOpts::default()
        };
        assert!(opts.validate().is_err());
    }

    #[test]
    fn landmark_query_beats_a_coarse_grid() {
        let high = array![[0.0, 0.0], [0.4, 0.0], [0.0, 0.4]];
        let low = array![[0.15, 0.15], [0.55, 0.10], [0.10, 0.55]];
        let emb = Embedding::from_landmarks(
            high.clone(),
            low.clone(),
            &Euclid,
            Transfer::identity(),
            Transfer::identity(),
            0.0,
            None,
        )
        .unwrap();
        let opts = ProjOpts {
            gridw: 2.0,
            grid_coarse: 3,
            grid_fine: 3,
            cg_steps: 0,
            ..ProjOpts::default()
        };
        let r = project_report(&emb, high.row(0), &Euclid, &opts).unwrap();
        assert_eq!(r.nearest_idx, 0);
        let err = (r.coords[0] - low[(0, 0)]).hypot(r.coords[1] - low[(0, 1)]);
        assert!(
            err < 1e-12,
            "landmark query must stay on the landmark, got {err} at {:?}",
            r.coords
        );
    }

    #[test]
    fn nearest_seed_when_grid_off() {
        let high = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let low = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let emb = Embedding::from_landmarks(
            high,
            low,
            &Euclid,
            Transfer::identity(),
            Transfer::identity(),
            0.0,
            None,
        )
        .unwrap();
        let opts = ProjOpts {
            gridw: 1.0,
            grid_coarse: 1,
            grid_fine: 1,
            cg_steps: 0,
            ..ProjOpts::default()
        };
        let r = project_report(&emb, array![1.0, 0.0].view(), &Euclid, &opts).unwrap();
        assert_eq!(r.nearest_idx, 1);
        assert!((r.nearest).abs() < 1e-14);
        assert!((r.coords[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn rejects_mutated_embedding_state() {
        let high = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let low = high.clone();
        let mut emb = Embedding::from_landmarks(
            high,
            low,
            &Euclid,
            Transfer::identity(),
            Transfer::identity(),
            0.0,
            None,
        )
        .unwrap();
        let opts = ProjOpts {
            gridw: 1.0,
            grid_coarse: 1,
            grid_fine: 1,
            cg_steps: 0,
            ..ProjOpts::default()
        };
        emb.low = array![[0.0, 0.0], [1.0, 0.0]];
        assert!(project_report(&emb, array![0.0, 0.0].view(), &Euclid, &opts).is_err());

        emb.low = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        emb.weights = array![1.0, 1.0];
        assert!(project_report(&emb, array![0.0, 0.0].view(), &Euclid, &opts).is_err());

        emb.weights = array![1.0, 1.0, 1.0];
        emb.hd[(1, 0)] = 2.0;
        assert!(project_report(&emb, array![0.0, 0.0].view(), &Euclid, &opts).is_err());
    }

    #[test]
    fn rejects_overflowed_high_dimensional_transfer() {
        struct HugeMetric;

        impl Metric for HugeMetric {
            fn dist_unchecked(&self, a: &[f64], b: &[f64]) -> f64 {
                if a == b { 0.0 } else { 1.0e200 }
            }
        }

        let mut emb = Embedding::from_landmarks(
            array![[0.0], [1.0]],
            array![[0.0], [1.0]],
            &HugeMetric,
            Transfer::identity(),
            Transfer::identity(),
            0.0,
            None,
        )
        .unwrap();
        emb.high[(1, 0)] = 1.0e200;
        emb.tfun_hd = Transfer::xsigmoid(1.0, 8.0, 1.0).unwrap();
        let opts = ProjOpts {
            gridw: 1.0,
            grid_coarse: 1,
            grid_fine: 1,
            cg_steps: 0,
            ..ProjOpts::default()
        };
        assert!(project_report(&emb, array![0.0].view(), &HugeMetric, &opts).is_err());
    }

    #[test]
    fn rejects_invalid_imix_in_landmark_embeddings() {
        let high = array![[0.0, 0.0], [1.0, 0.0]];
        let low = high.clone();
        assert!(
            Embedding::from_landmarks(
                high,
                low,
                &Euclid,
                Transfer::identity(),
                Transfer::identity(),
                f64::NAN,
                None,
            )
            .is_err()
        );
    }
}
