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
use crate::stress::query_chi;

#[derive(Clone, Debug)]
pub struct ProjOpts {
    pub gridw: f64,
    pub grid_coarse: usize,
    pub grid_fine: usize,
    pub cg_steps: usize,
}

impl Default for ProjOpts {
    fn default() -> Self {
        Self {
            gridw: 1.0,
            grid_coarse: 21,
            grid_fine: 201,
            cg_steps: 0,
        }
    }
}

impl ProjOpts {
    fn validate(&self) -> Result<()> {
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
                Ok(Self {
                    gridw: *w,
                    grid_coarse: *g1 as usize,
                    grid_fine: (*g2 as usize).max(1),
                    cg_steps: 0,
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
    opts.validate()?;
    let n = emb.high.nrows();
    let d_hi = emb.high.ncols();
    if query.len() != d_hi {
        return Err(LandfoldError::MetricSize {
            left: query.len(),
            right: d_hi,
        });
    }
    if n == 0 {
        return Err(LandfoldError::Empty);
    }
    let qslice: Vec<f64> = query.iter().copied().collect();
    let mut hd_row = Array1::<f64>::zeros(n);
    let mut landmark = vec![0.0; d_hi];
    let mut nearest = f64::INFINITY;
    let mut nearest_idx = 0usize;
    for i in 0..n {
        for (h, value) in landmark.iter_mut().enumerate().take(d_hi) {
            *value = emb.high[(i, h)];
        }
        let d = metric.dist(&qslice, &landmark)?;
        hd_row[i] = d;
        if d < nearest {
            nearest = d;
            nearest_idx = i;
        }
    }
    let mut fhd_row = Array1::<f64>::zeros(n);
    for i in 0..n {
        fhd_row[i] = emb.tfun_hd.f(hd_row[i]);
    }
    let d = emb.low.ncols();
    let mut best = emb.low.row(nearest_idx).to_owned();
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

    if d == 2 && opts.grid_coarse >= 2 {
        let w = opts.gridw;
        let g1 = opts.grid_coarse;
        let mut best_f = f64::INFINITY;
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
        let mut best_f = f64::INFINITY;
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

    let (chi, _) = eval(best.view());
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

fn scan_grid_2d(x0: f64, x1: f64, y0: f64, y1: f64, n: usize, mut visit: impl FnMut(Array1<f64>)) {
    let denom = (n.saturating_sub(1) as f64).max(1.0);
    for i in 0..n {
        let x = x0 + (x1 - x0) * (i as f64) / denom;
        for j in 0..n {
            let y = y0 + (y1 - y0) * (j as f64) / denom;
            visit(Array1::from(vec![x, y]));
        }
    }
}

fn scan_grid_1d(x0: f64, x1: f64, n: usize, mut visit: impl FnMut(Array1<f64>)) {
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
        };
        let r = project_report(&emb, array![1.0, 0.0].view(), &Euclid, &opts).unwrap();
        assert_eq!(r.nearest_idx, 1);
        assert!((r.nearest).abs() < 1e-14);
        assert!((r.coords[0] - 1.0).abs() < 1e-14);
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
