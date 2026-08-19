//! Out-of-sample placement of new high-D points.
//!
//! A new point is dropped onto the low-D map by minimising its χ against
//! the landmarks (Ceriotti, Tribello, Parrinello, *J. Chem. Theory Comput.*
//! **9**, 1521 (2013), <https://doi.org/10.1021/ct3010563>).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{Result, LandfoldError};
use crate::iter::Embedding;
use crate::metric::Metric;
use crate::stress::Stress;

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
            [w, g1, g2] => Ok(Self {
                gridw: *w,
                grid_coarse: *g1 as usize,
                grid_fine: *g2 as usize,
                cg_steps: 0,
            }),
            _ => Err(LandfoldError::Parse(
                "-grid needs gw,g1,g2".into(),
            )),
        }
    }
}

/// Project one high-D query onto the landmark embedding.
pub fn project_one(
    emb: &Embedding,
    query: ArrayView1<f64>,
    metric: &dyn Metric,
    opts: &ProjOpts,
) -> Result<Array1<f64>> {
    if emb.low.ncols() != 2 && opts.grid_coarse > 1 {
        // Coarse+fine grid search is implemented for 2-D embeddings.
    }
    let n = emb.high.nrows();
    let d_hi = emb.high.ncols();
    if query.len() != d_hi {
        return Err(LandfoldError::MetricSize {
            left: query.len(),
            right: d_hi,
        });
    }
    let mut hd_row = Array1::<f64>::zeros(n);
    for i in 0..n {
        let row: Vec<f64> = emb.high.row(i).iter().copied().collect();
        hd_row[i] = metric.dist_unchecked(query.as_slice().unwrap_or(&row), &row);
    }
    // Build a 1-vs-landmarks stress by grafting the query as extra row 0
    // of a tiny chi1 evaluation against the existing fhd/hd of landmarks
    // plus the new distances.
    let mut fhd_row = Array1::<f64>::zeros(n);
    for i in 0..n {
        fhd_row[i] = emb.tfun_hd.f(hd_row[i]);
    }
    let stress = Stress::new(
        emb.hd.clone(),
        emb.fhd.clone(),
        emb.tfun_ld.clone(),
        emb.imix,
        None,
        None,
    );
    let d = emb.low.ncols();
    let mut best = Array1::<f64>::zeros(d);
    let mut best_f = f64::INFINITY;

    if d == 2 && opts.grid_coarse >= 2 {
        let g1 = opts.grid_coarse;
        let w = opts.gridw;
        for i in 0..g1 {
            for j in 0..g1 {
                let x = -w + (2.0 * w) * (i as f64) / (g1 as f64 - 1.0);
                let y = -w + (2.0 * w) * (j as f64) / (g1 as f64 - 1.0);
                let q = array_xy(x, y);
                let (f, _) = stress.chi_query(
                    q.view(),
                    emb.low.view(),
                    hd_row.view(),
                    fhd_row.view(),
                    emb.weights.view(),
                );
                if f < best_f {
                    best_f = f;
                    best = q;
                }
            }
        }
        let g2 = opts.grid_fine.max(g1);
        let span = 2.0 * w / (g1 as f64);
        let cx = best[0];
        let cy = best[1];
        for i in 0..g2 {
            for j in 0..g2 {
                let x = cx - span + (2.0 * span) * (i as f64) / (g2 as f64 - 1.0);
                let y = cy - span + (2.0 * span) * (j as f64) / (g2 as f64 - 1.0);
                let q = array_xy(x, y);
                let (f, _) = stress.chi_query(
                    q.view(),
                    emb.low.view(),
                    hd_row.view(),
                    fhd_row.view(),
                    emb.weights.view(),
                );
                if f < best_f {
                    best_f = f;
                    best = q;
                }
            }
        }
    } else {
        // Nearest-neighbour landmark seed.
        let mut nn = 0usize;
        let mut nd = f64::INFINITY;
        for i in 0..n {
            if hd_row[i] < nd {
                nd = hd_row[i];
                nn = i;
            }
        }
        best = emb.low.row(nn).to_owned();
    }

    if opts.cg_steps > 0 {
        // Pack [query_ld | landmarks] and freeze landmarks by a 1-point CG
        // on chi1 via a tiny custom loop.
        let mut packed = Array1::<f64>::zeros((n + 1) * d);
        for h in 0..d {
            packed[h] = best[h];
        }
        for i in 0..n {
            for h in 0..d {
                packed[(i + 1) * d + h] = emb.low[(i, h)];
            }
        }
        // Only the query coordinates move: run a few gradient steps on them.
        for _ in 0..opts.cg_steps {
            let (f, g) = stress.chi_query(
                best.view(),
                emb.low.view(),
                hd_row.view(),
                fhd_row.view(),
                emb.weights.view(),
            );
            let _ = f;
            let gn: f64 = g.iter().map(|x| x * x).sum::<f64>().sqrt();
            if gn < 1e-12 {
                break;
            }
            let step = 0.1 / gn.max(1e-12);
            for h in 0..d {
                best[h] -= step * g[h];
            }
        }
    }
    Ok(best)
}

pub fn project_many(
    emb: &Embedding,
    queries: ArrayView2<f64>,
    metric: &dyn Metric,
    opts: &ProjOpts,
) -> Result<Array2<f64>> {
    let nq = queries.nrows();
    let d = emb.low.ncols();
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let rows: crate::error::Result<Vec<Array1<f64>>> = (0..nq)
            .into_par_iter()
            .map(|i| project_one(emb, queries.row(i), metric, opts))
            .collect();
        let rows = rows?;
        let mut out = Array2::<f64>::zeros((nq, d));
        for i in 0..nq {
            for h in 0..d {
                out[(i, h)] = rows[i][h];
            }
        }
        return Ok(out);
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut out = Array2::<f64>::zeros((nq, d));
        for i in 0..nq {
            let p = project_one(emb, queries.row(i), metric, opts)?;
            for h in 0..d {
                out[(i, h)] = p[h];
            }
        }
        Ok(out)
    }
}

fn array_xy(x: f64, y: f64) -> Array1<f64> {
    Array1::from(vec![x, y])
}
