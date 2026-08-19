//! Randomised pair search on χ.
//!
//! Mini-batches of pairs are drawn uniformly and a Robbins-Monro step is
//! taken on the corresponding partial gradient (Robbins and Monro, *Ann.
//! Math. Statist.* **22**, 400 (1951),
//! <https://doi.org/10.1214/aoms/1177729586>). Use this when the full
//! `n²` standard CG pass is too expensive. The default solver remains
//! [`crate::cg::minimize`].

use ndarray::{Array1, ArrayView1};

use crate::cg::CgReport;
use crate::metric::Euclid;
use crate::stress::{Stress, OVERLAP};

#[derive(Clone, Debug)]
pub struct StochOpts {
    pub steps: usize,
    pub batch: usize,
    pub seed: u64,
    pub step0: f64,
}

impl Default for StochOpts {
    fn default() -> Self {
        Self {
            steps: 200,
            batch: 64,
            seed: 1,
            step0: 0.05,
        }
    }
}

/// SplitMix64. Deterministic, no extra crate.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_index(state: &mut u64, n: usize) -> usize {
    (splitmix(state) as usize) % n
}

pub fn minimize_stochastic(
    stress: &Stress,
    init: ArrayView1<f64>,
    d: usize,
    opts: &StochOpts,
) -> CgReport {
    let n = stress.n;
    let mut pos = init.to_owned();
    let mut rng = opts.seed | 1;
    let metric = Euclid;
    let omix = 1.0 - stress.imix;
    let mut last = 0.0;

    for t in 0..opts.steps {
        let mut grad = vec![0.0; n * d];
        let mut tw = 0.0;
        let mut pval = 0.0;
        let b = opts.batch.max(1);
        for _ in 0..b {
            let i = rand_index(&mut rng, n);
            let mut j = rand_index(&mut rng, n);
            if i == j {
                j = (j + 1) % n;
            }
            let (lo, hi) = if i > j { (j, i) } else { (i, j) };
            let xi = &pos.as_slice().unwrap()[lo * d..(lo + 1) * d];
            let xj = &pos.as_slice().unwrap()[hi * d..(hi + 1) * d];
            let ld = metric.dist_unchecked(xi, xj);
            let (fld, dfld) = stress.tfun_ld.fdf(ld);
            let wij = stress.weights.as_ref().map(|w| w[(lo, hi)]).unwrap_or(1.0);
            tw += wij;
            let df = stress.fhd[(lo, hi)] - fld;
            let dd = stress.hd[(lo, hi)] - ld;
            pval += (df * df * omix + stress.imix * dd * dd) * wij;
            let dld = if ld < OVERLAP { OVERLAP } else { ld };
            let gij = (df * dfld * omix + stress.imix * dd) / dld * wij;
            for h in 0..d {
                let delta = xi[h] - xj[h];
                grad[lo * d + h] += gij * delta;
                grad[hi * d + h] -= gij * delta;
            }
        }
        if tw <= 0.0 {
            tw = 1.0;
        }
        last = pval / tw;
        let eta = opts.step0 / (1.0 + t as f64).sqrt();
        let scale = -2.0 * eta / tw;
        for k in 0..pos.len() {
            pos[k] += scale * grad[k];
        }
    }
    CgReport {
        value: last,
        coords: pos,
        steps: opts.steps,
    }
}
