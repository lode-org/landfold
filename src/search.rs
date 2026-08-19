//! Randomised pair search on χ.
//!
//! Mini-batches of pairs are drawn uniformly and a Robbins-Monro step is
//! taken on the corresponding partial gradient (Robbins and Monro, *Ann.
//! Math. Statist.* **22**, 400 (1951),
//! <https://doi.org/10.1214/aoms/1177729586>). Use this when the full
//! `n²` standard CG pass is too expensive. The default solver remains
//! [`crate::cg::minimize`].

use ndarray::ArrayView1;

use crate::cg::{CgReport, validate_packed_init};
use crate::metric::{Euclid, Metric};
use crate::stress::{OVERLAP, Stress};

#[derive(Clone, Debug)]
pub struct StochOpts {
    pub steps: usize,
    pub batch: usize,
    pub seed: u64,
    pub step0: f64,
}

impl StochOpts {
    pub(crate) fn validate(&self) -> crate::error::Result<()> {
        if self.batch == 0 {
            return Err(crate::error::LandfoldError::Msg(
                "stochastic batch must be > 0".into(),
            ));
        }
        if !self.step0.is_finite() || self.step0 < 0.0 {
            return Err(crate::error::LandfoldError::Msg(
                "stochastic step must be finite and nonnegative".into(),
            ));
        }
        Ok(())
    }
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
pub(crate) fn splitmix(state: &mut u64) -> u64 {
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
) -> crate::error::Result<CgReport> {
    let n = stress.n;
    validate_packed_init(init, n, d)?;
    opts.validate()?;
    let mut pos = init.to_owned();
    if n < 2 {
        return Ok(CgReport {
            value: stress.eval(pos.view(), d).value,
            coords: pos,
            steps: 0,
        });
    }
    let mut rng = opts.seed | 1;
    let metric = Euclid;
    let omix = 1.0 - stress.imix;
    let mut first_moment = vec![0.0; n * d];
    let mut second_moment = vec![0.0; n * d];
    const BETA1: f64 = 0.9;
    const BETA2: f64 = 0.999;
    const EPSILON: f64 = 1e-8;

    for t in 0..opts.steps {
        let mut grad = vec![0.0; n * d];
        let mut tw = 0.0;
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
            let ld = metric.dist(xi, xj)?;
            let (fld, dfld) = stress.tfun_ld.fdf(ld);
            let wij = stress.weights.as_ref().map(|w| w[(lo, hi)]).unwrap_or(1.0);
            tw += wij;
            let df = stress.fhd[(lo, hi)] - fld;
            let dd = stress.hd[(lo, hi)] - ld;
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
        let scale = -2.0 / tw;
        let bias1 = 1.0 - BETA1.powi((t + 1) as i32);
        let bias2 = 1.0 - BETA2.powi((t + 1) as i32);
        for k in 0..pos.len() {
            let gradient = scale * grad[k];
            first_moment[k] = BETA1 * first_moment[k] + (1.0 - BETA1) * gradient;
            second_moment[k] = BETA2 * second_moment[k] + (1.0 - BETA2) * gradient * gradient;
            let mean = first_moment[k] / bias1;
            let variance = (second_moment[k] / bias2).sqrt();
            pos[k] -= opts.step0 * mean / (variance + EPSILON);
        }
    }
    let value = stress.eval(pos.view(), d).value;
    Ok(CgReport {
        value,
        coords: pos,
        steps: opts.steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stress::Stress;
    use crate::transfer::Transfer;
    use ndarray::{Array2, array};

    #[test]
    fn reports_objective_at_returned_coordinates() {
        let hd = Array2::from_shape_vec((2, 2), vec![0.0, 2.0, 2.0, 0.0]).unwrap();
        let stress = Stress::new(hd.clone(), hd, Transfer::identity(), 1.0, None, None).unwrap();
        let opts = StochOpts {
            steps: 1,
            batch: 1,
            seed: 1,
            step0: 0.1,
        };
        let report = minimize_stochastic(&stress, array![0.0, 1.0].view(), 1, &opts).unwrap();
        let expected = stress.eval(report.coords.view(), 1).value;
        assert_eq!(report.value, expected);
    }

    #[test]
    fn rejects_overflowed_transfer_during_search() {
        let hd = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let mut stress = Stress::new(hd.clone(), hd, Transfer::identity(), 1.0, None, None).unwrap();
        stress.tfun_ld = Transfer::xsigmoid(1.0, 8.0, 1.0).unwrap();
        let opts = StochOpts {
            steps: 1,
            batch: 1,
            seed: 1,
            step0: 0.1,
        };
        assert!(minimize_stochastic(&stress, array![0.0, 1.0e154].view(), 1, &opts).is_err());
    }

    #[test]
    fn handles_empty_and_singleton_point_sets() {
        for n in [0, 1] {
            let hd = Array2::zeros((n, n));
            let init = ndarray::Array1::zeros(n);
            let stress =
                Stress::new(hd.clone(), hd, Transfer::identity(), 0.0, None, None).unwrap();
            let report =
                minimize_stochastic(&stress, init.view(), 1, &StochOpts::default()).unwrap();
            assert_eq!(report.steps, 0);
            assert_eq!(report.coords, init);
            assert_eq!(report.value, 0.0);
        }
    }

    #[test]
    fn rejects_invalid_options() {
        let hd = Array2::zeros((2, 2));
        let stress = Stress::new(hd.clone(), hd, Transfer::identity(), 0.0, None, None).unwrap();
        let init = array![0.0, 0.0];
        assert!(
            minimize_stochastic(
                &stress,
                init.view(),
                1,
                &StochOpts {
                    batch: 0,
                    ..StochOpts::default()
                }
            )
            .is_err()
        );
        assert!(
            minimize_stochastic(
                &stress,
                init.view(),
                1,
                &StochOpts {
                    step0: f64::NAN,
                    ..StochOpts::default()
                }
            )
            .is_err()
        );
    }
}
