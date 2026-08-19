//! Simulated annealing on χ, then optional CG polish.
//!
//! Geometric cooling and Metropolis updates follow Kirkpatrick, Gelatt
//! and Vecchi, *Science* **220**, 671 (1983),
//! <https://doi.org/10.1126/science.220.4598.671>. Per-axis step
//! adaptation matches the C++ annealing template. After the
//! schedule, Polak-Ribiere CG projects into the local basin.

use ndarray::ArrayView1;

use crate::cg::{CgOpts, CgReport, minimize, validate_packed_init};
use crate::search::splitmix;
use crate::stress::Stress;

#[derive(Clone, Debug)]
pub struct AnnealOpts {
    pub steps: usize,
    pub temp_init: f64,
    pub temp_final: f64,
    pub mc_step: f64,
    pub adapt: f64,
    pub seed: u64,
    /// Run standard CG from the last accepted point.
    pub polish: bool,
}

impl Default for AnnealOpts {
    fn default() -> Self {
        Self {
            steps: 200,
            temp_init: 1e-4,
            temp_final: 1e-20,
            mc_step: 0.1,
            adapt: 1.05,
            seed: 1,
            polish: true,
        }
    }
}

impl AnnealOpts {
    pub(crate) fn validate(&self) -> crate::error::Result<()> {
        if !self.temp_init.is_finite()
            || !self.temp_final.is_finite()
            || self.temp_init <= 0.0
            || self.temp_final <= 0.0
        {
            return Err(crate::error::LandfoldError::Msg(
                "annealing temperatures must be finite and > 0".into(),
            ));
        }
        if !self.mc_step.is_finite() || self.mc_step < 0.0 {
            return Err(crate::error::LandfoldError::Msg(
                "annealing step must be finite and nonnegative".into(),
            ));
        }
        if !self.adapt.is_finite() || self.adapt <= 0.0 {
            return Err(crate::error::LandfoldError::Msg(
                "annealing adaptation must be finite and > 0".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn urand(state: &mut u64) -> f64 {
    (splitmix(state) as f64) * (1.0 / ((u64::MAX as f64) + 1.0))
}

/// Axis-aligned Metropolis SA on packed coordinates, then optional CG.
pub fn minimize_anneal(
    stress: &Stress,
    init: ArrayView1<f64>,
    d: usize,
    opts: &AnnealOpts,
    cg: &CgOpts,
) -> crate::error::Result<CgReport> {
    validate_packed_init(init, stress.n, d)?;
    opts.validate()?;
    let nv = init.len();
    let mut pos = init.to_owned();
    let mut ev = stress.eval(pos.view(), d);
    let mut nrg = ev.value;
    let mut step = vec![opts.mc_step; nv];
    let mut accept = vec![0u64; nv];
    let mut tstep = vec![0u64; nv];
    let mut rng = opts.seed | 1;
    let t0 = opts.temp_init.max(1e-300);
    let t1 = opts.temp_final.max(1e-300);
    let ts = if opts.steps == 0 {
        1.0
    } else {
        (t1 / t0).ln() / opts.steps as f64
    };
    let mut temp = t0;
    let nsteps = opts.steps.max(1);

    for _ in 0..nsteps {
        for iu in 0..nv {
            tstep[iu] += 1;
            let mut npos = pos.clone();
            npos[iu] += step[iu] * (urand(&mut rng) - 0.5);
            let nnrg = stress.eval(npos.view(), d).value;
            let accept_p = if nnrg <= nrg {
                1.0
            } else {
                ((nrg - nnrg) / temp).exp()
            };
            if urand(&mut rng) <= accept_p {
                accept[iu] += 1;
                pos = npos;
                nrg = nnrg;
            }
            if accept[iu] * 2 > tstep[iu] {
                step[iu] *= opts.adapt;
            } else if opts.adapt > 0.0 {
                step[iu] /= opts.adapt;
            }
        }
        temp *= ts.exp();
    }

    if opts.polish {
        return minimize(stress, pos.view(), d, cg);
    }
    ev = stress.eval(pos.view(), d);
    Ok(CgReport {
        value: ev.value,
        coords: pos,
        steps: nsteps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pairwise::{apply_transfer, pairwise_euclid};
    use crate::transfer::Transfer;
    use ndarray::{Array, array};

    #[test]
    fn anneal_lowers_or_matches_init() {
        let pts = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.1, 0.9],
            [0.9, 0.1],
            [3.0, 3.0],
            [3.1, 2.9]
        ];
        let hd = pairwise_euclid(pts.view()).unwrap();
        let t = Transfer::xsigmoid(1.0, 4.0, 2.0).unwrap();
        let mut fhd = hd.clone();
        apply_transfer(&mut fhd, &t).unwrap();
        let s = Stress::new(hd, fhd, t, 0.0, None, None);
        let init = Array::from_iter([0.0, 0.0, 0.2, 0.1, -0.1, 0.3, 0.4, -0.2, 1.0, 1.1, 1.2, 0.8]);
        let ev0 = s.eval(init.view(), 2);
        let ao = AnnealOpts {
            steps: 40,
            polish: true,
            ..AnnealOpts::default()
        };
        let rep = minimize_anneal(&s, init.view(), 2, &ao, &CgOpts::default()).unwrap();
        assert!(rep.value <= ev0.value + 1e-12);
    }

    #[test]
    fn rejects_invalid_options() {
        assert!(
            AnnealOpts {
                mc_step: f64::NAN,
                ..AnnealOpts::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            AnnealOpts {
                adapt: 0.0,
                ..AnnealOpts::default()
            }
            .validate()
            .is_err()
        );
    }
}
