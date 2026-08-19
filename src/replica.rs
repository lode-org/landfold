//! Replica-exchange (parallel tempering) on χ.
//!
//! Adjacent replicas swap with the Metropolis rule of Hukushima and
//! Nemoto, *J. Phys. Soc. Jpn.* **65**, 1604 (1996),
//! <https://doi.org/10.1143/JPSJ.65.1604>. Each replica walks with
//! the same axis Metropolis kernel as [`crate::anneal`]. Extra arm
//! only: [`crate::iter::Solver::Standard`] stays the default.

use ndarray::ArrayView1;

use crate::anneal::urand;
use crate::cg::{minimize, CgOpts, CgReport};
use crate::search::splitmix;
use crate::stress::Stress;

#[derive(Clone, Debug)]
pub struct ReplicaOpts {
    pub replicas: usize,
    pub steps: usize,
    pub sweep: usize,
    pub temp_init: f64,
    pub temp_final: f64,
    pub mc_step: f64,
    pub seed: u64,
    pub polish: bool,
}

impl Default for ReplicaOpts {
    fn default() -> Self {
        Self {
            replicas: 4,
            steps: 80,
            sweep: 4,
            temp_init: 1e-3,
            temp_final: 1e-6,
            mc_step: 0.1,
            seed: 1,
            polish: true,
        }
    }
}

fn metropolis_sweep(
    stress: &Stress,
    pos: &mut ndarray::Array1<f64>,
    nrg: &mut f64,
    temp: f64,
    step: f64,
    rng: &mut u64,
    d: usize,
) {
    let nv = pos.len();
    for iu in 0..nv {
        let mut npos = pos.clone();
        npos[iu] += step * (urand(rng) - 0.5);
        let nnrg = stress.eval(npos.view(), d).value;
        let accept = if nnrg <= *nrg {
            1.0
        } else {
            ((*nrg - nnrg) / temp).exp()
        };
        if urand(rng) <= accept {
            *pos = npos;
            *nrg = nnrg;
        }
    }
}

/// Geometric ladder of temperatures, Metropolis sweeps, adjacent swaps.
pub fn minimize_replica(
    stress: &Stress,
    init: ArrayView1<f64>,
    d: usize,
    opts: &ReplicaOpts,
    cg: &CgOpts,
) -> crate::error::Result<CgReport> {
    let nr = opts.replicas.max(2);
    let t0 = opts.temp_init.max(1e-300);
    let t1 = opts.temp_final.max(1e-300);
    let mut temps = vec![0.0; nr];
    if nr == 1 {
        temps[0] = t0;
    } else {
        let ratio = (t1 / t0).powf(1.0 / (nr - 1) as f64);
        temps[0] = t0;
        for i in 1..nr {
            temps[i] = temps[i - 1] * ratio;
        }
    }
    let mut pos: Vec<ndarray::Array1<f64>> = (0..nr).map(|_| init.to_owned()).collect();
    let mut nrg: Vec<f64> = pos
        .iter()
        .map(|p| stress.eval(p.view(), d).value)
        .collect();
    let mut rng = opts.seed | 1;
    let mut best = nrg[0];
    let mut best_pos = pos[0].clone();
    let nsteps = opts.steps.max(1);
    let sweep = opts.sweep.max(1);

    for s in 0..nsteps {
        for r in 0..nr {
            for _ in 0..sweep {
                metropolis_sweep(
                    stress,
                    &mut pos[r],
                    &mut nrg[r],
                    temps[r],
                    opts.mc_step,
                    &mut rng,
                    d,
                );
            }
            if nrg[r] < best {
                best = nrg[r];
                best_pos = pos[r].clone();
            }
        }
        let start = s % 2;
        let mut i = start;
        while i + 1 < nr {
            let beta_i = 1.0 / temps[i];
            let beta_j = 1.0 / temps[i + 1];
            let delta = (beta_i - beta_j) * (nrg[i] - nrg[i + 1]);
            if delta >= 0.0 || urand(&mut rng) < delta.exp() {
                pos.swap(i, i + 1);
                nrg.swap(i, i + 1);
            }
            i += 2;
        }
        let _ = splitmix(&mut rng);
    }

    if opts.polish {
        return minimize(stress, best_pos.view(), d, cg);
    }
    Ok(CgReport {
        value: best,
        coords: best_pos,
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
    fn replica_lowers_or_matches_init() {
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
        apply_transfer(&mut fhd, &t);
        let s = Stress::new(hd, fhd, t, 0.0, None, None);
        let init =
            Array::from_iter([0.0, 0.0, 0.2, 0.1, -0.1, 0.3, 0.4, -0.2, 1.0, 1.1, 1.2, 0.8]);
        let ev0 = s.eval(init.view(), 2);
        let mut ro = ReplicaOpts::default();
        ro.steps = 12;
        ro.sweep = 2;
        ro.replicas = 3;
        ro.polish = true;
        let rep = minimize_replica(&s, init.view(), 2, &ro, &CgOpts::default()).unwrap();
        assert!(rep.value <= ev0.value + 1e-12);
    }
}
