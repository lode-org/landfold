//! Same two-well set, same transfers, four solvers. Prints χ and wall time.
//! MDS init and a scrambled init. No loosened CG tolerances.

use std::time::Instant;

use landfold::{AnnealOpts, Euclid, IterOpts, ReplicaOpts, Solver, StochOpts, Transfer, embed};
use ndarray::Array2;

fn two_wells() -> Array2<f64> {
    let mut pts = Array2::<f64>::zeros((16, 6));
    for i in 0..8 {
        pts[(i, 0)] = 0.01 * i as f64;
        pts[(i + 8, 0)] = 6.0 + 0.01 * i as f64;
        pts[(i + 8, 2)] = 0.5;
    }
    pts
}

fn base_opts() -> IterOpts {
    IterOpts {
        lowdim: 2,
        tfun_hd: Transfer::xsigmoid(3.0, 4.0, 2.0).unwrap(),
        tfun_ld: Transfer::xsigmoid(3.0, 2.0, 2.0).unwrap(),
        ..IterOpts::default()
    }
}

fn run(label: &str, pts: &Array2<f64>, opts: IterOpts, init: Option<ndarray::ArrayView2<f64>>) {
    let t0 = Instant::now();
    let (emb, report) = embed(pts.view(), &Euclid, &opts, init, None, None).unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let max_abs = emb.low.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    println!(
        "{:<18} chi={:.8e} steps={} ms={:.2} max|x|={:.4}",
        label, emb.stress, report.steps, ms, max_abs
    );
}

fn main() {
    let pts = two_wells();
    println!("# two-well n=16 D=6 fun-hd=3,4,2 fun-ld=3,2,2  (algorithm.rs contract)");
    println!("# MDS init");
    run("standard", &pts, base_opts(), None);
    let mut a = base_opts();
    a.solver = Solver::Anneal(AnnealOpts::default());
    run("anneal+cg", &pts, a, None);
    let mut r = base_opts();
    r.solver = Solver::Replica(ReplicaOpts::default());
    run("replica+cg", &pts, r, None);
    let mut s = base_opts();
    s.solver = Solver::Stochastic(StochOpts::default());
    run("stoch", &pts, s, None);

    // Same points, scrambled 2-D start (not MDS).
    let mut bad = Array2::<f64>::zeros((16, 2));
    let mut st = 1u64;
    for i in 0..16 {
        for h in 0..2 {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            bad[(i, h)] = ((st >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)) * 4.0 - 2.0;
        }
    }
    println!("# scrambled init in [-2,2]^2");
    run("standard", &pts, base_opts(), Some(bad.view()));
    let mut a = base_opts();
    a.solver = Solver::Anneal(AnnealOpts::default());
    run("anneal+cg", &pts, a, Some(bad.view()));
    let mut r = base_opts();
    r.solver = Solver::Replica(ReplicaOpts::default());
    run("replica+cg", &pts, r, Some(bad.view()));
    let mut s = base_opts();
    s.solver = Solver::Stochastic(StochOpts::default());
    run("stoch", &pts, s, Some(bad.view()));

    #[cfg(feature = "highs")]
    {
        use landfold::HighsOpts;
        println!("# HiGHS L-BFGS-QP, MDS init");
        let mut h = base_opts();
        h.solver = Solver::Highs(HighsOpts::default());
        run("highs", &pts, h, None);
        println!("# HiGHS box [-0.3,0.3], scrambled init");
        let hb = HighsOpts {
            lo: Some(-0.3),
            hi: Some(0.3),
            ..HighsOpts::default()
        };
        let mut h = base_opts();
        h.solver = Solver::Highs(hb);
        run("highs-box", &pts, h, Some(bad.view()));
        println!("# standard, same scrambled init (unconstrained)");
        run("standard", &pts, base_opts(), Some(bad.view()));
    }
}
