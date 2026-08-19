//! Time n chi+grad evals. Same point set as oracle/bench_chi.cpp.
fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let dim_hi = 8usize;
    let mut st = 1u64;
    let mut hi = ndarray::Array2::<f64>::zeros((n, dim_hi));
    for i in 0..n {
        for h in 0..dim_hi {
            hi[(i, h)] = lcg(&mut st);
        }
    }
    let hd = landfold::pairwise_euclid(hi.view()).unwrap();
    let t_hd = landfold::Transfer::xsigmoid(5.0, 8.0, 1.0).unwrap();
    let t_ld = landfold::Transfer::xsigmoid(5.0, 2.0, 2.0).unwrap();
    let mut fhd = hd.clone();
    landfold::apply_transfer(&mut fhd, &t_hd).unwrap();
    let s = landfold::stress::Stress::new(hd, fhd, t_ld, 0.0, None, None).unwrap();
    let mut low = ndarray::Array1::<f64>::zeros(n * 2);
    for k in 0..n * 2 {
        low[k] = lcg(&mut st) - 0.5;
    }
    let ev0 = s.eval(low.view(), 2);
    let t0 = std::time::Instant::now();
    let mut last = ev0.value;
    for _ in 0..reps {
        last = s.eval(low.view(), 2).value;
    }
    let ns = t0.elapsed().as_nanos() / reps as u128;
    println!("impl=landfold n={n} reps={reps} ns_per_eval={ns} chi={last:.17e}");
}
