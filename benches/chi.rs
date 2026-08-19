//! Criterion on χ+∇ for the teaching x-sigmoid, n=400.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use landfold::stress::Stress;
use landfold::{apply_transfer, pairwise_euclid, Transfer};

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

fn bench_chi(c: &mut Criterion) {
    let n = 400usize;
    let dim_hi = 8usize;
    let mut st = 1u64;
    let mut hi = ndarray::Array2::<f64>::zeros((n, dim_hi));
    for i in 0..n {
        for h in 0..dim_hi {
            hi[(i, h)] = lcg(&mut st);
        }
    }
    let hd = pairwise_euclid(hi.view()).unwrap();
    let t_hd = Transfer::xsigmoid(5.0, 8.0, 1.0).unwrap();
    let t_ld = Transfer::xsigmoid(5.0, 2.0, 2.0).unwrap();
    let mut fhd = hd.clone();
    apply_transfer(&mut fhd, &t_hd).unwrap();
    let s = Stress::new(hd, fhd, t_ld, 0.0, None, None).unwrap();
    let mut low = ndarray::Array1::<f64>::zeros(n * 2);
    for k in 0..n * 2 {
        low[k] = lcg(&mut st) - 0.5;
    }
    c.bench_function("chi_eval_n400", |b| {
        b.iter(|| black_box(s.eval(black_box(low.view()), 2)))
    });
}

criterion_group!(benches, bench_chi);
criterion_main!(benches);
