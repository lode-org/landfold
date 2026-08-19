//! Algorithm contracts: closed-form transfer, χ descent, two-well separation.

use approx::assert_relative_eq;
use ndarray::Array2;
use landfold::{
    classical_mds, embed, pairwise_euclid, Euclid, IterOpts, Transfer,
};

fn xsigmoid_closed(x: f64, sigma: f64, a: f64, b: f64) -> f64 {
    let u = (2.0_f64.powf(a / b) - 1.0) * (x / sigma).powf(a);
    1.0 - (1.0 + u).powf(-b / a)
}

#[test]
fn xsigmoid_matches_ceriotti_formula() {
    let t = Transfer::xsigmoid(5.0, 8.0, 1.0).unwrap();
    for x in [0.0, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0] {
        assert_relative_eq!(t.f(x), xsigmoid_closed(x, 5.0, 8.0, 1.0), epsilon = 1e-14);
    }
    assert_relative_eq!(t.f(5.0), 0.5, epsilon = 1e-14);
}

#[test]
fn embed_lowers_chi_below_mds() {
    let mut pts = Array2::<f64>::zeros((12, 4));
    for i in 0..6 {
        pts[(i, 0)] = i as f64 * 0.02;
        pts[(i + 6, 0)] = 4.0 + i as f64 * 0.02;
        pts[(i + 6, 1)] = 0.3;
    }
    let hd = pairwise_euclid(pts.view()).unwrap();
    let mds = classical_mds(hd.view(), 2).unwrap().0;
    let mut opts = IterOpts::default();
    opts.lowdim = 2;
    opts.tfun_hd = Transfer::xsigmoid(2.0, 4.0, 3.0).unwrap();
    opts.tfun_ld = Transfer::xsigmoid(2.0, 2.0, 3.0).unwrap();
    opts.cg.maxiter = 25;
    let (emb, _) = embed(pts.view(), &Euclid, &opts, None, None, None).unwrap();
    let stress = landfold::stress::Stress::new(
        hd.clone(),
        {
            let mut f = hd.clone();
            landfold::apply_transfer(&mut f, &opts.tfun_hd);
            f
        },
        opts.tfun_ld.clone(),
        0.0,
        None,
        None,
    );
    let chi_mds = stress.eval(
        ndarray::Array1::from_iter(mds.iter().copied()).view(),
        2,
    );
    assert!(
        emb.stress <= chi_mds.value + 1e-10,
        "embed χ {} must not exceed MDS χ {}",
        emb.stress,
        chi_mds.value
    );
    assert!(emb.stress.is_finite());
}

#[test]
fn two_wells_separate_in_2d() {
    let mut pts = Array2::<f64>::zeros((16, 6));
    for i in 0..8 {
        pts[(i, 0)] = 0.01 * i as f64;
        pts[(i + 8, 0)] = 6.0 + 0.01 * i as f64;
        pts[(i + 8, 2)] = 0.5;
    }
    let mut opts = IterOpts::default();
    opts.lowdim = 2;
    opts.tfun_hd = Transfer::xsigmoid(3.0, 4.0, 2.0).unwrap();
    opts.tfun_ld = Transfer::xsigmoid(3.0, 2.0, 2.0).unwrap();
    opts.cg.maxiter = 30;
    let (emb, _) = embed(pts.view(), &Euclid, &opts, None, None, None).unwrap();
    let mut intra = 0.0;
    let mut n_in = 0;
    for i in 0..8 {
        for j in 0..i {
            let dx = emb.low[(i, 0)] - emb.low[(j, 0)];
            let dy = emb.low[(i, 1)] - emb.low[(j, 1)];
            intra += (dx * dx + dy * dy).sqrt();
            n_in += 1;
        }
    }
    intra /= n_in as f64;
    let mut inter = 0.0;
    let mut n_out = 0;
    for i in 0..8 {
        for j in 8..16 {
            let dx = emb.low[(i, 0)] - emb.low[(j, 0)];
            let dy = emb.low[(i, 1)] - emb.low[(j, 1)];
            inter += (dx * dx + dy * dy).sqrt();
            n_out += 1;
        }
    }
    inter /= n_out as f64;
    assert!(
        inter > 2.0 * intra,
        "wells mixed: intra={intra} inter={inter}"
    );
}
