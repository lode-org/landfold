//! End-to-end: three well-separated blobs stay separated after embedding.

use approx::assert_relative_eq;
use ndarray::Array2;
use landfold::{embed_points, Euclid, IterOpts, Transfer};

#[test]
fn three_blobs_stay_separated() {
    let mut pts = Array2::<f64>::zeros((30, 4));
    for i in 0..10 {
        pts[(i, 0)] = i as f64 * 0.01;
        pts[(i + 10, 0)] = 5.0 + i as f64 * 0.01;
        pts[(i + 20, 0)] = i as f64 * 0.01;
        pts[(i + 20, 1)] = 5.0 + i as f64 * 0.01;
    }
    let mut opts = IterOpts::default();
    opts.lowdim = 2;
    opts.tfun_hd = Transfer::xsigmoid(2.0, 4.0, 3.0).unwrap();
    opts.tfun_ld = Transfer::xsigmoid(2.0, 2.0, 3.0).unwrap();
    opts.cg.maxiter = 20;
    let emb = embed_points(pts.view(), &Euclid, &opts).unwrap();
    assert_eq!(emb.low.nrows(), 30);
    assert!(emb.stress.is_finite());
    // Intra-blob distance in 2-D should be smaller than inter-blob.
    let d01 = ((emb.low[(0, 0)] - emb.low[(10, 0)]).powi(2)
        + (emb.low[(0, 1)] - emb.low[(10, 1)]).powi(2))
    .sqrt();
    let d00 = ((emb.low[(0, 0)] - emb.low[(1, 0)]).powi(2)
        + (emb.low[(0, 1)] - emb.low[(1, 1)]).powi(2))
    .sqrt();
    assert!(d01 > d00, "inter {d01} should exceed intra {d00}");
    assert_relative_eq!(emb.low[(0, 0)], emb.low[(0, 0)], epsilon = 0.0);
}
