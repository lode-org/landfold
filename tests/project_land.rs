//! Out-of-sample projection, farthest-point landmarks, and FES invert.

use approx::assert_relative_eq;
use landfold::{
    coordination_histogram, embed_points, farthest_point, project_one, Euclid, FreeEnergy,
    Histogram2d, IterOpts, ProjOpts, Transfer,
};
use ndarray::array;

#[test]
fn farthest_point_returns_k_distinct() {
    let pts = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ];
    let lm = farthest_point(pts.view(), &Euclid, 3, None, 0).unwrap();
    assert_eq!(lm.index.len(), 3);
    let mut s = lm.index.clone();
    s.sort_unstable();
    s.dedup();
    assert_eq!(s.len(), 3);
}

#[test]
fn project_recovers_held_out_corner() {
    let pts = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ];
    let mut opts = IterOpts::default();
    opts.lowdim = 2;
    opts.tfun_hd = Transfer::identity();
    opts.tfun_ld = Transfer::identity();
    opts.cg.maxiter = 40;
    let emb = embed_points(pts.view(), &Euclid, &opts).unwrap();
    let po = ProjOpts {
        gridw: 2.0,
        grid_coarse: 21,
        grid_fine: 41,
        cg_steps: 8,
    };
    let q = project_one(&emb, pts.row(0), &Euclid, &po).unwrap();
    let target = emb.low.row(0);
    let d = ((q[0] - target[0]).powi(2) + (q[1] - target[1]).powi(2)).sqrt();
    assert!(d < 0.15, "held-out corner landed {d} from its embedding");
}

#[test]
fn fes_minimum_sits_on_the_dense_bin() {
    let pts = array![
        [0.0, 0.0],
        [0.05, 0.0],
        [0.0, 0.05],
        [0.05, 0.05],
        [3.0, 3.0]
    ];
    let mut h = Histogram2d::new(-1.0, 4.0, 20, -1.0, 4.0, 20).unwrap();
    for i in 0..pts.nrows() {
        h.add(pts[(i, 0)], pts[(i, 1)], 1.0);
    }
    let fes = FreeEnergy::from_histogram(&h, 1.0);
    let mut best = f64::INFINITY;
    let mut ix = 0usize;
    let mut iy = 0usize;
    for y in 0..fes.f.nrows() {
        for x in 0..fes.f.ncols() {
            if let Some(v) = fes.f[(y, x)] {
                if v < best {
                    best = v;
                    ix = x;
                    iy = y;
                }
            }
        }
    }
    assert!(fes.x_centers[ix] < 1.0);
    assert!(fes.y_centers[iy] < 1.0);
    assert_relative_eq!(best, 0.0, epsilon = 1e-12);
}

#[test]
fn coordination_histogram_covers_zero_to_max() {
    let frame = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.87, 0.0],
        [0.5, 0.29, 0.82]
    ];
    let h = coordination_histogram(frame.view(), 1.5, 12);
    assert_eq!(h.counts.len(), 13);
    assert!(h.counts.iter().sum::<f64>() > 0.0);
}
