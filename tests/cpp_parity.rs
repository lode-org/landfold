//! Compare landfold kernels to goldens dumped from the C++ oracle (`interpol=false`).
//!
//! Goldens are produced by `scripts/gen_cpp_goldens.sh` on the remote builder
//! against the HaoZeke addLocks tree. Missing file is a hard fail: no
//! paper-only stand-in. MDS coordinates are not compared; `NLDRProjection.p`
//! is private, so the oracle dumps i<j Euclidean distances of `p`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use approx::assert_relative_eq;
use landfold::pairwise::apply_transfer;
use landfold::stress::Stress;
use landfold::{
    classical_mds, farthest_point, pairwise, pairwise_euclid, query_chi, Dot, Euclid, IterOpts,
    Metric, Periodic, Solver, Transfer,
};
use ndarray::{array, ArrayView2};

const XFER_ABS: f64 = 1e-14;
const XFER_REL: f64 = 1e-13;
const METRIC_ABS: f64 = 1e-15;
const CHI_ABS: f64 = 1e-14;
const CHI_REL: f64 = 1e-13;
const PAIR_ABS: f64 = 1e-14;
const PAIR_REL: f64 = 1e-13;
const MDS_ABS: f64 = 1e-12;
const MDS_REL: f64 = 1e-11;

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/cpp_oracle.txt")
}

fn load_blocks() -> BTreeMap<String, Vec<Vec<f64>>> {
    let path = golden_path();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "missing {}; run scripts/gen_cpp_goldens.sh on the remote builder",
            path.display()
        );
    });
    parse_blocks(&text)
}

fn parse_blocks(text: &str) -> BTreeMap<String, Vec<Vec<f64>>> {
    let mut out = BTreeMap::new();
    let mut cur: Option<String> = None;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("BEGIN ") {
            cur = Some(rest.trim().to_string());
            rows.clear();
        } else if line == "END" {
            if let Some(k) = cur.take() {
                out.insert(k, rows.clone());
            }
        } else if cur.is_some() && !line.is_empty() {
            let nums: Vec<f64> = line
                .split_whitespace()
                .map(|s| s.parse().expect("golden f64"))
                .collect();
            rows.push(nums);
        }
    }
    out
}

fn upper_triangle(dist: ArrayView2<f64>) -> Vec<f64> {
    let n = dist.nrows();
    let mut out = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            out.push(dist[(i, j)]);
        }
    }
    out
}

fn scalar_col(rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter()
        .map(|r| {
            assert_eq!(r.len(), 1, "expected one scalar per golden row");
            r[0]
        })
        .collect()
}

fn match_abs_rel(got: f64, gold: f64, eps: f64, max_rel: f64) -> bool {
    let diff = (got - gold).abs();
    diff <= eps || diff <= max_rel * got.abs().max(gold.abs())
}

#[test]
fn cpp_oracle_goldens_exist_and_match() {
    let blocks = load_blocks();
    for key in [
        "xfer xsigmoid_5_8_1",
        "pair euclid_triangle",
        "pair euclid_tetra",
        "pair pbc_3pt",
        "mds torgerson_triangle",
        "mds torgerson_tetra",
        "chi xsig_1_4_3_imix01",
        "chi1 identity_square",
        "landmarks fps_square_k3",
    ] {
        assert!(
            blocks.contains_key(key),
            "oracle did not dump `{key}`; regenerate tests/goldens/cpp_oracle.txt"
        );
    }

    let cases: &[(&str, Transfer)] = &[
        ("xfer identity", Transfer::identity()),
        ("xfer sigmoid_1", Transfer::sigmoid(1.0).unwrap()),
        ("xfer compress_1", Transfer::compress(1.0).unwrap()),
        (
            "xfer xsigmoid_5_8_1",
            Transfer::xsigmoid(5.0, 8.0, 1.0).unwrap(),
        ),
        (
            "xfer xsigmoid_6_8_8",
            Transfer::xsigmoid(6.0, 8.0, 8.0).unwrap(),
        ),
        (
            "xfer warp_5_8_1_2_2",
            Transfer::warp(5.0, 8.0, 1.0, 2.0, 2.0).unwrap(),
        ),
    ];
    for (key, t) in cases {
        let rows = blocks
            .get(*key)
            .unwrap_or_else(|| panic!("missing block {key}"));
        for row in rows {
            assert_eq!(row.len(), 3, "{key} row");
            let x = row[0];
            let (f, df) = t.fdf(x);
            assert_relative_eq!(f, row[1], epsilon = XFER_ABS, max_relative = XFER_REL);
            if x != 0.0 {
                assert_relative_eq!(df, row[2], epsilon = 1e-13, max_relative = 1e-12);
            }
        }
    }

    let e = blocks["metric euclid_3_4_5"][0][0];
    assert_relative_eq!(
        Euclid.dist(&[0.0, 0.0, 0.0], &[3.0, 4.0, 0.0]).unwrap(),
        e,
        epsilon = METRIC_ABS
    );
    let p = blocks["metric pbc_wrap"][0][0];
    let pbc = Periodic::isotropic(1, 1.0).unwrap();
    assert_relative_eq!(pbc.dist(&[0.05], &[0.95]).unwrap(), p, epsilon = METRIC_ABS);
    let d = blocks["metric dot_self"][0][0];
    let u = [1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()];
    assert_relative_eq!(Dot.dist(&u, &u).unwrap(), d, epsilon = 1e-14);

    let tri = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let tetra = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ];
    let pbc_pts = array![[0.05], [0.50], [0.95]];

    let pair_euclid_tri = scalar_col(&blocks["pair euclid_triangle"]);
    let rust_tri = upper_triangle(pairwise_euclid(tri.view()).unwrap().view());
    assert_eq!(pair_euclid_tri.len(), rust_tri.len());
    for (got, gold) in rust_tri.iter().zip(pair_euclid_tri.iter()) {
        assert_relative_eq!(*got, *gold, epsilon = PAIR_ABS, max_relative = PAIR_REL);
    }

    let pair_euclid_tet = scalar_col(&blocks["pair euclid_tetra"]);
    let rust_tet = upper_triangle(pairwise_euclid(tetra.view()).unwrap().view());
    assert_eq!(pair_euclid_tet.len(), rust_tet.len());
    for (got, gold) in rust_tet.iter().zip(pair_euclid_tet.iter()) {
        assert_relative_eq!(*got, *gold, epsilon = PAIR_ABS, max_relative = PAIR_REL);
    }

    let pair_pbc = scalar_col(&blocks["pair pbc_3pt"]);
    let rust_pbc = upper_triangle(pairwise(pbc_pts.view(), &pbc).unwrap().view());
    assert_eq!(pair_pbc.len(), rust_pbc.len());
    for (got, gold) in rust_pbc.iter().zip(pair_pbc.iter()) {
        assert_relative_eq!(*got, *gold, epsilon = PAIR_ABS, max_relative = PAIR_REL);
    }

    // NLDRProjection.p is private; compare invariant pairwise distances.
    check_mds_pairwise(&blocks, "mds torgerson_triangle", tri.view(), 2);
    check_mds_pairwise(&blocks, "mds torgerson_tetra", tetra.view(), 2);

    // Same 3-point χ that oracle.cpp dumps from NLDRITERChi.
    let hd = array![[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]];
    let coords = array![0.0, 0.0, 0.8, 0.1, -0.2, 0.7];
    let t = Transfer::xsigmoid(1.0, 4.0, 3.0).unwrap();

    for (key, imix, check_grad) in [
        ("chi xsig_1_4_3_imix01", 0.1, true),
        ("chi xsig_1_4_3_imix00", 0.0, false),
        ("chi identity_imix00", 0.0, false),
    ] {
        let rows = blocks
            .get(key)
            .unwrap_or_else(|| panic!("missing block {key}"));
        assert_eq!(rows.len(), 7, "{key} value + 6 grad");
        let tfun = if key.contains("identity") {
            Transfer::identity()
        } else {
            t.clone()
        };
        let mut f = hd.clone();
        apply_transfer(&mut f, &tfun);
        let ev = Stress::new(hd.clone(), f, tfun, imix, None, None).eval(coords.view(), 2);
        assert_relative_eq!(
            ev.value,
            rows[0][0],
            epsilon = CHI_ABS,
            max_relative = CHI_REL
        );
        // C++ NLDRITERChi imix==0 skips the /d_ij factor on gij; literature χ does not.
        if check_grad {
            for k in 0..6 {
                assert_relative_eq!(
                    ev.grad[k],
                    rows[1 + k][0],
                    epsilon = 1e-12,
                    max_relative = 1e-11
                );
            }
        }
    }

    let chi1 = scalar_col(&blocks["chi1 identity_square"]);
    assert_eq!(chi1.len(), 3, "chi1 value + 2 grad");
    let lm_ld = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let lm_hd = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ];
    let qhd = array![0.2, 0.1, 0.1];
    let xld = array![0.3, 0.2];
    let mut hd_row = ndarray::Array1::<f64>::zeros(4);
    let mut fhd_row = ndarray::Array1::<f64>::zeros(4);
    for i in 0..4 {
        hd_row[i] = Euclid
            .dist(qhd.as_slice().unwrap(), lm_hd.row(i).as_slice().unwrap())
            .unwrap();
        fhd_row[i] = Transfer::identity().f(hd_row[i]);
    }
    let w = ndarray::Array1::ones(4);
    let (vv, vg) = query_chi(
        xld.view(),
        lm_ld.view(),
        hd_row.view(),
        fhd_row.view(),
        &Transfer::identity(),
        0.0,
        w.view(),
    );
    assert_relative_eq!(vv, chi1[0], epsilon = CHI_ABS, max_relative = CHI_REL);
    assert_relative_eq!(vg[0], chi1[1], epsilon = 1e-12, max_relative = 1e-11);
    assert_relative_eq!(vg[1], chi1[2], epsilon = 1e-12, max_relative = 1e-11);

    let fps = scalar_col(&blocks["landmarks fps_square_k3"]);
    let square = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
    let lm = farthest_point(square.view(), &Euclid, 3, None, 0).unwrap();
    assert_eq!(lm.index.len(), 3);
    for (i, expected) in fps.iter().enumerate().take(3) {
        assert_eq!(lm.index[i] as f64, *expected);
    }
}

fn check_mds_pairwise(
    blocks: &BTreeMap<String, Vec<Vec<f64>>>,
    key: &str,
    pts: ArrayView2<f64>,
    lowdim: usize,
) {
    let gold = scalar_col(
        blocks
            .get(key)
            .unwrap_or_else(|| panic!("missing block {key}")),
    );
    let hd = pairwise_euclid(pts).unwrap();
    let (emb, _) = classical_mds(hd.view(), lowdim).unwrap();
    let got = upper_triangle(pairwise_euclid(emb.view()).unwrap().view());
    assert_eq!(got.len(), gold.len(), "{key} pair count");
    for (g, y) in got.iter().zip(gold.iter()) {
        assert_relative_eq!(*g, *y, epsilon = MDS_ABS, max_relative = MDS_REL);
    }
}

#[test]
fn one_e8_perturbation_of_a_golden_is_rejected() {
    let blocks = load_blocks();
    let gold = blocks["metric euclid_3_4_5"][0][0];
    let rust = Euclid.dist(&[0.0, 0.0, 0.0], &[3.0, 4.0, 0.0]).unwrap();
    assert!(
        match_abs_rel(rust, gold, METRIC_ABS, 0.0),
        "unperturbed euclid golden must match"
    );
    assert!(
        !match_abs_rel(rust, gold + 1e-8, METRIC_ABS, 0.0),
        "a 1e-8 shift of metric euclid_3_4_5 must fail the golden compare"
    );

    let chi_gold = blocks["chi xsig_1_4_3_imix01"][0][0];
    let hd = array![[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]];
    let coords = array![0.0, 0.0, 0.8, 0.1, -0.2, 0.7];
    let tfun = Transfer::xsigmoid(1.0, 4.0, 3.0).unwrap();
    let mut f = hd.clone();
    apply_transfer(&mut f, &tfun);
    let ev = Stress::new(hd, f, tfun, 0.1, None, None).eval(coords.view(), 2);
    assert!(match_abs_rel(ev.value, chi_gold, CHI_ABS, CHI_REL));
    assert!(
        !match_abs_rel(ev.value, chi_gold + 1e-8, CHI_ABS, CHI_REL),
        "a 1e-8 shift of chi xsig_1_4_3_imix01 must fail the golden compare"
    );

    let mds_gold = blocks["mds torgerson_triangle"][0][0];
    let tri = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let hd = pairwise_euclid(tri.view()).unwrap();
    let (emb, _) = classical_mds(hd.view(), 2).unwrap();
    let d01 = pairwise_euclid(emb.view()).unwrap()[(0, 1)];
    assert!(match_abs_rel(d01, mds_gold, MDS_ABS, MDS_REL));
    assert!(
        !match_abs_rel(d01, mds_gold + 1e-8, MDS_ABS, MDS_REL),
        "a 1e-8 shift of mds torgerson_triangle must fail the golden compare"
    );
}

#[test]
fn extras_are_not_the_default_solver() {
    // Halko / stoch / tensor stay extra arms; the default path is full-pair CG.
    let opts = IterOpts::default();
    assert!(matches!(opts.solver, Solver::Standard));
}
