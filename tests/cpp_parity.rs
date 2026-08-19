//! Compare landfold kernels to goldens dumped from the COSMO C++ (`interpol=false`).
//!
//! Goldens are produced by `scripts/gen_cpp_goldens.sh` on the remote builder.
//! This test fails if the file is missing: no silent paper-only stand-in.

use std::collections::BTreeMap;
use std::path::PathBuf;

use approx::assert_relative_eq;
use landfold::{Dot, Euclid, Metric, Periodic, Transfer};

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/cpp_oracle.txt")
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

#[test]
fn cpp_oracle_goldens_exist_and_match() {
    let path = golden_path();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "missing {}; run scripts/gen_cpp_goldens.sh on the remote builder",
            path.display()
        );
    });
    let blocks = parse_blocks(&text);
    assert!(
        blocks.contains_key("xfer xsigmoid_5_8_1"),
        "oracle did not dump xsigmoid_5_8_1"
    );

    let cases: &[(&str, Transfer)] = &[
        ("xfer identity", Transfer::identity()),
        ("xfer sigmoid_1", Transfer::sigmoid(1.0).unwrap()),
        ("xfer compress_1", Transfer::compress(1.0).unwrap()),
        ("xfer xsigmoid_5_8_1", Transfer::xsigmoid(5.0, 8.0, 1.0).unwrap()),
        ("xfer xsigmoid_6_8_8", Transfer::xsigmoid(6.0, 8.0, 8.0).unwrap()),
        ("xfer warp_5_8_1_2_2", Transfer::warp(5.0, 8.0, 1.0, 2.0, 2.0).unwrap()),
    ];
    for (key, t) in cases {
        let rows = blocks.get(*key).unwrap_or_else(|| panic!("missing block {key}"));
        for row in rows {
            assert_eq!(row.len(), 3, "{key} row");
            let x = row[0];
            let (f, df) = t.fdf(x);
            assert_relative_eq!(f, row[1], epsilon = 1e-14, max_relative = 1e-13);
            if x != 0.0 {
                assert_relative_eq!(df, row[2], epsilon = 1e-13, max_relative = 1e-12);
            }
        }
    }

    let e = blocks["metric euclid_3_4_5"][0][0];
    assert_relative_eq!(
        Euclid.dist(&[0.0, 0.0, 0.0], &[3.0, 4.0, 0.0]).unwrap(),
        e,
        epsilon = 1e-15
    );
    let p = blocks["metric pbc_wrap"][0][0];
    let pbc = Periodic::isotropic(1, 1.0);
    assert_relative_eq!(pbc.dist(&[0.05], &[0.95]).unwrap(), p, epsilon = 1e-15);
    let d = blocks["metric dot_self"][0][0];
    let u = [1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()];
    assert_relative_eq!(Dot.dist(&u, &u).unwrap(), d, epsilon = 1e-14);
}
