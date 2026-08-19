//! Out-of-sample projection, farthest-point landmarks, and FES invert.

use std::path::PathBuf;
use std::process::Command;

use approx::assert_relative_eq;
use landfold::{
    coordination_histogram, embed_points, farthest_point, project_many, project_one, Euclid,
    FreeEnergy, Histogram2d, IterOpts, ProjOpts, Transfer,
};
use ndarray::array;

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens")
        .join(name)
}

fn landfold_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_landfold"))
}

fn scratch(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "landfold-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

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

#[test]
fn proj_opts_from_cli_parses_grid() {
    let p = ProjOpts::from_cli("2.5,11,41").unwrap();
    assert_relative_eq!(p.gridw, 2.5);
    assert_eq!(p.grid_coarse, 11);
    assert_eq!(p.grid_fine, 41);
    assert_eq!(p.cg_steps, 0);
    assert!(ProjOpts::from_cli("1,2").is_err());
    assert!(ProjOpts::from_cli("nope").is_err());
}

#[test]
fn farthest_point_matches_square_golden() {
    let pts = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5]
    ];
    let lm = farthest_point(pts.view(), &Euclid, 3, None, 0).unwrap();
    let text = std::fs::read_to_string(golden("landmarks_square.txt")).unwrap();
    let want: Vec<usize> = text
        .lines()
        .find(|l| !l.starts_with('#') && !l.trim().is_empty())
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(lm.index, want);
}

#[test]
fn farthest_point_rejects_empty_or_too_many() {
    let pts = array![[0.0, 0.0], [1.0, 0.0]];
    assert!(farthest_point(pts.view(), &Euclid, 0, None, 0).is_err());
    assert!(farthest_point(pts.view(), &Euclid, 3, None, 0).is_err());
}

#[test]
fn farthest_point_weight_pulls_a_near_point() {
    let pts = array![[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]];
    let w = array![1.0, 1000.0, 1.0];
    let lm = farthest_point(pts.view(), &Euclid, 2, Some(w.view()), 0).unwrap();
    assert_eq!(lm.index[0], 0);
    assert_eq!(lm.index[1], 1);
}

#[test]
fn project_many_matches_project_one() {
    let pts = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut opts = IterOpts::default();
    opts.lowdim = 2;
    opts.tfun_hd = Transfer::identity();
    opts.tfun_ld = Transfer::identity();
    opts.cg.maxiter = 20;
    let emb = embed_points(pts.view(), &Euclid, &opts).unwrap();
    let po = ProjOpts {
        gridw: 2.0,
        grid_coarse: 11,
        grid_fine: 21,
        cg_steps: 4,
    };
    let many = project_many(&emb, pts.view(), &Euclid, &po).unwrap();
    for i in 0..pts.nrows() {
        let one = project_one(&emb, pts.row(i), &Euclid, &po).unwrap();
        assert_relative_eq!(many[(i, 0)], one[0], epsilon = 1e-12);
        assert_relative_eq!(many[(i, 1)], one[1], epsilon = 1e-12);
    }
}

#[test]
fn project_without_grid_uses_nearest_landmark() {
    let pts = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut opts = IterOpts::default();
    opts.lowdim = 2;
    opts.tfun_hd = Transfer::identity();
    opts.tfun_ld = Transfer::identity();
    opts.cg.maxiter = 10;
    let emb = embed_points(pts.view(), &Euclid, &opts).unwrap();
    let po = ProjOpts {
        gridw: 1.0,
        grid_coarse: 1,
        grid_fine: 1,
        cg_steps: 0,
    };
    let q = project_one(&emb, pts.row(0), &Euclid, &po).unwrap();
    assert_relative_eq!(q[0], emb.low[(0, 0)], epsilon = 1e-15);
    assert_relative_eq!(q[1], emb.low[(0, 1)], epsilon = 1e-15);
}

#[test]
fn fes_csv_matches_half_density_golden() {
    let mut h = Histogram2d::new(0.0, 2.0, 2, 0.0, 2.0, 2).unwrap();
    h.add(0.5, 0.5, 1.0);
    h.add(0.5, 0.5, 1.0);
    h.add(1.5, 1.5, 1.0);
    let fes = FreeEnergy::from_histogram(&h, 1.0);
    let mut got = Vec::new();
    fes.write_csv(&mut got).unwrap();
    let want = std::fs::read_to_string(golden("fes_half_density.csv")).unwrap();
    assert_eq!(String::from_utf8(got).unwrap(), want);
    assert_relative_eq!(fes.f[(1, 1)].unwrap(), 2.0_f64.ln(), epsilon = 1e-12);
}

#[test]
fn fes_svg_is_a_heatmap() {
    let mut h = Histogram2d::new(0.0, 2.0, 2, 0.0, 2.0, 2).unwrap();
    h.add(0.5, 0.5, 1.0);
    let fes = FreeEnergy::from_histogram(&h, 1.0);
    let mut svg = Vec::new();
    fes.write_svg(&mut svg, 80, 60).unwrap();
    let s = String::from_utf8(svg).unwrap();
    assert!(s.contains("<svg"));
    assert!(s.contains("</svg>"));
    assert!(s.contains("rgb("));
}

#[test]
fn coordination_histogram_matches_dimer_golden() {
    let frame = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
    let h = coordination_histogram(frame.view(), 1.5, 12);
    let want = std::fs::read_to_string(golden("cn_dimer.csv")).unwrap();
    let mut got = Vec::new();
    h.write_cn_csv(&mut got).unwrap();
    assert_eq!(String::from_utf8(got).unwrap(), want);
    assert_eq!(h.counts[0], 1.0);
    assert_eq!(h.counts[1], 2.0);
}

#[test]
fn cli_landmarks_writes_indices_and_k_rows() {
    let dir = scratch("lm");
    let input = dir.join("pts.dat");
    std::fs::write(&input, "0 0\n1 0\n0 1\n1 1\n0.5 0.5\n").unwrap();
    let out = Command::new(landfold_bin())
        .args(["landmarks", "-D", "2", "-n", "3", "--seed", "0", "--indices"])
        .stdin(std::fs::File::open(&input).unwrap())
        .output()
        .unwrap();
    assert!(out.status.success(), "stderr {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(
        stdout.lines().next().unwrap().starts_with("# indices 0 3 1"),
        "{stdout}"
    );
    let rows: Vec<_> = stdout
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .collect();
    assert_eq!(rows.len(), 3);
}

#[test]
fn cli_fes_csv_svg_and_cn() {
    let dir = scratch("fes");
    let ld = dir.join("ld.dat");
    let frames = dir.join("xyz.dat");
    let csv = dir.join("fes.csv");
    let svg = dir.join("fes.svg");
    let cn = dir.join("cn.csv");
    std::fs::write(&ld, "0.5 0.5\n0.5 0.5\n1.5 1.5\n").unwrap();
    std::fs::write(&frames, "0 0 0\n1 0 0\n10 0 0\n").unwrap();
    let out = Command::new(landfold_bin())
        .args([
            "fes",
            "--input",
            ld.to_str().unwrap(),
            "--nx",
            "2",
            "--ny",
            "2",
            "--xmin",
            "0",
            "--xmax",
            "2",
            "--ymin",
            "0",
            "--ymax",
            "2",
            "--csv",
            csv.to_str().unwrap(),
            "--svg",
            svg.to_str().unwrap(),
            "--frames",
            frames.to_str().unwrap(),
            "--cn-csv",
            cn.to_str().unwrap(),
            "--cn-cutoff",
            "1.5",
            "--cn-max",
            "12",
        ])
        .output()
        .unwrap();
    assert!(out.status.success(), "stderr {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(
        std::fs::read_to_string(&csv).unwrap(),
        std::fs::read_to_string(golden("fes_half_density.csv")).unwrap()
    );
    let svg_txt = std::fs::read_to_string(&svg).unwrap();
    assert!(svg_txt.contains("<svg"));
    assert_eq!(
        std::fs::read_to_string(&cn).unwrap(),
        std::fs::read_to_string(golden("cn_dimer.csv")).unwrap()
    );
}

#[test]
fn cli_project_lands_on_a_known_landmark() {
    let dir = scratch("proj");
    let hi = dir.join("hi.dat");
    let lo = dir.join("lo.dat");
    let q = dir.join("q.dat");
    std::fs::write(&hi, "0 0\n1 0\n0 1\n").unwrap();
    std::fs::write(&lo, "0 0\n1 0\n0 1\n").unwrap();
    std::fs::write(&q, "0 0\n").unwrap();
    let out = Command::new(landfold_bin())
        .args([
            "project",
            "-D",
            "2",
            "-d",
            "2",
            "--high-file",
            hi.to_str().unwrap(),
            "--low-file",
            lo.to_str().unwrap(),
            "--grid",
            "2.0,11,21",
            "--refine",
            "4",
        ])
        .stdin(std::fs::File::open(&q).unwrap())
        .output()
        .unwrap();
    assert!(out.status.success(), "stderr {}", String::from_utf8_lossy(&out.stderr));
    let nums: Vec<f64> = String::from_utf8(out.stdout)
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(nums.len(), 2);
    let d = (nums[0] * nums[0] + nums[1] * nums[1]).sqrt();
    assert!(d < 0.15, "projected origin landed at {nums:?}");
}

#[test]
fn cli_landmarks_voronoi_weights_sum_to_one() {
    let dir = scratch("lm-voro");
    let input = dir.join("pts.dat");
    std::fs::write(&input, "0 0\n0.1 0\n0.2 0\n10 0\n").unwrap();
    let out = Command::new(landfold_bin())
        .args([
            "landmarks",
            "-D",
            "2",
            "-n",
            "2",
            "--seed",
            "0",
            "--voronoi",
            "--indices",
        ])
        .stdin(std::fs::File::open(&input).unwrap())
        .output()
        .unwrap();
    assert!(out.status.success(), "stderr {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("# indices 0 3"), "{stdout}");
    let mut wsum = 0.0;
    for line in stdout.lines().filter(|l| !l.starts_with('#')) {
        let last = line.split_whitespace().last().unwrap().parse::<f64>().unwrap();
        wsum += last;
    }
    assert!((wsum - 1.0).abs() < 1e-12, "voronoi weights {wsum}");
}

#[test]
fn cli_cn_csv_without_frames_fails() {
    let dir = scratch("cn-err");
    let ld = dir.join("ld.dat");
    std::fs::write(&ld, "0 0\n1 1\n").unwrap();
    let out = Command::new(landfold_bin())
        .args([
            "fes",
            "--input",
            ld.to_str().unwrap(),
            "--cn-csv",
            dir.join("cn.csv").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
}
