//! Manifest and lockfile pin: one pyo3 major with `--features python`.
//!
//! dlpk 0.4.1 optionally pins pyo3 0.29. landfold matches that major
//! and does not turn on `dlpk/pyo3`, so `pyo3-ffi` `links = "python"`
//! appears once. The graph check is `cargo tree -e features --features
//! python` on the remote builder (`scripts/check_pyo3_pin.sh`).

use std::collections::BTreeSet;
use std::path::PathBuf;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn golden(name: &str) -> PathBuf {
    crate_root().join("tests/goldens").join(name)
}

fn lock_packages(text: &str) -> Vec<(String, String, Vec<String>)> {
    let mut out = Vec::new();
    let mut name = String::new();
    let mut version = String::new();
    let mut deps: Vec<String> = Vec::new();
    let mut in_deps = false;
    let flush = |name: &mut String,
                 version: &mut String,
                 deps: &mut Vec<String>,
                 out: &mut Vec<(String, String, Vec<String>)>| {
        if !name.is_empty() {
            out.push((name.clone(), version.clone(), deps.clone()));
        }
        name.clear();
        version.clear();
        deps.clear();
    };
    for line in text.lines() {
        if line == "[[package]]" {
            flush(&mut name, &mut version, &mut deps, &mut out);
            in_deps = false;
            continue;
        }
        if line == "dependencies = [" {
            in_deps = true;
            continue;
        }
        if in_deps {
            if line == "]" {
                in_deps = false;
            } else if let Some(dep) = line
                .trim()
                .trim_start_matches('"')
                .split(|c: char| c == '"' || c.is_whitespace())
                .next()
                .filter(|dep| !dep.is_empty())
            {
                deps.push(dep.to_string());
            }
            continue;
        }
        if let Some(v) = line.strip_prefix("name = \"") {
            name = v.trim_end_matches('"').to_string();
        } else if let Some(v) = line.strip_prefix("version = \"") {
            if name.is_empty() {
                continue;
            }
            version = v.trim_end_matches('"').to_string();
        }
    }
    flush(&mut name, &mut version, &mut deps, &mut out);
    out
}

fn toml_quoted(text: &str, key: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(&format!("{key} = ")) {
            if let Some(v) = rest.strip_prefix('"') {
                return Some(v.split('"').next().unwrap_or("").to_string());
            }
            if let Some(inner) = rest.strip_prefix('{') {
                for part in inner.split(',') {
                    let part = part.trim();
                    if let Some(v) = part.strip_prefix("version = \"") {
                        return Some(v.split('"').next().unwrap_or("").to_string());
                    }
                }
            }
        }
    }
    None
}

fn toml_python_feature(text: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("python = ") {
            return Some(rest.to_string());
        }
    }
    None
}

fn golden_field(text: &str, key: &str) -> String {
    text.lines()
        .find_map(|l| l.strip_prefix(&format!("{key} ")))
        .unwrap_or_else(|| panic!("missing {key} in pyo3_majors.txt"))
        .to_string()
}

#[test]
fn python_feature_matches_dlpk_pyo3_major() {
    let want = std::fs::read_to_string(golden("pyo3_majors.txt")).unwrap();
    let pyo3_major = golden_field(&want, "pyo3");
    let numpy_major = golden_field(&want, "numpy");
    let dlpk_ver = golden_field(&want, "dlpk");
    assert_eq!(golden_field(&want, "dlpk.pyo3"), "off");

    let toml = std::fs::read_to_string(crate_root().join("Cargo.toml")).unwrap();
    let pyo3 = toml_quoted(&toml, "pyo3").expect("pyo3 pin in Cargo.toml");
    let numpy = toml_quoted(&toml, "numpy").expect("numpy pin in Cargo.toml");
    let dlpk = toml_quoted(&toml, "dlpk").expect("dlpk pin in Cargo.toml");
    assert!(
        pyo3.starts_with(&pyo3_major),
        "Cargo.toml pyo3 {pyo3} must stay on dlpk major {pyo3_major}"
    );
    assert!(
        numpy.starts_with(&numpy_major),
        "Cargo.toml numpy {numpy} must stay on dlpk major {numpy_major}"
    );
    assert_eq!(dlpk, dlpk_ver);
    let python = toml_python_feature(&toml).expect("python feature");
    assert!(
        python.contains("dep:pyo3") && python.contains("dep:numpy"),
        "python feature must pull landfold's pyo3/numpy: {python}"
    );
    assert!(
        !python.contains("dlpk/pyo3"),
        "dlpk pyo3 feature stays off so pyo3-ffi links=python is unique: {python}"
    );

    let lock = std::fs::read_to_string(crate_root().join("Cargo.lock")).unwrap();
    let pkgs = lock_packages(&lock);
    let pyo3_vers: BTreeSet<_> = pkgs
        .iter()
        .filter(|(n, _, _)| n == "pyo3")
        .map(|(_, v, _)| v.clone())
        .collect();
    assert_eq!(
        pyo3_vers.len(),
        1,
        "Cargo.lock must resolve one pyo3, got {pyo3_vers:?}"
    );
    let locked = pyo3_vers.iter().next().unwrap();
    assert!(
        locked.starts_with(&format!("{pyo3_major}.")),
        "locked pyo3 {locked} is not major {pyo3_major}"
    );
    let numpy_vers: BTreeSet<_> = pkgs
        .iter()
        .filter(|(n, _, _)| n == "numpy")
        .map(|(_, v, _)| v.clone())
        .collect();
    assert_eq!(numpy_vers.len(), 1, "one numpy, got {numpy_vers:?}");
    assert!(
        numpy_vers
            .iter()
            .next()
            .unwrap()
            .starts_with(&format!("{numpy_major}.")),
        "locked numpy {numpy_vers:?} is not major {numpy_major}"
    );
    let dlpk_pkg = pkgs
        .iter()
        .find(|(n, v, _)| n == "dlpk" && v == &dlpk_ver)
        .unwrap_or_else(|| panic!("lock missing dlpk {dlpk_ver}"));
    assert!(
        !dlpk_pkg.2.iter().any(|d| d == "pyo3"),
        "dlpk {dlpk_ver} must not pull pyo3 when its pyo3 feature is off: {:?}",
        dlpk_pkg.2
    );
}

#[test]
fn xtsci_dependency_is_immutably_pinned() {
    let manifest = std::fs::read_to_string(crate_root().join("Cargo.toml")).unwrap();
    assert!(manifest.contains("git = \"https://github.com/HaoZeke/xtsci-optimize.git\""));
    assert!(manifest.contains("rev = \"d7849538b85fa25a1118f9a22c379e311cb26a3c\""));
}

#[test]
fn default_solver_stays_standard() {
    assert!(matches!(
        landfold::IterOpts::default().solver,
        landfold::Solver::Standard
    ));
}
