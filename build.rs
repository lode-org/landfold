use std::path::Path;
use std::process::Command;

fn valid_revision(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn locked_eindir_revision(manifest_dir: &Path) -> Option<String> {
    let lockfile = manifest_dir.join("Cargo.lock");
    let mut package = None;
    for line in std::fs::read_to_string(lockfile).ok()?.lines() {
        if line == "name = \"eindir-core\"" {
            package = Some(());
            continue;
        }
        if package.is_some()
            && let Some(source) =
                line.strip_prefix("source = \"git+https://github.com/HaoZeke/eindir.git")
            && let Some(revision) = source
                .split('#')
                .nth(1)
                .map(|value| value.trim_end_matches('"'))
            && valid_revision(revision)
        {
            return Some(revision.to_owned());
        }
        if line == "[[package]]" {
            package = None;
        }
    }
    None
}

fn main() {
    println!("cargo:rerun-if-env-changed=LANDFOLD_EINDIR_REVISION");
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=../eindir/.git/HEAD");
    let sibling = Path::new(env!("CARGO_MANIFEST_DIR")).join("../eindir");
    let head_path = sibling.join(".git/HEAD");
    if let Ok(head) = std::fs::read_to_string(&head_path)
        && let Some(reference) = head.strip_prefix("ref: ").map(str::trim)
    {
        println!("cargo:rerun-if-changed=../eindir/.git/{reference}");
    }
    let revision = match std::env::var("LANDFOLD_EINDIR_REVISION") {
        Ok(value) if valid_revision(&value) => Some(value),
        Ok(_) => panic!("LANDFOLD_EINDIR_REVISION must be a 40-digit hexadecimal commit"),
        Err(_) => None,
    }
    .or_else(|| {
        sibling.exists().then(|| {
            Command::new("git")
                .args([
                    "-C",
                    sibling.to_str().unwrap_or_default(),
                    "rev-parse",
                    "HEAD",
                ])
                .output()
                .ok()
                .filter(|output| output.status.success())
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .map(|value| value.trim().to_owned())
                .filter(|value| valid_revision(value))
        })?
    })
    .or_else(|| locked_eindir_revision(Path::new(env!("CARGO_MANIFEST_DIR"))))
    .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=LANDFOLD_EINDIR_REVISION={revision}");
}
