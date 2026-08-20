use std::path::Path;
use std::process::Command;

fn valid_revision(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn main() {
    println!("cargo:rerun-if-env-changed=LANDFOLD_EINDIR_REVISION");
    let revision = std::env::var("LANDFOLD_EINDIR_REVISION")
        .ok()
        .filter(|value| valid_revision(value))
        .or_else(|| {
            let sibling = Path::new(env!("CARGO_MANIFEST_DIR")).join("../eindir");
            sibling.exists().then(|| {
                Command::new("git")
                    .args(["-C", sibling.to_str().unwrap_or_default(), "rev-parse", "HEAD"])
                    .output()
                    .ok()
                    .filter(|output| output.status.success())
                    .and_then(|output| String::from_utf8(output.stdout).ok())
                    .map(|value| value.trim().to_owned())
                    .filter(|value| valid_revision(value))
            })?
        })
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=LANDFOLD_EINDIR_REVISION={revision}");
}
