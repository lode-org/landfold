use std::path::Path;
use std::process::Command;

fn valid_revision(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn main() {
    println!("cargo:rerun-if-env-changed=LANDFOLD_EINDIR_REVISION");
    println!("cargo:rerun-if-changed=../eindir/.git/HEAD");
    let revision = match std::env::var("LANDFOLD_EINDIR_REVISION") {
        Ok(value) if valid_revision(&value) => Some(value),
        Ok(_) => panic!("LANDFOLD_EINDIR_REVISION must be a 40-digit hexadecimal commit"),
        Err(_) => None,
    }
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
