use indoc::indoc;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::tempdir;

// When perf monitoring is disabled, the wrapper should be a no-op and produce identical assembly.
#[test]
fn assembly_matches_when_feature_disabled() {
    let dir = tempdir().expect("tempdir");
    let manifest = dir.path().join("Cargo.toml");
    let src_bin = dir.path().join("src").join("bin");
    fs::create_dir_all(&src_bin).expect("mkdirs");

    let perf_path = Path::new(env!("CARGO_MANIFEST_DIR"));

    write_manifest(&manifest, perf_path, false);
    write_bins(&src_bin, false);

    let target_dir = dir.path().join("target");

    build_asm(dir.path(), "with", &target_dir, false);
    build_asm(dir.path(), "plain", &target_dir, false);

    let with_s = find_asm(&target_dir, "with").expect("with asm");
    let plain_s = find_asm(&target_dir, "plain").expect("plain asm");

    let with_norm = normalize_asm(&fs::read_to_string(with_s).expect("read with.s"));
    let plain_norm = normalize_asm(&fs::read_to_string(plain_s).expect("read plain.s"));

    assert_eq!(
        with_norm, plain_norm,
        "assembly should match when perf-mon is disabled"
    );
}

#[test]
fn assembly_differs_when_perf_enabled() {
    let dir = tempdir().expect("tempdir");
    let manifest = dir.path().join("Cargo.toml");
    let src_bin = dir.path().join("src").join("bin");
    fs::create_dir_all(&src_bin).expect("mkdirs");

    let perf_path = Path::new(env!("CARGO_MANIFEST_DIR"));

    write_manifest(&manifest, perf_path, true);
    write_bins(&src_bin, true);

    let target_dir = dir.path().join("target");

    build_asm(dir.path(), "with", &target_dir, true);
    build_asm(dir.path(), "plain", &target_dir, true);

    let with_s = find_asm(&target_dir, "with").expect("with asm");
    let plain_s = find_asm(&target_dir, "plain").expect("plain asm");

    let with_norm = normalize_asm(&fs::read_to_string(with_s).expect("read with.s"));
    let plain_norm = normalize_asm(&fs::read_to_string(plain_s).expect("read plain.s"));

    assert_ne!(
        with_norm, plain_norm,
        "assembly should differ when perf-mon is enabled"
    );
}

fn build_asm(workspace: &Path, bin: &str, target_dir: &Path, enable_perf: bool) {
    let mut cmd = Command::new("cargo");
    cmd.current_dir(workspace)
        .arg("rustc")
        .arg("--bin")
        .arg(bin)
        .arg("--release");

    if enable_perf {
        cmd.arg("--features").arg("perf-mon");
    }

    let status = cmd
        .arg("--")
        .arg("--emit=asm")
        .arg("-Copt-level=3")
        .arg("-Cdebuginfo=0")
        .arg("-Ccodegen-units=1")
        .arg("-Cstrip=symbols")
        .env("CARGO_TARGET_DIR", target_dir)
        .status()
        .expect("spawn cargo rustc");
    assert!(status.success(), "cargo rustc failed for bin {bin}");
}

fn find_asm(target_dir: &Path, stem: &str) -> Option<PathBuf> {
    let deps = target_dir.join("release").join("deps");
    fs::read_dir(&deps)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| {
            p.extension().map(|e| e == "s").unwrap_or(false)
                && p.file_stem()
                    .map(|s| s.to_string_lossy().starts_with(stem))
                    .unwrap_or(false)
        })
}

fn normalize_asm(src: &str) -> String {
    extract_opcodes(src).join("\n")
}

fn write_manifest(manifest: &Path, perf_path: &Path, enable_perf: bool) {
    let dep = if enable_perf {
        format!(
            "llkv-perf-monitor = {{ path = \"{}\", features = [\"perf-mon\"] }}",
            perf_path.display()
        )
    } else {
        format!(
            "llkv-perf-monitor = {{ path = \"{}\" }}",
            perf_path.display()
        )
    };

    fs::write(
        manifest,
        format!(
            indoc!(
                r#"
                [package]
                name = "asm-check"
                version = "0.0.0"
                edition = "2021"

                [features]
                perf-mon = ["llkv-perf-monitor/perf-mon"]

                [dependencies]
                {dep}
            "#
            ),
            dep = dep,
        ),
    )
    .expect("write Cargo.toml");
}

fn write_bins(src_bin: &Path, enable_perf: bool) {
    let with_body = if enable_perf {
        indoc!(
            r#"
            use llkv_perf_monitor::{maybe_record, PerfContext};

            fn main() {
                let ctx = PerfContext::new("root");
                maybe_record!(["perf-mon"], ctx, "hello", { println!("hello world"); });
            }
            "#
        )
    } else {
        indoc!(
            r#"
            use llkv_perf_monitor::{maybe_record, PerfContext};

            fn main() {
                let ctx = PerfContext::default();
                maybe_record!(["perf-mon"], ctx, "hello", { println!("hello world"); });
            }
            "#
        )
    };

    fs::write(src_bin.join("with.rs"), with_body).expect("write with.rs");

    fs::write(
        src_bin.join("plain.rs"),
        indoc!(
            r#"
            fn main() {
                println!("hello world");
            }
            "#
        ),
    )
    .expect("write plain.rs");
}

// Extract opcode sequence from instructions, skipping labels and directives so
// we compare only the executed instruction stream, which is stable across
// platforms when the compiled code is identical.
fn extract_opcodes(src: &str) -> Vec<String> {
    src.lines()
        .map(|line| line.trim())
        .filter(|line| {
            !(line.is_empty()
                || line.starts_with('.')
                || line.starts_with('#')
                || line.ends_with(':'))
        })
        .map(|line| line.split_whitespace().next().unwrap_or("").to_string())
        .filter(|op| !op.is_empty())
        .collect()
}
