use sqllogictest::Runner;

// Reuse the existing EngineHarness implementation in slt.rs by declaring a module.
mod slt;

#[tokio::test]
async fn run_all_slt_files() {
    let base_dir = std::path::PathBuf::from("tests/slt");

    // Collect .slt files recursively, sort them for deterministic order.
    let mut slt_files: Vec<std::path::PathBuf> = Vec::new();
    fn collect(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let path = e.path();
                if path.is_dir() {
                    collect(&path, out);
                } else if path.extension().and_then(|s| s.to_str()) == Some("slt") {
                    out.push(path);
                }
            }
        }
    }

    collect(&base_dir, &mut slt_files);
    slt_files.sort();

    for path in slt_files {
        // Show progress: which .slt file is running.
        println!("Running slt: {}", path.display());
        // Create a fresh runner for each file so that each script gets a clean DB state.
        let mut runner = Runner::new(|| async { Ok(slt::EngineHarness::new()) });
        match runner.run_file_async(&path).await {
            Ok(()) => println!("  PASS: {}", path.display()),
            Err(e) => panic!("  FAIL: {}: {e}", path.display()),
        }
    }
}
