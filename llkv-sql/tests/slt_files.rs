use std::sync::Arc;

use llkv_storage::pager::MemPager;
use sqllogictest::Runner;

use crate::slt::EngineHarness;

// Reuse the existing EngineHarness implementation in slt.rs by declaring a module.
mod slt;

#[tokio::test]
async fn run_all_slt_files() {
    let mut runner = Runner::new(|| async { Ok(slt::EngineHarness::new()) });

    let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.push("tests/slt");

    let entries = std::fs::read_dir(dir).expect("read slt dir");
    for entry in entries {
        let path = entry.expect("read entry").path();
        if path.extension().and_then(|s| s.to_str()) == Some("slt") {
            runner
                .run_file_async(&path)
                .await
                .expect(&format!("running {} failed", path.display()));
        }
    }
}
