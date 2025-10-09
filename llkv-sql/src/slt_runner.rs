// Minimal test-only wrapper that delegates to the shared helpers in
// `llkv-test-utils`. The engine-specific adapter remains here so the
// shared crate stays engine-agnostic.

#![cfg(test)]

use std::path::Path;
use std::sync::Arc;

use crate::SqlEngine;
use crate::SqlResult;
use llkv_storage::pager::MemPager;

/// Run a single slt file by delegating to `llkv-test-utils`.
pub async fn run_slt_file<P: AsRef<Path>>(path: P) -> SqlResult<()> {
    // Engine adapter factory: construct the EngineHarness used by tests.
    struct EngineHarness {
        engine: SqlEngine<MemPager>,
    }
    impl EngineHarness {
        pub fn new() -> Self {
            let pager = Arc::new(MemPager::default());
            Self {
                engine: SqlEngine::new(pager),
            }
        }
    }

    // Delegate to the shared implementation in llkv-test-utils. We construct
    // a factory closure which creates the engine adapter for each run.
    llkv_test_utils::slt::run_slt_file_with_factory(path.as_ref(), || async {
        Ok::<_, ()>(EngineHarness::new())
    })
    .await
}
