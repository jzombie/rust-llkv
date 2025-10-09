use llkv_test_utils::slt;

#[test]
fn slt_harness() {
    // Create an engine factory closure that will be passed to the shared
    // harness. Keep this file minimal — the real harness lives in
    // `llkv-test-utils`.
    use llkv_sql::SqlEngine;
    use llkv_storage::pager::MemPager;
    use std::sync::Arc;

    struct EngineFactory;
    impl EngineFactory {
        async fn make() -> Result<impl sqllogictest::AsyncDB<Error = llkv_result::Error> + Send + 'static, ()> {
            let pager = Arc::new(MemPager::default());
            let engine = SqlEngine::new(pager);
            // Wrap engine in the same adapter used previously in this crate's tests.
            Ok(crate::tests::slt_engine_adapter::Adapter::new(engine))
        }
    }

    // Delegate to the test-utils harness. The harness is behind the `harness`
    // feature and provides discovery + libtest_mimic wiring.
    slt::run_slt_harness("tests/slt", || EngineFactory::make());
}

// Unit tests for the mapping/expansion helpers live in `llkv-test-utils`.
