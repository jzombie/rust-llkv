use libtest_mimic::Arguments;
use llkv_slt_tester as slt;
use llkv_test_utils::init_tracing_for_tests;

const SLT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/slt");

/// Note: For easier debugging you can run SQL directly against the main binary.
/// ```bash
/// cargo run -p llkv -- <<'SQL'
/// CREATE TABLE tab2(col0 INTEGER, col1 INTEGER, col2 INTEGER);
/// INSERT INTO tab2 VALUES (7,31,27);
/// INSERT INTO tab2 VALUES (79,17,38);
/// INSERT INTO tab2 VALUES (78,59,26);
/// SELECT DISTINCT col2 FROM tab2 ORDER BY col2;
/// SQL
/// ```
fn main() {
    init_tracing_for_tests();
    let args = Arguments::from_args();
    let conclusion = slt::run_slt_harness_with_args(
        SLT_DIR,
        slt::slt_test_engine::make_in_memory_factory_factory(),
        args,
    );
    conclusion.exit();
}
