use libtest_mimic::Arguments;
use llkv_test_utils::init_tracing_for_tests;
use llkv_sql::slt;

const SLT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/slt");

mod slt_engine;

fn main() {
    init_tracing_for_tests();
    let args = Arguments::from_args();
    let conclusion =
        slt::run_slt_harness_with_args(SLT_DIR, slt_engine::make_factory_factory(), args);
    conclusion.exit();
}
