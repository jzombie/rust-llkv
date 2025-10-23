use libtest_mimic::Arguments;
use llkv_slt_tester as slt;
use llkv_test_utils::init_tracing_for_tests;

const SLT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/slt");

fn main() {
    init_tracing_for_tests();
    let args = Arguments::from_args();
    let conclusion = slt::run_slt_harness_with_args(
        SLT_DIR,
        slt::engine::make_in_memory_factory_factory(),
        args,
    );
    conclusion.exit();
}
