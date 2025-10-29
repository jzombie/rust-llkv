use criterion::{Criterion, criterion_group, criterion_main};
use llkv_slt_tester::LlkvSltRunner;
use std::path::PathBuf;

// This particular test performance will degrade significantly if the underlying
// optimizations are not working correctly, since it relies on hash joins to
// efficiently execute the large multi-join query.
const SELECT4_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/slt/sqlite/select4.slturl"
);

fn bench_select4_slturl(c: &mut Criterion) {
    let mut group = c.benchmark_group("select4_slturl");

    // Configure for minimal iterations since each run takes 12-13 seconds
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(100));
    group.measurement_time(std::time::Duration::from_secs(123));

    group.bench_function("select4_full", |b| {
        b.iter(|| {
            let runner = LlkvSltRunner::in_memory();
            let path = PathBuf::from(SELECT4_PATH);
            runner.run_file(&path).expect("SLT test failed");
        });
    });

    group.finish();
}

criterion_group!(benches, bench_select4_slturl);
criterion_main!(benches);
