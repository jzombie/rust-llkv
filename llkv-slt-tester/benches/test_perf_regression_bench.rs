use criterion::{Criterion, criterion_group, criterion_main};
use llkv_slt_tester::LlkvSltRunner;
use std::collections::BTreeMap;
use std::path::PathBuf;

// These test performances will degrade significantly if the underlying
// optimizations are not working correctly.

// ETA: Roughtly 12-14 seconds
const SELECT4_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/slt/sqlite/select4.slturl"
);
// ETA: Roughtly .5 ms
const DELETE1000_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/slt/sqlite/index/delete/1000/slt_good_0.slturl"
);

fn bench_slturl_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("slturl_cases");

    // Configure for minimal iterations since each run takes 12-13 seconds
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(1));
    group.measurement_time(std::time::Duration::from_secs(12));

    let paths: BTreeMap<&str, &str> = [
        ("select4_full", SELECT4_PATH),
        ("delete1000_full", DELETE1000_PATH),
    ]
    .into();

    for (label, path) in paths {
        group.bench_function(label, |b| {
            b.iter(|| {
                let runner = LlkvSltRunner::in_memory();
                let path = PathBuf::from(path);
                runner.run_file(&path).expect("SLT test failed");
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_slturl_cases);
criterion_main!(benches);
