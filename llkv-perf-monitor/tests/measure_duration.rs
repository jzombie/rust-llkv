use std::time::Duration;

use llkv_perf_monitor::{measure_with_duration, PerfContext};

#[test]
fn returns_duration_even_when_perf_feature_disabled() {
    let ctx = PerfContext::disabled();

    let (result, elapsed) = measure_with_duration!(
        [],
        ctx,
        "sleep",
        {
            std::thread::sleep(Duration::from_millis(1));
            42
        }
    );

    assert_eq!(result, 42);
    assert!(elapsed > Duration::ZERO, "duration should be captured even when perf-mon is off");
    assert!(ctx.measurements().is_empty(), "disabled contexts must not record measurements");
}

#[test]
fn returns_duration_when_feature_list_not_matched() {
    let ctx = PerfContext::disabled();

    let (result, elapsed) = measure_with_duration!(
        ["never-on"],
        ctx,
        "sleep",
        {
            std::thread::sleep(Duration::from_millis(1));
            7
        }
    );

    assert_eq!(result, 7);
    assert!(elapsed > Duration::ZERO, "duration should be captured even when the listed feature is disabled");
    assert!(ctx.measurements().is_empty(), "no measurements should be recorded when context is disabled or feature is off");
}
