use std::sync::Arc;

use arrow::array::Int64Array;
use arrow::record_batch::RecordBatch;
use llkv_runtime::RuntimeStatementResult;
use llkv_sql::SqlEngine;
use llkv_storage::pager::{InstrumentedPager, MemPager};

fn collect_single_column_i64(engine: &SqlEngine, sql: &str) -> Vec<i64> {
    let mut results = engine.execute(sql).expect("execute select statement");
    assert_eq!(results.len(), 1, "expected single statement result");
    match results.pop().unwrap() {
        RuntimeStatementResult::Select { execution, .. } => {
            let mut out = Vec::new();
            execution
                .stream(|batch| {
                    append_column(batch, &mut out);
                    Ok(())
                })
                .expect("collect select results");
            out
        }
        other => panic!("expected SELECT result, got {other:?}"),
    }
}

fn append_column(batch: RecordBatch, out: &mut Vec<i64>) {
    let col = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("first column is Int64");
    for i in 0..col.len() {
        out.push(col.value(i));
    }
}

fn seed_table(engine: &SqlEngine) {
    engine
        .execute("CREATE TABLE scans(id INTEGER PRIMARY KEY, val INTEGER)")
        .expect("create table");

    // Three disjoint ranges to make pruning visible: 0-199_999, 400_000-599_999, 800_000-999_999.
    // The gap between ranges ensures that a narrow predicate can target only the middle band.
    let mut values = Vec::new();
    let mut id: i64 = 1;
    for (start, end) in [(0, 200_000), (400_000, 600_000), (800_000, 1_000_000)] {
        for v in start..end {
            values.push(format!("({id}, {v})"));
            id += 1;
        }
    }
    let sql = format!("INSERT INTO scans VALUES {};", values.join(","));
    engine.execute(&sql).expect("insert seed rows");
}

#[test]
fn sql_scan_paths_pruning_and_ordering() {
    let (pager, stats) = InstrumentedPager::new(MemPager::default());
    let engine = SqlEngine::new(Arc::new(pager));

    seed_table(&engine);

    // Baseline full scan.
    stats.reset();
    let mut full = collect_single_column_i64(&engine, "SELECT val FROM scans");
    let full_gets = stats.snapshot().physical_gets;

    // Range-pruned scan should touch fewer pages than the full scan.
    stats.reset();
    let pruned = collect_single_column_i64(
        &engine,
        "SELECT val FROM scans WHERE val BETWEEN 420000 AND 440000",
    );
    let pruned_gets = stats.snapshot().physical_gets;
    assert!(
        pruned_gets < full_gets,
        "expected pruned scan to fetch fewer pages than full scan (pruned: {pruned_gets}, full: {full_gets})"
    );
    assert_eq!(pruned.len(), 20_001, "unexpected row count for pruned scan");
    assert!(pruned.iter().all(|v| (420_000..=440_000).contains(v)));

    // Build a sort index on the data column so ORDER BY can use the sorted scan path.
    let ctx = engine.runtime_context();
    let table = ctx.lookup_table("scans").expect("table loaded");
    let val_col = table
        .schema
        .column_by_name("val")
        .expect("column exists")
        .field_id;
    table
        .table
        .register_sort_index(val_col)
        .expect("register sort index");

    // Sorted scan via the sort index should yield ordered results without errors.
    stats.reset();
    let ordered = collect_single_column_i64(&engine, "SELECT val FROM scans ORDER BY val");
    let ordered_gets = stats.snapshot().physical_gets;

    full.sort_unstable();
    assert_eq!(ordered, full, "ORDER BY did not return sorted values");
    assert!(
        ordered_gets > 0,
        "instrumented pager should record reads for ordered scan"
    );
}
