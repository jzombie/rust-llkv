#![forbid(unsafe_code)]

use std::hint::black_box;
use std::sync::Arc;

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use criterion::{Criterion, criterion_group, criterion_main};

use llkv_column_map::ROW_ID_COLUMN_NAME;
use llkv_column_map::store::Projection;
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::{Expr, Filter, Operator};
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use llkv_table::table::ScanStreamOptions;
use llkv_table::types::FieldId;

const NUM_ROWS: usize = 1_000_000;
const EXPECTED_MATCHES: usize = NUM_ROWS / 1000;
const TABLE_ID: llkv_table::types::TableId = 4242;
const FIELD_ID: FieldId = 1_001;

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    contains_case_sensitive: bool,
    starts_with_case_sensitive: bool,
}

const SCENARIOS: [Scenario; 4] = [
    Scenario {
        name: "both_case_sensitive",
        contains_case_sensitive: true,
        starts_with_case_sensitive: true,
    },
    Scenario {
        name: "contains_case_insensitive",
        contains_case_sensitive: false,
        starts_with_case_sensitive: true,
    },
    Scenario {
        name: "starts_case_insensitive",
        contains_case_sensitive: true,
        starts_with_case_sensitive: false,
    },
    Scenario {
        name: "both_case_insensitive",
        contains_case_sensitive: false,
        starts_with_case_sensitive: false,
    },
];

fn schema_with_row_id(field: Field) -> Arc<Schema> {
    let row_id = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    Arc::new(Schema::new(vec![row_id, field]))
}

fn setup_table() -> Table {
    let pager = Arc::new(MemPager::default());
    let table = Table::new(TABLE_ID, Arc::clone(&pager)).expect("table creation");

    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        llkv_table::constants::FIELD_ID_META_KEY.to_string(),
        FIELD_ID.to_string(),
    );
    let data_field = Field::new("data", DataType::Utf8, false).with_metadata(metadata);
    let schema = schema_with_row_id(data_field);

    // Build NUM_ROWS strings; every 1000th row contains "needle".
    let mut row_ids = Vec::with_capacity(NUM_ROWS);
    let mut values = Vec::with_capacity(NUM_ROWS);
    for i in 0..NUM_ROWS {
        row_ids.push(i as u64);
        if i % 1000 == 0 {
            values.push(format!("row-{}-payload-needle", i));
        } else {
            values.push(format!("row-{}-payload", i));
        }
    }

    let row_id_array = Arc::new(arrow::array::UInt64Array::from(row_ids));
    let value_array = Arc::new(StringArray::from(values));

    let batch =
        RecordBatch::try_new(schema, vec![row_id_array, value_array]).expect("record batch");
    table.append(&batch).expect("append batch");

    table
}

fn bench_planner_fused(table: &Table, scenario: Scenario) -> usize {
    let logical_field_id = LogicalFieldId::for_user(TABLE_ID, FIELD_ID);
    let projections = vec![Projection::from(logical_field_id)];
    // Build AND of two contains predicates; planner should fuse these into a single run
    let filter = Expr::And(vec![
        Expr::Pred(Filter {
            field_id: FIELD_ID,
            op: Operator::contains("needle", scenario.contains_case_sensitive),
        }),
        Expr::Pred(Filter {
            field_id: FIELD_ID,
            op: Operator::starts_with("row-", scenario.starts_with_case_sensitive),
        }),
    ]);

    let mut found = 0usize;
    table
        .scan_stream(
            &projections,
            &filter,
            ScanStreamOptions::default(),
            |batch| {
                found += batch.num_rows();
            },
        )
        .expect("scan_stream fused");
    found
}

fn run_store_sequential(table: &Table, scenario: Scenario) -> usize {
    use llkv_column_map::store::scan::filter::Utf8Filter;
    let lf = LogicalFieldId::for_user(TABLE_ID, FIELD_ID);
    let p1 = llkv_expr::typed_predicate::build_var_width_predicate(&Operator::contains(
        "needle",
        scenario.contains_case_sensitive,
    ))
    .unwrap();
    let p2 = llkv_expr::typed_predicate::build_var_width_predicate(&Operator::starts_with(
        "row-",
        scenario.starts_with_case_sensitive,
    ))
    .unwrap();
    let ids1 = table
        .store()
        .filter_row_ids::<Utf8Filter<i32>>(lf, &p1)
        .expect("ids1");
    let ids2 = table
        .store()
        .filter_row_ids::<Utf8Filter<i32>>(lf, &p2)
        .expect("ids2");
    // intersect
    let mut i = 0usize;
    let mut j = 0usize;
    let mut out = Vec::new();
    while i < ids1.len() && j < ids2.len() {
        match ids1[i].cmp(&ids2[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(ids1[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.len()
}

fn run_store_fused(table: &Table, scenario: Scenario) -> usize {
    let lf = LogicalFieldId::for_user(TABLE_ID, FIELD_ID);
    let p1 = llkv_expr::typed_predicate::build_var_width_predicate(&Operator::contains(
        "needle",
        scenario.contains_case_sensitive,
    ))
    .unwrap();
    let p2 = llkv_expr::typed_predicate::build_var_width_predicate(&Operator::starts_with(
        "row-",
        scenario.starts_with_case_sensitive,
    ))
    .unwrap();
    let preds = vec![p1, p2];
    <llkv_column_map::store::scan::filter::Utf8Filter<i32> as llkv_column_map::store::scan::filter::FilterDispatch>::run_fused(
        table.store(),
        lf,
        &preds,
    )
    .expect("fused run")
    .len()
}

fn bench_fused_vs_sequential(c: &mut Criterion) {
    let table = setup_table();

    let mut group = c.benchmark_group("string_substring_fused_vs_sequential");
    group.sample_size(20);

    for scenario in SCENARIOS {
        let fused_label = format!("planner_fused_{}", scenario.name);
        group.bench_function(fused_label, |b| {
            b.iter(|| {
                let n = bench_planner_fused(&table, scenario);
                assert_eq!(n, EXPECTED_MATCHES);
                black_box(n);
            })
        });

        let seq_label = format!("store_sequential_{}", scenario.name);
        group.bench_function(seq_label, |b| {
            b.iter(|| {
                let n = run_store_sequential(&table, scenario);
                assert_eq!(n, EXPECTED_MATCHES);
                black_box(n);
            })
        });
    }

    group.finish();

    // --- Storage-only microbench (no gather/scan_stream overhead) ---
    let mut group2 = c.benchmark_group("string_substring_storage_only");
    group2.sample_size(20);

    for scenario in SCENARIOS {
        let fused_label = format!("storage_fused_{}", scenario.name);
        group2.bench_function(fused_label, |b| {
            b.iter(|| {
                let n = run_store_fused(&table, scenario);
                assert_eq!(n, EXPECTED_MATCHES);
                black_box(n);
            })
        });

        let seq_label = format!("storage_sequential_{}", scenario.name);
        group2.bench_function(seq_label, |b| {
            b.iter(|| {
                let n = run_store_sequential(&table, scenario);
                assert_eq!(n, EXPECTED_MATCHES);
                black_box(n);
            })
        });
    }

    group2.finish();
}

criterion_group!(benches, bench_fused_vs_sequential);
criterion_main!(benches);
