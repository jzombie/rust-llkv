use arrow::array::{StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::store::Projection;
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::{Expr, Filter, Operator};
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use llkv_table::table::ScanStreamOptions;
use std::sync::Arc;

fn make_string_table(table_id: llkv_table::types::TableId) -> Table {
    let pager = Arc::new(MemPager::default());
    let table = Table::from_id(table_id, Arc::clone(&pager)).unwrap();

    const FIELD: llkv_table::types::FieldId = 1;
    let schema = Arc::new(Schema::new(vec![
        Field::new(llkv_column_map::ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("s", DataType::Utf8, false).with_metadata({
            let mut m = std::collections::HashMap::new();
            m.insert(
                llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                FIELD.to_string(),
            );
            m
        }),
    ]));

    let mut row_ids = Vec::new();
    let mut values = Vec::new();
    for i in 0..10000u64 {
        row_ids.push(i);
        if i % 100 == 0 {
            values.push(format!("row-{}-payload-needle", i));
        } else {
            values.push(format!("row-{}-payload", i));
        }
    }

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(row_ids)),
            Arc::new(StringArray::from(values)),
        ],
    )
    .unwrap();

    table.append(&batch).unwrap();
    table
}

fn make_table_from_values(
    table_id: llkv_table::types::TableId,
    field_id: llkv_table::types::FieldId,
    values: &[&str],
) -> Table {
    let pager = Arc::new(MemPager::default());
    let table = Table::from_id(table_id, Arc::clone(&pager)).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new(llkv_column_map::ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("s", DataType::Utf8, false).with_metadata({
            let mut m = std::collections::HashMap::new();
            m.insert(
                llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                field_id.to_string(),
            );
            m
        }),
    ]));

    let row_ids: Vec<u64> = (0..values.len() as u64).collect();
    let string_values: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(row_ids)),
            Arc::new(StringArray::from(string_values)),
        ],
    )
    .unwrap();

    table.append(&batch).unwrap();
    table
}

fn collect_matches(
    table: &Table,
    field_id: llkv_table::types::FieldId,
    expr: &Expr<'_, llkv_table::types::FieldId>,
) -> Vec<String> {
    let projection = Projection::from(LogicalFieldId::for_user(table.table_id(), field_id));
    let mut out = Vec::new();
    table
        .scan_stream(&[projection], expr, ScanStreamOptions::default(), |batch| {
            let arr = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for value in arr.iter().flatten() {
                out.push(value.to_string());
            }
        })
        .unwrap();
    out
}

#[test]
fn fused_equals_sequential_string_contains() {
    let table = make_string_table(900);
    const FIELD: llkv_table::types::FieldId = 1;
    // Two contains predicates on same field
    let pred1 = Expr::Pred(Filter {
        field_id: FIELD,
        op: Operator::contains("needle", true),
    });
    let pred2 = Expr::Pred(Filter {
        field_id: FIELD,
        op: Operator::starts_with("row-1", true),
    });

    // Fused: planner should detect AND of same-field preds
    let and_expr = Expr::And(vec![pred1.clone(), pred2.clone()]);
    let mut fused_count = 0usize;
    table
        .scan_stream(
            &[Projection::from(LogicalFieldId::for_user(
                table.table_id(),
                FIELD,
            ))],
            &and_expr,
            ScanStreamOptions::default(),
            |b| {
                fused_count += b.num_rows();
            },
        )
        .unwrap();

    // Sequential: run two single-predicate scans and intersect row ids
    use llkv_column_map::store::scan::filter::Utf8Filter;
    let lf = LogicalFieldId::for_user(table.table_id(), FIELD);
    let p1 =
        llkv_expr::typed_predicate::build_var_width_predicate(&Operator::contains("needle", true))
            .unwrap();
    let p2 = llkv_expr::typed_predicate::build_var_width_predicate(&Operator::starts_with(
        "row-1", true,
    ))
    .unwrap();
    let ids1 = table
        .store()
        .filter_row_ids::<Utf8Filter<i32>>(lf, &p1)
        .unwrap();
    let ids2 = table
        .store()
        .filter_row_ids::<Utf8Filter<i32>>(lf, &p2)
        .unwrap();
    let mut i = 0usize;
    let mut j = 0usize;
    let mut seq_ids = Vec::new();
    while i < ids1.len() && j < ids2.len() {
        match ids1[i].cmp(&ids2[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                seq_ids.push(ids1[i]);
                i += 1;
                j += 1;
            }
        }
    }

    // Validate counts match
    assert_eq!(fused_count, seq_ids.len());
}

#[test]
fn scan_string_predicates_case_sensitivity() {
    const FIELD: llkv_table::types::FieldId = 7;

    let contains_table = make_table_from_values(901, FIELD, &["needle", "Needle", "other"]);
    let mut sensitive = collect_matches(
        &contains_table,
        FIELD,
        &Expr::Pred(Filter {
            field_id: FIELD,
            op: Operator::contains("needle", true),
        }),
    );
    sensitive.sort();
    assert_eq!(sensitive, vec!["needle".to_string()]);

    let mut insensitive = collect_matches(
        &contains_table,
        FIELD,
        &Expr::Pred(Filter {
            field_id: FIELD,
            op: Operator::contains("needle", false),
        }),
    );
    insensitive.sort();
    assert_eq!(
        insensitive,
        vec!["Needle".to_string(), "needle".to_string()]
    );

    let starts_table = make_table_from_values(902, FIELD, &["row-one", "Row-two", "beta"]);
    let mut starts_sensitive = collect_matches(
        &starts_table,
        FIELD,
        &Expr::Pred(Filter {
            field_id: FIELD,
            op: Operator::starts_with("row-", true),
        }),
    );
    starts_sensitive.sort();
    assert_eq!(starts_sensitive, vec!["row-one".to_string()]);

    let mut starts_insensitive = collect_matches(
        &starts_table,
        FIELD,
        &Expr::Pred(Filter {
            field_id: FIELD,
            op: Operator::starts_with("row-", false),
        }),
    );
    starts_insensitive.sort();
    assert_eq!(
        starts_insensitive,
        vec!["Row-two".to_string(), "row-one".to_string()]
    );

    let ends_table = make_table_from_values(903, FIELD, &["tail", "Tail", "other"]);
    let mut ends_sensitive = collect_matches(
        &ends_table,
        FIELD,
        &Expr::Pred(Filter {
            field_id: FIELD,
            op: Operator::ends_with("tail", true),
        }),
    );
    ends_sensitive.sort();
    assert_eq!(ends_sensitive, vec!["tail".to_string()]);

    let mut ends_insensitive = collect_matches(
        &ends_table,
        FIELD,
        &Expr::Pred(Filter {
            field_id: FIELD,
            op: Operator::ends_with("tail", false),
        }),
    );
    ends_insensitive.sort();
    assert_eq!(
        ends_insensitive,
        vec!["Tail".to_string(), "tail".to_string()]
    );
}

#[test]
fn fused_case_sensitivity_combinations() {
    const FIELD: llkv_table::types::FieldId = 8;
    let values = [
        "row-000-needle",
        "row-001-Needle",
        "ROW-002-NEEDLE",
        "Row-003-needle",
        "row-004-other",
    ];
    let table = make_table_from_values(904, FIELD, &values);

    let scenarios = [
        (true, true, vec!["row-000-needle"]),
        (false, true, vec!["row-000-needle", "row-001-Needle"]),
        (true, false, vec!["Row-003-needle", "row-000-needle"]),
        (
            false,
            false,
            vec![
                "ROW-002-NEEDLE",
                "Row-003-needle",
                "row-000-needle",
                "row-001-Needle",
            ],
        ),
    ];

    for (contains_case_sensitive, starts_case_sensitive, expected_values) in scenarios {
        let expr = Expr::And(vec![
            Expr::Pred(Filter {
                field_id: FIELD,
                op: Operator::contains("needle", contains_case_sensitive),
            }),
            Expr::Pred(Filter {
                field_id: FIELD,
                op: Operator::starts_with("row-", starts_case_sensitive),
            }),
        ]);

        let mut actual = collect_matches(&table, FIELD, &expr);
        actual.sort();
        let mut expected: Vec<String> = expected_values.iter().map(|s| s.to_string()).collect();
        expected.sort();
        assert_eq!(actual, expected);
    }
}
