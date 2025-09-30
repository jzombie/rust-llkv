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
    let table = Table::new(table_id, Arc::clone(&pager)).unwrap();

    const FIELD: llkv_table::types::FieldId = 1;
    let schema = Arc::new(Schema::new(vec![
        Field::new(llkv_column_map::ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("s", DataType::Utf8, false).with_metadata({
            let mut m = std::collections::HashMap::new();
            m.insert("field_id".to_string(), FIELD.to_string());
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
