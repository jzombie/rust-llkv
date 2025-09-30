use std::collections::HashMap;
use std::io::Write;
use std::ops::Bound;
use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, StringArray};
use llkv_column_map::store::Projection;
use llkv_column_map::types::LogicalFieldId;
use llkv_csv::{CsvReadOptions, append_csv_into_table};
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use llkv_table::expr::{Expr, Filter, Operator};
use llkv_table::table::ScanStreamOptions;
use llkv_table::types::FieldId;
use tempfile::NamedTempFile;

fn write_sample_csv() -> NamedTempFile {
    let mut tmp = NamedTempFile::new().expect("create tmp csv");
    writeln!(tmp, "row_id,int_col,float_col,text_col").unwrap();
    writeln!(tmp, "0,10,1.5,hello").unwrap();
    writeln!(tmp, "1,20,2.5,world").unwrap();
    writeln!(tmp, "2,30,3.5,test").unwrap();
    tmp
}

#[test]
fn csv_append_roundtrip() {
    let pager = Arc::new(MemPager::default());
    let table = Table::new(42, Arc::clone(&pager)).expect("create table");

    let csv_file = write_sample_csv();
    let options = CsvReadOptions::default();

    let mut field_mapping: HashMap<String, FieldId> = HashMap::new();
    field_mapping.insert("int_col".to_string(), 1);
    field_mapping.insert("float_col".to_string(), 2);
    field_mapping.insert("text_col".to_string(), 3);

    append_csv_into_table(&table, csv_file.path(), &field_mapping, &options)
        .expect("append csv into table");

    let projections = vec![
        Projection::from(LogicalFieldId::for_user(table.table_id(), 1)),
        Projection::from(LogicalFieldId::for_user(table.table_id(), 2)),
        Projection::from(LogicalFieldId::for_user(table.table_id(), 3)),
    ];

    let filter_all_rows = Expr::Pred(Filter {
        field_id: 1,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    });

    let mut ints: Vec<i64> = Vec::new();
    let mut floats: Vec<f64> = Vec::new();
    let mut texts: Vec<String> = Vec::new();

    table
        .scan_stream(
            &projections,
            &filter_all_rows,
            ScanStreamOptions::default(),
            |batch| {
                let int_col = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("int column");
                let float_col = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("float column");
                let text_col = batch
                    .column(2)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("text column");

                ints.extend_from_slice(int_col.values());
                floats.extend_from_slice(float_col.values());
                texts.extend(text_col.iter().map(|s| s.unwrap().to_string()));
            },
        )
        .expect("scan appended rows");

    assert_eq!(ints, vec![10, 20, 30]);
    assert_eq!(floats, vec![1.5, 2.5, 3.5]);
    assert_eq!(texts, vec!["hello", "world", "test"]);
}
