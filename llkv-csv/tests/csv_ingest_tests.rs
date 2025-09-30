use std::collections::HashMap;
use std::io::Write;
use std::ops::Bound;
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, Date32Array, Float64Array, Int64Array, StringArray};
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
    writeln!(tmp, "row_id,int_col,float_col,text_col,bool_col,date_col").unwrap();
    writeln!(tmp, "0,10,1.5,hello,true,2024-01-01").unwrap();
    writeln!(tmp, "1,20,2.5,world,false,2024-01-02").unwrap();
    writeln!(tmp, "2,30,3.5,test,true,2024-01-03").unwrap();
    tmp
}

fn write_sample_csv_with_nulls() -> NamedTempFile {
    let mut tmp = NamedTempFile::new().expect("create tmp csv with nulls");
    writeln!(
        tmp,
        "row_id,int_col,float_col,text_col,bool_col,date_col,anchor_col"
    )
    .unwrap();
    writeln!(tmp, "0,10,1.5,hello,true,2024-01-01,anchor").unwrap();
    writeln!(tmp, "1,,2.5,,false,2024-01-02,anchor").unwrap();
    writeln!(tmp, "2,30,,world,,,anchor").unwrap();
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
    field_mapping.insert("bool_col".to_string(), 4);
    field_mapping.insert("date_col".to_string(), 5);

    append_csv_into_table(&table, csv_file.path(), &field_mapping, &options)
        .expect("append csv into table");

    let projections = vec![
        Projection::from(LogicalFieldId::for_user(table.table_id(), 1)),
        Projection::from(LogicalFieldId::for_user(table.table_id(), 2)),
        Projection::from(LogicalFieldId::for_user(table.table_id(), 3)),
        Projection::from(LogicalFieldId::for_user(table.table_id(), 4)),
        Projection::from(LogicalFieldId::for_user(table.table_id(), 5)),
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
    let mut bools: Vec<bool> = Vec::new();
    let mut dates: Vec<i32> = Vec::new();

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
                let bool_col = batch
                    .column(3)
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .expect("bool column");
                let date_col = batch
                    .column(4)
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .expect("date column");

                ints.extend_from_slice(int_col.values());
                floats.extend_from_slice(float_col.values());
                texts.extend(text_col.iter().map(|s| s.unwrap().to_string()));
                bools.extend(bool_col.iter().map(|b| b.unwrap()));
                dates.extend(date_col.values().iter().copied());
            },
        )
        .expect("scan appended rows");

    assert_eq!(ints, vec![10, 20, 30]);
    assert_eq!(floats, vec![1.5, 2.5, 3.5]);
    assert_eq!(texts, vec!["hello", "world", "test"]);
    assert_eq!(bools, vec![true, false, true]);
    assert_eq!(dates, vec![19723, 19724, 19725]);
}

#[test]
fn csv_append_preserves_nulls() {
    let pager = Arc::new(MemPager::default());
    let table = Table::new(43, Arc::clone(&pager)).expect("create table");

    let csv_file = write_sample_csv_with_nulls();
    let options = CsvReadOptions::default();

    let (schema, _) = llkv_csv::open_csv_reader(csv_file.path(), &options).expect("open reader");
    use arrow::datatypes::DataType;
    assert_eq!(schema.field(1).data_type(), &DataType::Int64);
    assert_eq!(schema.field(2).data_type(), &DataType::Float64);
    assert_eq!(schema.field(3).data_type(), &DataType::Utf8);
    assert_eq!(schema.field(4).data_type(), &DataType::Boolean);
    assert_eq!(schema.field(5).data_type(), &DataType::Date32);
    assert_eq!(schema.field(6).data_type(), &DataType::Utf8);

    let mut field_mapping: HashMap<String, FieldId> = HashMap::new();
    field_mapping.insert("int_col".to_string(), 1);
    field_mapping.insert("float_col".to_string(), 2);
    field_mapping.insert("text_col".to_string(), 3);
    field_mapping.insert("bool_col".to_string(), 4);
    field_mapping.insert("date_col".to_string(), 5);
    field_mapping.insert("anchor_col".to_string(), 6);

    append_csv_into_table(&table, csv_file.path(), &field_mapping, &options)
        .expect("append csv with nulls");

    let projections = vec![
        Projection::from(LogicalFieldId::for_user(table.table_id(), 1)),
        Projection::from(LogicalFieldId::for_user(table.table_id(), 2)),
        Projection::from(LogicalFieldId::for_user(table.table_id(), 3)),
        Projection::from(LogicalFieldId::for_user(table.table_id(), 4)),
        Projection::from(LogicalFieldId::for_user(table.table_id(), 5)),
    ];

    let filter_all_rows = Expr::Pred(Filter {
        field_id: 6,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    });

    let mut ints: Vec<Option<i64>> = Vec::new();
    let mut floats: Vec<Option<f64>> = Vec::new();
    let mut texts: Vec<Option<String>> = Vec::new();
    let mut bools: Vec<Option<bool>> = Vec::new();
    let mut dates: Vec<Option<i32>> = Vec::new();

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
                let bool_col = batch
                    .column(3)
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .expect("bool column");
                let date_col = batch
                    .column(4)
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .expect("date column");

                for row in 0..batch.num_rows() {
                    ints.push(if int_col.is_null(row) {
                        None
                    } else {
                        Some(int_col.value(row))
                    });
                    floats.push(if float_col.is_null(row) {
                        None
                    } else {
                        Some(float_col.value(row))
                    });
                    texts.push(if text_col.is_null(row) {
                        None
                    } else {
                        Some(text_col.value(row).to_string())
                    });
                    bools.push(if bool_col.is_null(row) {
                        None
                    } else {
                        Some(bool_col.value(row))
                    });
                    dates.push(if date_col.is_null(row) {
                        None
                    } else {
                        Some(date_col.value(row))
                    });
                }
            },
        )
        .expect("scan appended rows with nulls");

    assert_eq!(ints, vec![Some(10), None, Some(30)]);
    assert_eq!(floats, vec![Some(1.5), Some(2.5), None]);
    assert_eq!(
        texts,
        vec![Some("hello".to_string()), None, Some("world".to_string())]
    );
    assert_eq!(bools, vec![Some(true), Some(false), None]);
    assert_eq!(dates, vec![Some(19723), Some(19724), None]);
}
