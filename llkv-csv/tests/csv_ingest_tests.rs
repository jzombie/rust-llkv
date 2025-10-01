use std::collections::HashMap;
use std::io::Write;
use std::ops::Bound;
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, Date32Array, Float64Array, Int64Array, StringArray};
use llkv_column_map::store::Projection;
use llkv_column_map::types::LogicalFieldId;
use llkv_csv::{CsvReadOptions, append_csv_into_table, append_csv_into_table_with_mapping};
use llkv_storage::pager::MemPager;
use llkv_table::expr::{Expr, Filter, Operator};
use llkv_table::table::ScanStreamOptions;
use llkv_table::{Table, types::FieldId};
use tempfile::NamedTempFile;

use rand::{seq::SliceRandom, rngs::StdRng, SeedableRng};

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

fn write_additional_csv_rows() -> NamedTempFile {
    let mut tmp = NamedTempFile::new().expect("create tmp csv with extra rows");
    writeln!(tmp, "row_id,int_col,float_col,text_col,bool_col,date_col").unwrap();
    writeln!(tmp, "3,40,4.5,again,false,2024-01-04").unwrap();
    writeln!(tmp, "4,50,5.5,more,true,2024-01-05").unwrap();
    tmp
}

#[test]
fn csv_persists_colmeta_names() {
    let pager = Arc::new(MemPager::default());
    let table = Table::new(2001, Arc::clone(&pager)).expect("create table");

    let options = CsvReadOptions::default();
    let csv = write_sample_csv();
    append_csv_into_table(&table, csv.path(), &options).expect("append csv");

    // Query the catalog for the first two user column metas
    let metas = table.get_cols_meta(&[1, 2]);
    assert!(metas.len() == 2);
    assert!(metas[0].as_ref().and_then(|m| m.name.clone()).is_some());
    assert!(metas[1].as_ref().and_then(|m| m.name.clone()).is_some());
    assert_eq!(metas[0].as_ref().unwrap().name.as_ref().unwrap(), "int_col");
}

#[test]
fn csv_infer_fuzz_permutations() {
    // Deterministic-ish permutation test: shuffle column order a few times and
    // make sure inference succeeds and produces unique mappings.
    let mut rng = StdRng::seed_from_u64(42);
    let base_cols = vec!["row_id", "int_col", "float_col", "text_col", "bool_col", "date_col"];

    // Increase seeds and write multiple rows for broader coverage.
    for seed in 0..50 {
        let mut cols = base_cols.clone();
        cols[1..].shuffle(&mut rng);

        // Build a temporary CSV with this column order, using the same values.
        let mut tmp = NamedTempFile::new().expect("tmp csv");
        writeln!(tmp, "{}", cols.join(",")).unwrap();
        // write two rows of data to exercise multiple rows handling
        let row_vals1 = vec!["0","10","1.5","hello","true","2024-01-01"];
        let row_vals2 = vec!["1","20","2.5","world","false","2024-01-02"];
        let mut ordered1: Vec<&str> = Vec::new();
        let mut ordered2: Vec<&str> = Vec::new();
        for c in &cols {
            let idx = base_cols.iter().position(|b| b == c).unwrap();
            ordered1.push(row_vals1[idx]);
            ordered2.push(row_vals2[idx]);
        }
        writeln!(tmp, "{}", ordered1.join(",")).unwrap();
        writeln!(tmp, "{}", ordered2.join(",")).unwrap();

    let pager = Arc::new(MemPager::default());
    let table = Table::new(3000 + seed as u16, Arc::clone(&pager)).expect("create table");
        let options = CsvReadOptions::default();
        append_csv_into_table(&table, tmp.path(), &options).expect("append permuted csv");

        // Ensure metas exist and mapping is unique
        let logicals = table.store().user_field_ids_for_table(table.table_id());
        let mut seen = std::collections::HashSet::new();
        for l in logicals {
            let fid = l.field_id();
            assert!(fid != 0);
            assert!(seen.insert(fid), "duplicate fid seen");
        }
    }
}

#[test]
fn csv_append_roundtrip() {
    let pager = Arc::new(MemPager::default());
    let table = Table::new(42, Arc::clone(&pager)).expect("create table");

    let csv_file = write_sample_csv();
    let options = CsvReadOptions::default();

    append_csv_into_table(&table, csv_file.path(), &options).expect("append csv into table");

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
fn csv_infer_reuse_regression() {
    // Regression test for schema inference reuse: ensure appending a CSV and
    // then appending additional rows reuses field ids instead of erroring.
    let pager = Arc::new(MemPager::default());
    let table = Table::new(1001, Arc::clone(&pager)).expect("create table");

    let options = CsvReadOptions::default();
    let first = write_sample_csv();
    append_csv_into_table(&table, first.path(), &options).expect("append first csv");

    let second = write_additional_csv_rows();
    append_csv_into_table(&table, second.path(), &options).expect("append second csv");

    // Confirm the mapping exists by asking for schema and checking column names
    let schema = table.schema().expect("schema");
    assert_eq!(schema.fields().len(), 6); // row_id + 5 user columns
    assert_eq!(schema.field(1).name(), "int_col");
}

#[test]
fn csv_auto_schema_reuses_field_ids() {
    let pager = Arc::new(MemPager::default());
    let table = Table::new(44, Arc::clone(&pager)).expect("create table");

    let options = CsvReadOptions::default();

    let first_csv = write_sample_csv();
    append_csv_into_table(&table, first_csv.path(), &options).expect("append initial csv");

    let second_csv = write_additional_csv_rows();
    append_csv_into_table(&table, second_csv.path(), &options).expect("append additional csv");

    let projections = vec![
        Projection::from(LogicalFieldId::for_user(table.table_id(), 1)),
        Projection::from(LogicalFieldId::for_user(table.table_id(), 2)),
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

                ints.extend_from_slice(int_col.values());
                floats.extend_from_slice(float_col.values());
            },
        )
        .expect("scan appended rows");

    assert_eq!(ints, vec![10, 20, 30, 40, 50]);
    assert_eq!(floats, vec![1.5, 2.5, 3.5, 4.5, 5.5]);
}

#[test]
fn csv_append_with_manual_mapping() {
    let pager = Arc::new(MemPager::default());
    let table = Table::new(45, Arc::clone(&pager)).expect("create table");

    let mut field_mapping: HashMap<String, FieldId> = HashMap::new();
    field_mapping.insert("int_col".to_string(), 10);
    field_mapping.insert("float_col".to_string(), 20);
    field_mapping.insert("text_col".to_string(), 30);
    field_mapping.insert("bool_col".to_string(), 40);
    field_mapping.insert("date_col".to_string(), 50);

    let options = CsvReadOptions::default();
    let csv_file = write_sample_csv();

    append_csv_into_table_with_mapping(&table, csv_file.path(), &field_mapping, &options)
        .expect("append csv with manual mapping");

    let projections = vec![
        Projection::from(LogicalFieldId::for_user(table.table_id(), 10)),
        Projection::from(LogicalFieldId::for_user(table.table_id(), 20)),
        Projection::from(LogicalFieldId::for_user(table.table_id(), 30)),
        Projection::from(LogicalFieldId::for_user(table.table_id(), 40)),
        Projection::from(LogicalFieldId::for_user(table.table_id(), 50)),
    ];

    let filter_all_rows = Expr::Pred(Filter {
        field_id: 10,
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
        .expect("scan manual mapping rows");

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

    append_csv_into_table(&table, csv_file.path(), &options).expect("append csv with nulls");

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
