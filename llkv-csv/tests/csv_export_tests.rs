use std::io::Write;
use std::ops::Bound;
use std::sync::Arc;

use llkv_column_map::store::Projection;
use llkv_column_map::types::LogicalFieldId;
use llkv_csv::CsvReadOptions;
use llkv_csv::csv_export::{
    CsvExportColumn, CsvWriteOptions, export_csv_from_table, export_csv_from_table_with_filter,
    export_csv_from_table_with_projections, export_csv_to_writer_with_filter,
    export_csv_to_writer_with_projections,
};
use llkv_csv::csv_ingest::append_csv_into_table;
use llkv_result::Result as LlkvResult;
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use llkv_table::expr::{BinaryOp, Expr, Filter, Operator, ScalarExpr};
use llkv_table::table::ScanProjection;
use tempfile::NamedTempFile;

fn write_sample_csv() -> NamedTempFile {
    let mut tmp = NamedTempFile::new().expect("create tmp csv");
    writeln!(tmp, "rowid,int_col,float_col,text_col,bool_col,date_col").unwrap();
    writeln!(tmp, "0,10,1.5,hello,true,2024-01-01").unwrap();
    writeln!(tmp, "1,20,2.5,world,false,2024-01-02").unwrap();
    writeln!(tmp, "2,30,3.5,test,true,2024-01-03").unwrap();
    tmp
}

fn setup_table_with_sample_data() -> LlkvResult<Table> {
    let pager = Arc::new(MemPager::default());
    let table = Table::from_id(200, Arc::clone(&pager))?;
    let csv_file = write_sample_csv();
    let options = CsvReadOptions::default();
    append_csv_into_table(&table, csv_file.path(), &options)?;
    Ok(table)
}

#[test]
fn export_basic_projection() {
    let table = setup_table_with_sample_data().expect("create table with data");
    let out_file = NamedTempFile::new().expect("create export csv");

    let columns = vec![
        CsvExportColumn::with_alias(1, "int_col_alias"),
        CsvExportColumn::with_alias(3, "text_col_alias"),
    ];

    export_csv_from_table(
        &table,
        out_file.path(),
        &columns,
        &CsvWriteOptions::default(),
    )
    .expect("export csv");

    let contents = std::fs::read_to_string(out_file.path()).expect("read exported csv");
    let expected = "int_col_alias,text_col_alias\n10,hello\n20,world\n30,test\n";
    assert_eq!(contents, expected);
}

#[test]
fn export_with_filter_and_custom_options() {
    let table = setup_table_with_sample_data().expect("create table with data");
    let out_file = NamedTempFile::new().expect("create export csv");

    let columns = vec![
        CsvExportColumn::with_alias(1, "int_col_alias"),
        CsvExportColumn::with_alias(2, "float_col_alias"),
    ];

    let filter_expr = Expr::Pred(Filter {
        field_id: 1,
        op: Operator::Range {
            lower: Bound::Included(20.into()),
            upper: Bound::Unbounded,
        },
    });

    let options = CsvWriteOptions {
        delimiter: b'\t',
        ..Default::default()
    };

    export_csv_from_table_with_filter(&table, out_file.path(), &columns, &filter_expr, &options)
        .expect("export filtered csv");

    let contents = std::fs::read_to_string(out_file.path()).expect("read exported csv");
    let expected = "int_col_alias\tfloat_col_alias\n20\t2.5\n30\t3.5\n";
    assert_eq!(contents, expected);
}

#[test]
fn export_to_memory_writer() {
    let table = setup_table_with_sample_data().expect("create table with data");
    let mut buffer: Vec<u8> = Vec::new();

    let columns = vec![
        CsvExportColumn::with_alias(1, "int_col_alias"),
        CsvExportColumn::with_alias(4, "bool_col_alias"),
    ];

    let filter_expr = Expr::Pred(Filter {
        field_id: 1,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    });

    export_csv_to_writer_with_filter(
        &table,
        &mut buffer,
        &columns,
        &filter_expr,
        &CsvWriteOptions::default(),
    )
    .expect("export to memory");

    let contents = String::from_utf8(buffer).expect("utf8 output");
    let expected = "int_col_alias,bool_col_alias\n10,true\n20,false\n30,true\n";
    assert_eq!(contents, expected);
}

#[test]
fn export_with_projection_alias_inference() {
    let table = setup_table_with_sample_data().expect("create table with data");
    let out_file = NamedTempFile::new().expect("create export csv");

    let projections = vec![
        ScanProjection::from(Projection::with_alias(
            LogicalFieldId::for_user(table.table_id(), 1),
            "int_col_alias",
        )),
        ScanProjection::from(Projection::with_alias(
            LogicalFieldId::for_user(table.table_id(), 3),
            "text_col_alias",
        )),
    ];

    let filter_expr = Expr::Pred(Filter {
        field_id: 1,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    });

    export_csv_from_table_with_projections(
        &table,
        out_file.path(),
        projections,
        &filter_expr,
        &CsvWriteOptions::default(),
    )
    .expect("export csv");

    let contents = std::fs::read_to_string(out_file.path()).expect("read exported csv");
    let expected = "int_col_alias,text_col_alias\n10,hello\n20,world\n30,test\n";
    assert_eq!(contents, expected);
}

#[test]
fn export_with_computed_projection_alias() {
    let table = setup_table_with_sample_data().expect("create table with data");
    let mut buffer: Vec<u8> = Vec::new();

    let projections = vec![
        ScanProjection::from(Projection::with_alias(
            LogicalFieldId::for_user(table.table_id(), 1),
            "int_col_alias",
        )),
        ScanProjection::computed(
            ScalarExpr::binary(
                ScalarExpr::column(1),
                BinaryOp::Multiply,
                ScalarExpr::literal(2),
            ),
            "int_times_two_alias",
        ),
    ];

    let filter_expr = Expr::Pred(Filter {
        field_id: 1,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    });

    export_csv_to_writer_with_projections(
        &table,
        &mut buffer,
        projections,
        &filter_expr,
        &CsvWriteOptions::default(),
    )
    .expect("export csv with computed projection");

    let contents = String::from_utf8(buffer).expect("utf8 output");
    let mut lines = contents.lines();
    assert_eq!(lines.next(), Some("int_col_alias,int_times_two_alias"));
    let mut rows: Vec<(i64, f64)> = Vec::new();
    for line in lines {
        let parts: Vec<&str> = line.split(',').collect();
        assert_eq!(parts.len(), 2);
        let base: i64 = parts[0].parse().expect("int value");
        let computed: f64 = parts[1].parse().expect("float value");
        rows.push((base, computed));
    }

    let expected = vec![(10, 20.0), (20, 40.0), (30, 60.0)];
    assert_eq!(rows, expected);
}
