use std::io::Write;
use std::ops::Bound;
use std::sync::Arc;

use llkv_csv::csv_export::{
    CsvExportColumn, CsvWriteOptions, export_csv_from_table, export_csv_from_table_with_filter,
    export_csv_to_writer_with_filter,
};
use llkv_csv::CsvReadOptions;
use llkv_csv::csv_ingest::append_csv_into_table;
use llkv_result::Result as LlkvResult;
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use llkv_table::expr::{Expr, Filter, Operator};
use tempfile::NamedTempFile;

fn write_sample_csv() -> NamedTempFile {
    let mut tmp = NamedTempFile::new().expect("create tmp csv");
    writeln!(tmp, "row_id,int_col,float_col,text_col,bool_col,date_col").unwrap();
    writeln!(tmp, "0,10,1.5,hello,true,2024-01-01").unwrap();
    writeln!(tmp, "1,20,2.5,world,false,2024-01-02").unwrap();
    writeln!(tmp, "2,30,3.5,test,true,2024-01-03").unwrap();
    tmp
}

fn setup_table_with_sample_data() -> LlkvResult<Table> {
    let pager = Arc::new(MemPager::default());
    let table = Table::new(200, Arc::clone(&pager))?;
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
        CsvExportColumn::with_alias(1, "int_col"),
        CsvExportColumn::with_alias(3, "text_col"),
    ];

    export_csv_from_table(
        &table,
        out_file.path(),
        &columns,
        &CsvWriteOptions::default(),
    )
    .expect("export csv");

    let contents = std::fs::read_to_string(out_file.path()).expect("read exported csv");
    let expected = "int_col,text_col\n10,hello\n20,world\n30,test\n";
    assert_eq!(contents, expected);
}

#[test]
fn export_with_filter_and_custom_options() {
    let table = setup_table_with_sample_data().expect("create table with data");
    let out_file = NamedTempFile::new().expect("create export csv");

    let columns = vec![
        CsvExportColumn::with_alias(1, "int_col"),
        CsvExportColumn::with_alias(2, "float_col"),
    ];

    let filter_expr = Expr::Pred(Filter {
        field_id: 1,
        op: Operator::Range {
            lower: Bound::Included(20.into()),
            upper: Bound::Unbounded,
        },
    });

    let mut options = CsvWriteOptions::default();
    options.delimiter = b'\t';

    export_csv_from_table_with_filter(&table, out_file.path(), &columns, &filter_expr, &options)
        .expect("export filtered csv");

    let contents = std::fs::read_to_string(out_file.path()).expect("read exported csv");
    let expected = "int_col\tfloat_col\n20\t2.5\n30\t3.5\n";
    assert_eq!(contents, expected);
}

#[test]
fn export_to_memory_writer() {
    let table = setup_table_with_sample_data().expect("create table with data");
    let mut buffer: Vec<u8> = Vec::new();

    let columns = vec![
        CsvExportColumn::with_alias(1, "int_col"),
        CsvExportColumn::with_alias(4, "bool_col"),
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
    let expected = "int_col,bool_col\n10,true\n20,false\n30,true\n";
    assert_eq!(contents, expected);
}
