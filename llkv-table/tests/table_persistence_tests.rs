#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    Float32Array, Float64Array, Int16Array, Int32Array, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::ROW_ID_COLUMN_NAME;
use llkv_table::Table;
use llkv_table::types::{FieldId, RowId, TableId};

use llkv_storage::pager::simd_r_drive_pager::SimdRDrivePager;
use tempfile::TempDir;

/// Attach required `field_id` metadata to a data column field.
fn field_with_fid(name: &str, dt: DataType, fid: FieldId, nullable: bool) -> Field {
    Field::new(name, dt, nullable)
        .with_metadata(HashMap::from([("field_id".to_string(), fid.to_string())]))
}

#[test]
fn table_persistence_many_columns_simd_r_drive() {
    // Use a disk-backed pager: write, drop, reopen, verify.
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("table.db");

    // ---------- Test fixture ----------
    const TID: TableId = 100;

    // Distinct field ids for several numeric columns.
    const F_U64: FieldId = 10;
    const F_I32: FieldId = 11;
    const F_U32: FieldId = 12;
    const F_I16: FieldId = 13;
    const F_U8: FieldId = 14;
    const F_F64: FieldId = 15;
    const F_F32: FieldId = 16;
    const F_U16: FieldId = 18;

    // 4 rows
    let rows: Vec<RowId> = vec![1, 2, 3, 4];

    // Column payloads
    let v_u64 = vec![10_u64, 20, 30, 40];
    let v_i32 = vec![-5_i32, 15, 25, 35];
    let v_u32 = vec![7_u32, 11, 13, 17];
    let v_i16 = vec![-2_i16, 4, -6, 8];
    let v_u8 = vec![1_u8, 2, 3, 4];
    let v_f64 = vec![1.5_f64, 2.5, 3.5, 4.5];
    let v_f32 = vec![1.0_f32, 2.0, 3.0, 4.0];
    let v_u16 = vec![100_u16, 200, 300, 400];

    // ---------- Session 1: create pager + table, write a batch ----------
    {
        let pager = Arc::new(SimdRDrivePager::open(&path).expect("open SimdRDrivePager"));
        let table = Table::new(TID, pager).expect("new table");

        let schema = Arc::new(Schema::new(vec![
            // row_id must be non-nullable u64
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            field_with_fid("u64_col", DataType::UInt64, F_U64, false),
            field_with_fid("i32_col", DataType::Int32, F_I32, false),
            field_with_fid("u32_col", DataType::UInt32, F_U32, false),
            field_with_fid("i16_col", DataType::Int16, F_I16, false),
            field_with_fid("u8_col", DataType::UInt8, F_U8, false),
            field_with_fid("f64_col", DataType::Float64, F_F64, false),
            field_with_fid("f32_col", DataType::Float32, F_F32, false),
            field_with_fid("u16_col", DataType::UInt16, F_U16, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(rows.clone())),
                Arc::new(UInt64Array::from(v_u64.clone())),
                Arc::new(Int32Array::from(v_i32.clone())),
                Arc::new(UInt32Array::from(v_u32.clone())),
                Arc::new(Int16Array::from(v_i16.clone())),
                Arc::new(UInt8Array::from(v_u8.clone())),
                Arc::new(Float64Array::from(v_f64.clone())),
                Arc::new(Float32Array::from(v_f32.clone())),
                Arc::new(UInt16Array::from(v_u16.clone())),
            ],
        )
        .unwrap();

        table.append(&batch).expect("append");
        // drop(table), drop(pager) at end of scope
    }

    // ---------- Session 2: reopen from same file and verify ----------
    {
        let pager = Arc::new(SimdRDrivePager::open(&path).expect("reopen SimdRDrivePager"));
        let table = Table::new(TID, pager).expect("reopen table");
        let store = table.store();

        use llkv_column_map::types::{LogicalFieldId, Namespace};
        let lfid = |fid: FieldId| {
            LogicalFieldId::new()
                .with_namespace(Namespace::UserData)
                .with_table_id(TID)
                .with_field_id(fid)
        };

        // Helper to assert a primitive column exactly matches `expect`.
        macro_rules! assert_prim_eq {
            ($fid:expr, $dt:expr, $cast:ty, $expect:expr) => {{
                assert_eq!(store.data_type(lfid($fid)).unwrap(), $dt);
                let arr = store.gather_rows(lfid($fid), &rows).unwrap();
                let arr = arr.as_any().downcast_ref::<$cast>().unwrap();
                assert_eq!(arr.values(), $expect.as_slice());
            }};
        }

        assert_prim_eq!(F_U64, DataType::UInt64, UInt64Array, v_u64);
        assert_prim_eq!(F_I32, DataType::Int32, Int32Array, v_i32);
        assert_prim_eq!(F_U32, DataType::UInt32, UInt32Array, v_u32);
        assert_prim_eq!(F_I16, DataType::Int16, Int16Array, v_i16);
        assert_prim_eq!(F_U8, DataType::UInt8, UInt8Array, v_u8);
        assert_prim_eq!(F_F64, DataType::Float64, Float64Array, v_f64);
        assert_prim_eq!(F_F32, DataType::Float32, Float32Array, v_f32);
        assert_prim_eq!(F_U16, DataType::UInt16, UInt16Array, v_u16);
    }
}
