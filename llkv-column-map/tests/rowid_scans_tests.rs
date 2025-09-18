use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::store::ColumnStore;
use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanBuilder, ScanOptions,
};
use llkv_column_map::types::{LogicalFieldId, Namespace};

use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};

fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

fn seed_u64_perm(
    n: usize,
    seed: u64,
) -> (
    ColumnStore<MemPager>,
    LogicalFieldId,
    LogicalFieldId,
    Vec<u64>,
    Vec<usize>,
) {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    let field_id = fid(1001);

    // Schema: row_id (u64), data (u64)
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md),
    ]));

    // row_ids 0..n, values shuffled 0..n
    let rid: Vec<u64> = (0..n as u64).collect();
    let mut vals: Vec<u64> = (0..n as u64).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    vals.as_mut_slice().shuffle(&mut rng);
    let mut pos_of_val = vec![0usize; n];
    for (i, &v) in vals.iter().enumerate() {
        pos_of_val[v as usize] = i;
    }

    let rid_arr = Arc::new(UInt64Array::from(rid.clone()));
    let val_arr = Arc::new(UInt64Array::from(vals.clone()));
    let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();
    store.append(&batch).unwrap();
    store.create_sort_index(field_id).unwrap();

    let rid_fid = field_id.with_namespace(Namespace::RowIdShadow);
    (store, field_id, rid_fid, vals, pos_of_val)
}

#[test]
fn row_ids_unsorted_and_sorted_paths_u64() {
    const N: usize = 50_000;
    let (store, fid, rid_fid, vals, pos) = seed_u64_perm(N, 0xABCD_EF01_2345_6789);

    // Unsorted with row ids: should emit in append order, rids 0..N-1 and values == vals
    struct UnsortedCheck {
        r: Vec<u64>,
        v: Vec<u64>,
    }
    impl PrimitiveWithRowIdsVisitor for UnsortedCheck {
        fn u64_chunk_with_rids(&mut self, v: &UInt64Array, r: &UInt64Array) {
            for i in 0..v.len() {
                self.v.push(v.value(i));
                self.r.push(r.value(i));
            }
        }
    }
    impl PrimitiveVisitor for UnsortedCheck {}
    impl PrimitiveSortedVisitor for UnsortedCheck {}
    impl PrimitiveSortedWithRowIdsVisitor for UnsortedCheck {}
    let mut u = UnsortedCheck {
        r: Vec::with_capacity(N),
        v: Vec::with_capacity(N),
    };
    store
        .scan(
            fid,
            ScanOptions {
                sorted: false,
                reverse: false,
                with_row_ids: true,
                
                limit: None,
                offset: 0,
                include_nulls: false,
                nulls_first: false,
                anchor_row_id_field: None,
            },
            &mut u,
        )
        .unwrap();
    assert_eq!(u.r.len(), N);
    assert!(u.r.iter().copied().eq(0..N as u64));
    assert_eq!(u.v, vals);

    // Sorted ascending with row ids: values 0..N-1; rids equal to position of each value in original order
    struct SortedAsc {
        r: Vec<u64>,
        v: Vec<u64>,
    }
    impl PrimitiveSortedWithRowIdsVisitor for SortedAsc {
        fn u64_run_with_rids(&mut self, v: &UInt64Array, r: &UInt64Array, s: usize, l: usize) {
            let e = s + l;
            for i in s..e {
                self.v.push(v.value(i));
                self.r.push(r.value(i));
            }
        }
    }
    impl PrimitiveVisitor for SortedAsc {}
    impl PrimitiveSortedVisitor for SortedAsc {}
    impl PrimitiveWithRowIdsVisitor for SortedAsc {}
    let mut sa = SortedAsc {
        r: Vec::with_capacity(N),
        v: Vec::with_capacity(N),
    };
    store
        .scan(
            fid,
            ScanOptions {
                sorted: true,
                reverse: false,
                with_row_ids: true,
                
                limit: None,
                offset: 0,
                include_nulls: false,
                nulls_first: false,
                anchor_row_id_field: None,
            },
            &mut sa,
        )
        .unwrap();
    assert!(sa.v.len() == N && sa.r.len() == N);
    assert!(sa.v.windows(2).all(|w| w[0] <= w[1]));
    for i in 0..N {
        assert_eq!(sa.v[i] as usize, i);
        assert_eq!(sa.r[i] as usize, pos[i]);
    }

    // Sorted descending with row ids: values N-1..0; rids equal to position of each value
    struct SortedDesc {
        r: Vec<u64>,
        v: Vec<u64>,
    }
    impl PrimitiveSortedWithRowIdsVisitor for SortedDesc {
        fn u64_run_with_rids(&mut self, v: &UInt64Array, r: &UInt64Array, s: usize, l: usize) {
            let mut i = s + l;
            while i > s {
                i -= 1;
                self.v.push(v.value(i));
                self.r.push(r.value(i));
            }
        }
    }
    impl PrimitiveVisitor for SortedDesc {}
    impl PrimitiveSortedVisitor for SortedDesc {}
    impl PrimitiveWithRowIdsVisitor for SortedDesc {}
    let mut sd = SortedDesc {
        r: Vec::with_capacity(N),
        v: Vec::with_capacity(N),
    };
    store
        .scan(
            fid,
            ScanOptions {
                sorted: true,
                reverse: true,
                with_row_ids: true,
                
                limit: None,
                offset: 0,
                include_nulls: false,
                nulls_first: false,
                anchor_row_id_field: None,
            },
            &mut sd,
        )
        .unwrap();
    assert!(sd.v.windows(2).all(|w| w[0] >= w[1]));
    for i in 0..N {
        let v = sd.v[i] as usize;
        assert_eq!(sd.r[i] as usize, pos[v]);
    }

    // Builder with range + row ids: subset correctness
    struct RangeCollect {
        r: Vec<u64>,
        v: Vec<u64>,
    }
    impl PrimitiveSortedWithRowIdsVisitor for RangeCollect {
        fn u64_run_with_rids(&mut self, v: &UInt64Array, r: &UInt64Array, s: usize, l: usize) {
            let e = s + l;
            for i in s..e {
                self.v.push(v.value(i));
                self.r.push(r.value(i));
            }
        }
    }
    impl PrimitiveVisitor for RangeCollect {}
    impl PrimitiveSortedVisitor for RangeCollect {}
    impl PrimitiveWithRowIdsVisitor for RangeCollect {}
    let mut rc = RangeCollect {
        r: Vec::new(),
        v: Vec::new(),
    };
    ScanBuilder::new(&store, fid)
        .options(ScanOptions {
            sorted: true,
            reverse: false,
            with_row_ids: true,
            limit: None,
            offset: 0,
            include_nulls: false,
            nulls_first: false,
            anchor_row_id_field: None,
        })
        .with_range::<u64, _>(10_000..=20_000)
        .run(&mut rc)
        .unwrap();
    assert!(!rc.v.is_empty());
    assert!(rc.v.first().copied() == Some(10_000) && rc.v.last().copied() == Some(20_000));
    for i in 0..rc.v.len() {
        let v = rc.v[i] as usize;
        assert_eq!(rc.r[i] as usize, pos[v]);
    }
}
