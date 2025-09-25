use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    Int8Array, Int16Array, Int32Array, Int64Array, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, RangeKey, ScanBuilder, ScanOptions,
};
use llkv_column_map::store::{ColumnStore, IndexKind};
use llkv_column_map::types::{LogicalFieldId, Namespace};

use rand::seq::SliceRandom;

fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

fn make_schema(field_id: LogicalFieldId, dt: DataType) -> Arc<Schema> {
    let rid = Field::new("row_id", DataType::UInt64, false);
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let data_f = Field::new("data", dt, false).with_metadata(md);
    Arc::new(Schema::new(vec![rid, data_f]))
}

fn append_random<T, A>(
    store: &ColumnStore<MemPager>,
    field_id: LogicalFieldId,
    dt: DataType,
    values: &mut [T],
) where
    T: Copy + Ord + Into<i128>,
    A: From<Vec<T>> + arrow::array::Array + 'static,
{
    let schema = make_schema(field_id, dt);
    let n = values.len();
    let rid: Vec<u64> = (0..n as u64).collect();
    let rid_arr = Arc::new(UInt64Array::from(rid));
    let vals_arr_any: arrow::array::ArrayRef = {
        let a: A = A::from(values.to_vec());
        Arc::new(a) as arrow::array::ArrayRef
    };
    let batch = RecordBatch::try_new(schema, vec![rid_arr, Arc::clone(&vals_arr_any)]).unwrap();
    store.append(&batch).unwrap();
}

macro_rules! impl_collect_sorted_methods {
    ($($method:ident, $arr:ty),+ $(,)?) => {
        $(
            fn $method(&mut self, a: &$arr, start: usize, len: usize) {
                let end = start + len;
                for i in start..end {
                    self.values.push(a.value(i) as i128);
                }
            }
        )+
    };
}

struct CollectSortedI128 {
    values: Vec<i128>,
}

impl CollectSortedI128 {
    fn new() -> Self {
        Self { values: Vec::new() }
    }
}

impl PrimitiveVisitor for CollectSortedI128 {}
impl PrimitiveWithRowIdsVisitor for CollectSortedI128 {}
impl PrimitiveSortedWithRowIdsVisitor for CollectSortedI128 {}
impl PrimitiveSortedVisitor for CollectSortedI128 {
    impl_collect_sorted_methods!(
        u64_run,
        UInt64Array,
        u32_run,
        UInt32Array,
        u16_run,
        UInt16Array,
        u8_run,
        UInt8Array,
        i64_run,
        Int64Array,
        i32_run,
        Int32Array,
        i16_run,
        Int16Array,
        i8_run,
        Int8Array
    );
}

macro_rules! impl_collect_row_ids_methods {
    ($($method:ident, $arr:ty),+ $(,)?) => {
        $(
            fn $method(
                &mut self,
                _values: &$arr,
                rids: &UInt64Array,
                start: usize,
                len: usize,
            ) {
                let end = start + len;
                for i in start..end {
                    self.rids.push(rids.value(i));
                }
            }
        )+
    };
}

struct CollectRowIds {
    rids: Vec<u64>,
}

impl CollectRowIds {
    fn new() -> Self {
        Self { rids: Vec::new() }
    }
}

impl PrimitiveVisitor for CollectRowIds {}
impl PrimitiveSortedVisitor for CollectRowIds {}
impl PrimitiveWithRowIdsVisitor for CollectRowIds {}
impl PrimitiveSortedWithRowIdsVisitor for CollectRowIds {
    impl_collect_row_ids_methods!(
        u64_run_with_rids,
        UInt64Array,
        u32_run_with_rids,
        UInt32Array,
        u16_run_with_rids,
        UInt16Array,
        u8_run_with_rids,
        UInt8Array,
        i64_run_with_rids,
        Int64Array,
        i32_run_with_rids,
        Int32Array,
        i16_run_with_rids,
        Int16Array,
        i8_run_with_rids,
        Int8Array
    );
}

fn run_builder_range_case<T, Arr>(mut values: Vec<T>, dt: DataType, low: T, high: T)
where
    T: Copy + Ord + Into<i128> + RangeKey,
    Arr: From<Vec<T>> + arrow::array::Array + 'static,
{
    let mut rng = rand::rng();
    values.shuffle(&mut rng);

    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(Arc::clone(&pager)).unwrap();
    let field_id = fid(1);

    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("data", dt, false).with_metadata(md),
    ]));

    let rid: Vec<u64> = (0..values.len() as u64).collect();
    let rid_arr = Arc::new(UInt64Array::from(rid.clone()));
    let val_arr = Arc::new(Arr::from(values.clone())) as arrow::array::ArrayRef;
    let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();
    store.append(&batch).unwrap();
    store.register_index(field_id, IndexKind::Sort).unwrap();

    let mut sorted_pairs: Vec<(i128, u64)> = values
        .iter()
        .enumerate()
        .map(|(idx, &v)| (v.into(), rid[idx]))
        .collect();
    sorted_pairs.sort_by_key(|(value, _)| *value);
    let low_i128 = low.into();
    let high_i128 = high.into();
    let mut expected_values = Vec::new();
    let mut expected_rids = Vec::new();
    for (v, r) in sorted_pairs {
        if v >= low_i128 && v <= high_i128 {
            expected_values.push(v);
            expected_rids.push(r);
        }
    }

    let mut val_collect = CollectSortedI128::new();
    ScanBuilder::new(&store, field_id)
        .options(ScanOptions {
            sorted: true,
            ..Default::default()
        })
        .with_range::<T, _>(low..=high)
        .run(&mut val_collect)
        .unwrap();

    let mut rid_collect = CollectRowIds::new();
    ScanBuilder::new(&store, field_id)
        .options(ScanOptions {
            sorted: true,
            with_row_ids: true,
            ..Default::default()
        })
        .with_range::<T, _>(low..=high)
        .run(&mut rid_collect)
        .unwrap();

    assert_eq!(val_collect.values, expected_values);
    assert_eq!(rid_collect.rids, expected_rids);
}

#[test]
fn scan_all_integer_types_sorted_and_ranges() {
    const N: usize = 200_000;
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    let mut rng = rand::rng();

    // Helper to test one type end-to-end
    let mut test_type = |dt: DataType, gen_fn: &mut dyn FnMut(usize) -> Vec<i128>| {
        let field_id = fid(match dt {
            DataType::Int8 => 1,
            DataType::Int16 => 2,
            DataType::Int32 => 3,
            DataType::Int64 => 4,
            DataType::UInt8 => 5,
            DataType::UInt16 => 6,
            DataType::UInt32 => 7,
            DataType::UInt64 => 8,
            _ => 999,
        });
        let mut vals_i128 = gen_fn(N);
        vals_i128.as_mut_slice().shuffle(&mut rng);

        // Append as concrete array
        match dt {
            DataType::Int8 => {
                let mut v: Vec<i8> = vals_i128.iter().map(|&x| x as i8).collect();
                append_random::<i8, Int8Array>(&store, field_id, dt.clone(), &mut v);
            }
            DataType::Int16 => {
                let mut v: Vec<i16> = vals_i128.iter().map(|&x| x as i16).collect();
                append_random::<i16, Int16Array>(&store, field_id, dt.clone(), &mut v);
            }
            DataType::Int32 => {
                let mut v: Vec<i32> = vals_i128.iter().map(|&x| x as i32).collect();
                append_random::<i32, Int32Array>(&store, field_id, dt.clone(), &mut v);
            }
            DataType::Int64 => {
                let mut v: Vec<i64> = vals_i128.iter().map(|&x| x as i64).collect();
                append_random::<i64, Int64Array>(&store, field_id, dt.clone(), &mut v);
            }
            DataType::UInt8 => {
                let mut v: Vec<u8> = vals_i128.iter().map(|&x| x as u8).collect();
                append_random::<u8, UInt8Array>(&store, field_id, dt.clone(), &mut v);
            }
            DataType::UInt16 => {
                let mut v: Vec<u16> = vals_i128.iter().map(|&x| x as u16).collect();
                append_random::<u16, UInt16Array>(&store, field_id, dt.clone(), &mut v);
            }
            DataType::UInt32 => {
                let mut v: Vec<u32> = vals_i128.iter().map(|&x| x as u32).collect();
                append_random::<u32, UInt32Array>(&store, field_id, dt.clone(), &mut v);
            }
            DataType::UInt64 => {
                let mut v: Vec<u64> = vals_i128.iter().map(|&x| x as u64).collect();
                append_random::<u64, UInt64Array>(&store, field_id, dt.clone(), &mut v);
            }
            _ => unreachable!(),
        }

        // Sort index
        store.register_index(field_id, IndexKind::Sort).unwrap();

        // Validate ascending order and collect using visitor
        let asc: RefCell<Vec<i128>> = RefCell::new(Vec::with_capacity(N));
        struct CollectAsc<'a> {
            out: &'a RefCell<Vec<i128>>,
        }
        impl<'a> llkv_column_map::store::scan::PrimitiveVisitor for CollectAsc<'a> {}
        impl<'a> llkv_column_map::store::scan::PrimitiveWithRowIdsVisitor for CollectAsc<'a> {}
        impl<'a> llkv_column_map::store::scan::PrimitiveSortedWithRowIdsVisitor for CollectAsc<'a> {}
        impl<'a> PrimitiveSortedVisitor for CollectAsc<'a> {
            fn u64_run(&mut self, a: &UInt64Array, s: usize, l: usize) {
                let e = s + l;
                let mut v = self.out.borrow_mut();
                for i in s..e {
                    v.push(a.value(i) as i128);
                }
            }
            fn u32_run(&mut self, a: &UInt32Array, s: usize, l: usize) {
                let e = s + l;
                let mut v = self.out.borrow_mut();
                for i in s..e {
                    v.push(a.value(i) as i128);
                }
            }
            fn u16_run(&mut self, a: &UInt16Array, s: usize, l: usize) {
                let e = s + l;
                let mut v = self.out.borrow_mut();
                for i in s..e {
                    v.push(a.value(i) as i128);
                }
            }
            fn u8_run(&mut self, a: &UInt8Array, s: usize, l: usize) {
                let e = s + l;
                let mut v = self.out.borrow_mut();
                for i in s..e {
                    v.push(a.value(i) as i128);
                }
            }
            fn i64_run(&mut self, a: &Int64Array, s: usize, l: usize) {
                let e = s + l;
                let mut v = self.out.borrow_mut();
                for i in s..e {
                    v.push(a.value(i) as i128);
                }
            }
            fn i32_run(&mut self, a: &Int32Array, s: usize, l: usize) {
                let e = s + l;
                let mut v = self.out.borrow_mut();
                for i in s..e {
                    v.push(a.value(i) as i128);
                }
            }
            fn i16_run(&mut self, a: &Int16Array, s: usize, l: usize) {
                let e = s + l;
                let mut v = self.out.borrow_mut();
                for i in s..e {
                    v.push(a.value(i) as i128);
                }
            }
            fn i8_run(&mut self, a: &Int8Array, s: usize, l: usize) {
                let e = s + l;
                let mut v = self.out.borrow_mut();
                for i in s..e {
                    v.push(a.value(i) as i128);
                }
            }
        }
        let mut visitor = CollectAsc { out: &asc };
        store
            .scan(
                field_id,
                ScanOptions {
                    sorted: true,
                    reverse: false,
                    with_row_ids: false,

                    limit: None,
                    offset: 0,
                    include_nulls: false,
                    nulls_first: false,
                    anchor_row_id_field: None,
                },
                &mut visitor,
            )
            .unwrap();
        let asc = asc.into_inner();
        // Accept that non-matching closures are no-ops for other types (we used typed append above)
        // Verify ascending is sorted
        assert!(asc.windows(2).all(|w| w[0] <= w[1]));

        // Reverse check using store's reverse scanner
        let desc: RefCell<Vec<i128>> = RefCell::new(Vec::with_capacity(N));
        struct CollectDesc<'a> {
            out: &'a RefCell<Vec<i128>>,
        }
        impl<'a> llkv_column_map::store::scan::PrimitiveVisitor for CollectDesc<'a> {}
        impl<'a> llkv_column_map::store::scan::PrimitiveWithRowIdsVisitor for CollectDesc<'a> {}
        impl<'a> llkv_column_map::store::scan::PrimitiveSortedWithRowIdsVisitor for CollectDesc<'a> {}
        impl<'a> PrimitiveSortedVisitor for CollectDesc<'a> {
            fn u64_run(&mut self, a: &UInt64Array, s: usize, l: usize) {
                let mut v = self.out.borrow_mut();
                let mut i = s + l;
                while i > s {
                    i -= 1;
                    v.push(a.value(i) as i128);
                }
            }
            fn u32_run(&mut self, a: &UInt32Array, s: usize, l: usize) {
                let mut v = self.out.borrow_mut();
                let mut i = s + l;
                while i > s {
                    i -= 1;
                    v.push(a.value(i) as i128);
                }
            }
            fn u16_run(&mut self, a: &UInt16Array, s: usize, l: usize) {
                let mut v = self.out.borrow_mut();
                let mut i = s + l;
                while i > s {
                    i -= 1;
                    v.push(a.value(i) as i128);
                }
            }
            fn u8_run(&mut self, a: &UInt8Array, s: usize, l: usize) {
                let mut v = self.out.borrow_mut();
                let mut i = s + l;
                while i > s {
                    i -= 1;
                    v.push(a.value(i) as i128);
                }
            }
            fn i64_run(&mut self, a: &Int64Array, s: usize, l: usize) {
                let mut v = self.out.borrow_mut();
                let mut i = s + l;
                while i > s {
                    i -= 1;
                    v.push(a.value(i) as i128);
                }
            }
            fn i32_run(&mut self, a: &Int32Array, s: usize, l: usize) {
                let mut v = self.out.borrow_mut();
                let mut i = s + l;
                while i > s {
                    i -= 1;
                    v.push(a.value(i) as i128);
                }
            }
            fn i16_run(&mut self, a: &Int16Array, s: usize, l: usize) {
                let mut v = self.out.borrow_mut();
                let mut i = s + l;
                while i > s {
                    i -= 1;
                    v.push(a.value(i) as i128);
                }
            }
            fn i8_run(&mut self, a: &Int8Array, s: usize, l: usize) {
                let mut v = self.out.borrow_mut();
                let mut i = s + l;
                while i > s {
                    i -= 1;
                    v.push(a.value(i) as i128);
                }
            }
        }
        let mut visitor_desc = CollectDesc { out: &desc };
        store
            .scan(
                field_id,
                ScanOptions {
                    sorted: true,
                    reverse: true,
                    with_row_ids: false,

                    limit: None,
                    offset: 0,
                    include_nulls: false,
                    nulls_first: false,
                    anchor_row_id_field: None,
                },
                &mut visitor_desc,
            )
            .unwrap();
        let desc = desc.into_inner();
        assert_eq!(desc.len(), N);
        // desc should be strictly descending (allow equals for duplicate values)
        assert!(desc.windows(2).all(|w| w[0] >= w[1]));
        // And is the reverse of asc
        let mut asc_r = asc.clone();
        asc_r.reverse();
        assert_eq!(desc, asc_r);

        // Range scan emulation via closure (phase 1): lo..=hi
        let lo = asc[N / 3];
        let hi = asc[2 * N / 3];
        let ranged: RefCell<Vec<i128>> = RefCell::new(Vec::new());
        struct CollectRange<'a> {
            out: &'a RefCell<Vec<i128>>,
            lo: i128,
            hi: i128,
        }
        impl<'a> llkv_column_map::store::scan::PrimitiveVisitor for CollectRange<'a> {}
        impl<'a> llkv_column_map::store::scan::PrimitiveWithRowIdsVisitor for CollectRange<'a> {}
        impl<'a> llkv_column_map::store::scan::PrimitiveSortedWithRowIdsVisitor for CollectRange<'a> {}
        impl<'a> PrimitiveSortedVisitor for CollectRange<'a> {
            fn u64_run(&mut self, a: &UInt64Array, s: usize, l: usize) {
                let e = s + l;
                let mut w = self.out.borrow_mut();
                for i in s..e {
                    let v = a.value(i) as i128;
                    if v >= self.lo && v <= self.hi {
                        w.push(v);
                    }
                }
            }
            fn u32_run(&mut self, a: &UInt32Array, s: usize, l: usize) {
                let e = s + l;
                let mut w = self.out.borrow_mut();
                for i in s..e {
                    let v = a.value(i) as i128;
                    if v >= self.lo && v <= self.hi {
                        w.push(v);
                    }
                }
            }
            fn u16_run(&mut self, a: &UInt16Array, s: usize, l: usize) {
                let e = s + l;
                let mut w = self.out.borrow_mut();
                for i in s..e {
                    let v = a.value(i) as i128;
                    if v >= self.lo && v <= self.hi {
                        w.push(v);
                    }
                }
            }
            fn u8_run(&mut self, a: &UInt8Array, s: usize, l: usize) {
                let e = s + l;
                let mut w = self.out.borrow_mut();
                for i in s..e {
                    let v = a.value(i) as i128;
                    if v >= self.lo && v <= self.hi {
                        w.push(v);
                    }
                }
            }
            fn i64_run(&mut self, a: &Int64Array, s: usize, l: usize) {
                let e = s + l;
                let mut w = self.out.borrow_mut();
                for i in s..e {
                    let v = a.value(i) as i128;
                    if v >= self.lo && v <= self.hi {
                        w.push(v);
                    }
                }
            }
            fn i32_run(&mut self, a: &Int32Array, s: usize, l: usize) {
                let e = s + l;
                let mut w = self.out.borrow_mut();
                for i in s..e {
                    let v = a.value(i) as i128;
                    if v >= self.lo && v <= self.hi {
                        w.push(v);
                    }
                }
            }
            fn i16_run(&mut self, a: &Int16Array, s: usize, l: usize) {
                let e = s + l;
                let mut w = self.out.borrow_mut();
                for i in s..e {
                    let v = a.value(i) as i128;
                    if v >= self.lo && v <= self.hi {
                        w.push(v);
                    }
                }
            }
            fn i8_run(&mut self, a: &Int8Array, s: usize, l: usize) {
                let e = s + l;
                let mut w = self.out.borrow_mut();
                for i in s..e {
                    let v = a.value(i) as i128;
                    if v >= self.lo && v <= self.hi {
                        w.push(v);
                    }
                }
            }
        }
        let mut visitor2 = CollectRange {
            out: &ranged,
            lo,
            hi,
        };
        store
            .scan(
                field_id,
                ScanOptions {
                    sorted: true,
                    reverse: false,
                    with_row_ids: false,

                    limit: None,
                    offset: 0,
                    include_nulls: false,
                    nulls_first: false,
                    anchor_row_id_field: None,
                },
                &mut visitor2,
            )
            .unwrap();
        let ranged = ranged.into_inner();
        assert!(ranged.windows(2).all(|w| w[0] <= w[1]));
        assert!(ranged.first().unwrap() >= &lo && ranged.last().unwrap() <= &hi);
    };

    // Generators per dtype
    let gen_signed = |n: usize, bits: u32| -> Vec<i128> {
        let range = 1i128 << (bits - 1);
        (0..n).map(|i| i as i128 % (2 * range) - range).collect()
    };
    let gen_unsigned = |n: usize, bits: u32| -> Vec<i128> {
        let max = 1i128 << bits;
        (0..n).map(|i| i as i128 % max).collect()
    };

    // Exercise all integer types
    test_type(DataType::Int8, &mut |n| gen_signed(n, 8));
    test_type(DataType::Int16, &mut |n| gen_signed(n, 16));
    test_type(DataType::Int32, &mut |n| gen_signed(n, 32));
    test_type(DataType::Int64, &mut |n| gen_signed(n, 64));
    test_type(DataType::UInt8, &mut |n| gen_unsigned(n, 8));
    test_type(DataType::UInt16, &mut |n| gen_unsigned(n, 16));
    test_type(DataType::UInt32, &mut |n| gen_unsigned(n, 32));
    test_type(DataType::UInt64, &mut |n| gen_unsigned(n, 64));
}

#[test]
fn scan_builder_range_all_integer_types() {
    run_builder_range_case::<i8, Int8Array>(
        (0..200).map(|i| (i - 100) as i8).collect(),
        DataType::Int8,
        -40i8,
        60i8,
    );

    run_builder_range_case::<i16, Int16Array>(
        (0..1024).map(|i| ((i - 512) * 4) as i16).collect(),
        DataType::Int16,
        -1_000i16,
        1_200i16,
    );

    run_builder_range_case::<i32, Int32Array>(
        (0..1024).map(|i| ((i as i64 - 512) * 256) as i32).collect(),
        DataType::Int32,
        -32_000i32,
        40_000i32,
    );

    run_builder_range_case::<i64, Int64Array>(
        (0..1024)
            .map(|i| ((i as i128 - 512) * 1_048_576) as i64)
            .collect(),
        DataType::Int64,
        -50_000_000i64,
        60_000_000i64,
    );

    run_builder_range_case::<u8, UInt8Array>(
        (0..256).map(|i| i as u8).collect(),
        DataType::UInt8,
        30u8,
        180u8,
    );

    run_builder_range_case::<u16, UInt16Array>(
        (0..1024).map(|i| (i as u16) * 3).collect(),
        DataType::UInt16,
        150u16,
        900u16,
    );

    run_builder_range_case::<u32, UInt32Array>(
        (0..1024).map(|i| (i as u32) * 65_537).collect(),
        DataType::UInt32,
        500_000u32,
        40_000_000u32,
    );

    run_builder_range_case::<u64, UInt64Array>(
        (0..1024).map(|i| (i as u64) * 1_000_000).collect(),
        DataType::UInt64,
        50_000_000u64,
        400_000_000u64,
    );
}
