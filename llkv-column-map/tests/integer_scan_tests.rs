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
use llkv_column_map::store::ColumnStore;
use llkv_column_map::store::scan::{PrimitiveSortedVisitor, ScanOptions};
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
        store.create_sort_index(field_id).unwrap();

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
                    row_id_field: None,
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
                    row_id_field: None,
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
                    row_id_field: None,
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
        (0..n)
            .map(|i| ((i as i128 % (2 * range)) - range))
            .collect()
    };
    let gen_unsigned = |n: usize, bits: u32| -> Vec<i128> {
        let max = 1i128 << bits;
        (0..n).map(|i| (i as i128 % max)).collect()
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
