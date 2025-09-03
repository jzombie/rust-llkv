#![forbid(unsafe_code)]

use crate::expr::{Expr, Filter, Operator};
use crossbeam_channel as xchan;
use llkv_btree::codecs::{BigEndianIdCodec, BigEndianKeyCodec, KeyCodec};
use llkv_btree::errors::Error;
use llkv_btree::iter::{BPlusTreeIter, ScanOpts};
use llkv_btree::pager::{MemPager64, Pager as BTreePager, SharedPager};
use llkv_btree::prelude::*;
use llkv_btree::shared_bplus_tree::SharedBPlusTree;
use llkv_btree::views::value_view::ValueRef;
use std::collections::{BTreeMap, HashMap};
use std::ops::Bound;
use std::thread;

// --- Correct, Concrete Tree Types for our Table ---
type Pager = SharedPager<MemPager64>;

// Stores column data: maps RowId (u64) -> Value (Vec<u8>)
type ColumnTree = SharedBPlusTree<Pager, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>>;

// A B-Tree that maps an indexed value (e.g., user ID 100) to the root ID of a RowIdSetTree.
type PrimaryIndexTree = SharedBPlusTree<Pager, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>>;

// A B-Tree used as a set to store RowIds for a single indexed value.
type RowIdSetTree = SharedBPlusTree<Pager, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>>;

// --- Type Aliases for Readability ---
pub type FieldId = u32;
pub type RowId = u64;
pub type RowIdCmp = fn(&[u8], &[u8]) -> std::cmp::Ordering;

pub struct TableCfg {
    pub row_id_cmp: RowIdCmp,
}

/// The Table manages a collection of B-Trees for columns and indexes.
pub struct Table {
    columns: BTreeMap<FieldId, ColumnTree>,
    indexes: BTreeMap<FieldId, PrimaryIndexTree>,
    pager: Pager,
}

impl Table {
    pub fn new(page_size: usize) -> Self {
        let base_pager = MemPager64::new(page_size);
        let shared_pager = SharedPager::new(base_pager);
        Self {
            columns: BTreeMap::new(),
            indexes: BTreeMap::new(),
            pager: shared_pager,
        }
    }

    pub fn add_column(&mut self, field_id: FieldId) {
        let column_tree = ColumnTree::create_empty(self.pager.clone(), None).unwrap();
        self.columns.insert(field_id, column_tree);
    }

    pub fn add_index(&mut self, field_id: FieldId) {
        let index_tree = PrimaryIndexTree::create_empty(self.pager.clone(), None).unwrap();
        self.indexes.insert(field_id, index_tree);
    }

    pub fn insert_many(
        &mut self,
        rows: &[(RowId, HashMap<FieldId, (u64, Vec<u8>)>)],
    ) -> Result<(), Error> {
        // --- Step 1: Aggregate all writes across all rows ---
        let mut all_column_writes: BTreeMap<FieldId, Vec<(RowId, &[u8])>> = BTreeMap::new();
        let mut all_index_updates: BTreeMap<FieldId, Vec<(u64, RowId)>> = BTreeMap::new();

        for (row_id, data) in rows {
            for (field_id, (index_key, value)) in data {
                all_column_writes
                    .entry(*field_id)
                    .or_default()
                    .push((*row_id, value.as_slice()));

                if self.indexes.contains_key(field_id) {
                    all_index_updates
                        .entry(*field_id)
                        .or_default()
                        .push((*index_key, *row_id));
                }
            }
        }

        // --- Step 2: Perform batched column writes ---
        for (field_id, writes) in all_column_writes {
            if let Some(column_tree) = self.columns.get_mut(&field_id) {
                column_tree.insert_many(&writes)?;
            }
        }

        // --- Step 3: Perform batched index writes ---
        for (field_id, updates) in all_index_updates {
            if let Some(primary_index) = self.indexes.get_mut(&field_id) {
                // Group updates by index_key to handle multiple rows updating the same index value.
                let mut updates_by_key: BTreeMap<u64, Vec<RowId>> = BTreeMap::new();
                for (index_key, row_id) in updates {
                    updates_by_key.entry(index_key).or_default().push(row_id);
                }

                let unique_keys: Vec<u64> = updates_by_key.keys().cloned().collect();
                let snapshot = primary_index.snapshot();

                // Get all existing secondary tree roots in one batch.
                let existing_roots_map: HashMap<u64, u64> = snapshot
                    .get_many(&unique_keys)?
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, root_ref_opt)| {
                        root_ref_opt.map(|r| {
                            (
                                unique_keys[i],
                                u64::from_be_bytes(r.as_ref().try_into().unwrap()),
                            )
                        })
                    })
                    .collect();

                let mut new_primary_entries = Vec::new();

                for (index_key, row_ids) in updates_by_key {
                    if let Some(root_id) = existing_roots_map.get(&index_key) {
                        // UPDATE existing secondary tree with all its new row_ids in one batch.
                        let mut row_id_set = RowIdSetTree::open(self.pager.clone(), *root_id, None);
                        let writes: Vec<_> = row_ids.iter().map(|rid| (*rid, &[][..])).collect();
                        row_id_set.insert_many(&writes)?;
                    } else {
                        // CREATE new secondary tree and add all its row_ids in one batch.
                        let mut new_row_id_set =
                            RowIdSetTree::create_empty(self.pager.clone(), None)?;
                        let writes: Vec<_> = row_ids.iter().map(|rid| (*rid, &[][..])).collect();
                        new_row_id_set.insert_many(&writes)?;
                        let new_root_id = new_row_id_set.snapshot().root_id();
                        new_primary_entries.push((index_key, new_root_id.to_be_bytes()));
                    }
                }

                // Batch insert all new primary index entries.
                if !new_primary_entries.is_empty() {
                    let writes_slice: Vec<(u64, &[u8])> = new_primary_entries
                        .iter()
                        .map(|(k, v)| (*k, v.as_slice()))
                        .collect();
                    primary_index.insert_many(&writes_slice)?;
                }
            }
        }
        Ok(())
    }

    pub fn scan<'a>(
        &'a self,
        expr: &Expr<'a>,
        projection: &[FieldId],
        on_row: &mut dyn FnMut(
            RowId,
            &mut dyn Iterator<Item = ValueRef<<Pager as BTreePager>::Page>>,
        ),
    ) -> Result<(), Error> {
        let row_id_stream = self.get_row_id_stream(expr)?;
        let row_id_stream = match row_id_stream {
            Some(rx) => rx,
            None => return Ok(()),
        };
        for row_id in row_id_stream {
            let mut value_refs: Vec<ValueRef<<Pager as BTreePager>::Page>> =
                Vec::with_capacity(projection.len());
            for field_id in projection {
                if let Some(column_tree) = self.columns.get(field_id) {
                    if let Some(value_ref) = column_tree.snapshot().get(&row_id)? {
                        value_refs.push(value_ref);
                    }
                }
            }
            let mut values_iter = value_refs.into_iter();
            on_row(row_id, &mut values_iter);
        }
        Ok(())
    }

    fn get_row_id_stream<'b>(
        &'b self,
        expr: &Expr<'b>,
    ) -> Result<Option<xchan::Receiver<u64>>, Error> {
        match expr {
            Expr::Pred(filter) => self.get_stream_for_predicate(filter),
            Expr::And(sub_expressions) => {
                let streams: Vec<_> = sub_expressions
                    .iter()
                    .map(|sub_expr| self.get_row_id_stream(sub_expr))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect();

                if streams.is_empty() {
                    let (tx, rx) = xchan::unbounded();
                    drop(tx);
                    return Ok(Some(rx));
                }

                let (tx, rx) = xchan::unbounded();
                thread::spawn(move || {
                    let mut heads: Vec<Option<u64>> =
                        streams.iter().map(|s| s.recv().ok()).collect();
                    loop {
                        if heads.iter().any(|h| h.is_none()) {
                            break;
                        }
                        let first = heads[0].unwrap();
                        if heads.iter().all(|h| h.unwrap() == first) {
                            if tx.send(first).is_err() {
                                break;
                            }
                            for i in 0..streams.len() {
                                heads[i] = streams[i].recv().ok();
                            }
                        } else {
                            let min_head = heads.iter().filter_map(|&h| h).min().unwrap();
                            for i in 0..streams.len() {
                                if heads[i] == Some(min_head) {
                                    heads[i] = streams[i].recv().ok();
                                }
                            }
                        }
                    }
                });
                Ok(Some(rx))
            }
            Expr::Or(sub_expressions) => {
                let streams: Vec<_> = sub_expressions
                    .iter()
                    .map(|sub_expr| self.get_row_id_stream(sub_expr))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect();

                let (tx, rx) = xchan::unbounded();
                thread::spawn(move || {
                    let mut heads: Vec<Option<u64>> =
                        streams.iter().map(|s| s.recv().ok()).collect();
                    let mut last_sent: Option<u64> = None;
                    loop {
                        let min_head = heads.iter().filter_map(|&h| h).min();
                        if min_head.is_none() {
                            break;
                        }
                        let min_val = min_head.unwrap();
                        if last_sent != Some(min_val) {
                            if tx.send(min_val).is_err() {
                                break;
                            }
                            last_sent = Some(min_val);
                        }
                        for i in 0..streams.len() {
                            if heads[i] == Some(min_val) {
                                heads[i] = streams[i].recv().ok();
                            }
                        }
                    }
                });
                Ok(Some(rx))
            }
            Expr::Not(_) => {
                // `Not` is complex and often requires a full table scan, which is not
                // implemented. Returning an empty stream is a safe default.
                // let (_tx, rx) = xchan::unbounded();
                // Ok(Some(rx))

                unimplemented!("`Not` is currently unimplemented");
            }
        }
    }

    fn get_stream_for_predicate<'b>(
        &'b self,
        filter: &Filter<'b>,
    ) -> Result<Option<xchan::Receiver<u64>>, Error> {
        if let Some(primary_index) = self.indexes.get(&filter.field) {
            match filter.op {
                Operator::Equals(val) => {
                    let index_key = BigEndianKeyCodec::<u64>::decode_from(val)?;
                    let snapshot = primary_index.snapshot();
                    if let Some(root_id_ref) = snapshot.get(&index_key)? {
                        let root_id = u64::from_be_bytes(root_id_ref.as_ref().try_into().unwrap());
                        let row_id_set = RowIdSetTree::open(self.pager.clone(), root_id, None);
                        let row_id_stream = row_id_set.start_stream_with_opts(ScanOpts::forward());
                        let (tx, rx) = xchan::unbounded();
                        std::thread::spawn(move || {
                            for (row_id, _) in row_id_stream {
                                if tx.send(row_id).is_err() {
                                    break;
                                }
                            }
                        });
                        return Ok(Some(rx));
                    }
                }
                Operator::Range { lower, upper } => {
                    let lower_key = match lower {
                        Bound::Included(b) | Bound::Excluded(b) => {
                            Some(BigEndianKeyCodec::<u64>::decode_from(b)?)
                        }
                        Bound::Unbounded => None,
                    };
                    let upper_key = match upper {
                        Bound::Included(b) | Bound::Excluded(b) => {
                            Some(BigEndianKeyCodec::<u64>::decode_from(b)?)
                        }
                        Bound::Unbounded => None,
                    };
                    let lower_bound = match (lower, lower_key.as_ref()) {
                        (Bound::Included(_), Some(k)) => Bound::Included(k),
                        (Bound::Excluded(_), Some(k)) => Bound::Excluded(k),
                        _ => Bound::Unbounded,
                    };
                    let upper_bound = match (upper, upper_key.as_ref()) {
                        (Bound::Included(_), Some(k)) => Bound::Included(k),
                        (Bound::Excluded(_), Some(k)) => Bound::Excluded(k),
                        _ => Bound::Unbounded,
                    };
                    let opts = ScanOpts::forward().with_bounds(lower_bound, upper_bound);
                    let snapshot = primary_index.snapshot();
                    let mut root_ids = Vec::new();
                    if let Ok(primary_iter) = BPlusTreeIter::with_opts(&snapshot, opts) {
                        for (_, root_id_ref) in primary_iter {
                            root_ids
                                .push(u64::from_be_bytes(root_id_ref.as_ref().try_into().unwrap()));
                        }
                    }
                    let (tx, rx) = xchan::unbounded();
                    let pager = self.pager.clone();
                    std::thread::spawn(move || {
                        for root_id in root_ids {
                            let row_id_set = RowIdSetTree::open(pager.clone(), root_id, None);
                            let row_id_rx = row_id_set.start_stream_with_opts(ScanOpts::forward());
                            for (row_id, _) in row_id_rx {
                                if tx.send(row_id).is_err() {
                                    return;
                                }
                            }
                        }
                    });
                    return Ok(Some(rx));
                }
                _ => {
                    unimplemented!()
                }
            }
        }
        let (_tx, rx) = xchan::unbounded();
        Ok(Some(rx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Filter;
    use std::collections::HashSet;

    #[test]
    fn it_works_insert_and_scan_using_index() {
        let mut table = Table::new(4096);
        table.add_column(1); // Name
        table.add_index(1);
        table.add_column(2); // Country

        let row1 = HashMap::from([
            (1, (100u64, b"Alice".to_vec())),
            (2, (0u64, b"USA".to_vec())),
        ]);
        let row2 = HashMap::from([
            (1, (200u64, b"Bob".to_vec())),
            (2, (0u64, b"Canada".to_vec())),
        ]);
        let row3 = HashMap::from([
            (1, (100u64, b"Alice".to_vec())),
            (2, (0u64, b"UK".to_vec())),
        ]);

        table
            .insert_many(&[(101, row1), (102, row2), (103, row3)])
            .unwrap();

        let filter = Filter {
            field: 1,
            op: Operator::Equals(&100u64.to_be_bytes()),
        };
        let expr = Expr::Pred(filter);
        let projection = &[2];

        let mut results = Vec::new();
        table
            .scan(&expr, projection, &mut |row_id, values_iter| {
                let values: Vec<Vec<u8>> = values_iter.map(|v| v.as_slice().to_vec()).collect();
                results.push((row_id, values));
            })
            .unwrap();

        assert_eq!(results.len(), 2);

        let results_set: HashSet<(u64, Vec<Vec<u8>>)> = results.into_iter().collect();
        let expected_set =
            HashSet::from([(101, vec![b"USA".to_vec()]), (103, vec![b"UK".to_vec()])]);

        assert_eq!(results_set, expected_set);
    }

    #[test]
    fn it_works_range_scan_using_index() {
        let mut table = Table::new(4096);
        table.add_column(1); // User ID
        table.add_index(1);
        table.add_column(2); // Data

        table
            .insert_many(&[
                (
                    1,
                    HashMap::from([(1, (100u64, vec![])), (2, (0u64, b"data_100".to_vec()))]),
                ),
                (
                    2,
                    HashMap::from([(1, (101u64, vec![])), (2, (0u64, b"data_101".to_vec()))]),
                ),
                (
                    3,
                    HashMap::from([(1, (102u64, vec![])), (2, (0u64, b"data_102".to_vec()))]),
                ),
                (
                    4,
                    HashMap::from([(1, (103u64, vec![])), (2, (0u64, b"data_103".to_vec()))]),
                ),
                (
                    5,
                    HashMap::from([(1, (104u64, vec![])), (2, (0u64, b"data_104".to_vec()))]),
                ),
                (
                    6,
                    HashMap::from([(1, (105u64, vec![])), (2, (0u64, b"data_105".to_vec()))]),
                ),
            ])
            .unwrap();

        // Define bounds for the range scan: keys >= 102 and < 105
        let lower_bound_bytes = 102u64.to_be_bytes();
        let upper_bound_bytes = 105u64.to_be_bytes();

        let filter = Filter {
            field: 1,
            op: Operator::Range {
                lower: Bound::Included(&lower_bound_bytes),
                upper: Bound::Excluded(&upper_bound_bytes),
            },
        };
        let expr = Expr::Pred(filter);
        let projection = &[2]; // Project the data column

        let mut results = Vec::new();
        table
            .scan(&expr, projection, &mut |row_id, values_iter| {
                let values: Vec<Vec<u8>> = values_iter.map(|v| v.as_slice().to_vec()).collect();
                results.push((row_id, values));
            })
            .unwrap();

        // Expect to find rows with index keys 102, 103, and 104
        assert_eq!(results.len(), 3);

        let results_set: HashSet<(u64, Vec<Vec<u8>>)> = results.into_iter().collect();
        let expected_set = HashSet::from([
            (3, vec![b"data_102".to_vec()]),
            (4, vec![b"data_103".to_vec()]),
            (5, vec![b"data_104".to_vec()]),
        ]);

        assert_eq!(results_set, expected_set);
    }

    #[test]
    fn it_works_complex_multi_column_scans() {
        let mut table = Table::new(4096);

        // --- Setup: 10 columns, 3 of which are indexed ---
        for i in 1..=10 {
            table.add_column(i);
        }
        table.add_index(1); // High cardinality (unique)
        table.add_index(2); // Medium cardinality (100 unique values)
        table.add_index(3); // Low cardinality (5 unique values)

        // --- Insert 5000 rows ---
        let num_rows = 5000;
        let mut rows_to_insert = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let row_id = i as u64;
            let mut row_data = HashMap::new();

            // Indexed columns
            row_data.insert(1, (1000 + row_id, vec![])); // Unique key
            row_data.insert(2, (row_id % 100, vec![])); // 100 different keys
            row_data.insert(3, (row_id % 5, vec![])); // 5 different keys

            // Non-indexed data payload columns
            for j in 4..=10 {
                row_data.insert(j, (0, format!("col{}_row{}", j, row_id).into_bytes()));
            }
            rows_to_insert.push((row_id, row_data));
        }
        table.insert_many(&rows_to_insert).unwrap();

        // --- Test 1: Forward Range Scan on high-cardinality index (Field 1) ---
        let lower_1 = 3000u64.to_be_bytes();
        let upper_1 = 4000u64.to_be_bytes();
        let filter_1 = Filter {
            field: 1,
            op: Operator::Range {
                lower: Bound::Included(&lower_1),
                upper: Bound::Excluded(&upper_1),
            },
        };
        let expr_1 = Expr::Pred(filter_1);
        let mut results_1 = Vec::new();
        table
            .scan(&expr_1, &[4], &mut |row_id, values_iter| {
                results_1.push((row_id, values_iter.last().unwrap().as_slice().to_vec()));
            })
            .unwrap();

        assert_eq!(results_1.len(), 1000); // Should find 1000 rows
        results_1.sort_by_key(|(k, _)| *k);
        assert_eq!(results_1[0].0, 2000); // 1000 + 2000 = 3000
        assert_eq!(results_1[0].1, b"col4_row2000".to_vec());
        assert_eq!(results_1[999].0, 2999); // 1000 + 2999 = 3999
        assert_eq!(results_1[999].1, b"col4_row2999".to_vec());

        // --- Test 2: Equality Scan on medium-cardinality index (Field 2) ---
        let key_2 = 42u64.to_be_bytes();
        let filter_2 = Filter {
            field: 2,
            op: Operator::Equals(&key_2),
        };
        let expr_2 = Expr::Pred(filter_2);
        let mut results_2 = Vec::new();
        table
            .scan(&expr_2, &[], &mut |row_id, _| {
                results_2.push(row_id);
            })
            .unwrap();

        assert_eq!(results_2.len(), 50); // 5000 / 100 = 50
        assert!(results_2.iter().all(|&rid| rid % 100 == 42));

        // --- Test 3: Range Scan on low-cardinality index (Field 3) ---
        let lower_3 = 2u64.to_be_bytes();
        let upper_3 = 4u64.to_be_bytes();
        let filter_3 = Filter {
            field: 3,
            op: Operator::Range {
                lower: Bound::Included(&lower_3),
                upper: Bound::Excluded(&upper_3),
            },
        };
        let expr_3 = Expr::Pred(filter_3);
        let mut results_3 = Vec::new();
        table
            .scan(&expr_3, &[], &mut |row_id, _| {
                results_3.push(row_id);
            })
            .unwrap();

        // Should find rows where row_id % 5 is 2 or 3.
        // Each category has 5000 / 5 = 1000 rows. So 2000 total.
        assert_eq!(results_3.len(), 2000);
        assert!(results_3.iter().all(|&rid| rid % 5 == 2 || rid % 5 == 3));
    }

    #[test]
    fn it_works_intersections_and_unions() {
        let mut table = Table::new(4096);
        table.add_column(1);
        table.add_index(1); // City ID
        table.add_column(2);
        table.add_index(2); // Category ID
        table.add_column(3); // Payload

        let rows = vec![
            (
                10,
                HashMap::from([
                    (1, (1, vec![])),
                    (2, (10, vec![])),
                    (3, (0, b"Restaurant in NYC".to_vec())),
                ]),
            ),
            (
                20,
                HashMap::from([
                    (1, (1, vec![])),
                    (2, (20, vec![])),
                    (3, (0, b"Museum in NYC".to_vec())),
                ]),
            ),
            (
                30,
                HashMap::from([
                    (1, (2, vec![])),
                    (2, (10, vec![])),
                    (3, (0, b"Restaurant in London".to_vec())),
                ]),
            ),
            (
                40,
                HashMap::from([
                    (1, (2, vec![])),
                    (2, (30, vec![])),
                    (3, (0, b"Theatre in London".to_vec())),
                ]),
            ),
            (
                50,
                HashMap::from([
                    (1, (1, vec![])),
                    (2, (10, vec![])),
                    (3, (0, b"Another Restaurant in NYC".to_vec())),
                ]),
            ),
        ];
        table.insert_many(&rows).unwrap();

        // --- Test 1: Intersection (AND) ---
        // Find: Restaurants (category 10) in NYC (city 1)
        let city_nyc = 1u64.to_be_bytes();
        let cat_restaurant = 10u64.to_be_bytes();
        let expr_and = Expr::And(vec![
            Expr::Pred(Filter {
                field: 1,
                op: Operator::Equals(&city_nyc),
            }),
            Expr::Pred(Filter {
                field: 2,
                op: Operator::Equals(&cat_restaurant),
            }),
        ]);

        let mut results_and = Vec::new();
        table
            .scan(&expr_and, &[], &mut |row_id, _| {
                results_and.push(row_id);
            })
            .unwrap();

        let expected_and: HashSet<u64> = [10, 50].iter().cloned().collect();
        let got_and: HashSet<u64> = results_and.into_iter().collect();
        assert_eq!(got_and, expected_and);

        // --- Test 2: Union (OR) ---
        // Find: anything in NYC (city 1) OR any Restaurant (category 10)
        let expr_or = Expr::Or(vec![
            Expr::Pred(Filter {
                field: 1,
                op: Operator::Equals(&city_nyc),
            }),
            Expr::Pred(Filter {
                field: 2,
                op: Operator::Equals(&cat_restaurant),
            }),
        ]);

        let mut results_or = Vec::new();
        table
            .scan(&expr_or, &[], &mut |row_id, _| {
                results_or.push(row_id);
            })
            .unwrap();

        let expected_or: HashSet<u64> = [10, 20, 30, 50].iter().cloned().collect();
        let got_or: HashSet<u64> = results_or.into_iter().collect();
        assert_eq!(got_or, expected_or);
    }

    #[test]
    fn it_works_large_scale_intersections_and_unions() {
        let mut table = Table::new(4096);

        // --- Setup: 10 columns, 3 with indexes ---
        for i in 1..=10 {
            table.add_column(i);
        }
        table.add_index(1); // Country ID (10 values)
        table.add_index(2); // Status ID (5 values)
        table.add_index(3); // Category ID (100 values)

        // --- Insert 15,000 rows ---
        let num_rows = 15_000;
        let mut rows_to_insert = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let row_id = i as u64;
            let mut row_data = HashMap::new();
            row_data.insert(1, (row_id % 10, vec![]));
            row_data.insert(2, (row_id % 5, vec![]));
            row_data.insert(3, (row_id % 100, vec![]));
            for j in 4..=10 {
                row_data.insert(j, (0, vec![]));
            }
            rows_to_insert.push((row_id, row_data));
        }
        table.insert_many(&rows_to_insert).unwrap();

        // --- Test 1: Intersection (AND) ---
        // Find rows where Country=5 AND Status=2
        let country_5 = 5u64.to_be_bytes();
        let status_2 = 2u64.to_be_bytes();
        let expr_and = Expr::And(vec![
            Expr::Pred(Filter {
                field: 1,
                op: Operator::Equals(&country_5),
            }),
            Expr::Pred(Filter {
                field: 2,
                op: Operator::Equals(&status_2),
            }),
        ]);

        let mut results_and = Vec::new();
        table
            .scan(&expr_and, &[], &mut |row_id, _| {
                results_and.push(row_id);
            })
            .unwrap();

        // Expected: row_id % 10 == 5 AND row_id % 5 == 2.
        // This is true for row_ids ending in 5 that are also 2 mod 5 (impossible).
        // Let's adjust: row_id % 10 == 2 AND row_id % 5 == 2. True for all numbers ending in 2.
        // Let's use Chinese Remainder Theorem logic. row_id = 5 (mod 10) and row_id = 2 (mod 5).
        // The first implies row_id = 0 or 5 (mod 5). So only row_id = 5 (mod 10) and row_id = 0 (mod 5) works, which is row_id ends in 5.
        // Let's find row_id % 10 = 7 and row_id % 5 = 2. These are the same condition.
        // Let's do Country=7 and Status=2.
        let country_7 = 7u64.to_be_bytes();
        let status_2 = 2u64.to_be_bytes();
        let expr_and_2 = Expr::And(vec![
            Expr::Pred(Filter {
                field: 1,
                op: Operator::Equals(&country_7),
            }),
            Expr::Pred(Filter {
                field: 2,
                op: Operator::Equals(&status_2),
            }),
        ]);
        let mut results_and_2 = Vec::new();
        table
            .scan(&expr_and_2, &[], &mut |row_id, _| {
                results_and_2.push(row_id);
            })
            .unwrap();

        let mut expected_and = HashSet::new();
        for i in 0..num_rows {
            if i % 10 == 7 && i % 5 == 2 {
                expected_and.insert(i as u64);
            }
        }
        assert_eq!(
            results_and_2.into_iter().collect::<HashSet<u64>>(),
            expected_and
        );
        assert_eq!(expected_and.len(), 1500); // 15000 / 10 = 1500, then check mod 5 is 2. 7 mod 5 is 2. So all of them.

        // --- Test 2: Union (OR) ---
        // Find rows where Status=1 OR Status=3
        let status_1 = 1u64.to_be_bytes();
        let status_3 = 3u64.to_be_bytes();
        let expr_or = Expr::Or(vec![
            Expr::Pred(Filter {
                field: 2,
                op: Operator::Equals(&status_1),
            }),
            Expr::Pred(Filter {
                field: 2,
                op: Operator::Equals(&status_3),
            }),
        ]);

        let mut results_or = Vec::new();
        table
            .scan(&expr_or, &[], &mut |row_id, _| {
                results_or.push(row_id);
            })
            .unwrap();

        let mut expected_or = HashSet::new();
        for i in 0..num_rows {
            if i % 5 == 1 || i % 5 == 3 {
                expected_or.insert(i as u64);
            }
        }
        assert_eq!(
            results_or.into_iter().collect::<HashSet<u64>>(),
            expected_or
        );
        assert_eq!(expected_or.len(), 15000 * 2 / 5); // 6000
    }
}
