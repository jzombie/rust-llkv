#![forbid(unsafe_code)]

use crate::expr::{Expr, Operator};
use crossbeam_channel as xchan;
use llkv_btree::codecs::{BigEndianIdCodec, BigEndianKeyCodec, KeyCodec};
use llkv_btree::errors::Error;
use llkv_btree::iter::{BPlusTreeIter, ScanOpts};
use llkv_btree::pager::{MemPager64, Pager as BTreePager, SharedPager};
use llkv_btree::prelude::*;
use llkv_btree::shared_bplus_tree::SharedBPlusTree;
use llkv_btree::views::value_view::ValueRef;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Bound;

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

    pub fn insert(
        &mut self,
        row_id: RowId,
        data: HashMap<FieldId, (u64, Vec<u8>)>,
    ) -> Result<(), Error> {
        // --- Step 1: Group all writes by FieldId ---
        let mut column_writes: BTreeMap<FieldId, Vec<(RowId, &[u8])>> = BTreeMap::new();
        let mut index_updates: BTreeMap<FieldId, Vec<(u64, RowId)>> = BTreeMap::new();

        for (field_id, (index_key, value)) in &data {
            column_writes
                .entry(*field_id)
                .or_default()
                .push((row_id, value.as_slice()));
            if self.indexes.contains_key(field_id) {
                index_updates
                    .entry(*field_id)
                    .or_default()
                    .push((*index_key, row_id));
            }
        }

        // --- Step 2: Perform batched column writes ---
        for (field_id, writes) in column_writes {
            if let Some(column_tree) = self.columns.get_mut(&field_id) {
                column_tree.insert_many(&writes)?;
            }
        }

        // --- Step 3: Perform batched index writes ---
        for (field_id, updates) in index_updates {
            if let Some(primary_index) = self.indexes.get_mut(&field_id) {
                // --- Step 3a: Partition updates based on whether the index key is new or existing ---
                let mut existing_roots_to_update: BTreeMap<u64, Vec<RowId>> = BTreeMap::new();
                let mut new_roots_to_create: BTreeMap<u64, Vec<RowId>> = BTreeMap::new();

                // Find all unique keys and get their status in one batch
                let unique_keys: Vec<u64> = updates
                    .iter()
                    .map(|(k, _)| *k)
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();

                let snapshot = primary_index.snapshot();
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

                // Partition the updates based on whether the key already exists in the primary index
                for (index_key, row_id) in updates {
                    if existing_roots_map.contains_key(&index_key) {
                        existing_roots_to_update
                            .entry(index_key)
                            .or_default()
                            .push(row_id);
                    } else {
                        new_roots_to_create
                            .entry(index_key)
                            .or_default()
                            .push(row_id);
                    }
                }

                // --- Step 3b: Process updates to existing secondary trees ---
                for (index_key, row_ids) in existing_roots_to_update {
                    let root_id = existing_roots_map.get(&index_key).unwrap();
                    let mut row_id_set = RowIdSetTree::open(self.pager.clone(), *root_id, None);
                    let writes: Vec<_> = row_ids.iter().map(|rid| (*rid, &[][..])).collect();
                    row_id_set.insert_many(&writes)?;
                }

                // --- Step 3c: Process creations of new secondary trees and batch the primary index update ---
                if !new_roots_to_create.is_empty() {
                    let mut new_primary_entries = Vec::new();
                    for (index_key, row_ids) in new_roots_to_create {
                        let mut new_row_id_set =
                            RowIdSetTree::create_empty(self.pager.clone(), None)?;
                        let writes: Vec<_> = row_ids.iter().map(|rid| (*rid, &[][..])).collect();
                        new_row_id_set.insert_many(&writes)?;
                        let new_root_id = new_row_id_set.snapshot().root_id();
                        new_primary_entries.push((index_key, new_root_id.to_be_bytes()));
                    }
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
        if let Expr::Pred(filter) = expr {
            if let Some(primary_index) = self.indexes.get(&filter.field) {
                match filter.op {
                    Operator::Equals(val) => {
                        let index_key = BigEndianKeyCodec::<u64>::decode_from(val)?;
                        let snapshot = primary_index.snapshot();
                        if let Some(root_id_ref) = snapshot.get(&index_key)? {
                            let root_id =
                                u64::from_be_bytes(root_id_ref.as_ref().try_into().unwrap());
                            let row_id_set = RowIdSetTree::open(self.pager.clone(), root_id, None);
                            let row_id_stream =
                                row_id_set.start_stream_with_opts(ScanOpts::forward());
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
                                root_ids.push(u64::from_be_bytes(
                                    root_id_ref.as_ref().try_into().unwrap(),
                                ));
                            }
                        }
                        let (tx, rx) = xchan::unbounded();
                        let pager = self.pager.clone();
                        std::thread::spawn(move || {
                            for root_id in root_ids {
                                let row_id_set = RowIdSetTree::open(pager.clone(), root_id, None);
                                let row_id_rx =
                                    row_id_set.start_stream_with_opts(ScanOpts::forward());
                                for (row_id, _) in row_id_rx {
                                    if tx.send(row_id).is_err() {
                                        return;
                                    }
                                }
                            }
                        });
                        return Ok(Some(rx));
                    }
                    _ => { /* Other operators not implemented yet */ }
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

        table.insert(101, row1).unwrap();
        table.insert(102, row2).unwrap();
        table.insert(103, row3).unwrap();

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

        // Insert data with user IDs from 100 to 105
        table
            .insert(
                1,
                HashMap::from([(1, (100u64, vec![])), (2, (0u64, b"data_100".to_vec()))]),
            )
            .unwrap();
        table
            .insert(
                2,
                HashMap::from([(1, (101u64, vec![])), (2, (0u64, b"data_101".to_vec()))]),
            )
            .unwrap();
        table
            .insert(
                3,
                HashMap::from([(1, (102u64, vec![])), (2, (0u64, b"data_102".to_vec()))]),
            )
            .unwrap();
        table
            .insert(
                4,
                HashMap::from([(1, (103u64, vec![])), (2, (0u64, b"data_103".to_vec()))]),
            )
            .unwrap();
        table
            .insert(
                5,
                HashMap::from([(1, (104u64, vec![])), (2, (0u64, b"data_104".to_vec()))]),
            )
            .unwrap();
        table
            .insert(
                6,
                HashMap::from([(1, (105u64, vec![])), (2, (0u64, b"data_105".to_vec()))]),
            )
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
}
