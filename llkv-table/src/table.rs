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
use std::collections::{BTreeMap, HashMap};
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
            // Group column data
            column_writes
                .entry(*field_id)
                .or_default()
                .push((row_id, value.as_slice()));

            // Group index data
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
                let snapshot = primary_index.snapshot();

                // Collect all unique index keys to look them up in one batch
                let index_keys: Vec<u64> = updates.iter().map(|(k, _v)| *k).collect();
                let existing_roots = snapshot.get_many(&index_keys)?;

                let mut new_primary_entries = Vec::new();

                for (i, (index_key, new_row_id)) in updates.into_iter().enumerate() {
                    match &existing_roots[i] {
                        Some(root_id_ref) => {
                            // Secondary tree exists: open it and add the new row_id
                            let root_id_bytes: [u8; 8] = root_id_ref.as_ref().try_into().unwrap();
                            let root_id = u64::from_be_bytes(root_id_bytes);
                            let mut row_id_set =
                                RowIdSetTree::open(self.pager.clone(), root_id, None);
                            row_id_set.insert_many(&[(new_row_id, &[])])?;
                        }
                        None => {
                            // Secondary tree does not exist: create it
                            let mut new_row_id_set =
                                RowIdSetTree::create_empty(self.pager.clone(), None)?;
                            new_row_id_set.insert_many(&[(new_row_id, &[])])?;

                            // Get its root and stage it for insertion into the primary index
                            let new_root_id = new_row_id_set.snapshot().root_id();
                            new_primary_entries.push((index_key, new_root_id.to_be_bytes()));
                        }
                    }
                }

                // Insert all new primary index entries in a single batch
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
        if let Expr::Pred(filter) = expr {
            if let Some(primary_index) = self.indexes.get(&filter.field) {
                if let Operator::Equals(val) = filter.op {
                    let index_key = BigEndianKeyCodec::<u64>::decode_from(val)?;

                    let snapshot = primary_index.snapshot();
                    if let Some(root_id_ref) = snapshot.get(&index_key)? {
                        let root_id_bytes: [u8; 8] = root_id_ref.as_ref().try_into().unwrap();
                        let root_id = u64::from_be_bytes(root_id_bytes);

                        // Open the specific RowIdSetTree for this index value.
                        let row_id_set = RowIdSetTree::open(self.pager.clone(), root_id, None);

                        // Stream all keys (which are the row IDs) from this tree.
                        let original_rx = row_id_set.start_stream_with_opts(ScanOpts::forward());
                        let (tx, transformed_rx) = xchan::unbounded();

                        std::thread::spawn(move || {
                            for (row_id, _empty_value) in original_rx {
                                if tx.send(row_id).is_err() {
                                    break;
                                }
                            }
                        });

                        return Ok(Some(transformed_rx));
                    }
                }
            }
        }

        // If no matching index entry is found, return an empty stream.
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
}
