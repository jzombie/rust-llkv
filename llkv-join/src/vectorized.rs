use crate::JoinType;
use arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch, UInt32Builder};
use arrow::compute::take;
use arrow::datatypes::SchemaRef;
use arrow::row::{RowConverter, SortField};
use llkv_result::Error;
use rustc_hash::FxHashMap;
use std::sync::Arc;

pub type JoinFilter = Box<dyn Fn(&RecordBatch) -> Result<BooleanArray, Error> + Send + Sync>;

enum JoinMap {
    Hash(FxHashMap<Vec<u8>, Vec<u32>>),
    Cross(u32),
}

pub struct VectorizedHashJoinStream<I> {
    schema: SchemaRef,
    left_stream: I,
    right_batch: RecordBatch,
    join_map: JoinMap,
    join_type: JoinType,
    left_key_indices: Vec<usize>,
    left_converter: RowConverter,
    filter: Option<JoinFilter>,
    // Cross join state
    cross_left_batch: Option<RecordBatch>,
    cross_left_row: usize,
    cross_right_row: usize,
    cross_finished: bool,
    cross_left_row_matched: bool,
}

impl<I> VectorizedHashJoinStream<I>
where
    I: Iterator<Item = Result<RecordBatch, Error>>,
{
    pub fn try_new(
        schema: SchemaRef,
        left_stream: I,
        right_batch: RecordBatch,
        join_type: JoinType,
        left_key_indices: Vec<usize>,
        right_key_indices: Vec<usize>,
        filter: Option<JoinFilter>,
    ) -> Result<Self, Error> {
        let join_map = if left_key_indices.is_empty() {
            // Cross Join
            JoinMap::Cross(right_batch.num_rows() as u32)
        } else {
            // Hash Join
            let right_key_columns: Vec<ArrayRef> = right_key_indices
                .iter()
                .map(|&i| Arc::clone(right_batch.column(i)))
                .collect();

            let sort_fields: Vec<SortField> = right_key_columns
                .iter()
                .map(|c| SortField::new(c.data_type().clone()))
                .collect();

            let converter = RowConverter::new(sort_fields.clone())
                .map_err(|e| Error::Internal(e.to_string()))?;

            let rows = converter
                .convert_columns(&right_key_columns)
                .map_err(|e| Error::Internal(e.to_string()))?;

            let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
            for i in 0..rows.num_rows() {
                let row = rows.row(i);
                map.entry(row.as_ref().to_vec()).or_default().push(i as u32);
            }
            JoinMap::Hash(map)
        };

        let sort_fields: Vec<SortField> = right_key_indices
            .iter()
            .map(|&i| SortField::new(right_batch.column(i).data_type().clone()))
            .collect();

        let left_converter =
            RowConverter::new(sort_fields).map_err(|e| Error::Internal(e.to_string()))?;

        Ok(Self {
            schema,
            left_stream,
            right_batch,
            join_map,
            join_type,
            left_key_indices,
            left_converter,
            filter,
            cross_left_batch: None,
            cross_left_row: 0,
            cross_right_row: 0,
            cross_finished: false,
            cross_left_row_matched: false,
        })
    }
}

impl<I> Iterator for VectorizedHashJoinStream<I>
where
    I: Iterator<Item = Result<RecordBatch, Error>>,
{
    type Item = Result<RecordBatch, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.join_map {
            JoinMap::Cross(_) => return self.next_cross_batch(),
            JoinMap::Hash(_) => {}
        }

        self.next_hash_batch()
    }
}

impl<I> VectorizedHashJoinStream<I>
where
    I: Iterator<Item = Result<RecordBatch, Error>>,
{
    const CROSS_CHUNK: usize = 4096;

    fn next_cross_batch(&mut self) -> Option<Result<RecordBatch, Error>> {
        if self.cross_finished {
            return None;
        }

        let right_len = match self.join_map {
            JoinMap::Cross(count) => count as usize,
            _ => unreachable!(),
        };

        // Defensive: ensure stored count reflects the batch we built from.
        debug_assert_eq!(right_len, self.right_batch.num_rows());
        if right_len == 0 {
            tracing::debug!("VectorizedHashJoinStream: right_len is 0, finishing cross join");
            self.cross_finished = true;
            return None;
        }

        // Pull a left batch if we do not have one or exhausted it.
        while self.cross_left_batch.is_none() {
            match self.left_stream.next()? {
                Ok(b) if b.num_rows() == 0 => continue,
                Ok(b) => {
                    self.cross_left_batch = Some(b);
                    self.cross_left_row = 0;
                    self.cross_right_row = 0;
                    self.cross_left_row_matched = false;
                }
                Err(e) => return Some(Err(e)),
            }
        }

        let left_batch = self.cross_left_batch.as_ref().unwrap().clone();
        let left_batch_num_rows = left_batch.num_rows();

        let mut cand_left = UInt32Builder::new();
        let mut cand_right = UInt32Builder::new();

        let mut temp_left = self.cross_left_row;
        let mut temp_right = self.cross_right_row;
        let mut produced = 0;

        while produced < Self::CROSS_CHUNK {
            if temp_left >= left_batch_num_rows {
                break;
            }

            cand_left.append_value(temp_left as u32);
            cand_right.append_value(temp_right as u32);

            produced += 1;

            temp_right += 1;
            if temp_right >= right_len {
                temp_right = 0;
                temp_left += 1;
            }
        }

        let cand_left_arr = cand_left.finish();
        let cand_right_arr = cand_right.finish();

        if cand_left_arr.is_empty() {
            // No data produced but stream not finished yet; recurse to grab next left batch.
            self.cross_left_batch = None;
            self.cross_left_row = 0;
            self.cross_right_row = 0;
            self.cross_left_row_matched = false;
            return self.next_cross_batch();
        }

        // Build candidate batch
        let mut candidate_columns = Vec::new();
        for col in left_batch.columns() {
            match take(col, &cand_left_arr, None) {
                Ok(a) => candidate_columns.push(a),
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            }
        }
        for col in self.right_batch.columns() {
            match take(col, &cand_right_arr, None) {
                Ok(a) => candidate_columns.push(a),
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            }
        }
        let candidate_batch = match RecordBatch::try_new(Arc::clone(&self.schema), candidate_columns) {
            Ok(b) => b,
            Err(e) => return Some(Err(Error::Internal(e.to_string()))),
        };

        // Apply filter
        let mask = if let Some(filter) = &self.filter {
            match filter(&candidate_batch) {
                Ok(m) => m,
                Err(e) => return Some(Err(e)),
            }
        } else {
            BooleanArray::from(vec![true; candidate_batch.num_rows()])
        };

        // Build final indices
        let mut final_left = UInt32Builder::new();
        let mut final_right = UInt32Builder::new();
        let mut matched = self.cross_left_row_matched;

        for i in 0..cand_left_arr.len() {
            let l_idx = cand_left_arr.value(i);
            let r_idx = cand_right_arr.value(i);
            let passed = mask.is_valid(i) && mask.value(i);

            if passed {
                final_left.append_value(l_idx);
                final_right.append_value(r_idx);
                matched = true;
            }

            if r_idx as usize == right_len - 1 {
                if !matched && matches!(self.join_type, JoinType::Left) {
                    final_left.append_value(l_idx);
                    final_right.append_null();
                }
                matched = false;
            }
        }

        self.cross_left_row_matched = matched;
        self.cross_left_row = temp_left;
        self.cross_right_row = temp_right;

        if self.cross_left_row >= left_batch_num_rows {
            self.cross_left_batch = None;
            self.cross_left_row = 0;
            self.cross_right_row = 0;
            self.cross_left_row_matched = false;
        }

        let final_left_arr = final_left.finish();
        let final_right_arr = final_right.finish();

        if final_left_arr.is_empty() {
            return self.next_cross_batch();
        }

        let mut output_columns = Vec::new();

        for col in left_batch.columns() {
            match take(col, &final_left_arr, None) {
                Ok(a) => output_columns.push(a),
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            }
        }

        for col in self.right_batch.columns() {
            match take(col, &final_right_arr, None) {
                Ok(a) => output_columns.push(a),
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            }
        }

        Some(
            RecordBatch::try_new(Arc::clone(&self.schema), output_columns)
                .map_err(|e| Error::Internal(e.to_string())),
        )
    }

    fn next_hash_batch(&mut self) -> Option<Result<RecordBatch, Error>> {
        let map = match &self.join_map {
            JoinMap::Hash(map) => map,
            _ => unreachable!(),
        };

        let left_batch = match self.left_stream.next()? {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };

        let mut cand_left = UInt32Builder::new();
        let mut cand_right = UInt32Builder::new();

        let left_key_columns: Vec<ArrayRef> = self
            .left_key_indices
            .iter()
            .map(|&i| Arc::clone(left_batch.column(i)))
            .collect();

        let rows = match self.left_converter.convert_columns(&left_key_columns) {
            Ok(r) => r,
            Err(e) => return Some(Err(Error::Internal(e.to_string()))),
        };

        for i in 0..rows.num_rows() {
            let row = rows.row(i);
            if let Some(matches) = map.get(row.as_ref()) {
                for &right_idx in matches {
                    cand_left.append_value(i as u32);
                    cand_right.append_value(right_idx);
                }
            }
        }

        let cand_left_arr = cand_left.finish();
        let cand_right_arr = cand_right.finish();

        // Build candidate batch
        let mut candidate_columns = Vec::new();
        for col in left_batch.columns() {
            match take(col, &cand_left_arr, None) {
                Ok(a) => candidate_columns.push(a),
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            }
        }
        for col in self.right_batch.columns() {
            match take(col, &cand_right_arr, None) {
                Ok(a) => candidate_columns.push(a),
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            }
        }
        let candidate_batch = match RecordBatch::try_new(Arc::clone(&self.schema), candidate_columns) {
            Ok(b) => b,
            Err(e) => return Some(Err(Error::Internal(e.to_string()))),
        };

        // Apply filter
        let mask = if let Some(filter) = &self.filter {
            match filter(&candidate_batch) {
                Ok(m) => m,
                Err(e) => return Some(Err(e)),
            }
        } else {
            BooleanArray::from(vec![true; candidate_batch.num_rows()])
        };

        let mut final_left = UInt32Builder::new();
        let mut final_right = UInt32Builder::new();

        let mut matched_left = if matches!(self.join_type, JoinType::Left) {
            vec![false; left_batch.num_rows()]
        } else {
            Vec::new()
        };

        for i in 0..cand_left_arr.len() {
            if mask.is_valid(i) && mask.value(i) {
                let l_idx = cand_left_arr.value(i);
                let r_idx = cand_right_arr.value(i);
                final_left.append_value(l_idx);
                final_right.append_value(r_idx);

                if matches!(self.join_type, JoinType::Left) {
                    matched_left[l_idx as usize] = true;
                }
            }
        }

        if matches!(self.join_type, JoinType::Left) {
            for (i, &matched) in matched_left.iter().enumerate() {
                if !matched {
                    final_left.append_value(i as u32);
                    final_right.append_null();
                }
            }
        }

        let final_left_arr = final_left.finish();
        let final_right_arr = final_right.finish();

        let mut output_columns = Vec::new();
        for col in left_batch.columns() {
            match take(col, &final_left_arr, None) {
                Ok(a) => output_columns.push(a),
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            }
        }
        for col in self.right_batch.columns() {
            match take(col, &final_right_arr, None) {
                Ok(a) => output_columns.push(a),
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            }
        }

        Some(
            RecordBatch::try_new(Arc::clone(&self.schema), output_columns)
                .map_err(|e| Error::Internal(e.to_string())),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn cross_join_with_empty_right_ends_stream() -> Result<(), Error> {
        let left_schema = Arc::new(Schema::new(vec![Field::new("l", DataType::Int32, true)]));
        let right_schema = Arc::new(Schema::new(vec![Field::new("r", DataType::Int32, true)]));

        let left_batch = RecordBatch::try_new(
            left_schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef],
        )?;
        let right_batch = RecordBatch::new_empty(right_schema);

        let output_schema = Arc::new(Schema::new(vec![
            Field::new("l", DataType::Int32, true),
            Field::new("r", DataType::Int32, true),
        ]));

        let left_stream = vec![Ok(left_batch)].into_iter();
        let stream = VectorizedHashJoinStream::try_new(
            output_schema,
            left_stream,
            right_batch,
            JoinType::Inner,
            Vec::new(),
            Vec::new(),
            None,
        )?;

        let batches = stream.collect::<Result<Vec<_>, _>>()?;
        assert!(batches.is_empty());
        Ok(())
    }
}
