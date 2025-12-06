use std::sync::Arc;
use arrow::array::{UInt32Builder, ArrayRef, RecordBatch};
use arrow::compute::take;
use arrow::datatypes::{SchemaRef};
use arrow::row::{RowConverter, SortField};
use rustc_hash::FxHashMap;
use crate::JoinType;
use llkv_result::Error;

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
            
            let rows = converter.convert_columns(&right_key_columns)
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

        let left_converter = RowConverter::new(sort_fields)
            .map_err(|e| Error::Internal(e.to_string()))?;

        Ok(Self {
            schema,
            left_stream,
            right_batch,
            join_map,
            join_type,
            left_key_indices,
            left_converter,
        })
    }
}

impl<I> Iterator for VectorizedHashJoinStream<I>
where
    I: Iterator<Item = Result<RecordBatch, Error>>,
{
    type Item = Result<RecordBatch, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let left_batch = match self.left_stream.next()? {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };

        let mut left_indices = UInt32Builder::new();
        let mut right_indices = UInt32Builder::new();

        match &self.join_map {
            JoinMap::Cross(count) => {
                for i in 0..left_batch.num_rows() {
                    for j in 0..*count {
                        left_indices.append_value(i as u32);
                        right_indices.append_value(j);
                    }
                }
            }
            JoinMap::Hash(map) => {
                let left_key_columns: Vec<ArrayRef> = self.left_key_indices
                    .iter()
                    .map(|&i| Arc::clone(left_batch.column(i)))
                    .collect();
                
                let rows = match self.left_converter.convert_columns(&left_key_columns) {
                    Ok(r) => r,
                    Err(e) => return Some(Err(Error::Internal(e.to_string()))),
                };

                for i in 0..rows.num_rows() {
                    let row = rows.row(i);
                    match map.get(row.as_ref()) { 
                        Some(matches) => {
                            for &right_idx in matches {
                                left_indices.append_value(i as u32);
                                right_indices.append_value(right_idx);
                            }
                        }
                        None => {
                            if matches!(self.join_type, JoinType::Left) {
                                left_indices.append_value(i as u32);
                                right_indices.append_null();
                            }
                        }
                    }
                }
            }
        }

        let left_indices_array = left_indices.finish();
        let right_indices_array = right_indices.finish();

        if left_indices_array.is_empty() {
             return Some(Ok(RecordBatch::new_empty(self.schema.clone())));
        }

        // Take columns
        let mut output_columns = Vec::new();
        
        // Left columns
        for col in left_batch.columns() {
            match take(col, &left_indices_array, None) {
                Ok(a) => output_columns.push(a),
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            }
        }

        // Right columns
        for col in self.right_batch.columns() {
            match take(col, &right_indices_array, None) {
                Ok(a) => output_columns.push(a),
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            }
        }

        match RecordBatch::try_new(Arc::clone(&self.schema), output_columns) {
            Ok(b) => Some(Ok(b)),
            Err(e) => Some(Err(Error::Internal(e.to_string()))),
        }
    }
}
