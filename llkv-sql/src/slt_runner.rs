#![cfg(test)]

use std::path::Path;
use std::sync::Arc;

use sqllogictest::{AsyncDB, DBOutput, DefaultColumnType, Runner};

use crate::SqlEngine;
use crate::SqlResult;
use llkv_storage::pager::MemPager;

struct EngineHarness {
    engine: SqlEngine<MemPager>,
}
impl EngineHarness {
    pub fn new() -> Self {
        let pager = Arc::new(MemPager::default());
        Self {
            engine: SqlEngine::new(pager),
        }
    }
}

#[async_trait::async_trait]
impl AsyncDB for EngineHarness {
    type Error = llkv_result::Error;
    type ColumnType = DefaultColumnType;

    async fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        match self.engine.execute(sql) {
            Ok(mut results) => {
                if results.is_empty() {
                    return Ok(DBOutput::StatementComplete(0));
                }
                let result = results.remove(0);
                match result {
                    llkv_dsl::StatementResult::Select { execution, .. } => {
                        let batches = execution.collect()?;
                        let mut rows: Vec<Vec<String>> = Vec::new();
                        for batch in &batches {
                            for row_idx in 0..batch.num_rows() {
                                let mut row: Vec<String> = Vec::new();
                                for col in 0..batch.num_columns() {
                                    let array = batch.column(col);
                                    let val = match array.data_type() {
                                        arrow::datatypes::DataType::Int64 => {
                                            let a = array
                                                .as_any()
                                                .downcast_ref::<arrow::array::Int64Array>()
                                                .unwrap();
                                            if a.is_null(row_idx) {
                                                "NULL".to_string()
                                            } else {
                                                a.value(row_idx).to_string()
                                            }
                                        }
                                        arrow::datatypes::DataType::UInt64 => {
                                            let a = array
                                                .as_any()
                                                .downcast_ref::<arrow::array::UInt64Array>()
                                                .unwrap();
                                            if a.is_null(row_idx) {
                                                "NULL".to_string()
                                            } else {
                                                a.value(row_idx).to_string()
                                            }
                                        }
                                        arrow::datatypes::DataType::Float64 => {
                                            let a = array
                                                .as_any()
                                                .downcast_ref::<arrow::array::Float64Array>()
                                                .unwrap();
                                            if a.is_null(row_idx) {
                                                "NULL".to_string()
                                            } else {
                                                a.value(row_idx).to_string()
                                            }
                                        }
                                        arrow::datatypes::DataType::Utf8 => {
                                            let a = array
                                                .as_any()
                                                .downcast_ref::<arrow::array::StringArray>()
                                                .unwrap();
                                            if a.is_null(row_idx) {
                                                "NULL".to_string()
                                            } else {
                                                a.value(row_idx).to_string()
                                            }
                                        }
                                        _ => "".to_string(),
                                    };
                                    row.push(val);
                                }
                                rows.push(row);
                            }
                        }

                        let types = if let Some(first) = batches.first() {
                            (0..first.num_columns())
                                .map(|col| match first.column(col).data_type() {
                                    arrow::datatypes::DataType::Int64
                                    | arrow::datatypes::DataType::UInt64 => {
                                        DefaultColumnType::Integer
                                    }
                                    arrow::datatypes::DataType::Float64 => {
                                        DefaultColumnType::FloatingPoint
                                    }
                                    arrow::datatypes::DataType::Utf8 => DefaultColumnType::Text,
                                    _ => DefaultColumnType::Any,
                                })
                                .collect()
                        } else {
                            vec![]
                        };

                        Ok(DBOutput::Rows { types, rows })
                    }
                    llkv_dsl::StatementResult::Insert { rows_inserted, .. } => {
                        Ok(DBOutput::StatementComplete(rows_inserted as u64))
                    }
                    llkv_dsl::StatementResult::Update { rows_updated, .. } => {
                        Ok(DBOutput::StatementComplete(rows_updated as u64))
                    }
                    llkv_dsl::StatementResult::CreateTable { .. } => {
                        Ok(DBOutput::StatementComplete(0))
                    }
                    llkv_dsl::StatementResult::Transaction { .. } => {
                        Ok(DBOutput::StatementComplete(0))
                    }
                }
            }
            Err(e) => Err(e),
        }
    }

    async fn shutdown(&mut self) {}
}

/// Run a single slt file by path using the existing Async Runner.
pub async fn run_slt_file<P: AsRef<Path>>(path: P) -> SqlResult<()> {
    fn expand_loops(lines: &[String]) -> SqlResult<Vec<String>> {
        let mut out: Vec<String> = Vec::new();
        let mut i = 0usize;
        while i < lines.len() {
            let line = lines[i].trim_start().to_string();
            if line.starts_with("loop ") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 4 {
                    return Err(llkv_result::Error::Internal(format!(
                        "malformed loop directive: {}",
                        line
                    )));
                }
                let var = parts[1];
                let start: i64 = parts[2].parse().map_err(|e| {
                    llkv_result::Error::Internal(format!("invalid loop start: {}", e))
                })?;
                let count: i64 = parts[3].parse().map_err(|e| {
                    llkv_result::Error::Internal(format!("invalid loop count: {}", e))
                })?;

                let mut j = i + 1;
                while j < lines.len() && lines[j].trim_start() != "endloop" {
                    j += 1;
                }
                if j >= lines.len() {
                    return Err(llkv_result::Error::Internal(
                        "unterminated loop in slt".to_string(),
                    ));
                }

                let inner = &lines[i + 1..j];
                for k in 0..count {
                    let val = (start + k).to_string();
                    let substituted: Vec<String> = inner
                        .iter()
                        .map(|l| l.replace(&format!("${}", var), &val))
                        .collect();
                    let rec = expand_loops(&substituted)?;
                    out.extend(rec);
                }

                i = j + 1;
            } else {
                out.push(lines[i].clone());
                i += 1;
            }
        }
        Ok(out)
    }

    let text = std::fs::read_to_string(path.as_ref())
        .map_err(|e| llkv_result::Error::Internal(format!("failed to read slt file: {}", e)))?;
    let raw_lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
    let expanded_lines = expand_loops(&raw_lines)?;

    let mut tmp = std::env::temp_dir();
    tmp.push(format!("llkv_slt_{}.slt", std::process::id()));
    let expanded_text = expanded_lines.join("\n");
    std::fs::write(&tmp, expanded_text).map_err(|e| {
        llkv_result::Error::Internal(format!("failed to write temp slt file: {}", e))
    })?;

    let mut runner = Runner::new(|| async { Ok(EngineHarness::new()) });
    runner
        .run_file_async(&tmp)
        .await
        .map_err(|e| llkv_result::Error::Internal(format!("slt runner failed: {}", e)))?;

    let _ = std::fs::remove_file(&tmp);
    Ok(())
}
