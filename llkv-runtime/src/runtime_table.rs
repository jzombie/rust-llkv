use std::sync::Arc;

use rustc_hash::FxHashSet;

use arrow::record_batch::RecordBatch;
use llkv_executor::ExecutorRowBatch;
use llkv_plan::{
    CreateTablePlan, InsertConflictAction, InsertPlan, InsertSource, PlanColumnSpec, PlanValue,
};
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use crate::{RuntimeContext, RuntimeLazyFrame, RuntimeStatementResult, canonical_table_name};

pub struct RuntimeCreateTableBuilder<'ctx, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    ctx: &'ctx RuntimeContext<P>,
    plan: CreateTablePlan,
}

impl<'ctx, P> RuntimeCreateTableBuilder<'ctx, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub(crate) fn new(ctx: &'ctx RuntimeContext<P>, name: &str) -> Self {
        Self {
            ctx,
            plan: CreateTablePlan::new(name),
        }
    }

    pub fn if_not_exists(mut self) -> Self {
        self.plan.if_not_exists = true;
        self
    }

    pub fn or_replace(mut self) -> Self {
        self.plan.or_replace = true;
        self
    }

    pub fn with_column(
        mut self,
        name: impl Into<String>,
        data_type: arrow::datatypes::DataType,
    ) -> Self {
        self.plan
            .columns
            .push(PlanColumnSpec::new(name.into(), data_type, true));
        self
    }

    pub fn with_not_null_column(
        mut self,
        name: impl Into<String>,
        data_type: arrow::datatypes::DataType,
    ) -> Self {
        self.plan
            .columns
            .push(PlanColumnSpec::new(name.into(), data_type, false));
        self
    }

    pub fn with_column_spec(mut self, spec: PlanColumnSpec) -> Self {
        self.plan.columns.push(spec);
        self
    }

    pub fn finish(self) -> Result<RuntimeStatementResult<P>> {
        self.ctx.execute_create_table(self.plan)
    }
}

#[derive(Clone, Debug, Default)]
pub struct RuntimeRow {
    values: Vec<(String, PlanValue)>,
}

impl RuntimeRow {
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    pub fn with(mut self, name: impl Into<String>, value: impl Into<PlanValue>) -> Self {
        self.set(name, value);
        self
    }

    pub fn set(&mut self, name: impl Into<String>, value: impl Into<PlanValue>) -> &mut Self {
        let name = name.into();
        let value = value.into();
        if let Some((_, existing)) = self.values.iter_mut().find(|(n, _)| *n == name) {
            *existing = value;
        } else {
            self.values.push((name, value));
        }
        self
    }

    fn columns(&self) -> Vec<String> {
        self.values.iter().map(|(n, _)| n.clone()).collect()
    }

    fn values_for_columns(&self, columns: &[String]) -> Result<Vec<PlanValue>> {
        let mut out = Vec::with_capacity(columns.len());
        for column in columns {
            let value = self
                .values
                .iter()
                .find(|(name, _)| name == column)
                .ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "insert row missing value for column '{}'",
                        column
                    ))
                })?;
            out.push(value.1.clone());
        }
        Ok(out)
    }
}

pub fn row() -> RuntimeRow {
    RuntimeRow::new()
}

#[doc(hidden)]
pub enum RuntimeInsertRowKind {
    Named {
        columns: Vec<String>,
        values: Vec<PlanValue>,
    },
    Positional(Vec<PlanValue>),
}

pub trait IntoInsertRow {
    fn into_insert_row(self) -> Result<RuntimeInsertRowKind>;
}

impl IntoInsertRow for RuntimeRow {
    fn into_insert_row(self) -> Result<RuntimeInsertRowKind> {
        let row = self;
        if row.values.is_empty() {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one column".into(),
            ));
        }
        let columns = row.columns();
        let values = row.values_for_columns(&columns)?;
        Ok(RuntimeInsertRowKind::Named { columns, values })
    }
}

impl<T> IntoInsertRow for Vec<T>
where
    T: Into<PlanValue>,
{
    fn into_insert_row(self) -> Result<RuntimeInsertRowKind> {
        if self.is_empty() {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one column".into(),
            ));
        }
        Ok(RuntimeInsertRowKind::Positional(
            self.into_iter().map(Into::into).collect(),
        ))
    }
}

impl<T, const N: usize> IntoInsertRow for [T; N]
where
    T: Into<PlanValue>,
{
    fn into_insert_row(self) -> Result<RuntimeInsertRowKind> {
        if N == 0 {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one column".into(),
            ));
        }
        Ok(RuntimeInsertRowKind::Positional(
            self.into_iter().map(Into::into).collect(),
        ))
    }
}

/// Macro to implement `IntoInsertRow` for tuples of various sizes.
///
/// This enables ergonomic tuple-based row insertion syntax:
/// ```no_run
/// # use llkv_runtime::*;
/// # use llkv_storage::pager::mem_pager::MemPager;
/// # let table: RuntimeTableHandle<MemPager> = todo!();
/// table.insert_rows([(1_i64, "alice"), (2_i64, "bob")])?;
/// # Ok::<(), llkv_result::Error>(())
/// ```
///
/// Without this, users would need to use the more verbose `RuntimeRow` builder:
/// ```no_run
/// # use llkv_runtime::*;
/// # use llkv_storage::pager::mem_pager::MemPager;
/// # let table: RuntimeTableHandle<MemPager> = todo!();
/// table.insert_rows([
///     row().with("id", 1_i64).with("name", "alice"),
///     row().with("id", 2_i64).with("name", "bob"),
/// ])?;
/// # Ok::<(), llkv_result::Error>(())
/// ```
///
/// The macro generates implementations for tuples of 1-8 elements, covering
/// most common table schemas. Each tuple element must implement `Into<PlanValue>`.
macro_rules! impl_into_insert_row_tuple {
    ($($type:ident => $value:ident),+) => {
        impl<$($type,)+> IntoInsertRow for ($($type,)+)
        where
            $($type: Into<PlanValue>,)+
        {
            fn into_insert_row(self) -> Result<RuntimeInsertRowKind> {
                let ($($value,)+) = self;
                Ok(RuntimeInsertRowKind::Positional(vec![$($value.into(),)+]))
            }
        }
    };
}

// Generate IntoInsertRow implementations for tuples of size 1 through 8.
// This covers the vast majority of table schemas in practice.
impl_into_insert_row_tuple!(T1 => v1);
impl_into_insert_row_tuple!(T1 => v1, T2 => v2);
impl_into_insert_row_tuple!(T1 => v1, T2 => v2, T3 => v3);
impl_into_insert_row_tuple!(T1 => v1, T2 => v2, T3 => v3, T4 => v4);
impl_into_insert_row_tuple!(T1 => v1, T2 => v2, T3 => v3, T4 => v4, T5 => v5);
impl_into_insert_row_tuple!(T1 => v1, T2 => v2, T3 => v3, T4 => v4, T5 => v5, T6 => v6);
impl_into_insert_row_tuple!(
    T1 => v1,
    T2 => v2,
    T3 => v3,
    T4 => v4,
    T5 => v5,
    T6 => v6,
    T7 => v7
);
impl_into_insert_row_tuple!(
    T1 => v1,
    T2 => v2,
    T3 => v3,
    T4 => v4,
    T5 => v5,
    T6 => v6,
    T7 => v7,
    T8 => v8
);

pub struct RuntimeTableHandle<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    context: Arc<RuntimeContext<P>>,
    display_name: String,
    _canonical_name: String,
}

impl<P> RuntimeTableHandle<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(context: Arc<RuntimeContext<P>>, name: &str) -> Result<Self> {
        let (display_name, canonical_name) = canonical_table_name(name)?;
        context.lookup_table(&canonical_name)?;
        Ok(Self {
            context,
            display_name,
            _canonical_name: canonical_name,
        })
    }

    pub fn lazy(&self) -> Result<RuntimeLazyFrame<P>> {
        RuntimeLazyFrame::scan(Arc::clone(&self.context), &self.display_name)
    }

    pub fn insert_rows<R>(
        &self,
        rows: impl IntoIterator<Item = R>,
    ) -> Result<RuntimeStatementResult<P>>
    where
        R: IntoInsertRow,
    {
        enum InsertMode {
            Named,
            Positional,
        }

        let table = self.context.lookup_table(&self._canonical_name)?;
        let schema = table.schema.as_ref();
        let schema_column_names: Vec<String> =
            schema.columns.iter().map(|col| col.name.clone()).collect();
        let mut normalized_rows: Vec<Vec<PlanValue>> = Vec::new();
        let mut mode: Option<InsertMode> = None;
        let mut column_names: Option<Vec<String>> = None;
        let mut row_count = 0usize;

        for row in rows.into_iter() {
            row_count += 1;
            match row.into_insert_row()? {
                RuntimeInsertRowKind::Named { columns, values } => {
                    if let Some(existing) = &mode {
                        if !matches!(existing, InsertMode::Named) {
                            return Err(Error::InvalidArgumentError(
                                "cannot mix positional and named insert rows".into(),
                            ));
                        }
                    } else {
                        mode = Some(InsertMode::Named);
                        let mut seen =
                            FxHashSet::with_capacity_and_hasher(columns.len(), Default::default());
                        for column in &columns {
                            if !seen.insert(column.clone()) {
                                return Err(Error::InvalidArgumentError(format!(
                                    "duplicate column '{}' in insert row",
                                    column
                                )));
                            }
                        }
                        column_names = Some(columns.clone());
                    }

                    let expected = column_names
                        .as_ref()
                        .expect("column names must be initialized for named insert");
                    if columns != *expected {
                        return Err(Error::InvalidArgumentError(
                            "insert rows must specify the same columns".into(),
                        ));
                    }
                    if values.len() != expected.len() {
                        return Err(Error::InvalidArgumentError(format!(
                            "insert row expected {} values, found {}",
                            expected.len(),
                            values.len()
                        )));
                    }
                    normalized_rows.push(values);
                }
                RuntimeInsertRowKind::Positional(values) => {
                    if let Some(existing) = &mode {
                        if !matches!(existing, InsertMode::Positional) {
                            return Err(Error::InvalidArgumentError(
                                "cannot mix positional and named insert rows".into(),
                            ));
                        }
                    } else {
                        mode = Some(InsertMode::Positional);
                        column_names = Some(schema_column_names.clone());
                    }

                    if values.len() != schema.columns.len() {
                        return Err(Error::InvalidArgumentError(format!(
                            "insert row expected {} values, found {}",
                            schema.columns.len(),
                            values.len()
                        )));
                    }
                    normalized_rows.push(values);
                }
            }
        }

        if row_count == 0 {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one row".into(),
            ));
        }

        let columns = column_names.unwrap_or_else(|| schema_column_names.clone());
        self.insert_row_batch(ExecutorRowBatch {
            columns,
            rows: normalized_rows,
        })
    }

    pub fn insert_row_batch(&self, batch: ExecutorRowBatch) -> Result<RuntimeStatementResult<P>> {
        if batch.rows.is_empty() {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one row".into(),
            ));
        }
        if batch.columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one column".into(),
            ));
        }
        for row in &batch.rows {
            if row.len() != batch.columns.len() {
                return Err(Error::InvalidArgumentError(
                    "insert rows must have values for every column".into(),
                ));
            }
        }

        let plan = InsertPlan {
            table: self.display_name.clone(),
            columns: batch.columns,
            source: InsertSource::Rows(batch.rows),
            on_conflict: InsertConflictAction::None,
        };
        let snapshot = self.context.default_snapshot();
        self.context.insert(plan, snapshot)
    }

    pub fn insert_batches(&self, batches: Vec<RecordBatch>) -> Result<RuntimeStatementResult<P>> {
        let plan = InsertPlan {
            table: self.display_name.clone(),
            columns: Vec::new(),
            source: InsertSource::Batches(batches),
            on_conflict: InsertConflictAction::None,
        };
        let snapshot = self.context.default_snapshot();
        self.context.insert(plan, snapshot)
    }

    pub fn insert_lazy(&self, frame: RuntimeLazyFrame<P>) -> Result<RuntimeStatementResult<P>> {
        let ExecutorRowBatch { columns, rows } = frame.collect_rows()?;
        self.insert_row_batch(ExecutorRowBatch { columns, rows })
    }

    pub fn name(&self) -> &str {
        &self.display_name
    }
}
