use std::sync::Arc;

use arrow::array::RecordBatch;
use croaring::Treemap;
use llkv_column_map::store::{GatherNullPolicy, MultiGatherContext};
use llkv_expr::Expr;
use llkv_plan::{PlanGraph, ProgramSet};
use llkv_result::{Error, Result as ExecutorResult};
use llkv_table::constants::STREAM_BATCH_ROWS;
use llkv_table::table as table_types;
use llkv_types::{FieldId, LogicalFieldId, TableId};
use simd_r_drive_entry_handle::EntryHandle;

use llkv_storage::pager::Pager;
use llkv_scan::{
    ColumnProjectionInfo, ComputedProjectionInfo, ProjectionEval, RowIdSource, RowStream,
    RowStreamBuilder, ScanProjection, ScanStorage, ScanStreamOptions,
};
use rustc_hash::{FxHashMap, FxHashSet};

impl<P> ScanStorage<P> for crate::types::TableStorageAdapter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn table_id(&self) -> TableId {
        self.table().table_id()
    }

    fn field_data_type(&self, fid: LogicalFieldId) -> ExecutorResult<arrow::datatypes::DataType> {
        self.table()
            .store()
            .data_type(fid)
            .map_err(Error::from)
    }

    fn total_rows(&self) -> ExecutorResult<u64> {
        self.table().total_rows().map_err(Error::from)
    }

    fn prepare_gather_context(
        &self,
        logical_fields: &[LogicalFieldId],
    ) -> ExecutorResult<MultiGatherContext> {
        self.table()
            .store()
            .prepare_gather_context(logical_fields)
            .map_err(Error::from)
    }

    fn gather_row_window_with_context(
        &self,
        logical_fields: &[LogicalFieldId],
        row_ids: &[u64],
        null_policy: GatherNullPolicy,
        ctx: Option<&mut MultiGatherContext>,
    ) -> ExecutorResult<RecordBatch> {
        self.table()
            .store()
            .gather_row_window_with_context(logical_fields, row_ids, null_policy, ctx)
            .map_err(Error::from)
    }

    fn filter_row_ids<'expr>(&self, filter_expr: &Expr<'expr, FieldId>) -> ExecutorResult<Treemap> {
        self.table().filter_row_ids(filter_expr).map_err(Error::from)
    }

    fn stream_row_ids(
        &self,
        chunk_size: usize,
        on_chunk: &mut dyn FnMut(Vec<u64>) -> ExecutorResult<()>,
    ) -> ExecutorResult<()> {
        use llkv_expr::{Expr, Filter, Operator};
        use std::ops::Bound;

        let ids = self.table().filter_row_ids(&Expr::Pred(Filter {
            field_id: llkv_table::ROW_ID_FIELD_ID,
            op: Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            },
        }))?;
        let mut buffer = Vec::new();
        for chunk in ids.iter().collect::<Vec<_>>().chunks(chunk_size.max(1)) {
            buffer.clear();
            buffer.extend_from_slice(chunk);
            on_chunk(buffer.clone())?;
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn to_table_projection(proj: &ScanProjection) -> table_types::ScanProjection {
    match proj {
        ScanProjection::Column(p) => table_types::ScanProjection::Column(p.clone()),
        ScanProjection::Computed { expr, alias } => table_types::ScanProjection::Computed {
            expr: expr.clone(),
            alias: alias.clone(),
        },
    }
}

fn to_table_order(order: llkv_scan::ScanOrderSpec) -> table_types::ScanOrderSpec {
    table_types::ScanOrderSpec {
        field_id: order.field_id,
        direction: match order.direction {
            llkv_scan::ScanOrderDirection::Ascending => table_types::ScanOrderDirection::Ascending,
            llkv_scan::ScanOrderDirection::Descending => table_types::ScanOrderDirection::Descending,
        },
        nulls_first: order.nulls_first,
        transform: match order.transform {
            llkv_scan::ScanOrderTransform::IdentityInt64 => {
                table_types::ScanOrderTransform::IdentityInt64
            }
            llkv_scan::ScanOrderTransform::IdentityInt32 => {
                table_types::ScanOrderTransform::IdentityInt32
            }
            llkv_scan::ScanOrderTransform::IdentityUtf8 => {
                table_types::ScanOrderTransform::IdentityUtf8
            }
            llkv_scan::ScanOrderTransform::CastUtf8ToInteger => {
                table_types::ScanOrderTransform::CastUtf8ToInteger
            }
        },
    }
}

fn to_table_options<P>(options: ScanStreamOptions<P>) -> table_types::ScanStreamOptions<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table_types::ScanStreamOptions {
        include_nulls: options.include_nulls,
        order: options.order.map(to_table_order),
        row_id_filter: options
            .row_id_filter
            .map(|f| std::sync::Arc::new(ScanRowIdFilterShim { inner: f }) as _),
    }
}

struct ScanRowIdFilterShim<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    inner: std::sync::Arc<dyn llkv_scan::RowIdFilter<P>>,
}

impl<P> table_types::RowIdFilter<P> for ScanRowIdFilterShim<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn filter(
        &self,
        table: &llkv_table::table::Table<P>,
        row_ids: Treemap,
    ) -> llkv_result::Result<Treemap> {
        self.inner
            .filter(table.table_id(), table, row_ids)
            .map_err(llkv_result::Error::from)
    }
}

/// Thin wrapper capturing what the executor needs to run a scan.
pub struct ScanExecutor<'a, P, S>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    storage: &'a S,
    _phantom: std::marker::PhantomData<P>,
}

impl<'a, P, S> ScanExecutor<'a, P, S>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    pub fn new(storage: &'a S) -> Self {
        Self {
            storage,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn table_id(&self) -> TableId {
        self.storage.table_id()
    }

    /// Execute a scan using the executor's storage abstraction.
    pub fn execute<'expr, F>(
        &self,
        _plan_graph: PlanGraph,
        _programs: ProgramSet<'expr>,
        projections: &[ScanProjection],
        filter_expr: &Expr<'expr, FieldId>,
        options: ScanStreamOptions<P>,
        on_batch: &mut F,
    ) -> ExecutorResult<()>
    where
        F: FnMut(RecordBatch),
    {
        // Features still routed through llkv-table's scan_stream for correctness.
        if options.order.is_some() || options.row_id_filter.is_some() {
            if let Some(adapter) = self.storage.as_any().downcast_ref::<crate::types::TableStorageAdapter<P>>() {
                let table_projections: Vec<table_types::ScanProjection> =
                    projections.iter().map(to_table_projection).collect();
                let table_options = to_table_options(options);
                return adapter
                    .table()
                    .scan_stream_with_exprs(&table_projections, filter_expr, table_options, on_batch)
                    .map_err(Error::from);
            }
            return Err(Error::InvalidArgumentError(
                "order or row_id_filter requires table-backed storage".into(),
            ));
        }

        if projections.is_empty() {
            return Err(Error::InvalidArgumentError(
                "scan requires at least one projection".into(),
            ));
        }

        // Collect row ids via storage filter.
        let row_ids = self.storage.filter_row_ids(filter_expr)?;
        let row_source = if row_ids.is_empty() {
            RowIdSource::Bitmap(row_ids)
        } else {
            RowIdSource::Bitmap(row_ids)
        };

        // Determine projection evaluation plan and output schema.
        let mut projection_evals = Vec::with_capacity(projections.len());
        let mut unique_index = FxHashMap::default();
        let mut unique_lfids = Vec::new();
        let mut numeric_fields = FxHashSet::default();
        let mut lfid_dtypes = FxHashMap::default();

        for proj in projections {
            match proj {
                ScanProjection::Column(p) => {
                    let lfid = p.logical_field_id;
                    let dtype = self.storage.field_data_type(lfid)?;
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        unique_index.entry(lfid)
                    {
                        entry.insert(unique_lfids.len());
                        unique_lfids.push(lfid);
                    }
                    let fallback = lfid.field_id().to_string();
                    let output_name = p.alias.clone().unwrap_or(fallback);
                    projection_evals.push(ProjectionEval::Column(
                        ColumnProjectionInfo {
                            logical_field_id: lfid,
                            data_type: dtype,
                            output_name,
                        },
                    ));
                }
                ScanProjection::Computed { expr, alias } => {
                    let simplified = llkv_table::NumericKernels::simplify(expr);
                    let mut fields_set: FxHashSet<llkv_table::types::FieldId> =
                        FxHashSet::default();
                    llkv_table::NumericKernels::collect_fields(&simplified, &mut fields_set);
                    for fid in fields_set.iter().copied() {
                        numeric_fields.insert(fid);
                        let lfid = LogicalFieldId::for_user(self.table_id(), fid);
                        let dtype = self.storage.field_data_type(lfid)?;
                        lfid_dtypes.entry(lfid).or_insert_with(|| dtype.clone());
                        if let std::collections::hash_map::Entry::Vacant(entry) =
                            unique_index.entry(lfid)
                        {
                            entry.insert(unique_lfids.len());
                            unique_lfids.push(lfid);
                        }
                    }
                    projection_evals.push(ProjectionEval::Computed(
                        ComputedProjectionInfo {
                            expr: simplified,
                            alias: alias.clone(),
                        },
                    ));
                }
            }
        }

        let passthrough_fields: Vec<Option<llkv_table::types::FieldId>> = projection_evals
            .iter()
            .map(|eval| match eval {
                ProjectionEval::Computed(info) => {
                    llkv_table::NumericKernels::passthrough_column(&info.expr)
                }
                _ => None,
            })
            .collect();

        let null_policy = if options.include_nulls {
            GatherNullPolicy::IncludeNulls
        } else {
            GatherNullPolicy::DropNulls
        };

        let requires_numeric = projection_evals.iter().enumerate().any(|(idx, eval)| {
            matches!(
                eval,
                ProjectionEval::Computed(info)
                if passthrough_fields[idx].is_none()
                    && llkv_compute::analysis::computed_expr_requires_numeric(&info.expr)
            )
        });

        let mut schema_fields: Vec<arrow::datatypes::Field> =
            Vec::with_capacity(projection_evals.len());
        for (idx, eval) in projection_evals.iter().enumerate() {
            match eval {
                ProjectionEval::Column(info) => schema_fields.push(
                    arrow::datatypes::Field::new(
                        info.output_name.clone(),
                        info.data_type.clone(),
                        true,
                    ),
                ),
                ProjectionEval::Computed(info) => {
                    if let Some(fid) = passthrough_fields[idx] {
                        let lfid = LogicalFieldId::for_user(self.table_id(), fid);
                        let dtype = lfid_dtypes.get(&lfid).cloned().ok_or_else(|| {
                            Error::Internal("missing dtype for passthrough".into())
                        })?;
                        schema_fields.push(arrow::datatypes::Field::new(
                            info.alias.clone(),
                            dtype,
                            true,
                        ));
                    } else {
                        let dtype = llkv_compute::projection::infer_computed_dtype(
                            &info.expr,
                            self.table_id(),
                            &lfid_dtypes,
                        )?;
                        schema_fields.push(arrow::datatypes::Field::new(
                            info.alias.clone(),
                            dtype,
                            true,
                        ));
                    }
                }
            }
        }
        let out_schema = Arc::new(arrow::datatypes::Schema::new(schema_fields));

        let mut row_stream = RowStreamBuilder::new(
            self.storage,
            self.table_id(),
            Arc::clone(&out_schema),
            Arc::new(unique_lfids),
            Arc::new(projection_evals),
            Arc::new(passthrough_fields),
            Arc::new(unique_index),
            Arc::new(numeric_fields),
            requires_numeric,
            null_policy,
            row_source,
            STREAM_BATCH_ROWS,
        )
        .build()?;

        let expected_columns = row_stream.schema().fields().len();
        while let Some(chunk) = row_stream.next_chunk()? {
            let batch = chunk.to_record_batch();
            debug_assert_eq!(batch.num_columns(), expected_columns);
            if batch.num_rows() > 0 {
                on_batch(batch);
            }
        }

        Ok(())
    }
}
