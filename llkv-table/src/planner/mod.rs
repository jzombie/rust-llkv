pub mod plan_graph;

use std::cell::RefCell;
use std::cmp::{self, Ordering};
use std::convert::TryFrom;
use std::mem;
use std::ops::Bound;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, OffsetSizeTrait, RecordBatch,
    StringArray, UInt64Array, new_null_array,
};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, Schema};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamOutcome {
    Handled,
    Fallback,
}

impl StreamOutcome {
    #[inline]
    fn is_handled(self) -> bool {
        matches!(self, StreamOutcome::Handled)
    }
}

use llkv_column_map::ScanBuilder;
use llkv_column_map::scan::{FilterPrimitive, FilterRun, dense_row_runs};
use llkv_column_map::store::scan::ScanOptions;
use llkv_column_map::store::{GatherNullPolicy, MultiGatherContext};
use llkv_column_map::types::{LogicalFieldId, Namespace};
use llkv_column_map::{llkv_for_each_arrow_boolean, llkv_for_each_arrow_numeric, llkv_for_each_arrow_string};
use llkv_expr::literal::{FromLiteral, Literal};
use llkv_expr::typed_predicate::{
    PredicateValue, build_bool_predicate, build_fixed_width_predicate, build_var_width_predicate,
};
use llkv_expr::{BinaryOp, CompareOp, Expr, Filter, Operator, ScalarExpr};
use llkv_result::{Error, Result as LlkvResult};
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;

use llkv_storage::pager::Pager;

use crate::constants::STREAM_BATCH_ROWS;
use crate::scalar_eval::{NumericArrayMap, NumericKernels};
use crate::schema_ext::CachedSchema;
use crate::table::{
    ScanOrderDirection, ScanOrderSpec, ScanOrderTransform, ScanProjection, ScanStreamOptions, Table,
};
use crate::types::{FieldId, ROW_ID_FIELD_ID, RowId, TableId};

use self::plan_graph::{
    PlanEdge, PlanExpression, PlanField, PlanGraph, PlanGraphBuilder, PlanGraphError, PlanNode,
    PlanNodeId, PlanOperator,
};
use crate::stream::{RowStream, RowStreamBuilder};

// NOTE: Planning and execution currently live together; once the dedicated
// executor crate stabilizes we can migrate these components into `llkv-plan`.

macro_rules! impl_single_column_emit_chunk {
    (
        $_base:ident,
        $chunk:ident,
        $_chunk_with_rids:ident,
        $_run:ident,
        $_run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $chunk(&mut self, values: &$array_ty) {
            self.emit_array(values);
        }
    };
}

macro_rules! impl_single_column_reject_chunk_with_rids {
    (
        $_base:ident,
        $_chunk:ident,
        $chunk_with_rids:ident,
        $_run:ident,
        $_run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $chunk_with_rids(&mut self, _: &$array_ty, _: &arrow::array::UInt64Array) {
            self.reject_row_ids();
        }
    };
}
macro_rules! impl_single_column_emit_sorted_run {
    (
        $_base:ident,
        $_chunk:ident,
        $_chunk_with_rids:ident,
        $run:ident,
        $_run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $run(&mut self, values: &$array_ty, start: usize, len: usize) {
            if self.error.is_some() || len == 0 {
                return;
            }
            let slice = values.slice(start, len);
            let array: ArrayRef = Arc::new(slice) as ArrayRef;
            self.emit_array_ref(array);
        }
    };
}

macro_rules! impl_single_column_reject_sorted_run_with_rids {
    (
        $_base:ident,
        $_chunk:ident,
        $_chunk_with_rids:ident,
        $_run:ident,
        $run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $run_with_rids(
            &mut self,
            _: &$array_ty,
            _: &arrow::array::UInt64Array,
            _: usize,
            _: usize,
        ) {
            self.reject_row_ids();
        }
    };
}

macro_rules! impl_computed_emit_chunk {
    (
        $_base:ident,
        $chunk:ident,
        $_chunk_with_rids:ident,
        $_run:ident,
        $_run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $chunk(&mut self, values: &$array_ty) {
            self.process_values(values);
        }
    };
}

macro_rules! impl_computed_reject_chunk_with_rids {
    (
        $_base:ident,
        $_chunk:ident,
        $chunk_with_rids:ident,
        $_run:ident,
        $_run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $chunk_with_rids(&mut self, _: &$array_ty, _: &arrow::array::UInt64Array) {
            self.reject_row_ids();
        }
    };
}

macro_rules! impl_computed_emit_sorted_run {
    (
        $_base:ident,
        $_chunk:ident,
        $_chunk_with_rids:ident,
        $run:ident,
        $_run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $run(&mut self, values: &$array_ty, start: usize, len: usize) {
            if self.error.is_some() || len == 0 {
                return;
            }
            let slice = values.slice(start, len);
            let array: ArrayRef = Arc::new(slice) as ArrayRef;
            self.process_array(array);
        }
    };
}

macro_rules! impl_computed_reject_sorted_run_with_rids {
    (
        $_base:ident,
        $_chunk:ident,
        $_chunk_with_rids:ident,
        $_run:ident,
        $run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $run_with_rids(
            &mut self,
            _: &$array_ty,
            _: &arrow::array::UInt64Array,
            _: usize,
            _: usize,
        ) {
            self.reject_row_ids();
        }
    };
}

macro_rules! impl_affine_emit_chunk {
    (
        $_base:ident,
        $chunk:ident,
        $_chunk_with_rids:ident,
        $_run:ident,
        $_run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $cast:expr
    ) => {
        fn $chunk(&mut self, values: &$array_ty) {
            let cast = $cast;
            if values.null_count() == 0 {
                let slice = values.values();
                self.emit_no_nulls(slice.len(), |idx| cast(slice[idx]));
            } else {
                let len = values.len();
                self.emit_with_nulls(
                    len,
                    |idx| values.is_null(idx),
                    |idx| cast(values.value(idx)),
                );
            }
        }
    };
}

macro_rules! impl_affine_reject_chunk_with_rids {
    (
        $_base:ident,
        $_chunk:ident,
        $chunk_with_rids:ident,
        $_run:ident,
        $_run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $chunk_with_rids(&mut self, _: &$array_ty, _: &arrow::array::UInt64Array) {
            self.reject_row_ids();
        }
    };
}

macro_rules! impl_affine_emit_sorted_run {
    (
        $_base:ident,
        $_chunk:ident,
        $_chunk_with_rids:ident,
        $run:ident,
        $_run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $cast:expr
    ) => {
        fn $run(&mut self, values: &$array_ty, start: usize, len: usize) {
            if len == 0 {
                return;
            }
            let cast = $cast;
            if values.null_count() == 0 {
                let slice = values.values();
                self.emit_no_nulls(len, |idx| cast(slice[start + idx]));
            } else {
                self.emit_with_nulls(
                    len,
                    |idx| values.is_null(start + idx),
                    |idx| cast(values.value(start + idx)),
                );
            }
        }
    };
}

macro_rules! impl_affine_reject_sorted_run_with_rids {
    (
        $_base:ident,
        $_chunk:ident,
        $_chunk_with_rids:ident,
        $_run:ident,
        $run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $run_with_rids(
            &mut self,
            _: &$array_ty,
            _: &arrow::array::UInt64Array,
            _: usize,
            _: usize,
        ) {
            self.reject_row_ids();
        }
    };
}

macro_rules! impl_row_id_ignore_chunk {
    (
        $_base:ident,
        $chunk:ident,
        $_chunk_with_rids:ident,
        $_run:ident,
        $_run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $chunk(&mut self, _values: &$array_ty) {}
    };
}

macro_rules! impl_row_id_ignore_sorted_run {
    (
        $_base:ident,
        $_chunk:ident,
        $_chunk_with_rids:ident,
        $run:ident,
        $_run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $run(&mut self, _values: &$array_ty, _start: usize, _len: usize) {}
    };
}

macro_rules! impl_row_id_collect_chunk_with_rids {
    (
        $_base:ident,
        $_chunk:ident,
        $chunk_with_rids:ident,
        $_run:ident,
        $_run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $chunk_with_rids(&mut self, _values: &$array_ty, row_ids: &UInt64Array) {
            self.extend_from_array(row_ids);
        }
    };
}

macro_rules! impl_row_id_collect_sorted_run_with_rids {
    (
        $_base:ident,
        $_chunk:ident,
        $_chunk_with_rids:ident,
        $_run:ident,
        $run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $run_with_rids(
            &mut self,
            _values: &$array_ty,
            row_ids: &UInt64Array,
            start: usize,
            len: usize,
        ) {
            self.extend_from_slice(row_ids, start, len);
        }
    };
}

macro_rules! impl_row_id_stream_chunk_with_rids {
    (
        $_base:ident,
        $_chunk:ident,
        $chunk_with_rids:ident,
        $_run:ident,
        $_run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $chunk_with_rids(&mut self, _: &$array_ty, row_ids: &UInt64Array) {
            self.extend_from_array(row_ids);
        }
    };
}

macro_rules! impl_row_id_stream_sorted_run_with_rids {
    (
        $_base:ident,
        $_chunk:ident,
        $_chunk_with_rids:ident,
        $_run:ident,
        $run_with_rids:ident,
        $array_ty:ty,
        $_arrow_ty:ty,
        $_dtype:expr,
        $_native_ty:ty,
        $_cast:expr
    ) => {
        fn $run_with_rids(
            &mut self,
            _: &$array_ty,
            row_ids: &UInt64Array,
            start: usize,
            len: usize,
        ) {
            self.extend_from_slice(row_ids, start, len);
        }
    };
}

#[derive(Clone)]
pub(crate) struct ColumnProjectionInfo {
    logical_field_id: LogicalFieldId,
    data_type: DataType,
    output_name: String,
}

#[derive(Clone)]
pub(crate) struct ComputedProjectionInfo {
    expr: ScalarExpr<FieldId>,
    alias: String,
}

#[derive(Clone)]
pub(crate) enum ProjectionEval {
    Column(ColumnProjectionInfo),
    Computed(ComputedProjectionInfo),
}

struct PlannedScan<'expr, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    projections: Vec<ScanProjection>,
    filter_expr: Expr<'expr, FieldId>,
    options: ScanStreamOptions<P>,
    plan_graph: PlanGraph,
}

#[derive(Default, Clone, Copy)]
struct FieldPredicateStats {
    total: usize,
    contains: usize,
}

#[derive(Default)]
struct PredicateFusionCache {
    per_field: FxHashMap<FieldId, FieldPredicateStats>,
}

impl PredicateFusionCache {
    fn from_expr(expr: &Expr<'_, FieldId>) -> Self {
        let mut cache = Self::default();
        cache.record_expr(expr);
        cache
    }

    fn record_expr(&mut self, expr: &Expr<'_, FieldId>) {
        match expr {
            Expr::Pred(filter) => {
                let entry = self.per_field.entry(filter.field_id).or_default();
                entry.total += 1;
                if matches!(filter.op, Operator::Contains { .. }) {
                    entry.contains += 1;
                }
            }
            Expr::And(children) | Expr::Or(children) => {
                for child in children {
                    self.record_expr(child);
                }
            }
            Expr::Not(inner) => self.record_expr(inner),
            Expr::Compare { .. } => {}
            Expr::Literal(_) => {}
        }
    }

    fn should_fuse(&self, field_id: FieldId, dtype: &DataType) -> bool {
        let Some(stats) = self.per_field.get(&field_id) else {
            return false;
        };

        match dtype {
            DataType::Utf8 | DataType::LargeUtf8 => stats.contains >= 1 && stats.total >= 2,
            _ => stats.total >= 2,
        }
    }
}

pub(crate) struct TableExecutor<'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table: &'a Table<P>,
    row_id_cache: RefCell<Option<Vec<RowId>>>,
}

pub(crate) struct TablePlanner<'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table: &'a Table<P>,
}

impl<'a, P> TablePlanner<'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub(crate) fn new(table: &'a Table<P>) -> Self {
        Self { table }
    }

    pub(crate) fn scan_stream_with_exprs<'expr, F>(
        &self,
        projections: &[ScanProjection],
        filter_expr: &Expr<'expr, FieldId>,
        options: ScanStreamOptions<P>,
        on_batch: F,
    ) -> LlkvResult<()>
    where
        F: FnMut(RecordBatch),
    {
        let plan = self.plan_scan(projections, filter_expr, options)?;
        TableExecutor::new(self.table).execute(plan, on_batch)
    }

    // TODO: Make streamable. Don't pre-buffer all row ids in memory at once.
    // NOTE: Planner currently buffers row IDs eagerly; revisit once executor
    // supports incremental pagination to reduce peak memory usage.
    fn plan_scan<'expr>(
        &self,
        projections: &[ScanProjection],
        filter_expr: &Expr<'expr, FieldId>,
        options: ScanStreamOptions<P>,
    ) -> LlkvResult<PlannedScan<'expr, P>> {
        if projections.is_empty() {
            return Err(Error::InvalidArgumentError(
                "scan_stream requires at least one projection".into(),
            ));
        }

        let projections_vec = projections.to_vec();
        let filter_clone = filter_expr.clone();
        let plan_graph = self.build_plan_graph(&projections_vec, &filter_clone, options.clone())?;

        Ok(PlannedScan {
            projections: projections_vec,
            filter_expr: filter_clone,
            options,
            plan_graph,
        })
    }

    fn build_plan_graph(
        &self,
        projections: &[ScanProjection],
        filter_expr: &Expr<'_, FieldId>,
        options: ScanStreamOptions<P>,
    ) -> LlkvResult<PlanGraph> {
        let mut builder = PlanGraphBuilder::new();

        let scan_node_id = PlanNodeId::new(1);
        let mut scan_node = PlanNode::new(scan_node_id, PlanOperator::TableScan);
        scan_node
            .metadata
            .insert("table_id", self.table.table_id().to_string());
        scan_node
            .metadata
            .insert("projection_count", projections.len().to_string());
        builder.add_node(scan_node).map_err(plan_graph_err)?;
        builder.add_root(scan_node_id).map_err(plan_graph_err)?;

        let mut next_node = 2u32;
        let mut parent = scan_node_id;

        if !is_trivial_filter(filter_expr) {
            let filter_node_id = PlanNodeId::new(next_node);
            next_node += 1;
            let mut filter_node = PlanNode::new(filter_node_id, PlanOperator::Filter);
            filter_node.add_predicate(PlanExpression::new(format_expr(filter_expr)));
            builder.add_node(filter_node).map_err(plan_graph_err)?;
            builder
                .add_edge(PlanEdge::new(parent, filter_node_id))
                .map_err(plan_graph_err)?;
            parent = filter_node_id;
        }

        let project_node_id = PlanNodeId::new(next_node);
        next_node += 1;
        let mut project_node = PlanNode::new(project_node_id, PlanOperator::Project);

        for projection in projections {
            match projection {
                ScanProjection::Column(proj) => {
                    let alias = proj
                        .alias
                        .clone()
                        .unwrap_or_else(|| proj.logical_field_id.field_id().to_string());
                    let dtype = self.table.store().data_type(proj.logical_field_id)?;
                    project_node.add_projection(PlanExpression::new(format!("column({})", alias)));
                    project_node.add_field(
                        PlanField::new(alias.clone(), format!("{dtype:?}")).with_nullability(true),
                    );
                }
                ScanProjection::Computed { expr, alias } => {
                    project_node.add_projection(PlanExpression::new(format!(
                        "{} := {}",
                        alias,
                        format_scalar_expr(expr)
                    )));
                    project_node
                        .add_field(PlanField::new(alias.clone(), "Float64").with_nullability(true));
                }
            }
        }

        builder.add_node(project_node).map_err(plan_graph_err)?;
        builder
            .add_edge(PlanEdge::new(parent, project_node_id))
            .map_err(plan_graph_err)?;
        parent = project_node_id;

        let output_node_id = PlanNodeId::new(next_node);
        let mut output_node = PlanNode::new(output_node_id, PlanOperator::Output);
        output_node
            .metadata
            .insert("include_nulls", options.include_nulls.to_string());
        builder.add_node(output_node).map_err(plan_graph_err)?;
        builder
            .add_edge(PlanEdge::new(parent, output_node_id))
            .map_err(plan_graph_err)?;

        let annotations = builder.annotations_mut();
        annotations.description = Some("table.scan_stream".to_string());
        annotations
            .properties
            .insert("table_id".to_string(), self.table.table_id().to_string());

        builder.finish().map_err(plan_graph_err)
    }
}

impl<'a, P> TableExecutor<'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub(crate) fn new(table: &'a Table<P>) -> Self {
        Self {
            table,
            row_id_cache: RefCell::new(None),
        }
    }

    fn table_row_ids(&self) -> LlkvResult<Vec<RowId>> {
        if let Some(cached) = self.row_id_cache.borrow().as_ref() {
            return Ok(cached.clone());
        }
        let computed = self.compute_table_row_ids()?;
        *self.row_id_cache.borrow_mut() = Some(computed.clone());
        Ok(computed)
    }

    fn compute_table_row_ids(&self) -> LlkvResult<Vec<RowId>> {
        use llkv_column_map::store::rowid_fid;

        let fields = self
            .table
            .store()
            .user_field_ids_for_table(self.table.table_id());
        if fields.is_empty() {
            return Ok(Vec::new());
        }

        // Optimization: Try scanning the dedicated row_id shadow column first.
        // This is significantly faster than scanning multiple user columns because:
        // 1. It's a single column scan instead of multiple
        // 2. Row IDs are already in the format we need (no deduplication required)
        // 3. The shadow column is guaranteed to have all row IDs for the table
        //
        // We use the first user field to construct the shadow column ID.
        if let Some(&first_field) = fields.first() {
            let rid_shadow = rowid_fid(first_field);
            let mut collector = RowIdScanCollector::default();

            // Try to scan the row_id shadow column
            match ScanBuilder::new(self.table.store(), rid_shadow)
                .options(ScanOptions {
                    with_row_ids: true,
                    ..Default::default()
                })
                .run(&mut collector)
            {
                Ok(_) => {
                    // Success! We got all row IDs from the shadow column
                    let mut row_ids = collector.into_inner();
                    row_ids.sort_unstable();
                    tracing::trace!(
                        "[PERF] Fast path: collected {} row_ids from shadow column for table {}",
                        row_ids.len(),
                        self.table.table_id()
                    );
                    return Ok(row_ids);
                }
                Err(llkv_result::Error::NotFound) => {
                    // Shadow column doesn't exist, fall back to multi-column scan
                    tracing::trace!(
                        "[PERF] Shadow column not found for table {}, using multi-column scan",
                        self.table.table_id()
                    );
                }
                Err(e) => {
                    // Other error, fall back but log it
                    tracing::debug!(
                        "[PERF] Error scanning shadow column for table {}: {}, falling back to multi-column scan",
                        self.table.table_id(),
                        e
                    );
                }
            }
        }

        // Fallback: Multi-column scan with deduplication
        // This is needed when:
        // - The shadow column doesn't exist (for whatever reason)
        // - Columns may have different row_id sets (sparse columns)
        // - We need the union of all visible rows
        let expected = self
            .table
            .store()
            .total_rows_for_table(self.table.table_id())
            .unwrap_or_default();

        let mut seen: FxHashSet<RowId> = FxHashSet::default();
        let mut collected: Vec<RowId> = Vec::new();

        for lfid in fields {
            let mut collector = RowIdScanCollector::default();
            ScanBuilder::new(self.table.store(), lfid)
                .options(ScanOptions {
                    with_row_ids: true,
                    ..Default::default()
                })
                .run(&mut collector)?;
            for rid in collector.into_inner() {
                if seen.insert(rid) {
                    collected.push(rid);
                }
            }
            if expected > 0 && (seen.len() as u64) >= expected {
                break;
            }
        }

        collected.sort_unstable();
        tracing::trace!(
            "[PERF] Multi-column scan: collected {} row_ids for table {}",
            collected.len(),
            self.table.table_id()
        );
        Ok(collected)
    }

    fn stream_table_row_ids<F>(
        &self,
        chunk_size: usize,
        on_chunk: &mut F,
    ) -> LlkvResult<StreamOutcome>
    where
        F: FnMut(Vec<RowId>) -> LlkvResult<()>,
    {
        use llkv_column_map::store::rowid_fid;

        let fields = self
            .table
            .store()
            .user_field_ids_for_table(self.table.table_id());
        if fields.is_empty() {
            return Ok(StreamOutcome::Handled);
        }

        let Some(&first_field) = fields.first() else {
            return Ok(StreamOutcome::Handled);
        };

        let rid_shadow = rowid_fid(first_field);
        let store = self.table.store();
        let mut emitter = RowIdChunkEmitter::new(chunk_size, on_chunk);
        let scan_result = ScanBuilder::new(store, rid_shadow)
            .options(ScanOptions {
                with_row_ids: true,
                ..Default::default()
            })
            .run(&mut emitter);

        match scan_result {
            Ok(()) => {
                emitter.finish()?;
                Ok(StreamOutcome::Handled)
            }
            Err(Error::NotFound) => Ok(StreamOutcome::Fallback),
            Err(err) => {
                let _ = emitter.finish();
                Err(err)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn try_stream_full_table_scan<F>(
        &self,
        unique_lfids: &[LogicalFieldId],
        projection_evals: &[ProjectionEval],
        passthrough_fields: &[Option<FieldId>],
        unique_index: &FxHashMap<LogicalFieldId, usize>,
        numeric_fields: &FxHashSet<FieldId>,
        requires_numeric: bool,
        null_policy: GatherNullPolicy,
        options: &ScanStreamOptions<P>,
        out_schema: &Arc<Schema>,
        on_batch: &mut F,
    ) -> LlkvResult<StreamOutcome>
    where
        F: FnMut(RecordBatch),
    {
        if options.order.is_some() {
            return Ok(StreamOutcome::Fallback);
        }

        let store = self.table.store();
        let table_id = self.table.table_id();
        let mut gather_ctx = if unique_lfids.is_empty() {
            None
        } else {
            Some(store.prepare_gather_context(unique_lfids)?)
        };

        let unique_lfids_arc = Arc::new(unique_lfids.to_vec());
        let projection_evals_arc = Arc::new(projection_evals.to_vec());
        let passthrough_fields_arc = Arc::new(passthrough_fields.to_vec());
        let unique_index_arc = Arc::new(unique_index.clone());
        let numeric_fields_arc = Arc::new(numeric_fields.clone());

        let mut any_emitted = false;

        let mut process_chunk = |mut chunk: Vec<RowId>| -> LlkvResult<()> {
            if let Some(filter) = options.row_id_filter.as_ref() {
                chunk = filter.filter(self.table, chunk)?;
            }
            if chunk.is_empty() {
                return Ok(());
            }

            let mut builder = RowStreamBuilder::new(
                store,
                table_id,
                Arc::clone(out_schema),
                Arc::clone(&unique_lfids_arc),
                Arc::clone(&projection_evals_arc),
                Arc::clone(&passthrough_fields_arc),
                Arc::clone(&unique_index_arc),
                Arc::clone(&numeric_fields_arc),
                requires_numeric,
                null_policy,
                chunk,
                STREAM_BATCH_ROWS,
            );

            if let Some(ctx) = gather_ctx.take() {
                builder = builder.with_gather_context(ctx);
            }

            let mut stream = builder.build()?;
            let expected_columns = stream.schema().fields().len();

            while let Some(chunk) = stream.next_chunk()? {
                let batch = chunk.to_record_batch();
                debug_assert_eq!(batch.num_columns(), expected_columns);
                if batch.num_rows() > 0 {
                    any_emitted = true;
                    on_batch(batch);
                }
            }

            gather_ctx = stream.into_gather_context();
            Ok(())
        };
        if !self
            .stream_table_row_ids(STREAM_BATCH_ROWS, &mut process_chunk)?
            .is_handled()
        {
            return Ok(StreamOutcome::Fallback);
        }

        if !any_emitted && options.row_id_filter.is_none() {
            let total_rows = self.table.total_rows()?;
            let row_count = usize::try_from(total_rows).map_err(|_| {
                Error::InvalidArgumentError("table row count exceeds supported range".into())
            })?;
            emit_synthetic_null_batch(projection_evals, out_schema, row_count, on_batch)?;
        }

        Ok(StreamOutcome::Handled)
    }

    fn collect_row_ids_for_rowid_filter(&self, op: &Operator<'_>) -> LlkvResult<Vec<RowId>> {
        let all_row_ids = self.table_row_ids()?;
        if all_row_ids.is_empty() {
            return Ok(Vec::new());
        }
        filter_row_ids_by_operator(&all_row_ids, op)
    }

    fn execute<'expr, F>(&self, plan: PlannedScan<'expr, P>, mut on_batch: F) -> LlkvResult<()>
    where
        F: FnMut(RecordBatch),
    {
        let PlannedScan {
            projections,
            filter_expr,
            options,
            plan_graph: _plan_graph,
        } = plan;

        if self
            .try_single_column_direct_scan(
                &projections,
                &filter_expr,
                options.clone(),
                &mut on_batch,
            )?
            .is_handled()
        {
            return Ok(());
        }

        let mut projection_evals = Vec::with_capacity(projections.len());
        let mut unique_index: FxHashMap<LogicalFieldId, usize> = FxHashMap::default();
        let mut unique_lfids: Vec<LogicalFieldId> = Vec::new();
        let mut numeric_fields: FxHashSet<FieldId> = FxHashSet::default();
        let mut lfid_dtypes: FxHashMap<LogicalFieldId, DataType> = FxHashMap::default();

        for proj in &projections {
            match proj {
                ScanProjection::Column(p) => {
                    let lfid = p.logical_field_id;
                    if lfid.table_id() != self.table.table_id() {
                        return Err(Error::InvalidArgumentError(format!(
                            "Projection targets table {} but scan_stream is on table {}",
                            lfid.table_id(),
                            self.table.table_id()
                        )));
                    }
                    if lfid.namespace() != Namespace::UserData {
                        return Err(Error::InvalidArgumentError(format!(
                            "Projection {:?} must target user data namespace",
                            lfid
                        )));
                    }

                    let dtype = self.table.store().data_type(lfid)?;
                    lfid_dtypes.entry(lfid).or_insert_with(|| dtype.clone());
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        unique_index.entry(lfid)
                    {
                        entry.insert(unique_lfids.len());
                        unique_lfids.push(lfid);
                    }
                    let fallback = lfid.field_id().to_string();
                    let output_name = p.alias.clone().unwrap_or(fallback);
                    projection_evals.push(ProjectionEval::Column(ColumnProjectionInfo {
                        logical_field_id: lfid,
                        data_type: dtype,
                        output_name,
                    }));
                }
                ScanProjection::Computed { expr, alias } => {
                    if alias.trim().is_empty() {
                        return Err(Error::InvalidArgumentError(
                            "Computed projection requires a non-empty alias".into(),
                        ));
                    }

                    let simplified = NumericKernels::simplify(expr);
                    let mut fields_set: FxHashSet<FieldId> = FxHashSet::default();
                    NumericKernels::collect_fields(&simplified, &mut fields_set);
                    for field_id in fields_set.iter().copied() {
                        numeric_fields.insert(field_id);
                        let lfid = LogicalFieldId::for_user(self.table.table_id(), field_id);
                        if lfid.namespace() != Namespace::UserData {
                            return Err(Error::InvalidArgumentError(format!(
                                "Computed projection field {:?} must target user data namespace",
                                lfid
                            )));
                        }
                        let dtype = self.table.store().data_type(lfid)?;
                        lfid_dtypes.entry(lfid).or_insert_with(|| dtype.clone());
                        if let std::collections::hash_map::Entry::Vacant(entry) =
                            unique_index.entry(lfid)
                        {
                            entry.insert(unique_lfids.len());
                            unique_lfids.push(lfid);
                        }
                    }

                    projection_evals.push(ProjectionEval::Computed(ComputedProjectionInfo {
                        expr: simplified,
                        alias: alias.clone(),
                    }));
                }
            }
        }

        let passthrough_fields: Vec<Option<FieldId>> = projection_evals
            .iter()
            .map(|eval| match eval {
                ProjectionEval::Computed(info) => NumericKernels::passthrough_column(&info.expr),
                _ => None,
            })
            .collect();

        let null_policy = if options.include_nulls {
            GatherNullPolicy::IncludeNulls
        } else {
            GatherNullPolicy::DropNulls
        };

        let requires_numeric = projection_evals.iter().enumerate().any(|(idx, eval)| {
            matches!(eval, ProjectionEval::Computed(info) if passthrough_fields[idx].is_none()
                    && computed_expr_requires_numeric(&info.expr))
        });

        let mut schema_fields: Vec<Field> = Vec::with_capacity(projection_evals.len());
        for (idx, eval) in projection_evals.iter().enumerate() {
            match eval {
                ProjectionEval::Column(info) => schema_fields.push(Field::new(
                    info.output_name.clone(),
                    info.data_type.clone(),
                    true,
                )),
                ProjectionEval::Computed(info) => {
                    if let Some(fid) = passthrough_fields[idx] {
                        let lfid = LogicalFieldId::for_user(self.table.table_id(), fid);
                        let dtype = lfid_dtypes.get(&lfid).cloned().ok_or_else(|| {
                            Error::Internal("missing dtype for passthrough".into())
                        })?;
                        schema_fields.push(Field::new(info.alias.clone(), dtype, true));
                    } else {
                        let dtype = match &info.expr {
                            ScalarExpr::Literal(Literal::Integer(_)) => DataType::Int64,
                            ScalarExpr::Literal(Literal::Float(_)) => DataType::Float64,
                            ScalarExpr::Literal(Literal::Boolean(_)) => DataType::Boolean,
                            ScalarExpr::Literal(Literal::String(_)) => DataType::Utf8,
                            ScalarExpr::Literal(Literal::Null) => DataType::Null,
                            ScalarExpr::Literal(Literal::Struct(fields)) => {
                                // Infer struct type from the literal fields
                                let struct_fields = fields
                                    .iter()
                                    .map(|(name, lit)| {
                                        let field_dtype = infer_literal_datatype(lit.as_ref())?;
                                        Ok(Field::new(name.clone(), field_dtype, true))
                                    })
                                    .collect::<LlkvResult<Vec<_>>>()?;
                                DataType::Struct(struct_fields.into())
                            }
                            ScalarExpr::Binary { .. } => DataType::Float64,
                            ScalarExpr::Column(fid) => {
                                let lfid = LogicalFieldId::for_user(self.table.table_id(), *fid);
                                lfid_dtypes.get(&lfid).cloned().ok_or_else(|| {
                                    Error::Internal("missing dtype for computed column".into())
                                })?
                            }
                            ScalarExpr::Aggregate(_) => {
                                // Aggregates in computed columns return Int64.
                                // TODO: Fix: This is a simplification - ideally we'd determine type from the aggregate
                                // NOTE: This assumes SUM-like semantics; extend once planner
                                // carries precise aggregate signatures.
                                DataType::Int64
                            }
                            ScalarExpr::GetField { base, field_name } => {
                                // Determine the data type of the field being extracted
                                fn get_field_dtype(
                                    expr: &ScalarExpr<FieldId>,
                                    field_name: &str,
                                    table_id: TableId,
                                    lfid_dtypes: &FxHashMap<LogicalFieldId, DataType>,
                                ) -> LlkvResult<DataType> {
                                    let base_dtype = match expr {
                                        ScalarExpr::Column(fid) => {
                                            let lfid = LogicalFieldId::for_user(table_id, *fid);
                                            lfid_dtypes.get(&lfid).cloned().ok_or_else(|| {
                                                Error::Internal("missing dtype for column".into())
                                            })?
                                        }
                                        ScalarExpr::GetField {
                                            base: inner_base,
                                            field_name: inner_field,
                                        } => get_field_dtype(
                                            inner_base,
                                            inner_field,
                                            table_id,
                                            lfid_dtypes,
                                        )?,
                                        _ => return Err(Error::InvalidArgumentError(
                                            "GetField base must be a column or another GetField"
                                                .into(),
                                        )),
                                    };

                                    if let DataType::Struct(fields) = base_dtype {
                                        fields
                                            .iter()
                                            .find(|f| f.name() == field_name)
                                            .map(|f| f.data_type().clone())
                                            .ok_or_else(|| {
                                                Error::InvalidArgumentError(format!(
                                                    "Field '{}' not found in struct",
                                                    field_name
                                                ))
                                            })
                                    } else {
                                        Err(Error::InvalidArgumentError(
                                            "GetField can only be applied to struct types".into(),
                                        ))
                                    }
                                }

                                get_field_dtype(
                                    base,
                                    field_name,
                                    self.table.table_id(),
                                    &lfid_dtypes,
                                )?
                            }
                        };
                        schema_fields.push(Field::new(info.alias.clone(), dtype, true));
                    }
                }
            }
        }
        let out_schema = Arc::new(Schema::new(schema_fields));

        if options.order.is_none()
            && is_trivial_filter(&filter_expr)
            && self
                .try_stream_full_table_scan(
                    unique_lfids.as_slice(),
                    &projection_evals,
                    &passthrough_fields,
                    &unique_index,
                    &numeric_fields,
                    requires_numeric,
                    null_policy,
                    &options,
                    &out_schema,
                    &mut on_batch,
                )?
                .is_handled()
        {
            return Ok(());
        }

        let fusion_cache = PredicateFusionCache::from_expr(&filter_expr);
        let mut all_rows_cache: FxHashMap<FieldId, Vec<RowId>> = FxHashMap::default();

        tracing::debug!(
            expr = %format_expr(&filter_expr),
            "collect_row_ids_for_expr root"
        );

        // When we have a trivial filter (no predicates), enumerate ALL row IDs.
        // This is necessary for:
        // 1. MVCC filtering to check visibility of all rows including NULL rows
        // 2. Aggregates like COUNT_NULLS that need to count NULL rows
        // We scan the MVCC created_by column which exists for every row.
        let mut row_ids = if is_trivial_filter(&filter_expr) {
            use arrow::datatypes::UInt64Type;
            use llkv_expr::typed_predicate::Predicate;
            let created_lfid = LogicalFieldId::for_mvcc_created_by(self.table.table_id());
            tracing::trace!(
                "[SCAN_STREAM] MVCC + trivial filter: scanning created_by column for all row IDs"
            );
            // Get all rows where created_by exists (which is all rows that have been written)
            self.table
                .store()
                .filter_row_ids::<UInt64Type>(created_lfid, &Predicate::All)?
        } else {
            self.collect_row_ids_for_expr(&filter_expr, &fusion_cache, &mut all_rows_cache)?
        };

        tracing::trace!(
            "[SCAN_STREAM] collected {} row_ids, has_row_filter={}",
            row_ids.len(),
            options.row_id_filter.is_some()
        );
        if let Some(filter) = options.row_id_filter.as_ref() {
            let before_len = row_ids.len();
            row_ids = filter.filter(self.table, row_ids)?;
            tracing::trace!(
                "[SCAN_STREAM] after MVCC filter: {} -> {} row_ids",
                before_len,
                row_ids.len()
            );
        }
        if row_ids.is_empty() {
            tracing::trace!(
                "[SCAN_STREAM] row_ids is empty after filtering, returning early (no synthetic batch)"
            );
            // MVCC Note: If row_ids is empty after MVCC filtering, don't create synthetic batches!
            // The old optimization of creating NULL rows for trivial filters breaks MVCC visibility.
            // An empty row_ids list means NO visible rows, so we should return empty results.
            if options.row_id_filter.is_none() && is_trivial_filter(&filter_expr) {
                let total_rows = self.table.total_rows()?;
                let row_count = usize::try_from(total_rows).map_err(|_| {
                    Error::InvalidArgumentError("table row count exceeds supported range".into())
                })?;
                emit_synthetic_null_batch(
                    &projection_evals,
                    &out_schema,
                    row_count,
                    &mut on_batch,
                )?;
            }
            return Ok(());
        }

        if let Some(order_spec) = options.order {
            row_ids = self.sort_row_ids_with_order(row_ids, order_spec)?;
        }

        let table_id = self.table.table_id();
        let store = self.table.store();
        let unique_lfids = Arc::new(unique_lfids);
        let projection_evals = Arc::new(projection_evals);
        let passthrough_fields = Arc::new(passthrough_fields);
        let unique_index = Arc::new(unique_index);
        let numeric_fields = Arc::new(numeric_fields);

        let mut row_stream = RowStreamBuilder::new(
            store,
            table_id,
            Arc::clone(&out_schema),
            Arc::clone(&unique_lfids),
            Arc::clone(&projection_evals),
            Arc::clone(&passthrough_fields),
            Arc::clone(&unique_index),
            Arc::clone(&numeric_fields),
            requires_numeric,
            null_policy,
            row_ids,
            STREAM_BATCH_ROWS,
        )
        .build()?;

        let expected_columns = row_stream.schema().fields().len();

        while let Some(chunk) = row_stream.next_chunk()? {
            let batch = chunk.to_record_batch();
            debug_assert_eq!(batch.num_columns(), expected_columns);
            tracing::debug!(
                rows = batch.num_rows(),
                columns = batch.num_columns(),
                "TableExecutor produced batch"
            );
            on_batch(batch);
        }

        Ok(())
    }

    fn try_single_column_direct_scan<'expr, F>(
        &self,
        projections: &[ScanProjection],
        filter_expr: &Expr<'expr, FieldId>,
        options: ScanStreamOptions<P>,
        on_batch: &mut F,
    ) -> LlkvResult<StreamOutcome>
    where
        F: FnMut(RecordBatch),
    {
        if options.include_nulls || projections.len() != 1 {
            return Ok(StreamOutcome::Fallback);
        }

        match &projections[0] {
            ScanProjection::Column(p) => {
                if p.logical_field_id.table_id() != self.table.table_id() {
                    return Err(Error::InvalidArgumentError(format!(
                        "Projection targets table {} but scan_stream is on table {}",
                        p.logical_field_id.table_id(),
                        self.table.table_id()
                    )));
                }

                if !is_full_range_filter(filter_expr, p.logical_field_id.field_id()) {
                    return Ok(StreamOutcome::Fallback);
                }

                let dtype = self.table.store().data_type(p.logical_field_id)?;
                if matches!(dtype, DataType::Utf8 | DataType::LargeUtf8) {
                    return Ok(StreamOutcome::Fallback);
                }
                let field_name = p
                    .alias
                    .clone()
                    .unwrap_or_else(|| p.logical_field_id.field_id().to_string());
                let schema = Arc::new(Schema::new(vec![Field::new(
                    field_name,
                    dtype.clone(),
                    true,
                )]));

                let mut visitor = SingleColumnStreamVisitor::new(schema, on_batch);
                ScanBuilder::new(self.table.store(), p.logical_field_id)
                    .options(ScanOptions::default())
                    .run(&mut visitor)?;
                visitor.finish()?;
                Ok(StreamOutcome::Handled)
            }
            ScanProjection::Computed { expr, alias } => {
                let simplified = NumericKernels::simplify(expr);
                let mut fields_set: FxHashSet<FieldId> = FxHashSet::default();
                NumericKernels::collect_fields(&simplified, &mut fields_set);
                if fields_set.len() != 1 {
                    return Ok(StreamOutcome::Fallback);
                }

                let field_id = *fields_set.iter().next().unwrap();
                let lfid = LogicalFieldId::for_user(self.table.table_id(), field_id);
                if !is_full_range_filter(filter_expr, field_id) {
                    return Ok(StreamOutcome::Fallback);
                }

                if lfid.namespace() != Namespace::UserData {
                    return Err(Error::InvalidArgumentError(format!(
                        "Computed projection field {:?} must target user data namespace",
                        lfid
                    )));
                }

                let field_alias = alias.clone();
                let dtype = self.table.store().data_type(lfid)?;

                if let Some(passthrough_fid) = NumericKernels::passthrough_column(&simplified) {
                    if passthrough_fid != field_id {
                        return Err(Error::InvalidArgumentError(
                            "computed projection passthrough resolved to unexpected field".into(),
                        ));
                    }
                    let schema = Arc::new(Schema::new(vec![Field::new(
                        field_alias.clone(),
                        dtype.clone(),
                        true,
                    )]));
                    let mut visitor = SingleColumnStreamVisitor::new(schema, on_batch);
                    ScanBuilder::new(self.table.store(), lfid)
                        .options(ScanOptions::default())
                        .run(&mut visitor)?;
                    visitor.finish()?;
                    return Ok(StreamOutcome::Handled);
                }

                if let Some(affine) = NumericKernels::extract_affine(&simplified) {
                    if affine.field != field_id {
                        return Err(Error::InvalidArgumentError(
                            "affine extraction resolved to unexpected field".into(),
                        ));
                    }
                    if !is_supported_numeric(&dtype) {
                        return Ok(StreamOutcome::Fallback);
                    }
                    let schema = Arc::new(Schema::new(vec![Field::new(
                        field_alias.clone(),
                        DataType::Float64,
                        true,
                    )]));
                    let mut visitor = AffineSingleColumnVisitor::new(
                        schema,
                        affine.scale,
                        affine.offset,
                        on_batch,
                    );
                    ScanBuilder::new(self.table.store(), lfid)
                        .options(ScanOptions::default())
                        .run(&mut visitor)?;
                    visitor.finish()?;
                    return Ok(StreamOutcome::Handled);
                }

                let mut numeric_fields: FxHashSet<FieldId> = FxHashSet::default();
                numeric_fields.insert(field_id);
                let schema = Arc::new(Schema::new(vec![Field::new(
                    field_alias.clone(),
                    DataType::Float64,
                    true,
                )]));
                let mut visitor = ComputedSingleColumnVisitor::new(
                    schema,
                    simplified,
                    lfid,
                    numeric_fields,
                    on_batch,
                );
                ScanBuilder::new(self.table.store(), lfid)
                    .options(ScanOptions::default())
                    .run(&mut visitor)?;
                visitor.finish()?;
                Ok(StreamOutcome::Handled)
            }
        }
    }

    fn try_collect_and_fused(
        &self,
        children: &[Expr<'_, FieldId>],
        fusion_cache: &PredicateFusionCache,
    ) -> LlkvResult<Option<Vec<RowId>>> {
        let mut same_field: Option<FieldId> = None;
        for child in children {
            match child {
                Expr::Pred(filter) => match same_field {
                    Some(existing) if existing != filter.field_id => return Ok(None),
                    None => same_field = Some(filter.field_id),
                    _ => {}
                },
                _ => return Ok(None),
            }
        }

        let Some(fid) = same_field else {
            return Ok(None);
        };

        let filter_lfid = LogicalFieldId::for_user(self.table.table_id(), fid);
        let dtype = self.table.store().data_type(filter_lfid)?;

        let ops: Vec<Operator<'_>> = children
            .iter()
            .map(|child| match child {
                Expr::Pred(filter) => filter.op.clone(),
                _ => unreachable!("non-predicate child filtered earlier"),
            })
            .collect();

        if !ops
            .iter()
            .all(|op| !matches!(op, Operator::IsNull | Operator::IsNotNull))
        {
            return Ok(None);
        }

        if !fusion_cache.should_fuse(fid, &dtype) {
            return Ok(None);
        }

        let row_ids = match &dtype {
            DataType::Utf8 => self.collect_matching_row_ids_string_fused::<i32>(filter_lfid, &ops),
            DataType::LargeUtf8 => {
                self.collect_matching_row_ids_string_fused::<i64>(filter_lfid, &ops)
            }
            DataType::Boolean => self.collect_matching_row_ids_bool_fused(filter_lfid, &ops),
            other => llkv_column_map::with_integer_arrow_type!(
                other.clone(),
                |ArrowTy| self.collect_matching_row_ids_fused::<ArrowTy>(filter_lfid, &ops),
                Err(Error::Internal(format!(
                    "Filtering on type {:?} is not supported",
                    other
                ))),
            ),
        }?;

        Ok(Some(normalize_row_ids(row_ids)))
    }

    fn collect_row_ids_for_expr(
        &self,
        expr: &Expr<'_, FieldId>,
        fusion_cache: &PredicateFusionCache,
        all_rows_cache: &mut FxHashMap<FieldId, Vec<RowId>>,
    ) -> LlkvResult<Vec<RowId>> {
        enum EvalFrame<'a> {
            And {
                children: &'a [Expr<'a, FieldId>],
                next_child: usize,
                acc: Option<Vec<RowId>>,
            },
            Or {
                children: &'a [Expr<'a, FieldId>],
                next_child: usize,
                acc: Vec<RowId>,
            },
            Not { domain: Vec<RowId> },
        }

        let mut frames: Vec<EvalFrame<'_>> = Vec::new();
        let mut current: Option<&Expr<'_, FieldId>> = Some(expr);
        let mut last_result: Vec<RowId> = Vec::new();

        'outer: loop {
            if let Some(node) = current.take() {
                match node {
                    Expr::Pred(filter) => {
                        last_result = self.collect_row_ids_for_filter(filter)?;
                    }
                    Expr::Compare { left, op, right } => {
                        last_result =
                            self.collect_row_ids_for_compare(left, *op, right, all_rows_cache)?;
                    }
                    Expr::And(children) => {
                        if children.is_empty() {
                            return Err(Error::InvalidArgumentError(
                                "AND expression requires at least one predicate".into(),
                            ));
                        }

                        if let Some(fused) =
                            self.try_collect_and_fused(children, fusion_cache)?
                        {
                            last_result = fused;
                        } else {
                            frames.push(EvalFrame::And {
                                children,
                                next_child: 0,
                                acc: None,
                            });
                            current = Some(&children[0]);
                            continue 'outer;
                        }
                    }
                    Expr::Or(children) => {
                        if children.is_empty() {
                            return Err(Error::InvalidArgumentError(
                                "OR expression requires at least one predicate".into(),
                            ));
                        }
                        frames.push(EvalFrame::Or {
                            children,
                            next_child: 0,
                            acc: Vec::new(),
                        });
                        current = Some(&children[0]);
                        continue 'outer;
                    }
                    Expr::Not(inner) => {
                        let domain = self.collect_row_ids_domain(inner, all_rows_cache)?;
                        if domain.is_empty() {
                            last_result = Vec::new();
                        } else {
                            frames.push(EvalFrame::Not { domain });
                            current = Some(inner);
                            continue 'outer;
                        }
                    }
                    Expr::Literal(value) => {
                        if *value {
                            last_result = self.collect_all_row_ids(all_rows_cache)?;
                        } else {
                            last_result.clear();
                        }
                    }
                }
            }

            loop {
                match frames.last_mut() {
                    Some(EvalFrame::And {
                        children,
                        next_child,
                        acc,
                    }) => {
                        let child_result = mem::take(&mut last_result);
                        if *next_child == 0 {
                            *acc = Some(child_result);
                        } else {
                            let previous = acc.take().expect("AND accumulator must be set");
                            if previous.is_empty() || child_result.is_empty() {
                                *acc = Some(Vec::new());
                            } else {
                                *acc = Some(intersect_sorted(previous, child_result));
                            }
                        }

                        *next_child += 1;

                        let acc_ref = acc.as_ref().expect("AND accumulator initialized");
                        if acc_ref.is_empty() {
                            last_result.clear();
                            frames.pop();
                            continue;
                        }

                        if *next_child < children.len() {
                            current = Some(&children[*next_child]);
                            continue 'outer;
                        } else {
                            last_result = acc.take().unwrap();
                            frames.pop();
                            continue;
                        }
                    }
                    Some(EvalFrame::Or {
                        children,
                        next_child,
                        acc,
                    }) => {
                        let child_result = mem::take(&mut last_result);
                        if *next_child == 0 {
                            *acc = child_result;
                        } else if !child_result.is_empty() {
                            if acc.is_empty() {
                                *acc = child_result;
                            } else {
                                *acc = union_sorted(mem::take(acc), child_result);
                            }
                        }

                        *next_child += 1;

                        if *next_child < children.len() {
                            current = Some(&children[*next_child]);
                            continue 'outer;
                        } else {
                            last_result = mem::take(acc);
                            frames.pop();
                            continue;
                        }
                    }
                    Some(EvalFrame::Not { domain }) => {
                        let matched = mem::take(&mut last_result);
                        let domain_vec = mem::take(domain);
                        frames.pop();
                        last_result = difference_sorted(domain_vec, matched);
                        continue;
                    }
                    None => return Ok(last_result),
                }
            }
        }
    }

    fn collect_row_ids_for_filter(&self, filter: &Filter<'_, FieldId>) -> LlkvResult<Vec<RowId>> {
        if filter.field_id == ROW_ID_FIELD_ID {
            let row_ids = self.collect_row_ids_for_rowid_filter(&filter.op)?;
            tracing::debug!(
                field = "rowid",
                row_count = row_ids.len(),
                "collect_row_ids_for_filter rowid"
            );
            return Ok(row_ids);
        }

        let filter_lfid = LogicalFieldId::for_user(self.table.table_id(), filter.field_id);
        let dtype = self.table.store().data_type(filter_lfid)?;

        match &filter.op {
            Operator::IsNotNull => {
                let mut cache = FxHashMap::default();
                let non_null = self.collect_all_row_ids_for_field(filter.field_id, &mut cache)?;
                tracing::debug!(
                    field = ?filter_lfid,
                    row_count = non_null.len(),
                    "collect_row_ids_for_filter NOT NULL fast path"
                );
                return Ok(non_null);
            }
            Operator::IsNull => {
                let all_row_ids = self.table_row_ids()?;
                if all_row_ids.is_empty() {
                    return Ok(Vec::new());
                }
                let mut cache = FxHashMap::default();
                let non_null = self.collect_all_row_ids_for_field(filter.field_id, &mut cache)?;
                let null_ids = difference_sorted(all_row_ids, non_null);
                tracing::debug!(
                    field = ?filter_lfid,
                    row_count = null_ids.len(),
                    "collect_row_ids_for_filter NULL fast path"
                );
                return Ok(null_ids);
            }
            _ => {}
        }

        if let Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        } = &filter.op
            && let Some(runs) = dense_row_runs(self.table.store(), filter_lfid)?
        {
            let rows = expand_filter_runs(&runs);
            tracing::debug!(
                field = ?filter_lfid,
                row_count = rows.len(),
                "collect_row_ids_for_filter using dense runs"
            );
            return Ok(rows);
        }

        let row_ids = match &dtype {
            DataType::Utf8 => self.collect_matching_row_ids_string::<i32>(filter_lfid, &filter.op),
            DataType::LargeUtf8 => {
                self.collect_matching_row_ids_string::<i64>(filter_lfid, &filter.op)
            }
            DataType::Boolean => self.collect_matching_row_ids_bool(filter_lfid, &filter.op),
            other => llkv_column_map::with_integer_arrow_type!(
                other.clone(),
                |ArrowTy| self.collect_matching_row_ids::<ArrowTy>(filter_lfid, &filter.op),
                Err(Error::Internal(format!(
                    "Filtering on type {:?} is not supported",
                    other
                ))),
            ),
        }?;
        tracing::debug!(
            field = ?filter_lfid,
            row_count = row_ids.len(),
            "collect_row_ids_for_filter general path"
        );

        Ok(normalize_row_ids(row_ids))
    }

    /// Evaluate a constant-only comparison expression.
    ///
    /// Used for comparisons with no column references, such as those produced
    /// by materializing IN (SELECT ...) subqueries.
    fn evaluate_constant_compare(
        left: &ScalarExpr<FieldId>,
        op: CompareOp,
        right: &ScalarExpr<FieldId>,
    ) -> LlkvResult<bool> {
        use llkv_expr::literal::Literal;
        
        // Extract literal values
        let left_lit = match left {
            ScalarExpr::Literal(lit) => lit,
            _ => return Err(Error::InvalidArgumentError(
                "constant comparison requires literal expressions".into(),
            )),
        };
        let right_lit = match right {
            ScalarExpr::Literal(lit) => lit,
            _ => return Err(Error::InvalidArgumentError(
                "constant comparison requires literal expressions".into(),
            )),
        };

        // Helper to compare literals
        fn compare_literals(left: &Literal, right: &Literal) -> Option<std::cmp::Ordering> {
            match (left, right) {
                (Literal::Null, _) | (_, Literal::Null) => None,
                (Literal::Boolean(l), Literal::Boolean(r)) => l.partial_cmp(r),
                (Literal::Integer(l), Literal::Integer(r)) => l.partial_cmp(r),
                (Literal::Float(l), Literal::Float(r)) => l.partial_cmp(r),
                (Literal::String(l), Literal::String(r)) => l.partial_cmp(r),
                (Literal::Integer(l), Literal::Float(r)) => (*l as f64).partial_cmp(r),
                (Literal::Float(l), Literal::Integer(r)) => l.partial_cmp(&(*r as f64)),
                _ => None,
            }
        }

        // Evaluate based on operation
        let result = match op {
            CompareOp::Eq => left_lit == right_lit,
            CompareOp::NotEq => left_lit != right_lit,
            CompareOp::Lt => {
                compare_literals(left_lit, right_lit) == Some(std::cmp::Ordering::Less)
            }
            CompareOp::LtEq => matches!(
                compare_literals(left_lit, right_lit),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            ),
            CompareOp::Gt => {
                compare_literals(left_lit, right_lit) == Some(std::cmp::Ordering::Greater)
            }
            CompareOp::GtEq => matches!(
                compare_literals(left_lit, right_lit),
                Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
            ),
        };

        Ok(result)
    }

    /// Collect all row IDs from the table.
    ///
    /// Used when a constant-only comparison evaluates to true and we need
    /// to return all rows.
    fn collect_all_row_ids(
        &self,
        all_rows_cache: &mut FxHashMap<FieldId, Vec<RowId>>,
    ) -> LlkvResult<Vec<RowId>> {
        // Use ROW_ID_FIELD_ID to get all row IDs from the table
        self.collect_all_row_ids_for_field(ROW_ID_FIELD_ID, all_rows_cache)
    }

    fn collect_row_ids_for_compare(
        &self,
        left: &ScalarExpr<FieldId>,
        op: CompareOp,
        right: &ScalarExpr<FieldId>,
        all_rows_cache: &mut FxHashMap<FieldId, Vec<RowId>>,
    ) -> LlkvResult<Vec<RowId>> {
        let mut fields = FxHashSet::default();
        NumericKernels::collect_fields(left, &mut fields);
        NumericKernels::collect_fields(right, &mut fields);
        
        // Handle constant-only comparisons (e.g., from materialized IN subqueries)
        // These are comparisons like "5 IN (1,2,3)" with no column references
        if fields.is_empty() {
            // Evaluate the constant comparison
            let result = Self::evaluate_constant_compare(left, op, right)?;
            
            // If true, return all rows; if false, return empty
            return if result {
                self.collect_all_row_ids(all_rows_cache)
            } else {
                Ok(Vec::new())
            };
        }
        
        let mut domain: Option<Vec<RowId>> = None;
        let mut ordered_fields: Vec<FieldId> = fields.into_iter().collect();
        ordered_fields.sort_unstable();
        for fid in &ordered_fields {
            let rows = self.collect_all_row_ids_for_field(*fid, all_rows_cache)?;
            domain = Some(match domain {
                Some(existing) => intersect_sorted(existing, rows),
                None => rows,
            });
            if let Some(ref d) = domain
                && d.is_empty()
            {
                return Ok(Vec::new());
            }
        }
        if let Some(ref domain_rows) = domain {
            tracing::debug!(
                ?ordered_fields,
                domain_len = domain_rows.len(),
                "collect_row_ids_for_compare domain"
            );
        } else {
            tracing::debug!(
                ?ordered_fields,
                "collect_row_ids_for_compare domain empty"
            );
        }
        let domain = domain.unwrap_or_default();
        if domain.is_empty() {
            return Ok(domain);
        }
        let result = self.evaluate_compare_over_rows(&domain, &ordered_fields, left, op, right)?;
        tracing::debug!(
            ?ordered_fields,
            result_len = result.len(),
            "collect_row_ids_for_compare result"
        );
        Ok(result)
    }

    fn evaluate_compare_over_rows(
        &self,
        row_ids: &[RowId],
        fields: &[FieldId],
        left: &ScalarExpr<FieldId>,
        op: CompareOp,
        right: &ScalarExpr<FieldId>,
    ) -> LlkvResult<Vec<RowId>> {
        if row_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut numeric_fields: FxHashSet<FieldId> = fields.iter().copied().collect();
        let has_row_id = numeric_fields.remove(&ROW_ID_FIELD_ID);

        let table_id = self.table.table_id();
        let store = self.table.store();

        let physical_fields: Vec<FieldId> = fields
            .iter()
            .copied()
            .filter(|fid| *fid != ROW_ID_FIELD_ID)
            .collect();

        let schema = self.table.schema()?;
        let cached_schema = CachedSchema::new(Arc::clone(&schema));

        let mut projection_evals: Vec<ProjectionEval> = Vec::with_capacity(physical_fields.len());
        let mut output_fields: Vec<Field> = Vec::with_capacity(physical_fields.len());
        let mut unique_index: FxHashMap<LogicalFieldId, usize> = FxHashMap::default();

        for (idx, field_id) in physical_fields.iter().copied().enumerate() {
            let schema_idx = cached_schema.index_of_field_id(field_id).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "field_id {} missing from table schema",
                    field_id
                ))
            })?;
            let field = schema.field(schema_idx).clone();
            let lfid = LogicalFieldId::for_user(table_id, field_id);

            projection_evals.push(ProjectionEval::Column(ColumnProjectionInfo {
                logical_field_id: lfid,
                data_type: field.data_type().clone(),
                output_name: field.name().to_string(),
            }));
            output_fields.push(field);
            unique_index.insert(lfid, idx);
        }

        let logical_fields: Vec<LogicalFieldId> = physical_fields
            .iter()
            .map(|&fid| LogicalFieldId::for_user(table_id, fid))
            .collect();
        let logical_fields_for_arrays = logical_fields.clone();

        let requires_numeric = !numeric_fields.is_empty();
        let numeric_fields_arc = Arc::new(numeric_fields);

        let out_schema = Arc::new(Schema::new(output_fields));
        let unique_lfids_arc = Arc::new(logical_fields.clone());
        let projection_evals_arc = Arc::new(projection_evals);
        let passthrough_fields_arc = Arc::new(vec![None; projection_evals_arc.len()]);
        let unique_index_arc = Arc::new(unique_index);

        let mut result: Vec<RowId> = Vec::with_capacity(row_ids.len());

        let mut process_chunk = |window: &[RowId], columns: &[ArrayRef]| -> LlkvResult<()> {
            if window.is_empty() {
                return Ok(());
            }

            let mut numeric_arrays: NumericArrayMap = if columns.is_empty() {
                NumericKernels::prepare_numeric_arrays(&[], &[], numeric_fields_arc.as_ref())?
            } else {
                NumericKernels::prepare_numeric_arrays(
                    &logical_fields_for_arrays,
                    columns,
                    numeric_fields_arc.as_ref(),
                )?
            };

            if has_row_id {
                let rid_values: Vec<f64> = window.iter().map(|rid| *rid as f64).collect();
                numeric_arrays.insert(ROW_ID_FIELD_ID, Arc::new(Float64Array::from(rid_values)));
            }

            for (offset, &row_id) in window.iter().enumerate() {
                let left_val = NumericKernels::evaluate_value(left, offset, &numeric_arrays)?;
                let right_val = NumericKernels::evaluate_value(right, offset, &numeric_arrays)?;
                if let (Some(lv), Some(rv)) = (left_val, right_val)
                    && NumericKernels::compare(op, lv, rv)
                {
                    result.push(row_id);
                }
            }

            Ok(())
        };

        if logical_fields_for_arrays.is_empty() {
            let mut start = 0usize;
            while start < row_ids.len() {
                let end = cmp::min(start + STREAM_BATCH_ROWS, row_ids.len());
                process_chunk(&row_ids[start..end], &[])?;
                start = end;
            }
            return Ok(result);
        }

        let mut row_stream = RowStreamBuilder::new(
            store,
            table_id,
            Arc::clone(&out_schema),
            Arc::clone(&unique_lfids_arc),
            Arc::clone(&projection_evals_arc),
            Arc::clone(&passthrough_fields_arc),
            Arc::clone(&unique_index_arc),
            Arc::clone(&numeric_fields_arc),
            requires_numeric,
            GatherNullPolicy::IncludeNulls,
            row_ids.to_vec(),
            STREAM_BATCH_ROWS,
        )
        .build()?;

        while let Some(chunk) = row_stream.next_chunk()? {
            let window = chunk.row_ids.values();
            let batch = chunk.to_record_batch();
            process_chunk(window, batch.columns())?;
        }

        Ok(result)
    }

    fn collect_row_ids_domain(
        &self,
        expr: &Expr<'_, FieldId>,
        all_rows_cache: &mut FxHashMap<FieldId, Vec<RowId>>,
    ) -> LlkvResult<Vec<RowId>> {
        let mut stack: Vec<&Expr<'_, FieldId>> = vec![expr];
        let mut domain: Option<Vec<RowId>> = None;

        while let Some(node) = stack.pop() {
            match node {
                Expr::Pred(filter) => {
                    let rows =
                        self.collect_all_row_ids_for_field(filter.field_id, all_rows_cache)?;
                    if !rows.is_empty() {
                        let previous = domain.take();
                        domain = Some(match previous {
                            Some(existing) if existing.is_empty() => rows,
                            Some(existing) => union_sorted(existing, rows),
                            None => rows,
                        });
                    }
                }
                Expr::Compare { left, right, op } => {
                    let mut fields = FxHashSet::default();
                    NumericKernels::collect_fields(left, &mut fields);
                    NumericKernels::collect_fields(right, &mut fields);

                    if fields.is_empty() {
                        if Self::evaluate_constant_compare(left, *op, right)? {
                            let rows = self.collect_all_row_ids(all_rows_cache)?;
                            if !rows.is_empty() {
                                let previous = domain.take();
                                domain = Some(match previous {
                                    Some(existing) if existing.is_empty() => rows,
                                    Some(existing) => union_sorted(existing, rows),
                                    None => rows,
                                });
                            }
                        }
                        continue;
                    }

                    let mut compare_domain: Option<Vec<RowId>> = None;
                    let mut ordered_fields: Vec<FieldId> = fields.into_iter().collect();
                    ordered_fields.sort_unstable();
                    for fid in ordered_fields.iter().copied() {
                        let rows = self.collect_all_row_ids_for_field(fid, all_rows_cache)?;
                        compare_domain = Some(match compare_domain {
                            Some(existing) => intersect_sorted(existing, rows),
                            None => rows,
                        });
                        if let Some(ref d) = compare_domain
                            && d.is_empty()
                        {
                            break;
                        }
                    }

                    if let Some(ref domain_rows) = compare_domain {
                        tracing::debug!(
                            domain_len = domain_rows.len(),
                            "collect_row_ids_domain compare"
                        );
                    } else {
                        tracing::debug!("collect_row_ids_domain compare empty");
                    }

                    if let Some(rows) = compare_domain {
                        if !rows.is_empty() {
                            let previous = domain.take();
                            domain = Some(match previous {
                                Some(existing) if existing.is_empty() => rows,
                                Some(existing) => union_sorted(existing, rows),
                                None => rows,
                            });
                        }
                    }
                }
                Expr::And(children) | Expr::Or(children) => {
                    for child in children {
                        stack.push(child);
                    }
                }
                Expr::Not(inner) => stack.push(inner),
                Expr::Literal(value) => {
                    if *value {
                        let rows = self.collect_all_row_ids(all_rows_cache)?;
                        if !rows.is_empty() {
                            let previous = domain.take();
                            domain = Some(match previous {
                                Some(existing) if existing.is_empty() => rows,
                                Some(existing) => union_sorted(existing, rows),
                                None => rows,
                            });
                        }
                    }
                }
            }
        }

        Ok(domain.unwrap_or_default())
    }

    fn sort_row_ids_with_order(
        &self,
        mut row_ids: Vec<RowId>,
        order: ScanOrderSpec,
    ) -> LlkvResult<Vec<RowId>> {
        if row_ids.len() <= 1 {
            return Ok(row_ids);
        }

        let lfid = LogicalFieldId::for_user(self.table.table_id(), order.field_id);
        let store = self.table.store();
        let ascending = matches!(order.direction, ScanOrderDirection::Ascending);
        let schema = self.table.schema()?;
        let cached_schema = CachedSchema::new(Arc::clone(&schema));
        let schema_index = cached_schema
            .index_of_field_id(order.field_id)
            .ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "ORDER BY field {} missing from table schema",
                    order.field_id
                ))
            })?;
        let field = schema.field(schema_index).clone();

        let out_schema = Arc::new(Schema::new(vec![field.clone()]));
        let projection_evals_arc = Arc::new(vec![ProjectionEval::Column(ColumnProjectionInfo {
            logical_field_id: lfid,
            data_type: field.data_type().clone(),
            output_name: field.name().to_string(),
        })]);
        let passthrough_fields_arc = Arc::new(vec![None]);
        let mut unique_index_map = FxHashMap::default();
        unique_index_map.insert(lfid, 0usize);
        let unique_index_arc = Arc::new(unique_index_map);
        let logical_fields_arc = Arc::new(vec![lfid]);
        let numeric_fields_arc = Arc::new(FxHashSet::default());

        match order.transform {
            ScanOrderTransform::IdentityInteger => {
                let mut row_stream = RowStreamBuilder::new(
                    store,
                    self.table.table_id(),
                    Arc::clone(&out_schema),
                    Arc::clone(&logical_fields_arc),
                    Arc::clone(&projection_evals_arc),
                    Arc::clone(&passthrough_fields_arc),
                    Arc::clone(&unique_index_arc),
                    Arc::clone(&numeric_fields_arc),
                    false,
                    GatherNullPolicy::IncludeNulls,
                    row_ids.clone(),
                    STREAM_BATCH_ROWS,
                )
                .build()?;

                let mut chunks: Vec<Int64Array> = Vec::new();
                let mut positions: Vec<(usize, usize)> = Vec::with_capacity(row_ids.len());

                while let Some(chunk) = row_stream.next_chunk()? {
                    let batch = chunk.to_record_batch();
                    if batch.num_rows() == 0 {
                        continue;
                    }
                    if batch.num_columns() != 1 {
                        return Err(Error::Internal(
                            "ORDER BY gather produced unexpected column count".into(),
                        ));
                    }
                    let array = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| {
                            Error::InvalidArgumentError(
                                "ORDER BY integer expects Int64 column".into(),
                            )
                        })?
                        .clone();

                    let chunk_idx = chunks.len();
                    let row_count = batch.num_rows();
                    chunks.push(array);
                    for offset in 0..row_count {
                        positions.push((chunk_idx, offset));
                    }
                }

                if positions.len() != row_ids.len() {
                    return Err(Error::Internal(
                        "ORDER BY gather produced inconsistent row counts".into(),
                    ));
                }

                let mut indices: Vec<(usize, RowId)> =
                    row_ids.iter().copied().enumerate().collect();
                indices.sort_by(|(ai, arid), (bi, brid)| {
                    let (chunk_a, offset_a) = positions[*ai];
                    let (chunk_b, offset_b) = positions[*bi];
                    let array_a = &chunks[chunk_a];
                    let array_b = &chunks[chunk_b];
                    let left = if array_a.is_null(offset_a) {
                        None
                    } else {
                        Some(array_a.value(offset_a))
                    };
                    let right = if array_b.is_null(offset_b) {
                        None
                    } else {
                        Some(array_b.value(offset_b))
                    };
                    let ord = compare_option_values(left, right, ascending, order.nulls_first);
                    if ord == Ordering::Equal {
                        arid.cmp(brid)
                    } else {
                        ord
                    }
                });
                row_ids = indices.into_iter().map(|(_, rid)| rid).collect();
            }
            ScanOrderTransform::IdentityUtf8 => {
                let mut row_stream = RowStreamBuilder::new(
                    store,
                    self.table.table_id(),
                    Arc::clone(&out_schema),
                    Arc::clone(&logical_fields_arc),
                    Arc::clone(&projection_evals_arc),
                    Arc::clone(&passthrough_fields_arc),
                    Arc::clone(&unique_index_arc),
                    Arc::clone(&numeric_fields_arc),
                    false,
                    GatherNullPolicy::IncludeNulls,
                    row_ids.clone(),
                    STREAM_BATCH_ROWS,
                )
                .build()?;

                let mut chunks: Vec<StringArray> = Vec::new();
                let mut positions: Vec<(usize, usize)> = Vec::with_capacity(row_ids.len());

                while let Some(chunk) = row_stream.next_chunk()? {
                    let batch = chunk.to_record_batch();
                    if batch.num_rows() == 0 {
                        continue;
                    }
                    if batch.num_columns() != 1 {
                        return Err(Error::Internal(
                            "ORDER BY gather produced unexpected column count".into(),
                        ));
                    }
                    let array = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| {
                            Error::InvalidArgumentError("ORDER BY text expects Utf8 column".into())
                        })?
                        .clone();

                    let chunk_idx = chunks.len();
                    let row_count = batch.num_rows();
                    chunks.push(array);
                    for offset in 0..row_count {
                        positions.push((chunk_idx, offset));
                    }
                }

                if positions.len() != row_ids.len() {
                    return Err(Error::Internal(
                        "ORDER BY gather produced inconsistent row counts".into(),
                    ));
                }

                let mut indices: Vec<(usize, RowId)> =
                    row_ids.iter().copied().enumerate().collect();
                indices.sort_by(|(ai, arid), (bi, brid)| {
                    let (chunk_a, offset_a) = positions[*ai];
                    let (chunk_b, offset_b) = positions[*bi];
                    let array_a = &chunks[chunk_a];
                    let array_b = &chunks[chunk_b];
                    let left = if array_a.is_null(offset_a) {
                        None
                    } else {
                        Some(array_a.value(offset_a))
                    };
                    let right = if array_b.is_null(offset_b) {
                        None
                    } else {
                        Some(array_b.value(offset_b))
                    };
                    let ord = compare_option_values(left, right, ascending, order.nulls_first);
                    if ord == Ordering::Equal {
                        arid.cmp(brid)
                    } else {
                        ord
                    }
                });
                row_ids = indices.into_iter().map(|(_, rid)| rid).collect();
            }
            ScanOrderTransform::CastUtf8ToInteger => {
                let mut row_stream = RowStreamBuilder::new(
                    store,
                    self.table.table_id(),
                    Arc::clone(&out_schema),
                    Arc::clone(&logical_fields_arc),
                    Arc::clone(&projection_evals_arc),
                    Arc::clone(&passthrough_fields_arc),
                    Arc::clone(&unique_index_arc),
                    Arc::clone(&numeric_fields_arc),
                    false,
                    GatherNullPolicy::IncludeNulls,
                    row_ids.clone(),
                    STREAM_BATCH_ROWS,
                )
                .build()?;

                let mut keys: Vec<Option<i64>> = Vec::with_capacity(row_ids.len());

                while let Some(chunk) = row_stream.next_chunk()? {
                    let batch = chunk.to_record_batch();
                    if batch.num_rows() == 0 {
                        continue;
                    }
                    if batch.num_columns() != 1 {
                        return Err(Error::Internal(
                            "ORDER BY gather produced unexpected column count".into(),
                        ));
                    }
                    let array = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| {
                            Error::InvalidArgumentError("ORDER BY CAST expects Utf8 column".into())
                        })?;

                    for offset in 0..batch.num_rows() {
                        let key = if array.is_null(offset) {
                            None
                        } else {
                            array.value(offset).parse::<i64>().ok()
                        };
                        keys.push(key);
                    }
                }

                if keys.len() != row_ids.len() {
                    return Err(Error::Internal(
                        "ORDER BY gather produced inconsistent row counts".into(),
                    ));
                }

                let mut indices: Vec<(usize, RowId)> =
                    row_ids.iter().copied().enumerate().collect();
                indices.sort_by(|(ai, arid), (bi, brid)| {
                    let left = keys[*ai];
                    let right = keys[*bi];
                    let ord = compare_option_values(left, right, ascending, order.nulls_first);
                    if ord == Ordering::Equal {
                        arid.cmp(brid)
                    } else {
                        ord
                    }
                });
                row_ids = indices.into_iter().map(|(_, rid)| rid).collect();
            }
        }

        Ok(row_ids)
    }

    fn collect_all_row_ids_for_field(
        &self,
        field_id: FieldId,
        cache: &mut FxHashMap<FieldId, Vec<RowId>>,
    ) -> LlkvResult<Vec<RowId>> {
        if let Some(existing) = cache.get(&field_id) {
            return Ok(existing.clone());
        }

        let filter = Filter {
            field_id,
            op: Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            },
        };

        let ids = self.collect_row_ids_for_filter(&filter)?;
        cache.insert(field_id, ids.clone());
        Ok(ids)
    }

    fn collect_matching_row_ids<T>(
        &self,
        field_id: LogicalFieldId,
        op: &Operator<'_>,
    ) -> LlkvResult<Vec<RowId>>
    where
        T: FilterPrimitive<Native = <T as ArrowPrimitiveType>::Native> + ArrowPrimitiveType,
        <T as ArrowPrimitiveType>::Native: FromLiteral + Copy + PredicateValue,
    {
        let predicate = build_fixed_width_predicate::<T>(op).map_err(Error::predicate_build)?;
        self.table.store().filter_row_ids::<T>(field_id, &predicate)
    }

    fn collect_matching_row_ids_string<O>(
        &self,
        field_id: LogicalFieldId,
        op: &Operator<'_>,
    ) -> LlkvResult<Vec<RowId>>
    where
        O: OffsetSizeTrait + llkv_column_map::store::scan::filter::StringContainsKernel,
    {
        let predicate = build_var_width_predicate(op).map_err(Error::predicate_build)?;
        self.table
            .store()
            .filter_row_ids::<llkv_column_map::store::scan::filter::Utf8Filter<O>>(
                field_id, &predicate,
            )
    }

    fn collect_matching_row_ids_string_fused<O>(
        &self,
        field_id: LogicalFieldId,
        ops: &[Operator<'_>],
    ) -> LlkvResult<Vec<RowId>>
    where
        O: OffsetSizeTrait + llkv_column_map::store::scan::filter::StringContainsKernel,
    {
        // Build predicates for each operator and call the fused dispatch on Utf8Filter
        let mut preds: Vec<llkv_expr::typed_predicate::Predicate<String>> =
            Vec::with_capacity(ops.len());
        for op in ops {
            let p = build_var_width_predicate(op).map_err(Error::predicate_build)?;
            preds.push(p);
        }
        // Call the specialized fused runtime on Utf8Filter via the FilterDispatch trait
        let fused = <llkv_column_map::store::scan::filter::Utf8Filter<O> as llkv_column_map::store::scan::filter::FilterDispatch>::run_fused(
            self.table.store(),
            field_id,
            &preds,
        )?;
        Ok(fused)
    }

    fn collect_matching_row_ids_bool(
        &self,
        field_id: LogicalFieldId,
        op: &Operator<'_>,
    ) -> LlkvResult<Vec<RowId>> {
        let predicate = build_bool_predicate(op).map_err(Error::predicate_build)?;
        self.table
            .store()
            .filter_row_ids::<arrow::datatypes::BooleanType>(field_id, &predicate)
    }

    fn collect_matching_row_ids_bool_fused(
        &self,
        field_id: LogicalFieldId,
        ops: &[Operator<'_>],
    ) -> LlkvResult<Vec<RowId>> {
        let mut preds: Vec<llkv_expr::typed_predicate::Predicate<bool>> =
            Vec::with_capacity(ops.len());
        for op in ops {
            let p = build_bool_predicate(op).map_err(Error::predicate_build)?;
            preds.push(p);
        }
        let fused = <arrow::datatypes::BooleanType as llkv_column_map::store::scan::filter::FilterDispatch>::run_fused(
            self.table.store(),
            field_id,
            &preds,
        )?;
        Ok(fused)
    }

    fn collect_matching_row_ids_fused<T>(
        &self,
        field_id: LogicalFieldId,
        ops: &[Operator<'_>],
    ) -> LlkvResult<Vec<RowId>>
    where
        T: FilterPrimitive<Native = <T as ArrowPrimitiveType>::Native> + ArrowPrimitiveType,
        <T as ArrowPrimitiveType>::Native: FromLiteral + Copy + PredicateValue,
    {
        let mut preds: Vec<
            llkv_expr::typed_predicate::Predicate<<T as ArrowPrimitiveType>::Native>,
        > = Vec::with_capacity(ops.len());
        for op in ops {
            let p = build_fixed_width_predicate::<T>(op).map_err(Error::predicate_build)?;
            preds.push(p);
        }
        // Call the fused dispatch implemented on FilterDispatch for primitives
        let fused = <T as llkv_column_map::store::scan::filter::FilterDispatch>::run_fused(
            self.table.store(),
            field_id,
            &preds,
        )?;
        Ok(fused)
    }
}

pub(crate) fn collect_row_ids_for_table<'expr, P>(
    table: &Table<P>,
    filter_expr: &Expr<'expr, FieldId>,
) -> LlkvResult<Vec<RowId>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let executor = TableExecutor::new(table);
    let fusion_cache = PredicateFusionCache::from_expr(filter_expr);
    let mut all_rows_cache: FxHashMap<FieldId, Vec<RowId>> = FxHashMap::default();
    executor.collect_row_ids_for_expr(filter_expr, &fusion_cache, &mut all_rows_cache)
}

fn computed_expr_requires_numeric(expr: &ScalarExpr<FieldId>) -> bool {
    match expr {
        ScalarExpr::Literal(_) => false,
        ScalarExpr::Column(_) => true,
        ScalarExpr::Binary { .. } => true,
        ScalarExpr::Aggregate(_) => false, // Aggregates are computed separately
        ScalarExpr::GetField { .. } => false, // GetField requires raw arrays, not numeric conversion
    }
}

fn infer_literal_datatype(literal: &Literal) -> LlkvResult<DataType> {
    match literal {
        Literal::Integer(_) => Ok(DataType::Int64),
        Literal::Float(_) => Ok(DataType::Float64),
        Literal::Boolean(_) => Ok(DataType::Boolean),
        Literal::String(_) => Ok(DataType::Utf8),
        Literal::Null => Ok(DataType::Null),
        Literal::Struct(fields) => {
            let inferred_fields = fields
                .iter()
                .map(|(name, nested)| {
                    let dtype = infer_literal_datatype(nested.as_ref())?;
                    Ok(Field::new(name.clone(), dtype, true))
                })
                .collect::<LlkvResult<Vec<_>>>()?;
            Ok(DataType::Struct(inferred_fields.into()))
        }
    }
}

fn synthesize_computed_literal_array(
    info: &ComputedProjectionInfo,
    data_type: &DataType,
    row_count: usize,
) -> LlkvResult<ArrayRef> {
    if row_count == 0 {
        return Ok(new_null_array(data_type, 0));
    }

    match &info.expr {
        ScalarExpr::Literal(Literal::Integer(value)) => {
            let v = i64::try_from(*value).map_err(|_| {
                Error::InvalidArgumentError(
                    "integer literal exceeds supported range for INT64 column".into(),
                )
            })?;
            Ok(Arc::new(Int64Array::from(vec![v; row_count])) as ArrayRef)
        }
        ScalarExpr::Literal(Literal::Float(value)) => {
            Ok(Arc::new(Float64Array::from(vec![*value; row_count])) as ArrayRef)
        }
        ScalarExpr::Literal(Literal::Boolean(value)) => {
            Ok(Arc::new(BooleanArray::from(vec![*value; row_count])) as ArrayRef)
        }
        ScalarExpr::Literal(Literal::String(value)) => {
            Ok(Arc::new(StringArray::from(vec![value.clone(); row_count])) as ArrayRef)
        }
        ScalarExpr::Literal(Literal::Null) => Ok(new_null_array(data_type, row_count)),
        ScalarExpr::Literal(Literal::Struct(fields)) => {
            // Build a struct array from the literal fields
            use arrow::array::StructArray;

            // Convert each field to an array
            let mut field_arrays = Vec::new();
            let mut arrow_fields = Vec::new();

            for (field_name, field_literal) in fields {
                // Infer the proper data type from the literal
                let field_dtype = infer_literal_datatype(field_literal.as_ref())?;
                arrow_fields.push(Field::new(field_name.clone(), field_dtype.clone(), true));

                // Create the array for this field
                let field_array = match field_literal.as_ref() {
                    Literal::Integer(v) => {
                        let int_val = i64::try_from(*v).unwrap_or(0);
                        Arc::new(Int64Array::from(vec![int_val; row_count])) as ArrayRef
                    }
                    Literal::Float(v) => {
                        Arc::new(Float64Array::from(vec![*v; row_count])) as ArrayRef
                    }
                    Literal::Boolean(v) => {
                        Arc::new(BooleanArray::from(vec![*v; row_count])) as ArrayRef
                    }
                    Literal::String(v) => {
                        Arc::new(StringArray::from(vec![v.clone(); row_count])) as ArrayRef
                    }
                    Literal::Null => new_null_array(&field_dtype, row_count),
                    Literal::Struct(nested_fields) => {
                        // Recursively build nested struct
                        // Create a temporary ComputedProjectionInfo for nested struct
                        let nested_info = ComputedProjectionInfo {
                            expr: ScalarExpr::Literal(Literal::Struct(nested_fields.clone())),
                            alias: field_name.clone(),
                        };
                        synthesize_computed_literal_array(&nested_info, &field_dtype, row_count)?
                    }
                };

                field_arrays.push(field_array);
            }

            let struct_array = StructArray::try_new(
                arrow_fields.into(),
                field_arrays,
                None, // No null buffer
            )
            .map_err(|e| Error::Internal(format!("failed to create struct array: {}", e)))?;

            Ok(Arc::new(struct_array) as ArrayRef)
        }
        ScalarExpr::Column(_)
        | ScalarExpr::Binary { .. }
        | ScalarExpr::Aggregate(_)
        | ScalarExpr::GetField { .. } => Ok(new_null_array(data_type, row_count)),
    }
}

fn plan_graph_err(err: PlanGraphError) -> Error {
    Error::Internal(format!("plan graph construction failed: {err}"))
}

fn is_trivial_filter(expr: &Expr<'_, FieldId>) -> bool {
    matches!(
        expr,
        Expr::Pred(Filter {
            op: Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            },
            ..
        })
    )
}

fn format_expr(expr: &Expr<'_, FieldId>) -> String {
    match expr {
        Expr::And(children) => {
            if children.is_empty() {
                "TRUE".to_string()
            } else {
                children
                    .iter()
                    .map(format_expr)
                    .collect::<Vec<_>>()
                    .join(" AND ")
            }
        }
        Expr::Or(children) => {
            if children.is_empty() {
                "FALSE".to_string()
            } else {
                children
                    .iter()
                    .map(format_expr)
                    .collect::<Vec<_>>()
                    .join(" OR ")
            }
        }
        Expr::Not(inner) => format!("NOT ({})", format_expr(inner)),
        Expr::Pred(filter) => format_filter(filter),
        Expr::Compare { left, op, right } => format!(
            "{} {} {}",
            format_scalar_expr(left),
            format_compare_op(*op),
            format_scalar_expr(right)
        ),
        Expr::Literal(value) => {
            if *value {
                "TRUE".to_string()
            } else {
                "FALSE".to_string()
            }
        }
    }
}

fn format_filter(filter: &Filter<'_, FieldId>) -> String {
    format!("field#{} {}", filter.field_id, format_operator(&filter.op))
}

fn format_operator(op: &Operator<'_>) -> String {
    match op {
        Operator::Equals(lit) => format!("= {}", format_literal(lit)),
        Operator::Range { lower, upper } => format!(
            "IN {} .. {}",
            format_range_bound_lower(lower),
            format_range_bound_upper(upper)
        ),
        Operator::GreaterThan(lit) => format!("> {}", format_literal(lit)),
        Operator::GreaterThanOrEquals(lit) => format!(">= {}", format_literal(lit)),
        Operator::LessThan(lit) => format!("< {}", format_literal(lit)),
        Operator::LessThanOrEquals(lit) => format!("<= {}", format_literal(lit)),
        Operator::In(values) => {
            let rendered: Vec<String> = values.iter().map(format_literal).collect();
            format!("IN {{{}}}", rendered.join(", "))
        }
        Operator::StartsWith {
            pattern,
            case_sensitive,
        } => format_pattern_op("STARTS WITH", pattern, *case_sensitive),
        Operator::EndsWith {
            pattern,
            case_sensitive,
        } => format_pattern_op("ENDS WITH", pattern, *case_sensitive),
        Operator::Contains {
            pattern,
            case_sensitive,
        } => format_pattern_op("CONTAINS", pattern, *case_sensitive),
        Operator::IsNull => "IS NULL".to_string(),
        Operator::IsNotNull => "IS NOT NULL".to_string(),
    }
}

fn format_pattern_op(op_name: &str, pattern: &str, case_sensitive: bool) -> String {
    let mut rendered = format!("{} \"{}\"", op_name, escape_string(pattern));
    if !case_sensitive {
        rendered.push_str(" (case-insensitive)");
    }
    rendered
}

fn format_range_bound_lower(bound: &Bound<Literal>) -> String {
    match bound {
        Bound::Unbounded => "-inf".to_string(),
        Bound::Included(lit) => format!("[{}", format_literal(lit)),
        Bound::Excluded(lit) => format!("({}", format_literal(lit)),
    }
}

fn format_range_bound_upper(bound: &Bound<Literal>) -> String {
    match bound {
        Bound::Unbounded => "+inf".to_string(),
        Bound::Included(lit) => format!("{}]", format_literal(lit)),
        Bound::Excluded(lit) => format!("{})", format_literal(lit)),
    }
}

fn format_scalar_expr(expr: &ScalarExpr<FieldId>) -> String {
    match expr {
        ScalarExpr::Column(fid) => format!("col#{}", fid),
        ScalarExpr::Literal(lit) => format_literal(lit),
        ScalarExpr::Binary { left, op, right } => format!(
            "({} {} {})",
            format_scalar_expr(left),
            format_binary_op(*op),
            format_scalar_expr(right)
        ),
        ScalarExpr::Aggregate(agg) => format!("AGG({:?})", agg),
        ScalarExpr::GetField { base, field_name } => {
            format!("{}.{}", format_scalar_expr(base), field_name)
        }
    }
}

fn format_binary_op(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Subtract => "-",
        BinaryOp::Multiply => "*",
        BinaryOp::Divide => "/",
        BinaryOp::Modulo => "%",
    }
}

fn format_compare_op(op: CompareOp) -> &'static str {
    match op {
        CompareOp::Eq => "=",
        CompareOp::NotEq => "!=",
        CompareOp::Lt => "<",
        CompareOp::LtEq => "<=",
        CompareOp::Gt => ">",
        CompareOp::GtEq => ">=",
    }
}

fn format_literal(lit: &Literal) -> String {
    match lit {
        Literal::Integer(i) => i.to_string(),
        Literal::Float(f) => f.to_string(),
        Literal::Boolean(b) => b.to_string(),
        Literal::String(s) => format!("\"{}\"", escape_string(s)),
        Literal::Null => "NULL".to_string(),
        Literal::Struct(fields) => {
            let field_strs: Vec<_> = fields
                .iter()
                .map(|(name, lit)| format!("{}: {}", name, format_literal(lit)))
                .collect();
            format!("{{{}}}", field_strs.join(", "))
        }
    }
}

fn escape_string(value: &str) -> String {
    value.chars().flat_map(|c| c.escape_default()).collect()
}

fn compare_option_values<T: Ord>(
    left: Option<T>,
    right: Option<T>,
    ascending: bool,
    nulls_first: bool,
) -> Ordering {
    match (left, right) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => {
            if nulls_first {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (Some(_), None) => {
            if nulls_first {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
        (Some(a), Some(b)) => {
            if ascending {
                a.cmp(&b)
            } else {
                b.cmp(&a)
            }
        }
    }
}

fn is_full_range_filter(expr: &Expr<'_, FieldId>, expected_field: FieldId) -> bool {
    matches!(
        expr,
        Expr::Pred(Filter {
            field_id,
            op:
                Operator::Range {
                    lower: Bound::Unbounded,
                    upper: Bound::Unbounded,
                },
        }) if *field_id == expected_field
    )
}

struct SingleColumnStreamVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    schema: Arc<Schema>,
    on_batch: &'a mut F,
    error: Option<Error>,
}

impl<'a, F> SingleColumnStreamVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    fn new(schema: Arc<Schema>, on_batch: &'a mut F) -> Self {
        Self {
            schema,
            on_batch,
            error: None,
        }
    }

    fn emit_array_ref(&mut self, array: ArrayRef) {
        if self.error.is_some() {
            return;
        }

        match RecordBatch::try_new(self.schema.clone(), vec![array]) {
            Ok(batch) => (self.on_batch)(batch),
            Err(e) => {
                self.error = Some(Error::Internal(format!("arrow error: {e}")));
            }
        }
    }

    fn emit_array<A>(&mut self, values: &A)
    where
        A: Array + Clone + 'static,
    {
        if self.error.is_some() {
            return;
        }

        let array: ArrayRef = Arc::new(values.clone()) as ArrayRef;
        self.emit_array_ref(array);
    }

    fn reject_row_ids(&mut self) {
        if self.error.is_none() {
            self.error = Some(Error::Internal(
                "unexpected row-id payload in single-column fast path".into(),
            ));
        }
    }

    fn finish(self) -> LlkvResult<()> {
        match self.error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

struct ComputedSingleColumnVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    schema: Arc<Schema>,
    expr: ScalarExpr<FieldId>,
    lfid: LogicalFieldId,
    numeric_fields: FxHashSet<FieldId>,
    on_batch: &'a mut F,
    error: Option<Error>,
}

impl<'a, F> ComputedSingleColumnVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    fn new(
        schema: Arc<Schema>,
        expr: ScalarExpr<FieldId>,
        lfid: LogicalFieldId,
        numeric_fields: FxHashSet<FieldId>,
        on_batch: &'a mut F,
    ) -> Self {
        Self {
            schema,
            expr,
            lfid,
            numeric_fields,
            on_batch,
            error: None,
        }
    }

    fn process_array(&mut self, array: ArrayRef) {
        if self.error.is_some() || array.is_empty() {
            return;
        }

        let lfids = [self.lfid];
        let arrays = vec![array.clone()];
        let numeric_arrays =
            match NumericKernels::prepare_numeric_arrays(&lfids, &arrays, &self.numeric_fields) {
                Ok(map) => map,
                Err(err) => {
                    self.error = Some(err);
                    return;
                }
            };

        let len = array.len();
        let evaluated = match NumericKernels::evaluate_batch(&self.expr, len, &numeric_arrays) {
            Ok(arr) => arr,
            Err(err) => {
                self.error = Some(err);
                return;
            }
        };

        if let Err(err) = RecordBatch::try_new(self.schema.clone(), vec![evaluated])
            .map(|batch| (self.on_batch)(batch))
        {
            self.error = Some(Error::Internal(format!("arrow error: {err}")));
        }
    }

    fn process_values<A>(&mut self, values: &A)
    where
        A: Array + Clone + 'static,
    {
        if self.error.is_some() {
            return;
        }
        let array: ArrayRef = Arc::new(values.clone()) as ArrayRef;
        self.process_array(array);
    }

    fn reject_row_ids(&mut self) {
        if self.error.is_none() {
            self.error = Some(Error::Internal(
                "unexpected row-id payload in computed single-column fast path".into(),
            ));
        }
    }

    fn finish(self) -> LlkvResult<()> {
        match self.error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

impl<'a, F> llkv_column_map::scan::PrimitiveVisitor for ComputedSingleColumnVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    llkv_for_each_arrow_numeric!(impl_computed_emit_chunk);
}

impl<'a, F> llkv_column_map::scan::PrimitiveWithRowIdsVisitor for ComputedSingleColumnVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    llkv_for_each_arrow_numeric!(impl_computed_reject_chunk_with_rids);
}

impl<'a, F> llkv_column_map::scan::PrimitiveSortedVisitor for ComputedSingleColumnVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    llkv_for_each_arrow_numeric!(impl_computed_emit_sorted_run);
}

impl<'a, F> llkv_column_map::scan::PrimitiveSortedWithRowIdsVisitor
    for ComputedSingleColumnVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    llkv_for_each_arrow_numeric!(impl_computed_reject_sorted_run_with_rids);
}

impl<'a, F> llkv_column_map::scan::PrimitiveVisitor for SingleColumnStreamVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    llkv_for_each_arrow_numeric!(impl_single_column_emit_chunk);
}

impl<'a, F> llkv_column_map::scan::PrimitiveWithRowIdsVisitor for SingleColumnStreamVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    llkv_for_each_arrow_numeric!(impl_single_column_reject_chunk_with_rids);
}

impl<'a, F> llkv_column_map::scan::PrimitiveSortedVisitor for SingleColumnStreamVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    llkv_for_each_arrow_numeric!(impl_single_column_emit_sorted_run);
}

impl<'a, F> llkv_column_map::scan::PrimitiveSortedWithRowIdsVisitor
    for SingleColumnStreamVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    llkv_for_each_arrow_numeric!(impl_single_column_reject_sorted_run_with_rids);
}

struct AffineSingleColumnVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    schema: Arc<Schema>,
    scale: f64,
    offset: f64,
    on_batch: &'a mut F,
    error: Option<Error>,
}

impl<'a, F> AffineSingleColumnVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    fn new(schema: Arc<Schema>, scale: f64, offset: f64, on_batch: &'a mut F) -> Self {
        Self {
            schema,
            scale,
            offset,
            on_batch,
            error: None,
        }
    }

    fn emit_array(&mut self, array: arrow::array::Float64Array) {
        if self.error.is_some() {
            return;
        }
        let array_ref: ArrayRef = Arc::new(array) as ArrayRef;
        match RecordBatch::try_new(self.schema.clone(), vec![array_ref]) {
            Ok(batch) => (self.on_batch)(batch),
            Err(err) => {
                self.error = Some(Error::Internal(format!("arrow error: {err}")));
            }
        }
    }

    fn emit_no_nulls<G>(&mut self, len: usize, mut value_at: G)
    where
        G: FnMut(usize) -> f64,
    {
        if self.error.is_some() || len == 0 {
            return;
        }
        let scale = self.scale;
        let offset = self.offset;
        let iter = (0..len).map(|idx| {
            let base = value_at(idx);
            scale * base + offset
        });
        let array = arrow::array::Float64Array::from_iter_values(iter);
        self.emit_array(array);
    }

    fn emit_with_nulls<GNull, GValue>(
        &mut self,
        len: usize,
        mut is_null: GNull,
        mut value_at: GValue,
    ) where
        GNull: FnMut(usize) -> bool,
        GValue: FnMut(usize) -> f64,
    {
        if self.error.is_some() || len == 0 {
            return;
        }
        let scale = self.scale;
        let offset = self.offset;
        let iter = (0..len).map(|idx| {
            if is_null(idx) {
                None
            } else {
                let base = value_at(idx);
                Some(scale * base + offset)
            }
        });
        let array = arrow::array::Float64Array::from_iter(iter);
        self.emit_array(array);
    }

    fn reject_row_ids(&mut self) {
        if self.error.is_none() {
            self.error = Some(Error::Internal(
                "unexpected row-id payload in affine single-column fast path".into(),
            ));
        }
    }

    fn finish(self) -> LlkvResult<()> {
        match self.error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

impl<'a, F> llkv_column_map::scan::PrimitiveVisitor for AffineSingleColumnVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    llkv_for_each_arrow_numeric!(impl_affine_emit_chunk);
}

impl<'a, F> llkv_column_map::scan::PrimitiveWithRowIdsVisitor for AffineSingleColumnVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    llkv_for_each_arrow_numeric!(impl_affine_reject_chunk_with_rids);
}

impl<'a, F> llkv_column_map::scan::PrimitiveSortedVisitor for AffineSingleColumnVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    llkv_for_each_arrow_numeric!(impl_affine_emit_sorted_run);
}

impl<'a, F> llkv_column_map::scan::PrimitiveSortedWithRowIdsVisitor
    for AffineSingleColumnVisitor<'a, F>
where
    F: FnMut(RecordBatch),
{
    llkv_for_each_arrow_numeric!(impl_affine_reject_sorted_run_with_rids);
}

#[derive(Default)]
struct RowIdScanCollector {
    row_ids: Vec<RowId>,
}

impl RowIdScanCollector {
    fn extend_from_array(&mut self, row_ids: &UInt64Array) {
        for idx in 0..row_ids.len() {
            self.row_ids.push(row_ids.value(idx));
        }
    }

    fn extend_from_slice(&mut self, row_ids: &UInt64Array, start: usize, len: usize) {
        if len == 0 {
            return;
        }
        let end = (start + len).min(row_ids.len());
        for idx in start..end {
            self.row_ids.push(row_ids.value(idx));
        }
    }

    fn into_inner(self) -> Vec<RowId> {
        self.row_ids
    }
}

impl llkv_column_map::scan::PrimitiveVisitor for RowIdScanCollector {
    llkv_for_each_arrow_numeric!(impl_row_id_ignore_chunk);
    llkv_for_each_arrow_boolean!(impl_row_id_ignore_chunk);
    llkv_for_each_arrow_string!(impl_row_id_ignore_chunk);
}

impl llkv_column_map::scan::PrimitiveSortedVisitor for RowIdScanCollector {
    llkv_for_each_arrow_numeric!(impl_row_id_ignore_sorted_run);
    llkv_for_each_arrow_boolean!(impl_row_id_ignore_sorted_run);
    llkv_for_each_arrow_string!(impl_row_id_ignore_sorted_run);
}

impl llkv_column_map::scan::PrimitiveWithRowIdsVisitor for RowIdScanCollector {
    llkv_for_each_arrow_numeric!(impl_row_id_collect_chunk_with_rids);
    llkv_for_each_arrow_boolean!(impl_row_id_collect_chunk_with_rids);
    llkv_for_each_arrow_string!(impl_row_id_collect_chunk_with_rids);
}

impl llkv_column_map::scan::PrimitiveSortedWithRowIdsVisitor for RowIdScanCollector {
    llkv_for_each_arrow_numeric!(impl_row_id_collect_sorted_run_with_rids);
    llkv_for_each_arrow_boolean!(impl_row_id_collect_sorted_run_with_rids);
    llkv_for_each_arrow_string!(impl_row_id_collect_sorted_run_with_rids);

    fn null_run(&mut self, row_ids: &UInt64Array, start: usize, len: usize) {
        self.extend_from_slice(row_ids, start, len);
    }
}

struct RowIdChunkEmitter<'a, F>
where
    F: FnMut(Vec<RowId>) -> LlkvResult<()>,
{
    chunk_size: usize,
    buffer: Vec<RowId>,
    on_chunk: &'a mut F,
    error: Option<Error>,
}

impl<'a, F> RowIdChunkEmitter<'a, F>
where
    F: FnMut(Vec<RowId>) -> LlkvResult<()>,
{
    fn new(chunk_size: usize, on_chunk: &'a mut F) -> Self {
        let chunk_size = cmp::max(1, chunk_size);
        Self {
            chunk_size,
            buffer: Vec::with_capacity(chunk_size),
            on_chunk,
            error: None,
        }
    }

    fn extend_from_array(&mut self, row_ids: &UInt64Array) {
        if self.error.is_some() {
            return;
        }
        for idx in 0..row_ids.len() {
            self.push(row_ids.value(idx));
        }
    }

    fn extend_from_slice(&mut self, row_ids: &UInt64Array, start: usize, len: usize) {
        if self.error.is_some() || len == 0 {
            return;
        }
        let end = (start + len).min(row_ids.len());
        for idx in start..end {
            self.push(row_ids.value(idx));
        }
    }

    fn push(&mut self, value: u64) {
        if self.error.is_some() {
            return;
        }
        self.buffer.push(value);
        if self.buffer.len() >= self.chunk_size {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.error.is_some() || self.buffer.is_empty() {
            return;
        }
        let chunk = mem::take(&mut self.buffer);
        match (self.on_chunk)(chunk) {
            Ok(_) => {}
            Err(err) => self.error = Some(err),
        }
    }

    fn finish(mut self) -> LlkvResult<()> {
        self.flush();
        match self.error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

impl<'a, F> llkv_column_map::scan::PrimitiveVisitor for RowIdChunkEmitter<'a, F>
where
    F: FnMut(Vec<RowId>) -> LlkvResult<()>,
{
    llkv_for_each_arrow_numeric!(impl_row_id_ignore_chunk);
    llkv_for_each_arrow_boolean!(impl_row_id_ignore_chunk);
}

impl<'a, F> llkv_column_map::scan::PrimitiveSortedVisitor for RowIdChunkEmitter<'a, F>
where
    F: FnMut(Vec<RowId>) -> LlkvResult<()>,
{
    llkv_for_each_arrow_numeric!(impl_row_id_ignore_sorted_run);
    llkv_for_each_arrow_boolean!(impl_row_id_ignore_sorted_run);
}

impl<'a, F> llkv_column_map::scan::PrimitiveWithRowIdsVisitor for RowIdChunkEmitter<'a, F>
where
    F: FnMut(Vec<RowId>) -> LlkvResult<()>,
{
    llkv_for_each_arrow_numeric!(impl_row_id_stream_chunk_with_rids);
    llkv_for_each_arrow_boolean!(impl_row_id_stream_chunk_with_rids);
}

impl<'a, F> llkv_column_map::scan::PrimitiveSortedWithRowIdsVisitor for RowIdChunkEmitter<'a, F>
where
    F: FnMut(Vec<RowId>) -> LlkvResult<()>,
{
    llkv_for_each_arrow_numeric!(impl_row_id_stream_sorted_run_with_rids);
    llkv_for_each_arrow_boolean!(impl_row_id_stream_sorted_run_with_rids);

    fn null_run(&mut self, row_ids: &UInt64Array, start: usize, len: usize) {
        self.extend_from_slice(row_ids, start, len);
    }
}

fn normalize_row_ids(mut row_ids: Vec<RowId>) -> Vec<RowId> {
    row_ids.sort_unstable();
    row_ids.dedup();
    row_ids
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn materialize_row_window<P>(
    store: &llkv_column_map::store::ColumnStore<P>,
    table_id: TableId,
    unique_lfids: &[LogicalFieldId],
    projection_evals: &[ProjectionEval],
    passthrough_fields: &[Option<FieldId>],
    unique_index: &FxHashMap<LogicalFieldId, usize>,
    numeric_fields: &FxHashSet<FieldId>,
    requires_numeric: bool,
    null_policy: GatherNullPolicy,
    out_schema: &Arc<Schema>,
    window: &[RowId],
    mut gather_ctx: Option<&mut MultiGatherContext>,
) -> LlkvResult<Option<RecordBatch>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    if window.is_empty() {
        return Ok(None);
    }

    let mut gathered_batch: Option<RecordBatch> = None;
    let (batch_len, numeric_arrays) = if unique_lfids.is_empty() {
        let numeric_arrays = if requires_numeric {
            Some(NumericKernels::prepare_numeric_arrays(
                &[],
                &[],
                numeric_fields,
            )?)
        } else {
            None
        };
        (window.len(), numeric_arrays)
    } else {
        let mut local_ctx;
        let ctx = match gather_ctx.as_mut() {
            Some(ctx) => ctx,
            None => {
                local_ctx = store.prepare_gather_context(unique_lfids)?;
                &mut local_ctx
            }
        };
        let batch = store.gather_rows_with_reusable_context(ctx, window, null_policy)?;
        if batch.num_rows() == 0 {
            return Ok(None);
        }
        let batch_len = batch.num_rows();
        let numeric_arrays = if requires_numeric {
            Some(NumericKernels::prepare_numeric_arrays(
                unique_lfids,
                batch.columns(),
                numeric_fields,
            )?)
        } else {
            None
        };
        gathered_batch = Some(batch);
        (batch_len, numeric_arrays)
    };

    if batch_len == 0 {
        return Ok(None);
    }

    let gathered_columns: &[ArrayRef] = if let Some(batch) = gathered_batch.as_ref() {
        batch.columns()
    } else {
        &[]
    };

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(projection_evals.len());
    for (idx, eval) in projection_evals.iter().enumerate() {
        match eval {
            ProjectionEval::Column(info) => {
                let arr_idx = *unique_index
                    .get(&info.logical_field_id)
                    .expect("logical field id missing from index");
                columns.push(Arc::clone(&gathered_columns[arr_idx]));
            }
            ProjectionEval::Computed(info) => {
                if let Some(fid) = passthrough_fields[idx] {
                    let lfid = LogicalFieldId::for_user(table_id, fid);
                    let arr_idx = *unique_index
                        .get(&lfid)
                        .expect("passthrough field missing from index");
                    columns.push(Arc::clone(&gathered_columns[arr_idx]));
                    continue;
                }

                let array: ArrayRef = match &info.expr {
                    ScalarExpr::Literal(_) => synthesize_computed_literal_array(
                        info,
                        out_schema.field(idx).data_type(),
                        batch_len,
                    )?,
                    ScalarExpr::GetField { base, field_name } => {
                        fn eval_get_field(
                            expr: &ScalarExpr<FieldId>,
                            field_name: &str,
                            gathered_columns: &[ArrayRef],
                            unique_index: &FxHashMap<LogicalFieldId, usize>,
                            table_id: TableId,
                        ) -> LlkvResult<ArrayRef> {
                            let base_array = match expr {
                                ScalarExpr::Column(fid) => {
                                    let lfid = LogicalFieldId::for_user(table_id, *fid);
                                    let arr_idx = *unique_index.get(&lfid).ok_or_else(|| {
                                        Error::Internal("field missing from unique arrays".into())
                                    })?;
                                    Arc::clone(&gathered_columns[arr_idx])
                                }
                                ScalarExpr::GetField {
                                    base: inner_base,
                                    field_name: inner_field,
                                } => eval_get_field(
                                    inner_base,
                                    inner_field,
                                    gathered_columns,
                                    unique_index,
                                    table_id,
                                )?,
                                _ => {
                                    return Err(Error::InvalidArgumentError(
                                        "GetField base must be a column or another GetField".into(),
                                    ));
                                }
                            };

                            let struct_array = base_array
                                .as_any()
                                .downcast_ref::<arrow::array::StructArray>()
                                .ok_or_else(|| {
                                    Error::InvalidArgumentError(
                                        "GetField can only be applied to struct types".into(),
                                    )
                                })?;

                            struct_array
                                .column_by_name(field_name)
                                .ok_or_else(|| {
                                    Error::InvalidArgumentError(format!(
                                        "Field '{}' not found in struct",
                                        field_name
                                    ))
                                })
                                .map(Arc::clone)
                        }

                        eval_get_field(base, field_name, gathered_columns, unique_index, table_id)?
                    }
                    _ => {
                        let numeric_arrays = numeric_arrays
                            .as_ref()
                            .expect("numeric arrays should exist for computed projection");
                        NumericKernels::evaluate_batch(&info.expr, batch_len, numeric_arrays)?
                    }
                };
                columns.push(array);
            }
        }
    }

    let batch = RecordBatch::try_new(Arc::clone(out_schema), columns)?;
    Ok(Some(batch))
}

fn emit_synthetic_null_batch<F>(
    projection_evals: &[ProjectionEval],
    out_schema: &Arc<Schema>,
    row_count: usize,
    on_batch: &mut F,
) -> LlkvResult<()>
where
    F: FnMut(RecordBatch),
{
    if row_count == 0 {
        return Ok(());
    }

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(projection_evals.len());
    for (idx, eval) in projection_evals.iter().enumerate() {
        let array = match eval {
            ProjectionEval::Column(info) => new_null_array(&info.data_type, row_count),
            ProjectionEval::Computed(info) => synthesize_computed_literal_array(
                info,
                out_schema.field(idx).data_type(),
                row_count,
            )?,
        };
        columns.push(array);
    }

    let batch = RecordBatch::try_new(Arc::clone(out_schema), columns)?;
    on_batch(batch);
    Ok(())
}

fn literal_to_u64(lit: &Literal) -> LlkvResult<u64> {
    u64::from_literal(lit)
        .map_err(|err| Error::InvalidArgumentError(format!("rowid literal cast failed: {err}")))
}

fn lower_bound_index(row_ids: &[RowId], bound: &Bound<Literal>) -> LlkvResult<usize> {
    Ok(match bound {
        Bound::Unbounded => 0,
        Bound::Included(lit) => {
            let value = literal_to_u64(lit)?;
            row_ids.partition_point(|&rid| rid < value)
        }
        Bound::Excluded(lit) => {
            let value = literal_to_u64(lit)?;
            row_ids.partition_point(|&rid| rid <= value)
        }
    })
}

fn upper_bound_index(row_ids: &[RowId], bound: &Bound<Literal>) -> LlkvResult<usize> {
    Ok(match bound {
        Bound::Unbounded => row_ids.len(),
        Bound::Included(lit) => {
            let value = literal_to_u64(lit)?;
            row_ids.partition_point(|&rid| rid <= value)
        }
        Bound::Excluded(lit) => {
            let value = literal_to_u64(lit)?;
            row_ids.partition_point(|&rid| rid < value)
        }
    })
}

fn filter_row_ids_by_operator(row_ids: &[RowId], op: &Operator<'_>) -> LlkvResult<Vec<RowId>> {
    use Operator::*;

    match op {
        Equals(lit) => {
            let value = literal_to_u64(lit)?;
            match row_ids.binary_search(&value) {
                Ok(idx) => Ok(vec![row_ids[idx]]),
                Err(_) => Ok(Vec::new()),
            }
        }
        GreaterThan(lit) => {
            let value = literal_to_u64(lit)?;
            let idx = row_ids.partition_point(|&rid| rid <= value);
            Ok(row_ids[idx..].to_vec())
        }
        GreaterThanOrEquals(lit) => {
            let value = literal_to_u64(lit)?;
            let idx = row_ids.partition_point(|&rid| rid < value);
            Ok(row_ids[idx..].to_vec())
        }
        LessThan(lit) => {
            let value = literal_to_u64(lit)?;
            let idx = row_ids.partition_point(|&rid| rid < value);
            Ok(row_ids[..idx].to_vec())
        }
        LessThanOrEquals(lit) => {
            let value = literal_to_u64(lit)?;
            let idx = row_ids.partition_point(|&rid| rid <= value);
            Ok(row_ids[..idx].to_vec())
        }
        Range { lower, upper } => {
            let start = lower_bound_index(row_ids, lower)?;
            let end = upper_bound_index(row_ids, upper)?;
            if start >= end {
                Ok(Vec::new())
            } else {
                Ok(row_ids[start..end].to_vec())
            }
        }
        In(literals) => {
            if literals.is_empty() {
                return Ok(Vec::new());
            }
            let mut targets = FxHashSet::default();
            for lit in *literals {
                targets.insert(literal_to_u64(lit)?);
            }
            if targets.is_empty() {
                return Ok(Vec::new());
            }
            let mut matches = Vec::with_capacity(targets.len());
            for &rid in row_ids {
                if targets.remove(&rid) {
                    matches.push(rid);
                    if targets.is_empty() {
                        break;
                    }
                }
            }
            Ok(matches)
        }
        StartsWith { .. } | EndsWith { .. } | Contains { .. } => Err(Error::InvalidArgumentError(
            "rowid predicates do not support string pattern matching".into(),
        )),
        IsNull | IsNotNull => Err(Error::InvalidArgumentError(
            "rowid predicates do not support null checks".into(),
        )),
    }
}

fn intersect_sorted(left: Vec<RowId>, right: Vec<RowId>) -> Vec<RowId> {
    let mut result = Vec::with_capacity(left.len().min(right.len()));
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        let lv = left[i];
        let rv = right[j];
        if lv == rv {
            result.push(lv);
            i += 1;
            j += 1;
        } else if lv < rv {
            i += 1;
        } else {
            j += 1;
        }
    }
    result
}

fn union_sorted(left: Vec<RowId>, right: Vec<RowId>) -> Vec<RowId> {
    if left.is_empty() {
        return right;
    }
    if right.is_empty() {
        return left;
    }

    let mut result = Vec::with_capacity(left.len() + right.len());
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        let lv = left[i];
        let rv = right[j];
        if lv == rv {
            result.push(lv);
            i += 1;
            j += 1;
        } else if lv < rv {
            result.push(lv);
            i += 1;
        } else {
            result.push(rv);
            j += 1;
        }
    }

    while i < left.len() {
        result.push(left[i]);
        i += 1;
    }
    while j < right.len() {
        result.push(right[j]);
        j += 1;
    }

    result.dedup();
    result
}

fn difference_sorted(base: Vec<RowId>, subtract: Vec<RowId>) -> Vec<RowId> {
    if base.is_empty() || subtract.is_empty() {
        return base;
    }

    let mut result = Vec::with_capacity(base.len());
    let mut i = 0;
    let mut j = 0;
    while i < base.len() && j < subtract.len() {
        let bv = base[i];
        let sv = subtract[j];
        if bv == sv {
            i += 1;
            j += 1;
        } else if bv < sv {
            result.push(bv);
            i += 1;
        } else {
            j += 1;
        }
    }
    while i < base.len() {
        result.push(base[i]);
        i += 1;
    }
    result
}

fn expand_filter_runs(runs: &[FilterRun]) -> Vec<RowId> {
    let total: usize = runs.iter().map(|r| r.len).sum();
    let mut row_ids = Vec::with_capacity(total);
    for run in runs {
        let mut row_id = run.start_row_id;
        for _ in 0..run.len {
            row_ids.push(row_id);
            row_id += 1;
        }
    }
    row_ids
}

fn is_supported_numeric(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::UInt64
            | DataType::UInt32
            | DataType::UInt16
            | DataType::UInt8
            | DataType::Int64
            | DataType::Int32
            | DataType::Int16
            | DataType::Int8
            | DataType::Float64
            | DataType::Float32
    )
}

// Ensure macro_rules! definitions above are considered used by the compiler.
// We do this by invoking `llkv_for_each_arrow_numeric!` with a tiny no-op
// implementation macro inside a private module. This avoids exporting the
// macros from this crate while preventing `unused_macros` warnings.
mod __planner_macro_uses {
    // Import the helper macro from the column-map crate.
    use llkv_column_map::llkv_for_each_arrow_numeric;

    // A no-op macro that matches the same shape as the impl_* macros and
    // expands to nothing. It's only used here to exercise the macro_rules!
    // definitions so the compiler treats them as referenced.
    macro_rules! __planner_noop_impl {
        (
            $_base:ident,
            $_chunk:ident,
            $_chunk_with_rids:ident,
            $_run:ident,
            $_run_with_rids:ident,
            $array_ty:ty,
            $_arrow_ty:ty,
            $_dtype:expr,
            $_native_ty:ty,
            $_cast:expr
        ) => {};
    }

    // Invoke the dispatcher to expand `__planner_noop_impl` for each numeric
    // type. The expansion is a no-op, but it counts as a use of the impl_*
    // macros above because they are referenced indirectly by other macro
    // expansions in this module.
    llkv_for_each_arrow_numeric!(__planner_noop_impl);
}
