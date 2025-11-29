use croaring::Treemap;
use std::cell::RefCell;
use std::cmp::{self, Ordering};
use std::convert::TryFrom;
use std::mem;
use std::ops::Bound;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Int64Array, OffsetSizeTrait, PrimitiveArray, RecordBatch,
    UInt64Array,
};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, IntervalUnit, Schema};
use arrow_array::types::{Int32Type, Int64Type};

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
use llkv_column_map::{
    llkv_for_each_arrow_boolean, llkv_for_each_arrow_numeric, llkv_for_each_arrow_string,
};
use llkv_compute::analysis::{
    computed_expr_prefers_float, computed_expr_requires_numeric, get_field_dtype,
    scalar_expr_contains_coalesce,
};
use llkv_compute::eval::ScalarExprTypeExt;
use llkv_compute::projection::{
    ComputedLiteralInfo, ProjectionLiteral, emit_synthetic_null_batch, infer_literal_datatype,
    synthesize_computed_literal_array,
};
use llkv_compute::scalar::interval::compare_interval_values;
use llkv_compute::{RowIdFilter, compare_option_values, sort_row_ids_by_primitive};
use llkv_expr::literal::{FromLiteral, Literal};
use llkv_expr::typed_predicate::{
    PredicateValue, build_bool_predicate, build_fixed_width_predicate, build_var_width_predicate,
};
use llkv_expr::{CompareOp, Expr, Filter, Operator, ScalarExpr};
use llkv_result::{Error, Result as LlkvResult};
use llkv_types::decimal::DecimalValue;
use llkv_types::{LogicalFieldId, LogicalStorageNamespace};
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;

use llkv_storage::pager::Pager;

use crate::constants::STREAM_BATCH_ROWS;
use crate::reserved::is_information_schema_table;
use crate::{NumericArrayMap, NumericKernels};

use crate::schema_ext::CachedSchema;
use crate::table::{
    ScanOrderDirection, ScanOrderSpec, ScanOrderTransform, ScanProjection, ScanStreamOptions, Table,
};
use crate::types::{FieldId, ROW_ID_FIELD_ID, RowId, TableId};

use crate::stream::{RowIdSource, RowStream, RowStreamBuilder};
use llkv_plan::{
    DomainOp, DomainProgramId, EvalOp, OwnedFilter, OwnedOperator, PlanEdge, PlanExpression,
    PlanField, PlanGraph, PlanGraphBuilder, PlanGraphError, PlanNode, PlanNodeId, PlanOperator,
    ProgramCompiler, ProgramSet, normalize_predicate,
};
use llkv_scan::sort_row_ids_with_order;

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

pub(crate) type ComputedProjectionInfo = ComputedLiteralInfo<FieldId>;

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
    filter_expr: Arc<Expr<'expr, FieldId>>,
    options: ScanStreamOptions<P>,
    plan_graph: PlanGraph,
    programs: ProgramSet<'expr>,
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
        // Iterative traversal using work stack pattern.
        // See llkv-plan::traversal module documentation for pattern details.
        //
        // This avoids stack overflow on deeply nested expressions (50k+ nodes).
        let mut stack = vec![expr];

        while let Some(node) = stack.pop() {
            match node {
                Expr::Pred(filter) => {
                    let entry = self.per_field.entry(filter.field_id).or_default();
                    entry.total += 1;
                    if matches!(filter.op, Operator::Contains { .. }) {
                        entry.contains += 1;
                    }
                }
                Expr::And(children) | Expr::Or(children) => {
                    for child in children {
                        stack.push(child);
                    }
                }
                Expr::Not(inner) => stack.push(inner),
                Expr::Compare { .. } => {}
                Expr::InList { .. } => {}
                Expr::IsNull { .. } => {}
                Expr::Literal(_) => {}
                Expr::Exists(_) => {}
            }
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
    row_id_cache: RefCell<Option<Treemap>>,
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
        let normalized_filter = normalize_predicate(filter_expr.clone());
        let filter_arc = Arc::new(normalized_filter);
        let plan_graph = self.build_plan_graph(&projections_vec, filter_expr, options.clone())?;
        let programs = ProgramCompiler::new(Arc::clone(&filter_arc)).compile()?;

        Ok(PlannedScan {
            projections: projections_vec,
            filter_expr: Arc::clone(&filter_arc),
            options,
            plan_graph,
            programs,
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
                        expr.format_display()
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

    fn table_row_ids(&self) -> LlkvResult<std::cell::Ref<'_, Treemap>> {
        if self.row_id_cache.borrow().is_none() {
            let computed = self.compute_table_row_ids()?;
            *self.row_id_cache.borrow_mut() = Some(computed);
        }
        let borrow = self.row_id_cache.borrow();
        Ok(std::cell::Ref::map(borrow, |opt| opt.as_ref().unwrap()))
    }

    fn compute_table_row_ids(&self) -> LlkvResult<Treemap> {
        use llkv_column_map::store::rowid_fid;

        let fields = self
            .table
            .store()
            .user_field_ids_for_table(self.table.table_id());
        if fields.is_empty() {
            return Ok(Treemap::new());
        }

        let expected = self
            .table
            .store()
            .total_rows_for_table(self.table.table_id())
            .unwrap_or_default();

        // Optimization: Try scanning the dedicated row_id shadow column first.
        // This is significantly faster than scanning multiple user columns because:
        // 1. It's a single column scan instead of multiple
        // 2. Row IDs are already in the format we need (no deduplication required)
        // 3. The shadow column is guaranteed to have all row IDs for the table
        //
        // We use the first user field to construct the shadow column ID.
        if expected > 0
            && let Some(&first_field) = fields.first()
        {
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
                    let row_ids = collector.into_inner();
                    tracing::trace!(
                        "[PERF] Fast path: collected {} row_ids from shadow column for table {}",
                        row_ids.cardinality(),
                        self.table.table_id()
                    );
                    if row_ids.cardinality() == expected {
                        return Ok(row_ids);
                    }
                    tracing::debug!(
                        "[PERF] Shadow column for table {} returned {} row_ids but expected {}; falling back",
                        self.table.table_id(),
                        row_ids.cardinality(),
                        expected
                    );
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
        let mut collected = Treemap::new();

        for lfid in fields.clone() {
            let mut collector = RowIdScanCollector::default();
            ScanBuilder::new(self.table.store(), lfid)
                .options(ScanOptions {
                    with_row_ids: true,
                    ..Default::default()
                })
                .run(&mut collector)?;
            let rows = collector.into_inner();
            collected.or_inplace(&rows);

            if expected > 0 && collected.cardinality() >= expected {
                break;
            }
        }

        tracing::trace!(
            "[PERF] Multi-column scan: collected {} row_ids for table {}",
            collected.cardinality(),
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
                let bitmap: Treemap = chunk.iter().copied().collect();
                let filtered = filter.filter(self.table, bitmap)?;
                chunk = filtered.iter().collect();
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
            let projection_literals = build_projection_literals(projection_evals, out_schema);
            if let Some(batch) =
                emit_synthetic_null_batch(&projection_literals, out_schema, row_count)?
            {
                on_batch(batch);
            }
        }

        Ok(StreamOutcome::Handled)
    }

    fn collect_row_ids_for_rowid_filter(&self, op: &Operator<'_>) -> LlkvResult<Treemap> {
        let all_row_ids = self.table_row_ids()?;
        if all_row_ids.is_empty() {
            return Ok(Treemap::new());
        }
        all_row_ids.filter_by_operator(op)
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
            programs,
        } = plan;

        if self
            .try_single_column_direct_scan(
                &projections,
                filter_expr.as_ref(),
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
                    if lfid.namespace() != LogicalStorageNamespace::UserData {
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
                        if lfid.namespace() != LogicalStorageNamespace::UserData {
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
                            ScalarExpr::Literal(Literal::Int128(_)) => DataType::Int64,
                            ScalarExpr::Literal(Literal::Float64(_)) => DataType::Float64,
                            ScalarExpr::Literal(Literal::Decimal128(value)) => {
                                DataType::Decimal128(value.precision(), value.scale())
                            }
                            ScalarExpr::Literal(Literal::Boolean(_)) => DataType::Boolean,
                            ScalarExpr::Literal(Literal::String(_)) => DataType::Utf8,
                            ScalarExpr::Literal(Literal::Date32(_)) => DataType::Date32,
                            ScalarExpr::Literal(Literal::Interval(_)) => {
                                DataType::Interval(IntervalUnit::MonthDayNano)
                            }
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
                            ScalarExpr::Cast { data_type, .. } => data_type.clone(),
                            ScalarExpr::Not(_) => DataType::Int64,
                            ScalarExpr::IsNull { .. } => DataType::Int64,
                            ScalarExpr::Binary { .. }
                            | ScalarExpr::Compare { .. }
                            | ScalarExpr::Case { .. }
                            | ScalarExpr::Coalesce(_) => {
                                let mut resolver = |fid: FieldId| {
                                    let lfid = LogicalFieldId::for_user(self.table.table_id(), fid);
                                    lfid_dtypes.get(&lfid).cloned()
                                };

                                let inferred_type = info.expr.infer_result_type(&mut resolver);

                                if let Some(dtype) = inferred_type {
                                    dtype
                                } else if computed_expr_prefers_float(
                                    &info.expr,
                                    self.table.table_id(),
                                    &lfid_dtypes,
                                )? {
                                    DataType::Float64
                                } else {
                                    DataType::Int64
                                }
                            }
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
                            ScalarExpr::GetField { base, field_name } => get_field_dtype(
                                base,
                                field_name,
                                self.table.table_id(),
                                &lfid_dtypes,
                            )?,
                            ScalarExpr::Random => {
                                // RANDOM() returns a float64 value
                                DataType::Float64
                            }
                            ScalarExpr::ScalarSubquery(_) => {
                                // Scalar subqueries will be resolved at execution time
                                // For now, assume they can be null and return a generic type
                                // TODO: Infer type from subquery plan
                                DataType::Utf8
                            }
                        };
                        schema_fields.push(Field::new(info.alias.clone(), dtype, true));
                    }
                }
            }
        }
        let out_schema = Arc::new(Schema::new(schema_fields));

        if options.order.is_none()
            && is_trivial_filter(filter_expr.as_ref())
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

        let fusion_cache = PredicateFusionCache::from_expr(filter_expr.as_ref());
        let mut all_rows_cache: FxHashMap<FieldId, Treemap> = FxHashMap::default();

        // Intentionally skip logging the full filter expression here; deeply nested
        // expressions can exceed the default thread stack when rendered recursively.

        // When we have a trivial filter (no predicates), enumerate ALL row IDs.
        // This is necessary for:
        // 1. MVCC filtering to check visibility of all rows including NULL rows
        // 2. Aggregates like COUNT_NULLS that need to count NULL rows
        // We scan the MVCC created_by column which exists for every row.
        //
        // Historical note: information_schema tables used to have issues with complex
        // filter evaluation when many columns were projected. We previously attempted
        // to fall back to a "scan everything" path when the predicate returned zero
        // matches. That trade-off caused incorrect results for legitimate empty
        // queries (like looking up a dropped table), so we now trust the predicate
        // evaluation and simply log when it yields zero rows.
        let is_info_schema = is_information_schema_table(self.table.table_id());
        let is_trivial = is_trivial_filter(filter_expr.as_ref());

        let row_ids_source = if is_trivial {
            use arrow::datatypes::UInt64Type;
            use llkv_expr::typed_predicate::Predicate;
            let created_lfid = LogicalFieldId::for_mvcc_created_by(self.table.table_id());
            tracing::trace!(
                "[SCAN_STREAM] MVCC + trivial filter: scanning created_by column for all row IDs"
            );
            // Get all rows where created_by exists (which is all rows that have been written)
            let rows = self
                .table
                .store()
                .filter_row_ids::<UInt64Type>(created_lfid, &Predicate::All)?;
            RowIdSource::Vector(rows)
        } else if is_info_schema {
            let row_ids =
                self.collect_row_ids_for_program(&programs, &fusion_cache, &mut all_rows_cache)?;
            let is_empty = match &row_ids {
                RowIdSource::Bitmap(b) => b.is_empty(),
                RowIdSource::Vector(v) => v.is_empty(),
            };
            if is_empty {
                tracing::debug!(
                    "[SCAN_STREAM] information_schema table {}: predicate returned 0 rows",
                    self.table.table_id()
                );
            }
            row_ids
        } else {
            self.collect_row_ids_for_program(&programs, &fusion_cache, &mut all_rows_cache)?
        };

        let row_count = match &row_ids_source {
            RowIdSource::Bitmap(b) => b.cardinality(),
            RowIdSource::Vector(v) => v.len() as u64,
        };

        tracing::trace!(
            "[SCAN_STREAM] collected {} row_ids, has_row_filter={}",
            row_count,
            options.row_id_filter.is_some()
        );

        let final_row_ids: RowIdSource = if let Some(filter) = options.row_id_filter.as_ref() {
            let bitmap = match row_ids_source {
                RowIdSource::Bitmap(b) => b,
                RowIdSource::Vector(v) => Treemap::from_iter(v),
            };
            let before_len = bitmap.cardinality();
            let filtered = filter.filter(self.table, bitmap)?;
            tracing::trace!(
                "[SCAN_STREAM] after MVCC filter: {} -> {} row_ids",
                before_len,
                filtered.cardinality()
            );

            if let Some(order_spec) = options.order {
                let sorted = sort_row_ids_with_order(self.table, &filtered, order_spec)?;
                RowIdSource::Vector(sorted)
            } else {
                RowIdSource::Bitmap(filtered)
            }
        } else if let Some(order_spec) = options.order {
            let bitmap = match row_ids_source {
                RowIdSource::Bitmap(b) => b,
                RowIdSource::Vector(v) => Treemap::from_iter(v),
            };
            let sorted = sort_row_ids_with_order(self.table, &bitmap, order_spec)?;
            RowIdSource::Vector(sorted)
        } else {
            row_ids_source
        };

        let is_empty = match &final_row_ids {
            RowIdSource::Bitmap(b) => b.is_empty(),
            RowIdSource::Vector(v) => v.is_empty(),
        };

        if is_empty {
            tracing::trace!(
                "[SCAN_STREAM] row_ids is empty after filtering, returning early (no synthetic batch)"
            );
            // MVCC Note: If row_ids is empty after MVCC filtering, don't create synthetic batches!
            // The old optimization of creating NULL rows for trivial filters breaks MVCC visibility.
            // An empty row_ids list means NO visible rows, so we should return empty results.
            if options.row_id_filter.is_none() && is_trivial_filter(filter_expr.as_ref()) {
                let total_rows = self.table.total_rows()?;
                let row_count = usize::try_from(total_rows).map_err(|_| {
                    Error::InvalidArgumentError("table row count exceeds supported range".into())
                })?;
                let projection_literals = build_projection_literals(&projection_evals, &out_schema);
                if let Some(batch) =
                    emit_synthetic_null_batch(&projection_literals, &out_schema, row_count)?
                {
                    on_batch(batch);
                }
            }
            return Ok(());
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
            final_row_ids,
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

                if !filter_expr.is_full_range_for(&p.logical_field_id.field_id()) {
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
                if !filter_expr.is_full_range_for(&field_id) {
                    return Ok(StreamOutcome::Fallback);
                }

                if lfid.namespace() != LogicalStorageNamespace::UserData {
                    return Err(Error::InvalidArgumentError(format!(
                        "Computed projection field {:?} must target user data namespace",
                        lfid
                    )));
                }

                let field_alias = alias.clone();
                let dtype = self.table.store().data_type(lfid)?;

                let mut resolver = |fid: FieldId| {
                    if fid == field_id {
                        Some(dtype.clone())
                    } else {
                        None
                    }
                };
                let output_dtype = simplified
                    .infer_result_type(&mut resolver)
                    .unwrap_or(DataType::Float64);

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
                    if !NumericKernels::is_supported_numeric(&dtype) {
                        return Ok(StreamOutcome::Fallback);
                    }
                    if matches!(
                        output_dtype,
                        DataType::Float16 | DataType::Float32 | DataType::Float64
                    ) {
                        let schema = Arc::new(Schema::new(vec![Field::new(
                            field_alias.clone(),
                            output_dtype.clone(),
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
                }

                let mut numeric_fields: FxHashSet<FieldId> = FxHashSet::default();
                numeric_fields.insert(field_id);
                let schema = Arc::new(Schema::new(vec![Field::new(
                    field_alias.clone(),
                    output_dtype,
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

    fn collect_row_ids_for_filter(&self, filter: &OwnedFilter) -> LlkvResult<RowIdSource> {
        if filter.field_id == ROW_ID_FIELD_ID {
            let op = filter.op.to_operator();
            let row_ids = self.collect_row_ids_for_rowid_filter(&op)?;
            tracing::debug!(
                field = "rowid",
                row_count = row_ids.cardinality(),
                "collect_row_ids_for_filter rowid"
            );
            return Ok(RowIdSource::Bitmap(row_ids));
        }

        let filter_lfid = LogicalFieldId::for_user(self.table.table_id(), filter.field_id);
        let dtype = self.table.store().data_type(filter_lfid)?;

        match &filter.op {
            OwnedOperator::IsNotNull => {
                let mut cache = FxHashMap::default();
                let non_null = self.collect_all_row_ids_for_field(filter.field_id, &mut cache)?;
                tracing::debug!(
                    field = ?filter_lfid,
                    row_count = non_null.cardinality(),
                    "collect_row_ids_for_filter NOT NULL fast path"
                );
                return Ok(RowIdSource::Bitmap(non_null));
            }
            OwnedOperator::IsNull => {
                let all_row_ids = self.table_row_ids()?;
                if all_row_ids.is_empty() {
                    return Ok(RowIdSource::Bitmap(Treemap::new()));
                }
                let mut cache = FxHashMap::default();
                let non_null = self.collect_all_row_ids_for_field(filter.field_id, &mut cache)?;
                let null_ids = all_row_ids.clone() - non_null;
                tracing::debug!(
                    field = ?filter_lfid,
                    row_count = null_ids.cardinality(),
                    "collect_row_ids_for_filter NULL fast path"
                );
                return Ok(RowIdSource::Bitmap(null_ids));
            }
            _ => {}
        }

        if let OwnedOperator::Range {
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
            return Ok(RowIdSource::Vector(rows));
        }

        let op = filter.op.to_operator();
        let row_ids = match &dtype {
            DataType::Utf8 => self.collect_matching_row_ids_string::<i32>(filter_lfid, &op),
            DataType::LargeUtf8 => self.collect_matching_row_ids_string::<i64>(filter_lfid, &op),
            DataType::Boolean => self.collect_matching_row_ids_bool(filter_lfid, &op),
            other => llkv_column_map::with_integer_arrow_type!(
                other.clone(),
                |ArrowTy| self.collect_matching_row_ids::<ArrowTy>(filter_lfid, &op),
                Err(Error::Internal(format!(
                    "Filtering on type {:?} is not supported",
                    other
                ))),
            ),
        }?;
        tracing::debug!(
            field = ?filter_lfid,
            row_count = row_ids.cardinality(),
            "collect_row_ids_for_filter general path"
        );

        Ok(RowIdSource::Bitmap(row_ids))
    }

    /// Evaluate a constant-only comparison expression.
    ///
    /// Used for comparisons with no column references, such as those produced
    /// by materializing IN (SELECT ...) subqueries or constant expressions in
    /// WHERE clauses like `WHERE NULL BETWEEN NULL AND (16 / -28)`. Returns
    /// `Ok(Some(_))` when the predicate resolves to a concrete boolean and
    /// `Ok(None)` when NULL-propagation makes the outcome indeterminate.
    fn evaluate_constant_compare(
        left: &ScalarExpr<FieldId>,
        op: CompareOp,
        right: &ScalarExpr<FieldId>,
    ) -> LlkvResult<Option<bool>> {
        let left_lit_opt = NumericKernels::evaluate_constant_literal_expr(left)?;
        let right_lit_opt = NumericKernels::evaluate_constant_literal_expr(right)?;

        let left_lit = match left_lit_opt {
            Some(Literal::Null) | None => return Ok(None),
            Some(lit) => lit,
        };
        let right_lit = match right_lit_opt {
            Some(Literal::Null) | None => return Ok(None),
            Some(lit) => lit,
        };

        // Helper to compare literals
        fn compare_literals(left: &Literal, right: &Literal) -> Option<std::cmp::Ordering> {
            match (left, right) {
                (Literal::Boolean(l), Literal::Boolean(r)) => l.partial_cmp(r),
                (Literal::Int128(l), Literal::Int128(r)) => l.partial_cmp(r),
                (Literal::Float64(l), Literal::Float64(r)) => l.partial_cmp(r),
                (Literal::Decimal128(l), Literal::Decimal128(r)) => Some(l.cmp(r)),
                (Literal::Decimal128(l), Literal::Int128(r)) => {
                    DecimalValue::new(*r, 0).ok().map(|int| l.cmp(&int))
                }
                (Literal::Int128(l), Literal::Decimal128(r)) => {
                    DecimalValue::new(*l, 0).ok().map(|int| int.cmp(r))
                }
                (Literal::Decimal128(l), Literal::Float64(r)) => l.to_f64().partial_cmp(r),
                (Literal::Float64(l), Literal::Decimal128(r)) => l.partial_cmp(&r.to_f64()),
                (Literal::String(l), Literal::String(r)) => l.partial_cmp(r),
                (Literal::Date32(l), Literal::Date32(r)) => l.partial_cmp(r),
                (Literal::Interval(l), Literal::Interval(r)) => {
                    Some(compare_interval_values(*l, *r))
                }
                (Literal::Int128(l), Literal::Float64(r)) => (*l as f64).partial_cmp(r),
                (Literal::Float64(l), Literal::Int128(r)) => l.partial_cmp(&(*r as f64)),
                _ => None,
            }
        }

        // Evaluate based on operation
        let result = match op {
            CompareOp::Eq => left_lit == right_lit,
            CompareOp::NotEq => left_lit != right_lit,
            CompareOp::Lt => {
                compare_literals(&left_lit, &right_lit) == Some(std::cmp::Ordering::Less)
            }
            CompareOp::LtEq => matches!(
                compare_literals(&left_lit, &right_lit),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            ),
            CompareOp::Gt => {
                compare_literals(&left_lit, &right_lit) == Some(std::cmp::Ordering::Greater)
            }
            CompareOp::GtEq => matches!(
                compare_literals(&left_lit, &right_lit),
                Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
            ),
        };

        Ok(Some(result))
    }

    /// Collect all row IDs from the table.
    ///
    /// Used when a constant-only comparison evaluates to true and we need
    /// to return all rows.
    fn collect_all_row_ids(
        &self,
        all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
    ) -> LlkvResult<Treemap> {
        // Use ROW_ID_FIELD_ID to get all row IDs from the table
        self.collect_all_row_ids_for_field(ROW_ID_FIELD_ID, all_rows_cache)
    }

    fn collect_row_ids_for_compare(
        &self,
        left: &ScalarExpr<FieldId>,
        op: CompareOp,
        right: &ScalarExpr<FieldId>,
        all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
    ) -> LlkvResult<Treemap> {
        let mut fields = FxHashSet::default();
        NumericKernels::collect_fields(left, &mut fields);
        NumericKernels::collect_fields(right, &mut fields);

        // Handle constant-only comparisons (e.g., from materialized IN subqueries)
        // These are comparisons like "5 IN (1,2,3)" with no column references
        if fields.is_empty() {
            // Evaluate the constant comparison
            return match Self::evaluate_constant_compare(left, op, right)? {
                Some(true) => self.collect_all_row_ids(all_rows_cache),
                Some(false) | None => Ok(Treemap::new()),
            };
        }

        let mut ordered_fields: Vec<FieldId> = fields.into_iter().collect();
        ordered_fields.sort_unstable();

        let requires_full_scan =
            scalar_expr_contains_coalesce(left) || scalar_expr_contains_coalesce(right);

        let domain = if requires_full_scan {
            let mut union_rows = Treemap::new();
            for fid in &ordered_fields {
                let rows = self.collect_all_row_ids_for_field(*fid, all_rows_cache)?;
                union_rows |= rows;
            }
            union_rows
        } else {
            let mut domain: Option<Treemap> = None;
            for fid in &ordered_fields {
                let rows = self.collect_all_row_ids_for_field(*fid, all_rows_cache)?;
                domain = Some(match domain {
                    Some(existing) => existing & rows,
                    None => rows,
                });
                if let Some(ref d) = domain
                    && d.is_empty()
                {
                    return Ok(Treemap::new());
                }
            }
            if let Some(ref domain_rows) = domain {
                tracing::debug!(
                    ?ordered_fields,
                    domain_len = domain_rows.cardinality(),
                    "collect_row_ids_for_compare domain"
                );
            } else {
                tracing::debug!(?ordered_fields, "collect_row_ids_for_compare domain empty");
            }
            domain.unwrap_or_default()
        };

        if domain.is_empty() {
            return Ok(domain);
        }
        let result = self.evaluate_compare_over_rows(&domain, &ordered_fields, left, op, right)?;
        tracing::debug!(
            ?ordered_fields,
            result_len = result.cardinality(),
            "collect_row_ids_for_compare result"
        );
        Ok(result)
    }

    fn evaluate_constant_in_list(
        &self,
        expr: &ScalarExpr<FieldId>,
        list: &[ScalarExpr<FieldId>],
        negated: bool,
        all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
    ) -> LlkvResult<(Treemap, Treemap)> {
        let arrays: NumericArrayMap = FxHashMap::default();
        let target_array = NumericKernels::evaluate_value(expr, 0, &arrays)?;

        if target_array.data_type() == &arrow::datatypes::DataType::Null || target_array.is_null(0)
        {
            return Ok((Treemap::new(), Treemap::new()));
        }

        let mut matched = false;
        let mut saw_null = false;
        for value_expr in list {
            let value_array = NumericKernels::evaluate_value(value_expr, 0, &arrays)?;
            if value_array.data_type() == &arrow::datatypes::DataType::Null
                || value_array.is_null(0)
            {
                saw_null = true;
            } else {
                let cmp_array =
                    llkv_compute::compute_compare(&value_array, CompareOp::Eq, &target_array)?;
                let cmp = cmp_array.as_any().downcast_ref::<BooleanArray>().unwrap();
                if cmp.value(0) {
                    matched = true;
                    break;
                }
            }
        }

        let outcome = if matched {
            Some(!negated)
        } else if saw_null {
            None
        } else if negated {
            Some(true)
        } else {
            Some(false)
        };

        match outcome {
            Some(true) => {
                let rows = self.collect_all_row_ids(all_rows_cache)?;
                Ok((rows.clone(), rows))
            }
            Some(false) => {
                let rows = self.collect_all_row_ids(all_rows_cache)?;
                Ok((Treemap::new(), rows))
            }
            None => Ok((Treemap::new(), Treemap::new())),
        }
    }

    fn evaluate_constant_is_null(
        &self,
        expr: &ScalarExpr<FieldId>,
        negated: bool,
        _all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
    ) -> LlkvResult<bool> {
        let arrays: NumericArrayMap = FxHashMap::default();
        let value = NumericKernels::evaluate_value(expr, 0, &arrays)?;
        let is_null = value.data_type() == &arrow::datatypes::DataType::Null || value.is_null(0);
        Ok(if negated { !is_null } else { is_null })
    }

    fn collect_row_ids_for_is_null(
        &self,
        expr: &ScalarExpr<FieldId>,
        negated: bool,
        all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
    ) -> LlkvResult<Treemap> {
        let mut fields = FxHashSet::default();
        NumericKernels::collect_fields(expr, &mut fields);

        if fields.is_empty() {
            let matches = self.evaluate_constant_is_null(expr, negated, all_rows_cache)?;
            if matches {
                return self.collect_all_row_ids(all_rows_cache);
            } else {
                return Ok(Treemap::new());
            }
        }

        // Build domain from all referenced fields
        let mut domain: Option<Treemap> = None;
        let mut ordered_fields: Vec<FieldId> = fields.into_iter().collect();
        ordered_fields.sort_unstable();
        for fid in &ordered_fields {
            let rows = self.collect_all_row_ids_for_field(*fid, all_rows_cache)?;
            domain = Some(match domain {
                Some(existing) => existing & rows,
                None => rows,
            });
            if let Some(ref d) = domain
                && d.is_empty()
            {
                return Ok(Treemap::new());
            }
        }

        let domain_rows = domain.unwrap_or_default();
        if domain_rows.is_empty() {
            return Ok(Treemap::new());
        }

        // Evaluate IS NULL over the domain rows using the same pattern as in_list
        self.evaluate_is_null_over_rows(&domain_rows, &ordered_fields, expr, negated)
    }

    fn evaluate_is_null_over_rows(
        &self,
        row_ids: &Treemap,
        fields: &[FieldId],
        expr: &ScalarExpr<FieldId>,
        negated: bool,
    ) -> LlkvResult<Treemap> {
        if row_ids.is_empty() {
            return Ok(Treemap::new());
        }

        // Mirrors llkv_scan::predicate::evaluate_in_list_over_rows until planner
        // routes through the shared storage path
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

        let mut matched_rows = Treemap::new();

        let mut process_chunk = |window: &[RowId], columns: &[ArrayRef]| -> LlkvResult<()> {
            if window.is_empty() {
                return Ok(());
            }

            let mut numeric_arrays: NumericArrayMap = FxHashMap::default();
            if !columns.is_empty() {
                for (lfid, array) in logical_fields_for_arrays.iter().zip(columns.iter()) {
                    let fid = lfid.field_id();
                    if numeric_fields_arc.contains(&fid) {
                        numeric_arrays.insert(fid, array.clone());
                    }
                }
            }

            if has_row_id {
                let rid_values: Vec<i64> = window.iter().map(|rid| *rid as i64).collect();
                let array = Int64Array::from(rid_values);
                numeric_arrays.insert(ROW_ID_FIELD_ID, Arc::new(array));
            }

            for (offset, &row_id) in window.iter().enumerate() {
                let value = NumericKernels::evaluate_value(expr, offset, &numeric_arrays)?;
                let is_null = value.is_null(0);
                let matches = if negated { !is_null } else { is_null };
                if matches {
                    matched_rows.add(row_id);
                }
            }

            Ok(())
        };

        let mut row_stream = RowStreamBuilder::new(
            store,
            self.table.table_id(),
            Arc::clone(&out_schema),
            Arc::clone(&unique_lfids_arc),
            Arc::clone(&projection_evals_arc),
            Arc::clone(&passthrough_fields_arc),
            Arc::clone(&unique_index_arc),
            Arc::clone(&numeric_fields_arc),
            requires_numeric,
            GatherNullPolicy::IncludeNulls,
            row_ids.iter().collect::<Vec<_>>(),
            STREAM_BATCH_ROWS,
        )
        .build()?;

        while let Some(chunk) = row_stream.next_chunk()? {
            let window = chunk.row_ids.values();
            let batch = chunk.to_record_batch();
            process_chunk(window, batch.columns())?;
        }

        Ok(matched_rows)
    }

    fn collect_row_ids_for_program(
        &self,
        programs: &ProgramSet<'_>,
        fusion_cache: &PredicateFusionCache,
        all_rows_cache: &mut FxHashMap<FieldId, Treemap>,
    ) -> LlkvResult<RowIdSource> {
        llkv_scan::predicate::collect_row_ids_for_program(
            self.table,
            programs,
            fusion_cache,
            all_rows_cache,
        )
    }
}

pub(crate) fn collect_row_ids_for_table<'expr, P>(
    table: &Table<P>,
    filter_expr: &Expr<'expr, FieldId>,
) -> LlkvResult<RowIdSource>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let executor = TableExecutor::new(table);
    let fusion_cache = PredicateFusionCache::from_expr(filter_expr);
    let mut all_rows_cache: FxHashMap<FieldId, Treemap> = FxHashMap::default();
    let filter_arc = Arc::new(filter_expr.clone());
    let programs = ProgramCompiler::new(filter_arc).compile()?;
    executor.collect_row_ids_for_program(&programs, &fusion_cache, &mut all_rows_cache)
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
    expr.format_display()
}
struct PrimitiveOrderContext<'a> {
    out_schema: &'a Arc<Schema>,
    logical_fields: &'a Arc<Vec<LogicalFieldId>>,
    projection_evals: &'a Arc<Vec<ProjectionEval>>,
    passthrough_fields: &'a Arc<Vec<Option<FieldId>>>,
    unique_index: &'a Arc<FxHashMap<LogicalFieldId, usize>>,
    numeric_fields: &'a Arc<FxHashSet<FieldId>>,
}

impl<'a> PrimitiveOrderContext<'a> {
    fn new(
        out_schema: &'a Arc<Schema>,
        logical_fields: &'a Arc<Vec<LogicalFieldId>>,
        projection_evals: &'a Arc<Vec<ProjectionEval>>,
        passthrough_fields: &'a Arc<Vec<Option<FieldId>>>,
        unique_index: &'a Arc<FxHashMap<LogicalFieldId, usize>>,
        numeric_fields: &'a Arc<FxHashSet<FieldId>>,
    ) -> Self {
        Self {
            out_schema,
            logical_fields,
            projection_evals,
            passthrough_fields,
            unique_index,
            numeric_fields,
        }
    }
}

struct PrimitiveOrderData<T>
where
    T: ArrowPrimitiveType,
{
    chunks: Vec<PrimitiveArray<T>>,
    positions: Vec<(usize, usize)>,
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

        let mut numeric_arrays: NumericArrayMap = FxHashMap::default();
        if self.numeric_fields.contains(&self.lfid.field_id()) {
            numeric_arrays.insert(self.lfid.field_id(), array.clone());
        }

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
    row_ids: Treemap,
}

impl RowIdScanCollector {
    fn extend_from_array(&mut self, row_ids: &UInt64Array) {
        for idx in 0..row_ids.len() {
            self.row_ids.add(row_ids.value(idx));
        }
    }

    fn extend_from_slice(&mut self, row_ids: &UInt64Array, start: usize, len: usize) {
        if len == 0 {
            return;
        }
        let end = (start + len).min(row_ids.len());
        for idx in start..end {
            self.row_ids.add(row_ids.value(idx));
        }
    }

    fn into_inner(self) -> Treemap {
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
            Some(FxHashMap::default())
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
            let mut map: NumericArrayMap = FxHashMap::default();
            for (lfid, array) in unique_lfids.iter().zip(batch.columns().iter()) {
                let fid = lfid.field_id();
                if numeric_fields.contains(&fid) {
                    map.insert(fid, array.clone());
                }
            }
            Some(map)
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
                    ScalarExpr::Cast { .. } if !computed_expr_requires_numeric(&info.expr) => {
                        synthesize_computed_literal_array(
                            info,
                            out_schema.field(idx).data_type(),
                            batch_len,
                        )?
                    }
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

fn build_projection_literals(
    projection_evals: &[ProjectionEval],
    out_schema: &Arc<Schema>,
) -> Vec<ProjectionLiteral<FieldId>> {
    projection_evals
        .iter()
        .enumerate()
        .map(|(idx, eval)| match eval {
            ProjectionEval::Column(_) => ProjectionLiteral::Column {
                data_type: out_schema.field(idx).data_type().clone(),
            },
            ProjectionEval::Computed(info) => ProjectionLiteral::Computed {
                info: info.clone(),
                data_type: out_schema.field(idx).data_type().clone(),
            },
        })
        .collect()
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
