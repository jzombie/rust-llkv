pub mod plan_graph;

use std::cell::RefCell;
use std::cmp::{self, Ordering};
use std::convert::TryFrom;
use std::ops::Bound;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float64Array, Int64Array, OffsetSizeTrait, RecordBatch, StringArray,
    UInt64Array, new_null_array,
};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, Schema};

use llkv_column_map::ScanBuilder;
use llkv_column_map::parallel;
use llkv_column_map::scan::{FilterPrimitive, FilterRun, dense_row_runs};
use llkv_column_map::store::GatherNullPolicy;
use llkv_column_map::store::scan::ScanOptions;
use llkv_column_map::types::{LogicalFieldId, Namespace};
use llkv_column_map::{llkv_for_each_arrow_boolean, llkv_for_each_arrow_numeric};
use llkv_expr::literal::{FromLiteral, Literal};
use llkv_expr::typed_predicate::{
    PredicateValue, build_bool_predicate, build_fixed_width_predicate, build_var_width_predicate,
};
use llkv_expr::{BinaryOp, CompareOp, Expr, Filter, Operator, ScalarExpr};
use llkv_result::{Error, Result as LlkvResult};
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;

use llkv_storage::pager::Pager;
use rayon::prelude::*;

use crate::constants::STREAM_BATCH_ROWS;
use crate::scalar_eval::{NumericArrayMap, NumericKernels};
use crate::table::{
    ScanOrderDirection, ScanOrderSpec, ScanOrderTransform, ScanProjection, ScanStreamOptions, Table,
};
use crate::types::{FieldId, ROW_ID_FIELD_ID};

use self::plan_graph::{
    PlanEdge, PlanExpression, PlanField, PlanGraph, PlanGraphBuilder, PlanGraphError, PlanNode,
    PlanNodeId, PlanOperator,
};

// TODO: Refactor into executor and potentially migrate any remnants to `llkv-plan`

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

#[derive(Clone)]
struct ColumnProjectionInfo {
    logical_field_id: LogicalFieldId,
    data_type: DataType,
    output_name: String,
}

#[derive(Clone)]
struct ComputedProjectionInfo {
    expr: ScalarExpr<FieldId>,
    alias: String,
}

#[derive(Clone)]
enum ProjectionEval {
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
    row_id_cache: RefCell<Option<Vec<u64>>>,
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

    // TODO: Return `LlkvResult<Vec<RowId>>`
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

    fn table_row_ids(&self) -> LlkvResult<Vec<u64>> {
        if let Some(cached) = self.row_id_cache.borrow().as_ref() {
            return Ok(cached.clone());
        }
        let computed = self.compute_table_row_ids()?;
        *self.row_id_cache.borrow_mut() = Some(computed.clone());
        Ok(computed)
    }

    fn compute_table_row_ids(&self) -> LlkvResult<Vec<u64>> {
        let fields = self
            .table
            .store()
            .user_field_ids_for_table(self.table.table_id());
        if fields.is_empty() {
            return Ok(Vec::new());
        }

        let expected = self
            .table
            .store()
            .total_rows_for_table(self.table.table_id())
            .unwrap_or_default();

        let mut seen: FxHashSet<u64> = FxHashSet::default();
        let mut collected: Vec<u64> = Vec::new();

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
        Ok(collected)
    }

    fn collect_row_ids_for_rowid_filter(&self, op: &Operator<'_>) -> LlkvResult<Vec<u64>> {
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

        if self.try_single_column_direct_scan(
            &projections,
            &filter_expr,
            options.clone(),
            &mut on_batch,
        )? {
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
                            ScalarExpr::Literal(Literal::String(_)) => DataType::Utf8,
                            ScalarExpr::Binary { .. } => DataType::Float64,
                            ScalarExpr::Column(fid) => {
                                let lfid = LogicalFieldId::for_user(self.table.table_id(), *fid);
                                lfid_dtypes.get(&lfid).cloned().ok_or_else(|| {
                                    Error::Internal("missing dtype for computed column".into())
                                })?
                            }
                            ScalarExpr::Aggregate(_) => {
                                // Aggregates in computed columns return Int64
                                // TODO: Fix: This is a simplification - ideally we'd determine type from the aggregate
                                DataType::Int64
                            }
                        };
                        schema_fields.push(Field::new(info.alias.clone(), dtype, true));
                    }
                }
            }
        }
        let out_schema = Arc::new(Schema::new(schema_fields));

        let fusion_cache = PredicateFusionCache::from_expr(&filter_expr);
        let mut all_rows_cache: FxHashMap<FieldId, Vec<u64>> = FxHashMap::default();
        let mut row_ids =
            self.collect_row_ids_for_expr(&filter_expr, &fusion_cache, &mut all_rows_cache)?;
        if let Some(filter) = options.row_id_filter.as_ref() {
            row_ids = filter.filter(self.table, row_ids)?;
        }
        if row_ids.is_empty() {
            if is_trivial_filter(&filter_expr) {
                let total_rows = self.table.total_rows()?;
                let row_count = usize::try_from(total_rows).map_err(|_| {
                    Error::InvalidArgumentError("table row count exceeds supported range".into())
                })?;
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

                let batch = RecordBatch::try_new(Arc::clone(&out_schema), columns)?;
                on_batch(batch);
            }
            return Ok(());
        }

        if let Some(order_spec) = options.order {
            row_ids = self.sort_row_ids_with_order(row_ids, order_spec)?;
        }

        let null_policy = if options.include_nulls {
            GatherNullPolicy::IncludeNulls
        } else {
            GatherNullPolicy::DropNulls
        };

        let requires_numeric = projection_evals.iter().enumerate().any(|(idx, eval)| {
            matches!(eval, ProjectionEval::Computed(info)
                if passthrough_fields[idx].is_none()
                    && computed_expr_requires_numeric(&info.expr))
        });

        let table_id = self.table.table_id();
        let store = self.table.store();
        let unique_lfids = Arc::new(unique_lfids);
        let projection_evals = Arc::new(projection_evals);
        let passthrough_fields = Arc::new(passthrough_fields);
        let unique_index = Arc::new(unique_index);
        let numeric_fields = Arc::new(numeric_fields);

        let chunk_ranges: Vec<(usize, usize)> = (0..row_ids.len())
            .step_by(STREAM_BATCH_ROWS)
            .map(|start| {
                let end = cmp::min(start + STREAM_BATCH_ROWS, row_ids.len());
                (start, end)
            })
            .collect();

        let chunk_batches: Vec<LlkvResult<Option<RecordBatch>>> = parallel::with_thread_pool(
            move || {
                chunk_ranges
                    .into_par_iter()
                    .map(|(start, end)| -> LlkvResult<Option<RecordBatch>> {
                        let window = &row_ids[start..end];
                        if window.is_empty() {
                            return Ok(None);
                        }

                        let unique_lfids = Arc::clone(&unique_lfids);
                        let projection_evals = Arc::clone(&projection_evals);
                        let passthrough_fields = Arc::clone(&passthrough_fields);
                        let unique_index = Arc::clone(&unique_index);
                        let numeric_fields = Arc::clone(&numeric_fields);

                        let (batch_len, unique_arrays_owned, numeric_arrays): (
                            usize,
                            Vec<ArrayRef>,
                            Option<NumericArrayMap>,
                        ) = if unique_lfids.is_empty() {
                            let numeric_arrays = if requires_numeric {
                                Some(Default::default())
                            } else {
                                None
                            };
                            (window.len(), Vec::new(), numeric_arrays)
                        } else {
                            let mut gather_ctx =
                                match store.prepare_gather_context(unique_lfids.as_ref()) {
                                    Ok(ctx) => ctx,
                                    Err(e) => {
                                        tracing::debug!(
                                            ?unique_lfids,
                                            error = %e,
                                            "prepare_gather_context failed"
                                        );
                                        return Err(e);
                                    }
                                };
                            let batch = store.gather_rows_with_reusable_context(
                                &mut gather_ctx,
                                window,
                                null_policy,
                            )?;
                            if batch.num_rows() == 0 {
                                return Ok(None);
                            }

                            let unique_arrays_owned: Vec<ArrayRef> =
                                batch.columns().iter().map(Arc::clone).collect();
                            let numeric_arrays = if requires_numeric {
                                Some(NumericKernels::prepare_numeric_arrays(
                                    unique_lfids.as_ref(),
                                    unique_arrays_owned.as_slice(),
                                    numeric_fields.as_ref(),
                                )?)
                            } else {
                                None
                            };
                            (batch.num_rows(), unique_arrays_owned, numeric_arrays)
                        };

                        if batch_len == 0 {
                            return Ok(None);
                        }

                        let mut columns: Vec<ArrayRef> = Vec::with_capacity(projection_evals.len());
                        for (idx, eval) in projection_evals.iter().enumerate() {
                            match eval {
                                ProjectionEval::Column(info) => {
                                    let arr_idx = *unique_index
                                        .get(&info.logical_field_id)
                                        .expect("logical field id missing from index");
                                    columns.push(Arc::clone(&unique_arrays_owned[arr_idx]));
                                }
                                ProjectionEval::Computed(info) => {
                                    if let Some(fid) = passthrough_fields[idx] {
                                        let lfid = LogicalFieldId::for_user(table_id, fid);
                                        let arr_idx = *unique_index
                                            .get(&lfid)
                                            .expect("passthrough field missing from index");
                                        columns.push(Arc::clone(&unique_arrays_owned[arr_idx]));
                                    } else {
                                        let array: ArrayRef = match &info.expr {
                                            ScalarExpr::Literal(Literal::Integer(value)) => {
                                                let cast = i64::try_from(*value).map_err(|_| {
                                                    Error::InvalidArgumentError(
                                                        "integer literal exceeds 64-bit range"
                                                            .into(),
                                                    )
                                                })?;
                                                Arc::new(Int64Array::from(vec![cast; batch_len]))
                                                    as ArrayRef
                                            }
                                            ScalarExpr::Literal(Literal::Float(value)) => {
                                                Arc::new(Float64Array::from(vec![
                                                    *value;
                                                    batch_len
                                                ])) as ArrayRef
                                            }
                                            ScalarExpr::Literal(Literal::String(value)) => {
                                                Arc::new(StringArray::from(vec![
                                                    value.clone();
                                                    batch_len
                                                ])) as ArrayRef
                                            }
                                            _ => {
                                                let numeric_arrays = numeric_arrays.as_ref().expect(
                                                    "numeric arrays should exist for computed projection",
                                                );
                                                NumericKernels::evaluate_batch(
                                                    &info.expr,
                                                    batch_len,
                                                    numeric_arrays,
                                                )?
                                            }
                                        };
                                        columns.push(array);
                                    }
                                }
                            }
                        }

                        let output_batch = RecordBatch::try_new(Arc::clone(&out_schema), columns)?;
                        Ok(Some(output_batch))
                    })
                    .collect()
            },
        );

        for batch_result in chunk_batches {
            if let Some(batch) = batch_result? {
                tracing::debug!(
                    rows = batch.num_rows(),
                    columns = batch.num_columns(),
                    "TableExecutor produced batch"
                );
                on_batch(batch);
            }
        }

        Ok(())
    }

    fn try_single_column_direct_scan<'expr, F>(
        &self,
        projections: &[ScanProjection],
        filter_expr: &Expr<'expr, FieldId>,
        options: ScanStreamOptions<P>,
        on_batch: &mut F,
    ) -> LlkvResult<bool>
    where
        F: FnMut(RecordBatch),
    {
        if options.include_nulls || projections.len() != 1 {
            return Ok(false);
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
                    return Ok(false);
                }

                let dtype = self.table.store().data_type(p.logical_field_id)?;
                if matches!(dtype, DataType::Utf8 | DataType::LargeUtf8) {
                    return Ok(false);
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
                Ok(true)
            }
            ScanProjection::Computed { expr, alias } => {
                let simplified = NumericKernels::simplify(expr);
                let mut fields_set: FxHashSet<FieldId> = FxHashSet::default();
                NumericKernels::collect_fields(&simplified, &mut fields_set);
                if fields_set.len() != 1 {
                    return Ok(false);
                }

                let field_id = *fields_set.iter().next().unwrap();
                let lfid = LogicalFieldId::for_user(self.table.table_id(), field_id);
                if !is_full_range_filter(filter_expr, field_id) {
                    return Ok(false);
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
                    return Ok(true);
                }

                if let Some(affine) = NumericKernels::extract_affine(&simplified) {
                    if affine.field != field_id {
                        return Err(Error::InvalidArgumentError(
                            "affine extraction resolved to unexpected field".into(),
                        ));
                    }
                    if !is_supported_numeric(&dtype) {
                        return Ok(false);
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
                    return Ok(true);
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
                Ok(true)
            }
        }
    }

    fn collect_row_ids_for_expr(
        &self,
        expr: &Expr<'_, FieldId>,
        fusion_cache: &PredicateFusionCache,
        all_rows_cache: &mut FxHashMap<FieldId, Vec<u64>>,
    ) -> LlkvResult<Vec<u64>> {
        match expr {
            Expr::Pred(filter) => self.collect_row_ids_for_filter(filter),
            Expr::Compare { left, op, right } => {
                self.collect_row_ids_for_compare(left, *op, right, all_rows_cache)
            }
            Expr::And(children) => {
                if children.is_empty() {
                    return Err(Error::InvalidArgumentError(
                        "AND expression requires at least one predicate".into(),
                    ));
                }
                // Fast-path fusion: if this AND is composed solely of Pred nodes on the
                // same field, build typed predicates and call the storage fused runtime
                // entrypoint. This avoids multiple scans over the same column.
                let mut same_field: Option<FieldId> = None;
                let mut all_preds_same_field = true;
                for child in children.iter() {
                    if let Expr::Pred(f) = child {
                        if let Some(sf) = same_field {
                            if f.field_id != sf {
                                all_preds_same_field = false;
                                break;
                            }
                        } else {
                            same_field = Some(f.field_id);
                        }
                    } else {
                        all_preds_same_field = false;
                        break;
                    }
                }

                if all_preds_same_field {
                    let fid = same_field.expect("non-empty children checked");
                    let filter_lfid = LogicalFieldId::for_user(self.table.table_id(), fid);
                    let dtype = self.table.store().data_type(filter_lfid)?;

                    if fusion_cache.should_fuse(fid, &dtype) {
                        // Collect operator list
                        let ops: Vec<Operator<'_>> = children
                            .iter()
                            .map(|c| match c {
                                Expr::Pred(f) => f.op.clone(),
                                _ => unreachable!(),
                            })
                            .collect();

                        let row_ids = match &dtype {
                            DataType::Utf8 => {
                                self.collect_matching_row_ids_string_fused::<i32>(filter_lfid, &ops)
                            }
                            DataType::LargeUtf8 => {
                                self.collect_matching_row_ids_string_fused::<i64>(filter_lfid, &ops)
                            }
                            DataType::Boolean => {
                                self.collect_matching_row_ids_bool_fused(filter_lfid, &ops)
                            }
                            other => llkv_column_map::with_integer_arrow_type!(
                                other.clone(),
                                |ArrowTy| self
                                    .collect_matching_row_ids_fused::<ArrowTy>(filter_lfid, &ops,),
                                Err(Error::Internal(format!(
                                    "Filtering on type {:?} is not supported",
                                    other
                                ))),
                            ),
                        }?;

                        return Ok(normalize_row_ids(row_ids));
                    }
                }

                // Fallback to existing iterative intersection for mixed expressions.
                let mut iter = children.iter();
                let mut acc = self.collect_row_ids_for_expr(
                    iter.next().unwrap(),
                    fusion_cache,
                    all_rows_cache,
                )?;
                for child in iter {
                    let next_ids =
                        self.collect_row_ids_for_expr(child, fusion_cache, all_rows_cache)?;
                    if acc.is_empty() || next_ids.is_empty() {
                        return Ok(Vec::new());
                    }
                    acc = intersect_sorted(acc, next_ids);
                    if acc.is_empty() {
                        break;
                    }
                }
                Ok(acc)
            }
            Expr::Or(children) => {
                if children.is_empty() {
                    return Err(Error::InvalidArgumentError(
                        "OR expression requires at least one predicate".into(),
                    ));
                }
                let mut acc = Vec::new();
                for child in children {
                    let next_ids =
                        self.collect_row_ids_for_expr(child, fusion_cache, all_rows_cache)?;
                    if acc.is_empty() {
                        acc = next_ids;
                    } else if !next_ids.is_empty() {
                        acc = union_sorted(acc, next_ids);
                    }
                }
                Ok(acc)
            }
            Expr::Not(inner) => {
                let domain = self.collect_row_ids_domain(inner, all_rows_cache)?;
                if domain.is_empty() {
                    return Ok(Vec::new());
                }
                let matched = self.collect_row_ids_for_expr(inner, fusion_cache, all_rows_cache)?;
                Ok(difference_sorted(domain, matched))
            }
        }
    }

    fn collect_row_ids_for_filter(&self, filter: &Filter<'_, FieldId>) -> LlkvResult<Vec<u64>> {
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

    fn collect_row_ids_for_compare(
        &self,
        left: &ScalarExpr<FieldId>,
        op: CompareOp,
        right: &ScalarExpr<FieldId>,
        all_rows_cache: &mut FxHashMap<FieldId, Vec<u64>>,
    ) -> LlkvResult<Vec<u64>> {
        let mut fields = FxHashSet::default();
        NumericKernels::collect_fields(left, &mut fields);
        NumericKernels::collect_fields(right, &mut fields);
        if fields.is_empty() {
            return Err(Error::InvalidArgumentError(
                "Comparison expression must reference at least one column".into(),
            ));
        }
        let mut domain: Option<Vec<u64>> = None;
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
        let domain = domain.unwrap_or_default();
        if domain.is_empty() {
            return Ok(domain);
        }
        self.evaluate_compare_over_rows(&domain, &ordered_fields, left, op, right)
    }

    fn evaluate_compare_over_rows(
        &self,
        row_ids: &[u64],
        fields: &[FieldId],
        left: &ScalarExpr<FieldId>,
        op: CompareOp,
        right: &ScalarExpr<FieldId>,
    ) -> LlkvResult<Vec<u64>> {
        if row_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut numeric_fields: FxHashSet<FieldId> = fields.iter().copied().collect();
        let has_row_id = numeric_fields.remove(&ROW_ID_FIELD_ID);
        let lfids: Vec<LogicalFieldId> = fields
            .iter()
            .copied()
            .filter(|fid| *fid != ROW_ID_FIELD_ID)
            .map(|fid| LogicalFieldId::for_user(self.table.table_id(), fid))
            .collect();
        let mut result = Vec::new();
        let mut start = 0usize;
        while start < row_ids.len() {
            let end = cmp::min(start + STREAM_BATCH_ROWS, row_ids.len());
            let window = &row_ids[start..end];
            let mut numeric_arrays: NumericArrayMap = if lfids.is_empty() {
                NumericKernels::prepare_numeric_arrays(&[], &[], &numeric_fields)?
            } else {
                let gathered = self.table.store().gather_rows(
                    &lfids,
                    window,
                    GatherNullPolicy::IncludeNulls,
                )?;
                if gathered.num_rows() == 0 {
                    start = end;
                    continue;
                }
                let arrays = gathered.columns();
                NumericKernels::prepare_numeric_arrays(&lfids, arrays, &numeric_fields)?
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
            start = end;
        }
        Ok(result)
    }

    fn collect_row_ids_domain(
        &self,
        expr: &Expr<'_, FieldId>,
        all_rows_cache: &mut FxHashMap<FieldId, Vec<u64>>,
    ) -> LlkvResult<Vec<u64>> {
        match expr {
            Expr::Pred(filter) => {
                self.collect_all_row_ids_for_field(filter.field_id, all_rows_cache)
            }
            Expr::Compare { left, right, .. } => {
                let mut fields = FxHashSet::default();
                NumericKernels::collect_fields(left, &mut fields);
                NumericKernels::collect_fields(right, &mut fields);
                if fields.is_empty() {
                    return Err(Error::InvalidArgumentError(
                        "Comparison expression must reference at least one column".into(),
                    ));
                }
                let mut domain: Option<Vec<u64>> = None;
                let mut ordered_fields: Vec<FieldId> = fields.into_iter().collect();
                ordered_fields.sort_unstable();
                for fid in ordered_fields {
                    let rows = self.collect_all_row_ids_for_field(fid, all_rows_cache)?;
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
                Ok(domain.unwrap_or_default())
            }
            Expr::And(children) | Expr::Or(children) => {
                let mut acc = Vec::new();
                for child in children {
                    let next_ids = self.collect_row_ids_domain(child, all_rows_cache)?;
                    if acc.is_empty() {
                        acc = next_ids;
                    } else if !next_ids.is_empty() {
                        acc = union_sorted(acc, next_ids);
                    }
                }
                Ok(acc)
            }
            Expr::Not(inner) => self.collect_row_ids_domain(inner, all_rows_cache),
        }
    }

    fn sort_row_ids_with_order(
        &self,
        mut row_ids: Vec<u64>,
        order: ScanOrderSpec,
    ) -> LlkvResult<Vec<u64>> {
        if row_ids.len() <= 1 {
            return Ok(row_ids);
        }

        let lfid = LogicalFieldId::for_user(self.table.table_id(), order.field_id);
        let batch =
            self.table
                .store()
                .gather_rows(&[lfid], &row_ids, GatherNullPolicy::IncludeNulls)?;

        if batch.num_columns() != 1 || batch.num_rows() != row_ids.len() {
            return Err(Error::Internal(
                "ORDER BY gather produced unexpected column count".into(),
            ));
        }

        let ascending = matches!(order.direction, ScanOrderDirection::Ascending);

        match order.transform {
            ScanOrderTransform::IdentityInteger => {
                let array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError("ORDER BY integer expects Int64 column".into())
                    })?;
                let mut indices: Vec<(usize, u64)> = row_ids.iter().copied().enumerate().collect();
                indices.sort_by(|(ai, arid), (bi, brid)| {
                    let left = if array.is_null(*ai) {
                        None
                    } else {
                        Some(array.value(*ai))
                    };
                    let right = if array.is_null(*bi) {
                        None
                    } else {
                        Some(array.value(*bi))
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
                let array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError("ORDER BY text expects Utf8 column".into())
                    })?;
                let mut indices: Vec<(usize, u64)> = row_ids.iter().copied().enumerate().collect();
                indices.sort_by(|(ai, arid), (bi, brid)| {
                    let left = if array.is_null(*ai) {
                        None
                    } else {
                        Some(array.value(*ai))
                    };
                    let right = if array.is_null(*bi) {
                        None
                    } else {
                        Some(array.value(*bi))
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
                let array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError("ORDER BY CAST expects Utf8 column".into())
                    })?;
                let mut keys: Vec<Option<i64>> = Vec::with_capacity(row_ids.len());
                for idx in 0..row_ids.len() {
                    let key = if array.is_null(idx) {
                        None
                    } else {
                        array.value(idx).parse::<i64>().ok()
                    };
                    keys.push(key);
                }
                let mut indices: Vec<(usize, u64)> = row_ids.iter().copied().enumerate().collect();
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
        cache: &mut FxHashMap<FieldId, Vec<u64>>,
    ) -> LlkvResult<Vec<u64>> {
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
    ) -> LlkvResult<Vec<u64>>
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
    ) -> LlkvResult<Vec<u64>>
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
    ) -> LlkvResult<Vec<u64>>
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
    ) -> LlkvResult<Vec<u64>> {
        let predicate = build_bool_predicate(op).map_err(Error::predicate_build)?;
        self.table
            .store()
            .filter_row_ids::<arrow::datatypes::BooleanType>(field_id, &predicate)
    }

    fn collect_matching_row_ids_bool_fused(
        &self,
        field_id: LogicalFieldId,
        ops: &[Operator<'_>],
    ) -> LlkvResult<Vec<u64>> {
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
    ) -> LlkvResult<Vec<u64>>
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

// TODO: Return `LlkvResult<Vec<RowId>>`
pub(crate) fn collect_row_ids_for_table<'expr, P>(
    table: &Table<P>,
    filter_expr: &Expr<'expr, FieldId>,
) -> LlkvResult<Vec<u64>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let executor = TableExecutor::new(table);
    let fusion_cache = PredicateFusionCache::from_expr(filter_expr);
    let mut all_rows_cache: FxHashMap<FieldId, Vec<u64>> = FxHashMap::default();
    executor.collect_row_ids_for_expr(filter_expr, &fusion_cache, &mut all_rows_cache)
}

fn computed_expr_requires_numeric(expr: &ScalarExpr<FieldId>) -> bool {
    match expr {
        ScalarExpr::Literal(_) => false,
        ScalarExpr::Column(_) => true,
        ScalarExpr::Binary { .. } => true,
        ScalarExpr::Aggregate(_) => false, // Aggregates are computed separately
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
        ScalarExpr::Literal(Literal::String(value)) => {
            Ok(Arc::new(StringArray::from(vec![value.clone(); row_count])) as ArrayRef)
        }
        ScalarExpr::Column(_) | ScalarExpr::Binary { .. } | ScalarExpr::Aggregate(_) => {
            Ok(new_null_array(data_type, row_count))
        }
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
        Literal::String(s) => format!("\"{}\"", escape_string(s)),
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
    row_ids: Vec<u64>,
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

    fn into_inner(self) -> Vec<u64> {
        self.row_ids
    }
}

impl llkv_column_map::scan::PrimitiveVisitor for RowIdScanCollector {
    llkv_for_each_arrow_numeric!(impl_row_id_ignore_chunk);
    llkv_for_each_arrow_boolean!(impl_row_id_ignore_chunk);
}

impl llkv_column_map::scan::PrimitiveSortedVisitor for RowIdScanCollector {
    llkv_for_each_arrow_numeric!(impl_row_id_ignore_sorted_run);
    llkv_for_each_arrow_boolean!(impl_row_id_ignore_sorted_run);
}

impl llkv_column_map::scan::PrimitiveWithRowIdsVisitor for RowIdScanCollector {
    llkv_for_each_arrow_numeric!(impl_row_id_collect_chunk_with_rids);
    llkv_for_each_arrow_boolean!(impl_row_id_collect_chunk_with_rids);
}

impl llkv_column_map::scan::PrimitiveSortedWithRowIdsVisitor for RowIdScanCollector {
    llkv_for_each_arrow_numeric!(impl_row_id_collect_sorted_run_with_rids);
    llkv_for_each_arrow_boolean!(impl_row_id_collect_sorted_run_with_rids);

    fn null_run(&mut self, row_ids: &UInt64Array, start: usize, len: usize) {
        self.extend_from_slice(row_ids, start, len);
    }
}

fn normalize_row_ids(mut row_ids: Vec<u64>) -> Vec<u64> {
    row_ids.sort_unstable();
    row_ids.dedup();
    row_ids
}

fn literal_to_u64(lit: &Literal) -> LlkvResult<u64> {
    u64::from_literal(lit)
        .map_err(|err| Error::InvalidArgumentError(format!("rowid literal cast failed: {err}")))
}

fn lower_bound_index(row_ids: &[u64], bound: &Bound<Literal>) -> LlkvResult<usize> {
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

fn upper_bound_index(row_ids: &[u64], bound: &Bound<Literal>) -> LlkvResult<usize> {
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

fn filter_row_ids_by_operator(row_ids: &[u64], op: &Operator<'_>) -> LlkvResult<Vec<u64>> {
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
    }
}

fn intersect_sorted(left: Vec<u64>, right: Vec<u64>) -> Vec<u64> {
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

fn union_sorted(left: Vec<u64>, right: Vec<u64>) -> Vec<u64> {
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

fn difference_sorted(base: Vec<u64>, subtract: Vec<u64>) -> Vec<u64> {
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

fn expand_filter_runs(runs: &[FilterRun]) -> Vec<u64> {
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
