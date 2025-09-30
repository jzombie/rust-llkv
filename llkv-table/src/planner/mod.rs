pub mod plan_graph;

use std::cmp;
use std::ops::Bound;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, OffsetSizeTrait, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};

use llkv_column_map::ScanBuilder;
use llkv_column_map::llkv_for_each_arrow_numeric;
use llkv_column_map::scan::{FilterPrimitive, FilterRun, dense_row_runs};
use llkv_column_map::store::GatherNullPolicy;
use llkv_column_map::store::scan::ScanOptions;
use llkv_column_map::types::{LogicalFieldId, Namespace};
use llkv_expr::literal::{FromLiteral, Literal};
use llkv_expr::typed_predicate::{
    PredicateValue, build_fixed_width_predicate, build_var_width_predicate,
};
use llkv_expr::{BinaryOp, CompareOp, Expr, Filter, Operator, ScalarExpr};
use llkv_result::{Error, Result as LlkvResult};
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;

use llkv_storage::pager::Pager;

use crate::constants::STREAM_BATCH_ROWS;
use crate::scalar_eval::{NumericArrayMap, NumericKernels};
use crate::table::{ScanProjection, ScanStreamOptions, Table};
use crate::types::FieldId;

use self::plan_graph::{
    PlanEdge, PlanExpression, PlanField, PlanGraph, PlanGraphBuilder, PlanGraphError, PlanNode,
    PlanNodeId, PlanOperator,
};

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

struct PlannedScan<'expr> {
    projections: Vec<ScanProjection>,
    filter_expr: Expr<'expr, FieldId>,
    options: ScanStreamOptions,
    plan_graph: PlanGraph,
}

pub(crate) struct TableExecutor<'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table: &'a Table<P>,
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
        options: ScanStreamOptions,
        on_batch: F,
    ) -> LlkvResult<()>
    where
        F: FnMut(RecordBatch),
    {
        let plan = self.plan_scan(projections, filter_expr, options)?;
        TableExecutor::new(self.table).execute(plan, on_batch)
    }

    fn plan_scan<'expr>(
        &self,
        projections: &[ScanProjection],
        filter_expr: &Expr<'expr, FieldId>,
        options: ScanStreamOptions,
    ) -> LlkvResult<PlannedScan<'expr>> {
        if projections.is_empty() {
            return Err(Error::InvalidArgumentError(
                "scan_stream requires at least one projection".into(),
            ));
        }

        let projections_vec = projections.to_vec();
        let filter_clone = filter_expr.clone();
        let plan_graph = self.build_plan_graph(&projections_vec, &filter_clone, options)?;

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
        options: ScanStreamOptions,
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
        Self { table }
    }

    fn execute<'expr, F>(&self, plan: PlannedScan<'expr>, mut on_batch: F) -> LlkvResult<()>
    where
        F: FnMut(RecordBatch),
    {
        let PlannedScan {
            projections,
            filter_expr,
            options,
            plan_graph: _plan_graph,
        } = plan;

        if self.try_single_column_direct_scan(&projections, &filter_expr, options, &mut on_batch)? {
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
                    if fields_set.is_empty() {
                        return Err(Error::InvalidArgumentError(
                            "Computed projection must reference at least one column".into(),
                        ));
                    }

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
                        schema_fields.push(Field::new(info.alias.clone(), DataType::Float64, true));
                    }
                }
            }
        }
        let out_schema = Arc::new(Schema::new(schema_fields));

        let mut all_rows_cache: FxHashMap<FieldId, Vec<u64>> = FxHashMap::default();
        let row_ids = self.collect_row_ids_for_expr(&filter_expr, &mut all_rows_cache)?;
        if row_ids.is_empty() {
            return Ok(());
        }

        let null_policy = if options.include_nulls {
            GatherNullPolicy::IncludeNulls
        } else {
            GatherNullPolicy::DropNulls
        };

        let mut gather_ctx = self.table.store().prepare_gather_context(&unique_lfids)?;

        let requires_numeric = projection_evals.iter().enumerate().any(|(idx, eval)| {
            matches!(eval, ProjectionEval::Computed(_)) && passthrough_fields[idx].is_none()
        });

        let mut offset = 0usize;
        while offset < row_ids.len() {
            let end = cmp::min(offset + STREAM_BATCH_ROWS, row_ids.len());
            let window = &row_ids[offset..end];

            let batch = self.table.store().gather_rows_with_reusable_context(
                &mut gather_ctx,
                window,
                null_policy,
            )?;

            if batch.num_rows() == 0 {
                offset = end;
                continue;
            }

            let unique_arrays = batch.columns();
            let numeric_arrays: Option<NumericArrayMap> = if requires_numeric {
                Some(NumericKernels::prepare_numeric_arrays(
                    &unique_lfids,
                    unique_arrays,
                    &numeric_fields,
                )?)
            } else {
                None
            };

            let mut columns: Vec<ArrayRef> = Vec::with_capacity(projection_evals.len());
            for (idx, eval) in projection_evals.iter().enumerate() {
                match eval {
                    ProjectionEval::Column(info) => {
                        let arr_idx = *unique_index
                            .get(&info.logical_field_id)
                            .expect("logical field id missing from index");
                        columns.push(Arc::clone(&unique_arrays[arr_idx]));
                    }
                    ProjectionEval::Computed(info) => {
                        if let Some(fid) = passthrough_fields[idx] {
                            let lfid = LogicalFieldId::for_user(self.table.table_id(), fid);
                            let arr_idx = *unique_index
                                .get(&lfid)
                                .expect("passthrough field missing from index");
                            columns.push(Arc::clone(&unique_arrays[arr_idx]));
                        } else {
                            let numeric_arrays = numeric_arrays
                                .as_ref()
                                .expect("numeric arrays should exist for computed projection");
                            let array = NumericKernels::evaluate_batch(
                                &info.expr,
                                batch.num_rows(),
                                numeric_arrays,
                            )?;
                            columns.push(array);
                        }
                    }
                }
            }

            let output_batch = RecordBatch::try_new(out_schema.clone(), columns)?;
            on_batch(output_batch);
            offset = end;
        }

        Ok(())
    }

    fn try_single_column_direct_scan<'expr, F>(
        &self,
        projections: &[ScanProjection],
        filter_expr: &Expr<'expr, FieldId>,
        options: ScanStreamOptions,
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

                // Fallback to existing iterative intersection for mixed expressions.
                let mut iter = children.iter();
                let mut acc =
                    self.collect_row_ids_for_expr(iter.next().unwrap(), all_rows_cache)?;
                for child in iter {
                    let next_ids = self.collect_row_ids_for_expr(child, all_rows_cache)?;
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
                    let next_ids = self.collect_row_ids_for_expr(child, all_rows_cache)?;
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
                let matched = self.collect_row_ids_for_expr(inner, all_rows_cache)?;
                Ok(difference_sorted(domain, matched))
            }
        }
    }

    fn collect_row_ids_for_filter(&self, filter: &Filter<'_, FieldId>) -> LlkvResult<Vec<u64>> {
        let filter_lfid = LogicalFieldId::for_user(self.table.table_id(), filter.field_id);
        let dtype = self.table.store().data_type(filter_lfid)?;

        if let Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        } = &filter.op
            && let Some(runs) = dense_row_runs(self.table.store(), filter_lfid)?
        {
            return Ok(expand_filter_runs(&runs));
        }

        let row_ids = match &dtype {
            DataType::Utf8 => self.collect_matching_row_ids_string::<i32>(filter_lfid, &filter.op),
            DataType::LargeUtf8 => {
                self.collect_matching_row_ids_string::<i64>(filter_lfid, &filter.op)
            }
            other => llkv_column_map::with_integer_arrow_type!(
                other.clone(),
                |ArrowTy| self.collect_matching_row_ids::<ArrowTy>(filter_lfid, &filter.op),
                Err(Error::Internal(format!(
                    "Filtering on type {:?} is not supported",
                    other
                ))),
            ),
        }?;

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
        let lfids: Vec<LogicalFieldId> = fields
            .iter()
            .map(|fid| LogicalFieldId::for_user(self.table.table_id(), *fid))
            .collect();
        let mut result = Vec::new();
        let numeric_fields: FxHashSet<FieldId> = fields.iter().copied().collect();
        let mut start = 0usize;
        while start < row_ids.len() {
            let end = cmp::min(start + STREAM_BATCH_ROWS, row_ids.len());
            let window = &row_ids[start..end];
            let gathered =
                self.table
                    .store()
                    .gather_rows(&lfids, window, GatherNullPolicy::IncludeNulls)?;
            if gathered.num_rows() == 0 {
                start = end;
                continue;
            }
            let arrays = gathered.columns();
            let numeric_arrays: NumericArrayMap =
                NumericKernels::prepare_numeric_arrays(&lfids, arrays, &numeric_fields)?;
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
        T: FilterPrimitive,
        T::Native: FromLiteral + Copy + PredicateValue,
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

    fn collect_matching_row_ids_fused<T>(
        &self,
        field_id: LogicalFieldId,
        ops: &[Operator<'_>],
    ) -> LlkvResult<Vec<u64>>
    where
        T: FilterPrimitive,
        T::Native: FromLiteral + Copy + PredicateValue,
    {
        let mut preds: Vec<llkv_expr::typed_predicate::Predicate<T::Native>> =
            Vec::with_capacity(ops.len());
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
        Operator::StartsWith(prefix) => format!("STARTS WITH \"{}\"", escape_string(prefix)),
        Operator::EndsWith(suffix) => format!("ENDS WITH \"{}\"", escape_string(suffix)),
        Operator::Contains(fragment) => format!("CONTAINS \"{}\"", escape_string(fragment)),
    }
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
    }
}

fn format_binary_op(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Subtract => "-",
        BinaryOp::Multiply => "*",
        BinaryOp::Divide => "/",
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

fn normalize_row_ids(mut row_ids: Vec<u64>) -> Vec<u64> {
    row_ids.sort_unstable();
    row_ids.dedup();
    row_ids
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
