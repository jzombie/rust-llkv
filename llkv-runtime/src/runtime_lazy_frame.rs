use std::sync::Arc;

use llkv_executor::{ExecutorRowBatch, SelectExecution};
use llkv_expr::expr::Expr as LlkvExpr;
use llkv_plan::{AggregateExpr, PlanValue, SelectPlan, SelectProjection};
use llkv_result::Result;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use crate::{RuntimeContext, canonical_table_name};

/// Lazily built logical plan (thin wrapper over SelectPlan).
pub struct RuntimeLazyFrame<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    context: Arc<RuntimeContext<P>>,
    plan: SelectPlan,
}

impl<P> RuntimeLazyFrame<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn scan(context: Arc<RuntimeContext<P>>, table: &str) -> Result<Self> {
        let (display, canonical) = canonical_table_name(table)?;
        context.lookup_table(&canonical)?;
        Ok(Self {
            context,
            plan: SelectPlan::new(display),
        })
    }

    pub fn filter(mut self, predicate: LlkvExpr<'static, String>) -> Self {
        self.plan.filter = Some(predicate);
        self
    }

    pub fn select_all(mut self) -> Self {
        self.plan.projections = vec![SelectProjection::AllColumns];
        self
    }

    pub fn select_columns<S>(mut self, columns: impl IntoIterator<Item = S>) -> Self
    where
        S: AsRef<str>,
    {
        self.plan.projections = columns
            .into_iter()
            .map(|name| SelectProjection::Column {
                name: name.as_ref().to_string(),
                alias: None,
            })
            .collect();
        self
    }

    pub fn select(mut self, projections: Vec<SelectProjection>) -> Self {
        self.plan.projections = projections;
        self
    }

    pub fn aggregate(mut self, aggregates: Vec<AggregateExpr>) -> Self {
        self.plan.aggregates = aggregates;
        self
    }

    pub fn collect(self) -> Result<SelectExecution<P>> {
        let snapshot = self.context.default_snapshot();
        self.context.execute_select(self.plan, snapshot)
    }

    pub fn collect_rows(self) -> Result<ExecutorRowBatch> {
        let snapshot = self.context.default_snapshot();
        let execution = self.context.execute_select(self.plan, snapshot)?;
        execution.collect_rows()
    }

    pub fn collect_rows_vec(self) -> Result<Vec<Vec<PlanValue>>> {
        Ok(self.collect_rows()?.rows)
    }
}
