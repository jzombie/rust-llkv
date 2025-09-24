# Is it a DataFrame?

Short answer: yes. You already have the core of a lean, effective
dataframe.

## What you have that maps directly to a DF engine

* Columnar store backed by Arrow arrays.
* Untyped query literals with late coercion.
* Predicate building and typed dispatch.
* Row-id creation once, reused for projections.
* Streaming projection via `scan_stream` (no concat).
* Kernel interop for per-batch math, min/max, etc.
* A system catalog and stable field ids.

## What is missing to feel like a dataframe

1. Multi-column streaming

   * Align chunk/gather for N projected columns without materializing.
   * Shared row-id windows across columns.

2. Basic relational ops

   * Projection: select columns, rename, simple compute columns.
   * Filter chaining: `where().where()` composing to one predicate.
   * Aggregates: SUM/MIN/MAX/COUNT/AVG with and without GROUP BY.
   * Order by and limit (can still be streamed in windows).

3. Expression eval

   * Tiny expression IR over Arrow kernels (arithmetic, casts, boolean,
     string contains/starts/ends). Keep it minimal; no heavy planner.

4. Null and type coverage

   * Consistent null semantics across ops.
   * Extend literal casting coverage and error messages.

5. Pruning and perf

   * Per-chunk min/max stats to skip non-matching chunks.
   * Hash-set path for large IN lists.
   * Parallel window execution with local partials then merge.

6. API surface

   * A thin DF facade that builds a small logical plan:

     * `DataFrame::from(table)`
     * `.select(cols | exprs)`
     * `.filter(expr)`
     * `.aggregate(group_keys, aggs)`
     * `.collect()` or `.scan_into(sink)` for streaming.
   * Keep it chainable and lazy; execute in `collect/scan_into`.

7. Testing and benchmarks

   * Microbench: selective filters, IN, range with pruning.
   * End-to-end: streamed aggregates vs Polars/DataFusion on same data.

## A minimal shape (signatures only)

* `struct DataFrame { table: Arc<Table>, plan: Plan }`
* `fn select(self, exprs: impl Into<Vec<Expr>>)` -> `Self`
* `fn filter(self, pred: Expr)` -> `Self`
* `fn aggregate(self, keys: Vec<Expr>, aggs: Vec<Agg>)` -> `Self`
* `fn collect(self) -> Result<RecordBatch, Error>`
* `fn scan_into(self, f: impl FnMut(&RecordBatch)) -> Result<(), Error>`

## Why this can be fast

* Zero-copy Arrow everywhere.
* Late casting of literals avoids array creation.
* Single predicate to row-ids, then windowed gathers.
* Kernel-per-batch computation with stream reductions.
* Tiny planner and execution runtime.

If you add multi-column streaming, chunk pruning, and a tiny lazy plan,
you will have a very credible, lightweight dataframe that is competitive
for single-table analytics and selective scans.
