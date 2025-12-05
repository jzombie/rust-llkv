# TPC-H Planner Performance Tasks

- Plan representation: extend `LogicalPlan` with joins (type/keys/filters), aggregates (group keys/outputs), and compounds; resolve all columns to IDs and run every SELECT through it.
- Rewrite + pushdown: add predicate pushdown and projection pruning across joins/aggregates; exploit LIMIT/OFFSET/ORDER BY to favor sorted/index scans and early stopping.
- Stats + costs: collect row counts, min/max, null counts, index presence (plus key histograms for TPC-H); use them for scan/path selection, join strategy, and a greedy join order.
- Physical planning overhaul: choose index/sorted/range scans, join algorithms and ordering from the logical tree; push projections/filters to scans; reuse order to avoid extra sorts.
- Executor de-hardening: remove executor-side ad hoc translation for joins/aggregates/compounds; enforce SQL → logical plan → rewrites → physical plan → executor as the single path.
- Early TPC-H wins: gather stats on TPC-H tables; implement join-key resolution and join-order heuristic; push date/qty/price filters to scans; prefer sort indexes/merge joins for ORDER BY/LIMIT.

## Zero-Copy Plan (column-map → join)
1) Replace row materialization with row-id streams
- Redesign llkv-join to emit joined row-id pairs (or dense ranges) instead of gathered RecordBatches.
- Keep build/probe batches resident; output a JoinResult stream of (left_batch_idx, row_idx, right_batch_idx, row_idx|None) tuples. Semi/anti emit left indices only. No Arrow take/concat.

2) Projection-as-view over row-ids
- Add a projection stage that maps row-id tuples to zero-offset column slices. Use slice for contiguous runs; fall back to a single UInt32Array index for take only when unavoidable.

3) Column-map gather fast-paths
- Extend gather_indices* to detect monotonic/contiguous selections and return slice views; keep current path as fallback. For varlen columns, prefer offset-preserving slices via zero_offset instead of take.

4) Storage row-stream reuse
- Expose a view-only gather in column-map that returns Arc<ArrayData> slices from chunk caches when ranges align; avoid re-encoding buffers. Keep chunk buffers cached so join/projection can borrow, not clone.

5) Hash join adaptation
- Build phase: keep build batches; hash table stores (batch_idx, row_idx).
- Probe phase: emit vectors of index pairs and accumulate into ranges when keys repeat for sliceability.
- Output schema remains logical; physical columns are produced later by the projection stage using stored batches.

6) Cross join handling
- Emit dense Cartesian row-id ranges (left row repeats across right run) so projection forms slices without per-row take.

7) Executor wiring
- Thread join outputs as JoinRowSet + batch stash through the executor; defer RecordBatch materialization to final projection/aggregate. Apply post-join filters as bitmaps over row-id tuples before projection.

8) Memory safety
- Enforce zero-offset slices on zero-copy paths; prevent buffer over-retention by sharing Arc<ArrayData> from cached batches.

Scope/impact
- Touches: llkv-join (hash + cartesian), gather.rs, storage chunk cache, new projection/view adapter in executor or column-map, planner/executor interface for JoinRowSet.
- Behavior change: join no longer materializes intermediate batches; downstream consumers must accept row-id streams.

## Concerns

- Table & column resolvers seem duplicated between llkv-table, llkv-executor, llkv-plan
