
# LLKV Performance Execution Plan (Vectorized, Fused, Zero-Copy)

## Non-Negotiable Principles
- End-to-end vectorized pipelines: scans → filters → joins → projections → aggregates without per-operator materialization.
- Late materialization: carry row-id sets through joins/filters; materialize columns only at projection/aggregate boundaries.
- Zero-copy first: slices/offset-preserving views; avoid Arrow `take`/`concat` unless unavoidable; enforce zero-offset arrays.
- Order-aware: track and preserve ordering; prefer merge paths when sorted; avoid re-sorts when LIMIT/OFFSET present.
- Cost-based: use real stats (row counts, NDVs, min/max, histograms, index/sort flags) to pick join order, algorithm, and scan path.
- Parallel and spill-safe: partitioned hash/merge joins and aggregates with bounded memory and columnar spill.

## Step-by-Step Plan (do in order, no rework)
1) **Late-materialized joins (zero-copy core)**
	- Change `llkv-join` to output row-id sets (batch_idx, row_idx[, right]) instead of `RecordBatch`es.
	- Stash build/probe batches; no `take`/`concat` in join hot path.
	- Add projection/view layer that maps row-id runs to zero-offset slices; fall back to a single index array only when non-contiguous.
	- Apply post-join filters as bitmap operations on row-id sets before projection.

2) **Column-map zero-copy gather and buffer reuse**
	- Ensure `gather_indices*` and varlen gathers detect contiguous/monotonic regions and return slices; keep offset-preserving slices for Utf8/Binary.
	- Keep chunk buffers in `Arc<ArrayData>` caches; expose view-only gathers for reuse without copies.

3) **Executor integration + multi-table SELECT**
	- Wire executor/planner to consume row-id join outputs and late-materialize projections/aggregates.
	- Preserve MVCC row filters and ensure SELECT semantics/SLT stay green.

4) **Parallel, partitioned hash join with spill**
	- Add radix/range partitioning; parallel build/probe; adaptive batch sizing.
	- Implement bounded-memory hash table with columnar spill and rehash; keep join outputs as row-id sets.

5) **Ordering-aware pipeline + merge join**
	- Propagate order metadata through scans/filters/projections/joins.
	- Use existing sort indexes to satisfy ORDER BY/LIMIT without resort.
	- Add merge join when both sides ordered on join keys; keep late materialization.

6) **Stats and cost-based planning**
	- Collect and persist row counts, NDVs, min/max, null counts, histograms, index/sort flags per column.
	- Use costs to pick join order, join algo (hash vs merge), and scan path (sorted/index/range/full); prefer order-preserving plans when they save sorts.

7) **Expression fusion + bitmap filters**
	- Fuse scalar expressions and predicates into vectorized kernels; keep bool masks as bitmaps.
	- Specialize hot shapes (arith + compare, COALESCE/CASE) for SIMD types; avoid per-row branching.

8) **Aggregations: partitioned + spill-safe**
	- Two-phase (partial/final) grouped aggregation over row-id streams; partitioned hash agg with spill.
	- Maintain late materialization for non-agg columns; output zero-copy slices when possible.

9) **I/O and prefetch**
	- Add async/page prefetch for scans; tune batch sizes to cache; enable compressed page caching.
	- Avoid duplicate decompression when buffers are reused for projection.

10) **Validation and perf targets**
	 - Keep SLT and workspace tests green after each major step.
	 - Add micro/TPCH benches: require parity or better vs prior commit; target within competitive range of DuckDB/DataFusion for comparable workloads.

## Success Criteria
- Joins and projections do not allocate new buffers on hot paths when inputs are sliceable.
- Parallel hash join and agg stay bounded with spill, remain columnar.
- ORDER BY/LIMIT prefers existing order/sort indexes; merge join path active when sorted inputs exist.
- Cost-based planner chooses plans using collected stats and avoids unnecessary sorts/scans.
- Benchmarks show no regressions and trend toward DataFusion/DuckDB performance class.

## End-to-end performance blueprint (no future refactor passes)

- Vectorized, fused pipelines: scans → filters → joins → projections without intermediate materialization; fuse scalar expressions, comparisons, and casts into batch kernels; prefer SIMD-aware kernels for int/float/utf8 and keep bool masks as bitmaps.
- Parallel + partitioned operators: parallel scan with chunk prefetch; partitioned hash join (radix or range) with build/probe in parallel and adaptive batch sizing; streaming/partitioned group-by with two-phase aggregation; spill paths that stay columnar and zero-copy slice-friendly.
- Cost-based planning with real stats: collect row counts, NDVs, min/max, null counts, histograms, sortedness flags, and index metadata; pick join order, join type (hash vs merge), and scan path (sorted/range/index/full) via costs; exploit existing ordering to skip sorts.
- Ordering-aware execution: choose merge join when inputs sorted; propagate order metadata through projections/filters; avoid resorting when LIMIT/OFFSET present; keep sort indexes hot for ORDER BY/LIMIT.
- Late materialization everywhere: carry row-id sets through joins/filters; only materialize projected columns at the edge or for aggregates that truly need data; apply post-join filters on row-id bitmaps before projection.
- Projection/scan zero-copy: push slices down; varlen columns keep offset-preserving slices; reuse buffer pools/arenas; enforce zero-offset slices to avoid kernel slow paths.
- Memory + spill discipline: bounded hash tables with partition + spill; reuse buffers; compact bitmaps; avoid per-batch allocs in hot loops; keep caches in `Arc<ArrayData>` to enable zero-copy sharing across operators.
- Expression engine: specialize common shapes (arith + comparisons, COALESCE/CASE) and avoid per-row branching; optional codegen/JIT tier for hot pipelines if profiling shows a gap versus DuckDB/DataFusion.
- I/O throughput: async/page prefetch on scans; wide batches tuned to cache; compressed page caching; avoid redundant decompression on reuse.

Implementation order to avoid rework
1) Make joins late-materialized and zero-copy (row-id streams + slice-first projection) and keep existing semantics green.
2) Add parallel + partitioned hash join with spill; keep zero-copy projection on top of row-id outputs.
3) Wire ordering metadata and merge-join path; exploit existing sort indexes for ORDER BY/LIMIT; keep order through pipelines.
4) Introduce stats collection + cost-based join ordering/scan selection; use stats in planner to pick algorithms.
5) Fuse expression kernels and bitmap filters end-to-end; ensure projection/filter stays vectorized on row-id sets.
6) Add buffer reuse pools and enforce zero-offset slices across column-map/storage.
7) Add spill-safe, partitioned group-by/aggregate and keep late materialization for non-agg columns.
8) Profile; add specialization/codegen only if gaps remain vs DuckDB/DataFusion baselines.

## Concerns

- Table & column resolvers seem duplicated between llkv-table, llkv-executor, llkv-plan
- HashMap usage instead of FxHashMap
- llkv-join's vectorized vs. non-vectorized join paths
- Previous compile paths, are they still used at all?
- Perf overhead from potentially using TreeMap in the wrong places
