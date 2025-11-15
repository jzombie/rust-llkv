[ ] Stop cloning whole parent tables for uniqueness/FK checks. Even after the UniqueKey refactor we still call scan_multi_column_values which materializes every parent row into Vec<Vec<PlanValue>>. That allocates ~150–200 MB per FK/unique constraint at scale 1 and lives until the batch finishes. Short-term fix: add an index-aware fetch path in RuntimeContext::scan_multi_column_values (or a sibling helper) that, when a UNIQUE/PK index exists, probes the index per candidate key instead of scanning/allocating the entire table. That alone cuts both runtime and RSS.

[ ] Clear per-constraint caches as soon as a table finishes loading. ConstraintService::validate_insert_foreign_keys now builds FxHashSet<UniqueKey> per constraint but we hold them until the batch ends. When load_table_with_rows finishes a table, explicitly drop the cache (e.g., new ConstraintService::clear_fk_cache(table_id)), so the parent key sets from previous tables stop contributing to swap.

[x] Loader now resolves its batch size from the column-store write hints (and exposes `--batch-size` for overrides). Next step is to tweak the hint math instead of hard-coding constants when future tuning is required.

[ ] Defer FK/unique enforcement for bulk loads. Add a “bulk load mode” flag that records candidate violations but skips per-row constraint checks until the end of the table (or even per transaction). Once the data is loaded, run a streaming validator that scans once per constraint. This keeps correctness but turns the hot inserts into linear appends with minimal extra memory.

[ ] Profile scan_multi_column_values with heaptrack/Instruments. The flamegraph still shows ~30% there. Instrument it with tracing::trace_span! + counters so you can confirm how many rows each constraint is reading and use that data to prioritize which parent tables need indexes or special-case logic.

[x] Track pager churn and overwrite rates during large imports. Implemented via the pager diagnostics module (table spans plus totals) and now visible in the TPCH loader. Follow-up: reduce alloc/put batch counts by pooling key reservations so we are not making 1:1 alloc calls.

[ ] Capture loader timing per stage: break load_table_with_rows into phases (row formatting, FK cache validation, INSERT execution, flush) and emit tracing spans or metrics. If the extra minute sits in SQL planning vs. storage writes, we’ll know.

[ ] Tune hint math based on telemetry: once we see actual chunk sizes and pager churn, adjust ColumnStoreWriteHints::from_config (e.g., reduce recommended_insert_batch_rows only for varwidth-heavy tables, or grow TARGET_CHUNK_BYTES when the pager has headroom) instead of reverting to hard-coded defaults.

[ ] Explore FK cache lifetime: now that batch sizes dropped, we can revisit dropping caches even earlier (per chunk) or parallelizing constraint checks; telemetry will show whether FK validation still dominates.

[ ] Optional follow-up: add a “fast path” for append-only tables with no FK/unique checks—if telemetry says constraint work is negligible, skip this; otherwise, this could reclaim the lost minute without raising memory.

[ ] Batch pager key allocations. Current diagnostics show alloc_batches == physical_allocs because upstream layers call `alloc_many(1)` per chunk. Introduce small key pools (per ColumnWriter or reservation helper) so we request dozens/hundreds of keys at a time, which should shrink alloc batch counts and pager contention.


=====

## Bulk ingest parity plan (target: ≤15 s SF1 on laptop)

[ ] **Arrow-native batch ingest.** Replace the loader's `Vec<Vec<PlanValue>>` batches with Arrow `RecordBatch` producers and add `InsertSource::ArrowBatch` so the runtime receives typed column data directly. Executor must feed those batches through the existing `TableInserter` path so foreign keys, uniques, and triggers still fire.

[ ] **Column-store slice appends.** Extend `ColumnStore`/`ColumnWriter` with an `append_batch` API that ingests entire Arrow arrays (or contiguous slices) per column. This removes the per-value descriptor lookups visible in Instruments and is required to make Arrow-native batches pay off.

[ ] **Constraint-friendly bulk mode.** Keep FK/unique enforcement correct while avoiding per-row scans by (a) logging candidate keys during insert, (b) running streaming validators once the batch completes, and (c) integrating with the existing FK cache lifetime controls. This combines with the deferred-enforcement checkbox above.

[ ] **FK cache instrumentation + tuning.** Add tracing around `enable_fk_cache_for_table`/`clear_fk_cache_for_table` to confirm cache reuse. If caches churn every batch, introduce per-table reuse windows or chunk-level invalidation.

[ ] **Parallel table loading.** Allow independent tables (REGION/NATION/SUPPLIER/CUSTOMER) to load concurrently once the loader emits Arrow batches. Rayon-style task fan-out should be enough; coordination layer must keep schema install + FK ordering deterministic.

[ ] **I/O + tpchgen profiling.** Run `fs_usage`/`iostat` + tpchgen timing to guarantee we are CPU-bound. If file generation dominates, precompute `.tbl` files or pipe from `tpchgen --threads` so we benchmark the storage stack rather than the generator.

[ ] **Benchmark harness + regression targets.** Add a `cargo bench -p llkv-tpch --bench ingest` (or a Criterion harness) that loads SF0.1, SF1, and SF10, recording wall-clock + rows/sec. Track these against DuckDB COPY numbers so we know when we close the gap.

[ ] **Optional fast path:** once Arrow batches land, expose an “unsafe” bulk-ingest toggle for tables without FK/unique constraints. This lets append-only analytical tables skip even the deferred checks when the user opts in.
