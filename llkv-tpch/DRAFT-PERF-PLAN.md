- Stop cloning whole parent tables for uniqueness/FK checks. Even after the UniqueKey refactor we still call scan_multi_column_values which materializes every parent row into Vec<Vec<PlanValue>>. That allocates ~150–200 MB per FK/unique constraint at scale 1 and lives until the batch finishes. Short-term fix: add an index-aware fetch path in RuntimeContext::scan_multi_column_values (or a sibling helper) that, when a UNIQUE/PK index exists, probes the index per candidate key instead of scanning/allocating the entire table. That alone cuts both runtime and RSS.

- Clear per-constraint caches as soon as a table finishes loading. ConstraintService::validate_insert_foreign_keys now builds FxHashSet<UniqueKey> per constraint but we hold them until the batch ends. When load_table_with_rows finishes a table, explicitly drop the cache (e.g., new ConstraintService::clear_fk_cache(table_id)), so the parent key sets from previous tables stop contributing to swap.

- Loader now resolves its batch size from the column-store write hints (and exposes `--batch-size` for overrides), so tweak the hint math instead of hard-coding constants when future tuning is required.

- Defer FK/unique enforcement for bulk loads. Add a “bulk load mode” flag that records candidate violations but skips per-row constraint checks until the end of the table (or even per transaction). Once the data is loaded, run a streaming validator that scans once per constraint. This keeps correctness but turns the hot inserts into linear appends with minimal extra memory.

- Profile scan_multi_column_values with heaptrack/Instruments. The flamegraph still shows ~30% there. Instrument it with tracing::trace_span! + counters so you can confirm how many rows each constraint is reading and use that data to prioritize which parent tables need indexes or special-case logic.

- Ideally, we should also track how often the pager is being paged (and subsequently, how often existing chunks are overwritten, etc.) during these large imports to help ease storage thrashing and potential waste due to append-only nature.

