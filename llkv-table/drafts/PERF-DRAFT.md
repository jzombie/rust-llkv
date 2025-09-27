Chasing that 0.75 ms target next, here’s the roadmap I’d focus on:

Quantify what 0.75 ms means for per-row work and relate it to today’s 5.5 ms profile.
Pin down the remaining hot spots (row-id handling, predicate dispatch, aggregation) and outline concrete reductions for each.
Flag the architectural shifts—bitmask-based filtering, vectorized aggregates, row-id elimination—that get us into sub-millisecond territory.
where we stand
Current median: ~5.5 ms for 1 M rows ⇒ ~180 M rows/s.
Goal: 0.75 ms ⇒ ~1.3 B rows/s—roughly a 7× throughput jump.
Samply shows three dominant buckets:
FilterVisitor::u64_chunk_with_rids: ~20 %.
Predicate<T>::matches: ~9 %.
RawVec::reserve_and_handle (row-id pushes): ~8 %.
Aggregation in llkv_table::table::collect_rows_for_filter etc. still accounts for another ~30 % once the filter path is optimized, so we have to slim both filtering and the follow-on gather/aggregate stages.
key leaps to reach 0.75 ms
1. stop materializing row IDs in the hot path
Today every filter returns a Vec<u64> of row ids, which then drives gather_rows_with_context.
Switch to a dense, zero-copy mask (MutableBuffer/BooleanArray) or reuse the row locator’s contiguous windows. Downstream consumers iterate over the mask without vector pushes.
Expected impact: remove the 20 % filter loop cost + 8 % allocator time (no push/reserve) and cut gather work substantially because we no longer expand sparse row ids back into contiguous spans.
Estimate: ~30–35 % gain (5.5 ms → ~3.5 ms).
2. fuse predicate evaluation into vectorized kernels
Precompile simple predicates (equals, ranges) into SIMD-friendly closures that operate on whole chunks with packed_simd, std::arch, or Arrow’s compute kernels (arrow::compute::eq_float64 etc.) applied directly to array slices.
For In lists, pre-sort tiny arrays and use binary_search, or build a bitmap for high-selectivity cases.
Combined with bitmask output, you get branch-free operations that can leverage CPU vector units rather than scalar loops.
Estimate: another ~40–50 % off (3.5 ms → ~2.0 ms).
3. run aggregation on columnar buffers
Once we keep filtering columnar, feed the surviving slices straight into the reducer (sum/count) using Arrow’s compute kernels or hand-written SIMD loops, avoiding any gather+Materialize-to-RecordBatch step.
For numeric aggregates, you can accumulate directly over the filtered value buffer using the mask delivered in step 1.
Estimate: 2.0 ms → ~1.1 ms.
4. pipeline / parallel tweaks (optional but helpful)
Chunk the scan into larger contiguous spans (e.g., 128 K rows) to reduce per-batch overhead and keep CPU caches warm.
Consider double-buffering: while one chunk is filtering, prefetch the next chunk’s raw data via prefetch_read_data or OS-level hints. Saves a little scheduling overhead—maybe 5–10 %.
With the earlier wins, this can slide the runtime from ~1.1 ms to ≲0.9 ms.
5. SIMD + specialization polish
Specialize the entire scan for the handful of common predicate/aggregate pairs (equality filter + sum, range + count). Codegen narrow, monomorphic loops using #[inline(always)] and target_feature(enable = "avx2").
With everything vectorized and branch-free, hitting 0.75 ms (1.3 B rows/s) becomes feasible on modern desktop CPUs.
execution order
Replace row-id accumulation with a bitmap (and adjust gather consumers accordingly).
Integrate Arrow / SIMD predicate kernels to operate directly on primitive buffers.
Rewrite aggregation to consume the bitmask + raw values without materialization.
Profile again; if we’re still above 1 ms, introduce chunk-size tuning and hardware prefetch.
Add specialized AVX2/NEON loops for top predicates to shave the final fractions.
fast follow checks
Add Criterion configurations with larger sample sizes and --measurement-time to stabilize perf numbers after each major change.
Keep Samply traces focused on the bench binary to verify each optimization moves the right hotspot.
Ensure correctness by extending the existing tests to cover the new bitmap path (especially for sparse filters and null-heavy columns).
With these structural changes—especially eliminating row-id vectors and letting predicates/aggregates stay columnar—we’re aligning the scan_stream path with the column-map benchmark’s zero-copy, vectorized architecture. That’s the lever you’ll need to break through the 1 ms barrier and chase 0.75 ms.
