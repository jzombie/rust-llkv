The gap isn’t the endian swap. One‑by‑one sum is ~3.20 Gelem/s; decode_reduce is ~2.89–2.92 Gelem/s (closure + enum overhead).
The scan path is 100 ms to stream 1M items (~10 Melem/s end‑to‑end). That’s orders slower than pure decode, so the iterator’s per‑item work (PQ + shadow checks + reseed) dominates.
What to try next (surgical, low-risk)

Instrument next(): add counters for:

PQ pops/pushes
Active-shadow Bloom checks, precise probes
Binary searches for value/key bounds
Nodes skipped due to shadow
Then expose them via a debug method or test-only feature. This will pinpoint where cycles go on your data shape.
No-shadow fast path: add a policy that skips inter-segment membership probes when you know the workload has disjoint keys across generations (or you’re scanning a dataset with no overlap). Example:

ConflictPolicy::None (or a scan option flag). If set:
Don’t compute dominators
Don’t do shadow checks in next()
Keep everything else identical
Expect large gains when overlap is rare.
Order-aware dominators: you already precompute and cache bounds. Two cheap boosts:

Sort dominators by rank best-first (LWW: newest→older; FWW: oldest→newer) to increase early exits.
Track a small per-seg hit/miss heuristic and try frequent hitters first.
Stream decode in bench without Vec: to isolate iterator overhead even further, add a bench variant that:

For x in it { sum += u64::from_be_bytes(value.as_slice()[..8]) }
No Vec allocation, no decode_reduce closure; compares raw iterator speed vs pure buffer decode.
Keep scans on encoded bytes: don’t flip in column-map. Do bulk decode at the edges only when needed.

Why decode_reduce won’t match the baseline without a typed path

Per-item enum (DecodedValue) + closure call remain even with inlining. If you want a general fast path, consider an experimental typed reducer:
decode_reduce_typed<I, C, T, F>(inputs, init, f) where C: Codec<Owned = u64>
One match outside the loop chooses C; loop runs on native u64 (no enum/closure indirection in the hot decode).
If you want, I can:

Add an optional ConflictPolicy::None and a debug counter struct to read_scan (feature-gated).
Add the “raw streaming decode from iterator” bench variant to your file so you can compare iterator-only cost head-to-head with your memory decode baseline.
