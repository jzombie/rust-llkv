# TPC-H Planner Performance Tasks

- Plan representation: extend `LogicalPlan` with joins (type/keys/filters), aggregates (group keys/outputs), and compounds; resolve all columns to IDs and run every SELECT through it.
- Rewrite + pushdown: add predicate pushdown and projection pruning across joins/aggregates; exploit LIMIT/OFFSET/ORDER BY to favor sorted/index scans and early stopping.
- Stats + costs: collect row counts, min/max, null counts, index presence (plus key histograms for TPC-H); use them for scan/path selection, join strategy, and a greedy join order.
- Physical planning overhaul: choose index/sorted/range scans, join algorithms and ordering from the logical tree; push projections/filters to scans; reuse order to avoid extra sorts.
- Executor de-hardening: remove executor-side ad hoc translation for joins/aggregates/compounds; enforce SQL → logical plan → rewrites → physical plan → executor as the single path.
- Early TPC-H wins: gather stats on TPC-H tables; implement join-key resolution and join-order heuristic; push date/qty/price filters to scans; prefer sort indexes/merge joins for ORDER BY/LIMIT.
