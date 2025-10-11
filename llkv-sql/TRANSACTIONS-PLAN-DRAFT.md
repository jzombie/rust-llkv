Transaction delta context

On BEGIN, we don’t copy any base table data. Instead we create a brand‑new SqlEngine<MemPager> whose pager and context start empty.
Every mutating statement inside the transaction runs against that delta engine. Inserts only add rows there; updates will materialise just the changed rows (so we can replace them later) rather than duplicating whole tables.
Reads with overlays

Whenever a SELECT (or any reader) runs inside a transaction, the main engine will run the query on the base context and, in parallel, re-run it against the transaction delta. We’ll collect those batches and stitch them together while honouring replacements: if a row got updated in the delta, we’ll filter out its original base row and include the new one instead.
This gives us snapshot isolation semantics without copying everything up front.
Commit / rollback

COMMIT will lock the affected tables and replay only the delta rows into the main engine by calling the normal runtime APIs—no low-level pager manipulation, so we preserve the logical integrity guarantees.
ROLLBACK simply drops the delta context; no work required on the primary pager.
Efficiency safeguards

We’ll stream batches when materialising changes so we never hold an entire table in memory.
We keep a compact log (per table: inserted rows, replaced row IDs, deleted row IDs) so we can compose queries efficiently without scanning unrelated tables.
