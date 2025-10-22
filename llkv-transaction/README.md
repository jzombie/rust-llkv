# LLKV Transaction

**Work in Progress**

`llkv-transaction` provides transaction management and MVCC (Multi-Version Concurrency Control) for the [LLKV](../) toolkit.

## Purpose

- Implement snapshot isolation using MVCC semantics.
- Allocate and track transaction IDs with commit/abort status.
- Provide transaction context for coordinating operations across [`llkv-runtime`](../llkv-runtime/).
- Enforce row visibility rules based on transaction snapshots.

## Design Notes

- Each transaction operates with a consistent snapshot determined by its transaction ID and snapshot timestamp.
- Row versioning tracks when each row was created and deleted via `created_by` and `deleted_by` columns stored in [`llkv-table`](../llkv-table/).
- The transaction manager (`TxnIdManager`) allocates monotonic transaction IDs and tracks their commit status.
- Transactions see a consistent view of data as of their start time, preventing read anomalies.
- Used by [`llkv-runtime`](../llkv-runtime/) to coordinate session-level transactions and by [`llkv-executor`](../llkv-executor/) for visibility filtering during SELECT queries.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
