# llkv-plan

`llkv-plan` hosts the reusable planner intermediate representation (IR) and
related utilities for the LLKV toolkit.  It captures optimized plans as a
directed acyclic graph (DAG) of operators, supports deterministic DOT/JSON
serialization for snapshot testing, and enables embedders to hydrate plans into
runtime executors.

## Features

- Stable [`PlanGraph`] data model with node/edge metadata.
- Validation helpers that guarantee the graph is acyclic and internally
  consistent.
- DOT and JSON serializers for visualization and golden tests.

## Testing

```bash
cargo test -p llkv-plan
```

## License

Apache-2.0
