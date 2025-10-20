# Feature Placement Training

This guide keeps feature work aligned with our architecture so we avoid wasteful refactors and protect runtime performance.

## Know the Stack
- Start with the current linkage diagram in `docs/crate-linkage.md` to understand which crates sit above or below your target area.
- Default to placing new code as low in the stack as it can live **without** sacrificing the performance profile of the higher layers.

## Placement Process
1. Define the feature’s responsibility and data flow.
2. Identify the lowest crate that already owns similar responsibilities.
3. Confirm the crate can expose the needed API without pulling in higher-layer dependencies.
4. If you must stay in a higher layer for performance or ownership reasons, note the exception in the PR description.

## Guardrails
- **Performance first:** moving logic downward must not introduce cross-crate chatter that slows hot paths.
- **No duplicates:** extend existing modules instead of cloning functionality elsewhere.
- **No loops:** ensure dependency arrows still point downward; adding a reference that forces a circular dependency is a blocker.

## Before You Open a PR
- [ ] Verified that the chosen crate is the lowest layer that satisfies the feature.
- [ ] Reviewed existing modules to avoid re-implementing the same capability.
- [ ] Ran benchmarks or targeted tests when performance-sensitive paths are touched.
- [ ] Confirmed Cargo manifests stay acyclic.
- [ ] Updated the crate linkage diagram if the dependency graph changes.

Keep this checklist close; following it is cheaper than repeating large-scale refactors later.
