# LLKV Test Utils

**Work in Progress**

Utilities for running integration tests across the [LLKV](../) toolkit.

The crate currently provides a tiny, test-focused tracing initializer and a convenient `auto-init` feature for automatically installing a tracing subscriber at test-binary startup.

This README documents what the crate does, how the `auto-init` feature works, and how to use the helper manually if you prefer explicit control.

Note: This crate does not include the SLT test harness defined in [llkv-slt-tester](../llkv-slt-tester/) in order to prevent circular dependencies when these test utils are used throughout the LLKV workspace.

## What this crate provides

- `pub fn init_tracing_for_tests()`
  - Installs a `tracing-subscriber` `fmt` subscriber with an `EnvFilter` that reads `RUST_LOG`.
  - Safe to call multiple times (uses `std::sync::Once` internally).
- `auto-init` feature (optional)
  - When enabled, this feature pulls in the `ctor` crate and registers a function annotated with `#[ctor]` so `init_tracing_for_tests()` runs at binary initialization time. This means test binaries that depend on the crate with `auto-init` enabled will have tracing initialized automatically without each test needing to call the helper.

## How auto-init works (important notes)

- `ctor` executes the annotated function during the process startup sequence, before `main` or the test harness starts executing tests. That makes it convenient because you do not need to call the initializer from each test file.
- Because it runs at binary init, the auto-init path affects every test binary that links the `llkv-test-utils` crate with the `auto-init` feature enabled. In practice that means every crate that lists `llkv-test-utils = { ..., features = ["auto-init"] }` in its `dev-dependencies` will get tracing automatically for its tests.
- `-- --nocapture` is unrelated to auto-init. The `--nocapture` flag only tells the Rust test harness not to capture and hide `stdout`/`stderr`. The tracing `ctor` runs regardless of `--nocapture` if the crate and feature are compiled into the test binary.

## Control and best practices

- If you prefer explicit control (recommended for incremental adoption), add the crate as a dev-dependency without the `auto-init` feature and call `llkv_test_utils::init_tracing_for_tests()` at the top of tests or in a small test helper module you import into tests.
- If you prefer automatic behavior (convenient but broader effect), enable `auto-init` for the crate in the `dev-dependencies` of the crates you want to see tracing for.

## Examples

Manual init (explicit per-test):

1. Add the dev-dependency in `Cargo.toml` (without `auto-init`):

```toml
[dev-dependencies]
llkv-test-utils = { path = "../llkv-test-utils" }
```

2. In your test file:

```rust
mod common;

#[test]
fn my_test() {
    llkv_test_utils::init_tracing_for_tests();
    tracing::debug!("debug message visible if RUST_LOG allows");
    // ... test body ...
}
```

Automatic init (enabled feature):

1. Add the dev-dependency and enable `auto-init`:

```toml
[dev-dependencies]
llkv-test-utils = { path = "../llkv-test-utils", features = ["auto-init"] }
```

2. Run tests with desired log level and optionally `--nocapture`:

```bash
# Print debug logs from tests
RUST_LOG=debug cargo test -p llkv-sql -- --nocapture
```

Notes:

- Default log level is `info` when `RUST_LOG` is not set. Use `RUST_LOG=<module>=debug` or `RUST_LOG=debug` to see debug traces.
- If you enable `auto-init` in a crate's `dev-dependencies` you do not need (and should not) call the initializer manually; it will already have run.

## How to turn it off

- Remove the `auto-init` feature from the `dev-dependencies` entry in the crate(s) where you do not want automatic startup.
- Or remove the `llkv-test-utils` dev-dependency entirely and switch to the manual initialization pattern only where needed.

## Security and test isolation

Because the auto-init attaches a subscriber early for the whole test binary, the tracing subscriber will capture logs from any code executed in that binary. This is usually desirable for debugging, but be mindful if some tests rely on specific subscriber configurations or global state—those tests may be affected by a globally installed subscriber. The manual approach avoids installing global state for tests that don't opt in.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
