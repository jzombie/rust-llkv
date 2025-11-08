# Rustdoc Preview Guide

This guide walks through generating and inspecting the workspace documentation so you can verify narrative changes before publishing. Commands assume the repository root as the working directory. Replace `open` with the appropriate launcher for your platform (for example, `xdg-open` on Linux or `start` on Windows PowerShell).

## 1. Build the Documentation

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

- `RUSTDOCFLAGS="-D warnings"` ensures broken links, missing docs, and other warnings fail the build instead of slipping through.
- `--workspace` generates docs for every crate so cross-links stay intact.
- `--no-deps` keeps the build focused on workspace crates, avoiding time spent regenerating upstream documentation.

Expect the rendered HTML under `target/doc/`. If the command fails, address the reported warnings or errors before continuing.

## 2. Preview in a Browser

Open the main entrypoint generated for the `llkv` crate:

```bash
xdg-open target/doc/llkv/index.html   # Linux
open target/doc/llkv/index.html       # macOS
start target\doc\llkv\index.html    # Windows PowerShell
```

Alternatively, run a static file server from `target/doc` (for example, `python3 -m http.server 9000`) and browse to `http://localhost:9000/llkv/`.

## 3. Spot-Check Other Crates

When reviewing documentation for supporting crates (for example, `llkv-column-map` or `llkv-runtime`), either follow sidebar links from the main `llkv` page or open them directly:

```bash
xdg-open target/doc/llkv_column_map/index.html
xdg-open target/doc/llkv_runtime/index.html
```

Swap in `open` or `start` if you are on macOS or Windows respectively.

Remember that crate names with hyphens are rendered with underscores in the generated paths.

## 4. Clean Up (Optional)

If you want a fresh rebuild after large changes, clear the previous artifacts:

```zsh
cargo clean --doc
```

Then repeat the build step. This helps catch missing references that might otherwise persist from stale HTML files.

---

Keep this guide handy while iterating on documentation. Running the rustdoc build before every commit protects against broken links and ensures the published story matches the source tree.
