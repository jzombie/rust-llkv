#!/usr/bin/env bash
set -euo pipefail

# Usage: bench-package.sh <package-name>
# Runs `cargo bench --manifest-path <repo-root>/<package>/Cargo.toml`

pkg="$1"

# Determine repo root: prefer GITHUB_WORKSPACE (set by GitHub Actions),
# otherwise fall back to the git top-level or current working dir.
if [ -n "${GITHUB_WORKSPACE:-}" ]; then
  repo_root="$GITHUB_WORKSPACE"
else
  repo_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
fi

manifest="$repo_root/$pkg/Cargo.toml"

if [ ! -f "$manifest" ]; then
  echo "Error: manifest not found at $manifest" >&2
  exit 1
fi

echo "Running cargo bench for package '$pkg' using manifest: $manifest"
exec cargo bench --manifest-path "$manifest"
