# Remote SLT Test Support

## Overview

The SLT test harness now supports `.slturl` pointer files that reference remote test files instead of storing large test files locally in the repository.

## How It Works

### `.slturl` Pointer Files

Instead of committing large `.slt` test files, you can create small `.slturl` pointer files that contain a single URL pointing to the remote test content.

**Example:** `slt_good_0.slturl`
```
https://raw.githubusercontent.com/jzombie/sqlite-sqllogictest-corpus/refs/heads/main/test/index/in/10/slt_good_0.test
```

### Test Execution

When the test harness discovers a `.slturl` file:

1. It reads the URL from the file
2. Fetches the test content from the remote URL using HTTP(S)
3. Executes the test exactly as if it were a local `.slt` file
4. Reports results with the same trial naming convention

### Benefits

- **Reduced Repository Size**: Large test files are not committed to the repo
- **Centralized Test Corpus**: Tests can be managed in a separate repository
- **Easy Updates**: Update tests by changing the remote content, no repo commits needed
- **Identical Behavior**: Tests run exactly the same whether local or remote

## Usage

### Creating Pointer Files

Create a `.slturl` file with a single line containing the URL:

```bash
echo "https://example.com/path/to/test.slt" > my_test.slturl
```

### Running Tests

The harness automatically detects both `.slt` and `.slturl` files:

```bash
# Run the test harness (discovers all .slt and .slturl files)
cargo test --test slt_harness

# Run a specific pointer file programmatically
let runner = LlkvSltRunner::in_memory();
runner.run_file("tests/slt/sqlite/index/in/10/slt_good_0.slturl")?;
```

### Network Requirements

- Tests using `.slturl` files require network access during execution
- Consider using `#[ignore]` for network-dependent tests in CI environments without internet
- The `reqwest` crate with TLS support is required (already included)

## Implementation Details

### Modified Components

1. **`src/runner.rs`**: Updated `run_slt_harness_with_args` to:
   - Discover both `.slt` and `.slturl` files
   - Fetch remote content when processing `.slturl` files
   - Execute remote tests with proper error handling

2. **`src/lib.rs`**: Updated `LlkvSltRunner::run_file` to:
   - Detect `.slturl` extension
   - Delegate to `run_url` for pointer files

### File Format

`.slturl` files must contain:
- A single line with the complete URL
- Whitespace is trimmed automatically
- URL must be accessible via HTTP(S) GET request

### Error Handling

Errors are reported at multiple levels:
- File read errors (cannot open `.slturl` file)
- Network errors (cannot fetch remote URL)
- Content errors (remote content is invalid SLT)

All errors include context about the source file and URL.

## Migration Guide

To convert existing local `.slt` files to remote pointers:

1. Upload your `.slt` files to a remote repository (e.g., GitHub)
2. Create `.slturl` pointer files with the raw content URLs
3. Remove the original large `.slt` files
4. Commit only the small `.slturl` pointer files

**Example:**

```bash
# Before: 58,007 lines, 1.2MB
tests/slt/sqlite/index/in/10/slt_good_0.slt

# After: 1 line, ~150 bytes
tests/slt/sqlite/index/in/10/slt_good_0.slturl
```

## Testing

A test is provided to verify pointer file functionality:

```bash
# Run network-dependent tests
cargo test --test slturl_test -- --ignored
```
