import os
import time
import tempfile
import numpy as np
import pandas as pd
import duckdb
import polars as pl
import timeit

# --------------------------
# Config
# --------------------------
NUM_ROWS = 1_000_000
PARQUET_COMPRESSION = "zstd"  # "snappy" | "zstd" | "uncompressed"
REPEATS = 10  # warm runs (steady-state avg)
WARMUP = 1  # warmup runs (not reported)

# --------------------------
# Data generation
# --------------------------
rng = np.random.default_rng(0)
data = rng.integers(0, 100, size=NUM_ROWS, dtype=np.int64)
pandas_df = pd.DataFrame({"numbers": data})


# --------------------------
# Helpers
# --------------------------
def bench_once(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0  # ms


def bench_many(fn, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    times = [bench_once(fn) for _ in range(repeats)]
    return sum(times) / len(times)


def print_row(name, cold_ms, warm_ms):
    print(f"{name:30s} cold-ish: {cold_ms:8.2f} ms   warm avg: {warm_ms:8.2f} ms")


# --------------------------
# Disk-backed benchmarks
# --------------------------
with tempfile.TemporaryDirectory() as td:
    pq_path = os.path.join(td, "numbers.parquet")
    db_path = os.path.join(td, "duck.db")

    # Write Parquet to disk (via pandas/pyarrow)
    pandas_df.to_parquet(pq_path, engine="pyarrow", compression=PARQUET_COMPRESSION)
    size_mb = os.path.getsize(pq_path) / (1024 * 1024)
    print(
        f"Parquet file: {pq_path}  ({size_mb:.2f} MiB, compression={PARQUET_COMPRESSION})"
    )

    # Prepare a DuckDB database file containing a table with the data
    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=4")
    con.execute(f"CREATE TABLE numbers AS SELECT * FROM parquet_scan('{pq_path}')")
    con.close()

    # --- Polars: Parquet scan (lazy; projection/pushdown capable) ---
    def polars_disk_sum():
        # Using scan_parquet to avoid reading unneeded columns; then collect the scalar.
        out = pl.scan_parquet(pq_path).select(pl.col("numbers").sum()).collect()
        _ = out.item()  # ensure work is materialized

    # --- DuckDB: direct scan of parquet file (no DB load) ---
    def duckdb_parquet_sum():
        # New connection per call (ensures file read path is exercised)
        con = duckdb.connect()
        con.execute("PRAGMA threads=4")
        _ = con.execute(
            f"SELECT SUM(numbers) FROM read_parquet('{pq_path}')"
        ).fetchone()
        con.close()

    # --- DuckDB: query from database file (table stored on disk) ---
    def duckdb_dbfile_sum():
        con = duckdb.connect(db_path)
        con.execute("PRAGMA threads=4")
        _ = con.execute("SELECT SUM(numbers) FROM numbers").fetchone()
        con.close()

    # --- Pandas: Parquet read + sum ---
    def pandas_disk_sum():
        df = pd.read_parquet(pq_path, engine="pyarrow")
        _ = int(df["numbers"].sum())

    # Measure cold-ish (first call) and warm (avg of many)
    print("\n--- Disk-backed: cold-ish vs warm ---")
    results = []
    for name, fn in [
        ("Polars (scan_parquet)", polars_disk_sum),
        ("DuckDB (parquet scan)", duckdb_parquet_sum),
        ("DuckDB (db file)", duckdb_dbfile_sum),
        ("Pandas (read_parquet)", pandas_disk_sum),
    ]:
        cold = bench_once(fn)
        warm = bench_many(fn)
        results.append((name, cold, warm))
        print_row(name, cold, warm)

# --------------------------
# In-memory baselines (your originals)
# --------------------------
print("\n--- In-memory baselines (originals) ---")

# Polars in-memory
polars_df = pl.DataFrame({"numbers": data})
polars_time = timeit.timeit(lambda: polars_df["numbers"].sum(), number=100) / 100 * 1000

# DuckDB in-memory (querying the pandas_df object)
duckdb_time = (
    timeit.timeit(
        lambda: duckdb.query("SELECT SUM(numbers) FROM pandas_df").fetchone(),
        number=100,
    )
    / 100
    * 1000
)

# Pandas/NumPy in-memory
pandas_time = timeit.timeit(lambda: pandas_df["numbers"].sum(), number=100) / 100 * 1000

# Pure Python loop
python_list = data.tolist()


def sum_python_list():
    total = 0
    for i in python_list:
        total += i
    return total


python_time = timeit.timeit(sum_python_list, number=10) / 10 * 1000

print(f"Polars (in-mem)       : {polars_time:.3f} ms")
print(f"DuckDB (in-mem Pandas): {duckdb_time:.3f} ms")
print(f"Pandas/NumPy (in-mem) : {pandas_time:.3f} ms")
print(f"Pure Python (in-mem)  : {python_time:.3f} ms")

print("\nNotes:")
print(
    "- 'cold-ish' means first run after (re)opening; true cold cache is OS-dependent."
)
print(
    "- Parquet compression affects file size and decode cost. Try PARQUET_COMPRESSION='uncompressed' for decode-free comparisons."
)
print(
    "- DuckDB DB file path benchmarks an on-disk table; parquet scan benchmarks direct columnar file reads."
)
