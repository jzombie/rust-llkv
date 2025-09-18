import numpy as np
import pandas as pd
import duckdb
import polars as pl
import timeit

# Create a dataset of 1 million 64-bit integers
num_rows = 1_000_000
data = np.random.randint(0, 100, size=num_rows, dtype=np.int64)

# Convert to Pandas DataFrame and Polars DataFrame
pandas_df = pd.DataFrame({'numbers': data})
polars_df = pl.DataFrame({'numbers': data})

# --- Benchmarking ---

# 1. Polars
polars_time = timeit.timeit(lambda: polars_df.sum(), number=100) / 100

# 2. DuckDB
# DuckDB can query Pandas DataFrames directly, avoiding read/write overhead
duckdb_time = timeit.timeit(lambda: duckdb.query("SELECT SUM(numbers) FROM pandas_df").fetchone(), number=100) / 100

# 3. Pandas (which uses NumPy)
pandas_time = timeit.timeit(lambda: pandas_df['numbers'].sum(), number=100) / 100

# 4. Pure Python Loop (for comparison)
python_list = data.tolist()
def sum_python_list():
    total = 0
    for i in python_list:
        total += i
    return total

python_time = timeit.timeit(sum_python_list, number=10) / 10

# --- Print Results ---
print("--- Benchmark: Summing 1 Million Integers ---")
# timeit returns seconds, so multiply by 1000 for milliseconds (ms)
print(f"Polars:       {polars_time * 1000:.3f} ms")
print(f"DuckDB:       {duckdb_time * 1000:.3f} ms")
print(f"Pandas/NumPy: {pandas_time * 1000:.3f} ms")
print(f"Pure Python:  {python_time * 1000:.3f} ms")