#!/usr/bin/env python3
"""
omxcore quickstart — write queries the way you'd expect.

Run:
    pixi run python scripts/quickstart.py
    pixi run python scripts/quickstart.py --device gpu
    pixi run python scripts/quickstart.py --rows 5_000_000

—————————————————————————————————————————————————————————————
AVAILABLE OPERATIONS (everything the engine supports today)
—————————————————————————————————————————————————————————————

  LOADING DATA
    mxf.from_arrow(pa.Table)          — from an existing PyArrow table
    mxf.from_pandas(df)               — from a Pandas DataFrame
    mxf.from_polars(df)               — from a Polars DataFrame
    mxf.read_csv("path.csv")          — from a CSV file
    mxf.read_parquet("path.parquet")  — from a Parquet file

  DATA EXPLORATION  (returns concrete Python results)
    lf.compute()                      — execute and return a PyArrow table
    lf.compute().to_pandas()          — get a Pandas DataFrame
    lf.compute().schema               — column names + types

  FILTERING (lazy — no work until .compute())
    lf.filter(col("amount") > lit(100))
    lf.filter((col("region") == lit("West")) & (col("amount") >= lit(50)))
    lf.filter(col("category").isin(["Electronics", "Clothing"]))
    lf.filter(col("name").startswith("A"))
    lf.filter(col("name").contains("pro"))
    lf.filter(~col("cancelled"))                     — NOT
    lf.filter(col("amount").between(10, 100))        — 10 ≤ amount ≤ 100 (inclusive)

  COMPUTED / PROJECTED COLUMNS
    lf.with_columns(
        (col("price") * col("qty")).alias("revenue"),
        when(col("qty") > lit(10), lit("bulk"), lit("single")).alias("order_type"),
    )
    lf.select(col("id"), col("revenue"))             — keep only those columns

  AGGREGATION
    lf.groupby("region")
      .agg(
        col("amount").sum().alias("total"),
        col("amount").mean().alias("avg"),
        col("amount").min().alias("min"),
        col("amount").max().alias("max"),
        col("amount").count().alias("n"),
      )

    lf.groupby("year", "region").agg(...)            — multi-key groupby

    lf.groupby()                                     — global (whole table) agg
      .agg(col("amount").sum().alias("grand_total"))

  SORTING / LIMITING / DEDUP
    lf.sort("amount")                                — ascending (default)
    lf.sort(col("amount"), descending=True)          — descending
    lf.limit(10)
    lf.tail(10)                                      — last N rows (complement of limit)
    lf.distinct()

  DATA INSPECTION
    lf.schema                                        — pa.Schema without executing the query
    lf.describe()                                    — count/mean/std/min/25%/50%/75%/max

  JOINS
    orders.join(customers, left_on="cust_id", right_on="id")
    orders.join(products,  left_on=["store","sku"], right_on=["store","sku"])
    orders.join(other, left_on="id", right_on="id", how="left")  — left outer

  DATE HELPERS (dates stored as int32 YYYYMMDD)
    col("order_date").year()                         — extract year

  WINDOW / ANALYTIC FUNCTIONS  (computed via with_columns)
    col("amount").rank()                             — global rank (1,2,2,4…)
    col("amount").dense_rank()                       — dense rank (1,2,2,3…)
    row_number()                                     — sequential 1..n
    col("amount").cum_sum()                          — running sum
    col("amount").lag(1)                             — previous row value
    col("amount").lead(1)                            — next row value

    All of the above accept .over() for partitioned evaluation:
    col("amount").sum().over("region")               — group sum broadcast to every row
    col("score").rank().over("region", order_by="score")
    row_number().over("region", order_by="date").alias("rn")
    col("amount").cum_sum().over("region", order_by="date").alias("running_total")
    col("amount").lag(1).over("region", order_by="date").alias("prev_amount")

  CASE / WHEN
    when(condition, then_expr, else_expr)
    when(col("score") >= lit(90), lit("A"), lit("B")).alias("grade")

  EXPRESSION ARITHMETIC
    col("a") + col("b")   col("a") - col("b")
    col("a") * col("b")   col("a") / col("b")
    col("a") * lit(0.1)                              — multiply by scalar

  SQL INTERFACE
    mxf.sql("SELECT region, SUM(amount) FROM orders GROUP BY region",
            orders=orders_arrow_table)

  EXECUTION
    lf.compute(device="cpu")   — force CPU Mojo kernels
    lf.compute(device="gpu")   — force GPU kernels
    lf.compute(device="auto")  — auto-pick (GPU if available + large table)

  CACHE MANAGEMENT
    mxf.warmup("cpu")          — pre-JIT so first query is fast
    mxf.clear_cache()          — wipe compiled graph cache

—————————————————————————————————————————————————————————————
"""

import argparse
import time

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv

import omxcore as mxf
from omxcore import col, lit, when


# ── CLI args ─────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--rows",   type=int, default=1_000_000)
ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "auto"])
ap.add_argument("--save-csv", action="store_true",
                help="also write orders.csv / customers.csv so you can load them yourself")
args = ap.parse_args()

N      = args.rows
DEVICE = args.device


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1: generate synthetic data
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nGenerating {N:,}-row sales dataset...")
rng = np.random.default_rng(42)

regions    = ["North", "South", "East", "West", "Central"]
categories = ["Electronics", "Clothing", "Food", "Furniture", "Sports"]
statuses   = ["completed", "pending", "refunded"]

n_customers = max(N // 20, 100)

customers_arrow = pa.table({
    "customer_id": pa.array(np.arange(n_customers, dtype=np.int32)),
    "name":        [f"Customer_{i}" for i in range(n_customers)],
    "region":      rng.choice(regions, n_customers).tolist(),
    "tier":        rng.choice(["gold", "silver", "bronze"], n_customers).tolist(),
})

orders_arrow = pa.table({
    "order_id":    pa.array(np.arange(N, dtype=np.int32)),
    "customer_id": pa.array(rng.integers(0, n_customers, N, dtype=np.int32)),
    "category":    rng.choice(categories, N).tolist(),
    "amount":      rng.uniform(5.0, 2000.0, N).astype(np.float32),
    "qty":         pa.array(rng.integers(1, 50, N, dtype=np.int32)),
    "unit_cost":   rng.uniform(1.0, 500.0, N).astype(np.float32),
    "status":      rng.choice(statuses, N).tolist(),
    "order_date":  pa.array(rng.integers(20200101, 20241231, N, dtype=np.int32)),
})

if args.save_csv:
    import scratch   # scripts/scratch.py: ~/.cache/mxframe-test-tmp, never the shared /tmp
    pa_csv.write_csv(orders_arrow,    str(scratch.root() / "orders.csv"))
    pa_csv.write_csv(customers_arrow, str(scratch.root() / "customers.csv"))
    print(f"  Saved {scratch.root()}/orders.csv and {scratch.root()}/customers.csv")

print(f"  orders:    {orders_arrow.num_rows:>10,} rows  {orders_arrow.nbytes/1e6:.0f} MB")
print(f"  customers: {customers_arrow.num_rows:>10,} rows")


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2: load into omxcore  (LazyFrame — nothing executes yet)
# ─────────────────────────────────────────────────────────────────────────────
orders    = mxf.from_arrow(orders_arrow)
customers = mxf.from_arrow(customers_arrow)

# Could also do:
#   orders = mxf.read_csv("orders.csv")
#   orders = mxf.read_parquet("orders.parquet")
#   orders = mxf.from_pandas(some_pandas_df)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3: warm up (pre-JIT — absorbs the one-time Mojo compile cost)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nWarming up {DEVICE.upper()} kernels...")
t_warmup = mxf.warmup(DEVICE)
print(f"  done in {t_warmup:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
#  Helper — time a query and print a preview
# ─────────────────────────────────────────────────────────────────────────────
def run(label: str, lazy_frame):
    # Run 3 times, report min
    times = []
    result = None
    for _ in range(3):
        t0 = time.perf_counter()
        result = lazy_frame.compute(device=DEVICE)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"\n{'─'*56}")
    print(f"  {label}")
    print(f"  {result.num_rows} rows  |  {min(times):.1f}ms  (best of 3)")
    print(f"{'─'*56}")
    print(result.to_pandas().head(5).to_string(index=False))
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4: write your queries below
# ─────────────────────────────────────────────────────────────────────────────

# ── Query A: total revenue and order count by category ─────────────────────
run("Total revenue & order count by category",
    orders
    .filter(col("status") == lit("completed"))
    .groupby("category")
    .agg(
        col("amount").sum().alias("total_revenue"),
        col("amount").count().alias("num_orders"),
        col("amount").mean().alias("avg_order"),
    )
    .sort(col("total_revenue"), descending=True)
)

# ── Query B: high-value orders (> $500) joined with customer region ─────────
run("High-value completed orders with customer region",
    orders
    .filter((col("amount") > lit(500.0)) & (col("status") == lit("completed")))
    .join(customers, left_on="customer_id", right_on="customer_id")
    .groupby("region", "category")
    .agg(
        col("amount").sum().alias("revenue"),
        col("amount").count().alias("orders"),
    )
    .sort(col("revenue"), descending=True)
)

# ── Query C: profit margin column + group by tier ──────────────────────────
# with_columns adds computed columns without aggregating
run("Avg profit margin per customer tier",
    orders
    .with_columns(
        (col("amount") - col("unit_cost") * col("qty")).alias("profit"),
    )
    .join(customers, left_on="customer_id", right_on="customer_id")
    .filter(col("status") == lit("completed"))
    .groupby("tier")
    .agg(
        col("profit").mean().alias("avg_profit"),
        col("profit").sum().alias("total_profit"),
        col("amount").count().alias("orders"),
    )
    .sort("tier")
)

# ── Query D: yearly order trend ────────────────────────────────────────────
run("Yearly revenue trend (order_date is YYYYMMDD int)",
    orders
    .filter(col("status") != lit("refunded"))
    .groupby(col("order_date").year().alias("year"))
    .agg(
        col("amount").sum().alias("revenue"),
        col("amount").count().alias("orders"),
    )
    .sort("year")
)

# ── Query E: between + tail (Phase 8 polish) ───────────────────────────────
run(
    "Mid-range orders (between $100–$500), last 5",
    orders
    .filter(col("amount").between(100, 500))
    .tail(5),
)

# ── Query F: schema inspection (no .compute() needed) ──────────────────────
print("\n── Schema (no compute required) ────────────────────────────────────────────")
schema = orders.filter(col("amount") > lit(0)).schema
for field in schema:
    print(f"  {field.name:20s} {field.type}")

# ── Query G: describe numeric columns ──────────────────────────────────────
print("\n── Describe orders.amount ──────────────────────────────────────────────────")
desc = orders.select("amount").describe()
for row in range(desc.num_rows):
    stat = desc["statistic"][row].as_py()
    val  = desc["amount"][row].as_py()
    print(f"  {stat:<8s} {val:>10.2f}")

# ── add your own queries here ───────────────────────────────────────────────
# e.g.
# run("Electronics only",
#     orders
#     .filter(col("category") == lit("Electronics"))
#     .groupby("status")
#     .agg(col("amount").sum().alias("total"))
# )
#
# run("Top 10 spenders",
#     orders
#     .join(customers, left_on="customer_id", right_on="customer_id")
#     .groupby("name", "region")
#     .agg(col("amount").sum().alias("total_spent"))
#     .sort(col("total_spent"), descending=True)
#     .limit(10)
# )
#
# SQL interface also works:
# result = mxf.sql(
#     "SELECT category, SUM(amount) AS rev FROM orders GROUP BY category",
#     orders=orders_arrow,
# ).compute()

print("\nDone.\n")
