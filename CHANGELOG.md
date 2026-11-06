# 📋 Changelog

All notable changes to omxcore (called MX frame until the F-165 rename; older entries keep the old name) are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Changed
- **Renamed to omxcore (F-165).** `import omxcore as mx`; `import mxframe` still works and gives the same modules (`compat/mxframe/`), with no warning yet. The Mojo package is `omxcore` (`from omxcore import ...`, `build/omxcore.mojoc`); `scripts/build_mojo_pkg.sh` also builds `build/mxframe.mojoc` from the same source for one stable release, so apps that write `from mxframe import ...` keep building. The PyPI distribution is still `mxframe`, the `mx` command line keeps its name, and every `MXFRAME_*` variable, the kernel library names and `~/.mxframe` / `~/.cache/mxframe` are unchanged. Docs and site moved to `abhisheksreesaila/omxcore` (the repo rename follows). After updating a checkout, run `pixi run setup-omxcore` once: an older environment still has the 0.6.0 editable install, where `import mxframe` finds the old package instead of the alias. New page: [the story](https://abhisheksreesaila.github.io/omxcore/story.html).

### Added
- **SQL `COUNT(DISTINCT e) ... GROUP BY g` as two phases (F-206).** `GROUP BY (g, e)` through the dense / hashed machinery (fused into joins), then a count per `g`; a trial over the first 1/32 of the rows keeps the old pair set when nearly every row would make its own (g, e) group. SF1 random queries r64 140 to 11 ms (DuckDB 19), r68 40 to 22 (15), r40 26 to 17 (15); SF10 r11 106 to 17 (10.5), r9 118 to 82 (43). An inner join whose side an exact count found empty is not read a second time.
- **SQL sums past int64 are DuckDB's answer (F-192).** `mx sql` / `sql()`: a `SUM` or `AVG` of INT or DECIMAL values whose total passes int64 is exact in 128 bits like DuckDB's HUGEINT / DECIMAL(38) (global, grouped and window), including HAVING, ORDER BY, `+ - *`, comparisons and a scalar subquery over such totals: TPC-H Q11 runs (all 22 equal DuckDB). The result column keeps the upper 64 bits in `Column.high`. A total past int64 inside a derived table, CTE, UNION, window input or Parquet output still raises the int64 error.
- **SQL timestamps (F-108).** `TIMESTAMP '..'`, `CAST('..' AS TIMESTAMP)` and `TIMESTAMP_S / _MS / _NS '..'` literals, compared in the column's unit and zone like DuckDB (a DATE next to a timestamp is promoted, so `HAVING min(ts) > DATE '2024-01-01'` works; nanoseconds past 2^53 stay exact); CASE and COALESCE of a timestamp and a DATE give a timestamp; `MIN` / `MAX` of a computed string, date, timestamp or 64-bit integer (CASE, COALESCE, NULLIF, `d + INTERVAL ..`) keeps its type instead of going through float64 (`custom_ops._computed_extremes`); `date_or_timestamp +/- INTERVAL n DAY | HOUR | MINUTE | SECOND` on a column (a TIMESTAMP, like DuckDB).
- **SQL `COALESCE`, `IFNULL`, `NULLIF`, `ABS` and `CAST(x AS FLOAT | DOUBLE | TINYINT | SMALLINT | INT | BIGINT)` (F-100).** COALESCE takes the common type of its arguments like CASE does (`compiler.coalesce_arrow`, sharing `_supertype` with `case_when_arrow`; a FLOAT column next to an integer or DECIMAL literal stays float32, only a DOUBLE widens it); NULLIF is `CASE WHEN a = b THEN NULL ELSE a END`; ABS keeps its argument's type (an integer minimum raises); a float cast to an integer rounds half to even and raises when it does not fit. `expr IS [NOT] NULL` works over all of them. `mxframe.lazy_expr.coalesce(...)`, `Expr.abs()` and `Expr.cast("float32")` are the DataFrame forms.
- **Data bigger than GPU memory no longer fails (F-056 step 1).** Each GPU piece (fused aggregate, join, sort, gather) estimates its device memory first; over 90% of the GPU's free memory it runs on the CPU. A piece that runs out of memory anyway is halved and retried once (aggregates) and then rerun on the CPU, and the shape is remembered. `device="gpu"` warns, `device="auto"` stays quiet, `strict=True` raises `MemoryError` with the estimate; `explain(include_runtime=True)` (`gpu_pieces`) and the auto decision say when and why a piece left the GPU. `MXFRAME_GPU_MEM_BUDGET` (e.g. `512M`, or `0:512M,1:2G` per GPU) fakes a smaller GPU for tests.
- **Joins of two tables bigger than memory finish (F-056 step 3b).** When the other side of a join over a `scan_parquet` source does not fit, when the big file is the right side of a left / semi / anti / IN / NOT IN join, or when a file is joined with itself, both sides are hash-partitioned by the join key to `MXFRAME_SPILL_DIR` and joined pair by pair (a Grace hash join): the aggregate or row query above runs per pair on `cpu`, `gpu` (when the pair fits the GPU) or `auto`, partials merge as in step 2, a partition that still does not fit is split again, and a key repeated too often to split raises a `MemoryError` naming it. `explain()` says "grace join, 16 partitions". Equal keys meet by SQL rules (-0.0 and 0.0, NaN, an integer and the float of the same value, strings with different dictionaries; NULL never matches).
- **A Parquet file bigger than memory is streamed (F-056 step 2).** When the columns a `scan_parquet` query reads (after row-group pruning) are bigger than the host budget (`MXFRAME_HOST_MEM_BUDGET`, e.g. `64M`; by default 70% of free RAM), the query runs in batches of row groups: each batch is the ordinary query (filter, join to the dimension tables, partial aggregate) on `cpu`, `gpu` or `auto`, the partial aggregates are merged, and whatever sits above the aggregate (HAVING, sort, limit) runs on the merged result. Memory is one batch plus the group state; the dimension tables are read once and stay resident. Sums, counts, means, minimums and maximums are supported (integer sums stay exact past 2^53; NULL and NaN follow the in-memory rules; `reproducible=True` gives the same bits every run). `lf.last_compile_provenance["path"]` is `"streamed"` with the batch count, devices and budgets under `"streamed"`. A plan that cannot stream (no aggregate, a Sort / Distinct below the aggregate, count distinct, two big tables joined) warns and runs in memory as before; group state over half the budget raises a `MemoryError` saying so. Design and the step 3 list: `docs/design/streamed-execution.md`.
- **Changed:** `device="gpu"` served a join or sort that ran out of GPU memory with a `MemoryError`; it now returns the CPU's answer with a warning (`strict=True` keeps the error).

- **Subquery-shaped plans run natively (F-033).** `LazyFrame.scalar(agg)` is a scalar subquery for a later filter, `with_columns` or HAVING (computed natively first, used as a literal; NULL over no rows keeps no rows). A subquery's HAVING under a join (`WHERE k IN (SELECT k ... GROUP BY k HAVING ...)`: TPC-H Q18, Q21) is the executor's HAVING instead of an Arrow predicate over every group. `SELECT DISTINCT` on dictionary / small-range integer keys is the keyed group-by with a hidden `MIN(row id)` (first appearance kept): one `mx_execute` call fused with the filter, or with the star join's lookups, no filtered table and no gather; wide key spaces and plain strings keep `mx_distinct`. `MXFRAME_DISTINCT_GROUPBY=0` switches it off.

### Fixed
- **Float SUM / MEAN reproducibility (F-064).** The native CPU aggregates handed row blocks to whichever worker was free and, with many groups, added into one shared output with atomic float adds, so the last bits of a float sum changed from run to run (5,076 of 10k group sums at SF1) and a query comparing a view with its own maximum (TPC-H Q15 with an uncomputed view) returned 0 rows. Three changes. (1) A scalar subquery over an aggregate plan the outer plan also contains (`view.filter(col("t") == view.scalar(col("t").max()))`) now shares ONE materialized result with the outer plan, so such self-comparisons are exact at any group count (only plans ending in an aggregate are shared; a CTE used twice in a FROM is still computed twice). (2) A group's sum has one defined value: per 65,536-row morsel the group's rows are added in row order into a sub-total, and the sub-totals are added in morsel order; it does not depend on the thread count, on which worker ran what, or on the other aggregates in the query. By default this order runs only where it is as fast as before or faster: global aggregates and fused joins with 16,384 or more groups (up to 2x faster than the atomic shared output they replace); every other group-by keeps the previous fastest code, whose last bits can still differ between runs (a table per morsel measured 10-20% slower there; A/B on 6M rows, 12 threads). (3) `compute(reproducible=True)` / `MXFRAME_REPRODUCIBLE=1` (the variable is only the default; the flag travels with the call, thread-safe) uses the defined order on every native CPU path, including the older pipeline when `MXFRAME_NATIVE=0` and the CPU share of a CPU+GPU split (a wave of partial tables folded into the output in morsel order up to 131,072 groups at any number of rows, otherwise rows staged per morsel and applied per group range in the same order; 0-20% slower on keyed group-bys). Guarantee: same input table (same chunking, filter) gives the same bits. Not covered: up to 1,024 groups each group keeps 4 accumulator lanes (a global aggregate one per SIMD slot), so the order is fixed but not plain sequential; GPU float atomics outside `reproducible=True` (F-067 covers the GPU fused aggregate with it); the per-call CPU kernels (`group_sum`, `group_mean`, masked sums) take morsels in a fixed order per worker, reproducible for a fixed thread count only.
- **Kernel libraries find the Mojo runtime from the environment they are installed in** (F-036). The 0.6.0 wheel still ships `kernels_aot/*.so` with an absolute RUNPATH of the build machine's pixi env; the next release carries a relocatable one (`scripts/set_runpath.sh`). Linux only; macOS is out of scope.
- **Integers past 2^53 (F-037).** Integer `+ - *` whose result could pass 2^53 no longer rounds in the float64 interpreter: the compiler bounds each result (column types first, the data's min / max only when needed) and the exact path answers. An integer `SUM` whose total could pass 2^53 is now added exactly in int64.
- **Changed result type:** a `SUM` of an integer column that has nulls, or whose total could pass 2^53, now returns int64 (it was float64). Sums that fit stay float64 on the native path for now; making every integer `SUM` int64 is the follow-up (F-060, below).
- **Integer `+ - *` in int64, with no pass over the data (F-060b).** F-037 read each integer column's min / max once per query whenever the column types could not prove a result within 2^53, and answered in Arrow when the data passed it. The kernels (native executor, fused pipeline, hashed groups, GPU) now compute such arithmetic in int64 registers (new ops `IADD`, `ISUB`, `IMUL`, `F2I`, `I2F`): exact at any size, nothing read first. A result outside int64 raises like DuckDB; an int64 past 2^53 that meets a float operation makes the kernel decline and Arrow answer. An int64 beside a double in Arrow's `+ - *` and comparisons rounds like DuckDB's cast instead of raising `Integer value ... not in range`. `min` / `max` of an integer expression is int64 too.
- **Integer `SUM` is int64 on every path (F-060).** The result type depends on the input type only: the native executor, the fused pipeline, the GPU kernel, the NULL-aware Arrow path and a streamed query all add integers in int64 (new bytecode op `AGG_ISUM`; the Mojo kernels accumulate int64 instead of float64, so totals past 2^53 are exact) and return int64 whatever the values, row count, NULLs or batching, where the native paths answered a double when a bound proved the total fits in 2^53 and int64 otherwise. A total that does not fit int64 raises `OverflowError: Overflow in sum of INT64` (DuckDB's `HUGEINT` holds it; ours stays int64); a running total that passes int64 and comes back is exact. `avg` of integers is a double whose total is added in int64. The streamed-execution `SumProbe`, which only existed to keep batches and memory on the same type, is removed. Window sums are still float64.
- **Integer overflow past int64 is an error, not a wrong answer (F-061).** Integer `+ - *` on the exact path used to wrap at 2^63 (`a*a*a` with a = 2^30 gave 0 and the filter dropped rows); it now raises `OverflowError: Overflow in multiplication of INT64`, like DuckDB. An integer `SUM` (global or grouped) whose total does not fit int64 raises `OverflowError: Overflow in sum of INT64` instead of wrapping (DuckDB returns a 128-bit HUGEINT there; our SUM column stays int64). When every fast pipeline declines, an aggregate or filter that holds integer `+ - *` that could pass 2^53 (each expression bounded on its own, so an unrelated predicate such as an ordered string comparison cannot hide it; one the compiler cannot bound counts as unbounded) is answered by Arrow's checked kernels instead of the compiled graph, which wrapped silently. Group keys and projected columns (`with_columns`, `select`, sorts on them) are computed in Arrow and already raise.

### Docs
- **Docs site** at https://abhisheksreesaila.github.io/mxframe/ (GitHub Pages; `docs/site/`). It has interactive benchmarks, how `device="auto"` decides, the path a query takes (for porting), how the engine got fast, and the kernel visualizer. Every number on it is computed in the browser from `docs/bench`.
- **README** is short now: why, the latest numbers, install and run, API reference tables. A test runs every example in it and checks its numbers against `docs/bench`.
- `docs/api.md` and `CONTRIBUTING.md` are rewritten for the native executor.
- **LICENSE** file added (Apache-2.0, as already declared in `pyproject.toml`).

### Changed
- **`when(cond, then)` without `otherwise` is NULL, not 0 (F-105).** It was `otherwise=0`, so `when(p, 1).count()` counted every row and `.mean()` averaged the zeros; pass `0` explicitly to keep the old result. SQL `CASE` without `ELSE` follows (it was 0 too). A CASE result now has the common type of its branches like DuckDB (strings, dates, integers, `float32` next to an integer or DECIMAL literal; any other number mix is double) instead of always float; `CASE x WHEN v` compares `x = v`; `THEN NULL`, `THEN TRUE` and `THEN DATE '...'` work. A CASE that can be NULL is answered by the NULL-aware Arrow path (`sum(case ... else 0 end)` stays native).
- `device="auto"` now runs the native executor on every CPU decision (F-053); the co-processing policy is set only for a predicted CPU+GPU split or an explicit `gpu_fraction`.

## [0.6.0] — 2026-10-14

The CPU engine is now a native Mojo executor: one call per query stage runs the scan, filter,
joins and aggregate over Arrow memory, compiled ahead of time. Engine-only TPC-H and a seeded
random-query suite are the benchmarks, with every result checked against DuckDB. Numbers and the
story behind each step: `docs/optimizations.md`.

### 🧠 Native Mojo executor (F-030, F-031, M2a–M2n)
- One native call (`libmxexec_cpu.so`, over the Arrow C Data Interface) runs a whole plan stage: a
  bytecode interpreter for predicates and expressions, group-by specialized at compile time
  (direct-mapped, compacted or hashed keys), and joins fused into the scan as lookups (star joins
  in one pass, semi / anti joins without a DISTINCT, nested stars).
- Native DISTINCT, ORDER BY ... LIMIT (per-worker heaps), WHERE-only filters, `str_slice` and
  utf8 `isin`, COUNT(DISTINCT) on sorted runs.
- One thread per physical core; fewer workers for small inputs; no `pyarrow.compute` or pandas
  import on the query path; the Mojo runtime loads once.

### ⚡ Closing the gaps to DuckDB (F-043, F-046)
- Millions of groups aggregate into one shared output behind a per-worker cache; keys double as
  ORDER BY tie-breaks; a runtime join filter for large semi-join partners.
- Dictionary strings stay as codes through joins, single-table filters, SELECT DISTINCT and
  COUNT(DISTINCT), and are decoded only for the rows that reach the output.
- COUNT(DISTINCT) on unsorted keys: a byte map for small key x value ranges, hash partitions by
  key range or by (key, value) pair, so few keys no longer stall on one worker.
- WHERE + ORDER BY + LIMIT in one pass (`mx_filter_topk`).

### 📥 Parquet, SQL and planning
- `mx.scan_parquet` / `mx.read_parquet_table` with projection and filter pushdown and dictionary
  strings (F-010, F-017); `mx.sql` front end on sqlglot (joins of every side, aliases, GROUP BY
  keys by alias, F-041); `device="auto"` planner with explain and fallback (F-005).
- Fixed: a group-by on an aliased column (F-040); a SQL `SELECT` alias on a group key (F-041).

### 📊 Benchmarks
- Engine-only TPC-H scoreboard (`scripts/bench_user_felt.py --mode engine`, also `--shuffle`d
  rows) and a seeded random-query suite (`scripts/bench_random/`) rendered for MXFrame, Polars and
  DuckDB, all checked against DuckDB. DuckDB's lazy pyarrow import is now staged like every
  engine's import (it had inflated DuckDB's times).


### 🔗 Join engine: lean, reordered, late-materialized (F-018)
- **New `join_engine.py`.** A chain of inner joins is flattened into leaves (the inputs, filtered and cut to the columns the query reads) and key edges, joined in an order chosen from the leaves' real post-filter row counts (small filtered sides first, the larger side probes and the smaller builds), and carried as row-index vectors: a join gathers only the key columns it compares and composes the indices, every output column is gathered once at the end. Output column names and order are exactly those of the joins as written; left joins are barriers (their two sides are executed on their own). A `Filter` comparing columns of two leaves between joins (TPC-H Q5) is applied as soon as its leaves meet.
- **Projection pushdown through joins:** each `Join` produces only the columns read above it (through `Filter`, `Project`, `Sort`, `Aggregate`, `Distinct`).
- **Optimizer:** a filter is now optimized before the Parquet sources are read, so a predicate moved below a join lands on its `ParquetScan` and is applied while decoding (F-017). A disjunction across both sides of a join (`(a.x = 1 AND b.y = 2) OR (a.x = 3 AND b.y = 4)`, Q7, Q19) pushes what it implies to each side (`a.x IN (1, 3)`); the exact predicate stays above the join.
- **Join keys:** a composite integer key joins on its widest column and checks the others on the matching pairs (dictionary-encoding 11M `(partkey, suppkey)` pairs took 2 s); sparse integer keys up to 32x the row count use the direct-address join instead of hashing every row into a dictionary; the prefilter kernel `libmxjoin_cpu.so` (`kernels_aot/joinsemi_cpu.mojo`) drops probe rows whose key is not in a small build side with a cache-resident bitmap before the table join (456K orders against 60M lineitem rows: 0.57 s to 0.25 s); NumPy passes over long key arrays run on several threads.
- Strings of a small joined side (`n_name` of 25 nations over millions of rows) are gathered as dictionary codes when the consumer above is a filter, projection or aggregate.
- A one-shot gather after a GPU join runs on the CPU (0.12 s against 0.5 s for 3.3M rows x 6 columns: the GPU one uploads its sources and downloads the result).
- **Fixed:** the composite group id of two or more high-cardinality keys was `int32` and wrapped around, merging distinct groups (`Distinct` on `(l_orderkey, l_suppkey)` dropped 3,080 of 6M rows, TPC-H Q21 was wrong); it is `int64`, with a rank-fold fallback beyond 2^62 combinations. `count(col)` of a right-side column after a left join counts unmatched rows as 0 (regression test). `np_arrow.take_rows` no longer imports pandas for string columns.
- **Optimizer, projection pushdown:** a new pass `prune_unused_columns` drops the outputs of a `Project` that its parent does not read. `with_columns` passes every input column through, so after a join it named every column of both tables, and `resolve_plan` read them all from Parquet (TPC-H Q8 SF10: 16 lineitem columns, 5.4 s of reading instead of 5 columns, 0.8 s).
- **Join output limit:** the join kernels index their output with int32 offsets; a many-to-many join with more than 2,147,483,647 output rows (two 46,500-row tables on one key) wrote through wrapped offsets and segfaulted in `join_scatter`. It now raises `MemoryError` before any allocation. (The GPU join's prefix sum is int32 as well and is not guarded.)
- New `MXFRAME_JOIN_TRACE=1` prints one line per join step (rows in and out, time of the key, encode, join and compose stages).

### ✅ Correctness against DuckDB: counts, nulls, TPC-H queries (F-011)
- **Counts** are exact `int64` on every path (grouped and global, fused and legacy, empty results); the fused path used to return float64 and the GPU count kernel saturates at 2^24 per group (the host now counts labels).
- **Nulls follow SQL.** An aggregate that reads a column with nulls runs on Arrow's hash aggregate: a null group key is its own group, `sum/mean/min/max/count` skip null values, a comparison with null is not true, `AND`/`OR` are three-valued, a null `CASE` condition takes `ELSE`. Sorting with null keys puts them last (it raised `IndexError`).
- A `Sort/Limit/Distinct/Tail` below a `Filter/Project/Aggregate` is applied first (a `Limit` under an `Aggregate` was ignored).
- Join keys of different integer (or float) widths are cast to the wider type instead of raising; the join-result cache is keyed by device; the GPU gather no longer rounds `float64` columns to `float32` (it moves the 8-byte words) and defers nulls / other widths to Arrow.
- A numeric literal next to a `float32` column is compared in `float32` on every path (`l_discount <= 0.07` dropped the rows equal to 0.07 on the unfused paths); `case_when` values are `float64` on the CPU; `min`/`max` of a group of only ±inf returns that infinity on the legacy AOT paths.
- New `Expr.year_of_days()` for day-number date columns (`year()` reads YYYYMMDD integers).
- **TPC-H benchmark queries** (`scripts/benchmark_tpch.py`) follow the TPC-H text and use day-number dates (Q4, Q5, Q7, Q8, Q10, Q15, Q20 returned no rows on real Parquet; Q12's date constants were 5 days off; Q19 lacked the ship mode/instruction filter, Q13 the comment filter, Q21 the EXISTS clause). Q5 and Q8 need the supplier table: `supplier=` keyword. The pandas and Polars versions changed with them.
- `scripts/gen_tpch_parquet.py` casts DECIMAL to `float32` through `float64` (nearest); the direct cast truncated 0.05 to 0.049999997 and TPC-H Q6 came out 28% low. Regenerated data lives in `data/tpch_sf{1,10}_exact`.
- **GPU memory pool:** `mxframe.aot_kernels` defaults `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT` to 25 (the Mojo runtime otherwise reserves ~90% of free GPU memory at the first kernel-side buffer and our own allocations ran out at SF10). It is process-wide, respects a value you set, and is documented in the README. `hash_join` raises instead of returning an empty join if its prefix sum is 0 over non-zero match counts.
- Join outputs never hold two columns of one name: a right-side column whose name is taken is renamed `<name>_right` (`_right2`, ...), for inner and left joins; the optimizer's filter pushdown uses the same names.
- `tests/test_tpch_duckdb.py`: all 22 queries against DuckDB (official SQL over the same Parquet files) on `cpu`, `gpu` and `auto`, plus the pandas/Polars baselines.

### 🩺 Kernel health: GPU crashes, hangs and per-call overhead (F-012)
- **GPU grouped aggregates** (`group_sum/min/max/count`): grid-stride kernels sized to the GPU with a block-private table in dynamic shared memory sized to the real group count (up to 12,288 groups) or straight global atomics beyond it. `group_min`/`group_max` no longer fault above 8,192 groups. The big win was in `_atomic.mojo`: atomics were `seq_cst`, which made NVPTX put a fence around every atomic; they are `monotonic` now (CUDA `atomicAdd` semantics). 30M rows, resident input: 8 to 4,096 groups went from 44 to 610 ms down to about 1 to 3 ms.
- Skewed high-cardinality group-bys no longer collapse: a float `atomicrmw fadd` lowers to a compare-exchange loop, so a hot group cost O(k^2) retries (10M rows / 100k groups with one key at 25%: 40 s). The global path for sum/count now uses native `red.global.add.f32` and combines equal groups within a warp first (`match.any`); Zipf(1.3) is within 1.5x of uniform.
- GPU wrappers bind the CUDA context to the calling thread, so the Router's GPU worker thread can call them.
- **GPU `group_encode_i32`** no longer hangs (spin-wait on plain loads could read a stale L1 line) and accepts every int32 key including `INT32_MIN`. The Q21 hash-table kernel had the same spin-wait and was fixed the same way.
- **GPU sort** is stable, correct for any length (the old network skipped out-of-range partners, so any non-power-of-two length could come back out of order), ascending or descending, and about 2.6x faster (64-bit key|row words, shared-memory tiles). Multi-key composite sort keys no longer overflow int32 (also fixed on the CPU path).
- **GPU hash join** runs its prefix sums and the right-side argsort on the device.
- **Device-resident chaining:** the `AOTKernelsGPU` wrappers take `DeviceArray` inputs (`DeviceArray.borrow(ptr, dtype, n)` for residency-tracker or stager pointers) and `device_out=True` to leave results on the device. NumPy signatures are unchanged.
- **CPU `sort_indices`** is a stable parallel LSD radix sort for int32/int64/float32/float64 keys (10M rows: about 60 ms, was O(n^2) and hung at 10,000 rows); `unique_mask`, `prefix_sum_count` and `filter_gather_*` are parallel. Kernel ABI version 3 (these gained the trailing `threads` argument).
- Auto mode no longer treats the GPU sort as broken; the GPU calibration probe for `group_encode` is enabled and the min/max group-count limit is gone.

### ⚡ Multi-core CPU kernels with float64 accumulation (F-002)
- CPU AOT kernels (`libmxkernels_aot.so`) now run on all cores: grouped sum/min/max/mean/count, masked global sum/min/max/sum-product (SIMD), join build/probe/scatter, gather, and group-key composition. Grouped kernels use per-worker partial arrays merged in parallel; the join build is parallel by key range.
- Sums and means accumulate in **float64** and return float64 (counts are exact). At TPC-H SF10 the float32 accumulators were off by up to 4.6% (counts by 42%); Q1-style grouped sums now match DuckDB summing the same float32 values as doubles to ~1e-14 relative.
- Compound aggregate expressions (`price * (1 - discount)`) are evaluated in float64 on the CPU path.
- New thread budget: env `MXFRAME_CPU_THREADS` (default: all cores), `mxframe.set_cpu_threads(n)` / `mxframe.get_cpu_threads()`, and a per-call `threads=` on every `AOTKernels` method. The library keeps no global state, so several Python threads can call kernels at once.
- The exported symbols gained a trailing `threads` argument (ABI version 2); a stale `.so` is rejected with a clear error. `device="gpu"` is untouched.

---

## [0.5.0] — 2026-07-11

Full narrative: [v0.5.0 release notes](RELEASE_NOTES_0.5.0.md).

### ⚡ Native GPU UTF-8 predicates
- Added AOT `utf8_startswith_mask_gpu`, `utf8_contains_mask_gpu`, `utf8_equal_mask_gpu`, and packed-literal `utf8_isin_mask_gpu` kernels over Arrow validity, offsets, and byte buffers.
- Supports sliced arrays, nulls, empty strings/patterns, duplicate literals, and multibyte UTF-8 while matching Arrow filter semantics.
- GPU expression dispatch covers string `==`, `!=`, `startswith`, literal `contains`, and non-null literal `isin`; unsupported/null-membership cases retain Arrow fallback.
- Added permanent CPU/GPU Phase 6 tests and `pixi run bench-utf8` comparisons against PyArrow and Polars.
- At 1M rows, packed `isin` and `contains` beat local Arrow and Polars; exact equality remains faster in Polars' optimized CPU path, so no universal string-performance claim is made.

### ⚡ Weak-query GPU fusion
- Q6 dispatches `range5_sum_product_f32_gpu` instead of materializing five masks and a product array.
- Q4 uses `semi_range_group_count_i32_gpu` for fused qualifying-key marking, semi-join probing, and grouped counts.
- Q13 uses `key_match_counts_i32_gpu` to emit one compact count per customer without materializing a left join.
- Q21 uses `single_late_supplier_counts_i32_gpu` for exact `(order, supplier)` deduplication and deterministic compact supplier counts.
- Representative 1M/10M GPU results versus Polars: Q4 **3.4/28.6 vs 20.2/105.4 ms**; Q13 **1.8/28.5 vs 26.5/442.0 ms**; Q21 **6.6/49.8 vs 30.2/99.1 ms**.

### 📊 Bounded 22-query benchmark
- Every query and engine now runs in its own process, releasing Arrow, Pandas, and CUDA allocations between measurements.
- Added per-engine timeouts, five-query progress groups, merged CSV output, and optional synchronized `cudf.pandas` workers.
- MX CPU beat Polars on **22/22 at 1M** and **18/22 at 10M**.
- MX GPU beat Polars on **15/17 comparable paths at 1M** and **14/17 at 10M**.
- Q8 GPU remains `N/A:JIT`; Q15/Q17/Q18/Q20 GPU workers exceeded the bounded timeout.
- cuDF was unavailable locally, so RAPIDS remains `N/A` and no RAPIDS parity claim is made.

### 🎬 Visualization and identity
- Q4, Q6, Q12, Q13, Q14, and Q21 visualize their logical rewrite, dispatch boundary, and step-by-step Mojo kernel algorithm.
- Added a cool-gray MX mark whose rising X stroke is shaped as orange lightning.
- Verified responsive rendering and zero horizontal overflow in Playwright.

### Remaining performance-parity work
- Replace Q8's JIT path and fix Q15/Q17/Q18/Q20 GPU timeouts.
- Improve Q1 and Q14 GPU pipelines at 10M.
- Run synchronized 1M/10M benchmarks on a compatible RAPIDS/cuDF environment.

### Remaining feature-parity work
- Native GPU UTF-8 dictionary construction and string/date gather.
- GPU windows and additional semi/anti/cross/as-of/range joins.
- Broader regex/string, datetime, nested-type, reshape, SQL, streaming, and I/O pushdown support.
- Cross-generation NVIDIA testing plus AMD and Apple Silicon validation.

---

## [0.4.0] — 2026-07-11

### 🔗 String and composite joins
- Join keys from both tables now use one shared Arrow dictionary before dispatch, so equal strings receive identical dense `int32` IDs regardless of encounter order.
- Multi-column joins jointly densify shared per-column codes, avoiding mixed-radix overflow and supporting mixed string/integer composite keys.
- SQL null semantics are preserved: null keys receive side-specific IDs and never match, including left joins.
- Compact nonnegative integer joins retain their direct `int32` fast path; sparse and non-integer keys are safely densified.
- Fixed GPU upload-cache aliasing by retaining each cached NumPy host owner, preventing recycled host pointers from returning stale device buffers during joined-table gather.
- String encoding runs in Arrow on CPU; resulting IDs use the Mojo AOT CPU/GPU join kernels. Native GPU UTF-8 dictionary construction remains future work.

---

## [0.3.0] — 2026-07-10

### 📚 Docs & Benchmarks
- README restructured: 543 lines → 119 lines. Intro, quick start, and 4-row teaser only — all detail moved to dedicated docs.
- New `docs/benchmarks.md`: full 22-query TPC-H tables (1 M + 10 M rows), Mojo kernel catalogue, current limitations, roadmap, and reproduce instructions.
- New `docs/api.md`: LazyFrame method reference, expression syntax, SQL frontend, supported operations, project structure.
- `docs/vision-and-architecture.md`: fixed garbled sections 2.5/2.6/3; success criteria table repaired; all milestones marked ✅.
- `docs/archive/` added to `.gitignore` — old dev notes moved there, not published.
- All stale `Mojo 0.26.2` / `MAX Graph shape-cached join` references updated to reflect v0.2.3 reality (Mojo 26.4, AOT ctypes everywhere).

### 📊 Benchmark updates
- Q5/Q7/Q8/Q10/Q13/Q19 re-measured with per-run join-cache clearing (fresh-join). Previous v0.2.0 numbers for those queries were warm join-cache hits.
- Summary corrected: CPU 20/22 (Q13 loses fresh-join), GPU 12/22 (Q5/Q7 no longer win at 1 M without cache warmup).
- Q8 GPU marked `—` (MAX Graph `year()` groupby JIT too slow to benchmark; Phase 6 target).

### 🔨 Infrastructure
- `modular >= 26.4` requirement added to `pyproject.toml` runtime deps and Dependencies table in README.
- `docs/archive/` gitignored; previously committed archive files removed from history.

---

## [0.2.3] — 2026-07-09

### ⚡ Performance
- **Phase 4 — Device-resident GPU join pipeline** (`join_count_i32_gpu` + `join_scatter_i32_gpu`):
  GPU inner joins now bypass MAX Graph JIT entirely. Cold-start join latency drops from ~300 ms (MAX session init + JIT) to ~70 ms. `AOTKernelsGPU.hash_join()` caches key arrays via `_cached_upload`; `AOTKernelsGPU.gather_table()` gathers all numeric columns with a single shared device-resident index upload.
- **Phase 5 — GPU sort + top-k** (`sort_topk_i32_gpu`): `ORDER BY … LIMIT k` fuses sort and limit in a single AOT bitonic sort kernel, downloading only `k × 4` bytes instead of `N × 4`. `_apply_post_ops_custom` peek-ahead passes `topk=k` directly to the sort kernel.
- **Phase 3 — Vectorised rank mapping**: `_encode_sort_key` replaces the per-unique-value Python `for` loop with a single vectorised NumPy scatter (`rank_of[sorted_order] = np.arange(n_unique)`). ~1.5× speedup for sort-key encoding at 5 M rows.

### 🛠 Fixed
- **GPU sort+limit crash**: `_apply_sort_custom` was calling `_gpu_gather_table` which referenced `gather_f32` as a MAX Graph custom op — that kernel was never registered in the mojopkg. Any `ORDER BY` on `device="gpu"` would crash. Now routes through `_aot_gpu.gather_table` (AOT ctypes, no MAX Graph).
- `_apply_sort_custom` CPU path now uses `np.argpartition` (O(N) partial sort) when `topk < N`, instead of a full O(N log N) argsort.

### 🔨 Infrastructure
- **Mojo/MAX 26.4 migration**: all `kernels/*.mojo` and `kernels_aot/*.mojo` migrated from 26.2 API (`fn` → `def`, `from tensor import` → `from extensibility import`, `DeviceContextPtr` → `DeviceContext`, `enqueue_function_experimental` → `enqueue_function`).
- **`kernels/_atomic.mojo`** (new): drop-in replacement for the missing `std.os.atomic.Atomic`. Implements `fetch_add`, `min`, `max`, `compare_exchange` via MLIR `llvm.atomicrmw` / `llvm.cmpxchg` instructions — portable across NVIDIA and AMD; works in both `mojo package` (JIT) and `mojo build --emit shared-lib` (AOT) contexts.
- All compiled artifacts rebuilt with Mojo 1.0.0b2 (modular 26.4): `kernels.mojopkg`, `libmxkernels_aot.so`, `libmxkernels_aot_gpu.so`.

---

## [0.2.2] — 2026-07-08

### 🔨 Infrastructure
- First release targeting **Mojo/MAX 26.4** (`modular = "26.4.*"` in `pixi.toml`).
- **`kernels/_atomic.mojo`** shim introduced to replace the removed `std.os.atomic` module.
- All kernel sources migrated to 26.4 syntax; compiled artefacts rebuilt.
- `pyproject.toml` `runtime` extra pinned to `modular>=26.4` to prevent 26.2 wheels from being installed by Colab/pip users.

---

## [0.2.1] — 2026-06-xx

### 🛠 Fixed
- TestPyPI re-upload to validate wheel packaging. No functional changes.

---

## [0.2.0] — 2026-05-xx

### ✨ Added
- **Phase 1 — GPU buffer cache + masked global min/max**:
  - `masked_global_min_f32_gpu` / `masked_global_max_f32_gpu` Mojo AOT kernels.
  - `AOTKernelsGPU` device-buffer cache (`_cached_upload`, eviction, `clear_buf_cache`) — eliminates redundant PCIe uploads on repeated hot queries.
  - `group_mean_f32` GPU kernel (sum + count on GPU, divide on CPU).
- **Phase 2 — Masked global aggregation for any expression**:
  - `_compute_masked_global_agg` extended to handle any inner expression via `_eval_expr_arrow` (e.g. `a*(1-b)`, `case_when`, `startswith`, `isin`) — SUM reduction stays GPU-resident.
  - Join-only plans emit `"join_mojo_shortcut"` provenance (not `"pyarrow_shortcut"`) — Q19 restructured to single-pass join+filter+global_agg.
- **Phase 3 — GPU integer key encoding**:
  - `group_encode_i32_gpu` open-addressing hash-table kernel for integer group keys; replaces PyArrow `dictionary_encode` for integer-keyed group-by on GPU.
- **Audit tool** (`scripts/audit_gpu_paths.py`): classifies each of 22 TPC-H queries as `GPU-CLEAN` / `MOJO-CPU` / `FALLBACK` via `last_compile_provenance["path"]`.

### ⚡ Performance
- **22/22 TPC-H queries GPU-CLEAN** (100% Mojo-backed on GPU, zero PyArrow fallbacks in the reduction and join hot paths).

---

## [0.1.2] — 2026-04-22

### 🛠 Fixed
- `pip install "mxframe[runtime]"` now gives a working install. The `[runtime]` extra pulls `modular>=25.5`, which provides the `max.engine` / `max.driver` modules that `mxframe.compiler` imports at module load. (Pixi users get `modular` from the Modular conda channel and should install `mxframe` without the extra.)
- `sqlglot` import in `mxframe.sql_frontend` is now lazy: `import mxframe` no longer requires the `[sql]` extra. Calling `mxframe.sql(…)` without `sqlglot` installed raises a clear `ImportError` with the install hint.
- `__init__.py` `__version__` now matches `pyproject.toml`.

---

## [0.1.1] — 2026-04-22

### 🛠 Fixed
- Packaging: `readme` now points to `README.md` so PyPI shows the full project page with benchmarks and quickstart (was accidentally showing the internal vision/architecture doc).
- CI: `AOTKernelsGPU` init no longer hard-fails on CPU-only runners (`CUDA driver unavailable` is caught, `self._aot_gpu` falls back to `None`).
- CPU-only import and compute path validated in a clean venv without `modular`.

---

## [0.1.0] — 2026-04-21

Initial public release.