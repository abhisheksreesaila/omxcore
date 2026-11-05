# The Mojo engine: what it does and how it got fast

omxcore's core is a Mojo engine: no Python, it links into any app (Linux, macOS arm64) the way SQLite does, and it
aims to beat DuckDB and Polars on CPU and GPU. Python (`pip install mxframe`, `import omxcore`) and apps such as
[omxplorer](https://github.com/abhisheksreesaila/mxapp) are plugins on top of it. This page records what the engine
does and how each part got fast, with the numbers, including where it is still behind.

**About the numbers:** all timings below are informal: measured on a shared 12-core / 24-thread Ryzen with an RTX
3090 while other work ran (load 2 to 10), best or median of a few runs, DuckDB 1.5.1 and pyarrow on the same files in
the same minute. The quiet-machine scoreboard (Mojo engine vs DuckDB and Polars, CPU and GPU) has not been run yet.
Every answer quoted here is checked against DuckDB or pyarrow by the test suites; every ticket below was attacked by an
independent reviewer with adversarial inputs before it counted as done (ticket numbers refer to `docs/plan.md`).

## At a glance

| Area | What it does | Headline |
|---|---|---|
| 📄 Reading Parquet | bulk, parallel, SIMD decode; lazy dictionaries | numeric read 0.15 s for 20M rows (pyarrow 0.25 s) |
| 🔤 Reading text | words found in parallel, codes in file order | 5M-row text read 1.29 s → 0.17 s |
| 💾 Bigger than RAM | streamed queries a row group at a time, a block per core | 1.6 GB file in 72 MB; 1.3-1.4x DuckDB, faster with few row groups |
| ✏️ Edits on big files | the in-memory edit layer over a streamed file; streamed save | 20M rows: 1,350 edits in 0.8 s, save in 3.1 s at 180 MB |
| 🟩 GPU | in-memory tables and streamed files; AUTO picks the device | streamed: up to 2.6x the CPU on 2-4 cores; AUTO keeps the CPU on 12 |
| 📊 Aggregates | sum, count, min, max, mean, count distinct, median, percentile, top-k | count distinct and median 1.5-5x DuckDB at 1M groups |
| 🔢 Types | INT, DECIMAL (exact), TEXT, DATE, BOOL, FLOAT64 | float sums reproducible on any core count |
| 🧮 SQL | SELECT … WHERE … GROUP BY … HAVING … ORDER BY … LIMIT, EXPLAIN | 1M groups out in 159 ms (DuckDB 1.9 s); top-k 126 ms → 3 ms |
| 🔗 Joins | inner, LEFT, SEMI, ANTI; IN / EXISTS / NOT IN; 3-9 tables with a cost-based planner | join + group sum 110 ms (DuckDB 148); NOT EXISTS 9 ms (35); TPC-H Q9 73 ms (112) |
| 🧩 Expressions | CASE, CAST, text functions once per word, dates and intervals | date_trunc group 4.3 ms (DuckDB 20.6) |
| 🤖 `mx` | SQL in, JSON / CSV / Parquet out, for agents | 12 ms cold start, no Python |

## 📄 Reading Parquet at memory speed (F-140, F-141, F-145)

- **Bulk decode.** A row group's column is sized once and written in place: PLAIN int64 is one copy, int32 one
  widening loop, definition levels unpacked straight into the validity bitmap, dictionary indices unpacked and mapped to
  the file-wide codes in one loop. Before, every value was appended one at a time.
- **Every core inside a row group.** Page headers are walked once; pages decode in parallel into slices of the one
  column; dictionary pages decode as parallel jobs; a single page holding a million rows is cut by row range (at RLE
  run boundaries or inside bit-packed runs) so no core waits on one big page.
- **Lazy dictionaries.** A text column's word list is built only when a query reads that column: an unused unique-ID
  column used to cost 943 MB and 2 s at open, now 39 MB and 0.25 s.

| 20M rows, all cores | before | after | pyarrow |
|---|---|---|---|
| `read_parquet`, numeric | 3.78 s | **0.15 s** | 0.25 s |

Safety: crafted files (a 2 KB file claiming billions of rows, row counts that overflow, pages past the file) get a
readable "corrupt file" error before anything is allocated; thousands of byte-flipped files never crash.

## 🔤 Text without a string per value (F-142)

Text is stored as dictionary codes. Decoding a page with new words used to run on one core in file order. Now each page
looks its words up in parallel, records the ones it has not seen, a single ordered pass names them (so codes are
exactly the first-seen order), and a parallel pass fills them in.

| 5M-row text file | before | after | pyarrow, same output | DuckDB, hashing every word |
|---|---|---|---|---|
| all cores | 1.29 s | **0.17 s** | 0.5-0.6 s | 0.16 s (whole query) |

## 💾 Files bigger than memory (F-140, F-143, F-149)

`Query(scan_parquet(path))` reads one row group at a time, and only the columns the query needs.

- **Overlap.** The next row group decodes while the current one is aggregated (about 20%).
- **Morsels.** Each core owns a row group and walks it in 32k-row blocks, decoding and aggregating each block while it
  is in cache: no core waits at a barrier inside a group. Files with fewer row groups than cores are cut into row
  ranges so every core still works.

| Streamed numeric query, 20M rows | memory | time | DuckDB |
|---|---|---|---|
| one row group at a time (F-140) | 72 MB for a 1.6 GB file | 2.69 s | |
| bulk decode (F-141) | | 0.24 s | 0.10 s |
| overlap + parallel pages (F-143, F-145) | ~2 row groups | 0.150 s (2.4x) | 0.07 s |
| morsels (F-149), 20 row groups | 142 MB | **0.13 s (1.3-1.4x)** | 0.09-0.10 s |
| morsels, 4 row groups | 143 MB (was 448) | **0.14 s (0.8x: faster)** | 0.18 s |

A file truncated while it is open is refused cleanly. Errors are reported in the same order on every core count.

## ✏️ Editing a file bigger than memory (F-147)

The same edit layer as in-memory tables (sets, nulls, appends, removes, merges of spellings, formulas, undo) works over
a streamed file. Queries apply the edits while streaming; reading cells goes through an O(1) cache of decoded chunks;
`save` streams the edited file out a row group at a time and replaces it atomically (you can save over the file you
have open). Saving refuses only when values would be lost (a timestamp held as its date), never silently.

| 20M rows, 567 MB file | time | peak memory |
|---|---|---|
| 1,350 edits | 0.8 s | 56 MB |
| a query over the edited file | 0.1 s | 70 MB |
| saved | 3.1 s | 180 MB |

## 🟩 The GPU, chosen by cost (F-137, F-146)

In-memory tables can be uploaded once (`DeviceTable`) and queried on the GPU; streamed files decode straight into pinned
memory while the previous row group uploads and aggregates on the device. Results equal the CPU exactly. AUTO compares
the plan's work per row, per core, per column uploaded, and records why it chose:

| 20M-row streamed query | CPU | GPU | AUTO |
|---|---|---|---|
| 12 cores, light plan | 0.109 s | 0.132 s | CPU |
| 4 cores, heavy plan | 0.366 s | 0.178 s | GPU |
| 2 cores, heavy plan | 0.597 s | 0.220 s | GPU (2.7x) |

On a big CPU, streaming from disk is decode-bound, so the CPU wins; the GPU pays where data is already on it or cores
are few. A failed GPU query always gives its device memory back.

## 📊 Aggregates (F-133, F-148)

The core aggregation is one bytecode kernel, run per block on every core. Holistic aggregates add their own algorithms:
count distinct with HyperLogLog-sized hash sets, percentiles with quickselect or a parallel radix select, top-k with
bounded heaps.

| 20M rows | 8 groups | 1M groups | DuckDB (1M groups) |
|---|---|---|---|
| count distinct | 148 ms | 372 ms | 617 ms |
| median | 120 ms | 398 ms | 2,208 ms |
| top-k | 19 ms | 449 ms | 2,295 ms |

`median` matches DuckDB (the mean of the two middle values, kept exact one decimal place finer).

## 🔢 Types and exactness (F-157)

INT and DECIMAL are exact scaled integers; a sum past int64 raises rather than wrapping. FLOAT64 (Parquet DOUBLE and
FLOAT, CSV floats) follows DuckDB: NaN equals itself and sorts last, -0.0 equals 0.0, 32-bit columns compare in 32 bits.
Float sums run in a fixed order, so the bits are the same on any core count. Floats print as the shortest text that
reads back to the same bits (all 2M doubles tested equal Python's `repr`). CSV parsing is locale-independent.

## 🧮 SQL in Mojo (F-150, F-152)

`sql(catalog, text)` over tables, edited tables and streamed files returns an owned `Table`. Standard SQL rules for
NULLs (three-valued logic, a NULL group, aggregates of nothing are NULL); text sorts ignoring case. A logical plan sits
between the parser and the executor, and a planner behind a trait (reading only statistics) picks dense or hashed
grouping, top-k heaps or a full sort, and early stops; `EXPLAIN` shows every decision and why.

Steps for apps (F-160): `catalog.view(name, sql | Query)` names a query that later queries read like a table; `preview(cat, name, rows)` is
its first rows with the LIMIT inside the view; a result cache keyed by the view's plan text and the versions of its inputs (frame edits, file size and
mtime, upstream views) means an edit re-runs exactly the views below it; `COPY (query) TO 'f.parquet'` writes any query through the Parquet writer.
Details: `docs/omxcore-mojo-api.md`.

| 5M / 1.2M rows | omxcore | DuckDB |
|---|---|---|
| 8 groups, 4 aggregates | 8 ms | 6 ms |
| 1M groups, all rows out | **159 ms** | 1.9 s |
| `ORDER BY … LIMIT 10` | 3 ms (was 126) | 1.4 ms |
| `SELECT DISTINCT s` | **1.3 ms** | 4.0 ms |
| full sort, 1.1M rows | 114 ms | 47 ms |

A Parquet file whose statistics lie cannot give a wrong answer: they are checked while reading and the query re-runs
without them. Malformed SQL never crashes (fuzzed); every unsupported form raises with its line and column.

## 🔗 Joins (F-153, F-154)

A hash-join kernel (`libmxhashjoin_cpu.so`, a C interface any language can call) builds a direct array when keys are
dense and radix-partitioned tables when they are not, in parallel without atomics. Unique build keys join inside the
aggregation kernel (no joined row is written); repeated keys or row output materialize pairs under a memory budget.
Keys follow SQL and DuckDB exactly: NULL never matches, text matches byte for byte across tables, decimals are aligned,
NaN matches NaN.

| 6M x 1.5M x 150k rows | omxcore | DuckDB |
|---|---|---|
| join + group sum | **110 ms** | 148 ms |
| orders x customer | **3.6 ms** | 6.6 ms |
| LEFT join + GROUP BY | **18 ms** | 48 ms |
| EXISTS | **20 ms** | 70 ms |
| NOT EXISTS | **9 ms** | 35 ms |
| two-column key | 49 ms | 35 ms |
| filter on the probe side | 34 ms | 22 ms |
| one-to-many LEFT | 34 ms | 13 ms |

### Many tables and a planner (F-155)

Three to nine tables in one query (`JOIN … ON` chains or comma joins). Each table's filter runs first, so the planner
orders the joins from exact row counts and sampled distinct counts; a fact table with several unique-key dimensions
(a star) probes them all in one fused pass with nothing written in between. `EXPLAIN ANALYZE` shows actual rows and
milliseconds per step beside the estimates. Every TPC-H join shape equals DuckDB under the planner's order and up to
24 forced orders.

| TPC-H SF1 | omxcore | DuckDB |
|---|---|---|
| Q9 | **73 ms** | 112 ms |
| Q19 | 24 ms | 23 ms |
| star: lineitem with orders, part and supplier | **39 ms** | 80 ms |
| Q3 | 32 ms | 27 ms |
| Q5 | 41 ms | 13 ms |
| Q7 | 72 ms | 16 ms |

Q5 and Q7 lose because the plan joins one table at a time and writes out 1.2-1.8M rows between steps; a star-aware
planner that probes every dimension in one pass is the next step (F-161).

## 🧩 Expressions (F-156)

`CASE`, `COALESCE`, `NULLIF`, `CAST` / `TRY_CAST` with DuckDB's rounding and range rules, `IN` lists of any length,
text functions (`lower`, `upper`, `trim`, `substr`, `replace`, `LIKE` / `ILIKE`, `||`, …) evaluated **once per distinct
word, never per row**, and dates (`EXTRACT`, `date_trunc`, `± INTERVAL`, date differences).

| 6M rows | omxcore | DuckDB |
|---|---|---|
| `date_trunc` + GROUP BY | **4.3 ms** | 20.6 ms |
| `date + INTERVAL` filter | **3.7 ms** | 18.8 ms |
| `l_comment LIKE '%special%'` (3.6M distinct words) | 23.7 ms | 12.7 ms |
| `count(DISTINCT lower(h))` (1M distinct words) | 336 ms | 58 ms |

The last row builds new words on one thread; parallel per-word building is on the speed list.

## 🤖 `mx`, for agents (F-151)

```sh
mx sql "SELECT region, sum(revenue) FROM 'orders.parquet' GROUP BY region"     # JSON by default
mx sql "…" --format parquet --out result.parquet       mx explain --analyze "…"       mx schema orders.parquet
```

Errors come back as JSON with line, column and a hint (exit 1 for a query, 2 for usage); output is always valid UTF-8;
big files stream automatically. 12 ms cold start, 21 MB, no Python.

## Built so it stays correct

- **Every ticket reviewed adversarially** against DuckDB / pyarrow: thousands of generated queries, corrupt and lying
  files, fuzzed SQL. Bugs this caught before they shipped include lying Parquet statistics, 32-bit float comparisons,
  merged text in joins, a LEFT join's WHERE treated as a key, subquery scoping, and a leaked GPU context.
- **Structure over vigilance:** an owner object (`KernelRun`) makes "a buffer freed before the kernel reads it"
  impossible; a lint catches the rest; a pre-commit hook refuses stale compiled kernels; a test fails if a kernel
  library is not committed; test runs and package builds never share files.

### Joins over streamed files and edited frames (F-158)

A Parquet scan or a Frame on the probe side of a fused join is never read into a table: each row group (a Frame: the shared
rows, then the changed and appended ones, removed rows masked out) is joined as it arrives, its keys planned once from the
build side and applied to every chunk, the build side's hash map shared by all workers. This holds for aggregates and for row
queries (`ORDER BY ... LIMIT` keeps the best rows of each chunk in per-worker heaps), for inner, LEFT, SEMI and ANTI joins, and
for a star of unique-key dimensions. The note `join path: fused: ... chunk by chunk` says so. Peak memory is the build side plus
the row groups being decoded (up to min(threads, 8) at once, a worker each, for every scan aggregate), not the file: a 6M-row,
6-column scan joined to a 1,000-row dimension peaked at 384 MB read whole and 75 MB streamed (`ru_maxrss` of a fresh `mx`
process; not a timing). When the build keys repeat or are hashed, the join falls back to materializing the probe side.

The build side is collected row by row and counted against the join's memory budget while it grows
(`PlannerOptions.join_budget`, `mx --join-budget`): a build side past it raises "the build side of the join (t) has N rows of
C columns so far, past the join's memory budget of ...", before it is allocated. A join that spills to disk is not built yet.

A big GROUP BY over a scan merges each row group's partial in 256 hash partitions in parallel (1M groups: 1.85 s to 0.56 s on
the day it was built; not re-measured), with exact int64 sums that raise on overflow across row groups. A Frame's formula
columns are columns of a join too, on either side (`s.net`, `s.*`); a formula as a join key is refused by name. Tests:
`tests/test_f158_streamed_join.py`, `mojo/tests/test_sql.mojo` (a Frame probe and a Frame build side equal the join over
`bake()`).

## Still behind DuckDB (next)

- Multi-table joins that write intermediate rows (TPC-H Q5, Q7, Q10: 2-4x) and one-to-many LEFT joins: F-161, a
  star-aware planner that probes every dimension in one pass, filters kept inside the join.
- Full sorts and million-row outputs (~2x), and text functions that build new words over huge dictionaries (F-167).
- Still to build: the gate: all 22
  TPC-H queries through `mx` equal to DuckDB (F-162), and the quiet-machine scoreboard against DuckDB and Polars on
  CPU and GPU.
