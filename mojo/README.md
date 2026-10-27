# The Mojo package (no Python)

omxcore's Python-free API (`docs/omxcore-mojo-api.md`, brief `docs/briefs/mojo-api.md`).

- **Source:** the API in `mojo/omxcore/`. The native kernels stay in `kernels_aot/` (the Python path builds its
  `.so` files from them); `scripts/build_mojo_pkg.sh` copies them into the package as `omxcore/kernels/` with their
  imports made package-relative. Edit kernels in `kernels_aot/`, never in `build/`.
- **Build:** `pixi run bash scripts/build_mojo_pkg.sh` -> `build/omxcore.mojoc`. A program uses it with
  `mojo build app.mojo -I build`; on Linux also `-Xlinker -lzstd -Xlinker -lsnappy` (the Parquet codecs, C libraries) and
  their `-L` / `-rpath` (the pixi env's `lib/` here). Every other target, and Linux with `-D MXFRAME_NO_CODECS`, needs
  no library: it uses the Mojo codecs in `mojo/omxcore/codec.mojo` (below).
- **Tests:** `pixi run mojo-test` builds the package, then builds and runs each `mojo/tests/test_*.mojo` (a `main()`
  that raises on failure). A test binary that links libpython fails.
- **Lifetimes:** Mojo frees a value at its last use. A raw address taken from a `List` is dangling once the list
  is no longer used: keep the owner alive past the call (`_ = owner^`), or better, pass owned types, not addresses.

Why the kernels are copied, not imported from `-I kernels_aot`: a `.mojoc` does not carry modules that were only on
the `-I` path (a consumer then fails with "unable to locate module"), and a shared library that only re-imports
`@export` functions exports nothing, so the `.so` builds must keep compiling the original files.

## SQL (F-150, F-152)

`sql(catalog, "SELECT ...")` runs a query over one table, a Frame or a Parquet scan registered by name (aggregates, or rows: `SELECT *`,
expressions, `DISTINCT`, `ORDER BY`, `LIMIT` / `OFFSET`) and returns an owned `Table` plus the planner's notes and EXPLAIN
(`docs/omxcore-mojo-api.md` section 8). The modules, in the order a query passes them: `sql.mojo` (lexer, parser, syntax tree), `bind.mojo`
(Catalog, typed tree, binder), `plan.mojo` (the logical plan: Scan, Filter, Aggregate, Project, Sort, Limit), `planner.mojo` (`Planner` trait,
`RulePlanner`, `Stats`, `PlannerOptions`; the group strategy and the sink), `sqlprog.mojo` (the kernel program in SQL's mode), then the
executor: `exec.mojo` (Sources: table, frame mask + delta, scan row groups; shared helpers), `sqlexec.mojo` (the aggregate Sinks: dense,
hashed, holistic; `sql`), `rows.mojo` (the row Sinks: materialize through the kernel's block hook, top-k heaps), `post.mojo` (HAVING, the
select list, ORDER BY and LIMIT after aggregation, run as a row query over the groups); `sqleval.mojo` holds the result column type and the
exact AVG. `mojo/tools/sql_probe.mojo` runs one SQL string over Parquet files and prints typed JSON (tests/test_mojo_sql.py and
tests/test_mojo_rows.py compare it, and `mx`, with DuckDB); `mojo/tests/test_sql.mojo` and `test_sql_parse.mojo` run in `pixi run mojo-test`.

## Joins (F-153, F-154, F-155)

The equi-join (inner, left, semi, anti; IN / EXISTS subqueries are lifted to semi and anti joins by the binder): `joincore.mojo` (key normalization, pairs, gather: tables in, a table out, no query language), `join.mojo` (the SQL side: what each side must
deliver, the queries around the join), `jfront.mojo` (the fused front stage for unique build keys), `Query.join` in `query.mojo`, and the kernel
`kernels_aot/hashjoin_cpu.mojo` (int64 pointers in, rows out; `libmxhashjoin_cpu.so` is its C ABI for the Python engine). Tests: `mojo/tests/test_hashjoin.mojo` (kernel against a
brute-force loop), `test_join_api.mojo` (`Query.join`), `tests/test_mojo_join.py` (SQL against DuckDB: fused vs materialized, the build side, every key rule), `tests/test_hashjoin_lib.py`, `tests/test_mojo_outer_joins.py` (LEFT / SEMI / ANTI / subqueries against DuckDB), `mojo/tests/test_frame_merge.mojo` (edited Frames against their baked tables).
F-155: three or more tables: the planner (`planner.mojo`: `JoinStats`, `JoinPlan`, `rule_join_order`, `Planner.join_order`), the join graph in the binder, the multi-way driver and the star in `sqlexec.mojo`, the star's front stage in `jfront.mojo`; tests `tests/test_mojo_multiway.py` (TPC-H Q3 / Q5 / Q7 / Q8 / Q9 / Q10 / Q19 shapes on dbgen SF 0.1 against DuckDB, every join order) and `mojo/tests/test_multiway.mojo`.
Rules and paths: `docs/omxcore-mojo-api.md` section 8b.

## FLOAT64 (F-157)

`FLOAT` is a full column kind (float64 bits in the Int64 slot): Parquet DOUBLE / FLOAT, CSV exponents, the builder (`lit_float`, `Query.threads`,
`QueryResult.floats`), SQL, Frames. The rules (NaN order, -0.0, the reproducible sum, the 32-bit column) are in `docs/omxcore-mojo-api.md`.
`mojo/tools/float_probe.mojo` runs builder plans for `tests/test_mojo_float_builder.py`; `mojo/tests/test_float.mojo` is the pure-Mojo test.
`strtod` (libSystem) parses the digits a fast path cannot: `check_mac_link.sh` lists it for the app's stub.

## A CSV bigger than memory (F-178)

`scan_csv(path, CsvOptions(group_rows=..., block_bytes=...))` opens a CSV or TSV as a scan (`ParquetScan`, also named `CsvScan`): one pass finds its
row groups (`csvscan.mojo`: byte ranges cut where a newline ends a row outside quotes, the state machine `read_csv` uses) and keeps the kinds (from a
sample), the null counts and a failed-row report (`scan.csv_report()`); a group is parsed with `csv.mojo`'s `ChunkSink` when a query reads it, so
queries, SQL, `Frame(scan)` with edits and `save` are the Parquet scan's, and the peak memory is a row group, whatever the file (`docs/omxcore-mojo-api.md`
section 2b). Tests: `mojo/tests/test_csv_scan.mojo` (against `read_csv`), `tests/test_mojo_csv_scan.py` (SQL against DuckDB's `read_csv`, via
`mojo/tools/sql_probe.mojo`), `tests/test_csv_scan_memory.py` (`mojo/tools/csv_scan_probe.mojo`: peak resident memory on 50 MB and 150 MB files).

## SQL expressions (F-156)

CASE, COALESCE / NULLIF, IS DISTINCT FROM, CAST, LIKE, text functions, dates and intervals, IN lists of any length (rules and the table of what is supported: `docs/omxcore-mojo-api.md`,
"Expressions"). The parser (`sql.mojo`) and binder (`bind.mojo`, a marked section: types shared by the branches, derived TEXT dictionaries, the date and interval rules) make typed nodes;
`sqlprog.mojo` emits them: a text function is evaluated **once per distinct dictionary word** (`DictAccess` holds the query's derived dictionaries in a shared `DerivedStore`; the kernel
looks codes up with `OP_TAB`, predicates use `OP_IN_SET`), dates are integer civil-date arithmetic in the integer opcodes, a long `IN` is the kernel's hash set (`OP_ISET`), and a failing CAST sets
the kernel's error flag (`OP_FLAG`), gated by the CASE branch it sits in. The three opcodes are in `kernels_aot/pipeline_cpu.mojo` (the committed `libmxpipeline_cpu.so` / `libmxexec_cpu.so` carry them).
`sqlfn.mojo` has the scalar functions as plain functions on values (what DuckDB answers for one word or one day, including its quirks), `unicase.mojo` is DuckDB's case mapping
(`scripts/gen_unicase.py` asks DuckDB for it). Tests: `tests/test_mojo_exprs.py` (every function over a table, an edited frame, a streamed file and a frame over it), `test_mojo_expr_scale.py`
(every day 1800-2400 and 60,000 UTF-8 words, value by value), `test_mojo_expr_fuzz.py` (random expressions), `test_mojo_cast_fuzz.py` (`mojo/tools/cast_probe.mojo`: how every text cast reads
thousands of generated words, against DuckDB's TRY_CAST), `test_mx_exprs.py` (through `build/mx`). No 128-bit division in the package: Apple's libSystem has none (`scripts/check_mac_link.sh`).

## Ranges and blocks over an order (F-176)

`range_aggregate(source, [order,] cols, a, b)` (count / sum / mean / min / max of each column over positions a..b of a row list, exact at each column's scale) and
`read_block(source, [order,] cols, a, b, values, nulls)` (the window's cells and null flags into the caller's buffers) in `blocks.mojo`; a `SharedTable`, a `Frame` with edits and a `Frame` over a Parquet
file are all sources (rules, errors and timings: `docs/omxcore-mojo-api.md` section 6b). A column of the window is gathered into a dense buffer and reduced by tight loops, in morsels of 16,384 positions on every
core (a Frame over a file gathers on the calling thread through its chunk cache). Tests: `mojo/tests/test_blocks.mojo` (against the per-cell loop: tables, edited frames, scans with odd row groups, bounds, timings),
`tests/test_mojo_range.py` (against DuckDB through `mojo/tools/range_probe.mojo`; the FLOAT sum's bits equal with one thread and many), `mojo/tools/range_bench.mojo` (the cost of edits). No app concept is in the module:
it takes row numbers and a half-open range.

## `mx`, the agent command line (F-151)

`mojo/cli/mx.mojo` is `mx sql | explain | schema` over the SQL above: JSON in, JSON out, exit 1 for a query error and 2 for a usage error
(`docs/omxcore-mojo-api.md`, "The `mx` command line"). `mojo/omxcore/output.mojo` writes a Table as JSON, JSON lines or CSV (exact decimals,
ISO dates). Build: `pixi run bash scripts/build_mx.sh` → `build/mx` (it fails if libpython or Mojo's Python interop is linked). tests/mx_oracle.py
runs a SQL string through `build/mx` and DuckDB and compares typed values; tests/test_mx.py uses it.

## Background jobs (F-175)

`compute_job`, `sql_job`, `read_csv_job`, `read_parquet_job`, `save_job` and `write_parquet_job` (`jobs.mojo`) run the plain call on a thread of its own and hand back a `Job[T]` (`job.mojo`): `poll()`, `wait()`,
`result()`, `cancel()`, `progress()` and an optional completion callback. The work checks for a cancel between row ranges, windows, row groups, blocks and chunks and moves one atomic progress counter per block; a cancelled
save keeps the old file; dropping a handle cancels and frees. Engine functions take `job: JobCtl = JobCtl()` (a plain call passes none; SQL carries it in `PlannerOptions.job`). The serial build runs a job inside its start call.
Rules, what is checked where, and the tests: `docs/omxcore-mojo-api.md` section 8c; `mojo/tests/test_job.mojo`, `test_jobs.mojo`.

## Build switches (F-138b): the package on an app with no GPU, thread pool or C codecs

Three compile-time switches, read when the app builds (`mojo build ... -D NAME`), not when the package is built. Each
is **on for every target but Linux** (an Apple object needs no flags) and `-D` turns it on for Linux too, which is how
`pixi run mojo-test` tests the other side (`mojo/tests/switches/test_switches.mojo`). **Android counts as Linux** to
Mojo: an Android build gets the GPU code, the thread pool and the codecs unless it passes all three `-D` flags.
**The GPU on a Mac** (F-182): a build for macOS whose accelerator is an Apple GPU (a native build on Apple Silicon, or
`--target-accelerator metal:<n>`) has the GPU code too, so `MXFRAME_NO_GPU` is off there; it needs Apple's Metal
toolchain at build time. A Linux cross-build for macOS that names only the triple keeps the GPU off without a flag.

| Switch | Off (Linux default) | On |
|---|---|---|
| `MXFRAME_NO_GPU` (`HAS_GPU` in `gpu.mojo`) | `DeviceTable`, `compute(GPU)`, the GPU kernel (Linux, and macOS with an Apple GPU) | `gpu_available()` is False; `DeviceTable(...)` and `compute(GPU)` raise "no GPU on this build"; AUTO is the CPU. No `AsyncRT_Device*` symbol. |
| `MXFRAME_SERIAL` (`kernels/_fpenv.mojo`) | the thread pool (`par`, `par_n`) | tasks run in a loop on the calling thread, `core_count()` is 1, no `AsyncRT_*` symbol |
| `MXFRAME_NO_CODECS` (`parquet.mojo`) | zstd and snappy from the C libraries (libzstd, libsnappy) | zstd and snappy from `codec.mojo` (F-180): `write_parquet` compresses as asked and `read_parquet` reads what any writer made. No `ZSTD_*` / `snappy_*` symbol, nothing to link or ship |

**The Mojo codecs** (F-180, `mojo/omxcore/codec.mojo`). Parquet pages are compressed the same on every target: on Linux
by libzstd / libsnappy, everywhere else by zstd and snappy written in Mojo, so an `.omx` saved on a Mac is zstd-compressed
like one saved on Linux and each build opens the other's files. No source is vendored, so there is no licence to carry:
the code is written from RFC 8878 (zstd) and snappy's `format_description.txt`, and tested against the C libraries
(`mojo/tests/test_codec.mojo`: both directions, zstd levels 1 / 3 / 9 / 19, damaged input) and against pyarrow and DuckDB
(`tests/test_codec_mojo.py`, through `mojo/tools/pq_recode.mojo` built with `-D MXFRAME_NO_CODECS`). The decoder reads
all of zstd but dictionaries, and is strict: checksums verified, offsets bounded by frame start and window, sizes
checked before allocating; the encoder writes one frame with a 1 MB
window per page. Against libzstd level 3 / libsnappy on a realistic table (`mojo/tools/bench_codec_write.mojo <rows> <path>
[threads]`: 2 million orders, 8 columns with nulls, text and dates, 74.6 MB plain; Linux, Ryzen 9 5900X, best of five inside
the bench, run twice, load average 2 on a 24-thread machine that is not quiet):

| | file | save, 1 thread | save, all threads | open (`read_parquet`) |
|---|---|---|---|---|
| zstd, libzstd level 3 | 15,994,016 bytes (4.66x) | 0.220 s | 0.035 s | 0.017 s |
| zstd, Mojo | 16,025,403 bytes (4.65x) | 0.36 s | 0.058 to 0.071 s | 0.022 to 0.028 s |
| snappy, libsnappy | 31,420,051 bytes (2.37x) | 0.093 s | 0.034 to 0.038 s | 0.019 to 0.028 s |
| snappy, Mojo | 31,209,841 bytes (2.39x) | 0.124 s | 0.034 to 0.043 s | 0.016 s |
| no codec | 74,589,554 bytes | 0.073 s | 0.063 s | 0.013 s |

Before F-187 the same save took 0.595 s with the Mojo zstd and 0.268 s with libzstd (one thread each, files 16,067,618 and
16,058,417 bytes). The Mojo zstd save is now 1.6x libzstd's on one thread (was 2.2x) and 1.7 to 2x with every thread (the pool has
the 12 physical cores); compressing alone, on the columns of that table plus TPC-H lineitem columns, source text and a shared
library (13 inputs, 64 MB), the encoder takes 342 ms against libzstd level 3's 186 ms on one thread (1.8x; it was 528 ms,
2.8x) and writes 0.5% fewer bytes. The 1.5x target was not reached on one thread.

**Threads in a save** (F-187, `omxcore/fan.mojo`). A column of a row group is cut into data pages of about 1 MB (131,072
INT rows; smaller columns stay one page), and every page is built and compressed on its own; the pages are then written in
order, so the file is the same bytes for any number of threads (`write_parquet(..., threads=n)`: 0 is every thread the build
has, 1 the calling thread; tested for 1, 2, 3, 5, 8 and all, both codecs, with pyarrow and DuckDB reading the result).
The threads are the build's own: Linux with the pool uses `par_n` (workers take the next page as they finish one, because
pages differ 50-fold in cost), a **plain Mac build stays serial** (no async runtime, as before), and a serial build
with `-D MXFRAME_JOB_THREADS` fans out over pthreads with the calls its jobs already need (`pthread_create`, `pthread_join`,
`pthread_attr_*`) and `sysconf`: `scripts/check_mac_link.sh` reports no new symbol. The pages a save holds in flight follow a byte budget (`budget=`, F-188), not a thread cap: a page waits until what the pages in flight hold, with the coder's scratch, fits (`WRITE_BUDGET` 256 MB for a table; `STREAM_WRITE_BUDGET` 57 MB, four pages, for a save that streams a file). The bytes written are the same for any budget. A small budget means fewer pages at once and a slower save (a 1-4 MB budget on a 125 MB table: about 0.35 s against 0.06 s). Reading is
faster too: a file saved this way has several pages per column, which `read_parquet` decodes in parallel (0.040 s to
0.022 s on the table above).

AUTO and the GPU: `compute(AUTO)` goes to the CPU when the GPU path fails for one of two reasons only, an allocation
failure for the query's buffers or a program the kernel declines (-5 shared memory, -6 group table; on an Apple GPU
also a float program), and says which in
`QueryResult.fallback_reason`. A launch error or an illegal address raises on AUTO too. `compute(GPU)` never falls back.

**Link check:** `pixi run bash scripts/check_mac_link.sh` compiles `mojo/tools/mac_link_probe.mojo` over the package
for `arm64-apple-macosx14.0`, links it with `ld64.lld` against the app's `apple/sdk/macos/*.tbd` and its
`mojoui/hosts/mojo_rt.mojo` stand-ins, and fails on any AsyncRT, Tracy, KGEN (beyond the stand-ins), zstd, snappy,
Python or CUDA symbol (the probe saves and reads both codecs, so the Mojo ones are linked in). It prints the libSystem calls the package adds to the app's list (file I/O: `open`, `read`,
`mmap`, `rename`, ...): the app's stub and its `ALLOWED` set need them (F-138c).
