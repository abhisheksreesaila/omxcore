# Stable engine version for apps

Apps that vendor the Mojo package (`omxcore.mojoc`, called `mxframe.mojoc` until the F-165 rename) read this file to know which commit to build from.
Scripts read the one-line file `STABLE` (just the commit). When a new version becomes stable, an issue labelled
`mxframe` (the label keeps its name) is also opened on the app's repo (abhisheksreesaila/mxapp), and the app's scheduled check updates itself.
Only commits listed here are stable: every ticket in them passed adversarial review, and the full test suites were
green. Anything newer on `main` may still be in review: don't vendor it.

## Current stable

**`834ef9a`** (2026-10-29): SQL CTEs, derived tables, views, subqueries (correlated ones too), UNION / INTERSECT /
EXCEPT and QUALIFY; GPU shared memory bounded by the device (omxcore#19). Includes everything in 5e4cdb1 and earlier.

```sh
# build the package from a clean copy of the stable commit (never from a working checkout), on Linux as before
git -C <omxcore> archive 834ef9a | tar -x -C <scratch>
cd <scratch> && pixi run bash scripts/build_mojo_pkg.sh      # -> build/omxcore.mojoc (and build/mxframe.mojoc)
cp build/omxcore.mojoc <app>/vendor/ && echo 834ef9a > <app>/vendor/OMXCORE_COMMIT
```

The Mac build commands are unchanged from 3d2455d (below). **No new libSystem symbols** (`scripts/check_mac_link.sh`
OK). For parallel saves on the Mac, build the app with `-D MXFRAME_JOB_THREADS` (the pthread symbols listed under
23edc59, nothing more); without it, saves run on one thread.

## What changed since the previous stable (5e4cdb1), for apps

| Area | What you get | Ticket |
|---|---|---|
| SQL: CTEs, subqueries, set operations | `WITH` (a CTE read once runs in place, read more than once runs once; `MATERIALIZED` hints), derived tables in FROM (with column aliases), `Catalog.view`; scalar subqueries (empty → typed NULL, more than one row raises), `IN` / `EXISTS`, correlated ones decorrelated into a join; `UNION ALL` / `UNION` / `INTERSECT` / `EXCEPT` (TEXT dictionaries merged, numbers promoted as DuckDB does); `QUALIFY` and filtering on a window result; `SELECT` with no FROM. Equal to DuckDB; 21 of 22 TPC-H through `mx sql`. Refused by name: `WITH RECURSIVE`, `LATERAL`, correlated `NOT IN`, correlated subqueries outside WHERE or under OR, `INTERSECT` / `EXCEPT ALL`, `UNION BY NAME`. Every step's result is held in memory | F-159 |
| GPU shared memory | no kernel asks for more shared memory than the target has: the Python path's group kernel caps its table at 32 KB on an Apple GPU (8192 groups; 12288 on NVIDIA as before), and the pipeline kernel refuses a launch past `SMEM_LIMIT`. No change in answers; 10,000 groups on the GPU under Apple's plan equal the CPU | F-195 |

## What changed in 5e4cdb1 (since 5053a44), for apps

| Area | What you get | Ticket |
|---|---|---|
| Window functions | SQL `f(...) OVER (PARTITION BY … ORDER BY … [ROWS / RANGE frame])` in the select list and ORDER BY: `row_number`, `rank`, `dense_rank`, `percent_rank`, `cume_dist`, `ntile`, `lag` / `lead` (offset, default), `first_value` / `last_value` / `nth_value`, and `sum` / `count` / `avg` / `min` / `max` over a frame (running totals: `ROWS UNBOUNDED PRECEDING`; moving averages: `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW`). With WHERE and GROUP BY, on a Table, Frame, Parquet or CSV scan. Builder: `Window().partition_by(…).order_by(…).rows(a, b)`, `Agg.over(w)`, `Query.window(...)`, `compute_windows(q, device)`. Equal to DuckDB; exact INT / DECIMAL; CPU only (AUTO says so). Not yet: filtering on a window result (QUALIFY / a subquery), windows bigger than memory | F-168 |
| Sort bigger than memory | `Frame.permutation(c, descending, job, budget=, spill_dir=)` and `permutation_job(..., budget=, spill_dir=)`: past the budget, sorted runs go to disk and merge into the same order as before (3M rows of a streamed file: 169 MB to 64 MB at a 16 MiB budget). Pass `budget` (e.g. a quarter of the memory the app allows itself; the default is a flat 2 GiB) and `spill_dir` (the app's temporary directory). SQL `ORDER BY` spills the same way (`PlannerOptions.sort_budget` / `spill_dir`), and `sql_to_parquet(cat, sql, path, options)` sorts straight into a Parquet file without holding the result. Spill files are unlinked as soon as they open | F-167 |
| GPU memory | the GPU never fails on data bigger than it: a `DeviceTable` over the GPU's free memory holds no device copy (`is_resident()` False); `compute(GPU)` then runs row ranges that fit and merges them, `compute(AUTO)` answers on the CPU with a `fallback_reason` naming the sizes; a failed GPU allocation reruns that piece on the CPU (same answer), and AUTO remembers it until much more memory is free. `QueryResult.pieces` / `gpu_pieces`; `device` says CPU when any piece ran there. On the Mac this matters most on small-memory machines | F-189 |

## What changed in 5053a44 (since 8017543), for apps

| Area | What you get | Ticket |
|---|---|---|
| Save memory | `write_parquet(..., budget=)` and `Frame.save(..., budget=)` (bytes; 0 = default): a parallel save keeps the pages it holds in flight, the coder's working memory included, within the budget, for any data. Defaults: 256 MB for a table, 57 MB for a streamed Frame. Same file bytes for any thread count and budget. Streamed saves keep 8017543's speed and peak; the fixed 4-thread cap is gone. A tiny budget (1-4 MB) saves one page at a time: a 125 MB table takes ~0.35 s instead of 0.06 s | F-188 |

## What changed in 8017543 (since 2585dc9), for apps

| Area | What you get | Ticket |
|---|---|---|
| Parallel saves | `write_parquet(..., threads=n)` (0 = every thread the build has) compresses a row group's pages on several threads: Linux always; macOS / iOS with `-D MXFRAME_JOB_THREADS`; otherwise serial. The file is byte-identical for any thread count. 2M rows, zstd with the Mojo codec: 0.60 s before, 0.36 s on one thread, 0.06-0.07 s on 12 threads (libzstd: 0.22 s / 0.035 s). Saves of a streamed Frame used at most 4 threads (replaced by a byte budget in 5053a44) | F-187 |
| Pages | big columns are written as ~1 MB data pages (every build, Linux included): pyarrow and DuckDB read them, and files open faster (0.040 s to 0.022 s) because pages decode in parallel | F-187 |
| Size | zstd files within 0.2% of libzstd level 3 | F-187 |

## What changed in 2585dc9 (since 3d2455d), for apps

| Area | What you get | Ticket |
|---|---|---|
| Spread | `stddev` / `stddev_pop` / `variance` / `var_pop` (builder; SQL also `stddev_samp` / `var_samp`), per group and whole column, on a Table, Frame or scan. Float64 result, NULLs skipped, the sample forms NULL below two values; exact on data where DuckDB loses digits (1e15 + small numbers), equal to DuckDB elsewhere; NaN / inf among two or more values raises `out of range`, as DuckDB | F-177 |
| ISO weeks | `EXTRACT(week FROM d)`, `EXTRACT(isoyear FROM d)`, `week()`, `weekofyear()`, `isoyear()`: expressions and group keys, equal to DuckDB for every day 1900-2100 | F-177 |
| Quantile bins | SQL `quantile_bin(x, n)` as a GROUP BY key (bins 1..n, n at most 32; edge k is the value at sorted place ceil(k N / n) of the non-NULL values the WHERE keeps; ties share a bin; equals `ntile(n)` when n divides N). Builder: `q.quantile_bins(e, n)` then `.key(e)` / `.bins()` in `group_by` (bins 0..n-1). Elsewhere in a query it raises by name | F-177 |
| GPU and these | AUTO answers spread aggregates and quantile bins on the CPU (`fallback_reason` says so); `compute(GPU)` raises with the same words | F-177 |
| Streamed CSV | `scan_csv(path, CsvOptions(...))` (or `scan_csv_job`, with progress and cancel) opens a CSV of any size: queries, SQL (`Catalog.add_scan`), `Frame(scan)` edits and `save` (writes Parquet) work as on a Parquet scan. A 1.4 GB file peaks at ~60 MB (read_csv: 2 GB); opening ~165 MB/s on one thread. `scan.csv_report()` (per column failed / empty counts, ragged rows) and `failures()` (row, column, text, why): a failed cell is a NULL, so check the report. Kinds come from a sample cut at `block_bytes`: with very long rows they can differ from `read_csv` | F-178 |
| Undo for formulas | adding, replacing and removing a formula are steps of the Frame's one undo log; new `Frame.redo()` (every step kind; a new edit clears redo) and `Frame.remove_formula(m)` (refused while another formula reads it); `edits` advances on each step, undo and redo. Drop the app's second undo log | F-179 |
| Compressed Parquet everywhere | builds without C codecs (macOS, iOS) now write and read ZSTD and SNAPPY with codecs written in Mojo (no library to ship, no licence to carry); the same file size as libzstd level 3, zstd save ~2x slower than Linux's C library, open about the same; checksums verified. Linux is unchanged (libzstd / libsnappy). Write `.omx` files with ZSTD on every platform | F-180 |

**Not yet:** formula columns in `range_aggregate` / `read_block` still raise; saving a CSV scan as CSV; parallel zstd
compression; spread aggregates on the GPU.

## What changed in 3d2455d (since 23edc59), for apps

**Building the Mac app with the GPU.** The same `omxcore.mojoc` (built on Linux) gives the Apple GPU path when
the *app* is compiled for macOS with an Apple GPU as the accelerator; the switches resolve at the app's build, not the
package's. Metal kernels are compiled with Apple's Metal toolchain, so that build runs **on a Mac** (or a macOS runner):

```sh
xcodebuild -downloadComponent MetalToolchain                 # once per machine
mojo build -I vendor app_entry.mojo -o <app>                 # native build on Apple Silicon: the GPU is on, no -D flag
mojo build -I vendor --target-accelerator metal:1 ...        # only where MAX cannot name the GPU (GitHub's virtual one)
mojo build -I vendor -D MXFRAME_NO_GPU ...                   # opt out: CPU only, as before
```

A Linux cross-build of the Mac app (`--target-triple arm64-apple-macosx...` without a Metal accelerator) still has no
GPU and needs no flag: it links exactly as before (`scripts/check_mac_link.sh`).

| Area | What you get | Ticket |
|---|---|---|
| Apple GPU | `gpu_available()` is True on an Apple Silicon Mac built as above, and `DeviceTable` / `compute(GPU)` / `compute(AUTO)` run the kernels on Metal. `gpu_device()` returns "<name> · <api>" for the status bar ("Apple M4 · metal"; "" without a GPU); `QueryResult.device` (CPU / GPU) says where a query ran, `fallback_reason` why AUTO used the CPU | F-182 |
| How it runs there | Metal has no float64 and no general 64-bit atomics, so on an Apple GPU the kernel's word is an **Int64** and only the **atomic-free modes** run: 0 / 3 (few groups), 4 (the block's table in 32 KB of shared memory, up to 4,096 (slots + 1) x groups) and 5 (many groups: the table in device memory, one owner thread per group, up to 2^20). Plans are sized to **32 KB** of shared memory; big programs use 64-thread blocks. Device addresses become pointers in Metal's device address space (`addrspace(1)`). Results are exactly the CPU's (int64 totals, the same overflow error) | F-183, F-184, F-185 |
| When AUTO answers on the CPU (Mac) | a FLOAT column or float arithmetic (`mean` of INT / DECIMAL still runs on the GPU: its division is on the host); more than 2^20 (slots + 1) x groups; a program whose registers do not fit 64-thread blocks in 32 KB (more than ~44 aggregates); count_distinct / percentile / top_k (as on NVIDIA); a table under `AUTO_GPU_MIN_ROWS` (1M rows). `compute(GPU)` raises in those cases instead, with the same reason | F-184 |
| NVIDIA | unchanged: the kernels' PTX is byte-identical to 3a33697 (`ci/mac/nvidia_ptx.sha`), timings equal | F-186 |
| Name | `from omxcore import ...` (F-165). `mxframe.mojoc` is built one last time for code that still says `from mxframe` | F-165 |

Checked on GitHub's macos-26 runner (a virtual Apple GPU, M1 family: correctness only, not speed, not yet real
silicon): 12 orders queries on 1M rows equal pyarrow, including 5,000 groups and a 30-aggregate pivot
(`.github/workflows/apple-gpu.yml`, `scripts/check_apple_gpu.py`, `docs/goals/apple-gpu/report.md`).

## What changed in 23edc59 (since 321870f), for apps

| Area | What you get | Ticket |
|---|---|---|
| Background work | `compute_job`, `sql_job`, `read_csv_job`, `read_parquet_job`, `save_job`, `write_parquet_job`, `permutation_job` return a `Job[T]`: `poll()`, `wait()`, `result()`, `cancel()` (answered at once; a cancelled save keeps the old file), `progress()` (done / total / stage), an optional `JobDone` callback. On Linux each job runs on its own thread; on macOS / serial builds jobs run inside the start call unless the app builds with `-D MXFRAME_JOB_THREADS` and allows `_pthread_create _pthread_join _pthread_detach _pthread_attr_init _pthread_attr_setstacksize _pthread_attr_destroy _sched_yield` (check `JOBS_ARE_THREADED`) | F-175 |
| Ranges and blocks | `range_aggregate(source, [order,] cols, a, b)` (count / sum / mean / min / max per column, exact at each scale) and `read_block(source, [order,] cols, a, b, values, nulls)` (rows a..b of n columns in one call, column-major) over a Table or Frame (edits, merges, over a file); formula columns raise for now | F-176 |

## What changed in 321870f (since 1889b7f), for apps

| Area | What you get | Ticket |
|---|---|---|
| SQL | `sql(catalog, text) -> SqlResult` (an owned `Table`): SELECT, WHERE, GROUP BY (several keys, NULL is its own group), HAVING, ORDER BY (several keys, NULLS FIRST / LAST), LIMIT / OFFSET, `SELECT *`, DISTINCT, EXPLAIN [ANALYZE] | F-150, F-152 |
| Aggregates | sum, count, avg, min, max, count(DISTINCT), median, quantile_disc per group | F-148, F-150 |
| Expressions | CASE, COALESCE, NULLIF, CAST / TRY_CAST, long IN lists, text functions once per word (lower, upper, trim, substr, replace, LIKE / ILIKE, contains, `\|\|`), dates (EXTRACT year / quarter / month / dow, date_trunc, ± INTERVAL, date differences) | F-156 |
| Joins | inner, LEFT, SEMI, ANTI; IN / EXISTS / NOT IN subqueries; 3-9 tables with a cost-based planner | F-153..F-155 |
| Types | FLOAT64 columns (Parquet DOUBLE / FLOAT, CSV floats); `CsvOptions(decimal_columns=)` for exact money | F-157 |
| `mx` | command line for agents and scripts: SQL in, JSON / CSV / Parquet out | F-151 |

**API changes to check when you update:** `Frame` base nulls come from validity bits; `median` is interpolated (scale
+1); `word_count` / `text_of` / `code_of` raise; CSV columns with more than 4 decimal places read as FLOAT unless
named in `decimal_columns`.

**Mac link (apps that build for macOS):** allow these libSystem symbols in the app's stub and allowed list:
`_sysconf _memset_pattern16 _strtod _strtod_l _newlocale _freelocale` (the package's `scripts/check_mac_link.sh`
prints the current list).

## Asking for something
Open an issue on this repo with the label **`omxplorer`** (or your app's name): what you need, what you do meanwhile,
and a repro for a bug. Needs that are generic go into the engine; app-specific ones are built as plugins outside it.

## History
| Stable | Date | Highlights |
|---|---|---|
| `834ef9a` | 2026-10-29 | SQL CTEs, views, subqueries, set operations, QUALIFY; GPU shared memory bounded by the device |
| `5e4cdb1` | 2026-10-28 | window functions; sort bigger than memory (SQL and `permutation`); the GPU never fails on data bigger than it |
| `5053a44` | 2026-10-27 | a save's memory follows a byte budget (`budget=`) |
| `8017543` | 2026-10-26 | parallel zstd saves, faster Mojo encoder, ~1 MB pages |
| `2585dc9` | 2026-10-26 | stddev / variance, quantile bins, ISO week; streamed CSV; undo for formulas; compressed Parquet on the Mac |
| `3d2455d` | 2026-10-25 | Apple GPU (Metal): Int64 word, atomic-free modes, 32 KB; `gpu_device()`; the omxcore name |
| `23edc59` | 2026-10-20 | background jobs (cancel, progress), range aggregates and block reads |
| `321870f` | 2026-10-19 | SQL, joins, planner, expressions, FLOAT64, `mx` |
| `1889b7f` | 2026-10-19 | GPU for streamed files |
| `7cbcc04` | 2026-10-18 | editing streamed files, count distinct / median / top-k, faster streaming |
| `581250b` | 2026-10-18 | faster Parquet decode |
