# Stable engine version for apps

Apps that vendor the Mojo package (`omxcore.mojoc`, called `mxframe.mojoc` until the F-165 rename) read this file to know which commit to build from.
Scripts read the one-line file `STABLE` (just the commit). When a new version becomes stable, an issue labelled
`mxframe` (the label keeps its name) is also opened on the app's repo (abhisheksreesaila/mxapp), and the app's scheduled check updates itself.
Only commits listed here are stable: every ticket in them passed adversarial review, and the full test suites were
green. Anything newer on `main` may still be in review: don't vendor it.

## Current stable

**`3d2455d`** (2026-10-25): omxcore on Apple GPUs (Metal), and the first stable after the omxcore rename.

```sh
# build the package from a clean copy of the stable commit (never from a working checkout), on Linux as before
git -C <omxcore> archive 3d2455d | tar -x -C <scratch>
cd <scratch> && pixi run bash scripts/build_mojo_pkg.sh      # -> build/omxcore.mojoc (and build/mxframe.mojoc, the old name, one last time)
cp build/omxcore.mojoc <app>/vendor/ && echo 3d2455d > <app>/vendor/OMXCORE_COMMIT
```

**Building the Mac app with the GPU (new).** The same `omxcore.mojoc` (built on Linux) gives the Apple GPU path when
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

## What changed since the previous stable (23edc59), for apps

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
| `3d2455d` | 2026-10-25 | Apple GPU (Metal): Int64 word, atomic-free modes, 32 KB; `gpu_device()`; the omxcore name |
| `23edc59` | 2026-10-20 | background jobs (cancel, progress), range aggregates and block reads |
| `321870f` | 2026-10-19 | SQL, joins, planner, expressions, FLOAT64, `mx` |
| `1889b7f` | 2026-10-19 | GPU for streamed files |
| `7cbcc04` | 2026-10-18 | editing streamed files, count distinct / median / top-k, faster streaming |
| `581250b` | 2026-10-18 | faster Parquet decode |
