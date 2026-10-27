# omxcore: the Mojo API omxapp needs

**For:** the omxcore project. **From:** omxapp (the framework) and omxplorer (the Numbers-style data app), as of goal 15 (2026-10-17).
**Replaces:** the goal 8–9 wish list that was here (items 1–22, kept in git history). Everything in it is folded in below.

## Why
Today omxplorer runs on a **stand-in engine** inside this repo: `mojoui/frame.mojo` (columns, edits, formulas, sort),
`mojoui/frame_gpu.mojo` (GPU group-by) and two pyarrow shims (`mojoui/csvio.mojo` reads CSV, `mojoui/omxfile.mojo`
reads and writes Parquet). It is checked against an independent pass over the CSV and works to 100M rows
(`docs/scale.md`). But:
- the shims need Python, so Mac, iPhone and Android builds can't open a file;
- it's capped at 16 columns, int32 on the GPU, no nulls, one table;
- it's the app's own engine, not the product's.

The switch to omxcore should be **one layer**: the components (DataGrid, PivotTable, ChartView, KpiView, the
sheet and report pages) don't change. Only `Frame`, `GpuFrame`, `group_sum`, `read_csv` and `save_omx/open_omx`
change, and they're shaped like the API below on purpose.

## Hard requirements
- **A Mojo package (`.mojopkg`) with no Python or libpython**, so it links into the GTK, AppKit, UIKit and Android hosts.
- **Exact integer arithmetic:** money and quantities are scaled int64; sums are exact 64-bit (the checks compare to the cent).
- **Raises with readable messages** ("column `cost` not found", "row 41,207: `12.x` isn't a number").
- **GPU optional:** every operation has a CPU path giving identical results; the GPU is used when there is one.

## 1. Tables and columns

```mojo
from omxcore import Table, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL

var t = Table.empty()                              # "New blank sheet"
t.add_column("revenue", DECIMAL, scale=2)          # values stored as int64 hundredths
t.add_column("region", TEXT)                       # dictionary-encoded: codes + a string table (omxcore stores the codes as Int64, like every value)
t.rows(); t.columns(); t.name(c); t.kind(c); t.scale(c); t.index("region") -> Int  # -1 if absent
t.get(c, row) -> Int64                             # the stored value (a code for TEXT, days for DATE)
t.is_null(c, row) -> Bool
t.dictionary(c) -> Span[String]; t.text_of(c, code) -> String; t.code_of(c, "West") -> Int  # -1
```

| Need | Today (stand-in) | Why |
|---|---|---|
| Kinds INT, DECIMAL(scale), TEXT (dictionary), DATE (days since 1970), BOOL; and FLOAT (float64 bits in the value slot; `append_float`, `get_float`; F-157, the section below) | INT, DECIMAL, TEXT, DATE; bools import as 0/1 | the Import step and the format inspector branch on kind |
| **Any number of columns** | 16 (`MAX_COLS`, a SIMD row vector) | wider CSVs lose columns past 16 today |
| **Nulls per column** (a validity bitmap) | none: empty cells read as 0 or "" and are only counted | Delete on a number cell; honest averages and counts |
| Column-major, contiguous int64 buffers the app can borrow as `Span[Int64]` | `List[Int64]` per column | the grid reads a screenful at a time; the GPU upload reads whole columns |
| Shared, read-only base data across sessions (web: many tabs, one table in memory) | `share()` returns an address; `release_columns()` empties it | the web host serves one process to many sessions |

### FLOAT64 (F-157)

FLOAT is a full column kind: the float64's bits in the Int64 value slot, the validity layout unchanged (`t.append_float(c, x)`, `t.get_float(c, row)`;
`Frame.set_float`, `Frame.get_float`). Bits are kept as given (-0.0, a NaN's payload, subnormals) in storage; the rules below make them one value where a
query compares, groups or counts them.

| Where | Rule |
|---|---|
| **Parquet** | DOUBLE is read as it is, FLOAT (32-bit) is widened to float64 exactly (its column has `scale == F32_COLUMN` (32), see "a 32-bit column" below); PLAIN, dictionary and BYTE_STREAM_SPLIT pages (BYTE_STREAM_SPLIT is read on INT32, INT64 and fixed-length DECIMAL columns too). Written as DOUBLE (PLAIN, no statistics): every bit survives a round trip. A float column's footer min / max are never read (`ParquetScan.column_range` says "unknown" for a FLOAT), so NaN in a footer cannot reach the planner's dense-group logic; null counts are used as for any column. |
| **CSV** | A column is FLOAT when its first `infer_rows` rows show a number with an exponent (`1e5`, `2.5E-3`), `nan` / `inf` / `-inf` / `infinity` (any case) beside numbers, **more than 4 decimal places** (trailing zeros do not count), or a number too wide for int64 at its scale (DuckDB reads those as DOUBLE too); or when named in `CsvOptions(float_columns=[...])`. Up to 4 places stays an exact DECIMAL (scale 2, or 4). Five or more places are a float by default because an exact DECIMAL(6) has no wider integer behind it: `sum(rate * qty)` over a six-place rate would raise "passed the int64 range" where DuckDB answers. The exact path is an opt-in, `CsvOptions(decimal_columns=[...])`: the named columns are an exact DECIMAL at the scale their first rows need (2, 4, or their number of places, up to 18); every value of the first rows is checked to fit int64 at it and the read raises, naming the column, when one is not a plain number or does not fit (a later value that does not fit is a failed cell); arithmetic on such a column still raises "passed the int64 range" when a total does, and `rounded` counts values with more places than the scale. The value of a FLOAT cell is the correctly rounded float64 (an exact fast path, `strtod_l` in the C locale for the rest, whatever locale the host set); every NaN is the one positive quiet NaN; a cell that is not a number, or is past 1.8e308, is a failed cell (null). Past the window an exponent in an INT or DECIMAL column is a failed cell. |
| **Comparisons** | DuckDB's order: NaN equals itself and is above every number (+inf included); -0.0 equals 0.0; NULL is unknown (SQL) or false (builder). |
| **Arithmetic** | INT and DECIMAL next to a FLOAT become float64 as DuckDB casts them: `float(v) / 10^scale` (an integer past 2^53 raises: it is not exact as a double). `+ - *` and `/` (always a division of doubles, IEEE: `x / 0` is inf, -inf or NaN); `%` on a FLOAT is not supported (`//` is a division with NULL for a zero divisor, F-156). A literal with an exponent is a DOUBLE; a decimal literal next to a FLOAT is read as `float(v) / 10^scale`. |
| **SUM / AVG** | float64 in a defined order, the kernel's reproducible mode (F-064): within one run, each 65,536-row morsel adds its rows in order and the morsels' totals are added in order, so the bits do **not** depend on the number of threads (`Query.threads(n)`, `PlannerOptions.threads`), the morsel setting or the block size (tested). They do depend on the **plan and the source layout**, because those change which additions happen in which order: (1) dense and hashed groups give different last bits for the same query (a dense run keeps up to 4 partial sums per group below 1,025 groups, a hashed run one per morsel: `PlannerOptions.group` forces one); (2) a table read whole is one run over its rows, whereas a Parquet scan is one run per row group folded in file order (morsels restart at each group's first row), so `read_parquet` + a query and `scan_parquet` + the same query can differ in the last bits (the exact integers, counts, min, max and medians never do); (3) a Frame is its shared rows, then the changed rows added one by one; (4) the SIMD width of the machine sets the lanes of a global sum, so two machines may differ. The same query on the same data, plan and machine gives the same bits every run. A scan with a float sum always runs row group by row group, never by row-range morsels (those are cut by the number of cores). The tests compare with DuckDB to 1e-12 relative. AVG is the float sum over the count. |
| **MIN / MAX** | NaN is the largest value: MAX is NaN when the group holds one, MIN is NaN only when it holds nothing else (a hidden count of the NaNs, so the kernel never compares one). |
| **Group keys** | SQL only (the builder's key is a dense integer): hashed, never dense. -0.0 and 0.0 are one group, every NaN is one group (reported as the canonical NaN), NULL is its own group; ORDER BY puts NaN after +inf and NULLs last. `PlannerOptions(group=GROUP_DENSE)` on a FLOAT key raises and says why. |
| **count distinct, median, quantile_disc, top_k** | on order-preserving keys (-0.0 = 0.0, one NaN above +inf): `median` is the middle value, or for an even count `lo * 0.5 + hi * 0.5` (DuckDB's interpolation); `quantile_disc` is an actual value. `QueryResult.floats(i)` gives a float aggregate's values as Float64 (`is_float(i)`); `values(i)` and `top(i, g)` give their bits. |
| **Frame** | `set_float`, `set_text` (`"1e3"`, `"nan"`, `"-0.0"`, `""` is null), `append` strings; a formula with a float (an exponent literal, a FLOAT column, `/` of a FLOAT) is a FLOAT formula column (`+ - * /`, `min`, `max`; `//` and `%` are refused); `permutation` orders floats (NaN last, nulls last either way); `bake` and `save` write DOUBLE. A base cell is null by its validity bit only, so any INT / DECIMAL / DATE / FLOAT value is data, including 9221120237041090561 and -9223372036854775807. Edited and appended cells mark a null with the NaN `0x7FF8000000000001` (`NULL_EDIT`; `set` refuses it as a value, `set_float` stores that NaN as the canonical NaN), and the changed rows a query reads (`Overlay.delta`) carry their own `null_marker`, which differs from it when a changed row holds it as data. |
| **A 32-bit column** | DuckDB's REAL. A Parquet FLOAT column is read widened to float64 (`scale == 32`), and SQL gives it DuckDB's type rules: REAL next to REAL, INT or DECIMAL is **REAL** (a literal or DECIMAL column is cast to 32 bits, so `l_discount <= 0.07` keeps the rows holding 0.07, and `f = d` compares in 32 bits); REAL next to a DOUBLE (a double column, an exponent literal such as `7e-2`) is **DOUBLE**; one DOUBLE in an `IN` list makes the whole list DOUBLE. REAL `+ - * /` and negation stay REAL (each result rounded to 32 bits, as DuckDB computes it); `sum` and `avg` of REAL give DOUBLE, `min`, `max`, `median` and `quantile_disc` give REAL. Forms not modelled raise by name: REAL meeting a non-literal INT or DECIMAL, or REAL arithmetic, *after* aggregation (`max(f) + min(f)`), and a DECIMAL with more than 10 places next to REAL. The builder has no 32-bit rule: it reads the column as a double, with `lit_float(x)` (a double) and `lit_f32(x)` (rounded to 32 bits). |
| **GPU** | A query that reads a FLOAT column or does float arithmetic runs on the CPU: `compute(GPU)` raises, AUTO says why in `fallback_reason` (a float64 sum has one defined order there). |
| **Builder** | `col("x") * 1.5 + col("y") / col("n")`, `lit_float`, `lit_f32`, `Query.threads(n)`; a float group key is refused (use SQL). Aggregates: `sum`, `mean`, `min_of`, `max_of`, `count_distinct`, `median`, `percentile`, `top_k`, `stddev`, `stddev_pop`, `variance`, `var_pop`; equal-count bins as a key: `Query.quantile_bins(e, n)` then `QuantileBins.key(e)` (bins 0 .. n-1, SQL's `quantile_bin(e, n)` minus one; see Quantile bins); read them with `r.floats(i)`. |

## 2. Reading and writing files (the biggest gap: Python today)

```mojo
var r = read_csv(path, options=CsvOptions(max_columns=..., infer_rows=100_000))
r.table; r.rows; r.seconds
for c in r.columns: c.name, c.kind, c.scale, c.empty, c.distinct, c.sample   # the Import step's preview
write_parquet(path, table, metadata: String)    # metadata: one JSON string under a key ("omx")
var p = read_parquet(path); p.table; p.metadata -> String
var shared = r.take_table().share()             # moves the table out; the report / metadata stay readable
```

| Need | Detail |
|---|---|
| **CSV/TSV reader** with type inference | integers → INT; decimals → DECIMAL with scale 2 if every value has ≤ 2 places, else 4; an exponent, more than 4 places or nan / inf → FLOAT (F-157); ISO dates and timestamps → DATE; true/false → BOOL; else TEXT. Quoting, embedded newlines, BOM, `\r\n`. |
| Per-column import report | empty cells, distinct count (text), the first few raw values, rows that failed to parse (row number + text) |
| **Parquet compression** (F-180) | `write_parquet(..., compression=ZSTD)` (the default; also `SNAPPY`, `UNCOMPRESSED`) compresses every page on **every target**. On Linux with the C libraries by default; on macOS, iOS and every other target, and on Linux with `-D MXFRAME_NO_CODECS`, with the codecs written in Mojo (`omxcore/codec.mojo`): nothing to link, nothing to ship, no `ZSTD_*` / `snappy_*` symbol, and `scripts/check_mac_link.sh` stays green. The files are standard zstd / snappy Parquet: pyarrow and DuckDB read them, and each build reads the other's files. On a 75 MB table the Mojo coder's files are the size of libzstd level 3's (4.65x for zstd, 2.39x for snappy); a zstd save takes 0.36 s on one thread against 0.22 s for libzstd (was 0.6 s against 0.27 s before F-187) and 0.06 s with every thread against 0.035 s, snappy about the same time as libsnappy, and an open about as long (numbers in `mojo/README.md`). **Threads (F-187):** `write_parquet(..., threads=0)` compresses the pages of a row group on every thread the build has (`threads=1`: the calling thread); a column of a row group is cut into data pages of about 1 MB, which the reader decodes in parallel, and the file is the same bytes for any thread count. A plain Mac build (no async runtime) stays serial; a Mac build with `-D MXFRAME_JOB_THREADS` fans out over pthreads, with no symbol its jobs do not already need; **Budget (F-188):** `write_parquet(..., budget=0)` and `Frame.save(..., budget=0)` bound the bytes a save holds in flight, not the threads. A page waits until what the pages in flight hold plus its own cost fits the budget; a page being made counts its body, its compressed bytes and its coder's scratch (12 MB for zstd, 256 KB for snappy, per 1 MB page; calibrated, it includes what the allocator keeps), a finished page waiting for its turn counts its compressed bytes; pages are written in file order as the oldest finish and freed. The oldest page always goes, so one page is the least. Defaults: 256 MB for a table (`WRITE_BUDGET`; the table is resident already, and a 2M-row save is as fast as unbounded) and 57 MB for a save over a scan (`STREAM_WRITE_BUDGET`: four pages being made, what F-187's cap of 4 threads allowed). Measured on a 125 MB table with 24 threads, random and compressible values, libzstd and the Mojo coder: the save adds no more than its budget (1 MB: about 1 MB; 4 MB: 1-2 MB; 16 MB: 2 MB; 64 MB: up to 47 MB; no bound: 220 MB). The file is the same bytes for any thread count and any budget; an app that has a memory budget passes a share of it. **A small budget makes a save slower**: fewer pages at once means fewer busy threads (a 1-4 MB budget on a 125 MB table: about 0.35 s against 0.06 s). The streamed save's peak is its decoded row group (about 101 MB on the test's 332 MB file with one thread), which the budget does not change: F-188 keeps F-187's peak and speed there and does not enlarge the margin. The Mojo reader is strict: a zstd content checksum is verified (xxHash64) and a mismatch is a corrupt file; a match may reach back no further than its frame's start and declared window; a page size that the input could not possibly hold is refused before any memory is allocated. Zstd dictionaries are refused on read (no Parquet writer uses them). |
| Speed | today 1M rows: 0.5 s; 100M rows: 106 s on a loaded machine (pyarrow, multi-threaded). Parity or better, streaming (no full-file copy in memory). |
| **Parquet read/write with key-value metadata** | the `.omx` document is one Parquet file; its JSON metadata holds formats, formulas, charts, pivots and the Report. Edits are baked into the columns on save (D-138). |
| **Bigger than RAM** (F-140) | `var scan = scan_parquet(path)`; `Query(scan).where(...).group_by(col("region"), groups=scan.word_count(scan.index("region"))).agg(...).compute()` decodes one row group at a time (only the columns the plan reads) and equals `read_parquet` + the same query exactly; TEXT codes are the whole-file read's (`scan.code_of(c, word)`); CPU only. `count_distinct` and `top_k` stream too, exactly, by merging their state per row group; `percentile` / `median` keep every non-null value (12 bytes a row), see §5 | Edit it with `Frame(scan)` (§3, F-147) |
| **A CSV bigger than RAM** (F-178) | `var scan = scan_csv(path, CsvOptions(group_rows=..., block_bytes=...))`: the same scan as `scan_parquet` (queries, SQL, `Frame(scan)` with edits and `save`), over a CSV or TSV; kinds from a sample, a failed-row report, memory bounded by a row group. Section 2b |
| Later | XLSX, JSON lines, Arrow IPC; the Arrow C Data Interface both ways (zero-copy hand-over) |

### 2b. A CSV bigger than memory (F-178, `csvscan.mojo`)

```mojo
var scan = scan_csv("sales.csv", CsvOptions(group_rows=262_144, block_bytes=16 << 20, infer_rows=100_000))   # a ParquetScan (alias CsvScan)
scan.rows(); scan.row_groups(); scan.columns(); scan.name(c); scan.kind(c); scan.scale(c); scan.is_csv()
var report = scan.csv_report()                        # what opening found: the import report of read_csv, for the whole file
report.rows; report.columns[c].empty / .failed_count / .failed / .rounded / .sample; report.ragged / .ragged_count; report.skipped
for f in report.failures(): f.row, f.column, f.text, f.reason; f.message()    # "row 41,207, column `qty`: `12.x` isn't a whole number"
Query(scan).where(...).group_by(col("region"), groups=scan.word_count(scan.index("region"))).agg(...).compute()
cat.add_scan("sales", scan); sql(cat, "SELECT region, sum(qty) FROM sales GROUP BY region")
var f = Frame(scan); f.set(...); f.save("edited.parquet", metadata)           # as section 3 (F-147): the edited rows stream out as Parquet
var job = scan_csv_job(path, options)                 # Job[ParquetScan]: the opening pass in the background (progress counts bytes, cancel between row groups)
```

A scan over a CSV **is** a `ParquetScan` (`scan_csv` returns it; `CsvScan` is another name for the type): everything section 2 says about a scan, section 3 says about a Frame over one, and sections 5 and 8 say about queries and SQL over one holds, because
the engine asks a scan for row groups and only the decoding differs. The notes of a SQL plan call it a "CSV scan".

| What | How it behaves |
|---|---|
| **Opening** | Reads the file once, a block at a time, and keeps an index: the header and kinds (from the first `infer_rows` rows, as many as fit in `block_bytes`, at least the header and one row), then for each **row group** its byte range, row count and null count per column. The pass also parses every cell once, for the report and the null counts. Nothing but the index and the report stays in memory. ~165 MB/s on one thread (a 326 MB file: 2.1 s, a 1.42 GB file: 8.5 s). `scan_csv_job` runs it in the background |
| **Row groups** | About `group_rows` rows (262,144), never more than `block_bytes` bytes (16 MiB), found at a row boundary with the quote state machine `read_csv` splits its chunks with (a newline ends a row only outside quotes), so a quoted field with newlines in it is never cut and the rows are exactly the whole-file reader's, whatever the block boundaries (tested with groups of one row and blocks of one byte). A row longer than the block makes its window grow (doubling), up to `max_row_bytes` (64 MiB): past that, a readable error ("row 1 is longer than ... an unmatched quote makes the rest of the file one row"), since one open quote would otherwise make the rest of the file one row. Blank lines are not rows |
| **Kinds** | `read_csv`'s rules and options (INT, DECIMAL, FLOAT, DATE, BOOL, TEXT; `float_columns`, `decimal_columns`, `max_columns`, `has_header`, `delimiter`), decided by the sample. **A DECIMAL's scale covers the whole column:** inferred 2, it becomes 4 when any later value has 3 or 4 places; the opening pass finds that and is run again at the wider scale, so a column's kind and scale never change under a query |
| **Sample caveat** | The sample is cut at `block_bytes`: with long rows (or a small block) the kinds come from fewer than `infer_rows` rows, so a scan and `read_csv` of the same file can infer different kinds. Compare with equal windows (a large `block_bytes`) when it matters. A cell that failed its kind is a null in the table and **only `csv_report()` shows that it failed** |
| **Failed rows** | As `read_csv`: a cell that does not fit its column's kind (a value past the sample) is a **null**, the rest of the row is read, and the report names it: row number (data rows from 1), column, the text (cut at 200 bytes) and why. The first 1,000 per column are kept, `failed_count` is the true number; `failures()` lists them all in file order. Rows with more cells than the header and rows that end in an unmatched quote are in `ragged`. Nothing about bad values raises. A file that is not UTF-8 raises ("is not valid UTF-8 (byte N)"; UTF-16 is named), as `read_csv` does |
| **Exactness** | A scan gives **exactly** what `read_csv` of the same bytes and options gives: every cell, every TEXT code (first seen in file order), the report; and a query, SQL, `Frame` edits and `save` over it give what the same call over that table gives (tested for 5 block shapes, with and without morsels, against `read_csv`, and SQL against DuckDB's `read_csv`). The two exceptions are the ones of a Parquet scan: a float sum is added row group by row group (the last bits can differ), and a failed cell of a column that had to widen its scale reads "isn't a number" |
| **Reading a group** | Maps the group's bytes, parses the columns the plan reads (the others are skipped, not parsed), unmaps. Peak: the group's columns (8 bytes a value) and its mapped bytes, per worker thread. The TEXT dictionary of a column is built the first time the column is read (one extra pass over the file for that column, in file order, as `ensure_words`) and stays in memory, as for Parquet: **a high-cardinality text column that is read costs memory in its distinct words**; a column no query reads is never touched (a SQL plan no longer builds the dictionary of a TEXT column it does not read: that was a full pass here, and a Parquet dictionary read before) |
| **Statistics** | A CSV has none beyond the null counts (exact, from the opening pass): INT and DATE keys are grouped by hash, not by dense slots, and `column_range` says "unknown" |
| **Memory** | **Bounded by the options, not the file.** Measured with `mojo/tools/csv_scan_probe.mojo`, a fresh process each, peak resident memory (file pages included), `SELECT region, count(*), sum(qty), sum(amount), min(day), max(day), sum(flag) ... GROUP BY region` over 7 columns, one of them a TEXT column of distinct 32-byte words that the query does not read, `group_rows` 262,144, 16 MiB blocks: a **326 MB** file (4.4M rows) 60.6 MB, a **1.42 GB** file (17.6M rows) 60.2 MB (opening 8.5 s, the query 7.8 s; equal to DuckDB's answer); the same 326 MB file read whole by `read_csv`: 2.06 GB. With `group_rows` 50,000 and 2 MiB blocks the 326 MB file peaks at 32 MB. The peak is the process (about 30 MB of runtime) plus, for each worker thread, one group's columns and its mapped bytes. `tests/test_csv_scan_memory.py` checks that files of 50 MB and 150 MB have the same peak, under 100 MB |
| **Edits and save** | `Frame(scan)` is section 3 unchanged (row groups through the chunk cache, `cache_limit` 128 MiB by default; `save` streams one row group in and one out, to **Parquet**; the file it reads may be replaced). `save` writes a TEXT column as the file's dictionary words, a DECIMAL as DECIMAL(18, s), a FLOAT as DOUBLE |
| **Not there** | Saving back to CSV; a CSV scan in the CLI (`mx` reads a `.csv` whole); the GPU is as for Parquet (`read_group_into` works; a scan runs on the CPU); a lone `\r` as a line end, thousands separators (as `read_csv`); a file changed while it is open is refused at the next read ("the file changed since it was opened": size only, as for Parquet); opening is one thread |

## 3. Edits, rows and undo (a session's overlay on shared data)

```mojo
var f = Frame(table)              # one session's view: its edits over the shared table
f.set(c, row, value); f.set_text(c, row, "West")     # set_text parses by kind; new words get new codes
f.append(values: List[String]) -> Int                # a new row, returns its number
f.remove(row)                                        # a tombstone; f.alive(row)
f.undo() -> Bool; f.redo() -> Bool                   # one log: sets, appends, removes, merges and formulas (F-179); a new edit forgets what was undone
f.edits -> Int                                       # edit counter: hosts re-aggregate when it moves (every edit, undo and redo, formula steps too)
f.edited(c, row) -> Bool                              # the cell holds a session edit (grids mark it)
f.word_count(c) -> Int                                # shared words + words this session added
f.changed_rows() -> List[Int]                        # rows that differ from the shared base
f.merge(c, codes, into) -> Int                       # fold text values into one (a cleaning step), undoable
f.bake() -> Table                                    # edits applied, for saving (refused over a scan: use save)
f.save(path, metadata="", compression=ZSTD)          # the edited rows to a new Parquet file (F-147; below)
```

| Need | Why |
|---|---|
| Sparse cell edits, appended rows and removed rows over a read-only base | many sessions share one 100M-row table |
| One undo / redo log covering sets, appends, removes, merges and formula columns (added, replaced, removed) | the toolbar's Undo and Redo |
| **Edits visible to GPU aggregations without re-uploading** | today: the GPU sums the base, then the CPU applies each changed row as an exact delta (`group_sum` in `frame_gpu.mojo`). omxcore can do this its own way (patch kernel, dirty columns), as long as results include edits. |

### Editing a file bigger than memory (F-147)

```mojo
var f = Frame(scan_parquet(path))      # the same API as Frame(shared): set, set_null, set_text, append, remove, merge,
                                       # formula, undo, edited, word_count, text_of, code_of, get, is_null, permutation
f.set_cache_limit(64 * 1024 * 1024)    # optional; 128 MiB by default
var r = Query(f).where(...).group_by(...).agg(...).compute()     # streams the file with the edits applied
f.save("edited.parquet", metadata)     # streams the edited rows out; the file may replace the one the frame reads
```

| What | How it behaves |
|---|---|
| **Exactness** | A query, `get`, `is_null`, `edited`, `permutation` and `save` give exactly what `Frame(read_parquet(path))` with the same edits gives (tested: random sets, nulls, typed text with new words, appends, removes, merges and formulas over a 13-row-group file, again after every undo). TEXT codes are `scan.code_of`'s; session words follow the file's dictionary, which is built on first use, so codes never move. |
| **`get` / `is_null`** | Read the row group that holds the row through a cache of decoded **column chunks** (one column of one row group, 8 bytes a value plus its null bits). **The bound is `cache_limit` = 128 MiB by default** (`set_cache_limit(bytes)`, `cache_held()`, `cache_loads()`); least recently used chunks go first (lookup, use and eviction are O(1)), and the chunk just read is always kept (so one chunk bigger than the limit still reads: the bound is `max(limit, one chunk)`). A screenful of rows of one group costs one decode per column shown, not per cell; a grid wider than `limit / chunk` columns decodes on every pass, so size the limit to `columns shown x rows per group x 8`. A formula cell reads only the columns it uses. |
| **Queries** | `Query(frame)` streams the file as `Query(scan)` does (row groups on the workers by blocks, or group by group), each group or block skipping the rows the frame changed or removed (one bit per file row, 12.5 MB per 100M rows), and the changed and appended rows are added once at the end, exactly as for a frame over a table. Merges, formula columns, count_distinct, median, percentile and top_k work too (text order includes the session's words). CPU only. |
| **`permutation`** | Streams the column one row group at a time but holds **O(rows) keys** (about 50 bytes a row: 20M rows peaked at 1.0 GB, 1.75 s). A formula column evaluates row by row through the cache. |
| **`bake()`** | **Refused** over a scan ("a Frame over a streamed file cannot be baked into memory: save it to a new file with `save(path)`"). |
| **`save(path, metadata="", compression=ZSTD)`** | Streams the edited rows into a temporary file and renames it over `path` (atomic; an error removes the temporary file). One output row group per input row group (less its removed rows; a group with none left is skipped), then the appended rows in groups of 1M rows; all the frame's columns of one group are decoded at once, so the memory is one row group's columns, the output group, the file's dictionaries and the edits. A column is written OPTIONAL when the file declared it so or an edit put a null in it. Merged spellings are written as the word they read as; each group's dictionary page holds all the column's words (as `write_parquet` does). Formulas are not stored (their texts go in `metadata`). Over a table, `save` writes `bake()`. |
| **Metadata kept** | `save` writes `metadata` under "omx" (the caller's JSON; the source's is not carried over) and **writes back the file's other key-value metadata** (`scan.extra_metadata()`). Arrow's own `ARROW:schema` key is not kept: it describes the source's types, which the saved file no longer has. |
| **Narrowed columns** | Only a loss of values is refused: reading holds a TIMESTAMP as its UTC day, so **`save` raises, before writing anything and for any path, the source's included**: "column `ts` is TIMESTAMP in the file and is held as a DATE: saving would lose the time of day. ... pass allow_narrowing=True to save anyway" (`scan.narrowing(c)`, "" for none). A change of type that keeps every value is written as held, without asking: DECIMAL(p, s) as DECIMAL(18, s), ENUM / JSON as text, a narrower or unsigned integer as int64 (`scan.type_change(c)` says which). Arrow's own schema key is not written back, so Arrow-only types (dictionary, large_string, a timezone) come back plain. |
| **Threads** | **A Frame is not thread-safe**: `get`, `is_null` and `text_of` on a scan frame update the chunk cache (and build dictionaries), so one thread at a time per Frame; give each thread its own `Frame(scan)`. (`Query(frame)` copies what it needs and reads nothing from the frame while it runs.) |
| **API change** | `Frame.word_count`, `text_of` and `code_of` now `raises` (a file's dictionary is built on first use and can fail to decode); `get`, `is_null`, `permutation` and `overlay` already did. |
| **Measured** (12-core, 20M rows, 20 row groups, 567 MB file, 5 columns, 16 MiB cache) | edits 0.8 s for 1,350 edits (a cold chunk per set); query over the edited frame 0.1 s, 70 MB peak; a grid walk (1 row in 997, `get`) 25 ms, 88 MB; `save` 3.1 s, 180 MB peak (a 1M-row group is 40 MB decoded); the process alone is 21 MB. With 250k-row groups the save peaks at 84 MB on a 332 MB file. |


## 4. Formula columns

```mojo
var m = f.formula("margin = revenue - cost")   # parses, type-checks, returns the new column's index
f.formula_text(m); f.depends(m, c); f.dependants(c) -> List[String]
f.get(m, row)                                  # evaluated
f.program(m) -> Expr                           # the expression with other formulas inlined, for aggregation
f.remove_formula(m)                            # the columns after it move up by one; refused while another formula reads it
```

- **Undo and redo (F-179):** `formula(...)` (adding, or replacing the formula of that name) and `remove_formula(m)` are steps of the same log as cell
  edits, so `undo()` / `redo()` interleave them with sets, appends, removes and merges in the order they were made. Undo restores the exact earlier state: the
  formula texts, their column positions, and the scale of every formula that read a replaced one; the values that depend on them follow. `f.edits` moves on
  each of these steps and on every undo and redo. A refused formula (cycle, bad text, a name that is a stored column) adds no step. Any new edit, formula steps
  included, empties what `undo` took back (`redo()` is then False). Works the same over a table, an edited frame and a Parquet scan frame.

- Column expressions, never cell addresses: `+ - * / // %`, `min`, `max`, unary minus, constants, parentheses.
- **Decimal-aware:** `revenue / 2` keeps revenue's scale; `revenue * qty` rescales; division rounds the stand-in's way (pre-scaled numerator).
- Formulas may use formulas; aggregations over them must not materialise them (the stand-in inlines with `flatten`).
- Later: `year()`, `month()`, `day()` on dates; `if`; text functions; column plugins (embeddings, classifiers).

## 5. Aggregation (the engine's hot path)

```mojo
var q = Query(f).where(col("region") == code)             # filters: ranges and equality on any column, ANDed
              .group_by(col("carrier"), groups=n)            # a key expression with a known dense group count
              .agg(sum(col("revenue")), count())
var r = q.compute(device=AUTO)    # r.sums: List[Int] (index = key), r.counts, r.milliseconds
```

What omxplorer runs today, all through one call, `group_sum(frame, gf, key, value, ranges, groups)`:

| Caller | Key | Value | Filter |
|---|---|---|---|
| Chart on the sheet (`sheetpage.compute`) | a text column's code | a value column or formula | none |
| Pivot | `row_code × n_cols + col_code` (`pivot_program`) | value column | none |
| Report Sum / Average / Count tiles | constant 0 | value column | the Report filter (`region == West`) |
| Report chart tiles | a text column | value column | the Report filter |
| Column header histograms, status bar sum | a binned expression / constant | the column | none |

| Need | Detail |
|---|---|
| Group by an **integer key expression** with a known small group count (dense output, missing groups = 0) | covers text columns, pivots (combined keys) and bins |
| Aggregates | sum (exact int64), count, count(e), min, max, mean, **count distinct, median / percentile, top-k** (below) |
| Filters | `lo ≤ col ≤ hi` and `==` on any column (up to 8 today), ANDed; next: `is_in`, OR, nulls |
| **Includes the session's edits and formula columns** | §3 and §4 |
| Speed | today, 100M rows on the GPU: 19 ms for a sum, under 1 ms at 1M |
| Next | several group-bys in one pass (`compute_many`); reuse a compiled query with new constants (slider drags) |

### Count distinct, percentiles and top-k (F-148)

Aggregates that need more than a running number per group, exact, over a table, a Frame with edits or a Parquet scan:

```mojo
var r = Query(f).where(col("day").between(0, 364))
              .group_by(col("region"), groups=n)
              .agg(count_distinct(col("customer")),          # r.values(0)[g]   distinct non-null values
                   median(col("revenue")),                   # r.values(1)[g] (scale + 1)   r.present(1, g)
                   percentile(col("revenue"), 0.9),          # r.values(2)[g]   (the value itself, unscaled)
                   top_k(col("revenue"), 3),                 # r.top(3, g)      List[Int], largest first
                   sum(col("revenue")))                      # plain aggregates sit beside them, one pass
              .compute()
```

| Aggregate | Meaning | Result |
|---|---|---|
| `count_distinct(e)` | the number of different non-null values of `e` in the group (text: different words; a merged spelling counts once) | `r.values(i)[g]`, 0 for a group with none |
| `percentile(e, p)` (discrete; `percentile_cont`, interpolating, is a follow-up) | `p` in 0..1: the value at index `max(ceil(p * n) - 1, 0)` of the group's `n` non-null values sorted ascending. This is DuckDB's **`quantile_disc`** (checked against it), so the answer is always an actual value of the column, in the column's stored units (a DECIMAL(2) keeps its scale 2), and exact | `r.values(i)[g]`; `r.present(i, g)` is False for a group with no non-null value (its value is then 0) |
| `median(e)` | DuckDB's **`median`** (`quantile_cont` at 0.5): the middle value, or for an even count the mean of the two middle values, **exact**: the result is the stored value times 10, i.e. at the column's scale plus one decimal place, so [1, 2, 3, 4] gives 25 (2.5), and 5 * (a + b) never rounds. A median whose middle passes about 9.2e17 does not fit one more place in int64 and raises. A **text** column has no mean: its median stays discrete, the lower middle word (a code), unscaled | `r.values(i)[g]` as above |
| `top_k(e, k)` | the `k` largest non-null values of the group, repeats kept (`ORDER BY e DESC LIMIT k`; DuckDB's `max(e, k)`); fewer when the group has fewer | `r.top(i, g)`: `List[Int]`, largest first |

- **Shape.** Distinct counts and percentile values are a number per group like the other aggregates (`values(i)`, `present(i, g)`). Top-k is a list per group (`top(i, g)`) because its size varies; it holds values, not row numbers: ties make "which row" ambiguous, and an app that wants the rows filters on `value >= top(i, g)[k-1]`. Several aggregates over the same expression share one pass over its values.
- **Nulls** are ignored, as in SQL: a null value is not counted, not ranked, not listed. A row whose group is outside `0 .. groups-1`, or null, or that a filter drops, is left out as for every aggregate.
- **Text** columns use their codes for `count_distinct`. For `percentile`, `median` and `top_k` the order is the words' alphabetical order (ASCII case ignored, bytes breaking ties: the order of `permutation` and "apple" before "Zebra"), not the codes'; the values come back as codes (`text_of`). They need the plain column: a text column inside an expression raises. In a Frame the order includes the words the session added, and merged spellings are one word. **This differs from DuckDB's default**, which orders `VARCHAR` by bytes ("Zebra" before "apple"); DuckDB's `NOCASE` collation is not honoured by its `quantile_disc` / `max(e, k)`, so the tests compare against `lower(s)`. For words that differ only in case the order is the bytes', as in `permutation`.
- **Algorithms** (all on every core, `holistic.mojo`). The query kernel's per-block hook writes the (group, value) pairs of the selected rows, packed; then: *count distinct* is a hash set of (group, value) pairs in 4 x cores hash partitions (scatter the pairs, then each partition inserts alone: no locks), tables sized from a HyperLogLog sketch of the pairs and probed with software prefetch; the first window is counted in tables that live one task, only a second window (a scan's next row group, or a table over 20M rows) turns them into persistent tables. *Percentile* groups the values with a parallel counting sort and **selects** instead of sorting: three-way quickselect for small groups, a parallel most-significant-byte radix select for groups of 262,144 values or more (two passes for values in a small range, eight at most). *Top-k* is a bounded min-heap of `k` values per group (one compare rejects most rows). When top-k is the only aggregate reading its values and the groups are few, the kernel's own hook offers each value to a heap private to the worker, so no pairs are written at all (20M rows, 8 groups: 17 ms, a plain sum is 15 ms); otherwise private heaps per core over the pairs when groups are few, and the pairs scattered by group when many.
- **Streamed scans.** Distinct and top-k stay exact by merging state per row group (the hash set; the heaps), so they use memory for their state, not the file. **Percentiles need every value:** the state is the non-null (group, value) pairs of the whole file, 12 bytes a non-null row, and the answer is computed from a grouped copy (8 bytes a row more): a peak of about 27 bytes per non-null row measured (19M rows: 520 MB over the scan's 77 MB; a 100M-row column: about 2.7 GB). Distinct state is about 35 bytes per distinct (group, value) pair measured (19M distinct pairs: 680 MB), however many row groups; top-k is `k` words a group (`groups * k` may not pass 268M), nothing per row.
- **Tables and frames** run in windows of up to 256 MB of pairs (about 20M rows for one value source) so the transient pairs are bounded; the plain aggregates are merged across windows. A frame's changed and appended rows are added after the kernel pass (the shared rows they replace are skipped by the mask), as for the plain aggregates.
- **GPU:** CPU only. These need every value, not a running number; the GPU kernel's per-group accumulators cannot hold them, and moving the pairs to the device and back costs more than the CPU's pass. `compute(AUTO)` runs them on the CPU and says why in `fallback_reason` when it would have used a device table; `compute(GPU)` raises.
- **Also:** `r.top(i, g)` is empty for an aggregate that is not a `top_k`. `k` is clamped to the table's or file's row count (a Frame cannot know, so a `k` with `groups * k` past 268M raises). `r.sums` stays right across windows. `percentile` raises for `p` outside 0..1, `top_k` for `k < 1`; a value that is a comparison raises as for any aggregate.

Informal timings (F-148; 12-core Ryzen 9 5900X, 24 threads, load about 3, a 20M-row table in memory, `compute()` alone, best of 3, results checked against DuckDB elsewhere; the DuckDB column is the same query in DuckDB 1.5.1 on 24 threads over a table it loaded, not a quiet-machine benchmark):

| 20M rows | plain `sum` + `count` | `count_distinct` | `median` | `top_k(.., 10)` | all three + `sum` | DuckDB: distinct / median / top-k |
|---|---|---|---|---|---|---|
| 8 groups, ~19M distinct values | 17 ms | 148 ms | 120 ms | 19 ms | 265 ms | 258 / 322 / 14 ms |
| 1,000 groups, 1,000 values | 18 ms | 93 ms | 94 ms | 21 ms | 190 ms | 132 / 248 / 17 ms |
| 1,000,000 groups, ~19M distinct values | 108 ms | 372 ms | 398 ms | 449 ms | 739 ms | 617 / 2,208 / 2,295 ms |

A scan of the same 20 row-group file (decode included): `sum` 107 ms, `count_distinct` 490 ms (756 MB peak), `median` 187 ms (598 MB), `top_k` 105 ms (77 MB, the scan's own).

### Standard deviation and variance (F-177)

```mojo
from omxcore import stddev, stddev_pop, variance, var_pop

var r = Query(f).group_by(col("region"), groups=n)
              .agg(stddev(col("revenue")),         # r.floats(0)[g]   sample standard deviation (DuckDB's stddev = stddev_samp)
                   var_pop(col("revenue")),        # r.floats(1)[g]   population variance
                   mean(col("revenue")))
              .compute()
# r.present(0, g) is False for a group with fewer than two values (the sample variants are NULL there)
```

| Builder | SQL | Meaning |
|---|---|---|
| `stddev(e)` | `stddev(x)`, `stddev_samp(x)` | square root of `variance(e)`; NULL below two values |
| `stddev_pop(e)` | `stddev_pop(x)` | square root of `var_pop(e)`; 0 for one value, NULL for none |
| `variance(e)` | `variance(x)`, `var_samp(x)` | the sum of squared deviations from the mean over `n - 1`; NULL below two values |
| `var_pop(e)` | `var_pop(x)` | the same over `n`; 0 for one value, NULL for none |

- **Result:** always a float64 (`r.floats(i)`, `r.values(i)` holds its bits; in SQL the column kind is FLOAT), whatever the input kind. INT, DECIMAL and FLOAT inputs are
  read as DuckDB casts them to DOUBLE (`float(v) / 10^scale`; an int64 past 2^53 is rounded, never an error); TEXT, DATE and BOOL are refused (`stddev() needs numbers, found DATE`).
  NULLs are skipped, as for every aggregate; a group with no rows or only NULLs has no value (`present` false, SQL NULL).
- **Numerically stable.** The state is a count, a mean and the sum of squared deviations `M2` per group, O(groups) whatever the data. Values are cut into pieces of 65,536 in
  their order; each piece is two-pass (a Welford mean, then deviations from it with a correction term, compensated sums), and pieces and windows are merged with Chan's pairwise formula, the
  means kept as a pair (hi + lo) so that a difference of means near 1e15 stays exact. The variance of `1e15 + 1, 2, 3` is exactly 1; a sum-of-squares formula gives 0 or garbage.
  Because the pieces are fixed, **the bits do not depend on the number of threads** (`Query.threads(n)`), as for float sums. DuckDB's own Welford is accurate to ~1e-9 relative on
  such data and far worse on 1e15 + small (var_pop of 1e15 + 1..3 over 100k rows: DuckDB 0.68168, exact 0.66629, ours exact). So the engine is **equal to DuckDB where DuckDB is stable, and exact (more accurate) on cancellation-heavy data**: the tests compare to exact rational arithmetic there and to DuckDB, to 1e-9, elsewhere.
- **Non-finite results are DuckDB's error:** one NaN or infinity alone is a spread of 0 (`stddev` NULL); two or more values with a NaN or an infinity among them, or an overflow
  (`1e308, -1e308`), raise `stddev_samp is out of range: ...` (DuckDB: `STDDEV_SAMP is out of range!`).
- **Sources:** a table, a Frame with edits (the changed rows are folded in after the kernel pass), a streamed Parquet scan (row group by row group, state per group, so memory does not grow with the file),
  and SQL over all three; with other holistic aggregates they share the one pass over the values (the kernel hook writes the (group, value) pairs, `holistic.mojo` folds them into the moments).
- **GPU:** CPU only, like the other holistic aggregates: `compute(AUTO)` runs it on the CPU and says `count_distinct, percentile, top_k, stddev and variance run on the CPU` in `fallback_reason`;
  `compute(GPU)` raises with the same words. (`pipeline_gpu.mojo` and the pinned NVIDIA PTX are untouched.)

### Quantile bins (F-177)

Equal-count bins of a column as a group key, for histograms with even buckets.

```sql
SELECT quantile_bin(x, 10) AS decile, count(*), min(x), max(x) FROM t WHERE region = 'West' GROUP BY decile ORDER BY decile
```
```mojo
var b = Query(f).where(col("region") == 2).quantile_bins(col("x"), 10)         # the edges: one exact pass
var r = Query(f).where(col("region") == 2).group_by(b.key(col("x")), groups=b.bins()).agg(count(), min_of(col("x")), max_of(col("x"))).compute()
```

**The edge rule.** Take the `N` non-NULL values of `x` that the query's `WHERE` keeps (a frame's edits included), sorted. Edge `k` (`k = 1 .. n-1`) is the value at sorted place
`ceil(k * N / n)` (1-based; whole-number arithmetic, never a float `p`). A value's bin is `1 + the number of edges below it`, so:

- equal values always share a bin; a run of ties that crosses an edge falls into the lower bin (the first bin it touches), so bins are equal-count only for distinct values;
- a NULL has no bin: SQL gives it the NULL group, the builder leaves the row out;
- a value above every edge is in bin `n`; with fewer values than bins the top bins are empty (nothing is raised);
- this is `quantile_disc(x, k / n)` wherever `k / n` is exact, and for distinct values and an `N` that `n` divides it is exactly `ntile(n) OVER (ORDER BY x)` (the tests check both);
- FLOAT: NaN is above every number (DuckDB's order), so it is bin `n`, and `-0.0 = 0.0`; DATE and DECIMAL bin on the stored values.

| | SQL | Builder |
|---|---|---|
| Call | `quantile_bin(x, n)` as a `GROUP BY` key (it may be repeated in the select list, `HAVING` and `ORDER BY` only as that key; in `WHERE`, inside an aggregate or inside another `quantile_bin` it raises by name); `x` a column or an expression of INT, DECIMAL, FLOAT or DATE (not a constant), `n` a whole number 2 .. 32 | `Query.quantile_bins(e, n) -> QuantileBins` (edges, `bins()`, `key(e)`) |
| Bin numbers | `1 .. n` (INT), dense: the planner uses the dense strategy | `0 .. n-1` (`key(e)` is the number of edges below `e`): `group_by(key, groups=n)` |
| Edges come from | an exact pass of its own before the query is planned (`n - 1` order statistics over the rows the `WHERE` keeps, the holistic machinery; `EXPLAIN` runs it too), then the bin is one comparison and one add per edge | the same, on `Query.quantile_bins` (CPU) |
| Not supported | in a join or a row query (no `GROUP BY`/aggregate), in `WHERE` or an aggregate argument, `quantile_bin(sum(x), n)`, TEXT or BOOL | after `group_by` / `agg`; TEXT, BOOL |

The most bins is 32: each edge is one of the kernel program's 64 constants. The cost is `n - 1` comparisons a row on top of the pass that finds the edges
(a second query, so `WHERE`s are evaluated twice); an app that bins the same column often keeps the `QuantileBins` and passes `key` to later queries.

## 6. Sort

```mojo
f.permutation(c, descending=False) -> List[Int]   # row order; the data never moves
```
Stable, over any column including formulas and edited cells; nulls last in both directions, removed rows left out. Over a `Frame(scan)` it streams the column but holds O(rows) keys (§3).
Text sorts alphabetically ignoring ASCII case, byte order breaking ties ("apple" before "Zebra"; pending the owner's
review). A radix sort over Int64 keys: about 25-30 ms at 1M rows for text and narrow integers, 55-75 ms for full-range
int64 (informal, F-138a). Next: multi-column sort.

## 6b. Ranges and blocks over an order (F-176)

A grid shows a window of a sorted view; a status bar totals a selection of it. Both are "positions a..b of an order", and both used to cost a `get` / `is_null` per cell
(about 1,000 calls a scroll frame; a 100,000-cell cap on a total). Two calls read the window in bulk (`mojo/omxcore/blocks.mojo`). The engine knows nothing of grids or
status bars: it takes a list of row numbers and a half-open range.

```mojo
from omxcore import range_aggregate, read_block, RangeResult, RangeAgg

var order = f.permutation(cost)                     # any list of row numbers: Frame.permutation, the rows of an ORDER BY, a selection; rows may repeat
var cols: List[Int] = [cost, region, day]
var r = range_aggregate(f, order, cols, 1_000, 2_000_000)    # positions a <= p < b of `order`, for every column in one call
r.rows; r.columns[0].count; r.columns[0].sum; r.columns[0].mean; r.columns[0].min; r.columns[0].max; r.columns[0].has_sum; r.columns[0].has_value()
var s = range_aggregate(t, cols, 0, t.rows())                # no order: the rows themselves (a SharedTable; a Frame works too)

var values = List[Int64](length=len(cols) * (b - a), fill=0)
var nulls = List[UInt8](length=len(cols) * (b - a), fill=0)
read_block(f, order, cols, a, b, values, nulls)              # cell (column k, row i) is values[k * (b - a) + i]; nulls[...] == 1 for a null (its value is 0)
                                                            # after an error the buffers' contents are unspecified (writes stay inside the block)
```

Both calls come in four forms: a `SharedTable` or a `Frame`, with an `order` list or without (then position p is row p; over a Frame that holds a removed row the range must not
include it). A Frame is a table with edits, appended rows, removed rows and merges, or a table on a Parquet file (`Frame(scan_parquet(path))`): the same two calls read all of them.

| What | Rule |
|---|---|
| **Range** | half-open, `0 <= a <= b <= len(order)` (the rows, with no order). `a == b` is an empty range: `count` 0, no sum / min / max (SQL's NULL). Errors are plain sentences: `the range 5..3 ends before it starts`, `the range 0..11 goes past the end: the order has 10 rows`, `position 2 of the order is row 12, but there are only 10 rows`, `row 4 (position 4) was removed`, `column 99 does not exist (the table has 8)`, `the buffers hold 3 cells but the block has 10 (columns x rows)` |
| **Columns** | stored columns only; a formula column raises `column `twice` is a formula: ranges read the stored columns` (still: evaluating a formula per row, batched, is not done; F-179 made formulas undoable only) |
| **Cells** | the stored values, as `Table.get` / `Frame.get` give them: edits applied, merged text codes read as the code they were merged into, appended rows included; a null is skipped by every aggregate |
| **count** | non-null cells, every kind |
| **sum, mean** | INT and DECIMAL: the exact int64 sum at the column's own scale (`sum` is the stored integer, a DECIMAL(2) keeps two places); only a total outside int64 raises (`the sum of column `x` passed the int64 range (about 9.2e18)`; partial sums inside the range may pass it). `mean` is a Float64, DuckDB's AVG bit for bit (the exact sum over `count * 10^scale`, `ld_div`). FLOAT: `sum_float()` and `mean` in float64, see below. DATE, BOOL and TEXT have no sum or mean (`has_sum` is False) |
| **min, max** | every kind. INT / DECIMAL / DATE / BOOL: the stored value. TEXT: the code of the smallest / largest word in the case-insensitive order of `Frame.permutation` (bytes break ties), `f.text_of(c, code)` for the word. FLOAT: NaN is above +inf (MAX is NaN when the range holds one, MIN only when it holds nothing else), -0.0 equals 0.0 (the first in order is reported) |
| **FLOAT sum** | float64 added in position order inside each morsel of 16,384 positions counted from `a`, the morsels' sums added in morsel order, so the bits are the same on every run and every thread count (they differ from a plain left-to-right sum, and from DuckDB's, in the last places). NaN, +inf and -inf follow IEEE |
| **Block layout** | column-major: a column's window is contiguous (`values[k * (b - a) ...]`). `values` holds the stored value: INT the integer, DECIMAL the scaled integer, DATE days, BOOL 0 / 1, TEXT the code (the word is `text_of(c, code)`, a session's new words and merges included), FLOAT the float64's bits (`Float64(from_bits=UInt64(v))`; NaN payloads and -0.0 are kept as stored) |

**How it runs.** No cell is read by a call. A column of the window is gathered into a dense buffer by one tight loop over the order (the shared column read at the row numbers, the validity bit
beside it) and then reduced by another; the changed rows of a Frame are patched in by a small binary search over the rows the session touched, built per call from the cells of the
window's rows (a window no longer than 4x the frame's edits probes its own rows, so a 50-row read costs the same with 3 edits as with 300,000; a longer range takes every changed row, about 0.3 microseconds
an edited cell). A Table or a Frame over one runs on every core: the range is cut into morsels of 16,384 positions, each worker gathers and reduces its morsels into private records, and the records are
folded in morsel order (that fold is what makes sums reproducible). A Frame over a Parquet file runs on the calling thread (its cache is not thread-safe): the window's rows are bucketed by row group
so each (group, column) chunk is fetched once through the frame's cache of decoded chunks (§3, `set_cache_limit`): a repeated window decodes nothing, a first window decodes the chunks of the groups it
touches, and a window wider than the cache decodes on every pass. TEXT min / max marks the codes met and compares only those words.

**Exactness is tested** against a per-cell loop over the same source (`mojo/tests/test_blocks.mojo`: a table, a Frame with 300 edits, appended and removed rows, new words and a merge, again after
undoing them all, and a Frame over a 13-row-group file with 3,000-row groups, ranges of 0, 1, a morsel, a morsel boundary, several morsels and the whole table, random orders, repeated rows), and
against DuckDB (`tests/test_mojo_range.py`: rows a..b of `row_number() OVER (ORDER BY key, rid)` on 60,000 rows with nulls, DECIMAL(18,2) and DECIMAL(9,4), a float column with NaN / +-inf / -0.0 / subnormals,
text with case variants, dates and booleans; a table, a scan of 6,999-row groups, an edited Frame, an edited Frame over the file; ten orders; sums, counts, means (bit for bit), min and max exactly,
float sums to 1e-12 of the sum of absolute values, the block cell by cell).

Informal timings (F-176; 12-core Ryzen 9 5900X, the machine busy with other jobs (load average 5 to 6), best of several runs in one process, `mojo/tests/test_blocks.mojo`, `mojo/tools/range_bench.mojo`):

| call | time |
|---|---|
| `range_aggregate`, one INT / DECIMAL(2) / DATE / BOOL column, 1,000,000 rows, random order (the column is in cache from the previous run) | 0.7 ms |
| the same in table order (no order list) | 0.6 ms |
| one FLOAT column (sum, mean, min, max, NaN count), random order / table order | 1.2 ms / 1.1 ms |
| one TEXT column (13 words), random order / table order | 1.2 ms / 0.7 ms |
| five columns (INT, DECIMAL x2, FLOAT, TEXT) of 1,000,000 rows, random order (the gathers now miss the cache: DRAM-bound) | 12 ms |
| three columns (INT, DECIMAL, FLOAT) of 10,000,000 rows, random order | 106 ms |
| `range_aggregate` of a Frame over a 10-row-group file, 4 columns, 200,000 rows, random order, chunks in the cache | 7.3 ms |
| `read_block`, 50 rows x 40 columns, table, random order | 0.009 ms best, 0.012 median, 0.055 worst |
| the same from a Frame with 302 edited rows / with 300,000 edited cells | 0.014 ms / 0.048 ms |
| the same from a Frame over a file, chunks in the cache | 0.017 ms best, 0.018 median, 0.070 worst |
| the first such window of a file (400 chunks decoded: 40 columns of 10 row groups, the whole file, because the order is random) | 65 ms |
| a long range over a Frame with edits: 3 columns, 1,000,000 rows, random order, no edits / 300 / 30,000 / 300,000 edited cells | 8 / 4 / 11 / 119 ms (machine busy; the edits cost about 0.3 microseconds a cell) |

The single-column rows time a column that stays in the cache between runs; a first read of a column from memory in random order costs several times that (five columns: 12 ms). The one-million-row range
the ticket asked about is under 1 ms for one column and a few milliseconds for several; a grid window is two orders of magnitude under the 1 ms target, and stays so with 300,000 edits.

## 7. GPU memory
- A device copy of a table, **uploaded once per process**, shared by every session; `release()` to give it back when a document closes.
- Values over 32 bits must not break it (today the stand-in uploads int32 and refuses to upload otherwise; a single amount over $21.4M, stored in cents, doesn't fit).
- Share the app's `DeviceContext` so the app's own kernels (charts, point clouds) and omxcore's use the same device memory.
- `compute(AUTO)` falls back to the CPU only when the device cannot hold the query's buffers or the kernel declines the program; `QueryResult.fallback_reason` says which ("" when it ran where it was sent). `compute(GPU)` never falls back.
- `gpu_device()` names the GPU queries run on, "<name> · <api>" ("NVIDIA GeForce RTX 3090 · cuda", "Apple M4 · metal"), for a status bar; "" without one. `QueryResult.device` says where a query ran.
- **Apple GPUs (Metal, F-181..F-186).** A native build on Apple Silicon (or any build with `--target-accelerator metal:<n>`) has the GPU, on by default. Metal has no float64 and no general 64-bit atomics, so on an Apple GPU the kernel's word is an Int64 and only the atomic-free modes run: integer programs (INT, DECIMAL, TEXT keys, DATE: counts, sums, min, max, filters, NULLs) run on the GPU with the same answers as the CPU. Up to 4,096 (slots + 1) x groups the block's table is in shared memory (mode 4); up to about a million (2^20) it is in device memory (mode 5, each group owned by one thread). The CPU answers a program with a FLOAT column or float arithmetic, more (slots + 1) x groups than that, or more registers than 64-thread blocks hold (AUTO says why in `fallback_reason`; `compute(GPU)` raises). Shared memory is planned to 32 KB; narrow blocks have 64 threads. Metal kernels are compiled with Apple's Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`), so a Mac build with the GPU is built on a Mac; a Linux cross-build for macOS (`--target-triple arm64-apple-macosx...` without a Metal accelerator) keeps no GPU, as before. Checked on GitHub's macos-26 runner (`.github/workflows/apple-gpu.yml`, `scripts/check_apple_gpu.py`), whose GPU is a virtual one ("Apple Paravirtual device", M1's Metal family): correctness, not speed, and not yet on real Apple silicon; and on Linux with `-D OMXCORE_APPLE_GPU_SHAPE`, which gives an NVIDIA card Apple's plan for the GPU test suites.
- A build without a GPU (`-D MXFRAME_NO_GPU`, or a target that is neither Linux nor macOS with an Apple GPU): `gpu_available()` is False, `DeviceTable(...)` and `compute(GPU)` raise "no GPU on this build", AUTO is the CPU (see `mojo/README.md`, build switches).
- Later: data bigger than GPU or host memory (chunked, streamed); float programs on Apple GPUs (float32).

## 8. SQL over one table (F-150, row queries F-152)

```mojo
var cat = Catalog()
cat.add_table("orders", orders)              # a SharedTable
cat.add_frame("edited", frame)               # a Frame, snapshotted now (its edit counter is `SqlSource.version`): add it again after more edits
cat.add_scan("big", scan_parquet(path))      # a Parquet file, read a row group at a time (a Frame over a scan works too)
var r = sql(cat, "SELECT region, sum(revenue) AS total FROM orders WHERE qty > 2 GROUP BY region HAVING count(*) > 5 ORDER BY total DESC LIMIT 10")
r.table                                      # an owned Table: INT, DECIMAL, TEXT, DATE, BOOL and FLOAT
r.notes                                      # what the planner decided and what ran, one line each
r.explain; r.ms; r.device (CPU); r.fallback_reason ("")
var e = sql(cat, "EXPLAIN SELECT ...")       # the plan as a one-column Table `plan` (also `e.explain`); nothing runs
var o = PlannerOptions(); o.group = GROUP_HASHED          # GROUP_AUTO, GROUP_DENSE or GROUP_HASHED; o.dense_limit; o.threads
var h = sql(cat, "SELECT ...", o)
```

**The dialect** is a subset of DuckDB's; anything else raises `SQL error at line L, column C: <what>`, naming the construct
(`` `JOIN` is not supported yet (F-153) ``, `column `revenu` not found in table `orders` (did you mean `revenue`?)`).

| Part | Supported |
|---|---|
| Statement | `[EXPLAIN] SELECT ... FROM table [AS alias]` (table names and columns match without regard to case; `;` and `--` / `/* */` comments allowed) |
| Select list | expressions with aliases; group keys, aggregates, and expressions over them (`sum(a) / count(*)`, `max(d) - min(d)`, `g + 1`); a number, date or string constant; `CASE`, `COALESCE`, `CAST`, text and date functions (see Expressions below); with no aggregate and no `GROUP BY` the query returns rows (below): `*`, `t.*`, columns and expressions; `SELECT DISTINCT` |
| `WHERE` | `= <> != < <= > >=`, `BETWEEN`, `IN (literals)` of any length, `LIKE` / `ILIKE`, `IS [NOT] NULL`, `IS [NOT] DISTINCT FROM`, `AND OR NOT`, a BOOL column; text, DATE (`DATE '2024-01-31'` or a string next to a DATE column) and decimal literals (aligned to the column's scale); `+ - *`, `//`, `%` on INT and DECIMAL (scales aligned as DuckDB does) and `date + n`, `date - date` |
| `GROUP BY` | columns, expressions, ordinals (`GROUP BY 1`), select aliases; none at all (one group) |
| `HAVING`, `ORDER BY` | aggregates and group keys in any expression (a row query: any column or expression, selected or not, several keys); `ORDER BY` keys, aggregates, aliases, ordinals; `ASC` / `DESC`; `NULLS FIRST` / `LAST` (NULLS LAST by default, also for `DESC`, as DuckDB); `LIMIT n [OFFSET m]` |
| Aggregates | `SUM`, `COUNT(*)`, `COUNT(x)`, `COUNT(DISTINCT x)`, `MIN`, `MAX` (also of TEXT), `AVG`, `MEDIAN`, `QUANTILE_DISC(x, p)`, `STDDEV` / `STDDEV_SAMP`, `STDDEV_POP`, `VARIANCE` / `VAR_SAMP`, `VAR_POP` (F-177: a FLOAT, NULL below the count they need; section 5) |

**SQL's rules** (the Query builder above keeps its own; both run on the same kernels):
three-valued logic in `WHERE` / `HAVING` (a row is kept only when the condition is true; `x IN (1, NULL)` is true or unknown; `NOT` of unknown is unknown);
NULL is a group of its own; only groups with rows exist (a global aggregate over no rows still returns its one row); `SUM` / `MIN` / `MAX` / `AVG` / `MEDIAN`
of no non-null value are NULL, `COUNT` is 0; `x // 0` and `x % 0` are NULL and both truncate toward zero; a word the column never held matches nothing;
`/` always gives a FLOAT with IEEE division by zero as DuckDB (`1 / 0` is inf, `-1 / 0` is -inf, `0 / 0` is NaN; NaN equals itself and sorts after every number), `AVG` gives a FLOAT that is exactly what DuckDB computes (its long-double division `sum / (count * 10^scale)`, reproduced bit for bit).
Text order everywhere (`ORDER BY`, `MIN` / `MAX`, `QUANTILE_DISC`, `<` and `BETWEEN` against a literal) is alphabetical ignoring ASCII case, with byte order breaking ties; `=` and `IN` compare bytes.

**Differences from DuckDB, on purpose**: a `SUM` (or any int64 arithmetic) past int64 raises `a sum or an expression passed the int64 range (about 9.2e18)` where DuckDB
widens to HUGEINT (decimal `*` too); `MEDIAN` of TEXT is the lower middle word (DuckDB's, in the text order), `MEDIAN` of a DATE raises (DuckDB's is a TIMESTAMP, a kind omxcore does not have; use `quantile_disc(d, 0.5)`), `MEDIAN` of a DECIMAL is exact at one more decimal place (DuckDB rounds to the column's scale; the median of integers is a FLOAT, as DuckDB's DOUBLE);
`%` on a FLOAT raises (F-156: `//` of a DECIMAL or FLOAT is DuckDB's, a division with NULL for a zero divisor); the text order above; FLOAT64 in SQL follows the section below.

**How a query runs.** The statement is parsed (`sql.mojo`), bound to a typed tree (`bind.mojo`: types, scales, which expressions run per row and which after aggregation), turned into a
logical plan (`plan.mojo`: Scan, Filter, Aggregate, Filter (HAVING), Project, Sort, Limit, with the rewrites that look at its shape), planned from statistics only (`planner.mojo`: the `Planner`
trait, `RulePlanner`, `Stats`, `PlannerOptions`; it chooses the group strategy and the sink), lowered to one kernel program in SQL's mode (`sqlprog.mojo`), and run by the executor over the
source's chunks (`exec.mojo`: the Sources; `sqlexec.mojo`: the aggregate Sinks; `rows.mojo`: the row Sinks; `post.mojo`: the stage after aggregation). A table is one chunk; a frame is its shared table with
the changed rows masked out plus a small table of the changed and appended rows; a scan is one chunk per row group. The chunks' partial results (sums and counts add, min / max combine,
hashed groups merge by key) fold into one.

After aggregation the groups become a small table and HAVING, the select list's expressions, ORDER BY, OFFSET and LIMIT run over it as a row query, by the same executor and the same kernel
program lowering as any other row query (there is no second expression evaluator; a FLOAT is a float register, a number that meets one is converted as DuckDB casts it).

| Group strategy | When | How |
|---|---|---|
| none | no `GROUP BY` | one group, the plain kernel |
| dense | every key is a column whose values lie in a known range: a TEXT dictionary, BOOL, or the min / max of an INT / DECIMAL / DATE column of a table or frame (one pass, cached per source), and the domain (a NULL slot per nullable key, keys mixed-radix) is at most `dense_limit` (4,194,304) and either at most 65,536 or at most 4x the rows | the kernel adds into slot `g` directly |
| hashed | a key is an expression, or has no known range (an INT column of a scan), or the domain is too wide or too sparse | `hashagg_cpu.hash_aggregate`: per-worker hash tables over the key words (the value, plus a NULL flag word for a nullable key), a partitioned merge |

`count(DISTINCT)`, `MEDIAN`, `QUANTILE_DISC` and `MIN` / `MAX` of TEXT use the holistic states of §5; over hashed groups they take a second pass that maps each row to its group
by key (`holistic: a second pass ...`). `ORDER BY` with `LIMIT + OFFSET <= 4096` over more than 8x as many groups takes a bounded heap; otherwise stable radix passes, last key first.

`r.notes` is the contract the tests assert: `source: ...`, `group: dense, 2 keys g 7 (with a NULL slot) x s 17 ... = 119 slots: <reason>` or `group: hashed on N key words: <reason>`
(`forced by the options` when `PlannerOptions` said so), `kernel: N instructions a row (M slots)`, `ran: table|frame|scan ...: <chunks>`, `holistic: ...`, `order: ...`, `having: ...`, `time: ...`.

**A file's footer statistics are a hint, not a promise.** The planner uses a Parquet footer's min / max (dense groups for an integer key) and null counts (no NULL checks) but every row group is verified as it is read: a decoded group whose null count differs from the stated one, or a row outside a stated range (dense keys over a file carry one extra group that such rows fall into), makes the query run again without the statistics (hashed groups, every column nullable) and `notes` has `statistics: the footer of `t` cannot be believed (...); ran again without them`.

**Nesting**: an expression may nest 256 levels deep (parentheses, `NOT`, a repeated minus, or a chain of that many terms joined by one operator); more raises a positioned error.

**A frame's formula columns** (§4) are columns of the table: the query reads the formula's expression over the stored columns, inlined, with the builder's own rules for it (floor `//` and `%`,
`x // 0` is 0, `min` / `max`) and exact decimals; they work in `WHERE`, `GROUP BY` (hashed) and the aggregates, over a table frame and a frame over a scan.

**Limits**: joins: inner, left, semi, anti over two tables, inner over three or more (see 8b); `DISTINCT` only without aggregates; expressions as in "Expressions (F-156)"; CPU only;
at most 48 registers (about 20 aggregates over different expressions, fewer when an expression is long: `the query needs more than 48 registers`) and 64 distinct constants in one program, said as an SQL error when passed; an AVG whose exact sum passes int64 raises like SUM.

Informal timings (F-150; 12-core Ryzen 9 5900X under a load average of about 6 from other jobs, so read them as rough; 5,000,000 rows generated with 8 / 10,000 / 1,000,000 groups; omxcore is
`sql()` on a table already in memory, best of 3 in one process, which includes planning and, the first time, the pass that finds a key's range; DuckDB 1.5.1 on a table it loaded, best of 3, its result
fetched into Python; neither is a quiet-machine benchmark). Every result was checked against DuckDB first.

| query | omxcore SQL | DuckDB | group strategy |
|---|---|---|---|
| `count(*), sum(v), avg(d), min(dt)` | 7 ms | 5 ms | none (one result row) |
| 8 groups, 4 aggregates | 8 ms | 6 ms | dense |
| `WHERE v > 100`, 2 keys (56 groups) | 8 ms | 11 ms | dense |
| 10k groups, `ORDER BY 2 DESC, 1 LIMIT 10` | 9 ms | 13 ms | dense |
| 1M groups, `ORDER BY 2 DESC, 1 LIMIT 10` | 133 ms | 92 ms | dense |
| 1M groups, all 1M rows out | 159 ms | 1,900 ms | dense |
| key `g % 100000` (an expression, bounded by interval arithmetic) | 21 ms | 70 ms | dense |
| `count(DISTINCT)` + `median`, 8 groups | 82 ms | 44 ms | dense |
| `quantile_disc`, 10k groups | 38 ms | 67 ms | dense |
| 10k x 7 groups, hashed (forced) | 152 ms | 71 ms | hashed |

The same queries over the Parquet file (one chunk per row group) are decode-bound: 80 to 110 ms to decode the three columns of 50 row groups, plus the kernel. A scan with very many groups merges a
partial result per row group (a dense partial per group is the whole domain), which is slow at 1M groups (1 to 2.5 s): batching row groups and the other big-GROUP-BY work over streamed files is F-158.

### Expressions (F-156)

CASE, COALESCE, CAST, text and date functions run in the same kernel program as the arithmetic, in a select list, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, an aggregate's argument, and
after aggregation (`CASE WHEN count(*) > 5 THEN 'big' ELSE 'small' END`). Every one is checked against DuckDB value by value (`tests/test_mojo_exprs.py`: table, edited frame, streamed file,
edited frame over a file; `test_mojo_expr_scale.py`: every day from 1800 to 2400 and 60,000 UTF-8 words; `test_mojo_expr_fuzz.py`: 1,500 random expressions; `test_mojo_cast_fuzz.py`: thousands of
generated words per text cast).

| Part | Supported |
|---|---|
| Conditional | `CASE WHEN c THEN v ... [ELSE v] END` and `CASE x WHEN v THEN ...`; `COALESCE(a, b, ...)`, `IFNULL`, `NVL`, `NULLIF(a, b)`; `a IS [NOT] DISTINCT FROM b` (NULL is a value, NaN equals NaN); `b IS [NOT] TRUE / FALSE`. The results share one type as DuckDB's rule gives it: a DECIMAL with the most places, a FLOAT when one is (32-bit when all are REAL), a string next to dates is a date; mixing text with numbers raises. Only the branch a row takes can fail: a CAST in an untaken branch never raises |
| `IN` | any number of literals (INT, DECIMAL, DATE, TEXT, FLOAT: -0.0 is 0.0, one NaN); more than 8 numbers go through the kernel's hash set (`OP_ISET`), text and a short list through `OP_IN_SET` / compares |
| `CAST(x AS t)`, `TRY_CAST`, `x::t` | to INTEGER, BIGINT, SMALLINT, TINYINT (range checked), DECIMAL(p, s) (p up to 18), DOUBLE, REAL, VARCHAR, DATE, BOOLEAN. INT / DECIMAL / FLOAT / BOOL convert as DuckDB does (DECIMAL to INT or to fewer places and FLOAT to DECIMAL round halves away from zero, DOUBLE to INT rounds halves to even, DOUBLE to REAL past the 32-bit range fails); a value that does not fit raises `a value could not be converted (CAST(...))` (TRY_CAST: NULL). **TEXT to INT / DECIMAL / DOUBLE / REAL / DATE / BOOLEAN** reads each distinct word once, as DuckDB reads it (spaces around, `+` and `-`, `1_000`, `0x1F`, `.5`, exponents for integers and doubles, `YYYY-M-D` with `-` `/` ` ` or `\` as separators and trailing text, the boolean words); a word that is not one is an error where a row holds it (TRY_CAST: NULL). **To VARCHAR**: BOOLEAN, and an INT, DECIMAL or DATE *column* (the words of its lowest-to-highest range, up to 4,194,304 values, are made once). DATE to anything but VARCHAR, INT to DATE and FLOAT to VARCHAR are refused (DuckDB refuses the first two) |
| Text functions | `lower`, `upper` (DuckDB's own case mapping, generated from it: `İ` lowers to `i`, `ß` uppers to `ẞ`), `trim`, `ltrim`, `rtrim` (spaces, or the characters given), `substr` / `substring` (also `FROM .. FOR ..`; DuckDB's window rules: a negative start counts from the end, a negative length reads backwards), `length` (code points), `left`, `right`, `reverse`, `replace`, `strpos` / `instr`, `starts_with` / `prefix`, `ends_with` / `suffix`, `contains`, `concat` and `\|\|` (`\|\|` is NULL with a NULL, `concat` reads it as ''), `LIKE` / `ILIKE` / `NOT LIKE` with `_`, `%` and `ESCAPE` (no escape character unless named; the pattern is a constant), `IN` lists of words |
| Text compared with text | `=`, `<>` between two columns or expressions compare bytes (a word is translated into the other's code space once); `<`, `<=`, `>`, `>=` compare in the owner's order (ignoring ASCII case, bytes break ties) through one ranking of the words of both; across the two tables of a join too |
| Dates | `EXTRACT(part FROM d)`, `date_part('part', d)`, `year()`, `month()`, `day()`, `quarter()`, `dayofweek()`, `isodow()`, `dayofyear()`, `decade()` (parts: year month day dow isodow doy quarter epoch decade century millennium **week isoyear weekofyear**, F-177: `week(d)` / `weekofyear(d)` / `EXTRACT(week FROM d)` is the ISO 8601 week number 1 .. 53 and `isoyear(d)` the ISO week-numbering year, DuckDB's, so 2021-01-03 is week 53 of 2020 and 2024-12-30 is week 1 of 2025; computed from the Thursday of the date's week; both are dense group keys: 1 .. 53 and the dates' range of years); `date_trunc(part, d)` (year quarter month week day decade century millennium); `date + INTERVAL n DAY / WEEK / MONTH / YEAR` and `date - INTERVAL ...` (the amount a constant or an INT expression; months clamp to the end of the month: Jan 31 + 1 month is Feb 28 / 29); `date - date` (days), `datediff(part, a, b)` / `date_diff` (day, week, month, quarter, year, decade, century, millennium: DuckDB's boundary counting). Constants fold at bind time (`DATE '1998-12-01' - INTERVAL '90' DAY` is one DATE). `EXTRACT(epoch ...)` is a FLOAT, as DuckDB's. All in integer civil-date arithmetic (floor divisions: right before 1970) in the existing opcodes; a date function of one column is instead answered once per day of the column's range (a table the kernel looks the day up in) when that range is short against the rows, and `GROUP BY` of a date part, a truncated date or a date moved by months is dense when the range is known |
| Arithmetic | as before, plus: `//` of a DECIMAL or FLOAT is a division of doubles with a zero divisor NULL (DuckDB's, not a floor), `x // 0` and `x % 0` on integers and decimals are NULL, `/` is always a FLOAT, decimal sums and products keep DuckDB's scales (a product past 18 places raises) |

**How the text functions run.** A TEXT column is a dictionary and codes; a function of it is evaluated **once per distinct word**, not per row, and the kernel looks the codes up (`OP_TAB`: a table of
the answer per code; a predicate such as `LIKE` is the byte set `OP_IN_SET`). A function that gives text makes a new dictionary of its own (a TEXTMAP): the distinct answers, in the order met, and the
table that maps each input code to its answer's code; `lower(s)` is therefore a group key of integers, and the result table holds only the words present. `CASE` / `COALESCE` over words, `||` and a
cast to VARCHAR work the same way (a union of the branches' words, or one word per pair of words: two columns whose dictionaries multiply past 1,048,576 pairs raise instead). `r.notes` says it
ran: `text: 2 per-word tables over 11 dictionary words in 0.23 ms (each text function ran once per word, never per row)`. The tables of predicates and of `length` are built in parallel over the
words, and the `LIKE` shapes `lit`, `lit%`, `%lit`, `%lit%` need no backtracking: 3,610,733 distinct comments are searched for `%special%` in 20 ms.

**Differences from DuckDB, on purpose** (each raises or is documented, none answers differently):
- `date + INTERVAL` and `date_trunc` give a DATE (DuckDB: a TIMESTAMP at midnight; omxcore has no timestamp kind). The oracle casts them to DATE. Intervals in hours, minutes or seconds raise.
- `%` on a FLOAT raises (the kernel's remainder is floored and stepped back, not C's fmod: exact only below 2^53 quotients); `//` on REAL (32-bit) raises.
- `reverse` of a word with combining marks, joiners, variation selectors, regional indicators or an Indic script raises: DuckDB reverses by grapheme clusters, omxcore by code points, which agree only for words whose every character stands alone.
- A text word that DuckDB reads by a quirk this engine does not (`infinity`, `epoch`, `now` and their abbreviations as dates; BC dates; years past 9999; an exponent on a DECIMAL, whose rounding is odd; a long mantissa with an exponent) is an error in a CAST *and* a TRY_CAST (never a NULL or a guess); `tests/test_mojo_cast_fuzz.py` bounds how many such words there are.
- A computed date before year 1 (e.g. `DATE '0001-01-01' - INTERVAL '1' MONTH`) prints in results as `0000-12-01` (ISO 8601 astronomical years: year 0 is 1 BC); DuckDB prints the same day as `0001-12-01 (BC)`.
- `CAST(expr AS VARCHAR)` needs a column of INT, DECIMAL or DATE (or a BOOLEAN): the words come from the column's range; a date before year 1 or after year 9999 is an error where a row holds it.
- A `LIKE` pattern is a constant (a column as the pattern would need the product of two dictionaries); `||` of an INT needs `CAST(.. AS VARCHAR)` (DuckDB casts for you), unless it is a constant; `concat` and `||` of two columns need the dictionaries' product to be at most 1,048,576.
- An integer overflow or a failing CAST in a branch the row does not take: the conversion flags are gated by the branch, but an int64 overflow is not (it raises as it always did).
- DuckDB's own wrong answer: it treats `INTERVAL '30' DAY` and `INTERVAL '1' MONTH` as equal in its common-subexpression elimination, so a SELECT holding both gives the month's result for the day's column; the test oracle turns the optimizer off.

Informal timings (F-156; the 12-core Ryzen 9 5900X under a load average of 6 to 10 from other jobs, so rough; omxcore is `sql()` on a table in memory, best of 3 runs in one process, planning included; DuckDB 1.5.1 on
a table it loaded, best of 3, its result fetched into Python; every result was checked against DuckDB first). 5,000,000 generated rows (a DATE over 9,000 days, 400 words with case and accents, 1,000,003 distinct words, a number as text),
and TPC-H SF1 `lineitem` (6,001,215 rows):

| query | omxcore | DuckDB |
|---|---|---|
| `EXTRACT(year FROM dt)` group, count, sum | 4.9 ms | 3.6 ms |
| `date_trunc('month', dt)` group | 4.3 ms | 20.6 ms |
| `dt + INTERVAL 3 MONTH > DATE ... AND EXTRACT(dow FROM dt) = 1` | 3.7 ms | 18.8 ms |
| `sum(datediff('month', dt, DATE '2024-01-01'))` | 2.6 ms | 7.7 ms |
| `sum(CASE WHEN x > 0 THEN dd ELSE 0 END), count(CASE WHEN ...)` | 5.0 ms | 4.8 ms |
| `lower(s)` group (400 words) | 5.8 ms | 16.5 ms |
| `s LIKE '%a%'` (400 words) | 2.4 ms | 5.5 ms |
| `s \|\| '-' \|\| s2` group | 7.5 ms | 21.8 ms |
| `substr(s, 1, 2)` group, `sum(length(s))` | 5.6 ms | 24.9 ms |
| `sum(CAST(num AS INTEGER))` (1,000 words) | 2.5 ms | 6.1 ms |
| `id IN (1,000 numbers)` | 2.9 ms | 6.5 ms |
| `h LIKE '%99%'` (1,000,003 distinct words) | 4.6 ms | 8.4 ms |
| `count(DISTINCT lower(h))` (1,000,003 distinct words) | 336 ms | 58 ms |
| lineitem: `sum(CASE WHEN l_discount > 0.05 THEN ... ELSE 0 END), count(CASE ...)` | 6.4 ms | 9.4 ms |
| lineitem: Q12's `CASE` pair per ship mode | 8.4 ms | 16.3 ms |
| lineitem: `l_comment LIKE '%special%'` (3,610,733 distinct words) | 23.7 ms | 12.7 ms |

A date function of a column is answered once per *day of the column's range* when the range is short against the rows (a per-day table the kernel looks up: `dates: 2 per-day tables over 9,000 days`), and the planner
bounds the date parts by that range, so `GROUP BY EXTRACT(year ...)` is dense. The slow row is a text function that *makes text* over a million distinct words: it builds each answer once (about 0.3 µs a word, one thread)
and merges equal answers into one code; a predicate or `length` over the same words is built in parallel and takes 2 ms.

### Row queries (F-152)

A query with no aggregate and no `GROUP BY` returns rows: `SELECT *`, `SELECT t.*, x + 1 AS y`, `WHERE`, `ORDER BY` (several keys, ASC / DESC, NULLS FIRST / LAST, any expression), `LIMIT`, `OFFSET`, `SELECT DISTINCT`.
The result is an owned Table, as always; a frame's formula columns are columns (and part of `*`); text keeps the source's words (the result's dictionary holds the words present). A select item can be a column, an
arithmetic expression over INT / DECIMAL / DATE columns, a comparison or logical expression (a BOOL column of true / false / NULL) or a constant; a string constant is a one-word TEXT column (F-156) and a bare `NULL` is refused by name; a FLOAT (a column of the file, or `/`) is selected, ordered and made distinct like any other kind, in DuckDB's order (NaN equal to itself and above +inf, -0.0 equal to 0.0).

| Part | What runs |
|---|---|
| materialize (the sink) | each chunk runs one kernel program: the WHERE (and a frame's mask) and the expressions of the select list and ORDER BY that are not plain columns, into registers. The kernel's block hook (`rows._rows_hook`, the `_mat_hook` pattern) writes, per block of 1,024 rows, the numbers of the selected rows and the computed values at the block's own place, and a prefix sum over the blocks' counts compacts them. A plain column is not carried through registers: it is gathered from the chunk by the selected rows' numbers, so a wide `SELECT *` costs no registers |
| `ORDER BY` | the collected rows are sorted by a stable LSD radix pass per key, from the last key to the first (text under the owner's order, NULLS FIRST / LAST per key); ties keep the source order (a table's rows; a frame's, in the frame's own row order, see below) |
| `ORDER BY ... LIMIT k [OFFSET m]` | when `k + m` is at most 4,096 and the source has more than 8x that many rows, no row is materialized: each worker keeps its best `k + m` rows in a heap inside the hook (a first-key test drops most rows before anything is copied), the workers' heaps are merged with the best so far after each chunk, and the result is cut to `[m, m + k)`. Ties keep the source order. EXPLAIN says `order: top-K by per-worker heaps in the hook` and the notes `rows: the best K of N rows kept in per-worker heaps`; otherwise `order: all rows sorted (why)` |
| `LIMIT n` with no `ORDER BY` | the scan stops once it has `n + offset` rows: a table is read in slices, a scan stops reading row groups (`ran: scan t: 1 of 13 row groups read`, `rows: N rows selected (stopped early: the LIMIT was met)`). Which rows come first is the source order |
| `SELECT DISTINCT a, b` | the group-by of the select list (the binder rewrites it; EXPLAIN shows `Distinct keys=[a, b]`), so it takes the dense or hashed strategy with the reason; `DISTINCT` with aggregates or `GROUP BY` raises (use `GROUP BY`); its `ORDER BY` terms must be in the select list |

**Row order on an edited frame.** A query with no `ORDER BY` lists a frame's rows as the frame holds them: a row the session changed stays where it was (the shared row is masked out and its new values come from the delta chunk, merged back by row number), an appended row comes last, a removed row is gone. Equal sort keys and a `LIMIT` follow the same order, so a preview after an edit does not reorder the sheet. The merge is linear (two ascending runs), not a sort; a `LIMIT` with no `ORDER BY` reads the changed rows first (they are few) and the shared rows until `n + offset` of them are in hand.

Informal timings (F-152; the same 12-core Ryzen 9 5900X under a load average of about 5 from other jobs, so rough; the F-150 test generator's table of 1,200,000 rows and 9 columns written to Parquet; omxcore is the in-process
`sql()` time reported by `mx` (the file read apart), best of 3 fresh processes, the result a Table; DuckDB 1.5.1 on a table it loaded, best of 3, the result an Arrow table; every result is checked against DuckDB in `tests/test_mojo_rows.py`):

| query | rows out | omxcore | DuckDB |
|---|---|---|---|
| `SELECT id, x, d FROM t WHERE x > 500` | 275,347 | 12 ms | 9 ms |
| `SELECT * FROM t` | 1,200,000 | 114 ms | 59 ms |
| `SELECT id, x * 2 + 1 AS y, d + e AS de, dt FROM t WHERE k = 1` | 215,586 | 16 ms | 12 ms |
| `... WHERE x > 0 LIMIT 100` (the scan stops early) | 100 | 0.2 ms | 0.2 ms |
| `SELECT id, x FROM t ORDER BY x DESC, id LIMIT 10` (heaps) | 10 | 2.9 ms | 1.4 ms |
| `SELECT id, s, dt FROM t ORDER BY s, dt DESC LIMIT 50 OFFSET 100` (heaps, text key) | 50 | 3.5 ms | 6.0 ms |
| `SELECT id, x, s FROM t WHERE x IS NOT NULL ORDER BY x, id` (full sort) | 1,103,764 | 114 ms | 47 ms |
| `SELECT id, s FROM t ORDER BY s, id` (full sort, text) | 1,200,000 | 116 ms | 104 ms |
| `SELECT DISTINCT s FROM t` | 16 | 1.3 ms | 4.0 ms |
| `SELECT DISTINCT k, b FROM t` | 21 | 2.4 ms | 3.5 ms |
| `SELECT DISTINCT g FROM t` (hashed, 983,770 groups) | 983,770 | 45 ms | 22 ms |

Aggregates after the move of HAVING / ORDER BY / the select list onto the executor (same data; 983,770 groups for `g`): `GROUP BY g ORDER BY 2 DESC, 1 LIMIT 10` 57 ms (DuckDB 20),
`GROUP BY g` with all groups out 108 ms (61), `GROUP BY g HAVING count(*) > 1 ORDER BY sum(x) / count(*) DESC LIMIT 10` 67 ms (18), 7 groups 3.6 ms (2.0). DuckDB is faster where the result is a million rows or the query is a
full sort; the heaps, the early stop and `DISTINCT` over few values are where this executor wins or ties.

### The `mx` command line (F-151, for agents)

`mx` is the SQL of section 8 as one native program, built by `pixi run bash scripts/build_mx.sh` into `build/mx` (no Python in it; the script checks that). A cold `mx sql` on a small
file starts, runs and exits in about 11 ms wall (20 fresh processes in a shell loop, process creation included).

```
mx sql "<SQL>" [-t name=path ...] [-f query.sql] [--format json|jsonl|csv|parquet] [--out file] [--max-rows N] [--device cpu|gpu|auto]
mx explain [--analyze] "<SQL>" [-t name=path ...] [--format text|json]
mx schema <file> [--rows N]
```

- **Files.** `FROM 'orders.parquet'` and `FROM 'orders.csv'` open the file named in the query; `-t name=path` registers a file under a name (`FROM name`). A Parquet file is streamed a row group at a
  time when it is over a quarter of the machine's memory (or `--stream always`; `--stream never` reads it whole; `--memory BYTES` changes the memory it compares with), otherwise read into memory.
  The notes say which. A CSV is read whole, and as UTF-8 only: a file with a byte that is not UTF-8 (Latin-1 `caf\xe9`) or a UTF-16 byte order mark is an error naming the byte and line, with a hint to re-save it (`iconv -f LATIN1 -t UTF-8`); a UTF-8 byte order mark is skipped. The result is therefore always valid UTF-8. The advanced switches `--group auto|dense|hashed`, `--dense-limit N` and `--threads N` are F-150's `PlannerOptions`.
- **Device.** SQL runs on the CPU. `--device gpu` is accepted and the notes say it ran on the CPU; `device` in the result is `"cpu"`.
- **Result (stdout, one JSON object).**
  `{"columns":[{"name","kind","scale"}],"rows":[[...]],"row_count","truncated","ms","device","notes"}`.
  `kind` is INT, DECIMAL, TEXT, DATE, BOOL or FLOAT. An INT is a JSON integer. **A DECIMAL is a JSON number whose text is exact** (`1234.50`, scale digits kept): parse it with a decimal parser
  (`json.loads(text, parse_float=Decimal)`), never a float one. A DATE is `"yyyy-mm-dd"`, TEXT is the word, BOOL is `true` / `false`, NULL is `null`. A FLOAT is its shortest round-trip number;
  infinity and NaN, which JSON cannot hold, are the strings `"Infinity"`, `"-Infinity"` and `"NaN"`. `rows` holds at most `--max-rows` rows (default 1000; 0 means all); `row_count` is the true
  count and `truncated` says whether `rows` is shorter. `notes` is the planner's account of what ran (section 8): read it to see the group strategy, the columns read, the chunks, any re-run.
- **Other formats.** `jsonl` is one object per row (`{"k":0,"total":12.50}`), `csv` has a header and is written as DuckDB writes it: a cell is quoted when it holds a comma, a quote or a line break, **the empty string is `""` and NULL is an empty cell**, dates and decimals as in JSON. `--out file` writes every row to the file
  (whatever `--max-rows` says) and prints the JSON summary (`columns`, `row_count`, `ms`, `notes`, `out`, no `rows`). `--format parquet` needs `--out` (a FLOAT result column is written as Parquet DOUBLE). `--out` never creates directories: a missing one is an exit-1 error `couldn't write ...`; JSON, JSON lines and CSV are written in chunks of about a megabyte.
- **Errors.** A query error (bad SQL, unknown column, unsupported form, a file that cannot be opened) prints `{"error":{"message","line","column","hint"}}` on stdout and exits **1**; `line` and `column`
  are 1-based into the SQL text (null when the error is not about a position), `hint` carries the "did you mean" suggestion or what to do. A usage error (unknown command or option, wrong
  number of arguments, a bad value) prints the same object with the usage text as the hint and exits **2**. Exit 0 is success.
- **`mx explain`** prints the logical and physical plan as text (`--format json`: `{"plan":[lines],"analyzed","ms","notes"}`); `--analyze` also runs the SELECT and adds what ran and the time.
  `mx sql "EXPLAIN SELECT ..."` works too.
- **`mx schema file`** (Parquet or CSV): `{"file","format","bytes","rows","row_groups","columns":[{"name","kind","scale","nulls","distinct_words","first_values"}]}`; `distinct_words` is the size of a TEXT
  column's dictionary (null for other kinds), `first_values` the first `--rows` values (default 5) in the result's JSON form. A Parquet file is read from its footer and a few row groups.

Things an agent should know:

- **A long OR chain fails.** The parser stops expressions nested 256 levels deep, and an `a OR b OR c ...` chain of about 255 terms is that deep. Write `x IN (a, b, c)` instead. A list of **numbers** in
  an IN takes at most 40 values (`the most is 40`); split a longer one into several queries or filter by a range. A list of text words has no such limit.
- **A table is one file.** `FROM` is a registered name or a quoted file name; a second file is registered with another `-t name=path` or named in the query (joins: section 8b). A row query (`SELECT *`, `WHERE`, `ORDER BY`, `LIMIT`) works; `--max-rows` caps what is printed
  (default 1000, `row_count` is the real count).
- **A Parquet file with a DOUBLE (or FLOAT) column opens (F-157).** FLOAT is a column kind: see "FLOAT64 (F-157)" in section 1 for the rules (NaN order, -0.0, the reproducible sum, a 32-bit column compared with a literal in 32 bits). A Parquet text value that is not UTF-8 is refused.
- **A footer that lies.** When a Parquet footer's min / max or null count turns out wrong while reading, the query runs again without the statistics and the notes say `statistics: ... ran again without
  them`; an EXPLAIN of that run says `statistics not believed` where a file with no statistics says `the source has no statistics on it`.
- Text order is the owner's (case-insensitive, bytes as the tie-break), so `min(s)` / `ORDER BY s` on text differ from DuckDB's byte order; DECIMAL `median` has one more digit of scale than DuckDB.

## 8b. Joins (F-153, F-154, F-155)

The equi-join (inner, left, semi, anti) is an engine feature: a catalog of named sources (tables, Frames, Parquet scans), the PlanNode IR with two Scans under a Join, a hash join
kernel, and two ways to run it. It names no application; SQL, `Query.join`, and any other caller use the same code (the Python engine does not call the kernel yet: F-161). `Query.join` always materializes every column of both tables (it cannot know yet what the query after it reads): fusing it and carrying only the needed columns is F-161.

**SQL**: `FROM a JOIN b ON a.k = b.k [AND ...]`, `JOIN ... USING (k)` (`SELECT *` keeps the left columns, then the right minus the USING columns), and a comma join
whose WHERE has the equalities (`FROM a, b WHERE a.k = b.k`). A name is `alias.column` or a bare column that only one side has; a bare name both sides have raises
(`ambiguous`), joining a table with itself needs two aliases. `RIGHT`, `FULL` and `CROSS` joins are refused by name (`RIGHT JOIN is not supported yet`). Non-equality conditions in ON
and WHERE conditions over both sides run after the join (inner joins only: see below for the other kinds). The aggregates, `GROUP BY`, `ORDER BY` and `LIMIT` of section 8 work over the joined rows.

**LEFT, SEMI, ANTI (F-154)**: `a LEFT [OUTER] JOIN b ON ...` keeps every row of `a`; where nothing matches, `b`'s columns are NULL (a NULL key matches nothing). `a SEMI JOIN b ON ...`
keeps the rows of `a` that have a partner, `a ANTI JOIN b ON ...` the rows that have none; both return `a`'s columns only (`b`'s are visible in ON, nowhere after it). All three build on the
right table. Subqueries are these joins: `x IN (SELECT c FROM t [WHERE p])` and `EXISTS (SELECT .. FROM t WHERE t.k = outer.k [AND p])` are semi joins, `NOT EXISTS` is an anti join,
`x NOT IN (SELECT c FROM t [WHERE p])` is a null-aware anti join: it keeps no row when the list holds a NULL, keeps every row (a NULL key too) when the list is empty, and drops a row whose key
is NULL otherwise. In a subquery's WHERE a bare name is the subquery table's column when it has one, else the outer table's; an equality between a column of each is the join key, the other
conditions on the subquery's table filter it. Shapes that need a residual predicate during the match are refused by name: a non-equality condition across the tables, or a condition on the
left table alone, in the ON of a LEFT or ANTI join (F-155); one subquery per WHERE, joined to the rest by AND (F-155); a subquery with GROUP BY, a join, a scalar subquery or an EXISTS with no
correlated equality (F-159). A LEFT join on keys too wide to pack into one int64 is refused (F-155).

**Three or more tables (F-155)**: `FROM a, b, c WHERE ...` and `a JOIN b ON .. JOIN c ON ..` (inner joins; `USING` and the other kinds raise by name: a LEFT, SEMI or ANTI join with a third
table is F-159). The WHERE and the ON conditions are one set of conjuncts: an equality between columns of two tables is an edge of the join graph (the tables must be connected by them: no
cross product), a condition on one table filters that table first, a condition that reads several (an OR across tables, an inequality) is checked as soon as every table it reads is joined. The
common factors of an OR come out first: `(A AND x) OR (A AND y)` is `A AND (x OR y)` (TPC-H Q19: the shared `p_partkey = l_partkey` becomes the join key). Names resolve as for two tables
(an alias per use of a table; the same table twice needs two aliases).

How it runs, in the order of `EXPLAIN`'s notes: every table runs its own filter first (a row query over its source), so the planner works from exact filtered row counts and sampled
distinct counts of the join columns (`statistics:` notes; a table of 65,536 rows or more whose filter passes at least half of it is only counted and read unfiltered when it is the last join's
probe side: its filter then runs in the pipeline and no filtered copy is written); `Planner.join_order(JoinStats, PlannerOptions) -> JoinPlan` picks the order and the build side of each join
(the default `rule_join_order`: a greedy chain from every table as the start, the next table being the connected one that leaves the fewest rows by `rows x rows / max(distinct counts)`, the
cheapest chain by hashing the smaller input twice, probing the larger once and producing the output once; the written order is kept when it connects and its cost is within 10% of the best);
each join but the last gathers its pairs into a table that keeps only the columns later steps read; the last join runs with the rest of the query (aggregate, ORDER BY, LIMIT), fused when its build
keys are unique. A tail of joins on dimensions with unique keys runs as one **star**: the probe rows meet every dimension's map in one pass of blocks (each map's partner row is kept per block row,
so a miss in any drops the row), and no joined row or intermediate table is written; a dimension reached only through another (nation through supplier) is joined to it first. `PlannerOptions`
forces each choice: `join_order` (the tables' positions in FROM), `written_order`, `build_side` (BUILD_LEFT / BUILD_RIGHT), `join_budget` (bytes of joined values a materialized join may take),
`fuse` (False: no fused join, no star); `mx sql --order written|2,0,1  --build left|right  --join-budget BYTES  --join materialize`. A custom `Planner` overrides `join_order` and the executor does
not change (`mojo/tests/test_multiway.mojo`).

`EXPLAIN` shows the plan as written (a left-deep chain of the tables), then every decision with its numbers: the leaves' rows, the distinct counts, the chosen order and why, each join's hash side
and estimated rows. `EXPLAIN ANALYZE` (also `mx explain --analyze`) runs the query and adds an `Actual` section: the rows and milliseconds of each scan, each join (actual rows next to the estimate), each
filter across tables, the star, and the result.

**Builder**: `Query(left).join(right: SharedTable, left_on, right_on, how)` (`how` = "inner", "left", "semi" or "anti") returns a query over the joined rows (this table's columns, then the other's; a repeated name gets a
suffix `_1`). It must come first and works over plain tables; the rest of the builder's rules apply after it.

**Keys (exactness)**: a NULL key never matches. TEXT keys match by bytes across dictionaries (a once-per-dictionary translation of codes), so `West` is not `west`
(only ordering ignores case). INT is a DECIMAL of scale 0 and scales are aligned (a value that overflows when aligned cannot equal anything). FLOAT keys follow
DuckDB: -0.0 = 0.0 and NaN = NaN. int64 keys past 2^53 are exact. DATE against INT, text against a number and a narrowed column are refused with a message. Several
keys are packed into one int64 when the build side's ranges multiply to fewer than 2^62 values; otherwise they are hashed and the candidate pairs are verified against the columns.

**Paths** (the `notes` say which ran: `join build side:`, `join keys:`, `join path:`, `join time:`):
- The build side is the side with fewer rows after its own filters (each side is first reduced to the columns the query needs, by a row query over its source, so a Frame's edits,
  a scan's row groups and a one-side filter are all applied; an unfiltered plain table is read in place).
- *fused*: the build keys are unique (aggregates, and row queries including top-k and ORDER BY / LIMIT). The probe table is scanned by the pipeline; each block's rows look their partner up (direct array or hash table, with prefetch) and
  the partner's columns are gathered into per-worker scratch. No joined row is written.
- A **word dictionary is shared, not copied**: a join's gathered columns and the fused join's schema hold the source column's dictionary (an `ArcPointer`; the first `add_word` on a shared one copies it), so a join over a 150,000-word text column costs no per-word work.
- *fused* LEFT: the same with unique build keys; an unmatched row stays in the block with its build columns NULL (validity bits clear). *Fused* SEMI / ANTI: any build keys (a presence-only
  map: one pass of marks, no repeated-key rows); the block's rows are kept or dropped by whether their key is present, and nothing is gathered.
- *materialized*: repeated build keys, hashed composite keys, or holistic aggregates (median, percentiles, distinct): the matching (probe, build) pairs are counted
  first, gathered into a Table, and the query runs over that (a LEFT join's unmatched rows are pairs with build row -1; a semi / anti join gathers the selected left rows). Past the memory budget (3 GiB of joined values) it raises a readable error (spill is deferred). `mx --join materialize` forces it.

**The kernel** (`kernels_aot/hashjoin_cpu.mojo`; Mojo API and `libmxhashjoin_cpu.so` with a C ABI over a handle, `hashjoin_lib.mojo`): int64 keys and an optional `ok` byte array in, build rows
out. `hj_build` picks a direct array (range at most max(8n, 2^20) and 2^28; CSR rows when keys repeat) or radix partitions sized from L2 with linear probing and a presence bitmap; the build
is parallel without atomics. `hj_probe_unique`, `hj_probe_pairs` (budgeted; `outer` adds a LEFT join's unmatched rows as build row -1) and `hj_probe_block` / `hj_probe_left_block` (fused) probe it; `hj_build(presence=True)` and `hj_probe_select` / `hj_filter_block` (modes semi, anti, null-aware anti, none) are the semi and anti joins' form. The engine-level steps (`normalize_keys`, `join_pairs`, `gather_join`) are in `joincore.mojo`.

## 8c. Background jobs (F-175, `job.mojo`, `jobs.mojo`)

An app runs a query, a file read or a save off its UI thread, polls it (or takes a callback), shows progress and can cancel. Every job is a `Job[T]` for the `T` the plain call returns.

```mojo
var job = compute_job(query, AUTO)                                  # Job[QueryResult]  : Query.compute
var job = sql_job(catalog, "SELECT ...", options)                   # Job[SqlResult]    : sql(catalog, text, options)
var job = read_csv_job(path, options)                               # Job[CsvResult]    : read_csv
var job = scan_csv_job(path, options)                               # Job[ParquetScan]  : scan_csv, the pass that finds a CSV's row groups (F-178)
var job = read_parquet_job(path)                                    # Job[ParquetResult]: read_parquet
var job = save_job(frame, path, metadata, compression, allow_narrowing)   # Job[Int]     : Frame.save, the result is the rows written
var job = write_parquet_job(table, path, metadata, compression, row_group_rows)   # Job[Int]: write_parquet
var job = permutation_job(frame, column, descending)               # Job[List[Int]]   : Frame.permutation (a sort), over a snapshot

job.poll()        # JOB_RUNNING, JOB_DONE, JOB_FAILED or JOB_CANCELLED (state_name(s) is the word)
job.wait()        # blocks until the job's thread has ended; returns the state
job.result()      # waits, then moves the result out (once). Raises the job's own error when it failed, "cancelled" when it was
job.cancel()      # JOB_CANCELLED from this call on (unless the job had ended); the work stops at its next check
job.progress()    # JobProgress: .done of .total rows, .stage ("planning", "reading", "aggregating", "writing", "finishing"), .bytes
```

A completion callback is a struct of the app's with `done(mut self, state: Int)` (`JobDone`), passed as the last argument (`compute_job(q, AUTO, MyCallback(token))`). It runs once, on the job's thread,
after the state is final (`poll()` already answers it), with JOB_DONE, JOB_FAILED or JOB_CANCELLED. It is skipped when the handle was dropped first. It must not block, and the UI wakes itself from it (an event post), never draws.

**What a job owns.** The start function takes what the job needs by value or copies it: a `Query` is moved in (build it again for another run), `sql_job` works on a copy of the `Catalog` (frames in a catalog are
snapshots already), `save_job` takes a snapshot of the `Frame` (`frame_snapshot`: its edits as of the call; later edits are not in the file), and tables, files and scans are shared read only handles. The caller can
read cells, edit the frame and register new sources while a job runs; only a `ParquetScan` stays shared with the job, and its lazy TEXT dictionary build is guarded by a lock, so both may ask for it.
Nothing of the job's is the caller's address: the job's body (a struct) owns its inputs and frees them the moment the work ends, and the result lives in a slot freed by the last of the two owners (the handle and the thread).

**Cancel.** `cancel()` answers at once and never returns a partial result. The work checks between:
- *queries over a table*: row ranges of 2M rows (integer sums, counts, min and max merge exactly, so the answer is the one call's; a query with a float sum or a reproducible plan keeps its one call, since its float64 sum has one defined order); the windows of count_distinct / percentile / top_k;
- *over a Parquet file or a Frame over one*: the blocks of a morsel (32,768 rows by default), the waves of row groups, the row groups;
- *SQL*: the chunks (a scan's row groups), and inside a big table chunk ranges of 2M rows: dense and hashed group-bys (the ranges' groups are merged with `_merge_hashed`, as a scan's row groups are), the probe side of a fused join, row queries (slices of 1M rows). Same rule as queries: a float sum or a reproducible plan keeps its one call. A materialized join's pair building and gather run to their end; the steps between them check. A cancel is noticed within about one range (about 10 ms at the engine's speed);
- *sorts* (`permutation_job`): every 1M elements of the key pass and of each radix pass;
- *CSV*: the chunks the file is cut into (`CsvOptions.chunk_bytes`, 4 MB; a file read by one thread is one chunk); `scan_csv_job`: between row groups, progress in bytes;
- *Parquet read and write*: row groups.
The thread that notices unwinds (everything it owns is freed on the way) and the job ends. Dropping the handle is a cancel that waits for nothing: the thread ends by itself and frees the rest.
A save writes a temporary file and renames it; `cancel()` before the rename removes the temporary file and the old file stays whole. Just before the rename the save passes its point of no return
(`JobCtl.commit()`): a `cancel()` after that changes nothing and the job ends JOB_DONE with the new file in place, so a job that says "cancelled" never replaced a file.

**Saves and files.** Every writer has a temporary file of its own (`path.<pid>.<allocation address>.<clock>.tmp`), so two saves to one path, or a cancelled save still unwinding beside the one that replaced it, never touch each other's file; the last
rename wins and the file is always exactly one of the inputs. A save survives a cancel and a `kill -9` (the old file stays) but not a power loss: there is no `fsync` before the rename. A killed process leaves its temporary file behind (the name says whose it was).

**Handles and callbacks.** A handle dropped while its callback is running does not stop the callback: the app must not free what the callback touches on drop (a callback that posts an event to a long-lived queue is safe). `JOBS_ARE_THREADED` (`omxcore`, `omxcore.job`) is
the build's answer to "do jobs run on threads?"; an app can assert it. When it is False a job runs, and its callback is called, inside the start call: the callback runs **before** the handle is returned.

**A scan shared with a job.** The file's TEXT dictionaries are built once, behind a lock, by whichever of the job and the caller asks first (the flag that says "built" is an atomic release store, read with an acquire load, so a core with a weak memory order such as arm64 sees the words
once it sees the flag). The other waits (a spin that yields the core on a threaded build). To avoid waiting, the caller can check `scan.dictionary_loaded(c)` first.

**Progress.** `done` and `total` count rows (`bytes` is True for a CSV, whose rows are unknown before they are parsed, so it counts the file's bytes). One atomic add per block, never per row. `done` only grows; `total` can grow
while the job finds more work (a join reads its sides one after another, a hashed holistic query makes a second pass); a job that ends done has `done == total`.

**Threads.** A job is a `pthread` of its own (`pthread_create`, then the work calls the pool as any caller does: `start_pool` first), so jobs run side by side with each other and with the caller. A plain
call with no job passes `JobCtl()`: never cancelled, nothing counted, the code path unchanged.
**The serial build** (`MXFRAME_SERIAL`, and every target but Linux) has no async runtime and, by default, no job threads: the job runs to its end inside its start call and the handle comes back done (or failed), progress at the
total, the callback already called; `cancel()` then has nothing to stop. No new symbol: the Mac link check (`scripts/check_mac_link.sh`) lists the same libSystem calls as before the jobs (`pthread_*` is compiled out of that build).
**Jobs with threads on a serial build** (`-D MXFRAME_JOB_THREADS`, for the app that wants its Mac UI free): the job gets a `pthread` with a 16 MB stack and its kernels still run in a loop (no async runtime, no `AsyncRT_*` symbol);
the app's libSystem stub and its `ALLOWED` set then need `pthread_create`, `pthread_join`, `pthread_detach`, `pthread_attr_init`, `pthread_attr_setstacksize` and `pthread_attr_destroy` and `sched_yield` (`scripts/check_mac_link.sh` prints them under "with job threads").
To add a job for another kind of work: a struct with `comptime Out = T` and `def run(mut self, ctl: JobCtl) raises -> T` (`JobBody`), started with `start_job(body)` or `start_job_then(body, callback)`; call `ctl.check()`,
`ctl.add_total(n)` and `ctl.advance(n)` at block boundaries.

Tests: `mojo/tests/test_job.mojo` (the handle with made-up work: state, error, cancel, commit, callback, dropped handles, memory over 300 cycles), `mojo/tests/test_jobs.mojo` (every kind of query, read and save equal to the plain call, progress
monotonic and ending at the total, a cancel storm with the memory watched, cancelled saves keeping the old file byte for byte, jobs at once, a job over an edited Frame while the caller reads cells),
`mojo/tests/switches/test_switches.mojo` (the serial build).

## 9. Later (Query and Explore modes)
RIGHT / FULL joins, LEFT / SEMI / ANTI with three tables, scalar subqueries, expression functions (M8: F-156 to F-162); pipelines with cached steps; versioned column
chunks for a time machine and branches; TPC-H through SQL.

## Memory rules
- A shared `Table` is read-only once shared and outlives every `Frame` built on it.
- Borrowed spans stay valid until the next edit to that column.
- Results own their buffers. Device buffers belong to the `DeviceContext` they were made on.

## Done means
omxplorer imports a CSV, edits, adds formulas, sorts, builds charts, pivots and a filtered Report, and saves and
reopens a `.omx`, with `from omxcore import …` in place of `mojoui/frame*.mojo`, `csvio.mojo` and `omxfile.mojo`;
`scripts/check` passes with the same numbers; and `scripts/build.sh omxplorer mac` links with no libpython.


**FLOAT as text.** Wherever a FLOAT is written as text (`mx` JSON / JSON lines / CSV), it is the shortest decimal that reads back to the same bits
(`0.1` is `0.1`), in Python's `repr` layout (`float_text` in `floatfmt.mojo`; Mojo's own `String(Float64)` is used up to 15 significant digits; a value of 16 or 17 is rebuilt from the exact decimal expansion, rounded half to even, because Mojo's text can be one digit short or break an exact tie upward: the result equals Python's `repr` and DuckDB's text digit for digit. A value needing 16 or 17 digits costs microseconds, more for extreme exponents). inf, -inf and NaN are `"Infinity"`, `"-Infinity"` and `"NaN"` in JSON.
