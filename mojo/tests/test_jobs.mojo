# F-175: the real jobs (mojo/omxcore/jobs.mojo): every kind of query, read and save, run as a background job, gives exactly what the plain
# call gives; progress only grows and ends at the total; a cancel (at any point) returns no partial result, frees everything and
# leaves a saved file as it was; several jobs run at once; a job over an edited Frame runs while the caller reads cells.
from std.ffi import external_call, c_int
from std.os import makedirs
from omxcore import (Table, SharedTable, Frame, Query, QueryResult, col, lit, sum, count, mean, min_of, max_of, count_distinct, median,
                     percentile, top_k, INT, DECIMAL, TEXT, FLOAT, write_parquet, read_parquet, scan_parquet, ParquetResult, ZSTD,
                     UNCOMPRESSED, CPU, AUTO, MORSELS_ON, MORSELS_OFF, Job, JOB_RUNNING, JOB_DONE, JOB_FAILED, JOB_CANCELLED,
                     compute_job, permutation_job, sql_job, read_csv_job, read_parquet_job, save_job, write_parquet_job, Catalog, PlannerOptions,
                     GROUP_HASHED, sql, read_csv, CsvOptions, SqlResult, DeviceTable, gpu_available, GPU)
from std.os import listdir
from std.time import perf_counter_ns
from omxcore import JOBS_ARE_THREADED
from omxcore.procmem import rss_kb

comptime DIR = "mxframe_jobs_test/"


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed * 0x9E3779B97F4A7C15 + 1

    def next(mut self) -> UInt64:
        self.s ^= self.s << 13
        self.s ^= self.s >> 7
        self.s ^= self.s << 17
        return self.s

    def int(mut self, lo: Int, hi: Int) -> Int:       # [lo, hi]
        return lo + Int(self.next() % UInt64(hi - lo + 1))


def pause(us: Int):
    external_call["usleep", c_int](Int32(us))


# ── comparing results ────────────────────────────────────────────────────────────────────────────

def same_ints(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def same_rows(a: List[List[Int]], b: List[List[Int]]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if not same_ints(a[i], b[i]):
            return False
    return True


def same_result(a: QueryResult, b: QueryResult) -> Bool:
    return (same_ints(a.kinds, b.kinds) and same_ints(a.slot_of, b.slot_of) and same_ints(a.count_slot_of, b.count_slot_of)
            and same_rows(a.slots, b.slots) and same_ints(a.counts, b.counts) and same_ints(a.sums, b.sums)
            and same_ints(a.hol_of, b.hol_of) and same_rows(a.hvals, b.hvals) and same_rows(a.htop, b.htop)
            and same_rows(a.htop_off, b.htop_off) and same_rows(a.hcount, b.hcount))


def same_table(a: Table, b: Table) -> Bool:
    if a.columns() != b.columns() or a.rows() != b.rows():
        return False
    for c in range(a.columns()):
        ref x = a.cols[c]
        ref y = b.cols[c]
        if x.name != y.name or x.kind != y.kind or x.scale != y.scale or x.nulls != y.nulls:
            return False
        for r in range(a.rows()):
            if x.is_null(r) != y.is_null(r) or x.values[r] != y.values[r]:
                return False
        if x.kind == TEXT:
            if len(x.dict[].words) != len(y.dict[].words):
                return False
            for w in range(len(x.dict[].words)):
                if x.dict[].words[w] != y.dict[].words[w]:
                    return False
    return True


# ── watching a job ───────────────────────────────────────────────────────────────────────────────

def watch[T: Movable & Deinitable](mut job: Job[T], what: String) raises -> Int:
    """Polls the job until it ends, checking progress is monotonic; returns how many distinct progress values were seen."""
    var last = 0
    var seen = 0
    while job.poll() == JOB_RUNNING:
        var p = job.progress()
        expect(p.done >= last, what + ": progress went back from " + String(last) + " to " + String(p.done))
        expect(p.done <= p.total or p.total == 0, what + ": progress passed its total: " + String(p.done) + "/" + String(p.total))
        if p.done != last:
            seen += 1
        last = p.done
        pause(200)
    var p = job.progress()
    expect(p.done >= last, what + ": progress went back at the end")
    if job.poll() == JOB_DONE:
        expect(p.done == p.total, what + ": progress ends at the total: " + String(p.done) + "/" + String(p.total))
    return seen


# ── the data ─────────────────────────────────────────────────────────────────────────────────────

def big_table(n: Int) raises -> SharedTable:
    """k INT 0..6 (some null), v INT, w DECIMAL(2) with nulls, x FLOAT, name TEXT."""
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var v = t.add_column("v", INT)
    var w = t.add_column("w", DECIMAL, scale=2)
    var x = t.add_column("x", FLOAT)
    var nm = t.add_column("name", TEXT)
    var names: List[String] = ["alpha", "beta", "gamma", "delta", "epsilon"]
    var rng = Rng(11)
    for i in range(n):
        if i % 97 == 5:
            t.append_null(k)
        else:
            t.append(k, Int64(i % 7))
        t.append(v, Int64(rng.int(-1000, 1000)))
        if i % 31 == 2:
            t.append_null(w)
        else:
            t.append(w, Int64(rng.int(0, 99999)))
        t.append_float(x, Float64(rng.int(-100000, 100000)) / 64.0)
        t.append_text(nm, names[rng.int(0, 4)])
    return t^.share()


def agg_query(t: SharedTable) raises -> Query:
    return Query(t).where(col("v") > -900).group_by(col("k"), groups=7).agg(sum(col("v")), count(), count(col("w")), sum(col("w")),
                                                                              min_of(col("v")), max_of(col("w")), mean(col("v")))


def test_compute_equals_plain() raises:
    var t = big_table(6_000_000)                       # more than two job ranges: the query runs in several
    var plain = agg_query(t).compute(CPU)
    var job = compute_job(agg_query(t), CPU)
    var seen = watch(job, "compute")
    var r = job.result()
    var most = seen
    for _ in range(9):                                  # (ranges of 2M rows: progress is seen part way when it runs by ranges)
        var again = compute_job(agg_query(t), CPU)
        most = max(most, watch(again, "compute again"))
    comptime if JOBS_ARE_THREADED:                      # (a job that runs inside its start call is done before anyone can look)
        expect(most >= 1, "progress was seen part way through a 6M-row query: the ranges ran (" + String(most) + ")")
    expect(same_result(r, plain), "integer aggregates over 6M rows: the job's result is the plain call's")
    expect(r.counts[0] > 0 and r.sums[0] == plain.sums[0], "a real answer (group 0 has rows)")
    var flo = Query(t).group_by(col("k"), groups=7).agg(sum(col("x")), min_of(col("x")), count()).compute(CPU)
    var fj = compute_job(Query(t).group_by(col("k"), groups=7).agg(sum(col("x")), min_of(col("x")), count()), CPU)
    expect(same_result(fj.result(), flo), "a float64 sum is the same bits as the plain call (it keeps its defined order)")
    var g = Query(t).where(col("w") > 5000).agg(sum(col("w")), count()).compute(CPU)
    var gj = compute_job(Query(t).where(col("w") > 5000).agg(sum(col("w")), count()), CPU)
    expect(same_result(gj.result(), g), "no group key")


def test_holistic_equals_plain() raises:
    var t = big_table(400_000)
    var q = Query(t).group_by(col("k"), groups=7).agg(count_distinct(col("v")), median(col("w")), percentile(col("v"), 0.9), top_k(col("v"), 3))
    q._window_rows = 65536 * 2                          # six windows
    var plain = q.compute(CPU)
    var q2 = Query(t).group_by(col("k"), groups=7).agg(count_distinct(col("v")), median(col("w")), percentile(col("v"), 0.9), top_k(col("v"), 3))
    q2._window_rows = 65536 * 2
    var job = compute_job(q2^, CPU)
    _ = watch(job, "holistic")
    expect(same_result(job.result(), plain), "count_distinct, median, percentile and top_k over windows: the plain call's answer")


def scan_queries(path: String, mode: Int, which: Int) raises -> Query:
    var q = Query(scan_parquet(path)).morsels(mode).block_rows(8192)
    if which == 0:
        return q^.where(col("v") > -900).group_by(col("k"), groups=7).agg(sum(col("v")), count(), count(col("w")), sum(col("w")), min_of(col("v")))
    if which == 1:
        return q^.group_by(col("k"), groups=7).agg(sum(col("x")), min_of(col("x")), count())          # float: file order
    if which == 2:
        return q^.group_by(col("name"), groups=5).agg(count(), sum(col("v")))                        # a TEXT key
    return q^.group_by(col("k"), groups=7).agg(count_distinct(col("v")), median(col("w")), top_k(col("v"), 2))     # holistic: row group by row group


def test_scan_equals_plain() raises:
    var t = big_table(1_500_000)
    var path = DIR + "scan.parquet"
    write_parquet(path, t, "", UNCOMPRESSED, 100_000)         # fifteen row groups
    for mode in [MORSELS_ON, MORSELS_OFF]:
        for which in range(4):
            var plain = scan_queries(path, mode, which).compute(CPU)
            var job = compute_job(scan_queries(path, mode, which), CPU)
            _ = watch(job, "scan " + String(which))
            var r = job.result()
            expect(same_result(r, plain), "scan query " + String(which) + " (morsels " + String(mode) + "): the plain call's answer")
            var p = job.progress()
            expect(p.done == 1_500_000 and p.total == 1_500_000, "the scan's progress is its rows: " + String(p.done) + "/" + String(p.total))


def test_cancel_storm() raises:
    """Cancels at random moments, drops handles mid-run, lets some finish: never a partial result, never a crash, and the memory
    after hundreds of cycles is the memory before (the first cycles warm the allocator)."""
    var t = big_table(5_000_000)
    var path = DIR + "storm.parquet"
    write_parquet(path, t, "", UNCOMPRESSED, 250_000)
    var want_mem = agg_query(t).compute(CPU)
    var want_scan = scan_queries(path, MORSELS_ON, 0).compute(CPU)
    var want_hol = scan_queries(path, MORSELS_OFF, 3).compute(CPU)
    var wants = List[QueryResult]()
    wants.append(want_mem^)
    wants.append(want_scan^)
    wants.append(want_hol^)
    var rng = Rng(99)
    var done = 0
    var cancelled = 0
    var base = 0
    comptime CYCLES = 300
    for cycle in range(CYCLES):
        if cycle == 40:
            base = rss_kb()
        var kind = cycle % 3
        var job = compute_job(agg_query(t) if kind == 0 else (scan_queries(path, MORSELS_ON, 0) if kind == 1 else scan_queries(path, MORSELS_OFF, 3)), CPU)
        pause(rng.int(0, 4000))
        var how = rng.int(0, 3)
        if how == 0:                                   # dropped while it may still run
            _ = job^
            continue
        if how != 1:
            job.cancel()
        var state = job.wait()
        if state == JOB_CANCELLED:
            cancelled += 1
            var msg = String()
            try:
                _ = job.result()
            except e:
                msg = String(e)
            expect(msg == "cancelled", "a cancelled job has no result, it says `cancelled`: `" + msg + "`")
        else:
            expect(state == JOB_DONE, "a job ends done or cancelled, not " + String(state))
            done += 1
            var got = job.result()
            expect(same_result(got, wants[kind]), "a job that finished before its cancel gives the full answer (kind " + String(kind) + ")")
    var grown = rss_kb() - base
    print("cancel storm:", CYCLES, "cycles,", done, "done,", cancelled, "cancelled, memory grew", grown, "KB after the first 40")
    expect(grown < 60_000, "memory grew " + String(grown) + " KB over 260 cycles (a leak per cycle would be tens of MB)")
    comptime if JOBS_ARE_THREADED:
        expect(cancelled > 20, "the cancels reached running jobs (" + String(cancelled) + ")")


def test_cancel_under_load() raises:
    """Eight jobs of every kind at once, a random half cancelled at random moments: the survivors answer exactly, the cancelled have no
    result, and the memory after many rounds is the memory before."""
    var t = big_table(3_000_000)
    var path = DIR + "load.parquet"
    write_parquet(path, t, "", UNCOMPRESSED, 250_000)
    var cat = Catalog()
    cat.add_table("t", t)
    cat.add_scan("s", scan_parquet(path))
    var want_q = agg_query(t).compute(CPU)
    var want_s = scan_queries(path, MORSELS_ON, 0).compute(CPU)
    var sql_text = "SELECT name, count(*), sum(v) FROM s GROUP BY name ORDER BY name"
    var want_sql = sql(cat, sql_text)
    var want_rows = sql(cat, "SELECT k, v FROM t WHERE v > 990 ORDER BY v DESC, k LIMIT 50")
    var rng = Rng(21)
    var base = 0
    var survivors = 0
    for round in range(30):
        if round == 5:
            base = rss_kb()
        var jobs = List[Job[QueryResult]]()
        var sqls = List[Job[SqlResult]]()
        var reads = List[Job[ParquetResult]]()
        for i in range(3):
            jobs.append(compute_job(agg_query(t) if i % 2 == 0 else scan_queries(path, MORSELS_ON, 0), CPU))
        sqls.append(sql_job(cat, sql_text))
        sqls.append(sql_job(cat, "SELECT k, v FROM t WHERE v > 990 ORDER BY v DESC, k LIMIT 50"))
        reads.append(read_parquet_job(path))
        pause(rng.int(0, 3000))
        for i in range(len(jobs)):
            if rng.int(0, 1) == 1:
                jobs[i].cancel()
        for i in range(len(sqls)):
            if rng.int(0, 1) == 1:
                sqls[i].cancel()
        if rng.int(0, 1) == 1:
            reads[0].cancel()
        for i in range(len(jobs)):
            if jobs[i].wait() == JOB_DONE:
                survivors += 1
                var got = jobs[i].result()
                expect(same_result(got, want_q if i % 2 == 0 else want_s), "a surviving query job answers exactly")
            else:
                expect(jobs[i].wait() == JOB_CANCELLED, "a job ends done or cancelled")
        for i in range(len(sqls)):
            if sqls[i].wait() == JOB_DONE:
                survivors += 1
                expect(same_table(sqls[i].result().table, (want_sql if i == 0 else want_rows).table), "a surviving sql job answers exactly")
        if reads[0].wait() == JOB_DONE:
            survivors += 1
            expect(reads[0].result().table.rows() == 3_000_000, "a surviving read has every row")
        else:
            expect(reads[0].wait() == JOB_CANCELLED, "a read ends done or cancelled")
    var grown = rss_kb() - base
    print("cancel under load: 30 rounds of 6 jobs,", survivors, "survived, memory grew", grown, "KB after round 5")
    expect(grown < 120_000, "memory grew " + String(grown) + " KB")


def test_gpu_job() raises:
    if not gpu_available():
        return
    var t = big_table(2_000_000)
    var d = DeviceTable(t)
    var plain = Query(t).on_device(d).where(col("v") > -900).group_by(col("k"), groups=7).agg(sum(col("v")), count()).compute(GPU)
    var job = compute_job(Query(t).on_device(d).where(col("v") > -900).group_by(col("k"), groups=7).agg(sum(col("v")), count()), GPU)
    _ = watch(job, "gpu")
    expect(same_result(job.result(), plain), "a job on the GPU gives the plain call's answer")


# ── SQL ──────────────────────────────────────────────────────────────────────────────────────────

def sql_catalog(t: SharedTable, path: String) raises -> Catalog:
    var cat = Catalog()
    cat.add_table("t", t)
    cat.add_scan("s", scan_parquet(path))
    var f = Frame(t)
    f.set(f.column("v"), 5, Int64(777))
    f.set(f.column("v"), 123_456, Int64(-5))
    f.remove(77)
    cat.add_frame("f", f)
    var u = Table.empty()
    var uv = u.add_column("v", INT)
    var ug = u.add_column("grp", INT)
    for i in range(-1000, 1001):
        u.append(uv, Int64(i))
        u.append(ug, Int64(abs(i) % 5))
    cat.add_table("u", u^.share())
    return cat^


def test_sql_equals_plain() raises:
    var t = big_table(3_000_000)
    var path = DIR + "sql.parquet"
    write_parquet(path, t, "", UNCOMPRESSED, 250_000)
    var cat = sql_catalog(t, path)
    var queries: List[String] = [
        "SELECT k, count(*), sum(v), avg(w), min(v), max(w) FROM {} WHERE v > -900 GROUP BY k ORDER BY k",
        "SELECT name, count(*), sum(v) FROM {} GROUP BY name ORDER BY name",
        "SELECT k, count(DISTINCT v), median(w) FROM {} GROUP BY k HAVING count(*) > 10 ORDER BY k",
        "SELECT k, v, w FROM {} WHERE v > 990 AND w IS NOT NULL ORDER BY v DESC, k LIMIT 25",
        "SELECT * FROM {} WHERE v = 999 LIMIT 40",
        "SELECT DISTINCT k, name FROM {} ORDER BY k, name",
        "SELECT k, v + 1, w FROM {} WHERE v > 995 ORDER BY 2, 1",
        "SELECT count(*), sum(x) FROM {}"]
    var hashed = PlannerOptions()
    hashed.group = GROUP_HASHED
    for source in ["t", "s", "f"]:
        for qi in range(len(queries)):
            var text = String(queries[qi]).replace("{}", source)
            for h in range(2):
                var opts = hashed.copy() if h == 1 else PlannerOptions()
                var plain = sql(cat, text, opts)
                var job = sql_job(cat, text, opts)
                _ = watch(job, text)
                var r = job.result()
                expect(same_table(r.table, plain.table), "sql job == sql: " + text + " (hashed " + String(h) + ")")
                expect(len(r.notes) > 0 and r.notes[0] == plain.notes[0], "same plan: " + text)
    for source in ["t", "s", "f"]:
        var jtext = "SELECT b.grp, count(*), sum(a.w) FROM " + source + " a JOIN u b ON a.v = b.v GROUP BY b.grp ORDER BY b.grp"
        var j2 = sql_job(cat, jtext)
        var want = sql(cat, jtext)
        expect(same_table(j2.result().table, want.table), "a join of " + source + " with a small table")
    var bad = sql_job(cat, "SELECT nope FROM t")
    expect(bad.wait() == JOB_FAILED, "a query that cannot be bound fails the job")
    var msg = String()
    try:
        _ = bad.result()
    except e:
        msg = String(e)
    var plain_msg = String()
    try:
        _ = sql(cat, "SELECT nope FROM t")
    except e:
        plain_msg = String(e)
    expect(msg == plain_msg and msg != "", "and result() raises the plain call's error: `" + msg + "`")


# ── CSV and Parquet reads ─────────────────────────────────────────────────────────────────────────

def write_csv(path: String, rows: Int) raises:
    var f = open(path, "w")
    f.write("id,name,amount,when,flag,ratio\n")
    var text = String()
    var rng = Rng(5)
    for i in range(rows):
        if i % 17 == 3:
            text += String(i) + ",x,,,,\n"
        else:
            text += (String(i) + ",n" + String(rng.int(0, 40)) + "," + String(rng.int(0, 99999) // 100) + "." + String(rng.int(10, 99)) +
                     ",2024-0" + String(rng.int(1, 9)) + "-1" + String(rng.int(0, 9)) + "," + ("true" if rng.int(0, 1) == 1 else "false") +
                     "," + String(rng.int(1, 1000)) + ".5e-1\n")
        if text.byte_length() > 1_000_000:
            f.write(text)
            text = String()
    f.write(text)
    f.close()


def test_csv_job() raises:
    var path = DIR + "rows.csv"
    write_csv(path, 400_000)
    for threads in [0, 1]:
        var opts = CsvOptions(threads=threads, chunk_bytes=1 << 16)
        var plain = read_csv(path, opts)
        var job = read_csv_job(path, opts)
        _ = watch(job, "csv")
        var r = job.result()
        expect(same_table(r.table, plain.table) and r.rows == plain.rows and r.rows == 400_000, "read_csv job == read_csv (threads " + String(threads) + ")")
        expect(len(r.columns) == len(plain.columns), "same import report")
        for c in range(len(r.columns)):
            expect(r.columns[c].failed_count == plain.columns[c].failed_count and r.columns[c].empty == plain.columns[c].empty, "same report for column " + String(c))
        var p = job.progress()
        expect(p.bytes and p.done == p.total and p.total > 1_000_000, "a CSV's progress counts its bytes: " + String(p.done) + "/" + String(p.total))
    var gone = read_csv_job(DIR + "no-such.csv")
    expect(gone.wait() == JOB_FAILED, "an unreadable file fails the job")


def test_parquet_read_job() raises:
    var t = big_table(1_200_000)
    var path = DIR + "read.parquet"
    write_parquet(path, t, "{\"note\": 1}", UNCOMPRESSED, 100_000)
    var plain = read_parquet(path)
    var job = read_parquet_job(path)
    _ = watch(job, "read_parquet")
    var r = job.result()
    expect(same_table(r.table, plain.table) and r.metadata == plain.metadata, "read_parquet job == read_parquet")
    var p = job.progress()
    expect(p.done == 1_200_000 and p.total == 1_200_000, "rows done of rows: " + String(p.done) + "/" + String(p.total))


# ── saves ────────────────────────────────────────────────────────────────────────────────────────

def file_bytes(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var b = f.read_bytes()
    f.close()
    return b^


def same_bytes(a: List[UInt8], b: List[UInt8]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def no_temp_files() raises -> Bool:
    for name in listdir(DIR):
        if name.endswith(".tmp"):
            return False
    return True


def edited_frame(t: SharedTable) raises -> Frame:
    var f = Frame(t)
    f.set(f.column("v"), 3, Int64(31))
    f.set_null(f.column("w"), 40)
    f.set_text(f.column("name"), 41, "zeta")
    f.remove(1000)
    _ = f.append(["3", "5", "1.25", "2.5", "omega"])
    return f^


def test_save_equals_plain() raises:
    var t = big_table(600_000)
    var f = edited_frame(t)
    f.save(DIR + "plain.parquet", "{\"a\": 1}", UNCOMPRESSED)
    var job = save_job(f, DIR + "job.parquet", "{\"a\": 1}", UNCOMPRESSED)
    _ = watch(job, "save")
    expect(job.result() == f.rows() - 1, "the job's result is the rows written (one row was removed)")
    expect(same_bytes(file_bytes(DIR + "plain.parquet"), file_bytes(DIR + "job.parquet")), "the saved file is the plain save's, byte for byte")
    # a frame over a file
    var path = DIR + "src.parquet"
    write_parquet(path, t, "", UNCOMPRESSED, 100_000)
    var sf = Frame(scan_parquet(path))
    sf.set(sf.column("v"), 5, Int64(55))
    sf.remove(250_000)
    _ = sf.append(["1", "2", "3.5", "4.5", "beta"])
    sf.save(DIR + "plain2.parquet", "", UNCOMPRESSED)
    var j2 = save_job(sf, DIR + "job2.parquet", "", UNCOMPRESSED)
    _ = watch(j2, "save over a file")
    var written = j2.result()
    expect(written == sf.rows() - 1 and written == read_parquet(DIR + "job2.parquet").table.rows(), "rows written " + String(written) + " are the saved file's rows")
    var p = j2.progress()
    expect(p.done == p.total and p.total >= 600_000, "progress ends at its total: " + String(p.done) + "/" + String(p.total))
    expect(same_bytes(file_bytes(DIR + "plain2.parquet"), file_bytes(DIR + "job2.parquet")), "a frame over a file: the plain save's bytes")
    var wj = write_parquet_job(t, DIR + "wjob.parquet", "", UNCOMPRESSED, 100_000)
    expect(wj.result() == 600_000, "write_parquet_job writes the table")
    write_parquet(DIR + "wplain.parquet", t, "", UNCOMPRESSED, 100_000)
    expect(same_bytes(file_bytes(DIR + "wplain.parquet"), file_bytes(DIR + "wjob.parquet")), "write_parquet_job == write_parquet, byte for byte")


def test_cancelled_save_keeps_the_old_file() raises:
    var old = big_table(1000)
    var path = DIR + "keep.parquet"
    write_parquet(path, old, "", UNCOMPRESSED)
    var old_bytes = file_bytes(path)
    var big = big_table(2_000_000)
    var f = edited_frame(big)
    var path_new = DIR + "new-want.parquet"
    f.save(path_new, "", UNCOMPRESSED)
    var new_bytes = file_bytes(path_new)
    var rng = Rng(3)
    var kept = 0
    var replaced = 0
    for cycle in range(60):
        write_parquet(path, old, "", UNCOMPRESSED)
        var job = save_job(f, path, "", UNCOMPRESSED)
        pause(rng.int(0, 12000))
        var how = rng.int(0, 2)
        if how == 0:
            _ = job^                                       # dropped: the save is cancelled or finishes on its own
            pause(60000)
            var now = file_bytes(path)
            expect(same_bytes(now, old_bytes) or same_bytes(now, new_bytes), "a dropped save leaves the old file or the whole new one")
            continue
        job.cancel()
        var state = job.wait()
        var now = file_bytes(path)
        if state == JOB_CANCELLED:
            kept += 1
            expect(same_bytes(now, old_bytes), "a cancelled save leaves the old file as it was (cycle " + String(cycle) + ")")
        else:
            replaced += 1
            expect(state == JOB_DONE and same_bytes(now, new_bytes), "a save that was past cancelling is whole")
    expect(no_temp_files(), "no temporary file is left behind")
    print("cancelled saves:", kept, "kept the old file,", replaced, "finished first")
    comptime if JOBS_ARE_THREADED:
        expect(kept > 5, "the cancels reached running saves (" + String(kept) + ")")


# ── at once, and over an edited frame ─────────────────────────────────────────────────────────────

def test_concurrent_jobs() raises:
    var tables = List[SharedTable]()
    var wants = List[QueryResult]()
    for i in range(6):
        tables.append(big_table(700_000 + 100_000 * i))
        wants.append(agg_query(tables[i]).compute(CPU))
    var jobs = List[Job[QueryResult]]()
    for i in range(6):
        jobs.append(compute_job(agg_query(tables[i]), CPU))
    var cat = Catalog()
    cat.add_table("t", tables[0])
    var sj = sql_job(cat, "SELECT k, count(*), sum(v) FROM t GROUP BY k ORDER BY k")
    var cj = read_csv_job(DIR + "rows.csv")
    for i in range(6):
        expect(same_result(jobs[i].result(), wants[i]), "job " + String(i) + " over its own table gives its own answer")
    var want_sql = sql(cat, "SELECT k, count(*), sum(v) FROM t GROUP BY k ORDER BY k")
    expect(same_table(sj.result().table, want_sql.table), "the sql job beside them")
    expect(cj.result().rows == 400_000, "and the csv job")


def test_job_over_edited_frame() raises:
    var t = big_table(4_000_000)
    var f = edited_frame(t)
    var vc = f.column("v")
    var want = Query(f).where(col("v") > -900).group_by(col("k"), groups=7).agg(sum(col("v")), count(), min_of(col("v"))).compute(CPU)
    var job = compute_job(Query(f).where(col("v") > -900).group_by(col("k"), groups=7).agg(sum(col("v")), count(), min_of(col("v"))), CPU)
    # the caller reads cells, edits more and starts a SQL job on the frame while the first job runs
    var reads = 0
    var sum_read = Int64(0)
    var cat = Catalog()
    cat.add_frame("f", f)
    var sj = sql_job(cat, "SELECT k, count(*), sum(v) FROM f GROUP BY k ORDER BY k")
    var before_sql = sql(cat, "SELECT k, count(*), sum(v) FROM f GROUP BY k ORDER BY k")
    while job.poll() == JOB_RUNNING or reads < 200_000:
        sum_read += f.get(vc, reads % 4_000_000)
        reads += 1
        if reads % 50_000 == 0:
            f.set(vc, reads % 4_000_000, Int64(-reads))        # later edits are not in the job's answer
    expect(same_result(job.result(), want), "the job answers over the frame as it was when it started")
    expect(same_table(sj.result().table, before_sql.table), "so does the sql job (the catalog's frame was snapshotted)")
    expect(reads >= 200_000 and sum_read != 0, "the caller read cells meanwhile")
    # a frame over a file: the caller reads cells (and so the file's text dictionary) while the job reads the file
    var path = DIR + "frame.parquet"
    write_parquet(path, t, "", UNCOMPRESSED, 250_000)
    var sf = Frame(scan_parquet(path))
    sf.set(sf.column("v"), 9, Int64(99))
    var swant = Query(sf).group_by(col("name"), groups=sf.word_count(sf.column("name"))).agg(sum(col("v")), count()).compute(CPU)
    var sjob = compute_job(Query(sf).group_by(col("name"), groups=sf.word_count(sf.column("name"))).agg(sum(col("v")), count()), CPU)
    var n = 0
    while sjob.poll() == JOB_RUNNING or n < 2000:
        _ = sf.get(sf.column("v"), (n * 7919) % 4_000_000)
        _ = sf.text_of(sf.column("name"), Int64(n % 5))
        n += 1
    expect(same_result(sjob.result(), swant), "a frame over a file: the job's answer while the caller reads cells")


def temp_files() raises -> Int:
    var n = 0
    for name in listdir(DIR):
        if name.endswith(".tmp"):
            n += 1
    return n


def test_save_restart_and_two_saves() raises:
    """A save cancelled and at once started again, and two saves to one path at once: the file is always exactly one of the inputs."""
    var a = edited_frame(big_table(1_500_000))
    var b = Frame(big_table(1_000_000))
    var pa = DIR + "want-a.parquet"
    var pb = DIR + "want-b.parquet"
    a.save(pa, "", UNCOMPRESSED)
    b.save(pb, "", UNCOMPRESSED)
    var want_a = file_bytes(pa)
    var want_b = file_bytes(pb)
    var path = DIR + "contested.parquet"
    var rng = Rng(8)
    var failed = 0
    for cycle in range(40):
        var j1 = save_job(a, path, "", UNCOMPRESSED)
        pause(rng.int(0, 15000))
        j1.cancel()
        var j2 = save_job(b, path, "", UNCOMPRESSED)                    # no wait: the cancelled save may still be unwinding
        var s2 = j2.wait()
        _ = j1.wait()
        expect(s2 == JOB_DONE, "the restarted save is done, not failed (cycle " + String(cycle) + ")")
        var now = file_bytes(path)
        expect(same_bytes(now, want_b), "the file is the restarted save's, whole (cycle " + String(cycle) + ")")
    for cycle in range(20):
        var j1 = save_job(a, path, "", UNCOMPRESSED)
        var j2 = save_job(b, path, "", UNCOMPRESSED)
        var s1 = j1.wait()
        var s2 = j2.wait()
        expect(s1 == JOB_DONE and s2 == JOB_DONE, "two saves at once both finish")
        var now = file_bytes(path)
        expect(same_bytes(now, want_a) or same_bytes(now, want_b), "the file is exactly one of the two inputs (cycle " + String(cycle) + ")")
    expect(temp_files() == 0, "no temporary file is left: " + String(temp_files()))


def test_permutation_job() raises:
    var t = big_table(5_000_000)
    var f = edited_frame(t)
    for c in ["v", "w", "name", "x"]:
        for desc in [False, True]:
            var col_i = f.column(c)
            var plain = f.permutation(col_i, desc)
            var job = permutation_job(f, col_i, desc)
            _ = watch(job, "sort " + String(c))
            var got = job.result()
            expect(len(got) == len(plain) and same_ints(got, plain), "permutation_job == permutation: " + String(c) + " descending " + String(desc))
    comptime if JOBS_ARE_THREADED:                      # (a job that runs inside its start call is done before cancel() is called)
        var slow = permutation_job(f, f.column("v"))
        pause(5000)
        slow.cancel()
        var t0 = perf_counter_ns()
        expect(slow.wait() == JOB_CANCELLED, "a cancelled sort has no result")
        var ms = Float64(perf_counter_ns() - t0) / 1e6
        expect(ms < 400.0, "the sort stops within a few checks of the cancel: " + String(ms) + " ms")


def test_cancel_latency() raises:
    """A hashed group-by over a big table, and a join's probe side, are cut into ranges: a cancel is noticed within about 100 ms."""
    var t = big_table(30_000_000)
    var u = Table.empty()
    var uv = u.add_column("v", INT)
    var ug = u.add_column("grp", INT)
    for i in range(-1000, 1001):
        u.append(uv, Int64(i))
        u.append(ug, Int64(abs(i) % 5))
    var cat = Catalog()
    cat.add_table("t", t)
    cat.add_table("u", u^.share())
    var hashed = PlannerOptions()
    hashed.group = GROUP_HASHED
    var texts: List[String] = ["SELECT k, name, count(*), sum(v), min(w) FROM t GROUP BY k, name ORDER BY k, name",
                               "SELECT b.grp, count(*), sum(a.w) FROM t a JOIN u b ON a.v = b.v GROUP BY b.grp ORDER BY b.grp"]
    for qi in range(len(texts)):
        var want = sql(cat, texts[qi], hashed)
        var full = sql_job(cat, texts[qi], hashed)
        _ = watch(full, "latency " + String(qi))
        expect(same_table(full.result().table, want.table), "ranged hashed / joined sql job == sql: " + texts[qi])
        var worst = 0.0
        var hit = 0
        var stopped_early = 0
        for k in range(6):
            var job = sql_job(cat, texts[qi], hashed)
            pause(20000 + 15000 * k)
            job.cancel()
            var t0 = perf_counter_ns()
            if job.wait() == JOB_CANCELLED:
                hit += 1
                var pr = job.progress()
                if pr.done < pr.total:                       # the work stopped between ranges, not after its one call
                    stopped_early += 1
            worst = max(worst, Float64(perf_counter_ns() - t0) / 1e6)
        print("cancel latency, query", qi, ": worst", worst, "ms,", hit, "of 6 cancels reached a running job; a whole run takes", want.ms, "ms")
        comptime if JOBS_ARE_THREADED:
            expect(hit >= 3, "the cancels reached running jobs (" + String(hit) + ")")
            expect(stopped_early >= 1, "a cancelled job stopped part way through its rows (" + String(stopped_early) + " of " + String(hit) + ")")
        expect(worst < 250.0, "a cancel stops the job within about 100 ms (worst " + String(worst) + " ms): " + texts[qi])


def test_cancel_multiway() raises:
    """F-158: a three-table join as a job equals `sql`, and a cancel reaches it between the join's steps and tables."""
    var t = big_table(30_000_000)                      # (30M rows: a whole run takes tens of ms, so at 4M the job finished before the cancel landed)
    var u = Table.empty()
    var uv = u.add_column("v", INT)
    var ug = u.add_column("grp", INT)
    for i in range(-1000, 1001):
        u.append(uv, Int64(i))
        u.append(ug, Int64(abs(i) % 5))
    var x = Table.empty()
    var xg = x.add_column("grp", INT)
    var xl = x.add_column("label", INT)
    for g in range(5):
        x.append(xg, Int64(g))
        x.append(xl, Int64(100 + g))
    var cat = Catalog()
    cat.add_table("t", t)
    cat.add_table("u", u^.share())
    cat.add_table("x", x^.share())
    var text = "SELECT x.label, count(*), sum(a.w) FROM t a JOIN u b ON a.v = b.v JOIN x ON b.grp = x.grp GROUP BY x.label ORDER BY x.label"
    var want = sql(cat, text)
    var full = sql_job(cat, text)
    _ = watch(full, "multiway")
    expect(same_table(full.result().table, want.table), "a multi-way sql job == sql")
    var hit = 0
    var worst = 0.0
    for k in range(6):
        var job = sql_job(cat, text)
        pause(500 + 1500 * k)
        job.cancel()
        var t0 = perf_counter_ns()
        if job.wait() == JOB_CANCELLED:
            hit += 1
        worst = max(worst, Float64(perf_counter_ns() - t0) / 1e6)
    print("cancel of a multi-way join: worst", worst, "ms,", hit, "of 6 reached a running job; a whole run takes", want.ms, "ms")
    comptime if JOBS_ARE_THREADED:
        expect(hit >= 3, "the cancels reached running multi-way jobs (" + String(hit) + ")")
    expect(worst < 400.0, "a cancel stops a multi-way join within a few hundred ms (worst " + String(worst) + " ms)")


def main() raises:
    makedirs(DIR, exist_ok=True)
    test_compute_equals_plain()
    test_holistic_equals_plain()
    test_scan_equals_plain()
    test_cancel_storm()
    test_cancel_under_load()
    test_gpu_job()
    test_sql_equals_plain()
    test_csv_job()
    test_parquet_read_job()
    test_save_equals_plain()
    test_cancelled_save_keeps_the_old_file()
    test_concurrent_jobs()
    test_job_over_edited_frame()
    test_save_restart_and_two_saves()
    test_permutation_job()
    test_cancel_latency()
    test_cancel_multiway()
    print("test_jobs ok")
