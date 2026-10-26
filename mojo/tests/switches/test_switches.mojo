# F-138b: the package built for an app with no GPU, no thread pool and no codecs. scripts/mojo_test.sh builds this one
# file with `-D MXFRAME_NO_GPU -D MXFRAME_SERIAL -D MXFRAME_NO_CODECS` (the switches a Mac build has by default) and
# runs it with two Parquet files the normal build wrote: argv[1] zstd, argv[2] snappy (from tools/pq_write_sample).
# Everything an app does must still give the right answer; the GPU and the thread pool are gone and say so readably, and
# the codecs are the Mojo ones (F-180): files are saved compressed and the C libraries' files are read.
from std.sys import argv
from std.os import getenv
from omxcore import (Table, SharedTable, Frame, Query, QueryResult, INT, DECIMAL, col, sum, count, CPU, GPU, AUTO, DeviceTable,
                     gpu_available, read_csv, CsvOptions, write_parquet, read_parquet, scan_parquet, min_of, max_of, ZSTD, SNAPPY,
                     count_distinct, median, percentile, top_k, Catalog, PlannerOptions, GROUP_HASHED, GROUP_DENSE, sql,
                     compute_job, sql_job, read_csv_job, read_parquet_job, save_job, write_parquet_job, permutation_job, JOB_DONE, JOB_FAILED, JOB_CANCELLED)
from omxcore.job import THREADS
from omxcore.query import AUTO_GPU_MIN_ROWS
from omxcore.kernels._fpenv import core_count


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


def main() raises:
    var args = argv()
    # no GPU
    expect(not gpu_available(), "gpu_available() is False")
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var v = t.add_column("v", INT)
    for i in range(AUTO_GPU_MIN_ROWS + 5):                    # over AUTO's row threshold
        t.append(k, Int64(i % 3))
        t.append(v, Int64(i))
    var shared = t^.share()
    var msg = String("")
    try:
        _ = DeviceTable(shared)
    except e:
        msg = String(e)
    expect(msg == "no GPU on this build", "DeviceTable raises: `" + msg + "`")
    try:
        _ = Query(shared).group_by(col("k"), groups=3).agg(sum(col("v")), count()).compute(GPU)
    except e:
        msg = String(e)
    expect("no GPU on this build" in msg, "compute(GPU) raises: `" + msg + "`")
    var auto = Query(shared).group_by(col("k"), groups=3).agg(sum(col("v")), count()).compute(AUTO)
    expect(auto.device == CPU and auto.fallback_reason == "", "AUTO runs on the CPU, not as a fallback")
    var n = AUTO_GPU_MIN_ROWS + 5
    expect(auto.counts[0] == (n + 2) // 3 and auto.counts[1] == (n + 1) // 3 and auto.counts[2] == n // 3,
           "the serial kernel counts every row: " + String(auto.counts[0]) + " " + String(auto.counts[1]) + " " + String(auto.counts[2]))
    var want0 = 0
    for i in range(0, n, 3):
        want0 += i
    expect(auto.sums[0] == want0, "and sums them exactly: " + String(auto.sums[0]) + " vs " + String(want0))

    # count_distinct / percentile / top_k with no thread pool (every task runs in a loop on this thread), against a hand count
    var hol = Query(shared).group_by(col("k"), groups=3).agg(count_distinct(col("v") % 1000), median(col("v")), percentile(col("v"), 0.9),
                                                             top_k(col("v"), 2)).compute(AUTO)
    for g in range(3):
        var m = (n - g + 2) // 3                                  # group g holds v = g, g + 3, g + 6, ... below n
        var idx = Int(0.5 * Float64(m))
        if Float64(idx) < 0.5 * Float64(m):
            idx += 1
        var idx9 = Int(0.9 * Float64(m))
        if Float64(idx9) < 0.9 * Float64(m):
            idx9 += 1
        expect(hol.values(0)[g] == 1000, "count_distinct(v % 1000) of group " + String(g) + ": " + String(hol.values(0)[g]))
        expect(hol.values(1)[g] == 5 * ((g + 3 * (idx - 1)) + (g + 3 * (m // 2))), "median of group " + String(g) + ": " + String(hol.values(1)[g]))
        expect(hol.values(2)[g] == g + 3 * (idx9 - 1), "p90 of group " + String(g) + ": " + String(hol.values(2)[g]))
        var top = hol.top(3, g)
        expect(len(top) == 2 and top[0] == g + 3 * (m - 1) and top[1] == g + 3 * (m - 2), "top 2 of group " + String(g))

    # SQL with no thread pool: dense groups, hashed groups (hashagg's start_pool and its partitioned merge on this thread) and the
    # two-pass holistic aggregates, against hand counts
    var cat = Catalog()
    cat.add_table("t", shared)
    for strategy in [GROUP_DENSE, GROUP_HASHED]:
        var po = PlannerOptions()
        po.group = strategy
        var r = sql(cat, "SELECT k, count(*), sum(v), count(DISTINCT v % 1000), median(v) FROM t GROUP BY k ORDER BY k", po)
        var hashed = strategy == GROUP_HASHED
        var seen = False
        for note in r.notes:
            if note.startswith("group: hashed" if hashed else "group: dense"):
                seen = True
        expect(seen, "SQL ran the " + ("hashed" if hashed else "dense") + " strategy")
        expect(r.table.rows() == 3, "SQL gives three groups")
        for g in range(3):
            var m = (n - g + 2) // 3
            expect(r.table.get(1, g) == Int64(m), "SQL count of group " + String(g) + ": " + String(r.table.get(1, g)))
            expect(r.table.get(2, g) == Int64(m * g + 3 * (m * (m - 1) // 2)), "SQL sum of group " + String(g) + ": " + String(r.table.get(2, g)))
            expect(r.table.get(3, g) == 1000, "SQL count(DISTINCT) of group " + String(g))
            var idx = Int(0.5 * Float64(m))
            if Float64(idx) < 0.5 * Float64(m):
                idx += 1
            var mid = Float64((g + 3 * (idx - 1)) + (g + 3 * (m // 2))) / 2.0
            expect(Float64(from_bits=UInt64(r.table.get(4, g))) == mid, "SQL median of group " + String(g))

    # F-152: row queries on this thread: the materialize sink (WHERE + ORDER BY), the top-k heaps (a small LIMIT), SELECT DISTINCT
    var top_v = (n - 1) - ((n - 1 - 7) % 100)                     # the largest v below n with v % 100 == 7
    var rr = sql(cat, "SELECT k, v FROM t WHERE v % 100 = 7 ORDER BY v DESC")
    expect(rr.table.rows() == (n - 1 - 7) // 100 + 1, "row query: " + String(rr.table.rows()) + " rows")
    expect(rr.table.get(1, 0) == Int64(top_v) and rr.table.get(1, 1) == Int64(top_v - 100) and rr.table.get(0, 0) == Int64(top_v % 3), "row query ORDER BY v DESC")
    var tk = sql(cat, "SELECT k, v FROM t WHERE v % 100 = 7 ORDER BY v DESC LIMIT 2 OFFSET 1")
    expect(tk.table.rows() == 2 and tk.table.get(1, 0) == Int64(top_v - 100) and tk.table.get(1, 1) == Int64(top_v - 200), "top-k rows")
    var heaps = False
    var topk3 = sql(cat, "SELECT v FROM t ORDER BY v DESC LIMIT 3")
    for note in topk3.notes:
        if note.startswith("order: top-3 by per-worker heaps"):
            heaps = True
    expect(heaps and topk3.table.get(0, 0) == Int64(n - 1), "a small LIMIT over many rows keeps its best rows in per-worker heaps")
    var dist = sql(cat, "SELECT DISTINCT k FROM t ORDER BY k")
    expect(dist.table.rows() == 3 and dist.table.get(0, 2) == 2, "SELECT DISTINCT")

    # F-153: the inner join with no thread pool: unique keys fused, the same join materialized, and repeated keys (many to many)
    var jf = sql(cat, "SELECT count(*), sum(b.k) FROM t a JOIN t b ON a.v = b.v")
    var nofuse = PlannerOptions()
    nofuse.fuse = False
    var jm = sql(cat, "SELECT count(*), sum(b.k) FROM t a JOIN t b ON a.v = b.v", nofuse)
    var fused = False
    for note in jf.notes:
        if note.startswith("join path: fused"):
            fused = True
    expect(fused and jf.table.get(0, 0) == Int64(n) and jm.table.get(0, 0) == Int64(n) and jf.table.get(1, 0) == jm.table.get(1, 0), "join, fused and materialized")
    var mm = sql(cat, "SELECT count(*) FROM t a JOIN t b ON a.k = b.k WHERE a.v < 3000 AND b.v < 3000")      # 1,000 rows of each k on both sides
    expect(mm.table.get(0, 0) == Int64(3_000_000), "join with repeated keys: " + String(mm.table.get(0, 0)) + " wanted 3000000")
    var refused = False
    try:
        _ = sql(cat, "SELECT count(*) FROM t a JOIN t b ON a.k = b.k")
    except e:
        refused = "memory budget" in String(e)
    expect(refused, "a join past the budget is refused before it allocates")

    # no thread pool: the parallel CSV reader runs its chunks one after the other
    var csv = String("mxframe-switches.csv")
    var text = String("a,b\n")
    for i in range(50_000):
        text += String(i) + "," + String(i % 7) + "\n"
    with open(csv, "w") as f:
        f.write(text)
    var r = read_csv(csv)
    expect(r.table.rows() == 50_000, "read_csv rows: " + String(r.table.rows()))
    var total = 0
    for i in range(r.table.rows()):
        total += Int(r.table.get(1, i))
    expect(total == 149_997, "read_csv values: " + String(total))      # 7142 full cycles of 0..6 (21 each) plus 0+1+2+3+4+5 = 15

    # the serial `par` / `par_n` really run every chunk: many small chunks (quoted newlines across their edges)
    # read cell for cell as one chunk does
    expect(core_count() == 1, "core_count() is 1 on a serial build: " + String(core_count()))
    var qcsv = String("mxframe-switches-quoted.csv")
    var qtext = String("id,note,n\n")
    for i in range(20_000):
        qtext += String(i) + ",\"line " + String(i) + "\nnext, " + String(i % 13) + "\"," + String(i % 5) + "\n"
    with open(qcsv, "w") as f:
        f.write(qtext)
    var one = read_csv(qcsv, CsvOptions(threads=1))
    var many = read_csv(qcsv, CsvOptions(threads=4, chunk_bytes=4096))
    expect(one.table.rows() == 20_000 and many.table.rows() == 20_000,
           "chunked rows: " + String(one.table.rows()) + " / " + String(many.table.rows()))
    for i in range(20_000):
        expect(many.table.get(0, i) == one.table.get(0, i) and many.table.get(2, i) == one.table.get(2, i) and
               many.table.text_of(1, Int(many.table.get(1, i))) == one.table.text_of(1, Int(one.table.get(1, i))), "chunked read equals one chunk at row " + String(i))

    # no C codecs: the Mojo zstd and snappy save compressed files (1,000 values of ten kinds: 8,000 bytes plain), and they read back
    var path = String("mxframe-switches.parquet")
    var tens = Table.empty()
    var ten = tens.add_column("ten", INT)
    for i in range(1000):
        tens.append(ten, Int64(i % 10))
    for codec in [ZSTD, SNAPPY]:
        write_parquet(path + ".tens", tens, "", codec)
        var tsize = len(open(path + ".tens", "r").read_bytes())
        expect(tsize < 700, "1,000 repeating values are stored compressed (" + String(tsize) + " bytes)")
        var tback = read_parquet(path + ".tens")
        expect(tback.table.rows() == 1000 and tback.table.get(0, 997) == 7 and tback.table.get(0, 3) == 3, "and read back exactly")
    var big = Table.empty()
    var x = big.add_column("x", INT)
    for i in range(1000):
        big.append(x, Int64(i) * 0x9E3779B97F4A7C15)
    write_parquet(path, big, "", ZSTD)
    var back = read_parquet(path)
    expect(back.table.rows() == 1000 and back.table.get(0, 999) == Int64(999) * 0x9E3779B97F4A7C15, "and read back exactly")

    # a streamed query (F-140) over it: the same numbers, on the CPU
    var streamed = Query(scan_parquet(path)).agg(count(), min_of(col("x")), max_of(col("x"))).compute(AUTO)
    var whole = Query(back.take_table().share()).agg(count(), min_of(col("x")), max_of(col("x"))).compute(AUTO)
    expect(streamed.device == CPU and streamed.counts[0] == 1000 and streamed.values(1)[0] == whole.values(1)[0] and
           streamed.values(2)[0] == whole.values(2)[0], "a streamed query gives the whole-file answer")

    # F-146: no GPU on this build: a streamed query on GPU says so, on AUTO it is the CPU with nothing to explain
    var gmsg = String("")
    try:
        _ = Query(scan_parquet(path)).agg(count()).compute(GPU)
    except e:
        gmsg = String(e)
    expect(gmsg == "device GPU: no GPU on this build", "a streamed query on GPU raises: `" + gmsg + "`")
    var gauto = Query(scan_parquet(path)).auto_gpu_ops(0).agg(count()).compute(AUTO)
    expect(gauto.device == CPU and gauto.fallback_reason == "" and gauto.counts[0] == 1000, "AUTO streams on the CPU")

    # an edited streamed file (F-147): the overlay, the sort and the save work with no pool and no codecs
    var sframe = Frame(scan_parquet(path))
    sframe.set(0, 5, 42)
    sframe.remove(7)
    var edited = Query(sframe).agg(count(), max_of(col("x"))).compute(AUTO)
    var want_max = Int64.MIN
    for i in range(1000):
        if i != 7:
            want_max = max(want_max, Int64(42) if i == 5 else Int64(i) * 0x9E3779B97F4A7C15)
    expect(edited.counts[0] == 999 and edited.values(1)[0] == Int(want_max), "a scan frame's query sees its edits")
    expect(sframe.get(0, 5) == 42 and sframe.permutation(0).__len__() == 999, "get and sort see them")
    sframe.save(path + ".edited", "", ZSTD)
    var resaved = read_parquet(path + ".edited")
    expect(resaved.table.rows() == 999 and resaved.table.get(0, 5) == 42, "and the saved file holds them")

    # F-175: no thread pool to run a job on: by default every job runs inside its start call and the handle comes back finished (with
    # -D MXFRAME_JOB_THREADS it gets a thread of its own whose kernels still run in a loop); either way the plain call's answer, its
    # progress at the total, and nothing left to cancel
    var jq = compute_job(Query(shared).group_by(col("k"), groups=3).agg(sum(col("v")), count()), AUTO)
    comptime if not THREADS:
        expect(jq.poll() == JOB_DONE, "a job on a serial build is done when it is started")
    expect(jq.wait() == JOB_DONE, "the job is done")
    var jr = jq.result()
    expect(jr.counts[0] == auto.counts[0] and jr.sums[0] == auto.sums[0], "compute_job gives the plain answer")
    var jp = jq.progress()
    expect(jp.done == jp.total and jp.total == n, "and its progress is at the total: " + String(jp.done) + "/" + String(jp.total))
    jq.cancel()
    expect(jq.poll() == JOB_DONE, "cancel on a finished job changes nothing")
    var js = sql_job(cat, "SELECT k, count(*) FROM t GROUP BY k ORDER BY k")
    expect(js.wait() == JOB_DONE and js.result().table.get(1, 0) == Int64((n + 2) // 3), "sql_job")
    var jc = read_csv_job(csv)
    expect(jc.wait() == JOB_DONE and jc.result().rows == 50_000, "read_csv_job")
    var jbad = read_csv_job("mxframe-switches-no-such.csv")
    expect(jbad.wait() == JOB_FAILED, "a failing job is failed, not raised at the start")
    var jw = write_parquet_job(shared, path + ".job", "", ZSTD)
    expect(jw.wait() == JOB_DONE and jw.result() == n, "write_parquet_job")
    var jrd = read_parquet_job(path + ".job")
    expect(jrd.wait() == JOB_DONE and jrd.result().table.rows() == n, "read_parquet_job")
    var jperm = permutation_job(sframe, 0)
    expect(jperm.wait() == JOB_DONE and jperm.result().__len__() == sframe.permutation(0).__len__(), "permutation_job")
    var jv = save_job(sframe, path + ".job-edited", "", ZSTD)
    var vstate = jv.wait()
    var vmsg = String()
    try:
        var wrote = jv.result()
        vmsg = "wrote " + String(wrote)
    except e:
        vmsg = String(e)
    expect(vstate == JOB_DONE and vmsg == "wrote 999", "save_job: state " + String(vstate) + ", " + vmsg)
    comptime if THREADS:                                   # a job thread with serial kernels, cancelled mid-run
        var slow = compute_job(Query(shared).group_by(col("k"), groups=3).agg(sum(col("v")), count_distinct(col("v")), median(col("v"))), AUTO)
        slow.cancel()
        expect(slow.wait() == JOB_CANCELLED or slow.poll() == JOB_DONE, "a job thread on a serial build can be cancelled")

    # ... and reads the files the C libraries wrote
    expect(len(args) >= 3, "run with a zstd file and a snappy file")
    for k in range(1, 3):
        var theirs = read_parquet(String(args[k]))
        expect(theirs.table.rows() == 500, "a file from the C codecs has its 500 rows")
        for i in range(500):
            expect(theirs.table.get(0, i) == Int64(i * 7 - 3000), "qty of row " + String(i) + " in file " + String(k))
        var words = scan_parquet(String(args[k]))
        expect(words.word_count(words.index("region")) == 5, "and its text column scans: " + String(words.word_count(words.index("region"))) + " words")
    print("ok")
