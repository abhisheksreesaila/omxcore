# F-138b: a small program over the whole package, compiled as an object for arm64-apple-macosx14.0 by
# scripts/check_mac_link.sh and linked the way the omxplorer Mac build links. It is never run: it exists so the
# linker sees every path an app uses (CSV import, queries on the CPU / AUTO / GPU, a frame's edits, Parquet save
# and open). `mojo_entry` is the entry point the app's Mac build links with (`-e _mojo_entry`).
from std.ffi import c_int
from omxcore import (Table, SharedTable, Frame, INT, DECIMAL, TEXT, Query, col, sum, count, CPU, GPU, AUTO,
                     DeviceTable, gpu_available, read_csv, scan_csv, CsvOptions, write_parquet, read_parquet, scan_parquet, UNCOMPRESSED, ZSTD, SNAPPY,
                     count_distinct, median, percentile, top_k, stddev, var_pop, Catalog, PlannerOptions, GROUP_HASHED, sql, range_aggregate, read_block,
                     compute_job, sql_job, read_csv_job, scan_csv_job, read_parquet_job, save_job, write_parquet_job, permutation_job, JOB_DONE, sql_to_parquet)


def run() raises:
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var v = t.add_column("v", DECIMAL, scale=2)
    for i in range(10):
        t.append(k, Int64(i % 3))
        t.append(v, Int64(i * 100))
    var shared = t^.share()
    var f = Frame(shared)
    f.set(1, 2, Int64(7))
    var q = Query(f).group_by(col("k"), 3).agg(sum(col("v")), count())
    var r = q.compute(AUTO)
    print(r.counts[0], r.device, gpu_available())
    var h = Query(f).group_by(col("k"), 3).agg(count_distinct(col("v")), median(col("v")), percentile(col("v"), 0.9), top_k(col("v"), 2)).compute(AUTO)
    print(h.values(0)[0], h.values(1)[0], h.top(3, 0).__len__())
    var sp = Query(f).group_by(col("k"), 3).agg(stddev(col("v")), var_pop(col("v"))).compute(AUTO)      # F-177
    var qbins = Query(f).quantile_bins(col("v"), 4)
    print(sp.floats(0)[0], sp.present(1, 0), qbins.bins(), Query(f).group_by(qbins.key(col("v")), qbins.bins()).agg(count()).compute(AUTO).counts[0])
    try:
        var d = DeviceTable(shared)                  # raises "no GPU on this build" off Linux
        _ = Query(shared).on_device(d).agg(count()).compute(GPU)
    except e:
        print(e)
    write_parquet("probe.parquet", shared, "{}", UNCOMPRESSED)
    write_parquet("probe-z.parquet", shared, "{}", ZSTD)         # F-180: the Mojo zstd and snappy, no C library
    write_parquet("probe-s.parquet", shared, "{}", SNAPPY)
    print(read_parquet("probe-z.parquet").table.rows(), read_parquet("probe-s.parquet").table.rows())
    var back = read_parquet("probe.parquet")
    print(back.table.rows())
    print(Query(scan_parquet("probe.parquet")).group_by(col("k"), 3).agg(sum(col("v")), count()).compute(AUTO).counts[0])
    var sf = Frame(scan_parquet("probe.parquet"))          # F-147: edits over a streamed file, queried, sorted, saved
    sf.set(1, 2, Int64(9))
    sf.remove(0)
    print(Query(sf).group_by(col("k"), 3).agg(sum(col("v")), count()).compute(AUTO).counts[0], sf.permutation(1).__len__(), sf.get(1, 2))
    sf.save("probe-edited.parquet", "{}", ZSTD)
    var c = read_csv("probe.csv")
    print(c.table.rows())
    # F-150: SQL over a table, a frame and a scan, dense and hashed groups, the two-pass holistic aggregates
    var cat = Catalog()
    cat.add_table("t", shared)
    cat.add_frame("f", f)
    cat.add_scan("s", scan_parquet("probe.parquet"))
    var hashed = PlannerOptions()
    hashed.group = GROUP_HASHED
    for name in ["t", "f", "s"]:
        var a = sql(cat, "SELECT k, count(*), sum(v), avg(v) FROM " + name + " WHERE v > 0 GROUP BY k ORDER BY 2 DESC LIMIT 2")
        var b = sql(cat, "SELECT k, count(DISTINCT v), median(v), min(v) FROM " + name + " GROUP BY k HAVING count(*) > 1", hashed)
        var sd = sql(cat, "SELECT quantile_bin(v, 3) AS b, stddev(v), var_pop(v) FROM " + name + " GROUP BY b")                      # F-177
        var iso = sql(cat, "SELECT count(*) FROM " + name + " WHERE week(DATE '2021-01-03') = 53 AND isoyear(DATE '2021-01-03') = 2020")
        print(sd.table.rows(), iso.table.rows())
        print(a.table.rows(), b.table.rows(), a.notes[0], sql(cat, "EXPLAIN SELECT count(*) FROM " + name).explain)
        # F-152: row queries: the materialize sink, a top-k, SELECT DISTINCT
        var r1 = sql(cat, "SELECT k, v, v + 1 FROM " + name + " WHERE v > 0 ORDER BY v DESC LIMIT 3")
        var r2 = sql(cat, "SELECT DISTINCT k FROM " + name)
        var r3 = sql(cat, "SELECT * FROM " + name + " LIMIT 5")
        print(r1.table.rows(), r2.table.rows(), r3.table.rows())
        # F-167: a sort that spills runs to disk and merges them (into a table, and straight into a file)
        var spillo = PlannerOptions()
        spillo.sort_run_rows = 4
        spillo.spill_dir = "."
        var r4 = sql(cat, "SELECT k, v FROM " + name + " ORDER BY v DESC, k", spillo)
        var r5 = sql_to_parquet(cat, "SELECT k, v FROM " + name + " ORDER BY k, v", "probe-sorted.parquet", spillo)
        print(r4.table.rows(), r5.rows_written)
        # F-153: an inner join (fused, then materialized) of each source with the table, and the builder's join
        var j1 = sql(cat, "SELECT a.k, count(*), sum(b.v) FROM " + name + " a JOIN t b ON a.v = b.v GROUP BY a.k")
        var nofuse = PlannerOptions()
        nofuse.fuse = False
        var j2 = sql(cat, "SELECT a.k, b.v FROM " + name + " a JOIN t b ON a.v = b.v WHERE a.k = 1 ORDER BY 2 LIMIT 3", nofuse)
        print(j1.table.rows(), j2.table.rows())
    print(Query(shared).join(shared, ["v"], ["v"]).agg(count()).compute(AUTO).counts[0])
    # F-176: range aggregates and block reads over a table, a frame with edits and a frame over a streamed file
    var rcols: List[Int] = [0, 1]
    var rorder = sf.permutation(1)
    var ra = range_aggregate(shared, rcols, 0, 10)
    var rb = range_aggregate(f, rcols, 0, 10)
    var rc = range_aggregate(sf, rorder, rcols, 0, len(rorder))
    var bvals = List[Int64](length=20, fill=0)
    var bnulls = List[UInt8](length=20, fill=0)
    read_block(shared, rcols, 0, 10, bvals, bnulls)
    read_block(f, rcols, 0, 10, bvals, bnulls)
    read_block(sf, rorder, rcols, 0, len(rorder), bvals, bnulls)
    print(ra.columns[1].sum, rb.columns[1].sum, rc.columns[1].sum, bvals[3])
    # F-175: background jobs. The serial build runs each job in the call and hands back a finished handle (no thread, no runtime symbol).
    var j1 = compute_job(Query(f).group_by(col("k"), 3).agg(sum(col("v")), count()), AUTO)
    var j2 = sql_job(cat, "SELECT k, count(*), sum(v) FROM t GROUP BY k ORDER BY k")
    var j3 = read_csv_job("probe.csv")
    var j4 = read_parquet_job("probe.parquet")
    var j5 = save_job(sf, "probe-job.parquet", "{}", ZSTD)
    var j6 = write_parquet_job(shared, "probe-job2.parquet", "{}", UNCOMPRESSED)
    var j7 = permutation_job(sf, 0)
    print(j7.wait(), j7.result().__len__())
    print(j1.wait() == JOB_DONE, j2.wait(), j3.wait(), j4.wait(), j5.wait(), j6.wait())
    print(j1.result().counts[0], j2.result().table.rows(), j3.result().rows, j4.result().table.rows(), j5.result(), j6.result(), j1.progress().stage)
    # F-178: a CSV streamed a row group at a time: opened (kinds, row groups, the failed-row report), queried, edited and saved
    var cs = scan_csv("probe.csv", CsvOptions(group_rows=4, block_bytes=64))
    print(cs.rows(), cs.row_groups(), cs.csv_report().failed_count(), Query(cs).group_by(col("k"), 3).agg(sum(col("v")), count()).compute(AUTO).counts[0])
    cat.add_scan("c", cs)
    print(sql(cat, "SELECT k, count(*), sum(v) FROM c GROUP BY k ORDER BY k").table.rows())
    var j8 = scan_csv_job("probe.csv", CsvOptions(group_rows=4))
    print(j8.wait(), j8.result().rows())
    var cf = Frame(cs)
    cf.set(1, 0, Int64(5))
    cf.save("probe-csv.parquet", "{}", ZSTD)


@export("mojo_entry")
def entry(argc: c_int, argv: Int) abi("C") -> c_int:
    try:
        run()
        return 0
    except e:
        print(e)
        return 1
