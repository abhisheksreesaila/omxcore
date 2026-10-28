# F-167: a sort bigger than its memory budget spills sorted runs to disk and merges them. Every query runs twice, in memory and with the
# sort forced to spill in runs of 1,024 rows (many runs), 3,072 (two), and one run holding every row; the two results are the same table,
# cell by cell, for a Table, a Frame with edits, a Parquet scan and a CSV scan. Ties keep the source's order (id is in the select list), and
# NULLS FIRST / LAST, DESC, TEXT case order, NaN and -0.0 are in the data. (tests/test_sort_spill.py compares the answers with DuckDB.)
from std.os import listdir, mkdir
from std.math import nan, inf
from omxcore import (Table, SharedTable, Frame, Catalog, PlannerOptions, sql, sql_to_parquet, SqlResult, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT,
                     write_parquet, scan_parquet, read_parquet, scan_csv, CsvOptions, ZSTD, sql_job, JOB_DONE, JOB_CANCELLED)

comptime N = 6000
comptime DIR = "spill-dir"


def fail(msg: String) raises:
    raise Error(msg)


def cell(t: Table, c: Int, r: Int) raises -> String:
    if t.is_null(c, r):
        return "NULL"
    var k = t.kind(c).value
    var v = t.get(c, r)
    if k == TEXT.value:
        return "'" + t.text_of(c, Int(v)) + "'"
    if k == BOOL.value:
        return "true" if v != 0 else "false"
    if k == FLOAT.value:
        return String(Float64(from_bits=UInt64(v)))
    return String(v)


def rows_of(t: Table) raises -> String:
    var out = String()
    for r in range(t.rows()):
        for c in range(t.columns()):
            out += (" " if c > 0 else "") + cell(t, c, r)
        out += "\n"
    return out^


def words() -> List[String]:
    return ["apple", "Apple", "APPLE", "banana", "Banana", "cherry", "Cherry", "", "date", "Date", "elder", "fig", "Fig", "grape", "apple pie", "Zebra", "zebra"]


def make_table() raises -> SharedTable:
    var t = Table.empty()
    var id = t.add_column("id", INT)
    var k1 = t.add_column("k1", INT)
    var txt = t.add_column("txt", TEXT)
    var f = t.add_column("f", FLOAT)
    var d = t.add_column("d", DECIMAL, scale=2)
    var dt = t.add_column("dt", DATE)
    var b = t.add_column("b", BOOL)
    var w = words()
    var specials: List[Float64] = [nan[DType.float64](), -0.0, 0.0, inf[DType.float64](), -inf[DType.float64](), 1.5, -2.25]
    for i in range(N):
        t.append(id, Int64(i))
        if i % 11 == 0:
            t.append_null(k1)
        else:
            t.append(k1, Int64((i * 7) % 13 - 6))
        if i % 17 == 0:
            t.append_null(txt)
        else:
            t.append_text(txt, w[(i * 5) % len(w)])
        if i % 19 == 0:
            t.append_null(f)
        elif i % 3 == 0:
            t.append_float(f, specials[(i // 3) % len(specials)])
        else:
            t.append_float(f, Float64((i * 31) % 50) / 4.0 - 6.0)
        t.append(d, Int64((i * 37) % 2000 - 1000))
        t.append(dt, Int64(18000 + (i * 3) % 40))
        t.append(b, Int64(1 if i % 5 < 2 else 0))
    return t^.share()


def queries() -> List[String]:
    return [
        "SELECT id, k1 FROM t ORDER BY k1",
        "SELECT id, k1, txt, f FROM t ORDER BY k1 DESC NULLS FIRST, txt, f DESC",
        "SELECT id, txt FROM t ORDER BY txt DESC, id DESC",
        "SELECT id, f FROM t ORDER BY f NULLS LAST",
        "SELECT id, f FROM t ORDER BY f DESC NULLS FIRST",
        "SELECT id, b, dt, d FROM t ORDER BY b, dt DESC, d",
        "SELECT id, k1 FROM t ORDER BY k1 NULLS FIRST LIMIT 1500 OFFSET 700",
        "SELECT id, k1 + 1 AS z, txt FROM t WHERE id % 3 <> 0 ORDER BY z DESC, txt",
        "SELECT id FROM t ORDER BY txt, f DESC, id",
        "SELECT txt, k1 FROM t ORDER BY txt, k1",
        "SELECT id FROM t WHERE id < 0 ORDER BY id",
    ]


def spilled(r: SqlResult, runs: Int) -> Bool:
    for n in r.notes:
        if " runs spilled to disk" in n and n.startswith("order: sorted") and (runs < 0 or (" in " + String(runs) + " runs") in n):
            return True
    return False


def spilled_one(r: SqlResult) -> Bool:
    for n in r.notes:
        if " in 1 run spilled to disk" in n:
            return True
    return False


def no_files_left() raises:
    var left = listdir(DIR)
    if len(left) != 0:
        fail("the sort left " + String(len(left)) + " temporary file(s) in " + DIR + ": " + left[0])


def options(run_rows: Int) -> PlannerOptions:
    var o = PlannerOptions()
    o.sort_run_rows = run_rows
    o.spill_dir = DIR
    return o^


def check_source(mut cat: Catalog, what: String, expect_rows: Int = -1) raises:
    """Every query, in memory and spilled in many / two / one run(s): the same table."""
    for q in queries():
        var want = sql(cat, q)
        for note in want.notes:
            if "spilled to disk" in note:
                fail(what + ": the plain sort spilled: " + q)
        var text = rows_of(want.table)
        if expect_rows >= 0 and q.startswith("SELECT id, k1 FROM t ORDER BY k1") and want.table.rows() != expect_rows and "LIMIT" not in q:
            fail(what + ": " + String(want.table.rows()) + " rows, wanted " + String(expect_rows))
        var plan: List[Int] = [1024, 3072, 100000]
        for run_rows in plan:
            var got = sql(cat, q, options(run_rows))
            var gt = rows_of(got.table)
            if gt != text:
                fail(what + ", runs of " + String(run_rows) + " rows: `" + q + "`\n--- spilled ---\n" + String(gt[byte=:600]) + "\n--- in memory ---\n" + String(text[byte=:600]))
            if want.table.rows() > 0:
                if run_rows == 100000 and not spilled_one(got):
                    fail(what + ": one run wanted: " + q)
                if run_rows == 1024 and not spilled(got, -1) and "id < 0" not in q:
                    fail(what + ": many runs wanted: " + q)
            no_files_left()


def test_table() raises:
    var cat = Catalog()
    cat.add_table("t", make_table())
    check_source(cat, "table")
    # the runs: 6000 rows in runs of 1024 are 6 runs, in runs of 3072 are 2 runs
    var r = sql(cat, "SELECT id, k1 FROM t ORDER BY k1", options(1024))
    if not spilled(r, 6):
        fail("6 runs of 1,024 rows wanted: " + String(r.notes[len(r.notes) - 1]))
    var r2 = sql(cat, "SELECT id, k1 FROM t ORDER BY k1", options(3072))
    if not spilled(r2, 2):
        fail("2 runs wanted")


def test_frame_with_edits() raises:
    var f = Frame(make_table())
    f.set(1, 5, Int64(3))
    f.set_null(1, 6)
    f.set_text(2, 7, "Mango")
    f.set_float(3, 8, 99.5)
    f.set_null(3, 9)
    f.set(1, 5999, Int64(-6))
    f.set(1, 4000, Int64(-6))
    f.remove(10)
    f.remove(3000)
    _ = f.append(["6000", "0", "banana", "1.5", "0.00", "1970-01-01", "true"])
    _ = f.append(["6001", "", "Zed", "nan", "3.00", "1970-01-02", "false"])
    var cat = Catalog()
    cat.add_frame("t", f)
    check_source(cat, "frame with edits")


def test_parquet_scan() raises:
    write_parquet("spill-scan.parquet", make_table(), "", ZSTD, 704)
    var cat = Catalog()
    cat.add_scan("t", scan_parquet("spill-scan.parquet"))
    check_source(cat, "parquet scan")
    var f = Frame(scan_parquet("spill-scan.parquet"))
    f.set(1, 2, Int64(-6))
    f.remove(100)
    var cat2 = Catalog()
    cat2.add_frame("t", f)
    check_source(cat2, "frame over a parquet scan")


def test_csv_scan() raises:
    var text = String("id,k1,txt,d\n")
    var w = words()
    for i in range(N):
        text += String(i) + "," + ("" if i % 11 == 0 else String((i * 7) % 13 - 6)) + "," + ("" if i % 17 == 0 else w[(i * 5) % len(w)]) + "," + String((i * 37) % 2000 - 1000) + ".25\n"
    with open("spill-scan.csv", "w") as fh:
        fh.write(text)
    var cat = Catalog()
    cat.add_scan("t", scan_csv("spill-scan.csv", CsvOptions(group_rows=800)))
    for q in ["SELECT id, k1 FROM t ORDER BY k1", "SELECT id, txt, d FROM t ORDER BY txt DESC NULLS FIRST, d, id", "SELECT id FROM t ORDER BY d DESC, k1 LIMIT 2500 OFFSET 17"]:
        var want = rows_of(sql(cat, q).table)
        var plan: List[Int] = [1024, 3072, 100000]
        for run_rows in plan:
            var got = sql(cat, q, options(run_rows))
            if rows_of(got.table) != want:
                fail("csv scan, runs of " + String(run_rows) + ": " + q)
            if not spilled(got, -1) and not spilled_one(got):
                fail("csv scan: the sort did not spill: " + q)
            no_files_left()


def test_text_order_across_runs() raises:
    """TEXT codes are in first-seen order: a word met in the last run is the smallest, and the merge must still put it first."""
    var t = Table.empty()
    var id = t.add_column("id", INT)
    var s = t.add_column("s", TEXT)
    var last = String("aaa")
    for i in range(4096):
        t.append(id, Int64(i))
        t.append_text(s, String("w") + String(i % 40))
    for i in range(1024):
        t.append(id, Int64(4096 + i))
        t.append_text(s, last if i % 2 == 0 else String("AAA"))
    var cat = Catalog()
    cat.add_table("t", t^.share())
    var want = rows_of(sql(cat, "SELECT id, s FROM t ORDER BY s, id DESC").table)
    var got = sql(cat, "SELECT id, s FROM t ORDER BY s, id DESC", options(1024))
    if rows_of(got.table) != want:
        fail("a word first seen in the last run sorted wrong")
    if not got.table.text_of(1, Int(got.table.get(1, 0))).startswith("aaa") and not got.table.text_of(1, Int(got.table.get(1, 0))).startswith("AAA"):
        fail("the smallest word is not first: " + got.table.text_of(1, Int(got.table.get(1, 0))))


def test_top_k_does_not_spill() raises:
    """A small LIMIT keeps the top-k heaps: no runs, even when the sort is told to spill."""
    var cat = Catalog()
    cat.add_table("t", make_table())
    var r = sql(cat, "SELECT id, k1 FROM t ORDER BY k1, id LIMIT 10", options(1024))
    for n in r.notes:
        if "spilled" in n:
            fail("a top-k spilled: " + n)
    no_files_left()
    if r.table.rows() != 10:
        fail("top-10 wanted")


def test_save_sorted_file() raises:
    """ORDER BY into a Parquet file: the rows are merged straight into the file; the file holds the sorted table, in row groups of the size asked."""
    var cat = Catalog()
    cat.add_table("t", make_table())
    for q in queries():
        var want = sql(cat, q)
        var res = sql_to_parquet(cat, q, "sorted-out.parquet", options(1024), "", ZSTD, 1000)
        if res.rows_written != want.table.rows():
            fail("rows written " + String(res.rows_written) + ", wanted " + String(want.table.rows()) + ": " + q)
        var back = read_parquet("sorted-out.parquet")
        if rows_of(back.table) != rows_of(want.table):
            fail("the sorted file differs from the sort in memory: " + q + "\n" + String(rows_of(back.table)[byte=:300]))
        no_files_left()
    var res2 = sql_to_parquet(cat, "SELECT id, k1 FROM t ORDER BY k1", "sorted-out.parquet", options(1024), "", ZSTD, 1000)
    var streamed = False
    for n in res2.notes:
        if n.startswith("output: 6,000 rows merged straight into"):
            streamed = True
    if not streamed:
        fail("the rows were not streamed into the file")
    if scan_parquet("sorted-out.parquet").row_groups() != 6:
        fail("6 row groups of 1,000 rows wanted, got " + String(scan_parquet("sorted-out.parquet").row_groups()))
    # a sort that fits the budget is written from memory
    var res3 = sql_to_parquet(cat, "SELECT id, k1 FROM t ORDER BY k1", "sorted-out2.parquet")
    if res3.rows_written != N or read_parquet("sorted-out2.parquet").table.rows() != N:
        fail("an in-memory sort written to a file")


def test_error_removes_the_runs() raises:
    """An error after the runs are written (the output folder is a file) leaves no temporary file and no output."""
    var cat = Catalog()
    cat.add_table("t", make_table())
    var failed = False
    try:
        _ = sql_to_parquet(cat, "SELECT id, k1 FROM t ORDER BY k1", "sorted-out.parquet/out.parquet", options(1024))
    except e:
        failed = True
    if not failed:
        fail("a save into a folder that is a file should fail")
    no_files_left()
    var bad = options(1024)
    bad.spill_dir = "sorted-out.parquet/runs"
    var failed2 = False
    try:
        _ = sql(cat, "SELECT id, k1 FROM t ORDER BY k1", bad)
    except e:
        failed2 = "no such folder" in String(e)
    if not failed2:
        fail("a spill folder that is a file should give a readable error")


def test_cancel_removes_the_runs() raises:
    var cat = Catalog()
    cat.add_table("t", make_table())
    var cancelled = 0
    for round in range(12):
        var job = sql_job(cat, "SELECT id, k1, txt FROM t ORDER BY txt, k1, id", options(1024))
        if round % 2 == 1:
            job.cancel()
        var state = job.wait()
        if state != JOB_DONE and state != JOB_CANCELLED:
            fail("a sort job ends done or cancelled")
        if state == JOB_CANCELLED:
            cancelled += 1
        no_files_left()
    if cancelled == 0:
        fail("no cancel reached a sort")


def main() raises:
    mkdir(DIR)
    test_table()
    print("ok   table")
    test_frame_with_edits()
    print("ok   frame with edits")
    test_parquet_scan()
    print("ok   parquet scan")
    test_csv_scan()
    print("ok   csv scan")
    test_text_order_across_runs()
    print("ok   text order across runs")
    test_top_k_does_not_spill()
    print("ok   top-k")
    test_save_sorted_file()
    print("ok   sorted file")
    test_error_removes_the_runs()
    print("ok   error")
    test_cancel_removes_the_runs()
    print("ok   cancel")
