# F-167 review: `Frame.permutation` (and `permutation_job`) honour a sort budget. A permutation too big for the budget is sorted in runs of
# (key, row) pairs on disk and merged; it is the in-memory permutation, element for element, for every key kind (INT with NULLs, TEXT of
# mixed case, FLOAT with NaN / -0.0 / inf, DECIMAL, DATE, BOOL, a formula), both directions, a Table, a Frame with edits, a Frame over a
# Parquet scan and over a CSV scan, with runs of 1,000 rows (many), 3,000 (two) and 100,000 (one run holding every row).
from std.os import listdir, mkdir
from std.math import nan, inf
from omxcore import (Table, SharedTable, Frame, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, write_parquet, scan_parquet, scan_csv, CsvOptions, ZSTD,
                     permutation_job, JOB_DONE, JOB_CANCELLED, JobCtl)

comptime N = 5000
comptime DIR = "perm-spill"


def fail(msg: String) raises:
    raise Error(msg)


def make_table() raises -> SharedTable:
    var t = Table.empty()
    var id = t.add_column("id", INT)
    var k1 = t.add_column("k1", INT)
    var txt = t.add_column("txt", TEXT)
    var f = t.add_column("f", FLOAT)
    var d = t.add_column("d", DECIMAL, scale=2)
    var dt = t.add_column("dt", DATE)
    var b = t.add_column("b", BOOL)
    var w: List[String] = ["apple", "Apple", "APPLE", "banana", "Banana", "cherry", "", "date", "Zebra", "zebra", "apple pie"]
    var sp: List[Float64] = [nan[DType.float64](), -0.0, 0.0, inf[DType.float64](), -inf[DType.float64](), 1.5, -2.25]
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
            t.append_float(f, sp[(i // 3) % len(sp)])
        else:
            t.append_float(f, Float64((i * 31) % 50) / 4.0 - 6.0)
        t.append(d, Int64((i * 37) % 2000 - 1000))
        t.append(dt, Int64(18000 + (i * 3) % 40))
        t.append(b, Int64(1 if i % 5 < 2 else 0))
    return t^.share()


def no_files_left() raises:
    var left = listdir(DIR)
    if len(left) != 0:
        fail("the sort left a temporary file in " + DIR + ": " + left[0])


def same(a: List[Int], b: List[Int], what: String) raises:
    if len(a) != len(b):
        fail(what + ": " + String(len(a)) + " rows, wanted " + String(len(b)))
    for i in range(len(a)):
        if a[i] != b[i]:
            fail(what + ": row " + String(i) + " is " + String(a[i]) + ", wanted " + String(b[i]))


def check(f: Frame, what: String) raises:
    for c in range(f.columns()):
        for desc in [False, True]:
            var want = f.permutation(c, desc)
            var runs: List[Int] = [1000, 3000, 100000]
            for r in runs:
                var got = f.permutation(c, desc, JobCtl(), 0, DIR, r)
                same(got, want, what + ", column " + f.name(c) + (" desc" if desc else "") + ", runs of " + String(r))
                no_files_left()
            # a budget small enough that the sort cannot fit: spills by itself, no override
            same(f.permutation(c, desc, JobCtl(), 20000, DIR), want, what + ", column " + f.name(c) + ", a 20 KB budget")
            no_files_left()


def main() raises:
    mkdir(DIR)
    var t = Frame(make_table())
    _ = t.formula("twice = k1 + k1")
    check(t, "table")
    var e = Frame(make_table())
    e.set(1, 5, Int64(3))
    e.set_null(1, 6)
    e.set_text(2, 7, "Mango")
    e.set_float(3, 8, 99.5)
    e.set_null(3, 9)
    e.remove(10)
    e.remove(3000)
    _ = e.append(["5000", "0", "banana", "1.5", "0.00", "1970-01-01", "true"])
    _ = e.append(["5001", "", "Zed", "nan", "3.00", "1970-01-02", "false"])
    check(e, "frame with edits")
    write_parquet("perm.parquet", make_table(), "", ZSTD, 704)
    var s = Frame(scan_parquet("perm.parquet"))
    s.set(1, 2, Int64(-6))
    s.remove(100)
    check(s, "frame over a parquet scan")
    var text = String("id,k1,txt,d\n")
    for i in range(N):
        text += String(i) + "," + ("" if i % 11 == 0 else String((i * 7) % 13 - 6)) + "," + ("" if i % 17 == 0 else "w" + String((i * 5) % 9)) + "," + String((i * 37) % 2000 - 1000) + ".25\n"
    with open("perm.csv", "w") as fh:
        fh.write(text)
    var c = Frame(scan_csv("perm.csv", CsvOptions(group_rows=800)))
    check(c, "frame over a csv scan")
    print("ok   permutation spill: table, edits, parquet scan, csv scan")
    # the job: a cancel mid-run leaves no file
    var cancelled = 0
    for round in range(10):
        var job = permutation_job(Frame(make_table()), 2, False, budget=20000, spill_dir=DIR)
        if round % 2 == 1:
            job.cancel()
        var state = job.wait()
        if state != JOB_DONE and state != JOB_CANCELLED:
            fail("a permutation job ends done or cancelled")
        if state == JOB_DONE:
            same(job.result(), Frame(make_table()).permutation(2), "a permutation job that spilled")
        else:
            cancelled += 1
        no_files_left()
    if cancelled == 0:
        fail("no cancel reached a permutation job")
    print("ok   permutation job spills and cancels clean")
