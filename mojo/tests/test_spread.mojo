# F-177: stddev / stddev_pop / variance / var_pop through Query, over a table, a Frame with edits and a Parquet scan of several row
# groups. The oracle is exact: n * sum(x^2) - sum(x)^2 in 128-bit integers over the stored values, divided once at the end, so a
# column of 1e12 + small numbers (where a sum-of-squares formula in doubles loses every digit) is checked to 1e-12 relative.
from std.math import sqrt
from omxcore import (Table, SharedTable, Query, QueryResult, Frame, col, lit, sum, count, mean, stddev, stddev_pop, variance, var_pop, median,
                     INT, DECIMAL, TEXT, write_parquet, read_parquet, scan_parquet, ZSTD, CPU, GPU, AUTO, DeviceTable, gpu_available)


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


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


def near(got: Float64, want: Float64, rel: Float64) -> Bool:
    if got == want:
        return True
    var d = got - want
    if d < 0.0:
        d = -d
    var w = want if want >= 0.0 else -want
    return d <= rel * w


comptime N = 150_000
comptime BIG = 1_000_000_000_000


def make_table(n: Int, seed: UInt64) raises -> SharedTable:
    """carrier INT -1..7 (nullable); big INT = 1e12 + (-50..50) (nullable); rev DECIMAL(2) (nullable, negatives); day INT 0..400."""
    var t = Table.empty()
    var c = t.add_column("carrier", INT)
    var b = t.add_column("big", INT)
    var r = t.add_column("rev", DECIMAL, scale=2)
    var d = t.add_column("day", INT)
    var rng = Rng(seed)
    for i in range(n):
        if i % 31 == 7:
            t.append_null(c)
        else:
            t.append(c, Int64(rng.int(-1, 7)))
        if i % 13 == 1:
            t.append_null(b)
        else:
            t.append(b, Int64(BIG + rng.int(-50, 50)))
        if i % 11 == 3:
            t.append_null(r)
        else:
            t.append(r, Int64(rng.int(-90_000, 90_000)))
        t.append(d, Int64(rng.int(0, 400)))
    return t^.share()


struct Exact(Copyable, Movable):
    var n: Int
    var sum: Int128
    var sumsq: Int128

    def __init__(out self):
        self.n = 0
        self.sum = 0
        self.sumsq = 0

    def add(mut self, v: Int64):
        self.n += 1
        self.sum += Int128(v)
        self.sumsq += Int128(v) * Int128(v)

    def num(self) -> Float64:
        return Float64(Int128(self.n) * self.sumsq - self.sum * self.sum)

    def var_pop(self, scale: Int) -> Float64:
        return self.num() / (Float64(self.n) * Float64(self.n)) / pow10(2 * scale)

    def var_samp(self, scale: Int) -> Float64:
        return self.num() / (Float64(self.n) * Float64(self.n - 1)) / pow10(2 * scale)


def pow10(k: Int) -> Float64:
    var x = 1.0
    for _ in range(k):
        x *= 10.0
    return x


def gather(t: SharedTable, vname: String, groups: Int, keyed: Bool, filtered: Bool) -> List[Exact]:
    var out = List[Exact]()
    for _ in range(groups):
        out.append(Exact())
    var kc = t.index("carrier")
    var vc = t.index(vname)
    var dc = t.index("day")
    for row in range(t.rows()):
        if filtered and not (t.get(dc, row) >= 10 and t.get(dc, row) <= 300):
            continue
        var g = 0
        if keyed:
            if t.is_null(kc, row):
                continue
            g = Int(t.get(kc, row))
            if g < 0 or g >= groups:
                continue
        if t.is_null(vc, row):
            continue
        out[g].add(t.get(vc, row))
    return out^


def check_spread(r: QueryResult, first: Int, ex: List[Exact], scale: Int, rel: Float64, label: String) raises:
    """Aggregates first .. first + 3 are stddev, stddev_pop, variance, var_pop of the same values."""
    for g in range(len(ex)):
        var e = ex[g].copy()
        var tag = label + " group " + String(g)
        expect(r.present(first + 1, g) == (e.n >= 1), tag + ": stddev_pop present")
        expect(r.present(first + 3, g) == (e.n >= 1), tag + ": var_pop present")
        expect(r.present(first, g) == (e.n >= 2), tag + ": stddev present")
        expect(r.present(first + 2, g) == (e.n >= 2), tag + ": variance present")
        if e.n >= 1:
            var vp = e.var_pop(scale)
            expect(near(r.floats(first + 3)[g], vp, rel), tag + ": var_pop " + String(r.floats(first + 3)[g]) + " != " + String(vp))
            expect(near(r.floats(first + 1)[g], sqrt(vp), rel), tag + ": stddev_pop " + String(r.floats(first + 1)[g]) + " != " + String(sqrt(vp)))
        if e.n >= 2:
            var vs = e.var_samp(scale)
            expect(near(r.floats(first + 2)[g], vs, rel), tag + ": variance " + String(r.floats(first + 2)[g]) + " != " + String(vs))
            expect(near(r.floats(first)[g], sqrt(vs), rel), tag + ": stddev " + String(r.floats(first)[g]) + " != " + String(sqrt(vs)))
        expect(r.is_float(first), tag + ": a float64 aggregate")


def run_grouped(var q: Query, groups: Int) raises -> QueryResult:
    return q^.group_by(col("carrier"), groups=groups).agg(stddev(col("big")), stddev_pop(col("big")), variance(col("big")), var_pop(col("big")),
                                                         stddev(col("rev")), stddev_pop(col("rev")), variance(col("rev")), var_pop(col("rev")),
                                                         count()).compute(CPU)


def run_global(var q: Query) raises -> QueryResult:
    return q^.agg(stddev(col("big")), stddev_pop(col("big")), variance(col("big")), var_pop(col("big")),
                  stddev(col("rev")), stddev_pop(col("rev")), variance(col("rev")), var_pop(col("rev"))).compute(CPU)


def test_over_a_table() raises:
    var t = make_table(N, 1)
    var r = run_grouped(Query(t), 10)                         # carriers 8 and 9 never occur: empty groups
    check_spread(r, 0, gather(t, "big", 10, True, False), 0, 1e-12, "table big")
    check_spread(r, 4, gather(t, "rev", 10, True, False), 2, 1e-12, "table rev")
    expect(not r.present(0, 8) and not r.present(1, 9), "an empty group has no value")
    var g = run_global(Query(t))
    check_spread(g, 0, gather(t, "big", 1, False, False), 0, 1e-12, "global big")
    check_spread(g, 4, gather(t, "rev", 1, False, False), 2, 1e-12, "global rev")
    var f = Query(t).where((col("day") >= 10) & (col("day") <= 300)).group_by(col("carrier"), groups=7).agg(
        stddev(col("big")), stddev_pop(col("big")), variance(col("big")), var_pop(col("big"))).compute(CPU)
    check_spread(f, 0, gather(t, "big", 7, True, True), 0, 1e-12, "filtered big")


def test_the_same_bits_for_any_number_of_threads() raises:
    var t = make_table(N, 2)
    var a = Query(t).threads(1).group_by(col("carrier"), groups=7).agg(stddev(col("big")), var_pop(col("rev"))).compute(CPU)
    var b = Query(t).threads(8).group_by(col("carrier"), groups=7).agg(stddev(col("big")), var_pop(col("rev"))).compute(CPU)
    for g in range(7):
        expect(a.values(0)[g] == b.values(0)[g] and a.values(1)[g] == b.values(1)[g], "bits equal for 1 and 8 threads, group " + String(g))
    var one = Query(t).threads(1).agg(stddev(col("big"))).compute(CPU)           # one group of 150,000 values: cut into pieces over the threads
    var many = Query(t).threads(8).agg(stddev(col("big"))).compute(CPU)
    expect(one.values(0)[0] == many.values(0)[0], "one big group: bits equal for 1 and 8 threads")


def test_small_groups_and_nulls() raises:
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var v = t.add_column("v", INT)
    # group 0: 1, 2 (variance 0.5); group 1: one value 7; group 2: only NULLs; group 3: nothing; group 4: -3, -3, -3 (zero spread)
    t.append(k, 0)
    t.append(v, 1)
    t.append(k, 0)
    t.append(v, 2)
    t.append(k, 1)
    t.append(v, 7)
    t.append(k, 2)
    t.append_null(v)
    t.append(k, 2)
    t.append_null(v)
    for _ in range(3):
        t.append(k, 4)
        t.append(v, -3)
    var s = t^.share()
    var r = Query(s).group_by(col("k"), groups=5).agg(stddev(col("v")), stddev_pop(col("v")), variance(col("v")), var_pop(col("v"))).compute(CPU)
    expect(r.floats(2)[0] == 0.5 and r.floats(3)[0] == 0.25 and r.floats(1)[0] == 0.5 and r.floats(0)[0] == sqrt(0.5), "two values")
    expect(not r.present(0, 1) and not r.present(2, 1) and r.present(1, 1) and r.present(3, 1) and r.floats(3)[1] == 0.0 and r.floats(1)[1] == 0.0, "one value")
    for i in range(4):
        expect(not r.present(i, 2) and not r.present(i, 3), "all NULL and empty groups have no value: aggregate " + String(i))
    expect(r.floats(0)[4] == 0.0 and r.floats(2)[4] == 0.0 and r.present(0, 4), "equal values: zero spread")
    var e = Table.empty()
    _ = e.add_column("v", INT)
    var none = Query(e^.share()).agg(stddev(col("v")), var_pop(col("v"))).compute(CPU)
    expect(not none.present(0, 0) and not none.present(1, 0), "no rows")


def test_cancellation_and_extremes() raises:
    var t = Table.empty()
    var v = t.add_column("v", INT)
    var f = t.add_column("f", INT)
    # 1e15 + 1, +2, +3: a naive (sum of squares - n * mean^2) formula in doubles answers 0 or garbage; the variance is exactly 1
    for x in range(1, 4):
        t.append(v, Int64(1_000_000_000_000_000 + x))
        t.append(f, Int64(x))
    var s = t^.share()
    var r = Query(s).agg(variance(col("v")), var_pop(col("v")), stddev(col("v")), variance(col("f"))).compute(CPU)
    expect(r.floats(0)[0] == 1.0, "cancellation: variance of 1e15 + 1, 2, 3 is 1, got " + String(r.floats(0)[0]))
    expect(near(r.floats(1)[0], 2.0 / 3.0, 1e-15), "var_pop 2/3")
    expect(r.floats(2)[0] == 1.0 and r.floats(3)[0] == 1.0, "stddev and a small column")
    # values near the ends of int64 read as doubles (rounded past 2^53, never an error): the spread is of the rounded values
    var w = Table.empty()
    var x = w.add_column("x", INT)
    w.append(x, Int64.MAX)
    w.append(x, Int64.MAX - 1)
    w.append(x, Int64.MIN + 1)
    var rw = Query(w^.share()).agg(var_pop(col("x")), stddev(col("x"))).compute(CPU)
    expect(rw.floats(0)[0] > 1.0e37 and rw.floats(1)[0] > 1.0e18, "extreme values do not overflow")


def test_decimal_scales() raises:
    var t = Table.empty()
    var a = t.add_column("a", DECIMAL, scale=1)
    var b = t.add_column("b", DECIMAL, scale=4)
    # a = 0.1, 0.2, 0.3 -> variance 0.01; b = 1.0000, 2.5000 -> variance 1.125
    t.append(a, 1)
    t.append(a, 2)
    t.append(a, 3)
    t.append(b, 10000)
    t.append(b, 25000)
    t.append_null(b)
    var s = t^.share()
    var r = Query(s).agg(variance(col("a")), variance(col("b")), stddev_pop(col("b"))).compute(CPU)
    expect(near(r.floats(0)[0], 0.01, 1e-14), "decimal(1): " + String(r.floats(0)[0]))
    expect(near(r.floats(1)[0], 1.125, 1e-14) and near(r.floats(2)[0], 0.75, 1e-14), "decimal(4)")


def test_over_a_frame_with_edits() raises:
    var t = make_table(60_000, 3)
    var f = Frame(t)
    var rng = Rng(9)
    for step in range(400):
        var row = rng.int(0, f.rows() - 1)
        while not f.alive(row):
            row = rng.int(0, f.rows() - 1)
        var what = rng.int(0, 6)
        if what <= 1:
            f.set(f.column("big"), row, Int64(BIG + rng.int(-5000, 5000)))
        elif what == 2:
            f.set(f.column("rev"), row, Int64(rng.int(-70_000, 70_000)))
        elif what == 3:
            f.set_null(f.column("big"), row)
        elif what == 4:
            _ = f.append(["3", String(BIG + 77), "12.50", "100"])
        elif what == 5:
            f.remove(row)
        else:
            f.set(f.column("carrier"), row, Int64(rng.int(0, 6)))
    var baked = f.bake().share()
    var r = run_grouped(Query(f), 10)
    check_spread(r, 0, gather(baked, "big", 10, True, False), 0, 1e-12, "frame big")
    check_spread(r, 4, gather(baked, "rev", 10, True, False), 2, 1e-12, "frame rev")
    var shared = run_grouped(Query(t), 10)
    var differs = False
    for g in range(7):
        differs = differs or shared.values(0)[g] != r.values(0)[g]
    expect(differs, "the edits change the answer")


def test_over_a_scan_of_several_row_groups() raises:
    var mem = make_table(N, 5)
    var path = String("build/mojo-tests/f177_scan.parquet")
    write_parquet(path, mem, "{}", ZSTD, 20_000)
    var scan = scan_parquet(path)
    expect(scan.row_groups() == 8, "the file has " + String(scan.row_groups()) + " row groups")
    var r = run_grouped(Query(scan), 10)
    check_spread(r, 0, gather(mem, "big", 10, True, False), 0, 1e-12, "scan big")
    check_spread(r, 4, gather(mem, "rev", 10, True, False), 2, 1e-12, "scan rev")
    var g = run_global(Query(scan))
    check_spread(g, 0, gather(mem, "big", 1, False, False), 0, 1e-12, "scan global big")


def test_errors_and_the_cpu() raises:
    var t = make_table(1000, 6)
    var msg = String()
    try:
        _ = Query(t).agg(stddev(col("big"))).compute(GPU)
    except e:
        msg = String(e)
    expect("CPU" in msg, "GPU: " + msg)
    var r = Query(t).agg(stddev(col("big"))).compute(AUTO)
    expect(r.device == CPU, "AUTO runs it on the CPU")
    var w = Table.empty()
    var c = w.add_column("w", TEXT)
    w.append_text(c, "a")
    msg = ""
    try:
        _ = Query(w^.share()).agg(stddev(col("w"))).compute(CPU)
    except e:
        msg = String(e)
    expect("need numbers" in msg, "text: " + msg)


def test_a_device_table_does_not_move_them_to_the_gpu() raises:
    """AUTO with a DeviceTable attached and enough rows uses the GPU for plain aggregates; stddev and variance run on the CPU and say why,
    next to a sum that would have gone to the GPU. compute(GPU) refuses. (Skipped without a GPU.)"""
    if not gpu_available():
        return
    var t = make_table(1_000_000, 8)
    var dt = DeviceTable(t)
    var r = Query(t).on_device(dt).group_by(col("carrier"), groups=10).agg(stddev(col("big")), var_pop(col("rev")), sum(col("day"))).compute(AUTO)
    expect(r.device == CPU and "stddev and variance run on the CPU" in r.fallback_reason, "AUTO ran on the CPU and said why: " + r.fallback_reason)
    expect(r.present(0, 3) and r.present(1, 3), "a value")
    var plain = Query(t).group_by(col("carrier"), groups=10).agg(stddev(col("big")), var_pop(col("rev")), sum(col("day"))).compute(CPU)
    for g in range(10):
        expect(r.values(0)[g] == plain.values(0)[g] and r.values(1)[g] == plain.values(1)[g] and r.values(2)[g] == plain.values(2)[g],
               "the same answers as the CPU run, group " + String(g))
    var msg = String("")
    try:
        _ = Query(t).on_device(dt).agg(variance(col("big"))).compute(GPU)
    except e:
        msg = String(e)
    expect("run on the CPU" in msg, "compute(GPU): " + msg)
    dt.release()


def main() raises:
    test_over_a_table()
    test_the_same_bits_for_any_number_of_threads()
    test_small_groups_and_nulls()
    test_cancellation_and_extremes()
    test_decimal_scales()
    test_over_a_frame_with_edits()
    test_over_a_scan_of_several_row_groups()
    test_errors_and_the_cpu()
    test_a_device_table_does_not_move_them_to_the_gpu()
    print("ok")
