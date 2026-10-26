# F-177: Query.quantile_bins / QuantileBins.key through the builder, over a table, a Frame with edits and a Parquet scan. The oracle is written
# here: sort the cells the plan keeps, edge k is the value at sorted place ceil(k * N / bins), a row's bin is the number of edges below its value.
from omxcore import (Table, SharedTable, Query, QuantileBins, Frame, col, lit, lit_float, sum, count, min_of, max_of, INT, DECIMAL, TEXT,
                     write_parquet, scan_parquet, ZSTD, CPU, FLOAT)


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


def msort(mut v: List[Int64]):
    var n = len(v)
    var tmp = List[Int64](length=n, fill=0)
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = min(lo + width, n)
            var hi = min(lo + 2 * width, n)
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                if v[j] < v[i]:
                    tmp[k] = v[j]
                    j += 1
                else:
                    tmp[k] = v[i]
                    i += 1
                k += 1
            while i < mid:
                tmp[k] = v[i]
                i += 1
                k += 1
            while j < hi:
                tmp[k] = v[j]
                j += 1
                k += 1
            lo += 2 * width
        swap(v, tmp)
        width *= 2


comptime N = 100_000


def make_table(n: Int, seed: UInt64) raises -> SharedTable:
    """x INT (-200 .. 200, ties, 7% null), r DECIMAL(2) (nullable, wide), k INT 0..2, w TEXT."""
    var t = Table.empty()
    var x = t.add_column("x", INT)
    var r = t.add_column("r", DECIMAL, scale=2)
    var k = t.add_column("k", INT)
    var w = t.add_column("w", TEXT)
    var rng = Rng(seed)
    for i in range(n):
        if i % 14 == 3:
            t.append_null(x)
        else:
            t.append(x, Int64(rng.int(-200, 200)))
        if i % 9 == 4:
            t.append_null(r)
        else:
            t.append(r, Int64(rng.int(-5_000_000, 5_000_000)))
        t.append(k, Int64(rng.int(0, 2)))
        t.append_text(w, "a")
    return t^.share()


def the_edges(t: SharedTable, name: String, bins: Int, filter_k: Int) -> List[Int64]:
    """Edge k = the value at sorted place ceil(k * N / bins) (1-based), by a plain loop over the rows with k == filter_k (-1: all)."""
    var vals = List[Int64]()
    var c = t.index(name)
    var kc = t.index("k")
    for row in range(t.rows()):
        if filter_k >= 0 and t.get(kc, row) != Int64(filter_k):
            continue
        if not t.is_null(c, row):
            vals.append(t.get(c, row))
    msort(vals)
    var edges = List[Int64]()
    var total = len(vals)
    for k in range(1, bins):
        edges.append(vals[max((k * total + bins - 1) // bins - 1, 0)])
    return edges^


def bin_of(edges: List[Int64], v: Int64) -> Int:
    var b = 0
    for e in edges:
        if e < v:
            b += 1
    return b


def check_counts(t: SharedTable, name: String, bins: Int, filter_k: Int, edges: List[Int64], got: Query, label: String) raises:
    """The bins' row counts and sums of `name` from the plain loop equal the query's."""
    var cnt = List[Int](length=bins, fill=0)
    var sm = List[Int](length=bins, fill=0)
    var c = t.index(name)
    var kc = t.index("k")
    for row in range(t.rows()):
        if filter_k >= 0 and t.get(kc, row) != Int64(filter_k):
            continue
        if t.is_null(c, row):
            continue
        var b = bin_of(edges, t.get(c, row))
        cnt[b] += 1
        sm[b] += Int(t.get(c, row))
    var r = got.compute(CPU)
    var total = 0
    for b in range(bins):
        expect(r.counts[b] == cnt[b], label + ": bin " + String(b) + " holds " + String(r.counts[b]) + " rows, the loop counts " + String(cnt[b]))
        expect(Int(r.sums[b]) == sm[b], label + ": bin " + String(b) + " sums to " + String(r.sums[b]) + ", not " + String(sm[b]))
        total += cnt[b]
    expect(total > 0, label + ": rows were binned")


def test_a_table() raises:
    var t = make_table(N, 1)
    for bins in [2, 3, 10, 32]:
        var qb = Query(t).quantile_bins(col("x"), bins)
        var want = the_edges(t, "x", bins, -1)
        expect(len(qb.edges) == bins - 1 and qb.bins() == bins, "edges and bins")
        for i in range(bins - 1):
            expect(qb.edges[i] == want[i], "bins " + String(bins) + ": edge " + String(i + 1) + " is " + String(qb.edges[i]) + ", the loop says " + String(want[i]))
        check_counts(t, "x", bins, -1, want, Query(t).group_by(qb.key(col("x")), groups=bins).agg(count(), sum(col("x"))), "x " + String(bins))
    # a DECIMAL column (edges are the stored hundredths) under a filter
    var qd = Query(t).where(col("k") == 1).quantile_bins(col("r"), 7)
    var wd = the_edges(t, "r", 7, 1)
    for i in range(6):
        expect(qd.edges[i] == wd[i], "decimal edge " + String(i + 1))
    check_counts(t, "r", 7, 1, wd, Query(t).where(col("k") == 1).group_by(qd.key(col("r")), groups=7).agg(count(), sum(col("r"))), "r")


def test_ties_and_small_inputs() raises:
    var t = Table.empty()
    var v = t.add_column("v", INT)
    for x in [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]:                # N = 12, 3 bins: edges at places 4 and 8 -> 1 and 2
        t.append(v, Int64(x))
    var s = t^.share()
    var qb = Query(s).quantile_bins(col("v"), 3)
    expect(qb.edges[0] == 1 and qb.edges[1] == 2, "edges of the tied column")
    var r = Query(s).group_by(qb.key(col("v")), groups=3).agg(count(), min_of(col("v")), max_of(col("v"))).compute(CPU)
    expect(r.counts[0] == 5 and r.counts[1] == 4 and r.counts[2] == 3, "ties share a bin: 5, 4, 3 rows")
    var e = Table.empty()                                           # no rows: edges 0, no bin is reached
    _ = e.add_column("v", INT)
    var qe = Query(e^.share()).quantile_bins(col("v"), 4)
    expect(len(qe.edges) == 3 and qe.edges[0] == 0, "an empty table")


def test_a_frame_with_edits_and_a_scan() raises:
    var t = make_table(60_000, 3)
    var f = Frame(t)
    var rng = Rng(5)
    for _ in range(300):
        var row = rng.int(0, f.rows() - 1)
        while not f.alive(row):
            row = rng.int(0, f.rows() - 1)
        var what = rng.int(0, 3)
        if what == 0:
            f.set(f.column("x"), row, Int64(rng.int(-300, 300)))
        elif what == 1:
            f.set_null(f.column("x"), row)
        elif what == 2:
            _ = f.append(["5", "12.50", "1", "a"])
        else:
            f.remove(row)
    var baked = f.bake().share()
    var qb = Query(f).quantile_bins(col("x"), 8)
    var want = the_edges(baked, "x", 8, -1)
    for i in range(7):
        expect(qb.edges[i] == want[i], "frame edge " + String(i + 1) + ": " + String(qb.edges[i]) + " vs " + String(want[i]))
    check_counts(baked, "x", 8, -1, want, Query(f).group_by(qb.key(col("x")), groups=8).agg(count(), sum(col("x"))), "frame")
    var mem = make_table(N, 6)
    var path = String("build/mojo-tests/f177_bins.parquet")
    write_parquet(path, mem, "{}", ZSTD, 20_000)
    var scan = scan_parquet(path)
    var qs = Query(scan).quantile_bins(col("x"), 12)
    var ws = the_edges(mem, "x", 12, -1)
    for i in range(11):
        expect(qs.edges[i] == ws[i], "scan edge " + String(i + 1))
    check_counts(mem, "x", 12, -1, ws, Query(scan).group_by(qs.key(col("x")), groups=12).agg(count(), sum(col("x"))), "scan")


def test_a_float_column() raises:
    var t = Table.empty()
    var f = t.add_column("f", FLOAT)
    var rng = Rng(11)
    var xs = List[Float64]()
    for i in range(4000):
        if i % 17 == 0:
            t.append_null(f)
        else:
            var x = Float64(rng.int(-300, 300)) * 0.25 + 0.1
            xs.append(x)
            t.append_float(f, x)
    var s = t^.share()
    var qb = Query(s).quantile_bins(col("f"), 5)
    expect(qb.is_float, "float edges")
    var sorted = List[Float64]()
    for x in xs:
        var at = len(sorted)
        sorted.append(x)
        while at > 0 and sorted[at - 1] > sorted[at]:
            var tmp = sorted[at - 1]
            sorted[at - 1] = sorted[at]
            sorted[at] = tmp
            at -= 1
    for k in range(1, 5):
        var want = sorted[max((k * len(sorted) + 4) // 5 - 1, 0)]
        expect(Float64(from_bits=UInt64(qb.edges[k - 1])) == want, "float edge " + String(k))
    var r = Query(s).group_by(qb.key(col("f")), groups=5).agg(count()).compute(CPU)
    var total = 0
    for b in range(5):
        total += r.counts[b]
    expect(total == len(xs), "every non-null row is in a bin")
    expect(r.counts[0] >= len(xs) // 5 and r.counts[4] <= len(xs) // 5 + len(xs) // 10, "the bins are about equal")


def test_errors() raises:
    var t = make_table(1000, 7)
    for bins in [1, 33, 0, -4]:
        var msg = String()
        try:
            _ = Query(t).quantile_bins(col("x"), bins)
        except e:
            msg = String(e)
        expect("from 2 to 32" in msg, "bins " + String(bins) + ": " + msg)
    var msg = String()
    try:
        _ = Query(t).quantile_bins(col("w"), 4)
    except e:
        msg = String(e)
    expect("need numbers" in msg, "text: " + msg)
    msg = ""
    try:
        _ = Query(t).agg(count()).quantile_bins(col("x"), 4)
    except e:
        msg = String(e)
    expect("before group_by and agg" in msg, "after agg: " + msg)


def main() raises:
    test_a_table()
    test_ties_and_small_inputs()
    test_a_frame_with_edits_and_a_scan()
    test_a_float_column()
    test_errors()
    print("ok")
