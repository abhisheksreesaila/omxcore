# F-157 (M8): FLOAT64 as a column kind, from Mojo with no Python: tables, Parquet, CSV, Query, Frame and SQL, with exact expected values
# (sums of multiples of 0.25 are exact in any order). tests/test_mojo_float*.py compare many more queries with DuckDB on random doubles.
from std.os import makedirs
from omxcore.wide import wide_make
from std.ffi import external_call
from std.sys import CompilationTarget
from omxcore import (float_text, Table, SharedTable, Frame, Catalog, Query, QueryResult, INT, DECIMAL, TEXT, FLOAT, F32_COLUMN, col, lit_float, lit_f32,
                     sum, count, mean, min_of, max_of, count_distinct, median, percentile, sql, PlannerOptions, GROUP_DENSE,
                     read_parquet, write_parquet, scan_parquet, read_csv, CsvOptions, ZSTD, CPU, AUTO)

comptime DIR = "/home/abhisheksreesaila/.cache/mxframe_float_test/"
comptime NAN_BITS = Int64(0x7FF8000000000000)


def fail(msg: String) raises:
    raise Error("FAILED: " + msg)


def expect(cond: Bool, what: String) raises:
    if not cond:
        fail(what)


def bits(x: Float64) -> Int64:
    return Int64(x.to_bits())


def is_nan(x: Float64) -> Bool:
    return x != x


def same(a: Float64, b: Float64) -> Bool:
    """Equal values, NaN equal to NaN (bits of zeros may differ: -0.0 == 0.0)."""
    return a == b or (is_nan(a) and is_nan(b))


struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed * 0x9E3779B97F4A7C15 + 1

    def next(mut self) -> UInt64:
        self.s ^= self.s << 13
        self.s ^= self.s >> 7
        self.s ^= self.s << 17
        return self.s

    def int(mut self, lo: Int, hi: Int) -> Int:
        return lo + Int(self.next() % UInt64(hi - lo + 1))


def specials() -> List[Float64]:
    var inf = Float64(1.0) / Float64(0.0)
    return [0.0, -0.0, 1.5, -2.25, Float64(from_bits=UInt64(0x7FF8000000000000)), Float64(from_bits=UInt64(0xFFF8000000000123)),
            Float64(from_bits=UInt64(0x7FF800000000BEEF)), inf, -inf, 5e-324, -5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 0.1]


def special_table() raises -> SharedTable:
    var t = Table.empty()
    var x = t.add_column("x", FLOAT)
    var sp = specials()
    for i in range(2000):
        if i % 11 == 4:
            t.append_null(x)
        else:
            t.append_float(x, sp[i % len(sp)])
    return t^.share()


def test_table_float_cells() raises:
    var t = special_table()
    expect(t.kind(0) == FLOAT, "the column is FLOAT")
    var sp = specials()
    for i in range(2000):
        if i % 11 == 4:
            expect(t.is_null(0, i), "null cell")
        else:
            expect(t.get(0, i) == bits(sp[i % len(sp)]), "the bits of a float are kept as given (-0.0, a NaN's payload, subnormals)")
    var bad = Table.empty()
    _ = bad.add_column("n", INT)
    var raised = False
    try:
        bad.append_float(0, 1.0)
    except:
        raised = True
    expect(raised, "append_float to an INT column raises")
    raised = False
    try:
        _ = bad.add_column("w", FLOAT, scale=3)
    except e:
        raised = String(e).find("a FLOAT's scale is 0") >= 0
    expect(raised, "a FLOAT's scale is 0 or 32")


def test_parquet_round_trip() raises:
    makedirs(DIR, exist_ok=True)
    var t = special_table()
    var path = DIR + "special.parquet"
    write_parquet(path, t, "", ZSTD, 704)
    var rp = read_parquet(path)
    var back = rp.take_table().share()
    expect(back.kind(0) == FLOAT and back.rows() == 2000, "read back as FLOAT")
    for i in range(2000):
        expect(back.is_null(0, i) == t.is_null(0, i), "nulls survive")
        if not t.is_null(0, i):
            expect(back.get(0, i) == t.get(0, i), "every bit survives a Parquet round trip")
    var scan = scan_parquet(path)
    expect(scan.kind(0) == FLOAT, "a scan sees FLOAT")
    var g = scan.read_group(1, [0])
    expect(g.rows() == 704 and g.get(0, 0) == t.get(0, 704), "a row group of a scan decodes the same bits")
    var range_known = scan.column_range(0)
    expect(not range_known[0], "a float column states no range: the footer's min / max are never used")


def test_csv_rule() raises:
    makedirs(DIR, exist_ok=True)
    var path = DIR + "f.csv"
    with open(path, "w") as f:
        f.write("a,b,c,d,e\n1,1.5,1.25,1e3,x\n2,2.25,2.5001,2.5E-3,y\n3,,3.1234567890123,-7,z\n")
    var r = read_csv(path)
    expect(r.table.kind(0) == INT, "whole numbers stay INT")
    expect(r.table.kind(1) == DECIMAL and r.table.scale(1) == 2, "up to 2 places: DECIMAL(2)")
    expect(r.table.kind(2) == FLOAT, "more than 9 places in the first rows: FLOAT")
    expect(r.table.kind(3) == FLOAT, "an exponent: FLOAT")
    expect(r.table.get_float(2, 2) == 3.1234567890123 and r.table.get_float(3, 1) == 0.0025 and r.table.get_float(3, 2) == -7.0, "the values are parsed exactly")
    expect(r.table.get_float(2, 1) == 2.5001, "2.5001 (4 places) in a FLOAT column is the double 2.5001")
    var opts = CsvOptions()
    opts.float_columns.append("b")
    var r2 = read_csv(path, opts)
    expect(r2.table.kind(1) == FLOAT and r2.table.get_float(1, 1) == 2.25 and r2.table.is_null(1, 2), "float_columns reads the named columns as FLOAT")


def quarter_table() raises -> SharedTable:
    """g INT 0..2 (one null), x FLOAT: multiples of 0.25 (one NaN, one null, one -inf), i INT."""
    var t = Table.empty()
    var g = t.add_column("g", INT)
    var x = t.add_column("x", FLOAT)
    var i = t.add_column("i", INT)
    var rng = Rng(5)
    for r in range(3000):
        if r == 1000:
            t.append_null(g)
        else:
            t.append(g, Int64(r % 3))
        if r == 7:
            t.append_float(x, Float64(from_bits=UInt64(0x7FF8000000000000)))
        elif r == 9:
            t.append_null(x)
        elif r == 11:
            t.append_float(x, -Float64(1.0) / Float64(0.0))
        else:
            t.append_float(x, Float64(rng.int(-40, 40)) * 0.25)
        t.append(i, Int64(r % 7))
    return t^.share()


def test_builder_global_and_grouped() raises:
    var t = quarter_table()
    var r = Query(t).agg(count(), count(col("x")), sum(col("x")), mean(col("x")), min_of(col("x")), max_of(col("x"))).compute(CPU)
    expect(r.counts[0] == 3000, "count()")
    expect(r.values(1)[0] == 2999, "count(x) skips the null (a NaN is a value)")
    expect(is_nan(r.floats(2)[0]), "the sum of a column holding a NaN is NaN")
    expect(is_nan(r.floats(5)[0]), "MAX is NaN when the group holds a NaN (it is above every number)")
    expect(r.floats(4)[0] == -(Float64(1.0) / Float64(0.0)), "MIN skips the NaN and finds -inf")
    # the rows without the NaN and -inf: sums of multiples of 0.25 are exact in any order
    var q = Query(t).where(col("x") > lit_float(-1e300)).where(col("x") < lit_float(1e300)).agg(count(), sum(col("x")), min_of(col("x")), max_of(col("x")), mean(col("x")))
    var r2 = q^.compute(CPU)
    var n = 0
    var total = Float64(0.0)
    var lo = Float64(1e300)
    var hi = Float64(-1e300)
    var xs = t.values(1)
    for k in range(3000):
        if t.is_null(1, k):
            continue
        var v = Float64(from_bits=UInt64(xs[k]))
        if is_nan(v) or v < -1e300:
            continue
        n += 1
        total += v
        lo = min(lo, v)
        hi = max(hi, v)
    expect(r2.counts[0] == n, "where on floats counts the rows")
    expect(r2.floats(1)[0] == total and r2.floats(2)[0] == lo and r2.floats(3)[0] == hi, "sum / min / max of floats")
    expect(r2.means(4)[0] == total / Float64(n), "the mean of floats")
    var gq = Query(t).group_by(col("g"), groups=3).where(col("x") > lit_float(-1e300)).where(col("x") < lit_float(1e300)).agg(count(), sum(col("x") * 2.0 + col("i")), max_of(col("x") - col("i")))
    var rg = gq^.compute(CPU)
    var sums = List[Float64](length=3, fill=0.0)
    var counts = List[Int](length=3, fill=0)
    var best = List[Float64](length=3, fill=-1e300)
    var ints = t.values(2)
    var gs = t.values(0)
    for k in range(3000):
        if t.is_null(0, k) or t.is_null(1, k):
            continue
        var v = Float64(from_bits=UInt64(xs[k]))
        if is_nan(v) or v < -1e300:
            continue
        var g = Int(gs[k])
        counts[g] += 1
        sums[g] += v * 2.0 + Float64(ints[k])
        best[g] = max(best[g], v - Float64(ints[k]))
    for g in range(3):
        expect(rg.counts[g] == counts[g], "group counts")
        expect(rg.floats(1)[g] == sums[g] and rg.floats(2)[g] == best[g], "grouped float arithmetic")


def test_the_sum_is_the_same_for_any_threads() raises:
    var t = Table.empty()
    var x = t.add_column("x", FLOAT)
    var g = t.add_column("g", INT)
    var rng = Rng(9)
    for r in range(400_000):
        t.append_float(x, Float64(rng.int(-1_000_000, 1_000_000)) * 1.0e-3 / 3.0 * (1.0 + 1e-7 * Float64(r % 13)))
        t.append(g, Int64(rng.int(0, 9)))
    var shared = t^.share()
    var first = Query(shared).threads(1).group_by(col("g"), groups=10).agg(sum(col("x")), mean(col("x"))).compute(CPU)
    for n in [2, 3, 5, 8, 0]:
        var again = Query(shared).threads(n).group_by(col("g"), groups=10).agg(sum(col("x")), mean(col("x"))).compute(CPU)
        for k in range(10):
            expect(bits(again.floats(0)[k]) == bits(first.floats(0)[k]), "the float sum is the same bits for " + String(n) + " threads")
            expect(bits(again.floats(1)[k]) == bits(first.floats(1)[k]), "the float mean is the same bits for " + String(n) + " threads")
    var global1 = Query(shared).threads(1).agg(sum(col("x"))).compute(CPU)
    var global7 = Query(shared).threads(7).agg(sum(col("x"))).compute(CPU)
    expect(bits(global1.floats(0)[0]) == bits(global7.floats(0)[0]), "a global float sum is the same bits for any threads")


def test_holistic_floats() raises:
    var t = Table.empty()
    var x = t.add_column("x", FLOAT)
    var vals: List[Float64] = [3.0, 1.0, 2.0, 4.0, Float64(from_bits=UInt64(0x7FF8000000000000)), 2.0, -0.0, 0.0]
    for v in vals:
        t.append_float(x, v)
    t.append_null(x)
    var r = Query(t^.share()).agg(count_distinct(col("x")), median(col("x")), percentile(col("x"), 0.0), percentile(col("x"), 1.0), percentile(col("x"), 0.5)).compute(CPU)
    expect(r.values(0)[0] == 6, "count distinct: -0.0 and 0.0 are one value, NaN is one value, the null is not counted: 0, 1, 2, 3, 4, NaN")
    # sorted: -0.0 0.0 1 2 2 3 4 NaN (8 values): the median is the mean of the 4th and 5th: 2
    expect(r.floats(1)[0] == 2.0, "median of 8 values: the mean of 2 and 2")
    expect(r.floats(2)[0] == 0.0, "quantile_disc(0) is the smallest")
    expect(is_nan(r.floats(3)[0]), "quantile_disc(1) is NaN: NaN is above every number")
    expect(r.floats(4)[0] == 2.0, "quantile_disc(0.5) is the value at index ceil(0.5 * 8) - 1 = 3")


def test_frame_floats() raises:
    makedirs(DIR, exist_ok=True)
    var t = Table.empty()
    var x = t.add_column("x", FLOAT)
    var n = t.add_column("n", INT)
    var d = t.add_column("d", DECIMAL, scale=2)
    var base: List[Float64] = [3.5, -1.0, 2.0, 8.25]
    for i in range(4):
        t.append_float(x, base[i])
        t.append(n, Int64(i + 1))
        t.append(d, Int64(100 * (i + 1) + 25))
    var f = Frame(t^.share())
    f.set_float(0, 1, Float64(from_bits=UInt64(0x7FF8000000000000)))
    f.set_text(0, 2, "1.5e1")
    f.set_text(0, 3, "-0.0")
    var raised = False
    try:
        f.set_text(0, 0, "abc")
    except e:
        raised = String(e).find("column `x`: `abc` isn't a number") >= 0
    expect(raised, "unparseable float text is refused, naming the column")
    raised = False
    try:
        f.set_text(0, 0, "1e999")
    except e:
        raised = String(e).find("past the largest float64") >= 0
    expect(raised, "a float past float64 is refused")
    expect(f.get_float(0, 2) == 15.0 and f.get_float(0, 0) == 3.5 and f.get_float(0, 3) == 0.0, "edited float cells")
    _ = f.formula("m = x * 2 + n / 4 - d")
    expect(f.kind(f.column("m")) == FLOAT, "a formula with a float is a FLOAT column")
    expect(f.get_float(f.column("m"), 0) == 3.5 * 2.0 + 1.0 / 4.0 - 1.25, "the formula reads the decimal as float(v) / 100")
    expect(f.get_float(f.column("m"), 2) == 15.0 * 2.0 + 3.0 / 4.0 - 3.25, "the formula follows the edit")
    var order = f.permutation(0)
    expect(order[0] == 3 and order[1] == 0 and order[2] == 2 and order[3] == 1, "sorted ascending: -0.0, 3.5, 15, NaN last")
    var desc = f.permutation(0, descending=True)
    expect(desc[0] == 1 and desc[1] == 2 and desc[2] == 0 and desc[3] == 3, "descending: NaN first")
    var r = Query(f).agg(count(), sum(col("x")), max_of(col("x")), min_of(col("x")), sum(col("m"))).compute(CPU)
    expect(is_nan(r.floats(1)[0]) and is_nan(r.floats(2)[0]) and r.floats(3)[0] == 0.0, "a query over the edited frame (the NaN cell is in the delta)")
    f.set_float(0, 1, 100.0)
    var r2 = Query(f).agg(sum(col("x")), max_of(col("x")), min_of(col("x"))).compute(CPU)
    expect(r2.floats(0)[0] == 100.0 + 3.5 + 15.0 + 0.0 and r2.floats(1)[0] == 100.0 and r2.floats(2)[0] == 0.0, "after setting the NaN to 100")
    var path = DIR + "frame.parquet"
    f.save(path)
    var rp = read_parquet(path)
    var back = rp.take_table().share()
    expect(back.kind(0) == FLOAT and back.get_float(0, 1) == 100.0 and back.get_float(0, 2) == 15.0, "save writes the edited floats as DOUBLE")


def test_the_smallest_negative_subnormal_is_not_a_null() raises:
    """-5e-324 has the bits INT64_MIN + 1, which the Frame once used to mark a null edit: a float cell must keep it."""
    var t = Table.empty()
    var x = t.add_column("x", FLOAT)
    t.append_float(x, -5e-324)
    t.append_float(x, Float64(from_bits=UInt64(0x7FF8000000000001)))
    t.append_null(x)
    var f = Frame(t^.share())
    expect(not f.is_null(0, 0) and f.get_float(0, 0) == -5e-324, "-5e-324 is a value")
    expect(not f.is_null(0, 1) and is_nan(f.get_float(0, 1)), "the NaN with the marker's payload is a NaN, not a null")
    expect(f.is_null(0, 2), "the null is a null")
    var b = f.bake()
    expect(not b.cols[0].is_null(0) and b.get_float(0, 0) == -5e-324 and not b.cols[0].is_null(1) and b.cols[0].is_null(2), "bake keeps them")
    var r = Query(f).agg(count(col("x")), min_of(col("x"))).compute(CPU)
    expect(r.values(0)[0] == 2 and r.floats(1)[0] == -5e-324, "a query counts them")


def test_marker_values_are_data_in_base_columns() raises:
    """The Frame marks an edited null with a NaN's bits (9221120237041090561), but a base cell is null by its validity bit only: an INT or
    DECIMAL cell may hold that value, or the previous marker (-9223372036854775807), as data."""
    var new_marker = Int64(0x7FF8000000000001)
    var old_marker = Int64.MIN + 1
    var t = Table.empty()
    var n = t.add_column("n", INT)
    var d = t.add_column("d", DECIMAL, scale=2)
    var g = t.add_column("g", INT)
    var vals: List[Int64] = [new_marker, old_marker, 7, new_marker, 5]
    for i in range(5):
        t.append(n, vals[i])
        t.append(d, new_marker if i % 2 == 0 else Int64(100))
        t.append(g, Int64(i % 2))
    t.append_null(n)
    t.append(d, 3)
    t.append(g, 0)
    var shared = t^.share()
    var f = Frame(shared)
    for i in range(5):
        expect(not f.is_null(0, i) and f.get(0, i) == vals[i], "a base cell holding a marker value is data")
    expect(f.is_null(0, 5), "the null is a null")
    var r0 = Query(f).agg(count(col("n")), min_of(col("n")), max_of(col("n"))).compute(CPU)
    expect(r0.values(0)[0] == 5 and Int64(r0.values(1)[0]) == old_marker and Int64(r0.values(2)[0]) == new_marker, "a query over the frame counts them")
    f.set(2, 1, 9)                                            # an edit puts the row in the delta, with its marker-valued cells
    f.set_null(0, 3)
    f.set(1, 4, 11)
    var r1 = Query(f).group_by(col("g"), groups=2).agg(count(col("n")), max_of(col("n")), min_of(col("d"))).compute(CPU)
    expect(r1.values(0)[0] == 3 and r1.values(0)[1] == 0, "counts of the non-null cells after the edits")
    expect(Int64(r1.values(1)[0]) == new_marker and r1.values(2)[0] == 3, "a marker-valued base cell survives in the delta rows")
    var cat = Catalog()
    cat.add_frame("t", f)
    var s = sql(cat, "SELECT count(n), max(n), min(n) FROM t")
    expect(s.table.get(0, 0) == 4 and s.table.get(1, 0) == new_marker and s.table.get(2, 0) == old_marker, "SQL over the frame sees them as data")
    var b = f.bake()
    expect(b.get(0, 0) == new_marker and b.get(0, 1) == old_marker and b.cols[0].is_null(3) and b.cols[0].is_null(5), "bake keeps them")
    var order = f.permutation(0)
    expect(order[0] == 1 and order[3] == 0 and order[4] == 3, "sorted: the old marker is the smallest, the new one the largest, nulls last")


def test_digits_parse_in_the_c_locale_whatever_the_process_locale() raises:
    """The slow path of the float parser (strtod_l with the C locale) must not follow LC_NUMERIC: a host that called setlocale with a comma
    for the decimal point would otherwise stop every long number at its point. Tries the locales a machine may have (none installed here
    means the parse is checked under "C" only)."""
    makedirs(DIR, exist_ok=True)
    var path = DIR + "locale.csv"
    with open(path, "w") as f:
        f.write("x\n1.00000000000000000000001e0\n123456789012345678901.5\n0.1\n")
    var names: List[String] = ["de_DE.UTF-8", "fr_FR.UTF-8", "de_DE.utf8", "fr_FR.utf8", "de_DE", "fr_FR", "pt_BR.UTF-8", "ru_RU.UTF-8"]
    comptime LC_NUMERIC = 1 if CompilationTarget.is_linux() else 4
    var switched = 0
    for n in names:
        var c = List[UInt8](n.as_bytes())
        c.append(0)
        var got = external_call["setlocale", Int](Int32(LC_NUMERIC), Int(c.unsafe_ptr()))
        if got != 0:
            switched += 1
            var r = read_csv(path)
            expect(r.table.kind(0) == FLOAT and r.table.get_float(0, 0) == 1.0 and r.table.get_float(0, 1) == 123456789012345678901.5
                   and r.table.get_float(0, 2) == 0.1, "a long number parses to the same float under " + n)
            var c2 = List[UInt8](length=2, fill=0)
            c2[0] = 67
            _ = external_call["setlocale", Int](Int32(LC_NUMERIC), Int(c2.unsafe_ptr()))
        _ = c^
    var r = read_csv(path)
    expect(r.table.kind(0) == FLOAT and r.table.get_float(0, 1) == 123456789012345678901.5, "and under C")


def test_float_text_is_the_shortest_that_reads_back() raises:
    expect(float_text(0.1) == "0.1" and float_text(0.30000000000000004) == "0.30000000000000004" and float_text(100.0) == "100.0", "plain values")
    expect(float_text(1e-7) == "1e-07" and float_text(5e-324) == "5e-324" and float_text(-0.0) == "-0.0" and float_text(1e16) == "1e+16", "exponents")
    # the double Mojo's own formatter prints with one digit too few (8.69563793934264e+17 reads back as the neighbour)
    var x = Float64(from_bits=UInt64(0x4389E3C2C7D3B6A9))
    var t = float_text(x)
    var back = Float64(0.0)
    var ok = False
    with open(DIR + "t.csv", "w") as f:
        f.write("x\n" + t + "\n1e0\n")
    var rr = read_csv(DIR + "t.csv")
    expect(rr.table.get_float(0, 0) == x, "it reads back: " + t)


def test_money_with_many_places_is_float_by_default_and_exact_on_request() raises:
    """fx.csv: rate has 6 places and qty is large. Read as DuckDB does (FLOAT) `sum(rate * qty)` answers; as an exact DECIMAL(6) the product
    passes int64 and is kept in 128 bits (F-192), while `sum(rate)` is exact."""
    makedirs(DIR, exist_ok=True)
    var path = DIR + "fx.csv"
    var rng = Rng(3)
    var exact_total = Int64(0)
    var float_total = Float64(0.0)
    var exact_product = Int128(0)
    with open(path, "w") as f:
        f.write("rate,qty\n")
        for _ in range(200_000):
            var micro = rng.int(1_000_000, 999_999_999)
            var qty = rng.int(100_000, 1_000_000)
            exact_total += Int64(micro)
            float_total += Float64(micro) / 1.0e6 * Float64(qty)
            exact_product += Int128(micro) * Int128(qty)
            f.write(String(micro // 1_000_000) + "." + String(1_000_000 + micro % 1_000_000)[byte=1:] + "," + String(qty) + "\n")
    var by_default = read_csv(path)
    expect(by_default.table.kind(0) == FLOAT, "6 places: FLOAT by default")
    var cat = Catalog()
    cat.add_table("fx", by_default.take_table().share())
    var r = sql(cat, "SELECT sum(rate * qty), sum(rate) FROM fx")
    expect(r.table.get_float(0, 0) > 1.0e13 and abs(r.table.get_float(0, 0) / float_total - 1.0) < 1e-9, "the product sum answers, as DuckDB's does")
    var opts = CsvOptions()
    opts.decimal_columns.append("rate")
    var exact = read_csv(path, opts)
    expect(exact.table.kind(0) == DECIMAL and exact.table.scale(0) == 6, "decimal_columns: DECIMAL(6)")
    var cat2 = Catalog()
    cat2.add_table("fx", exact.take_table().share())
    var s1 = sql(cat2, "SELECT sum(rate) FROM fx")
    expect(s1.table.get(0, 0) == exact_total, "the exact sum of the six-place rates, to the last digit")
    var p = sql(cat2, "SELECT sum(rate * qty) FROM fx")           # F-192: the product sum passes int64 and is kept in 128 bits, as DuckDB's DECIMAL(38)
    expect(p.table.is_wide(0), "the exact product sum is a total past int64")
    expect(wide_make(p.table.get(0, 0), p.table.get_high(0, 0)) == exact_product, "and it is the exact sum of the products, to the last digit")


def test_float_text_of_powers_of_two() raises:
    """At a power of two the double below is twice as close as the one above, so the nearest 16 digits can fall outside the interval while a
    neighbour of the last digit is inside (2^-77 is 6.617444900424222e-24, not ...2214e-24). Every power of two reads back; tests/test_mx.py compares each with Python repr."""
    makedirs(DIR, exist_ok=True)
    var path = DIR + "pow2.csv"
    var texts = List[String]()
    var xs = List[Float64]()
    for e in range(-1022, 1024):
        var x = Float64(from_bits=UInt64((e + 1023) << 52))          # 2^e: mantissa 0
        xs.append(x)
        texts.append(float_text(x))
    with open(path, "w") as f:
        f.write("x\n")
        for t in texts:
            f.write(t + "\n")
    var opts = CsvOptions()
    opts.float_columns.append("x")
    var r = read_csv(path, opts)
    for i in range(len(xs)):
        expect(r.table.get_float(0, i) == xs[i], "2^e reads back from " + texts[i])
    var tiny = Float64(from_bits=UInt64((1023 - 77) << 52))
    expect(float_text(tiny) == "6.617444900424222e-24", "2^-77: " + float_text(tiny))


def sql_table() raises -> Table:
    var t = Table.empty()
    var k = t.add_column("k", FLOAT)
    var v = t.add_column("v", FLOAT)
    var n = t.add_column("n", INT)
    var f = t.add_column("f", FLOAT, scale=F32_COLUMN)
    var nan = Float64(from_bits=UInt64(0x7FF8000000000000))
    var negnan = Float64(from_bits=UInt64(0xFFF8000000000123))
    var keys: List[Float64] = [1.5, -0.0, 0.0, nan, negnan, 1.5, 2.0, -(Float64(1.0) / Float64(0.0))]
    for i in range(8):
        t.append_float(k, keys[i])
        t.append_float(v, Float64(i + 1) * 0.25)
        t.append(n, Int64(i))
        t.append_float(f, Float64(Float32(0.07)) if i % 2 == 0 else Float64(Float32(0.05)))
    t.append_null(k)
    t.append_float(v, 10.0)
    t.append(n, 8)
    t.append_float(f, Float64(Float32(0.07)))
    return t^


def cell(t: Table, c: Int, r: Int) -> String:
    if t.is_null(c, r):
        return "NULL"
    if t.kind(c) == FLOAT:
        var x = t.get_float(c, r)
        return "nan" if is_nan(x) else String(x)
    return String(t.get(c, r))


def test_sql_floats() raises:
    var cat = Catalog()
    cat.add_table("t", sql_table()^.share())
    var r = sql(cat, "SELECT k, count(*), sum(v), max(n) FROM t GROUP BY k ORDER BY k")
    expect(r.table.rows() == 6, "-0.0 and 0.0 are one group, every NaN is one group, NULL is a group: 1.5, 0, 2, -inf, nan, NULL")
    var want = String("-inf 1 2.0 7;0.0 2 1.25 2;1.5 2 1.75 5;2.0 1 1.75 6;nan 2 2.25 4;NULL 1 10.0 8")
    var got = String()
    for i in range(r.table.rows()):
        got += (";" if i > 0 else "") + cell(r.table, 0, i) + " " + cell(r.table, 1, i) + " " + cell(r.table, 2, i) + " " + cell(r.table, 3, i)
    expect(got == want, "group by a float, ordered: -inf first, NaN after every number, NULL last\n" + got)
    var hashed = False
    for note in r.notes:
        hashed = hashed or note.find("group: hashed") >= 0
    expect(hashed, "a float key is grouped by hashing")
    var forced = PlannerOptions()
    forced.group = GROUP_DENSE
    var raised = False
    try:
        _ = sql(cat, "SELECT k, count(*) FROM t GROUP BY k", forced)
    except e:
        raised = String(e).find("is a FLOAT: floats are grouped by hashing, never by a dense range") >= 0
    expect(raised, "forcing dense groups on a float key is refused, with the reason")
    var c = sql(cat, "SELECT count(*), sum(v * 2 + n), avg(v), min(k), max(k), count(DISTINCT k), median(v) FROM t WHERE v >= 0.5 AND k = k")
    expect(cell(c.table, 0, 0) == "7" and cell(c.table, 1, 0) == "45.5", "v * 2 + n over the seven rows with v >= 0.5 and a key: 17.5 + 28")
    expect(cell(c.table, 2, 0) == String(Float64(2 + 3 + 4 + 5 + 6 + 7 + 8) * 0.25 / 7.0), "avg of floats")
    expect(cell(c.table, 3, 0) == "-inf" and cell(c.table, 4, 0) == "nan" and cell(c.table, 5, 0) == "5", "min skips NaN, max is NaN, 5 distinct keys")
    var m = sql(cat, "SELECT count(*) FROM t WHERE f <= 0.07")
    expect(cell(m.table, 0, 0) == "9", "a 32-bit float column is compared with a literal in 32 bits: the five 0.07 rows are kept")
    var m2 = sql(cat, "SELECT count(*) FROM t WHERE f + 0 <= 0.07")
    expect(cell(m2.table, 0, 0) == "9", "REAL arithmetic stays REAL (rounded to 32 bits), so the literal is a REAL too")
    var m3 = sql(cat, "SELECT count(*) FROM t WHERE f <= 7e-2")
    expect(cell(m3.table, 0, 0) == "4", "an exponent literal is a DOUBLE: the double 0.07 is below the float 0.07, so those five rows are out")


def main() raises:
    test_table_float_cells()
    test_parquet_round_trip()
    test_csv_rule()
    test_builder_global_and_grouped()
    test_the_sum_is_the_same_for_any_threads()
    test_holistic_floats()
    test_frame_floats()
    test_the_smallest_negative_subnormal_is_not_a_null()
    test_marker_values_are_data_in_base_columns()
    test_sql_floats()
    test_digits_parse_in_the_c_locale_whatever_the_process_locale()
    test_float_text_is_the_shortest_that_reads_back()
    test_float_text_of_powers_of_two()
    test_money_with_many_places_is_float_by_default_and_exact_on_request()
    print("ok")
