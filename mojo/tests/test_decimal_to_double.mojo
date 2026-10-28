# F-190: an INT or DECIMAL read as a DOUBLE rounds to the nearest double, as DuckDB's cast does (float(v) / 10^scale), also past 2^53.
# It used to raise "int64 range exceeded", so TPC-H Q14 (100.00 * sum(l_extendedprice) / sum(...)) failed at SF1. tests/test_f190_decimal_to_double.py
# compares the same values with DuckDB bit for bit; this file checks the engine alone (no Python): the SQL cast, the builder, a sum, and that
# an int64 that really overflows still raises.
from omxcore import (Table, SharedTable, Catalog, Query, INT, DECIMAL, col, sum, count, min_of, max_of, sql, CPU, GPU, AUTO)


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


def p10(scale: Int) -> Float64:
    var f = Float64(1.0)
    for _ in range(scale):
        f *= 10.0
    return f


def duckdb_double(raw: Int64, scale: Int) -> Float64:
    """DuckDB's DECIMAL -> DOUBLE as probed in tests/test_f190_decimal_to_double.py: up to +-2^53 float(raw) / 10^scale; past it the whole
    part and the fraction apart, float(raw / 10^scale) + float(raw % 10^scale) / 10^scale, the division truncating."""
    var f = p10(scale)
    if raw >= -(Int64(1) << 53) and raw <= (Int64(1) << 53):
        return Float64(raw) / f
    var p = Int64(1)
    for _ in range(scale):
        p *= 10
    var neg = raw < 0
    var m = UInt64(0) - UInt64(raw) if neg else UInt64(raw)
    var q = m // UInt64(p)
    var x = Float64(q) + Float64(m - q * UInt64(p)) / f
    return -x if neg else x


def raws() -> List[Int64]:
    """Values around 2^53, 2^62 and the ends of int64, both signs."""
    var two53 = Int64(1) << 53
    var two62 = Int64(1) << 62
    return [Int64(0), Int64(1), -Int64(7), two53 - 1, two53, two53 + 1, two53 + 2, two53 + 3, -(two53 + 1), -(two53 + 3), two53 * 3 + 1,
            two62 + 12345, -(two62 + 12345), Int64.MAX, Int64.MAX - 1, Int64.MIN + 1, Int64(1_000_000_000_000_000_001), Int64(7_654_321_098_765_432_109)]


def one_column(raw: List[Int64], scale: Int) raises -> SharedTable:
    var t = Table.empty()
    var id = t.add_column("id", INT)
    var v = t.add_column("v", DECIMAL if scale > 0 else INT, scale=scale)
    for i in range(len(raw)):
        t.append(id, Int64(i))
        t.append(v, raw[i])
    return t^.share()


def test_sql_cast_rounds_like_float_over_ten_to_the_scale() raises:
    var raw = raws()
    for scale in range(0, 19):
        var cat = Catalog()
        cat.add_table("t", one_column(raw, scale))
        var r = sql(cat, "SELECT cast(v as double) FROM t ORDER BY id")
        for i in range(len(raw)):
            var want = duckdb_double(raw[i], scale)
            var got = r.table.get_float(0, i)
            expect(got == want, "cast(" + String(raw[i]) + " at scale " + String(scale) + " as double) = " + String(want) + ", got " + String(got))


def test_builder_float_arithmetic_rounds() raises:
    var raw = raws()
    for scale in [0, 2, 9, 18]:
        for i in range(len(raw)):
            var one = Table.empty()
            var v = one.add_column("v", DECIMAL if scale > 0 else INT, scale=scale)
            one.append(v, raw[i])
            var r = Query(one^.share()).agg(max_of(col("v") * 1.0), min_of(col("v") + 0.0)).compute(CPU)
            var want = duckdb_double(raw[i], scale)
            expect(r.floats(0)[0] == want and r.floats(1)[0] == want, "builder: " + String(raw[i]) + " at scale " + String(scale) + " read as a double is " + String(want))


def test_a_big_decimal_sum_divided_is_a_double() raises:
    """The Q14 shape: 100.00 * sum(x) / sum(y), where 100.00 * sum(x) is a DECIMAL(4) whose raw integer is past 2^53."""
    var t = Table.empty()
    var x = t.add_column("x", DECIMAL, scale=2)
    var y = t.add_column("y", DECIMAL, scale=2)
    for _ in range(10):
        t.append(x, 3_000_000_000_001)
        t.append(y, 7_000_000_000_003)
    var cat = Catalog()
    cat.add_table("t", t^.share())
    var r = sql(cat, "SELECT 100.00 * sum(x) / sum(y), sum(x) / 3, 100.00 * sum(x) / 2.0 FROM t")
    var sx = Int64(30_000_000_000_010)                  # raw sums, scale 2
    var sy = Int64(70_000_000_000_030)
    var scaled = sx * 10000                             # 100.00 * sum(x): raw 3.0000000000010e17 at scale 4, past 2^53
    expect(scaled > (Int64(1) << 53), "the shape is past 2^53")
    var want0 = duckdb_double(scaled, 4) / duckdb_double(sy, 2)
    var want1 = duckdb_double(sx, 2) / duckdb_double(30, 1)
    var want2 = duckdb_double(scaled, 4) / duckdb_double(20, 1)
    expect(r.table.get_float(0, 0) == want0, "100.00 * sum(x) / sum(y) = " + String(want0) + ", got " + String(r.table.get_float(0, 0)))
    expect(r.table.get_float(1, 0) == want1, "sum(x) / 3 = " + String(want1) + ", got " + String(r.table.get_float(1, 0)))
    expect(r.table.get_float(2, 0) == want2, "100.00 * sum(x) / 2.0 = " + String(want2) + ", got " + String(r.table.get_float(2, 0)))


def test_the_gpu_keeps_declining_a_program_that_converts() raises:
    """The GPU kernel still refuses an integer past 2^53 meeting a double, so a program with a conversion never goes there: GPU says so,
    AUTO runs it on the CPU and answers what the CPU answers."""
    var raw = raws()
    var t = one_column(raw, 2)
    var want = Float64(0.0)
    for i in range(len(raw)):
        want = max(want, duckdb_double(raw[i], 2))
    var raised = False
    try:
        _ = Query(t).agg(max_of(col("v") * 1.0)).compute(GPU)
    except e:
        raised = String(e).find("int64 range") < 0              # a refusal that names the CPU (or the missing GPU), never an overflow
    expect(raised, "device GPU refuses a float conversion")
    var r = Query(t).agg(max_of(col("v") * 1.0)).compute(AUTO)
    expect(r.floats(0)[0] == want, "AUTO answers on the CPU: " + String(want) + ", got " + String(r.floats(0)[0]))


def test_a_sum_past_int64_still_raises() raises:
    var t = Table.empty()
    var v = t.add_column("v", DECIMAL, scale=2)
    t.append(v, Int64.MAX)
    t.append(v, Int64.MAX)
    var cat = Catalog()
    cat.add_table("t", t^.share())
    var raised = False
    try:
        _ = sql(cat, "SELECT sum(v) / 2.0 FROM t")
    except e:
        raised = String(e).find("passed the int64 range") >= 0
    expect(raised, "an int64 sum that really overflows raises")


def main() raises:
    test_sql_cast_rounds_like_float_over_ten_to_the_scale()
    test_builder_float_arithmetic_rounds()
    test_a_big_decimal_sum_divided_is_a_double()
    test_the_gpu_keeps_declining_a_program_that_converts()
    test_a_sum_past_int64_still_raises()
    print("ok")
