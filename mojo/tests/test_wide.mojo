# F-192: totals past int64. The 128-bit helpers (text, parsing, multiplication, the DOUBLE cast DuckDB does), then SQL from Mojo with no Python: a SUM
# whose total passes int64 (global, grouped, a NULL-only group, negatives), what HAVING / ORDER BY / the select list do with it, a scalar subquery
# that is itself past int64, and the places that cannot hold such a total yet saying so. Expected values were computed in Python; the DOUBLE casts
# are DuckDB 1.5.1's (tests/test_f192_wide_sums.py compares whole queries with DuckDB; this file needs no Python at run time).
from omxcore import Table, Catalog, Frame, sql, INT, DECIMAL
from omxcore.wide import wide_text, wide_make, wide_lo, wide_hi, parse_wide, wide_mul, wide_to_float, fits_int64, within_decimal


def fail(msg: String) raises:
    raise Error(msg)


def check(got: String, want: String, what: String) raises:
    if got != want:
        fail(what + "\n--- got ---\n" + got + "\n--- wanted ---\n" + want)


def check_float(digits: String, scale: Int, want: Float64) raises:
    var w = parse_wide(digits)
    if not w.ok:
        fail("cannot read " + digits)
    var got = wide_to_float(w.value, scale)
    if got != want:
        fail("the DOUBLE of " + digits + " at scale " + String(scale) + " is " + String(got) + ", DuckDB's is " + String(want))


def cell(t: Table, c: Int, r: Int) raises -> String:
    if t.is_null(c, r):
        return "NULL"
    if t.is_wide(c):
        return wide_text(wide_make(t.get(c, r), t.get_high(c, r)), t.scale(c))
    if t.kind(c) == DECIMAL:
        return wide_text(Int128(t.get(c, r)), t.scale(c))
    return String(t.get(c, r))


def rows_of(t: Table) raises -> String:
    var out = String()
    for r in range(t.rows()):
        for c in range(t.columns()):
            out += (" " if c > 0 else "") + cell(t, c, r)
        out += "\n"
    return out^


def fails_with(mut cat: Catalog, query: String, message: String) raises:
    try:
        _ = sql(cat, query)
    except e:
        if message not in String(e):
            fail("`" + query + "` failed with\n  " + String(e) + "\nwanted a message with\n  " + message)
        return
    fail("`" + query + "` should have failed")


def helpers() raises:
    # text: 128-bit values with the point in place, both signs, the extremes
    var m3 = Int128(-3) << Int128(100)
    check(wide_text(m3, 0), "-3802951800684688204490109616128", "text, scale 0")
    check(wide_text(m3, 5), "-38029518006846882044901096.16128", "text, scale 5")
    check(wide_text(Int128(5), 2), "0.05", "text, a fraction below one")
    check(wide_text(Int128(0), 2), "0.00", "text, zero")
    check(wide_text((Int128(1) << Int128(127)) - Int128(1), 18), "170141183460469231731.687303715884105727", "text, the largest 128-bit value")
    # the two words and back
    var big = (Int128(7) << Int128(90)) + Int128(12345)
    check(String(wide_lo(big)) + " " + String(wide_hi(big)), "12345 " + String(Int64(7) << 26), "words")
    if wide_make(wide_lo(m3), wide_hi(m3)) != m3:
        fail("the words of a negative number do not make it again")
    if not fits_int64(Int128(Int64.MAX)) or fits_int64(Int128(Int64.MAX) + Int128(1)) or fits_int64(Int128(Int64.MIN) - Int128(1)):
        fail("fits_int64")
    # reading: up to 38 digits, 18 places
    var p = parse_wide("-123456789012345678901234567890.123456")
    if not p.ok or p.scale != 6:
        fail("parse_wide of a 36 digit decimal")
    check(wide_text(p.value, p.scale), "-123456789012345678901234567890.123456", "parse and print")
    if parse_wide("1000000000000000000000000000000000000000").ok:                       # 40 digits
        fail("40 digits are past DECIMAL(38)")
    if parse_wide("1.0000000000000000001").ok:                                         # 19 places
        fail("19 decimal places are past a column's scale")
    # multiplication is exact up to 38 digits, an error past them
    check(wide_text(wide_mul(Int128(10000000000000000000), Int128(1000000000000000000)), 0), "10000000000000000000000000000000000000", "10^19 * 10^18")
    check(wide_text(wide_mul(Int128(-4294967296), Int128(4294967296)), 0), "-18446744073709551616", "negative product")
    try:
        _ = wide_mul(Int128(100000000000000000000), Int128(10000000000000000000))      # 10^20 * 10^19 = 10^39
        fail("10^39 should not fit")
    except e:
        if "38 digits" not in String(e):
            fail("wrong message: " + String(e))
    try:
        _ = within_decimal(Int128(10000000000000000000) * Int128(10000000000000000000))     # 10^38 itself is one digit too many
        fail("10^38 should not fit")
    except e:
        if "38 digits" not in String(e):
            fail("wrong message: " + String(e))
    # the cast to DOUBLE (DuckDB 1.5.1's values)
    check_float("2098206791939208999857975122728244029", 0, 2.098206791939209e+36)
    check_float("-9992202491720960686413560282700905388", 0, -9.992202491720961e+36)
    check_float("2528836280092140138817910268287397981", 2, 2.52883628009214e+34)
    check_float("-7030240824550432356229226129568669232", 2, -7.030240824550433e+34)
    check_float("2976838678420647517255524174302584370", 5, 2.9768386784206475e+31)
    check_float("-5286607336075297273700771391229527778", 5, -5.286607336075297e+31)
    check_float("2033514896705622078741250060234047260", 10, 2.0335148967056222e+26)
    check_float("-4131807498659831318244549131333755779", 10, -4.131807498659831e+26)
    check_float("1247436639261327261599956977199404298", 18, 1.2474366392613274e+18)
    check_float("-2768560034571581511205326095771604279", 18, -2.7685600345715814e+18)
    check_float("-1", 2, -0.01)
    check_float("123456789012345678901234567890", 2, 1.2345678901234569e+27)
    check_float("-559077926033271240080182095887350247", 0, -5.590779260332713e+35)       # (a negative at scale 0 rounds its two's-complement words)


def table_of() raises -> Table:
    var t = Table.empty()
    var g = t.add_column("g", INT)
    var v = t.add_column("v", DECIMAL, scale=2)
    var b = t.add_column("b", INT)
    var gs: List[Int] = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
    var vs: List[Int] = [9000000000000000000, 9000000000000000000, 9000000000000000000, 5, -9000000000000000000, -9000000000000000000, -9000000000000000000, 1, 2, 0]
    for i in range(10):
        t.append(g, Int64(gs[i]))
        if i == 9:
            t.append_null(Int(1))
            t.append_null(Int(2))
        else:
            t.append(1, Int64(vs[i]))
            t.append(2, Int64(vs[i]))
    return t^


def sums() raises:
    var cat = Catalog()
    cat.add_table("t", table_of()^.share())
    # global and grouped: past int64 in two groups (a positive and a negative total), small in one, NULL in the last
    var r = sql(cat, "SELECT sum(v), sum(b), count(*) FROM t")
    check(rows_of(r.table), "0.08 8 10\n", "the global totals are small again")
    r = sql(cat, "SELECT g, sum(v), sum(b) FROM t GROUP BY g ORDER BY g")
    check(rows_of(r.table), "1 270000000000000000.05 27000000000000000005\n2 -270000000000000000.00 -27000000000000000000\n3 0.03 3\n4 NULL NULL\n", "grouped totals")
    r = sql(cat, "SELECT sum(v) FROM t WHERE g = 1")
    check(rows_of(r.table), "270000000000000000.05\n", "a global total past int64")
    # what a query does with a wide total
    r = sql(cat, "SELECT g, sum(b) AS s FROM t GROUP BY g ORDER BY s DESC")
    check(rows_of(r.table), "1 27000000000000000005\n3 3\n2 -27000000000000000000\n4 NULL\n", "ORDER BY a wide total (NULLs last)")
    r = sql(cat, "SELECT g FROM t GROUP BY g HAVING sum(b) > 20000000000000000000 ORDER BY g")
    check(rows_of(r.table), "1\n", "HAVING a wide total above a wide constant")
    r = sql(cat, "SELECT g, sum(v) * 2, sum(v) - sum(v), -sum(v) FROM t GROUP BY g ORDER BY g")
    check(rows_of(r.table), "1 540000000000000000.10 0.00 -270000000000000000.05\n2 -540000000000000000.00 0.00 270000000000000000.00\n3 0.06 0.00 -0.03\n4 NULL NULL NULL\n",
          "arithmetic on totals")
    # a scalar subquery that is past int64 itself, and a scale of 12 that is
    r = sql(cat, "SELECT g FROM t GROUP BY g HAVING sum(b) < (SELECT sum(b) * 10000000000 FROM t WHERE g = 1) ORDER BY g")
    check(rows_of(r.table), "1\n2\n3\n", "HAVING over a scalar subquery that is past int64 itself")
    r = sql(cat, "SELECT g, sum(v) * 0.0001000000 FROM t WHERE g <= 2 GROUP BY g ORDER BY g")
    check(rows_of(r.table), "1 27000000000000.000005000000\n2 -27000000000000.000000000000\n", "a total times a scale-10 constant")
    # a window sum
    r = sql(cat, "SELECT g, sum(b) OVER (PARTITION BY g) AS s FROM t WHERE g <= 2 ORDER BY g, s")
    check(rows_of(r.table), "1 27000000000000000005\n1 27000000000000000005\n1 27000000000000000005\n1 27000000000000000005\n"
                            "2 -27000000000000000000\n2 -27000000000000000000\n2 -27000000000000000000\n", "a window sum past int64")
    # what cannot hold a wide total yet says why (the int64 message)
    fails_with(cat, "SELECT g, s FROM (SELECT g, sum(b) AS s FROM t GROUP BY g) WHERE s > 1", "passed the int64 range")
    fails_with(cat, "SELECT sum(sum(b)) OVER () FROM t GROUP BY g", "passed the int64 range")
    fails_with(cat, "SELECT sum(b) OVER (PARTITION BY g) + 1 FROM t WHERE g <= 2", "passed the int64 range")
    fails_with(cat, "SELECT b * 4000000000 FROM t", "passed the int64 range")                  # a single value's arithmetic still raises
    # a Frame over such a column cannot hold the upper words: it says so when it is read, queried or baked, and Table.check sees a short `high`
    var wide = sql(cat, "SELECT g, sum(b) AS s FROM t GROUP BY g ORDER BY g")
    var held = wide.take_table().share()
    var f = Frame(held)
    var said = False
    try:
        _ = f.bake()
    except e:
        said = "totals past int64" in String(e)
    if not said:
        fail("a Frame baked a wide column")
    said = False
    try:
        var cat3 = Catalog()
        cat3.add_frame("f", f)
    except e:
        said = "totals past int64" in String(e)
    if not said:
        fail("a Frame over a wide column was queried")
    var sr = sql(cat, "SELECT g, sum(b) AS s FROM t GROUP BY g ORDER BY g")
    var short = sr.take_table()
    short.cols[1].high.pop()
    said = False
    try:
        short.check()
    except e:
        said = "upper words" in String(e)
    if not said:
        fail("Table.check missed a short `high`")
    # a total cast to a narrower integer type raises like DuckDB's Conversion Error, and a total that fits casts
    fails_with(cat, "SELECT sum(b)::BIGINT FROM t WHERE g <= 2 GROUP BY g HAVING g = 1", "out of range")
    r = sql(cat, "SELECT sum(b)::BIGINT FROM t WHERE g = 3")
    check(rows_of(r.table), "3\n", "a total that fits casts to BIGINT")
    print("ok")


def main() raises:
    helpers()
    sums()
