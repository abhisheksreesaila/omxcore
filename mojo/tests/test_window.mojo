# F-168: window functions in SQL over a table, a Frame and a Parquet scan, from Mojo with no Python: exact results (the expected values are DuckDB's,
# except where the engine's text order differs from DuckDB's binary one), the notes, EXPLAIN, the errors. (tests/test_mojo_window.py compares
# many more queries with DuckDB on generated data; mojo/tests/test_window_builder.mojo covers the builder.)
from omxcore import (Table, SharedTable, Frame, Catalog, PlannerOptions, sql, SqlResult, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, write_parquet, scan_parquet,
                     ZSTD)


def fail(msg: String) raises:
    raise Error(msg)


def cell(t: Table, c: Int, r: Int) raises -> String:
    if t.is_null(c, r):
        return "NULL"
    var k = t.kind(c).value
    var v = t.get(c, r)
    if k == TEXT.value:
        return t.text_of(c, Int(v))
    if k == BOOL.value:
        return "true" if v != 0 else "false"
    if k == FLOAT.value:
        return String(Float64(from_bits=UInt64(v)))
    if k == DECIMAL.value:
        var scale = t.scale(c)
        var neg = v < 0
        var digits = String(UInt64(v) if not neg else UInt64(0) - UInt64(v))
        while digits.byte_length() <= scale:
            digits = "0" + digits
        var cut = digits.byte_length() - scale
        return ("-" if neg else "") + String(digits[byte=:cut]) + "." + String(digits[byte=cut:])
    return String(v)


def rows_of(t: Table) raises -> String:
    var out = String()
    for r in range(t.rows()):
        for c in range(t.columns()):
            out += (" " if c > 0 else "") + cell(t, c, r)
        out += "\n"
    return out^


def check(got: String, want: String, what: String) raises:
    if got != want:
        fail(what + "\n--- got ---\n" + got + "--- wanted ---\n" + want)


def has_note(r: SqlResult, prefix: String) raises:
    for n in r.notes:
        if n.startswith(prefix):
            return
    var all = String()
    for n in r.notes:
        all += n + "\n"
    fail("no note starts with `" + prefix + "`; the notes are:\n" + all)


def fails_with(mut cat: Catalog, query: String, message: String) raises:
    try:
        _ = sql(cat, query)
    except e:
        if message not in String(e):
            fail("`" + query + "` failed with\n  " + String(e) + "\nwanted a message with\n  " + message)
        return
    fail("`" + query + "` should have failed")


def orders() raises -> SharedTable:
    var t = Table.empty()
    var region = t.add_column("region", TEXT)
    var qty = t.add_column("qty", INT)
    var price = t.add_column("price", DECIMAL, scale=2)
    var day = t.add_column("day", DATE)
    var regions: List[String] = ["West", "east", "West", "North", "east", "West", "", "North", "east", "West"]
    var qtys: List[Int] = [3, 1, 5, 4, 7, 2, 6, 0, 9, -2]
    var prices: List[Int] = [1250, 400, 2025, 999, 3000, -150, 700, 0, 12345, 500]
    for i in range(10):
        if regions[i].byte_length() == 0:
            t.append_null(region)
        else:
            t.append_text(region, regions[i])
        if i == 7:
            t.append_null(qty)
        else:
            t.append(qty, Int64(qtys[i]))
        t.append(price, Int64(prices[i]))
        t.append(day, Int64(18000 + i * 10))
    return t^.share()


def big() raises -> SharedTable:
    """50,000 rows in 700 partitions of uneven size, with ties and NULLs in x."""
    var t = Table.empty()
    var g = t.add_column("g", INT)
    var x = t.add_column("x", INT)
    var id = t.add_column("id", INT)
    var seed = UInt64(12345)
    for i in range(50000):
        seed = seed * 6364136223846793005 + 1442695040888963407
        var r = Int((seed >> 33) % 1000)
        t.append(g, Int64((r * r) % 700))
        if r % 17 == 0:
            t.append_null(x)
        else:
            t.append(x, Int64(r % 97) - 40)
        t.append(id, Int64(i))
    return t^.share()


def main() raises:
    var cat = Catalog()
    cat.add_table("orders", orders())

    # running totals and ranks: the default frame (rows up to the last peer), NULLs last, a NULL group of its own
    var r = sql(cat, "SELECT region, qty, row_number() OVER (PARTITION BY region ORDER BY qty) AS rn, sum(qty) OVER (PARTITION BY region ORDER BY qty) AS run FROM orders ORDER BY day")
    check(rows_of(r.table), "West 3 3 3\neast 1 1 1\nWest 5 4 8\nNorth 4 1 4\neast 7 2 8\nWest 2 2 0\nNULL 6 1 6\nNorth NULL 2 4\neast 9 3 17\nWest -2 1 -2\n", "running totals")
    check(r.table.name(0) + "|" + r.table.name(2) + "|" + r.table.name(3), "region|rn|run", "names")
    check(String(r.table.kind(2)) + String(r.table.kind(3)), "INTINT", "kinds")
    has_note(r, "window: 2 functions in 1 sort over 10 rows")
    check(r.fallback_reason, "window functions run on the CPU", "the fallback reason")

    # lag, lead (offset, default), first_value, last_value over the whole partition
    r = sql(cat, "SELECT lag(qty) OVER (PARTITION BY region ORDER BY day) AS l, lead(qty, 2, -1) OVER (PARTITION BY region ORDER BY day) AS d, "
                 "first_value(qty) OVER (PARTITION BY region ORDER BY day) AS f, "
                 "last_value(qty) OVER (PARTITION BY region ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS la FROM orders")
    check(rows_of(r.table), "NULL 2 3 -2\nNULL 9 1 9\n3 -2 3 -2\nNULL -1 4 NULL\n1 -1 1 9\n5 -1 3 -2\nNULL -1 6 6\n4 -1 4 NULL\n7 -1 1 9\n2 -1 3 -2\n",
          "lag, lead, first_value, last_value")

    # moving averages and sums over ROWS frames (DECIMAL stays exact, AVG is a FLOAT), the frame cut at the table's edges
    r = sql(cat, "SELECT avg(price) OVER (ORDER BY day ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS a, sum(price) OVER (ORDER BY day ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS s, "
                 "min(qty) OVER (ORDER BY day ROWS 2 PRECEDING) AS m FROM orders")
    check(rows_of(r.table), "12.5 16.50 3\n8.25 36.75 1\n12.125 34.24 1\n15.12 60.24 1\n19.995 38.49 4\n14.25 35.50 2\n2.75 5.50 2\n3.5 130.45 2\n61.725 128.45 6\n64.225 128.45 -2\n",
          "moving frames")
    check(String(r.table.kind(0)) + String(r.table.kind(1)) + String(r.table.scale(1)), "FLOATDECIMAL2", "kinds and scale")

    # rank, dense_rank, percent_rank, cume_dist (peers share a rank; the engine's text order ignores case), ntile
    r = sql(cat, "SELECT rank() OVER (ORDER BY region) AS r, dense_rank() OVER (ORDER BY region) AS d, percent_rank() OVER (ORDER BY region) AS p, "
                 "cume_dist() OVER (ORDER BY region) AS c, ntile(3) OVER (ORDER BY day) AS n FROM orders")
    check(rows_of(r.table),
          "6 3 0.5555555555555556 0.9 1\n1 1 0.0 0.3 1\n6 3 0.5555555555555556 0.9 1\n4 2 0.3333333333333333 0.5 1\n1 1 0.0 0.3 2\n6 3 0.5555555555555556 0.9 2\n"
          "10 4 1.0 1.0 2\n4 2 0.3333333333333333 0.5 3\n1 1 0.0 0.3 3\n6 3 0.5555555555555556 0.9 3\n", "ranks")

    # a window over grouped results, and ORDER BY / LIMIT after the windows
    r = sql(cat, "SELECT region, sum(qty) AS s, rank() OVER (ORDER BY sum(qty) DESC) AS r, sum(sum(qty)) OVER (ORDER BY sum(qty) DESC ROWS UNBOUNDED PRECEDING) AS cum "
                 "FROM orders GROUP BY region ORDER BY r")
    check(rows_of(r.table), "east 17 1 17\nWest 8 2 25\nNULL 6 3 31\nNorth 4 4 35\n", "windows over groups")
    r = sql(cat, "SELECT qty, row_number() OVER (ORDER BY qty DESC NULLS FIRST, day) AS rn FROM orders ORDER BY rn DESC LIMIT 3")
    check(rows_of(r.table), "-2 10\n1 9\n2 8\n", "ORDER BY and LIMIT after the window")

    # WHERE runs first; nothing left gives no rows; an expression over windows
    r = sql(cat, "SELECT qty, rank() OVER (ORDER BY qty) AS r FROM orders WHERE qty > 100")
    check(rows_of(r.table), "", "no rows")
    r = sql(cat, "SELECT qty - lag(qty) OVER (ORDER BY day) AS delta, sum(qty) OVER () AS total FROM orders WHERE region = 'West' ORDER BY day")
    check(rows_of(r.table), "NULL 8\n2 8\n-3 8\n-4 8\n", "an expression over a window")

    # a Frame sees its edits (a changed row, a removed row, an appended row)
    var f = Frame(orders())
    f.set_text(f.column("qty"), 0, "100")
    _ = f.append(["East", "1", "1.00", "2020-01-01"])
    f.remove(1)
    cat.add_frame("edited", f)
    r = sql(cat, "SELECT region, qty, sum(qty) OVER (PARTITION BY region ORDER BY day) AS run FROM edited ORDER BY day")
    check(rows_of(r.table), "West 100 100\nWest 5 105\nNorth 4 4\neast 7 7\nWest 2 107\nNULL 6 6\nNorth NULL 4\neast 9 16\nWest -2 105\nEast 1 1\n", "a frame")

    # a streamed Parquet file: the columns the query needs are read, the answers equal the table's
    write_parquet("window-test-orders.parquet", orders(), "", ZSTD, 8)
    cat.add_scan("filed", scan_parquet("window-test-orders.parquet"))
    var a = sql(cat, "SELECT region, day, rank() OVER (PARTITION BY region ORDER BY qty) AS r, avg(price) OVER (ORDER BY day ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS a FROM orders ORDER BY day")
    var b = sql(cat, "SELECT region, day, rank() OVER (PARTITION BY region ORDER BY qty) AS r, avg(price) OVER (ORDER BY day ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS a FROM filed ORDER BY day")
    check(rows_of(b.table), rows_of(a.table), "the scan equals the table")
    has_note(b, "ran: scan filed: ")

    # unsupported forms raise by name
    fails_with(cat, "SELECT qty FROM orders WHERE rank() OVER (ORDER BY qty) = 1", "window functions are only allowed in the SELECT list and ORDER BY")
    fails_with(cat, "SELECT sum(qty) OVER (ORDER BY day RANGE BETWEEN 5 PRECEDING AND CURRENT ROW) FROM orders", "`RANGE` with an offset (5) is not supported yet")
    fails_with(cat, "SELECT sum(qty) OVER (ORDER BY day GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM orders", "a `GROUPS` frame is not supported yet")
    fails_with(cat, "SELECT median(qty) OVER () FROM orders", "`median` is not supported as a window function")
    fails_with(cat, "SELECT sum(region) OVER () FROM orders", "sum(...) OVER is not supported for TEXT values")
    fails_with(cat, "EXPLAIN ANALYZE SELECT rank() OVER (ORDER BY qty) FROM orders", "EXPLAIN ANALYZE of a query with window functions is not supported yet")

    # EXPLAIN shows the window step and runs nothing
    r = sql(cat, "EXPLAIN SELECT rank() OVER (PARTITION BY region ORDER BY qty) FROM orders")
    if not rows_of(r.table).startswith("Window functions (after the query below, on the CPU)"):
        fail("EXPLAIN did not start with the window step:\n" + rows_of(r.table))

    # a running sum is exact in 128 bits inside (adding the next row before the old one leaves passes int64), and raises when a frame's own sum does
    var t = Table.empty()
    var i = t.add_column("i", INT)
    var v = t.add_column("v", INT)
    var vals: List[Int] = [9000000000000000000, 9000000000000000000, -9000000000000000000, -9000000000000000000]
    for k in range(4):
        t.append(i, Int64(k))
        t.append(v, Int64(vals[k]))
    cat.add_table("wide", t^.share())
    r = sql(cat, "SELECT sum(v) OVER (ORDER BY i ROWS BETWEEN CURRENT ROW AND CURRENT ROW) AS s FROM wide")
    check(rows_of(r.table), "9000000000000000000\n9000000000000000000\n-9000000000000000000\n-9000000000000000000\n", "sums that fit")
    fails_with(cat, "SELECT sum(v) OVER (ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND 1 FOLLOWING) FROM wide", "a window sum passed the int64 range")

    # 1 thread and many give the same rows, FLOAT sums bit for bit
    cat.add_table("big", big())
    var one = PlannerOptions()
    one.threads = 1
    var many = PlannerOptions()
    many.threads = 8
    var bq = ("SELECT id, sum(x / 3.0) OVER (PARTITION BY g ORDER BY id ROWS BETWEEN 3 PRECEDING AND 2 FOLLOWING) AS f, avg(x) OVER (PARTITION BY g ORDER BY x) AS a, "
              "rank() OVER (PARTITION BY g ORDER BY x) AS r, lag(x, 2) OVER (PARTITION BY g ORDER BY id) AS l, min(x) OVER (PARTITION BY g ORDER BY id ROWS 5 PRECEDING) AS m FROM big ORDER BY id")
    var s1 = sql(cat, bq, one)
    var s8 = sql(cat, bq, many)
    check(rows_of(s8.table), rows_of(s1.table), "1 thread and 8 threads")
    if s1.table.rows() != 50000:
        fail("the big table lost rows")
    print("ok")
