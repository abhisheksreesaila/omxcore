# F-168: window functions from the builder (`Window`, `Query.window`, `compute_windows`): the same answers as the SQL text over a table, a Frame
# (edits) and a Parquet scan, the CPU fallback, and the refusals.
from omxcore import (Table, SharedTable, Frame, Query, col, lit, sum, count, mean, min_of, max_of, median, Window, row_number, rank, dense_rank, percent_rank, ntile,
                     lag, lead, first_value, last_value, nth_value, CURRENT_ROW, UNBOUNDED_PRECEDING, UNBOUNDED_FOLLOWING, compute_windows, window_sql, CPU, GPU, AUTO,
                     INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, write_parquet, scan_parquet, ZSTD)


def fail(msg: String) raises:
    raise Error(msg)


def cell(t: Table, c: Int, r: Int) raises -> String:
    if t.is_null(c, r):
        return "NULL"
    var k = t.kind(c).value
    var v = t.get(c, r)
    if k == TEXT.value:
        return t.text_of(c, Int(v))
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


def orders() raises -> SharedTable:
    var t = Table.empty()
    var region = t.add_column("region", TEXT)
    var qty = t.add_column("qty", INT)
    var price = t.add_column("price", DECIMAL, scale=2)
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
    return t^.share()


def fails_with(what: String, message: String, got: String) raises:
    if message not in got:
        fail(what + " failed with\n  " + got + "\nwanted a message with\n  " + message)


def main() raises:
    var orders_t = orders()
    var by_region = Window().partition_by(col("region")).order_by(col("qty"))

    # running totals and ranks, the rows `where` keeps, the source's columns first and then one column per function
    var q = Query(orders_t).where(col("qty") >= 0).window(row_number().over(by_region.copy()).alias("rn"), sum(col("qty")).over(by_region.copy()).alias("run"),
                                                          rank().over(Window().order_by(col("qty"), descending=True)))
    var r = compute_windows(q)
    check(rows_of(r.table), "West 3 12.50 2 5 6\neast 1 4.00 1 1 8\nWest 5 20.25 3 10 4\nNorth 4 9.99 1 4 5\neast 7 30.00 2 8 2\nWest 2 -1.50 1 2 7\nNULL 6 7.00 1 6 3\neast 9 123.45 3 17 1\n",
          "running totals")
    check(r.table.name(3) + "|" + r.table.name(4) + "|" + r.table.name(5), "rn|run|w2", "names")
    check(r.fallback_reason, "window functions run on the CPU", "AUTO falls back to the CPU")
    check(compute_windows(q, CPU).fallback_reason, "", "CPU asked for")

    # moving average (DECIMAL in, FLOAT out), lag with a default, last / nth value over the whole partition, ntile
    var moving = Window().order_by(col("qty"), nulls_first=True).rows(-1, CURRENT_ROW)
    var all_rows = Window().order_by(col("qty")).rows(UNBOUNDED_PRECEDING, UNBOUNDED_FOLLOWING)
    var wq = Query(orders_t).window(mean(col("price")).over(moving.copy()).alias("avg2"), lag(col("qty"), 1, lit(-9)).over(Window().order_by(col("qty"))).alias("prev"),
                                    last_value(col("qty")).over(all_rows.copy()).alias("last"), nth_value(col("qty"), 2).over(all_rows.copy()).alias("second"),
                                    ntile(3).over(Window().order_by(col("qty"))).alias("tile"))
    r = compute_windows(wq)
    check(rows_of(r.table),
          "West 3 12.50 5.5 2 NULL 1 1\neast 1 4.00 4.5 -2 NULL 1 1\nWest 5 20.25 15.12 4 NULL 1 2\nNorth 4 9.99 11.245 3 NULL 1 2\neast 7 30.00 18.5 6 NULL 1 3\n"
          "West 2 -1.50 1.25 1 NULL 1 1\nNULL 6 7.00 13.625 5 NULL 1 2\nNorth NULL 0.00 0.0 9 NULL 1 3\neast 9 123.45 76.725 7 NULL 1 3\nWest -2 5.00 2.5 -9 NULL 1 1\n",
          "moving average, lag, last, nth, ntile")

    # the SQL the builder writes
    check(window_sql(Query(orders_t).where((col("qty") > 2) & (col("price") < 10)).window(sum(col("qty")).over(Window().partition_by(col("region")).order_by(col("qty"), True).rows(-2, 0)))),
          "SELECT *, sum(\"qty\") OVER (PARTITION BY \"region\" ORDER BY \"qty\" DESC NULLS LAST ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS \"w0\" "
          "FROM __query WHERE ((\"qty\" > 2) AND (\"price\" < 10))", "the SQL text")

    # a Frame: the edits are seen; the builder equals the SQL text over the same frame
    var f = Frame(orders_t)
    f.set_text(f.column("qty"), 0, "100")
    _ = f.append(["East", "1", "1.00"])
    f.remove(1)
    var fr = compute_windows(Query(f).window(sum(col("qty")).over(by_region.copy()).alias("run"), lag(col("qty")).over(by_region.copy()).alias("prev")))
    check(rows_of(fr.table), "West 100 12.50 105 5\nWest 5 20.25 5 2\nNorth 4 9.99 4 NULL\neast 7 30.00 7 NULL\nWest 2 -1.50 0 -2\nNULL 6 7.00 6 NULL\n"
          "North NULL 0.00 4 4\neast 9 123.45 16 7\nWest -2 5.00 -2 NULL\nEast 1 1.00 1 NULL\n", "a frame")

    # a streamed Parquet file gives the table's answer
    write_parquet("window-builder-orders.parquet", orders_t, "", ZSTD, 8)
    var sr = compute_windows(Query(scan_parquet("window-builder-orders.parquet")).window(sum(col("qty")).over(by_region.copy()).alias("run"), dense_rank().over(by_region.copy())))
    var tr = compute_windows(Query(orders_t).window(sum(col("qty")).over(by_region.copy()).alias("run"), dense_rank().over(by_region.copy())))
    check(rows_of(sr.table), rows_of(tr.table), "the scan equals the table")

    # refusals by name
    try:
        _ = compute_windows(q, GPU)
        fail("GPU should have been refused")
    except e:
        fails_with("GPU", "window functions run on the CPU only", String(e))
    try:
        _ = compute_windows(Query(orders_t).group_by(col("qty"), 10).agg(count()).window(rank().over(Window().order_by(col("qty")))))
        fail("a grouped query should have been refused")
    except e:
        fails_with("a grouped query", "takes no window functions here", String(e))
    try:
        _ = compute_windows(Query(orders_t).window(rank()))
        fail("a function with no window should have been refused")
    except e:
        fails_with("no window", "has no window", String(e))
    try:
        _ = compute_windows(Query(orders_t).window(sum(col("qty")).over(Window().order_by(col("qty")).range(-2, 0))))
        fail("RANGE with an offset should have been refused")
    except e:
        fails_with("RANGE with an offset", "`RANGE` with an offset", String(e))
    try:
        _ = median(col("qty")).over(Window())
        fail("median over a window should have been refused")
    except e:
        fails_with("median", "not supported as a window function", String(e))
    print("ok")
