# F-160: named views, preview with the LIMIT pushed down, the result cache (counted by runs, not by time), and COPY (query) TO 'file.parquet'.
# The expected rows are DuckDB's on the same ten rows (create view ... as ...; the edits as UPDATEs).
from omxcore import (Table, SharedTable, Frame, Catalog, Query, col, lit, sql, preview, SqlResult, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, write_parquet, read_parquet,
                     scan_parquet, ZSTD)


def fail(msg: String) raises:
    raise Error(msg)


def cell(t: Table, c: Int, r: Int) raises -> String:
    if t.is_null(c, r):
        return "NULL"
    var k = t.kind(c).value
    var v = t.get(c, r)
    if k == TEXT.value:
        return t.text_of(c, Int(v))
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


def check_int(got: Int, want: Int, what: String) raises:
    if got != want:
        fail(what + ": got " + String(got) + ", wanted " + String(want))


def has_note(r: SqlResult, text: String) raises:
    for n in r.notes:
        if text in n:
            return
    var all = String()
    for n in r.notes:
        all += n + "\n"
    fail("no note has `" + text + "`; the notes are:\n" + all)


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


def dims() raises -> SharedTable:
    var t = Table.empty()
    var region = t.add_column("region", TEXT)
    var rank = t.add_column("rank", INT)
    t.append_text(region, "West")
    t.append(rank, 1)
    t.append_text(region, "east")
    t.append(rank, 2)
    t.append_text(region, "North")
    t.append(rank, 3)
    return t^.share()


def views_read_views() raises:
    var cat = Catalog()
    cat.add_table("orders", orders())
    cat.view("big", "SELECT region, qty FROM orders WHERE qty >= 5")
    cat.view("by_region", "SELECT region, sum(qty) AS q FROM big GROUP BY region")
    var r = sql(cat, "SELECT region, q FROM by_region ORDER BY q DESC")
    check(rows_of(r.table), "east 16\nNULL 6\nWest 5\n", "a view reads the view before it")
    # a view added later does not change the earlier one; replacing a view's text changes what it reads
    cat.view("big", "SELECT region, qty FROM orders WHERE qty >= 7")
    r = sql(cat, "SELECT region, q FROM by_region ORDER BY q DESC")
    check(rows_of(r.table), "east 16\n", "a view with new text runs the new text")


def view_from_a_query() raises:
    var cat = Catalog()
    cat.add_table("orders", orders())
    cat.view("q_big", Query(orders()).where(col("qty") > lit(2)))
    var r = sql(cat, "SELECT sum(qty) FROM q_big")
    check(rows_of(r.table), "34\n", "a view made from a Query keeps the rows its where keeps")
    cat.view("after", "SELECT region, qty FROM q_big WHERE qty > 5 ORDER BY qty")
    r = sql(cat, "SELECT region, qty FROM after")
    check(rows_of(r.table), "NULL 6\neast 7\neast 9\n", "a view reads a view made from a Query")
    # a grouped Query is refused by name
    var refused = False
    try:
        cat.view("g", Query(orders()).group_by(col("qty"), groups=10))
    except e:
        refused = "write the query in SQL" in String(e)
    if not refused:
        fail("a Query with a group_by cannot be a view")


def preview_pushes_the_limit() raises:
    var cat = Catalog()
    cat.add_table("orders", orders())
    cat.view("ranked", "SELECT region, qty FROM orders ORDER BY qty DESC")
    var r = preview(cat, "ranked", 3)
    check(rows_of(r.table), "east 9\neast 7\nNULL 6\n", "the first 3 rows of an ordered view")
    has_note(r, "preview ranked: LIMIT 3 pushed into the view's query")
    r = preview(cat, "ranked", 0)
    check_int(r.table.rows(), 0, "0 rows")
    r = preview(cat, "ranked")
    check_int(r.table.rows(), 10, "the default 100 rows lists all 10")
    # a view with a LIMIT of its own: the preview is the smaller of the two
    cat.view("two", "SELECT region, qty FROM orders ORDER BY qty DESC LIMIT 2;")
    r = preview(cat, "two", 5)
    check(rows_of(r.table), "east 9\neast 7\n", "a view's own LIMIT stays")
    r = preview(cat, "two", 1)
    check(rows_of(r.table), "east 9\n", "a smaller preview of a limited view")
    # no ORDER BY: the first rows the view lists
    cat.view("pos", "SELECT region, qty FROM orders WHERE qty > 0")
    r = preview(cat, "pos", 4)
    check(rows_of(r.table), "West 3\neast 1\nWest 5\nNorth 4\n", "a preview of a filter")
    # a view that reads a view: the upstream one comes whole (and is kept), the preview's own query is cut
    cat.view("top", "SELECT region, qty FROM pos ORDER BY qty DESC")
    r = preview(cat, "top", 2)
    check(rows_of(r.table), "east 9\neast 7\n", "a preview reads the views before it")
    check_int(cat.view_runs(), 1, "the upstream view ran once")
    r = preview(cat, "top", 3)
    check_int(cat.view_runs(), 1, "the upstream view is read from the cache")
    check_int(cat.view_hits(), 1, "one read of it was a hit")
    var refused = False
    try:
        _ = preview(cat, "orders", 3)
    except e:
        refused = "is not a view" in String(e)
    if not refused:
        fail("preview of a name that is not a view raises")


def chain(mut cat: Catalog, f: Frame) raises:
    cat.add_frame("orders", f)
    cat.add_table("dims", dims())
    cat.view("v1", "SELECT region, qty FROM orders WHERE qty > 0")
    cat.view("v2", "SELECT region, sum(qty) AS q FROM v1 GROUP BY region")
    cat.view("v3", "SELECT region, rank FROM dims")
    cat.view("v4", "SELECT v2.q, v3.rank FROM v2 JOIN v3 ON v2.region = v3.region")


def an_edit_runs_only_the_steps_below_it() raises:
    var cat = Catalog()
    var f = Frame(orders())
    chain(cat, f)
    var r = sql(cat, "SELECT q FROM v2 ORDER BY q")
    check(rows_of(r.table), "4\n6\n10\n17\n", "v2 before the edit")
    check_int(cat.view_runs(), 2, "v2 and v1 ran")
    r = sql(cat, "SELECT q, rank FROM v4 ORDER BY q")
    check(rows_of(r.table), "4 3\n10 1\n17 2\n", "v4 joins v2 and v3")
    check_int(cat.view_runs(), 4, "v4 and v3 ran; v2 was read from the cache")
    r = sql(cat, "SELECT q, rank FROM v4 ORDER BY q")
    has_note(r, "step: view v4: read from the cache (3 rows)")
    check_int(cat.view_runs(), 4, "a repeated read runs nothing")

    # edit the frame: the qty of the row 1 (1 -> 10) and of row 5 (2 -> -5), then give the catalog the edited frame
    f.set(1, 1, 10)
    f.set(1, 5, -5)
    cat.add_frame("orders", f)
    r = sql(cat, "SELECT rank FROM v3 ORDER BY rank")
    check(rows_of(r.table), "1\n2\n3\n", "v3 reads no table of the frame")
    check_int(cat.view_runs(), 4, "v3 does not read the frame: not run again")
    r = sql(cat, "SELECT q FROM v2 ORDER BY q")
    check(rows_of(r.table), "4\n6\n8\n26\n", "v2 after the edit")
    check_int(cat.view_runs(), 6, "v1 and v2 ran again")
    r = sql(cat, "SELECT q, rank FROM v4 ORDER BY q")
    check(rows_of(r.table), "4 3\n8 1\n26 2\n", "v4 after the edit")
    check_int(cat.view_runs(), 7, "only v4 ran again (v2 and v3 were kept)")
    r = sql(cat, "SELECT q, rank FROM v4 ORDER BY q")
    check_int(cat.view_runs(), 7, "and now it is kept")

    # the same frame added again with no edits keeps every result
    cat.add_frame("orders", f)
    r = sql(cat, "SELECT q, rank FROM v4 ORDER BY q")
    check_int(cat.view_runs(), 7, "the same edits, the same results")

    # undo returns the table to what it was, but the frame's edit counter has moved: results are recomputed, never stale
    f.undo()
    f.undo()
    cat.add_frame("orders", f)
    r = sql(cat, "SELECT q FROM v2 ORDER BY q")
    check(rows_of(r.table), "4\n6\n10\n17\n", "v2 after the undo")


def the_cache_is_bounded_by_bytes() raises:
    var cat = Catalog()
    cat.add_table("orders", orders())
    cat.view("a", "SELECT qty FROM orders")
    cat.view("b", "SELECT qty FROM orders WHERE qty > 5")
    _ = sql(cat, "SELECT count(*) FROM a")
    _ = sql(cat, "SELECT count(*) FROM b")
    check_int(cat.view_runs(), 2, "both ran")
    # a: 10 rows x 9 bytes = 90; b: 3 rows x 9 = 27. Room for 100: the least recently used (a) goes
    cat.set_cache_budget(100)
    _ = sql(cat, "SELECT count(*) FROM b")
    check_int(cat.view_runs(), 2, "b is kept")
    _ = sql(cat, "SELECT count(*) FROM a")
    check_int(cat.view_runs(), 3, "a was dropped and runs again; keeping it (117 bytes in all) drops b, the one read before it")
    _ = sql(cat, "SELECT count(*) FROM a")
    check_int(cat.view_runs(), 3, "a is kept")
    _ = sql(cat, "SELECT count(*) FROM b")
    check_int(cat.view_runs(), 4, "b was dropped")
    # a budget of 0 keeps nothing
    cat.set_cache_budget(0)
    _ = sql(cat, "SELECT count(*) FROM b")
    _ = sql(cat, "SELECT count(*) FROM b")
    check_int(cat.view_runs(), 6, "nothing is kept: both reads run")
    var r = sql(cat, "SELECT count(*) FROM b")
    check(rows_of(r.table), "3\n", "and the answer is the same")


def a_file_the_view_names_is_an_input() raises:
    var cat = Catalog()
    cat.add_table("orders", orders())
    cat.view("v", "SELECT qty FROM orders WHERE region <> 'views-test-marker.txt' AND qty > 6")
    _ = sql(cat, "SELECT count(*) FROM v")
    _ = sql(cat, "SELECT count(*) FROM v")
    check_int(cat.view_runs(), 1, "kept while the file does not exist")
    var out = open("views-test-marker.txt", "w")
    out.write("one")
    out.close()
    _ = sql(cat, "SELECT count(*) FROM v")
    check_int(cat.view_runs(), 2, "the file now exists: the key changed")
    _ = sql(cat, "SELECT count(*) FROM v")
    check_int(cat.view_runs(), 2, "and is kept again")
    out = open("views-test-marker.txt", "w")
    out.write("one, longer")
    out.close()
    _ = sql(cat, "SELECT count(*) FROM v")
    check_int(cat.view_runs(), 3, "the file's size changed: run again")


def copy_writes_parquet() raises:
    var cat = Catalog()
    cat.add_table("orders", orders())
    var r = sql(cat, "COPY (SELECT region, qty, price FROM orders WHERE qty > 2 ORDER BY qty) TO 'views-copy.parquet'")
    check(rows_of(r.table), "6\n", "COPY answers the rows it wrote")
    check(String(r.table.name(0)), "Count", "in a column called Count")
    var back = read_parquet("views-copy.parquet")
    check(rows_of(back.table), "West 3 12.50\nNorth 4 9.99\nWest 5 20.25\nNULL 6 7.00\neast 7 30.00\neast 9 123.45\n", "the file holds the query's rows")
    # options, a trailing semicolon, parentheses and quotes inside the query, a view as the source
    cat.view("big", "SELECT region, qty FROM orders WHERE qty >= 5")
    r = sql(cat, "copy (select region, qty from big where region <> ')' order by qty desc) to 'views-copy2.parquet' (format parquet, compression snappy, row_group_size 8);")
    check(rows_of(r.table), "3\n", "a lower case COPY with options reading a view")
    back = read_parquet("views-copy2.parquet")
    check(rows_of(back.table), "east 9\neast 7\nWest 5\n", "options and a view")
    # it is atomic and refuses what it cannot write
    var msgs: List[String] = ["COPY (SELECT qty FROM orders) TO 'x.csv'", "COPY (SELECT qty FROM orders) TO 'x.parquet' (FORMAT CSV)",
                              "COPY (SELECT qty FROM orders) TO 'x.parquet' (COMPRESSION lz4)", "COPY SELECT qty FROM orders TO 'x.parquet'",
                              "COPY (SELECT qty FROM orders) 'x.parquet'", "COPY (SELECT qty FROM orders TO 'x.parquet'",
                              "COPY (SELECT qty FROM orders) TO 'x.parquet' (DELIMITER ',')", "COPY (SELECT qty FROM orders) TO 'x.parquet' (ROW_GROUP_SIZE 7)"]
    var wants: List[String] = ["Parquet only", "Parquet only", "zstd, snappy or uncompressed", "COPY takes a query in parentheses", "needs TO", "never closed",
                               "not supported", "multiple of 8"]
    for i in range(len(msgs)):
        var failed = False
        try:
            _ = sql(cat, msgs[i])
        except e:
            failed = wants[i] in String(e)
            if not failed:
                fail("`" + msgs[i] + "` failed with\n  " + String(e) + "\nwanted a message with `" + wants[i] + "`")
        if not failed:
            fail("`" + msgs[i] + "` should have failed")


def main() raises:
    views_read_views()
    view_from_a_query()
    preview_pushes_the_limit()
    an_edit_runs_only_the_steps_below_it()
    the_cache_is_bounded_by_bytes()
    a_file_the_view_names_is_an_input()
    copy_writes_parquet()
    print("ok")
