# F-159: queries made of several statements from Mojo with no Python: WITH, derived tables, catalog views, subqueries (scalar, IN, EXISTS, correlated),
# UNION / INTERSECT / EXCEPT, QUALIFY, filtering on a window result. The expected rows are DuckDB's (generated once on the same ten rows). Every query runs over a table,
# over a Frame whose last edit removed an appended row, and over a streamed Parquet file, and the three answers are the same; tests/test_mojo_compound.py compares
# many more queries with DuckDB on generated data and on TPC-H.
from omxcore import (Table, SharedTable, Frame, Catalog, PlannerOptions, sql, sql_to_parquet, SqlResult, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, write_parquet,
                     scan_parquet, ZSTD, parse_query, query_files)


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


def catalog_of(source: Int) raises -> Catalog:
    """0: a table, 1: a Frame whose last edit removed an appended row, 2: a streamed Parquet file (2 row groups). The view big_orders reads it."""
    var cat = Catalog()
    if source == 0:
        cat.add_table("orders", orders())
    elif source == 1:
        var f = Frame(orders())
        _ = f.append(["East", "99", "1.00", "2020-01-01"])
        f.remove(10)
        cat.add_frame("orders", f)
    else:
        write_parquet("sqlquery-test-orders.parquet", orders(), "", ZSTD, 8)
        cat.add_scan("orders", scan_parquet("sqlquery-test-orders.parquet"))
    cat.view("big_orders", "SELECT region, qty FROM orders WHERE qty >= 5")
    return cat^


def add(mut qs: List[String], mut ws: List[String], q: String, w: String):
    qs.append(q)
    ws.append(w)


def main() raises:
    var queries = List[String]()
    var wants = List[String]()

    add(queries, wants, "WITH a AS (SELECT region, qty FROM orders WHERE qty > 0) SELECT region, sum(qty) FROM a GROUP BY region ORDER BY 2 DESC", "east 17\nWest 10\nNULL 6\nNorth 4\n")
    add(queries, wants, "SELECT region, qty, r FROM (SELECT region, qty, rank() OVER (PARTITION BY region ORDER BY qty DESC) AS r FROM orders) WHERE r = 1 ORDER BY qty",
        "North 4 1\nWest 5 1\nNULL 6 1\neast 9 1\n")
    add(queries, wants, "SELECT region, qty FROM orders o WHERE qty > (SELECT avg(qty) FROM orders i WHERE i.region = o.region) ORDER BY qty", "West 3\nWest 5\neast 7\neast 9\n")
    add(queries, wants, "SELECT qty FROM orders o WHERE NOT EXISTS (SELECT 1 FROM orders i WHERE i.region = o.region AND i.qty > o.qty) ORDER BY qty", "4\n5\n6\n9\nNULL\n")
    add(queries, wants, "SELECT qty FROM orders WHERE region IN (SELECT region FROM orders WHERE qty > 6) ORDER BY qty", "1\n7\n9\n")
    add(queries, wants, "SELECT qty FROM orders WHERE qty > 4 UNION ALL SELECT qty FROM orders WHERE qty < 2 ORDER BY 1", "-2\n1\n5\n6\n7\n9\n")
    add(queries, wants, "SELECT qty FROM orders WHERE qty > 2 INTERSECT SELECT qty FROM orders WHERE qty < 7 ORDER BY 1", "3\n4\n5\n6\n")
    add(queries, wants, "SELECT qty FROM orders WHERE qty > 2 EXCEPT SELECT qty FROM orders WHERE qty > 6 ORDER BY 1", "3\n4\n5\n6\n")
    add(queries, wants, "SELECT qty FROM orders WHERE qty > 2 UNION SELECT qty FROM orders WHERE qty < 7 ORDER BY 1", "-2\n1\n2\n3\n4\n5\n6\n7\n9\n")
    add(queries, wants, "SELECT count(*), count(DISTINCT region) FROM (SELECT region FROM orders UNION ALL SELECT region FROM orders)", "20 3\n")
    add(queries, wants, "SELECT region, qty FROM orders QUALIFY row_number() OVER (PARTITION BY region ORDER BY qty DESC) = 1 ORDER BY qty", "North 4\nWest 5\nNULL 6\neast 9\n")
    add(queries, wants, "SELECT sum(qty) FROM big_orders", "27\n")
    add(queries, wants, "SELECT count(*) FROM big_orders a JOIN big_orders b ON a.qty = b.qty", "4\n")
    add(queries, wants, "SELECT qty FROM orders o WHERE EXISTS (SELECT 1 FROM orders i WHERE i.region = o.region AND i.qty <> o.qty) AND qty > (SELECT min(qty) FROM orders i WHERE i.region = o.region) ORDER BY qty",
        "2\n3\n5\n7\n9\n")
    add(queries, wants, "SELECT price FROM orders WHERE price > (SELECT avg(price) FROM orders) ORDER BY price", "30.00\n123.45\n")
    add(queries, wants, "SELECT region, (SELECT max(qty) FROM orders WHERE qty < 0) AS m FROM orders WHERE qty = 1", "east -2\n")
    add(queries, wants, "SELECT qty FROM orders WHERE qty > (SELECT max(qty) FROM orders WHERE qty > 100) ORDER BY qty", "")

    for source in range(3):
        var cat = catalog_of(source)
        for i in range(len(queries)):
            var r = sql(cat, queries[i])
            check(rows_of(r.table), wants[i], "source " + String(source) + ": " + queries[i])

    var cat = catalog_of(0)

    # the plan says what ran: a CTE read once is inlined, read twice is materialized once; a derived table is evaluated in place
    var r = sql(cat, "WITH a AS (SELECT qty FROM orders) SELECT x.qty FROM a x JOIN a y ON x.qty = y.qty")
    has_note(r, "step: CTE a: materialized once (2 reads, 10 rows)")
    r = sql(cat, "WITH a AS (SELECT qty FROM orders) SELECT qty FROM a WHERE qty > 5")
    has_note(r, "step: CTE a: inlined into the query that reads it (its filter and columns merged, nothing materialized)")
    r = sql(cat, "WITH a AS (SELECT qty, count(*) AS c FROM orders GROUP BY qty) SELECT qty FROM a WHERE c > 0")
    has_note(r, "step: CTE a: inlined into its one read (")
    r = sql(cat, "SELECT qty FROM (SELECT qty FROM orders) z WHERE qty > 5")
    has_note(r, "step: derived table z: inlined into the query that reads it")
    r = sql(cat, "SELECT qty FROM (SELECT qty, count(*) AS c FROM orders GROUP BY qty) z WHERE c > 0")
    has_note(r, "step: derived table z: evaluated in place (")
    r = sql(cat, "EXPLAIN WITH a AS (SELECT qty FROM orders) SELECT x.qty FROM a x JOIN a y ON x.qty = y.qty")
    if "CTE a: materialized once (2 reads, 10 rows)" not in rows_of(r.table):
        fail("EXPLAIN does not show the CTE:\n" + rows_of(r.table))
    r = sql(cat, "SELECT qty FROM big_orders")
    has_note(r, "step: view big_orders: evaluated in place (")

    # a scalar subquery's types: DECIMAL, DATE, TEXT, and an empty one is a NULL of its type
    r = sql(cat, "SELECT qty FROM orders WHERE price = (SELECT max(price) FROM orders) AND day = (SELECT max(day) FROM orders WHERE qty > 8)")
    check(rows_of(r.table), "9\n", "decimal and date scalars")
    r = sql(cat, "SELECT qty FROM orders WHERE region = (SELECT min(region) FROM orders WHERE qty > 8)")
    check(rows_of(r.table), "1\n7\n9\n", "a text scalar")
    r = sql(cat, "SELECT (SELECT max(price) FROM orders WHERE qty > 100) AS p FROM orders WHERE qty = 1")
    check(rows_of(r.table) + String(r.table.kind(0)) + String(r.table.scale(0)), "NULL\nDECIMAL2", "an empty decimal scalar is a NULL of its type")
    r = sql(cat, "SELECT (SELECT max(day) FROM orders WHERE qty > 100) AS d FROM orders WHERE qty = 1")
    check(rows_of(r.table) + String(r.table.kind(0)), "NULL\nDATE", "an empty date scalar is a NULL of its type")

    # the caller's catalog gains no name from a compound query
    var before = cat.names()
    _ = sql(cat, "WITH a AS (SELECT qty FROM orders) SELECT x.qty FROM a x JOIN a y ON x.qty = y.qty WHERE x.qty IN (SELECT qty FROM (SELECT qty, count(*) AS c FROM orders GROUP BY qty) z)")
    var after = cat.names()
    if len(before) != len(after):
        fail("the catalog had " + String(len(before)) + " sources and has " + String(len(after)) + " after a compound query")

    # a view reads the views before it, and a source and a view cannot share a name
    cat.view("bigger", "SELECT qty FROM big_orders WHERE qty > 6")
    r = sql(cat, "SELECT qty FROM bigger ORDER BY qty")
    check(rows_of(r.table), "7\n9\n", "a view over a view")
    try:
        cat.view("orders", "SELECT 1 FROM orders")
        fail("a view named like a source should be refused")
    except e:
        if "is a source of this catalog" not in String(e):
            fail(String(e))

    # the answer of a compound query goes to a Parquet file like any other
    var opts = PlannerOptions()
    opts.sort_run_rows = 4
    opts.spill_dir = "."
    var w = sql_to_parquet(cat, "WITH a AS (SELECT qty, price FROM orders ORDER BY price) SELECT qty FROM a WHERE qty > 2 ORDER BY qty DESC", "sqlquery-test-out.parquet", opts)
    check(String(w.rows_written), "6", "rows written")
    cat.add_scan("written", scan_parquet("sqlquery-test-out.parquet"))
    r = sql(cat, "SELECT qty FROM written")
    check(rows_of(r.table), "9\n7\n6\n5\n4\n3\n", "the file")

    # the files a query names, in every statement of it
    var q = parse_query("WITH a AS (SELECT x FROM 'a.parquet') SELECT y FROM 'b.csv' WHERE y IN (SELECT x FROM 'a.parquet' UNION SELECT z FROM 'c.parquet')")
    var files = query_files(q)
    var all_files = String()
    for f in files:
        all_files += f + ","
    for want in ["a.parquet,", "b.csv,", "c.parquet,"]:
        if want not in all_files:
            fail("files: " + all_files + " lacks " + want)
    check(String(len(files)), "3", "files, each once")

    # unsupported forms raise by name
    fails_with(cat, "WITH RECURSIVE a AS (SELECT 1) SELECT * FROM a", "`WITH RECURSIVE` is not supported yet")
    fails_with(cat, "SELECT * FROM orders, LATERAL (SELECT 1) x", "`LATERAL` is not supported yet")
    fails_with(cat, "SELECT qty FROM orders WHERE qty = (SELECT qty FROM orders)", "the scalar subquery returned 10 rows: a scalar subquery must return at most one row")
    fails_with(cat, "SELECT qty FROM orders o WHERE qty NOT IN (SELECT qty FROM orders i WHERE i.region = o.region)", "a correlated NOT IN (SELECT ...) is not supported yet")
    fails_with(cat, "SELECT qty FROM orders UNION ALL SELECT qty, price FROM orders", "the operands of the set operation have 1 and 2 columns")
    fails_with(cat, "SELECT qty FROM orders UNION ALL SELECT region FROM orders", "cannot be combined: column 1 is INT in one and TEXT in the other")
    fails_with(cat, "SELECT qty FROM orders INTERSECT ALL SELECT qty FROM orders", "`INTERSECT ALL` is not supported yet")
    print("ok")
