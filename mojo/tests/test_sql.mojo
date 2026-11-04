# F-150: SQL aggregate queries over a table, a Frame and a Parquet scan, from Mojo with no Python: exact results, the
# planner's notes (which path ran), EXPLAIN and the error messages. (tests/test_mojo_sql.py compares many more queries with
# DuckDB on generated data.)
from std.sys import argv
from omxcore import (Table, SharedTable, Frame, Catalog, PlannerOptions, GROUP_HASHED, GROUP_DENSE, BUILD_LEFT, BUILD_RIGHT, sql, sql_with, SqlResult, INT, DECIMAL, TEXT,
                     DATE, BOOL, FLOAT, write_parquet, scan_parquet, ZSTD, Planner, RulePlanner, PhysicalPlan, Stats, BoundQuery)


def fail(msg: String) raises:
    raise Error(msg)


def cell(t: Table, c: Int, r: Int) raises -> String:
    """A cell as text: NULL, a number, a decimal at its scale, a word, true / false."""
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


def orders() raises -> SharedTable:
    var t = Table.empty()
    var region = t.add_column("region", TEXT)
    var qty = t.add_column("qty", INT)
    var price = t.add_column("price", DECIMAL, scale=2)
    var day = t.add_column("day", DATE)
    var flag = t.add_column("flag", BOOL)
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
        if i % 4 == 3:
            t.append_null(flag)
        else:
            t.append(flag, Int64(i % 2))
    return t^.share()


struct AlwaysHashed(Planner, Copyable, Movable):
    def __init__(out self):
        pass

    def plan(self, q: BoundQuery, st: Stats, opts: PlannerOptions) raises -> PhysicalPlan:
        var o = opts.copy()
        o.group = GROUP_HASHED
        var p = RulePlanner().plan(q, st, o)
        p.notes.append("planner: AlwaysHashed")
        return p^


def main() raises:
    var cat = Catalog()
    cat.add_table("orders", orders())

    # groups, aggregates, the NULL group, ORDER BY with the owner's text order (ignoring case)
    var r = sql(cat, "SELECT region, count(*), count(qty), sum(qty), sum(price), min(qty), max(day) FROM orders GROUP BY region ORDER BY region")
    check(rows_of(r.table),
          "east 3 3 17 157.45 1 18080\nNorth 2 1 4 9.99 4 18070\nWest 4 4 8 36.25 -2 18090\nNULL 1 1 6 7.00 6 18060\n",
          "group by a text column")
    has_note(r, "group: dense, 1 key region 4 (with a NULL slot) = 4 slots")
    has_note(r, "ran: table orders: 1 chunk of 10 rows")
    check(r.table.name(0) + "|" + r.table.name(1) + "|" + r.table.name(2), "region|count_star()|count(qty)", "column names")

    # three-valued logic: the unknown row is not kept; NOT of unknown is unknown; IS NULL sees it
    r = sql(cat, "SELECT count(*) FROM orders WHERE qty > 2 OR flag")
    check(rows_of(r.table), "9\n", "OR with unknown")
    r = sql(cat, "SELECT count(*) FROM orders WHERE NOT (qty > 2)")
    check(rows_of(r.table), "3\n", "NOT with unknown")
    r = sql(cat, "SELECT count(*) FROM orders WHERE (qty > 2) IS NULL")
    check(rows_of(r.table), "1\n", "IS NULL of a comparison")

    # nothing matches: one row for a global aggregate, SUM / MIN / AVG of nothing are NULL, no row for a group
    r = sql(cat, "SELECT count(*), sum(qty), min(price), avg(price) FROM orders WHERE qty > 100")
    check(rows_of(r.table), "0 NULL NULL NULL\n", "a global aggregate over no row")
    r = sql(cat, "SELECT region, count(*) FROM orders WHERE qty > 100 GROUP BY region")
    check(rows_of(r.table), "", "groups over no row")

    # AVG is a FLOAT, `/` and expressions over aggregates; HAVING; LIMIT / OFFSET
    r = sql(cat, "SELECT region, avg(qty), sum(price) / count(*) AS unit FROM orders GROUP BY region HAVING count(*) >= 2 ORDER BY 2 DESC")
    check(rows_of(r.table), "east 5.666666666666667 52.48333333333333\nNorth 4.0 4.995\nWest 2.0 9.0625\n", "avg and /")
    check(r.table.name(2), "unit", "an alias")
    r = sql(cat, "SELECT qty, count(*) FROM orders GROUP BY qty ORDER BY qty DESC NULLS FIRST LIMIT 3 OFFSET 1")
    check(rows_of(r.table), "9 1\n7 1\n6 1\n", "ORDER BY DESC NULLS FIRST, LIMIT and OFFSET")
    has_note(r, "order: ")

    # decimals: scales aligned, // and % truncate toward zero, x / 0 is NULL
    r = sql(cat, "SELECT sum(price * 2), sum(price + 1), max(price - 0.5) FROM orders")
    check(rows_of(r.table), "421.38 220.69 122.95\n", "decimal arithmetic")
    r = sql(cat, "SELECT sum(qty // 2), sum(qty % 3), count(qty // (qty - 3)) FROM orders")
    check(rows_of(r.table), "15 5 8\n", "integer division and remainder")

    # text predicates: a word the column never held matches nothing; ordered comparison ignores case
    r = sql(cat, "SELECT count(*) FROM orders WHERE region = 'nowhere' OR region IN ('West', 'WEST')")
    check(rows_of(r.table), "4\n", "IN with a missing word")
    r = sql(cat, "SELECT count(*) FROM orders WHERE region >= 'north' AND region < 'west'")
    check(rows_of(r.table), "4\n", "ordered text comparison")

    # the strategy can be forced: hashed groups give the same rows
    var hashed = PlannerOptions()
    hashed.group = GROUP_HASHED
    r = sql(cat, "SELECT region, count(*), sum(price), count(DISTINCT qty), median(qty) FROM orders GROUP BY region ORDER BY region", hashed)
    check(rows_of(r.table), "east 3 157.45 3 7.0\nNorth 2 9.99 1 4.0\nWest 4 36.25 4 2.5\nNULL 1 7.00 1 6.0\n", "hashed groups, holistic aggregates")
    has_note(r, "group: hashed on 2 key words: forced by the options")
    has_note(r, "ran: holistic second pass")
    var dense = PlannerOptions()
    dense.group = GROUP_DENSE
    r = sql(cat, "SELECT region, count(*), sum(price), count(DISTINCT qty), median(qty) FROM orders GROUP BY region ORDER BY region", dense)
    check(rows_of(r.table), "east 3 157.45 3 7.0\nNorth 2 9.99 1 4.0\nWest 4 36.25 4 2.5\nNULL 1 7.00 1 6.0\n", "dense groups, holistic aggregates")

    # EXPLAIN
    r = sql(cat, "EXPLAIN SELECT region, sum(price) FROM orders WHERE qty > 1 GROUP BY region")
    check(rows_of(r.table),
          "Logical plan\n  Project region, sum(price)\n    Aggregate keys=[region] aggs=[sum(price)]\n      Filter (WHERE) qty > 1\n        Scan orders\n"
          "Physical plan\n  source: orders (table), 10 rows\n  group: dense, 1 key region 4 (with a NULL slot) = 4 slots: the domain of 4 slots is small\n"
          "  columns read: region, qty, price (3 of 5)\n  kernel: 19 instructions a row (1 slot)\n  device: CPU (SQL runs on the CPU)\n",
          "EXPLAIN")

    # a frame: the edits are seen (a changed row, an appended row, a removed row)
    var f = Frame(orders())
    f.set_text(f.column("qty"), 0, "100")
    _ = f.append(["East", "1", "1.00", "2020-01-01", "true"])
    f.remove(1)
    cat.add_frame("edited", f)
    r = sql(cat, "SELECT region, count(*), sum(qty) FROM edited GROUP BY region ORDER BY region")
    check(rows_of(r.table), "East 1 1\neast 2 16\nNorth 2 4\nWest 4 105\nNULL 1 6\n", "frame")
    has_note(r, "ran: frame edited:")

    # a streamed file: the same answers, one chunk per row group
    write_parquet("build/mojo-tests/sql-test-orders.parquet", orders(), "", ZSTD, 8)
    cat.add_scan("filed", scan_parquet("build/mojo-tests/sql-test-orders.parquet"))
    var a = sql(cat, "SELECT region, count(*), sum(price), min(day) FROM orders GROUP BY region ORDER BY region")
    var b = sql(cat, "SELECT region, count(*), sum(price), min(day) FROM filed GROUP BY region ORDER BY region")
    check(rows_of(b.table), rows_of(a.table), "the scan equals the table")
    has_note(b, "ran: scan filed: 2 of 2 row groups read")

    # a join over a frame and over a streamed file: the frame's edits are seen, the file's row groups are read; text keys match by bytes ('east' is not 'East')
    var rt = Table.empty()
    var r_reg = rt.add_column("region", TEXT)
    var r_w = rt.add_column("w", INT)
    var r_names: List[String] = ["West", "North", "East"]
    for i in range(3):
        rt.append_text(r_reg, r_names[i])
        rt.append(r_w, Int64(10 * (i + 1)))
    var reg_t = rt^.share()
    cat.add_table("reg", reg_t)
    r = sql(cat, "SELECT r.w, count(*), sum(e.qty) FROM edited e JOIN reg r ON e.region = r.region GROUP BY r.w ORDER BY r.w")
    check(rows_of(r.table), "10 4 105\n20 2 4\n30 1 1\n", "a join over a frame")
    has_note(r, "join build side: r (3 rows")
    has_note(r, "join path: fused: ")
    var streamed_frame = False                                  # F-158: the frame's chunks probe one at a time (its edits seen), never read into a table
    for n in r.notes:
        streamed_frame = streamed_frame or "chunk by chunk" in n
    if not streamed_frame:
        fail("a Frame on the probe side streams")
    var fb = Catalog()                                          # ... and equals the join over the baked frame (edit + append + removed row)
    fb.add_table("edited", f.bake().share())
    fb.add_table("reg", reg_t)
    var baked = sql(fb, "SELECT r.w, count(*), sum(e.qty) FROM edited e JOIN reg r ON e.region = r.region GROUP BY r.w ORDER BY r.w")
    check(rows_of(r.table), rows_of(baked.table), "a join over a frame equals the join over bake()")
    var jt = sql(cat, "SELECT r.w, count(*), sum(e.qty) FROM orders e JOIN reg r ON e.region = r.region GROUP BY r.w ORDER BY r.w")
    var js = sql(cat, "SELECT r.w, count(*), sum(e.qty) FROM filed e JOIN reg r ON e.region = r.region GROUP BY r.w ORDER BY r.w")
    check(rows_of(js.table), rows_of(jt.table), "a join over a scan equals the join over the table")
    check(rows_of(js.table), "10 4 8\n20 2 4\n", "the join over a scan")

    # any Planner can plan: this one always asks for hashed groups
    r = sql_with(cat, "SELECT region, count(*) FROM orders GROUP BY region ORDER BY region", AlwaysHashed())
    check(rows_of(r.table), "east 3\nNorth 2\nWest 4\nNULL 1\n", "a custom planner")
    has_note(r, "planner: AlwaysHashed")
    has_note(r, "group: hashed on 2 key words: forced by the options")

    # a frame's formula columns are columns: the builder's rules (floor division, x // 0 is 0, min / max) and exact decimals
    var ff = Frame(orders())
    _ = ff.formula("net = price - qty")                         # DECIMAL(2) minus INT: qty brought to cents
    _ = ff.formula("half = qty // 2")
    _ = ff.formula("lo = min(qty, 3)")
    ff.set_text(ff.column("qty"), 0, "100")
    cat.add_frame("sheet", ff)
    r = sql(cat, "SELECT region, sum(net), sum(half), max(lo), count(net) FROM sheet GROUP BY region ORDER BY region")
    check(rows_of(r.table), "east 140.45 7 3 3\nNorth 5.99 2 3 1\nWest -68.75 52 3 4\nNULL 1.00 3 3 1\n", "formula columns")
    r = sql(cat, "SELECT sum(net) FROM sheet WHERE half > 1 AND net IS NOT NULL")
    check(rows_of(r.table), "72.19\n", "a formula in WHERE")

    # F-158: a Frame on the BUILD side: its formula columns, edits (qty 3 -> 100), an append and a removed row are seen, equal to the join over bake()
    var bq = "SELECT count(*), sum(o.price), sum(s.net), sum(s.half) FROM orders o JOIN sheet s ON o.qty = s.qty"
    var bopts = PlannerOptions()
    bopts.build_side = BUILD_RIGHT                              # the Frame builds the hash table
    r = sql(cat, bq, bopts)
    has_note(r, "join build side: s (")
    var sb = Catalog()
    sb.add_table("orders", orders())
    sb.add_table("sheet", ff.bake().share())
    var sbr = sql(sb, "SELECT count(*), sum(o.price), sum(s.price - s.qty), sum(s.qty // 2) FROM orders o JOIN sheet s ON o.qty = s.qty", bopts)   # (bake() keeps no formulas: they are written out)
    check(rows_of(r.table), rows_of(sbr.table), "a join with a Frame build side (formulas) equals the join over bake()")
    check(rows_of(r.table), "8 198.19 166.19 14\n", "a Frame build side with formula columns")
    try:                                                        # a formula column as a join key is refused by name
        _ = sql(cat, "SELECT count(*) FROM orders o JOIN sheet s ON o.qty = s.half", bopts)
        fail("a formula as a join key should raise")
    except e:
        check(String(e), "SQL error at line 1, column 53: a join key that is a frame's formula column (`half`) is not supported yet: join on the stored columns (a formula column can be read, filtered and aggregated after the join)", "formula join key")
    r = sql(cat, "SELECT s.* FROM orders o JOIN sheet s ON o.qty = s.qty WHERE o.qty = 5", bopts)             # s.* holds the formula columns too
    check(String(r.table.columns()), "8", "s.* of a Frame in a join: 5 stored + 3 formula columns")
    bopts.build_side = BUILD_LEFT                               # ... and as the probe side the same answer
    r = sql(cat, bq, bopts)
    check(rows_of(r.table), rows_of(sbr.table), "the same join with the Frame on the probe side")

    # errors say what and where
    try:
        _ = sql(cat, "SELECT sum(qtty) FROM orders")
        fail("a misspelled column should raise")
    except e:
        check(String(e), "SQL error at line 1, column 12: column `qtty` not found in table `orders` (did you mean `qty`?)", "misspelled column")
    try:
        _ = sql(cat, "SELECT region FROM orders RIGHT JOIN other ON 1")
        fail("a RIGHT JOIN should raise")
    except e:
        check(String(e), "SQL error at line 1, column 27: `RIGHT JOIN` is not supported yet", "join")
    print("ok")
