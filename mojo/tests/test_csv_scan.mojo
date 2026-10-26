# F-178: a CSV file queried a block at a time (`scan_csv`), docs/omxcore-mojo-api.md section 2b. The oracle is the whole-file reader: for the same
# bytes and the same options, `scan_csv` must give exactly what `read_csv` gives (every cell, every TEXT code, the failed-row report), and a
# query / SQL / Frame over the scan exactly what the same call over the table gives, whatever the block boundaries cut through.
from std.os import makedirs
from omxcore import (Table, SharedTable, Query, QueryResult, col, lit, sum, count, mean, min_of, max_of, count_distinct,
                     INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, read_csv, CsvOptions, scan_csv, ParquetScan, read_parquet,
                     Frame, Catalog, sql, MORSELS_ON, MORSELS_OFF, AUTO)

comptime DIR = "mxframe_csv_scan_test/"


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


def write_file(name: String, content: String) raises -> String:
    makedirs(DIR, exist_ok=True)
    var f = open(DIR + name, "w")
    f.write(content)
    f.close()
    return DIR + name


def write_bytes(name: String, content: List[UInt8]) raises -> String:
    makedirs(DIR, exist_ok=True)
    var f = open(DIR + name, "w")
    f.write_bytes(content)
    f.close()
    return DIR + name


def cell(t: Table, c: Int, row: Int) -> String:
    """A cell as a user reads it back: null, a word, or the stored number."""
    if t.is_null(c, row):
        return "null"
    if t.kind(c) == TEXT:
        return t.text_of(c, Int(t.get(c, row)))
    return String(t.get(c, row))


def mixed_csv(n: Int) -> String:
    """id INT, region TEXT (some quoted, with a comma), note TEXT (quoted, some with a newline or doubled quotes, some empty), carrier INT
    (empty cells), revenue DECIMAL(2) (empty cells), price FLOAT (an exponent in the first row, so the whole column is FLOAT), day DATE, flag BOOL,
    cost INT. Lines end in \\r\\n on every 7th row."""
    var words: List[String] = ["West", "East", "North", "South", "Central", "\"North, East\"", "Nörth-West", "Far"]
    var out = String("id,region,note,carrier,revenue,price,day,flag,cost\n")
    var rng = Rng(11)
    for i in range(n):
        var line = String(i + 1) + "," + words[(i // 40 + rng.int(0, 2)) % 8] + ","
        var k = rng.int(0, 9)
        if k == 0:
            line += ""
        elif k == 1:
            line += "\"two\nlines " + String(rng.int(0, 5)) + "\""
        elif k == 2:
            line += "\"say \"\"hi\"\", " + String(rng.int(0, 3)) + "\""
        else:
            line += "n" + String(rng.int(0, 6))
        line += "," + ("" if i % 11 == 3 else String(rng.int(-1, 6)))
        line += "," + ("" if i % 13 == 1 else String(rng.int(-99999, 99999) // 100) + "." + String(rng.int(10, 99)))
        line += "," + ("1.5e3" if i == 0 else String(rng.int(0, 99)) + "." + String(rng.int(0, 99999)))
        line += ",2024-" + ("0" + String(rng.int(1, 9)) if rng.int(0, 1) == 0 else String(rng.int(10, 12))) + "-" + String(rng.int(10, 28))
        line += "," + ("true" if rng.int(0, 1) == 0 else "false")
        line += "," + String(rng.int(-5000, 5000))
        out += line + ("\r\n" if i % 7 == 0 else "\n")
    return out^


def same_result(a: QueryResult, b: QueryResult, label: String) raises:
    """Every field a caller can read."""
    expect(len(a.kinds) == len(b.kinds), label + ": same aggregates")
    expect(len(a.slots) == len(b.slots), label + ": same slot count " + String(len(a.slots)) + " vs " + String(len(b.slots)))
    for s in range(len(a.slots)):
        expect(len(a.slots[s]) == len(b.slots[s]), label + ": slot " + String(s) + " has the same groups")
        for g in range(len(a.slots[s])):
            expect(a.slots[s][g] == b.slots[s][g], label + ": slot " + String(s) + " group " + String(g) + " is " +
                   String(a.slots[s][g]) + ", wanted " + String(b.slots[s][g]))
    expect(len(a.counts) == len(b.counts), label + ": same groups")
    for g in range(len(a.counts)):
        expect(a.counts[g] == b.counts[g], label + ": count group " + String(g) + " is " + String(a.counts[g]) + ", wanted " + String(b.counts[g]))
        for i in range(len(a.kinds)):
            expect(a.present(i, g) == b.present(i, g), label + ": present(" + String(i) + ", " + String(g) + ")")


def same_table(a: Table, b: Table, label: String) raises:
    expect(a.columns() == b.columns() and a.rows() == b.rows(), label + ": shape " + String(a.rows()) + "x" + String(a.columns()) +
           " vs " + String(b.rows()) + "x" + String(b.columns()))
    for c in range(a.columns()):
        expect(a.name(c) == b.name(c) and a.kind(c) == b.kind(c) and a.scale(c) == b.scale(c), label + ": column " + String(c) + " is the same kind")
        for r in range(a.rows()):
            expect(cell(a, c, r) == cell(b, c, r), label + ": row " + String(r) + " column `" + a.name(c) + "` reads `" + cell(a, c, r) +
                   "`, wanted `" + cell(b, c, r) + "`")


def scan_table(scan: ParquetScan) raises -> SharedTable:
    """The whole scan as one table (row group by row group; TEXT words from the file-wide dictionary)."""
    var all = List[Int]()
    for c in range(scan.columns()):
        all.append(c)
    var t = Table.empty()
    for c in range(scan.columns()):
        _ = t.add_column(scan.name(c), scan.kind(c), scale=scan.scale(c))
        if scan.kind(c) == TEXT:
            for w in range(scan.word_count(c)):
                _ = t.add_word(c, scan.text_of(c, w))
    for g in range(scan.row_groups()):
        var part = scan.read_group(g, all)
        for r in range(scan.group_rows(g)):
            for c in range(scan.columns()):
                if part.is_null(c, r):
                    t.append_null(c)
                else:
                    t.append(c, part.get(c, r))
    return t^.share()


def check_same_as_whole(path: String, options: CsvOptions, label: String) raises:
    var whole = read_csv(path, options)
    var wt = whole.take_table().share()
    var scan = scan_csv(path, options)
    expect(scan.rows() == whole.rows, label + ": rows " + String(scan.rows()) + " vs " + String(whole.rows))
    same_table(scan_table(scan)._table[], wt._table[], label + " cells")
    # the import report: the same failed cells, in the same order, with the same words
    var rep = scan.csv_report()
    expect(rep.rows == whole.rows and len(rep.columns) == len(whole.columns), label + ": report shape")
    for c in range(len(whole.columns)):
        expect(rep.columns[c].empty == whole.columns[c].empty, label + ": empty cells of `" + whole.columns[c].name + "`")
        expect(rep.columns[c].failed_count == whole.columns[c].failed_count, label + ": failed cells of `" + whole.columns[c].name + "`")
        expect(len(rep.columns[c].failed) == len(whole.columns[c].failed), label + ": failed list of `" + whole.columns[c].name + "`")
        for k in range(len(whole.columns[c].failed)):
            expect(rep.columns[c].failed[k].message() == whole.columns[c].failed[k].message(), label + ": failed cell " + rep.columns[c].failed[k].message())
    expect(rep.ragged_count == whole.ragged_count, label + ": ragged rows")


def test_every_cell_equals_read_csv() raises:
    var path = write_file("mixed.csv", mixed_csv(3000))
    var shapes: List[Tuple[Int, Int]] = [(500, 1 << 20), (1 << 20, 4096), (37, 64), (1, 1), (100000, 1 << 30)]
    for s in shapes:
        var options = CsvOptions(group_rows=s[0], block_bytes=s[1])
        check_same_as_whole(path, options, "group_rows " + String(s[0]) + ", block_bytes " + String(s[1]))


def check_queries(path: String, options: CsvOptions, mode: Int, label: String) raises:
    var whole = read_csv(path, options)
    var mem = whole.take_table().share()
    var scan = scan_csv(path, options)
    var nreg = scan.word_count(scan.index("region"))
    var nnote = scan.word_count(scan.index("note"))
    expect(nreg == len(mem.dictionary(mem.index("region"))), label + ": the dictionary has every word")
    var far = mem.code_of(mem.index("region"), "Far")
    same_result(Query(scan).morsels(mode).block_rows(700).group_by(col("region"), groups=nreg)
                .agg(sum(col("revenue")), count(), count(col("revenue")), min_of(col("cost")), max_of(col("cost")), mean(col("revenue")), mean(col("cost"))).compute(),
                Query(mem).group_by(col("region"), groups=nreg)
                .agg(sum(col("revenue")), count(), count(col("revenue")), min_of(col("cost")), max_of(col("cost")), mean(col("revenue")), mean(col("cost"))).compute(), label + " text key")
    same_result(Query(scan).morsels(mode).block_rows(700).where(col("region") == lit(far)).group_by(col("carrier"), groups=7)
                .agg(sum(col("revenue")), count(), min_of(col("revenue"))).compute(),
                Query(mem).where(col("region") == lit(far)).group_by(col("carrier"), groups=7)
                .agg(sum(col("revenue")), count(), min_of(col("revenue"))).compute(), label + " filter on a word, nullable key")
    same_result(Query(scan).morsels(mode).block_rows(700).group_by(col("note"), groups=nnote).agg(count(), sum(col("cost"))).compute(),
                Query(mem).group_by(col("note"), groups=nnote).agg(count(), sum(col("cost"))).compute(), label + " nullable text key")
    same_result(Query(scan).morsels(mode).block_rows(700).agg(count(), sum(col("id")), min_of(col("day")), max_of(col("day")), count(col("flag"))).compute(),
                Query(mem).agg(count(), sum(col("id")), min_of(col("day")), max_of(col("day")), count(col("flag"))).compute(), label + " global")
    same_result(Query(scan).morsels(mode).block_rows(700).where(col("cost") > 99999).group_by(col("carrier"), groups=7).agg(sum(col("revenue")), max_of(col("cost"))).compute(),
                Query(mem).where(col("cost") > 99999).group_by(col("carrier"), groups=7).agg(sum(col("revenue")), max_of(col("cost"))).compute(), label + " nothing kept")
    same_result(Query(scan).morsels(mode).block_rows(700).group_by(col("flag"), groups=2).agg(count_distinct(col("region")), min_of(col("price")), max_of(col("price"))).compute(),
                Query(mem).group_by(col("flag"), groups=2).agg(count_distinct(col("region")), min_of(col("price")), max_of(col("price"))).compute(), label + " distinct, FLOAT min / max")
    # a float sum is folded per row group (api doc section 1): equal to the table's within rounding, not bit for bit
    var fa = Query(scan).morsels(mode).block_rows(700).group_by(col("flag"), groups=2).agg(sum(col("price"))).compute()
    var fb = Query(mem).group_by(col("flag"), groups=2).agg(sum(col("price"))).compute()
    for g in range(2):
        var x = Float64(from_bits=UInt64(fa.slots[0][g]))
        var y = Float64(from_bits=UInt64(fb.slots[0][g]))
        expect(abs(x - y) <= 1e-9 * abs(y), label + ": float sum of group " + String(g) + " is " + String(x) + ", wanted " + String(y))


def test_queries_equal_the_table() raises:
    var path = write_file("queries.csv", mixed_csv(4000))
    var shapes: List[Tuple[Int, Int]] = [(500, 1 << 20), (1 << 20, 4096), (37, 64), (100000, 1 << 30)]
    for s in shapes:
        var options = CsvOptions(group_rows=s[0], block_bytes=s[1])
        check_queries(path, options, MORSELS_OFF, "group_rows " + String(s[0]) + ", block_bytes " + String(s[1]))
        check_queries(path, options, MORSELS_ON, "morsels, group_rows " + String(s[0]) + ", block_bytes " + String(s[1]))


def test_sql_over_a_csv_scan() raises:
    var path = write_file("sql.csv", mixed_csv(2500))
    var options = CsvOptions(group_rows=400, block_bytes=8192)
    var whole = read_csv(path, options)
    var cat = Catalog()
    cat.add_table("t", whole.take_table().share())
    cat.add_scan("s", scan_csv(path, options))
    var queries: List[String] = [
        "SELECT region, count(*), sum(revenue), min(day), max(cost) FROM {} GROUP BY region ORDER BY region",
        "SELECT carrier, count(*) FROM {} WHERE region = 'Far' OR note LIKE 'n%' GROUP BY carrier ORDER BY carrier",
        "SELECT flag, count(DISTINCT note), median(cost) FROM {} GROUP BY flag ORDER BY flag",
        "SELECT id, region, note, revenue FROM {} WHERE cost > 4900 ORDER BY id LIMIT 20",
        "SELECT count(*), sum(id), avg(cost) FROM {}",
        "SELECT region, note FROM {} WHERE note LIKE '%lines%' ORDER BY id LIMIT 5",
    ]
    for q in queries:
        var want = sql(cat, String(q).replace("{}", "t"))
        var got = sql(cat, String(q).replace("{}", "s"))
        same_table(got.table, want.table, "sql `" + String(q) + "`")
        var streamed = False
        for note in got.notes:
            streamed = streamed or note.startswith("ran: scan s")
        expect(streamed, "`" + String(q) + "` ran as a scan: " + String(got.notes.__len__()) + " notes")


def same_file(path: String, want: Table, label: String) raises:
    """The saved Parquet file read back equals the baked table, cell by cell."""
    var rp = read_parquet(path)
    var got = rp.take_table().share()
    same_table(got._table[], want, label)


def test_frame_edits_and_save_over_a_csv_scan() raises:
    var path = write_file("frame.csv", mixed_csv(2000))
    var options = CsvOptions(group_rows=300, block_bytes=16384)
    var whole = read_csv(path, options)
    var a = Frame(whole.take_table().share())
    var scan = scan_csv(path, options)
    var b = Frame(scan)
    b.set_cache_limit(3 * 8 * 300)                           # a handful of column chunks: the cache evicts all the time
    var rng = Rng(5)
    var rv = a.column("revenue")
    var rc = a.column("region")
    var nc = a.column("note")
    var cs = a.column("cost")
    var ca = a.column("carrier")
    for _ in range(300):
        var row = rng.int(0, a.rows() - 1)
        while not a.alive(row):
            row = rng.int(0, a.rows() - 1)
        var what = rng.int(0, 9)
        if what <= 1:
            var v = Int64(rng.int(-100000, 300000))
            a.set(rv, row, v)
            b.set(rv, row, v)
        elif what == 2:
            a.set_null(rv, row)
            b.set_null(rv, row)
        elif what == 3:
            var w = "Fresh" + String(rng.int(0, 3)) if rng.int(0, 1) == 0 else String("West")
            a.set_text(rc, row, w)
            b.set_text(rc, row, w)
        elif what == 4:
            var memo = "memo, " + String(rng.int(0, 5))
            a.set_text(nc, row, memo)
            b.set_text(nc, row, memo)
        elif what == 5:
            var vals: List[String] = ["9000001", "East", "n2", "3", "12.50", "7.25", "2019-05-03", "true", "17"]
            _ = a.append(vals)
            _ = b.append(vals)
        elif what == 6 or what == 7:
            a.remove(row)
            b.remove(row)
        elif what == 8:
            a.set_null(ca, row)
            b.set_null(ca, row)
        else:
            a.set_text(cs, row, "4242")
            b.set_text(cs, row, "4242")
    expect(a.rows() == b.rows(), "rows after the edits")
    for _ in range(400):
        var row = rng.int(0, a.rows() - 1)
        for c in range(a.columns()):
            expect(a.alive(row) == b.alive(row) and a.is_null(c, row) == b.is_null(c, row) and a.get(c, row) == b.get(c, row) and
                   a.edited(c, row) == b.edited(c, row), "cell " + String(c) + ", " + String(row) + " of the edited frame")
    var n = a.word_count(rc)
    expect(n == b.word_count(rc), "words after the edits")
    same_result(Query(b).group_by(col("region"), groups=n).agg(sum(col("revenue")), count(), max_of(col("cost"))).compute(),
                Query(a).group_by(col("region"), groups=n).agg(sum(col("revenue")), count(), max_of(col("cost"))).compute(), "query over the edited scan frame")
    for c in range(a.columns()):
        var pa = a.permutation(c)
        var pb = b.permutation(c)
        expect(len(pa) == len(pb), "order of `" + a.name(c) + "`")
        for i in range(len(pa)):
            expect(pa[i] == pb[i], "order of `" + a.name(c) + "` at " + String(i))
    b.save(DIR + "frame.saved.parquet", "{}")
    same_file(DIR + "frame.saved.parquet", a.bake(), "the saved file")
    # the scan itself is untouched, and the file can be opened again
    expect(scan.rows() == 2000, "the scan still has the file's rows")


def same_in_shapes(name: String, content: String, base: CsvOptions) raises:
    """`content` read whole and scanned in several block shapes gives the same cells and the same report."""
    var path = write_file(name, content)
    var shapes: List[Tuple[Int, Int]] = [(1, 1), (2, 7), (3, 64), (1000, 1 << 20)]
    for s in shapes:
        var o = CsvOptions(base.delimiter, base.has_header, base.max_columns, base.infer_rows, group_rows=s[0], block_bytes=s[1])
        check_same_as_whole(path, o, name + " group_rows " + String(s[0]) + ", block_bytes " + String(s[1]))


def test_the_failed_row_report() raises:
    var path = write_file("bad.csv", "id,qty,price\n1,5,1.50\n2,6,2.50\n3,7,3.50\n4,x9,4.00\n5,8,abc\n6,,6.0\n7,1.5,7.00\n")
    var scan = scan_csv(path, CsvOptions(infer_rows=3, group_rows=2))
    expect(scan.rows() == 7 and scan.row_groups() == 4, "seven rows in groups of two: " + String(scan.row_groups()))
    expect(scan.kind(1) == INT and scan.kind(2) == DECIMAL and scan.scale(2) == 2, "kinds come from the first three rows")
    var rep = scan.csv_report()
    var f = rep.failures()
    expect(len(f) == 3, "three failed cells, found " + String(len(f)))
    expect(f[0].message() == "row 4, column `qty`: `x9` isn't a whole number", f[0].message())
    expect(f[1].message() == "row 5, column `price`: `abc` isn't a number", f[1].message())
    expect(f[2].message() == "row 7, column `qty`: `1.5` isn't a whole number", f[2].message())
    expect(rep.failed_count() == 3 and rep.columns[1].empty == 1 and rep.rows == 7, "counts: failed " + String(rep.failed_count()) + ", empty qty " + String(rep.columns[1].empty))
    # the bad cells are nulls, the rest of their rows is read
    var all: List[Int] = [0, 1, 2]
    var g1 = scan.read_group(1, all)                         # rows 3 and 4
    expect(g1.get(0, 1) == 4 and g1.is_null(1, 1) and g1.get(2, 1) == 400, "row 4 keeps its id and price, qty is null")
    var total = Query(scan).agg(sum(col("qty")), count(col("qty")), count()).compute()
    expect(total.values(0)[0] == 5 + 6 + 7 + 8 and total.values(1)[0] == 4 and total.counts[0] == 7, "the query sees the nulls: " + String(total.values(0)[0]))
    same_in_shapes("bad2.csv", "id,qty,price\n1,5,1.50\n2,6,2.50\n3,7,3.50\n4,x9,4.00\n5,8,abc\n6,,6.0\n7,1.5,7.00\n", CsvOptions(infer_rows=3))


def test_awkward_files() raises:
    var base = CsvOptions()
    same_in_shapes("header_only.csv", "a,b,c\n", base)
    same_in_shapes("header_no_newline.csv", "a,b,c", base)
    same_in_shapes("empty.csv", "", base)
    same_in_shapes("blank.csv", "\n\n\r\n", base)
    same_in_shapes("crlf.csv", "a,b\r\n1,x\r\n2,\"y\r\nz\"\r\n3,w\r\n", base)
    same_in_shapes("no_final_newline.csv", "a,b\n1,x\n2,y", base)
    same_in_shapes("blank_lines.csv", "a,b\n\n1,x\n\n\n2,y\n\n", base)
    same_in_shapes("short_long.csv", "a,b,c\n1,x,1.5\n2,y\n3\n4,z,2.5,extra,more\n5,q,3.5\n", base)
    same_in_shapes("quoted_edges.csv", "a,b\n1,\"\"\n2,\"a\"\"b\"\n3,\"x\ny\nz\"\n4,\"tail\"junk\n5,5\" pipe\n6,end\n", base)
    same_in_shapes("unmatched.csv", "a,b\n1,x\n2,\"open\n3,y\n4,z\n", base)
    same_in_shapes("widen.csv", "a,b\n1,1.50\n2,2.25\n3,3.125\n4,4.5\n5,12345.6789\n6,7\n", CsvOptions(infer_rows=2))
    same_in_shapes("exponent_late.csv", "a,b\n1,1\n2,2\n3,3e5\n4,4\n", CsvOptions(infer_rows=2))
    same_in_shapes("dates_bools.csv", "d,f\n2024-01-05,true\n2024-02-30,FALSE\nx,maybe\n2024-03-01,\n", CsvOptions(infer_rows=1))
    same_in_shapes("no_header.csv", "1,x\n2,y\n3,z\n", CsvOptions(has_header=False))
    same_in_shapes("semicolons.csv", "a;b\n1;\"x;y\"\n2;z\n", CsvOptions(delimiter=";"))
    var tsv = write_file("t.tsv", "a\tb\n1\tx y\n2\t\"p\tq\"\n")
    check_same_as_whole(tsv, CsvOptions(group_rows=1, block_bytes=5), "tsv")


def test_a_byte_order_mark_is_not_part_of_the_first_name() raises:
    var bytes: List[UInt8] = [0xEF, 0xBB, 0xBF]
    bytes.extend(String("id,name\n1,a\n2,b\n").as_bytes())
    var path = write_bytes("bom.csv", bytes)
    var scan = scan_csv(path, CsvOptions(group_rows=1, block_bytes=4))
    expect(scan.name(0) == "id" and scan.name(1) == "name" and scan.rows() == 2, "BOM skipped: `" + scan.name(0) + "`")
    check_same_as_whole(path, CsvOptions(group_rows=1, block_bytes=4), "bom")


def test_very_long_lines() raises:
    var long = String()
    for i in range(30000):
        long += "word" + String(i % 10) + " "
    var content = String("id,note,tail\n1,\"") + long + "\nsecond line\"," + "a\n2," + long + ",b\n3,short,c\n"
    var path = write_file("long.csv", content)
    var scan = scan_csv(path, CsvOptions(group_rows=1, block_bytes=1024))
    expect(scan.rows() == 3 and scan.row_groups() == 3, "three rows, each its own group")
    check_same_as_whole(path, CsvOptions(group_rows=1, block_bytes=1024), "long lines, one row per group")
    check_same_as_whole(path, CsvOptions(group_rows=100, block_bytes=1 << 20), "long lines, one group")
    # a row longer than max_row_bytes is a readable error, not a file read into memory
    var msg = String()
    try:
        _ = scan_csv(path, CsvOptions(block_bytes=1024, max_row_bytes=4096))
    except e:
        msg = String(e)
    expect("row 1 is longer than 4096 bytes" in msg and "max_row_bytes" in msg, "a row past the limit: " + msg)


def test_files_it_refuses() raises:
    var bad = List[UInt8]()
    bad.extend(String("a,b\n1,caf").as_bytes())
    bad.append(0xE9)
    bad.extend(String("\n2,x\n").as_bytes())
    var msg = String()
    try:
        _ = scan_csv(write_bytes("latin1.csv", bad), CsvOptions(group_rows=1, block_bytes=4))
    except e:
        msg = String(e)
    expect("is not valid UTF-8" in msg, "Latin-1: " + msg)
    var u16: List[UInt8] = [0xFF, 0xFE, 0x61, 0x00, 0x0A, 0x00]
    msg = ""
    try:
        _ = scan_csv(write_bytes("u16.csv", u16))
    except e:
        msg = String(e)
    expect("UTF-16" in msg, "UTF-16: " + msg)
    msg = ""
    try:
        _ = scan_csv(DIR + "does_not_exist.csv")
    except e:
        msg = String(e)
    expect("does_not_exist.csv" in msg, "a missing file names it: " + msg)
    msg = ""
    try:
        _ = scan_csv(DIR)
    except e:
        msg = String(e)
    expect("not a regular file" in msg, "a directory: " + msg)
    msg = ""
    try:
        _ = scan_csv(write_file("x.csv", "a\n1\n"), CsvOptions(group_rows=0))
    except e:
        msg = String(e)
    expect("at least 1" in msg, "bad options: " + msg)


def test_a_file_that_changed_is_refused() raises:
    var path = write_file("changing.csv", "a,b\n1,x\n2,y\n3,z\n")
    var scan = scan_csv(path, CsvOptions(group_rows=1))
    var f = open(path, "w")                                  # truncated in place under the open scan
    f.write("a,b\n1,x\n")
    f.close()
    var msg = String()
    try:
        _ = Query(scan).agg(count(col("a"))).compute()
    except e:
        msg = String(e)
    expect("the file changed since it was opened" in msg, "a changed file: " + msg)


def test_header_only_and_empty_scans_query() raises:
    var scan = scan_csv(write_file("h.csv", "a,b\n"))
    expect(scan.columns() == 2 and scan.rows() == 0 and scan.row_groups() == 0, "a header only: two columns, no rows")
    var r = Query(scan).agg(count()).compute()
    expect(r.counts[0] == 0, "count over no rows is 0: " + String(r.counts[0]))
    var empty = scan_csv(write_file("e.csv", ""))
    expect(empty.columns() == 0 and empty.rows() == 0, "an empty file: no columns, no rows")


def main() raises:
    test_every_cell_equals_read_csv()
    test_queries_equal_the_table()
    test_sql_over_a_csv_scan()
    test_frame_edits_and_save_over_a_csv_scan()
    test_the_failed_row_report()
    test_awkward_files()
    test_a_byte_order_mark_is_not_part_of_the_first_name()
    test_very_long_lines()
    test_files_it_refuses()
    test_a_file_that_changed_is_refused()
    test_header_only_and_empty_scans_query()
    print("test_csv_scan ok")
