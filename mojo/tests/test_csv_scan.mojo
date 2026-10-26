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


def cell(t: SharedTable, c: Int, row: Int) -> String:
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


def same_table(a: SharedTable, b: SharedTable, label: String) raises:
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
    same_table(scan_table(scan), wt, label + " cells")
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


def main() raises:
    test_every_cell_equals_read_csv()
    test_queries_equal_the_table()
    print("test_csv_scan ok")
