# F-150: runs one SQL string over Parquet files and prints the result as typed JSON (tests/test_mojo_sql.py compares it with DuckDB).
#
#   sql_probe [--mode table|scan|frame|framescan] [--edits FILE] [--group auto|dense|hashed] [--dense-limit N]
#             [--threads N] [--repeat N] [--group-rows N] [--block-bytes N] [--infer-rows N]
#             "<SQL>" name=path.parquet|path.csv [name=path ...]
# A path ending in .csv or .tsv is read with read_csv (table, frame) or streamed with scan_csv (scan, framescan; F-178: --group-rows,
# --block-bytes and --infer-rows are its CsvOptions).
#
# --mode  table: read the file whole; scan: stream it a row group at a time; frame: read it whole, then apply the edit script
#         to a Frame over it; framescan: the same over a streamed file. The edits go to the first file.
# --edits a tab-separated script: `set COL ROW TEXT`, `append V1 V2 ...` (one per column, empty = NULL), `remove ROW`, `formula name = expr`,
#         `merge COL INTO W1,W2,...` (words folded into INTO).
# Output, one JSON object: {"columns":[{"name","kind","scale"}], "rows":[[...]], "notes":[...], "explain":"...", "ms":x, "times":[...]}
# values: INT a number, DECIMAL / DATE / TEXT a string (a decimal as exact text, a date as yyyy-mm-dd), BOOL true / false,
# FLOAT the float64's bits as "0x....", NULL null. On a SQL error: {"error": "the message"} and exit status 1.
from std.sys import argv
from std.time import perf_counter_ns
from omxcore import (Catalog, sql, PlannerOptions, GROUP_AUTO, GROUP_DENSE, GROUP_HASHED, Frame, Table, SharedTable, read_parquet,
                     scan_parquet, scan_csv, read_csv, CsvOptions, ParquetScan, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT)


def is_csv(path: String) -> Bool:
    return path.endswith(".csv") or path.endswith(".tsv")


def read_whole(path: String, options: CsvOptions) raises -> SharedTable:
    if is_csv(path):
        var r = read_csv(path, options)
        return r.take_table().share()
    var rp = read_parquet(path)
    return rp.take_table().share()


def open_scan(path: String, options: CsvOptions) raises -> ParquetScan:
    return scan_csv(path, options) if is_csv(path) else scan_parquet(path)


def hex_digit(d: Int) -> String:
    var digits = "0123456789abcdef"
    return String(digits[byte=d:d + 1])


def esc_utf8(s: String) -> String:
    """JSON string of s, non-ASCII bytes kept (the text is UTF-8)."""
    var out = String('"')
    var i = 0
    var b = s.as_bytes()
    var start = 0
    while i < len(b):
        var c = b[i]
        if c == UInt8(34) or c == UInt8(92) or c < UInt8(32):
            out += String(unsafe_from_utf8=b[start:i])
            if c == UInt8(34):
                out += '\\"'
            elif c == UInt8(92):
                out += "\\\\"
            else:
                out += "\\u00" + hex_digit(Int(c) // 16) + hex_digit(Int(c) % 16)
            start = i + 1
        i += 1
    out += String(unsafe_from_utf8=b[start:len(b)])
    return out + '"'


def civil(days: Int) -> String:
    var z = days + 719468
    var era = (z if z >= 0 else z - 146096) // 146097
    var doe = z - era * 146097
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var y = yoe + era * 400
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    var mp = (5 * doy + 2) // 153
    var d = doy - (153 * mp + 2) // 5 + 1
    var m = mp + 3 if mp < 10 else mp - 9
    var yy = y + (1 if m <= 2 else 0)
    return pad(yy, 4) + "-" + pad(m, 2) + "-" + pad(d, 2)


def pad(v: Int, w: Int) -> String:
    var s = String(v)
    while s.byte_length() < w:
        s = "0" + s
    return s


def decimal_text(v: Int64, scale: Int) -> String:
    var neg = v < 0
    var digits = String(UInt64(v) if not neg else UInt64(0) - UInt64(v))
    if scale == 0:
        return ("-" if neg else "") + digits
    while digits.byte_length() <= scale:
        digits = "0" + digits
    var cut = digits.byte_length() - scale
    return ("-" if neg else "") + String(digits[byte=:cut]) + "." + String(digits[byte=cut:])


def hex64(v: Int64) -> String:
    var u = UInt64(v)
    var out = String()
    for k in range(16):
        var shift = UInt64(60 - 4 * k)
        out += hex_digit(Int((u >> shift) & 0xF))
    return "0x" + out


def kind_text(k: Int) -> String:
    var names: List[String] = ["INT", "DECIMAL", "TEXT", "DATE", "BOOL", "FLOAT"]
    return names[k]


def json_result(t: Table, notes: List[String], explain: String, ms: Float64, times: List[Float64]) raises -> String:
    var out = String('{"columns": [')
    for c in range(t.columns()):
        out += ("" if c == 0 else ", ") + '{"name": ' + esc_utf8(t.name(c)) + ', "kind": "' + kind_text(t.kind(c).value) + '", "scale": ' + String(t.scale(c)) + "}"
    out += '], "rows": ['
    for r in range(t.rows()):
        out += ("" if r == 0 else ", ") + "["
        for c in range(t.columns()):
            out += "" if c == 0 else ", "
            if t.is_null(c, r):
                out += "null"
                continue
            var k = t.kind(c).value
            var v = t.get(c, r)
            if k == INT.value:
                out += String(v)
            elif k == DECIMAL.value:
                out += '"' + decimal_text(v, t.scale(c)) + '"'
            elif k == TEXT.value:
                out += esc_utf8(t.text_of(c, Int(v)))
            elif k == DATE.value:
                out += '"' + civil(Int(v)) + '"'
            elif k == BOOL.value:
                out += "true" if v != 0 else "false"
            else:
                out += '"' + hex64(v) + '"'
        out += "]"
    out += '], "notes": ['
    for i in range(len(notes)):
        out += ("" if i == 0 else ", ") + esc_utf8(notes[i])
    out += '], "explain": ' + esc_utf8(explain) + ', "ms": ' + String(ms) + ', "times": ['
    for i in range(len(times)):
        out += ("" if i == 0 else ", ") + String(times[i])
    return out + "]}"


def split(s: String, sep: String) -> List[String]:
    var out = List[String]()
    for part in s.split(sep):
        out.append(String(part))
    return out^


def apply_edits(mut f: Frame, path: String) raises:
    var text = String()
    with open(path, "r") as fh:
        text = fh.read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        var p = split(String(line), "\t")
        if p[0] == "set":
            f.set_text(f.column(p[1]), Int(p[2]), p[3] if len(p) > 3 else "")
        elif p[0] == "append":
            var vals = List[String]()
            for i in range(1, len(p)):
                vals.append(p[i])
            _ = f.append(vals)
        elif p[0] == "formula":
            _ = f.formula(p[1])
        elif p[0] == "remove":
            f.remove(Int(p[1]))
        elif p[0] == "merge":
            var c = f.column(p[1])
            var into = f.code_of(c, p[2])
            var codes = List[Int64]()
            for w in split(p[3], ","):
                var k = f.code_of(c, w)
                if k >= 0:
                    codes.append(Int64(k))
            _ = f.merge(c, codes, Int64(into))
        else:
            raise Error("edit script: unknown edit `" + p[0] + "`")


def main() raises:
    var args = argv()
    var mode = String("table")
    var edits = String()
    var opts = PlannerOptions()
    var repeat = 1
    var csv = CsvOptions()
    var positional = List[String]()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--mode":
            mode = String(args[i + 1])
            i += 2
        elif a == "--edits":
            edits = String(args[i + 1])
            i += 2
        elif a == "--group":
            var g = String(args[i + 1])
            opts.group = GROUP_DENSE if g == "dense" else GROUP_HASHED if g == "hashed" else GROUP_AUTO
            i += 2
        elif a == "--dense-limit":
            opts.dense_limit = Int(String(args[i + 1]))
            i += 2
        elif a == "--threads":
            opts.threads = Int(String(args[i + 1]))
            i += 2
        elif a == "--group-rows":
            csv.group_rows = Int(String(args[i + 1]))
            i += 2
        elif a == "--block-bytes":
            csv.block_bytes = Int(String(args[i + 1]))
            i += 2
        elif a == "--infer-rows":
            csv.infer_rows = Int(String(args[i + 1]))
            i += 2
        elif a == "--repeat":
            repeat = Int(String(args[i + 1]))
            i += 2
        else:
            positional.append(a)
            i += 1
    var text = positional[0]
    var cat = Catalog()
    for k in range(1, len(positional)):
        var kv = split(positional[k], "=")
        var name = kv[0]
        var path = kv[1]
        if mode == "table" or (mode == "frame" and k > 1):
            cat.add_table(name, read_whole(path, csv))
        elif mode == "scan" or (mode == "framescan" and k > 1):
            cat.add_scan(name, open_scan(path, csv))
        elif mode == "frame":
            var f = Frame(read_whole(path, csv))
            if edits.byte_length() > 0:
                apply_edits(f, edits)
            cat.add_frame(name, f)
        else:
            var f = Frame(open_scan(path, csv))
            if edits.byte_length() > 0:
                apply_edits(f, edits)
            cat.add_frame(name, f)
    try:
        var times = List[Float64]()
        var last = sql(cat, text, opts)
        times.append(last.ms)
        for _ in range(1, repeat):
            last = sql(cat, text, opts)
            times.append(last.ms)
        print(json_result(last.table, last.notes, last.explain, last.ms, times))
    except e:
        print('{"error": ' + esc_utf8(String(e)) + "}")
        raise Error("sql failed")
