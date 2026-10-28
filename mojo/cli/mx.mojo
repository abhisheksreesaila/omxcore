# F-151: `mx`, the command line for agents (and people): SQL in, JSON / CSV / Parquet out, EXPLAIN, schema.
#
#   mx sql "<SQL>" [-t name=path ...] [--format json|jsonl|csv|parquet] [--out file] [--max-rows N] [--device cpu|gpu|auto]
#   mx explain [--analyze] "<SQL>" [-t name=path ...] [--format text|json]
#   mx schema <file> [--rows N]
#
# Files are read where the SQL names them (FROM 'orders.parquet', FROM 'sales.csv') or where `-t name=path` registers them (FROM name).
# Every answer is a JSON object on stdout (an error too: {"error": {...}}); the exit status is 0, 1 for a query error and 2 for a usage error.
from std.sys import argv, exit
from std.time import perf_counter_ns
from std.os.path import dirname, isdir
from omxcore import (Catalog, PlannerOptions, GROUP_AUTO, GROUP_DENSE, GROUP_HASHED, BUILD_LEFT, BUILD_RIGHT, sql, SqlResult, Table, SharedTable, read_csv, CsvOptions,
                     read_parquet, scan_parquet, write_parquet, sql_to_parquet, ZSTD, parse_sql, TEXT, INT, DECIMAL, DATE, BOOL, FLOAT)
from omxcore.output import (json_string, columns_json, rows_json, rows_json_one, row_jsonl, csv_header, csv_row, cell_json, kind_text)
from omxcore.parquet import physical_memory, DEFAULT_ROW_GROUP

comptime USAGE = ("usage: mx sql \"<SQL>\" [-t name=path ...] [-f query.sql] [--format json|jsonl|csv|parquet] [--out file] [--max-rows N] [--device cpu|gpu|auto]\n"
                  "       mx explain [--analyze] \"<SQL>\" [-t name=path ...] [--format text|json]\n"
                  "       mx schema <file> [--rows N]\n"
                  "advanced: --group auto|dense|hashed  --dense-limit N  --threads N  --stream auto|always|never (Parquet: stream row groups; auto streams a file over a quarter of memory)  --memory BYTES (the memory auto compares with)  --join auto|materialize (a join with unique build keys is fused into the pipeline; materialize writes its rows first)  --order written|P,P,P (the order to join three or more tables in: the written one, or the tables' positions in FROM, 0 first; default: the planner's)  --build left|right (the side of each hash join the table is built on; default: the smaller)  --join-budget BYTES (the bytes of joined values a materialized join may take)  --sort-budget BYTES (the bytes of rows a sort holds in memory: past it ORDER BY sorts runs on disk and merges them; with --format parquet --out the merged rows go straight into the file)  --spill-dir DIR (where the runs go; default TMPDIR or /tmp)  --row-group-rows N (rows of each Parquet row group written)")


# ── errors ───────────────────────────────────────────────────────────────────────────────────────────

def error_json(message: String, line: Int, col: Int, hint: String) -> String:
    return ('{"error":{"message":' + json_string(message) + ',"line":' + (String(line) if line > 0 else "null") + ',"column":' +
            (String(col) if col > 0 else "null") + ',"hint":' + (json_string(hint) if hint.byte_length() > 0 else "null") + "}}")


def usage_error(message: String) -> Int:
    print(error_json(message, 0, 0, USAGE))
    exit(2)
    return 2


def query_error(text: String) raises -> Int:
    """Prints a failed query as JSON and exits 1; an SQL error is split into its position, message and hint."""
    var message = text
    var line = 0
    var col = 0
    var hint = String()
    if text.startswith("SQL error at line "):
        var c_at = text.find(", column ")
        var colon = text.find(": ", c_at if c_at >= 0 else 0)
        if c_at >= 0 and colon > c_at:
            line = Int(String(text[byte=18:c_at]))
            col = Int(String(text[byte=c_at + 9:colon]))
            message = String(text[byte=colon + 2:])
    var dym = message.find(" (did you mean ")
    if dym >= 0 and message.endswith("?)"):
        var h = String(message[byte=dym + 2:message.byte_length() - 1])
        var m = String(message[byte=:dym])
        hint = h^
        message = m^
    elif "(the catalog is empty)" in message or "(the catalog has " in message or "is not open" in message:
        hint = "name the file in the query (FROM 'orders.parquet') or register it with -t name=path"
    elif "joined by `OR`" in message or "joined by `AND`" in message:
        hint = "a chain of about 255 OR terms hits the depth limit: write x IN (a, b, c) instead (a list of numbers takes up to 40 values; text lists are longer)"
    elif "levels deep" in message:
        hint = "flatten the parentheses or NOT chain"
    elif "not valid UTF-8" in message or "is UTF-16" in message:
        hint = "mx reads UTF-8 only: re-save the file as UTF-8 (for example iconv -f LATIN1 -t UTF-8 in.csv > out.csv)"
    print(error_json(message, line, col, hint))
    exit(1)
    return 1


# ── arguments ────────────────────────────────────────────────────────────────────────────────────────

struct Args(Movable):
    var positional: List[String]
    var tables: List[String]         # name=path
    var format: String
    var out: String
    var max_rows: Int
    var device: String
    var group: String
    var dense_limit: Int
    var threads: Int
    var stream: String
    var memory: Int
    var join: String
    var order: String
    var build: String
    var join_budget: Int
    var sort_budget: Int
    var sort_run_rows: Int
    var spill_dir: String
    var row_group_rows: Int
    var analyze: Bool
    var rows: Int
    var query_file: String

    def __init__(out self):
        self.positional = List[String]()
        self.tables = List[String]()
        self.format = String()
        self.out = String()
        self.max_rows = 1000
        self.device = "auto"
        self.group = "auto"
        self.dense_limit = 0
        self.threads = 0
        self.stream = "auto"
        self.memory = 0
        self.join = "auto"
        self.order = String()
        self.build = "auto"
        self.join_budget = 0
        self.sort_budget = 0
        self.sort_run_rows = 0
        self.spill_dir = String()
        self.row_group_rows = 0
        self.analyze = False
        self.rows = 5
        self.query_file = String()


def parse_int(what: String, text: String) raises -> Int:
    var b = text.as_bytes()
    if len(b) == 0 or len(b) > 12:
        _ = usage_error(what + " needs a whole number, found `" + text + "`")
    var v = 0
    for x in b:
        if x < UInt8(48) or x > UInt8(57):
            _ = usage_error(what + " needs a whole number, found `" + text + "`")
        v = v * 10 + Int(x - 48)
    return v


def parse_args(first: Int) raises -> Args:
    var a = Args()
    var all = argv()
    var i = first
    while i < len(all):
        var arg = String(all[i])
        var takes = arg == "-t" or arg == "--table" or arg == "--format" or arg == "--out" or arg == "--max-rows" or arg == "--device" or \
                    arg == "--group" or arg == "--dense-limit" or arg == "--threads" or arg == "--stream" or arg == "--join" or arg == "--order" or arg == "--build" or arg == "--join-budget" or arg == "--sort-budget" or arg == "--sort-run-rows" or arg == "--spill-dir" or arg == "--row-group-rows" or arg == "--memory" or arg == "--rows" or arg == "-f" or arg == "--file"
        if takes:
            if i + 1 >= len(all):
                _ = usage_error(arg + " needs a value")
            var v = String(all[i + 1])
            i += 2
            if arg == "-t" or arg == "--table":
                if v.find("=") <= 0:
                    _ = usage_error("-t needs name=path, found `" + v + "`")
                a.tables.append(v)
            elif arg == "--format":
                a.format = v
            elif arg == "--out":
                a.out = v
            elif arg == "--max-rows":
                a.max_rows = parse_int("--max-rows", v)
            elif arg == "--device":
                a.device = v
            elif arg == "--group":
                if v != "auto" and v != "dense" and v != "hashed":
                    _ = usage_error("--group is auto, dense or hashed, found `" + v + "`")
                a.group = v
            elif arg == "--dense-limit":
                a.dense_limit = parse_int("--dense-limit", v)
            elif arg == "--threads":
                a.threads = parse_int("--threads", v)
            elif arg == "--join":
                if v != "auto" and v != "materialize":
                    _ = usage_error("--join is auto or materialize, found `" + v + "`")
                a.join = v
            elif arg == "--order":
                a.order = v
            elif arg == "--build":
                if v != "auto" and v != "left" and v != "right":
                    _ = usage_error("--build is auto, left or right, found `" + v + "`")
                a.build = v
            elif arg == "--sort-budget":
                a.sort_budget = parse_int("--sort-budget", v)
            elif arg == "--sort-run-rows":
                a.sort_run_rows = parse_int("--sort-run-rows", v)
            elif arg == "--spill-dir":
                a.spill_dir = v
            elif arg == "--row-group-rows":
                a.row_group_rows = parse_int("--row-group-rows", v)
            elif arg == "--join-budget":
                a.join_budget = parse_int("--join-budget", v)
            elif arg == "--stream":
                if v != "auto" and v != "always" and v != "never":
                    _ = usage_error("--stream is auto, always or never, found `" + v + "`")
                a.stream = v
            elif arg == "--rows":
                a.rows = parse_int("--rows", v)
            elif arg == "--memory":
                a.memory = parse_int("--memory", v)
            else:
                a.query_file = v
            continue
        if arg == "--analyze":
            a.analyze = True
        elif arg.startswith("-") and arg.byte_length() > 1:
            _ = usage_error("unknown option `" + arg + "`")
        else:
            a.positional.append(arg)
        i += 1
    return a^


# ── files ────────────────────────────────────────────────────────────────────────────────────────────

def extension(path: String) -> String:
    var dot = path.rfind(".")
    if dot < 0:
        return String()
    var e = String()
    for b in String(path[byte=dot + 1:]).as_bytes():
        e += chr(Int(b + 32)) if UInt8(65) <= b <= UInt8(90) else chr(Int(b))
    return e^


def load_file(mut cat: Catalog, name: String, path: String, stream: String, memory_bytes: Int, mut notes: List[String]) raises:
    var t0 = perf_counter_ns()
    var ext = extension(path)
    if ext == "parquet" or ext == "pq":
        var scan = scan_parquet(path)
        var memory = memory_bytes if memory_bytes > 0 else physical_memory()
        var streamed = stream == "always" or (stream == "auto" and memory > 0 and scan.file_bytes() * 4 > memory)
        if streamed:
            cat.add_scan(name, scan)
            notes.append("file " + path + ": streamed a row group at a time (" + String(scan.row_groups()) + " row groups, " + String(scan.rows()) + " rows)")
        else:
            var rp = read_parquet(path)
            var table = rp.take_table().share()
            notes.append("file " + path + ": read into memory (" + String(table.rows()) + " rows, " + String(Int(Float64(perf_counter_ns() - t0) / 1e6)) + " ms)")
            cat.add_table(name, table)
    elif ext == "csv" or ext == "tsv" or ext == "tab" or ext == "txt":
        var r = read_csv(path, CsvOptions())
        var table = r.take_table().share()
        notes.append("file " + path + ": read into memory (" + String(table.rows()) + " rows, " + String(Int(Float64(perf_counter_ns() - t0) / 1e6)) + " ms)")
        cat.add_table(name, table)
    else:
        raise Error("cannot tell the format of `" + path + "`: use a .parquet, .csv or .tsv file")


def open_tables(mut cat: Catalog, args: Args, text: String, mut notes: List[String]) raises:
    for spec in args.tables:
        var eq = spec.find("=")
        load_file(cat, String(spec[byte=:eq]), String(spec[byte=eq + 1:]), args.stream, args.memory, notes)
    try:
        var st = parse_sql(text)
        if st.table_is_file and cat.find(st.table) < 0:
            load_file(cat, st.table, st.table, args.stream, args.memory, notes)
        if st.has_right and st.right_is_file and cat.find(st.right_table) < 0:
            load_file(cat, st.right_table, st.right_table, args.stream, args.memory, notes)
    except e:
        _ = query_error(String(e))


def options_of(args: Args) raises -> PlannerOptions:
    var o = PlannerOptions()
    if args.group == "dense":
        o.group = GROUP_DENSE
    elif args.group == "hashed":
        o.group = GROUP_HASHED
    elif args.group != "auto":
        _ = usage_error("--group is auto, dense or hashed, found `" + args.group + "`")
    if args.dense_limit > 0:
        o.dense_limit = args.dense_limit
    o.threads = args.threads
    o.fuse = args.join != "materialize"
    if args.order == "written":
        o.written_order = True
    elif args.order.byte_length() > 0:
        for part in args.order.split(","):
            o.join_order.append(parse_int("--order", String(part)))
    if args.build == "left":
        o.build_side = BUILD_LEFT
    elif args.build == "right":
        o.build_side = BUILD_RIGHT
    o.join_budget = args.join_budget
    o.sort_budget = args.sort_budget
    o.sort_run_rows = args.sort_run_rows
    o.spill_dir = args.spill_dir
    return o^


def query_text(args: Args, what: String) raises -> String:
    if args.query_file.byte_length() > 0:
        try:
            with open(args.query_file, "r") as fh:
                return fh.read()
        except e:
            _ = usage_error("cannot read the query file `" + args.query_file + "`")
    if len(args.positional) != 1:
        _ = usage_error(what + " takes one SQL string, found " + String(len(args.positional)) + " arguments")
    return args.positional[0]


def check_device(args: Args):
    if args.device != "cpu" and args.device != "auto" and args.device != "gpu":
        _ = usage_error("--device is cpu, gpu or auto, found `" + args.device + "`")


def notes_json(notes: List[String]) -> String:
    var out = String("[")
    for i in range(len(notes)):
        out += ("," if i > 0 else "") + json_string(notes[i])
    return out + "]"


comptime FLUSH_AT = 1 << 20


def check_out_directory(path: String) raises:
    """`path`'s directory must exist: mx never creates directories (Mojo's open would)."""
    var dir = dirname(path)
    if dir.byte_length() > 0 and not isdir(dir):
        raise Error("couldn't write `" + path + "`: the directory `" + dir + "` does not exist")


def open_for_write(path: String) raises -> FileHandle:
    """Opens `path` for writing."""
    check_out_directory(path)
    try:
        return open(path, "w")
    except:
        raise Error("couldn't write `" + path + "`: no such directory, or no permission to write there")


def put(mut fh: FileHandle, mut buf: String, piece: String) raises:
    """Adds to the buffer and writes it out in chunks of about a megabyte, so a big result never sits in one String."""
    buf += piece
    if buf.byte_length() >= FLUSH_AT:
        fh.write(buf)
        buf = String()


def write_rows(path: String, t: Table, fmt: String) raises:
    """All rows of `t` to `path` as json, jsonl or csv."""
    var fh = open_for_write(path)
    var buf = String()
    var n = t.rows()
    if fmt == "csv":
        put(fh, buf, csv_header(t) + "\n")
        for i in range(n):
            put(fh, buf, csv_row(t, i) + "\n")
    elif fmt == "jsonl":
        for i in range(n):
            put(fh, buf, row_jsonl(t, i) + "\n")
    else:
        put(fh, buf, '{"columns":' + columns_json(t) + ',"rows":[')
        for i in range(n):
            put(fh, buf, ("," if i > 0 else "") + rows_json_one(t, i))
        put(fh, buf, "]}\n")
    fh.write(buf)
    fh.close()


# ── commands ─────────────────────────────────────────────────────────────────────────────────────────

def cmd_sql(args: Args) raises:
    check_device(args)
    var fmt = args.format if args.format.byte_length() > 0 else "json"
    if fmt != "json" and fmt != "jsonl" and fmt != "csv" and fmt != "parquet":
        _ = usage_error("--format is json, jsonl, csv or parquet, found `" + fmt + "`")
    if fmt == "parquet" and args.out.byte_length() == 0:
        _ = usage_error("--format parquet needs --out file.parquet")
    var text = query_text(args, "mx sql")
    var cat = Catalog()
    var notes = List[String]()
    if args.device == "gpu":
        notes.append("device: --device gpu asked for; SQL runs on the CPU in this version")
    var r = SqlResult()
    try:
        open_tables(cat, args, text, notes)
        if fmt == "parquet":                    # F-167: an ORDER BY bigger than the sort budget is merged straight into the file
            check_out_directory(args.out)
            r = sql_to_parquet(cat, text, args.out, options_of(args), "", ZSTD, args.row_group_rows if args.row_group_rows > 0 else DEFAULT_ROW_GROUP)
        else:
            r = sql(cat, text, options_of(args))
    except e:
        _ = query_error(String(e))
    for n in r.notes:
        notes.append(n)
    var total = r.rows_written if r.rows_written >= 0 else r.table.rows()
    var to_file = args.out.byte_length() > 0
    var shown = total if (to_file or args.max_rows == 0) else min(total, args.max_rows)
    if to_file:
        try:
            if fmt != "parquet":                    # (a Parquet file was written by sql_to_parquet above)
                write_rows(args.out, r.table, fmt)
        except e:
            var why = String(e)
            _ = query_error(why if args.out in why else "couldn't write `" + args.out + "`: " + why)
        notes.append("wrote " + String(total) + " rows to " + args.out + " as " + fmt)
    if fmt == "jsonl" and not to_file:
        for i in range(shown):
            print(row_jsonl(r.table, i))
        return
    if fmt == "csv" and not to_file:
        print(csv_header(r.table))
        for i in range(shown):
            print(csv_row(r.table, i))
        return
    var rows = rows_json(r.table, 0 if to_file else shown)
    print('{"columns":' + columns_json(r.table) + ',"rows":' + rows + ',"row_count":' + String(total) + ',"truncated":' +
          ("true" if shown < total else "false") + ',"ms":' + String(r.ms) + ',"device":"cpu","notes":' + notes_json(notes) +
          (',"out":' + json_string(args.out) if to_file else "") + "}")


def without_explain(text: String) -> String:
    """`text` without a leading EXPLAIN [ANALYZE] (any case), so the SELECT itself can be opened and run."""
    var b = text.as_bytes()
    var n = len(b)
    var i = 0
    for word in ["explain", "analyze"]:
        var j = i
        while j < n and (b[j] == UInt8(32) or b[j] == UInt8(9) or b[j] == UInt8(10) or b[j] == UInt8(13)):
            j += 1
        var k = 0
        var w = word.as_bytes()
        while k < len(w) and j + k < n and (b[j + k] | UInt8(32)) == w[k]:
            k += 1
        var ends = j + k >= n or b[j + k] == UInt8(32) or b[j + k] == UInt8(9) or b[j + k] == UInt8(10) or b[j + k] == UInt8(13)
        if k == len(w) and ends:
            i = j + k
        elif word == "explain":
            return text
    return String(text[byte=i:])


def cmd_explain(args: Args) raises:
    check_device(args)
    var fmt = args.format if args.format.byte_length() > 0 else "text"
    if fmt != "text" and fmt != "json":
        _ = usage_error("mx explain: --format is text or json, found `" + fmt + "`")
    var text = query_text(args, "mx explain")
    var body = without_explain(text)              # EXPLAIN [ANALYZE] SELECT ... is accepted: the SELECT is what is opened and run
    var cat = Catalog()
    var notes = List[String]()
    var r = SqlResult()
    try:
        open_tables(cat, args, body, notes)
        r = sql(cat, ("EXPLAIN ANALYZE " if args.analyze else "EXPLAIN ") + body, options_of(args))
    except e:
        _ = query_error(String(e))
    for n in r.notes:
        notes.append(n)
    if fmt == "text":
        print(r.explain.rstrip("\n"))
        return
    var lines = String("[")
    var first = True
    for line in r.explain.split("\n"):
        if line.byte_length() > 0:
            lines += ("" if first else ",") + json_string(String(line))
            first = False
    print('{"plan":' + lines + '],"analyzed":' + ("true" if args.analyze else "false") + ',"ms":' + String(r.ms) + ',"notes":' + notes_json(notes) + "}")


def cmd_schema(args: Args) raises:
    if len(args.positional) != 1:
        _ = usage_error("mx schema takes one file, found " + String(len(args.positional)) + " arguments")
    var path = args.positional[0]
    var ext = extension(path)
    var out = String()
    try:
        if ext == "parquet" or ext == "pq":
            var scan = scan_parquet(path)
            var cols = List[Int]()
            for c in range(scan.columns()):
                cols.append(c)
            var first = scan.read_group(0, cols) if scan.row_groups() > 0 and scan.group_rows(0) > 0 else Table.empty()
            out = ('{"file":' + json_string(path) + ',"format":"parquet","bytes":' + String(scan.file_bytes()) + ',"rows":' + String(scan.rows()) +
                   ',"row_groups":' + String(scan.row_groups()) + ',"columns":[')
            for c in range(scan.columns()):
                var nulls = scan.null_count(c)
                if nulls < 0:                                          # the footer states none: count them
                    nulls = 0
                    for g in range(scan.row_groups()):
                        if scan.group_rows(g) > 0:
                            nulls += scan.read_group(g, [c]).null_count(0)
                var k = scan.kind(c).value
                var words = String("null")
                if k == TEXT.value:
                    words = String(scan.word_count(c))
                var vals = String("[")
                for r in range(min(args.rows, first.rows())):
                    var cell: String
                    if first.is_null(c, r):
                        cell = "null"
                    elif k == TEXT.value:
                        cell = json_string(scan.text_of(c, Int(first.get(c, r))))
                    else:
                        cell = cell_json(first, c, r)
                    vals += ("," if r > 0 else "") + cell
                out += (("," if c > 0 else "") + '{"name":' + json_string(scan.name(c)) + ',"kind":"' + kind_text(k) + '","scale":' + String(scan.scale(c)) +
                        ',"nulls":' + String(nulls) + ',"distinct_words":' + words + ',"first_values":' + vals + "]}")
            out += "]}"
        elif ext == "csv" or ext == "tsv" or ext == "tab" or ext == "txt":
            var r = read_csv(path, CsvOptions())
            var t = r.take_table()
            out = ('{"file":' + json_string(path) + ',"format":"csv","rows":' + String(t.rows()) + ',"row_groups":null,"columns":[')
            for c in range(t.columns()):
                var k = t.kind(c).value
                var vals = String("[")
                for row in range(min(args.rows, t.rows())):
                    vals += ("," if row > 0 else "") + cell_json(t, c, row)
                out += (("," if c > 0 else "") + '{"name":' + json_string(t.name(c)) + ',"kind":"' + kind_text(k) + '","scale":' + String(t.scale(c)) +
                        ',"nulls":' + String(t.null_count(c)) + ',"distinct_words":' + (String(len(t.dictionary(c))) if k == TEXT.value else "null") +
                        ',"first_values":' + vals + "]}")
            out += "]}"
        else:
            _ = usage_error("cannot tell the format of `" + path + "`: use a .parquet, .csv or .tsv file")
    except e:
        _ = query_error(String(e))
    print(out)


def main() raises:
    var all = argv()
    if len(all) < 2:
        _ = usage_error("a command is needed: sql, explain or schema")
    var cmd = String(all[1])
    if cmd == "help" or cmd == "--help" or cmd == "-h":
        print(USAGE)
        return
    var args = parse_args(2)
    if cmd == "sql":
        cmd_sql(args)
    elif cmd == "explain":
        cmd_explain(args)
    elif cmd == "schema":
        cmd_schema(args)
    else:
        _ = usage_error("unknown command `" + cmd + "`: sql, explain or schema")
