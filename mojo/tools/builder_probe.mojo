# F-204: runs one builder query (`Query.join`, `where`, `group_by`, `agg`) over Parquet files and prints the result as JSON, for tests/test_f204_builder_join.py to
# compare with DuckDB and with `sql_probe` (the same join as SQL text).
#
#   builder_probe [--mode table|frame] [--right-mode table|frame] [--edits FILE] [--redits FILE] [--how inner|left|semi|anti] [--on lk=rk,lk2=rk2]
#                 [--where "a<5;b>=3|c==2"] [--group COL:GROUPS] [--aggs "count,sum:v,min:v,max:v,mean:v,countof:v,distinct:v,sum:a*b"] [--threads N] [--repeat N]
#                 left=path.parquet right=path.parquet
#
# --mode / --right-mode  table: the file read whole; frame: read whole, then the edit script (the format of sql_probe) applied to a Frame over it.
# --where   conjuncts separated by `;`, each a `|` list of `column op integer` (op: < <= > >= == !=); every conjunct is one `.where(...)`.
# --aggs    count, count of a column, count_distinct (F-206: `distinct:v`), sum, min, max, mean of a column or of `a*b` / `a+b` / `a-b`.
# Output: {"notes": [...], "rows": [[g, count(*), agg...]], "ms": x}: the groups that have rows (all of the one group of no `--group`); a sum / min / max / mean with no
# value in a group is null, a mean is the float64's text.
from std.sys import argv
from std.time import perf_counter_ns
from omxcore import (Frame, Table, SharedTable, read_parquet, Query, QueryResult, Agg, Expr, col, lit, count, sum, mean, min_of, max_of, count_distinct, float_text, CPU)


def split(s: String, sep: String) -> List[String]:
    var out = List[String]()
    for part in s.split(sep):
        out.append(String(part))
    return out^


def esc(s: String) -> String:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


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
        elif p[0] == "remove":
            f.remove(Int(p[1]))
        else:
            raise Error("edit script: unknown edit `" + p[0] + "`")


def operand(text: String) raises -> Expr:
    """`name`, `a*b`, `a+b`, `a-b`."""
    for op in ["*", "+", "-"]:
        var parts = split(text, op)
        if len(parts) == 2:
            if op == "*":
                return col(parts[0]) * col(parts[1])
            if op == "+":
                return col(parts[0]) + col(parts[1])
            return col(parts[0]) - col(parts[1])
    return col(text)


def term(text: String) raises -> Expr:
    for op in ["<=", ">=", "==", "!=", "<", ">"]:
        var parts = text.split(op, 1)
        if len(parts) == 2:
            var l = col(String(parts[0]))
            var v = lit(Int(String(parts[1])))
            if op == "<=":
                return l <= v
            if op == ">=":
                return l >= v
            if op == "==":
                return l == v
            if op == "!=":
                return l != v
            if op == "<":
                return l < v
            return l > v
    raise Error("where: cannot read `" + text + "`")


def load(path: String, mode: String, edits: String) raises -> Frame:
    var rp = read_parquet(path)
    var f = Frame(rp.take_table().share())
    if mode == "frame" and edits.byte_length() > 0:
        apply_edits(f, edits)
    return f^


def main() raises:
    var args = argv()
    var mode = String("table")
    var rmode = String("table")
    var edits = String()
    var redits = String()
    var how = String("inner")
    var on = String()
    var wh = String()
    var group = String()
    var aggs = String("count")
    var threads = 0
    var repeat = 1
    var left_path = String()
    var right_path = String()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--mode":
            mode = String(args[i + 1])
            i += 2
        elif a == "--right-mode":
            rmode = String(args[i + 1])
            i += 2
        elif a == "--edits":
            edits = String(args[i + 1])
            i += 2
        elif a == "--redits":
            redits = String(args[i + 1])
            i += 2
        elif a == "--how":
            how = String(args[i + 1])
            i += 2
        elif a == "--on":
            on = String(args[i + 1])
            i += 2
        elif a == "--where":
            wh = String(args[i + 1])
            i += 2
        elif a == "--group":
            group = String(args[i + 1])
            i += 2
        elif a == "--aggs":
            aggs = String(args[i + 1])
            i += 2
        elif a == "--threads":
            threads = Int(String(args[i + 1]))
            i += 2
        elif a == "--repeat":
            repeat = Int(String(args[i + 1]))
            i += 2
        elif a.startswith("left="):
            left_path = String(a[byte=5:])
            i += 1
        elif a.startswith("right="):
            right_path = String(a[byte=6:])
            i += 1
        else:
            raise Error("unknown argument " + a)
    try:
        var lf = load(left_path, mode, edits)
        var rf = load(right_path, rmode, redits)
        var lon = List[String]()
        var ron = List[String]()
        for pair in split(on, ","):
            var kv = split(pair, "=")
            lon.append(kv[0])
            ron.append(kv[1])
        var best = Float64(1e30)
        var out = String()
        for _ in range(repeat):
            var t0 = perf_counter_ns()
            var q: Query
            if mode == "frame":
                q = Query(lf)
            else:
                q = Query(lf.base)
            if rmode == "frame":
                q = q^.join(rf, lon, ron, how)
            else:
                q = q^.join(rf.base, lon, ron, how)
            if threads > 0:
                q = q^.threads(threads)
            if wh.byte_length() > 0:
                for conj in split(wh, ";"):
                    var terms = split(conj, "|")
                    var e = term(terms[0])
                    for k in range(1, len(terms)):
                        e = e | term(terms[k])
                    q = q^.where(e^)
            var groups = 1
            if group.byte_length() > 0:
                var gp = split(group, ":")
                groups = Int(gp[1])
                q = q^.group_by(col(gp[0]), groups)
            var names = List[String]()
            for spec in split(aggs, ","):
                var p = split(spec, ":")
                names.append(spec)
                if p[0] == "count":
                    q = q^.agg(count())
                elif p[0] == "countof":
                    q = q^.agg(count(operand(p[1])))
                elif p[0] == "distinct":
                    q = q^.agg(count_distinct(operand(p[1])))
                elif p[0] == "sum":
                    q = q^.agg(sum(operand(p[1])))
                elif p[0] == "min":
                    q = q^.agg(min_of(operand(p[1])))
                elif p[0] == "max":
                    q = q^.agg(max_of(operand(p[1])))
                elif p[0] == "mean":
                    q = q^.agg(mean(operand(p[1])))
                else:
                    raise Error("unknown aggregate " + p[0])
            var r = q.compute(CPU)
            var ms = Float64(perf_counter_ns() - t0) / 1e6
            if ms < best:
                best = ms
            out = String('{"notes": [')
            for n in range(len(r.notes)):
                out += ("" if n == 0 else ", ") + esc(r.notes[n])
            out += '], "rows": ['
            var first = True
            for g in range(groups):
                if r.counts[g] == 0 and group.byte_length() > 0:
                    continue
                out += ("" if first else ", ") + "[" + String(g) + ", " + String(r.counts[g])
                first = False
                for k in range(len(names)):
                    var kind = split(names[k], ":")[0]
                    if kind == "count":
                        continue
                    out += ", "
                    if kind == "countof" or kind == "distinct":
                        out += String(r.values(k)[g])
                    elif not r.present(k, g):
                        out += "null"
                    elif kind == "mean":
                        out += esc(float_text(r.means(k)[g]))
                    else:
                        out += String(r.values(k)[g])
                out += "]"
            out += '], "ms": ' + String(best) + "}"
        print(out)
    except e:
        print('{"error": ' + esc(String(e)) + "}")
