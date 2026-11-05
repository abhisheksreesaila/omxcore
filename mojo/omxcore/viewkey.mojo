"""The cache key of a named view (F-160): its canonical plan text and the versions of everything it reads.

    "SELECT region , sum ( qty ) FROM orders GROUP BY region | source orders=table#1403…v2 ; view totals=[…]"

- the plan text is the view's SQL as tokens (white space and comments gone), so the same query written another way over the lines keeps its result;
- a source it names is its identity (the shared table it reads) and, for a Frame, the number of edits when it was added to the catalog, so an edit changes
  the key and a catalog that gets the same table again keeps it; a Parquet scan or a file named in FROM adds its size and modification time;
- a view it names adds that view's own key, so an edit changes the keys of the views below it and of no other.

The names are found by reading the tokens, not the parse: a word that is also a column name is an extra input at worst (a key that changes with a table
the view does not read is a result recomputed, never a stale one). The tables the identities name are returned too, for the cache to hold: while an
entry lives no other table can take a dead one's address.

What is left out on purpose: `PlannerOptions`. The device (CPU, GPU, AUTO), threads, grouping strategy, join order and build side, fusing and the sort
budgets change how a query runs, not which rows it returns (an option the engine cannot honour raises, and an error is never kept); only the order of
rows a query leaves unordered may differ, and then the first result kept is the one every read gets. FROM names a file or a table, never a glob or a
folder in this version, so a file's stamp is its own size and mtime.
"""
from std.os import stat
from .sql import lex_sql, T_EOF, T_IDENT, T_QIDENT, T_STR, lower_ascii
from .bind import Catalog, SqlSource
from .table import SharedTable


def canonical_text(text: String) raises -> String:
    """The SQL as its tokens, one space apart: comments and spacing do not matter, anything else does."""
    var toks = lex_sql(text)
    var out = String()
    for i in range(len(toks)):
        ref t = toks[i]
        if t.kind == T_EOF:
            break
        if t.kind == T_QIDENT:
            out += '"' + t.text.replace('"', '""') + '" '
        elif t.kind == T_STR:
            out += "'" + t.text.replace("'", "''") + "' "
        else:
            out += t.text + " "
    return out^


def file_stamp(path: String) -> String:
    """`path@size:mtime`, or empty when there is no such file."""
    try:
        var s = stat(path)
        return path + "@" + String(s.st_size) + ":" + String(s.st_mtimespec.as_nanoseconds())
    except:
        return String()


def source_stamp(src: SqlSource, mut inputs: List[SharedTable]) -> String:
    inputs.append(src.table)
    var out = "table#" + String(Int(UnsafePointer(to=src.table._table[]))) + " rows=" + String(src.table.rows()) + " v" + String(src.version)
    if src.scan:
        out += " " + file_stamp(src.scan.value()._info[].path)
    return out^


def _seen(names: List[String], key: String) -> Bool:
    for i in range(len(names)):
        if names[i] == key:
            return True
    return False


def view_key(cat: Catalog, vi: Int, mut inputs: List[SharedTable]) raises -> String:
    """The key of view `vi`, read against the sources and the views before it (a view reads no later one)."""
    var text = cat.view_text(vi)
    var parts = List[String]()
    var seen = List[String]()
    var toks = lex_sql(text)
    for i in range(len(toks)):
        ref t = toks[i]
        if t.kind == T_IDENT or t.kind == T_QIDENT:
            var key = lower_ascii(t.text)
            if _seen(seen, key):
                continue
            var up = cat.view_index(t.text)
            if up >= 0 and up < vi:
                seen.append(key)
                parts.append("view " + key + "=[" + view_key(cat, up, inputs) + "]")
                continue
            var si = cat.find(t.text)
            if si >= 0:
                seen.append(key)
                parts.append("source " + key + "=" + source_stamp(cat.sources[si], inputs))
        elif t.kind == T_STR:
            var f = file_stamp(t.text)
            if f.byte_length() > 0 and not _seen(seen, f):
                seen.append(f)
                parts.append("file " + f)
    var out = canonical_text(text) + "|"
    for i in range(len(parts)):
        out += (";" if i > 0 else "") + parts[i]
    return out^
