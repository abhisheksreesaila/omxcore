"""Window functions from the builder (F-168): `compute_windows(query)`.

    var w = Window().partition_by(col("region")).order_by(col("day")).rows(-2, CURRENT_ROW)
    var q = Query(orders).where(col("qty") > 0).window(row_number().over(Window().partition_by(col("region")).order_by(col("day"))),
                                                          sum(col("price")).over(w).alias("moving"))
    var r = compute_windows(q)            # r.table: the source's columns for the rows `where` keeps, then one column per function
    r.device; r.fallback_reason; r.notes; r.ms

The query's rows are those its `where`s keep, over a table, a Frame (its edits and formula columns) or a Parquet scan (which materialises the columns the
functions and the source's other columns need: the result holds them all). The plan is written as SQL (the expressions as text, columns quoted) and runs
through the SQL path, so the answers are the SQL path's (`winsql.mojo`, `window.mojo`) and the builder has no second implementation. `device`: the CPU runs
windows; AUTO answers on the CPU with `fallback_reason` saying so, and GPU raises. A Query with a group_by or an aggregate takes no windows (write that in SQL).
"""
from .table import Table
from .expr import Expr, Node, N_COL, N_LIT, N_LITF, N_DIV, N_ADD, N_SUB, N_MUL, N_FLOORDIV, N_MOD, N_NEG, N_LEAST, N_GREATEST, N_LT, N_LE, N_GT, N_GE, N_EQ, N_NE, N_AND, N_OR, N_NOT
from .floatfmt import float_text
from .query import Query, P_FILTER, P_AGGREGATE, CPU, GPU, AUTO
from .bind import Catalog, SqlSource, SRC_TABLE, SRC_FRAME, SRC_SCAN
from .planner import PlannerOptions
from .sqlexec import SqlResult
from .sqlquery import sql
from .sql import FM_NONE, FM_ROWS
from .window import (WindowFn, Window, UNBOUNDED_PRECEDING, UNBOUNDED_FOLLOWING, W_ROW_NUMBER, W_RANK, W_DENSE_RANK, W_PERCENT_RANK, W_CUME_DIST, W_NTILE, W_LAG,
                     W_LEAD, W_FIRST, W_LAST, W_NTH, W_SUM, W_COUNT, W_COUNT_STAR, W_AVG, W_MIN, W_MAX)

comptime SOURCE_NAME = "__query"


def _quote(name: String) -> String:
    return "\"" + name.replace("\"", "\"\"") + "\""


def _sql_of(e: Expr) raises -> String:
    """The expression as SQL text, fully parenthesised (the nodes are in post-order: a stack of texts)."""
    var stack = List[String]()
    for n in e.nodes:
        if n.op == N_COL:
            stack.append(_quote(n.name))
        elif n.op == N_LIT:
            stack.append("(" + String(n.value) + ")" if n.value < 0 else String(n.value))
        elif n.op == N_LITF:
            stack.append("CAST('" + float_text(Float64(from_bits=UInt64(n.value))) + "' AS DOUBLE)")
        elif n.op == N_NEG or n.op == N_NOT:
            var a = stack.pop()
            stack.append("(" + ("-" if n.op == N_NEG else "NOT ") + a + ")")
        else:
            var b = stack.pop()
            var a = stack.pop()
            var op = String()
            if n.op == N_ADD:
                op = "+"
            elif n.op == N_SUB:
                op = "-"
            elif n.op == N_MUL:
                op = "*"
            elif n.op == N_DIV:
                op = "/"
            elif n.op == N_FLOORDIV:
                op = "//"
            elif n.op == N_MOD:
                op = "%"
            elif n.op == N_LT:
                op = "<"
            elif n.op == N_LE:
                op = "<="
            elif n.op == N_GT:
                op = ">"
            elif n.op == N_GE:
                op = ">="
            elif n.op == N_EQ:
                op = "="
            elif n.op == N_NE:
                op = "<>"
            elif n.op == N_AND:
                op = "AND"
            elif n.op == N_OR:
                op = "OR"
            elif n.op == N_LEAST:
                stack.append("(CASE WHEN " + a + " <= " + b + " THEN " + a + " ELSE " + b + " END)")
                continue
            elif n.op == N_GREATEST:
                stack.append("(CASE WHEN " + a + " >= " + b + " THEN " + a + " ELSE " + b + " END)")
                continue
            else:
                raise Error("this expression (an internal node, op " + String(n.op) + ") cannot be written for a window query")
            stack.append("(" + a + " " + op + " " + b + ")")
    if len(stack) != 1:
        raise Error("an expression with no value cannot be written for a window query")
    return stack.pop()


def _bound(v: Int, start: Bool) -> String:
    if v <= UNBOUNDED_PRECEDING:
        return "UNBOUNDED PRECEDING"
    if v >= UNBOUNDED_FOLLOWING:
        return "UNBOUNDED FOLLOWING"
    if v == 0:
        return "CURRENT ROW"
    return String(-v) + " PRECEDING" if v < 0 else String(v) + " FOLLOWING"


def _window_sql(w: Window) raises -> String:
    var s = String()
    if len(w.partition) > 0:
        s += "PARTITION BY "
        for i in range(len(w.partition)):
            s += (", " if i > 0 else "") + _sql_of(w.partition[i])
    if len(w.order) > 0:
        s += (" " if s.byte_length() > 0 else "") + "ORDER BY "
        for i in range(len(w.order)):
            s += (", " if i > 0 else "") + _sql_of(w.order[i]) + (" DESC" if w.order_desc[i] else "") + (" NULLS FIRST" if w.order_nulls_first[i] else " NULLS LAST")
    if w.mode != FM_NONE:
        s += (" " if s.byte_length() > 0 else "") + ("ROWS" if w.mode == FM_ROWS else "RANGE") + " BETWEEN " + _bound(w.start, True) + " AND " + _bound(w.end, False)
    return s


def _call_sql(f: WindowFn) raises -> String:
    var k = f.kind
    if k == W_ROW_NUMBER:
        return "row_number()"
    if k == W_RANK:
        return "rank()"
    if k == W_DENSE_RANK:
        return "dense_rank()"
    if k == W_PERCENT_RANK:
        return "percent_rank()"
    if k == W_CUME_DIST:
        return "cume_dist()"
    if k == W_NTILE:
        return "ntile(" + String(f.n) + ")"
    if k == W_LAG or k == W_LEAD:
        return ("lag(" if k == W_LAG else "lead(") + _sql_of(f.arg) + ", " + String(f.n) + ((", " + _sql_of(f.default)) if f.has_default else "") + ")"
    if k == W_FIRST:
        return "first_value(" + _sql_of(f.arg) + ")"
    if k == W_LAST:
        return "last_value(" + _sql_of(f.arg) + ")"
    if k == W_NTH:
        return "nth_value(" + _sql_of(f.arg) + ", " + String(f.n) + ")"
    if k == W_SUM:
        return "sum(" + _sql_of(f.arg) + ")"
    if k == W_COUNT:
        return "count(" + _sql_of(f.arg) + ")"
    if k == W_COUNT_STAR:
        return "count(*)"
    if k == W_AVG:
        return "avg(" + _sql_of(f.arg) + ")"
    if k == W_MIN:
        return "min(" + _sql_of(f.arg) + ")"
    return "max(" + _sql_of(f.arg) + ")"


def window_sql(q: Query) raises -> String:
    """The SQL text of a Query with `window(...)` functions: `SELECT *, f(...) OVER (...) AS name, ... FROM __query [WHERE ...]`."""
    if len(q.windows) == 0:
        raise Error("the query has no window functions: add some with Query.window(...)")
    var text = String("SELECT *")
    for i in range(len(q.windows)):
        ref f = q.windows[i]
        if not f.has_over:
            raise Error("window function " + String(i) + " has no window: write `.over(Window()...)`")
        text += ", " + _call_sql(f) + " OVER (" + _window_sql(f.win) + ") AS " + _quote(f.name if f.name.byte_length() > 0 else "w" + String(i))
    text += " FROM " + SOURCE_NAME
    var where = String()
    for node in q.nodes:
        if node.kind == P_AGGREGATE:
            raise Error("a Query with a group_by or an aggregate takes no window functions here: write the query in SQL (`sql()`)")
        if node.kind == P_FILTER:
            where += (" AND " if where.byte_length() > 0 else "") + _sql_of(node.exprs[0])
    if where.byte_length() > 0:
        text += " WHERE " + where
    return text^


def compute_windows(q: Query, device: Int = AUTO) raises -> SqlResult:
    """Runs a Query's window functions (see the module doc): `.table` holds the rows' source columns and the functions' columns. `device` AUTO / CPU run on
    the CPU (`fallback_reason` says so under AUTO); GPU raises."""
    if device == GPU:
        raise Error("window functions run on the CPU only: use device=CPU or AUTO")
    var text = window_sql(q)
    var cat = Catalog()
    var src = SqlSource(SOURCE_NAME, q.table)
    if q.scan:
        src.kind = SRC_SCAN
        src.scan = q.scan.copy()
    elif q.framed:
        src.kind = SRC_FRAME
    src.overlay = q.overlay.copy()
    src.framed = q.framed
    cat._put(src^)
    var opts = PlannerOptions()
    opts.threads = q._threads
    var r = sql(cat, text, opts)
    r.fallback_reason = "window functions run on the CPU" if device == AUTO else String()
    r.notes.append("builder: " + text)
    return r^
