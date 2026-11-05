"""Expressions and window functions written as SQL text (F-168; moved out of winquery.mojo by F-160 so `Query` can write itself as a view's SQL).

The nodes of an `Expr` are in post-order: a stack of texts, fully parenthesised; column names are quoted.
"""
from .expr import Expr, Node, N_COL, N_LIT, N_LITF, N_DIV, N_ADD, N_SUB, N_MUL, N_FLOORDIV, N_MOD, N_NEG, N_LEAST, N_GREATEST, N_LT, N_LE, N_GT, N_GE, N_EQ, N_NE, N_AND, N_OR, N_NOT
from .floatfmt import float_text
from .sql import FM_NONE, FM_ROWS
from .window import (WindowFn, Window, UNBOUNDED_PRECEDING, UNBOUNDED_FOLLOWING, W_ROW_NUMBER, W_RANK, W_DENSE_RANK, W_PERCENT_RANK, W_CUME_DIST, W_NTILE, W_LAG,
                     W_LEAD, W_FIRST, W_LAST, W_NTH, W_SUM, W_COUNT, W_COUNT_STAR, W_AVG, W_MIN, W_MAX)



def quote_name(name: String) -> String:
    return "\"" + name.replace("\"", "\"\"") + "\""


def sql_of(e: Expr) raises -> String:
    """The expression as SQL text, fully parenthesised (the nodes are in post-order: a stack of texts)."""
    var stack = List[String]()
    for n in e.nodes:
        if n.op == N_COL:
            stack.append(quote_name(n.name))
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


def window_clause(w: Window) raises -> String:
    var s = String()
    if len(w.partition) > 0:
        s += "PARTITION BY "
        for i in range(len(w.partition)):
            s += (", " if i > 0 else "") + sql_of(w.partition[i])
    if len(w.order) > 0:
        s += (" " if s.byte_length() > 0 else "") + "ORDER BY "
        for i in range(len(w.order)):
            s += (", " if i > 0 else "") + sql_of(w.order[i]) + (" DESC" if w.order_desc[i] else "") + (" NULLS FIRST" if w.order_nulls_first[i] else " NULLS LAST")
    if w.mode != FM_NONE:
        s += (" " if s.byte_length() > 0 else "") + ("ROWS" if w.mode == FM_ROWS else "RANGE") + " BETWEEN " + _bound(w.start, True) + " AND " + _bound(w.end, False)
    return s


def call_sql(f: WindowFn) raises -> String:
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
        return ("lag(" if k == W_LAG else "lead(") + sql_of(f.arg) + ", " + String(f.n) + ((", " + sql_of(f.default)) if f.has_default else "") + ")"
    if k == W_FIRST:
        return "first_value(" + sql_of(f.arg) + ")"
    if k == W_LAST:
        return "last_value(" + sql_of(f.arg) + ")"
    if k == W_NTH:
        return "nth_value(" + sql_of(f.arg) + ", " + String(f.n) + ")"
    if k == W_SUM:
        return "sum(" + sql_of(f.arg) + ")"
    if k == W_COUNT:
        return "count(" + sql_of(f.arg) + ")"
    if k == W_COUNT_STAR:
        return "count(*)"
    if k == W_AVG:
        return "avg(" + sql_of(f.arg) + ")"
    if k == W_MIN:
        return "min(" + sql_of(f.arg) + ")"
    return "max(" + sql_of(f.arg) + ")"
