"""The stage after aggregation (F-152): HAVING, the select list, ORDER BY and LIMIT as a row query over the groups.

Aggregation leaves one row per group: the group keys and the aggregates' values. What the query does with them next is the same
thing a row query does with a table's rows, so the executor runs it the same way: the groups become a small Table (`groups`),
the post-aggregate expressions (`BoundQuery.pexprs`, whose leaves are `B_KEY` and `B_AGG`) are rewritten as ordinary expressions
over its columns, HAVING becomes the WHERE, and the select list, ORDER BY, OFFSET and LIMIT run through `rows.mojo` (the kernel
program, the materialize sink, the stable radix sort or the top-k heaps). There is no second expression evaluator.

FLOAT values (AVG, `/`) are float registers in the program. An integer or decimal leaf that meets a FLOAT is read from a shadow
column (its value converted to a double here, exactly as DuckDB casts it), so an aggregate past 2^53 does not stop the kernel's
integer-to-float conversion.
"""
from .table import Table, SharedTable, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .bind import (decimal_to_float, BoundQuery, BExpr, SqlSource, SortKey, B_COL, B_LIT, B_KEY, B_AGG, B_ADD, B_SUB, B_MUL, B_DIV, B_CMP, B_NEG, B_I2F)
from .plan import build_plan
from .planner import RulePlanner, PlannerOptions, commas, Stats, plan_sinks
from .sqlprog import DictAccess
from .sqleval import ValCol, float_bits
from .exec import _put_column, _inputs, make_stats
from .rows import prepare_rows, execute_rows


def _is_number(kind: Int) -> Bool:
    return kind == INT.value or kind == DECIMAL.value


struct _Shadows(Movable):
    var of_col: List[Int]            # the groups' column that is shadowed
    var node: List[Int]              # the expression node reading its shadow

    def __init__(out self):
        self.of_col = List[Int]()
        self.node = List[Int]()


def _shadow(mut q2: BoundQuery, mut sh: _Shadows, ncols: Int, operand: Int) -> Int:
    """The node to read for `operand` in a FLOAT context (after aggregation the binder leaves integers and decimals unconverted): for a leaf
    holding an integer or decimal, a FLOAT column holding the same number converted to a double; for another integer or decimal expression, the
    kernel's conversion (`B_I2F`); a FLOAT is read as it is."""
    var leaf = q2.rexprs[operand].copy()
    if not _is_number(leaf.kind):
        return operand
    if leaf.op != B_COL:                                          # an expression: the program converts it (float(v) / 10^scale)
        q2.rexprs.append(BExpr(B_I2F, FLOAT.value, 0, operand, -1, Int64(leaf.scale), "", leaf.line, leaf.col))
        return len(q2.rexprs) - 1
    var col = Int(leaf.value)
    for i in range(len(sh.of_col)):
        if sh.of_col[i] == col:
            return sh.node[i]
    var at = ncols + len(sh.of_col)
    q2.rexprs.append(BExpr(B_COL, FLOAT.value, 0, -1, -1, Int64(at), leaf.text, leaf.line, leaf.col))
    sh.of_col.append(col)
    sh.node.append(len(q2.rexprs) - 1)
    return len(q2.rexprs) - 1


def run_post(q: BoundQuery, keys: List[ValCol], aggs: List[ValCol], dicts: DictAccess, opts: PlannerOptions, n: Int,
             mut notes: List[String]) raises -> Table:
    """The result table of `q` from its `n` groups' key and aggregate values."""
    var nk = len(keys)
    var ncols = nk + len(aggs)
    var q2 = BoundQuery()
    q2.rows = True
    q2.source = 0
    q2.text = q.text
    q2.line = q.line
    q2.col = q.col
    q2.rexprs = q.pexprs.copy()
    for i in range(len(q2.rexprs)):                                  # leaves: the groups' columns
        if q2.rexprs[i].op == B_KEY or q2.rexprs[i].op == B_AGG:
            var col = Int(q2.rexprs[i].value) + (nk if q2.rexprs[i].op == B_AGG else 0)
            q2.rexprs[i].op = B_COL
            q2.rexprs[i].value = Int64(col)
            if q2.rexprs[i].kind == TEXT.value:
                q2.rexprs[i].src = col
    var sh = _Shadows()
    for i in range(len(q2.rexprs)):                                  # numbers that meet a FLOAT are read as doubles
        var x = q2.rexprs[i].copy()
        var float_ctx = False
        if x.op == B_DIV:
            float_ctx = True
        elif (x.op == B_ADD or x.op == B_SUB or x.op == B_MUL) and x.kind == FLOAT.value:
            float_ctx = True
        elif x.op == B_CMP and (q2.rexprs[x.a].kind == FLOAT.value or q2.rexprs[x.b].kind == FLOAT.value):
            float_ctx = True
        if float_ctx:
            var a = _shadow(q2, sh, ncols, x.a)
            var b = _shadow(q2, sh, ncols, x.b)
            q2.rexprs[i].a = a
            q2.rexprs[i].b = b
    q2.where = q.having
    q2.where_text = q.having_text
    q2.routs = q.outputs.copy()
    q2.rorder = q.order.copy()
    q2.out_names = q.out_names.copy()
    q2.out_text = q.out_text.copy()
    q2.order_text = q.order_text.copy()
    q2.limit = q.limit
    q2.offset = q.offset
    # the groups as a table
    var g = Table.empty()
    for k in range(nk):
        _put_column(g, "k" + String(k), keys[k], dicts)
    for a in range(len(aggs)):
        _put_column(g, "a" + String(a), aggs[a], dicts)
    for s in range(len(sh.of_col)):
        var src_col = sh.of_col[s]
        var v = keys[src_col].copy() if src_col < nk else aggs[src_col - nk].copy()
        var f = ValCol(FLOAT.value, 0, v.n())
        for i in range(v.n()):
            f.v[i] = float_bits(decimal_to_float(v.v[i], v.scale)) if not v.is_null(i) else 0
        f.nul = v.nul.copy()
        _put_column(g, "f" + String(s), f, dicts)
    if g.columns() == 0:                                             # no key and no aggregate (`SELECT 1 HAVING true`): the rows of a table with no column are none, so a column carries the groups
        _put_column(g, "n", ValCol(INT.value, 0, n), dicts)
    var shared = g^.share()
    var src = SqlSource("groups", shared)
    var dicts2 = DictAccess(shared, None, List[List[String]]())
    var st = make_stats(src, q2, dicts2, True)
    var plan = RulePlanner().plan(q2, st, opts)
    var lp2 = build_plan(q2, "groups")
    plan_sinks(lp2, q2, st, opts, plan)
    var inp = _inputs(src, q2, True)
    var scratch = List[String]()
    var stage = prepare_rows(src, q2, plan, dicts2, inp, scratch, False)
    var table = execute_rows(src, q2, plan, opts, dicts2, scratch, inp, stage^, True)
    var selected = n
    for note in scratch:
        if note.startswith("rows: ") and " rows selected" in note:
            var digits = String()
            for b in String(note[byte=6:]).as_bytes():
                if UInt8(48) <= b <= UInt8(57):
                    digits += chr(Int(b))
                elif b != UInt8(44):
                    break
            if digits.byte_length() > 0:
                selected = Int(digits)
    if len(q.order) > 0:
        if plan.top_k > 0:
            notes.append("order: top-" + commas(plan.top_k) + " by per-worker heaps over " + commas(n) + " groups")
        else:
            notes.append("order: sorted " + commas(selected) + " groups by " + String(len(q.order)) + (" key" if len(q.order) == 1 else " keys") +
                         " (stable radix passes)")
    if q.having >= 0 and plan.top_k == 0:
        notes.append("having: " + commas(selected) + " of " + commas(n) + " groups kept")
    return table^
