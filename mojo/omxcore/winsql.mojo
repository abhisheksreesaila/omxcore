"""SQL window functions (F-168): the rewrite that turns `SELECT ..., f(x) OVER (...) FROM ...` into two plain queries and a window step.

    inner:  SELECT <every window-free part the select list, the windows and ORDER BY read> FROM ... WHERE ... [GROUP BY ... HAVING ...]
    window: one column per window function over the inner result (`window.mojo`), aligned with its rows
    outer:  SELECT <the original select list, windows replaced by their columns> FROM __win [ORDER BY ...] [LIMIT ...]

The inner query is whatever the engine runs (a table, a frame's edits, a Parquet or CSV scan, a join, a GROUP BY: the window then runs over the
grouped result), so a scan materialises the columns it needs (streaming a window bigger than memory is F-167's work). The outer query is an
ordinary row query over the inner result: DISTINCT, ORDER BY, LIMIT and OFFSET apply after the windows, as in SQL. Each maximal window-free
subexpression becomes one inner column (`__w0`, `__w1`, ...), so `sum(x * 2) OVER (PARTITION BY g)` computes `x * 2` and `g` once, in the kernel.

Refused by name: a window function in WHERE, GROUP BY, HAVING or a join condition; `*` beside a window over a join; DISTINCT, FILTER, IGNORE NULLS
and a named window; GROUPS frames, RANGE with an offset and EXCLUDE (the parser); a window function that is not in the list of `window.mojo`.
"""
from .sql import (SqlStatement, SqlExpr, SelectItem, OrderItem, WinSpec, sql_error, lower_ascii, upper_ascii, EXPLAIN_NONE,
                  X_INT, X_DEC, X_STR, X_NULL, X_TRUE, X_FALSE, X_DATE, X_COL, X_NEG, X_BIN, X_NOT, X_ISNULL, X_BETWEEN, X_IN, X_CALL, X_FLT, X_SUBIN, X_EXISTS,
                  X_CASE, X_CAST, X_LIKE, X_DISTINCT, X_INTERVAL, X_ISBOOL, FM_NONE, FM_ROWS)
from .bind import Catalog, _is_aggregate_name, parse_date_literal
from .table import Table, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .window import (WinFn, WinGroup, WinOrder, apply_windows, function_name, W_ROW_NUMBER, W_RANK, W_DENSE_RANK, W_PERCENT_RANK, W_CUME_DIST, W_NTILE, W_LAG,
                     W_LEAD, W_FIRST, W_LAST, W_NTH, W_SUM, W_COUNT, W_COUNT_STAR, W_AVG, W_MIN, W_MAX)
from .sqleval import float_bits

comptime WIN_TABLE = "__win"
comptime WINDOW_LIST = "row_number, rank, dense_rank, percent_rank, cume_dist, ntile, lag, lead, first_value, last_value, nth_value, sum, count, avg, min, max"


struct _Default(Copyable, Movable):
    """A lag / lead default as the SQL wrote it; it is read once the argument's column type is known."""
    var group: Int
    var fn_at: Int
    var kind: Int
    var text: String
    var neg: Bool
    var line: Int
    var col: Int

    def __init__(out self, group: Int, fn_at: Int, kind: Int, text: String, neg: Bool, line: Int, col: Int):
        self.group = group
        self.fn_at = fn_at
        self.kind = kind
        self.text = text
        self.neg = neg
        self.line = line
        self.col = col


struct WindowPlan(Movable):
    var inner: SqlStatement
    var outer: SqlStatement
    var groups: List[WinGroup]
    var defaults: List[_Default]
    var descriptions: List[String]       # one per function, for the notes

    def __init__(out self):
        self.inner = SqlStatement(String())
        self.outer = SqlStatement(String())
        self.groups = List[WinGroup]()
        self.defaults = List[_Default]()
        self.descriptions = List[String]()


def _uses_a(k: Int) -> Bool:
    return k == X_NEG or k == X_NOT or k == X_ISNULL or k == X_ISBOOL or k == X_INTERVAL or k == X_BIN or k == X_DISTINCT or k == X_BETWEEN or \
           k == X_IN or k == X_LIKE or k == X_CASE or k == X_CAST


def _uses_b(k: Int) -> Bool:
    return k == X_BIN or k == X_DISTINCT or k == X_BETWEEN or k == X_LIKE


def _uses_c(k: Int) -> Bool:
    return k == X_BETWEEN or k == X_LIKE or k == X_CASE


def _uses_items(k: Int) -> Bool:
    return k == X_CALL or k == X_IN or k == X_CASE


def _is_literal(k: Int) -> Bool:
    return k == X_INT or k == X_DEC or k == X_FLT or k == X_STR or k == X_NULL or k == X_TRUE or k == X_FALSE or k == X_DATE


def _children(x: SqlExpr) -> List[Int]:
    """The expressions a node reads (the arguments and, for a window call, its PARTITION BY and ORDER BY are added by the caller)."""
    var out = List[Int]()
    if _uses_a(x.kind) and x.a >= 0:
        out.append(x.a)
    if _uses_b(x.kind) and x.b >= 0:
        out.append(x.b)
    if _uses_c(x.kind) and x.c >= 0:
        out.append(x.c)
    if _uses_items(x.kind):
        for it in x.items:
            out.append(it)
    return out^


def _mark(st: SqlStatement, e: Int, mut seen: List[Bool]):
    seen[e] = True
    ref x = st.exprs[e]
    for ch in _children(x):
        _mark(st, ch, seen)
    if x.kind == X_CALL and x.c >= 0:
        ref w = st.wins[x.c]
        for p in w.partition:
            _mark(st, p, seen)
        for o in w.order:
            _mark(st, o.expr, seen)


def _function_kind(x: SqlExpr) raises -> Int:
    var up = x.text
    if up == "ROW_NUMBER":
        return W_ROW_NUMBER
    if up == "RANK":
        return W_RANK
    if up == "DENSE_RANK":
        return W_DENSE_RANK
    if up == "PERCENT_RANK":
        return W_PERCENT_RANK
    if up == "CUME_DIST":
        return W_CUME_DIST
    if up == "NTILE":
        return W_NTILE
    if up == "LAG":
        return W_LAG
    if up == "LEAD":
        return W_LEAD
    if up == "FIRST_VALUE" or up == "FIRST":
        return W_FIRST
    if up == "LAST_VALUE" or up == "LAST":
        return W_LAST
    if up == "NTH_VALUE":
        return W_NTH
    if up == "SUM":
        return W_SUM
    if up == "COUNT":
        return W_COUNT_STAR if x.a == 1 else W_COUNT
    if up == "AVG" or up == "MEAN":
        return W_AVG
    if up == "MIN":
        return W_MIN
    if up == "MAX":
        return W_MAX
    raise sql_error(x.line, x.col, "`" + x.text.lower() + "` is not supported as a window function (the window functions are " + WINDOW_LIST + ")")


struct _Rewrite(Movable):
    var st: SqlStatement                 # the original statement: it becomes the inner query
    var out: SqlStatement                # the outer query
    var inner_items: List[SelectItem]
    var inner_keys: List[String]
    var plan: WindowPlan
    var fn_count: Int

    def __init__(out self, var st: SqlStatement):
        self.out = SqlStatement(st.text)
        self.st = st^
        self.inner_items = List[SelectItem]()
        self.inner_keys = List[String]()
        self.plan = WindowPlan()
        self.fn_count = 0

    def has_window(self, e: Int) -> Bool:
        ref x = self.st.exprs[e]
        if x.kind == X_CALL and x.c >= 0:
            return True
        for ch in _children(x):
            if self.has_window(ch):
                return True
        return False

    def add_out(mut self, var x: SqlExpr) -> Int:
        self.out.exprs.append(x^)
        return len(self.out.exprs) - 1

    def out_col(mut self, name: String, line: Int, col: Int) -> Int:
        return self.add_out(SqlExpr(X_COL, name, String(), -1, -1, -1, List[Int](), False, line, col))

    def slot(mut self, e: Int) -> Int:
        """The inner column that holds window-free expression `e` (made on first use)."""
        var key = self.st.render(e)
        for i in range(len(self.inner_keys)):
            if self.inner_keys[i] == key:
                return i
        ref x = self.st.exprs[e]
        self.inner_keys.append(key)
        self.inner_items.append(SelectItem(e, "__w" + String(len(self.inner_items)), True, x.line, x.col))
        return len(self.inner_items) - 1

    def column_of(mut self, e: Int, what: String) raises -> Int:
        ref x = self.st.exprs[e]
        if self.has_window(e):
            raise sql_error(x.line, x.col, "a window function cannot be inside " + what + " of another window function")
        return self.slot(e)

    def extract(mut self, e: Int) raises -> Int:
        """The outer expression for `e`: a window call becomes its column, a window-free part an inner column, anything else is rebuilt."""
        var x = self.st.exprs[e].copy()
        if x.kind == X_CALL and x.c >= 0:
            return self.add_window(e)
        if _is_literal(x.kind):
            return self.add_out(x^)
        if not self.has_window(e):
            return self.out_col("__w" + String(self.slot(e)), x.line, x.col)
        if x.kind == X_SUBIN or x.kind == X_EXISTS:
            raise sql_error(x.line, x.col, "a subquery next to a window function is not supported")
        if x.kind == X_CALL and _is_aggregate_name(x.text):
            raise sql_error(x.line, x.col, "a window function cannot be inside the aggregate " + x.text.lower() + "(...)")
        var nx = x.copy()
        if _uses_a(x.kind) and x.a >= 0:
            nx.a = self.extract(x.a)
        if _uses_b(x.kind) and x.b >= 0:
            nx.b = self.extract(x.b)
        if _uses_c(x.kind) and x.c >= 0:
            nx.c = self.extract(x.c)
        if _uses_items(x.kind):
            for i in range(len(x.items)):
                nx.items[i] = self.extract(x.items[i])
        return self.add_out(nx^)

    def literal_int(self, e: Int, what: String) raises -> Int:
        ref x = self.st.exprs[e]
        var neg = False
        var at = e
        if x.kind == X_NEG:
            neg = True
            at = x.a
        ref y = self.st.exprs[at]
        if y.kind != X_INT or y.text.byte_length() > 15:
            raise sql_error(x.line, x.col, what + " must be a whole-number literal")
        return -Int(y.text) if neg else Int(y.text)

    def add_window(mut self, e: Int) raises -> Int:
        var x = self.st.exprs[e].copy()
        var w = self.st.wins[x.c].copy()
        var kind = _function_kind(x)
        if x.flag:
            raise sql_error(x.line, x.col, "DISTINCT inside a window function (" + x.text.lower() + "(DISTINCT ...) OVER) is not supported")
        var nargs = len(x.items)
        var fname = function_name(kind)
        var f = WinFn(kind, "__wf" + String(self.fn_count))
        var has_dflt = False
        var dkind = 0
        var dtext = String()
        var dneg = False
        var dline = 0
        var dcol = 0
        var lo = 0
        var hi = 0
        if kind == W_ROW_NUMBER or kind == W_RANK or kind == W_DENSE_RANK or kind == W_PERCENT_RANK or kind == W_CUME_DIST:
            lo = 0
            hi = 0
        elif kind == W_NTILE or kind == W_FIRST or kind == W_LAST or kind == W_SUM or kind == W_COUNT or kind == W_AVG or kind == W_MIN or kind == W_MAX:
            lo = 1
            hi = 1
        elif kind == W_NTH:
            lo = 2
            hi = 2
        elif kind == W_LAG or kind == W_LEAD:
            lo = 1
            hi = 3
        elif kind == W_COUNT_STAR:
            lo = 0
            hi = 0
        if nargs < lo or nargs > hi:
            raise sql_error(x.line, x.col, fname + "() takes " + (String(lo) if lo == hi else String(lo) + " to " + String(hi)) + " argument" + ("" if hi == 1 else "s") + ", found " + String(nargs))
        # arguments
        if kind == W_NTILE:
            f.n = self.literal_int(x.items[0], "the bucket count of ntile")
            if f.n < 1:
                raise sql_error(x.line, x.col, "ntile needs at least one bucket")
        elif kind == W_NTH:
            f.arg = self.column_of(x.items[0], "the argument")
            f.n = self.literal_int(x.items[1], "the position of nth_value")
            if f.n < 1:
                raise sql_error(x.line, x.col, "nth_value counts from 1")
        elif kind != W_ROW_NUMBER and kind != W_RANK and kind != W_DENSE_RANK and kind != W_PERCENT_RANK and kind != W_CUME_DIST and kind != W_COUNT_STAR:
            f.arg = self.column_of(x.items[0], "the argument")
        if kind == W_LAG or kind == W_LEAD:
            var off = 1
            if nargs >= 2:
                off = self.literal_int(x.items[1], "the offset of " + fname)
            f.n = -off if kind == W_LAG else off
            if nargs == 3:
                f.has_default = True
                var d = x.items[2]
                var neg = False
                if self.st.exprs[d].kind == X_NEG:
                    neg = True
                    d = self.st.exprs[d].a
                ref dx = self.st.exprs[d]
                if not _is_literal(dx.kind) or (neg and dx.kind != X_INT and dx.kind != X_DEC and dx.kind != X_FLT):
                    raise sql_error(dx.line, dx.col, "the default of " + fname + " must be a literal")
                if dx.kind != X_NULL:
                    f.def_null = False
                    has_dflt = True
                    dkind = dx.kind
                    dtext = dx.text
                    dneg = neg
                    dline = dx.line
                    dcol = dx.col
        f.mode = w.mode
        f.s_kind = w.s_kind
        f.s_n = w.s_n
        f.e_kind = w.e_kind
        f.e_n = w.e_n
        # the sort it shares
        var part = List[Int]()
        for p in w.partition:
            part.append(self.column_of(p, "PARTITION BY"))
        var order = List[WinOrder]()
        for o in w.order:
            order.append(WinOrder(self.column_of(o.expr, "ORDER BY"), o.desc, o.nulls == 1))
        var g = -1
        for i in range(len(self.plan.groups)):
            ref cand = self.plan.groups[i]
            if len(cand.part) != len(part) or len(cand.order) != len(order):
                continue
            var same = True
            for j in range(len(part)):
                if cand.part[j] != part[j]:
                    same = False
            for j in range(len(order)):
                if cand.order[j].col != order[j].col or cand.order[j].desc != order[j].desc or cand.order[j].nulls_first != order[j].nulls_first:
                    same = False
            if same:
                g = i
                break
        if g < 0:
            var ng = WinGroup()
            ng.part = part^
            ng.order = order^
            self.plan.groups.append(ng^)
            g = len(self.plan.groups) - 1
        if has_dflt:
            self.plan.defaults.append(_Default(g, len(self.plan.groups[g].fns), dkind, dtext, dneg, dline, dcol))
        self.plan.groups[g].fns.append(f^)
        self.plan.descriptions.append(self.st.render(e))
        var name = "__wf" + String(self.fn_count)
        self.fn_count += 1
        return self.out_col(name, x.line, x.col)


def plan_windows(var st: SqlStatement, catalog: Catalog) raises -> WindowPlan:
    """Splits a statement with window functions into its inner query, the window step and its outer query (module doc)."""
    # windows belong to the select list and ORDER BY only
    var seen = List[Bool](length=len(st.exprs), fill=False)
    for it in st.items:
        if it.expr >= 0:
            _mark(st, it.expr, seen)
    for o in st.order_by:
        _mark(st, o.expr, seen)
    for i in range(len(st.exprs)):
        if st.exprs[i].kind == X_CALL and st.exprs[i].c >= 0 and not seen[i]:
            raise sql_error(st.exprs[i].line, st.exprs[i].col, "window functions are only allowed in the SELECT list and ORDER BY (not in WHERE, GROUP BY, HAVING or ON)")
    # `*` becomes the table's columns (the order the query would have given them)
    var items = List[SelectItem]()
    for it in st.items:
        if not it.star:
            items.append(it.copy())
            continue
        if st.has_right or len(st.extra) > 0 or it.alias.byte_length() > 0:
            raise sql_error(it.line, it.col, "`*` next to a window function over a join (or `t.*`) is not supported: name the columns")
        var si = catalog.find(st.table)
        if si < 0:
            raise sql_error(st.table_line, st.table_col, "table `" + st.table + "` not found")
        ref src = catalog.sources[si]
        var names = List[String]()
        for c in range(src.table.columns()):
            names.append(src.table.name(c))
        for f in src.overlay.formulas:
            names.append(f.name)
        for nm in names:
            st.exprs.append(SqlExpr(X_COL, nm, String(), -1, -1, -1, List[Int](), False, it.line, it.col))
            items.append(SelectItem(len(st.exprs) - 1, String(), False, it.line, it.col))
    st.items = items^
    # GROUP BY ordinals and select aliases name select items: say what they are before the select list changes
    if len(st.group_by) > 0:
        var table_cols = List[String]()
        if not st.has_right:
            var si = catalog.find(st.table)
            if si >= 0:
                for c in range(catalog.sources[si].table.columns()):
                    table_cols.append(lower_ascii(catalog.sources[si].table.name(c)))
        for g in range(len(st.group_by)):
            ref gx = st.exprs[st.group_by[g]]
            if gx.kind == X_INT and gx.text.byte_length() < 9:
                var k = Int(gx.text)
                if k >= 1 and k <= len(st.items):
                    st.group_by[g] = st.items[k - 1].expr
            elif gx.kind == X_COL and gx.text2.byte_length() == 0 and not st.has_right and lower_ascii(gx.text) not in table_cols:
                for it in st.items:
                    if it.has_alias and lower_ascii(it.alias) == lower_ascii(gx.text):
                        st.group_by[g] = it.expr
                        break
    var orig_items = st.items.copy()
    var orig_order = st.order_by.copy()
    var limit = st.limit
    var offset = st.offset
    var distinct = st.distinct
    var rw = _Rewrite(st^)
    rw.out.table = WIN_TABLE
    rw.out.table_line = rw.st.table_line
    rw.out.table_col = rw.st.table_col
    # the select list
    for it in orig_items:
        var e = rw.extract(it.expr)
        var alias = it.alias
        if not it.has_alias:
            ref ox = rw.st.exprs[it.expr]
            alias = ox.text if (ox.kind == X_COL) else rw.st.render(it.expr)
        rw.out.items.append(SelectItem(e, alias, True, it.line, it.col))
    # ORDER BY: ordinals and select aliases stay as written, anything else is an expression over the inner columns
    for o in orig_order:
        ref ox = rw.st.exprs[o.expr]
        var keep = ox.kind == X_INT
        if ox.kind == X_COL and ox.text2.byte_length() == 0:
            for it in rw.out.items:
                if lower_ascii(it.alias) == lower_ascii(ox.text):
                    keep = True
        var e = o.expr
        if keep:
            e = rw.add_out(ox.copy())
        else:
            e = rw.extract(o.expr)
        rw.out.order_by.append(OrderItem(e, o.desc, o.nulls))
    rw.out.limit = limit
    rw.out.offset = offset
    rw.out.distinct = distinct
    # the inner query: the same FROM / WHERE / GROUP BY / HAVING, selecting the columns the rest reads
    if len(rw.inner_items) == 0:                                       # (only row_number() and the like: any column says how many rows there are)
        rw.st.exprs.append(SqlExpr(X_INT, "1", String(), -1, -1, -1, List[Int](), False, rw.st.items[0].line, rw.st.items[0].col))
        rw.inner_items.append(SelectItem(len(rw.st.exprs) - 1, "__w0", True, rw.st.items[0].line, rw.st.items[0].col))
    rw.st.items = rw.inner_items.copy()
    rw.st.order_by = List[OrderItem]()
    rw.st.limit = -1
    rw.st.offset = 0
    rw.st.distinct = False
    rw.st.explain = EXPLAIN_NONE
    var plan = WindowPlan()
    swap(plan, rw.plan)
    swap(plan.inner, rw.st)
    swap(plan.outer, rw.out)
    return plan^


# ── the window step ──────────────────────────────────────────────────────────────────────────────────

def _read_default(var d: _Default, mut f: WinFn, t: Table) raises:
    ref ac = t.cols[f.arg]
    var k = ac.kind.value
    var what = String("a ") + String(ac.kind) + " value"
    if d.kind == X_NULL:
        return
    if k == TEXT.value:
        if d.kind != X_STR:
            raise sql_error(d.line, d.col, "the default must be text, like the column it follows")
        f.def_is_text = True
        f.def_text = d.text
        return
    if k == DATE.value:
        var days = parse_date_literal(d.text) if (d.kind == X_DATE or d.kind == X_STR) else Int.MIN
        if days == Int.MIN:
            raise sql_error(d.line, d.col, "the default must be a date, like the column it follows")
        f.def_val = Int64(days)
        return
    if k == BOOL.value:
        if d.kind != X_TRUE and d.kind != X_FALSE:
            raise sql_error(d.line, d.col, "the default must be true or false, like the column it follows")
        f.def_val = 1 if d.kind == X_TRUE else 0
        return
    if d.kind != X_INT and d.kind != X_DEC and d.kind != X_FLT:
        raise sql_error(d.line, d.col, "the default must be a number, like the column it follows")
    if k == FLOAT.value:
        var v = atof(d.text)
        f.def_val = float_bits(-v if d.neg else v)
        return
    var scale = ac.scale if k == DECIMAL.value else 0
    if d.kind == X_FLT:
        raise sql_error(d.line, d.col, "the default has an exponent, which " + what + " cannot hold")
    var digits = String()
    var frac = 0
    var seen_dot = False
    for b in d.text.as_bytes():
        if b == UInt8(46):
            seen_dot = True
        else:
            digits += chr(Int(b))
            if seen_dot:
                frac += 1
    if frac > scale:
        raise sql_error(d.line, d.col, "the default has more decimal places than " + what + " holds (" + String(scale) + ")")
    for _ in range(scale - frac):
        digits += "0"
    if digits.byte_length() > 18:
        raise sql_error(d.line, d.col, "the default is too large for " + what)
    var v = Int64(Int(digits))
    f.def_val = -v if d.neg else v


def finish_windows(mut plan: WindowPlan, mut t: Table, threads: Int, mut notes: List[String]) raises:
    """Adds the window columns to the inner result `t`."""
    for i in range(len(plan.defaults)):
        var d = plan.defaults[i].copy()
        _read_default(d^, plan.groups[plan.defaults[i].group].fns[plan.defaults[i].fn_at], t)
    apply_windows(t, plan.groups, threads)
    var n_fns = 0
    for g in plan.groups:
        n_fns += len(g.fns)
    notes.append("window: " + String(n_fns) + (" function" if n_fns == 1 else " functions") + " in " + String(len(plan.groups)) + (" sort" if len(plan.groups) == 1 else " sorts") +
                 " over " + String(t.rows()) + " rows, on the CPU (the rows are the inner query's result; partitions run in parallel)")
    for dsc in plan.descriptions:
        notes.append("window: " + dsc)
