"""Taking a subquery apart (F-159): the pure half. `sqlquery.mojo` runs the statements; this file reads and builds syntax trees.

A correlated subquery `K` of an outer statement `S` is decorrelated into a derived table `D` and a join:

    EXISTS (SELECT .. FROM r WHERE r.a = s.x AND p)            D = SELECT DISTINCT r.a AS __k0 FROM r WHERE p           S semi-joined with D on __k0 = s.x
    EXISTS (.. AND r.b <> s.y)  (one inequality)               D = SELECT r.a AS __k0, min(r.b) AS __lo, max(r.b) AS __hi FROM r WHERE p GROUP BY r.a
                                                               S joined with D on __k0 = s.x, kept when __lo <> s.y OR __hi <> s.y
    s.v < (SELECT agg(..) FROM r WHERE r.a = s.x AND p)        D = SELECT r.a AS __k0, agg(..) AS __v FROM r WHERE p GROUP BY r.a
                                                               S joined with D on __k0 = s.x, the subquery's place reads __v
    s.v IN (SELECT r.c FROM r WHERE r.a = s.x AND p)           D = SELECT r.a AS __k0, r.c AS __v ..., S semi-joined on __k0 = s.x AND __v = s.v

The conditions of K that read only K's tables stay in D; a correlation is an equality between an expression of K's tables and a column of S (or, once
per EXISTS, an inequality: `min` and `max` of the inner side decide whether any row satisfies it). Anything else raises by name.
"""
from .sql import (SqlStatement, SqlExpr, SelectItem, OrderItem, TableRef, SubQuery, WinSpec, sql_error, lower_ascii, closest, X_INT, X_DEC, X_STR, X_NULL,
                  X_TRUE, X_FALSE, X_DATE, X_COL, X_NEG, X_BIN, X_NOT, X_ISNULL, X_BETWEEN, X_IN, X_CALL, X_FLT, X_SUBIN, X_EXISTS, X_CASE, X_CAST,
                  X_LIKE, X_DISTINCT, X_INTERVAL, X_ISBOOL, X_SCALAR_Q, X_IN_Q, X_EXISTS_Q, EXPLAIN_NONE, SET_NONE)
from .bind import Catalog, _split_and_expr, _and_chain, _has_aggregate
from .table import Table, SharedTable, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, ColumnKind
from .sqlfn import date_text, decimal_text
from .floatfmt import float_text


# ── names: what a FROM clause shows ──────────────────────────────────────────────────────────────────

struct Scope(Copyable, Movable):
    """The tables of a FROM clause as the statement names them: the alias (or the table's name) and the columns, lower case."""
    var shown: List[String]
    var tables: List[String]
    var cols: List[List[String]]       # lower-case names
    var real: List[List[String]]       # the names as the source spells them

    def __init__(out self):
        self.shown = List[String]()
        self.tables = List[String]()
        self.cols = List[List[String]]()
        self.real = List[List[String]]()

    def entries(self) -> Int:
        return len(self.shown)


def _add_entry(mut sc: Scope, catalog: Catalog, table: String, alias: String):
    var i = catalog.find(table)
    sc.shown.append(lower_ascii(alias) if alias.byte_length() > 0 else lower_ascii(table))
    sc.tables.append(lower_ascii(table))
    var lo = List[String]()
    var re = List[String]()
    if i >= 0:
        ref src = catalog.sources[i]
        for c in range(src.table.columns()):
            re.append(src.table.name(c))
            lo.append(lower_ascii(src.table.name(c)))
        for f in src.overlay.formulas:
            re.append(f.name)
            lo.append(lower_ascii(f.name))
    sc.cols.append(lo^)
    sc.real.append(re^)


def scope_of(st: SqlStatement, catalog: Catalog) -> Scope:
    var sc = Scope()
    _add_entry(sc, catalog, st.table, st.table_alias)
    if st.has_right:
        _add_entry(sc, catalog, st.right_table, st.right_alias)
    for ex in st.extra:
        _add_entry(sc, catalog, ex.name, ex.alias)
    return sc^


def find_entry(sc: Scope, qualifier: String, name: String) -> Tuple[Int, Int]:
    """(entry, column) of a column reference in the scope; entry -1 when it names nothing here, -2 when a bare name is in more than one table."""
    var key = lower_ascii(name)
    if qualifier.byte_length() > 0:
        var q = lower_ascii(qualifier)
        for e in range(sc.entries()):
            if sc.shown[e] == q:                                                # (an alias hides the table's own name)
                for c in range(len(sc.cols[e])):
                    if sc.cols[e][c] == key:
                        return (e, c)
                return (-1, -1)
        return (-1, -1)
    var found = -1
    var at = -1
    for e in range(sc.entries()):
        for c in range(len(sc.cols[e])):
            if sc.cols[e][c] == key:
                if found >= 0 and found != e:
                    return (-2, -1)
                found = e
                at = c
                break
    return (found, at)


# ── walking an expression ────────────────────────────────────────────────────────────────────────────

def _uses_a(k: Int) -> Bool:
    return k == X_NEG or k == X_NOT or k == X_ISNULL or k == X_ISBOOL or k == X_INTERVAL or k == X_BIN or k == X_DISTINCT or k == X_BETWEEN or \
           k == X_IN or k == X_LIKE or k == X_CASE or k == X_CAST or k == X_IN_Q or k == X_SUBIN


def _uses_b(k: Int) -> Bool:
    return k == X_BIN or k == X_DISTINCT or k == X_BETWEEN or k == X_LIKE


def _uses_c(k: Int) -> Bool:
    return k == X_BETWEEN or k == X_LIKE or k == X_CASE


def _uses_items(k: Int) -> Bool:
    return k == X_CALL or k == X_IN or k == X_CASE


def children_of(st: SqlStatement, e: Int) -> List[Int]:
    """The expressions an expression reads (a call's window clause included, a subquery's statement not)."""
    ref x = st.exprs[e]
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
    if x.kind == X_CALL and x.c >= 0:
        ref w = st.wins[x.c]
        for p in w.partition:
            out.append(p)
        for o in w.order:
            out.append(o.expr)
    return out^


def collect(st: SqlStatement, e: Int, mut out: List[Int]):
    """Every node of the expression, itself first."""
    if e < 0:
        return
    out.append(e)
    for ch in children_of(st, e):
        collect(st, ch, out)


def is_subquery_node(k: Int) -> Bool:
    return k == X_SCALAR_Q or k == X_IN_Q or k == X_EXISTS_Q


def contains_subquery(st: SqlStatement, e: Int) -> Bool:
    var nodes = List[Int]()
    collect(st, e, nodes)
    for n in nodes:
        if is_subquery_node(st.exprs[n].kind):
            return True
    return False


def copy_expr(src: SqlStatement, e: Int, mut dst: SqlStatement) -> Int:
    """The expression (a call without a window clause, no subquery) copied into another statement's pool; returns its index there."""
    var x = src.exprs[e].copy()
    var node = SqlExpr(x.kind, x.text, x.text2, x.a, x.b, x.c, List[Int](), x.flag, x.line, x.col)
    if _uses_a(x.kind) and x.a >= 0:
        node.a = copy_expr(src, x.a, dst)
    if _uses_b(x.kind) and x.b >= 0:
        node.b = copy_expr(src, x.b, dst)
    if _uses_c(x.kind) and x.c >= 0:
        node.c = copy_expr(src, x.c, dst)
    if _uses_items(x.kind):
        for it in x.items:
            node.items.append(copy_expr(src, it, dst))
    else:
        node.items = x.items.copy()
    dst.exprs.append(node^)
    return len(dst.exprs) - 1


def add_node(mut st: SqlStatement, kind: Int, text: String, a: Int, b: Int, line: Int, col: Int, qual: String = String()) -> Int:
    st.exprs.append(SqlExpr(kind, text, qual, a, b, -1, List[Int](), False, line, col))
    return len(st.exprs) - 1


def column_node(mut st: SqlStatement, name: String, qual: String, line: Int, col: Int) -> Int:
    return add_node(st, X_COL, name, -1, -1, line, col, qual)


def equal_node(mut st: SqlStatement, a: Int, b: Int, line: Int, col: Int) -> Int:
    return add_node(st, X_BIN, "=", a, b, line, col)


def and_node(mut st: SqlStatement, a: Int, b: Int) -> Int:
    if a < 0:
        return b
    if b < 0:
        return a
    return add_node(st, X_BIN, "AND", a, b, st.exprs[b].line, st.exprs[b].col)


# ── a cell as a literal ──────────────────────────────────────────────────────────────────────────────

def set_typed_null(mut st: SqlStatement, node: Int, k: ColumnKind, scale: Int, line: Int, col: Int):
    """Overwrites `node` with CAST(NULL AS <the type>): a NULL that has the subquery's column type."""
    st.exprs.append(SqlExpr(X_NULL, String("NULL"), String(), -1, -1, -1, List[Int](), False, line, col))
    var operand = len(st.exprs) - 1
    var name = String("BIGINT")
    var b = -1
    var c = -1
    if k == DECIMAL:
        name = "DECIMAL"
        b = 18
        c = scale
    elif k == FLOAT:
        name = "DOUBLE"
    elif k == TEXT:
        name = "VARCHAR"
    elif k == DATE:
        name = "DATE"
    elif k == BOOL:
        name = "BOOLEAN"
    st.exprs[node] = SqlExpr(X_CAST, name, String(), operand, b, c, List[Int](), False, line, col)


def set_literal(mut st: SqlStatement, node: Int, t: Table, row: Int, c: Int) raises:
    """Overwrites `node` (a subquery) with the value of cell (row, c) of a result table as a literal of its type."""
    var line = st.exprs[node].line
    var col = st.exprs[node].col
    if t.is_null(c, row):
        set_typed_null(st, node, t.kind(c), t.scale(c), line, col)
        return
    var k = t.kind(c)
    var v = t.get(c, row)
    var kind = X_INT
    var text = String()
    var neg = False
    if k == INT:
        neg = v < 0
        text = String(UInt64(v) if not neg else UInt64(0) - UInt64(v))
    elif k == DECIMAL:
        neg = v < 0
        text = decimal_text(v if not neg else Int64(0) - v, t.scale(c))
        kind = X_DEC if t.scale(c) > 0 else X_INT
    elif k == FLOAT:
        var f = t.get_float(c, row)
        if f != f or f == Float64.MAX * 2.0 or f == -Float64.MAX * 2.0:
            raise sql_error(line, col, "the subquery's value is " + String(f) + ", which has no SQL literal")
        neg = f < 0.0 or (f == 0.0 and UInt64(f.to_bits()) >> 63 != 0)
        text = float_text(-f if neg else f)
        if text.find(".") < 0 and text.find("e") < 0:
            text += ".0"
        kind = X_FLT
    elif k == TEXT:
        kind = X_STR
        text = t.text_of(c, Int(v))
    elif k == DATE:
        kind = X_DATE
        text = date_text(Int(v))
    else:
        kind = X_TRUE if v != 0 else X_FALSE
        text = "true" if v != 0 else "false"
    if neg:
        st.exprs.append(SqlExpr(kind, text, String(), -1, -1, -1, List[Int](), False, line, col))
        st.exprs[node] = SqlExpr(X_NEG, String(), String(), len(st.exprs) - 1, -1, -1, List[Int](), False, line, col)
    else:
        st.exprs[node] = SqlExpr(kind, text, String(), -1, -1, -1, List[Int](), False, line, col)


def literal_node(mut st: SqlStatement, t: Table, row: Int, c: Int, line: Int, col: Int) raises -> Int:
    st.exprs.append(SqlExpr(X_NULL, String("NULL"), String(), -1, -1, -1, List[Int](), False, line, col))
    var n = len(st.exprs) - 1
    set_literal(st, n, t, row, c)
    return n


# ── cloning a statement's shell ──────────────────────────────────────────────────────────────────────

def clone_shell(st: SqlStatement) -> SqlStatement:
    """A statement with st's expressions, FROM and join, and nothing else: the caller sets the select list and the clauses it wants."""
    var c = SqlStatement(st.text)
    c.exprs = st.exprs.copy()
    c.table = st.table
    c.table_alias = st.table_alias
    c.table_is_file = st.table_is_file
    c.table_line = st.table_line
    c.table_col = st.table_col
    c.has_right = st.has_right
    c.right_table = st.right_table
    c.right_alias = st.right_alias
    c.right_is_file = st.right_is_file
    c.right_line = st.right_line
    c.right_col = st.right_col
    c.join_on = st.join_on
    c.join_using = st.join_using.copy()
    c.comma_join = st.comma_join
    c.join_kind = st.join_kind
    c.extra = st.extra.copy()
    c.wins = st.wins.copy()
    c.subs = st.subs.copy()
    return c^


# ── correlation ──────────────────────────────────────────────────────────────────────────────────────

comptime SIDE_INNER = 0
comptime SIDE_OUTER = 1


def side_of(x: SqlExpr, inner: Scope, outer: Scope) -> Int:
    """Whether a column reference of a subquery means one of the subquery's tables or the outer statement's."""
    var f = find_entry(inner, x.text2, x.text)
    if f[0] != -1:
        return SIDE_INNER
    var g = find_entry(outer, x.text2, x.text)
    if g[0] >= 0:
        return SIDE_OUTER
    return SIDE_INNER


def outer_refs(st: SqlStatement, e: Int, inner: Scope, outer: Scope) -> List[Int]:
    """The column references under `e` that mean the outer statement's tables."""
    var nodes = List[Int]()
    collect(st, e, nodes)
    var out = List[Int]()
    for n in nodes:
        if st.exprs[n].kind == X_COL and side_of(st.exprs[n], inner, outer) == SIDE_OUTER:
            out.append(n)
    return out^


def statement_is_correlated(k: SqlStatement, inner: Scope, outer: Scope) -> Bool:
    """Whether any expression of the subquery reads a column of the outer statement (a nested subquery has its own pool and is looked at when it runs)."""
    for i in range(len(k.exprs)):
        if k.exprs[i].kind == X_COL and side_of(k.exprs[i], inner, outer) == SIDE_OUTER:
            return True
    return False


struct Correlation(Movable):
    """What a correlated subquery asks of the outer row: equalities `inner expression = outer column`, and at most one inequality."""
    var inner_conj: List[Int]         # conjuncts of the subquery that read only its own tables
    var key_in: List[Int]             # the inner expression of each equality (the subquery's pool)
    var key_name: List[String]        # the outer column of each equality, as written
    var key_qual: List[String]
    var theta_in: Int                 # -1 for none
    var theta_op: String              # the operator with the inner side on the left: < <= > >= <>
    var theta_name: String
    var theta_qual: String
    var line: Int
    var col: Int

    def __init__(out self):
        self.inner_conj = List[Int]()
        self.key_in = List[Int]()
        self.key_name = List[String]()
        self.key_qual = List[String]()
        self.theta_in = -1
        self.theta_op = String()
        self.theta_name = String()
        self.theta_qual = String()
        self.line = 0
        self.col = 0


def _flip(op: String) -> String:
    if op == "<":
        return ">"
    if op == "<=":
        return ">="
    if op == ">":
        return "<"
    if op == ">=":
        return "<="
    return op


def analyze(k: SqlStatement, inner: Scope, outer: Scope, allow_theta: Bool, what: String) raises -> Correlation:
    """Splits the WHERE of a correlated subquery into what stays inside and what ties it to the outer row. Raises by name for the shapes it does not take apart."""
    var cor = Correlation()
    cor.line = k.table_line
    cor.col = k.table_col
    if len(k.group_by) > 0 or k.having >= 0:
        raise sql_error(k.table_line, k.table_col, "a correlated " + what + " with GROUP BY or HAVING is not supported yet")
    if k.limit >= 0 or k.offset > 0:
        raise sql_error(k.table_line, k.table_col, "a correlated " + what + " with LIMIT or OFFSET is not supported yet")
    if k.distinct and what != "EXISTS" and what != "IN":
        raise sql_error(k.table_line, k.table_col, "a correlated " + what + " with DISTINCT is not supported yet")
    if len(k.wins) > 0:
        raise sql_error(k.table_line, k.table_col, "a correlated " + what + " with window functions is not supported yet")
    for it in k.items:
        if it.expr >= 0 and len(outer_refs(k, it.expr, inner, outer)) > 0:
            raise sql_error(it.line, it.col, "a correlated " + what + " that reads an outer column in its select list is not supported yet (the outer column belongs in WHERE: `inner = outer`)")
    var conj = List[Int]()
    _split_and_expr(k, k.where, conj)
    if k.join_on >= 0 and len(outer_refs(k, k.join_on, inner, outer)) > 0:
        raise sql_error(k.table_line, k.table_col, "a correlated " + what + " that reads an outer column in a join condition is not supported yet: move it to WHERE")
    for c in conj:
        var outs = outer_refs(k, c, inner, outer)
        if len(outs) == 0:
            cor.inner_conj.append(c)
            continue
        ref x = k.exprs[c]
        var ok = x.kind == X_BIN and (x.text == "=" or x.text == "<>" or x.text == "<" or x.text == "<=" or x.text == ">" or x.text == ">=")
        if not ok:
            raise sql_error(x.line, x.col, "this condition of a correlated " + what + " reads an outer column in a form that is not supported: an equality between an expression of the subquery and an outer column is (`inner.k = outer.k`)")
        var a_outs = outer_refs(k, x.a, inner, outer)
        var b_outs = outer_refs(k, x.b, inner, outer)
        var inner_side = -1
        var outer_col = -1
        if len(a_outs) == 0 and len(b_outs) == 1 and k.exprs[x.b].kind == X_COL:
            inner_side = x.a
            outer_col = x.b
        elif len(b_outs) == 0 and len(a_outs) == 1 and k.exprs[x.a].kind == X_COL:
            inner_side = x.b
            outer_col = x.a
        else:
            raise sql_error(x.line, x.col, "this condition of a correlated " + what + " must compare an expression of the subquery's tables with one outer column")
        var op = x.text if inner_side == x.a else _flip(x.text)
        var oc = k.exprs[outer_col].copy()
        if op == "=":
            cor.key_in.append(inner_side)
            cor.key_name.append(oc.text)
            cor.key_qual.append(oc.text2)
        else:
            if not allow_theta:
                raise sql_error(x.line, x.col, "a correlated " + what + " with `" + x.text + "` against an outer column is not supported yet (only EXISTS takes one such condition)")
            if cor.theta_in >= 0:
                raise sql_error(x.line, x.col, "a correlated " + what + " with more than one inequality against outer columns is not supported yet")
            cor.theta_in = inner_side
            cor.theta_op = op
            cor.theta_name = oc.text
            cor.theta_qual = oc.text2
    if len(cor.key_in) == 0 and cor.theta_in < 0:
        raise sql_error(k.table_line, k.table_col, "a correlated " + what + " needs an equality between its tables and the outer row")
    return cor^


# ── building D ───────────────────────────────────────────────────────────────────────────────────────

def set_items(mut k: SqlStatement, exprs: List[Int], names: List[String]):
    k.items = List[SelectItem]()
    for i in range(len(exprs)):
        k.items.append(SelectItem(exprs[i], names[i], True, k.exprs[exprs[i]].line, k.exprs[exprs[i]].col))


def call_node(mut k: SqlStatement, name: String, arg: Int) -> Int:
    var items = List[Int]()
    items.append(arg)
    k.exprs.append(SqlExpr(X_CALL, name, String(), 0, -1, -1, items^, False, k.exprs[arg].line, k.exprs[arg].col))
    return len(k.exprs) - 1


def key_name(i: Int) -> String:
    return "__k" + String(i)


def build_exists_source(mut k: SqlStatement, cor: Correlation):
    """K becomes D: the distinct inner sides of the equalities (EXISTS, IN), or per key the min and max of the inequality's inner side."""
    var exprs = List[Int]()
    var names = List[String]()
    for i in range(len(cor.key_in)):
        exprs.append(cor.key_in[i])
        names.append(key_name(i))
    k.group_by = List[Int]()
    if cor.theta_in >= 0:
        var lo = call_node(k, "MIN", cor.theta_in)
        var hi = call_node(k, "MAX", cor.theta_in)
        exprs.append(lo)
        names.append("__lo")
        exprs.append(hi)
        names.append("__hi")
        for i in range(len(cor.key_in)):
            k.group_by.append(cor.key_in[i])
    else:
        k.distinct = True
    set_items(k, exprs, names)
    k.where = _and_chain(k, cor.inner_conj)
    k.order_by = List[OrderItem]()
    k.limit = -1
    k.offset = 0


def build_value_source(mut k: SqlStatement, cor: Correlation, value: Int):
    """K becomes D with the keys and its value (the scalar subquery's expression, or the IN subquery's column), grouped by the keys when the value is an aggregate."""
    var exprs = List[Int]()
    var names = List[String]()
    for i in range(len(cor.key_in)):
        exprs.append(cor.key_in[i])
        names.append(key_name(i))
    exprs.append(value)
    names.append("__v")
    set_items(k, exprs, names)
    k.group_by = List[Int]()
    k.distinct = False
    k.where = _and_chain(k, cor.inner_conj)
    k.order_by = List[OrderItem]()
    k.limit = -1
    k.offset = 0


def group_by_keys(mut k: SqlStatement, cor: Correlation):
    k.group_by = List[Int]()
    for i in range(len(cor.key_in)):
        k.group_by.append(cor.key_in[i])


# ── the first stage: the outer statement's tables as one table ───────────────────────────────────────

struct Flat(Movable):
    """The columns of a joined FROM clause that the rest of a statement reads, as one table's columns."""
    var entry: List[Int]
    var column: List[Int]
    var name: List[String]
    var scope: Scope

    def __init__(out self):
        self.entry = List[Int]()
        self.column = List[Int]()
        self.name = List[String]()
        self.scope = Scope()


def flatten(mut st: SqlStatement, sc: Scope, mut flat: Flat) raises:
    """Names every column the statement reads once, in a table of its own: the select items without an alias keep the name they would have had, every
    column reference loses its qualifier (and gets a `name_2` when two tables have a column of one name)."""
    flat.scope = sc.copy()
    # the names the select items would show, before any reference changes
    var shown = List[String]()
    for it in st.items:
        if it.star:
            raise sql_error(it.line, it.col, "`*` in a query with a correlated subquery over several tables is not supported yet: name the columns")
        shown.append(String())
    for k in range(len(st.items)):
        ref it = st.items[k]
        if it.has_alias:
            continue
        ref x = st.exprs[it.expr]
        if x.kind == X_COL:
            var f = find_entry(sc, x.text2, x.text)
            shown[k] = sc.real[f[0]][f[1]] if f[0] >= 0 else x.text
        else:
            shown[k] = st.render(it.expr)
    for e in range(len(st.exprs)):
        if st.exprs[e].kind != X_COL:
            continue
        var f = find_entry(sc, st.exprs[e].text2, st.exprs[e].text)
        if f[0] < 0:
            continue
        var at = -1
        for j in range(len(flat.entry)):
            if flat.entry[j] == f[0] and flat.column[j] == f[1]:
                at = j
        if at < 0:
            var nm = sc.real[f[0]][f[1]]
            var taken = True
            var n = 1
            var cand = nm
            while taken:
                taken = False
                for j in range(len(flat.name)):
                    if lower_ascii(flat.name[j]) == lower_ascii(cand):
                        taken = True
                if taken:
                    n += 1
                    cand = nm + "_" + String(n)
            flat.entry.append(f[0])
            flat.column.append(f[1])
            flat.name.append(cand)
            at = len(flat.name) - 1
        st.exprs[e].text = flat.name[at]
        st.exprs[e].text2 = String()
    for k in range(len(st.items)):
        if not st.items[k].has_alias:
            st.items[k].alias = shown[k]
            st.items[k].has_alias = True


def flat_source(st: SqlStatement, flat: Flat, plain: List[Int]) -> SqlStatement:
    """The statement that makes the flat table: the columns, over the FROM and join of `st` as it was before `flatten`, filtered by the conjuncts `plain`."""
    var t0 = clone_shell(st)
    var names = List[String]()
    var exprs = List[Int]()
    for j in range(len(flat.name)):
        var e = flat.entry[j]
        exprs.append(column_node(t0, flat.scope.real[e][flat.column[j]], flat.scope.shown[e], st.table_line, st.table_col))
        names.append(flat.name[j])
    if len(exprs) == 0:                                                         # nothing read: one column carries the rows
        exprs.append(column_node(t0, flat.scope.real[0][0], flat.scope.shown[0], st.table_line, st.table_col))
        names.append(flat.scope.real[0][0])
    set_items(t0, exprs, names)
    t0.where = _and_chain(t0, plain)
    return t0^
