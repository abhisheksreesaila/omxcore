"""Queries made of several statements (F-159): WITH, derived tables in FROM, subqueries as values, IN and EXISTS, UNION / INTERSECT / EXCEPT, QUALIFY.

`sql(catalog, text)` parses the text into a `SqlQuery` (sql.mojo: one `SqlStatement` per SELECT, kids first, the one the text ends with last). A query
of one plain statement goes straight to the engine, as before. Any other runs here, on a copy of the catalog (the sources are shared handles) that
collects the intermediate results under internal names (`__d3`, `__c1`):

- a derived table `FROM (SELECT ...) x` runs first, its table joins the catalog and the statement's FROM names it (alias `x` kept);
- a CTE `WITH c AS (...)` read once runs at its use (inlined); read more than once, it runs once at the first use and every use reads that table
  (the plan, EXPLAIN, says which); `AS MATERIALIZED` and `AS NOT MATERIALIZED` force either;
- a view of the catalog (`Catalog.view`) is a CTE that every query can read.

Each statement then goes to the engine as the one-SELECT statement it was before F-159: the joins, hashed and dense groups, DISTINCT, the sort
(F-167's spilling one included) and window functions are the existing paths, with no second copy of any of them. Subqueries are taken apart in
`subq.mojo`, set operations and QUALIFY below.
"""
from std.time import perf_counter_ns
from .table import Table, SharedTable, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, F32_COLUMN
from .sql import (SqlStatement, SqlQuery, SqlExpr, SelectItem, TableRef, WinSpec, Cte, SubQuery, OrderItem, parse_query, sql_error, lower_ascii, EXPLAIN_NONE, EXPLAIN_ANALYZE,
                  SET_NONE, SET_UNION_ALL, SET_UNION, SET_INTERSECT, SET_EXCEPT, CTE_AUTO, CTE_MATERIALIZED, CTE_INLINED,
                  X_SCALAR_Q, X_IN_Q, X_EXISTS_Q, X_SUBIN, X_EXISTS, X_NOT, X_COL, X_INT, X_NULL, X_CAST, X_TRUE, X_FALSE, X_IN, X_BIN, X_CALL, X_ISBOOL, JK_INNER, JK_LEFT)
from .bind import Catalog, _split_and_expr, _and_chain, _has_aggregate
from .subq import (SIDE_OUTER, side_of, Scope, scope_of, Flat, flatten, flat_source, clone_shell, collect, is_subquery_node, add_node, column_node, equal_node, and_node,
                   copy_expr, set_literal, set_typed_null, literal_node, statement_is_correlated, analyze, build_exists_source, build_value_source, group_by_keys, key_name)
from .setops import union_all
from .planner import Planner, RulePlanner, PlannerOptions
from .sqlexec import SqlResult, run_plain
from .parquet import write_parquet, ZSTD, DEFAULT_ROW_GROUP
from .sink import ResultSink
from std.memory.alloc import unsafe_alloc


struct _Cte(Copyable, Movable):
    """A WITH name in scope: its statement, how many times the query reads it and, once it ran for a second reader, the catalog name of its table."""
    var name: String
    var kid: Int
    var uses: Int
    var hint: Int
    var columns: List[String]
    var table: String
    var hidden: Int                  # while its own body (or an earlier one's) runs, a name sees only the names before it
    var group_end: Int               # the end of the WITH clause it belongs to

    def __init__(out self, name: String, kid: Int, uses: Int, hint: Int, var columns: List[String], group_end: Int = 0):
        self.hidden = 0
        self.group_end = group_end
        self.name = name
        self.kid = kid
        self.uses = uses
        self.hint = hint
        self.columns = columns^
        self.table = String()


struct QueryRun(Movable):
    """One run of a query: the statements still to run (taken out of the store when they run), the CTEs in scope, the steps for EXPLAIN."""
    var stmts: List[SqlStatement]
    var ctes: List[_Cte]
    var steps: List[String]
    var next: Int
    var text: String

    def __init__(out self, mut q: SqlQuery, text: String):
        self.stmts = List[SqlStatement]()
        swap(self.stmts, q.stmts)
        self.ctes = List[_Cte]()
        self.steps = List[String]()
        self.next = 0
        self.text = text

    def fresh(mut self, prefix: String) -> String:
        self.next += 1
        return prefix + String(self.next)

    def take(mut self, idx: Int) -> SqlStatement:
        var st = SqlStatement(self.text)
        swap(st, self.stmts[idx])
        return st^

    def find_cte(self, name: String) -> Int:
        var key = lower_ascii(name)
        var i = len(self.ctes) - 1
        while i >= 0:
            if lower_ascii(self.ctes[i].name) == key and self.ctes[i].hidden == 0:
                return i
            i -= 1
        return -1


def expr_kids(st: SqlStatement) -> List[Int]:
    """The statements the expressions of `st` hold: its subqueries."""
    var out = List[Int]()
    for x in st.exprs:
        if x.kind == X_SCALAR_Q or x.kind == X_IN_Q or x.kind == X_EXISTS_Q:
            out.append(x.items[0])
    return out^


def _count_uses(run: QueryRun, idx: Int, name: String) -> Int:
    """How many times the statement `idx` and everything below it read the table `name` (a FROM entry that is not a file)."""
    ref st = run.stmts[idx]
    var n = 0
    var key = lower_ascii(name)
    if st.table_kid < 0 and not st.table_is_file and lower_ascii(st.table) == key:
        n += 1
    if st.has_right and st.right_kid < 0 and not st.right_is_file and lower_ascii(st.right_table) == key:
        n += 1
    for ex in st.extra:
        if ex.kid < 0 and not ex.is_file and lower_ascii(ex.name) == key:
            n += 1
    var kids = List[Int]()
    if st.table_kid >= 0:
        kids.append(st.table_kid)
    if st.right_kid >= 0:
        kids.append(st.right_kid)
    for ex in st.extra:
        if ex.kid >= 0:
            kids.append(ex.kid)
    for c in st.ctes:
        kids.append(c.kid)
    if st.set_op != SET_NONE:
        kids.append(st.set_left)
        kids.append(st.set_right)
    for k in expr_kids(st):
        kids.append(k)
    for k in kids:
        var shadowed = False
        for c in run.stmts[k].ctes:
            if lower_ascii(c.name) == key:
                shadowed = True
        if not shadowed:
            n += _count_uses(run, k, name)
    return n


def _rename_columns(mut t: Table, names: List[String], line: Int, col: Int) raises:
    if len(names) == 0:
        return
    if len(names) > t.columns():
        raise sql_error(line, col, "the column list names " + String(len(names)) + " columns, the query has " + String(t.columns()))
    t.by_name = Dict[String, Int]()
    for c in range(t.columns()):
        if c < len(names):
            t.cols[c].name = names[c]
        if t.cols[c].name in t.by_name:
            raise sql_error(line, col, "column `" + t.cols[c].name + "` is named twice")
        t.by_name[t.cols[c].name] = c


def _commas(n: Int) -> String:
    var s = String(n)
    var out = String()
    var k = 0
    for i in range(s.byte_length() - 1, -1, -1):
        out = String(s[byte=i:i + 1]) + out
        k += 1
        if k % 3 == 0 and i > 0:
            out = "," + out
    return out


def register_result(mut run: QueryRun, mut catalog: Catalog, mut r: SqlResult, prefix: String) -> String:
    """Puts a result's table in the catalog under a fresh internal name."""
    var name = run.fresh(prefix)
    catalog.add_table(name, r.take_table().share())
    return name


def _resolve_slot[P: Planner](mut run: QueryRun, mut catalog: Catalog, name: String, is_file: Bool, kid: Int, alias: String, line: Int, col: Int,
                              planner: P, opts: PlannerOptions) raises -> Tuple[String, String]:
    """A FROM entry: a derived table runs now; a CTE name runs (or reads the table it already made); anything else is left to the engine."""
    if kid >= 0:
        var r = exec_statement(run, catalog, kid, planner, opts)
        var rows = r.table.rows()
        var nm = register_result(run, catalog, r, "__d")
        run.steps.append("derived table " + (alias if alias.byte_length() > 0 else nm) + ": evaluated in place (" + _commas(rows) + " rows)")
        return (nm, alias)
    if is_file:
        return (name, alias)
    var ci = run.find_cte(name)
    if ci < 0:
        var vi = catalog.view_index(name)
        if vi >= 0:
            return _view_slot(run, catalog, vi, name, alias, planner, opts)
        return (name, alias)
    var shown = alias if alias.byte_length() > 0 else run.ctes[ci].name
    var kept = run.ctes[ci].hint == CTE_MATERIALIZED or run.ctes[ci].uses >= 2     # (NOT MATERIALIZED read twice is kept too: a statement runs once)
    if kept and run.ctes[ci].table.byte_length() > 0:
        return (run.ctes[ci].table, shown)
    var kid2 = run.ctes[ci].kid
    var cols = run.ctes[ci].columns.copy()
    var cname = run.ctes[ci].name
    var uses = run.ctes[ci].uses
    for h in range(ci, run.ctes[ci].group_end):
        run.ctes[h].hidden += 1
    var r: SqlResult
    try:
        r = exec_statement(run, catalog, kid2, planner, opts)
    except e:
        for h in range(ci, run.ctes[ci].group_end):
            run.ctes[h].hidden -= 1
        raise e^
    for h in range(ci, run.ctes[ci].group_end):
        run.ctes[h].hidden -= 1
    _rename_columns(r.table, cols, line, col)
    var rows = r.table.rows()
    var nm = register_result(run, catalog, r, "__c" if kept else "__d")
    if kept:
        run.ctes[ci].table = nm
        run.steps.append("CTE " + cname + ": materialized once (" + String(uses) + " reads, " + _commas(rows) + " rows)")
    else:
        run.steps.append("CTE " + cname + ": inlined into its one read (" + _commas(rows) + " rows)")
    return (nm, shown)


def _view_slot[P: Planner](mut run: QueryRun, mut catalog: Catalog, vi: Int, name: String, alias: String, planner: P,
                           opts: PlannerOptions) raises -> Tuple[String, String]:
    """A named view of the catalog: its SQL runs (in a run of its own) and the result is read as a table."""
    var text = catalog.view_text(vi)
    var vcat = catalog.without_view(vi)
    var r = sql_with(vcat, text, planner, opts)
    var rows = r.table.rows()
    var nm = register_result(run, catalog, r, "__v")
    run.steps.append("view " + name + ": evaluated in place (" + _commas(rows) + " rows)")
    return (nm, alias if alias.byte_length() > 0 else name)


def exec_statement[P: Planner](mut run: QueryRun, mut catalog: Catalog, idx: Int, planner: P, opts: PlannerOptions) raises -> SqlResult:
    """Runs statement `idx` of the query: its WITH names come into scope, the statements it reads run first, then the engine runs it."""
    var st = run.take(idx)
    var mark = len(run.ctes)
    for c in st.ctes:
        run.ctes.append(_Cte(c.name, c.kid, 0, c.hint, c.columns.copy(), mark + len(st.ctes)))
    for i in range(mark, len(run.ctes)):                                        # reads of a name: the statement's body and the CTEs after it
        var n = _count_uses_in(run, st, run.ctes[i].name)
        for j in range(i + 1, len(run.ctes)):
            n += _count_uses(run, run.ctes[j].kid, run.ctes[i].name)
        run.ctes[i].uses = n
    var explain = st.explain
    st.explain = EXPLAIN_NONE
    var r: SqlResult
    try:
        r = _exec_body(run, catalog, st^, planner, opts, explain)
    except e:
        run.ctes.shrink(mark)
        raise e^
    run.ctes.shrink(mark)
    return r^


def _count_uses_in(run: QueryRun, st: SqlStatement, name: String) -> Int:
    """Reads of `name` by a statement that is not in the store (it was taken out to run) and what it reads."""
    var n = 0
    var key = lower_ascii(name)
    if st.table_kid < 0 and not st.table_is_file and lower_ascii(st.table) == key:
        n += 1
    if st.has_right and st.right_kid < 0 and not st.right_is_file and lower_ascii(st.right_table) == key:
        n += 1
    for ex in st.extra:
        if ex.kid < 0 and not ex.is_file and lower_ascii(ex.name) == key:
            n += 1
    var kids = List[Int]()
    if st.table_kid >= 0:
        kids.append(st.table_kid)
    if st.right_kid >= 0:
        kids.append(st.right_kid)
    for ex in st.extra:
        if ex.kid >= 0:
            kids.append(ex.kid)
    if st.set_op != SET_NONE:
        kids.append(st.set_left)
        kids.append(st.set_right)
    for k in expr_kids(st):
        kids.append(k)
    for k in kids:
        var shadowed = False
        for c in run.stmts[k].ctes:
            if lower_ascii(c.name) == key:
                shadowed = True
        if not shadowed:
            n += _count_uses(run, k, name)
    return n


def _exec_body[P: Planner](mut run: QueryRun, mut catalog: Catalog, var st: SqlStatement, planner: P, opts: PlannerOptions,
                           explain: Int) raises -> SqlResult:
    var inner = opts.copy()                                    # (the sink of sql_to_parquet takes the last statement's rows only)
    inner.sink = None
    if st.set_op != SET_NONE:
        var left = exec_statement(run, catalog, st.set_left, planner, inner)
        var right = exec_statement(run, catalog, st.set_right, planner, inner)
        var lt = left.take_table()
        var rt = right.take_table()
        var names = List[String]()
        for c in range(lt.columns()):
            names.append(lt.name(c))
        var combined = union_all(lt, rt, st.table_line, st.table_col)
        var op = st.set_op
        var line = st.table_line
        var col = st.table_col
        st.set_op = SET_NONE
        var plain_union = op == SET_UNION_ALL and len(st.order_by) == 0 and st.limit < 0 and st.offset == 0
        var nrows = combined.rows()
        if op != SET_UNION_ALL:
            combined = distinct_rows(run, catalog, combined^, names, op, rt.rows(), planner, inner)
        run.steps.append(_set_name(op) + ": " + _commas(nrows) + " rows in, " + _commas(combined.rows()) + " out")
        if plain_union or (len(st.order_by) == 0 and st.limit < 0 and st.offset == 0):
            var out = SqlResult()
            out.table = combined^
            out.explain = String()
            if explain != EXPLAIN_NONE:
                out = _explained_steps(run, out)
            return out^
        st.table = register_table(run, catalog, combined^, "__u")
        st.table_alias = String()
        st.table_is_file = False
        st.items.append(SelectItem(-1, String(), False, line, col, True))
    _ = try_inline(run, catalog, st, planner, inner)
    _resolve_slots(run, catalog, st, planner, inner)
    var steps = lower_subqueries(run, catalog, st, planner, inner)
    if len(steps) > 0:
        return run_stages(run, catalog, st^, steps^, planner, inner, opts, explain)
    if st.qualify >= 0:
        return qualify_run(run, catalog, st^, planner, inner, opts, explain)
    var r = run_plain_explained(run, catalog, st^, planner, opts, explain)
    return r^


def _resolve_slots[P: Planner](mut run: QueryRun, mut catalog: Catalog, mut st: SqlStatement, planner: P, opts: PlannerOptions) raises:
    """The FROM entries of a statement that are derived tables, CTEs or views run (or are read) now and the statement names the tables they made."""
    if st.set_op != SET_NONE:
        return
    var slot = _resolve_slot(run, catalog, st.table, st.table_is_file, st.table_kid, st.table_alias, st.table_line, st.table_col, planner, opts)
    st.table = slot[0]
    st.table_alias = slot[1]
    st.table_is_file = False
    st.table_kid = -1
    if st.has_right:
        var rs = _resolve_slot(run, catalog, st.right_table, st.right_is_file, st.right_kid, st.right_alias, st.right_line, st.right_col, planner, opts)
        st.right_table = rs[0]
        st.right_alias = rs[1]
        st.right_is_file = False
        st.right_kid = -1
    for i in range(len(st.extra)):
        var xs = _resolve_slot(run, catalog, st.extra[i].name, st.extra[i].is_file, st.extra[i].kid, st.extra[i].alias, st.extra[i].line, st.extra[i].col, planner, opts)
        st.extra[i].name = xs[0]
        st.extra[i].alias = xs[1]
        st.extra[i].is_file = False
        st.extra[i].kid = -1


def _set_name(op: Int) -> String:
    if op == SET_UNION_ALL:
        return "UNION ALL"
    if op == SET_UNION:
        return "UNION"
    if op == SET_INTERSECT:
        return "INTERSECT"
    return "EXCEPT"


def run_plain_explained[P: Planner](mut run: QueryRun, mut catalog: Catalog, var st: SqlStatement, planner: P, opts: PlannerOptions,
                                    explain: Int) raises -> SqlResult:
    """The engine runs the statement; for EXPLAIN the steps that ran before it come first in the plan."""
    st.explain = explain
    var r = run_plain(catalog, st^, run.text, planner, opts)
    if explain != EXPLAIN_NONE and len(run.steps) > 0:
        var shown = Table.empty()
        try:
            var c = shown.add_column("plan", TEXT)
            shown.append_text(c, "Steps (they ran first, to learn the columns the query reads)")
            for s in run.steps:
                shown.append_text(c, "  " + s)
            shown.append_text(c, "Query")
            for k in range(r.table.rows()):
                shown.append_text(c, "  " + r.table.text_of(0, Int(r.table.get(0, k))))
            r.table = shown^
            r.explain = String()
            for k in range(r.table.rows()):
                r.explain += r.table.text_of(0, Int(r.table.get(0, k))) + "\n"
        except e:
            pass
    return r^


def register_table(mut run: QueryRun, mut catalog: Catalog, var t: Table, prefix: String) -> String:
    var name = run.fresh(prefix)
    catalog.add_table(name, t^.share())
    return name


def _quote(name: String) -> String:
    return '"' + name.replace('"', '""') + '"'


def _explained_steps(run: QueryRun, r: SqlResult) raises -> SqlResult:
    """For EXPLAIN of a query the engine did not plan as one statement: the steps are the plan."""
    var out = SqlResult()
    var shown = Table.empty()
    var c = shown.add_column("plan", TEXT)
    for s in run.steps:
        shown.append_text(c, s)
    out.table = shown^
    for s in run.steps:
        out.explain += s + "\n"
    return out^


def distinct_rows[P: Planner](mut run: QueryRun, mut catalog: Catalog, var combined: Table, names: List[String], op: Int, right_rows: Int, planner: P,
                              opts: PlannerOptions) raises -> Table:
    """UNION / INTERSECT / EXCEPT of the rows `combined` holds (the left operand's rows first, then the right's): the rows of the answer, each once, by
    GROUP BY over every column, which puts all NULLs in one group, as these operations do."""
    var n = combined.columns()
    var left_rows = combined.rows() - right_rows
    # side column: 0 for the left operand's rows, 1 for the right's
    var side = combined.add_column("__side", INT)
    for r in range(combined.rows()):
        combined.append(side, Int64(0 if r < left_rows else 1))
    var cols = String()
    for c in range(n):
        combined.cols[c].name = "c" + String(c)
    combined.by_name = Dict[String, Int]()
    for c in range(combined.columns()):
        combined.by_name[combined.cols[c].name] = c
    var name = register_table(run, catalog, combined^, "__u")
    for c in range(n):
        cols += ("" if c == 0 else ", ") + "c" + String(c)
    var text: String
    if op == SET_UNION:
        text = "SELECT " + cols + " FROM " + name + " GROUP BY " + cols
    else:
        var having = "l > 0 AND r > 0" if op == SET_INTERSECT else "l > 0 AND r = 0"
        text = ("SELECT " + cols + " FROM (SELECT " + cols + ", sum(CASE WHEN __side = 0 THEN 1 ELSE 0 END) AS l, sum(CASE WHEN __side = 1 THEN 1 ELSE 0 END) AS r FROM " +
                name + " GROUP BY " + cols + ") AS g WHERE " + having)
    var res = sql_with(catalog, text, planner, opts)
    var t = res.take_table()
    t.by_name = Dict[String, Int]()
    for c in range(t.columns()):
        t.cols[c].name = names[c]
        t.by_name[names[c]] = c
    return t^


def qualify_run[P: Planner](mut run: QueryRun, mut catalog: Catalog, var st: SqlStatement, planner: P, opts: PlannerOptions,
                            final_opts: PlannerOptions, explain: Int) raises -> SqlResult:
    """`QUALIFY cond`: the select list and the condition are computed with their window functions, the rows whose condition is true stay, and DISTINCT, ORDER BY,
    LIMIT and OFFSET run over what stays (a query over the first one's table, like any filter on a window result)."""
    var q = st.qualify
    st.qualify = -1
    var order = st.order_by.copy()
    var limit = st.limit
    var offset = st.offset
    var distinct = st.distinct
    st.order_by = List[OrderItem]()
    st.limit = -1
    st.offset = 0
    st.distinct = False
    var line = st.exprs[q].line
    var col = st.exprs[q].col
    var known = List[String]()
    var has_star = False
    for it in st.items:
        if it.star:
            has_star = True
        elif it.has_alias:
            known.append(lower_ascii(it.alias))
        elif st.exprs[it.expr].kind == X_COL:
            known.append(lower_ascii(st.exprs[it.expr].text))
    var under = List[Int]()
    collect(st, q, under)
    for u in under:                                  # a select item's alias in the condition means the item
        if st.exprs[u].kind == X_COL and st.exprs[u].text2.byte_length() == 0:
            for it in st.items:
                if it.has_alias and lower_ascii(it.alias) == lower_ascii(st.exprs[u].text):
                    st.exprs[u] = st.exprs[it.expr].copy()
                    break
    st.items.append(SelectItem(q, String("__q"), True, line, col))
    var frags = List[String]()                    # the ORDER BY's keys as text; "#n" is output column n, resolved when the columns are known
    var hidden = 1
    for k in range(len(order)):
        var ex = st.exprs[order[k].expr].copy()
        var frag: String
        if ex.kind == X_INT:
            frag = "#" + ex.text
        elif ex.kind == X_COL and ex.text2.byte_length() == 0 and (has_star or lower_ascii(ex.text) in known):
            frag = _quote(ex.text)
        else:
            st.items.append(SelectItem(order[k].expr, "__o" + String(k), True, ex.line, ex.col))
            frag = _quote("__o" + String(k))
            hidden += 1
        frags.append(frag)
    var inner_explain = explain
    var r = run_plain_explained(run, catalog, st^, planner, opts, inner_explain)
    if explain != EXPLAIN_NONE:
        return r^
    var t = r.take_table()
    var nvis = t.columns() - hidden
    var text = String("SELECT ") + ("DISTINCT " if distinct else "")
    for c in range(nvis):
        text += ("" if c == 0 else ", ") + _quote(t.name(c))
    var vis_names = List[String]()
    for c in range(nvis):
        vis_names.append(t.name(c))
    var tname = register_table(run, catalog, t^, "__q")
    text += " FROM " + _quote(tname) + " WHERE \"__q\""
    if len(order) > 0:
        text += " ORDER BY "
        for k in range(len(order)):
            var f = frags[k]
            if f.startswith("#"):
                var n = Int(String(f[byte=1:]))
                if n < 1 or n > nvis:
                    raise sql_error(line, col, "ORDER BY position " + String(n) + " is not in the select list")
                f = _quote(vis_names[n - 1])
            text += ("" if k == 0 else ", ") + f + (" DESC" if order[k].desc else "") + (" NULLS FIRST" if order[k].nulls == 1 else (" NULLS LAST" if order[k].nulls == 0 else ""))
    if limit >= 0:
        text += " LIMIT " + String(limit)
    if offset > 0:
        text += " OFFSET " + String(offset)
    return sql_with(catalog, text, planner, final_opts)


# ── subqueries ───────────────────────────────────────────────────────────────────────────────────────

struct Step(Movable):
    """One condition of a WHERE that holds a correlated (or IN) subquery, ready to run on the outer rows: the condition (an expression of the outer statement's
    pool), the table D it joins (empty for none), the join's condition and whether the join is a LEFT one."""
    var conj: Int
    var d: String
    var join_on: Int
    var left: Bool

    def __init__(out self, conj: Int, d: String, join_on: Int, left: Bool):
        self.conj = conj
        self.d = d
        self.join_on = join_on
        self.left = left


def _prepare_kid[P: Planner](mut run: QueryRun, mut catalog: Catalog, kid: Int, planner: P, opts: PlannerOptions) raises:
    """Runs the derived tables and CTEs in the FROM of a subquery, so that its columns are known; the subquery stays where it is."""
    if run.stmts[kid].set_op != SET_NONE or len(run.stmts[kid].ctes) > 0:
        return
    var k = run.take(kid)
    try:
        _resolve_slots(run, catalog, k, planner, opts)
    except e:
        run.stmts[kid] = k^
        raise e^
    run.stmts[kid] = k^


def _is_correlated[P: Planner](mut run: QueryRun, mut catalog: Catalog, outer: Scope, kid: Int, planner: P, opts: PlannerOptions) raises -> Bool:
    if run.stmts[kid].set_op != SET_NONE or len(run.stmts[kid].ctes) > 0:
        return False
    _prepare_kid(run, catalog, kid, planner, opts)
    var inner = scope_of(run.stmts[kid], catalog)
    return statement_is_correlated(run.stmts[kid], inner, outer)


def _run_kid[P: Planner](mut run: QueryRun, mut catalog: Catalog, kid: Int, planner: P, opts: PlannerOptions) raises -> Table:
    var r = exec_statement(run, catalog, kid, planner, opts)
    return r.take_table()


def _substitute[P: Planner](mut run: QueryRun, mut catalog: Catalog, mut st: SqlStatement, n: Int, planner: P, opts: PlannerOptions) raises:
    """An uncorrelated subquery runs once and its answer takes its place in the expression: a value, TRUE or FALSE, or a list of values."""
    var kid = st.exprs[n].items[0]
    var kind = st.exprs[n].kind
    var line = st.exprs[n].line
    var col = st.exprs[n].col
    var neg = st.exprs[n].flag
    var t = _run_kid(run, catalog, kid, planner, opts)
    if kind == X_EXISTS_Q:
        var has = t.rows() > 0
        st.exprs[n] = SqlExpr(X_TRUE if (has != neg) else X_FALSE, String("true") if (has != neg) else String("false"), String(), -1, -1, -1, List[Int](), False, line, col)
        run.steps.append("subquery EXISTS: " + ("rows" if has else "no rows") + ", a constant")
        return
    if t.columns() != 1:
        raise sql_error(line, col, "a subquery in " + ("IN" if kind == X_IN_Q else "a value position") + " must return one column, this one returns " + String(t.columns()))
    if kind == X_SCALAR_Q:
        if t.rows() > 1:
            raise sql_error(line, col, "the scalar subquery returned " + _commas(t.rows()) + " rows: a scalar subquery must return at most one row")
        if t.rows() == 0:
            set_typed_null(st, n, t.kind(0), t.scale(0), line, col)
        else:
            set_literal(st, n, t, 0, 0)
        run.steps.append("scalar subquery: " + _commas(t.rows()) + " row, a constant")
        return
    if t.rows() > MAX_IN_LIST:
        raise sql_error(line, col, "IN (SELECT ...) here returned " + _commas(t.rows()) + " rows, more than the " + _commas(MAX_IN_LIST) +
                        " a list takes (a column on the left of IN, in a condition of WHERE, makes it a semi join)")
    var items = List[Int]()
    for r in range(t.rows()):
        items.append(literal_node(st, t, r, 0, line, col))
    st.exprs[n].kind = X_IN
    st.exprs[n].items = items^
    run.steps.append("subquery IN: a list of " + _commas(t.rows()) + " values")


comptime MAX_IN_LIST = 10000


def _is_simple(k: SqlStatement) -> Bool:
    """A subquery of the shape F-154 lifts into a semi (anti) join as it stands: one table, a select item, a WHERE, nothing else."""
    if k.set_op != SET_NONE or len(k.ctes) > 0 or k.has_right or len(k.extra) > 0 or k.table_kid >= 0 or k.table_is_file or k.join_on >= 0:
        return False
    if len(k.group_by) > 0 or k.having >= 0 or k.limit >= 0 or k.offset > 0 or len(k.wins) > 0 or k.qualify >= 0:
        return False
    for i in range(len(k.exprs)):
        if is_subquery_node(k.exprs[i].kind):
            return False
    if len(k.items) != 1:
        return False
    if k.items[0].expr >= 0 and _has_aggregate(k, k.items[0].expr):
        return False
    if k.where >= 0 and _has_aggregate(k, k.where):
        return False
    var conj = List[Int]()
    _split_and_expr(k, k.where, conj)
    for c in conj:                                  # (a condition of no column, `1 = 0`, is not one a semi join's ON takes)
        var under = List[Int]()
        collect(k, c, under)
        var has_col = False
        for u in under:
            if k.exprs[u].kind == X_COL:
                has_col = True
        if not has_col:
            return False
    return True


def _native_sub(mut st: SqlStatement, k: SqlStatement, n: Int, with_item: Bool, inner: Scope, outer: Scope) -> Int:
    """The subquery as F-154 reads it: its table, its selected expression (IN) and its WHERE, copied into the statement's pool. The references that mean the
    subquery's table and the ones that mean the outer statement's are told apart (`SubQuery.inner`, `.outer`)."""
    var sub = SubQuery()
    var first = len(st.exprs)
    sub.table = k.table
    sub.alias = k.table_alias
    sub.line = st.exprs[n].line
    sub.col = st.exprs[n].col
    if with_item:
        sub.sel = copy_expr(k, k.items[0].expr, st)
    if k.where >= 0:
        sub.where = copy_expr(k, k.where, st)
    for e in range(first, len(st.exprs)):
        if st.exprs[e].kind == X_COL:
            if side_of(st.exprs[e], inner, outer) == SIDE_OUTER:
                sub.outer.append(e)
            else:
                sub.inner.append(e)
    st.subs.append(sub^)
    return len(st.subs) - 1


def _native_in[P: Planner](mut run: QueryRun, mut catalog: Catalog, mut st: SqlStatement, n: Int, planner: P, opts: PlannerOptions) raises:
    """`x [NOT] IN (SELECT c ...)` that reads no outer column: the subquery runs, and the condition is the semi (anti) join with its table, as F-154 has it."""
    var kid = st.exprs[n].items[0]
    var line = st.exprs[n].line
    var col = st.exprs[n].col
    if _is_simple(run.stmts[kid]) and not run.stmts[kid].items[0].star:
        var at = _native_sub(st, run.stmts[kid], n, True, scope_of(run.stmts[kid], catalog), Scope())
        st.exprs[n].kind = X_SUBIN
        st.exprs[n].items = List[Int]()
        st.exprs[n].items.append(at)
        run.steps.append("subquery IN: " + run.stmts[kid].table + ", a semi join")
        return
    var r = exec_statement(run, catalog, kid, planner, opts)
    if r.table.columns() != 1:
        raise sql_error(line, col, "a subquery in IN must return one column, this one returns " + String(r.table.columns()))
    var cname = r.table.name(0)
    var rows = r.table.rows()
    var d = register_result(run, catalog, r, "__d")
    run.steps.append("subquery IN: " + d + " (" + _commas(rows) + " rows), a semi join")
    var sub = SubQuery()
    sub.table = d
    sub.line = line
    sub.col = col
    sub.sel = column_node(st, cname, String(), line, col)
    st.subs.append(sub^)
    st.exprs[n].kind = X_SUBIN
    st.exprs[n].items = List[Int]()
    st.exprs[n].items.append(len(st.subs) - 1)


def _correlated_exists[P: Planner](mut run: QueryRun, mut catalog: Catalog, mut st: SqlStatement, n: Int, negated: Bool, sc: Scope, planner: P,
                                   opts: PlannerOptions) raises -> Step:
    var kid = st.exprs[n].items[0]
    var line = st.exprs[n].line
    var col = st.exprs[n].col
    var flag = st.exprs[n].flag
    if _is_simple(run.stmts[kid]):
        var inner0 = scope_of(run.stmts[kid], catalog)
        var cor0 = analyze(run.stmts[kid], inner0, sc, True, "EXISTS")
        var bare = True
        for ke in cor0.key_in:
            if run.stmts[kid].exprs[ke].kind != X_COL:
                bare = False
        if cor0.theta_in < 0 and bare:                                          # equalities between columns only: F-154's semi (anti) join takes the subquery as it is
            var at = _native_sub(st, run.stmts[kid], n, False, inner0, sc)
            st.exprs[n].kind = X_EXISTS
            st.exprs[n].items = List[Int]()
            st.exprs[n].items.append(at)
            run.steps.append("subquery EXISTS: " + run.stmts[kid].table + ", a semi join")
            return Step(-1, String(), -1, False)
    var k = run.take(kid)
    var inner = scope_of(k, catalog)
    var cor = analyze(k, inner, sc, True, "EXISTS")
    build_exists_source(k, cor)
    run.stmts[kid] = k^
    var r = exec_statement(run, catalog, kid, planner, opts)
    var rows = r.table.rows()
    var keys = len(cor.key_in)
    var lo = -1
    var hi = -1
    if keys == 0 and cor.theta_in >= 0:                                         # no key: D is one row, its two values are constants
        lo = literal_node(st, r.table, 0, 0, line, col)
        hi = literal_node(st, r.table, 0, 1, line, col)
    var d = register_result(run, catalog, r, "__d")
    if cor.theta_in < 0:                                                        # keys only: the semi (anti) join of F-154 against D
        var sub = SubQuery()
        sub.table = d
        sub.line = line
        sub.col = col
        var on = -1
        for i in range(keys):
            var dk = column_node(st, key_name(i), d, line, col)
            var oc = column_node(st, cor.key_name[i], cor.key_qual[i], line, col)
            on = and_node(st, on, equal_node(st, dk, oc, line, col))
        sub.where = on
        st.subs.append(sub^)
        st.exprs[n].kind = X_EXISTS
        st.exprs[n].items = List[Int]()
        st.exprs[n].items.append(len(st.subs) - 1)
        run.steps.append("subquery EXISTS: " + d + " (" + _commas(rows) + " keys), a semi join")
        return Step(-1, String(), -1, False)
    # one inequality: D holds the min and max of the inner side per key; the outer row is kept by what they say
    var o = column_node(st, cor.theta_name, cor.theta_qual, line, col)
    var cond: Int
    var dlo = lo if keys == 0 else column_node(st, "__lo", d, line, col)
    var dhi = hi if keys == 0 else column_node(st, "__hi", d, line, col)
    var op = cor.theta_op
    if op == "<>":
        var a = add_node(st, X_BIN, "<>", dlo, o, line, col)
        var o2 = column_node(st, cor.theta_name, cor.theta_qual, line, col)
        var b = add_node(st, X_BIN, "<>", dhi, o2, line, col)
        cond = add_node(st, X_BIN, "OR", a, b, line, col)
    elif op == "<" or op == "<=":
        cond = add_node(st, X_BIN, op, dlo, o, line, col)
    else:
        cond = add_node(st, X_BIN, op, dhi, o, line, col)
    var on = -1
    for i in range(keys):
        var dk = column_node(st, key_name(i), d, line, col)
        var oc = column_node(st, cor.key_name[i], cor.key_qual[i], line, col)
        on = and_node(st, on, equal_node(st, oc, dk, line, col))
    var absent = flag != negated
    run.steps.append("subquery " + ("NOT EXISTS" if absent else "EXISTS") + ": " + d + " (" + _commas(rows) + " keys with their min and max), a " + ("left join" if absent else "join"))
    if absent:
        st.exprs.append(SqlExpr(X_ISBOOL, String("TRUE"), String(), cond, -1, -1, List[Int](), True, line, col))
        return Step(len(st.exprs) - 1, d if keys > 0 else String(), on, True)
    return Step(cond, d if keys > 0 else String(), on, False)


def _correlated_in[P: Planner](mut run: QueryRun, mut catalog: Catalog, mut st: SqlStatement, n: Int, negated: Bool, sc: Scope, planner: P,
                               opts: PlannerOptions) raises -> Step:
    var kid = st.exprs[n].items[0]
    var line = st.exprs[n].line
    var col = st.exprs[n].col
    if st.exprs[n].flag or negated:
        raise sql_error(line, col, "a correlated NOT IN (SELECT ...) is not supported yet (NOT EXISTS says it without the NULL rule)")
    var left = st.exprs[n].a
    if st.exprs[left].kind != X_COL:
        raise sql_error(line, col, "a correlated IN (SELECT ...) needs a column on its left side")
    var k = run.take(kid)
    if len(k.items) != 1 or k.items[0].star:
        raise sql_error(line, col, "a subquery in IN must return one column")
    var inner = scope_of(k, catalog)
    var cor = analyze(k, inner, sc, False, "IN")
    build_value_source(k, cor, k.items[0].expr)
    k.distinct = True
    run.stmts[kid] = k^
    var r = exec_statement(run, catalog, kid, planner, opts)
    var rows = r.table.rows()
    var d = register_result(run, catalog, r, "__d")
    var sub = SubQuery()
    sub.table = d
    sub.line = line
    sub.col = col
    var on = -1
    for i in range(len(cor.key_in)):
        var dk = column_node(st, key_name(i), d, line, col)
        var oc = column_node(st, cor.key_name[i], cor.key_qual[i], line, col)
        on = and_node(st, on, equal_node(st, dk, oc, line, col))
    var dv = column_node(st, "__v", d, line, col)
    on = and_node(st, on, equal_node(st, dv, left, line, col))
    sub.where = on
    st.subs.append(sub^)
    st.exprs[n].kind = X_EXISTS
    st.exprs[n].items = List[Int]()
    st.exprs[n].items.append(len(st.subs) - 1)
    st.exprs[n].flag = False
    run.steps.append("subquery IN: " + d + " (" + _commas(rows) + " rows), a semi join on the correlation and the value")
    return Step(-1, String(), -1, False)


def _correlated_scalar[P: Planner](mut run: QueryRun, mut catalog: Catalog, mut st: SqlStatement, n: Int, conj: Int, sc: Scope, planner: P,
                                   opts: PlannerOptions) raises -> Step:
    var kid = st.exprs[n].items[0]
    var line = st.exprs[n].line
    var col = st.exprs[n].col
    var k = run.take(kid)
    if len(k.items) != 1 or k.items[0].star:
        raise sql_error(line, col, "a scalar subquery must return one column")
    var value = k.items[0].expr
    if not _has_aggregate(k, value):
        raise sql_error(line, col, "a correlated scalar subquery must be an aggregate (without one it can return several rows per outer row)")
    var is_count = k.exprs[value].kind == X_CALL and k.exprs[value].text == "COUNT"
    var inner = scope_of(k, catalog)
    var cor = analyze(k, inner, sc, False, "scalar subquery")
    var under = List[Int]()
    collect(k, value, under)
    var calls = 0
    for a in under:
        if k.exprs[a].kind == X_CALL and k.exprs[a].text == "COUNT":
            calls += 1
    if calls > 0 and not is_count:
        raise sql_error(line, col, "a correlated scalar subquery with COUNT inside an expression is not supported yet (an empty group is 0, not NULL)")
    build_value_source(k, cor, value)
    group_by_keys(k, cor)
    run.stmts[kid] = k^
    var r = exec_statement(run, catalog, kid, planner, opts)
    var rows = r.table.rows()
    var d = register_result(run, catalog, r, "__d")
    var on = -1
    for i in range(len(cor.key_in)):
        var dk = column_node(st, key_name(i), d, line, col)
        var oc = column_node(st, cor.key_name[i], cor.key_qual[i], line, col)
        on = and_node(st, on, equal_node(st, oc, dk, line, col))
    var cmp = st.exprs[conj].kind == X_BIN and (st.exprs[conj].text == "=" or st.exprs[conj].text == "<>" or st.exprs[conj].text == "<" or
                                                 st.exprs[conj].text == "<=" or st.exprs[conj].text == ">" or st.exprs[conj].text == ">=") and \
          (st.exprs[conj].a == n or st.exprs[conj].b == n)
    var left = is_count or not cmp
    var vnode = SqlExpr(X_COL, String("__v"), d, -1, -1, -1, List[Int](), False, line, col)
    if is_count:
        st.exprs.append(vnode^)
        var zero = SqlExpr(X_INT, String("0"), String(), -1, -1, -1, List[Int](), False, line, col)
        st.exprs.append(zero^)
        var items = List[Int]()
        items.append(len(st.exprs) - 2)
        items.append(len(st.exprs) - 1)
        st.exprs[n] = SqlExpr(X_CALL, String("COALESCE"), String(), 0, -1, -1, items^, False, line, col)
    else:
        st.exprs[n] = vnode^
    run.steps.append("correlated scalar subquery: " + d + " (" + _commas(rows) + " keys), a " + ("left join" if left else "join"))
    return Step(conj, d, on, left)


def lower_subqueries[P: Planner](mut run: QueryRun, mut catalog: Catalog, mut st: SqlStatement, planner: P, opts: PlannerOptions) raises -> List[Step]:
    """Takes the subqueries of a statement apart (module doc of subq.mojo). Uncorrelated ones run now and become constants, lists or a semi join; correlated
    ones become a derived table and a Step each. On return the statement's WHERE holds only the conditions that need no step."""
    var steps = List[Step]()
    var nodes = List[Int]()
    for i in range(len(st.exprs)):
        if is_subquery_node(st.exprs[i].kind):
            nodes.append(i)
    if len(nodes) == 0:
        return steps^
    var sc = scope_of(st, catalog)
    var conj = List[Int]()
    _split_and_expr(st, st.where, conj)
    var owner = List[Int](length=len(st.exprs), fill=-1)
    for ci in range(len(conj)):
        var under = List[Int]()
        collect(st, conj[ci], under)
        for u in under:
            owner[u] = ci
    var correlated = List[Bool]()
    for n in nodes:
        correlated.append(_is_correlated(run, catalog, sc, st.exprs[n].items[0], planner, opts))
    # nodes outside the WHERE's conjuncts, and uncorrelated ones that are not an IN condition of their own, run first
    var form_node = List[Int](length=len(conj), fill=-1)
    var form_not = List[Bool](length=len(conj), fill=False)
    for ci in range(len(conj)):
        var e = conj[ci]
        if st.exprs[e].kind == X_NOT and (st.exprs[st.exprs[e].a].kind == X_EXISTS_Q or st.exprs[st.exprs[e].a].kind == X_IN_Q):
            form_node[ci] = st.exprs[e].a
            form_not[ci] = True
        elif st.exprs[e].kind == X_EXISTS_Q or st.exprs[e].kind == X_IN_Q:
            form_node[ci] = e
    for j in range(len(nodes)):
        var n = nodes[j]
        var ci = owner[n]
        if ci < 0:
            if correlated[j]:
                raise sql_error(st.exprs[n].line, st.exprs[n].col, "a correlated subquery outside WHERE is not supported yet")
            _substitute(run, catalog, st, n, planner, opts)
        elif not correlated[j] and not (form_node[ci] == n and st.exprs[n].kind == X_IN_Q and st.exprs[st.exprs[n].a].kind == X_COL):
            _substitute(run, catalog, st, n, planner, opts)         # (an IN whose left side is no column cannot be a semi join: its values become a list)
    var plain = List[Int]()
    for ci in range(len(conj)):
        var rest = List[Int]()
        var under = List[Int]()
        collect(st, conj[ci], under)
        for u in under:
            if is_subquery_node(st.exprs[u].kind):
                rest.append(u)
        if len(rest) == 0:
            plain.append(conj[ci])
            continue
        if len(rest) > 1:
            raise sql_error(st.exprs[rest[1]].line, st.exprs[rest[1]].col, "more than one subquery that reads an outer column in one condition is not supported yet: put each in a condition of its own")
        var n = rest[0]
        var kind = st.exprs[n].kind
        if form_node[ci] == n and kind == X_IN_Q and not _node_is_correlated(nodes, correlated, n):
            _native_in(run, catalog, st, n, planner, opts)
            steps.append(Step(conj[ci], String(), -1, False))
        elif form_node[ci] == n and kind == X_EXISTS_Q:
            var s = _correlated_exists(run, catalog, st, n, form_not[ci], sc, planner, opts)
            steps.append(Step(conj[ci] if s.conj < 0 else s.conj, s.d, s.join_on, s.left))
        elif form_node[ci] == n and kind == X_IN_Q:
            var s = _correlated_in(run, catalog, st, n, form_not[ci], sc, planner, opts)
            steps.append(Step(conj[ci], s.d, s.join_on, s.left))
        elif kind == X_SCALAR_Q:
            steps.append(_correlated_scalar(run, catalog, st, n, conj[ci], sc, planner, opts))
        else:
            raise sql_error(st.exprs[n].line, st.exprs[n].col, "a subquery that reads an outer column is supported as a condition of its own (EXISTS, IN, or a comparison with a scalar subquery), not inside OR, NOT or an expression")
    st.where = _and_chain(st, plain)
    return steps^


def _node_is_correlated(nodes: List[Int], correlated: List[Bool], n: Int) -> Bool:
    for j in range(len(nodes)):
        if nodes[j] == n:
            return correlated[j]
    return False


def _shown(table: String, alias: String) -> String:
    return alias if alias.byte_length() > 0 else table


def run_stages[P: Planner](mut run: QueryRun, mut catalog: Catalog, var st: SqlStatement, var steps: List[Step], planner: P, opts: PlannerOptions,
                           final_opts: PlannerOptions, explain: Int) raises -> SqlResult:
    """The statement with correlated subqueries, one step at a time over the outer rows: with several tables the outer rows are first made one table; each step is a
    query over the table the step before made (joined with its D when it has one); the last step joins the statement's own query."""
    var plain = List[Int]()
    _split_and_expr(st, st.where, plain)
    if st.has_right or len(st.extra) > 0:
        var orig = clone_shell(st)
        var sc = scope_of(st, catalog)
        var flat = Flat()
        flatten(st, sc, flat)
        var t0 = flat_source(orig, flat, plain)
        t0.text = st.text
        var r0 = run_plain(catalog, t0^, run.text, planner, opts)
        var rows0 = r0.table.rows()
        var name0 = register_result(run, catalog, r0, "__t")
        run.steps.append("outer tables read by a subquery: " + name0 + " (" + _commas(rows0) + " rows)")
        for si in range(len(st.subs)):                                          # a subquery's outer references name the flat table (its own columns may share a name)
            for e in st.subs[si].outer:
                st.exprs[e].text2 = name0
        st.table = name0
        st.table_alias = String()
        st.table_is_file = False
        st.has_right = False
        st.right_table = String()
        st.right_alias = String()
        st.join_on = -1
        st.join_using = List[String]()
        st.comma_join = False
        st.join_kind = JK_INNER
        st.extra = List[TableRef]()
        plain = List[Int]()
    var carried = _shown(st.table, st.table_alias) if not (st.table_alias.byte_length() == 0 and st.table.startswith("__t")) else String()
    var merge_last = True
    for it in st.items:
        if it.star:
            merge_last = False
    var stages = len(steps) - (1 if merge_last else 0)
    for i in range(stages):
        var stage = clone_shell(st)
        stage.wins = List[WinSpec]()
        var shown = _shown(st.table, st.table_alias)
        stage.items.append(SelectItem(-1, shown if (steps[i].d.byte_length() > 0) else String(), False, st.table_line, st.table_col, True))
        stage.table_alias = st.table_alias
        var w = steps[i].conj
        if i == 0:
            for p in plain:
                w = and_node(stage, w, p) if w >= 0 else p
        stage.where = w
        if steps[i].d.byte_length() > 0:
            stage.has_right = True
            stage.right_table = steps[i].d
            stage.right_alias = String()
            stage.join_kind = JK_LEFT if steps[i].left else JK_INNER
            stage.join_on = steps[i].join_on
        var r = run_plain(catalog, stage^, run.text, planner, opts)
        var rows = r.table.rows()
        var nm = register_result(run, catalog, r, "__t")
        run.steps.append("subquery step " + String(i + 1) + ": " + nm + " (" + _commas(rows) + " rows)")
        st.table = nm
        st.table_alias = carried
    var rest = List[Int]()
    if stages == 0:
        for p in plain:
            rest.append(p)
    if merge_last:
        var last = len(steps) - 1
        rest.append(steps[last].conj)
        if steps[last].d.byte_length() > 0:
            st.has_right = True
            st.right_table = steps[last].d
            st.right_alias = String()
            st.join_kind = JK_LEFT if steps[last].left else JK_INNER
            st.join_on = steps[last].join_on
    st.where = _and_chain(st, rest)
    return run_plain_explained(run, catalog, st^, planner, final_opts, explain)


# ── inlining ─────────────────────────────────────────────────────────────────────────────────────────

def _inlinable(k: SqlStatement) -> Bool:
    """A derived table or CTE that is only a filter and a projection of one source: `SELECT cols | expressions FROM t [WHERE p]`."""
    if k.set_op != SET_NONE or len(k.ctes) > 0 or k.has_right or len(k.extra) > 0 or k.table_kid >= 0 or k.join_on >= 0:
        return False
    if len(k.group_by) > 0 or k.having >= 0 or k.distinct or len(k.order_by) > 0 or k.limit >= 0 or k.offset > 0 or len(k.wins) > 0 or k.qualify >= 0:
        return False
    for i in range(len(k.exprs)):
        if is_subquery_node(k.exprs[i].kind):
            return False
    for it in k.items:
        if it.star:
            if len(k.items) != 1 or it.alias.byte_length() > 0:
                return False
        elif _has_aggregate(k, it.expr):
            return False
    if k.where >= 0 and _has_aggregate(k, k.where):
        return False
    return True


def _out_name(k: SqlStatement, i: Int) -> String:
    """The name the select item `i` of a query shows (the binder's rule: the alias, a column's own name, else the expression as written)."""
    ref it = k.items[i]
    if it.has_alias:
        return it.alias
    if k.exprs[it.expr].kind == X_COL:
        return k.exprs[it.expr].text
    return k.render(it.expr)


def try_inline[P: Planner](mut run: QueryRun, mut catalog: Catalog, mut st: SqlStatement, planner: P, opts: PlannerOptions) raises -> Bool:
    """A statement over one derived table (or a CTE read once) that is just a filter and a projection of a source reads that source: the filter joins the statement's
    WHERE and the statement's references to the derived table's columns become the expressions behind them. Nothing is materialized, and a Parquet scan under it is
    filtered as it streams. Returns whether it did (the statement is untouched when it did not)."""
    if st.set_op != SET_NONE or st.has_right or len(st.extra) > 0 or len(st.subs) > 0:
        return False
    for i in range(len(st.exprs)):
        if is_subquery_node(st.exprs[i].kind):                  # (a subquery may read the derived table's columns through its own pool)
            return False
    var kid = -1
    var ci = -1
    var shown = st.table_alias
    if st.table_kid >= 0:
        kid = st.table_kid
    elif not st.table_is_file:
        ci = run.find_cte(st.table)
        if ci < 0:
            return False
        if run.ctes[ci].uses != 1 or run.ctes[ci].hint == CTE_MATERIALIZED or len(run.ctes[ci].columns) > 0:
            return False
        kid = run.ctes[ci].kid
        if shown.byte_length() == 0:
            shown = run.ctes[ci].name
    if kid < 0:
        return False
    var inner_taken = run.take(kid)                     # a derived table of the derived table folds into it first
    var hide_to = run.ctes[ci].group_end if ci >= 0 else 0
    for h in range(ci if ci >= 0 else 0, hide_to):      # (a CTE's own body does not see its name, nor the names after it)
        run.ctes[h].hidden += 1
    try:
        _ = try_inline(run, catalog, inner_taken, planner, opts)
    except e:
        for h in range(ci if ci >= 0 else 0, hide_to):
            run.ctes[h].hidden -= 1
        run.stmts[kid] = inner_taken^
        raise e^
    for h in range(ci if ci >= 0 else 0, hide_to):
        run.ctes[h].hidden -= 1
    run.stmts[kid] = inner_taken^
    ref k = run.stmts[kid]
    if not _inlinable(k) or k.table_is_file:
        return False
    if k.table_kid >= 0 or run.find_cte(k.table) >= 0 or catalog.view_index(k.table) >= 0:
        return False
    var star_kid = len(k.items) == 1 and k.items[0].star
    var key = lower_ascii(shown)
    for it in st.items:
        if it.star and not star_kid and it.alias.byte_length() > 0 and lower_ascii(it.alias) != key:
            return False
    # which column references mean the derived table's columns (and which expression each is)
    var names = List[String]()
    var real = List[String]()
    var exprs = List[Int]()
    if not star_kid:
        for i in range(len(k.items)):
            real.append(_out_name(k, i))
            names.append(lower_ascii(real[i]))
            exprs.append(k.items[i].expr)
    var own_aliases = List[String]()
    for it in st.items:
        if it.has_alias:
            own_aliases.append(lower_ascii(it.alias))
    var before = len(st.exprs)
    var pick = List[Int](length=before, fill=-1)
    for n in range(before):
        if st.exprs[n].kind != X_COL:
            continue
        var q = lower_ascii(st.exprs[n].text2)
        var nm = lower_ascii(st.exprs[n].text)
        if q.byte_length() > 0 and q != key:
            return False
        if star_kid:
            continue
        var at = -1
        for j in range(len(names)):
            if names[j] == nm:
                at = j
        if at < 0:
            if q.byte_length() == 0 and nm in own_aliases:
                continue
            return False
        pick[n] = at
    # an item without an alias that reads the derived table's columns is named after what it reads (an aggregate's name is the engine's own and cannot be kept)
    for i in range(len(st.items)):
        if st.items[i].star or st.items[i].has_alias or st.exprs[st.items[i].expr].kind == X_COL:
            continue
        var under = List[Int]()
        collect(st, st.items[i].expr, under)
        var touched = False
        for u in under:
            if u < before and st.exprs[u].kind == X_COL and (pick[u] >= 0 or (star_kid and st.exprs[u].text2.byte_length() > 0)):
                touched = True
        if touched and _has_aggregate(st, st.items[i].expr):
            return False
    # nothing can fail from here on
    for i in range(len(st.items)):
        if st.items[i].star and star_kid:
            st.items[i].alias = k.table_alias                   # `x.*` is the source's `*`
        elif st.items[i].star or st.items[i].has_alias:
            pass
        elif st.exprs[st.items[i].expr].kind == X_COL:
            if not star_kid and pick[st.items[i].expr] >= 0:
                st.items[i].alias = real[pick[st.items[i].expr]]    # the column keeps the name the derived table gave it
                st.items[i].has_alias = True
        elif not _has_aggregate(st, st.items[i].expr):
            st.items[i].alias = st.render(st.items[i].expr)         # an expression keeps the name it had over the derived table
            st.items[i].has_alias = True
    var kcopy = k.where
    for n in range(before):
        if st.exprs[n].kind != X_COL:
            continue
        if star_kid:
            st.exprs[n].text2 = k.table_alias
        elif pick[n] >= 0:
            var root = copy_expr(k, exprs[pick[n]], st)
            st.exprs[n] = st.exprs[root].copy()
    if not star_kid:                                             # `*` over a projection is the projection's columns
        var items = List[SelectItem]()
        for i in range(len(st.items)):
            if st.items[i].star:
                for j in range(len(real)):
                    items.append(SelectItem(copy_expr(k, exprs[j], st), real[j], True, st.items[i].line, st.items[i].col))
            else:
                items.append(st.items[i].copy())
        st.items = items^
    var kw = -1
    if kcopy >= 0:
        kw = copy_expr(k, kcopy, st)
    st.where = and_node(st, kw, st.where)
    var table = k.table
    var talias = k.table_alias
    st.table = table
    st.table_alias = talias
    st.table_is_file = False
    st.table_kid = -1
    if ci >= 0:
        run.steps.append("CTE " + run.ctes[ci].name + ": inlined into the query that reads it (its filter and columns merged, nothing materialized)")
    else:
        run.steps.append("derived table " + (shown if shown.byte_length() > 0 else String("")) + ": inlined into the query that reads it (its filter and columns merged, nothing materialized)")
    return True


def sql_with[P: Planner](mut catalog: Catalog, text: String, planner: P, options: PlannerOptions = PlannerOptions()) raises -> SqlResult:
    """`sql` with the physical planner of your choice (anything that implements `Planner`)."""
    var q = parse_query(text)
    var plain = len(q.stmts) == 1
    if plain:
        ref st = q.stmts[0]
        plain = len(st.ctes) == 0 and st.qualify < 0 and st.set_op == SET_NONE and not catalog.reads_a_view(st)
    if plain:
        return run_plain(catalog, q.stmts.pop(), text, planner, options)
    var work = catalog.copy()
    var root = len(q.stmts) - 1
    var run = QueryRun(q, text)
    var t0 = perf_counter_ns()
    var r = exec_statement(run, work, root, planner, options)
    for step in run.steps:
        r.notes.append("step: " + step)
    r.ms = Float64(perf_counter_ns() - t0) / 1e6
    return r^


def sql(mut catalog: Catalog, text: String, options: PlannerOptions = PlannerOptions()) raises -> SqlResult:
    """Runs one SELECT (or EXPLAIN SELECT, WITH, UNION ...) over the catalog's sources with the rule-based planner. The result owns its table."""
    return sql_with(catalog, text, RulePlanner(), options)


def sql_to_parquet_with[P: Planner](mut catalog: Catalog, text: String, path: String, planner: P, options: PlannerOptions = PlannerOptions(),
                                    metadata: String = "", compression: Int = ZSTD, row_group_rows: Int = DEFAULT_ROW_GROUP) raises -> SqlResult:
    """F-167: `sql_to_parquet` with the planner of your choice."""
    if row_group_rows < 8 or row_group_rows % 8 != 0:
        raise Error("row_group_rows must be a positive multiple of 8, got " + String(row_group_rows))
    var counter = unsafe_alloc[Int64](1)
    counter[0] = -1
    var opts = options.copy()
    opts.sink = ResultSink(path, metadata, compression, row_group_rows, Int(counter))
    var r = SqlResult()
    try:
        r = sql_with(catalog, text, planner, opts)
        if counter[0] < 0:                                   # not streamed (the rows fitted the sort budget, or the query is not a sorting row query)
            write_parquet(path, r.table, metadata, compression, row_group_rows, options.job)
            counter[0] = Int64(r.table.rows())
    except e:
        counter.free()
        raise e^
    r.rows_written = Int(counter[0])
    counter.free()
    return r^


def sql_to_parquet(mut catalog: Catalog, text: String, path: String, options: PlannerOptions = PlannerOptions(), metadata: String = "",
                   compression: Int = ZSTD, row_group_rows: Int = DEFAULT_ROW_GROUP) raises -> SqlResult:
    """Runs one SELECT and writes its rows to a Parquet file, atomically (a temporary file renamed into place; an error or a cancel leaves
    `path` as it was). `ORDER BY` over rows bigger than the sort budget (`PlannerOptions.sort_budget`) is sorted in runs on disk and the merged
    rows go straight into row groups of `row_group_rows`: a file bigger than memory sorted into a new file. The result's `table` is then
    the empty table of the result's columns; `rows_written` says how many rows the file has."""
    return sql_to_parquet_with(catalog, text, path, RulePlanner(), options, metadata, compression, row_group_rows)
