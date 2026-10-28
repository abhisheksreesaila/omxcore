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
from .sql import (SqlStatement, SqlQuery, SqlExpr, SelectItem, TableRef, Cte, parse_query, sql_error, lower_ascii, EXPLAIN_NONE, EXPLAIN_ANALYZE,
                  SET_NONE, SET_UNION_ALL, SET_UNION, SET_INTERSECT, SET_EXCEPT, CTE_AUTO, CTE_MATERIALIZED, CTE_INLINED,
                  X_SCALAR_Q, X_IN_Q, X_EXISTS_Q)
from .bind import Catalog
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
    var steps_before = len(run.steps)
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
    var t0 = perf_counter_ns()
    if st.set_op != SET_NONE:
        raise sql_error(st.table_line, st.table_col, _set_name(st.set_op) + " is not supported yet")
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
    if st.qualify >= 0:
        return qualify_run(run, catalog, st^, planner, opts, explain)
    var r = run_plain_explained(run, catalog, st^, planner, opts, explain)
    return r^


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


def qualify_run[P: Planner](mut run: QueryRun, mut catalog: Catalog, var st: SqlStatement, planner: P, opts: PlannerOptions,
                            explain: Int) raises -> SqlResult:
    return run_plain_explained(run, catalog, st^, planner, opts, explain)


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
