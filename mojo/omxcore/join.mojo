"""The SQL side of the inner equi-join (F-153): what a bound query needs of each table, and the queries that run before and after the join.

The join runs on two in-memory sides (`joincore.mojo`). Each side is first reduced to what the query needs of it: the executor runs it as a row
query over its source (so a Frame's edits, a Parquet scan's row groups and the filter that reads only that side are all applied, and a TEXT column
gets a compact dictionary), or uses the source table itself when nothing filters it. The rest of the query (the filter that reads both sides,
aggregates, ORDER BY, LIMIT) runs over the joined rows: a table made from the pairs, or, when the build side's keys are unique, the probe table with
the build side's columns gathered per block inside the pipeline (`jfront.mojo`).
"""
from .table import Table, SharedTable, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .bind import (BoundQuery, BExpr, Catalog, KeySpec, AggSpec, SortKey, copy_query, B_COL, B_LIT, B_AND, B_AGG, F_COUNT_STAR)
from .joincore import JoinSide, _unique_name


# ── what each side must deliver ──────────────────────────────────────────────────────────────────────

def _mark(q: BoundQuery, e: Int, mut cols: List[Bool], mut seen: List[Bool]):
    if e < 0 or seen[e]:
        return
    seen[e] = True
    ref x = q.rexprs[e]
    if x.op == B_COL:
        cols[Int(x.value)] = True
    _mark(q, x.a, cols, seen)
    _mark(q, x.b, cols, seen)
    for it in x.items:
        _mark(q, it, cols, seen)


def _mark_rows(q: BoundQuery, e: Int, mut seen: List[Bool]):
    if e < 0 or seen[e]:
        return
    seen[e] = True
    _mark_rows(q, q.rexprs[e].a, seen)
    _mark_rows(q, q.rexprs[e].b, seen)
    for it in q.rexprs[e].items:
        _mark_rows(q, it, seen)


def total_columns(q: BoundQuery) -> Int:
    """The columns of all the tables of FROM, numbered one after the other."""
    return q.tab_start[len(q.tab_start) - 1] + q.tab_cols[len(q.tab_cols) - 1]


def live_roots(q: BoundQuery) -> List[Int]:
    """The row-level expressions that still run on the joined rows: the filter the sides leave, the group keys, the aggregates'
    arguments, the select list and ORDER BY of a row query."""
    var roots = List[Int]()
    roots.append(q.where)
    for k in q.keys:
        roots.append(k.expr)
    for a in q.aggs:
        roots.append(a.arg)
    for e in q.routs:
        roots.append(e)
    for o in q.rorder:
        roots.append(o.expr)
    return roots^


def needed_columns(q: BoundQuery, t: Int) -> List[Int]:
    """The global columns of table `t` the joined rows must carry: what the live expressions read, and the join keys."""
    var total = len(q.rexprs)
    var cols = List[Bool](length=total_columns(q), fill=False)
    var seen = List[Bool](length=total, fill=False)
    for r in live_roots(q):
        _mark(q, r, cols, seen)
    for c in q.jk_l:
        cols[c] = True
    for c in q.jk_r:
        cols[c] = True
    var out = List[Int]()
    for c in range(q.tab_start[t], q.tab_start[t] + q.tab_cols[t]):
        if cols[c]:
            out.append(c)
    return out^


def side_query(catalog: Catalog, q: BoundQuery, t: Int, needed: List[Int]) -> BoundQuery:
    """The row query that delivers side `t`: its needed columns, filtered by the conjuncts that read only that side."""
    var sq = copy_query(q)
    var start = q.tab_start[t]
    sq.explain = 0                                                  # the side runs, whatever the query is asked for
    sq.joined = False
    sq.sources = List[Int]()
    sq.sources.append(q.sources[t])
    sq.source = q.sources[t]
    sq.rows = True
    sq.distinct = False
    sq.keys = List[KeySpec]()
    sq.aggs = List[AggSpec]()
    sq.outputs = List[Int]()
    sq.order = List[SortKey]()
    sq.rorder = List[SortKey]()
    sq.having = -1
    sq.limit = -1
    sq.offset = 0
    sq.jk_l = List[Int]()
    sq.jk_r = List[Int]()
    sq.conj_l = List[Int]()
    sq.conj_r = List[Int]()
    sq.conj_x = List[Int]()
    for i in range(len(sq.rexprs)):                                 # this side's columns by their own numbers
        if sq.rexprs[i].op == B_COL:
            var g = Int(sq.rexprs[i].value)
            if g >= start and g < start + q.tab_cols[t]:
                sq.rexprs[i].value = Int64(g - start)
                if sq.rexprs[i].kind == TEXT.value:
                    sq.rexprs[i].src = g - start
    var conj = (q.conj_l.copy() if t == 0 else q.conj_r.copy()) if len(q.sources) == 2 else q.leaf_conj[t].copy()
    sq.where = -1
    if len(conj) > 0:
        var acc = conj[0]
        for k in range(1, len(conj)):
            sq.rexprs.append(BExpr(B_AND, BOOL.value, 0, acc, conj[k], 0, "", 0, 0))
            acc = len(sq.rexprs) - 1
        sq.where = acc
    ref src = catalog.sources[q.sources[t]]
    sq.routs = List[Int]()
    sq.out_names = List[String]()
    sq.out_text = List[String]()
    for g in needed:
        var c = g - start
        sq.rexprs.append(BExpr(B_COL, src.table.kind(c).value, src.table.scale(c), -1, -1, Int64(c), src.table.name(c), 0, 0))
        var at = len(sq.rexprs) - 1
        if src.table.kind(c).value == TEXT.value:
            sq.rexprs[at].src = c
        sq.routs.append(at)
        sq.out_names.append(src.table.name(c))
        sq.out_text.append(src.table.name(c))
    sq.where_text = "the filter that reads only this table"
    sq.order_text = List[String]()
    # the other table's columns (and everything else this query does not read) are not this query's: nothing of them is left
    var seen = List[Bool](length=len(sq.rexprs), fill=False)
    var cols = List[Bool](length=total_columns(q), fill=False)
    _mark_rows(sq, sq.where, seen)
    for e in sq.routs:
        _mark_rows(sq, e, seen)
    for i in range(len(sq.rexprs)):
        if not seen[i]:
            sq.rexprs[i].op = B_LIT
            sq.rexprs[i].kind = INT.value
            sq.rexprs[i].scale = 0
            sq.rexprs[i].value = 0
            sq.rexprs[i].a = -1
            sq.rexprs[i].b = -1
            sq.rexprs[i].items = List[Int]()
    return sq^


def joined_query(q: BoundQuery, lneeded: List[Int], rneeded: List[Int]) -> BoundQuery:
    """The query that runs over the joined table (a single source whose columns are the left side's needed columns, then the right's)."""
    var jq = copy_query(q)
    var total = total_columns(q)
    var jmap = List[Int](length=total, fill=-1)
    for i in range(len(lneeded)):
        jmap[lneeded[i]] = i
    for i in range(len(rneeded)):
        jmap[rneeded[i]] = len(lneeded) + i
    var seen = List[Bool](length=len(jq.rexprs), fill=False)
    var cols = List[Bool](length=total, fill=False)
    for r in live_roots(q):
        _mark(q, r, cols, seen)
    for i in range(len(jq.rexprs)):
        if not seen[i]:                                             # read only before the join (a side's filter): nothing is left of it
            jq.rexprs[i].op = B_LIT
            jq.rexprs[i].kind = INT.value
            jq.rexprs[i].scale = 0
            jq.rexprs[i].value = 0
            jq.rexprs[i].a = -1
            jq.rexprs[i].b = -1
            jq.rexprs[i].items = List[Int]()
            continue
        if jq.rexprs[i].op == B_COL:
            var g = Int(jq.rexprs[i].value)
            jq.rexprs[i].value = Int64(jmap[g])
            if jq.rexprs[i].kind == TEXT.value:
                jq.rexprs[i].src = jmap[g]
    jq.joined = False
    jq.sources = List[Int]()
    jq.sources.append(0)
    jq.source = 0
    jq.jk_l = List[Int]()
    jq.jk_r = List[Int]()
    jq.conj_l = List[Int]()
    jq.conj_r = List[Int]()
    jq.conj_x = List[Int]()
    return jq^


def virtual_schema(left: JoinSide, right: JoinSide, outer_right: Bool = False) raises -> Table:
    """A table with no rows that has the joined columns' names, kinds, scales, dictionaries and NULL counts: what the planner and the
    program's lowering read of the joined source when the join is fused (the rows come from the probe table and the build side's gather)."""
    var t = Table.empty()
    for side in range(2):
        ref sd = left if side == 0 else right
        for c in sd.cols:
            var kind = sd.table.kind(c)
            var col = t.add_column(_unique_name(t, sd.table.name(c)), kind, scale=sd.table.scale(c))
            if kind == TEXT:
                t.share_dictionary(col, sd.table, c)
            t.cols[col].nulls = sd.table.null_count(c)
            if outer_right and side == 1:
                t.cols[col].nulls = max(t.cols[col].nulls, 1)         # a LEFT join: an unmatched row is NULL here, so the program checks
    return t^


# ── three or more tables (F-155): what each step of a join order needs, and the two-table query the last step runs ──────────────────────────

def table_of_column(q: BoundQuery, g: Int) -> Int:
    var t = 0
    for u in range(len(q.tab_start)):
        if g >= q.tab_start[u]:
            t = u
    return t


def multi_roots(q: BoundQuery) -> List[Int]:
    """The row-level expressions of a join of three or more tables that read columns after the leaves' own filters: the group keys, the aggregates'
    arguments, the select list, ORDER BY of a row query. (The conjuncts that read several tables are `q.resid`; checked apart.)"""
    return live_roots(q)


def needed_after(q: BoundQuery, joined: Int, applied: List[Bool]) -> List[Int]:
    """The global columns, of the tables in the bit set `joined`, that later steps read: the select list's expressions, the conjuncts across tables not yet
    checked (`applied[k]` False), and the keys of the equalities to tables not joined yet."""
    var total = total_columns(q)
    var cols = List[Bool](length=total, fill=False)
    var seen = List[Bool](length=len(q.rexprs), fill=False)
    for r in multi_roots(q):
        _mark(q, r, cols, seen)
    for k in range(len(q.resid)):
        if not applied[k]:
            _mark(q, q.resid[k], cols, seen)
    for k in range(len(q.edge_a)):
        var ta = table_of_column(q, q.edge_a[k])
        var tb = table_of_column(q, q.edge_b[k])
        var ina = (joined >> ta) & 1 != 0
        var inb = (joined >> tb) & 1 != 0
        if ina and not inb:
            cols[q.edge_a[k]] = True
        if inb and not ina:
            cols[q.edge_b[k]] = True
    var out = List[Int]()
    for g in range(total):
        if cols[g] and (joined >> table_of_column(q, g)) & 1 != 0:
            out.append(g)
    return out^


def all_needed(q: BoundQuery) -> List[Int]:
    """Every global column some operator reads: the live expressions, the conjuncts across tables, the equalities' keys (not the leaves' own filters:
    their side queries read what they filter)."""
    var total = total_columns(q)
    var cols = List[Bool](length=total, fill=False)
    var seen = List[Bool](length=len(q.rexprs), fill=False)
    for r in multi_roots(q):
        _mark(q, r, cols, seen)
    for e in q.resid:
        _mark(q, e, cols, seen)
    for g in q.edge_a:
        cols[g] = True
    for g in q.edge_b:
        cols[g] = True
    var out = List[Int]()
    for g in range(total):
        if cols[g]:
            out.append(g)
    return out^


def _neutral(mut x: BExpr):
    x.op = B_LIT
    x.kind = INT.value
    x.scale = 0
    x.value = 0
    x.a = -1
    x.b = -1
    x.items = List[Int]()


def _remap_columns(mut jq: BoundQuery, colmap: List[Int], roots: List[Int]):
    """The expressions reachable from `roots` read columns by their new numbers (`colmap[old]`); everything else is neutral."""
    var seen = List[Bool](length=len(jq.rexprs), fill=False)
    for r in roots:
        _mark_rows(jq, r, seen)
    for i in range(len(jq.rexprs)):
        if not seen[i]:
            _neutral(jq.rexprs[i])
            continue
        if jq.rexprs[i].op == B_COL:
            var g = Int(jq.rexprs[i].value)
            jq.rexprs[i].value = Int64(colmap[g])
            if jq.rexprs[i].kind == TEXT.value:
                jq.rexprs[i].src = colmap[g]


def filter_query(q: BoundQuery, conj: List[Int], gcols: List[Int], t: SharedTable) -> BoundQuery:
    """The row query that keeps the rows of table `t` (whose columns are the global columns `gcols`, in order) that pass the conjuncts `conj`: every column
    comes out. Used for a condition across tables as soon as the tables it reads are joined."""
    var fq = copy_query(q)
    var total = total_columns(q)
    var colmap = List[Int](length=total, fill=-1)
    for i in range(len(gcols)):
        colmap[gcols[i]] = i
    fq.joined = False
    fq.sources = List[Int]()
    fq.sources.append(0)
    fq.source = 0
    fq.rows = True
    fq.distinct = False
    fq.keys = List[KeySpec]()
    fq.aggs = List[AggSpec]()
    fq.outputs = List[Int]()
    fq.order = List[SortKey]()
    fq.rorder = List[SortKey]()
    fq.having = -1
    fq.limit = -1
    fq.offset = 0
    fq.explain = 0
    fq.jk_l = List[Int]()
    fq.jk_r = List[Int]()
    fq.conj_l = List[Int]()
    fq.conj_r = List[Int]()
    fq.conj_x = List[Int]()
    fq.leaf_conj = List[List[Int]]()
    fq.edge_a = List[Int]()
    fq.edge_b = List[Int]()
    fq.resid = List[Int]()
    fq.tab_names = List[String]()
    fq.tab_names.append("rows")
    fq.tab_alias = List[String]()
    fq.tab_alias.append(String())
    fq.tab_start = List[Int]()
    fq.tab_start.append(0)
    fq.tab_cols = List[Int]()
    fq.tab_cols.append(len(gcols))
    var acc = conj[0]
    for k in range(1, len(conj)):
        fq.rexprs.append(BExpr(B_AND, BOOL.value, 0, acc, conj[k], 0, "", 0, 0))
        acc = len(fq.rexprs) - 1
    fq.where = acc
    var roots = List[Int]()
    roots.append(acc)
    fq.routs = List[Int]()
    fq.out_names = List[String]()
    fq.out_text = List[String]()
    for i in range(len(gcols)):
        fq.rexprs.append(BExpr(B_COL, t.kind(i).value, t.scale(i), -1, -1, Int64(i), t.name(i), 0, 0))
        var at = len(fq.rexprs) - 1
        if t.kind(i).value == TEXT.value:
            fq.rexprs[at].src = i
        fq.routs.append(at)
        roots.append(at)
        fq.out_names.append(t.name(i))
        fq.out_text.append(t.name(i))
    # the conjuncts' own columns by the new numbers (the columns just added are already in them)
    var seen = List[Bool](length=len(fq.rexprs), fill=False)
    _mark_rows(fq, acc, seen)
    for i in range(len(fq.rexprs) - len(gcols)):
        if not seen[i]:
            _neutral(fq.rexprs[i])
            continue
        if fq.rexprs[i].op == B_COL:
            var g = Int(fq.rexprs[i].value)
            fq.rexprs[i].value = Int64(colmap[g])
            if fq.rexprs[i].kind == TEXT.value:
                fq.rexprs[i].src = colmap[g]
    fq.where_text = "the condition across the tables joined so far"
    fq.order_text = List[String]()
    return fq^


def final_query(q: BoundQuery, left: List[Int], right: List[Int], lt: SharedTable, rt: SharedTable, key_l: List[Int], key_r: List[Int], conj: List[Int],
                rconj: List[Int] = List[Int](), rconj_text: List[String] = List[String]()) -> BoundQuery:
    """The two-table query that runs the last step of a join order: the rows joined so far (`lt`, whose columns are the global columns `left`) and the
    last table (`rt`, the global columns `right`), joined on the pairs `key_l[i] = key_r[i]` (global numbers), then the rest of the query. `conj`: the conjuncts
    across tables still to check, now over the joined rows."""
    var fq = copy_query(q)
    var total = total_columns(q)
    var colmap = List[Int](length=total, fill=-1)
    for i in range(len(left)):
        colmap[left[i]] = i
    for i in range(len(right)):
        colmap[right[i]] = len(left) + i
    var roots = live_roots(q)
    for e in conj:
        roots.append(e)
    for e in rconj:
        roots.append(e)
    _remap_columns(fq, colmap, roots)
    fq.joined = True
    fq.join_kind = 0
    fq.explain = 0
    fq.sources = List[Int]()
    fq.sources.append(0)
    fq.sources.append(1)
    fq.source = 0
    fq.tab_names = List[String]()
    fq.tab_names.append("rows joined so far")
    fq.tab_names.append("last table")
    fq.tab_alias = List[String]()
    fq.tab_alias.append(String())
    fq.tab_alias.append(String())
    fq.tab_start = List[Int]()
    fq.tab_start.append(0)
    fq.tab_start.append(len(left))
    fq.tab_cols = List[Int]()
    fq.tab_cols.append(len(left))
    fq.tab_cols.append(len(right))
    fq.jk_l = List[Int]()
    fq.jk_r = List[Int]()
    fq.jk_l_names = List[String]()
    fq.jk_r_names = List[String]()
    for k in range(len(key_l)):
        fq.jk_l.append(colmap[key_l[k]])
        fq.jk_r.append(colmap[key_r[k]])
        fq.jk_l_names.append(lt.name(colmap[key_l[k]]))
        fq.jk_r_names.append(rt.name(colmap[key_r[k]] - len(left)))
    fq.conj_l = List[Int]()
    fq.conj_r = rconj.copy()
    fq.conj_x = List[Int]()
    fq.conj_l_text = List[String]()
    fq.conj_r_text = rconj_text.copy()
    fq.conj_x_text = List[String]()
    fq.where = -1
    if len(conj) > 0:
        var acc = conj[0]
        fq.conj_x.append(conj[0])
        for k in range(1, len(conj)):
            fq.conj_x.append(conj[k])
            fq.rexprs.append(BExpr(B_AND, BOOL.value, 0, acc, conj[k], 0, "", 0, 0))
            acc = len(fq.rexprs) - 1
        fq.where = acc
    fq.where_text = "the conditions across the tables"
    fq.leaf_conj = List[List[Int]]()
    fq.edge_a = List[Int]()
    fq.edge_b = List[Int]()
    fq.resid = List[Int]()
    fq.using_cols = List[String]()
    return fq^


# ── a side's filter left to the pipeline, and counted (F-155) ────────────────────────────────────────────────────────

def count_query(catalog: Catalog, q: BoundQuery, t: Int) -> BoundQuery:
    """`SELECT count(*)` over side `t` with the conjuncts that read only it: its exact filtered row count without writing the rows."""
    var cq = side_query(catalog, q, t, List[Int]())
    cq.rows = False
    cq.routs = List[Int]()
    cq.keys = List[KeySpec]()
    cq.aggs = List[AggSpec]()
    cq.aggs.append(AggSpec(F_COUNT_STAR, -1, 0.0, INT.value, 0, "count_star()", "count_star:*:0"))
    cq.pexprs = List[BExpr]()
    cq.pexprs.append(BExpr(B_AGG, INT.value, 0, -1, -1, 0, "count_star()", 0, 0))
    cq.outputs = List[Int]()
    cq.outputs.append(0)
    cq.out_names = List[String]()
    cq.out_names.append("n")
    cq.out_text = List[String]()
    cq.out_text.append("count(*)")
    return cq^


def with_probe_filter(q: BoundQuery, t: Int) -> BoundQuery:
    """The query with side `t`'s own conjuncts added to the filter that runs on the joined rows: the probe side is read unfiltered and the pipeline applies
    its filter (no filtered copy of it is written)."""
    var q2 = copy_query(q)
    var conj = q.conj_l.copy() if t == 0 else q.conj_r.copy()
    for c in conj:
        if q2.where < 0:
            q2.where = c
        else:
            q2.rexprs.append(BExpr(B_AND, BOOL.value, 0, q2.where, c, 0, "", 0, 0))
            q2.where = len(q2.rexprs) - 1
    return q2^


# ── a star: one probe table, several dimensions with unique keys, one pass (F-155) ──────────────────────────────────

def final_needed(q: BoundQuery, conj: List[Int]) -> List[Bool]:
    """The global columns the rest of the query reads over the joined rows: the live expressions and the conjuncts `conj` across tables."""
    var cols = List[Bool](length=total_columns(q), fill=False)
    var seen = List[Bool](length=len(q.rexprs), fill=False)
    for r in multi_roots(q):
        _mark(q, r, cols, seen)
    for e in conj:
        _mark(q, e, cols, seen)
    return cols^


def virtual_schema_multi(tabs: List[SharedTable], cols: List[List[Int]]) raises -> Table:
    """A table with no rows that has the star's columns (the probe table's, then each dimension's): names, kinds, scales, dictionaries and NULL counts."""
    var t = Table.empty()
    for s in range(len(tabs)):
        for c in cols[s]:
            var kind = tabs[s].kind(c)
            var col = t.add_column(_unique_name(t, tabs[s].name(c)), kind, scale=tabs[s].scale(c))
            if kind == TEXT:
                t.share_dictionary(col, tabs[s], c)
            t.cols[col].nulls = tabs[s].null_count(c)
    return t^


def star_query(q: BoundQuery, vglobal: List[Int], conj: List[Int]) -> BoundQuery:
    """The single-source query over a star's virtual columns (`vglobal[i]` is the global column that virtual column i is): the rest of the query, with the
    conjuncts `conj` across tables (and the probe table's own filter) as its WHERE."""
    var sq = copy_query(q)
    var colmap = List[Int](length=total_columns(q), fill=-1)
    for i in range(len(vglobal)):
        colmap[vglobal[i]] = i
    var roots = live_roots(q)
    for e in conj:
        roots.append(e)
    _remap_columns(sq, colmap, roots)
    sq.joined = False
    sq.join_kind = 0
    sq.explain = 0
    sq.sources = List[Int]()
    sq.sources.append(0)
    sq.source = 0
    sq.tab_names = List[String]()
    sq.tab_names.append("join")
    sq.tab_alias = List[String]()
    sq.tab_alias.append(String())
    sq.tab_start = List[Int]()
    sq.tab_start.append(0)
    sq.tab_cols = List[Int]()
    sq.tab_cols.append(len(vglobal))
    sq.jk_l = List[Int]()
    sq.jk_r = List[Int]()
    sq.conj_l = List[Int]()
    sq.conj_r = List[Int]()
    sq.conj_x = List[Int]()
    sq.leaf_conj = List[List[Int]]()
    sq.edge_a = List[Int]()
    sq.edge_b = List[Int]()
    sq.resid = List[Int]()
    sq.where = -1
    if len(conj) > 0:
        var acc = conj[0]
        for k in range(1, len(conj)):
            sq.rexprs.append(BExpr(B_AND, BOOL.value, 0, acc, conj[k], 0, "", 0, 0))
            acc = len(sq.rexprs) - 1
        sq.where = acc
    sq.where_text = "the conditions across the tables"
    return sq^
