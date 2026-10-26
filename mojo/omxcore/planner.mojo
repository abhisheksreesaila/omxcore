"""The physical planner (F-150): how a bound query runs, decided from statistics alone.

`Planner` is a trait: `plan(query, stats, options)` returns a `PhysicalPlan` and the reasons for each decision in
`notes`, which EXPLAIN prints and the tests assert. `RulePlanner` is the first implementation; a cost-based one
(F-155) replaces it without touching the executor. `PlannerOptions` forces each choice, so a test can reach every path
and check which one ran.

What it decides today: the group strategy.
- no GROUP BY: one group, the plain kernel;
- dense: the keys are columns whose values lie in a known range (a TEXT column's dictionary, BOOL, the min / max of an
  integer, decimal or date column), laid out mixed-radix with a NULL slot per nullable key, and the kernel addresses
  the group directly: no hashing, the fastest path while the domain is not much bigger than the rows;
- hashed: key words go through per-worker hash tables and a partitioned merge (`hashagg_cpu`): any keys, any range.
"""
from .job import JobCtl                                   # F-175
from .table import TEXT, BOOL, FLOAT
from .plan import LogicalPlan, P_SORT, P_LIMIT
from .bind import (BoundQuery, SRC_TABLE, SRC_FRAME, SRC_SCAN, B_COL, B_LIT, B_NULL, B_ADD, B_SUB, B_MUL, B_IDIV, B_IMOD,
                   B_NEG, B_RESCALE, B_CASE, B_COALESCE, B_DISTINCT, B_BOOLIS, B_CAST, B_TFN, B_CONCAT, B_STR,
                   B_DATEPART, B_DATETRUNC, B_ADDMONTHS, B_DATEDIFF, B_QBIN)
from .sqlfn import date_part, date_trunc_days, add_months, DP_YEAR, DP_MONTH, DP_DAY, DP_DOW, DP_ISODOW, DP_DOY, DP_QUARTER, DP_WEEK, DP_ISOYEAR, DP_DECADE, DP_CENTURY, DP_MILLENNIUM

comptime BUILD_AUTO = 0
comptime BUILD_LEFT = 1
comptime BUILD_RIGHT = 2
comptime GROUP_AUTO = 0
comptime GROUP_DENSE = 1
comptime GROUP_HASHED = 2
comptime G_NONE = 0          # a plan's strategy: one group
comptime G_DENSE = 1
comptime G_HASHED = 2
comptime S_AGG = 0           # a plan's sink (F-152): the aggregate sinks (dense, hashed, holistic), or ...
comptime S_ROWS = 1          # ... materialized rows (a row query)
comptime TOPK_MAX = 4096     # the most rows a top-k keeps in per-worker heaps; past it the rows are sorted whole
comptime TOPK_FACTOR = 8     # a top-k pays off when the source has more than this many times the rows wanted

comptime DEFAULT_DENSE_LIMIT = 4_194_304     # the most group slots a dense layout may take
comptime SPARSE_FACTOR = 4                   # past this many slots a row, the domain is too sparse for dense (unless small)
comptime SMALL_DOMAIN = 65_536               # a domain this small is always dense


struct PlannerOptions(Copyable, Movable):
    var job: JobCtl                          # F-175: the background job this query runs for (cancel and progress); JobCtl() for a plain call
    var group: Int                           # GROUP_AUTO, GROUP_DENSE or GROUP_HASHED
    var dense_limit: Int
    var threads: Int                         # 0: every core
    var fuse: Bool                           # a join with unique build keys is fused into the pipeline (False: its pairs are materialized)
    # F-155: joins. Each choice the planner makes can be forced, so a test reaches every order and path
    var written_order: Bool                  # join the tables in the order FROM writes them (an error when that order meets a table no equality reaches yet)
    var join_order: List[Int]                # the tables' positions in FROM, in the order to join them (empty: the planner chooses)
    var build_side: Int                      # BUILD_AUTO, BUILD_LEFT (the rows joined so far / the first table) or BUILD_RIGHT (the table joined in / the second)
    var join_budget: Int                     # the bytes of joined values a materialized join may take (0: the engine's default)
    var stay_within: Float64                 # the written order is kept when the best order's estimated cost is not this much cheaper

    def __init__(out self):
        self.job = JobCtl()
        self.written_order = False
        self.join_order = List[Int]()
        self.build_side = BUILD_AUTO
        self.join_budget = 0
        self.stay_within = 0.10
        self.fuse = True
        self.group = GROUP_AUTO
        self.dense_limit = DEFAULT_DENSE_LIMIT
        self.threads = 0


struct ColStat(Copyable, Movable):
    var kind: Int
    var scale: Int
    var nullable: Bool           # may hold a NULL (unknown counts as yes)
    var has_range: Bool
    var lo: Int64
    var hi: Int64
    var words: Int               # a TEXT column's dictionary size

    def __init__(out self, kind: Int, scale: Int):
        self.kind = kind
        self.scale = scale
        self.nullable = True
        self.has_range = False
        self.lo = 0
        self.hi = 0
        self.words = 0


struct Stats(Movable):
    var source: String
    var kind: Int                # SRC_TABLE, SRC_FRAME or SRC_SCAN
    var rows: Int
    var cols: List[ColStat]
    var row_groups: Int
    var edits: Int               # a frame: the changed and appended rows that go through the delta
    var version: Int             # a frame: its edit counter when it was registered
    var distrusted: Bool         # a file's footer statistics were found wrong in this query: none is used

    def __init__(out self, source: String, kind: Int, rows: Int):
        self.source = source
        self.kind = kind
        self.rows = rows
        self.cols = List[ColStat]()
        self.row_groups = 0
        self.edits = 0
        self.version = 0
        self.distrusted = False


@fieldwise_init
struct KeyLayout(Copyable, Movable):
    var expr: Int                # the key's row-level expression
    var kind: Int
    var scale: Int
    var nullable: Bool
    var lo: Int64                # dense: the smallest value (a TEXT code or BOOL starts at 0)
    var radix: Int               # dense: the slots this key takes, its NULL slot included
    var stride: Int              # dense: the slot step of this key
    var src: Int                 # the source column of a plain-column key, else -1
    var words: Int               # hashed: key words (the value, and a NULL flag when nullable)
    var guarded: Bool            # dense: its range came from a file's footer statistics, so rows outside it are looked for


struct PhysicalPlan(Movable):
    var group: Int
    var keys: List[KeyLayout]
    var domain: Int
    var holistic: Bool
    var source_kind: Int
    var notes: List[String]
    var sink: Int                    # S_AGG or S_ROWS
    var top_k: Int                   # rows: the rows wanted when per-worker heaps replace the full sort (0: sort them all, or no ORDER BY)
    var early_stop: Bool             # rows: no ORDER BY, so the scan stops once it has limit + offset rows

    def __init__(out self):
        self.sink = S_AGG
        self.top_k = 0
        self.early_stop = False
        self.group = G_NONE
        self.keys = List[KeyLayout]()
        self.domain = 1
        self.holistic = False
        self.source_kind = SRC_TABLE
        self.notes = List[String]()

    def key_words(self) -> Int:
        var n = 0
        for k in self.keys:
            n += k.words
        return n


trait Planner:
    def plan(self, q: BoundQuery, st: Stats, opts: PlannerOptions) raises -> PhysicalPlan:
        ...

    def join_order(self, js: JoinStats, opts: PlannerOptions) raises -> JoinPlan:
        """The order to join three or more tables in, and the build side of each join, from the tables' statistics alone. The default is the greedy
        rule `rule_join_order`; a planner with a better search (DPccp, sampled selectivities) overrides it and the executor does not change."""
        return rule_join_order(js, opts)


def expr_nullable(q: BoundQuery, e: Int, st: Stats) -> Bool:
    """Can the row-level expression be NULL? (a nullable column, a NULL literal, a division that may meet zero)"""
    ref x = q.rexprs[e]
    if x.op == B_COL:
        return st.cols[Int(x.value)].nullable
    if x.op == B_NULL:
        return True
    # F-156: the expressions whose operands are not just `a` and `b`
    if x.op == B_LIT or x.op == B_STR:
        return False
    if x.op == B_DISTINCT or x.op == B_BOOLIS:                     # never NULL: NULL is a value to them
        return False
    if x.op == B_CASE:                                             # the values (not the conditions): the THENs and the ELSE
        var nw = Int(x.value)
        for k in range(nw):
            if expr_nullable(q, x.items[2 * k + 1], st):
                return True
        return expr_nullable(q, x.items[2 * nw], st)
    if x.op == B_COALESCE:                                         # NULL only where every argument is
        for it in x.items:
            if not expr_nullable(q, it, st):
                return False
        return True
    if x.op == B_CAST:                                             # a TRY_CAST can fail into NULL
        return (Int(x.value) & 1) != 0 or expr_nullable(q, x.a, st)
    if x.op == B_QBIN:                                             # NULL where the value is
        return expr_nullable(q, x.a, st)
    if x.op == B_CONCAT and x.value != 0:                          # concat() reads NULL as ''
        return False
    if x.op == B_TFN or x.op == B_CONCAT or x.op == B_DATEPART or x.op == B_DATETRUNC or x.op == B_ADDMONTHS or x.op == B_DATEDIFF:
        var rr = expr_nullable(q, x.a, st)
        if x.b >= 0:
            rr = rr or expr_nullable(q, x.b, st)
        return rr
    if x.op >= 50 and x.op != B_CASE:                              # an operator this function does not know: it may be NULL
        return True
    var r = False
    if x.op == B_IDIV or x.op == B_IMOD:                           # a division is NULL where the divisor is zero: not by a nonzero constant
        if not (q.rexprs[x.b].op == B_LIT and q.rexprs[x.b].value != 0):
            r = True
    if x.a >= 0:
        r = r or expr_nullable(q, x.a, st)
    if x.b >= 0:
        r = r or expr_nullable(q, x.b, st)
    return r


struct Interval(Copyable, Movable):
    """The values an integer-like expression can take, when the statistics bound them."""
    var ok: Bool
    var lo: Int64
    var hi: Int64

    def __init__(out self, ok: Bool, lo: Int64 = 0, hi: Int64 = 0):
        self.ok = ok
        self.lo = lo
        self.hi = hi


comptime _BOUND = Int64(1) << 60             # intervals past this are given up on (the arithmetic stays far from int64's edge)


def _in_bound(x: Int64) -> Bool:
    return -_BOUND < x < _BOUND


def interval(q: BoundQuery, e: Int, st: Stats) -> Interval:
    """Interval arithmetic over a row-level expression from the columns' ranges: a key such as `x % 10`, `x // 100` or `a + 1` then
    has a known domain and can use the dense strategy. Not ok when a column has no range, or the expression is a division by a
    non-constant."""
    ref x = q.rexprs[e]
    if x.op == B_LIT:
        return Interval(True, x.value, x.value)
    if x.op == B_COL:
        ref cs = st.cols[Int(x.value)]
        if x.kind == TEXT.value:
            return Interval(True, 0, Int64(max(cs.words - 1, 0)))
        if x.kind == BOOL.value:
            return Interval(True, 0, 1)
        return Interval(cs.has_range and _in_bound(cs.lo) and _in_bound(cs.hi), cs.lo, cs.hi)
    if x.op == B_DATEPART or x.op == B_DATETRUNC or x.op == B_ADDMONTHS:
        # F-156: a date function of a column whose range is known stays in a range: the part of a date has a fixed one (month 1..12), a year, a truncated date or
        # a date moved by a constant number of months grows with the date, so the ends of the range give the ends of the answer: GROUP BY EXTRACT(year ...) is dense
        var da = interval(q, x.a, st)
        if not da.ok or da.lo < -719162 or da.hi > 2932896:
            return Interval(False)
        var part = Int(x.value)
        if x.op == B_DATEPART:
            if part == DP_MONTH:
                return Interval(True, 1, 12)
            if part == DP_DAY:
                return Interval(True, 1, 31)
            if part == DP_DOW:
                return Interval(True, 0, 6)
            if part == DP_ISODOW:
                return Interval(True, 1, 7)
            if part == DP_QUARTER:
                return Interval(True, 1, 4)
            if part == DP_DOY:
                return Interval(True, 1, 366)
            if part == DP_WEEK:
                return Interval(True, 1, 53)
            if part == DP_YEAR or part == DP_ISOYEAR or part == DP_DECADE or part == DP_CENTURY or part == DP_MILLENNIUM:
                return Interval(True, Int64(date_part(Int(da.lo), part)), Int64(date_part(Int(da.hi), part)))
            return Interval(False)
        if x.op == B_DATETRUNC:
            return Interval(True, Int64(date_trunc_days(Int(da.lo), part)), Int64(date_trunc_days(Int(da.hi), part)))
        ref n = q.rexprs[x.b]
        if n.op != B_LIT or n.value > 1000000 or n.value < -1000000:
            return Interval(False)
        return Interval(True, Int64(add_months(Int(da.lo), Int(n.value))), Int64(add_months(Int(da.hi), Int(n.value))))
    if x.op == B_QBIN:
        return Interval(True, 1, x.value)
    if x.op == B_NEG or x.op == B_RESCALE:
        var a = interval(q, x.a, st)
        if not a.ok:
            return Interval(False)
        if x.op == B_NEG:
            return Interval(True, -a.hi, -a.lo)
        var m = Int64(1)
        for _ in range(Int(x.value)):
            m *= 10
        if not (_in_bound(a.lo) and _in_bound(a.hi)) or (a.hi != 0 and abs(a.hi) > _BOUND // m) or (a.lo != 0 and abs(a.lo) > _BOUND // m):
            return Interval(False)
        return Interval(True, a.lo * m, a.hi * m)
    if x.op == B_ADD or x.op == B_SUB or x.op == B_MUL or x.op == B_IDIV or x.op == B_IMOD:
        var a = interval(q, x.a, st)
        var b = interval(q, x.b, st)
        if not a.ok or not b.ok:
            return Interval(False)
        if x.op == B_ADD:
            return Interval(_in_bound(a.lo + b.lo) and _in_bound(a.hi + b.hi), a.lo + b.lo, a.hi + b.hi)
        if x.op == B_SUB:
            return Interval(_in_bound(a.lo - b.hi) and _in_bound(a.hi - b.lo), a.lo - b.hi, a.hi - b.lo)
        if x.op == B_MUL:
            var corners = [a.lo, a.hi]
            var lo = Int64.MAX
            var hi = Int64.MIN
            for u in corners:
                for v in [b.lo, b.hi]:
                    var big = Float64(u) * Float64(v)
                    if big > 1.0e18 or big < -1.0e18:
                        return Interval(False)
                    lo = min(lo, u * v)
                    hi = max(hi, u * v)
            return Interval(_in_bound(lo) and _in_bound(hi), lo, hi)
        if b.lo != b.hi or b.lo == 0:                                # a division by a constant only
            return Interval(False)
        var c = b.lo
        if x.op == B_IDIV:
            var p = a.lo // c if (a.lo >= 0) == (c > 0) or a.lo % c == 0 else a.lo // c + 1     # truncating
            var r = a.hi // c if (a.hi >= 0) == (c > 0) or a.hi % c == 0 else a.hi // c + 1
            return Interval(True, min(p, r), max(p, r))
        var m = (abs(c) - 1)
        if a.lo >= 0:
            return Interval(True, 0, m)
        if a.hi <= 0:
            return Interval(True, -m, 0)
        return Interval(True, -m, m)
    return Interval(False)


# ── join order (F-155) ───────────────────────────────────────────────────────────────────────────────

struct JoinStats(Copyable, Movable):
    """What a join order is chosen from: the tables (leaves) of FROM in written order with their base rows and their rows after their own filters (exact:
    the leaf's filter has run), and the equalities between them (one entry per pair of columns)."""
    var names: List[String]
    var base_rows: List[Int]
    var rows: List[Int]
    var edge_a: List[Int]            # the leaves an equality joins
    var edge_b: List[Int]
    var ndv_a: List[Int]             # the distinct values of the equality's column in each of the two leaves (after their filters), -1 unknown
    var ndv_b: List[Int]

    def __init__(out self):
        self.names = List[String]()
        self.base_rows = List[Int]()
        self.rows = List[Int]()
        self.edge_a = List[Int]()
        self.edge_b = List[Int]()
        self.ndv_a = List[Int]()
        self.ndv_b = List[Int]()

    def connected(self, a: Int, b: Int) -> Bool:
        for k in range(len(self.edge_a)):
            if (self.edge_a[k] == a and self.edge_b[k] == b) or (self.edge_a[k] == b and self.edge_b[k] == a):
                return True
        return False


struct JoinStep(Copyable, Movable):
    var leaf: Int                    # the table joined in
    var build_new: Bool              # the hash table is built on the table joined in (else on the rows joined so far)
    var est_rows: Int                # the estimated rows after the step
    var cost: Int                    # the estimated cost of the step (rows hashed twice, probed once, produced once)

    def __init__(out self, leaf: Int, build_new: Bool, est_rows: Int, cost: Int):
        self.leaf = leaf
        self.build_new = build_new
        self.est_rows = est_rows
        self.cost = cost


struct JoinPlan(Movable):
    var start: Int
    var steps: List[JoinStep]
    var cost: Int
    var notes: List[String]          # every decision with its numbers: EXPLAIN prints them

    def __init__(out self):
        self.start = 0
        self.steps = List[JoinStep]()
        self.cost = 0
        self.notes = List[String]()


def _min_int(a: Int, b: Int) -> Int:
    return a if a < b else b


def _step_estimate(js: JoinStats, in_set: List[Bool], acc_rows: Int, leaf: Int) -> Int:
    """The rows after joining `leaf` to the rows joined so far. Each equality to a table already joined gives rows x leaf rows / the larger of the two
    columns' distinct counts (the rows joined so far hold at most `acc_rows` distinct values of theirs); with no distinct counts known, the smaller of
    the two base tables stands for the key's domain (a foreign key into the smaller). The most selective equality counts (a composite key is at least
    that selective). Never more than the product."""
    var best = -1
    var f = js.rows[leaf]
    for k in range(len(js.edge_a)):
        var other = -1
        var d_leaf = -1
        var d_other = -1
        if js.edge_a[k] == leaf and in_set[js.edge_b[k]]:
            other = js.edge_b[k]
            d_leaf = js.ndv_a[k]
            d_other = js.ndv_b[k]
        elif js.edge_b[k] == leaf and in_set[js.edge_a[k]]:
            other = js.edge_a[k]
            d_leaf = js.ndv_b[k]
            d_other = js.ndv_a[k]
        if other < 0:
            continue
        var div: Int
        if d_leaf >= 0 and d_other >= 0:
            div = max(max(d_leaf, _min_int(d_other, max(acc_rows, 1))), 1)
        else:
            div = _min_int(max(js.base_rows[other], 1), max(js.base_rows[leaf], 1))
        var est = Int(Float64(acc_rows) * Float64(f) / Float64(div))
        if best < 0 or est < best:
            best = est
    if best < 0:
        return -1
    return max(min(best, acc_rows * f), 1 if (acc_rows > 0 and f > 0) else 0)


def _step_cost(acc_rows: Int, leaf_rows: Int, out_rows: Int) -> Int:
    var small = _min_int(acc_rows, leaf_rows)
    var big = max(acc_rows, leaf_rows)
    return 2 * small + big + out_rows


def _plan_from(js: JoinStats, order: List[Int], opts: PlannerOptions) -> JoinPlan:
    """Costs a given order (it must connect: every table after the first has an equality to one before it; `ok` of the caller checks)."""
    var plan = JoinPlan()
    var n = len(js.names)
    var in_set = List[Bool](length=n, fill=False)
    plan.start = order[0]
    in_set[order[0]] = True
    var acc = js.rows[order[0]]
    for i in range(1, len(order)):
        var leaf = order[i]
        var est = _step_estimate(js, in_set, acc, leaf)
        var build_new = js.rows[leaf] <= acc
        if opts.build_side == BUILD_LEFT:
            build_new = False
        elif opts.build_side == BUILD_RIGHT:
            build_new = True
        var c = _step_cost(acc, js.rows[leaf], max(est, 0))
        plan.steps.append(JoinStep(leaf, build_new, est, c))
        plan.cost += c
        in_set[leaf] = True
        acc = max(est, 0)
    return plan^


def _order_connects(js: JoinStats, order: List[Int]) -> Bool:
    var n = len(js.names)
    var in_set = List[Bool](length=n, fill=False)
    in_set[order[0]] = True
    for i in range(1, len(order)):
        var ok = False
        for k in range(len(js.edge_a)):
            if (js.edge_a[k] == order[i] and in_set[js.edge_b[k]]) or (js.edge_b[k] == order[i] and in_set[js.edge_a[k]]):
                ok = True
        if not ok:
            return False
        in_set[order[i]] = True
    return True


def _greedy_from(js: JoinStats, start: Int) -> List[Int]:
    """From `start`, the connected table that gives the fewest rows next (ties: the smaller table), until all are joined."""
    var n = len(js.names)
    var in_set = List[Bool](length=n, fill=False)
    var order = List[Int]()
    order.append(start)
    in_set[start] = True
    var acc = js.rows[start]
    while len(order) < n:
        var best = -1
        var best_est = 0
        for leaf in range(n):
            if in_set[leaf]:
                continue
            var est = _step_estimate(js, in_set, acc, leaf)
            if est < 0:
                continue
            if best < 0 or est < best_est or (est == best_est and js.rows[leaf] < js.rows[best]):
                best = leaf
                best_est = est
        if best < 0:
            break
        order.append(best)
        in_set[best] = True
        acc = max(best_est, 0)
    return order^


def _join_names(js: JoinStats, order: List[Int]) -> String:
    var s = String()
    for i in range(len(order)):
        s += (" > " if i > 0 else "") + js.names[order[i]]
    return s


def rule_join_order(js: JoinStats, opts: PlannerOptions) raises -> JoinPlan:
    """The rule-based join order: every table is tried as the start of a greedy chain (the next table is the connected one that leaves the fewest rows), the
    cheapest chain by the cost model wins, and the written order is kept when it connects and costs within `stay_within` of the best. The build side of
    each step is the smaller input (hash table on the table joined in when it has fewer rows than the rows joined so far). Forced: `join_order`,
    `written_order`, `build_side`."""
    var n = len(js.names)
    var written = List[Int]()
    for i in range(n):
        written.append(i)
    var notes = List[String]()
    var plan: JoinPlan
    if len(opts.join_order) > 0:
        if len(opts.join_order) != n:
            raise Error("join order: it names " + String(len(opts.join_order)) + " tables, the query joins " + String(n))
        var seen = List[Bool](length=n, fill=False)
        for t in opts.join_order:
            if t < 0 or t >= n or seen[t]:
                raise Error("join order: each of the " + String(n) + " tables (0 to " + String(n - 1) + ") appears once")
            seen[t] = True
        if not _order_connects(js, opts.join_order):
            raise Error("join order " + _join_names(js, opts.join_order) + " reaches a table no equality joins yet (a cross product): choose another order")
        plan = _plan_from(js, opts.join_order, opts)
        notes.append("join order: " + _join_names(js, opts.join_order) + " (forced by the options)")
    elif opts.written_order and _order_connects(js, written):
        plan = _plan_from(js, written, opts)
        notes.append("join order: " + _join_names(js, written) + " (the written order, forced by the options)")
    else:
        var best_order = List[Int]()
        var best_cost = -1
        for start in range(n):
            var o = _greedy_from(js, start)
            if len(o) != n:
                continue
            var c = _plan_from(js, o, opts).cost
            if best_cost < 0 or c < best_cost:
                best_cost = c
                best_order = o^
        if best_cost < 0:
            raise Error("the tables are not all reached by equalities: a cross join is not supported")
        var wcost = -1
        if _order_connects(js, written):
            wcost = _plan_from(js, written, opts).cost
        if opts.written_order:
            notes.append("the written order meets a table no equality reaches yet: the planner chose the order")
        if wcost >= 0 and Float64(wcost) <= Float64(best_cost) * (1.0 + opts.stay_within):
            plan = _plan_from(js, written, opts)
            notes.append("join order: " + _join_names(js, written) + " (the written order: its cost " + commas(wcost) + " is within " + String(Int(opts.stay_within * 100.0)) +
                         "% of the best found, " + _join_names(js, best_order) + " at " + commas(best_cost) + ")")
        else:
            plan = _plan_from(js, best_order, opts)
            notes.append("join order: " + _join_names(js, best_order) + " (greedy from every start table, cost " + commas(best_cost) + "" +
                         ((", the written order costs " + commas(wcost)) if wcost >= 0 else ", the written order meets a table no equality reaches yet") + ")")
    for nt in notes:
        plan.notes.append(nt)
    plan.notes.append("join start: " + js.names[plan.start] + " (" + commas(js.rows[plan.start]) + " of " + commas(js.base_rows[plan.start]) + " rows after its filter)")
    var acc_names = js.names[plan.start]
    var acc_rows = js.rows[plan.start]
    for i in range(len(plan.steps)):
        ref st = plan.steps[i]
        plan.notes.append("join " + String(i + 1) + ": " + acc_names + " (" + commas(acc_rows) + " rows) with " + js.names[st.leaf] + " (" + commas(js.rows[st.leaf]) + " of " +
                          commas(js.base_rows[st.leaf]) + " rows): hash table on " + (js.names[st.leaf] if st.build_new else "the rows joined so far") +
                          (", forced" if opts.build_side != BUILD_AUTO else ", the smaller side") + "; estimated " + commas(st.est_rows) + " rows, cost " + commas(st.cost))
        acc_names += " + " + js.names[st.leaf]
        acc_rows = st.est_rows
    return plan^


def commas(n: Int) -> String:
    var s = String(n if n >= 0 else -n)
    var out = String()
    var k = s.byte_length()
    for i in range(k):
        if i > 0 and (k - i) % 3 == 0:
            out += ","
        out += String(s[byte=i:i + 1])
    return ("-" if n < 0 else "") + out


struct RulePlanner(Planner, Copyable, Movable):
    def __init__(out self):
        pass

    def plan(self, q: BoundQuery, st: Stats, opts: PlannerOptions) raises -> PhysicalPlan:
        var plan = PhysicalPlan()
        plan.source_kind = st.kind
        for a in q.aggs:
            if a.hol:
                plan.holistic = True
        var what = "table" if st.kind == SRC_TABLE else "frame at edit " + String(st.version) + ": the edits as a mask over the shared rows plus " + \
                   commas(st.edits) + " changed or appended rows" if st.kind == SRC_FRAME else "Parquet scan, " + commas(st.row_groups) + " row groups"
        plan.notes.append("source: " + st.source + " (" + what + "), " + commas(st.rows) + " rows")
        if q.rows:
            plan.sink = S_ROWS
            return plan^
        if len(q.keys) == 0:
            plan.group = G_NONE
            plan.notes.append("group: none (one result row)")
            return plan^
        # per key: can it be addressed directly, and how many slots does it take?
        var dense_ok = True
        var why_not = String()
        var domain = 1
        var parts = String()
        for k in range(len(q.keys)):
            var e = q.rexprs[q.keys[k].expr].copy()
            var kind = e.kind
            var layout = KeyLayout(q.keys[k].expr, kind, e.scale, True, 0, 1, 1, -1, 1, False)
            if e.op == B_COL:
                var c = Int(e.value)
                layout.src = c
                ref cs = st.cols[c]
                layout.nullable = cs.nullable
                var radix = 0
                if kind == TEXT.value:
                    radix = cs.words
                    layout.lo = 0
                elif kind == BOOL.value:
                    radix = 2
                    layout.lo = 0
                elif cs.has_range:
                    layout.lo = cs.lo
                    var span = Int(cs.hi) - Int(cs.lo)
                    radix = span + 1 if span >= 0 and span < (1 << 40) else -1
                else:
                    radix = -1
                if kind == FLOAT.value:                         # a float has no range to address by: its min / max (a footer's too) are never used
                    radix = -1
                if radix < 0:
                    dense_ok = False
                    if why_not.byte_length() == 0:
                        why_not = ("`" + q.keys[k].name + "` is a FLOAT: floats are grouped by hashing, never by a dense range" if kind == FLOAT.value else
                                   "the range of `" + q.keys[k].name + "` is " + ("too wide" if cs.has_range else "unknown (" + ("statistics not believed" if st.distrusted else "the source has no statistics on it") + ")"))
                    radix = 1
                if layout.nullable:
                    radix += 1
                layout.radix = radix
            else:
                layout.nullable = expr_nullable(q, q.keys[k].expr, st)
                var iv = interval(q, q.keys[k].expr, st)
                var radix = -1
                if iv.ok:
                    layout.lo = iv.lo
                    var span = Int(iv.hi) - Int(iv.lo)
                    radix = span + 1 if span >= 0 and span < (1 << 40) else -1
                if kind == FLOAT.value:
                    radix = -1
                if radix < 0:
                    dense_ok = False
                    if why_not.byte_length() == 0:
                        why_not = ("`" + q.keys[k].name + "` is a FLOAT: floats are grouped by hashing, never by a dense range" if kind == FLOAT.value else
                                   "the range of `" + q.keys[k].name + "` is " + ("too wide" if iv.ok else "unknown: it is not a column or an expression the statistics bound"))
                    radix = 1
                if layout.nullable:
                    radix += 1
                layout.radix = radix
            layout.words = 1 + (1 if layout.nullable else 0)
            layout.guarded = st.kind == SRC_SCAN and kind != TEXT.value and kind != BOOL.value
            if dense_ok:
                if domain > (1 << 40) // max(layout.radix, 1):
                    dense_ok = False
                    why_not = "the domain passes 2^40 slots"
                else:
                    domain *= layout.radix
            parts += (" x " if k > 0 else "") + q.keys[k].name + " " + String(layout.radix) + (" (with a NULL slot)" if layout.nullable else "")
            plan.keys.append(layout^)
        var use_dense = False
        var reason = String()
        var est_slots = 1 + 2 * len(q.aggs)                         # the dense kernel holds a word per slot per group
        if dense_ok and domain > (1 << 27) // est_slots:
            dense_ok = False
            why_not = "the domain of " + commas(domain) + " slots times " + String(est_slots) + " accumulators passes the memory bound"
        if opts.group == GROUP_DENSE:
            if not dense_ok:
                raise Error("cannot force dense groups: " + why_not)
            if domain > opts.dense_limit:
                raise Error("cannot force dense groups: the domain of " + commas(domain) + " slots passes the limit of " + commas(opts.dense_limit))
            use_dense = True
            reason = "forced by the options"
        elif opts.group == GROUP_HASHED:
            reason = "forced by the options"
        elif dense_ok and domain <= opts.dense_limit and (domain <= SMALL_DOMAIN or domain <= SPARSE_FACTOR * max(st.rows, 1)):
            use_dense = True
            reason = "the domain of " + commas(domain) + " slots is " + ("small" if domain <= SMALL_DOMAIN else "no more than " + String(SPARSE_FACTOR) + "x the rows")
        elif not dense_ok:
            reason = why_not
        elif domain > opts.dense_limit:
            reason = "the domain of " + commas(domain) + " slots passes the dense limit of " + commas(opts.dense_limit)
        else:
            reason = "the domain of " + commas(domain) + " slots is sparse for " + commas(st.rows) + " rows"
        if use_dense:
            plan.group = G_DENSE
            plan.domain = domain
            var stride = 1
            var k = len(plan.keys) - 1
            while k >= 0:
                plan.keys[k].stride = stride
                stride *= plan.keys[k].radix
                k -= 1
            plan.notes.append("group: dense, " + String(len(plan.keys)) + (" key" if len(plan.keys) == 1 else " keys") + " " + parts +
                              " = " + commas(domain) + " slots: " + reason)
        else:
            plan.group = G_HASHED
            plan.domain = 0
            plan.notes.append("group: hashed on " + String(plan.key_words()) + " key " + ("word" if plan.key_words() == 1 else "words") + ": " + reason)
        return plan^


def plan_sinks(lp: LogicalPlan, q: BoundQuery, st: Stats, opts: PlannerOptions, mut plan: PhysicalPlan):
    """F-152: what the logical plan's shape says about the sink. A row query materializes its rows; a Sort under a Limit
    keeps a top-k in per-worker heaps when few rows are wanted out of many, otherwise sorts them all; a Limit with no Sort
    stops the scan early. (An aggregate query's order is decided where its groups are known: the finish.)"""
    if not q.rows:
        return
    plan.notes.append("sink: rows (the kernel's block hook writes the selected rows; text keeps its words)")
    var srt = lp.find(P_SORT)
    if srt >= 0:
        var want = lp.nodes[srt].limit
        if lp.top_k and want > 0 and want <= TOPK_MAX and st.rows > TOPK_FACTOR * want:
            plan.top_k = want
            plan.notes.append("order: top-" + commas(want) + " by per-worker heaps in the hook (" + commas(st.rows) + " rows, " + commas(want) + " wanted)")
        elif lp.top_k and want > 0:
            plan.notes.append("order: all rows sorted (" + ("top-" + commas(want) + " wanted but " + ("more than " + commas(TOPK_MAX) if want > TOPK_MAX else "the source has no more than " + String(TOPK_FACTOR) + "x that many rows") + ")"))
        else:
            plan.notes.append("order: all rows sorted")
    elif lp.early_stop:
        plan.early_stop = True
        ref lim = lp.nodes[lp.find(P_LIMIT)]
        plan.notes.append("limit: no ORDER BY, so the scan stops once it has " + commas(lim.limit + lim.offset) + " rows")
