"""Window functions (F-168): `apply_windows(table, groups)` adds one column per window function to a Table.

The SQL front end (`winsql.mojo`) and the builder (`Query.window`) both end here. The input is a Table of plain columns (the rows a query
produced); a `WinGroup` names the PARTITION BY columns, the ORDER BY keys and the functions that share them. For each group:

  1. the rows are put in (partition, order) order by the same stable LSD radix passes as ORDER BY (text under the engine's order, NULLS
     FIRST / LAST per key, FLOAT in DuckDB's order: NaN equal to itself and above +inf, -0.0 equal to 0.0). Ties keep the input order;
  2. partitions and peer runs (rows equal on every ORDER BY key) are found in one pass;
  3. every function runs over each partition, in tasks of whole partitions on the thread pool (`fan.run_tasks`): a task writes only the
     output slots of its own rows, so the answer is the same for 1 and N threads.

Functions: row_number, rank, dense_rank, percent_rank, cume_dist, ntile(k), lag / lead(x [, offset [, default]]), first_value / last_value /
nth_value(x, n), and sum, count, count(*), avg, min, max over a frame. Frames are monotone (both ends only move forward), so a sum, count or
average is a running total that adds the rows entering and subtracts the rows leaving, and min / max a monotone deque: O(rows). INT and
DECIMAL sums are exact (a 128-bit accumulator; a result outside int64 raises, like the engine's other sums); a FLOAT sum adds in partition
order when the frame starts at the partition's start and otherwise combines a fixed binary tree (no cancellation from subtracting).
Frames: ROWS with any offsets, RANGE with UNBOUNDED / CURRENT ROW bounds (peers included; the default frame with an ORDER BY is RANGE
UNBOUNDED PRECEDING to CURRENT ROW, the whole partition without one). The ranking functions, ntile, lag and lead ignore the frame.
"""
from std.memory import ArcPointer
from .table import Table, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, WordDict
from .frame import _radix_order_job, text_ranks
from .job import JobCtl
from .holistic import float_key
from .sqleval import float_bits, ld_div
from .fan import ParTask, run_tasks
from .sql import FM_NONE, FM_ROWS, FM_RANGE, FB_UNB_PREC, FB_PREC, FB_CUR, FB_FOL, FB_UNB_FOL
from .expr import Expr

comptime W_ROW_NUMBER = 0
comptime W_RANK = 1
comptime W_DENSE_RANK = 2
comptime W_PERCENT_RANK = 3
comptime W_CUME_DIST = 4
comptime W_NTILE = 5
comptime W_LAG = 6
comptime W_LEAD = 7
comptime W_FIRST = 8
comptime W_LAST = 9
comptime W_NTH = 10
comptime W_SUM = 11
comptime W_COUNT = 12
comptime W_COUNT_STAR = 13
comptime W_AVG = 14
comptime W_MIN = 15
comptime W_MAX = 16

comptime TASK_ROWS = 16384        # a task takes whole partitions until it holds about this many rows
comptime WINDOW_OVERFLOW = "a window sum passed the int64 range (about 9.2e18)"


struct WinOrder(Copyable, Movable):
    var col: Int
    var desc: Bool
    var nulls_first: Bool

    def __init__(out self, col: Int, desc: Bool, nulls_first: Bool):
        self.col = col
        self.desc = desc
        self.nulls_first = nulls_first


struct WinFn(Copyable, Movable):
    """One window function. `arg` is the input column (-1 for none); `n` the ntile bucket count, nth_value's n, or lag / lead's offset (lag
    positive looks back); a lag / lead default is `def_null` (the default is NULL), a stored value `def_val`, or a word `def_text`."""
    var kind: Int
    var arg: Int
    var n: Int
    var has_default: Bool
    var def_null: Bool
    var def_val: Int64
    var def_text: String
    var def_is_text: Bool
    var mode: Int
    var s_kind: Int
    var s_n: Int
    var e_kind: Int
    var e_n: Int
    var name: String

    def __init__(out self, kind: Int, name: String):
        self.kind = kind
        self.arg = -1
        self.n = 1
        self.has_default = False
        self.def_null = True
        self.def_val = 0
        self.def_text = String()
        self.def_is_text = False
        self.mode = FM_NONE
        self.s_kind = FB_UNB_PREC
        self.s_n = 0
        self.e_kind = FB_CUR
        self.e_n = 0
        self.name = name


struct WinGroup(Copyable, Movable):
    """Functions that share one sort: the same PARTITION BY columns and ORDER BY keys."""
    var part: List[Int]
    var order: List[WinOrder]
    var fns: List[WinFn]

    def __init__(out self):
        self.part = List[Int]()
        self.order = List[WinOrder]()
        self.fns = List[WinFn]()


def function_name(kind: Int) -> String:
    var names: List[String] = ["row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile", "lag", "lead", "first_value", "last_value",
                               "nth_value", "sum", "count", "count", "avg", "min", "max"]
    return names[kind]


def _kind_text(k: Int) -> String:
    return String(ColumnKind(k))


def output_type(f: WinFn, t: Table) raises -> Tuple[Int, Int]:
    """The (kind, scale) of the column `f` produces over `t`; raises for an argument the function cannot take."""
    var k = f.kind
    if k == W_ROW_NUMBER or k == W_RANK or k == W_DENSE_RANK or k == W_NTILE or k == W_COUNT or k == W_COUNT_STAR:
        return (INT.value, 0)
    if k == W_PERCENT_RANK or k == W_CUME_DIST:
        return (FLOAT.value, 0)
    var ak = t.cols[f.arg].kind.value
    var asc = t.cols[f.arg].scale
    if k == W_SUM:
        if ak == INT.value or ak == DECIMAL.value:
            return (ak, asc)
        if ak == FLOAT.value:
            return (FLOAT.value, 0)
        raise Error("sum(...) OVER is not supported for " + _kind_text(ak) + " values (INT, DECIMAL and FLOAT only)")
    if k == W_AVG:
        if ak == INT.value or ak == DECIMAL.value or ak == FLOAT.value:
            return (FLOAT.value, 0)
        raise Error("avg(...) OVER is not supported for " + _kind_text(ak) + " values (INT, DECIMAL and FLOAT only)")
    return (ak, asc)


struct _FnRun(Copyable, Movable):
    var kind: Int
    var n: Int
    var mode: Int
    var s_kind: Int
    var s_n: Int
    var e_kind: Int
    var e_n: Int
    var has_default: Bool
    var def_null: Bool
    var def_val: Int64
    var argv: Int                    # address of the argument column's values (0: none)
    var argvalid: Int                # address of its validity bitmap (0: no nulls)
    var is_float: Bool
    var scale: Int
    var keys: List[Int64]            # min / max: a comparable key per row
    var outv: Int
    var outn: Int

    def __init__(out self):
        self.kind = 0
        self.n = 0
        self.mode = 0
        self.s_kind = 0
        self.s_n = 0
        self.e_kind = 0
        self.e_n = 0
        self.has_default = False
        self.def_null = True
        self.def_val = 0
        self.argv = 0
        self.argvalid = 0
        self.is_float = False
        self.scale = 0
        self.keys = List[Int64]()
        self.outv = 0
        self.outn = 0


struct _Ctx(Movable):
    var n: Int
    var perm: List[Int]              # position -> input row
    var has_order: Bool
    var peer_lo: List[Int]           # per position: where its peer run starts (empty without ORDER BY: the partition start)
    var peer_hi: List[Int]           # ... and where it ends (exclusive)
    var pbounds: List[Int]           # the partitions' first positions, then n
    var task_at: List[Int]           # the partition each task starts at, then the partition count
    var fns: List[_FnRun]
    var job: Int                     # the job's words (0: none): a task stops taking partitions once it is cancelled
    var err: List[Int]               # set to 1 by a task whose sum left the int64 range

    def __init__(out self):
        self.n = 0
        self.perm = List[Int]()
        self.has_order = False
        self.peer_lo = List[Int]()
        self.peer_hi = List[Int]()
        self.pbounds = List[Int]()
        self.task_at = List[Int]()
        self.fns = List[_FnRun]()
        self.job = 0
        self.err = List[Int](length=1, fill=0)


struct _Out(Movable):
    var kind: Int
    var scale: Int
    var vals: List[Int64]
    var nul: List[UInt8]             # 1 where the result is NULL
    var dict: ArcPointer[WordDict]   # TEXT results: the words their codes belong to
    var name: String

    def __init__(out self, kind: Int, scale: Int, n: Int, name: String):
        self.kind = kind
        self.scale = scale
        self.vals = List[Int64](length=n, fill=0)
        self.nul = List[UInt8](length=n, fill=0)
        self.dict = ArcPointer(WordDict())
        self.name = name


# ── the builder's side: `Window` and `WindowFn` (Query.window, compute_windows in winquery.mojo) ─────────

comptime UNBOUNDED_PRECEDING = -(1 << 62)      # a frame bound of `Window.rows` / `Window.range`: every row from the partition's start
comptime UNBOUNDED_FOLLOWING = 1 << 62         # ... every row to the partition's end
comptime CURRENT_ROW = 0                       # ... the row itself (`range`: with its peers); a negative number is rows PRECEDING, a positive one FOLLOWING


struct Window(Copyable, Movable):
    """`OVER (PARTITION BY ... ORDER BY ... [frame])` as a builder:

        var w = Window().partition_by(col("region")).order_by(col("day")).rows(-2, CURRENT_ROW)    # ROWS BETWEEN 2 PRECEDING AND CURRENT ROW

    With no `rows` / `range` the frame is the SQL default: RANGE UNBOUNDED PRECEDING to CURRENT ROW (the rows up to the last peer), the whole partition without
    an `order_by`. NULLs sort last unless `nulls_first`, in either direction (DuckDB's default)."""
    var partition: List[Expr]
    var order: List[Expr]
    var order_desc: List[Bool]
    var order_nulls_first: List[Bool]
    var mode: Int
    var start: Int
    var end: Int

    def __init__(out self):
        self.partition = List[Expr]()
        self.order = List[Expr]()
        self.order_desc = List[Bool]()
        self.order_nulls_first = List[Bool]()
        self.mode = FM_NONE
        self.start = UNBOUNDED_PRECEDING
        self.end = CURRENT_ROW

    def partition_by(var self, *keys: Expr) -> Self:
        for k in keys:
            self.partition.append(k.copy())
        return self^

    def order_by(var self, key: Expr, descending: Bool = False, nulls_first: Bool = False) -> Self:
        self.order.append(key.copy())
        self.order_desc.append(descending)
        self.order_nulls_first.append(nulls_first)
        return self^

    def rows(var self, start: Int, end: Int) -> Self:
        """ROWS BETWEEN `start` AND `end`: a negative number is that many rows PRECEDING, 0 the current row, a positive number FOLLOWING; `UNBOUNDED_PRECEDING` /
        `UNBOUNDED_FOLLOWING` reach the partition's edges."""
        self.mode = FM_ROWS
        self.start = start
        self.end = end
        return self^

    def range(var self, start: Int, end: Int) -> Self:
        """RANGE BETWEEN `start` AND `end`, the bounds `UNBOUNDED_PRECEDING`, `CURRENT_ROW` (with its peers) and `UNBOUNDED_FOLLOWING` only: another offset raises
        when the windows are computed."""
        self.mode = FM_RANGE
        self.start = start
        self.end = end
        return self^


struct WindowFn(Copyable, Movable):
    """A window function with its `Window` (`row_number().over(w)`, `sum(col("v")).over(w)`) and the name of its result column (`alias`; `w0`, `w1`, ... when none)."""
    var kind: Int
    var arg: Expr
    var n: Int                       # ntile's buckets, nth_value's n, lag / lead's offset
    var has_default: Bool
    var default: Expr                # lag / lead: a number literal
    var win: Window
    var has_over: Bool
    var name: String

    def __init__(out self, kind: Int):
        self.kind = kind
        self.arg = Expr(0)
        self.n = 1
        self.has_default = False
        self.default = Expr(0)
        self.win = Window()
        self.has_over = False
        self.name = String()

    def over(var self, var w: Window) -> Self:
        self.win = w^
        self.has_over = True
        return self^

    def alias(var self, name: String) -> Self:
        self.name = name
        return self^


def row_number() -> WindowFn:
    return WindowFn(W_ROW_NUMBER)


def rank() -> WindowFn:
    """The place of a row's peer group in the partition's order, counting rows (1, 1, 3, ...)."""
    return WindowFn(W_RANK)


def dense_rank() -> WindowFn:
    return WindowFn(W_DENSE_RANK)


def percent_rank() -> WindowFn:
    return WindowFn(W_PERCENT_RANK)


def cume_dist() -> WindowFn:
    return WindowFn(W_CUME_DIST)


def ntile(buckets: Int) -> WindowFn:
    var f = WindowFn(W_NTILE)
    f.n = buckets
    return f^


def lag(e: Expr, offset: Int = 1) -> WindowFn:
    """The value `offset` rows before the current one in the partition's order (NULL when there is none)."""
    var f = WindowFn(W_LAG)
    f.arg = e.copy()
    f.n = offset
    return f^


def lag(e: Expr, offset: Int, default: Expr) -> WindowFn:
    """`lag` with a default (a number literal) for rows with no row that far back."""
    var f = lag(e, offset)
    f.has_default = True
    f.default = default.copy()
    return f^


def lead(e: Expr, offset: Int = 1) -> WindowFn:
    var f = WindowFn(W_LEAD)
    f.arg = e.copy()
    f.n = offset
    return f^


def lead(e: Expr, offset: Int, default: Expr) -> WindowFn:
    var f = lead(e, offset)
    f.has_default = True
    f.default = default.copy()
    return f^


def first_value(e: Expr) -> WindowFn:
    var f = WindowFn(W_FIRST)
    f.arg = e.copy()
    return f^


def last_value(e: Expr) -> WindowFn:
    var f = WindowFn(W_LAST)
    f.arg = e.copy()
    return f^


def nth_value(e: Expr, n: Int) -> WindowFn:
    var f = WindowFn(W_NTH)
    f.arg = e.copy()
    f.n = n
    return f^


# ── ordering ─────────────────────────────────────────────────────────────────────────────────────────

def _col_keys(t: Table, c: Int) -> List[Int64]:
    """Int64 keys whose signed order is the column's order (a NULL's key is 0; the caller orders NULLs apart)."""
    var n = t.rows()
    var keys = List[Int64](length=n, fill=0)
    ref col = t.cols[c]
    if col.kind == FLOAT:
        for i in range(n):
            keys[i] = 0 if col.is_null(i) else float_key(col.values[i])
    elif col.kind == TEXT:
        var ranks = text_ranks(col.dict[].words)
        for i in range(n):
            keys[i] = 0 if col.is_null(i) else ranks[Int(col.values[i])]
    else:
        for i in range(n):
            keys[i] = 0 if col.is_null(i) else col.values[i]
    return keys^


def _sort_pass(t: Table, c: Int, keys: List[Int64], desc: Bool, nulls_first: Bool, perm: List[Int], job: JobCtl) raises -> List[Int]:
    """One stable pass over `perm` by column `c` (values, then the NULL flag so NULLS FIRST / LAST holds in either direction)."""
    var n = len(perm)
    var aligned = List[Int64](length=n, fill=0)
    for i in range(n):
        aligned[i] = keys[perm[i]]
    var out = _radix_order_job(aligned, perm, desc, job)
    if t.cols[c].nulls > 0:
        var flags = List[Int64](length=n, fill=0)
        for i in range(n):
            var isn = t.cols[c].is_null(out[i])
            flags[i] = Int64((0 if isn else 1) if nulls_first else (1 if isn else 0))
        out = _radix_order_job(flags, out, False, job)
    return out^


def _mark_changes(t: Table, c: Int, keys: List[Int64], perm: List[Int], mut flag: List[UInt8], v: UInt8):
    """Marks in `flag` (to at least `v`) every position whose row differs from the one before it on column `c` (NULLs equal each other)."""
    var n = len(perm)
    var kp = keys.unsafe_ptr()
    var pp = perm.unsafe_ptr()
    var fp = flag.unsafe_ptr()
    if t.cols[c].nulls == 0:
        for p in range(1, n):
            if kp[pp[p - 1]] != kp[pp[p]] and fp[p] < v:
                fp[p] = v
    else:
        var vp = t.cols[c].validity.unsafe_ptr()
        for p in range(1, n):
            var a = pp[p - 1]
            var b = pp[p]
            var na = (vp[a >> 3] >> UInt8(a & 7)) & 1 == 0
            var nb = (vp[b >> 3] >> UInt8(b & 7)) & 1 == 0
            if (na != nb or (not na and kp[a] != kp[b])) and fp[p] < v:
                fp[p] = v


# ── one group ────────────────────────────────────────────────────────────────────────────────────────

def _prepare(t: Table, g: WinGroup, n: Int, threads: Int, job: JobCtl, mut outs: List[_Out]) raises:
    job.check()
    var ctx = _Ctx()
    ctx.job = job.words
    ctx.n = n
    ctx.has_order = len(g.order) > 0
    # 1. the order
    var pkeys = List[List[Int64]]()
    for c in g.part:
        pkeys.append(_col_keys(t, c))
    var okeys = List[List[Int64]]()
    for o in g.order:
        okeys.append(_col_keys(t, o.col))
    var perm = List[Int](length=n, fill=0)
    for i in range(n):
        perm[i] = i
    var k = len(g.order) - 1
    while k >= 0:
        perm = _sort_pass(t, g.order[k].col, okeys[k], g.order[k].desc, g.order[k].nulls_first, perm, job)
        k -= 1
    k = len(g.part) - 1
    while k >= 0:
        perm = _sort_pass(t, g.part[k], pkeys[k], False, False, perm, job)
        k -= 1
    # 2. partitions and peer runs: flag 2 where a partition starts, 1 where a run of peers does
    var flag = List[UInt8](length=n, fill=0)
    for j in range(len(g.part)):
        _mark_changes(t, g.part[j], pkeys[j], perm, flag, 2)
    for j in range(len(g.order)):
        _mark_changes(t, g.order[j].col, okeys[j], perm, flag, 1)
    var pb = List[Int]()
    if ctx.has_order:
        ctx.peer_lo = List[Int](length=n, fill=0)
        ctx.peer_hi = List[Int](length=n, fill=0)
    for p in range(n):
        if p == 0 or flag[p] == 2:
            pb.append(p)
        if ctx.has_order:
            ctx.peer_lo[p] = p if (p == 0 or flag[p] != 0) else ctx.peer_lo[p - 1]
    pb.append(n)
    if ctx.has_order:
        var p = n - 1
        while p >= 0:
            ctx.peer_hi[p] = n if p == n - 1 else (p + 1 if ctx.peer_lo[p + 1] == p + 1 else ctx.peer_hi[p + 1])
            p -= 1
    ctx.perm = perm^
    ctx.pbounds = pb^
    # 3. tasks of whole partitions
    var parts = len(ctx.pbounds) - 1
    var start = 0
    for i in range(parts):
        if ctx.pbounds[i] - ctx.pbounds[start] >= TASK_ROWS:
            ctx.task_at.append(start)
            start = i
    if parts > 0:
        ctx.task_at.append(start)
    ctx.task_at.append(parts)
    # 4. the functions
    for f in g.fns:
        var ot = output_type(f, t)
        var out = _Out(ot[0], ot[1], n, f.name)
        var fr = _FnRun()
        fr.kind = f.kind
        fr.n = f.n
        fr.mode = f.mode
        fr.s_kind = f.s_kind
        fr.s_n = f.s_n
        fr.e_kind = f.e_kind
        fr.e_n = f.e_n
        fr.has_default = f.has_default
        fr.def_null = f.def_null
        fr.def_val = f.def_val
        if f.arg >= 0:
            ref ac = t.cols[f.arg]
            fr.argv = Int(ac.values.unsafe_ptr())
            fr.argvalid = ac.validity_address()
            fr.is_float = ac.kind == FLOAT
            fr.scale = ac.scale
            if ac.kind == TEXT:
                out.dict = ac.dict
            if f.kind == W_MIN or f.kind == W_MAX:
                fr.keys = _col_keys(t, f.arg)
        if f.kind == W_LAG or f.kind == W_LEAD:
            if f.def_is_text:                                            # a default word: the result's dictionary is a copy with the word added
                var wd = WordDict()
                wd.words = out.dict[].words.copy()
                wd.codes = out.dict[].codes.copy()
                var code = wd.codes.get(f.def_text, -1)
                if code < 0:
                    code = len(wd.words)
                    wd.words.append(f.def_text)
                    wd.codes[f.def_text] = code
                out.dict = ArcPointer(wd^)
                fr.def_val = Int64(code)
        fr.outv = Int(out.vals.unsafe_ptr())
        fr.outn = Int(out.nul.unsafe_ptr())
        ctx.fns.append(fr^)
        outs.append(out^)
    # 5. run
    var ntasks = len(ctx.task_at) - 1
    var task = _WinTask(Int(UnsafePointer(to=ctx)))
    run_tasks(task, ntasks, threads)
    job.check()                                                      # (a cancelled run left some partitions undone: no result)
    var failed = ctx.err[0]
    if failed != 0:
        raise Error(WINDOW_OVERFLOW)


@fieldwise_init
struct _WinTask(ParTask):
    var ctx: Int

    def run(self, i: Int):
        var cp = UnsafePointer[_Ctx, MutAnyOrigin](unsafe_from_address=self.ctx)
        var c0 = cp[].task_at[i]
        var c1 = cp[].task_at[i + 1]
        var job = JobCtl(cp[].job)
        for pi in range(c0, c1):
            if job.cancelled():
                return
            var ps = cp[].pbounds[pi]
            var pe = cp[].pbounds[pi + 1]
            for f in range(len(cp[].fns)):
                _eval(cp[], cp[].fns[f], ps, pe)


# ── evaluating one function over one partition ───────────────────────────────────────────────────────

@always_inline
def _isnull(valid: Int, row: Int) -> Bool:
    if valid == 0:
        return False
    var b = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=valid)[row >> 3]
    return (b >> UInt8(row & 7)) & 1 == 0


@always_inline
def _put(fr: _FnRun, row: Int, v: Int64):
    UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=fr.outv)[row] = v


@always_inline
def _put_null(fr: _FnRun, row: Int):
    UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=fr.outn)[row] = 1


def _frame(c: _Ctx, fr: _FnRun, p: Int, ps: Int, pe: Int) -> Tuple[Int, Int]:
    """The frame of position `p` as [lo, hi) positions inside its partition [ps, pe); empty when hi == lo."""
    var l = ps
    var h = pe - 1
    if fr.mode == FM_ROWS:
        if fr.s_kind == FB_PREC:
            l = p - fr.s_n
        elif fr.s_kind == FB_CUR:
            l = p
        elif fr.s_kind == FB_FOL:
            l = p + fr.s_n
        if fr.e_kind == FB_PREC:
            h = p - fr.e_n
        elif fr.e_kind == FB_CUR:
            h = p
        elif fr.e_kind == FB_FOL:
            h = p + fr.e_n
    else:
        if fr.s_kind == FB_CUR:
            l = c.peer_lo[p] if c.has_order else ps
        if fr.e_kind == FB_CUR:
            h = (c.peer_hi[p] if c.has_order else pe) - 1
    l = min(max(l, ps), pe)                    # (a start past the partition's end is an empty frame at its end, not a row of the next partition)
    h = min(h, pe - 1)
    return (l, max(h + 1, l))


def _eval(c: _Ctx, fr: _FnRun, ps: Int, pe: Int):
    var k = fr.kind
    var cnt = pe - ps
    var perm = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(c.perm.unsafe_ptr()))
    if k == W_ROW_NUMBER:
        for p in range(ps, pe):
            _put(fr, perm[p], Int64(p - ps + 1))
    elif k == W_RANK:
        for p in range(ps, pe):
            _put(fr, perm[p], Int64((c.peer_lo[p] if c.has_order else ps) - ps + 1))
    elif k == W_DENSE_RANK:
        var d = 0
        for p in range(ps, pe):
            if p == ps or (c.has_order and c.peer_lo[p] == p):
                d += 1
            _put(fr, perm[p], Int64(d))
    elif k == W_PERCENT_RANK:
        for p in range(ps, pe):
            var rank = (c.peer_lo[p] if c.has_order else ps) - ps + 1
            _put(fr, perm[p], float_bits(0.0 if cnt == 1 else Float64(rank - 1) / Float64(cnt - 1)))
    elif k == W_CUME_DIST:
        for p in range(ps, pe):
            var upto = (c.peer_hi[p] if c.has_order else pe) - ps
            _put(fr, perm[p], float_bits(Float64(upto) / Float64(cnt)))
    elif k == W_NTILE:
        var small = cnt // fr.n
        var large_n = cnt % fr.n
        var boundary = large_n * (small + 1)
        for p in range(ps, pe):
            var r = p - ps
            var b = r + 1
            if small > 0:
                b = r // (small + 1) + 1 if r < boundary else (r - boundary) // small + large_n + 1
            _put(fr, perm[p], Int64(b))
    elif k == W_LAG or k == W_LEAD:
        var argv = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=fr.argv)
        for p in range(ps, pe):
            var q = p + fr.n
            var row = perm[p]
            if q < ps or q >= pe:
                if fr.has_default and not fr.def_null:
                    _put(fr, row, fr.def_val)
                else:
                    _put_null(fr, row)
            else:
                var src = perm[q]
                if _isnull(fr.argvalid, src):
                    _put_null(fr, row)
                else:
                    _put(fr, row, argv[src])
    elif k == W_FIRST or k == W_LAST or k == W_NTH:
        var argv = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=fr.argv)
        for p in range(ps, pe):
            var fh = _frame(c, fr, p, ps, pe)
            var at = fh[0] if k == W_FIRST else (fh[1] - 1 if k == W_LAST else fh[0] + fr.n - 1)
            var row = perm[p]
            if fh[1] <= fh[0] or at >= fh[1] or at < fh[0]:
                _put_null(fr, row)
            else:
                var src = perm[at]
                if _isnull(fr.argvalid, src):
                    _put_null(fr, row)
                else:
                    _put(fr, row, argv[src])
    else:
        _eval_agg(c, fr, ps, pe)


def _eval_agg(c: _Ctx, fr: _FnRun, ps: Int, pe: Int):
    var k = fr.kind
    var cnt_all = pe - ps
    var perm = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(c.perm.unsafe_ptr()))
    var argv = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=fr.argv)
    var is_minmax = k == W_MIN or k == W_MAX
    var is_float = fr.is_float
    var use_tree = is_float and (k == W_SUM or k == W_AVG) and fr.s_kind != FB_UNB_PREC
    var dq = List[Int](length=cnt_all if is_minmax else 0, fill=0)
    var head = 0
    var tail = 0
    var tree = List[Float64](length=2 * cnt_all if use_tree else 0, fill=0.0)
    var pc = List[Int](length=cnt_all + 1 if use_tree else 0, fill=0)
    if use_tree:
        for i in range(cnt_all):
            var row = perm[ps + i]
            var nn = _isnull(fr.argvalid, row)
            tree[cnt_all + i] = 0.0 if nn else Float64(from_bits=UInt64(argv[row]))
            pc[i + 1] = pc[i] + (0 if nn else 1)
        var i = cnt_all - 1
        while i > 0:
            tree[i] = tree[2 * i] + tree[2 * i + 1]
            i -= 1
    var cur_lo = ps
    var cur_hi = ps
    var acc = Int128(0)
    var facc = 0.0
    var cnt = 0
    for p in range(ps, pe):
        var fh = _frame(c, fr, p, ps, pe)
        var lo = fh[0]
        var hi = fh[1]
        var row = perm[p]
        if use_tree:
            var l = lo - ps + cnt_all
            var r = hi - ps + cnt_all
            var left = 0.0
            var right = 0.0
            while l < r:
                if l & 1 == 1:
                    left += tree[l]
                    l += 1
                if r & 1 == 1:
                    r -= 1
                    right += tree[r]
                l >>= 1
                r >>= 1
            facc = left + right
            cnt = pc[hi - ps] - pc[lo - ps]
        else:
            while cur_hi < hi:
                var src = perm[cur_hi]
                if k == W_COUNT_STAR:
                    cnt += 1
                elif not _isnull(fr.argvalid, src):
                    cnt += 1
                    if k == W_SUM or k == W_AVG:
                        if is_float:
                            facc += Float64(from_bits=UInt64(argv[src]))
                        else:
                            acc += Int128(argv[src])
                    elif is_minmax:
                        var key = fr.keys[src]
                        while tail > head:
                            var back = fr.keys[perm[dq[tail - 1]]]
                            if (k == W_MIN and back >= key) or (k == W_MAX and back <= key):
                                tail -= 1
                            else:
                                break
                        dq[tail] = cur_hi
                        tail += 1
                cur_hi += 1
            while cur_lo < lo:
                var src = perm[cur_lo]
                if k == W_COUNT_STAR:
                    cnt -= 1
                elif not _isnull(fr.argvalid, src):
                    cnt -= 1
                    if k == W_SUM or k == W_AVG:
                        if is_float:
                            facc -= Float64(from_bits=UInt64(argv[src]))
                        else:
                            acc -= Int128(argv[src])
                    elif is_minmax:
                        if head < tail and dq[head] == cur_lo:
                            head += 1
                cur_lo += 1
        if k == W_COUNT or k == W_COUNT_STAR:
            _put(fr, row, Int64(cnt))
        elif cnt == 0 or (is_minmax and head >= tail):
            _put_null(fr, row)
        elif k == W_SUM:
            if is_float:
                _put(fr, row, float_bits(facc))
            elif acc > Int128(Int64.MAX) or acc < Int128(Int64.MIN):
                UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(c.err.unsafe_ptr()))[0] = 1
            else:
                _put(fr, row, Int64(acc))
        elif k == W_AVG:
            if is_float:
                _put(fr, row, float_bits(facc / Float64(cnt)))
            elif acc > Int128(Int64.MAX) or acc < Int128(Int64.MIN):
                UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(c.err.unsafe_ptr()))[0] = 1
            else:
                var den = UInt128(cnt)
                for _ in range(fr.scale):
                    den *= UInt128(10)
                _put(fr, row, float_bits(ld_div(Int64(acc), den)))
        else:                                                              # min / max: the deque's front
            var v = argv[perm[dq[head]]]
            if is_float and Float64(from_bits=UInt64(v)) != Float64(from_bits=UInt64(v)):
                v = Int64(0x7FF8000000000000)                              # (every NaN is one value; the engine's other aggregates answer this one)
            _put(fr, row, v)


# ── the entry ────────────────────────────────────────────────────────────────────────────────────────

def apply_windows(mut t: Table, groups: List[WinGroup], threads: Int = 0, job: JobCtl = JobCtl()) raises:
    """Adds one column per function of `groups` to `t` (named `WinFn.name`, in order), each aligned with the rows of `t`."""
    var n = t.rows()
    var outs = List[_Out]()
    for g in groups:
        _prepare(t, g, n, threads, job, outs)
    for i in range(len(outs)):
        var ci = t.add_column(outs[i].name, ColumnKind(outs[i].kind), outs[i].scale)
        var nulls = 0
        for j in range(n):
            if outs[i].nul[j] != 0:
                nulls += 1
        swap(t.cols[ci].values, outs[i].vals)
        if nulls > 0:
            var bits = List[UInt8](length=(n + 7) // 8, fill=0xFF)
            for j in range(n):
                if outs[i].nul[j] != 0:
                    bits[j // 8] &= ~(UInt8(1) << UInt8(j % 8))
            t.cols[ci].validity = bits^
            t.cols[ci].nulls = nulls
        if outs[i].kind == TEXT.value:
            t.cols[ci].dict = outs[i].dict
