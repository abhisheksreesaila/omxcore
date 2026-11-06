"""Range aggregates and block reads over a row order (F-176; docs/omxcore-mojo-api.md "Ranges and blocks").

A grid shows a window of a sorted view and a status bar totals a selection of it. Both are "positions a..b of an order": the order is a
list of row numbers (`Frame.permutation`, the row numbers of a `sql ... ORDER BY`, any row list) or, with no list, the rows themselves.

    var s = range_aggregate(frame, order, [cost, region, day], 1_000, 2_000_000)   # positions a <= p < b of `order`
    s.columns[0].sum, s.columns[0].mean_value, s.columns[1].min_word(frame) ...
    var values = List[Int64](length=3 * n, fill=0); var nulls = List[UInt8](length=3 * n, fill=0)
    read_block(frame, order, [cost, region, day], a, b, values, nulls)               # the window, column-major

The cells are gathered a column at a time into dense buffers and reduced by tight loops: no per-cell call. An in-memory source (a table, a
frame over one, with its edits, appended rows and merges) runs on every core in fixed morsels of `MORSEL` positions; a Frame over a
Parquet file gathers on the calling thread through the frame's cache of decoded column chunks (a chunk is decoded once per call however many
rows of it the window holds, and stays in the cache for the next one).

Exactness: INT / DECIMAL sums are exact int64 (kept as a high and a low half per morsel, so only a total outside int64 raises); a FLOAT
sum is float64 added in position order inside each morsel and the morsel totals added in morsel order, so the bits do not depend on the
number of threads; MIN / MAX of a FLOAT put NaN above +inf (MAX is NaN when the range holds one) and -0.0 equal to 0.0 (the first in order
wins); MIN / MAX of TEXT use the case-insensitive order of `Frame.permutation` (bytes break ties); nulls are skipped as SQL skips them.
"""
from std.memory import Pointer
from .table import SharedTable, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .frame import Frame, _text_less
from .kernels._fpenv import core_count, par_n
from .kernels.pipeline_cpu import LP, BP
from .sqleval import avg_div

comptime MORSEL = 16384            # positions per morsel: the unit of work, and the unit of the (reproducible) float sum
comptime NP = Pointer[Int, MutUntrackedOrigin]
comptime _REC = 6                  # Int64s a morsel leaves per column: count, sum (hi | float bits), lo, min, max, nan count
comptime _NAN_BITS = Int64(0x7FF8000000000000)
comptime _INF_BITS = Int64(0x7FF0000000000000)


# ── results ─────────────────────────────────────────────────────────────────────────────────────────

struct RangeAgg(Copyable, Movable):
    """The aggregates of one column over a range. `count` is the non-null cells; with none, there is no sum, mean, min or max
    (`has_value()` is False, as SQL gives NULL). INT, DECIMAL and FLOAT have a sum and a mean (`has_sum`); every kind has a min and a max.

    Values are the stored ones: INT the integer, DECIMAL the value times 10^scale (the sum is exact at the same scale), DATE days, BOOL 0 / 1,
    TEXT a code (`min_word` / `max_word` give the word), FLOAT a float64's bits (`sum_float`, `min_float`, `max_float`)."""
    var kind: ColumnKind
    var scale: Int
    var count: Int
    var sum: Int64
    var min: Int64
    var max: Int64
    var mean: Float64
    var has_sum: Bool

    def __init__(out self, kind: ColumnKind, scale: Int):
        self.kind = kind
        self.scale = scale
        self.count = 0
        self.sum = 0
        self.min = 0
        self.max = 0
        self.mean = 0.0
        self.has_sum = False

    def has_value(self) -> Bool:
        return self.count > 0

    def sum_float(self) -> Float64:
        return Float64(from_bits=UInt64(self.sum))

    def min_float(self) -> Float64:
        return Float64(from_bits=UInt64(self.min))

    def max_float(self) -> Float64:
        return Float64(from_bits=UInt64(self.max))


struct RangeResult(Movable):
    """One `RangeAgg` per requested column, in the order asked, and the number of positions in the range (`rows`, nulls included)."""
    var rows: Int
    var columns: List[RangeAgg]

    def __init__(out self, rows: Int):
        self.rows = rows
        self.columns = List[RangeAgg]()


# ── what a call needs of its source ─────────────────────────────────────────────────────────────────

struct _Prep(Movable):
    """A call's view of its source as plain arrays (the workers read it through `desc`). For a Frame: `touched` marks the base rows the
    session changed, `crow` lists every changed or appended row (ascending), and `pvals` / `pnull` / `palive` hold their cells as the
    session sees them (edits applied, merges applied, tombstones marked), so the gather reads the shared columns directly except for
    those few rows."""
    var cols: List[Int]
    var kinds: List[Int]
    var scales: List[Int]
    var nbase: Int
    var total: Int
    var meta: List[Int]
    var touched: List[UInt8]
    var crow: List[Int]
    var palive: List[UInt8]
    var pvals: List[Int64]
    var pnull: List[UInt8]
    var afrom: List[Int64]
    var ato: List[Int64]
    var seen: List[List[UInt8]]
    var words: List[Int]

    def __init__(out self):
        self.cols = List[Int]()
        self.kinds = List[Int]()
        self.scales = List[Int]()
        self.nbase = 0
        self.total = 0
        self.meta = List[Int]()
        self.touched = List[UInt8]()
        self.crow = List[Int]()
        self.palive = List[UInt8]()
        self.pvals = List[Int64]()
        self.pnull = List[UInt8]()
        self.afrom = List[Int64]()
        self.ato = List[Int64]()
        self.seen = List[List[UInt8]]()
        self.words = List[Int]()

    def desc(self) -> List[Int]:
        """The addresses the workers read: [nbase, total, columns, touched, crow, changed rows, palive, pvals, pnull, meta, afrom, ato]."""
        var d = List[Int]()
        d.append(self.nbase)
        d.append(self.total)
        d.append(len(self.cols))
        d.append(Int(self.touched.unsafe_ptr()) if len(self.touched) > 0 else 0)
        d.append(Int(self.crow.unsafe_ptr()))
        d.append(len(self.crow))
        d.append(Int(self.palive.unsafe_ptr()))
        d.append(Int(self.pvals.unsafe_ptr()))
        d.append(Int(self.pnull.unsafe_ptr()))
        d.append(Int(self.meta.unsafe_ptr()))
        d.append(Int(self.afrom.unsafe_ptr()))
        d.append(Int(self.ato.unsafe_ptr()))
        return d^


def _p10(k: Int) -> Int64:
    var v: Int64 = 1
    for _ in range(k):
        v *= 10
    return v


def _words_noun(n: Int) -> String:
    return String(n) + (" row" if n == 1 else " rows")


def _check_columns(cols: List[Int], n: Int, what: String) raises:
    for c in cols:
        if c < 0 or c >= n:
            raise Error("column " + String(c) + " does not exist (" + what + " has " + String(n) + ")")


def _check_range(a: Int, b: Int, n: Int, order: Bool) raises:
    var what = String("the order has ") if order else String("there are ")
    if a < 0:
        raise Error("the range " + String(a) + ".." + String(b) + " starts before the first position")
    if b < a:
        raise Error("the range " + String(a) + ".." + String(b) + " ends before it starts")
    if b > n:
        raise Error("the range " + String(a) + ".." + String(b) + " goes past the end: " + what + _words_noun(n))


def _prep_table(t: SharedTable, cols: List[Int], need_seen: Bool) raises -> _Prep:
    _check_columns(cols, t.columns(), "the table")
    var p = _Prep()
    p.cols = cols.copy()
    p.nbase = t.rows()
    p.total = t.rows()
    for k in range(len(cols)):
        var c = cols[k]
        p.kinds.append(t.kind(c).value)
        p.scales.append(t.scale(c))
        var w = len(t.dictionary(c)) if t.kind(c) == TEXT and need_seen else 0
        p.words.append(w)
        p.seen.append(List[UInt8](length=max(w, 1), fill=0))
        p.meta.append(Int(t.values(c).unsafe_ptr()))
        p.meta.append(t._table[].cols[c].validity_address())
        p.meta.append(t.kind(c).value)
        p.meta.append(t.scale(c))
        p.meta.append(0)
        p.meta.append(0)
        p.meta.append(Int(p.seen[k].unsafe_ptr()))
        p.meta.append(w)
    return p^


def _touched_rows(f: Frame, cols: List[Int], order: List[Int], identity: Bool, a: Int, b: Int) -> List[Int]:
    """The rows among positions a..b whose cells the session changed (edited in one of `cols`, appended, or removed), ascending and once each.
    A window no longer than 4x the frame's edits probes each of its rows (`len(cols)` lookups a row), so a grid's read costs the same
    with 3 edits or 300,000; a longer range takes every changed row of the frame, which costs less than reading the range does."""
    var out = List[Int]()
    var edits = len(f.cells) + len(f.removed) + len(f.added)
    if edits == 0:
        return out^
    if b - a > 4 * edits:
        return f.changed_rows()
    var total = f.rows()
    for p in range(a, b):
        var r = p if identity else order[p]
        if r < 0 or r >= total:
            continue
        var hit = r >= f.nbase or r in f.removed
        if not hit:
            for c in cols:
                if ((r << 20) | c) in f.cells:
                    hit = True
                    break
        if hit:
            out.append(r)
    sort(out)
    var uniq = List[Int]()
    for r in out:
        if len(uniq) == 0 or uniq[len(uniq) - 1] != r:
            uniq.append(r)
    return uniq^


def _prep_frame(f: Frame, cols: List[Int], order: List[Int], identity: Bool, a: Int, b: Int, need_seen: Bool) raises -> _Prep:
    for c in cols:
        if c >= f.stored() and c < f.columns():
            raise Error("column `" + f.name(c) + "` is a formula: ranges read the stored columns")
    _check_columns(cols, f.stored(), "the frame")
    var p = _Prep()
    p.cols = cols.copy()
    p.nbase = f.nbase
    p.total = f.rows()
    var ncols = len(cols)
    var changed = _touched_rows(f, cols, order, identity, a, b)
    var nrow = len(changed)
    if nrow > 0:
        if nrow > 64:                                        # many changed rows: a bitmap rejects the untouched ones before the binary search
            p.touched = List[UInt8](length=(f.nbase + 7) // 8 + 1, fill=0)
            for row in changed:
                if row < f.nbase:
                    p.touched[row >> 3] |= UInt8(1) << UInt8(row & 7)
        for row in changed:
            p.crow.append(row)
            p.palive.append(UInt8(1) if f.alive(row) else UInt8(0))
        p.pvals = List[Int64](length=ncols * nrow, fill=0)
        p.pnull = List[UInt8](length=ncols * nrow, fill=0)
    for k in range(ncols):
        var c = cols[k]
        var kind = f.base.kind(c)
        p.kinds.append(kind.value)
        p.scales.append(f.base.scale(c))
        var w = f.word_count(c) if kind == TEXT and need_seen else 0
        p.words.append(w)
        p.seen.append(List[UInt8](length=max(w, 1), fill=0))
        for j in range(nrow):
            var cell = f._cell(c, changed[j])
            if cell[1]:
                p.pnull[k * nrow + j] = 1
            else:
                p.pvals[k * nrow + j] = f.alias(c, cell[0])
        # merges: (from, to) pairs of this column, sorted by `from`, for the workers' binary search
        var off = len(p.afrom)
        if kind == TEXT and len(f.aliases) > 0:
            var pairs = List[Tuple[Int64, Int64]]()
            for item in f.aliases.items():
                if (item.key >> 32) == c:
                    pairs.append((Int64(item.key & 0xFFFFFFFF), item.value))
            sort(pairs)
            for pr in pairs:
                p.afrom.append(pr[0])
                p.ato.append(pr[1])
        p.meta.append(0 if f.scan else Int(f.base.values(c).unsafe_ptr()))
        p.meta.append(0 if f.scan else f.base._table[].cols[c].validity_address())
        p.meta.append(kind.value)
        p.meta.append(f.base.scale(c))
        p.meta.append(len(p.afrom) - off)
        p.meta.append(off)
        p.meta.append(Int(p.seen[k].unsafe_ptr()))
        p.meta.append(w)
    return p^


# ── the gather (workers: plain addresses in, no allocation) ─────────────────────────────────────────

@always_inline
def _lower_bound(rows: NP, n: Int, r: Int) -> Int:
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = (lo + hi) >> 1
        if rows[mid] < r:
            lo = mid + 1
        else:
            hi = mid
    return lo


@always_inline
def _alias_of(afrom: LP, ato: LP, off: Int, n: Int, v: Int64) -> Int64:
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = (lo + hi) >> 1
        if afrom[off + mid] < v:
            lo = mid + 1
        else:
            hi = mid
    if lo < n and afrom[off + lo] == v:
        return ato[off + lo]
    return v


def _gather_mem(d: Int, k: Int, order: Int, lo: Int, hi: Int, vout: LP, nout: BP) -> Int:
    """Column k (of the call's list) at positions lo..hi of the order (`order` is the address of the row list, 0 for the rows themselves)
    into vout / nout [0, hi - lo) (null: the flag is 1 and the value 0). Returns -1, or the first position whose row does not exist or was removed."""
    var dp = NP(unsafe_from_address=d)
    var nbase = dp[0]
    var total = dp[1]
    var tbits = dp[3]
    var touched = BP(unsafe_from_address=tbits)
    var ncrow = dp[5]
    var crow = NP(unsafe_from_address=dp[4])
    var palive = BP(unsafe_from_address=dp[6])
    var pvals = LP(unsafe_from_address=dp[7]) + k * ncrow
    var pnull = BP(unsafe_from_address=dp[8]) + k * ncrow
    var m = NP(unsafe_from_address=dp[9]) + k * 8
    var vals = LP(unsafe_from_address=m[0])
    var vaddr = m[1]
    var valid = BP(unsafe_from_address=vaddr)
    var na = m[4]
    var afrom = LP(unsafe_from_address=dp[10])
    var ato = LP(unsafe_from_address=dp[11])
    var op = NP(unsafe_from_address=order)
    var patched = ncrow > 0
    for p in range(lo, hi):
        var r = op[p] if order != 0 else p
        var i = p - lo
        if r < 0 or r >= total:
            return p
        var j = 0
        var hit = False
        if patched and (r >= nbase or tbits == 0 or (touched[r >> 3] >> UInt8(r & 7)) & 1 != 0):
            j = _lower_bound(crow, ncrow, r)
            hit = j < ncrow and crow[j] == r
        if not hit and r >= nbase:
            return p
        if not hit:
            if vaddr != 0 and (valid[r >> 3] >> UInt8(r & 7)) & 1 == 0:
                vout[i] = 0
                nout[i] = 1
            else:
                var v = vals[r]
                if na > 0:
                    v = _alias_of(afrom, ato, m[5], na, v)
                vout[i] = v
                nout[i] = 0
        else:
            if palive[j] == 0:
                return p
            vout[i] = pvals[j]
            nout[i] = pnull[j]
    return -1


def _gather_scan(f: Frame, cols: List[Int], d: Int, order: Int, lo: Int, hi: Int, vout: LP, nout: BP) raises -> Int:
    """Every column of the call at positions lo..hi of the order of a Frame over a Parquet file, column-major into vout / nout (column k at
    k * (hi - lo)). The base rows are bucketed by row group so that each (group, column) chunk is fetched once; the fetch goes through the
    frame's cache (decoded on a miss, kept up to its limit). Returns -1 or the first bad position (as `_gather_mem`)."""
    var dp = NP(unsafe_from_address=d)
    var nbase = dp[0]
    var total = dp[1]
    var ncols = dp[2]
    var tbits = dp[3]
    var touched = BP(unsafe_from_address=tbits)
    var ncrow = dp[5]
    var crow = NP(unsafe_from_address=dp[4])
    var palive = BP(unsafe_from_address=dp[6])
    var pvals = LP(unsafe_from_address=dp[7])
    var pnull = BP(unsafe_from_address=dp[8])
    var meta = NP(unsafe_from_address=dp[9])
    var afrom = LP(unsafe_from_address=dp[10])
    var ato = LP(unsafe_from_address=dp[11])
    var op = NP(unsafe_from_address=order)
    var n = hi - lo
    var groups = len(f.starts) - 1
    var gs = List[Int](length=max(n, 1), fill=-1)
    var counts = List[Int](length=groups + 1, fill=0)
    for p in range(lo, hi):
        var r = op[p] if order != 0 else p
        var i = p - lo
        if r < 0 or r >= total:
            return p
        var j = 0
        var hit = False
        if ncrow > 0 and (r >= nbase or tbits == 0 or (touched[r >> 3] >> UInt8(r & 7)) & 1 != 0):
            j = _lower_bound(crow, ncrow, r)
            hit = j < ncrow and crow[j] == r
        if not hit and r >= nbase:
            return p
        if not hit:
            var g = f._group_of(r)
            gs[i] = g
            counts[g + 1] += 1
        else:
            if palive[j] == 0:
                return p
            for k in range(ncols):
                vout[k * n + i] = pvals[k * ncrow + j]
                nout[k * n + i] = pnull[k * ncrow + j]
    for g in range(groups):
        counts[g + 1] += counts[g]
    var at = counts.copy()
    var idx = List[Int](length=max(n, 1), fill=0)
    for i in range(n):
        if gs[i] >= 0:
            idx[at[gs[i]]] = i
            at[gs[i]] += 1
    ref cache = f.cache[]
    for k in range(ncols):
        var na = meta[k * 8 + 4]
        var off = meta[k * 8 + 5]
        for g in range(groups):
            var s = counts[g]
            var e = counts[g + 1]
            if s == e:
                continue
            var slot = cache.fetch(f.scan.value(), g, cols[k])
            ref t = cache.tables[slot]
            var vals = t.cols[0].values.unsafe_ptr()
            var vaddr = t.cols[0].validity_address()
            var valid = BP(unsafe_from_address=vaddr)
            var start = f.starts[g]
            for q in range(s, e):
                var i = idx[q]
                var r = (op[lo + i] if order != 0 else lo + i) - start
                if vaddr != 0 and (valid[r >> 3] >> UInt8(r & 7)) & 1 == 0:
                    vout[k * n + i] = 0
                    nout[k * n + i] = 1
                else:
                    var v = vals[r]
                    if na > 0:
                        v = _alias_of(afrom, ato, off, na, v)
                    vout[k * n + i] = v
                    nout[k * n + i] = 0
    return -1


# ── the reductions ──────────────────────────────────────────────────────────────────────────────────

def _reduce(kind: Int, vals: LP, nulls: BP, n: Int, rec: LP, seen: BP):
    """One morsel of one column: count, sum, min, max (see _REC). A float's sum adds in position order; a text column marks the codes it met."""
    var cnt = Int64(0)
    if kind == FLOAT.value:
        var s = Float64(0.0)
        var nan = Int64(0)
        var mn = Float64(from_bits=UInt64(_INF_BITS))
        var mx = -mn
        for i in range(n):
            if nulls[i] == 0:
                var x = Float64(from_bits=UInt64(vals[i]))
                cnt += 1
                s += x
                if not (x == x):
                    nan += 1
                else:
                    if x < mn:
                        mn = x
                    if x > mx:
                        mx = x
        rec[0] = cnt
        rec[1] = Int64(s.to_bits())
        rec[3] = Int64(mn.to_bits())
        rec[4] = Int64(mx.to_bits())
        rec[5] = nan
    elif kind == TEXT.value:
        for i in range(n):
            if nulls[i] == 0:
                cnt += 1
                seen[Int(vals[i])] = 1
        rec[0] = cnt
    else:
        var hi = Int64(0)
        var lo = Int64(0)
        var mn = Int64.MAX
        var mx = Int64.MIN
        for i in range(n):
            if nulls[i] == 0:
                var v = vals[i]
                cnt += 1
                hi += v >> 32
                lo += v & 0xFFFFFFFF
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
        rec[0] = cnt
        rec[1] = hi
        rec[2] = lo
        rec[3] = mn
        rec[4] = mx


def _bad_position(total: Int, order: List[Int], identity: Bool, pos: Int, removed: Bool) -> Error:
    var r = pos if identity else order[pos]
    if removed:
        return Error("row " + String(r) + " (position " + String(pos) + ") was removed")
    return Error("position " + String(pos) + " of the order is row " + String(r) + ", but there are only " + _words_noun(total))


def _first_bad(errs: List[Int]) -> Int:
    var bad = -1
    for e in errs:
        if e >= 0 and (bad < 0 or e < bad):
            bad = e
    return bad


def _run_mem(p: _Prep, order: Int, a: Int, b: Int) -> Tuple[List[Int64], Int]:
    """The morsels' records of an in-memory source over positions a..b, on every core; (records, first bad position or -1)."""
    var n = b - a
    var ncols = len(p.cols)
    var nm = (n + MORSEL - 1) // MORSEL
    var recs = List[Int64](length=max(nm * ncols * _REC, 1), fill=0)
    if n == 0:
        return (recs^, -1)
    var d = p.desc()
    var da = Int(d.unsafe_ptr())
    var ra = Int(recs.unsafe_ptr())
    var tasks = max(1, min(core_count(), nm))
    var errs = List[Int](length=tasks, fill=-1)
    var ea = Int(errs.unsafe_ptr())

    def work(t: Int) {imm}:
        var vb = List[Int64](unsafe_uninit_length=MORSEL)
        var nb = List[UInt8](unsafe_uninit_length=MORSEL)
        var vp = LP(unsafe_from_address=Int(vb.unsafe_ptr()))
        var np = BP(unsafe_from_address=Int(nb.unsafe_ptr()))
        var meta = NP(unsafe_from_address=NP(unsafe_from_address=da)[9])
        for m in range(nm * t // tasks, nm * (t + 1) // tasks):
            var lo = a + m * MORSEL
            var hi = min(lo + MORSEL, b)
            for k in range(ncols):
                var bad = _gather_mem(da, k, order, lo, hi, vp, np)
                if bad >= 0:
                    NP(unsafe_from_address=ea)[t] = bad
                    _ = vb^
                    _ = nb^
                    return
                _reduce(meta[k * 8 + 2], vp, np, hi - lo, LP(unsafe_from_address=ra) + (m * ncols + k) * _REC,
                        BP(unsafe_from_address=meta[k * 8 + 6]))
        _ = vb^
        _ = nb^

    if tasks == 1:
        work(0)
    else:
        par_n(work, tasks, tasks)
    var bad = _first_bad(errs)
    _ = d^
    return (recs^, bad)


def _run_scan(f: Frame, p: _Prep, order: Int, a: Int, b: Int) raises -> Tuple[List[Int64], Int]:
    """The same over a Frame on a Parquet file: a morsel at a time on the calling thread (the cache is not thread-safe)."""
    var n = b - a
    var ncols = len(p.cols)
    var nm = (n + MORSEL - 1) // MORSEL
    var recs = List[Int64](length=max(nm * ncols * _REC, 1), fill=0)
    if n == 0:
        return (recs^, -1)
    var d = p.desc()
    var da = Int(d.unsafe_ptr())
    var vb = List[Int64](unsafe_uninit_length=ncols * min(n, MORSEL))
    var nb = List[UInt8](unsafe_uninit_length=ncols * min(n, MORSEL))
    var vp = LP(unsafe_from_address=Int(vb.unsafe_ptr()))
    var np = BP(unsafe_from_address=Int(nb.unsafe_ptr()))
    var ra = LP(unsafe_from_address=Int(recs.unsafe_ptr()))
    var meta = NP(unsafe_from_address=NP(unsafe_from_address=da)[9])
    var bad = -1
    for m in range(nm):
        var lo = a + m * MORSEL
        var hi = min(lo + MORSEL, b)
        bad = _gather_scan(f, p.cols, da, order, lo, hi, vp, np)
        if bad >= 0:
            break
        for k in range(ncols):
            _reduce(meta[k * 8 + 2], vp + k * (hi - lo), np + k * (hi - lo), hi - lo, ra + (m * ncols + k) * _REC,
                    BP(unsafe_from_address=meta[k * 8 + 6]))
    _ = vb^
    _ = nb^
    _ = d^
    _ = recs.__len__()
    return (recs^, bad)


def _fold(p: _Prep, recs: List[Int64], nm: Int, rows: Int, names: List[String]) raises -> RangeResult:
    """The morsels' records of each column folded in morsel order into the column's aggregates."""
    var out = RangeResult(rows)
    var ncols = len(p.cols)
    for k in range(ncols):
        var kind = p.kinds[k]
        var agg = RangeAgg(ColumnKind(kind), p.scales[k])
        var is_float = kind == FLOAT.value
        var numeric = kind == INT.value or kind == DECIMAL.value
        var cnt = 0
        var hi_t = Int64(0)
        var lo_t = Int64(0)
        var s = Float64(0.0)
        var nan = Int64(0)
        var mn = Int64.MAX
        var mx = Int64.MIN
        var fmn = Float64(from_bits=UInt64(_INF_BITS))
        var fmx = -fmn
        for m in range(nm):
            var at = (m * ncols + k) * _REC
            if recs[at] == 0:
                continue
            cnt += Int(recs[at])
            if is_float:
                s += Float64(from_bits=UInt64(recs[at + 1]))
                nan += recs[at + 5]
                var x = Float64(from_bits=UInt64(recs[at + 3]))
                if x < fmn:
                    fmn = x
                x = Float64(from_bits=UInt64(recs[at + 4]))
                if x > fmx:
                    fmx = x
            elif kind != TEXT.value:
                hi_t += recs[at + 1]
                lo_t += recs[at + 2]
                mn = min(mn, recs[at + 3])
                mx = max(mx, recs[at + 4])
        agg.count = cnt
        if cnt > 0 and is_float:
            agg.has_sum = True
            agg.sum = Int64(s.to_bits())
            agg.mean = s / Float64(cnt)
            agg.min = _NAN_BITS if Int64(cnt) == nan else Int64(fmn.to_bits())    # MIN is NaN only when nothing else is there
            agg.max = _NAN_BITS if nan > 0 else Int64(fmx.to_bits())              # MAX is NaN when one is
        elif cnt > 0 and numeric:
            var h = hi_t + (lo_t >> 32)
            if h < -2147483648 or h > 2147483647:
                raise Error("the sum of column `" + names[k] + "` passed the int64 range (about 9.2e18)")
            agg.has_sum = True
            agg.sum = (h << 32) + (lo_t & 0xFFFFFFFF)
            agg.mean = avg_div(Int128(agg.sum), Int(cnt), agg.scale)
            agg.min = mn
            agg.max = mx
        elif cnt > 0 and kind != TEXT.value:
            agg.min = mn
            agg.max = mx
        out.columns.append(agg^)
    return out^


def _text_extent_table(t: SharedTable, p: _Prep, mut r: RangeResult):
    """MIN / MAX of each TEXT column from the codes its morsels marked: the words, compared in the case-insensitive order."""
    for k in range(len(p.cols)):
        if p.kinds[k] != TEXT.value or r.columns[k].count == 0:
            continue
        var words = t.dictionary(p.cols[k])
        var lo = -1
        var hi = -1
        for code in range(p.words[k]):
            if p.seen[k][code] == 0:
                continue
            if lo < 0:
                lo = code
                hi = code
                continue
            if _text_less(words[code], words[lo]):
                lo = code
            if _text_less(words[hi], words[code]):
                hi = code
        r.columns[k].min = Int64(lo)
        r.columns[k].max = Int64(hi)


def _text_extent_frame(f: Frame, p: _Prep, mut r: RangeResult) raises:
    for k in range(len(p.cols)):
        if p.kinds[k] != TEXT.value or r.columns[k].count == 0:
            continue
        var c = p.cols[k]
        var lo = -1
        var hi = -1
        var lo_word = String()
        var hi_word = String()
        for code in range(p.words[k]):
            if p.seen[k][code] == 0:
                continue
            var w = f.text_of(c, Int64(code))
            if lo < 0:
                lo = code
                hi = code
                lo_word = w
                hi_word = w
                continue
            if _text_less(w, lo_word):
                lo = code
                lo_word = w
            if _text_less(hi_word, w):
                hi = code
                hi_word = w
        r.columns[k].min = Int64(lo)
        r.columns[k].max = Int64(hi)


# ── range aggregates ────────────────────────────────────────────────────────────────────────────────

def _aggregate_table(t: SharedTable, order: List[Int], identity: Bool, cols: List[Int], a: Int, b: Int) raises -> RangeResult:
    _check_range(a, b, t.rows() if identity else len(order), not identity)
    var p = _prep_table(t, cols, True)
    var names = List[String]()
    for c in cols:
        names.append(t.name(c))
    var run = _run_mem(p, 0 if identity else Int(order.unsafe_ptr()), a, b)
    if run[1] >= 0:
        raise _bad_position(p.total, order, identity, run[1], False)
    var r = _fold(p, run[0], (b - a + MORSEL - 1) // MORSEL, b - a, names)
    _text_extent_table(t, p, r)
    return r^


def _aggregate_frame(f: Frame, order: List[Int], identity: Bool, cols: List[Int], a: Int, b: Int) raises -> RangeResult:
    _check_range(a, b, f.rows() if identity else len(order), not identity)
    var p = _prep_frame(f, cols, order, identity, a, b, True)
    var names = List[String]()
    for c in cols:
        names.append(f.name(c))
    var oa = 0 if identity else Int(order.unsafe_ptr())
    var run = _run_scan(f, p, oa, a, b) if f.scan else _run_mem(p, oa, a, b)
    if run[1] >= 0:
        var row = run[1] if identity else order[run[1]]
        raise _bad_position(p.total, order, identity, run[1], 0 <= row < p.total)
    var r = _fold(p, run[0], (b - a + MORSEL - 1) // MORSEL, b - a, names)
    _text_extent_frame(f, p, r)
    return r^


def range_aggregate(t: SharedTable, order: List[Int], cols: List[Int], a: Int, b: Int) raises -> RangeResult:
    """count / sum / mean / min / max of each column of `cols` over positions a <= p < b of `order` (a list of row numbers)."""
    return _aggregate_table(t, order, False, cols, a, b)


def range_aggregate(t: SharedTable, cols: List[Int], a: Int, b: Int) raises -> RangeResult:
    """The same over rows a <= r < b in table order."""
    return _aggregate_table(t, List[Int](), True, cols, a, b)


def range_aggregate(f: Frame, order: List[Int], cols: List[Int], a: Int, b: Int) raises -> RangeResult:
    """The same over a Frame: its edits, appended rows and merges are included; a removed row in the range raises."""
    return _aggregate_frame(f, order, False, cols, a, b)


def range_aggregate(f: Frame, cols: List[Int], a: Int, b: Int) raises -> RangeResult:
    return _aggregate_frame(f, List[Int](), True, cols, a, b)


# ── block reads ─────────────────────────────────────────────────────────────────────────────────────

def _read_mem(p: _Prep, order: Int, a: Int, b: Int, vaddr: Int, naddr: Int) -> Int:
    """Gathers every column of the call at positions a..b into the caller's buffers (column k at k * (b - a)); on every core when the range is long."""
    var n = b - a
    var ncols = len(p.cols)
    var d = p.desc()
    var da = Int(d.unsafe_ptr())
    var tasks = max(1, min(core_count(), n // MORSEL))
    var errs = List[Int](length=tasks, fill=-1)
    var ea = Int(errs.unsafe_ptr())

    def work(t: Int) {imm}:
        var lo = a + n * t // tasks
        var hi = a + n * (t + 1) // tasks
        for k in range(ncols):
            var bad = _gather_mem(da, k, order, lo, hi, LP(unsafe_from_address=vaddr) + (k * n + lo - a), BP(unsafe_from_address=naddr) + (k * n + lo - a))
            if bad >= 0:
                NP(unsafe_from_address=ea)[t] = bad
                return

    if n > 0:
        if tasks == 1:
            work(0)
        else:
            par_n(work, tasks, tasks)
    var bad = _first_bad(errs)
    _ = d^
    return bad


def _check_buffers(values: List[Int64], nulls: List[UInt8], need: Int) raises:
    if len(values) < need or len(nulls) < need:
        raise Error("the buffers hold " + String(min(len(values), len(nulls))) + " cells but the block has " + String(need) + " (columns x rows)")


def _read_table(t: SharedTable, order: List[Int], identity: Bool, cols: List[Int], a: Int, b: Int, vaddr: Int, naddr: Int) raises:
    var p = _prep_table(t, cols, False)
    var bad = _read_mem(p, 0 if identity else Int(order.unsafe_ptr()), a, b, vaddr, naddr)
    if bad >= 0:
        raise _bad_position(p.total, order, identity, bad, False)


def _read_frame(f: Frame, order: List[Int], identity: Bool, cols: List[Int], a: Int, b: Int, vaddr: Int, naddr: Int) raises:
    var p = _prep_frame(f, cols, order, identity, a, b, False)
    var oa = 0 if identity else Int(order.unsafe_ptr())
    var bad = -1
    if b > a:
        if f.scan:
            var d = p.desc()
            bad = _gather_scan(f, cols, Int(d.unsafe_ptr()), oa, a, b, LP(unsafe_from_address=vaddr), BP(unsafe_from_address=naddr))
            _ = d^
        else:
            bad = _read_mem(p, oa, a, b, vaddr, naddr)
    if bad >= 0:
        var row = bad if identity else order[bad]
        raise _bad_position(p.total, order, identity, bad, 0 <= row < p.total)


def read_block(t: SharedTable, order: List[Int], cols: List[Int], a: Int, b: Int, mut values: List[Int64], mut nulls: List[UInt8]) raises:
    """Positions a <= p < b of `order` for each column of `cols`, into the caller's buffers, column-major: cell (column k, row i) is at
    `k * (b - a) + i`. `values` holds the stored value (a DECIMAL scaled, a DATE in days, a TEXT code (`t.text_of`), a FLOAT's bits), `nulls` is
    1 for a null cell (its value 0). Both need `len(cols) * (b - a)` cells."""
    _check_range(a, b, len(order), True)
    _check_columns(cols, t.columns(), "the table")
    _check_buffers(values, nulls, len(cols) * (b - a))
    _read_table(t, order, False, cols, a, b, Int(values.unsafe_ptr()), Int(nulls.unsafe_ptr()))


def read_block(t: SharedTable, cols: List[Int], a: Int, b: Int, mut values: List[Int64], mut nulls: List[UInt8]) raises:
    """The same over rows a <= r < b in table order."""
    _check_range(a, b, t.rows(), False)
    _check_columns(cols, t.columns(), "the table")
    _check_buffers(values, nulls, len(cols) * (b - a))
    _read_table(t, List[Int](), True, cols, a, b, Int(values.unsafe_ptr()), Int(nulls.unsafe_ptr()))


def read_block(f: Frame, order: List[Int], cols: List[Int], a: Int, b: Int, mut values: List[Int64], mut nulls: List[UInt8]) raises:
    """The same over a Frame: edits, appended rows and merges included (a TEXT code is `f.text_of`'s); a removed row in the range raises."""
    _check_range(a, b, len(order), True)
    _check_buffers(values, nulls, len(cols) * (b - a))
    _read_frame(f, order, False, cols, a, b, Int(values.unsafe_ptr()), Int(nulls.unsafe_ptr()))


def read_block(f: Frame, cols: List[Int], a: Int, b: Int, mut values: List[Int64], mut nulls: List[UInt8]) raises:
    _check_range(a, b, f.rows(), False)
    _check_buffers(values, nulls, len(cols) * (b - a))
    _read_frame(f, List[Int](), True, cols, a, b, Int(values.unsafe_ptr()), Int(nulls.unsafe_ptr()))
