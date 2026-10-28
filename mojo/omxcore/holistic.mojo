"""Holistic aggregates (F-148): count distinct, percentiles and top-k per group, exact, on all cores.

A sum, min or max folds one row at a time into a number per group; these three need more than that, so the query
kernel does not hold them. Instead its per-block hook (query.mojo `_mat_hook`) writes, for each distinct value
expression ("source"), the (group, value) pair of every row that is selected, whose group is in range and whose value
is not null, packed block by block (a `Window`). This file consumes those pairs:

- count distinct: a hash set of (group, value) pairs, split into P partitions by the pair's hash. A window's pairs are
  first scattered to their partitions (parallel, two passes), then each partition inserts its own pairs (parallel, no
  locks). The set persists between windows, so a streamed scan merges row group after row group exactly. Memory:
  16 bytes a slot, about 1.5 slots per distinct (group, value) pair.
- top-k: the k largest values of a group, duplicates kept (SQL `ORDER BY e DESC LIMIT k`): a bounded min-heap of k
  values per group (O(rows log k), almost all rows rejected by one compare with the heap's smallest). Few groups: every
  task keeps private heaps over its blocks and they are merged (when top-k is the only aggregate on its values, the
  kernel hook feeds per-worker heaps directly and writes no pairs, query.mojo `_mat_hook`); many groups: the pairs are
  scattered by group so that one task owns each group's heap. The state is k values a group, and merging windows is
  just offering their values.
- percentile: the value at index max(ceil(p * n) - 1, 0) of the group's sorted non-null values, which is DuckDB's
  `quantile_disc` (an actual value, exact in the stored integers; `median` is DuckDB's `median` (quantile_cont at 0.5): the mean of the lower and the
  upper middle value, found with one extra select and kept exact as 5 * (a + b), the stored value x 10; text stays the
  lower middle word). Exact percentiles need every value: the
  pairs are kept (12 bytes a non-null row, plus 8 bytes a row while the answer is computed), grouped with a parallel
  counting sort, then each group is selected, not sorted: quickselect (three-way, O(n)) for small groups, and for a
  group of 262,144 values or more a parallel most-significant-byte radix select, which reads the values
  without moving them (about two passes for values in a small range, at most eight).
- variance and standard deviation (F-177): O(groups) state, a count, a mean and the sum of squared deviations M2 per group. A window's values
  are grouped, each group is cut into pieces of SPREAD_PIECE values in their order, each piece is two-pass (a Welford mean, then
  deviations from it with a correction term, sums compensated), and pieces and windows are merged left to right with Chan's formula: stable
  against cancellation (1e9 + 1, 1e9 + 2 ...), and the same bits for any number of threads. Integers and decimals become float64 as DuckDB
  casts them (`float(v) / 10^scale`, past 2^53 rounded, never an error); float64 NaN / inf make the result NaN.
- text: percentile and top-k order words alphabetically (the column's sort order, `Frame.permutation`'s) not by code:
  the values are mapped to their rank before ordering and back to the code after (`ranks`, `unranks`).
"""
from std.memory import Pointer
from std.sys import prefetch, PrefetchOptions
from std.atomic import Atomic
from std.bit import count_leading_zeros
from std.math import ceil, log, sqrt, isfinite
from .kernels._fpenv import core_count, par_n

comptime LP = Pointer[Int64, MutUntrackedOrigin]
comptime IP = Pointer[Int32, MutUntrackedOrigin]
comptime NP = Pointer[Int, MutUntrackedOrigin]
comptime BP = Pointer[UInt8, MutUntrackedOrigin]
comptime HLL_M = 256               # registers of a HyperLogLog sketch (standard error about 6.5%)

# aggregate kinds (the plain ones are in query.mojo)
comptime A_DISTINCT = 6
comptime A_PCT = 7
comptime A_TOPK = 8
comptime A_MEDIAN = 9                 # DuckDB's `median` (quantile_cont): the mean of the two middle values, at scale + 1
comptime A_VAR_POP = 10               # var_pop / var_samp / stddev_pop / stddev_samp (F-177): a float64 from the group's count, mean and sum of squared deviations
comptime A_VAR_SAMP = 11
comptime A_STDDEV_POP = 12
comptime A_STDDEV_SAMP = 13
comptime A_QEDGE = 14                # edge k of n quantile bins (F-177): the value at sorted place ceil(k * N / n), by integer arithmetic (`p` holds k, `k` holds n)
comptime SPREAD_PIECE = 65536        # values folded into one (count, mean, M2) piece: fixed, so the bits do not depend on the number of threads
comptime UPPER_MIDDLE = 2.0          # a `p` that asks _percentiles for the upper middle value, index n // 2

comptime BLOCK = 65536               # entries in a block made from a flat array
comptime RADIX_MIN = 262_144         # a group at least this big is selected by the parallel radix select
comptime HIST_BUDGET = 4_000_000     # counters in the per-task histograms of a counting sort
comptime PRIVATE_HEAPS = 1 << 22     # heap words all tasks may hold privately (top-k with few groups)
comptime MAX_TOPK_WORDS = 1 << 28    # groups x k a top-k state may hold (2 GiB)
comptime SIGN = UInt64(1) << 63


@always_inline
def float_key(bits: Int64) -> Int64:
    """A float64's bits as an int64 whose signed order is the float order DuckDB uses: -0.0 equal to 0.0, every NaN one value
    above +inf, negative numbers below positive ones (F-157). Equal keys are equal values, so count distinct works on them."""
    var x = Float64(from_bits=UInt64(bits))
    var b = bits
    if x != x:
        b = Int64(0x7FF8000000000000)
    elif x == 0.0:
        b = 0
    return b ^ ((b >> 63) & Int64(0x7FFFFFFFFFFFFFFF))


@always_inline
def float_unkey(k: Int64) -> Int64:
    """The float64 bits of a `float_key`."""
    return k ^ ((k >> 63) & Int64(0x7FFFFFFFFFFFFFFF))


def median_of_floats(lo: Float64, hi: Float64, n: Int) -> Float64:
    """DuckDB's `median` of doubles from the two middle values of n sorted ones: the middle value for an odd n, else
    lo * 0.5 + hi * 0.5 (its interpolation: it neither overflows for two huge values nor loses the sign of an infinity)."""
    if n % 2 == 1:
        return lo
    return lo * 0.5 + hi * 0.5


def _lp(a: Int) -> LP:
    return LP(unsafe_from_address=a)


def _ip(a: Int) -> IP:
    return IP(unsafe_from_address=a)


def _np(a: Int) -> NP:
    return NP(unsafe_from_address=a)


def _bp(a: Int) -> BP:
    return BP(unsafe_from_address=a)


@always_inline
def _cut(n: Int, parts: Int, i: Int) -> Int:
    return n * i // parts


@always_inline
def _mix(v: Int64, g: Int32) -> UInt64:
    var x = UInt64(v) ^ (UInt64(Int(g)) * 0x9E3779B97F4A7C15)
    x = (x ^ (x >> 32)) * 0xD6E8FEB86659FD93
    x = (x ^ (x >> 32)) * 0xD6E8FEB86659FD93
    return x ^ (x >> 32)


def _tasks(total: Int, blocks: Int, threads: Int) -> Int:
    """How many tasks to split `blocks` blocks over: one for a small input or a serial build."""
    if total < 32768 or threads <= 1:
        return 1
    return max(1, min(threads, blocks))


# ── the pairs ────────────────────────────────────────────────────────────────────────────────────────

struct Blocks(Movable):
    """A list of runs of (group, value) pairs: block i has `n[i]` Int32 groups at address `g[i]` and as many Int64
    values at `v[i]`. Every group is in range and every value is non-null (the kernel hook and `HolStates.absorb_rows`
    leave the others out)."""
    var g: List[Int]
    var v: List[Int]
    var n: List[Int]
    var total: Int

    def __init__(out self):
        self.g = List[Int]()
        self.v = List[Int]()
        self.n = List[Int]()
        self.total = 0

    def add(mut self, g: Int, v: Int, n: Int):
        if n > 0:
            self.g.append(g)
            self.v.append(v)
            self.n.append(n)
            self.total += n

    def add_flat(mut self, g: Int, v: Int, n: Int):
        """`n` consecutive pairs, cut into blocks that tasks can share out."""
        var at = 0
        while at < n:
            var m = min(BLOCK, n - at)
            self.add(g + at * 4, v + at * 8, m)
            at += m


def window_blocks(gids: Int, vals: Int, blk_n: Int, nblocks: Int, blk: Int) -> Blocks:
    """The blocks of a kernel window: block b holds `blk_n[b]` pairs at entry b * blk."""
    var out = Blocks()
    var bn = _ip(blk_n)
    for b in range(nblocks):
        out.add(gids + b * blk * 4, vals + b * blk * 8, Int(bn[b]))
    return out^


struct Scattered(Movable):
    """The pairs of some blocks regrouped by part: part p is entries seg[p] .. seg[p+1] of `vals` and `gids`."""
    var seg: List[Int]
    var vals: List[Int64]
    var gids: List[Int32]
    var hll: List[UInt8]       # when asked for: HyperLogLog registers, `tasks x parts x HLL_M`, of the pairs' hashes

    def __init__(out self):
        self.seg = List[Int]()
        self.vals = List[Int64]()
        self.gids = List[Int32]()
        self.hll = List[UInt8]()


@always_inline
def _owner(v: Int64, g: Int32, nparts: Int, by_hash: Bool) -> Int:
    if by_hash:
        return Int((_mix(v, g) >> 40) & UInt64(nparts - 1))
    return Int(g) % nparts


def _scatter(b: Blocks, nparts: Int, by_hash: Bool, rank: Int, threads: Int, want_hll: Bool = False) -> Scattered:
    """Regroup the pairs into `nparts` parts: by the hash of (group, value) (`nparts` a power of two) or by group
    modulo `nparts`; values are mapped through `rank` (an Int64 table, 0: none). Two parallel passes over the blocks
    (count, then place); the order inside a part follows the blocks. `want_hll` (by hash): the count pass also fills
    a HyperLogLog sketch per task and part, to size the tables the pairs go into."""
    var nb = len(b.n)
    var tp = _tasks(b.total, nb, threads)
    var cnt = List[Int](length=tp * nparts, fill=0)
    var cnt_at = Int(cnt.unsafe_ptr())
    var sketch = List[UInt8](length=tp * nparts * HLL_M if want_hll else 1, fill=0)
    var sketch_at = Int(sketch.unsafe_ptr())
    var bg = Int(b.g.unsafe_ptr())
    var bv = Int(b.v.unsafe_ptr())
    var bn = Int(b.n.unsafe_ptr())

    def count(t: Int) {imm}:
        var c = _np(cnt_at) + t * nparts
        var regs = _bp(sketch_at) + t * nparts * HLL_M
        for k in range(_cut(nb, tp, t), _cut(nb, tp, t + 1)):
            var gp = _ip(_np(bg)[k])
            var vp = _lp(_np(bv)[k])
            if want_hll:
                for j in range(_np(bn)[k]):
                    var h = _mix(vp[j], gp[j])
                    var p = Int((h >> 40) & UInt64(nparts - 1))
                    c[p] += 1
                    var h2 = (h ^ (h >> 29)) * 0xBF58476D1CE4E5B9
                    var r = regs + (p * HLL_M + Int(h2 >> 56))
                    var rho = UInt8(count_leading_zeros((h2 << 8) | 128)) + 1
                    if r[0] < rho:
                        r[0] = rho
            else:
                for j in range(_np(bn)[k]):
                    c[_owner(vp[j], gp[j], nparts, by_hash)] += 1

    par_n(count, tp, tp)
    var out = Scattered()
    if want_hll:
        out.hll = sketch^
    out.seg = List[Int](length=nparts + 1, fill=0)
    var run = 0
    for p in range(nparts):
        out.seg[p] = run
        for t in range(tp):
            var c = cnt[t * nparts + p]
            cnt[t * nparts + p] = run
            run += c
    out.seg[nparts] = run
    out.vals = List[Int64](unsafe_uninit_length=max(run, 1))
    out.gids = List[Int32](unsafe_uninit_length=max(run, 1))
    var ov = Int(out.vals.unsafe_ptr())
    var og = Int(out.gids.unsafe_ptr())

    def place(t: Int) {imm}:
        var pos = _np(cnt_at) + t * nparts
        var dv = _lp(ov)
        var dg = _ip(og)
        for k in range(_cut(nb, tp, t), _cut(nb, tp, t + 1)):
            var gp = _ip(_np(bg)[k])
            var vp = _lp(_np(bv)[k])
            for j in range(_np(bn)[k]):
                var p = _owner(vp[j], gp[j], nparts, by_hash)
                var v = vp[j]
                if rank != 0:
                    v = _lp(rank)[Int(v)]
                dv[pos[p]] = v
                dg[pos[p]] = gp[j]
                pos[p] += 1

    par_n(place, tp, tp)
    _ = cnt^
    return out^


struct Grouped(Movable):
    """The values of the blocks sorted into groups by a counting sort: group g's values are
    `vals[off[g] : off[g + 1]]`, in no particular order."""
    var off: List[Int]
    var vals: List[Int64]

    def __init__(out self):
        self.off = List[Int]()
        self.vals = List[Int64]()


def _group(b: Blocks, groups: Int, rank: Int, threads: Int) -> Grouped:
    """Counting sort of the pairs by group, in parallel: each task counts the groups of its blocks, the counters give
    every (group, task) its place, and each task writes its values there. The counters are `tasks x groups`, so the
    tasks are fewer when the groups are many (down to one for 4 million groups and more)."""
    var nb = len(b.n)
    var tp = min(_tasks(b.total, nb, threads), max(1, HIST_BUDGET // max(groups, 1)))
    var hist = List[Int](length=tp * groups, fill=0)
    var hist_at = Int(hist.unsafe_ptr())
    var bg = Int(b.g.unsafe_ptr())
    var bv = Int(b.v.unsafe_ptr())
    var bn = Int(b.n.unsafe_ptr())

    def count(t: Int) {imm}:
        var h = _np(hist_at) + t * groups
        for k in range(_cut(nb, tp, t), _cut(nb, tp, t + 1)):
            var gp = _ip(_np(bg)[k])
            for j in range(_np(bn)[k]):
                h[Int(gp[j])] += 1

    par_n(count, tp, tp)
    var out = Grouped()
    out.off = List[Int](length=groups + 1, fill=0)
    var run = 0
    for g in range(groups):
        out.off[g] = run
        for t in range(tp):
            var c = hist[t * groups + g]
            hist[t * groups + g] = run
            run += c
    out.off[groups] = run
    out.vals = List[Int64](unsafe_uninit_length=max(run, 1))
    var ov = Int(out.vals.unsafe_ptr())

    def place(t: Int) {imm}:
        var pos = _np(hist_at) + t * groups
        var dv = _lp(ov)
        for k in range(_cut(nb, tp, t), _cut(nb, tp, t + 1)):
            var gp = _ip(_np(bg)[k])
            var vp = _lp(_np(bv)[k])
            for j in range(_np(bn)[k]):
                var g = Int(gp[j])
                var v = vp[j]
                if rank != 0:
                    v = _lp(rank)[Int(v)]
                dv[pos[g]] = v
                pos[g] += 1

    par_n(place, tp, tp)
    _ = hist^
    return out^


# ── selection ────────────────────────────────────────────────────────────────────────────────────────

def _select_nth(p: LP, n: Int, k: Int) -> Int64:
    """The value of rank k (0-based) among the n values at p, by quickselect with a three-way partition (equal values
    cost nothing); the values are left permuted. Pivots are the median of three pseudo-random elements, so no
    input order is a worst case."""
    var lo = 0
    var hi = n - 1
    var rng = UInt64(0x2545F4914F6CDD1D) ^ UInt64(n)
    while True:
        if hi - lo < 24:                              # insertion sort the small remainder
            for i in range(lo + 1, hi + 1):
                var x = p[i]
                var j = i - 1
                while j >= lo and p[j] > x:
                    p[j + 1] = p[j]
                    j -= 1
                p[j + 1] = x
            return p[k]
        var span = UInt64(hi - lo + 1)
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        var a = p[lo + Int(rng % span)]
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        var b = p[lo + Int(rng % span)]
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        var c = p[lo + Int(rng % span)]
        var pivot = max(min(a, b), min(max(a, b), c))
        var lt = lo
        var i = lo
        var gt = hi
        while i <= gt:
            var x = p[i]
            if x < pivot:
                p[i] = p[lt]
                p[lt] = x
                lt += 1
                i += 1
            elif x > pivot:
                p[i] = p[gt]
                p[gt] = x
                gt -= 1
            else:
                i += 1
        if k < lt:
            hi = lt - 1
        elif k > gt:
            lo = gt + 1
        else:
            return pivot


def _radix_select(addr: Int, n: Int, k: Int, threads: Int) -> Int64:
    """The value of rank k (0-based) among the n values at `addr`, read-only and in parallel: one pass finds the
    smallest and largest key (the value with its sign bit flipped, so keys order as unsigned), the bytes they share
    are skipped, then one pass per remaining byte counts how many values of the prefix fall in each of its 256
    digits and narrows the prefix to the digit holding rank k. At most eight passes; values in a small range need
    two."""
    var tp = max(1, min(threads, n // 65536))
    var lo_k = List[UInt64](length=tp, fill=UInt64.MAX)
    var hi_k = List[UInt64](length=tp, fill=0)
    var lo_at = Int(lo_k.unsafe_ptr())
    var hi_at = Int(hi_k.unsafe_ptr())

    def extent(t: Int) {imm}:
        var p = _lp(addr)
        var lo = UInt64.MAX
        var hi = UInt64(0)
        for i in range(_cut(n, tp, t), _cut(n, tp, t + 1)):
            var key = UInt64(p[i]) ^ SIGN
            lo = min(lo, key)
            hi = max(hi, key)
        Pointer[UInt64, MutUntrackedOrigin](unsafe_from_address=lo_at)[t] = lo
        Pointer[UInt64, MutUntrackedOrigin](unsafe_from_address=hi_at)[t] = hi

    par_n(extent, tp, tp)
    var lo = UInt64.MAX
    var hi = UInt64(0)
    for t in range(tp):
        lo = min(lo, lo_k[t])
        hi = max(hi, hi_k[t])
    if lo == hi:
        return Int64(lo ^ SIGN)
    var lead = Int(count_leading_zeros(lo ^ hi)) // 8          # bytes both ends share
    var prefix = (lo >> UInt64(64 - 8 * lead)) if lead > 0 else UInt64(0)
    var rem = k
    var hist = List[Int](length=tp * 256, fill=0)
    var hist_at = Int(hist.unsafe_ptr())
    for pass_ in range(lead, 8):
        var shift = 56 - 8 * pass_
        var want = prefix
        for i in range(tp * 256):
            hist[i] = 0

        def digits(t: Int) {imm}:
            var p = _lp(addr)
            var h = _np(hist_at) + t * 256
            for i in range(_cut(n, tp, t), _cut(n, tp, t + 1)):
                var key = UInt64(p[i]) ^ SIGN
                if shift == 56 or (key >> UInt64(shift + 8)) == want:
                    h[Int((key >> UInt64(shift)) & 255)] += 1

        par_n(digits, tp, tp)
        var d = 0
        while True:
            var c = 0
            for t in range(tp):
                c += hist[t * 256 + d]
            if rem < c or d == 255:
                break
            rem -= c
            d += 1
        prefix = (prefix << 8) | UInt64(d)
    _ = lo_k^
    _ = hi_k^
    _ = hist^
    return Int64(prefix ^ SIGN)


@always_inline
def _percentile_index(p: Float64, n: Int, bins: Int = 0) -> Int:
    """DuckDB's `quantile_disc` index in n sorted values: ceil(p * n) - 1, at least 0. With `bins` > 0, `p` is a whole number k: the index of
    bin edge k of `bins` bins, ceil(k * n / bins) - 1, in integers (a float p = k / bins could land on the wrong side of a whole product)."""
    if bins > 0:
        return min(max((Int(p) * n + bins - 1) // bins - 1, 0), n - 1)
    if p > 1.5:                                  # UPPER_MIDDLE: the lower middle, or the one after it when n is even
        return n // 2
    var i = Int(ceil(p * Float64(n))) - 1
    return min(max(i, 0), n - 1)


struct Percentiles(Movable):
    var vals: List[List[Int]]       # per requested p: per group, the value (0 for a group with no values)
    var counts: List[Int]           # per group: its non-null values

    def __init__(out self):
        self.vals = List[List[Int]]()
        self.counts = List[Int]()


def _percentiles(b: Blocks, groups: Int, ps: List[Float64], bins: List[Int], rank: Int, unrank: Int, threads: Int) -> Percentiles:
    """Every p in `ps` for every group over the pairs: group them, then select (see the file comment)."""
    var gr = _group(b, groups, rank, threads)
    var np = len(ps)
    var out_at = List[Int](length=max(np * groups, 1), fill=0)
    var out_p = Int(out_at.unsafe_ptr())
    var ps_p = Int(ps.unsafe_ptr())
    var bins_p = Int(bins.unsafe_ptr())
    var off_at = Int(gr.off.unsafe_ptr())
    var vals_at = Int(gr.vals.unsafe_ptr())
    # small groups: balanced chunks of groups, one task each; big groups are left for the parallel select below
    var tp = _tasks(b.total, groups, threads)
    var chunks = max(1, tp * 4) if tp > 1 else 1
    var cuts = List[Int](length=chunks + 1, fill=groups)
    cuts[0] = 0
    if chunks > 1:
        var c = 1
        var g = 0
        while g < groups and c < chunks:
            if gr.off[g] >= b.total * c // chunks:
                cuts[c] = g
                c += 1
            else:
                g += 1
        for k in range(c, chunks):
            cuts[k] = groups
    var cuts_at = Int(cuts.unsafe_ptr())
    var big_ok = threads > 1

    def small(ci: Int) {imm}:
        var off = _np(off_at)
        var vals = _lp(vals_at)
        var out = _np(out_p)
        var pp = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=ps_p)
        var bp = _np(bins_p)
        for g in range(_np(cuts_at)[ci], _np(cuts_at)[ci + 1]):
            var m = off[g + 1] - off[g]
            if m == 0 or (big_ok and m >= RADIX_MIN):
                continue
            for q in range(np):
                var v = _select_nth(vals + off[g], m, _percentile_index(pp[q], m, bp[q]))
                if unrank != 0:
                    v = _lp(unrank)[Int(v)]
                out[q * groups + g] = Int(v)

    par_n(small, chunks, threads)
    if big_ok:
        for g in range(groups):
            var m = gr.off[g + 1] - gr.off[g]
            if m >= RADIX_MIN:
                for q in range(np):
                    var v = _radix_select(vals_at + gr.off[g] * 8, m, _percentile_index(ps[q], m, bins[q]), threads)
                    if unrank != 0:
                        v = _lp(unrank)[Int(v)]
                    out_at[q * groups + g] = Int(v)
    var res = Percentiles()
    for q in range(np):
        var row = List[Int](length=groups, fill=0)
        for g in range(groups):
            row[g] = out_at[q * groups + g]
        res.vals.append(row^)
    for g in range(groups):
        res.counts.append(gr.off[g + 1] - gr.off[g])
    _ = gr^
    _ = cuts^
    return res^


# ── variance and standard deviation (F-177) ──────────────────────────────────────────────────────────

@always_inline
def is_spread(kind: Int) -> Bool:
    """The aggregate is var_pop, var_samp, stddev_pop or stddev_samp."""
    return kind >= A_VAR_POP and kind <= A_STDDEV_SAMP


def spread_needs(kind: Int) -> Int:
    """How many non-null values the sample variants need for a number (they answer NULL below): 2; the population ones 1."""
    return 2 if kind == A_VAR_SAMP or kind == A_STDDEV_SAMP else 1


def spread_name(kind: Int) -> String:
    return "var_pop" if kind == A_VAR_POP else "var_samp" if kind == A_VAR_SAMP else "stddev_pop" if kind == A_STDDEV_POP else "stddev_samp"


def spread_value(kind: Int, n: Int, m2: Float64) raises -> Float64:
    """The aggregate from a group's count and sum of squared deviations (the caller checks `spread_needs`). As DuckDB, a result that is not a
    finite number (a NaN or an infinity among two or more values, or an overflow) is an error, not a value."""
    var v = m2 / Float64(n) if kind == A_VAR_POP or kind == A_STDDEV_POP else m2 / Float64(n - 1)
    if kind == A_STDDEV_POP or kind == A_STDDEV_SAMP:
        v = sqrt(v)
    if not isfinite(v):
        raise Error(spread_name(kind) + " is out of range: the values give a result that is not a finite number (a NaN or an infinity among them, or an overflow), as in DuckDB")
    return v


@always_inline
def _dbl(v: Int64, is_float: Bool, div: Float64) -> Float64:
    if is_float:
        return Float64(from_bits=UInt64(float_unkey(v)))
    if v > 9007199254740992 or v < -9007199254740992:           # F-190: DuckDB's DECIMAL -> DOUBLE past 2^53 (bind.decimal_to_float, kernel _decimal_f64)
        var pu = UInt64(div)
        var m = UInt64(0) - UInt64(v) if v < 0 else UInt64(v)
        var q = m // pu
        var x = Float64(q) + Float64(m - q * pu) / div
        return -x if v < 0 else x
    return Float64(v) / div


@always_inline
def _neumaier(mut sum: Float64, mut comp: Float64, x: Float64):
    var t = sum + x
    if abs(sum) >= abs(x):
        comp += (sum - t) + x
    else:
        comp += (x - t) + sum
    sum = t


def _piece_moments(p: LP, m: Int, is_float: Bool, div: Float64) -> Tuple[Float64, Float64, Float64]:
    """(mean, the mean's remainder, M2) of the m values at p: a Welford mean, then the deviations from it (compensated sums of d and d*d;
    M2 = sum d*d - (sum d)^2 / m). The mean is returned as a pair (hi + lo) because the pieces are merged by their difference: means of 1e12 + small
    numbers are only good to 1e-4 as one double, and the merge would lose the small part."""
    var mean = 0.0
    for i in range(m):
        mean += (_dbl(p[i], is_float, div) - mean) / Float64(i + 1)
    if m == 1:                                     # (DuckDB: one value is a spread of 0, even when the value is NaN or infinite)
        return (mean, 0.0, 0.0)
    var s1 = 0.0
    var c1 = 0.0
    var s2 = 0.0
    var c2 = 0.0
    for i in range(m):
        var d = _dbl(p[i], is_float, div) - mean
        _neumaier(s1, c1, d)
        _neumaier(s2, c2, d * d)
    var t1 = s1 + c1
    var m2 = (s2 + c2) - t1 * t1 / Float64(m)
    if m2 < 0.0:
        m2 = 0.0
    return (mean, t1 / Float64(m), m2)


@always_inline
def _chan(mut n: Int, mut mean: Float64, mut lo: Float64, mut m2: Float64, nb: Int, mean_b: Float64, lo_b: Float64, m2_b: Float64):
    """Folds the moments of nb more values into (n, mean + lo, M2): Chan's pairwise formula, the means kept as pairs so that their difference is exact."""
    if nb == 0:
        return
    if n == 0:
        n = nb
        mean = mean_b
        lo = lo_b
        m2 = m2_b
        return
    var tot = n + nb
    var delta = (mean_b - mean) + (lo_b - lo)
    m2 = m2 + m2_b + delta * delta * Float64(n) * Float64(nb) / Float64(tot)
    var step = delta * Float64(nb) / Float64(tot)
    var hi = mean + step                           # (mean + lo) + step, renormalized into a pair
    var bb = hi - mean
    var err = (mean - (hi - bb)) + (step - bb)
    var l2 = lo + err
    var top = hi + l2
    lo = l2 - (top - hi)
    mean = top
    n = tot


struct VarState(Movable):
    """Per group the count, mean and sum of squared deviations of the values seen so far (one per value source that a spread aggregate reads)."""
    var n: List[Int]
    var mean: List[Float64]
    var lo: List[Float64]            # the mean is mean + lo
    var m2: List[Float64]

    def __init__(out self, groups: Int):
        self.n = List[Int](length=groups, fill=0)
        self.mean = List[Float64](length=groups, fill=0.0)
        self.lo = List[Float64](length=groups, fill=0.0)
        self.m2 = List[Float64](length=groups, fill=0.0)

    def absorb(mut self, b: Blocks, is_float: Bool, div: Float64, threads: Int):
        var groups = len(self.n)
        var gr = _group(b, groups, 0, threads)
        var off_at = Int(gr.off.unsafe_ptr())
        var vals_at = Int(gr.vals.unsafe_ptr())
        var n_at = Int(self.n.unsafe_ptr())
        var mean_at = Int(self.mean.unsafe_ptr())
        var lo_at = Int(self.lo.unsafe_ptr())
        var m2_at = Int(self.m2.unsafe_ptr())
        var tp = _tasks(b.total, groups, threads)
        var chunks = max(1, tp * 4) if tp > 1 else 1
        var big = 4 * SPREAD_PIECE if threads > 1 else Int.MAX            # a group this big is cut over the threads, piece by piece
        # balanced chunks of groups, one task each
        var cuts = List[Int](length=chunks + 1, fill=groups)
        cuts[0] = 0
        if chunks > 1:
            var c = 1
            var g = 0
            while g < groups and c < chunks:
                if gr.off[g] >= b.total * c // chunks:
                    cuts[c] = g
                    c += 1
                else:
                    g += 1
            for k in range(c, chunks):
                cuts[k] = groups
        var cuts_at = Int(cuts.unsafe_ptr())

        def small(ci: Int) {imm}:
            var off = _np(off_at)
            var vals = _lp(vals_at)
            var cn = _np(n_at)
            var cmean = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=mean_at)
            var clo = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=lo_at)
            var cm2 = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=m2_at)
            for g in range(_np(cuts_at)[ci], _np(cuts_at)[ci + 1]):
                var m = off[g + 1] - off[g]
                if m == 0 or m >= big:
                    continue
                var at = 0
                while at < m:
                    var k = min(SPREAD_PIECE, m - at)
                    var pm = _piece_moments(vals + off[g] + at, k, is_float, div)
                    _chan(cn[g], cmean[g], clo[g], cm2[g], k, pm[0], pm[1], pm[2])
                    at += k

        par_n(small, chunks, threads)
        for g in range(groups):
            var m = gr.off[g + 1] - gr.off[g]
            if m < big:
                continue
            var pieces = (m + SPREAD_PIECE - 1) // SPREAD_PIECE
            var pmean = List[Float64](length=pieces, fill=0.0)
            var plo = List[Float64](length=pieces, fill=0.0)
            var pm2 = List[Float64](length=pieces, fill=0.0)
            var pmean_at = Int(pmean.unsafe_ptr())
            var plo_at = Int(plo.unsafe_ptr())
            var pm2_at = Int(pm2.unsafe_ptr())
            var base = vals_at + gr.off[g] * 8
            var tpc = min(threads, pieces)

            def piece(t: Int) {imm}:
                for k in range(_cut(pieces, tpc, t), _cut(pieces, tpc, t + 1)):
                    var len_k = min(SPREAD_PIECE, m - k * SPREAD_PIECE)
                    var r = _piece_moments(_lp(base) + k * SPREAD_PIECE, len_k, is_float, div)
                    Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=pmean_at)[k] = r[0]
                    Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=plo_at)[k] = r[1]
                    Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=pm2_at)[k] = r[2]

            par_n(piece, tpc, tpc)
            for k in range(pieces):
                _chan(self.n[g], self.mean[g], self.lo[g], self.m2[g], min(SPREAD_PIECE, m - k * SPREAD_PIECE), pmean[k], plo[k], pm2[k])
            _ = pmean^
            _ = plo^
            _ = pm2^
        _ = gr^
        _ = cuts^


# ── count distinct ───────────────────────────────────────────────────────────────────────────────────

comptime PREFETCH_AHEAD = 12     # entries ahead whose table slot is prefetched while inserting


def _hll_estimate(regs: BP) -> Float64:
    """The distinct count a HyperLogLog sketch of HLL_M registers stands for."""
    var total = 0.0
    var zeros = 0
    for i in range(HLL_M):
        total += 1.0 / Float64(UInt64(1) << UInt64(regs[i]))
        zeros += Int(regs[i] == 0)
    var m = Float64(HLL_M)
    var e = 0.7213 / (1.0 + 1.079 / m) * m * m / total
    if e <= 2.5 * m and zeros > 0:
        e = m * log(m / Float64(zeros))               # few distinct values: linear counting
    return e


def _cap_for(est: Float64) -> Int:
    """Slots for `est` pairs, plus a little for the sketch's error, at about 70% full (any number: a slot is found by
    multiplying, not masking)."""
    return max(Int(est * 1.1 / 0.7) + 64, 1024)


@always_inline
def _slot(h: UInt64, cap: Int) -> Int:
    """The home slot of hash h in a table of `cap` slots: the low 32 bits scaled to 0 .. cap-1."""
    return Int(((h & 0xFFFFFFFF) * UInt64(cap)) >> 32)


def _dist_grow(mut tab: List[Int64], old_cap: Int, cap: Int):
    """Rehash one partition's table of `old_cap` slots into a new one of `cap`. A slot is two words:
    the value, and the group + 1 (0: empty); both on one cache line, so a probe is one miss."""
    var old = tab^
    tab = List[Int64](length=2 * cap, fill=0)
    var op = _lp(Int(old.unsafe_ptr()))
    var np = _lp(Int(tab.unsafe_ptr()))
    for i in range(old_cap):
        var e = op[2 * i + 1]
        if e != 0:
            var s = _slot(_mix(op[2 * i], Int32(e - 1)), cap)
            while np[2 * s + 1] != 0:
                s += 1
                if s == cap:
                    s = 0
            np[2 * s] = op[2 * i]
            np[2 * s + 1] = e
    _ = old^


def _dist_add(mut tab: List[Int64], mut cap: Int, used: Int, sv: LP, sg: IP, n: Int, min_cap: Int) -> Int:
    """Insert n (value, group) pairs into one partition's open-addressing table of `cap` slots (the first 2 * cap
    words of `tab`; linear probing). It doubles when 4/5 full, after being made at least `min_cap` slots up front,
    which a sketch's estimate makes the usual size. Returns the number of distinct pairs it holds. The slots of the
    next PREFETCH_AHEAD pairs are prefetched: with millions of distinct pairs every probe is a cache miss, and
    overlapping them is most of the speed."""
    var u = used
    if cap * 6 < min_cap * 5:                    # well under the size the sketch asks for: at least 1.5 times, so a scan whose
        var target = max(min_cap, cap * 3 // 2)   # every row group brings new values rehashes O(log) times, not once a group
        _dist_grow(tab, cap, target)
        cap = target
    var tp = _lp(Int(tab.unsafe_ptr()))
    for i in range(n):
        if (u + 1) * 5 > cap * 4:
            var bigger = max(cap * 2, 1024)
            _dist_grow(tab, cap, bigger)
            cap = bigger
            tp = _lp(Int(tab.unsafe_ptr()))
        if i + PREFETCH_AHEAD < n:
            var hs = _slot(_mix(sv[i + PREFETCH_AHEAD], sg[i + PREFETCH_AHEAD]), cap)
            prefetch[PrefetchOptions().for_write().high_locality().to_data_cache()](tp + 2 * hs)
        var v = sv[i]
        var g = sg[i]
        var s = _slot(_mix(v, g), cap)
        var tag = Int64(g) + 1
        while True:
            var e = tp[2 * s + 1]
            if e == 0:
                tp[2 * s] = v
                tp[2 * s + 1] = tag
                u += 1
                break
            if e == tag and tp[2 * s] == v:
                break
            s += 1
            if s == cap:
                s = 0
    return u


def _tally(tab: LP, slots: Int, dense: IP, atomic: LP, use_dense: Bool):
    """Add one to the counter of the group of every pair in the table (a private counter per partition, or an atomic one)."""
    for i in range(slots):
        var e = tab[2 * i + 1]
        if e != 0:
            if use_dense:
                dense[Int(e) - 1] += 1
            else:
                _ = Atomic[Scalar[DType.int64]].fetch_add(atomic + (Int(e) - 1), 1)


struct DistinctState(Movable):
    """The distinct (group, value) pairs seen so far, in `parts` hash partitions (a power of two).

    The first window's pairs are only partitioned and kept (`pending`), not inserted: a query that fits one window
    (most tables) counts them with a table per partition that lives for one task, never holding more than a few
    partitions' tables at once. From a second window on, the pairs go into persistent tables, sized from a
    HyperLogLog sketch per partition (of everything seen, so a window that repeats values does not grow them)."""
    var parts: Int
    var tabs: List[List[Int64]]        # per partition: 2 words a slot, (value, group + 1), group + 1 = 0 for an empty slot
    var used: List[Int]
    var hll: List[UInt8]               # parts x HLL_M
    var pending: Scattered
    var has_pending: Bool
    var tabled: Bool                   # the persistent tables hold the pairs seen so far

    def __init__(out self, parts: Int):
        self.parts = parts
        self.tabs = List[List[Int64]]()
        self.used = List[Int]()
        for _ in range(parts):
            self.tabs.append(List[Int64]())
            self.used.append(0)
        self.hll = List[UInt8](length=parts * HLL_M, fill=0)
        self.pending = Scattered()
        self.has_pending = False
        self.tabled = False

    def absorb(mut self, b: Blocks, threads: Int):
        var sc = _scatter(b, self.parts, True, 0, threads, True)
        var tp = len(sc.hll) // (self.parts * HLL_M)
        for t in range(tp):
            for i in range(self.parts * HLL_M):
                self.hll[i] = max(self.hll[i], sc.hll[t * self.parts * HLL_M + i])
        if not self.tabled and not self.has_pending:
            self.pending = sc^
            self.has_pending = True
            return
        if self.has_pending:
            _insert(self.tabs, self.used, self.hll, self.pending, self.parts, threads)
            self.pending = Scattered()
            self.has_pending = False
        self.tabled = True
        _insert(self.tabs, self.used, self.hll, sc, self.parts, threads)

    def counts(mut self, groups: Int, threads: Int) -> List[Int]:
        """The number of distinct values of each group."""
        var P = self.parts
        var out = List[Int](length=groups, fill=0)
        if not self.tabled and not self.has_pending:
            return out^
        var dense = P * groups <= (1 << 23)           # a counter per (partition, group), summed after; many groups: atomics
        var loc = List[Int32](length=P * groups if dense else 1, fill=0)
        var acc = List[Int64](length=1 if dense else groups, fill=0)
        var loc_at = Int(loc.unsafe_ptr())
        var acc_at = Int(acc.unsafe_ptr())
        if self.has_pending:
            var seg_at = Int(self.pending.seg.unsafe_ptr())
            var sv = Int(self.pending.vals.unsafe_ptr())
            var sg = Int(self.pending.gids.unsafe_ptr())
            var hll_at = Int(self.hll.unsafe_ptr())

            var workers = max(1, min(threads, P))

            def transient(w: Int) {imm}:
                # worker w takes partitions w, w + workers, ... and reuses one table buffer (cleared, not reallocated):
                # a fresh table per partition spends more on page faults and zeroing than on the counting
                var tab = List[Int64]()
                for p in range(w, P, workers):
                    var lo = _np(seg_at)[p]
                    var n = _np(seg_at)[p + 1] - lo
                    if n == 0:
                        continue
                    var cap = _cap_for(_hll_estimate(_bp(hll_at) + p * HLL_M))
                    if len(tab) >= 2 * cap:
                        var tq = _lp(Int(tab.unsafe_ptr()))
                        for i in range(2 * cap):
                            tq[i] = 0
                    else:
                        tab = List[Int64](length=2 * cap, fill=0)
                    _ = _dist_add(tab, cap, 0, _lp(sv) + lo, _ip(sg) + lo, n, 0)
                    _tally(_lp(Int(tab.unsafe_ptr())), cap, _ip(loc_at) + p * groups if dense else _ip(loc_at), _lp(acc_at), dense)    # lifetime: ok (the loop uses `tab` again)

            par_n(transient, workers, workers)
        else:
            var tabs_at = Int(self.tabs.unsafe_ptr())

            def persistent(p: Int) {imm}:
                ref tl = (Pointer[List[Int64], MutUntrackedOrigin](unsafe_from_address=tabs_at) + p)[]
                _tally(_lp(Int(tl.unsafe_ptr())), len(tl) // 2, _ip(loc_at) + p * groups if dense else _ip(loc_at), _lp(acc_at), dense)

            par_n(persistent, P, threads)
        if dense:
            var out_at = Int(out.unsafe_ptr())
            var tp = _tasks(groups, groups, threads)

            def total(t: Int) {imm}:
                var o = _np(out_at)
                var c = _ip(loc_at)
                for g in range(_cut(groups, tp, t), _cut(groups, tp, t + 1)):
                    var s = 0
                    for p in range(P):
                        s += Int(c[p * groups + g])
                    o[g] = s

            par_n(total, tp, tp)
        else:
            for g in range(groups):
                out[g] = Int(acc[g])
        _ = loc^
        _ = acc^
        return out^


def _insert(mut tabs: List[List[Int64]], mut used: List[Int], hll: List[UInt8], sc: Scattered, parts: Int, threads: Int):
    """Insert a window's partitioned pairs into the persistent tables, a task per partition."""
    var tabs_at = Int(tabs.unsafe_ptr())
    var used_at = Int(used.unsafe_ptr())
    var hll_at = Int(hll.unsafe_ptr())
    var seg_at = Int(sc.seg.unsafe_ptr())
    var sv = Int(sc.vals.unsafe_ptr())
    var sg = Int(sc.gids.unsafe_ptr())

    def insert(p: Int) {imm}:
        var lo = _np(seg_at)[p]
        var n = _np(seg_at)[p + 1] - lo
        if n == 0:
            return
        var tp = Pointer[List[Int64], MutUntrackedOrigin](unsafe_from_address=tabs_at) + p
        var up = _np(used_at) + p
        var cap = len(tp[]) // 2
        up[0] = _dist_add(tp[], cap, up[0], _lp(sv) + lo, _ip(sg) + lo, n, _cap_for(_hll_estimate(_bp(hll_at) + p * HLL_M)))

    par_n(insert, parts, threads)


# ── top-k ────────────────────────────────────────────────────────────────────────────────────────────

@always_inline
def heap_offer(h: LP, n: Int, k: Int, v: Int64) -> Int:
    """Offer v to the min-heap of the n (at most k) largest values seen at h; returns the heap's new size."""
    if n < k:
        var i = n
        while i > 0:
            var par = (i - 1) // 2
            if h[par] <= v:
                break
            h[i] = h[par]
            i = par
        h[i] = v
        return n + 1
    if v <= h[0]:
        return n
    var i = 0
    while True:
        var c = 2 * i + 1
        if c >= k:
            break
        if c + 1 < k and h[c + 1] < h[c]:
            c += 1
        if h[c] >= v:
            break
        h[i] = h[c]
        i = c
    h[i] = v
    return n


def _offer_blocks(b: Blocks, k0: Int, k1: Int, heap: LP, size: NP, k: Int, rank: Int):
    """Offer the values of blocks k0 .. k1-1 to their groups' heaps (group g's heap is k words at g * k)."""
    for q in range(k0, k1):
        var gp = _ip(b.g[q])
        var vp = _lp(b.v[q])
        for j in range(b.n[q]):
            var g = Int(gp[j])
            var v = vp[j]
            if rank != 0:
                v = _lp(rank)[Int(v)]
            size[g] = heap_offer(heap + g * k, size[g], k, v)


def _heap_sort_desc(h: LP, n: Int):
    """Sort the min-heap at h of n values into descending order in place (heapsort: pop the minimum to the end)."""
    var m = n
    while m > 1:
        var top = h[0]
        m -= 1
        var x = h[m]
        h[m] = top
        var i = 0
        while True:
            var c = 2 * i + 1
            if c >= m:
                break
            if c + 1 < m and h[c + 1] < h[c]:
                c += 1
            if h[c] >= x:
                break
            h[i] = h[c]
            i = c
        h[i] = x


struct TopKState(Movable):
    """The k largest values of each group so far, a min-heap of up to k values per group."""
    var k: Int
    var groups: Int
    var heap: List[Int64]
    var size: List[Int]

    def __init__(out self, k: Int, groups: Int):
        self.k = k
        self.groups = groups
        self.heap = List[Int64](length=max(k * groups, 1), fill=0)
        self.size = List[Int](length=max(groups, 1), fill=0)

    def absorb(mut self, b: Blocks, rank: Int, threads: Int):
        var k = self.k
        var G = self.groups
        var nb = len(b.n)
        var tp = _tasks(b.total, nb, threads)
        var heap = _lp(Int(self.heap.unsafe_ptr()))
        var size = _np(Int(self.size.unsafe_ptr()))
        if tp == 1:
            _offer_blocks(b, 0, nb, heap, size, k, rank)
        elif k <= PRIVATE_HEAPS // (tp * G):
            # few groups: every task keeps its own heaps over its blocks; they are merged into the state after
            var ph = List[Int64](length=tp * G * k, fill=0)
            var ps = List[Int](length=tp * G, fill=0)
            var ph_at = Int(ph.unsafe_ptr())
            var ps_at = Int(ps.unsafe_ptr())

            def mine(t: Int) {imm}:
                _offer_blocks(b, _cut(nb, tp, t), _cut(nb, tp, t + 1), _lp(ph_at) + t * G * k, _np(ps_at) + t * G, k, rank)

            par_n(mine, tp, tp)
            self.merge_workers(ph_at, ps_at, tp)
            _ = ph^
            _ = ps^
        else:
            # many groups: scatter by group so one task owns each group's heap
            var sc = _scatter(b, tp, False, rank, threads)
            var seg_at = Int(sc.seg.unsafe_ptr())
            var sv = Int(sc.vals.unsafe_ptr())
            var sg = Int(sc.gids.unsafe_ptr())
            var heap_at = Int(heap)
            var size_at = Int(size)

            def own(p: Int) {imm}:
                var vp = _lp(sv)
                var gp = _ip(sg)
                var h = _lp(heap_at)
                var s = _np(size_at)
                for i in range(_np(seg_at)[p], _np(seg_at)[p + 1]):
                    var g = Int(gp[i])
                    s[g] = heap_offer(h + g * k, s[g], k, vp[i])

            par_n(own, tp, tp)
            _ = sc^

    def merge_workers(mut self, heaps: Int, sizes: Int, workers: Int):
        """Offer every value in `workers` private sets of heaps (worker w's heap of group g is k words at
        `heaps + (w * groups + g) * k` words, `sizes + w * groups + g` of them used) to the state's."""
        var G = self.groups
        var k = self.k
        var heap = _lp(Int(self.heap.unsafe_ptr()))
        var size = _np(Int(self.size.unsafe_ptr()))
        for w in range(workers):
            for g in range(G):
                for i in range(_np(sizes)[w * G + g]):
                    size[g] = heap_offer(heap + g * k, size[g], k, _lp(heaps)[(w * G + g) * k + i])

    def finish(mut self, unrank: Int) -> Tuple[List[Int], List[Int]]:
        """(values, offsets): group g's values, largest first, are `values[offsets[g] : offsets[g + 1]]`."""
        var vals = List[Int]()
        var off = List[Int]()
        var heap = _lp(Int(self.heap.unsafe_ptr()))
        for g in range(self.groups):
            off.append(len(vals))
            _heap_sort_desc(heap + g * self.k, self.size[g])
            for i in range(self.size[g]):
                var v = heap[g * self.k + i]
                if unrank != 0:
                    v = _lp(unrank)[Int(v)]
                vals.append(Int(v))
        off.append(len(vals))
        return (vals^, off^)


# ── the states of one query ──────────────────────────────────────────────────────────────────────────

@fieldwise_init
struct HolSpec(Copyable, Movable):
    var agg: Int            # the aggregate's index in the query
    var kind: Int
    var p: Float64
    var k: Int
    var src: Int            # the value source (a distinct value expression) it reads


struct RawState(Movable):
    """Every non-null (group, value) pair of a source, copied from the windows (percentiles need them all)."""
    var cg: List[List[Int32]]
    var cv: List[List[Int64]]

    def __init__(out self):
        self.cg = List[List[Int32]]()
        self.cv = List[List[Int64]]()

    def absorb(mut self, b: Blocks, threads: Int):
        var n = b.total
        if n == 0:
            return
        var ng = List[Int32](unsafe_uninit_length=n)
        var nv = List[Int64](unsafe_uninit_length=n)
        var nb = len(b.n)
        var at = List[Int](length=nb + 1, fill=0)
        for q in range(nb):
            at[q + 1] = at[q] + b.n[q]
        var tp = _tasks(n, nb, threads)
        var ng_at = Int(ng.unsafe_ptr())
        var nv_at = Int(nv.unsafe_ptr())
        var at_at = Int(at.unsafe_ptr())
        var bg = Int(b.g.unsafe_ptr())
        var bv = Int(b.v.unsafe_ptr())
        var bn = Int(b.n.unsafe_ptr())

        def copy(t: Int) {imm}:
            for q in range(_cut(nb, tp, t), _cut(nb, tp, t + 1)):
                var m = _np(bn)[q]
                var o = _np(at_at)[q]
                var sg = _ip(_np(bg)[q])
                var sv = _lp(_np(bv)[q])
                var dg = _ip(ng_at) + o
                var dv = _lp(nv_at) + o
                for j in range(m):
                    dg[j] = sg[j]
                    dv[j] = sv[j]

        par_n(copy, tp, tp)
        _ = at^
        self.cg.append(ng^)
        self.cv.append(nv^)

    def blocks(self) -> Blocks:
        var out = Blocks()
        for c in range(len(self.cg)):
            out.add_flat(Int(self.cg[c].unsafe_ptr()), Int(self.cv[c].unsafe_ptr()), len(self.cg[c]))
        return out^


struct HolResult(Movable):
    """Per holistic aggregate (in `specs` order): its value per group (distinct count, percentile) or its top-k
    values (flat, with offsets); and whether a group has a value at all."""
    var vals: List[List[Int]]
    var top: List[List[Int]]
    var top_off: List[List[Int]]
    var counts: List[List[Int]]      # per aggregate: the group's non-null values (distinct and percentile; top-k: its size)

    def __init__(out self):
        self.vals = List[List[Int]]()
        self.top = List[List[Int]]()
        self.top_off = List[List[Int]]()
        self.counts = List[List[Int]]()


struct HolStates(Movable):
    var groups: Int
    var threads: Int
    var specs: List[HolSpec]
    var ranks: List[List[Int64]]         # per source: code -> its place in the text order (empty: not text)
    var floats: List[Bool]               # per source: its values are float64s, held as `float_key`s (the kernel hook made them)
    var unranks: List[List[Int64]]       # per source: the place -> code
    var scales: List[Int]                # per source: the decimal places of its integer values (a spread aggregate reads them as float(v) / 10^scale)
    var has_distinct: List[Bool]
    var has_raw: List[Bool]
    var has_var: List[Bool]
    var spread: List[VarState]           # per source
    var distinct: List[DistinctState]    # per source
    var raw: List[RawState]              # per source
    var topk: List[TopKState]            # per spec (k = 0 for the others)

    def __init__(out self, groups: Int, var specs: List[HolSpec], var ranks: List[List[Int64]], max_rows: Int = -1,
                 var floats: List[Bool] = List[Bool](), var scales: List[Int] = List[Int]()) raises:
        self.groups = groups
        while len(floats) < len(ranks):
            floats.append(False)
        self.floats = floats^
        while len(scales) < len(ranks):
            scales.append(0)
        self.scales = scales^
        self.threads = core_count()
        var nsrc = len(ranks)
        self.has_distinct = List[Bool](length=nsrc, fill=False)
        self.has_raw = List[Bool](length=nsrc, fill=False)
        self.has_var = List[Bool](length=nsrc, fill=False)
        self.spread = List[VarState]()
        self.distinct = List[DistinctState]()
        self.raw = List[RawState]()
        self.topk = List[TopKState]()
        self.unranks = List[List[Int64]]()
        var parts = 4
        while parts < 4 * self.threads:
            parts *= 2
        for s in range(nsrc):
            var un = List[Int64](length=len(ranks[s]), fill=0)
            for c in range(len(ranks[s])):
                un[Int(ranks[s][c])] = Int64(c)
            self.unranks.append(un^)
        for i in range(len(specs)):
            if specs[i].kind == A_TOPK and specs[i].k >= 1 and max_rows >= 0:
                specs[i].k = min(specs[i].k, max(max_rows, 1))        # no group holds more values than the input has rows
        for spec in specs:
            if spec.kind == A_PCT or spec.kind == A_MEDIAN or spec.kind == A_QEDGE:
                if spec.kind == A_QEDGE and (spec.k < 2 or spec.p < 1.0 or spec.p >= Float64(spec.k)):
                    raise Error("quantile bins: edge " + String(Int(spec.p)) + " of " + String(spec.k) + " bins does not exist")
                if spec.kind == A_PCT and not (spec.p >= 0.0 and spec.p <= 1.0):
                    raise Error("percentile: p must be between 0 and 1, got " + String(spec.p))
                self.has_raw[spec.src] = True
            elif spec.kind == A_DISTINCT:
                self.has_distinct[spec.src] = True
            elif is_spread(spec.kind):
                self.has_var[spec.src] = True
            elif spec.kind == A_TOPK:
                if spec.k < 1:
                    raise Error("top_k: k must be at least 1, got " + String(spec.k))
                if spec.k > MAX_TOPK_WORDS // groups:           # (a product could overflow Int)
                    raise Error("top_k: k=" + String(spec.k) + " for " + String(groups) + " groups would hold more than "
                                + String(MAX_TOPK_WORDS) + " values")
        for s in range(nsrc):
            self.distinct.append(DistinctState(parts if self.has_distinct[s] else 1))
            self.raw.append(RawState())
            self.spread.append(VarState(groups if self.has_var[s] else 0))
        for spec in specs:
            self.topk.append(TopKState(spec.k if spec.kind == A_TOPK else 1, groups if spec.kind == A_TOPK else 1))
        self.specs = specs^
        self.ranks = ranks^

    def rank_of(self, s: Int) -> Int:
        """Address of source s's code -> place table (0: not text)."""
        return Int(self.ranks[s].unsafe_ptr()) if len(self.ranks[s]) > 0 else 0

    def direct_topk(self, s: Int, workers: Int) -> Int:
        """The top_k aggregate (its place among the holistic ones) when it is the only one reading source s and
        `workers` private sets of its heaps fit PRIVATE_HEAPS: the kernel hook then offers the values to per-worker heaps
        itself and writes no pairs. -1 otherwise."""
        var found = -1
        for h in range(len(self.specs)):
            if self.specs[h].src == s:
                if self.specs[h].kind != A_TOPK or found >= 0:
                    return -1
                found = h
        if found >= 0 and self.specs[found].k > PRIVATE_HEAPS // (workers * self.groups):
            return -1
        return found

    def merge_workers(mut self, h: Int, heaps: Int, sizes: Int, workers: Int):
        """Take the per-worker heaps the kernel hook filled for top_k aggregate h."""
        self.topk[h].merge_workers(heaps, sizes, workers)

    def _unrank(self, s: Int) -> Int:
        return Int(self.unranks[s].unsafe_ptr()) if len(self.unranks[s]) > 0 else 0

    def absorb(mut self, s: Int, b: Blocks):
        """Take one window's pairs of source s into every state that reads it."""
        if b.total == 0:
            return
        if self.has_distinct[s]:
            self.distinct[s].absorb(b, self.threads)
        if self.has_raw[s]:
            self.raw[s].absorb(b, self.threads)
        if self.has_var[s]:
            var div = 1.0
            for _ in range(self.scales[s]):
                div *= 10.0
            self.spread[s].absorb(b, self.floats[s], div, self.threads)
        for h in range(len(self.specs)):
            if self.specs[h].kind == A_TOPK and self.specs[h].src == s:
                self.topk[h].absorb(b, self.rank_of(s), self.threads)

    def absorb_rows(mut self, s: Int, gids: List[Int32], vals: List[Int64]):
        """The same for a few single pairs (a frame's changed and appended rows)."""
        var b = Blocks()
        b.add_flat(Int(gids.unsafe_ptr()), Int(vals.unsafe_ptr()), len(gids))
        self.absorb(s, b)
        _ = gids
        _ = vals

    def _entries(self, h: Int) -> Int:
        """How many selections aggregate h asks of its source's raw values: a percentile one, a median of numbers two
        (the lower and the upper middle), a median of text one (the lower middle: words have no mean)."""
        if self.specs[h].kind == A_PCT or self.specs[h].kind == A_QEDGE:
            return 1
        if self.specs[h].kind == A_MEDIAN:
            return 1 if len(self.ranks[self.specs[h].src]) > 0 else 2
        return 0

    def finish(mut self) raises -> HolResult:
        var res = HolResult()
        var G = self.groups
        var distinct_of = List[List[Int]]()
        for s in range(len(self.distinct)):
            distinct_of.append(self.distinct[s].counts(G, self.threads) if self.has_distinct[s] else List[Int]())
        var pct_of = List[Percentiles]()           # per source
        for s in range(len(self.distinct)):
            var ps = List[Float64]()
            var bins = List[Int]()                  # per p: the number of bins when it is the k of a bin edge, else 0
            for h in range(len(self.specs)):
                if self.specs[h].src == s and self._entries(h) > 0:
                    var edge = self.specs[h].kind == A_QEDGE
                    ps.append(self.specs[h].p if self.specs[h].kind == A_PCT or edge else 0.5)
                    bins.append(self.specs[h].k if edge else 0)
                    if self._entries(h) == 2:
                        ps.append(UPPER_MIDDLE)
                        bins.append(0)
            if self.has_raw[s]:
                pct_of.append(_percentiles(self.raw[s].blocks(), G, ps, bins, self.rank_of(s), self._unrank(s), self.threads))
            else:
                pct_of.append(Percentiles())
        for h in range(len(self.specs)):
            ref spec = self.specs[h]
            if spec.kind == A_DISTINCT:
                res.vals.append(distinct_of[spec.src].copy())
                res.counts.append(distinct_of[spec.src].copy())
                res.top.append(List[Int]())
                res.top_off.append(List[Int]())
            elif is_spread(spec.kind):
                # a float64 per group (its bits); the count is the group's non-null values (the sample variants answer NULL below two)
                var m = List[Int](length=G, fill=0)
                for g in range(G):
                    var n = self.spread[spec.src].n[g]
                    if n >= spread_needs(spec.kind):
                        m[g] = Int(Int64(spread_value(spec.kind, n, self.spread[spec.src].m2[g]).to_bits()))
                res.vals.append(m^)
                res.counts.append(self.spread[spec.src].n.copy())
                res.top.append(List[Int]())
                res.top_off.append(List[Int]())
            elif spec.kind == A_PCT or spec.kind == A_MEDIAN or spec.kind == A_QEDGE:
                var q = 0                          # this aggregate's first selection among its source's
                for h2 in range(h):
                    if self.specs[h2].src == spec.src:
                        q += self._entries(h2)
                if self._entries(h) == 2 and self.floats[spec.src]:
                    # a median of doubles: the middle value, or the interpolation of the two middle ones (their float bits are the answer)
                    var m = List[Int](length=G, fill=0)
                    for g in range(G):
                        var n = pct_of[spec.src].counts[g]
                        if n == 0:
                            continue
                        var lo = Float64(from_bits=UInt64(float_unkey(Int64(pct_of[spec.src].vals[q][g]))))
                        var hi = Float64(from_bits=UInt64(float_unkey(Int64(pct_of[spec.src].vals[q + 1][g]))))
                        m[g] = Int(Int64(median_of_floats(lo, hi, n).to_bits()))
                    res.vals.append(m^)
                elif self._entries(h) == 2:
                    # mean of the two middle values at scale + 1: 10 * (a + b) / 2 = 5 * (a + b), exact
                    var m = List[Int](length=G, fill=0)
                    for g in range(G):
                        var lo = pct_of[spec.src].vals[q][g]
                        var hi = pct_of[spec.src].vals[q + 1][g]
                        var sum = Int64(lo) + Int64(hi)
                        if ((Int64(lo) ^ sum) & (Int64(hi) ^ sum)) < 0 or sum > Int64.MAX // 5 or sum < -(Int64.MAX // 5):   # (Int64.MIN // 5 floors: one step too far)
                            raise Error("median: the middle of group " + String(g) + " is too large for its mean at one more decimal place in int64 (about 9.2e17 at the column's scale)")
                        m[g] = Int(sum * 5)
                    res.vals.append(m^)
                elif self.floats[spec.src]:
                    var m = pct_of[spec.src].vals[q].copy()
                    for g in range(G):
                        m[g] = Int(float_unkey(Int64(m[g])))
                    res.vals.append(m^)
                else:
                    res.vals.append(pct_of[spec.src].vals[q].copy())
                res.counts.append(pct_of[spec.src].counts.copy())
                res.top.append(List[Int]())
                res.top_off.append(List[Int]())
            else:
                var t = self.topk[h].finish(self._unrank(spec.src))
                res.counts.append(self.topk[h].size.copy())
                res.vals.append(List[Int](length=G, fill=0))
                if self.floats[spec.src]:
                    var tv = t[0].copy()
                    for i in range(len(tv)):
                        tv[i] = Int(float_unkey(Int64(tv[i])))
                    res.top.append(tv^)
                else:
                    res.top.append(t[0].copy())
                res.top_off.append(t[1].copy())
        return res^
