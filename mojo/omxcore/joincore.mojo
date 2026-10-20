"""The inner equi-join of two tables, as an engine feature (F-153): key normalization, the hash join kernel's calls, the pairs, the gather.

Everything here works on `SharedTable`s and names no query language and no application: the SQL layer (`join.mojo`, `sqlexec.mojo`), the
Query builder (`Query.join`) and any other caller turn their input into two tables, the columns that are the keys and the columns to carry, and
call these.

- `normalize_keys` turns each key column into one int64 per row: integers as they are, DECIMAL aligned to the larger scale (INT is scale 0; a value
  that overflows when aligned cannot equal anything on the other side, so it never matches), DATE and BOOL as stored, TEXT through a code
  translation from the probe side's dictionary to the build side's (byte-exact: join equality is byte equality; a word the build side lacks never
  matches), FLOAT by `float_key` (-0.0 equals 0.0 and every NaN equals every NaN, DuckDB's rule), NULL never matches; several keys are packed into one
  int64 when their ranges multiply to fewer than 2^62, otherwise hashed to one and the candidate pairs are verified against the key columns. A single
  key that needs no change is the column's own values: nothing is copied.
- `join_pairs` probes the kernel's map (`kernels/hashjoin_cpu.mojo`) and returns every matching (probe row, build row) pair in probe-row order, or
  refuses past a budget before allocating.
- `gather_join` makes the joined table from the pairs: the carried columns of each side at its rows, gathered in parallel.
"""
from std.memory.alloc import unsafe_alloc
from .table import Table, SharedTable, Column, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .holistic import float_key
from .kernels._fpenv import par_n, core_count
from .kernels.hashjoin_cpu import JoinMap, hj_build, hj_probe_pairs, hj_probe_select, hj_release, E_BUDGET, PM_SEMI, PM_ANTI, PM_ANTI_NA

comptime JOIN_BUDGET_BYTES = 3 << 30        # the joined rows a join may materialize (8 bytes a value); past it a readable error (spill is later)


def _pow10(k: Int) raises -> Int64:
    if k < 0 or k > 18:
        raise Error("a power of ten outside 10^0 .. 10^18")
    var v: Int64 = 1
    for _ in range(k):
        v *= 10
    return v


def commas(n: Int) -> String:
    var s = String(n if n >= 0 else -n)
    var out = String()
    var k = s.byte_length()
    for i in range(k):
        if i > 0 and (k - i) % 3 == 0:
            out += ","
        out += String(s[byte=i:i + 1])
    return ("-" if n < 0 else "") + out


def _unique_name(t: Table, name: String) -> String:
    if t.index(name) < 0:
        return name
    var i = 1
    while t.index(name + "_" + String(i)) >= 0:
        i += 1
    return name + "_" + String(i)


# ── keys ─────────────────────────────────────────────────────────────────────────────────────────────

struct JoinKeys(Movable):
    var bkeys: List[Int64]           # the build side's key of each row, the probe side's ...
    var bok: List[UInt8]             # ... and whether the row can match at all
    var pkeys: List[Int64]
    var pok: List[UInt8]
    var verify: Bool                 # the keys are hashes of tuples: candidate pairs are checked against the columns
    var bcomp: List[List[Int64]]     # verify: each key column's normalized values
    var pcomp: List[List[Int64]]
    var layout: String               # for the notes: how several keys became one
    var bk_addr: Int                 # the arrays the kernel reads: the lists above, or a column's own values when no key had to change
    var bo_addr: Int                 # (a `ok` address of 0: every row may match)
    var pk_addr: Int
    var po_addr: Int

    def __init__(out self):
        self.bk_addr = 0
        self.bo_addr = 0
        self.pk_addr = 0
        self.po_addr = 0
        self.bkeys = List[Int64]()
        self.bok = List[UInt8]()
        self.pkeys = List[Int64]()
        self.pok = List[UInt8]()
        self.verify = False
        self.bcomp = List[List[Int64]]()
        self.pcomp = List[List[Int64]]()
        self.layout = String()


def _mix(h: UInt64) -> UInt64:
    var z = h
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _valid_bits(t: SharedTable, c: Int) -> Int:
    return t._table[].cols[c].validity_address()


def _column_keys(t: SharedTable, c: Int, kind: Int, mul: Int64, translate: List[Int32], translating: Bool, mut vals: List[Int64], mut ok: List[UInt8]):
    """One key column as int64 values and `ok` flags (see the module doc)."""
    var n = t.rows()
    vals = List[Int64](length=n, fill=0)
    ok = List[UInt8](length=n, fill=1)
    var src = t.values(c).unsafe_ptr()
    var valid = _valid_bits(t, c)
    var vp = vals.unsafe_ptr()
    var op = ok.unsafe_ptr()
    for i in range(n):
        if valid != 0 and (Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=valid)[i >> 3] >> UInt8(i & 7)) & 1 == 0:
            op[i] = 0
            continue
        var v = src[i]
        if kind == FLOAT.value:
            vp[i] = float_key(v)
        elif kind == TEXT.value:
            if translating:
                var tr = Int(translate[Int(v)])
                if tr < 0:
                    op[i] = 0
                    continue
                vp[i] = Int64(tr)
            else:
                vp[i] = v
        elif mul != 1:
            var r = v * mul
            if (v != 0 and r // v != mul) or (v == Int64.MIN):
                op[i] = 0
                continue
            vp[i] = r
        else:
            vp[i] = v


def column_ndv(t: SharedTable, c: Int, threads: Int = 0) -> Int:
    """The number of distinct non-NULL values of a column (the join planner's statistic), from a sample of 16,384 distinct rows chosen at random (every row of
    a smaller table: exact). Rows are drawn at random positions, not at a stride: a column that is clustered (sorted keys, each repeated on neighbouring
    rows) would otherwise show no repeats. The sample's repeated values give the estimate: m rows of d equally frequent values collide in about m(m-1) / (2d)
    pairs, so d is about m(m-1) / (2 x pairs) (a key that never collides is taken as unique; the estimate is at least the distinct values seen and at most
    the rows). An order of magnitude is what the join order needs."""
    var n = t.rows()
    if n == 0:
        return 0
    var src = t.values(c).unsafe_ptr()
    var valid = _valid_bits(t, c)
    var bits = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=valid)
    var target = min(n, 16384)
    var slots = 1 << 17
    var keys = List[Int64](length=slots, fill=0)
    var counts = List[Int32](length=slots, fill=0)
    var seen_rows = List[Int32](length=slots, fill=-1)              # the rows drawn so far (each is drawn once)
    var distinct = 0
    var pairs = 0
    var m = 0
    var drawn = 0
    var seed = UInt64(0x9E3779B97F4A7C15)
    while drawn < target:
        var i = drawn
        if target < n:
            seed = _mix(seed)
            i = Int(seed % UInt64(n))
            var rh = Int(_mix(UInt64(i)) >> 47) & (slots - 1)
            var dup = False
            while seen_rows[rh] >= 0:
                if Int(seen_rows[rh]) == i:
                    dup = True
                    break
                rh = (rh + 1) & (slots - 1)
            if dup:
                continue
            seen_rows[rh] = Int32(i)
        drawn += 1
        if valid == 0 or (bits[i >> 3] >> UInt8(i & 7)) & 1 != 0:
            var v = src[i]
            var h = Int(_mix(UInt64(v)) >> 47) & (slots - 1)
            while counts[h] != 0 and keys[h] != v:
                h = (h + 1) & (slots - 1)
            if counts[h] == 0:
                keys[h] = v
                distinct += 1
            else:
                pairs += Int(counts[h])                            # a new row of a value seen `counts[h]` times collides with each of them
            counts[h] += 1
            m += 1
    if m == 0:
        return 0
    if target == n:
        return distinct                                            # every row was looked at: exact
    var est = n
    if pairs > 0:
        est = Int(Float64(m) * Float64(m - 1) / (2.0 * Float64(pairs)))
    return max(min(est, n), distinct)


def unmatchable_not_null(mut keys: JoinKeys, lo: Int64, hi: Int64, probe: SharedTable, c: Int, n: Int):
    """NOT IN: a probe row whose value can equal no build key (a TEXT word the build side lacks, a decimal that overflows when aligned) is not a NULL key,
    and a NULL key is the only one NOT IN drops. `ok` cannot tell the two apart, so such a row becomes an ordinary key outside the build map's range
    (it is present nowhere). Only keys the normalization built itself are touched (a column's own values are never changed, and have no such rows)."""
    if keys.po_addr == 0 or len(keys.pkeys) != n:
        return
    var sentinel = hi + 1 if hi < Int64.MAX else lo - 1
    var valid = _valid_bits(probe, c)
    var po = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=keys.po_addr)
    var bits = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=valid)
    for i in range(n):
        if po[i] == 0 and (valid == 0 or (bits[i >> 3] >> UInt8(i & 7)) & 1 != 0):
            po[i] = 1
            keys.pkeys[i] = sentinel


def _ok_bytes(t: SharedTable, c: Int, mut ok: List[UInt8]) -> Int:
    """The `ok` bytes of a column's validity (0 when it has no NULL: every row may match)."""
    if t.null_count(c) == 0:
        return 0
    var n = t.rows()
    ok = List[UInt8](unsafe_uninit_length=n)
    var valid = _valid_bits(t, c)
    var bits = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=valid)
    for i in range(n):
        ok[i] = (bits[i >> 3] >> UInt8(i & 7)) & 1
    return Int(ok.unsafe_ptr())


def _pack_plain(bt: SharedTable, pt: SharedTable, bcols: List[Int], pcols: List[Int], mut keys: JoinKeys, threads: Int) raises -> Bool:
    """Several key columns that need no change (integers or decimals of one scale, dates, bools) and have no NULL: packed into one int64 straight
    from the columns, in parallel slices (the build side's ranges, then both sides' keys). False when the columns are not that plain or the ranges
    do not fit 2^62 values (the general path takes over)."""
    var nk = len(bcols)
    for j in range(nk):
        var kind = bt.kind(bcols[j]).value
        var plain = kind == DATE.value or kind == BOOL.value or ((kind == INT.value or kind == DECIMAL.value) and bt.scale(bcols[j]) == pt.scale(pcols[j]))
        if not plain or bt.null_count(bcols[j]) != 0 or pt.null_count(pcols[j]) != 0:
            return False
    var nb = bt.rows()
    var np = pt.rows()
    var bsrc = List[Int](length=nk, fill=0)
    var psrc = List[Int](length=nk, fill=0)
    for j in range(nk):
        bsrc[j] = Int(bt.values(bcols[j]).unsafe_ptr())
        psrc[j] = Int(pt.values(pcols[j]).unsafe_ptr())
    var T = max(1, min(threads if threads > 0 else core_count(), nb // 65536 + 1))
    var mins = List[Int64](length=T * nk, fill=Int64.MAX)
    var maxs = List[Int64](length=T * nk, fill=Int64.MIN)
    var mnp = mins.unsafe_ptr()
    var mxp = maxs.unsafe_ptr()

    def ranges(t: Int) {imm}:
        var a = nb * t // T
        var b = nb * (t + 1) // T
        for j in range(nk):
            var src = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=bsrc[j])
            var lo = Int64.MAX
            var hi = Int64.MIN
            for i in range(a, b):
                var v = src[i]
                lo = min(lo, v)
                hi = max(hi, v)
            mnp[t * nk + j] = lo
            mxp[t * nk + j] = hi

    par_n(ranges, T, T)
    var los = List[Int64]()
    var spans = List[UInt64]()
    var strides = List[UInt64]()
    var product = UInt64(1)
    for j in range(nk):
        var lo = Int64.MAX
        var hi = Int64.MIN
        for t in range(T):
            lo = min(lo, mins[t * nk + j])
            hi = max(hi, maxs[t * nk + j])
        if lo > hi:
            lo = 0
            hi = 0
        var span = UInt64(hi) - UInt64(lo) + 1
        if span == 0 or product > (UInt64(1) << 62) // span:
            return False
        los.append(lo)
        spans.append(span)
        strides.append(product)
        product *= span
    keys.bkeys = List[Int64](unsafe_uninit_length=nb)
    keys.pkeys = List[Int64](unsafe_uninit_length=np)
    keys.pok = List[UInt8](unsafe_uninit_length=np)
    var bk = keys.bkeys.unsafe_ptr()
    var pk = keys.pkeys.unsafe_ptr()
    var po = keys.pok.unsafe_ptr()
    var TB = max(1, min(threads if threads > 0 else core_count(), nb // 65536 + 1))
    var TP = max(1, min(threads if threads > 0 else core_count(), np // 65536 + 1))

    def build_keys(t: Int) {imm}:
        var a = nb * t // TB
        var b = nb * (t + 1) // TB
        for i in range(a, b):
            bk[i] = 0
        for j in range(nk):
            var src = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=bsrc[j])
            var lo = los[j]
            var st = strides[j]
            for i in range(a, b):
                bk[i] = Int64(UInt64(bk[i]) + UInt64(src[i] - lo) * st)

    def probe_keys(t: Int) {imm}:
        var a = np * t // TP
        var b = np * (t + 1) // TP
        for i in range(a, b):
            pk[i] = 0
            po[i] = 1
        for j in range(nk):
            var src = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=psrc[j])
            var lo = los[j]
            var sp = spans[j]
            var st = strides[j]
            for i in range(a, b):
                var d = UInt64(src[i] - lo)
                if d >= sp:
                    po[i] = 0                                         # outside the build side's range: it cannot match
                else:
                    pk[i] = Int64(UInt64(pk[i]) + d * st)

    par_n(build_keys, TB, TB)
    par_n(probe_keys, TP, TP)
    keys.layout = String(nk) + " keys packed into one int64 (" + String(product) + " combinations)"
    keys.bk_addr = Int(keys.bkeys.unsafe_ptr())
    keys.pk_addr = Int(keys.pkeys.unsafe_ptr())
    keys.bo_addr = 0
    keys.po_addr = Int(keys.pok.unsafe_ptr())
    return True


def normalize_keys(bt: SharedTable, pt: SharedTable, bcols: List[Int], pcols: List[Int], mut keys: JoinKeys, threads: Int = 0) raises:
    """The keys of the build side `bt` and the probe side `pt`, columns `bcols[j]` and `pcols[j]` being compared pairwise."""
    var nk = len(bcols)
    if bt.rows() >= (1 << 31) or pt.rows() >= (1 << 31):
        raise Error("join: a side of 2^31 rows or more is not supported (the join's row numbers are 32-bit); filter or split the table first")
    if nk == 1:
        var kind = bt.kind(bcols[0]).value
        var bs = bt.scale(bcols[0])
        var ps = pt.scale(pcols[0])
        var numeric = kind == INT.value or kind == DECIMAL.value
        if (numeric and bs == ps) or kind == DATE.value or kind == BOOL.value:
            # the key is the column itself: no pass over the rows, only the NULL flags (when there are any)
            keys.bk_addr = Int(bt.values(bcols[0]).unsafe_ptr())
            keys.pk_addr = Int(pt.values(pcols[0]).unsafe_ptr())
            keys.bo_addr = _ok_bytes(bt, bcols[0], keys.bok)
            keys.po_addr = _ok_bytes(pt, pcols[0], keys.pok)
            keys.layout = "one key (the column itself)"
            return
    if nk > 1 and _pack_plain(bt, pt, bcols, pcols, keys, threads):
        return
    var bvals = List[List[Int64]]()
    var boks = List[List[UInt8]]()
    var pvals = List[List[Int64]]()
    var poks = List[List[UInt8]]()
    for j in range(nk):
        var kind = bt.kind(bcols[j]).value
        var bs = bt.scale(bcols[j])
        var ps = pt.scale(pcols[j])
        var bmul = Int64(1)
        var pmul = Int64(1)
        var numeric = kind == INT.value or kind == DECIMAL.value
        if numeric and bs != ps:
            if bs < ps:
                bmul = _pow10(ps - bs)
            else:
                pmul = _pow10(bs - ps)
        var translate = List[Int32]()
        if kind == TEXT.value:                                       # the probe side's code -> the build side's, once per word
            var words = pt.dictionary(pcols[j])
            for w in range(len(words)):
                translate.append(Int32(bt.code_of(bcols[j], String(words[w]))))
        var bv = List[Int64]()
        var bo = List[UInt8]()
        var pv = List[Int64]()
        var po = List[UInt8]()
        _column_keys(bt, bcols[j], kind, bmul, List[Int32](), False, bv, bo)
        _column_keys(pt, pcols[j], kind, pmul, translate, kind == TEXT.value, pv, po)
        bvals.append(bv^)
        boks.append(bo^)
        pvals.append(pv^)
        poks.append(po^)
    var nb = bt.rows()
    var np = pt.rows()
    keys.bok = List[UInt8](length=nb, fill=1)
    keys.pok = List[UInt8](length=np, fill=1)
    for j in range(nk):
        for i in range(nb):
            if boks[j][i] == 0:
                keys.bok[i] = 0
        for i in range(np):
            if poks[j][i] == 0:
                keys.pok[i] = 0
    if nk == 1:
        keys.bkeys = bvals[0].copy()
        keys.pkeys = pvals[0].copy()
        keys.layout = "one key"
        keys.bk_addr = Int(keys.bkeys.unsafe_ptr())
        keys.pk_addr = Int(keys.pkeys.unsafe_ptr())
        keys.bo_addr = Int(keys.bok.unsafe_ptr())
        keys.po_addr = Int(keys.pok.unsafe_ptr())
        return
    # several keys: packed into one int64 when the build side's ranges multiply to fewer than 2^62 values
    var los = List[Int64]()
    var spans = List[UInt64]()
    var fits = True
    var product = UInt64(1)
    for j in range(nk):
        var lo = Int64.MAX
        var hi = Int64.MIN
        for i in range(nb):
            if keys.bok[i] != 0:
                lo = min(lo, bvals[j][i])
                hi = max(hi, bvals[j][i])
        if lo > hi:
            lo = 0
            hi = 0
        var span = UInt64(hi) - UInt64(lo) + 1
        los.append(lo)
        spans.append(span)
        if span == 0 or product > (UInt64(1) << 62) // span:
            fits = False
        else:
            product *= span
    keys.bkeys = List[Int64](length=nb, fill=0)
    keys.pkeys = List[Int64](length=np, fill=0)
    if fits:
        keys.layout = String(nk) + " keys packed into one int64 (" + String(product) + " combinations)"
        var stride = UInt64(1)
        for j in range(nk):
            for i in range(nb):
                keys.bkeys[i] = Int64(UInt64(keys.bkeys[i]) + UInt64(bvals[j][i] - los[j]) * stride)
            for i in range(np):
                if keys.pok[i] != 0 and (pvals[j][i] < los[j] or UInt64(pvals[j][i] - los[j]) >= spans[j]):
                    keys.pok[i] = 0                                  # outside the build side's range: it cannot match
                else:
                    keys.pkeys[i] = Int64(UInt64(keys.pkeys[i]) + UInt64(pvals[j][i] - los[j]) * stride)
            stride *= spans[j]
        keys.bk_addr = Int(keys.bkeys.unsafe_ptr())
        keys.pk_addr = Int(keys.pkeys.unsafe_ptr())
        keys.bo_addr = Int(keys.bok.unsafe_ptr())
        keys.po_addr = Int(keys.pok.unsafe_ptr())
        return
    keys.verify = True
    keys.layout = String(nk) + " keys hashed to one int64 (their ranges are too wide to pack), pairs verified against the columns"
    for i in range(nb):
        var h = UInt64(0x9E3779B97F4A7C15)
        for j in range(nk):
            h = _mix(h ^ UInt64(bvals[j][i]))
        keys.bkeys[i] = Int64(h)
    for i in range(np):
        var h = UInt64(0x9E3779B97F4A7C15)
        for j in range(nk):
            h = _mix(h ^ UInt64(pvals[j][i]))
        keys.pkeys[i] = Int64(h)
    keys.bcomp = bvals^
    keys.pcomp = pvals^
    keys.bk_addr = Int(keys.bkeys.unsafe_ptr())
    keys.pk_addr = Int(keys.pkeys.unsafe_ptr())
    keys.bo_addr = Int(keys.bok.unsafe_ptr())
    keys.po_addr = Int(keys.pok.unsafe_ptr())


# ── the keys of a probe side that arrives in chunks (F-158) ────────────────────────────────────────────

struct KeyPlan(Movable):
    """How a probe chunk's key columns become the int64 keys the build side's map holds: planned once from the build side and the probe's schema, applied to
    each chunk (a scan's row group, a frame's shared rows, its changed rows). `hashed`: the keys are hashes of tuples (a join the chunked path leaves to the
    materialized one: its pairs must be verified against the columns)."""
    var nk: Int
    var own: Bool                    # one key that is the column itself
    var kinds: List[Int]
    var pmuls: List[Int64]           # the probe's decimal scale brought to the build side's
    var translates: List[List[Int32]]   # a TEXT key: the probe's code -> the build side's (-1: the word has no partner)
    var packed: Bool
    var hashed: Bool
    var los: List[Int64]
    var spans: List[UInt64]
    var layout: String

    def __init__(out self):
        self.nk = 0
        self.own = False
        self.kinds = List[Int]()
        self.pmuls = List[Int64]()
        self.translates = List[List[Int32]]()
        self.packed = False
        self.hashed = False
        self.los = List[Int64]()
        self.spans = List[UInt64]()
        self.layout = String()


def make_key_plan(bt: SharedTable, ps: SharedTable, bcols: List[Int], pcols: List[Int], pwords: List[List[String]], aliases: List[List[Int]],
                  mut keys: JoinKeys) raises -> KeyPlan:
    """The plan, and the build side's keys in `keys` (`bk_addr`, `bo_addr`). `ps` holds the probe's kinds, scales and (when `pwords[j]` is empty) the words
    of its TEXT key columns by code; `pwords[j]` are a column's words by code when a frame's session words go beyond the shared table's, `aliases[j]` its merged
    codes (from, to, from, to ...)."""
    var plan = KeyPlan()
    var nk = len(bcols)
    plan.nk = nk
    if bt.rows() >= (1 << 31):
        raise Error("join: a side of 2^31 rows or more is not supported (the join's row numbers are 32-bit); filter or split the table first")
    if nk == 1:
        var kind = bt.kind(bcols[0]).value
        var numeric = kind == INT.value or kind == DECIMAL.value
        if (numeric and bt.scale(bcols[0]) == ps.scale(pcols[0])) or kind == DATE.value or kind == BOOL.value:
            plan.own = True
            plan.layout = "one key (the column itself)"
            keys.bk_addr = Int(bt.values(bcols[0]).unsafe_ptr())
            keys.bo_addr = _ok_bytes(bt, bcols[0], keys.bok)
            plan.kinds.append(kind)
            return plan^
    var bvals = List[List[Int64]]()
    keys.bok = List[UInt8](length=bt.rows(), fill=1)
    for j in range(nk):
        var kind = bt.kind(bcols[j]).value
        var bs = bt.scale(bcols[j])
        var psc = ps.scale(pcols[j])
        var bmul = Int64(1)
        var pmul = Int64(1)
        if (kind == INT.value or kind == DECIMAL.value) and bs != psc:
            if bs < psc:
                bmul = _pow10(psc - bs)
            else:
                pmul = _pow10(bs - psc)
        var translate = List[Int32]()
        if kind == TEXT.value:
            var words = List[String]()
            if len(pwords[j]) > 0:
                words = pwords[j].copy()
            else:
                var dw = ps.dictionary(pcols[j])
                for w in range(len(dw)):
                    words.append(String(dw[w]))
            for w in range(len(words)):
                var target = w
                var a = 0
                while a + 1 < len(aliases[j]):                       # a merged code reads as the code it was merged into
                    if aliases[j][a] == w:
                        target = aliases[j][a + 1]
                    a += 2
                translate.append(Int32(bt.code_of(bcols[j], words[target]) if target < len(words) else -1))
        var bv = List[Int64]()
        var bo = List[UInt8]()
        _column_keys(bt, bcols[j], kind, bmul, List[Int32](), False, bv, bo)
        bvals.append(bv^)
        for i in range(bt.rows()):
            if bo[i] == 0:
                keys.bok[i] = 0
        plan.kinds.append(kind)
        plan.pmuls.append(pmul)
        plan.translates.append(translate^)
    var nb = bt.rows()
    if nk == 1:
        keys.bkeys = bvals[0].copy()
        plan.layout = "one key"
    else:
        var product = UInt64(1)
        var fits = True
        for j in range(nk):
            var lo = Int64.MAX
            var hi = Int64.MIN
            for i in range(nb):
                if keys.bok[i] != 0:
                    lo = min(lo, bvals[j][i])
                    hi = max(hi, bvals[j][i])
            if lo > hi:
                lo = 0
                hi = 0
            var span = UInt64(hi) - UInt64(lo) + 1
            plan.los.append(lo)
            plan.spans.append(span)
            if span == 0 or product > (UInt64(1) << 62) // span:
                fits = False
            else:
                product *= span
        if not fits:
            plan.hashed = True
            plan.layout = String(nk) + " keys hashed to one int64 (their ranges are too wide to pack)"
            return plan^
        plan.packed = True
        plan.layout = String(nk) + " keys packed into one int64 (" + String(product) + " combinations)"
        keys.bkeys = List[Int64](length=nb, fill=0)
        var stride = UInt64(1)
        for j in range(nk):
            for i in range(nb):
                keys.bkeys[i] = Int64(UInt64(keys.bkeys[i]) + UInt64(bvals[j][i] - plan.los[j]) * stride)
            stride *= plan.spans[j]
    keys.bk_addr = Int(keys.bkeys.unsafe_ptr())
    keys.bo_addr = Int(keys.bok.unsafe_ptr())
    return plan^


def probe_chunk_keys(plan: KeyPlan, chunk: SharedTable, pcols: List[Int], mut keys: JoinKeys) raises:
    """The probe keys of one chunk (`pcols` are its key columns' positions in the chunk): into `keys.pk_addr` / `keys.po_addr` (the lists it owns live as long as
    `keys`)."""
    var n = chunk.rows()
    if plan.own:
        keys.pk_addr = Int(chunk.values(pcols[0]).unsafe_ptr())
        keys.po_addr = _ok_bytes(chunk, pcols[0], keys.pok)
        return
    keys.pok = List[UInt8](length=n, fill=1)
    var pvals = List[List[Int64]]()
    for j in range(plan.nk):
        var kind = plan.kinds[j]
        var pv = List[Int64](length=n, fill=0)
        var src = chunk.values(pcols[j]).unsafe_ptr()
        var valid = _valid_bits(chunk, pcols[j])
        var bits = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=valid)
        for i in range(n):
            if valid != 0 and (bits[i >> 3] >> UInt8(i & 7)) & 1 == 0:
                keys.pok[i] = 0
                continue
            var v = src[i]
            if kind == FLOAT.value:
                pv[i] = float_key(v)
            elif kind == TEXT.value:
                if v < 0 or Int(v) >= len(plan.translates[j]) or plan.translates[j][Int(v)] < 0:
                    keys.pok[i] = 0                                  # a word the build side lacks never matches
                    continue
                pv[i] = Int64(plan.translates[j][Int(v)])
            elif plan.pmuls[j] != 1:
                var r = v * plan.pmuls[j]
                if (v != 0 and r // v != plan.pmuls[j]) or v == Int64.MIN:
                    keys.pok[i] = 0
                    continue
                pv[i] = r
            else:
                pv[i] = v
        pvals.append(pv^)
    keys.pkeys = List[Int64](length=n, fill=0)
    if plan.nk == 1:
        keys.pkeys = pvals[0].copy()
    else:
        var stride = UInt64(1)
        for j in range(plan.nk):
            for i in range(n):
                if keys.pok[i] != 0 and (pvals[j][i] < plan.los[j] or UInt64(pvals[j][i] - plan.los[j]) >= plan.spans[j]):
                    keys.pok[i] = 0                                  # outside the build side's range: it cannot match
                else:
                    keys.pkeys[i] = Int64(UInt64(keys.pkeys[i]) + UInt64(pvals[j][i] - plan.los[j]) * stride)
            stride *= plan.spans[j]
    keys.pk_addr = Int(keys.pkeys.unsafe_ptr())
    keys.po_addr = Int(keys.pok.unsafe_ptr())


# ── the pairs ────────────────────────────────────────────────────────────────────────────────────────

struct JoinPairs(Movable):
    """The matching (probe row, build row) pairs, in probe-row order: two Int32 arrays the kernel allocated (freed with the pairs)."""
    var n: Int
    var probe_a: Int
    var build_a: Int

    def __init__(out self):
        self.n = 0
        self.probe_a = 0
        self.build_a = 0

    def __del__(deinit self):
        hj_release(self.probe_a)
        hj_release(self.build_a)


def join_pairs(m: JoinMap, keys: JoinKeys, n_probe: Int, budget_rows: Int, threads: Int, outer: Bool = False) raises -> JoinPairs:
    """Every (probe row, build row) pair in probe-row order. `outer`: a probe row with no partner is one pair whose build row is -1 (a LEFT join)."""
    var out = JoinPairs()
    if n_probe == 0:
        return out^
    if outer and keys.verify:
        raise Error("a LEFT join on keys too wide to pack into one int64 (several columns whose value ranges multiply past 2^62) is not supported yet: " +
                    "join on fewer or narrower keys, or write it as an inner join (F-155)")
    var info = List[Int64](length=4, fill=0)
    var rc = hj_probe_pairs(m, keys.pk_addr, keys.po_addr, n_probe, threads, budget_rows, Int(info.unsafe_ptr()), outer)
    if rc == E_BUDGET:
        raise Error("the join produces " + commas(Int(info[0])) + " rows, past the join's memory budget of " + commas(budget_rows) + " rows: filter either side first, " +
                    "join on a unique key, or raise the budget (`PlannerOptions.join_budget`, `mx --join-budget`); a join that spills to disk is not built yet")
    if rc != 0:
        raise Error("the hash join failed with code " + String(rc))
    var n = Int(info[0])
    var pr = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(info[1]))
    var br = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(info[2]))
    if keys.verify:                                                  # hashed tuples: keep the pairs whose key columns are really equal
        var kept = 0
        for i in range(n):
            var same = True
            for j in range(len(keys.bcomp)):
                if keys.bcomp[j][Int(br[i])] != keys.pcomp[j][Int(pr[i])]:
                    same = False
                    break
            if same:
                pr[kept] = pr[i]
                br[kept] = br[i]
                kept += 1
        n = kept
    out.n = n
    out.probe_a = Int(info[1])
    out.build_a = Int(info[2])
    return out^


def select_rows(m: JoinMap, keys: JoinKeys, n_probe: Int, fmode: Int, threads: Int) raises -> JoinPairs:
    """The probe rows that pass a presence filter (`PM_SEMI`: a partner exists, `PM_ANTI`: none does, `PM_ANTI_NA`: none does and the key is not NULL,
    `PM_NONE`: no row), in row order, as the `probe_a` rows of the result (`build_a` is 0). Hashed composite keys are verified against the columns."""
    var out = JoinPairs()
    if n_probe == 0:
        return out^
    if not keys.verify:
        var info = List[Int64](length=2, fill=0)
        _ = hj_probe_select(m, keys.pk_addr, keys.po_addr, n_probe, threads, fmode, Int(info.unsafe_ptr()))
        out.n = Int(info[0])
        out.probe_a = Int(info[1])
        return out^
    var pairs = join_pairs(m, keys, n_probe, 1 << 40, threads)
    var mark = List[UInt8](length=n_probe, fill=0)
    var pr = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=pairs.probe_a)
    for i in range(pairs.n):
        mark[Int(pr[i])] = 1
    var po = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=keys.po_addr)
    var count = 0
    var keep = List[UInt8](length=n_probe, fill=0)
    for i in range(n_probe):
        var usable = keys.po_addr == 0 or po[i] != 0
        var k = False
        if fmode == PM_SEMI:
            k = mark[i] != 0
        elif fmode == PM_ANTI:
            k = mark[i] == 0
        elif fmode == PM_ANTI_NA:
            k = mark[i] == 0 and usable
        if k:
            keep[i] = 1
            count += 1
    if count == 0:
        return out^
    var rows = unsafe_alloc[Int32](count)
    var at = 0
    for i in range(n_probe):
        if keep[i] != 0:
            rows[at] = Int32(i)
            at += 1
    out.n = count
    out.probe_a = Int(rows)
    return out^


struct JoinSide(Movable):
    """One side of the join: its table, the table's columns the joined rows carry (the needed ones, in order) and the table's column of each
    join key. A side with no filter over a plain table is the source table itself (nothing is copied); any other side is the result of a row
    query over its source."""
    var table: SharedTable
    var cols: List[Int]
    var key_cols: List[Int]
    var zero_copy: Bool

    def __init__(out self, table: SharedTable, var cols: List[Int], var key_cols: List[Int], zero_copy: Bool):
        self.table = table
        self.cols = cols^
        self.key_cols = key_cols^
        self.zero_copy = zero_copy


def _gather_column(src: Int, valid: Int, rows_a: Int, n: Int, dst_a: Int, nul_a: Int, threads: Int):
    """dst[i] = src[rows[i]] (and the NULL byte from the validity bits), over slices of the rows in parallel. A row number of -1 (a LEFT join's
    unmatched row) is a NULL: the value 0 and a NULL byte."""
    var tasks = max(1, min(threads if threads > 0 else core_count(), n // 32768 + 1))

    def work(t: Int) {imm}:
        var s = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=src)
        var rows = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=rows_a)
        var dst = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=dst_a)
        var a = n * t // tasks
        var b = n * (t + 1) // tasks
        for i in range(a, b):
            var r = Int(rows[i])
            dst[i] = s[r] if r >= 0 else 0
        if nul_a != 0:
            var nd = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=nul_a)
            var bits = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=valid)
            for i in range(a, b):
                var r = Int(rows[i])
                if r < 0:
                    nd[i] = 1
                elif valid != 0:
                    nd[i] = 1 - ((bits[r >> 3] >> UInt8(r & 7)) & 1)
                else:
                    nd[i] = 0

    par_n(work, tasks, tasks)


def gather_join(left: JoinSide, right: JoinSide, pairs: JoinPairs, left_builds: Bool, threads: Int, outer: Bool = False) raises -> Table:
    """The joined table: the carried columns of the left side at its rows of the pairs, then the right side's (the build side's rows are the
    pairs' build rows, the probe side's the pairs' probe rows). `outer`: the build side's rows may be -1 (its columns are NULL there)."""
    var n = pairs.n
    var t = Table.empty()
    for side in range(2):
        ref sd = left if side == 0 else right
        var rows_a = (pairs.build_a if left_builds else pairs.probe_a) if side == 0 else (pairs.probe_a if left_builds else pairs.build_a)
        for c in sd.cols:
            var kind = sd.table.kind(c)
            var col = t.add_column(_unique_name(t, sd.table.name(c)), kind, scale=sd.table.scale(c))
            var vals = List[Int64](unsafe_uninit_length=n)
            var has_nulls = sd.table.null_count(c) > 0 or (outer and (side == 0) == left_builds)       # outer: the build side's columns hold NULLs for unmatched rows
            var nul = List[UInt8](length=n if has_nulls else 0, fill=0)
            _gather_column(Int(sd.table.values(c).unsafe_ptr()), _valid_bits(sd.table, c), rows_a, n, Int(vals.unsafe_ptr()),
                           Int(nul.unsafe_ptr()) if has_nulls else 0, threads)
            if kind == TEXT:                                          # the side's whole dictionary, codes unchanged (shared, not copied)
                t.share_dictionary(col, sd.table, c)
            _fill_column_owned(t.cols[col], vals^, nul^)
    return t^


def _fill_column_owned(mut col: Column, var vals: List[Int64], var nul: List[UInt8]):
    col.values = vals^
    if len(nul) > 0:
        var n = len(col.values)
        col.validity = List[UInt8](length=(n + 7) // 8, fill=0xFF)
        var nulls = 0
        for i in range(n):
            if nul[i] != 0:
                col.validity[i // 8] &= ~(UInt8(1) << UInt8(i % 8))
                nulls += 1
        col.nulls = nulls
        if nulls == 0:
            col.validity = List[UInt8]()


