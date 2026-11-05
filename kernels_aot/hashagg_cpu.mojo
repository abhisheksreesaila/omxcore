"""
F-069: partitioned hash aggregate for the native executor (`executor_cpu.mojo`).

Group keys that cannot be addressed directly (float64, key tuples wider than the direct-address space,
integers spanning the whole int64 range, expression keys of any value) become tuples of int64 *key words*;
the aggregate then runs in two phases, with no n_rows-sized array and no atomics:

  1. Workers claim morsels (MORSEL rows) and run the program exactly as the dense paths do
     (`pipeline_cpu._process_range`: predicate, fused join front stage, aggregate inputs) but with a
     per-worker LOCAL hash table numbering the key tuples of the surviving rows; the aggregates accumulate
     into per-worker partial arrays indexed by the local number (`_accumulate`, the same code as the dense
     paths). The local table holds `cap` groups; when a block could overflow it, and at the end, the worker
     FLUSHES: one record (hash, rows, one sub-total per aggregate, the key words) per local group is
     appended to the worker's buffer of the partition the group's hash selects, and the table is reset.
  2. Partitions (a power of two chosen from the row count, about 32k rows each) are merged in parallel, one
     worker per partition at a time: the records of every worker are folded into a table sized to the
     partition (open addressing, groups numbered in order of first record), then the partitions' groups
     are copied side by side into the result arrays the executor's finishing code reads (HAVING, ORDER BY,
     key order, output): key words per group, `vals[agg * G + g]`, `cnt[g]`.

Default mode (fastest): the local table lives across morsels (it flushes only when full), four partial
lanes per group, partitions merged in the order the workers' buffers happen to be (the last bits of a
float sum can differ between runs).

Reproducible mode (`repro`, F-064): a group's rows are added in the defined order: per morsel, in row
order, into ONE sub-total (a single lane; the local table has room for a whole morsel, so it never flushes
inside one), and the morsels' sub-totals are added in morsel order. A flush at the end of every morsel
emits that morsel's records tagged with its number; the merge of a partition walks the workers' buffers
in tag order. The result does not depend on the thread count, the aggregate list or the schedule, and equals
the dense paths' order for more than 1,024 groups (a NumPy reference in tests/test_f069_hashed_groups.py).

Key words: int32 / int64 / dictionary entry ids as they are; float64 as order-preserving bits with every
NaN one value and -0.0 equal to 0.0 (`float_word`), so equal words are equal keys and the words sort as the
keys do (NaN after +inf, as DuckDB). The caller supplies `keyw`, which fills `words[k * VEC + j]` for the
j-th selected row of a block (the same block contract as `_encode_sel`).

Result words (`res`, caller-allocated, 3 + n_keys): 0 G, 1 vals* (float64[n_aggs * max(G, 1)]),
2 cnt* (int64[max(G, 1)]), 3 + k key words* (int64[max(G, 1)]); all from `unsafe_alloc`.
Return: 0 ok, -1 bad program, -3 too many registers, -5 `keyw` failed (invalid dictionary code),
-6 the flushed records passed `rec_limit` bytes (0: no limit; the caller's other path answers),
-7 an integer sum (OP_AGG_ISUM) passed int64 (the caller's checked path answers).
"""

from std.memory.alloc import unsafe_alloc
from std.memory import Pointer, bitcast
from _fpenv import par, par_n, start_pool
from std.atomic import Atomic
from std.ffi import external_call

from pipeline_cpu import Prog, _run_ops, _process_range, _worker_state, _no_front, _ibits, _iadd, _is_iext, _iext, _fill_identity, VEC, MORSEL, MAX_REGS, WS_BYTES_MASK, morsel_rows

comptime FP = Pointer[Float64, MutUntrackedOrigin]
comptime IP = Pointer[Int32, MutUntrackedOrigin]
comptime LP = Pointer[Int64, MutUntrackedOrigin]
comptime BP = Pointer[UInt8, MutUntrackedOrigin]

comptime OP_AGG_SUM = 50
comptime OP_AGG_MIN = 51
comptime OP_AGG_MAX = 52
comptime OP_AGG_ISUM = 53
comptime INF = Float64(1.7976931348623157e308) * 10.0

comptime DEF_CAP = 8192          # default mode: groups a worker's local table holds before it flushes
comptime DEF_LANES = 4           # default mode: partial lanes per group (breaks the add dependency chain)
comptime PART_ROWS = 32768       # rows per partition (sizes the partition count)
comptime MAX_PARTS = 1024
comptime SEED = UInt64(0x9E3779B97F4A7C15)

# per-worker state words (hs[S_*])
comptime S_TABLE = 0     # Int32[tsize]: slot -> local group, or -1
comptime S_KEYS = 1      # Int64[cap * nk]: key words of each local group
comptime S_HASH = 2      # Int64[cap]
comptime S_PV = 3        # Float64[A * cap * lanes]: partial values, (a * cap + g) * lanes + lane
comptime S_PC = 4        # Int64[cap * lanes]: partial row counts
comptime S_USED = 5      # Int32[cap]: the table slot of each local group (to reset the table cheaply)
comptime S_COUNT = 6     # local groups now
comptime S_SEG = 7       # tag of the next flush: (morsel << 20) | flushes in the morsel so far
comptime S_WORDS = 8     # Int64[nk * VEC]: key words of a block
comptime S_HH = 9        # UInt64[VEC]: hashes of a block
comptime S_BUF = 10      # Int64[3 * nparts]: this worker's buffer per partition (address, words used, words capacity)
comptime S_RB = 11       # address of the shared Int64: record bytes flushed so far, all workers
comptime S_LIM = 12      # the limit on it (0: none)
comptime S_OVER = 13     # set when the record limit is passed (the morsel loop stops)
comptime S_N = 14

# partition result words (pr[i * PR_N + PR_*])
comptime PR_G = 0
comptime PR_KEYS = 1
comptime PR_CNT = 2
comptime PR_VALS = 3
comptime PR_NREC = 4
comptime PR_N = 5


@always_inline
def _fp(a: Int) -> FP:
    return FP(unsafe_from_address=a)


@always_inline
def _ip(a: Int) -> IP:
    return IP(unsafe_from_address=a)


@always_inline
def _lp(a: Int) -> LP:
    return LP(unsafe_from_address=a)


@always_inline
def _free(a: Int):
    Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=a).free()


@always_inline
def _mix(h: UInt64) -> UInt64:
    var z = h
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


@always_inline
def _u(v: Int64) -> UInt64:
    return bitcast[DType.uint64, 1](SIMD[DType.int64, 1](v))[0]


@always_inline
def _i(v: UInt64) -> Int64:
    return bitcast[DType.int64, 1](SIMD[DType.uint64, 1](v))[0]


@always_inline
def float_word(x: Float64) -> Int64:
    """Key word of a float64: every NaN one value (a quiet positive NaN, above +inf), -0.0 equal to 0.0, the
    rest order-preserving (negative values have their low 63 bits flipped), so words compare as the floats do."""
    if x != x:
        return Int64(0x7FF8000000000000)
    if x == 0.0:
        return Int64(0)
    var b = bitcast[DType.int64, 1](SIMD[DType.float64, 1](x))[0]
    return b ^ ((b >> 63) & Int64(0x7FFFFFFFFFFFFFFF))


@always_inline
def word_float(w: Int64) -> Float64:
    """The float64 a key word stands for."""
    return bitcast[DType.float64, 1](SIMD[DType.int64, 1](word_bits(w)))[0]


@always_inline
def word_bits(w: Int64) -> Int64:
    """The IEEE bits of the float64 a key word stands for (the inverse of float_word; zero comes back as +0.0)."""
    return w ^ ((w >> 63) & Int64(0x7FFFFFFFFFFFFFFF))


def _pow2_at_least(n: Int, lo: Int) -> Int:
    var t = lo
    while t < n:
        t *= 2
    return t


@always_inline
def _kind(p: Prog, a: Int) -> Int:
    return Int(p.instr(p.n_pro + p.n_pred + p.n_body, a)[0])


def _reserve(bufs: LP, part: Int, words: Int) -> Int:
    """Address of `words` more int64 at the end of partition `part`'s buffer (grown by doubling)."""
    var addr = Int(bufs[3 * part])
    var used = Int(bufs[3 * part + 1])
    var capw = Int(bufs[3 * part + 2])
    if used + words > capw:
        var ncap = max(256, capw * 2)
        while ncap < used + words:
            ncap *= 2
        var nb = Int(unsafe_alloc[Int64](ncap))
        if used > 0:
            _ = external_call["memcpy", Int](nb, addr, used * 8)
        if addr != 0:
            _free(addr)
        bufs[3 * part] = Int64(nb)
        bufs[3 * part + 2] = Int64(ncap)
        addr = nb
    bufs[3 * part + 1] = Int64(used + words)
    return addr + used * 8


def _flush(p: Prog, hs: LP, nk: Int, cap: Int, lanes: Int, nparts: Int, pshift: Int):
    """Write one record per local group to its partition's buffer and reset the table and the partials."""
    var count = Int(hs[S_COUNT])
    if count == 0:
        return
    var A = p.n_aggs
    var table = _ip(Int(hs[S_TABLE]))
    var keys = _lp(Int(hs[S_KEYS]))
    var lhash = _lp(Int(hs[S_HASH]))
    var pv = _fp(Int(hs[S_PV]))
    var pc = _lp(Int(hs[S_PC]))
    var used = _ip(Int(hs[S_USED]))
    var bufs = _lp(Int(hs[S_BUF]))
    var rw = 3 + A + nk
    var seg = hs[S_SEG]
    for g in range(count):
        var h = lhash[g]
        var part = 0 if nparts == 1 else Int(_u(h) >> UInt64(pshift))
        var rec = _reserve(bufs, part, rw)
        var r = _lp(rec)
        r[0] = seg
        r[1] = h
        var c = Int64(0)
        for l in range(lanes):
            c += pc[g * lanes + l]
            pc[g * lanes + l] = 0
        r[2] = c
        for a in range(A):
            var kind = _kind(p, a)
            var init = 0.0
            if kind == OP_AGG_MIN:
                init = INF
            elif kind == OP_AGG_MAX:
                init = -INF
            var q = pv + (a * cap + g) * lanes
            if _is_iext(kind):                            # F-060c: an int64 extreme, reduced over the lanes
                var xq = _ibits(q)
                var xacc = xq[0]
                for l in range(1, lanes):
                    xacc = _iext(kind, xacc, xq[l])
                _fill_identity(q, lanes, kind)
                _lp(rec + (3 + a) * 8)[0] = xacc
                continue
            if kind == OP_AGG_ISUM:                       # an int64 total: the same words, read as int64
                var iq = _ibits(q)
                var iacc = iq[0]
                iq[0] = 0
                for l in range(1, lanes):
                    iacc = _iadd(iacc, iq[l], p.ovf)
                    iq[l] = 0
                _lp(rec + (3 + a) * 8)[0] = iacc
                continue
            var acc = q[0]
            q[0] = init
            for l in range(1, lanes):
                if kind == OP_AGG_SUM:
                    acc += q[l]
                elif kind == OP_AGG_MIN:
                    acc = min(acc, q[l])
                else:
                    acc = max(acc, q[l])
                q[l] = init
            _fp(rec + (3 + a) * 8)[0] = acc
        for k in range(nk):
            r[3 + A + k] = keys[g * nk + k]
        table[Int(used[g])] = -1
    hs[S_COUNT] = 0
    hs[S_SEG] = seg + 1
    var tot = Int(Atomic[Scalar[DType.int64]].fetch_add(_lp(Int(hs[S_RB])), Int64(count * rw * 8))) + count * rw * 8
    if Int(hs[S_LIM]) > 0 and tot > Int(hs[S_LIM]):
        hs[S_OVER] = 1


def hash_aggregate[KW: def(Int, Int, IP, Int, Bool, LP, FP, BP) -> Bool,
                   FR: def(Int, Int, Int, IP, BP) -> Int, fused: Bool](
    code_a: Int, consts_a: Int, set_bytes_a: Int, set_meta_a: Int, tbl_a: Int, n_chunks: Int, dtypes_a: Int,
    nk: Int, n_rows: Int, threads: Int, repro: Bool, keyw: KW, front: FR, ws_arg: Int, res: LP,
    rec_limit: Int,
) -> Int:
    var code = _ip(code_a)
    if Int(code[0]) != 0x4d585031:
        return -1
    var ws_bytes = ws_arg & WS_BYTES_MASK
    var MR = morsel_rows(ws_arg)             # probe rows per morsel (MORSEL, less when the front stage expands rows: F-161)
    var cap = (MORSEL + VEC) if repro else DEF_CAP
    var lanes = 1 if repro else DEF_LANES
    var ovf = unsafe_alloc[Int64](1)                  # set by an integer sum that passes int64 (pipeline_cpu._iadd)
    ovf[0] = 0
    var p = Prog(code, _fp(consts_a), BP(unsafe_from_address=set_bytes_a), _lp(set_meta_a), _lp(tbl_a),
                 n_chunks, _ip(dtypes_a), _ip(0), cap, True, ovf)
    if p.n_regs > MAX_REGS:
        return -3
    p.nl = lanes
    var A = p.n_aggs
    var rw = 3 + A + nk
    var morsels = (n_rows + MR - 1) // MR
    var nt = max(1, min(threads if threads > 0 else 1, morsels))
    var nparts = 1
    while nparts < MAX_PARTS and nparts * PART_ROWS < n_rows:
        nparts *= 2
    var pbits = 0
    while (1 << pbits) < nparts:
        pbits += 1
    var pshift = 64 - pbits
    var tsize = _pow2_at_least(2 * cap, 1024)
    var mask = UInt64(tsize - 1)

    var hstate = unsafe_alloc[Int64](nt * S_N)
    var wbufs = unsafe_alloc[Int64](nt * 3 * nparts)
    var wst = unsafe_alloc[Int64](nt * 4)
    for i in range(nt * S_N):
        hstate[i] = 0
    for i in range(nt * 3 * nparts):
        wbufs[i] = 0
    for i in range(nt * 4):
        wst[i] = 0
    var counter = unsafe_alloc[Int64](1)
    counter[0] = 0
    var failed = unsafe_alloc[Int64](1)
    failed[0] = 0
    var rbytes = unsafe_alloc[Int64](1)
    rbytes[0] = 0

    def worker(wid: Int) {imm}:
        var st = wst + wid * 4
        _worker_state(p, ws_bytes, st)
        var regs = _fp(Int(st[0]))
        var sel = _ip(Int(st[1]))
        var gscratch = _ip(Int(st[2]))
        var ws = BP(unsafe_from_address=Int(st[3]))
        var hs = hstate + wid * S_N
        var table = unsafe_alloc[Int32](tsize)
        var keys = unsafe_alloc[Int64](cap * nk)
        var lhash = unsafe_alloc[Int64](cap)
        var pv = unsafe_alloc[Float64](max(A, 1) * cap * lanes)
        var pc = unsafe_alloc[Int64](cap * lanes)
        var used = unsafe_alloc[Int32](cap)
        var words = unsafe_alloc[Int64](nk * VEC)
        var hh = unsafe_alloc[UInt64](VEC)
        for i in range(tsize):
            table[i] = -1
        for a in range(A):
            _fill_identity(pv + a * cap * lanes, cap * lanes, _kind(p, a))
        for i in range(cap * lanes):
            pc[i] = 0
        hs[S_TABLE] = Int64(Int(table))
        hs[S_KEYS] = Int64(Int(keys))
        hs[S_HASH] = Int64(Int(lhash))
        hs[S_PV] = Int64(Int(pv))
        hs[S_PC] = Int64(Int(pc))
        hs[S_USED] = Int64(Int(used))
        hs[S_BUF] = Int64(Int(wbufs + wid * 3 * nparts))
        hs[S_COUNT] = 0
        hs[S_SEG] = Int64(wid) << 20
        hs[S_RB] = Int64(Int(rbytes))
        hs[S_LIM] = Int64(rec_limit)

        def enc(c: Int, i0: Int, sl: IP, nsel: Int, use_sel: Bool, dst: IP, rg: FP) {imm} -> Bool:
            if not keyw(c, i0, sl, nsel, use_sel, words, rg, ws):
                return False
            if Int(hs[S_COUNT]) + nsel > cap:
                _flush(p, hs, nk, cap, lanes, nparts, pshift)
            for j in range(nsel):
                hh[j] = SEED
            for k in range(nk):
                var wk = words + k * VEC
                for j in range(nsel):
                    hh[j] = _mix(hh[j] ^ _u(wk[j]))
            var count = Int(hs[S_COUNT])
            for j in range(nsel):
                var h = hh[j]
                var hi = _i(h)
                var pos = Int(h & mask)
                while True:
                    var g = Int(table[pos])
                    if g < 0:
                        g = count
                        count += 1
                        table[pos] = Int32(g)
                        used[g] = Int32(pos)
                        lhash[g] = hi
                        for k in range(nk):
                            keys[g * nk + k] = words[k * VEC + j]
                        dst[j] = Int32(g)
                        break
                    if lhash[g] == hi:
                        var same = True
                        for k in range(nk):
                            if keys[g * nk + k] != words[k * VEC + j]:
                                same = False
                                break
                        if same:
                            dst[j] = Int32(g)
                            break
                    pos = (pos + 1) & Int(mask)
            hs[S_COUNT] = Int64(count)
            return True

        while True:
            var m = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
            var lo = m * MR
            if lo >= n_rows or Int(hs[S_OVER]) != 0 or Int(failed[0]) != 0:
                break
            if repro:
                hs[S_SEG] = Int64(m) << 20
            if not _process_range[keyed=True, mode=0, fused=fused](
                p, regs, sel, gscratch, pv, pc, lo, min(lo + MR, n_rows), enc, front, ws):
                failed[0] = 1
                break
            if repro:
                _flush(p, hs, nk, cap, lanes, nparts, pshift)
        _flush(p, hs, nk, cap, lanes, nparts, pshift)
        table.free()
        keys.free()
        lhash.free()
        pv.free()
        pc.free()
        used.free()
        words.free()
        hh.free()
        for f in range(4):
            _free(Int(st[f]))

    if nt == 1:
        worker(0)
    else:
        start_pool()
        par(worker, nt)

    var over = False
    for w in range(nt):
        over = over or Int(hstate[w * S_N + S_OVER]) != 0
    var rc = -5 if Int(failed[0]) != 0 else (-6 if over else 0)
    var pres = unsafe_alloc[Int64](nparts * PR_N)
    for i in range(nparts * PR_N):
        pres[i] = 0
    if rc == 0:
        counter[0] = 0

        def merge_part(wid: Int) {imm}:
            var cur = unsafe_alloc[Int64](nt)
            while True:
                var pi = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
                if pi >= nparts:
                    break
                var nrec = 0
                for w in range(nt):
                    nrec += Int(wbufs[(w * nparts + pi) * 3 + 1]) // rw
                if nrec == 0:
                    continue
                var ts = _pow2_at_least(2 * nrec, 16)
                var tmask = ts - 1
                var tab = unsafe_alloc[Int32](ts)
                for i in range(ts):
                    tab[i] = -1
                var ekeys = unsafe_alloc[Int64](nrec * nk)
                var ehash = unsafe_alloc[Int64](nrec)
                var ecnt = unsafe_alloc[Int64](nrec)
                var evals = unsafe_alloc[Float64](max(A, 1) * nrec)
                var ne = 0
                for w in range(nt):
                    cur[w] = 0

                while True:
                    # the next segment to fold: reproducible mode, the smallest tag among the workers'
                    # buffers (their morsel order); default mode, the workers one after another
                    var best = -1
                    var bseg = Int64(0)
                    for w in range(nt):
                        var b = wbufs + (w * nparts + pi) * 3
                        if Int(cur[w]) < Int(b[1]):
                            var s = _lp(Int(b[0]) + Int(cur[w]) * 8)[0]
                            if best < 0 or (repro and s < bseg):
                                best = w
                                bseg = s
                            if not repro:
                                break
                    if best < 0:
                        break
                    var b = wbufs + (best * nparts + pi) * 3
                    var end = Int(b[1])
                    while Int(cur[best]) < end:
                        var rec = Int(b[0]) + Int(cur[best]) * 8
                        var r = _lp(rec)
                        if repro and r[0] != bseg:
                            break
                        var h = r[1]
                        var pos = Int(h) & tmask
                        var e = 0
                        while True:
                            e = Int(tab[pos])
                            if e < 0:
                                e = ne
                                ne += 1
                                tab[pos] = Int32(e)
                                ehash[e] = h
                                for k in range(nk):
                                    ekeys[e * nk + k] = r[3 + A + k]
                                ecnt[e] = 0
                                for a in range(A):
                                    _fill_identity(evals + (a * nrec + e), 1, _kind(p, a))
                                break
                            if ehash[e] == h:
                                var same = True
                                for k in range(nk):
                                    if ekeys[e * nk + k] != r[3 + A + k]:
                                        same = False
                                        break
                                if same:
                                    break
                            pos = (pos + 1) & tmask
                        ecnt[e] += r[2]
                        for a in range(A):
                            var kind = _kind(p, a)
                            var x = _fp(rec + (3 + a) * 8)[0]
                            var q = evals + (a * nrec + e)
                            if kind == OP_AGG_SUM:
                                q[0] += x
                            elif kind == OP_AGG_ISUM:
                                var iq = _ibits(q)
                                iq[0] = _iadd(iq[0], _lp(rec + (3 + a) * 8)[0], p.ovf)
                            elif _is_iext(kind):
                                var xq = _ibits(q)
                                xq[0] = _iext(kind, xq[0], _lp(rec + (3 + a) * 8)[0])
                            elif kind == OP_AGG_MIN:
                                q[0] = min(q[0], x)
                            else:
                                q[0] = max(q[0], x)
                        cur[best] += Int64(rw)
                tab.free()
                ehash.free()
                for w in range(nt):                       # this partition's records are folded: give them back now
                    var b = wbufs + (w * nparts + pi) * 3
                    if Int(b[0]) != 0:
                        _free(Int(b[0]))
                        b[0] = 0
                        b[1] = 0
                        b[2] = 0
                # keep the groups, not the record-sized scratch: arrays of exactly ne entries
                var nkeys2 = unsafe_alloc[Int64](max(ne * nk, 1))
                var ncnt2 = unsafe_alloc[Int64](max(ne, 1))
                var nvals2 = unsafe_alloc[Float64](max(A * ne, 1))
                for e in range(ne * nk):
                    nkeys2[e] = ekeys[e]
                for e in range(ne):
                    ncnt2[e] = ecnt[e]
                for a in range(A):
                    for e in range(ne):
                        nvals2[a * ne + e] = evals[a * nrec + e]
                ekeys.free()
                ecnt.free()
                evals.free()
                var o = pres + pi * PR_N
                o[PR_G] = Int64(ne)
                o[PR_KEYS] = Int64(Int(nkeys2))
                o[PR_CNT] = Int64(Int(ncnt2))
                o[PR_VALS] = Int64(Int(nvals2))
                o[PR_NREC] = Int64(ne)                    # (the values' stride)
            cur.free()

        var nm = max(1, min(nt, nparts))
        if nm == 1:
            merge_part(0)
        else:
            start_pool()
            par(merge_part, nm)

    if rc == 0 and ovf[0] != 0:
        rc = -7                                           # an integer sum passed int64: the caller's checked path answers

    # the partitions' groups side by side
    var total = 0
    var offs = unsafe_alloc[Int64](nparts + 1)
    for i in range(nparts):
        offs[i] = Int64(total)
        total += Int(pres[i * PR_N + PR_G])
    offs[nparts] = Int64(total)
    var G = max(total, 1)
    var vals = unsafe_alloc[Float64](max(A, 1) * G)
    var cnt = unsafe_alloc[Int64](G)
    var kw = unsafe_alloc[Int64](nk)
    cnt[0] = 0
    for a in range(A):
        vals[a * G] = 0.0
    for k in range(nk):
        kw[k] = Int64(Int(unsafe_alloc[Int64](G)))
        _lp(Int(kw[k]))[0] = 0
    if rc == 0 and total > 0:
        counter[0] = 0

        def copy_part(wid: Int) {imm}:
            while True:
                var pi = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
                if pi >= nparts:
                    break
                var o = pres + pi * PR_N
                var ne = Int(o[PR_G])
                if ne == 0:
                    continue
                var at = Int(offs[pi])
                var nrec = Int(o[PR_NREC])
                var ekeys = _lp(Int(o[PR_KEYS]))
                var ecnt = _lp(Int(o[PR_CNT]))
                var evals = _fp(Int(o[PR_VALS]))
                for e in range(ne):
                    cnt[at + e] = ecnt[e]
                for a in range(A):
                    for e in range(ne):
                        vals[a * G + at + e] = evals[a * nrec + e]
                for k in range(nk):
                    var dstk = _lp(Int(kw[k]))
                    for e in range(ne):
                        dstk[at + e] = ekeys[e * nk + k]

        var nm = max(1, min(nt, nparts))
        if nm == 1:
            copy_part(0)
        else:
            start_pool()
            par(copy_part, nm)

    for i in range(nparts):
        var o = pres + i * PR_N
        if Int(o[PR_G]) > 0 or Int(o[PR_NREC]) > 0:
            _free(Int(o[PR_KEYS]))
            _free(Int(o[PR_CNT]))
            _free(Int(o[PR_VALS]))
    for i in range(nt * nparts):
        if Int(wbufs[3 * i]) != 0:
            _free(Int(wbufs[3 * i]))
    if rc != 0:
        _free(Int(vals))
        _free(Int(cnt))
        for k in range(nk):
            _free(Int(kw[k]))
        res[0] = 0
    else:
        res[0] = Int64(total)
        res[1] = Int64(Int(vals))
        res[2] = Int64(Int(cnt))
        for k in range(nk):
            res[3 + k] = kw[k]
    pres.free()
    offs.free()
    kw.free()
    hstate.free()
    wbufs.free()
    wst.free()
    counter.free()
    failed.free()
    rbytes.free()
    ovf.free()
    return rc
