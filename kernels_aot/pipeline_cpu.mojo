"""
MXFrame fused pipeline kernel, CPU side (F-008).

A vectorized interpreter in the style of MonetDB/X100: Python compiles a filter
predicate plus a list of aggregates into a small bytecode program
(`pipeline_program.py`); this kernel runs that program over a row range
[start, end) of raw column pointers, without JIT and without materialising
filtered columns.

Per worker thread, per vector of VEC (1024) rows:
  1. PRED ops evaluate the predicate over all rows of the vector into a
     register (1.0 / 0.0) and a selection vector is built from it.
  2. BODY ops evaluate the aggregate input expressions on the selected rows
     only (LOAD_COL gathers through the selection vector; dense when every row
     survived). Registers are cache-resident float64 vectors (SIMD).
  3. AGG ops accumulate into worker-local float64 partials per group.
Workers claim morsels dynamically. Each morsel's partial table is folded into the output in morsel
order (F-064: a float sum has one defined value, whatever the thread count; see _pipeline for the
default mode, the reproducible mode and the staged many-groups path). All arithmetic is float64
(integers and float32 columns are widened on load), so sums accumulate in float64.

Build (mirrors scripts/release.sh):
    pixi run mojo build kernels_aot/pipeline_cpu.mojo \
        --emit shared-lib -o kernels_aot/libmxpipeline_cpu.so

Program/ABI contract (keep in sync with pipeline_program.py):
  code   int32[]  : header (16 ints) then 5-int instructions [op,dst,a,b,c]
                    in the order prologue, pred, body, aggs.
  consts float64[], set_bytes uint8[], set_meta int64[] (offset, length, lo)*
  column dtypes: 0 i32, 1 i64, 2 f32, 3 f64, 4 u8, 5 | (slot << 8) = int32 dictionary codes plus the
  integer in chunk-table column `slot` (the native executor's string predicates, F-031),
  6 | (slot << 8) = utf8 offsets (int32; the column's base address is rebased like any other) plus, in
  chunk-table columns `slot`, `slot + 1`, `slot + 2`: the chunk's bytes address, its validity bitmap
  address (0 = no nulls) and the bit index of row 0 of the chunk table's row numbering (F-034),
  7 | (slot << 8) = a column only tested by ISNULL (F-097): its values are never read (its chunk-table
  address is unused); chunk-table columns `slot`, `slot + 1` hold the chunk's validity bitmap address
  (0 = no nulls) and the bit index of row 0, as for dtype 6
  chunk table int64[n_chunks, 2 + n_cols]: row_start, row_end, then per column the
                    chunk's base address minus row_start * itemsize, so that
                    `addr + row * itemsize` is valid for every row of the chunk
                    (columns arrive as Arrow chunks; nothing is concatenated).
                    A single contiguous column set is one chunk [0, n).
  dtypes int32[] (0 i32,1 i64,2 f32,3 f64,4 u8)
  gids   int32[]  : group id per row (0 address = global aggregate, n_groups==1)
Outputs: out_vals float64[n_aggs * n_groups] (sum/min/max per agg and group),
         out_cnt int64[n_groups] (rows that passed the predicate per group).
"""

from std.memory.alloc import unsafe_alloc
from std.memory import Pointer, unsafe_memset_zero
from _fpenv import par, par_n, start_pool
from std.sys import simd_width_of, CompilationTarget
from std.atomic import Atomic
from std.math import floor
from std.ffi import external_call

comptime F64 = DType.float64
comptime W = simd_width_of[F64]()
comptime VEC = 1024
comptime MORSEL = 65536
comptime MAX_REGS = 64
comptime HDR = 16
comptime BUDGET_BYTES = 128 * 1024 * 1024   # partial tables / staging slots alive at once
comptime SHARED_MIN_ROWS = 1 << 20           # legacy shared output: inputs of at least this many rows
comptime FUSED_GROUPS = 65536               # default mode, fused join: the table path up to this many groups
comptime FUSED_MIN = 16384                  # default mode, fused join: the defined order from this many groups
comptime LEGACY_MERGE_GROUPS = 16384        # the previous code's merge task size
comptime MERGE_GROUPS = 256                 # groups per parallel merge task
comptime TABLE_GROUPS = 131072              # more groups than this: staged output, no table per morsel
comptime LANE_GROUPS = 1024                 # up to this many groups: 4 accumulator lanes per group

comptime FP = Pointer[Float64, MutUntrackedOrigin]
comptime IP = Pointer[Int32, MutUntrackedOrigin]
comptime LP = Pointer[Int64, MutUntrackedOrigin]
comptime BP = Pointer[UInt8, MutUntrackedOrigin]

# ── opcodes (mirror pipeline_program.py) ───────────────────────────────────
comptime OP_LOAD_COL = 1
comptime OP_LOAD_CONST = 2
comptime OP_LOADI = 3         # the column's values as int64 BITS in the register (an "int" register: F-060c, F-060b)
comptime OP_ADD = 10
comptime OP_SUB = 11
comptime OP_MUL = 12
comptime OP_DIV = 13
comptime OP_MOD = 14        # fmod: the sign follows the dividend (integer % a nonzero constant, F-101)
comptime OP_IADD = 15       # int64 + - * of two "int" registers (int64 bits in the float64 words); a result outside int64 sets *ovf (F-060b)
comptime OP_ISUB = 16
comptime OP_IMUL = 17
comptime OP_F2I = 18        # dst = int64(a): a float register of whole numbers within +-2^53
comptime OP_I2F = 19        # dst = float64(a) of an "int" register; a value outside +-2^53 sets *ovf
# Exact integer operations over "int" registers, for the Mojo API (docs/mxframe-mojo-api.md: formulas, bins, int64
# filters). The Python compiler does not emit them. Division and modulo follow Mojo's Int64 (floor, the remainder
# takes the divisor's sign) and give 0 for a zero divisor, as the app's formula columns do.
comptime OP_IDIV = 42       # dst = a // b
comptime OP_IMOD = 43       # dst = a % b
comptime OP_IMIN = 44       # dst = min(a, b)
comptime OP_IMAX = 45       # dst = max(a, b)
comptime OP_ICMP = 46       # dst = 1.0 if a <cmp> b else 0.0, cmp in operand c (OP_LT .. OP_NE): a float register
comptime OP_LT = 20
comptime OP_LE = 21
comptime OP_GT = 22
comptime OP_GE = 23
comptime OP_EQ = 24
comptime OP_NE = 25
comptime OP_AND = 30
comptime OP_OR = 31
comptime OP_NOT = 32
comptime OP_FLOOR = 33
comptime OP_F32 = 34        # float64(float32(a)): a FLOAT result rounded after each operation (F-110)
comptime OP_SELECT = 35
comptime OP_BETWEEN = 36
comptime OP_IN_SET = 37
comptime OP_CMPC = 38
comptime OP_ROWID = 39      # the row's number in the whole input (SELECT DISTINCT: first appearance, F-033)
comptime OP_STR = 40        # a plain utf8 column LIKE a pattern (F-034)
comptime OP_ISNULL = 41     # the column's validity bit is clear (F-097)
# F-156 (the Mojo API's SQL expressions; the Python compiler does not emit them): a table lookup, a hash set and an error flag.
comptime OP_TAB = 47        # dst = table[a - lo]: an "int" register in, int64 bits out (a float's bits too); out of range gives 0. Set s holds the 8-byte entries
                            # (set_meta[s*3] = their byte offset in set_bytes, 8-aligned; [s*3+1] = how many; [s*3+2] = lo): a TEXT function's answer per dictionary word
comptime OP_ISET = 48       # dst = 1.0 if the int64 in register a is in the hash set s, else 0.0 (an IN list of any length). Set s: set_meta[s*3] = the table's byte offset
                            # (8-aligned), [s*3+1] = its capacity (a power of two), [s*3+2] = 1 when the set holds INT64_MIN; the table's slots hold keys, empty = INT64_MIN
comptime OP_FLAG = 49       # *ovf = 1 when any of the first n lanes of register a is not 0.0 (a CAST met a value it cannot convert)
comptime OP_AGG_SUM = 50
comptime OP_AGG_MIN = 51
comptime OP_AGG_MAX = 52
comptime OP_AGG_ISUM = 53   # sum of an "int" register, accumulated in int64 (F-060, F-060b); the slot holds int64 bits
comptime OP_AGG_IMIN = 54   # min / max of an "int" register, accumulated in int64 (F-060c); the slot holds int64 bits
comptime OP_AGG_IMAX = 55
comptime E_OVERFLOW = -21   # an int64 sum or result passed its range, or an int64 value past 2^53 met a float op (the caller's exact path answers)
comptime LIM53 = Int64(1) << 53
comptime B62 = 4611686018427387904.0     # 2^62 as a double: |a double product| below it proves an int64 product fits

comptime INF = Float64(1.7976931348623157e308) * 10.0


@always_inline
def _fp(addr: Int) -> FP:
    return FP(unsafe_from_address=addr)

@always_inline
def _ibits(p: FP) -> LP:
    """An integer sum's slots (OP_AGG_ISUM) are float64 words that hold int64 values: same memory, read as int64."""
    return LP(unsafe_from_address=Int(p))

@always_inline
def _iadd(a: Int64, b: Int64, ovf: LP) -> Int64:
    """a + b in int64 (wrapping); sets *ovf when the true sum does not fit. A partial total that passes int64 and
    comes back (rows of both signs) flags too: the caller then adds exactly in the checked path."""
    var s = a + b
    if ((a ^ s) & (b ^ s)) < 0:
        ovf[0] = 1
    return s

@always_inline
def _is_iext(kind: Int) -> Bool:
    return kind == OP_AGG_IMIN or kind == OP_AGG_IMAX

@always_inline
def _iext(kind: Int, a: Int64, b: Int64) -> Int64:
    """The integer extreme of two slot words (OP_AGG_IMIN / OP_AGG_IMAX, F-060c)."""
    return min(a, b) if kind == OP_AGG_IMIN else max(a, b)

@always_inline
def _imul_ovf(x: Int64, y: Int64) -> Bool:
    """Does x * y pass int64? (exact; the wrapped product is checked by dividing back, and the two quotients that
    trap are decided first)."""
    if x == 0 or y == 0:
        return False
    if x == -1:
        return y == Int64.MIN
    if y == -1:
        return x == Int64.MIN
    return (x * y) // x != y

@always_inline
def _int_bin[OP: Int](d: FP, a: FP, b: FP, n: Int, ovf: LP):
    """OP_IADD / OP_ISUB / OP_IMUL over the first n lanes of two "int" registers (int64 bits in the float64 words), in
    int64. A result outside int64 sets *ovf (the lanes past n hold stale words: never looked at). A product is
    checked from its double: below 2^62 in magnitude it fits, otherwise the lane is checked exactly."""
    var dd = _ibits(d)
    var xa = _ibits(a)
    var xb = _ibits(b)
    var ov = SIMD[DType.int64, W](0)
    var i = 0
    while i + W <= n:
        var x = xa.load[width=W](i)
        var y = xb.load[width=W](i)
        comptime if OP == OP_IADD:
            var r = x + y
            ov |= (x ^ r) & (y ^ r)
            dd.store(i, r)
        elif OP == OP_ISUB:
            var r = x - y
            ov |= (x ^ y) & (x ^ r)
            dd.store(i, r)
        else:
            var fp = x.cast[F64]() * y.cast[F64]()
            var big = fp.ge(SIMD[F64, W](B62)) | fp.le(SIMD[F64, W](-B62))
            dd.store(i, x * y)
            if big.reduce_or():
                for l in range(W):
                    if big[l] and _imul_ovf(x[l], y[l]):
                        ovf[0] = 1
        i += W
    if ov.reduce_or() < 0:
        ovf[0] = 1
    while i < n:
        var x = xa[i]
        var y = xb[i]
        comptime if OP == OP_IADD:
            var r = x + y
            if ((x ^ r) & (y ^ r)) < 0:
                ovf[0] = 1
            dd[i] = r
        elif OP == OP_ISUB:
            var r = x - y
            if ((x ^ y) & (x ^ r)) < 0:
                ovf[0] = 1
            dd[i] = r
        else:
            if _imul_ovf(x, y):
                ovf[0] = 1
            dd[i] = x * y
        i += 1

def _f2i(d: FP, a: FP, n: Int):
    """OP_F2I: the first n lanes of a float register of whole numbers within +-2^53, as int64 bits."""
    var dd = _ibits(d)
    var i = 0
    while i + W <= n:
        dd.store(i, a.load[width=W](i).cast[DType.int64]())
        i += W
    while i < n:
        dd[i] = Int64(a[i])
        i += 1

def _decimal_f64(v: Int64, scale_u: UInt64, scale_f: Float64) -> Float64:
    """An int64 DECIMAL storage integer as a double, DuckDB's rule (probed bit for bit in tests/test_f190_decimal_to_double.py): up to
    +-2^53 the integer is exact and the double is v / 10^scale; past it, the whole part and the fraction are converted apart,
    double(v / 10^scale) + double(v % 10^scale) / 10^scale (truncating division). scale_u = 10^scale as an integer, scale_f as a double."""
    if v >= -LIM53 and v <= LIM53:
        return Float64(v) / scale_f
    var neg = v < 0
    var m = UInt64(0) - UInt64(v) if neg else UInt64(v)
    var q = m // scale_u
    var x = Float64(q) + Float64(m - q * scale_u) / scale_f
    return -x if neg else x

def _i2f(d: FP, a: FP, n: Int, ovf: LP, mode: Int):
    """OP_I2F: the first n lanes of an "int" register as doubles. `mode` is the instruction's last field. 0 (the Python API): a value
    outside +-2^53 is not exact as a double, *ovf is set and the caller's exact path answers. 1 + k (the Mojo API, F-190, which has no
    exact path and whose SQL is DuckDB's): the DECIMAL(k) integer read as a double the way DuckDB casts it, v / 10^k, rounded
    past 2^53 (`_decimal_f64`); k = 0 is a plain integer cast."""
    var xa = _ibits(a)
    var lim = SIMD[DType.int64, W](LIM53)
    var i = 0
    if mode > 0:
        var scale_u = UInt64(1)
        var scale_f = Float64(1.0)
        for _ in range(mode - 1):
            scale_u *= 10
            scale_f *= 10.0
        while i + W <= n:
            var v = xa.load[width=W](i)
            if (v.gt(lim) | v.lt(-lim)).reduce_or():
                for k in range(W):
                    d[i + k] = _decimal_f64(v[k], scale_u, scale_f)
            else:
                d.store(i, v.cast[F64]() / scale_f)
            i += W
        while i < n:
            d[i] = _decimal_f64(xa[i], scale_u, scale_f)
            i += 1
        return
    var oor = SIMD[DType.bool, W](fill=False)
    while i + W <= n:
        var v = xa.load[width=W](i)
        d.store(i, v.cast[F64]())
        oor = oor | v.gt(lim) | v.lt(-lim)
        i += W
    if oor.reduce_or():
        ovf[0] = 1
    while i < n:
        var v = xa[i]
        d[i] = Float64(v)
        if v > LIM53 or v < -LIM53:
            ovf[0] = 1
        i += 1

def _int_op2[OP: Int](d: FP, a: FP, b: FP, cmp: Int, n: Int, ovf: LP):
    """OP_IDIV / OP_IMOD / OP_IMIN / OP_IMAX / OP_ICMP over the first n lanes of two "int" registers. INT64_MIN // -1
    does not fit: *ovf is set (the caller's exact path or error answers)."""
    var dd = _ibits(d)
    var xa = _ibits(a)
    var xb = _ibits(b)
    for i in range(n):
        var x = xa[i]
        var y = xb[i]
        comptime if OP == OP_IDIV:
            if y == 0:
                dd[i] = 0
            elif y == -1 and x == Int64.MIN:
                ovf[0] = 1
                dd[i] = x
            else:
                dd[i] = x // y
        elif OP == OP_IMOD:
            dd[i] = 0 if y == 0 or y == -1 else x % y
        elif OP == OP_IMIN:
            dd[i] = min(x, y)
        elif OP == OP_IMAX:
            dd[i] = max(x, y)
        else:
            var r = (x < y) if cmp == OP_LT else (x <= y) if cmp == OP_LE else (x > y) if cmp == OP_GT else \
                    (x >= y) if cmp == OP_GE else (x == y) if cmp == OP_EQ else (x != y)
            d[i] = 1.0 if r else 0.0

def _fill_identity(base: FP, count: Int, kind: Int):
    """`count` slot words at their aggregate's identity: 0 for sums, +-inf for float extremes, INT64_MAX / INT64_MIN
    (as bits) for the integer extremes."""
    if _is_iext(kind):
        var ib = _ibits(base)
        var v = Int64.MAX if kind == OP_AGG_IMIN else Int64.MIN
        for i in range(count):
            ib[i] = v
        return
    var init = 0.0
    if kind == OP_AGG_MIN:
        init = INF
    elif kind == OP_AGG_MAX:
        init = -INF
    for i in range(count):
        base[i] = init

@always_inline
def _ip(addr: Int) -> IP:
    return IP(unsafe_from_address=addr)

@always_inline
def _lp(addr: Int) -> LP:
    return LP(unsafe_from_address=addr)

@always_inline
def _bp(addr: Int) -> BP:
    return BP(unsafe_from_address=addr)


# ── column loads (widen to float64) ────────────────────────────────────────

@always_inline
def _load_dense[T: DType](dst: FP, addr: Int, i0: Int, n: Int):
    var src = Pointer[Scalar[T], MutUntrackedOrigin](unsafe_from_address=addr) + i0
    var i = 0
    while i + W <= n:
        dst.store(i, src.load[width=W](i).cast[F64]())
        i += W
    while i < n:
        dst[i] = src[i].cast[F64]()
        i += 1

@always_inline
def _load_gather[T: DType](dst: FP, addr: Int, i0: Int, sel: IP, n: Int):
    var src = Pointer[Scalar[T], MutUntrackedOrigin](unsafe_from_address=addr) + i0
    for j in range(n):
        dst[j] = src[Int(sel[j])].cast[F64]()

@always_inline
def _coli_t[T: DType](dst: FP, addr: Int, i0: Int, sel: IP, use_sel: Bool, n: Int):
    """OP_LOADI: the column's values widened to int64 and stored as that int64's bits (no float64 on the way)."""
    var d = _ibits(dst)
    var src = Pointer[Scalar[T], MutUntrackedOrigin](unsafe_from_address=addr) + i0
    if use_sel:
        for j in range(n):
            d[j] = src[Int(sel[j])].cast[DType.int64]()
    else:
        for j in range(n):
            d[j] = src[j].cast[DType.int64]()

def _load_coli(dst: FP, addr: Int, dtype: Int, i0: Int, sel: IP, use_sel: Bool, n: Int):
    if dtype == 0:
        _coli_t[DType.int32](dst, addr, i0, sel, use_sel, n)
    elif dtype == 4:
        _coli_t[DType.uint8](dst, addr, i0, sel, use_sel, n)
    else:                                         # 1: int64, 8: int64 read only here (exact past 2^53)
        _coli_t[DType.int64](dst, addr, i0, sel, use_sel, n)

def _load_col(dst: FP, addr: Int, dtype: Int, i0: Int, sel: IP, use_sel: Bool, n: Int):
    if use_sel:
        if dtype == 0:
            _load_gather[DType.int32](dst, addr, i0, sel, n)
        elif dtype == 1:
            _load_gather[DType.int64](dst, addr, i0, sel, n)
        elif dtype == 2:
            _load_gather[DType.float32](dst, addr, i0, sel, n)
        elif dtype == 3:
            _load_gather[DType.float64](dst, addr, i0, sel, n)
        else:
            _load_gather[DType.uint8](dst, addr, i0, sel, n)
    else:
        if dtype == 0:
            _load_dense[DType.int32](dst, addr, i0, n)
        elif dtype == 1:
            _load_dense[DType.int64](dst, addr, i0, n)
        elif dtype == 2:
            _load_dense[DType.float32](dst, addr, i0, n)
        elif dtype == 3:
            _load_dense[DType.float64](dst, addr, i0, n)
        else:
            _load_dense[DType.uint8](dst, addr, i0, n)


# ── vector ops: n is padded up to a multiple of W by the caller ────────────

# ── NaN-aware compares (F-070): DuckDB puts NaN above every number (+inf too) and NaN = NaN ────────
# Branch-free: IEEE's compare, then NaN's rows fixed with the (x != x) masks. Only float operands get them.

@always_inline
def _lt_na(x: SIMD[F64, W], y: SIMD[F64, W]) -> SIMD[DType.bool, W]:
    return x.lt(y) | (~y.eq(y) & x.eq(x))

@always_inline
def _le_na(x: SIMD[F64, W], y: SIMD[F64, W]) -> SIMD[DType.bool, W]:
    return x.le(y) | ~y.eq(y)

@always_inline
def _gt_na(x: SIMD[F64, W], y: SIMD[F64, W]) -> SIMD[DType.bool, W]:
    return x.gt(y) | (~x.eq(x) & y.eq(y))

@always_inline
def _ge_na(x: SIMD[F64, W], y: SIMD[F64, W]) -> SIMD[DType.bool, W]:
    return x.ge(y) | ~x.eq(x)

@always_inline
def _eq_na(x: SIMD[F64, W], y: SIMD[F64, W]) -> SIMD[DType.bool, W]:
    return x.eq(y) | (~x.eq(x) & ~y.eq(y))


@always_inline
def _bin[OP: Int, NA: Bool](d: FP, a: FP, b: FP, n: Int):
    var one = SIMD[F64, W](1.0)
    var zero = SIMD[F64, W](0.0)
    var i = 0
    while i < n:
        var x = a.load[width=W](i)
        var y = b.load[width=W](i)
        var r: SIMD[F64, W]
        comptime if OP == OP_ADD:
            r = x + y
        elif OP == OP_SUB:
            r = x - y
        elif OP == OP_MUL:
            r = x * y
        elif OP == OP_DIV:
            r = x / y
        elif OP == OP_MOD:
            # Mojo's float % is floored (the sign follows the divisor); fmod's follows the dividend: step back
            # by the divisor when the two signs differ and the remainder is not zero (exact for |x| < 2^53)
            var m = x % y
            r = (m.ne(zero) & (x.lt(zero) ^ y.lt(zero))).select(m - y, m)
        elif OP == OP_LT:
            comptime if NA:
                r = _lt_na(x, y).select(one, zero)
            else:
                r = x.lt(y).select(one, zero)
        elif OP == OP_LE:
            comptime if NA:
                r = _le_na(x, y).select(one, zero)
            else:
                r = x.le(y).select(one, zero)
        elif OP == OP_GT:
            comptime if NA:
                r = _gt_na(x, y).select(one, zero)
            else:
                r = x.gt(y).select(one, zero)
        elif OP == OP_GE:
            comptime if NA:
                r = _ge_na(x, y).select(one, zero)
            else:
                r = x.ge(y).select(one, zero)
        elif OP == OP_EQ:
            comptime if NA:
                r = _eq_na(x, y).select(one, zero)
            else:
                r = x.eq(y).select(one, zero)
        elif OP == OP_NE:
            comptime if NA:
                r = _eq_na(x, y).select(zero, one)
            else:
                r = x.eq(y).select(zero, one)
        elif OP == OP_AND:
            r = (x.ne(zero) & y.ne(zero)).select(one, zero)
        else:
            r = (x.ne(zero) | y.ne(zero)).select(one, zero)
        d.store(i, r)
        i += W

def _binop(op: Int, d: FP, a: FP, b: FP, n: Int, na: Bool):
    if op == OP_ADD:
        _bin[OP_ADD, False](d, a, b, n)
    elif op == OP_SUB:
        _bin[OP_SUB, False](d, a, b, n)
    elif op == OP_MUL:
        _bin[OP_MUL, False](d, a, b, n)
    elif op == OP_DIV:
        _bin[OP_DIV, False](d, a, b, n)
    elif op == OP_MOD:
        _bin[OP_MOD, False](d, a, b, n)
    elif op == OP_LT:
        if na:
            _bin[OP_LT, True](d, a, b, n)
        else:
            _bin[OP_LT, False](d, a, b, n)
    elif op == OP_LE:
        if na:
            _bin[OP_LE, True](d, a, b, n)
        else:
            _bin[OP_LE, False](d, a, b, n)
    elif op == OP_GT:
        if na:
            _bin[OP_GT, True](d, a, b, n)
        else:
            _bin[OP_GT, False](d, a, b, n)
    elif op == OP_GE:
        if na:
            _bin[OP_GE, True](d, a, b, n)
        else:
            _bin[OP_GE, False](d, a, b, n)
    elif op == OP_EQ:
        if na:
            _bin[OP_EQ, True](d, a, b, n)
        else:
            _bin[OP_EQ, False](d, a, b, n)
    elif op == OP_NE:
        if na:
            _bin[OP_NE, True](d, a, b, n)
        else:
            _bin[OP_NE, False](d, a, b, n)
    elif op == OP_AND:
        _bin[OP_AND, False](d, a, b, n)
    else:
        _bin[OP_OR, False](d, a, b, n)

@always_inline
def _cmp_thr[OP: Int, NA: Bool](d: FP, thr: Float64, n: Int):
    """In place: d = 1.0 if d <OP> thr else 0.0 (NA: NaN above every number, as DuckDB)."""
    var t = SIMD[F64, W](thr)
    var one = SIMD[F64, W](1.0)
    var zero = SIMD[F64, W](0.0)
    var i = 0
    while i < n:
        var x = d.load[width=W](i)
        var r: SIMD[F64, W]
        comptime if OP == OP_LT:
            comptime if NA:
                r = _lt_na(x, t).select(one, zero)
            else:
                r = x.lt(t).select(one, zero)
        elif OP == OP_LE:
            comptime if NA:
                r = _le_na(x, t).select(one, zero)
            else:
                r = x.le(t).select(one, zero)
        elif OP == OP_GT:
            comptime if NA:
                r = (x.gt(t) | ~x.eq(x)).select(one, zero)     # (thr is never NaN here: the host keeps NaN constants out of CMPC)
            else:
                r = x.gt(t).select(one, zero)
        elif OP == OP_GE:
            comptime if NA:
                r = _ge_na(x, t).select(one, zero)
            else:
                r = x.ge(t).select(one, zero)
        elif OP == OP_EQ:
            comptime if NA:
                r = _eq_na(x, t).select(one, zero)
            else:
                r = x.eq(t).select(one, zero)
        else:
            comptime if NA:
                r = _eq_na(x, t).select(zero, one)
            else:
                r = x.eq(t).select(zero, one)
        d.store(i, r)
        i += W

def _cmp_const(op: Int, d: FP, thr: Float64, n: Int, na: Bool):
    if op == OP_LT:
        if na:
            _cmp_thr[OP_LT, True](d, thr, n)
        else:
            _cmp_thr[OP_LT, False](d, thr, n)
    elif op == OP_LE:
        if na:
            _cmp_thr[OP_LE, True](d, thr, n)
        else:
            _cmp_thr[OP_LE, False](d, thr, n)
    elif op == OP_GT:
        if na:
            _cmp_thr[OP_GT, True](d, thr, n)
        else:
            _cmp_thr[OP_GT, False](d, thr, n)
    elif op == OP_GE:
        if na:
            _cmp_thr[OP_GE, True](d, thr, n)
        else:
            _cmp_thr[OP_GE, False](d, thr, n)
    elif op == OP_EQ:
        if na:
            _cmp_thr[OP_EQ, True](d, thr, n)
        else:
            _cmp_thr[OP_EQ, False](d, thr, n)
    else:
        if na:
            _cmp_thr[OP_NE, True](d, thr, n)
        else:
            _cmp_thr[OP_NE, False](d, thr, n)

def _not(d: FP, a: FP, n: Int):
    var one = SIMD[F64, W](1.0)
    var zero = SIMD[F64, W](0.0)
    var i = 0
    while i < n:
        d.store(i, a.load[width=W](i).eq(zero).select(one, zero))
        i += W

def _floor(d: FP, a: FP, n: Int):
    var i = 0
    while i < n:
        d.store(i, floor(a.load[width=W](i)))
        i += W

def _f32(d: FP, a: FP, n: Int):
    var i = 0
    while i < n:
        d.store(i, a.load[width=W](i).cast[DType.float32]().cast[DType.float64]())
        i += W

def _select(d: FP, c: FP, a: FP, b: FP, n: Int):
    var zero = SIMD[F64, W](0.0)
    var i = 0
    while i < n:
        d.store(i, c.load[width=W](i).ne(zero).select(a.load[width=W](i), b.load[width=W](i)))
        i += W

def _between(d: FP, x: FP, lo: FP, hi: FP, n: Int):
    var one = SIMD[F64, W](1.0)
    var zero = SIMD[F64, W](0.0)
    var i = 0
    while i < n:
        var v = x.load[width=W](i)
        d.store(i, (v.ge(lo.load[width=W](i)) & v.le(hi.load[width=W](i))).select(one, zero))
        i += W

def _tab(d: FP, a: FP, bytes: BP, off: Int, length: Int, lo: Int, n: Int):
    """OP_TAB: dst = table[a - lo] for the first n lanes ("int" registers: the bits are the values), 0 outside the table."""
    var xa = _ibits(a)
    var dd = _ibits(d)
    var tab = LP(unsafe_from_address=Int(bytes) + off)
    var base = Int64(lo)
    var len_u = UInt64(length)
    for i in range(n):
        var k = xa[i] - base
        dd[i] = tab[Int(k)] if UInt64(k) < len_u else Int64(0)


def _iset(d: FP, a: FP, bytes: BP, off: Int, cap: Int, has_min: Int, n: Int):
    """OP_ISET: dst = 1.0 where the int64 in `a` is in the hash set (open addressing, linear probing, a multiplicative hash)."""
    var xa = _ibits(a)
    var tab = LP(unsafe_from_address=Int(bytes) + off)
    var lg = 0
    while (1 << lg) < cap:
        lg += 1
    var shift = UInt64(64 - lg)
    var mask = cap - 1
    for i in range(n):
        var x = xa[i]
        var hit = False
        if x == Int64.MIN:
            hit = has_min != 0
        else:
            var h = Int((UInt64(x) * UInt64(0x9E3779B97F4A7C15)) >> shift)
            while True:
                var y = tab[h]
                if y == x:
                    hit = True
                    break
                if y == Int64.MIN:
                    break
                h = (h + 1) & mask
        d[i] = 1.0 if hit else 0.0


def _in_set(d: FP, a: FP, bytes: BP, off: Int, length: Int, lo: Int, n: Int):
    for i in range(n):
        var x = a[i] - Float64(lo)
        var hit = 0.0
        if x >= 0.0 and x < Float64(length):
            if bytes[off + Int(x)] != 0:
                hit = 1.0
        d[i] = hit


# ── LIKE over a plain utf8 column (F-034) ──────────────────────────────────────────────────────
# Pattern bytes: UTF-8 literals, 0xFF = '%' (any run), 0xFE = '_' (one UTF-8 character); neither byte
# occurs in UTF-8 text. Byte semantics like DuckDB's: literals compare as bytes, '_' skips one code point.

@always_inline
def _ulen(b: UInt8) -> Int:
    if b < 0x80:
        return 1
    if b < 0xE0:
        return 2
    if b < 0xF0:
        return 3
    return 4


@always_inline
def _eq_bytes(a: BP, b: BP, n: Int) -> Bool:
    for i in range(n):
        if a[i] != b[i]:
            return False
    return True


def _like_general(s: BP, L: Int, pat: BP, pl: Int) -> Bool:
    """Glob match with '%' and '_': one backtrack point (the last '%'), O(len * pattern) at worst."""
    var si = 0
    var pi = 0
    var star = -1
    var mark = 0
    while si < L:
        var pch = Int(pat[pi]) if pi < pl else -1
        if pch == 0xFF:
            star = pi
            pi += 1
            if pi == pl:
                return True
            mark = si
        elif pch == 0xFE:
            si += _ulen(s[si])
            pi += 1
        elif pch >= 0 and pch == Int(s[si]):
            si += 1
            pi += 1
        elif star >= 0:
            mark += _ulen(s[mark])
            si = mark
            pi = star + 1
        else:
            return False
    while pi < pl and pat[pi] == 0xFF:
        pi += 1
    return pi == pl


def _is_null(d: FP, valid_addr: Int, bit_base: Int, i0: Int, sel: IP, use_sel: Bool, n: Int):
    """d[j] = 1.0 where row j is NULL (its validity bit is clear), 0.0 where it is not or the chunk has no bitmap."""
    if valid_addr == 0:
        for j in range(n):
            d[j] = 0.0
        return
    var valid = _bp(valid_addr)
    for j in range(n):
        var bit = bit_base + i0 + (Int(sel[j]) if use_sel else j)
        d[j] = 1.0 if ((Int(valid[bit >> 3]) >> (bit & 7)) & 1) == 0 else 0.0


def _str_like(d: FP, offs_addr: Int, data_addr: Int, valid_addr: Int, bit_base: Int, pat: BP, pl: Int,
              negate: Bool, isin: Bool, i0: Int, sel: IP, use_sel: Bool, n: Int):
    """d[j] = 1.0 if row j's string matches the pattern (negated: does not), 0.0 for a NULL string.
    isin: `pat` is a list of values, each a 4-byte little-endian length and its bytes; a row matches when
    it equals one of them."""
    var offs = IP(unsafe_from_address=offs_addr)
    var data = _bp(data_addr)
    var valid = _bp(valid_addr)
    # the pattern's shape, once per call: 0 general, 1 exact, 2 prefix, 3 suffix, 4 substring
    var markers = 0
    var wild_one = False
    for k in range(0 if isin else pl):
        if pat[k] == 0xFF or pat[k] == 0xFE:
            markers += 1
        if pat[k] == 0xFE:
            wild_one = True
    var shape = 0
    var lit = pat
    var m = pl
    if isin:
        shape = 5
    elif not wild_one:
        if markers == 0:
            shape = 1
        elif markers == 1 and pat[pl - 1] == 0xFF:
            shape = 2
            m = pl - 1
        elif markers == 1 and pat[0] == 0xFF:
            shape = 3
            lit = pat + 1
            m = pl - 1
        elif markers == 2 and pl >= 3 and pat[0] == 0xFF and pat[pl - 1] == 0xFF:
            shape = 4
            lit = pat + 1
            m = pl - 2
    var first = Int(lit[0]) if m > 0 else 0
    for j in range(n):
        var r = i0 + (Int(sel[j]) if use_sel else j)
        if valid_addr != 0:
            var bit = bit_base + r
            if ((Int(valid[bit >> 3]) >> (bit & 7)) & 1) == 0:
                d[j] = 0.0
                continue
        var a = Int(offs[r])
        var L = Int(offs[r + 1]) - a
        var s = data + a
        var hit = False
        if shape == 1:
            hit = L == m and _eq_bytes(s, lit, m)
        elif shape == 2:
            hit = L >= m and _eq_bytes(s, lit, m)
        elif shape == 3:
            hit = L >= m and _eq_bytes(s + (L - m), lit, m)
        elif shape == 5:
            var q = 0
            while q < pl:
                var vl = Int(pat[q]) | (Int(pat[q + 1]) << 8) | (Int(pat[q + 2]) << 16) | (Int(pat[q + 3]) << 24)
                if vl == L and _eq_bytes(s, pat + (q + 4), L):
                    hit = True
                    break
                q += 4 + vl
        elif shape == 4:
            if m == 0:
                hit = True
            else:
                var i = 0
                while i + m <= L:
                    if Int(s[i]) == first and _eq_bytes(s + i, lit, m):
                        hit = True
                        break
                    i += 1
        else:
            hit = _like_general(s, L, pat, pl)
        d[j] = 1.0 if hit != negate else 0.0


comptime HUGE = 1 << 21


def big_alloc(n: Int) -> Int:
    """Address of n bytes that are written right after allocation, in parallel: 2 MiB-aligned and
    marked for transparent huge pages above 2 MiB, so first-touch faults are 512x fewer (24 threads
    faulting 4 KiB pages serialize on the kernel's mm lock: ~25 ms for 50 MB). Free with free_big."""
    if n < HUGE:
        return Int(unsafe_alloc[UInt8](max(n, 1), alignment=64))
    var size = (n + HUGE - 1) // HUGE * HUGE
    var b = unsafe_alloc[UInt8](size, alignment=HUGE)
    comptime if CompilationTarget.is_linux():                         # 14 is Linux's MADV_HUGEPAGE: not Darwin's
        _ = external_call["madvise", Int32](Int(b), size, Int32(14))
    return Int(b)


def free_big(a: Int):
    Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=a).free()


# ── per-run immutable state ────────────────────────────────────────────────

struct Prog(Copyable, Movable):
    var code: IP
    var consts: FP
    var set_bytes: BP
    var set_meta: LP
    var tbl: LP
    var n_chunks: Int
    var tbl_stride: Int
    var dtypes: IP
    var gids: IP
    var n_groups: Int
    var has_gids: Bool
    var n_regs: Int
    var n_pro: Int
    var n_pred: Int
    var n_body: Int
    var n_aggs: Int
    var pred_reg: Int
    var nl: Int      # partial lanes per (agg, group)
    var ovf: LP      # set to 1 by any integer sum (OP_AGG_ISUM) that passes int64; the caller's flag cell

    def __init__(out self, code: IP, consts: FP, set_bytes: BP, set_meta: LP,
                 tbl: LP, n_chunks: Int, dtypes: IP, gids: IP, n_groups: Int, has_gids: Bool, ovf: LP):
        self.ovf = ovf
        self.code = code
        self.consts = consts
        self.set_bytes = set_bytes
        self.set_meta = set_meta
        self.tbl = tbl
        self.n_chunks = n_chunks
        self.tbl_stride = 2 + Int(code[7])
        self.dtypes = dtypes
        self.gids = gids
        self.n_groups = n_groups
        self.has_gids = has_gids
        self.n_regs = Int(code[1])
        self.n_pro = Int(code[2])
        self.n_pred = Int(code[3])
        self.n_body = Int(code[4])
        self.n_aggs = Int(code[5])
        self.pred_reg = Int(code[6])
        if not has_gids:
            self.nl = W
        elif n_groups <= LANE_GROUPS:       # a function of the group count alone: the order in which a
            self.nl = 4                     # group's rows are added must not depend on the aggregate list
        else:
            self.nl = 1

    @always_inline
    def instr(self, base: Int, k: Int) -> IP:
        return self.code + (HDR + (base + k) * 5)


def _run_ops(p: Prog, regs: FP, cols: LP, base: Int, count: Int, i0: Int, sel: IP,
             use_sel: Bool, n: Int):
    var npad = ((n + W - 1) // W) * W
    for k in range(count):
        var ins = p.instr(base, k)
        var op = Int(ins[0])
        var d = regs + Int(ins[1]) * VEC
        if op == OP_LOAD_COL:
            var c = Int(ins[2])
            var dt = Int(p.dtypes[c])
            if (dt & 255) == 5:
                # dictionary codes of this chunk, shifted to its dictionary's place in the
                # concatenation of all chunks' dictionaries (the offset sits in tbl slot dt >> 8)
                _load_col(d, Int(cols[c]), 0, i0, sel, use_sel, n)
                var off = Float64(Int(cols[dt >> 8]))
                for i in range(n):
                    d[i] += off
            else:
                _load_col(d, Int(cols[c]), dt, i0, sel, use_sel, n)
        elif op == OP_LOADI:
            var c = Int(ins[2])
            _load_coli(d, Int(cols[c]), Int(p.dtypes[c]), i0, sel, use_sel, n)
        elif op == OP_LOAD_CONST:
            var v = SIMD[F64, W](p.consts[Int(ins[2])])
            var i = 0
            while i < VEC:
                d.store(i, v)
                i += W
        elif op == OP_CMPC:
            # column <cmp> constant: widen exactly, compare against the pre-rounded threshold
            var c = Int(ins[2])
            _load_col(d, Int(cols[c]), Int(p.dtypes[c]), i0, sel, use_sel, n)
            _cmp_const(Int(ins[4]) & 255, d, p.consts[Int(ins[3])], npad, (Int(ins[4]) >> 8) != 0)
        elif op == OP_STR:
            var c = Int(ins[2])
            var slot = Int(p.dtypes[c]) >> 8        # dtype 6 | (slot << 8): the chunk's bytes, validity, bit base
            var si = Int(ins[3])
            _str_like(d, Int(cols[c]), Int(cols[slot]), Int(cols[slot + 1]), Int(cols[slot + 2]),
                      p.set_bytes + Int(p.set_meta[si * 3]), Int(p.set_meta[si * 3 + 1]), (Int(ins[4]) & 1) != 0,
                      (Int(ins[4]) & 2) != 0, i0, sel, use_sel, n)
        elif op == OP_ISNULL:
            var slot = Int(p.dtypes[Int(ins[2])]) >> 8        # dtype 7 | (slot << 8): validity bitmap, bit base
            _is_null(d, Int(cols[slot]), Int(cols[slot + 1]), i0, sel, use_sel, n)
        elif op == OP_ROWID:
            # i0 is the block's first row in the whole input (the column bases are rebased to it); sel
            # holds the block offsets of the rows kept so far
            for i in range(n):
                d[i] = Float64(i0 + (Int(sel[i]) if use_sel else i))
        elif op >= OP_IADD and op <= OP_I2F:
            if op == OP_IADD:
                _int_bin[OP_IADD](d, regs + Int(ins[2]) * VEC, regs + Int(ins[3]) * VEC, n, p.ovf)
            elif op == OP_ISUB:
                _int_bin[OP_ISUB](d, regs + Int(ins[2]) * VEC, regs + Int(ins[3]) * VEC, n, p.ovf)
            elif op == OP_IMUL:
                _int_bin[OP_IMUL](d, regs + Int(ins[2]) * VEC, regs + Int(ins[3]) * VEC, n, p.ovf)
            elif op == OP_F2I:
                _f2i(d, regs + Int(ins[2]) * VEC, n)
            else:
                _i2f(d, regs + Int(ins[2]) * VEC, n, p.ovf, Int(ins[4]))
        elif op >= OP_IDIV and op <= OP_ICMP:
            var ra = regs + Int(ins[2]) * VEC
            var rb = regs + Int(ins[3]) * VEC
            if op == OP_IDIV:
                _int_op2[OP_IDIV](d, ra, rb, 0, n, p.ovf)
            elif op == OP_IMOD:
                _int_op2[OP_IMOD](d, ra, rb, 0, n, p.ovf)
            elif op == OP_IMIN:
                _int_op2[OP_IMIN](d, ra, rb, 0, n, p.ovf)
            elif op == OP_IMAX:
                _int_op2[OP_IMAX](d, ra, rb, 0, n, p.ovf)
            else:
                _int_op2[OP_ICMP](d, ra, rb, Int(ins[4]), n, p.ovf)
        elif op < OP_NOT:
            _binop(op, d, regs + Int(ins[2]) * VEC, regs + Int(ins[3]) * VEC, npad, Int(ins[4]) != 0)
        elif op == OP_NOT:
            _not(d, regs + Int(ins[2]) * VEC, npad)
        elif op == OP_FLOOR:
            _floor(d, regs + Int(ins[2]) * VEC, npad)
        elif op == OP_F32:
            _f32(d, regs + Int(ins[2]) * VEC, npad)
        elif op == OP_SELECT:
            _select(d, regs + Int(ins[2]) * VEC, regs + Int(ins[3]) * VEC,
                    regs + Int(ins[4]) * VEC, npad)
        elif op == OP_BETWEEN:
            _between(d, regs + Int(ins[2]) * VEC, regs + Int(ins[3]) * VEC,
                     regs + Int(ins[4]) * VEC, npad)
        elif op == OP_IN_SET:
            var s = Int(ins[3])
            _in_set(d, regs + Int(ins[2]) * VEC, p.set_bytes, Int(p.set_meta[s * 3]),
                    Int(p.set_meta[s * 3 + 1]), Int(p.set_meta[s * 3 + 2]), n)
        elif op == OP_TAB:
            var s = Int(ins[3])
            _tab(d, regs + Int(ins[2]) * VEC, p.set_bytes, Int(p.set_meta[s * 3]),
                 Int(p.set_meta[s * 3 + 1]), Int(p.set_meta[s * 3 + 2]), n)
        elif op == OP_ISET:
            var s = Int(ins[3])
            _iset(d, regs + Int(ins[2]) * VEC, p.set_bytes, Int(p.set_meta[s * 3]),
                  Int(p.set_meta[s * 3 + 1]), Int(p.set_meta[s * 3 + 2]), n)
        elif op == OP_FLAG:
            var ra = regs + Int(ins[2]) * VEC
            for i in range(n):
                if ra[i] != 0.0:
                    p.ovf[0] = 1


def _accumulate(p: Prog, regs: FP, gp: IP, n: Int, pv: FP, pc: LP):
    """Accumulate n compacted rows into the worker partials.

    Layout: pv[(a * G + g) * NL + lane], pc[g * NL + lane]; lane = j & (NL-1),
    which breaks the dependency chain when consecutive rows hit one group.
    """
    var G = p.n_groups
    var NL = p.nl
    var A = p.n_aggs
    if not p.has_gids:
        # Global aggregate: NL == W, accumulate straight into SIMD lanes.
        pc[0] += Int64(n)
        for a in range(A):
            var ins = p.instr(p.n_pro + p.n_pred + p.n_body, a)
            var kind = Int(ins[0])
            var r = regs + Int(ins[2]) * VEC
            var acc = pv + a * NL
            if kind == OP_AGG_ISUM:
                # exact: the input is an "int" register, its int64 values added in the SIMD lanes
                var iacc = _ibits(acc)
                var ir = _ibits(r)
                var iv = iacc.load[width=W]()
                var ov = SIMD[DType.int64, W](0)
                var k = 0
                while k + W <= n:
                    var x = ir.load[width=W](k)
                    var t = iv + x
                    ov |= (iv ^ t) & (x ^ t)
                    iv = t
                    k += W
                iacc.store(iv)
                if ov.reduce_or() < 0:
                    p.ovf[0] = 1
                while k < n:
                    iacc[0] = _iadd(iacc[0], ir[k], p.ovf)
                    k += 1
                continue
            if _is_iext(kind):
                # F-060c: the extreme of int64 values (the register holds their bits), in SIMD lanes
                var xacc = _ibits(acc)
                var xr = _ibits(r)
                var xv = xacc.load[width=W]()
                var k = 0
                while k + W <= n:
                    var x = xr.load[width=W](k)
                    xv = min(xv, x) if kind == OP_AGG_IMIN else max(xv, x)
                    k += W
                xacc.store(xv)
                while k < n:
                    xacc[0] = _iext(kind, xacc[0], xr[k])
                    k += 1
                continue
            var accv = acc.load[width=W]()
            var j = 0
            while j + W <= n:
                var x = r.load[width=W](j)
                if kind == OP_AGG_SUM:
                    accv = accv + x
                elif kind == OP_AGG_MIN:
                    accv = min(accv, x)
                else:
                    accv = max(accv, x)
                j += W
            acc.store(accv)
            while j < n:
                var x = r[j]
                if kind == OP_AGG_SUM:
                    acc[0] += x
                elif kind == OP_AGG_MIN:
                    acc[0] = min(acc[0], x)
                else:
                    acc[0] = max(acc[0], x)
                j += 1
        return
    var lm = NL - 1
    for j in range(n):
        var g = Int(gp[j])
        if g >= 0 and g < G:
            pc[g * NL + (j & lm)] += 1
    for a in range(A):
        var ins = p.instr(p.n_pro + p.n_pred + p.n_body, a)
        var kind = Int(ins[0])
        var r = regs + Int(ins[2]) * VEC
        var base = pv + a * G * NL
        if kind == OP_AGG_SUM:
            for j in range(n):
                var g = Int(gp[j])
                if g >= 0 and g < G:
                    base[g * NL + (j & lm)] += r[j]
        elif kind == OP_AGG_ISUM:
            var ib = _ibits(base)
            for j in range(n):
                var g = Int(gp[j])
                if g >= 0 and g < G:
                    var q = ib + (g * NL + (j & lm))
                    q[0] = _iadd(q[0], _ibits(r)[j], p.ovf)
        elif _is_iext(kind):
            var xb = _ibits(base)
            var xr = _ibits(r)
            for j in range(n):
                var g = Int(gp[j])
                if g >= 0 and g < G:
                    var q = xb + (g * NL + (j & lm))
                    q[0] = _iext(kind, q[0], xr[j])
        elif kind == OP_AGG_MIN:
            for j in range(n):
                var g = Int(gp[j])
                if g >= 0 and g < G:
                    var q = base + (g * NL + (j & lm))
                    q[0] = min(q[0], r[j])
        else:
            for j in range(n):
                var g = Int(gp[j])
                if g >= 0 and g < G:
                    var q = base + (g * NL + (j & lm))
                    q[0] = max(q[0], r[j])


def _stage(p: Prog, regs: FP, gp: IP, n: Int, sv: FP, sc: LP):
    """Append the n compacted rows of a block to the morsel's slot (staged output, see _pipeline):
    sc[0] rows so far, sc[1] the address of the slot's group ids, sv[a * MORSEL + i] the value of
    row i for aggregate a. Rows keep their order; nothing is added to a float here."""
    var G = p.n_groups
    var k = Int(sc[0])
    var k0 = k
    var sg = _ip(Int(sc[1]))
    for j in range(n):
        var g = Int(gp[j])
        if g >= 0 and g < G:
            sg[k] = Int32(g)
            k += 1
    for a in range(p.n_aggs):
        var r = regs + Int(p.instr(p.n_pro + p.n_pred + p.n_body, a)[2]) * VEC
        var q = sv + a * MORSEL + k0
        var i = 0
        for j in range(n):
            var g = Int(gp[j])
            if g >= 0 and g < G:
                q[i] = r[j]
                i += 1
    sc[0] = Int64(k)


def _init_values(p: Prog, pv: FP):
    """Identity of every aggregate's slot for one morsel's table (the row counts are kept elsewhere)."""
    var n = p.n_groups * p.nl
    for a in range(p.n_aggs):
        var kind = Int(p.instr(p.n_pro + p.n_pred + p.n_body, a)[0])
        _fill_identity(pv + a * n, n, kind)


comptime SC = 4096          # per-worker cache of groups in front of the shared output (power of 2)


def _flush_slot(p: Prog, cv: FP, cl: LP, s: Int):
    """Add cache slot s into the shared output (atomically) and empty it."""
    var g = Int(cl[2 + s])
    if g < 0:
        return
    var G = p.n_groups
    var ov = _fp(Int(cl[0]))
    var oc = _lp(Int(cl[1]))
    _ = Atomic[Scalar[DType.int64]].fetch_add(oc + g, cl[2 + SC + s])
    for a in range(p.n_aggs):
        var kind = Int(p.instr(p.n_pro + p.n_pred + p.n_body, a)[0])
        var x = cv[a * SC + s]
        if kind == OP_AGG_SUM:
            _ = Atomic[Scalar[DType.float64]].fetch_add(ov + a * G + g, x)
        elif kind == OP_AGG_ISUM:
            var xi = _ibits(cv + (a * SC + s))[0]
            var old = Atomic[Scalar[DType.int64]].fetch_add(_ibits(ov + a * G + g), xi)
            _ = _iadd(old, xi, p.ovf)
        elif kind == OP_AGG_IMIN:
            _ = Atomic[Scalar[DType.int64]].min(_ibits(ov + a * G + g), _ibits(cv + (a * SC + s))[0])
        elif kind == OP_AGG_IMAX:
            _ = Atomic[Scalar[DType.int64]].max(_ibits(ov + a * G + g), _ibits(cv + (a * SC + s))[0])
        elif kind == OP_AGG_MIN:
            _ = Atomic[Scalar[DType.float64]].min(ov + a * G + g, x)
        else:
            _ = Atomic[Scalar[DType.float64]].max(ov + a * G + g, x)
    cl[2 + s] = -1


def _flush_all(p: Prog, cv: FP, cl: LP):
    for s in range(SC):
        _flush_slot(p, cv, cl, s)


def _accumulate_shared(p: Prog, regs: FP, gp: IP, n: Int, cv: FP, cl: LP):
    """Accumulate n compacted rows for the output shared by all workers (group counts too large for
    a partial table per worker). Rows go through this worker's small direct-mapped cache of groups
    (cl: out_vals and out_cnt addresses, SC tags, SC counts; cv: SC values per aggregate): a repeat
    of a cached group is added locally, another group evicts the slot into the output with atomic
    add / min / max. Skewed keys stay in the cache instead of every worker retrying atomics on one
    hot group."""
    var G = p.n_groups
    var A = p.n_aggs
    var st = cl + (2 + 2 * SC)                     # hits, rows seen, direct (1: cache bypassed)
    if Int(st[2]) != 0:
        # keys spread over many groups: the cache only adds evictions, every row goes straight in
        var ov = _fp(Int(cl[0]))
        var oc = _lp(Int(cl[1]))
        for j in range(n):
            var g = Int(gp[j])
            if g < 0 or g >= G:
                continue
            _ = Atomic[Scalar[DType.int64]].fetch_add(oc + g, 1)
            for a in range(A):
                var ins = p.instr(p.n_pro + p.n_pred + p.n_body, a)
                var kind = Int(ins[0])
                var x = regs[Int(ins[2]) * VEC + j]
                if kind == OP_AGG_SUM:
                    _ = Atomic[Scalar[DType.float64]].fetch_add(ov + a * G + g, x)
                elif kind == OP_AGG_ISUM:
                    var xi = _ibits(regs + Int(ins[2]) * VEC)[j]
                    var old = Atomic[Scalar[DType.int64]].fetch_add(_ibits(ov + a * G + g), xi)
                    _ = _iadd(old, xi, p.ovf)
                elif kind == OP_AGG_IMIN:
                    _ = Atomic[Scalar[DType.int64]].min(_ibits(ov + a * G + g), _ibits(regs + Int(ins[2]) * VEC)[j])
                elif kind == OP_AGG_IMAX:
                    _ = Atomic[Scalar[DType.int64]].max(_ibits(ov + a * G + g), _ibits(regs + Int(ins[2]) * VEC)[j])
                elif kind == OP_AGG_MIN:
                    _ = Atomic[Scalar[DType.float64]].min(ov + a * G + g, x)
                else:
                    _ = Atomic[Scalar[DType.float64]].max(ov + a * G + g, x)
        return
    for j in range(n):
        var g = Int(gp[j])
        if g < 0 or g >= G:
            continue
        var s = Int((UInt64(g) * 0x9E3779B97F4A7C15) >> 52)        # top 12 bits: SC slots
        st[1] += 1
        if Int(cl[2 + s]) != g:
            _flush_slot(p, cv, cl, s)
            cl[2 + s] = Int64(g)
            cl[2 + SC + s] = 1
            for a in range(A):
                var ins = p.instr(p.n_pro + p.n_pred + p.n_body, a)
                var x0 = regs[Int(ins[2]) * VEC + j]
                if Int(ins[0]) == OP_AGG_ISUM or _is_iext(Int(ins[0])):
                    _ibits(cv + (a * SC + s))[0] = _ibits(regs + Int(ins[2]) * VEC)[j]
                else:
                    cv[a * SC + s] = x0
            continue
        st[0] += 1
        cl[2 + SC + s] += 1
        for a in range(A):
            var ins = p.instr(p.n_pro + p.n_pred + p.n_body, a)
            var kind = Int(ins[0])
            var x = regs[Int(ins[2]) * VEC + j]
            var q = cv + (a * SC + s)
            if kind == OP_AGG_SUM:
                q[0] += x
            elif kind == OP_AGG_ISUM:
                var iq = _ibits(q)
                iq[0] = _iadd(iq[0], _ibits(regs + Int(ins[2]) * VEC)[j], p.ovf)
            elif _is_iext(kind):
                var iq = _ibits(q)
                iq[0] = _iext(kind, iq[0], _ibits(regs + Int(ins[2]) * VEC)[j])
            elif kind == OP_AGG_MIN:
                q[0] = min(q[0], x)
            else:
                q[0] = max(q[0], x)
    if Int(st[1]) >= 65536 and Int(st[0]) * 16 < Int(st[1]):    # under 1/16 hits: keys spread out
        _flush_all(p, cv, cl)
        st[2] = 1


def _init_partials(p: Prog, pv: FP, pc: LP):
    var G = p.n_groups
    var NL = p.nl
    for a in range(p.n_aggs):
        var kind = Int(p.instr(p.n_pro + p.n_pred + p.n_body, a)[0])
        _fill_identity(pv + a * G * NL, G * NL, kind)
    for i in range(G * NL):
        pc[i] = 0


def _worker_state(p: Prog, ws_bytes: Int, st: LP):
    """A worker's block state (registers, selection and group-id scratch, the fused front stage's scratch),
    allocated by the worker itself on its first morsel (in parallel, not in the calling thread) and kept
    for the waves after it: st[0..3] = regs, sel, gscratch, ws."""
    var regs = unsafe_alloc[Float64](p.n_regs * VEC + W, alignment=64)
    for i in range(p.n_regs * VEC + W):
        regs[i] = 0.0
    var sel = unsafe_alloc[Int32](VEC)
    var gscratch = unsafe_alloc[Int32](VEC)
    var ws = unsafe_alloc[UInt8](max(ws_bytes, 64), alignment=64)      # the fused front stage's block state
    unsafe_memset_zero(ws, max(ws_bytes, 64))
    _run_ops(p, regs, p.tbl, 0, p.n_pro, 0, sel, False, VEC)           # prologue (constants)
    st[0] = Int64(Int(regs))
    st[1] = Int64(Int(sel))
    st[2] = Int64(Int(gscratch))
    st[3] = Int64(Int(ws))


def _find_chunk(p: Prog, row: Int) -> Int:
    """Index of the chunk holding `row` (binary search on row_start)."""
    var lo = 0
    var hi = p.n_chunks - 1
    while lo < hi:
        var mid = (lo + hi + 1) // 2
        if Int(p.tbl[mid * p.tbl_stride]) <= row:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _process_range[E: def(Int, Int, IP, Int, Bool, IP, FP) -> Bool, keyed: Bool, mode: Int,
                   FR: def(Int, Int, Int, IP, BP) -> Int, fused: Bool](
    p: Prog, regs: FP, sel: IP, gscratch: IP, pv: FP, pc: LP, lo: Int, hi: Int, encode: E,
    front: FR, ws: BP,
) -> Bool:
    """Rows [lo, hi) in VEC blocks: predicate, selection, body, accumulate. `keyed`: group ids of
    the selected rows only, from `encode(chunk, row0, sel, nsel, use_sel, dst, regs)` (regs: the program's registers, for keys it computed) (False: bad key).
    `fused` (F-046a slice 3, join + aggregate): the rows come from `front(chunk, row0, n, sel, ws)`,
    which filters the block, looks the join partners up and fills `sel` with the surviving rows (their
    count is returned) and the worker's column table at `ws` (the partner columns of those rows
    gathered into block scratch); the program's predicate then runs on those rows only."""
    var pos = lo
    while pos < hi:
        var k = _find_chunk(p, pos)
        var row = p.tbl + k * p.tbl_stride
        var cols = row + 2
        var stop = min(hi, Int(row[1]))
        var i0 = pos
        while i0 < stop:
            var n = min(VEC, stop - i0)
            var nsel = n
            var use_sel = False
            var bcols = cols
            comptime if fused:
                nsel = front(k, i0, n, sel, ws)
                bcols = LP(unsafe_from_address=Int(ws))
                use_sel = nsel < n
                if nsel > 0 and p.pred_reg >= 0:
                    _run_ops(p, regs, bcols, p.n_pro, p.n_pred, i0, sel, use_sel, nsel)
                    var pr = regs + p.pred_reg * VEC
                    var kk = 0
                    for j in range(nsel):
                        sel[kk] = sel[j] if use_sel else Int32(j)
                        kk += 1 if pr[j] != 0.0 else 0
                    nsel = kk
                    use_sel = nsel < n
            else:
                if p.pred_reg >= 0:
                    _run_ops(p, regs, cols, p.n_pro, p.n_pred, i0, sel, False, n)
                    var pr = regs + p.pred_reg * VEC
                    var kk = 0
                    for i in range(n):
                        sel[kk] = Int32(i)
                        kk += 1 if pr[i] != 0.0 else 0
                    nsel = kk
                    use_sel = nsel < n
            if nsel > 0:
                _run_ops(p, regs, bcols, p.n_pro + p.n_pred, p.n_body, i0, sel, use_sel, nsel)
                var gp = gscratch
                comptime if keyed:
                    if not encode(k, i0, sel, nsel, use_sel, gscratch, regs):
                        return False
                else:
                    if p.has_gids:
                        var gsrc = p.gids + i0
                        if use_sel:
                            for j in range(nsel):
                                gscratch[j] = gsrc[Int(sel[j])]
                        else:
                            gp = gsrc
                comptime if mode == 2:
                    _stage(p, regs, gp, nsel, pv, pc)
                elif mode == 1:
                    _accumulate_shared(p, regs, gp, nsel, pv, pc)
                else:
                    _accumulate(p, regs, gp, nsel, pv, pc)
            i0 += n
        pos = stop
    return True


def _pipeline[E: def(Int, Int, IP, Int, Bool, IP, FP) -> Bool, keyed: Bool,
              FR: def(Int, Int, Int, IP, BP) -> Int, fused: Bool](
    code_a: Int, consts_a: Int, set_bytes_a: Int, set_meta_a: Int,
    tbl_a: Int, n_chunks: Int, dtypes_a: Int, gids_a: Int, n_groups: Int,
    start: Int, end: Int, out_vals_a: Int, out_cnt_a: Int, threads: Int, encode: E,
    front: FR, ws_bytes: Int, repro: Bool,
) -> Int:
    """`_pipeline_run`, then E_OVERFLOW when an integer sum (OP_AGG_ISUM) passed int64 anywhere on the way:
    the totals are then wrong, and the caller's checked path (Arrow) answers or raises."""
    var ovf = unsafe_alloc[Int64](1)
    ovf[0] = 0
    var rc = _pipeline_run[keyed=keyed, fused=fused](
        code_a, consts_a, set_bytes_a, set_meta_a, tbl_a, n_chunks, dtypes_a, gids_a, n_groups, start, end,
        out_vals_a, out_cnt_a, threads, encode, front, ws_bytes, repro, ovf)
    if rc == 0 and ovf[0] != 0:
        rc = E_OVERFLOW
    ovf.free()
    return rc


def _pipeline_run[E: def(Int, Int, IP, Int, Bool, IP, FP) -> Bool, keyed: Bool,
                  FR: def(Int, Int, Int, IP, BP) -> Int, fused: Bool](
    code_a: Int, consts_a: Int, set_bytes_a: Int, set_meta_a: Int,
    tbl_a: Int, n_chunks: Int, dtypes_a: Int, gids_a: Int, n_groups: Int,
    start: Int, end: Int, out_vals_a: Int, out_cnt_a: Int, threads: Int, encode: E,
    front: FR, ws_bytes: Int, repro: Bool, ovf: LP,
) -> Int:
    """The interpreter over rows [start, end). `keyed`: no group-id array; the group ids of the rows
    that pass the predicate are made per VEC block by `encode` (see _process_range), so keys of
    filtered-out rows are never read and no n_rows-sized array is written. -5 when `encode` fails."""
    var code = _ip(code_a)
    if Int(code[0]) != 0x4d585031:
        return -1
    var has_gids = keyed or gids_a != 0
    if (not has_gids) and n_groups != 1:
        return -2
    var p = Prog(code, _fp(consts_a), _bp(set_bytes_a), _lp(set_meta_a), _lp(tbl_a),
                 n_chunks, _ip(dtypes_a), _ip(gids_a), n_groups, has_gids, ovf)
    if p.n_regs > MAX_REGS:
        return -3
    if n_chunks < 1 and end > start:
        return -4
    var A = p.n_aggs
    var G = n_groups
    var NL = p.nl
    var out_vals = _fp(out_vals_a)
    var out_cnt = _lp(out_cnt_a)

    # identities for the output
    for a in range(A):
        var kind = Int(p.instr(p.n_pro + p.n_pred + p.n_body, a)[0])
        _fill_identity(out_vals + a * G, G, kind)
    for g in range(G):
        out_cnt[g] = 0
    var total = end - start
    if total <= 0:
        return 0

    # Which implementation (F-064). Reproducible mode (`compute(reproducible=True)`, plan flag `repro`)
    # always uses the defined order below. The default keeps main's speed everywhere it was measured
    # (A/B, 6M rows, 12 threads, quiet machine): the previous code (a partial table per worker, or the
    # shared output with atomic adds; its float sums can differ in the last bits between runs) runs every
    # keyed aggregate and every fused join under FUSED_MIN groups, where a table per morsel costs 10-20%
    # more; the defined order runs the global aggregate (no groups: as fast) and fused joins from
    # FUSED_MIN groups (up to 2x faster than the atomic shared output it replaces).
    var nt0 = max(1, min(threads if threads > 0 else 1, (total + MORSEL - 1) // MORSEL))
    var mo0 = (total + MORSEL - 1) // MORSEL
    var legacy = has_gids and not repro and (not fused or G < FUSED_MIN)
    # the table path needs a wave of tables (two per worker, or all of them when fewer) and a count table
    # per worker inside the budget; one wave when all tables fit
    var table_bytes0 = A * G * NL * 8
    var table_ok = (not has_gids) or (min(mo0, 2 * nt0) * table_bytes0 + nt0 * G * NL * 8 <= BUDGET_BYTES
                                      and G <= (TABLE_GROUPS if repro else FUSED_GROUPS))
    if legacy:
        p.nl = 4 if G * (A + 1) <= 2048 else 1          # the previous lane rule (depends on the aggregate list)
        NL = p.nl
        var per_worker = (A + 1) * G * NL * 8
        var nthreads = threads if threads > 0 else 1
        var morsels = (total + MORSEL - 1) // MORSEL
        nthreads = min(nthreads, morsels)
        if per_worker > 0:
            # every worker initializes and merges a partial per group: with many groups, fewer workers
            # (at most rows / groups of them) beat the extra passes over a large partial each
            nthreads = min(nthreads, max(1, BUDGET_BYTES // per_worker))
            nthreads = min(nthreads, max(1, total // max(G, 1)))
        nthreads = max(nthreads, 1)
        # Too many groups for a partial per worker (the caps above leave under half the workers): all
        # workers update the output itself with atomic add / min / max; no partials, no merge
        var want = max(1, min(threads if threads > 0 else 1, morsels))
        var shared = A > 0 and want >= 4 and nthreads * 2 <= want and total >= SHARED_MIN_ROWS
        if shared:
            nthreads = want

        var pv_all = _fp(big_alloc(1 if shared else nthreads * max(A * G * NL, 1) * 8))
        var pc_all = _lp(big_alloc(1 if shared else nthreads * max(G * NL, 1) * 8))
        var counter = unsafe_alloc[Int64](1)
        counter[0] = 0
        var failed = unsafe_alloc[Int64](1)
        failed[0] = 0
        var pv_stride = max(A * G * NL, 1)
        var pc_stride = max(G * NL, 1)

        def lworker(wid: Int) {imm}:
            var regs = unsafe_alloc[Float64](p.n_regs * VEC + W, alignment=64)
            for i in range(p.n_regs * VEC + W):
                regs[i] = 0.0
            var sel = unsafe_alloc[Int32](VEC)
            var gscratch = unsafe_alloc[Int32](VEC)
            var ws = unsafe_alloc[UInt8](max(ws_bytes, 64), alignment=64)      # the fused front stage's block state
            unsafe_memset_zero(ws, max(ws_bytes, 64))
            var pv = pv_all + wid * pv_stride
            var pc = pc_all + wid * pc_stride
            if shared:                                 # this worker's group cache (see _accumulate_shared)
                pc = unsafe_alloc[Int64](2 + 2 * SC + 3)
                pv = unsafe_alloc[Float64](max(A, 1) * SC)
                pc[0] = Int64(Int(out_vals))
                pc[1] = Int64(Int(out_cnt))
                for s in range(SC):
                    pc[2 + s] = -1
                for s in range(3):
                    pc[2 + 2 * SC + s] = 0
            else:
                _init_partials(p, pv, pc)
            # prologue (constants) once per worker
            _run_ops(p, regs, p.tbl, 0, p.n_pro, 0, sel, False, VEC)
            while True:
                var m = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
                var lo = start + m * MORSEL
                if lo >= end:
                    break
                var ok = True
                if shared:
                    ok = _process_range[keyed=keyed, mode=1, fused=fused](
                        p, regs, sel, gscratch, pv, pc, lo, min(lo + MORSEL, end), encode, front, ws)
                else:
                    ok = _process_range[keyed=keyed, mode=0, fused=fused](
                        p, regs, sel, gscratch, pv, pc, lo, min(lo + MORSEL, end), encode, front, ws)
                if not ok:
                    failed[0] = 1
                    break
            if shared:
                _flush_all(p, pv, pc)
                pv.free()
                pc.free()
            regs.free()
            sel.free()
            gscratch.free()
            ws.free()

        if nthreads == 1:
            lworker(0)
        else:
            # A host process (Python via ctypes) has no Mojo async runtime yet:
            # create the global CPU device (thread pool) once, then parallelize.
            start_pool()
            par(lworker, nthreads)

        # merge worker partials into the output (parallel over group ranges when there are many groups)
        def lmerge(chunk: Int) {imm}:
            var g0 = chunk * LEGACY_MERGE_GROUPS
            var g1 = min(g0 + LEGACY_MERGE_GROUPS, G)
            for a in range(A):
                var kind = Int(p.instr(p.n_pro + p.n_pred + p.n_body, a)[0])
                if kind == OP_AGG_ISUM:
                    var iout = _ibits(out_vals)
                    for g in range(g0, g1):
                        var iacc = iout[a * G + g]
                        for t in range(nthreads):
                            var iq = _ibits(pv_all + t * pv_stride + (a * G + g) * NL)
                            for l in range(NL):
                                iacc = _iadd(iacc, iq[l], p.ovf)
                        iout[a * G + g] = iacc
                    continue
                if _is_iext(kind):
                    var xout = _ibits(out_vals)
                    for g in range(g0, g1):
                        var xacc = xout[a * G + g]
                        for t in range(nthreads):
                            var xq = _ibits(pv_all + t * pv_stride + (a * G + g) * NL)
                            for l in range(NL):
                                xacc = _iext(kind, xacc, xq[l])
                        xout[a * G + g] = xacc
                    continue
                for g in range(g0, g1):
                    var acc = out_vals[a * G + g]
                    for t in range(nthreads):
                        var q = pv_all + t * pv_stride + (a * G + g) * NL
                        for l in range(NL):
                            if kind == OP_AGG_SUM:
                                acc += q[l]
                            elif kind == OP_AGG_MIN:
                                acc = min(acc, q[l])
                            else:
                                acc = max(acc, q[l])
                    out_vals[a * G + g] = acc
            for g in range(g0, g1):
                var c = Int64(0)
                for t in range(nthreads):
                    var q = pc_all + t * pc_stride + g * NL
                    for l in range(NL):
                        c += q[l]
                out_cnt[g] = c

        var merge_chunks = 0 if shared else (G + LEGACY_MERGE_GROUPS - 1) // LEGACY_MERGE_GROUPS
        if nthreads > 1 and merge_chunks > 1:
            par_n(lmerge, merge_chunks, nthreads)
        else:
            for chunk in range(merge_chunks):
                lmerge(chunk)

        free_big(Int(pv_all))
        free_big(Int(pc_all))
        counter.free()
        var rc = -5 if Int(failed[0]) != 0 else 0
        failed.free()
        return rc

    var nthreads = nt0
    var morsels = mo0

    # Reproducible sums (F-064). Float adds are not associative, so a group's value is defined by an
    # order that depends on neither the workers, the thread count nor the aggregate list: for each
    # morsel (MORSEL rows from `start`), each group's rows are added in row order into a sub-total
    # (`nl` lanes, a function of the group count only, summed in lane order), and the morsels'
    # sub-totals are added in morsel order. Two ways to get there, with the same result:
    #  - a table of sub-totals per MORSEL: morsels run in waves of `nslots` (all of them when their tables fit
    #    the memory budget, so usually one wave, never more than the budget whatever the number of rows); workers claim a wave's morsels dynamically (work
    #    stealing, so a busy core does not stall the tail) and fill that morsel's table; after each wave
    #    the tables are folded into the output in morsel order, in parallel over group ranges;
    #  - staged (too many groups for that): see below.
    # (Row counts are integers, so they are kept per worker and summed in any order.)
    var staged = has_gids and not table_ok

    var nslots = min(morsels, (BUDGET_BYTES - nthreads * G * NL * 8) // max(table_bytes0, 1))
    var pv_all = _fp(big_alloc(1 if staged else nslots * max(A * G * NL, 1) * 8))
    var pc_all = _lp(big_alloc(1 if staged else nthreads * max(G * NL, 1) * 8))
    var counter = unsafe_alloc[Int64](1)
    counter[0] = 0
    var failed = unsafe_alloc[Int64](1)
    failed[0] = 0
    var pv_stride = max(A * G * NL, 1)
    var pc_stride = max(G * NL, 1)

    if staged:
        # Morsels run in waves of `nslots`: (1) workers claim the wave's morsels dynamically, run them
        # into a worker-local staging area (group id and one value per aggregate of each surviving
        # row, in row order) and then copy the rows, grouped by group range (`nbins` ranges, stable),
        # into the morsel's own slot; (2) workers claim group ranges and each range is applied by one
        # worker, slot by slot in morsel order: a group's rows of one slot are added in row order into
        # a sub-total that is then added to the output. No atomics.
        var nbins_target = min(4 * nthreads, 256)
        var shift = 0
        while ((G - 1) >> shift) >= nbins_target:
            shift += 1
        var nbins = ((G - 1) >> shift) + 1
        var offs_bytes = ((nbins + 1) * 8 + 63) // 64 * 64
        var ids_bytes = MORSEL * 4
        var stage_bytes = ids_bytes + A * MORSEL * 8 + nbins * 8
        var slot_bytes = offs_bytes + ids_bytes + A * MORSEL * 8        # offsets, group ids, values
        var nslots = max(2, min(2 * nthreads, (BUDGET_BYTES // 2 - nthreads * stage_bytes) // slot_bytes))
        nslots = min(nslots, morsels)
        var slots = _bp(big_alloc(nslots * slot_bytes))
        # per worker: regs, sel, gscratch, ws, [row count, ids address], staged ids, staged values, bin cursors
        var wst = unsafe_alloc[Int64](nthreads * 8)             # made by each worker on its first morsel
        for i in range(nthreads * 8):
            wst[i] = 0
        var ctl = unsafe_alloc[Int64](2)             # first morsel of the wave, morsels in it

        def stage_morsels(wid: Int) {imm}:
            var st = wst + wid * 8
            if Int(st[0]) == 0:
                _worker_state(p, ws_bytes, st)
                var sc0 = unsafe_alloc[Int64](2)
                var sids0 = unsafe_alloc[Int32](MORSEL)
                sc0[0] = 0
                sc0[1] = Int64(Int(sids0))
                st[4] = Int64(Int(sc0))
                st[5] = Int64(Int(sids0))
                st[6] = Int64(Int(unsafe_alloc[Float64](max(A, 1) * MORSEL)))
                st[7] = Int64(Int(unsafe_alloc[Int64](nbins)))
            var regs = _fp(Int(st[0]))
            var sel = _ip(Int(st[1]))
            var gscratch = _ip(Int(st[2]))
            var ws = _bp(Int(st[3]))
            var sc = _lp(Int(st[4]))
            var sids = _ip(Int(st[5]))
            var sv = _fp(Int(st[6]))
            var cur = _lp(Int(st[7]))
            while True:
                var i = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
                if i >= Int(ctl[1]):
                    break
                var lo = start + (Int(ctl[0]) + i) * MORSEL
                sc[0] = 0
                if not _process_range[keyed=keyed, mode=2, fused=fused](
                    p, regs, sel, gscratch, sv, sc, lo, min(lo + MORSEL, end), encode, front, ws):
                    failed[0] = 1
                    break
                var n = Int(sc[0])
                var base = slots + i * slot_bytes
                var offs = _lp(Int(base))
                var gids = _ip(Int(base) + offs_bytes)
                var vals = _fp(Int(base) + offs_bytes + ids_bytes)
                for b in range(nbins + 1):
                    offs[b] = 0
                for r in range(n):
                    offs[1 + (Int(sids[r]) >> shift)] += 1
                for b in range(nbins):
                    offs[b + 1] += offs[b]
                    cur[b] = offs[b]
                for r in range(n):
                    var g = Int(sids[r])
                    var b = g >> shift
                    var pos = Int(cur[b])
                    cur[b] = Int64(pos + 1)
                    gids[pos] = Int32(g)
                    for a in range(A):
                        vals[a * MORSEL + pos] = sv[a * MORSEL + r]

        def apply_bins(wid: Int) {imm}:
            var width = 1 << shift
            var sub = unsafe_alloc[Float64](max(A, 1) * width)        # a slot's sub-totals of the range
            var cnt = unsafe_alloc[Int32](width)
            var touched = unsafe_alloc[Int32](width)
            for i in range(max(A, 1) * width):
                sub[i] = 0.0
            for i in range(width):
                cnt[i] = 0
            while True:
                var b = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
                if b >= nbins:
                    break
                var g0 = b << shift
                for i in range(Int(ctl[1])):
                    var base = slots + i * slot_bytes
                    var offs = _lp(Int(base))
                    var gids = _ip(Int(base) + offs_bytes)
                    var vals = _fp(Int(base) + offs_bytes + ids_bytes)
                    var r0 = Int(offs[b])
                    var r1 = Int(offs[b + 1])
                    var nt = 0
                    for r in range(r0, r1):
                        var g = Int(gids[r]) - g0
                        if cnt[g] == 0:
                            touched[nt] = Int32(g)
                            nt += 1
                        cnt[g] += 1
                    for a in range(A):
                        var kind = Int(p.instr(p.n_pro + p.n_pred + p.n_body, a)[0])
                        var q = vals + a * MORSEL
                        var o = out_vals + a * G + g0
                        var sb = sub + a * width
                        if kind == OP_AGG_SUM:
                            for r in range(r0, r1):
                                sb[Int(gids[r]) - g0] += q[r]
                        elif kind == OP_AGG_ISUM:
                            var isb = _ibits(sb)
                            for r in range(r0, r1):
                                var at = Int(gids[r]) - g0
                                isb[at] = _iadd(isb[at], _ibits(q)[r], p.ovf)
                        elif _is_iext(kind):
                            var xo = _ibits(o)
                            var xq = _ibits(q)
                            for r in range(r0, r1):
                                var t = xo + (Int(gids[r]) - g0)
                                t[0] = _iext(kind, t[0], xq[r])
                        elif kind == OP_AGG_MIN:
                            for r in range(r0, r1):
                                var t = o + (Int(gids[r]) - g0)
                                t[0] = min(t[0], q[r])
                        else:
                            for r in range(r0, r1):
                                var t = o + (Int(gids[r]) - g0)
                                t[0] = max(t[0], q[r])
                    for u in range(nt):
                        var g = Int(touched[u])
                        out_cnt[g0 + g] += Int64(cnt[g])
                        cnt[g] = 0
                    for a in range(A):
                        var akind = Int(p.instr(p.n_pro + p.n_pred + p.n_body, a)[0])
                        if akind == OP_AGG_SUM:
                            var o = out_vals + a * G + g0
                            var sb = sub + a * width
                            for u in range(nt):
                                var g = Int(touched[u])
                                o[g] += sb[g]
                                sb[g] = 0.0
                        elif akind == OP_AGG_ISUM:
                            var io = _ibits(out_vals + a * G + g0)
                            var isb = _ibits(sub + a * width)
                            for u in range(nt):
                                var g = Int(touched[u])
                                io[g] = _iadd(io[g], isb[g], p.ovf)
                                isb[g] = 0
            sub.free()
            cnt.free()
            touched.free()

        start_pool()
        var wave0 = 0
        while wave0 < morsels and failed[0] == 0:
            ctl[0] = Int64(wave0)
            ctl[1] = Int64(min(nslots, morsels - wave0))
            counter[0] = 0
            par(stage_morsels, nthreads)
            if failed[0] != 0:
                break
            counter[0] = 0
            par(apply_bins, nthreads)
            wave0 += nslots
        for w in range(nthreads):
            var st = wst + w * 8
            if Int(st[0]) != 0:
                for f in range(8):
                    Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(st[f])).free()
        wst.free()
        ctl.free()
        free_big(Int(slots))
        free_big(Int(pv_all))
        free_big(Int(pc_all))
        counter.free()
        var rc_s = -5 if Int(failed[0]) != 0 else 0
        failed.free()
        return rc_s

    # per worker: regs, sel, gscratch, ws (made by the worker on its first morsel; the waves reuse them)
    var wst = unsafe_alloc[Int64](nthreads * 4)
    for i in range(nthreads * 4):
        wst[i] = 0
    var ctl = unsafe_alloc[Int64](2)                 # first morsel of the wave, morsels in it

    def worker(wid: Int) {imm}:
        var st = wst + wid * 4
        var pc = pc_all + wid * pc_stride
        if Int(st[0]) == 0:
            _worker_state(p, ws_bytes, st)
            for i in range(pc_stride):                                     # this worker's row counts
                pc[i] = 0
        var regs = _fp(Int(st[0]))
        var sel = _ip(Int(st[1]))
        var gscratch = _ip(Int(st[2]))
        var ws = _bp(Int(st[3]))
        while True:
            var j = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
            if j >= Int(ctl[1]):
                break
            var lo = start + (Int(ctl[0]) + j) * MORSEL
            var pv = pv_all + j * pv_stride
            _init_values(p, pv)
            if not _process_range[keyed=keyed, mode=0, fused=fused](
                p, regs, sel, gscratch, pv, pc, lo, min(lo + MORSEL, end), encode, front, ws):
                failed[0] = 1
                break

    # fold the wave's tables into the output, in morsel order (parallel over group ranges)
    def merge(chunk: Int) {imm}:
        var g0 = chunk * MERGE_GROUPS
        var g1 = min(g0 + MERGE_GROUPS, G)
        for a in range(A):
            var kind = Int(p.instr(p.n_pro + p.n_pred + p.n_body, a)[0])
            var o = out_vals + a * G
            for j in range(Int(ctl[1])):
                var q = pv_all + j * pv_stride + a * G * NL
                if kind == OP_AGG_ISUM:
                    var iq = _ibits(q)
                    var io = _ibits(o)
                    for g in range(g0, g1):
                        var xi = iq[g * NL]
                        for l in range(1, NL):
                            xi = _iadd(xi, iq[g * NL + l], p.ovf)
                        io[g] = _iadd(io[g], xi, p.ovf)
                    continue
                if _is_iext(kind):
                    var xq = _ibits(q)
                    var xo = _ibits(o)
                    for g in range(g0, g1):
                        var xi = xq[g * NL]
                        for l in range(1, NL):
                            xi = _iext(kind, xi, xq[g * NL + l])
                        xo[g] = _iext(kind, xo[g], xi)
                    continue
                for g in range(g0, g1):
                    var x = q[g * NL]
                    for l in range(1, NL):
                        if kind == OP_AGG_SUM:
                            x += q[g * NL + l]
                        elif kind == OP_AGG_MIN:
                            x = min(x, q[g * NL + l])
                        else:
                            x = max(x, q[g * NL + l])
                    if kind == OP_AGG_SUM:
                        o[g] += x
                    elif kind == OP_AGG_MIN:
                        o[g] = min(o[g], x)
                    else:
                        o[g] = max(o[g], x)

    var merge_chunks = (G + MERGE_GROUPS - 1) // MERGE_GROUPS
    if nthreads > 1:
        # A host process (Python via ctypes) has no Mojo async runtime yet:
        # create the global CPU device (thread pool) once, then parallelize.
        start_pool()
    var wave0 = 0
    while wave0 < morsels and failed[0] == 0:
        ctl[0] = Int64(wave0)
        ctl[1] = Int64(min(nslots, morsels - wave0))
        counter[0] = 0
        if nthreads == 1:
            worker(0)
        else:
            par(worker, nthreads)
        if failed[0] != 0:
            break
        if A > 0:
            if nthreads > 1 and merge_chunks > 1:
                par_n(merge, merge_chunks, nthreads)
            else:
                for chunk in range(merge_chunks):
                    merge(chunk)
        wave0 += nslots
    for t in range(nthreads):
        if Int(wst[t * 4]) == 0:                 # a worker that never started has no counts
            continue
        var q = pc_all + t * pc_stride
        for g in range(G):
            var c = Int64(0)
            for l in range(NL):
                c += q[g * NL + l]
            out_cnt[g] += c
    for w in range(nthreads):
        var st = wst + w * 4
        if Int(st[0]) != 0:
            for f in range(4):
                Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(st[f])).free()
    wst.free()
    ctl.free()
    free_big(Int(pv_all))
    free_big(Int(pc_all))
    counter.free()
    var rc = -5 if Int(failed[0]) != 0 else 0
    failed.free()
    return rc


def _no_keys(chunk: Int, row0: Int, sel: IP, nsel: Int, use_sel: Bool, dst: IP, regs: FP) -> Bool:
    return True


def _no_front(chunk: Int, row0: Int, n: Int, sel: IP, ws: BP) -> Int:
    return n


@export
def mx_pipeline_cpu(
    code_a: Int, consts_a: Int, set_bytes_a: Int, set_meta_a: Int,
    tbl_a: Int, n_chunks: Int, dtypes_a: Int, gids_a: Int, n_groups: Int,
    start: Int, end: Int, out_vals_a: Int, out_cnt_a: Int, threads: Int, repro: Int,
) abi("C") -> Int:
    """`repro` (15th argument, F-064): non-zero = float sums in the defined order whatever the group count."""
    return _pipeline[keyed=False, fused=False](code_a, consts_a, set_bytes_a, set_meta_a, tbl_a, n_chunks,
                                               dtypes_a, gids_a, n_groups, start, end, out_vals_a, out_cnt_a,
                                               threads, _no_keys, _no_front, 0, repro != 0)


def pipeline_plain(
    code_a: Int, consts_a: Int, set_bytes_a: Int, set_meta_a: Int,
    tbl_a: Int, n_chunks: Int, dtypes_a: Int, gids_a: Int, n_groups: Int,
    start: Int, end: Int, out_vals_a: Int, out_cnt_a: Int, threads: Int, repro: Bool,
) -> Int:
    """mx_pipeline_cpu with the reproducible-sums flag (the native executor's calls)."""
    return _pipeline[keyed=False, fused=False](code_a, consts_a, set_bytes_a, set_meta_a, tbl_a, n_chunks,
                                               dtypes_a, gids_a, n_groups, start, end, out_vals_a, out_cnt_a,
                                               threads, _no_keys, _no_front, 0, repro)


def pipeline_keyed[E: def(Int, Int, IP, Int, Bool, IP, FP) -> Bool](
    code_a: Int, consts_a: Int, set_bytes_a: Int, set_meta_a: Int,
    tbl_a: Int, n_chunks: Int, dtypes_a: Int, n_groups: Int,
    start: Int, end: Int, out_vals_a: Int, out_cnt_a: Int, threads: Int, encode: E, repro: Bool,
) -> Int:
    """mx_pipeline_cpu with group ids made per morsel by `encode` (see _pipeline)."""
    return _pipeline[keyed=True, fused=False](code_a, consts_a, set_bytes_a, set_meta_a, tbl_a, n_chunks,
                                              dtypes_a, 0, n_groups, start, end, out_vals_a, out_cnt_a,
                                              threads, encode, _no_front, 0, repro)


def pipeline_fused[E: def(Int, Int, IP, Int, Bool, IP, FP) -> Bool, FR: def(Int, Int, Int, IP, BP) -> Int,
                   keyed: Bool](
    code_a: Int, consts_a: Int, set_bytes_a: Int, set_meta_a: Int,
    tbl_a: Int, n_chunks: Int, dtypes_a: Int, n_groups: Int,
    start: Int, end: Int, out_vals_a: Int, out_cnt_a: Int, threads: Int, encode: E,
    front: FR, ws_bytes: Int, repro: Bool,
) -> Int:
    """The pipeline over the rows `front` lets through (a fused join + aggregate, see _process_range)."""
    return _pipeline[keyed=keyed, fused=True](code_a, consts_a, set_bytes_a, set_meta_a, tbl_a, n_chunks,
                                              dtypes_a, 0, n_groups, start, end, out_vals_a, out_cnt_a,
                                              threads, encode, front, ws_bytes, repro)
