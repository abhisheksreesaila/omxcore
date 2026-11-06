"""Values of an aggregate result's columns, and the exact AVG (F-150, F-152).

`ValCol` is a column of results: an Int64 per group (a FLOAT holds a float64's bits) and a flag per group for NULL. The
expressions that combine the groups' values (HAVING, the select list, ORDER BY) are no longer evaluated here: since F-152 they run
as a row query over the groups through the executor (`post.mojo`). What stays is the column type, the float helpers, and
`avg_div`, which divides as DuckDB's AVG does on this platform.
"""
from std.math import fma
from std.sys import CompilationTarget
from .table import INT, DECIMAL, TEXT, BOOL, FLOAT
from .wide import wide_make, wide_lo, wide_hi, fits_int64, hug


struct ValCol(Copyable, Movable):
    var kind: Int
    var scale: Int
    var src: Int                     # a TEXT value: the source column whose dictionary its codes belong to
    var v: List[Int64]
    var hi: List[Int64]              # F-192: a total past int64: the upper 64 bits of each value, `v` holding the lower 64; empty while every value fits int64
    var nul: List[UInt8]             # 1 where NULL; empty when none is

    def __init__(out self, kind: Int, scale: Int, n: Int):
        self.kind = kind
        self.scale = scale
        self.src = -1
        self.v = List[Int64](length=n, fill=0)
        self.hi = List[Int64]()
        self.nul = List[UInt8]()

    def n(self) -> Int:
        return len(self.v)

    def is_null(self, i: Int) -> Bool:
        return len(self.nul) > 0 and self.nul[i] != 0

    def set_null(mut self, i: Int):
        if len(self.nul) == 0:
            self.nul = List[UInt8](length=len(self.v), fill=0)
        self.nul[i] = 1

    def is_wide(self) -> Bool:
        return len(self.hi) > 0

    def wide_at(self, i: Int) -> Int128:
        """Value i as 128 bits (a column that is not wide holds int64s)."""
        return wide_make(self.v[i], self.hi[i]) if len(self.hi) > 0 else Int128(self.v[i])

    def set_wide(mut self, i: Int, x: Int128):
        """Stores a 128-bit value; the upper words appear (for every value) with the first one that does not fit int64."""
        if len(self.hi) == 0:
            if fits_int64(x):
                self.v[i] = Int64(x)
                return
            self.hi = List[Int64](length=len(self.v), fill=0)
            for j in range(len(self.v)):
                self.hi[j] = self.v[j] >> 63
        self.v[i] = wide_lo(x)
        self.hi[i] = wide_hi(x)

    def has_nulls(self) -> Bool:
        for x in self.nul:
            if x != 0:
                return True
        return False


def _fcmp(a: Float64, b: Float64) -> Int:
    if a != a:
        return 0 if b != b else 1
    if b != b:
        return -1
    return -1 if a < b else 1 if a > b else 0



def float_bits(x: Float64) -> Int64:
    return Int64(x.to_bits())


def avg_div(num: Int128, cnt: Int, scale: Int) -> Float64:
    """AVG of a total `num` (the stored integers' sum at `scale` decimal places) over `cnt` rows, as DuckDB divides it on this platform: its
    C++ `long double` is the x87 80-bit type on x86-64 (`avg_div_x87`) and a plain double elsewhere (`avg_div_double`). F-211."""
    comptime if CompilationTarget.is_x86():
        return avg_div_x87(num, cnt, scale)
    else:
        return avg_div_double(num, cnt, scale)


def avg_div_x87(num: Int128, cnt: Int, scale: Int) -> Float64:
    """x86-64: `ld_div_wide` over count * 10^scale where DuckDB's power of ten is a double (probed: scales 2 to 38; an exact 10^25 gave a different last
    bit for 3 of 3 cases) and the product with the count is rounded to the 64-bit mantissa. The double is m * 2^e, so the product is cnt * m (below 2^117)
    rounded to 64 bits, divided exactly and scaled back by 2^-e."""
    var p = Int128(1)
    for _ in range(scale):
        p *= Int128(10)
    var pb = UInt64(hug(p).to_bits())
    var m = UInt128((pb & UInt64(0xFFFFFFFFFFFFF)) | (UInt64(1) << 52))
    var e = Int((pb >> 52) & UInt64(0x7FF)) - 1075
    var den = UInt128(cnt) * m
    var bits = _bitlen128(den)
    if bits > 64:
        var sh = bits - 64
        var low = den & ((UInt128(1) << UInt128(sh)) - UInt128(1))
        var half = UInt128(1) << UInt128(sh - 1)
        den >>= UInt128(sh)
        if low > half or (low == half and (den & UInt128(1)) != UInt128(0)):
            den += UInt128(1)
        e += sh
    return ld_div_wide(num, den) * Float64(from_bits=UInt64((1023 - e) << 52))


def avg_div_double(num: Int128, cnt: Int, scale: Int) -> Float64:
    """arm64 (probed on an Apple M4, DuckDB 1.5.6: all 2,900 differing cases, scales 0 to 38) and any platform whose `long double` is a double:
    DuckDB's HUGEINT cast of the total (`hug`, the same on both platforms), divided by double(count) * double(10^scale)."""
    var p = Int128(1)
    for _ in range(scale):
        p *= Int128(10)
    return hug(num) / (Float64(cnt) * hug(p))


def ld_div_wide(num: Int128, den: UInt128) -> Float64:
    """num / den as an x87 `long double` division rounded to a double, which is how DuckDB computes AVG: the quotient is
    rounded to a 64-bit mantissa, then to 53 bits. That is not always the correctly rounded quotient (about 1 average
    in 20,000 differs by one ulp), and the oracle is DuckDB's, so this reproduces its rounding exactly.

    The two roundings differ only when the 64-bit quotient lands exactly on a midpoint of the 53-bit grid, so unless the
    exact quotient lies within 1/512 ulp of one, the plain double division (correctly rounded) is the answer: found
    from the exact residual (an FMA). Otherwise, and for operands past 2^53, the quotient is computed bit by bit.
    The numerator may be a total past int64 (F-192): only the bit loop reads it, and its quotient has 67 or 68 bits whatever the size."""
    if num == Int128(0):
        return 0.0
    var neg = num < Int128(0)
    var n = UInt128(-(num + Int128(1))) + UInt128(1) if neg else UInt128(num)
    if n < (UInt128(1) << 53) and den < (UInt128(1) << 53):
        var nf = Float64(UInt64(n))
        var df = Float64(UInt64(den))
        var q0 = nf / df
        var r = fma(-q0, df, nf)
        if r == 0.0:
            return -q0 if neg else q0
        var q0_bits = UInt64(q0.to_bits())
        var neighbor = Float64(from_bits=q0_bits + 1 if r > 0.0 else q0_bits - 1)
        var spacing = neighbor - q0 if r > 0.0 else q0 - neighbor
        var frac = (r if r > 0.0 else -r) / df
        var gap = frac - spacing * 0.5
        if (gap if gap > 0.0 else -gap) > spacing * (1.0 / 512.0):
            return -q0 if neg else q0
    var bn = _bitlen128(n)
    var bd = _bitlen128(den)
    var s = 67 - bn + bd                                        # (n << s) / den has 67 or 68 bits
    var total = bn + s
    var rem = UInt128(0)
    var q = UInt128(0)
    for i in range(total):
        var idx = total - 1 - i
        var bit = UInt128(0)
        if idx >= s:
            bit = (n >> UInt128(idx - s)) & UInt128(1)
        rem = (rem << UInt128(1)) | bit
        q = q << UInt128(1)
        if rem >= den:
            rem -= den
            q |= UInt128(1)
    var bq = _bitlen128(q)
    var drop = bq - 64
    var low = q & ((UInt128(1) << UInt128(drop)) - UInt128(1))
    var m64 = q >> UInt128(drop)
    var half = UInt128(1) << UInt128(drop - 1)
    if low > half or (low == half and (rem != UInt128(0) or (m64 & UInt128(1)) != UInt128(0))):
        m64 += UInt128(1)
    var exp = drop - s                                          # the value is m64 * 2^exp
    var low11 = m64 & UInt128(0x7FF)
    var m53 = m64 >> UInt128(11)
    if low11 > UInt128(0x400) or (low11 == UInt128(0x400) and (m53 & UInt128(1)) != UInt128(0)):
        m53 += UInt128(1)
    exp += 11
    var out = Float64(UInt64(m53)) * Float64(from_bits=UInt64((1023 + exp) << 52))
    return -out if neg else out


def _bitlen64(x: UInt64) -> Int:
    var n = 0
    var v = x
    while v != 0:
        n += 1
        v >>= 1
    return n


def _bitlen128(x: UInt128) -> Int:
    var n = 0
    var v = x
    while v != UInt128(0):
        n += 1
        v >>= UInt128(1)
    return n
