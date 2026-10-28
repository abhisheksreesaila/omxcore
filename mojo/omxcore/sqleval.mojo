"""Values of an aggregate result's columns, and the exact AVG (F-150, F-152).

`ValCol` is a column of results: an Int64 per group (a FLOAT holds a float64's bits) and a flag per group for NULL. The
expressions that combine the groups' values (HAVING, the select list, ORDER BY) are no longer evaluated here: since F-152 they run
as a row query over the groups through the executor (`post.mojo`). What stays is the column type, the float helpers, and
`ld_div`, which divides as DuckDB's AVG does.
"""
from std.math import fma
from .table import INT, DECIMAL, TEXT, BOOL, FLOAT


struct ValCol(Copyable, Movable):
    var kind: Int
    var scale: Int
    var src: Int                     # a TEXT value: the source column whose dictionary its codes belong to
    var v: List[Int64]
    var nul: List[UInt8]             # 1 where NULL; empty when none is

    def __init__(out self, kind: Int, scale: Int, n: Int):
        self.kind = kind
        self.scale = scale
        self.src = -1
        self.v = List[Int64](length=n, fill=0)
        self.nul = List[UInt8]()

    def n(self) -> Int:
        return len(self.v)

    def is_null(self, i: Int) -> Bool:
        return len(self.nul) > 0 and self.nul[i] != 0

    def set_null(mut self, i: Int):
        if len(self.nul) == 0:
            self.nul = List[UInt8](length=len(self.v), fill=0)
        self.nul[i] = 1

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


def ld_div(num: Int64, den: UInt128) -> Float64:
    """num / den as an x87 `long double` division rounded to a double, which is how DuckDB computes AVG: the quotient is
    rounded to a 64-bit mantissa, then to 53 bits. That is not always the correctly rounded quotient (about 1 average
    in 20,000 differs by one ulp), and the oracle is DuckDB's, so this reproduces its rounding exactly.

    The two roundings differ only when the 64-bit quotient lands exactly on a midpoint of the 53-bit grid, so unless the
    exact quotient lies within 1/512 ulp of one, the plain double division (correctly rounded) is the answer: found
    from the exact residual (an FMA). Otherwise, and for operands past 2^53, the quotient is computed bit by bit."""
    if num == 0:
        return 0.0
    var neg = num < 0
    var n = UInt64(0) - UInt64(num) if neg else UInt64(num)
    if n < (UInt64(1) << 53) and den < (UInt128(1) << 53):
        var nf = Float64(n)
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
    var bn = _bitlen64(n)
    var bd = _bitlen128(den)
    var s = 67 - bn + bd                                        # (n << s) / den has 67 or 68 bits
    var total = bn + s
    var rem = UInt128(0)
    var q = UInt128(0)
    for i in range(total):
        var idx = total - 1 - i
        var bit = UInt128(0)
        if idx >= s:
            bit = UInt128((n >> UInt64(idx - s)) & 1)
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
