"""128-bit values (F-192): the totals a SUM of INT or DECIMAL values reaches past int64, which DuckDB keeps in a HUGEINT / DECIMAL(38).

A column stores every value as an Int64, so a wide total travels in two places: the SUM itself is the sum of two exact int64 sums (the
value's high and low 32-bit halves, `sqlprog.lower_sql` with `wide`), and the stage after aggregation (`post.mojo`) reads such a total
as an `Int128`. What the engine hands out is a pair of words: `Column.high` (the upper 64 bits, empty while every value fits in int64)
next to `Column.values` (the lower 64 bits as they are).

No 128-bit `//` or `%` anywhere: Apple's libSystem lacks `___udivti3` (docs/lessons.md). Text is made and read through 32-bit limbs held in UInt64
(a 64-bit division is a hardware instruction), and a conversion to a double is done on the words.
"""

comptime DECIMAL_DIGITS = 38        # DuckDB's DECIMAL holds 38 digits; a result past them is an error, not a wrapped number


def pow10_wide(k: Int) raises -> Int128:
    if k < 0 or k > DECIMAL_DIGITS:
        raise Error("a power of ten outside 10^0 .. 10^38")
    var v = Int128(1)
    for _ in range(k):
        v *= Int128(10)
    return v


def wide_make(lo: Int64, hi: Int64) -> Int128:
    """The 128-bit value whose lower 64 bits are `lo` (as bits) and upper 64 bits `hi`."""
    return (Int128(hi) << Int128(64)) | Int128(UInt64(lo))


def wide_lo(v: Int128) -> Int64:
    return Int64(UInt64(v & Int128(0xFFFFFFFFFFFFFFFF)))


def wide_hi(v: Int128) -> Int64:
    return Int64(v >> Int128(64))


def fits_int64(v: Int128) -> Bool:
    return v >= Int128(Int64.MIN) and v <= Int128(Int64.MAX)


def _magnitude(v: Int128) -> UInt128:
    return UInt128(-(v + Int128(1))) + UInt128(1) if v < Int128(0) else UInt128(v)


def bit_length(m: UInt128) -> Int:
    var n = 0
    var x = m
    while x != UInt128(0):
        n += 1
        x >>= UInt128(1)
    return n


@always_inline
def _limit() -> Int128:
    """10^38."""
    return Int128(10000000000000000000) * Int128(10000000000000000000)


def within_decimal(v: Int128) raises -> Int128:
    """`v`, or an error when it has more than 38 digits (DuckDB: "Overflow in multiplication of DECIMAL(38)")."""
    var limit = _limit()
    if v >= limit or v <= -limit:
        raise Error("a value passed the 38 digits a DECIMAL holds")
    return v


def wide_mul(a: Int128, b: Int128) raises -> Int128:
    """a * b, exactly, or an error when the product has more than 38 digits."""
    if a == Int128(0) or b == Int128(0):
        return Int128(0)
    var ma = _magnitude(a)
    var mb = _magnitude(b)
    if bit_length(ma) + bit_length(mb) > 128:                       # (the product is at least 2^127: more than 38 digits)
        raise Error("a value passed the 38 digits a DECIMAL holds")
    var p = ma * mb                                                 # (below 2^128: no wrap)
    if p >= UInt128(_limit()):
        raise Error("a value passed the 38 digits a DECIMAL holds")
    var neg = (a < Int128(0)) != (b < Int128(0))
    return -Int128(p) if neg else Int128(p)


def wide_add(a: Int128, b: Int128) raises -> Int128:
    return within_decimal(a + b)                                    # (both within 38 digits, so the sum cannot wrap 128 bits)


def wide_sub(a: Int128, b: Int128) raises -> Int128:
    return within_decimal(a - b)


def wide_text(v: Int128, scale: Int) -> String:
    """The value as DuckDB prints a HUGEINT / DECIMAL(38, scale): 12345 at scale 2 is 123.45."""
    var neg = v < Int128(0)
    var m = _magnitude(v)
    var limbs = List[UInt64](length=4, fill=0)
    for i in range(4):
        limbs[i] = UInt64(m & UInt128(0xFFFFFFFF))
        m >>= UInt128(32)
    var chunks = List[UInt64]()                                     # base 10^9, least significant first
    while limbs[0] != 0 or limbs[1] != 0 or limbs[2] != 0 or limbs[3] != 0:
        var rem = UInt64(0)
        var i = 3
        while i >= 0:
            var cur = (rem << 32) | limbs[i]
            limbs[i] = cur // 1000000000
            rem = cur % 1000000000
            i -= 1
        chunks.append(rem)
    var digits = String()
    if len(chunks) == 0:
        digits = "0"
    else:
        digits = String(chunks[len(chunks) - 1])
        var k = len(chunks) - 2
        while k >= 0:
            var part = String(chunks[k])
            while part.byte_length() < 9:
                part = "0" + part
            digits += part
            k -= 1
    if scale > 0:
        while digits.byte_length() <= scale:
            digits = "0" + digits
        var cut = digits.byte_length() - scale
        digits = String(digits[byte=:cut]) + "." + String(digits[byte=cut:])
    return ("-" if neg else "") + digits


struct WideNum(Movable):
    var value: Int128
    var scale: Int
    var ok: Bool

    def __init__(out self):
        self.value = Int128(0)
        self.scale = 0
        self.ok = False


def parse_wide(text: String) -> WideNum:
    """A literal such as `-12.50` read into 128 bits: ok is False past 38 digits or 18 decimal places (the most a column's scale takes)."""
    var out = WideNum()
    var b = text.as_bytes()
    var neg = False
    var i = 0
    if len(b) > 0 and b[0] == UInt8(45):
        neg = True
        i = 1
    var v = Int128(0)
    var scale = 0
    var digits = 0
    var seen_dot = False
    while i < len(b):
        if b[i] == UInt8(46):
            seen_dot = True
            i += 1
            continue
        if b[i] < UInt8(48) or b[i] > UInt8(57):
            return out^
        digits += 1
        if digits > DECIMAL_DIGITS:
            return out^
        v = v * Int128(10) + Int128(Int(b[i] - 48))
        if seen_dot:
            scale += 1
        i += 1
    if scale > 18:
        return out^
    out.value = -v if neg else v
    out.scale = scale
    out.ok = True
    return out^


def _hug(m: UInt128) -> Float64:
    """An unsigned 128-bit integer as a double the way DuckDB's HUGEINT cast does it: the upper word times 2^64 plus the lower word (two roundings)."""
    return Float64(UInt64(m >> UInt128(64))) * 18446744073709551616.0 + Float64(UInt64(m & UInt128(0xFFFFFFFFFFFFFFFF)))


def _divmod_small(m: UInt128, p: UInt64) -> Tuple[UInt128, UInt64]:
    """m / p and m % p by shifting and subtracting (no 128-bit division: docs/lessons.md)."""
    var q = UInt128(0)
    var rem = UInt128(0)
    var d = UInt128(p)
    var i = 127
    while i >= 0:
        rem = (rem << UInt128(1)) | ((m >> UInt128(i)) & UInt128(1))
        if rem >= d:
            rem -= d
            q |= UInt128(1) << UInt128(i)
        i -= 1
    return (q, UInt64(rem))


def wide_to_float(v: Int128, scale: Int) -> Float64:
    """The value read as a DOUBLE, as DuckDB casts a DECIMAL(38, scale) (probed bit for bit against DuckDB 1.5.1 on 2,500 values, scales 0 to 18):
    scale 0, the two's-complement words (the upper one signed): upper * 2^64 + lower; otherwise the whole part and the fraction apart (the
    quotient and remainder by 10^scale, both truncating toward zero), each made a double from its words: whole + fraction / 10^scale."""
    if scale == 0:
        return Float64(Int64(v >> Int128(64))) * 18446744073709551616.0 + Float64(UInt64(v & Int128(0xFFFFFFFFFFFFFFFF)))
    var neg = v < Int128(0)
    var m = _magnitude(v)
    var pu = UInt64(1)
    var pf = Float64(1.0)
    for _ in range(scale):
        pu *= 10
        pf *= 10.0
    var qr = _divmod_small(m, pu)
    var x = _hug(qr[0]) + Float64(qr[1]) / pf
    return -x if neg else x
