"""The scalar functions of SQL as plain functions on values (F-156): what a text function answers for one word, what a date function answers for one day.

No query machinery in here: the binder folds a function of constants with these, and the emitter (`sqlprog.mojo`) evaluates a text function once per word of a
dictionary and hands the answers to the kernel as a table. Everything follows DuckDB 1.5 (the oracle: `tests/test_mojo_exprs.py`), including its
quirks: `length` counts code points, `lower` and `upper` are utf8proc's simple mappings (`scripts/gen_unicase.py` asked DuckDB for them), `trim` strips
spaces only, `substr` clips its window to the string, a negative length reads backwards, `LIKE` has no escape character unless ESCAPE names one.
"""
from .csv import days_from_civil, _days_in_month, parse_float_text, F_OK, F_RANGE
from .table import TEXT, INT, DECIMAL, DATE, BOOL, FLOAT, F32_COLUMN
from .unicase import LOWER_RUNS, UPPER_RUNS

# ── UTF-8 ────────────────────────────────────────────────────────────────────────────────────────────

def ulen(b: UInt8) -> Int:
    """The length of the UTF-8 sequence that starts with byte b (1 for a stray continuation byte)."""
    if b < UInt8(0x80):
        return 1
    if b < UInt8(0xE0):
        return 2 if b >= UInt8(0xC0) else 1
    if b < UInt8(0xF0):
        return 3
    return 4


def decode(s: String) -> List[Int]:
    """The code points of s."""
    var out = List[Int]()
    var b = s.as_bytes()
    var n = len(b)
    var i = 0
    while i < n:
        var c = Int(b[i])
        var k = ulen(b[i])
        if k == 1 or i + k > n:
            out.append(c)
            i += 1
            continue
        var cp = c & (0x1F if k == 2 else 0x0F if k == 3 else 0x07)
        for j in range(1, k):
            cp = (cp << 6) | (Int(b[i + j]) & 0x3F)
        out.append(cp)
        i += k
    return out^


def encode_into(cp: Int, mut out: List[UInt8]):
    if cp < 0x80:
        out.append(UInt8(cp))
    elif cp < 0x800:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))


def encode(cps: List[Int], a: Int, b: Int) -> String:
    """The characters cps[a:b] as a string."""
    var out = List[UInt8]()
    for i in range(a, b):
        encode_into(cps[i], out)
    return String(unsafe_from_utf8=out)


def char_count(s: String) -> Int:
    """The number of code points (DuckDB's length())."""
    var n = 0
    for b in s.as_bytes():
        if (b & UInt8(0xC0)) != UInt8(0x80):
            n += 1
    return n


# ── case mapping ─────────────────────────────────────────────────────────────────────────────────────

struct CaseTables(Movable):
    """lower / upper: dense delta arrays below U+20000, built from the generated runs (`unicase.mojo`)."""
    var lower: List[Int32]
    var upper: List[Int32]

    def __init__(out self) raises:
        self.lower = _expand(LOWER_RUNS)
        self.upper = _expand(UPPER_RUNS)

    def map(self, s: String, up: Bool) -> String:
        var b = s.as_bytes()
        var ascii = True
        for x in b:
            if x >= UInt8(0x80):
                ascii = False
                break
        var out = List[UInt8](capacity=len(b) + 4)
        if ascii:
            var changed = False
            for x in b:
                if not up and UInt8(65) <= x <= UInt8(90):
                    out.append(x + 32)
                    changed = True
                elif up and UInt8(97) <= x <= UInt8(122):
                    out.append(x - 32)
                    changed = True
                else:
                    out.append(x)
            if not changed:
                return s
            return String(unsafe_from_utf8=out)
        ref tab = self.upper if up else self.lower
        for cp in decode(s):
            var m = cp
            if cp < len(tab):
                m = cp + Int(tab[cp])
            encode_into(m, out)
        return String(unsafe_from_utf8=out)


def ascii_case(s: String, up: Bool) -> String:
    """lower / upper of a string that holds only ASCII."""
    var out = List[UInt8]()
    for x in s.as_bytes():
        if not up and UInt8(65) <= x <= UInt8(90):
            out.append(x + 32)
        elif up and UInt8(97) <= x <= UInt8(122):
            out.append(x - 32)
        else:
            out.append(x)
    return String(unsafe_from_utf8=out)


def _expand(spec: StaticString) raises -> List[Int32]:
    var tab = List[Int32](length=0x20000, fill=0)
    for run in spec.split(","):
        var f = run.split(":")
        var a = Int(f[0])
        var b = Int(f[1])
        var d = Int(f[2])
        var step = Int(f[3])
        var c = a
        while c <= b:
            tab[c] = Int32(d)
            c += step
    return tab^


# ── text functions ───────────────────────────────────────────────────────────────────────────────────

def substr(s: String, start: Int, length: Int, has_len: Bool) -> String:
    """DuckDB's substr: 1-based characters; a negative start counts from the end; the window [p, p + length - 1] (backwards for a negative length) is
    clipped to the string; no length reads to the end."""
    var cps = decode(s)
    var n = len(cps)
    var p = start if start > 0 else (n + start + 1 if start < 0 else 0)
    var lo = p
    var hi = n
    if has_len:
        if length > 0:
            hi = p + length - 1
        elif length < 0:
            lo = p + length
            hi = p - 1
        else:
            return String()
    if lo < 1:
        lo = 1
    if hi > n:
        hi = n
    if lo > hi:
        return String()
    return encode(cps, lo - 1, hi)


def left_of(s: String, k: Int) -> String:
    """left(s, k): the first k characters; a negative k drops the last -k."""
    var cps = decode(s)
    var n = len(cps)
    var m = k if k >= 0 else n + k
    if m <= 0:
        return String()
    return encode(cps, 0, min(m, n))


def right_of(s: String, k: Int) -> String:
    var cps = decode(s)
    var n = len(cps)
    var m = k if k >= 0 else n + k
    if m <= 0:
        return String()
    return encode(cps, n - min(m, n), n)


def reverse_of(s: String) -> String:
    var cps = decode(s)
    var out = List[UInt8]()
    for i in range(len(cps) - 1, -1, -1):
        encode_into(cps[i], out)
    return String(unsafe_from_utf8=out)


def reverse_is_exact(s: String) -> Bool:
    """DuckDB reverses a word by grapheme clusters (a letter and its combining marks, an emoji sequence, a flag stay together); reversing code points is the same
    answer when every character stands alone. True when every code point is one that always does: below U+0300, the Latin, Greek, Cyrillic, Hangul-syllable,
    kana and CJK letters, and the emoji faces. Anything else (combining marks, joiners, variation selectors, regional indicators, Indic scripts ...) is not decided."""
    for c in decode(s):
        if c < 0x300:
            continue
        var ok = (0x370 <= c <= 0x482) or (0x48A <= c <= 0x52F) or (0x1E00 <= c <= 0x1EFF) or (0x3041 <= c <= 0x3096) or (0x30A1 <= c <= 0x30FA) or \
                 (0x4E00 <= c <= 0x9FFF) or (0xAC00 <= c <= 0xD7A3) or (0x1F600 <= c <= 0x1F64F)
        if not ok:
            return False
    return True


def trim_chars(s: String, chars: String, left: Bool, right: Bool) -> String:
    """trim / ltrim / rtrim with a set of characters (DuckDB's default is the space only)."""
    var cps = decode(s)
    var set = decode(chars)
    var a = 0
    var b = len(cps)
    if left:
        while a < b and _in(set, cps[a]):
            a += 1
    if right:
        while b > a and _in(set, cps[b - 1]):
            b -= 1
    return encode(cps, a, b)


def _in(set: List[Int], c: Int) -> Bool:
    for x in set:
        if x == c:
            return True
    return False


def strpos_of(s: String, sub: String) -> Int:
    """The 1-based character position of the first `sub` in s, 0 when it is not there (1 for the empty string)."""
    if sub.byte_length() == 0:
        return 1
    var i = s.find(sub)
    if i < 0:
        return 0
    return char_count(String(s[byte=0:i])) + 1


def replace_all(s: String, find: String, repl: String) -> String:
    if find.byte_length() == 0:
        return s
    return s.replace(find, repl)


struct LikePattern(Copyable, Movable):
    """A LIKE pattern as tokens: 0 any run (`%`), 1 any one character (`_`), 2 a literal character (its bytes in `lit`)."""
    var kind: List[Int]
    var lo: List[Int]
    var hi: List[Int]
    var lit: List[UInt8]
    var shape: Int             # 0 general, 1 `lit`, 2 `lit%`, 3 `%lit`, 4 `%lit%`, 5 `%` (anything): the shapes that need no backtracking
    var needle: List[UInt8]    # the literal of shapes 1 to 4

    def __init__(out self, pattern: String, escape: String, has_escape: Bool):
        self.kind = List[Int]()
        self.lo = List[Int]()
        self.hi = List[Int]()
        self.lit = List[UInt8]()
        self.shape = 0
        self.needle = List[UInt8]()
        var b = pattern.as_bytes()
        var e = -1
        if has_escape and escape.byte_length() > 0:
            e = decode(escape)[0]
        var n = len(b)
        var i = 0
        while i < n:
            var k = ulen(b[i])
            if i + k > n:
                k = n - i
            var cp = Int(b[i])
            if k > 1:
                cp = decode(String(unsafe_from_utf8=b[i:i + k]))[0]
            var literal = False
            if e >= 0 and cp == e and i + k < n:                  # the escape makes the next character literal
                i += k
                k = ulen(b[i])
                if i + k > n:
                    k = n - i
                literal = True
                cp = 0
            if not literal and b[i] == UInt8(37):
                self.kind.append(0)
                self.lo.append(0)
                self.hi.append(0)
            elif not literal and b[i] == UInt8(95):
                self.kind.append(1)
                self.lo.append(0)
                self.hi.append(0)
            else:
                self.kind.append(2)
                self.lo.append(len(self.lit))
                for j in range(k):
                    self.lit.append(b[i + j])
                self.hi.append(len(self.lit))
            i += k
        var first = 0
        var last = len(self.kind)
        while first < last and self.kind[first] == 0:
            first += 1
        while last > first and self.kind[last - 1] == 0:
            last -= 1
        var plain = True
        for t in range(first, last):
            plain = plain and self.kind[t] == 2
        if plain:
            if len(self.kind) > 0 and first == last:
                self.shape = 5                                   # only `%`
            else:
                var lead = first > 0
                var trail = last < len(self.kind)
                self.shape = 4 if (lead and trail) else 3 if lead else 2 if trail else 1
                for t in range(first, last):
                    for j in range(self.lo[t], self.hi[t]):
                        self.needle.append(self.lit[j])

    def matches(self, s: String) -> Bool:
        return self.matches_bytes(s.as_bytes())

    def matches_bytes(self, b: Span[UInt8, _]) -> Bool:
        var n = len(b)
        var sh = self.shape
        if sh != 0:
            var m = len(self.needle)
            if sh == 5:
                return True
            if sh == 1:
                return n == m and self._eq_at(b, 0)
            if sh == 2:
                return n >= m and self._eq_at(b, 0)
            if sh == 3:
                return n >= m and self._eq_at(b, n - m)
            if m == 0:
                return True
            var first = self.needle[0]
            var i = 0
            while i + m <= n:
                if b[i] == first and self._eq_at(b, i):
                    return True
                i += 1
            return False
        var m = len(self.kind)
        var si = 0
        var pi = 0
        var star_pi = -1
        var star_si = 0
        while si < n:
            if pi < m and self.kind[pi] == 1:
                si += min(ulen(b[si]), n - si)
                pi += 1
            elif pi < m and self.kind[pi] == 2 and self._lit_at(b, si, pi):
                si += self.hi[pi] - self.lo[pi]
                pi += 1
            elif pi < m and self.kind[pi] == 0:
                star_pi = pi
                star_si = si
                pi += 1
            elif star_pi >= 0:
                star_si += min(ulen(b[star_si]), n - star_si)
                si = star_si
                pi = star_pi + 1
            else:
                return False
        while pi < m and self.kind[pi] == 0:
            pi += 1
        return pi == m

    def _eq_at(self, b: Span[UInt8, _], at: Int) -> Bool:
        for j in range(len(self.needle)):
            if b[at + j] != self.needle[j]:
                return False
        return True

    def _lit_at(self, b: Span[UInt8, _], si: Int, pi: Int) -> Bool:
        var k = self.hi[pi] - self.lo[pi]
        if si + k > len(b):
            return False
        for j in range(k):
            if b[si + j] != self.lit[self.lo[pi] + j]:
                return False
        return True


# ── dates ────────────────────────────────────────────────────────────────────────────────────────────

def civil_from_days(days: Int) -> Tuple[Int, Int, Int]:
    """(year, month, day) of a day number counted from 1970-01-01 (proleptic Gregorian; floor divisions, so dates before 1970 are right)."""
    var z = days + 719468
    var era = z // 146097
    var doe = z - era * 146097
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var y = yoe + era * 400
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    var mp = (5 * doy + 2) // 153
    var d = doy - (153 * mp + 2) // 5 + 1
    var m = mp + 3 if mp < 10 else mp - 9
    return (y + (1 if m <= 2 else 0), m, d)


def is_leap(y: Int) -> Bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def add_months(days: Int, months: Int) -> Int:
    """The date `months` after `days`; a day past the end of the target month clamps to its last day (Jan 31 + 1 month = Feb 28 / 29)."""
    var c = civil_from_days(days)
    var total = c[0] * 12 + (c[1] - 1) + months
    var y = total // 12
    var m = total - y * 12 + 1
    return days_from_civil(y, m, min(c[2], _days_in_month(y, m)))


comptime DP_YEAR = 1
comptime DP_MONTH = 2
comptime DP_DAY = 3
comptime DP_DOW = 4          # 0 = Sunday
comptime DP_ISODOW = 5       # 1 = Monday .. 7 = Sunday
comptime DP_DOY = 6
comptime DP_QUARTER = 7
comptime DP_EPOCH = 8        # seconds since 1970-01-01
comptime DP_DECADE = 9
comptime DP_WEEK = 10        # trunc / diff: weeks start on Monday; date_part: the ISO week number
comptime DP_CENTURY = 11
comptime DP_MILLENNIUM = 12
comptime DP_ISOYEAR = 13     # ISO 8601 week-numbering year (F-177); `DP_WEEK` in `date_part` is the ISO week number 1 .. 53 (as a trunc / diff part it is the Monday)


def iso_thursday(days: Int) -> Int:
    """The Thursday of the ISO week the day number falls in (weeks run Monday to Sunday): the ISO year and week number are the year of that day and its
    day of year counted in sevens."""
    return days - (days + 3) % 7 + 3


def date_part(days: Int, part: Int) -> Int:
    var c = civil_from_days(days)
    if part == DP_YEAR:
        return c[0]
    if part == DP_MONTH:
        return c[1]
    if part == DP_DAY:
        return c[2]
    if part == DP_DOW:
        return (days + 4) % 7
    if part == DP_ISODOW:
        return (days + 3) % 7 + 1
    if part == DP_DOY:
        return days - days_from_civil(c[0], 1, 1) + 1
    if part == DP_QUARTER:
        return (c[1] - 1) // 3 + 1
    if part == DP_ISOYEAR or part == DP_WEEK:
        var th = iso_thursday(days)
        var y = civil_from_days(th)[0]
        return y if part == DP_ISOYEAR else (th - days_from_civil(y, 1, 1)) // 7 + 1
    if part == DP_EPOCH:
        return days * 86400
    if part == DP_DECADE:
        return c[0] // 10
    if part == DP_CENTURY:
        return (c[0] + 99) // 100 if c[0] > 0 else c[0] // 100
    return (c[0] + 999) // 1000 if c[0] > 0 else c[0] // 1000


def date_trunc_days(days: Int, part: Int) -> Int:
    var c = civil_from_days(days)
    if part == DP_YEAR:
        return days_from_civil(c[0], 1, 1)
    if part == DP_QUARTER:
        return days_from_civil(c[0], ((c[1] - 1) // 3) * 3 + 1, 1)
    if part == DP_MONTH:
        return days_from_civil(c[0], c[1], 1)
    if part == DP_WEEK:
        return days - (days + 3) % 7                              # back to Monday
    if part == DP_DECADE:
        return days_from_civil((c[0] // 10) * 10, 1, 1)
    if part == DP_CENTURY:                                        # DuckDB: the year divided by 100, floored (1948 is 1900, not 1901)
        return days_from_civil((c[0] // 100) * 100, 1, 1)
    if part == DP_MILLENNIUM:
        return days_from_civil((c[0] // 1000) * 1000, 1, 1)
    return days


def date_diff_days(a: Int, b: Int, part: Int) -> Int:
    """DuckDB's date_diff(part, a, b): the number of `part` boundaries crossed going from a to b (months, quarters, years, decades), whole weeks (the days
    over 7, truncated toward zero), or days."""
    if part == DP_DAY:
        return b - a
    if part == DP_WEEK:
        var d = b - a
        return d // 7 if d >= 0 else -((-d) // 7)
    var ca = civil_from_days(a)
    var cb = civil_from_days(b)
    if part == DP_YEAR:
        return cb[0] - ca[0]
    if part == DP_DECADE:
        return cb[0] // 10 - ca[0] // 10
    if part == DP_CENTURY:
        return cb[0] // 100 - ca[0] // 100
    if part == DP_MILLENNIUM:
        return cb[0] // 1000 - ca[0] // 1000
    var ma = ca[0] * 12 + ca[1] - 1
    var mb = cb[0] * 12 + cb[1] - 1
    if part == DP_QUARTER:
        return mb // 3 - ma // 3
    return mb - ma


def part_code(name: String) -> Int:
    """A date part's code from its name (lower case), -1 when there is no such part."""
    if name == "year" or name == "years" or name == "y":
        return DP_YEAR
    if name == "month" or name == "months" or name == "mon":
        return DP_MONTH
    if name == "day" or name == "days" or name == "d":
        return DP_DAY
    if name == "dow" or name == "dayofweek":
        return DP_DOW
    if name == "isodow":
        return DP_ISODOW
    if name == "doy" or name == "dayofyear":
        return DP_DOY
    if name == "quarter":
        return DP_QUARTER
    if name == "epoch":
        return DP_EPOCH
    if name == "decade":
        return DP_DECADE
    if name == "week" or name == "weeks" or name == "weekofyear":
        return DP_WEEK
    if name == "isoyear":
        return DP_ISOYEAR
    if name == "century":
        return DP_CENTURY
    if name == "millennium":
        return DP_MILLENNIUM
    return -1


# ── text of numbers and dates ────────────────────────────────────────────────────────────────────────

def pad_left(v: Int, w: Int) -> String:
    var s = String(v)
    while s.byte_length() < w:
        s = "0" + s
    return s


def decimal_text(v: Int64, scale: Int) -> String:
    """A stored decimal as DuckDB prints it: 12.50, -0.05, 5 (scale 0)."""
    var neg = v < 0
    var digits = String(UInt64(v) if not neg else UInt64(0) - UInt64(v))
    if scale == 0:
        return ("-" if neg else "") + digits
    while digits.byte_length() <= scale:
        digits = "0" + digits
    var cut = digits.byte_length() - scale
    return ("-" if neg else "") + String(digits[byte=:cut]) + "." + String(digits[byte=cut:])


def date_text(days: Int) -> String:
    """YYYY-MM-DD, the year padded to four digits (years 1 to 9999)."""
    var c = civil_from_days(days)
    return pad_left(c[0], 4) + "-" + pad_left(c[1], 2) + "-" + pad_left(c[2], 2)


# ── a call of a text function, as data ───────────────────────────────────────────────────────────────

comptime TF_LOWER = 1
comptime TF_UPPER = 2
comptime TF_TRIM = 3          # flags: 1 the left end, 2 the right end; the character set is the first string (space when none)
comptime TF_SUBSTR = 4        # ints: start, [length]
comptime TF_LENGTH = 5
comptime TF_STARTS = 6        # the first string
comptime TF_ENDS = 7
comptime TF_CONTAINS = 8
comptime TF_LIKE = 9          # strs: pattern, escape; flags: 1 ILIKE (the pattern and the word are lower-cased first), 2 an escape is given
comptime TF_STRPOS = 10
comptime TF_LEFT = 11         # ints: count
comptime TF_RIGHT = 12
comptime TF_REVERSE = 13
comptime TF_REPLACE = 14      # strs: what to find, what to put


def tf_kind(func: Int) -> Int:
    """What the function answers: TEXT, INT or BOOL (a ColumnKind value)."""
    if func == TF_LENGTH or func == TF_STRPOS:
        return INT.value
    if func == TF_STARTS or func == TF_ENDS or func == TF_CONTAINS or func == TF_LIKE:
        return BOOL.value
    return TEXT.value


struct TFn(Copyable, Movable):
    """One text function with its constant arguments; `to_text`, `to_int` and `to_bool` answer for one word (LOWER and UPPER need `CaseTables`:
    the caller maps the word, and for ILIKE lower-cases it before `to_bool`)."""
    var func: Int
    var flags: Int
    var strs: List[String]
    var ints: List[Int]
    var like: LikePattern

    def __init__(out self, func: Int, flags: Int, strs: List[String], ints: List[Int]):
        self.func = func
        self.flags = flags
        self.strs = strs.copy()
        self.ints = ints.copy()
        self.like = LikePattern(strs[0] if func == TF_LIKE and len(strs) > 0 else String(), strs[1] if func == TF_LIKE and len(strs) > 1 else String(),
                                (flags & 2) != 0)

    def to_text(self, w: String) -> String:
        if self.func == TF_TRIM:
            return trim_chars(w, self.strs[0] if len(self.strs) > 0 else String(" "), (self.flags & 1) != 0, (self.flags & 2) != 0)
        if self.func == TF_SUBSTR:
            return substr(w, self.ints[0], self.ints[1] if len(self.ints) > 1 else 0, len(self.ints) > 1)
        if self.func == TF_LEFT:
            return left_of(w, self.ints[0])
        if self.func == TF_RIGHT:
            return right_of(w, self.ints[0])
        if self.func == TF_REVERSE:
            return reverse_of(w)
        if self.func == TF_REPLACE:
            return replace_all(w, self.strs[0], self.strs[1])
        return w

    def to_int(self, w: String) -> Int:
        if self.func == TF_STRPOS:
            return strpos_of(w, self.strs[0])
        return char_count(w)

    def to_bool(self, w: String) -> Bool:
        if self.func == TF_STARTS:
            return w.startswith(self.strs[0])
        if self.func == TF_ENDS:
            return w.endswith(self.strs[0])
        if self.func == TF_CONTAINS:
            return w.find(self.strs[0]) >= 0
        return self.like.matches(w)


# ── reading a word as a number, a date or a boolean (CAST from TEXT), once per word ────────────────────
#
# DuckDB's text casts have quirks (digit separators, hex and binary integers, a doubled sign on doubles, date words like `inf`, exponents on decimals
# that round the wrong way), found by comparing against DuckDB word for word (tests/test_mojo_cast_fuzz.py). A word is read in one of three ways:
# it reads (`ok`), it certainly does not (a TRY_CAST makes it NULL, a CAST fails), or its meaning in DuckDB is a quirk not reproduced here (`unsure`: a
# CAST and a TRY_CAST both fail, so no answer is ever wrong).

struct Parsed(Copyable, Movable):
    """`ok`: the word reads as the type; `value`: its stored value (an integer, a scaled decimal, a day number, 0 / 1, a float's bits); `unsure`: not decided."""
    var ok: Bool
    var value: Int64
    var unsure: Bool

    def __init__(out self, ok: Bool, value: Int64 = 0, unsure: Bool = False):
        self.ok = ok
        self.value = value
        self.unsure = unsure


def _is_space(b: UInt8) -> Bool:
    return b == UInt8(32) or (b >= UInt8(9) and b <= UInt8(13))


def _is_digit_b(b: UInt8) -> Bool:
    return b >= UInt8(48) and b <= UInt8(57)


def _is_hex_b(b: UInt8) -> Bool:
    return _is_digit_b(b) or (b >= UInt8(97) and b <= UInt8(102)) or (b >= UInt8(65) and b <= UInt8(70))


def _trimmed(w: String) -> List[UInt8]:
    var b = w.as_bytes()
    var i = 0
    var end = len(b)
    while i < end and _is_space(b[i]):
        i += 1
    while end > i and _is_space(b[end - 1]):
        end -= 1
    var out = List[UInt8]()
    for k in range(i, end):
        out.append(b[k])
    return out^


def _unscore(t: List[UInt8], hexdigits: Bool, mut out: List[UInt8]) -> Bool:
    """t without its digit separators: an underscore is one only between two digits (hex digits for a hex number); any other underscore is False."""
    for i in range(len(t)):
        if t[i] == UInt8(95):
            var ok = i > 0 and i + 1 < len(t)
            if ok:
                ok = (_is_hex_b(t[i - 1]) and _is_hex_b(t[i + 1])) if hexdigits else (_is_digit_b(t[i - 1]) and _is_digit_b(t[i + 1]))
            if not ok:
                return False
            continue
        out.append(t[i])
    return True


struct _Num(Movable):
    var ok: Bool
    var neg: Bool
    var dig: List[UInt8]       # the significant digits (leading zeros dropped), at most 38
    var e10: Int               # the value is dig * 10^e10
    var has_exp: Bool
    var odd: Bool              # a number whose exponent has no digits (`1e`, `5e-`): DuckDB reads some of those and not others

    def __init__(out self):
        self.odd = False
        self.ok = False
        self.neg = False
        self.dig = List[UInt8]()
        self.e10 = 0
        self.has_exp = False


def _scan_number(t: List[UInt8]) -> _Num:
    """`[+-] digits [. digits] [e [+-] digits]` with at least one digit before the exponent (underscores already removed)."""
    var r = _Num()
    var n = len(t)
    var i = 0
    if n > 0 and (t[0] == UInt8(45) or t[0] == UInt8(43)):
        r.neg = t[0] == UInt8(45)
        i = 1
    var digits_seen = 0
    var frac = 0
    var in_frac = False
    while i < n:
        var c = t[i]
        if c == UInt8(46):
            if in_frac:
                return r^
            in_frac = True
            i += 1
            continue
        if not _is_digit_b(c):
            break
        digits_seen += 1
        if len(r.dig) == 0 and c == UInt8(48):
            pass                                             # a leading zero carries no weight
        else:
            if len(r.dig) >= 38:
                return r^                                    # more digits than this reads
            r.dig.append(c - 48)
        if in_frac:
            frac += 1
        i += 1
    if digits_seen == 0:
        return r^
    var ex = 0
    if i < n:
        if (t[i] | UInt8(32)) != UInt8(101):
            return r^
        i += 1
        var eneg = False
        if i < n and (t[i] == UInt8(45) or t[i] == UInt8(43)):
            eneg = t[i] == UInt8(45)
            i += 1
        if i >= n:
            r.odd = True
            return r^
        var ed = 0
        while i < n:
            if not _is_digit_b(t[i]):
                return r^
            if ed < 100000:
                ed = ed * 10 + Int(t[i]) - 48
            i += 1
        ex = -ed if eneg else ed
        r.has_exp = True
    r.e10 = ex - frac
    r.ok = True
    return r^


def _scaled(r: _Num, scale: Int, mut out: Int64) -> Bool:
    """The number times 10^scale as an int64, halves rounded away from zero (the first dropped digit decides); False when it does not fit. (No 128-bit
    arithmetic: the package is linked into apps whose C library has no 128-bit division.)"""
    var n = len(r.dig)
    var k = r.e10 + scale
    if n == 0:
        out = 0
        return True
    var keep = n
    var round_up = False
    if k < 0:
        var drop = -k
        if drop > n:
            out = 0
            return True
        keep = n - drop
        round_up = r.dig[keep] >= UInt8(5)
    var total = keep + (k if k > 0 else 0)
    if total > 19:
        return False
    var limit = UInt64(9223372036854775808) if r.neg else UInt64(9223372036854775807)
    var v = UInt64(0)
    for j in range(total):
        var d = UInt64(r.dig[j]) if j < keep else UInt64(0)
        if v > (limit - d) // 10:
            return False
        v = v * 10 + d
    if round_up:
        if v >= limit:
            return False
        v += 1
    out = (Int64.MIN if v == UInt64(9223372036854775808) else Int64(0) - Int64(v)) if r.neg else Int64(v)
    return True


def _parse_radix(t: List[UInt8], radix: Int, bits: Int) -> Parsed:
    """`0x1F_FF` / `0b1010`: an unsigned number of the integer type (underscores between digits)."""
    var digits = List[UInt8]()
    var body = List[UInt8]()
    for k in range(2, len(t)):
        body.append(t[k])
    if not _unscore(body, radix == 16, digits) or len(digits) == 0:
        return Parsed(False)
    var v = UInt64(0)
    var limit = UInt64(9223372036854775807)
    for d in digits:
        var x = 0
        if _is_digit_b(d):
            x = Int(d) - 48
        elif radix == 16 and _is_hex_b(d):
            x = (Int(d | UInt8(32)) - 97) + 10
        else:
            return Parsed(False)
        if x >= radix:
            return Parsed(False)
        if v > (limit - UInt64(x)) // UInt64(radix):
            return Parsed(False)
        v = v * UInt64(radix) + UInt64(x)
    if bits < 64 and v >= (UInt64(1) << UInt64(bits - 1)):
        return Parsed(False)
    return Parsed(True, Int64(v))


def parse_text(w: String, kind: Int, scale: Int, bits: Int, prec: Int) -> Parsed:
    """The word as a value of `kind` (a ColumnKind value): INT (its width in `bits`), DECIMAL (`scale` places, at most `prec` digits), FLOAT (`scale` is
    F32_COLUMN for a 32-bit float), DATE or BOOL; see the note above for `unsure`."""
    if kind == INT.value or kind == DECIMAL.value:
        var t = _trimmed(w)
        if len(t) == 0:
            return Parsed(False)
        var plain = len(t) == len(w.as_bytes())                   # (hex and binary only without spaces around)
        if kind == INT.value and len(t) > 2 and t[0] == UInt8(48) and plain:
            var c1 = t[1] | UInt8(32)
            if c1 == UInt8(120):
                return _parse_radix(t, 16, bits)
            if c1 == UInt8(98):
                return _parse_radix(t, 2, bits)
        var u = List[UInt8]()
        if not _unscore(t, False, u):
            return Parsed(False)
        var r = _scan_number(u)
        if not r.ok:
            return Parsed(False, 0, r.odd)
        if r.has_exp and (kind == DECIMAL.value or len(r.dig) > 18 or r.e10 > 30 or r.e10 < -30):
            return Parsed(False, 0, True)                     # DuckDB's exponents on decimals round oddly; a long mantissa with one overflows differently
        var v = Int64(0)
        if not _scaled(r, scale, v):
            return Parsed(False)
        if kind == INT.value and bits < 64:
            var lim = Int64(1) << Int64(bits - 1)
            if v < -lim or v >= lim:
                return Parsed(False)
        if kind == DECIMAL.value:
            var p = Int64(1)
            for _ in range(prec):
                p *= 10
            if v <= -p or v >= p:
                return Parsed(False)
        return Parsed(True, v)
    if kind == FLOAT.value:
        var t = _trimmed(w)
        if len(t) == 0:
            return Parsed(False)
        var u = List[UInt8]()
        if not _unscore(t, False, u):
            return Parsed(False)
        var at = 0
        if u[0] == UInt8(43):                                # DuckDB skips one plus sign itself, and the parser then reads an optional minus
            at = 1
            if len(u) == 1 or u[1] == UInt8(43):
                return Parsed(False)
        var text = String(unsafe_from_utf8=u[at:])
        var f = Float64(0.0)
        var st = parse_float_text(text, f)
        if st == F_RANGE:
            f = Float64(from_bits=UInt64(0xFFF0000000000000 if text.startswith("-") else 0x7FF0000000000000))
        elif st != F_OK:
            return Parsed(False)
        if scale == F32_COLUMN:
            f = Float64(Float32(f))
        return Parsed(True, Int64(f.to_bits()))
    if kind == DATE.value:
        return _parse_date_word(w)
    if kind == BOOL.value:
        var low = ascii_case(w, False)
        if low == "true" or low == "t" or low == "yes" or low == "y" or low == "1":
            return Parsed(True, 1)
        if low == "false" or low == "f" or low == "no" or low == "n" or low == "0":
            return Parsed(True, 0)
        return Parsed(False)
    return Parsed(False)


def _parse_date_word(w: String) -> Parsed:
    """DuckDB's date reader: leading spaces, [-] year digits, one of ` - / \\` twice, a month and a day of one or two digits, then anything that does not start
    with a digit (a time, text). The words that mean other dates (`infinity`, `epoch`, `now`, `today`, ... and their abbreviations), BC dates and years past 9999
    are not decided here."""
    var b = w.as_bytes()
    var n = len(b)
    var pos = 0
    while pos < n and _is_space(b[pos]):
        pos += 1
    if pos >= n:
        return Parsed(False)
    var neg = False
    if b[pos] == UInt8(45):
        neg = True
        pos += 1
        if pos >= n:
            return Parsed(False)
    var c0 = b[pos] | UInt8(32)
    if c0 >= UInt8(97) and c0 <= UInt8(122):                  # a word: the specials (infinity, epoch, now, today, tomorrow, yesterday) or not a date
        if c0 == UInt8(105) or c0 == UInt8(101) or c0 == UInt8(110) or c0 == UInt8(116) or c0 == UInt8(121):
            return Parsed(False, 0, True)
        return Parsed(False)
    var year = 0
    var ydigits = 0
    while pos < n and _is_digit_b(b[pos]):
        year = year * 10 + Int(b[pos]) - 48
        ydigits += 1
        pos += 1
        if year > 99999999:
            return Parsed(False)
    if pos >= n or ydigits == 0:
        return Parsed(False)
    var sep = b[pos]
    pos += 1
    if sep != UInt8(32) and sep != UInt8(45) and sep != UInt8(47) and sep != UInt8(92):
        return Parsed(False)
    var month = 0
    var md = 0
    while pos < n and md < 2 and _is_digit_b(b[pos]):
        month = month * 10 + Int(b[pos]) - 48
        md += 1
        pos += 1
    if md == 0 or pos >= n or b[pos] != sep:
        return Parsed(False)
    pos += 1
    var day = 0
    var dd = 0
    while pos < n and dd < 2 and _is_digit_b(b[pos]):
        day = day * 10 + Int(b[pos]) - 48
        dd += 1
        pos += 1
    if dd == 0:
        return Parsed(False)
    if pos < n and _is_digit_b(b[pos]):
        return Parsed(False)
    var y = -year if neg else year
    if month < 1 or month > 12 or day < 1:
        return Parsed(False)
    if y < 1 or y > 9999:
        # a valid date in DuckDB's range that this engine's text and arithmetic do not cover (BC dates, five-digit years), else not a date
        if day <= 31 and y > -5900000 and y <= 5900000:
            return Parsed(False, 0, True)
        return Parsed(False)
    if day > _days_in_month(y, month):
        return Parsed(False)
    # a BC marker after the day changes the year: not decided
    var rest = pos
    while rest < n and _is_space(b[rest]):
        rest += 1
    if rest + 1 < n and (b[rest] | UInt8(32)) == UInt8(98) and (b[rest + 1] | UInt8(32)) == UInt8(99):
        return Parsed(False, 0, True)
    if rest < n and b[rest] == UInt8(40):
        return Parsed(False, 0, True)
    return Parsed(True, Int64(days_from_civil(y, month, day)))
