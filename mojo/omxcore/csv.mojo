"""CSV and TSV reader (docs/omxcore-mojo-api.md section 2): a file in, a typed `Table` and an import report out.

    var r = read_csv("sales.csv", options=CsvOptions(infer_rows=100_000))
    r.table; r.rows; r.seconds
    for c in r.columns: c.name, c.kind, c.scale, c.empty, c.distinct, c.sample, c.failed

How it reads (no Python; a float is parsed from its digits, correctly rounded, F-157):
  1. The file is memory-mapped (no copy of the whole file); a UTF-8 byte order mark is skipped.
  2. The header row and then the first `infer_rows` rows choose each column's kind (below).
  3. Rows are split at safe boundaries so threads can parse in parallel. A newline ends a row only outside quotes, and
     quotes pair up (a doubled quote is two toggles), so the quote state at any byte is the parity of the quotes
     before it: a parallel pass counts each chunk's quotes, a prefix XOR gives every chunk's starting state, and each
     chunk then finds its first row start exactly. No guessing, no fallback.
  4. Each chunk parses its rows straight into int64 columns (no String per cell: numbers are parsed from the bytes,
     text goes through a byte-keyed dictionary), then the chunks are stitched together in file order.

Kinds, by the first `infer_rows` rows, empty cells ignored:
  every value a whole number                       INT
  numbers, some with a decimal point               DECIMAL, scale 2 if no value has more than 2 decimal places
                                                   (trailing zeros do not count), else scale 4. The stored value is
                                                   parsed from the digits (exact); extra digits round half away
                                                   from zero.
  a number with an exponent (`1e5`, `2.5E-3`), nan /    FLOAT (float64, correctly rounded; every NaN is one NaN).
  inf / -inf / infinity (any case) beside numbers,      A whole number or a decimal of up to 4 places stays INT /
  more than 4 decimal places (trailing zeros do not     DECIMAL, which are exact. As DuckDB does, 5 or more places
  count), or a number too wide for int64 at its scale   is a float: an exact DECIMAL at 5 to 9 places has no wider
  (DuckDB reads those as DOUBLE)                        integer behind it, so `sum(rate * qty)` would raise
                                                        "passed the int64 range" where DuckDB answers.
                                                        `CsvOptions(float_columns=[...])` reads the named columns
                                                        as FLOAT, `decimal_columns=[...]` as an exact DECIMAL at
                                                        the scale the first rows need (every value of them checked
                                                        to fit int64; a later value that does not fit is a failed
                                                        cell). A FLOAT cell that is not a number, or is past about
                                                        1.8e308, is a failed cell (a null).
  YYYY-MM-DD, or an ISO timestamp                  DATE (days since 1970). A timestamp with a zone offset is
                                                   converted to UTC first, then floored to the day (as the Parquet
                                                   reader does); one without is taken as written.
  true / false in any case                         BOOL
  anything else (or nothing but empty cells)       TEXT
An empty cell (or `""`) is a null. A value past the window that does not fit its column's kind is not an error: the
cell becomes a null and the column's report gets a failed row, "row 41,207: `12.x` isn't a number".

File format decisions:
  - RFC 4180 quoting, lenient like pyarrow: a quote opens a quoted field only at the start of a field, a doubled
    quote inside one is a literal quote, and a quote anywhere else (`5" pipe`) is an ordinary character. A quote
    left open runs to the end of the file as one cell and is reported in `ragged` ("has an unmatched quote").
    The parallel pre-scan runs the same four-state machine over every chunk from each possible start state and
    composes the results, so chunk starts are exact.
  - Rows end at `\\n`; a `\\r` right before it is dropped. Completely empty lines are skipped.
  - A row with fewer cells than the header gets nulls; a row with more keeps its first cells and is reported in
    `ragged` (row number and the row's text), the extra cells are dropped.
  - Row numbers count data rows from 1 (the header is not a row). Failed rows are capped at 1000 per column
    (`failed_count` has the true number).
  - Columns past `max_columns` are not read (`skipped` names them). Empty and repeated header names become
    `column_N` and `name_2`, `name_3`...
  - A DECIMAL's scale covers the whole column: inferred 2, it widens to 4 when a later value has 3 or 4 places
    (earlier values times 100; a value that no longer fits becomes a failed cell). More than 4 places round half
    away from zero and are counted in the column's `rounded`. A column whose window values overflow int64 at its
    scale is TEXT.
  - Also: a file truncated while it is mapped can crash the reader (SIGBUS); the delimiter is chosen by file
    extension only; leading zeros are lost in numbers (`007` is 7).
  - Past the window, an exponent in an INT or DECIMAL column is a failed cell (its kind is already chosen).
  - Not supported yet: thousands separators, a lone `\\r` as a line end, a
    non-UTF-8 encoding. The parsed columns are held twice for a moment while the chunks are stitched together.
"""
from std.ffi import external_call
from std.sys import CompilationTarget
from std.memory import Pointer
from std.time import perf_counter_ns
from std.collections import Dict
from .job import JobCtl, STAGE_READ, STAGE_FINISH      # F-175
from .utf8 import utf8_seq_len
from .table import Table, Column, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .kernels._fpenv import par_n, core_count

comptime BP = Pointer[UInt8, MutUntrackedOrigin]
comptime LP = Pointer[Int64, MutUntrackedOrigin]

comptime _QUOTE = 34
comptime _NL = 10
comptime _CR = 13
comptime _MAX_FAILED = 1000
comptime _SAMPLE_ROWS = 8
comptime _MAX_CHUNKS = 512


# ── public types ────────────────────────────────────────────────────────────────────────────────


struct CsvOptions(Copyable, Movable):
    """`delimiter`: one byte, "" picks a tab for `.tsv` / `.tab` files and a comma for the rest. `threads` 0 uses
    every core, 1 parses on one thread (the reference the tests compare to). `chunk_bytes`: the smallest piece
    given to a thread (lower it only to test). `float_columns`: header names read as FLOAT whatever the first rows show. `decimal_columns`: header names read as an exact DECIMAL at the scale their first rows need (up to 18 places), raising when a value of the first rows is not a plain number or does not fit int64.

    Only `scan_csv` (a file read a block at a time, csvscan.mojo) uses the last three: `group_rows`: the rows of one row group (the unit a
    query decodes), `block_bytes`: the most bytes of the file looked at at once (the sample for the kinds, and the window a row group is
    found in), `max_row_bytes`: the longest row accepted (a quote that never closes would otherwise make the rest of the file one row)."""

    var delimiter: String
    var has_header: Bool
    var max_columns: Int
    var infer_rows: Int
    var threads: Int
    var chunk_bytes: Int
    var float_columns: List[String]
    var decimal_columns: List[String]
    var group_rows: Int
    var block_bytes: Int
    var max_row_bytes: Int

    def __init__(out self, delimiter: String = "", has_header: Bool = True, max_columns: Int = 4096,
                 infer_rows: Int = 100_000, threads: Int = 0, chunk_bytes: Int = 1 << 22,
                 var float_columns: List[String] = List[String](), var decimal_columns: List[String] = List[String](),
                 group_rows: Int = 262_144, block_bytes: Int = 16 << 20, max_row_bytes: Int = 64 << 20):
        self.group_rows = group_rows
        self.block_bytes = block_bytes
        self.max_row_bytes = max_row_bytes
        self.float_columns = float_columns^
        self.decimal_columns = decimal_columns^
        self.delimiter = delimiter
        self.has_header = has_header
        self.max_columns = max_columns
        self.infer_rows = infer_rows
        self.threads = threads
        self.chunk_bytes = chunk_bytes


struct FailedRow(Copyable, Movable):
    """A row (or a cell in it) that could not be read: its number and the raw text, plus why."""

    var row: Int
    var text: String
    var reason: String

    def __init__(out self, row: Int, text: String, reason: String):
        self.row = row
        self.text = text
        self.reason = reason

    def message(self) -> String:
        return "row " + _commas(self.row) + ": `" + self.text + "` " + self.reason


struct CsvColumn(Copyable, Movable):
    """The Import step's view of one column."""

    var name: String
    var kind: ColumnKind
    var scale: Int
    var empty: Int  # empty cells (nulls from short rows count too)
    var distinct: Int  # TEXT: words in its dictionary, 0 for the other kinds
    var sample: List[String]  # the first rows, as written (quotes removed)
    var failed: List[FailedRow]  # cells that did not fit `kind`, now null (at most 1000)
    var failed_count: Int
    var rounded: Int  # DECIMAL: cells with more than 4 decimal places, rounded to 4

    def __init__(out self, name: String, kind: ColumnKind, scale: Int):
        self.rounded = 0
        self.name = name
        self.kind = kind
        self.scale = scale
        self.empty = 0
        self.distinct = 0
        self.sample = List[String]()
        self.failed = List[FailedRow]()
        self.failed_count = 0


struct CsvResult(Movable):
    var table: Table
    var rows: Int
    var seconds: Float64
    var columns: List[CsvColumn]
    var skipped: List[String]  # header names past `max_columns`
    var ragged: List[FailedRow]  # rows with more cells than the header (at most 1000)
    var ragged_count: Int

    def __init__(out self):
        self.table = Table()
        self.rows = 0
        self.seconds = 0.0
        self.columns = List[CsvColumn]()
        self.skipped = List[String]()
        self.ragged = List[FailedRow]()
        self.ragged_count = 0

    def take_table(mut self) -> Table:
        """The table, moved out (the result keeps an empty one), so `.share()` works while the report is still read."""
        var t = Table.empty()
        swap(t, self.table)
        return t^


# ── cell parsers (bytes in, exact int64 out; no String, no float) ──────────────────────────────


@always_inline
def _digit_ok(v: UInt64, d: UInt64, extra: UInt64) -> Bool:
    """Whether v * 10 + d still fits int64 (extra = 1 allows 2^63, for a negative number)."""
    return v < 922337203685477580 or (v == 922337203685477580 and d <= 7 + extra)


def _parse_int(p: BP, n: Int, mut out: Int64) -> Bool:
    var i = 0
    var neg = p[0] == 45
    if neg or p[0] == 43:
        i = 1
    if i >= n:
        return False
    var extra = UInt64(1) if neg else UInt64(0)
    var v = UInt64(0)
    while i < n:
        var d = Int(p[i]) - 48
        if d < 0 or d > 9 or not _digit_ok(v, UInt64(d), extra):
            return False
        v = v * 10 + UInt64(d)
        i += 1
    if neg:
        out = Int64.MIN if v == 9223372036854775808 else -Int64(v)
    else:
        out = Int64(v)
    return True


def _parse_decimal(p: BP, n: Int, scale: Int, mut out: Int64, mut places: Int) -> Bool:
    """"-12.5" at scale 2 is -1250. `places`: decimal places without trailing zeros (0 for "5", 1 for "1.50")."""
    var i = 0
    var neg = p[0] == 45
    if neg or p[0] == 43:
        i = 1
    var extra = UInt64(1) if neg else UInt64(0)
    var v = UInt64(0)
    var digits = 0
    var frac = 0  # decimal places seen
    var kept = 0  # decimal places stored
    var in_frac = False
    var round_up = False
    places = 0
    while i < n:
        var c = Int(p[i])
        i += 1
        if c == 46:
            if in_frac:
                return False
            in_frac = True
            continue
        var d = c - 48
        if d < 0 or d > 9:
            return False
        digits += 1
        if in_frac:
            frac += 1
            if d != 0:
                places = frac
            if kept == scale:
                if frac == scale + 1:
                    round_up = d >= 5
                continue
            kept += 1
        if not _digit_ok(v, UInt64(d), extra):
            return False
        v = v * 10 + UInt64(d)
    if digits == 0:
        return False
    while kept < scale:
        if not _digit_ok(v, 0, extra):
            return False
        v *= 10
        kept += 1
    if round_up:
        if v >= 9223372036854775807 + extra:
            return False
        v += 1
    if neg:
        out = Int64.MIN if v == 9223372036854775808 else -Int64(v)
    else:
        out = Int64(v)
    return True


comptime F_BAD = 0          # _parse_float: not a number
comptime F_OK = 1
comptime F_RANGE = 2        # a number past about 1.8e308 (or an exponent too long to be one)
comptime F_NAN_BITS = 0x7FF8000000000000


@always_inline
def _pow10(k: Int) -> Float64:
    """10^k for 0 <= k <= 22, exact (every partial product is below 1e22, whose odd part fits 53 bits)."""
    var r = Float64(1.0)
    if k & 1 != 0:
        r *= 1e1
    if k & 2 != 0:
        r *= 1e2
    if k & 4 != 0:
        r *= 1e4
    if k & 8 != 0:
        r *= 1e8
    if k & 16 != 0:
        r *= 1e16
    return r


@always_inline
def _lower(p: BP, at: Int) -> Int:
    return Int(p[at] | 32)


def _special(p: BP, i: Int, n: Int) -> Int:
    """What the bytes from i are: 1 `nan`, 2 `inf` or `infinity` (any case), 0 neither."""
    var m = n - i
    if m == 3 and _lower(p, i) == 110 and _lower(p, i + 1) == 97 and _lower(p, i + 2) == 110:
        return 1
    if m == 3 and _lower(p, i) == 105 and _lower(p, i + 1) == 110 and _lower(p, i + 2) == 102:
        return 2
    if m == 8 and _lower(p, i) == 105 and _lower(p, i + 1) == 110 and _lower(p, i + 2) == 102 and _lower(p, i + 3) == 105 \
       and _lower(p, i + 4) == 110 and _lower(p, i + 5) == 105 and _lower(p, i + 6) == 116 and _lower(p, i + 7) == 121:
        return 2
    return 0


def _parse_float(p: BP, n: Int, mut out: Float64) -> Int:
    """A float64 from its text: [+-] digits [. digits] [e|E [+-] digits], or nan / inf / infinity with an optional sign (any
    case). Correctly rounded: a mantissa that fits 53 bits with an exponent within +-22 is one exact multiplication or
    division (Clinger), everything else goes to the C library's strtod on a NUL-terminated copy. Every NaN is the one
    positive quiet NaN. Returns F_OK, F_BAD (not a number) or F_RANGE (past the largest float64)."""
    var i = 0
    var neg = False
    if n > 0 and (p[0] == 45 or p[0] == 43):
        neg = p[0] == 45
        i = 1
    if i >= n:
        return F_BAD
    var c0 = Int(p[i]) | 32
    if c0 == 110 or c0 == 105:                          # n, i: nan, inf, infinity
        var sp = _special(p, i, n)
        if sp == 0:
            return F_BAD
        out = Float64(from_bits=UInt64(F_NAN_BITS if sp == 1 else (0xFFF0000000000000 if neg else 0x7FF0000000000000)))
        return F_OK
    var w = UInt64(0)
    var kept = 0            # significant digits folded into w
    var dropped = 0         # integer-part digits past the 19th (each one is a factor of 10 lost)
    var inexact = False     # a nonzero digit was dropped
    var digits = 0
    var frac = 0            # fraction digits folded into w
    var in_frac = False
    var started = False     # a nonzero digit has been seen
    while i < n:
        var c = Int(p[i])
        if c == 46:
            if in_frac:
                return F_BAD
            in_frac = True
            i += 1
            continue
        var d = c - 48
        if d < 0 or d > 9:
            break
        digits += 1
        i += 1
        if d != 0:
            started = True
        if not started:                 # leading zeros carry no weight (but a fraction's still shift the exponent)
            if in_frac:
                frac += 1
            continue
        if kept < 19:
            w = w * 10 + UInt64(d)
            kept += 1
            if in_frac:
                frac += 1
        else:
            if d != 0:
                inexact = True
            if not in_frac:
                dropped += 1
    if digits == 0:
        return F_BAD
    var ex = 0
    if i < n:
        if (p[i] | 32) != 101:
            return F_BAD
        i += 1
        var eneg = False
        if i < n and (p[i] == 45 or p[i] == 43):
            eneg = p[i] == 45
            i += 1
        if i >= n:
            return F_BAD
        var ed = 0
        while i < n:
            var d = Int(p[i]) - 48
            if d < 0 or d > 9:
                return F_BAD
            if ed < 100000:
                ed = ed * 10 + d
            i += 1
        ex = -ed if eneg else ed
    var e10 = ex - frac + dropped
    if w == 0:
        out = -Float64(0.0) if neg else Float64(0.0)
        return F_OK
    if not inexact and w <= (UInt64(1) << 53) and -22 <= e10 <= 22:
        var f = Float64(w)
        f = f * _pow10(e10) if e10 >= 0 else f / _pow10(-e10)
        out = -f if neg else f
        return F_OK
    # the general case: strtod, which rounds correctly (glibc, libSystem), on a NUL-terminated copy
    var buf = List[UInt8](length=n + 1, fill=0)
    for k in range(n):
        buf[k] = p[k]
    # strtod follows LC_NUMERIC: a host that called setlocale (a comma for the point) would stop at the point. strtod_l with the "C" locale
    # does not (glibc, bionic and libSystem all have newlocale / strtod_l; the numeric category's mask is 2 on glibc, 16 on Apple).
    var name = List[UInt8](length=2, fill=0)
    name[0] = 67
    comptime NUMERIC_MASK = 2 if CompilationTarget.is_linux() else 16
    var loc = external_call["newlocale", Int](Int32(NUMERIC_MASK), Int(name.unsafe_ptr()), Int(0))
    var v: Float64
    if loc != 0:
        v = external_call["strtod_l", Float64](Int(buf.unsafe_ptr()), Int(0), loc)
        external_call["freelocale", NoneType](loc)
    else:
        v = external_call["strtod", Float64](Int(buf.unsafe_ptr()), Int(0))     # (no locale object: the process locale, normally "C")
    _ = name^
    _ = buf^
    if v - v != 0.0:                                    # inf: the text was past the largest float64
        return F_RANGE
    out = v
    return F_OK


@always_inline
def _two(p: BP, at: Int) -> Int:
    """The two digits at p[at], p[at + 1] as a number, or -1."""
    var a = Int(p[at]) - 48
    var b = Int(p[at + 1]) - 48
    return a * 10 + b if 0 <= a <= 9 and 0 <= b <= 9 else -1


def days_from_civil(y: Int, m: Int, d: Int) -> Int:
    """Days since 1970-01-01 of a calendar date (Howard Hinnant's algorithm)."""
    var yy = y - (1 if m <= 2 else 0)
    var era = (yy if yy >= 0 else yy - 399) // 400
    var yoe = yy - era * 400
    var doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    var doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _days_in_month(y: Int, m: Int) -> Int:
    if m == 2:
        return 29 if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0 else 28
    return 30 if m == 4 or m == 6 or m == 9 or m == 11 else 31


def _parse_date(p: BP, n: Int, mut out: Int64) -> Bool:
    """YYYY-MM-DD, or that followed by `T` or a space and HH:MM[:SS[.fraction]][Z | +HH[:MM]]."""
    if n < 10 or p[4] != 45 or p[7] != 45:
        return False
    var hi = _two(p, 0)
    var lo = _two(p, 2)
    var m = _two(p, 5)
    var d = _two(p, 8)
    if hi < 0 or lo < 0 or m < 1 or m > 12 or d < 1:
        return False
    var y = hi * 100 + lo
    if d > _days_in_month(y, m):
        return False
    var shift = 0  # whole days to add after converting the time to UTC
    if n > 10:
        if (p[10] != 84 and p[10] != 32) or n < 16 or p[13] != 58:
            return False
        var hh = _two(p, 11)
        var mm = _two(p, 14)
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            return False
        var secs = hh * 3600 + mm * 60
        var i = 16
        if i < n and p[i] == 58:
            if i + 3 > n:
                return False
            var ss = _two(p, i + 1)
            if ss < 0 or ss > 60:
                return False
            secs += ss
            i += 3
            if i < n and p[i] == 46:
                i += 1
                var start = i
                while i < n and p[i] >= 48 and p[i] <= 57:
                    i += 1
                if i == start:
                    return False
        if i < n:
            if p[i] == 90:
                i += 1
            elif p[i] == 43 or p[i] == 45:
                var oh = _two(p, i + 1) if i + 3 <= n else -1
                if oh < 0 or oh > 23:
                    return False
                var om = 0
                var sign = 1 if p[i] == 43 else -1
                i += 3
                if i < n and p[i] == 58:
                    i += 1
                if i < n:
                    om = _two(p, i) if i + 2 == n else -1
                    if om < 0 or om > 59:
                        return False
                    i += 2
                secs -= sign * (oh * 3600 + om * 60)  # to UTC
            if i != n:
                return False
        shift = secs // 86400  # floor: 23:30 at -05:00 is 04:30 UTC the next day
    out = Int64(days_from_civil(y, m, d) + shift)
    return True


def _parse_bool(p: BP, n: Int, mut out: Int64) -> Bool:
    """true / false in any case."""
    if n == 4 and (p[0] | 32) == 116 and (p[1] | 32) == 114 and (p[2] | 32) == 117 and (p[3] | 32) == 101:
        out = 1
        return True
    if n == 5 and (p[0] | 32) == 102 and (p[1] | 32) == 97 and (p[2] | 32) == 108 and (p[3] | 32) == 115 \
            and (p[4] | 32) == 101:
        out = 0
        return True
    return False


# ── small helpers ───────────────────────────────────────────────────────────────────────────────

def first_invalid_utf8(p: BP, n: Int) -> Int:
    """The byte offset of the first byte that is not well-formed UTF-8, or -1 when all of it is."""
    var i = 0
    while i < n:
        var b = p[i]
        if b < 0x80:
            i += 1
            continue
        var k = utf8_seq_len(b, p[i + 1] if i + 1 < n else 0, p[i + 2] if i + 2 < n else 0, p[i + 3] if i + 3 < n else 0, n - i)
        if k == 0:
            return i
        i += k
    return -1


def _check_text_encoding(p: BP, n: Int, path: String) raises:
    """A CSV is read as UTF-8 and nothing else: a UTF-16 byte order mark or a byte that is not UTF-8 (Latin-1 `caf\\xe9`) raises, naming the byte and line."""
    if n >= 2 and ((p[0] == 0xFF and p[1] == 0xFE) or (p[0] == 0xFE and p[1] == 0xFF)):
        raise Error("`" + path + "` is UTF-16 (it starts with a byte order mark), and omxcore reads UTF-8 only")
    var bad = first_invalid_utf8(p, n)
    if bad >= 0:
        var line = 1
        for i in range(bad):
            if p[i] == 10:
                line += 1
        raise Error("`" + path + "` is not valid UTF-8 (byte " + String(bad) + ", line " + String(line) + ")")


def _str(p: BP, n: Int) -> String:
    return String(unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](unsafe_ptr=p, length=n))


def _clip(p: BP, n: Int, limit: Int) -> String:
    """The bytes as text, cut to at most `limit` bytes without splitting a UTF-8 character."""
    if n <= limit:
        return _str(p, n)
    var m = limit
    while m > 0 and (p[m] & 0xC0) == 0x80:
        m -= 1
    return _str(p, m) + "..."


def _commas(n: Int) -> String:
    var s = String(n)
    var out = String()
    var first = s.byte_length() % 3
    for i in range(s.byte_length()):
        if i > 0 and (i - first) % 3 == 0:
            out += ","
        out += s[byte=i]
    return out


def _add_fail_in_order(mut fails: List[_Fail], var f: _Fail):
    """Keeps a chunk's failed cells in row order (the first `_MAX_FAILED`), whichever pass found them."""
    var at = len(fails)
    while at > 0 and fails[at - 1].row > f.row:
        at -= 1
    if at >= _MAX_FAILED:
        return
    fails.insert(at, f^)
    if len(fails) > _MAX_FAILED:
        _ = fails.pop()


def _scaled_text(v: Int64, scale: Int) -> String:
    """A stored decimal as written: 123456 at scale 2 is "1234.56"."""
    var digits = String(UInt64(v) if v >= 0 else UInt64(0) - UInt64(v))  # INT64_MIN has no positive Int64
    while digits.byte_length() <= scale:
        digits = "0" + digits
    var cut = digits.byte_length() - scale
    return ("-" if v < 0 else "") + String(digits[byte=:cut]) + "." + String(digits[byte=cut:])


def _reason(kind: Int) -> String:
    if kind == INT.value:
        return "isn't a whole number"
    if kind == DECIMAL.value or kind == FLOAT.value:
        return "isn't a number"
    if kind == DATE.value:
        return "isn't a date (YYYY-MM-DD)"
    return "isn't true or false"


struct WordTable(Movable):
    """A dictionary keyed by bytes: the code of a text cell is found without building a String; a word is stored once."""

    var arena: List[UInt8]
    var starts: List[Int]
    var lens: List[Int]
    var hashes: List[UInt64]
    var slots: List[Int32]  # code + 1, 0 = free
    var mask: Int

    def __init__(out self):
        self.arena = List[UInt8]()
        self.starts = List[Int]()
        self.lens = List[Int]()
        self.hashes = List[UInt64]()
        self.slots = List[Int32](length=64, fill=0)
        self.mask = 63

    def code_of(mut self, p: BP, n: Int) -> Int:
        var h = UInt64(14695981039346656037)
        for i in range(n):
            h = (h ^ UInt64(p[i])) * 1099511628211
        h ^= h >> 29
        var i = Int(h) & self.mask
        while True:
            var s = Int(self.slots[i])
            if s == 0:
                break
            var code = s - 1
            if self.hashes[code] == h and self.lens[code] == n:
                var base = self.arena.unsafe_ptr() + self.starts[code]
                var same = True
                for k in range(n):
                    if base[k] != p[k]:
                        same = False
                        break
                if same:
                    return code
            i = (i + 1) & self.mask
        var code = len(self.starts)
        self.starts.append(len(self.arena))
        self.lens.append(n)
        self.hashes.append(h)
        for k in range(n):
            self.arena.append(p[k])
        self.slots[i] = Int32(code + 1)
        if (code + 1) * 2 > len(self.slots):
            self._grow()
        return code

    def _grow(mut self):
        var size = len(self.slots) * 2
        self.slots = List[Int32](length=size, fill=0)
        self.mask = size - 1
        for code in range(len(self.starts)):
            var i = Int(self.hashes[code]) & self.mask
            while self.slots[i] != 0:
                i = (i + 1) & self.mask
            self.slots[i] = Int32(code + 1)

    def word(self, code: Int) -> String:
        return _str(BP(unsafe_from_address=Int(self.arena.unsafe_ptr()) + self.starts[code]), self.lens[code])


# ── the row walker: splits bytes into rows and cells, hands each cell to a sink ────────────────


trait CellSink:
    def cell(mut self, col: Int, p: BP, n: Int): ...
    def end_row(mut self, ncells: Int, row: BP, row_len: Int): ...
    def unmatched_quote(mut self, row: BP, row_len: Int): ...


# The quote rules are one small state machine, shared by the walker and the parallel pre-scan so they cannot differ:
#   A  at the start of a field        a quote opens a quoted field (Q); a delimiter or newline ends the field
#   U  inside an unquoted field       a quote is an ordinary character
#   Q  inside a quoted field          a quote is either the closing one or the first of a doubled pair (C)
#   C  just after a quote in Q        another quote: a literal quote, back to Q; otherwise the field closed
#                                     (delimiter / newline end it; anything else is kept literally, state U)
# A newline ends a row in every state but Q.
comptime _A = 0
comptime _U = 1
comptime _Q = 2
comptime _C = 3


@always_inline
def _step(state: Int, b: UInt8, delim: UInt8) -> Int:
    if state == _Q:
        return _C if b == _QUOTE else _Q
    if b == delim or b == _NL:
        return _A
    if b == _QUOTE and (state == _A or state == _C):
        return _Q
    return _U


def _walk[S: CellSink](data: BP, start: Int, end: Int, delim: UInt8, max_rows: Int, mut sink: S) -> Int:
    """Reads rows from `start` up to `end` (a row start, or the end of the data) or `max_rows` rows, whichever
    comes first; returns the position after the last row read. Quoted cells are unquoted into a scratch buffer
    before the sink sees them; everything else is passed in place."""
    var pos = start
    var rows = 0
    var scratch = List[UInt8]()
    while pos < end and rows < max_rows:
        if data[pos] == _NL:
            pos += 1
            continue
        if data[pos] == _CR and (pos + 1 == end or data[pos + 1] == _NL):
            pos += 2 if pos + 1 < end else 1
            continue
        var row_start = pos
        var col = 0
        var unmatched = False
        while True:
            var fstart = pos
            var st = _A
            var quoted = pos < end and data[pos] == _QUOTE
            while pos < end:
                var b = data[pos]
                if st == _Q:
                    if b == _QUOTE:
                        st = _C
                elif b == delim or b == _NL:
                    break
                else:
                    st = _step(st, b, delim)
                pos += 1
            var fend = pos
            var row_done = pos >= end or data[pos] == _NL
            if st == _Q:
                unmatched = True
            elif row_done and fend > fstart and data[fend - 1] == _CR:
                fend -= 1
            if not quoted:
                sink.cell(col, data + fstart, fend - fstart)
            else:
                if len(scratch) < fend - fstart:
                    scratch.resize(fend - fstart, 0)
                var out = scratch.unsafe_ptr()    # lifetime: ok (`scratch` lives through the enclosing loop)
                var n = 0
                var q = _A
                for i in range(fstart, fend):
                    var b = data[i]
                    if b != _QUOTE:
                        out[n] = b
                        n += 1
                    elif q == _C:  # the second of a doubled pair
                        out[n] = _QUOTE
                        n += 1
                    elif q == _U:  # a quote inside an unquoted tail: literal
                        out[n] = _QUOTE
                        n += 1
                    q = _step(q, b, delim)
                sink.cell(col, BP(unsafe_from_address=Int(out)), n)
            col += 1
            if row_done:
                if unmatched:
                    sink.unmatched_quote(data + row_start, fend - row_start)
                sink.end_row(col, data + row_start, fend - row_start)
                if pos < end:
                    pos += 1
                break
            pos += 1
        rows += 1
    return pos


# ── sinks ───────────────────────────────────────────────────────────────────────────────────────


struct HeaderSink(CellSink, Movable):
    var names: List[String]
    var cells: Int

    def __init__(out self):
        self.names = List[String]()
        self.cells = 0

    def cell(mut self, col: Int, p: BP, n: Int):
        self.names.append(_str(p, n))

    def end_row(mut self, ncells: Int, row: BP, row_len: Int):
        self.cells = ncells

    def unmatched_quote(mut self, row: BP, row_len: Int):
        pass


struct InferCol(Copyable, Movable):
    var can_int: Bool
    var can_dec: Bool
    var can_date: Bool
    var can_bool: Bool
    var can_float: Bool
    var float_only: Bool     # a value only a float can be: an exponent, nan / inf (a number too wide for int64 is decided in `_parse`)
    var numeric: Bool        # some value had a digit
    var places: Int
    var nonempty: Int
    var sample: List[String]

    def __init__(out self):
        self.can_int = True
        self.can_dec = True
        self.can_date = True
        self.can_bool = True
        self.can_float = True
        self.float_only = False
        self.numeric = False
        self.places = 0
        self.nonempty = 0
        self.sample = List[String]()


struct InferSink(CellSink, Movable):
    """Chooses each column's kind from the rows it is given."""

    var ncols: Int
    var cols: List[InferCol]

    def __init__(out self, ncols: Int):
        self.ncols = ncols
        self.cols = List[InferCol]()
        for _ in range(ncols):
            self.cols.append(InferCol())

    def cell(mut self, col: Int, p: BP, n: Int):
        if col >= self.ncols:
            return
        ref s = self.cols[col]
        if len(s.sample) < _SAMPLE_ROWS:
            s.sample.append(_clip(p, n, 100))
        if n == 0:
            return
        s.nonempty += 1
        var v = Int64(0)
        if s.can_int and not _parse_int(p, n, v):
            s.can_int = False
        if s.can_dec:
            var places = 0
            if _parse_decimal(p, n, 0, v, places):  # (the scale is chosen from `places` afterwards)
                if places > s.places:
                    s.places = places
            else:
                s.can_dec = False
        if s.can_date and not _parse_date(p, n, v):
            s.can_date = False
        if s.can_bool and not _parse_bool(p, n, v):
            s.can_bool = False
        if s.can_float:
            var f = Float64(0.0)
            if _parse_float(p, n, f) == F_BAD:               # (a number past float64 is still a number: its cell fails later)
                s.can_float = False
            else:
                var plain = _special(p, 1 if (p[0] == 45 or p[0] == 43) else 0, n) == 0
                s.numeric = s.numeric or plain
                for k in range(n):                       # an exponent, or nan / inf: only a float reads it
                    if (p[k] | 32) == 101 or not plain:
                        s.float_only = True
                        break

    def end_row(mut self, ncells: Int, row: BP, row_len: Int):
        pass

    def unmatched_quote(mut self, row: BP, row_len: Int):
        pass


struct FitSink(CellSink, Movable):
    """Checks that a DECIMAL column's window values fit int64 at the scale chosen for it (a column that does not
    fit is read as text instead)."""

    var scales: List[Int]  # -1: not a DECIMAL column
    var fits: List[Bool]

    def __init__(out self, kinds: List[Int], scales: List[Int]):
        self.scales = List[Int]()
        self.fits = List[Bool]()
        for c in range(len(kinds)):
            self.scales.append(scales[c] if kinds[c] == DECIMAL.value else -1)
            self.fits.append(True)

    def cell(mut self, col: Int, p: BP, n: Int):
        if col >= len(self.scales) or self.scales[col] < 0 or n == 0 or not self.fits[col]:
            return
        var v = Int64(0)
        var places = 0
        if not _parse_decimal(p, n, self.scales[col], v, places):
            self.fits[col] = False

    def end_row(mut self, ncells: Int, row: BP, row_len: Int):
        pass

    def unmatched_quote(mut self, row: BP, row_len: Int):
        pass


struct _Fail(Copyable, Movable):
    var row: Int  # within the chunk
    var text: String
    var reason: String

    def __init__(out self, row: Int, text: String, reason: String):
        self.row = row
        self.text = text
        self.reason = reason


struct ChunkCol(Movable):
    var kind: Int
    var scale: Int
    var vals: List[Int64]
    var nulls: List[Int]  # rows (within the chunk) that are null
    var words: WordTable
    var remap: List[Int64]  # chunk code -> table code
    var empty: Int
    var fails: List[_Fail]
    var fail_count: Int
    var wide_rows: List[Int]  # DECIMAL at scale 2: cells with 3 or 4 places, kept at scale 4 here (the column widens)
    var wide_vals: List[Int64]
    var rounded: Int  # cells with more than 4 decimal places
    var active: Bool  # False: the column is not read at all (a scan reads the columns a query uses)
    var count_only: Bool  # a TEXT column whose words nobody asked for: its cells are only counted

    def __init__(out self, kind: Int, scale: Int):
        self.active = True
        self.count_only = False
        self.kind = kind
        self.scale = scale
        self.vals = List[Int64]()
        self.nulls = List[Int]()
        self.words = WordTable()
        self.remap = List[Int64]()
        self.empty = 0
        self.fails = List[_Fail]()
        self.fail_count = 0
        self.wide_rows = List[Int]()
        self.wide_vals = List[Int64]()
        self.rounded = 0


struct ChunkSink(CellSink, Movable):
    """Parses the rows of one chunk into typed columns."""

    var ncols: Int
    var cols: List[ChunkCol]
    var rows: Int
    var ragged: List[_Fail]
    var ragged_count: Int
    var header_cells: Int

    def __init__(out self, kinds: List[Int], scales: List[Int], header_cells: Int, want: List[Bool] = List[Bool](),
                 text_words: Bool = True):
        """`want`: the columns to read (empty: all). `text_words` False: a TEXT column's words are not collected (its empty cells are still counted)."""
        self.header_cells = header_cells
        self.ncols = len(kinds)
        self.cols = List[ChunkCol]()
        for c in range(len(kinds)):
            self.cols.append(ChunkCol(kinds[c], scales[c]))
            if len(want) > 0 and not want[c]:
                self.cols[c].active = False
            self.cols[c].count_only = kinds[c] == TEXT.value and not text_words
        self.rows = 0
        self.ragged = List[_Fail]()
        self.ragged_count = 0

    def cell(mut self, col: Int, p: BP, n: Int):
        if col >= self.ncols:
            return
        ref c = self.cols[col]
        if not c.active:
            return
        if n == 0:
            c.vals.append(0)
            c.nulls.append(self.rows)
            c.empty += 1
            return
        var v = Int64(0)
        var ok = True
        var reason = String()
        if c.kind == TEXT.value:
            v = 0 if c.count_only else Int64(c.words.code_of(p, n))
        elif c.kind == INT.value:
            ok = _parse_int(p, n, v)
        elif c.kind == DECIMAL.value:
            var places = 0
            ok = _parse_decimal(p, n, c.scale, v, places)
            if ok and places > c.scale:
                if c.scale == 2:  # more places than the column was inferred with: it widens to scale 4
                    ok = _parse_decimal(p, n, 4, v, places)
                    if ok:
                        c.wide_rows.append(self.rows)
                        c.wide_vals.append(v)
                        v = 0  # (the scale-2 slot is filled when the column is widened)
                if ok and places > 4:
                    c.rounded += 1
        elif c.kind == DATE.value:
            ok = _parse_date(p, n, v)
        elif c.kind == FLOAT.value:
            var f = Float64(0.0)
            var st = _parse_float(p, n, f)
            ok = st == F_OK
            if ok:
                v = Int64(f.to_bits())
            elif st == F_RANGE:
                reason = "is too big for a float64 (about 1.8e308)"
        else:
            ok = _parse_bool(p, n, v)
        if ok:
            c.vals.append(v)
        else:
            c.vals.append(0)
            c.nulls.append(self.rows)
            c.fail_count += 1
            if len(c.fails) < _MAX_FAILED:
                c.fails.append(_Fail(self.rows, _clip(p, n, 200), reason if reason.byte_length() > 0 else _reason(c.kind)))

    def end_row(mut self, ncells: Int, row: BP, row_len: Int):
        for col in range(ncells, self.ncols):  # a short row: the missing cells are empty
            if not self.cols[col].active:
                continue
            self.cols[col].vals.append(0)
            self.cols[col].nulls.append(self.rows)
            self.cols[col].empty += 1
        if ncells > self.header_cells:
            self._report(row, row_len, "has " + String(ncells) + " cells, the header has " +
                         String(self.header_cells) + "; the extra cells were dropped")
        self.rows += 1

    def unmatched_quote(mut self, row: BP, row_len: Int):
        self._report(row, row_len, "has an unmatched quote: the rest of the file was read as one cell")

    def _report(mut self, row: BP, row_len: Int, reason: String):
        self.ragged_count += 1
        if len(self.ragged) < _MAX_FAILED:
            self.ragged.append(_Fail(self.rows, _clip(row, row_len, 200), reason))


# ── the reader ──────────────────────────────────────────────────────────────────────────────────


def read_csv(path: String, options: CsvOptions = CsvOptions(), job: JobCtl = JobCtl()) raises -> CsvResult:
    """Reads a CSV or TSV file into a `Table` with an import report. Raises (readable message) only when the file
    cannot be opened or the options are unusable; bad values never raise, they become nulls and failed rows.
    `job` (F-175, job.mojo): the background job this runs for. Its progress counts the file's bytes (the rows are not known
    before they are parsed), and a cancel is seen between the chunks the file is cut into (a file read by one thread is one chunk)."""
    var start_ns = perf_counter_ns()
    var delim = _delimiter(path, options.delimiter)
    if options.max_columns < 1:
        raise Error("max_columns must be at least 1, not " + String(options.max_columns))
    var file: FileHandle
    try:
        file = open(path, "r")
    except:
        raise Error("couldn't open `" + path + "`: no such file, or no permission to read it")
    var fd = file._get_raw_fd()
    var one = List[UInt8](length=1, fill=0)
    var got = external_call["read", Int](fd, one.unsafe_ptr(), Int(1))
    _ = one^
    if got < 0:                                                       # a directory opens, but cannot be read
        file.close()
        raise Error("couldn't read `" + path + "`: not a regular file")
    var size = external_call["lseek", Int](fd, Int(0), Int32(2))
    if size < 0:
        file.close()
        raise Error("couldn't read `" + path + "`: not a regular file")
    var result = CsvResult()
    if size > 0:
        var addr = external_call["mmap", Int](Int(0), size, Int32(1), Int32(2), fd, Int(0))
        file.close()  # the mapping stays valid
        if addr == -1:
            raise Error("couldn't read `" + path + "`: the file could not be mapped into memory")
        try:
            job.stage(STAGE_READ)
            job.count_bytes()
            job.add_total(size)
            _check_text_encoding(BP(unsafe_from_address=addr), size, path)
            _parse(BP(unsafe_from_address=addr), size, delim, options, result, job)
        except e:
            _ = external_call["munmap", Int32](addr, size)
            raise Error(String(e))
        _ = external_call["munmap", Int32](addr, size)
    else:
        file.close()
    result.seconds = Float64(perf_counter_ns() - start_ns) / 1e9
    return result^


def _delimiter(path: String, requested: String) raises -> UInt8:
    if requested == "":
        return 9 if path.endswith(".tsv") or path.endswith(".tab") else 44
    if len(requested.as_bytes()) != 1:
        raise Error("the delimiter must be one character, not `" + requested + "`")
    var b = requested.as_bytes()[0]
    if b == _QUOTE or b == _NL or b == _CR:
        raise Error("the delimiter can't be a quote or a line break")
    return b


struct CsvShape(Movable):
    """What the first rows say about a file: the columns' names, kinds and scales, and where the data starts."""
    var names: List[String]
    var kinds: List[Int]
    var scales: List[Int]
    var header_cells: Int            # 0: nothing but blank lines
    var first_row: Int               # the byte offset of the first data row
    var skipped: List[String]        # header names past `max_columns`
    var infer: InferSink

    def __init__(out self):
        self.names = List[String]()
        self.kinds = List[Int]()
        self.scales = List[Int]()
        self.header_cells = 0
        self.first_row = 0
        self.skipped = List[String]()
        self.infer = InferSink(0)


def _infer_shape(data: BP, size: Int, delim: UInt8, options: CsvOptions) raises -> CsvShape:
    """The header and the kinds, from the first `infer_rows` rows of `data[:size]` (which ends at a row boundary)."""
    var shape = CsvShape()
    var body = 0
    if size >= 3 and data[0] == 0xEF and data[1] == 0xBB and data[2] == 0xBF:
        body = 3

    # header: the first row's cells (or their count, when the file has no header)
    var head = HeaderSink()
    var after_first = _walk(data, body, size, delim, 1, head)
    if head.cells == 0:  # nothing but blank lines
        return shape^
    shape.header_cells = head.cells
    var ncols = min(head.cells, options.max_columns)
    var names = List[String]()
    var seen = Dict[String, Int]()
    for c in range(ncols):
        var name = head.names[c] if options.has_header else String()
        if name == "":
            name = "column_" + String(c + 1)
        if name in seen:
            var k = seen[name] + 1
            seen[name] = k
            var alt = name + "_" + String(k)
            while alt in seen:
                k += 1
                alt = name + "_" + String(k)
            seen[name] = k
            seen[alt] = 1
            name = alt
        else:
            seen[name] = 1
        names.append(name)
    if options.has_header:
        for c in range(ncols, head.cells):
            shape.skipped.append(head.names[c])
    var first_row = after_first if options.has_header else body

    # kinds from the first `infer_rows` rows
    var infer = InferSink(ncols)
    _ = _walk(data, first_row, size, delim, options.infer_rows, infer)
    var kinds = List[Int]()
    var scales = List[Int]()
    for c in range(ncols):
        ref s = infer.cols[c]
        var kind = TEXT.value
        var scale = 0
        var asked = False
        for name in options.float_columns:
            asked = asked or name == names[c]
        var exact = False
        for name in options.decimal_columns:
            exact = exact or name == names[c]
        if asked and exact:
            raise Error("column `" + names[c] + "` is named in both float_columns and decimal_columns")
        if asked:
            kind = FLOAT.value
        elif exact:
            if s.nonempty > 0 and not s.can_dec:
                raise Error("column `" + names[c] + "` cannot be read as an exact DECIMAL: a value in its first rows is not a plain number, or does not fit int64 (a DECIMAL holds at most 18 digits)")
            kind = DECIMAL.value                                  # opt-in: exact at the scale its first rows need (2, 4, or the places up to 18)
            scale = 2 if s.places <= 2 else 4 if s.places <= 4 else s.places
        elif s.nonempty > 0:
            if s.can_float and s.numeric and s.float_only:
                kind = FLOAT.value                                # an exponent, nan / inf
            elif s.can_int:
                kind = INT.value
            elif s.can_dec and s.places <= 4:
                kind = DECIMAL.value
                scale = 2 if s.places <= 2 else 4
            elif s.can_float and s.numeric:
                kind = FLOAT.value                                # more than 4 places, or a number too wide for int64 (DuckDB reads those as DOUBLE)
            elif s.can_date:
                kind = DATE.value
            elif s.can_bool:
                kind = BOOL.value
        kinds.append(kind)
        scales.append(scale)
    var any_decimal = False
    for c in range(ncols):
        any_decimal = any_decimal or kinds[c] == DECIMAL.value
    if any_decimal:  # the scale is known now: a column whose window values overflow int64 at it is read as text
        var fit = FitSink(kinds, scales)
        _ = _walk(data, first_row, size, delim, options.infer_rows, fit)
        for c in range(ncols):
            if kinds[c] == DECIMAL.value and not fit.fits[c]:
                for name in options.decimal_columns:
                    if name == names[c]:
                        raise Error("column `" + names[c] + "` cannot be read as an exact DECIMAL: a value in its first rows does not fit int64 at " + String(scales[c]) + " decimal places")
                kinds[c] = FLOAT.value if infer.cols[c].can_float and infer.cols[c].numeric else TEXT.value     # too wide at its scale
                scales[c] = 0
    shape.names = names^
    shape.kinds = kinds^
    shape.scales = scales^
    shape.first_row = first_row
    shape.infer = infer^
    return shape^


def _parse(data: BP, size: Int, delim: UInt8, options: CsvOptions, mut result: CsvResult, job: JobCtl = JobCtl()) raises:
    var shape = _infer_shape(data, size, delim, options)
    if shape.header_cells == 0:  # nothing but blank lines
        return
    result.skipped = shape.skipped.copy()
    var names = shape.names.copy()
    var kinds = shape.kinds.copy()
    var scales = shape.scales.copy()
    var first_row = shape.first_row
    var head = HeaderSink()
    head.cells = shape.header_cells
    job.advance(first_row)                                       # (the bytes before the first row count as done)

    # chunks: safe row boundaries found with a parallel pre-scan of the quote state machine
    var workers = options.threads if options.threads > 0 else core_count()
    var span = size - first_row
    var nch = 1
    if workers > 1 and span > 0:
        nch = max(1, min(span // max(options.chunk_bytes, 1), _MAX_CHUNKS))
    var bounds = List[Int](length=nch + 1, fill=0)
    for i in range(nch + 1):
        bounds[i] = first_row + span * i // nch
    var rstart = List[Int](length=nch + 1, fill=0)
    rstart[nch] = size
    if nch > 1:
        # The quote state at a byte depends on everything before it, so each chunk is run from all four possible
        # start states in parallel (what state does it end in, from each?). Composing those one after another
        # gives every chunk's true start state, exactly; then each chunk looks for its first row start. A chunk
        # begins one byte early (at the boundary - 1), so a row starting exactly at the boundary is found.
        var ends = List[Int](length=nch * 4, fill=0)
        var ends_at = Int(ends.unsafe_ptr())
        var bounds_at = Int(bounds.unsafe_ptr())
        def run_states(i: Int) {imm}:
            if job.cancelled():
                return
            var b = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=bounds_at)
            var out = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=ends_at)
            var lo = first_row if i == 0 else b[i] - 1
            var hi = b[i + 1] - 1 if i + 1 < nch else size
            var s0 = _A
            var s1 = _U
            var s2 = _Q
            var s3 = _C
            for q in range(lo, hi):
                var byte = data[q]
                s0 = _step(s0, byte, delim)
                s1 = _step(s1, byte, delim)
                s2 = _step(s2, byte, delim)
                s3 = _step(s3, byte, delim)
            out[i * 4 + _A] = s0
            out[i * 4 + _U] = s1
            out[i * 4 + _Q] = s2
            out[i * 4 + _C] = s3
        par_n(run_states, nch, workers)
        var state = List[Int](length=nch, fill=_A)  # the state at each chunk's first byte
        for i in range(1, nch):
            state[i] = ends[(i - 1) * 4 + state[i - 1]]
        var rstart_at = Int(rstart.unsafe_ptr())
        var state_at = Int(state.unsafe_ptr())
        def find_start(t: Int) {imm}:
            if job.cancelled():
                return
            var i = t + 1  # task t finds where chunk t + 1 starts
            var b = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=bounds_at)
            var st = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=state_at)
            var out = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=rstart_at)
            var q = b[i] - 1
            var s = st[i]
            var found = size
            while q < size:
                var byte = data[q]
                if byte == _NL and s != _Q:
                    found = q + 1
                    break
                s = _step(s, byte, delim)
                q += 1
            out[i] = found
        par_n(find_start, nch - 1, workers)
        _ = state^
        _ = ends^
    rstart[0] = first_row
    for i in range(1, nch):  # keep the starts in order (a chunk with no row start of its own is empty)
        if rstart[i] < rstart[i - 1]:
            rstart[i] = rstart[i - 1]
    _ = bounds^

    var sinks = List[ChunkSink]()
    for _ in range(nch):
        sinks.append(ChunkSink(kinds, scales, head.cells))
    var sinks_at = Int(sinks.unsafe_ptr())
    var rstart_at2 = Int(rstart.unsafe_ptr())
    def parse_chunk(i: Int) {imm}:
        if job.cancelled():                                      # F-175: between chunks
            return
        var rs = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=rstart_at2)
        var sp = Pointer[ChunkSink, MutUntrackedOrigin](unsafe_from_address=sinks_at)
        _ = _walk(data, rs[i], rs[i + 1], delim, Int.MAX, (sp + i)[])
        job.advance(rs[i + 1] - rs[i])
    if nch == 1:
        parse_chunk(0)
    else:
        par_n(parse_chunk, nch, workers)
    _ = rstart^  # (the tasks read it by address)
    job.check()                                                  # a cancelled round left chunks unparsed: none is used
    job.stage(STAGE_FINISH)
    _stitch(sinks, names, kinds, scales, shape.infer, result, job)


def _stitch(mut sinks: List[ChunkSink], names: List[String], kinds: List[Int], mut scales: List[Int],
            infer: InferSink, mut result: CsvResult, job: JobCtl = JobCtl()) raises:
    """Joins the chunks, in file order, into the table and the report."""
    var nch = len(sinks)
    var ncols = len(names)
    var offsets = List[Int](length=nch + 1, fill=0)
    for i in range(nch):
        offsets[i + 1] = offsets[i] + sinks[i].rows
    var total = offsets[nch]
    var kind_of = List[ColumnKind]()
    var widen = List[Bool]()  # a scale-2 DECIMAL column that met values with 3 or 4 places: it becomes scale 4
    for c in range(ncols):
        var w = False
        for i in range(nch):
            w = w or len(sinks[i].cols[c].wide_rows) > 0
        widen.append(w)
        if w:
            scales[c] = 4
        kind_of.append(ColumnKind(kinds[c]))
        _ = result.table.add_column(names[c], kind_of[c], scales[c])
        result.table.cols[c].values.resize(total, 0)
        # a TEXT column's words: each chunk's words, in order, so codes follow first appearance in the file
        if kinds[c] == TEXT.value:
            for i in range(nch):
                ref cc = sinks[i].cols[c]
                for k in range(len(cc.words.starts)):
                    cc.remap.append(Int64(result.table.add_word(c, cc.words.word(k))))

    var sinks_at = Int(sinks.unsafe_ptr())
    var offsets_at = Int(offsets.unsafe_ptr())
    var tbl = Int(result.table.cols.unsafe_ptr())
    def copy_chunk(i: Int) {imm}:
        if job.cancelled():
            return
        var sp = Pointer[ChunkSink, MutUntrackedOrigin](unsafe_from_address=sinks_at)
        var off = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=offsets_at)
        var cols = Pointer[Column, MutUntrackedOrigin](unsafe_from_address=tbl)
        for c in range(ncols):
            var dst = LP(unsafe_from_address=Int(cols[c].values.unsafe_ptr())) + off[i]
            ref cc = (sp + i)[].cols[c]
            var n = len(cc.vals)
            var src = LP(unsafe_from_address=Int(cc.vals.unsafe_ptr()))
            if cc.kind == TEXT.value and len(cc.remap) > 0:
                var map = LP(unsafe_from_address=Int(cc.remap.unsafe_ptr()))
                for r in range(n):
                    dst[r] = map[Int(src[r])]
            elif widen[c]:
                var limit = Int64.MAX // 100
                for r in range(n):
                    var v = src[r]
                    if v > limit or v < -limit:  # too big to keep 4 places: a failed cell
                        dst[r] = 0
                        cc.nulls.append(r)
                        cc.fail_count += 1
                        _add_fail_in_order(cc.fails, _Fail(r, _scaled_text(v, 2),
                                           "is too big to keep 4 decimal places (the column needed them)"))
                    else:
                        dst[r] = v * 100
                for k in range(len(cc.wide_rows)):
                    dst[cc.wide_rows[k]] = cc.wide_vals[k]
            else:
                for r in range(n):
                    dst[r] = src[r]
    if nch == 1:
        copy_chunk(0)
    else:
        par_n(copy_chunk, nch, core_count())
    job.check()

    for c in range(ncols):
        var report = CsvColumn(names[c], kind_of[c], scales[c])
        report.sample = infer.cols[c].sample.copy()
        ref col = result.table.cols[c]
        var nulls = 0
        for i in range(nch):
            nulls += len(sinks[i].cols[c].nulls)
        if nulls > 0:
            col.validity = List[UInt8](length=(total + 7) // 8, fill=0xFF)
            col.nulls = nulls
            for i in range(nch):
                ref cc = sinks[i].cols[c]
                for k in range(len(cc.nulls)):
                    var row = offsets[i] + cc.nulls[k]
                    col.validity[row >> 3] &= ~(UInt8(1) << UInt8(row & 7))
                    col.values[row] = 0  # a null's slot holds 0, whatever the cell said
        for i in range(nch):
            ref cc = sinks[i].cols[c]
            report.empty += cc.empty
            report.failed_count += cc.fail_count
            report.rounded += cc.rounded
            for k in range(len(cc.fails)):
                if len(report.failed) < _MAX_FAILED:
                    report.failed.append(FailedRow(offsets[i] + cc.fails[k].row + 1, cc.fails[k].text,
                                                   cc.fails[k].reason))
        if kinds[c] == TEXT.value:
            report.distinct = len(col.dict[].words)
        result.columns.append(report^)
    for i in range(nch):
        result.ragged_count += sinks[i].ragged_count
        for k in range(len(sinks[i].ragged)):
            if len(result.ragged) < _MAX_FAILED:
                result.ragged.append(FailedRow(offsets[i] + sinks[i].ragged[k].row + 1, sinks[i].ragged[k].text,
                                               sinks[i].ragged[k].reason))
    result.rows = total


def parse_float_text(text: String, mut out: Float64) -> Int:
    """`_parse_float` over a String (a SQL literal): F_OK, F_BAD or F_RANGE."""
    var n = text.byte_length()
    if n == 0:
        return F_BAD
    return _parse_float(BP(unsafe_from_address=Int(text.unsafe_ptr())), n, out)
