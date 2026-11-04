"""Tables as text: JSON, JSON lines and CSV (F-151, used by the `mx` command).

Values are written exactly: an INT as its digits, a DECIMAL as its digits with the point in place (`12.50`, a JSON number whose text is
exact: read it with a decimal parser, not a float one), a DATE as "yyyy-mm-dd", TEXT as the word, BOOL as true / false, NULL as null
(CSV: an empty cell). A FLOAT is its shortest round-trip text; inf, -inf and NaN, which JSON cannot hold, are the strings "Infinity",
"-Infinity" and "NaN".
"""
from .utf8 import utf8_seq_len
from .table import Table, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .wide import wide_text, wide_make
from .floatfmt import float_text


def civil_date(days: Int) -> String:
    """'yyyy-mm-dd' of a day count since 1970-01-01."""
    var z = days + 719468
    var era = (z if z >= 0 else z - 146096) // 146097
    var doe = z - era * 146097
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var y = yoe + era * 400
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    var mp = (5 * doy + 2) // 153
    var d = doy - (153 * mp + 2) // 5 + 1
    var m = mp + 3 if mp < 10 else mp - 9
    return _pad(y + (1 if m <= 2 else 0), 4) + "-" + _pad(m, 2) + "-" + _pad(d, 2)


def _pad(v: Int, w: Int) -> String:
    var s = String(v)
    while s.byte_length() < w:
        s = "0" + s
    return s


def decimal_text(v: Int64, scale: Int) -> String:
    """A stored decimal as written: 123456 at scale 2 is 1234.56."""
    var neg = v < 0
    var digits = String(UInt64(v) if not neg else UInt64(0) - UInt64(v))
    if scale == 0:
        return ("-" if neg else "") + digits
    while digits.byte_length() <= scale:
        digits = "0" + digits
    var cut = digits.byte_length() - scale
    return ("-" if neg else "") + String(digits[byte=:cut]) + "." + String(digits[byte=cut:])


def _hex(d: Int) -> String:
    var digits = "0123456789abcdef"
    return String(digits[byte=d:d + 1])


def _utf8_cleaned(s: String) -> String:
    """`s`, with each byte that is not well-formed UTF-8 replaced by U+FFFD (a valid string comes back unchanged)."""
    var b = s.as_bytes()
    var n = len(b)
    var bad = False
    var i = 0
    while i < n:
        if b[i] < UInt8(0x80):
            i += 1
            continue
        var k = utf8_seq_len(b[i], b[i + 1] if i + 1 < n else UInt8(0), b[i + 2] if i + 2 < n else UInt8(0), b[i + 3] if i + 3 < n else UInt8(0), n - i)
        if k == 0:
            bad = True
            break
        i += k
    if not bad:
        return s
    var out = List[UInt8]()
    i = 0
    while i < n:
        var k = 1
        if b[i] >= UInt8(0x80):
            k = utf8_seq_len(b[i], b[i + 1] if i + 1 < n else UInt8(0), b[i + 2] if i + 2 < n else UInt8(0), b[i + 3] if i + 3 < n else UInt8(0), n - i)
        if k == 0:
            out.append(UInt8(0xEF))
            out.append(UInt8(0xBF))
            out.append(UInt8(0xBD))
            i += 1
        else:
            for j in range(k):
                out.append(b[i + j])
            i += k
    return String(unsafe_from_utf8=out)


def json_string(text: String) -> String:
    """`text` as a JSON string (UTF-8 kept, quotes, backslashes and control characters escaped; a byte that is not UTF-8, which
    the readers refuse, would still come out as U+FFFD, so the output is always valid JSON)."""
    var s = _utf8_cleaned(text)
    var b = s.as_bytes()
    var out = String('"')
    var start = 0
    for i in range(len(b)):
        var c = b[i]
        if c == UInt8(34) or c == UInt8(92) or c < UInt8(32):
            out += String(unsafe_from_utf8=b[start:i])
            if c == UInt8(34):
                out += '\\"'
            elif c == UInt8(92):
                out += "\\\\"
            elif c == UInt8(10):
                out += "\\n"
            elif c == UInt8(9):
                out += "\\t"
            elif c == UInt8(13):
                out += "\\r"
            else:
                out += "\\u00" + _hex(Int(c) // 16) + _hex(Int(c) % 16)
            start = i + 1
    out += String(unsafe_from_utf8=b[start:len(b)])
    return out + '"'


def _float_text(v: Int64, quote_special: Bool) -> String:
    var x = Float64(from_bits=UInt64(v))
    if x != x:
        return '"NaN"' if quote_special else "NaN"
    if x - x != 0.0:                       # inf (Float64.MAX is not a bound: it is inf itself)
        if x > 0.0:
            return '"Infinity"' if quote_special else "inf"
        return '"-Infinity"' if quote_special else "-inf"
    return float_text(x)


def cell_json(t: Table, c: Int, r: Int) -> String:
    if t.is_null(c, r):
        return "null"
    var k = t.kind(c).value
    var v = t.get(c, r)
    if t.is_wide(c):                                            # F-192: a total past int64
        return wide_text(wide_make(v, t.get_high(c, r)), t.scale(c))
    if k == INT.value:
        return String(v)
    if k == DECIMAL.value:
        return decimal_text(v, t.scale(c))
    if k == TEXT.value:
        return json_string(t.text_of(c, Int(v)))
    if k == DATE.value:
        return '"' + civil_date(Int(v)) + '"'
    if k == BOOL.value:
        return "true" if v != 0 else "false"
    return _float_text(v, True)


def cell_csv(t: Table, c: Int, r: Int) -> String:
    if t.is_null(c, r):
        return String()
    var k = t.kind(c).value
    var v = t.get(c, r)
    if t.is_wide(c):
        return wide_text(wide_make(v, t.get_high(c, r)), t.scale(c))
    if k == INT.value:
        return String(v)
    if k == DECIMAL.value:
        return decimal_text(v, t.scale(c))
    if k == DATE.value:
        return civil_date(Int(v))
    if k == BOOL.value:
        return "true" if v != 0 else "false"
    if k == FLOAT.value:
        return _float_text(v, False)
    return csv_quote(t.text_of(c, Int(v)))


def csv_quote(text: String) -> String:
    """A CSV cell as DuckDB writes it: quoted when it holds a comma, a quote or a line break, and the empty string as two quotes (a
    NULL is an empty cell, so the two stay apart)."""
    var s = _utf8_cleaned(text)
    if s.byte_length() == 0:
        return '""'
    var plain = True
    for x in s.as_bytes():
        if x == UInt8(44) or x == UInt8(34) or x == UInt8(10) or x == UInt8(13):
            plain = False
    if plain:
        return s
    return '"' + s.replace('"', '""') + '"'


def kind_text(k: Int) -> String:
    var names: List[String] = ["INT", "DECIMAL", "TEXT", "DATE", "BOOL", "FLOAT"]
    return names[k]


def columns_json(t: Table) -> String:
    var out = String("[")
    for c in range(t.columns()):
        out += ("" if c == 0 else ",") + '{"name":' + json_string(t.name(c)) + ',"kind":"' + kind_text(t.kind(c).value) + '","scale":' + String(t.scale(c)) + "}"
    return out + "]"


def rows_json_one(t: Table, r: Int) -> String:
    """One row as a JSON array."""
    var out = String("[")
    for c in range(t.columns()):
        out += ("," if c > 0 else "") + cell_json(t, c, r)
    return out + "]"


def rows_json(t: Table, rows: Int) -> String:
    """The first `rows` rows as a JSON array of arrays."""
    var out = String("[")
    for r in range(rows):
        out += ("," if r > 0 else "") + rows_json_one(t, r)
    return out + "]"


def row_jsonl(t: Table, r: Int) -> String:
    """One row as a JSON object keyed by column name (JSON lines)."""
    var out = String("{")
    for c in range(t.columns()):
        out += ("," if c > 0 else "") + json_string(t.name(c)) + ":" + cell_json(t, c, r)
    return out + "}"


def csv_header(t: Table) -> String:
    var out = String()
    for c in range(t.columns()):
        out += ("," if c > 0 else "") + csv_quote(t.name(c))
    return out


def csv_row(t: Table, r: Int) -> String:
    var out = String()
    for c in range(t.columns()):
        out += ("," if c > 0 else "") + cell_csv(t, c, r)
    return out
