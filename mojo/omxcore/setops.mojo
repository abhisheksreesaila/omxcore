"""UNION ALL (F-159): two result tables as one. The columns are matched by position and named by the first table; a column the two tables type alike is
copied; INT with DECIMAL is a DECIMAL, DECIMALs of two scales the larger scale, a number with a FLOAT a FLOAT (DuckDB's rule); any other pair raises. A TEXT
column keeps the first table's dictionary and adds the second's new words to it, so the codes of the first table stay and the second's are mapped once per word."""
from .table import Table, SharedTable, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, ColumnKind
from .sql import sql_error
from .bind import pow10_checked, decimal_to_float, kind_name


def _numeric(k: ColumnKind) -> Bool:
    return k == INT or k == DECIMAL or k == FLOAT


def _result_kind(ka: ColumnKind, sa: Int, kb: ColumnKind, sb: Int, c: Int, line: Int, col: Int) raises -> Tuple[Int, Int]:
    """(kind, scale) of the column both tables' column `c` become."""
    if ka == kb:
        if ka == DECIMAL:
            return (DECIMAL.value, max(sa, sb))
        if ka == FLOAT:
            return (FLOAT.value, 0)
        return (ka.value, 0)
    if _numeric(ka) and _numeric(kb):
        if ka == FLOAT or kb == FLOAT:
            return (FLOAT.value, 0)
        return (DECIMAL.value, sa if ka == DECIMAL else sb)
    raise sql_error(line, col, "the operands of the set operation cannot be combined: column " + String(c + 1) + " is " + kind_name(ka.value, sa) +
                    " in one and " + kind_name(kb.value, sb) + " in the other")


def _convert(v: Int64, kind: ColumnKind, scale: Int, to_kind: Int, to_scale: Int) raises -> Int64:
    if kind.value == to_kind and scale == to_scale:
        return v
    if to_kind == FLOAT.value:
        if kind == FLOAT:
            return v
        var f = Float64(v) if kind == INT else decimal_to_float(v, scale)
        return Int64(f.to_bits())
    var m = pow10_checked(to_scale - (scale if kind == DECIMAL else 0))
    var r = v * m
    if v != 0 and r // v != m:
        raise Error("a value does not fit the decimal column the set operation makes of it")
    return r


def union_all(a: Table, b: Table, line: Int, col: Int) raises -> Table:
    if a.columns() != b.columns():
        raise sql_error(line, col, "the operands of the set operation have " + String(a.columns()) + " and " + String(b.columns()) + " columns")
    var out = Table.empty()
    for c in range(a.columns()):
        var rk = _result_kind(a.kind(c), a.scale(c), b.kind(c), b.scale(c), c, line, col)
        var oc = out.add_column(a.name(c), ColumnKind(rk[0]), rk[1])
        if a.kind(c) == TEXT:
            for w in a.dictionary(c):
                _ = out.add_word(oc, w)
            var mapping = List[Int]()
            for w in b.dictionary(c):
                mapping.append(out.add_word(oc, w))
            for r in range(a.rows()):
                out.cols[oc].push(a.get(c, r), not a.is_null(c, r))
            for r in range(b.rows()):
                var nb = b.is_null(c, r)
                out.cols[oc].push(Int64(0) if nb else Int64(mapping[Int(b.get(c, r))]), not nb)
            continue
        for r in range(a.rows()):
            var na = a.is_null(c, r)
            out.cols[oc].push(Int64(0) if na else _convert(a.get(c, r), a.kind(c), a.scale(c), rk[0], rk[1]), not na)
        for r in range(b.rows()):
            var nb = b.is_null(c, r)
            out.cols[oc].push(Int64(0) if nb else _convert(b.get(c, r), b.kind(c), b.scale(c), rk[0], rk[1]), not nb)
    return out^
