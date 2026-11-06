"""Arrow arrays from NumPy without importing pandas.

``pa.array`` asks PyArrow's lazy pandas shim whether its input is a pandas
object, which imports pandas (~150 ms, ~300 modules) the first time it runs in a
process, even for plain NumPy input. The engine turns NumPy results into Arrow
on every query, so a user who never touches pandas paid that on their first
query. ``array`` builds the same Arrow array straight from the NumPy buffer
(zero-copy for contiguous numeric input) and defers to ``pa.array`` for
anything else.
"""

from __future__ import annotations

import sys

import numpy as np
import pyarrow as pa

from ._lazy_pc import pc

_NUMERIC = {
    np.dtype("int8"): pa.int8(), np.dtype("int16"): pa.int16(),
    np.dtype("int32"): pa.int32(), np.dtype("int64"): pa.int64(),
    np.dtype("uint8"): pa.uint8(), np.dtype("uint16"): pa.uint16(),
    np.dtype("uint32"): pa.uint32(), np.dtype("uint64"): pa.uint64(),
    np.dtype("float32"): pa.float32(), np.dtype("float64"): pa.float64(),
}


def empty(schema: pa.Schema) -> pa.Table:
    """A table of `schema` with no rows. (the schema method for this imports pandas, 110-160 ms per process: F-026.)"""
    return pa.Table.from_batches([], schema)


def _is_masked(obj) -> bool:
    ma = sys.modules.get("numpy.ma")          # touching np.ma would import it (~9 ms)
    return ma is not None and isinstance(obj, ma.MaskedArray)


def _strings(values) -> pa.Array:
    data = [v.encode("utf-8") for v in values]
    offsets = np.zeros(len(data) + 1, dtype=np.int32)
    np.cumsum([len(b) for b in data], out=offsets[1:])
    return pa.Array.from_buffers(pa.string(), len(data),
                                 [None, pa.py_buffer(offsets), pa.py_buffer(b"".join(data))])


def _same_kind(kind, type) -> bool:
    """True when a Python value of ``kind`` converts to ``type`` exactly as PyArrow would."""
    if type is None:
        return True
    if kind is bool:
        return pa.types.is_boolean(type)
    if kind is int:
        return pa.types.is_integer(type) or pa.types.is_floating(type)
    if kind is float:
        return pa.types.is_floating(type)
    if kind is str:
        return pa.types.is_string(type) or pa.types.is_large_string(type)
    return False


def _from_pylist(values, type):
    """Homogeneous list of bool / int / float / str (no None) without pandas, else None.

    Only takes the fast path when PyArrow would produce the same result; anything
    else (typed conversions like int->decimal, str->date, numpy scalars of other
    widths) returns None so the caller defers to PyArrow.
    """
    if not values:
        return None
    if all(isinstance(v, np.generic) for v in values):
        a = np.asarray(values)
        if a.dtype in _NUMERIC or a.dtype == np.bool_:
            out = array(a)
            return out if type is None or out.type == type else None
        return None
    if any(isinstance(v, np.generic) for v in values):
        return None
    kinds = {bool if isinstance(v, bool) else int if isinstance(v, int) else
             float if isinstance(v, float) else str if isinstance(v, str) else None for v in values}
    if kinds == {int, float}:
        k = float
    elif len(kinds) == 1:
        k = next(iter(kinds))
    else:
        return None
    if k is float and type is not None and pa.types.is_integer(type) \
            and all(float(v).is_integer() and abs(v) < 2 ** 53 for v in values):
        k = int                  # whole floats into an integer type: PyArrow gives the same ints
        values = [int(v) for v in values]
    if k is None or not _same_kind(k, type):
        return None
    if k is str:
        out = _strings(values)
    elif k is bool:
        out = array(np.asarray(values, dtype=np.bool_))
    elif k is int:
        try:
            out = array(np.asarray(values, dtype=np.int64))
        except OverflowError:
            return None
    else:
        out = array(np.asarray(values, dtype=np.float64))
    if type is not None and out.type != type:
        try:
            out = pc.cast(out, type, safe=True)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            return None          # let PyArrow raise its own error
    return out


def scalar(value, type=None):
    """``pa.scalar(value, type)`` without importing pandas (bool/int/float/str; else PyArrow)."""
    if isinstance(value, pa.Scalar):
        return pc.cast(value, type) if type is not None and value.type != type else value
    import datetime as _dt
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return array(np.array([value]))[0]                    # a timestamp of the value's own unit (ns too)
    if type is None and isinstance(value, _dt.date) and not isinstance(value, _dt.datetime):
        days = (value - _dt.date(1970, 1, 1)).days
        return pc.cast(array(np.array([days], dtype=np.int32)), pa.date32())[0]
    if type is None and isinstance(value, _dt.datetime) and value.tzinfo is None:
        us = (value - _dt.datetime(1970, 1, 1)) // _dt.timedelta(microseconds=1)
        return pc.cast(array(np.array([us], dtype=np.int64)), pa.timestamp("us"))[0]
    arr = _from_pylist([value], type) if value is not None else None
    if arr is None:
        return pa.scalar(value, type=type)
    return arr[0]


def combine(arr) -> pa.Array:
    """``ChunkedArray.combine_chunks()`` without pandas: PyArrow builds the zero-chunk case with
    ``pa.array([])``, which imports pandas, and an empty intermediate (a filter that keeps no
    rows) is common on small data. One chunk is returned as is (unlike ``combine_chunks``, which
    copies it, the result can keep a non-zero offset); an Array passes through."""
    if not isinstance(arr, pa.ChunkedArray):
        return arr
    if arr.num_chunks == 0:
        return pa.nulls(0, arr.type)
    if arr.num_chunks == 1:
        return arr.chunk(0)
    return arr.combine_chunks()


def value_set(values, like) -> pa.Array:
    """``value_set`` for ``pc.is_in`` against a column of type ``like``. An empty list has no type
    to infer, and ``pa.array([])`` imports pandas: build it empty with the column's value type."""
    if len(values) == 0:
        return pa.nulls(0, like.value_type if pa.types.is_dictionary(like) else like)
    return array(values)


def is_in(col, values):
    """SQL ``col IN (values)`` as a boolean array: NULL where ``col`` is NULL (``pc.is_in`` answers False there,
    which made ``NOT IN`` keep the NULL rows)."""
    values = list(values)
    has_null = any(v is None for v in values)          # x IN (1, NULL) is true or NULL, never false
    values = [v for v in values if v is not None]
    if pa.types.is_floating(col.type):
        # DuckDB: -0.0 = 0.0 and NaN = NaN; Arrow's hash lookup tells -0.0 from 0.0 (F-070). Adding 0.0 turns
        # -0.0 into 0.0, and NaN is looked up with a mask of its own.
        has_nan = any(v != v for v in values)
        zero = scalar(0.0, type=col.type)
        r = pc.is_in(pc.add(col, zero), value_set=value_set([v + 0.0 for v in values if v == v], col.type))
        if has_nan:
            r = pc.or_kleene(r, pc.is_nan(col))
    else:
        vs = value_set(values, col.type)
        if pa.types.is_timestamp(col.type) and pa.types.is_timestamp(vs.type) and vs.type != col.type:
            units = ("s", "ms", "us", "ns")         # literals are TIMESTAMP (us): compare in the finer unit, in the column's zone
            fine = pa.timestamp(max(col.type.unit, vs.type.unit, key=units.index), col.type.tz)
            col, vs = pc.cast(col, fine), pc.cast(vs, fine)
        r = pc.is_in(col, value_set=vs)
    if has_null:
        r = pc.if_else(r, scalar(True, type=pa.bool_()), pa.nulls(1, pa.bool_())[0])
    if getattr(col, "null_count", 0):
        r = pc.if_else(pc.is_valid(col), r, pa.nulls(1, pa.bool_())[0])
    return r


def _masked(obj, type):
    """1-D masked numeric/bool array: values plus a validity bitmap, None if not supported."""
    data = np.ma.getdata(obj)
    if data.ndim != 1 or not (data.dtype in _NUMERIC or data.dtype == np.bool_):
        return None
    mask = np.ma.getmaskarray(obj)
    base = array(np.ascontiguousarray(data))
    if not mask.any():
        out = base
    else:
        valid = np.packbits(~mask, bitorder="little")
        out = pa.Array.from_buffers(base.type, len(base), [pa.py_buffer(valid), base.buffers()[1]],
                                    null_count=int(mask.sum()))
    return pc.cast(out, type, safe=True) if type is not None and out.type != type else out


def array(obj, type=None, **kwargs) -> pa.Array:
    """``pa.array(obj, type)`` for 1-D numeric/bool NumPy input, without pandas."""
    if not kwargs and isinstance(obj, (list, tuple)):
        if type is not None and all(v is None for v in obj):
            return pa.nulls(len(obj), type)          # empty, or all NULL (an aggregate over no rows)
        out = _from_pylist(list(obj), type)
        if out is not None:
            return out
    if not kwargs and _is_masked(obj):
        out = _masked(obj, type)
        if out is not None:
            return out
    if kwargs or not isinstance(obj, np.ndarray) or obj.ndim != 1 or _is_masked(obj):
        return pa.array(obj, type=type, **kwargs)
    a = obj
    if a.dtype == np.bool_:
        n = len(a)
        bits = np.packbits(a, bitorder="little") if n else np.zeros(0, np.uint8)
        out = pa.Array.from_buffers(pa.bool_(), n, [None, pa.py_buffer(bits)])
    else:
        at = _NUMERIC.get(a.dtype.newbyteorder("=") if a.dtype.byteorder not in "=|" else a.dtype)
        if at is None or a.dtype.byteorder == ">":
            return pa.array(obj, type=type)
        if not a.flags["C_CONTIGUOUS"]:
            a = np.ascontiguousarray(a)
        out = pa.Array.from_buffers(at, len(a), [None, pa.py_buffer(a)])
    if type is not None and out.type != type:
        out = pc.cast(out, type, safe=True)
    return out


_NP_OF = {pa.int8(): np.int8, pa.int16(): np.int16, pa.int32(): np.int32, pa.int64(): np.int64,
          pa.uint8(): np.uint8, pa.uint16(): np.uint16, pa.uint32(): np.uint32, pa.uint64(): np.uint64,
          pa.float32(): np.float32, pa.float64(): np.float64,
          pa.date32(): np.int32}


def to_numpy(obj, zero_copy_only: bool = False, **kwargs):
    """``obj.to_numpy(...)`` for Arrow arrays without importing pandas.

    Null-free fixed-width numeric (and bool) arrays become NumPy views of the Arrow
    buffer, exactly what PyArrow returns for them. Everything else - chunked arrays
    with several chunks, nulls, strings, dictionaries, non-Arrow objects such as
    MAX tensors - goes through the object's own ``to_numpy``.
    """
    if isinstance(obj, pa.ChunkedArray):
        if obj.num_chunks == 1:
            return to_numpy(obj.chunk(0), zero_copy_only=zero_copy_only, **kwargs)
        if obj.num_chunks == 0 and obj.type != pa.date32():
            dt = _NP_OF.get(obj.type)
            if dt is not None and not kwargs:
                return np.empty(0, dtype=dt)
        if not kwargs and obj.num_chunks > 1 and _NP_OF.get(obj.type) is not None \
                and obj.type != pa.date32() and obj.null_count == 0:
            return np.concatenate([to_numpy(c) for c in obj.chunks])
    if not isinstance(obj, pa.Array) or kwargs or obj.null_count:
        return obj.to_numpy(zero_copy_only=zero_copy_only, **kwargs) if isinstance(obj, (pa.Array, pa.ChunkedArray)) \
            else obj.to_numpy(**kwargs)
    t = obj.type
    if t == pa.bool_():
        if zero_copy_only:
            return obj.to_numpy(zero_copy_only=True)       # raises, as PyArrow does
        n = len(obj)
        if n == 0:
            return np.zeros(0, dtype=np.bool_)
        bits = np.frombuffer(obj.buffers()[1], dtype=np.uint8)
        return np.unpackbits(bits, bitorder="little", count=obj.offset + n)[obj.offset:].astype(np.bool_)
    dt = _NP_OF.get(t)
    if dt is None or t == pa.date32():
        return obj.to_numpy(zero_copy_only=zero_copy_only)
    buf = obj.buffers()[1]
    if buf is None:
        return np.empty(0, dtype=dt)
    out = np.frombuffer(buf, dtype=dt, count=len(obj), offset=obj.offset * np.dtype(dt).itemsize)
    out.flags.writeable = False          # a view of immutable Arrow memory, as PyArrow returns
    return out


# --- gathering rows -------------------------------------------------------------------------

TAKE_CHUNK_ROWS = 1 << 22        # rows per slice for columns whose values cannot be measured
TAKE_CHUNK_BYTES = 1 << 30       # value bytes per slice of a string / binary column
TAKE_SLICE_MIN_ROWS = 1 << 17    # gathers of fewer rows are never sliced


def _has_32bit_offsets(t) -> bool:
    if pa.types.is_struct(t):
        return any(_has_32bit_offsets(f.type) for f in t)
    return pa.types.is_string(t) or pa.types.is_binary(t) or pa.types.is_list(t)


def _byte_slices(col, idx):
    """Slice bounds over ``idx`` such that each slice gathers at most TAKE_CHUNK_BYTES of
    string / binary values (one row alone may exceed it), or None for other types."""
    if not (pa.types.is_string(col.type) or pa.types.is_binary(col.type)):
        return None
    lens = pc.binary_length(col)
    lens = to_numpy(pc.fill_null(lens, scalar(0, type=lens.type))).astype(np.int64, copy=False)   # not a Python 0: that imports pandas
    cum = np.cumsum(lens[idx])
    n, bounds, start, base = len(idx), [], 0, 0
    while start < n:
        end = int(np.searchsorted(cum, base + TAKE_CHUNK_BYTES, side="right"))
        end = min(max(end, start + 1), n)
        bounds.append((start, end))
        base, start = int(cum[end - 1]), end
    return bounds


def take_rows(obj, idx):
    """``obj.take(idx)`` for a table, chunked or plain array and a NumPy row-index array.

    A column with 32-bit offsets (string, binary, list) gathered into more than 2 GiB of
    values comes back as a corrupt array (``Negative offsets in binary array``, visible only
    to ``validate(full=True)``): joining 60M TPC-H lineitem rows to their orders gathers 3 GB
    of ``o_comment``.  String and binary columns are gathered in slices of at most
    ``TAKE_CHUNK_BYTES`` of values (measured from the value lengths); lists and structs
    holding strings in slices of rows sized from the average value size, each checked with
    ``validate`` (an ``ArrowInvalid`` is raised rather than a corrupt column returned).
    The result is a chunked array.  Every other column is gathered in one call."""
    idx = np.ascontiguousarray(idx, dtype=np.int64)
    n = len(idx)

    def one(col):
        if n <= TAKE_SLICE_MIN_ROWS or not _has_32bit_offsets(col.type):
            return pc.take(col, array(idx))
        bounds = _byte_slices(col, idx)
        checked = bounds is None
        if bounds is None:
            per_row = max(1.0, col.nbytes / max(1, len(col)))          # average value bytes per source row
            step = int(max(1, min(TAKE_CHUNK_ROWS, TAKE_CHUNK_BYTES // per_row)))
            bounds = [(i, min(i + step, n)) for i in range(0, n, step)]
        parts = [pc.take(col, array(idx[a:b])) for a, b in bounds]
        chunks = [c for p in parts for c in (p.chunks if isinstance(p, pa.ChunkedArray) else [p])]
        if checked:
            for c in chunks:
                c.validate(full=True)
        return pa.chunked_array(chunks, type=col.type)

    if isinstance(obj, pa.Table):
        if n <= TAKE_SLICE_MIN_ROWS or not any(_has_32bit_offsets(t) for t in obj.schema.types):
            return pc.take(obj, array(idx))
        return pa.Table.from_arrays([one(obj.column(i)) for i in range(obj.num_columns)], schema=obj.schema)
    return one(obj)


# ---- pandas-free replacements for Table.group_by -----------------------------------------------
# PyArrow's Table.group_by / Table.join import pandas (through pyarrow.dataset) on every call, even
# on non-empty data. These cover the engine's fallback uses: DISTINCT and grouped aggregates over
# columns with nulls. Groups come out in first-appearance order, like Table.group_by.

def _codes(col) -> np.ndarray:
    """Dense int64 codes per row, nulls encoded as their own value."""
    a = combine(col)
    if pa.types.is_dictionary(a.type):
        a = a.dictionary_decode()
    a = canonical_floats(a)
    return to_numpy(pc.dictionary_encode(a, null_encoding="encode").indices).astype(np.int64)


def canonical_floats(a):
    """A float array grouped like SQL: -0.0 equals 0.0 and every NaN (any sign or payload) equals
    every other NaN; NULLs stay NULL. Arrow's dictionary encoding compares floats bit for bit and
    gave each of those its own group (F-077). Used for the codes only: a group still reports the
    key value of its first row, which is what DuckDB does too."""
    if a.type not in (pa.float32(), pa.float64()) or len(a) == 0:
        return a
    dt = np.dtype(np.float32 if a.type == pa.float32() else np.float64)
    x = np.frombuffer(a.buffers()[1], dtype=dt, count=a.offset + len(a))
    with np.errstate(invalid="ignore"):                     # a signalling NaN raises "invalid" here
        out = x + dt.type(0)                                # -0.0 + 0.0 is 0.0 (a new array)
    nan = np.isnan(x)
    if nan.any():
        out[nan] = dt.type("nan")
    return pa.Array.from_buffers(a.type, len(a), [a.buffers()[0], pa.py_buffer(out)],
                                 null_count=a.null_count, offset=a.offset)


def _group_ids(table: pa.Table, cols):
    """(group id per row, first row of each group), groups numbered by first appearance."""
    codes = np.stack([_codes(table.column(c)) for c in cols], axis=1)
    _, first, inv = np.unique(codes, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[inv.reshape(-1)], first[order]


def _float_values(a) -> np.ndarray:
    """float64 values of a numeric Array, null slots unspecified (callers mask them). Reads the
    data buffer: ``to_numpy`` on an array with nulls goes through pandas."""
    if a.type != pa.float64():
        a = pc.cast(a, pa.float64())
    if len(a) == 0:
        return np.zeros(0, np.float64)
    return np.frombuffer(a.buffers()[1], dtype=np.float64, count=a.offset + len(a))[a.offset:]


_INT64_MIN, _INT64_MAX = -(1 << 63), (1 << 63) - 1


def _sum_overflow():
    return OverflowError("Overflow in sum of INT64: the total does not fit in a 64-bit integer "
                         "(DuckDB widens it to a 128-bit HUGEINT)")


def _int_group_limbs(a, valid, g, n_groups):
    """Exact per-group sums of an integer Array as three 21-bit limb sums (total = s2 * 2^42 + s1 * 2^21 + s0):
    np.bincount adds in float64, which loses digits past 2^53, but each limb's sum is exact there."""
    if a.type != pa.int64():
        a = pc.cast(a, pa.int64())
    v = np.frombuffer(a.buffers()[1], dtype=np.int64, count=a.offset + len(a))[a.offset:][valid]
    mask = (1 << 21) - 1
    s0 = np.bincount(g, weights=(v & mask), minlength=n_groups)
    s1 = np.bincount(g, weights=((v >> 21) & mask), minlength=n_groups)
    s2 = np.bincount(g, weights=(v >> 42), minlength=n_groups)
    return s0, s1, s2


def _limb_total(s0, s1, s2, i) -> int:
    return (int(s2[i]) << 42) + (int(s1[i]) << 21) + int(s0[i])


def _int_group_sum(a, valid, g, n_groups) -> np.ndarray:
    """Exact int64 per-group sum: the limb sums are recombined in int64. A group whose total does not
    fit int64 raises OverflowError instead of wrapping."""
    s0, s1, s2 = _int_group_limbs(a, valid, g, n_groups)
    # float64 estimate of the total: only groups near or past the int64 limits are rechecked exactly
    for i in np.flatnonzero(np.abs(s2 * 2.0 ** 42 + s1 * 2.0 ** 21 + s0) > 2.0 ** 62):
        if not _INT64_MIN <= _limb_total(s0, s1, s2, i) <= _INT64_MAX:
            raise _sum_overflow()
    return (s2.astype(np.int64) << 42) + (s1.astype(np.int64) << 21) + s0.astype(np.int64)


# DuckDB divides its HUGEINT total in a C++ `long double`: the x87 80-bit type on x86-64, a plain double on arm64 (Apple
# Silicon) and Windows (F-211: probed on an Apple M4, DuckDB 1.5.6, 106 of the 108 cases where the formulas differ were
# `double(total) / n` with the HUGEINT cast of the total, none the correctly rounded quotient). Where numpy's longdouble is only a
# double the x87 imitation cannot be built (upper * 2^64 + lower cancels), and DuckDB itself divides in doubles there.
# Tests flip this flag to run each path on either machine.
_X87 = np.finfo(np.longdouble).nmant >= 63


def _hug(total: int) -> float:
    """DuckDB's HUGEINT to DOUBLE cast, the same on both platforms (mojo/omxcore/wide.mojo `hug`): an upper word of -1 is
    `-double(UINT64_MAX - lower) - 1.0`, any other `double(upper) * 2^64 + double(lower)`."""
    upper, lower = total >> 64, total & ((1 << 64) - 1)
    if upper == -1:
        return -float((1 << 64) - 1 - lower) - 1.0
    return float(upper) * 2.0 ** 64 + float(lower)


def _ld_div(total: int, n: int) -> float:
    """DuckDB's ``avg`` of an integer, on this platform. x86-64: the HUGEINT total, taken as a 64-bit-mantissa ``long double`` (upper * 2^64 +
    lower, one rounding), divided by the count in ``long double`` and rounded to a double (F-126: probed, it is not the correctly rounded
    total / n: 19 of 19 differing cases agree with this and with no other formula). Without an x87 (arm64, Windows): the cast of the total
    to a double (`_hug`) divided by the count in doubles (F-211)."""
    if not _X87:
        return _hug(total) / n
    L = np.longdouble
    return float((L(total >> 64) * L(2.0 ** 64) + L(total & ((1 << 64) - 1))) / L(n))


def _int_group_mean(a, valid, g, n_groups, cnt) -> np.ndarray:
    """Per-group mean of an integer Array as DuckDB computes it (F-126): the total is exact in a wider integer
    (it may pass int64: a mean never raises), then divided as `_ld_div` does."""
    s0, s1, s2 = _int_group_limbs(a, valid, g, n_groups)
    est = np.abs(s2 * 2.0 ** 42 + s1 * 2.0 ** 21 + s0)
    tot = (s2.astype(np.int64) << 42) + (s1.astype(np.int64) << 21) + s0.astype(np.int64)     # wraps only where est is huge
    res = int_mean_of_totals(tot, cnt)
    for i in np.flatnonzero((est > 2.0 ** 62) & (cnt > 0)):          # a total that may not fit int64
        res[i] = _ld_div(_limb_total(s0, s1, s2, i), int(cnt[i]))
    return res


def int_mean_of_totals(tot: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """Means from exact int64 totals and counts, divided as DuckDB does (`_ld_div`). A count of 0 gives 0.0
    (callers mask it)."""
    safe = np.maximum(cnt, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        if _X87:
            res = (tot.astype(np.longdouble) / safe.astype(np.longdouble)).astype(np.float64)
        else:                                          # DuckDB's HUGEINT cast of the total (`_hug`), divided in doubles (F-211)
            res = np.where(tot < 0, -(~tot).astype(np.float64) - 1.0, tot.astype(np.float64)) / safe
    return np.where(cnt > 0, res, 0.0)


def int_mean_of_halves(lo: np.ndarray, hi: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """`int_mean_of_totals` for totals kept as ``hi * 2^32 + lo`` (a total that may pass int64: streamed batches)."""
    big = (hi > (1 << 29)) | (hi < -(1 << 29)) | (lo > (1 << 61)) | (lo < -(1 << 61))   # else the int64 total is exact
    res = int_mean_of_totals(np.where(big, 0, hi << 32) + np.where(big, 0, lo), cnt)
    for i in np.flatnonzero(big & (cnt > 0)):
        res[i] = _ld_div((int(hi[i]) << 32) + int(lo[i]), int(cnt[i]))
    return res


def int_mean_column(tot, cnt) -> pa.Array:
    """A float64 mean Array from an int64 total column and a count column (a NULL total, an empty global
    aggregate, stays NULL)."""
    tot, cnt = combine(tot), combine(cnt)
    if tot.null_count:
        return pa.nulls(len(tot), pa.float64())
    return array(int_mean_of_totals(to_numpy(tot), to_numpy(cnt)))


def int_total(a):
    """The exact total of an integer Array as a Python int, whatever its size (None when no value is valid)."""
    n = len(a) - a.null_count
    if n == 0:
        return None
    valid = to_numpy(pc.is_valid(a)).astype(bool) if a.null_count else np.ones(len(a), bool)
    s0, s1, s2 = _int_group_limbs(a, valid, np.zeros(int(valid.sum()), np.intp), 1)
    return _limb_total(s0, s1, s2, 0)


def int_mean(a):
    """DuckDB's ``avg`` of an integer Array as a Python float (None when no value is valid); never an overflow."""
    tot = int_total(a)
    return None if tot is None else _ld_div(tot, len(a) - a.null_count)


def int_sum(a):
    """``pc.sum`` of an integer Array as a Python int (None when no value is valid); OverflowError
    when the total does not fit int64 (``pc.sum`` wraps). Only a column whose rows * largest
    magnitude could pass 2^63 pays for the exact limb sum."""
    n = len(a) - a.null_count
    if n == 0:
        return None
    mm = pc.min_max(a)
    if max(abs(mm["min"].as_py()), abs(mm["max"].as_py())) * n <= _INT64_MAX:
        return pc.sum(a).as_py()
    valid = to_numpy(pc.is_valid(a)).astype(bool) if a.null_count else np.ones(len(a), bool)
    return int(_int_group_sum(a, valid, np.zeros(int(valid.sum()), np.intp), 1)[0])


_ARITH = {"add": ("addition", "add_checked"), "sub": ("subtraction", "subtract_checked"),
          "mul": ("multiplication", "multiply_checked")}


def _int_beside_double(lhs, rhs):
    """An integer operand next to a DOUBLE one converts to double the way DuckDB's cast does (rounding to the nearest
    double). Arrow's own promotion refuses an int64 past 2^53 ("Integer value ... not in range", F-060b)."""
    lt, rt = getattr(lhs, "type", None), getattr(rhs, "type", None)
    if lt is not None and rt is not None:
        if pa.types.is_integer(lt) and rt == pa.float64():
            lhs = pc.cast(lhs, pa.float64(), safe=False)
        elif pa.types.is_integer(rt) and lt == pa.float64():
            rhs = pc.cast(rhs, pa.float64(), safe=False)
    return lhs, rhs


def checked_arith(op, lhs, rhs):
    """``add`` / ``sub`` / ``mul`` of two operands, raising OverflowError ("Overflow in
    multiplication of INT64", DuckDB's message) where Arrow's plain kernels would wrap. Float
    operands behave as with the plain kernels."""
    word, name = _ARITH[op]
    lhs, rhs = _int_beside_double(lhs, rhs)
    try:
        return getattr(pc, name)(lhs, rhs)
    except pa.ArrowInvalid as e:
        if "overflow" not in str(e).lower():
            raise
        t = lhs.type if hasattr(lhs, "type") else rhs.type
        raise OverflowError(f"Overflow in {word} of {str(t).upper()}") from e


def modulo(lhs, rhs, decimal=False):
    """SQL ``lhs % rhs`` as DuckDB computes it: C ``fmod``, so the sign follows the dividend (-7 % 3 = -1, 7 % -3
    = 1, not NumPy's ``%``), NULL when either side is NULL or an integer divisor is 0 (a float one gives NaN), NaN
    stays NaN, ``x % inf`` is x. Operands are arrays or scalars, and the arrays decide the arithmetic: integers
    stay integers (the operands' common type; int64 when signedness differs), exact for uint64 too (a mix of
    uint64 and signed values is computed exactly and must fit int64 or uint64, or it raises); anything with a
    float is float64 (float32 when both are float32). ``INT64_MIN % -1`` raises like DuckDB's division."""
    def parts(x):
        if isinstance(x, pa.Scalar):
            v = x.as_py() if x.is_valid else 0
            return np.array([v]), np.array([x.is_valid]), x.type, True
        a = combine(x)
        valid = to_numpy(pc.is_valid(a)).astype(bool) if a.null_count else np.ones(len(a), bool)
        if pa.types.is_floating(a.type):
            return _float_values(a), valid, a.type, False
        return to_numpy(pc.fill_null(a, scalar(0, type=a.type))), valid, a.type, False
    (a, av, at, a_sc), (b, bv, bt, b_sc) = parts(lhs), parts(rhs)
    valid = av & bv
    n = max(len(a) if not a_sc else 0, len(b) if not b_sc else 0) if not (a_sc and b_sc) else 1       # an empty array and a scalar: no rows
    lit_a = decimal and a_sc and a.dtype.kind == "f" and b.dtype.kind in "iu" and bool(np.isfinite(a[0]))
    lit_b = decimal and b_sc and b.dtype.kind == "f" and a.dtype.kind in "iu" and bool(np.isfinite(b[0]))
    if lit_a or lit_b:       # an integer array against a decimal literal k/s (2.5 = 25/10): DuckDB is exact
        from decimal import Decimal
        d = Decimal(repr(float((a if lit_a else b)[0])))
        s_ = 10 ** max(-d.as_tuple().exponent, 0)
        k = int(d * s_)
        if abs(k) * s_ > np.iinfo(np.int64).max:
            raise OverflowError(f"{d} has too many digits for an exact modulo")
        x = b if lit_a else a
        x = np.broadcast_to(x, (n,))
        if lit_b:                                      # x % k/s = ((x % k) * s % k) / s, exact while k*s fits
            valid = valid & (k != 0)
            if x.dtype == np.uint64:
                kk = np.uint64(abs(k) or 1)
                r = np.fmod(np.fmod(x, kk) * np.uint64(s_), kk)
            else:
                kk = abs(k) or 1
                r = np.fmod(np.fmod(x.astype(np.int64), kk) * s_, kk)
        else:                                          # k/s % x = (k % (x * s)) / s
            c = np.iinfo(np.int64).max // s_       # |x| * s overflows past c, and then |x| * s > |k|: the result is k
            big = (x > c) | (x < -c) if x.dtype != np.uint64 else x > np.uint64(c)
            valid = valid & (x != 0)
            r = np.where(big, k, np.fmod(np.int64(k), np.where(valid & ~big, np.where(big, 0, x).astype(np.int64) * s_, 1)))
        out_t, res = pa.float64(), r / s_
    elif a.dtype.kind == "f" or b.dtype.kind == "f":
        out_t = pa.float32() if pa.types.is_float32(at) and pa.types.is_float32(bt) else pa.float64()
        with np.errstate(all="ignore"):
            res = np.fmod(a.astype(np.float64), b.astype(np.float64))
    else:
        valid = valid & (b != 0)
        arrays = [t for t, sc in ((at, a_sc), (bt, b_sc)) if not sc]
        signed = [pa.types.is_signed_integer(t) for t in arrays]
        if a.dtype == np.uint64 or b.dtype == np.uint64:           # exact in uint64 unless a signed value is in play
            a_u = a.dtype == np.uint64 or (a_sc and a[0] >= 0)
            b_u = b.dtype == np.uint64 or (b_sc and b[0] >= 0)
            if a_u and b_u and not any(signed):
                out_t, res = pa.uint64(), np.fmod(a.astype(np.uint64), np.where(valid, b, 1).astype(np.uint64))
            else:
                ints = [(0 if not v else int(x)) for x, v in zip(np.broadcast_to(a.astype(object), (n,)), valid)]
                den = [(1 if not v else int(y)) for y, v in zip(np.broadcast_to(b.astype(object), (n,)), valid)]
                rem = [(abs(x) % abs(y)) * (-1 if x < 0 else 1) for x, y in zip(ints, den)]
                lo, hi = (min(rem, default=0), max(rem, default=0))
                if lo >= -(2 ** 63) and hi < 2 ** 63:
                    out_t, res = pa.int64(), np.array(rem, np.int64)
                elif lo >= 0 and hi < 2 ** 64:
                    out_t, res = pa.uint64(), np.array(rem, np.uint64)
                else:
                    raise OverflowError("Overflow in modulo of UINT64 and a signed integer")
        else:
            a, b = a.astype(np.int64), np.where(valid, b, 1).astype(np.int64)
            same_sign = bool(arrays) and (all(signed) or not any(signed))
            out_t = max(arrays, key=lambda t: t.bit_width) if same_sign else pa.int64()
            # the minimum of the result's type % -1 overflows (DuckDB: INT32_MIN % -1 raises, INT32_MIN % BIGINT -1 is 0) (F-126)
            low = np.iinfo(f"int{out_t.bit_width}").min if pa.types.is_signed_integer(out_t) else None
            if low is not None and np.any(valid & (a == low) & (b == -1)):
                raise OverflowError(f"Overflow in division of {low} / -1")
            res = np.fmod(a, b)
    return array(np.ma.array(np.broadcast_to(res, (n,)).copy(), mask=~np.broadcast_to(valid, (n,))), type=out_t)


def _nonzero(x):
    """*x* with its zeros replaced by NULL (a divisor: DuckDB gives NULL for a zero one, integer or float)."""
    if isinstance(x, pa.Scalar):
        return pa.scalar(None, x.type) if x.is_valid and x.as_py() == 0 else x
    return pc.if_else(pc.equal(x, scalar(0, type=x.type)), pa.scalar(None, x.type), x)


def int_divide(lhs, rhs):
    """SQL ``lhs // rhs`` as DuckDB computes it: integers divide truncating toward zero (-7 // 2 = -3) in the
    operands' common type (an integer literal takes the column's type when it fits: int32 // 2 is int32), any
    float or decimal operand makes it a plain division in double, and a zero divisor gives NULL (float too).
    NULL in, NULL out. ``INT64_MIN // -1`` raises like DuckDB's."""
    if isinstance(lhs, pa.Scalar) != isinstance(rhs, pa.Scalar) and all(
            pa.types.is_integer(x.type) for x in (lhs, rhs)):      # a literal next to a column: the column's type
        lit_, col_ = (lhs, rhs) if isinstance(lhs, pa.Scalar) else (rhs, lhs)
        try:
            lit_ = lit_.cast(col_.type) if lit_.is_valid else pa.scalar(None, col_.type)
            lhs, rhs = (lit_, col_) if isinstance(lhs, pa.Scalar) else (col_, lit_)
        except pa.ArrowInvalid:
            pass
    rhs = _nonzero(rhs)
    try:
        return pc.divide_checked(lhs, rhs)
    except pa.ArrowInvalid as e:
        if "overflow" not in str(e).lower():
            raise
        raise OverflowError("Overflow in division of INT64") from e


def power(base, exponent):
    """SQL ``power(x, y)`` / ``x ** y``: a DOUBLE whatever the arguments (a negative base to a fraction is NaN,
    ``0 ** -1`` is inf, an overflow is inf); NULL in, NULL out."""
    return pc.power(pc.cast(base, pa.float64(), safe=False), pc.cast(exponent, pa.float64(), safe=False))


def _domain_checked(x, bad, message):
    """``x`` as a DOUBLE, after raising DuckDB's "Out of Range Error" for the first value the function rejects."""
    x = pc.cast(x, pa.float64(), safe=False)
    rejected = pc.filter(x, bad(x))
    if len(rejected):
        raise ValueError(message(rejected[0].as_py()))
    return x


def sqrt(x):
    """SQL ``sqrt(x)``: a DOUBLE; a negative argument raises like DuckDB's, NULL gives NULL, NaN stays NaN."""
    return pc.sqrt(_domain_checked(x, lambda v: pc.less(v, scalar(0.0)),
                                   lambda _: "cannot take square root of a negative number"))


def ln(x):
    """SQL ``ln(x)``: a DOUBLE; zero and negative arguments raise like DuckDB's, NULL gives NULL, NaN stays NaN."""
    return pc.ln(_domain_checked(x, lambda v: pc.less_equal(v, scalar(0.0)),
                                 lambda v: "cannot take logarithm of zero" if v == 0 else "cannot take logarithm of a negative number"))


def nullify(x):
    """NULL of *x*'s type on every row (``x + NULL``: arithmetic with a NULL keeps the other operand's type)."""
    return pa.nulls(len(x), x.type)


SQL_MATH = {"idiv": int_divide, "pow": power, "sqrt": sqrt, "ln": ln, "nullify": nullify}   # (op name -> function of its evaluated arguments)


def distinct_rows(table: pa.Table) -> pa.Table:
    """Distinct rows of ``table`` (nulls equal to each other), first-appearance order."""
    if table.num_rows == 0:
        return table
    _, first = _group_ids(table, table.column_names)
    return pc.take(table, array(first.astype(np.int64)))


def float_extreme(a, op: str):
    """``min`` / ``max`` of an Arrow array as a Python number (None: no valid value), ordering NaN above every
    number like DuckDB (F-070): min skips NaN (NaN if that is all there is), max is NaN if there is one."""
    v = pc.min_max(a)[op].as_py()
    if pa.types.is_floating(a.type):
        n_nan = pc.sum(pc.is_nan(a)).as_py() or 0
        if n_nan and (op == "max" or n_nan == len(a) - a.null_count):
            return float("nan")
    return v


def group_aggregate(table: pa.Table, keys, spec) -> pa.Table:
    """``table.group_by(keys).aggregate(spec)`` for spec ops sum / min / max / mean / count (nulls
    skipped; a group with no valid value gives NULL, count 0). Columns: keys, then ``<col>_<op>``."""
    names = list(keys) + [f"{c}_{op}" for c, op in spec]
    if table.num_rows == 0:
        cols = [pa.nulls(0, table.schema.field(k).type) for k in keys]
        for c, op in spec:
            t = table.schema.field(c).type
            cols.append(pa.nulls(0, pa.int64() if op == "count" else
                                 t if op in ("min", "max") else
                                 pa.int64() if op == "sum" and pa.types.is_integer(t) else pa.float64()))
        return pa.Table.from_arrays(cols, names=names)
    gid, first = _group_ids(table, keys)
    n_groups = len(first)
    take = array(first.astype(np.int64))
    out = [pc.take(combine(table.column(k)), take) for k in keys]
    for c, op in spec:
        a = combine(table.column(c))
        valid = to_numpy(pc.is_valid(a)).astype(bool) if a.null_count else np.ones(len(a), bool)
        g = gid[valid]
        cnt = np.bincount(g, minlength=n_groups)
        if op == "count":
            out.append(array(cnt.astype(np.int64)))
            continue
        if op == "sum" and pa.types.is_integer(a.type):
            out.append(array(np.ma.array(_int_group_sum(a, valid, g, n_groups), mask=cnt == 0), type=pa.int64()))
            continue
        if op == "mean" and pa.types.is_integer(a.type):
            # F-060 / F-126: the integers are added exactly (wider than int64: a mean never raises), then divided once
            res = _int_group_mean(a, valid, g, n_groups, cnt)
            out.append(array(np.ma.array(res, mask=cnt == 0), type=pa.float64()))
            continue
        if op in ("min", "max") and (pa.types.is_integer(a.type) or pa.types.is_date(a.type) or pa.types.is_timestamp(a.type)):
            # exact: float64 rounds past 2^53 (F-093: a date / timestamp is its integer storage)
            own = a.type
            if not pa.types.is_integer(own):
                a = pc.cast(a, pa.int32() if pa.types.is_date32(own) else pa.int64())
            dt = np.dtype(f"{'u' if pa.types.is_unsigned_integer(a.type) else 'i'}{a.type.bit_width // 8}")
            xi = np.frombuffer(a.buffers()[1], dtype=dt, count=a.offset + len(a))[a.offset:][valid] if len(a) else np.zeros(0, dt)
            res = np.zeros(n_groups, dtype=dt)
            init = np.iinfo(dt).max if op == "min" else np.iinfo(dt).min
            res[:] = init
            (np.minimum if op == "min" else np.maximum).at(res, g, xi)
            res = array(np.ma.array(res, mask=cnt == 0), type=a.type)
            out.append(res if own == a.type else pc.cast(res, own))
            continue
        x = _float_values(a)[valid]
        if op in ("sum", "mean"):
            res = np.bincount(g, weights=x, minlength=n_groups)
            if op == "mean":
                with np.errstate(invalid="ignore", divide="ignore"):
                    res = res / cnt
        else:
            # DuckDB orders NaN above every number (F-070): min skips NaN (a group of only NaN gives NaN), max is
            # NaN when the group holds one (Table.group_by skips NaN in both and gives +-inf for only-NaN)
            num = ~np.isnan(x)
            res = np.full(n_groups, np.inf if op == "min" else -np.inf)
            (np.minimum if op == "min" else np.maximum).at(res, g[num], x[num])
            res[np.bincount(g[num], minlength=n_groups) == 0] = np.nan
            if op == "max":
                res[np.bincount(g[~num], minlength=n_groups) > 0] = np.nan
        t = a.type
        typ = (t if op in ("min", "max") else
               pa.int64() if op == "sum" and pa.types.is_integer(t) else pa.float64())
        out.append(array(np.ma.array(res, mask=cnt == 0), type=typ))
    return pa.Table.from_arrays(out, names=names)
