"""F-081: window partitions over a float column group like DuckDB (-0.0 = 0.0, every NaN one
partition, NULL its own partition, +-inf their own). Expected values come from DuckDB."""
import struct

import duckdb
from conftest import oracle_connection
import pyarrow as pa
import pytest

from omxcore import LazyFrame, col, row_number


def _nan(payload):
    return struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000000 | payload))[0]


def _table(keys, key_type=pa.float64(), k2=None, v=None):
    n = len(keys)
    d = {"k": pa.array(keys, type=key_type), "v": pa.array(v if v is not None else [float(i + 1) for i in range(n)], pa.float64()),
         "id": pa.array(list(range(n)), pa.int64())}
    if k2 is not None:
        d["k2"] = pa.array(k2)
    return pa.table(d)


def _duck(t, select):
    con = oracle_connection()
    con.register("t", t)
    return con.execute(f"SELECT {select} AS r FROM t ORDER BY id").fetch_arrow_table().column("r").to_pylist()


def _mx(t, expr):
    out = LazyFrame(t).with_columns(expr.alias("r")).compute(device="cpu")
    ids = out.column("id").to_pylist()
    r = out.column("r").to_pylist()
    return [r[ids.index(i)] for i in range(t.num_rows)]


FLOAT_KEYS = [_nan(1), 0.0, -0.0, None, float("inf"), _nan(2), float("-inf"), None, float("inf"), 0.0, _nan(1)]

CASES = [
    ("sum", col("v").sum().over("k"), "sum(v) OVER (PARTITION BY k)"),
    ("mean", col("v").mean().over("k"), "avg(v) OVER (PARTITION BY k)"),
    ("min", col("v").min().over("k"), "min(v) OVER (PARTITION BY k)"),
    ("max", col("v").max().over("k"), "max(v) OVER (PARTITION BY k)"),
    ("rank", col("v").rank().over("k"), "rank() OVER (PARTITION BY k ORDER BY v)"),
    ("row_number", row_number().over("k", order_by="v"), "row_number() OVER (PARTITION BY k ORDER BY v)"),
]


@pytest.mark.parametrize("name,expr,sql", CASES, ids=[c[0] for c in CASES])
def test_float_partition_key_matches_duckdb(name, expr, sql):
    t = _table(FLOAT_KEYS)
    assert _mx(t, expr) == _duck(t, sql)


def test_sum_over_two_nan_payloads_is_one_partition():
    t = _table([_nan(1), _nan(2), 1.0, 1.0])
    assert _mx(t, col("v").sum().over("k")) == [3.0, 3.0, 7.0, 7.0]


def test_negative_zero_and_zero_are_one_partition():
    t = _table([0.0, -0.0, 5.0], v=[1.0, 2.0, 10.0])
    assert _mx(t, col("v").sum().over("k")) == [3.0, 3.0, 10.0]


def test_null_is_its_own_partition_apart_from_nan():
    t = _table([None, float("nan"), None, float("nan")])
    assert _mx(t, col("v").sum().over("k")) == [4.0, 6.0, 4.0, 6.0]


def test_multi_column_partition_with_float_key():
    t = _table([_nan(1), _nan(2), _nan(1), 0.0, -0.0, None], k2=["a", "a", "b", "x", "x", "x"])
    got = _mx(t, col("v").sum().over("k", "k2"))
    assert got == _duck(t, "sum(v) OVER (PARTITION BY k, k2)")


def test_window_min_max_over_nan_values_follow_duckdb():
    nan = float("nan")
    t = pa.table({"k": [1, 1, 1, 2, 2, 3], "v": [nan, 2.0, 5.0, nan, nan, 4.0], "id": list(range(6))})
    for op, expr in (("min", col("v").min().over("k")), ("max", col("v").max().over("k"))):
        got, want = _mx(t, expr), _duck(t, f"{op}(v) OVER (PARTITION BY k)")
        assert repr(got) == repr(want), (op, got, want)


@pytest.mark.parametrize("keys,typ", [([1, 2, 1, None, 2, None], pa.int64()),
                                      (["a", "b", "a", None, "b", None], pa.string())],
                         ids=["int", "string"])
def test_int_and_string_partition_keys_still_match_duckdb(keys, typ):
    t = _table(keys, key_type=typ)
    assert _mx(t, col("v").sum().over("k")) == _duck(t, "sum(v) OVER (PARTITION BY k)")
