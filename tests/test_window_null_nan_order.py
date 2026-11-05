"""F-084: window functions treat NULL, NaN and DESC order like DuckDB. Expected values come from
DuckDB itself, over several partitions, float32 and float64, with and without ORDER BY."""
import duckdb
from conftest import oracle_connection
import pyarrow as pa
import pytest

from omxcore import LazyFrame, col, row_number

NAN = float("nan")
# three partitions; NULLs, NaNs (ties), -0.0 / 0.0, and one all-NULL partition (k=3)
K = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3]
V = [NAN, 2.0, None, 5.0, NAN, -0.0, 0.0, None, 0.0, 1.5, None, None]
O = [3.0, None, NAN, 1.0, 3.0, -1.0, NAN, 2.0, None, 2.0, 1.0, 1.0]


def _table(typ):
    n = len(K)
    return pa.table({"k": pa.array(K, pa.int64()), "v": pa.array(V, typ), "o": pa.array(O, typ),
                     "id": pa.array(range(n), pa.int64())})


def _duck(t, select):
    con = oracle_connection()
    con.register("t", t)
    return con.execute(f"SELECT {select} AS r FROM t ORDER BY id").to_arrow_table().column("r").to_pylist()


def _mx(t, expr):
    out = LazyFrame(t).with_columns(expr.alias("r")).compute(device="cpu")
    ids, r = out.column("id").to_pylist(), out.column("r").to_pylist()
    return [r[ids.index(i)] for i in range(t.num_rows)]


def _same(got, want):
    # repr tells NaN from NULL and 0.0 from -0.0 apart; numbers compare as floats (DuckDB rank is int)
    def f(xs):
        return [None if x is None else float(x) for x in xs]
    assert repr(f(got)) == repr(f(want)), (got, want)


TYPES = [pa.float64(), pa.float32()]


@pytest.mark.parametrize("typ", TYPES, ids=str)
@pytest.mark.parametrize("op", ["min", "max"])
def test_window_min_max_skip_null_and_give_null_when_none_valid(typ, op):
    t = _table(typ)
    expr = getattr(col("v"), op)().over("k")
    _same(_mx(t, expr), _duck(t, f"{op}(v) OVER (PARTITION BY k)"))


@pytest.mark.parametrize("typ", TYPES, ids=str)
@pytest.mark.parametrize("fn", ["rank", "dense_rank"])
@pytest.mark.parametrize("desc", [False, True])
def test_rank_ties_nan_zero_and_null(typ, fn, desc):
    t = _table(typ)
    expr = getattr(col("v"), fn)().over("k", order_by="v", descending=desc)
    _same(_mx(t, expr), _duck(t, f"{fn}() OVER (PARTITION BY k ORDER BY v {'DESC' if desc else 'ASC'})"))


@pytest.mark.parametrize("typ", TYPES, ids=str)
@pytest.mark.parametrize("desc", [False, True])
def test_row_number_order_by_nan_and_null(typ, desc):
    t = _table(typ)
    expr = row_number().over("k", order_by="o", descending=desc)
    # id breaks ties so the expected answer is fixed (the engine keeps row order on ties)
    _same(_mx(t, expr), _duck(t, f"row_number() OVER (PARTITION BY k ORDER BY o {'DESC' if desc else 'ASC'}, id)"))


@pytest.mark.parametrize("typ", TYPES, ids=str)
@pytest.mark.parametrize("desc", [False, True])
def test_running_sum_follows_the_window_order(typ, desc):
    t = _table(typ)
    t = t.set_column(1, "v", pa.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 1.0, 2.0, 4.0, 8.0, 1.0, 2.0], typ))
    expr = col("v").cum_sum().over("k", order_by="o", descending=desc)
    d = "DESC" if desc else "ASC"
    _same(_mx(t, expr), _duck(t, f"sum(v) OVER (PARTITION BY k ORDER BY o {d}, id ROWS UNBOUNDED PRECEDING)"))


def test_nan_is_first_under_desc_and_last_under_asc_with_null_after_it():
    t = pa.table({"k": [1, 1, 1, 1], "o": [1.0, NAN, None, 3.0], "id": [0, 1, 2, 3]})
    assert _mx(t, row_number().over("k", order_by="o")) == [1.0, 3.0, 4.0, 2.0]
    assert _mx(t, row_number().over("k", order_by="o", descending=True)) == [3.0, 1.0, 4.0, 2.0]


def test_window_dtype_is_float64():
    t = pa.table({"k": [1, 1], "v": pa.array([1, 4], pa.int64()), "id": [0, 1]})
    out = LazyFrame(t).with_columns(col("v").max().over("k").alias("r")).compute(device="cpu")
    assert out.schema.field("r").type == pa.float64()
    assert out.column("r").to_pylist() == [4.0, 4.0]


# sum / mean / cum_sum skip NULLs: k=1 leading NULLs then values, k=2 a NaN among values,
# k=3 all NULL, k=4 no NULLs
SK = [1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4]
SV = [None, None, 2.0, None, 5.0, NAN, 1.0, None, None, 3.0, 4.0]
SO = [1.0, 2, 3, 4, 5, 1, 2, 1, 2, 2, 1]


def _sum_table(typ):
    return pa.table({"k": pa.array(SK, pa.int64()), "v": pa.array(SV, typ), "o": pa.array(SO, typ),
                     "id": pa.array(range(len(SK)), pa.int64())})


@pytest.mark.parametrize("typ", TYPES, ids=str)
@pytest.mark.parametrize("op,sql", [("sum", "sum"), ("mean", "avg")])
def test_window_sum_mean_skip_null_and_give_null_when_none_valid(typ, op, sql):
    t = _sum_table(typ)
    _same(_mx(t, getattr(col("v"), op)().over("k")), _duck(t, f"{sql}(v) OVER (PARTITION BY k)"))


@pytest.mark.parametrize("typ", TYPES, ids=str)
def test_running_sum_is_null_until_first_valid_value(typ):
    t = _sum_table(typ)
    expr = col("v").cum_sum().over("k", order_by="o")
    _same(_mx(t, expr), _duck(t, "sum(v) OVER (PARTITION BY k ORDER BY o, id ROWS UNBOUNDED PRECEDING)"))


def test_running_sum_without_partition_skips_nulls():
    t = pa.table({"v": [None, 1.0, None, 2.0], "id": [0, 1, 2, 3]})
    assert _mx(t, col("v").cum_sum().over(order_by="id")) == [None, 1.0, 1.0, 3.0]
