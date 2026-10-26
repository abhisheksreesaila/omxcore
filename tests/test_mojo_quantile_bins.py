"""F-177: `quantile_bin(x, n)`, an equal-count bin number 1 .. n of a value, as a group key.

The edge rule (docs/omxcore-mojo-api.md, "Quantile bins"): take the N non-NULL values of x that the query's WHERE lets through, sorted. Edge k
(k = 1 .. n-1) is the value at sorted place ceil(k * N / n) (1-based; whole-number arithmetic, never a float p). The bin of a value v is
1 + the number of edges below v, so equal values always share a bin (a run of ties that crosses an edge falls into the lower bin), a NULL has no
bin (NULL), and a value above every edge is in bin n. This is `quantile_disc(x, k / n)` at the places where k / n is exact, and, for distinct values
and an N that n divides, `ntile(n) OVER (ORDER BY x)`.

The oracle is written here, independent of the engine: Python sorts the column that DuckDB holds, picks the edges by the rule, and DuckDB counts
the rows in each bin with a CASE over those edges (so the comparison also covers DECIMAL, DATE and FLOAT edges). The Mojo side is
mojo/tools/sql_probe.mojo (the harness of tests/test_mojo_sql.py).
"""
import math
from decimal import Decimal

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mx_oracle import assert_equal
from test_mojo_sql import probe, run_probe, got_rows, assert_source_path, Dataset  # noqa: F401  (probe is a fixture)


@pytest.fixture(scope="module")
def sets(tmp_path_factory):
    d = tmp_path_factory.mktemp("qbins")
    return {"g8": Dataset("g8", 20_000, 8, d, 21), "g10k": Dataset("g10k", 200_000, 10_000, d, 22)}


def sql_literal(v):
    if isinstance(v, float):
        return "'nan'::DOUBLE" if math.isnan(v) else repr(v)
    if isinstance(v, Decimal):
        return str(v)
    if hasattr(v, "isoformat"):
        return f"DATE '{v.isoformat()}'"
    return str(v)


def edges_by_the_rule(con, expr, where, n):
    """The n - 1 edges of `expr` over the rows `where` keeps, by the documented rule, from a sort done here."""
    values = [r[0] for r in con.execute(f"SELECT {expr} FROM t WHERE ({where}) AND {expr} IS NOT NULL ORDER BY 1").fetchall()]
    total = len(values)
    return [values[max(-(-k * total // n) - 1, 0)] for k in range(1, n)] if total else []


def bin_case(expr, edges):
    adds = " + ".join(f"({expr} > {sql_literal(e)})::INTEGER" for e in edges)
    return f"CASE WHEN {expr} IS NULL THEN NULL ELSE 1 + {adds or '0'} END"


CASES = [
    # (expression, n, WHERE, extra aggregates)
    ("x", 10, "TRUE", "min(x), max(x), sum(d)"),
    ("x", 4, "TRUE", "count(x), avg(d)"),
    ("d", 7, "k = 1", "min(d), max(d), count(*)"),
    ("e", 16, "x > 0 OR x IS NULL", "min(e), max(e)"),
    ("dt", 5, "TRUE", "min(dt), max(dt)"),
    ("x * 2 + 1", 6, "g < 4", "min(x), max(x)"),
    ("x", 32, "TRUE", "min(x), max(x)"),
    ("x", 2, "x IS NOT NULL", "min(x), max(x)"),
]


@pytest.mark.parametrize("mode", ["table", "scan", "frame", "framescan"])
@pytest.mark.parametrize("ds_name", ["g8", "g10k"])
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}-{c[1]}")
def test_equals_the_rule(probe, sets, ds_name, mode, case):
    expr, n, where, extra = case
    ds = sets[ds_name]
    con = ds.oracle(mode)
    edges = edges_by_the_rule(con, expr, where, n)
    sql = f"SELECT quantile_bin({expr}, {n}) AS qb, count(*), {extra} FROM t WHERE {where} GROUP BY qb"
    res = run_probe(probe, sql, {"t": ds.path}, mode=mode, edits=ds.edits_path if mode in ("frame", "framescan") else None)
    assert_source_path(res, mode)
    oracle = f"SELECT {bin_case(expr, edges)} AS qb, count(*), {extra} FROM t WHERE {where} GROUP BY qb"
    assert_equal(got_rows(res), con.execute(oracle).fetchall(), False, f"{ds_name}/{mode}: {sql}")
    group_note = next(n_ for n_ in res["notes"] if n_.startswith("group:"))
    assert group_note.startswith("group: dense"), group_note               # the bin number is 1 .. n: a dense key
    bins = {r[0] for r in got_rows(res)}
    assert bins - {None} <= set(range(1, n + 1))


def test_two_keys_and_a_having(probe, sets):
    ds = sets["g8"]
    con = ds.table
    edges = edges_by_the_rule(con, "x", "TRUE", 4)
    sql = "SELECT g, quantile_bin(x, 4), count(*), sum(d) FROM t GROUP BY g, quantile_bin(x, 4) HAVING count(*) > 10"
    res = run_probe(probe, sql, {"t": ds.path})
    oracle = f"SELECT g, {bin_case('x', edges)}, count(*), sum(d) FROM t GROUP BY 1, 2 HAVING count(*) > 10"
    assert_equal(got_rows(res), con.execute(oracle).fetchall(), False, sql)


@pytest.mark.parametrize("group_by", ["qb, qc", "1, 2", "quantile_bin(x, 4), quantile_bin(d, 3)"])
def test_two_bin_keys_in_one_group_by(probe, sets, group_by):
    ds = sets["g8"]
    con = ds.table
    ex, ed = edges_by_the_rule(con, "x", "TRUE", 4), edges_by_the_rule(con, "d", "TRUE", 3)
    sql = f"SELECT quantile_bin(x, 4) qb, quantile_bin(d, 3) qc, count(*), sum(e) FROM t GROUP BY {group_by}"
    res = run_probe(probe, sql, {"t": ds.path})
    oracle = f"SELECT {bin_case('x', ex)}, {bin_case('d', ed)}, count(*), sum(e) FROM t GROUP BY 1, 2"
    assert_equal(got_rows(res), con.execute(oracle).fetchall(), False, sql)


def test_three_bin_keys_and_one_column_binned_twice(probe, sets):
    ds = sets["g8"]
    con = ds.table
    e4, e7 = edges_by_the_rule(con, "x", "TRUE", 4), edges_by_the_rule(con, "x", "TRUE", 7)
    ed, ee = edges_by_the_rule(con, "d", "TRUE", 3), edges_by_the_rule(con, "e", "TRUE", 2)
    sql = "SELECT quantile_bin(x, 4), quantile_bin(x, 7), quantile_bin(d, 3), count(*) FROM t GROUP BY 1, 2, 3"
    res = run_probe(probe, sql, {"t": ds.path})
    oracle = f"SELECT {bin_case('x', e4)}, {bin_case('x', e7)}, {bin_case('d', ed)}, count(*) FROM t GROUP BY 1, 2, 3"
    assert_equal(got_rows(res), con.execute(oracle).fetchall(), False, sql)
    sql = "SELECT quantile_bin(x, 4), quantile_bin(d, 3), quantile_bin(e, 2), count(*) FROM t GROUP BY 1, 2, 3"
    res = run_probe(probe, sql, {"t": ds.path})
    oracle = f"SELECT {bin_case('x', e4)}, {bin_case('d', ed)}, {bin_case('e', ee)}, count(*) FROM t GROUP BY 1, 2, 3"
    assert_equal(got_rows(res), con.execute(oracle).fetchall(), False, sql)


def test_distinct_values_and_an_n_that_divides_the_rows_are_ntile(probe, tmp_path):
    path = tmp_path / "distinct.parquet"
    values = [(i * 7919) % 1000 for i in range(1000)]               # a permutation of 0 .. 999
    pq.write_table(pa.table({"x": pa.array(values, pa.int64())}), path)
    con = duckdb.connect()
    con.execute(f"create table t as select * from read_parquet('{path}')")
    for n in (2, 4, 8, 10, 20):
        res = run_probe(probe, f"SELECT x, quantile_bin(x, {n}) FROM t GROUP BY x, quantile_bin(x, {n})", {"t": path})
        want = con.execute(f"SELECT x, ntile({n}) OVER (ORDER BY x) FROM t").fetchall()
        assert_equal(got_rows(res), want, False, f"ntile({n})")


def test_ties_share_a_bin_and_the_run_across_an_edge_goes_to_the_lower_bin(probe, tmp_path):
    path = tmp_path / "ties.parquet"
    # N = 12 sorted: 1 1 1 1 1 2 2 2 2 3 3 3; n = 3: edges are places 4 and 8 -> 1 and 2; so 1 -> bin 1, 2 -> bin 2, 3 -> bin 3
    pq.write_table(pa.table({"x": pa.array([1] * 5 + [2] * 4 + [3] * 3, pa.int64())}), path)
    res = run_probe(probe, "SELECT x, quantile_bin(x, 3), count(*) FROM t GROUP BY x, quantile_bin(x, 3)", {"t": path})
    assert sorted(got_rows(res)) == [(1, 1, 5), (2, 2, 4), (3, 3, 3)]
    # N = 6: 5 5 5 5 7 9, n = 3: edges at places 2 and 4 -> 5 and 5: 5 -> bin 1, 7 and 9 -> bin 3 (1 + the edges below them: both)
    pq.write_table(pa.table({"x": pa.array([5, 5, 5, 5, 7, 9], pa.int64())}), path)
    res = run_probe(probe, "SELECT x, quantile_bin(x, 3) FROM t GROUP BY x, quantile_bin(x, 3)", {"t": path})
    assert sorted(got_rows(res)) == [(5, 1), (7, 3), (9, 3)]


def test_null_has_no_bin_and_a_column_of_nothing_is_one_null_group(probe, tmp_path):
    path = tmp_path / "nulls.parquet"
    pq.write_table(pa.table({"x": pa.array([None, 4, None, 8, 6, 2], pa.int64()), "y": pa.array([None] * 6, pa.int64())}), path)
    res = run_probe(probe, "SELECT quantile_bin(x, 2), count(*), min(x) FROM t GROUP BY quantile_bin(x, 2)", {"t": path})
    assert sorted(got_rows(res), key=lambda r: (r[0] is None, r[0])) == [(1, 2, 2), (2, 2, 6), (None, 2, None)]      # edge 1 = place ceil(4/2) = 2 -> 4
    res = run_probe(probe, "SELECT quantile_bin(y, 4), count(*) FROM t GROUP BY quantile_bin(y, 4)", {"t": path})
    assert got_rows(res) == [(None, 6)]
    res = run_probe(probe, "SELECT quantile_bin(x, 4), count(*) FROM t WHERE x > 100 GROUP BY quantile_bin(x, 4)", {"t": path})
    assert got_rows(res) == []


def test_float_columns_with_nan_and_negative_zero(probe, tmp_path):
    path = tmp_path / "floats.parquet"
    vals = [float(i % 37) * 0.25 - 3.0 for i in range(500)] + [float("nan")] * 30 + [0.0, -0.0] * 5 + [None] * 12
    pq.write_table(pa.table({"f": pa.array(vals, pa.float64()), "v": pa.array(list(range(len(vals))), pa.int64())}), path)
    con = duckdb.connect()
    con.execute(f"create table t as select * from read_parquet('{path}')")
    for n in (3, 10):
        edges = edges_by_the_rule(con, "f", "TRUE", n)
        res = run_probe(probe, f"SELECT quantile_bin(f, {n}) AS qb, count(*), sum(v) FROM t GROUP BY qb", {"t": path})
        oracle = f"SELECT {bin_case('f', edges)} AS qb, count(*), sum(v) FROM t GROUP BY qb"
        assert_equal(got_rows(res), con.execute(oracle).fetchall(), False, f"float n={n}")


def test_it_is_refused_where_it_means_nothing(probe, sets):
    ds = sets["g8"]
    for sql, text in (("SELECT quantile_bin(x, 1) FROM t GROUP BY 1", "from 2 to 32"),
                      ("SELECT quantile_bin(x, 33) FROM t GROUP BY 1", "from 2 to 32"),
                      ("SELECT quantile_bin(x, k) FROM t GROUP BY 1", "from 2 to 32"),
                      ("SELECT quantile_bin(5, 4) FROM t", "not a constant"),
                      ("SELECT quantile_bin(s, 4) FROM t GROUP BY 1", "needs numbers or dates"),
                      ("SELECT quantile_bin(x) FROM t", "takes a value and the number of bins"),
                      ("SELECT quantile_bin(sum(x), 4) FROM t", "cannot take an aggregate"),
                      ("SELECT quantile_bin(x, 4) FROM t", "not in a join or a row query"),
                      ("SELECT count(*) FROM t WHERE quantile_bin(x, 2) = 1", "supported only in GROUP BY"),
                      ("SELECT count(*) FROM t WHERE quantile_bin(x, 2) = 1 GROUP BY k", "supported only in GROUP BY"),
                      ("SELECT quantile_bin(quantile_bin(x, 3), 2), count(*) FROM t GROUP BY 1", "supported only in GROUP BY"),
                      ("SELECT k, sum(quantile_bin(x, 4)) FROM t GROUP BY k", "supported only in GROUP BY"),
                      ("SELECT k, count(*) FROM t GROUP BY k HAVING sum(CASE WHEN quantile_bin(x, 2) = 1 THEN 1 ELSE 0 END) > 1", "supported only in GROUP BY"),
                      ("SELECT k, CASE WHEN quantile_bin(x, 2) = 1 THEN 1 ELSE 0 END FROM t GROUP BY k", "column 21: quantile_bin() bins the rows of a table: it cannot take an aggregate"),
                      ("SELECT k, count(*) FROM t GROUP BY k ORDER BY quantile_bin(x, 2)", "column 47: quantile_bin() bins the rows of a table: it cannot take an aggregate"),
                      ("SELECT count(*) FROM t a JOIN t b ON a.id = b.id GROUP BY quantile_bin(a.x, 2)", "not in a join or a row query yet")):
        res = run_probe(probe, sql, {"t": ds.path}, expect_error=True)
        assert text in res["error"], (sql, res)
