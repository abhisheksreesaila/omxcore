"""F-177: stddev / stddev_samp / stddev_pop / variance / var_samp / var_pop in SQL, equal to DuckDB.

The Mojo side is mojo/tools/sql_probe.mojo (the harness of tests/test_mojo_sql.py). Every query runs over a table read whole, a Parquet
file streamed as 13 row groups, a Frame after 300 edits and a streamed file with the same edits, against DuckDB's answer. DuckDB's own
algorithm (Welford) and ours (a two-pass piece, merged with Chan's formula) are not bit for bit the same, so floats are compared to 1e-9
relative; everything else (NULLs, which groups have a value, the kinds) is exact. A column of 1e15 + small numbers checks the cancellation
a sum-of-squares formula loses.
"""
import math
from fractions import Fraction

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_mojo_sql import probe, run_probe, got_rows, assert_source_path, Dataset  # noqa: F401  (probe is a fixture)

REL = 1e-9


@pytest.fixture(scope="module")
def sets(tmp_path_factory):
    d = tmp_path_factory.mktemp("spread")
    return {"g8": Dataset("g8", 20_000, 8, d, 11), "g10k": Dataset("g10k", 200_000, 10_000, d, 12)}


def close(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        a, b = float(a), float(b)
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        if math.isinf(a) or math.isinf(b):
            return a == b
        return abs(a - b) <= REL * max(abs(a), abs(b))
    return a == b


def assert_close(got, want, nkeys, label):
    def key(r):
        return tuple((0, 0) if v is None else (1, v) for v in r[:nkeys])
    g = sorted(got, key=key)
    w = sorted((tuple(r) for r in want), key=key)
    assert len(g) == len(w), f"{label}: {len(g)} rows, DuckDB has {len(w)}"
    for i, (a, b) in enumerate(zip(g, w)):
        assert len(a) == len(b) and all(close(x, y) for x, y in zip(a, b)), f"{label}: row {i}: got {a}, DuckDB has {b}"


ALL4 = "stddev({0}), stddev_pop({0}), variance({0}), var_pop({0})"

QUERIES = [
    ("global", 0, "SELECT " + ", ".join(ALL4.format(c) for c in ("x", "d", "e")) + ", stddev_samp(x), var_samp(e) FROM t"),
    ("by-k", 1, "SELECT k, " + ", ".join(ALL4.format(c) for c in ("x", "d", "e")) + ", count(*) FROM t GROUP BY k"),
    ("by-g", 1, "SELECT g, " + ALL4.format("x") + ", " + ALL4.format("d") + " FROM t GROUP BY g"),
    ("empty", 0, "SELECT stddev(x), var_pop(d), count(*) FROM t WHERE x > 5000"),
    ("empty-groups", 1, "SELECT k, stddev(x), var_pop(x) FROM t WHERE x > 5000 GROUP BY k"),
    ("expressions", 1, "SELECT k, stddev(x * d), variance(x + 1), var_pop(d - e), stddev_pop(x * 2) FROM t GROUP BY k"),
    ("mixed", 1, "SELECT k, count(*), avg(x), stddev(x), median(x), count(DISTINCT s), min(dt), var_pop(e) FROM t GROUP BY k",),
    ("by-text", 1, "SELECT s, stddev(d), var_pop(x) FROM t GROUP BY s",),
    ("by-two", 2, "SELECT b, k, stddev(e), stddev_pop(e) FROM t GROUP BY b, k"),
    ("filtered", 1, "SELECT k, stddev(x), variance(d) FROM t WHERE x BETWEEN -500 AND 500 AND d IS NOT NULL GROUP BY k"),
    ("having", 1, "SELECT k, stddev(x) AS sd FROM t GROUP BY k HAVING stddev(x) > 100 AND count(*) > 2"),
    ("arithmetic", 1, "SELECT k, stddev(x) * 2.5 - var_pop(x), variance(x) + 1.5, stddev(x) + stddev_pop(x) FROM t GROUP BY k"),
]


def ids(q):
    return q[0]


@pytest.mark.parametrize("mode", ["table", "scan", "frame", "framescan"])
@pytest.mark.parametrize("ds_name", ["g8", "g10k"])
@pytest.mark.parametrize("q", QUERIES, ids=ids)
def test_equals_duckdb(probe, sets, ds_name, mode, q):
    name, nkeys, sql = q
    ds = sets[ds_name]
    res = run_probe(probe, sql, {"t": ds.path}, mode=mode, edits=ds.edits_path if mode in ("frame", "framescan") else None)
    assert_source_path(res, mode)
    cur = ds.oracle(mode).execute(sql)
    want = cur.fetchall()
    kinds = [c["kind"] for c in res["columns"]]
    for i, desc in enumerate(cur.description):
        if "stddev" in sql or "var" in sql:
            if i >= nkeys and str(desc[1]) == "DOUBLE":
                assert kinds[i] == "FLOAT", f"column {i} of `{sql}`: {kinds[i]}"
    assert_close(got_rows(res), want, nkeys, f"{ds_name}/{mode}: {sql}")


def test_the_sample_variants_are_null_below_two_values_the_population_ones_below_one(probe, sets):
    ds = sets["g10k"]
    sql = "SELECT g, count(x), stddev(x), stddev_pop(x), variance(x), var_pop(x) FROM t WHERE id < 200 GROUP BY g"      # 200 groups of one row
    res = run_probe(probe, sql, {"t": ds.path})
    rows = got_rows(res)
    seen = {1: False, 0: False}
    for g, n, sd, sdp, v, vp in rows:
        if n == 0:
            assert sd is None and sdp is None and v is None and vp is None, (g, n)
            seen[0] = True
        elif n == 1:
            assert sd is None and v is None and sdp == 0.0 and vp == 0.0, (g, n, sd, sdp)
            seen[1] = True
        else:
            assert sd is not None and v is not None
    assert seen[1], "the data has a group of one value"


def test_a_hashed_group_key_and_the_holistic_second_pass(probe, sets):
    ds = sets["g8"]
    sql = "SELECT s, k, stddev(x), var_pop(d), count(*) FROM t GROUP BY s, k"
    for strategy in ("dense", "hashed"):
        res = run_probe(probe, sql, {"t": ds.path}, group=strategy)
        assert_close(got_rows(res), ds.table.execute(sql).fetchall(), 2, f"{strategy}: {sql}")
        if strategy == "hashed":
            assert any(n.startswith("ran: holistic second pass") for n in res["notes"]), res["notes"]


def test_it_is_refused_on_text_dates_and_booleans(probe, sets):
    ds = sets["g8"]
    for col, kind in (("s", "TEXT"), ("dt", "DATE"), ("b", "BOOL")):
        for fn in ("stddev", "var_pop"):
            res = run_probe(probe, f"SELECT {fn}({col}) FROM t", {"t": ds.path}, expect_error=True)
            assert "needs numbers" in res["error"] and kind in res["error"], res
    res = run_probe(probe, "SELECT stddev(x, 2) FROM t", {"t": ds.path}, expect_error=True)
    assert "takes 1 argument" in res["error"], res
    res = run_probe(probe, "SELECT stddev(DISTINCT x) FROM t", {"t": ds.path}, expect_error=True)
    assert "not supported" in res["error"], res


# ── numbers where a naive formula fails ──────────────────────────────────────────────────────────────

def make_hard_file(path, n=60_000):
    """g (7 groups, one of a single row and one NULL group), small BIGINT, big BIGINT (1e15 + a few), huge BIGINT (near +-4e18),
    d DECIMAL(18,6), f DOUBLE (1e9 + fractions, with NaN and inf in group 5), all NULL in group 6."""
    import random
    rng = random.Random(5)
    g, small, big, huge, d, f = [], [], [], [], [], []
    for i in range(n):
        gi = rng.randrange(0, 5) if i > 2 else i + 5          # groups 5, 6, 7: rows 0, 1, 2 are alone in theirs
        g.append(gi)
        small.append(rng.randint(-5, 5))
        big.append(10**15 + rng.randint(1, 9) if rng.random() > 0.05 else None)
        huge.append(rng.choice([-1, 1]) * (4 * 10**18 - rng.randint(0, 10**6)) if rng.random() > 0.3 else None)
        d.append(None if rng.random() < 0.1 else __import__("decimal").Decimal(rng.randint(-10**9, 10**9)).scaleb(-6))
        f.append(None if gi == 6 else 1e9 + rng.random() * 100)
    f[2] = float("nan")                                       # group 7: a single NaN
    t = pa.table({"g": pa.array(g, pa.int64()), "small": pa.array(small, pa.int64()), "big": pa.array(big, pa.int64()),
                  "huge": pa.array(huge, pa.int64()), "d": pa.array(d, pa.decimal128(18, 6)), "f": pa.array(f, pa.float64())})
    pq.write_table(t, path, row_group_size=n // 7 + 1)


@pytest.fixture(scope="module")
def hard(tmp_path_factory):
    path = tmp_path_factory.mktemp("spread-hard") / "hard.parquet"
    make_hard_file(path)
    con = duckdb.connect()
    con.execute(f"create table t as select * from read_parquet('{path}')")
    return path, con


def exact_spread(values):
    """(stddev, stddev_pop, variance, var_pop) of the finite doubles in `values` by exact rational arithmetic (None below the needed count),
    the oracle where DuckDB's own Welford loses digits."""
    if not values:
        return (None, None, None, None)
    if len(values) == 1:                                      # one value is a spread of 0, even a NaN (DuckDB)
        return (None, 0.0, None, 0.0)
    xs = [Fraction(v) for v in values]
    n = len(xs)
    mean = sum(xs) / n
    m2 = sum((x - mean) ** 2 for x in xs)
    vp = float(m2 / n)
    vs = float(m2 / (n - 1)) if n > 1 else None
    return (math.sqrt(vs) if vs is not None else None, math.sqrt(vp), vs, vp)


def as_doubles(column, scale=0):
    """The values as DuckDB casts them to DOUBLE: float(int) for an integer, float(unscaled) / 10^scale for a decimal."""
    out = []
    for v in column:
        if v is None:
            out.append(None)
        elif scale:
            out.append(float(int(v.scaleb(scale))) / 10.0 ** scale)
        else:
            out.append(float(v))
    return out


NAN_GROUP = 7                                                 # the group whose only value is a NaN: left out of the whole-table query (DuckDB raises for it)
EXACT_COLS = [("big", 0), ("huge", 0), ("f", 0), ("small", 0), ("d", 6)]


def expected_exact(path, by_group):
    t = pq.read_table(path).to_pydict()
    cols = {name: as_doubles(t[name], scale) for name, scale in EXACT_COLS}
    groups = sorted(set(t["g"])) if by_group else [None]
    out = {}
    for g in groups:
        row = []
        for name, _ in EXACT_COLS:
            vals = [v for gi, v in zip(t["g"], cols[name]) if v is not None and (g == gi if by_group else gi != NAN_GROUP)]
            row.append(exact_spread(vals))
        out[g] = row
    return out


@pytest.mark.parametrize("mode", ["table", "scan"])
@pytest.mark.parametrize("by_group", [True, False])
def test_equals_exact_arithmetic_where_a_sum_of_squares_loses_every_digit(probe, hard, mode, by_group):
    """Values 1e15 + (1..9), 4e18 - small, 1e9 + a fraction: DuckDB's Welford is off by 1e-5 on the first and 1e-9 on the third; the exact
    rational result is what the engine must give (to 1e-12), for every one of the four functions."""
    path, _ = hard
    cols = ", ".join(", ".join(f"{fn}({c})" for fn in ("stddev", "stddev_pop", "variance", "var_pop")) for c, _ in EXACT_COLS)
    sql = f"SELECT g, {cols} FROM t GROUP BY g" if by_group else f"SELECT {cols} FROM t WHERE g <> {NAN_GROUP}"
    res = run_probe(probe, sql, {"t": path}, mode=mode)
    rows = got_rows(res)
    want = expected_exact(path, by_group)
    assert len(rows) == len(want)
    for r in rows:
        g = r[0] if by_group else None
        vals = r[1:] if by_group else r
        for ci, (name, _) in enumerate(EXACT_COLS):
            for k, label in enumerate(("stddev", "stddev_pop", "variance", "var_pop")):
                got, exp = vals[ci * 4 + k], want[g][ci][k]
                assert got is None and exp is None or (got is not None and exp is not None and (
                    (math.isnan(got) and math.isnan(exp)) or abs(got - exp) <= 1e-12 * abs(exp))), f"{mode} group {g} {label}({name}): {got} != {exp}"


DUCK_OK = [
    "SELECT g, stddev(small), var_pop(small), stddev(d), variance(d), stddev_pop(d) FROM t GROUP BY g",
    "SELECT stddev(d), var_pop(small), stddev_pop(d), variance(small) FROM t",
]


@pytest.mark.parametrize("mode", ["table", "scan"])
@pytest.mark.parametrize("sql", DUCK_OK)
def test_where_duckdb_is_accurate_it_is_equal(probe, hard, mode, sql):
    path, con = hard
    res = run_probe(probe, sql, {"t": path}, mode=mode)
    assert_close(got_rows(res), con.execute(sql).fetchall(), 1 if sql.startswith("SELECT g") else 0, f"{mode}: {sql}")


def test_the_variance_of_1e15_plus_1_2_3_is_exactly_1(probe, tmp_path):
    path = tmp_path / "c.parquet"
    pq.write_table(pa.table({"v": pa.array([10**15 + 1, 10**15 + 2, 10**15 + 3], pa.int64())}), path)
    res = run_probe(probe, "SELECT variance(v), stddev(v), var_pop(v) FROM t", {"t": path})
    v, sd, vp = got_rows(res)[0]
    assert v == 1.0 and sd == 1.0 and abs(vp - 2 / 3) < 1e-15, (v, sd, vp)


def test_nan_and_infinity_as_in_duckdb(probe, tmp_path):
    """One NaN or infinity is a spread of 0 (the sample variants NULL); two or more values with one of them, or an overflow, is an error."""
    path = tmp_path / "n.parquet"
    pq.write_table(pa.table({"k": pa.array([1, 2, 3, 3], pa.int64()),
                             "f": pa.array([float("nan"), float("inf"), 2.0, 4.0], pa.float64())}), path)
    con = duckdb.connect()
    con.execute(f"create table t as select * from read_parquet('{path}')")
    sql = "SELECT k, stddev(f), stddev_pop(f), variance(f), var_pop(f) FROM t WHERE k < 3 GROUP BY k"
    res = run_probe(probe, sql, {"t": path})
    assert_close(got_rows(res), con.execute(sql).fetchall(), 1, sql)
    assert sorted(got_rows(res)) == [(1, None, 0.0, None, 0.0), (2, None, 0.0, None, 0.0)]
    for rows, fn in (([1.0, float("nan"), 3.0], "stddev"), ([1.0, float("inf")], "var_pop"), ([1e308, -1e308], "stddev_pop"), ([1e154, -1e154], "var_samp")):
        p = tmp_path / f"{fn}.parquet"
        pq.write_table(pa.table({"f": pa.array(rows, pa.float64())}), p)
        res = run_probe(probe, f"SELECT {fn}(f) FROM t", {"t": p}, expect_error=True)
        assert res["error"].startswith(f"{'stddev_samp' if fn == 'stddev' else fn} is out of range"), res
        with pytest.raises(duckdb.OutOfRangeException):
            duckdb.connect().execute(f"select {fn}(f) from read_parquet('{p}')")
