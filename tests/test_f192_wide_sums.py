"""F-192: a SUM of INT or DECIMAL values whose total passes int64 is DuckDB's answer (DuckDB sums into HUGEINT / DECIMAL(38)), where the engine raised
"a sum or an expression passed the int64 range". Grouped, global and window sums, with NULLs and negatives, then what a query does with such a total
(ORDER BY, HAVING, arithmetic with a constant, a scalar subquery: TPC-H Q11's shape). Every answer is compared with DuckDB through the real `mx`
command (tests/mx_oracle.py): kinds, DECIMAL scales and every digit.

The data is built so that a group's total, and the whole column's, are far past 9.2e18 while single values stay in range (a DECIMAL(18, 2) value is
at most 9.99e17 stored as 99999999999999999), with some groups that stay small (so one column mixes narrow and wide totals), one group of NULLs only,
and negative totals.
"""
import random
import subprocess
from decimal import Decimal

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mx_oracle import check, mx_binary, run_mx

ROWS = 600


@pytest.fixture(scope="module")
def mx():
    return mx_binary()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    rnd = random.Random(192)
    g, d, b, p, q, o = [], [], [], [], [], []
    for i in range(ROWS):
        grp = i % 7
        g.append(None if grp == 6 else grp)
        if grp == 4:                                          # a group of NULLs only: its sums are NULL
            d.append(None)
            b.append(None)
        elif grp == 3:                                        # a small group: its total stays in int64
            d.append(Decimal(rnd.randrange(-5000, 5000)).scaleb(-2))
            b.append(rnd.randrange(-5000, 5000))
        elif grp == 5:                                        # negative values: a negative total far past int64
            d.append(Decimal(-rnd.randrange(10**17, 10**18 - 1)).scaleb(-2))
            b.append(-rnd.randrange(1 << 61, 1 << 62))
        else:
            d.append(Decimal(rnd.randrange(-10**17, 10**18 - 1)).scaleb(-2) if rnd.random() > 0.05 else None)
            b.append(rnd.randrange(-(1 << 60), 1 << 62) if rnd.random() > 0.05 else None)
        p.append(Decimal(rnd.randrange(1, 10**9)).scaleb(-2))   # TPC-H's ps_supplycost: DECIMAL(15, 2)
        q.append(rnd.randrange(1, 10**5))                       # ps_availqty
        o.append(i)
    path = tmp_path_factory.mktemp("f192") / "t.parquet"
    pq.write_table(pa.table({"o": pa.array(o, pa.int64()), "g": pa.array(g, pa.int64()), "d": pa.array(d, pa.decimal128(18, 2)),
                             "b": pa.array(b, pa.int64()), "p": pa.array(p, pa.decimal128(15, 2)), "q": pa.array(q, pa.int32())}),
                   path, row_group_size=97)
    con = duckdb.connect()
    con.execute(f"create table t as select * from read_parquet('{path}')")
    return {"files": {"t": str(path)}, "con": con}


GLOBAL = [
    "select sum(case when d > 0 then d else 0 end), sum(case when d > 0 then 1 else 0 end) from t",
    "select sum(d), min(d), max(d), count(d), avg(d), sum(b), avg(b) from t",
    "select sum(d) from t",
    "select sum(b) from t",
    "select sum(d), sum(b), count(*), count(d) from t",
    "select sum(d) from t where g = 3",                       # stays in int64: the same answer as before
    "select sum(d) from t where g = 4",                       # NULLs only
    "select sum(d) from t where g = 99",                      # no rows
    "select sum(d) from t where g = 5",                       # a negative total
]

GROUPED = [
    "select g, sum(d), sum(b) from t group by g order by g",
    "select g, sum(d) s from t group by g order by s desc",
    "select g, sum(d) s from t group by g order by s",
    "select g, sum(d) s, count(*) c from t group by g order by g",
    "select g, sum(b) s from t group by g having sum(b) > 1000000000000000000 order by g",
    "select g, sum(d) s from t group by g having sum(d) < 0 order by s",
    "select g, sum(d) s from t group by g order by s desc limit 2",
    "select g, sum(d) from t group by g order by g limit 3 offset 1",
    "select o % 3 as k, sum(d) from t group by o % 3 order by k",
    "select sum(d) from t group by g order by 1",
    "select g, sum(d), min(d), max(d), count(d), avg(d), sum(b), min(b), max(b), avg(b) from t group by g order by g",
    "select g, sum(d) s from t group by g having sum(d) > 100000000000000000000.00 order by g",
    "select g, sum(d) s from t group by g having sum(d) is not null order by g",
    "select g, sum(d) s from t group by g having sum(d) is null order by g",
    "select g, sum(d) s from t group by g order by sum(d) * 2 desc, g",
    "select g, sum(d) s from t group by g order by -sum(d), g",
    "select g, sum(d) from t where d > 0 group by g order by g",
    "select o % 5 as k, g, sum(d), sum(b) from t group by o % 5, g order by k, g",
    "select o % 50 as k, sum(p * q), sum(q * 50000000000000) from t group by o % 50 order by k",
    "select g, case when sum(d) > 0 then 1 else 0 end, sum(d) between -1 and 1 from t group by g order by g",
]

ARITHMETIC = [
    "select g, sum(d) + 1 from t group by g order by g",
    "select g, -sum(d) from t group by g order by g",
    "select g, sum(d) * 2 from t group by g order by g",
    "select g, sum(d) * 0.0001 from t group by g order by g",
    "select g, sum(d) * 1.5 - sum(d) from t group by g order by g",
    "select g, sum(d) - sum(d) from t group by g order by g",
    "select g, sum(d) + sum(b) from t group by g order by g",
    "select sum(d) + sum(d), sum(d) * 3 from t",
    "select g, sum(d) > 0, sum(d) = sum(d), sum(d) <= 5 from t group by g order by g",
    "select g, sum(b) * 2 from t group by g order by g",
    "select g, sum(d) / 100 from t group by g order by g",
    "select g, avg(d), avg(b) from t group by g order by g",
    "select avg(d), avg(b) from t",
    "select sum(d) / count(*) from t",
]

Q11_SHAPE = [
    # TPC-H Q11: the per-group total compared with a fraction of the grand total, which is itself past int64 once scaled to 12 places
    "select g, sum(p * q) v from t group by g having sum(p * q) > (select sum(p * q) * 0.0001000000 from t) order by v desc",
    "select g, sum(d) v from t group by g having sum(d) > (select sum(d) * 0.0001 from t) order by v desc",
    "select g, sum(d) v from t group by g having sum(d) > (select sum(d) from t where g = 2) order by g",
    "select g from t group by g having sum(d) > (select sum(d) / 2 from t) order by g",
    "select sum(p * q) * 0.0001000000 from t",
    "select sum(p * q) from t",
]

WINDOWS = [
    "select o, avg(d) over (partition by g), avg(b) over (partition by g) from t order by o",
    "select o, sum(d) over (partition by g order by o rows between 3 preceding and 3 following) from t order by o",
    "select o, sum(d) over (partition by g) as s from t order by s desc, o limit 10",
    "select o, sum(d) over (partition by g) as s from t order by o limit 5 offset 2",
    "select distinct sum(d) over (partition by g) as s from t order by s",
    "select g, sum(d) over (partition by g), sum(d) over (partition by g) from t where o < 40 order by o",
    "select o, sum(d) over (partition by g) from t order by o",
    "select o, sum(d) over (partition by g order by o) from t order by o",
    "select o, sum(b) over (order by o rows between 5 preceding and current row) from t order by o",
    "select o, sum(d) over () from t order by o",
    "select o, sum(d) over (partition by g order by o rows between unbounded preceding and unbounded following) from t order by o",
    "select o, sum(d) over (partition by g) s from t order by s, o",
    "select o, g, sum(d) over (partition by g), sum(b) over (partition by g), count(*) over (partition by g) from t order by o",
]


@pytest.mark.parametrize("sql", GLOBAL)
def test_global_sum(mx, world, sql):
    check(mx, sql, world["con"], world["files"])


@pytest.mark.parametrize("sql", GROUPED)
def test_grouped_sum(mx, world, sql):
    check(mx, sql, world["con"], world["files"], ordered=True)


@pytest.mark.parametrize("sql", ARITHMETIC)
def test_what_a_query_does_with_a_wide_total(mx, world, sql):
    check(mx, sql, world["con"], world["files"], ordered=True)


@pytest.mark.parametrize("sql", Q11_SHAPE)
def test_scalar_subquery_and_having_over_wide_totals(mx, world, sql):
    check(mx, sql, world["con"], world["files"], ordered=True)


@pytest.mark.parametrize("sql", WINDOWS)
def test_window_sum(mx, world, sql):
    check(mx, sql, world["con"], world["files"], ordered=True)


WAYS = [("--group", "hashed"), ("--group", "dense"), ("--threads", "1"), ("--threads", "8"), ("--stream", "always"), ("--stream", "never")]
WAY_QUERIES = [
    "select g, sum(d), sum(b), avg(d) from t group by g order by g",
    "select o % 7 as k, sum(q * 50000000000000), sum(p * q) from t group by o % 7 order by k",
    "select sum(d), sum(b), avg(b) from t",
    "select g, sum(d) s from t group by g having sum(d) > 100000000000000000000.00 order by s desc",
]


@pytest.mark.parametrize("extra", WAYS, ids=lambda e: " ".join(e))
def test_every_way_of_running_the_query_gives_the_same_wide_totals(mx, world, extra):
    """Dense or hashed groups, one thread or many, a file read whole or streamed row group by row group (partial totals merge past int64)."""
    for sql in WAY_QUERIES:
        check(mx, sql, world["con"], world["files"], ordered=True, extra=extra)


@pytest.fixture(scope="module")
def edges(tmp_path_factory):
    """Totals at the very edge of int64: exactly 2^63 - 1 and -2^63 (they fit), one past either (they do not), in groups that mix the two."""
    top, bottom = (1 << 63) - 1, -(1 << 63)
    rows = [(0, top - 5), (0, 5),                  # 2^63 - 1: fits
            (1, top - 5), (1, 6),                  # 2^63: does not
            (2, bottom + 5), (2, -5),              # -2^63: fits
            (3, bottom + 5), (3, -6),              # -2^63 - 1: does not
            (4, top), (4, top), (4, bottom), (4, bottom), (4, 7),     # partial sums pass either end, the total is 7
            (5, None), (5, None),                  # NULLs only
            (6, 1 << 62), (6, 1 << 62), (6, -(1 << 62)), (6, -(1 << 62)), (6, 1 << 62), (6, 1 << 62)]   # climbs past the top and comes back, then ends past it
    path = tmp_path_factory.mktemp("f192e") / "e.parquet"
    pq.write_table(pa.table({"g": pa.array([r[0] for r in rows], pa.int64()), "v": pa.array([r[1] for r in rows], pa.int64())}), path, row_group_size=3)
    con = duckdb.connect()
    con.execute(f"create table e as select * from read_parquet('{path}')")
    return {"files": {"e": str(path)}, "con": con}


EDGES = [
    "select g, sum(v) from e group by g order by g",
    "select g, sum(v) - 1 from e group by g order by g",
    "select g, sum(v) + 1 from e group by g order by g",
    "select g, sum(v) = 9223372036854775807, sum(v) > 9223372036854775807, sum(v) < -9223372036854775808 from e group by g order by g",
    "select sum(v) from e",
    "select sum(v) from e where g in (0, 2)",
    "select sum(v) from e where g in (1, 3)",
    "select sum(v) from e where g = 4",
    "select g, avg(v) from e group by g order by g",
    "select sum(v) over (order by g, v rows between unbounded preceding and current row) from e order by g, v",
]


@pytest.mark.parametrize("sql", EDGES)
def test_totals_at_the_edge_of_int64(mx, edges, sql):
    check(mx, sql, edges["con"], edges["files"], ordered=True)


@pytest.mark.parametrize("extra", WAYS, ids=lambda e: " ".join(e))
def test_the_edges_in_every_way_of_running(mx, edges, extra):
    check(mx, "select g, sum(v), avg(v) from e group by g order by g", edges["con"], edges["files"], ordered=True, extra=extra)


def test_csv_prints_the_digits(mx, world):
    out = subprocess.run([str(mx), "sql", "select g, sum(d), sum(b) from t where g = 1 group by g", "-t", f"t={world['files']['t']}", "--format", "csv"],
                         capture_output=True, text=True)
    want = world["con"].execute("select g, sum(d), sum(b) from t where g = 1 group by g").fetchone()
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines() == ["g,sum(d),sum(b)", f"{want[0]},{want[1]},{want[2]}"]


def test_a_parquet_file_cannot_hold_the_total_yet_and_says_so(mx, world, tmp_path):
    out = subprocess.run([str(mx), "sql", "select g, sum(d) from t group by g", "-t", f"t={world['files']['t']}", "--format", "parquet", "--out", str(tmp_path / "o.parquet")],
                         capture_output=True, text=True)
    assert out.returncode != 0
    assert "holds totals past int64" in out.stdout + out.stderr, out.stdout + out.stderr
    assert not (tmp_path / "o.parquet").exists()


REFUSED = [
    # a total past int64 inside a step that still reads int64s says so with the int64 message (DuckDB answers these; they are not built yet)
    "select s from (select g, sum(d) s from t group by g) x where s > 1",
    "select g, sum(sum(d)) over () from t group by g",
    "select o, sum(d) over (partition by g) + 1 from t",
    "select o, sum(d) over (partition by g) > 0 from t",
    "select g, sum(d) from t group by g union all select g, sum(d) from t group by g",
    "select g, case when g = 1 then sum(d) else 0 end from t group by g",
    "select b * 4 from t",
]


@pytest.mark.parametrize("sql", REFUSED)
def test_what_cannot_hold_a_total_yet_raises_the_int64_error(mx, world, sql):
    out = subprocess.run([str(mx), "sql", sql, "-t", f"t={world['files']['t']}"], capture_output=True, text=True)
    assert out.returncode == 1
    assert "a sum or an expression passed the int64 range (about 9.2e18)" in out.stdout, out.stdout


def test_a_constant_past_int64_in_a_row_filter_says_what_it_can_do(mx, world):
    """DuckDB keeps `d > 1e20` as a DECIMAL(38) comparison; here such a number is only read after aggregation (HAVING, the select list, ORDER BY)."""
    out = subprocess.run([str(mx), "sql", "select o from t where d > 100000000000000000000.00", "-t", f"t={world['files']['t']}"], capture_output=True, text=True)
    assert out.returncode == 1
    assert "a number past the int64 range can only be compared with or combined with totals after aggregation" in out.stdout, out.stdout
    assert "100000000000000000000.00" in out.stdout, out.stdout


def test_the_totals_really_pass_int64(world):
    """The data is what the tests say it is: the global sums and most groups pass int64 in DuckDB's own answer."""
    con = world["con"]
    big = 2**63
    assert abs(con.execute("select sum(d) * 100 from t").fetchone()[0]) > big or abs(con.execute("select sum(b) from t").fetchone()[0]) > big
    rows = con.execute("select g, sum(d) * 100, sum(b) from t group by g").fetchall()
    assert sum(1 for r in rows if r[1] is not None and abs(r[1]) > big) >= 3
    assert sum(1 for r in rows if r[2] is not None and abs(r[2]) > big) >= 3
    assert any(r[1] is not None and abs(r[1]) < big for r in rows)            # the small group


def test_an_expression_that_really_overflows_still_raises(mx, world):
    """Only a total gets 128 bits: one row's value past int64 is still an error (DuckDB's BIGINT arithmetic raises too)."""
    run_mx(mx, ["sql", "select b * 4 from t"], world["files"], expect_exit=1)
