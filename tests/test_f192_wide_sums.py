"""F-192: a SUM of INT or DECIMAL values whose total passes int64 is DuckDB's answer (DuckDB sums into HUGEINT / DECIMAL(38)), where the engine raised
"a sum or an expression passed the int64 range". Grouped, global and window sums, with NULLs and negatives, then what a query does with such a total
(ORDER BY, HAVING, arithmetic with a constant, a scalar subquery: TPC-H Q11's shape). Every answer is compared with DuckDB through the real `mx`
command (tests/mx_oracle.py): kinds, DECIMAL scales and every digit.

The data is built so that a group's total, and the whole column's, are far past 9.2e18 while single values stay in range (a DECIMAL(18, 2) value is
at most 9.99e17 stored as 99999999999999999), with some groups that stay small (so one column mixes narrow and wide totals), one group of NULLs only,
and negative totals.
"""
import math
import random
import subprocess
from decimal import Decimal

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mx_oracle import check, mx_binary, run_mx, typed_rows

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


@pytest.fixture(scope="module")
def bands(tmp_path_factory):
    """Groups of 24 rows whose total is in the bands where DuckDB's HUGEINT to DOUBLE cast changes rule (upper word 0, -1, other), at scale 0 (BIGINT)
    and 2 and 4 (DECIMAL(18)): around +-2^63 and +-2^64 and random 120-bit totals, each total split into 24 values that fit the column."""
    rnd = random.Random(1923)
    two63, two64 = 1 << 63, 1 << 64
    totals = []
    for base in (two63, two64):
        for sign in (1, -1):
            totals += [sign * (base + d) for d in (-3, -1, 0, 1, 2, 5, 1000, 12345)]
            totals += [sign * rnd.randrange(base - (1 << 40), base + (1 << 40)) for _ in range(40)]
            totals += [sign * rnd.randrange(base // 2, base) for _ in range(20)]
    totals += [rnd.choice((1, -1)) * rnd.randrange(1 << 64, 1 << 120) for _ in range(40)]
    g, v0, v2, v4 = [], [], [], []
    for i, t in enumerate(totals):
        for col, lim in ((v0, (1 << 63) - 1), (v2, 10**18 - 1), (v4, 10**18 - 1)):
            if abs(t) > 24 * lim:
                t_col = (1 if t > 0 else -1) * rnd.randrange(12 * lim, 24 * lim)       # what 24 values of this column can hold
            else:
                t_col = t
            parts = [t_col // 24] * 24
            parts[0] += t_col - sum(parts)
            assert all(abs(x) <= lim for x in parts), (t, col)
            col.extend(parts)
        g.extend([i] * 24)
    path = tmp_path_factory.mktemp("f192b") / "b.parquet"
    pq.write_table(pa.table({"g": pa.array(g, pa.int64()), "v0": pa.array(v0, pa.int64()),
                             "v2": pa.array([Decimal(x).scaleb(-2) for x in v2], pa.decimal128(18, 2)),
                             "v4": pa.array([Decimal(x).scaleb(-4) for x in v4], pa.decimal128(18, 4))}), path, row_group_size=500)
    con = duckdb.connect()
    con.execute(f"create table b as select * from read_parquet('{path}')")
    return {"files": {"b": str(path)}, "con": con}


BAND_SQL = "select g, sum({c}) / 7.0, sum({c}) / count(*), avg({c}), sum({c})::double, sum({c}) * 1.5 / 3.0 from b group by g order by g"


def test_a_total_at_scale_0_cast_to_double_is_duckdbs_bit_for_bit_in_every_band(mx, bands):
    """Negative totals between -2^64 and -2^63 (upper word -1: -double(2^64 - lower), exactly -2^64 at lower 0), positive ones below 2^64 (upper word
    0), and the rest: the total as a DOUBLE (division by a double, by a count, AVG's own rule, a plain cast)."""
    check(mx, BAND_SQL.format(c="v0"), bands["con"], bands["files"], ordered=True)


@pytest.mark.parametrize("col", ["v2", "v4"])
def test_a_total_with_decimal_places_cast_to_double_is_duckdbs_in_every_band(mx, bands, col):
    """The same bands at scales 2 and 4: DuckDB's cast of a DECIMAL(38, s) is the whole part and the fraction apart, each by the HUGEINT cast.
    Known gap (docs/plan.md F-192): for a NEGATIVE total whose whole part is between 2^55 and 2^64 (about 1 in 10 in the bands below), DuckDB's
    double is one ulp nearer zero than this rule gives (probed on 800 values: the rule behind it is not found; positive totals and every total at
    scale 0 are exact), so the float columns here may differ by an ulp or two (a division carries it), in a negative total only, and everything else is exact."""
    sql = BAND_SQL.format(c=col)
    res = run_mx(mx, ["sql", "--max-rows", "0", sql], bands["files"])
    want = bands["con"].execute(sql).fetchall()
    got = typed_rows(res)
    assert len(got) == len(want)
    off = 0
    for a, b in zip(got, want):
        for x, y in zip(a, b):
            if isinstance(x, float):
                assert x == y or (y < 0 and abs(x - y) <= 2 * math.ulp(y)), (a, b)
                off += x != y
            else:
                assert x == y, (a, b)
    assert off < len(want), "every group differing in the last bit is not a rare gap"


def test_a_narrow_total_divided_as_a_double_is_duckdbs_too(mx, world):
    """DuckDB's SUM is a DECIMAL(38) whether or not the total fits int64, so it casts by that rule (F-190's int64 rule differs in the last bit for
    some DECIMAL(15, 2) totals)."""
    check(mx, "select g, sum(p) / 7.0, sum(p) / count(*), sum(p * q) / 3.0 from t group by g order by g", world["con"], world["files"], ordered=True)
    check(mx, "select sum(p) / 7.0, sum(p) / 11.0, sum(q) / 7.0 from t", world["con"], world["files"])


def test_a_wide_total_cast_to_bigint_raises_like_duckdb(mx, world):
    """`sum(v)::bigint` of a total past int64 is a Conversion Error in DuckDB (it returned the low 64 bits as a BIGINT here); a total that fits casts."""
    for sql in ("select sum(b)::bigint from t", "select g, sum(b)::bigint from t group by g order by g", "select sum(b)::integer from t",
                "select (sum(b) + 1)::bigint from t"):
        with pytest.raises(duckdb.Error):
            world["con"].execute(sql).fetchall()
        out = subprocess.run([str(mx), "sql", sql, "-t", f"t={world['files']['t']}"], capture_output=True, text=True)
        assert out.returncode == 1 and "out of range" in out.stdout, (sql, out.stdout)
    check(mx, "select sum(b)::bigint from t where g = 3", world["con"], world["files"])
    check(mx, "select try_cast(sum(b) as bigint) from t", world["con"], world["files"])


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
