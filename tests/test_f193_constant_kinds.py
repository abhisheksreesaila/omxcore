"""F-193: a constant in the select list keeps its own kind whatever else is in the list.

Before: `SELECT CAST(NULL AS DATE) AS m, CAST(NULL AS BIGINT) AS q` returned q as DATE. The row collector shares one column
between select items with the same expression key, and the key of a typed NULL (and of 1 against CAST(1 AS BIGINT), or
2.50 against 2.5) did not hold the type, so the second item took the first one's column.

Every list is run through the `mx` binary and through the Mojo API (the SQL probe: a table read whole and a Parquet file
streamed), and compared with DuckDB: names, kinds, scales and values.
"""
import random

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mx_oracle import assert_equal, check, duck_kind, mx_binary
from test_mojo_sql import got_rows, probe, run_probe  # noqa: F401  (probe is a fixture)

NULLS = [f"CAST(NULL AS {t})" for t in ("BIGINT", "INTEGER", "DOUBLE", "VARCHAR", "BOOLEAN", "DATE", "DECIMAL(9,2)", "DECIMAL(18,4)")]
VALUES = ["1", "CAST(1 AS BIGINT)", "3000000000", "1.5", "CAST(2.5 AS DECIMAL(9,3))", "CAST(2.50 AS DECIMAL(9,2))", "'a'", "true",
          "DATE '2020-01-02'", "CAST(7 AS DOUBLE)", "1 + 1"]
COLUMNS = ["id", "k", "d", "dt", "s", "b"]


@pytest.fixture(scope="module")
def mx():
    return mx_binary()


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    c.execute("""create table t as select i::bigint as id, (i % 3)::bigint as k, (i * 1.25)::decimal(9,2) as d,
        date '2020-01-01' + i::int as dt, case when i % 4 = 0 then null else 'w' || (i % 3)::varchar end as s, i % 2 = 0 as b
        from range(0, 6) r(i)""")
    return c


@pytest.fixture(scope="module")
def path(tmp_path_factory, con):
    p = tmp_path_factory.mktemp("f193") / "t.parquet"
    pq.write_table(con.execute("select * from t").to_arrow_table(), p, row_group_size=2)
    return p


def probe_check(probe, con, q, path, mode):
    res = run_probe(probe, q, {"t": path}, mode=mode)
    cur = con.execute(q)
    for c, desc in zip(res["columns"], cur.description):
        dk = duck_kind(desc[1])
        assert (c["kind"], c["scale"] if dk[0] == "DECIMAL" else 0) == dk, f"`{q}`: got {c}, DuckDB has {dk}"
    assert_equal(got_rows(res), cur.fetchall(), False, q)


def test_the_ticket_example(mx, con):
    res = check(mx, "SELECT CAST(NULL AS DATE) AS m, CAST(NULL AS BIGINT) AS q", con)
    assert [c["kind"] for c in res["columns"]] == ["DATE", "INT"]


def test_the_ticket_example_next_to_a_column(mx, con, path):
    check(mx, "SELECT id, CAST(NULL AS DATE) AS m, CAST(NULL AS BIGINT) AS q FROM t", con, files={"t": path})


def test_equal_looking_constants_of_two_types(mx, con):
    res = check(mx, "SELECT 1 AS a, CAST(1 AS BIGINT) AS b, CAST(2.5 AS DECIMAL(9,3)) AS c, CAST(2.50 AS DECIMAL(9,2)) AS d", con)
    assert [(c["kind"], c["scale"]) for c in res["columns"]] == [("INT", 0), ("INT", 0), ("DECIMAL", 3), ("DECIMAL", 2)]


def _lists(seed, n):
    rng = random.Random(seed)
    pool = NULLS + VALUES + COLUMNS
    for _ in range(n):
        items = [rng.choice(pool) for _ in range(rng.randint(2, 5))]
        if not any(i in COLUMNS for i in items):
            yield "SELECT " + ", ".join(f"{e} AS c{k}" for k, e in enumerate(items))      # constants only: no FROM
        yield "SELECT " + ", ".join(f"{e} AS c{k}" for k, e in enumerate(items)) + " FROM t"


QUERIES = list(_lists(193, 120))


@pytest.mark.parametrize("q", QUERIES)
def test_constant_lists_through_mx(mx, con, path, q):
    check(mx, q, con, files={"t": path})


@pytest.mark.parametrize("mode", ["table", "scan"])
@pytest.mark.parametrize("q", QUERIES[:60])
def test_constant_lists_through_the_mojo_api(probe, con, path, mode, q):    # noqa: F811
    if " FROM t" not in q:
        pytest.skip("the probe reads a table")
    probe_check(probe, con, q, path, mode)


AGGREGATES = [
    "SELECT count(*) AS n, CAST(NULL AS DATE) AS m, CAST(NULL AS BIGINT) AS q FROM t",
    "SELECT k, sum(id) AS s, CAST(NULL AS DATE) AS m, CAST(NULL AS BIGINT) AS q, 1 AS a, CAST(1 AS BIGINT) AS b FROM t GROUP BY k",
    "SELECT k, CAST(NULL AS VARCHAR) AS m, CAST(NULL AS DECIMAL(9,2)) AS q, CAST(2.5 AS DECIMAL(9,3)) AS c, CAST(2.50 AS DECIMAL(9,2)) AS e "
    "FROM t GROUP BY k",
]


@pytest.mark.parametrize("q", AGGREGATES)
def test_constants_next_to_aggregates(mx, con, path, q):
    check(mx, q, con, files={"t": path})
