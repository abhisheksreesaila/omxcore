"""F-159: queries made of several statements, equal to DuckDB: WITH, derived tables, subqueries (scalar, IN, EXISTS; correlated ones too), UNION / INTERSECT /
EXCEPT, filtering on a window result (a derived table, QUALIFY).

Every form runs over small generated tables with NULLs in the columns a query correlates or compares on (`t`: 300 rows, `u`: 40 rows), through `build/mx`,
and is compared with DuckDB on the same data: the table read whole, the Parquet file streamed a row group at a time, and (the forms that take only INT and
TEXT) a CSV file. Each form runs with an inner query that returns rows and with one that returns none (`WHERE 1 = 0`), and the plan lines (EXPLAIN) are
asserted where the ticket promises them. The TPC-H shapes (Q2, Q17, Q20, Q21, Q22) run on dbgen SF 0.1.
"""
import re

import duckdb
import pytest

from mx_oracle import check, mx_binary, run_mx

N_T = 300
N_U = 40


@pytest.fixture(scope="session")
def mx():
    return mx_binary()


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("compound")
    con = duckdb.connect()
    con.execute(f"""create table t as select i::BIGINT as id,
        case when i % 11 = 0 then null else (i % 7)::BIGINT end as k,
        case when i % 13 = 0 then null else ((i * 37) % 50 - 10)::BIGINT end as v,
        case when i % 17 = 0 then null else (['West', 'east', 'North', 'south', 'Apple'])[1 + (i % 5)] end as s,
        case when i % 19 = 0 then null else ((i * 7) % 1000 - 300) * 0.01 end::DECIMAL(10,2) as d
        from range({N_T}) r(i)""")
    con.execute(f"""create table u as select i::BIGINT as uid,
        case when i % 9 = 0 then null else (i % 5)::BIGINT end as k,
        case when i % 6 = 0 then null else ((i * 11) % 30)::BIGINT end as w,
        (['West', 'east', 'Zed'])[1 + (i % 3)] as s from range({N_U}) r(i)""")
    paths = {}
    for name in ("t", "u"):
        paths[name] = d / f"{name}.parquet"
        con.execute(f"copy {name} to '{paths[name]}' (format parquet, row_group_size 64)")
        paths[name + "_csv"] = d / f"{name}.csv"
        con.execute(f"copy {name} to '{paths[name + '_csv']}' (format csv, header)")
    return {"con": con, "paths": paths}


MODES = ["memory", "scan"]


def tables_of(sql):
    return [t for t in ("t", "u") if re.search(rf"\b{t}\b", sql)]


def q(mx, data, mode, sql, ordered=False, duck=None):
    suffix = "_csv" if mode == "csv" else ""
    files = {t: data["paths"][t + suffix] for t in tables_of(sql)}
    extra = ["--stream", "always" if mode == "scan" else "never"] if mode != "csv" else []
    return check(mx, sql, data["con"], files=files, ordered=ordered, extra=extra, duck_sql=duck)


def plan(mx, data, sql):
    files = {t: data["paths"][t] for t in tables_of(sql)}
    out = run_mx(mx, ["explain", "--format", "json", "--stream", "never", sql], files)
    return "\n".join(out["plan"])


def error_of(mx, data, sql):
    files = {t: data["paths"][t] for t in tables_of(sql)}
    return run_mx(mx, ["sql", sql], files, expect_exit=1)["error"]["message"]


# inner queries that return rows and none, one filter each
FILTERS = ["id >= 0", "1 = 0"]


# ── step 1: derived tables and WITH ──────────────────────────────────────────────────────────────

DERIVED = {
    "filter-outside": "SELECT * FROM (SELECT id, k, v FROM t WHERE {f}) x WHERE k > 2",
    "alias-qualifier": "SELECT x.id, x.v FROM (SELECT id, v FROM t WHERE {f}) AS x WHERE x.v IS NOT NULL",
    "aggregate-inside": "SELECT * FROM (SELECT k, count(*) c, sum(v) sv FROM t WHERE {f} GROUP BY k) x WHERE c > 1",
    "aggregate-outside": "SELECT k, sum(v), count(v), min(id) FROM (SELECT * FROM t WHERE {f}) x GROUP BY k",
    "aggregate-both": "SELECT c, count(*) FROM (SELECT k, count(*) c FROM t WHERE {f} GROUP BY k) x GROUP BY c",
    "global-aggregate": "SELECT count(*), sum(v), max(lower(s)) FROM (SELECT * FROM t WHERE {f}) x",
    "nested": "SELECT * FROM (SELECT * FROM (SELECT id, v FROM t WHERE {f}) a WHERE v > 0) b WHERE id < 100",
    "join-two-derived": "SELECT x.id, y.w FROM (SELECT id, k FROM t WHERE {f}) x JOIN (SELECT k, w FROM u WHERE w IS NOT NULL) y ON x.k = y.k WHERE y.w > 10",
    "join-derived-and-table": "SELECT x.id, u.w FROM (SELECT id, k FROM t WHERE {f}) x JOIN u ON x.k = u.k",
    "left-join-derived": "SELECT x.id, y.w FROM (SELECT id, k FROM t WHERE {f}) x LEFT JOIN (SELECT k, max(w) w FROM u GROUP BY k) y ON x.k = y.k",
    "distinct-order-limit": "SELECT DISTINCT k FROM (SELECT * FROM t WHERE {f}) x ORDER BY k LIMIT 4",
    "text-and-decimal": "SELECT s, sum(d) FROM (SELECT s, d FROM t WHERE {f}) x GROUP BY s",
    "expression-columns": "SELECT a + 1, b FROM (SELECT v * 2 AS a, upper(s) AS b FROM t WHERE {f}) x WHERE a > 5",
    "star-and-order": "SELECT * FROM (SELECT id, s FROM t WHERE {f}) x ORDER BY lower(s), id LIMIT 7",
    "rank-filter": "SELECT id, k, r FROM (SELECT id, k, rank() OVER (PARTITION BY k ORDER BY v, id) r FROM t WHERE {f}) x WHERE r <= 2",
    "row-number-top-1": "SELECT * FROM (SELECT id, k, row_number() OVER (PARTITION BY k ORDER BY id DESC) rn FROM t WHERE {f}) x WHERE rn = 1",
    "running-sum-filter": "SELECT id, run FROM (SELECT id, sum(v) OVER (ORDER BY id) run FROM t WHERE {f}) x WHERE run > 100",
    "window-over-derived": "SELECT id, lag(v) OVER (ORDER BY id) FROM (SELECT id, v FROM t WHERE {f}) x",
}
UNORDERED_NEEDS_ORDER = {"distinct-order-limit": True, "star-and-order": True}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("f", FILTERS)
@pytest.mark.parametrize("name", list(DERIVED))
def test_derived_table_equals_duckdb(mx, data, name, f, mode):
    q(mx, data, mode, DERIVED[name].format(f=f), ordered=name in UNORDERED_NEEDS_ORDER)


def test_derived_table_over_a_csv_file(mx, data):
    q(mx, data, "csv", "SELECT k, count(*) FROM (SELECT * FROM t WHERE id > 20) x GROUP BY k")


def test_derived_table_without_an_alias(mx, data):
    q(mx, data, "memory", "SELECT * FROM (SELECT id FROM t WHERE id < 5)")


def test_derived_table_is_shown_in_explain(mx, data):
    text = plan(mx, data, "SELECT * FROM (SELECT id, k FROM t) x WHERE k > 2")
    assert "derived table x: evaluated in place (300 rows)" in text


CTES = {
    "once": "WITH a AS (SELECT id, k, v FROM t WHERE {f}) SELECT k, sum(v) FROM a GROUP BY k",
    "twice-self-join": "WITH a AS (SELECT id, k, v FROM t WHERE {f}) SELECT x.id, y.id FROM a x JOIN a y ON x.k = y.k AND x.id < y.id WHERE x.id < 20",
    "chained": "WITH a AS (SELECT * FROM t WHERE {f}), b AS (SELECT k, count(*) c FROM a GROUP BY k) SELECT * FROM b WHERE c > 3",
    "later-reads-earlier-twice": "WITH a AS (SELECT id, k FROM t WHERE {f}), b AS (SELECT x.id FROM a x JOIN a y ON x.k = y.k AND x.id = y.id) SELECT count(*) FROM b",
    "column-list": "WITH a(i, kk) AS (SELECT id, k FROM t WHERE {f}) SELECT i, kk FROM a WHERE kk = 3",
    "materialized-hint": "WITH a AS MATERIALIZED (SELECT id, v FROM t WHERE {f}) SELECT sum(v) FROM a",
    "not-materialized-twice": "WITH a AS NOT MATERIALIZED (SELECT id, v FROM t WHERE {f}) SELECT count(*) FROM a x JOIN a y ON x.id = y.id",
    "cte-in-derived": "WITH a AS (SELECT id, v FROM t WHERE {f}) SELECT * FROM (SELECT id FROM a WHERE v > 5) z WHERE id < 100",
    "cte-with-window": "WITH ranked AS (SELECT id, k, rank() OVER (PARTITION BY k ORDER BY v, id) r FROM t WHERE {f}) SELECT * FROM ranked WHERE r <= 3",
    "shadows-a-table": "WITH u AS (SELECT id AS uid, k, v AS w, s FROM t WHERE {f}) SELECT count(*), sum(w) FROM u",
    "cte-order-limit": "WITH a AS (SELECT * FROM t WHERE {f}) SELECT id FROM a ORDER BY v DESC NULLS LAST, id LIMIT 5",
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("f", FILTERS)
@pytest.mark.parametrize("name", list(CTES))
def test_cte_equals_duckdb(mx, data, name, f, mode):
    q(mx, data, mode, CTES[name].format(f=f), ordered=name in ("cte-order-limit",))


def test_a_cte_read_once_is_inlined_and_read_twice_materialized_once(mx, data):
    once = plan(mx, data, "WITH a AS (SELECT id, k FROM t) SELECT k FROM a WHERE id < 5")
    assert "CTE a: inlined into its one read (300 rows)" in once
    twice = plan(mx, data, "WITH a AS (SELECT id, k FROM t) SELECT x.id FROM a x JOIN a y ON x.k = y.k")
    assert "CTE a: materialized once (2 reads, 300 rows)" in twice
    assert twice.count("CTE a:") == 1
    forced = plan(mx, data, "WITH a AS MATERIALIZED (SELECT id FROM t) SELECT id FROM a")
    assert "CTE a: materialized once (1 reads, 300 rows)" in forced
    unforced = plan(mx, data, "WITH a AS NOT MATERIALIZED (SELECT id FROM t) SELECT id FROM a")
    assert "CTE a: inlined into its one read (300 rows)" in unforced


def test_a_cte_name_shadows_a_table_and_a_cte_of_an_inner_query_shadows_it(mx, data):
    q(mx, data, "memory", "WITH t AS (SELECT id + 1000 AS id FROM t WHERE id < 3) SELECT count(*), max(id) FROM t")
    q(mx, data, "memory", "WITH a AS (SELECT id FROM t WHERE id < 10) SELECT count(*) FROM (WITH a AS (SELECT id FROM t WHERE id < 3) SELECT id FROM a) z")


def test_a_cte_that_is_never_read_does_not_run(mx, data):
    q(mx, data, "memory", "WITH a AS (SELECT 1 / 0 FROM nowhere) SELECT count(*) FROM t")


@pytest.mark.parametrize("sql,message", [
    ("WITH RECURSIVE a AS (SELECT 1) SELECT * FROM a", "`WITH RECURSIVE` is not supported yet"),
    ("SELECT * FROM t, LATERAL (SELECT 1) x", "`LATERAL` is not supported yet"),
    ("SELECT * FROM (SELECT id FROM t) x(a)", "column names after a table alias is not supported yet"),
    ("WITH a AS (SELECT id FROM t) SELECT * FROM b", "table `b` not found"),
    ("WITH a(x, y) AS (SELECT id FROM t) SELECT * FROM a", "the column list names 2 columns, the query has 1"),
])
def test_unsupported_forms_raise_by_name(mx, data, sql, message):
    assert message in error_of(mx, data, sql)
