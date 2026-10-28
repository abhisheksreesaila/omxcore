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
    con.execute(f"""create table u as select i::BIGINT as uid, i::BIGINT as id,
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


def test_derived_table_with_column_names(mx, data):
    q(mx, data, "memory", "SELECT c_k, n FROM (SELECT k, count(*) FROM t GROUP BY k) AS c (c_k, n) WHERE n > 20")
    q(mx, data, "memory", "SELECT n, count(*) FROM (SELECT k, count(v) FROM t GROUP BY k) c (kk, n) GROUP BY n")
    q(mx, data, "memory", "SELECT a.x, b.y FROM (SELECT id, k FROM t) a (id2, x) JOIN (SELECT uid, k FROM u) b (uid2, y) ON a.x = b.y")


def test_derived_table_without_an_alias(mx, data):
    q(mx, data, "memory", "SELECT * FROM (SELECT id FROM t WHERE id < 5)")


def test_derived_table_is_shown_in_explain(mx, data):
    text = plan(mx, data, "SELECT * FROM (SELECT id, k FROM t) x WHERE k > 2")
    assert "derived table x: inlined into the query that reads it" in text
    text = plan(mx, data, "SELECT * FROM (SELECT k, count(*) c FROM t GROUP BY k) x WHERE c > 5")
    assert "derived table x: evaluated in place (8 rows)" in text


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
    assert "CTE a: inlined into the query that reads it (its filter and columns merged, nothing materialized)" in once
    assert "CTE a: materialized" not in once and "source: t (table), 300 rows" in once
    not_merged = plan(mx, data, "WITH a AS (SELECT k, count(*) c FROM t GROUP BY k) SELECT k FROM a WHERE c > 5")
    assert "CTE a: inlined into its one read (8 rows)" in not_merged
    twice = plan(mx, data, "WITH a AS (SELECT id, k FROM t) SELECT x.id FROM a x JOIN a y ON x.k = y.k")
    assert "CTE a: materialized once (2 reads, 300 rows)" in twice
    assert twice.count("CTE a:") == 1
    forced = plan(mx, data, "WITH a AS MATERIALIZED (SELECT id FROM t) SELECT id FROM a")
    assert "CTE a: materialized once (1 reads, 300 rows)" in forced
    unforced = plan(mx, data, "WITH a AS NOT MATERIALIZED (SELECT id FROM t) SELECT id FROM a")
    assert "CTE a: inlined into the query that reads it" in unforced


def test_a_cte_name_shadows_a_table_and_a_cte_of_an_inner_query_shadows_it(mx, data):
    q(mx, data, "memory", "WITH t AS (SELECT id + 1000 AS id FROM t WHERE id < 3) SELECT count(*), max(id) FROM t")
    q(mx, data, "memory", "WITH a AS (SELECT id FROM t WHERE id < 10) SELECT count(*) FROM (WITH a AS (SELECT id FROM t WHERE id < 3) SELECT id FROM a) z")


def test_a_cte_that_is_never_read_does_not_run(mx, data):
    q(mx, data, "memory", "WITH a AS (SELECT 1 / 0 FROM nowhere) SELECT count(*) FROM t")


@pytest.mark.parametrize("sql,message", [
    ("WITH RECURSIVE a AS (SELECT 1) SELECT * FROM a", "`WITH RECURSIVE` is not supported yet"),
    ("SELECT * FROM t, LATERAL (SELECT 1) x", "`LATERAL` is not supported yet"),
    ("SELECT * FROM (SELECT id FROM t) x(a, b)", "the column list names 2 columns, the query has 1"),
    ("SELECT * FROM (SELECT id FROM t) (a)", "expected a table alias before the column names"),
    ("WITH a AS (SELECT id FROM t) SELECT * FROM b", "table `b` not found"),
    ("WITH a(x, y) AS (SELECT id FROM t) SELECT * FROM a", "the column list names 2 columns, the query has 1"),
])
def test_unsupported_forms_raise_by_name(mx, data, sql, message):
    assert message in error_of(mx, data, sql)


# ── step 2: subqueries ───────────────────────────────────────────────────────────────────────────

SUBQUERIES = {
    # scalar, uncorrelated (an empty inner is NULL)
    "scalar-compare": "SELECT id FROM t WHERE v > (SELECT avg(w) FROM u WHERE {f})",
    "scalar-equal-int": "SELECT id FROM t WHERE k = (SELECT min(k) FROM u WHERE {f})",
    "scalar-in-select-list": "SELECT id, (SELECT max(w) FROM u WHERE {f}) AS m FROM t WHERE id < 5",
    "scalar-text": "SELECT id FROM t WHERE lower(s) = (SELECT max(lower(s)) FROM u WHERE {f})",
    "scalar-decimal": "SELECT id FROM t WHERE d > (SELECT max(d) FROM t WHERE {f} AND id < 100)",
    "scalar-in-having": "SELECT k, count(*) FROM t GROUP BY k HAVING count(*) > (SELECT count(*) FROM u WHERE {f}) / 10",
    "scalar-in-arithmetic": "SELECT id FROM t WHERE v * 2 < 3 * (SELECT sum(w) FROM u WHERE {f}) / 40",
    "scalar-no-rows-is-null": "SELECT id, coalesce((SELECT max(w) FROM u WHERE {f} AND w < 0), -1) FROM t WHERE id < 3",
    "scalar-over-a-derived-table": "SELECT id FROM t WHERE v = (SELECT max(w) FROM (SELECT w FROM u WHERE {f}) x)",
    # IN, uncorrelated
    "in": "SELECT id FROM t WHERE k IN (SELECT k FROM u WHERE {f})",
    "not-in-with-nulls": "SELECT id FROM t WHERE k NOT IN (SELECT k FROM u WHERE {f})",
    "not-in-without-nulls": "SELECT id FROM t WHERE k NOT IN (SELECT k FROM u WHERE {f} AND k IS NOT NULL)",
    "in-or": "SELECT id FROM t WHERE k IN (SELECT k FROM u WHERE {f}) OR id < 3",
    "in-group-by-having": "SELECT id FROM t WHERE k IN (SELECT k FROM u WHERE {f} GROUP BY k HAVING count(*) > 5)",
    "in-beside-a-join": "SELECT t.id, u.uid FROM t JOIN u ON t.k = u.k WHERE t.v IN (SELECT w FROM u WHERE {f})",
    "in-expression-on-the-left": "SELECT id FROM t WHERE id + 0 IN (SELECT uid FROM u WHERE {f})",
    "not-in-expression-on-the-left": "SELECT id FROM t WHERE id * 2 NOT IN (SELECT uid FROM u WHERE {f} AND uid IS NOT NULL)",
    "in-two-subqueries": "SELECT id FROM t WHERE k IN (SELECT k FROM u WHERE {f}) AND v NOT IN (SELECT w FROM u WHERE w IS NOT NULL AND {f})",
    "in-derived-inner": "SELECT id FROM t WHERE k IN (SELECT k FROM (SELECT k, count(*) c FROM u WHERE {f} GROUP BY k) x WHERE c > 2)",
    # EXISTS
    "exists": "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE {f}) AND id < 5",
    "not-exists": "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE {f}) AND id < 5",
    "exists-correlated": "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND {f})",
    "not-exists-correlated": "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND {f})",
    "exists-correlated-filter": "SELECT id FROM t WHERE EXISTS (SELECT * FROM u WHERE u.k = t.k AND u.w > 10 AND {f})",
    "exists-correlated-expression": "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k + 1 = t.k AND {f})",
    "exists-two-keys": "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.w = t.v AND {f})",
    "exists-not-equal": "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.w <> t.v AND {f})",
    "not-exists-not-equal": "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.w <> t.v AND {f})",
    "exists-less": "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.w < t.v AND {f})",
    "exists-greater-equal-flipped": "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE t.v >= u.w AND u.k = t.k AND {f})",
    "not-exists-greater": "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.w > t.v AND {f})",
    "exists-only-inequality": "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.w > t.v AND {f})",
    "not-exists-only-inequality": "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.w > t.v AND {f})",
    "exists-and-not-exists": "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.w <> t.v AND {f}) AND NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.w > 20)",
    "exists-beside-a-join": "SELECT a.id, b.uid FROM t a JOIN u b ON a.k = b.k WHERE EXISTS (SELECT 1 FROM t c WHERE c.k = a.k AND c.id <> a.id AND {f})",
    "exists-with-alias-and-plain-filter": "SELECT o.id FROM t AS o WHERE o.v > 0 AND NOT EXISTS (SELECT 1 FROM u WHERE u.k = o.k AND {f}) AND o.id < 150",
    "exists-inner-join": "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u JOIN t t2 ON u.k = t2.k WHERE u.k = t.k AND u.w > 5 AND {fq})",
    "exists-outer-only-condition": "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE t.v > 3 AND {f})",
    "in-correlated": "SELECT id FROM t WHERE v IN (SELECT w FROM u WHERE u.k = t.k AND {f})",
    "in-correlated-two-keys": "SELECT id FROM t WHERE v IN (SELECT w FROM u WHERE u.k = t.k AND u.s = t.s AND {f})",
    # scalar, correlated
    "scalar-correlated-avg": "SELECT id FROM t WHERE v > (SELECT avg(w) FROM u WHERE u.k = t.k AND {f})",
    "scalar-correlated-min": "SELECT id FROM t WHERE v = (SELECT min(w) FROM u WHERE u.k = t.k AND {f})",
    "scalar-correlated-sum-flipped": "SELECT id FROM t WHERE (SELECT sum(w) FROM u WHERE u.k = t.k AND {f}) > v",
    "scalar-correlated-count-zero": "SELECT id FROM t WHERE (SELECT count(*) FROM u WHERE u.k = t.k AND {f}) = 0",
    "scalar-correlated-count-positive": "SELECT id FROM t WHERE 2 <= (SELECT count(w) FROM u WHERE u.k = t.k AND {f})",
    "scalar-correlated-arithmetic": "SELECT id FROM t WHERE v * 2 < (SELECT 0.5 * avg(w) FROM u WHERE u.k = t.k AND {f})",
    "scalar-correlated-two-keys": "SELECT id FROM t WHERE v < (SELECT max(w) FROM u WHERE u.k = t.k AND u.s = t.s AND {f})",
    "scalar-correlated-isnull": "SELECT id FROM t WHERE (SELECT max(w) FROM u WHERE u.k = t.k AND {f}) IS NULL AND id < 30",
    "scalar-correlated-beside-a-join": "SELECT a.id, b.uid FROM t a JOIN u b ON a.k = b.k WHERE a.v > (SELECT avg(w) FROM u WHERE u.k = a.k AND {f})",
    "scalar-correlated-aggregate-outside": "SELECT k, count(*) FROM t WHERE v > (SELECT avg(w) FROM u WHERE u.k = t.k AND {f}) GROUP BY k",
    "scalar-correlated-and-plain": "SELECT id FROM t WHERE id < 200 AND v >= (SELECT min(w) FROM u WHERE u.k = t.k AND {f}) AND s <> 'east'",
    "scalar-correlated-text-key": "SELECT id FROM t WHERE d > (SELECT min(d) FROM t t2 WHERE t2.s = t.s AND {f})",
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("f", FILTERS)
@pytest.mark.parametrize("name", list(SUBQUERIES))
def test_subquery_equals_duckdb(mx, data, name, f, mode):
    q(mx, data, mode, SUBQUERIES[name].format(f=f, fq=f.replace("id", "u.id")))


def test_a_subquery_over_a_csv_file(mx, data):
    q(mx, data, "csv", "SELECT id FROM t WHERE k IN (SELECT k FROM u WHERE w > 10) AND EXISTS (SELECT 1 FROM u WHERE u.k = t.k)")


def test_a_scalar_subquery_with_more_than_one_row_raises(mx, data):
    msg = error_of(mx, data, "SELECT id FROM t WHERE v = (SELECT w FROM u)")
    assert "the scalar subquery returned 40 rows: a scalar subquery must return at most one row" in msg


def test_a_scalar_subquery_with_two_columns_raises(mx, data):
    assert "must return one column, this one returns 2" in error_of(mx, data, "SELECT id FROM t WHERE v = (SELECT w, k FROM u LIMIT 1)")


@pytest.mark.parametrize("sql,message", [
    ("SELECT id FROM t WHERE v NOT IN (SELECT w FROM u WHERE u.k = t.k)", "a correlated NOT IN (SELECT ...) is not supported yet"),
    ("SELECT id, (SELECT max(w) FROM u WHERE u.k = t.k) FROM t", "a correlated subquery outside WHERE is not supported yet"),
    ("SELECT id FROM t WHERE id < 3 OR EXISTS (SELECT 1 FROM u WHERE u.k = t.k)", "is supported as a condition of its own"),
    ("SELECT id FROM t WHERE v > (SELECT w FROM u WHERE u.k = t.k)", "a correlated scalar subquery must be an aggregate"),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.w < t.v AND u.uid > t.id)", "more than one inequality"),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.w + t.v = 3)", "must compare an expression of the subquery's tables with one outer column"),
    ("SELECT id FROM t WHERE EXISTS (SELECT k FROM u WHERE u.k = t.k GROUP BY k)", "with GROUP BY or HAVING is not supported yet"),
    ("SELECT id FROM t WHERE 3 = (SELECT count(*) + sum(1) FROM u WHERE u.k = t.k)", "COUNT inside an expression"),
    ("SELECT id FROM t WHERE v < (SELECT max(w) / 0.0 FROM u)", "which has no SQL literal"),
])
def test_unsupported_subquery_forms_raise_by_name(mx, data, sql, message):
    assert message in error_of(mx, data, sql)


def test_the_plan_names_each_subquery_step(mx, data):
    text = plan(mx, data, "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.w <> t.v)")
    assert "subquery EXISTS:" in text and "min and max" in text
    text = plan(mx, data, "SELECT id FROM t WHERE k IN (SELECT k FROM u)")
    assert "subquery IN:" in text and "a semi join" in text
    text = plan(mx, data, "SELECT id FROM t WHERE v > (SELECT max(w) FROM u)")
    assert "scalar subquery: 1 row, a constant" in text


# ── the TPC-H shapes (dbgen SF 0.1) ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tpch(tmp_path_factory):
    d = tmp_path_factory.mktemp("tpch_compound")
    con = duckdb.connect()
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    con.execute("CALL dbgen(sf=0.1)")
    paths = {}
    for (t,) in con.execute("show tables").fetchall():
        paths[t] = d / f"{t}.parquet"
        con.execute(f"copy {t} to '{paths[t]}' (format parquet, row_group_size 30000)")
    return {"con": con, "paths": paths}


def tpch_q(mx, tpch, sql, ordered=False, extra=()):
    names = [t for t in ("customer", "orders", "lineitem", "supplier", "nation", "region", "part", "partsupp") if re.search(rf"\b{t}\b", sql)]
    return check(mx, sql, tpch["con"], files={t: tpch["paths"][t] for t in names}, ordered=ordered, extra=list(extra))


TPCH = {
    "Q2": ("""SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment
FROM part, supplier, partsupp, nation, region
WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND p_size = 15 AND p_type LIKE '%BRASS' AND s_nationkey = n_nationkey
  AND n_regionkey = r_regionkey AND r_name = 'EUROPE'
  AND ps_supplycost = (SELECT min(ps_supplycost) FROM partsupp, supplier, nation, region
                       WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'EUROPE')
ORDER BY s_acctbal DESC, n_name, s_name, p_partkey LIMIT 100""", True),
    "Q4": ("""SELECT o_orderpriority, count(*) AS order_count FROM orders
WHERE o_orderdate >= DATE '1993-07-01' AND o_orderdate < DATE '1993-07-01' + INTERVAL '3' MONTH
  AND EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate)
GROUP BY o_orderpriority ORDER BY o_orderpriority""", True),
    "Q17": ("""SELECT sum(l_extendedprice) / 7.0 AS avg_yearly FROM lineitem, part
WHERE p_partkey = l_partkey AND p_brand = 'Brand#23' AND p_container = 'MED BOX'
  AND l_quantity < (SELECT 0.2 * avg(l_quantity) FROM lineitem WHERE l_partkey = p_partkey)""", False),
    "Q18": ("""SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity) FROM customer, orders, lineitem
WHERE o_orderkey IN (SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING sum(l_quantity) > 300)
  AND c_custkey = o_custkey AND o_orderkey = l_orderkey
GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice ORDER BY o_totalprice DESC, o_orderdate LIMIT 100""", True),
    "Q20": ("""SELECT s_name, s_address FROM supplier, nation
WHERE s_suppkey IN (SELECT ps_suppkey FROM partsupp
                    WHERE ps_partkey IN (SELECT p_partkey FROM part WHERE p_name LIKE 'forest%')
                      AND ps_availqty > (SELECT 0.5 * sum(l_quantity) FROM lineitem
                                         WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey
                                           AND l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR))
  AND s_nationkey = n_nationkey AND n_name = 'CANADA' ORDER BY s_name""", True),
    "Q21": ("""SELECT s_name, count(*) AS numwait FROM supplier, lineitem l1, orders, nation
WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey AND o_orderstatus = 'F' AND l1.l_receiptdate > l1.l_commitdate
  AND EXISTS (SELECT * FROM lineitem l2 WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey)
  AND NOT EXISTS (SELECT * FROM lineitem l3 WHERE l3.l_orderkey = l1.l_orderkey AND l3.l_suppkey <> l1.l_suppkey AND l3.l_receiptdate > l3.l_commitdate)
  AND s_nationkey = n_nationkey AND n_name = 'SAUDI ARABIA'
GROUP BY s_name ORDER BY numwait DESC, s_name LIMIT 100""", True),
    "Q22": ("""SELECT cntrycode, count(*) AS numcust, sum(c_acctbal) AS totacctbal FROM (
  SELECT substring(c_phone FROM 1 FOR 2) AS cntrycode, c_acctbal FROM customer
  WHERE substring(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')
    AND c_acctbal > (SELECT avg(c_acctbal) FROM customer WHERE c_acctbal > 0.00 AND substring(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17'))
    AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)) AS custsale
GROUP BY cntrycode ORDER BY cntrycode""", True),
}


@pytest.mark.parametrize("stream", ["never", "always"])
@pytest.mark.parametrize("name", list(TPCH))
def test_tpch_shape_equals_duckdb(mx, tpch, name, stream):
    sql, ordered = TPCH[name]
    res = tpch_q(mx, tpch, sql, ordered=ordered, extra=("--stream", stream))
    assert len(res["rows"]) > 0 or name in ("Q17",)


# ── step 3: UNION ALL, UNION, INTERSECT, EXCEPT ──────────────────────────────────────────────────

SETOPS = {
    "union-all": "SELECT id, k FROM t WHERE {f} UNION ALL SELECT uid, k FROM u",
    "union-all-empty-right": "SELECT id, k FROM t WHERE {f} UNION ALL SELECT uid, k FROM u WHERE 1 = 0",
    "union-all-text-dictionaries": "SELECT s FROM t WHERE {f} UNION ALL SELECT s FROM u",
    "union-all-int-and-decimal": "SELECT v FROM t WHERE {f} UNION ALL SELECT d FROM t WHERE id < 20",
    "union-all-decimal-scales": "SELECT d FROM t WHERE {f} UNION ALL SELECT CAST(d AS DECIMAL(12,4)) FROM t WHERE id < 20",
    "union-all-int-and-double": "SELECT v FROM t WHERE {f} UNION ALL SELECT w / 4.0 FROM u",
    "union-all-columns-of-every-kind": "SELECT id, s, d, v IS NULL AS b FROM t WHERE {f} UNION ALL SELECT uid, s, w * 0.5, w > 3 FROM u",
    "union-all-three": "SELECT id FROM t WHERE {f} UNION ALL SELECT uid FROM u UNION ALL SELECT id FROM t WHERE id < 3",
    "union-all-names-from-the-first": "SELECT id AS a, k AS b FROM t WHERE {f} UNION ALL SELECT uid, k FROM u",
    "union-all-aggregates": "SELECT k, count(*) FROM t WHERE {f} GROUP BY k UNION ALL SELECT k, sum(w) FROM u GROUP BY k",
    "union-all-order-limit": "SELECT id, k FROM t WHERE {f} UNION ALL SELECT uid, k FROM u ORDER BY 1 DESC LIMIT 7",
    "union-all-order-by-name": "SELECT id AS a, k FROM t WHERE {f} UNION ALL SELECT uid, k FROM u ORDER BY k NULLS FIRST, a LIMIT 9 OFFSET 2",
    "union-all-parenthesized": "(SELECT id FROM t WHERE {f}) UNION ALL (SELECT uid FROM u)",
    "union": "SELECT k FROM t WHERE {f} UNION SELECT k FROM u",
    "union-several-columns": "SELECT k, s FROM t WHERE {f} UNION SELECT k, s FROM u",
    "union-every-kind": "SELECT s, d, v IS NULL AS b, k FROM t WHERE {f} UNION SELECT s, w * 0.01, w > 3, k FROM u",
    "union-order-by": "SELECT k FROM t WHERE {f} UNION SELECT k FROM u ORDER BY k NULLS LAST",
    "union-then-union-all": "SELECT k FROM t WHERE {f} UNION SELECT k FROM u UNION ALL SELECT k FROM u WHERE k = 1",
    "intersect": "SELECT k FROM t WHERE {f} INTERSECT SELECT k FROM u",
    "intersect-two-columns": "SELECT k, s FROM t WHERE {f} INTERSECT SELECT k, s FROM u",
    "except": "SELECT k FROM t WHERE {f} EXCEPT SELECT k FROM u",
    "except-reversed": "SELECT k FROM u EXCEPT SELECT k FROM t WHERE {f}",
    "except-two-columns": "SELECT k, s FROM t WHERE {f} EXCEPT SELECT k, s FROM u",
    "intersect-binds-tighter": "SELECT k FROM t WHERE {f} UNION SELECT k FROM u INTERSECT SELECT k FROM t WHERE k > 2",
    "left-to-right": "SELECT k FROM t WHERE {f} EXCEPT SELECT k FROM u WHERE k = 1 UNION SELECT k FROM u WHERE k = 4",
    "parenthesized-changes-the-order": "SELECT k FROM t WHERE {f} EXCEPT (SELECT k FROM u WHERE k = 1 UNION SELECT k FROM u WHERE k = 4)",
    "set-op-in-a-derived-table": "SELECT count(*), sum(k) FROM (SELECT k FROM t WHERE {f} UNION ALL SELECT k FROM u) x",
    "set-op-in-a-cte-read-twice": "WITH a AS (SELECT k FROM t WHERE {f} UNION SELECT k FROM u) SELECT x.k, y.k FROM a x JOIN a y ON x.k = y.k",
    "set-op-in-a-subquery": "SELECT id FROM t WHERE k IN (SELECT k FROM u WHERE {f} UNION SELECT 6)",
    "set-op-with-a-window": "SELECT id, rank() OVER (ORDER BY id DESC) FROM t WHERE {f} UNION ALL SELECT uid, rank() OVER (ORDER BY uid) FROM u",
}
SETOPS_ORDERED = {"union-all-order-limit", "union-all-order-by-name", "union-order-by"}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("f", FILTERS)
@pytest.mark.parametrize("name", list(SETOPS))
def test_set_operation_equals_duckdb(mx, data, name, f, mode):
    sql = SETOPS[name].format(f=f)
    if name == "set-op-in-a-subquery":
        sql = sql.replace("SELECT 6", "SELECT k FROM u WHERE k = 4")
    q(mx, data, mode, sql, ordered=name in SETOPS_ORDERED)


def test_the_result_columns_of_a_union_are_the_first_operands(mx, data):
    res = q(mx, data, "memory", "SELECT id AS a, s AS word FROM t UNION ALL SELECT uid, s FROM u")
    assert [c["name"] for c in res["columns"]] == ["a", "word"]


def test_a_union_over_csv_files(mx, data):
    q(mx, data, "csv", "SELECT k FROM t UNION SELECT k FROM u")


@pytest.mark.parametrize("sql,message", [
    ("SELECT id FROM t UNION ALL SELECT uid, k FROM u", "the operands of the set operation have 1 and 2 columns"),
    ("SELECT id FROM t UNION ALL SELECT s FROM u", "the operands of the set operation cannot be combined: column 1 is INT in one and TEXT in the other"),
    ("SELECT id FROM t INTERSECT ALL SELECT uid FROM u", "`INTERSECT ALL` is not supported yet"),
    ("SELECT id FROM t EXCEPT ALL SELECT uid FROM u", "`EXCEPT ALL` is not supported yet"),
    ("SELECT id FROM t UNION BY NAME SELECT uid FROM u", "`UNION BY NAME` is not supported yet"),
])
def test_unsupported_set_operations_raise_by_name(mx, data, sql, message):
    assert message in error_of(mx, data, sql)


def test_the_plan_names_the_set_operation(mx, data):
    text = plan(mx, data, "SELECT k FROM t UNION SELECT k FROM u")
    assert "UNION: 340 rows in, 8 out" in text


# ── step 4: filtering on a window result: QUALIFY ────────────────────────────────────────────────

QUALIFY = {
    "row-number-one": "SELECT id, k FROM t WHERE {f} QUALIFY row_number() OVER (PARTITION BY k ORDER BY id) = 1",
    "rank-alias": "SELECT id, k, rank() OVER (PARTITION BY k ORDER BY v, id) AS r FROM t WHERE {f} QUALIFY r <= 2",
    "window-not-selected": "SELECT id FROM t WHERE {f} QUALIFY dense_rank() OVER (PARTITION BY s ORDER BY v DESC NULLS LAST, id) <= 3",
    "running-total": "SELECT id, v FROM t WHERE {f} QUALIFY sum(v) OVER (ORDER BY id) > 150",
    "over-groups": "SELECT k, count(*) AS c FROM t WHERE {f} GROUP BY k QUALIFY rank() OVER (ORDER BY count(*) DESC, k) <= 3",
    "and-or": "SELECT id FROM t WHERE {f} QUALIFY row_number() OVER (PARTITION BY k ORDER BY id) = 1 OR lag(v) OVER (ORDER BY id) IS NULL",
    "order-limit": "SELECT id, k FROM t WHERE {f} QUALIFY row_number() OVER (PARTITION BY k ORDER BY id DESC) <= 2 ORDER BY k NULLS LAST, id LIMIT 6",
    "order-by-ordinal": "SELECT id, k FROM t WHERE {f} QUALIFY row_number() OVER (PARTITION BY k ORDER BY id) <= 3 ORDER BY 2 NULLS FIRST, 1 DESC",
    "order-by-an-unselected-column": "SELECT id FROM t WHERE {f} QUALIFY row_number() OVER (ORDER BY id) <= 8 ORDER BY v NULLS LAST, id",
    "order-by-an-expression": "SELECT id, k FROM t WHERE {f} QUALIFY row_number() OVER (ORDER BY id) <= 8 ORDER BY id % 3, id",
    "distinct": "SELECT DISTINCT k FROM t WHERE {f} QUALIFY row_number() OVER (PARTITION BY k, s ORDER BY id) <= 2",
    "star": "SELECT * FROM t WHERE {f} QUALIFY row_number() OVER (PARTITION BY k ORDER BY id) = 1",
    "in-a-derived-table": "SELECT count(*) FROM (SELECT id FROM t WHERE {f} QUALIFY row_number() OVER (PARTITION BY k ORDER BY id) = 1) x",
    "in-a-cte": "WITH a AS (SELECT id, k FROM t WHERE {f} QUALIFY row_number() OVER (PARTITION BY k ORDER BY id) <= 2) SELECT k, count(*) FROM a GROUP BY k",
    "limit-offset": "SELECT id FROM t WHERE {f} QUALIFY rank() OVER (PARTITION BY k ORDER BY id) = 1 ORDER BY id LIMIT 3 OFFSET 1",
}
QUALIFY_ORDERED = {"order-limit", "order-by-ordinal", "order-by-an-unselected-column", "order-by-an-expression", "limit-offset"}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("f", FILTERS)
@pytest.mark.parametrize("name", list(QUALIFY))
def test_qualify_equals_duckdb(mx, data, name, f, mode):
    q(mx, data, mode, QUALIFY[name].format(f=f), ordered=name in QUALIFY_ORDERED)


def test_qualify_over_a_csv_file(mx, data):
    q(mx, data, "csv", "SELECT id, k FROM t QUALIFY row_number() OVER (PARTITION BY k ORDER BY id) = 1")


# ── a compound query written to a Parquet file: only the last statement's rows go to the file ────

def test_a_compound_query_into_a_parquet_file_with_inner_sorts_that_spill(mx, data, tmp_path):
    sql = ("WITH a AS (SELECT id, k, v FROM t WHERE id >= 0 ORDER BY v, id) "
           "SELECT x.id, x.v FROM (SELECT id, v FROM a ORDER BY id DESC) x WHERE x.id IN (SELECT id FROM a WHERE k > 2) ORDER BY x.v NULLS LAST, x.id")
    out = tmp_path / "out.parquet"
    run_mx(mx, ["sql", "--format", "parquet", "--out", str(out), "--sort-budget", "1024", "--sort-run-rows", "40", "--spill-dir", str(tmp_path), sql],
           {"t": data["paths"]["t"]})
    got = duckdb.connect().execute(f"select * from read_parquet('{out}')").fetchall()
    want = data["con"].execute(sql).fetchall()
    assert got == want and len(got) > 50


# ── more shapes ──────────────────────────────────────────────────────────────────────────────────

MORE = {
    "star-with-a-correlated-inequality": "SELECT * FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.w > t.v AND {f})",
    "star-with-a-scalar": "SELECT * FROM t WHERE v < (SELECT avg(w) FROM u WHERE u.k = t.k AND {f}) AND id < 100",
    "between-two-scalars": "SELECT id FROM t WHERE v BETWEEN (SELECT min(w) FROM u WHERE {f}) AND (SELECT max(w) FROM u WHERE {f})",
    "derived-with-a-subquery-inside": "SELECT count(*) FROM (SELECT id FROM t WHERE k IN (SELECT k FROM u WHERE {f}) AND v > (SELECT avg(w) FROM u WHERE u.k = t.k)) x",
    "subquery-over-a-cte": "WITH a AS (SELECT k, w FROM u WHERE {f}) SELECT id FROM t WHERE v > (SELECT avg(w) FROM a WHERE a.k = t.k) OR id < 0",
    "correlated-through-a-cte-read-twice": "WITH a AS (SELECT k, w FROM u WHERE {f}) SELECT id FROM t WHERE EXISTS (SELECT 1 FROM a WHERE a.k = t.k) AND v >= (SELECT max(w) FROM a WHERE a.k = t.k)",
    "having-with-a-scalar": "SELECT k, sum(v) FROM t GROUP BY k HAVING sum(v) > (SELECT sum(w) FROM u WHERE {f}) / 3",
    "order-by-on-a-subquery-result": "SELECT id FROM t WHERE k IN (SELECT k FROM u WHERE {f}) ORDER BY v NULLS LAST, id LIMIT 5",
    "two-tables-two-correlated-conditions": ("SELECT t.id, u.uid FROM t JOIN u ON t.k = u.k WHERE EXISTS (SELECT 1 FROM u u2 WHERE u2.k = t.k AND u2.w <> u.w) "
                                              "AND t.v > (SELECT avg(w) FROM u u3 WHERE u3.k = u.k AND {f})"),
}
MORE_ORDERED = {"order-by-on-a-subquery-result"}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("f", FILTERS)
@pytest.mark.parametrize("name", list(MORE))
def test_more_subquery_shapes_equal_duckdb(mx, data, name, f, mode):
    q(mx, data, mode, MORE[name].format(f=f), ordered=name in MORE_ORDERED)


def test_star_next_to_a_correlated_subquery_over_several_tables_raises_by_name(mx, data):
    msg = error_of(mx, data, "SELECT * FROM t JOIN u ON t.k = u.k WHERE EXISTS (SELECT 1 FROM u u2 WHERE u2.k = t.k AND u2.w <> u.w)")
    assert "`*` in a query with a correlated subquery over several tables is not supported yet" in msg


def test_explain_analyze_of_a_compound_query(mx, data):
    out = run_mx(mx, ["explain", "--analyze", "--format", "json", "--stream", "never", "WITH a AS (SELECT id, k FROM t) SELECT x.id FROM a x JOIN a y ON x.k = y.k"],
                 {"t": data["paths"]["t"]})
    text = "\n".join(out["plan"])
    assert "CTE a: materialized once (2 reads, 300 rows)" in text and "result:" in text


# ── inlining: a derived table or CTE that only filters and projects is merged into the query that reads it ───────

def names_of(mx, data, sql):
    files = {t: data["paths"][t] for t in tables_of(sql)}
    return [c["name"] for c in run_mx(mx, ["sql", "--stream", "never", sql], files)["columns"]]


# (the query, the same query with an ORDER BY inside the derived table, which keeps it from being merged: it is evaluated first, and names its columns by the engine's own rules)
INLINED = [
    ("SELECT a, b + 1, a * 2 AS c, sum(a) AS sa FROM (SELECT id AS a, v AS b FROM t WHERE id < 50) x GROUP BY a, b",
     "SELECT a, b + 1, a * 2 AS c, sum(a) AS sa FROM (SELECT id AS a, v AS b FROM t WHERE id < 50 ORDER BY id) x GROUP BY a, b"),
    ("SELECT x.a, upper(x.s), count(*) FROM (SELECT id AS a, s FROM t WHERE k = 2) x GROUP BY x.a, x.s",
     "SELECT x.a, upper(x.s), count(*) FROM (SELECT id AS a, s FROM t WHERE k = 2 ORDER BY id) x GROUP BY x.a, x.s"),
    ("SELECT a, sum(b) AS total, count(*) FROM (SELECT id AS a, v AS b FROM t WHERE id < 50) x GROUP BY a",
     "SELECT a, sum(b) AS total, count(*) FROM (SELECT id AS a, v AS b FROM t WHERE id < 50 ORDER BY id) x GROUP BY a"),
    ("SELECT * FROM (SELECT id, v * 2 AS dv FROM t WHERE id < 5) x", "SELECT * FROM (SELECT id, v * 2 AS dv FROM t WHERE id < 5 ORDER BY id) x"),
    ("SELECT x.* FROM (SELECT id, v * 2 AS dv FROM t WHERE id < 5) x", "SELECT x.* FROM (SELECT id, v * 2 AS dv FROM t WHERE id < 5 ORDER BY id) x"),
    ("WITH a AS (SELECT id AS i, k FROM t WHERE v > 0) SELECT i, k, i + k FROM a WHERE i < 30 ORDER BY i",
     "WITH a AS (SELECT id AS i, k FROM t WHERE v > 0 ORDER BY id) SELECT i, k, i + k FROM a WHERE i < 30 ORDER BY i"),
    ("SELECT dv, dv + 1 FROM (SELECT v * 2 AS dv FROM t) WHERE dv > 10 ORDER BY dv, 2 LIMIT 5",
     "SELECT dv, dv + 1 FROM (SELECT v * 2 AS dv FROM t ORDER BY id) WHERE dv > 10 ORDER BY dv, 2 LIMIT 5"),
]


@pytest.mark.parametrize("sql,materialized", INLINED)
def test_an_inlined_query_keeps_the_column_names_it_would_have_had(mx, data, sql, materialized):
    assert "inlined into the query that reads it" in plan(mx, data, sql)
    assert "inlined into the query that reads it" not in plan(mx, data, materialized)
    assert names_of(mx, data, sql) == names_of(mx, data, materialized)
    q(mx, data, "memory", sql, ordered="ORDER BY" in sql)


def test_an_unaliased_aggregate_over_a_derived_column_is_not_merged_so_its_name_is_kept(mx, data):
    sql = "SELECT sum(a), max(b) FROM (SELECT id AS a, v AS b FROM t) x"
    assert "inlined into the query that reads it" not in plan(mx, data, sql)
    assert names_of(mx, data, sql) == ["sum(a)", "max(b)"]


def test_a_column_the_derived_table_does_not_have_is_an_error_not_the_source_column(mx, data):
    assert "column `s` not found" in error_of(mx, data, "SELECT s FROM (SELECT id FROM t) x")
    assert "column `s` not found" in error_of(mx, data, "WITH a AS (SELECT id FROM t) SELECT s FROM a")


def test_an_inlined_scan_streams_with_the_filter_pushed_into_it(mx, data):
    out = run_mx(mx, ["sql", "--stream", "always", "SELECT count(*) FROM (SELECT id, k FROM t WHERE id >= 200) x WHERE k = 3"], {"t": data["paths"]["t"]})
    assert any(n.startswith("ran: scan t") for n in out["notes"]), out["notes"]
    assert out["rows"] == [[sum(1 for i in range(200, 300) if i % 11 != 0 and i % 7 == 3)]]


# ── DuckDB's own text of the TPC-H queries that need F-159 (Q11 sums DECIMAL x DECIMAL past int64: its own ticket; Q15 is written with WITH, DuckDB's text makes a view) ──

Q15_WITH = """WITH revenue0 AS (SELECT l_suppkey AS supplier_no, sum(l_extendedprice * (1 - l_discount)) AS total_revenue FROM lineitem
 WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-01-01' + INTERVAL '3' MONTH GROUP BY l_suppkey)
SELECT s_suppkey, s_name, s_address, s_phone, total_revenue FROM supplier, revenue0 WHERE s_suppkey = supplier_no
 AND total_revenue = (SELECT max(total_revenue) FROM revenue0) ORDER BY s_suppkey"""


@pytest.mark.parametrize("n", [2, 4, 7, 8, 9, 13, 15, 16, 17, 18, 20, 21, 22])
def test_duckdbs_own_tpch_text_equals_duckdb(mx, tpch, n):
    sql = Q15_WITH if n == 15 else tpch["con"].execute(f"select query from tpch_queries() where query_nr = {n}").fetchone()[0].strip().rstrip(";")
    res = tpch_q(mx, tpch, sql, ordered=True)
    assert len(res["rows"]) > 0


# ── F-159 review: a substituted subquery is a value, never a position; INT = correlated FLOAT ────

ORDER_BY_VALUE = [
    "SELECT k, id FROM t WHERE id < 40 ORDER BY (SELECT min(w) FROM u), id",
    "SELECT k, id FROM t WHERE id < 40 ORDER BY (SELECT min(uid) + 2 FROM u), id",
    "SELECT k, id FROM t WHERE id < 40 ORDER BY (SELECT max(w) FROM u) DESC, id DESC",
    "SELECT k, count(*) AS c FROM t GROUP BY k ORDER BY (SELECT 2 FROM u LIMIT 1), k",
    "SELECT k, count(*) AS c FROM t GROUP BY k ORDER BY 2 DESC, k",
    "WITH a AS (SELECT k, id FROM t WHERE id < 40) SELECT k, id FROM a ORDER BY (SELECT min(w) FROM u), id",
    "SELECT k, id FROM t WHERE id < 40 ORDER BY 2, 1",
    "SELECT k, id FROM t WHERE id < 40 QUALIFY row_number() OVER (ORDER BY id) < 30 ORDER BY (SELECT min(uid) + 2 FROM u), id",
    "SELECT id FROM t WHERE id < 40 GROUP BY id, (SELECT min(uid) + 1 FROM u) ORDER BY id",
]


@pytest.mark.parametrize("sql", ORDER_BY_VALUE)
def test_a_subquery_in_order_by_is_a_value_not_a_position(mx, data, sql):
    q(mx, data, "memory", sql, ordered=True)


@pytest.mark.parametrize("f", FILTERS)
def test_an_int_equal_to_a_correlated_float_subquery(mx, data, f):
    q(mx, data, "memory", f"SELECT id FROM t WHERE v = (SELECT avg(w) FROM u WHERE u.k = t.k AND {f})")
    q(mx, data, "memory", f"SELECT id FROM t WHERE (SELECT avg(w) FROM u WHERE u.k = t.k AND {f}) = v * 1.0")
    q(mx, data, "memory", f"SELECT id FROM t WHERE d = (SELECT max(d) FROM t t2 WHERE t2.k = t.k AND {f.replace('id', 't2.id')})")


# ── F-159 review 2: = against a correlated scalar of any kind; SELECT with no FROM ────────────────

NON_NUMERIC_EQUALS = [
    "SELECT id FROM t WHERE s = (SELECT min(s) FROM u WHERE u.k = t.k AND u.s = 'Zed' AND {f})",
    "SELECT id FROM t WHERE (SELECT max(s) FROM u WHERE u.k = t.k AND u.s = 'west' AND {f}) = s",
    "SELECT id FROM (SELECT id, k, v > 0 AS b FROM t ORDER BY id) x WHERE b = (SELECT max(w) > 3 FROM u WHERE u.k = x.k AND {f})",
    "SELECT id FROM (SELECT id, k, DATE '2020-01-01' + CAST(id % 30 AS INTEGER) AS dt FROM t ORDER BY id) x WHERE dt = (SELECT max(DATE '2020-01-01' + CAST(w % 30 AS INTEGER)) FROM u WHERE u.k = x.k AND {f})",
    "SELECT id FROM t WHERE d = (SELECT max(d) FROM t t2 WHERE t2.k = t.k AND {fq})",
]


@pytest.mark.parametrize("f", FILTERS)
@pytest.mark.parametrize("sql", NON_NUMERIC_EQUALS)
def test_text_bool_date_equal_to_a_correlated_scalar(mx, data, sql, f):
    q(mx, data, "memory", sql.format(f=f, fq=f.replace("id", "t2.id")))


@pytest.mark.parametrize("sql", [
    "SELECT id FROM t WHERE v > (SELECT 3) AND id < 30",
    "SELECT id, (SELECT 1) AS one FROM t WHERE id < 3",
    "SELECT id FROM t WHERE k IN (SELECT 1 UNION SELECT 2)",
    "SELECT id FROM t WHERE EXISTS (SELECT 1) AND id < 3",
    "SELECT id FROM t WHERE id < (SELECT 2 + 3)",
])
def test_select_with_no_from(mx, data, sql):
    q(mx, data, "memory", sql)
