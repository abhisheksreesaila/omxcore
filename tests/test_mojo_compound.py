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
