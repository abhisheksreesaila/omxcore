"""F-155: joins of three to eight tables (the TPC-H Q3, Q5, Q7, Q8, Q9, Q10 and Q19 shapes on dbgen SF 0.1), equal to DuckDB.

Every query runs through `build/mx` (tests/mx_oracle.py: typed values compared exactly) and through DuckDB on the same Parquet files, under the planner's
order and under every join order the options can force (every permutation that connects, for up to four tables; the written order and a spread of others
for more). The notes say which order ran, the build side of each step and how the last join ran.

The queries keep the join structure of the TPC-H queries and drop what SQL expressions (F-156) have not reached yet: EXTRACT(year), CASE, LIKE.
"""
import itertools
import re

import duckdb
import pytest

from mx_oracle import check, mx_binary, run_mx


@pytest.fixture(scope="session")
def mx():
    return mx_binary()


@pytest.fixture(scope="session")
def tpch(tmp_path_factory):
    d = tmp_path_factory.mktemp("tpch01")
    con = duckdb.connect()
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    con.execute("CALL dbgen(sf=0.1)")
    paths = {}
    for (t,) in con.execute("show tables").fetchall():
        paths[t] = d / f"{t}.parquet"
        con.execute(f"copy {t} to '{paths[t]}' (format parquet, row_group_size 30000)")
    return {"con": con, "paths": paths}


def tables_of(sql):
    return [t for t in ("customer", "orders", "lineitem", "supplier", "nation", "region", "part", "partsupp") if re.search(rf"\b{t}\b", sql)]


def q(mx, tpch, sql, ordered=False, extra=(), duck_sql=None):
    return check(mx, sql, tpch["con"], files={t: tpch["paths"][t] for t in tables_of(sql)}, ordered=ordered, extra=extra, duck_sql=duck_sql)


def note(res, prefix):
    hits = [n for n in res["notes"] if n.startswith(prefix)]
    assert hits, (prefix, res["notes"])
    return hits[0]


Q3 = """SELECT l_orderkey, sum(l_extendedprice * (1 - l_discount)) AS revenue, o_orderdate, o_shippriority
FROM customer, orders, lineitem
WHERE c_mktsegment = 'BUILDING' AND c_custkey = o_custkey AND l_orderkey = o_orderkey
  AND o_orderdate < DATE '1995-03-15' AND l_shipdate > DATE '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority ORDER BY revenue DESC, o_orderdate, l_orderkey LIMIT 10"""

Q3_JOIN_SYNTAX = """SELECT l_orderkey, sum(l_extendedprice * (1 - l_discount)) AS revenue, o_orderdate, o_shippriority
FROM customer JOIN orders ON c_custkey = o_custkey JOIN lineitem ON l_orderkey = o_orderkey
WHERE c_mktsegment = 'BUILDING' AND o_orderdate < DATE '1995-03-15' AND l_shipdate > DATE '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority ORDER BY revenue DESC, o_orderdate, l_orderkey LIMIT 10"""

Q5 = """SELECT n_name, sum(l_extendedprice * (1 - l_discount)) AS revenue
FROM customer, orders, lineitem, supplier, nation, region
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND l_suppkey = s_suppkey AND c_nationkey = s_nationkey
  AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'ASIA'
  AND o_orderdate >= DATE '1994-01-01' AND o_orderdate < DATE '1995-01-01'
GROUP BY n_name ORDER BY revenue DESC"""

Q7 = """SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation, sum(l_extendedprice * (1 - l_discount)) AS revenue, count(*) AS n
FROM supplier, lineitem, orders, customer, nation n1, nation n2
WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey AND c_custkey = o_custkey AND s_nationkey = n1.n_nationkey AND c_nationkey = n2.n_nationkey
  AND ((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY') OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE'))
  AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
GROUP BY n1.n_name, n2.n_name ORDER BY supp_nation, cust_nation"""

Q8 = """SELECT n2.n_name AS supp_nation, sum(l_extendedprice * (1 - l_discount)) AS volume, count(*) AS n
FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
WHERE p_partkey = l_partkey AND s_suppkey = l_suppkey AND l_orderkey = o_orderkey AND o_custkey = c_custkey AND c_nationkey = n1.n_nationkey
  AND n1.n_regionkey = r_regionkey AND r_name = 'AMERICA' AND s_nationkey = n2.n_nationkey
  AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31' AND p_type = 'ECONOMY ANODIZED STEEL'
GROUP BY n2.n_name ORDER BY supp_nation"""

Q9 = """SELECT n_name, o_orderdate, sum(l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity) AS profit
FROM part, supplier, lineitem, partsupp, orders, nation
WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey AND ps_partkey = l_partkey AND p_partkey = l_partkey AND o_orderkey = l_orderkey
  AND s_nationkey = n_nationkey AND p_size < 4
GROUP BY n_name, o_orderdate ORDER BY n_name, o_orderdate DESC LIMIT 60"""

Q10 = """SELECT c_custkey, c_name, sum(l_extendedprice * (1 - l_discount)) AS revenue, c_acctbal, n_name, c_address, c_phone
FROM customer, orders, lineitem, nation
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND o_orderdate >= DATE '1993-10-01' AND o_orderdate < DATE '1994-01-01'
  AND l_returnflag = 'R' AND c_nationkey = n_nationkey
GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address ORDER BY revenue DESC, c_custkey LIMIT 20"""

Q19 = """SELECT sum(l_extendedprice * (1 - l_discount)) AS revenue
FROM lineitem, part
WHERE (p_partkey = l_partkey AND p_brand = 'Brand#12' AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') AND l_quantity >= 1 AND l_quantity <= 11 AND p_size BETWEEN 1 AND 5
       AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON')
   OR (p_partkey = l_partkey AND p_brand = 'Brand#23' AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') AND l_quantity >= 10 AND l_quantity <= 20 AND p_size BETWEEN 1 AND 10
       AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON')
   OR (p_partkey = l_partkey AND p_brand = 'Brand#34' AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') AND l_quantity >= 20 AND l_quantity <= 30 AND p_size BETWEEN 1 AND 15
       AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON')"""

SHAPES = {"q3": Q3, "q3j": Q3_JOIN_SYNTAX, "q5": Q5, "q7": Q7, "q8": Q8, "q9": Q9, "q10": Q10}
ORDERED = {"q3": True, "q3j": True, "q5": True, "q7": True, "q8": True, "q9": True, "q10": True}


@pytest.mark.parametrize("name", list(SHAPES))
def test_the_planners_choice_equals_duckdb(mx, tpch, name):
    res = q(mx, tpch, SHAPES[name], ordered=ORDERED[name])
    assert note(res, "join order:").startswith("join order: "), res["notes"]
    assert any(n.startswith("last join:") or n.startswith("star dimension") for n in res["notes"]), res["notes"]
    assert any(n.startswith("join path:") for n in res["notes"]), res["notes"]


@pytest.mark.parametrize("name", list(SHAPES))
def test_the_written_order_equals_duckdb(mx, tpch, name):
    res = q(mx, tpch, SHAPES[name], ordered=ORDERED[name], extra=["--order", "written"])
    assert "join order: " in " ".join(res["notes"])


def n_tables(name):
    return len(tables_of(SHAPES[name])) + (1 if name in ("q7", "q8") else 0) + (1 if name == "q8" else 0)


def connected_orders(name, n, limit):
    """Positions in FROM, permuted; the ones the planner accepts (every table reaches an equality to the ones before it) are the tests: a refused one is checked apart."""
    orders = list(itertools.permutations(range(n)))
    if len(orders) > limit:
        step = len(orders) // limit
        orders = orders[::step][:limit]
    return orders


@pytest.mark.parametrize("name,n", [("q3", 3), ("q3j", 3), ("q10", 4), ("q9", 6), ("q5", 6), ("q7", 6)])
def test_every_forced_order_equals_duckdb_or_is_refused_by_name(mx, tpch, name, n):
    ran = 0
    refused = 0
    for order in connected_orders(name, n, 24):
        extra = ["--order", ",".join(map(str, order))]
        try:
            q(mx, tpch, SHAPES[name], ordered=ORDERED[name], extra=extra)
            ran += 1
        except AssertionError as e:
            text = str(e)
            # a refused order: a cross product, or a join that explodes past the memory budget (an order that joins on a low-cardinality key first)
            assert "reaches a table no equality joins yet" in text or "memory budget" in text, text[:600]
            refused += 1
    assert ran >= 4, (ran, refused)


@pytest.mark.parametrize("name", ["q3", "q5", "q10"])
@pytest.mark.parametrize("build", ["left", "right"])
def test_a_forced_build_side_equals_duckdb(mx, tpch, name, build):
    res = q(mx, tpch, SHAPES[name], ordered=ORDERED[name], extra=["--build", build])
    assert "forced" in " ".join(res["notes"]), res["notes"]


@pytest.mark.parametrize("name", ["q3", "q10", "q9"])
def test_materialized_and_threads_equal_duckdb(mx, tpch, name):
    res = q(mx, tpch, SHAPES[name], ordered=ORDERED[name], extra=["--join", "materialize"])
    assert note(res, "join path").startswith("join path: materialized:"), res["notes"]
    q(mx, tpch, SHAPES[name], ordered=ORDERED[name], extra=["--threads", "1"])


def test_streamed_scans_equal_duckdb(mx, tpch):
    q(mx, tpch, Q3, ordered=True, extra=["--stream", "always"])
    q(mx, tpch, Q5, ordered=True, extra=["--stream", "always"])


def test_q19_takes_the_common_equality_out_of_the_or(mx, tpch):
    """The three disjuncts share `p_partkey = l_partkey`: it becomes the join key, the rest of the OR is checked on the joined rows."""
    res = q(mx, tpch, Q19)
    assert "join build side:" in " ".join(res["notes"]), res["notes"]


def test_q19_in_join_syntax(mx, tpch):
    sql = Q19.replace("FROM lineitem, part", "FROM lineitem JOIN part ON p_partkey = l_partkey")
    q(mx, tpch, sql)


# ── EXPLAIN and EXPLAIN ANALYZE ───────────────────────────────────────────────────────────────────────

def test_explain_shows_every_decision_and_estimate(mx, tpch):
    files = {t: tpch["paths"][t] for t in tables_of(Q5)}
    out = run_mx(mx, ["sql", "EXPLAIN " + Q5], files)
    text = "\n".join(r[0] for r in out["rows"])
    assert "Logical plan" in text and "Join (inner) on" in text and "Scan customer" in text
    assert "join order:" in text and "join start:" in text and "join 1:" in text and "join 5:" in text, text
    assert "estimated" in text and "hash table on" in text, text


def test_explain_analyze_shows_actual_rows_and_milliseconds_per_operator(mx, tpch):
    files = {t: tpch["paths"][t] for t in tables_of(Q5)}
    out = run_mx(mx, ["sql", "EXPLAIN ANALYZE " + Q5], files)
    text = "\n".join(r[0] for r in out["rows"])
    assert "\nActual" in "\n" + text and "scan region:" in text and ("join step 1" in text or "star dimension group, join 1" in text) and "result: 5 rows in" in text, text
    assert re.search(r"scan orders: 150,000 rows, [\d,]+ after its filter", text), text


def test_explain_analyze_of_a_two_table_join_and_a_single_table(mx, tpch):
    files = {"orders": tpch["paths"]["orders"], "customer": tpch["paths"]["customer"]}
    out = run_mx(mx, ["sql", "EXPLAIN ANALYZE SELECT count(*) FROM orders JOIN customer ON o_custkey = c_custkey WHERE c_mktsegment = 'BUILDING'"], files)
    text = "\n".join(r[0] for r in out["rows"])
    assert "Actual" in text and "join (inner):" in text and "result: 1 rows in" in text, text
    out = run_mx(mx, ["sql", "EXPLAIN ANALYZE SELECT count(*) FROM orders"], {"orders": tpch["paths"]["orders"]})
    text = "\n".join(r[0] for r in out["rows"])
    assert "Actual" in text and "result: 1 rows in" in text, text


# ── what is refused ───────────────────────────────────────────────────────────────────────────────────

def err(mx, tpch, sql):
    files = {t: tpch["paths"][t] for t in tables_of(sql)}
    return run_mx(mx, ["sql", sql], files, expect_exit=1)["error"]


def test_a_table_no_equality_reaches_is_refused(mx, tpch):
    e = err(mx, tpch, "SELECT count(*) FROM customer, orders, nation WHERE c_custkey = o_custkey")
    assert "is not joined to the other tables by an equality" in e["message"], e


def test_a_bad_forced_order_is_refused(mx, tpch):
    files = {t: tpch["paths"][t] for t in tables_of(Q3)}
    e = run_mx(mx, ["sql", Q3, "--order", "0,1"], files, expect_exit=1)["error"]
    assert "it names 2 tables, the query joins 3" in e["message"], e
    e = run_mx(mx, ["sql", Q3, "--order", "0,0,1"], files, expect_exit=1)["error"]
    assert "each of the 3 tables (0 to 2) appears once" in e["message"], e
    e = run_mx(mx, ["sql", Q3, "--order", "0,2,1"], files, expect_exit=1)["error"]
    assert "reaches a table no equality joins yet" in e["message"], e


def test_left_join_with_a_third_table_is_refused_by_name(mx, tpch):
    e = err(mx, tpch, "SELECT count(*) FROM customer LEFT JOIN orders ON c_custkey = o_custkey JOIN nation ON c_nationkey = n_nationkey")
    assert "a `LEFT JOIN` together with a third table is not supported yet" in e["message"], e


def test_using_with_three_tables_is_refused(mx, tpch):
    e = err(mx, tpch, "SELECT count(*) FROM customer JOIN orders ON c_custkey = o_custkey JOIN nation USING (n_nationkey)")
    assert "USING" in e["message"], e


def test_the_join_budget_is_a_readable_error(mx, tpch):
    files = {t: tpch["paths"][t] for t in tables_of(Q10)}
    e = run_mx(mx, ["sql", Q10, "--join-budget", "100000", "--join", "materialize"], files, expect_exit=1)["error"]
    assert "past the join's memory budget" in e["message"], e


# ── a star: the fact table probes several dimensions with unique keys in one pass ─────────────────────

STAR3 = """SELECT o_orderpriority, p_brand, count(*), sum(l_extendedprice * (1 - l_discount))
FROM lineitem, orders, part, supplier
WHERE l_orderkey = o_orderkey AND l_partkey = p_partkey AND l_suppkey = s_suppkey AND p_size < 20 AND s_acctbal > 0
GROUP BY o_orderpriority, p_brand"""

STAR_FACT_FILTER = """SELECT s_nationkey, p_brand, count(*), sum(l_quantity), min(o_orderdate), max(p_retailprice)
FROM lineitem, part, supplier, orders
WHERE l_partkey = p_partkey AND l_suppkey = s_suppkey AND l_orderkey = o_orderkey AND l_quantity > 10 AND l_discount < 0.09 AND o_orderstatus <> 'P'
GROUP BY s_nationkey, p_brand"""

SNOWFLAKE = """SELECT n_name, p_brand, count(*), sum(l_extendedprice)
FROM lineitem, orders, part, supplier, nation
WHERE l_orderkey = o_orderkey AND l_partkey = p_partkey AND l_suppkey = s_suppkey AND s_nationkey = n_nationkey AND o_totalprice > 100000
GROUP BY n_name, p_brand"""

STAR_ROWS = """SELECT l_orderkey, l_linenumber, p_brand, s_name, o_orderdate
FROM lineitem, part, supplier, orders
WHERE l_partkey = p_partkey AND l_suppkey = s_suppkey AND l_orderkey = o_orderkey AND p_size = 7 AND s_acctbal > 9000 AND l_quantity > 45
ORDER BY l_orderkey, l_linenumber LIMIT 50"""


@pytest.mark.parametrize("sql,extra", [(STAR3, []), (STAR_FACT_FILTER, []), (SNOWFLAKE, ["--order", "written"])])
def test_a_star_runs_in_one_fused_pass_and_equals_duckdb(mx, tpch, sql, extra):
    """The fact table first (the written order): every other table is a dimension with unique keys; nation, reached through supplier, is joined to it first."""
    ordered = "LIMIT" in sql
    res = q(mx, tpch, sql, ordered=ordered, extra=extra)
    path = note(res, "join path")
    assert path.startswith("join path: fused: a star:"), res["notes"]
    assert any(n.startswith("star dimension") for n in res["notes"]), res["notes"]
    if sql == SNOWFLAKE:
        assert any("supplier+nation" in n for n in res["notes"]), res["notes"]


@pytest.mark.parametrize("sql", [STAR3, STAR_FACT_FILTER, SNOWFLAKE, STAR_ROWS])
def test_a_star_equals_duckdb_when_materialized_and_on_one_thread_and_streamed(mx, tpch, sql):
    ordered = "LIMIT" in sql
    res = q(mx, tpch, sql, ordered=ordered, extra=["--join", "materialize", "--order", "written"])
    assert "star" not in note(res, "join path"), res["notes"]
    q(mx, tpch, sql, ordered=ordered, extra=["--threads", "1", "--order", "written"])
    q(mx, tpch, sql, ordered=ordered, extra=["--stream", "always", "--order", "written"])


def test_a_star_with_the_facts_filter_in_the_pipeline_says_so(mx, tpch):
    res = q(mx, tpch, STAR_FACT_FILTER, extra=["--order", "written"])
    assert "the probe table's filter runs in the pipeline" in note(res, "join path"), res["notes"]


@pytest.mark.parametrize("order", ["0,1,2,3", "1,0,2,3", "2,1,0,3", "3,0,1,2"])
def test_a_star_under_forced_orders_equals_duckdb(mx, tpch, order):
    try:
        q(mx, tpch, STAR3, extra=["--order", order])
    except AssertionError as e:
        assert "reaches a table no equality joins yet" in str(e) or "memory budget" in str(e), str(e)[:400]
