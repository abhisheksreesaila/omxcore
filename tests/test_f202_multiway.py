"""F-202: multi-way joins at DuckDB speed. What each change must keep true, on tables that are not TPC-H (NULL keys, duplicates, empty sides, dates, text):

- an OR of ANDs across tables implies a filter for each table alone (the filter runs on that table first; the OR itself is still checked after the joins);
- a big filtered table that is joined behind a small one is read only where its key has a partner (the runtime join filter), or stays unwritten as the last probe side;
- a TEXT group key that the select list only passes through is looked up after ORDER BY ... LIMIT picked its groups, not built for every group.
Every answer is equal to DuckDB; the notes (the engine's own EXPLAIN) prove which path ran.
"""

import pytest

from conftest import oracle_connection
from mx_oracle import check, mx_binary


@pytest.fixture(scope="session")
def mx():
    return mx_binary()


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("f202")
    con = oracle_connection()
    con.execute("create table nat as select i as nk, 'N' || lpad(i::varchar, 2, '0') as nname from range(25) r(i)")
    # a dimension of unique keys that points at a nation (some NULL), a second that does
    con.execute("""create table da as select i as a, 'A' || (i % 97) as an, case when i % 41 = 0 then null else i % 25 end as ak, i % 5 as seg from range(3000) r(i)""")
    con.execute("""create table db as select i as b, 'B-' || i as bn, (i * 7) % 25 as bk, ((i % 13) * 0.5)::decimal(8,1) as w from range(4000) r(i)""")
    # the fact: 200,000 rows, NULL and dangling keys, dates, text, a decimal with NULLs
    con.execute("""create table f as select i as id,
        case when i % 31 = 0 then null when i % 53 = 0 then 90000 + i else (i * 7) % 3000 end as a,
        case when i % 37 = 0 then null else ((i // 25) * 11) % 4000 end as b,
        case when i % 19 = 0 then null else ((i % 700) * 0.25)::decimal(10,2) end as x,
        date '2020-01-01' + (i % 800)::int as d,
        (['red','Green','blue','amber'])[1 + i % 4] as s from range(200000) r(i)""")
    # a table whose key repeats (about 3 rows a key)
    con.execute("create table dup as select (i * 5) % 3000 as a, i as did, (i % 9) as v from range(9000) r(i)")
    con.execute("create table nothing as select * from da where false")
    paths = {}
    for t, rg in [("nat", 25), ("da", 700), ("db", 900), ("f", 50000), ("dup", 3000), ("nothing", 100)]:
        paths[t] = d / f"{t}.parquet"
        con.execute(f"copy {t} to '{paths[t]}' (format parquet, row_group_size {rg})")
    return {"con": con, "paths": paths}


def q(mx, data, sql, tables, ordered=False, extra=()):
    return check(mx, sql, data["con"], files={t: data["paths"][t] for t in tables}, ordered=ordered, extra=extra)


def note(res, text):
    hits = [n for n in res["notes"] if text in n]
    assert hits, (text, res["notes"])
    return hits[0]


# Q7's shape: two aliases of one small table, an OR of ANDs across them
OR_PAIRS = """SELECT n1.nname AS sn, n2.nname AS cn, extract(year FROM f.d) AS y, sum(f.x) AS total, count(*) AS c
FROM da, f, db, nat n1, nat n2
WHERE da.a = f.a AND db.b = f.b AND da.ak = n1.nk AND db.bk = n2.nk
  AND ((n1.nname = 'N03' AND n2.nname = 'N07') OR (n1.nname = 'N07' AND n2.nname = 'N03'))
  AND f.d BETWEEN date '2020-06-01' AND date '2021-12-31'
GROUP BY 1, 2, 3 ORDER BY 1, 2, 3"""


def test_an_or_across_tables_filters_each_table_first(mx, data):
    res = q(mx, data, OR_PAIRS, ["da", "f", "db", "nat"], ordered=True)
    assert res["rows"], "the OR has rows to find"
    note(res, "n1 (2 of 25 rows)")                                  # the OR implies n_name IN ('N03', 'N07') for each alias
    note(res, "n2 (2 of 25 rows)")


@pytest.mark.parametrize("sql", [
    # one branch has nothing to say about a table: no implied filter for it (and the answer is the same)
    "SELECT count(*), sum(f.x) FROM da, f, db WHERE da.a = f.a AND db.b = f.b AND ((da.seg = 1 AND db.w > 3) OR (db.w < 1))",
    "SELECT da.seg, count(*) FROM da, f, db WHERE da.a = f.a AND db.b = f.b AND ((da.seg = 1 AND db.w > 3) OR (da.seg = 2 AND db.w < 1)) GROUP BY da.seg ORDER BY da.seg",
    # a NULL in one branch's test
    "SELECT count(*) FROM da, f, db WHERE da.a = f.a AND db.b = f.b AND ((da.ak = 3 AND f.x > 100) OR (da.ak = 4 AND db.w < 2))",
    # an OR nested in an AND, a NOT, an IN
    "SELECT count(*), sum(f.x) FROM da, f, db WHERE da.a = f.a AND db.b = f.b AND (da.seg IN (1, 2) AND (db.w > 4 OR f.d < date '2020-03-01'))",
    "SELECT count(*) FROM da, f, db WHERE da.a = f.a AND db.b = f.b AND NOT (da.seg = 1 OR db.w > 2)",
    # an OR whose branches have no condition on one table at all
    "SELECT count(*) FROM da, f, db WHERE da.a = f.a AND db.b = f.b AND (f.s = 'red' OR f.d > date '2021-06-01')",
])
def test_or_conditions_equal_duckdb(mx, data, sql):
    q(mx, data, sql, ["da", "f", "db"])


def test_an_empty_dimension_and_null_keys(mx, data):
    q(mx, data, "SELECT count(*), sum(f.x) FROM nothing, f, db WHERE nothing.a = f.a AND db.b = f.b", ["nothing", "f", "db"])
    q(mx, data, "SELECT count(*) FROM da, f, nat WHERE da.a = f.a AND da.ak = nat.nk AND f.x > 100 AND ((nat.nname = 'N01' AND da.seg = 2) OR (nat.nname = 'N02' AND da.seg = 3))", ["da", "f", "nat"])


# the runtime join filter: f (big, filtered) is read only where its key has a partner among the rows joined so far
REDUCED = [
    "SELECT da.an, count(*), sum(f.x) FROM da, f, db WHERE da.a = f.a AND f.b = db.b AND da.seg = 2 AND da.ak = 7 AND f.d >= date '2020-02-01' AND f.x > 10 GROUP BY da.an ORDER BY da.an",
    "SELECT db.bn, count(f.x), min(f.d) FROM db, f, da WHERE db.b = f.b AND f.a = da.a AND db.w = 3.0 AND db.bk = 4 AND f.s <> 'red' GROUP BY db.bn ORDER BY db.bn",
    # duplicates among the keys joined first
    "SELECT dup.v, count(*), sum(f.x) FROM dup, f, da WHERE dup.a = f.a AND f.a = da.a AND dup.v = 2 AND f.d > date '2020-05-01' GROUP BY dup.v",
]


@pytest.mark.parametrize("sql", REDUCED)
def test_a_leaf_behind_a_small_table_is_read_where_its_key_has_a_partner(mx, data, sql):
    tables = [t for t in ["da", "f", "db", "dup"] if f" {t}," in sql or f" {t} " in sql or sql.endswith(f" {t}")]
    res = q(mx, data, sql, tables, ordered=True, extra=["--order", "0,1,2"])        # the small table first, then f, then the rest
    note(res, "runtime join filter: f is read only where its key has a partner")


def test_the_reduced_read_with_no_partner_at_all(mx, data):
    q(mx, data, "SELECT count(*), sum(f.x) FROM da, f, db WHERE da.a = f.a AND f.b = db.b AND da.seg = 3 AND da.ak = 99 AND f.d > date '2020-02-01'", ["da", "f", "db"], extra=["--order", "0,1,2"])


# a TEXT key that is only passed through, with ORDER BY ... LIMIT over many groups
MANY_GROUPS = [
    ("SELECT f.id, da.an, db.bn, sum(f.x) AS t FROM f, da, db WHERE f.a = da.a AND f.b = db.b GROUP BY f.id, da.an, db.bn ORDER BY t DESC, f.id LIMIT 10", True),
    ("SELECT f.id, da.an, db.bn, sum(f.x) AS t FROM f, da, db WHERE f.a = da.a AND f.b = db.b GROUP BY f.id, da.an, db.bn ORDER BY t DESC NULLS LAST, f.id LIMIT 7 OFFSET 3", True),
    # the key is also what ORDER BY reads: it must be real for every group
    ("SELECT da.an, f.d, sum(f.x) AS t FROM f, da, db WHERE f.a = da.a AND f.b = db.b GROUP BY da.an, f.d ORDER BY da.an, f.d DESC LIMIT 60", True),
    ("SELECT f.id, db.bn, sum(f.x) AS t FROM f, da, db WHERE f.a = da.a AND f.b = db.b GROUP BY f.id, db.bn ORDER BY db.bn DESC, f.id LIMIT 25", True),
    # a text key used twice in the select list, and one with NULLs (a LEFT join leaves them)
    ("SELECT f.id, da.an, da.an AS again, sum(f.x) AS t FROM f, da, db WHERE f.a = da.a AND f.b = db.b GROUP BY f.id, da.an ORDER BY t DESC, f.id LIMIT 5", True),
    ("SELECT f.id, da.an, sum(f.x) AS t FROM f LEFT JOIN da ON f.a = da.a GROUP BY f.id, da.an ORDER BY t DESC NULLS LAST, f.id LIMIT 10", True),
    ("SELECT f.id, da.an, count(*) AS c FROM f LEFT JOIN da ON f.a = da.a GROUP BY f.id, da.an ORDER BY c DESC, f.id DESC LIMIT 10", True),
]


@pytest.mark.parametrize("sql,ordered", MANY_GROUPS)
def test_text_keys_passed_through_order_by_limit(mx, data, sql, ordered):
    q(mx, data, sql, [t for t in ["f", "da", "db"] if f" {t} " in sql or f" {t}," in sql], ordered=ordered)
