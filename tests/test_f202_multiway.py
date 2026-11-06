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
    # the OR implies nname IN ('N03', 'N07') for each alias: each is joined as 2 rows, not 25
    import re
    for alias in ("n1", "n2"):
        assert any(re.search(rf"{alias}\.nk [\d,]+ in 2$", n) or re.search(rf"{alias}\.nk has [\d,]+ distinct values in 2 rows", n) for n in res["notes"]), res["notes"]


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

# LIKE patterns of several literals and `%` (TPC-H Q13's '%special%requests%'): matched by finding each literal after the one before it
@pytest.fixture(scope="session")
def words(tmp_path_factory):
    d = tmp_path_factory.mktemp("f202w")
    con = oracle_connection()
    con.execute("""create table w as select i as id, case when i % 17 = 0 then null else
        (['', 'a', 'ab', 'aa', 'aab', 'abab', 'special requests', 'special  requests', 'requestsspecial', 'xspecialyrequestsz', 'special', 'requests',
          'aXbXc', 'abc', 'cba', 'a%b', 'a_b', 'café au lait', 'naïve café', 'éé', 'ba', 'bab', 'aXXb', 'the special offer requests more'])[1 + i % 24] end as s
        from range(240) r(i)""")
    p = d / "w.parquet"
    con.execute(f"copy w to '{p}' (format parquet)")
    return {"con": con, "paths": {"w": p}}


PATTERNS = ["%special%requests%", "special%requests", "%special%requests", "special%requests%", "a%b%", "%a%b", "a%b", "a%a", "aa%a", "%a%a%", "%a%%b%", "a%%b",
            "%café%", "%é%é%", "%naïve%café", "%X%X%", "a%X%c", "%c%b%a%", "ab%ab", "%ab%ab%", "%%a%b%%", "a_b%c", "%a%_b%"]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_like_with_several_literals_equals_duckdb(mx, words, pattern):
    esc = pattern.replace("'", "''")
    check(mx, f"SELECT id, s LIKE '{esc}' AS m, s NOT LIKE '{esc}' AS n FROM w ORDER BY id", words["con"], files={"w": words["paths"]["w"]}, ordered=True)
    check(mx, f"SELECT count(*) FROM w WHERE s LIKE '{esc}'", words["con"], files={"w": words["paths"]["w"]})


# a star from the largest table: unique keys into every dimension, and an equality between two dimension groups checked on the joined values (TPC-H Q5's shape)
STARS = [
    "SELECT da.an, count(*), sum(f.x) FROM f, da, db WHERE f.a = da.a AND f.b = db.b AND da.ak = db.bk GROUP BY da.an ORDER BY da.an",
    "SELECT nat.nname, sum(f.x) FROM f, da, db, nat WHERE f.a = da.a AND f.b = db.b AND da.ak = db.bk AND da.ak = nat.nk AND f.d > date '2021-01-01' GROUP BY nat.nname ORDER BY nat.nname",
    "SELECT db.bk, count(*) FROM f, da, db WHERE f.a = da.a AND f.b = db.b AND da.ak = db.bk AND da.seg = 2 AND db.w > 3 GROUP BY db.bk ORDER BY db.bk",
    "SELECT count(*), min(f.d), max(f.x) FROM f, da, db WHERE f.a = da.a AND f.b = db.b AND da.ak = db.bk AND f.s = 'blue'",
    # a dimension of a dimension, and nothing selected from the second group
    "SELECT nat.nname, count(*) FROM f, db, da, nat WHERE f.b = db.b AND db.bk = nat.nk AND f.a = da.a AND da.seg = 1 GROUP BY nat.nname ORDER BY nat.nname",
]


@pytest.mark.parametrize("sql", STARS)
def test_a_star_from_the_fact_table_equals_duckdb(mx, data, sql):
    res = q(mx, data, sql, [t for t in ["f", "da", "db", "nat"] if f" {t}," in sql or f" {t} " in sql or sql.split(" WHERE ")[0].endswith(f" {t}")], ordered=True)
    note(res, "a star from f")
    note(res, "join path: fused: a star")


def test_a_dimension_with_repeated_keys_is_not_a_star(mx, data):
    # dup repeats its key: the planner's order stands, and the answer is the same
    res = q(mx, data, "SELECT da.seg, count(*), sum(f.x) FROM f, da, dup WHERE f.a = da.a AND dup.a = f.a AND dup.v < 3 GROUP BY da.seg ORDER BY da.seg", ["f", "da", "dup"], ordered=True)
    assert not [n for n in res["notes"] if "a star from" in n], res["notes"]


def test_a_star_with_an_empty_dimension_and_null_cross_keys(mx, data):
    q(mx, data, "SELECT count(*), sum(f.x) FROM f, nothing, db WHERE f.a = nothing.a AND f.b = db.b AND nothing.ak = db.bk", ["f", "nothing", "db"])
    q(mx, data, "SELECT count(*) FROM f, da, db WHERE f.a = da.a AND f.b = db.b AND da.ak = db.bk AND da.ak IS NULL", ["f", "da", "db"])


# the star's boundaries: STAR_RATIO = 3 (the fact table has at least 3x the rows of every other), and keys a sample calls unique that are not
@pytest.fixture(scope="session")
def bounds(tmp_path_factory):
    d = tmp_path_factory.mktemp("f202b")
    con = oracle_connection()
    con.execute("create table bf as select i as id, (i * 7) % 70000 as k70, (i * 11) % 60000 as k60, (i * 13) % 50 as ks, i % 17 as g, ((i % 400) * 0.5)::decimal(10,1) as x from range(200000) r(i)")
    con.execute("create table d70 as select i as k, 'a' || (i % 31) as an, i % 5 as seg from range(70000) r(i)")
    con.execute("create table d60 as select i as k, 'b' || (i % 29) as bn, i % 7 as seg from range(60000) r(i)")
    con.execute("create table ds as select i as k, 'c' || (i % 4) as cn from range(50) r(i)")
    # 40,000 keys, a handful of them repeated: a sample of the column can call it unique, the build then finds the repeats
    con.execute("create table dd as select case when i % 9973 = 0 then i - 1 else i end as k, 'd' || (i % 13) as dn from range(40000) r(i)")
    con.execute("create table bf2 as select i as id, (i * 7) % 40000 as kd, (i * 13) % 50 as ks, ((i % 300) * 0.5)::decimal(10,1) as x from range(300000) r(i)")
    paths = {}
    for t in ["bf", "d70", "d60", "ds", "dd", "bf2"]:
        paths[t] = d / f"{t}.parquet"
        con.execute(f"copy {t} to '{paths[t]}' (format parquet, row_group_size 40000)")
    return {"con": con, "paths": paths}


def test_just_under_the_star_ratio_the_planners_order_stands(mx, bounds):
    # 200,000 rows against a 70,000-row dimension: 2.86x, under 3
    sql = "SELECT d70.an, ds.cn, count(*), sum(bf.x) FROM bf, d70, ds WHERE bf.k70 = d70.k AND bf.ks = ds.k GROUP BY d70.an, ds.cn ORDER BY d70.an, ds.cn"
    res = q(mx, bounds, sql, ["bf", "d70", "ds"], ordered=True)
    assert not [n for n in res["notes"] if "a star from" in n], res["notes"]


def test_over_the_star_ratio_the_fact_table_probes_every_dimension(mx, bounds):
    # 200,000 rows against a 60,000-row dimension: 3.33x
    sql = "SELECT d60.bn, ds.cn, count(*), sum(bf.x) FROM bf, d60, ds WHERE bf.k60 = d60.k AND bf.ks = ds.k GROUP BY d60.bn, ds.cn ORDER BY d60.bn, ds.cn"
    res = q(mx, bounds, sql, ["bf", "d60", "ds"], ordered=True)
    note(res, "a star from bf")


def test_keys_that_look_unique_but_repeat_fall_back_and_stay_correct(mx, bounds):
    sql = "SELECT dd.dn, ds.cn, count(*), sum(bf2.x) FROM bf2, dd, ds WHERE bf2.kd = dd.k AND bf2.ks = ds.k GROUP BY dd.dn, ds.cn ORDER BY dd.dn, ds.cn"
    res = q(mx, bounds, sql, ["bf2", "dd", "ds"], ordered=True)
    note(res, "star: dd has repeated keys: the joins run one by one")      # the plan was a star (the ndv said unique); the build found the repeats and the joins ran one by one


# LIKE with ESCAPE and ILIKE (the case-folded pattern takes the same literal matcher)
ESC_PATTERNS = [("a\\%b%", "\\"), ("%a\\_b%", "\\"), ("a!%b", "!"), ("%special%requests%", "\\"), ("a\\%%b", "\\")]


@pytest.mark.parametrize("pattern,esc", ESC_PATTERNS)
def test_like_with_escape_equals_duckdb(mx, words, pattern, esc):
    p = pattern.replace("'", "''")
    check(mx, f"SELECT id, s LIKE '{p}' ESCAPE '{esc}' AS m FROM w ORDER BY id", words["con"], files={"w": words["paths"]["w"]}, ordered=True)


ILIKE_PATTERNS = ["%SPECIAL%Requests%", "SPECIAL%REQUESTS", "A%B%", "%CAFÉ%", "%X%x%", "%É%é%", "a%B%c"]


@pytest.mark.parametrize("pattern", ILIKE_PATTERNS)
def test_ilike_with_several_literals_equals_duckdb(mx, words, pattern):
    p = pattern.replace("'", "''")
    check(mx, f"SELECT id, s ILIKE '{p}' AS m, s NOT ILIKE '{p}' AS n FROM w ORDER BY id", words["con"], files={"w": words["paths"]["w"]}, ordered=True)


# a correlated scalar subquery whose table is joined to others (TPC-H Q2's shape): D is cut to the outer rows' keys with an IN list
@pytest.mark.parametrize("sql", [
    "SELECT da.a, da.an FROM da WHERE da.seg = 2 AND da.a < 400 AND da.a = (SELECT min(f.a) FROM f, db WHERE f.a = da.a AND f.b = db.b AND db.bk < 12) ORDER BY da.a",
    "SELECT da.a, da.an FROM da WHERE da.seg = 1 AND da.a < 600 AND (SELECT min(f.x) FROM f, db WHERE f.a = da.a AND f.b = db.b AND db.bk < 12) > 40 ORDER BY da.a",
    "SELECT da.a FROM da WHERE da.a < 500 AND (SELECT count(*) FROM f, db WHERE f.a = da.a AND f.b = db.b AND db.bk > 20) >= 2 ORDER BY da.a",
])
def test_a_correlated_subquery_over_joined_tables_is_cut_to_the_outer_keys(mx, data, sql):
    res = q(mx, data, sql, ["da", "f", "db"], ordered=True)
    note(res, "cut to the keys of the")
