"""F-201: subqueries that become joins, at join speed. A step (IN / EXISTS / a correlated aggregate) that reads one table of a joined FROM, and whose
table D is small beside it, runs on that table before the joins (Q18's shape); a correlated step's D is cut to the keys the outer rows have (Q17, Q20,
Q21's shapes). Every answer equals DuckDB, with NULLs in keys and values, duplicates, and empty inner sides; the notes prove which path ran.
"""

import random

import pytest

from conftest import oracle_connection
from mx_oracle import check, mx_binary

PUSHED = "before the tables are joined"


@pytest.fixture(scope="session")
def mx():
    return mx_binary()


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("f201")
    con = oracle_connection()
    # customers, orders (a NULL customer, a customer that is not there), a fact table with repeated order keys, NULL keys and NULL quantities;
    # 50 orders (ok % 100 = 7) are heavy: their facts carry a large quantity
    con.execute("""create table cust as select i as ck, 'n' || (i % 7) as nation, case when i % 11 = 0 then null else 'c' || i end as name from range(300) r(i)""")
    con.execute("""create table ord as select i as ok, case when i % 97 = 0 then null when i % 41 = 0 then 9999 else (i * 7) % 300 end as ck,
        case when i % 13 = 0 then null else ((i % 500) * 1.5)::decimal(10,2) end as tot from range(5000) r(i)""")
    con.execute("""create table fact as select i as fid, case when i % 211 = 0 then null else (i * 3) % 5000 end as ok,
        case when i % 17 = 0 then null when (((i * 3) % 5000) % 100) = 7 then 60 + (i % 30) else 1 + (i % 9) end as qty, (i * 11) % 300 as cust
        from range(20000) r(i)""")
    con.execute("create table nofact as select * from fact where false")
    # Q20's shape: partsupp rows keyed by two columns, line items of the same pairs (some pairs have none, some line items have a NULL part)
    con.execute("""create table ps as select (i % 400) as pk, (i % 7) as sk, ((i * 13) % 90)::int as avail from range(2000) r(i) where (i % 400, i % 7) is not null""")
    con.execute("""create table li as select case when i % 97 = 0 then null else (i * 11) % 450 end as pk, (i % 7) as sk, (1 + i % 9) as qty,
        date '2020-01-01' + (i % 700)::int as d from range(30000) r(i)""")
    paths = {}
    for t in ["cust", "ord", "fact", "nofact", "ps", "li"]:
        paths[t] = d / f"{t}.parquet"
        con.execute(f"copy {t} to '{paths[t]}' (format parquet, row_group_size 4000)")
    return {"con": con, "paths": paths}


def q(mx, data, sql, tables, ordered=False):
    return check(mx, sql, data["con"], files={t: data["paths"][t] for t in tables}, ordered=ordered)


def note(res, text):
    hits = [n for n in res["notes"] if text in n]
    assert hits, (text, res["notes"])


def no_note(res, text):
    assert not [n for n in res["notes"] if text in n], (text, res["notes"])


Q18_SHAPE = """SELECT c.name, o.ok, o.tot, sum(f.qty) AS s FROM cust c, ord o, fact f
    WHERE o.ok IN (SELECT ok FROM fact GROUP BY ok HAVING sum(qty) > {t}) AND c.ck = o.ck AND o.ok = f.ok
    GROUP BY c.name, o.ok, o.tot ORDER BY o.ok"""


def test_an_in_with_a_small_inner_side_runs_on_its_table_before_the_joins(mx, data):
    res = q(mx, data, Q18_SHAPE.format(t=150), ["cust", "ord", "fact"], ordered=True)
    assert len(res["rows"]) > 0
    note(res, PUSHED)


def test_the_same_in_over_an_empty_inner_side_keeps_nothing(mx, data):
    res = q(mx, data, Q18_SHAPE.format(t=10000000), ["cust", "ord", "fact"], ordered=True)
    assert res["rows"] == []


def test_not_in_with_a_null_in_the_inner_side_keeps_no_row(mx, data):
    # fact.ok holds NULLs, so no order is NOT IN the list
    res = q(mx, data, """SELECT count(*), sum(o.tot) FROM cust c, ord o
        WHERE o.ok NOT IN (SELECT ok FROM fact GROUP BY ok HAVING sum(qty) > 150 OR ok IS NULL) AND c.ck = o.ck""", ["cust", "ord", "fact"])
    assert res["rows"][0][0] == 0


def test_not_in_without_nulls_in_the_inner_side(mx, data):
    res = q(mx, data, """SELECT c.nation, count(*), sum(o.tot) FROM cust c, ord o
        WHERE o.ok NOT IN (SELECT ok FROM fact WHERE ok IS NOT NULL GROUP BY ok HAVING sum(qty) > 150) AND c.ck = o.ck GROUP BY c.nation ORDER BY c.nation""",
          ["cust", "ord", "fact"], ordered=True)
    assert len(res["rows"]) > 0


def test_a_correlated_aggregate_is_cut_to_the_keys_of_the_outer_rows(mx, data):
    res = q(mx, data, """SELECT c.nation, count(*) FROM cust c, ord o
        WHERE c.ck = o.ck AND o.tot < (SELECT sum(f.qty) FROM fact f WHERE f.ok = o.ok AND f.qty > 80) GROUP BY c.nation ORDER BY c.nation""",
          ["cust", "ord", "fact"], ordered=True)
    note(res, "cut to the keys of the")


def test_a_count_zero_keeps_the_orders_the_inner_side_has_no_row_for(mx, data):
    res = q(mx, data, """SELECT c.nation, count(*) FROM cust c, ord o
        WHERE c.ck = o.ck AND (SELECT count(*) FROM fact f WHERE f.ok = o.ok AND f.qty > 80) = 0 GROUP BY c.nation ORDER BY c.nation""",
          ["cust", "ord", "fact"], ordered=True)
    assert len(res["rows"]) > 0


def test_a_table_of_a_left_join_is_not_filtered_before_the_join(mx, data):
    res = q(mx, data, """SELECT count(*), sum(o.tot) FROM cust c LEFT JOIN ord o ON c.ck = o.ck
        WHERE o.ok IN (SELECT ok FROM fact GROUP BY ok HAVING sum(qty) > 150)""", ["cust", "ord", "fact"])
    no_note(res, PUSHED)


def test_the_same_table_twice_under_two_aliases(mx, data):
    res = q(mx, data, """SELECT o1.ok, o2.ok FROM ord o1, ord o2
        WHERE o1.ok IN (SELECT ok FROM fact GROUP BY ok HAVING sum(qty) > 150) AND o1.ck = o2.ck AND o2.ok < 400 ORDER BY o1.ok, o2.ok""",
          ["ord", "fact"], ordered=True)
    assert len(res["rows"]) > 0


def test_exists_with_a_second_customer_value_per_order(mx, data):
    # one inequality (min / max of the inner side), a small inner side
    res = q(mx, data, """SELECT c.nation, count(*) FROM cust c, ord o
        WHERE c.ck = o.ck AND EXISTS (SELECT 1 FROM fact f WHERE f.ok = o.ok AND f.cust <> o.ck AND f.qty > 80) GROUP BY c.nation ORDER BY c.nation""",
          ["cust", "ord", "fact"], ordered=True)
    assert len(res["rows"]) > 0


def test_not_exists_over_an_inner_table_with_no_rows(mx, data):
    res = q(mx, data, """SELECT count(*) FROM cust c, ord o WHERE c.ck = o.ck AND NOT EXISTS (SELECT 1 FROM nofact f WHERE f.ok = o.ok AND f.cust <> o.ck)""",
            ["cust", "ord", "nofact"])
    assert res["rows"][0][0] > 0


SEMI = "SELECT count(*), sum(o.tot) FROM ord o WHERE o.ok < {m} AND {neg} EXISTS (SELECT 1 FROM fact f WHERE f.ok = o.ok AND f.qty > 5)"


@pytest.mark.parametrize("neg", ["", "NOT"])
def test_exists_with_a_filtered_inner_table_is_cut_to_the_few_outer_rows(mx, data, neg):
    res = q(mx, data, SEMI.format(m=300, neg=neg), ["ord", "fact"])
    note(res, "outer rows after the statement's filters")
    note(res, "cut to the keys of the")
    assert res["rows"][0][0] > 0


@pytest.mark.parametrize("neg", ["", "NOT"])
def test_exists_with_many_outer_rows_keeps_the_semi_join_with_the_inner_table(mx, data, neg):
    res = q(mx, data, SEMI.format(m=4000, neg=neg), ["ord", "fact"])
    no_note(res, "cut to the keys of the")


@pytest.mark.parametrize("neg", ["", "NOT"])
def test_exists_over_an_inner_table_with_no_rows_and_few_outer_rows(mx, data, neg):
    res = q(mx, data, "SELECT count(*) FROM ord o WHERE o.ok < 300 AND " + neg + " EXISTS (SELECT 1 FROM nofact f WHERE f.ok = o.ok AND f.qty > 5)", ["ord", "nofact"])
    assert res["rows"][0][0] == (0 if neg == "" else 300)


TWO_KEYS = """SELECT count(*), sum(ps.avail) FROM ps WHERE ps.pk IN (SELECT pk FROM ps WHERE avail > {a})
    AND ps.avail > (SELECT 0.5 * sum(li.qty) FROM li WHERE li.pk = ps.pk AND li.sk = ps.sk AND li.d >= DATE '2020-06-01' AND li.d < DATE '2021-06-01')"""


@pytest.mark.parametrize("a", [20, 80, 1000])
def test_a_correlation_on_two_columns_is_cut_to_the_first_key_and_still_matches_on_both(mx, data, a):
    res = q(mx, data, TWO_KEYS.format(a=a), ["ps", "li"])
    if a < 1000:
        note(res, "cut to the keys of the")


# ── random worlds ───────────────────────────────────────────────────────────────────────────────────

TEMPLATES = [
    "SELECT a.v, b.w, count(*), sum(a.x) FROM a, b WHERE a.k = b.k AND a.k IN (SELECT k FROM c GROUP BY k HAVING sum(x) > {t}) GROUP BY a.v, b.w ORDER BY 1, 2",
    "SELECT a.v, count(*), sum(b.w) FROM a, b WHERE a.k = b.k AND a.k NOT IN (SELECT k FROM c WHERE k IS NOT NULL GROUP BY k HAVING sum(x) > {t}) GROUP BY a.v ORDER BY 1",
    "SELECT a.v, count(*), sum(b.w) FROM a, b WHERE a.k = b.k AND a.k NOT IN (SELECT k FROM c GROUP BY k HAVING sum(x) > {t}) GROUP BY a.v ORDER BY 1",
    "SELECT a.v, count(*) FROM a, b WHERE a.k = b.k AND a.x > (SELECT max(x) FROM c WHERE c.k = a.k AND c.x > {t}) GROUP BY a.v ORDER BY 1",
    "SELECT a.v, count(*) FROM a, b WHERE a.k = b.k AND NOT EXISTS (SELECT 1 FROM c WHERE c.k = a.k AND c.x <> a.x AND c.x > {t}) GROUP BY a.v ORDER BY 1",
    "SELECT a.v, count(*) FROM a, b WHERE a.k = b.k AND (SELECT count(*) FROM c WHERE c.k = a.k AND c.x > {t}) = 0 GROUP BY a.v ORDER BY 1",
    "SELECT a.v, count(*) FROM a, b WHERE a.k = b.k AND EXISTS (SELECT 1 FROM c WHERE c.k = a.k AND c.x <> a.x AND c.x > {t}) GROUP BY a.v ORDER BY 1",
    "SELECT a.v, count(*) FROM a, b WHERE a.k = b.k AND a.x IN (SELECT x FROM c WHERE c.k = a.k AND c.x > {t}) GROUP BY a.v ORDER BY 1",
    "SELECT a.v, count(*), sum(a.x) FROM a WHERE a.x < 8 AND EXISTS (SELECT 1 FROM c WHERE c.k = a.k AND c.x > {t}) GROUP BY a.v ORDER BY 1",
    "SELECT a.v, count(*), sum(a.x) FROM a WHERE a.x < 8 AND NOT EXISTS (SELECT 1 FROM c WHERE c.k = a.k AND c.x > {t}) GROUP BY a.v ORDER BY 1",
    "SELECT a.v, count(*) FROM a WHERE a.x < 8 AND a.v > (SELECT count(*) FROM c WHERE c.k = a.k AND c.x > {t}) GROUP BY a.v ORDER BY 1",
]


@pytest.fixture(scope="session")
def worlds(tmp_path_factory):
    d = tmp_path_factory.mktemp("f201w")
    out = {}
    for seed in range(6):
        rnd = random.Random(seed)
        con = oracle_connection()
        nk = rnd.choice([40, 400, 3000])

        def key():
            return "null" if rnd.random() < 0.05 else str(rnd.randrange(nk))

        def num(null_p):
            return "null" if rnd.random() < null_p else str(rnd.randrange(100))

        con.execute("create table a(k bigint, v integer, x integer)")
        con.execute("create table b(k bigint, w integer)")
        con.execute("create table c(k bigint, x integer)")
        con.execute("insert into a values " + ",".join(f"({key()}, {rnd.randrange(6)}, {num(0.1)})" for _ in range(3000)))
        con.execute("insert into b values " + ",".join(f"({key()}, {num(0.1)})" for _ in range(2000)))
        nc = rnd.choice([0, 500, 9000])
        if nc:
            con.execute("insert into c values " + ",".join(f"({key()}, {num(0.1)})" for _ in range(nc)))
        paths = {}
        for t in "abc":
            paths[t] = d / f"{seed}_{t}.parquet"
            con.execute(f"copy {t} to '{paths[t]}' (format parquet, row_group_size 1000)")
        out[seed] = {"con": con, "paths": paths}
    return out


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("ti", range(len(TEMPLATES)))
def test_random_worlds_equal_duckdb(mx, worlds, seed, ti):
    for t in (30, 90, 99999):
        sql = TEMPLATES[ti].format(t=t)
        w = worlds[seed]
        check(mx, sql, w["con"], files={n: w["paths"][n] for n in "abc"}, ordered=True)


# ── text functions over a dictionary (Q22's substring): the same call written twice is answered once, ASCII words are cut by bytes ──────────

@pytest.fixture(scope="session")
def words(tmp_path_factory):
    d = tmp_path_factory.mktemp("f201t")
    con = oracle_connection()
    con.execute("""create table w as select i as id,
        case when i % 17 = 0 then null when i % 7 = 0 then 'é' || (i * 31 % 1000)::varchar when i % 11 = 0 then '日本' || (i % 50)::varchar || 'x'
             when i % 13 = 0 then '' else lpad((i * 7919 % 100000)::varchar, (1 + i % 9)::int, '0') end as s,
        i % 5 as g from range(40000) r(i)""")
    path = d / "w.parquet"
    con.execute(f"copy w to '{path}' (format parquet, row_group_size 8000)")
    return {"con": con, "paths": {"w": path}}


TEXT_QUERIES = [
    "SELECT substring(s FROM 1 FOR 2) AS p, count(*) FROM w WHERE substring(s FROM 1 FOR 2) IN ('00', '12', 'é1', '日本') GROUP BY substring(s FROM 1 FOR 2) ORDER BY 1",
    "SELECT substr(s, 2, 3) AS p, count(*), min(id) FROM w GROUP BY 1 ORDER BY 1",
    "SELECT substr(s, -3) AS p, count(*) FROM w GROUP BY 1 ORDER BY 1",
    "SELECT substr(s, -3, 2) AS p, substr(s, 0, 2) AS q, substr(s, 3, -2) AS r, count(*) FROM w GROUP BY 1, 2, 3 ORDER BY 1, 2, 3",
    "SELECT left(s, 2) AS a, right(s, 2) AS b, count(*) FROM w GROUP BY 1, 2 ORDER BY 1, 2",
    "SELECT left(s, -2) AS a, right(s, -2) AS b, left(s, 0) AS c, count(*) FROM w GROUP BY 1, 2, 3 ORDER BY 1, 2, 3",
    "SELECT g, count(DISTINCT substr(s, 1, 3)), count(*) FROM w WHERE substr(s, 1, 3) <> '000' GROUP BY g ORDER BY g",
]


@pytest.mark.parametrize("sql", TEXT_QUERIES)
def test_substring_left_right_equal_duckdb(mx, words, sql):
    res = check(mx, sql, words["con"], files={"w": words["paths"]["w"]}, ordered=True)
    assert len(res["rows"]) > 1


def test_the_same_substring_written_three_times_is_answered_once(mx, words):
    res = check(mx, "SELECT substring(s FROM 1 FOR 2) AS p, count(*) FROM w WHERE substring(s FROM 1 FOR 2) IN ('00', '12') GROUP BY substring(s FROM 1 FOR 2) ORDER BY 1",
                words["con"], files={"w": words["paths"]["w"]}, ordered=True)
    tabs = [n for n in res["notes"] if n.startswith("text:")]
    assert len(tabs) == 1 and tabs[0].startswith("text: 1 per-word table over"), tabs
