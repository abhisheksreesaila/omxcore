"""F-161: join speed. An aggregate over a many-to-many join (and over a LEFT join with repeated build keys) runs inside the fused front stage:
each probe row is expanded to its partners there, no pairs are materialized. A cheap filter of the probe side runs before the probe. Every
answer is equal to DuckDB; the notes (the engine's own EXPLAIN) prove which path ran. Speed is not measured here.
"""

import pytest

from conftest import oracle_connection
from mx_oracle import check, mx_binary, run_mx


@pytest.fixture(scope="session")
def mx():
    return mx_binary()


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("f161")
    con = oracle_connection()
    con.execute("""create table cust as select i as id, 'n' || (i % 7) as nation, (i % 13) as seg,
        case when i % 11 = 0 then null else ((i * 3) % 50)::decimal(10,2) end as bal from range(6000) r(i)""")
    # 1..many: orders (the smaller table, so the build side) repeat their customer key about 2.7 times; customers 1500.. have none; some orders have no customer
    # (a NULL key, or one that is missing), dates, nulls in price
    con.execute("""create table orders as select i as oid,
        case when i % 53 = 0 then null when i % 29 = 0 then 100000 + i else (i * 7) % 1500 end as cust,
        case when i % 17 = 0 then null else ((i % 300) * 0.25)::decimal(12,2) end as price,
        date '2021-01-01' + (i % 500)::int as odate, i % 4 as pri,
        case when i % 9 = 0 then null else (['A','b','C','d'])[1 + i % 4] end as flag
        from range(4000) r(i)""")
    # one key repeated 5000 times (past the expansion limit) among otherwise light keys
    con.execute("""create table skew as select case when i % 2 = 0 then 7 else i % 3000 end as cust, i as sid, i % 10 as v from range(5000) r(i)""")
    con.execute("create table nothing as select * from orders where false")
    # many groups: past 65,536 the kernel stages a morsel's rows (a slot holds MORSEL rows, so a morsel of probe rows must not put out more)
    con.execute("create table bigc as select i as id, i % 9 as seg, ((i % 1000) * 0.5)::double as w from range(200000) r(i)")
    con.execute("create table bigo as select i as oid, (i * 7) % 3000 as cust, ((i % 977) * 0.1)::decimal(12,1) as price, i % 5 as pri from range(150000) r(i)")   # 50 orders per customer
    con.execute("create table bigo2 as select i as oid, (i * 7) % 70001 as cust, ((i % 977) * 0.1)::decimal(12,1) as price from range(120000) r(i)")                # 1 or 2 per customer
    paths = {}
    for t, rg in [("cust", 700), ("orders", 3000), ("skew", 2500), ("nothing", 100), ("bigc", 50000), ("bigo", 40000), ("bigo2", 40000)]:
        paths[t] = d / f"{t}.parquet"
        con.execute(f"copy {t} to '{paths[t]}' (format parquet, row_group_size {rg})")
    return {"con": con, "paths": paths}


BUILD_RIGHT = ["--build", "right"]       # the repeated side builds (the planner builds on the smaller side after the filters)


def q(mx, data, sql, tables, extra=(), ordered=False):
    return check(mx, sql, data["con"], files={t: data["paths"][t] for t in tables}, ordered=ordered, extra=extra)


def note(res, text):
    hits = [n for n in res["notes"] if text in n]
    assert hits, (text, res["notes"])
    return hits[0]


def no_note(res, text):
    assert not [n for n in res["notes"] if text in n], (text, res["notes"])


EXPANDED = [
    # the probe side is the larger table, the build side repeats its keys; aggregates of every kind
    "SELECT c.nation, count(*), sum(o.price) FROM cust c JOIN orders o ON c.id = o.cust GROUP BY c.nation",
    "SELECT count(*), sum(o.price), min(o.odate), max(o.price), avg(o.price) FROM cust c JOIN orders o ON c.id = o.cust",
    "SELECT c.seg, o.pri, count(*), sum(c.bal), count(o.price), count(c.bal) FROM cust c JOIN orders o ON c.id = o.cust GROUP BY c.seg, o.pri",
    "SELECT c.id, count(*), sum(o.price) FROM cust c JOIN orders o ON c.id = o.cust GROUP BY c.id",
    "SELECT o.flag, count(*), sum(c.bal) FROM cust c JOIN orders o ON c.id = o.cust WHERE c.seg < 6 AND o.pri > 0 GROUP BY o.flag",
    "SELECT c.nation, count(*) FROM cust c JOIN orders o ON c.id = o.cust WHERE c.bal > 10 AND o.price < 50 AND o.odate > date '2021-06-01' GROUP BY c.nation",
    "SELECT sum(o.price * c.seg) FROM cust c JOIN orders o ON c.id = o.cust WHERE o.price > c.bal",
    "SELECT count(*) FROM cust c JOIN orders o ON c.id = o.cust AND c.seg = o.pri",
    "SELECT a.seg, count(*) FROM cust a JOIN cust b ON a.seg = b.seg AND a.id < b.id WHERE b.id < 300 GROUP BY a.seg",
]

LEFT = [
    "SELECT c.nation, count(*), count(o.oid), sum(o.price) FROM cust c LEFT JOIN orders o ON c.id = o.cust GROUP BY c.nation",
    "SELECT count(*), count(o.oid), sum(o.price), min(o.odate) FROM cust c LEFT JOIN orders o ON c.id = o.cust",
    "SELECT c.id, count(o.oid), sum(o.price) FROM cust c LEFT JOIN orders o ON c.id = o.cust GROUP BY c.id",
    "SELECT c.seg, o.flag, count(*), sum(c.bal) FROM cust c LEFT JOIN orders o ON c.id = o.cust AND o.pri = 1 WHERE c.id < 2000 GROUP BY c.seg, o.flag",
    "SELECT c.nation, count(*) FROM cust c LEFT JOIN orders o ON c.id = o.cust WHERE c.seg > 2 GROUP BY c.nation",
    "SELECT count(*) FROM cust c LEFT JOIN orders o ON c.id = o.cust WHERE o.oid IS NULL",
]


@pytest.mark.parametrize("sql", EXPANDED + LEFT)
def test_aggregates_over_repeated_build_keys_are_fused(mx, data, sql):
    res = q(mx, data, sql, ["cust", "orders"], extra=BUILD_RIGHT)
    assert "expanded to its partners" in note(res, "join path"), res["notes"]
    no_note(res, "materialized")


@pytest.mark.parametrize("sql", EXPANDED[:3] + LEFT[:3])
def test_forced_materialized_is_the_same_answer(mx, data, sql):
    res = q(mx, data, sql, ["cust", "orders"], extra=["--join", "materialize", *BUILD_RIGHT])
    assert note(res, "join path").startswith("join path: materialized:"), res["notes"]


@pytest.mark.parametrize("sql", [EXPANDED[0], EXPANDED[3], LEFT[0], LEFT[2]])
@pytest.mark.parametrize("threads", ["1", "3"])
def test_expansion_with_other_thread_counts(mx, data, sql, threads):
    q(mx, data, sql, ["cust", "orders"], extra=["--threads", threads, *BUILD_RIGHT])


@pytest.mark.parametrize("sql", [EXPANDED[0], LEFT[0], LEFT[3]])
def test_a_streamed_probe_expands_too(mx, data, sql):
    res = q(mx, data, sql, ["cust", "orders"], extra=["--stream", "always", *BUILD_RIGHT])
    assert "expanded to its partners" in note(res, "join path"), res["notes"]
    assert "chunk by chunk" in note(res, "join path"), res["notes"]


def test_rows_stay_materialized(mx, data):
    res = q(mx, data, "SELECT c.id, o.oid FROM cust c JOIN orders o ON c.id = o.cust WHERE c.id < 40", ["cust", "orders"], extra=BUILD_RIGHT)
    assert note(res, "join path").startswith("join path: materialized:"), res["notes"]


def test_a_hot_key_past_the_limit_is_materialized(mx, data):
    res = q(mx, data, "SELECT s.v, count(*), sum(c.bal) FROM cust c JOIN skew s ON c.id = s.cust GROUP BY s.v", ["cust", "skew"], extra=BUILD_RIGHT)
    note(res, "repeated keys")
    assert "one up to" in note(res, "join path"), res["notes"]
    assert note(res, "join path").startswith("join path: materialized:"), res["notes"]
    res = q(mx, data, "SELECT count(*), count(s.sid) FROM cust c LEFT JOIN skew s ON c.id = s.cust", ["cust", "skew"])
    assert note(res, "join path").startswith("join path: materialized:"), res["notes"]


def test_empty_and_missing_partners(mx, data):
    q(mx, data, "SELECT count(*), sum(o.price) FROM cust c JOIN nothing o ON c.id = o.cust", ["cust", "nothing"])
    q(mx, data, "SELECT c.nation, count(o.oid) FROM cust c LEFT JOIN nothing o ON c.id = o.cust GROUP BY c.nation", ["cust", "nothing"])
    q(mx, data, "SELECT count(*) FROM cust c JOIN orders o ON c.id = o.cust WHERE c.id < 0", ["cust", "orders"])


MANY_GROUPS = [
    "SELECT c.id, count(*), sum(o.price) FROM bigc c JOIN bigo o ON c.id = o.cust GROUP BY c.id",
    "SELECT c.id, count(o.oid), sum(o.price), sum(c.w) FROM bigc c LEFT JOIN bigo2 o ON c.id = o.cust GROUP BY c.id",
    "SELECT c.id, count(*), min(o.price), max(o.pri) FROM bigc c LEFT JOIN bigo o ON c.id = o.cust GROUP BY c.id",
    "SELECT o.oid, count(*), sum(c.w) FROM bigc c JOIN bigo o ON c.id = o.cust GROUP BY o.oid",
    "SELECT c.seg, count(*), sum(o.price), avg(c.w) FROM bigc c LEFT JOIN bigo o ON c.id = o.cust GROUP BY c.seg",
]


@pytest.mark.parametrize("group", ["auto", "dense", "hashed"])
@pytest.mark.parametrize("sql", MANY_GROUPS)
def test_many_groups_and_staged_morsels(mx, data, sql, group):
    res = q(mx, data, sql, ["bigc", "bigo", "bigo2"], extra=[*BUILD_RIGHT, "--group", group])
    assert "expanded to its partners" in note(res, "join path"), res["notes"]
