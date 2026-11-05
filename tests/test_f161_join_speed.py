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
    # a fact table of 300,000 rows (a probe side the engine counts first) and a dimension with unique keys, a small table with repeated keys
    con.execute("""create table bigf as select i as fid, case when i % 97 = 0 then null else (i * 37) % 6000 end as k, i % 11 as g,
        case when i % 50 = 0 then null else (i * 13) % 1000 end as x, date '2020-01-01' + (i % 400)::int as dt,
        case when i % 9 = 8 then null else (['apple','Banana','cherry','date','elder','fig','grape','kiwi'])[1 + i % 8] end as s,
        ((i % 700) * 0.01)::decimal(10,2) as m from range(300000) r(i)""")
    con.execute("create table dimk as select i as k, 'name' || (i % 40) as name, (i % 17) as seg, (i * 0.5)::decimal(10,1) as w from range(5000) r(i)")
    con.execute("create table repg as select i % 11 as g, i as rid, (i % 6) as v from range(2000) r(i)")                                         # 11 keys, about 180 rows each
    con.execute("create table repg2 as select i % 11 as g, i as rid, (i % 6) as v from range(300) r(i)")                                         # 11 keys, about 27 rows each
    con.execute("create table smallg as select i as g, 'grp' || (i % 4) as gname from range(11) r(i)")
    paths = {}
    for t, rg in [("cust", 700), ("orders", 3000), ("skew", 2500), ("nothing", 100), ("bigc", 50000), ("bigo", 40000), ("bigo2", 40000), ("bigf", 100000), ("dimk", 2000), ("repg", 500), ("repg2", 100), ("smallg", 100)]:
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


FIRST = "probe filter first"
SELECTIVE = [
    "SELECT d.name, count(*), sum(f.x), sum(f.m) FROM bigf f JOIN dimk d ON f.k = d.k WHERE f.x < 50 GROUP BY d.name",
    "SELECT d.seg, f.g, count(*), min(f.dt), max(d.w) FROM bigf f JOIN dimk d ON f.k = d.k WHERE f.x < 50 AND f.g > 2 GROUP BY d.seg, f.g",
    "SELECT count(*), sum(f.x) FROM bigf f JOIN dimk d ON f.k = d.k WHERE f.s LIKE 'ap%' AND f.dt > date '2020-06-01' AND f.x > 900",
    "SELECT d.seg, count(*) FROM bigf f JOIN dimk d ON f.k = d.k WHERE f.x IS NULL OR f.x > 985 GROUP BY d.seg",
    "SELECT count(*), count(d.k), sum(f.x) FROM bigf f LEFT JOIN dimk d ON f.k = d.k WHERE f.x < 40",
    "SELECT f.g, count(*), sum(r.v), sum(f.x) FROM bigf f JOIN repg2 r ON f.g = r.g WHERE f.x < 20 GROUP BY f.g",
    "SELECT count(*), sum(r.v) FROM bigf f LEFT JOIN repg2 r ON f.g = r.g WHERE f.x < 20 AND f.s = 'kiwi'",
    "SELECT count(*), sum(f.x) FROM bigf f WHERE f.k IN (SELECT k FROM dimk WHERE seg < 5) AND f.x < 30",
    "SELECT count(*), sum(f.x) FROM bigf f WHERE NOT EXISTS (SELECT 1 FROM dimk d WHERE d.k = f.k) AND f.x < 30",
]


@pytest.mark.parametrize("sql", SELECTIVE)
def test_a_selective_probe_filter_runs_before_the_probe(mx, data, sql):
    res = q(mx, data, sql, ["bigf", "dimk", "repg2"], extra=BUILD_RIGHT)
    assert note(res, "join probe filter").endswith("no filtered copy of it is written"), res["notes"]
    assert FIRST in " ".join(res["notes"]), res["notes"]
    no_note(res, "join path: materialized")


@pytest.mark.parametrize("threads", ["1", "3"])
def test_probe_filter_first_with_other_thread_counts(mx, data, threads):
    q(mx, data, SELECTIVE[0], ["bigf", "dimk"], extra=["--threads", threads])
    q(mx, data, SELECTIVE[5], ["bigf", "repg2"], extra=["--threads", threads, *BUILD_RIGHT])


def test_a_filter_that_most_rows_pass_keeps_the_filter_after_the_probe(mx, data):
    res = q(mx, data, "SELECT d.name, count(*), sum(f.x) FROM bigf f JOIN dimk d ON f.k = d.k WHERE f.x < 900 GROUP BY d.name", ["bigf", "dimk"])
    assert "join probe filter" in " ".join(res["notes"]), res["notes"]
    assert FIRST not in " ".join(res["notes"]), res["notes"]


def test_a_filter_across_both_tables_is_not_moved_before_the_probe(mx, data):
    res = q(mx, data, "SELECT d.name, count(*) FROM bigf f JOIN dimk d ON f.k = d.k WHERE f.x < 50 AND f.x > d.seg GROUP BY d.name", ["bigf", "dimk"])
    assert FIRST not in " ".join(res["notes"]), res["notes"]


def test_forced_materialized_with_a_selective_filter(mx, data):
    q(mx, data, SELECTIVE[0], ["bigf", "dimk"], extra=["--join", "materialize"])


STAR = [
    "SELECT d.name, sg.gname, count(*), sum(f.x) FROM bigf f JOIN dimk d ON f.k = d.k JOIN smallg sg ON f.g = sg.g WHERE f.x < 50 GROUP BY d.name, sg.gname",
    "SELECT sg.gname, count(*), sum(d.w) FROM bigf f JOIN dimk d ON f.k = d.k JOIN smallg sg ON f.g = sg.g WHERE f.x < 100 AND f.s LIKE '%e%' GROUP BY sg.gname",
]


@pytest.mark.parametrize("sql", STAR)
def test_a_star_probe_filter_runs_before_the_probes(mx, data, sql):
    res = q(mx, data, sql, ["bigf", "dimk", "smallg"])
    assert "join path: fused: a star" in " ".join(res["notes"]), res["notes"]
    assert FIRST in " ".join(res["notes"]), res["notes"]


def test_tpch_q12_and_q14_expand_in_the_front_stage():
    """TPC-H at SF 1 (DuckDB's dbgen, the gate's data): Q12 and Q14 build on the small filtered lineitem side, whose keys repeat (up to 4 and 5 rows of a key): each
    probe row is expanded to its partners in the fused pass; both equal DuckDB. (Q10 and Q3 keep their written order; the gate has them equal too.)"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import gate_common as gate
    from gate_compare import compare_with_duckdb
    mx = mx_binary()
    paths = gate.table_paths(1)
    con = gate.duck_views(paths)
    qs = gate.queries()
    for n in (12, 14):
        compare_with_duckdb(mx, qs[n], paths, con, label=f"TPC-H Q{n} SF1")
        files = {t: paths[t] for t in ("orders", "lineitem", "part")}
        out = run_mx(mx, ["sql", "--max-rows", "0", qs[n]], files)
        joined = " ".join(out["notes"])
        assert "expanded to its partners" in joined and "join path: materialized" not in joined, out["notes"]
