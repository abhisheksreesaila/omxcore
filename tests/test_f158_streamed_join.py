"""F-158: a Parquet scan on the probe side of a join streams chunk by chunk (never read into a table), a big GROUP BY over a scan merges its
row groups' partials, and a build side past the join's memory budget raises a readable error. Every answer is equal to DuckDB's on the same
files and to the same query with the scan read whole (`--stream never`); the notes prove which path ran.
"""
import subprocess

import duckdb
import pytest

from mx_oracle import check, mx_binary


@pytest.fixture(scope="session")
def mx():
    return mx_binary()


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("f158")
    con = duckdb.connect()
    con.execute("""create table facts as select
        i as fid,
        case when i % 97 = 0 then null else (i * 37) % 2600 end as k,
        (i * 13) % 11 as g,
        case when i % 50 = 0 then null else (i * 13) % 1000 - 500 end as x,
        case when i % 9 = 8 then null else (['West','east','North','South','NORTH','mango','Zebra','apple'])[1 + i % 8] end as s
        from range(60000) r(i)""")
    con.execute("""create table d1 as select i as k, 'name' || (i % 50) as name, (i * 1.0001)::decimal(10,4) as w from range(2000) r(i)""")
    con.execute("""create table d2 as select i as g, 'grp' || i as gname from range(11) r(i)""")
    con.execute("create table big as select i as id, (i * 7919) % 1000003 as k, i % 1000 as v from range(1300000) r(i)")
    con.execute("""create table f2 as select i as id,
        case when i % 13 = 0 then null else 'key' || (i % 300) end as s,
        case when i % 11 = 0 then null else i % 20 end as a, (i * 7) % 15 as b,
        ((i % 400) * 0.25)::decimal(10,2) as d, i % 100 as x from range(40000) r(i)""")
    con.execute("""create table m2 as select 'key' || i as s, i % 20 as a, i % 15 as b, (i * 0.25)::decimal(12,4) as d, i as mid from range(300) r(i)""")
    con.execute("create table m2n as select * from m2 union all select null, 3, 4, null, 999")   # a NULL in the build side (NOT IN)
    con.execute("create table rep as select i % 40 as k, i as v from range(5000) r(i)")            # repeated keys: the streamed join falls back to materialized
    con.execute("create table bigb as select i as k, i as w from range(1000000) r(i)")
    for name, vals in [("ov", (3000000000000000000, 6000000000000000000, 9000000000000000000)), ("ovfit", (3000000000000000000, 3000000000000000000, 1))]:
        # 3 row groups of 50,000 rows; key 1 once per group (row 0 of it) with the group's value, every other key distinct with value 1
        con.execute(f"create table {name} as select case when i % 50000 = 0 then 1 else 100 + i end as k, "
                    f"case when i % 50000 = 0 then (case i // 50000 when 0 then {vals[0]} when 1 then {vals[1]} else {vals[2]} end)::bigint else 1::bigint end as v from range(150000) r(i)")
    paths = {}
    for t, rg in [("facts", 7000), ("d1", 700), ("d2", 100), ("big", 100000), ("f2", 5000), ("m2", 100), ("m2n", 100), ("rep", 1000), ("bigb", 100000), ("ov", 50000), ("ovfit", 50000)]:
        paths[t] = d / f"{t}.parquet"
        con.execute(f"copy {t} to '{paths[t]}' (format parquet, row_group_size {rg})")
    return {"con": con, "paths": paths}


def q(mx, data, sql, tables, extra=()):
    return check(mx, sql, data["con"], files={t: data["paths"][t] for t in tables}, ordered=False, extra=extra)


def note(res, text):
    hits = [n for n in res["notes"] if text in n]
    assert hits, (text, res["notes"])
    return hits[0]


STREAMED = [
    "SELECT d.name, count(*), sum(f.x) FROM facts f JOIN d1 d ON f.k = d.k GROUP BY d.name",
    "SELECT f.g, count(*), sum(d.w) FROM facts f JOIN d1 d ON f.k = d.k WHERE f.x > 0 GROUP BY f.g",
    "SELECT d.name, count(*), sum(f.x) FROM facts f LEFT JOIN d1 d ON f.k = d.k GROUP BY d.name",
    "SELECT count(*), sum(f.x) FROM facts f WHERE f.k IN (SELECT k FROM d1 WHERE k < 900)",
    "SELECT count(*), sum(f.x) FROM facts f WHERE NOT EXISTS (SELECT 1 FROM d1 d WHERE d.k = f.k)",
]


@pytest.mark.parametrize("sql", STREAMED)
def test_scan_probe_streams_and_equals_duckdb(mx, data, sql):
    res = q(mx, data, sql, ["facts", "d1"], extra=["--stream", "always"])
    assert "chunk by chunk" in note(res, "join path"), res["notes"]
    q(mx, data, sql, ["facts", "d1"], extra=["--stream", "never"])


def test_star_over_a_scan_probe_streams(mx, data):
    sql = "SELECT d2.gname, d1.name, count(*), sum(f.x) FROM facts f JOIN d1 ON f.k = d1.k JOIN d2 ON f.g = d2.g GROUP BY d2.gname, d1.name"
    res = q(mx, data, sql, ["facts", "d1", "d2"], extra=["--stream", "always"])
    assert "a star" in note(res, "join path") and "chunk by chunk" in note(res, "join path"), res["notes"]


def test_group_by_one_million_groups_over_a_scan_equals_duckdb(mx, data):
    """1,000,003 possible keys over 1.3M rows in 13 row groups: every group's partial merges across row groups (exact int64 sums)."""
    sql = "SELECT count(*), sum(s), max(c), min(s) FROM (SELECT k, sum(v) AS s, count(*) AS c FROM big GROUP BY k) t"
    q(mx, data, sql, ["big"], extra=["--stream", "always"])
    res = q(mx, data, "SELECT k, sum(v), count(*) FROM big GROUP BY k HAVING sum(v) > 1500", ["big"], extra=["--stream", "always"])
    assert "13 row groups" in note(res, "source: big") and "1,000,003 groups" in note(res, "having:"), res["notes"]
    assert note(res, "group: hashed"), res["notes"]
    assert 0 < res["row_count"] < 200000, res["row_count"]


KEYS = [
    ("text keys", "SELECT m.mid % 7, count(*), sum(f.x) FROM f2 f JOIN m2 m ON f.s = m.s GROUP BY m.mid % 7"),
    ("packed multi-column key with NULLs", "SELECT count(*), sum(f.x), sum(m.mid) FROM f2 f JOIN m2 m ON f.a = m.a AND f.b = m.b AND f.s = m.s"),
    ("decimal rescale key", "SELECT count(*), sum(f.x), sum(m.mid) FROM f2 f JOIN m2 m ON f.d = m.d"),
    ("left join, NULL keys stay", "SELECT count(*), count(m.mid), sum(f.x) FROM f2 f LEFT JOIN m2 m ON f.a = m.a AND f.b = m.b AND f.s = m.s"),
]


@pytest.mark.parametrize("name,sql", KEYS, ids=[k[0] for k in KEYS])
def test_key_shapes_stream_and_equal_duckdb(mx, data, name, sql):
    res = q(mx, data, sql, ["f2", "m2"], extra=["--stream", "always"])
    assert "chunk by chunk" in note(res, "join path") or "join path: materialized" in note(res, "join path"), res["notes"]
    q(mx, data, sql, ["f2", "m2"], extra=["--stream", "never"])


def test_not_in_with_a_NULL_in_the_build_side_equals_duckdb(mx, data):
    """A NULL among the build keys makes NOT IN unknown for every probe row that has no match (DuckDB: none kept)."""
    for t in ["m2", "m2n"]:
        q(mx, data, f"SELECT count(*) FROM f2 WHERE f2.s NOT IN (SELECT s FROM {t})", ["f2", t], extra=["--stream", "always"])


def test_repeated_build_keys_fall_back_to_materialized_without_a_probe_budget(mx, data):
    """The probe scan (1M rows) is not the build side: the fallback must not charge it to the join budget (it did, and blamed the wrong table)."""
    sql = "SELECT count(*) FROM rep a JOIN bigb b ON a.k = b.k"
    for stream in ["never", "always"]:
        res = q(mx, data, sql, ["rep", "bigb"], extra=["--stream", stream, "--join-budget", "100000"])
        assert res["rows"] == [[5000]], res
        if stream == "always":
            assert "join path: materialized" in note(res, "join path"), res["notes"]


def _sum_k1(mx, data, table, stream):
    cmd = [str(mx), "sql", "--stream", stream, "--group", "hashed", "-t", f"{table}={data['paths'][table]}", f"SELECT sum(v) FROM {table} WHERE k = 1 GROUP BY k"]
    return subprocess.run(cmd, capture_output=True).stdout.decode()


def test_sum_overflow_across_row_groups_raises(mx, data):
    """Key 1 sums 3e18, 6e18, 9e18 in three row groups: each partial fits int64, the total (1.8e19) does not. Hashed groups run `PartAcc` across the
    row groups; the merge raises instead of wrapping (DuckDB widens to HUGEINT and answers). Read whole, the one pass overflows the same way."""
    out = _sum_k1(mx, data, "ov", "always")
    assert "passed the int64 range" in out and '"rows"' not in out, out[-300:]
    cmd = [str(mx), "sql", "--stream", "always", "--group", "hashed", "-t", f"ov={data['paths']['ov']}", "SELECT sum(v) FROM ov GROUP BY k HAVING sum(v) < 0"]
    assert "passed the int64 range" in subprocess.run(cmd, capture_output=True).stdout.decode()


def test_sum_across_row_groups_that_fits_is_exact(mx, data):
    """3e18 + 3e18 + 1 in three row groups merges to exactly 6000000000000000001."""
    res = q(mx, data, "SELECT k, sum(v) FROM ovfit WHERE k = 1 GROUP BY k", ["ovfit"], extra=["--stream", "always", "--group", "hashed"])
    assert [r[1] for r in res["rows"]] == [6000000000000000001] or [str(r[1]) for r in res["rows"]] == ["6000000000000000001"], res["rows"]


@pytest.mark.parametrize("stream", ["never", "always"])
def test_build_side_past_the_budget_raises_readable_error(mx, data, stream):
    """The build side is collected row by row (its filter runs on the way in): the budget is checked while it grows, before it is allocated whole."""
    sql = "SELECT count(*) FROM facts f JOIN d1 d ON f.k = d.k WHERE d.k < 1500"
    cmd = [str(mx), "sql", "--stream", stream, "-t", f"facts={data['paths']['facts']}", "-t", f"d1={data['paths']['d1']}"]
    p = subprocess.run(cmd + ["--join-budget", "1000", sql], capture_output=True)
    out = (p.stdout + p.stderr).decode()
    assert "the build side of the join (d1) has 1,500 rows of 1 column so far, past the join's memory budget of 1000 B: filter it" in out, out[-400:]
    q(mx, data, sql, ["facts", "d1"], extra=["--stream", stream, "--join-budget", "100000"])
