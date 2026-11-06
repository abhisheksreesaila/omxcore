"""F-153: the inner equi-join of two tables in SQL (JOIN ... ON, USING, a comma join with WHERE equalities), equal to DuckDB.

Every query runs through `build/mx` (tests/mx_oracle.py: typed values compared exactly) and through DuckDB on the same data. The data
are generated with DuckDB and written to Parquet: a fact table with NULL keys, keys that miss, decimals, dates, text and floats; a
dimension table with unique keys, a repeated key column and different text dictionaries (case variants); a table whose keys repeat
heavily (a hot key); keys past 2^53; float keys with -0.0, NaN and infinities; an empty table. Each test also asserts the path that ran,
from the notes: which side was the build side, the layout of the table, and how the join ran.
"""
import re

import duckdb
import pytest

from mx_oracle import check, mx_binary, run_mx


@pytest.fixture(scope="session")
def mx():
    return mx_binary()


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("joins")
    con = duckdb.connect()
    con.execute("""create table facts as select
        i as fid,
        case when i % 97 = 0 then null else (i * 37) % 2600 end as k,
        (i * 13) % 1000 as k2,
        i % 11 as g,
        case when i % 50 = 0 then null else (i * 13) % 1000 - 500 end as x,
        ((i % 500) * 0.25)::decimal(12,2) as d,
        case when i % 9 = 8 then null else (['West','east','North','South','NORTH','mango','Zebra','apple'])[1 + i % 8] end as s,
        date '2020-01-01' + (i % 400)::int as dt,
        i % 3 = 0 as b,
        case i % 8 when 0 then 0.0 when 1 then -0.0 when 2 then 'nan'::double when 3 then 1.5 when 4 then 'inf'::double when 5 then null else (i % 20)::double end as f
        from range(50000) r(i)""")
    con.execute("""create table dims as select
        i as k,
        i % 100 as kk,
        (i * 7) % 1000 as k2,
        'name' || (i % 50) as name,
        (i * 1.0001)::decimal(10,4) as w,
        ((i % 500) * 0.25)::decimal(18,4) as d4,
        (['west','East','North','south','Zebra','APPLE','Mango','west '])[1 + i % 8] as s,
        date '2020-01-01' + (i % 400)::int as dt,
        i % 3 = 0 as b,
        case i % 8 when 0 then -0.0 when 1 then 0.0 when 2 then 'nan'::double when 3 then 1.5 when 4 then 'inf'::double else (i % 20)::double end as f
        from range(2000) r(i)""")
    con.execute("""create table hot as select (case when i % 2 = 0 then 7 else i % 300 end) as k, i as hid, (i % 5) as v from range(3000) r(i)""")
    con.execute("""create table bigk_a as select (9007199254740992 + (i % 6))::bigint as k, i as a from range(600) r(i)
                   union all select 9223372036854775807::bigint, 1000 union all select (-9223372036854775808)::bigint, 1001 union all select null::bigint, 1002""")
    con.execute("""create table bigk_b as select (9007199254740992 + i)::bigint as k, i as b from range(8) r(i)
                   union all select 9223372036854775807::bigint, 100 union all select (-9223372036854775808)::bigint, 101 union all select 0::bigint, 102""")
    con.execute("""create table bigf as select i as fid, case when i % 101 = 0 then null else (i * 37) % 2600 end as k, i % 11 as g, (i % 1000) - 500 as x,
        ((i % 700) * 0.01)::decimal(10,2) as m from range(600000) r(i)""")
    con.execute("create table nothing as select * from dims where false")
    con.execute("create table wide as select i as a, ((i::hugeint * 4503599627370449) % 9007199254740881)::bigint as b, i % 7 as g from range(3000) r(i)")
    con.execute("create table wide2 as select i as a, ((i::hugeint * 4503599627370449) % 9007199254740881)::bigint as b, i * 2 as v from range(2000) r(i)")
    tables = ["facts", "dims", "hot", "bigk_a", "bigk_b", "nothing", "wide", "wide2", "bigf"]
    paths = {}
    for t in tables:
        paths[t] = d / f"{t}.parquet"
        con.execute(f"copy {t} to '{paths[t]}' (format parquet, row_group_size 7000)")
    return {"con": con, "paths": paths}


def q(mx, data, sql, tables, ordered=False, duck_sql=None, extra=()):
    return check(mx, sql, data["con"], files={t: data["paths"][t] for t in tables}, ordered=ordered, duck_sql=duck_sql, extra=extra)


def note(res, prefix):
    hits = [n for n in res["notes"] if n.startswith(prefix)]
    assert hits, (prefix, res["notes"])
    return hits[0]


# ── the answers ──────────────────────────────────────────────────────────────────────────────────────

FK_PK = [
    "SELECT count(*), sum(f.x), sum(d.w), min(f.dt) FROM facts f JOIN dims d ON f.k = d.k",
    "SELECT d.name, count(*), sum(f.d) FROM facts f JOIN dims d ON f.k = d.k GROUP BY d.name",
    "SELECT f.g, d.kk, count(*), sum(f.x * 2 + 1) FROM facts f JOIN dims d ON f.k = d.k WHERE d.b GROUP BY f.g, d.kk",
    "SELECT f.fid, f.x, d.name, d.w FROM facts f JOIN dims d ON f.k = d.k WHERE f.fid < 3000",
    "SELECT f.fid, d.name FROM facts f JOIN dims d ON f.k = d.k WHERE f.g = 3 AND d.kk < 20 AND f.x > d.kk",
    "SELECT d.dt, count(*) FROM facts f JOIN dims d ON f.k = d.k AND f.dt = d.dt GROUP BY d.dt",
    "SELECT count(*) FROM facts f JOIN dims d ON f.k = d.k WHERE f.x + d.kk > 400",
    "SELECT d.s, count(*), avg(f.x) FROM facts f JOIN dims d ON d.k = f.k GROUP BY d.s",
]


@pytest.mark.parametrize("join", ["auto", "materialize"])
@pytest.mark.parametrize("sql", FK_PK)
def test_fk_to_pk_joins_equal_duckdb(mx, data, sql, join):
    """Unique build keys run fused (the probe inside the aggregate pipeline); `--join materialize` writes the pairs and the joined table first."""
    res = q(mx, data, sql, ["facts", "dims"], extra=["--join", join])
    assert re.match(r"join build side: d \([\d,]+ rows, the smaller side\): direct array, unique keys", note(res, "join build side")), res["notes"]
    if join == "auto":
        assert note(res, "join path").startswith("join path: fused:"), res["notes"]
    else:
        assert note(res, "join path").startswith("join path: materialized:"), res["notes"]


MANY_TO_MANY = [
    "SELECT count(*), sum(f.x) FROM facts f JOIN dims d ON f.g = d.kk",
    "SELECT f.g, count(*), sum(d.w) FROM facts f JOIN dims d ON f.g = d.kk GROUP BY f.g",
    "SELECT f.fid, d.k FROM facts f JOIN dims d ON f.g = d.kk WHERE f.fid < 200 AND d.k < 150",
    "SELECT d.kk, count(*) FROM dims d JOIN dims e ON d.kk = e.kk GROUP BY d.kk",
]


@pytest.mark.parametrize("sql", MANY_TO_MANY)
def test_many_to_many_joins_equal_duckdb(mx, data, sql):
    res = q(mx, data, sql, ["facts", "dims"])
    assert "repeated keys" in note(res, "join build side"), res["notes"]
    if sql.startswith("SELECT f.fid"):                                   # a row query gathers the pairs; an aggregate expands in the front stage (F-161)
        assert "materialized" in note(res, "join path")
    else:
        assert "expanded to its partners" in note(res, "join path"), res["notes"]


def test_a_self_join_needs_aliases_and_works_with_them(mx, data):
    res = q(mx, data, "SELECT a.kk, count(*) FROM dims a JOIN dims b ON a.kk = b.kk AND a.k < b.k GROUP BY a.kk", ["dims"])
    assert note(res, "join build side")
    err = run_mx(mx, ["sql", "SELECT count(*) FROM dims JOIN dims ON dims.k = dims.k"], {"dims": data["paths"]["dims"]}, expect_exit=1)
    assert "needs a different alias for each side" in err["error"]["message"]


def test_composite_keys_are_packed_into_one_int64(mx, data):
    res = q(mx, data, "SELECT count(*), sum(f.x) FROM facts f JOIN dims d ON f.k = d.k AND f.k2 = d.k2", ["facts", "dims"])
    assert "2 keys packed into one int64" in note(res, "join keys")
    res = q(mx, data, "SELECT d.name, count(*) FROM facts f JOIN dims d ON f.g = d.kk AND f.k2 = d.k2 GROUP BY d.name", ["facts", "dims"])
    assert "packed" in note(res, "join keys")


def test_composite_keys_too_wide_to_pack_are_hashed_and_verified(mx, data):
    res = q(mx, data, "SELECT count(*), sum(v) FROM wide w JOIN wide2 u ON w.a = u.a AND w.b = u.b", ["wide", "wide2"])
    assert "hashed to one int64" in note(res, "join keys")
    q(mx, data, "SELECT w.g, count(*) FROM wide w JOIN wide2 u ON w.b = u.b AND w.a = u.a GROUP BY w.g", ["wide", "wide2"])


def test_text_keys_across_dictionaries_match_by_bytes(mx, data):
    # 'West' / 'west' / 'west ' / 'NORTH' / 'North' are different words to a join: only ordering ignores case
    res = q(mx, data, "SELECT f.s, count(*) FROM facts f JOIN dims d ON f.s = d.s GROUP BY f.s", ["facts", "dims"])
    assert note(res, "join build side")
    q(mx, data, "SELECT count(*) FROM facts f JOIN dims d ON f.s = d.s AND f.g = d.kk", ["facts", "dims"])
    q(mx, data, "SELECT d.s, f.s, count(*) FROM dims d JOIN facts f ON d.s = f.s WHERE f.g < 5 GROUP BY d.s, f.s", ["facts", "dims"])


def test_decimal_scales_are_aligned_and_int_is_scale_zero(mx, data):
    q(mx, data, "SELECT count(*), sum(f.x) FROM facts f JOIN dims d ON f.d = d.d4", ["facts", "dims"])           # (12,2) with (18,4)
    q(mx, data, "SELECT count(*) FROM facts f JOIN dims d ON f.g = d.d4", ["facts", "dims"])                      # INT with DECIMAL(18,4)
    q(mx, data, "SELECT count(*) FROM facts f JOIN dims d ON d.w = f.d", ["facts", "dims"])


def test_dates_and_bools_join(mx, data):
    q(mx, data, "SELECT count(*), sum(f.x) FROM facts f JOIN dims d ON f.dt = d.dt", ["facts", "dims"])
    q(mx, data, "SELECT f.b, count(*) FROM facts f JOIN dims d ON f.b = d.b AND f.g = d.kk GROUP BY f.b", ["facts", "dims"])


def test_null_keys_never_match(mx, data):
    res = q(mx, data, "SELECT count(*), count(f.k) FROM facts f JOIN dims d ON f.k = d.k", ["facts", "dims"])
    nulls = data["con"].execute("select count(*) from facts where k is null").fetchone()[0]
    assert nulls > 0
    q(mx, data, "SELECT count(*) FROM bigk_a a JOIN bigk_b b ON a.k = b.k", ["bigk_a", "bigk_b"])


def test_int64_keys_past_2_53_are_exact(mx, data):
    # 2^53 + 1 and 2^53 + 3 differ from 2^53 + 2 as integers, not as doubles
    q(mx, data, "SELECT a.k, count(*), sum(b.b) FROM bigk_a a JOIN bigk_b b ON a.k = b.k GROUP BY a.k", ["bigk_a", "bigk_b"])
    q(mx, data, "SELECT a.a, b.b FROM bigk_a a JOIN bigk_b b ON a.k = b.k WHERE a.a > 590", ["bigk_a", "bigk_b"])


def test_float_keys_follow_duckdbs_rules(mx, data):
    # -0.0 equals 0.0, NaN equals NaN, inf equals inf, NULL never matches
    q(mx, data, "SELECT count(*), sum(f.x) FROM facts f JOIN dims d ON f.f = d.f", ["facts", "dims"])
    q(mx, data, "SELECT f.f, count(*) FROM facts f JOIN dims d ON f.f = d.f GROUP BY f.f", ["facts", "dims"])


def test_empty_sides(mx, data):
    res = q(mx, data, "SELECT count(*), sum(f.x) FROM facts f JOIN nothing n ON f.k = n.k", ["facts", "nothing"])
    assert "empty" in note(res, "join build side")
    q(mx, data, "SELECT f.fid FROM facts f JOIN nothing n ON f.k = n.k", ["facts", "nothing"])
    q(mx, data, "SELECT count(*) FROM nothing n JOIN facts f ON f.k = n.k", ["facts", "nothing"])
    q(mx, data, "SELECT count(*) FROM facts f JOIN dims d ON f.k = d.k WHERE d.k < 0", ["facts", "dims"])


def test_a_hot_key(mx, data):
    # the hot key is on the probe side: the build side (dims) has unique keys, so the join is fused
    res = q(mx, data, "SELECT count(*), sum(h.v) FROM hot h JOIN dims d ON h.k = d.k", ["hot", "dims"])
    # (no column of dims is read after the join: its unique keys are a presence bitmap, F-207; reading one keeps the direct array)
    assert re.match(r"join build side: d \(2,000 rows, the smaller side\): presence bitmap:", note(res, "join build side")), res["notes"]
    res = q(mx, data, "SELECT count(*), sum(h.v), max(d.k) FROM hot h JOIN dims d ON h.k = d.k", ["hot", "dims"])
    assert re.match(r"join build side: d \(2,000 rows, the smaller side\): direct array, unique keys", note(res, "join build side")), res["notes"]
    assert note(res, "join path").startswith("join path: fused:"), res["notes"]
    # the hot key on both sides: repeated build keys, materialized pairs
    res = q(mx, data, "SELECT h.v, count(*) FROM hot h JOIN hot g ON h.k = g.k WHERE h.hid < 400 AND g.hid < 400 GROUP BY h.v", ["hot"])
    assert "repeated keys (CSR rows)" in note(res, "join build side"), res["notes"]
    assert note(res, "join path").startswith("join path: materialized:"), res["notes"]
    res = q(mx, data, "SELECT count(*) FROM hot h JOIN hot g ON h.k = g.k", ["hot"])
    assert "repeated keys" in note(res, "join build side") and note(res, "join path").startswith("join path: materialized:"), res["notes"]


# ── the build side is the side with fewer rows after its filters ────────────────────────────────────

def test_the_build_side_is_the_smaller_side_after_filters(mx, data):
    res = q(mx, data, "SELECT count(*) FROM facts f JOIN dims d ON f.k = d.k", ["facts", "dims"])
    assert note(res, "join build side").startswith("join build side: d (2,000 rows")
    res = q(mx, data, "SELECT count(*) FROM facts f JOIN dims d ON f.k = d.k WHERE f.fid < 300", ["facts", "dims"])
    assert note(res, "join build side").startswith("join build side: f (300 rows"), res["notes"]
    res = q(mx, data, "SELECT count(*) FROM dims d JOIN facts f ON f.k = d.k WHERE f.fid < 300", ["facts", "dims"])
    assert note(res, "join build side").startswith("join build side: f (300 rows"), res["notes"]
    assert "20,000" not in note(res, "join:") and "300 rows of f" in note(res, "join:")


# ── the other ways to write a join ──────────────────────────────────────────────────────────────────

def test_using_and_comma_joins(mx, data):
    q(mx, data, "SELECT count(*), sum(x) FROM facts JOIN dims USING (k)", ["facts", "dims"])
    q(mx, data, "SELECT k, count(*) FROM facts JOIN dims USING (k) WHERE x > 0 GROUP BY k", ["facts", "dims"])
    q(mx, data, "SELECT * FROM facts JOIN dims USING (k) WHERE fid < 40", ["facts", "dims"])
    q(mx, data, "SELECT count(*), sum(f.x) FROM facts f, dims d WHERE f.k = d.k AND d.kk < 30", ["facts", "dims"])
    q(mx, data, "SELECT d.kk, count(*) FROM facts f, dims d WHERE d.k = f.k GROUP BY d.kk", ["facts", "dims"])
    q(mx, data, "SELECT count(*) FROM facts f JOIN dims d ON f.k = d.k WHERE f.g = d.kk", ["facts", "dims"])


def test_star_and_qualified_star(mx, data):
    q(mx, data, "SELECT * FROM facts f JOIN dims d ON f.k = d.k WHERE f.fid < 30", ["facts", "dims"])
    q(mx, data, "SELECT d.* FROM facts f JOIN dims d ON f.k = d.k WHERE f.fid < 30", ["facts", "dims"])
    q(mx, data, "SELECT f.*, d.name FROM facts f JOIN dims d ON f.k = d.k WHERE f.fid < 30", ["facts", "dims"])


def test_order_by_limit_distinct_over_a_join(mx, data):
    q(mx, data, "SELECT f.fid, d.name FROM facts f JOIN dims d ON f.k = d.k ORDER BY f.fid DESC LIMIT 25", ["facts", "dims"], ordered=True,
      duck_sql="SELECT f.fid, d.name FROM facts f JOIN dims d ON f.k = d.k ORDER BY f.fid DESC NULLS LAST LIMIT 25")
    q(mx, data, "SELECT DISTINCT d.kk, f.g FROM facts f JOIN dims d ON f.k = d.k", ["facts", "dims"])
    q(mx, data, "SELECT d.kk, sum(f.x) AS s FROM facts f JOIN dims d ON f.k = d.k GROUP BY d.kk HAVING sum(f.x) > 100 ORDER BY s DESC NULLS LAST, d.kk LIMIT 10",
      ["facts", "dims"], ordered=True)


# ── names ───────────────────────────────────────────────────────────────────────────────────────────

def err(mx, data, sql, tables):
    return run_mx(mx, ["sql", sql], {t: data["paths"][t] for t in tables}, expect_exit=1)["error"]


def test_an_ambiguous_bare_name_raises_and_a_qualified_one_works(mx, data):
    e = err(mx, data, "SELECT k FROM facts f JOIN dims d ON f.k = d.k", ["facts", "dims"])
    assert e["message"] == "column `k` is ambiguous: it is in both `f` and `d` (write f.k or d.k)" and (e["line"], e["column"]) == (1, 8), e
    q(mx, data, "SELECT f.k, d.k FROM facts f JOIN dims d ON f.k = d.k WHERE f.fid < 20", ["facts", "dims"])
    q(mx, data, "SELECT name, w FROM facts f JOIN dims d ON f.k = d.k WHERE fid < 20", ["facts", "dims"])           # names in one table only
    q(mx, data, "SELECT facts.fid, dims.name FROM facts JOIN dims ON facts.k = dims.k WHERE facts.fid < 20", ["facts", "dims"])


def test_unknown_names_say_where(mx, data):
    e = err(mx, data, "SELECT f.nope FROM facts f JOIN dims d ON f.k = d.k", ["facts", "dims"])
    assert e["message"] == "column `nope` not found in `f`" and (e["line"], e["column"]) == (1, 8), e
    e = err(mx, data, "SELECT z.k FROM facts f JOIN dims d ON f.k = d.k", ["facts", "dims"])
    assert e["message"] == "unknown table or alias `z` in `z.k`", e
    e = err(mx, data, "SELECT nope FROM facts f JOIN dims d ON f.k = d.k", ["facts", "dims"])
    assert e["message"].startswith("column `nope` not found in `f` or `d`"), e
    e = err(mx, data, "SELECT count(*) FROM facts f JOIN nowhere d ON f.k = d.k", ["facts"])
    assert e["message"].startswith("table `nowhere` not found"), e


def test_unsupported_joins_say_so(mx, data):
    e = err(mx, data, "SELECT count(*) FROM facts f JOIN dims d ON f.x > d.k", ["facts", "dims"])
    assert e["message"].startswith("the join has no equality between a column of `f` and a column of `d`"), e
    e = err(mx, data, "SELECT count(*) FROM facts f RIGHT JOIN dims d ON f.k = d.k", ["facts", "dims"])
    assert e["message"] == "`RIGHT JOIN` is not supported yet", e
    e = err(mx, data, "SELECT count(*) FROM facts f LEFT JOIN dims d ON f.k = d.k JOIN hot h ON h.k = d.k", ["facts", "dims", "hot"])
    assert e["message"] == "a `LEFT JOIN` together with a third table is not supported yet (F-159)", e


@pytest.mark.parametrize("sql,message", [
    ("SELECT count(*) FROM facts f JOIN dims d ON f.dt = d.k", "cannot join f.dt (DATE) with d.k (INT)"),
    ("SELECT count(*) FROM facts f JOIN dims d ON f.s = d.k", "cannot join f.s (TEXT) with d.k (INT)"),
    ("SELECT count(*) FROM facts f JOIN dims d ON f.f = d.k", "cannot join f.f (FLOAT) with d.k (INT)"),
    ("SELECT count(*) FROM facts f JOIN dims d ON f.b = d.k", "cannot join f.b (BOOL) with d.k (INT)"),
])
def test_keys_of_different_kinds_are_refused_readably(mx, data, sql, message):
    e = err(mx, data, sql, ["facts", "dims"])
    assert e["message"].startswith(message), e
    assert (e["line"], e["column"]) == (1, sql.index(" = ") + 2), e          # the `=` of the ON condition (1-based)


def test_a_bad_on_condition_is_called_ON_not_WHERE(mx, data):
    e = err(mx, data, "SELECT count(*) FROM facts f JOIN dims d ON f.k = d.k AND f.x + 1", ["facts", "dims"])
    assert e["message"].startswith("ON needs a true / false value"), e
    e = err(mx, data, "SELECT count(*) FROM facts f JOIN dims d ON f.k = d.k WHERE f.x + 1", ["facts", "dims"])
    assert e["message"].startswith("WHERE needs a true / false value"), e


def test_explain_shows_the_join_tree(mx, data):
    out = run_mx(mx, ["sql", "EXPLAIN SELECT d.name, count(*) FROM facts f JOIN dims d ON f.k = d.k WHERE f.g = 3 AND d.kk < 20 AND f.x > d.kk GROUP BY d.name"],
                 {"facts": data["paths"]["facts"], "dims": data["paths"]["dims"]})
    lines = [r[0] for r in out["rows"]]
    assert lines[:11] == ["Logical plan", "  Project d.name, count(*)", "    Aggregate keys=[name] aggs=[count_star()]", "      Filter f.x > d.kk",
                          "        Join (inner) on f.k = d.k", "          Filter f.g = 3", "            Scan facts AS f", "          Filter d.kk < 20",
                          "            Scan dims AS d", "Physical plan", lines[10]], lines
    assert lines[10].startswith("  join: inner, on k = k"), lines


# ── the fused path in detail ─────────────────────────────────────────────────────────────────────────

FUSED = [
    # the build side's columns of every kind come through the block scratch: TEXT, DECIMAL, DATE, BOOL, FLOAT, with NULLs
    "SELECT d.s, d.b, count(*), sum(d.w), min(d.dt), max(d.f) FROM facts f JOIN dims d ON f.k = d.k GROUP BY d.s, d.b",
    "SELECT f.g, d.name, sum(d.d4 + f.d) FROM facts f JOIN dims d ON f.k = d.k WHERE d.dt > DATE '2020-06-01' GROUP BY f.g, d.name",
    "SELECT count(*), sum(f.x), avg(d.w) FROM facts f JOIN dims d ON f.k = d.k",
    # rows: partner columns through registers, probe columns gathered
    "SELECT f.fid, d.name, d.w, d.s, d.b, d.f FROM facts f JOIN dims d ON f.k = d.k WHERE f.fid < 5000",
    "SELECT f.fid, d.name FROM facts f JOIN dims d ON f.k = d.k WHERE d.k < 100",
    "SELECT d.k, d.name, f.fid FROM facts f JOIN dims d ON f.k = d.k WHERE f.x > 400",
    # keys that are expressions of the partner's columns
    "SELECT d.kk % 7, count(*) FROM facts f JOIN dims d ON f.k = d.k GROUP BY d.kk % 7",
    "SELECT count(*) FROM facts f JOIN dims d ON f.k = d.k WHERE d.name <> 'name3' AND f.s = 'North'",
    # the probe side is the left table or the right one
    "SELECT d.kk, sum(f.x) FROM dims d JOIN facts f ON d.k = f.k GROUP BY d.kk",
]


@pytest.mark.parametrize("sql", FUSED)
def test_fused_joins_equal_duckdb(mx, data, sql):
    res = q(mx, data, sql, ["facts", "dims"])
    assert note(res, "join path").startswith("join path: fused:"), res["notes"]


def test_fused_row_queries_order_limit_and_stop_early(mx, data):
    for sql, duck in [
        ("SELECT f.fid, d.name FROM facts f JOIN dims d ON f.k = d.k ORDER BY f.fid DESC, d.name LIMIT 30",
         "SELECT f.fid, d.name FROM facts f JOIN dims d ON f.k = d.k ORDER BY f.fid DESC NULLS LAST, lower(d.name), d.name LIMIT 30"),
        ("SELECT f.fid, d.w FROM facts f JOIN dims d ON f.k = d.k ORDER BY d.w DESC, f.fid LIMIT 30 OFFSET 5",
         "SELECT f.fid, d.w FROM facts f JOIN dims d ON f.k = d.k ORDER BY d.w DESC NULLS LAST, f.fid NULLS LAST LIMIT 30 OFFSET 5"),
        ("SELECT f.fid, d.name FROM facts f JOIN dims d ON f.k = d.k ORDER BY d.name, f.fid",
         "SELECT f.fid, d.name FROM facts f JOIN dims d ON f.k = d.k ORDER BY lower(d.name), d.name, f.fid"),
    ]:
        res = q(mx, data, sql, ["facts", "dims"], ordered=True, duck_sql=duck)
        assert note(res, "join path").startswith("join path: fused:"), res["notes"]
    res = run_mx(mx, ["sql", "--max-rows", "0", "SELECT f.fid, d.name FROM facts f JOIN dims d ON f.k = d.k LIMIT 7"],
                 {"facts": data["paths"]["facts"], "dims": data["paths"]["dims"]})
    assert len(res["rows"]) == 7 and note(res, "join path").startswith("join path: fused:")
    want = {tuple(r) for r in data["con"].execute("SELECT f.fid, d.name FROM facts f JOIN dims d ON f.k = d.k").fetchall()}
    assert all((r[0], r[1]) in want for r in res["rows"])


def test_an_aggregate_that_needs_every_row_materializes(mx, data):
    res = q(mx, data, "SELECT d.kk, count(DISTINCT f.x), median(f.x) FROM facts f JOIN dims d ON f.k = d.k GROUP BY d.kk", ["facts", "dims"])
    assert note(res, "join path").startswith("join path: materialized:") and "holistic" in note(res, "join path"), res["notes"]


def test_a_filter_can_flip_the_sides(mx, data):
    res = q(mx, data, "SELECT count(*), sum(d.w) FROM facts f JOIN dims d ON f.k = d.k WHERE f.fid < 100", ["facts", "dims"])
    assert note(res, "join build side").startswith("join build side: f (100 rows"), res["notes"]


BIG = [
    "SELECT count(*), sum(b.x), sum(d.w) FROM bigf b JOIN dims d ON b.k = d.k",
    "SELECT d.s, count(*), sum(b.m) FROM bigf b JOIN dims d ON b.k = d.k GROUP BY d.s",
    "SELECT d.kk, b.g, count(*), min(b.x), max(d.dt) FROM bigf b JOIN dims d ON b.k = d.k WHERE b.x > -200 GROUP BY d.kk, b.g",
    "SELECT d.name, sum(b.m) FROM bigf b JOIN dims d ON b.k = d.k GROUP BY d.name",
]


@pytest.mark.parametrize("sql", BIG)
def test_fused_joins_over_many_blocks_and_workers_equal_duckdb(mx, data, sql):
    """600,000 probe rows are ten morsels of the pipeline: every worker has its own scratch."""
    res = q(mx, data, sql, ["bigf", "dims"])
    assert note(res, "join path").startswith("join path: fused:"), res["notes"]
    q(mx, data, sql, ["bigf", "dims"], extra=["--join", "materialize"])
    q(mx, data, sql, ["bigf", "dims"], extra=["--threads", "1"])


def test_fused_rows_over_many_workers(mx, data):
    q(mx, data, "SELECT b.fid, d.name FROM bigf b JOIN dims d ON b.k = d.k WHERE b.fid % 5000 = 3", ["bigf", "dims"])
    q(mx, data, "SELECT b.fid, d.w FROM bigf b JOIN dims d ON b.k = d.k ORDER BY d.w DESC, b.fid LIMIT 40", ["bigf", "dims"], ordered=True,
      duck_sql="SELECT b.fid, d.w FROM bigf b JOIN dims d ON b.k = d.k ORDER BY d.w DESC NULLS LAST, b.fid NULLS LAST LIMIT 40")


@pytest.mark.parametrize("sql", [
    "SELECT d.name, count(*), sum(f.x) FROM facts f JOIN dims d ON f.k = d.k GROUP BY d.name",
    "SELECT f.fid, d.name FROM facts f JOIN dims d ON f.k = d.k WHERE f.fid < 500",
    "SELECT count(*) FROM facts f JOIN dims d ON f.g = d.kk",
])
def test_joins_over_streamed_parquet_scans_equal_duckdb(mx, data, sql):
    """`--stream always` reads each file by row groups (7,000 rows each); the join sees the same rows."""
    q(mx, data, sql, ["facts", "dims"], extra=["--stream", "always"])
