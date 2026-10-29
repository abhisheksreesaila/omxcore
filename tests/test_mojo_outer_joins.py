"""F-154: LEFT, SEMI and ANTI joins and the subqueries that mean them (IN, NOT IN, EXISTS, NOT EXISTS), equal to DuckDB.

Every query runs through `build/mx` (tests/mx_oracle.py: typed values compared exactly) and through DuckDB on the same Parquet files. The tables have
NULL keys on both sides, keys that miss, repeated keys, text with different dictionaries, decimals, dates; an empty table; a table whose key column holds
a NULL (NOT IN). Each test asserts the path that ran from the notes (fused or materialized, the build side, the filter mode of a presence join).
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
    d = tmp_path_factory.mktemp("outer")
    con = duckdb.connect()
    # customers: unique ck; seg with NULLs; text; decimal; date
    con.execute("""create table cust as select i as ck, case when i % 7 = 0 then null else i % 50 end as seg, case when i % 23 = 0 then null else ['West','east','North','South','NORTH'][1 + i % 5] end as s,
        ((i % 40) * 0.25)::decimal(10,2) as bal, date '2021-01-01' + (i % 300)::int as dt from range(400) r(i)""")
    # orders: ck NULL every 11th, many keys that miss (up to 460), keys repeat (3 orders a key at most more); amt with NULLs
    con.execute("""create table ords as select i as oid, case when i % 11 = 0 then null else (i * 3) % 460 end as ck,
        case when i % 13 = 0 then null else i % 100 end as amt,
        ['east','West','NORTH','x'][1 + i % 4] as tag, ((i % 70) * 0.5)::decimal(10,2) as price from range(1500) r(i)""")
    # a dimension with unique keys and a fact table that probes it (fused LEFT): fact has NULL keys and misses
    con.execute("""create table dim as select i as k, 'name' || (i % 40) as name, (i * 1.5)::decimal(10,2) as w, date '2020-01-01' + (i % 200)::int as dt,
        (i % 9)::double / 2 as f, i % 3 = 0 as b from range(1200) r(i)""")
    con.execute("""create table fact as select i as fid, case when i % 17 = 0 then null else (i * 7) % 1500 end as k, i % 11 as g, (i % 500) - 250 as x,
        ['West','east','zebra'][1 + i % 3] as s from range(30000) r(i)""")
    # a list with a NULL, a list without, an empty table, and an empty-keyed table
    con.execute("create table withnull as select * from (values (3::bigint), (5), (null), (8)) t(ck)")
    con.execute("create table nonull as select * from (values (3::bigint), (5), (8), (400)) t(ck)")
    con.execute("create table nothing as select * from ords where false")
    con.execute("create table nothing_c as select * from cust where false")
    paths = {}
    for t in ["cust", "ords", "dim", "fact", "withnull", "nonull", "nothing", "nothing_c"]:
        paths[t] = d / f"{t}.parquet"
        con.execute(f"copy {t} to '{paths[t]}' (format parquet, row_group_size 4000)")
    return {"con": con, "paths": paths}


def q(mx, data, sql, tables, ordered=False, duck_sql=None, extra=()):
    return check(mx, sql, data["con"], files={t: data["paths"][t] for t in tables}, ordered=ordered, duck_sql=duck_sql, extra=extra)


def note(res, prefix):
    hits = [n for n in res["notes"] if n.startswith(prefix)]
    assert hits, (prefix, res["notes"])
    return hits[0]


def err(mx, data, sql, tables):
    out = run_mx(mx, ["sql", sql], {t: data["paths"][t] for t in tables}, expect_exit=1)
    return out["error"]


# ── LEFT JOIN ────────────────────────────────────────────────────────────────────────────────────────

LEFT_FUSED = [
    "SELECT count(*), count(d.name), sum(d.w), min(d.dt), max(d.f) FROM fact f LEFT JOIN dim d ON f.k = d.k",
    "SELECT d.name, count(*), sum(f.x) FROM fact f LEFT JOIN dim d ON f.k = d.k GROUP BY d.name",
    "SELECT f.g, d.b, count(*), count(d.w), avg(f.x) FROM fact f LEFT JOIN dim d ON f.k = d.k WHERE f.x > -100 GROUP BY f.g, d.b",
    "SELECT f.fid, f.k, d.name, d.w, d.dt, d.b FROM fact f LEFT JOIN dim d ON f.k = d.k WHERE f.fid < 400",
    "SELECT f.fid, d.name FROM fact f LEFT JOIN dim d ON f.k = d.k WHERE d.k IS NULL AND f.fid < 2000",
    "SELECT count(*) FROM fact f LEFT JOIN dim d ON f.k = d.k AND d.w > 100",
    "SELECT f.g, count(d.k) FROM fact f LEFT JOIN dim d ON f.k = d.k AND d.b GROUP BY f.g",
    "SELECT count(*), sum(d.w) FROM fact f LEFT JOIN dim d ON f.k = d.k WHERE d.w > 500",
    "SELECT f.s, count(*), count(d.name) FROM fact f LEFT JOIN dim d ON f.k = d.k GROUP BY f.s",
]


@pytest.mark.parametrize("join", ["auto", "materialize"])
@pytest.mark.parametrize("sql", LEFT_FUSED)
def test_left_join_with_unique_build_keys_equals_duckdb(mx, data, sql, join):
    """Unique build keys run fused: unmatched probe rows stay, their build columns are NULL; `--join materialize` writes the pairs first."""
    res = q(mx, data, sql, ["fact", "dim"], extra=["--join", join])
    assert re.match(r"join build side: d \([\d,]+ rows, the right side: a left join builds on it\): direct array, unique keys", note(res, "join build side")), res["notes"]
    path = note(res, "join path")
    if join == "auto":
        assert path.startswith("join path: fused:") and "NULL" in path, res["notes"]
    else:
        assert path.startswith("join path: materialized:") and "unmatched rows included" in path, res["notes"]
    q(mx, data, sql, ["fact", "dim"], extra=["--threads", "1"])


LEFT_REPEATED = [
    "SELECT c.ck, count(o.oid) FROM cust c LEFT JOIN ords o ON c.ck = o.ck GROUP BY c.ck",
    "SELECT c.ck, count(o.oid), sum(o.price) FROM cust c LEFT JOIN ords o ON c.ck = o.ck AND o.amt > 20 GROUP BY c.ck",
    "SELECT c.ck, o.oid, o.tag FROM cust c LEFT JOIN ords o ON c.ck = o.ck WHERE c.ck < 60",
    "SELECT count(*), count(o.oid), count(c.ck) FROM cust c LEFT JOIN ords o ON c.ck = o.ck",
    "SELECT c.s, count(*), sum(o.amt) FROM cust c LEFT JOIN ords o ON c.ck = o.ck WHERE c.seg IS NOT NULL GROUP BY c.s",
    "SELECT c.ck FROM cust c LEFT JOIN ords o ON c.ck = o.ck WHERE o.oid IS NULL",
    "SELECT c.ck, o.tag, o.price FROM cust c LEFT JOIN ords o ON c.seg = o.amt WHERE c.ck < 30 AND o.price > 10",
    "SELECT o.tag, count(*) FROM cust c LEFT JOIN ords o ON c.ck = o.ck GROUP BY o.tag",
    "SELECT c.ck, o.oid FROM cust c LEFT JOIN ords o ON c.seg = o.amt WHERE c.ck < 100",
]


@pytest.mark.parametrize("sql", LEFT_REPEATED)
def test_left_join_with_repeated_build_keys_equals_duckdb(mx, data, sql):
    """Repeated build keys: the pairs (unmatched rows included) are gathered into a table and the query runs over it."""
    res = q(mx, data, sql, ["cust", "ords"])
    assert re.match(r"join build side: o \([\d,]+ rows, the right side: a left join builds on it\): .*repeated keys", note(res, "join build side")), res["notes"]
    assert note(res, "join path").startswith("join path: materialized:"), res["notes"]


def test_a_left_join_counted_per_left_row_is_the_q13_shape(mx, data):
    """Every customer appears once, with the number of its orders (0 for none): the unmatched rows are counted, not dropped."""
    sql = "SELECT c.ck, count(o.oid) AS n FROM cust c LEFT JOIN ords o ON c.ck = o.ck AND o.tag <> 'x' GROUP BY c.ck ORDER BY n DESC, c.ck LIMIT 25"
    q(mx, data, sql, ["cust", "ords"], ordered=True,
      duck_sql="SELECT c.ck, count(o.oid) AS n FROM cust c LEFT JOIN ords o ON c.ck = o.ck AND o.tag <> 'x' GROUP BY c.ck ORDER BY n DESC, c.ck LIMIT 25")
    res = q(mx, data, "SELECT count(*) FROM cust c LEFT JOIN ords o ON c.ck = o.ck AND o.tag <> 'x' WHERE o.oid IS NULL", ["cust", "ords"])
    assert res["rows"][0][0] > 0


@pytest.mark.parametrize("left,right", [("cust", "nothing"), ("nothing_c", "ords"), ("nothing_c", "nothing")])
def test_left_join_with_an_empty_side(mx, data, left, right):
    sqls = {
        ("cust", "nothing"): ["SELECT count(*), count(o.oid) FROM cust c LEFT JOIN nothing o ON c.ck = o.ck", "SELECT c.ck, o.oid FROM cust c LEFT JOIN nothing o ON c.ck = o.ck WHERE c.ck < 5"],
        ("nothing_c", "ords"): ["SELECT count(*), count(o.oid) FROM nothing_c c LEFT JOIN ords o ON c.ck = o.ck", "SELECT c.ck, o.oid FROM nothing_c c LEFT JOIN ords o ON c.ck = o.ck"],
        ("nothing_c", "nothing"): ["SELECT count(*) FROM nothing_c c LEFT JOIN nothing o ON c.ck = o.ck", "SELECT c.ck, o.oid FROM nothing_c c LEFT JOIN nothing o ON c.ck = o.ck"],
    }[(left, right)]
    for sql in sqls:
        q(mx, data, sql, [left, right])
        q(mx, data, sql, [left, right], extra=["--join", "materialize"])


def test_left_join_using_and_star(mx, data):
    q(mx, data, "SELECT * FROM cust LEFT JOIN withnull USING (ck) WHERE ck < 12", ["cust", "withnull"])
    q(mx, data, "SELECT c.ck, w.ck FROM cust c LEFT JOIN withnull w ON c.ck = w.ck WHERE c.ck < 12", ["cust", "withnull"])


def test_left_join_text_and_composite_keys(mx, data):
    q(mx, data, "SELECT count(*), count(o.oid) FROM cust c LEFT JOIN ords o ON c.ck = o.ck AND c.seg = o.amt", ["cust", "ords"])
    q(mx, data, "SELECT count(*), count(d.name) FROM fact f LEFT JOIN dim d ON f.k = d.k AND f.g = d.k", ["fact", "dim"])
    q(mx, data, "SELECT c.s, o.tag, count(*) FROM cust c LEFT JOIN ords o ON c.s = o.tag GROUP BY c.s, o.tag", ["cust", "ords"])


WHERE_EQUALITIES = [
    # a WHERE equality between the two tables is a filter after a LEFT join (it removes the NULL-extended rows), not a join key
    "SELECT count(*), count(d.name) FROM fact f LEFT JOIN dim d ON f.k = d.k WHERE f.g = d.k",
    "SELECT count(*), count(o.oid) FROM cust c LEFT JOIN ords o ON c.ck = o.ck WHERE c.seg = o.amt",
    "SELECT count(*), count(d.name) FROM cust c LEFT JOIN dim d ON c.ck = d.k WHERE c.dt = d.dt",
    "SELECT count(*) FROM cust c LEFT JOIN dim d ON c.seg = d.k WHERE c.dt = d.dt",
    "SELECT c.ck, o.oid FROM cust c LEFT JOIN ords o ON c.ck = o.ck WHERE c.seg = o.amt AND c.ck < 100",
    "SELECT c.s, count(*) FROM cust c LEFT JOIN ords o ON c.ck = o.ck WHERE o.amt = c.seg GROUP BY c.s",
    "SELECT count(*) FROM cust c LEFT JOIN ords o ON c.ck = o.ck WHERE c.ck = o.ck",
    "SELECT count(*) FROM cust c LEFT JOIN ords o ON c.ck = o.ck AND c.seg = o.amt WHERE o.oid = c.ck",
]


@pytest.mark.parametrize("extra", [[], ["--join", "materialize"], ["--stream", "always"], ["--threads", "1"]])
@pytest.mark.parametrize("sql", WHERE_EQUALITIES)
def test_a_where_equality_after_a_left_join_is_a_filter_not_a_key(mx, data, sql, extra):
    q(mx, data, sql, ["cust", "ords", "dim", "fact"], extra=extra)


# ── SEMI and ANTI JOIN, IN / NOT IN / EXISTS / NOT EXISTS ───────────────────────────────────────────

PRESENCE = [
    # (semi form, anti form) pairs checked together below
    ("SELECT count(*), sum(c.seg), min(c.dt) FROM cust c SEMI JOIN ords o ON c.ck = o.ck",
     "SELECT count(*), sum(c.seg), min(c.dt) FROM cust c WHERE EXISTS (SELECT 1 FROM ords o WHERE o.ck = c.ck)"),
    ("SELECT c.s, count(*), sum(c.bal) FROM cust c SEMI JOIN ords o ON c.ck = o.ck AND o.amt > 40 GROUP BY c.s",
     "SELECT c.s, count(*), sum(c.bal) FROM cust c WHERE EXISTS (SELECT * FROM ords o WHERE o.ck = c.ck AND o.amt > 40) GROUP BY c.s"),
    ("SELECT c.ck, c.s FROM cust c SEMI JOIN ords o ON c.ck = o.ck WHERE c.ck < 80",
     "SELECT c.ck, c.s FROM cust c WHERE c.ck < 80 AND c.ck IN (SELECT ck FROM ords)"),
    ("SELECT count(*) FROM cust SEMI JOIN ords USING (ck)",
     "SELECT count(*) FROM cust WHERE ck IN (SELECT ck FROM ords WHERE amt IS NOT NULL OR amt IS NULL)"),
]


@pytest.mark.parametrize("pair", PRESENCE)
def test_semi_join_and_its_subquery_forms_equal_duckdb(mx, data, pair):
    explicit, sub = pair
    duck_explicit = explicit
    for sql in (explicit, sub):
        res = q(mx, data, sql, ["cust", "ords"])
        assert re.match(r"join build side: (o|ords) \([\d,]+ rows, the right side: a semi join builds on it\)", note(res, "join build side")), res["notes"]
        assert note(res, "join path").startswith("join path: fused:") and "semi join" in note(res, "join path"), res["notes"]
        res = q(mx, data, sql, ["cust", "ords"], extra=["--join", "materialize"])
        assert note(res, "join path").startswith("join path: materialized:") and "pass the semi join" in note(res, "join path"), res["notes"]
        q(mx, data, sql, ["cust", "ords"], extra=["--threads", "1"])


ANTI = [
    "SELECT count(*), sum(c.seg) FROM cust c ANTI JOIN ords o ON c.ck = o.ck",
    "SELECT count(*), sum(c.seg) FROM cust c WHERE NOT EXISTS (SELECT 1 FROM ords o WHERE o.ck = c.ck)",
    "SELECT c.ck, c.seg FROM cust c WHERE NOT EXISTS (SELECT * FROM ords o WHERE o.ck = c.ck AND o.amt > 40) AND c.ck < 120",
    "SELECT c.s, count(*) FROM cust c ANTI JOIN ords o ON c.ck = o.ck GROUP BY c.s",
    "SELECT * FROM cust c ANTI JOIN ords o ON c.ck = o.ck WHERE c.ck < 150",
]


@pytest.mark.parametrize("sql", ANTI)
def test_anti_join_and_not_exists_equal_duckdb(mx, data, sql):
    """A NULL key has no partner, so NOT EXISTS keeps it (NOT IN does not: see below)."""
    res = q(mx, data, sql, ["cust", "ords"])
    assert note(res, "join path").startswith("join path: fused:") and "anti join" in note(res, "join path"), res["notes"]
    res = q(mx, data, sql, ["cust", "ords"], extra=["--join", "materialize"])
    assert note(res, "join path").startswith("join path: materialized:"), res["notes"]


IN_LISTS = [
    "SELECT count(*), sum(seg) FROM cust WHERE seg IN (SELECT ck FROM ords)",
    "SELECT ck FROM cust WHERE ck IN (SELECT ck FROM ords WHERE amt > 90) ORDER BY ck",
    "SELECT ck FROM cust WHERE ck IN (SELECT ck FROM withnull)",
    "SELECT ck FROM cust WHERE ck IN (SELECT ck FROM nothing)",
    "SELECT count(*) FROM nothing_c WHERE ck IN (SELECT ck FROM ords)",
    "SELECT count(*), sum(ck) FROM cust WHERE seg IN (SELECT seg FROM cust WHERE ck < 10)",           # the same table on both sides
    "SELECT count(*) FROM cust c WHERE EXISTS (SELECT * FROM cust c2 WHERE c2.seg = c.seg AND c2.ck < 10)",
    "SELECT count(*) FROM cust WHERE EXISTS (SELECT * FROM cust c2 WHERE c2.seg = cust.seg AND c2.ck < c2.seg)",
    "SELECT count(*) FROM cust WHERE ck IN (SELECT c2.ck FROM cust c2 WHERE c2.seg = 3)",
    "SELECT count(*) FROM cust WHERE ck NOT IN (SELECT ck FROM cust WHERE seg > 20)",
]


@pytest.mark.parametrize("sql", IN_LISTS)
def test_in_subquery_equals_duckdb(mx, data, sql):
    tables = ["cust", "ords", "withnull", "nothing", "nothing_c"]
    res = q(mx, data, sql, tables)
    assert ("semi" if "NOT IN" not in sql else "anti") in note(res, "join: "), res["notes"]


NOT_IN = [
    ("SELECT count(*) FROM cust WHERE seg NOT IN (SELECT ck FROM nonull)", "no NULL in the list: a NULL key is dropped, the others are kept when absent", "join: anti (null-aware)"),
    ("SELECT ck FROM cust WHERE ck NOT IN (SELECT ck FROM nonull) AND ck < 20", "no NULL in the list", "a NULL key is dropped"),
    ("SELECT count(*) FROM cust WHERE ck NOT IN (SELECT ck FROM withnull)", "a NULL in the list: no row at all", "the list holds a NULL: no row passes"),
    ("SELECT ck FROM cust WHERE ck NOT IN (SELECT ck FROM withnull)", "a NULL in the list: no row at all", "the list holds a NULL: no row passes"),
    ("SELECT count(*) FROM cust WHERE ck NOT IN (SELECT ck FROM nothing)", "an empty list: every row, NULL keys too", "the list is empty: every row is kept, a NULL key too"),
    ("SELECT count(*) FROM cust WHERE seg NOT IN (SELECT ck FROM nothing)", "an empty list keeps a NULL probe key", "the list is empty"),
    ("SELECT count(*) FROM cust WHERE seg NOT IN (SELECT ck FROM ords)", "ords.ck holds NULLs", "the list holds a NULL"),
    ("SELECT count(*) FROM cust WHERE seg NOT IN (SELECT ck FROM ords WHERE ck IS NOT NULL)", "the NULLs filtered out of the list", "a NULL key is dropped"),
    ("SELECT count(*) FROM cust c WHERE NOT EXISTS (SELECT * FROM cust c2 WHERE c2.seg = c.seg AND c2.ck < 10)", "the same table, own alias", "join: anti"),
    ("SELECT count(*) FROM nothing_c WHERE ck NOT IN (SELECT ck FROM nonull)", "an empty probe side", "join: anti (null-aware)"),
    ("SELECT s, count(*) FROM cust WHERE s NOT IN (SELECT tag FROM ords) GROUP BY s", "words the list lacks are kept, a NULL text key is dropped", "a NULL key is dropped"),
    ("SELECT s, count(*) FROM fact WHERE s NOT IN (SELECT tag FROM ords) GROUP BY s", "a word absent from the list's dictionary is not a NULL", "a NULL key is dropped"),
]


@pytest.mark.parametrize("sql,why,expect_note", NOT_IN)
def test_not_in_is_null_aware(mx, data, sql, why, expect_note):
    tables = ["cust", "ords", "withnull", "nonull", "nothing", "nothing_c", "fact"]
    res = q(mx, data, sql, tables)
    assert any(expect_note in n for n in res["notes"]), (why, res["notes"])
    q(mx, data, sql, tables, extra=["--join", "materialize"])


SCOPES = [
    # a name in a subquery means the subquery's own table when it is that table's alias (or its name, with no alias); the outer query is reached only by what the subquery does not define
    "SELECT count(*) FROM cust WHERE seg IN (SELECT seg FROM cust WHERE cust.ck > 300)",
    "SELECT count(*) FROM cust c1 WHERE seg IN (SELECT seg FROM cust WHERE cust.ck > 300)",
    "SELECT count(*) FROM cust c1 WHERE NOT EXISTS (SELECT * FROM cust WHERE cust.seg = c1.seg AND cust.ck > 300)",
    "SELECT count(*) FROM cust c1 WHERE EXISTS (SELECT * FROM cust WHERE cust.seg = c1.seg AND cust.ck > 300)",
    "SELECT count(*) FROM cust WHERE EXISTS (SELECT * FROM cust c2 WHERE c2.seg = cust.seg AND c2.ck > 300)",
    "SELECT count(*) FROM cust WHERE NOT EXISTS (SELECT * FROM cust c2 WHERE c2.seg = cust.seg AND c2.ck > 300)",
    "SELECT count(*) FROM cust AS cust WHERE cust.ck IN (SELECT ck FROM cust WHERE cust.seg < 10)",
    "SELECT count(*) FROM cust c1 WHERE c1.ck IN (SELECT cust.ck FROM cust WHERE cust.seg < 10 AND cust.ck > 100)",
    "SELECT count(*) FROM cust c1 WHERE c1.seg IN (SELECT cust.amt FROM ords cust WHERE cust.amt > 20)",
    "SELECT count(*) FROM cust c1 WHERE c1.seg NOT IN (SELECT cust.amt FROM ords cust WHERE cust.amt > 20 AND cust.amt IS NOT NULL)",
    "SELECT count(*) FROM cust c1 WHERE EXISTS (SELECT 1 FROM ords o WHERE o.ck = c1.ck AND o.amt > 50)",
    "SELECT count(*) FROM cust WHERE EXISTS (SELECT 1 FROM ords WHERE ords.ck = cust.ck AND ords.amt > 50)",
    "SELECT c1.s, count(*) FROM cust c1 WHERE NOT EXISTS (SELECT * FROM ords WHERE ords.ck = c1.ck AND amt > 50) GROUP BY c1.s",
    "SELECT ck FROM cust WHERE seg IN (SELECT seg FROM cust c3 WHERE c3.ck < 20) AND cust.ck > 380",
]


@pytest.mark.parametrize("extra", [[], ["--join", "materialize"], ["--stream", "always"]])
@pytest.mark.parametrize("sql", SCOPES)
def test_subquery_names_resolve_innermost_scope_first(mx, data, sql, extra):
    q(mx, data, sql, ["cust", "ords"], extra=extra)


def test_presence_joins_over_a_multi_block_probe(mx, data):
    """30,000 fact rows are several blocks of the pipeline and several workers; the dimension has unique and the orders repeated keys."""
    for sql in [
        "SELECT count(*), sum(x) FROM fact WHERE k IN (SELECT k FROM dim WHERE b)",
        "SELECT g, count(*) FROM fact WHERE NOT EXISTS (SELECT 1 FROM dim WHERE dim.k = fact.k) GROUP BY g",
        "SELECT g, count(*) FROM fact WHERE k NOT IN (SELECT k FROM dim) GROUP BY g",
        "SELECT s, count(*), sum(x) FROM fact f SEMI JOIN dim d ON f.k = d.k AND f.g = d.k GROUP BY s",
    ]:
        q(mx, data, sql, ["fact", "dim"])
        q(mx, data, sql, ["fact", "dim"], extra=["--join", "materialize"])
        q(mx, data, sql, ["fact", "dim"], extra=["--threads", "1"])


def test_presence_and_left_joins_over_streamed_scans(mx, data):
    for sql in [
        "SELECT count(*), count(o.oid) FROM cust c LEFT JOIN ords o ON c.ck = o.ck",
        "SELECT c.s, count(*) FROM cust c WHERE EXISTS (SELECT 1 FROM ords o WHERE o.ck = c.ck AND o.amt < 50) GROUP BY c.s",
        "SELECT count(*) FROM fact WHERE k NOT IN (SELECT k FROM dim)",
    ]:
        q(mx, data, sql, ["cust", "ords", "fact", "dim"], extra=["--stream", "always"])


def test_explain_names_the_kind(mx, data):
    out = run_mx(mx, ["sql", "EXPLAIN SELECT count(*) FROM cust c WHERE NOT EXISTS (SELECT 1 FROM ords o WHERE o.ck = c.ck AND o.amt > 40)"],
                 {"cust": data["paths"]["cust"], "ords": data["paths"]["ords"]})
    lines = [r[0] for r in out["rows"]]
    assert any(l.strip().startswith("Join (anti) on c.ck = o.ck") for l in lines), lines
    out = run_mx(mx, ["sql", "EXPLAIN SELECT c.ck FROM cust c LEFT JOIN ords o ON c.ck = o.ck"], {"cust": data["paths"]["cust"], "ords": data["paths"]["ords"]})
    assert any(r[0].strip().startswith("Join (left) on c.ck = o.ck") for r in out["rows"]), out["rows"]


# ── what is refused, by name ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sql,message", [
    ("SELECT count(*) FROM cust c RIGHT JOIN ords o ON c.ck = o.ck", "`RIGHT JOIN` is not supported yet"),
    ("SELECT count(*) FROM cust c RIGHT OUTER JOIN ords o ON c.ck = o.ck", "`RIGHT OUTER JOIN` is not supported yet"),
    ("SELECT count(*) FROM cust c FULL JOIN ords o ON c.ck = o.ck", "`FULL JOIN` is not supported yet"),
    ("SELECT count(*) FROM cust c FULL OUTER JOIN ords o ON c.ck = o.ck", "`FULL OUTER JOIN` is not supported yet"),
    ("SELECT count(*) FROM cust c CROSS JOIN ords o", "`CROSS JOIN` is not supported yet"),
])
def test_right_full_and_cross_joins_are_refused_by_name(mx, data, sql, message):
    e = err(mx, data, sql, ["cust", "ords"])
    assert e["message"].startswith(message), e
    assert e["line"] == 1


def test_unsupported_subquery_shapes_say_so(mx, data):
    # (F-159 lifted the limits on OR, several subqueries, subqueries beside a join, uncorrelated EXISTS and GROUP BY inside: tests/test_mojo_compound.py)
    t = ["cust", "ords", "dim"]
    e = err(mx, data, "SELECT count(*) FROM cust WHERE ck IN (SELECT ck, amt FROM ords)", t)
    assert e["message"].startswith("a subquery in IN must return one column, this one returns 2"), e
    e = err(mx, data, "SELECT count(*) FROM cust c LEFT JOIN ords o ON c.ck = o.ck AND c.seg > 3", ["cust", "ords"])
    assert "not supported in a LEFT JOIN yet" in e["message"], e
    e = err(mx, data, "SELECT o.oid FROM cust c SEMI JOIN ords o ON c.ck = o.ck", ["cust", "ords"])
    assert "unknown table or alias `o`" in e["message"] or "not found" in e["message"], e


def test_a_semi_join_select_star_has_the_left_columns_only(mx, data):
    res = run_mx(mx, ["sql", "SELECT * FROM cust c SEMI JOIN ords o ON c.ck = o.ck LIMIT 3"], {"cust": data["paths"]["cust"], "ords": data["paths"]["ords"]})
    assert [c["name"] for c in res["columns"]] == ["ck", "seg", "s", "bal", "dt"], res["columns"]
