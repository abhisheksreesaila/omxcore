"""F-206: `COUNT(DISTINCT e) ... GROUP BY g` runs as DuckDB's two phases: GROUP BY (g, e) through the normal machinery (fused into a join, dense or hashed groups),
then the number of non-NULL e groups per g (`count(DISTINCT): two phases` in the notes). Every answer equals DuckDB, through SQL over tables and SQL over edited
Frames; shapes that keep the holistic path (two COUNT(DISTINCT) of different columns, no GROUP BY, a median or a SUM beside it) answer as before and carry no note.

World: fact (150,000 rows) with a small group key g1 (NULLs), a large one g2, a float one gf (-0.0, 0.0, NaN, NULL), values of every type (NULLs in each) and a join
key k (NULL and dangling keys); dim (45,000 rows, unique keys) with a group column and a text column.
"""
import pytest

from conftest import oracle_connection
from test_mojo_sql import probe, run_probe, got_rows          # noqa: F401  (the probe fixture builds mojo/tools/sql_probe.mojo)
from test_f202_frames import make_edits
from mx_oracle import assert_equal, duck_kind

NOTE = "count(DISTINCT): two phases"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("f206")
    base = oracle_connection()
    base.execute("""create table fact as select i as id,
        case when i % 40 = 0 then null else i % 50 end as g1,
        (i * 7919) % 30011 as g2,
        case when i % 97 = 0 then null when i % 31 = 0 then -0.0 when i % 29 = 0 then 0.0 when i % 53 = 0 then 'nan'::double else (i % 7)::double end as gf,
        case when i % 20 = 0 then null when i % 11 = 0 then 70000 + i else (i * 7) % 45000 end as k,
        case when i % 9 = 0 then null else (i * 13) % 17 end as xi,
        (i * 31) % 70001 as xb,
        case when i % 8 = 0 then null else 'w' || ((i * 5) % 13) end as xt,
        case when i % 7 = 0 then null else date '2021-01-01' + ((i * 3) % 60)::int end as xd,
        case when i % 6 = 0 then null else ((i * 17) % 41) / 4.0 end::decimal(12, 2) as xc,
        case when i % 12 = 0 then null when i % 10 = 0 then -0.0 when i % 14 = 0 then 0.0 when i % 61 = 0 then 'nan'::double else ((i * 11) % 9)::double end as xf,
        case when i % 5 = 0 then null else i % 3 = 0 end as xl,
        i % 1000 as v from range(150000) r(i)""")
    base.execute("""create table dim as select i as dk,
        case when i % 33 = 0 then null else i % 25 end as dg,
        case when i % 7 = 0 then null else 'd' || (i % 11) end as dt from range(45000) r(i)""")
    paths = {}
    for t, rg in [("fact", 40000), ("dim", 30000)]:
        paths[t] = d / f"{t}.parquet"
        base.execute(f"copy {t} to '{paths[t]}' (format parquet, row_group_size {rg})")
    out = {"paths": paths, "dir": d, "con": base}
    # the Frame world: the columns an edit script can write (int, text, date, decimal), in file order, under the table name `fact`
    base.execute("create table ffact as select id, g1, g2, k, xi, xb, xt, xd, xc, v from fact")
    paths["ffact"] = d / "ffact.parquet"
    base.execute(f"copy ffact to '{paths['ffact']}' (format parquet, row_group_size 40000)")
    con = oracle_connection()
    con.execute(f"create table fact as select row_number() over () - 1 as rid, * from read_parquet('{paths['ffact']}')")
    con.execute(f"create table dim as select * from read_parquet('{paths['dim']}')")
    cols = [("g1", "int"), ("g2", "int"), ("k", "int"), ("xi", "int"), ("xb", "int"), ("xt", "text"), ("xd", "date"), ("xc", "dec2"), ("v", "int")]
    edits = d / "fact.edits"
    edits.write_text(make_edits(con, "fact", "id", cols, 150000, 5, 60, 45000))
    out["fact_frame"] = {"con": con, "edits": edits}
    return out


def files_of(world, tables, first=None):
    order = ([first] if first else []) + [t for t in tables if t != first]
    return {t: world["paths"][t] for t in order}


def same(res, cur, sql, label):
    want = cur.fetchall()
    for i, desc in enumerate(cur.description):
        dk = duck_kind(desc[1])
        c = res["columns"][i]
        assert c["kind"] == dk[0] and (dk[0] != "DECIMAL" or c["scale"] == dk[1]), f"column {i} of `{sql}`: got {c}, DuckDB has {dk}"
    assert_equal(got_rows(res), want, "ORDER BY" in sql.upper(), f"{label}: {sql}")


def two_phase(res):
    hits = [n for n in res["notes"] if n.startswith(NOTE)]
    return hits[0] if hits else None


# (name, sql, tables): every one runs the two phases
TWO_PHASE = [
    ("small key, int value", "select g1, count(distinct xi) as n from fact group by g1 order by g1", ["fact"]),
    ("large key, text value", "select g2, count(distinct xt) as n from fact group by g2 order by g2", ["fact"]),
    ("large key, int value, a filter", "select g2, count(distinct xi) as n from fact where v < 300 group by g2 order by g2", ["fact"]),
    ("big domain value", "select g1, count(distinct xb) as n from fact group by g1 order by g1", ["fact"]),
    ("date value", "select g1, count(distinct xd) as n from fact group by g1 order by g1", ["fact"]),
    ("decimal value", "select g1, count(distinct xc) as n from fact group by g1 order by g1", ["fact"]),
    ("float value (-0.0, 0.0, NaN)", "select g1, count(distinct xf) as n from fact group by g1 order by g1", ["fact"]),
    ("bool value", "select g1, count(distinct xl) as n from fact group by g1 order by g1", ["fact"]),
    ("float key (-0.0, 0.0, NaN, NULL)", "select gf, count(distinct xi) as n, count(*) as c from fact group by gf order by gf", ["fact"]),
    ("text key", "select xt, count(distinct g1) as n from fact group by xt order by n, xt", ["fact"]),
    ("count(*) and count(col) beside it", "select g1, count(*) as c, count(xi) as ci, count(distinct xi) as n from fact group by g1 order by g1", ["fact"]),
    ("HAVING on it", "select g1, count(distinct xi) as n from fact group by g1 having count(distinct xi) > 14 order by g1", ["fact"]),
    ("HAVING on another aggregate", "select g1, count(distinct xi) as n from fact group by g1 having count(*) > 2990 order by g1", ["fact"]),
    ("ORDER BY it, LIMIT", "select g2, count(distinct xt) as n from fact group by g2 order by n desc, g2 limit 25", ["fact"]),
    ("expression key and value", "select g1 % 5 as b, count(distinct xi + 1) as n from fact group by g1 % 5 order by b", ["fact"]),
    ("expression of the count in the select list", "select g1, count(distinct xi) * 2 + 1 as n2 from fact group by g1 order by g1", ["fact"]),
    ("the key is the value", "select xi, count(distinct xi) as n from fact group by xi order by xi", ["fact"]),
    ("two keys", "select g1, v % 4 as q, count(distinct xt) as n from fact group by g1, v % 4 order by g1, q", ["fact"]),
    ("groups whose value is all NULL count 0", "select g2, count(distinct xi) as n from fact where xi is null or v < 5 group by g2 order by g2", ["fact"]),
    ("nothing passes: no groups", "select g1, count(distinct xi) as n from fact where v < 0 group by g1", ["fact"]),
    ("join: group from the build side", "select dg, count(distinct xt) as n from fact join dim on k = dk group by dg order by dg", ["fact", "dim"]),
    ("join: value from the build side", "select g1, count(distinct dt) as n from fact join dim on k = dk group by g1 order by g1", ["fact", "dim"]),
    ("join: both from the probe side, filtered", "select g1, count(distinct xi) as n from fact join dim on k = dk where dg < 12 and v < 700 group by g1 order by g1", ["fact", "dim"]),
    ("join: large key", "select g2, count(distinct dt) as n, count(*) as c from fact join dim on k = dk group by g2 order by g2", ["fact", "dim"]),
    ("left join", "select dg, count(distinct xi) as n, count(*) as c from fact left join dim on k = dk group by dg order by dg", ["fact", "dim"]),
    ("join, HAVING on it", "select dg, count(distinct xt) as n from fact join dim on k = dk group by dg having count(distinct xt) > 11 order by dg", ["fact", "dim"]),
]

# the holistic path stays: no note, same answers
HOLISTIC = [
    ("two COUNT(DISTINCT) of different columns", "select g1, count(distinct xi) as a, count(distinct xt) as b from fact group by g1 order by g1", ["fact"]),
    ("no GROUP BY", "select count(distinct xi) as n, count(distinct xt) as m from fact", ["fact"]),
    ("a median beside it", "select g1, count(distinct xi) as n, median(xb) as m from fact group by g1 order by g1", ["fact"]),
    ("a SUM beside it", "select g1, count(distinct xi) as n, sum(xb) as s from fact group by g1 order by g1", ["fact"]),
]


@pytest.mark.parametrize("name,sql,tables", TWO_PHASE, ids=[c[0] for c in TWO_PHASE])
def test_sql_over_tables(probe, world, name, sql, tables):
    res = run_probe(probe, sql, files_of(world, tables), mode="table")
    same(res, world["con"].execute(sql), sql, name)
    assert two_phase(res), res["notes"]


@pytest.mark.parametrize("name,sql,tables", HOLISTIC, ids=[c[0] for c in HOLISTIC])
def test_the_holistic_path_answers_as_before(probe, world, name, sql, tables):
    res = run_probe(probe, sql, files_of(world, tables), mode="table")
    same(res, world["con"].execute(sql), sql, name)
    assert not two_phase(res), res["notes"]


FRAME_CASES = [c for c in TWO_PHASE if not any(w in c[1] for w in ("gf", "xf", "xl"))]       # (an edit script writes no FLOAT or BOOL cells)


@pytest.mark.parametrize("name,sql,tables", FRAME_CASES, ids=[c[0] for c in FRAME_CASES])
def test_sql_over_an_edited_frame(probe, world, name, sql, tables):
    """The fact table is an edited Frame (changed cells, NULLs, appended and removed rows): the same two phases, the same answers as DuckDB over the edited rows."""
    files = files_of(world, tables, "fact")
    files["fact"] = world["paths"]["ffact"]
    res = run_probe(probe, sql, files, mode="frame", edits=world["fact_frame"]["edits"])
    same(res, world["fact_frame"]["con"].execute(sql), sql, name)
    assert two_phase(res), res["notes"]


def test_the_note_counts_the_groups_it_folded(probe, world):
    sql = "select g1, count(distinct xi) as n from fact group by g1"
    res = run_probe(probe, sql, files_of(world, ["fact"]), mode="table")
    inner = world["con"].execute("select count(*) from (select g1, xi from fact group by g1, xi)").fetchone()[0]
    outer = world["con"].execute("select count(distinct g1) + 1 from fact").fetchone()[0]       # (the NULL key is a group)
    assert two_phase(res).startswith(f"{NOTE}: {inner:,} groups of (keys, value) folded into {outer:,} groups")


def test_a_join_stays_fused(probe, world):
    """The point of the rewrite: a holistic aggregate declined the fused join (6M rows gathered); the two phases are an ordinary aggregate over it."""
    sql = "select dg, count(distinct xt) as n from fact join dim on k = dk group by dg"
    res = run_probe(probe, sql, files_of(world, ["fact", "dim"]), mode="table")
    assert any("join path: fused" in n for n in res["notes"]), res["notes"]
    assert not any("materialized" in n and "join path" in n for n in res["notes"]), res["notes"]


# ── the cost rule: many (keys, value) groups keep the holistic path ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def big(tmp_path_factory):
    """1.2M rows (past the trial's 262,144): gh few groups under a large key domain (hashed), gu unique, gp clustered (2 rows a key), x small."""
    d = tmp_path_factory.mktemp("f206big")
    con = oracle_connection()
    con.execute("""create table bt as select i as id, (i % 400) * 1000003 as gh, i as gu, i // 2 as gp,
        case when i % 17 = 0 then null else (i * 7) % 13 end as x, (i * 13) % 1000 as jk from range(1200000) r(i)""")
    con.execute("create table bd as select i as dk, case when i % 50 = 0 then null else i % 7 end as dv from range(1000) r(i)")
    path = d / "bt.parquet"
    con.execute(f"copy bt to '{path}' (format parquet, row_group_size 100000)")
    dpath = d / "bd.parquet"
    con.execute(f"copy bd to '{dpath}' (format parquet)")
    return {"path": path, "dpath": dpath, "con": con}


@pytest.mark.parametrize("mode", ["table", "scan"])
def test_few_groups_over_many_rows_run_two_phases(probe, big, mode):
    """The first 1/32 of the rows (a table) or the first row groups (a scan) make few groups: the trial passes, the rest is folded in."""
    sql = "select gh, count(distinct x) as n, count(*) as c from bt group by gh order by gh"
    res = run_probe(probe, sql, {"bt": big["path"]}, mode=mode)
    same(res, big["con"].execute(sql), sql, mode)
    assert two_phase(res), res["notes"]


@pytest.mark.parametrize("mode", ["table", "scan"])
@pytest.mark.parametrize("key", ["gu", "gp"])
def test_many_groups_keep_the_pair_set_of_the_holistic_path(probe, big, mode, key):
    """A unique key (1.2M pairs) and a clustered one (600,000, one pair for every second row) are declined by the trial: same answers, no two-phase note."""
    sql = f"select {key}, count(distinct x) as n from bt group by {key} having count(distinct x) = 0 or {key} % 4999 = 0 order by {key}"
    res = run_probe(probe, sql, {"bt": big["path"]}, mode=mode)
    same(res, big["con"].execute(sql), sql, f"{mode} {key}")
    assert not two_phase(res), res["notes"]


def test_the_trial_runs_through_a_fused_join(probe, big):
    """A probe table of 1.2M rows joined with a 1,000-row dimension: the trial range goes through the fused join, the answer equals DuckDB; with a pair for every row it is declined."""
    files = {"bt": big["path"], "bd": big["dpath"]}
    few = "select gh, count(distinct dv) as n from bt join bd on jk = dk group by gh order by gh"
    res = run_probe(probe, few, files, mode="table")
    same(res, big["con"].execute(few), few, "few pairs")
    assert two_phase(res) and any("join path: fused" in n for n in res["notes"]), res["notes"]
    many = "select gu, count(distinct dv) as n from bt join bd on jk = dk group by gu having count(distinct dv) = 0 or gu % 4999 = 0 order by gu"
    res = run_probe(probe, many, files, mode="table")
    same(res, big["con"].execute(many), many, "a pair for every row")
    assert not two_phase(res), res["notes"]


# ── random queries ──────────────────────────────────────────────────────────────────────────────────────────────────────

KEYS = ["g1", "g2", "gf", "xt", "g1 % 5", "v % 7", "xi", "dg", "dt"]
VALUES = ["xi", "xb", "xt", "xd", "xc", "xf", "xl", "xi + v", "g1", "dg", "dt"]
FILTERS = ["", "where v < 500", "where xi is not null", "where xb > 20000 and v % 3 = 0", "where v < 30", "where xt <> 'w3'", "where false"]


def random_query(rng):
    """(sql, tables): one key, one COUNT(DISTINCT value), maybe counts beside it, a filter, a join, HAVING; ordered over every output column, so LIMIT is exact."""
    key = rng.choice(KEYS)
    value = rng.choice(VALUES)
    joined = rng.random() < 0.5 or any(c in (key, value) for c in ("dg", "dt"))
    extra = rng.choice(["", ", count(*) as c", ", count(xi) as ci"])
    having = rng.choice(["", f" having count(distinct {value}) > 1", " having count(*) > 3"])
    limit = rng.choice(["", " limit 40"])
    inner = (f"select {key} as k1, count(distinct {value}) as n{extra} from {'fact join dim on k = dk' if joined else 'fact'} {rng.choice(FILTERS)} "
             f"group by {key}{having}")
    cols = ["n"] + ([extra.split(" as ")[1]] if extra else [])
    sql = f"select * from ({inner}) order by n desc, " + "".join(c + ", " for c in cols[1:]) + f"k1{limit}"
    return sql, ["fact", "dim"] if joined else ["fact"]


@pytest.mark.parametrize("seed", range(40))
def test_random_queries_equal_duckdb(probe, world, seed):
    import random
    sql, tables = random_query(random.Random(seed))
    res = run_probe(probe, sql, files_of(world, tables), mode="table")
    same(res, world["con"].execute(sql), sql, f"seed {seed}")


@pytest.mark.parametrize("sql", [
    "select dg, count(distinct xt) as n from fact join dim on k = dk where v < 0 group by dg order by dg",
    "select count(*) as c, sum(v) as s, count(distinct dt) as n from fact join dim on k = dk where v < 0",
    "select dg, count(*) as c, sum(v) as s from dim join fact on dk = k where v < 0 group by dg order by dg",
    "select count(*) as c, count(dk) as cd from fact left join dim on k = dk where v < 0",
    "select count(*) as c, count(dk) as cd from dim left join fact on k = dk where v < 0",
    "select count(*) as c from fact join dim on k = dk where v < 3 and xi < 0",
], ids=["distinct", "global", "dim first", "left, filter on the preserved side", "left, filter on the right (WHERE)", "two filters"])
def test_a_join_side_that_keeps_no_row_is_not_read_again(probe, world, sql):
    """The 150,000-row table's filter keeps nothing: an exact count says so, and the inner join is empty without a second pass over the table (a LEFT join keeps its answer)."""
    res = run_probe(probe, sql, files_of(world, ["fact", "dim"]), mode="table")
    same(res, world["con"].execute(sql), sql, "no rows")
