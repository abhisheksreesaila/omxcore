"""F-205: a join's build side is not copied when its filter keeps most rows: the filter runs as one flag byte a row over the whole table and the hash map is
built from the table with those flags as the keys' `ok` bytes (`join build filter:` in the notes). A filter that keeps few rows still writes the (small) copy, and a
filtered side that ends up the probe side is gathered by its flags. Every answer equals DuckDB, through SQL over tables and SQL over edited Frames (a Frame's
removed rows keep the flag path; changed or appended rows decline it, by design, and the answer is still DuckDB's).

World: fact (150,000 rows: NULL and dangling keys), dim (90,000 rows: keys repeat twice, NULL keys, a NULL value, text, a date), small (1,500 rows).
"""
import random

import pytest

from conftest import oracle_connection
from test_mojo_sql import probe, run_probe, got_rows          # noqa: F401  (the probe fixture builds mojo/tools/sql_probe.mojo)
from test_f202_frames import make_edits
from mx_oracle import assert_equal, duck_kind

FLAG = "join build filter:"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("f205")
    base = oracle_connection()
    base.execute("""create table fact as select i as id,
        case when i % 20 = 0 then null when i % 11 = 0 then 70000 + i else (i * 7) % 45000 end as k,
        i % 1000 as v from range(150000) r(i)""")
    base.execute("""create table dim as select i as did,
        case when i % 33 = 0 then null else (i * 3) % 45000 end as dk,
        case when i % 25 = 0 then null else i % 100 end as dv,
        case when i % 40 = 0 then 'a special request and more requests' when i % 7 = 0 then 'special only' else 'plain ' || (i % 13) end as dt,
        date '2021-01-01' + (i % 400)::int as dd from range(90000) r(i)""")
    base.execute("create table small as select i as sk, (i * 31) % 45000 as sv from range(1500) r(i)")
    paths = {}
    for t, rg in [("fact", 40000), ("dim", 30000), ("small", 700)]:
        paths[t] = d / f"{t}.parquet"
        base.execute(f"copy {t} to '{paths[t]}' (format parquet, row_group_size {rg})")
    out = {"paths": paths, "dir": d, "con": base}
    # two edited oracles: fact edited (changes, appends, removals), dim with changed cells
    spec = {"fact": ("fact", [("k", "int"), ("v", "int")], 150000, 45000),
            "dim_changed": ("dim", [("dk", "int"), ("dv", "int"), ("dt", "text"), ("dd", "date")], 90000, 45000)}
    for which, (table, cols, n_rows, key_range) in spec.items():
        con = oracle_connection()
        for t in ("fact", "dim", "small"):
            if t == table:
                con.execute(f"create table {t} as select row_number() over () - 1 as rid, * from read_parquet('{paths[t]}')")
            else:
                con.execute(f"create table {t} as select * from read_parquet('{paths[t]}')")
        key = "id" if table == "fact" else "did"
        script = make_edits(con, table, key, cols, n_rows, 5, 40, key_range)
        edits = d / f"{which}.edits"
        edits.write_text(script)
        out[which] = {"con": con, "edits": edits, "table": table}
    # dim with removals only
    con = oracle_connection()
    for t in ("fact", "dim", "small"):
        if t == "dim":
            con.execute(f"create table dim as select row_number() over () - 1 as rid, * from read_parquet('{paths['dim']}')")
        else:
            con.execute(f"create table {t} as select * from read_parquet('{paths[t]}')")
    rnd = random.Random(11)
    gone = sorted(rnd.sample(range(90000), 300))
    for r in gone:
        con.execute(f"delete from dim where rid = {r}")
    (d / "dim_removed.edits").write_text("".join(f"remove\t{r}\n" for r in gone))
    out["dim_removed"] = {"con": con, "edits": d / "dim_removed.edits", "table": "dim"}
    # dim with appended rows only (nothing changed or removed): the new rows live outside the table, so the flag path must decline
    con = oracle_connection()
    for t in ("fact", "dim", "small"):
        if t == "dim":
            con.execute(f"create table dim as select row_number() over () - 1 as rid, * from read_parquet('{paths['dim']}')")
        else:
            con.execute(f"create table {t} as select * from read_parquet('{paths[t]}')")
    lines = []
    for j in range(40):
        rid = 90000 + j
        vals = (rid, j * 1000 % 45000, j % 100, "plain 1" if j % 2 else "special requests", f"2021-03-{1 + j % 28:02d}")
        lines.append("append\t" + "\t".join(str(v) for v in vals))
        con.execute(f"insert into dim (rid, did, dk, dv, dt, dd) values ({rid}, {vals[0]}, {vals[1]}, {vals[2]}, '{vals[3]}', date '{vals[4]}')")
    (d / "dim_appended.edits").write_text("\n".join(lines) + "\n")
    out["dim_appended"] = {"con": con, "edits": d / "dim_appended.edits", "table": "dim"}
    return out


def files_of(world, tables, first=None):
    order = ([first] if first else []) + [t for t in tables if t != first]
    return {t: world["paths"][t] for t in order}


def run_table(probe, world, sql, tables):
    res = run_probe(probe, sql, files_of(world, tables), mode="table")
    return res, world["con"].execute(sql)


def same(res, cur, sql, label):
    want = cur.fetchall()
    for i, desc in enumerate(cur.description):
        dk = duck_kind(desc[1])
        c = res["columns"][i]
        assert c["kind"] == dk[0] and (dk[0] != "DECIMAL" or c["scale"] == dk[1]), f"column {i} of `{sql}`: got {c}, DuckDB has {dk}"
    assert_equal(got_rows(res), want, "ORDER BY" in sql.upper(), f"{label}: {sql}")


def has(res, text):
    return any(text in n for n in res["notes"])


def note(res, text):
    hits = [n for n in res["notes"] if text in n]
    assert hits, (text, res["notes"])
    return hits[0]


# (name, sql, tables, flags): `flags` says the build side was not copied (a note), else it was gathered or read as before
LEFT_HIGH = "select count(*) as c, count(dv) as cv, sum(dv) as s, max(did) as m from fact left join dim on k = dk and dv < 90"
Q13 = ("select id, count(did) as cnt from fact left join dim on k = dk and dt not like '%special%requests%' group by id order by cnt desc, id limit 50")
CASES = [
    ("left, the ON filter keeps 90%", LEFT_HIGH, ["fact", "dim"], True),
    ("left, group by the probe's value", "select v % 10 as g, count(dv) as cv, sum(dv) as s, count(*) as c from fact left join dim on k = dk and dv < 90 group by v % 10 order by g",
     ["fact", "dim"], True),
    ("left, the ON filter keeps 10%", "select count(*) as c, count(dv) as cv, sum(dv) as s from fact left join dim on k = dk and dv < 10", ["fact", "dim"], False),
    ("left, the ON filter keeps nothing", "select count(*) as c, count(dv) as cv, sum(dv) as s from fact left join dim on k = dk and dv < 0", ["fact", "dim"], False),
    ("left, the Q13 shape: text NOT LIKE in the ON", Q13, ["fact", "dim"], True),
    ("left, a date filter on the right", "select count(did) as c, min(dd) as lo, max(dd) as hi from fact left join dim on k = dk and dd < date '2021-11-01'", ["fact", "dim"], True),
    ("left, the right filter reads two columns", "select count(*) as c, sum(dv) as s from fact left join dim on k = dk and dv < 95 and dt <> 'plain 3'", ["fact", "dim"], True),
    ("inner, the filtered table is the smaller side and builds", "select v % 7 as g, count(*) as c, sum(dv) as s from fact join dim on k = dk where dv < 90 group by v % 7 order by g",
     ["fact", "dim"], True),
    ("inner, a filter on each side", "select count(*) as c, sum(dv) as s, sum(v) as sv from fact join dim on k = dk where dv < 90 and v < 500", ["fact", "dim"], True),
    ("inner, the filtered table is the larger side and probes", "select count(*) as c, sum(dv) as s from small join dim on sv = dk where dv < 90", ["small", "dim"], False),
    ("inner, nothing passes", "select count(*) as c, sum(dv) as s from fact join dim on k = dk where dv < 0", ["fact", "dim"], False),
]


@pytest.mark.parametrize("name,sql,tables,flags", CASES, ids=[c[0] for c in CASES])
def test_sql_over_tables(probe, world, name, sql, tables, flags):
    res, cur = run_table(probe, world, sql, tables)
    same(res, cur, sql, name)
    if flags:
        assert "no filtered copy is written" in note(res, FLAG)
    else:
        assert not has(res, FLAG), res["notes"]


def test_the_flag_note_counts_the_rows_that_pass(probe, world):
    res, cur = run_table(probe, world, LEFT_HIGH, ["fact", "dim"])
    kept = world["con"].execute("select count(*) from dim where dv < 90").fetchone()[0]
    assert note(res, FLAG) == f"join build filter: dim keeps {kept:,} of 90,000 rows: the filter's flags are the keys' ok bytes, no filtered copy is written"
    assert has(res, "join path: fused")


def test_a_null_key_and_the_filter_both_keep_a_row_out(probe, world):
    sql = "select count(*) as c, count(did) as cd from fact left join dim on k = dk and dv < 90 and dk is not null"
    res, cur = run_table(probe, world, sql, ["fact", "dim"])
    same(res, cur, sql, "null keys")


# ── edited Frames ─────────────────────────────────────────────────────────────────────────────────────────

def run_frame(probe, world, sql, tables, which):
    first = world[which]["table"]
    res = run_probe(probe, sql, files_of(world, tables, first), mode="frame", edits=world[which]["edits"])
    return res, world[which]["con"].execute(sql)


FRAME_SQL = [
    LEFT_HIGH,
    "select count(*) as c, sum(dv) as s from fact join dim on k = dk where dv < 90",
    Q13,
]


@pytest.mark.parametrize("sql", FRAME_SQL)
def test_fact_frame_edited_build_side_a_table(probe, world, sql):
    """The probe table is an edited Frame (changes, appends, removals); the build side is a plain table: its flags still replace its copy."""
    res, cur = run_frame(probe, world, sql, ["fact", "dim"], "fact")
    same(res, cur, sql, "fact frame")
    assert "no filtered copy is written" in note(res, FLAG)


@pytest.mark.parametrize("sql", FRAME_SQL)
def test_dim_frame_with_removed_rows_keeps_the_flag_path(probe, world, sql):
    """A Frame whose edits are removals only: its mask hides the rows (their flag stays 0), nothing is copied."""
    res, cur = run_frame(probe, world, sql, ["fact", "dim"], "dim_removed")
    same(res, cur, sql, "dim frame, removals")
    assert "no filtered copy is written" in note(res, FLAG)


@pytest.mark.parametrize("sql", FRAME_SQL)
def test_dim_frame_with_appended_rows_only_declines_the_flags(probe, world, sql):
    """Appended rows are a chunk outside the table: flags over the table would miss them, so the guard declines and the filtered rows are read (appends included)."""
    res, cur = run_frame(probe, world, sql, ["fact", "dim"], "dim_appended")
    same(res, cur, sql, "dim frame, appends only")
    assert not has(res, FLAG), res["notes"]


def test_left_join_with_a_where_filter_on_the_preserved_side(probe, world):
    """The preserved (left, probing) side is a 150,000-row table with a WHERE filter: it never builds, so no flags stay on it; the right side may still build from its flags."""
    sql = "select count(*) as c, count(did) as cd, sum(v) as sv from fact left join dim on k = dk where v < 500"
    res, cur = run_table(probe, world, sql, ["fact", "dim"])
    same(res, cur, sql, "left join, filter on the left")
    assert not has(res, FLAG), res["notes"]
    sql = "select count(*) as c, count(dv) as cv from fact left join dim on k = dk and dv < 90 where v < 500"
    res, cur = run_table(probe, world, sql, ["fact", "dim"])
    same(res, cur, sql, "left join, filter on both sides")
    assert "dim keeps" in note(res, FLAG)


@pytest.mark.parametrize("sql", FRAME_SQL)
def test_dim_frame_with_changed_rows_is_read_as_before(probe, world, sql):
    """Changed and appended rows live outside the table: the flag path declines (the filtered rows are read), and the answer is DuckDB's."""
    res, cur = run_frame(probe, world, sql, ["fact", "dim"], "dim_changed")
    same(res, cur, sql, "dim frame, changes")
    assert not has(res, FLAG), res["notes"]


# ── every entry point on one world: SQL over tables, SQL over edited Frames, the builder over tables and over Frames ────────────

from test_f204_builder_join import (probes, run, sql_rows, builder_rows, same as same_rows, fast, make_edits as make_int_edits, write, FACT_SPEC, DIM_SPEC, sql_text, GROUPS)   # noqa: F401,E402


def big_int_world(tmp_path, seed):
    rnd = random.Random(seed)
    n, m = 150000, 90000
    fact = [[i, None if rnd.random() < 0.04 else rnd.randrange(0, 60000), rnd.randrange(0, GROUPS), None if rnd.random() < 0.05 else rnd.randrange(0, 1000)] for i in range(n)]
    dim = [[None if rnd.random() < 0.03 else rnd.randrange(0, 45000), None if rnd.random() < 0.08 else rnd.randrange(0, 100), rnd.randrange(0, 6)] for _ in range(m)]
    write(tmp_path / "fact_base.parquet", ["id", "k", "g", "v"], fact)
    write(tmp_path / "dim_base.parquet", ["dk", "w", "h"], dim)
    (tmp_path / "f.edits").write_text(make_int_edits(rnd, fact, FACT_SPEC, 300, 80, 200))
    (tmp_path / "d.edits").write_text(make_int_edits(rnd, dim, DIM_SPEC, 0, 0, 150))         # the dimension: removed rows only
    write(tmp_path / "fact.parquet", ["id", "k", "g", "v"], fact)
    write(tmp_path / "dim.parquet", ["dk", "w", "h"], dim)
    return fact, dim


ENTRY_SHAPES = [
    ("inner, the filtered dimension builds", "inner", "w>10", "count,sum:v,sum:w"),
    ("inner, a filter on each side", "inner", "w>10;v<500", "count,sum:w,mean:v"),
    ("left, the filter on the dimension", "left", "w>10", "count,sum:v,countof:w"),
]


@pytest.mark.parametrize("seed", [0, 1])
def test_every_entry_point_equals_duckdb_on_a_big_world(probes, tmp_path, seed):
    fact, dim = big_int_world(tmp_path, seed)
    con = oracle_connection()
    con.execute("create table fact(id bigint, k bigint, g bigint, v bigint)")
    con.execute("create table dim(dk bigint, w bigint, h bigint)")
    con.executemany("insert into fact values (?, ?, ?, ?)", [r for r in fact if r is not None])
    con.executemany("insert into dim values (?, ?, ?)", [r for r in dim if r is not None])
    sqlp, bp = probes["sql_probe"], probes["builder_probe"]
    t = tmp_path
    for name, how, where, aggs in ENTRY_SHAPES:
        sql = sql_text(how, where, aggs, True)
        want = con.execute(sql).fetchall()
        what = f"{name} (seed {seed})"
        a = run(sqlp, ["--mode", "table", sql, f"fact={t}/fact.parquet", f"dim={t}/dim.parquet"])
        same_rows(sql_rows(a), want, "(a) sql table: " + what)
        fast(a, "(a) " + what)
        assert has(a, FLAG) == (how == "inner"), (what, a["notes"])        # (a left join filtered in WHERE is an inner join: its filter is on the right side too)
        b = run(sqlp, ["--mode", "frame", "--edits", str(t / "f.edits"), "--redits", str(t / "d.edits"), sql, f"fact={t}/fact_base.parquet", f"dim={t}/dim_base.parquet"])
        same_rows(sql_rows(b), want, "(b) sql frame: " + what)
        fast(b, "(b) " + what)
        assert has(b, FLAG) == (how == "inner"), (what, b["notes"])        # the dimension Frame has removed rows only: its flags stand in for its copy
        common = ["--how", how, "--on", "k=dk", "--aggs", aggs, "--where", where, "--group", f"g:{GROUPS}"]
        c = run(bp, ["--mode", "table"] + common + [f"left={t}/fact.parquet", f"right={t}/dim.parquet"])
        same_rows(builder_rows(c, True), want, "(c) builder table: " + what)
        fast(c, "(c) " + what)
        d = run(bp, ["--mode", "frame", "--edits", str(t / "f.edits"), "--right-mode", "frame", "--redits", str(t / "d.edits")] + common +
                [f"left={t}/fact_base.parquet", f"right={t}/dim_base.parquet"])
        same_rows(builder_rows(d, True), want, "(d) builder frame: " + what)
        fast(d, "(d) " + what)
