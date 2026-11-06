"""F-202 over Frames: the same joins as tests/test_f202_multiway.py, but one table is a Frame with an edit script applied (changed cells, appended rows,
removed rows: the overlay's mask and delta) and the answer is compared with DuckDB over the same edits. The probe (mojo/tools/sql_probe.mojo) applies the
script to the FIRST file it is given, so each case runs twice: with the fact table first and with a dimension first. The notes say which path ran.
"""

import datetime
import random

import pytest

from conftest import oracle_connection
from test_mojo_sql import probe, run_probe, got_rows          # noqa: F401  (the probe fixture builds mojo/tools/sql_probe.mojo)
from mx_oracle import assert_equal, duck_kind

F_COLS = [("a", "int"), ("b", "int"), ("x", "dec2"), ("d", "date"), ("s", "text")]
DA_COLS = [("an", "text"), ("ak", "int"), ("seg", "int")]
WORDS = ["red", "green", "blue", "amber", "zzz-new"]       # (lower case: omxcore orders text case-insensitively, DuckDB by bytes, so mixed-case ties would differ in ORDER BY)


def _lit(kind, v):
    if v == "":
        return "NULL"
    return v if kind in ("int", "dec2") else "'" + v.replace("'", "''") + "'"


def make_edits(con, table, key, cols, n_rows, seed, n_edits, key_range):
    """An edit script in the probe's format, applied to DuckDB's table `table` (which has a `rid` column) the same way."""
    rng = random.Random(seed)
    lines = []
    alive = set(range(n_rows))
    next_rid = n_rows
    names = [key] + [c for c, _ in cols]

    def value(name, kind):
        if kind == "int":
            return str(rng.randint(0, key_range))
        if kind == "dec2":
            return f"{rng.randint(0, 20000) / 100:.2f}"
        if kind == "date":
            return (datetime.date(2020, 1, 1) + datetime.timedelta(days=rng.randint(0, 900))).isoformat()
        return rng.choice(WORDS)

    for _ in range(n_edits):
        what = rng.randint(0, 9)
        if what <= 5:
            row = rng.randrange(n_rows)
            while row not in alive:
                row = rng.randrange(n_rows)
            name, kind = rng.choice(cols)
            v = "" if what == 5 else value(name, kind)
            lines.append(f"set\t{name}\t{row}\t{v}")
            con.execute(f"update {table} set {name} = {_lit(kind, v)} where rid = {row}")
        elif what in (6, 7):
            vals = [str(next_rid)] + ["" if rng.random() < 0.15 else value(n, k) for n, k in cols]
            lines.append("append\t" + "\t".join(vals))
            kinds = [("int")] + [k for _, k in cols]
            con.execute(f"insert into {table} (rid, {', '.join(names)}) values ({next_rid}, {', '.join(_lit(k, v) for k, v in zip(kinds, vals))})")
            alive.add(next_rid)
            next_rid += 1
        else:
            row = rng.randrange(n_rows)
            if row in alive:
                lines.append(f"remove\t{row}")
                con.execute(f"delete from {table} where rid = {row}")
                alive.discard(row)
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("f202frames")
    base = oracle_connection()
    base.execute("create table nat as select i as nk, 'N' || lpad(i::varchar, 2, '0') as nname from range(25) r(i)")
    base.execute("create table da as select i as id, 'A' || (i % 97) as an, case when i % 41 = 0 then null else i % 25 end as ak, i % 5 as seg from range(3000) r(i)")
    base.execute("create table db as select i as id, 'B-' || i as bn, (i * 7) % 25 as bk, ((i % 13) * 0.5)::decimal(8,1) as w from range(4000) r(i)")
    base.execute("create table dbig as select i as id, 'G' || (i % 23) as gn, i % 9 as seg from range(100000) r(i)")                 # 100,000 rows: at least a third of f's
    base.execute("""create table f as select i as id,
        case when i % 31 = 0 then null when i % 53 = 0 then 90000 + i else (i * 7) % 3000 end as a,
        case when i % 37 = 0 then null else ((i // 25) * 11) % 4000 end as b,
        case when i % 19 = 0 then null else ((i % 700) * 0.25)::decimal(10,2) end as x,
        date '2020-01-01' + (i % 800)::int as d,
        (['red','Green','blue','amber'])[1 + i % 4] as s from range(200000) r(i)""")
    paths = {}
    for t, rg in [("nat", 25), ("da", 700), ("db", 900), ("f", 50000), ("dbig", 25000)]:
        paths[t] = d / f"{t}.parquet"
        base.execute(f"copy {t} to '{paths[t]}' (format parquet, row_group_size {rg})")
    # two oracles: the fact table edited, a dimension edited (each over the other tables as they are)
    out = {"paths": paths}
    for which, table, cols, n_rows, key_range in (("f", "f", F_COLS, 200000, 3100), ("da", "da", DA_COLS, 3000, 30), ("dbig", "dbig", [("gn", "text"), ("seg", "int")], 100000, 8)):
        con = oracle_connection()
        for t in ("nat", "da", "db", "f", "dbig"):
            if t == table:
                con.execute(f"create table {t} as select row_number() over () - 1 as rid, * from read_parquet('{paths[t]}')")
            else:
                con.execute(f"create table {t} as select * from read_parquet('{paths[t]}')")
        script = make_edits(con, table, "id", cols, n_rows, 7 if which == "f" else 8, 60, key_range)
        edits = d / f"{which}.edits"
        edits.write_text(script)
        out[which] = {"con": con, "edits": edits}
    return out


def run_case(probe, world, sql, tables, frame_table):
    """`sql` over `tables`, `frame_table` first (so it is the Frame), compared with DuckDB over the same edits."""
    order = [frame_table] + [t for t in tables if t != frame_table]
    files = {t: world["paths"][t] for t in order}
    res = run_probe(probe, sql, files, mode="frame", edits=world[frame_table]["edits"])
    cur = world[frame_table]["con"].execute(sql)
    want = cur.fetchall()
    for i, desc in enumerate(cur.description):
        dk = duck_kind(desc[1])
        c = res["columns"][i]
        assert c["kind"] == dk[0] and (dk[0] != "DECIMAL" or c["scale"] == dk[1]), f"column {i} of `{sql}`: got {c}, DuckDB has {dk}"
    assert_equal(got_rows(res), want, "ORDER BY" in sql, f"frame {frame_table}: {sql}")
    return res


def has(res, text):
    return any(text in n for n in res["notes"])


Q_OR = """SELECT n1.nname AS sn, n2.nname AS cn, extract(year FROM f.d) AS y, sum(f.x) AS total, count(*) AS c
FROM da, f, db, nat n1, nat n2
WHERE da.id = f.a AND db.id = f.b AND da.ak = n1.nk AND db.bk = n2.nk
  AND ((n1.nname = 'N03' AND n2.nname = 'N07') OR (n1.nname = 'N07' AND n2.nname = 'N03'))
  AND f.d BETWEEN date '2020-06-01' AND date '2021-12-31'
GROUP BY 1, 2, 3 ORDER BY 1, 2, 3"""

CASES = [
    ("or_pairs", Q_OR, ["da", "f", "db", "nat"]),
    ("star_cross", "SELECT da.an, count(*), sum(f.x) FROM f, da, db WHERE f.a = da.id AND f.b = db.id AND da.ak = db.bk GROUP BY da.an ORDER BY da.an", ["f", "da", "db"]),
    ("star_nat", "SELECT nat.nname, sum(f.x) FROM f, da, db, nat WHERE f.a = da.id AND f.b = db.id AND da.ak = db.bk AND da.ak = nat.nk AND f.d > date '2021-01-01' GROUP BY nat.nname ORDER BY nat.nname",
     ["f", "da", "db", "nat"]),
    ("star_filtered", "SELECT db.bk, count(*) FROM f, da, db WHERE f.a = da.id AND f.b = db.id AND da.ak = db.bk AND da.seg = 2 AND db.w > 3 GROUP BY db.bk ORDER BY db.bk", ["f", "da", "db"]),
    ("reduced", "SELECT da.an, count(*), sum(f.x) FROM da, f, db WHERE da.id = f.a AND f.b = db.id AND da.seg = 2 AND da.ak = 7 AND f.d >= date '2020-02-01' AND f.x > 10 GROUP BY da.an ORDER BY da.an",
     ["da", "f", "db"]),
    ("many_groups_text", "SELECT f.id, da.an, db.bn, sum(f.x) AS t FROM f, da, db WHERE f.a = da.id AND f.b = db.id GROUP BY f.id, da.an, db.bn ORDER BY t DESC, f.id LIMIT 10", ["f", "da", "db"]),
    ("order_by_text", "SELECT da.an, f.d, sum(f.x) AS t FROM f, da, db WHERE f.a = da.id AND f.b = db.id GROUP BY da.an, f.d ORDER BY da.an, f.d DESC LIMIT 60", ["f", "da", "db"]),
    ("left_join_text", "SELECT f.id, da.an, sum(f.x) AS t FROM f LEFT JOIN da ON f.a = da.id GROUP BY f.id, da.an ORDER BY t DESC NULLS LAST, f.id LIMIT 10", ["f", "da"]),
    ("correlated", "SELECT da.id, da.an FROM da WHERE da.seg = 1 AND da.id < 600 AND (SELECT min(f.x) FROM f, db WHERE f.a = da.id AND f.b = db.id AND db.bk < 12) > 40 ORDER BY da.id", ["da", "f", "db"]),
    ("chain", "SELECT nat.nname, count(*), sum(f.x) FROM f, db, nat WHERE f.b = db.id AND db.bk = nat.nk AND f.s <> 'red' GROUP BY nat.nname ORDER BY nat.nname", ["f", "db", "nat"]),
]


STAR_CASES = {"star_cross", "star_nat", "star_filtered", "many_groups_text", "order_by_text"}
STREAMED_STAR_CASES = STAR_CASES


@pytest.mark.parametrize("frame_table", ["f", "da"])
@pytest.mark.parametrize("name,sql,tables", CASES, ids=[c[0] for c in CASES])
def test_a_join_over_a_frame_equals_duckdb(probe, world, name, sql, tables, frame_table):
    if frame_table not in tables:
        pytest.skip("the case does not read this table")
    res = run_case(probe, world, sql, tables, frame_table)
    if name in STAR_CASES:                                       # the fast path stays on whether the fact table or a dimension is the Frame
        assert has(res, "a star from f"), res["notes"]
        assert has(res, "join path: fused: a star"), res["notes"]
    if name == "reduced" and frame_table == "f":                 # the Frame is read where its key has a partner: its own mask narrowed, its changed rows kept
        assert has(res, "runtime join filter: f is read only where its key has a partner"), res["notes"]
    if name in STREAMED_STAR_CASES and frame_table == "f":       # a Frame fact table streams through the star (its changed rows apart): never read into a table
        assert has(res, "a scan or Frame streamed chunk by chunk"), res["notes"]
        assert has(res, "changed or appended rows"), res["notes"]
    print("PATHS", name, frame_table, [n for n in res["notes"] if n.startswith(("join order", "join path", "star", "runtime join filter", "ran:", "join build"))][:8])


def test_a_frame_dimension_at_a_third_of_the_fact_table_still_streams(probe, world):
    """100,000 rows against 200,000 is over the 1/3 ratio: the Frame is a probe candidate and stays streamed (not read into a table); the answer is DuckDB's."""
    sql = "SELECT dbig.gn, count(*), sum(f.x) FROM f, dbig, db WHERE f.a = dbig.id AND f.b = db.id GROUP BY dbig.gn ORDER BY dbig.gn"
    res = run_case(probe, world, sql, ["f", "dbig", "db"], "dbig")
    assert has(res, "dbig.id an unknown number in 100,000"), res["notes"]       # it stayed a streamed leaf (no distinct counts: nothing was read of it to count)
    assert not has(res, "a star from f"), res["notes"]


def test_a_frame_dimension_under_a_third_is_read_and_can_be_a_star_dimension(probe, world):
    sql = "SELECT da.an, count(*), sum(f.x) FROM f, da, db WHERE f.a = da.id AND f.b = db.id GROUP BY da.an ORDER BY da.an"
    res = run_case(probe, world, sql, ["f", "da", "db"], "da")
    assert has(res, "a star from f"), res["notes"]
