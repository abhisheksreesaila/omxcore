"""F-207: the presence bitmap of an inner join's build side (no dimension column read) through every entry point, on the random worlds of
tests/test_f204_builder_join.py: SQL over tables, SQL over edited Frames, the builder over tables, the builder over edited Frames. Unique dimension keys give a bitmap
(`join build side: ... presence bitmap`), repeated keys the usual map; NULL keys, dangling keys, appended / changed / removed rows; every answer is DuckDB's.
"""
import pytest

from test_f204_builder_join import probes, con, build_world, sql_text, run, sql_rows, builder_rows, same as same_rows, fast, GROUPS   # noqa: F401

BITS = "presence bitmap"

# (name, how, where, aggs): no column of the dimension is read
NO_DIM_COLUMN = [
    ("inner, a fact filter", "inner", "v<500", "count,sum:v"),
    ("inner, no filter", "inner", "", "count,sum:v,min:v,max:v"),
    ("inner or on a column that is never NULL", "inner", "g<3|g>8", "count,mean:v"),
]


@pytest.mark.parametrize("seed,dup,dim_edits", [(0, False, False), (1, True, False), (2, False, True), (3, True, True), (4, False, False), (5, True, True)])
def test_every_entry_point_equals_duckdb(probes, con, tmp_path, seed, dup, dim_edits):
    fact, dim = build_world(tmp_path, seed, dup, dim_edits)
    con.execute("drop table if exists fact")
    con.execute("drop table if exists dim")
    con.execute("create table fact(id bigint, k bigint, g bigint, v bigint)")
    con.execute("create table dim(dk bigint, w bigint, h bigint)")
    con.executemany("insert into fact values (?, ?, ?, ?)", [r for r in fact if r is not None])
    con.executemany("insert into dim values (?, ?, ?)", [r for r in dim if r is not None])
    sqlp, bp = probes["sql_probe"], probes["builder_probe"]
    t = tmp_path
    keys = [r[0] for r in dim if r is not None and r[0] is not None]
    unique = len(keys) == len(set(keys))                               # (the tables hold the rows after the edits: they may repeat a key the base did not)
    for name, how, where, aggs in NO_DIM_COLUMN:
        for grouped in (True, False):
            sql = sql_text(how, where, aggs, grouped)
            want = con.execute(sql).fetchall()
            what = f"{name} {'grouped' if grouped else 'global'} (seed {seed}, dup {dup}, dim edits {dim_edits})"
            a = run(sqlp, ["--mode", "table", sql, f"fact={t}/fact.parquet", f"dim={t}/dim.parquet"])
            same_rows(sql_rows(a), want, "(a) sql table: " + what)
            fast(a, "(a) " + what)
            line = [n for n in a["notes"] if n.startswith("join build side:")][0]
            assert (BITS in line) == unique, (what, line)                    # unique keys: a bitmap; repeated keys: the usual map
            b = run(sqlp, ["--mode", "frame", "--edits", str(t / "f.edits")] + (["--redits", str(t / "d.edits")] if dim_edits else []) +
                    [sql, f"fact={t}/fact_base.parquet", f"dim={t}/dim_base.parquet"])
            same_rows(sql_rows(b), want, "(b) sql frame: " + what)
            common = ["--how", how, "--on", "k=dk", "--aggs", aggs] + (["--where", where] if where else []) + (["--group", f"g:{GROUPS}"] if grouped else [])
            c = run(bp, ["--mode", "table"] + common + [f"left={t}/fact.parquet", f"right={t}/dim.parquet"])
            same_rows(builder_rows(c, grouped), want, "(c) builder table: " + what)
            fast(c, "(c) " + what)
            line = [n for n in c["notes"] if n.startswith("join build side:")][0]
            assert (BITS in line) == unique, (what, line)
            d = run(bp, ["--mode", "frame", "--edits", str(t / "f.edits")] + (["--right-mode", "frame", "--redits", str(t / "d.edits")] if dim_edits else []) +
                    common + [f"left={t}/fact_base.parquet", f"right={t}/dim_base.parquet"])
            same_rows(builder_rows(d, grouped), want, "(d) builder frame: " + what)
