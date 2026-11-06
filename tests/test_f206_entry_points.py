"""F-206: COUNT(DISTINCT) per group through every entry point equals DuckDB, and each says which path ran.

The world and the edits are F-204's (fact(id, k, g, v) with NULL and dangling keys, dim(dk, w, h) with repeated and NULL keys, inserts / updates / deletes on either);
the same query goes through
  (a) SQL over tables               two phases: `count(DISTINCT): two phases` and `join path: fused`
  (b) SQL over edited Frames        the same, over the edited rows
  (c) the builder over tables       `count_distinct` keeps its own path by design: the builder groups by one key with a stated number of groups (dense only), so a
  (d) the builder over edited Frames  (group, value) table has no layout to fold into; the pair set answers, a join writes the columns it reads first
"""
import pytest

from test_f204_builder_join import probes, con, build_world, run, builder_rows, sql_rows, same, GROUPS     # noqa: F401  (the probes and con fixtures)

TWO_PHASE = "count(DISTINCT): two phases"

# (name, how, where, group column, distinct column): SQL text and the builder's arguments come from the same description (count(*) beside it, as the builder prints it)
SHAPES = [
    ("inner, value from the probe side", "inner", "", "g", "v"),
    ("inner, value from the build side", "inner", "", "g", "w"),
    ("inner, group from the build side", "inner", "", "h", "v"),
    ("inner, filtered", "inner", "v<500;w>10", "g", "v"),
    ("inner, or filter", "inner", "g<3|h>4", "h", "w"),
    ("left join", "left", "", "g", "w"),
    ("left join, filtered on the right", "left", "w>10", "g", "v"),
]


def sql_text(how, where, grp, val):
    conds = []
    for conj in (where.split(";") if where else []):
        conds.append("(" + " or ".join(t.replace("==", "=").replace("!=", "<>") for t in conj.split("|")) + ")")
    cond = (" where " + " and ".join(conds)) if conds else ""
    return f"select {grp}, count(*), count(distinct {val}) from fact {'left ' if how == 'left' else ''}join dim on k = dk{cond} group by {grp}"


@pytest.mark.parametrize("seed,dup,dim_edits", [(0, False, False), (1, True, False), (2, False, True), (3, True, True)])
def test_every_entry_point_equals_duckdb(probes, con, tmp_path, seed, dup, dim_edits):      # noqa: F811
    fact, dim = build_world(tmp_path, seed, dup, dim_edits)
    live_f = [r for r in fact if r is not None]
    live_d = [r for r in dim if r is not None]
    con.execute("drop table if exists fact")
    con.execute("drop table if exists dim")
    con.execute("create table fact(id bigint, k bigint, g bigint, v bigint)")
    con.execute("create table dim(dk bigint, w bigint, h bigint)")
    con.executemany("insert into fact values (?, ?, ?, ?)", live_f)
    con.executemany("insert into dim values (?, ?, ?)", live_d)
    sqlp, bp = probes["sql_probe"], probes["builder_probe"]
    t = tmp_path
    for name, how, where, grp, val in SHAPES:
        sql = sql_text(how, where, grp, val)
        want = con.execute(sql).fetchall()
        what = f"{name} (seed {seed}, dup {dup}, dim edits {dim_edits})"
        a = run(sqlp, ["--mode", "table", sql, f"fact={t}/fact.parquet", f"dim={t}/dim.parquet"])
        same(sql_rows(a), want, "(a) sql table: " + what)
        assert any(n.startswith(TWO_PHASE) for n in a["notes"]) and any("join path: fused" in n for n in a["notes"]), ("(a) " + what, a["notes"])
        b = run(sqlp, ["--mode", "frame", "--edits", str(t / "f.edits")] + (["--redits", str(t / "d.edits")] if dim_edits else []) +
                [sql, f"fact={t}/fact_base.parquet", f"dim={t}/dim_base.parquet"])
        same(sql_rows(b), want, "(b) sql frame: " + what)
        assert any(n.startswith(TWO_PHASE) for n in b["notes"]) and any("join path: fused" in n for n in b["notes"]), ("(b) " + what, b["notes"])
        groups = GROUPS if grp == "g" else 6
        common = ["--how", how, "--on", "k=dk", "--aggs", f"distinct:{val}", "--group", f"{grp}:{groups}"] + (["--where", where] if where else [])
        c = run(bp, ["--mode", "table"] + common + [f"left={t}/fact.parquet", f"right={t}/dim.parquet"])
        same(builder_rows(c, True), want, "(c) builder table: " + what)
        assert not any(n.startswith(TWO_PHASE) for n in c["notes"]), ("(c) " + what, c["notes"])
        d = run(bp, ["--mode", "frame", "--edits", str(t / "f.edits")] + (["--right-mode", "frame", "--redits", str(t / "d.edits")] if dim_edits else []) +
                common + [f"left={t}/fact_base.parquet", f"right={t}/dim_base.parquet"])
        same(builder_rows(d, True), want, "(d) builder frame: " + what)
        assert not any(n.startswith(TWO_PHASE) for n in d["notes"]), ("(d) " + what, d["notes"])
