"""F-204: a join and a group-by through every entry point of the engine give DuckDB's answer, and the fast path runs on each.

The same query goes through
  (a) SQL over tables               (sql_probe --mode table, the world after its edits written as a file)
  (b) SQL over edited Frames        (sql_probe --mode frame, `Catalog.add_frame`: the file as it was, then the edit script)
  (c) the builder over tables       (builder_probe --mode table: `Query.join` + `where` + `group_by` + `agg`)
  (d) the builder over edited Frames (builder_probe --mode frame, the fact Frame and, in some worlds, the dimension Frame too)
and every answer equals DuckDB's on the world after the edits (inserts, updates, deletes, NULL keys, repeated keys, LEFT / semi / anti joins). Each entry
point's notes must say `join path: fused` (the build side's keys are unique, repeat up to EXPAND_MAX_REP times, or the join is a semi / anti one).
"""
import fcntl
import json
import random
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import oracle_connection
from engine_sources import engine_sources

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
GROUPS = 12


@pytest.fixture(scope="session")
def probes():
    """sql_probe and builder_probe, built once (a lock keeps xdist workers apart)."""
    BUILD.mkdir(exist_ok=True)
    out = {}
    for name in ("sql_probe", "builder_probe"):
        path = BUILD / name
        sources = engine_sources(ROOT / f"mojo/tools/{name}.mojo")
        with open(BUILD / f".{name}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not path.exists() or path.stat().st_mtime < max(p.stat().st_mtime for p in sources):
                subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / f"mojo/tools/{name}.mojo"), str(path)], check=True, capture_output=True, text=True)
        out[name] = path
    return out


# ── a random world: fact(id, k, g, v) and dim(dk, w, h), then edits to either ─────────────────────────────────────────

def make_world(rnd, dup):
    n, m = 3000, 280
    fact = [[i, None if rnd.random() < 0.04 else rnd.randrange(0, 360), rnd.randrange(0, GROUPS), None if rnd.random() < 0.05 else rnd.randrange(0, 1000)] for i in range(n)]
    if dup:                                                   # keys repeat (up to 4 times) and some are NULL
        dim = [[None if rnd.random() < 0.03 else rnd.randrange(0, 300), None if rnd.random() < 0.08 else rnd.randrange(0, 100), rnd.randrange(0, 6)] for _ in range(int(m * 1.6))]
    else:
        keys = rnd.sample(range(0, 300), m)
        dim = [[keys[i], None if rnd.random() < 0.08 else rnd.randrange(0, 100), rnd.randrange(0, 6)] for i in range(m)]
    return fact, dim


def make_edits(rnd, rows, spec, n_sets, n_append, n_remove):
    """An edit script (the sql_probe / builder_probe format) and the rows it leaves (`rows` is mutated; None: removed). `spec` per column: ("id", "id") (the
    row's number, never set), (name, "int", high, nullable) or (name, "group") (0 .. GROUPS-1)."""
    names = [s[0] for s in spec]
    value = lambda s: rnd.randrange(0, GROUPS) if s[1] == "group" else (None if s[3] and rnd.random() < 0.12 else rnd.randrange(0, s[2]))
    lines = []
    changeable = [c for c, s in enumerate(spec) if s[1] != "id"]
    originals = len(rows)
    for _ in range(n_sets):
        r = rnd.randrange(originals)
        c = rnd.choice(changeable)
        rows[r][c] = value(spec[c])
        lines.append(f"set\t{names[c]}\t{r}\t{'' if rows[r][c] is None else rows[r][c]}")
    for _ in range(n_append):
        row = [len(rows) if s[1] == "id" else value(s) for s in spec]
        rows.append(row)
        lines.append("append\t" + "\t".join("" if v is None else str(v) for v in row))
    for r in rnd.sample(range(originals), n_remove):
        rows[r] = None
        lines.append(f"remove\t{r}")
    return "\n".join(lines) + "\n"


FACT_SPEC = [("id", "id"), ("k", "int", 400, True), ("g", "group"), ("v", "int", 1000, True)]
DIM_SPEC = [("dk", "int", 330, True), ("w", "int", 100, True), ("h", "int", 6, False)]


def write(path, names, rows):
    live = [r for r in rows if r is not None]
    pq.write_table(pa.table({n: pa.array([r[i] for r in live], pa.int64()) for i, n in enumerate(names)}), path, row_group_size=900)


@pytest.fixture(scope="module")
def con():
    return oracle_connection()


def build_world(tmp_path, seed, dup, dim_edits):
    rnd = random.Random(seed * 7919 + (1 if dup else 0))
    fact, dim = make_world(rnd, dup)
    write(tmp_path / "fact_base.parquet", ["id", "k", "g", "v"], fact)
    write(tmp_path / "dim_base.parquet", ["dk", "w", "h"], dim)
    f_script = make_edits(rnd, fact, FACT_SPEC, 90, 25, 40)
    d_script = make_edits(rnd, dim, DIM_SPEC, 25, 6, 12) if dim_edits else ""
    (tmp_path / "f.edits").write_text(f_script)
    (tmp_path / "d.edits").write_text(d_script)
    write(tmp_path / "fact.parquet", ["id", "k", "g", "v"], fact)
    write(tmp_path / "dim.parquet", ["dk", "w", "h"], dim)
    return fact, dim


# ── the queries: one description, written as SQL, and as the builder's arguments ──────────────────────────────────────

# (name, how, where, aggs). `where`: conjuncts joined by `;`, each a `|` list of `column op integer`; aggs: the builder_probe spelling (count, sum:v, ...).
SHAPES = [
    ("inner by g", "inner", "", "count,sum:v,sum:w,min:w,max:v"),
    ("inner filtered", "inner", "v<500;w>10", "count,sum:w,mean:v"),
    ("inner or on non-null columns", "inner", "g<3|h>4", "count,sum:v"),
    ("inner product", "inner", "v>100", "count,sum:v*w,countof:w"),
    ("left by g", "left", "", "count,sum:w,countof:w,min:w,max:w"),
    ("left filtered on the right", "left", "w>10", "count,sum:v"),
    ("semi by g", "semi", "v>5", "count,sum:v"),
    ("anti by g", "anti", "", "count,sum:v,max:v"),
]


def sql_text(how, where, aggs, grouped):
    sels = ["count(*)"]
    for spec in aggs.split(","):
        kind, _, arg = spec.partition(":")
        if kind == "count":
            continue
        fn = {"countof": "count", "mean": "avg"}.get(kind, kind)
        sels.append(f"{fn}({arg})")
    conds = []
    for conj in (where.split(";") if where else []):
        conds.append("(" + " or ".join(t.replace("==", "=").replace("!=", "<>") for t in conj.split("|")) + ")")
    cond = (" where " + " and ".join(conds)) if conds else ""
    if how == "inner" or how == "left":
        frm = f"from fact {'left ' if how == 'left' else ''}join dim on k = dk{cond}"
    else:
        sub = f"{'not ' if how == 'anti' else ''}exists (select 1 from dim where dk = k)"
        frm = f"from fact where {sub}" + ("".join(f" and {c}" for c in conds))
    return f"select {'g, ' if grouped else ''}{', '.join(sels)} {frm}" + (" group by g" if grouped else "")


def run(binary, args):
    p = subprocess.run([str(binary)] + args, capture_output=True)
    out = json.loads(p.stdout.decode("utf-8"))
    assert "error" not in out, f"{binary.name} {args[:12]}: {out.get('error')}"
    return out


def sql_rows(res):
    kinds = [c["kind"] for c in res["columns"]]
    out = []
    for r in res["rows"]:
        row = []
        for v, k in zip(r, kinds):
            if v is None:
                row.append(None)
            elif k == "FLOAT":
                import struct
                row.append(struct.unpack(">d", bytes.fromhex(v[2:]))[0])
            else:
                row.append(int(v) if k in ("INT", "DECIMAL") else v)
        out.append(tuple(row))
    return out


def builder_rows(res, grouped):
    out = []
    for r in res["rows"]:
        row = [None if v is None else (float(v) if isinstance(v, str) else v) for v in r]
        out.append(tuple(row if grouped else row[1:]))
    return out


def same(got, want, what):
    key = lambda r: tuple((0, 0) if v is None else (1, v) for v in r[:1])
    got = sorted(got, key=key)
    want = sorted([tuple(None if v is None else (float(v) if isinstance(v, float) else int(v)) for v in r) for r in want], key=key)
    assert len(got) == len(want), (what, len(got), len(want), got[:3], want[:3])
    for a, b in zip(got, want):
        for x, y in zip(a, b):
            if isinstance(x, float) or isinstance(y, float):
                assert x is not None and y is not None and abs(x - y) <= 1e-9 * max(1.0, abs(y)), (what, a, b)
            else:
                assert x == y, (what, a, b)


def fast(res, what):
    assert any("join path: fused" in n for n in res["notes"]), (what, res["notes"])


@pytest.mark.parametrize("seed,dup,dim_edits", [(0, False, False), (1, True, False), (2, False, True), (3, True, True), (4, False, False), (5, True, True), (6, False, True), (7, True, False)])
def test_every_entry_point_equals_duckdb(probes, con, tmp_path, seed, dup, dim_edits):
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
    for name, how, where, aggs in SHAPES:
        for grouped in (True, False):
            sql = sql_text(how, where, aggs, grouped)
            want = con.execute(sql).fetchall()
            what = f"{name} {'grouped' if grouped else 'global'} (seed {seed}, dup {dup}, dim edits {dim_edits})"
            # (a) SQL over tables, (b) SQL over edited Frames
            a = run(sqlp, ["--mode", "table", sql, f"fact={t}/fact.parquet", f"dim={t}/dim.parquet"])
            same(sql_rows(a), want, "(a) sql table: " + what)
            fast(a, "(a) " + what)
            b = run(sqlp, ["--mode", "frame", "--edits", str(t / "f.edits")] + (["--redits", str(t / "d.edits")] if dim_edits else []) +
                    [sql, f"fact={t}/fact_base.parquet", f"dim={t}/dim_base.parquet"])
            same(sql_rows(b), want, "(b) sql frame: " + what)
            fast(b, "(b) " + what)
            # (c) the builder over tables, (d) over edited Frames
            common = ["--how", how, "--on", "k=dk", "--aggs", aggs] + (["--where", where] if where else []) + (["--group", f"g:{GROUPS}"] if grouped else [])
            c = run(bp, ["--mode", "table"] + common + [f"left={t}/fact.parquet", f"right={t}/dim.parquet"])
            same(builder_rows(c, grouped), want, "(c) builder table: " + what)
            fast(c, "(c) " + what)
            d = run(bp, ["--mode", "frame", "--edits", str(t / "f.edits")] + (["--right-mode", "frame", "--redits", str(t / "d.edits")] if dim_edits else []) +
                    common + [f"left={t}/fact_base.parquet", f"right={t}/dim_base.parquet"])
            same(builder_rows(d, grouped), want, "(d) builder frame: " + what)
            fast(d, "(d) " + what)
