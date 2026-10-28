"""F-150: SQL aggregate queries over one table in the Mojo API, equal to DuckDB.

The Mojo side is mojo/tools/sql_probe.mojo run as a subprocess: it parses, binds, plans and runs one SQL string and prints the
result as typed JSON (values exact: decimals as text, floats as their bits). Every query runs through

- a table read whole (`table`),
- a Frame after 300 random edits (`frame`: sets, nulls, new words, appended and removed rows, one merge; the oracle applies the
  same edits to a DuckDB table),
- a Parquet file streamed as 13 row groups (`scan`),

on generated data with nulls, mixed-scale decimals, dates, a boolean and text with case variants, at 8, 10,000 and about
1,000,000 groups, and equals DuckDB's answer: ORDER BY without ties compared in order, the rest as sets. Each test also asserts
the path that ran (the planner's notes: dense or hashed groups and the reason, the kind of source, the chunks read), not only
the answer. Text order is the owner's (alphabetical ignoring ASCII case, bytes as the tie-break), so the oracle orders and
compares text under (lower(s), s).
"""
import datetime
import fcntl
import json
import os
import random
import re
import struct
import subprocess
from decimal import Decimal
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import pytest

from mx_oracle import assert_equal, duck_kind, sort_key
from engine_sources import engine_sources

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
WORDS = ["west", "West", "WEST", "east", "East", "North", "NORTH", "south", "Apple", "apple", "banana", "Zebra", "zebra", "mango", "Mango"]
N_EDITS = 300
ROW_GROUPS = 13


# ── the probe ────────────────────────────────────────────────────────────────────────────────────────

def _newest(paths):
    return max(p.stat().st_mtime for p in paths)


@pytest.fixture(scope="session")
def probe():
    """Builds mojo/tools/sql_probe.mojo (once, even under xdist: the workers take a lock) and returns its path."""
    BUILD.mkdir(exist_ok=True)
    out = BUILD / "sql_probe"
    sources = engine_sources(ROOT / "mojo/tools/sql_probe.mojo")
    with open(BUILD / ".sql_probe.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not out.exists() or out.stat().st_mtime < _newest(sources):
            subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/sql_probe.mojo"), str(out)],
                           check=True, capture_output=True, text=True)
    return out


def run_probe(probe, sql, files, mode="table", group=None, edits=None, dense_limit=None, repeat=None, expect_error=False):
    cmd = [str(probe), "--mode", mode]
    if group:
        cmd += ["--group", group]
    if edits:
        cmd += ["--edits", str(edits)]
    if dense_limit:
        cmd += ["--dense-limit", str(dense_limit)]
    if repeat:
        cmd += ["--repeat", str(repeat)]
    cmd += [sql] + [f"{k}={v}" for k, v in files.items()]
    p = subprocess.run(cmd, capture_output=True)
    text = p.stdout.decode("utf-8")
    out = json.loads(text) if text.strip() else {"error": "no output: " + p.stderr.decode("utf-8", "replace")[-400:]}
    if "error" in out and not expect_error:
        raise AssertionError(f"the probe failed on `{sql}`: {out['error']}")
    return out


# ── typed values ─────────────────────────────────────────────────────────────────────────────────────

def got_rows(res):
    kinds = [c["kind"] for c in res["columns"]]
    rows = []
    for r in res["rows"]:
        out = []
        for v, k in zip(r, kinds):
            if v is None:
                out.append(None)
            elif k == "DECIMAL":
                out.append(Decimal(v))
            elif k == "DATE":
                out.append(datetime.date.fromisoformat(v))
            elif k == "FLOAT":
                out.append(struct.unpack("<d", struct.pack("<Q", int(v, 16)))[0])
            else:
                out.append(v)
        rows.append(tuple(out))
    return rows


# ── generated data ───────────────────────────────────────────────────────────────────────────────────

def make_source(con, n, groups, seed):
    """The table `src` in DuckDB: rid (the row number), id, g (about `groups` values, 2% NULL), k (a few values, 10% NULL), x (BIGINT,
    ties, negatives, 8% NULL), d DECIMAL(18,2), e DECIMAL(9,4), s (words with case variants), dt DATE, b BOOLEAN."""
    con.execute("set threads=1")
    con.execute(f"select setseed({(seed % 100) / 100})")
    words = "[" + ", ".join(f"'{w}'" for w in WORDS) + "]"
    con.execute(f"""
        create table src as select
          i::BIGINT as rid,
          i::BIGINT as id,
          case when random() < 0.02 then null else ((i * 2654435761) % {groups})::BIGINT end as g,
          case when random() < 0.10 then null else (random() * 5)::int::BIGINT end as k,
          case when random() < 0.08 then null else ((random() * 2000)::int - 1000)::BIGINT end as x,
          case when random() < 0.10 then null else (((random() * 200000)::int - 100000) * 0.01)::decimal(18,2) end as d,
          case when random() < 0.10 then null else (((random() * 2000000)::int - 1000000) * 0.0001)::decimal(9,4) end as e,
          case when random() < 0.10 then null else {words}[1 + (random() * {len(WORDS) - 1})::int] end as s,
          case when random() < 0.05 then null else date '2018-01-01' + (random() * 1800)::int end as dt,
          case when random() < 0.05 then null else random() < 0.4 end as b
        from range({n}) r(i)""")


COLS = [("id", "int"), ("g", "int"), ("k", "int"), ("x", "int"), ("d", "dec2"), ("e", "dec4"), ("s", "text"), ("dt", "date"), ("b", "bool")]


def make_edits(con, n_rows, groups, seed):
    """A deterministic edit script (the probe's format) and the same edits applied to the DuckDB table `src`."""
    rng = random.Random(seed)
    lines = []
    alive = set(range(n_rows))
    next_rid = n_rows
    names = [c for c, _ in COLS]
    pool = WORDS + ["zzz-new", "aaa-new", "Mid", "q"]

    def value(name, kind):
        if kind == "int":
            if name == "g":
                return str(rng.randint(-5, groups + 5))
            if name == "id":
                return str(rng.randint(0, 10 * n_rows))
            return str(rng.randint(-1200, 1200))
        if kind == "dec2":
            return f"{rng.randint(-100000, 100000) / 100:.2f}"
        if kind == "dec4":
            return f"{rng.randint(-1000000, 1000000) / 10000:.4f}"
        if kind == "text":
            return rng.choice(pool)
        if kind == "date":
            return (datetime.date(2017, 6, 1) + datetime.timedelta(days=rng.randint(0, 2200))).isoformat()
        return rng.choice(["true", "false"])

    def lit(kind, v):
        if v == "":
            return "NULL"
        return v if kind in ("int", "dec2", "dec4", "bool") else "'" + v.replace("'", "''") + "'"

    for step in range(N_EDITS):
        what = rng.randint(0, 12)
        if what <= 6:
            row = rng.choice(sorted(alive)) if len(alive) < 5000 else rng.randrange(n_rows)
            while row not in alive:
                row = rng.randrange(n_rows)
            name, kind = rng.choice(COLS[1:])
            v = "" if what == 6 else value(name, kind)
            lines.append(f"set\t{name}\t{row}\t{v}")
            con.execute(f"update src set {name} = {lit(kind, v)} where rid = {row}")
        elif what in (7, 8):
            vals = ["" if rng.random() < 0.15 else value(name, kind) for name, kind in COLS]
            lines.append("append\t" + "\t".join(vals))
            con.execute(f"insert into src (rid, {', '.join(names)}) values ({next_rid}, {', '.join(lit(k, v) for (n, k), v in zip(COLS, vals))})")
            alive.add(next_rid)
            next_rid += 1
        elif what == 9:
            row = rng.randrange(n_rows)
            if row in alive:
                lines.append(f"remove\t{row}")
                con.execute(f"delete from src where rid = {row}")
                alive.discard(row)
        else:
            row = rng.randrange(n_rows)
            while row not in alive:
                row = rng.randrange(n_rows)
            name, kind = rng.choice(COLS[1:])
            v = value(name, kind)
            lines.append(f"set\t{name}\t{row}\t{v}")
            con.execute(f"update src set {name} = {lit(kind, v)} where rid = {row}")
    lines.append("merge\ts\tWest\tzzz-new,Mid")                       # two words folded into one
    con.execute("update src set s = 'West' where s in ('zzz-new', 'Mid')")
    return "\n".join(lines) + "\n"


class Dataset:
    def __init__(self, name, n, groups, directory, seed):
        self.name, self.n, self.groups = name, n, groups
        self.path = directory / f"{name}.parquet"
        self.edits_path = directory / f"{name}.edits"
        group_rows = -(-n // ROW_GROUPS)
        con = duckdb.connect()
        make_source(con, n, groups, seed)
        table = con.execute("select * exclude (rid) from src order by rid").to_arrow_table()
        pq.write_table(table, self.path, row_group_size=group_rows)
        self.nostats = directory / f"{name}.nostats.parquet"
        pq.write_table(table, self.nostats, row_group_size=group_rows, write_statistics=False)
        self.table = duckdb.connect()
        self.table.execute(f"create table t as select * from read_parquet('{self.path}')")
        self.table.execute("set threads=4")
        self.edited = duckdb.connect()
        self.edited.execute(f"create table src as select row_number() over () - 1 as rid, * from read_parquet('{self.path}')")
        self.edits_path.write_text(make_edits(self.edited, n, groups, seed + 1))
        self.edited.execute("create view t as select * exclude (rid) from src")
        self.edited.execute("set threads=4")

    def oracle(self, mode):
        return self.edited if mode in ("frame", "framescan") else self.table


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("sqldata")
    return {
        "g8": Dataset("g8", 20_000, 8, d, 1),
        "g10k": Dataset("g10k", 200_000, 10_000, d, 2),
        "g1m": Dataset("g1m", 1_200_000, 1_000_000, d, 3),
    }


# ── what the planner should have chosen ──────────────────────────────────────────────────────────────

def expected_group(keys, ds, mode, kinds, con, rows):
    """The strategy RulePlanner's documented rule picks: dense when every key is a column with a known range (a TEXT / BOOL column
    always has one; an integer column has the table's, the frame's or the row groups' statistics) and the domain is small or no more
    than 4x the rows. An expression key is hashed here (the planner bounds some of them: the tests that use one say what they expect)."""
    if not keys:
        return "none"
    domain = 1
    for key in keys:
        if key == "expr":
            return "hashed"
        if key == "dense":                           # an expression the planner bounds from the columns' ranges (x % 10, x // 3)
            continue
        kind = kinds[key]
        if kind == "text":
            n_words = con.execute(f"select count(distinct {key}) from t").fetchone()[0]
            domain *= n_words + 1
        elif kind == "bool":
            domain *= 3
        else:
            span = con.execute(f"select max({key}) - min({key}) from t").fetchone()[0]
            domain *= (span + 1) + 1
    return "dense" if domain <= 4_194_304 and (domain <= 65_536 or domain <= 4 * rows) else "hashed"


# ── the queries ──────────────────────────────────────────────────────────────────────────────────────

KINDS = {"g": "int", "k": "int", "x": "int", "dt": "date", "s": "text", "b": "bool", "id": "int"}


class Q:
    def __init__(self, sql, keys=(), ordered=False, duck=None, fix=None, skip=(), tag=None):
        self.sql, self.keys, self.ordered, self.duck, self.fix, self.skip = sql, list(keys), ordered, duck, fix, set(skip)
        self.tag = tag or sql


def text_cmp(col, op, lit):
    """`col op 'lit'` under the owner's text order: ignoring ASCII case, bytes as the tie-break."""
    strict = ">" if op in (">", ">=") else "<"
    return f"(lower({col}) {strict} lower('{lit}') OR (lower({col}) = lower('{lit}') AND {col} {op} '{lit}'))"


def median_dec(rows, positions, extra_scale=1):
    """DuckDB's median of a DECIMAL rounds to the column's scale; ours is exact at one more place, so the oracle asks for
    median(d * 10) (exact at the column's scale) and the result is read back divided by ten."""
    out = []
    for r in rows:
        r = list(r)
        for p in positions:
            if r[p] is not None:
                r[p] = Decimal(r[p]) / 10
        out.append(tuple(r))
    return out


QUERIES = [
    Q("SELECT count(*), count(x), sum(x), min(x), max(x), avg(x) FROM t"),
    Q("SELECT sum(d), min(d), max(d), avg(d), sum(e), avg(e), sum(d * e), sum(d + e), max(e - d) FROM t"),
    Q("SELECT min(dt), max(dt), min(b), max(b), count(DISTINCT s), count(DISTINCT dt), count(DISTINCT d) FROM t"),
    Q("SELECT min(s), max(s) FROM t", duck="SELECT arg_min(s, (lower(s), s)), arg_max(s, (lower(s), s)) FROM t"),
    Q("SELECT count(*), sum(x), min(x), avg(x), max(s) FROM t WHERE x > 5000", duck="SELECT count(*), sum(x), min(x), avg(x), arg_max(s, (lower(s), s)) FROM t WHERE x > 5000"),
    Q("SELECT median(x), quantile_disc(x, 0.9), quantile_disc(x, 0), quantile_disc(d, 0.25), count(DISTINCT x), median(d), quantile_disc(s, 0.5) FROM t",
      duck="SELECT median(x), quantile_disc(x, 0.9), quantile_disc(x, 0), quantile_disc(d, 0.25), count(DISTINCT x), median(d * 10), "
           "(list(s ORDER BY lower(s), s) FILTER (WHERE s IS NOT NULL))[greatest(ceil(0.5 * count(s))::int, 1)] FROM t",
      fix=lambda rows: [r[:5] + (Decimal(r[5]) / 10 if r[5] is not None else None,) + r[6:] for r in rows]),
    Q("SELECT g, count(*), sum(x), avg(d) FROM t GROUP BY g", keys=["g"], tag="by-g", skip=("g1m",)),
    Q("SELECT g, count(*) AS n, sum(d) AS sd FROM t GROUP BY g ORDER BY n DESC, g LIMIT 20", keys=["g"], ordered=True),
    Q("SELECT g, k, sum(x), min(e), max(e), count(DISTINCT s) FROM t GROUP BY g, k HAVING count(*) > 2", keys=["g", "k"], skip=("g1m",)),
    Q("SELECT k, s, count(*), sum(d) FROM t WHERE x BETWEEN -500 AND 500 AND d IS NOT NULL GROUP BY k, s", keys=["k", "s"]),
    Q("SELECT g, sum(x) FROM t GROUP BY g HAVING sum(x) > 100 AND count(*) > 2 AND g IS NOT NULL", keys=["g"]),
    Q("SELECT s, count(*), avg(x), median(x), min(dt) FROM t GROUP BY s ORDER BY s",
      keys=["s"], ordered=True, duck="SELECT s, count(*), avg(x), median(x), min(dt) FROM t GROUP BY s ORDER BY lower(s) NULLS LAST, s NULLS LAST"),
    Q("SELECT dt, count(*) FROM t GROUP BY dt ORDER BY dt DESC LIMIT 5", keys=["dt"], ordered=True),
    Q("SELECT b, count(*), sum(x), min(s) FROM t GROUP BY b", keys=["b"], duck="SELECT b, count(*), sum(x), arg_min(s, (lower(s), s)) FROM t GROUP BY b"),
    Q("SELECT x % 10, x // 3, count(*) FROM t GROUP BY x % 10, x // 3", keys=["dense"], tag="expr-keys"),
    Q("SELECT g, sum(x) / count(*), max(d) - min(d), sum(d) / sum(x), sum(x) + 1.5, count(*) * 2 FROM t GROUP BY g HAVING count(*) > 1",
      keys=["g"], skip=("g1m",)),
    Q("SELECT count(*), sum(x) FROM t WHERE s >= 'm' AND s < 'x'",
      duck=f"SELECT count(*), sum(x) FROM t WHERE {text_cmp('s', '>=', 'm')} AND {text_cmp('s', '<', 'x')}"),
    Q("SELECT s, count(*) FROM t WHERE s IN ('west', 'East', 'NORTH', 'nope') OR s = 'Zebra' GROUP BY s", keys=["s"]),
    Q("SELECT s, count(*) FROM t WHERE s NOT IN ('west', 'East') GROUP BY s", keys=["s"]),
    Q("SELECT k, count(*) FROM t WHERE dt >= '2020-01-01' AND dt < DATE '2021-01-01' AND NOT b GROUP BY k", keys=["k"]),
    Q("SELECT k, count(*), sum(x) FROM t WHERE x > 0 OR d < 0 GROUP BY k", keys=["k"]),
    Q("SELECT k, count(*) FROM t WHERE NOT (x > 0 AND d > 0) AND (b OR x IS NULL) GROUP BY k", keys=["k"]),
    Q("SELECT count(*) FROM t WHERE x IN (1, 2, NULL) OR k NOT IN (1, 2, NULL)", tag="in-null"),
    Q("SELECT count(*) FROM t WHERE (x > 0) IS NULL OR (b IS NOT NULL AND d IS NULL)", tag="is-null"),
    Q("SELECT k, sum(x // (k - 2)), count(x % (k - 2)), sum(x % 7), min(x // 7) FROM t GROUP BY k", keys=["k"]),
    Q("SELECT k, avg(x), avg(d), avg(e), avg(d * e) FROM t GROUP BY k", keys=["k"]),
    Q("SELECT k, sum(x) AS sx FROM t GROUP BY k ORDER BY count(*) DESC, k NULLS FIRST LIMIT 4 OFFSET 1", keys=["k"], ordered=True),
    Q("SELECT g AS grp, count(*) FROM t GROUP BY 1 ORDER BY 2 DESC, grp DESC NULLS FIRST LIMIT 7", keys=["g"], ordered=True),
    Q("SELECT s, k, count(*), sum(d) FROM t GROUP BY s, k ORDER BY s DESC, k LIMIT 10", keys=["s", "k"], ordered=True,
      duck="SELECT s, k, count(*), sum(d) FROM t GROUP BY s, k ORDER BY lower(s) DESC NULLS LAST, s DESC NULLS LAST, k LIMIT 10"),
    Q("SELECT g, quantile_disc(x, 0.5), median(d), count(DISTINCT s), min(s) FROM t WHERE g < 50 GROUP BY g", keys=["g"], tag="holistic-by-g",
      duck="SELECT g, quantile_disc(x, 0.5), median(d * 10), count(DISTINCT s), arg_min(s, (lower(s), s)) FROM t WHERE g < 50 GROUP BY g",
      fix=lambda rows: [r[:2] + (Decimal(r[2]) / 10 if r[2] is not None else None,) + r[3:] for r in rows]),
    Q("SELECT count(*), sum(x) FROM t WHERE x > 5000 GROUP BY k", keys=["k"], tag="empty-groups"),
    Q("SELECT k, median(s), quantile_disc(s, 0.3), quantile_disc(dt, 0.5), quantile_disc(dt, 0.9), quantile_disc(b, 0.5) FROM t GROUP BY k", keys=["k"],
      tag="median-text-date",
      duck="SELECT k, (list(s ORDER BY lower(s), s) FILTER (WHERE s IS NOT NULL))[greatest(ceil(0.5 * count(s))::int, 1)], "
           "(list(s ORDER BY lower(s), s) FILTER (WHERE s IS NOT NULL))[greatest(ceil(0.3 * count(s))::int, 1)], "
           "quantile_disc(dt, 0.5), quantile_disc(dt, 0.9), quantile_disc(b, 0.5) FROM t GROUP BY k"),
    Q("SELECT s, median(s), quantile_disc(dt, 0.5) FROM t WHERE x > 0 GROUP BY s", keys=["s"], tag="median-text-by-text",
      duck="SELECT s, (list(s ORDER BY lower(s), s) FILTER (WHERE s IS NOT NULL))[greatest(ceil(0.5 * count(s))::int, 1)], quantile_disc(dt, 0.5) "
           "FROM t WHERE x > 0 GROUP BY s"),
    Q("SELECT count(*), sum(x) FROM t WHERE d > x AND e < d AND x * d > e", tag="mixed-scales"),
    Q("SELECT k, count(*) FROM t WHERE d >= -12.5 AND e < 0.00005 AND x = 1.0 GROUP BY k", keys=["k"], tag="decimal-literals"),
    Q("SELECT k, max(dt) - min(dt), min(dt) + 7, count(*) FROM t WHERE dt + 30 > '2020-06-01' GROUP BY k", keys=["k"], tag="date-arithmetic"),
    Q("SELECT k, sum(x) FROM t GROUP BY k ORDER BY -sum(x), k LIMIT 5", keys=["k"], ordered=True, tag="order-by-expression"),
    Q("SELECT 1, 2.5, count(*), sum(1.5) FROM t WHERE TRUE", tag="constants"),
    Q("SELECT count(*), sum(x) FROM t WHERE FALSE OR x > NULL OR NULL", tag="constant-conditions"),
    Q("SELECT k, count(*) FROM t GROUP BY k ORDER BY k LIMIT 0", keys=["k"], ordered=True, tag="limit-0"),
    Q("SELECT k, count(*) FROM t GROUP BY k ORDER BY k OFFSET 100", keys=["k"], ordered=True, tag="offset-past-the-end"),
    Q("SELECT k, sum(b), min(b), max(b), count(b) FROM t GROUP BY k", keys=["k"], tag="bool-aggregates"),
]


def query_id(q):
    return re.sub(r"[^A-Za-z0-9]+", "-", q.tag)[:48]


MODES = ["table", "frame", "scan"]


def run_case(probe, data, ds_name, mode, q, group=None, **kw):
    ds = data[ds_name]
    files = {"t": ds.path}
    res = run_probe(probe, q.sql, files, mode=mode, group=group, edits=ds.edits_path if mode in ("frame", "framescan") else None, **kw)
    assert_source_path(res, mode)
    con = ds.oracle(mode)
    cur = con.execute(q.duck or q.sql)
    want = cur.fetchall()
    if q.fix:
        want = q.fix(want)
    got = got_rows(res)
    kinds = [(c["kind"], c["scale"]) for c in res["columns"]]
    if not q.fix:
        for i, desc in enumerate(cur.description):
            dk = duck_kind(desc[1])
            assert kinds[i][0] == dk[0] and (dk[0] != "DECIMAL" or kinds[i][1] == dk[1]), \
                f"column {i} of `{q.sql}`: got {kinds[i]}, DuckDB has {dk}"
    assert_equal(got, want, q.ordered, f"{ds_name}/{mode}: {q.sql}")
    return res, ds


@pytest.mark.parametrize("ds_name", ["g8", "g10k", "g1m"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("q", QUERIES, ids=query_id)
def test_equals_duckdb_and_path(probe, data, ds_name, mode, q):
    if ds_name in q.skip:
        pytest.skip("a result too big to compare row by row")
    res, ds = run_case(probe, data, ds_name, mode, q)
    notes = res["notes"]
    # the path that ran
    want = expected_group(q.keys, ds, mode, KINDS, ds.oracle(mode), ds.n)
    group_note = next(n for n in notes if n.startswith("group:"))
    assert group_note.startswith("group: " + want), f"{q.sql} on {ds_name}/{mode}: the planner said `{group_note}`, expected {want}"
    assert_source_path(res, mode)


def assert_source_path(res, mode):
    """The notes say which source ran how: a table in one chunk, a frame as its shared rows (changed ones masked) plus the changed rows,
    a scan as every one of its 13 row groups; and the planner's strategy and the kernel's size are always there."""
    notes = res["notes"]
    source_note = next(n for n in notes if n.startswith("source:"))
    ran_note = next(n for n in notes if n.startswith("ran:"))
    if mode == "table":
        assert "(table)" in source_note and ran_note.startswith("ran: table t: 1 chunk of"), notes
    elif mode == "frame":
        assert "(frame at edit" in source_note and ran_note.startswith("ran: frame t:") and "changed or appended rows" in ran_note, notes
    elif mode == "framescan":
        assert "Parquet scan, 13 row groups" in source_note and ran_note.startswith("ran: scan t: 13 of 13 row groups read") and \
               "changed or appended rows" in ran_note, notes
    else:
        assert "Parquet scan, 13 row groups" in source_note and ran_note.startswith("ran: scan t: 13 of 13 row groups read"), notes
    assert any(n.startswith("group:") for n in notes) and any(n.startswith("kernel:") for n in notes), notes


# ── forced paths, empty and edge cases ───────────────────────────────────────────────────────────────

FORCED = [
    Q("SELECT g, k, count(*), sum(x), min(d), max(dt), avg(e) FROM t GROUP BY g, k", keys=["g", "k"]),
    Q("SELECT s, count(*), sum(d), count(DISTINCT x), median(x) FROM t GROUP BY s", keys=["s"]),
    Q("SELECT b, k, count(*), quantile_disc(d, 0.4), min(s) FROM t GROUP BY b, k", keys=["b", "k"],
      duck="SELECT b, k, count(*), quantile_disc(d, 0.4), arg_min(s, (lower(s), s)) FROM t GROUP BY b, k"),
]


@pytest.mark.parametrize("mode", ["table", "frame", "scan", "framescan"])
@pytest.mark.parametrize("strategy", ["dense", "hashed"])
@pytest.mark.parametrize("q", FORCED, ids=query_id)
def test_forced_group_strategy_gives_the_same_answer(probe, data, mode, strategy, q):
    """PlannerOptions force either strategy; the answer is DuckDB's whichever runs, and the notes say which did."""
    res, ds = run_case(probe, data, "g8", mode, q, group=strategy)
    group_note = next(n for n in res["notes"] if n.startswith("group:"))
    assert group_note.startswith(f"group: {strategy}") and group_note.endswith("forced by the options"), group_note
    if strategy == "hashed" and any(a in q.sql for a in ("DISTINCT", "median", "quantile_disc", "min(s)")):
        assert any(n.startswith("ran: holistic second pass") for n in res["notes"]), res["notes"]


def test_dense_limit_sends_a_wide_domain_to_hashed_groups(probe, data):
    q = QUERIES[6]
    res, _ = run_case(probe, data, "g10k", "table", q, dense_limit=500)
    group_note = next(n for n in res["notes"] if n.startswith("group:"))
    assert group_note == "group: hashed on 2 key words: the domain of 10,001 slots passes the dense limit of 500", group_note


def test_forcing_dense_beyond_the_limit_raises(probe, data):
    res = run_probe(probe, "SELECT g, count(*) FROM t GROUP BY g", {"t": data["g10k"].path}, group="dense", dense_limit=500, expect_error=True)
    assert res["error"] == "cannot force dense groups: the domain of 10,001 slots passes the limit of 500"


def test_sum_past_int64_raises_in_every_source(probe, data):
    for mode in ["table", "scan"]:
        res = run_probe(probe, "SELECT sum(id * id * id) FROM t", {"t": data["g1m"].path}, mode=mode, expect_error=True)
        assert res["error"] == "a sum or an expression passed the int64 range (about 9.2e18)", (mode, res)
        res = run_probe(probe, "SELECT k, sum(id * id * id) FROM t GROUP BY k", {"t": data["g1m"].path}, mode=mode, group="hashed", expect_error=True)
        assert res["error"] == "a sum or an expression passed the int64 range (about 9.2e18)", (mode, res)


def test_empty_inputs(probe, data):
    for mode in MODES:
        q = Q("SELECT count(*), count(x), sum(x), min(x), max(d), avg(e), median(x), count(DISTINCT s), min(s) FROM t WHERE id < 0",
              duck="SELECT count(*), count(x), sum(x), min(x), max(d), avg(e), median(x), count(DISTINCT s), arg_min(s, (lower(s), s)) FROM t WHERE id < 0")
        res, _ = run_case(probe, data, "g8", mode, q)
        assert got_rows(res) == [(0, 0, None, None, None, None, None, 0, None)]
        res, _ = run_case(probe, data, "g8", mode, Q("SELECT k, count(*) FROM t WHERE id < 0 GROUP BY k", keys=["k"]))
        assert res["rows"] == []


def test_null_is_a_group_of_its_own_and_sorts_last(probe, data):
    q = Q("SELECT k, count(*) AS n FROM t GROUP BY k ORDER BY k", keys=["k"], ordered=True)
    for mode in MODES:
        res, _ = run_case(probe, data, "g8", mode, q)
        rows = res["rows"]
        assert rows[-1][0] is None and rows[-1][1] > 0 and all(r[0] is not None for r in rows[:-1]), rows
        assert [r[0] for r in rows[:-1]] == sorted(r[0] for r in rows[:-1])


def test_text_order_ignores_case_and_bytes_break_ties(probe, data):
    res = run_probe(probe, "SELECT s, count(*) FROM t GROUP BY s ORDER BY s", {"t": data["g8"].path})
    words = [r[0] for r in res["rows"] if r[0] is not None]
    assert words == sorted(WORDS, key=lambda w: (w.lower(), w)), words
    res = run_probe(probe, "SELECT s, count(*) FROM t GROUP BY s ORDER BY s DESC", {"t": data["g8"].path})
    assert [r[0] for r in res["rows"] if r[0] is not None] == sorted(WORDS, key=lambda w: (w.lower(), w), reverse=True)
    res = run_probe(probe, "SELECT min(s), max(s) FROM t", {"t": data["g8"].path})
    assert res["rows"] == [["Apple", "zebra"]], res["rows"]


def test_the_select_list_keeps_names_aliases_and_kinds(probe, data):
    res = run_probe(probe, "SELECT k AS kk, count(*), sum(d) AS total, avg(x), min(dt), max(b) FROM t GROUP BY k LIMIT 1", {"t": data["g8"].path})
    assert [c["name"] for c in res["columns"]] == ["kk", "count_star()", "total", "avg(x)", "min(dt)", "max(b)"]
    assert [(c["kind"], c["scale"]) for c in res["columns"]] == [("INT", 0), ("INT", 0), ("DECIMAL", 2), ("FLOAT", 0), ("DATE", 0), ("BOOL", 0)]


def test_explain_shows_the_logical_and_the_physical_plan(probe, data):
    sql = "EXPLAIN SELECT s, k, sum(d) AS total FROM t WHERE x > 5 GROUP BY s, k HAVING count(*) > 2 ORDER BY total DESC LIMIT 3"
    res = run_probe(probe, sql, {"t": data["g8"].path})
    assert res["columns"] == [{"name": "plan", "kind": "TEXT", "scale": 0}]
    lines = [r[0] for r in res["rows"]]
    assert lines == [
        "Logical plan",
        "  Limit 3",
        "    Sort total DESC",
        "      Project s, k, sum(d)",
        "        Filter (HAVING) count(*) > 2",
        "          Aggregate keys=[s, k] aggs=[sum(d), count_star()]",
        "            Filter (WHERE) x > 5",
        "              Scan t",
        "Physical plan",
        "  source: t (table), 20,000 rows",
        "  group: dense, 2 keys s 16 (with a NULL slot) x k 7 (with a NULL slot) = 112 slots: the domain of 112 slots is small",
        "  columns read: k, x, d, s (4 of 9)",
        "  kernel: 36 instructions a row (2 slots)",
        "  device: CPU (SQL runs on the CPU)",
    ], lines
    assert res["explain"] == "\n".join(lines) + "\n"
    assert not any(n.startswith("ran:") for n in res["notes"]), "EXPLAIN does not run the query"


def test_explain_of_a_hashed_plan_says_why(probe, data):
    lines = lambda sql, mode="table", name="g8": [r[0] for r in run_probe(probe, sql, {"t": getattr(data[name], "path" if mode != "nostats" else "nostats")},
                                                                         mode="scan" if mode == "nostats" else mode)["rows"]]
    assert "  group: hashed on 2 key words: the domain of 2,000,002 slots is sparse for 20,000 rows" in lines("EXPLAIN SELECT x * x, count(*) FROM t GROUP BY x * x")
    assert "  group: dense, 1 key x % 7 14 (with a NULL slot) = 14 slots: the domain of 14 slots is small" in lines("EXPLAIN SELECT x % 7, count(*) FROM t GROUP BY x % 7")
    # a file written without statistics has no ranges to plan from
    assert "  group: hashed on 2 key words: the range of `g` is unknown (the source has no statistics on it)" in lines("EXPLAIN SELECT g, count(*) FROM t GROUP BY g", "nostats")
    assert "  group: dense, 1 key g 9 (with a NULL slot) = 9 slots: the domain of 9 slots is small" in lines("EXPLAIN SELECT g, count(*) FROM t GROUP BY g", "scan")


ERRORS = [
    ("SELECT sum(revenu) FROM t", "SQL error at line 1, column 12: column `revenu` not found in table `t`"),
    ("SELECT sum(xx) FROM t", "SQL error at line 1, column 12: column `xx` not found in table `t` (did you mean `x`?)"),
    ("SELECT sum(x) FROM tt", "SQL error at line 1, column 20: table `tt` not found (did you mean `t`?)"),
    ("SELECT t.k, count(*) FROM u GROUP BY t.k", "SQL error at line 1, column 27: table `u` not found (the catalog has t)"),
    ("SELCT sum(x) FROM t", "SQL error at line 1, column 1: expected SELECT or EXPLAIN, found `SELCT` (did you mean SELECT?)"),
    ("SELECT sum(x) FORM t", "SQL error at line 1, column 15: expected FROM, found `FORM` (did you mean FROM?)"),
    ("SELECT count(*) FROM t WHERE", "SQL error at line 1, column 29: expected an expression, found the end of the statement"),
    ("SELECT count(*) FROM t WHERE s = 'oops", "SQL error at line 1, column 34: this string is never closed (expected a closing ')"),
    ("SELECT count(*) FROM t; SELECT 1", "SQL error at line 1, column 25: expected the end of the statement, found `SELECT`"),
    ("SELECT sum(x) FROM t RIGHT JOIN u ON t.id = u.id", "SQL error at line 1, column 22: `RIGHT JOIN` is not supported yet"),
    ("SELECT DISTINCT k, count(*) FROM t GROUP BY k", "SQL error at line 1, column 17: `SELECT DISTINCT` together with aggregates, GROUP BY or HAVING is not supported yet: use GROUP BY alone (it returns the same distinct rows), or DISTINCT without aggregates"),
    ("SELECT DISTINCT ON (k) k FROM t", "SQL error at line 1, column 8: `SELECT DISTINCT ON` is not supported yet"),
    ("SELECT z.* FROM t", "SQL error at line 1, column 8: unknown table or alias `z` in `z.*`"),
    ("SELECT sum(x) FROM t WHERE s GLOB 'a*'", "SQL error at line 1, column 30: `GLOB` is not supported yet"),
    ("SELECT sum(x) FROM t WHERE s SIMILAR TO 'a'", "SQL error at line 1, column 30: `SIMILAR` is not supported yet"),
    ("SELECT count(*) FROM t WHERE s IN (SELECT s, x FROM t)", "SQL error at line 1, column 44: a subquery with more than one column is not supported yet"),
    ("SELECT sum(x) OVER w FROM t", "SQL error at line 1, column 20: `OVER w` (a named window) is not supported yet (write the window out: OVER (PARTITION BY ... ORDER BY ...))"),
    ("SELECT 1e3x, count(*) FROM t", "SQL error at line 1, column 8: the number `1e3` runs into the letters after it"),
    ("SELECT skewness(x) FROM t", "SQL error at line 1, column 8: function `skewness` is not supported: SQL here has the aggregates sum, count, min, max, avg, median, quantile_disc, stddev, stddev_samp, stddev_pop, variance, var_samp, var_pop and the scalar functions coalesce, ifnull, nvl, nullif, lower, upper, trim, ltrim, rtrim, substr, substring, length, starts_with, ends_with, contains, strpos, instr, left, right, reverse, replace, concat, extract, date_part, year, month, day, quarter, dayofweek, isodow, dayofyear, date_trunc, datediff, date_diff, week, weekofyear, isoyear, quantile_bin"),
    ("SELECT summ(x) FROM t", "SQL error at line 1, column 8: function `summ` is not supported (did you mean `sum`?)"),
    ("SELECT sum(x), k FROM t", "SQL error at line 1, column 16: column `k` must appear in the GROUP BY clause or be used in an aggregate function"),
    ("SELECT k, count(*) FROM t GROUP BY k HAVING x > 1", "SQL error at line 1, column 45: column `x` must appear in the GROUP BY clause or be used in an aggregate function"),
    ("SELECT sum(sum(x)) FROM t", "SQL error at line 1, column 12: aggregate functions cannot be nested"),
    ("SELECT count(*) FROM t WHERE sum(x) > 1", "SQL error at line 1, column 37: aggregate functions are not allowed in WHERE: filter the groups with HAVING"),
    ("SELECT sum(s) FROM t", "SQL error at line 1, column 12: sum() needs numbers, found TEXT"),
    ("SELECT median(dt) FROM t", "SQL error at line 1, column 15: median() of a DATE is a TIMESTAMP in DuckDB (the middle of two dates can be half a day), which omxcore has no kind for yet (F-157): use quantile_disc(dt, 0.5)"),
    ("SELECT count(*) FROM t WHERE dt = '1965-02-29'", "SQL error at line 1, column 33: cannot read '1965-02-29' as a DATE: the day 29 does not exist: February 1965 has 28 days"),
    ("SELECT count(*) FROM t WHERE dt = DATE '2020-04-31'", "SQL error at line 1, column 35: `2020-04-31` is not a date: the day 31 does not exist: April 2020 has 30 days"),
    ("SELECT count(*) FROM t WHERE x = 'a'", "SQL error at line 1, column 32: cannot compare INT with TEXT (`=`)"),
    ("SELECT count(*) FROM t WHERE s < d", "SQL error at line 1, column 32: cannot compare TEXT with DECIMAL(2) (`<`)"),
    ("SELECT count(*) FROM t WHERE b + 1 > 2", "SQL error at line 1, column 32: cannot apply `+` to BOOL and INT"),
    ("SELECT count(*) FROM t WHERE dt > '2020-13-45'", "SQL error at line 1, column 33: cannot read '2020-13-45' as a DATE: there is no month 13"),
    ("SELECT quantile_disc(x, 1.5) FROM t", "SQL error at line 1, column 25: the quantile must be between 0 and 1, found 1.5"),
    ("SELECT count(*) FROM t LIMIT x", "SQL error at line 1, column 30: LIMIT needs a whole number, found `x` (expressions are not supported there)"),
    ("SELECT count(*) FROM t GROUP BY 3", "SQL error at line 1, column 33: GROUP BY 3 is out of range: the select list has 1 items"),
]


@pytest.mark.parametrize("sql,message", ERRORS, ids=[e[0][:50] for e in ERRORS])
def test_unsupported_and_misspelled_queries_say_what_and_where(probe, data, sql, message):
    for mode in ("table", "scan"):
        res = run_probe(probe, sql, {"t": data["g8"].path}, mode=mode, expect_error=True)
        assert res.get("error") == message, f"{mode}: {res}"


# ── random queries ───────────────────────────────────────────────────────────────────────────────────

class _Gen:
    """Random aggregate queries over the generated columns (conditions with AND / OR / NOT and NULLs, expressions with
    + - * // %, mixed decimal scales, several group keys, HAVING), seeded so a failure repeats."""

    INTS = ["x", "k", "g", "id"]
    DECS = ["d", "e"]

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def typed(self, depth=0):
        """(an expression, whether it is an integer): `//` is only generated between integers (on a DECIMAL it is a FLOAT in DuckDB)."""
        r = self.rng
        if depth > 2 or r.random() < 0.35:
            c = r.choice(self.INTS + self.DECS + [str(r.randint(-5, 9)), f"{r.randint(-300, 300) / 100:.2f}"])
            return c, not (c in self.DECS or "." in c)
        a, ai = self.typed(depth + 1)
        b, bi = self.typed(depth + 1)
        op = r.choice(["+", "-", "*", "+", "-", "//", "%"])
        if op == "//" and not (ai and bi):
            op = "+"
        return f"({a} {op} {b})", ai and bi

    def num(self, depth=0):
        return self.typed(depth)[0]

    def cond(self, depth=0):
        r = self.rng
        if depth > 2 or r.random() < 0.4:
            t = r.random()
            if t < 0.55:
                a, b = self.num(1), self.num(1)
                return f"{a} {r.choice(['=', '<>', '<', '<=', '>', '>='])} {b}"
            if t < 0.65:
                return f"{r.choice(['x', 'k', 'd', 'e', 's', 'dt', 'b'])} IS {r.choice(['', 'NOT '])}NULL"
            if t < 0.75:
                items = ", ".join(str(r.randint(-3, 5)) for _ in range(r.randint(1, 4))) + (", NULL" if r.random() < 0.2 else "")
                return f"{r.choice(self.INTS)} {r.choice(['', 'NOT '])}IN ({items})"
            if t < 0.82:
                items = ", ".join(repr(r.choice(WORDS + ["zz"])) for _ in range(r.randint(1, 3)))
                return f"s {r.choice(['', 'NOT '])}IN ({items})"
            if t < 0.88:
                return f"s {r.choice(['=', '<>'])} {repr(r.choice(WORDS + ['zz']))}"
            if t < 0.94:
                return f"dt {r.choice(['<', '>=', '='])} DATE '20{r.randint(18, 22)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}'"
            return r.choice(["b", "NOT b", "b = true", "b <> false"])
        c = f"({self.cond(depth + 1)} {r.choice(['AND', 'OR'])} {self.cond(depth + 1)})"
        return f"NOT {c}" if r.random() < 0.15 else c

    def agg(self):
        r = self.rng
        f = r.choice(["sum", "sum", "count", "min", "max", "avg", "count", "countstar", "countd", "median", "q"])
        e = self.num(1)
        if f == "countstar":
            return "count(*)"
        if f == "countd":
            return f"count(DISTINCT {r.choice(self.INTS + self.DECS + ['s', 'dt', 'b'])})"
        if f in ("min", "max") and r.random() < 0.3:
            return f"{f}({r.choice(['dt', 'b', 'd', 'x'])})"
        if f == "median":
            return f"median({r.choice(self.INTS)})"
        if f == "q":
            return f"quantile_disc({r.choice(self.INTS + self.DECS)}, {r.choice([0.1, 0.5, 0.9, 0, 1])})"
        return f"{f}({e})"

    def query(self):
        r = self.rng
        keys = r.choice([[], ["k"], ["s"], ["b"], ["k", "s"], ["g"], ["dt"], ["x % 5"], ["k", "b"], ["x // 100"], ["s", "k", "b"]])
        q = f"SELECT {', '.join(keys + [self.agg() for _ in range(r.randint(1, 4))])} FROM t"
        if r.random() < 0.6:
            q += f" WHERE {self.cond()}"
        if keys:
            q += " GROUP BY " + ", ".join(keys)
        if keys and r.random() < 0.25:
            q += f" HAVING count(*) > {r.randint(0, 5)}"
        return q


@pytest.mark.parametrize("mode", ["table", "frame", "scan"])
@pytest.mark.parametrize("seed", [11, 12])
def test_random_queries_equal_duckdb(probe, data, mode, seed):
    ds = data["g8"]
    gen = _Gen(seed * 100 + len(mode))
    compared = 0
    for _ in range(40):
        q = gen.query()
        res = run_probe(probe, q, {"t": ds.path}, mode=mode, edits=ds.edits_path if mode == "frame" else None, expect_error=True)
        if "error" in res:
            raise AssertionError(f"{mode}: `{q}`: {res['error']}")
        compared += 1
        assert_source_path(res, mode)
        want = ds.oracle(mode).execute(q).fetchall()
        assert_equal(got_rows(res), want, False, f"{mode}: {q}")
    assert compared >= 25, f"only {compared} of 40 random queries ran"


@pytest.mark.parametrize("keys", ["g", "k", "dt", "k, b", "g, k", "d", "e", "x + 1", "k * 3 - 2", "x // 100, k"])
def test_a_scans_statistics_plan_like_the_table(probe, data, keys):
    """The row groups' min / max statistics give a streamed file the ranges a table gets from a pass over its column, so the
    same keys get the same strategy with the same domain."""
    sql = f"EXPLAIN SELECT {keys}, count(*) FROM t GROUP BY {keys}"
    plans = []
    for mode in ("table", "scan"):
        lines = [r[0] for r in run_probe(probe, sql, {"t": data["g8"].path}, mode=mode)["rows"]]
        plans.append(next(l for l in lines if l.startswith("  group:")))
    assert plans[0] == plans[1], plans


def test_a_decimal_product_past_int64_raises(probe, data):
    for mode in ("table", "scan"):
        res = run_probe(probe, "SELECT sum(d * e * x * id) FROM t", {"t": data["g1m"].path}, mode=mode, expect_error=True)
        assert res["error"] == "a sum or an expression passed the int64 range (about 9.2e18)", (mode, res)


def test_names_are_case_insensitive_quoted_names_and_aliases_work(probe, data):
    ds = data["g8"]
    sql = 'SELECT a."K" AS "My Group", COUNT(*) AS "Total Rows", SUM(A.X) FROM "T" AS a WHERE a.D IS NOT NULL GROUP BY a.k ORDER BY 2 DESC, 1'
    res = run_probe(probe, sql, {"T": ds.path})
    assert [c["name"] for c in res["columns"]] == ["My Group", "Total Rows", "sum(A.X)"]
    want = ds.table.execute('SELECT k, COUNT(*), SUM(x) FROM t WHERE d IS NOT NULL GROUP BY k ORDER BY 2 DESC, 1').fetchall()
    assert_equal(got_rows(res), want, True, sql)


@pytest.mark.parametrize("mode", ["frame", "framescan"])
def test_a_frames_formula_columns_are_columns(probe, data, mode):
    """f1 = d - x (a decimal minus an integer, brought to one scale), f2 = f1 * 2 (a formula over a formula), f3 = d + e (two scales):
    the frame's formulas, after its 300 edits, queried like stored columns, equal the same expressions in DuckDB."""
    ds = data["g8"]
    script = ds.edits_path.with_suffix(".formulas")
    script.write_text(ds.edits_path.read_text() + "formula\tf1 = d - x\nformula\tf2 = f1 * 2\nformula\tf3 = d + e\n")
    con = ds.edited
    con.execute("create or replace view tf as select *, d - x as f1, (d - x) * 2 as f2, d + e as f3 from (select * exclude (rid) from src)")
    queries = [
        ("SELECT k, sum(f1), min(f2), max(f3), count(f1), avg(f3) FROM t GROUP BY k", False),
        ("SELECT count(*), sum(f3), min(f1) FROM t WHERE f1 > f3 OR f2 IS NULL", False),
        ("SELECT f2, count(*) FROM t WHERE k = 1 GROUP BY f2 HAVING count(*) > 1", False),
        ("SELECT s, sum(f2 + 1), sum(f3 - f1) FROM t WHERE f3 BETWEEN -50 AND 50 GROUP BY s", False),
        ("SELECT k, sum(f1) AS total FROM t GROUP BY k ORDER BY total DESC NULLS LAST, k", True),
    ]
    for sql, ordered in queries:
        res = run_probe(probe, sql, {"t": ds.path}, mode=mode, edits=script)
        assert_source_path(res, mode)
        want = con.execute(sql.replace(" FROM t", " FROM tf")).fetchall()
        assert_equal(got_rows(res), want, ordered, f"{mode}: {sql}")


def test_a_duplicate_output_name_gets_a_suffix(probe, data):
    res = run_probe(probe, "SELECT k, k, count(*) AS k FROM t GROUP BY k", {"t": data["g8"].path})
    assert [c["name"] for c in res["columns"]] == ["k", "k_1", "k_2"]


def test_the_register_limit_is_an_sql_error(probe, data):
    aggs = ", ".join(f"sum(x + {i})" for i in range(40))
    res = run_probe(probe, f"SELECT k, {aggs} FROM t GROUP BY k", {"t": data["g8"].path}, expect_error=True)
    assert res["error"] == ("SQL error at line 1, column 8: the query needs more than 48 registers (the select list's aggregates and expressions are too many "
                            "or too long for one program: about 20 aggregates over different expressions, fewer when an expression is long); split it into two queries"), res


# ── files whose footer statistics lie ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def lying(tmp_path_factory):
    """Four row groups of 500 rows. `honest.parquet` has g in 0..9 except one 1000 in row group 3 and one NULL in `gn` there. In
    the lying copies the footer says that row group's maximum is 9 (`lie_max`), and also that it holds no NULL (`lie_both`)."""
    import struct
    import pyarrow as pa
    d = tmp_path_factory.mktemp("lying")
    rng = random.Random(5)
    n = 2000
    g = [rng.randint(0, 9) for _ in range(n)]
    k = [rng.randint(0, 3) for _ in range(n)]
    g[1500] = 1000
    gn = list(g)
    gn[1700] = None
    tbl = pa.table({"g": pa.array(g, pa.int64()), "k": pa.array(k, pa.int64()), "gn": pa.array(gn, pa.int64())})
    honest = d / "honest.parquet"
    pq.write_table(tbl, honest, row_group_size=500, write_page_index=False)
    raw = bytearray(honest.read_bytes())
    flen = struct.unpack("<I", raw[-8:-4])[0]
    start = len(raw) - 8 - flen
    foot = bytes(raw[start:-8])
    assert foot.count(struct.pack("<q", 1000)) >= 2                       # the maxima of g and gn in row group 3
    foot = foot.replace(struct.pack("<q", 1000), struct.pack("<q", 9))
    (d / "lie_max.parquet").write_bytes(bytes(raw[:start]) + foot + bytes(raw[-8:]))
    nulls = bytearray(foot)
    patched = 0
    for i in range(len(nulls) - 1):
        if nulls[i] == 0x16 and nulls[i + 1] == 0x02:                      # Statistics.null_count = 1 (zigzag 2)
            nulls[i + 1] = 0x00
            patched += 1
    assert patched >= 1
    (d / "lie_both.parquet").write_bytes(bytes(raw[:start]) + bytes(nulls) + bytes(raw[-8:]))
    m = pq.ParquetFile(d / "lie_both.parquet").metadata
    assert m.row_group(3).column(2).statistics.null_count == 0 and m.row_group(3).column(0).statistics.max == 9
    return {"honest": honest, "lie_max": d / "lie_max.parquet", "lie_both": d / "lie_both.parquet", "con": duckdb.connect()}


LIES = [      # (the query, whether `lie_max` makes it run again, whether `lie_both` does)
    ("SELECT g, count(*) FROM t GROUP BY g", True, True),
    ("SELECT k, g, count(*) FROM t GROUP BY k, g", True, True),            # a key past its range used to land in the next key's slot
    ("SELECT g % 4, count(*), sum(g) FROM t GROUP BY g % 4", False, False),   # `g % 4` stays in 0..3 whatever g's range
    ("SELECT gn, count(*) FROM t GROUP BY gn", True, True),
    ("SELECT k, gn, count(*) FROM t GROUP BY k, gn", True, True),
    ("SELECT count(*), count(gn), sum(gn), min(gn), max(gn) FROM t", False, True),     # a NULL read as 0 without the null count's check
]


@pytest.mark.parametrize("mode", ["scan", "framescan"])
@pytest.mark.parametrize("file", ["honest", "lie_max", "lie_both"])
@pytest.mark.parametrize("sql,again_max,again_both", LIES)
def test_a_footer_that_lies_cannot_change_an_answer(probe, lying, mode, file, sql, again_max, again_both):
    """The statistics are a hint, checked as the row groups are read: a maximum that is not one, or a null count that is too small,
    makes the query run again without them (hashed groups, every column nullable), and the notes say so. The answer is DuckDB's on
    the honest file's data either way."""
    want = lying["con"].execute(sql.replace("FROM t", f"FROM read_parquet('{lying['honest']}')")).fetchall()
    res = run_probe(probe, sql, {"t": lying[file]}, mode=mode)
    assert_equal(got_rows(res), want, False, f"{file}/{mode}: {sql}")
    again = [n for n in res["notes"] if n.startswith("statistics:")]
    expect = {"honest": False, "lie_max": again_max, "lie_both": again_both}[file]
    if expect:
        assert len(again) == 1 and "ran again without them" in again[0], res["notes"]
        if "GROUP BY" in sql:
            assert next(n for n in res["notes"] if n.startswith("group:")).startswith("group: hashed"), res["notes"]
    else:
        assert not again, again
    if file == "honest" and "GROUP BY" in sql and "% 4" not in sql:
        assert next(n for n in res["notes"] if n.startswith("group:")).startswith("group: dense"), res["notes"]    # the statistics were used


@pytest.mark.parametrize("mode", MODES)
def test_division_by_zero_is_ieee_like_duckdb(probe, data, mode):
    """`/` is a double division: 1 / 0 is inf, -1 / 0 is -inf, 0 / 0 is NaN, 1.5 / 0 is inf; `//` and `%` by zero are NULL."""
    for q in [Q("SELECT k, sum(x) / 0, -sum(x) / 0, sum(x) * 0 / 0, sum(d) / 0, sum(x) / sum(k - k), 1 / 0, -1 / 0, 0 / 0, 1.5 / 0 FROM t GROUP BY k", keys=["k"]),
              Q("SELECT k, sum(x) // 0, sum(x) % 0, 7 // 0, 7 % 0 FROM t GROUP BY k", keys=["k"]),
              Q("SELECT k, sum(x) / (sum(k) - sum(k)) AS r FROM t GROUP BY k ORDER BY r, k", keys=["k"], ordered=True),
              Q("SELECT k, count(*) FROM t GROUP BY k HAVING sum(x) / 0 > 1e300 OR sum(x) * 0 / 0 = sum(x) * 0 / 0".replace("1e300", "1000000"), keys=["k"])]:
        run_case(probe, data, "g8", mode, q)


def test_constant_divisions_by_zero(probe, data):
    res = run_probe(probe, "SELECT 1 / 0, -1 / 0, 0 / 0, 1.5 / 0, count(*) FROM t", {"t": data["g8"].path})
    inf, ninf = float("inf"), float("-inf")
    row = got_rows(res)[0]
    assert (row[0], row[1], row[3], row[4]) == (inf, ninf, inf, 20000) and row[2] != row[2]


# ── a limit on nesting: the parser, the binder and the program builder recurse ───────────────────────

def _depth_error(probe, data, where):
    res = run_probe(probe, f"SELECT count(*) FROM t WHERE {where}", {"t": data["g8"].path}, expect_error=True)
    return res.get("error")


def test_deeply_nested_expressions_raise_a_positioned_error_instead_of_crashing(probe, data):
    prefix = len("SELECT count(*) FROM t WHERE ")
    parens = _depth_error(probe, data, "(" * 300 + "x > 1" + ")" * 300)
    assert parens == f"SQL error at line 1, column {prefix + 257}: this expression is nested more than 256 levels deep (parentheses, NOT or a minus sign repeated)", parens
    nots = _depth_error(probe, data, "NOT " * 3000 + "b")
    assert nots == f"SQL error at line 1, column {prefix + 4 * 255 + 1}: this expression is nested more than 256 levels deep (parentheses, NOT or a minus sign repeated)", nots
    minus = _depth_error(probe, data, "x > " + "- " * 400 + "1")
    assert minus.startswith("SQL error at line 1, column ") and "nested more than 256 levels deep" in minus, minus


def test_expressions_at_the_limit_run(probe, data):
    ds = data["g8"]
    for where, want_sql in [("(" * 250 + "x > 1" + ")" * 250, "x > 1"), ("NOT " * 250 + "b", "b"), (" OR ".join(["x = 1"] * 255), "x = 1"),
                            ("x + " * 254 + "1 > 5", "x * 254 + 1 > 5")]:
        res = run_probe(probe, f"SELECT count(*) FROM t WHERE {where}", {"t": ds.path})
        assert got_rows(res) == ds.table.execute(f"SELECT count(*) FROM t WHERE {want_sql}").fetchall(), want_sql


def test_long_chains_of_operators_are_capped_and_a_shorter_one_runs(probe, data):
    prefix = len("SELECT count(*) FROM t WHERE ")
    for op, make in [("OR", lambda n: " OR ".join(f"x = {i % 5}" for i in range(n))), ("AND", lambda n: " AND ".join(f"x <> {i % 5}" for i in range(n)))]:
        msg = _depth_error(probe, data, make(3000))
        assert msg.startswith("SQL error at line 1, column ") and f"more than the 256 allowed (a chain of that many terms joined by `{op}` counts)" in msg, msg
        assert _depth_error(probe, data, make(200)) is None
    res = run_probe(probe, "SELECT sum(" + " + ".join(["x"] * 4000) + ") FROM t", {"t": data["g8"].path}, expect_error=True)
    assert res["error"].startswith("SQL error at line 1, column ") and "more than the 256 allowed (a chain of that many terms joined by `+` counts)" in res["error"], res
    ok = run_probe(probe, "SELECT sum(" + " + ".join(["x"] * 200) + ") FROM t", {"t": data["g8"].path})
    want = data["g8"].table.execute("SELECT sum(x) * 200 FROM t").fetchall()
    assert_equal(got_rows(ok), want, False, "a 200-term sum")
