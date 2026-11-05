"""F-176: range aggregates and block reads over a row order, equal to DuckDB.

mojo/tools/range_probe.mojo is run as a subprocess: it opens a Parquet file as a table, as a streamed scan, or as a Frame (over either)
after an edit script, sorts it with `Frame.permutation` (or takes the rows as they are), and prints count / sum / mean / min / max of every
column over positions a..b of that order, plus the block of cells itself, as typed JSON. DuckDB gets the same question as
`row_number() OVER (ORDER BY key, rid)` and a `WHERE pos >= a AND pos < b`:

- INT and DECIMAL: sums and counts exactly, the mean to the last bit (DuckDB's AVG), min and max exactly;
- FLOAT: min and max exactly (NaN above +inf, -0.0 = 0.0), the sum to 1e-12 of the sum of the absolute values (DuckDB adds in another order),
  the same class (NaN / +inf / -inf) when a NaN or an infinity is in the range;
- TEXT: min and max in the case-insensitive order (DuckDB under (lower(s), s)); DATE and BOOL: min and max;
- NULL cells are skipped; an empty range has no sum, mean, min or max;
- the block, cell by cell.

The data has nulls, decimals of two scales, a float column with NaN / +-inf / -0.0 / subnormals, text with case variants, a Parquet file of
row groups of 6,999 rows (a range crosses them), and 300 edits (sets, nulls, new words, appended and removed rows, a merge).
"""
import datetime
import fcntl
import json
import math
import os
import random
import struct
import subprocess
from decimal import Decimal
from pathlib import Path

import duckdb
from conftest import oracle_connection
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from engine_sources import engine_sources

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
N = 60_000
ROW_GROUP = 6_999
WORDS = ["West", "west", "WEST", "east", "East", "North", "apple", "Zebra", "zebra", "mango", "Mango", "ändern", "Zoë"]
COLS = [("i", "int"), ("d2", "dec2"), ("d4", "dec4"), ("f", "float"), ("fs", "floats"), ("s", "text"), ("dt", "date"), ("b", "bool")]
NAMES = [c for c, _ in COLS]


def _build_probe(name, flags=""):
    BUILD.mkdir(exist_ok=True)
    out = BUILD / name
    src = engine_sources(ROOT / "mojo/tools/range_probe.mojo")
    with open(BUILD / ".range_probe.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not out.exists() or out.stat().st_mtime < max(p.stat().st_mtime for p in src):
            subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/range_probe.mojo"), str(out)],
                           check=True, capture_output=True, text=True, env={**os.environ, "MXFRAME_BUILD_FLAGS": flags})
    return out


@pytest.fixture(scope="session")
def probe():
    return _build_probe("range_probe")


@pytest.fixture(scope="session")
def probe_serial():
    """The same program built with `-D MXFRAME_SERIAL`: every task runs on the calling thread (what an Apple build gets)."""
    return _build_probe("range_probe_serial", "-D MXFRAME_SERIAL")


# ── data ─────────────────────────────────────────────────────────────────────────────────────────────

def make_arrow(seed):
    rng = np.random.default_rng(seed)

    def mask(p):
        return rng.random(N) < p

    i = pa.array(rng.integers(-1000, 1001, N), mask=mask(0.10))
    d2 = pa.array([None if m else Decimal(int(v)).scaleb(-2) for v, m in zip(rng.integers(-100_000, 100_000, N), mask(0.10))], type=pa.decimal128(18, 2))
    d4 = pa.array([None if m else Decimal(int(v)).scaleb(-4) for v, m in zip(rng.integers(-10_000_000, 10_000_000, N), mask(0.12))], type=pa.decimal128(9, 4))
    f = pa.array(np.round(rng.uniform(-1e4, 1e4, N), 3), mask=mask(0.10))
    fs = rng.uniform(-1e4, 1e4, N)
    pick = rng.random(N)
    special = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 5e-324, -5e-324])
    fs = np.where(pick < 0.02, special[rng.integers(0, len(special), N)], fs)
    fs = pa.array(fs, mask=mask(0.08))
    s = pa.array([None if m else WORDS[k] for k, m in zip(rng.integers(0, len(WORDS), N), mask(0.10))], type=pa.string())
    dt = pa.array(rng.integers(-3000, 25000, N).astype(np.int32), type=pa.date32(), mask=mask(0.05))
    b = pa.array(rng.random(N) < 0.4, mask=mask(0.08))
    return pa.table({"i": i, "d2": d2, "d4": d4, "f": f, "fs": fs, "s": s, "dt": dt, "b": b})


def duck_over(path):
    con = oracle_connection()
    con.execute("set threads=1")
    con.execute(f"create table src as select row_number() over () - 1 as rid, * from read_parquet('{path}')")
    return con


def make_edits(con, seed):
    """A deterministic edit script (the probe's format) and the same edits applied to the DuckDB table `src`."""
    rng = random.Random(seed)
    lines = []
    alive = set(range(N))
    next_rid = N
    pool = WORDS + ["zzz-new", "aaa-new", "Mid", "q"]

    def value(kind):
        if kind == "int":
            return str(rng.randint(-1500, 1500))
        if kind == "dec2":
            return f"{rng.randint(-100000, 100000) / 100:.2f}"
        if kind == "dec4":
            return f"{rng.randint(-1000000, 1000000) / 10000:.4f}"
        if kind == "float":
            return f"{rng.uniform(-1e4, 1e4):.3f}"
        if kind == "floats":
            return rng.choice(["nan", "inf", "-inf", "-0.0", "0", f"{rng.uniform(-1e4, 1e4):.3f}", f"{rng.uniform(-1e4, 1e4):.3f}"])
        if kind == "text":
            return rng.choice(pool)
        if kind == "date":
            return (datetime.date(2017, 6, 1) + datetime.timedelta(days=rng.randint(0, 2200))).isoformat()
        return rng.choice(["true", "false"])

    def lit(kind, v):
        if v == "":
            return "NULL"
        if kind in ("int", "dec2", "dec4", "bool"):
            return v
        if kind in ("float", "floats"):
            return f"'{v}'::DOUBLE"
        return "'" + v.replace("'", "''") + "'"

    def alive_row():
        while True:
            row = rng.randrange(N)
            if row in alive:
                return row

    for _ in range(300):
        what = rng.randint(0, 11)
        if what <= 6:
            row = alive_row()
            name, kind = rng.choice(COLS)
            v = "" if what == 6 else value(kind)
            lines.append(f"set\t{name}\t{row}\t{v}")
            con.execute(f"update src set {name} = {lit(kind, v)} where rid = {row}")
        elif what in (7, 8):
            vals = ["" if rng.random() < 0.15 else value(kind) for _, kind in COLS]
            lines.append("append\t" + "\t".join(vals))
            con.execute(f"insert into src (rid, {', '.join(NAMES)}) values ({next_rid}, {', '.join(lit(k, v) for (_, k), v in zip(COLS, vals))})")
            alive.add(next_rid)
            next_rid += 1
        elif what == 9:
            row = alive_row()
            lines.append(f"remove\t{row}")
            con.execute(f"delete from src where rid = {row}")
            alive.discard(row)
        else:
            row = alive_row()
            name, kind = rng.choice(COLS)
            v = value(kind)
            lines.append(f"set\t{name}\t{row}\t{v}")
            con.execute(f"update src set {name} = {lit(kind, v)} where rid = {row}")
    lines.append("merge\ts\tWest\twest,WEST")
    con.execute("update src set s = 'West' where s in ('west', 'WEST')")
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("rangedata")
    path = d / "t.parquet"
    pq.write_table(make_arrow(7), path, row_group_size=ROW_GROUP)
    plain = duck_over(path)
    edited = duck_over(path)
    edits = d / "edits.txt"
    edits.write_text(make_edits(edited, 11))
    return {"path": path, "edits": edits, "plain": plain, "edited": edited}


# ── the question to DuckDB ───────────────────────────────────────────────────────────────────────────

def order_sql(order):
    if order == "none":
        return "rid"
    name, _, direction = order.partition(":")
    desc = " DESC" if direction == "desc" else ""
    if name == "s":
        return f"lower(s){desc} NULLS LAST, s{desc} NULLS LAST, rid"
    return f"{name}{desc} NULLS LAST, rid"


def duck_aggregates(con, order, a, b):
    parts = []
    for name, kind in COLS:
        if kind in ("int", "dec2", "dec4"):
            parts.append(f"count({name}), sum({name}), avg({name}), min({name}), max({name})")
        elif kind in ("float", "floats"):
            parts.append(f"count({name}), sum({name}), avg({name}), min({name}), max({name}), sum(abs({name})) filter (where isfinite({name}))")
        elif kind == "text":
            parts.append(f"count({name}), arg_min({name}, (lower({name}), {name})), arg_max({name}, (lower({name}), {name}))")
        else:
            parts.append(f"count({name}), min({name}), max({name})")
    q = (f"with o as (select *, row_number() over (order by {order_sql(order)}) - 1 as pos from src) "
         f"select {', '.join(parts)} from o where pos >= {a} and pos < {b}")
    return con.execute(q).fetchone()


def duck_block(con, order, a, b):
    q = (f"with o as (select *, row_number() over (order by {order_sql(order)}) - 1 as pos from src) "
         f"select {', '.join(NAMES)} from o where pos >= {a} and pos < {b} order by pos")
    return con.execute(q).fetchall()


def run_probe(probe, data, mode, order, a, b, out="both"):
    cmd = [str(probe), "--mode", mode, "--order", order, "--out", out, "--a", str(a), "--b", str(b), "--cols", ",".join(NAMES)]
    if mode in ("frame", "framescan"):
        cmd += ["--edits", str(data["edits"])]
    cmd.append(str(data["path"]))
    p = subprocess.run(cmd, capture_output=True)
    text = p.stdout.decode("utf-8")
    res = json.loads(text) if text.strip() else {"error": "no output: " + p.stderr.decode("utf-8", "replace")[-400:]}
    assert "error" not in res, res["error"]
    return res


def bits(h):
    return struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]


def typed(kind, v):
    if v is None:
        return None
    if kind in ("dec2", "dec4"):
        return Decimal(v)
    if kind == "date":
        return datetime.date.fromisoformat(v)
    if kind in ("float", "floats"):
        return bits(v)
    return v


def same_float(a, b):
    return (a is None and b is None) or (a is not None and b is not None and ((math.isnan(a) and math.isnan(b)) or a == b))


def near(a, b, scale):
    if a is None or b is None:
        return a is None and b is None
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return same_float(a, b)
    return abs(a - b) <= 1e-12 * max(scale, 1e-300) + 1e-300


def check_aggregates(res, row, label):
    at = 0
    for k, (name, kind) in enumerate(COLS):
        g = res["columns"][k]
        tag = f"{label} {name}"
        assert g["name"] == name
        count = row[at]
        assert g["count"] == count, f"{tag}: count {g['count']} vs {count}"
        if kind in ("int", "dec2", "dec4"):
            s, avg, mn, mx = row[at + 1:at + 5]
            at += 5
            assert (typed(kind, g["sum"]) if g["sum"] is not None else None) == s, f"{tag}: sum {g['sum']} vs {s}"
            got_mean = bits(g["mean"]) if g["mean"] is not None else None
            assert (got_mean is None and avg is None) or struct.pack("<d", got_mean) == struct.pack("<d", avg), f"{tag}: mean {got_mean!r} vs {avg!r}"
            assert (typed(kind, g["min"]), typed(kind, g["max"])) == (mn, mx), f"{tag}: min/max {g['min']},{g['max']} vs {mn},{mx}"
        elif kind in ("float", "floats"):
            s, avg, mn, mx, sabs = row[at + 1:at + 6]
            at += 6
            scale = sabs or 0.0
            ours_sum = bits(g["sum"]) if g["sum"] is not None else None
            assert near(ours_sum, s, scale), f"{tag}: sum {ours_sum!r} vs {s!r}"
            assert near(bits(g["mean"]) if g["mean"] is not None else None, avg, scale / max(count, 1)), f"{tag}: mean"
            assert same_float(bits(g["min"]) if g["min"] is not None else None, mn), f"{tag}: min {g['min']} vs {mn!r}"
            assert same_float(bits(g["max"]) if g["max"] is not None else None, mx), f"{tag}: max {g['max']} vs {mx!r}"
        else:
            mn, mx = row[at + 1:at + 3]
            at += 3
            assert g["sum"] is None, f"{tag}: a {kind} has no sum"
            assert (typed(kind, g["min"]), typed(kind, g["max"])) == (mn, mx), f"{tag}: min/max {g['min']},{g['max']} vs {mn},{mx}"


def check_block(res, rows, label):
    assert len(res["block"]) == len(rows), f"{label}: {len(res['block'])} rows vs {len(rows)}"
    for r, (got, want) in enumerate(zip(res["block"], rows)):
        for k, (name, kind) in enumerate(COLS):
            g = typed(kind, got[k])
            w = want[k]
            if kind in ("float", "floats"):
                assert same_float(g, w), f"{label}: row {r} {name}: {g!r} vs {w!r}"
            else:
                assert g == w, f"{label}: row {r} {name}: {g!r} vs {w!r}"


ORDERS = ["none", "i", "d2:desc", "f", "fs", "fs:desc", "s", "s:desc", "dt:desc", "b"]
MODES = ["table", "scan", "frame", "framescan"]


def ranges(n):
    return [(0, 0), (5, 6), (0, n), (1_000, 40_000), (n - 1, n), (16_383, 16_385), (6_998, 7_001), (12_345, 12_346), (n // 3, n // 3 + 20_000), (n - 50, n)]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("order", ORDERS)
def test_equals_duckdb(probe, data, mode, order):
    edited = mode in ("frame", "framescan")
    con = data["edited"] if edited else data["plain"]
    if edited and order == "none":
        pytest.skip("the rows themselves include removed rows: they raise (tests/test_mojo_range.py::test_removed_rows)")
    n = con.execute("select count(*) from src").fetchone()[0]
    for a, b in ranges(n):
        label = f"{mode} {order} {a}..{b}"
        res = run_probe(probe, data, mode, order, a, b, "agg" if b - a > 300 else "both")
        assert res["rows"] == b - a
        check_aggregates(res, duck_aggregates(con, order, a, b), label)
        if b - a <= 300:
            check_block(res, duck_block(con, order, a, b), label)


@pytest.mark.parametrize("mode", ["frame", "framescan"])
def test_removed_rows(probe, data, mode):
    cmd = [str(probe), "--mode", mode, "--order", "none", "--a", "0", "--b", "60000", "--cols", "i", "--edits", str(data["edits"]), str(data["path"])]
    out = subprocess.run(cmd, capture_output=True).stdout.decode()
    err = json.loads(out)["error"]
    assert err.endswith(" was removed") and err.startswith("row "), err


def test_out_of_range(probe, data):
    cmd = [str(probe), "--mode", "table", "--order", "none", "--a", "10", "--b", "70000", "--cols", "i", str(data["path"])]
    err = json.loads(subprocess.run(cmd, capture_output=True).stdout.decode())["error"]
    assert err == "the range 10..70000 goes past the end: there are 60000 rows"


@pytest.mark.parametrize("order", ["none", "fs", "f:desc"])
@pytest.mark.parametrize("a,b", [(0, 60_000), (7, 59_990), (1_000, 40_000)])
def test_float_sums_do_not_depend_on_threads(probe, probe_serial, data, order, a, b):
    """The FLOAT sum is added in morsels from the first position and folded in order: one thread and many give the same bits."""
    cols = "f,fs,d2,i"
    outs = []
    for exe in (probe, probe_serial):
        cmd = [str(exe), "--mode", "table", "--order", order, "--out", "agg", "--a", str(a), "--b", str(b), "--cols", cols, str(data["path"])]
        outs.append(json.loads(subprocess.run(cmd, capture_output=True).stdout.decode())["columns"])
    for x, y in zip(*outs):
        assert (x["sum"], x["mean"], x["min"], x["max"]) == (y["sum"], y["mean"], y["min"], y["max"]), x["name"]
