"""F-178: SQL over a CSV file streamed a row group at a time (`scan_csv`) equals DuckDB's `read_csv` of the same bytes.

The Mojo side is mojo/tools/sql_probe.mojo (a path ending in .csv is read whole in `table` / `frame` modes and streamed with `scan_csv` in
`scan` / `framescan` modes, `--group-rows` and `--block-bytes` pick the row groups). The oracle is DuckDB reading the same file with the
column types the data has. The data has nulls, mixed-scale decimals, dates, booleans, and text with a comma, a quote and a newline in it,
so a quoted field is cut by a block boundary in the small-block runs. `framescan` applies the 300 random edits of test_mojo_sql to the
streamed file and to a DuckDB table.
"""
import re

import duckdb
import pytest

from mx_oracle import assert_equal, duck_kind
from test_mojo_sql import QUERIES, Q, make_source, make_edits, got_rows, run_probe, query_id, probe  # noqa: F401  (probe is a fixture)

N = 30_000
GROUPS = 400
TYPES = {"id": "BIGINT", "g": "BIGINT", "k": "BIGINT", "x": "BIGINT", "d": "DECIMAL(18,2)", "e": "DECIMAL(18,4)", "s": "VARCHAR", "dt": "DATE", "b": "BOOLEAN"}
TYPE_SQL = "{" + ", ".join(f"'{c}': '{t}'" for c, t in TYPES.items()) + "}"
AWKWARD = "update src set s = case s when 'banana' then 'ban,ana' when 'mango' then 'man\"go' when 'Mango' then 'Man' || chr(10) || 'go' else s end"

# (rows in a row group, bytes of file looked at at once): many small groups in one big window, and groups cut by a small window
SHAPES = [(2_500, 16 << 20), (1_000_000, 4096)]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("csvscan")
    con = duckdb.connect()
    make_source(con, N, GROUPS, 5)
    con.execute(AWKWARD)
    path = d / "t.csv"
    con.execute(f"copy (select * exclude (rid) from src order by rid) to '{path}' (header)")
    text = path.read_text(encoding="utf8")
    assert '"Man\ngo"' in text and '"ban,ana"' in text and 'man""go' in text          # the awkward fields are in the file, quoted
    table = duckdb.connect()
    table.execute(f"create table t as select * from read_csv('{path}', header=true, columns={TYPE_SQL})")
    table.execute("set threads=4")
    edits_path = d / "t.edits"
    edited = duckdb.connect()
    edited.execute("set threads=1")
    edited.execute(f"create table src as select row_number() over () - 1 as rid, * from read_csv('{path}', header=true, columns={TYPE_SQL})")
    edits_path.write_text(make_edits(edited, N, GROUPS, 6))
    edited.execute("create view t as select * exclude (rid) from src")
    return {"path": path, "table": table, "edited": edited, "edits": edits_path}


def run_case(probe, data, mode, q, shape=SHAPES[0]):
    extra = ["--group-rows", str(shape[0]), "--block-bytes", str(shape[1])]
    cmd_edits = data["edits"] if mode in ("frame", "framescan") else None
    res = run_probe_with(probe, q.sql, {"t": data["path"]}, mode, cmd_edits, extra)
    con = data["edited"] if mode in ("frame", "framescan") else data["table"]
    cur = con.execute(q.duck or q.sql)
    want = cur.fetchall()
    if q.fix:
        want = q.fix(want)
    if not q.fix:
        for i, desc in enumerate(cur.description):
            dk = duck_kind(desc[1])
            c = res["columns"][i]
            assert c["kind"] == dk[0] and (dk[0] != "DECIMAL" or c["scale"] == dk[1]), f"column {i} of `{q.sql}`: got {c}, DuckDB has {dk}"
    assert_equal(got_rows(res), want, q.ordered, f"{mode} {shape}: {q.sql}")
    return res


def run_probe_with(probe, sql, files, mode, edits, extra):
    """run_probe with the CSV flags in front of the SQL (run_probe puts only its own flags there)."""
    import json
    import subprocess
    cmd = [str(probe), "--mode", mode, *extra]
    if edits:
        cmd += ["--edits", str(edits)]
    cmd += [sql] + [f"{k}={v}" for k, v in files.items()]
    p = subprocess.run(cmd, capture_output=True)
    text = p.stdout.decode("utf-8")
    out = json.loads(text) if text.strip() else {"error": "no output: " + p.stderr.decode("utf-8", "replace")[-400:]}
    assert "error" not in out, f"the probe failed on `{sql}`: {out['error']}"
    return out


ROW_QUERIES = [
    Q("SELECT id, s, dt, d FROM t WHERE x > 900 ORDER BY id LIMIT 50", ordered=True, tag="rows-ordered"),
    Q("SELECT s, count(*) FROM t WHERE s LIKE '%o%' GROUP BY s", keys=["s"], tag="like"),
    Q("SELECT DISTINCT k, b FROM t", tag="distinct"),
]
ALL = [q for q in QUERIES + ROW_QUERIES if "g1m" not in q.skip]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"group_rows={s[0]},block_bytes={s[1]}")
@pytest.mark.parametrize("q", ALL, ids=query_id)
def test_a_csv_scan_equals_duckdb(probe, data, q, shape):
    res = run_case(probe, data, "scan", q, shape)
    ran = next(n for n in res["notes"] if n.startswith("ran:"))
    source = next(n for n in res["notes"] if n.startswith("source:"))
    assert "CSV scan, " in source, source
    m = re.match(r"ran: scan t: (\d+) of (\d+) row groups read", ran)
    assert m and m.group(1) == m.group(2) or "LIMIT" in q.sql, ran


@pytest.mark.parametrize("q", ALL, ids=query_id)
def test_the_whole_file_reader_equals_duckdb(probe, data, q):
    run_case(probe, data, "table", q)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"group_rows={s[0]},block_bytes={s[1]}")
@pytest.mark.parametrize("q", ALL[::2], ids=query_id)
def test_edits_over_a_csv_scan_equal_duckdb(probe, data, q, shape):
    run_case(probe, data, "framescan", q, shape)


def test_bad_values_past_the_sample_are_nulls_and_the_rest_of_the_row_is_read(probe, tmp_path):
    """Values that do not fit their column's kind (the kind comes from the first 100 rows) become nulls, as in read_csv; DuckDB reads the
    same data with those cells empty."""
    n = 5_000
    bad = {(150, "x"): "oops", (2_001, "x"): "1.5", (4_000, "dt"): "2024-13-45", (4_001, "d"): "n/a", (4_999, "b"): "maybe"}
    lines, clean = ["id,x,d,dt,b,s"], ["id,x,d,dt,b,s"]
    for i in range(n):
        cells = {"id": str(i), "x": str(i % 97 - 40), "d": f"{(i % 500) / 10:.2f}", "dt": f"2020-{1 + i % 12:02d}-{1 + i % 28:02d}",
                 "b": "true" if i % 3 else "false", "s": f"w{i % 11}"}
        lines.append(",".join({**cells, **{c: v for (r, c), v in bad.items() if r == i}}.values()))
        clean.append(",".join({**cells, **{c: "" for (r, c), v in bad.items() if r == i}}.values()))
    path = tmp_path / "bad.csv"
    path.write_text("\n".join(lines) + "\n")
    oracle_path = tmp_path / "clean.csv"
    oracle_path.write_text("\n".join(clean) + "\n")
    con = duckdb.connect()
    con.execute(f"create table t as select * from read_csv('{oracle_path}', header=true, columns={{'id': 'BIGINT', 'x': 'BIGINT', 'd': 'DECIMAL(18,2)', "
                f"'dt': 'DATE', 'b': 'BOOLEAN', 's': 'VARCHAR'}})")
    for sql in ["SELECT count(*), count(x), sum(x), count(d), sum(d), count(dt), min(dt), max(dt), count(b), sum(b) FROM t",
                "SELECT s, count(x), sum(x), count(dt) FROM t GROUP BY s",
                "SELECT id, x, d, dt, b FROM t WHERE id IN (150, 2001, 4000, 4001, 4999) ORDER BY id"]:
        res = run_probe_with(probe, sql, {"t": path}, "scan", None, ["--infer-rows", "100", "--group-rows", "700", "--block-bytes", "4096"])
        assert_equal(got_rows(res), con.execute(sql).fetchall(), "ORDER BY" in sql, sql)
