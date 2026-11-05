"""F-151: the `mx` agent command line (build/mx), checked as a user runs it: arguments in, JSON / text / files out, exit codes.

Answers are compared with DuckDB through tests/mx_oracle.py (typed values, exact decimals); the rest checks the shape an agent
reads: the JSON keys, errors with line / column / hint and exit 1, usage errors with exit 2, output formats, `schema`, `explain`.
"""
import csv
import datetime
import io
import json
import re
import struct
import subprocess
from decimal import Decimal

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mx_oracle import check, mx_binary, run_mx, typed_rows


@pytest.fixture(scope="session")
def mx():
    return mx_binary()


@pytest.fixture(scope="session")
def con():
    return duckdb.connect()


@pytest.fixture(scope="session")
def files(tmp_path_factory, con):
    d = tmp_path_factory.mktemp("mxdata")
    con.execute("""create table src as select i as id, (i % 7)::bigint as k,
        case when i % 11 = 0 then null else ((i * 37) % 100000)::decimal(12,2) end as d,
        date '2020-01-01' + (i % 400)::int as dt,
        case i % 5 when 0 then 'a b' when 1 then 'Q"uo\\te' when 2 then 'new' || chr(10) || 'line' when 3 then 'ünï' else null end as s,
        i % 3 = 0 as b
        from range(0, 5000) r(i)""")
    pq_path = d / "t.parquet"
    pq.write_table(con.execute("select * from src").to_arrow_table(), pq_path, row_group_size=700)
    csv_path = d / "c.csv"
    con.execute(f"copy (select id, k, s from src where s is not null and s not like '%' || chr(10) || '%') to '{csv_path}' (header)")
    return {"parquet": pq_path, "csv": csv_path, "dir": d}


QUERIES = [
    "SELECT count(*), sum(d), min(dt), max(dt), avg(d), count(s) FROM t",
    "SELECT k, count(*), sum(d), avg(d) FROM t GROUP BY k ORDER BY k",
    "SELECT s, count(*) c FROM t GROUP BY s ORDER BY c DESC, s",
    "SELECT b, k, max(dt), count(d) FROM t GROUP BY b, k",
    "SELECT k, sum(d) s FROM t GROUP BY k HAVING sum(d) > 1000000 ORDER BY k",
    "SELECT k % 3 m, count(*) FROM t WHERE d IS NOT NULL AND s <> 'ünï' GROUP BY k % 3",
    "SELECT count(*), avg(d) / 0, sum(id) // 0 FROM t",
    "SELECT k, quantile_disc(dt, 0.5), count(DISTINCT s) FROM t GROUP BY k",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_a_query_equals_duckdb_by_table_name_and_by_file_name(mx, con, files, sql):
    check(mx, sql, con, files={"t": files["parquet"]}, duck_sql=sql.replace("FROM t", f"FROM read_parquet('{files['parquet']}')"))
    quoted = sql.replace("FROM t", f"FROM '{files['parquet']}'")
    check(mx, quoted, con, duck_sql=quoted)


@pytest.mark.parametrize("sql", QUERIES[:4])
def test_streaming_and_reading_whole_give_the_same_answer(mx, con, files, sql):
    want = sql.replace("FROM t", f"FROM read_parquet('{files['parquet']}')")
    streamed = check(mx, sql, con, files={"t": files["parquet"]}, duck_sql=want, extra=["--stream", "always"])
    whole = check(mx, sql, con, files={"t": files["parquet"]}, duck_sql=want, extra=["--stream", "never"])
    assert any("streamed a row group at a time (8 row groups, 5000 rows)" in n for n in streamed["notes"]), streamed["notes"]
    assert any("read into memory (5000 rows" in n for n in whole["notes"]), whole["notes"]


def test_a_parquet_file_over_a_quarter_of_memory_streams(mx, files):
    size = files["parquet"].stat().st_size
    small = run_mx(mx, ["sql", "SELECT count(*) FROM t", "--memory", str(size * 3)], {"t": files["parquet"]})
    big = run_mx(mx, ["sql", "SELECT count(*) FROM t", "--memory", str(size * 5)], {"t": files["parquet"]})
    assert small["rows"] == [[5000]] and any("streamed" in n for n in small["notes"]), small["notes"]
    assert big["rows"] == [[5000]] and any("read into memory" in n for n in big["notes"]), big["notes"]


def test_a_csv_file_is_read_by_name(mx, con, files):
    path = files["csv"]
    sql = f"SELECT k, count(*), min(id), max(s) FROM '{path}' GROUP BY k"
    check(mx, sql, con, duck_sql=sql)


def test_the_json_shape(mx, files):
    out = run_mx(mx, ["sql", "SELECT k, sum(d) total, min(dt) first_day FROM t GROUP BY k ORDER BY k LIMIT 2"], {"t": files["parquet"]})
    assert list(out) == ["columns", "rows", "row_count", "truncated", "ms", "device", "notes"]
    assert out["columns"] == [{"name": "k", "kind": "INT", "scale": 0}, {"name": "total", "kind": "DECIMAL", "scale": 2},
                              {"name": "first_day", "kind": "DATE", "scale": 0}]
    assert out["row_count"] == 2 and out["truncated"] is False and out["device"] == "cpu"
    assert isinstance(out["ms"], Decimal) and all(isinstance(n, str) for n in out["notes"])
    k0 = out["rows"][0]
    assert k0[0] == 0 and isinstance(k0[1], Decimal) and k0[2] == "2020-01-01"


def test_decimals_are_exact_number_text_and_text_is_escaped(mx, files):
    p = subprocess.run([str(mx), "sql", "SELECT s, sum(d) FROM t GROUP BY s ORDER BY s", "-t", f"t={files['parquet']}"], capture_output=True)
    raw = p.stdout.decode("utf-8")
    assert '"Q\\"uo\\\\te"' in raw and '"new\\nline"' in raw and '"ünï"' in raw
    assert json.loads(raw)["rows"][-1][0] is None                                    # NULL text sorts last
    assert re.search(r",\d+\.\d\d\]", raw) and not re.search(r',"\d+\.\d\d"', raw)      # a number, never a quoted string
    for row in json.loads(raw, parse_float=Decimal)["rows"]:
        assert row[1] is None or isinstance(row[1], Decimal)


def test_device_gpu_on_a_scan_runs_on_the_cpu_and_the_notes_say_why(mx, files):
    out = run_mx(mx, ["sql", "SELECT count(*) FROM t", "--device", "gpu", "--stream", "always"], {"t": files["parquet"]})
    assert out["rows"] == [[5000]] and out["device"] == "cpu"
    assert any(n.startswith("device: CPU (GPU asked for; ") for n in out["notes"]), out["notes"]


def test_max_rows_truncates_and_says_so(mx, files):
    out = run_mx(mx, ["sql", "SELECT id, count(*) FROM t GROUP BY id ORDER BY id", "--max-rows", "3"], {"t": files["parquet"]})
    assert out["rows"] == [[0, 1], [1, 1], [2, 1]] and out["row_count"] == 5000 and out["truncated"] is True
    default = run_mx(mx, ["sql", "SELECT id, count(*) FROM t GROUP BY id ORDER BY id"], {"t": files["parquet"]})
    assert len(default["rows"]) == 1000 and default["row_count"] == 5000 and default["truncated"] is True
    everything = run_mx(mx, ["sql", "SELECT id, count(*) FROM t GROUP BY id", "--max-rows", "0"], {"t": files["parquet"]})
    assert len(everything["rows"]) == 5000 and everything["truncated"] is False


def test_jsonl_and_csv_output(mx, files):
    sql = "SELECT k, sum(d) total, min(dt) day, min(s) s FROM t GROUP BY k ORDER BY k LIMIT 2"
    base = [str(mx), "sql", sql, "-t", f"t={files['parquet']}"]
    lines = subprocess.run(base + ["--format", "jsonl"], capture_output=True, check=True).stdout.decode().splitlines()
    first = json.loads(lines[0], parse_float=Decimal)
    assert len(lines) == 2 and list(first) == ["k", "total", "day", "s"] and first["day"] == "2020-01-01" and isinstance(first["total"], Decimal)
    text = subprocess.run(base + ["--format", "csv"], capture_output=True, check=True).stdout.decode()
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["k", "total", "day", "s"] and len(rows) == 3 and rows[1][2] == "2020-01-01"


def test_parquet_output_writes_every_row_and_reads_back(mx, con, files):
    out = files["dir"] / "out.parquet"
    summary = run_mx(mx, ["sql", "SELECT k, sum(d) total, min(dt) day FROM t GROUP BY k", "--format", "parquet", "--out", str(out)], {"t": files["parquet"]})
    assert summary["row_count"] == 7 and summary["out"] == str(out)
    got = con.execute(f"select k, total, day from read_parquet('{out}') order by k").fetchall()
    want = con.execute(f"select k, sum(d), min(dt) from read_parquet('{files['parquet']}') group by k order by k").fetchall()
    assert got == want
    assert len(want) == 7


def test_json_out_writes_the_file_and_prints_a_summary(mx, files):
    out = files["dir"] / "out.json"
    summary = run_mx(mx, ["sql", "SELECT k, count(*) FROM t GROUP BY k ORDER BY k", "--out", str(out)], {"t": files["parquet"]})
    assert summary["row_count"] == 7 and summary["out"] == str(out)
    assert len(json.loads(out.read_text())["rows"]) == 7


def test_float_specials_are_strings_in_json(mx, files):
    out = run_mx(mx, ["sql", "SELECT avg(d) / 0, 0 * (avg(d) / 0), -avg(d) / 0 FROM t"], {"t": files["parquet"]})
    assert out["rows"] == [["Infinity", "NaN", "-Infinity"]], out["rows"]
    f = typed_rows(out)[0]
    assert f[0] == float("inf") and f[1] != f[1] and f[2] == float("-inf")


# ── errors ───────────────────────────────────────────────────────────────────────────────────────────

def test_a_query_error_is_json_with_position_and_exit_1(mx, files):
    out = run_mx(mx, ["sql", "SELECT k, sum(dd) FROM t GROUP BY k"], {"t": files["parquet"]}, expect_exit=1)
    e = out["error"]
    assert list(e) == ["message", "line", "column", "hint"]
    assert (e["line"], e["column"]) == (1, 15) and "dd" in e["message"] and "column" in e["message"]
    assert "did you mean" not in e["message"]
    multi = run_mx(mx, ["sql", "SELECT k,\n  sum(d)\nFROM t\nGROUP BY kk"], {"t": files["parquet"]}, expect_exit=1)
    assert multi["error"]["line"] == 4


def test_a_near_miss_gets_a_hint(mx, files):
    out = run_mx(mx, ["sql", "SELECT sum(dt2) FROM t"], {"t": files["parquet"]}, expect_exit=1)
    assert out["error"]["hint"] and "dt" in out["error"]["hint"]


def test_a_missing_table_names_what_is_open(mx, files):
    out = run_mx(mx, ["sql", "SELECT count(*) FROM orders"], {"t": files["parquet"]}, expect_exit=1)
    assert "orders" in out["error"]["message"] and "t" in out["error"]["hint"]
    none = run_mx(mx, ["sql", "SELECT count(*) FROM orders"], expect_exit=1)
    assert none["error"]["hint"] and "-t" in none["error"]["hint"]


def test_a_missing_file_is_a_query_error(mx, files):
    out = run_mx(mx, ["sql", "SELECT count(*) FROM '/nonexistent/x.parquet'"], expect_exit=1)
    assert "/nonexistent/x.parquet" in out["error"]["message"] and "no such file" in out["error"]["message"]


@pytest.mark.parametrize("args", [
    ["sql"],
    ["sql", "SELECT 1", "SELECT 2"],
    ["sql", "SELECT count(*) FROM t", "--format", "xml"],
    ["sql", "SELECT count(*) FROM t", "--max-rows", "many"],
    ["sql", "SELECT count(*) FROM t", "--bogus"],
    ["sql", "SELECT count(*) FROM t", "--device", "fpga"],
    ["sql", "SELECT count(*) FROM t", "--format", "parquet"],
    ["frobnicate"],
    [],
])
def test_a_usage_error_is_json_with_the_usage_and_exit_2(mx, files, args):
    out = run_mx(mx, args, {"t": files["parquet"]}, expect_exit=2)
    assert out["error"]["message"] and out["error"]["line"] is None and "mx sql" in out["error"]["hint"]


def test_a_long_or_chain_hits_the_depth_limit_and_in_is_the_way(mx, con, files):
    chain = " OR ".join(f"k = {i}" for i in range(300))
    out = run_mx(mx, ["sql", f"SELECT count(*) FROM t WHERE {chain}"], {"t": files["parquet"]}, expect_exit=1)
    assert "IN" in out["error"]["hint"], out["error"]
    ints = ", ".join(str(i) for i in range(40))
    check(mx, f"SELECT count(*) FROM t WHERE k IN ({ints})", con, files={"t": files["parquet"]},
          duck_sql=f"SELECT count(*) FROM read_parquet('{files['parquet']}') WHERE k IN ({ints})")
    words = ", ".join(f"'w{i}'" for i in range(300)) + ", 'ünï'"
    check(mx, f"SELECT count(*) FROM t WHERE s IN ({words})", con, files={"t": files["parquet"]},
          duck_sql=f"SELECT count(*) FROM read_parquet('{files['parquet']}') WHERE s IN ({words})")


# ── schema and explain ───────────────────────────────────────────────────────────────────────────────

def test_schema_of_a_parquet_file(mx, files):
    out = run_mx(mx, ["schema", str(files["parquet"]), "--rows", "2"])
    assert out["format"] == "parquet" and out["rows"] == 5000 and out["row_groups"] == 8
    cols = {c["name"]: c for c in out["columns"]}
    assert list(cols) == ["id", "k", "d", "dt", "s", "b"]
    assert cols["d"]["kind"] == "DECIMAL" and cols["d"]["scale"] == 2 and cols["d"]["nulls"] == 455
    assert cols["s"]["kind"] == "TEXT" and cols["s"]["nulls"] == 1000 and cols["s"]["distinct_words"] == 4
    assert cols["id"]["first_values"] == [0, 1] and cols["id"]["nulls"] == 0 and cols["dt"]["first_values"][0] == "2020-01-01"
    assert cols["b"]["kind"] == "BOOL"


def test_schema_of_a_csv_file(mx, files):
    out = run_mx(mx, ["schema", str(files["csv"])])
    assert out["format"] == "csv" and [c["name"] for c in out["columns"]] == ["id", "k", "s"]
    assert [c["kind"] for c in out["columns"]] == ["INT", "INT", "TEXT"]


def test_explain_prints_both_plans_and_analyze_runs_it(mx, files):
    base = [str(mx), "explain", "SELECT k, sum(d) FROM t GROUP BY k", "-t", f"t={files['parquet']}"]
    text = subprocess.run(base, capture_output=True, check=True).stdout.decode()
    assert text.startswith("Logical plan\n") and "Physical plan" in text and "group: dense" in text and "result:" not in text
    ran = subprocess.run(base[:2] + ["--analyze"] + base[2:], capture_output=True, check=True).stdout.decode()
    assert "result: 7 rows in" in ran and "Actual" in ran and "ran:" in ran
    js = run_mx(mx, ["explain", "--format", "json", "SELECT k, sum(d) FROM t GROUP BY k"], {"t": files["parquet"]})
    assert js["analyzed"] is False and js["plan"][0] == "Logical plan" and any(l.strip().startswith("group: dense") for l in js["plan"])


def test_explain_of_a_bad_query_is_an_error(mx, files):
    out = run_mx(mx, ["explain", "SELECT nope FROM t GROUP BY nope"], {"t": files["parquet"]}, expect_exit=1)
    assert out["error"]["line"] == 1


# ── a footer that lies ───────────────────────────────────────────────────────────────────────────────

def test_a_lying_footer_runs_again_and_the_note_says_statistics_not_believed(mx, con, tmp_path):
    n = 2000
    g = [i % 10 for i in range(n)]
    g[1500] = 1000
    honest = tmp_path / "honest.parquet"
    pq.write_table(pa.table({"g": pa.array(g, pa.int64())}), honest, row_group_size=500, write_page_index=False)
    raw = bytearray(honest.read_bytes())
    flen = struct.unpack("<I", raw[-8:-4])[0]
    start = len(raw) - 8 - flen
    foot = bytes(raw[start:-8]).replace(struct.pack("<q", 1000), struct.pack("<q", 9))
    lie = tmp_path / "lie.parquet"
    lie.write_bytes(bytes(raw[:start]) + foot + bytes(raw[-8:]))
    sql = "SELECT g, count(*) FROM t GROUP BY g"
    out = check(mx, sql, con, files={"t": lie}, duck_sql=sql.replace("FROM t", f"FROM read_parquet('{honest}')"), extra=["--stream", "always"])
    notes = "\n".join(out["notes"])
    assert "statistics: " in notes and "ran again without them" in notes
    assert "the range of `g` is unknown (statistics not believed)" in notes or "group: hashed" in notes


def test_cold_start_is_fast(mx, files):
    import time
    best = min(_time(mx, files) for _ in range(5))
    assert best < 0.25, f"a cold `mx sql` took {best * 1000:.0f} ms"


def _time(mx, files):
    import time
    t = time.perf_counter()
    subprocess.run([str(mx), "sql", "SELECT count(*) FROM t", "-t", f"t={files['parquet']}"], capture_output=True, check=True)
    return time.perf_counter() - t


# ── F-151 review: text encodings, CSV output, --out, flags ───────────────────────────────────────────

def test_a_latin1_csv_is_refused_as_not_utf8_never_passed_through(mx, tmp_path):
    p = tmp_path / "latin.csv"
    p.write_bytes("id,city\n1,caf\xe9\n2,plain\n".encode("latin-1"))
    for args in (["sql", f"SELECT city, count(*) FROM '{p}' GROUP BY city"], ["schema", str(p)]):
        out = run_mx(mx, args, expect_exit=1)
        assert "not valid UTF-8" in out["error"]["message"] and "byte 13" in out["error"]["message"] and "line 2" in out["error"]["message"]
        assert "UTF-8" in out["error"]["hint"]


def test_a_utf16_csv_is_refused_by_its_byte_order_mark(mx, tmp_path):
    p = tmp_path / "wide.csv"
    p.write_bytes("id,city\n1,x\n".encode("utf-16"))
    out = run_mx(mx, ["schema", str(p)], expect_exit=1)
    assert "UTF-16" in out["error"]["message"] and "UTF-8" in out["error"]["hint"]


def test_a_non_utf8_column_name_is_refused(mx, tmp_path):
    p = tmp_path / "head.csv"
    p.write_bytes(b"id,caf\xe9\n1,2\n")
    out = run_mx(mx, ["schema", str(p)], expect_exit=1)
    assert "not valid UTF-8" in out["error"]["message"] and "line 1" in out["error"]["message"]


def test_a_utf8_csv_with_a_byte_order_mark_and_accents_reads(mx, tmp_path):
    p = tmp_path / "ok.csv"
    p.write_bytes(b"\xef\xbb\xbfid,city\n1,caf\xc3\xa9\n2,\xe2\x82\xac\n")
    out = run_mx(mx, ["sql", f"SELECT city, count(*) c FROM '{p}' GROUP BY city ORDER BY city"])
    assert [r[0] for r in out["rows"]] == ["café", "€"]


def test_bad_utf8_in_a_parquet_text_value_is_refused(mx, tmp_path):
    p = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"w": pa.array(["cafe", "ok"], pa.string()), "n": [1, 2]}), p, compression="none")
    p.write_bytes(p.read_bytes().replace(b"cafe", b"caf\xe9"))                         # not UTF-8, as a careless writer leaves it
    for args in (["sql", f"SELECT w, count(*) FROM '{p}' GROUP BY w"], ["schema", str(p)]):
        out = run_mx(mx, args, expect_exit=1)
        assert "not valid UTF-8" in out["error"]["message"] and "`w`" in out["error"]["message"] and "UTF-8" in out["error"]["hint"]


HOSTILE = ["", " ", "a,b", 'say "hi"', "line\nbreak", "cr\rhere", "crlf\r\nx", "ünï €", " lead", "trail ", "NULL", "null", "\\N", "tab\there", ",", '"', "plain"]


@pytest.fixture(scope="session")
def hostile(tmp_path_factory):
    d = tmp_path_factory.mktemp("hostile")
    ids = list(range(len(HOSTILE) + 1))
    p = d / "h.parquet"
    pq.write_table(pa.table({"id": ids, "w": pa.array(HOSTILE + [None], pa.string())}), p)
    return p


@pytest.mark.parametrize("to_file", [False, True])
def test_csv_output_is_byte_for_byte_what_duckdb_writes(mx, con, hostile, tmp_path, to_file):
    sql = f"SELECT id, w, count(*) c FROM '{hostile}' GROUP BY id, w ORDER BY id"
    want_file = tmp_path / "duck.csv"
    con.execute(f"COPY ({sql}) TO '{want_file}' (FORMAT csv, HEADER)")
    want = want_file.read_bytes()
    if to_file:
        out = tmp_path / "mx.csv"
        run_mx(mx, ["sql", sql, "--format", "csv", "--out", str(out)])
        got = out.read_bytes()
    else:
        got = subprocess.run([str(mx), "sql", sql, "--format", "csv", "--max-rows", "0"], capture_output=True, check=True).stdout
    assert got.replace(b"\r\n", b"\n") == want.replace(b"\r\n", b"\n"), (got, want)


def test_csv_output_tells_null_from_the_empty_string(mx, hostile):
    got = subprocess.run([str(mx), "sql", f"SELECT id, w, count(*) c FROM '{hostile}' GROUP BY id, w ORDER BY id", "--format", "csv", "--max-rows", "0"],
                         capture_output=True, check=True).stdout.decode()
    lines = got.split("\n")
    assert lines[1] == '0,"",1'                                                        # the empty string is two quotes
    assert lines[-2] == f"{len(HOSTILE)},,1"                                           # NULL is an empty cell


def test_parquet_output_writes_a_float_as_double(mx, con, files, tmp_path):
    sql = "SELECT k, avg(d) a, count(*) c FROM t GROUP BY k ORDER BY k"
    out = tmp_path / "f.parquet"
    run_mx(mx, ["sql", sql, "--format", "parquet", "--out", str(out)], {"t": files["parquet"]})
    t = pq.read_table(out)
    assert str(t.schema.field("a").type) == "double"
    want = con.execute(sql.replace("FROM t", f"FROM read_parquet('{files['parquet']}')")).fetchall()
    assert [(r["k"], r["a"], r["c"]) for r in t.to_pylist()] == want


def test_a_directory_named_like_a_csv_is_an_error(mx, tmp_path):
    d = tmp_path / "x.csv"
    d.mkdir()
    out = run_mx(mx, ["sql", f"SELECT count(*) FROM '{d}'"], expect_exit=1)
    assert "not a regular file" in out["error"]["message"]
    out = run_mx(mx, ["schema", str(d)], expect_exit=1)
    assert "not a regular file" in out["error"]["message"]


def test_an_unknown_stream_or_group_value_is_a_usage_error(mx, files):
    for flag, val in (("--stream", "sometimes"), ("--group", "maybe")):
        out = run_mx(mx, ["sql", "SELECT count(*) FROM t", flag, val], {"t": files["parquet"]}, expect_exit=2)
        assert flag in out["error"]["message"] and val in out["error"]["message"]


def test_explain_accepts_an_explain_prefix_and_a_file_name(mx, files):
    p = files["parquet"]
    for analyze, text in ((True, f"EXPLAIN ANALYZE SELECT k, count(*) FROM '{p}' GROUP BY k"),
                          (False, f"EXPLAIN SELECT k, count(*) FROM '{p}' GROUP BY k"),
                          (True, f"explain select k, count(*) FROM '{p}' GROUP BY k")):
        r = subprocess.run([str(mx), "explain"] + (["--analyze"] if analyze else []) + [text], capture_output=True)
        assert r.returncode == 0, r.stdout
        assert b"Logical plan" in r.stdout and b"group: dense" in r.stdout
        assert (b"result: 7 rows in" in r.stdout) == analyze


def test_sql_with_an_explain_prefix_over_a_file_name_runs(mx, files):
    out = run_mx(mx, ["sql", f"EXPLAIN SELECT k, count(*) FROM '{files['parquet']}' GROUP BY k"])
    assert out["rows"][0] == ["Logical plan"]


@pytest.mark.parametrize("fmt", ["json", "csv", "jsonl", "parquet"])
def test_out_to_an_unwritable_path_is_a_clean_error_and_creates_no_directories(mx, files, tmp_path, fmt):
    target = tmp_path / "missing" / "deeper" / "o.out"
    out = run_mx(mx, ["sql", "SELECT k, count(*) FROM t GROUP BY k", "--format", fmt, "--out", str(target)], {"t": files["parquet"]}, expect_exit=1)
    assert str(target) in out["error"]["message"]
    assert not (tmp_path / "missing").exists()
    out = run_mx(mx, ["sql", "SELECT k, count(*) FROM t GROUP BY k", "--format", fmt, "--out", "/proc/nope/o.out"], {"t": files["parquet"]}, expect_exit=1)
    assert "/proc/nope/o.out" in out["error"]["message"]


@pytest.mark.parametrize("fmt", ["json", "csv", "jsonl"])
def test_a_big_out_file_is_complete(mx, files, tmp_path, fmt):
    target = tmp_path / f"big.{fmt}"
    res = run_mx(mx, ["sql", "SELECT id, s, count(*) c FROM t GROUP BY id, s", "--format", fmt, "--out", str(target)], {"t": files["parquet"]})
    assert res["row_count"] == 5000
    text = target.read_text()
    if fmt == "json":
        assert len(json.loads(text)["rows"]) == 5000
    elif fmt == "jsonl":
        assert len([json.loads(l) for l in text.splitlines()]) == 5000
    else:
        assert len(list(csv.reader(io.StringIO(text, newline="")))) == 5001


def test_a_parquet_double_column_opens_and_equals_duckdb(mx, con, tmp_path):
    """F-157: a DOUBLE column (and a 32-bit FLOAT one) opens, even when the query never reads it, and answers equal DuckDB's."""
    p = tmp_path / "dbl.parquet"
    price = [1.5, 2.25, None, float("nan"), -0.0, 0.0, float("inf"), 4.0, 0.5, 2.25]
    pq.write_table(pa.table({"id": list(range(10)), "g": [i % 3 for i in range(10)], "price": pa.array(price, pa.float64()),
                             "w": pa.array([0.25 * i for i in range(10)], pa.float32())}), p)
    assert check(mx, f"SELECT count(*) FROM '{p}' WHERE id > 2", con)                      # the DOUBLE column is never read
    for sql, ordered in [(f"SELECT count(price), min(price), max(price), count(DISTINCT price) FROM '{p}'", False),
                         (f"SELECT g, sum(w), max(price) FROM '{p}' GROUP BY g ORDER BY g", True),
                         (f"SELECT price, count(*) FROM '{p}' GROUP BY price ORDER BY price", True),
                         (f"SELECT count(*) FROM '{p}' WHERE price > 2 AND price <> 4", False),
                         (f"SELECT median(price), sum(w * 2) FROM '{p}'", False)]:
        check(mx, sql, con, ordered=ordered)
    out = run_mx(mx, ["schema", str(p)])
    assert [c["kind"] for c in out["columns"]] == ["INT", "INT", "FLOAT", "FLOAT"] if "columns" in out else True


def test_a_float_is_written_as_the_shortest_text_that_reads_back(mx, tmp_path):
    """F-157: 0.1 prints as 0.1 and every double reads back to its bits, in JSON and in CSV (Mojo's own formatter alone fails 1 in 1,000)."""
    import random
    r = random.Random(11)
    vals = [2.0 ** e for e in range(-1074, 1024)] + [-(2.0 ** e) for e in range(-1074, 1024, 7)]      # every power of two: the gap below is half the gap above
    vals += [2.0 ** 49 + 0.25, 2.0 ** 50 + 0.5, 4503599627370497.5, 0.1, 0.2, 0.30000000000000004, 1e-7, 5e-324, 1.7976931348623157e308, 123456789.123456789, 1e16, 1e15, 8.695637939342639e17, -3.3854499886543212e16,
            100.0, 1e22, 2.5e-5, 0.0001]
    vals += [struct.unpack("<d", struct.pack("<Q", r.getrandbits(64)))[0] for _ in range(4000)]
    vals += [round(r.uniform(-1e6, 1e6), r.randint(0, 6)) for _ in range(5000)]
    vals += [r.randint(2**48, 2**52) + r.choice([0.25, 0.5, 0.75, 0.125]) for _ in range(5000)]      # exact ties at 16-17 digits
    vals = sorted({v for v in vals if v == v and abs(v) != float("inf") and v != 0.0})
    p = tmp_path / "f.parquet"
    pq.write_table(pa.table({"x": pa.array(vals, pa.float64())}), p)
    target = tmp_path / "x.csv"
    run_mx(mx, ["sql", f"SELECT x, count(*) FROM '{p}' GROUP BY x ORDER BY x", "--format", "csv", "--out", str(target)])
    lines = [l.split(",")[0] for l in target.read_text().splitlines()[1:]]
    assert len(lines) == len(vals)
    for v, t in zip(vals, lines):
        assert float(t) == v and struct.pack("<d", float(t)) == struct.pack("<d", v), (v, t)
        digits = lambda x: len(x.lower().split("e")[0].replace("-", "").replace(".", "").strip("0"))
        assert digits(t) <= digits(repr(v)), (v, t)       # as short as Python's repr (which may pick another digit of the same length)
        assert t == repr(v), (v, t)                       # round half to even on exact ties: Python's repr, digit for digit


def test_a_csv_of_money_with_many_places_answers_like_duckdb(mx, con, tmp_path):
    """F-157: more than 4 decimal places is FLOAT (DuckDB's DOUBLE): `sum(rate * qty)` answers, where an exact DECIMAL(6) would pass int64."""
    import random
    r = random.Random(5)
    p = tmp_path / "fx.csv"
    p.write_text("rate,qty,price9\n" + "".join(f"{r.uniform(1, 999):.6f},{r.randint(100000, 1000000)},{r.uniform(1, 3000):.9f}\n" for _ in range(100000)))
    sql = f"SELECT sum(rate * qty), sum(rate), avg(price9), count(*), max(qty) FROM '{p}'"
    out = run_mx(mx, ["sql", sql])
    assert [c["kind"] for c in out["columns"]] == ["FLOAT", "FLOAT", "FLOAT", "INT", "INT"]
    want = con.execute(sql).fetchone()
    for got, w in zip(typed_rows(out)[0], want):
        assert abs(float(got) - float(w)) <= 1e-9 * abs(float(w)), (got, w)
