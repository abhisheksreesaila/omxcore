"""F-167: ORDER BY bigger than the sort budget spills sorted runs to disk and merges them; the answers equal DuckDB's.

The real `mx` binary runs each query over a Parquet file (read whole, and streamed), a CSV file (streamed) and writes the sorted rows
into a Parquet file (`--format parquet --out`: merged straight into the file). `--sort-run-rows` is the test override that makes
every run that many rows, so 1,024 rows make many runs, 4,096 a few, and a number bigger than the file one run. Every query has a
unique last key (`id`) so DuckDB's answer is one order; text is ordered the engine's way (ignoring ASCII case, bytes as the
tie-break), so DuckDB sorts `lower(txt), txt`. A second test compares the spilled order with the in-memory order on queries whose
keys tie (the source's order is kept). No temporary file is left behind.
"""
import json
import os
import random
import subprocess
from pathlib import Path

import duckdb
import pytest

from mx_oracle import mx_binary, run_mx, check, typed_rows, assert_equal

ROWS = 9000
WORDS = ["apple", "Apple", "APPLE", "banana", "Banana", "cherry", "Cherry", "", "date", "Date", "elder", "fig", "Fig", "grape", "apple pie", "Zebra", "zebra"]
RUNS = [1024, 4096, 1_000_000]

# (mx SQL, DuckDB SQL): the same ordering in each dialect. TEXT keys are `lower(txt), txt` for DuckDB.
QUERIES = [
    ("SELECT id, k1 FROM t ORDER BY k1 NULLS LAST, id", "SELECT id, k1 FROM t ORDER BY k1 NULLS LAST, id"),
    ("SELECT id, k1, txt FROM t ORDER BY k1 DESC NULLS FIRST, txt NULLS LAST, id DESC",
     "SELECT id, k1, txt FROM t ORDER BY k1 DESC NULLS FIRST, lower(txt) NULLS LAST, txt, id DESC"),
    ("SELECT id, txt FROM t ORDER BY txt DESC NULLS LAST, id", "SELECT id, txt FROM t ORDER BY lower(txt) DESC NULLS LAST, txt DESC, id"),
    ("SELECT id, f FROM t ORDER BY f NULLS LAST, id", "SELECT id, f FROM t ORDER BY f NULLS LAST, id"),
    ("SELECT id, f FROM t ORDER BY f DESC NULLS FIRST, id", "SELECT id, f FROM t ORDER BY f DESC NULLS FIRST, id"),
    ("SELECT id, b, dt, d FROM t ORDER BY b, dt DESC, d, id", "SELECT id, b, dt, d FROM t ORDER BY b, dt DESC, d, id"),
    ("SELECT id, k1 FROM t ORDER BY k1 NULLS FIRST, id LIMIT 1500 OFFSET 700", "SELECT id, k1 FROM t ORDER BY k1 NULLS FIRST, id LIMIT 1500 OFFSET 700"),
    ("SELECT id, k1 + 1 AS z, txt FROM t WHERE id % 3 <> 0 ORDER BY z DESC NULLS LAST, txt NULLS LAST, id",
     "SELECT id, k1 + 1 AS z, txt FROM t WHERE id % 3 <> 0 ORDER BY z DESC NULLS LAST, lower(txt) NULLS LAST, txt, id"),
    ("SELECT id FROM t ORDER BY txt NULLS LAST, f DESC NULLS LAST, id", "SELECT id FROM t ORDER BY lower(txt) NULLS LAST, txt, f DESC NULLS LAST, id"),
]


def make_data(path_parquet, path_csv):
    rng = random.Random(167)
    con = duckdb.connect()
    con.execute("create table t (id bigint, k1 int, txt varchar, f double, d decimal(9,2), dt date, b boolean)")
    specials = ["'nan'::double", "-0.0", "0.0", "'inf'::double", "'-inf'::double", "1.5", "-2.25"]
    rows = []
    for i in range(ROWS):
        k1 = "null" if i % 11 == 0 else str(rng.randint(-6, 6))
        txt = "null" if i % 17 == 0 else "'" + rng.choice(WORDS) + "'"
        if i % 19 == 0:
            f = "null"
        elif i % 3 == 0:
            f = rng.choice(specials)
        else:
            f = str(rng.randint(-24, 24) / 4)
        rows.append(f"({i}, {k1}, {txt}, {f}, {rng.randint(-1000, 1000)}.{rng.randint(10, 99)}, date '2019-01-01' + {rng.randint(0, 39)}, {rng.random() < 0.4})")
    con.execute("insert into t values " + ",".join(rows))
    con.execute(f"copy t to '{path_parquet}' (format parquet, row_group_size 1000)")
    con.execute(f"copy (select id, k1, txt, d from t) to '{path_csv}' (header)")
    return con


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    base = Path(os.path.expanduser("~/.cache/mxframe-test-tmp")) / "f167"
    base.mkdir(parents=True, exist_ok=True)
    d = Path(os.path.expanduser(str(base))) / f"run-{os.getpid()}"
    d.mkdir(exist_ok=True)
    pq, csv = d / "t.parquet", d / "t.csv"
    con = make_data(pq, csv)
    spill = d / "spill"
    spill.mkdir(exist_ok=True)
    yield {"mx": mx_binary(), "con": con, "pq": pq, "csv": csv, "spill": spill, "dir": d}
    for p in list(spill.iterdir()) + list(d.glob("*")):
        if p.is_file():
            p.unlink()


def spilled_runs(res):
    for n in res["notes"]:
        if n.startswith("order: sorted") and "spilled to disk" in n:
            return n
    return None


def no_spill_files(env):
    assert list(env["spill"].iterdir()) == [], list(env["spill"].iterdir())


@pytest.mark.parametrize("stream", ["never", "always"])
@pytest.mark.parametrize("run_rows", RUNS)
def test_parquet_equals_duckdb(env, run_rows, stream):
    for sql, duck in QUERIES:
        extra = ["--stream", stream, "--sort-run-rows", str(run_rows), "--spill-dir", str(env["spill"])]
        res = check(env["mx"], sql, env["con"], files={"t": env["pq"]}, duck_sql=duck, ordered=True, extra=extra)
        note = spilled_runs(res)
        assert note is not None, (sql, res["notes"])
        if "WHERE" not in sql:
            runs = -(-ROWS // run_rows)
            assert f" in {runs} run{'s' if runs > 1 else ''} spilled" in note, (sql, note)
        no_spill_files(env)


@pytest.mark.parametrize("run_rows", RUNS)
def test_csv_scan_equals_duckdb(env, run_rows):
    queries = [
        ("SELECT id, k1 FROM t ORDER BY k1 NULLS LAST, id", None),
        ("SELECT id, txt, d FROM t ORDER BY txt DESC NULLS FIRST, d, id", "SELECT id, txt, d FROM t ORDER BY lower(txt) DESC NULLS FIRST, txt DESC, d, id"),
    ]
    for sql, duck in queries:
        extra = ["--stream", "always", "--sort-run-rows", str(run_rows), "--spill-dir", str(env["spill"])]
        con = duckdb.connect()
        con.execute(f"create view t as select * from read_csv('{env['csv']}', header=true, types={{'id': 'BIGINT', 'k1': 'INTEGER', 'txt': 'VARCHAR', 'd': 'DECIMAL(9,2)'}})")
        res = check(env["mx"], sql, con, files={"t": env["csv"]}, duck_sql=duck, ordered=True, extra=extra)
        assert spilled_runs(res) is not None, res["notes"]
        no_spill_files(env)


@pytest.mark.parametrize("run_rows", [1024, 1_000_000])
def test_merged_straight_into_a_parquet_file(env, run_rows):
    """`--format parquet --out`: the sorted rows go into the file as row groups of the size asked, and the file equals DuckDB's sort."""
    out = env["dir"] / f"sorted-{run_rows}.parquet"
    sql, duck = QUERIES[1]
    run_mx(env["mx"], ["sql", "--format", "parquet", "--out", str(out), "--stream", "always", "--sort-run-rows", str(run_rows), "--row-group-rows", "1000",
                       "--spill-dir", str(env["spill"]), sql], files={"t": env["pq"]})
    assert out.exists()
    con = env["con"]
    got = con.execute(f"select * from read_parquet('{out}')").fetchall()
    want = con.execute(duck).fetchall()
    assert_equal(got, want, True, "sorted file")
    groups = con.execute(f"select count(*) from parquet_metadata('{out}') where column_id = 0").fetchone()[0]
    assert groups == ROWS // 1000 + 1 or groups == ROWS // 1000, groups
    assert set(r[0] for r in con.execute(f"select row_group_num_rows from parquet_metadata('{out}')").fetchall()) <= {1000, ROWS % 1000}
    no_spill_files(env)


def test_streamed_note_says_the_result_is_not_held(env):
    out = env["dir"] / "note.parquet"
    res = run_mx(env["mx"], ["sql", "--format", "parquet", "--out", str(out), "--stream", "always", "--sort-run-rows", "1024", "--spill-dir", str(env["spill"]),
                             "SELECT id, k1 FROM t ORDER BY k1 NULLS LAST, id"], files={"t": env["pq"]})
    assert res["row_count"] == ROWS
    assert any(n.startswith(f"output: {ROWS:,} rows merged straight into") for n in res["notes"]), res["notes"]
    assert [c["name"] for c in res["columns"]] == ["id", "k1"]


@pytest.mark.parametrize("stream", ["never", "always"])
def test_ties_keep_the_source_order(env, stream):
    """Keys that tie: the spilled sort gives exactly the in-memory order (the source's order among equals), for every run size."""
    queries = [
        "SELECT id, k1 FROM t ORDER BY k1",
        "SELECT id, k1, b FROM t ORDER BY b DESC, k1 NULLS FIRST",
        "SELECT id, txt FROM t ORDER BY txt",
        "SELECT id, f FROM t ORDER BY f DESC",
    ]
    for sql in queries:
        plain = run_mx(env["mx"], ["sql", "--max-rows", "0", "--stream", stream, sql], files={"t": env["pq"]})
        assert spilled_runs(plain) is None
        for run_rows in RUNS:
            spilled = run_mx(env["mx"], ["sql", "--max-rows", "0", "--stream", stream, "--sort-run-rows", str(run_rows), "--spill-dir", str(env["spill"]), sql],
                             files={"t": env["pq"]})
            assert spilled_runs(spilled) is not None
            assert spilled["rows"] == plain["rows"], (sql, run_rows)
            no_spill_files(env)


def test_a_spill_folder_that_is_a_file_is_a_readable_error(env):
    bad = env["dir"] / "notafolder"
    bad.write_text("x")
    res = run_mx(env["mx"], ["sql", "--sort-run-rows", "1024", "--spill-dir", str(bad / "x"), "SELECT id FROM t ORDER BY k1, id"], files={"t": env["pq"]}, expect_exit=1)
    assert "temporary file for the sort" in res["error"]["message"]
    assert "spill_dir" in res["error"]["message"]
