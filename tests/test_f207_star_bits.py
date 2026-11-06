"""F-207: a star dimension that contributes no column (it only filters the fact table, or only joins on its key) is a presence bitmap of its keys, not a
hash table: one bit per key of its range, built by marking the keys (a duplicate key falls back to the hash table at once). The notes say `presence bitmap` on
`star dimension ...`; every answer equals DuckDB, over the same worlds as tests/test_f202_frames.py: tables, the fact table as an edited Frame (changes, appended rows,
removed rows, new words), a dimension as an edited Frame. A dimension whose column the query reads keeps the hash table (its rows are gathered).
"""
import pytest

from test_mojo_sql import probe, run_probe, got_rows          # noqa: F401  (the probe fixture builds mojo/tools/sql_probe.mojo)
from test_f202_frames import world, run_case, has            # noqa: F401  (the world fixture: nat, da, db, f, dbig and the edit scripts)
from mx_oracle import assert_equal, duck_kind

BITS = "presence bitmap"

# (name, sql, tables, dimensions that are bitmaps)
CASES = [
    ("two filter-only dimensions", "SELECT f.a % 5 AS g, count(*), sum(f.x) FROM f, da, db WHERE f.a = da.id AND f.b = db.id AND da.seg = 2 AND db.w > 3 GROUP BY f.a % 5 ORDER BY g",
     ["f", "da", "db"], ["da", "db"]),
    ("one dimension's column is read", "SELECT da.an, count(*), sum(f.x) FROM f, da, db WHERE f.a = da.id AND f.b = db.id AND da.seg = 2 AND db.w > 3 GROUP BY da.an ORDER BY da.an",
     ["f", "da", "db"], ["db"]),
    ("no filter on the dimensions: their keys only", "SELECT f.a % 5 AS g, count(*), sum(f.x) FROM f, da, db WHERE f.a = da.id AND f.b = db.id GROUP BY f.a % 5 ORDER BY g",
     ["f", "da", "db"], ["da", "db"]),
    ("a global aggregate", "SELECT count(*), sum(f.x), min(f.d), max(f.x) FROM f, da, db WHERE f.a = da.id AND f.b = db.id AND da.seg <= 3 AND db.w < 5",
     ["f", "da", "db"], ["da", "db"]),
    ("the fact table's filter and a dimension", "SELECT f.a % 5 AS g, count(*), avg(f.x) FROM f, da, db WHERE f.a = da.id AND f.b = db.id AND f.d > date '2021-01-01' AND f.x > 20 AND da.seg = 1 GROUP BY f.a % 5 ORDER BY g",
     ["f", "da", "db"], ["da", "db"]),
    ("the dimension's key is read", "SELECT da.id, count(*) AS c FROM f, da, db WHERE f.a = da.id AND f.b = db.id AND da.seg = 4 GROUP BY da.id ORDER BY c DESC, da.id LIMIT 8",
     ["f", "da", "db"], ["db"]),
    ("a dimension that matches nothing", "SELECT f.a % 5 AS g, count(*) FROM f, da, db WHERE f.a = da.id AND f.b = db.id AND da.seg = 99 GROUP BY f.a % 5 ORDER BY g",
     ["f", "da", "db"], ["db"]),               # (da keeps no row: an empty map; db has no filter: its keys are the bitmap)
]


@pytest.mark.parametrize("name,sql,tables,bitmaps", CASES, ids=[c[0] for c in CASES])
def test_tables(probe, world, name, sql, tables, bitmaps):
    """SQL over tables read whole."""
    files = {t: world["paths"][t] for t in tables}
    res = run_probe(probe, sql, files, mode="table")
    # the Frame oracles hold edited copies: a pristine connection over the same files answers
    from conftest import oracle_connection
    ref = oracle_connection()
    for t in tables:
        ref.execute(f"create table {t} as select * from read_parquet('{world['paths'][t]}')")
    cur = ref.execute(sql)
    want = cur.fetchall()
    for i, desc in enumerate(cur.description):
        dk = duck_kind(desc[1])
        assert res["columns"][i]["kind"] == dk[0], (sql, res["columns"][i], dk)
    assert_equal(got_rows(res), want, "ORDER BY" in sql, f"table: {sql}")
    assert has(res, "a star from f"), res["notes"]
    for dim in ["da", "db"]:
        line = [n for n in res["notes"] if n.startswith(f"star dimension {dim}:")]
        assert line, res["notes"]
        assert (BITS in line[0]) == (dim in bitmaps), (dim, line[0])


@pytest.mark.parametrize("frame_table", ["f", "da"])
@pytest.mark.parametrize("name,sql,tables,bitmaps", CASES, ids=[c[0] for c in CASES])
def test_edited_frames(probe, world, name, sql, tables, bitmaps, frame_table):
    """The fact table (streamed in chunks: shared rows, then changed rows) or a dimension is a Frame with edits; the answer is DuckDB's over the same edits."""
    res = run_case(probe, world, sql, tables, frame_table)
    assert has(res, "a star from f"), res["notes"]
    for dim in bitmaps:
        line = [n for n in res["notes"] if n.startswith(f"star dimension {dim}:")]
        assert line, res["notes"]
        if not (frame_table == dim):          # a Frame dimension is read as its edited rows: appended rows can repeat a key, and then the usual layout answers
            assert BITS in line[0], (dim, line[0])
