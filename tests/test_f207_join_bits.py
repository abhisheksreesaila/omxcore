"""F-207: the build side of a two-table inner join is a presence bitmap of its keys when no column of it is read after the join (its filter ran before, its key
is the probe's): `join build side: ... presence bitmap` in the notes. A repeated key, a column that is read, a LEFT join and a join that is not fused (a holistic
aggregate) keep the usual map, and the answer is DuckDB's everywhere: SQL over tables and over edited Frames (the worlds of tests/test_f202_frames.py).
"""
import pytest

from conftest import oracle_connection
from test_mojo_sql import probe, run_probe, got_rows          # noqa: F401  (the probe fixture builds mojo/tools/sql_probe.mojo)
from test_f202_frames import world, run_case                  # noqa: F401  (the world fixture: nat, da, db, f, dbig and the edit scripts)
from mx_oracle import assert_equal

BITS = "presence bitmap"

# (name, sql, tables, a bitmap on the build side)
TWO = [
    ("a filter-only dimension", "SELECT f.a % 5 AS g, count(*), sum(f.x) FROM f JOIN da ON f.a = da.id WHERE da.seg = 2 GROUP BY f.a % 5 ORDER BY g", ["f", "da"], True),
    ("no filter: the keys only", "SELECT f.b % 4 AS g, count(*), max(f.d) FROM f JOIN db ON f.b = db.id GROUP BY f.b % 4 ORDER BY g", ["f", "db"], True),
    ("the fact table's filter beside the dimension", "SELECT f.a % 3 AS g, count(*), avg(f.x) FROM f JOIN da ON f.a = da.id WHERE da.seg < 4 AND f.x > 25 GROUP BY f.a % 3 ORDER BY g",
     ["f", "da"], True),
    ("a dimension column is read", "SELECT da.an, count(*) FROM f JOIN da ON f.a = da.id WHERE da.seg = 2 GROUP BY da.an ORDER BY da.an", ["f", "da"], False),
    ("the dimension's key is read", "SELECT da.id, count(*) AS c FROM f JOIN da ON f.a = da.id WHERE da.seg = 2 GROUP BY da.id ORDER BY c DESC, da.id LIMIT 6", ["f", "da"], False),
    ("repeated build keys", "SELECT f.a % 5 AS g, count(*) FROM f JOIN db ON f.a = db.bk GROUP BY f.a % 5 ORDER BY g", ["f", "db"], False),
    ("a left join", "SELECT f.a % 5 AS g, count(*), count(da.id) FROM f LEFT JOIN da ON f.a = da.id AND da.seg = 2 GROUP BY f.a % 5 ORDER BY g", ["f", "da"], False),
    ("a holistic aggregate (the join is not fused)", "SELECT f.a % 5 AS g, count(DISTINCT f.x) FROM f JOIN da ON f.a = da.id WHERE da.seg = 2 GROUP BY f.a % 5 ORDER BY g",
     ["f", "da"], False),
    ("a row query", "SELECT f.id, f.x FROM f JOIN da ON f.a = da.id WHERE da.seg = 3 AND f.x > 40 ORDER BY f.x DESC, f.id LIMIT 12", ["f", "da"], True),
    # semi and anti joins only ask whether a key is there: a bitmap too, repeated keys included (db.bk repeats)
    ("a semi join (IN)", "SELECT f.a % 5 AS g, count(*), sum(f.x) FROM f WHERE f.a IN (SELECT id FROM da WHERE seg = 2) GROUP BY f.a % 5 ORDER BY g", ["f", "da"], True),
    ("a semi join over repeated keys", "SELECT f.a % 5 AS g, count(*) FROM f WHERE f.a IN (SELECT bk FROM db) GROUP BY f.a % 5 ORDER BY g", ["f", "db"], True),
    ("an anti join (NOT EXISTS)", "SELECT f.a % 5 AS g, count(*), sum(f.x) FROM f WHERE NOT EXISTS (SELECT 1 FROM da WHERE da.id = f.a AND da.seg = 2) GROUP BY f.a % 5 ORDER BY g", ["f", "da"], True),
    ("NOT IN over a list without NULL", "SELECT f.a % 5 AS g, count(*) FROM f WHERE f.a NOT IN (SELECT id FROM da WHERE seg < 3) GROUP BY f.a % 5 ORDER BY g", ["f", "da"], True),
    ("NOT IN over a list with a NULL", "SELECT count(*) FROM f WHERE f.a NOT IN (SELECT ak FROM da)", ["f", "da"], None),
]


@pytest.mark.parametrize("name,sql,tables,bits", TWO, ids=[c[0] for c in TWO])
def test_two_tables(probe, world, name, sql, tables, bits):
    files = {t: world["paths"][t] for t in tables}
    res = run_probe(probe, sql, files, mode="table")
    ref = oracle_connection()
    for t in tables:
        ref.execute(f"create table {t} as select * from read_parquet('{world['paths'][t]}')")
    want = ref.execute(sql).fetchall()
    assert_equal(got_rows(res), want, "ORDER BY" in sql, f"table: {sql}")
    line = [n for n in res["notes"] if n.startswith("join build side:")]
    assert line, res["notes"]
    if bits is not None:
        assert (BITS in line[0]) == bits, line[0]


@pytest.mark.parametrize("frame_table", ["f", "da"])
@pytest.mark.parametrize("name,sql,tables,bits", TWO, ids=[c[0] for c in TWO])
def test_two_tables_over_edited_frames(probe, world, name, sql, tables, bits, frame_table):
    if frame_table not in tables:
        pytest.skip("the case does not read this table")
    run_case(probe, world, sql, tables, frame_table)
