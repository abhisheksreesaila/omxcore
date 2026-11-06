"""F-207: ORDER BY ... LIMIT over a filtered scan keeps the best rows in per-worker heaps, and the heap's worst first key is a dynamic filter: a row that cannot
enter the heap is dropped by a cheap pass over that one column, before the WHERE reads its columns and before the other sort keys are loaded
(`top-N front` in the notes). Every answer is DuckDB's through a table read whole, an edited Frame and a Parquet file streamed as row groups, including ties
at the LIMIT boundary (a first key with 40 distinct values, LIMIT 10: the later keys decide), OFFSET, a filter that keeps nothing, a LIMIT larger than the rows
that pass, DESC, NULLS FIRST / LAST. A first key that is nullable, an expression, TEXT or a float takes the plain heap path (no front).
"""
import pytest

from conftest import oracle_connection
from test_mojo_sql import probe, run_probe, got_rows          # noqa: F401  (the probe fixture builds mojo/tools/sql_probe.mojo)
from test_f202_frames import make_edits
from mx_oracle import assert_equal, duck_kind

ROWS = 150000
COLS = [("q", "dec2"), ("k", "int"), ("d", "date"), ("s", "text")]
FRONT = "top-N front"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("f207n")
    base = oracle_connection()
    # q: 40 distinct values, never NULL; qn: the same with NULLs; k: unique-ish, never NULL; f: a float
    base.execute(f"""create table t as select i as id,
        ((i * 7919) % 40)::decimal(10,2) as q,
        (i * 104729) % 1000003 as k,
        date '2020-01-01' + (i % 700)::int as d,
        (['red','green','blue','amber'])[1 + i % 4] as s,
        case when i % 29 = 0 then null else ((i * 7919) % 40)::decimal(10,2) end as qn,
        (((i * 31) % 997) * 0.5)::double as f,
        ((i * 104729) % 1000003) * 4611686018427 as h,
        i % 7 as ln from range({ROWS}) r(i)""")
    path = d / "t.parquet"
    base.execute(f"copy t to '{path}' (format parquet, row_group_size 40000)")
    fpath = d / "tf.parquet"                       # the Frame's file: only the columns an edit script names (the probe appends one value per column)
    base.execute(f"copy (select id, q, k, d, s from t) to '{fpath}' (format parquet, row_group_size 40000)")
    frame = oracle_connection()
    frame.execute(f"create table t as select row_number() over () - 1 as rid, * from read_parquet('{fpath}')")
    edits = d / "t.edits"
    edits.write_text(make_edits(frame, "t", "id", COLS, ROWS, 5, 150, 40))
    return {"path": path, "fpath": fpath, "con": base, "frame": frame, "edits": edits}


def run(probe, world, sql, mode):
    if mode == "frame":
        res = run_probe(probe, sql, {"t": world["fpath"]}, mode="frame", edits=world["edits"])
        cur = world["frame"].execute(sql)
    else:
        res = run_probe(probe, sql, {"t": world["path"]}, mode=mode)
        cur = world["con"].execute(sql)
    want = cur.fetchall()
    for i, desc in enumerate(cur.description):
        dk = duck_kind(desc[1])
        c = res["columns"][i]
        assert c["kind"] == dk[0] and (dk[0] != "DECIMAL" or c["scale"] == dk[1]), f"column {i} of `{sql}`: got {c}, DuckDB has {dk}"
    assert_equal(got_rows(res), want, True, f"{mode}: {sql}")
    return res


def frame_ok(sql):
    """A Frame has only the columns the edit script names."""
    return not ({"ln", "h", "qn", "f"} & set(sql.replace(",", " ").split()))


def has(res, text):
    return any(text in n for n in res["notes"])


MODES = ["table", "frame", "scan"]
W = "where d < date '2021-06-01'"

# (name, sql): the first key is a plain column that is never NULL in the file: the front applies to the file's rows
FRONT_CASES = [
    ("asc, ties at the limit", f"select id, ln, q, k from t {W} order by q, id, ln limit 10"),
    ("desc, ties at the limit", f"select id, ln, q, k from t {W} order by q desc, id, ln limit 10"),
    ("a unique first key", f"select id, q, k from t {W} order by k, id limit 25"),
    ("offset", f"select id, q, k from t {W} order by q desc, k, id limit 7 offset 13"),
    ("the filter keeps few rows", "select id, q, k from t where d = date '2020-03-05' order by q, k, id limit 5"),
    ("the filter keeps nothing", "select id, q, k from t where d < date '2019-01-01' order by q, k, id limit 5"),
    ("the filter keeps fewer rows than the limit", "select id, q, k from t where d = date '2020-03-05' and q = 3 order by q, k, id limit 500"),
    ("a limit past every row", f"select id, q from t {W} order by q, id limit 4000"),
    ("a text key after the first", f"select id, s, q from t {W} order by q, s, id limit 12"),
    ("a filter on the key column too", "select id, q, k from t where q < 12 and d < date '2021-01-01' order by q desc, k, id limit 9"),
    ("a first key past 2^53", f"select id, h, q from t {W} order by h desc, id limit 11"),
    ("every row passes", "select id, q, k from t order by q desc, id desc limit 10"),
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name,sql", FRONT_CASES, ids=[c[0] for c in FRONT_CASES])
def test_the_heap_threshold_filters_before_the_where(probe, world, name, sql, mode):
    if mode == "frame" and not frame_ok(sql):
        pytest.skip("the Frame has no such column")
    res = run(probe, world, sql, mode)
    assert has(res, "per-worker heaps"), res["notes"]
    if mode != "frame":                 # (a Frame's changed rows are a chunk of their own; the shared rows still use the front, asserted below)
        assert has(res, FRONT), res["notes"]


def test_a_frame_uses_the_front_on_its_shared_rows(probe, world):
    res = run(probe, world, f"select id, q, k from t {W} order by q, id, k limit 10", "frame")
    assert has(res, FRONT), res["notes"]


PLAIN_CASES = [
    ("a nullable first key, nulls last", f"select id, qn, k from t {W} order by qn, k, id limit 10"),
    ("a nullable first key, nulls first desc", f"select id, qn, k from t {W} order by qn desc nulls first, k, id limit 10"),
    ("an expression as first key", f"select id, q, k from t {W} order by q + k % 3, id limit 10"),
    ("a text first key", f"select id, s, k from t {W} order by s, k, id limit 10"),
    ("a float first key", f"select id, f, k from t {W} order by f desc, k, id limit 10"),
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name,sql", PLAIN_CASES, ids=[c[0] for c in PLAIN_CASES])
def test_other_first_keys_take_the_plain_heap(probe, world, name, sql, mode):
    if mode == "frame" and not frame_ok(sql):
        pytest.skip("the Frame has no such column")
    res = run(probe, world, sql, mode)
    assert has(res, "per-worker heaps"), res["notes"]
    assert not has(res, FRONT), res["notes"]
