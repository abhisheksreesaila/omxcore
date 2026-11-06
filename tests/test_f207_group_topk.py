"""F-207: GROUP BY keys that the WHERE pins to one value, and ORDER BY an aggregate LIMIT over groups with TEXT parts.

A key fixed by `WHERE col = literal` takes one slot of the dense layout (the planner folds it: `(fixed by the WHERE)` in the group note), so a 3-key
group-by whose TEXT keys are constants is as small as its one free key. The answer is DuckDB's through the three SQL entry points: a table read whole, an
edited Frame (changed cells, new words, NULLs, appended and removed rows) and a Parquet file streamed as row groups. A literal the data does not hold, an
OR, a NOT or a NULL never fold. A hashed group-by (a wide key beside TEXT keys) still finds its top groups on the partial slots, ties at the LIMIT included.
"""
import random

import pytest

from conftest import oracle_connection
from test_mojo_sql import probe, run_probe, got_rows          # noqa: F401  (the probe fixture builds mojo/tools/sql_probe.mojo)
from test_f202_frames import make_edits, WORDS
from mx_oracle import assert_equal, duck_kind

ROWS = 120000
COLS = [("cust", "int"), ("prio", "text"), ("stat", "text"), ("price", "dec2"), ("d", "date")]
FIXED = "(fixed by the WHERE)"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("f207")
    base = oracle_connection()
    base.execute(f"""create table o as select i as id,
        case when i % 53 = 0 then null else (i * 7919) % 3000 end as cust,
        case when i % 61 = 0 then null else (['red','green','blue','amber','zzz-new'])[1 + (i % 5)] end as prio,
        (['green','blue','amber'])[1 + (i % 3)] as stat,
        case when i % 47 = 0 then null else ((i % 9000) * 0.25)::decimal(10,2) end as price,
        date '2020-01-01' + (i % 300)::int as d,
        i % 7 as seg,
        (i % 40000) * 1000003 as big from range({ROWS}) r(i)""")
    path = d / "o.parquet"
    base.execute(f"copy o to '{path}' (format parquet, row_group_size 30000)")
    frame = oracle_connection()
    frame.execute(f"create table o as select row_number() over () - 1 as rid, * from read_parquet('{path}')")
    edits = d / "o.edits"
    edits.write_text(make_edits(frame, "o", "id", COLS, ROWS, 3, 120, 3100))
    return {"path": path, "con": base, "frame": frame, "edits": edits}


def run(probe, world, sql, mode):
    if mode == "frame":
        res = run_probe(probe, sql, {"o": world["path"]}, mode="frame", edits=world["edits"])
        cur = world["frame"].execute(sql)
    else:
        res = run_probe(probe, sql, {"o": world["path"]}, mode=mode)
        cur = world["con"].execute(sql)
    want = cur.fetchall()
    for i, desc in enumerate(cur.description):
        dk = duck_kind(desc[1])
        c = res["columns"][i]
        assert c["kind"] == dk[0] and (dk[0] != "DECIMAL" or c["scale"] == dk[1]), f"column {i} of `{sql}`: got {c}, DuckDB has {dk}"
    assert_equal(got_rows(res), want, "ORDER BY" in sql.upper(), f"{mode}: {sql}")
    return res


def note(res, text):
    hits = [n for n in res["notes"] if text in n]
    assert hits, (text, res["notes"])
    return hits[0]


MODES = ["table", "frame", "scan"]

# (name, sql): the keys a WHERE equality pins are folded
FOLDED = [
    ("two text keys fixed, top 10", "select cust, prio, stat, sum(price) as a0, min(price) as a1, count(*) as c from o where prio = 'red' and price >= 400 "
     "group by cust, prio, stat order by a0 desc, cust, prio, stat limit 10"),
    ("a text key and an int key fixed", "select seg, prio, cust, count(*) as c, sum(price) as s from o where seg = 3 and prio = 'blue' group by seg, prio, cust order by c desc, cust limit 7"),
    ("a date key fixed", "select d, cust, count(*) as c, max(price) as m from o where d = date '2020-03-05' group by d, cust order by c desc, cust limit 5"),
    ("every key fixed", "select prio, seg, count(*) as c, sum(price) as s from o where prio = 'green' and seg = 2 group by prio, seg"),
    ("the equality beside a range", "select prio, cust, sum(price) as s from o where prio = 'amber' and cust < 500 and d > date '2020-02-01' group by prio, cust order by s desc, cust limit 10"),
    ("the equality written the other way round", "select prio, cust, count(*) as c from o where 'blue' = prio group by prio, cust order by c desc, cust limit 10"),
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name,sql", FOLDED, ids=[c[0] for c in FOLDED])
def test_fixed_keys_fold(probe, world, name, sql, mode):
    res = run(probe, world, sql, mode)
    assert FIXED in note(res, "group: dense"), res["notes"]


NOT_FOLDED = [
    ("an OR", "select prio, cust, count(*) as c from o where prio = 'red' or prio = 'blue' group by prio, cust order by c desc, prio, cust limit 10"),
    ("a NOT", "select prio, cust, count(*) as c from o where not (prio = 'red') group by prio, cust order by c desc, prio, cust limit 10"),
    ("a <>", "select prio, cust, count(*) as c from o where prio <> 'red' group by prio, cust order by c desc, prio, cust limit 10"),
    ("a word the data does not hold", "select prio, cust, count(*) as c from o where prio = 'purple' group by prio, cust order by c desc, cust limit 10"),
    ("IS NULL", "select prio, cust, count(*) as c from o where prio is null group by prio, cust order by c desc, cust limit 10"),
    ("a range on the key", "select seg, cust, count(*) as c from o where seg <= 2 group by seg, cust order by c desc, seg, cust limit 10"),
    ("the key is not the pinned column", "select cust, count(*) as c from o where prio = 'red' group by cust order by c desc, cust limit 10"),
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name,sql", NOT_FOLDED, ids=[c[0] for c in NOT_FOLDED])
def test_other_filters_do_not_fold(probe, world, name, sql, mode):
    res = run(probe, world, sql, mode)
    assert not any(FIXED in n for n in res["notes"]), res["notes"]


# ── hashed groups: the wide key keeps the group-by hashed; the top groups are found on the partial slots ──────────────────────────

HASHED = [
    ("sum desc", "select big, prio, stat, sum(price) as a0, count(*) as c from o group by big, prio, stat order by a0 desc, big, prio, stat limit 10", True),
    ("sum asc, NULL sums first", "select big, prio, sum(price) as a0 from o group by big, prio order by a0 nulls first, big, prio limit 10", True),
    ("sum desc, NULL sums last", "select big, prio, sum(price) as a0 from o group by big, prio order by a0 desc nulls last, big, prio limit 10", True),
    ("min asc with NULL groups", "select big, prio, min(price) as m from o group by big, prio order by m, big, prio limit 12", True),
    ("max desc, offset", "select big, prio, max(price) as m from o group by big, prio order by m desc, big, prio limit 8 offset 5", True),
    ("a filter on the text key", "select big, prio, stat, sum(price) as a0 from o where prio = 'red' and price >= 100 group by big, prio, stat order by a0 desc, big, prio, stat limit 10", True),
    ("count desc: every group ties at the limit", "select big, prio, stat, count(*) as c from o group by big, prio, stat order by c desc, big, prio, stat limit 25", False),
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name,sql,slots", HASHED, ids=[c[0] for c in HASHED])
def test_hashed_composite_keys_find_the_top_groups_on_the_slots(probe, world, name, sql, slots, mode):
    res = run(probe, world, sql, mode)
    assert "group: hashed" in note(res, "group:"), res["notes"]
    if slots:
        assert "found on the groups' partial slots" in note(res, "order: top"), res["notes"]


def test_random_worlds_fold_equal_duckdb(probe, tmp_path):
    rnd = random.Random(207)
    for seed in range(3):
        n = 60000
        con = oracle_connection()
        con.execute(f"""create table o as select i as id, case when i % 41 = 0 then null else (i * {7 + seed}) % 700 end as cust,
            (['a','b','c','d'])[1 + ((i * {3 + seed}) % 4)] as prio,
            case when i % 17 = 0 then null else (['x','y'])[1 + (i % 2)] end as stat,
            ((i % 500) * 0.5)::decimal(10,2) as price, date '2021-01-01' + (i % 90)::int as d, i % 5 as seg, i as wide
            from range({n}) r(i)""")
        path = tmp_path / f"r{seed}.parquet"
        con.execute(f"copy o to '{path}' (format parquet, row_group_size 20000)")
        for _ in range(6):
            pick = rnd.choice(["prio = 'a'", "prio = 'c' and stat = 'x'", "seg = 4", "stat = 'y' and seg = 1", "d = date '2021-02-03'"])
            keys = rnd.choice(["cust, prio, stat", "prio, stat, cust", "seg, prio, cust", "d, cust, prio"])
            lim = rnd.choice([1, 5, 10, 50])
            sql = f"select {keys}, sum(price) as s, count(*) as c from o where {pick} group by {keys} order by s desc nulls last, {keys} limit {lim}"
            for mode in MODES:
                res = run_probe(probe, sql, {"o": path}, mode=mode)
                cur = con.execute(sql)
                assert_equal(got_rows(res), cur.fetchall(), True, f"seed {seed} {mode}: {sql}")
