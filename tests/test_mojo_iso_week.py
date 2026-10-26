"""F-177: ISO week and ISO year (`EXTRACT(week / isoyear FROM d)`, `week(d)`, `weekofyear(d)`, `isoyear(d)`), as expressions and as group keys,
equal to DuckDB for every day from 1900 to 2100 (that covers every one of the 28 year patterns: the 53-week years, the 1 January that belongs to
the year before, the 31 December that belongs to the year after) and for a table whose dates repeat over a short range (the kernel looks the answer
up once per day instead of computing it per row).

The Mojo side is mojo/tools/sql_probe.mojo (the harness of tests/test_mojo_sql.py).
"""
import datetime

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mx_oracle import assert_equal
from test_mojo_sql import probe, run_probe, got_rows  # noqa: F401  (probe is a fixture)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("isoweek")
    day = datetime.date(1900, 1, 1)
    days = []
    while day <= datetime.date(2100, 12, 31):
        days.append(day)
        day += datetime.timedelta(days=1)
    days += [None] * 40
    wide = pa.table({"dt": pa.array(days, pa.date32()), "v": pa.array(list(range(len(days))), pa.int64())})
    pq.write_table(wide, d / "wide.parquet", row_group_size=len(days) // 5 + 1)
    # few dates, many rows: the day table path (span 400 days * 20 < 300,000 rows)
    start = datetime.date(2019, 12, 20)
    narrow_days = [start + datetime.timedelta(days=(i * 7919) % 400) if i % 50 else None for i in range(300_000)]
    narrow = pa.table({"dt": pa.array(narrow_days, pa.date32()), "v": pa.array([i % 1000 for i in range(300_000)], pa.int64())})
    pq.write_table(narrow, d / "narrow.parquet", row_group_size=60_000)
    out = {}
    for name in ("wide", "narrow"):
        con = duckdb.connect()
        con.execute(f"create table t as select * from read_parquet('{d / (name + '.parquet')}')")
        out[name] = (d / f"{name}.parquet", con)
    return out


ROW_QUERIES = [
    "SELECT dt, extract(isoyear FROM dt), extract(week FROM dt), isoyear(dt), week(dt), weekofyear(dt), date_part('week', dt), date_part('isoyear', dt) FROM t",
    "SELECT dt, isoyear(dt) * 100 + week(dt), week(dt) - 1, week(dt) + isoyear(dt) FROM t WHERE week(dt) = 53 OR week(dt) = 1",
]


@pytest.mark.parametrize("mode", ["table", "scan"])
@pytest.mark.parametrize("name", ["wide", "narrow"])
@pytest.mark.parametrize("sql", ROW_QUERIES)
def test_every_day_equals_duckdb(probe, files, name, mode, sql):
    path, con = files[name]
    res = run_probe(probe, sql, {"t": path}, mode=mode)
    assert_equal(got_rows(res), con.execute(sql).fetchall(), False, f"{name}/{mode}: {sql}")


KEY_QUERIES = [
    ("SELECT isoyear(dt), week(dt), count(*), sum(v) FROM t GROUP BY 1, 2", "hashed or dense"),
    ("SELECT extract(week FROM dt) AS w, count(*), min(dt), max(dt) FROM t GROUP BY w", "dense"),
    ("SELECT extract(isoyear FROM dt), count(*) FROM t GROUP BY extract(isoyear FROM dt)", "dense"),
    ("SELECT extract(isoyear FROM dt) AS y, week(dt) AS w, sum(v) FROM t WHERE dt >= '2019-06-01' GROUP BY y, w ORDER BY y, w", "hashed or dense"),
    ("SELECT week(dt), count(DISTINCT isoyear(dt)), max(v) FROM t GROUP BY week(dt) HAVING count(*) > 3", "dense"),
]


@pytest.mark.parametrize("mode", ["table", "scan"])
@pytest.mark.parametrize("name", ["wide", "narrow"])
@pytest.mark.parametrize("q", KEY_QUERIES, ids=lambda q: q[0][:50])
def test_as_group_keys(probe, files, name, mode, q):
    sql, plan = q
    path, con = files[name]
    res = run_probe(probe, sql, {"t": path}, mode=mode)
    assert_equal(got_rows(res), con.execute(sql).fetchall(), "ORDER BY" in sql, f"{name}/{mode}: {sql}")
    group_note = next(n for n in res["notes"] if n.startswith("group:"))
    if plan == "dense":
        assert group_note.startswith("group: dense"), group_note       # the week (1..53) and the year are bounded by the dates' range


def test_constants_are_folded_with_the_same_rule(probe, files):
    path, con = files["wide"]
    days = ["1900-01-01", "1969-12-31", "1970-01-01", "2015-12-31", "2016-01-01", "2016-01-03", "2020-12-31", "2021-01-01", "2021-01-03", "2024-12-30", "2026-01-01",
            "2098-01-01", "2100-12-31"]
    sql = "SELECT " + ", ".join(f"week(DATE '{d}'), isoyear(DATE '{d}')" for d in days) + " FROM t LIMIT 1"
    res = run_probe(probe, sql, {"t": path})
    assert_equal(got_rows(res), con.execute(sql).fetchall(), False, sql)


def test_a_null_date_gives_null_and_a_non_date_is_refused(probe, files):
    path, con = files["wide"]
    res = run_probe(probe, "SELECT count(*), count(week(dt)), count(isoyear(dt)) FROM t", {"t": path})
    total, dated = con.execute("SELECT count(*), count(dt) FROM t").fetchone()
    assert dated == 73414 and total == dated + 40, (total, dated)
    assert got_rows(res) == [(total, dated, dated)], got_rows(res)
    res = run_probe(probe, "SELECT week(v) FROM t", {"t": path}, expect_error=True)
    assert "needs a DATE" in res["error"], res
    res = run_probe(probe, "SELECT date_trunc('isoyear', dt) FROM t", {"t": path}, expect_error=True)
    assert "is not a date part" in res["error"], res
