"""F-168: window functions in SQL (`f(...) OVER (PARTITION BY ... ORDER BY ... [frame])`), equal to DuckDB.

The data are test_mojo_sql.py's generated datasets (ties, NULLs, mixed-scale decimals, dates, a boolean, text with case variants; 8 groups,
10,000 groups, and so on). Every query runs through a table read whole, a Parquet file streamed as 13 row groups, a Frame after 300 random
edits and a streamed file with the same edits, and is compared with DuckDB. A window's ORDER BY ends in the tie-break columns `TB`, so the
order is total (two rows equal on all of them are equal rows), except where a test is about peers (the default RANGE frame, rank, dense_rank).
(DuckDB 1.5.1 answers count(*) = -1 for a frame that ends two or more rows before it starts, `ROWS BETWEEN 3 PRECEDING AND 5 PRECEDING`: the
empty-frame test uses a frame inverted by one row, where its count is 0.) The rows of a result are compared as a multiset (FLOAT to 1e-9 relative: DuckDB's own segment trees add in another order); the queries that
end in ORDER BY are compared in order. Kinds and decimal scales are compared too.
"""
import math
from collections import Counter

import pytest

from mx_oracle import duck_kind, run_mx, typed_rows
from test_mojo_sql import Dataset, data, probe, run_probe, got_rows  # noqa: F401  (fixtures)
from test_mojo_rows import fetch, mx  # noqa: F401  (fixtures)

TB = "id, d, e, dt, g, k"          # ends every window ORDER BY that must be total
ORD = f"x, {TB}"


def close(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        a, b = float(a), float(b)
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        if math.isinf(a) or math.isinf(b):
            return a == b
        return abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1.0)
    return a == b


def key(row):
    return tuple((0, "") if v is None else (1, repr(round(v, 6) if isinstance(v, float) and v == v and abs(v) != math.inf else v)) for v in row)


def compare(got, want, ordered, label):
    assert len(got) == len(want), f"{label}: {len(got)} rows, DuckDB has {len(want)}"
    if not ordered:
        got = sorted(got, key=key)
        want = sorted((tuple(r) for r in want), key=key)
    for i, (a, b) in enumerate(zip(got, want)):
        assert len(a) == len(b) and all(close(x, y) for x, y in zip(a, b)), f"{label}: row {i}: got {a}, DuckDB has {b}"


def check(mode, ds, mx, probe, sql, duck=None, ordered=False):
    got, cols, notes = fetch(mode, ds, mx, probe, sql)
    cur = ds.oracle(mode).execute(duck or sql)
    want = cur.fetchall()
    for i, desc in enumerate(cur.description):
        dk = duck_kind(desc[1])
        assert cols[i]["kind"] == dk[0] and (dk[0] != "DECIMAL" or cols[i]["scale"] == dk[1]), f"{sql}: column {i}: got {cols[i]}, DuckDB has {dk}"
    compare(got, want, ordered, sql)
    return notes


def over(spec, *fns, first=0):
    return ", ".join(f"{f} OVER ({spec}) AS w{first + i}" for i, f in enumerate(fns))


P = f"PARTITION BY k ORDER BY {ORD}"
PX = "PARTITION BY k ORDER BY x"           # ties: peers

QUERIES = [
    # ranking
    ("row-number", f"SELECT id, {over(P, 'row_number()')} FROM t"),
    ("row-number-one-partition", f"SELECT id, {over(f'ORDER BY {ORD}', 'row_number()')} FROM t"),
    ("rank-dense-rank-peers", f"SELECT id, {over(PX, 'rank()', 'dense_rank()')} FROM t"),
    ("rank-desc-nulls-first", f"SELECT id, {over('ORDER BY x DESC NULLS FIRST', 'rank()', 'dense_rank()')} FROM t"),
    ("rank-nulls-last-asc", f"SELECT id, {over('PARTITION BY g ORDER BY x NULLS LAST', 'rank()')} FROM t"),
    ("percent-rank-cume-dist", f"SELECT id, {over('PARTITION BY g ORDER BY x', 'percent_rank()', 'cume_dist()')} FROM t"),
    ("percent-rank-cume-dist-k", f"SELECT id, {over(PX, 'percent_rank()', 'cume_dist()')} FROM t"),
    ("ntile", f"SELECT id, {over(P, 'ntile(4)', 'ntile(7)', 'ntile(1)', 'ntile(100000)')} FROM t"),
    ("rank-no-order", f"SELECT id, {over('PARTITION BY k', 'rank()', 'dense_rank()', 'percent_rank()', 'cume_dist()')} FROM t"),
    ("text-order-key", f"SELECT id, {over(f'PARTITION BY k ORDER BY lower(s), {TB}', 'row_number()')} FROM t",
     f"SELECT id, row_number() OVER (PARTITION BY k ORDER BY lower(s), {TB}) AS w0 FROM t"),
    ("two-order-keys-directions", f"SELECT id, {over(f'PARTITION BY k ORDER BY b DESC, x ASC NULLS FIRST, {TB}', 'row_number()', 'rank()')} FROM t"),
    # lag and lead
    ("lag-lead-int", f"SELECT id, {over(P, 'lag(x)', 'lead(x, 2)', 'lag(x, 3, -1)', 'lead(x, 1, 7)', 'lag(x, 0)')} FROM t"),
    ("lag-negative-offset", f"SELECT id, {over(P, 'lag(x, -2)', 'lead(x, -1, 5)')} FROM t"),
    ("lag-lead-decimal", f"SELECT id, {over(P, 'lag(d, 2, 0)', 'lead(d, 1, 1.25)', 'lag(e, 1, -3)', 'lead(e)')} FROM t"),
    ("lag-lead-text", f"SELECT id, {over(P, 'lag(s)', 'lead(s, 1, ' + chr(39) + 'none' + chr(39) + ')', 'lag(s, 2, ' + chr(39) + 'West' + chr(39) + ')')} FROM t"),
    ("lag-lead-date-bool", f"SELECT id, {over(P, 'lag(dt, 1, DATE ' + chr(39) + '2000-01-01' + chr(39) + ')', 'lead(dt, 2)', 'lag(b)', 'lead(b, 1, true)')} FROM t"),
    ("lag-lead-float", f"SELECT id, {over(P, 'lag(x / 3.0, 1, 0.5)', 'lead(x / 7.0)')} FROM t"),
    ("lag-lead-many-partitions", f"SELECT id, {over(f'PARTITION BY g ORDER BY {ORD}', 'lag(x)', 'lead(x, 1, 0)')} FROM t"),
    # first, last, nth
    ("first-last-default-frame", f"SELECT id, {over(P, 'first_value(x)', 'last_value(x)', 'first_value(s)', 'last_value(d)')} FROM t"),
    ("first-last-peers", f"SELECT id, {over(PX, 'last_value(x)', 'first_value(x)', 'first_value(x * 2)')} FROM t"),
    ("first-last-whole-partition", f"SELECT id, {over(P + ' ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING', 'first_value(x)', 'last_value(x)', 'last_value(s)')} FROM t"),
    ("nth-value", f"SELECT id, {over(P + ' ROWS BETWEEN 2 PRECEDING AND CURRENT ROW', 'nth_value(x, 2)', 'nth_value(d, 3)', 'nth_value(s, 1)')} FROM t"),
    ("first-last-sliding", f"SELECT id, {over(P + ' ROWS BETWEEN 3 PRECEDING AND 1 FOLLOWING', 'first_value(x)', 'last_value(x)')} FROM t"),
    # aggregates over the default frame (peers) and the whole partition
    ("sum-running-peers", f"SELECT id, {over(PX, 'sum(x)', 'count(*)', 'count(x)', 'avg(x)', 'min(x)', 'max(x)')} FROM t"),
    ("sum-running-decimal-peers", f"SELECT id, {over('PARTITION BY g ORDER BY x', 'sum(d)', 'avg(d)', 'min(e)', 'max(d)', 'sum(e)')} FROM t"),
    ("whole-partition", f"SELECT id, {over('PARTITION BY k', 'sum(x)', 'count(*)', 'avg(d)', 'min(dt)', 'max(dt)', 'max(b)', 'min(lower(s))')} FROM t"),
    ("whole-table", f"SELECT id, {over('', 'sum(x)', 'count(*)', 'avg(e)', 'max(x)')} FROM t"),
    # rows frames
    ("rows-moving-sum", f"SELECT id, {over(f'ORDER BY {ORD} ROWS BETWEEN 2 PRECEDING AND CURRENT ROW', 'sum(x)', 'avg(d)', 'count(x)')} FROM t"),
    ("rows-centred", f"SELECT id, {over(f'PARTITION BY g ORDER BY {ORD} ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING', 'avg(x)', 'sum(d)', 'min(x)', 'max(e)')} FROM t"),
    ("rows-minmax-window", f"SELECT id, {over(P + ' ROWS BETWEEN 5 PRECEDING AND 2 FOLLOWING', 'min(x)', 'max(x)', 'min(dt)', 'max(lower(s))')} FROM t"),
    ("rows-before-current", f"SELECT id, {over(P + ' ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING', 'sum(x)', 'count(*)', 'count(x)', 'min(x)', 'avg(d)')} FROM t"),
    ("rows-after-current", f"SELECT id, {over(P + ' ROWS BETWEEN 1 FOLLOWING AND 3 FOLLOWING', 'sum(x)', 'count(*)', 'max(x)')} FROM t"),
    ("rows-to-the-end", f"SELECT id, {over(P + ' ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING', 'sum(x)', 'min(d)', 'avg(e)')} FROM t"),
    ("rows-from-the-start", f"SELECT id, {over(P + ' ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING', 'sum(x)', 'max(x)')} FROM t"),
    ("rows-unbounded-both", f"SELECT id, {over(P + ' ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING', 'sum(x)', 'count(*)', 'avg(x)')} FROM t"),
    ("rows-shorthand", f"SELECT id, {over(P + ' ROWS 2 PRECEDING', 'sum(x)', 'max(d)')}, {over(P + ' ROWS UNBOUNDED PRECEDING', 'sum(d)', first=9)} FROM t"),
    ("rows-large-offsets", f"SELECT id, {over(P + ' ROWS BETWEEN 1000000 PRECEDING AND 1000000 FOLLOWING', 'sum(x)', 'min(x)')} FROM t"),
    ("rows-empty-frame", f"SELECT id, {over(P + ' ROWS BETWEEN 1 PRECEDING AND 2 PRECEDING', 'sum(x)', 'count(*)', 'min(x)')} FROM t"),
    ("rows-current-only", f"SELECT id, {over(P + ' ROWS BETWEEN CURRENT ROW AND CURRENT ROW', 'sum(x)', 'count(x)', 'min(d)')} FROM t"),
    # range frames (peers)
    ("range-unbounded-both", f"SELECT id, {over(PX + ' RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING', 'sum(x)', 'count(*)')} FROM t"),
    ("range-current-to-end", f"SELECT id, {over(PX + ' RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING', 'sum(x)', 'min(d)', 'count(*)')} FROM t"),
    ("range-current-only", f"SELECT id, {over(PX + ' RANGE BETWEEN CURRENT ROW AND CURRENT ROW', 'sum(x)', 'count(*)', 'avg(d)')} FROM t"),
    ("range-explicit-default", f"SELECT id, {over(PX + ' RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW', 'sum(x)', 'max(d)')} FROM t"),
    ("range-shorthand", f"SELECT id, {over(PX + ' RANGE UNBOUNDED PRECEDING', 'sum(x)')} FROM t"),
    ("range-no-order", f"SELECT id, {over('PARTITION BY k RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING', 'sum(x)', 'count(*)')} FROM t"),
    # floats
    ("float-sums", f"SELECT id, {over(PX, 'sum(x / 3.0)', 'avg(x / 7.0)', 'min(x / 3.0)', 'max(x / 3.0)')} FROM t"),
    ("float-moving", f"SELECT id, {over(P + ' ROWS BETWEEN 3 PRECEDING AND 2 FOLLOWING', 'sum(x / 3.0)', 'avg(x / 7.0)', 'min(x / 3.0)', 'max(d / 3.0)')} FROM t"),
    ("float-whole", f"SELECT id, {over('PARTITION BY k', 'sum(x / 3.0)', 'avg(e / 7.0)', 'count(x / 3.0)')} FROM t"),
    ("float-order-key", f"SELECT id, {over(f'PARTITION BY k ORDER BY x / 3.0 DESC, {TB}', 'row_number()', 'rank()')} FROM t"),
    ("float-partition-key", f"SELECT id, {over('PARTITION BY x / 3.0', 'count(*)', 'sum(d)')} FROM t"),
    # expressions as arguments, several windows, shared sorts
    ("expression-args", f"SELECT id, {over(P + ' ROWS BETWEEN 2 PRECEDING AND CURRENT ROW', 'sum(x * 2 + 1)', 'sum(d + e)', 'max(x - 3)')} FROM t"),
    ("several-sorts", f"SELECT id, {over(P, 'row_number()')}, {over('PARTITION BY g ORDER BY x DESC, id', 'rank()', first=8)}, sum(x) OVER (PARTITION BY s) AS w7 FROM t"),
    ("arithmetic-over-windows", f"SELECT id, x - lag(x) OVER ({P}) AS delta, sum(x) OVER ({P}) / count(x) OVER ({P}) AS ratio, "
                                f"coalesce(lag(x) OVER ({P}), -1) + 1 AS c FROM t"),
    ("case-over-window", f"SELECT id, CASE WHEN rank() OVER ({PX}) = 1 THEN 'first' ELSE 'later' END AS pos FROM t"),
    ("window-spec-spacing", "SELECT id, sum(x) over(partition by k order by id rows between 1 preceding and current row) AS w FROM t"),
    ("star", f"SELECT *, rank() OVER ({PX}) AS r FROM t"),
    # with WHERE, GROUP BY, DISTINCT, empty
    ("where", f"SELECT id, {over(P + ' ROWS BETWEEN 2 PRECEDING AND CURRENT ROW', 'sum(x)', 'row_number()')} FROM t WHERE x > 0 AND s IS NOT NULL"),
    ("where-nothing", f"SELECT id, {over(P, 'row_number()', 'sum(x)', 'lag(x)')} FROM t WHERE x > 100000"),
    ("group-by-rank", "SELECT k, sum(x) AS s, count(*) AS c, rank() OVER (ORDER BY sum(x) DESC) AS r, sum(sum(x)) OVER (ORDER BY k) AS cum FROM t GROUP BY k",
     "SELECT k, sum(x) AS s, count(*) AS c, rank() OVER (ORDER BY sum(x) DESC) AS r, sum(sum(x)) OVER (ORDER BY k) AS cum FROM t GROUP BY k"),
    ("group-by-partitions", "SELECT g, k, count(*) AS c, row_number() OVER (PARTITION BY g ORDER BY count(*) DESC, k) AS rn, "
                            "sum(count(*)) OVER (PARTITION BY g) AS tot FROM t GROUP BY g, k"),
    ("group-by-having-lag", "SELECT g, sum(d) AS s, lag(sum(d)) OVER (ORDER BY g) AS prev FROM t GROUP BY g HAVING count(*) > 2"),
    ("group-by-ordinal", "SELECT k, count(*) AS c, rank() OVER (ORDER BY count(*) DESC) AS r FROM t GROUP BY 1"),
    ("distinct", f"SELECT DISTINCT rank() OVER ({PX}) AS r, k FROM t"),
    ("window-of-window-free-expression", "SELECT x + 1 AS xp, sum(x + 1) OVER (PARTITION BY k) AS s FROM t WHERE k IS NOT NULL AND x IS NOT NULL"),
]


def qid(q):
    return q[0]


@pytest.mark.parametrize("mode", ["table", "scan", "frame", "framescan"])
@pytest.mark.parametrize("ds_name", ["g8", "g10k"])
@pytest.mark.parametrize("q", QUERIES, ids=qid)
def test_windows_equal_duckdb(mx, probe, data, ds_name, mode, q):
    if ds_name == "g10k" and mode in ("frame", "framescan") and q[0] not in ("rows-moving-sum", "sum-running-peers", "lag-lead-many-partitions"):
        pytest.skip("the 200,000-row edited frame runs three representative queries")
    check(mode, data[ds_name], mx, probe, q[1], q[2] if len(q) > 2 else None)


ORDERED = [
    ("order-by-window-alias", "SELECT id, row_number() OVER (ORDER BY x, id) AS rn FROM t ORDER BY rn DESC LIMIT 7"),
    ("order-by-window-expression", f"SELECT id, x FROM t ORDER BY rank() OVER ({PX}), id LIMIT 25"),
    ("order-by-ordinal-and-limit", f"SELECT id, sum(x) OVER ({P}) AS s FROM t ORDER BY 2 DESC NULLS LAST, 1 LIMIT 40 OFFSET 5"),
    ("order-by-unselected-column", f"SELECT id, lag(x) OVER ({P}) AS p FROM t ORDER BY x DESC NULLS LAST, id LIMIT 30"),
    ("order-by-several", f"SELECT id, k, row_number() OVER (PARTITION BY k ORDER BY id) AS rn FROM t ORDER BY k NULLS LAST, rn LIMIT 60"),
    ("limit-only-after-windows", "SELECT id, sum(x) OVER (ORDER BY id) AS run FROM t ORDER BY id LIMIT 12"),
]


@pytest.mark.parametrize("mode", ["table", "scan"])
@pytest.mark.parametrize("ds_name", ["g8", "g10k"])
@pytest.mark.parametrize("q", ORDERED, ids=qid)
def test_windows_then_order_by_limit_equal_duckdb(mx, probe, data, ds_name, mode, q):
    check(mode, data[ds_name], mx, probe, q[1], q[2] if len(q) > 2 else None, ordered=True)


MILLION = [
    ("million-partitions-running", f"SELECT id, {over(f'PARTITION BY g ORDER BY {ORD}', 'row_number()', 'lag(x)', 'sum(d)')} FROM t"),
    ("million-partitions-peers", f"SELECT id, {over('PARTITION BY g ORDER BY x', 'rank()', 'cume_dist()', 'min(d)')} FROM t"),
    ("one-partition-moving", f"SELECT id, {over(f'ORDER BY {ORD} ROWS BETWEEN 100 PRECEDING AND 100 FOLLOWING', 'sum(x)', 'max(d)', 'avg(e)')} FROM t"),
]


@pytest.mark.parametrize("mode", ["table", "scan"])
@pytest.mark.parametrize("q", MILLION, ids=qid)
def test_a_million_rows_in_a_million_partitions_and_in_one(mx, probe, data, mode, q):
    check(mode, data["g1m"], mx, probe, q[1])


def test_threads_do_not_change_a_window_result(mx, data):
    """1 thread and 8 threads give the same rows, bit for bit (a FLOAT sum included)."""
    ds = data["g10k"]
    sql = (f"SELECT id, sum(x / 3.0) OVER ({P} ROWS BETWEEN 3 PRECEDING AND 2 FOLLOWING) AS f, avg(x / 7.0) OVER (PARTITION BY g ORDER BY id) AS a, "
           f"rank() OVER (PARTITION BY g ORDER BY x) AS r, lag(s) OVER (PARTITION BY g ORDER BY id) AS l FROM t ORDER BY id")
    one = run_mx(mx, ["sql", "--max-rows", "0", "--threads", "1", sql], {"t": ds.path})
    many = run_mx(mx, ["sql", "--max-rows", "0", "--threads", "8", sql], {"t": ds.path})
    assert one["rows"] == many["rows"]
    assert len(one["rows"]) == ds.n


def test_the_notes_say_the_windows_ran_on_the_cpu(mx, data):
    out = run_mx(mx, ["sql", "--max-rows", "3", f"SELECT id, rank() OVER ({PX}) AS r, sum(x) OVER ({PX}) AS s FROM t"], {"t": data["g8"].path})
    notes = out["notes"]
    assert any(n.startswith("window: 2 functions in 1 sort over 20,000 rows") or n.startswith("window: 2 functions in 1 sort over 20000 rows") for n in notes), notes
    assert any(n.startswith("window: ") and "ms" in n for n in notes), notes


def test_explain_shows_the_window_step_and_runs_nothing(mx, data):
    out = run_mx(mx, ["sql", "--max-rows", "0", f"EXPLAIN SELECT id, rank() OVER ({PX}) AS r FROM t"], {"t": data["g8"].path})
    text = "\n".join(r[0] for r in out["rows"])
    assert text.startswith("Window functions (after the query below, on the CPU)"), text
    assert "rank() OVER (PARTITION BY k ORDER BY x)" in text, text
    assert "Query" in text and "Scan" in text, text


def test_an_int_sum_past_int64_raises_like_the_other_sums(mx, data):
    import subprocess
    r = subprocess.run([str(mx), "sql", "SELECT sum(x * 1000000000000000) OVER (ORDER BY id) AS s FROM t", "-t", f"t={data['g8'].path}"], capture_output=True)
    assert r.returncode != 0
    assert "a window sum passed the int64 range" in (r.stdout + r.stderr).decode(), (r.stdout + r.stderr)[-300:]


ERRORS = [
    ("SELECT id FROM t WHERE sum(x) OVER () > 3", "SQL error at line 1, column 24: window functions are only allowed in the SELECT list and ORDER BY (not in WHERE, GROUP BY, HAVING or ON)"),
    ("SELECT k FROM t GROUP BY k HAVING rank() OVER (ORDER BY k) = 1", "window functions are only allowed in the SELECT list and ORDER BY"),
    ("SELECT sum(x) OVER (ORDER BY id RANGE BETWEEN 5 PRECEDING AND CURRENT ROW) FROM t", "`RANGE` with an offset (5) is not supported yet"),
    ("SELECT sum(x) OVER (ORDER BY id GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t", "a `GROUPS` frame is not supported yet"),
    ("SELECT sum(x) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW) FROM t", "`EXCLUDE` in a window frame is not supported yet"),
    ("SELECT sum(x) OVER w FROM t", "`OVER w` (a named window) is not supported yet"),
    ("SELECT median(x) OVER () FROM t", "`median` is not supported as a window function"),
    ("SELECT sum(DISTINCT x) OVER () FROM t", "DISTINCT inside a window function"),
    ("SELECT lag(x) IGNORE NULLS OVER (ORDER BY id) FROM t", "`IGNORE NULLS` is not supported yet"),
    ("SELECT sum(x) FILTER (WHERE x > 0) OVER () FROM t", "`FILTER` is not supported yet"),
    ("SELECT sum(s) OVER () FROM t", "sum(...) OVER is not supported for TEXT values"),
    ("SELECT ntile(0) OVER (ORDER BY id) FROM t", "ntile needs at least one bucket"),
    ("SELECT ntile(x) OVER (ORDER BY id) FROM t", "the bucket count of ntile must be a whole-number literal"),
    ("SELECT lag(x, 1, 'a') OVER (ORDER BY id) FROM t", "the default must be a number, like the column it follows"),
    ("SELECT lag(s, 1, 3) OVER (ORDER BY id) FROM t", "the default must be text, like the column it follows"),
    ("SELECT rank(x) OVER (ORDER BY id) FROM t", "rank() takes 0 arguments, found 1"),
    ("SELECT sum(rank() OVER (ORDER BY id)) OVER () FROM t", "a window function cannot be inside the argument of another window function"),
    ("SELECT sum(sum(x) OVER ()) FROM t", "a window function cannot be inside the aggregate sum(...)"),
    ("SELECT sum(x) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED FOLLOWING AND CURRENT ROW) FROM t", "a window frame cannot start at UNBOUNDED FOLLOWING"),
    ("SELECT sum(x) OVER (ORDER BY id ROWS BETWEEN CURRENT ROW AND UNBOUNDED PRECEDING) FROM t", "a window frame cannot end at UNBOUNDED PRECEDING"),
    ("SELECT sum(x) OVER (ORDER BY id ROWS BETWEEN 2 FOLLOWING AND 1 PRECEDING) FROM t", "frame starting from following row cannot have preceding rows"),
    ("SELECT sum(x) OVER (ORDER BY id ROWS BETWEEN 2 FOLLOWING AND CURRENT ROW) FROM t", "frame starting from following row cannot end with current row"),
    ("SELECT sum(x) OVER (ORDER BY id ROWS BETWEEN CURRENT ROW AND 1 PRECEDING) FROM t", "frame starting from current row cannot have preceding rows"),
    ("SELECT sum(x) OVER (ORDER BY id ROWS BETWEEN y PRECEDING AND CURRENT ROW) FROM t", "a frame offset that is not a whole-number literal"),
    ("SELECT *, rank() OVER (ORDER BY a.id) FROM t a JOIN t b ON a.id = b.id", "`*` next to a window function over a join"),
    ("SELECT rank() OVER (PARTITION BY nope ORDER BY id) FROM t", "column `nope` not found"),
]


@pytest.mark.parametrize("sql,message", ERRORS, ids=[e[0][:55] for e in ERRORS])
def test_unsupported_window_forms_say_what_and_where(probe, data, sql, message):
    res = run_probe(probe, sql, {"t": data["g8"].path}, expect_error=True)
    assert "error" in res, f"`{sql}` should fail, it gave {str(res)[:200]}"
    assert message in res["error"], f"`{sql}`: {res['error']}"
    assert res["error"].startswith("SQL error at line") or "window sum" in res["error"] or "sum(...) OVER" in res["error"] or "ntile" in res["error"], res["error"]


def test_a_window_over_a_join(mx, data):
    ds = data["g8"]
    sql = "SELECT a.id, b.x AS bx, sum(a.x) OVER (PARTITION BY a.k ORDER BY a.id, b.id) AS s FROM t a JOIN t b ON a.id = b.id AND b.id < 3000"
    out = run_mx(mx, ["sql", "--max-rows", "0", sql], {"t": ds.path})
    want = ds.table.execute(sql).fetchall()
    compare(typed_rows(out), want, False, sql)


def test_a_csv_file_is_a_source(mx, data, tmp_path):
    """A CSV file is read into memory (the columns the query needs) and the windows run over it."""
    ds = data["g8"]
    path = tmp_path / "w.csv"
    ds.table.execute(f"COPY (SELECT id, k, x FROM t) TO '{path}' (HEADER, DELIMITER ',')")
    sql = ("SELECT id, sum(x) OVER (PARTITION BY k ORDER BY x, id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS s, rank() OVER (PARTITION BY k ORDER BY x) AS r, "
           "lag(x, 2, -1) OVER (PARTITION BY k ORDER BY id) AS l FROM c")
    out = run_mx(mx, ["sql", "--max-rows", "0", sql], {"c": path})
    want = ds.table.execute(sql.replace("FROM c", "FROM t")).fetchall()
    compare(typed_rows(out), want, False, sql)


@pytest.fixture(scope="module")
def specials(tmp_path_factory):
    """3,000 rows of DOUBLEs with NaN, inf, -inf, -0.0, 0.0 and NULL among ordinary values, in 12 groups."""
    import duckdb
    d = tmp_path_factory.mktemp("specials")
    con = duckdb.connect()
    con.execute("set threads=1")
    con.execute("select setseed(0.5)")
    con.execute("""create table t as select i::BIGINT as id, (i % 12)::BIGINT as g,
        case (random() * 14)::int
          when 0 then 'nan'::double when 1 then 'inf'::double when 2 then '-inf'::double when 3 then -0.0::double when 4 then 0.0::double
          when 5 then null when 6 then 1e300 when 7 then -1e300 else ((random() * 2000)::int - 1000) / 8.0 end as v
        from range(3000) r(i)""")
    path = d / "specials.parquet"
    con.execute(f"copy t to '{path}' (format parquet, row_group_size 700)")
    return con, path


SPECIAL_QUERIES = [
    ("rank-by-v", "SELECT id, rank() OVER (PARTITION BY g ORDER BY v) AS a, dense_rank() OVER (ORDER BY v DESC) AS b, percent_rank() OVER (ORDER BY v NULLS FIRST) AS c FROM t"),
    ("partition-by-v", "SELECT id, count(*) OVER (PARTITION BY v) AS n, row_number() OVER (PARTITION BY v ORDER BY id) AS r FROM t"),
    ("minmax-running", "SELECT id, min(v) OVER (PARTITION BY g ORDER BY id) AS lo, max(v) OVER (PARTITION BY g ORDER BY id) AS hi FROM t"),
    ("minmax-moving", "SELECT id, min(v) OVER (PARTITION BY g ORDER BY id ROWS BETWEEN 3 PRECEDING AND 1 FOLLOWING) AS lo, "
                      "max(v) OVER (PARTITION BY g ORDER BY id ROWS BETWEEN 3 PRECEDING AND 1 FOLLOWING) AS hi FROM t"),
    ("sums-running", "SELECT id, sum(v) OVER (PARTITION BY g ORDER BY id) AS s, avg(v) OVER (PARTITION BY g ORDER BY id) AS a, count(v) OVER (PARTITION BY g ORDER BY id) AS c FROM t"),
    ("sums-whole", "SELECT id, sum(v) OVER (PARTITION BY g) AS s, avg(v) OVER () AS a FROM t"),
    ("lag-first-last", "SELECT id, lag(v, 1, 0.5) OVER (PARTITION BY g ORDER BY id) AS l, first_value(v) OVER (PARTITION BY g ORDER BY id) AS f, "
                       "last_value(v) OVER (PARTITION BY g ORDER BY id ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING) AS la FROM t"),
]


@pytest.mark.parametrize("mode", ["table", "scan"])
@pytest.mark.parametrize("q", SPECIAL_QUERIES, ids=qid)
def test_nan_infinities_signed_zeros_and_nulls_equal_duckdb(mx, specials, mode, q):
    con, path = specials
    out = run_mx(mx, ["sql", "--max-rows", "0", "--stream", "always" if mode == "scan" else "never", q[1]], {"t": path})
    cur = con.execute(q[1])
    want = cur.fetchall()
    for i, desc in enumerate(cur.description):
        dk = duck_kind(desc[1])
        assert out["columns"][i]["kind"] == dk[0], f"{q[1]}: column {i}: {out['columns'][i]}, DuckDB has {dk}"
    got = typed_rows(out)
    compare(sorted(got, key=lambda r: r[0]), sorted(want, key=lambda r: r[0]), True, q[1])
