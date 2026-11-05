"""F-152: SQL row queries in the Mojo API (SELECT *, expressions, DISTINCT, ORDER BY, top-k, LIMIT / OFFSET), equal to DuckDB.

The data are test_mojo_sql.py's generated datasets (nulls, mixed-scale decimals, dates, a boolean, text with case variants, at
20,000 / 200,000 / 1.2 million rows). Every query runs through

- a table read whole (`mx sql` with --stream never),
- a Parquet file streamed as 13 row groups (`mx sql` with --stream always),
- a Frame after 300 random edits (the probe's `frame` mode; `mx` opens no Frames),

and is compared with DuckDB (tests/mx_oracle.py's typed comparison). ORDER BY has ties, which no two engines break the same way, so an
ordered query is compared in two parts: the sequence of its sort-key columns must equal DuckDB's exactly (text under the owner's
case-insensitive order, `lower(s), s`), and the rows must be DuckDB's rows (as a multiset; with a LIMIT, a sub-multiset of the
unlimited result). Each test also asserts the path that ran, from the planner's notes.
"""
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

import pytest
from conftest import oracle_connection

from mx_oracle import duck_kind, mx_binary, run_mx, typed_rows
from test_mojo_sql import Dataset, data, probe, run_probe, got_rows, ROW_GROUPS, COLS  # noqa: F401  (fixtures)


@pytest.fixture(scope="session")
def mx():
    return mx_binary()


@dataclass
class Q:
    core: str                          # SELECT ... [ORDER BY ...] with no LIMIT / OFFSET, in our dialect
    duck: Optional[str] = None         # the same for DuckDB (text sort keys written as lower(s), s); core if None
    keys: Optional[List[int]] = None   # the output columns that are the sort keys, in order (None: no ORDER BY)
    limit: Optional[int] = None
    offset: int = 0
    tag: str = ""

    @property
    def sql(self):
        s = self.core
        if self.limit is not None:
            s += f" LIMIT {self.limit}"
        if self.offset:
            s += f" OFFSET {self.offset}"
        return s

    def id(self):
        return self.tag or self.sql[:60]


def fetch(mode, ds, mx, probe, sql):
    """(rows, columns) of `sql` over the dataset in the given mode, typed."""
    if mode in ("frame", "framescan"):
        res = run_probe(probe, sql, {"t": ds.path}, mode=mode, edits=ds.edits_path)
        return got_rows(res), res["columns"], res["notes"]
    out = run_mx(mx, ["sql", "--max-rows", "0", "--stream", "always" if mode == "scan" else "never", sql], {"t": ds.path})
    return typed_rows(out), out["columns"], out["notes"]


def check_query(mode, ds, mx, probe, q: Q, big_ok=True):
    got, cols, notes = fetch(mode, ds, mx, probe, q.sql)
    con = ds.oracle(mode)
    cur = con.execute(q.duck or q.core)
    full = cur.fetchall()
    for i, desc in enumerate(cur.description):
        dk = duck_kind(desc[1])
        assert cols[i]["kind"] == dk[0] and (dk[0] != "DECIMAL" or cols[i]["scale"] == dk[1]), f"{q.sql}: column {i}: got {cols[i]}, DuckDB has {dk}"
    if q.keys is not None:
        want_keys = [tuple(r[i] for i in q.keys) for r in full]
        want_keys = want_keys[q.offset:q.offset + q.limit] if q.limit is not None else want_keys[q.offset:]
        got_keys = [tuple(r[i] for i in q.keys) for r in got]
        assert got_keys == want_keys, f"{q.sql}: the sort keys differ (first mismatch at {next((i for i, (a, b) in enumerate(zip(got_keys, want_keys)) if a != b), None)}; {len(got_keys)} rows, DuckDB has {len(want_keys)})"
    else:
        assert q.limit is None or len(got) == min(q.limit, max(len(full) - q.offset, 0)), f"{q.sql}: {len(got)} rows"
    if q.limit is None and q.offset == 0:
        assert len(got) == len(full), f"{q.sql}: {len(got)} rows, DuckDB has {len(full)}"
        assert Counter(map(_canon, got)) == Counter(map(_canon, full)), f"{q.sql}: the rows differ"
    elif q.limit is None:
        assert len(got) == max(len(full) - q.offset, 0)
    else:
        extra = Counter(map(_canon, got)) - Counter(map(_canon, full))
        assert not extra, f"{q.sql}: rows DuckDB does not have: {list(extra)[:3]}"
    return notes


def _canon(row):
    return tuple("nan" if isinstance(v, float) and v != v else v for v in row)


QUERIES = [
    Q("SELECT * FROM t", tag="select-star"),
    Q("SELECT t.*, x + 1 AS x1 FROM t WHERE x > 0", tag="qualified-star-and-expression"),
    Q("SELECT id, x, d FROM t WHERE x > 500 AND d IS NOT NULL", tag="where-and-null"),
    Q("SELECT id, x * 2 + 1 AS y, d + 0.5 AS d5, dt, e - d AS ed FROM t WHERE k = 1 OR b", tag="expressions"),
    Q("SELECT s, b, dt FROM t WHERE s IN ('West', 'east', 'NORTH') OR NOT b", tag="text-in-and-bool"),
    Q("SELECT id, x > 0 AS pos, b, x IS NULL AS nx FROM t WHERE id < 5000", tag="boolean-outputs"),
    Q("SELECT id, k // 2 AS h, x % 7 AS m, -x AS neg FROM t WHERE x IS NOT NULL AND k IS NOT NULL", tag="integer-division-and-modulo"),
    Q("SELECT id FROM t WHERE x > 100000", tag="empty"),
    Q("SELECT DISTINCT s FROM t", tag="distinct-text"),
    Q("SELECT DISTINCT k, b FROM t", tag="distinct-int-bool-null"),
    Q("SELECT DISTINCT x % 10 AS m, s FROM t", tag="distinct-expression-and-text"),
    Q("SELECT DISTINCT dt FROM t WHERE dt > DATE '2021-01-01'", tag="distinct-date"),
    Q("SELECT DISTINCT * FROM t WHERE id < 300", tag="distinct-star"),
    Q("SELECT id, x, s FROM t ORDER BY x DESC, id", duck="SELECT id, x, s FROM t ORDER BY x DESC NULLS LAST, id NULLS LAST", keys=[1, 0], tag="order-two-keys"),
    Q("SELECT id, k, x FROM t WHERE x IS NOT NULL ORDER BY k", duck="SELECT id, k, x FROM t WHERE x IS NOT NULL ORDER BY k NULLS LAST", keys=[1], limit=100, offset=20,
      tag="ties-limit-offset"),
    Q("SELECT id, x FROM t ORDER BY x DESC NULLS FIRST", duck="SELECT id, x FROM t ORDER BY x DESC NULLS FIRST", keys=[1], limit=10, tag="nulls-first-desc"),
    Q("SELECT id, s, dt FROM t ORDER BY s, dt DESC", duck="SELECT id, s, dt FROM t ORDER BY lower(s) NULLS LAST, s, dt DESC NULLS LAST", keys=[1, 2],
      limit=50, tag="text-order"),
    Q("SELECT id, s, k FROM t ORDER BY s DESC NULLS FIRST, k", duck="SELECT id, s, k FROM t ORDER BY lower(s) DESC NULLS FIRST, s DESC, k NULLS LAST", keys=[1, 2],
      limit=300, tag="text-desc-nulls-first"),
    Q("SELECT id, d, e FROM t ORDER BY d + e, id", duck="SELECT id, d, e FROM t ORDER BY d + e NULLS LAST, id NULLS LAST", keys=[0], limit=40, tag="order-by-expression"),
    Q("SELECT id, x FROM t ORDER BY 2 DESC, 1", duck="SELECT id, x FROM t ORDER BY 2 DESC NULLS LAST, 1", keys=[1, 0], limit=25, tag="order-by-ordinal"),
    Q("SELECT id, x AS xx FROM t ORDER BY xx NULLS LAST, id", duck="SELECT id, x AS xx FROM t ORDER BY xx NULLS LAST, id NULLS LAST", keys=[1, 0], limit=25,
      tag="order-by-alias"),
    Q("SELECT id, b FROM t ORDER BY b, id", duck="SELECT id, b FROM t ORDER BY b NULLS LAST, id NULLS LAST", keys=[1, 0], limit=30, tag="order-by-bool"),
    Q("SELECT s FROM t ORDER BY x, id", duck="SELECT s FROM t ORDER BY x NULLS LAST, id NULLS LAST", keys=None, limit=7, tag="order-by-a-column-not-selected"),
    Q("SELECT id FROM t ORDER BY id", duck="SELECT id FROM t ORDER BY id NULLS LAST", keys=[0], limit=10, offset=10_000_000, tag="offset-past-the-end"),
    Q("SELECT id FROM t ORDER BY id", duck="SELECT id FROM t ORDER BY id NULLS LAST", keys=[0], limit=0, tag="limit-zero"),
    Q("SELECT id, x FROM t WHERE x > 0", limit=17, tag="limit-alone"),
    Q("SELECT id, x FROM t WHERE x > 0", limit=5, offset=3, tag="limit-offset-alone"),
    Q("SELECT id, x FROM t WHERE x > 0", limit=0, tag="limit-zero-alone"),
    Q("SELECT id FROM t ORDER BY id", duck="SELECT id FROM t ORDER BY id NULLS LAST", keys=[0], offset=19990, tag="offset-without-limit"),
    Q("SELECT DISTINCT s FROM t ORDER BY s", duck="SELECT DISTINCT s FROM t ORDER BY lower(s) NULLS LAST, s", keys=[0], limit=4, tag="distinct-order-limit"),
]


@pytest.mark.parametrize("ds_name", ["g8", "g10k"])
@pytest.mark.parametrize("mode", ["table", "scan", "frame", "framescan"])
@pytest.mark.parametrize("q", QUERIES, ids=Q.id)
def test_row_queries_equal_duckdb(mx, probe, data, ds_name, mode, q):
    check_query(mode, data[ds_name], mx, probe, q)


BIG = [q for q in QUERIES if q.limit is not None or q.tag.startswith("distinct")]


@pytest.mark.parametrize("mode", ["table", "scan", "frame"])
@pytest.mark.parametrize("q", [q for q in BIG if q.tag not in ("distinct-star",)], ids=Q.id)
def test_row_queries_over_a_million_rows_with_ties_equal_duckdb(mx, probe, data, mode, q):
    check_query(mode, data["g1m"], mx, probe, q)


# ── the path that ran: the planner's notes ───────────────────────────────────────────────────────────

def notes_of(mode, ds, mx, probe, sql):
    return fetch(mode, ds, mx, probe, sql)[2]


def has(notes, prefix):
    return any(n.startswith(prefix) for n in notes)


@pytest.mark.parametrize("mode", ["table", "scan", "frame"])
def test_a_small_limit_over_many_rows_keeps_its_best_rows_in_heaps(mx, probe, data, mode):
    ds = data["g10k"]
    notes = notes_of(mode, ds, mx, probe, "SELECT id, x FROM t ORDER BY x DESC, id LIMIT 10")
    assert has(notes, "sink: rows"), notes
    assert has(notes, "order: top-10 by per-worker heaps in the hook ("), notes
    assert any(n.startswith("rows: the best 10 of ") and n.endswith("rows kept in per-worker heaps, 2 keys") for n in notes), notes
    assert not has(notes, "order: sorted"), notes


def test_a_limit_the_heaps_do_not_pay_for_sorts_all_the_rows(mx, probe, data):
    notes = notes_of("table", data["g8"], mx, probe, "SELECT id, x FROM t ORDER BY x, id LIMIT 3000")           # 20,000 rows are not 8x 3,000
    assert has(notes, "order: all rows sorted (top-3,000 wanted but the source has no more than 8x that many rows)"), notes
    assert any(n.startswith("order: sorted ") and n.endswith(" rows by 2 keys (stable radix passes)") for n in notes), notes
    notes = notes_of("table", data["g10k"], mx, probe, "SELECT id, x FROM t ORDER BY x, id LIMIT 5000")        # past the heaps' bound of 4,096
    assert has(notes, "order: all rows sorted (top-5,000 wanted but more than 4,096)"), notes


@pytest.mark.parametrize("mode", ["table", "scan"])
def test_a_limit_with_no_order_by_stops_the_scan_early(mx, probe, data, mode):
    notes = notes_of(mode, data["g1m"], mx, probe, "SELECT id, x FROM t WHERE x > 0 LIMIT 5")
    assert has(notes, "limit: no ORDER BY, so the scan stops once it has 5 rows"), notes
    selected = next(n for n in notes if n.startswith("rows: ") and " rows selected" in n)
    assert "stopped early" in selected, selected
    assert int(selected.split()[1].replace(",", "")) < 100_000, selected                       # not the 600,000 rows that qualify
    if mode == "scan":
        assert any(n.startswith("ran: scan t: 1 of 13 row groups read") for n in notes), notes


def test_ties_keep_the_source_order(mx, probe, data):
    ds = data["g10k"]
    for sql in ["SELECT id, k FROM t ORDER BY k", "SELECT id, k FROM t ORDER BY k LIMIT 300", "SELECT id, k FROM t ORDER BY k DESC NULLS FIRST LIMIT 3000"]:
        for mode in ["table", "scan"]:
            rows, _, _ = fetch(mode, ds, mx, probe, sql)
            for (i1, k1), (i2, k2) in zip(rows, rows[1:]):
                if k1 == k2:
                    assert i1 < i2, f"{mode}: {sql}: ids {i1}, {i2} of k = {k1} are out of source order"


def test_explain_of_a_row_query_shows_the_logical_plan_and_the_decisions(mx, probe, data):
    ds = data["g10k"]
    out = run_mx(mx, ["sql", "EXPLAIN SELECT id, x + 1 AS y FROM t WHERE k = 2 ORDER BY y DESC LIMIT 10 OFFSET 5"], {"t": ds.path})
    lines = [r[0] for r in out["rows"]]
    assert lines[:8] == ["Logical plan", "  Limit 10 offset 5", "    Sort y DESC", "      Project id, x + 1", "        Filter (WHERE) k = 2", "          Scan t",
                         "Physical plan", "  source: t (table), 200,000 rows"], lines
    assert "  sink: rows (the kernel's block hook writes the selected rows; text keeps its words)" in lines
    assert "  order: top-15 by per-worker heaps in the hook (200,000 rows, 15 wanted)" in lines
    out = run_mx(mx, ["sql", "EXPLAIN SELECT DISTINCT s, k FROM t"], {"t": ds.path})
    lines = [r[0] for r in out["rows"]]
    assert lines[1:4] == ["  Project s, k", "    Distinct keys=[s, k]", "      Scan t"], lines
    assert any(l.startswith("  group: dense, 2 keys s ") for l in lines), lines


ROW_ERRORS = [
    ("SELECT NULL FROM t", "SQL error at line 1, column 8: a bare NULL in the select list cannot be typed"),
    ("SELECT id FROM t ORDER BY 3", "SQL error at line 1, column 27: ORDER BY 3 is out of range: the select list has 1 items"),
    ("SELECT DISTINCT s FROM t ORDER BY x", "SQL error at line 1, column 35: for SELECT DISTINCT, ORDER BY expressions must appear in the select list: `x` does not"),
    ("SELECT idd FROM t", "SQL error at line 1, column 8: column `idd` not found in table `t` (did you mean `id`?)"),
    ("SELECT id FROM t ORDER BY idd", "SQL error at line 1, column 27: column `idd` not found in table `t` (did you mean `id`?)"),
    ("SELECT id FROM t WHERE sum(x) > 3", "SQL error at line 1, column 24: aggregate function sum() is not allowed here: aggregates belong in the select list, HAVING and ORDER BY"),
    ("SELECT z.* FROM t", "SQL error at line 1, column 8: unknown table or alias `z` in `z.*`"),
]


@pytest.mark.parametrize("sql,message", ROW_ERRORS, ids=[e[0][:40] for e in ROW_ERRORS])
def test_row_query_errors_say_what_and_where(probe, data, sql, message):
    res = run_probe(probe, sql, {"t": data["g8"].path}, expect_error=True)
    assert res.get("error") == message, res


@pytest.mark.parametrize("mode", ["frame", "framescan"])
def test_a_frames_formula_columns_come_out_of_row_queries(probe, data, mode):
    """Formulas (f1 = d - x, f2 = f1 * 2, f3 = d + e) after 300 edits: selected, filtered and ordered like stored columns, and part of `*`."""
    ds = data["g8"]
    script = ds.edits_path.with_suffix(".rowformulas")
    script.write_text(ds.edits_path.read_text() + "formula\tf1 = d - x\nformula\tf2 = f1 * 2\nformula\tf3 = d + e\n")
    con = ds.edited
    con.execute("create or replace view tf as select *, d - x as f1, (d - x) * 2 as f2, d + e as f3 from (select * exclude (rid) from src)")
    queries = [
        ("SELECT id, f1, f2, f3 FROM t WHERE f1 > 0", None),
        ("SELECT * FROM t", None),
        ("SELECT id, f2 FROM t ORDER BY f3 DESC NULLS LAST, id LIMIT 20", "SELECT id, f2 FROM tf ORDER BY f3 DESC NULLS LAST, id NULLS LAST LIMIT 20"),
        ("SELECT DISTINCT k, f3 FROM t WHERE f3 BETWEEN -5 AND 5", None),
    ]
    for sql, duck in queries:
        res = run_probe(probe, sql, {"t": ds.path}, mode=mode, edits=script)
        got = got_rows(res)
        cur = con.execute(duck or sql.replace(" FROM t", " FROM tf"))
        want = cur.fetchall()
        ordered = "ORDER BY" in sql
        if ordered:
            assert [tuple(r) for r in got] == [tuple(r) for r in want] or sorted(map(_canon, got), key=repr) == sorted(map(_canon, want), key=repr), f"{mode}: {sql}"
            assert [r[0] for r in got] == [r[0] for r in want], f"{mode}: {sql}: order"
        else:
            assert Counter(map(_canon, got)) == Counter(map(_canon, want)), f"{mode}: {sql}"
    star = run_probe(probe, "SELECT * FROM t", {"t": ds.path}, mode=mode, edits=script)
    assert [c["name"] for c in star["columns"]] == [c[0] for c in COLS] + ["f1", "f2", "f3"]


# ── random row queries ───────────────────────────────────────────────────────────────────────────────

from test_mojo_sql import _Gen  # noqa: E402


class _RowGen(_Gen):
    """Random row queries: expressions and conditions from the aggregate generator, ORDER BY keys among the selected columns (ordinals,
    with DESC and NULLS clauses), LIMIT / OFFSET, DISTINCT; seeded so a failure repeats."""

    OUTS = ["id", "k", "x", "g", "d", "e", "dt", "b", "s"]

    def typed(self, depth=0):
        """Like the aggregate generator's, but `%` only between integers: DuckDB widens a product of decimals past 18 digits and takes the
        remainder in doubles, where omxcore keeps the exact int64 decimal (the documented difference of decimal arithmetic past int64)."""
        r = self.rng
        if depth > 2 or r.random() < 0.35:
            c = r.choice(self.INTS + self.DECS + [str(r.randint(-5, 9)), f"{r.randint(-300, 300) / 100:.2f}"])
            return c, not (c in self.DECS or "." in c)
        a, ai = self.typed(depth + 1)
        b, bi = self.typed(depth + 1)
        op = r.choice(["+", "-", "*", "+", "-", "//", "%"])
        if op in ("//", "%") and not (ai and bi):
            op = "+"
        return f"({a} {op} {b})", ai and bi

    def query(self):
        r = self.rng
        outs = r.sample(self.OUTS, r.randint(1, 4))
        if r.random() < 0.5:
            outs.append(self.num(1))
        distinct = r.random() < 0.2
        sql = f"SELECT {'DISTINCT ' if distinct else ''}{', '.join(outs)} FROM t"
        if r.random() < 0.6:
            sql += f" WHERE {self.cond()}"
        keys = None
        duck = sql
        if r.random() < 0.7:
            sortable = [i for i, o in enumerate(outs) if o != "s"]
            if sortable:
                picked = r.sample(sortable, min(len(sortable), r.randint(1, 3)))
                terms, dterms = [], []
                for i in picked:
                    desc = r.random() < 0.4
                    nulls = r.choice(["", " NULLS FIRST", " NULLS LAST"])
                    terms.append(f"{i + 1}{' DESC' if desc else ''}{nulls}")
                    dterms.append(f"{i + 1}{' DESC' if desc else ''}{nulls or ' NULLS LAST'}")
                sql += " ORDER BY " + ", ".join(terms)
                duck += " ORDER BY " + ", ".join(dterms)
                keys = picked
        limit = offset = None
        if r.random() < 0.6:
            limit = r.choice([0, 1, 5, 40, 300, 5000, 100000])
            offset = r.choice([0, 0, 7, 600, 100000])
        return Q(core=sql, duck=duck, keys=keys, limit=limit, offset=offset or 0)


@pytest.mark.parametrize("mode", ["table", "scan", "frame"])
@pytest.mark.parametrize("seed", [21, 22, 23, 24])
def test_random_row_queries_equal_duckdb(mx, probe, data, mode, seed):
    gen = _RowGen(seed * 100 + len(mode))
    compared = 0
    for _ in range(25):
        q = gen.query()
        try:
            check_query(mode, data["g10k"] if seed % 2 else data["g8"], mx, probe, q)
        except AssertionError as e:
            if "passed the int64 range" in str(e):                     # a product of decimals past int64: omxcore raises where DuckDB widens
                continue
            raise
        compared += 1
    assert compared >= 15, f"only {compared} of 25 random queries ran"


# ── FLOAT in row queries (F-157 made it a column kind) ───────────────────────────────────────────────

FLOAT_QUERIES = [
    Q("SELECT id, x / 2 AS h, d / 3 AS t3, x * 1.5 AS m FROM t WHERE x IS NOT NULL", tag="float-expressions"),
    Q("SELECT id, x / 3 AS q FROM t ORDER BY q DESC, id", duck="SELECT id, x / 3 AS q FROM t ORDER BY q DESC NULLS LAST, id NULLS LAST", keys=[1, 0], limit=40,
      tag="order-by-a-float-expression-top-k"),
    Q("SELECT id, d / e AS r FROM t ORDER BY r NULLS FIRST, id", duck="SELECT id, d / e AS r FROM t ORDER BY r NULLS FIRST, id NULLS LAST", keys=[1, 0],
      tag="order-by-a-float-expression-full-sort"),
    Q("SELECT DISTINCT x / 7 AS q FROM t", tag="distinct-float"),
    Q("SELECT id, x / 0 AS inf, 0 * (x / 0) AS nan FROM t WHERE x IS NOT NULL AND id < 2000", tag="inf-and-nan"),
    Q("SELECT id FROM t WHERE x / 4 > d / 3", tag="float-comparison"),
]


@pytest.mark.parametrize("mode", ["table", "scan", "frame"])
@pytest.mark.parametrize("q", FLOAT_QUERIES, ids=Q.id)
def test_float_expressions_in_row_queries_equal_duckdb(mx, probe, data, mode, q):
    check_query(mode, data["g8"], mx, probe, q)


def test_a_float_column_is_selected_ordered_and_distinct(mx, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import duckdb
    vals = [1.5, -0.0, 0.0, None, float("nan"), float("inf"), float("-inf"), 2.25, 1.5, -7.0, float("nan"), None, 3.0] * 400
    ids = list(range(len(vals)))
    path = tmp_path / "f.parquet"
    pq.write_table(pa.table({"id": ids, "f": pa.array(vals, pa.float64())}), path, row_group_size=900)
    con = oracle_connection()
    for sql, duck, keys in [
        ("SELECT * FROM t", None, None),
        ("SELECT f, id FROM t ORDER BY f, id", "SELECT f, id FROM t ORDER BY f NULLS LAST, id", [0, 1]),
        ("SELECT f, id FROM t ORDER BY f DESC NULLS FIRST, id LIMIT 30", "SELECT f, id FROM t ORDER BY f DESC NULLS FIRST, id", [0, 1]),
        ("SELECT DISTINCT f FROM t", None, None),
        ("SELECT id, f + 1 AS g FROM t WHERE f > 1", None, None),
    ]:
        for stream in ("never", "always"):
            out = run_mx(mx, ["sql", "--max-rows", "0", "--stream", stream, sql], {"t": path})
            got = typed_rows(out)
            want = con.execute((duck or sql).replace(" FROM t", f" FROM read_parquet('{path}')")).fetchall()
            if " LIMIT " in sql:
                want = want[:30]
            if keys is not None:
                assert [_canon(r) for r in got] == [_canon(r) for r in want], f"{stream}: {sql}"
            else:
                assert Counter(map(_canon, got)) == Counter(map(_canon, want)), f"{stream}: {sql}"


# ── an edited Frame's rows come in the frame's order ─────────────────────────────────────────────────

FRAME_ORDER = [
    ("SELECT id, k, x FROM t", "SELECT id, k, x FROM src ORDER BY rid"),
    ("SELECT id, k, x FROM t WHERE x > 0", "SELECT id, k, x FROM src WHERE x > 0 ORDER BY rid"),
    ("SELECT id, k, x FROM t LIMIT 25", "SELECT id, k, x FROM src ORDER BY rid LIMIT 25"),
    ("SELECT id, k, x FROM t LIMIT 12 OFFSET 20000", "SELECT id, k, x FROM src ORDER BY rid LIMIT 12 OFFSET 20000"),
    ("SELECT id, k, x FROM t ORDER BY k", "SELECT id, k, x FROM src ORDER BY k NULLS LAST, rid"),
    ("SELECT id, k, x FROM t ORDER BY k LIMIT 50 OFFSET 7", "SELECT id, k, x FROM src ORDER BY k NULLS LAST, rid LIMIT 50 OFFSET 7"),
    ("SELECT id, k, x FROM t ORDER BY k DESC NULLS FIRST LIMIT 4000 OFFSET 96", "SELECT id, k, x FROM src ORDER BY k DESC NULLS FIRST, rid LIMIT 4000 OFFSET 96"),
    ("SELECT id, k, x FROM t ORDER BY k LIMIT 4097", "SELECT id, k, x FROM src ORDER BY k NULLS LAST, rid LIMIT 4097"),
]


@pytest.mark.parametrize("mode", ["frame", "framescan"])
@pytest.mark.parametrize("sql,duck", FRAME_ORDER, ids=[q[0][:60] for q in FRAME_ORDER])
def test_an_edited_frames_rows_come_in_the_frames_order(probe, data, mode, sql, duck):
    """After 300 edits (sets, nulls, appended and removed rows) a query with no ORDER BY lists the rows as the frame holds them (a changed
    row where it was, an appended row at the end), and equal sort keys keep that order -- exactly DuckDB's `ORDER BY rid`."""
    ds = data["g8"]
    res = run_probe(probe, sql, {"t": ds.path}, mode=mode, edits=ds.edits_path)
    want = ds.edited.execute(duck).fetchall()
    got = got_rows(res)
    assert [_canon(r) for r in got] == [_canon(r) for r in want], f"{mode}: {sql}"
