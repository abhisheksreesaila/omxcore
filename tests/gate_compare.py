"""F-162: the M8 gate's comparison. A SQL text goes through the real `mx sql` (over Parquet files) and through DuckDB (over the same
files); the answers must be equal: INT / DECIMAL / DATE / TEXT / BOOL exactly (kinds and DECIMAL scales too), FLOAT to a relative
1e-9 (the engine's parallel sums are not bitwise reproducible, see docs/lessons.md), rows in order when the query has an ORDER BY
and as a multiset when it has none."""
import math
import re
import sys
from pathlib import Path

from mx_oracle import duck_kind, run_mx, typed_rows

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from gate_common import TABLES  # noqa: E402

FLOAT_REL = 1e-9


def tables_named(sql):
    """The tables a query text names (the run registers only those: mx reads each registered file whole)."""
    return [t for t in TABLES if re.search(rf"\b{t}\b", sql)]


def has_order_by(sql):
    import sqlglot
    return sqlglot.parse_one(sql, dialect="duckdb").args.get("order") is not None


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        a, b = float(a), float(b)
        return (a != a and b != b) or a == b or math.isclose(a, b, rel_tol=FLOAT_REL, abs_tol=0.0)
    return a == b


def _key(row):
    return tuple((0, "") if v is None else (1, repr(round(v, 6)) if isinstance(v, float) else repr(v)) for v in row)


def compare_json_with_duckdb(res, sql, con, label=None):
    """`res` is `mx sql --max-rows 0`'s parsed JSON. Raises AssertionError saying what differs from DuckDB's answer to `sql`."""
    label = label or sql[:60]
    assert res and "columns" in res and not res["truncated"], f"{label}: no complete result: {res}"
    cur = con.execute(sql)
    want = cur.fetchall()
    assert len(res["columns"]) == len(cur.description), f"{label}: {len(res['columns'])} columns, DuckDB has {len(cur.description)}"
    for i, desc in enumerate(cur.description):
        kind, scale = duck_kind(desc[1])
        c = res["columns"][i]
        assert c["kind"] == kind and (kind != "DECIMAL" or c["scale"] == scale), \
            f"{label}: column {i} ({c['name']}) is {c['kind']}({c['scale']}), DuckDB has {desc[1]}"
    got = typed_rows(res)
    assert len(got) == len(want), f"{label}: {len(got)} rows, DuckDB has {len(want)}"
    if not has_order_by(sql):
        got, want = sorted(got, key=_key), sorted(want, key=_key)
    for i, (a, b) in enumerate(zip(got, want)):
        assert len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b)), f"{label}: row {i}: got {a}, DuckDB has {b}"


def compare_with_duckdb(mx, sql, paths, con, label=None, extra=()):
    """Runs `sql` through `mx sql` over the Parquet files it names, then compares with DuckDB; returns mx's JSON when equal."""
    res = run_mx(mx, ["sql", "--max-rows", "0", *extra, sql], {n: paths[n] for n in tables_named(sql)})
    compare_json_with_duckdb(res, sql, con, label)
    return res


def gap_must_still_fail_with(error, run):
    """A known gap: xfail when `run()` fails with exactly this engine error; any other failure, or a pass, fails the test."""
    import pytest
    try:
        run()
    except AssertionError as e:
        assert error in str(e), f"the engine's error is no longer `{error}`: {str(e)[:600]}"
        pytest.xfail(error)
    pytest.fail(f"the gap `{error}` is closed: delete its entry from GAPS")
