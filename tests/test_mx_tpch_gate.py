"""F-162, the M8 gate (1/2): the 22 TPC-H queries, as DuckDB's tpch extension writes them, through the real `mx sql` over typed
Parquet (DuckDB's dbgen at SF1: DECIMAL(15,2), DATE, written once to ~/.cache/omxcore-gate) equal DuckDB: INT / DECIMAL / DATE / TEXT
exactly, FLOAT to 1e-9, in ORDER BY order (tests/gate_compare.py).

A query the engine cannot run yet is `xfail(strict=True)` with the engine's own error in GAPS: when a gap closes the test XPASSes,
strict makes that a failure, and the entry must be deleted (that is how F-159 / F-155 show up here). `pixi run test --runxfail
tests/test_mx_tpch_gate.py` shows each gap's real error. Report: docs/bench/f162-coverage.md."""
import sys
from pathlib import Path

import pytest

from gate_compare import compare_with_duckdb
from mx_oracle import mx_binary

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gate_common as gate  # noqa: E402

SF = 1

# query -> the engine's error today. Keep the text exact (the xfail reason is what the F-159 / F-155 reviewer greps).
GAPS = {
    2: "a scalar subquery is not supported yet (F-159)",
    7: "a subquery in FROM is not supported yet (F-159)",
    8: "a subquery in FROM is not supported yet (F-159)",
    9: "a subquery in FROM is not supported yet (F-159)",
    11: "a scalar subquery is not supported yet (F-159)",
    13: "a subquery in FROM is not supported yet (F-159)",
    14: "a sum or an expression passed the int64 range (about 9.2e18), or an integer past 2^53 met a float64 (it is not exact as a double)",
    15: "`WITH` (common table expressions) is not supported yet (F-159)",
    16: "a subquery in a query that also joins tables is not supported yet (F-155)",
    17: "a scalar subquery is not supported yet (F-159)",
    18: "`GROUP` in a subquery is not supported yet (F-159)",
    20: "a scalar subquery is not supported yet (F-159)",
    21: "more than one subquery in WHERE is not supported yet (F-155)",
    22: "a subquery in FROM is not supported yet (F-159)",
}


@pytest.fixture(scope="module")
def world():
    paths = gate.table_paths(SF)
    return mx_binary(), paths, gate.duck_views(paths), gate.queries()


def _param(q):
    if q in GAPS:
        return pytest.param(q, marks=pytest.mark.xfail(strict=True, reason=GAPS[q]), id=f"Q{q}")
    return pytest.param(q, id=f"Q{q}")


@pytest.mark.parametrize("q", [_param(q) for q in range(1, 23)])
def test_tpch_query_equals_duckdb(world, q):
    mx, paths, con, qs = world
    compare_with_duckdb(mx, qs[q], paths, con, label=f"TPC-H Q{q} SF{SF}")
