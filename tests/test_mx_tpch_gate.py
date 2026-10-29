"""F-162, the M8 gate (1/2): the 22 TPC-H queries, as DuckDB's tpch extension writes them, through the real `mx sql` over typed
Parquet (DuckDB's dbgen at SF1: DECIMAL(15,2), DATE, written once to ~/.cache/omxcore-gate) equal DuckDB: INT / DECIMAL / DATE / TEXT
exactly, FLOAT to 1e-9, in ORDER BY order (tests/gate_compare.py).

A query the engine cannot run yet is xfailed with the engine's own error in GAPS, and only for exactly that error: a different
failure (a wrong answer, another error) fails the test, and so does a pass (the gap closed: delete the entry) (that is how F-159 / F-155 show up here). `pixi run test --runxfail
tests/test_mx_tpch_gate.py` shows each gap's real error. Report: docs/bench/f162-coverage.md."""
import sys
from pathlib import Path

import pytest

from gate_compare import compare_with_duckdb, gap_must_still_fail_with
from mx_oracle import mx_binary

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gate_common as gate  # noqa: E402

SF = 1

# query -> the engine's error today. Keep the text exact (the xfail reason is what the F-159 / F-155 reviewer greps).
GAPS = {
    11: "a sum or an expression passed the int64 range (about 9.2e18)",       # F-192
}


@pytest.fixture(scope="module")
def world():
    paths = gate.table_paths(SF)
    return mx_binary(), paths, gate.duck_views(paths), gate.queries()


@pytest.mark.parametrize("q", range(1, 23), ids=lambda q: f"Q{q}")
def test_tpch_query_equals_duckdb(world, q):
    mx, paths, con, qs = world
    if q in GAPS:
        gap_must_still_fail_with(GAPS[q], lambda: compare_with_duckdb(mx, qs[q], paths, con, label=f"TPC-H Q{q} SF{SF}"))
        return
    compare_with_duckdb(mx, qs[q], paths, con, label=f"TPC-H Q{q} SF{SF}")
