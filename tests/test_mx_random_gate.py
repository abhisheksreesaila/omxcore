"""F-162, the M8 gate (2/2): the random query suite (scripts/bench_random/gen.py, the F-044 / F-058 generator: filters, joins along the
foreign keys, group-bys, HAVING, top-N, COUNT(DISTINCT), SELECT DISTINCT) rendered as SQL over the typed TPC-H Parquet (SF1, see
scripts/gate_common.py), through the real `mx sql` and DuckDB, compared as in tests/test_mx_tpch_gate.py.

A seed the engine cannot run yet is `xfail(strict=True)` with the engine's own error in GAPS (when a gap closes the test XPASSes, strict
makes that a failure and the entry goes). `pixi run test --runxfail tests/test_mx_random_gate.py` shows each real error."""
import sys
from pathlib import Path

import pytest

from gate_compare import compare_with_duckdb
from mx_oracle import mx_binary

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gate_common as gate  # noqa: E402

SF = 1
SEEDS = range(80)
GAPS = {}


@pytest.fixture(scope="module")
def world():
    paths = gate.table_paths(SF)
    return mx_binary(), paths, gate.duck_views(paths), gate.random_stats()


def _param(seed):
    if seed in GAPS:
        return pytest.param(seed, marks=pytest.mark.xfail(strict=True, reason=GAPS[seed]), id=f"seed{seed}")
    return pytest.param(seed, id=f"seed{seed}")


@pytest.mark.parametrize("seed", [_param(s) for s in SEEDS])
def test_random_query_equals_duckdb(world, seed):
    mx, paths, con, stats = world
    sql = gate.random_sql(seed, stats)
    compare_with_duckdb(mx, sql, paths, con, label=f"random seed {seed}: {sql}")


def test_the_comparison_sees_a_different_answer(world):
    """The gate is only worth something if it fails on a wrong answer: DuckDB gets a region table with a row missing."""
    mx, paths, con, _ = world
    con.execute("CREATE OR REPLACE VIEW region AS SELECT * FROM read_parquet('%s') WHERE r_regionkey < 4" % paths["region"])
    with pytest.raises(AssertionError, match=r"5 rows, DuckDB has 4"):
        compare_with_duckdb(mx, "SELECT r_regionkey, r_name FROM region ORDER BY r_regionkey", paths, con)
    con.execute("CREATE OR REPLACE VIEW region AS SELECT * FROM read_parquet('%s')" % paths["region"])
