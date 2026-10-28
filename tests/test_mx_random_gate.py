"""F-162, the M8 gate (2/2): the random query suite (scripts/bench_random/gen.py, the F-044 / F-058 generator: filters, joins along the
foreign keys, group-bys, HAVING, top-N, COUNT(DISTINCT), SELECT DISTINCT) rendered as SQL over the typed TPC-H Parquet (SF1, see
scripts/gate_common.py), through the real `mx sql` and DuckDB, compared as in tests/test_mx_tpch_gate.py.

A seed the engine cannot run yet is xfailed with the engine's own error in GAPS, and only for exactly that error (another failure, or a
pass because the gap closed, fails the test). `pixi run test --runxfail tests/test_mx_random_gate.py` shows each real error."""
import sys
from pathlib import Path

import pytest

from gate_compare import compare_json_with_duckdb, compare_with_duckdb, gap_must_still_fail_with
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


@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed{s}")
def test_random_query_equals_duckdb(world, seed):
    mx, paths, con, stats = world
    sql = gate.random_sql(seed, stats)
    run = lambda: compare_with_duckdb(mx, sql, paths, con, label=f"random seed {seed}: {sql}")
    if seed in GAPS:
        gap_must_still_fail_with(GAPS[seed], run)
    else:
        run()


def test_the_comparison_sees_a_different_answer(world):
    """The gate is only worth something if it fails on a wrong answer: plant each kind of difference in mx's JSON, compare with DuckDB."""
    import copy
    from mx_oracle import run_mx
    mx, paths, con, _ = world
    sql = "SELECT r_regionkey, r_name, r_comment FROM region ORDER BY r_regionkey"
    good = run_mx(mx, ["sql", "--max-rows", "0", sql], {"region": paths["region"]})
    compare_json_with_duckdb(good, sql, con)                                   # the untouched answer is equal

    def planted(change):
        res = copy.deepcopy(good)
        change(res)
        return res

    def wrong_value(r): r["rows"][0][1] = "NOWHERE"
    def null_for_value(r): r["rows"][0][2] = None
    def order_swapped(r): r["rows"].reverse()
    def fewer_rows(r): r["rows"].pop()
    def wrong_kind(r): r["columns"][0]["kind"] = "TEXT"
    for change, why in ((wrong_value, r"row 0"), (null_for_value, r"row 0"), (order_swapped, r"row 0"),
                        (fewer_rows, r"4 rows, DuckDB has 5"), (wrong_kind, r"column 0")):
        with pytest.raises(AssertionError, match=why):
            compare_json_with_duckdb(planted(change), sql, con)
    sql = "SELECT c_acctbal FROM customer ORDER BY c_custkey LIMIT 3"
    res = run_mx(mx, ["sql", "--max-rows", "0", sql], {"customer": paths["customer"]})
    with pytest.raises(AssertionError, match=r"DECIMAL\(3\)"):                  # a wrong DECIMAL scale
        compare_json_with_duckdb(planted_scale(res), sql, con)


def planted_scale(res):
    import copy
    res = copy.deepcopy(res)
    res["columns"][0]["scale"] = 3
    return res
