"""F-011 engine correctness: counts, nulls, limits, caches (see tests/_f011_scenarios.py).

``cpu`` scenarios run in-process; ``gpu`` and ``auto`` in a subprocess (a fault must not
take the session down).
"""
import subprocess
import sys
from pathlib import Path

import pytest

import _f011_scenarios as sc
from omxcore.aot_kernels import GPU_AOT_AVAILABLE
from omxcore.pipeline_kernels import cuda_usable

SCRIPT = Path(__file__).with_name("_f011_scenarios.py")


def _run(name, device):
    if device == "cpu":
        sc.SCENARIOS[name](device)
        return
    if not GPU_AOT_AVAILABLE or not cuda_usable():
        pytest.skip("no GPU AOT kernels, or no CUDA device to run them (the Python path's GPU is NVIDIA only)")
    r = subprocess.run([sys.executable, str(SCRIPT), name, device], capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
@pytest.mark.parametrize("name", [
    "count_exact_beyond_2p24",
    "count_exact_beyond_2p24_unfused",
    "count_types_all_paths",
    "nulls_filter_drops_null_comparisons",
    "nulls_group_key_is_its_own_group",
    "nulls_global_aggregates_skip_nulls",
    "string_ne_with_null_keys",
    "sort_with_null_keys",
    "limit_below_aggregate_is_applied",
    "multi_key_sort_order",
    "join_keys_of_different_integer_widths",
    "join_cache_is_keyed_by_device",
    "case_when_sums_are_float64",
    "min_max_identity_with_infinities",
    "float32_column_keeps_boundary_rows",
    "gpu_scratch_does_not_reserve_the_device",
    "join_duplicate_names_get_a_right_suffix",
    "self_join_then_aggregate",
    "literal_aggregates_with_nulls",
])
def test_scenario(name, device):
    _run(name, device)


# --- gathering strings ---------------------------------------------------------------------

def test_take_rows_slices_string_columns_by_bytes(monkeypatch):
    import numpy as np
    import pyarrow as pa
    from omxcore import np_arrow
    monkeypatch.setattr(np_arrow, "TAKE_CHUNK_BYTES", 400)
    monkeypatch.setattr(np_arrow, "TAKE_SLICE_MIN_ROWS", 10)
    t = pa.table({"s": pa.array([f"v{i}" * (1 + i % 5) for i in range(1000)]),
                  "k": pa.array(np.arange(1000, dtype=np.int64))})
    idx = np.random.default_rng(1).integers(0, 1000, 950)
    got = np_arrow.take_rows(t, idx)
    assert got.equals(t.take(pa.array(idx)))
    assert got.column("s").num_chunks > 5 and got.column("k").num_chunks == 1     # only strings are sliced
    for c in got.column("s").chunks:
        assert sum(len(v.as_py()) for v in c) <= 400 + 25                          # a slice stays near the budget
    col = np_arrow.take_rows(t.column("s").combine_chunks(), idx)
    assert col.to_pylist() == [t.column("s")[int(i)].as_py() for i in idx]
    # structs holding strings are sliced by rows and validated
    monkeypatch.setattr(np_arrow, "TAKE_CHUNK_ROWS", 100)
    st = pa.table({"x": pa.array([{"a": f"s{i}", "b": i} for i in range(1000)])})
    assert np_arrow.take_rows(st, idx).equals(st.take(pa.array(idx)))


@pytest.mark.slow
def test_take_rows_survives_more_than_2gib_of_long_strings():
    """Arrow's take returns a corrupt column (`Negative offsets in binary array`, seen only by
    validate(full=True)) past 2 GiB of string data, however few rows that is: a lineitem-to-orders
    join at SF10 gathers 3 GB of o_comment; 3.2M rows of 703 bytes are 2.25 GB."""
    import numpy as np
    import pyarrow as pa
    from omxcore import np_arrow
    n = 100_000
    t = pa.table({"s": pa.array(["x" * 700 + f"{i % 1000:03d}" for i in range(n)])})    # 70 MB
    idx = np.tile(np.arange(n, dtype=np.int64), 32)                                      # 3.2M rows, 2.25 GB
    out = np_arrow.take_rows(t, idx)
    out.validate(full=True)
    assert out.num_rows == 32 * n and out.column("s")[32 * n - 1].as_py().endswith("999")
    st = pa.table({"x": pa.array([{"a": "y" * 700, "b": i} for i in range(n)])})
    out = np_arrow.take_rows(st, idx)
    out.validate(full=True)
