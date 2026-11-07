"""F-214: the GPU kernel library cannot be loaded on this machine (a Mac's committed file is x86-64 ELF, Metal builds none):
the hand-written TPC-H queries on device='gpu' answer on the CPU and say so, they do not crash with an OSError."""
import sys
from pathlib import Path

import pyarrow as pa
import pytest

import omxcore.aot_kernels as ak

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import benchmark_tpch as b  # noqa: E402


@pytest.fixture
def unloadable_gpu_lib(tmp_path, monkeypatch):
    bad = tmp_path / "libmxkernels_aot_gpu.so"
    bad.write_bytes(b"not a shared library")
    monkeypatch.setattr(ak, "_gpu_lib_path", bad)
    monkeypatch.setattr(ak, "GPU_AOT_AVAILABLE", True)
    monkeypatch.setattr(b, "GPU_AOT_AVAILABLE", True)
    monkeypatch.setattr(ak, "_aot_gpu_singleton", None)
    monkeypatch.setattr(ak, "_aot_gpu_failed", False)
    monkeypatch.setattr(b, "_BENCH_AOT_GPU", None)
    b._DIRECT_PROVENANCE.clear()
    yield bad
    b._DIRECT_PROVENANCE.clear()


NOTE = {"path": "cpu_fallback", "device": "cpu",
        "note": "GPU kernel library not loadable on this machine: ran on the CPU"}


@pytest.mark.parametrize("make,run", [
    (lambda: b.make_tpch_q4_tables(n_orders=2000, n_lineitem=8000), b.run_q4_mxframe),
    (lambda: b.make_tpch_q13_tables(n_customers=500, n_orders=3000), b.run_q13_mxframe),
])
def test_unloadable_gpu_library_answers_on_the_cpu_with_a_note(unloadable_gpu_lib, make, run):
    tables = make()
    want = run(*tables, device="cpu")
    b._DIRECT_PROVENANCE.clear()
    got = run(*tables, device="gpu")
    assert isinstance(got, pa.Table)
    assert got.to_pydict() == want.to_pydict()
    assert NOTE in b._DIRECT_PROVENANCE
