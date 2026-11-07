#!/usr/bin/env python3
"""
audit_gpu_paths.py — Execution-path coverage audit for all 22 TPC-H queries.

Instruments LazyFrame.compute() to record `last_compile_provenance` for every
compute call inside each query, then classifies each query:

  GPU-CLEAN : every compute call ran a GPU-resident path
              (gpu_aot, masked_global_agg_mojo on gpu, compiled on gpu)
  MOJO-CPU  : every compute call ran a Mojo-backed path, but on CPU
              (cpu_aot, masked_global_agg_mojo on cpu, compiled on cpu)
  FALLBACK  : at least one compute call used a PyArrow/NumPy path
              (pyarrow_shortcut, global_agg_pyarrow)

This is the regression guard for the "GPU-accelerated DataFrame" claim:
run it after every kernel/dispatch change. Use --min-gpu-coverage in CI to
fail the build when coverage drops.

Usage:
  python scripts/audit_gpu_paths.py                      # CPU audit (Mojo coverage)
  python scripts/audit_gpu_paths.py --device gpu         # GPU audit (the real claim)
  python scripts/audit_gpu_paths.py --device gpu --runs 3 --min-gpu-coverage 0.8
"""

import functools
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

from scripts import benchmark_tpch as b
from omxcore.lazy_frame import LazyFrame

# Paths considered GPU-resident vs Mojo-CPU vs Python-side fallback.
GPU_PATHS = {"gpu_aot"}
MOJO_PATHS = {"cpu_aot", "masked_global_agg_mojo",
              "join_mojo_shortcut"}  # join index generation ran in Mojo
COMPILED_PATH = "compiled"                            # MAX graph; device field decides
FALLBACK_PATHS = {"pyarrow_shortcut", "global_agg_pyarrow", "cpu_fallback"}  # cpu_fallback: F-214, no loadable GPU library

_records: list = []
_orig_compute = LazyFrame.compute


def _wrapped_compute(self, *args, **kwargs):
    out = _orig_compute(self, *args, **kwargs)
    _records.append(dict(self.last_compile_provenance or {}))
    return out


LazyFrame.compute = _wrapped_compute


def classify(records: list, device: str) -> str:
    """Classify a query run from its per-compute provenance records."""
    if not records:
        return "NO-COMPUTE"
    saw_fallback = False
    saw_cpu = False
    for r in records:
        path = r.get("path")
        dev = r.get("device")
        if path in FALLBACK_PATHS:
            saw_fallback = True
        elif path in GPU_PATHS:
            pass  # gpu-resident
        elif path == COMPILED_PATH or path in MOJO_PATHS:
            if dev != "gpu":
                saw_cpu = True
        else:
            saw_fallback = True  # unknown path — treat as fallback, force a look
    if saw_fallback:
        return "FALLBACK"
    if device == "gpu" and saw_cpu:
        return "MOJO-CPU"
    return "GPU-CLEAN" if device == "gpu" else "MOJO-CPU"


def build_queries(s: int, N: int):
    """Return {name: zero-arg callable} for all 22 queries at scale s."""
    li = b.make_lineitem(N)
    c3, o3, l3 = b.make_tpch_q3_tables(15_000 * s, 150_000 * s, 600_000 * s)
    o12, l12 = b.make_tpch_q12_tables(50_000 * s, 300_000 * s)
    p14, l14 = b.make_tpch_q14_tables(20_000 * s, 200_000 * s)
    na5, c5, o5, l5 = b.make_tpch_q5_tables(15_000 * s, 100_000 * s, 400_000 * s)
    na10, c10, o10, l10 = b.make_tpch_q10_tables(15_000 * s, 150_000 * s, 600_000 * s)
    na7, s7, c7, o7, l7 = b.make_tpch_q7_tables(2_000 * s, 10_000 * s, 80_000 * s, 200_000 * s)
    na8, r8, p8, c8, o8, l8 = b.make_tpch_q8_tables(5_000 * s, 10_000 * s, 80_000 * s, 200_000 * s)
    c13, o13 = b.make_tpch_q13_tables(150_000 * s, 600_000 * s)
    p19, l19 = b.make_tpch_q19_tables(20_000 * s, 200_000 * s)
    o4, l4 = b.make_tpch_q4_tables(150_000 * s, 600_000 * s)
    na9, s9, ps9, p9, o9, l9 = b.make_tpch_q9_tables(5_000 * s, 2_000 * s, 80_000 * s, 200_000 * s)
    na11, s11, ps11 = b.make_tpch_q11_tables()
    c18, o18, l18 = b.make_tpch_q18_tables(15_000 * s, 60_000 * s, 300_000 * s)
    p16, s16, ps16 = b.make_tpch_q16_tables()
    p17, l17 = b.make_tpch_q17_tables(5_000 * s, 200_000 * s)
    na2, r2, p2, s2, ps2 = b.make_tpch_q2_tables()
    s15, l15 = b.make_tpch_q15_tables(2_000 * s, 200_000 * s)
    na20, p20, s20, ps20, l20 = b.make_tpch_q20_tables(5_000 * s, 2_000 * s, 100_000 * s)
    na21, s21, o21, l21 = b.make_tpch_q21_tables(2_000 * s, 60_000 * s, 200_000 * s)
    c22, o22 = b.make_tpch_q22_tables(150_000 * s, 400_000 * s)

    def q(fn, *tables):
        return lambda device: fn(*tables, device)

    return {
        "Q1":  q(b.run_q1_mxframe, li),
        "Q2":  q(b.run_q2_mxframe, na2, r2, p2, s2, ps2),
        "Q3":  q(b.run_q3_mxframe, c3, o3, l3),
        "Q4":  q(b.run_q4_mxframe, o4, l4),
        "Q5":  q(functools.partial(b.run_q5_mxframe, supplier=b.make_synthetic_supplier()), na5, c5, o5, l5),
        "Q6":  q(b.run_q6_mxframe, li),
        "Q7":  q(b.run_q7_mxframe, na7, s7, c7, o7, l7),
        "Q8":  q(functools.partial(b.run_q8_mxframe, supplier=b.make_synthetic_supplier()), na8, r8, p8, c8, o8, l8),
        "Q9":  q(b.run_q9_mxframe, na9, s9, ps9, p9, o9, l9),
        "Q10": q(b.run_q10_mxframe, na10, c10, o10, l10),
        "Q11": q(b.run_q11_mxframe, na11, s11, ps11),
        "Q12": q(b.run_q12_mxframe, o12, l12),
        "Q13": q(b.run_q13_mxframe, c13, o13),
        "Q14": q(b.run_q14_mxframe, p14, l14),
        "Q15": q(b.run_q15_mxframe, s15, l15),
        "Q16": q(b.run_q16_mxframe, p16, s16, ps16),
        "Q17": q(b.run_q17_mxframe, p17, l17),
        "Q18": q(b.run_q18_mxframe, c18, o18, l18),
        "Q19": q(b.run_q19_mxframe, p19, l19),
        "Q20": q(b.run_q20_mxframe, na20, p20, s20, ps20, l20),
        "Q21": q(b.run_q21_mxframe, na21, s21, o21, l21),
        "Q22": q(b.run_q22_mxframe, c22, o22),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="TPC-H execution-path coverage audit")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--rows", type=int, default=1_000_000,
                    help="lineitem rows for Q1/Q6 (default 1M)")
    ap.add_argument("--scale", type=int, default=1,
                    help="scale factor for multi-table generators")
    ap.add_argument("--runs", type=int, default=1,
                    help="timed runs per query after 1 warmup (median reported)")
    ap.add_argument("--min-gpu-coverage", type=float, default=None,
                    help="exit non-zero if GPU-CLEAN fraction < threshold (gpu device only)")
    ap.add_argument("--min-mojo-coverage", type=float, default=None,
                    help="exit non-zero if non-FALLBACK fraction < threshold")
    args = ap.parse_args()

    print(f"Building tables (rows={args.rows:,}, scale={args.scale}) ...", flush=True)
    queries = build_queries(args.scale, args.rows)

    results = []
    for name, run in queries.items():
        global _records
        status, detail, ms = "ERR", "", None
        try:
            # Warmup (also captures provenance for classification)
            _records = []
            b._DIRECT_PROVENANCE.clear()
            run(args.device)
            records = [*_records, *b._DIRECT_PROVENANCE]
            status = classify(records, args.device)
            detail = " ".join(
                f"{r.get('path')}@{r.get('device')}" for r in records
            )
            # Timed hot runs
            samples = []
            for _ in range(args.runs):
                _records = []
                b._DIRECT_PROVENANCE.clear()
                t0 = time.perf_counter()
                run(args.device)
                samples.append((time.perf_counter() - t0) * 1000.0)
            samples.sort()
            ms = samples[len(samples) // 2]
        except Exception as e:
            detail = f"{type(e).__name__}: {str(e)[:80]}"
        results.append((name, status, ms, detail))
        ms_s = f"{ms:8.1f}ms" if ms is not None else "       —"
        print(f"  {name:<4} {status:<10} {ms_s}  {detail}")

    n = len(results)
    n_gpu = sum(1 for _, st, _, _ in results if st == "GPU-CLEAN")
    n_mojo = sum(1 for _, st, _, _ in results if st in ("GPU-CLEAN", "MOJO-CPU"))
    n_fb = sum(1 for _, st, _, _ in results if st == "FALLBACK")
    n_err = sum(1 for _, st, _, _ in results if st == "ERR")

    print("\n" + "=" * 60)
    print(f"  Device          : {args.device}")
    print(f"  GPU-clean       : {n_gpu}/{n}  ({n_gpu / n:.0%})")
    print(f"  Mojo (any dev)  : {n_mojo}/{n}  ({n_mojo / n:.0%})")
    print(f"  Fallback        : {n_fb}/{n}")
    print(f"  Errors          : {n_err}/{n}")
    print("=" * 60)

    if args.min_gpu_coverage is not None and n_gpu / n < args.min_gpu_coverage:
        print(f"FAIL: GPU coverage {n_gpu / n:.0%} < required {args.min_gpu_coverage:.0%}")
        return 1
    if args.min_mojo_coverage is not None and n_mojo / n < args.min_mojo_coverage:
        print(f"FAIL: Mojo coverage {n_mojo / n:.0%} < required {args.min_mojo_coverage:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
