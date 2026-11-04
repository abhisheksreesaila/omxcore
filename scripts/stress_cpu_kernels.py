#!/usr/bin/env python3
"""Stress the CPU kernels the way a loaded machine does (F-020, rare crashes under heavy load).

Several processes, each running several Python threads that call the native kernels concurrently
(ctypes releases the GIL, so the kernels of the four libraries - libmxkernels_aot, libmxpipeline_cpu,
libmxgroupkeys_cpu, libmxjoin_cpu - share one Mojo worker pool per process), and check every result
against NumPy.  A worker that dies by signal, or a wrong result, is reported with the seed/iteration.

    pixi run python scripts/stress_cpu_kernels.py --procs 6 --threads 4 --seconds 120
    pixi run python scripts/stress_cpu_kernels.py --child --threads 4 --seconds 20     # one process

Exit status 0 when every process finished cleanly.
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _import_mxframe():
    """`import omxcore` from THIS checkout (the repo root is the package; a worktree may have another name)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import scratch
    shim = scratch.shim_dir(ROOT)
    link = shim / "omxcore"
    if not link.exists():
        link.symlink_to(ROOT)
    sys.path.insert(0, str(shim))
    import omxcore
    return omxcore


# ── the workloads: each returns None or raises AssertionError ───────────────

def w_sort(rng, np, aot, **_):
    n = int(rng.integers(50_000, 3_000_000))
    kind = rng.integers(0, 3)
    if kind == 0:
        k = rng.integers(-1000, 1000, n).astype(np.int32)
    elif kind == 1:
        k = rng.integers(0, 10**12, n).astype(np.int64)
    else:
        k = rng.random(n)
    p = aot.sort_indices(k)
    assert len(p) == n and p.dtype == np.int32
    s = k[p]
    assert (s[1:] >= s[:-1]).all(), "sort_indices: keys not in order"
    assert int(p.astype(np.int64).sum()) == n * (n - 1) // 2 and len(np.unique(p)) == n, "not a permutation"


def w_group_sum(rng, np, aot, **_):
    n = int(rng.integers(50_000, 2_000_000))
    g = int(rng.integers(1, 5000))
    labels = rng.integers(0, g, n).astype(np.int32)
    vals = rng.random(n).astype(np.float32)
    got = aot.group_sum(vals, labels, g)
    want = np.bincount(labels, weights=vals.astype(np.float64), minlength=g)
    assert np.allclose(got, want, rtol=1e-4, atol=1e-2), "group_sum differs"


def w_join(rng, np, aot, **_):
    nl, nr = int(rng.integers(10_000, 1_000_000)), int(rng.integers(1_000, 200_000))
    lk = rng.integers(0, nr, nl).astype(np.int32)
    rk = np.arange(nr, dtype=np.int32)
    counts = aot.join_count(lk, rk)
    assert int(counts.sum(dtype=np.int64)) == nl, "join_count differs"
    lo, ro = aot.join_scatter(lk, rk, counts)
    assert len(lo) == nl and (rk[ro] == lk[lo]).all(), "join_scatter differs"


def w_int_keys(rng, np, gk, **_):
    n = int(rng.integers(60_000, 2_000_000))
    hi = int(rng.choice([7, 300, 50_000, 3_000_000]))
    a = rng.integers(0, hi, n).astype(np.int64)
    r = gk.encode_int(a)
    if r is not None:                      # None: range too wide for direct addressing, by design
        ids, uniq, first = r
        assert (uniq[ids] == a).all(), "encode_int: ids do not reproduce the keys"
        assert (a[first] == uniq).all()
    ranked = gk.dense_sorted(a * 7919 % 10**9)
    assert ranked is not None, "dense_sorted declined"
    ids2, uniq2, _ = ranked
    assert (uniq2[ids2] == a * 7919 % 10**9).all() and (np.diff(uniq2) > 0).all()


def w_fused_query(rng, np, mx, pa, **_):
    n = int(rng.integers(60_000, 1_500_000))
    t = pa.table({"g": pa.array(rng.integers(0, 200, n)), "a": pa.array(rng.random(n)),
                  "b": pa.array(rng.integers(0, 1000, n))})
    from omxcore import col
    out = (mx.LazyFrame(t).filter(col("b") < 500).groupby("g")
           .agg((col("a") * 2).sum().alias("s"), col("b").count().alias("c")).compute(device="cpu"))
    m = np.asarray(t.column("b")) < 500
    assert out.column("c").to_numpy().sum() == int(m.sum()), "fused aggregate count differs"
    assert abs(out.column("s").to_numpy().sum() - 2 * np.asarray(t.column("a"))[m].sum()) < 1e-6 * n, "sum differs"


def w_join_query(rng, np, mx, pa, **_):
    nl, nr = int(rng.integers(60_000, 1_000_000)), int(rng.integers(1_000, 100_000))
    left = pa.table({"k": pa.array(rng.integers(0, nr * 2, nl)), "v": pa.array(rng.random(nl))})
    right = pa.table({"k": pa.array(np.arange(nr)), "w": pa.array(rng.random(nr))})
    out = mx.LazyFrame(left).join(mx.LazyFrame(right), on="k").compute(device="cpu")
    assert out.num_rows == int((np.asarray(left.column("k")) < nr).sum()), "join row count differs"
    semi = mx.LazyFrame(left).join(mx.LazyFrame(right).select("k"), on="k").compute(device="cpu")
    assert semi.num_rows == out.num_rows


WORKLOADS = [w_sort, w_group_sum, w_join, w_int_keys, w_fused_query, w_join_query]


def child(args) -> int:
    faulthandler.enable()
    mx = _import_mxframe()
    import numpy as np
    import pyarrow as pa
    from omxcore import group_keys as gk
    from omxcore.aot_kernels import AOTKernels
    aot = AOTKernels()
    if args.warm:                      # the background warm-up thread that device="auto" starts, racing the queries
        try:
            from omxcore.custom_ops import _process_gpu_warmup
            _process_gpu_warmup()
        except Exception:
            pass
    stop = time.time() + args.seconds
    errors: list = []
    counts = [0] * args.threads

    def run(tid):
        rng = np.random.default_rng(args.seed * 1000 + tid)
        i = 0
        while time.time() < stop and not errors:
            fn = WORKLOADS[int(rng.integers(0, len(WORKLOADS)))]
            try:
                fn(rng, np, aot=aot, gk=gk, mx=mx, pa=pa)
            except AssertionError as e:
                errors.append(f"thread {tid} iter {i} {fn.__name__}: {e}")
                return
            except Exception as e:     # noqa: BLE001
                errors.append(f"thread {tid} iter {i} {fn.__name__}: {type(e).__name__}: {e}")
                return
            i += 1
            counts[tid] = i

    threads = [threading.Thread(target=run, args=(t,)) for t in range(args.threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        print("CHILD ERROR", errors[0], flush=True)
        return 2
    print(f"CHILD OK {sum(counts)} workloads", flush=True)
    return 0


def parent(args) -> int:
    if args.rounds > 1:                # many short-lived processes: the first calls of every library race each other
        bad = 0
        for r in range(args.rounds):
            a = argparse.Namespace(**{**vars(args), "rounds": 1, "seed": args.seed + r * args.procs})
            bad += parent_once(a, quiet=True)
        print(f"{args.rounds} rounds x {args.procs} processes: " + ("all clean" if not bad else f"{bad} failed"))
        return 1 if bad else 0
    return 1 if parent_once(args) else 0


def parent_once(args, quiet: bool = False) -> int:
    procs = []
    for p in range(args.procs):
        cmd = [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()), "--child",
               "--threads", str(args.threads), "--seconds", str(args.seconds), "--seed", str(args.seed + p)]
        if args.warm:
            cmd.append("--warm")
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                      env={**os.environ, "PYTHONUNBUFFERED": "1"}))
    bad = 0
    for p, pr in enumerate(procs):
        out, _ = pr.communicate()
        ok = pr.returncode == 0
        if not quiet or not ok:
            print(f"process {p}: {'ok' if ok else 'FAILED rc=' + str(pr.returncode)}")
        if not ok:
            bad += 1
            print("\n".join(out.strip().splitlines()[-25:]))
    if not quiet:
        print("all clean" if not bad else f"{bad} of {args.procs} processes failed")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--warm", action="store_true", help="also start the GPU warm-up thread in each process")
    ap.add_argument("--rounds", type=int, default=1, help="repeat the whole set of processes (use with a small --seconds)")
    ap.add_argument("--child", action="store_true")
    args = ap.parse_args()
    return child(args) if args.child else parent(args)


if __name__ == "__main__":
    sys.exit(main())
