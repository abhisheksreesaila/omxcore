"""F-002: multi-core CPU kernels with float64 accumulation.

Every kernel is checked against a NumPy float64 reference, for several thread
budgets, on the edge cases that break morsel/partial logic: 0 rows, 1 row, one
group, many groups, sizes that do not divide evenly between workers, all rows
masked, labels out of range.
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import omxcore
from omxcore import aot_kernels
from omxcore.aot_kernels import AOT_AVAILABLE, AOTKernels, get_cpu_threads, set_cpu_threads

pytestmark = pytest.mark.skipif(not AOT_AVAILABLE, reason="libmxkernels_aot.so not built")

F32_MAX = 3.4028234663852886e38
F64_MAX = 1.7976931348623157e308
THREADS = [1, 2, 3, 24]


@pytest.fixture(scope="module")
def aot():
    return AOTKernels()


@pytest.fixture(autouse=True)
def _restore_threads():
    before = get_cpu_threads()
    yield
    set_cpu_threads(before)


def rng(seed=0):
    return np.random.default_rng(seed)


# ─────────────────────────────── grouped aggregates ───────────────────────────────

GROUP_CASES = [
    (0, 1), (0, 5), (1, 1), (1, 4), (7, 3), (100, 1),
    (65_537, 4), (300_001, 1), (300_001, 5), (300_001, 1000),
    (300_001, 200_000),          # many groups relative to rows: stays serial
    (4_000_003, 600_000),        # many groups, workers capped so partials stay small
    (2_100_003, 1_500_000),      # 1.5M groups: fewer partial workers / serial fallback
]


def _ref_group(vals, labels, n_groups):
    v = vals.astype(np.float64)
    ok = (labels >= 0) & (labels < n_groups)
    lab, v = labels[ok], v[ok]
    s = np.bincount(lab, weights=v, minlength=n_groups)
    c = np.bincount(lab, minlength=n_groups).astype(np.float64)
    mean = np.where(c > 0, s / np.maximum(c, 1), 0.0)
    mn = np.full(n_groups, F32_MAX if vals.dtype == np.float32 else F64_MAX)
    mx = -mn
    if len(v):
        order = np.argsort(lab, kind="stable")
        ls, vs = lab[order], v[order]
        first = np.r_[0, np.flatnonzero(np.diff(ls)) + 1]
        present = ls[first]
        mn[present] = np.minimum.reduceat(vs, first)
        mx[present] = np.maximum.reduceat(vs, first)
    return s, c, mean, mn, mx


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_rows,n_groups", GROUP_CASES)
def test_group_aggregates_match_float64_reference(aot, n_rows, n_groups, dtype):
    r = rng(n_rows + n_groups)
    vals = (r.random(n_rows) * 100 - 30).astype(dtype)
    labels = r.integers(0, n_groups, n_rows).astype(np.int32)
    s, c, mean, mn, mx = _ref_group(vals, labels, n_groups)
    for t in THREADS:
        np.testing.assert_allclose(aot.group_sum(vals, labels, n_groups, threads=t), s, rtol=1e-12, atol=1e-9)
        np.testing.assert_array_equal(aot.group_count(labels, n_groups, threads=t), c)
        np.testing.assert_allclose(aot.group_mean(vals, labels, n_groups, threads=t), mean, rtol=1e-12, atol=1e-9)
        np.testing.assert_array_equal(aot.group_min(vals, labels, n_groups, threads=t), mn)
        np.testing.assert_array_equal(aot.group_max(vals, labels, n_groups, threads=t), mx)


def test_group_out_of_range_labels_are_skipped(aot):
    n, g = 200_000, 7
    r = rng(1)
    vals = r.random(n).astype(np.float32)
    labels = r.integers(-3, g + 3, n).astype(np.int32)
    s, c, mean, mn, mx = _ref_group(vals, labels, g)
    for t in THREADS:
        np.testing.assert_allclose(aot.group_sum(vals, labels, g, threads=t), s, rtol=1e-12)
        np.testing.assert_array_equal(aot.group_count(labels, g, threads=t), c)
        np.testing.assert_array_equal(aot.group_min(vals, labels, g, threads=t), mn)


def test_group_no_groups_is_a_noop(aot):
    assert aot.group_sum(np.zeros(5, np.float32), np.zeros(5, np.int32), 0).shape == (0,)


def test_empty_groups_keep_identities(aot):
    labels = np.array([0, 0, 2], dtype=np.int32)          # group 1 and 3 are empty
    vals = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    assert list(aot.group_sum(vals, labels, 4)) == [3.0, 0.0, 5.0, 0.0]
    assert list(aot.group_mean(vals, labels, 4)) == [1.5, 0.0, 5.0, 0.0]
    assert list(aot.group_min(vals, labels, 4)) == [1.0, F32_MAX, 5.0, F32_MAX]
    assert list(aot.group_max(vals, labels, 4)) == [2.0, -F32_MAX, 5.0, -F32_MAX]
    assert list(aot.group_count(labels, 4)) == [2.0, 0.0, 1.0, 0.0]


def test_group_sum_i64_exact(aot):
    r = rng(2)
    n, g = 400_003, 11
    vals = r.integers(-(1 << 40), 1 << 40, n).astype(np.int64)
    labels = r.integers(0, g, n).astype(np.int32)
    ref = np.zeros(g, dtype=np.int64)
    np.add.at(ref, labels, vals)
    for t in THREADS:
        np.testing.assert_array_equal(aot.group_sum_i64(vals, labels, g, threads=t), ref)


def test_float32_sum_accumulates_in_float64(aot):
    """20M x 0.1f: a float32 accumulator is off by ~1e-3 relative; float64 is exact to ~1e-15."""
    n = 20_000_003
    vals = np.full(n, 0.1, dtype=np.float32)
    labels = np.zeros(n, dtype=np.int32)
    exact = float(np.float32(0.1)) * n
    for t in (1, 4):
        got = float(aot.group_sum(vals, labels, 1, threads=t)[0])
        assert abs(got - exact) / exact < 1e-12
        mask = np.ones(n, dtype=np.int32)
        got = aot.masked_global_sum(vals, mask, threads=t)
        assert abs(got - exact) / exact < 1e-12


def test_exact_results_do_not_depend_on_thread_count(aot):
    """Counts, min/max and integer sums are exact for any thread budget; float sums agree to ~1e-15."""
    r = rng(3)
    n, g = 1_000_003, 300
    vals = r.random(n).astype(np.float32)
    labels = r.integers(0, g, n).astype(np.int32)
    base = {t: aot.group_sum(vals, labels, g, threads=t) for t in (1, 5, 24)}
    for t in (5, 24):
        np.testing.assert_allclose(base[t], base[1], rtol=1e-13)
        np.testing.assert_array_equal(aot.group_count(labels, g, threads=t), aot.group_count(labels, g, threads=1))
        np.testing.assert_array_equal(aot.group_min(vals, labels, g, threads=t), aot.group_min(vals, labels, g, threads=1))


# ─────────────────────────────── masked global aggregates ─────────────────────────

MASKED_SIZES = [0, 1, 3, 8, 15, 16, 17, 131_077, 1_000_003]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", MASKED_SIZES)
def test_masked_global_aggregates(aot, n, dtype):
    r = rng(n)
    a = (r.random(n) * 10 - 3).astype(dtype)
    b = (r.random(n) * 2).astype(dtype)
    mask = (r.random(n) < 0.4).astype(np.int32)
    sel = mask != 0
    ref_sum = a.astype(np.float64)[sel].sum()
    ref_prod = (a.astype(np.float64) * b.astype(np.float64))[sel].sum()
    big = F32_MAX if dtype == np.float32 else F64_MAX
    ref_min = float(a[sel].min()) if sel.any() else big
    ref_max = float(a[sel].max()) if sel.any() else -big
    for t in THREADS:
        assert aot.masked_global_sum(a, mask, threads=t) == pytest.approx(ref_sum, rel=1e-12, abs=1e-9)
        assert aot.masked_global_sum_product(a, b, mask, threads=t) == pytest.approx(ref_prod, rel=1e-12, abs=1e-9)
        assert aot.masked_global_min(a, mask, threads=t) == ref_min
        assert aot.masked_global_max(a, mask, threads=t) == ref_max


@pytest.mark.parametrize("n", [1, 17, 300_001])
def test_masked_all_masked_out(aot, n):
    a = rng(n).random(n).astype(np.float32)
    zero = np.zeros(n, dtype=np.int32)
    for t in THREADS:
        assert aot.masked_global_sum(a, zero, threads=t) == 0.0
        assert aot.masked_global_sum_product(a, a, zero, threads=t) == 0.0
        assert aot.masked_global_min(a, zero, threads=t) == F32_MAX
        assert aot.masked_global_max(a, zero, threads=t) == -F32_MAX


def test_masked_ignores_nan_in_masked_out_rows(aot):
    n = 100_003
    a = rng(5).random(n).astype(np.float32)
    mask = np.ones(n, dtype=np.int32)
    a[::7] = np.nan
    mask[::7] = 0
    ref = a[mask != 0].astype(np.float64).sum()
    for t in THREADS:
        assert aot.masked_global_sum(a, mask, threads=t) == pytest.approx(ref, rel=1e-12)
        assert aot.masked_global_min(a, mask, threads=t) == float(a[mask != 0].min())


def test_legacy_f32_method_names_still_work(aot):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    m = np.array([1, 0, 1, 1], dtype=np.int32)
    assert aot.masked_global_sum_f32(a, m) == 8.0
    assert aot.masked_global_sum_product_f32(a, a, m) == 26.0
    assert list(aot.group_sum_f32(a, np.array([0, 1, 0, 1], np.int32), 2)) == [4.0, 6.0]


# ─────────────────────────────── gather / composite ───────────────────────────────

@pytest.mark.parametrize("m", [0, 1, 5, 65_537, 300_003])
def test_gather(aot, m):
    r = rng(m)
    src_n = 100_000
    idx = r.integers(0, src_n, m).astype(np.int32)
    for dt, fn in ((np.float32, aot.gather_f32), (np.int32, aot.gather_i32), (np.int64, aot.gather_i64)):
        src = (r.random(src_n) * 1000).astype(dt)
        for t in THREADS:
            np.testing.assert_array_equal(fn(src, idx, threads=t), src[idx])


@pytest.mark.parametrize("n", [0, 1, 9, 300_001])
def test_group_composite(aot, n):
    r = rng(n)
    ks = [r.integers(0, 50, n).astype(np.int32) for _ in range(4)]
    strides = np.array([1, 50, 2500, 125_000], dtype=np.int64)
    ref = sum(k.astype(np.int64) * s for k, s in zip(ks, strides))
    for t in THREADS:
        got = aot.group_composite(*ks, strides, threads=t)
        assert got.dtype == np.int64
        np.testing.assert_array_equal(got, ref)


# ─────────────────────────────── joins ───────────────────────────────────────────

def _ref_join(lk, rk, left=False):
    n_left = len(lk)
    order = np.argsort(rk, kind="stable").astype(np.int32)
    srk = rk[order]
    lo = np.searchsorted(srk, lk, side="left")
    hi = np.searchsorted(srk, lk, side="right")
    cnt = (hi - lo).astype(np.int64)
    counts = np.maximum(cnt, 1) if left else cnt
    total = int(counts.sum())
    l_out = np.repeat(np.arange(n_left, dtype=np.int32), counts)
    if total == 0:
        return counts.astype(np.int32), l_out, np.empty(0, np.int32)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    within = np.arange(total) - starts
    has = np.repeat(cnt > 0, counts)
    src = np.where(has, np.repeat(lo, counts) + within, 0)
    r_out = np.where(has, order[np.minimum(src, max(len(order) - 1, 0))] if len(order) else -1, -1).astype(np.int32)
    return counts.astype(np.int32), l_out, r_out


JOIN_CASES = [
    # (n_left, n_right, key_range)
    (1, 1, 1), (10, 10, 3), (1000, 500, 400), (5000, 5000, 50), (100_003, 2000, 5000),
    (1_500_007, 1_200_003, 800_000),        # parallel key-range build (several owners)
]


@pytest.mark.parametrize("n_left,n_right,key_range", JOIN_CASES)
def test_inner_join_matches_reference(aot, n_left, n_right, key_range):
    r = rng(n_left)
    lk = r.integers(0, key_range, n_left).astype(np.int32)
    rk = r.integers(0, key_range, n_right).astype(np.int32)
    mc_ref, lo_ref, ro_ref = _ref_join(lk, rk)
    for t in (1, 3, 24):
        mc = aot.join_count(lk, rk, threads=t)
        np.testing.assert_array_equal(mc, mc_ref)
        lo, ro = aot.join_scatter(lk, rk, mc, threads=t)
        np.testing.assert_array_equal(lo, lo_ref)
        np.testing.assert_array_equal(ro, ro_ref)


@pytest.mark.parametrize("n_left,n_right,key_range", JOIN_CASES)
def test_left_join_matches_reference(aot, n_left, n_right, key_range):
    r = rng(n_right)
    lk = r.integers(0, key_range + key_range // 3 + 1, n_left).astype(np.int32)   # some unmatched
    rk = r.integers(0, key_range, n_right).astype(np.int32)
    mc_ref, lo_ref, ro_ref = _ref_join(lk, rk, left=True)
    for t in (1, 3, 24):
        mc = aot.join_count_left(lk, rk, threads=t)
        np.testing.assert_array_equal(mc, mc_ref)
        lo, ro = aot.join_scatter_left(lk, rk, mc, threads=t)
        np.testing.assert_array_equal(lo, lo_ref)
        np.testing.assert_array_equal(ro, ro_ref)


def test_join_with_no_matches_and_empty_inputs(aot):
    lk = np.array([1, 2, 3], dtype=np.int32)
    rk = np.array([7, 8], dtype=np.int32)
    mc = aot.join_count(lk, rk)
    assert list(mc) == [0, 0, 0]
    lo, ro = aot.join_scatter(lk, rk, mc)
    assert len(lo) == len(ro) == 0
    assert list(aot.join_count(np.empty(0, np.int32), rk)) == []
    mcl = aot.join_count_left(lk, rk)
    assert list(mcl) == [1, 1, 1]
    lo, ro = aot.join_scatter_left(lk, rk, mcl)
    assert list(lo) == [0, 1, 2] and list(ro) == [-1, -1, -1]


def test_join_negative_probe_keys_do_not_match(aot):
    lk = np.array([-1, 0, 5], dtype=np.int32)
    rk = np.array([0, 5, 5], dtype=np.int32)
    mc = aot.join_count(lk, rk)
    assert list(mc) == [0, 1, 2]


# ─────────────────────────────── thread-count knob ───────────────────────────────

def test_set_and_get_cpu_threads():
    assert set_cpu_threads(3) == 3
    assert get_cpu_threads() == 3
    assert omxcore.get_cpu_threads() == 3
    omxcore.set_cpu_threads(2)
    assert get_cpu_threads() == 2
    with pytest.raises(ValueError):
        set_cpu_threads(0)
    set_cpu_threads(None)                         # back to env / all cores
    assert get_cpu_threads() >= 1


def _threads_in_fresh_process(env_value):
    # the shim dir (tests/conftest.py) is the parent of the un-resolved package path
    shim = str(Path(os.path.abspath(omxcore.__file__)).parent.parent)
    env = dict(os.environ, PYTHONPATH=shim)
    env.pop("MXFRAME_CPU_THREADS", None)
    if env_value is not None:
        env["MXFRAME_CPU_THREADS"] = env_value
    out = subprocess.run(
        [sys.executable, "-c", "from omxcore.aot_kernels import get_cpu_threads; print(get_cpu_threads())"],
        env=env, capture_output=True, text=True, timeout=120, check=True)
    return int(out.stdout.strip().splitlines()[-1])


def test_env_var_sets_default_thread_count():
    assert _threads_in_fresh_process("3") == 3
    all_cores = _threads_in_fresh_process(None)
    assert all_cores >= 1
    assert _threads_in_fresh_process("garbage") == all_cores      # invalid -> all cores
    assert _threads_in_fresh_process("0") == all_cores


def _thread_cpu_ns():
    """CPU nanoseconds each thread of this process has run (kernel scheduler accounting)."""
    out = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/schedstat") as f:
                out[tid] = int(f.read().split()[0])
        except OSError:                    # thread ended between listdir and open
            pass
    return out


def _busy_threads(fn, min_ns=500_000):
    """How many threads each ran at least ``min_ns`` of CPU during ``fn``.  CPU time is the
    work a thread did, so load from other processes stretches wall time but not this count."""
    before = _thread_cpu_ns()
    fn()
    after = _thread_cpu_ns()
    return sum(1 for t, ns in after.items() if ns - before.get(t, 0) >= min_ns)


def test_thread_budget_changes_actual_parallelism(aot):
    """threads=1 keeps the work on one thread; threads=4 spreads it over four workers
    (the calling thread also runs a share, so up to five threads are busy)."""
    if (os.cpu_count() or 1) < 4:
        pytest.skip("needs >= 4 cores")
    from conftest import cpu_kernels_threaded
    if not cpu_kernels_threaded():
        pytest.skip("the CPU kernels of this build run on one thread (kernels_aot/_fpenv.mojo SERIAL): no budget to change")
    n = 30_000_000
    a = np.ones(n, dtype=np.float32)
    mask = np.ones(n, dtype=np.int32)
    aot.masked_global_sum(a, mask, threads=4)      # warm the pool and page in the data

    if not os.path.isdir("/proc/self/task"):       # no per-thread CPU time (macOS): CPU time over wall time says how many ran
        import time

        def cpu_over_wall(threads):
            w0, c0 = time.perf_counter(), time.process_time()
            aot.masked_global_sum(a, mask, threads=threads)
            return (time.process_time() - c0) / (time.perf_counter() - w0)

        assert max(cpu_over_wall(1) for _ in range(4)) < 1.3
        assert min(cpu_over_wall(4) for _ in range(4)) > 2.0
        return

    def busy(threads):
        return [_busy_threads(lambda: aot.masked_global_sum(a, mask, threads=threads))
                for _ in range(4)]

    assert max(busy(1)) == 1
    four = busy(4)
    assert 4 <= min(four) and max(four) <= 5        # 4 workers, plus the calling thread (measured: 5)


def test_concurrent_calls_from_python_threads(aot):
    """ctypes releases the GIL; kernels keep no global state, so parallel callers are safe."""
    n, g = 700_001, 64
    cases = []
    for seed in range(8):
        r = rng(100 + seed)
        vals = r.random(n).astype(np.float32)
        labels = r.integers(0, g, n).astype(np.int32)
        mask = (r.random(n) < 0.5).astype(np.int32)
        cases.append((vals, labels, mask, _ref_group(vals, labels, g)[0],
                      vals.astype(np.float64)[mask != 0].sum()))
    errors = []

    def worker(i):
        vals, labels, mask, ref_g, ref_m = cases[i]
        try:
            for _ in range(5):
                np.testing.assert_allclose(aot.group_sum(vals, labels, g, threads=3), ref_g, rtol=1e-12)
                assert aot.masked_global_sum(vals, mask, threads=2) == pytest.approx(ref_m, rel=1e-12)
        except BaseException as exc:                   # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors, errors[0]


def test_stale_library_is_rejected(tmp_path):
    """A .so without the F-002 ABI symbol must fail loudly, not segfault."""
    candidates = ["/usr/lib/libm.so.6", "/lib64/libm.so.6", "/usr/lib/x86_64-linux-gnu/libm.so.6"]
    libm = next((Path(c) for c in candidates if Path(c).exists()), None)
    if libm is None:
        pytest.skip("libm not found")
    fake = tmp_path / "libfake.so"
    fake.symlink_to(libm)                      # a real shared object with no mx_abi_version
    with pytest.raises(RuntimeError, match="stale"):
        AOTKernels(lib_path=fake)
