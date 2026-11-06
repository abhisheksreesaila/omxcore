"""Router runtime: CPU and GPU workers pull morsels from one queue."""
import threading
import time

import numpy as np
import pytest

from omxcore.coproc import MorselQueue, Worker, run_router


def _sum_worker(name, device, data, morsel_rows, delay_per_row=0.0, **kw):
    def compute(rng):
        s, e = rng
        if delay_per_row:
            time.sleep(delay_per_row * (e - s))
        return float(data[s:e].sum())

    return Worker(name=name, device=device, morsel_rows=morsel_rows, compute=compute, **kw)


def test_queue_covers_every_row_exactly_once_under_contention():
    q = MorselQueue(1_000_003)
    got = []
    lock = threading.Lock()

    def pull(size):
        while (r := q.take(size)) is not None:
            with lock:
                got.append(r)

    ts = [threading.Thread(target=pull, args=(s,)) for s in (1000, 7777, 50_000, 3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    got.sort()
    assert got[0][0] == 0 and got[-1][1] == 1_000_003
    assert all(a[1] == b[0] for a, b in zip(got, got[1:]))


def test_router_result_matches_single_device():
    data = np.arange(1_000_000, dtype=np.float64)
    ws = [_sum_worker("cpu", "cpu", data, 64_000), _sum_worker("gpu", "gpu", data, 256_000)]
    res = run_router(len(data), ws, merge=sum)
    assert res.value == pytest.approx(data.sum())
    assert sum(s.rows for s in res.stats.values()) == len(data)


def test_faster_worker_takes_more_rows():
    data = np.ones(400_000)
    slow = _sum_worker("cpu", "cpu", data, 10_000, delay_per_row=2e-7)
    fast = _sum_worker("gpu", "gpu", data, 10_000, delay_per_row=2e-8)
    res = run_router(len(data), [slow, fast], merge=sum)
    assert res.value == len(data)
    assert res.stats["gpu"].rows > res.stats["cpu"].rows


def test_merge_sees_partials_in_row_order():
    data = np.arange(100, dtype=np.float64)
    seen = []

    def merge(parts):
        seen.extend(parts)
        return sum(parts)

    # Each partial is its morsel's first row, so row order means ascending values.
    ws = [
        Worker(name="a", device="cpu", morsel_rows=7, compute=lambda r: r[0]),
        Worker(name="b", device="gpu", morsel_rows=13, compute=lambda r: r[0]),
    ]
    run_router(len(data), ws, merge=merge)
    assert seen == sorted(seen) and len(seen) >= 8


def test_static_split_honours_gpu_fraction():
    data = np.ones(1_000_000)
    ws = [_sum_worker("cpu", "cpu", data, 50_000), _sum_worker("gpu", "gpu", data, 50_000)]
    res = run_router(len(data), ws, merge=sum, static_fractions={"gpu": 0.3})
    assert res.value == len(data)
    assert res.stats["gpu"].rows == 300_000
    assert res.stats["cpu"].rows == 700_000


def _timed_sleep(seconds):
    t = time.perf_counter()
    time.sleep(seconds)
    return time.perf_counter() - t


def test_prepare_overlaps_with_compute():
    """While morsel k computes, morsel k+1 is being prepared (the copy stage)."""
    events = []
    lock = threading.Lock()

    def prepare(rng):
        with lock:
            events.append(("prep_start", rng[0], time.perf_counter()))
        time.sleep(0.02)
        with lock:
            events.append(("prep_end", rng[0], time.perf_counter()))
        return rng

    def compute(h):
        with lock:
            events.append(("comp_start", h[0], time.perf_counter()))
        time.sleep(0.02)
        with lock:
            events.append(("comp_end", h[0], time.perf_counter()))
        return h[1] - h[0]

    unit = min(_timed_sleep(0.02) for _ in range(5))      # what this OS's sleep(20 ms) costs, overshoot included (macOS: ~27 ms)
    w = Worker(name="gpu", device="gpu", morsel_rows=10, prepare=prepare, compute=compute)
    t0 = time.perf_counter()
    res = run_router(60, [w], merge=sum)
    wall = time.perf_counter() - t0
    assert res.value == 60
    # 6 morsels * (prep + compute) = 12 sleeps serial; overlapped ~ 7 sleeps (140 ms where a sleep is 20 ms).
    assert wall < 0.8 * 12 * unit


def test_worker_error_is_raised_not_swallowed():
    def compute(rng):
        raise MemoryError("gpu oom")

    ws = [Worker(name="gpu", device="gpu", morsel_rows=10, compute=compute)]
    with pytest.raises(MemoryError):
        run_router(100, ws, merge=sum)


def test_failed_gpu_worker_can_hand_rows_back_to_cpu():
    """Fragment fallback inside a Router: a failing worker's morsels are redone by the others."""
    data = np.ones(10_000)

    def bad(rng):
        raise MemoryError("gpu oom")

    ws = [
        _sum_worker("cpu", "cpu", data, 500, delay_per_row=2e-6),
        Worker(name="gpu", device="gpu", morsel_rows=2000, compute=bad, fallible=True),
    ]
    res = run_router(len(data), ws, merge=sum)
    assert res.value == len(data)
    assert res.stats["gpu"].failed and res.stats["gpu"].rows == 0
    assert res.fallbacks == 1


def test_empty_input():
    res = run_router(0, [_sum_worker("cpu", "cpu", np.ones(0), 10)], merge=lambda parts: sum(parts))
    assert res.value == 0
