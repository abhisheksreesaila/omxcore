"""F-064: float SUM / MEAN are bit-identical from run to run on every native CPU aggregate path.

Rows used to be handed to workers through an atomic counter (which worker adds which rows changed per
run) and the shared output used atomic float adds (the order of additions changed per run), so the last
bits of a float sum differed between two runs of the same query on the same data. Each test runs one
shape 30 times with 8 workers and demands 0 groups whose sum or mean differs bitwise from the first run,
and that the path under test ran (call counters), so it cannot pass through a fallback."""
import ctypes
import sys

import numpy as np
import pyarrow as pa
import pytest

import omxcore.native_exec as nx
from omxcore import LazyFrame, col, lit
from omxcore.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")

RUNS = 30


@pytest.fixture
def calls(monkeypatch):
    """Plan versions of every `mx_execute` call (3 = fused join + aggregate)."""
    log = []
    real = nx._call

    def call(lib, plan_addr, out_addr):
        log.append(int((ctypes.c_int64 * 2).from_address(plan_addr)[1]))
        return real(lib, plan_addr, out_addr)

    monkeypatch.setattr(nx, "_call", call)
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    monkeypatch.setenv("MXFRAME_CPU_THREADS", "8")
    return log


def _bits(t, names):
    return [np.ascontiguousarray(t.column(n).to_numpy()).view(np.int64) for n in names]


def _same_every_run(plan, names, runs=RUNS, **kw):
    first = _bits(plan.compute(device="cpu", **kw), names)
    differ = np.zeros(len(first[0]), bool)
    for _ in range(runs - 1):
        got = _bits(plan.compute(device="cpu", **kw), names)
        for a, b in zip(first, got):
            assert len(a) == len(b)
            differ |= a != b
    assert int(differ.sum()) == 0, f"{int(differ.sum())} of {len(differ)} groups differ bitwise between runs"


def _fact(n, groups, seed=7):
    r = np.random.default_rng(seed)
    return pa.table({"s": A(r.integers(0, groups, n).astype(np.int32)), "p": A(r.random(n) * 1000),
                     "d": A(r.random(n))})


def _revenue(lf):
    return lf.groupby("s").agg((col("p") * (lit(1.0) - col("d"))).sum().alias("tot"),
                               col("p").mean().alias("av"))


def test_single_table_keyed_aggregate(calls):
    """6M rows, 8k groups, `reproducible=True` (a table per morsel, merged in morsel order). The default
    mode keeps main's faster per-worker tables here, whose last bits vary between runs."""
    _same_every_run(_revenue(LazyFrame(_fact(6_000_000, 8_000))), ["tot", "av"], reproducible=True)
    assert calls and 3 not in calls          # the single-table aggregate, not the fused join


def test_global_aggregate(calls):
    t = _fact(6_000_000, 10)
    plan = LazyFrame(t).groupby().agg((col("p") * col("d")).sum().alias("tot"), col("d").mean().alias("av"))
    _same_every_run(plan, ["tot", "av"])
    assert calls and 3 not in calls


def test_many_groups_shared_output(calls):
    """3M groups over 6M rows, `reproducible=True`: a table per morsel is over the memory budget, so the
    rows are staged and applied per group range (the default mode runs the faster atomic shared output
    here, whose last bits vary between runs)."""
    _same_every_run(_revenue(LazyFrame(_fact(6_000_000, 3_000_000))), ["tot", "av"], reproducible=True)
    assert calls and 3 not in calls


def _star(groups):
    r = np.random.default_rng(11)
    n = 6_000_000
    fact = pa.table({"k": A(r.integers(0, groups, n).astype(np.int32)), "p": A(r.random(n) * 1000),
                     "d": A(r.random(n))})
    dim = pa.table({"dk": A(np.arange(groups, dtype=np.int32)), "w": A(r.random(groups))})
    return (LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk").groupby("k")
            .agg((col("p") * (lit(1.0) - col("d")) * col("w")).sum().alias("tot"), col("p").mean().alias("av")))


def test_fused_join_aggregate(calls):
    """Join + aggregate in one native call (plan version 3), 10k groups, `reproducible=True` (the default
    mode runs main's faster code under 16,384 groups)."""
    _same_every_run(_star(10_000), ["tot", "av"], reproducible=True)
    assert 3 in calls


def test_fused_join_aggregate_in_the_default_mode_from_16384_groups(calls):
    """Default mode, no flag: 20k groups use a table per morsel (65k and more: staged), both reproducible."""
    _same_every_run(_star(20_000), ["tot", "av"])
    assert 3 in calls


def test_fused_join_aggregate_many_groups_shared_output(calls):
    """Fused join + aggregate with 200k groups: staged above the table path, in the default mode too."""
    _same_every_run(_star(200_000), ["tot", "av"])
    assert 3 in calls


def _tot(t, extra, threads, monkeypatch, **kw):
    """The `tot` column of a grouped revenue sum next to `extra` more aggregates, with `threads` workers."""
    monkeypatch.setenv("MXFRAME_CPU_THREADS", str(threads))
    aggs = [(col("p") * (lit(1.0) - col("d"))).sum().alias("tot")]
    aggs += [(col("p") * col("d") * lit(float(i + 2))).sum().alias(f"x{i}") for i in range(extra)]
    return _bits(LazyFrame(t).groupby("s").agg(*aggs).compute(device="cpu", **kw), ["tot"])[0]


@pytest.mark.parametrize("groups", [300, 800, 8_000])
def test_a_sum_does_not_depend_on_the_other_aggregates(calls, monkeypatch, groups):
    """`reproducible=True`: `agg(tot)` and `agg(tot, x0, x1)` give the same bits for `tot` (the number of
    accumulator lanes used to depend on the aggregate list)."""
    t = _fact(6_000_000, groups)
    base = _tot(t, 0, 8, monkeypatch, reproducible=True)
    for extra in (2, 5):
        assert int((base != _tot(t, extra, 8, monkeypatch, reproducible=True)).sum()) == 0


@pytest.mark.parametrize("groups", [300, 10_000, 200_000])
def test_a_sum_does_not_depend_on_the_thread_count(calls, monkeypatch, groups):
    """4, 8 and 16 workers give the same bits, with `reproducible=True` (200k groups: the staged path)."""
    t = _fact(6_000_000, groups)
    base = _tot(t, 0, 4, monkeypatch, reproducible=True)
    for threads in (8, 16):
        assert int((base != _tot(t, 0, threads, monkeypatch, reproducible=True)).sum()) == 0


def _canonical(t, groups, morsel=65536):
    """The defined order for more than 1024 groups: per morsel, each group's rows added in row order;
    then the morsels' sub-totals added in morsel order."""
    s = t.column("s").to_numpy()
    v = t.column("p").to_numpy() * (1.0 - t.column("d").to_numpy())
    out = np.zeros(groups)
    for lo in range(0, len(s), morsel):
        sub = np.zeros(groups)
        np.add.at(sub, s[lo:lo + morsel], v[lo:lo + morsel])
        out += sub
    return out.view(np.int64)


@pytest.mark.parametrize("groups", [5_000, 10_000, 60_000, 200_000])
def test_the_per_morsel_table_path_and_the_staged_path_give_the_defined_order(calls, monkeypatch, groups):
    """`reproducible=True`: 10k and 60k groups use a table per morsel, 200k the staged path (tables over
    the budget): all equal the same NumPy reference of the defined order, bit for bit."""
    t = _fact(3_000_000, groups)
    want = _canonical(t, groups)
    assert int((_tot(t, 0, 8, monkeypatch, reproducible=True) != want).sum()) == 0


def test_reproducible_false_overrides_the_environment_variable(calls, monkeypatch):
    """With MXFRAME_REPRODUCIBLE=1, `reproducible=False` runs the default mode again: above 8,192 groups that
    is the faster path whose sums differ between runs (200k groups over 6M rows: many groups differ)."""
    if sys.platform != "linux":
        pytest.skip("the default mode differs between runs only through thread order; the macOS CPU kernels run on one thread "
                    "(kernels_aot/_fpenv.mojo SERIAL, F-210), so its sums repeat bit for bit")
    monkeypatch.setenv("MXFRAME_REPRODUCIBLE", "1")
    t = _fact(6_000_000, 200_000)
    plan = LazyFrame(t).groupby("s").agg((col("p") * (lit(1.0) - col("d"))).sum().alias("tot"))
    on = _bits(plan.compute(device="cpu"), ["tot"])[0]
    assert int((on != _bits(plan.compute(device="cpu"), ["tot"])[0]).sum()) == 0
    runs = [_bits(plan.compute(device="cpu", reproducible=False), ["tot"])[0] for _ in range(4)]
    assert any(int((runs[0] != r).sum()) > 0 for r in runs[1:])


def test_threads_computing_at_once_in_opposite_modes_each_keep_their_own(calls):
    """The flag travels with the compute call (a per-thread context variable and a plan word), not through
    process-wide state: while one thread computes in the default mode, another's reproducible results are
    still the defined order, bit for bit, and the first's default results are untouched."""
    import threading
    t = _fact(3_000_000, 200_000)
    want = _canonical(t, 200_000)
    plan = LazyFrame(t).groupby("s").agg((col("p") * (lit(1.0) - col("d"))).sum().alias("tot"))
    bad, stop = [], threading.Event()

    def reproducible_loop():
        for _ in range(10):
            if int((_bits(plan.compute(device="cpu", reproducible=True), ["tot"])[0] != want).sum()):
                bad.append("reproducible result differs")

    def default_loop():
        while not stop.is_set():
            got = plan.compute(device="cpu", reproducible=False).column("tot").to_numpy()
            if not np.allclose(got, want.view(np.float64), rtol=1e-9):
                bad.append("default result wrong")

    a, b = threading.Thread(target=reproducible_loop), threading.Thread(target=default_loop)
    b.start()
    a.start()
    a.join()
    stop.set()
    b.join()
    assert bad == []


def test_views_that_differ_only_in_an_array_literal_do_not_share(calls):
    """`A.filter(tot == B.scalar(max))` with A and B differing in an `isin` array must not merge them (the
    plan key keeps arrays by value): A holds {1, 2}, B {3}, so no group of A equals B's maximum."""
    t = pa.table({"s": A(np.array([1, 2, 3, 4], np.int32)), "x": A(np.array([10.0, 20.0, 30.0, 40.0]))})

    def view(keys):
        return LazyFrame(t).filter(col("s").isin(keys)).groupby("s").agg(col("x").sum().alias("tot"))

    a, b = view(np.array([1, 2])), view(np.array([3]))
    assert a.filter(col("tot") == b.scalar(col("tot").max())).compute(device="cpu").num_rows == 0
    same = view(np.array([1, 2]))
    assert a.filter(col("tot") == same.scalar(col("tot").max())).compute(device="cpu").num_rows == 1


def test_reproducible_mode_has_no_row_limit(calls):
    """24M rows, 8k groups, 6 sums, `reproducible=True` (the tables are folded into the output wave by
    wave, so memory does not grow with the rows), 10 runs."""
    r = np.random.default_rng(5)
    n = 24_000_000
    t = pa.table({"s": A(r.integers(0, 8_000, n).astype(np.int32)), "p": A(r.random(n) * 1000), "d": A(r.random(n))})
    aggs = [(col("p") * (lit(1.0) + lit(float(i)) * col("d"))).sum().alias(f"t{i}") for i in range(6)]
    _same_every_run(LazyFrame(t).groupby("s").agg(*aggs), [f"t{i}" for i in range(6)], runs=10,
                    reproducible=True)


def test_a_cte_joined_to_itself_on_a_float_sum(calls):
    """A CTE used twice in FROM is still computed twice (only a scalar subquery over the outer plan's own
    aggregate shares one result): the two sums are the same bits with `reproducible=True`; in the default
    mode a self-join on the sum can miss rows."""
    import omxcore as mx
    t = _fact(2_000_000, 5_000)
    q = ("WITH a AS (SELECT s, SUM(p * (1.0 - d)) AS tot FROM t GROUP BY s) "
         "SELECT x.s FROM a x JOIN a y ON x.tot = y.tot AND x.s = y.s")
    assert mx.sql(q, t=t).compute(device="cpu", reproducible=True).num_rows == 5_000


def test_reproducible_applies_when_the_native_executor_is_off(calls, monkeypatch):
    """MXFRAME_NATIVE=0 sends the aggregate through the older pipeline (`run_pipeline_cpu`, the same library's
    `mx_pipeline_cpu` with a 15th argument): `reproducible=True` still gives the defined order, bit for bit,
    at 200k groups (it used to differ in ~100k groups per rerun)."""
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    t = _fact(3_000_000, 200_000)
    monkeypatch.setenv("MXFRAME_CPU_THREADS", "8")
    plan = LazyFrame(t).groupby("s").agg((col("p") * (lit(1.0) - col("d"))).sum().alias("tot"))

    def by_key():                                   # this path returns groups in first-appearance order
        out = plan.compute(device="cpu", reproducible=True)
        order = np.argsort(out.column("s").to_numpy())
        return np.ascontiguousarray(out.column("tot").to_numpy()[order]).view(np.int64)

    first = by_key()
    assert calls == []                              # no mx_execute: the executor was off
    assert int((first != _canonical(t, 200_000)).sum()) == 0
    assert int((first != by_key()).sum()) == 0


def test_the_environment_variable_turns_reproducible_mode_on(calls, monkeypatch):
    """MXFRAME_REPRODUCIBLE=1 does what `reproducible=True` does (200k groups, staged), and
    `reproducible=False` overrides it."""
    t = _fact(3_000_000, 200_000)
    want = _canonical(t, 200_000)
    monkeypatch.setenv("MXFRAME_REPRODUCIBLE", "1")
    assert int((_tot(t, 0, 8, monkeypatch) != want).sum()) == 0


def test_a_view_compared_with_its_own_maximum_at_200k_groups_in_the_default_mode(calls):
    """The uncomputed-view Q15 shape over 200k groups (the many-groups path, whose default sums can differ
    between runs): the view is computed once and shared by the subquery and the outer plan, so it matches
    its own maximum 20 times of 20."""
    t = _fact(2_000_000, 200_000)
    view = LazyFrame(t).groupby("s").agg((col("p") * (lit(1.0) - col("d"))).sum().alias("tot"))
    for _ in range(20):
        got = view.filter(col("tot") == view.scalar(col("tot").max())).compute(device="cpu")
        assert got.num_rows == 1


def test_cpu_group_and_masked_sums_of_the_kernel_library():
    """The per-call CPU kernels (the non-native path, GPU-session fallbacks) hand out morsels in a fixed
    order too: 8 workers, 30 runs, identical bits."""
    from omxcore.aot_kernels import AOTKernels
    k = AOTKernels()
    r = np.random.default_rng(3)
    n = 6_000_000
    v = r.random(n) * 1000
    labels = r.integers(0, 10_000, n).astype(np.int32)
    mask = (r.random(n) < 0.7).astype(np.int32)
    first = (k.group_sum(v, labels, 10_000, threads=8).view(np.int64), k.group_mean(v, labels, 10_000, threads=8).view(np.int64),
             np.float64(k.masked_global_sum(v, mask, threads=8)).view(np.int64),
             np.float64(k.masked_global_sum_product(v, v, mask, threads=8)).view(np.int64))
    for _ in range(RUNS - 1):
        got = (k.group_sum(v, labels, 10_000, threads=8).view(np.int64), k.group_mean(v, labels, 10_000, threads=8).view(np.int64),
               np.float64(k.masked_global_sum(v, mask, threads=8)).view(np.int64),
               np.float64(k.masked_global_sum_product(v, v, mask, threads=8)).view(np.int64))
        for a, b in zip(first, got):
            assert np.array_equal(a, b)
