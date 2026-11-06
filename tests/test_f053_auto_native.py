"""F-053: device="auto" runs the native executor whenever it decides on the CPU.

The co-processing policy used to be set on every CPU decision, and the native executor declines
under it, so the default `.compute()` ran the older fused pipeline. Now the policy is set only when
the cost model predicts a CPU+GPU split beats the native CPU executor alone.

TODO (quiet-machine run, F-053 scoreboard item): `auto` within 10% of min(cpu, gpu) on >= 20/22
TPC-H queries at SF1 and SF10, median of fresh-process runs.
"""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import omxcore.lazy_frame as lfm
import omxcore.native_exec as nx
import omxcore.pipeline_kernels as pk
from omxcore import LazyFrame, auto_mode as am, calibration as cal, col, lit
from omxcore import coproc_fused as cf
from omxcore.np_arrow import array as A

needs_gpu = pytest.mark.skipif(not pk.gpu_available(), reason="the cpu+gpu split needs the pipeline GPU library and a CUDA device (NVIDIA)")

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")

N = 300_000      # above SMALL_QUERY_ROWS, so auto prices the plan instead of short-cutting


class _Calib:
    """A calibration that says whatever the test needs about the GPU."""

    def __init__(self, has_gpu, cpu_ms=100.0, gpu_ms=1.0):
        self.has_gpu, self.cpu_ms, self.gpu_ms = has_gpu, cpu_ms, gpu_ms

    def estimate(self, op, device, rows, card=None, **k):
        if op != "pipeline":              # whole-query pricing: the GPU never wins, so auto decides cpu
            return 10.0 if device == "cpu" else 1e6
        return self.cpu_ms if device == "cpu" else self.gpu_ms

    def estimate_gpu_join_total(self, *a, **k):
        return 1e6

    def transfer_ms(self, *a, **k):
        return 0.0

    def gpu_init_cost_ms(self):
        return 200.0


class _Warmup:
    available = True

    def __init__(self, ready):
        self.ready = ready

    def remaining_init_ms(self):
        return 0.0 if self.ready else 200.0

    def start(self):
        return self


def _fake_gpu(monkeypatch, ready, cpu_ms=100.0, gpu_ms=1.0):
    monkeypatch.setattr(cal, "get_calibration", lambda: _Calib(True, cpu_ms, gpu_ms))
    monkeypatch.setattr(am, "get_gpu_warmup", lambda calib=None, start=True: _Warmup(ready))
    monkeypatch.setattr(am, "gpu_warmup_started", lambda: ready)      # a ready GPU has started (F-058 cold shortcut)


@pytest.fixture(params=["small input", "no gpu", "cold gpu", "warm gpu, split not worth it"])
def gpu_state(request, monkeypatch):
    if request.param == "no gpu":
        monkeypatch.setattr(cal, "get_calibration", lambda: _Calib(False))
    elif request.param == "cold gpu":
        _fake_gpu(monkeypatch, ready=False)
    elif request.param == "warm gpu, split not worth it":
        monkeypatch.setattr(cf, "MIN_UPLOAD_BYTES", 1)        # the model, not the size gate, says no
        _fake_gpu(monkeypatch, ready=True, cpu_ms=100.0, gpu_ms=1000.0)
    return request.param


@pytest.fixture
def native_calls(monkeypatch):
    """Every native entry point that ran, by name."""
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    log = []
    for name in ("_call", "_call_filter", "_call_distinct", "_call_count_distinct", "_call_gather"):
        real = getattr(nx, name)

        def counting(*a, _real=real, _name=name):
            log.append(_name)
            return _real(*a)

        monkeypatch.setattr(nx, name, counting)
    return log


@pytest.fixture
def policies(monkeypatch):
    """The co-processing policy in force each time a device run starts."""
    seen = []
    real = lfm._run_on_device

    def spy(plan, device, **kw):
        seen.append(cf.current_policy())
        return real(plan, device, **kw)

    monkeypatch.setattr(lfm, "_run_on_device", spy)
    return seen


def _fact(n):
    r = np.random.default_rng(7)
    return pa.table({
        "k": A(r.integers(0, 50, n).astype(np.int64)),
        "dim": A(r.integers(0, 200, n).astype(np.int64)),
        "d": A(r.integers(0, 1000, n).astype(np.int32)),
        "v": A(r.random(n) * 100),
    })


def _dim():
    r = np.random.default_rng(8)
    return pa.table({"dim": A(np.arange(200, dtype=np.int64)), "grp": A(r.integers(0, 5, 200).astype(np.int64))})


def _distinct_tbl(n):
    r = np.random.default_rng(9)
    return pa.table({"a": A(np.sort(r.integers(0, n // 7, n)).astype(np.int64)),
                     "b": A(r.integers(0, 5, n).astype(np.int64))})


# name -> (frame builder, DuckDB SQL over tables f, dm, t, ordered)
PLANS = {
    "single-table aggregate": (
        lambda f, dm, t: LazyFrame(f).filter(col("d") < lit(900)).groupby("k").agg(col("v").sum().alias("s"),
                                                                                  col("v").count().alias("n")),
        "select k, sum(v) s, count(v) n from f where d < 900 group by k", False),
    "filter": (
        lambda f, dm, t: LazyFrame(f).filter(col("d") < lit(5)).select("k", "d"),
        "select k, d from f where d < 5", False),
    "top-k": (
        lambda f, dm, t: LazyFrame(f).groupby("k").agg(col("v").sum().alias("s")).sort("s", descending=[True]).limit(5),
        "select k, sum(v) s from f group by k order by s desc limit 5", True),
    "star join + aggregate": (
        lambda f, dm, t: LazyFrame(f).join(LazyFrame(dm), on="dim").groupby("grp").agg(col("v").sum().alias("s")),
        "select grp, sum(v) s from f join dm using (dim) group by grp", False),
    "count distinct": (
        lambda f, dm, t: LazyFrame(t).select("a", "b").distinct().groupby("a").agg(col("b").count().alias("n")),
        "select a, count(distinct b) n from t group by a", False),
}


def _rows(tbl, ordered):
    rows = list(zip(*[tbl.column(c).to_pylist() for c in tbl.column_names]))
    rows = [tuple(float(f"{x:.9g}") if isinstance(x, float) else x for x in r) for r in rows]
    return rows if ordered else sorted(rows)


def _duckdb(sql, **tables):
    con = duckdb.connect()
    for name, t in tables.items():
        con.register(name, t)
    out = con.sql(sql).arrow()
    return out.read_all() if hasattr(out, "read_all") else out


@pytest.mark.parametrize("name", list(PLANS))
def test_auto_on_a_cpu_decision_takes_the_native_path_of_device_cpu(gpu_state, native_calls, name):
    build, sql, ordered = PLANS[name]
    n = 20_000 if gpu_state == "small input" else N
    f, dm, t = _fact(n), _dim(), _distinct_tbl(n)

    cpu = build(f, dm, t).compute(device="cpu")
    cpu_calls = list(native_calls)
    native_calls.clear()
    lf = build(_fact(n), _dim(), _distinct_tbl(n))      # fresh tables: no result cache from the cpu run
    auto = lf.compute(device="auto")

    assert lf.last_auto_decision.device == "cpu"
    assert cpu_calls, "device='cpu' did not reach the native executor: the test would prove nothing"
    assert native_calls == cpu_calls
    want = _duckdb(sql, f=f, dm=dm, t=t)
    assert _rows(auto, ordered) == _rows(want, ordered) == _rows(cpu, ordered)


def test_no_coproc_policy_unless_a_split_is_predicted(monkeypatch, policies):
    _fake_gpu(monkeypatch, ready=False)
    LazyFrame(_fact(N)).groupby("k").agg(col("v").sum().alias("s")).compute(device="auto")
    assert policies == [None]


@needs_gpu
def test_a_predicted_split_sets_the_policy_and_keeps_the_answer(monkeypatch, policies):
    """Warm GPU, GPU priced far below the CPU, columns resident (F-057): the split path runs and is correct."""
    monkeypatch.setattr(cf, "MIN_UPLOAD_BYTES", 1)          # a test-sized input still qualifies
    monkeypatch.setattr(cf, "resident_for", lambda *a: True)
    _fake_gpu(monkeypatch, ready=True, cpu_ms=100.0, gpu_ms=30.0)
    monkeypatch.setattr(cf, "_gpu_usable", lambda: False)   # no real GPU work: try_split declines
    f = _fact(N)
    lf = LazyFrame(f).groupby("k").agg(col("v").sum().alias("s"))
    got = lf.compute(device="auto")
    assert len(policies) == 1 and policies[0] is not None, lf.last_auto_decision.explain()
    assert "cpu path: cpu+gpu split" in lf.last_auto_decision.explain()
    want = _duckdb("select k, sum(v) s from f group by k", f=f)
    assert _rows(got, False) == _rows(want, False)


@needs_gpu
def test_a_split_worthy_query_on_host_data_runs_native_and_queues_the_upload(monkeypatch, policies):
    """The first query of a split-worthy aggregate has no resident columns: it runs exactly as device="cpu"
    (native, no policy), and the upload that later queries split over starts once it has returned (F-057)."""
    n = 6_000_000                                             # 12 B/row of source: over the 64 MiB gate
    r = np.random.default_rng(3)
    f = pa.table({"k": A(r.integers(0, 50, n).astype(np.int64)), "v": A(r.random(n))})
    _fake_gpu(monkeypatch, ready=True, cpu_ms=100.0, gpu_ms=30.0)
    monkeypatch.setattr(cf, "_gpu_usable", lambda: True)
    monkeypatch.setattr(cf, "IDLE_BEFORE_BACKGROUND_S", 0.0)
    queued = []
    from omxcore import pipeline_exec
    monkeypatch.setattr(pipeline_exec, "queue_resident_upload", lambda plan: queued.append(1))
    lf = LazyFrame(f).groupby("k").agg(col("v").sum().alias("s"))
    got = lf.compute(device="auto")
    assert "cpu path: native executor (columns not resident on the GPU yet, upload queued" in lf.last_auto_decision.explain()
    assert policies == [None] and queued == [1]
    want = _duckdb("select k, sum(v) s from f group by k", f=f)
    assert _rows(got, False) == _rows(want, False)


def test_split_is_priced_on_source_rows_not_filtered_rows(monkeypatch):
    """A selective filter leaves few rows, but the pipeline (and the upload) read every source row."""
    n = 6_000_000
    f = pa.table({"k": A(np.zeros(n, dtype=np.int64)), "d": A(np.arange(n, dtype=np.int32) % 1000),
                  "v": A(np.ones(n))})
    _fake_gpu(monkeypatch, ready=True, cpu_ms=100.0, gpu_ms=30.0)
    monkeypatch.setattr(cf, "resident_for", lambda *a: True)
    lf = LazyFrame(f).filter(col("d") < lit(3)).groupby("k").agg(col("v").sum().alias("s"))
    assert "cpu path: cpu+gpu split" in lf.explain(device="auto", costs=True)


def test_shapes_the_split_cannot_take_are_never_predicted_a_split(monkeypatch):
    monkeypatch.setattr(cf, "MIN_UPLOAD_BYTES", 1)
    _fake_gpu(monkeypatch, ready=True, cpu_ms=100.0, gpu_ms=30.0)
    f, dm = _fact(N), _dim()
    joined = LazyFrame(f).join(LazyFrame(dm), on="dim").groupby("grp").agg(col("v").sum().alias("s"))
    assert "cpu path: native executor (no cpu+gpu split: plan is not Aggregate(Filter*(Scan))" in joined.explain(
        device="auto", costs=True)
    derived = LazyFrame(f).groupby((col("k") + lit(1)).alias("k1")).agg(col("v").sum().alias("s"))
    assert "cpu path: native executor (no cpu+gpu split: derived or renamed group keys" in derived.explain(
        device="auto", costs=True)
    renamed = LazyFrame(f).groupby(col("k").alias("kk")).agg(col("v").sum().alias("s"))
    assert "no cpu+gpu split: derived or renamed" in renamed.explain(device="auto", costs=True)


def test_gpu_fraction_forces_a_split_and_keeps_the_answer(monkeypatch, policies):
    _fake_gpu(monkeypatch, ready=False)                     # the model alone would say native
    monkeypatch.setattr(cf, "_gpu_usable", lambda: False)
    f = _fact(N)
    got = LazyFrame(f).groupby("k").agg(col("v").sum().alias("s")).compute(device="auto", gpu_fraction=0.5)
    assert len(policies) == 1 and policies[0].gpu_fraction == 0.5
    want = _duckdb("select k, sum(v) s from f group by k", f=f)
    assert _rows(got, False) == _rows(want, False)


def test_a_split_that_does_not_beat_native_cpu_is_not_chosen(monkeypatch):
    """GPU 10x slower than the CPU kernel: a split saves under the 10% margin, so the native executor runs."""
    monkeypatch.setattr(cf, "MIN_UPLOAD_BYTES", 1)
    _fake_gpu(monkeypatch, ready=True, cpu_ms=100.0, gpu_ms=1000.0)
    lf = LazyFrame(_fact(N)).groupby("k").agg(col("v").sum().alias("s"))
    assert "cpu path: native executor" in lf.explain(device="auto", costs=True)


def test_a_native_cpu_far_faster_than_the_gpu_is_not_split(monkeypatch):
    """The CPU share of a split runs on the native executor, so its price scales with NATIVE_CPU_FACTOR: at
    5 ms native against a 30 ms GPU (4.5 ms per call) no split is worth it."""
    monkeypatch.setattr(cf, "MIN_UPLOAD_BYTES", 1)
    monkeypatch.setattr(cf, "resident_for", lambda *a: True)
    monkeypatch.setattr(am, "NATIVE_CPU_FACTOR", 0.05)
    _fake_gpu(monkeypatch, ready=True, cpu_ms=100.0, gpu_ms=30.0)
    lf = LazyFrame(_fact(N)).groupby("k").agg(col("v").sum().alias("s"))
    text = lf.explain(device="auto", costs=True)
    assert "cpu path: native executor (no cpu+gpu split: est " in text and "cpu alone 5.0ms" in text
    monkeypatch.setattr(am, "NATIVE_CPU_FACTOR", 1.0)
    text = lf.explain(device="auto", costs=True)
    assert "cpu path: cpu+gpu split (est " in text and "cpu alone 100.0ms" in text


def test_explain_says_which_cpu_path_and_why(monkeypatch):
    monkeypatch.setattr(cal, "get_calibration", lambda: _Calib(False))
    lf = LazyFrame(_fact(N)).groupby("k").agg(col("v").sum().alias("s"))
    text = lf.explain(device="auto", costs=True)
    assert "auto -> cpu  reason: no gpu" in text
    assert "cpu path: native executor (no cpu+gpu split: no gpu)" in text


@pytest.mark.parametrize("source", ["memory", "parquet"])
def test_auto_keeps_count_distinct_columns_encoded_like_device_cpu(native_calls, tmp_path, source):
    """device='cpu' hands a dictionary column of COUNT(DISTINCT b) per a to the native count as codes;
    auto resolves the plan (decoding dictionaries) before it decides, so after a CPU decision it must
    resolve it the same way."""
    r = np.random.default_rng(5)
    n = 200_000
    words = ["MAIL", "SHIP", "AIR", "RAIL", "TRUCK"]
    t = pa.table({
        "a": A(r.integers(0, 3000, n).astype(np.int64)),
        "s": pa.DictionaryArray.from_arrays(A(r.integers(0, 5, n).astype(np.int32)), pa.array(words)),
    })

    if source == "parquet":
        import pyarrow.parquet as pq
        from omxcore import read_parquet
        pq.write_table(t, str(tmp_path / "t.parquet"), row_group_size=50_000)

    def frame():
        return (read_parquet(str(tmp_path / "t.parquet")) if source == "parquet" else LazyFrame(t)).select("a", "s").distinct().groupby("a").agg(col("s").count().alias("n"))

    cpu = frame().compute(device="cpu")
    cpu_calls = list(native_calls)
    native_calls.clear()
    lf = frame()
    auto = lf.compute(device="auto")
    assert lf.last_auto_decision.device == "cpu"
    assert "_call_count_distinct" in cpu_calls, cpu_calls
    assert native_calls == cpu_calls
    assert _rows(auto, False) == _rows(_duckdb("select a, count(distinct s) n from t group by a", t=t), False)


def test_coproc_env_off_never_sets_the_policy(monkeypatch, policies):
    monkeypatch.setenv("MXFRAME_COPROC", "0")
    monkeypatch.setattr(cf, "MIN_UPLOAD_BYTES", 1)
    _fake_gpu(monkeypatch, ready=True, cpu_ms=100.0, gpu_ms=5.0)
    lf = LazyFrame(_fact(N)).groupby("k").agg(col("v").sum().alias("s"))
    lf.compute(device="auto")
    assert policies == [None]
