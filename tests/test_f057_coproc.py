"""F-057: morsel co-processing that wins.

Residency of repeat queries over a file (scan_parquet / read_parquet) or an in-memory table; the native
CPU executor as the CPU share of a split; one predicate for what the split path takes; a split chosen
only when the model says it beats the faster device by 10%, with the numbers in `explain`.

TODO (quiet-machine run, F-057 scoreboard item): auto within 10% of min(cpu, gpu) on every query and
faster than both on resident high-volume aggregation; needs the machine to itself.
"""
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import omxcore as mx
from omxcore import LazyFrame, auto_mode as am, calibration as cal, col, lit
from omxcore import coproc_fused as cf
from omxcore import native_exec as nx
from omxcore import pipeline_exec as pe
from omxcore import pipeline_kernels as pk
from omxcore.lazy_frame import Aggregate

needs_gpu = pytest.mark.skipif(not pk.gpu_available() or not pk.cpu_available(),
                               reason="needs both pipeline libraries and a GPU")
needs_native = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")


def data(n=600_000, groups=37, seed=0, nan=False):
    rng = np.random.default_rng(seed)
    w = rng.random(n)
    if nan == "group":                 # every row of group 0 is NaN, so its min and max are NaN
        w[:] = np.where(np.arange(n) % groups == 0, np.nan, w)
    elif nan:
        w[rng.random(n) < 0.01] = np.nan
    return pa.table({
        "k": pa.array(rng.integers(0, groups, n).astype(np.int32)),
        "j": pa.array(rng.integers(0, 3, n).astype(np.int64)),
        "v": pa.array(rng.random(n).astype(np.float32)),
        "w": pa.array(w),
        "d": pa.array(rng.integers(0, 100, n).astype(np.int32)),
        "i": pa.array(rng.integers(0, 1000, n).astype(np.int64)),
    })


def agg_query(lf, grouped=True):
    f = lf.filter(col("d") > lit(20))
    aggs = [col("v").sum().alias("s"), (col("v") * col("w")).sum().alias("p"), col("w").max().alias("mx"),
            col("w").min().alias("mn"), col("v").mean().alias("a"), col("i").sum().alias("isum"),
            col("v").count().alias("n")]
    return f.groupby("k").agg(*aggs).sort("k") if grouped else f.groupby().agg(*aggs)


def same(a, b):
    assert a.column_names == b.column_names and a.num_rows == b.num_rows
    for c in a.column_names:
        x = np.asarray(a.column(c).to_pylist(), dtype=float)
        y = np.asarray(b.column(c).to_pylist(), dtype=float)
        np.testing.assert_allclose(x, y, rtol=1e-9, equal_nan=True, err_msg=c)


def duck(sql, **tables):
    con = duckdb.connect()
    for n, t in tables.items():
        con.register(n, t)
    out = con.sql(sql).arrow()
    return out.read_all() if hasattr(out, "read_all") else out


GROUPED_SQL = ("select k, sum(v) s, sum(v*w) p, max(w) mx, min(w) mn, avg(v) a, sum(i) isum, count(v) n "
               "from t where d > 20 group by k order by k")
GLOBAL_SQL = ("select sum(v) s, sum(v*w) p, max(w) mx, min(w) mn, avg(v) a, sum(i) isum, count(v) n "
              "from t where d > 20")


# ── 1. residency of repeat queries ──────────────────────────────────────────────────────────────────────────

@pytest.fixture
def parquet_file(tmp_path):
    t = data()
    path = str(tmp_path / "t.parquet")
    pq.write_table(t, path, row_group_size=100_000)
    return path, t


def _repeat(make_frame, runs=3, fraction=0.5):
    """Compute the query `runs` times on fresh frames (a new read each time for a file); (paths, results)."""
    paths, results = [], []
    with cf.policy(min_upload_bytes=0, gpu_fraction=fraction):
        for _ in range(runs):
            lf = agg_query(make_frame())
            results.append(lf.compute(device="cpu"))
            paths.append(lf.last_compile_provenance["path"])
            cf.wait_for_uploads(30)
    return paths, results


@needs_gpu
@needs_native
@pytest.mark.parametrize("source", ["scan_parquet", "read_parquet", "in-memory table"])
def test_repeat_queries_over_the_same_source_find_it_resident(parquet_file, source):
    """Each compute of a scan_parquet frame (and each read_parquet call) reads the file into new buffers: the
    residency key is the file's identity, so the second query still finds the first one's upload."""
    path, t = parquet_file
    make = {"scan_parquet": lambda: mx.scan_parquet(path), "read_parquet": lambda: mx.read_parquet(path),
            "in-memory table": lambda: LazyFrame(t)}[source]
    cf.clear_residency()
    paths, results = _repeat(make)
    assert paths == ["fused_pipeline_cpu", "fused_pipeline_split", "fused_pipeline_split"]
    assert cf.resident_count() == 1
    want = duck(GROUPED_SQL, t=t)
    for r in results:
        same(r, want)


@needs_gpu
@needs_native
def test_global_aggregate_over_a_file_is_resident_too(parquet_file):
    path, t = parquet_file
    cf.clear_residency()
    paths = []
    with cf.policy(min_upload_bytes=0, gpu_fraction=0.5):
        for _ in range(2):
            lf = agg_query(mx.scan_parquet(path), grouped=False)
            got = lf.compute(device="cpu")
            paths.append(lf.last_compile_provenance["path"])
            cf.wait_for_uploads(30)
    assert paths == ["fused_pipeline_cpu", "fused_pipeline_split"]
    same(got, duck(GLOBAL_SQL, t=t))


@needs_gpu
@needs_native
def test_a_changed_file_never_serves_the_old_device_copy(tmp_path):
    path = str(tmp_path / "t.parquet")
    old, new = data(seed=1), data(seed=2, n=650_000)
    pq.write_table(old, path, row_group_size=100_000)
    cf.clear_residency()
    _repeat(lambda: mx.scan_parquet(path), runs=2)
    assert cf.resident_count() == 1
    pq.write_table(new, path, row_group_size=100_000)          # same path, new content and size
    paths, results = _repeat(lambda: mx.scan_parquet(path), runs=2)
    assert paths == ["fused_pipeline_cpu", "fused_pipeline_split"]
    for r in results:
        same(r, duck(GROUPED_SQL, t=new))
    assert cf.resident_count() == 1                              # the old file's copy was freed, not kept beside


@needs_gpu
@needs_native
def test_different_filters_pushed_into_the_read_are_different_sources(tmp_path):
    """A pushed filter drops rows while reading: the two reads hold different rows and must not share a copy."""
    path = str(tmp_path / "t.parquet")
    t = data()
    pq.write_table(t, path, row_group_size=100_000)
    cf.clear_residency()

    def frame(limit):
        return (mx.scan_parquet(path).filter(col("i") < lit(limit)).groupby("k")
                .agg(col("v").sum().alias("s"), col("w").max().alias("mx")).sort("k"))

    from omxcore.io_parquet import resolve_plan, table_source
    tokens = {limit: table_source(_scan_table(resolve_plan(frame(limit).plan))) for limit in (300, 700)}
    assert tokens[300] != tokens[700] and all(tokens.values())
    with cf.policy(min_upload_bytes=0, gpu_fraction=0.5):
        for limit in (300, 700, 300, 700):
            got = frame(limit).compute(device="cpu")
            cf.wait_for_uploads(30)
            want = duck(f"select k, sum(v) s, max(w) mx from t where i < {limit} group by k order by k", t=t)
            same(got, want)


def _scan_table(plan):
    from omxcore.lazy_frame import Scan
    while not isinstance(plan, Scan):
        plan = plan.input
    return plan.table


@needs_gpu
@needs_native
def test_two_queries_alternating_on_one_file_are_both_resident(parquet_file):
    """Q1 then Q6 on one file: each read of the file is its own source and they must not evict each other."""
    path, t = parquet_file
    cf.clear_residency()

    def one(c):
        return mx.scan_parquet(path).groupby("k").agg(col(c).sum().alias("s")).sort("k")

    paths = []
    with cf.policy(min_upload_bytes=0, gpu_fraction=0.5):
        for _ in range(3):
            for c in ("v", "w"):
                lf = one(c)
                got = lf.compute(device="cpu")
                paths.append((c, lf.last_compile_provenance["path"]))
                cf.wait_for_uploads(30)
                same(got, duck(f"select k, sum({c}) s from t group by k order by k", t=t))
    assert paths[:2] == [("v", "fused_pipeline_cpu"), ("w", "fused_pipeline_cpu")]
    assert all(p == "fused_pipeline_split" for _, p in paths[2:]), paths
    assert cf.resident_count() == 2


def test_residency_is_asked_for_exactly_the_programs_columns():
    """A resident sum(v) says nothing about sum(v), sum(w): a split would be predicted and never run."""
    cf.clear_residency()
    src = ("parquet", "/x", 1, 2, 3, 4, ("v", "w"), "()", ())
    with cf._lock:
        cf._cache["fake"] = cf._Entry(dc=None, nbytes=0, owners=[], sig=(src, frozenset({"v"}), ("k",)))
    try:
        assert cf.resident_for(src, {"v"}, ("k",))
        assert not cf.resident_for(src, {"v", "w"}, ("k",))
        assert not cf.resident_for(src, {"v"}, ())
    finally:
        with cf._lock:
            cf._cache.pop("fake")


def test_the_prepared_registry_is_bounded_and_cleared(monkeypatch):
    cf.clear_residency()
    monkeypatch.setattr(cf, "MAX_PREPARED", 3)
    src = lambda i: ("parquet", f"/f{i}", 1, 2, 3, 4, (), "()", ())
    assert all(cf.begin_prepare(src(i), {"v"}, ()) for i in range(5))
    assert len(cf._prepared) == 3
    assert not cf.begin_prepare(src(4), {"v"}, ())          # still remembered: not redone
    cf.clear_residency()
    assert len(cf._prepared) == 0 and cf.begin_prepare(src(4), {"v"}, ())


def test_a_file_read_carries_a_source_token_and_a_derived_table_does_not(parquet_file):
    from omxcore.io_parquet import read_parquet_table, table_source
    path, t = parquet_file
    a, b = read_parquet_table(path), read_parquet_table(path)
    assert a is not b and table_source(a) == table_source(b) and table_source(a)[0] == "parquet"
    assert table_source(a.slice(0, 10)) is None and table_source(t) is None
    assert table_source(read_parquet_table(path, columns=["k"])) != table_source(a)


# ── 2. the native executor takes the CPU share of a split ─────────────────────────────────────────────────

def _inputs(table, grouped=True):
    plan = agg_query(LazyFrame(table), grouped=grouped).plan
    while not isinstance(plan, Aggregate):
        plan = plan.input
    got = pe.split_check(plan)
    assert not isinstance(got, str), got
    return plan, got


def _gids(compiler_cls, got):
    gids = None
    if got.gkeys:
        gids, n_groups, _ = compiler_cls._build_group_ids_cached(got.table, list(got.gkeys))
        return np.ascontiguousarray(gids, dtype=np.int32), n_groups
    return None, 1


@needs_native
@pytest.mark.skipif(not pk.cpu_available(), reason="needs libmxpipeline_cpu.so")
@pytest.mark.parametrize("grouped", [True, False])
@pytest.mark.parametrize("nan", [False, True, "group"])
def test_native_range_partials_merge_to_the_pipeline_kernels_partials(grouped, nan):
    """The executor's partials for two row ranges merge to what the pipeline kernel gives for all rows:
    every aggregate kind (sum, mean, min / max with and without NaN, count) and the NaN counts behind them."""
    from omxcore.custom_ops import CustomOpsCompiler
    from omxcore.pipeline_program import finalize, merge_partials
    t = data(200_000, groups=300, nan=nan)
    if nan == "group" and not grouped:
        pytest.skip("an all-NaN group needs keys")
    plan, got = _inputs(t, grouped)
    gids, n_groups = _gids(CustomOpsCompiler, got)
    comp = CustomOpsCompiler(device="cpu")
    n = t.num_rows
    cut = n // 3
    parts = []
    for a, b in ((0, cut), (cut, n)):
        p = nx.range_partials(comp, got.prog, t, [plan.input.predicate], plan, gids, n_groups, a, b)
        assert p is not None, nx.last_decline()
        parts.append(p)
    want = pk.run_pipeline_cpu(got.prog, got.cc, gids, n_groups, 0, n)
    have = merge_partials(got.prog, parts)
    pres_w, cols_w = finalize(got.prog, want, grouped)
    pres_h, cols_h = finalize(got.prog, have, grouped)
    assert (pres_w == pres_h).all()
    for name in cols_w:
        np.testing.assert_allclose(np.asarray(cols_h[name], dtype=float), np.asarray(cols_w[name], dtype=float),
                                   rtol=1e-9, equal_nan=True, err_msg=name)


@needs_native
def test_native_range_over_an_empty_or_filtered_out_range_is_the_identity():
    from omxcore.custom_ops import CustomOpsCompiler
    t = data(50_000, groups=5)
    plan, got = _inputs(t, grouped=False)
    comp = CustomOpsCompiler(device="cpu")
    empty = nx.range_partials(comp, got.prog, t, [plan.input.predicate], plan, None, 1, 100, 100)
    assert empty.cnt.tolist() == [0] and np.array_equal(empty.vals[:, 0], got.prog.identity)
    none_pass = Aggregate(plan.input, [], plan.aggs)
    none_pass.input.predicate = col("d") > lit(1000)
    got_none = nx.range_partials(comp, got.prog, t, [none_pass.input.predicate], none_pass, None, 1, 0, 50_000)
    assert got_none.cnt.tolist() == [0] and np.array_equal(got_none.vals[:, 0], got.prog.identity)


@needs_gpu
@needs_native
@pytest.mark.parametrize("fraction", [0.25, 0.6, 1.0])
@pytest.mark.parametrize("shape", ["grouped", "global", "nan", "nan group"])
def test_forced_splits_equal_device_cpu_and_duckdb(fraction, shape, monkeypatch):
    t = data(nan={"nan": True, "nan group": "group"}.get(shape, False))
    make = lambda: agg_query(LazyFrame(t), grouped=shape != "global")
    want_cpu = make().compute(device="cpu")
    ran = []
    real = nx.range_partials

    def spy(*a, **k):
        got = real(*a, **k)
        ran.append(got is not None)
        return got

    monkeypatch.setattr(nx, "range_partials", spy)
    cf.clear_residency()
    with cf.policy(min_upload_bytes=0, gpu_fraction=fraction):
        make().compute(device="cpu")
        cf.wait_for_uploads(30)
        lf = make()
        got = lf.compute(device="cpu")
    prov = lf.last_compile_provenance
    assert prov["path"] == ("fused_pipeline_gpu_resident" if fraction == 1.0 else "fused_pipeline_split"), prov
    assert prov["coproc"]["gpu_fraction"] == fraction
    assert any(ran), "the native executor never took a CPU share: the test would prove nothing"
    same(got, want_cpu)
    same(got, duck(GROUPED_SQL if shape != "global" else GLOBAL_SQL, t=t))


@needs_gpu
@needs_native
def test_the_cpu_share_runs_on_the_native_executor_not_the_old_kernel(monkeypatch):
    t = data()
    cf.clear_residency()
    old_kernel = []
    real = pk.run_pipeline_cpu
    monkeypatch.setattr(pk, "run_pipeline_cpu", lambda *a, **k: old_kernel.append(a[-2:]) or real(*a, **k))
    with cf.policy(min_upload_bytes=0, gpu_fraction=0.5):
        agg_query(LazyFrame(t)).compute(device="cpu")        # host data: the whole range on the CPU
        cf.wait_for_uploads(30)
        got = agg_query(LazyFrame(t)).compute(device="cpu")  # resident: a split
    assert old_kernel == [], f"the pipeline kernel ran for ranges {old_kernel}"
    same(got, duck(GROUPED_SQL, t=t))


# ── 3. one predicate for what the split takes ───────────────────────────────────────────────────────────────

def _shapes():
    t = data(60_000, groups=20)
    nulls = pa.table({"k": pa.array([1, 2, 3, None] * 15_000, type=pa.int32()),
                      "v": pa.array(np.ones(60_000, dtype=np.float64))})
    big = pa.table({"k": pa.array(np.zeros(60_000, dtype=np.int32)),
                    "i": pa.array(np.full(60_000, (1 << 53) + 1, dtype=np.int64)),
                    "v": pa.array(np.ones(60_000))})
    strings = pa.table({"k": pa.array(np.zeros(60_000, dtype=np.int32)),
                        "s": pa.array(["a", "b", "c", "d"] * 15_000), "v": pa.array(np.ones(60_000))})
    dim = pa.table({"k": pa.array(np.arange(20, dtype=np.int32)), "g": pa.array(np.arange(20) % 3)})
    s = lambda e: e.alias("s")
    return {
        "plain": (LazyFrame(t).groupby("k").agg(s(col("v").sum())), True),
        "filter": (LazyFrame(t).filter(col("d") > lit(5)).groupby("k").agg(s(col("v").sum())), True),
        "global": (LazyFrame(t).groupby().agg(s(col("v").sum())), True),
        "sort and limit on top": (LazyFrame(t).groupby("k").agg(s(col("v").sum())).sort("k").limit(3), True),
        "project below (inlined)": (LazyFrame(t).with_columns((col("v") * lit(2.0)).alias("v2"))
                                    .groupby("k").agg(s(col("v2").sum())), True),
        "distinct below": (LazyFrame(t).select("k", "j").distinct().groupby("k").agg(s(col("j").count())), False),
        "nulls": (LazyFrame(nulls).groupby("k").agg(s(col("v").sum())), False),
        "int64 past 2^53": (LazyFrame(big).filter(col("i") > lit(5)).groupby("k").agg(s(col("i").max())), False),
        "int64 max past 2^53, in int64 (F-060c)": (LazyFrame(big).groupby("k").agg(s(col("i").max())), True),
        "integer sum of int64 past 2^53, in int64 (F-060b)": (LazyFrame(big).groupby("k").agg(s(col("i").sum())), True),
        "dictionary-code filter, in-memory": (LazyFrame(strings).filter(col("s") == lit("a"))
                                              .groupby("k").agg(s(col("v").sum())), False),
        "join": (LazyFrame(t).join(LazyFrame(dim), on="k").groupby("g").agg(s(col("v").sum())), False),
        "derived key": (LazyFrame(t).groupby((col("k") + lit(1)).alias("k1")).agg(s(col("v").sum())), False),
        "renamed key": (LazyFrame(t).groupby(col("k").alias("kk")).agg(s(col("v").sum())), False),
    }


@pytest.mark.parametrize("name", list(_shapes()))
def test_split_predicate_matches_what_the_split_path_takes(name):
    """`split_check` accepts a plan exactly when `try_fused_aggregate` runs it AND the split can key its
    columns (stable group ids, buffers a device copy can be tied to): both answer the same question."""
    from omxcore.custom_ops import CustomOpsCompiler
    lf, accepted = _shapes()[name]
    verdict = pe.split_check(lf.plan)
    assert (not isinstance(verdict, str)) == accepted, verdict
    if not accepted:
        assert isinstance(verdict, str) and verdict
        return
    # what the split path does: the fused pipeline binds the same program and columns the verdict holds
    node = lf.plan
    while not isinstance(node, Aggregate):
        node = node.input
    inlined = CustomOpsCompiler._inline_project_into_aggregate(node)
    bound = pe._bind(*pe._table_of(inlined)[:2], pe._table_of(inlined)[2])
    assert not isinstance(bound, str)
    assert verdict.prog.columns == bound[0].columns


def test_a_shape_the_fused_pipeline_declines_is_declined_by_the_predicate_with_its_reason():
    shapes = _shapes()
    for name in ("nulls", "int64 past 2^53"):
        verdict = pe.split_check(shapes[name][0].plan)
        assert "nulls" in verdict if name == "nulls" else "2^53" in verdict
    assert "Aggregate(Filter*(Scan))" in pe.split_check(shapes["join"][0].plan)
    assert "derived or renamed" in pe.split_check(shapes["derived key"][0].plan)
    assert "converted on every query" in pe.split_check(shapes["dictionary-code filter, in-memory"][0].plan)


# ── 4. the split decision and its numbers ────────────────────────────────────────────────────────────────────

def test_plan_split_prices_the_split_against_the_faster_device_with_all_its_costs():
    p = cf.plan_split(cpu_ms=100.0, gpu_ms=30.0, n_groups=4, n_slots=3)
    # the cpu share runs x1.05 slower beside the gpu; the gpu call costs 4.5 ms on top of its 30 ms compute:
    # balance (1-f)*105 == 4.5 + f*30
    assert p.contended_cpu_ms == pytest.approx(105.0) and p.gpu_alone_ms == pytest.approx(34.5)
    assert p.gpu_fraction == pytest.approx((105.0 - 4.5) / (105.0 + 30.0))
    assert p.merge_ms == pytest.approx(0.5 + 4 * 4 * 8 / 4_000_000)
    assert p.split_ms == pytest.approx((1 - p.gpu_fraction) * 105.0 + p.merge_ms)
    assert p.wins and p.split_ms < 34.5 * 0.9


def test_a_split_that_saves_less_than_ten_percent_is_not_chosen():
    p = cf.plan_split(cpu_ms=8.0, gpu_ms=1.0)           # split ~ 5.4 ms against 5.5 ms for the gpu alone
    assert p.gpu_alone_ms == pytest.approx(5.5)
    assert p.split_ms >= min(p.cpu_alone_ms, p.gpu_alone_ms) * 0.9 and not p.wins
    assert cf.plan_split(cpu_ms=60.0, gpu_ms=30.0).wins            # a clear win keeps winning


def test_the_per_call_cost_decides_small_inputs():
    """6M rows of a Q1-shaped query: ~5.5 ms on the CPU and ~1.9 ms of GPU compute (this box's calibration)
    still lose to the CPU alone once the GPU's 4.5 ms per call is paid."""
    assert not cf.plan_split(cpu_ms=5.5, gpu_ms=1.9).wins


def test_a_hopeless_gpu_or_cpu_is_one_device_alone():
    assert cf.plan_split(1.0, 500.0).gpu_fraction == 0.0 and not cf.plan_split(1.0, 500.0).wins
    gpu_alone = cf.plan_split(500.0, 20.0)
    assert gpu_alone.gpu_fraction == 1.0 and gpu_alone.wins and "resident gpu alone" in gpu_alone.explain()


class _Calib:
    has_gpu = True

    def __init__(self, cpu_ms, gpu_ms):
        self.cpu_ms, self.gpu_ms = cpu_ms, gpu_ms

    def estimate(self, op, device, rows, card=None, **k):
        if op != "pipeline":
            return 10.0 if device == "cpu" else 1e6
        return self.cpu_ms if device == "cpu" else self.gpu_ms

    def estimate_gpu_join_total(self, *a, **k):
        return 1e6

    def transfer_ms(self, *a, **k):
        return 0.0

    def gpu_init_cost_ms(self):
        return 200.0

    def observe(self, *a, **k):
        pass


class _Warm:
    available = True

    def remaining_init_ms(self):
        return 0.0

    def start(self):
        return self


def _decide(monkeypatch, table_, cpu_ms=100.0, gpu_ms=30.0, resident=False, grouped=True):
    monkeypatch.setattr(cf, "MIN_UPLOAD_BYTES", 1)
    monkeypatch.setattr(cal, "get_calibration", lambda: _Calib(cpu_ms, gpu_ms))
    monkeypatch.setattr(am, "get_gpu_warmup", lambda calib=None, start=True: _Warm())
    monkeypatch.setattr(am, "gpu_warmup_started", lambda: True)
    monkeypatch.setattr(cf, "resident_for", lambda *a: resident)
    lf = agg_query(LazyFrame(table_), grouped=grouped)
    return lf, lf.explain(device="auto", costs=True)


def test_explain_shows_the_numbers_of_a_split_it_chose(monkeypatch):
    lf, text = _decide(monkeypatch, data(), resident=True)
    line = next(l for l in text.splitlines() if "cpu path:" in l)
    assert "cpu path: cpu+gpu split (est " in line
    for part in ("gpu share 0.", "cpu alone 100.0ms", "contention when sharing (assumed, unmeasured)", "gpu alone 34.5ms (compute 30.0 + 4.5 per call)",
                 "merge ", "vs best single device 34.5ms", "needs 10%"):
        assert part in line, line


def test_explain_says_why_not_when_the_split_does_not_win(monkeypatch):
    lf, text = _decide(monkeypatch, data(), cpu_ms=8.0, gpu_ms=1.0, resident=True)
    line = next(l for l in text.splitlines() if "cpu path:" in l)
    assert line.strip().startswith("cpu path: native executor (no cpu+gpu split: est ") and "needs 10%" in line, line


def test_without_resident_columns_the_query_runs_native_and_queues_the_upload(monkeypatch):
    lf, text = _decide(monkeypatch, data(), resident=False)
    line = next(l for l in text.splitlines() if "cpu path:" in l)
    assert "native executor (columns not resident on the GPU yet, upload queued (pending checks); once resident: est " in line
    d = am.choose_device(lf.plan, calib=_Calib(100.0, 30.0), gpu=_Warm())
    assert d.upload is True and d.split is False


def test_a_non_resident_auto_decision_reads_no_column(monkeypatch):
    """Deciding a query whose columns are not resident compiles the program and asks residency: no pass over
    int64 columns or integer sums (they cost as much as a native query, on every decision, possibly forever)."""
    scans = []
    import omxcore.join_engine as je
    import omxcore.pipeline_program as pp
    real_mm = je._minmax
    monkeypatch.setattr(je, "_minmax", lambda *a: scans.append("minmax") or real_mm(*a))
    real_col, real_sum = pe._column_past_2_53, __import__("omxcore.pipeline_program", fromlist=["x"]).int_sum_inexact
    monkeypatch.setattr(pe, "_column_past_2_53", lambda *a: scans.append("int64") or real_col(*a))
    import omxcore.custom_ops as co
    real_reason = co.CustomOpsCompiler._arrow_aggregate_reason
    monkeypatch.setattr(co.CustomOpsCompiler, "_arrow_aggregate_reason",
                        staticmethod(lambda *a, **k: scans.append("arrow") or real_reason(*a, **k)))
    lf, text = _decide(monkeypatch, data(), resident=False)
    assert "upload queued" in text
    assert scans == []
    # the same decision with the columns resident predicts the split, still without a scan
    lf, text = _decide(monkeypatch, data(), resident=True)
    assert "cpu path: cpu+gpu split" in text and scans == []
    # the full check (background upload, the run) does scan
    assert not isinstance(pe.split_check(lf.plan), str) and "int64" in scans and "arrow" in scans
    assert "minmax" not in scans      # F-060: an integer sum is not sized (no pass over its column)


@pytest.mark.parametrize("kind", ["multi-chunk dictionary", "plain string"])
def test_a_string_filter_in_a_non_resident_decision_does_no_row_work(monkeypatch, kind):
    """A scan_parquet read of several row groups holds a multi-chunk dictionary column; building its program's
    dictionary unifies and rewrites every row's code (and a plain string column is hash-encoded). The decision
    only names columns: none of that runs for it, while the full check (the run) still does it."""
    import omxcore.pipeline_program as pp
    n = 400_000
    rng = np.random.default_rng(4)
    words = np.array(["A", "N", "R"])[rng.integers(0, 3, n)]
    if kind == "plain string":
        f = pa.chunked_array([pa.array(words)])
    else:     # different dictionaries per chunk, as row groups give
        f = pa.chunked_array([pa.array(words[:n // 2]).dictionary_encode(),
                              pa.array(words[n // 2:][::-1]).dictionary_encode()])
    t = pa.table({"k": pa.array(rng.integers(0, 4, n).astype(np.int32)), "v": pa.array(rng.random(n)), "f": f})
    lf = LazyFrame(t).filter(col("f") == lit("R")).groupby("k").agg(col("v").sum().alias("s"))
    work = []
    real_pc, real_combine, real_single = pp.pc, pp._combine, pp._single_array

    class CountingPc:
        def __getattr__(self, name):
            fn = getattr(real_pc, name)
            return (lambda *a, **k: work.append(name) or fn(*a, **k)) if name == "dictionary_encode" else fn

    monkeypatch.setattr(pp, "pc", CountingPc())
    monkeypatch.setattr(pp, "_combine", lambda *a, **k: work.append("combine") or real_combine(*a, **k))
    monkeypatch.setattr(pp, "_single_array", lambda *a, **k: work.append("single_array") or real_single(*a, **k))
    light = pe.split_check(lf.plan, full=False)
    assert not isinstance(light, str), light
    assert light.columns == frozenset({"f", "v"}) and work == []
    full = pe.split_check(lf.plan)
    assert work, "the full check should build the dictionary: the test would prove nothing"


def test_the_light_compile_names_the_same_columns_as_the_real_one():
    from omxcore.pipeline_program import compile_pipeline
    t = data(50_000)
    t = t.append_column("day", pa.array(np.arange(50_000, dtype=np.int32) % 4000))
    aggs = [col("i").sum().alias("s"), (col("i") * col("j")).sum().alias("p"), col("v").sum().alias("v")]
    preds = [col("d") > lit(5)]
    full = compile_pipeline(preds, aggs, t)
    light = compile_pipeline(preds, aggs, t, exact_bounds=False)
    assert light.columns == full.columns and light.columns
    yr = [col("day").year_of_days().sum().alias("y")]
    assert compile_pipeline([], yr, t, exact_bounds=False).columns == compile_pipeline([], yr, t).columns


@needs_native
@needs_gpu
def test_a_cpu_decision_with_a_queued_upload_runs_native_then_starts_the_upload(monkeypatch):
    t = data(400_000)
    calls = []
    monkeypatch.setattr(pe, "queue_resident_upload", lambda plan: calls.append(plan))
    monkeypatch.setattr(cf, "MIN_UPLOAD_BYTES", 1)
    monkeypatch.setattr(cal, "get_calibration", lambda: _Calib(100.0, 30.0))
    monkeypatch.setattr(am, "get_gpu_warmup", lambda calib=None, start=True: _Warm())
    monkeypatch.setattr(am, "gpu_warmup_started", lambda: True)
    monkeypatch.setattr(cf, "IDLE_BEFORE_BACKGROUND_S", 0.0)
    policies = []
    import omxcore.lazy_frame as lfm
    real = lfm._run_on_device
    monkeypatch.setattr(lfm, "_run_on_device", lambda plan, device, **kw: policies.append(cf.current_policy()) or real(plan, device, **kw))
    lf = agg_query(LazyFrame(t))
    got = lf.compute(device="auto")
    assert policies == [None], "a non-resident query must run exactly as device='cpu' (native), no policy"
    assert len(calls) == 1
    same(got, duck(GROUPED_SQL, t=t))


@needs_gpu
@needs_native
def test_auto_splits_once_the_background_upload_has_landed():
    """End to end on the real GPU: first auto query native + queued upload, the repeat finds it and splits."""
    import time
    assert am.get_gpu_warmup().wait(120), "the GPU did not initialise"
    t = data(1_500_000)
    cf.clear_residency()
    orig, real_cal = cf.MIN_UPLOAD_BYTES, cal.get_calibration
    cf.MIN_UPLOAD_BYTES = 1
    fake = _Calib(100.0, 30.0)               # this box's calibration says "no split" at a size a test can afford
    fake.estimate_cpu_only = None
    cal.get_calibration = lambda: fake
    try:
        want = duck(GROUPED_SQL, t=t)
        first = agg_query(LazyFrame(t))
        same(first.compute(device="auto"), want)
        d1 = first.last_auto_decision
        if not d1.upload:                       # calibration says a split would not win here: nothing to prove
            pytest.skip(f"model does not predict a split on this box: {d1.cpu_path}")
        deadline = time.monotonic() + 30
        while not cf.resident_count() and time.monotonic() < deadline:
            time.sleep(0.05)
        cf.wait_for_uploads(30)
        assert cf.resident_count() == 1
        second = agg_query(LazyFrame(t))
        same(second.compute(device="auto"), want)
        assert "cpu+gpu split" in second.last_auto_decision.cpu_path
        assert second.last_compile_provenance["path"] in ("fused_pipeline_split", "fused_pipeline_gpu_resident")
    finally:
        cf.MIN_UPLOAD_BYTES, cal.get_calibration = orig, real_cal


@needs_gpu
@needs_native
@pytest.mark.parametrize("mode", [True, False])
def test_the_cpu_share_runs_in_the_callers_reproducible_mode(monkeypatch, mode):
    """The router's worker threads do not inherit the context variable behind `compute(reproducible=...)`
    (F-064): the share must be told, or a reproducible query's CPU half would sum in the default order."""
    t = data()
    seen = []
    real = nx.range_partials

    def spy(*a, **k):
        seen.append(nx.reproducible())
        return real(*a, **k)

    monkeypatch.setattr(nx, "range_partials", spy)
    cf.clear_residency()
    with cf.policy(min_upload_bytes=0, gpu_fraction=0.5):
        agg_query(LazyFrame(t)).compute(device="cpu", reproducible=mode)
        cf.wait_for_uploads(30)
        seen.clear()
        got = agg_query(LazyFrame(t)).compute(device="cpu", reproducible=mode)
    assert seen and set(seen) == {mode}
    same(got, duck(GROUPED_SQL, t=t))
