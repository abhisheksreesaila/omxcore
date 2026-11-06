"""F-126: `avg` of integers never raises on an integer overflow, and is DuckDB's exact double.

DuckDB adds an integer column in a HUGEINT and divides once, so `avg` of values near 2^60 is a double even
when the total passes int64 (F-060 made the engine's `sum` raise there, by design, and `avg` inherited the
raise: a query that used to answer failed). `avg` now falls back to an exact wider sum on every path, and its
division is rounded once: a total between 2^53 and 2^63 used to be converted to a double first and then
divided (1 ulp off DuckDB's value for about one table in five).

Every answer is compared with DuckDB's, bit for bit, and the path that answered is asserted. `sum` over the same
data still raises."""
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import omxcore as mx
import omxcore.native_exec as nx
from omxcore import LazyFrame, col, gpu_stream
from omxcore.custom_ops import CustomOpsCompiler
from omxcore.io_parquet import scan_parquet
from omxcore.np_arrow import array as A

HAS_GPU = gpu_stream.cuda_available()
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1
N = 30_000

MODES = [
    pytest.param("1", "cpu", id="native"),
    pytest.param("0", "cpu", id="pipeline"),
    pytest.param("0", "gpu", id="gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
]
KERNEL = {("1", "cpu"): "native", ("0", "cpu"): "fused_pipeline_cpu", ("0", "gpu"): "fused_pipeline_gpu"}
ARROW = "arrow_null_semantics"


def _table():
    r = np.random.default_rng(126)
    k = np.arange(N)
    sign = np.where(r.random(N) < 0.5, -1, 1)
    over = ((1 << 60) + r.integers(0, 1 << 40, N)).astype(np.int64)            # 6,000 per group: ~7e21, past int64
    over_mixed = (over * sign).astype(np.int64)                                # both signs: totals near zero
    wide = r.integers(1 << 52, 1 << 53, N).astype(np.int64)                    # each value fits a double; the total (2^65) does not fit int64
    fits = r.integers(1 << 42, 1 << 43, N).astype(np.int64)                    # ~2^57 total: fits, past 2^53
    extreme = np.where(k % 2 == 0, I64_MAX, I64_MIN + (k % 3)).astype(np.int64)
    extreme[:2] = [I64_MAX, I64_MIN]
    g = (k % 5).astype(np.int64)
    extreme[g == 4] = I64_MAX
    return pa.table({
        "over": A(over), "wide": A(wide), "mixed": A(over_mixed), "fits": A(fits), "extreme": A(extreme),
        "i32": A(r.integers(-2 ** 31, 2 ** 31, N).astype(np.int32)),
        "nul": pa.array([None if x % 7 == 0 else int(v) for x, v in zip(k, over)], pa.int64()),
        "g": A(g), "f": A(r.random(N)),
    })


T = _table()
_DUCK = duckdb.connect()
_DUCK.register("t", T)
AVGS = ["over", "wide", "mixed", "fits", "extreme", "i32"]


def _want(q):
    return _DUCK.execute(q).fetchall()


@pytest.fixture
def run(monkeypatch):
    """run(builder, native, device) -> (table, paths of every plan run)."""
    paths = []
    real = CustomOpsCompiler._compile_and_run

    def recording(self, *a, **k):
        out = real(self, *a, **k)
        paths.append((self.last_compile_provenance or {}).get("path"))
        return out

    monkeypatch.setattr(CustomOpsCompiler, "_compile_and_run", recording)

    def go(build, native, device, table=T, **kw):
        monkeypatch.setenv("MXFRAME_NATIVE", native)
        if native == "1" and not nx.available():
            pytest.skip("libmxexec_cpu.so not built")
        paths.clear()
        return build(LazyFrame(table)).compute(device=device, **kw), list(paths)
    return go


def _answered_by(c, native, device, paths):
    """A column with a value past 2^53 is not loaded by the kernels (they read float64 lanes), and a total past int64
    is declined by them (E_OVERFLOW): Arrow's wide sum answers both. The others stay on the kernels."""
    return {KERNEL[(native, device)]} if c in ("fits", "i32") else {ARROW}


# -- global and grouped, against DuckDB's doubles --------------------------------------------------------------

@pytest.mark.parametrize("c", AVGS)
@pytest.mark.parametrize("native,device", MODES)
def test_a_global_integer_avg_is_duckdbs_double(run, native, device, c):
    got, paths = run(lambda f: f.groupby().agg(col(c).mean().alias("m")), native, device)
    assert got.schema.field("m").type == pa.float64()
    assert got.column("m").to_pylist() == [_want(f"select avg({c}) from t")[0][0]]
    assert set(paths) == _answered_by(c, native, device, paths), paths


@pytest.mark.parametrize("c", AVGS)
@pytest.mark.parametrize("native,device", MODES)
def test_a_grouped_integer_avg_is_duckdbs_double(run, native, device, c):
    got, paths = run(lambda f: f.groupby("g").agg(col(c).mean().alias("m")).sort("g"), native, device)
    assert got.schema.field("m").type == pa.float64()
    assert got.column("m").to_pylist() == [r[1] for r in _want(f"select g, avg({c}) from t group by g order by g")]
    assert set(paths) == _answered_by(c, native, device, paths), paths


@pytest.mark.parametrize("native,device", MODES)
@pytest.mark.parametrize("c", ["over", "wide"])
def test_sum_over_the_same_data_still_raises_and_avg_does_not(run, native, device, c):
    with pytest.raises(OverflowError, match="sum of INT64"):
        run(lambda f: f.groupby().agg(col(c).sum().alias("s")), native, device)
    with pytest.raises(OverflowError, match="sum of INT64"):
        run(lambda f: f.groupby("g").agg(col(c).sum().alias("s")), native, device)
    got, _ = run(lambda f: f.groupby("g").agg(col(c).mean().alias("m")).sort("g"), native, device)
    assert got.column("m").to_pylist() == [r[1] for r in _want(f"select g, avg({c}) from t group by g order by g")]


@pytest.mark.parametrize("native,device", MODES)
def test_avg_next_to_other_aggregates_over_an_overflowing_column(run, native, device):
    got, paths = run(lambda f: f.groupby("g").agg(col("over").mean().alias("m"), col("over").min().alias("lo"),
                                                  col("over").max().alias("hi"), col("over").count().alias("n"),
                                                  col("f").mean().alias("fm")).sort("g"), native, device)
    want = _want("select g, avg(over), min(over), max(over), count(over), avg(f) from t group by g order by g")
    rows = [tuple(r[n] for n in ("g", "m", "lo", "hi", "n", "fm")) for r in got.to_pylist()]
    assert [r[:5] for r in rows] == [r[:5] for r in want]
    assert [r[5] for r in rows] == pytest.approx([r[5] for r in want], rel=1e-12)
    assert set(paths) == {ARROW}, paths


def test_avg_with_a_filter_and_an_expression():
    got = (LazyFrame(T).filter(col("g") > 1).groupby("g").agg((col("over") + 1).mean().alias("m")).sort("g")
           .compute(device="cpu"))
    assert got.column("m").to_pylist() == [r[1] for r in _want("select g, avg(over + 1) from t where g > 1 group by g order by g")]


# -- NULLs: the NULL-aware path counts the valid rows ------------------------------------------------------------

@pytest.mark.parametrize("native", ["1", "0"])
def test_avg_of_a_column_with_nulls_divides_by_the_valid_rows(run, native):
    got, paths = run(lambda f: f.groupby().agg(col("nul").mean().alias("m")), native, "cpu")
    assert got.column("m").to_pylist() == [_want("select avg(nul) from t")[0][0]]
    got, paths = run(lambda f: f.groupby("g").agg(col("nul").mean().alias("m")).sort("g"), native, "cpu")
    assert got.column("m").to_pylist() == [r[1] for r in _want("select g, avg(nul) from t group by g order by g")]
    assert set(paths) == {ARROW}, paths


def test_a_group_of_only_nulls_and_an_empty_input_average_to_null():
    t = pa.table({"x": pa.array([None, None, I64_MAX, I64_MAX, I64_MAX], pa.int64()), "g": A(np.array([0, 0, 1, 1, 1]))})
    got = LazyFrame(t).groupby("g").agg(col("x").mean().alias("m")).sort("g").compute(device="cpu")
    assert got.column("m").to_pylist() == [None, float(I64_MAX)]
    empty = LazyFrame(T).filter(col("g") > 99).groupby().agg(col("over").mean().alias("m")).compute(device="cpu")
    assert empty.column("m").to_pylist() == [None]


# -- rounding once: a total between 2^53 and 2^63 that fits int64 --------------------------------------------------

@pytest.mark.parametrize("native,device", MODES)
def test_a_total_that_fits_int64_but_not_a_double_is_divided_once(run, native, device):
    """float(total) / n and total / n differ by an ulp for some inputs: 60 tables, none may differ from DuckDB."""
    for seed in range(60):
        x = np.random.default_rng(seed).integers(1 << 42, 1 << 43, 20_001).astype(np.int64)
        t = pa.table({"x": A(x), "g": A((np.arange(len(x)) % 3).astype(np.int64))})
        con = duckdb.connect()
        con.register("t", t)
        got, paths = run(lambda f: f.groupby().agg(col("x").mean().alias("m")), native, device, table=t)
        assert got.column("m").to_pylist() == [con.execute("select avg(x) from t").fetchone()[0]], seed
        assert set(paths) == {KERNEL[(native, device)]}, paths
        got, paths = run(lambda f: f.groupby("g").agg(col("x").mean().alias("m")).sort("g"), native, device, table=t)
        assert got.column("m").to_pylist() == [r[1] for r in con.execute("select g, avg(x) from t group by g order by g").fetchall()], seed


def test_an_order_by_or_having_over_an_integer_mean_sees_the_mean():
    """The native executor sorts what it computes: a mean's total sorts like its mean only when counts agree."""
    r = np.random.default_rng(5)
    n = 40_000
    g = (r.integers(0, 6, n) ** 2).astype(np.int64)                 # groups of very different sizes
    x = r.integers(1 << 42, 1 << 43, n).astype(np.int64)
    t = pa.table({"x": A(x), "g": A(g)})
    con = duckdb.connect()
    con.register("t", t)
    got = LazyFrame(t).groupby("g").agg(col("x").mean().alias("m")).sort("m", descending=True).compute(device="cpu")
    want = con.execute("select g, avg(x) as m from t group by g order by m desc").fetchall()
    assert [(r["g"], r["m"]) for r in got.to_pylist()] == want
    got = mx.sql("select g, avg(x) as m from t group by g having avg(x) > 5400000000000 order by g", t=t).compute(device="cpu")
    assert [(r["g"], r["m"]) for r in got.to_pylist()] == con.execute(
        "select g, avg(x) as m from t group by g having avg(x) > 5400000000000 order by g").fetchall()


# -- SQL ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("native,device", MODES)
@pytest.mark.parametrize("q", [
    "select avg(over) as m from t", "select avg(wide) as m from t", "select g, avg(over) as m from t group by g order by g",
    "select g, avg(mixed) as m, avg(extreme) as e, avg(wide) as w from t group by g order by g",
    "select avg(extreme) / 2 as m from t",
])
def test_sql_avg_equals_duckdb(monkeypatch, native, device, q):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    got = mx.sql(q, t=T).compute(device=device)
    assert [tuple(r.values()) for r in got.to_pylist()] == _want(q)


# -- streamed (F-056): batches and the merge hold wide sums -------------------------------------------------------------

@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "global"])
@pytest.mark.parametrize("c", ["over", "wide", "mixed", "fits"])
def test_a_streamed_integer_avg_is_duckdbs_double_at_every_budget(tmp_path, monkeypatch, c, grouped):
    path = str(tmp_path / "t.parquet")
    pq.write_table(T, path, row_group_size=100)
    steps = []
    for budget in ("5K", "20K", "64G"):
        monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", budget)
        s = scan_parquet(path)
        lf = (s.groupby("g") if grouped else s.groupby()).agg(col(c).mean().alias("m"))
        got = lf.compute(device="cpu")
        steps.append(lf.last_compile_provenance.get("streamed", {}).get("batches"))
        want = _want(f"select {'g, ' if grouped else ''}avg({c}) from t{' group by g' if grouped else ''}")
        rows = sorted(got.to_pylist(), key=lambda r: r.get("g", 0))
        assert [tuple(r.values()) for r in rows] == sorted(want), budget
    assert steps[2] is None and steps[0] > steps[1] > 1, steps


def test_a_streamed_sum_of_the_same_column_still_raises(tmp_path, monkeypatch):
    path = str(tmp_path / "t.parquet")
    pq.write_table(T, path, row_group_size=100)
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "20K")
    with pytest.raises(OverflowError, match="sum of INT64"):
        scan_parquet(path).groupby("g").agg(col("over").sum().alias("s")).compute(device="cpu")


# -- a CPU+GPU split ------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")
@pytest.mark.parametrize("fraction", [0.0, 0.3, 0.7, 1.0])
@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "global"])
@pytest.mark.parametrize("c", ["over", "wide", "fits"])
def test_a_cpu_gpu_split_averages_like_duckdb(fraction, grouped, c):
    import omxcore.coproc_fused as cf
    cf.clear_residency()
    r = np.random.default_rng(7)
    rows = 600_000
    x = {"over": (1 << 60) + r.integers(0, 1 << 40, rows), "wide": r.integers(1 << 52, 1 << 53, rows),
         "fits": r.integers(1 << 30, 1 << 31, rows)}[c]
    t = pa.table({"x": A(x.astype(np.int64)), "g": A((np.arange(rows) % 37).astype(np.int64))})
    con = duckdb.connect()
    con.register("t", t)

    def query():
        s = LazyFrame(t).groupby("g") if grouped else LazyFrame(t).groupby()
        return s.agg(col("x").mean().alias("m"))
    with cf.policy(min_upload_bytes=0, gpu_fraction=fraction):
        query().compute(device="cpu")
        cf.wait_for_uploads(30)
        lf = query()
        got = lf.compute(device="cpu")
    want = con.execute(f"select {'g, ' if grouped else ''}avg(x) from t{' group by g' if grouped else ''}").fetchall()
    rows_out = sorted(got.to_pylist(), key=lambda r: r.get("g", 0))
    assert [tuple(r.values()) for r in rows_out] == sorted(want)
    if 0.0 < fraction < 1.0 and c == "fits":
        assert lf.last_compile_provenance["path"] == "fused_pipeline_split", lf.last_compile_provenance["path"]


# -- over a join ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "select grp, avg(big) as m from f join d on f.k = d.k group by grp order by grp",
    "select avg(big) as m from f join d on f.k = d.k where grp = 1",
    "select avg(small) as m from f join d on f.k = d.k",
])
def test_an_integer_avg_over_a_join_equals_duckdb(q):
    n = 50_000
    r = np.random.default_rng(3)
    f = pa.table({"k": A(r.integers(0, 100, n)), "big": A((1 << 60) + r.integers(0, 1000, n)),
                  "small": A(r.integers(1 << 40, 1 << 41, n))})
    d = pa.table({"k": A(np.arange(100)), "grp": A(np.arange(100) % 5)})
    con = duckdb.connect()
    con.register("f", f)
    con.register("d", d)
    got = mx.sql(q, f=f, d=d).compute(device="cpu")
    assert [tuple(r.values()) for r in got.to_pylist()] == con.execute(q).fetchall()


def test_a_streamed_avg_whose_group_state_spills_to_disk(tmp_path, monkeypatch):
    """The two sums of a wide mean merge partition by partition like any other hidden column."""
    n, groups = 60_000, 6_000
    r = np.random.default_rng(9)
    t = pa.table({"x": A(((1 << 62) + r.integers(0, 1 << 40, n)).astype(np.int64)),
                  "k": A(np.arange(n) % groups)})
    path = str(tmp_path / "t.parquet")
    pq.write_table(t, path, row_group_size=500)
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "300K")
    lf = scan_parquet(path).groupby("k").agg(col("x").mean().alias("m"), col("x").count().alias("n"))
    got = lf.compute(device="cpu")
    con = duckdb.connect()
    con.register("t", t)
    want = con.execute("select k, avg(x), count(x) from t group by k").fetchall()
    assert lf.last_compile_provenance["streamed"].get("spill"), lf.last_compile_provenance["streamed"]
    assert sorted(tuple(r.values()) for r in got.to_pylist()) == sorted(want)


@pytest.mark.parametrize("native,device", MODES)
def test_the_division_is_duckdbs_long_double_not_the_correctly_rounded_quotient(run, native, device):
    """DuckDB divides its HUGEINT total in a 64-bit-mantissa long double: on this data 1 group in ~2,000 differs from
    `total / n` rounded once (the first, group 575, is 4.6116867269789e+18, not 4.611686726978899e+18)."""
    n, groups = 60_000, 6_000
    r = np.random.default_rng(9)
    x = ((1 << 62) + r.integers(0, 1 << 40, n)).astype(np.int64)
    t = pa.table({"x": A(x), "k": A(np.arange(n) % groups)})
    con = duckdb.connect()
    con.register("t", t)
    want = con.execute("select k, avg(x) from t group by k order by k").fetchall()
    exact = [sum(int(v) for v in x[np.arange(n) % groups == k]) / 10 for k in range(groups)]
    assert sum(w[1] != e for w, e in zip(want, exact)) >= 3
    got, paths = run(lambda f: f.groupby("k").agg(col("x").mean().alias("m")).sort("k"), native, device, table=t)
    assert [tuple(r.values()) for r in got.to_pylist()] == want
    assert set(paths) == {ARROW}, paths


# -- no x87 long double (macOS ARM, Windows): DuckDB divides in doubles there (F-211, pinned against DuckDB's own bits in test_f211_avg_platform.py) ----------

@pytest.fixture
def no_x87(monkeypatch):
    import omxcore.np_arrow as npa
    monkeypatch.setattr(npa, "_X87", False)
    return npa


def _double_mean(total, n):
    """double(total) / n, the total read as DuckDB's HUGEINT cast does (a negative total below 2^64 is -double(-total - 1) - 1.0)."""
    t = -float(-total - 1) - 1.0 if -(1 << 64) <= total < 0 else float(total)
    return t / n


def test_without_an_x87_long_double_the_mean_is_the_total_as_a_double_over_the_count(no_x87):
    npa = no_x87
    big = (1 << 62) + 123
    for total, n in ((-3, 2), (-1 - 2, 2), (7, 3), (-7, 3), (big * 10 + 7, 10), (-(big * 10 + 7), 10), (big * 3, 7), (1 << 100, 3), (-(1 << 100) - 1, 7)):
        assert npa._ld_div(total, n) == _double_mean(total, n), (total, n)
    tot = np.array([-3, 7, -7, (1 << 53) + 1, -(1 << 53) - 3, 5 * (1 << 60) + 11, 0], dtype=np.int64)
    cnt = np.array([2, 3, 3, 7, 10, 9, 0], dtype=np.int64)
    got = npa.int_mean_of_totals(tot, cnt)
    assert got.tolist() == [-1.5, 7 / 3, -7 / 3, _double_mean((1 << 53) + 1, 7), _double_mean(-(1 << 53) - 3, 10), _double_mean(5 * (1 << 60) + 11, 9), 0.0]
    lo = np.array([-1, -2, (1 << 61) + 5, -(1 << 62), 9], dtype=np.int64)
    hi = np.array([0, 0, 3, -(1 << 30), (1 << 31) - 1], dtype=np.int64)
    c = np.array([2, 2, 5, 3, 7], dtype=np.int64)
    want = [_double_mean((int(h) << 32) + int(l), int(k)) for h, l, k in zip(hi, lo, c)]
    assert npa.int_mean_of_halves(lo, hi, c).tolist() == want


def test_the_halves_of_a_total_are_not_added_in_int64_when_a_half_is_large(monkeypatch):
    import omxcore.np_arrow as npa
    lo = np.array([(1 << 62) + 1], dtype=np.int64)          # hi << 32 + lo would wrap int64
    hi = np.array([(1 << 30) - 1], dtype=np.int64)
    c = np.array([3], dtype=np.int64)
    assert npa.int_mean_of_halves(lo, hi, c).tolist() == [npa._ld_div(((1 << 30) - 1 << 32) + (1 << 62) + 1, 3)]


@pytest.mark.parametrize("c", ["over", "wide", "mixed", "fits"])
def test_without_an_x87_long_double_an_average_is_within_an_ulp_of_duckdb(no_x87, c):
    """The x87 imitation switched off (on x86-64 DuckDB divides in a long double): never more than one rounding away."""
    got = LazyFrame(T).groupby("g").agg(col(c).mean().alias("m")).sort("g").compute(device="cpu").column("m").to_pylist()
    want = [r[1] for r in _want(f"select g, avg({c}) from t group by g order by g")]
    assert got == pytest.approx(want, rel=2e-16, abs=0)
    tiny = pa.table({"x": A(np.array([-1, -2], dtype=np.int64))})
    assert LazyFrame(tiny).groupby().agg(col("x").mean().alias("m")).compute(device="cpu").column("m").to_pylist() == [-1.5]


# -- a sink: rows written before an overflow are not written twice ------------------------------------------------------------

def test_a_sink_parquet_whose_streamed_run_overflows_while_merging_spilled_groups_is_written_once(tmp_path, monkeypatch):
    n, groups = 60_000, 6_000
    r = np.random.default_rng(10)
    k = np.arange(n) % groups
    x = np.where(k == 17, (1 << 62) + r.integers(0, 1 << 40, n), r.integers(0, 1 << 20, n)).astype(np.int64)   # one group overflows
    t = pa.table({"x": A(x), "k": A(k)})
    src, dst = str(tmp_path / "t.parquet"), str(tmp_path / "out.parquet")
    pq.write_table(t, src, row_group_size=500)
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "300K")
    lf = scan_parquet(src).groupby("k").agg(col("x").mean().alias("m"), col("x").count().alias("n"))
    import omxcore.streamed as st
    seen = []
    real_reset = st.ParquetOut.reset
    monkeypatch.setattr(st.ParquetOut, "reset", lambda self: (seen.append(self.rows + self.buffered), real_reset(self))[1])
    rows = st.sink_parquet(lf, dst)
    assert seen and seen[0] > 0, seen                 # the first run had already written (or buffered) rows when it overflowed
    con = duckdb.connect()
    con.register("t", t)
    want = con.execute("select k, avg(x), count(x) from t group by k").fetchall()
    out = pq.read_table(dst)
    assert rows == groups == out.num_rows
    assert sorted(tuple(r.values()) for r in out.to_pylist()) == sorted(want)
    assert [f for f in tmp_path.iterdir() if f.name.endswith(".partial")] == []
