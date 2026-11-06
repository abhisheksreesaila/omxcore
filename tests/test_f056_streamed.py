"""F-056 step 2: a query over a Parquet file bigger than memory finishes, with bounded memory.

`MXFRAME_HOST_MEM_BUDGET` and `MXFRAME_GPU_MEM_BUDGET` make a ~200 MB file with 60 row groups "bigger than
the machine". The query is cut into batches of row groups (filter, project, partial aggregate per batch), the
partials are merged, and the result must equal DuckDB's over the same file on cpu, gpu and auto. The path
(`provenance["path"] == "streamed"`, batch count) and the peak memory are asserted.
"""
import math
import os
import re
import subprocess
import sys
import textwrap
import warnings

import duckdb
from conftest import oracle_connection
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from omxcore import LazyFrame, col, gpu_stream
from omxcore.io_parquet import scan_parquet

N = 6_000_000
GROUP_ROWS = 100_000          # 60 row groups of ~3.4 MB decoded
HOST = "40M"
GPU = "8M"

HAS_GPU = gpu_stream.cuda_available()
DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
           pytest.param("auto", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU"))]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("f056")
    rng = np.random.default_rng(56)
    idx = np.arange(N)
    fact = pa.table({
        "k": (idx % 4).astype(np.int32),
        "j": rng.integers(0, 2000, N).astype(np.int32),
        "v": rng.random(N) * 100,
        "q": rng.integers(1, 51, N).astype(np.int64),
        "d": (idx * 2500 // N).astype(np.int32),                        # sorted: a filter on it prunes row groups
        "s": np.array(["A", "B", "C"])[rng.integers(0, 3, N)],
        "m": ((idx * 7919) % 300_000).astype(np.int32),                 # 300k groups
    })
    dim1 = pa.table({"id": np.arange(2000, dtype=np.int32), "region": (np.arange(2000) % 25).astype(np.int32),
                     "w": (np.arange(2000) % 7).astype(np.int64)})
    dim2 = pa.table({"region": np.arange(25, dtype=np.int32), "rname": np.array([f"R{i:02d}" for i in range(25)])})
    out = {n: str(d / f"{n}.parquet") for n in ("fact", "dim1", "dim2")}
    pq.write_table(fact, out["fact"], row_group_size=GROUP_ROWS)
    pq.write_table(dim1, out["dim1"])
    pq.write_table(dim2, out["dim2"])
    return out


@pytest.fixture(autouse=True)
def _budgets(monkeypatch):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", HOST)
    monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", GPU)


def sql(query, files):
    con = oracle_connection()
    for n, p in files.items():
        con.execute(f"CREATE VIEW {n} AS SELECT * FROM read_parquet('{p}')")
    return con.sql(query).arrow().read_all()


def close(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def same(got, want, keys=()):
    """Same columns, same rows (sorted by `keys` when given), floats to 1e-9 relative, everything else exact."""
    assert got.column_names == want.column_names
    assert got.num_rows == want.num_rows
    g, w = got.to_pylist(), want.to_pylist()
    if keys:
        order = lambda r: tuple((r[k] is None, r[k]) for k in keys)       # noqa: E731
        g, w = sorted(g, key=order), sorted(w, key=order)
    for gr, wr in zip(g, w):
        for name in want.column_names:
            assert close(gr[name], wr[name]), (name, gr, wr)


def prov(lf):
    return lf.last_compile_provenance


def run(lf, device, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)         # gpu says when a piece leaves the GPU
        return lf.compute(device=device, **kw)


def assert_streamed(lf, min_batches=3):
    p = prov(lf)
    assert p["path"] == "streamed", p
    assert p["streamed"]["batches"] >= min_batches
    return p["streamed"]


# --- TPC-H shaped aggregates ---------------------------------------------------------------------------------

def q1(files):
    lf = (scan_parquet(files["fact"]).filter(col("d") <= 2000).groupby("k", "s")
          .agg(col("q").sum().alias("sum_q"), col("v").sum().alias("sum_v"),
               (col("v") * col("q")).sum().alias("sum_vq"), col("v").mean().alias("avg_v"),
               col("q").mean().alias("avg_q"), col("v").min().alias("min_v"), col("q").max().alias("max_q"),
               col("v").count().alias("n")))
    want = sql("SELECT k, s, sum(q) AS sum_q, sum(v) AS sum_v, sum(v * q) AS sum_vq, avg(v) AS avg_v, "
               "avg(q) AS avg_q, min(v) AS min_v, max(q) AS max_q, count(v) AS n FROM fact WHERE d <= 2000 "
               "GROUP BY k, s", files)
    return lf, want, ("k", "s")


def q6(files):
    lf = (scan_parquet(files["fact"]).filter((col("d") >= 500) & (col("d") < 1500) & (col("q") < 24))
          .groupby().agg((col("v") * col("q")).sum().alias("revenue"), col("v").count().alias("n")))
    want = sql("SELECT sum(v * q) AS revenue, count(v) AS n FROM fact WHERE d >= 500 AND d < 1500 AND q < 24", files)
    return lf, want, ()


def q3(files):
    """Fact x two nested dimensions, a HAVING-like filter, ORDER BY ... LIMIT above the aggregate."""
    fact, dim1, dim2 = (scan_parquet(files[n]) for n in ("fact", "dim1", "dim2"))
    lf = (fact.filter(col("d") < 2200)
          .join(dim1.join(dim2, on="region"), left_on="j", right_on="id")
          .groupby("rname", "w").agg((col("v") * col("q")).sum().alias("revenue"), col("v").count().alias("n"))
          .sort("revenue", descending=True).limit(7))
    want = sql("SELECT rname, w, sum(v * q) AS revenue, count(v) AS n FROM fact JOIN dim1 ON fact.j = dim1.id "
               "JOIN dim2 ON dim1.region = dim2.region WHERE d < 2200 GROUP BY rname, w "
               "ORDER BY revenue DESC LIMIT 7", files)
    return lf, want, ()


def q5_dim_filter(files):
    """A filtered dimension (the build side shrinks) and a left-deep join; group by a dimension column."""
    fact, dim1 = scan_parquet(files["fact"]), scan_parquet(files["dim1"]).filter(col("w") < 3)
    lf = (fact.join(dim1, left_on="j", right_on="id").groupby("region")
          .agg(col("v").sum().alias("rev"), col("q").sum().alias("sq")))
    want = sql("SELECT region, sum(v) AS rev, sum(q) AS sq FROM fact JOIN dim1 ON fact.j = dim1.id "
               "WHERE w < 3 GROUP BY region", files)
    return lf, want, ("region",)


def many_groups(files):
    lf = (scan_parquet(files["fact"]).groupby("m")
          .agg(col("v").sum().alias("sv"), col("q").sum().alias("sq"), col("v").count().alias("n")))
    want = sql("SELECT m, sum(v) AS sv, sum(q) AS sq, count(v) AS n FROM fact GROUP BY m", files)
    return lf, want, ("m",)


QUERIES = {"q1": q1, "q6": q6, "q3": q3, "q5": q5_dim_filter, "many_groups": many_groups}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(QUERIES))
def test_a_file_bigger_than_the_budget_is_streamed_and_equals_duckdb(files, name, device):
    lf, want, keys = QUERIES[name](files)
    got = run(lf, device)
    info = assert_streamed(lf)
    same(got, want, keys)
    assert info["host_budget"] == 40 << 20
    assert sum(info["devices"].values()) == info["batches"]
    if device == "cpu":
        assert set(info["devices"]) == {"cpu"}


def test_without_a_budget_that_is_exceeded_nothing_streams(files, monkeypatch):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "64G")
    lf, want, keys = q1(files)
    got = run(lf, "cpu")
    assert prov(lf)["path"] != "streamed"
    same(got, want, keys)


def test_a_streamed_result_has_the_dtypes_of_the_in_memory_one(files, monkeypatch):
    lf, _, _ = q1(files)
    streamed = run(lf, "cpu")
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "64G")
    whole = run(q1(files)[0], "cpu")
    assert streamed.schema == whole.schema


def types_by_budget(path, build, budgets, device="cpu"):
    """{budget: (schema, rows)} of the same query at several host budgets (so several batch counts)."""
    out = {}
    before = os.environ.get("MXFRAME_HOST_MEM_BUDGET")
    try:
        for b in budgets:
            os.environ["MXFRAME_HOST_MEM_BUDGET"] = b
            lf = build(scan_parquet(path))
            got = run(lf, device)
            out[b] = (got.schema, got.sort_by(got.column_names[0]).to_pylist(), prov(lf).get("streamed", {}).get("batches"))
    finally:                   # the budget must not leak into later tests on this worker (it did, under xdist)
        if before is None:
            os.environ.pop("MXFRAME_HOST_MEM_BUDGET", None)
        else:
            os.environ["MXFRAME_HOST_MEM_BUDGET"] = before
    return out


def test_an_integer_sum_has_the_type_it_has_in_memory_whatever_the_batch_count(tmp_path, monkeypatch):
    """64k rows with one 2^41 value: the sum is int64 in memory and in every batch (F-060: the type is the input's)."""
    n = 64_000
    x = np.zeros(n, dtype=np.int64)
    x[0] = 2 ** 41
    path = small_file(tmp_path, pa.table({"k": np.zeros(n, dtype=np.int32), "x": x}), 1000)
    got = types_by_budget(path, lambda s: s.groupby("k").agg(col("x").sum().alias("s")), ["20K", "200K", "64G"])
    assert got["64G"][2] is None                                   # in memory
    assert got["20K"][2] > got["200K"][2] > 1                      # two different batch counts
    assert {b: str(v[0].field("s").type) for b, v in got.items()} == {"20K": "int64", "200K": "int64", "64G": "int64"}
    assert all(v[1] == [{"k": 0, "s": 2 ** 41}] for v in got.values())


def assert_same_type_at_every_budget(path, build, budgets=("6K", "40K", "200K", "64G"), column="s"):
    got = types_by_budget(path, build, list(budgets))
    batches = [v[2] for v in got.values()]
    assert batches[-1] is None and len({b for b in batches[:-1]}) > 1, batches       # several batch counts, then in memory
    types = {b: str(v[0].field(column).type) for b, v in got.items()}
    assert len(set(types.values())) == 1, types
    assert len({repr(v[1]) for v in got.values()}) == 1                                # and the same sums
    return types["64G"]


def test_a_selective_filter_leaves_the_integer_sum_type_as_in_memory(tmp_path):
    """A big value sits in the rows the filter drops: int64 at every budget, whichever batches hold it."""
    n = 64_000
    y = np.random.default_rng(0).integers(0, 1000, n).astype(np.int64)
    y[::5000] = 2 ** 44
    path = small_file(tmp_path, pa.table({"k": np.zeros(n, dtype=np.int32), "y": y}), 256)
    kind = assert_same_type_at_every_budget(path, lambda s: s.filter(col("y") < 1000).groupby("k").agg(col("y").sum().alias("s")))
    assert kind == "int64"


def test_a_filter_on_another_column_leaves_the_integer_sum_type_as_in_memory(tmp_path):
    n = 64_000
    y = np.random.default_rng(0).integers(0, 1000, n).astype(np.int64)
    y[5] = 2 ** 41
    f = (np.arange(n) * 7919 % 100).astype(np.int32)
    path = small_file(tmp_path, pa.table({"k": np.zeros(n, dtype=np.int32), "y": y, "f": f}), 1000)
    assert assert_same_type_at_every_budget(
        path, lambda s: s.filter(col("f") == 3).groupby("k").agg(col("y").sum().alias("s"))) == "int64"


def test_a_product_of_columns_whose_large_values_sit_in_different_batches(tmp_path):
    """max|a| x max|b| is 2^60 although no row has both: a batch holding only the large a and another only the
    large b give the type the whole table gives (int64)."""
    n = 64_000
    a, b = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    a[0], b[n - 1] = 2 ** 30, 2 ** 30
    path = small_file(tmp_path, pa.table({"k": np.zeros(n, dtype=np.int32), "a": a, "b": b}), 250)
    assert assert_same_type_at_every_budget(
        path, lambda s: s.groupby("k").agg((col("a") * col("b")).sum().alias("s")), budgets=("20K", "200K", "64G")) == "int64"


def test_an_integer_sum_after_a_join_has_the_in_memory_type(tmp_path):
    n = 64_000
    x = np.zeros(n, dtype=np.int64)
    x[0] = 2 ** 41
    fact = small_file(tmp_path, pa.table({"k": np.zeros(n, dtype=np.int32), "j": (np.arange(n) % 10).astype(np.int32), "x": x}), 1000)
    dim = str(tmp_path / "dim.parquet")
    pq.write_table(pa.table({"id": np.arange(10, dtype=np.int32), "g": np.zeros(10, dtype=np.int32)}), dim)
    assert assert_same_type_at_every_budget(
        fact, lambda s: s.join(scan_parquet(dim), left_on="j", right_on="id").groupby("g").agg(col("x").sum().alias("s")),
        budgets=("20K", "200K", "64G")) == "int64"


def test_a_sum_near_the_old_2_53_limit_is_int64_in_memory_and_at_every_budget(tmp_path):
    """rows x max|value| just under 2^53 used to be a double (and one more row int64): the type depended on the values
    (F-060). Both are int64 now, at every budget and in memory, with the exact totals."""
    n = 64_000
    big = (2 ** 53 - 1) // n                       # n x big < 2^53
    y = np.full(n, big, dtype=np.int64)
    f = (np.arange(n) % 4).astype(np.int32)
    path = small_file(tmp_path, pa.table({"k": np.zeros(n, dtype=np.int32), "y": y, "f": f}), 500)
    kind = assert_same_type_at_every_budget(
        path, lambda s: s.filter(col("f") == 1).groupby("k").agg(col("y").sum().alias("s")), budgets=("20K", "100K", "64G"))
    assert kind == "int64"
    path2 = small_file(tmp_path, pa.table({"k": np.zeros(n + 1, dtype=np.int32), "y": np.full(n + 1, big + 1, dtype=np.int64),
                                           "f": (np.arange(n + 1) % 4).astype(np.int32)}), 500)
    assert assert_same_type_at_every_budget(
        path2, lambda s: s.filter(col("f") == 1).groupby("k").agg(col("y").sum().alias("s")),
        budgets=("20K", "100K", "64G")) == "int64"


def test_a_small_integer_sum_is_int64_in_memory_and_streamed(tmp_path):
    n = 64_000
    path = small_file(tmp_path, pa.table({"k": (np.arange(n) % 2).astype(np.int32), "x": np.arange(n, dtype=np.int64)}), 1000)
    got = types_by_budget(path, lambda s: s.groupby("k").agg(col("x").sum().alias("s")), ["20K", "64G"])
    assert {b: str(v[0].field("s").type) for b, v in got.items()} == {"20K": "int64", "64G": "int64"}
    assert got["20K"][1] == got["64G"][1] == [{"k": 0, "s": sum(range(0, n, 2))}, {"k": 1, "s": sum(range(1, n, 2))}]


def test_float32_min_max_and_the_other_aggregates_have_the_in_memory_types(tmp_path):
    n = 40_000
    rng = np.random.default_rng(1)
    path = small_file(tmp_path, pa.table({"k": (np.arange(n) % 3).astype(np.int8), "f": rng.random(n).astype(np.float32),
                                          "i": np.arange(n, dtype=np.int16), "u": np.arange(n, dtype=np.uint32)}), 1000)

    def build(s):
        return s.groupby("k").agg(col("f").sum().alias("fs"), col("f").mean().alias("fm"), col("f").min().alias("fmin"),
                                  col("f").max().alias("fmax"), col("i").sum().alias("isum"), col("i").max().alias("imax"),
                                  col("u").sum().alias("us"), col("u").count().alias("n"))
    got = types_by_budget(path, build, ["20K", "64G"])
    assert got["20K"][2] > 3
    assert got["20K"][0] == got["64G"][0], (got["20K"][0], got["64G"][0])
    assert str(got["64G"][0].field("fmin").type) == "double"


def test_batches_come_from_the_row_groups_the_filter_leaves(files, monkeypatch):
    """d is sorted, so d < 250 keeps the first six of 60 row groups: the rest are never read."""
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "10M")
    small = scan_parquet(files["fact"]).filter(col("d") < 250).groupby("k").agg(col("v").sum().alias("s"))
    got = run(small, "cpu")
    info = assert_streamed(small, 3)
    assert info["row_groups"] == 60 and info["row_groups_read"] == 6
    same(got, sql("SELECT k, sum(v) AS s FROM fact WHERE d < 250 GROUP BY k", files), ("k",))


def test_every_batch_empty_gives_the_empty_answers(files):
    """The filter cannot be answered from statistics, so every batch is read and filtered down to nothing."""
    nothing = (col("q") + col("k")) > 500
    keyed = scan_parquet(files["fact"]).filter(nothing).groupby("k").agg(col("v").sum().alias("s"))
    assert run(keyed, "cpu").num_rows == 0
    assert_streamed(keyed)
    glob = (scan_parquet(files["fact"]).filter(nothing)
            .groupby().agg(col("v").sum().alias("s"), col("q").sum().alias("sq"), col("v").count().alias("n"),
                           col("v").mean().alias("m"), col("q").min().alias("mn")))
    got = run(glob, "cpu")
    assert_streamed(glob)
    assert got.to_pylist() == [{"s": None, "sq": None, "n": 0, "m": None, "mn": None}]
    same(got, sql("SELECT sum(v) AS s, sum(q) AS sq, count(v) AS n, avg(v) AS m, min(q) AS mn FROM fact "
                  "WHERE q + k > 500", files))


def test_a_group_found_in_some_batches_only_is_kept(files):
    """d is sorted: each of its 2,500 values lives in one or two batches, the other batches never see it."""
    lf = scan_parquet(files["fact"]).groupby("d").agg(col("v").count().alias("n"), col("v").sum().alias("s"))
    got = run(lf, "cpu")
    assert_streamed(lf)
    same(got, sql("SELECT d, count(v) AS n, sum(v) AS s FROM fact GROUP BY d", files), ("d",))


@pytest.mark.parametrize("device", DEVICES)
def test_batches_emptied_by_the_row_filter_next_to_batches_that_are_not(files, device):
    """d is sorted and `d + k` cannot be answered from statistics: the early batches filter down to nothing
    (a partial of no groups, whose column types may differ), the late ones do not."""
    lf = (scan_parquet(files["fact"]).filter((col("d") + col("k")) > 2000).groupby("k")
          .agg(col("q").sum().alias("sq"), col("v").mean().alias("m"), col("q").min().alias("lo"),
               col("v").count().alias("n")))
    got = run(lf, device)
    assert_streamed(lf)
    same(got, sql("SELECT k, sum(q) AS sq, avg(v) AS m, min(q) AS lo, count(v) AS n FROM fact WHERE d + k > 2000 "
                  "GROUP BY k", files), ("k",))


@pytest.mark.parametrize("how", ["left", "semi", "anti"])
def test_the_streamed_table_is_the_left_side_of_left_semi_and_anti_joins(files, how):
    fact, dim1 = scan_parquet(files["fact"]), scan_parquet(files["dim1"]).filter(col("w") < 3)
    lf = (fact.join(dim1, left_on="j", right_on="id", how=how).groupby("k")
          .agg(col("v").sum().alias("s"), col("v").count().alias("n")))
    got = run(lf, "cpu")
    assert_streamed(lf)
    kind = {"left": "LEFT JOIN", "semi": "SEMI JOIN", "anti": "ANTI JOIN"}[how]
    want = sql(f"SELECT k, sum(v) AS s, count(v) AS n FROM fact {kind} (SELECT * FROM dim1 WHERE w < 3) d "
               f"ON fact.j = d.id GROUP BY k", files)
    same(got, want, ("k",))


def test_a_join_where_the_big_file_is_the_right_side_of_a_left_join_is_a_grace_join(files):
    """Step 2 declined this (a build side cannot stream); step 3b partitions both sides (tests/test_f056_step3b_grace.py)."""
    fact, dim1 = scan_parquet(files["fact"]), scan_parquet(files["dim1"])
    lf = dim1.join(fact, left_on="id", right_on="j", how="left").groupby("region").agg(col("v").sum().alias("s"))
    got = run(lf, "cpu")
    assert prov(lf)["path"] == "streamed" and prov(lf)["streamed"]["grace"]["how"] == "left"
    same(got, sql("SELECT region, sum(v) AS s FROM dim1 LEFT JOIN fact ON dim1.id = fact.j GROUP BY region", files),
         ("region",))


# --- exactness ----------------------------------------------------------------------------------------------

def small_file(tmp_path, table, rows_per_group=2000):
    path = str(tmp_path / "small.parquet")
    pq.write_table(table, path, row_group_size=rows_per_group)
    return path


def tiny(monkeypatch, host="100K"):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", host)


@pytest.mark.parametrize("device", DEVICES)
def test_integer_sums_stay_exact_past_2_53(tmp_path, monkeypatch, device):
    tiny(monkeypatch)
    n = 200_000
    rng = np.random.default_rng(1)
    t = pa.table({"k": (np.arange(n) % 3).astype(np.int32),
                  "x": (2 ** 45 + rng.integers(0, 1000, n)).astype(np.int64)})     # every group sums to ~2^62
    path = small_file(tmp_path, t)
    lf = scan_parquet(path).groupby("k").agg(col("x").sum().alias("s"), col("x").min().alias("lo"),
                                            col("x").max().alias("hi"), col("x").count().alias("n"))
    got = run(lf, device)
    assert_streamed(lf, 5)
    exact = {k: int(t.column("x").to_numpy()[np.arange(n) % 3 == k].sum()) for k in range(3)}
    assert {r["k"]: r["s"] for r in got.to_pylist()} == exact
    assert max(exact.values()) > 2 ** 53
    ref = duckdb.sql(f"SELECT k, min(x) AS lo, max(x) AS hi, count(x) AS n FROM read_parquet('{path}') GROUP BY k").arrow().read_all()
    assert {r["k"]: (r["lo"], r["hi"], r["n"]) for r in got.to_pylist()} == \
        {r["k"]: (r["lo"], r["hi"], r["n"]) for r in ref.to_pylist()}


def test_integer_sums_whose_batches_are_small_but_total_is_not(tmp_path, monkeypatch):
    """Each batch's sum is below 2^53 (the engine answers it as a double); the total is not: the partials are
    merged as int64, not rounded as doubles."""
    tiny(monkeypatch, "20K")
    n = 40_000
    t = pa.table({"k": np.zeros(n, dtype=np.int32), "x": np.full(n, 2 ** 38 + 1, dtype=np.int64)})
    lf = scan_parquet(small_file(tmp_path, t, 1000)).groupby("k").agg(col("x").sum().alias("s"))
    got = run(lf, "cpu")
    info = assert_streamed(lf, 8)
    assert got.column("s").to_pylist() == [n * (2 ** 38 + 1)]          # ~2^53.3: a double would round it
    assert n * (2 ** 38 + 1) > 2 ** 53 and info["batches"] > 8


@pytest.mark.parametrize("device", DEVICES)
def test_nulls_and_nans_follow_the_engine_rules(tmp_path, monkeypatch, device):
    tiny(monkeypatch)
    n = 60_000
    rng = np.random.default_rng(2)
    k = (np.arange(n) % 5).astype(np.int32)
    v = rng.random(n)
    v[(np.arange(n) % 97) == 0] = np.nan
    v[k == 4] = np.nan                                    # group 4 is all NaN
    x = pa.array(rng.integers(0, 100, n).astype(np.float64), mask=(k == 3) | (np.arange(n) % 11 == 0))   # group 3 is all NULL
    t = pa.table({"k": pa.array(k, mask=(np.arange(n) % 50 == 7)), "v": v, "x": x})   # some NULL keys
    path = small_file(tmp_path, t, 1500)
    lf = scan_parquet(path).groupby("k").agg(col("v").max().alias("vmax"), col("v").min().alias("vmin"),
                                            col("x").sum().alias("xs"), col("x").mean().alias("xm"),
                                            col("x").count().alias("xn"), col("v").count().alias("vn"))
    got = run(lf, device)
    assert_streamed(lf, 5)
    want = duckdb.sql(f"SELECT k, max(v) AS vmax, min(v) AS vmin, sum(x) AS xs, avg(x) AS xm, count(x) AS xn, "
                      f"count(v) AS vn FROM read_parquet('{path}') GROUP BY k").arrow().read_all()
    same(got, want, ("k",))


def test_reproducible_mode_gives_the_same_bits_every_run(files):
    lf, _, _ = q1(files)
    a = run(lf, "cpu", reproducible=True).sort_by([("k", "ascending"), ("s", "ascending")])
    b = run(q1(files)[0], "cpu", reproducible=True).sort_by([("k", "ascending"), ("s", "ascending")])
    assert prov(lf)["path"] == "streamed"
    assert a.equals(b)
    for name in ("sum_v", "sum_vq", "avg_v"):
        assert a.column(name).to_numpy().tobytes() == b.column(name).to_numpy().tobytes()


def test_reproducible_batches_do_not_depend_on_free_memory(monkeypatch):
    """With no override the budget is read from MemTotal in reproducible mode, not from what is free now."""
    from omxcore import streamed
    monkeypatch.delenv("MXFRAME_HOST_MEM_BUDGET")
    monkeypatch.setattr(streamed, "_meminfo", lambda key: {"MemAvailable": 1 << 30, "MemTotal": 64 << 30}[key])
    a = streamed.host_budget(reproducible=True)
    monkeypatch.setattr(streamed, "_meminfo", lambda key: {"MemAvailable": 40 << 30, "MemTotal": 64 << 30}[key])
    assert a == streamed.host_budget(reproducible=True) == 32 << 30
    assert streamed.host_budget() == int(0.7 * (40 << 30))


# --- what does not stream ---------------------------------------------------------------------------------

def test_a_plan_that_streams_through_a_grace_join_does_not_warn(tmp_path, monkeypatch):
    tiny(monkeypatch, "20K")
    t = pa.table({"k": np.arange(20_000, dtype=np.int32), "v": np.arange(20_000, dtype=np.float64)})
    path = small_file(tmp_path, t, 1000)
    big = scan_parquet(path)
    lf = big.select("k", "v").join(big.select("k"), on="k").sort("v", descending=True).limit(3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)                       # (a self-join is a Grace join since step 3b)
        got = lf.compute(device="cpu")
    assert prov(lf)["path"] == "streamed" and prov(lf)["streamed"]["grace"]["how"] == "inner"
    assert got.column("v").to_pylist() == [19999.0, 19998.0, 19997.0]


def test_count_distinct_declines_with_the_reason(tmp_path, monkeypatch):
    tiny(monkeypatch, "20K")
    t = pa.table({"k": (np.arange(20_000) % 3).astype(np.int32), "v": (np.arange(20_000) % 10).astype(np.int32)})
    path = small_file(tmp_path, t, 1000)
    lf = scan_parquet(path).distinct("k", "v").groupby("k").agg(col("v").count().alias("n"))
    with pytest.warns(RuntimeWarning, match="not streamed: a Distinct below the aggregate"):
        got = lf.compute(device="cpu")
    assert sorted((r["k"], r["n"]) for r in got.to_pylist()) == [(0, 10), (1, 10), (2, 10)]


def test_a_result_of_groups_bigger_than_the_budget_raises_a_clear_memory_error(tmp_path, monkeypatch):
    """The group state spills to disk (F-056 step 3a); a result that is itself all the groups cannot come back in
    less memory than it takes, and says so (with a LIMIT or HAVING above the groups it would finish)."""
    tiny(monkeypatch, "400K")
    n = 100_000
    t = pa.table({"k": np.arange(n, dtype=np.int64), "v": np.ones(n)})          # every row its own group
    lf = scan_parquet(small_file(tmp_path, t, 5000)).groupby("k").agg(col("v").sum().alias("s"))
    with pytest.raises(MemoryError, match=r"result of this query is bigger than memory: .*budget of 400\.0 KB"):
        lf.compute(device="cpu")


def test_a_dimension_bigger_than_half_the_budget_is_joined_by_partitions(tmp_path, monkeypatch):
    tiny(monkeypatch, "300K")
    n = 40_000
    big = pa.table({"id": np.arange(n, dtype=np.int32), "x": np.ones(n)})
    other = pa.table({"id": (np.arange(n) % n).astype(np.int32), "y": np.ones(n)})
    a, b = str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")
    pq.write_table(big, a, row_group_size=2000)
    pq.write_table(other, b, row_group_size=2000)
    lf = scan_parquet(a).join(scan_parquet(b), on="id").groupby().agg(col("y").sum().alias("s"))
    got = run(lf, "cpu")                                                      # step 3b: both sides are partitioned
    assert prov(lf)["path"] == "streamed" and prov(lf)["streamed"]["grace"]["pairs"] >= 2
    assert got.column("s").to_pylist() == [float(n)]


# --- memory is bounded ------------------------------------------------------------------------------------

PEAK = textwrap.dedent("""
    import json, sys
    import pyarrow as pa
    from omxcore import col
    from omxcore.io_parquet import scan_parquet

    def status(key):
        with open("/proc/self/status") as f:
            return next(int(l.split()[1]) * 1024 for l in f if l.startswith(key + ":"))

    lf = (scan_parquet(sys.argv[1]).filter(col("d") < 2400).groupby("k", "s")
          .agg(col("v").sum().alias("sv"), (col("v") * col("q")).sum().alias("svq"), col("q").mean().alias("aq")))
    lf.schema                                       # imports and warms up before the peak is reset
    lf.compute(device="cpu")
    if sys.platform == "linux":                      # macOS has no /proc and no way to restart the peak
        open("/proc/self/clear_refs", "w").write("5")   # VmHWM starts again from the current size
    base = status("VmRSS") if sys.platform == "linux" else 0
    out = lf.compute(device=sys.argv[2])
    import omxcore
    print(json.dumps({"module": omxcore.__file__, "rows": out.num_rows, "path": lf.last_compile_provenance["path"],
                      "arrow_peak": pa.default_memory_pool().max_memory(), "rss_peak_growth": (status("VmHWM") - base) if sys.platform == "linux" else None}))
""")


def conftest_root():
    import conftest
    return os.path.realpath(conftest._shim / "omxcore")


def peak(files, device, budget):
    import json
    import conftest                                  # its shim directory holds `omxcore` -> this checkout
    env = dict(os.environ, MXFRAME_HOST_MEM_BUDGET=budget, MXFRAME_GPU_MEM_BUDGET=GPU,
               PYTHONPATH=os.pathsep.join([str(conftest._shim)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    out = subprocess.run([sys.executable, "-c", PEAK, files["fact"], device], env=env, capture_output=True, text=True,
                         check=True, timeout=600)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_peak_memory_is_bounded_by_the_budget_not_the_file(files):
    """Warm both runs (the first compute reads and frees everything once, so the allocator's pages are not
    the difference), then compare the peak of a second compute."""
    decoded = N * (4 + 4 + 8 + 8 + 4 + 1)          # the file's columns as Arrow holds them (~170 MB)
    streamed = peak(files, "cpu", HOST)
    whole = peak(files, "cpu", "64G")
    here = os.path.join(conftest_root(), "__init__.py")        # the checkout under test, not an installed omxcore
    assert os.path.realpath(streamed["module"]) == os.path.realpath(whole["module"]) == here
    assert streamed["path"] == "streamed" and whole["path"] != "streamed"
    assert streamed["rows"] == whole["rows"] == 12
    assert streamed["arrow_peak"] < 40 << 20, streamed          # under the 40 MB budget
    assert whole["arrow_peak"] > 0.7 * decoded                  # the contrast: reading it whole holds the file
    if sys.platform == "linux":                                 # the resident peak of one call needs /proc and clear_refs (not on macOS)
        assert streamed["rss_peak_growth"] < 0.5 * whole["rss_peak_growth"], (streamed, whole)


# --- explain says what will happen ------------------------------------------------------------------------

def test_explain_says_a_query_streams_in_so_many_batches(files):
    lf, _, _ = q1(files)
    text = lf.explain(optimized=True)
    assert "Streaming" in text
    line = next(l for l in text.splitlines() if l.startswith("streams in "))
    assert re.fullmatch(r"streams in \d+ batches of row groups \(49 of 60 row groups read, host budget 40\.0 MB\)", line), line
    run(lf, "cpu")
    assert line == f"streams in {prov(lf)['streamed']['batches']} batches of row groups (49 of 60 row groups read, host budget 40.0 MB)"


def test_explain_says_a_self_join_is_a_grace_join(files):
    fact = scan_parquet(files["fact"])
    lf = fact.select("j", "v").join(fact.select("j"), on="j").limit(5)
    assert "grace join, 16 partitions" in lf.explain()


def test_explain_says_when_the_data_fits(files, monkeypatch):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "64G")
    lf, _, _ = q1(files)
    assert "in memory: fits the host budget of 64.0 GB" in lf.explain()


# --- declines where a wrong answer would be silent --------------------------------------------------------

@pytest.fixture(scope="module")
def silent(tmp_path_factory):
    d = tmp_path_factory.mktemp("f056silent")
    n = 60_000
    rng = np.random.default_rng(5)
    fact = pa.table({"a": pa.array(rng.random(n), mask=np.arange(n) % 3 == 0),         # the first column holds NULLs
                     "k": pa.array((np.arange(n) % 5).astype(np.int32), mask=np.arange(n) % 17 == 0),
                     "j": rng.integers(0, 300, n).astype(np.int32), "x": rng.integers(0, 100, n).astype(np.int64)})
    dim = pa.table({"id": np.arange(200, dtype=np.int32), "g": (np.arange(200) % 7).astype(np.int32)})
    paths = {"fact": str(d / "fact.parquet"), "dim": str(d / "dim.parquet")}
    pq.write_table(fact, paths["fact"], row_group_size=2000)
    pq.write_table(dim, paths["dim"])
    return paths


SILENT = {
    "semi (EXISTS) with the big file on the right": (
        "SELECT g, count(*) AS n FROM dim WHERE EXISTS (SELECT 1 FROM fact WHERE fact.j = dim.id) GROUP BY g", "semi"),
    "anti (NOT EXISTS) with the big file on the right": (
        "SELECT g, count(*) AS n FROM dim WHERE NOT EXISTS (SELECT 1 FROM fact WHERE fact.j = dim.id AND fact.x > 98) "
        "GROUP BY g", "anti"),
    "IN (subquery over the big file)": (
        "SELECT g, count(*) AS n FROM dim WHERE id IN (SELECT j FROM fact WHERE x > 97) GROUP BY g", "semi"),
    "left join with the big file on the right": (
        "SELECT g, count(x) AS n, count(*) AS c FROM dim LEFT JOIN fact ON fact.j = dim.id GROUP BY g", "left"),
    "self-join of the big file": (
        "SELECT f1.k, count(*) AS n FROM fact f1 JOIN (SELECT j, max(x) AS mx FROM fact GROUP BY j) f2 "
        "ON f1.j = f2.j AND f1.x = f2.mx GROUP BY f1.k", "inner"),
    "scalar subquery over the big file in WHERE": (
        "SELECT k, count(*) AS n FROM fact WHERE x > (SELECT avg(x) FROM fact) GROUP BY k", None),
}


def silent_reference(query, paths):
    con = oracle_connection()
    for n, p in paths.items():
        con.execute(f"CREATE VIEW {n} AS SELECT * FROM read_parquet('{p}')")
    return con.sql(query).arrow().read_all()


@pytest.mark.parametrize("name", sorted(SILENT))
def test_a_shape_that_cannot_stream_gives_the_in_memory_answer(silent, monkeypatch, name):
    import omxcore as mx
    query, why = SILENT[name]
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "100K")
    lf = mx.sql(query, fact=scan_parquet(silent["fact"]), dim=scan_parquet(silent["dim"]))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        got = lf.compute(device="cpu")
    said = [str(x.message) for x in w if "not streamed" in str(x.message)]
    if why:                                    # a join of two big sides is a Grace join (step 3b), never declined
        assert not said, said
        assert prov(lf)["path"] == "streamed" and prov(lf)["streamed"]["grace"]["how"] == why
    want = silent_reference(query, silent)
    key = lambda r: tuple((v is None, str(v)) for v in r.values())            # noqa: E731
    g, w_ = sorted(got.to_pylist(), key=key), sorted(want.to_pylist(), key=key)
    assert len(g) == len(w_)
    for a, b in zip(g, w_):
        assert all(close(float(a[c]) if a[c] is not None and not isinstance(a[c], str) else a[c], b[c]) for c in b), (a, b)


def test_count_star_counts_rows_when_the_first_column_and_the_keys_hold_nulls(silent, monkeypatch):
    import omxcore as mx
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "100K")
    query = "SELECT k, count(*) AS n, count(a) AS na, avg(a) AS m FROM fact GROUP BY k"
    lf = mx.sql(query, fact=scan_parquet(silent["fact"]))
    got = lf.compute(device="cpu")
    assert prov(lf)["path"] == "streamed"
    want = silent_reference(query, silent)
    key = lambda r: (r["k"] is None, r["k"])                                      # noqa: E731
    g, w_ = sorted(got.to_pylist(), key=key), sorted(want.to_pylist(), key=key)
    assert [r["n"] for r in g] == [r["n"] for r in w_] and None in [r["k"] for r in g]
    assert all(close(a["m"], b["m"]) and a["na"] == b["na"] for a, b in zip(g, w_))
