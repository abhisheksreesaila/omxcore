"""F-056 step 3b: joins where both sides are bigger than memory (Grace hash join over spill partitions).

`MXFRAME_HOST_MEM_BUDGET` makes ~7 MB (orders) and ~15 MB (lineitem) Parquet files "bigger than the machine".
Both sides are hash-partitioned by the join key to disk and the pairs are joined one after another, with the plan
above the join (aggregate, filters, top-k) fed per pair. Every result must equal DuckDB's over the same files, on
cpu, gpu and auto; the path (`provenance["streamed"]["grace"]`), the bounded peak memory and the cleanup of the
spill directory are asserted. On `main` these queries are declined with a RuntimeWarning and run in memory.
"""
import json
import os
import subprocess
import sys
import textwrap
import warnings

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import omxcore as mx
from omxcore import col, gpu_stream
from omxcore.io_parquet import scan_parquet

from test_f056_streamed import same
from test_f056_step3a import spill_leftovers

HAS_GPU = gpu_stream.cuda_available()
DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
           pytest.param("auto", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU"))]

ORD = 200_000
LI = 400_000
HOST = "6M"


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("f056grace")
    rng = np.random.default_rng(7)
    idx = np.arange(ORD)
    o = pa.table({
        "o_id": idx.astype(np.int64),
        "o_cust": rng.integers(0, 5000, ORD).astype(np.int32),
        "o_prio": np.array(["1-URGENT", "2-HIGH", "3-MEDIUM", "4-LOW", "5-NONE"])[rng.integers(0, 5, ORD)],
        "o_tot": rng.random(ORD) * 1000,
    })
    li = np.arange(LI)
    l = pa.table({
        "l_id": li.astype(np.int64),
        "l_ok": rng.integers(0, int(ORD * 0.9), LI).astype(np.int64),           # a tenth of the orders has no lines
        "l_q": rng.integers(1, 51, LI).astype(np.int64),
        "l_p": rng.random(LI) * 100,
        "l_flag": np.array(["A", "N", "R"])[rng.integers(0, 3, LI)],
    })
    out = {"o": str(d / "o.parquet"), "l": str(d / "l.parquet")}
    pq.write_table(o, out["o"], row_group_size=20_000)
    pq.write_table(l, out["l"], row_group_size=20_000)
    return out


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", HOST)
    spill = tmp_path / "tmp"
    spill.mkdir()
    monkeypatch.setenv("MXFRAME_SPILL_DIR", str(spill))
    return spill


def duck(query, files):
    con = duckdb.connect()
    for n, p in files.items():
        con.execute(f"CREATE VIEW {n} AS SELECT * FROM read_parquet('{p}')")
    return con.sql(query).arrow().read_all()


def run_sql(query, files, device="cpu", **kw):
    lf = mx.sql(query, **{n: scan_parquet(p) for n, p in files.items()})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)           # a declined join warns: that is the failure on main
        warnings.filterwarnings("ignore", message=".*leaves the GPU.*|.*GPU.*")
        got = lf.compute(device=device, **kw)
    return lf, got


def grace(lf):
    p = lf.last_compile_provenance
    assert p["path"] == "streamed", p
    return p["streamed"]["grace"]


def check(query, files, device, keys=(), min_pairs=2):
    lf, got = run_sql(query, files, device)
    g = grace(lf)
    assert g["partitions"] == 16 and g["pairs"] >= min_pairs, g
    same(got, duck(query, files), keys)
    return lf, g


# --- 1. TPC-H shaped: lineitem x orders, both over budget, an aggregate above --------------------------------

Q3 = ("SELECT o_prio, l_flag, count(*) AS n, sum(l_q) AS sq, sum(l_p * (1 - o_cust / 10000.0)) AS rev, "
      "avg(l_p) AS ap, min(o_tot) AS lo, max(l_q) AS hi "
      "FROM l JOIN o ON l_ok = o_id WHERE l_q > 5 AND o_tot < 900 GROUP BY o_prio, l_flag")


@pytest.mark.parametrize("device", DEVICES)
def test_lineitem_join_orders_both_over_budget_with_an_aggregate_equals_duckdb(files, device, _env):
    lf, g = check(Q3, files, device, ("o_prio", "l_flag"))
    assert g["pairs"] >= 16
    devices = lf.last_compile_provenance["streamed"]["devices"]
    assert sum(devices.values()) == g["pairs"]
    if device == "gpu":
        assert devices == {"gpu": g["pairs"]}
    assert spill_leftovers(_env) == []


@pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")
def test_a_pair_that_does_not_fit_the_gpu_budget_runs_on_the_cpu(files, monkeypatch):
    monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", "100K")
    lf, g = check(Q3, files, "gpu", ("o_prio", "l_flag"))
    assert lf.last_compile_provenance["streamed"]["devices"] == {"cpu": g["pairs"]}


def test_three_big_tables_join_through_nested_grace_joins(tmp_path, monkeypatch, _env):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "1500K")
    rng = np.random.default_rng(21)
    n = 60_000
    f = {}
    for name, ks in (("a", ("x",)), ("b", ("x", "y")), ("c", ("y",))):
        cols = {k: rng.integers(0, 40_000, n).astype(np.int64) for k in ks}
        cols[f"v{name}"] = rng.integers(0, 10, n).astype(np.int64)
        f[name] = str(tmp_path / f"{name}.parquet")
        pq.write_table(pa.table(cols), f[name], row_group_size=5_000)
    q = "SELECT count(*) AS n, sum(va * vb * vc) AS s FROM a JOIN b ON a.x = b.x JOIN c ON b.y = c.y"
    lf, got = run_sql(q, f)
    g = grace(lf)
    assert g["pairs"] > 16 and g["partitions_total"] >= 64, g
    same(got, duck(q, f))
    assert spill_leftovers(_env) == []


def test_explain_says_grace_join_and_the_partition_count(files):
    lf = mx.sql(Q3, **{n: scan_parquet(p) for n, p in files.items()})
    text = lf.explain()
    assert "grace join, 16 partitions" in text, text


def test_a_join_that_fits_the_budget_is_not_a_grace_join(files, monkeypatch):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "64G")
    lf, _ = run_sql(Q3, files)
    assert lf.last_compile_provenance["path"] != "streamed"


def test_top_k_above_the_join_runs_on_the_pair_results(files):
    q = ("SELECT l_id, o_prio, l_p * o_tot AS w FROM l JOIN o ON l_ok = o_id WHERE l_q = 50 "
         "ORDER BY w DESC, l_id LIMIT 7 OFFSET 2")
    lf, got = run_sql(q, files)
    grace(lf)
    same(got, duck(q, files))


def test_a_selective_row_query_over_the_join_is_collected(files):
    q = "SELECT l_id, o_id, o_prio FROM l JOIN o ON l_ok = o_id WHERE l_q = 50 AND o_tot > 995 AND l_p > 90"
    lf, got = run_sql(q, files)
    grace(lf)
    assert got.num_rows > 0
    same(got, duck(q, files), ("l_id",))


# --- 2. the big file on the right of left / semi / anti / IN / NOT IN ------------------------------------------

LEFT = ("SELECT o_prio, count(*) AS rows, count(l_id) AS n, sum(l_q) AS sq, max(l_p) AS hi "
        "FROM o LEFT JOIN l ON o_id = l_ok GROUP BY o_prio")
SEMI = ("SELECT o_prio, count(*) AS n, sum(o_tot) AS t FROM o WHERE EXISTS "
        "(SELECT 1 FROM l WHERE l_ok = o_id AND l_q > 40) GROUP BY o_prio")
ANTI = ("SELECT o_prio, count(*) AS n, sum(o_tot) AS t FROM o WHERE NOT EXISTS "
        "(SELECT 1 FROM l WHERE l_ok = o_id AND l_q > 40) GROUP BY o_prio")
IN = "SELECT o_prio, count(*) AS n, sum(o_tot) AS t FROM o WHERE o_id IN (SELECT l_ok FROM l WHERE l_q > 40) GROUP BY o_prio"
NOT_IN = ("SELECT o_prio, count(*) AS n, sum(o_tot) AS t FROM o WHERE o_id NOT IN (SELECT l_ok FROM l WHERE l_q > 40) "
          "GROUP BY o_prio")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("query", [LEFT, SEMI, ANTI, IN, NOT_IN], ids=["left", "semi", "anti", "in", "not_in"])
def test_the_big_file_on_the_right_of_a_left_semi_anti_in_not_in_join_equals_duckdb(files, device, query, _env):
    check(query, files, device, ("o_prio",))
    assert spill_leftovers(_env) == []


def test_left_join_rows_keep_their_unmatched_orders_with_null_columns(files):
    q = ("SELECT o_id, l_id FROM o LEFT JOIN l ON o_id = l_ok WHERE o_id < 3000 AND (l_q IS NULL OR l_q > 49) "
         "ORDER BY o_id, l_id")
    lf, got = run_sql(q, files)
    grace(lf)
    same(got, duck(q, files))
    assert any(r["l_id"] is None for r in got.to_pylist())


@pytest.mark.parametrize("query", [Q3, LEFT, SEMI, ANTI], ids=["inner", "left", "semi", "anti"])
def test_result_types_are_those_of_the_in_memory_run(files, query, monkeypatch):
    """The partials of the pairs are merged as int64, and the result type is decided once at the end, as in memory."""
    lf, got = run_sql(query, files)
    grace(lf)
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "64G")
    whole = mx.sql(query, **{n: scan_parquet(p) for n, p in files.items()}).compute(device="cpu")
    assert got.schema == whole.schema


# --- 3. a self-join of the big file -----------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_a_self_join_of_the_big_file_equals_duckdb(files, device, _env):
    a = scan_parquet(files["l"]).filter(col("l_q") < 20)
    b = scan_parquet(files["l"]).filter(col("l_q") > 30).select(col("l_ok").alias("b_ok"), col("l_q").alias("b_q"))
    lf = (a.join(b, left_on="l_ok", right_on="b_ok").groupby("l_flag")
          .agg(col("l_id").count().alias("n"), (col("l_p") * col("b_q")).sum().alias("s")))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*GPU.*")
        got = lf.compute(device=device)
    g = grace(lf)
    assert g["pairs"] >= 16 and g["how"] == "inner"
    same(got, duck("SELECT a.l_flag AS l_flag, count(a.l_id) AS n, sum(a.l_p * b.l_q) AS s FROM l a JOIN l b ON a.l_ok = b.l_ok "
                   "WHERE a.l_q < 20 AND b.l_q > 30 GROUP BY a.l_flag", files), ("l_flag",))
    assert spill_leftovers(_env) == []


# --- 4. skew: one partition re-splits ---------------------------------------------------------------------------

def skewed_pair(tmp_path):
    """80% of the left rows (and the right rows) have keys that all hash to partition 0 of the first level."""
    from omxcore import spill
    cand = pa.table({"k": np.arange(2_000_000, dtype=np.int64)})
    part = ((spill.hash_keys(cand, ["k"], 0) >> np.uint64(33)) % np.uint64(spill.PARTITIONS)).astype(np.int64)
    hot = np.flatnonzero(part == 0)[:3000]
    cold = np.flatnonzero(part != 0)[:6000]
    rng = np.random.default_rng(11)
    n = 150_000
    pick = lambda m: np.where(rng.random(m) < 0.8, rng.choice(hot, m), rng.choice(cold, m)).astype(np.int64)    # noqa: E731
    left = pa.table({"k": pick(n), "v": rng.random(n)})
    m = 40_000
    right = pa.table({"k": pick(m), "w": rng.integers(0, 100, m).astype(np.int64)})
    out = {"a": str(tmp_path / "a.parquet"), "b": str(tmp_path / "b.parquet")}
    pq.write_table(left, out["a"], row_group_size=10_000)
    pq.write_table(right, out["b"], row_group_size=10_000)
    return out


def test_a_partition_that_is_still_too_big_is_split_again(tmp_path, monkeypatch, _env):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "1500K")
    f = skewed_pair(tmp_path)
    q = "SELECT count(*) AS n, sum(v * w) AS s, max(w) AS hi FROM a JOIN b ON a.k = b.k"
    lf, g = check(q, f, "cpu")
    assert g["depth"] >= 2, g
    assert g["partitions_total"] > 2 * 16, g
    assert spill_leftovers(_env) == []


def test_keys_that_never_split_raise_a_memory_error_and_clean_up(tmp_path, monkeypatch, _env):
    from omxcore import spill
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "2M")
    monkeypatch.setattr(spill, "MAX_DEPTH", 2)
    n = 100_000
    t = pa.table({"k": np.full(n, 7, dtype=np.int64), "v": np.arange(n, dtype=np.float64)})
    for name in ("a", "b"):
        pq.write_table(t, str(tmp_path / f"{name}.parquet"), row_group_size=10_000)
    q = "SELECT count(*) AS n FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.k = a.k)"
    f = {"a": str(tmp_path / "a.parquet"), "b": str(tmp_path / "b.parquet")}
    with pytest.raises(MemoryError, match=r"grace join.*cannot be split.*key"):
        run_sql(q, f)
    assert spill_leftovers(_env) == []


HOT = {
    "inner, build right, left in chunks": "SELECT count(*) AS n, sum(w) AS s FROM a JOIN b ON a.k = b.k",
    "inner, build left, right in chunks": "SELECT count(*) AS n, sum(w) AS s FROM b JOIN a ON a.k = b.k",
    "left join, right built, left in chunks": "SELECT count(*) AS n, count(w) AS nw, sum(w) AS s FROM a LEFT JOIN b ON a.k = b.k",
    "semi": "SELECT count(*) AS n, sum(v) AS s FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.k = a.k AND b.w > 100)",
    "anti": "SELECT count(*) AS n, sum(v) AS s FROM a WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.k = a.k AND b.w > 100)",
}


@pytest.mark.parametrize("name", sorted(HOT))
def test_a_hot_key_on_one_side_only_streams_the_big_side_against_the_small_partition(tmp_path, monkeypatch, _env, name):
    """90% of `a` has key 7: its partition is far over the budget, while the same partition of `b` is one small row
    group's worth. No split is needed (and none would help): the small side is built, the big one is read in chunks."""
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "2M")
    n = 200_000
    a = pa.table({"k": np.where(np.arange(n) % 10 < 9, 7, np.arange(n)).astype(np.int64), "v": np.ones(n)})
    b = pa.table({"k": np.arange(n // 4, dtype=np.int64), "w": np.arange(n // 4, dtype=np.int64)})
    pq.write_table(a, str(tmp_path / "a.parquet"), row_group_size=10_000)
    pq.write_table(b, str(tmp_path / "b.parquet"), row_group_size=10_000)
    f = {"a": str(tmp_path / "a.parquet"), "b": str(tmp_path / "b.parquet")}
    lf, g = check(HOT[name], f, "cpu")
    assert g["chunked"] >= 1 and g["split"] == 0, g
    assert spill_leftovers(_env) == []


def test_a_limit_above_a_grace_join_stops_early_and_cleans_up(files, _env):
    q = "SELECT l_id, o_id, o_tot, o_prio FROM l JOIN o ON l_ok = o_id LIMIT 5"
    lf, got = run_sql(q, files)
    g = grace(lf)
    assert got.num_rows == 5 and g["pairs"] < 16, g
    every = {(r["l_id"], r["o_id"]) for r in duck("SELECT l_id, o_id, o_tot, o_prio FROM l JOIN o ON l_ok = o_id", files).to_pylist()}
    assert {(r["l_id"], r["o_id"]) for r in got.to_pylist()} <= every
    assert spill_leftovers(_env) == []


# --- 5. keys that need SQL rules to co-locate --------------------------------------------------------------------

def key_files(tmp_path, left_ints=False):
    rng = np.random.default_rng(5)
    pool = np.array([np.nan, -0.0, 0.0, 1.5, 2.0, 3.0, 1e300, -7.25] + list(np.arange(10, 1500) * 0.5))

    def floats(n):
        v = pool[rng.integers(0, len(pool), n)]
        v[rng.random(n) < 0.05] = np.nan
        return pa.array(v, mask=rng.random(n) < 0.05)

    n, m = 40_000, 30_000
    a = pa.table({"k": pa.array(rng.integers(0, 60, n).astype(np.int64)) if left_ints else floats(n),
                  "v": rng.integers(0, 100, n).astype(np.int64)})
    b = pa.table({"k": floats(m), "w": rng.integers(0, 100, m).astype(np.int64)})
    out = {"a": str(tmp_path / "a.parquet"), "b": str(tmp_path / "b.parquet")}
    pq.write_table(a, out["a"], row_group_size=4_000)
    pq.write_table(b, out["b"], row_group_size=4_000)
    return out


@pytest.mark.parametrize("how", ["JOIN", "LEFT JOIN"])
def test_null_nan_and_signed_zero_keys_match_by_sql_rules(tmp_path, monkeypatch, how):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "500K")
    f = key_files(tmp_path)
    q = f"SELECT count(*) AS n, count(w) AS nw, sum(v) AS sv, sum(w) AS sw FROM a {how} b ON a.k = b.k"
    check(q, f, "cpu")
    assert duck(q, f).column("nw")[0].as_py() > 100_000              # the NaN / zero / NULL pool does meet


def test_integer_keys_meet_float_keys_by_value(tmp_path, monkeypatch):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "500K")
    f = key_files(tmp_path, left_ints=True)
    q = "SELECT count(*) AS n, sum(v) AS sv, sum(w) AS sw FROM a JOIN b ON a.k = b.k"
    check(q, f, "cpu")
    assert duck(q, f).column("n")[0].as_py() > 1_000


def test_string_keys_with_different_dictionaries_per_row_group_meet(tmp_path, monkeypatch):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "1M")
    rng = np.random.default_rng(9)
    names = np.array([f"cust-{i:05d}" for i in range(30_000)])
    n, m = 120_000, 60_000
    a = pa.table({"k": names[rng.integers(0, 30_000, n)], "v": rng.integers(0, 10, n).astype(np.int64)})
    b = pa.table({"k": names[rng.permutation(30_000)[:m // 2].repeat(2)], "w": rng.integers(0, 10, m).astype(np.int64)})
    f = {"a": str(tmp_path / "a.parquet"), "b": str(tmp_path / "b.parquet")}
    pq.write_table(a, f["a"], row_group_size=6_000)
    pq.write_table(b, f["b"], row_group_size=7_000)
    check("SELECT count(*) AS n, sum(v * w) AS s FROM a JOIN b ON a.k = b.k", f, "cpu")


def test_not_in_with_a_null_in_the_list_returns_nothing(tmp_path, monkeypatch, _env):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "1M")
    n = 100_000
    keys = np.arange(n, dtype=np.int64)
    a = pa.table({"k": pa.array(keys, mask=(keys % 1000) == 3), "v": np.ones(n)})         # NULL keys on the left too
    nulls = np.zeros(n, dtype=bool)
    nulls[77_777] = True                                  # one NULL, far from the first row group
    b = pa.table({"k": pa.array(keys, mask=nulls)})
    f = {"a": str(tmp_path / "a.parquet"), "b": str(tmp_path / "b.parquet")}
    pq.write_table(a, f["a"], row_group_size=5_000)
    pq.write_table(b, f["b"], row_group_size=5_000)
    b_cut = "SELECT k FROM b WHERE k IS NOT NULL AND k < 60000"
    for q, joined in (("SELECT count(*) AS n FROM a WHERE k NOT IN (SELECT k FROM b)", False),     # a NULL in the list: nothing
                      (f"SELECT count(*) AS n, sum(v) AS s FROM a WHERE k NOT IN ({b_cut})", True),
                      ("SELECT count(*) AS n FROM a WHERE k NOT IN (SELECT k FROM b WHERE k < 0)", False)):     # empty list: all (row groups pruned: nothing to partition)
        lf, got = run_sql(q, f)
        if joined:
            grace(lf)
        want = duck(q, f)
        same(got, want)
        assert got.column("n")[0].as_py() == want.column("n")[0].as_py()
    assert spill_leftovers(_env) == []


# --- 6. reproducible mode, memory, cleanup -----------------------------------------------------------------------

def test_a_reproducible_grace_join_is_the_same_bits_twice(files):
    q = "SELECT o_prio, sum(l_p * o_tot) AS s, avg(l_p) AS a FROM l JOIN o ON l_ok = o_id GROUP BY o_prio"
    outs = []
    for _ in range(2):
        lf = mx.sql(q, **{n: scan_parquet(p) for n, p in files.items()})
        outs.append(lf.compute(device="cpu", reproducible=True).sort_by("o_prio"))
        grace(lf)
    assert outs[0].equals(outs[1])
    assert outs[0].to_pylist() == outs[1].to_pylist()


def test_the_spill_directory_is_removed_after_an_error_in_a_pair(files, _env, monkeypatch):
    from omxcore import streamed
    real = streamed._spill.Partitioned.read
    seen = {}

    def broken(self, i):
        seen.setdefault("dirs", spill_leftovers(_env))
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(streamed._spill.Partitioned, "read", broken)
    with pytest.raises(RuntimeError, match="disk on fire"):
        run_sql(Q3, files)
    assert len(seen["dirs"]) == 1
    assert spill_leftovers(_env) == []
    monkeypatch.setattr(streamed._spill.Partitioned, "read", real)


PEAK = textwrap.dedent("""
    import json, sys
    import pyarrow as pa
    import omxcore as mx
    from omxcore.io_parquet import scan_parquet

    def status(key):
        with open("/proc/self/status") as f:
            return next(int(l.split()[1]) * 1024 for l in f if l.startswith(key + ":"))

    o, l = sys.argv[1], sys.argv[2]
    q = ("SELECT o_prio, l_flag, count(*) AS n, sum(l_p * o_tot) AS s FROM l JOIN o ON l_ok = o_id "
         "WHERE l_q > 5 GROUP BY o_prio, l_flag")
    lf = mx.sql(q, o=scan_parquet(o), l=scan_parquet(l))
    lf.compute(device="cpu")                         # warm: the first run decodes and frees everything once
    if sys.platform == "linux":                      # macOS has no /proc and no way to restart the peak
        open("/proc/self/clear_refs", "w").write("5")
    base = status("VmRSS") if sys.platform == "linux" else 0
    pa.default_memory_pool().release_unused()
    out = lf.compute(device="cpu")
    print(json.dumps({"rows": out.num_rows, "path": lf.last_compile_provenance["path"],
                      "arrow_peak": pa.default_memory_pool().max_memory(), "rss_peak_growth": (status("VmHWM") - base) if sys.platform == "linux" else None}))
""")


def peak(files, budget, tmp):
    import conftest
    env = dict(os.environ, MXFRAME_HOST_MEM_BUDGET=budget, MXFRAME_SPILL_DIR=str(tmp),
               PYTHONPATH=os.pathsep.join([str(conftest._shim)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    out = subprocess.run([sys.executable, "-c", PEAK, files["o"], files["l"]], env=env, capture_output=True, text=True,
                         check=True, timeout=600)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_peak_memory_of_a_grace_join_is_bounded_by_the_budget(files, _env):
    """orders (7 MB) + lineitem (15 MB) decode to more than twice the 6 MB budget; the Arrow pool never holds more than
    the budget while the pairs run, and the same query with a big budget holds both tables."""
    bounded = peak(files, HOST, _env)
    whole = peak(files, "64G", _env)
    assert bounded["path"] == "streamed" and whole["path"] != "streamed"
    assert bounded["rows"] == whole["rows"] == 15
    assert bounded["arrow_peak"] < 6 << 20, bounded
    assert whole["arrow_peak"] > 12 << 20, whole
    assert spill_leftovers(_env) == []


# --- 7. NULL keys, big integers against floats, the hot key in the error -------------------------------------------

NULL_JOINS = {
    "inner": "SELECT count(*) AS n, sum(v) AS sv, sum(w) AS sw FROM a JOIN b ON a.k = b.k",
    "left": "SELECT count(*) AS n, count(w) AS nw, sum(v) AS sv FROM a LEFT JOIN b ON a.k = b.k",
    "semi": "SELECT count(*) AS n, sum(v) AS sv FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.k = a.k)",
    "anti": "SELECT count(*) AS n, sum(v) AS sv FROM a WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.k = a.k)",
}


@pytest.mark.parametrize("how", sorted(NULL_JOINS))
def test_files_that_are_mostly_null_keys_join_without_a_hot_partition(tmp_path, monkeypatch, _env, how):
    """80% NULL keys on both sides all hash to one partition at every salt. They never match, so they are never
    partitioned: dropped, or (left, anti) emitted without a partner."""
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "1M")
    rng = np.random.default_rng(31)
    n = 150_000
    f = {}
    for name, extra in (("a", "v"), ("b", "w")):
        t = pa.table({"k": pa.array(rng.integers(0, 30_000, n).astype(np.int64), mask=rng.random(n) < 0.8),
                      extra: rng.integers(0, 10, n).astype(np.int64)})
        f[name] = str(tmp_path / f"{name}.parquet")
        pq.write_table(t, f[name], row_group_size=10_000)
    lf, g = check(NULL_JOINS[how], f, "cpu")
    assert g["null_key_rows"] > 100_000, g
    assert spill_leftovers(_env) == []


def test_an_integer_with_no_exact_double_against_a_float_key_is_a_type_error_like_in_memory(tmp_path, monkeypatch, _env):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "300K")
    n = 40_000
    a = pa.table({"k": np.arange(n, dtype=np.int64) + 2 ** 53 + 1, "v": np.ones(n)})
    b = pa.table({"k": np.arange(n, dtype=np.float64), "w": np.ones(n)})
    f = {}
    for name, t in (("a", a), ("b", b)):
        f[name] = str(tmp_path / f"{name}.parquet")
        pq.write_table(t, f[name], row_group_size=4_000)
    with pytest.raises(TypeError, match="join key types differ"):
        run_sql("SELECT count(*) AS n FROM a JOIN b ON a.k = b.k", f)
    assert spill_leftovers(_env) == []


def test_the_memory_error_names_the_hot_key_and_its_row_count(tmp_path, monkeypatch, _env):
    from omxcore import spill
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "2M")
    monkeypatch.setattr(spill, "MAX_DEPTH", 2)
    n = 100_000
    t = pa.table({"k": np.full(n, 7, dtype=np.int64), "v": np.arange(n, dtype=np.float64)})
    f = {}
    for name in ("a", "b"):
        f[name] = str(tmp_path / f"{name}.parquet")
        pq.write_table(t, f[name], row_group_size=10_000)
    with pytest.raises(MemoryError, match=r"most repeated key is k=7, 100,000 (left|right) rows"):
        run_sql("SELECT count(*) AS n FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.k = a.k)", f)
