"""F-046a slice 2: star lookups on the new map (`mx_filter` plan version 2).

A join partner whose key repeats no longer sends the star scan to the leaf-by-leaf join: the map keeps
its rows in CSR layout (build order) and the scan writes one joined row per matching partner row, the
rows of a fact row in partner build order. Hashed maps get a presence bitmap when the key range is small
enough, so a miss stops early. A partner much larger than the fact's filtered rows is first reduced to
the rows whose keys the fact's surviving rows hold (the runtime join filter, F-043, generalized).

Every result is compared with DuckDB; every test asserts the path: one star call, no leaf-by-leaf join,
no `-18` (E_DUP) decline, the map's kind (direct / hash, repeated keys, bitmap) and the partner rows the
star call was given."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import omxcore.custom_ops as co
import omxcore.join_engine as je
import omxcore.native_exec as nx
from omxcore import LazyFrame, col, lit
from omxcore.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")


@pytest.fixture
def seen(monkeypatch):
    """What the star path did: `star` = native calls that carry a join, `leaf` = leaf-by-leaf join steps,
    `build` = the partner rows (non-semi lookups) of the last star call, `maps` = its maps' description."""
    log = {"star": 0, "leaf": 0, "build": None, "maps": None}
    real_filter, real_idx = nx.try_filter, je._join_indices

    def try_filter(comp, table, preds, needed, lookups=(), keep_dict=()):
        out = real_filter(comp, table, preds, needed, lookups, keep_dict)
        if any(not lk.semi for lk in lookups):
            log["star"] += out is not None
            log["build"] = [lk.build.num_rows for lk in lookups if not lk.semi]
            log["maps"] = nx.last_filter_maps()
        return out

    def join_indices(*a, **k):
        log["leaf"] += 1
        return real_idx(*a, **k)

    monkeypatch.setattr(nx, "try_filter", try_filter)
    monkeypatch.setattr(je, "_join_indices", join_indices)
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    monkeypatch.setenv("MXFRAME_FUSED_JOIN_AGG", "0")      # these tests are about the star scan
    nx.reset_decline_counts()
    return log


def _fact(n=60_000, seed=1, kmax=800, scale=1, chunks=1, nulls=False):
    r = np.random.default_rng(seed)
    parts = []
    for i in range(chunks):
        m = n // chunks
        k = A(r.integers(-5, kmax, m).astype(np.int64) * scale)
        parts.append(pa.table({
            "id": A(np.arange(i * m, (i + 1) * m, dtype=np.int64)),
            "k": pa.array(k.to_pylist(), type=pa.int64()) if not nulls else
                 pa.array([None if x % 97 == 0 else x for x in k.to_pylist()], type=pa.int64()),
            "k2": A(r.integers(0, 40, m).astype(np.int32)),
            "q": A(r.integers(0, 100, m).astype(np.int32)),
            "v": A(np.round(r.random(m) * 50, 2)),
        }))
    return pa.concat_tables(parts)


def _dim(keys, tag="a", seed=2, chunks=1):
    """A partner with one row per entry of `keys` (repeats allowed): `dv` numbers the rows."""
    keys = np.asarray(keys, np.int64)
    r = np.random.default_rng(seed)
    n = len(keys)
    cuts = np.linspace(0, n, chunks + 1).astype(int)
    parts = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        parts.append(pa.table({
            "dk": A(keys[a:b]),
            "dv": A(np.arange(a, b, dtype=np.int32)),
            "name": pa.array([f"{tag}{x}" for x in range(a, b)], type=pa.string()),
            "w": A(np.round(r.random(b - a), 3)),
        }))
    return pa.concat_tables(parts)


def _repeating(n_keys, seed=3, max_rep=3, scale=1):
    """Keys 0..n_keys-1 repeated 0..max_rep times each (some absent), rows shuffled."""
    r = np.random.default_rng(seed)
    keys = np.repeat(np.arange(n_keys, dtype=np.int64), r.integers(0, max_rep + 1, n_keys)) * scale
    return keys[r.permutation(len(keys))]


def _duck(sql, **t):
    con = duckdb.connect()
    for k, v in t.items():
        con.register(k, v)
    return [tuple(r) for r in con.execute(sql).fetchall()]


def _rows(t, cols):
    d = t.select(cols).to_pydict()
    return [tuple(d[c][i] for c in cols) for i in range(t.num_rows)]


def _same(got, exp):
    key = lambda r: tuple((x is None, str(type(x)), x) for x in r)
    got = sorted(got, key=key)
    exp = sorted(exp, key=key)
    assert len(got) == len(exp), (len(got), len(exp))
    for g, e in zip(got, exp):
        for x, y in zip(g, e):
            if isinstance(x, float) or isinstance(y, float):
                assert x == pytest.approx(y, rel=1e-9), (g, e)
            else:
                assert x == y, (g, e)


def _star_only(seen):
    """One star call answered the join: no leaf-by-leaf step, no duplicate-key decline."""
    assert seen["star"] == 1, (seen, nx.decline_counts())
    assert seen["leaf"] == 0, seen
    assert not [k for k in nx.decline_counts() if "-18" in k], nx.decline_counts()


# ── repeated partner keys ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("scale", [1, 1_000_003])        # compact range: direct CSR; wide: hashed CSR
def test_partner_with_repeated_keys_expands_one_to_many(seen, scale):
    fact = _fact(scale=scale)
    dim = _dim(_repeating(900, scale=scale))
    got = (LazyFrame(fact).filter(col("q") < lit(70))
           .join(LazyFrame(dim), left_on="k", right_on="dk")
           .select("id", "k", "v", "dv", "name", "w").compute(device="cpu"))
    _star_only(seen)
    assert seen["maps"] == [{"kind": "direct" if scale == 1 else "hash", "multi": True, "bitmap": False}]
    exp = _duck("SELECT id, k, v, dv, name, w FROM fact JOIN dim ON k = dk WHERE q < 70", fact=fact, dim=dim)
    assert got.num_rows > fact.num_rows * 0.7 * 0.5        # a real expansion
    _same(_rows(got, ["id", "k", "v", "dv", "name", "w"]), exp)


def test_rows_come_out_in_fact_order_then_partner_build_order(seen):
    for scale in (1, 1_000_003):
        fact = pa.table({"id": A(np.arange(8, dtype=np.int64)), "k": A(np.array([5, 9, 5, 1, 7, 9, 9, 5], np.int64) * scale)})
        dim = _dim(np.array([9, 5, 1, 9, 5, 9, 2], np.int64) * scale)       # dv: 0..6 in this order
        got = LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk").select("id", "dv").compute(device="cpu")
        assert _rows(got, ["id", "dv"]) == [(0, 1), (0, 4), (1, 0), (1, 3), (1, 5), (2, 1), (2, 4), (3, 2),
                                            (5, 0), (5, 3), (5, 5), (6, 0), (6, 3), (6, 5), (7, 1), (7, 4)]


def test_two_partners_one_repeats(seen):
    fact = _fact()
    unique = _dim(np.arange(0, 40), tag="u", seed=5).rename_columns(["ek", "ev", "ename", "ew"])
    rep = _dim(_repeating(900))
    got = (LazyFrame(fact).join(LazyFrame(unique), left_on="k2", right_on="ek")
           .join(LazyFrame(rep), left_on="k", right_on="dk")
           .select("id", "ev", "dv", "ename", "name").compute(device="cpu"))
    _star_only(seen)
    assert sorted(m["multi"] for m in seen["maps"]) == [False, True]
    exp = _duck("SELECT id, ev, dv, ename, name FROM fact JOIN unique_ ON k2 = ek JOIN rep ON k = dk",
                fact=fact, unique_=unique, rep=rep)
    _same(_rows(got, ["id", "ev", "dv", "ename", "name"]), exp)


def test_two_partners_both_repeat_multiply(seen):
    fact = _fact(n=20_000, kmax=300)
    a = _dim(_repeating(300, seed=7, max_rep=2), tag="a")
    b = _dim(_repeating(40, seed=8, max_rep=3), tag="b", seed=9).rename_columns(["bk", "bv", "bname", "bw"])
    got = (LazyFrame(fact).join(LazyFrame(a), left_on="k", right_on="dk")
           .join(LazyFrame(b), left_on="k2", right_on="bk")
           .select("id", "dv", "bv").compute(device="cpu"))
    _star_only(seen)
    assert all(m["multi"] for m in seen["maps"])
    exp = _duck("SELECT id, dv, bv FROM fact JOIN a ON k = dk JOIN b ON k2 = bk", fact=fact, a=a, b=b)
    _same(_rows(got, ["id", "dv", "bv"]), exp)


def test_composite_key_partner_repeats(seen):
    fact = _fact(n=30_000, kmax=200)
    r = np.random.default_rng(4)
    pairs = np.stack([r.integers(0, 200, 4000), r.integers(0, 40, 4000)], 1)    # (k, k2) repeats
    dim = pa.table({"pk": A(pairs[:, 0].astype(np.int64)), "pk2": A(pairs[:, 1].astype(np.int32)),
                    "dv": A(np.arange(4000, dtype=np.int32))})
    got = (LazyFrame(fact).join(LazyFrame(dim), left_on=["k", "k2"], right_on=["pk", "pk2"])
           .select("id", "dv").compute(device="cpu"))
    _star_only(seen)
    exp = _duck("SELECT id, dv FROM fact JOIN dim ON k = pk AND k2 = pk2", fact=fact, dim=dim)
    _same(_rows(got, ["id", "dv"]), exp)


def test_repeated_keys_past_2_pow_53(seen):
    base = (1 << 53)
    keys = np.array([base, base + 1, base + 1, base + 2, base + 2, base + 2, (1 << 62) + 5], np.int64)
    dim = _dim(keys)
    r = np.random.default_rng(1)
    pick = np.concatenate([keys, [base + 3, base - 1]]).astype(np.int64)
    fk = pick[r.integers(0, len(pick), 5000)]
    fact = pa.table({"id": A(np.arange(5000, dtype=np.int64)), "k": A(fk)})
    got = LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk").select("id", "k", "dv").compute(device="cpu")
    _star_only(seen)
    assert seen["maps"][0]["kind"] == "hash" and seen["maps"][0]["multi"]
    exp = _duck("SELECT id, k, dv FROM fact JOIN dim ON k = dk", fact=fact, dim=dim)
    _same(_rows(got, ["id", "k", "dv"]), exp)


def test_shuffled_rows_and_chunks(seen):
    fact = _fact(n=90_000, chunks=3)
    r = np.random.default_rng(11)
    fact = fact.take(pa.array(r.permutation(fact.num_rows), type=pa.int64()))
    dim = _dim(_repeating(900, seed=12), chunks=3)
    got = (LazyFrame(fact).filter(col("q") >= lit(10)).join(LazyFrame(dim), left_on="k", right_on="dk")
           .select("id", "dv", "name").compute(device="cpu"))
    _star_only(seen)
    exp = _duck("SELECT id, dv, name FROM fact JOIN dim ON k = dk WHERE q >= 10", fact=fact, dim=dim)
    _same(_rows(got, ["id", "dv", "name"]), exp)


def test_null_keys_never_match(seen):
    """Null keys are not looked up natively: the star declines (naming nulls) and the leaf-by-leaf join answers,
    with null keys matching nothing."""
    fact = _fact(n=20_000, nulls=True)
    dim = _dim(_repeating(900))
    got = LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk").select("id", "dv").compute(device="cpu")
    assert seen["star"] == 0 and seen["leaf"] == 1, seen
    assert any("nulls" in k for k in nx.decline_counts()), nx.decline_counts()
    exp = _duck("SELECT id, dv FROM fact JOIN dim ON k = dk", fact=fact, dim=dim)
    _same(_rows(got, ["id", "dv"]), exp)


EXPLODE = """
import resource, sys
import numpy as np, pyarrow as pa
import omxcore.native_exec as nx
from omxcore import LazyFrame
from omxcore.np_arrow import array as A
n = 70_000                      # one key on both sides: 4.9e9 joined rows
fact = pa.table({"id": A(np.arange(n, dtype=np.int64)), "k": A(np.zeros(n, np.int64))})
dim = pa.table({"dk": A(np.zeros(n, np.int64)), "dv": A(np.arange(n, dtype=np.int32))})
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
try:
    LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk").select("id", "dv").compute(device="cpu")
except MemoryError:
    pass
else:
    sys.exit("no MemoryError")
print("GROWTH_MB", (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - base) // (1 << 20 if sys.platform == "darwin" else 1 << 10))   # ru_maxrss: bytes on macOS, kB on Linux
print("DECLINES", nx.decline_counts())
"""


def test_an_output_beyond_the_limit_declines_before_anything_is_written():
    """The star counts the expansion in pass 1 and declines (-14) without buffering joined rows; the leaf-by-leaf
    join then refuses it (MemoryError). Peak memory stays near the normal footprint (a buffered expansion
    needed 3.9 GB)."""
    import os
    import subprocess
    import sys
    from conftest import _shim
    env = dict(os.environ, MXFRAME_NATIVE="1", MXFRAME_STRICT_NATIVE="1", MXFRAME_FUSED_JOIN_AGG="0",
               PYTHONPATH=str(_shim) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-c", EXPLODE], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr[-2000:]
    growth = int(r.stdout.split("GROWTH_MB")[1].split()[0])
    assert growth < 600, growth
    assert "executor declined (code -14)" in r.stdout, r.stdout


def test_aggregate_over_a_repeating_partner_runs_through_the_star(seen, monkeypatch):
    """The fused aggregate declines repeated keys (E_DUP, -18); the star scan answers, once."""
    monkeypatch.delenv("MXFRAME_FUSED_JOIN_AGG")
    fact = _fact()
    dim = _dim(_repeating(900))
    got = (LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk")
           .groupby("k2").agg(col("v").sum().alias("s"), col("dv").count().alias("n")).sort("k2").compute(device="cpu"))
    assert seen["star"] == 1 and seen["leaf"] == 0, seen
    exp = _duck("SELECT k2, SUM(v), COUNT(dv) FROM fact JOIN dim ON k = dk GROUP BY k2 ORDER BY k2", fact=fact, dim=dim)
    _same(_rows(got, ["k2", "s", "n"]), exp)


# ── presence bitmap on hashed maps ───────────────────────────────────────────────────────────────────

def test_hashed_map_has_a_presence_bitmap_when_its_range_is_small(seen):
    r = np.random.default_rng(21)
    keys = np.unique(r.integers(0, 3_000_000, 20_000)).astype(np.int64)      # 20k keys in 3M: sparse
    fact = pa.table({"id": A(np.arange(100_000, dtype=np.int64)), "k": A(r.integers(0, 3_000_000, 100_000).astype(np.int64))})
    dim = _dim(keys)
    got = LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk").select("id", "dv").compute(device="cpu")
    _star_only(seen)
    assert seen["maps"] == [{"kind": "hash", "multi": False, "bitmap": True}]
    exp = _duck("SELECT id, dv FROM fact JOIN dim ON k = dk", fact=fact, dim=dim)
    assert 0 < got.num_rows < 5000                       # most probes miss
    _same(_rows(got, ["id", "dv"]), exp)


def test_hashed_repeating_map_with_a_bitmap(seen):
    r = np.random.default_rng(22)
    keys = np.repeat(np.unique(r.integers(0, 3_000_000, 15_000)), 2).astype(np.int64)
    fact = pa.table({"id": A(np.arange(100_000, dtype=np.int64)), "k": A(r.choice(keys, 100_000))})
    dim = _dim(keys)
    got = LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk").select("id", "dv").compute(device="cpu")
    _star_only(seen)
    assert seen["maps"] == [{"kind": "hash", "multi": True, "bitmap": True}]
    exp = _duck("SELECT id, dv FROM fact JOIN dim ON k = dk", fact=fact, dim=dim)
    assert got.num_rows == 2 * fact.num_rows
    _same(_rows(got, ["id", "dv"]), exp)


def test_a_range_too_wide_for_a_bitmap_has_none(seen):
    fact = _fact(scale=1_000_003)
    dim = _dim(np.unique(_repeating(900, scale=1_000_003)))
    LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk").select("id", "dv").compute(device="cpu")
    _star_only(seen)
    assert seen["maps"] == [{"kind": "hash", "multi": False, "bitmap": False}]


def test_a_large_hashed_map_without_a_bitmap_is_prefetched_and_exact(seen):
    """100k keys over a 2^40 range: a hash table of ~2.4 MB (past the prefetch threshold), no bitmap; repeats
    on a tenth of the keys; misses and hits both resolve."""
    r = np.random.default_rng(23)
    keys = np.unique(r.integers(0, 1 << 40, 100_000)).astype(np.int64)
    keys = np.concatenate([keys, keys[:10_000]])
    fact = pa.table({"id": A(np.arange(150_000, dtype=np.int64)),
                     "k": A(np.where(r.random(150_000) < 0.5, r.choice(keys, 150_000),
                                     r.integers(0, 1 << 40, 150_000)).astype(np.int64))})
    dim = _dim(keys)
    got = LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk").select("id", "dv").compute(device="cpu")
    _star_only(seen)
    assert seen["maps"] == [{"kind": "hash", "multi": True, "bitmap": False}]
    exp = _duck("SELECT id, dv FROM fact JOIN dim ON k = dk", fact=fact, dim=dim)
    _same(_rows(got, ["id", "dv"]), exp)


# ── the runtime join filter ──────────────────────────────────────────────────────────────────────────

def _runtime_case(q_below, dim_rows=100_000):
    r = np.random.default_rng(31)
    n = 200_000
    fact = pa.table({"id": A(np.arange(n, dtype=np.int64)), "k": A(r.integers(0, dim_rows, n).astype(np.int64)),
                     "q": A(r.integers(0, 100, n).astype(np.int32))})
    dim = _dim(np.arange(dim_rows, dtype=np.int64))
    plan = (LazyFrame(fact).filter(col("q") < lit(q_below))
            .join(LazyFrame(dim), left_on="k", right_on="dk").select("id", "dv", "name"))
    exp = _duck(f"SELECT id, dv, name FROM fact JOIN dim ON k = dk WHERE q < {q_below}", fact=fact, dim=dim)
    return fact, dim, plan, exp


def test_runtime_filter_is_taken_when_the_partner_dwarfs_the_filtered_fact(seen, monkeypatch):
    monkeypatch.setenv("MXFRAME_RUNTIME_JOIN_FILTER", "1")      # the row-count rules alone
    fact, dim, plan, exp = _runtime_case(q_below=3)
    kept = int(sum(x < 3 for x in fact.column("q").to_pylist()))
    got = plan.compute(device="cpu")
    _star_only(seen)
    keys = len(set(k for k, q in zip(fact.column("k").to_pylist(), fact.column("q").to_pylist()) if q < 3))
    assert seen["build"] == [keys] and keys <= kept < dim.num_rows // 2
    _same(_rows(got, ["id", "dv", "name"]), exp)


def test_runtime_filter_is_skipped_when_the_fact_keeps_most_rows(seen, monkeypatch):
    monkeypatch.setenv("MXFRAME_RUNTIME_JOIN_FILTER", "1")
    fact, dim, plan, exp = _runtime_case(q_below=80)
    got = plan.compute(device="cpu")
    _star_only(seen)
    assert seen["build"] == [dim.num_rows]
    _same(_rows(got, ["id", "dv", "name"]), exp)


def test_runtime_filter_is_skipped_when_the_partner_is_not_twice_the_filtered_fact(seen, monkeypatch):
    monkeypatch.setenv("MXFRAME_RUNTIME_JOIN_FILTER", "1")
    fact, dim, plan, exp = _runtime_case(q_below=3, dim_rows=5_000)      # ~6k fact rows survive, 5k partner rows
    got = plan.compute(device="cpu")
    _star_only(seen)
    assert seen["build"] == [5_000]
    _same(_rows(got, ["id", "dv", "name"]), exp)


def test_runtime_filter_waits_for_a_partner_map_that_leaves_the_cache(seen, monkeypatch):
    """Default mode: the second pass over the fact's filter pays only when the partner's map (16 bytes per
    row) would not fit the last-level cache. A 100k-row partner fits a real L3; with a 1 MB L3 it does not."""
    fact, dim, plan, exp = _runtime_case(q_below=3)
    got = plan.compute(device="cpu")
    _star_only(seen)
    assert seen["build"] == [dim.num_rows]
    _same(_rows(got, ["id", "dv", "name"]), exp)
    nx.reset_decline_counts()
    seen.update(star=0, leaf=0)
    monkeypatch.setattr(je, "_l3_bytes", lambda: 1 << 20)
    co._JOIN_RESULT_CACHE.clear()                 # (the first answer would be served from it)
    got = plan.compute(device="cpu")
    _star_only(seen)
    assert seen["build"][0] < dim.num_rows // 10
    _same(_rows(got, ["id", "dv", "name"]), exp)


def test_runtime_filter_can_be_switched_off(seen, monkeypatch):
    monkeypatch.setenv("MXFRAME_RUNTIME_JOIN_FILTER", "0")
    monkeypatch.setattr(je, "_l3_bytes", lambda: 1 << 20)
    fact, dim, plan, exp = _runtime_case(q_below=3)
    got = plan.compute(device="cpu")
    _star_only(seen)
    assert seen["build"] == [dim.num_rows]
    _same(_rows(got, ["id", "dv", "name"]), exp)


def test_runtime_filter_keeps_every_row_of_a_repeated_key(seen, monkeypatch):
    monkeypatch.setenv("MXFRAME_RUNTIME_JOIN_FILTER", "1")
    r = np.random.default_rng(32)
    n = 150_000
    fact = pa.table({"id": A(np.arange(n, dtype=np.int64)), "k": A(r.integers(0, 60_000, n).astype(np.int64)),
                     "q": A(r.integers(0, 100, n).astype(np.int32))})
    dim = _dim(np.repeat(np.arange(60_000, dtype=np.int64), 2))
    got = (LazyFrame(fact).filter(col("q") < lit(2)).join(LazyFrame(dim), left_on="k", right_on="dk")
           .select("id", "dv").compute(device="cpu"))
    _star_only(seen)
    assert seen["build"][0] < dim.num_rows // 4 and seen["maps"][0]["multi"]
    exp = _duck("SELECT id, dv FROM fact JOIN dim ON k = dk WHERE q < 2", fact=fact, dim=dim)
    _same(_rows(got, ["id", "dv"]), exp)
