"""F-056 step 3a: three things that used to be "decline with a warning or MemoryError" now finish, in bounded memory.

1. Group state bigger than the budget: the partial of every batch is hash-partitioned to disk (Arrow IPC under
   MXFRAME_SPILL_DIR), each partition is merged on its own and the outputs are concatenated.
2. A query that returns rows (no aggregate): ORDER BY + LIMIT keeps a running top-k, LIMIT alone stops early, and a
   filter with no LIMIT concatenates when the result fits and raises a clear error naming the size when it does not.
3. One row group bigger than the budget is read in slices of rows.

Budgets are artificial (`MXFRAME_HOST_MEM_BUDGET`) over small files; the answers equal DuckDB's over the same file.
"""
import json
import math
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

from omxcore import col, gpu_stream
from omxcore.io_parquet import scan_parquet

from test_f056_streamed import close, prov, run, same

HAS_GPU = gpu_stream.cuda_available()
DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
           pytest.param("auto", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU"))]

N = 400_000
GROUPS = 150_000                 # group state: 150k x (key + 3 partials) x 8 bytes = 4.8 MB
HOST = "6M"                      # half of it (3 MB) is smaller than the group state; batches of ~1 MB of input


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    d = tmp_path_factory.mktemp("f056s3a")
    idx = np.arange(N)
    rng = np.random.default_rng(3)
    t = pa.table({
        "g": ((idx * 7919) % GROUPS).astype(np.int64),
        "v": rng.random(N) * 100,
        "q": rng.integers(1, 1000, N).astype(np.int64),
        "id": idx.astype(np.int64),                                     # unique: a sort key without ties
        "s": np.array([f"k{i:06d}" for i in range(GROUPS)])[(idx * 31) % GROUPS],
    })
    path = str(d / "big.parquet")
    pq.write_table(t, path, row_group_size=20_000)
    return path


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", HOST)
    spill = tmp_path / "tmp"
    spill.mkdir()
    monkeypatch.setenv("MXFRAME_SPILL_DIR", str(spill))
    return spill


def duck(query, path):
    con = duckdb.connect()
    con.execute(f"CREATE VIEW big AS SELECT * FROM read_parquet('{path}')")
    return con.sql(query).arrow().read_all()


def spill_leftovers(tmp):
    return sorted(os.listdir(tmp))


# --- 1. group state bigger than the budget ------------------------------------------------------------------

def all_groups(path):
    return (scan_parquet(path).groupby("g")
            .agg(col("v").sum().alias("sv"), col("q").sum().alias("sq"), col("v").mean().alias("mv"),
                 col("v").min().alias("lo"), col("q").max().alias("hi"), col("v").count().alias("n")))


def light_groups(path):
    """150k groups and three result columns: the result (3.6 MB) fits the budget, the group state (4.8 MB) does not."""
    return scan_parquet(path).groupby("g").agg(col("v").sum().alias("sv"), col("q").mean().alias("mq"))


@pytest.mark.parametrize("device", DEVICES)
def test_group_state_bigger_than_the_budget_is_spilled_and_equals_duckdb(big, device, _env):
    lf = light_groups(big)
    got = run(lf, device)
    p = prov(lf)
    assert p["path"] == "streamed"
    assert p["streamed"]["spill"]["partitions"] >= 2
    assert p["streamed"]["spill"]["bytes"] > 0
    want = duck("SELECT g, sum(v) AS sv, avg(q) AS mq FROM big GROUP BY g", big)
    assert got.num_rows == GROUPS
    same(got, want, ("g",))
    assert spill_leftovers(_env) == []


@pytest.mark.parametrize("device", DEVICES)
def test_having_order_by_and_limit_run_on_the_spilled_groups(big, device):
    lf = (all_groups(big).filter(col("n") >= 3).sort("sq", "g", descending=[True, False]).limit(12, 4))
    got = run(lf, device)
    assert prov(lf)["path"] == "streamed" and prov(lf)["streamed"]["spill"]["partitions"] >= 2
    want = duck("SELECT g, sum(v) AS sv, sum(q) AS sq, avg(v) AS mv, min(v) AS lo, max(q) AS hi, count(v) AS n "
                "FROM big GROUP BY g HAVING count(v) >= 3 ORDER BY sq DESC, g LIMIT 12 OFFSET 4", big)
    assert got.num_rows == 12
    same(got, want)


def test_string_keys_spill_and_group_once(big):
    lf = scan_parquet(big).groupby("s").agg(col("q").sum().alias("sq"), col("id").count().alias("n"))
    got = run(lf, "cpu")
    assert prov(lf)["streamed"]["spill"]["partitions"] >= 2
    same(got, duck("SELECT s, sum(q) AS sq, count(id) AS n FROM big GROUP BY s", big), ("s",))


def test_nan_null_and_signed_zero_keys_spilled_are_one_group_each(tmp_path, monkeypatch):
    """F-077: NaN of any bit pattern, -0.0 / 0.0 and NULL are one group each, whichever batch they appear in."""
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "1M")
    n = 100_000
    idx = np.arange(n)
    key = (idx % 40_000).astype(np.float64) + 0.5
    nan_neg = np.frombuffer(np.uint64(0xFFF8000000000001).tobytes(), dtype=np.float64)[0]
    key[idx % 997 == 0] = np.nan
    key[idx % 991 == 0] = nan_neg
    key[idx % 983 == 0] = -0.0
    key[idx % 977 == 0] = 0.0
    valid = ~(idx % 971 == 0)
    t = pa.table({"k": pa.array(key, mask=~valid), "x": (2 ** 53 + (idx % 5)).astype(np.int64)})
    path = str(tmp_path / "k.parquet")
    pq.write_table(t, path, row_group_size=5_000)
    lf = scan_parquet(path).groupby("k").agg(col("x").sum().alias("sx"), col("x").count().alias("n"))
    got = run(lf, "cpu")
    assert prov(lf)["streamed"]["spill"]["partitions"] >= 2
    rows = got.to_pylist()
    assert sum(1 for r in rows if r["k"] is not None and math.isnan(r["k"])) == 1
    assert sum(1 for r in rows if r["k"] is not None and r["k"] == 0.0) == 1
    assert sum(1 for r in rows if r["k"] is None) == 1
    want = duck_file("SELECT k, sum(x) AS sx, count(x) AS n FROM t GROUP BY k", path)
    key_of = lambda r: (r["k"] is None, r["k"] is not None and math.isnan(r["k"]), r["k"] or 0.0)     # noqa: E731
    g, w = sorted(rows, key=key_of), sorted(want.to_pylist(), key=key_of)
    assert len(g) == len(w)
    for a, b in zip(g, w):
        assert close(a["k"], b["k"]) and a["sx"] == b["sx"] and a["n"] == b["n"], (a, b)


def duck_file(query, path):
    con = duckdb.connect()
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{path}')")
    return con.sql(query).arrow().read_all()


def test_integer_sums_past_2_53_stay_exact_when_spilled(tmp_path, monkeypatch):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "1M")
    n = 100_000
    idx = np.arange(n)
    t = pa.table({"k": (idx % 30_000).astype(np.int64), "x": (2 ** 53 + 1 + idx % 7).astype(np.int64)})
    path = str(tmp_path / "x.parquet")
    pq.write_table(t, path, row_group_size=5_000)
    lf = scan_parquet(path).groupby("k").agg(col("x").sum().alias("sx"), col("x").min().alias("lo"))
    got = run(lf, "cpu")
    assert prov(lf)["streamed"]["spill"]["partitions"] >= 2
    assert got.schema.field("sx").type == pa.int64()
    same(got, duck_file("SELECT k, sum(x)::BIGINT AS sx, min(x) AS lo FROM t GROUP BY k", path), ("k",))


def test_a_reproducible_spilled_run_is_the_same_bits_in_the_same_order(big):
    a = run(light_groups(big), "cpu", reproducible=True)
    b = run(light_groups(big), "cpu", reproducible=True)
    assert a.equals(b)                                  # same rows in the same order, bit for bit
    for name in ("sv", "mq"):
        assert a.column(name).to_numpy().tobytes() == b.column(name).to_numpy().tobytes()


def test_the_spill_directory_is_removed_after_a_success(big, _env):
    run(light_groups(big), "cpu")
    assert spill_leftovers(_env) == []


def tiny_state(tmp_path, monkeypatch, budget="300K"):
    """100k groups (2.4 MB of partials) under a 300 KB budget: one of 16 partitions (150 KB, x4) is still too big, so
    it splits again."""
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", budget)
    n = 100_000
    idx = np.arange(n)
    t = pa.table({"k": ((idx * 7919) % 100_000).astype(np.int64), "v": (idx % 10).astype(np.float64)})
    path = str(tmp_path / "t.parquet")
    pq.write_table(t, path, row_group_size=5_000)
    lf = (scan_parquet(path).groupby("k").agg(col("v").sum().alias("sv"), col("v").count().alias("n"))
          .sort("sv", "k", descending=[True, False]).limit(5))
    want = duck_file("SELECT k, sum(v) AS sv, count(v) AS n FROM t GROUP BY k ORDER BY sv DESC, k LIMIT 5", path)
    return lf, want


def test_a_partition_that_is_still_too_big_is_split_again(tmp_path, monkeypatch, _env):
    lf, want = tiny_state(tmp_path, monkeypatch)
    got = run(lf, "cpu")
    assert prov(lf)["streamed"]["spill"]["depth"] >= 2
    assert prov(lf)["streamed"]["spill"]["partitions"] >= 16 + 2 * 16 // 2        # the second level split some partitions
    same(got, want)
    assert spill_leftovers(_env) == []


def test_splitting_past_the_depth_limit_raises_a_memory_error_and_cleans_up(tmp_path, monkeypatch, _env):
    from omxcore import streamed
    lf, _ = tiny_state(tmp_path, monkeypatch)
    monkeypatch.setattr(streamed._spill, "MAX_DEPTH", 1)
    with pytest.raises(MemoryError, match=r"too big to merge in the host memory budget .* after splitting the groups 1 times"):
        run(lf, "cpu")
    assert spill_leftovers(_env) == []


def test_the_spill_directory_is_removed_after_a_keyboard_interrupt(big, _env, monkeypatch):
    from omxcore import streamed

    def interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(streamed, "_read_partition", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(light_groups(big), "cpu")
    assert spill_leftovers(_env) == []


def test_a_full_disk_is_reported_as_a_spill_failure_naming_the_directory(big, _env, monkeypatch):
    from omxcore import spill
    real = pa.ipc.new_stream
    calls = []

    def full(*a, **k):
        calls.append(1)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(spill.pa.ipc, "new_stream", full)
    with pytest.raises(spill.SpillError, match=rf"spilling group state to {_env}/mxframe-spill-.* failed after about [\d,]+ bytes were written: .*No space left"):
        run(light_groups(big), "cpu")
    assert calls and spill_leftovers(_env) == []
    monkeypatch.setattr(spill.pa.ipc, "new_stream", real)


def test_too_little_free_space_is_reported_before_writing(big, _env, monkeypatch):
    from omxcore import spill
    monkeypatch.setattr(spill.shutil, "disk_usage", lambda p: type("U", (), {"free": 1000})())
    with pytest.raises(spill.SpillError, match=rf"not enough disk space to spill group state: {_env}/mxframe-spill-.* has 1,000 bytes free, [\d,]+ are needed"):
        run(light_groups(big), "cpu")
    assert spill_leftovers(_env) == []


def test_an_unwritable_spill_directory_is_reported(big, monkeypatch, tmp_path):
    from omxcore import spill
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("MXFRAME_SPILL_DIR", str(blocker / "sub"))
    with pytest.raises(spill.SpillError, match=r"spill directory .*file/sub cannot be created .*MXFRAME_SPILL_DIR"):
        run(light_groups(big), "cpu")


def test_the_default_spill_directory_is_on_disk_not_tmpdir(monkeypatch, tmp_path):
    from omxcore import spill
    monkeypatch.delenv("MXFRAME_SPILL_DIR")
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmpfs"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert spill.spill_base() == str(tmp_path / "cache" / "mxframe" / "spill")
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert spill.spill_base() == os.path.join(os.path.expanduser("~"), ".cache", "mxframe", "spill")


def test_a_spill_directory_in_ram_warns(big, _env, monkeypatch):
    from omxcore import spill
    monkeypatch.setattr(spill, "ram_backed", lambda path: "tmpfs")
    with pytest.warns(RuntimeWarning, match=rf"spill directory {_env} is on tmpfs \(RAM\).*MXFRAME_SPILL_DIR"):
        lf = light_groups(big)
        lf.compute(device="cpu")


def test_ram_backed_reads_the_mount_table(tmp_path, monkeypatch):
    from omxcore import spill
    here = str(tmp_path.resolve())
    table = f"/dev/sda1 / ext4 rw 0 0\ntmpfs {here} tmpfs rw 0 0\n"
    import builtins
    real = builtins.open
    monkeypatch.setattr(builtins, "open", lambda p, *a, **k: __import__("io").StringIO(table) if p == "/proc/mounts" else real(p, *a, **k))
    assert spill.ram_backed(here + "/x/y") == "tmpfs"
    assert spill.ram_backed("/home/someone") is None


def test_spill_folders_of_dead_processes_of_this_host_are_removed_and_the_rest_kept(big, _env):
    """Only folders made by this host and pid namespace are swept: in a cache shared by several machines or containers
    another one's pid cannot be looked up here, so its folder may belong to a live query."""
    from omxcore import spill
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    mine = spill._owner()
    (_env / f"mxframe-spill-{mine}-{dead.pid}-abc").mkdir()
    (_env / f"mxframe-spill-{mine}-{dead.pid}-abc" / "s1").write_text("stale")
    live = _env / f"mxframe-spill-{mine}-{os.getppid()}-live"
    live.mkdir()
    elsewhere = _env / f"mxframe-spill-otherhost-4026531836-{dead.pid}-x"        # same dead pid, another machine
    elsewhere.mkdir()
    other_ns = _env / f"mxframe-spill-{mine.rsplit('-', 1)[0]}-1-{dead.pid}-y"     # same host, another pid namespace
    other_ns.mkdir()
    (_env / "unrelated").mkdir()
    run(light_groups(big), "cpu")
    assert spill_leftovers(_env) == sorted([live.name, elsewhere.name, other_ns.name, "unrelated"])


def test_the_folder_name_carries_the_host_and_pid_namespace(big, _env, monkeypatch):
    from omxcore import spill
    seen = []
    real = spill.Partitioned.finish
    monkeypatch.setattr(spill.Partitioned, "finish", lambda self: (seen.extend(os.listdir(_env)), real(self))[1])
    run(light_groups(big), "cpu")
    assert seen and all(n.startswith(f"mxframe-spill-{spill._owner()}-{os.getpid()}-") for n in seen), seen


def test_a_deleted_partition_file_is_an_error_not_missing_groups(big, _env, monkeypatch):
    from omxcore import spill
    real = spill.Partitioned.finish

    def finish_then_delete_one(self):
        real(self)
        victim = next(i for i in range(spill.PARTITIONS) if self.rows[i])
        os.remove(self._path(victim))

    monkeypatch.setattr(spill.Partitioned, "finish", finish_then_delete_one)
    with pytest.raises(spill.SpillError, match=r"spill partition \d+ \([\d,]+ rows\) is missing"):
        run(light_groups(big), "cpu")
    assert spill_leftovers(_env) == []


def test_a_truncated_partition_file_is_an_error(big, _env, monkeypatch):
    from omxcore import spill
    real = spill.Partitioned.finish

    def finish_then_truncate_one(self):
        real(self)
        victim = next(i for i in range(spill.PARTITIONS) if self.rows[i])
        with open(self._path(victim), "r+b") as f:
            f.truncate(os.path.getsize(self._path(victim)) // 2)

    monkeypatch.setattr(spill.Partitioned, "finish", finish_then_truncate_one)
    with pytest.raises((spill.SpillError, pa.ArrowInvalid, OSError)):
        run(light_groups(big), "cpu")
    assert spill_leftovers(_env) == []


def test_an_error_while_finishing_the_partition_files_is_a_spill_error(big, _env, monkeypatch):
    from omxcore import spill

    class Broken:
        def __init__(self, w):
            self.w = w

        def write_table(self, t):
            self.w.write_table(t)

        def close(self):
            self.w.close()
            raise OSError(28, "No space left on device")

    real = spill.Partitioned._write

    def write_then_break(self, table):
        real(self, table)
        for i in self._writers:
            self._writers[i] = Broken(self._writers[i])

    monkeypatch.setattr(spill.Partitioned, "_write", write_then_break)
    with pytest.raises(spill.SpillError, match=r"failed while finishing the partition files .*No space left"):
        run(light_groups(big), "cpu")
    assert spill_leftovers(_env) == []


def test_the_spill_directory_is_removed_after_an_error(big, _env, monkeypatch):
    from omxcore import streamed
    seen = {}

    def broken(*a, **k):
        seen["dirs"] = spill_leftovers(_env)            # the spill files exist while the partitions are merged
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(streamed, "_read_partition", broken)
    with pytest.raises(RuntimeError, match="disk on fire"):
        run(light_groups(big), "cpu")
    assert len(seen["dirs"]) == 1
    assert spill_leftovers(_env) == []


PEAK = textwrap.dedent("""
    import json, sys
    import pyarrow as pa
    from omxcore import col
    from omxcore.io_parquet import scan_parquet

    def status(key):
        with open("/proc/self/status") as f:
            return next(int(l.split()[1]) * 1024 for l in f if l.startswith(key + ":"))

    path, mode = sys.argv[1], sys.argv[2]
    lf = scan_parquet(path)
    if mode == "groups":
        lf = (lf.groupby("g").agg(col("v").sum().alias("sv"), col("q").sum().alias("sq"), col("v").count().alias("n"))
              .filter(col("n") >= 3).sort("sq", "g", descending=[True, False]).limit(5))
    else:
        lf = lf.groupby("g2").agg(col("v").sum().alias("sv"), col("q").count().alias("n"))
    lf.schema
    lf.compute(device="cpu")                         # warm: the first run decodes and frees everything once
    if sys.platform == "linux":                      # macOS has no /proc and no way to restart the peak
        open("/proc/self/clear_refs", "w").write("5")
    base = status("VmRSS") if sys.platform == "linux" else 0
    pa.default_memory_pool().release_unused()
    out = lf.compute(device="cpu")
    print(json.dumps({"rows": out.num_rows, "path": lf.last_compile_provenance["path"],
                      "arrow_peak": pa.default_memory_pool().max_memory(), "rss_peak_growth": (status("VmHWM") - base) if sys.platform == "linux" else None}))
""")


def peak(path, mode, budget, tmp):
    import conftest
    env = dict(os.environ, MXFRAME_HOST_MEM_BUDGET=budget, MXFRAME_SPILL_DIR=str(tmp),
               PYTHONPATH=os.pathsep.join([str(conftest._shim)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    out = subprocess.run([sys.executable, "-c", PEAK, path, mode], env=env, capture_output=True, text=True, check=True,
                         timeout=600)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_peak_memory_with_spilled_groups_is_bounded_by_the_budget(big, _env):
    """The group state is 4.8 MB and the file 12 MB decoded: the Arrow pool never holds more than the 6 MB budget,
    while the same query with a big budget holds the whole file."""
    spilled = peak(big, "groups", HOST, _env)
    whole = peak(big, "groups", "64G", _env)
    assert spilled["path"] == "streamed" and whole["path"] != "streamed"
    assert spilled["rows"] == whole["rows"] == 5
    assert spilled["arrow_peak"] < 6 << 20, spilled
    assert whole["arrow_peak"] > 10 << 20, whole
    assert spill_leftovers(_env) == []


# --- 2. queries that return rows ----------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("offset", [0, 7])
def test_order_by_limit_streams_a_running_top_k(big, device, offset):
    lf = (scan_parquet(big).filter(col("q") > 100).select("id", "g", "v").sort("v", descending=True).limit(10, offset))
    got = run(lf, device)
    p = prov(lf)
    assert p["path"] == "streamed" and p["streamed"]["batches"] >= 3
    assert p["streamed"]["rows"]["kind"] == "top-k"
    want = duck(f"SELECT id, g, v FROM big WHERE q > 100 ORDER BY v DESC LIMIT 10 OFFSET {offset}", big)
    assert got.num_rows == 10
    same(got, want)


def test_order_by_limit_in_sql_streams_and_equals_duckdb(big):
    import omxcore as mx
    q = "SELECT id, v * 2 AS w, g FROM big WHERE g % 3 = 0 ORDER BY w DESC, id LIMIT 9 OFFSET 2"
    lf = mx.sql(q, big=scan_parquet(big))
    got = lf.compute(device="cpu")
    assert prov(lf)["path"] == "streamed"
    same(got, duck(q, big))


def test_limit_alone_stops_reading_once_it_has_enough(big):
    lf = scan_parquet(big).filter(col("q") > 500).select("id", "q").limit(6, 2)
    got = run(lf, "cpu")
    p = prov(lf)
    assert p["path"] == "streamed"
    assert p["streamed"]["rows"]["kind"] == "limit"
    assert p["streamed"]["rows"]["batches_run"] == 1 < p["streamed"]["batches"]
    assert got.num_rows == 6
    allowed = set(duck("SELECT id FROM big WHERE q > 500", big).column("id").to_pylist())
    assert set(got.column("id").to_pylist()) <= allowed
    assert all(q > 500 for q in got.column("q").to_pylist())


def test_limit_larger_than_what_matches_reads_everything_and_returns_it(big):
    lf = scan_parquet(big).filter((col("id") % 16_000) == 3).select("id").limit(100)
    got = run(lf, "cpu")
    assert prov(lf)["streamed"]["rows"]["batches_run"] == prov(lf)["streamed"]["batches"]
    assert sorted(got.column("id").to_pylist()) == list(range(3, N, 16_000))


@pytest.mark.parametrize("device", DEVICES)
def test_a_filter_whose_result_fits_is_streamed_and_concatenated(big, device):
    lf = scan_parquet(big).filter((col("id") % 500) == 3).select("id", "g", "v")
    got = run(lf, device)
    p = prov(lf)
    assert p["path"] == "streamed" and p["streamed"]["rows"]["kind"] == "collect"
    want = duck("SELECT id, g, v FROM big WHERE id % 500 = 3", big)
    assert got.num_rows == 800
    same(got.sort_by("id"), want.sort_by("id"))


def test_a_bare_filter_returns_every_column_including_strings(big):
    lf = scan_parquet(big).filter((col("id") % 20_000) == 3)
    got = run(lf, "cpu")
    assert prov(lf)["path"] == "streamed"
    assert got.column_names == ["g", "v", "q", "id", "s"] and got.num_rows == 20
    same(got.sort_by("id"), duck("SELECT * FROM big WHERE id % 20000 = 3 ORDER BY id", big))


def test_a_sort_without_limit_whose_result_fits_equals_duckdb(big):
    lf = scan_parquet(big).filter((col("id") % 1000) == 3).select("id", "v").sort("v")
    got = run(lf, "cpu")
    assert prov(lf)["path"] == "streamed"
    same(got, duck("SELECT id, v FROM big WHERE id % 1000 = 3 ORDER BY v", big))


def test_a_result_bigger_than_the_budget_raises_an_error_naming_the_size(big):
    lf = scan_parquet(big).select("id", "g", "v", "q")                  # all 400k rows x 32 bytes = 12.8 MB
    with pytest.raises(MemoryError, match=r"result of this query is bigger than memory: .*[\d.]+ MB .*budget of 6\.0 MB"):
        run(lf, "cpu")


def test_a_sort_without_limit_of_more_than_fits_names_the_size_and_sink_parquet(big):
    """Step 3c sorts it on disk, but the sorted result is one in-memory table: it must fit; sink_parquet has no limit."""
    lf = scan_parquet(big).select("id", "g", "v", "q").sort("v")
    with pytest.raises(MemoryError, match=r"bigger than memory: the 400,000 sorted rows are about [\d.]+ MB.*sink_parquet"):
        run(lf, "cpu")


def test_a_join_to_a_small_table_keeps_the_top_k_streamed(big, tmp_path):
    dim = str(tmp_path / "dim.parquet")
    pq.write_table(pa.table({"g": np.arange(2000, dtype=np.int64), "w": (np.arange(2000) % 11).astype(np.int64)}), dim)
    lf = (scan_parquet(big).join(scan_parquet(dim), on="g").filter(col("w") == 4)
          .select("id", "w", "q").sort("q", "id", descending=[True, False]).limit(8))
    got = run(lf, "cpu")
    assert prov(lf)["path"] == "streamed"
    con = duckdb.connect()
    con.execute(f"CREATE VIEW big AS SELECT * FROM read_parquet('{big}')")
    con.execute(f"CREATE VIEW dim AS SELECT * FROM read_parquet('{dim}')")
    same(got, con.sql("SELECT id, w, q FROM big JOIN dim USING (g) WHERE w = 4 ORDER BY q DESC, id LIMIT 8").arrow().read_all())


def test_a_top_k_whose_k_is_bigger_than_the_budget_raises_a_clear_error(big):
    lf = scan_parquet(big).select("id", "g", "v", "q").sort("v").limit(399_000)
    with pytest.raises(MemoryError, match="bigger than memory"):
        run(lf, "cpu")


# --- 3. one row group bigger than the budget -----------------------------------------------------------------

@pytest.fixture(scope="module")
def one_group(tmp_path_factory):
    n = 200_000
    idx = np.arange(n)
    t = pa.table({"g2": (idx % 50).astype(np.int64), "v": (idx % 1000) / 10.0, "q": idx.astype(np.int64),
                  "g": (idx % 50).astype(np.int64)})
    path = str(tmp_path_factory.mktemp("onegroup") / "one.parquet")
    # a single 6.4 MB row group; 32 KB pages: Arrow streams a column chunk page by page, so what is held while a
    # slice is read is the slice plus about one page per column (the default page is 1 MB: more than this test's budget)
    pq.write_table(t, path, row_group_size=n, data_page_size=32768, use_dictionary=False)
    assert pq.ParquetFile(path).metadata.num_row_groups == 1
    return path


@pytest.mark.parametrize("device", DEVICES)
def test_a_row_group_bigger_than_the_budget_is_read_in_slices(one_group, device, monkeypatch):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "3M")
    lf = scan_parquet(one_group).groupby("g2").agg(col("v").sum().alias("sv"), col("q").sum().alias("sq"),
                                                   col("q").count().alias("n"), col("v").mean().alias("mv"))
    got = run(lf, device)
    p = prov(lf)
    assert p["path"] == "streamed"
    assert p["streamed"]["batches"] >= 4 and p["streamed"]["row_groups"] == 1
    assert p["streamed"]["slices"]["rows_per_slice"] < 200_000
    same(got, duck_one("SELECT g2, sum(v) AS sv, sum(q) AS sq, count(q) AS n, avg(v) AS mv FROM one GROUP BY g2", one_group),
         ("g2",))


def duck_one(query, path):
    con = duckdb.connect()
    con.execute(f"CREATE VIEW one AS SELECT * FROM read_parquet('{path}')")
    return con.sql(query).arrow().read_all()


def test_slices_of_a_big_row_group_feed_a_top_k_too(one_group, monkeypatch):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "3M")
    lf = scan_parquet(one_group).filter(col("g") == 7).select("q", "v").sort("q", descending=True).limit(5)
    got = run(lf, "cpu")
    assert prov(lf)["path"] == "streamed" and prov(lf)["streamed"]["batches"] >= 4
    same(got, duck_one("SELECT q, v FROM one WHERE g = 7 ORDER BY q DESC LIMIT 5", one_group))


def test_peak_memory_of_a_row_group_bigger_than_the_budget_is_bounded(one_group, _env):
    out = peak(one_group, "one", "3M", _env)
    assert out["path"] == "streamed" and out["rows"] == 50
    assert out["arrow_peak"] < 3 << 20, out


def test_a_slice_read_out_of_step_with_the_batch_loop_raises(one_group, monkeypatch):
    """The slice iterator and the batch loop count slices separately; a plan that read the streamed file twice in a
    batch would silently skip rows, so it raises."""
    from omxcore import io_parquet
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "3M")
    real = io_parquet._next_slice
    seen = []

    def twice(st, *a, **k):
        seen.append(st.slice_no)
        real(st, *a, **k)
        return real(st, *a, **k)

    monkeypatch.setattr(io_parquet, "_next_slice", twice)
    lf = scan_parquet(one_group).groupby("g2").agg(col("q").sum().alias("sq"))
    with pytest.raises(RuntimeError, match=r"slice 0 of .* was asked for but the reader is at slice 1"):
        run(lf, "cpu")


def test_memory_held_elsewhere_lowers_the_size_at_which_a_merge_spills(_env):
    """The same partials merge in memory with nothing held and spill when resident tables / the sink hold most of the
    budget: the merge trigger counts what the query already holds."""
    from omxcore import spill, streamed
    n = 20_000
    tab = pa.table({"k": np.arange(n, dtype=np.int64), "v": np.ones(n)})
    path = os.path.join(str(_env), "h.parquet")
    pq.write_table(tab, path)
    spec = streamed._spec(scan_parquet(path).groupby("k").agg(col("v").sum().alias("sv")).plan)
    streamed._argument_types(spec)
    partial = pa.table({"k": tab.column("k"), "__p0": tab.column("v")})
    results = {}
    for held in (0, 2_000_000):
        root = spill.SpillDir()
        state = streamed._State(spec, 4_000_000, root, held=lambda h=held: h)
        state.add(partial)
        state.add(partial)
        state.merge()
        results[held] = state.spilled
        root.close()
    assert results == {0: False, 2_000_000: True}
