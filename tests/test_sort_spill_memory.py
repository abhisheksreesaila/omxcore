"""F-167: a sort bigger than its budget stays inside a memory bound that does not grow with the file.

mojo/tools/sort_spill_probe.mojo runs one ORDER BY in a fresh process over a Parquet file streamed with a scan, with a sort budget, and
prints the process's peak resident memory (VmHWM, file pages included). The sorted rows go straight into a new Parquet file
(`sql_to_parquet`). The files here are 3 and 6 million rows (4 columns, 96 and 192 MB of values); with a 64 MiB budget the peak is the
same for both and a fraction of what the sort in memory needs; the new file holds the rows in DuckDB's order.

The bound by formula: the runtime and the open file (about 25 MB), then the sort budget, then the fixed extras outside the budget, a
row group of the scanned file being decoded, the merge's output group and the writer's pages (< 40 MB for this file), so
peak < 25 MB + budget + 40 MB whatever the file's size. What today's sort needs is measured too: the sort in memory peaks above 3 times
the bytes of the values it sorts (3.9x here: the columns, the keys, the row order and the radix passes' buffers).
"""
import fcntl
import json
import os
import subprocess
from pathlib import Path

import duckdb
import pytest

from engine_sources import engine_sources

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
BUDGET = 64 << 20
SMALL, BIG = 3_000_000, 6_000_000
QUERY = "SELECT id, k, txt, x FROM t ORDER BY txt DESC, k, id"
DUCK = "order by txt desc, k, id"
BOUND_KB = (25 << 10) + (BUDGET >> 10) + (40 << 10)


@pytest.fixture(scope="module")
def probe():
    BUILD.mkdir(exist_ok=True)
    out = BUILD / "sort_spill_probe"
    sources = engine_sources(ROOT / "mojo/tools/sort_spill_probe.mojo")
    with open(BUILD / ".sort_spill_probe.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not out.exists() or out.stat().st_mtime < max(p.stat().st_mtime for p in sources):
            subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/sort_spill_probe.mojo"), str(out)],
                           check=True, capture_output=True, text=True)
    return out


@pytest.fixture(scope="module")
def files():
    d = Path(os.path.expanduser("~/.cache/mxframe-test-tmp")) / f"f167-mem-{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "spill").mkdir(exist_ok=True)
    out = {"dir": d}
    con = duckdb.connect()
    for name, n in (("small", SMALL), ("big", BIG)):
        p = d / f"{name}.parquet"
        con.execute(f"""copy (select i as id, ((i * 2654435761) % 1000003)::int as k, 'word-' || ((i * 7919) % 5000)::text as txt,
            ((i * 2654435761) % 4294967296) / 1000.0 as x from range({n}) t(i)) to '{p}' (format parquet, row_group_size 100000)""")
        out[name] = p
    yield out
    for p in list(d.glob("**/*")):
        if p.is_file():
            p.unlink()


def run(probe, files, mode, name, budget, out=None, group=131072):
    args = [str(probe), mode, str(files[name]), QUERY, str(budget), str(out) if out else "", str(files["dir"] / "spill"), str(group)]
    p = subprocess.run(args, check=True, capture_output=True, text=True)
    return json.loads(p.stdout)


def test_the_peak_is_the_budget_not_the_file(probe, files):
    small = run(probe, files, "file", "small", BUDGET, files["dir"] / "small-out.parquet")
    big = run(probe, files, "file", "big", BUDGET, files["dir"] / "big-out.parquet")
    for r, n in ((small, SMALL), (big, BIG)):
        assert r["rows"] == n and r["streamed"] and r["runs"] >= 4, r
        assert r["spill_files_left"] == 0, r
        assert r["vm_hwm_kb"] < BOUND_KB, (r, BOUND_KB)                        # by formula, not tuned to the machine
    assert big["runs"] > small["runs"] * 1.8, (small, big)                  # twice the rows, twice the runs ...
    assert abs(big["vm_hwm_kb"] - small["vm_hwm_kb"]) < 15_000, (small, big)  # ... and the same peak


def test_the_sorted_file_equals_duckdb(probe, files):
    run(probe, files, "file", "small", BUDGET, files["dir"] / "small-out.parquet")
    con = duckdb.connect()
    src, out = files["small"], files["dir"] / "small-out.parquet"
    same = con.execute(f"""select count(*) from (select id, row_number() over () rn from read_parquet('{out}')) a
        join (select id, row_number() over ({DUCK}) rn from read_parquet('{src}')) b using (id, rn)""").fetchone()[0]
    assert same == SMALL
    cols = con.execute(f"select * from read_parquet('{out}') limit 1").description
    assert [c[0] for c in cols] == ["id", "k", "txt", "x"]


def test_the_sort_in_memory_needs_several_times_its_input(probe, files):
    """What today's sort needs: measured, the contrast the budget is for."""
    plain = run(probe, files, "scan", "small", 0)
    spilled = run(probe, files, "file", "small", BUDGET, files["dir"] / "small-out.parquet")
    values_kb = SMALL * 4 * 8 // 1024
    assert plain["runs"] == 0
    assert plain["vm_hwm_kb"] - plain["base_hwm_kb"] > 3 * values_kb, (plain, values_kb)
    assert plain["vm_hwm_kb"] > 3 * spilled["vm_hwm_kb"], (plain, spilled)
