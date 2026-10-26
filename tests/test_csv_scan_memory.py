"""F-178: a CSV streamed with `scan_csv` stays inside a memory bound that does not grow with the file.

mojo/tools/csv_scan_probe.mojo runs one SQL string in a fresh process, over the file read whole or streamed, and prints the answer's
checksum and the process's peak resident memory (VmHWM, file pages included). The files here are 3 and 9 times a 16 MB window; the
streamed peak is the same for both (and far below the file), the whole-file read's is several times the file, and the answers agree
with each other and with DuckDB.
"""
import fcntl
import json
import subprocess
from pathlib import Path

import duckdb
import pytest

from engine_sources import engine_sources

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
GROUP_ROWS = 100_000
BLOCK_BYTES = 16 << 20
QUERY = "SELECT region, count(*), sum(qty), sum(amount), min(day), max(day), sum(flag) FROM t GROUP BY region ORDER BY region"
SMALL, BIG = 700_000, 2_100_000          # rows: about 50 MB and 150 MB of CSV


@pytest.fixture(scope="module")
def probe():
    BUILD.mkdir(exist_ok=True)
    out = BUILD / "csv_scan_probe"
    sources = engine_sources(ROOT / "mojo/tools/csv_scan_probe.mojo")
    with open(BUILD / ".csv_scan_probe.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not out.exists() or out.stat().st_mtime < max(p.stat().st_mtime for p in sources):
            subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/csv_scan_probe.mojo"), str(out)],
                           check=True, capture_output=True, text=True)
    return out


def make_csv(path, n):
    con = duckdb.connect()
    v = "((i * 7) % 2000001 - 1000000)"
    amount = f"(case when {v} < 0 then '-' else '' end || (abs({v}) // 100)::text || '.' || lpad((abs({v}) % 100)::text, 2, '0'))"
    con.execute(f"""copy (select i as id, 'region-' || lpad(((i * 7919) % 40)::text, 2, '0') as region, (i * 31 % 2001 - 1000) as qty,
        {amount} as amount, date '2015-01-01' + (i % 3000)::int as day, (i % 3 = 0) as flag,
        md5(i::text) as note from range({n}) t(i)) to '{path}' (header)""")


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("csvmem")
    out = {}
    for name, n in (("small", SMALL), ("big", BIG)):
        out[name] = d / f"{name}.csv"
        make_csv(out[name], n)
    assert out["big"].stat().st_size > 8 * BLOCK_BYTES and out["big"].stat().st_size > 2.5 * out["small"].stat().st_size
    return out


def run(probe, mode, path, threads=2):
    p = subprocess.run([str(probe), mode, str(path), QUERY, str(GROUP_ROWS), str(BLOCK_BYTES), str(threads)], check=True, capture_output=True, text=True)
    return json.loads(p.stdout)


def test_the_streamed_peak_does_not_grow_with_the_file(probe, files):
    small = run(probe, "scan", files["small"])
    big = run(probe, "scan", files["big"])
    file_kb = files["big"].stat().st_size // 1024
    assert big["row_groups"] >= 20
    assert big["vm_hwm_kb"] < 100_000, big                                   # a 150 MB file in under 100 MB, runtime included
    assert big["vm_hwm_kb"] < small["vm_hwm_kb"] + 25_000, (small, big)       # three times the file, the same peak (a little for the bigger index)
    # the bound by formula: ~40 MB of runtime and index, then per worker thread one group (100,000 rows x 7 columns x 8 bytes) and its mapped bytes (< 30 MB)
    assert big["vm_hwm_kb"] < 40_000 + 2 * 30_000 and big["vm_hwm_kb"] < file_kb / 2, big


def test_the_whole_file_read_holds_the_file(probe, files):
    whole = run(probe, "table", files["big"])
    scan = run(probe, "scan", files["big"])
    assert whole["vm_hwm_kb"] > 3 * scan["vm_hwm_kb"], (whole, scan)           # the contrast the bound is for


def test_the_answers_are_equal_to_the_whole_read_and_to_duckdb(probe, files):
    for name in ("small", "big"):
        whole = run(probe, "table", files[name])
        scan = run(probe, "scan", files[name])
        assert (whole["rows"], whole["checksum"]) == (scan["rows"], scan["checksum"]) == (40, scan["checksum"]), (whole, scan)
        con = duckdb.connect()
        want = con.execute(f"""select region, count(*), sum(qty), sum(amount), min(day), max(day), sum(flag::int)
            from read_csv('{files[name]}', header=true, columns={{'id': 'BIGINT', 'region': 'VARCHAR', 'qty': 'BIGINT', 'amount': 'DECIMAL(18,2)',
            'day': 'DATE', 'flag': 'BOOLEAN', 'note': 'VARCHAR'}}) group by region order by region""").fetchall()
        # the probe prints a checksum, so compare the same checksum computed from DuckDB's rows
        epoch = __import__("datetime").date(1970, 1, 1)
        total = 0
        for region, n, qty, amount, lo, hi, flags in want:
            total += len(region.encode()) + n + qty + int(amount * 100) + (lo - epoch).days + (hi - epoch).days + flags
        assert scan["checksum"] == total, (name, scan, total)
