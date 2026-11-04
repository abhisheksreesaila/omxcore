#!/usr/bin/env python
"""F-141: Parquet decode timings, one thread (the serial build) and all cores.

Makes a 20M-row numeric file and a 5M-row text-heavy file under ~/.cache/mxframe-bench/f141_bench (pyarrow, 1M-row groups, snappy), builds
mojo/tools/pq_decode_bench.mojo twice (normal, and -D MXFRAME_SERIAL), and prints for each:
  scan numeric   Query(scan_parquet) end to end
  read numeric   read_parquet (decode) and then the in-memory kernel's time on the same query
  read text      read_parquet on the text-heavy file
Usage: pixi run python scripts/bench_parquet_decode.py [--keep] [--codec snappy]
Informal: run on a quiet machine for quotable numbers; the load average is printed.
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb

ROOT = Path(__file__).resolve().parent.parent
D = Path.home() / ".cache/mxframe-bench/f149"


def make_numeric(path, n, group, codec):
    rng = np.random.default_rng(1)
    schema = pa.schema([("carrier", pa.int64()), ("day", pa.date32()), ("revenue", pa.decimal128(18, 2)),
                        ("cost", pa.int64())])
    w = pq.ParquetWriter(path, schema, compression=codec)
    for lo in range(0, n, group):
        m = min(group, n - lo)
        cents = rng.integers(-10**12, 10**12, m)
        rev = pa.Array.from_buffers(pa.decimal128(18, 2), m, [None, pa.py_buffer(
            np.column_stack([cents, cents >> 63]).astype("<i8").tobytes())])
        carrier = pa.array(np.where(rng.random(m) < 0.02, -1, rng.integers(0, 10, m)))
        w.write_table(pa.table([carrier, pa.array(rng.integers(9000, 15000, m).astype("int32")).cast(pa.date32()),
                                rev, pa.array(rng.integers(-2**40, 2**40, m))], schema=schema))
    w.close()


def make_text(path, n, group, codec):
    rng = np.random.default_rng(2)
    words = pa.array([f"region-{i}" for i in range(8)])
    names = pa.array([f"customer name number {i:07d}" for i in range(200_000)])
    schema = pa.schema([("region", pa.string()), ("name", pa.string()), ("cost", pa.int64())])
    w = pq.ParquetWriter(path, schema, compression=codec)
    for lo in range(0, n, group):
        m = min(group, n - lo)
        nm = names.take(pa.array(rng.integers(0, 200_000, m)))
        region = words.take(pa.array(rng.integers(0, 8, m)))
        w.write_table(pa.table([region, nm, pa.array(rng.integers(0, 1000, m))], schema=schema))
    w.close()


def build(serial):
    out = D / ("bench_serial" if serial else "bench")
    lib = ROOT / ".pixi/envs/default/lib"
    cmd = ["mojo", "build", "--disable-warnings", str(ROOT / "mojo/tools/pq_decode_bench.mojo"), "-I", str(ROOT / "build"),
           "-Xlinker", f"-L{lib}", "-Xlinker", "-lzstd", "-Xlinker", "-lsnappy", f"-Xlinker", f"-rpath={lib}", "-o", str(out)]
    if serial:
        cmd[2:2] = ["-D", "MXFRAME_SERIAL"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


def run(binary, *args, reps=3):
    best = None
    for _ in range(reps):
        res = json.loads(subprocess.run([str(binary), *map(str, args)], check=True, capture_output=True, text=True).stdout)
        key = res.get("total_s") or res["decode_s"]
        if best is None or key < (best.get("total_s") or best["decode_s"]):
            best = res
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--codec", default="snappy")
    a = ap.parse_args()
    D.mkdir(exist_ok=True)
    try:
        subprocess.run(["bash", str(ROOT / "scripts/build_mojo_pkg.sh")], check=True, capture_output=True)
        num, txt = D / "numeric.parquet", D / "text.parquet"
        if not num.exists():
            make_numeric(num, 20_000_000, 1_000_000, a.codec)
            make_text(txt, 5_000_000, 1_000_000, a.codec)
        print(f"load average {os.getloadavg()}; numeric {num.stat().st_size >> 20} MB, text {txt.stat().st_size >> 20} MB, codec {a.codec}")
        sql = ("SELECT carrier, sum(cost), sum(CAST(revenue * 100 AS BIGINT)), count(*), max(CAST(day AS DATE) - DATE '1970-01-01') "
               f"FROM read_parquet('{num}') WHERE carrier BETWEEN 0 AND 9 GROUP BY carrier")
        for serial in (True, False):
            con = duckdb.connect()
            con.execute(f"SET threads={1 if serial else os.cpu_count()}")
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                con.execute(sql).fetchall()
                ts.append(time.perf_counter() - t0)
            print(f"DuckDB {'1 thread ' if serial else 'all cores'}: streamed numeric query {min(ts):.3f} s")
            b = build(serial)
            tag = "1 thread " if serial else "all cores"
            s = run(b, num, "scan", "numeric")
            r = run(b, num, "read", "numeric")
            t = run(b, txt, "read", "text")
            ts = run(b, txt, "scan", "text")
            print(f"{tag} | scan numeric 20M {s['total_s']:.3f} s | read numeric 20M {r['decode_s']:.3f} s (in-memory kernel {r['kernel_ms']:.0f} ms)"
                  f" | read text 5M {t['decode_s']:.3f} s | scan text {ts['total_s']:.3f} s | checks {s['check']} {r['check']} {t['check']}")
    finally:
        if not a.keep:
            shutil.rmtree(D, ignore_errors=True)


if __name__ == "__main__":
    main()
