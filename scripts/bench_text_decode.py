#!/usr/bin/env python
"""F-142: text Parquet decode against pyarrow, one thread and all cores.

Makes a 5M-row text-heavy file (region: 8 words, name: 200k words that fall back to PLAIN pages, cost: int64; 1M-row
groups) under ~/.cache/mxframe-bench/f142 and times
  read_parquet   mojo/tools/pq_decode_bench.mojo (normal build, and -D MXFRAME_SERIAL for one thread)
  pyarrow        pq.read_table with use_threads=True / False (strings stay strings, as ours become dictionary codes)
Best of 3 each; the load average is printed. Usage: pixi run python scripts/bench_text_decode.py [--keep] [--codec snappy]
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

ROOT = Path(__file__).resolve().parent.parent
D = Path.home() / ".cache/mxframe-bench/f142"


def make_text(path, n, group, codec):
    rng = np.random.default_rng(2)
    words = pa.array([f"region-{i}" for i in range(8)])
    names = pa.array([f"customer name number {i:07d}" for i in range(200_000)])
    schema = pa.schema([("region", pa.string()), ("name", pa.string()), ("cost", pa.int64())])
    w = pq.ParquetWriter(path, schema, compression=codec)
    for lo in range(0, n, group):
        m = min(group, n - lo)
        w.write_table(pa.table([words.take(pa.array(rng.integers(0, 8, m))), names.take(pa.array(rng.integers(0, 200_000, m))),
                                pa.array(rng.integers(0, 1000, m))], schema=schema))
    w.close()


def build(serial):
    out = D / ("bench_serial" if serial else "bench")
    lib = ROOT / ".pixi/envs/default/lib"
    cmd = ["mojo", "build", "--disable-warnings", str(ROOT / "mojo/tools/pq_decode_bench.mojo"), "-I", str(ROOT / "build"),
           "-Xlinker", f"-L{lib}", "-Xlinker", "-lzstd", "-Xlinker", "-lsnappy", "-Xlinker", "-rpath", "-Xlinker", str(lib), "-o", str(out)]
    if serial:
        cmd[2:2] = ["-D", "MXFRAME_SERIAL"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


def best(f, reps=7):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return min(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--codec", default="snappy")
    a = ap.parse_args()
    D.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["bash", str(ROOT / "scripts/build_mojo_pkg.sh")], check=True, capture_output=True)
        path = D / "text.parquet"
        if not path.exists():
            make_text(path, 5_000_000, 1_000_000, a.codec)
        print(f"load average {os.getloadavg()}; text file {path.stat().st_size >> 20} MB, codec {a.codec}")
        for serial in (True, False):
            b = build(serial)
            runs = [json.loads(subprocess.run([str(b), str(path), "read", "text"], check=True, capture_output=True, text=True).stdout)
                    for _ in range(7)]
            r = min(runs, key=lambda x: x["decode_s"])
            print(f"mojo read_parquet {'1 thread ' if serial else 'all cores'}: {r['decode_s']:.3f} s (check {r['check']})")
        pa.set_cpu_count(os.cpu_count())
        print(f"pyarrow all cores: {best(lambda: pq.read_table(path, use_threads=True)):.3f} s (plain strings); "
              f"{best(lambda: pq.read_table(path, use_threads=True, read_dictionary=['region', 'name'])):.3f} s (dictionary-encoded, as ours)")
        pa.set_cpu_count(1)
        print(f"pyarrow 1 thread : {best(lambda: pq.read_table(path, use_threads=False)):.3f} s (plain strings); "
              f"{best(lambda: pq.read_table(path, use_threads=False, read_dictionary=['region', 'name'])):.3f} s (dictionary-encoded, as ours)")
    finally:
        if not a.keep:
            shutil.rmtree(D, ignore_errors=True)


if __name__ == "__main__":
    main()
