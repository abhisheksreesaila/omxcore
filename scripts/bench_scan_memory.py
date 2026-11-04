#!/usr/bin/env python
"""F-140: the memory claim of a streamed Parquet query, measured.

Writes a Parquet file several times a memory limit (pyarrow, one row group at a time, so the generator itself stays
small), then runs the same query two ways in a fresh process each, with a sampler reading the process's
RssAnon / RssFile / RssShmem from /proc every 10 ms (as scripts/bench_spill.py's _MemPeak does):
  scan  Query(scan_parquet(path))         streamed, one row group at a time
  read  read_parquet(path) + Query        the whole file in memory
and prints the answers (they must be equal), the peaks and the sizes. With `--limit-mb N` each run is also started
under a cgroup memory cap (systemd-run --user --scope -p MemoryMax) so "exceeds the limit" is the kernel's verdict.

Usage: pixi run python scripts/bench_scan_memory.py [--rows 60000000] [--group 2000000] [--limit-mb 600]
       [--dir ~/.cache/mxframe-bench/f140_bench] [--keep] [--query big|by_region]
The file goes under --dir (default ~/.cache/mxframe-bench/f140_bench) and is deleted at the end unless --keep.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
WORDS = ["West", "East", "Nörth", "South-East", "Far", "Central", "Zed", "Mid"]


def write_file(path, rows, group, seed=1):
    rng = np.random.default_rng(seed)
    schema = pa.schema([("region", pa.string()), ("carrier", pa.int64()), ("day", pa.date32()),
                        ("revenue", pa.decimal128(18, 2)), ("cost", pa.int64())])
    words = pa.array(WORDS)
    w = pq.ParquetWriter(path, schema, compression="none", use_dictionary=["region"])
    done = 0
    while done < rows:
        n = min(group, rows - done)
        start = (done // group) % 4                                 # row-group dictionaries differ
        region = words.take(pa.array((start + rng.integers(0, 5, n)) % 8))
        carrier = pa.array(rng.integers(-1, 9, n))
        day = pa.array(rng.integers(9000, 15000, n).astype("int32"), pa.int32()).cast(pa.date32())
        cents = rng.integers(-10**15, 10**15, n)                      # incompressible 8-byte values
        revenue = pa.Array.from_buffers(pa.decimal128(18, 2), n, [None, pa.py_buffer(
            np.column_stack([cents, cents >> 63]).astype("<i8").tobytes())])
        cost = pa.array(rng.integers(-2**40, 2**40, n))
        w.write_table(pa.table([region, carrier, day, revenue, cost], schema=schema))
        done += n
    w.close()


class MemPeak:
    """Peak RssAnon / RssFile / RssShmem (MB) of a child process, sampled from /proc/<pid>/status."""

    def __init__(self, pid):
        self.pid, self.peak = pid, {"anon": 0, "file": 0, "shmem": 0, "total": 0}
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.wait(0.01):
            try:
                with open(f"/proc/{self.pid}/status") as f:
                    v = {}
                    for line in f:
                        for k, key in (("RssAnon:", "anon"), ("RssFile:", "file"), ("RssShmem:", "shmem")):
                            if line.startswith(k):
                                v[key] = int(line.split()[1]) // 1024
            except OSError:
                return
            if len(v) == 3:
                v["total"] = sum(v.values())
                for k in self.peak:
                    self.peak[k] = max(self.peak[k], v[k])

    def stop(self):
        self._stop.set()
        self._t.join()
        return self.peak


def run(probe, path, mode, query, limit_mb):
    cmd = [str(probe), str(path), mode, query]
    if limit_mb:
        cmd = ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={limit_mb}M", "-p", "MemorySwapMax=0"] + cmd
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sampler = MemPeak(p.pid)
    out, err = p.communicate()
    peak = sampler.stop()
    return p.returncode, out, err, peak, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=60_000_000)
    ap.add_argument("--group", type=int, default=2_000_000)
    ap.add_argument("--limit-mb", type=int, default=0)
    ap.add_argument("--dir", default=os.path.expanduser("~/.cache/mxframe-bench/f140_bench"))
    ap.add_argument("--query", default="by_region")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "big.parquet"
    try:
        if not path.exists():
            t0 = time.perf_counter()
            write_file(path, args.rows, args.group)
            print(f"wrote {path}: {path.stat().st_size / 2**20:.0f} MB, {args.rows:,} rows in "
                  f"{pq.read_metadata(path).num_row_groups} row groups ({time.perf_counter() - t0:.0f} s)")
        size_mb = path.stat().st_size / 2**20
        probe = d / "pq_scan_probe"
        subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_scan_probe.mojo"), str(probe)],
                       check=True, capture_output=True, text=True)
        decoded_group_mb = args.group * 8 * (3 if args.query == "by_region" else 2) / 2**20
        print(f"file {size_mb:.0f} MB; one row group of the columns the query reads, decoded: {decoded_group_mb:.0f} MB; "
              f"limit {args.limit_mb or 'none'} MB; query {args.query}")
        answers = {}
        for mode in ("scan", "read"):
            rc, out, err, peak, secs = run(probe, path, mode, args.query, args.limit_mb)
            if rc != 0:
                print(f"{mode:5s} FAILED (exit {rc}){' - killed by the memory cap' if rc in (-9, 137) else ''}: "
                      f"{(out + err).strip()[-200:]}  peak while alive: {peak}")
                continue
            res = json.loads(out.strip().split("\n")[-1])
            answers[mode] = {k: res[k] for k in ("words", "kinds", "slot_of", "count_slot_of", "counts", "slots", "present")}
            print(f"{mode:5s} peak RSS {peak['total']:>6} MB (anon {peak['anon']}, file {peak['file']}, shmem {peak['shmem']}); "
                  f"process high-water mark {res['peak_kb'] // 1024} MB; query {res['ms']:.0f} ms; wall {secs:.1f} s")
        if len(answers) == 2:
            print("answers equal:", answers["scan"] == answers["read"])
            print("rows counted:", sum(answers["scan"]["counts"]))
    finally:
        if not args.keep:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
