"""A/B one environment switch on the random queries of one shape, omxcore only, alternating runs.

    pixi run python scripts/bench_random/ab_env.py --shape select_distinct --env MXFRAME_DISTINCT_GROUPBY \
        --sf 1 --n 100 --reps 3

Each (query, setting) runs `worker.py mx` in a fresh process (columns loaded untimed, first run timed), the
two settings alternating so a loaded machine hits both alike; the median of `--reps` per setting is printed,
with whether the results are equal. Rough by construction (a shared machine): label the numbers.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import gen  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scratch as _scratch  # noqa: E402


def run(spec_path, data_dir, out, env):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from tpch_user.runner import worker_env
    e = dict(worker_env("-1"), **env)
    p = subprocess.run([sys.executable, os.path.join(HERE, "worker.py"), "mx", spec_path, data_dir, out],
                       capture_output=True, text=True, env=e, cwd=ROOT, timeout=300)
    lines = [l for l in p.stdout.splitlines() if l.startswith("@@RESULT ")]
    return json.loads(lines[-1][9:]) if lines else {"error": (p.stderr or p.stdout)[-200:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--env", required=True, help="the variable: set to 1 (on) and 0 (off)")
    ap.add_argument("--sf", default="1")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    data_dir = os.path.join(ROOT, "data", f"tpch_sf{a.sf}")
    st = gen.stats(os.path.join(ROOT, "data", "tpch_sf1"))
    tmp = tempfile.mkdtemp(prefix="mxab-", dir=_scratch.root())
    rows = []
    for seed in range(a.seed0, a.seed0 + a.n):
        spec = gen.make(seed, st)
        if spec["shape"] != a.shape:
            continue
        sp = os.path.join(tmp, f"q{seed}.json")
        json.dump(spec, open(sp, "w"))
        ms = {"1": [], "0": []}
        paths = {}
        outs = {}
        for rep in range(a.reps):
            for setting in ("0", "1") if rep % 2 == 0 else ("1", "0"):
                out = os.path.join(tmp, f"q{seed}-{setting}.parquet")
                r = run(sp, data_dir, out, {a.env: setting})
                if "error" in r:
                    print(f"q{seed} {setting}: {r['error']}")
                    continue
                ms[setting].append(r["ms"])
                paths[setting] = r.get("native_calls")
                outs[setting] = out
        if not (ms["0"] and ms["1"]):
            continue
        import pyarrow.parquet as pq
        same = pq.read_table(outs["0"]).equals(pq.read_table(outs["1"]))
        off, on = statistics.median(ms["0"]), statistics.median(ms["1"])
        rows.append((seed, off, on, same))
        print(f"q{seed:<4} {a.shape:16s} off {off:8.1f} ms  on {on:8.1f} ms  {off / on:5.2f}x  same={same}  "
              f"calls on={sorted(set(paths.get('1') or []))}", flush=True)
    if rows:
        import math
        gm = math.exp(sum(math.log(o / n) for _, o, n, _ in rows) / len(rows))
        print(f"\n{len(rows)} queries, geometric mean off/on {gm:.2f}x (>1: on is faster); all equal: {all(r[3] for r in rows)}")


if __name__ == "__main__":
    main()
