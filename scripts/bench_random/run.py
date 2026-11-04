"""F-044: random queries on the TPC-H data, engine-only, omxcore vs Polars vs DuckDB.

    pixi run python scripts/bench_random/run.py --sf 1 --n 100 [--shuffle] [--seed0 0] --out docs/bench

Each (query, engine) runs in a fresh process (worker.py): load untimed, first run timed. DuckDB's
result is the reference: ordered queries are compared row by row, others as sorted rows; floats
within 1e-3 relative (inputs are float32), `drift` up to 1e-2, else `WRONG`. omxcore also reports
which native kernels ran and why a part fell back to Python.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import gen  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scratch as _scratch  # noqa: E402

ENGINES = ["duckdb", "mx", "polars"]
TIE = 1e-6          # sort values this close (relative) may come out in either order


def _run(engine, spec_path, data_dir, out, shuffle, timeout, gpu="-1"):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from tpch_user.runner import worker_env
    cmd = [sys.executable, os.path.join(HERE, "worker.py"), engine, spec_path, data_dir, out] + (["--shuffle"] if shuffle else [])
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=worker_env(gpu), cwd=ROOT, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": "TIMEOUT"}
    lines = [l for l in p.stdout.splitlines() if l.startswith("@@RESULT ")]
    if not lines:
        return {"error": (p.stderr or p.stdout).strip().splitlines()[-1][:300] if (p.stderr or p.stdout).strip() else f"exit {p.returncode}"}
    return json.loads(lines[-1][9:])


def _norm(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return float(v)
    return v


def _rows(path, cols):
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    if not all(c in t.column_names for c in cols):
        raise ValueError(f"columns {t.column_names}, expected {cols}")
    d = t.select(cols).to_pydict()
    return [tuple(_norm(d[c][i]) for c in cols) for i in range(t.num_rows)]


def _close(a, b, rel):
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return abs(a - b) <= rel * max(1.0, abs(a), abs(b))
    return a == b


def _sortkey(r):
    return tuple((x is None, str(type(x)), x if not isinstance(x, float) else round(x, 6)) for x in r)


def _rows_close(r, g, rel):
    return all(_close(a, b, rel) for a, b in zip(r, g))


def _same_multiset(ref, got, rel):
    """Rows equal as multisets, floats within rel (rows matched after sorting both)."""
    return len(ref) == len(got) and all(_rows_close(a, b, rel) for a, b in zip(sorted(ref, key=_sortkey),
                                                                            sorted(got, key=_sortkey)))


def out_of_order(a, b, pos, desc):
    """True when row b follows row a against the ORDER BY. The first sort column on which the rows
    are not exactly equal decides: a float pair within TIE of each other is a near-tie (two engines
    sum in a different order and differ in the last bits), so either order is accepted and later
    columns are NOT consulted. Checking them (an earlier version skipped near-equal values and went
    on) demanded the key order inside runs of unequal values: with 30,000 groups the averages are
    ~1e-6 apart, and seed 7129 was reported WRONG for rows that equal DuckDB's bit for bit."""
    for p, d in zip(pos, desc):
        x, y = a[p], b[p]
        if x == y:
            continue
        if x is None or y is None:                 # DuckDB's default: NULLS LAST in both directions
            return x is None
        if _close(x, y, TIE):
            return False
        return (x < y) if d else (x > y)
    return False


def compare(spec, ref_path, got_path):
    """DuckDB's rows against an engine's. Unordered queries: equal as multisets. Ordered ones: the
    engine's rows follow the ORDER BY, and equal DuckDB's as multisets, except that rows tied with the
    LIMIT's last row (float sort values within tolerance) may be any of the tied ones: two groups
    whose averages differ in the 7th digit can come out in either order."""
    cols = (spec.get("select") or ((spec["group"] + [a[2] for a in spec["aggs"]]) if not spec["distinct"]
                                   else [spec["distinct"][0], "n"]))
    try:
        ref, got = _rows(ref_path, cols), _rows(got_path, cols)
    except Exception as e:
        return "WRONG", str(e)[:200]
    if len(ref) != len(got):
        return "WRONG", f"rows {len(got)} vs {len(ref)}"
    unordered = False
    for rel, verdict in ((1e-3, "ok"), (1e-2, "drift")):
        if not spec["order"]:
            if _same_multiset(ref, got, rel):
                return verdict, ""
            continue
        pos = [cols.index(c) for c, _ in spec["order"]]
        desc = [d for _, d in spec["order"]]

        def before(a, b):          # a sorts strictly before b; float sort values within 1e-6 tie
            for p, d in zip(pos, desc):
                x, y = a[p], b[p]
                if _close(x, y, TIE):
                    continue
                return (x > y) if d else (x < y)
            return False

        if any(out_of_order(got[i], got[i + 1], pos, desc) for i in range(len(got) - 1)):
            unordered = True
            continue                                       # not in ORDER BY order
        if _same_multiset(ref, got, rel):
            return verdict, ""
        if spec["limit"] and len(ref) == spec["limit"]:
            last = ref[-1]
            core = lambda rows: [x for x in rows if before(x, last)]
            nf = max([i for i, p in enumerate(pos) if isinstance(last[p], float)], default=-1)

            def at_cut(x):     # ties with the cutoff on its sort columns up to the last float one
                return all(_close(x[p], last[p], TIE) for p in pos[:nf + 1]) if nf >= 0 else \
                    not before(x, last) and not before(last, x)

            tied = all(at_cut(x) for x in got if not before(x, last))
            core = lambda rows: [x for x in rows if before(x, last) and not at_cut(x)]
            if tied and _same_multiset(core(ref), core(got), rel):   # differ only among rows tied with the cutoff
                return verdict, ""
    if unordered:
        return "WRONG", "not in ORDER BY order"
    bad = next((i for i, (r, g) in enumerate(zip(sorted(ref, key=_sortkey), sorted(got, key=_sortkey)))
                if not _rows_close(r, g, 1e-2)), 0)
    return "WRONG", f"first differing row (sorted) {bad}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", default="1")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "bench"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="cpu", help="device for omxcore's compute() (F-055: gpu)")
    ap.add_argument("--gpu", default="-1", help="CUDA_VISIBLE_DEVICES for the workers (a UUID or index)")
    ap.add_argument("--engines", default=",".join(ENGINES))
    a = ap.parse_args()
    data_dir = os.path.join(ROOT, "data", f"tpch_sf{a.sf}")
    st = gen.stats(os.path.join(ROOT, "data", "tpch_sf1"))
    name = a.name or f"random-sf{a.sf}{'-shuffled' if a.shuffle else ''}"
    os.environ["MXBENCH_DEVICE"] = a.device
    engines = sorted(a.engines.split(","), key=lambda e: e != "duckdb")     # the reference first
    tmp = tempfile.mkdtemp(prefix="mxrand-", dir=_scratch.root())
    rows = []
    for seed in range(a.seed0, a.seed0 + a.n):
        spec = gen.make(seed, st)
        sp = os.path.join(tmp, f"q{seed}.json")
        json.dump(spec, open(sp, "w"))
        rec = {"seed": seed, "shape": spec["shape"], "sql": gen.sql(spec),
               **{e: {"error": "not run"} for e in ENGINES if e not in engines}}
        for e in engines:
            out = os.path.join(tmp, f"q{seed}-{e}.parquet")
            r = _run(e, sp, data_dir, out, a.shuffle, a.timeout, a.gpu if e == "mx" else "-1")
            rec[e] = r
            if e != "duckdb" and "error" not in r and "error" not in rec["duckdb"]:
                rec[e]["verdict"], rec[e]["why"] = compare(spec, os.path.join(tmp, f"q{seed}-duckdb.parquet"), out)
        rows.append(rec)
        m, p, d = (rec[e].get("ms") for e in ("mx", "polars", "duckdb"))
        print(f"q{seed:<4} {spec['shape']:9s} mx {m or 0:8.1f}  polars {p or 0:8.1f}  duckdb {d or 0:8.1f}  "
              f"{rec['mx'].get('verdict', rec['mx'].get('error', ''))}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
    report(rows, os.path.join(a.out, name + ".md"), a)


def _gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def report(rows, path, a):
    ok = lambda r, e: "error" not in r[e] and r[e].get("verdict", "ok") in ("ok", "drift")
    L = [f"# Random queries on TPC-H SF{a.sf}{' (rows shuffled)' if a.shuffle else ''}: omxcore vs Polars vs DuckDB\n",
         f"{len(rows)} seeded random queries (seeds {a.seed0}..{a.seed0 + len(rows) - 1}, `scripts/bench_random/`), "
         "engine-only: each (query, engine) in a fresh process, columns loaded untimed, the first run timed "
         "(query + materialized result). DuckDB's result is the reference.\n"]
    by = defaultdict(list)
    for r in rows:
        by[r["shape"]].append(r)
    L.append("| shape | queries | omxcore correct | faster than Polars | faster than DuckDB | Polars / omxcore (geo. mean) | DuckDB / omxcore (geo. mean) | omxcore: no native decline logged |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    tot = defaultdict(int)
    for shape in sorted(by) + ["all"]:
        rs = rows if shape == "all" else by[shape]
        good = [r for r in rs if ok(r, "mx")]
        vp = [r for r in good if ok(r, "polars")]
        vd = [r for r in good if "error" not in r["duckdb"]]
        wp = sum(r["mx"]["ms"] < r["polars"]["ms"] for r in vp)
        wd = sum(r["mx"]["ms"] < r["duckdb"]["ms"] for r in vd)
        gp = _gm([r["polars"]["ms"] / r["mx"]["ms"] for r in vp])
        gd = _gm([r["duckdb"]["ms"] / r["mx"]["ms"] for r in vd])
        nat = sum(1 for r in good if not r["mx"].get("declines"))
        L.append(f"| {shape} | {len(rs)} | {len(good)}/{len(rs)} | {wp}/{len(vp)} | {wd}/{len(vd)} | {gp:.2f}x | {gd:.2f}x | {nat}/{len(good)} |")
    tm = {e: sum(r[e].get("ms", 0) for r in rows if all("error" not in r[x] for x in ENGINES)) for e in ENGINES}
    L.append(f"\nTotal time over queries every engine ran: omxcore {tm['mx']:.0f} ms, Polars {tm['polars']:.0f} ms, "
             f"DuckDB {tm['duckdb']:.0f} ms. Geo. mean ratio > 1: omxcore faster.\n")
    bad = [r for r in rows if not ok(r, "mx")]
    L.append(f"## omxcore results that differ from DuckDB or failed ({len(bad)})\n")
    for r in bad:
        L.append(f"- q{r['seed']} ({r['shape']}): {r['mx'].get('verdict', r['mx'].get('error'))} {r['mx'].get('why', '')}  \n  `{r['sql']}`")
    pb = [r for r in rows if not ok(r, "polars")]
    L.append(f"\n## Polars results that differ from DuckDB or failed ({len(pb)})\n")
    for r in pb:
        L.append(f"- q{r['seed']}: {r['polars'].get('verdict', r['polars'].get('error'))} {r['polars'].get('why', '')}")
    slow = sorted([r for r in rows if ok(r, "mx") and "error" not in r["duckdb"]],
                  key=lambda r: r["mx"]["ms"] / max(min(r["duckdb"]["ms"], r["polars"].get("ms", 1e9)), 1e-3), reverse=True)[:12]
    L.append("\n## Where omxcore is slowest relative to the faster of Polars / DuckDB\n")
    L.append("| query | shape | omxcore ms | Polars ms | DuckDB ms | fell back to Python for |")
    L.append("|---|---|---:|---:|---:|---|")
    for r in slow:
        L.append(f"| q{r['seed']} | {r['shape']} | {r['mx']['ms']:.1f} | {r['polars'].get('ms', float('nan')):.1f} | "
                 f"{r['duckdb']['ms']:.1f} | {'; '.join(r['mx'].get('declines', []))[:160]} |")
    reasons = defaultdict(int)
    for r in rows:
        for d in r["mx"].get("declines", []):
            reasons[d] += 1
    L.append("\n## Why parts of omxcore queries ran in Python (count of queries)\n")
    for d, c in sorted(reasons.items(), key=lambda x: -x[1]):
        L.append(f"- {c}: {d}")
    L.append("\n## Queries\n")
    for r in rows:
        L.append(f"- q{r['seed']} ({r['shape']}): `{r['sql']}`")
    open(path, "w").write("\n".join(L) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
