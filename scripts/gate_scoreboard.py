"""F-162 / F-058: the quiet-night scoreboard of the M8 gate. omxcore's `mx sql` (cpu / gpu / auto) against DuckDB on TPC-H (22 queries as
DuckDB writes them) and the random suite, SF1 and SF10, over typed Parquet (scripts/gate_common.py; generated on first use into
~/.cache/omxcore-gate). Writes docs/bench/<date>-gate.md (+ .json) with a per-query ratio and the 90% gate verdict.

    pixi run python scripts/gate_scoreboard.py                       # the night: SF 1 and 10, 22 queries + 50 random, cpu/gpu/auto, 3 fresh runs each
    pixi run python scripts/gate_scoreboard.py --smoke               # proof it works: SF1, Q6, 1 run, cpu; writes ~/.cache/omxcore-gate/gate-smoke.md
    pixi run python scripts/gate_scoreboard.py --sf 1 --devices cpu --repeats 5 --random 100 --queries 1,3,6

How a time is taken (the repo's benchmark rules, docs/bench/f058-report.md): ENGINE-ONLY FIRST RUN. Every (engine, query, run) is a fresh
process; the tables the query names are loaded first, outside the clock (omxcore: `mx sql` reads the Parquet files before the query and
reports the query's own `ms`; DuckDB: `CREATE TABLE ... AS SELECT * FROM read_parquet(...)` and `import pyarrow` before the clock);
no warm-up; the clock covers the query and the materialized result (omxcore's Table; DuckDB's `fetch_arrow_table()`). The reported time
is the median of the fresh runs. Polars is not in this scoreboard: it runs DataFrame code, not SQL text (the old runs: docs/bench/f058-*).

Correctness is checked after all the timing (so DuckDB's checking work never overlaps a timed run): the first run's result of every
omxcore query is compared with DuckDB's (tests/gate_compare.py: exact for INT / DECIMAL / DATE / TEXT, FLOAT to 1e-9, in ORDER BY order).
A query that fails or differs counts as a loss, and the report lists it with the reason.

The gate (docs/plan.md F-058): a device beats DuckDB on at least 90% of the queries (strictly faster, median, and equal to DuckDB).
The report also counts "within 10%" (omxcore time <= 1.1 x DuckDB's), which the owner called a close miss, and the geometric mean.
Run it on a quiet machine: it prints the load average before every run and the report gives the median and maximum.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tests"))
import gate_common as gate  # noqa: E402

GATE_FRACTION = 0.9
CLOSE = 1.1
DEVICES = ["cpu", "gpu", "auto"]


# ── the DuckDB worker: one query in a fresh process ────────────────────────────────────────────────

def duck_worker(sql_file: str, tables: list, paths_json: str):
    import duckdb                                              # noqa: E402
    import pyarrow                                             # noqa: F401  (staging: DuckDB imports it inside its first .arrow(), ~70 ms)
    sql = Path(sql_file).read_text()
    paths = json.loads(paths_json)
    con = duckdb.connect()
    for t in tables:
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_parquet('{paths[t]}')")
    t0 = time.perf_counter()
    out = con.execute(sql).fetch_arrow_table()
    ms = (time.perf_counter() - t0) * 1e3
    print("@@RESULT " + json.dumps({"ms": ms, "rows": out.num_rows}), flush=True)


# ── one timed run ──────────────────────────────────────────────────────────────────────────────────

def run_duck(sql: str, tables: list, paths: dict, workdir: Path, timeout: float) -> dict:
    f = workdir / "q.sql"
    f.write_text(sql)
    cmd = [sys.executable, __file__, "--duck-worker", str(f), ",".join(tables), json.dumps({t: paths[t] for t in tables})]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("@@RESULT ")]
    if not lines:
        return {"error": ((p.stderr or p.stdout).strip().splitlines() or [f"exit {p.returncode}"])[-1][:300]}
    return json.loads(lines[-1][9:])


def run_mx_once(mx: str, sql: str, tables: list, paths: dict, device: str, gpu: str, timeout: float) -> dict:
    cmd = [mx, "sql", "--max-rows", "0", "--device", device, sql]
    for t in tables:
        cmd += ["-t", f"{t}={paths[t]}"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu if device != "cpu" else "-1")
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    try:
        res = json.loads(p.stdout.decode("utf-8"), parse_float=__import__("decimal").Decimal)
    except ValueError:
        return {"error": f"exit {p.returncode}: {p.stderr.decode()[-200:] or p.stdout[-200:]!r}"}
    if p.returncode != 0 or "error" in res:
        return {"error": res.get("error", {}).get("message", f"exit {p.returncode}")}
    device_note = next((n for n in res["notes"] if n.startswith("device:")), "")
    return {"ms": float(res["ms"]), "rows": res["row_count"], "device_note": device_note, "json": res}


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


# ── the scoreboard ─────────────────────────────────────────────────────────────────────────────────

def measure(args, mx: str):
    """{(sf, name): {"sql", "tables", "duckdb" | device: {"ms": [...], "rows", "device_note"} or {"error"}, "first": {device: mx JSON}}}"""
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="gate-", dir=gate.GATE_DATA))
    tpch = gate.queries()
    stats = gate.random_stats(1) if args.random else None
    loads = []
    out = {}
    for sf in args.sf:
        paths = gate.table_paths(sf)
        work = [(f"Q{q}", tpch[q]) for q in args.queries] + [(f"r{s}", gate.random_sql(s, stats)) for s in range(args.random)]
        for name, sql in work:
            from gate_compare import tables_named
            tables = tables_named(sql)
            rec = {"sql": sql, "tables": tables, "first": {}}
            out[(sf, name)] = rec
            for dev in ["duckdb"] + args.devices:
                for i in range(args.repeats):
                    load = os.getloadavg()[0]
                    loads.append(load)
                    r = run_duck(sql, tables, paths, tmp, args.timeout) if dev == "duckdb" else run_mx_once(mx, sql, tables, paths, dev, args.gpu, args.timeout)
                    if dev != "duckdb" and i == 0:
                        rec["first"][dev] = r.pop("json", None)
                    r.pop("json", None)
                    if "error" in r:
                        rec[dev] = {"error": r["error"]}
                        print(f"SF{sf} {name:5s} {dev:6s} run {i + 1}: ERROR {r['error'][:100]}  (load {load:.1f})", flush=True)
                        break
                    rec.setdefault(dev, {"ms": [], "device_note": r.get("device_note", ""), "rows": r["rows"]})["ms"].append(r["ms"])
                    print(f"SF{sf} {name:5s} {dev:6s} run {i + 1}: {r['ms']:10.2f} ms  (load {load:.1f})", flush=True)
    return out, loads


def check(args, out):
    """Phase 2: every omxcore first-run result against DuckDB. Sets rec[device]["verdict"] = "equal" or why not."""
    from gate_compare import compare_json_with_duckdb
    for sf in args.sf:
        con = gate.duck_views(gate.table_paths(sf))
        for (s, name), rec in out.items():
            if s != sf:
                continue
            for dev in args.devices:
                r = rec.get(dev)
                if r is None or "error" in r:
                    continue
                try:
                    compare_json_with_duckdb(rec["first"][dev], rec["sql"], con, label=f"SF{sf} {name} {dev}")
                    r["verdict"] = "equal"
                except AssertionError as e:
                    r["verdict"] = "DIFFERS: " + str(e)[:300]
                print(f"check SF{sf} {name:5s} {dev:6s} {r['verdict'][:80]}", flush=True)


def ratio(rec, dev):
    """DuckDB median / omxcore median (> 1: omxcore faster), or None when omxcore failed, differs, or DuckDB failed."""
    r, d = rec.get(dev), rec.get("duckdb")
    if not r or "error" in r or r.get("verdict") != "equal" or not d or "error" in d:
        return None
    return median(d["ms"]) / median(r["ms"])


def report(args, out, loads, path: Path, started: str):
    L = [f"# M8 gate scoreboard: {datetime.date.today().isoformat()}", "",
         f"omxcore `mx sql` ({', '.join(args.devices)}) against DuckDB {__import__('duckdb').__version__}, engine-only first runs: a fresh process per (engine, query, run), the tables "
         f"loaded outside the clock, no warm-up, the median of {args.repeats} run(s). Typed TPC-H Parquet (DuckDB dbgen: DECIMAL, DATE), queries as DuckDB's "
         f"tpch extension writes them. Started {started}; load average before each run: median {median(loads):.1f}, max {max(loads):.1f}"
         f"{' (NOT QUIET: re-run on an idle machine)' if max(loads) > 4 else ''}. Command: `{' '.join(sys.argv)}`.", ""]
    notes = {r[dev]["device_note"] for r in out.values() for dev in args.devices if dev in r and "device_note" in r[dev]}
    if notes:
        L += ["Device as the CLI reported it: " + "; ".join(f"`{n}`" for n in sorted(notes)) +
              ". Where it says the SQL runs on the CPU, the `gpu` and `auto` columns time the same CPU engine.", ""]
    verdicts = []
    for sf in args.sf:
        for group, names in (("TPC-H", [n for (s, n) in out if s == sf and n.startswith("Q")]),
                             ("random", [n for (s, n) in out if s == sf and n.startswith("r")])):
            if not names:
                continue
            L += [f"## {group} SF{sf}", "",
                  "| query | DuckDB ms | " + " | ".join(f"omxcore {d} ms | ratio {d}" for d in args.devices) + " |",
                  "|---|---:|" + "---:|---:|" * len(args.devices)]
            for n in names:
                rec = out[(sf, n)]
                d = rec.get("duckdb", {})
                row = [n, f"{median(d['ms']):.1f}" if "ms" in d else f"ERROR {d.get('error', '?')[:40]}"]
                for dev in args.devices:
                    r = rec.get(dev, {})
                    if "ms" not in r:
                        row += [f"ERROR: {r.get('error', 'not run')[:70]}", "-"]
                    elif r.get("verdict") != "equal":
                        row += [f"{median(r['ms']):.1f}", f"WRONG: {r.get('verdict', '?')[:60]}"]
                    else:
                        row += [f"{median(r['ms']):.1f}", f"{ratio(rec, dev):.2f}x"]
                L.append("| " + " | ".join(row) + " |")
            L.append("")
            L += ["| device | queries | equal DuckDB | faster than DuckDB | within 10% | geo. mean DuckDB / omxcore | 90% gate |", "|---|---:|---:|---:|---:|---:|---|"]
            for dev in args.devices:
                rs = [(n, ratio(out[(sf, n)], dev)) for n in names]
                equal = sum(1 for n in names if out[(sf, n)].get(dev, {}).get("verdict") == "equal")
                wins = sum(1 for _, r in rs if r is not None and r > 1)
                close = sum(1 for _, r in rs if r is not None and r >= 1 / CLOSE)
                ok = wins >= GATE_FRACTION * len(names)
                verdicts.append((f"{group} SF{sf} {dev}", ok, wins, len(names)))
                L.append(f"| {dev} | {len(names)} | {equal} | {wins} | {close} | {gmean([r for _, r in rs]):.2f}x | "
                         f"{'PASS' if ok else 'NOT MET'} ({wins}/{len(names)}, needs {math.ceil(GATE_FRACTION * len(names))}) |")
            L.append("")
    bad = [(sf, n, dev, out[(sf, n)][dev]) for (sf, n) in out for dev in args.devices
           if dev in out[(sf, n)] and ("error" in out[(sf, n)][dev] or out[(sf, n)][dev].get("verdict") != "equal")]
    L += [f"## omxcore failures and differences ({len(bad)})", ""]
    for sf, n, dev, r in bad:
        L.append(f"- SF{sf} {n} {dev}: {r.get('error') or r.get('verdict')}")
    L += ["", "## Verdict", ""]
    for what, ok, wins, total in verdicts:
        L.append(f"- {what}: {'PASS' if ok else 'NOT MET'} ({wins}/{total} faster than DuckDB)")
    L.append("")
    L.append(f"**The 90% gate: {'MET' if verdicts and all(v[1] for v in verdicts) else 'NOT MET'}**" + (" (a --smoke run proves the harness; it says nothing about the gate)" if args.smoke else ""))
    L += ["", "## Queries run", ""]
    for (sf, n), rec in out.items():
        if sf == args.sf[0]:
            L.append(f"- {n}: `{' '.join(rec['sql'].split())[:400]}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
    slim = {f"SF{sf}/{n}": {k: v for k, v in rec.items() if k != "first"} for (sf, n), rec in out.items()}
    path.with_suffix(".json").write_text(json.dumps(slim, indent=1, default=str))
    print("wrote", path)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--duck-worker":
        duck_worker(sys.argv[2], sys.argv[3].split(","), sys.argv[4])
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sf", default="1,10", help="scale factors, comma separated (default 1,10)")
    ap.add_argument("--queries", default=",".join(map(str, range(1, 23))), help="TPC-H query numbers (default all 22)")
    ap.add_argument("--random", type=int, default=50, help="how many random-suite seeds (default 50; 0 for none)")
    ap.add_argument("--devices", default=",".join(DEVICES), help="omxcore devices (default cpu,gpu,auto)")
    ap.add_argument("--repeats", type=int, default=3, help="fresh-process runs per (engine, query); the median is reported")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for the gpu / auto runs (default GPU 0)")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--mx", default=None, help="the mx binary (default: tests/mx_oracle.mx_binary(), built when stale)")
    ap.add_argument("--out", default=None, help="report path (default docs/bench/<date>-gate.md)")
    ap.add_argument("--smoke", action="store_true", help="1 query (Q6), SF1, cpu, 1 run: prove the harness works (report under ~/.cache/omxcore-gate)")
    a = ap.parse_args()
    if a.smoke:
        a.sf, a.queries, a.random, a.devices, a.repeats = "1", "6", 0, "cpu", 1
    a.sf = [s.strip() for s in a.sf.split(",")]
    a.queries = [int(q) for q in a.queries.split(",") if q]
    a.devices = [d for d in a.devices.split(",") if d]
    bad = [d for d in a.devices if d not in DEVICES]
    if bad:
        ap.error(f"unknown device {bad}")
    gate.GATE_DATA.mkdir(parents=True, exist_ok=True)
    mx = a.mx
    if mx is None:
        from mx_oracle import mx_binary
        mx = str(mx_binary())
    path = Path(a.out) if a.out else (gate.GATE_DATA / "gate-smoke.md" if a.smoke else ROOT / "docs" / "bench" / f"{datetime.date.today().isoformat()}-gate.md")
    started = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"load average {os.getloadavg()[0]:.1f}; mx = {mx}", flush=True)
    out, loads = measure(a, mx)
    check(a, out)
    report(a, out, loads, path, started)


if __name__ == "__main__":
    main()
