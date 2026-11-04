#!/usr/bin/env python3
"""User-felt TPC-H scoreboard (ticket F-001).

What a user feels: a fresh Python process, data starting as Parquet on disk, the clock
around (Parquet read + query + materialized result).  No warm-up, ever.  One subprocess
per (engine, query, scale factor).  `import` time is reported separately; a repeat run in
the same process is a secondary number.  Every result is checked against DuckDB.

`--mode engine` (F-024) is the engine-only scoreboard: each engine first loads the columns the
query uses into its own in-memory form (outside the clock, recorded as load_ms) and the clock
covers query + materialized result only.  Since 2026-10-13 this is the headline measure (loading
data is staging); user-felt is the secondary number.

    pixi run python scripts/bench_user_felt.py --sf 1,10 --queries 1-22 \
        --engines pandas,polars,duckdb,mx-cpu,mx-gpu,mx-auto --timeout 300 --out docs/bench

Every measurement is appended to <out>/<name>.csv (and flushed) the moment it exists, so a crash
or Ctrl-C loses nothing; the markdown is rendered from the CSV at the end.  `--resume` skips the
(engine, sf, query) rows already in the CSV; `--report-only` just re-renders the markdown.

Environment: the subprocesses use CUDA_VISIBLE_DEVICES=<--gpu> (default 1) and import
omxcore from THIS checkout.  Data: $MXFRAME_TPCH_DATA or <repo>/data or the main checkout's
data/ (a worktree has none); files are <data>/tpch_sf<SF>_exact/<table>.parquet (exact DECIMALs; --legacy-data uses tpch_sf<SF>).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scratch as _scratch  # noqa: E402
from tpch_user import spec  # noqa: E402
from tpch_user.compare import compare_tables  # noqa: E402
from tpch_user.official import check_official  # noqa: E402
from tpch_user.report import (CSV_FIELDS, ENGINE_ORDER, environment_info, mode_of, new_row, read_csv,  # noqa: E402
                              render_markdown)
from tpch_user.runner import (REPO_ROOT, check_mxframe_path, resolve_gpu, run_one,  # noqa: E402
                              warm_page_cache)

ALL_ENGINES = ["pandas", "polars", "duckdb", "mx-cpu", "mx-gpu", "mx-auto"]


def parse_queries(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    bad = [q for q in out if q not in spec.OUTPUT]
    if bad:
        raise SystemExit(f"unknown queries: {bad}")
    return out


def find_data_root(cli: str | None) -> Path:
    cands = [cli, os.environ.get("MXFRAME_TPCH_DATA"), str(REPO_ROOT / "data")]
    try:
        common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        if common:
            cands.append(str(Path(common).parent / "data"))
    except OSError:
        pass
    for c in cands:
        if c and Path(c).is_dir():
            return Path(c)
    raise SystemExit("TPC-H data not found; pass --data-root (expects <root>/tpch_sf<SF>/*.parquet)")


EMPTY_SUM_IS_ZERO = ("pandas", "polars")      # engines that give 0, not NULL, over no rows


def compare_result(path, ref_path, q: int, engine: str = ""):
    """Compare a result Parquet with the reference; returns (verdict, correct, max_rel_err, note)."""
    import pyarrow.parquet as pq
    try:
        cmp = compare_tables(pq.read_table(path), pq.read_table(ref_path),
                             topn_metric=spec.TOPN_METRIC.get(q), order_by=spec.ORDER_BY.get(q),
                             empty_sum_is_zero=engine in EMPTY_SUM_IS_ZERO)
    except Exception as e:  # noqa: BLE001 - a broken comparison must not lose the measurement
        return "unchecked", None, None, f"comparison failed: {type(e).__name__}: {str(e)[:150]}"
    return cmp.verdict, cmp.ok, cmp.max_rel_err, cmp.message


def official_check(path, q: int) -> str | None:
    import pyarrow.parquet as pq
    try:
        return check_official(q, pq.read_table(path))
    except Exception as e:  # noqa: BLE001
        return f"check failed: {type(e).__name__}: {str(e)[:100]}"


def open_log(csv_path: Path, resume: bool, mode: str = "user-felt"):
    """(file, writer, rows already present). Without --resume the CSV starts empty. Resuming keeps
    the CSV's own header (pre-F-024 CSVs lack `mode` and `load_ms`) and refuses another mode's rows."""
    existing = read_csv(csv_path) if resume and csv_path.exists() else []
    fields = CSV_FIELDS
    if existing:
        if mode_of(existing) != mode:
            raise SystemExit(f"{csv_path} holds {mode_of(existing)} rows; cannot resume it in {mode} mode "
                             "(use another --name)")
        with open(csv_path, newline="") as f:
            fields = next(csv.reader(f))
    f = open(csv_path, "a" if existing else "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    if not existing:
        w.writeheader()
        f.flush()
    return f, w, existing


def write_report(csv_path: Path, md_path: Path, a, root, gpu_desc: str, elapsed: float,
                 queries: list[int], sfs: list[str]) -> None:
    rows = read_csv(csv_path)
    if not rows:
        return
    engines = [e for e in ENGINE_ORDER if any(r["engine"] == e for r in rows)]
    qs = sorted({r["query"] for r in rows} | set(queries))
    sf_all = sorted({r["sf"] for r in rows} | set(sfs), key=float)
    meta = environment_info(a, root, elapsed, gpu_desc)
    meta['data_variant'] = 'legacy (truncated float32 DECIMALs)' if a.legacy_data else 'exact'
    md_path.write_text(render_markdown(rows, meta, engines, qs, sf_all))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sf", default="1", help="scale factors, e.g. 1,10")
    ap.add_argument("--queries", default="1-22")
    ap.add_argument("--engines", default=",".join(ALL_ENGINES))
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per subprocess")
    ap.add_argument("--out", default="docs/bench")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--legacy-data", action="store_true",
                    help="use <root>/tpch_sf<SF> (float32 DECIMALs truncated: 0.05 -> 0.049999997) instead of "
                         "the default <root>/tpch_sf<SF>_exact")
    ap.add_argument("--gpu", default="1", help="nvidia-smi index (or GPU-UUID) of the GPU the workers may use")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--name", default=None,
                    help="output basename (default user-felt-<date>, or engine-only-<date> with --mode engine)")
    ap.add_argument("--mode", choices=["user-felt", "engine"], default="user-felt",
                    help="user-felt (secondary): clock around Parquet read + query; engine (headline): each engine loads "
                         "the query's columns into memory first, the clock covers query + result only")
    ap.add_argument("--shuffle", action="store_true",
                    help="engine mode: put each loaded table's rows in a random order (outside the clock); --report-only on such a CSV needs --mode engine --shuffle too, to keep the label")
    ap.add_argument("--resume", action="store_true",
                    help="keep the existing CSV and skip (engine, sf, query) rows already in it")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing non-empty CSV of the same name (default: refuse)")
    ap.add_argument("--report-only", action="store_true", help="re-render the markdown from the CSV and exit")
    a = ap.parse_args()
    if a.shuffle:
        if a.mode != "engine":
            ap.error("--shuffle needs --mode engine (the rows are shuffled after the untimed load)")
        os.environ["MXBENCH_SHUFFLE"] = "1"          # read by every worker (scripts/tpch_user/worker.py)

    engines = [e.strip() for e in a.engines.split(",") if e.strip()]
    unknown = [e for e in engines if e not in ALL_ENGINES]
    if unknown:
        raise SystemExit(f"unknown engines: {unknown}; choose from {ALL_ENGINES}")
    queries = parse_queries(a.queries)
    sfs = [s.strip() for s in a.sf.split(",") if s.strip()]
    root = find_data_root(a.data_root)
    out_dir = Path(a.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base = a.name or (f"engine-only{'-shuffled' if a.shuffle else ''}-{a.date}" if a.mode == "engine"
                      else f"user-felt-{a.date}")
    csv_path, md_path = out_dir / f"{base}.csv", out_dir / f"{base}.md"
    cuda_dev, gpu_desc = resolve_gpu(a.gpu)

    if a.report_only:
        write_report(csv_path, md_path, a, root, gpu_desc, 0.0, queries, sfs)
        print(f"wrote {md_path}")
        return

    if csv_path.exists() and csv_path.stat().st_size > 0 and not (a.resume or a.force):
        raise SystemExit(f"{csv_path} already has results; pass --resume to add to it, --force to overwrite, "
                         "or use another --name")
    # fail BEFORE the loop: data present, and workers import omxcore from THIS checkout
    suffix = "" if a.legacy_data else "_exact"
    data_dirs = {sf: root / f"tpch_sf{sf}{suffix}" for sf in sfs}
    missing = [str(d) for d in data_dirs.values() if not d.is_dir()]
    if missing:
        raise SystemExit(f"TPC-H data not found: {missing}")
    if any(e.startswith("mx-") for e in engines):
        mxp = check_mxframe_path()
        if os.path.realpath(mxp) != os.path.realpath(REPO_ROOT):
            raise SystemExit(f"workers import omxcore from {mxp}, not {REPO_ROOT}")

    # the DuckDB reference always runs first (it is only *reported* when requested)
    order = ["duckdb"] + [e for e in engines if e != "duckdb"]
    tmp = Path(tempfile.mkdtemp(prefix="bench-user-felt-", dir=_scratch.root()))
    f, writer, existing = open_log(csv_path, a.resume, a.mode)
    done = {(r["engine"], r["sf"], r["query"]) for r in existing}
    t_start = time.time()
    try:
        for sf in sfs:
            data_dir = data_dirs[sf]
            tables = sorted({t for q in queries for t in spec.TABLES[q]})
            warm_page_cache(str(data_dir), tables)
            for q in queries:
                todo = [e for e in engines if (e, sf, q) not in done]
                if not todo:
                    continue
                ref_path = tmp / f"ref_sf{sf}_q{q}.parquet"
                ref_ok = False
                for eng in order:
                    if eng == "duckdb":
                        if not any(e != "duckdb" for e in todo) and "duckdb" not in todo:
                            continue
                    elif eng not in todo:
                        continue
                    res_path = ref_path if eng == "duckdb" else tmp / f"{eng}_sf{sf}_q{q}.parquet"
                    rep_path = Path(str(res_path) + ".repeat")
                    load1 = os.getloadavg()[0]
                    rec = run_one(eng, q, str(data_dir), str(res_path), a.timeout, cuda_dev, a.mode)
                    rec.update(engine=eng, sf=sf, query=q, correct=None, max_rel_err=None,
                               load1=load1, verdict="ref" if eng == "duckdb" else "unchecked",
                               repeat_verdict="")
                    mxp = rec.pop("mxframe_path", None)
                    if mxp and os.path.realpath(mxp) != os.path.realpath(REPO_ROOT):
                        rec["error"] = f"imported omxcore from {mxp}, not {REPO_ROOT}"
                    if eng == "duckdb":
                        ref_ok = "error" not in rec and res_path.exists()
                        if ref_ok and sf == "1":
                            msg = official_check(res_path, q)
                            if msg:
                                rec["compare_note"] = f"official TPC-H SF1 answer: {msg}"
                        if rec.get("repeat_ms") is not None and ref_ok and rep_path.exists():
                            rec["repeat_verdict"] = compare_result(rep_path, ref_path, q, eng)[0]
                    elif "error" in rec:
                        rec["verdict"] = "error"
                    elif ref_ok and res_path.exists():
                        rec["verdict"], rec["correct"], rec["max_rel_err"], note = \
                            compare_result(res_path, ref_path, q, eng)
                        if note:
                            rec["compare_note"] = note
                        if rec.get("repeat_ms") is not None and rep_path.exists():
                            rec["repeat_verdict"] = compare_result(rep_path, ref_path, q, eng)[0]
                    elif res_path.exists():
                        rec["compare_note"] = "not verified: the DuckDB reference failed for this query"
                    if eng in todo:
                        writer.writerow(new_row(rec))
                        f.flush()
                        os.fsync(f.fileno())
                    for p in (res_path, rep_path):
                        if eng != "duckdb" and p.exists():
                            p.unlink()
                    status = rec.get("error") or f"first={rec.get('first_ms', 0):.0f}ms {rec['verdict']}"
                    print(f"SF{sf} Q{q:<2} {eng:<8} {status}", flush=True)
                for p in (ref_path, Path(str(ref_path) + ".repeat")):
                    if p.exists():
                        p.unlink()
    finally:
        f.close()
        shutil.rmtree(tmp, ignore_errors=True)
        write_report(csv_path, md_path, a, root, gpu_desc, time.time() - t_start, queries, sfs)
        print(f"\nwrote {md_path}\nwrote {csv_path}")


if __name__ == "__main__":
    main()
