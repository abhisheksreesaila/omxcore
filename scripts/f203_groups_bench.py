"""F-203: many-group GROUP BY timing, `mx sql` (cpu) against DuckDB in the same run, fresh process per run, median of --repeats (the gate scoreboard's rules:
scripts/gate_scoreboard.py run_mx_once / run_duck). Prints load, both medians, the ratio and mx's `finish:` note.

    pixi run python scripts/f203_groups_bench.py --sf 10 --repeats 3 [--mx build/mx]
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tests"))
import gate_common as gate  # noqa: E402
from gate_scoreboard import median, run_duck, run_mx_once  # noqa: E402

SHAPES = {
    "inner-q18": "select l_orderkey from lineitem group by l_orderkey having sum(l_quantity) > 300",
    "unique-sum": "select l_orderkey, sum(l_extendedprice) s from lineitem group by l_orderkey having sum(l_extendedprice) > 400000",
    "top-n": "select l_orderkey, sum(l_extendedprice) s from lineitem group by l_orderkey order by s desc, l_orderkey limit 10",
    "count-having": "select l_orderkey, count(*) c from lineitem group by l_orderkey having count(*) > 6",
    "two-keys": "select l_orderkey, l_linenumber, sum(l_quantity) from lineitem group by l_orderkey, l_linenumber having sum(l_quantity) > 49",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", default="10")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--mx", default=str(HERE.parent / "build/mx"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    paths = gate.table_paths(int(a.sf))
    tmp = Path(tempfile.mkdtemp(prefix="f203-", dir=gate.GATE_DATA))
    print(f"SF{a.sf}  mx={a.mx}  load at start {os.getloadavg()[0]:.1f}")
    for name, sql in SHAPES.items():
        if a.only and name not in a.only.split(","):
            continue
        mx_ms, duck_ms, note, rows = [], [], "", None
        for _ in range(a.repeats):
            d = run_duck(sql, ["lineitem"], paths, tmp, 900)
            m = run_mx_once(a.mx, sql, ["lineitem"], paths, "cpu", "0", 900)
            if "error" in d or "error" in m:
                print(f"  {name}: error {d.get('error')} {m.get('error')}")
                break
            duck_ms.append(d["ms"])
            mx_ms.append(m["ms"])
            rows = m["rows"]
            note = next((n for n in m["json"]["notes"] if n.startswith("finish:")), "")
        else:
            print(f"  {name:13s} mx {median(mx_ms):7.0f} ms  duckdb {median(duck_ms):7.0f} ms  ratio {median(mx_ms) / median(duck_ms):4.2f}  rows {rows}  load {os.getloadavg()[0]:.1f}\n      {note}")


main()
