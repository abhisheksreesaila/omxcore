# F-167 test tool (tests/test_sort_spill_memory.py): runs one sorting SQL string over a Parquet or CSV file streamed with a scan, in a fresh
# process, and prints one JSON line with the peak resident memory (VmHWM: file pages included) and what the sort did.
#
#   sort_spill_probe <table|scan|file> <in.parquet|in.csv> "<SQL>" <sort_budget_bytes|0> [out.parquet] [spill_dir] [row_group_rows] [threads]
#
#   table: the file read whole into a table; scan: streamed, the result is a table; file: streamed, the sorted rows go to out.parquet (sql_to_parquet).
# {"rows": N, "checksum": X, "runs": R, "vm_hwm_kb": .., "rss_anon_kb": .., "query_ms": .., "streamed": true|false, "spill_files_left": F}
# checksum: an order-sensitive hash of the first column of the result (a table result: its rows; a file: read back with the engine's reader).
from std.sys import argv
from std.time import perf_counter_ns
from std.os import listdir
from omxcore import (Catalog, PlannerOptions, sql, sql_to_parquet, read_parquet, scan_parquet, scan_csv, read_csv, CsvOptions, ZSTD, Table)


def status_kb(key: String) raises -> Int:
    var text = String()
    with open("/proc/self/status", "r") as fh:
        text = fh.read()
    for line in text.split("\n"):
        if line.startswith(key):
            var digits = String()
            for b in line.as_bytes():
                if b >= 48 and b <= 57:
                    digits += chr(Int(b))
            return Int(digits)
    return -1


def checksum(t: Table) raises -> Int:
    var h = 0
    for r in range(t.rows()):
        h = (h * 1000003 + Int(t.get(0, r)) + 7) % 1000000000000000007
    return h


def main() raises:
    var args = argv()
    var mode = String(args[1])
    var path = String(args[2])
    var text = String(args[3])
    var popts = PlannerOptions()
    popts.sort_budget = Int(String(args[4]))
    var out = String(args[5]) if len(args) > 5 else String("")
    var dir = String(args[6]) if len(args) > 6 else String("")
    popts.spill_dir = dir
    var group = Int(String(args[7])) if len(args) > 7 else (1 << 20)
    if len(args) > 8:
        popts.threads = Int(String(args[8]))
    var cat = Catalog()
    var csv = path.endswith(".csv")
    if mode == "table":
        if csv:
            var rc = read_csv(path)
            cat.add_table("t", rc.take_table().share())
        else:
            var rp = read_parquet(path)
            cat.add_table("t", rp.take_table().share())
    else:
        if csv:
            cat.add_scan("t", scan_csv(path, CsvOptions(group_rows=100000)))
        else:
            cat.add_scan("t", scan_parquet(path))
    var base_kb = status_kb("VmHWM:")
    var t0 = perf_counter_ns()
    var rows = 0
    var sum = 0
    var runs = 0
    var streamed = False
    if mode == "file":
        var r = sql_to_parquet(cat, text, out, popts, "", ZSTD, group)
        rows = r.rows_written
        for n in r.notes:
            if n.startswith("output:"):
                streamed = True
            var at = n.find(" runs spilled")
            if n.startswith("order: sorted") and at > 0:
                var digits = String()
                var j = at - 1
                while j >= 0 and n.as_bytes()[j] >= 48 and n.as_bytes()[j] <= 57:
                    digits = chr(Int(n.as_bytes()[j])) + digits
                    j -= 1
                runs = Int(digits)
    else:
        var r = sql(cat, text, popts)
        rows = r.table.rows()
        sum = checksum(r.table)
        for n in r.notes:
            var at = n.find(" runs spilled")
            if n.startswith("order: sorted") and at > 0:
                var digits = String()
                var j = at - 1
                while j >= 0 and n.as_bytes()[j] >= 48 and n.as_bytes()[j] <= 57:
                    digits = chr(Int(n.as_bytes()[j])) + digits
                    j -= 1
                runs = Int(digits)
    var t1 = perf_counter_ns()
    var left = 0
    if dir.byte_length() > 0:
        left = len(listdir(dir))
    print('{"rows": ' + String(rows) + ', "checksum": ' + String(sum) + ', "runs": ' + String(runs) + ', "vm_hwm_kb": ' + String(status_kb("VmHWM:")) +
          ', "base_hwm_kb": ' + String(base_kb) + ', "rss_anon_kb": ' + String(status_kb("RssAnon:")) + ', "query_ms": ' + String(Float64(t1 - t0) / 1e6) +
          ', "streamed": ' + ("true" if streamed else "false") + ', "spill_files_left": ' + String(left) + "}")
