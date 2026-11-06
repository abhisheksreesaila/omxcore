# F-147: edits a Parquet file through a Frame over a scan, queries it and saves it, and prints one JSON line with the
# time of each phase and the process's peak resident memory (VmHWM) after it. The edits are a fixed script, so an
# outside oracle can replay them (tests/test_mojo_parquet.py, scripts/bench_scan_memory.py --edit):
#   cost of row (k * 1000003) % rows := k * 3 - 1500           for k in 0..999
#   region of row (k * 7919 + 5) % rows := "Fresh" (a word the file lacks)   for k in 0..99
#   rows (k * 999983 + 17) % rows removed                      for k in 0..199
#   50 rows appended: region "Appended", carrier 3, day 2020-01-01, revenue 1.50, cost 77
# Usage: pq_edit_probe <path> <out path> <cache MiB> [permute] [allow]     (the file is the one pq_scan_probe reads)
from std.sys import argv
from std.time import perf_counter_ns
from omxcore.procmem import peak_kb
from omxcore import Frame, Query, col, sum, count, scan_parquet, AUTO



def ms_since(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e6


def main() raises:
    var args = argv()
    var path = String(args[1])
    var out = String(args[2])
    var cache = Int(String(args[3])) * 1024 * 1024
    var permute = False
    var allow = False                                        # "allow": save even if a column was narrowed on read
    for i in range(4, len(args)):
        permute = permute or String(args[i]) == "permute"
        allow = allow or String(args[i]) == "allow"
    var t0 = perf_counter_ns()
    var scan = scan_parquet(path)
    var f = Frame(scan)
    f.set_cache_limit(cache)
    var rows = f.rows()
    var cost = f.column("cost")
    var region = f.column("region")
    var json = String('{"rows": ') + String(rows) + ', "row_groups": ' + String(scan.row_groups())
    var opened_kb = peak_kb()
    t0 = perf_counter_ns()
    for k in range(1000):
        f.set(cost, (k * 1000003) % rows, Int64(k * 3 - 1500))
    for k in range(100):
        f.set_text(region, (k * 7919 + 5) % rows, "Fresh")
    for k in range(200):
        f.remove((k * 999983 + 17) % rows)
    for _ in range(50):
        _ = f.append(["Appended", "3", "2020-01-01", "1.50", "77"])
    json += ', "edit_ms": ' + String(ms_since(t0)) + ', "edit_peak_kb": ' + String(peak_kb())
    t0 = perf_counter_ns()
    var n = f.word_count(region)
    var r = Query(f).group_by(col("region"), groups=n).agg(sum(col("cost")), count()).compute(AUTO)
    json += ', "query_ms": ' + String(ms_since(t0)) + ', "query_peak_kb": ' + String(peak_kb()) + ', "words": ['
    for g in range(n):
        json += ("" if g == 0 else ", ") + '"' + f.text_of(region, Int64(g)) + '"'
    json += '], "sums": ['
    for g in range(n):
        json += ("" if g == 0 else ", ") + String(r.slots[r.slot_of[0]][g])
    json += '], "counts": ['
    for g in range(n):
        json += ("" if g == 0 else ", ") + String(r.counts[g])
    json += ']'
    t0 = perf_counter_ns()
    var seen: Int64 = 0                                      # a grid scrolling the file: a row every 997, every group
    var at = 0
    while at < rows:
        if f.alive(at) and not f.is_null(cost, at):
            seen += f.get(cost, at)
        at += 997
    json += ', "get_ms": ' + String(ms_since(t0)) + ', "get_sum": ' + String(seen) + ', "get_peak_kb": ' + String(peak_kb())
    json += ', "cache_loads": ' + String(f.cache_loads()) + ', "cache_held": ' + String(f.cache_held())
    if permute:
        t0 = perf_counter_ns()
        var order = f.permutation(cost, False)
        json += ', "permute_ms": ' + String(ms_since(t0)) + ', "permute_len": ' + String(len(order)) + ', "permute_first": ' + String(order[0]) + ', "permute_peak_kb": ' + String(peak_kb())
    t0 = perf_counter_ns()
    f.save(out, "{\"edited\": true}", allow_narrowing=allow)
    json += ', "save_ms": ' + String(ms_since(t0)) + ', "save_peak_kb": ' + String(peak_kb()) + ', "opened_peak_kb": ' + String(opened_kb)
    print(json + "}")
