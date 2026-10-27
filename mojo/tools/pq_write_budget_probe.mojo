# F-188: how much memory a save adds. Usage: pq_write_budget_probe <rows> <path> <threads> <budget bytes>
# Builds a table of eight incompressible INT columns in one row group (rows * 64 bytes), resets the process's peak resident memory
# (/proc/self/clear_refs 5), saves it with `threads` threads and a `budget`, and prints one JSON line: the table's size and how far the peak
# rose above what the process held before the save (tests/test_mojo_parquet.py).
from std.sys import argv
from std.time import perf_counter_ns
from omxcore import Table, INT, write_parquet, ZSTD


def status_kb(key: String) raises -> Int:
    with open("/proc/self/status", "r") as f:
        for line in f.read().split("\n"):
            if line.startswith(key):
                var parts = line.split()
                return Int(String(parts[1]))
    return -1


def main() raises:
    var args = argv()
    var rows = Int(String(args[1]))
    var path = String(args[2])
    var threads = Int(String(args[3]))
    var budget = Int(String(args[4]))
    var t = Table.empty()
    for c in range(8):
        _ = t.add_column("c" + String(c), INT)
    var s = UInt64(88172645463325252)
    for _ in range(rows):
        for c in range(8):
            s ^= s << 13
            s ^= s >> 7
            s ^= s << 17
            t.append(c, Int64(s >> 1))
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")
    var before = status_kb("VmRSS:")
    var t0 = perf_counter_ns()
    write_parquet(path, t, "", ZSTD, rows, threads=threads, budget=budget)
    var ms = Float64(perf_counter_ns() - t0) / 1e6
    print('{"table_kb": ' + String(rows * 64 // 1024) + ', "before_kb": ' + String(before) + ', "peak_kb": ' + String(status_kb("VmHWM:")) + ', "save_ms": ' + String(ms) + "}")
