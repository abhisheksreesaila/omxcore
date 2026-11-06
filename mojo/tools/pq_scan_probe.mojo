# F-140: runs one named query over a Parquet file either streamed (`scan`: Query(scan_parquet(path))) or after reading
# the whole file (`read`: read_parquet + Query), and prints one JSON line: the result's slots, counts and `present`, the
# region column's words (code order), the time and the process's peak resident memory (VmHWM).
# Usage: pq_scan_probe <path> <scan|read> <query>     (tests/test_mojo_parquet.py, scripts/bench_scan_memory.py)
# The file has: region (string), carrier (int64), day (date), revenue (decimal(.., 2)), cost (int64).
from std.sys import argv
from std.os import getenv
from omxcore.procmem import peak_kb
from omxcore import (Query, QueryResult, col, lit, least, sum, count, mean, min_of, max_of, read_parquet, scan_parquet,
                     CPU, GPU, AUTO, MORSELS_AUTO, MORSELS_ON, MORSELS_OFF)



def _morsel_mode() -> Int:
    """The scan's morsel switch from the environment (this tool only: the library reads no environment)."""
    var m = getenv("MXFRAME_SCAN_MORSELS")
    return MORSELS_ON if m == "1" else MORSELS_OFF if m == "0" else MORSELS_AUTO


def _device() -> Int:
    """The scan's device from the environment (this tool only): MXFRAME_SCAN_DEVICE=gpu or cpu, else AUTO."""
    var d = getenv("MXFRAME_SCAN_DEVICE")
    return GPU if d == "gpu" else CPU if d == "cpu" else AUTO


def _block_rows() raises -> Int:
    var n = getenv("MXFRAME_MORSEL_ROWS")
    return Int(n) if n.byte_length() > 0 else 0


def plan(var q: Query, which: String, nreg: Int, east: Int) raises -> Query:
    if which == "by_region":
        return q^.group_by(col("region"), groups=nreg).agg(sum(col("revenue")), count(), count(col("revenue")),
                                                            min_of(col("cost")), max_of(col("cost")), mean(col("revenue")),
                                                            min_of(col("revenue")), max_of(col("revenue")))
    if which == "filtered":
        return q^.where(col("region") == lit(east)).where(col("day").between(10_000, 14_000)) \
            .group_by(col("carrier"), groups=8).agg(sum(col("revenue")), count(), max_of(col("cost")))
    if which == "pivot":
        return q^.where(col("carrier").between(0, 7)).group_by(col("region") * 8 + col("carrier"), groups=nreg * 8).agg(sum(col("cost")), count(col("revenue")))
    if which == "bins":
        return q^.group_by(least(col("day") // 30, lit(99)), groups=100).agg(count(), sum(col("cost")))
    if which == "global":
        return q^.agg(sum(col("cost")), count(), min_of(col("revenue")), max_of(col("revenue")), mean(col("cost")))
    if which == "sum_v":                   # files with no region column (unique-id text + ints)
        return q^.agg(sum(col("v")), count())
    if which == "big":                     # the memory proof: two columns of the file
        return q^.group_by(col("region"), groups=nreg).agg(sum(col("cost")), count())
    raise Error("unknown query " + which)



def main() raises:
    var args = argv()
    var path = String(args[1])
    var mode = String(args[2])
    var which = String(args[3])
    var r: QueryResult
    var words = String("")
    if mode == "scan":
        var scan = scan_parquet(path)
        var c = scan.index("region")
        var nreg = 1
        var east = 0
        if c >= 0:
            nreg = scan.word_count(c)
            east = max(scan.code_of(c, "East"), 0)
            for w in range(nreg):
                words += scan.text_of(c, w) + "\x1f"
        r = plan(Query(scan), which, nreg, east).morsels(_morsel_mode()).block_rows(_block_rows()).compute(_device())
    else:
        var rp = read_parquet(path)
        var t = rp.take_table().share()
        var c = t.index("region")
        var nreg = 1
        var east = 0
        if c >= 0:
            nreg = len(t.dictionary(c))
            east = max(t.code_of(c, "East"), 0)
            for w in range(nreg):
                words += t.text_of(c, w) + "\x1f"
        r = plan(Query(t), which, nreg, east).compute(AUTO)
    var out = String('{"words": "') + words.replace("\x1f", "|") + '", "device": ' + String(r.device) + ', "reason": "' + r.fallback_reason.replace('"', "'") + '", "ms": ' + String(r.milliseconds)
    out += ', "peak_kb": ' + String(peak_kb()) + ', "kinds": ['
    for i in range(len(r.kinds)):
        out += ("" if i == 0 else ", ") + String(r.kinds[i])
    out += '], "slot_of": ['
    for i in range(len(r.kinds)):
        out += ("" if i == 0 else ", ") + String(r.slot_of[i])
    out += '], "count_slot_of": ['
    for i in range(len(r.kinds)):
        out += ("" if i == 0 else ", ") + String(r.count_slot_of[i])
    out += '], "counts": ['
    for g in range(len(r.counts)):
        out += ("" if g == 0 else ", ") + String(r.counts[g])
    out += '], "slots": ['
    for s in range(len(r.slots)):
        out += ("" if s == 0 else ", ") + "["
        for g in range(len(r.slots[s])):
            out += ("" if g == 0 else ", ") + String(r.slots[s][g])
        out += "]"
    out += '], "present": ['
    for i in range(len(r.kinds)):
        out += ("" if i == 0 else ", ") + "["
        for g in range(len(r.counts)):
            out += ("" if g == 0 else ", ") + ("1" if r.present(i, g) else "0")
        out += "]"
    print(out + "]}")
