# F-189: data bigger than GPU memory never fails. `MXFRAME_GPU_MEM_BUDGET` fakes a small GPU (docs/omxcore-mojo-api.md, GPU
# section), so a few MB of table stand in for a table bigger than the card, and `MXFRAME_GPU_FAIL_PIECE=n` makes the nth
# piece's device allocation fail as an out-of-memory would. The oracle is the CPU answer of the same query (every slot, count
# and presence flag equal, not close). Skips cleanly with no GPU. Every test that sets a variable puts it back (`Env`
# restores it when it goes out of scope, a failed test included). Modest allocations: GPU 0 is shared.
from omxcore import (Table, SharedTable, Frame, Query, QueryResult, col, sum, count, min_of, max_of, INT, write_parquet,
                     scan_parquet, scan_csv, CsvOptions, ZSTD, CPU, GPU, AUTO, DeviceTable, gpu_available)
from omxcore.gpu import gpu_budget_override, parse_size, GPU_BUDGET_ENV, GPU_FAIL_ENV
from std.os import getenv, setenv, unsetenv


struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed * 0x9E3779B97F4A7C15 + 1

    def next(mut self) -> UInt64:
        self.s ^= self.s << 13
        self.s ^= self.s >> 7
        self.s ^= self.s << 17
        return self.s

    def int(mut self, lo: Int, hi: Int) -> Int:       # [lo, hi]
        return lo + Int(self.next() % UInt64(hi - lo + 1))


struct Env(Movable):
    """Sets an environment variable for as long as it lives and puts it back as it was."""
    var name: String
    var had: Bool
    var saved: String

    def __init__(out self, name: String, value: String):
        self.name = name
        self.had = getenv(name, "").byte_length() > 0
        self.saved = getenv(name, "")
        _ = setenv(name, value)

    def set(mut self, value: String):
        _ = setenv(self.name, value)

    def clear(mut self):
        _ = unsetenv(self.name)

    def __del__(deinit self):
        if self.had:
            _ = setenv(self.name, self.saved)
        else:
            _ = unsetenv(self.name)


def budget(text: String) -> Env:
    return Env(GPU_BUDGET_ENV, text)


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


def same(a: QueryResult, b: QueryResult, label: String) raises:
    """Every slot, count and presence flag equal (the GPU is not allowed to be merely close)."""
    expect(len(a.slots) == len(b.slots), label + ": same number of slots")
    for s in range(len(a.slots)):
        for g in range(len(a.counts)):
            expect(a.slots[s][g] == b.slots[s][g], label + ": slot " + String(s) + " group " + String(g) + ": " +
                   String(a.slots[s][g]) + " vs cpu " + String(b.slots[s][g]))
    for g in range(len(a.counts)):
        expect(a.counts[g] == b.counts[g], label + ": count group " + String(g))
    for i in range(len(a.kinds)):
        for g in range(len(a.counts)):
            expect(a.present(i, g) == b.present(i, g), label + ": present " + String(i) + "/" + String(g))
    expect(len(a.sums) == len(b.sums), label + ": sums")
    for g in range(len(a.sums)):
        expect(a.sums[g] == b.sums[g], label + ": sum of group " + String(g))


def make_table(n: Int, seed: UInt64) raises -> SharedTable:
    """k INT 0..7 (every 29th null), v INT (every 13th null, some past int32), w INT -5000..5000: 3 columns, 24 bytes a row
    on the device plus the bitmaps."""
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var v = t.add_column("v", INT)
    var w = t.add_column("w", INT)
    var rng = Rng(seed)
    for i in range(n):
        if i % 29 == 3:
            t.append_null(k)
        else:
            t.append(k, Int64(rng.int(0, 7)))
        if i % 13 == 1:
            t.append_null(v)
        else:
            t.append(v, Int64(rng.int(-3_000_000_000, 3_000_000_000)))
        t.append(w, Int64(rng.int(-5000, 5000)))
    return t^.share()


comptime ROWS = 300_000          # 7.2 MB of columns: over a 1 MB fake GPU several times


def grouped(var q: Query) raises -> Query:
    return q^.where(col("w") > -4000).group_by(col("k"), groups=8).agg(sum(col("v")), count(), min_of(col("w")), max_of(col("w")))


def test_the_budget_is_read_from_the_environment() raises:
    expect(parse_size("512M") == 512 * 1024 * 1024 and parse_size("64KB") == 64 * 1024 and parse_size("1.5G") == 1536 * 1024 * 1024
           and parse_size("4096") == 4096, "sizes")
    var none = Env(GPU_BUDGET_ENV, "")
    none.clear()
    expect(gpu_budget_override(0) == -1, "unset: no override")
    none.set("2M")
    expect(gpu_budget_override(0) == 2 * 1024 * 1024 and gpu_budget_override(1) == 2 * 1024 * 1024, "every GPU")
    none.set("0:512M,1:2G")
    expect(gpu_budget_override(0) == 512 * 1024 * 1024 and gpu_budget_override(1) == 2048 * 1024 * 1024 and gpu_budget_override(2) == -1,
           "per GPU")
    _ = none^                                                 # (a value is destroyed at its last use: this keeps the variable set to here)


def test_a_table_over_the_budget_runs_in_pieces() raises:
    """compute(GPU) on a table 7x the GPU: row ranges are uploaded in turn, each fits, the merged answer is the CPU's."""
    var b = budget("1M")
    var t = make_table(ROWS, 1)
    var dt = DeviceTable(t)
    expect(not dt.is_resident() and dt.bytes() == 0 and dt.needed_bytes() >= 3 * 8 * ROWS, "no device copy: " + String(dt.bytes()))
    var g = grouped(Query(t).on_device(dt)).compute(GPU)
    expect(g.device == GPU and g.fallback_reason == "", "ran on the GPU: `" + g.fallback_reason + "`")
    expect(g.pieces >= 8 and g.gpu_pieces == g.pieces, "pieces that fit 0.9 MB: " + String(g.pieces) + " (" + String(g.gpu_pieces) + " on the GPU)")
    same(g, grouped(Query(t)).compute(CPU), "pieces")
    same(Query(t).on_device(dt).agg(count()).compute(GPU), Query(t).agg(count()).compute(CPU), "count() reads no column")
    same(Query(t).on_device(dt).where(col("v") > 0).agg(sum(col("v")), min_of(col("w"))).compute(GPU),
         Query(t).where(col("v") > 0).agg(sum(col("v")), min_of(col("w"))).compute(CPU), "no group by")
    dt.release()
    _ = b^


def test_a_table_under_the_budget_stays_resident_in_one_launch() raises:
    var b = budget("64M")
    var t = make_table(ROWS, 2)
    var dt = DeviceTable(t)
    expect(dt.is_resident() and dt.bytes() >= 3 * 8 * ROWS, "uploaded")
    var g = grouped(Query(t).on_device(dt)).compute(GPU)
    expect(g.device == GPU and g.pieces == 1 and g.gpu_pieces == 1, "one launch")
    same(g, grouped(Query(t)).compute(CPU), "resident")
    dt.release()
    _ = b^


def test_auto_answers_a_too_big_table_on_the_cpu_and_says_how_big() raises:
    var t = make_table(1_100_000, 3)
    var r = Env(GPU_BUDGET_ENV, "64M")
    var fits = DeviceTable(t)
    var small = grouped(Query(t).on_device(fits)).compute(AUTO)
    expect(small.device == GPU and small.fallback_reason == "", "a table that fits: AUTO takes the GPU")
    fits.release()
    r.set("2M")
    var dt = DeviceTable(t)
    var a = grouped(Query(t).on_device(dt)).compute(AUTO)
    expect(a.device == CPU and a.pieces == 0, "AUTO: the CPU")
    expect("budget of" in a.fallback_reason and "MB on the GPU" in a.fallback_reason and "1100000 rows" in a.fallback_reason,
           "and why, with the sizes: `" + a.fallback_reason + "`")
    same(a, grouped(Query(t)).compute(CPU), "AUTO over budget")
    var g = grouped(Query(t).on_device(dt)).compute(GPU)
    expect(g.device == GPU and g.pieces > 10, "compute(GPU) runs the same table in pieces: " + String(g.pieces))
    same(g, a, "GPU pieces vs AUTO")
    dt.release()
    _ = r^


def test_a_frames_edits_ride_along_with_every_piece() raises:
    var b = budget("1M")
    var t = make_table(ROWS, 4)
    var dt = DeviceTable(t)
    var f = Frame(t)
    var rng = Rng(7)
    for _ in range(3000):
        f.set(f.column("w"), rng.int(0, ROWS - 1), Int64(rng.int(-9000, 9000)))
    for _ in range(500):
        f.set_null(f.column("v"), rng.int(0, ROWS - 1))
    for _ in range(800):
        var row = rng.int(0, ROWS - 1)
        if f.alive(row):
            f.remove(row)
    for _ in range(20):
        _ = f.append(["3", "123456", "77"])
    var g = grouped(Query(f).on_device(dt)).compute(GPU)
    expect(g.device == GPU and g.pieces >= 8, "pieces: " + String(g.pieces))
    same(g, grouped(Query(f)).compute(CPU), "frame over pieces")
    dt.release()
    _ = b^


def test_the_gpu_scan_cuts_a_group_over_the_budget_into_ranges() raises:
    var b = budget("1M")
    var t = make_table(ROWS, 5)
    var want = grouped(Query(t)).compute(CPU)
    write_parquet("f189_small_groups.parquet", t, "", ZSTD, 10_000)                # 30 groups of 240 KB: each fits
    var fits = grouped(Query(scan_parquet("f189_small_groups.parquet"))).compute(GPU)
    expect(fits.device == GPU and fits.pieces == 30 and fits.gpu_pieces == 30, "a group at a time: " + String(fits.pieces))
    same(fits, want, "groups that fit")
    write_parquet("f189_big_groups.parquet", t, "", ZSTD, 150_000)                  # 2 groups of 3.6 MB: each over the budget
    var big = scan_parquet("f189_big_groups.parquet")
    var g = grouped(Query(big)).compute(GPU)
    expect(g.device == GPU and g.fallback_reason == "", "GPU: each group in row ranges, on the GPU: `" + g.fallback_reason + "`")
    same(g, want, "groups over the budget")
    var a = grouped(Query(big).auto_gpu_ops(0)).compute(AUTO)
    expect(a.device == CPU and "row group of 150000 rows needs" in a.fallback_reason and "budget of" in a.fallback_reason,
           "AUTO: the CPU, naming the size: `" + a.fallback_reason + "`")
    same(a, want, "AUTO scan over budget")
    _ = b^


def test_a_csv_scan_over_the_budget() raises:
    var b = budget("1M")
    var t = make_table(60_000, 6)
    var text = String("k,v,w\n")
    for i in range(t.rows()):
        for c in range(3):
            if c > 0:
                text += ","
            if not t.is_null(c, i):
                text += String(t.get(c, i))
        text += "\n"
    var f = open("f189_scan.csv", "w")
    f.write(text)
    f.close()
    var scan = scan_csv("f189_scan.csv", CsvOptions(group_rows=40_000))             # groups of 960 KB and 480 KB: the first is over the budget (Apple's plan keeps 254 KB of it for buffers)
    var g = grouped(Query(scan)).compute(GPU)
    expect(g.device == GPU, "a CSV file's groups cut into ranges: `" + g.fallback_reason + "`")
    same(g, grouped(Query(t)).compute(CPU), "csv scan over budget")
    var auto = grouped(Query(scan).auto_gpu_ops(0)).compute(AUTO)
    expect(auto.device == CPU and "budget of" in auto.fallback_reason, "AUTO: the CPU: `" + auto.fallback_reason + "`")
    same(auto, g, "csv AUTO")
    _ = b^


def test_a_failed_piece_reruns_on_the_cpu_with_the_same_answer() raises:
    var b = budget("1M")
    var t = make_table(ROWS, 8)
    var dt = DeviceTable(t)
    var want = grouped(Query(t)).compute(CPU)
    var fail = Env(GPU_FAIL_ENV, "3")
    var g = grouped(Query(t).on_device(dt)).compute(GPU)
    expect(g.device == CPU and g.pieces >= 8 and g.gpu_pieces == g.pieces - 1, "one piece left the GPU: " + String(g.gpu_pieces) + " of " + String(g.pieces))
    expect("could not hold a piece of" in g.fallback_reason and "out of memory" in g.fallback_reason, "and says why: `" + g.fallback_reason + "`")
    same(g, want, "piece 3 on the CPU")
    fail.set("1")
    same(grouped(Query(t).on_device(dt)).compute(GPU), want, "the first piece on the CPU")
    b.set("64M")                                              # (a budget the table fits, so the upload is tried)
    fail.set("0")
    var up = DeviceTable(t)
    expect(not up.is_resident() and up.bytes() == 0 and up.oom_free() > 0, "an upload that fails is undone and remembered")
    fail.clear()
    var after = grouped(Query(t).on_device(up)).compute(GPU)
    expect(after.device == GPU and after.pieces == 1, "the handle still answers on the GPU, uploading for the query")
    same(after, want, "after a failed upload")
    up.release()
    dt.release()
    _ = fail^
    _ = b^


def test_a_failure_on_a_resident_table_falls_back_and_auto_remembers_it() raises:
    var t = make_table(1_100_000, 9)
    var b = budget("64M")
    var dt = DeviceTable(t)
    expect(dt.is_resident(), "resident under 64M")
    var want = grouped(Query(t)).compute(CPU)
    var fail = Env(GPU_FAIL_ENV, "1")
    var g = grouped(Query(t).on_device(dt)).compute(GPU)
    expect(g.device == CPU and "could not allocate" in g.fallback_reason, "GPU: the CPU answered: `" + g.fallback_reason + "`")
    same(g, want, "GPU after a failed allocation")
    fail.clear()
    expect(dt.oom_free() == 64 * 1024 * 1024, "the free memory at the failure is remembered: " + String(dt.oom_free()))
    var a = grouped(Query(t).on_device(dt)).compute(AUTO)
    expect(a.device == CPU and "earlier query ran out of GPU memory" in a.fallback_reason, "AUTO remembers: `" + a.fallback_reason + "`")
    same(a, want, "AUTO while remembered")
    b.set("128M")                                              # twice the free memory it failed at: worth another try
    var again = grouped(Query(t).on_device(dt)).compute(AUTO)
    expect(again.device == GPU and again.fallback_reason == "", "AUTO tries the GPU again: `" + again.fallback_reason + "`")
    same(again, want, "AUTO retried")
    expect(dt.oom_free() == 0, "and forgets")
    dt.release()
    _ = fail^
    _ = b^


def test_a_failed_group_runs_on_the_cpu_and_the_scan_remembers() raises:
    var b = budget("64M")
    var t = make_table(ROWS, 10)
    write_parquet("f189_fail.parquet", t, "", ZSTD, 20_000)                          # 15 groups
    var scan = scan_parquet("f189_fail.parquet")
    var want = grouped(Query(t)).compute(CPU)
    var fail = Env(GPU_FAIL_ENV, "4")
    var g = grouped(Query(scan)).compute(GPU)
    expect(g.device == CPU and g.pieces == 15 and g.gpu_pieces == 3, "groups 1-3 on the GPU, the rest after the failure on the CPU: " + String(g.gpu_pieces))
    expect("could not hold a row group" in g.fallback_reason, "`" + g.fallback_reason + "`")
    same(g, want, "scan with a failed group")
    fail.clear()
    var a = grouped(Query(scan).auto_gpu_ops(0)).compute(AUTO)
    expect(a.device == CPU and "earlier query ran out of GPU memory" in a.fallback_reason, "AUTO remembers on the file: `" + a.fallback_reason + "`")
    same(a, want, "AUTO scan while remembered")
    b.set("128M")
    var again = grouped(Query(scan).auto_gpu_ops(0)).compute(AUTO)
    expect(again.device == GPU and again.pieces == 15 and again.gpu_pieces == 15, "AUTO tries again: `" + again.fallback_reason + "`")
    same(again, want, "AUTO scan retried")
    _ = fail^
    _ = b^


def test_buffers_that_cannot_fit_say_so() raises:
    """3M groups need slots x 3M x 8 bytes of output: more than a 32 MB GPU has, however the rows are cut."""
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var v = t.add_column("v", INT)
    for i in range(1_100_000):
        t.append(k, Int64(i % 3_000_000))
        t.append(v, Int64(i % 7))
    var s = t^.share()
    var b = budget("32M")
    var dt = DeviceTable(s)
    expect(dt.is_resident(), "the 17.6 MB table fits: " + String(dt.needed_bytes()))
    var a = Query(s).on_device(dt).group_by(col("k"), groups=3_000_000).agg(sum(col("v")), count()).compute(AUTO)
    expect(a.device == CPU and "buffers need" in a.fallback_reason and "budget of" in a.fallback_reason, "AUTO: `" + a.fallback_reason + "`")
    var msg = String("")
    try:
        _ = Query(s).on_device(dt).group_by(col("k"), groups=3_000_000).agg(sum(col("v")), count()).compute(GPU)
    except e:
        msg = String(e)
    expect("buffers need" in msg and "budget of" in msg, "GPU raises a readable error: `" + msg + "`")
    dt.release()
    _ = b^


def test_a_partial_total_past_int64_is_settled_by_the_whole() raises:
    """+2^62 for 30000 rows, then -2^62 for 30000: the running total passes int64 half way, the whole sums to 0. Whatever the CPU
    answers (a total, or its overflow error) the pieces answer too."""
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var v = t.add_column("v", INT)
    for i in range(60_000):
        t.append(k, Int64(i % 2))
        t.append(v, Int64(1 << 62) if i < 30_000 else -Int64(1 << 62))
    var s = t^.share()
    var b = budget("1M")
    var dt = DeviceTable(s)
    expect(not dt.is_resident(), "60000 rows x 16 bytes (960 KB) do not fit 0.9 MB: " + String(dt.needed_bytes()))
    var cpu_msg = String("")
    var cpu = QueryResult()
    try:
        cpu = Query(s).agg(sum(col("v")), count()).compute(CPU)
    except e:
        cpu_msg = String(e)
    var gpu_msg = String("")
    var g = QueryResult()
    try:
        g = Query(s).on_device(dt).agg(sum(col("v")), count()).compute(GPU)
    except e:
        gpu_msg = String(e)
    expect(gpu_msg == cpu_msg, "same outcome: gpu `" + gpu_msg + "`, cpu `" + cpu_msg + "`")
    if cpu_msg == "":
        same(g, cpu, "int64 partials")
    dt.release()
    _ = b^


def main() raises:
    test_the_budget_is_read_from_the_environment()
    if not gpu_available():
        print("no GPU: skipped")
        return
    test_a_table_over_the_budget_runs_in_pieces()
    test_a_table_under_the_budget_stays_resident_in_one_launch()
    test_auto_answers_a_too_big_table_on_the_cpu_and_says_how_big()
    test_a_frames_edits_ride_along_with_every_piece()
    test_the_gpu_scan_cuts_a_group_over_the_budget_into_ranges()
    test_a_csv_scan_over_the_budget()
    test_a_failed_piece_reruns_on_the_cpu_with_the_same_answer()
    test_a_failure_on_a_resident_table_falls_back_and_auto_remembers_it()
    test_a_failed_group_runs_on_the_cpu_and_the_scan_remembers()
    test_buffers_that_cannot_fit_say_so()
    test_a_partial_total_past_int64_is_settled_by_the_whole()
    print("all gpu oom tests passed")
