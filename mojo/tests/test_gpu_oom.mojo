# F-189: data bigger than GPU memory. `MXFRAME_GPU_MEM_BUDGET` fakes a small GPU (docs/omxcore-mojo-api.md, GPU section), so
# a few MB of table stand in for a table bigger than the card. The oracle is the CPU answer of the same query. Skips
# cleanly with no GPU. Every test that sets the variable puts it back (`Budget` restores it when it goes out of scope).
from omxcore import (Table, SharedTable, Frame, Query, QueryResult, col, sum, count, min_of, max_of, INT, write_parquet,
                     scan_parquet, ZSTD, CPU, GPU, AUTO, DeviceTable, gpu_available)
from omxcore.gpu import gpu_budget_override, parse_size, GPU_BUDGET_ENV
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


struct Budget(Movable):
    """Sets the fake GPU size for as long as it lives and puts the variable back as it was."""
    var had: Bool
    var saved: String

    def __init__(out self, text: String):
        self.had = getenv(GPU_BUDGET_ENV, "").byte_length() > 0
        self.saved = getenv(GPU_BUDGET_ENV, "")
        _ = setenv(GPU_BUDGET_ENV, text)

    def __del__(deinit self):
        if self.had:
            _ = setenv(GPU_BUDGET_ENV, self.saved)
        else:
            _ = unsetenv(GPU_BUDGET_ENV)


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
    var saved = getenv(GPU_BUDGET_ENV, "")
    _ = unsetenv(GPU_BUDGET_ENV)
    expect(gpu_budget_override(0) == -1, "unset: no override")
    _ = setenv(GPU_BUDGET_ENV, "2M")
    expect(gpu_budget_override(0) == 2 * 1024 * 1024 and gpu_budget_override(1) == 2 * 1024 * 1024, "every GPU")
    _ = setenv(GPU_BUDGET_ENV, "0:512M,1:2G")
    expect(gpu_budget_override(0) == 512 * 1024 * 1024 and gpu_budget_override(1) == 2048 * 1024 * 1024 and gpu_budget_override(2) == -1,
           "per GPU")
    if saved.byte_length() > 0:
        _ = setenv(GPU_BUDGET_ENV, saved)
    else:
        _ = unsetenv(GPU_BUDGET_ENV)


def test_today_a_device_table_over_the_budget_is_uploaded_whole() raises:
    """Before F-189: nothing reads the budget. The 7.2 MB table goes up whole to a GPU that was said to hold 1 MB."""
    var budget = Budget("1M")
    var t = make_table(ROWS, 1)
    var dt = DeviceTable(t)
    expect(dt.bytes() >= 3 * 8 * ROWS, "uploaded whole: " + String(dt.bytes()))
    var g = grouped(Query(t).on_device(dt)).compute(GPU)
    expect(g.device == GPU and g.fallback_reason == "", "compute(GPU) ran the whole table in one go")
    same(g, grouped(Query(t)).compute(CPU), "GPU over budget")
    dt.release()
    _ = budget^


def test_today_auto_ignores_the_budget() raises:
    var budget = Budget("1M")
    var t = make_table(1_100_000, 2)
    var dt = DeviceTable(t)
    var r = grouped(Query(t).on_device(dt)).compute(AUTO)
    expect(r.device == GPU and r.fallback_reason == "", "AUTO took the GPU for a table 20x the budget")
    dt.release()
    _ = budget^


def test_today_a_gpu_scan_goes_a_row_group_at_a_time() raises:
    """A streamed file is already cut into row groups, so a file over the budget runs; a group over the budget is not checked."""
    var budget = Budget("1M")
    var t = make_table(ROWS, 3)
    write_parquet("f189_scan.parquet", t, "", ZSTD, 10_000)           # 30 groups of 240 KB
    var scan = scan_parquet("f189_scan.parquet")
    var g = grouped(Query(scan)).compute(GPU)
    expect(g.device == GPU and g.fallback_reason == "", "the scan ran on the GPU")
    same(g, grouped(Query(t)).compute(CPU), "GPU scan over budget")
    write_parquet("f189_scan_big.parquet", t, "", ZSTD, 150_000)      # 2 groups of 3.6 MB: each over the budget
    var big = grouped(Query(scan_parquet("f189_scan_big.parquet"))).compute(GPU)
    expect(big.device == GPU, "a row group over the budget was uploaded whole")
    _ = budget^


def main() raises:
    test_the_budget_is_read_from_the_environment()
    if not gpu_available():
        print("no GPU: skipped")
        return
    test_today_a_device_table_over_the_budget_is_uploaded_whole()
    test_today_auto_ignores_the_budget()
    test_today_a_gpu_scan_goes_a_row_group_at_a_time()
    print("all gpu oom tests passed")
