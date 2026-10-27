# bench_codec_write <rows> <path> [threads]: saves a realistic orders table with each codec and reads it back, printing bytes and best-of-five
# seconds (F-180); threads (F-187): the most threads that compress, 0 or none given: all the build has, 1: one. Build it normally for libzstd / libsnappy, or with -D MXFRAME_NO_CODECS for the Mojo codecs, and compare.
from std.sys import argv
from std.time import perf_counter_ns
from omxcore import Table, INT, DECIMAL, TEXT, DATE, BOOL, write_parquet, read_parquet, UNCOMPRESSED, SNAPPY, ZSTD


def orders(n: Int) raises -> Table:
    var t = Table.empty()
    var id = t.add_column("order_id", INT)
    var customer = t.add_column("customer", INT)
    var qty = t.add_column("qty", INT)
    var price = t.add_column("price", DECIMAL, scale=2)
    var day = t.add_column("day", DATE)
    var status = t.add_column("status", TEXT)
    var region = t.add_column("region", TEXT)
    var paid = t.add_column("paid", BOOL)
    var statuses: List[String] = ["new", "paid", "shipped", "delivered", "returned"]
    for w in statuses:
        _ = t.add_word(status, w)
    for r in range(50):
        _ = t.add_word(region, "region-" + String(r))
    var s = UInt64(88172645463325252)
    for i in range(n):
        s ^= s << 13
        s ^= s >> 7
        s ^= s << 17
        t.append(id, Int64(1_000_000 + i))
        t.append(customer, Int64(s % 100_000))
        t.append(qty, Int64(1 + (s >> 20) % 12))
        if (s >> 32) % 97 == 0:
            t.append_null(price)
        else:
            t.append(price, Int64(500 + (s >> 33) % 20_000))
        t.append(day, Int64(19_000 + (i // 3000) % 365))
        t.append(status, Int64(((s >> 40) % 10) // 2))
        t.append(region, Int64((s >> 44) % 50))
        t.append(paid, Int64((s >> 50) % 4 != 0))
    return t^


def main() raises:
    var args = argv()
    var n = Int(String(args[1]))
    var path = String(args[2])
    var threads = Int(String(args[3])) if len(args) > 3 else 0
    var t = orders(n)
    var plain = 0
    for which in range(3):
        var codec = UNCOMPRESSED if which == 0 else (SNAPPY if which == 1 else ZSTD)
        var name = "none" if which == 0 else ("snappy" if which == 1 else "zstd")
        var best_w = 1 << 62
        var best_r = 1 << 62
        for _ in range(5):
            var a = perf_counter_ns()
            write_parquet(path, t, "", codec, threads=threads)
            var b = perf_counter_ns()
            var back = read_parquet(path)
            var c = perf_counter_ns()
            if back.table.rows() != n:
                raise Error("read back " + String(back.table.rows()) + " rows")
            best_w = min(best_w, Int(b - a))
            best_r = min(best_r, Int(c - b))
        var size = len(open(path, "r").read_bytes())
        if which == 0:
            plain = size
        print(name, "bytes", size, "ratio-vs-none", Float64(plain) / Float64(size), "write s", Float64(best_w) / 1e9, "read s", Float64(best_r) / 1e9)
