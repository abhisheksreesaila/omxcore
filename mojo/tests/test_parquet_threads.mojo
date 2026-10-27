# F-187: the columns of a row group are compressed on several threads, and the file is the same bytes for any number of them.
# The oracle is the single-thread file (threads=1 never starts a worker): every other count, every codec, must write exactly its
# bytes, and read_parquet must give back the table that was saved. A column that cannot be written raises the same error for
# every count and leaves no file behind.
from std.pathlib import Path
from omxcore import (Table, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, write_parquet, read_parquet, UNCOMPRESSED, SNAPPY, ZSTD)


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


def table(n: Int, with_bad_decimal: Bool = False) raises -> Table:
    var t = Table.empty()
    var id = t.add_column("id", INT)
    var cust = t.add_column("cust", INT)
    var price = t.add_column("price", DECIMAL, scale=2)
    var day = t.add_column("day", DATE)
    var status = t.add_column("status", TEXT)
    var region = t.add_column("region", TEXT)
    var paid = t.add_column("paid", BOOL)
    var ratio = t.add_column("ratio", FLOAT)
    var words: List[String] = ["new", "paid", "shipped", "delivered", "returned"]
    for w in words:
        _ = t.add_word(status, w)
    for r in range(40):
        _ = t.add_word(region, "region-" + String(r))
    var s = UInt64(88172645463325252)
    for i in range(n):
        s ^= s << 13
        s ^= s >> 7
        s ^= s << 17
        t.append(id, Int64(1_000_000 + i))
        t.append(cust, Int64(s % 50_000))
        if (s >> 32) % 97 == 0:
            t.append_null(price)
        else:
            t.append(price, Int64(500 + (s >> 33) % 20_000))
        t.append(day, Int64(19_000 + (i // 300) % 365))
        t.append(status, Int64(((s >> 40) % 10) // 2))
        t.append(region, Int64((s >> 44) % 40))
        t.append(paid, Int64((s >> 50) % 4 != 0))
        t.append_float(ratio, Float64(s % 1000) / 7.0)
    if with_bad_decimal:                       # the third column of the second row group cannot be written
        t.append(id, Int64(1))
        t.append(cust, Int64(1))
        t.append(price, Int64(1_000_000_000_000_000_000))
        t.append(day, Int64(1))
        t.append(status, Int64(0))
        t.append(region, Int64(0))
        t.append(paid, Int64(1))
        t.append_float(ratio, Float64(1))
    return t^


def bytes_of(path: String) raises -> List[UInt8]:
    return open(path, "r").read_bytes()


def main() raises:
    var n = 25_000
    var t = table(n)
    var counts: List[Int] = [2, 3, 5, 8, 0]
    var codecs: List[Int] = [UNCOMPRESSED, SNAPPY, ZSTD]
    for codec in codecs:
        write_parquet("one.parquet", t, "{}", codec, 8000, threads=1)          # four row groups
        var want = bytes_of("one.parquet")
        for c in counts:
            write_parquet("many.parquet", t, "{}", codec, 8000, threads=c)
            var got = bytes_of("many.parquet")
            expect(len(got) == len(want), "codec " + String(codec) + " threads " + String(c) + ": " + String(len(got)) + " bytes, one thread wrote " + String(len(want)))
            var same = True
            for i in range(len(want)):
                if got[i] != want[i]:
                    same = False
                    break
            expect(same, "codec " + String(codec) + " threads " + String(c) + ": the bytes differ from the single-thread file")
        var back = read_parquet("many.parquet")
        expect(back.table.rows() == n, "rows read back: " + String(back.table.rows()))
        expect(back.table.cols[0].values[24_999] == 1_024_999, "id of the last row: " + String(back.table.cols[0].values[24_999]))
        expect(back.table.cols[3].values[7] == 19_000, "day of row 7: " + String(back.table.cols[3].values[7]))

    # a column that cannot be written: one error, the same for every count, and no file (nor temporary file) is left
    var bad = table(16_000, with_bad_decimal=True)
    var msgs = List[String]()
    for c in [1, 3, 0]:
        var msg = String("")
        try:
            write_parquet("bad.parquet", bad, "{}", ZSTD, 8000, threads=c)
        except e:
            msg = String(e)
        expect(msg.byte_length() > 0, "an out-of-range DECIMAL raises (threads " + String(c) + ")")
        expect("does not fit DECIMAL(18, 2)" in msg, "the error names the column's problem: `" + msg + "`")
        expect(not Path("bad.parquet").exists(), "no file is left behind (threads " + String(c) + ")")
        msgs.append(msg)
    expect(msgs[0] == msgs[1] and msgs[1] == msgs[2], "the error does not depend on the thread count")
    print("ok")
