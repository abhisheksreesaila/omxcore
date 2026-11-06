# F-210: OP_ICMPCOL, one pass over a column for `column <cmp> int64 constant` (the host makes it from OP_LOADI + OP_LOAD_CONST + OP_ICMP, query.cpu_code).
# The kernel runs it over int32, int64 and uint8 columns, with all six comparisons, a row count that is not a multiple of a block or of a
# vector (the scalar tails), and negative and extreme constants. The count of rows that pass and the sum of a second column over them
# are compared with a plain loop.
from std.memory.alloc import unsafe_alloc
from omxcore.kernels.pipeline_cpu import pipeline_plain

comptime MAGIC = 0x4D585031
comptime OP_LT = 20
comptime OP_LE = 21
comptime OP_GT = 22
comptime OP_GE = 23
comptime OP_EQ = 24
comptime OP_NE = 25


def holds(cmp: Int, v: Int64, k: Int64) -> Bool:
    if cmp == OP_LT:
        return v < k
    if cmp == OP_LE:
        return v <= k
    if cmp == OP_GT:
        return v > k
    if cmp == OP_GE:
        return v >= k
    if cmp == OP_EQ:
        return v == k
    return v != k


def run(col0: Int, dtype: Int, col1: Int, n: Int, k: Int64, cmp: Int) raises -> Tuple[Int, Int64, Int64]:
    # SUM(col1) WHERE col0 <cmp> k:  ICMPCOL r0 = col0 <cmp> consts[0]; LOADI r0 = col1; AGG_ISUM slot0 += r0
    var code: List[Int32] = [MAGIC, 1, 0, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                             56, 0, 0, 0, Int32(cmp), 3, 0, 1, 0, 0, 53, 0, 0, 0, 0]
    var consts: List[Float64] = [Float64(from_bits=UInt64(k))]        # the pool holds int64 bits
    var set_bytes: List[UInt8] = [0]
    var set_meta: List[Int64] = [0, 0, 0]
    var tbl: List[Int64] = [0, Int64(n), Int64(col0), Int64(col1)]
    var dtypes: List[Int32] = [Int32(dtype), 1]
    var out_vals: List[Float64] = [0.0]
    var out_cnt: List[Int64] = [0]
    var rc = pipeline_plain(Int(code.unsafe_ptr()), Int(consts.unsafe_ptr()), Int(set_bytes.unsafe_ptr()),
                            Int(set_meta.unsafe_ptr()), Int(tbl.unsafe_ptr()), 1, Int(dtypes.unsafe_ptr()),
                            0, 1, 0, n, Int(out_vals.unsafe_ptr()), Int(out_cnt.unsafe_ptr()), 4, False)
    _ = code^, consts^, set_bytes^, set_meta^, tbl^, dtypes^      # Mojo frees at last use: keep them past the call
    return (rc, out_vals.unsafe_ptr().bitcast[Int64]()[0], out_cnt[0])


def main() raises:
    var n = 1_000_003
    var a32 = unsafe_alloc[Int32](n)
    var a64 = unsafe_alloc[Int64](n)
    var a8 = unsafe_alloc[UInt8](n)
    var v = unsafe_alloc[Int64](n)
    for i in range(n):
        a32[i] = Int32((i * 7919) % 2001 - 1000)             # negative and positive
        a64[i] = Int64((i * 104729) % 2001 - 1000) * 1_000_000_007
        a8[i] = UInt8(i % 251)
        v[i] = Int64(i % 1000) + 1
    var cmps: List[Int] = [OP_LT, OP_LE, OP_GT, OP_GE, OP_EQ, OP_NE]
    var consts32: List[Int64] = [-1000, -1, 0, 500, 1000, 5000, -5000]
    var consts64: List[Int64] = [-1000 * 1_000_000_007, 0, 17 * 1_000_000_007, 1000 * 1_000_000_007, Int64.MAX, Int64.MIN]
    var consts8: List[Int64] = [0, 1, 100, 250, 251, -1]
    var checks = 0
    for kind in range(3):
        var ks = consts32.copy() if kind == 0 else (consts64.copy() if kind == 1 else consts8.copy())
        for ci in range(len(cmps)):
            for ki in range(len(ks)):
                var k = ks[ki]
                var cmp = cmps[ci]
                var want_n = 0
                var want_sum = Int64(0)
                for i in range(n):
                    var x = Int64(a32[i]) if kind == 0 else (a64[i] if kind == 1 else Int64(a8[i]))
                    if holds(cmp, x, k):
                        want_n += 1
                        want_sum += v[i]
                var r = run(Int(a32) if kind == 0 else (Int(a64) if kind == 1 else Int(a8)), 0 if kind == 0 else (1 if kind == 1 else 4), Int(v), n, k, cmp)
                if r[0] != 0 or Int(r[2]) != want_n or r[1] != want_sum:
                    raise Error("FAILED: column kind " + String(kind) + " cmp " + String(cmp) + " const " + String(k) + ": rc " + String(r[0]) + " count " +
                                String(r[2]) + " sum " + String(r[1]) + ", wanted count " + String(want_n) + " sum " + String(want_sum))
                checks += 1
    print("icmpcol ok:", checks, "comparisons")
