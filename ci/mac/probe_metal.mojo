# F-181: does this machine's GPU do what omxcore's pipeline kernel relies on? Run on the macos-26 runner (Metal) and,
# as a control, on Linux (NVIDIA). One line per premise: `PROBE <name>: ok` or `PROBE <name>: wrong (...)`.
#   mojo run [--target-accelerator metal:1] ci/mac/probe_metal.mojo
# Premises: a kernel writes through a device address passed as an integer, and through one it reads out of another
# buffer (the engine's column table); dynamic shared memory of 16 and 32 KB; 64-bit integer + - * // % with negative
# operands; 32-bit warp shuffles. Copies are explicit (no zero-copy reads: modular/modular #6145 on Apple).
from std.memory import Pointer, bitcast
from max.gpu import thread_idx, block_idx, barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace, external_memory
from max.gpu.primitives.warp import shuffle_down

comptime LP = Pointer[Int64, MutUntrackedOrigin]
comptime N = 1024


def k_addr(addr: Int64):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    LP(unsafe_from_address=Int(addr))[i] = Int64(i) * 3


def k_table(table: Int64):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    var dst = LP(unsafe_from_address=Int(LP(unsafe_from_address=Int(table))[1]))
    dst[i] = Int64(i) + 1000


def k_shared(out_a: Int64, words: Int64):
    var smem = external_memory[Int64, address_space=AddressSpace.SHARED, alignment=16]()
    var tid = Int(thread_idx.x)
    var n = Int(words)
    var j = tid
    while j < n:
        smem[j] = Int64(j)
        j += 256
    barrier()
    if tid == 0:
        var s = Int64(0)
        for k in range(n):
            s += smem[k]
        LP(unsafe_from_address=Int(out_a))[0] = s


def k_int64(a_a: Int64, b_a: Int64, out_a: Int64):
    var i = Int(thread_idx.x)
    var x = LP(unsafe_from_address=Int(a_a))[i]
    var y = LP(unsafe_from_address=Int(b_a))[i]
    var o = LP(unsafe_from_address=Int(out_a))
    o[i * 5] = x + y
    o[i * 5 + 1] = x - y
    o[i * 5 + 2] = x * y
    o[i * 5 + 3] = x // y if y != 0 else 0
    o[i * 5 + 4] = x % y if y != 0 else 0


def k_shuffle(out_a: Int64):
    var tid = Int(thread_idx.x)
    var v = UInt32(tid * 7 + 1)
    var x = shuffle_down(SIMD[DType.uint32, 1](v), UInt32(1))
    LP(unsafe_from_address=Int(out_a))[tid] = Int64(x[0])


def main() raises:
    var count = DeviceContext.number_of_devices()
    if count == 0:
        print("PROBE DEVICE: none")
        return
    var ctx = DeviceContext()
    print("PROBE DEVICE:", ctx.name(), "·", ctx.api())

    # 1. an integer address
    var buf = ctx.enqueue_create_buffer[DType.int64](N)
    ctx.enqueue_memset(buf, 0)
    ctx.enqueue_function[k_addr](Int64(Int(buf.unsafe_ptr())), grid_dim=N // 256, block_dim=256)
    var h = List[Int64](length=N, fill=0)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var bad = 0
    for i in range(N):
        if h[i] != Int64(i) * 3:
            bad += 1
    print("PROBE int-address:", "ok" if bad == 0 else "wrong (" + String(bad) + " of " + String(N) + ")")

    # 2. an address read out of another buffer
    var dst = ctx.enqueue_create_buffer[DType.int64](N)
    ctx.enqueue_memset(dst, 0)
    var table = ctx.enqueue_create_buffer[DType.int64](4)
    var th: List[Int64] = [0, Int64(Int(dst.unsafe_ptr())), 0, 0]
    ctx.enqueue_copy(dst_buf=table, src_ptr=th.unsafe_ptr())
    ctx.enqueue_function[k_table](Int64(Int(table.unsafe_ptr())), grid_dim=N // 256, block_dim=256)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=dst)
    ctx.synchronize()
    bad = 0
    for i in range(N):
        if h[i] != Int64(i) + 1000:
            bad += 1
    print("PROBE address-in-buffer:", "ok" if bad == 0 else "wrong (" + String(bad) + " of " + String(N) + ")")
    _ = th^

    # 3. dynamic shared memory
    for kb in [16, 32]:
        var words = kb * 1024 // 8
        var o = ctx.enqueue_create_buffer[DType.int64](1)
        ctx.enqueue_memset(o, 0)
        try:
            ctx.enqueue_function[k_shared](Int64(Int(o.unsafe_ptr())), Int64(words), grid_dim=1, block_dim=256,
                                           shared_mem_bytes=kb * 1024)
            var r = List[Int64](length=1, fill=0)
            ctx.enqueue_copy(dst_ptr=r.unsafe_ptr(), src_buf=o)
            ctx.synchronize()
            var want = Int64(words) * Int64(words - 1) // 2
            print("PROBE shared-" + String(kb) + "kb:", "ok" if r[0] == want else "wrong (" + String(r[0]) + ", wanted " + String(want) + ")")
        except e:
            print("PROBE shared-" + String(kb) + "kb: wrong (" + String(e) + ")")

    # 4. 64-bit integer maths
    var xs = List[Int64]()
    var ys = List[Int64]()
    for i in range(64):
        xs.append((Int64(i) - 32) * 123456789012)
        ys.append(Int64(i % 7) - 3)
    var da = ctx.enqueue_create_buffer[DType.int64](64)
    var db = ctx.enqueue_create_buffer[DType.int64](64)
    var dout = ctx.enqueue_create_buffer[DType.int64](64 * 5)
    ctx.enqueue_copy(dst_buf=da, src_ptr=xs.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=db, src_ptr=ys.unsafe_ptr())
    ctx.enqueue_function[k_int64](Int64(Int(da.unsafe_ptr())), Int64(Int(db.unsafe_ptr())), Int64(Int(dout.unsafe_ptr())),
                                  grid_dim=1, block_dim=64)
    var got = List[Int64](length=64 * 5, fill=0)
    ctx.enqueue_copy(dst_ptr=got.unsafe_ptr(), src_buf=dout)
    ctx.synchronize()
    bad = 0
    for i in range(64):
        var x = xs[i]
        var y = ys[i]
        var want: List[Int64] = [x + y, x - y, x * y, x // y if y != 0 else 0, x % y if y != 0 else 0]
        for k in range(5):
            if got[i * 5 + k] != want[k]:
                bad += 1
    print("PROBE int64-maths:", "ok" if bad == 0 else "wrong (" + String(bad) + " of 320)")
    _ = xs^
    _ = ys^

    # 5. warp shuffles (32 lanes)
    var so = ctx.enqueue_create_buffer[DType.int64](64)
    ctx.enqueue_function[k_shuffle](Int64(Int(so.unsafe_ptr())), grid_dim=1, block_dim=64)
    var sh = List[Int64](length=64, fill=0)
    ctx.enqueue_copy(dst_ptr=sh.unsafe_ptr(), src_buf=so)
    ctx.synchronize()
    bad = 0
    for i in range(64):
        if i % 32 != 31 and sh[i] != Int64((i + 1) * 7 + 1):
            bad += 1
    print("PROBE shuffle-32:", "ok" if bad == 0 else "wrong (" + String(bad) + " of 62)")
    _ = buf^
    _ = dst^
    _ = table^
    _ = da^
    _ = db^
    _ = dout^
    _ = so^
