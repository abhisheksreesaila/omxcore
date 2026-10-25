# F-181: does this machine's GPU do what omxcore's pipeline kernel relies on? Run on the macos-26 runner (Metal) and,
# as a control, on Linux (NVIDIA). One line per premise: `PROBE <name>: ok` or `PROBE <name>: wrong (...)`.
#   mojo run [--target-accelerator metal:1] ci/mac/probe_metal.mojo
# Premises: a kernel writes through a device address passed as an integer, and through one it reads out of another
# buffer (the engine's column table); dynamic shared memory of 16 and 32 KB; 64-bit integer + - * // % with negative
# operands; 32-bit warp shuffles. Copies are explicit (no zero-copy reads: modular/modular #6145 on Apple).
# Run 1 on the runner (2026-10-24): every premise failed, typed-pointer kernels included in nothing, so this version
# also has controls: `typed-*` lines pass buffers as typed pointers (as MAX's own kernels do), `gpu-address` prints
# the address the kernel sees beside the host's `unsafe_ptr()`, and `learned-*` lines use the kernel's own view of an
# address (learned by a first kernel) for a buffer that is, or is not, also an argument (residency).
# Run 2 (40171534745): typed pointers fine and the host's address is the kernel's; every pointer made from an integer
# wrong, even of a bound buffer: in AIR it is a generic `ptr`, and Metal's device memory is `ptr addrspace(1)`. So the
# premises now use `KP`, the address space omxcore's kernels use (GLOBAL on an Apple GPU, generic elsewhere);
# `generic-*` lines keep the plain-pointer result as evidence.
from std.memory import Pointer, bitcast
from std.sys.info import is_apple_gpu
from max.gpu import thread_idx, block_idx, barrier
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.memory import AddressSpace, external_memory
from max.gpu.primitives.warp import shuffle_down

comptime GP = Pointer[Int64, MutUntrackedOrigin]                       # a plain (generic) pointer
comptime KAS = AddressSpace.GLOBAL if is_apple_gpu() else AddressSpace.GENERIC
comptime LP = Pointer[Int64, MutUntrackedOrigin, address_space=KAS]    # as kernels_aot/pipeline_gpu.mojo makes them
comptime N = 1024
comptime TP = UnsafePointer[Int64, MutAnyOrigin]


def k_typed(p: TP):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    p[i] = Int64(i) * 3


def k_where(p: TP, q: TP, o: TP):
    """The kernel's view of two buffers' addresses."""
    if thread_idx.x == 0 and block_idx.x == 0:
        o[0] = Int64(Int(p))
        o[1] = Int64(Int(q))


def k_addr(addr: Int64):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    LP(unsafe_from_address=Int(addr))[i] = Int64(i) * 3


def k_addr_with(p: TP, addr: Int64):
    """Writes through `addr` while `p` (the same buffer, or another) is an argument."""
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    LP(unsafe_from_address=Int(addr))[i] = Int64(i) * 5
    if i == 0:
        p[N] = 0                                    # touch the argument so it stays one


def k_roundtrip(p: TP):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    LP(unsafe_from_address=Int(p))[i] = Int64(i) * 7


def k_generic(addr: Int64):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    GP(unsafe_from_address=Int(addr))[i] = Int64(i) * 3


def k_table(table: TP, anchor: TP):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    var dst = LP(unsafe_from_address=Int(table[1]))
    dst[i] = Int64(i) + 1000
    if i == 0:
        anchor[N] = 0


def k_shared(o: TP, words: Int64):
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
        o[0] = s


def k_int64(a: TP, b: TP, o: TP):
    var i = Int(thread_idx.x)
    var x = a[i]
    var y = b[i]
    o[i * 5] = x + y
    o[i * 5 + 1] = x - y
    o[i * 5 + 2] = x * y
    o[i * 5 + 3] = x // y if y != 0 else 0
    o[i * 5 + 4] = x % y if y != 0 else 0


def k_shuffle(o: TP):
    var tid = Int(thread_idx.x)
    var v = UInt32(tid * 7 + 1)
    var x = shuffle_down(SIMD[DType.uint32, 1](v), UInt32(1))
    o[tid] = Int64(x[0])


def _back(ctx: DeviceContext, buf: DeviceBuffer[DType.int64], n: Int) raises -> List[Int64]:
    var h = List[Int64](length=n, fill=0)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    return h^


def _count_bad(h: List[Int64], mul: Int64, add: Int64) -> Int:
    var bad = 0
    for i in range(N):
        if h[i] != Int64(i) * mul + add:
            bad += 1
    return bad


def _say(name: String, bad: Int, of: Int):
    print("PROBE " + name + ":", "ok" if bad == 0 else "wrong (" + String(bad) + " of " + String(of) + ")")


def main() raises:
    var count = DeviceContext.number_of_devices()
    if count == 0:
        print("PROBE DEVICE: none")
        return
    var ctx = DeviceContext()
    print("PROBE DEVICE:", ctx.name(), "·", ctx.api())

    # controls: typed pointers, as MAX's kernels pass them
    var a = ctx.enqueue_create_buffer[DType.int64](N + 1)
    var b = ctx.enqueue_create_buffer[DType.int64](N + 1)
    ctx.enqueue_memset(a, 0)
    ctx.enqueue_function[k_typed](a.unsafe_ptr(), grid_dim=N // 256, block_dim=256)
    _say("typed-write", _count_bad(_back(ctx, a, N), 3, 0), N)

    var w = ctx.enqueue_create_buffer[DType.int64](2)
    ctx.enqueue_function[k_where](a.unsafe_ptr(), b.unsafe_ptr(), w.unsafe_ptr(), grid_dim=1, block_dim=32)
    var seen = _back(ctx, w, 2)
    var host_a = Int64(Int(a.unsafe_ptr()))
    var host_b = Int64(Int(b.unsafe_ptr()))
    print("PROBE gpu-address:", "same" if seen[0] == host_a and seen[1] == host_b else "differs",
          "(host", hex(host_a), hex(host_b), "kernel", hex(seen[0]), hex(seen[1]), ")")

    # 0. a plain pointer made from the integer (wrong on Metal: generic address space)
    ctx.enqueue_memset(a, 0)
    ctx.enqueue_function[k_generic](host_a, grid_dim=N // 256, block_dim=256)
    _say("generic-int-address", _count_bad(_back(ctx, a, N), 3, 0), N)

    # 1. an integer address (the host's unsafe_ptr), alone
    ctx.enqueue_memset(a, 0)
    ctx.enqueue_function[k_addr](host_a, grid_dim=N // 256, block_dim=256)
    _say("int-address", _count_bad(_back(ctx, a, N), 3, 0), N)

    # 1b. the pointer as an integer inside the kernel and back
    ctx.enqueue_memset(a, 0)
    ctx.enqueue_function[k_roundtrip](a.unsafe_ptr(), grid_dim=N // 256, block_dim=256)
    _say("typed-roundtrip", _count_bad(_back(ctx, a, N), 7, 0), N)

    # 1c. the kernel's own address of `a`, with `a` an argument too, then with only `b` an argument
    ctx.enqueue_memset(a, 0)
    ctx.enqueue_function[k_addr_with](a.unsafe_ptr(), seen[0], grid_dim=N // 256, block_dim=256)
    _say("learned-address-arg", _count_bad(_back(ctx, a, N), 5, 0), N)
    ctx.enqueue_memset(a, 0)
    ctx.enqueue_function[k_addr_with](b.unsafe_ptr(), seen[0], grid_dim=N // 256, block_dim=256)
    _say("learned-address-not-arg", _count_bad(_back(ctx, a, N), 5, 0), N)

    # 2. an address read out of another buffer (the kernel's own view of `a`), `a` not an argument
    var table = ctx.enqueue_create_buffer[DType.int64](4)
    var th: List[Int64] = [0, seen[0], 0, 0]
    ctx.enqueue_copy(dst_buf=table, src_ptr=th.unsafe_ptr())
    ctx.enqueue_memset(a, 0)
    ctx.enqueue_function[k_table](table.unsafe_ptr(), b.unsafe_ptr(), grid_dim=N // 256, block_dim=256)
    _say("address-in-buffer", _count_bad(_back(ctx, a, N), 1, 1000), N)
    _ = th^

    # 3. dynamic shared memory
    for kb in [16, 32]:
        var words = kb * 1024 // 8
        var o = ctx.enqueue_create_buffer[DType.int64](1)
        ctx.enqueue_memset(o, 0)
        try:
            ctx.enqueue_function[k_shared](o.unsafe_ptr(), Int64(words), grid_dim=1, block_dim=256, shared_mem_bytes=kb * 1024)
            var r = _back(ctx, o, 1)
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
    ctx.enqueue_function[k_int64](da.unsafe_ptr(), db.unsafe_ptr(), dout.unsafe_ptr(), grid_dim=1, block_dim=64)
    var got = _back(ctx, dout, 64 * 5)
    var bad = 0
    for i in range(64):
        var x = xs[i]
        var y = ys[i]
        var want: List[Int64] = [x + y, x - y, x * y, x // y if y != 0 else 0, x % y if y != 0 else 0]
        for k in range(5):
            if got[i * 5 + k] != want[k]:
                bad += 1
    _say("int64-maths", bad, 320)
    _ = xs^
    _ = ys^

    # 5. warp shuffles (32 lanes)
    var so = ctx.enqueue_create_buffer[DType.int64](64)
    ctx.enqueue_function[k_shuffle](so.unsafe_ptr(), grid_dim=1, block_dim=64)
    var sh = _back(ctx, so, 64)
    bad = 0
    for i in range(64):
        if i % 32 != 31 and sh[i] != Int64((i + 1) * 7 + 1):
            bad += 1
    _say("shuffle-32", bad, 62)
    _ = a^
    _ = b^
    _ = w^
    _ = table^
    _ = da^
    _ = db^
    _ = dout^
    _ = so^
