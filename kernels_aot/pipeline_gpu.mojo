"""
MXFrame fused pipeline kernel, GPU side (F-008).

Interprets the same bytecode program as pipeline_cpu.mojo. Each thread owns
RPT rows of a tile (TILE = BLOCK * RPT rows per block); the interpreter's
"registers" are a thread-private register file kept in dynamic shared memory
(`regs[reg][tile_row]`, conflict-free), so nothing spills to local memory and
no barriers are needed between instructions. Every thread of a warp executes
the same instruction (uniform control flow); the predicate only masks rows.
The program itself is staged in shared memory once per block.

Aggregation modes (chosen from the slot and group counts):
  * 0, few groups ((n_slots + 1) * n_groups <= SMALL, including the global
    aggregate with one group): every thread accumulates its own partial per
    (slot, group); at the end the block reduces them with warp shuffles and
    writes one row of `scratch`; a second kernel merges the per-block rows in
    block order. No atomics, and results are deterministic.
  * 1, a middle number of groups ((n_slots + 1) * n_groups shared-memory doubles
    fit next to the register file): the block keeps one table of accumulators in
    shared memory, updates it with shared float64 atomics, and adds the table to
    the output once per block with global atomics.
  * 2, many groups: per-row float64 atomics (add / min / max) into the output.
  * 5, many groups on an Int64 word (Apple's GPU, F-184): mode 4's ordered walk with the block's table in its row
    of `scratch` in device memory instead of shared memory, and group g owned by thread g % B (each slot is read and
    written by one thread only, so no device-memory barrier is needed). For more groups than mode 4 holds; up to
    MANY_MAX_NS slots, in at most MANY_SCRATCH_WORDS // ns blocks (`scratch_rows`).
  * 4, reproducible mode (F-067) replaces modes 1 and 2 when the table fits shared memory: no atomics.
    Per tile, every row's group id is written to shared memory; after a barrier the thread that owns a
    group (group g belongs to thread g % BLOCK) walks the tile's rows in row order and adds the matching
    ones to the block's table; at the end the table is written to a row of `scratch` and the same merge
    kernel as modes 0 and 3 adds the per-block rows in block order. The order of additions is a function
    of the row order, the grid size and the tile size only. More groups than fit, and mode 2's shape,
    return -6 (the caller runs the CPU's defined order).
  Every accumulation atomic uses monotonic ordering: the default sequentially
  consistent ordering makes NVPTX fence around each one (~20x slower). Modes 0
  and 3 finish with a second kernel that merges the per-block rows of `scratch`
  in block order, so their results are deterministic.

All arithmetic is float64 (columns are widened on load) on NVIDIA. On an Apple GPU (Metal: no double, no general
64-bit atomics; F-183, F-184) the kernels are instantiated with an Int64 word instead (`WORD`, chosen at compile time
for the build's GPU): the integer programs of the Mojo API run unchanged on it (int registers, 0 / 1 booleans, counts
and int64 totals), a float instruction sets the overflow flag (the host never sends one: `int_word_program_ok`), and
only the atomic-free modes 0, 3 and 4 run (more groups than mode 4 holds return -6). Device pointers are
passed as integers; the Python side owns device memory (CUDA driver API on the
primary context, like AOTKernelsGPU), so already-resident morsels are accepted
without another copy.

Build:
    pixi run mojo build kernels_aot/pipeline_gpu.mojo \
        --emit shared-lib -o kernels_aot/libmxpipeline_gpu.so
"""

from max.gpu.host import DeviceContext
from std.math import ceildiv, floor
from max.gpu import thread_idx, block_idx, grid_dim, barrier
from max.gpu.primitives.warp import shuffle_down
from max.gpu.memory import AddressSpace, external_memory
from std.memory import Pointer, stack_allocation, bitcast
from std.sys import has_apple_gpu_accelerator, is_defined
from std.sys.info import is_apple_gpu

comptime F64 = DType.float64
# Host side only (never read inside a kernel, which is compiled for the GPU target): is the accelerator this build
# compiles its kernels for an Apple GPU (a native Apple Silicon build, or `--target-accelerator metal:<n>`)? The
# kernels take their word as a parameter, `WORD`, and plans are sized to `SMEM_LIMIT`.
# `-D OMXCORE_APPLE_GPU_SHAPE` gives any GPU Apple's plan (the Int64 word, the atomic-free modes, 32 KB, 64-thread
# narrow blocks), so the Apple path is tested on an NVIDIA card too: scripts/check_apple_gpu.py's M6 runs the GPU
# Mojo suites and the orders check with it. Never set in a release build.
comptime APPLE_GPU = has_apple_gpu_accelerator() or is_defined["OMXCORE_APPLE_GPU_SHAPE"]()
comptime WORD = DType.int64 if APPLE_GPU else F64
comptime SMEM_LIMIT: Int = (32 if APPLE_GPU else 48) * 1024     # dynamic shared memory per block: Apple 32 KB, NVIDIA 48 KB
comptime BLOCK: Int = 256
comptime SMALL: Int = 48            # per-thread accumulators (mode 0)
comptime HDR: Int = 16
comptime MAX_CONSTS: Int = 64
comptime WARPS: Int = BLOCK // 32
comptime LIM53 = Int64(1) << 53         # an int64 value outside +-2^53 is not exact as a double
comptime REPRO_MAX_NS: Int = 4096   # most (slots + 1) * groups the ordered mode holds (pipeline_kernels.REPRO_MAX_NS)
comptime MANY_MAX_NS: Int = 1 << 20          # mode 5: most (slots + 1) * groups (more: the CPU answers)
comptime MANY_SCRATCH_WORDS: Int = 1 << 24   # mode 5: device words for the blocks' tables (128 MB); fewer blocks above
comptime NARROW_B: Int = 64 if APPLE_GPU else 128   # threads of a narrow block: half (NVIDIA) or a quarter (Apple's 32 KB) of BLOCK

comptime OP_LOAD_COL = 1
comptime OP_LOAD_CONST = 2
comptime OP_LOADI = 3           # the column's values as int64 BITS in the register (an "int" register: F-060c, F-060b)
comptime OP_ADD = 10
comptime OP_SUB = 11
comptime OP_MUL = 12
comptime OP_DIV = 13
comptime OP_MOD = 14            # fmod: the sign follows the dividend (integer % a nonzero constant, F-101)
comptime OP_IADD = 15           # int64 + - * of two "int" registers (int64 bits in the float64 words); a result outside int64 sets *ovf (F-060b)
comptime OP_ISUB = 16
comptime OP_IMUL = 17
comptime OP_F2I = 18            # dst = int64(a): a float register of whole numbers within +-2^53
comptime OP_I2F = 19            # dst = float64(a) of an "int" register; a value outside +-2^53 sets *ovf
comptime OP_LT = 20
comptime OP_LE = 21
comptime OP_GT = 22
comptime OP_GE = 23
comptime OP_EQ = 24
comptime OP_NE = 25
comptime OP_AND = 30
comptime OP_OR = 31
comptime OP_NOT = 32
comptime OP_FLOOR = 33
comptime OP_F32 = 34            # float64(float32(a)): a FLOAT result rounded after each operation (F-110)
comptime OP_SELECT = 35
comptime OP_BETWEEN = 36
comptime OP_IN_SET = 37
comptime OP_CMPC = 38
comptime OP_ISNULL = 41         # 1.0 where the column's validity bit is clear (device bitmap: 3-entry column table, as the CPU)
comptime OP_IDIV = 42           # exact int64 ops of the Mojo API's programs (F-137), the CPU's semantics:
comptime OP_IMOD = 43           #   floor //, Mojo's Int64 %, x // 0 = x % 0 = 0, x % -1 = 0,
comptime OP_IMIN = 44           #   INT64_MIN // -1 sets *ovf
comptime OP_IMAX = 45
comptime OP_ICMP = 46           # 1.0 / 0.0 float register: a <cmp in operand c> b
comptime OP_AGG_SUM = 50
comptime OP_AGG_MIN = 51
comptime OP_AGG_MAX = 52
comptime OP_AGG_ISUM = 53       # sum of an "int" register in int64 (F-060, F-060b): the slot's float64 word holds int64 bits
comptime OP_AGG_IMIN = 54       # min / max of an "int" register in int64 (F-060c): the slot's float64 word holds int64 bits
comptime OP_AGG_IMAX = 55

comptime SH = AddressSpace.SHARED
# Device memory reached through an address passed as an integer: on an Apple GPU a pointer made from an integer must
# be in the device (GLOBAL) address space, since Metal has no generic one (a plain pointer's writes went nowhere on
# the macos-26 runner, ci/mac/probe_metal.mojo); elsewhere the generic pointer as before. `is_apple_gpu()` is resolved
# for the target each kernel is compiled for.
comptime KAS = AddressSpace.GLOBAL if is_apple_gpu() else AddressSpace.GENERIC
comptime SFP = Pointer[Float64, MutUntrackedOrigin, address_space=SH]
comptime SIP = Pointer[Int32, MutUntrackedOrigin, address_space=SH]
comptime FP = Pointer[Float64, MutUntrackedOrigin, address_space=KAS]
comptime SWP[W: DType] = Pointer[Scalar[W], MutUntrackedOrigin, address_space=SH]
comptime WP[W: DType] = Pointer[Scalar[W], MutUntrackedOrigin, address_space=KAS]
comptime IP = Pointer[Int32, MutUntrackedOrigin, address_space=KAS]
comptime LP = Pointer[Int64, MutUntrackedOrigin, address_space=KAS]
comptime BP = Pointer[UInt8, MutUntrackedOrigin, address_space=KAS]


# ── monotonic atomics ────────────────────────────────────────────────────────
# llvm.atomicrmw with monotonic ordering (CUDA atomicAdd semantics). The stdlib's
# Atomic defaults to sequentially consistent, which makes NVPTX fence around every
# atomic (~20x slower). Same lowering as _atomic.mojo, for float64 / int64.
# bin_op: add=1 (int), fadd=11, fmax=13, fmin=14; ordering: monotonic=2.

@always_inline
def _llvm_ptr[space: AddressSpace, T: DType](ptr: Pointer[Scalar[T], MutUntrackedOrigin, address_space=space]) -> __mlir_type.`!llvm.ptr`:
    comptime if space == AddressSpace.SHARED:
        var p3 = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr<3>`](ptr._mlir_value)
        return __mlir_op.`llvm.addrspacecast`[_type = __mlir_type.`!llvm.ptr`](p3)
    else:
        return __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr._mlir_value)


@always_inline
def _atomic_add_f64[space: AddressSpace](ptr: Pointer[Float64, MutUntrackedOrigin, address_space=space], val: Float64):
    var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.f64](val)
    _ = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`11 : i64`, ordering = __mlir_attr.`2 : i64`, _type = __mlir_type.f64](_llvm_ptr[space, F64](ptr), v)


@always_inline
def _atomic_min_f64[space: AddressSpace](ptr: Pointer[Float64, MutUntrackedOrigin, address_space=space], val: Float64):
    var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.f64](val)
    _ = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`14 : i64`, ordering = __mlir_attr.`2 : i64`, _type = __mlir_type.f64](_llvm_ptr[space, F64](ptr), v)


@always_inline
def _atomic_max_f64[space: AddressSpace](ptr: Pointer[Float64, MutUntrackedOrigin, address_space=space], val: Float64):
    var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.f64](val)
    _ = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`13 : i64`, ordering = __mlir_attr.`2 : i64`, _type = __mlir_type.f64](_llvm_ptr[space, F64](ptr), v)


@always_inline
def _atomic_add_i64(ptr: LP, val: Int64):
    var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.i64](val)
    _ = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`1 : i64`, ordering = __mlir_attr.`2 : i64`, _type = __mlir_type.i64](_llvm_ptr[KAS, DType.int64](ptr), v)


@always_inline
def _atomic_add_bits[space: AddressSpace](ptr: Pointer[Float64, MutUntrackedOrigin, address_space=space], val: Int64) -> Int64:
    """int64 atomic add on a float64 slot that holds int64 bits (the pointer is untyped in LLVM); returns the old value."""
    var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.i64](val)
    var r = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`1 : i64`, ordering = __mlir_attr.`2 : i64`, _type = __mlir_type.i64](_llvm_ptr[space, F64](ptr), v)
    return __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type[Int64]](r)


@always_inline
def _atomic_min_bits[space: AddressSpace](ptr: Pointer[Float64, MutUntrackedOrigin, address_space=space], val: Int64):
    """int64 atomic min on a float64 slot that holds int64 bits (llvm bin_op 8: signed min)."""
    var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.i64](val)
    _ = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`8 : i64`, ordering = __mlir_attr.`2 : i64`, _type = __mlir_type.i64](_llvm_ptr[space, F64](ptr), v)


@always_inline
def _atomic_max_bits[space: AddressSpace](ptr: Pointer[Float64, MutUntrackedOrigin, address_space=space], val: Int64):
    """int64 atomic max on a float64 slot that holds int64 bits (llvm bin_op 7: signed max)."""
    var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.i64](val)
    _ = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`7 : i64`, ordering = __mlir_attr.`2 : i64`, _type = __mlir_type.i64](_llvm_ptr[space, F64](ptr), v)


@always_inline
def _i64_of[W: DType](x: Scalar[W]) -> Int64:
    """The int64 a slot word holds (its bits: a float64 word holds int64 bits, an Int64 word is the value)."""
    comptime if W == F64:
        return bitcast[DType.int64, 1](SIMD[F64, 1](rebind[Float64](x)))[0]
    else:
        return rebind[Int64](x)


@always_inline
def _word_of[W: DType = F64](x: Int64) -> Scalar[W]:
    comptime if W == F64:
        return rebind[Scalar[W]](bitcast[F64, 1](SIMD[DType.int64, 1](x))[0])
    else:
        return rebind[Scalar[W]](x)


@always_inline
def _note_ovf(a: Int64, b: Int64, ovf: LP):
    """Flags *ovf when a + b (computed with wrapping) passed int64."""
    var s = a + b
    if ((a ^ s) & (b ^ s)) < 0:
        ovf[0] = 1


@always_inline
def _iext_words[W: DType](kind: Int, a: Scalar[W], b: Scalar[W]) -> Scalar[W]:
    """The integer min / max (OP_AGG_IMIN / IMAX) of two slot words that hold int64s, as a slot word."""
    var x = _i64_of(a)
    var y = _i64_of(b)
    return _word_of[W](min(x, y) if kind == OP_AGG_IMIN else max(x, y))


@always_inline
def _iadd_words[W: DType](a: Scalar[W], b: Scalar[W], ovf: LP) -> Scalar[W]:
    """The sum of two slot words that hold int64s, as a slot word."""
    var x = _i64_of(a)
    var y = _i64_of(b)
    _note_ovf(x, y, ovf)
    return _word_of[W](x + y)


@always_inline
def _iadd_value[W: DType](a: Scalar[W], v: Scalar[W], ovf: LP) -> Scalar[W]:
    """The slot word `a` (int64 bits) plus `v`, an "int" register's word (int64 bits)."""
    var x = _i64_of(a)
    var y = _i64_of(v)
    _note_ovf(x, y, ovf)
    return _word_of[W](x + y)


@always_inline
def _imul_ovf(x: Int64, y: Int64) -> Bool:
    """Does x * y pass int64? Both inside [-2^31, 2^31) cannot; otherwise exact (divide back, after the two
    quotients that trap)."""
    if ((x + 2147483648) | (y + 2147483648)) >> 32 == 0:
        return False
    if x == 0 or y == 0:
        return False
    if x == -1:
        return y == Int64.MIN
    if y == -1:
        return x == Int64.MIN
    return (x * y) // x != y


@always_inline
def _fp(addr: Int) -> FP:
    return FP(unsafe_from_address=addr)

@always_inline
def _wp[W: DType](addr: Int) -> WP[W]:
    return WP[W](unsafe_from_address=addr)

@always_inline
def _ip(addr: Int) -> IP:
    return IP(unsafe_from_address=addr)

@always_inline
def _lp(addr: Int) -> LP:
    return LP(unsafe_from_address=addr)

@always_inline
def _bp(addr: Int) -> BP:
    return BP(unsafe_from_address=addr)


@always_inline
def _load(addr: Int, dtype: Int, row: Int) -> Float64:
    if dtype == 2:
        return Pointer[Float32, MutUntrackedOrigin, address_space=KAS](unsafe_from_address=addr)[row].cast[F64]()
    elif dtype == 0:
        return _ip(addr)[row].cast[F64]()
    elif dtype == 3:
        return _fp(addr)[row]
    elif dtype == 1:
        return _lp(addr)[row].cast[F64]()
    else:
        return _bp(addr)[row].cast[F64]()


@always_inline
def _load_i[W: DType = F64](addr: Int, dtype: Int, row: Int) -> Scalar[W]:
    """OP_LOADI: the column value widened to int64, as the bits of the register's word."""
    if dtype == 0:
        return _word_of[W](_ip(addr)[row].cast[DType.int64]())
    elif dtype == 4:
        return _word_of[W](_bp(addr)[row].cast[DType.int64]())
    return _word_of[W](_lp(addr)[row])               # 1: int64, 8: int64 read only here (exact past 2^53)


@always_inline
def _b[W: DType = F64](x: Bool) -> Scalar[W]:
    return Scalar[W](1) if x else Scalar[W](0)


@always_inline
def _nz[W: DType](x: Scalar[W]) -> Bool:
    """x != 0 for the 0 / 1 booleans, by integer test (float64 compares are 1/64 rate on GeForce)."""
    return _i64_of(x) != 0


@always_inline
def _cmp_f(cmp: Int, x: Float32, t: Float32, na: Bool) -> Bool:
    """x <cmp> t; na: NaN is above every number (+inf too) and NaN = NaN, as DuckDB (F-070)."""
    if cmp == OP_GE:
        return x >= t or (na and not (x == x))
    elif cmp == OP_LT:
        return x < t or (na and not (t == t) and x == x)
    elif cmp == OP_LE:
        return x <= t or (na and not (t == t))
    elif cmp == OP_GT:
        return x > t or (na and not (x == x) and t == t)
    elif cmp == OP_EQ:
        return x == t or (na and not (x == x) and not (t == t))
    return not (x == t or (na and not (x == x) and not (t == t)))


@always_inline
def _cmp_d(cmp: Int, x: Float64, t: Float64, na: Bool) -> Bool:
    """x <cmp> t; na: NaN is above every number (+inf too) and NaN = NaN, as DuckDB (F-070)."""
    if cmp == OP_GE:
        return x >= t or (na and not (x == x))
    elif cmp == OP_LT:
        return x < t or (na and not (t == t) and x == x)
    elif cmp == OP_LE:
        return x <= t or (na and not (t == t))
    elif cmp == OP_GT:
        return x > t or (na and not (x == x) and t == t)
    elif cmp == OP_EQ:
        return x == t or (na and not (x == x) and not (t == t))
    return not (x == t or (na and not (x == x) and not (t == t)))


@always_inline
def _cmp_i(cmp: Int, x: Int64, t: Int64) -> Bool:
    if cmp == OP_GE:
        return x >= t
    elif cmp == OP_LT:
        return x < t
    elif cmp == OP_LE:
        return x <= t
    elif cmp == OP_GT:
        return x > t
    elif cmp == OP_EQ:
        return x == t
    return x != t


@always_inline
def _cmpc(addr: Int, dtype: Int, row: Int, thr: Float64, cmp: Int, na: Bool) -> Float64:
    """column[row] <cmp> thr in the column's own type (thr was rounded on the host)."""
    if dtype == 2:
        var x = Pointer[Float32, MutUntrackedOrigin, address_space=KAS](unsafe_from_address=addr)[row]
        return _b(_cmp_f(cmp, x, thr.cast[DType.float32](), na))
    elif dtype == 0:
        return _b(_cmp_i(cmp, _ip(addr)[row].cast[DType.int64](), Int64(thr)))
    elif dtype == 3:
        return _b(_cmp_d(cmp, _fp(addr)[row], thr, na))
    elif dtype == 1:
        return _b(_cmp_i(cmp, _lp(addr)[row], Int64(thr)))
    return _b(_cmp_i(cmp, _bp(addr)[row].cast[DType.int64](), Int64(thr)))


@always_inline
def _shfl_down[W: DType](v: Scalar[W], off: Int) -> Scalar[W]:
    """A 64-bit word's warp shuffle as two 32-bit shuffles (the stdlib has no 64-bit shuffle)."""
    var bits = bitcast[DType.uint32, 2](SIMD[W, 1](v))
    var lo = shuffle_down(SIMD[DType.uint32, 1](bits[0]), UInt32(off))
    var hi = shuffle_down(SIMD[DType.uint32, 1](bits[1]), UInt32(off))
    return bitcast[W, 1](SIMD[DType.uint32, 2](lo[0], hi[0]))[0]


@always_inline
def _combine[W: DType](kind: Int, a: Scalar[W], b: Scalar[W], ovf: LP) -> Scalar[W]:
    if kind == OP_AGG_MIN:
        return min(a, b)
    elif kind == OP_AGG_MAX:
        return max(a, b)
    elif kind == OP_AGG_ISUM:
        return _iadd_words(a, b, ovf)
    elif kind == OP_AGG_IMIN or kind == OP_AGG_IMAX:
        return _iext_words(kind, a, b)
    return a + b


@always_inline
def _warp_reduce[W: DType](kind: Int, v: Scalar[W], ovf: LP) -> Scalar[W]:
    var x = v
    var off = 16
    while off > 0:
        x = _combine(kind, x, _shfl_down(x, off), ovf)
        off >>= 1
    return x


@always_inline
def _ident[W: DType](kind: Int) -> Scalar[W]:
    comptime if W == F64:
        if kind == OP_AGG_MIN:
            return rebind[Scalar[W]](Float64.MAX)
        elif kind == OP_AGG_MAX:
            return rebind[Scalar[W]](-Float64.MAX)
        elif kind == OP_AGG_IMIN:
            return _word_of[W](Int64.MAX)
        elif kind == OP_AGG_IMAX:
            return _word_of[W](Int64.MIN)
        return Scalar[W](0)
    else:
        if kind == OP_AGG_IMIN:
            return _word_of[W](Int64.MAX)
        elif kind == OP_AGG_IMAX:
            return _word_of[W](Int64.MIN)
        return Scalar[W](0)


def _run[RPT: Int, B: Int](
    code: SIP, base: Int, count: Int, regs: SFP, consts: SFP, cols: LP, dts: IP,
    set_bytes: BP, set_meta: LP, tid: Int, row_base: Int, mask: Int, ovf: LP,
):
    """Run `count` instructions for this thread's RPT rows; bit k of `mask` = row k is live. `ovf`: the flag cell an
    int64 op sets when its result passes int64, or an int64 value past 2^53 is converted to a double (F-060b)."""
    comptime TILE = B * RPT
    for n in range(count):
        var ins = code + (HDR + (base + n) * 5)
        var op = Int(ins[0])
        var d = regs + Int(ins[1]) * TILE + tid
        if op == OP_LOAD_COL:
            var c = Int(ins[2])
            var addr = Int(cols[c])
            var dt = Int(dts[c])
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    d[k * B] = _load(addr, dt, row_base + tid + k * B)
        elif op == OP_LOADI:
            var c = Int(ins[2])
            var addr = Int(cols[c])
            var dt = Int(dts[c])
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    d[k * B] = _load_i(addr, dt, row_base + tid + k * B)
        elif op == OP_LOAD_CONST:
            var v = consts[Int(ins[2])]
            comptime for k in range(RPT):
                d[k * B] = v
        elif op == OP_CMPC:
            var c = Int(ins[2])
            var addr = Int(cols[c])
            var dt = Int(dts[c])
            var thr = consts[Int(ins[3])]
            var cmp = Int(ins[4]) & 255
            var na = (Int(ins[4]) >> 8) != 0
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    d[k * B] = _cmpc(addr, dt, row_base + tid + k * B, thr, cmp, na)
        elif op >= OP_IADD and op <= OP_I2F:
            var a = regs + Int(ins[2]) * TILE + tid
            var b = regs + Int(ins[3]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    if op == OP_F2I:
                        d[k * B] = _word_of(Int64(a[k * B]))
                    elif op == OP_I2F:
                        var v = _i64_of(a[k * B])
                        if v > LIM53 or v < -LIM53:
                            ovf[0] = 1
                        d[k * B] = Float64(v)
                    else:
                        var x = _i64_of(a[k * B])
                        var y = _i64_of(b[k * B])
                        var r = x + y
                        if op == OP_ISUB:
                            r = x - y
                            if ((x ^ y) & (x ^ r)) < 0:
                                ovf[0] = 1
                        elif op == OP_IMUL:
                            r = x * y
                            if _imul_ovf(x, y):
                                ovf[0] = 1
                        elif ((x ^ r) & (y ^ r)) < 0:
                            ovf[0] = 1
                        d[k * B] = _word_of(r)
        elif op < OP_LT:
            var a = regs + Int(ins[2]) * TILE + tid
            var b = regs + Int(ins[3]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    var x = a[k * B]
                    var y = b[k * B]
                    var r = x + y
                    if op == OP_MUL:
                        r = x * y
                    elif op == OP_SUB:
                        r = x - y
                    elif op == OP_DIV:
                        r = x / y
                    elif op == OP_MOD:
                        r = x % y                      # floored (the sign follows the divisor); fmod's follows the dividend
                        if r != 0.0 and ((x < 0.0) != (y < 0.0)):
                            r = r - y
                    d[k * B] = r
        elif op < OP_AND:
            var a = regs + Int(ins[2]) * TILE + tid
            var b = regs + Int(ins[3]) * TILE + tid
            var na = Int(ins[4]) != 0
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    d[k * B] = _b(_cmp_d(op, a[k * B], b[k * B], na))
        elif op < OP_NOT:
            var a = regs + Int(ins[2]) * TILE + tid
            var b = regs + Int(ins[3]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    var x = _nz(a[k * B])
                    var y = _nz(b[k * B])
                    d[k * B] = _b(x and y) if op == OP_AND else _b(x or y)
        elif op == OP_NOT:
            var a = regs + Int(ins[2]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    d[k * B] = _b(not _nz(a[k * B]))
        elif op == OP_FLOOR:
            var a = regs + Int(ins[2]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    d[k * B] = floor(a[k * B])
        elif op == OP_F32:
            var a = regs + Int(ins[2]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    d[k * B] = Float64(Float32(a[k * B]))
        elif op == OP_SELECT:
            var c = regs + Int(ins[2]) * TILE + tid
            var a = regs + Int(ins[3]) * TILE + tid
            var b = regs + Int(ins[4]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    d[k * B] = a[k * B] if _nz(c[k * B]) else b[k * B]
        elif op == OP_BETWEEN:
            var x = regs + Int(ins[2]) * TILE + tid
            var lo = regs + Int(ins[3]) * TILE + tid
            var hi = regs + Int(ins[4]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    var v = x[k * B]
                    d[k * B] = _b(v >= lo[k * B] and v <= hi[k * B])
        elif op == OP_IN_SET:
            var s = Int(ins[3])
            var off = Int(set_meta[s * 3])
            var length = Float64(Int(set_meta[s * 3 + 1]))
            var lo = Float64(Int(set_meta[s * 3 + 2]))
            var a = regs + Int(ins[2]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    var x = a[k * B] - lo
                    var hit = 0.0
                    if x >= 0.0 and x < length:
                        if set_bytes[off + Int(x)] != 0:
                            hit = 1.0
                    d[k * B] = hit
        elif op == OP_ISNULL:
            var slot = Int(dts[Int(ins[2])]) >> 8              # dtype 7 | (slot << 8): bitmap address, bit base
            var valid = _bp(Int(cols[slot]))
            var has = Int(cols[slot]) != 0
            var bit_base = Int(cols[slot + 1])
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    var null = 0.0
                    if has:
                        var bit = bit_base + row_base + tid + k * B
                        if ((Int(valid[bit >> 3]) >> (bit & 7)) & 1) == 0:
                            null = 1.0
                    d[k * B] = null
        elif op >= OP_IDIV and op <= OP_ICMP:
            var a = regs + Int(ins[2]) * TILE + tid
            var b = regs + Int(ins[3]) * TILE + tid
            var cmp = Int(ins[4])
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    var x = _i64_of(a[k * B])
                    var y = _i64_of(b[k * B])
                    if op == OP_IDIV:
                        var q = Int64(0)
                        if y == -1 and x == Int64.MIN:
                            ovf[0] = 1
                            q = x
                        elif y != 0:
                            q = x // y
                        d[k * B] = _word_of(q)
                    elif op == OP_IMOD:
                        d[k * B] = _word_of(0 if y == 0 or y == -1 else x % y)
                    elif op == OP_IMIN:
                        d[k * B] = _word_of(min(x, y))
                    elif op == OP_IMAX:
                        d[k * B] = _word_of(max(x, y))
                    else:
                        d[k * B] = _b(_cmp_i(cmp, x, y))


comptime SLP = Pointer[Int64, MutUntrackedOrigin, address_space=SH]


def _run_int[RPT: Int, B: Int](
    code: SIP, base: Int, count: Int, regs: SLP, consts: SLP, cols: LP, dts: IP, tid: Int, row_base: Int, mask: Int,
    ovf: LP,
):
    """`_run` on an Int64 word (Apple's GPU, F-183): the integer instructions of the Mojo API's programs, with the same
    results and overflow rules; booleans are 0 / 1. A float instruction (the host never sends one, see
    `int_word_program_ok`) sets the overflow flag instead of computing."""
    comptime TILE = B * RPT
    for n in range(count):
        var ins = code + (HDR + (base + n) * 5)
        var op = Int(ins[0])
        var d = regs + Int(ins[1]) * TILE + tid
        if op == OP_LOADI:
            var c = Int(ins[2])
            var addr = Int(cols[c])
            var dt = Int(dts[c])
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    d[k * B] = _load_i[DType.int64](addr, dt, row_base + tid + k * B)
        elif op == OP_LOAD_CONST:
            var v = consts[Int(ins[2])]
            comptime for k in range(RPT):
                d[k * B] = v
        elif op == OP_IADD or op == OP_ISUB or op == OP_IMUL:
            var a = regs + Int(ins[2]) * TILE + tid
            var b = regs + Int(ins[3]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    var x = a[k * B]
                    var y = b[k * B]
                    var r = x + y
                    if op == OP_ISUB:
                        r = x - y
                        if ((x ^ y) & (x ^ r)) < 0:
                            ovf[0] = 1
                    elif op == OP_IMUL:
                        r = x * y
                        if _imul_ovf(x, y):
                            ovf[0] = 1
                    elif ((x ^ r) & (y ^ r)) < 0:
                        ovf[0] = 1
                    d[k * B] = r
        elif op == OP_F2I:                               # a 0 / 1 boolean as the number: already one
            var a = regs + Int(ins[2]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    d[k * B] = a[k * B]
        elif op == OP_AND or op == OP_OR:
            var a = regs + Int(ins[2]) * TILE + tid
            var b = regs + Int(ins[3]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    var x = a[k * B] != 0
                    var y = b[k * B] != 0
                    d[k * B] = Int64(1) if (x and y if op == OP_AND else x or y) else Int64(0)
        elif op == OP_NOT:
            var a = regs + Int(ins[2]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    d[k * B] = Int64(1) if a[k * B] == 0 else Int64(0)
        elif op == OP_SELECT:
            var c = regs + Int(ins[2]) * TILE + tid
            var a = regs + Int(ins[3]) * TILE + tid
            var b = regs + Int(ins[4]) * TILE + tid
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    d[k * B] = a[k * B] if c[k * B] != 0 else b[k * B]
        elif op == OP_ISNULL:
            var slot = Int(dts[Int(ins[2])]) >> 8              # dtype 7 | (slot << 8): bitmap address, bit base
            var valid = _bp(Int(cols[slot]))
            var has = Int(cols[slot]) != 0
            var bit_base = Int(cols[slot + 1])
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    var null = Int64(0)
                    if has:
                        var bit = bit_base + row_base + tid + k * B
                        if ((Int(valid[bit >> 3]) >> (bit & 7)) & 1) == 0:
                            null = 1
                    d[k * B] = null
        elif op >= OP_IDIV and op <= OP_ICMP:
            var a = regs + Int(ins[2]) * TILE + tid
            var b = regs + Int(ins[3]) * TILE + tid
            var cmp = Int(ins[4])
            comptime for k in range(RPT):
                if (mask >> k) & 1 != 0:
                    var x = a[k * B]
                    var y = b[k * B]
                    if op == OP_IDIV:
                        var q = Int64(0)
                        if y == -1 and x == Int64.MIN:
                            ovf[0] = 1
                            q = x
                        elif y != 0:
                            q = x // y
                        d[k * B] = q
                    elif op == OP_IMOD:
                        d[k * B] = 0 if y == 0 or y == -1 else x % y
                    elif op == OP_IMIN:
                        d[k * B] = min(x, y)
                    elif op == OP_IMAX:
                        d[k * B] = max(x, y)
                    else:
                        d[k * B] = Int64(1) if _cmp_i(cmp, x, y) else Int64(0)
        else:
            ovf[0] = 1                                   # a float instruction: not on an Int64 word


@always_inline
def _run_w[RPT: Int, B: Int, W: DType](
    code: SIP, base: Int, count: Int, regs: SWP[W], consts: SWP[W], cols: LP, dts: IP,
    set_bytes: BP, set_meta: LP, tid: Int, row_base: Int, mask: Int, ovf: LP,
):
    """The interpreter for the kernel's word: `_run` (float64, NVIDIA) or `_run_int` (Int64, Apple)."""
    comptime if W == F64:
        _run[RPT, B](code, base, count, rebind[SFP](regs), rebind[SFP](consts), cols, dts, set_bytes, set_meta, tid,
                     row_base, mask, ovf)
    else:
        _run_int[RPT, B](code, base, count, rebind[SLP](regs), rebind[SLP](consts), cols, dts, tid, row_base, mask, ovf)


comptime INT_WORD_OPS: List[Int] = [OP_LOAD_CONST, OP_LOADI, OP_IADD, OP_ISUB, OP_IMUL, OP_F2I, OP_AND, OP_OR, OP_NOT,
                                    OP_SELECT, OP_ISNULL, OP_IDIV, OP_IMOD, OP_IMIN, OP_IMAX, OP_ICMP]


def int_word_program_ok(code: List[Int32], n_ins: Int, agg_base: Int) -> Bool:
    """Can an Int64-word kernel (Apple's GPU) run this program? Its first `agg_base` instructions (after the header)
    must be integer ones, and its aggregates integer sums, mins and maxes. The Mojo API sends float programs to the
    CPU before this (`Program.has_float`); this is the check at the kernel's door."""
    for i in range(n_ins):
        var op = Int(code[HDR + i * 5])
        if i >= agg_base:
            if op != OP_AGG_ISUM and op != OP_AGG_IMIN and op != OP_AGG_IMAX:
                return False
            continue
        var ok = False
        for o in materialize[INT_WORD_OPS]():
            if op == o:
                ok = True
        if not ok:
            return False
    return True


@always_inline
def _key_group[W: DType](word: Scalar[W], groups: Int) -> Int:
    """Keyed mode: register 0 holds the row's group key as int64 bits; a key outside 0 .. groups-1 leaves the row out."""
    var v = _i64_of(word)
    return Int(v) if v >= 0 and v < Int64(groups) else -1


def pipeline_kernel[RPT: Int, B: Int, W: DType](
    code_a_i64: Int64, consts_a_i64: Int64, set_bytes_a_i64: Int64, set_meta_a_i64: Int64,
    cols_a_i64: Int64, dtypes_a_i64: Int64, gids_a_i64: Int64, n_groups_i64: Int64, n_slots_i64: Int64,
    row0_i64: Int64, row_end_i64: Int64, out_vals_a_i64: Int64, out_cnt_a_i64: Int64, scratch_a_i64: Int64,
    n_regs_i64: Int64, code_ints_i64: Int64, mode_i64: Int64, ovf_a_i64: Int64,
):
    var code_a = Int(code_a_i64)
    var consts_a = Int(consts_a_i64)
    var set_bytes_a = Int(set_bytes_a_i64)
    var set_meta_a = Int(set_meta_a_i64)
    var cols_a = Int(cols_a_i64)
    var dtypes_a = Int(dtypes_a_i64)
    var gids_a = Int(gids_a_i64)
    var n_groups = Int(n_groups_i64)
    var n_slots = Int(n_slots_i64)
    var row0 = Int(row0_i64)
    var row_end = Int(row_end_i64)
    var out_vals_a = Int(out_vals_a_i64)
    var out_cnt_a = Int(out_cnt_a_i64)
    var scratch_a = Int(scratch_a_i64)
    var n_regs = Int(n_regs_i64)
    var code_ints = Int(code_ints_i64)
    var mode = Int(mode_i64)
    var ovf = _lp(Int(ovf_a_i64))
    comptime TILE = B * RPT
    comptime NWARPS = B // 32
    var tid = Int(thread_idx.x)
    var G = n_groups
    var A = n_slots

    # dynamic shared memory: [register file | constants | warp partials | program]
    # sized at launch by _smem_bytes; bound: SMEM_LIMIT (run_pipeline_gpu refuses a larger launch with -5)
    var smem = external_memory[Scalar[W], address_space=SH, alignment=16]()
    var regs = smem
    var consts = smem + n_regs * TILE
    var wpart = consts + MAX_CONSTS
    var accs = wpart + NWARPS * SMALL     # per-thread accumulators in shared memory (mode 3)
    var acc_words = (n_slots + 1) * n_groups * B if mode == 3 else ((n_slots + 1) * n_groups if (mode == 1 or mode == 4) else 0)
    var tg = (accs + acc_words).bitcast[Int32]()      # mode 4: this tile's group id per row (-1: not counted)
    if mode == 4:
        acc_words += TILE // 2
    comptime if W != F64:
        if mode == 5:
            acc_words += TILE // 2
    var code = (accs + acc_words).bitcast[Int32]()
    var i = tid
    while i < code_ints:
        code[i] = _ip(code_a)[i]
        i += B
    i = tid
    while i < MAX_CONSTS:
        consts[i] = _wp[W](consts_a)[i]
        i += B
    barrier()

    var n_pro = Int(code[2])
    var n_pred = Int(code[3])
    var n_body = Int(code[4])
    var pred_reg = Int(code[6])
    var agg_base = n_pro + n_pred + n_body
    var cols = _lp(cols_a)
    var dts = _ip(dtypes_a)
    var set_bytes = _bp(set_bytes_a)
    var set_meta = _lp(set_meta_a)
    var keyed = gids_a == -1                     # keyed mode (F-137): the group id comes from register 0 of the body
    var gids = _ip(gids_a) if not keyed else _ip(0)
    var has_gids = gids_a != 0 and not keyed
    var out_vals = _wp[W](out_vals_a)
    var out_cnt = _lp(out_cnt_a)

    var all_rows = (1 << RPT) - 1
    _run_w[RPT, B, W](code, 0, n_pro, regs, consts, cols, dts, set_bytes, set_meta, tid, 0, all_rows, ovf)

    var acc = stack_allocation[SMALL, Scalar[W]]()
    var ns = (A + 1) * G
    if mode == 0:
        for a in range(A):
            var init = _ident[W](Int(code[HDR + (agg_base + a) * 5]))
            for g in range(G):
                acc[a * G + g] = init
        for g in range(G):
            acc[A * G + g] = 0
    elif mode == 3:
        for a in range(A):
            var init = _ident[W](Int(code[HDR + (agg_base + a) * 5]))
            for g in range(G):
                accs[(a * G + g) * B + tid] = init
        for g in range(G):
            accs[(A * G + g) * B + tid] = 0
    elif mode == 1 or mode == 4:
        var j = tid
        while j < ns:
            accs[j] = _ident[W](Int(code[HDR + (agg_base + j // G) * 5])) if j < A * G else 0
            j += B
        barrier()

    comptime if W != F64:
        if mode == 5:                            # the block's table in device memory, each group by its owner
            var row = _lp(scratch_a) + Int(block_idx.x) * ns
            var g = tid
            while g < G:
                for a in range(A):
                    row[a * G + g] = _i64_of(_ident[W](Int(code[HDR + (agg_base + a) * 5])))
                row[A * G + g] = 0
                g += B

    var n_tiles = ceildiv(row_end - row0, TILE)
    var tile = Int(block_idx.x)
    var step = Int(grid_dim.x)
    while tile < n_tiles:
        var row_base = row0 + tile * TILE
        var valid = 0
        comptime for k in range(RPT):
            if row_base + tid + k * B < row_end:
                valid |= 1 << k
        var active = valid
        if pred_reg >= 0:
            _run_w[RPT, B, W](code, n_pro, n_pred, regs, consts, cols, dts, set_bytes, set_meta,
                      tid, row_base, valid, ovf)
            active = 0
            comptime for k in range(RPT):
                if (valid >> k) & 1 != 0 and _nz(regs[pred_reg * TILE + tid + k * B]):
                    active |= 1 << k
        if mode == 4:
            comptime for k in range(RPT):
                tg[tid + k * B] = -1
        comptime if W != F64:
            if mode == 5:
                comptime for k in range(RPT):
                    tg[tid + k * B] = -1
        if active != 0:
            _run_w[RPT, B, W](code, n_pro + n_pred, n_body, regs, consts, cols, dts, set_bytes,
                      set_meta, tid, row_base, active, ovf)
            if mode == 4:
                comptime for k in range(RPT):
                    if (active >> k) & 1 != 0:
                        var g = 0
                        if keyed:
                            g = _key_group(regs[tid + k * B], G)
                        elif has_gids:
                            g = Int(gids[row_base + tid + k * B])
                        if g >= 0 and g < G:
                            tg[tid + k * B] = Int32(g)
            comptime if W != F64:
                if mode == 5:
                    comptime for k in range(RPT):
                        if (active >> k) & 1 != 0:
                            var g = 0
                            if keyed:
                                g = _key_group(regs[tid + k * B], G)
                            elif has_gids:
                                g = Int(gids[row_base + tid + k * B])
                            if g >= 0 and g < G:
                                tg[tid + k * B] = Int32(g)
            comptime for k in range(RPT):
                if (active >> k) & 1 != 0 and mode != 4:   # (mode 5 on the Int64 word lands in the atomic modes' branch, empty there)
                    var g = 0
                    if keyed:
                        g = _key_group(regs[tid + k * B], G)
                    elif has_gids:
                        g = Int(gids[row_base + tid + k * B])
                    if g >= 0 and g < G:
                        if mode == 0:
                            acc[A * G + g] += 1
                            for a in range(A):
                                var ins = code + (HDR + (agg_base + a) * 5)
                                var kind = Int(ins[0])
                                var v = regs[Int(ins[2]) * TILE + tid + k * B]
                                var q = a * G + g
                                if kind == OP_AGG_SUM:
                                    acc[q] += v
                                elif kind == OP_AGG_ISUM:
                                    acc[q] = _iadd_value(acc[q], v, ovf)
                                elif kind == OP_AGG_IMIN or kind == OP_AGG_IMAX:
                                    acc[q] = _iext_words(kind, acc[q], v)
                                elif kind == OP_AGG_MIN:
                                    acc[q] = min(acc[q], v)
                                else:
                                    acc[q] = max(acc[q], v)
                        elif mode == 3:
                            accs[(A * G + g) * B + tid] += 1
                            for a in range(A):
                                var ins = code + (HDR + (agg_base + a) * 5)
                                var kind = Int(ins[0])
                                var v = regs[Int(ins[2]) * TILE + tid + k * B]
                                var p = accs + ((a * G + g) * B + tid)
                                if kind == OP_AGG_SUM:
                                    p[0] += v
                                elif kind == OP_AGG_ISUM:
                                    p[0] = _iadd_value(p[0], v, ovf)
                                elif kind == OP_AGG_IMIN or kind == OP_AGG_IMAX:
                                    p[0] = _iext_words(kind, p[0], v)
                                elif kind == OP_AGG_MIN:
                                    p[0] = min(p[0], v)
                                else:
                                    p[0] = max(p[0], v)
                        else:
                            comptime if W == F64:
                                var faccs = rebind[SFP](accs)
                                var fvals = rebind[FP](out_vals)
                                if mode == 1:
                                    _atomic_add_f64(faccs + (A * G + g), 1.0)
                                    for a in range(A):
                                        var ins = code + (HDR + (agg_base + a) * 5)
                                        var kind = Int(ins[0])
                                        var v = rebind[Float64](regs[Int(ins[2]) * TILE + tid + k * B])
                                        var p = faccs + (a * G + g)
                                        if kind == OP_AGG_SUM:
                                            _atomic_add_f64(p, v)
                                        elif kind == OP_AGG_ISUM:
                                            var yv = _i64_of(v)
                                            _note_ovf(_atomic_add_bits(p, yv), yv, ovf)
                                        elif kind == OP_AGG_IMIN:
                                            _atomic_min_bits(p, _i64_of(v))
                                        elif kind == OP_AGG_IMAX:
                                            _atomic_max_bits(p, _i64_of(v))
                                        elif kind == OP_AGG_MIN:
                                            _atomic_min_f64(p, v)
                                        else:
                                            _atomic_max_f64(p, v)
                                else:
                                    _atomic_add_i64(out_cnt + g, Int64(1))
                                    for a in range(A):
                                        var ins = code + (HDR + (agg_base + a) * 5)
                                        var kind = Int(ins[0])
                                        var v = rebind[Float64](regs[Int(ins[2]) * TILE + tid + k * B])
                                        var p = fvals + (a * G + g)
                                        if kind == OP_AGG_SUM:
                                            _atomic_add_f64(p, v)
                                        elif kind == OP_AGG_ISUM:
                                            var yv = _i64_of(v)
                                            _note_ovf(_atomic_add_bits(p, yv), yv, ovf)
                                        elif kind == OP_AGG_IMIN:
                                            _atomic_min_bits(p, _i64_of(v))
                                        elif kind == OP_AGG_IMAX:
                                            _atomic_max_bits(p, _i64_of(v))
                                        elif kind == OP_AGG_MIN:
                                            _atomic_min_f64(p, v)
                                        else:
                                            _atomic_max_f64(p, v)
        if mode == 4:
            # the owner of group g walks this tile's rows in row order (no two threads add to one slot)
            barrier()
            var g = tid
            while g < G:
                for r in range(TILE):
                    if Int(tg[r]) == g:
                        accs[A * G + g] += 1
                        for a in range(A):
                            var ins = code + (HDR + (agg_base + a) * 5)
                            var kind = Int(ins[0])
                            var v = regs[Int(ins[2]) * TILE + r]
                            var p = accs + (a * G + g)
                            if kind == OP_AGG_SUM:
                                p[0] += v
                            elif kind == OP_AGG_ISUM:
                                p[0] = _iadd_value(p[0], v, ovf)
                            elif kind == OP_AGG_IMIN or kind == OP_AGG_IMAX:
                                p[0] = _iext_words(kind, p[0], v)
                            elif kind == OP_AGG_MIN:
                                p[0] = min(p[0], v)
                            else:
                                p[0] = max(p[0], v)
                g += B
            barrier()
        comptime if W != F64:
            if mode == 5:
                # each row's group is added by its owner (thread g % B), rows in row order
                barrier()
                var row = _lp(scratch_a) + Int(block_idx.x) * ns
                for r in range(TILE):
                    var g = Int(tg[r])
                    if g >= 0 and (g & (B - 1)) == tid:
                        row[A * G + g] += 1
                        for a in range(A):
                            var ins = code + (HDR + (agg_base + a) * 5)
                            var kind = Int(ins[0])
                            var v = _i64_of(regs[Int(ins[2]) * TILE + r])
                            var q = a * G + g
                            if kind == OP_AGG_ISUM:
                                _note_ovf(row[q], v, ovf)
                                row[q] = row[q] + v
                            elif kind == OP_AGG_IMIN:
                                row[q] = min(row[q], v)
                            elif kind == OP_AGG_IMAX:
                                row[q] = max(row[q], v)
                barrier()
        tile += step

    if mode == 4:
        # the block's table is its row of `scratch`; merge_kernel adds the rows in block order
        var j = tid
        while j < ns:
            _wp[W](scratch_a)[Int(block_idx.x) * ns + j] = accs[j]
            j += B
    elif mode == 0 or mode == 3:
        # reduce the block's per-thread partials: warp shuffles, then across warps
        var lane = tid & 31
        var warp = tid >> 5
        for s_ in range(ns):
            var kind = OP_AGG_SUM
            if s_ < A * G:
                kind = Int(code[HDR + (agg_base + s_ // G) * 5])
            var w = _warp_reduce(kind, acc[s_] if mode == 0 else accs[s_ * B + tid], ovf)
            if lane == 0:
                wpart[warp * SMALL + s_] = w
        barrier()
        if tid < ns:
            var kind = OP_AGG_SUM
            if tid < A * G:
                kind = Int(code[HDR + (agg_base + tid // G) * 5])
            var r = wpart[tid]
            for w in range(1, NWARPS):
                r = _combine(kind, r, wpart[w * SMALL + tid], ovf)
            _wp[W](scratch_a)[Int(block_idx.x) * ns + tid] = r
    else:
        comptime if W == F64:
            var faccs = rebind[SFP](accs)
            var fvals = rebind[FP](out_vals)
            if mode == 1:
                barrier()
                var j = tid
                while j < ns:
                    var kind = OP_AGG_SUM
                    if j < A * G:
                        kind = Int(code[HDR + (agg_base + j // G) * 5])
                    var v = faccs[j]
                    if j >= A * G:
                        if v != 0.0:
                            _atomic_add_i64(out_cnt + (j - A * G), Int64(v))
                    elif kind == OP_AGG_ISUM:
                        if _i64_of(v) != 0:
                            var yv = _i64_of(v)
                            _note_ovf(_atomic_add_bits(fvals + j, yv), yv, ovf)
                    elif kind == OP_AGG_SUM:
                        if v != 0.0:
                            _atomic_add_f64(fvals + j, v)
                    elif kind == OP_AGG_IMIN:
                        if _i64_of(v) != Int64.MAX:
                            _atomic_min_bits(fvals + j, _i64_of(v))
                    elif kind == OP_AGG_IMAX:
                        if _i64_of(v) != Int64.MIN:
                            _atomic_max_bits(fvals + j, _i64_of(v))
                    elif kind == OP_AGG_MIN:
                        if v != Float64.MAX:
                            _atomic_min_f64(fvals + j, v)
                    else:
                        if v != -Float64.MAX:
                            _atomic_max_f64(fvals + j, v)
                    j += B


@always_inline
def _combine_f64(kind: Int, a: Float64, b: Float64, ovf: LP) -> Float64:
    """`_combine` on float64 words exactly as it was before the word became a parameter: NVIDIA's merge kernel uses
    it so its PTX stays the pinned one (ci/mac/nvidia_ptx.sha)."""
    if kind == OP_AGG_MIN:
        return min(a, b)
    elif kind == OP_AGG_MAX:
        return max(a, b)
    elif kind == OP_AGG_ISUM:
        var x = bitcast[DType.int64, 1](SIMD[F64, 1](a))[0]
        var y = bitcast[DType.int64, 1](SIMD[F64, 1](b))[0]
        _note_ovf(x, y, ovf)
        return bitcast[F64, 1](SIMD[DType.int64, 1](x + y))[0]
    elif kind == OP_AGG_IMIN or kind == OP_AGG_IMAX:
        var x = bitcast[DType.int64, 1](SIMD[F64, 1](a))[0]
        var y = bitcast[DType.int64, 1](SIMD[F64, 1](b))[0]
        return bitcast[F64, 1](SIMD[DType.int64, 1](min(x, y) if kind == OP_AGG_IMIN else max(x, y)))[0]
    return a + b


def init_kernel[W: DType](out_vals_a_i64: Int64, out_cnt_a_i64: Int64, ident_a_i64: Int64, n_groups_i64: Int64, total_i64: Int64):
    var out_vals_a = Int(out_vals_a_i64)
    var out_cnt_a = Int(out_cnt_a_i64)
    var ident_a = Int(ident_a_i64)
    var n_groups = Int(n_groups_i64)
    var total = Int(total_i64)
    var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    if t < total:
        _wp[W](out_vals_a)[t] = _wp[W](ident_a)[t // n_groups]
    if t < n_groups:
        _lp(out_cnt_a)[t] = 0


def merge_kernel(
    code_a_i64: Int64, scratch_a_i64: Int64, out_vals_a_i64: Int64, out_cnt_a_i64: Int64, n_blocks_i64: Int64,
    n_groups_i64: Int64, n_slots_i64: Int64, agg_base_i64: Int64, ident_a_i64: Int64, ovf_a_i64: Int64,
):
    var code_a = Int(code_a_i64)
    var scratch_a = Int(scratch_a_i64)
    var out_vals_a = Int(out_vals_a_i64)
    var out_cnt_a = Int(out_cnt_a_i64)
    var n_blocks = Int(n_blocks_i64)
    var n_groups = Int(n_groups_i64)
    var n_slots = Int(n_slots_i64)
    var agg_base = Int(agg_base_i64)
    var ident_a = Int(ident_a_i64)
    var ovf = _lp(Int(ovf_a_i64))
    """Merge the per-block partials in block order (deterministic)."""
    var s = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    var A = n_slots
    var G = n_groups
    var ns = (A + 1) * G
    if s >= ns:
        return
    var kind = OP_AGG_SUM
    var r = 0.0
    if s < A * G:
        kind = Int(_ip(code_a)[HDR + (agg_base + s // G) * 5])
        r = _fp(ident_a)[s // G]
    for b in range(n_blocks):
        r = _combine_f64(kind, r, _fp(scratch_a)[b * ns + s], ovf)
    if s < A * G:
        _fp(out_vals_a)[s] = r
    else:
        _lp(out_cnt_a)[s - A * G] = Int64(r)


def merge_kernel_int(
    code_a_i64: Int64, scratch_a_i64: Int64, out_vals_a_i64: Int64, out_cnt_a_i64: Int64, n_blocks_i64: Int64,
    n_groups_i64: Int64, n_slots_i64: Int64, agg_base_i64: Int64, ident_a_i64: Int64, ovf_a_i64: Int64,
):
    var code_a = Int(code_a_i64)
    var scratch_a = Int(scratch_a_i64)
    var out_vals_a = Int(out_vals_a_i64)
    var out_cnt_a = Int(out_cnt_a_i64)
    var n_blocks = Int(n_blocks_i64)
    var n_groups = Int(n_groups_i64)
    var n_slots = Int(n_slots_i64)
    var agg_base = Int(agg_base_i64)
    var ident_a = Int(ident_a_i64)
    var ovf = _lp(Int(ovf_a_i64))
    """Merge the per-block partials in block order (deterministic)."""
    var s = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    var A = n_slots
    var G = n_groups
    var ns = (A + 1) * G
    if s >= ns:
        return
    var kind = OP_AGG_SUM
    var r = Int64(0)
    if s < A * G:
        kind = Int(_ip(code_a)[HDR + (agg_base + s // G) * 5])
        r = _lp(ident_a)[s // G]
    for b in range(n_blocks):
        r = _combine(kind, r, _lp(scratch_a)[b * ns + s], ovf)
    if s < A * G:
        _lp(out_vals_a)[s] = r
    else:
        _lp(out_cnt_a)[s - A * G] = r


def _rows_per_thread(n_regs: Int, n_groups: Int, n_slots: Int) -> Int:
    return 2 if (n_regs <= 6 and n_groups == 1 and n_slots <= 2) else 1


def _block_threads(mode: Int) -> Int:
    return 128 if mode == 3 else BLOCK


def _smem_bytes(mode: Int, n_groups: Int, n_slots: Int, n_regs: Int, code_ints: Int, narrow: Bool = False) -> Int:
    """Dynamic shared memory of a launch in `mode`: the one formula the mode choice and the launch share. `narrow`
    (the Mojo API's big programs): NARROW_B-thread blocks (128; 64 on an Apple GPU), a smaller register file per block."""
    var ns = (n_slots + 1) * n_groups
    var bsz = NARROW_B if narrow else _block_threads(mode)
    var tile = bsz * _rows_per_thread(n_regs, n_groups, n_slots)
    var smem = n_regs * tile * 8 + MAX_CONSTS * 8 + code_ints * 4 + (bsz // 32) * SMALL * 8
    if mode == 3:
        smem += ns * bsz * 8
    elif mode == 1:
        smem += ns * 8
    elif mode == 4:
        smem += ns * 8 + tile * 4
    elif mode == 5:
        smem += tile * 4
    return smem


def _choose_mode(n_groups: Int, n_slots: Int, n_regs: Int, code_ints: Int, repro_asked: Int, allow_narrow: Bool = False) -> Int:
    """The aggregation mode (see the header) a program with these sizes runs in, or -5 (needs more shared
    memory than a block has) / -6 (reproducible mode and the group table does not fit). On an Apple GPU the
    atomic-free modes only (as if reproducible mode were asked: modes 1 and 2 need 64-bit atomics, which Metal lacks),
    within its 32 KB of shared memory."""
    var repro = repro_asked
    comptime if APPLE_GPU:
        repro = 1
    var G = n_groups
    var ns = (n_slots + 1) * G
    var mode = 0 if G == 1 and ns <= SMALL else 2
    if G > 1:
        # few groups: per-thread accumulators in shared memory (128-thread blocks); otherwise one shared
        # table of accumulators per block, folded into the output with global atomics, if it fits
        if ns <= SMALL and _smem_bytes(3, G, n_slots, n_regs, code_ints) <= SMEM_LIMIT:
            mode = 3
        elif _smem_bytes(1, G, n_slots, n_regs, code_ints) <= SMEM_LIMIT:
            mode = 1
    if repro != 0 and (mode == 1 or mode == 2):
        # no atomics: the ordered per-block table (mode 4), if the group count fits in shared memory (in
        # 128-thread blocks when the caller allows them: on an Apple GPU, big programs have no other mode)
        var fits4 = ns <= REPRO_MAX_NS and (_smem_bytes(4, G, n_slots, n_regs, code_ints) <= SMEM_LIMIT or
                                            (allow_narrow and _smem_bytes(4, G, n_slots, n_regs, code_ints, True) <= SMEM_LIMIT))
        if fits4:
            mode = 4
        else:
            comptime if APPLE_GPU:               # many groups on the Int64 word: the table in device memory (mode 5)
                if G > 1 and ns <= MANY_MAX_NS and (_smem_bytes(5, G, n_slots, n_regs, code_ints) <= SMEM_LIMIT or
                                                    (allow_narrow and _smem_bytes(5, G, n_slots, n_regs, code_ints, True) <= SMEM_LIMIT)):
                    mode = 5
                else:
                    return -6
            else:
                return -6
    if _smem_bytes(mode, G, n_slots, n_regs, code_ints) > SMEM_LIMIT:
        if allow_narrow and _smem_bytes(mode, G, n_slots, n_regs, code_ints, True) <= SMEM_LIMIT:
            return mode
        return -5
    return mode


def scratch_rows(mode: Int, n_groups: Int, n_slots: Int, n_blocks: Int) -> Int:
    """How many per-block rows of (slots + 1) * groups words `scratch` needs for a launch in `mode` on a device with
    `n_blocks` cores (0 for the modes that write the output directly). The launch never uses more blocks than this
    (`_grid_blocks`), so the two always agree (mojo/tests/test_gpu_plan.mojo)."""
    if mode == 0 or mode == 3 or mode == 4:
        return max(n_blocks, 1) * 8
    if mode == 5:
        return max(1, min(max(n_blocks, 1) * 8, MANY_SCRATCH_WORDS // ((n_slots + 1) * n_groups)))
    return 0


def _grid_blocks(mode: Int, n_groups: Int, n_slots: Int, n_regs: Int, code_ints: Int, narrow: Bool, rows: Int, n_blocks: Int) -> Int:
    """The blocks a launch of `rows` rows in `mode` uses: one per tile up to `per_sm` resident blocks per core, and never
    more than `scratch_rows` in the modes that keep per-block rows."""
    var bsz = NARROW_B if narrow else _block_threads(mode)
    var rpt = 1 if narrow else _rows_per_thread(n_regs, n_groups, n_slots)
    var smem = _smem_bytes(mode, n_groups, n_slots, n_regs, code_ints, narrow)
    var per_sm = max(1, min(8, (100 * 1024) // smem, 1536 // bsz))
    var blocks = min(ceildiv(rows, bsz * rpt), max(n_blocks, 1) * per_sm)
    if mode == 5:
        blocks = min(blocks, scratch_rows(mode, n_groups, n_slots, n_blocks))
    return max(blocks, 1)


def _narrow(mode: Int, n_groups: Int, n_slots: Int, n_regs: Int, code_ints: Int) -> Bool:
    """The Mojo API's programs keep many registers (a pivot with ten aggregates needs 23): when 256-thread blocks do
    not fit in shared memory, 128-thread blocks (rpt 1) may. Mode 3 already has its own shape; mode 4 narrows only
    where it was chosen with narrow blocks allowed (an Apple GPU: the Mojo API never asks NVIDIA for mode 4)."""
    return mode != 3 and _smem_bytes(mode, n_groups, n_slots, n_regs, code_ints) > SMEM_LIMIT and \
        _smem_bytes(mode, n_groups, n_slots, n_regs, code_ints, True) <= SMEM_LIMIT


@export
def mx_pipeline_gpu_mode(n_groups: Int, n_slots: Int, n_regs: Int, code_ints: Int, repro: Int) abi("C") -> Int:
    """The mode ``mx_pipeline_gpu`` would run (0 to 4), or a negative status: Python asks this before it
    splits rows, so a reproducible split never hands the GPU a share it would decline."""
    return _choose_mode(n_groups, n_slots, n_regs, code_ints, repro)


def run_pipeline_gpu(
    ctx: DeviceContext,
    code_a: Int, consts_a: Int, set_bytes_a: Int, set_meta_a: Int,
    cols_a: Int, dtypes_a: Int, gids_a: Int, n_groups: Int, n_slots: Int,
    start: Int, end: Int, out_vals_a: Int, out_cnt_a: Int, ident_a: Int,
    scratch_a: Int, n_regs: Int, code_ints: Int, agg_base: Int,
    n_blocks: Int, repeat: Int, repro: Int, ovf_a: Int, narrow_ok: Bool = False,
) raises -> Int:
    """`mx_pipeline_gpu` on a DeviceContext the caller owns (F-137): the Mojo API runs the kernel on the app's
    context, so its buffers, the app's own kernels and this one share one stream and need no extra synchronization.
    `narrow_ok`: the Mojo API's programs keep many registers, so a launch that does not fit 256-thread blocks in
    shared memory may use 128-thread ones (the Python path never asks: it declines with -5 as before).
    `gids_a` == -1 selects keyed mode: the group id of each row is register 0 of the body (int64 bits), an id
    outside 0 .. n_groups-1 leaves the row out. Returns 0 or a negative status (see `mx_pipeline_gpu`)."""
    if n_slots < 0 or n_groups < 1:
        return -2
    var G = n_groups
    var A = n_slots
    if gids_a == 0 and G != 1:
        return -2
    var mode = _choose_mode(G, A, n_regs, code_ints, repro, narrow_ok)
    if mode < 0:
        return mode
    var narrow = narrow_ok and _narrow(mode, G, A, n_regs, code_ints)
    var bsz = NARROW_B if narrow else _block_threads(mode)
    var rpt = 1 if narrow else _rows_per_thread(n_regs, G, A)
    var smem = _smem_bytes(mode, G, A, n_regs, code_ints, narrow)
    if smem > SMEM_LIMIT:            # _choose_mode keeps every plan within it; never launch past the target's limit
        return -5
    var small = 1 if (mode == 0 or mode == 3 or mode == 4 or mode == 5) else 0
    var total = A * G
    if small == 0 or end <= start:
        ctx.enqueue_function[init_kernel[WORD]](
            Int64(out_vals_a), Int64(out_cnt_a), Int64(ident_a), Int64(G), Int64(total),
            grid_dim=ceildiv(max(total, G), BLOCK), block_dim=BLOCK)
    if end > start:
        var blocks = _grid_blocks(mode, G, A, n_regs, code_ints, narrow, end - start, n_blocks)
        for _ in range(max(repeat, 1)):
            comptime if NARROW_B == 64:                   # Apple: narrow blocks of 64 threads
                if bsz == 64:
                    ctx.enqueue_function[pipeline_kernel[1, 64, WORD]](
                        Int64(code_a), Int64(consts_a), Int64(set_bytes_a), Int64(set_meta_a), Int64(cols_a), Int64(dtypes_a), Int64(gids_a),
                        Int64(G), Int64(A), Int64(start), Int64(end), Int64(out_vals_a), Int64(out_cnt_a), Int64(scratch_a), Int64(n_regs), Int64(code_ints),
                        Int64(mode), Int64(ovf_a), grid_dim=blocks, block_dim=bsz, shared_mem_bytes=smem)
                    continue
            if bsz == 128 and rpt == 2:
                ctx.enqueue_function[pipeline_kernel[2, 128, WORD]](
                    Int64(code_a), Int64(consts_a), Int64(set_bytes_a), Int64(set_meta_a), Int64(cols_a), Int64(dtypes_a), Int64(gids_a),
                    Int64(G), Int64(A), Int64(start), Int64(end), Int64(out_vals_a), Int64(out_cnt_a), Int64(scratch_a), Int64(n_regs), Int64(code_ints),
                    Int64(mode), Int64(ovf_a), grid_dim=blocks, block_dim=bsz, shared_mem_bytes=smem)
            elif bsz == 128:
                ctx.enqueue_function[pipeline_kernel[1, 128, WORD]](
                    Int64(code_a), Int64(consts_a), Int64(set_bytes_a), Int64(set_meta_a), Int64(cols_a), Int64(dtypes_a), Int64(gids_a),
                    Int64(G), Int64(A), Int64(start), Int64(end), Int64(out_vals_a), Int64(out_cnt_a), Int64(scratch_a), Int64(n_regs), Int64(code_ints),
                    Int64(mode), Int64(ovf_a), grid_dim=blocks, block_dim=bsz, shared_mem_bytes=smem)
            elif rpt == 2:
                ctx.enqueue_function[pipeline_kernel[2, 256, WORD]](
                    Int64(code_a), Int64(consts_a), Int64(set_bytes_a), Int64(set_meta_a), Int64(cols_a), Int64(dtypes_a), Int64(gids_a),
                    Int64(G), Int64(A), Int64(start), Int64(end), Int64(out_vals_a), Int64(out_cnt_a), Int64(scratch_a), Int64(n_regs), Int64(code_ints),
                    Int64(mode), Int64(ovf_a), grid_dim=blocks, block_dim=bsz, shared_mem_bytes=smem)
            else:
                ctx.enqueue_function[pipeline_kernel[1, 256, WORD]](
                    Int64(code_a), Int64(consts_a), Int64(set_bytes_a), Int64(set_meta_a), Int64(cols_a), Int64(dtypes_a), Int64(gids_a),
                    Int64(G), Int64(A), Int64(start), Int64(end), Int64(out_vals_a), Int64(out_cnt_a), Int64(scratch_a), Int64(n_regs), Int64(code_ints),
                    Int64(mode), Int64(ovf_a), grid_dim=blocks, block_dim=bsz, shared_mem_bytes=smem)
        if small != 0:
            comptime if WORD == F64:
                ctx.enqueue_function[merge_kernel](
                    Int64(code_a), Int64(scratch_a), Int64(out_vals_a), Int64(out_cnt_a), Int64(blocks), Int64(G), Int64(A), Int64(agg_base), Int64(ident_a), Int64(ovf_a),
                    grid_dim=ceildiv((A + 1) * G, BLOCK), block_dim=BLOCK)
            else:
                ctx.enqueue_function[merge_kernel_int](
                    Int64(code_a), Int64(scratch_a), Int64(out_vals_a), Int64(out_cnt_a), Int64(blocks), Int64(G), Int64(A), Int64(agg_base), Int64(ident_a), Int64(ovf_a),
                    grid_dim=ceildiv((A + 1) * G, BLOCK), block_dim=BLOCK)
    ctx.synchronize()
    return 0


@export
def mx_pipeline_gpu(
    code_a: Int, consts_a: Int, set_bytes_a: Int, set_meta_a: Int,
    cols_a: Int, dtypes_a: Int, gids_a: Int, n_groups: Int, n_slots: Int,
    start: Int, end: Int, out_vals_a: Int, out_cnt_a: Int, ident_a: Int,
    scratch_a: Int, n_regs: Int, code_ints: Int, agg_base: Int,
    n_blocks: Int, repeat: Int, repro: Int, ovf_a: Int,
) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        return run_pipeline_gpu(ctx, code_a, consts_a, set_bytes_a, set_meta_a, cols_a, dtypes_a, gids_a, n_groups, n_slots,
                                start, end, out_vals_a, out_cnt_a, ident_a, scratch_a, n_regs, code_ints, agg_base,
                                n_blocks, repeat, repro, ovf_a)
    except:
        return -1
