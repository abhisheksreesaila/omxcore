"""ctypes bindings for the fused pipeline kernels (F-008).

``libmxpipeline_cpu.so`` and ``libmxpipeline_gpu.so`` are built ahead of time
from ``kernels_aot/pipeline_{cpu,gpu}.mojo`` (see ``scripts/build_pipeline.sh``);
nothing is compiled at run time. Both libraries are loaded lazily, and the GPU
library and the CUDA driver are touched only when a GPU function is called.

Entry points return :class:`~mxframe.pipeline_program.Partials` for a row range
``[start, end)`` so a Router can hand morsels to either device and merge the
results with :func:`merge_partials`.
"""

from __future__ import annotations

import ctypes
import os
import threading
import weakref
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from .pipeline_program import (
    Buf,
    ChunkedColumns,
    Partials,
    PipelineProgram,
    SumOverflow,
    dtype_code,
    empty_partials,
    finalize,
    merge_partials,
)

__all__ = [
    "cpu_available", "gpu_available", "default_threads", "run_pipeline_cpu",
    "run_pipeline_gpu", "GpuUnsupported", "DeviceColumns", "upload_columns", "merge_partials", "finalize",
]

_HERE = Path(__file__).parent
_lock = threading.Lock()
_cpu_lib = None
_gpu_lib = None

_I64 = ctypes.c_int64
E_OVERFLOW = -21          # the kernels' code for an integer sum past int64 (pipeline_cpu.mojo)


class GpuUnsupported(Exception):
    """This program or shape cannot run on the GPU kernel (the caller should use the CPU kernel)."""


def _find(name: str, env: str) -> Optional[Path]:
    override = os.environ.get(env)
    if override and Path(override).exists():
        return Path(override)
    for cand in (_HERE / "kernels_aot" / name, _HERE / name):
        if cand.exists():
            return cand
    return None


def cpu_available() -> bool:
    return _find("libmxpipeline_cpu.so", "MXFRAME_PIPELINE_CPU_LIB") is not None


def gpu_available() -> bool:
    return _find("libmxpipeline_gpu.so", "MXFRAME_PIPELINE_GPU_LIB") is not None


_PHYSICAL = None


def physical_cores() -> int:
    """Physical cores this process may run on (hyperthread siblings counted once); the logical CPU
    count when the topology is not readable. The kernels are memory-bound: a second hyperthread
    per core adds contention, not throughput (F-032 M2k: SF10 Q21 3.78 -> 3.54 s on 12 of 24)."""
    global _PHYSICAL
    if _PHYSICAL is None:
        try:
            cpus = os.sched_getaffinity(0)
        except (AttributeError, OSError):
            cpus = range(os.cpu_count() or 1)
        cores = set()
        try:
            for c in cpus:
                with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as f:
                    cores.add(f.read().strip())
        except OSError:
            cores = set(cpus)
        _PHYSICAL = max(1, len(cores))
    return _PHYSICAL


def default_threads(rows: int = None) -> int:
    """Worker threads for a kernel: MXFRAME_CPU_THREADS, else one per physical core, and fewer for
    a small input (``rows``): one per 128k rows, so a 800k-row table does not wake every core."""
    env = os.environ.get("MXFRAME_CPU_THREADS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    n = physical_cores()
    if rows is not None:
        n = min(n, max(1, rows >> 17))
    return n


def _load_cpu():
    global _cpu_lib
    with _lock:
        if _cpu_lib is None:
            path = _find("libmxpipeline_cpu.so", "MXFRAME_PIPELINE_CPU_LIB")
            if path is None:
                raise FileNotFoundError("libmxpipeline_cpu.so not found")
            lib = ctypes.CDLL(str(path))
            fn = lib.mx_pipeline_cpu
            fn.restype = _I64
            fn.argtypes = [_I64] * 15
            _cpu_lib = lib
    return _cpu_lib


def _reproducible_now() -> bool:
    from . import native_exec
    return native_exec.reproducible()


def _addr(a: np.ndarray) -> int:
    return int(a.ctypes.data)


def _as_chunked(program: PipelineProgram, columns) -> ChunkedColumns:
    """Normalise ``columns`` (numpy arrays, or already chunked) to :class:`ChunkedColumns`."""
    if isinstance(columns, ChunkedColumns):
        cc = columns
    else:
        arrays = []
        for a in columns:
            if not isinstance(a, np.ndarray) or not a.flags["C_CONTIGUOUS"]:
                a = np.ascontiguousarray(a)
            arrays.append(a)
        cc = ChunkedColumns.from_arrays(arrays)
        if not arrays:
            cc = ChunkedColumns([], [], 0)
    if cc.n_cols != len(program.columns):
        raise ValueError(f"program reads {len(program.columns)} columns, got {cc.n_cols}")
    return cc


def _check_gids(gids: Optional[np.ndarray], n_groups: int) -> Optional[np.ndarray]:
    if gids is not None:
        if gids.dtype != np.int32 or not gids.flags["C_CONTIGUOUS"]:
            gids = np.ascontiguousarray(gids, dtype=np.int32)
    elif n_groups != 1:
        raise ValueError("gids may only be omitted for a global aggregate (n_groups == 1)")
    return gids


def _column_dtypes(cc: ChunkedColumns) -> List[int]:
    if not cc.chunks:
        return [0] * cc.n_cols
    return [b.dt for b in cc.chunks[0]]


def _chunk_table(cc: ChunkedColumns, n_rows: int):
    """The kernel's chunk table (row range and per-column base address per chunk), built once per binding."""
    cached = getattr(cc, "_cpu_table", None)
    if cached is not None:
        return cached
    n_cols = cc.n_cols
    if cc.chunks:
        tbl = np.empty((len(cc.bounds), 2 + n_cols), dtype=np.int64)
        for k, ((s, e), arrs) in enumerate(zip(cc.bounds, cc.chunks)):
            tbl[k, 0], tbl[k, 1] = s, e
            for c, b in enumerate(arrs):
                tbl[k, 2 + c] = b.addr - s * b.itemsize
    else:  # no column data (e.g. count-only over gids): one dummy chunk spanning the rows
        tbl = np.array([[0, n_rows]], dtype=np.int64)
    col_dt = np.array(_column_dtypes(cc) or [0], dtype=np.int32)
    if cc.chunks:                       # the dummy table depends on the row count: not cached
        cc._cpu_table = (tbl, col_dt)
    return tbl, col_dt


def run_pipeline_cpu(program: PipelineProgram, columns, gids: Optional[np.ndarray], n_groups: int,
                     start: int, end: int, threads: Optional[int] = None,
                     reproducible: Optional[bool] = None) -> Partials:
    """Run the program over rows ``[start, end)`` on the CPU (all cores by default).

    ``reproducible``: float sums in the defined order whatever the group count (F-064); default: the mode of
    the ``compute()`` running in this thread (``native_exec.reproducible()``). A worker thread started by
    the caller does not inherit it: read it first and pass it in.

    ``columns`` is a sequence of numpy arrays or a :class:`ChunkedColumns`
    (from ``program.bind_chunks(table)``, which avoids concatenating Arrow chunks).
    """
    lib = _load_cpu()
    cc = _as_chunked(program, columns)
    gids = _check_gids(gids, n_groups)
    n_rows = cc.n_rows if cc.n_cols else (len(gids) if gids is not None else end)
    if gids is not None:
        n_rows = min(n_rows, len(gids))
    if start < 0 or end > n_rows or start > end:
        raise ValueError(f"row range [{start}, {end}) outside 0..{n_rows}")
    n_slots = program.n_slots
    vals = np.empty((n_slots, n_groups), dtype=np.float64)
    cnt = np.empty(n_groups, dtype=np.int64)
    tbl, col_dt = _chunk_table(cc, n_rows)
    rc = lib.mx_pipeline_cpu(
        _addr(program.code), _addr(program.consts), _addr(program.set_bytes),
        _addr(program.set_meta), _addr(tbl), int(tbl.shape[0]), _addr(col_dt),
        _addr(gids) if gids is not None else 0, n_groups, start, end,
        _addr(vals), _addr(cnt), int(threads) if threads else default_threads(),
        1 if (reproducible if reproducible is not None else _reproducible_now()) else 0,
    )
    if rc == E_OVERFLOW:
        raise SumOverflow("Overflow in sum of INT64")
    if rc != 0:
        raise RuntimeError(f"mx_pipeline_cpu failed with code {rc}")
    return Partials(vals=vals, cnt=cnt)


# ── GPU ─────────────────────────────────────────────────────────────────────
# The CUDA driver API on the device's primary context, the same context Mojo's
# DeviceContext() uses, so device pointers allocated here are visible in the
# kernels. Nothing below runs (and libcuda is not loaded) until a GPU function
# is called.

class _Cuda:
    """Minimal CUDA driver wrapper (memory + copies + SM count)."""

    def __init__(self):
        lib = None
        for name in ("libcuda.so.1", "libcuda.so"):
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                pass
        if lib is None:
            raise RuntimeError("libcuda.so not found")
        self.lib = lib
        u64, i32, sz, vp = ctypes.c_uint64, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p
        lib.cuInit.argtypes = [ctypes.c_uint]
        err = lib.cuInit(0)
        if err != 0:
            raise RuntimeError(f"cuInit failed: {err}")
        dev = ctypes.c_int(0)
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(i32), i32]
        lib.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(u64), i32]
        lib.cuCtxSetCurrent.argtypes = [u64]
        if lib.cuDeviceGet(ctypes.byref(dev), 0) != 0:
            raise RuntimeError("no CUDA device")
        ctx = u64(0)
        lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
        self.ctx = ctx
        lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(u64), sz]
        lib.cuMemFree_v2.argtypes = [u64]
        lib.cuMemcpyHtoD_v2.argtypes = [u64, vp, sz]
        lib.cuMemcpyDtoH_v2.argtypes = [vp, u64, sz]
        lib.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(i32), i32, i32]
        n = ctypes.c_int(0)
        lib.cuDeviceGetAttribute(ctypes.byref(n), 16, dev)  # MULTIPROCESSOR_COUNT
        self.sm_count = max(int(n.value), 1)
        self._tls = threading.local()
        self._pool_lock = threading.Lock()
        self._pool = {}
        self._sizes = {}

    def bind_thread(self) -> None:
        """Make the primary context current on the calling thread (Router workers)."""
        if not getattr(self._tls, "bound", False):
            self.lib.cuCtxSetCurrent(self.ctx)
            self._tls.bound = True

    def malloc(self, nbytes: int) -> int:
        self.bind_thread()
        ptr = ctypes.c_uint64(0)
        err = self.lib.cuMemAlloc_v2(ctypes.byref(ptr), max(int(nbytes), 8))
        if err != 0:
            if err == 2:
                raise MemoryError(f"CUDA_ERROR_OUT_OF_MEMORY: cuMemAlloc({nbytes}) failed")
            raise RuntimeError(f"CUDA_ERROR {err}: cuMemAlloc({nbytes}) failed")
        return int(ptr.value)

    def free(self, ptr: int) -> None:
        if ptr:
            self.bind_thread()
            self.lib.cuMemFree_v2(ctypes.c_uint64(ptr))

    def h2d(self, dev_ptr: int, arr: np.ndarray) -> None:
        self.bind_thread()
        if arr.nbytes:
            err = self.lib.cuMemcpyHtoD_v2(ctypes.c_uint64(dev_ptr), ctypes.c_void_p(arr.ctypes.data),
                                           ctypes.c_size_t(arr.nbytes))
            if err != 0:
                raise RuntimeError(f"CUDA_ERROR {err}: cuMemcpyHtoD failed")

    def h2d_addr(self, dev_ptr: int, host_addr: int, nbytes: int) -> None:
        """Copy ``nbytes`` from a host address (an Arrow buffer, say) to the device."""
        self.bind_thread()
        if nbytes:
            err = self.lib.cuMemcpyHtoD_v2(ctypes.c_uint64(dev_ptr), ctypes.c_void_p(host_addr),
                                           ctypes.c_size_t(nbytes))
            if err != 0:
                raise RuntimeError(f"CUDA_ERROR {err}: cuMemcpyHtoD failed")

    def d2h(self, arr: np.ndarray, dev_ptr: int) -> None:
        self.bind_thread()
        if arr.nbytes:
            err = self.lib.cuMemcpyDtoH_v2(ctypes.c_void_p(arr.ctypes.data), ctypes.c_uint64(dev_ptr),
                                           ctypes.c_size_t(arr.nbytes))
            if err != 0:
                raise RuntimeError(f"CUDA_ERROR {err}: cuMemcpyDtoH failed")

    def sync(self) -> None:
        self.bind_thread()
        err = self.lib.cuCtxSynchronize()
        if err == 2:
            raise MemoryError("CUDA_ERROR_OUT_OF_MEMORY: cuCtxSynchronize")
        if err != 0:
            raise RuntimeError(f"CUDA_ERROR {err}: cuCtxSynchronize failed")

    POOL_KEEP = 4                  # buffers kept per name
    POOL_MAX_BYTES = 32 << 20      # bigger ones are freed at check-in

    def checkout(self, name: str, nbytes: int) -> int:
        """A device buffer of at least ``nbytes`` from a shared, bounded pool (no allocation on
        the per-morsel path). Return it with :meth:`checkin`; threads come and go (the Router
        spawns one per run), so nothing is tied to a thread."""
        with self._pool_lock:
            free = self._pool.setdefault(name, [])
            for i, (ptr, size) in enumerate(free):
                if size >= nbytes:
                    del free[i]
                    self._sizes[ptr] = size
                    return ptr
            stale = free.pop(0) if len(free) >= self.POOL_KEEP else None
        if stale is not None:
            self.free(stale[0])
        ptr = self.malloc(nbytes)
        with self._pool_lock:
            self._sizes[ptr] = nbytes
        return ptr

    def checkin(self, name: str, ptr: int) -> None:
        with self._pool_lock:
            size = self._sizes.pop(ptr, 0)
            free = self._pool.setdefault(name, [])
            if size and size <= self.POOL_MAX_BYTES and len(free) < self.POOL_KEEP:
                free.append((ptr, size))
                return
        self.free(ptr)

    def upload(self, arr: np.ndarray) -> int:
        arr = np.ascontiguousarray(arr)
        ptr = self.malloc(arr.nbytes)
        self.h2d(ptr, arr)
        return ptr


_cuda: Optional[_Cuda] = None


def _get_cuda() -> _Cuda:
    global _cuda
    with _lock:
        if _cuda is None:
            _cuda = _Cuda()
    return _cuda


REPRO_MAX_NS = 4096     # (slots + 1) * groups the reproducible GPU kernel holds: REPRO_MAX_NS in pipeline_gpu.mojo


def _load_gpu():
    global _gpu_lib
    with _lock:
        if _gpu_lib is None:
            path = _find("libmxpipeline_gpu.so", "MXFRAME_PIPELINE_GPU_LIB")
            if path is None:
                raise FileNotFoundError("libmxpipeline_gpu.so not found")
            lib = ctypes.CDLL(str(path))
            fn = lib.mx_pipeline_gpu
            fn.restype = _I64
            fn.argtypes = [_I64] * 22
            mode = lib.mx_pipeline_gpu_mode
            mode.restype = _I64
            mode.argtypes = [_I64] * 5
            _gpu_lib = lib
    return _gpu_lib


def gpu_reproducible_fits(program: PipelineProgram, n_groups: int) -> bool:
    """Can the GPU kernel add this program's floats in a fixed order (mode 0, 3 or the ordered table)? Asked by
    the kernel library itself, so the answer is the one ``run_pipeline_gpu(reproducible=True)`` will get."""
    return _load_gpu().mx_pipeline_gpu_mode(n_groups, program.n_slots, int(program.code[1]),
                                            int(len(program.code)), 1) >= 0


class _DeviceProgram:
    """Bytecode, constants, sets and identities of one program, resident on the GPU."""

    def __init__(self, program: PipelineProgram, cu: _Cuda):
        self.cu = cu
        self.code = cu.upload(program.code)
        consts = np.zeros(64, dtype=np.float64)   # the kernel stages MAX_CONSTS = 64 doubles
        consts[:len(program.consts)] = program.consts
        self.consts = cu.upload(consts)
        self.n_regs = int(program.code[1])
        self.code_ints = int(len(program.code))
        self.agg_base = int(program.code[2] + program.code[3] + program.code[4])
        self.set_bytes = cu.upload(program.set_bytes)
        self.set_meta = cu.upload(program.set_meta)
        self.ident = cu.upload(program.identity if program.n_slots else np.zeros(1))
        self._ptrs = [self.code, self.consts, self.set_bytes, self.set_meta, self.ident]

    def free(self):
        for p in self._ptrs:
            self.cu.free(p)
        self._ptrs = []


def _device_program(program: PipelineProgram, cu: _Cuda) -> _DeviceProgram:
    dp = getattr(program, "_device_program", None)
    if dp is None:
        dp = _DeviceProgram(program, cu)
        program._device_program = dp
        weakref.finalize(program, dp.free)
    return dp


class DeviceColumns:
    """Columns (and group ids) resident on the GPU for rows ``[base_row, base_row + n_rows)``.

    Lets a Router upload a morsel itself (on its own copy stream) and hand the
    kernel already-resident pointers.
    """

    def __init__(self, ptrs: List[int], dtypes: List[int], gid_ptr: int, base_row: int, n_rows: int,
                 owned: Optional[List[int]] = None, cu: Optional[_Cuda] = None):
        self.ptrs, self.dtypes, self.gid_ptr = ptrs, dtypes, gid_ptr
        self.base_row, self.n_rows = base_row, n_rows
        self._owned = owned or []
        self._cu = cu

    def free(self) -> None:
        if self._cu is not None:
            for p in self._owned:
                self._cu.free(p)
        self._owned = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.free()


def upload_columns(program: PipelineProgram, columns, gids: Optional[np.ndarray],
                   start: int, end: int, before_copy=None) -> DeviceColumns:
    """Copy rows ``[start, end)`` of each column (and ``gids``) to the GPU.

    Chunked columns are copied chunk by chunk straight into one device buffer
    per column: no host-side concatenation.
    """
    cu = _get_cuda()
    cc = _as_chunked(program, columns)
    if gids is not None:
        gids = _check_gids(gids, 2)
    n = end - start
    itemsizes = [b.itemsize for b in cc.chunks[0]] if cc.chunks else [8] * cc.n_cols
    owned: List[int] = []
    try:
        ptrs = [cu.malloc(n * isz) for isz in itemsizes]
        owned += ptrs
        for (s, e), arrs in zip(cc.bounds, cc.chunks):
            lo, hi = max(s, start), min(e, end)
            if hi <= lo:
                continue
            for c, b in enumerate(arrs):
                if before_copy is not None:
                    before_copy()        # background uploads yield to running queries here
                cu.h2d_addr(ptrs[c] + (lo - start) * b.itemsize, b.addr + (lo - s) * b.itemsize,
                            (hi - lo) * b.itemsize)
        gp = 0
        if gids is not None:
            gp = cu.upload(gids[start:end])
            owned.append(gp)
        cu.sync()
    except BaseException:
        for p in owned:
            cu.free(p)
        raise
    return DeviceColumns(ptrs, _column_dtypes(cc), gp, start, n, owned, cu)


def run_pipeline_gpu(program: PipelineProgram, columns: Union[Sequence[np.ndarray], ChunkedColumns, DeviceColumns],
                     gids: Optional[np.ndarray], n_groups: int, start: int, end: int,
                     _repeat: int = 1, reproducible: Optional[bool] = None) -> Partials:
    """Run the program over rows ``[start, end)`` on the GPU.

    ``reproducible`` (default: the mode of the ``compute()`` running in this thread, as for the CPU kernel;
    a worker thread started by the caller does not inherit it, so pass it in): float sums without atomics,
    bit-identical from run to run (F-067). Raises :class:`GpuUnsupported` when the group count does not fit
    the ordered kernel (``REPRO_MAX_NS``): the caller's CPU path then runs the defined order.

    ``columns`` are host arrays / chunked columns (only the row range is copied
    over) or a :class:`DeviceColumns` already on the device. ``_repeat`` re-runs
    the main kernel (benchmarking only: results then accumulate).
    """
    lib = _load_gpu()
    cu = _get_cuda()
    dp = _device_program(program, cu)
    if start < 0 or start > end:
        raise ValueError(f"bad row range [{start}, {end})")
    n_slots = program.n_slots
    dev_cols: Optional[DeviceColumns] = None
    taken = []

    def take(name: str, nbytes: int) -> int:
        ptr = cu.checkout(name, nbytes)
        taken.append((name, ptr))
        return ptr

    try:
        if isinstance(columns, DeviceColumns):
            dc = columns
            if start < dc.base_row or end > dc.base_row + dc.n_rows:
                raise ValueError("row range outside the resident columns")
        else:
            cc = _as_chunked(program, columns)
            gids = _check_gids(gids, n_groups)
            n_rows = cc.n_rows if cc.n_cols else (len(gids) if gids is not None else end)
            if gids is not None:
                n_rows = min(n_rows, len(gids))
            if end > n_rows:
                raise ValueError(f"row range [{start}, {end}) outside 0..{n_rows}")
            dc = dev_cols = upload_columns(program, cc, gids, start, end)
        k_start, k_end = start - dc.base_row, end - dc.base_row
        if n_groups != 1 and not dc.gid_ptr:
            raise ValueError("gids may only be omitted for a global aggregate (n_groups == 1)")
        max_blocks = cu.sm_count * 8          # the kernel launches sm_count * (blocks that fit per SM)
        # pooled scratch: launch arguments, outputs and the block partials
        col_ptrs = take("cols", max(len(dc.ptrs), 1) * 8)
        col_dt = take("dts", max(len(dc.dtypes), 1) * 4)
        cu.h2d(col_ptrs, np.array(dc.ptrs or [0], dtype=np.int64))
        cu.h2d(col_dt, np.array(dc.dtypes or [0], dtype=np.int32))
        out_vals = take("out_vals", max(n_slots * n_groups, 1) * 8)
        out_cnt = take("out_cnt", n_groups * 8)
        repro = reproducible if reproducible is not None else _reproducible_now()
        ns = (n_slots + 1) * n_groups
        # (CUDA only: the Python path never runs on an Apple GPU, whose mode 5 would need scratch_rows' sizing)
        block_rows = ns if ns <= 48 or (repro and ns <= REPRO_MAX_NS) else 0
        scratch = take("scratch", max(max_blocks * block_rows * 8, 8))
        ovf = take("ovf", 8)                       # the kernel sets it when an integer sum passes int64 (F-060)
        cu.h2d(ovf, np.zeros(1, dtype=np.int64))
        cu.sync()
        rc = lib.mx_pipeline_gpu(
            dp.code, dp.consts, dp.set_bytes, dp.set_meta, col_ptrs, col_dt,
            dc.gid_ptr if n_groups > 1 else 0, n_groups, n_slots, k_start, k_end,
            out_vals, out_cnt, dp.ident, scratch, dp.n_regs, dp.code_ints, dp.agg_base,
            cu.sm_count, int(_repeat), 1 if repro else 0, ovf,
        )
        if rc == -6:
            raise GpuUnsupported("reproducible float sums need the group table in shared memory")
        if rc == -5:
            raise GpuUnsupported("program needs more shared memory than a block has")
        if rc != 0:
            raise RuntimeError(f"CUDA_ERROR: mx_pipeline_gpu failed with code {rc}")
        vals = np.empty((n_slots, n_groups), dtype=np.float64)
        cnt = np.empty(n_groups, dtype=np.int64)
        if n_slots:
            cu.d2h(vals, out_vals)
        cu.d2h(cnt, out_cnt)
        flag = np.zeros(1, dtype=np.int64)
        cu.d2h(flag, ovf)
        if flag[0]:
            raise SumOverflow("Overflow in sum of INT64")
        return Partials(vals=vals, cnt=cnt)
    finally:
        for name, ptr in taken:
            cu.checkin(name, ptr)
        if dev_cols is not None:
            dev_cols.free()
