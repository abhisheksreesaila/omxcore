"""A device copy of a shared table (docs/omxcore-mojo-api.md section 7).

    var dt = DeviceTable(shared)                 # uploads every column once (a default DeviceContext, GPU 0)
    var dt = DeviceTable(shared, app_ctx)        # or on the app's own DeviceContext
    var r = Query(frame).on_device(dt).where(...).group_by(...).agg(...).compute(device=GPU)
    dt.release()                                 # the document closed: give the memory back

Why it looks like this:
- int64 throughout: the stand-in uploaded int32 and refused a wider value (a single amount over $21.4M in cents
  does not fit); here the columns go up exactly as the table holds them, validity bitmaps (Arrow layout) beside them.
- A `DeviceTable` is a handle like `SharedTable`: copies share one device copy, so every session holds the one the
  process uploaded. `release()` frees it for all of them (a query on a released table raises on `GPU` and runs on
  the CPU on `AUTO`).
- The app's `DeviceContext` is used as is: buffers are allocated on it and the query kernels are enqueued on it, so
  the app's own kernels (charts, point clouds) and omxcore's share one device, one memory pool and one stream. With
  the default constructor omxcore creates its own context on device 0.
- It is read-only: a session's edits never touch it (a query uploads the small bitmap of changed rows instead).
- F-138b, where the GPU code lives: only on a GPU build. `HAS_GPU` is true on Linux and, since F-182, on Apple
  Silicon (Metal: the kernels' word is Int64 there, see kernels/pipeline_gpu.mojo); `-D MXFRAME_NO_GPU` turns it off
  (to test the other side, or for a Mac build cross-compiled on Linux: Metal kernels need Apple's Metal toolchain at
  build time). Anywhere else `gpu_available()` is False, `DeviceTable(...)` raises "no GPU on this build", `Query.compute(GPU)`
  raises the same and AUTO runs on the CPU. Every use of `DeviceContext`, a buffer or `kernels.pipeline_gpu` sits in
  a `comptime if HAS_GPU` block (the imports stay; a module-level import is not code), so an Apple object refers to
  no `AsyncRT_Device*` symbol. The device memory hangs off the shared state by address (`_GpuMemory`) for that reason:
  a field of that type would put its destructor, and so the GPU runtime, into every build.
"""
from std.memory import ArcPointer
from std.memory.alloc import unsafe_alloc
from std.sys import is_defined, CompilationTarget, has_apple_gpu_accelerator
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from .table import SharedTable

comptime HAS_GPU = (CompilationTarget.is_linux() or (CompilationTarget.is_macos() and has_apple_gpu_accelerator())) and not is_defined["MXFRAME_NO_GPU"]()
comptime NO_GPU_MESSAGE = "no GPU on this build"


def gpu_available() -> Bool:
    """Is there a GPU omxcore can use? (Creating a DeviceContext with none present crashes: ask first.) Always False
    on a build without GPU code."""
    comptime if HAS_GPU:
        try:
            return DeviceContext.number_of_devices() > 0
        except:
            return False
    else:
        return False


def gpu_device() -> String:
    """The GPU queries run on, as "<name> · <api>" ("NVIDIA GeForce RTX 3090 · cuda", "Apple M4 · metal"), for an
    app's status bar; "" when `gpu_available()` is False."""
    comptime if HAS_GPU:
        if not gpu_available():
            return String()
        try:
            var ctx = DeviceContext()
            return ctx.name() + " · " + ctx.api()
        except:
            return String()
    else:
        return String()


struct _GpuMemory(Movable):
    """What lives on the device. Built and used only inside `comptime if HAS_GPU`."""
    var ctx: DeviceContext
    var data: List[DeviceBuffer[DType.int64]]
    var valid: List[DeviceBuffer[DType.uint8]]       # only the columns that have nulls

    def __init__(out self, var ctx: DeviceContext):
        self.ctx = ctx^
        self.data = List[DeviceBuffer[DType.int64]]()
        self.valid = List[DeviceBuffer[DType.uint8]]()


struct _DeviceState(Movable):
    var table: SharedTable                           # the table this mirrors, held so its identity cannot be reused
    var rows: Int
    var live: Bool
    var bytes: Int
    var data_addr: List[Int]
    var valid_addr: List[Int]                        # 0: the column has no nulls
    var gpu: Int                                     # address of the _GpuMemory (0: none, or released)

    def __init__(out self, table: SharedTable, rows: Int):
        self.table = table
        self.rows = rows
        self.live = True
        self.bytes = 0
        self.data_addr = List[Int]()
        self.valid_addr = List[Int]()
        self.gpu = 0

    def _free_gpu(mut self):
        comptime if HAS_GPU:
            if self.gpu != 0:
                var p = UnsafePointer[_GpuMemory, MutAnyOrigin](unsafe_from_address=self.gpu)
                _ = p.take_pointee()
                p.free()
                self.gpu = 0

    def __del__(deinit self):
        self._free_gpu()


struct DeviceTable(ImplicitlyCopyable):
    """The columns of a `SharedTable` on the GPU, uploaded once and shared by every session that holds a copy."""
    var _s: ArcPointer[_DeviceState]

    def __init__(out self, table: SharedTable) raises:
        """Uploads `table` to the default GPU (device 0)."""
        comptime if HAS_GPU:
            if not gpu_available():
                raise Error("DeviceTable: no GPU found")
            self = Self(table, DeviceContext())
        else:
            raise Error(NO_GPU_MESSAGE)

    def __init__(out self, table: SharedTable, ctx: DeviceContext) raises:
        """Uploads `table` on the app's `DeviceContext`: the app's buffers and these share one device."""
        comptime if HAS_GPU:
            self._s = ArcPointer(_DeviceState(table, table.rows()))
            var mem = unsafe_alloc[_GpuMemory](1)
            mem.unsafe_write(_GpuMemory(ctx.copy()))
            self._s[].gpu = Int(mem)
            ref s = self._s[]
            var n = table.rows()
            for c in range(table.columns()):
                var buf = ctx.enqueue_create_buffer[DType.int64](max(n, 1))
                if n > 0:                                # an empty column has no host memory to copy from
                    ctx.enqueue_copy(dst_buf=buf, src_ptr=table.values(c).unsafe_ptr())
                s.data_addr.append(Int(buf.unsafe_ptr()))
                mem[].data.append(buf^)
                s.bytes += 8 * max(n, 1)
                var nbytes = len(table._table[].cols[c].validity)
                if nbytes > 0:
                    var vb = ctx.enqueue_create_buffer[DType.uint8](nbytes)
                    ctx.enqueue_copy(dst_buf=vb, src_ptr=table._table[].cols[c].validity.unsafe_ptr())
                    s.valid_addr.append(Int(vb.unsafe_ptr()))
                    mem[].valid.append(vb^)
                    s.bytes += nbytes
                else:
                    s.valid_addr.append(0)
            ctx.synchronize()                            # the host columns may be freed or changed after this returns
        else:
            raise Error(NO_GPU_MESSAGE)

    def context(self) raises -> DeviceContext:
        comptime if HAS_GPU:
            if self._s[].gpu == 0:
                raise Error("the DeviceTable was released")
            return UnsafePointer[_GpuMemory, MutAnyOrigin](unsafe_from_address=self._s[].gpu)[].ctx.copy()
        else:
            raise Error(NO_GPU_MESSAGE)

    def rows(self) -> Int:
        return self._s[].rows

    def bytes(self) -> Int:
        """Device memory held (0 after `release`)."""
        return self._s[].bytes if self._s[].live else 0

    def is_live(self) -> Bool:
        return self._s[].live

    def mirrors(self, table: SharedTable) -> Bool:
        """Was this made from `table` (the same shared columns, not merely an equal-looking table)? The device table
        holds the table's handle, so the columns it mirrors cannot be freed and their address reused by another."""
        return Int(self._s[].table._table[].cols.unsafe_ptr()) == Int(table._table[].cols.unsafe_ptr())

    def data_address(self, c: Int) -> Int:
        return self._s[].data_addr[c]

    def validity_address(self, c: Int) -> Int:
        return self._s[].valid_addr[c]

    def release(self):
        """Frees the device memory for every copy of this handle (the document closed). Call it when no query on this
        table is running: the app queries from one thread, so there is no lock, and a launch still in flight would
        read freed memory."""
        ref s = self._s[]
        if not s.live:
            return
        s._free_gpu()
        s.data_addr = List[Int]()
        s.valid_addr = List[Int]()
        s.bytes = 0
        s.live = False


struct _ScanDevice(Movable):
    """What one streamed GPU scan keeps on the device: its context and one set of column buffers (and validity bitmaps)
    that every decoded row group is copied into in turn, grown when a group is bigger than the last, never freed between
    groups (an allocation and free of the group's columns cost about as much as the copy). Built and used only inside
    `comptime if HAS_GPU`."""
    var ctx: DeviceContext
    var data: List[DeviceBuffer[DType.int64]]
    var data_cap: List[Int]
    var valid: List[DeviceBuffer[DType.uint8]]
    var valid_cap: List[Int]
    var data_addr: List[Int]                         # of the group last loaded
    var valid_addr: List[Int]                        # 0: that column has no nulls in it
    var host: List[List[HostBuffer[DType.int64]]]    # two sets of pinned host buffers, one per column: a decoded row group
    var host_cap: List[List[Int]]                    #   is written into one set while the other is copied to the device

    def __init__(out self, var ctx: DeviceContext):
        self.ctx = ctx^
        self.data = List[DeviceBuffer[DType.int64]]()
        self.data_cap = List[Int]()
        self.valid = List[DeviceBuffer[DType.uint8]]()
        self.valid_cap = List[Int]()
        self.data_addr = List[Int]()
        self.valid_addr = List[Int]()
        self.host = List[List[HostBuffer[DType.int64]]]()
        self.host_cap = List[List[Int]]()
        for _ in range(2):
            self.host.append(List[HostBuffer[DType.int64]]())
            self.host_cap.append(List[Int]())


def _new_context() raises -> Int:
    """The device side of a streamed scan (a DeviceContext on the default GPU and its reusable column buffers), on the
    heap, by address; 0 on a build without GPU code or a machine without a GPU. No GPU symbol is named outside
    `comptime if HAS_GPU`."""
    comptime if HAS_GPU:
        if not gpu_available():
            return 0
        var mem = unsafe_alloc[_ScanDevice](1)
        mem.unsafe_write(_ScanDevice(DeviceContext()))
        return Int(mem)
    else:
        return 0


def _free_context(addr: Int):
    comptime if HAS_GPU:
        if addr != 0:
            var p = UnsafePointer[_ScanDevice, MutAnyOrigin](unsafe_from_address=addr)
            try:
                p[].ctx.synchronize()                    # nothing in flight when the buffers go
            except:
                pass
            p[].host = List[List[HostBuffer[DType.int64]]]()       # the buffers first, then the context they hang off
            p[].data = List[DeviceBuffer[DType.int64]]()
            p[].valid = List[DeviceBuffer[DType.uint8]]()
            _ = p.take_pointee()
            p.free()


def _scan_context(addr: Int) raises -> DeviceContext:
    comptime if HAS_GPU:
        return UnsafePointer[_ScanDevice, MutAnyOrigin](unsafe_from_address=addr)[].ctx.copy()
    else:
        raise Error(NO_GPU_MESSAGE)


def _pinned_prepare(addr: Int, which: Int, ncols: Int, rows: Int) raises -> List[Int]:
    """Pinned host buffers of `rows` Int64 for `ncols` columns in set `which` (0 or 1), grown when too small (only a set the
    device is not copying from may be asked for), and their addresses: a decoded row group is written straight into them
    (`ParquetScan.read_group_into`) and copied to the device from there, at the bus's full speed and without a host copy."""
    var out = List[Int]()
    comptime if HAS_GPU:
        ref m = UnsafePointer[_ScanDevice, MutAnyOrigin](unsafe_from_address=addr)[]
        for c in range(ncols):
            if c >= len(m.host[which]):
                m.host[which].append(m.ctx.enqueue_create_host_buffer[DType.int64](max(rows, 1)))
                m.host_cap[which].append(max(rows, 1))
            elif m.host_cap[which][c] < rows:
                m.host[which][c] = m.ctx.enqueue_create_host_buffer[DType.int64](rows)
                m.host_cap[which][c] = rows
            out.append(Int(m.host[which][c].unsafe_ptr()))
        m.ctx.synchronize()                              # (the buffers exist before anything is written into them)
    return out^


def _stage_load(addr: Int, table: SharedTable, ext: List[Int] = List[Int](), rows: Int = -1) raises:
    """Copies the table's columns (and the validity bitmaps of those that have nulls) into the scan's device buffers,
    growing them when the table has more rows than any before, and waits for the copies: the host columns may change after
    this returns. `_stage_data` / `_stage_valid` then give the device addresses."""
    comptime if HAS_GPU:
        ref m = UnsafePointer[_ScanDevice, MutAnyOrigin](unsafe_from_address=addr)[]
        var n = table.rows() if rows < 0 else rows           # (ext: the values are in the caller's buffers, the table has none)
        m.data_addr = List[Int]()
        m.valid_addr = List[Int]()
        for c in range(table.columns()):
            if c >= len(m.data):                      # a column not seen before: its buffers
                m.data.append(m.ctx.enqueue_create_buffer[DType.int64](max(n, 1)))
                m.data_cap.append(max(n, 1))
                m.valid.append(m.ctx.enqueue_create_buffer[DType.uint8](1))
                m.valid_cap.append(1)
            elif m.data_cap[c] < n:                    # a bigger group: a bigger buffer (the old one is idle)
                m.data[c] = m.ctx.enqueue_create_buffer[DType.int64](n)
                m.data_cap[c] = n
            if n > 0:
                if len(ext) > 0:
                    m.ctx.enqueue_copy(dst_buf=m.data[c], src_ptr=UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=ext[c]))
                else:
                    m.ctx.enqueue_copy(dst_buf=m.data[c], src_ptr=table.values(c).unsafe_ptr())
            m.data_addr.append(Int(m.data[c].unsafe_ptr()))
            var nbytes = len(table._table[].cols[c].validity)
            if nbytes > 0:
                if m.valid_cap[c] < nbytes:
                    m.valid[c] = m.ctx.enqueue_create_buffer[DType.uint8](nbytes)
                    m.valid_cap[c] = nbytes
                m.ctx.enqueue_copy(dst_buf=m.valid[c], src_ptr=table._table[].cols[c].validity.unsafe_ptr())
                m.valid_addr.append(Int(m.valid[c].unsafe_ptr()))
            else:
                m.valid_addr.append(0)
        m.ctx.synchronize()


def _stage_data(addr: Int, c: Int) -> Int:
    comptime if HAS_GPU:
        return UnsafePointer[_ScanDevice, MutAnyOrigin](unsafe_from_address=addr)[].data_addr[c]
    else:
        return 0


def _stage_valid(addr: Int, c: Int) -> Int:
    comptime if HAS_GPU:
        return UnsafePointer[_ScanDevice, MutAnyOrigin](unsafe_from_address=addr)[].valid_addr[c]
    else:
        return 0


struct _ScanContext(Movable):
    """Owns the device side of one streamed GPU scan: `__del__` frees it, so every way out of the scan (a result, an
    overflow, a decode error, a failed allocation) gives the CUDA context and its buffers back."""
    var addr: Int                    # the address of the `_ScanDevice`, 0: none

    def __init__(out self, want: Bool) raises:
        self.addr = _new_context() if want else 0

    def __del__(deinit self):
        _free_context(self.addr)
