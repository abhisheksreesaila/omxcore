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
from std.os import getenv
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


# ---- F-189: how much GPU memory a query may take ----------------------------------------------------------------------
# `MXFRAME_GPU_MEM_BUDGET` fakes a smaller GPU (a test knob, as in the Python front end): "512M" for every GPU, or
# "0:512M,1:2G" per device id. K / M / G are 1024-based, a trailing B is allowed ("64KB"), a bare number is bytes. The
# real figure is what the MAX runtime reports as free on the context's device (`DeviceContext.get_memory_info`); the
# budget is the smaller of the two, less GPU_HEADROOM_PERCENT kept back (HeavyDB's 90%), so a query's own buffers
# and the driver's bookkeeping fit beside what omxcore holds.
comptime GPU_BUDGET_ENV = "MXFRAME_GPU_MEM_BUDGET"
comptime GPU_FAIL_ENV = "MXFRAME_GPU_FAIL_PIECE"
comptime GPU_USABLE_PERCENT = 90
comptime GPU_RETRY_FACTOR_PERCENT = 150          # a remembered out-of-memory shape gets another GPU try once this much more is free


def _drop_last(s: String) -> String:
    return String(s[byte=0 : s.byte_length() - 1])


def parse_size(text: String) raises -> Int:
    """"512M" -> 536870912; "64KB", "2G", "1.5G", "4096" (K, M, G are powers of 1024)."""
    var t = String(text.strip()).upper()
    if t.endswith("B") and t.byte_length() > 1:
        t = _drop_last(t)
    var unit = 1
    if t.endswith("K"):
        unit = 1 << 10
    elif t.endswith("M"):
        unit = 1 << 20
    elif t.endswith("G"):
        unit = 1 << 30
    if unit > 1:
        t = _drop_last(t)
    if t.byte_length() == 0:
        raise Error(GPU_BUDGET_ENV + ": `" + text + "` is not a size (try 512M)")
    try:
        return Int(Float64(t) * Float64(unit))
    except:
        raise Error(GPU_BUDGET_ENV + ": `" + text + "` is not a size (try 512M)")


def gpu_budget_override(device_id: Int = 0) -> Int:
    """The fake size of GPU `device_id` from MXFRAME_GPU_MEM_BUDGET, -1 when none is set. A value that is not a size
    raises where the budget is first used (`gpu_free_bytes`)."""
    return _budget_override(device_id)


def _budget_override(device_id: Int) -> Int:
    var raw = getenv(GPU_BUDGET_ENV, "")
    if raw.byte_length() == 0:
        return -1
    var dflt = -1
    for part in raw.split(","):
        var at = part.find(":")
        try:
            if at >= 0:
                if Int(String(part[byte=0:at]).strip()) == device_id:
                    return parse_size(String(part[byte=at + 1:]))
            else:
                dflt = parse_size(String(part))
        except:
            return -2                                    # (unreadable: `gpu_free_bytes` says so)
    return dflt


def gpu_free_bytes(ctx: DeviceContext) raises -> Int:
    """Free memory on the context's device, as the MAX runtime reports it, capped by the test override. A device whose
    runtime does not report memory counts as unlimited."""
    comptime if HAS_GPU:
        var free = Int.MAX
        try:
            free = Int(ctx.get_memory_info()[0])
        except:
            pass
        var o = _budget_override(Int(ctx.id()))
        if o == -2:
            raise Error(GPU_BUDGET_ENV + " is `" + getenv(GPU_BUDGET_ENV, "") + "`: a size like 512M, or 0:512M,1:2G")
        return free if o < 0 else min(free, o)
    else:
        raise Error(NO_GPU_MESSAGE)


def gpu_budget(ctx: DeviceContext) raises -> Int:
    """What one query may hold on the device at once: GPU_USABLE_PERCENT of `gpu_free_bytes`."""
    var free = gpu_free_bytes(ctx)
    return free // 100 * GPU_USABLE_PERCENT


def gpu_capacity(ctx: DeviceContext) raises -> Int:
    """What a query could take on this device if nothing else were using it: GPU_USABLE_PERCENT of the runtime's total, capped
    by the override. `gpu_budget` is what is free now; a query that does not fit even this can never run on this GPU."""
    comptime if HAS_GPU:
        var total = Int.MAX
        try:
            total = Int(ctx.get_memory_info()[1])
        except:
            pass
        var o = _budget_override(Int(ctx.id()))
        return (total if o < 0 else min(total, o)) // 100 * GPU_USABLE_PERCENT
    else:
        raise Error(NO_GPU_MESSAGE)


def human_bytes(n: Int) -> String:
    """12582912 -> "12.0 MB" for the messages that name a size."""
    if n < 1024:
        return String(n) + " B"
    var units: List[String] = ["KB", "MB", "GB", "TB"]
    var x = Float64(n)
    var u = -1
    while x >= 1024.0 and u < 3:
        x /= 1024.0
        u += 1
    var tenths = Int(x * 10.0 + 0.5)
    return String(tenths // 10) + "." + String(tenths % 10) + " " + units[u]


comptime INJECTED_OOM = "CUDA call failed: CUDA_ERROR_OUT_OF_MEMORY (out of memory; injected by MXFRAME_GPU_FAIL_PIECE)"


def gpu_fail_piece() -> Int:
    """MXFRAME_GPU_FAIL_PIECE (a test knob): the device allocation of the Nth piece of a query (1 = the first row range, row
    group or the query's own buffers) raises as an out-of-memory would; 0 fails the upload of a `DeviceTable`. -1: none."""
    var raw = getenv(GPU_FAIL_ENV, "")
    if raw.byte_length() == 0:
        return -1
    try:
        return Int(raw)
    except:
        return -1


def _table_bytes(table: SharedTable) -> Int:
    """Device bytes of a whole upload: each column's int64 values and its validity bitmap."""
    var n = 0
    for c in range(table.columns()):
        n += 8 * max(table.rows(), 1) + len(table._table[].cols[c].validity)
    return n


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
    var resident: Bool                               # F-189: the columns are on the device (False: too big for the budget, queries upload row ranges)
    var need: Int                                    # the bytes a whole upload takes
    var oom_free: Int                                # the free bytes when a query last ran out of memory (0: none did); AUTO waits for 1.5x

    def __init__(out self, table: SharedTable, rows: Int):
        self.table = table
        self.rows = rows
        self.live = True
        self.bytes = 0
        self.resident = False
        self.need = 0
        self.oom_free = 0
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
        """Uploads `table` on the app's `DeviceContext`: the app's buffers and these share one device. F-189: a table
        that does not fit the GPU budget (`gpu_budget`) is not uploaded: the handle holds the context and no device copy
        (`is_resident()` is False, `bytes()` 0), `compute(GPU)` then uploads row ranges that fit, one after the other,
        and AUTO answers on the CPU. An upload that runs out of memory half way is undone the same way."""
        comptime if HAS_GPU:
            self._s = ArcPointer(_DeviceState(table, table.rows()))
            var mem = unsafe_alloc[_GpuMemory](1)
            mem.unsafe_write(_GpuMemory(ctx.copy()))
            self._s[].gpu = Int(mem)
            ref s = self._s[]
            var n = table.rows()
            s.need = _table_bytes(table)
            var budget = gpu_budget(ctx)
            if s.need > budget:
                s.oom_free = 0
                return
            try:
                if gpu_fail_piece() == 0:
                    raise Error(INJECTED_OOM)
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
                s.resident = True
            except e:
                try:
                    ctx.synchronize()
                except:
                    pass
                mem[].data = List[DeviceBuffer[DType.int64]]()          # what went up is given back
                mem[].valid = List[DeviceBuffer[DType.uint8]]()
                s.data_addr = List[Int]()
                s.valid_addr = List[Int]()
                s.bytes = 0
                s.oom_free = max(gpu_free_bytes(ctx), 1)
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

    def is_resident(self) -> Bool:
        """F-189: the columns are on the device. False for a table over the GPU budget (queries upload row ranges)."""
        return self._s[].live and self._s[].resident

    def needed_bytes(self) -> Int:
        """The device bytes a whole upload of the table takes (what `bytes()` is once it is resident)."""
        return self._s[].need

    def oom_free(self) -> Int:
        """Free device bytes when a query on this table last ran out of memory, 0 if none did (AUTO stays on the CPU until
        GPU_RETRY_FACTOR_PERCENT of that is free)."""
        return self._s[].oom_free

    def remember_oom(self, free: Int):
        self._s[].oom_free = max(free, 1)

    def forget_oom(self):
        self._s[].oom_free = 0

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


def _context_on(ctx: DeviceContext) -> Int:
    """The same, on the caller's `DeviceContext` (F-189: a table's row ranges are staged on the app's device)."""
    comptime if HAS_GPU:
        var mem = unsafe_alloc[_ScanDevice](1)
        mem.unsafe_write(_ScanDevice(ctx.copy()))
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


def _stage_range(addr: Int, table: SharedTable, data_used: List[Bool], valid_used: List[Bool], r0: Int, rows: Int,
                 ext: List[Int] = List[Int]()) raises:
    """F-189: rows [r0, r0 + rows) of the columns a query reads (`data_used[c]`: its values, `valid_used[c]`: its validity
    bitmap) copied to the scan's device buffers, which are made exactly `rows` long (a copy moves a whole buffer: a longer one
    would read past the host column), and waited for. `r0` is a multiple of 8, so a bitmap slice starts on a byte.
    `ext`: the values are in these host buffers, not in the table (a scan's decoded row group). The bitmap goes up with 8 spare
    zero bytes after it (a kernel may read a word past the last row). `_stage_data` / `_stage_valid` give the addresses."""
    comptime if HAS_GPU:
        ref m = UnsafePointer[_ScanDevice, MutAnyOrigin](unsafe_from_address=addr)[]
        m.data_addr = List[Int]()
        m.valid_addr = List[Int]()
        var hosts = List[List[UInt8]]()                  # (kept to the end: the copies read them until the wait)
        for c in range(table.columns()):
            if c >= len(m.data):
                m.data.append(m.ctx.enqueue_create_buffer[DType.int64](1))
                m.data_cap.append(1)
                m.valid.append(m.ctx.enqueue_create_buffer[DType.uint8](1))
                m.valid_cap.append(1)
            if data_used[c] and rows > 0:
                if m.data_cap[c] != rows:
                    m.data[c] = m.ctx.enqueue_create_buffer[DType.int64](rows)
                    m.data_cap[c] = rows
                var src = (ext[c] if len(ext) > 0 else Int(table.values(c).unsafe_ptr())) + 8 * r0
                m.ctx.enqueue_copy(dst_buf=m.data[c], src_ptr=UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=src))
            m.data_addr.append(Int(m.data[c].unsafe_ptr()))
            var have = len(table._table[].cols[c].validity)
            if valid_used[c] and have > 0:
                var first = r0 // 8
                var nb = (rows + 7) // 8
                var host = List[UInt8](length=nb + 8, fill=0)
                var src = table._table[].cols[c].validity.unsafe_ptr()
                for i in range(min(nb, have - first)):
                    host[i] = src[first + i]
                if m.valid_cap[c] != nb + 8:
                    m.valid[c] = m.ctx.enqueue_create_buffer[DType.uint8](nb + 8)
                    m.valid_cap[c] = nb + 8
                m.ctx.enqueue_copy(dst_buf=m.valid[c], src_ptr=host.unsafe_ptr())
                hosts.append(host^)
                m.valid_addr.append(Int(m.valid[c].unsafe_ptr()))
            else:
                m.valid_addr.append(0)
        m.ctx.synchronize()
        _ = hosts^


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

    def __init__(out self, ctx: DeviceContext):
        self.addr = _context_on(ctx)

    def __del__(deinit self):
        _free_context(self.addr)
