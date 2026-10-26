"""A read-only mapping of part of a file (Parquet column chunks, CSV row groups): in the process only while it is held."""
from std.ffi import external_call

comptime _MAP_ALIGN = 16384          # mmap offsets must be page multiples: 4 KB (Linux x86), 16 KB (Apple arm64)


struct _Mapped(Movable):
    """A read-only mapping of `length` bytes of a file from `offset`; unmapped when released or dropped, so a
    chunk's pages are in the process only while it is decoded."""
    var addr: Int
    var span: Int
    var data: Int                    # the address of the byte at `offset`

    def __init__(out self):
        self.addr = 0
        self.span = 0
        self.data = 0

    def __init__(out self, fd: Int, offset: Int, length: Int, path: String) raises:
        var start = offset - offset % _MAP_ALIGN
        self.span = length + (offset - start)
        self.addr = external_call["mmap", Int](Int(0), self.span, Int32(1), Int32(2), fd, start)
        self.data = 0
        if self.addr == -1:
            self.span = 0
            raise Error("couldn't read `" + path + "`: part of the file could not be mapped into memory")
        self.data = self.addr + (offset - start)

    def release(mut self):
        if self.span > 0:
            _ = external_call["munmap", Int32](self.addr, self.span)
            self.span = 0

    def __del__(deinit self):
        if self.span > 0:
            _ = external_call["munmap", Int32](self.addr, self.span)
