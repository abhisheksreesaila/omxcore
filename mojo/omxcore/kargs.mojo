"""The arguments of a kernel call, with an owner (F-152).

A kernel takes addresses: of its code, its constants, its table of column pointers, and of the arrays its block hook writes
through a control block. Mojo frees a value at its last use, and a List whose address went into a call or a control block
has no use the compiler can see after that, so the memory was freed before the kernel read it (F-148, F-152 twice). Writing
`_ = xs^` by hand after each call is a rule people forget.

`KernelRun` owns every one of those lists. Nothing outside it holds an address: the arrays a hook needs are `region`s of the
run (`region`, `bytes_region`, handed back as an index), the control block is one of them, and the call is a *method* of the run
(`keyed`, `hashed`), so the run is alive for the whole call and for whatever the caller reads from it afterwards (`i64`, `i32`,
`u8` give a pointer to a region, valid while the run is). A function that must hand a region's address to a second function
(such as `window_blocks`) takes the run as an argument, so the run is borrowed for that call too.
"""
from .table import SharedTable
from .query import Program, MAGIC, cpu_code
from .kernels._fpenv import core_count
from .kernels.pipeline_cpu import IP, LP, FP, BP, pipeline_keyed, pipeline_fused, _no_front
from .kernels.hashagg_cpu import hash_aggregate
from .exec import _table_addresses


struct KernelRun(Movable):
    var code: List[Int32]
    var consts: List[Float64]
    var set_bytes: List[UInt8]
    var set_meta: List[Int64]
    var tbl: List[Int64]
    var dtypes: List[Int32]
    var out_vals: List[Float64]
    var out_cnt: List[Int64]
    var regions: List[List[Int64]]       # the arrays a hook or the kernel's result reads or writes; their addresses are stable (a List moves, its buffer does not)
    var ctl: Int                         # the region holding the hook's control block, -1 for none
    var ctl_const: Int                   # the constant that holds its address
    var repro: Bool

    def __init__(out self, p: Program, set_bytes: List[UInt8], set_meta: List[Int64], t: SharedTable, mask_addr: Int, groups: Int = 1, row0: Int = 0, rows: Int = -1):
        self.code = cpu_code(p)
        self.consts = List[Float64]()
        for v in p.consts:
            self.consts.append(Float64(from_bits=UInt64(v)))
        if len(self.consts) == 0:
            self.consts.append(0.0)
        self.set_bytes = set_bytes.copy()
        if len(self.set_bytes) == 0:
            self.set_bytes.append(0)
        self.set_meta = set_meta.copy()
        if len(self.set_meta) == 0:
            self.set_meta = [0, 0, 0]
        self.tbl = _table_addresses(t, p, mask_addr, row0, rows)
        self.dtypes = p.dtypes.copy()
        if len(self.dtypes) == 0:
            self.dtypes.append(1)
        self.out_vals = List[Float64](length=max(p.n_slots, 1) * max(groups, 1), fill=0.0)
        self.out_cnt = List[Int64](length=max(groups, 1), fill=0)
        self.regions = List[List[Int64]]()
        self.ctl = -1
        self.ctl_const = p.ctl_const
        self.repro = p.repro

    # -- the arrays of a hook
    def region(mut self, words: Int, zero: Bool = True) -> Int:
        """An array of `words` 8-byte words (zero-filled unless `zero` is False: the hook writes every entry it later reads); its index."""
        if zero:
            self.regions.append(List[Int64](length=max(words, 1), fill=0))
        else:
            self.regions.append(List[Int64](unsafe_uninit_length=max(words, 1)))
        return len(self.regions) - 1

    def bytes_region(mut self, nbytes: Int, zero: Bool = True) -> Int:
        return self.region((nbytes + 7) // 8, zero)

    def i64(self, r: Int) -> LP:
        return LP(unsafe_from_address=Int(self.regions[r].unsafe_ptr()))

    def i32(self, r: Int) -> IP:
        return IP(unsafe_from_address=Int(self.regions[r].unsafe_ptr()))

    def u8(self, r: Int) -> BP:
        return BP(unsafe_from_address=Int(self.regions[r].unsafe_ptr()))

    def control(mut self, r: Int):
        """Region `r` is the hook's control block: its address goes into the constant the program loads it from, at the call."""
        self.ctl = r

    # -- the calls: methods, so the run is alive while the kernel reads it
    def keyed[E: def(Int, Int, IP, Int, Bool, IP, FP) -> Bool](mut self, encode: E, n_groups: Int, start: Int, stop: Int, threads: Int) -> Int:
        """`pipeline_keyed` over rows [start, stop); the program's results are `out_vals` / `out_cnt`."""
        if self.ctl >= 0:
            self.consts[self.ctl_const] = Float64(from_bits=UInt64(Int(self.regions[self.ctl].unsafe_ptr())))
        return pipeline_keyed[E](Int(self.code.unsafe_ptr()), Int(self.consts.unsafe_ptr()), Int(self.set_bytes.unsafe_ptr()),
                                 Int(self.set_meta.unsafe_ptr()), Int(self.tbl.unsafe_ptr()), 1, Int(self.dtypes.unsafe_ptr()), n_groups,
                                 start, stop, Int(self.out_vals.unsafe_ptr()), Int(self.out_cnt.unsafe_ptr()),
                                 threads if threads > 0 else core_count(), encode, False)

    def hashed[KW: def(Int, Int, IP, Int, Bool, LP, FP, BP) -> Bool](mut self, keyw: KW, nkw: Int, n_rows: Int, threads: Int, res: Int) -> Int:
        """`hash_aggregate` (hashed groups); `res` is the region that receives the result's addresses."""
        return hash_aggregate[fused=False](Int(self.code.unsafe_ptr()), Int(self.consts.unsafe_ptr()), Int(self.set_bytes.unsafe_ptr()),
                                           Int(self.set_meta.unsafe_ptr()), Int(self.tbl.unsafe_ptr()), 1, Int(self.dtypes.unsafe_ptr()), nkw, n_rows,
                                           threads if threads > 0 else core_count(), self.repro, keyw, _no_front, 0,
                                           LP(unsafe_from_address=Int(self.regions[res].unsafe_ptr())), 0)

    def table_address(self) -> Int:
        """The program's column table (valid while the run is)."""
        return Int(self.tbl.unsafe_ptr())

    def fused[E: def(Int, Int, IP, Int, Bool, IP, FP) -> Bool, FR: def(Int, Int, Int, IP, BP) -> Int](
            mut self, encode: E, front: FR, ws_bytes: Int, keyed: Bool, n_groups: Int, start: Int, stop: Int, threads: Int) -> Int:
        """`pipeline_fused`: the program over the rows `front` lets through (a join probe: see jfront.mojo)."""
        if self.ctl >= 0:
            self.consts[self.ctl_const] = Float64(from_bits=UInt64(Int(self.regions[self.ctl].unsafe_ptr())))
        if keyed:
            return pipeline_fused[E, FR, True](Int(self.code.unsafe_ptr()), Int(self.consts.unsafe_ptr()), Int(self.set_bytes.unsafe_ptr()),
                                               Int(self.set_meta.unsafe_ptr()), Int(self.tbl.unsafe_ptr()), 1, Int(self.dtypes.unsafe_ptr()), n_groups,
                                               start, stop, Int(self.out_vals.unsafe_ptr()), Int(self.out_cnt.unsafe_ptr()),
                                               threads if threads > 0 else core_count(), encode, front, ws_bytes, self.repro)
        return pipeline_fused[E, FR, False](Int(self.code.unsafe_ptr()), Int(self.consts.unsafe_ptr()), Int(self.set_bytes.unsafe_ptr()),
                                            Int(self.set_meta.unsafe_ptr()), Int(self.tbl.unsafe_ptr()), 1, Int(self.dtypes.unsafe_ptr()), n_groups,
                                            start, stop, Int(self.out_vals.unsafe_ptr()), Int(self.out_cnt.unsafe_ptr()),
                                            threads if threads > 0 else core_count(), encode, front, ws_bytes, self.repro)

    def hashed_fused[KW: def(Int, Int, IP, Int, Bool, LP, FP, BP) -> Bool, FR: def(Int, Int, Int, IP, BP) -> Int](
            mut self, keyw: KW, front: FR, ws_bytes: Int, nkw: Int, n_rows: Int, threads: Int, res: Int) -> Int:
        """`hash_aggregate` over the rows `front` lets through."""
        return hash_aggregate[fused=True](Int(self.code.unsafe_ptr()), Int(self.consts.unsafe_ptr()), Int(self.set_bytes.unsafe_ptr()),
                                          Int(self.set_meta.unsafe_ptr()), Int(self.tbl.unsafe_ptr()), 1, Int(self.dtypes.unsafe_ptr()), nkw, n_rows,
                                          threads if threads > 0 else core_count(), self.repro, keyw, front, ws_bytes,
                                          LP(unsafe_from_address=Int(self.regions[res].unsafe_ptr())), 0)
