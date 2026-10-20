"""The fused probe of a join (F-153, F-154): the join runs inside the pipeline that aggregates (or lists) its rows.

When the build side's keys are unique, each probe row has at most one partner, so no joined row needs to exist: the kernel's front stage
(`pipeline_fused`) takes each 1,024-row block of the probe table, looks every row's key up (`hj_probe_block`: prefetched, a miss drops
the row), gathers the partner's columns that the program reads into per-worker scratch (the block's cache-resident "joined columns"), and
points the program's column table at the probe table's own columns and the scratch. The aggregate then runs over the surviving rows as if
the joined table were there. Nothing the size of the join result is written.

`FusedJoin` is what the SQL layer registers on the joined source: the probe table, the build table, the map and the normalized keys, with
the numbering of the joined columns (each is a column of the probe table or of the build table). `FrontSpec` is the plain-
value copy of it that the front closure captures; the arrays it points to live in the kernel run (`KernelRun` regions) or in `FusedJoin`,
both alive for the whole kernel call.
"""
from std.memory import ArcPointer
from .table import SharedTable, Table
from .bind import SqlSource
from .query import Program
from .kernels.pipeline_cpu import IP, LP, FP, BP, VEC
from .kernels.hashjoin_cpu import JoinMap, MapView, load_view, hj_probe_block, hj_probe_left_block, hj_filter_block
from .kargs import KernelRun
from .joincore import JoinKeys, KeyPlan, probe_chunk_keys

comptime DT_NULLMASK = 7

# how the front stage treats a probe row (F-154): the inner join drops rows without a partner; a LEFT join keeps them (their build columns are NULL);
# a semi / anti join keeps or drops rows by whether the key is present and gathers nothing (the numbers are the kernel's `PM_*` modes)
comptime FM_INNER = 0
comptime FM_LEFT = 1


struct StageData(Movable):
    """One dimension of a star (F-155): the map built on its table's keys, and the probe table's normalized keys for it."""
    var map: ArcPointer[JoinMap]
    var keys: JoinKeys

    def __init__(out self, map: ArcPointer[JoinMap], var keys: JoinKeys):
        self.map = map
        self.keys = keys^


struct StreamStage(Movable):
    """One join of a streamed probe (F-158): the build side's table and map (shared by every chunk and every worker, read only) and how a chunk's key columns
    become the map's keys."""
    var build: SharedTable
    var map: ArcPointer[JoinMap]
    var plan: KeyPlan
    var pcols: List[Int]             # the probe's key columns, by the probe source's column numbers

    def __init__(out self, build: SharedTable, map: ArcPointer[JoinMap], var plan: KeyPlan, var pcols: List[Int]):
        self.build = build
        self.map = map
        self.plan = plan^
        self.pcols = pcols^


struct ProbeStream(Movable):
    """A fused join whose probe side arrives in chunks: a scan's row groups, a frame's shared rows and then its changed rows, a table. Each chunk gets a
    `FusedJoin` of its own (`chunk_join`): its keys are planned once (`StreamStage.plan`), the build side's maps are shared. `used` are the probe source's
    columns a chunk holds (in order; all of them for a table or a frame), `pos` the position of each in a chunk."""
    var src: SqlSource
    var used: List[Int]
    var pos: List[Int]
    var stages: List[StreamStage]
    var fmode: Int                   # a single join: FM_INNER, FM_LEFT or a presence filter mode
    var star: Bool                   # several unique-key dimensions probed one after the other
    var vstage: List[Int]            # per joined column: -1 a probe column, else the stage whose build table it comes from
    var vreal: List[Int]             # ... the probe source's column number, or the build table's column

    def __init__(out self, var src: SqlSource, var used: List[Int], var pos: List[Int], var stages: List[StreamStage], fmode: Int, star: Bool, var vstage: List[Int], var vreal: List[Int]):
        self.src = src^
        self.used = used^
        self.pos = pos^
        self.stages = stages^
        self.fmode = fmode
        self.star = star
        self.vstage = vstage^
        self.vreal = vreal^


struct FusedJoin(Movable):
    var probe: SharedTable
    var build: SharedTable
    var vprobe: List[Bool]           # per joined column: is it a probe-table column
    var vreal: List[Int]             # ... and which column of its table
    var vpartner: List[Int]          # a build-side joined column: its number among the partner columns (-1 for a probe column)
    var partner: List[Int]           # the build table's columns the front stage gathers, in partner order
    var map: ArcPointer[JoinMap]
    var keys: JoinKeys               # the probe side's normalized keys (and the build side's, which built the map)
    var n_probe_cols: Int            # the columns of a probe chunk (the program's table columns after them are the dimensions' placeholders)
    var stream: Optional[ArcPointer[ProbeStream]]   # set on the join a streamed query is lowered against: its probe arrives in chunks
    var fmode: Int                   # FM_INNER, FM_LEFT, or a presence filter mode (semi / anti: no partner columns)
    var partner_table: List[SharedTable]   # the table each partner column comes from (one table for a join; a star has one per dimension)
    var partner_stage: List[Int]           # ... and which stage (dimension) of the star
    var stages: List[StageData]      # a star: the dimensions probed one after the other, each with unique keys (empty for a single join)

    def __init__(out self, probe: SharedTable, build: SharedTable, var vprobe: List[Bool], var vreal: List[Int], map: ArcPointer[JoinMap], var keys: JoinKeys, fmode: Int = 0):
        self.fmode = fmode
        self.n_probe_cols = probe.columns()
        self.stream = None
        self.probe = probe
        self.build = build
        self.vpartner = List[Int]()
        self.partner = List[Int]()
        self.partner_table = List[SharedTable]()
        self.partner_stage = List[Int]()
        self.stages = List[StageData]()
        for i in range(len(vprobe)):
            if vprobe[i]:
                self.vpartner.append(-1)
            else:
                self.vpartner.append(len(self.partner))
                self.partner.append(vreal[i])
                self.partner_table.append(build)
                self.partner_stage.append(0)
        self.vprobe = vprobe^
        self.vreal = vreal^
        self.map = map
        self.keys = keys^

    def __init__(out self, probe: SharedTable, var tables: List[SharedTable], var vstage: List[Int], var vreal: List[Int], var stages: List[StageData]):
        """A star: joined column i is the probe table's column `vreal[i]` when `vstage[i]` is -1, else column `vreal[i]` of dimension `vstage[i]`."""
        self.fmode = 0
        self.n_probe_cols = probe.columns()
        self.stream = None
        self.probe = probe
        self.build = probe
        self.vprobe = List[Bool]()
        self.vpartner = List[Int]()
        self.partner = List[Int]()
        self.partner_table = List[SharedTable]()
        self.partner_stage = List[Int]()
        for i in range(len(vstage)):
            if vstage[i] < 0:
                self.vprobe.append(True)
                self.vpartner.append(-1)
            else:
                self.vprobe.append(False)
                self.vpartner.append(len(self.partner))
                self.partner.append(vreal[i])
                self.partner_table.append(tables[vstage[i]])
                self.partner_stage.append(vstage[i])
        self.vreal = vreal^
        self.map = ArcPointer(JoinMap())
        self.keys = JoinKeys()
        self.stages = stages^

    def virtual_columns(self) -> Int:
        return len(self.vprobe)


struct FrontSpec(ImplicitlyCopyable, Movable):
    """Plain values for the front closure: nothing here owns memory."""
    var view: MapView
    var keys_a: Int
    var ok_a: Int
    var tbl_a: Int                   # the program's column table of the probe table ([row start, end, pointers...])
    var n_words: Int                 # program columns
    var n_partner: Int
    var n_data: Int                  # entries of (partner, program column) for data columns ...
    var data_a: Int                  # ... Int64 pairs at this address
    var n_valid: Int                 # ... and for validity bitmaps: (partner, slot) pairs
    var valid_a: Int
    var bvals_a: Int                 # per partner: its build column's values address and its validity bits address (0: none)
    var off_prow: Int
    var off_vals: Int
    var off_bits: Int
    var ws_bytes: Int
    var fmode: Int
    var n_stages: Int                # a star: how many dimensions, ...
    var stages_a: Int                # ... and their 15 words each (the map's 11, the probe keys' address, the NULL flags' address, 2 spare)

    def __init__(out self):
        self.n_stages = 0
        self.stages_a = 0
        self.fmode = 0
        self.view = MapView(JoinMap())
        self.keys_a = 0
        self.ok_a = 0
        self.tbl_a = 0
        self.n_words = 0
        self.n_partner = 0
        self.n_data = 0
        self.data_a = 0
        self.n_valid = 0
        self.valid_a = 0
        self.bvals_a = 0
        self.off_prow = 0
        self.off_vals = 0
        self.off_bits = 0
        self.ws_bytes = 0


def _align8(n: Int) -> Int:
    return (n + 7) // 8 * 8


def front_spec(mut run: KernelRun, fj: FusedJoin, p: Program) raises -> FrontSpec:
    """Which of the program's columns are the build side's, and where the front stage finds what it gathers. The arrays go into the run's
    regions (alive as long as the run, which the kernel call is a method of)."""
    var fs = FrontSpec()
    fs.view = MapView(fj.map[])
    fs.fmode = fj.fmode
    fs.keys_a = fj.keys.pk_addr
    fs.ok_a = fj.keys.po_addr
    fs.n_words = len(p.cols)
    fs.n_partner = len(fj.partner)
    var first = fj.n_probe_cols                                      # the program's table columns from here on are the build side's (placeholders)
    var data = List[Int]()
    var valid = List[Int]()
    for q in range(len(p.cols)):
        var dt = Int(p.dtypes[q]) & 255
        var tc = p.cols[q]
        if dt == DT_NULLMASK:
            if tc >= first:
                valid.append(tc - first)
                valid.append(Int(p.dtypes[q]) >> 8)
        elif tc >= first:
            data.append(tc - first)
            data.append(q)
    fs.n_data = len(data) // 2
    fs.n_valid = len(valid) // 2
    var rd = run.region(max(len(data), 1))
    var rv = run.region(max(len(valid), 1))
    for i in range(len(data)):
        run.i64(rd)[i] = Int64(data[i])
    for i in range(len(valid)):
        run.i64(rv)[i] = Int64(valid[i])
    fs.data_a = Int(run.i64(rd))
    fs.valid_a = Int(run.i64(rv))
    var rb = run.region(max(3 * fs.n_partner, 1))
    for j in range(fs.n_partner):
        var c = fj.partner[j]
        run.i64(rb)[3 * j] = Int64(Int(fj.partner_table[j].values(c).unsafe_ptr()))
        run.i64(rb)[3 * j + 1] = Int64(fj.partner_table[j]._table[].cols[c].validity_address())
        run.i64(rb)[3 * j + 2] = Int64(fj.partner_stage[j])
    fs.bvals_a = Int(run.i64(rb))
    fs.n_stages = len(fj.stages)
    var rs = run.region(max(15 * fs.n_stages, 1))
    for k in range(fs.n_stages):
        MapView(fj.stages[k].map[]).store(run.i64(rs) + 15 * k)
        run.i64(rs)[15 * k + 11] = Int64(fj.stages[k].keys.pk_addr)
        run.i64(rs)[15 * k + 12] = Int64(fj.stages[k].keys.po_addr)
    fs.stages_a = Int(run.i64(rs))
    fs.tbl_a = run.table_address()
    fs.off_prow = _align8(fs.n_words * 8)
    fs.off_vals = fs.off_prow + max(fs.n_stages, 1) * VEC * 4
    fs.off_bits = fs.off_vals + fs.n_partner * VEC * 8
    fs.ws_bytes = fs.off_bits + fs.n_partner * (VEC // 8) + 64
    return fs


def jf_front(fs: FrontSpec, i0: Int, n: Int, sel: IP, ws: BP) -> Int:
    """One block: probe, drop the misses, gather the partners' columns into the worker's scratch, and aim the program's column table at the
    probe table and the scratch (rebased by the block's first row, as the kernel's loads index `i0 + row`)."""
    for j in range(n):
        sel[j] = Int32(j)
    if fs.n_stages > 0:
        return _jf_star(fs, i0, n, sel, ws)
    var prow = IP(unsafe_from_address=Int(ws) + fs.off_prow)
    var keys = LP(unsafe_from_address=fs.keys_a)
    var okp = BP(unsafe_from_address=fs.ok_a if fs.ok_a != 0 else Int(ws))
    var kept: Int
    if fs.fmode == FM_INNER:
        kept = hj_probe_block(fs.view, keys, okp, fs.ok_a != 0, i0, sel, n, prow)
    elif fs.fmode == FM_LEFT:
        hj_probe_left_block(fs.view, keys, okp, fs.ok_a != 0, i0, sel, n, prow)
        kept = n
    else:
        kept = hj_filter_block(fs.view, keys, okp, fs.ok_a != 0, i0, sel, n, fs.fmode)
    if kept == 0:
        return 0
    var lc = LP(unsafe_from_address=Int(ws))
    var arow = LP(unsafe_from_address=fs.tbl_a) + 2
    for q in range(fs.n_words):
        lc[q] = arow[q]
    var bv = LP(unsafe_from_address=fs.bvals_a)
    var dat = LP(unsafe_from_address=fs.data_a)
    for e in range(fs.n_data):
        var u = Int(dat[2 * e])
        var q = Int(dat[2 * e + 1])
        var dst = LP(unsafe_from_address=Int(ws) + fs.off_vals + u * VEC * 8)
        var src = LP(unsafe_from_address=Int(bv[3 * u]))
        if fs.fmode == FM_LEFT:
            for t in range(kept):
                var r = Int(prow[t])
                if r >= 0:
                    dst[Int(sel[t])] = src[r]                         # an unmatched row's value is never read: its validity bit is clear
        else:
            for t in range(kept):
                dst[Int(sel[t])] = src[Int(prow[t])]
        lc[q] = Int64(Int(dst)) - Int64(i0 * 8)
    var vd = LP(unsafe_from_address=fs.valid_a)
    for e in range(fs.n_valid):
        var u = Int(vd[2 * e])
        var slot = Int(vd[2 * e + 1])
        var valid_bits = Int(bv[3 * u + 1])
        var left = fs.fmode == FM_LEFT
        if valid_bits == 0 and not left:
            continue                                                  # the build column has no NULL: the program reads it as always valid
        var bits = BP(unsafe_from_address=Int(ws) + fs.off_bits + u * (VEC // 8))
        for z in range(VEC // 8):
            bits[z] = 0
        var src = BP(unsafe_from_address=valid_bits)
        for t in range(kept):
            var r = Int(prow[t])
            if r < 0:
                continue                                              # a LEFT join's unmatched row: NULL
            if valid_bits == 0 or (src[r >> 3] >> UInt8(r & 7)) & 1 != 0:
                var b = Int(sel[t])
                bits[b >> 3] = bits[b >> 3] | (UInt8(1) << UInt8(b & 7))
        lc[slot] = Int64(Int(bits))
        lc[slot + 1] = Int64(-i0)
    return kept


def _jf_star(fs: FrontSpec, i0: Int, n: Int, sel: IP, ws: BP) -> Int:
    """A star's block: every dimension's map is probed in turn (a row without a partner in any of them is dropped from `sel`; each dimension keeps the
    partner row of each block row in its own array), then the partners' columns are gathered for the rows that survived all of them."""
    var kept = n
    for st in range(fs.n_stages):
        var w = LP(unsafe_from_address=fs.stages_a) + 15 * st
        var view = load_view(w)
        var keys = LP(unsafe_from_address=Int(w[11]))
        var has_ok = w[12] != 0
        var okp = BP(unsafe_from_address=Int(w[12]) if has_ok else Int(ws))
        kept = hj_probe_block(view, keys, okp, has_ok, i0, sel, kept, IP(unsafe_from_address=Int(ws) + fs.off_prow + st * VEC * 4), True)
        if kept == 0:
            return 0
    var lc = LP(unsafe_from_address=Int(ws))
    var arow = LP(unsafe_from_address=fs.tbl_a) + 2
    for q in range(fs.n_words):
        lc[q] = arow[q]
    var bv = LP(unsafe_from_address=fs.bvals_a)
    var dat = LP(unsafe_from_address=fs.data_a)
    for e in range(fs.n_data):
        var u = Int(dat[2 * e])
        var q = Int(dat[2 * e + 1])
        var dst = LP(unsafe_from_address=Int(ws) + fs.off_vals + u * VEC * 8)
        var src = LP(unsafe_from_address=Int(bv[3 * u]))
        var prow = IP(unsafe_from_address=Int(ws) + fs.off_prow + Int(bv[3 * u + 2]) * VEC * 4)
        for t in range(kept):
            var r0 = Int(sel[t])
            dst[r0] = src[Int(prow[r0])]
        lc[q] = Int64(Int(dst)) - Int64(i0 * 8)
    var vd = LP(unsafe_from_address=fs.valid_a)
    for e in range(fs.n_valid):
        var u = Int(vd[2 * e])
        var slot = Int(vd[2 * e + 1])
        var valid_bits = Int(bv[3 * u + 1])
        if valid_bits == 0:
            continue                                                  # the dimension's column has no NULL: the program reads it as always valid
        var prow = IP(unsafe_from_address=Int(ws) + fs.off_prow + Int(bv[3 * u + 2]) * VEC * 4)
        var bits = BP(unsafe_from_address=Int(ws) + fs.off_bits + u * (VEC // 8))
        for z in range(VEC // 8):
            bits[z] = 0
        var src = BP(unsafe_from_address=valid_bits)
        for t in range(kept):
            var r0 = Int(sel[t])
            var r = Int(prow[r0])
            if (src[r >> 3] >> UInt8(r & 7)) & 1 != 0:
                bits[r0 >> 3] = bits[r0 >> 3] | (UInt8(1) << UInt8(r0 & 7))
        lc[slot] = Int64(Int(bits))
        lc[slot + 1] = Int64(-i0)
    return kept


def chunk_join(ps: ProbeStream, chunk: SharedTable, layout_only: Bool = False) raises -> FusedJoin:
    """The fused join of one probe chunk: its keys from the chunk's columns, the build sides' maps shared. `layout_only`: no keys, no rows (the join the query is
    lowered against: it has the chunk's columns and the dimensions' placeholders)."""
    var vreal = List[Int]()
    for v in range(len(ps.vstage)):
        vreal.append(ps.pos[ps.vreal[v]] if ps.vstage[v] < 0 else ps.vreal[v])
    var fj: FusedJoin
    if not ps.star:
        var keys = JoinKeys()
        if not layout_only:
            var pc = List[Int]()
            for c in ps.stages[0].pcols:
                pc.append(ps.pos[c])
            probe_chunk_keys(ps.stages[0].plan, chunk, pc, keys)
        var vprobe = List[Bool]()
        for v in range(len(ps.vstage)):
            vprobe.append(ps.vstage[v] < 0)
        fj = FusedJoin(chunk, ps.stages[0].build, vprobe^, vreal^, ps.stages[0].map, keys^, ps.fmode)
    else:
        var tables = List[SharedTable]()
        var stages = List[StageData]()
        for k in range(len(ps.stages)):
            var keys = JoinKeys()
            if not layout_only:
                var pc = List[Int]()
                for c in ps.stages[k].pcols:
                    pc.append(ps.pos[c])
                probe_chunk_keys(ps.stages[k].plan, chunk, pc, keys)
            tables.append(ps.stages[k].build)
            stages.append(StageData(ps.stages[k].map, keys^))
        fj = FusedJoin(chunk, tables^, ps.vstage.copy(), vreal^, stages^)
    if layout_only:
        fj.n_probe_cols = len(ps.used)
    return fj^


def stream_template(ps: ArcPointer[ProbeStream]) raises -> FusedJoin:
    """The join a streamed query is lowered against."""
    var fj = chunk_join(ps[], Table.empty().share(), True)
    fj.stream = ps
    return fj^
