"""The executor's Sources (F-152): where the rows come from, and the helpers every Sink shares.

A Source is read in chunks, one at a time: a table (one chunk), a Frame (its shared table with the changed rows masked out, then a
small table of the changed and appended rows), or a Parquet scan (one chunk per row group, then the changed rows). `_Chunks`
opens them, `_load_chunk` gives chunk i and the address of its mask, `_check_footer` verifies what a file's footer stated.
`_Inputs` says how a kernel program sees the source (which columns, which may be NULL). The Sinks that consume the chunks are
in `sqlexec.mojo` (dense, hashed and holistic aggregates) and `rows.mojo` (materialized rows, top-k); both finish through the
generic helpers at the end of this file (`_order_rows`, `_top_rows`, `_to_table`), which work on columns of values (`ValCol`).
"""
from .joincore import _fill_column_owned
from .table import Table, SharedTable, Column, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .frame import NULL_EDIT, text_ranks, _radix_order
from .query import Program, _ScanMask, _group_mask
from .kernels._fpenv import core_count, par_n
from .kernels.pipeline_cpu import IP, LP, FP, BP
from .bind import Catalog, SqlSource, BoundQuery, SortKey, SRC_TABLE, SRC_FRAME, SRC_SCAN, B_COL, B_CAST, B_DATEPART, B_DATETRUNC, B_ADDMONTHS, B_DATEDIFF, B_NEG, B_ADD, B_SUB
from .planner import Stats, ColStat
from .sqlprog import DictAccess, collect_columns
from .sqleval import ValCol
from .holistic import float_key


def _table_addresses(t: SharedTable, p: Program, mask_addr: Int, row0: Int = 0, rows: Int = -1) -> List[Int64]:
    """The program's column table: a pointer per program column. A column the table does not have (a join's build-side column, which the
    fused front stage supplies per block) is a placeholder 0."""
    var tbl: List[Int64] = [0, Int64(t.rows() if rows < 0 else rows)]
    var n = t.columns()
    # F-175: rows row0 .. row0 + rows of the table as a table of its own (a multiple of 8, so the bitmaps line up): the kernel's hashed
    # aggregate has no start row, so a background job's ranges are views, not arguments
    var vo = Int64(8 * row0)
    var bo = Int64(row0 // 8)
    for c in p.cols:
        if c >= 0:
            tbl.append(Int64(Int(t.values(c).unsafe_ptr())) + vo if c < n else Int64(0))
        elif c == -1_000_000 or c == -5_000_000:
            tbl.append(0)
        elif c == -4_000_000:
            tbl.append(Int64(mask_addr) + (bo if mask_addr != 0 else 0))
        else:
            var va = Int64(t._table[].cols[-1 - c].validity_address()) if -1 - c < n else Int64(0)
            tbl.append(va + (bo if va != 0 else 0))
    return tbl^


# ── sources: statistics, the delta table, the program's inputs ───────────────────────────────────────

def _column_range(t: SharedTable, c: Int) -> Tuple[Bool, Int64, Int64]:
    """The lowest and highest non-null value of a column (and whether there is one): in parallel over slices of the rows."""
    var n = t.rows()
    var tasks = max(1, min(core_count(), n // 65536))
    var los = List[Int64](length=tasks, fill=Int64.MAX)
    var his = List[Int64](length=tasks, fill=Int64.MIN)
    var seen = List[Int64](length=tasks, fill=0)
    var vp = Int(t.values(c).unsafe_ptr())
    var valid = t._table[].cols[c].validity_address()
    var lp = Int(los.unsafe_ptr())
    var hp = Int(his.unsafe_ptr())
    var sp = Int(seen.unsafe_ptr())

    def work(i: Int) {imm}:
        var vals = LP(unsafe_from_address=vp)
        var bits = BP(unsafe_from_address=valid)
        var lo = Int64.MAX
        var hi = Int64.MIN
        var any = Int64(0)
        var a = n * i // tasks
        var b = n * (i + 1) // tasks
        if valid == 0:
            for r in range(a, b):
                var v = vals[r]
                lo = min(lo, v)
                hi = max(hi, v)
            any = Int64(1) if b > a else Int64(0)
        else:
            for r in range(a, b):
                if (bits[r >> 3] >> UInt8(r & 7)) & 1 != 0:
                    var v = vals[r]
                    lo = min(lo, v)
                    hi = max(hi, v)
                    any = 1
        LP(unsafe_from_address=lp)[i] = lo
        LP(unsafe_from_address=hp)[i] = hi
        LP(unsafe_from_address=sp)[i] = any

    if tasks == 1:
        work(0)
    else:
        par_n(work, tasks, tasks)
    var lo = Int64.MAX
    var hi = Int64.MIN
    var any = False
    for i in range(tasks):
        if seen[i] != 0:
            lo = min(lo, los[i])
            hi = max(hi, his[i])
            any = True
    return (any, lo, hi)


def _is_range_kind(k: Int) -> Bool:
    return k == INT.value or k == DECIMAL.value or k == DATE.value


def _key_columns(q: BoundQuery, e: Int, mut out: List[Int]):
    """The integer-like source columns a row-level expression reads."""
    ref x = q.rexprs[e]
    if x.op == B_COL and _is_range_kind(x.kind):
        var c = Int(x.value)
        for k in out:
            if k == c:
                return
        out.append(c)
    if x.a >= 0:
        _key_columns(q, x.a, out)
    if x.b >= 0:
        _key_columns(q, x.b, out)


def _date_leaves(q: BoundQuery, e: Int, mut out: List[Int]):
    """The DATE columns a date function's operands read through date functions, + and - (a per-day table is made for them: `sqlprog._Emitter.day_table`)."""
    ref x = q.rexprs[e]
    if x.op == B_COL and x.kind == DATE.value:
        out.append(Int(x.value))
    elif x.op == B_DATEPART or x.op == B_DATETRUNC or x.op == B_NEG:
        _date_leaves(q, x.a, out)
    elif x.op == B_ADDMONTHS or x.op == B_DATEDIFF or x.op == B_ADD or x.op == B_SUB:
        _date_leaves(q, x.a, out)
        _date_leaves(q, x.b, out)


def prepare_ranges(mut catalog: Catalog, q: BoundQuery) raises:
    """Caches the value range of each integer-like column a group key reads: one pass over a table or frame's column (the frame's
    changed rows included), the row groups' statistics for a scan (none when a row group states none, then it is hashed)."""
    ref src = catalog.sources[q.source]
    var cols = List[Int]()
    for k in q.keys:
        _key_columns(q, k.expr, cols)
    for x in q.rexprs:                                     # F-156: a CAST of a column to text writes the words of its range; a date function of a column answers once per day of it
        var cc = -1
        if x.op == B_CAST and x.kind == TEXT.value and q.rexprs[x.a].op == B_COL and _is_range_kind(q.rexprs[x.a].kind):
            cc = Int(q.rexprs[x.a].value)
        elif x.op == B_DATEPART or x.op == B_DATETRUNC or x.op == B_ADDMONTHS or x.op == B_DATEDIFF:
            var leaves = List[Int]()
            _date_leaves(q, x.a, leaves)
            if x.b >= 0:
                _date_leaves(q, x.b, leaves)
            for lc in leaves:
                var seen_leaf = False
                for k in cols:
                    seen_leaf = seen_leaf or k == lc
                if not seen_leaf:
                    cols.append(lc)
        if cc >= 0:
            var seen = False
            for k in cols:
                seen = seen or k == cc
            if not seen:
                cols.append(cc)
    for c in cols:
        var cached = False
        for x in src.range_cols:
            if x == c:
                cached = True
        if cached:
            continue
        var r = _column_range(src.table, c) if src.kind != SRC_SCAN else src.scan.value().column_range(c)
        if src.kind == SRC_SCAN and not r[0]:
            continue                                       # no statistics: the planner will say so
        var has = r[0] and r[1] <= r[2] if src.kind == SRC_SCAN else r[0]
        var lo = r[1]
        var hi = r[2]
        for row in src.overlay.delta:                      # a frame's changed and appended rows
            if row[c] != src.overlay.null_marker:
                lo = min(lo, row[c])
                hi = max(hi, row[c])
                has = True
        src.range_cols.append(c)
        src.range_lo.append(lo)
        src.range_hi.append(hi)
        src.range_has.append(has)


def _alias_lists(src: SqlSource) -> List[List[Int]]:
    var out = List[List[Int]]()
    for _ in range(src.table.columns()):
        out.append(List[Int]())
    for item in src.overlay.aliases.items():
        var c = Int(item.key >> 32)
        if c < len(out):
            out[c].append(Int(item.key & 0xFFFFFFFF))
            out[c].append(Int(item.value))
    return out^


def _delta_nulls(src: SqlSource, c: Int) -> Bool:
    for row in src.overlay.delta:
        if row[c] == src.overlay.null_marker:
            return True
    return False


def make_stats(src: SqlSource, q: BoundQuery, dicts: DictAccess, trust: Bool = True) raises -> Stats:
    """`trust` False: a file's footer statistics are not used (every column may hold NULLs, no range is known)."""
    var st = Stats(src.name, src.kind, src.rows() + (len(src.overlay.delta) if src.kind != SRC_SCAN else 0))
    st.edits = len(src.overlay.delta)
    st.version = src.version
    st.distrusted = not trust
    var scan_cols = collect_columns(q) if src.kind == SRC_SCAN else List[Int]()
    if src.scan:
        st.row_groups = src.scan.value().row_groups()
        if src.scan.value().is_csv():
            st.scan_name = "CSV scan"
    for c in range(src.table.columns()):
        var cs = ColStat(src.table.kind(c).value, src.table.scale(c))
        if src.kind == SRC_SCAN:
            cs.nullable = not trust or src.scan.value().null_count(c) != 0 or _delta_nulls(src, c)
        else:
            cs.nullable = src.table.null_count(c) > 0 or _delta_nulls(src, c)
        if cs.kind == TEXT.value and (src.kind != SRC_SCAN or c in scan_cols):      # (a scan builds a TEXT column's dictionary on first use: not for a column the query never reads, F-178)
            cs.words = dicts.count(c)
        for i in range(len(src.range_cols)):
            if src.range_cols[i] == c and src.range_has[i] and (trust or src.kind != SRC_SCAN):
                cs.has_range = True
                cs.lo = src.range_lo[i]
                cs.hi = src.range_hi[i]
        st.cols.append(cs^)
    return st^


def delta_table(src: SqlSource, cols: List[Int]) raises -> SharedTable:
    """The frame's changed and appended rows as a small table of `cols` (merged TEXT codes applied)."""
    var t = Table.empty()
    for c in cols:
        _ = t.add_column(src.table.name(c), src.table.kind(c), scale=src.table.scale(c))
    for row in src.overlay.delta:
        for k in range(len(cols)):
            var raw = row[cols[k]]
            if raw == src.overlay.null_marker:
                t.cols[k].push(0, False)
            else:
                if src.table.kind(cols[k]) == TEXT:
                    var hit = src.overlay.aliases.get((cols[k] << 32) | Int(raw))
                    if hit:
                        raw = hit.value()
                t.cols[k].push(raw, True)
    return t^.share()


# ── finishing: groups -> HAVING -> ORDER BY -> LIMIT -> the result table ─────────────────────────────

def _gather(vc: ValCol, idx: List[Int]) -> ValCol:
    var out = ValCol(vc.kind, vc.scale, len(idx))
    out.src = vc.src
    for i in range(len(idx)):
        out.v[i] = vc.v[idx[i]]
    if len(vc.nul) > 0:
        out.nul = List[UInt8](length=len(idx), fill=0)
        for i in range(len(idx)):
            out.nul[i] = vc.nul[idx[i]]
    return out^


def _sort_keys(vc: ValCol, dicts: DictAccess) raises -> List[Int64]:
    """Int64 keys whose signed order is the column's order: ints as they are, floats by their bits, TEXT by the place of
    the word among the words present (alphabetical, ignoring ASCII case, bytes as the tie-break)."""
    var n = vc.n()
    var keys = List[Int64](length=n, fill=0)
    if vc.kind == FLOAT.value:
        for i in range(n):
            var b = vc.v[i]
            keys[i] = 0 if vc.is_null(i) else float_key(b)       # -0.0 ties with 0.0 and every NaN with every NaN, above +inf
        return keys^
    if vc.kind == TEXT.value:
        var seen = List[Int](length=0, fill=0)
        var present = List[Int]()
        var index = List[Int]()
        var mx = 0
        for i in range(n):
            if not vc.is_null(i):
                mx = max(mx, Int(vc.v[i]))
        var at = List[Int32](length=mx + 1, fill=-1)
        for i in range(n):
            if vc.is_null(i):
                continue
            var c = Int(vc.v[i])
            if at[c] < 0:
                at[c] = Int32(len(present))
                present.append(c)
        var words = List[String]()
        for c in present:
            words.append(dicts.word(vc.src, c))
        var ranks = text_ranks(words)
        for i in range(n):
            if not vc.is_null(i):
                keys[i] = ranks[Int(at[Int(vc.v[i])])]
        return keys^
    for i in range(n):
        keys[i] = 0 if vc.is_null(i) else vc.v[i]                  # (a NULL's stored value means nothing: NULLs tie, whatever the kernel left there)
    return keys^


def _order_rows(cols: List[ValCol], order: List[SortKey], dicts: DictAccess, n: Int) raises -> List[Int]:
    """The rows 0 .. n-1 sorted by the ORDER BY keys: stable LSD passes from the last key to the first (and, within a
    key with NULLs, the values first and then the NULL flag, so NULLS LAST / FIRST holds whatever the direction)."""
    var perm = List[Int](length=n, fill=0)
    for i in range(n):
        perm[i] = i
    var k = len(order) - 1
    while k >= 0:
        var vc = cols[k].copy()
        var keys = _sort_keys(vc, dicts)
        var aligned = List[Int64](length=n, fill=0)
        for i in range(n):
            aligned[i] = keys[perm[i]]
        perm = _radix_order(aligned, perm, order[k].desc)
        if vc.has_nulls():
            var flags = List[Int64](length=n, fill=0)
            for i in range(n):
                var isn = vc.is_null(perm[i])
                flags[i] = Int64((0 if isn else 1) if order[k].nulls_first else (1 if isn else 0))
            perm = _radix_order(flags, perm, False)
        k -= 1
    return perm^


struct _SortCols(Movable):
    """The ORDER BY keys of a set of rows as comparable numbers, for `_top_rows`."""
    var keys: List[List[Int64]]
    var nulls: List[List[UInt8]]
    var desc: List[Bool]
    var nulls_first: List[Bool]

    def __init__(out self):
        self.keys = List[List[Int64]]()
        self.nulls = List[List[UInt8]]()
        self.desc = List[Bool]()
        self.nulls_first = List[Bool]()

    def cmp(self, a: Int, o: Self, b: Int) -> Int:
        """-1 when row a of this set sorts before row b of `o` (a set with the same keys: the merge of spilled runs, F-167), 1 when after,
        0 when every key ties. The one definition of the order after `_order_rows`."""
        for k in range(len(self.keys)):
            var an = len(self.nulls[k]) > 0 and self.nulls[k][a] != 0
            var bn = len(o.nulls[k]) > 0 and o.nulls[k][b] != 0
            if an or bn:
                if an and bn:
                    continue
                return -1 if an == self.nulls_first[k] else 1
            var x = self.keys[k][a]
            var y = o.keys[k][b]
            if x != y:
                return -1 if ((x > y) if self.desc[k] else (x < y)) else 1
        return 0

    def before(self, a: Int, b: Int) -> Bool:
        """Row a sorts before row b (ties: the lower row number, as the stable radix order)."""
        var c = self.cmp(a, self, b)
        return c < 0 or (c == 0 and a < b)


def _top_rows(cols: List[ValCol], order: List[SortKey], dicts: DictAccess, n: Int, want: Int) raises -> List[Int]:
    """The first `want` rows of 0 .. n-1 in ORDER BY order, by a bounded heap: one pass, no full sort (a big result with a
    small LIMIT)."""
    var sc = _SortCols()
    for k in range(len(order)):
        sc.keys.append(_sort_keys(cols[k], dicts))
        sc.nulls.append(cols[k].nul.copy())
        sc.desc.append(order[k].desc)
        sc.nulls_first.append(order[k].nulls_first)
    var heap = List[Int]()                  # a max-heap: the worst of the best `want` rows on top
    if want <= 0:
        return heap^
    for r in range(n):
        if len(heap) < want:
            heap.append(r)
            var i = len(heap) - 1
            while i > 0:
                var parent = (i - 1) // 2
                if sc.before(heap[parent], heap[i]):
                    var t = heap[parent]
                    heap[parent] = heap[i]
                    heap[i] = t
                    i = parent
                else:
                    break
        elif sc.before(r, heap[0]):
            heap[0] = r
            var i = 0
            while True:
                var l = 2 * i + 1
                var rr = l + 1
                var worst = i
                if l < len(heap) and sc.before(heap[worst], heap[l]):
                    worst = l
                if rr < len(heap) and sc.before(heap[worst], heap[rr]):
                    worst = rr
                if worst == i:
                    break
                var t = heap[worst]
                heap[worst] = heap[i]
                heap[i] = t
                i = worst
    # sort the survivors (insertion sort: `want` is small)
    for i in range(1, len(heap)):
        var j = i
        while j > 0 and sc.before(heap[j], heap[j - 1]):
            var t = heap[j]
            heap[j] = heap[j - 1]
            heap[j - 1] = t
            j -= 1
    return heap^


def _fill_column(mut col: Column, vals: List[Int64], nul: List[UInt8]):
    col.values = vals.copy()
    if len(nul) > 0:
        var n = len(vals)
        col.validity = List[UInt8](length=(n + 7) // 8, fill=0xFF)
        var nulls = 0
        for i in range(n):
            if nul[i] != 0:
                col.validity[i // 8] &= ~(UInt8(1) << UInt8(i % 8))
                nulls += 1
        col.nulls = nulls
        if nulls == 0:
            col.validity = List[UInt8]()


def _unique_name(t: Table, name: String) -> String:
    if t.index(name) < 0:
        return name
    var i = 1
    while t.index(name + "_" + String(i)) >= 0:
        i += 1
    return name + "_" + String(i)


def _put_column(mut t: Table, name: String, vc: ValCol, dicts: DictAccess) raises:
    """Adds `vc` to the table as a column called `name`; a TEXT column gets a compact dictionary of the words actually present."""
    var kind = ColumnKind(vc.kind)
    var c = t.add_column(_unique_name(t, name), kind, scale=vc.scale)
    if vc.kind == TEXT.value:
        var n = vc.n()
        var mx = 0
        for r in range(n):
            if not vc.is_null(r):
                mx = max(mx, Int(vc.v[r]))
        var remap = List[Int32](length=mx + 1, fill=-1)
        var codes = List[Int64](length=n, fill=0)
        for r in range(n):
            if vc.is_null(r):
                continue
            var old = Int(vc.v[r])
            if remap[old] < 0:
                remap[old] = Int32(t.add_word(c, dicts.word(vc.src, old)))
            codes[r] = Int64(remap[old])
        _fill_column(t.cols[c], codes, vc.nul)
    else:
        _fill_column(t.cols[c], vc.v, vc.nul)


def _put_column_owned(mut t: Table, name: String, mut vc: ValCol, dicts: DictAccess) raises:
    """`_put_column` for a column nobody needs afterwards: its values move into the table (a TEXT column's codes are remapped as before)."""
    if vc.kind == TEXT.value:
        _put_column(t, name, vc, dicts)
        return
    var kind = ColumnKind(vc.kind)
    var c = t.add_column(_unique_name(t, name), kind, scale=vc.scale)
    var vals = List[Int64]()
    var nul = List[UInt8]()
    swap(vals, vc.v)
    swap(nul, vc.nul)
    _fill_column_owned(t.cols[c], vals^, nul^)


def _to_table(q: BoundQuery, outs: List[ValCol], dicts: DictAccess) raises -> Table:
    var t = Table.empty()
    for i in range(len(outs)):
        _put_column(t, q.out_names[i], outs[i], dicts)
    return t^


def _ones_mask(rows: Int) -> List[UInt8]:
    return List[UInt8](length=(rows + 7) // 8 + 8, fill=0xFF)


struct _Chunks(Movable):
    """The chunks a source is read in, one at a time: a table (its rows, then the frame's changed rows), or a scan's row
    groups (then the changed rows). `load(i)` gives chunk i and sets `mask_addr` for it."""
    var groups: List[Int]            # a scan: the row groups with rows
    var base: Bool                   # a table or frame: the shared table has rows
    var has_delta: Bool
    var scan_mode: Bool
    var used: List[Int]              # the source columns the chunks hold (all of them for a table)
    var sm: _ScanMask
    var ones: List[UInt8]
    var gm: List[UInt8]
    var mask_addr: Int
    var masked: Bool
    var override: Optional[SharedTable]   # a fused join: the probe table is the chunk (the source's own table is only the joined schema)
    var starts: List[Int]            # a scan: the frame row where each of `groups` starts
    var delta_first: Bool            # read the changed rows' chunk before the shared ones (a LIMIT that stops early needs them all)

    def __init__(out self):
        self.override = None
        self.starts = List[Int]()
        self.delta_first = False
        self.groups = List[Int]()
        self.base = False
        self.has_delta = False
        self.scan_mode = False
        self.used = List[Int]()
        self.sm = _ScanMask()
        self.ones = List[UInt8]()
        self.gm = List[UInt8]()
        self.mask_addr = 0
        self.masked = False

    def n(self) -> Int:
        return (len(self.groups) if self.scan_mode else (1 if self.base else 0)) + (1 if self.has_delta else 0)

    def shared(self) -> Int:
        """How many chunks hold the shared rows (the delta chunk, when there is one, comes after them in numbering)."""
        return len(self.groups) if self.scan_mode else (1 if self.base else 0)

    def pick(self, k: Int) -> Int:
        """The k-th chunk to read: in numbering order, or the delta chunk first."""
        if self.delta_first and self.has_delta:
            return self.shared() if k == 0 else k - 1
        return k

    def is_delta(self, i: Int) -> Bool:
        return i >= self.shared()

    def row_base(self, i: Int) -> Int:
        """The frame row at which chunk i's rows start (a shared chunk)."""
        return self.starts[i] if self.scan_mode else 0


def _open_chunks(src: SqlSource, used: List[Int]) raises -> _Chunks:
    var ch = _Chunks()
    ch.scan_mode = src.kind == SRC_SCAN
    ch.used = used.copy()
    ch.masked = len(src.overlay.mask) > 0
    ch.has_delta = len(src.overlay.delta) > 0
    if ch.scan_mode:
        ref scan = src.scan.value()
        var at0 = 0
        for g in range(scan.row_groups()):
            if scan.group_rows(g) > 0:
                ch.groups.append(g)
                ch.starts.append(at0)
            at0 += scan.group_rows(g)
        if ch.masked:
            ch.sm.addr = Int(src.overlay.mask.unsafe_ptr())
            var at = 0
            for g in range(scan.row_groups()):
                ch.sm.starts.append(at)
                at += scan.group_rows(g)
            var mx = len(src.overlay.delta)
            for g in range(scan.row_groups()):
                mx = max(mx, scan.group_rows(g))
            ch.ones = _ones_mask(mx)
    else:
        ch.base = src.rows() > 0
        if ch.masked:
            ch.ones = _ones_mask(len(src.overlay.delta))
    return ch^


def chunk_rows_total(src: SqlSource, ch: _Chunks) -> Int:
    """The rows the chunks hold in all (a job's progress total), without loading them."""
    var n = 0
    if ch.scan_mode:
        for g in ch.groups:
            n += src.scan.value().group_rows(g)
    elif ch.base:
        n += ch.override.value().rows() if ch.override else src.rows()
    if ch.has_delta:
        n += len(src.overlay.delta)
    return n


def _load_chunk(src: SqlSource, mut ch: _Chunks, i: Int) raises -> SharedTable:
    var nb = len(ch.groups) if ch.scan_mode else (1 if ch.base else 0)
    if i >= nb:                                                   # the frame's changed and appended rows
        ch.mask_addr = Int(ch.ones.unsafe_ptr()) if ch.masked else 0
        return delta_table(src, ch.used)
    if not ch.scan_mode:
        ch.mask_addr = Int(src.overlay.mask.unsafe_ptr()) if ch.masked else 0
        if ch.override:
            return ch.override.value()
        return src.table
    var g = ch.groups[i]
    ref scan = src.scan.value()
    var chunk = scan.read_group(g, ch.used)^.share()
    ch.gm = _group_mask(ch.sm, g, scan.group_rows(g)) if ch.masked else List[UInt8]()
    ch.mask_addr = Int(ch.gm.unsafe_ptr()) if len(ch.gm) > 0 else (Int(ch.ones.unsafe_ptr()) if ch.masked else 0)
    return chunk^


comptime LIE = "footer statistics wrong: "


def _check_footer(src: SqlSource, ch: _Chunks, i: Int, chunk: SharedTable) raises:
    """A trusted footer states each column chunk's null count: a decoded row group that holds another number shows the footer
    cannot be believed (a NULL read as a value would otherwise be counted as 0)."""
    if not ch.scan_mode or i >= len(ch.groups):
        return
    ref scan = src.scan.value()
    for k in range(len(ch.used)):
        var stated = scan.group_null_count(ch.groups[i], ch.used[k])
        var real = chunk.null_count(k)
        if stated >= 0 and stated != real:
            raise Error(LIE + "row group " + String(ch.groups[i]) + " states " + String(stated) + (" null" if stated == 1 else " nulls") +
                        " in column `" + scan.name(ch.used[k]) + "` and holds " + String(real))



def _all(n: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(n):
        out.append(i)
    return out^


struct _Inputs(Movable):
    """How the program sees the source: which columns the chunks hold and where, which can be NULL, whether a mask applies."""
    var used: List[Int]
    var pos: List[Int]
    var chunk_cols: Int
    var nullable: List[Bool]
    var masked: Bool
    var scan_mode: Bool

    def __init__(out self):
        self.used = List[Int]()
        self.pos = List[Int]()
        self.chunk_cols = 0
        self.nullable = List[Bool]()
        self.masked = False
        self.scan_mode = False


def _inputs(src: SqlSource, q: BoundQuery, trust: Bool = True) raises -> _Inputs:
    var inp = _Inputs()
    var n_cols = src.table.columns()
    inp.used = collect_columns(q)
    inp.scan_mode = src.kind == SRC_SCAN
    # a table or frame is seen as it is; a scan's row groups hold only the columns used
    inp.pos = List[Int](length=n_cols, fill=-1)
    inp.chunk_cols = n_cols
    if inp.scan_mode:
        if len(inp.used) == 0 and n_cols > 0:
            inp.used.append(0)
        for k in range(len(inp.used)):
            inp.pos[inp.used[k]] = k
        inp.chunk_cols = len(inp.used)
        for c in inp.used:
            if src.table.kind(c) == TEXT:
                src.scan.value().ensure_words(c)
    else:
        inp.used = _all(n_cols)
        for c in range(n_cols):
            inp.pos[c] = c
    for c in range(n_cols):
        inp.nullable.append((not trust or src.scan.value().null_count(c) != 0 if inp.scan_mode else src.table.null_count(c) > 0) or _delta_nulls(src, c))
    inp.masked = len(src.overlay.mask) > 0
    return inp^
