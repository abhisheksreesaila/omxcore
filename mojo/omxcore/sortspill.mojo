"""A sort bigger than its memory budget (F-167): sorted runs on disk, merged.

A row query with ORDER BY (`rows.mojo`) collects the selected rows and sorts them. When the collected rows would pass the sort budget
(`PlannerOptions.sort_budget`) the rows so far are sorted as a run, written to one temporary file and freed; the reading goes on; the
last rows are a run too; then the runs are merged k-ways and the rows come out in the order a sort in memory gives, in batches: the
result table (when it fits) or row groups of a Parquet file (`sql_to_parquet`: a file bigger than memory sorted into a new file).

One definition of the order. A run is sorted by `_order_rows` (exec.mojo), the sort in memory; the merge compares rows with
`_SortCols.cmp`, the comparison the top-k heap uses, over keys made by `_sort_keys`. Ties keep the order the rows had in the source:
every row carries its number (`seq`: its place in the read, or a frame's row number when the frame has edits) as the last key.

A TEXT key is a dictionary code, and a code's place is relative to the words present: the runs keep codes, and when the last run is
written the words that occurred are ranked once (`_sort_keys` over them), so a merge compares ranks that mean the same in every run.

Run file layout: run after run, each column by column: n values (Int64), then n NULL flags (bytes); the last column is `seq`. A block
of a run is read with one seek and read per column. The file is removed when the sorter is dropped: on success, on an error, on a cancel.
"""
from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
from std.os import getenv
from std.time import perf_counter_ns
from .table import TEXT, FLOAT, INT
from .job import JobCtl
from .bind import SortKey
from .sqleval import ValCol
from .sqlprog import DictAccess
from .exec import _order_rows, _sort_keys, _SortCols
from .kernels.pipeline_cpu import VEC

comptime SORT_BUDGET_DEFAULT = 2 << 30         # bytes of rows a sort holds in memory at once, when the caller sets no budget
comptime ROW_COLLECT_BYTES = 9                 # a collected row costs this many bytes a column: its value and its NULL flag
comptime SORT_WORK_BASE = 96                   # sorting a run: the row order, one radix pass's buffers and the gathered column being written, per row ...

comptime SORT_WORK_PER_KEY = 16                # ... and per key: its copy and the numbers made of it (measured: docs/omxcore-mojo-api.md, F-167)
comptime MERGE_MIN_BLOCK = 64                  # rows of each run the merge keeps in memory, at the least
comptime MERGE_MAX_BLOCK = 1 << 16
comptime CANCEL_EVERY = 1 << 16                # merged rows between two cancel checks


def sort_budget_bytes(requested: Int) -> Int:
    return requested if requested > 0 else SORT_BUDGET_DEFAULT


def run_row_cap(budget: Int, columns: Int, keys: Int, forced: Int) -> Int:
    """The most rows a run may hold: what fits the budget with the sort's own working memory (`forced` > 0: that many, a test's override).
    A multiple of the kernel's block, at least one block."""
    if forced > 0:
        return max(forced, 1)
    var per_row = columns * ROW_COLLECT_BYTES + SORT_WORK_BASE + keys * SORT_WORK_PER_KEY
    return max(VEC, (budget // per_row // VEC) * VEC)


def _spill_name(dir: String) -> String:
    var base = dir
    if base.byte_length() == 0:
        base = getenv("TMPDIR")
    if base.byte_length() == 0:
        base = "/tmp"
    if base.byte_length() > 1 and base.endswith("/"):
        var trimmed = String(base[byte=:base.byte_length() - 1])
        base = trimmed^
    var uniq = unsafe_alloc[Int64](1)             # the process, the address of a fresh allocation and the clock: two sorts never share a file
    var name = base + "/omxcore-sort-" + String(external_call["getpid", Int32]()) + "-" + String(Int(uniq)) + "-" + String(perf_counter_ns()) + ".spill"
    uniq.free()
    return name^


def _write_all(fd: Int, addr: Int, count: Int, offset: Int, path: String) raises:
    if external_call["lseek", Int](fd, offset, Int32(0)) < 0:          # (lseek + write, not pwrite: the Mac app's libSystem stub has the first two)
        raise Error("couldn't write the sort's temporary file `" + path + "`")
    var done = 0
    while done < count:
        var n = external_call["write", Int](fd, UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=addr + done), count - done)
        if n <= 0:
            raise Error("couldn't write the sort's temporary file `" + path + "`: the disk may be full, or the folder not writable (PlannerOptions.spill_dir)")
        done += n


def _read_all(fd: Int, addr: Int, count: Int, offset: Int, path: String) raises:
    if external_call["lseek", Int](fd, offset, Int32(0)) < 0:
        raise Error("couldn't read the sort's temporary file `" + path + "` back")
    var done = 0
    while done < count:
        var n = external_call["read", Int](fd, UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=addr + done), count - done)
        if n <= 0:
            raise Error("couldn't read the sort's temporary file `" + path + "` back (was it removed?)")
        done += n


# (the lists are parameters here: the caller keeps them alive for the whole call, which a local's address handed to write / read would not)
def _put64(fd: Int, v: List[Int64], offset: Int, path: String) raises:
    _write_all(fd, Int(v.unsafe_ptr()), len(v) * 8, offset, path)


def _put8(fd: Int, v: List[UInt8], offset: Int, path: String) raises:
    _write_all(fd, Int(v.unsafe_ptr()), len(v), offset, path)


def _get64(fd: Int, mut v: List[Int64], offset: Int, path: String) raises:
    _read_all(fd, Int(v.unsafe_ptr()), len(v) * 8, offset, path)


def _get8(fd: Int, mut v: List[UInt8], offset: Int, path: String) raises:
    _read_all(fd, Int(v.unsafe_ptr()), len(v), offset, path)


struct _Run(Movable):
    """One sorted run in the file, and the block of it the merge holds."""
    var start: Int                   # byte offset of the run in the file
    var rows: Int
    var has_nul: List[Bool]          # per column (the last is seq): any NULL in the run, else its flags were not read
    var next: Int                    # the first row not yet loaded
    var cols: List[ValCol]           # the block: every column's values and flags
    var sc: _SortCols                # the block's sort keys (and, last, seq)
    var cur: Int                     # the row of the block the merge is at
    var n: Int                       # rows in the block

    def __init__(out self, start: Int, rows: Int, has_nul: List[Bool]):
        self.start = start
        self.rows = rows
        self.has_nul = has_nul.copy()
        self.next = 0
        self.cols = List[ValCol]()
        self.sc = _SortCols()
        self.cur = 0
        self.n = 0


struct SpillSort(Movable):
    """The runs of one sort. `add_run` sorts the rows given and writes them; `begin_merge` and `merge_batch` read them back in order."""
    var path: String
    var file: FileHandle
    var fd: Int
    var end: Int                     # bytes written
    var order: List[SortKey]         # the ORDER BY keys (desc, nulls_first)
    var key_col: List[Int]           # per key: its column among the rows
    var kind: List[Int]              # per column
    var scale: List[Int]
    var src: List[Int]
    var any_null: List[Bool]         # per column: a NULL in some run
    var seen: List[List[UInt8]]      # per key of TEXT: the codes that occur (1 at the code)
    var rank_of: List[List[Int64]]   # per key of TEXT: the rank of each code occurring, once the merge has begun
    var runs: List[_Run]
    var heap: List[Int]              # the runs by the sort order of their current rows (a binary heap)
    var block_rows: Int
    var to_skip: Int
    var remaining: Int               # rows still to give (-1: all)
    var merged: Int                  # rows given since the last cancel check
    var job: JobCtl
    var total_rows: Int

    def __init__(out self, dir: String, order: List[SortKey], key_col: List[Int], templates: List[ValCol], job: JobCtl) raises:
        var path = _spill_name(dir)
        self.path = String()                      # (set once the file exists: only then does dropping the sorter remove it)
        self.fd = -1
        self.end = 0
        self.order = order.copy()
        self.key_col = key_col.copy()
        self.kind = List[Int]()
        self.scale = List[Int]()
        self.src = List[Int]()
        self.any_null = List[Bool]()
        for t in templates:
            self.kind.append(t.kind)
            self.scale.append(t.scale)
            self.src.append(t.src)
            self.any_null.append(False)
        self.seen = List[List[UInt8]]()
        self.rank_of = List[List[Int64]]()
        for k in range(len(key_col)):
            self.seen.append(List[UInt8]())
            self.rank_of.append(List[Int64]())
        self.runs = List[_Run]()
        self.heap = List[Int]()
        self.block_rows = MERGE_MIN_BLOCK
        self.to_skip = 0
        self.remaining = -1
        self.merged = 0
        self.job = job
        self.total_rows = 0
        try:
            self.file = open(path, "w")
        except:
            raise Error("couldn't write a temporary file for the sort in `" + (dir if dir.byte_length() > 0 else "the temporary folder") +
                        "`: no such folder, or no permission to write there (PlannerOptions.spill_dir)")
        self.path = path^
        self.fd = self.file._get_raw_fd()

    def __del__(deinit self):
        if self.path.byte_length() > 0:
            _ = external_call["unlink", Int32](self.path.as_c_string_span().ptr())

    def run_count(self) -> Int:
        return len(self.runs)

    # ── writing the runs ─────────────────────────────────────────────────────────────────────────

    def add_run(mut self, mut cols: List[ValCol], seq: List[Int64], base: Int, dicts: DictAccess) raises:
        """Sorts the rows in `cols` (stable) and writes them as a run; the columns are emptied. `seq`: each row's number when it is not
        `base` + its place (a frame with edits); the rows then also tie by it."""
        var n = len(cols[0].v)
        if n == 0:
            return
        self.job.check()
        var kc = List[ValCol]()
        var order = self.order.copy()
        for k in self.key_col:
            kc.append(cols[k].copy())
        if len(seq) > 0:
            var s = ValCol(INT.value, 0, n)
            for i in range(n):
                s.v[i] = seq[i]
            kc.append(s^)
            order.append(SortKey(0, False, False))
        var perm = _order_rows(kc, order, dicts, n)
        kc = List[ValCol]()
        self.job.check()
        for j in range(len(self.key_col)):                    # the words of a TEXT key that occur, for the ranks
            ref vc = cols[self.key_col[j]]
            if vc.kind == TEXT.value:
                var mx = -1
                for i in range(n):
                    if not vc.is_null(i):
                        mx = max(mx, Int(vc.v[i]))
                if mx >= len(self.seen[j]):
                    self.seen[j].resize(mx + 1, 0)
                for i in range(n):
                    if not vc.is_null(i):
                        self.seen[j][Int(vc.v[i])] = 1
        var ncols = len(cols)
        var has_nul = List[Bool]()
        var start = self.end
        var vals = List[Int64](length=n, fill=0)
        var nuls = List[UInt8](length=n, fill=0)
        for c in range(ncols + 1):                            # the columns, then seq
            var any = False
            if c < ncols:
                ref vc = cols[c]
                for i in range(n):
                    vals[i] = vc.v[perm[i]]
                if len(vc.nul) > 0:
                    for i in range(n):
                        var f = vc.nul[perm[i]]
                        nuls[i] = f
                        if f != 0:
                            any = True
                else:
                    for i in range(n):
                        nuls[i] = 0
                if any:
                    self.any_null[c] = True
            else:
                for i in range(n):
                    vals[i] = seq[perm[i]] if len(seq) > 0 else Int64(base + perm[i])
                    nuls[i] = 0
            has_nul.append(any)
            var at = start + c * n * (8 + 1)
            _put64(self.fd, vals, at, self.path)
            _put8(self.fd, nuls, at + n * 8, self.path)
        self.end = start + (ncols + 1) * n * 9
        self.runs.append(_Run(start, n, has_nul))
        self.total_rows += n
        for c in range(ncols):                                # free the run's rows: a cleared list keeps its memory
            cols[c].v = List[Int64]()
            cols[c].nul = List[UInt8]()

    def column_has_nulls(self, c: Int) -> Bool:
        return self.any_null[c]

    # ── merging them ─────────────────────────────────────────────────────────────────────────────

    def begin_merge(mut self, dicts: DictAccess, budget: Int, skip: Int, limit: Int) raises:
        """Ranks the TEXT keys' words, sizes the blocks to the budget, loads the first block of every run. `skip` rows are dropped from the
        front of the order, then `limit` (-1: all) are given."""
        self.file.close()
        self.file = open(self.path, "r")
        self.fd = self.file._get_raw_fd()
        for j in range(len(self.key_col)):
            if self.kind[self.key_col[j]] != TEXT.value:
                continue
            var universe = ValCol(TEXT.value, 0, 0)
            universe.src = self.src[self.key_col[j]]
            for code in range(len(self.seen[j])):
                if self.seen[j][code] != 0:
                    universe.v.append(Int64(code))
            var ranks = _sort_keys(universe, dicts)
            self.rank_of[j] = List[Int64](length=len(self.seen[j]), fill=0)
            for i in range(len(universe.v)):
                self.rank_of[j][Int(universe.v[i])] = ranks[i]
            self.seen[j] = List[UInt8]()
        var k = max(len(self.runs), 1)
        var per_row = (len(self.kind) + len(self.key_col) + 2) * 9 + 16
        self.block_rows = max(MERGE_MIN_BLOCK, min(MERGE_MAX_BLOCK, budget // (2 * k * per_row)))
        self.to_skip = skip
        self.remaining = limit
        for r in range(len(self.runs)):
            self._load(r, dicts)
            if self.runs[r].n > 0:
                self.heap.append(r)
        var i = len(self.heap) // 2 - 1
        while i >= 0:
            self._sift(i)
            i -= 1

    def _load(mut self, r: Int, dicts: DictAccess) raises:
        """Reads the next block of run r (n = 0 when the run is used up) and makes its sort keys."""
        var nc = len(self.kind)
        var rows = self.runs[r].rows
        var from_row = self.runs[r].next
        var m = min(self.block_rows, rows - from_row)
        self.runs[r].n = m
        self.runs[r].cur = 0
        self.runs[r].cols = List[ValCol]()
        self.runs[r].sc = _SortCols()
        if m <= 0:
            return
        var start = self.runs[r].start
        var seqs = List[Int64](length=m, fill=0)
        for c in range(nc + 1):
            var at = start + c * rows * 9
            if c == nc:
                _get64(self.fd, seqs, at + from_row * 8, self.path)
                break
            var vc = ValCol(self.kind[c], self.scale[c], m)
            vc.src = self.src[c]
            _get64(self.fd, vc.v, at + from_row * 8, self.path)
            if self.runs[r].has_nul[c]:
                vc.nul = List[UInt8](length=m, fill=0)
                _get8(self.fd, vc.nul, at + rows * 8 + from_row, self.path)
            self.runs[r].cols.append(vc^)
        for j in range(len(self.key_col)):
            var src = self.runs[r].cols[self.key_col[j]].copy()
            if src.kind == TEXT.value:                        # codes -> the ranks all the runs share
                var rk = ValCol(INT.value, 0, m)
                for i in range(m):
                    if not src.is_null(i):
                        rk.v[i] = self.rank_of[j][Int(src.v[i])]
                rk.nul = src.nul.copy()
                src = rk^
            var keys = _sort_keys(src, dicts)
            self.runs[r].sc.keys.append(keys^)
            self.runs[r].sc.nulls.append(src.nul.copy())
            self.runs[r].sc.desc.append(self.order[j].desc)
            self.runs[r].sc.nulls_first.append(self.order[j].nulls_first)
        self.runs[r].sc.keys.append(seqs^)
        self.runs[r].sc.nulls.append(List[UInt8]())
        self.runs[r].sc.desc.append(False)
        self.runs[r].sc.nulls_first.append(False)
        self.runs[r].next = from_row + m

    def _less(self, a: Int, b: Int) -> Bool:
        return self.runs[a].sc.cmp(self.runs[a].cur, self.runs[b].sc, self.runs[b].cur) < 0

    def _sift(mut self, start: Int):
        var i = start
        var n = len(self.heap)
        while True:
            var l = 2 * i + 1
            var best = i
            if l < n and self._less(self.heap[l], self.heap[best]):
                best = l
            if l + 1 < n and self._less(self.heap[l + 1], self.heap[best]):
                best = l + 1
            if best == i:
                return
            var t = self.heap[i]
            self.heap[i] = self.heap[best]
            self.heap[best] = t
            i = best

    def merge_batch(mut self, want: Int, dicts: DictAccess) raises -> List[ValCol]:
        """The next `want` rows (fewer at the end, none when used up) in sort order, one column per column of the rows, after the rows
        skipped and within the limit."""
        var out = List[ValCol]()
        for c in range(len(self.kind)):
            var vc = ValCol(self.kind[c], self.scale[c], 0)
            vc.src = self.src[c]
            out.append(vc^)
        var got = 0
        while got < want and len(self.heap) > 0 and self.remaining != 0:
            var r = self.heap[0]
            if self.to_skip > 0:
                self.to_skip -= 1
            else:
                ref run = self.runs[r]
                for c in range(len(out)):
                    ref src = run.cols[c]
                    ref dst = out[c]
                    if len(src.nul) > 0:
                        if len(dst.nul) == 0:
                            dst.nul = List[UInt8](length=len(dst.v), fill=0)
                        dst.nul.append(src.nul[run.cur])
                    elif len(dst.nul) > 0:
                        dst.nul.append(0)
                    dst.v.append(src.v[run.cur])
                got += 1
                if self.remaining > 0:
                    self.remaining -= 1
            self.merged += 1
            if self.merged >= CANCEL_EVERY:
                self.merged = 0
                self.job.check()
            self.runs[r].cur += 1
            if self.runs[r].cur >= self.runs[r].n:
                self._load(r, dicts)
                if self.runs[r].n == 0:
                    self.heap[0] = self.heap[len(self.heap) - 1]
                    _ = self.heap.pop()
            if len(self.heap) > 0:
                self._sift(0)
        return out^
