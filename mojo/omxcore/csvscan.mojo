"""F-178: a CSV file bigger than memory, read a block at a time (docs/omxcore-mojo-api.md section 2b).

`scan_csv` (parquet.mojo, next to `scan_parquet`) opens a CSV so that it behaves as a Parquet scan does: row groups, queries, SQL, Frames
with edits, save. This file holds what is particular to text: finding the row groups, and decoding one.

Opening reads the whole file once, a block at a time, and keeps only a small index:
  1. The first block gives the header and the kinds (csv.mojo `_infer_shape`, the same rules as `read_csv`: the first `infer_rows` rows
     that fit in `block_bytes`).
  2. A pass over the file finds the row groups (about `group_rows` rows each, never more than `block_bytes`; a row longer than the block
     makes a bigger window, up to `max_row_bytes`) and parses every cell once to count nulls, collect the failed-row report, and see
     whether a DECIMAL column needs more places than the first rows showed (then the pass is run again at the wider scale, so the answer
     for a column never changes under a query). Nothing but the index and the report is kept: a group is the file's byte range [offset,
     offset + bytes), a row count and a null count per column.
A row group is found at a row boundary, and a row boundary is a newline outside quotes: the same state machine `read_csv` splits its
chunks with (`csv._step`), so the groups cut the file exactly where the whole-file reader's rows end, and a quoted field with a newline
in it is never split, whatever the block boundary. Reading a group maps its bytes, parses the columns asked for with the reader's own
`ChunkSink`, and lets the bytes go: the working set is a group, whatever the size of the file.
"""
from std.memory import Pointer
from .csv import (BP, CsvOptions, CsvColumn, FailedRow, ChunkSink, CsvShape, _Fail, _walk, _infer_shape, _delimiter, _step, _A, _Q, _NL,
                  _MAX_FAILED, first_invalid_utf8)
from .mapped import _Mapped
from .table import Table, SharedTable, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT


struct CsvFailure(Copyable, Movable):
    """A cell that did not fit its column's kind (now a null): where it is, what it said and why it failed."""

    var row: Int          # the data row, counting from 1 (the header is not a row)
    var column: String
    var text: String
    var reason: String

    def __init__(out self, row: Int, column: String, text: String, reason: String):
        self.row = row
        self.column = column
        self.text = text
        self.reason = reason

    def message(self) -> String:
        return "row " + String(self.row) + ", column `" + self.column + "`: `" + self.text + "` " + self.reason


struct CsvReport(Copyable, Movable):
    """What opening a CSV found: `read_csv`'s import report, for the whole file. `columns[c].failed` holds a column's failed cells (the first
    1,000 in file order; `failed_count` is the true number), `ragged` the rows with more cells than the header (and the rows that end in an
    unmatched quote). `columns[c].distinct` is 0: the words of a TEXT column are counted by `scan.word_count(c)` when asked."""

    var rows: Int
    var columns: List[CsvColumn]
    var skipped: List[String]
    var ragged: List[FailedRow]
    var ragged_count: Int

    def __init__(out self):
        self.rows = 0
        self.columns = List[CsvColumn]()
        self.skipped = List[String]()
        self.ragged = List[FailedRow]()
        self.ragged_count = 0

    def failed_count(self) -> Int:
        var n = 0
        for c in self.columns:
            n += c.failed_count
        return n

    def failures(self) -> List[CsvFailure]:
        """Every failed cell kept (at most 1,000 a column), in file order."""
        var keys = List[Int]()
        for c in range(len(self.columns)):
            for k in range(len(self.columns[c].failed)):
                keys.append((self.columns[c].failed[k].row << 22) | (c << 10) | k)
        sort(keys)
        var out = List[CsvFailure]()
        for key in keys:
            var c = (key >> 10) & 0xFFF
            var f = self.columns[c].failed[key & 0x3FF].copy()
            out.append(CsvFailure(f.row, self.columns[c].name, f.text, f.reason))
        return out^


struct CsvSource(Movable):
    """The index of an opened CSV file."""

    var delim: UInt8
    var header_cells: Int
    var first_row: Int            # the byte offset of the first data row
    var names: List[String]
    var kinds: List[Int]
    var scales: List[Int]
    var offs: List[Int]           # per row group: its first byte
    var lens: List[Int]           # and its length in bytes (it ends at a row boundary)
    var rows: List[Int]
    var nulls: List[List[Int]]    # [group][column]: the cells that are null (empty, short rows, failed cells)
    var report: CsvReport

    def __init__(out self):
        self.delim = 44
        self.header_cells = 0
        self.first_row = 0
        self.names = List[String]()
        self.kinds = List[Int]()
        self.scales = List[Int]()
        self.offs = List[Int]()
        self.lens = List[Int]()
        self.rows = List[Int]()
        self.nulls = List[List[Int]]()
        self.report = CsvReport()


def _find_cut(p: BP, n: Int, delim: UInt8, max_lines: Int, eof: Bool) -> Tuple[Int, Int]:
    """(cut, lines): the end of the first complete rows of `p[:n]` and how many lines they are: just after the `max_lines`-th newline that is
    outside quotes, or after the last one in the window; at the end of the file, the end of the window (a last row needs no newline). Cut 0:
    no row ends in the window. `p` starts at a row start."""
    var s = _A
    var safe = 0
    var lines = 0
    for i in range(n):
        var b = p[i]
        var was_quoted = s == _Q
        s = _step(s, b, delim)
        if b == _NL and not was_quoted:
            safe = i + 1
            lines += 1
            if lines >= max_lines:
                return (safe, lines)
    if eof and safe < n:
        return (n, lines + 1)
    return (safe, lines)


def _check_start(p: BP, n: Int, path: String) raises:
    if n >= 2 and ((p[0] == 0xFF and p[1] == 0xFE) or (p[0] == 0xFE and p[1] == 0xFF)):
        raise Error("`" + path + "` is UTF-16 (it starts with a byte order mark), and omxcore reads UTF-8 only")


def _too_long(path: String, row: Int, limit: Int) -> Error:
    return Error("`" + path + "`: row " + String(row) + " is longer than " + String(limit) + " bytes (an unmatched quote makes the rest of the file one row). " +
                 "Fix the quote, or raise CsvOptions.max_row_bytes")


def csv_open(fd: Int, size: Int, path: String, options: CsvOptions) raises -> CsvSource:
    """Opens a CSV for streaming (see the top of this file). `fd` is the open file, `size` its length."""
    if options.max_columns < 1:
        raise Error("max_columns must be at least 1, not " + String(options.max_columns))
    if options.group_rows < 1 or options.block_bytes < 1 or options.max_row_bytes < 1:
        raise Error("group_rows, block_bytes and max_row_bytes must be at least 1")
    var src = CsvSource()
    src.delim = _delimiter(path, options.delimiter)
    if size == 0:
        return src^
    # the sample: the first rows that fit in a block (a longer first row makes the window grow)
    var window = min(size, options.block_bytes)
    var shape = CsvShape()
    var want_lines = options.infer_rows + (1 if options.has_header else 0)
    while True:
        var m = _Mapped(fd, 0, window, path)
        var p = BP(unsafe_from_address=m.data)
        _check_start(p, window, path)
        var found = _find_cut(p, window, src.delim, max(want_lines, 1), window >= size)
        var cut = found[0]
        if found[1] < (2 if options.has_header else 1) and window < size:       # the header and one row at least, however small the block
            m.release()
            if window >= options.max_row_bytes:
                raise _too_long(path, 1, options.max_row_bytes)
            window = min(size, window * 2)
            continue
        var bad = first_invalid_utf8(p, cut)
        if bad >= 0:
            m.release()
            raise Error("`" + path + "` is not valid UTF-8 (byte " + String(bad) + ")")
        shape = _infer_shape(p, cut, src.delim, options)
        m.release()
        break
    if shape.header_cells == 0:       # nothing but blank lines
        return src^
    src.header_cells = shape.header_cells
    src.first_row = shape.first_row
    src.names = shape.names.copy()
    src.kinds = shape.kinds.copy()
    src.scales = shape.scales.copy()
    var samples = List[List[String]]()
    for c in range(len(src.names)):
        samples.append(shape.infer.cols[c].sample.copy())
    for _ in range(2):                # the second pass only when a DECIMAL column had to widen
        if not _sweep(fd, size, path, options, src, samples, shape.skipped):
            break
    return src^


def _sweep(fd: Int, size: Int, path: String, options: CsvOptions, mut src: CsvSource, samples: List[List[String]], skipped: List[String]) raises -> Bool:
    """One pass over the data rows: finds the row groups and parses every cell once for the report and the null counts. True: a DECIMAL
    column of scale 2 met a value with 3 or 4 places; its scale is now 4 and the pass has to be run again."""
    src.offs.clear()
    src.lens.clear()
    src.rows.clear()
    src.nulls.clear()
    src.report = CsvReport()
    src.report.skipped = skipped.copy()
    var ncols = len(src.names)
    for c in range(ncols):
        var col = CsvColumn(src.names[c], ColumnKind(src.kinds[c]), src.scales[c])
        col.sample = samples[c].copy()
        src.report.columns.append(col^)
    var widen = List[Bool](length=ncols, fill=False)
    var any_widen = False
    var at = src.first_row
    var row_base = 0
    while at < size:
        var window = min(options.block_bytes, size - at)
        var m = _Mapped()
        var cut = 0
        while True:
            m = _Mapped(fd, at, window, path)
            cut = _find_cut(BP(unsafe_from_address=m.data), window, src.delim, options.group_rows, at + window >= size)[0]
            if cut > 0:
                break
            m.release()
            if window >= options.max_row_bytes:
                raise _too_long(path, row_base + 1, options.max_row_bytes)
            window = min(size - at, window * 2)
        var p = BP(unsafe_from_address=m.data)
        var bad = first_invalid_utf8(p, cut)
        if bad >= 0:
            m.release()
            raise Error("`" + path + "` is not valid UTF-8 (byte " + String(at + bad) + ")")
        var sink = ChunkSink(src.kinds, src.scales, src.header_cells, List[Bool](), False)
        _ = _walk(p, 0, cut, src.delim, Int.MAX, sink)
        m.release()
        if sink.rows > 0:
            src.offs.append(at)
            src.lens.append(cut)
            src.rows.append(sink.rows)
            var nulls = List[Int]()
            for c in range(ncols):
                ref cc = sink.cols[c]
                nulls.append(len(cc.nulls))
                ref rep = src.report.columns[c]
                rep.empty += cc.empty
                rep.failed_count += cc.fail_count
                rep.rounded += cc.rounded
                for k in range(len(cc.fails)):
                    if len(rep.failed) < _MAX_FAILED:
                        rep.failed.append(FailedRow(row_base + cc.fails[k].row + 1, cc.fails[k].text, cc.fails[k].reason))
                if len(cc.wide_rows) > 0:
                    widen[c] = True
                    any_widen = True
            src.nulls.append(nulls^)
            src.report.ragged_count += sink.ragged_count
            for k in range(len(sink.ragged)):
                if len(src.report.ragged) < _MAX_FAILED:
                    src.report.ragged.append(FailedRow(row_base + sink.ragged[k].row + 1, sink.ragged[k].text, sink.ragged[k].reason))
        row_base += sink.rows
        at += cut
    src.report.rows = row_base
    if any_widen:
        for c in range(ncols):
            if widen[c]:
                src.scales[c] = 4
        return True
    return False


def csv_parse_group(src: CsvSource, fd: Int, path: String, g: Int, want: List[Bool], text_words: Bool) raises -> ChunkSink:
    """Row group `g` parsed: the columns marked in `want` (empty: all), a TEXT column's words collected unless `text_words` is False."""
    var m = _Mapped(fd, src.offs[g], src.lens[g], path)
    var sink = ChunkSink(src.kinds, src.scales, src.header_cells, want, text_words)
    _ = _walk(BP(unsafe_from_address=m.data), 0, src.lens[g], src.delim, Int.MAX, sink)
    m.release()
    if sink.rows != src.rows[g]:
        raise Error("the file changed since it was opened (`" + path + "`)")
    return sink^


def csv_table(mut sink: ChunkSink, cols: List[Int], schema: SharedTable, dicts: Table, ext: Int) raises -> Table:
    """The columns `cols` of a parsed row group as a table whose TEXT codes are the file-wide dictionary's (`dicts`, built beforehand).
    `ext` as in `ParquetScan.read_group_into`: the address of a List[Int] of destination addresses, the values written there and
    the table keeping only validity and null counts; 0: the values stay in the table."""
    var t = Table.empty()
    var n = sink.rows
    for k in range(len(cols)):
        var c = cols[k]
        _ = t.add_column(schema.name(c), schema.kind(c), scale=schema.scale(c))
        ref cc = sink.cols[c]
        if cc.kind == TEXT.value and len(cc.words.starts) > 0:
            var remap = List[Int64]()
            for w in range(len(cc.words.starts)):
                var code = dicts.code_of(c, cc.words.word(w))
                if code < 0:
                    raise Error("internal error: a word of column `" + schema.name(c) + "` is not in the file's dictionary (the file changed?)")
                remap.append(Int64(code))
            for r in range(n):
                cc.vals[r] = remap[Int(cc.vals[r])]
            for r in cc.nulls:
                cc.vals[r] = 0
        ref col = t.cols[k]
        if len(cc.nulls) > 0:
            col.validity = List[UInt8](length=(n + 7) // 8, fill=0xFF)
            col.nulls = len(cc.nulls)
            for r in cc.nulls:
                col.validity[r >> 3] &= ~(UInt8(1) << UInt8(r & 7))
        if ext == 0:
            swap(col.values, cc.vals)
        else:
            var dest = UnsafePointer[List[Int], MutAnyOrigin](unsafe_from_address=ext)[][k]
            var d = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=dest)
            for r in range(n):
                d[r] = cc.vals[r]
    return t^
