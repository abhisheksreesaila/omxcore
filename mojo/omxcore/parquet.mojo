"""Parquet files with no Python (docs/omxcore-mojo-api.md section 2): `write_parquet` and `read_parquet`.

The `.omx` document is one Parquet file whose key-value metadata holds one JSON string under the key "omx".

How omxcore's kinds are stored (any Parquet reader understands them):
  INT      INT64
  DECIMAL  INT64 with the DECIMAL(18, scale) logical type (the stored value is the scaled integer, as in memory)
  DATE     INT32 with the DATE logical type (days since 1970)
  BOOL     BOOLEAN
  FLOAT    DOUBLE (read: DOUBLE, and FLOAT widened to float64)
  TEXT     BYTE_ARRAY (STRING): a dictionary page with the column's words, data pages of dictionary indices
A column with nulls is OPTIONAL: its definition levels are the validity bitmap's bits (Parquet packs them the same
way, least significant first), written as bit-packed runs. Pages are compressed with zstd (default), snappy or not
at all; one data page per column per row group. `write_parquet` writes to `path + ".tmp"` and renames it into place,
so a crash or a refused table never destroys an existing file.

Reading takes what other writers produce: signed and unsigned integers of 8 to 64 bits (a UINT_64 past the int64 range
is refused, naming the column), TEXT only for string types (a BYTE_ARRAY with no string type is BINARY and is refused,
as are TIME columns, so invalid UTF-8 never reaches a table), and a timestamp becomes its UTC day: isAdjustedToUTC
or not, the stored instant is floored to a day since 1970 in UTC, never shifted to a local zone.

The file is untrusted input: every length, count, run and bit width is checked against what the page or footer can
hold before anything is allocated or read, and a bad file raises "(corrupt file)", never crashes.

F-138b, the no-codec build: zstd and snappy are C libraries. An Apple app links neither, so there (any target but
Linux, or `-D MXFRAME_NO_CODECS` on Linux to test it) `write_parquet` saves UNCOMPRESSED whatever `compression` says,
and `read_parquet` refuses a zstd or snappy column chunk with an error naming the column. No codec symbol is
referenced, so nothing links. On Linux both codecs work as before.
"""
from std.atomic import Atomic
from std.time import perf_counter_ns
from std.memory.alloc import unsafe_alloc
from std.ffi import external_call
from std.sys import is_defined, CompilationTarget
from std.bit import pop_count, byte_swap
from .job import JobCtl, STAGE_READ, STAGE_WRITE, THREADS      # F-175
from .kernels._fpenv import par_n, core_count
from std.collections import Dict
from std.memory import ArcPointer
from .table import Table, SharedTable, Column, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, F32_COLUMN
from .mapped import _Mapped, _MAP_ALIGN
from .csv import CsvOptions
from .csvscan import CsvSource, CsvReport, csv_open, csv_parse_group, csv_table
from .thrift import ThriftWriter, ThriftReader, T_I32, T_I64, T_BINARY, T_LIST, T_STRUCT, T_TRUE, T_FALSE, T_STOP, _valid_utf8

# Parquet enums (parquet.thrift)
comptime PT_BOOLEAN = 0
comptime PT_INT32 = 1
comptime PT_INT64 = 2
comptime PT_INT96 = 3
comptime PT_FLOAT = 4
comptime PT_DOUBLE = 5
comptime PT_BYTE_ARRAY = 6
comptime PT_FIXED = 7
comptime REQUIRED = 0
comptime OPTIONAL = 1
comptime REPEATED = 2
comptime CT_UTF8 = 0
comptime CT_ENUM = 4
comptime CT_DECIMAL = 5
comptime CT_DATE = 6
comptime CT_TIME_MILLIS = 7
comptime CT_TIME_MICROS = 8
comptime CT_TIMESTAMP_MILLIS = 9
comptime CT_TIMESTAMP_MICROS = 10
comptime CT_UINT_8 = 11
comptime CT_UINT_64 = 14
comptime CT_JSON = 19
comptime ENC_PLAIN = 0
comptime ENC_PLAIN_DICTIONARY = 2
comptime ENC_RLE = 3
comptime ENC_BIT_PACKED = 4
comptime ENC_DELTA_BINARY_PACKED = 5
comptime ENC_RLE_DICTIONARY = 8
comptime ENC_BYTE_STREAM_SPLIT = 9
comptime PAGE_DATA = 0
comptime PAGE_DICTIONARY = 2
comptime PAGE_DATA_V2 = 3

# compression
comptime UNCOMPRESSED = 0
comptime SNAPPY = 1
comptime GZIP = 2
comptime ZSTD = 6

comptime NO_CODECS = is_defined["MXFRAME_NO_CODECS"]() or not CompilationTarget.is_linux()

comptime DEFAULT_ROW_GROUP = 1 << 20
comptime INT32_MAX = 2147483647
comptime MAX_DECIMAL = 1_000_000_000_000_000_000      # DECIMAL(18, s): the stored integer has at most 18 digits


# ── bytes ────────────────────────────────────────────────────────────────────────────────────────────

def _put32(mut b: List[UInt8], v: Int):
    for k in range(4):
        b.append(UInt8((v >> (8 * k)) & 0xFF))


def _put64(mut b: List[UInt8], v: Int64):
    for k in range(8):
        b.append(UInt8((v >> Int64(8 * k)) & 0xFF))


def _put_varint(mut b: List[UInt8], v: Int):
    var x = UInt64(v)
    while x >= 0x80:
        b.append(UInt8(x & 0x7F) | 0x80)
        x >>= 7
    b.append(UInt8(x))


def _bit_width(max_value: Int) -> Int:
    var w = 0
    while (1 << w) <= max_value:
        w += 1
    return w


def _compress(codec: Int, src: List[UInt8]) raises -> List[UInt8]:
    if codec == UNCOMPRESSED:
        return src.copy()
    comptime if NO_CODECS:
        raise Error("this build has no zstd or snappy: pages are saved UNCOMPRESSED")
    else:
        if codec == ZSTD:
            var cap = external_call["ZSTD_compressBound", Int](len(src))
            var out = List[UInt8](length=max(cap, 1), fill=0)
            var n = external_call["ZSTD_compress", Int](out.unsafe_ptr(), cap, src.unsafe_ptr(), len(src), Int32(3))
            if external_call["ZSTD_isError", UInt32](n) != 0:
                raise Error("zstd compression failed")
            out.resize(n, 0)
            return out^
        if codec == SNAPPY:
            var cap = external_call["snappy_max_compressed_length", Int](len(src))
            var out = List[UInt8](length=max(cap, 1), fill=0)
            var n = cap
            if external_call["snappy_compress", Int32](src.unsafe_ptr(), len(src), out.unsafe_ptr(), UnsafePointer(to=n)) != 0:
                raise Error("snappy compression failed")
            out.resize(n, 0)
            return out^
        raise Error("compression codec " + String(codec) + " is not supported (UNCOMPRESSED, SNAPPY, ZSTD)")


def _decompress(codec: Int, src: Int, n: Int, size: Int) raises -> List[UInt8]:
    """`n` compressed bytes at address `src` into `size` bytes. Both sizes come from the file: they are checked
    (and the codec's own stated size compared) before the output is allocated."""
    if n < 0 or size < 0 or size > INT32_MAX:
        raise Error("a Parquet page has an impossible size (corrupt file)")
    if codec == UNCOMPRESSED:
        if n != size:
            raise Error("an uncompressed Parquet page has two different sizes (corrupt file)")
        var out = List[UInt8](unsafe_uninit_length=max(size, 1))
        var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=src)
        for i in range(size):
            out[i] = p[i]
        out.resize(size, 0)
        return out^
    comptime if NO_CODECS:
        if codec == ZSTD or codec == SNAPPY:
            raise Error("this build has no zstd or snappy, so it cannot read this page")
        raise Error("compression codec " + String(codec) + " is not supported (this build reads UNCOMPRESSED only)")
    else:
        if codec == ZSTD:
            var stated = external_call["ZSTD_getFrameContentSize", UInt64](src, n)
            if stated < (UInt64(1) << 62) and stated != UInt64(size):
                raise Error("a zstd page did not decompress to its stated size (corrupt file)")
            var out = List[UInt8](unsafe_uninit_length=max(size, 1))
            var got = external_call["ZSTD_decompress", Int](out.unsafe_ptr(), size, src, n)
            if external_call["ZSTD_isError", UInt32](got) != 0 or got != size:
                raise Error("a zstd page did not decompress to its stated size (corrupt file)")
            out.resize(size, 0)
            return out^
        if codec == SNAPPY:
            var stated = 0
            if external_call["snappy_uncompressed_length", Int32](src, n, UnsafePointer(to=stated)) != 0 or stated != size:
                raise Error("a snappy page did not decompress to its stated size (corrupt file)")
            var out = List[UInt8](unsafe_uninit_length=max(size, 1))
            var got = size
            if external_call["snappy_uncompress", Int32](src, n, out.unsafe_ptr(), UnsafePointer(to=got)) != 0 or got != size:
                raise Error("a snappy page did not decompress to its stated size (corrupt file)")
            out.resize(size, 0)
            return out^
        raise Error("compression codec " + String(codec) + " is not supported yet (UNCOMPRESSED, SNAPPY, ZSTD)")


# ── writing ──────────────────────────────────────────────────────────────────────────────────────────

@fieldwise_init
struct _Chunk(Copyable, Movable):
    var offset: Int              # where the column chunk starts (its dictionary page, else its data page)
    var data_offset: Int
    var dict_offset: Int         # -1 without a dictionary
    var compressed: Int
    var uncompressed: Int
    var values: Int


def write_parquet(path: String, t: SharedTable, metadata: String = "", compression: Int = ZSTD,
                  row_group_rows: Int = DEFAULT_ROW_GROUP, job: JobCtl = JobCtl()) raises:
    """Writes the table to `path`; `metadata` (a JSON string) is stored under the key "omx". `job` (F-175): the background job it
    runs for; a cancel is seen between row groups, and a cancelled write leaves `path` as it was."""
    _write(path, t._table[], metadata, compression, row_group_rows, job)


def write_parquet(path: String, t: Table, metadata: String = "", compression: Int = ZSTD,
                  row_group_rows: Int = DEFAULT_ROW_GROUP, job: JobCtl = JobCtl()) raises:
    _write(path, t, metadata, compression, row_group_rows, job)


def _page_header(mut b: List[UInt8], kind: Int, uncompressed: Int, compressed: Int, values: Int, encoding: Int):
    var w = ThriftWriter()
    w.i32(1, kind)
    w.i32(2, uncompressed)
    w.i32(3, compressed)
    if kind == PAGE_DICTIONARY:
        w.begin_struct(7)
        w.i32(1, values)
        w.i32(2, encoding)
        w.end_struct()
    else:
        w.begin_struct(5)
        w.i32(1, values)
        w.i32(2, encoding)
        w.i32(3, ENC_RLE)          # definition levels
        w.i32(4, ENC_RLE)          # repetition levels (none)
        w.end_struct()
    w.end()
    b.extend(w^.take())


def _check_page(col: Column, uncompressed: Int, compressed: Int) raises:
    """A page header stores both sizes as int32."""
    if uncompressed > INT32_MAX - 4096 or compressed > INT32_MAX - 4096:
        raise Error("column `" + col.name + "`: a page of " + String(uncompressed) + " bytes does not fit a Parquet page" +
                    " (use a smaller row_group_rows)")


def _column_pages(mut file: List[UInt8], col: Column, lo: Int, hi: Int, optional: Bool, codec: Int) raises -> _Chunk:
    """Appends one column chunk (rows lo..hi) to `file`."""
    if hi - lo > INT32_MAX:
        raise Error("column `" + col.name + "`: " + String(hi - lo) + " rows in one row group do not fit a Parquet page (use a smaller row_group_rows)")
    var chunk = _Chunk(len(file), 0, -1, 0, 0, hi - lo)
    var start = len(file)
    if col.kind == TEXT:                                # the dictionary page: every word, PLAIN
        var d = List[UInt8]()
        for word in col.dict[].words:
            var bytes = word.as_bytes()
            _put32(d, len(bytes))
            for c in bytes:
                d.append(c)
        var cd = _compress(codec, d)
        _check_page(col, len(d), len(cd))
        chunk.dict_offset = len(file)
        _page_header(file, PAGE_DICTIONARY, len(d), len(cd), len(col.dict[].words), ENC_PLAIN)
        chunk.uncompressed += len(file) - chunk.dict_offset + len(d)       # header + page, uncompressed
        file.extend(cd^)
    var body = List[UInt8]()
    if optional:                                        # definition levels: the validity bits, bit-packed runs
        var levels = List[UInt8]()
        var n = hi - lo
        var groups = (n + 7) // 8
        _put_varint(levels, (groups << 1) | 1)
        for g in range(groups):
            var byte = UInt8(0)
            for k in range(8):
                var row = lo + 8 * g + k
                if row < hi and not col.is_null(row):
                    byte |= UInt8(1) << UInt8(k)
            levels.append(byte)
        _put32(body, len(levels))
        body.extend(levels^)
    if col.kind == TEXT:                                # dictionary indices: bit width, then bit-packed runs
        var w = max(_bit_width(len(col.dict[].words) - 1), 1)
        body.append(UInt8(w))
        var idx = List[Int64]()
        for row in range(lo, hi):
            if not col.is_null(row):
                idx.append(col.values[row])
        var groups = (len(idx) + 7) // 8
        _put_varint(body, (groups << 1) | 1)
        var acc: UInt64 = 0
        var nbits = 0
        for i in range(groups * 8):
            var v = UInt64(idx[i]) if i < len(idx) else 0
            acc |= v << UInt64(nbits)
            nbits += w
            while nbits >= 8:
                body.append(UInt8(acc & 0xFF))
                acc >>= 8
                nbits -= 8
        if nbits > 0:
            body.append(UInt8(acc & 0xFF))
    elif col.kind == BOOL:                              # PLAIN booleans: one bit each, least significant first
        var acc = 0
        var k = 0
        for row in range(lo, hi):
            if col.is_null(row):
                continue
            if col.values[row] != 0:
                acc |= 1 << k
            k += 1
            if k == 8:
                body.append(UInt8(acc))
                acc = 0
                k = 0
        if k > 0:
            body.append(UInt8(acc))
    elif col.kind == DATE:
        for row in range(lo, hi):
            if not col.is_null(row):
                var v = col.values[row]
                if v < Int64(Int32.MIN) or v > Int64(Int32.MAX):
                    raise Error("column `" + col.name + "`: day " + String(v) + " does not fit a Parquet DATE")
                _put32(body, Int(v))
    else:                                               # INT, DECIMAL, FLOAT: PLAIN int64 (a FLOAT's bits are the DOUBLE's)
        for row in range(lo, hi):
            if not col.is_null(row):
                var v = col.values[row]
                if col.kind == DECIMAL and (v >= Int64(MAX_DECIMAL) or v <= -Int64(MAX_DECIMAL)):
                    raise Error("column `" + col.name + "`: " + String(v) + " does not fit DECIMAL(18, " +
                                String(col.scale) + ") (more than 18 digits)")
                _put64(body, v)
    var cb = _compress(codec, body)
    _check_page(col, len(body), len(cb))
    chunk.data_offset = len(file)
    _page_header(file, PAGE_DATA, len(body), len(cb), hi - lo, ENC_RLE_DICTIONARY if col.kind == TEXT else ENC_PLAIN)
    chunk.uncompressed += len(file) - chunk.data_offset + len(body)
    file.extend(cb^)
    chunk.compressed = len(file) - start
    return chunk^


def _write(path: String, t: Table, metadata: String, requested: Int, row_group_rows: Int, job: JobCtl = JobCtl()) raises:
    t.check()
    if t.columns() == 0:
        raise Error("a table with no columns cannot be written as Parquet")
    if row_group_rows < 8 or row_group_rows % 8 != 0:
        raise Error("row_group_rows must be a positive multiple of 8, got " + String(row_group_rows))
    var optional = List[Bool]()
    for c in range(t.columns()):
        optional.append(t.cols[c].nulls > 0)
    var w = ParquetWriter(path, t, optional, requested)
    try:
        var n = t.rows()
        var lo = 0
        job.stage(STAGE_WRITE)
        job.add_total(n)
        while lo < n or (n == 0 and w.group_count() == 0):
            job.check()                                      # F-175: between row groups
            var hi = min(lo + row_group_rows, n)
            w.write_rows(t, lo, hi)
            job.advance(hi - lo)
            if n == 0:
                break
            lo = hi
        job.commit()                                         # (raises when cancelled: the temporary file goes, the old one stays)
        w.finish(metadata)
    except e:
        w.abort()
        raise e^


struct ParquetWriter(Movable):
    """A Parquet file written one row group at a time (F-147): the row groups go to a temporary file as they are
    given, so the memory is one group, whatever the file's size; `finish` writes the footer and renames the file into
    place (atomic: a crash or an error never loses an existing file; `abort` removes the temporary file).

        var w = ParquetWriter(path, schema_table, optional, compression)
        w.write_group(group_table)      # once per row group: the same columns, TEXT words and all
        w.finish(metadata)

    `optional[c]`: the column is written with definition levels (it may hold a null); a column not marked must hold
    none. The TEXT dictionary of every group is the group table's words (each group page holds all of them)."""
    var path: String
    var tmp: String
    var file: FileHandle
    var offset: Int
    var codec: Int
    var schema: Table                    # the columns (name, kind, scale), no rows
    var optional: List[Bool]
    var groups: List[List[_Chunk]]
    var rows: List[Int]
    var total: Int

    def __init__(out self, path: String, shape: Table, optional: List[Bool], compression: Int) raises:
        self.codec = compression
        comptime if NO_CODECS:
            self.codec = UNCOMPRESSED                  # no zstd / snappy linked: whatever was asked for
        self.path = path
        # The temp name carries the process id, so two processes saving the same path never share a temp file.
        # (F-175) and the address of a fresh allocation and the clock: two saves to one path in one process (a cancelled save that is
        # still unwinding beside the one that replaced it, two jobs at once) must never share a temporary file.
        var uniq = unsafe_alloc[Int64](1)
        self.tmp = path + "." + String(external_call["getpid", Int32]()) + "." + String(Int(uniq)) + "." + String(perf_counter_ns()) + ".tmp"
        uniq.free()
        self.offset = 0
        self.schema = Table()
        for c in range(shape.columns()):
            _ = self.schema.add_column(shape.cols[c].name, shape.cols[c].kind, scale=shape.cols[c].scale)
        self.optional = optional.copy()
        self.groups = List[List[_Chunk]]()
        self.rows = List[Int]()
        self.total = 0
        try:
            self.file = open(self.tmp, "w")
        except:
            raise Error("couldn't write `" + path + "`: no such folder, or no permission to write there")
        var magic: List[UInt8] = [0x50, 0x41, 0x52, 0x31]                 # "PAR1"
        self.file.write_bytes(Span(magic))
        self.offset = 4

    def group_count(self) -> Int:
        return len(self.groups)

    def write_group(mut self, t: Table) raises:
        """Appends all of `t`'s rows as one row group (its columns are the writer's, in order)."""
        self.write_rows(t, 0, t.rows())

    def write_rows(mut self, t: Table, lo: Int, hi: Int) raises:
        """Appends rows lo..hi of `t` as one row group."""
        if t.columns() != self.schema.columns():
            raise Error("a row group has " + String(t.columns()) + " columns, the file " + String(self.schema.columns()))
        var buf = List[UInt8]()
        var chunks = List[_Chunk]()
        for c in range(t.columns()):
            var ch = _column_pages(buf, t.cols[c], lo, hi, self.optional[c], self.codec)
            ch.offset += self.offset                       # `buf` starts where the file stands
            ch.data_offset += self.offset
            if ch.dict_offset >= 0:
                ch.dict_offset += self.offset
            chunks.append(ch^)
        self.file.write_bytes(Span(buf))
        self.offset += len(buf)
        self.groups.append(chunks^)
        self.rows.append(hi - lo)
        self.total += hi - lo

    def abort(mut self):
        try:
            self.file.close()
        except:
            pass
        _ = external_call["unlink", Int32](self.tmp.as_c_string_span().ptr())

    def finish(mut self, metadata: String, extra: List[String] = List[String]()) raises:
        """Writes the footer and moves the file into place. `extra`: other key-value metadata to keep, key then value,
        key then value... (a scan's `extra_metadata()`)."""
        var meta = _footer(self.schema, self.optional, self.total, self.groups, self.rows, metadata, self.codec, extra)
        var tail = List[UInt8]()
        var footer = len(meta)
        tail.extend(meta^)
        _put32(tail, footer)
        tail.extend([UInt8(0x50), UInt8(0x41), UInt8(0x52), UInt8(0x31)])
        self.file.write_bytes(Span(tail))
        self.file.close()
        var dest = self.path
        if external_call["rename", Int32](self.tmp.as_c_string_span().ptr(), dest.as_c_string_span().ptr()) != 0:
            _ = external_call["unlink", Int32](self.tmp.as_c_string_span().ptr())
            raise Error("could not move `" + self.tmp + "` to `" + self.path + "`")


def _footer(t: Table, optional: List[Bool], n: Int, groups: List[List[_Chunk]], rows_of: List[Int], metadata: String,
            codec: Int, extra: List[String]) raises -> List[UInt8]:
    """The file metadata (Thrift) for the row groups written: `t` supplies the columns' names, kinds and scales."""
    var ncols = t.columns()
    var w = ThriftWriter()
    w.i32(1, 1)                                         # version
    w.list_header(2, T_STRUCT, ncols + 1)               # schema: the root, then one element per column
    w.begin_list_element_struct()
    w.binary(4, "schema")
    w.i32(5, ncols)
    w.end_struct()
    for c in range(ncols):
        ref col = t.cols[c]
        w.begin_list_element_struct()
        var pt = PT_BYTE_ARRAY if col.kind == TEXT else PT_BOOLEAN if col.kind == BOOL else PT_INT32 if col.kind == DATE else PT_DOUBLE if col.kind == FLOAT else PT_INT64
        w.i32(1, pt)
        w.i32(3, OPTIONAL if optional[c] else REQUIRED)
        w.binary(4, col.name)
        if col.kind == TEXT:
            w.i32(6, CT_UTF8)
            w.begin_struct(10)
            w.begin_struct(1)                           # LogicalType.STRING
            w.end_struct()
            w.end_struct()
        elif col.kind == DATE:
            w.i32(6, CT_DATE)
            w.begin_struct(10)
            w.begin_struct(6)                           # LogicalType.DATE
            w.end_struct()
            w.end_struct()
        elif col.kind == DECIMAL:
            w.i32(6, CT_DECIMAL)
            w.i32(7, col.scale)
            w.i32(8, 18)
            w.begin_struct(10)
            w.begin_struct(5)                           # LogicalType.DECIMAL
            w.i32(1, col.scale)
            w.i32(2, 18)
            w.end_struct()
            w.end_struct()
        w.end_struct()
    w.i64(3, Int64(n))
    w.list_header(4, T_STRUCT, len(groups))             # row groups
    for g in range(len(groups)):
        var rows = rows_of[g]
        var total = 0
        w.begin_list_element_struct()
        w.list_header(1, T_STRUCT, ncols)
        for c in range(ncols):
            ref ch = groups[g][c]
            ref col = t.cols[c]
            total += ch.uncompressed
            w.begin_list_element_struct()
            w.i64(2, Int64(ch.offset))
            w.begin_struct(3)                           # ColumnMetaData
            var pt = PT_BYTE_ARRAY if col.kind == TEXT else PT_BOOLEAN if col.kind == BOOL else PT_INT32 if col.kind == DATE else PT_DOUBLE if col.kind == FLOAT else PT_INT64
            w.i32(1, pt)
            if col.kind == TEXT:
                w.list_header(2, T_I32, 3)
                w.zigzag(ENC_PLAIN)
                w.zigzag(ENC_RLE)
                w.zigzag(ENC_RLE_DICTIONARY)
            else:
                w.list_header(2, T_I32, 2)
                w.zigzag(ENC_PLAIN)
                w.zigzag(ENC_RLE)
            w.list_header(3, T_BINARY, 1)
            w.raw_binary(col.name)
            w.i32(4, codec)
            w.i64(5, Int64(rows))
            w.i64(6, Int64(ch.uncompressed))
            w.i64(7, Int64(ch.compressed))
            w.i64(9, Int64(ch.data_offset))
            if ch.dict_offset >= 0:
                w.i64(11, Int64(ch.dict_offset))
            w.end_struct()
            w.end_struct()
        w.i64(2, Int64(total))
        w.i64(3, Int64(rows))
        w.end_struct()
    var pairs = len(extra) // 2 + (1 if metadata.byte_length() > 0 else 0)
    if pairs > 0:
        w.list_header(5, T_STRUCT, pairs)
        if metadata.byte_length() > 0:
            w.begin_list_element_struct()
            w.binary(1, "omx")
            w.binary(2, metadata)
            w.end_struct()
        for i in range(len(extra) // 2):
            w.begin_list_element_struct()
            w.binary(1, extra[2 * i])
            w.binary(2, extra[2 * i + 1])
            w.end_struct()
    w.binary(6, "omxcore version 0.6.0")
    w.end()
    return w^.take()


# ── reading ──────────────────────────────────────────────────────────────────────────────────────────

struct ParquetResult(Movable):
    var table: Table
    var metadata: String             # the "omx" key-value metadata, "" when there is none

    def __init__(out self, var table: Table, var metadata: String):
        self.table = table^
        self.metadata = metadata^

    def take_table(mut self) -> Table:
        """The table, moved out (the result keeps an empty one), so `.share()` works while `metadata` is still read."""
        var t = Table.empty()
        swap(t, self.table)
        return t^


@fieldwise_init
struct _Leaf(Copyable, Movable):
    var name: String
    var ptype: Int
    var type_length: Int
    var repetition: Int
    var converted: Int               # -1 when absent
    var scale: Int
    var precision: Int
    var logical: Int                 # the LogicalType union's field id, -1 when absent
    var ts_unit: Int                 # TIMESTAMP: 1 millis, 2 micros, 3 nanos
    var children: Int
    var unsigned: Bool               # an INTEGER logical type with isSigned false


@fieldwise_init
struct _ChunkMeta(Copyable, Movable):
    var ptype: Int
    var codec: Int
    var values: Int
    var data_off: Int
    var dict_off: Int
    var size: Int                    # total_compressed_size: the bytes of the chunk's pages (headers included)
    var group_rows: Int              # its row group's num_rows, -1 when the footer has none
    var nulls: Int                   # its Statistics' null_count, -1 when absent
    var stat_n: Int                  # the byte length of its Statistics' minimum and maximum (0: it states none)
    var min_le: Int64                # the minimum and maximum read little-endian (INT32 / INT64) and big-endian (a short decimal)
    var max_le: Int64
    var min_be: Int64
    var max_be: Int64


struct _Bytes:
    """A cursor over `n` bytes at `base`."""
    var base: Int
    var n: Int
    var pos: Int

    def __init__(out self, base: Int, n: Int):
        self.base = base
        self.n = n
        self.pos = 0

    def at(self, i: Int) -> Int:
        return Int(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=self.base)[i])

    def need(self, k: Int) raises:
        if k < 0 or k > self.n - self.pos:
            raise Error("a Parquet page ends early (corrupt file)")

    def byte(mut self) raises -> Int:
        self.need(1)
        self.pos += 1
        return self.at(self.pos - 1)

    def le(mut self, k: Int) raises -> Int64:
        self.need(k)
        var v: UInt64 = 0
        for i in range(k):
            v |= UInt64(self.at(self.pos + i)) << UInt64(8 * i)
        self.pos += k
        if k < 8 and (v >> UInt64(8 * k - 1)) & 1 == 1:          # sign-extend
            v |= ~UInt64(0) << UInt64(8 * k)
        return Int64(v)

    def varint(mut self) raises -> UInt64:
        var v: UInt64 = 0
        var shift = 0
        while True:
            var b = self.byte()
            v |= UInt64(b & 0x7F) << UInt64(shift)
            if b < 0x80:
                return v
            shift += 7
            if shift > 63:
                raise Error("a Parquet page has a bad varint (corrupt file)")

    def zigzag(mut self) raises -> Int64:
        var u = self.varint()
        return Int64(u >> 1) ^ -Int64(u & 1)

    def small(mut self) raises -> Int:
        """A varint count or header: below 2^62, so doubling or multiplying by 8 cannot overflow."""
        var u = self.varint()
        if u >= (UInt64(1) << 62):
            raise Error("a Parquet page has an impossible count (corrupt file)")
        return Int(u)


def _unpack(mut c: _Bytes, width: Int, count: Int, mut out: List[Int64]) raises:
    """`count` values of `width` bits, least significant first (Parquet's bit packing)."""
    if width < 0 or width > 64:
        raise Error("a Parquet page has a bit width of " + String(width) + " (corrupt file)")
    if count < 0:
        raise Error("a Parquet page has a negative count (corrupt file)")
    if width == 0:                                      # the callers cap `count` at the values still needed
        for _ in range(count):
            out.append(0)
        return
    if count > ((c.n - c.pos) * 8) // width:            # checked before count * width, which could overflow
        raise Error("a Parquet page ends early (corrupt file)")
    var nbytes = (count * width + 7) // 8
    var acc: UInt64 = 0
    var have = 0
    var p = c.pos
    var mask = (UInt64(1) << UInt64(width)) - 1 if width < 64 else ~UInt64(0)
    for _ in range(count):
        while have < width:
            acc |= UInt64(c.at(p)) << UInt64(have)
            p += 1
            have += 8
            if have > 56 and have < width:      # widths past 56 bits: take the rest in a second word
                break
        if have >= width:
            out.append(Int64(acc & mask))
            acc = acc >> UInt64(width) if width < 64 else 0
            have -= width
        else:
            var lo = acc
            var got = have
            acc = 0
            have = 0
            while got + have < width:
                acc |= UInt64(c.at(p)) << UInt64(have)
                p += 1
                have += 8
            var need = width - got
            out.append(Int64(lo | ((acc & ((UInt64(1) << UInt64(need)) - 1)) << UInt64(got))))
            acc >>= UInt64(need)
            have -= need
    c.pos += nbytes


def _rle_hybrid(mut c: _Bytes, width: Int, count: Int, mut out: List[Int64]) raises:
    """The RLE / bit-packing hybrid: runs of one repeated value and groups of 8 bit-packed values. Exactly `count`
    values are produced: a run longer than what is still needed is cut, never built."""
    if width < 0 or width > 64:
        raise Error("a Parquet page has a bit width of " + String(width) + " (corrupt file)")
    var start = len(out)
    while len(out) - start < count:
        var h = c.small()
        var left = count - (len(out) - start)
        if h & 1 == 1:
            var groups = h >> 1
            _unpack(c, width, left if groups >= (left + 7) // 8 else groups * 8, out)
        else:
            var v = c.le((width + 7) // 8) if width > 0 else 0
            if width > 0 and width < 64:
                v &= Int64((UInt64(1) << UInt64(width)) - 1)
            for _ in range(min(h >> 1, left)):
                out.append(v)


def _wrap32(v: Int64) -> Int64:
    """An INT32 column's value: the low 32 bits, sign-extended (its deltas wrap modulo 2^32)."""
    var low = v & 0xFFFFFFFF
    return low - (Int64(1) << 32) if low >= (Int64(1) << 31) else low


def _delta_binary_packed(mut c: _Bytes, int32: Bool, limit: Int, mut out: List[Int64]) raises -> Int:
    """DELTA_BINARY_PACKED integers (at most `limit`); returns how many were read. INT32 arithmetic wraps modulo 2^32, as the writer's."""
    var block = c.small()
    var minis = c.small()
    var total = c.small()
    var prev = c.zigzag()
    if total > limit:
        raise Error("a Parquet page holds more delta values than its count says (corrupt file)")
    if total == 0:
        return 0
    if block <= 0 or minis <= 0 or block % minis != 0:
        raise Error("a Parquet delta block has a bad shape (corrupt file)")
    out.append(prev)
    var got = 1
    var per_mini = block // minis
    while got < total:
        var min_delta = c.zigzag()
        var widths = List[Int]()
        for _ in range(minis):
            widths.append(c.byte())
        for m in range(minis):
            if got >= total:
                break                                   # trailing miniblocks of the last block may be absent
            var deltas = List[Int64]()
            _unpack(c, widths[m], min(per_mini, total - got), deltas)
            for d in deltas:
                prev = prev + min_delta + d
                if int32:
                    prev = _wrap32(prev)
                out.append(prev)
                got += 1
    return total


def _ptype_name(p: Int) -> String:
    var names: List[String] = ["BOOLEAN", "INT32", "INT64", "INT96", "FLOAT", "DOUBLE", "BYTE_ARRAY", "FIXED_LEN_BYTE_ARRAY"]
    return names[p] if 0 <= p < len(names) else "physical type " + String(p)


def _is_unsigned(leaf: _Leaf) -> Bool:
    return leaf.unsigned or (CT_UINT_8 <= leaf.converted <= CT_UINT_64)


def _is_timestamp(leaf: _Leaf) -> Bool:
    return leaf.ptype == PT_INT64 and (leaf.logical == 8 or leaf.converted == CT_TIMESTAMP_MILLIS or
                                       leaf.converted == CT_TIMESTAMP_MICROS)


def _kind_of(leaf: _Leaf) raises -> Tuple[ColumnKind, Int]:
    """The omxcore kind (and scale) a Parquet leaf column becomes."""
    if leaf.ptype < 0 or leaf.ptype > PT_FIXED:
        raise Error("column `" + leaf.name + "` has an unknown physical type (" + String(leaf.ptype) + ", corrupt file)")
    if leaf.logical == 7 or leaf.converted == CT_TIME_MILLIS or leaf.converted == CT_TIME_MICROS:
        raise Error("column `" + leaf.name + "` is a TIME: omxcore has no kind for it yet")
    var decimal = leaf.converted == CT_DECIMAL or leaf.logical == 5
    if decimal:
        if leaf.precision > 18:
            raise Error("column `" + leaf.name + "`: DECIMAL(" + String(leaf.precision) + ", " + String(leaf.scale) +
                        ") has more digits than an int64 holds (18)")
        if leaf.ptype == PT_INT32 or leaf.ptype == PT_INT64 or leaf.ptype == PT_FIXED:
            return (DECIMAL, leaf.scale)
        if leaf.ptype == PT_BYTE_ARRAY:
            raise Error("column `" + leaf.name + "` is a DECIMAL stored as BYTE_ARRAY: omxcore reads DECIMAL on INT32, INT64 and FIXED_LEN_BYTE_ARRAY only")
    if leaf.ptype == PT_BOOLEAN:
        return (BOOL, 0)
    if leaf.ptype == PT_BYTE_ARRAY:
        if leaf.logical == 1 or leaf.logical == 4 or leaf.logical == 12 or leaf.converted == CT_UTF8 or \
           leaf.converted == CT_ENUM or leaf.converted == CT_JSON:
            return (TEXT, 0)
        raise Error("column `" + leaf.name + "` is BINARY (BYTE_ARRAY without a string type): omxcore has no kind for it yet")
    if leaf.converted == CT_DATE or leaf.logical == 6:
        return (DATE, 0)
    if _is_timestamp(leaf):
        return (DATE, 0)                                # a timestamp becomes its UTC day (the CSV reader's rule)
    if leaf.ptype == PT_INT32 or leaf.ptype == PT_INT64:
        return (INT, 0)
    if leaf.ptype == PT_FLOAT:
        return (FLOAT, F32_COLUMN)                      # widened to float64 (exact); the scale remembers the 32 bits; saved as DOUBLE
    if leaf.ptype == PT_DOUBLE:
        return (FLOAT, 0)
    raise Error("column `" + leaf.name + "` is " + _ptype_name(leaf.ptype) + ": omxcore has no kind for it yet")


def _to_stored(leaf: _Leaf, raw: Int64) raises -> Int64:
    """A plain value of the leaf's physical type as omxcore stores it (timestamps become days, floored; unsigned
    integers keep their value: UINT_32 is zero-extended, UINT_64 past the int64 range is refused)."""
    if _is_timestamp(leaf):
        var per_day = Int64(86_400_000) if (leaf.ts_unit == 1 or leaf.converted == CT_TIMESTAMP_MILLIS) else \
                      Int64(86_400_000_000_000) if leaf.ts_unit == 3 else Int64(86_400_000_000)
        return raw // per_day
    if _is_unsigned(leaf) and leaf.logical != 5 and leaf.converted != CT_DECIMAL:
        if leaf.ptype == PT_INT32:
            return raw & 0xFFFFFFFF
        if raw < 0:
            raise Error("column `" + leaf.name + "` is UINT_64 and holds " + String(UInt64(raw)) +
                        ", past the int64 range omxcore stores")
    return raw


def read_parquet(path: String, job: JobCtl = JobCtl()) raises -> ParquetResult:
    """Reads a Parquet file (written by omxcore, pyarrow, DuckDB, ...) into a Table, plus its "omx" metadata. `job` (F-175): the
    background job it runs for; a cancel is seen between row groups."""
    var file = _open_file(path)
    var n = _file_size(file, path)
    if n < 12:
        raise Error("`" + path + "` is not a Parquet file (no PAR1 magic)")
    var map = _Mapped(file._get_raw_fd(), 0, n, path)
    var data = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=map.data)
    if data[0] != 0x50 or data[1] != 0x41 or data[2] != 0x52 or data[3] != 0x31 or \
       data[n - 4] != 0x50 or data[n - 3] != 0x41 or data[n - 2] != 0x52 or data[n - 1] != 0x31:
        raise Error("`" + path + "` is not a Parquet file (no PAR1 magic)")
    var flen = Int(data[n - 8]) | (Int(data[n - 7]) << 8) | (Int(data[n - 6]) << 16) | (Int(data[n - 5]) << 24)
    if flen <= 0 or flen > n - 12:
        raise Error("`" + path + "`: the Parquet footer length is wrong (corrupt file)")
    var base = map.data
    var r = ThriftReader(base + n - 8 - flen, flen)

    var footer = _read_footer(r)
    ref leaves = footer.leaves
    ref groups = footer.groups

    var cols = List[_Leaf]()
    var t = _schema_table(path, leaves, cols)
    var ids = List[Int]()
    var bases = List[Int]()
    var ns = List[Int]()
    var widx = List[_WordIndex]()
    var widxs = List[Int]()
    var total = 0
    for c in range(len(cols)):
        ids.append(c)
        bases.append(base)
        ns.append(n)
        widx.append(_WordIndex())
        widx[c].col = Int(t.cols.unsafe_ptr() + c)
    for c in range(len(cols)):
        widxs.append(Int(widx.unsafe_ptr() + c))
    for g in range(len(groups)):
        if len(cols) > 0 and len(groups[g]) == len(cols):
            var n_g = groups[g][0].values
            _check_claim("`" + path + "` row group " + String(g), n_g, len(cols))
            total += n_g                                        # each part checked, so the sum cannot wrap
            _check_claim("`" + path + "`", total, len(cols))
    job.stage(STAGE_READ)
    job.add_total(total)
    for c in range(len(cols)):                                  # size every column once; the pages write every row
        t.cols[c].values = List[Int64](unsafe_uninit_length=total)
    var row0 = 0
    for g in range(len(groups)):
        job.check()                                             # F-175: between row groups
        if len(groups[g]) != len(cols):
            raise Error("`" + path + "`: row group " + String(g) + " has " + String(len(groups[g])) +
                        " columns, the schema " + String(len(cols)))
        _decode_group(bases, ns, groups[g], ids, ids, widxs, True, cols, row0, t)
        if len(cols) > 0:
            row0 += groups[g][0].values
            job.advance(groups[g][0].values)
    for c in range(len(cols)):                                  # (a presized column holds exactly the rows read)
        if len(t.cols[c].values) > row0:
            t.cols[c].values.shrink(row0)
        if len(t.cols[c].validity) > (row0 + 7) // 8:
            t.cols[c].validity.shrink((row0 + 7) // 8)
    t.check()
    map.release()
    _ = widx^                                                   # the word indexes are read by address above
    return ParquetResult(t^, footer.omx.copy())


def _schema_table(path: String, leaves: List[_Leaf], mut cols: List[_Leaf]) raises -> Table:
    """An empty table with the file's columns (a flat schema: a root, then one element per column); `cols` gets
    their leaves."""
    if len(leaves) == 0:
        raise Error("`" + path + "`: the Parquet schema is empty")
    var t = Table.empty()
    for i in range(1, len(leaves)):
        if leaves[i].children > 0 or leaves[i].repetition == REPEATED:
            raise Error("column `" + leaves[i].name + "` is nested (a list or a struct): not supported yet")
        var k = _kind_of(leaves[i])
        _ = t.add_column(leaves[i].name, k[0], scale=k[1])
        cols.append(leaves[i].copy())
    return t^


@fieldwise_init
struct _Footer(Movable):
    var leaves: List[_Leaf]
    var groups: List[List[_ChunkMeta]]
    var omx: String
    var extra: List[String]          # the other key-value metadata, key then value (not "omx", not Arrow's schema)


def _read_footer(mut r: ThriftReader) raises -> _Footer:
    """The schema, the row groups' chunk metadata and the "omx" metadata from a Parquet footer."""
    var leaves = List[_Leaf]()
    var groups = List[List[_ChunkMeta]]()
    var omx = String("")
    var extra = List[String]()
    while True:
        var f = r.field()
        if f[1] == T_STOP:
            break
        if f[0] == 2 and f[1] == T_LIST:                # schema
            var h = r.list_header()
            for _ in range(h[1]):
                leaves.append(_read_schema_element(r))
        elif f[0] == 4 and f[1] == T_LIST:              # row groups
            var h = r.list_header()
            for _ in range(h[1]):
                groups.append(_read_row_group(r))
        elif f[0] == 5 and f[1] == T_LIST:              # key-value metadata
            var h = r.list_header()
            for _ in range(h[1]):
                r.enter()
                var key = String("")
                var value = String("")
                while True:
                    var g = r.field()
                    if g[1] == T_STOP:
                        break
                    if g[0] == 1 and g[1] == T_BINARY:
                        key = r.binary()
                    elif g[0] == 2 and g[1] == T_BINARY:
                        value = r.binary()
                    else:
                        r.skip(g[1])
                r.leave()
                if key == "omx":
                    omx = value
                elif key != "ARROW:schema":              # it describes the source's types, which a saved file does not keep
                    extra.append(key)
                    extra.append(value)
        else:
            r.skip(f[1])

    return _Footer(leaves^, groups^, omx^, extra^)


def _read_schema_element(mut r: ThriftReader) raises -> _Leaf:
    var leaf = _Leaf("", -1, 0, REQUIRED, -1, 0, 0, -1, 0, 0, False)
    r.enter()
    while True:
        var f = r.field()
        if f[1] == T_STOP:
            break
        if f[0] == 1 and f[1] == T_I32:
            leaf.ptype = Int(r.zigzag())
        elif f[0] == 2 and f[1] == T_I32:
            leaf.type_length = Int(r.zigzag())
        elif f[0] == 3 and f[1] == T_I32:
            leaf.repetition = Int(r.zigzag())
        elif f[0] == 4 and f[1] == T_BINARY:
            leaf.name = r.binary()
        elif f[0] == 5 and f[1] == T_I32:
            leaf.children = Int(r.zigzag())
        elif f[0] == 6 and f[1] == T_I32:
            leaf.converted = Int(r.zigzag())
        elif f[0] == 7 and f[1] == T_I32:
            leaf.scale = Int(r.zigzag())
        elif f[0] == 8 and f[1] == T_I32:
            leaf.precision = Int(r.zigzag())
        elif f[0] == 10 and f[1] == T_STRUCT:            # LogicalType: a union, one field set
            r.enter()
            while True:
                var u = r.field()
                if u[1] == T_STOP:
                    break
                leaf.logical = u[0]
                if u[0] == 5 and u[1] == T_STRUCT:       # DECIMAL(scale, precision)
                    r.enter()
                    while True:
                        var d = r.field()
                        if d[1] == T_STOP:
                            break
                        if d[0] == 1 and d[1] == T_I32:
                            leaf.scale = Int(r.zigzag())
                        elif d[0] == 2 and d[1] == T_I32:
                            leaf.precision = Int(r.zigzag())
                        else:
                            r.skip(d[1])
                    r.leave()
                elif u[0] == 10 and u[1] == T_STRUCT:    # INTEGER(bitWidth, isSigned)
                    r.enter()
                    while True:
                        var d = r.field()
                        if d[1] == T_STOP:
                            break
                        if d[0] == 2 and (d[1] == T_TRUE or d[1] == T_FALSE):
                            leaf.unsigned = d[1] == T_FALSE
                        else:
                            r.skip(d[1])
                    r.leave()
                elif u[0] == 8 and u[1] == T_STRUCT:     # TIMESTAMP(isAdjustedToUTC, unit)
                    r.enter()
                    while True:
                        var d = r.field()
                        if d[1] == T_STOP:
                            break
                        if d[0] == 2 and d[1] == T_STRUCT:   # TimeUnit: MILLIS 1, MICROS 2, NANOS 3
                            r.enter()
                            while True:
                                var tu = r.field()
                                if tu[1] == T_STOP:
                                    break
                                leaf.ts_unit = tu[0]
                                r.skip(tu[1])
                            r.leave()
                        else:
                            r.skip(d[1])
                    r.leave()
                else:
                    r.skip(u[1])
            r.leave()
        else:
            r.skip(f[1])
    r.leave()
    return leaf^


def _read_row_group(mut r: ThriftReader) raises -> List[_ChunkMeta]:
    var chunks = List[_ChunkMeta]()
    var group_rows = -1
    r.enter()
    while True:
        var f = r.field()
        if f[1] == T_STOP:
            break
        if f[0] == 1 and f[1] == T_LIST:
            var h = r.list_header()
            for _ in range(h[1]):
                var meta = _ChunkMeta(-1, 0, 0, -1, -1, 0, -1, -1, 0, 0, 0, 0, 0)
                r.enter()
                while True:
                    var g = r.field()
                    if g[1] == T_STOP:
                        break
                    if g[0] == 1 and g[1] == T_BINARY:
                        raise Error("a Parquet column chunk lives in another file: not supported")
                    if g[0] == 3 and g[1] == T_STRUCT:   # ColumnMetaData
                        r.enter()
                        while True:
                            var m = r.field()
                            if m[1] == T_STOP:
                                break
                            if m[0] == 1 and m[1] == T_I32:
                                meta.ptype = Int(r.zigzag())
                            elif m[0] == 4 and m[1] == T_I32:
                                meta.codec = Int(r.zigzag())
                            elif m[0] == 5 and m[1] == T_I64:
                                meta.values = Int(r.zigzag())
                            elif m[0] == 7 and m[1] == T_I64:
                                meta.size = Int(r.zigzag())
                            elif m[0] == 9 and m[1] == T_I64:
                                meta.data_off = Int(r.zigzag())
                            elif m[0] == 11 and m[1] == T_I64:
                                meta.dict_off = Int(r.zigzag())
                            elif m[0] == 12 and m[1] == T_STRUCT:      # Statistics (F-150: ranges and null counts for the planner)
                                r.enter()
                                var lo5 = (0, Int64(0), Int64(0))
                                var hi5 = (0, Int64(0), Int64(0))
                                while True:
                                    var st = r.field()
                                    if st[1] == T_STOP:
                                        break
                                    if st[0] == 3 and st[1] == T_I64:
                                        meta.nulls = Int(r.zigzag())
                                    elif st[0] == 5 and st[1] == T_BINARY:
                                        hi5 = r.raw_value()
                                    elif st[0] == 6 and st[1] == T_BINARY:
                                        lo5 = r.raw_value()
                                    else:
                                        r.skip(st[1])
                                r.leave()
                                if lo5[0] == hi5[0] and lo5[0] > 0 and lo5[0] <= 8:
                                    meta.stat_n = lo5[0]
                                    meta.min_le = lo5[1]
                                    meta.max_le = hi5[1]
                                    meta.min_be = lo5[2]
                                    meta.max_be = hi5[2]
                            else:
                                r.skip(m[1])
                        r.leave()
                    else:
                        r.skip(g[1])
                r.leave()
                chunks.append(meta^)
        elif f[0] == 3 and f[1] == T_I64:
            group_rows = Int(r.zigzag())
        else:
            r.skip(f[1])
    for i in range(len(chunks)):
        chunks[i].group_rows = group_rows
    r.leave()
    return chunks^


# ── decoding a row group: pages in bulk, in parallel (F-141) ─────────────────────────────────────────
#
# A row group's chunks (one per column read) are decoded in three steps:
#   1. walk: the page headers of every chunk, serially (cheap). A dictionary page is decoded here (its words become
#      the file-wide codes, or are added to the table's own dictionary for `read_parquet`), a data page becomes a
#      `_Page` job that knows its first row. The columns are then sized once, from the rows the pages add up to.
#   2. TEXT pages that must ADD words to a table (PLAIN text under `read_parquet`) are decoded now, in file order,
#      because adding words mutates the table.
#   3. every other page is decoded in parallel straight into its slice of the column: decompress, unpack the
#      definition levels into the page's validity bits, write the values (a PLAIN int64 is a copy, an int32 one
#      widening loop, dictionary indices are gathered into codes in the same loop that unpacks them), spread the
#      values over the null rows. The pages' validity bits are then merged into the column's bitmap in page order.
# Nothing in the parallel step mutates shared state: a page only writes its own rows and its own `_Page` record.

struct _Misses(Copyable, Movable):
    """A text page decoded in parallel against a frozen word index: the words it missed (F-142)."""
    var at: List[Int]                # address, length and hash of each distinct missed word, in page order
    var n: List[Int]
    var h: List[UInt64]
    var code: List[Int]              # their codes, given in file order by the serial phase
    var slots: List[Int32]           # a table over them (one entry per distinct word of the page)
    var canon: List[Int]             # (big columns) per miss: page << 32 | miss number of the first page that missed the word

    def __init__(out self):
        self.at = List[Int]()
        self.n = List[Int]()
        self.h = List[UInt64]()
        self.code = List[Int]()
        self.slots = List[Int32]()
        self.canon = List[Int]()


struct _Slices(Copyable, Movable):
    """A big page decoded in slices by row range (F-145): the value stream, and per slice its first non-null value, its
    row, its offset in the stream and, in a bit-packed run entered mid-way, the groups of 8 still to read."""
    var base: Int                    # the value stream: base address, size, bit width or value width
    var end: Int
    var width: Int
    var v: List[Int]
    var row: List[Int]
    var pos: List[Int]
    var pend: List[Int]
    var skip: List[Int]              # values to drop from the start of the entry point (a cut inside a run)
    var next: Int                    # the slice a block reader decodes next
    var err: List[String]

    def __init__(out self):
        self.base = 0
        self.end = 0
        self.width = 0
        self.v = List[Int]()
        self.row = List[Int]()
        self.pos = List[Int]()
        self.pend = List[Int]()
        self.skip = List[Int]()
        self.next = 0
        self.err = List[String]()


struct _Page(Copyable, Movable):
    var addr: Int                    # the page body (compressed) in the file or its mapping
    var foff: Int                    # and its offset in the file
    var csize: Int
    var usize: Int
    var nvals: Int
    var encoding: Int
    var v2: Bool
    var def_len: Int
    var def_enc: Int
    var v2_compressed: Bool
    var ctx: Int                     # its chunk's `_Ctx`
    var row: Int                     # its first row in the output column
    var is_dict: Bool                # a dictionary page: decoded into its chunk's `_Ctx.dict` in the first parallel round
    var defer: Bool                  # PLAIN text that may add words: misses are recorded, and coded in file order afterwards
    var split: Bool                  # a big page decoded in slices by row range (prepared, then one job per slice)
    var whole: Bool                  # a block reader decodes the whole page at once (an encoding it cannot cut)
    var values: List[Int64]          # ... into this, from which its blocks are copied
    var present: Int                 # non-null values (set when the page is opened)
    var nulls: Int
    var bits: List[UInt8]            # validity bits of its rows (an OPTIONAL column with nulls on this page)
    var err: String
    var keep: List[UInt8]            # the decompressed page, kept while its missed words / slices are read
    var miss: _Misses
    var sl: _Slices

    def __init__(out self, addr: Int, csize: Int, usize: Int, nvals: Int, encoding: Int, v2: Bool, def_len: Int,
                 def_enc: Int, v2_compressed: Bool, ctx: Int, row: Int, is_dict: Bool = False, defer: Bool = False,
                 foff: Int = 0):
        self.addr = addr
        self.foff = foff
        self.csize = csize
        self.usize = usize
        self.nvals = nvals
        self.encoding = encoding
        self.v2 = v2
        self.def_len = def_len
        self.def_enc = def_enc
        self.v2_compressed = v2_compressed
        self.ctx = ctx
        self.row = row
        self.is_dict = is_dict
        self.defer = defer
        self.split = False
        self.whole = False
        self.values = List[Int64]()
        self.present = 0
        self.nulls = 0
        self.bits = List[UInt8]()
        self.err = String()
        self.keep = List[UInt8]()
        self.miss = _Misses()
        self.sl = _Slices()


@fieldwise_init
struct _Ctx(Copyable, Movable):
    var leaf: Int                    # index into the leaves
    var col: Int                     # the output column
    var codec: Int
    var optional: Bool
    var widx: Int                    # address of the TEXT column's `_WordIndex` (0 for other kinds)
    var add: Bool                    # new words are added to the table (else a word that is missing is a corrupt file)
    var dict: List[Int64]            # the chunk's dictionary page as stored values (file-wide codes for TEXT)
    var row0: Int                    # the chunk's first row in the output column
    var dict_seen: Bool              # its dictionary page was read (a second one is a corrupt file)


struct _WordIndex(Movable):
    """A TEXT column's words by their bytes: an open-addressing table over the words of the table's `Column` (found by
    its address), so a value that is already a word is found without making a String. Reads are safe from many
    threads; `insert` runs on the calling thread only."""
    var col: Int                     # address of the table's TEXT Column (its `words` hold the bytes)
    var slots: List[Int32]           # -1 empty, else a code
    var hashes: List[UInt64]         # by code
    var mask: Int

    def __init__(out self):
        self.col = 0
        self.slots = List[Int32](length=1024, fill=-1)
        self.hashes = List[UInt64]()
        self.mask = 1023

    def find(self, p: UnsafePointer[UInt8, MutAnyOrigin], n: Int, h: UInt64) -> Int:
        var words = UnsafePointer[Column, MutAnyOrigin](unsafe_from_address=self.col)[].dict[].words.unsafe_ptr()
        var i = Int(h) & self.mask
        while True:
            var code = Int(self.slots[i])
            if code < 0:
                return -1
            if self.hashes[code] == h and words[code].byte_length() == n:
                var q = words[code].unsafe_ptr()    # lifetime: ok (`words` points into the column, which the table owns)
                var same = True
                for k in range(n):
                    if q[k] != p[k]:
                        same = False
                        break
                if same:
                    return code
            i = (i + 1) & self.mask

    def insert(mut self, h: UInt64, code: Int):
        if code >= len(self.hashes):
            self.hashes.append(h)
        if (len(self.hashes) + 1) * 2 > len(self.slots):         # load factor 1/2: double and re-place every code
            var size = len(self.slots) * 2
            self.slots = List[Int32](length=size, fill=-1)
            self.mask = size - 1
            for c in range(len(self.hashes)):
                var j = Int(self.hashes[c]) & self.mask
                while self.slots[j] >= 0:
                    j = (j + 1) & self.mask
                self.slots[j] = Int32(c)
            return
        var j = Int(h) & self.mask
        while self.slots[j] >= 0:
            j = (j + 1) & self.mask
        self.slots[j] = Int32(code)


def _hash_bytes(p: UnsafePointer[UInt8, MutAnyOrigin], n: Int) -> UInt64:
    var h = UInt64(0x9E3779B97F4A7C15) ^ UInt64(n)
    var i = 0
    while i + 8 <= n:
        h = (h ^ (p + i).bitcast[UInt64]().load[alignment=1]()) * UInt64(0xFF51AFD7ED558CCD)
        h ^= h >> 32
        i += 8
    var tail: UInt64 = 0
    var k = 0
    while i < n:
        tail |= UInt64(p[i]) << UInt64(8 * k)
        i += 1
        k += 1
    h = (h ^ tail) * UInt64(0xC4CEB9FE1A85EC53)
    h ^= h >> 29
    return h


def _get_bit(p: UnsafePointer[UInt8, MutAnyOrigin], i: Int) -> Bool:
    return (p[i >> 3] >> UInt8(i & 7)) & 1 == 1


def _assign_bits(dst: UnsafePointer[UInt8, MutAnyOrigin], pos: Int, src: UnsafePointer[UInt8, MutAnyOrigin], nbits: Int):
    """dst bits [pos, pos + nbits) = src bits [0, nbits) (least significant bit first, both)."""
    var i = 0
    while i < nbits and (pos + i) & 7 != 0:
        var m = UInt8(1) << UInt8((pos + i) & 7)
        if _get_bit(src, i):
            dst[(pos + i) >> 3] |= m
        else:
            dst[(pos + i) >> 3] &= ~m
        i += 1
    while i + 8 <= nbits:
        var sb = i >> 3
        var sh = i & 7
        var v = src[sb]
        if sh != 0:
            v = UInt8(((UInt16(src[sb]) | (UInt16(src[sb + 1]) << 8)) >> UInt16(sh)) & 0xFF)
        dst[(pos + i) >> 3] = v
        i += 8
    while i < nbits:
        var m = UInt8(1) << UInt8((pos + i) & 7)
        if _get_bit(src, i):
            dst[(pos + i) >> 3] |= m
        else:
            dst[(pos + i) >> 3] &= ~m
        i += 1


def _copy_bits(dst: UnsafePointer[UInt8, MutAnyOrigin], dpos: Int, src: UnsafePointer[UInt8, MutAnyOrigin], spos: Int, n: Int):
    """dst bits [dpos, dpos + n) = src bits [spos, spos + n)."""
    var i = 0
    while i < n and (spos + i) & 7 != 0:
        var m = UInt8(1) << UInt8((dpos + i) & 7)
        if _get_bit(src, spos + i):
            dst[(dpos + i) >> 3] |= m
        else:
            dst[(dpos + i) >> 3] &= ~m
        i += 1
    if i < n:
        _assign_bits(dst, dpos + i, src + ((spos + i) >> 3), n - i)


def _set_ones(dst: UnsafePointer[UInt8, MutAnyOrigin], pos: Int, count: Int):
    var i = 0
    while i < count and (pos + i) & 7 != 0:
        dst[(pos + i) >> 3] |= UInt8(1) << UInt8((pos + i) & 7)
        i += 1
    while i + 8 <= count:
        dst[(pos + i) >> 3] = 0xFF
        i += 8
    while i < count:
        dst[(pos + i) >> 3] |= UInt8(1) << UInt8((pos + i) & 7)
        i += 1


def _count_bits(p: UnsafePointer[UInt8, MutAnyOrigin], nbits: Int) -> Int:
    var n = 0
    var full = nbits >> 3
    for i in range(full):
        n += Int(pop_count(p[i]))
    if nbits & 7 != 0:
        n += Int(pop_count(p[full] & ((UInt8(1) << UInt8(nbits & 7)) - 1)))
    return n


def _levels_to_bits(mut c: _Bytes, count: Int, bits: UnsafePointer[UInt8, MutAnyOrigin]) raises:
    """`count` definition levels (RLE / bit-packed hybrid, width 1) into `bits`, already zero: bit i set = value i is
    present. A bit-packed run of 1-bit values is a run of bitmap bits; a repeated 1 sets a range."""
    var pos = 0
    while pos < count:
        var h = c.small()
        var left = count - pos
        if h & 1 == 1:
            var groups = h >> 1
            var take = left if groups >= (left + 7) // 8 else groups * 8
            var nbytes = (take + 7) // 8
            c.need(nbytes)
            _assign_bits(bits, pos, UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos), take)
            c.pos += nbytes
            pos += take
        else:
            var v = c.le(1) & 1
            var run = min(h >> 1, left)
            if v == 1:
                _set_ones(bits, pos, run)
            pos += run


def _page_bytes(codec: Int, src: Int, n: Int, size: Int, mut owner: List[UInt8]) raises -> Int:
    """The address of `size` uncompressed bytes: the page itself when it is not compressed, else a decompressed copy
    held in `owner`."""
    if codec == UNCOMPRESSED:
        if n != size or n < 0:
            raise Error("an uncompressed Parquet page has two different sizes (corrupt file)")
        return src
    owner = _decompress(codec, src, n, size)
    return Int(owner.unsafe_ptr())


def _identity(leaf: _Leaf) -> Bool:
    """A plain integer of this leaf is stored as it is (no timestamp-to-day, no unsigned handling)."""
    return not _is_timestamp(leaf) and not (_is_unsigned(leaf) and leaf.logical != 5 and leaf.converted != CT_DECIMAL)


def _note_miss(mut page: _Page, wp: UnsafePointer[UInt8, MutAnyOrigin], n: Int, h: UInt64) -> Int:
    """A word the frozen index lacks, seen by a page decoded in parallel: one entry per distinct word of the page, in
    first-seen order. Returns the placeholder -1 - (its miss number) for the value."""
    if len(page.miss.slots) == 0:
        page.miss.slots = List[Int32](length=1024, fill=-1)
    var mask = len(page.miss.slots) - 1
    var j = Int(h) & mask
    while True:
        var m = Int(page.miss.slots[j])
        if m < 0:
            break
        if page.miss.h[m] == h and page.miss.n[m] == n:
            var q = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=page.miss.at[m])
            var same = True
            for k in range(n):
                if q[k] != wp[k]:
                    same = False
                    break
            if same:
                return -1 - m
        j = (j + 1) & mask
    var m = len(page.miss.at)
    page.miss.at.append(Int(wp))
    page.miss.n.append(n)
    page.miss.h.append(h)
    if (m + 1) * 2 > len(page.miss.slots):               # grow at half full and re-place every miss
        var size = len(page.miss.slots) * 2
        page.miss.slots = List[Int32](length=size, fill=-1)
        mask = size - 1
        for i in range(m + 1):
            var k = Int(page.miss.h[i]) & mask
            while page.miss.slots[k] >= 0:
                k = (k + 1) & mask
            page.miss.slots[k] = Int32(i)
    else:
        page.miss.slots[j] = Int32(m)
    return -1 - m


def _plain_into(mut c: _Bytes, leaf: _Leaf, ctx: _Ctx, count: Int, dout: UnsafePointer[Int64, MutAnyOrigin], tbl: Int,
                store: Bool, pg: Int = 0) raises:
    """`count` PLAIN values to dout[0..count). `store` False (a dictionary build): TEXT words are added, nothing written."""
    if count < 0:
        raise Error("column `" + leaf.name + "`: a page has a negative value count (corrupt file)")
    if leaf.ptype == PT_BOOLEAN:
        c.need((count + 7) // 8)
        var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos)
        for i in range(count):
            dout[i] = Int64((p[i >> 3] >> UInt8(i & 7)) & 1)
        c.pos += (count + 7) // 8
    elif leaf.ptype == PT_INT32:
        if count > (c.n - c.pos) // 4:
            raise Error("a Parquet page ends early (corrupt file)")
        var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos).bitcast[Int32]()
        if _identity(leaf):
            var i = 0
            while i + 4 <= count:
                dout.store[width=4, alignment=1](i, (p + i).load[width=4, alignment=1]().cast[DType.int64]())
                i += 4
            while i < count:
                dout[i] = Int64((p + i).load[alignment=1]())
                i += 1
        else:
            for i in range(count):
                dout[i] = _to_stored(leaf, Int64((p + i).load[alignment=1]()))
        c.pos += count * 4
    elif leaf.ptype == PT_INT64:
        if count > (c.n - c.pos) // 8:
            raise Error("a Parquet page ends early (corrupt file)")
        var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos).bitcast[Int64]()
        if _identity(leaf):
            var i = 0
            while i + 4 <= count:
                dout.store[width=4, alignment=1](i, (p + i).load[width=4, alignment=1]())
                i += 4
            while i < count:
                dout[i] = (p + i).load[alignment=1]()
                i += 1
        else:
            for i in range(count):
                dout[i] = _to_stored(leaf, (p + i).load[alignment=1]())
        c.pos += count * 8
    elif leaf.ptype == PT_DOUBLE:                         # float64 bits, stored as they are
        if count > (c.n - c.pos) // 8:
            raise Error("a Parquet page ends early (corrupt file)")
        var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos).bitcast[Int64]()
        var i = 0
        while i + 4 <= count:
            dout.store[width=4, alignment=1](i, (p + i).load[width=4, alignment=1]())
            i += 4
        while i < count:
            dout[i] = (p + i).load[alignment=1]()
            i += 1
        c.pos += count * 8
    elif leaf.ptype == PT_FLOAT:                          # widened to float64 (exact), stored as its bits
        if count > (c.n - c.pos) // 4:
            raise Error("a Parquet page ends early (corrupt file)")
        var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos).bitcast[Float32]()
        var q = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=Int(dout))
        for i in range(count):
            q[i] = Float64((p + i).load[alignment=1]())
        c.pos += count * 4
    elif leaf.ptype == PT_BYTE_ARRAY:
        for i in range(count):
            var n = Int(c.le(4)) & 0xFFFFFFFF
            c.need(n)
            if not _valid_utf8(c.base + c.pos, n):
                raise Error("column `" + leaf.name + "`: a text value is not valid UTF-8")
            var wp = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos)
            c.pos += n
            var h = _hash_bytes(wp, n)
            var wi = UnsafePointer[_WordIndex, MutAnyOrigin](unsafe_from_address=ctx.widx)
            var code = wi[].find(wp, n, h)
            if code < 0:
                if not ctx.add:
                    raise Error("column `" + leaf.name + "`: a word is missing from the file's dictionary (corrupt file)")
                if pg != 0:                              # a page decoded in parallel: the index is frozen, note the miss;
                    ref page = UnsafePointer[_Page, MutAnyOrigin](unsafe_from_address=pg)[]
                    code = _note_miss(page, wp, n, h)    # its code comes later, in file order (-1 - miss number now)
                else:
                    var s = String(unsafe_from_utf8=Span[UInt8, MutAnyOrigin](unsafe_ptr=wp, length=n))
                    code = UnsafePointer[Table, MutAnyOrigin](unsafe_from_address=tbl)[].add_word(ctx.col, s)
                    wi[].insert(h, code)
            if store:
                dout[i] = Int64(code)
    elif leaf.ptype == PT_FIXED:                        # a big-endian two's-complement decimal
        var w = leaf.type_length
        if w < 1 or w > 16:
            raise Error("column `" + leaf.name + "` has a fixed length of " + String(w) + " bytes (corrupt file)")
        if count > (c.n - c.pos) // w:
            raise Error("a Parquet page ends early (corrupt file)")
        if w >= 8:                                       # only the last 8 bytes survive an int64: one byte swap each
            var q = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos + (w - 8))
            for i in range(count):
                dout[i] = byte_swap((q + i * w).bitcast[Int64]().load[alignment=1]())
            c.pos += count * w
        else:
            for i in range(count):
                var v: Int64 = -1 if c.at(c.pos) >= 0x80 else 0
                for k in range(w):
                    v = (v << 8) | Int64(c.at(c.pos + k))
                c.pos += w
                dout[i] = v
    else:
        raise Error("column `" + leaf.name + "`: its physical type cannot be read yet")


def _dict_into(mut c: _Bytes, leaf: _Leaf, dict: List[Int64], count: Int, dout: UnsafePointer[Int64, MutAnyOrigin]) raises:
    """`count` dictionary indices (a bit width, then the RLE / bit-packed hybrid) gathered through `dict` into dout."""
    var w = c.byte()
    if w > 64:
        raise Error("a Parquet page has a bit width of " + String(w) + " (corrupt file)")
    _dict_runs(c, leaf, w, dict, count, dout, 0)


def _bitpacked_run(mut c: _Bytes, leaf: _Leaf, w: Int, take: Int, d: UnsafePointer[Int64, MutAnyOrigin], dn: Int,
                   dout: UnsafePointer[Int64, MutAnyOrigin]) raises:
    """`take` bit-packed dictionary indices (width w) gathered through the dictionary d into dout."""
    if w == 0:
        if dn < 1:
            raise Error("column `" + leaf.name + "`: a dictionary index is out of range (corrupt file)")
        for i in range(take):
            dout[i] = d[0]
    elif w <= 32:
        if take > ((c.n - c.pos) * 8) // w:
            raise Error("a Parquet page ends early (corrupt file)")
        var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos)
        var mask = (UInt64(1) << UInt64(w)) - 1
        var acc: UInt64 = 0
        var have = 0
        var at = 0
        for i in range(take):
            while have < w:
                acc |= UInt64(p[at]) << UInt64(have)
                at += 1
                have += 8
            var ix = Int(acc & mask)
            acc >>= UInt64(w)
            have -= w
            if ix >= dn:
                raise Error("column `" + leaf.name + "`: a dictionary index is out of range (corrupt file)")
            dout[i] = d[ix]
        c.pos += (take * w + 7) // 8
    else:
        var idx = List[Int64]()
        _unpack(c, w, take, idx)
        for i in range(take):
            if idx[i] < 0 or Int(idx[i]) >= dn:
                raise Error("column `" + leaf.name + "`: a dictionary index is out of range (corrupt file)")
            dout[i] = d[Int(idx[i])]


def _dict_runs(mut c: _Bytes, leaf: _Leaf, w: Int, dict: List[Int64], count: Int, dout: UnsafePointer[Int64, MutAnyOrigin],
               pending: Int, skip: Int = 0) raises:
    """`count` values of a dictionary-index page from the cursor, which is at a run header, or (`pending` > 0) inside a
    bit-packed run with `pending` groups of 8 values still to read (a slice that starts mid-run); the first `skip` values
    from there are read and dropped (a cut inside a run)."""
    var d = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(dict.unsafe_ptr()))
    var dn = len(dict)
    var got = -skip                                      # negative while values are still being dropped
    var pend = pending
    while got < count:
        var h = (pend << 1) | 1 if pend > 0 else c.small()
        pend = 0
        var left = count - got
        if h & 1 == 1:
            var groups = h >> 1
            var take = left if groups >= (left + 7) // 8 else groups * 8
            if got < 0:                                  # decode, then keep only what follows the dropped values
                var tmp = List[Int64](unsafe_uninit_length=take)
                _bitpacked_run(c, leaf, w, take, d, dn, UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(tmp.unsafe_ptr())))
                for i in range(-got, min(take, count - got)):
                    dout[got + i] = tmp[i]
                _ = tmp^
            else:
                _bitpacked_run(c, leaf, w, take, d, dn, dout + got)
            got += take
        else:
            var v = c.le((w + 7) // 8) if w > 0 else 0
            if w > 0 and w < 64:
                v &= Int64((UInt64(1) << UInt64(w)) - 1)
            if v < 0 or Int(v) >= dn:
                raise Error("column `" + leaf.name + "`: a dictionary index is out of range (corrupt file)")
            var run = min(h >> 1, left)
            var x = d[Int(v)]
            for i in range(max(0, -got), run):
                dout[got + i] = x
            got += run


def _decode_dict_page(p: _Page, mut ctx: _Ctx, leaf: _Leaf, tbl: Int) raises:
    """A dictionary page as stored values (file-wide codes for TEXT looked up in the finished dictionary), put into its
    chunk's context. The only job of its round that writes a context; the others take theirs as plain (read) parameters."""
    var owner = List[UInt8]()
    var c = _Bytes(_page_bytes(ctx.codec, p.addr, p.csize, p.usize, owner), p.usize)
    var d = List[Int64](unsafe_uninit_length=p.nvals)
    _plain_into(c, leaf, ctx, p.nvals, UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(d.unsafe_ptr())), tbl, True)
    ctx.dict = d^
    _ = owner^


def _open_page(mut p: _Page, ctx: _Ctx, leaf: _Leaf, mut owner: List[UInt8]) raises -> _Bytes:
    """Decompresses a data page (into `owner` when it is compressed), unpacks its definition levels into `p.bits` and
    returns a cursor on its values; `p.present` is the number of non-null values."""
    var c: _Bytes
    var present = p.nvals
    var nobits = List[UInt8](length=8, fill=0)
    var bits_p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(nobits.unsafe_ptr()))
    if ctx.optional:
        p.bits = List[UInt8](length=(p.nvals + 7) // 8 + 1, fill=0)
        bits_p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(p.bits.unsafe_ptr()))
    if p.v2:
        var lv = p.def_len
        if lv < 0 or lv > p.csize or lv > p.usize:
            raise Error("column `" + leaf.name + "`: the definition levels are longer than their page (corrupt file)")
        if ctx.optional:
            var lc = _Bytes(p.addr, lv)
            _levels_to_bits(lc, p.nvals, bits_p)
        var at = _page_bytes(ctx.codec if p.v2_compressed else UNCOMPRESSED, p.addr + lv, p.csize - lv, p.usize - lv, owner)
        c = _Bytes(at, p.usize - lv)
    else:
        var at = _page_bytes(ctx.codec, p.addr, p.csize, p.usize, owner)
        c = _Bytes(at, p.usize)
        if ctx.optional:
            if p.def_enc == ENC_BIT_PACKED:                      # deprecated: most significant bit first, no length
                var nb = (p.nvals + 7) // 8
                c.need(nb)
                for i in range(p.nvals):
                    if (c.at(c.pos + i // 8) >> (7 - i % 8)) & 1 == 1:
                        bits_p[i >> 3] |= UInt8(1) << UInt8(i & 7)
                c.pos += nb
            elif p.def_enc == ENC_RLE:
                var ln = Int(c.le(4)) & 0xFFFFFFFF
                c.need(ln)
                var lc = _Bytes(c.base + c.pos, ln)
                _levels_to_bits(lc, p.nvals, bits_p)
                c.pos += ln
            else:
                raise Error("column `" + leaf.name + "`: definition levels use encoding " + String(p.def_enc) + ", not supported")
    if ctx.optional:
        present = _count_bits(bits_p, p.nvals)
    p.present = present
    _ = nobits^
    return c^


def _decode_page(mut p: _Page, ctx: _Ctx, leaf: _Leaf, dst: Int, tbl: Int) raises:
    """Decodes one data page into the column slice at `dst` (0: only add the words of a TEXT page, store nothing)."""
    var owner = List[UInt8]()
    var c = _open_page(p, ctx, leaf, owner)
    var present = p.present
    var nobits = List[UInt8](length=8, fill=0)
    var bits_p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(p.bits.unsafe_ptr() if ctx.optional else nobits.unsafe_ptr()))
    var store = dst != 0
    if not store and not (p.encoding == ENC_PLAIN and leaf.ptype == PT_BYTE_ARRAY):
        # a words-only build (dst 0) only adds the words of PLAIN text; any other page would write through a null
        # pointer, so it is refused here, before a decoder can reach one
        raise Error("column `" + leaf.name + "`: only PLAIN text pages are read for their words (internal error)")
    var dout = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=dst)
    if p.encoding == ENC_PLAIN:
        _plain_into(c, leaf, ctx, present, dout, tbl, store, Int(UnsafePointer(to=p)) if p.defer else 0)
    elif p.encoding == ENC_RLE_DICTIONARY or p.encoding == ENC_PLAIN_DICTIONARY:
        _dict_into(c, leaf, ctx.dict, present, dout)
    elif p.encoding == ENC_DELTA_BINARY_PACKED and (leaf.ptype == PT_INT32 or leaf.ptype == PT_INT64):
        var raw = List[Int64]()
        _ = _delta_binary_packed(c, leaf.ptype == PT_INT32, present, raw)
        if len(raw) < present:
            raise Error("column `" + leaf.name + "`: a page holds fewer values than its levels say (corrupt file)")
        for i in range(present):
            dout[i] = _to_stored(leaf, raw[i])
    elif p.encoding == ENC_BYTE_STREAM_SPLIT and leaf.ptype == PT_FIXED:
        # a big-endian two's-complement decimal of type_length bytes, byte k of value i at k * count + i
        var w = leaf.type_length
        if w < 1 or w > 16:
            raise Error("column `" + leaf.name + "` has a fixed length of " + String(w) + " bytes (corrupt file)")
        if present > (c.n - c.pos) // w:
            raise Error("a Parquet page ends early (corrupt file)")
        var src = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos)
        for i in range(present):
            var v: Int64 = -1 if src[i] >= 0x80 else 0
            for k in range(max(0, w - 8), w):                 # only the last 8 bytes survive an int64
                v = (v << 8) | Int64(src[k * present + i])
            dout[i] = v
        c.pos += present * w
    elif p.encoding == ENC_BYTE_STREAM_SPLIT and (leaf.ptype == PT_FLOAT or leaf.ptype == PT_DOUBLE or leaf.ptype == PT_INT32 or leaf.ptype == PT_INT64):
        # byte k of value i is stored at k * count + i (the page's non-null values): gather the bytes back; a FLOAT is widened to
        # float64, an INT32 sign-extended (then converted as a PLAIN value is: timestamps, unsigned, decimals)
        var w = 4 if (leaf.ptype == PT_FLOAT or leaf.ptype == PT_INT32) else 8
        if present > (c.n - c.pos) // w:
            raise Error("a Parquet page ends early (corrupt file)")
        var src = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=c.base + c.pos)
        for i in range(present):
            var v: UInt64 = 0
            for k in range(w):
                v |= UInt64(src[k * present + i]) << UInt64(8 * k)
            if leaf.ptype == PT_FLOAT:
                dout[i] = Int64(Float64(Float32(from_bits=UInt32(v))).to_bits())
            else:
                var x = Int64(v)
                if leaf.ptype == PT_INT32 and v >= 0x80000000:
                    x -= 0x100000000
                dout[i] = _to_stored(leaf, x)
        c.pos += present * w
    elif p.encoding == ENC_RLE and leaf.ptype == PT_BOOLEAN:
        _ = c.le(4)                                       # a length prefix
        var vals = List[Int64]()
        _rle_hybrid(c, 1, present, vals)
        if len(vals) < present:
            raise Error("column `" + leaf.name + "`: a page holds fewer values than its levels say (corrupt file)")
        for i in range(present):
            dout[i] = vals[i]
    else:
        raise Error("column `" + leaf.name + "`: Parquet encoding " + String(p.encoding) + " is not supported yet")
    p.nulls = p.nvals - present
    if p.nulls > 0 and store:                             # spread the values over the rows, backwards
        var k = present - 1
        for row in range(p.nvals - 1, -1, -1):
            if _get_bit(bits_p, row):
                dout[row] = dout[k]
                k -= 1
            else:
                dout[row] = 0
    elif p.nulls == 0:
        p.bits = List[UInt8]()
    if len(p.miss.at) > 0:
        p.keep = owner^                               # the missed words are read from it by the serial phase
    else:
        _ = owner^
    _ = nobits^


comptime SPLIT_MIN = 262_144        # a page with at least this many rows is decoded in slices by row range
comptime SLICE_MIN = 65_536         # and no slice is smaller than this many values


def _plan_dict_cuts(mut c: _Bytes, w: Int, present: Int, vts: List[Int], mut p: _Page) raises:
    """One serial pass over a dictionary-index page's run headers (the cursor is after the bit width): an entry point for
    each target value index in `vts` (ascending): inside a bit-packed run at the group of 8 holding it, inside a repeated
    run at the run's header, each with the number of values to drop to reach the target exactly."""
    var v = 0
    var idx = 0
    while idx < len(vts):
        if vts[idx] >= present:                          # nothing left to read from there
            p.sl.pos.append(c.pos)
            p.sl.pend.append(0)
            p.sl.skip.append(0)
            idx += 1
            continue
        var start = c.pos
        var h = c.small()
        if h & 1 == 1:
            var groups = h >> 1
            if w > 0 and groups > (c.n - c.pos) // w:
                raise Error("a Parquet page ends early (corrupt file)")
            var data = c.pos
            var span = groups * 8
            while idx < len(vts) and vts[idx] < v + span and vts[idx] < present:
                var off = vts[idx] - v
                var g = off // 8
                p.sl.pos.append(data + g * w)
                p.sl.pend.append(groups - g)
                p.sl.skip.append(off - 8 * g)
                idx += 1
            v += span
            c.pos = data + groups * w
        else:
            var run = h >> 1
            while idx < len(vts) and vts[idx] < v + run and vts[idx] < present:
                p.sl.pos.append(start)
                p.sl.pend.append(0)
                p.sl.skip.append(vts[idx] - v)
                idx += 1
            c.need((w + 7) // 8)
            c.pos += (w + 7) // 8
            v += run


def _step_cuts(nvals: Int) -> List[Int]:
    """The rows at which a big page is cut into slices for the workers of a row group (F-145)."""
    var step = max(SLICE_MIN, (nvals + 2 * core_count() - 1) // (2 * core_count()))
    var cuts = List[Int]()
    var r = step
    while r < nvals:
        cuts.append(r)
        r += step
    return cuts^


def _prep_page(mut p: _Page, ctx: _Ctx, leaf: _Leaf, cut_rows: List[Int]) raises:
    """The serial part of a page decoded in slices: decompress, definition levels, and the entry point of a slice at each
    of `cut_rows` (ascending rows of the page, each below its row count) besides the one at the start; the values are
    decoded by `_decode_slice` jobs. The decompressed page stays in `p.keep` until they are done."""
    var owner = List[UInt8]()
    var c = _open_page(p, ctx, leaf, owner)
    var present = p.present
    var vts = List[Int]()                                # the number of non-null values before each cut row
    if present == p.nvals:
        for r in cut_rows:
            vts.append(r)
    else:
        var bits = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(p.bits.unsafe_ptr()))
        var seen = 0
        var done_rows = 0
        for r in cut_rows:
            while done_rows + 8 <= r:
                seen += Int(pop_count(bits[done_rows >> 3]))
                done_rows += 8
            var extra = 0
            for row in range(done_rows, r):
                if _get_bit(bits, row):
                    extra += 1
            vts.append(seen + extra)
    p.sl.v.append(0)
    p.sl.row.append(0)
    p.sl.skip.append(0)
    for i in range(len(cut_rows)):
        p.sl.v.append(vts[i])
        p.sl.row.append(cut_rows[i])
    if p.encoding == ENC_RLE_DICTIONARY or p.encoding == ENC_PLAIN_DICTIONARY:
        var w = c.byte()
        if w > 64:
            raise Error("a Parquet page has a bit width of " + String(w) + " (corrupt file)")
        p.sl.width = w
        p.sl.pos.append(c.pos)
        p.sl.pend.append(0)
        _plan_dict_cuts(c, w, present, vts, p)
    else:
        var width = 4 if (leaf.ptype == PT_INT32 or leaf.ptype == PT_FLOAT) else 8 if (leaf.ptype == PT_INT64 or leaf.ptype == PT_DOUBLE) else leaf.type_length
        if width < 1 or width > 16:
            raise Error("column `" + leaf.name + "` has a fixed length of " + String(width) + " bytes (corrupt file)")
        if present > (c.n - c.pos) // width:
            raise Error("a Parquet page ends early (corrupt file)")
        p.sl.width = width
        p.sl.pos.append(c.pos)
        p.sl.pend.append(0)
        for v in vts:
            p.sl.pos.append(c.pos + v * width)
            p.sl.pend.append(0)
            p.sl.skip.append(0)
    p.sl.err = List[String](length=len(p.sl.v), fill=String())
    p.sl.base = c.base
    p.sl.end = c.n
    p.keep = owner^


def _decode_slice(p: _Page, k: Int, ctx: _Ctx, leaf: _Leaf, dst: Int) raises:
    """Slice k of a prepared page: its values, then spread over its rows (a slice holds exactly the rows from its first
    value's row to the next slice's first value's row, so the spread never leaves it)."""
    var v1 = p.sl.v[k + 1] if k + 1 < len(p.sl.v) else p.present
    var r1 = p.sl.row[k + 1] if k + 1 < len(p.sl.row) else p.nvals
    var cnt = v1 - p.sl.v[k]
    var r0 = p.sl.row[k]
    var dout = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=dst + r0 * 8)
    var c = _Bytes(p.sl.base, p.sl.end)
    c.pos = p.sl.pos[k]
    if p.encoding == ENC_RLE_DICTIONARY or p.encoding == ENC_PLAIN_DICTIONARY:
        _dict_runs(c, leaf, p.sl.width, ctx.dict, cnt, dout, p.sl.pend[k], p.sl.skip[k])
    else:
        _plain_into(c, leaf, ctx, cnt, dout, 0, True, 0)
    if r1 - r0 > cnt:
        var bits = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(p.bits.unsafe_ptr()))
        var j = cnt - 1
        for row in range(r1 - r0 - 1, -1, -1):
            if _get_bit(bits, r0 + row):
                dout[row] = dout[j]
                j -= 1
            else:
                dout[row] = 0


def _walk_chunk(base: Int, n: Int, meta: _ChunkMeta, leaf: _Leaf, mut ctx: _Ctx, ctx_at: Int, tbl: Int,
                mut pages: List[_Page], words_only: Bool) raises:
    """Reads a chunk's page headers (`base` + file offset = the page's address; `n` bounds the offsets). A dictionary
    page is decoded now; each data page is appended to `pages`."""
    var pos = meta.dict_off if meta.dict_off > 0 and meta.dict_off < meta.data_off else meta.data_off
    var done = 0
    comptime if NO_CODECS:
        if meta.codec == ZSTD or meta.codec == SNAPPY:
            raise Error("column `" + leaf.name + "` is " + ("zstd" if meta.codec == ZSTD else "snappy") +
                        "-compressed, and this build has no zstd or snappy (save the file UNCOMPRESSED, or open it on a build with codecs)")
    if meta.values < 0:
        raise Error("column `" + leaf.name + "`: a column chunk has a negative value count (corrupt file)")
    while done < meta.values:
        if pos <= 0 or pos >= n:
            raise Error("column `" + leaf.name + "`: a page offset is outside the file (corrupt file)")
        var h = ThriftReader(base + pos, n - pos)
        var ptype = -1
        var usize = 0
        var csize = 0
        var nvals = 0
        var encoding = 0
        var def_len = 0                  # v2: definition levels' byte length
        var def_enc = ENC_RLE            # v1: how the definition levels are encoded
        var v2_compressed = True
        while True:
            var f = h.field()
            if f[1] == T_STOP:
                break
            if f[0] == 1 and f[1] == T_I32:
                ptype = Int(h.zigzag())
            elif f[0] == 2 and f[1] == T_I32:
                usize = Int(h.zigzag())
            elif f[0] == 3 and f[1] == T_I32:
                csize = Int(h.zigzag())
            elif (f[0] == 5 or f[0] == 7 or f[0] == 8) and f[1] == T_STRUCT:
                h.enter()
                while True:
                    var g = h.field()
                    if g[1] == T_STOP:
                        break
                    if g[0] == 1 and g[1] == T_I32:
                        nvals = Int(h.zigzag())
                    elif f[0] != 8 and g[0] == 2 and g[1] == T_I32:
                        encoding = Int(h.zigzag())
                    elif f[0] == 5 and g[0] == 3 and g[1] == T_I32:
                        def_enc = Int(h.zigzag())
                    elif f[0] == 8 and g[0] == 4 and g[1] == T_I32:
                        encoding = Int(h.zigzag())
                    elif f[0] == 8 and g[0] == 5 and g[1] == T_I32:
                        def_len = Int(h.zigzag())
                    elif f[0] == 8 and g[0] == 6 and g[1] == T_I32:
                        if Int(h.zigzag()) != 0:
                            raise Error("column `" + leaf.name + "` has repetition levels: not supported")
                    elif f[0] == 8 and (g[1] == T_TRUE or g[1] == T_FALSE) and g[0] == 7:
                        v2_compressed = g[1] == T_TRUE
                    else:
                        h.skip(g[1])
                h.leave()
            else:
                h.skip(f[1])
        var body_at = pos + h.pos
        if csize < 0 or usize < 0 or nvals < 0 or nvals > INT32_MAX:
            raise Error("column `" + leaf.name + "`: a page header has a negative size or count (corrupt file)")
        if csize > n - body_at:
            raise Error("column `" + leaf.name + "`: a page runs past the end of the file (corrupt file)")
        pos = body_at + csize
        if ptype == PAGE_DICTIONARY:
            if ctx.dict_seen:
                raise Error("column `" + leaf.name + "`: a chunk has two dictionary pages (corrupt file)")
            ctx.dict_seen = True
            if nvals > 8 * usize + 8:
                raise Error("column `" + leaf.name + "`: a dictionary page holds more values than its bytes can (corrupt file)")
            if ctx.add and leaf.ptype == PT_BYTE_ARRAY:  # text words are added to the table here, in file order
                var page = List[UInt8]()
                var c = _Bytes(_page_bytes(meta.codec, base + body_at, csize, usize, page), usize)
                ctx.dict = List[Int64](unsafe_uninit_length=nvals)
                _plain_into(c, leaf, ctx, nvals, UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(ctx.dict.unsafe_ptr())),
                            tbl, True)
                _ = page^
            else:                                        # a job of the first parallel round
                pages.append(_Page(base + body_at, csize, usize, nvals, ENC_PLAIN, False, 0, ENC_RLE, True, ctx_at, 0, is_dict=True, foff=body_at))
            continue
        if ptype != PAGE_DATA and ptype != PAGE_DATA_V2:
            continue                                     # an index page: not needed
        if nvals > meta.values - done:
            raise Error("column `" + leaf.name + "`: a page holds more values than its column chunk (corrupt file)")
        var dict_coded = encoding == ENC_RLE_DICTIONARY or encoding == ENC_PLAIN_DICTIONARY
        if words_only and dict_coded:
            done += nvals
            continue
        var defer = leaf.ptype == PT_BYTE_ARRAY and ctx.add and encoding == ENC_PLAIN
        pages.append(_Page(base + body_at, csize, usize, nvals, encoding, ptype == PAGE_DATA_V2, def_len, def_enc,
                           v2_compressed, ctx_at, ctx.row0 + done, defer=defer, foff=body_at))
        done += nvals


def physical_memory() -> Int:
    """The machine's physical memory in bytes (sysconf), 0 when it cannot be told."""
    comptime if CompilationTarget.is_linux():
        var pages = external_call["sysconf", Int](Int32(85))         # _SC_PHYS_PAGES
        var size = external_call["sysconf", Int](Int32(30))          # _SC_PAGESIZE
        return pages * size if pages > 0 and size > 0 else 0
    else:
        var pages = external_call["sysconf", Int](Int32(200))        # Darwin's _SC_PHYS_PAGES
        var size = external_call["sysconf", Int](Int32(29))          # Darwin's _SC_PAGESIZE
        return pages * size if pages > 0 and size > 0 else 0


def _check_claim(what: String, rows: Int, ncols: Int) raises:
    """Refuses a row count whose decoded columns (8 bytes a value) could not fit in the machine's memory. The columns
    are sized from the counts a file states, so this is what stops a file of a few KB from asking for terabytes.
    Limits: a claim below the machine's memory (or an address-space cap lower than it) is still sized, untouched until
    its pages decode, and a corrupt page raises when it does."""
    if rows < 0:
        raise Error(what + " claims a negative row count (" + String(rows) + ") (corrupt file)")
    var ram = physical_memory()
    if ram > 0 and rows > ram // (8 * max(ncols, 1)):
        var need = Float64(rows) * 8.0 * Float64(max(ncols, 1))
        raise Error(what + " claims " + String(rows) + " rows: " + String(need / 1e9) + " GB for " + String(ncols) +
                    (" column" if ncols == 1 else " columns") + ", more than this machine's memory (" + String(Float64(ram) / 1e9) +
                    " GB); a corrupt file, or too big to read whole (stream it with scan_parquet)")


comptime DEDUPE_MIN = 50_000        # misses of a column in a row group from which the cross-page dedupe runs in parallel


def _dedupe_misses(mut pages: List[_Page], ctx: Int, total: Int):
    """For the misses of column chunk `ctx` (all absent from the frozen index), find for each the FIRST miss, in page
    and miss order, of the same word: the words are split into hash partitions and each partition is walked by one
    task, so the serial phase only has to name the first occurrences (in order) and copy their codes."""
    var parts = 64
    for i in range(len(pages)):
        if pages[i].ctx == ctx:
            pages[i].miss.canon = List[Int](length=len(pages[i].miss.at), fill=0)
    var pa = Int(pages.unsafe_ptr())
    var np = len(pages)
    def part(w: Int) {imm}:
        var pg = Pointer[_Page, MutUntrackedOrigin](unsafe_from_address=pa)
        var size = 1024
        while size < 4 * (total // parts + 1):
            size *= 2
        var slots = List[Int](length=size, fill=-1)
        var count = 0
        for i in range(np):
            ref p = (pg + i)[]
            if p.ctx != ctx:
                continue
            for m in range(len(p.miss.at)):
                if Int((p.miss.h[m] >> 32) & UInt64(parts - 1)) != w:
                    continue
                var mask = size - 1
                var j = Int(p.miss.h[m]) & mask
                var found = -1
                while slots[j] >= 0:
                    var r = slots[j]
                    ref q = (pg + (r >> 32))[]
                    var qm = r & 0xFFFFFFFF
                    if q.miss.h[qm] == p.miss.h[m] and q.miss.n[qm] == p.miss.n[m]:
                        var a = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=q.miss.at[qm])
                        var b = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=p.miss.at[m])
                        var same = True
                        for k in range(p.miss.n[m]):
                            if a[k] != b[k]:
                                same = False
                                break
                        if same:
                            found = r
                            break
                    j = (j + 1) & mask
                if found >= 0:
                    p.miss.canon[m] = found
                else:
                    p.miss.canon[m] = (i << 32) | m
                    slots[j] = (i << 32) | m
                    count += 1
                    if count * 2 > size:                 # a skewed partition: double the table and re-place its words
                        var bigger = List[Int](length=size * 2, fill=-1)
                        for e in range(size):
                            if slots[e] >= 0:
                                ref q = (pg + (slots[e] >> 32))[]
                                var k = Int(q.miss.h[slots[e] & 0xFFFFFFFF]) & (size * 2 - 1)
                                while bigger[k] >= 0:
                                    k = (k + 1) & (size * 2 - 1)
                                bigger[k] = slots[e]
                        slots = bigger^
                        size *= 2
    par_n(part, parts, min(parts, core_count()))


def _decode_group(bases: List[Int], ns: List[Int], metas: List[_ChunkMeta], leaf_ids: List[Int], out_cols: List[Int],
                  widxs: List[Int], add_words: Bool, leaves: List[_Leaf], row0: Int, mut t: Table, words_only: Bool = False,
                  ext: Int = 0) raises:
    """Decodes one row group: chunk k (`metas[k]`, at `bases[k]` / `ns[k]`, leaf `leaf_ids[k]`) becomes rows
    row0.. of table column `out_cols[k]`, whose TEXT words are found in `widxs[k]` (a `_WordIndex` over the table's
    column) and, when `add_words`, added to it. `words_only`: only the TEXT words are added to the table (a dictionary build).
    `ext` (F-146): the address of a List[Int] of one destination address per out column; the values are written there (host
    memory of the caller's, e.g. pinned for the GPU) and the table's columns keep no values, only their validity bitmaps and
    null counts."""
    if len(metas) == 0:
        return
    var tbl = Int(UnsafePointer(to=t))
    var ctxs = List[_Ctx]()
    var pages = List[_Page]()
    var rows = -1
    for k in range(len(metas)):
        ctxs.append(_Ctx(leaf_ids[k], out_cols[k], metas[k].codec, leaves[leaf_ids[k]].repetition == OPTIONAL, widxs[k],
                         add_words, List[Int64](), row0, False))
        if rows < 0:
            rows = metas[k].values
        elif metas[k].values != rows:
            raise Error("column `" + leaves[leaf_ids[k]].name + "` has " + String(metas[k].values) + " rows in this row group, column `" +
                        leaves[leaf_ids[0]].name + "` has " + String(rows) + " (corrupt file)")
        _walk_chunk(bases[k], ns[k], metas[k], leaves[leaf_ids[k]], ctxs[k], k, tbl, pages, words_only)
    if not words_only:
        for k in range(len(metas)):                     # the pages add up to the chunk (the walk); the chunk to the group
            if metas[k].group_rows >= 0 and metas[k].values != metas[k].group_rows:
                raise Error("column `" + leaves[leaf_ids[k]].name + "` has " + String(metas[k].values) + " values in a row group of " +
                            String(metas[k].group_rows) + " rows (corrupt file)")
        _check_claim("row group", rows, len(out_cols))
        for k in range(len(out_cols)):                  # size the columns once: the pages add up to the rows
            ref col = t.cols[out_cols[k]]
            if ext != 0:
                continue
            if len(col.values) < row0:
                raise Error("column `" + col.name + "`: rows do not line up (corrupt file)")
            if len(col.values) < row0 + rows:            # (read_parquet sizes the whole column up front)
                if row0 == 0:                            # every row is written below, values or a zero for a null
                    col.values = List[Int64](unsafe_uninit_length=rows)
                else:
                    col.values.resize(row0 + rows, 0)
            if 0 < len(col.validity) < (len(col.values) + 7) // 8:      # a bitmap that exists covers every row
                col.validity.resize((len(col.values) + 7) // 8, 0xFF)
    var ob = List[Int](length=len(t.cols), fill=0)      # where each out column's values are written
    if not words_only:
        for k in range(len(out_cols)):
            if ext != 0:
                ob[out_cols[k]] = UnsafePointer[List[Int], MutAnyOrigin](unsafe_from_address=ext)[][k]
            else:
                ob[out_cols[k]] = Int(t.cols[out_cols[k]].values.unsafe_ptr())
    var oba = Int(ob.unsafe_ptr())
    if core_count() > 1 and not words_only:             # big pages are decoded in slices (prepared by one job, then a job per slice)
        for i in range(len(pages)):
            ref p = pages[i]
            if p.is_dict or p.defer or p.nvals < SPLIT_MIN:
                continue
            var pt = leaves[ctxs[p.ctx].leaf].ptype
            if p.encoding == ENC_RLE_DICTIONARY or p.encoding == ENC_PLAIN_DICTIONARY or \
               (p.encoding == ENC_PLAIN and (pt == PT_INT32 or pt == PT_INT64 or pt == PT_FIXED or pt == PT_FLOAT or pt == PT_DOUBLE)):
                p.split = True
    # Two parallel rounds. A: the dictionary pages, the pages that need no dictionary (PLAIN, text that adds words) and
    # the preparation of every big page (decompress, levels, where the slices start). B: what needs a finished
    # dictionary or a prepared page: the small dictionary-indexed pages and one job per slice of the big ones.
    var ja = List[Int]()
    var jb = List[Int]()
    var kb = List[Int]()
    var dict_job = List[Bool](length=len(ctxs), fill=False)     # chunks whose dictionary is a job of round A
    for i in range(len(pages)):
        if pages[i].is_dict:
            ja.append(i)
            dict_job[pages[i].ctx] = True
    for i in range(len(pages)):
        ref p = pages[i]
        if p.is_dict:
            continue
        # (a dictionary decoded while the headers were walked, as a text dictionary under read_parquet is, is already
        # there: such a page needs no second round)
        if p.split or not (p.encoding == ENC_RLE_DICTIONARY or p.encoding == ENC_PLAIN_DICTIONARY) or not dict_job[p.ctx]:
            ja.append(i)
        else:
            jb.append(i)
            kb.append(-1)
    var pa = Int(pages.unsafe_ptr())
    var ca = Int(ctxs.unsafe_ptr())
    var la = Int(leaves.unsafe_ptr())
    var cola = Int(t.cols.unsafe_ptr())
    if len(ja) > 0:
        var jaa = Int(ja.unsafe_ptr())
        def job_a(j: Int) {imm}:
            ref p = (Pointer[_Page, MutUntrackedOrigin](unsafe_from_address=pa) + Pointer[Int, MutUntrackedOrigin](unsafe_from_address=jaa)[j])[]
            try:
                if p.is_dict:
                    # (the dictionary is written into the chunk's `_Ctx`; the other jobs of this round take it as a plain
                    # parameter. With `ref ctx` parameters the write was lost: see docs/lessons.md)
                    ref wctx = (Pointer[_Ctx, MutUntrackedOrigin](unsafe_from_address=ca) + p.ctx)[]
                    ref wleaf = (Pointer[_Leaf, MutUntrackedOrigin](unsafe_from_address=la) + wctx.leaf)[]
                    _decode_dict_page(p, wctx, wleaf, tbl)
                else:
                    ref ctx = (Pointer[_Ctx, MutUntrackedOrigin](unsafe_from_address=ca) + p.ctx)[]
                    ref leaf = (Pointer[_Leaf, MutUntrackedOrigin](unsafe_from_address=la) + ctx.leaf)[]
                    ref col = (Pointer[Column, MutUntrackedOrigin](unsafe_from_address=cola) + ctx.col)[]
                    if p.split:
                        _prep_page(p, ctx, leaf, _step_cuts(p.nvals))
                    else:
                        _decode_page(p, ctx, leaf, 0 if words_only else Pointer[Int, MutUntrackedOrigin](unsafe_from_address=oba)[ctx.col] + p.row * 8, tbl)
            except e:
                p.err = String(e)
        if len(ja) == 1:
            job_a(0)
        else:
            par_n(job_a, len(ja), min(len(ja), core_count()))
    _ = ja^
    for i in range(len(pages)):                         # a job per slice of every prepared page
        if pages[i].split and pages[i].err.byte_length() == 0:
            for k in range(len(pages[i].sl.v)):
                jb.append(i)
                kb.append(k)
    if len(jb) > 0:
        var jba = Int(jb.unsafe_ptr())
        var kba = Int(kb.unsafe_ptr())
        def job_b(j: Int) {imm}:
            ref p = (Pointer[_Page, MutUntrackedOrigin](unsafe_from_address=pa) + Pointer[Int, MutUntrackedOrigin](unsafe_from_address=jba)[j])[]
            var k = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=kba)[j]
            ref ctx = (Pointer[_Ctx, MutUntrackedOrigin](unsafe_from_address=ca) + p.ctx)[]
            ref leaf = (Pointer[_Leaf, MutUntrackedOrigin](unsafe_from_address=la) + ctx.leaf)[]
            ref col = (Pointer[Column, MutUntrackedOrigin](unsafe_from_address=cola) + ctx.col)[]
            try:
                if k < 0:
                    _decode_page(p, ctx, leaf, Pointer[Int, MutUntrackedOrigin](unsafe_from_address=oba)[ctx.col] + p.row * 8, tbl)
                else:
                    _decode_slice(p, k, ctx, leaf, Pointer[Int, MutUntrackedOrigin](unsafe_from_address=oba)[ctx.col] + p.row * 8)
            except e:
                if k < 0:
                    p.err = String(e)
                else:
                    p.sl.err[k] = String(e)
        if len(jb) == 1:
            job_b(0)
        else:
            par_n(job_b, len(jb), min(len(jb), core_count()))
    _ = jb^
    _ = kb^
    for i in range(len(pages)):
        if pages[i].split and pages[i].err.byte_length() == 0:
            for k in range(len(pages[i].sl.err)):
                if pages[i].sl.err[k].byte_length() > 0:
                    pages[i].err = pages[i].sl.err[k]
                    break
            pages[i].nulls = pages[i].nvals - pages[i].present
            if pages[i].nulls == 0:
                pages[i].bits = List[UInt8]()
            pages[i].keep = List[UInt8]()
    for i in range(len(pages)):
        if pages[i].err.byte_length() > 0:
            raise Error(pages[i].err)
    # the serial phase: the words the frozen index missed get their codes in file order (first seen first, as one
    # serial pass would give them), then the parallel phase writes those codes over the placeholders
    var fixes = List[Int]()
    for k in range(len(ctxs)):
        var total = 0
        for i in range(len(pages)):
            if pages[i].ctx == k:
                total += len(pages[i].miss.at)
        if total >= DEDUPE_MIN:
            _dedupe_misses(pages, k, total)
    for i in range(len(pages)):
        if len(pages[i].miss.at) > 0:
            ref ctx = ctxs[pages[i].ctx]
            var wi = UnsafePointer[_WordIndex, MutAnyOrigin](unsafe_from_address=ctx.widx)
            var deduped = len(pages[i].miss.canon) > 0
            for m in range(len(pages[i].miss.at)):
                var code = -1
                if deduped:
                    var r = pages[i].miss.canon[m]
                    if r != (i << 32) | m:               # the same word was missed first elsewhere: its code is ready
                        pages[i].miss.code.append(pages[r >> 32].miss.code[r & 0xFFFFFFFF])
                        continue
                else:
                    code = wi[].find(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=pages[i].miss.at[m]),
                                     pages[i].miss.n[m], pages[i].miss.h[m])
                if code < 0:
                    var wp = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=pages[i].miss.at[m])
                    var w = String(unsafe_from_utf8=Span[UInt8, MutAnyOrigin](unsafe_ptr=wp, length=pages[i].miss.n[m]))
                    code = t.add_word(ctx.col, w)
                    wi[].insert(pages[i].miss.h[m], code)
                pages[i].miss.code.append(code)
            fixes.append(i)
    if words_only:
        return
    if len(fixes) > 0:
        var fa = Int(fixes.unsafe_ptr())
        def fill(j: Int) {imm}:
            var page = Pointer[_Page, MutUntrackedOrigin](unsafe_from_address=pa)
            ref p = (page + Pointer[Int, MutUntrackedOrigin](unsafe_from_address=fa)[j])[]
            ref ctx = (Pointer[_Ctx, MutUntrackedOrigin](unsafe_from_address=ca) + p.ctx)[]
            ref col = (Pointer[Column, MutUntrackedOrigin](unsafe_from_address=cola) + ctx.col)[]
            var dst = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Pointer[Int, MutUntrackedOrigin](unsafe_from_address=oba)[ctx.col] + p.row * 8)
            var codes = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(p.miss.code.unsafe_ptr()))
            for r in range(p.nvals):
                var v = dst[r]
                if v < 0:
                    dst[r] = Int64(codes[Int(-1 - v)])
        if len(fixes) == 1:
            fill(0)
        else:
            par_n(fill, len(fixes), min(len(fixes), core_count()))
        for i in fixes:
            pages[i].keep = List[UInt8]()
    _ = fixes^
    for i in range(len(pages)):                         # nulls and validity bitmaps, in page order
        ref p = pages[i]
        ref col = t.cols[ctxs[p.ctx].col]
        col.nulls += p.nulls
        if p.nulls > 0:
            if len(col.validity) == 0:
                col.validity = List[UInt8](length=((len(col.values) if ext == 0 else row0 + rows) + 7) // 8, fill=0xFF)
            _assign_bits(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(col.validity.unsafe_ptr())), p.row,
                         UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(p.bits.unsafe_ptr())), p.nvals)
    _ = pages^
    _ = ctxs^
    _ = ob^


# ── scanning: a file bigger than memory, one row group at a time (F-140) ─────────────────────────────

def _open_file(path: String) raises -> FileHandle:
    var file: FileHandle
    try:
        file = open(path, "r")
    except:
        raise Error("couldn't open `" + path + "`: no such file, or no permission to read it")
    return file^


def _file_size(file: FileHandle, path: String) raises -> Int:
    var size = external_call["lseek", Int](file._get_raw_fd(), Int(0), Int32(2))
    if size < 0:
        raise Error("couldn't read `" + path + "`: not a regular file")
    return size


def _chunk_source(fd: Int, size: Int, path: String, meta: _ChunkMeta, leaf: _Leaf, mut maps: List[_Mapped]) raises -> Tuple[Int, Int]:
    """Maps one column chunk's bytes (never the whole file); returns the (base, bound) `_walk_chunk` takes: base + a
    file offset is that byte's address, bound is the offset just past the chunk. Known limit: if the file is truncated
    while the mapping is decoded, touching a vanished page is a bus error; a change between row groups is caught by
    `_check_unchanged`."""
    if meta.values == 0:
        return (0, 0)
    var lo = meta.dict_off if meta.dict_off > 0 and meta.dict_off < meta.data_off else meta.data_off
    if lo <= 0 or meta.size <= 0 or meta.size > size - lo:
        raise Error("column `" + leaf.name + "`: a column chunk lies outside the file (corrupt file)")
    var m = _Mapped(fd, lo, meta.size, path)
    var base = m.data - lo
    maps.append(m^)
    return (base, lo + meta.size)


struct _ScanInfo(Movable):
    var path: String
    var file: FileHandle             # kept open: a file replaced by rename is still the one that was opened
    var size: Int
    var leaves: List[_Leaf]
    var groups: List[List[_ChunkMeta]]
    var rows_of: List[Int]
    var metadata: String
    var dicts: Table                 # TEXT columns' file-wide words, each built on first use
    var widx: List[_WordIndex]       # and their indexes by bytes
    var loaded: List[Int64]          # 1 once the dictionary is built: written with a release store, read with an acquire load (F-175: a job's thread and its caller)
    var extra: List[String]          # the file's other key-value metadata (key, value, ...)
    var build_lock: List[Int64]      # F-175: one word, 1 while a thread builds a dictionary (a job and its caller share a scan)
    var is_csv: Bool                 # F-178: the rows are a CSV file's (`scan_csv`): `csv` holds its row groups, and `leaves` / `groups` are made from it
    var csv: CsvSource

    def __init__(out self, path: String, var file: FileHandle, size: Int, var leaves: List[_Leaf],
                 var groups: List[List[_ChunkMeta]], var rows_of: List[Int], var metadata: String, var dicts: Table):
        self.path = path
        self.file = file^
        self.size = size
        self.loaded = List[Int64](length=len(leaves), fill=0)
        self.build_lock = List[Int64](length=1, fill=0)
        self.is_csv = False
        self.csv = CsvSource()
        self.extra = List[String]()
        self.widx = List[_WordIndex]()
        for _ in range(len(leaves)):
            self.widx.append(_WordIndex())
        self.leaves = leaves^
        self.groups = groups^
        self.rows_of = rows_of^
        self.metadata = metadata^
        self.dicts = dicts^

    def check_unchanged(self) raises:
        """A truncated or resized file: refuse before mapping pages that may be gone (a same-size rewrite is not seen)."""
        if external_call["lseek", Int](self.file._get_raw_fd(), Int(0), Int32(2)) != self.size:
            raise Error("the file changed since it was opened (`" + self.path + "`)")


struct ParquetScan(Copyable, Movable):
    """A Parquet file opened for streaming: its footer and columns, none of the data. `Query(scan)` runs a query over it
    one row group at a time (query.mojo); copies share everything, including the open file.

    A TEXT code is the code `read_parquet` gives the word (first seen, in file order), so a filter literal or a
    group count taken here is right for the query. A TEXT column's file-wide dictionary is built the first time it is
    asked for (`word_count`, `code_of`, `text_of`) or a query reads the column, from its dictionary pages (a PLAIN
    text page, as after a writer's dictionary fallback, is decoded for its words); a column nobody reads costs
    nothing. Limit: a high-cardinality text column that IS read keeps all its distinct words in memory.

    The file is held open (a file replaced by rename keeps reading the one opened). If it is truncated or resized, the
    next row group raises "the file changed since it was opened". Known limits: a rewrite in place that keeps the size
    is not detected (only the size is compared), and a truncation during the decoding of one column chunk is a bus
    error. No reader is safe against a concurrent writer; `write_parquet` replaces files by rename."""
    var _info: ArcPointer[_ScanInfo]
    var schema: SharedTable          # the columns (kinds and scales), no rows and no words

    def __init__(out self, var info: _ScanInfo, var schema: SharedTable):
        self._info = ArcPointer(info^)
        self.schema = schema^

    def rows(self) -> Int:
        var n = 0
        for r in self._info[].rows_of:
            n += r
        return n

    def row_groups(self) -> Int:
        return len(self._info[].groups)

    def csv_report(self) raises -> CsvReport:
        """A CSV scan's import report (F-178): the columns' kinds, empty and failed cells, the failed cells themselves (`failures()`: row,
        column, the text, why), the ragged rows. Raises for a Parquet scan."""
        if not self._info[].is_csv:
            raise Error("not a scan of a CSV file")
        return self._info[].csv.report.copy()

    def group_rows(self, g: Int) -> Int:
        return self._info[].rows_of[g]

    def metadata(self) -> String:
        """The file's "omx" key-value metadata, "" when there is none."""
        return self._info[].metadata

    def extra_metadata(self) -> List[String]:
        """The file's key-value metadata other than "omx" and Arrow's schema, key then value, key then value...
        (`ParquetWriter.finish(.., extra)` writes it back)."""
        return self._info[].extra.copy()

    def narrowing(self, c: Int) -> String:
        """The VALUES reading column `c` lost that saving would not restore, "" when none: a TIMESTAMP is held as its
        UTC day (the time of day is gone). `save` refuses this unless asked. A change of type that keeps every value
        is `type_change(c)` instead."""
        ref leaf = self._info[].leaves[c]
        if _is_timestamp(leaf):
            return "TIMESTAMP in the file and is held as a DATE: saving would lose the time of day"
        return ""

    def type_change(self, c: Int) -> String:
        """How saving column `c` would change its type while keeping every value, "" when it would not: a DECIMAL(p, s)
        is written as DECIMAL(18, s), an ENUM or JSON string as plain text, a narrower or unsigned integer as a signed
        64-bit one (values past int64 are refused when read). `save` writes these without asking."""
        ref leaf = self._info[].leaves[c]
        var is_decimal = (leaf.converted == CT_DECIMAL or leaf.logical == 5)
        var is_date = (leaf.converted == CT_DATE or leaf.logical == 6)
        if leaf.ptype == PT_INT32 and not is_decimal and not is_date:
            return ("an unsigned " if _is_unsigned(leaf) else "a ") + "32-bit or narrower integer in the file: saving writes a signed 64-bit one"
        if leaf.ptype == PT_INT64 and _is_unsigned(leaf):
            return "an unsigned 64-bit integer in the file: saving writes a signed one"
        if leaf.ptype == PT_FLOAT:
            return "a 32-bit float in the file: saving writes a 64-bit DOUBLE (every value is kept exactly)"
        if (leaf.converted == CT_DECIMAL or leaf.logical == 5) and leaf.precision > 0 and leaf.precision != 18:
            return "DECIMAL(" + String(leaf.precision) + ", " + String(leaf.scale) + ") in the file: saving would write DECIMAL(18, " + \
                   String(leaf.scale) + ")"
        if leaf.converted == CT_ENUM or leaf.logical == 4:
            return "an ENUM in the file: saving would write plain text"
        if leaf.converted == CT_JSON or leaf.logical == 12:
            return "JSON in the file: saving would write plain text"
        return ""

    def null_count(self, c: Int) -> Int:
        """The nulls column `c` holds according to the file's statistics, -1 when a row group states none (or the column is
        REQUIRED, which holds none: 0)."""
        if self._info[].leaves[c].repetition != OPTIONAL:
            return 0
        var total = 0
        for g in range(len(self._info[].groups)):
            ref ch = self._info[].groups[g][c]
            if ch.nulls < 0:
                return -1
            total += ch.nulls
        return total

    def file_bytes(self) -> Int:
        """The size of the file in bytes."""
        return self._info[].size

    def group_null_count(self, g: Int, c: Int) -> Int:
        """The nulls row group `g` states for column `c` in its Statistics, -1 when it states none."""
        return self._info[].groups[g][c].nulls

    def column_range(self, c: Int) -> Tuple[Bool, Int64, Int64]:
        """(known, lowest, highest) stored value of column `c` from the row groups' statistics, for INT, DECIMAL and DATE columns on
        INT32, INT64 or a short FIXED_LEN_BYTE_ARRAY decimal: `known` is False when a row group states none or the column is of another
        kind (and when there is no value at all, known is True with lowest > highest)."""
        var k = self.schema.kind(c)
        if k != INT and k != DECIMAL and k != DATE:
            return (False, 0, 0)
        ref leaf = self._info[].leaves[c]
        var fixed = leaf.ptype == PT_FIXED and k == DECIMAL and leaf.type_length >= 1 and leaf.type_length <= 8
        if (leaf.ptype != PT_INT32 and leaf.ptype != PT_INT64 and not fixed) or (_is_unsigned(leaf) and leaf.ptype == PT_INT64):
            return (False, 0, 0)
        var want = 4 if leaf.ptype == PT_INT32 else 8 if leaf.ptype == PT_INT64 else leaf.type_length
        var lo = Int64.MAX
        var hi = Int64.MIN
        for g in range(len(self._info[].groups)):
            ref ch = self._info[].groups[g][c]
            if ch.values == 0 or (ch.nulls >= 0 and ch.nulls == ch.values):          # no value in the chunk
                continue
            if ch.stat_n != want:
                return (False, 0, 0)
            try:
                lo = min(lo, _to_stored(leaf, ch.min_be if fixed else ch.min_le))
                hi = max(hi, _to_stored(leaf, ch.max_be if fixed else ch.max_le))
            except:
                return (False, 0, 0)
        return (True, lo, hi)

    def nullable(self, c: Int) -> Bool:
        """Column `c` is declared OPTIONAL in the file (it may hold nulls; a REQUIRED column holds none)."""
        return self._info[].leaves[c].repetition == OPTIONAL

    def columns(self) -> Int:
        return self.schema.columns()

    def name(self, c: Int) -> String:
        return self.schema.name(c)

    def kind(self, c: Int) -> ColumnKind:
        return self.schema.kind(c)

    def scale(self, c: Int) -> Int:
        return self.schema.scale(c)

    def index(self, name: String) -> Int:
        return self.schema.index(name)

    def column(self, name: String) raises -> Int:
        return self.schema.column(name)

    def dictionary_loaded(self, c: Int) -> Bool:
        """Whether column `c`'s file-wide dictionary has been built (it is built on first use)."""
        return Atomic[Scalar[DType.int64]].load(Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(self._info[].loaded.unsafe_ptr())) + c) != 0

    def ensure_words(self, c: Int) raises:
        """Builds TEXT column `c`'s file-wide dictionary (once)."""
        if self.schema.kind(c) != TEXT or self.dictionary_loaded(c):
            return
        ref info = self._info[]
        # F-175: a background job and its caller may ask for the same dictionary at once: one builds it, the other waits for it
        var lock = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(info.build_lock.unsafe_ptr()))
        while True:
            var free = Int64(0)
            if Atomic[Scalar[DType.int64]].compare_exchange(lock, free, 1):
                break
            comptime if THREADS:
                _ = external_call["sched_yield", Int32]()      # the other thread is building: give it the core
        try:
            if Atomic[Scalar[DType.int64]].load(Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(info.loaded.unsafe_ptr())) + c) == 0:
                info.check_unchanged()
                info.widx[c].col = Int(info.dicts.cols.unsafe_ptr() + c)
                if info.is_csv:
                    var want = List[Bool](length=len(info.leaves), fill=False)
                    want[c] = True
                    for g in range(len(info.groups)):
                        var sink = csv_parse_group(info.csv, info.file._get_raw_fd(), info.path, g, want, True)
                        for w in range(len(sink.cols[c].words.starts)):
                            _ = info.dicts.add_word(c, sink.cols[c].words.word(w))
                else:
                    for g in range(len(info.groups)):
                        var maps = List[_Mapped]()
                        var src = _chunk_source(info.file._get_raw_fd(), info.size, info.path, info.groups[g][c], info.leaves[c], maps)
                        _decode_group([src[0]], [src[1]], [info.groups[g][c].copy()], [c], [c], [Int(info.widx.unsafe_ptr() + c)], True,
                                      info.leaves, 0, info.dicts, words_only=True)
                        _ = maps^
                Atomic[Scalar[DType.int64]].store(Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(info.loaded.unsafe_ptr())) + c, 1)
        except e:
            Atomic[Scalar[DType.int64]].store(lock, 0)
            raise e^
        Atomic[Scalar[DType.int64]].store(lock, 0)

    def word_count(self, c: Int) raises -> Int:
        """A TEXT column's number of words in the whole file: the `groups=` of a query keyed by it."""
        self.ensure_words(c)
        return len(self._info[].dicts.dictionary(c))

    def text_of(self, c: Int, code: Int) raises -> String:
        self.ensure_words(c)
        return self._info[].dicts.text_of(c, code)

    def words_address(self, c: Int) raises -> Int:
        """The address of TEXT column `c`'s file-wide words (an array of `word_count(c)` Strings), for a reader that goes through them in parallel."""
        self.ensure_words(c)
        return Int(self._info[].dicts.dictionary(c).unsafe_ptr())

    def code_of(self, c: Int, word: String) raises -> Int:
        """The word's code, -1 when the file never holds it."""
        self.ensure_words(c)
        return self._info[].dicts.code_of(c, word)

    def read_group(self, g: Int, cols: List[Int]) raises -> Table:
        """Row group `g` as a table of just the columns `cols` (in that order), decoded: a TEXT value is the code the
        whole-file read gives its word (looked up in the file-wide dictionary while the pages are decoded)."""
        return self._read(g, cols, 0)

    def read_group_into(self, g: Int, cols: List[Int], dest: List[Int]) raises -> Table:
        """`read_group`, the values written to the addresses in `dest` (one buffer of `group_rows(g)` Int64 per column
        read, the caller's, e.g. pinned memory) instead of into the table, which keeps only validity and null counts."""
        return self._read(g, cols, Int(UnsafePointer(to=dest)))

    def _read(self, g: Int, cols: List[Int], ext: Int) raises -> Table:
        for c in cols:
            self.ensure_words(c)
        ref info = self._info[]
        info.check_unchanged()
        if info.is_csv:
            var want = List[Bool](length=len(info.leaves), fill=False)
            for c in cols:
                want[c] = True
            var sink = csv_parse_group(info.csv, info.file._get_raw_fd(), info.path, g, want, True)
            return csv_table(sink, cols, self.schema, info.dicts, ext)
        var t = Table.empty()
        var maps = List[_Mapped]()
        var bases = List[Int]()
        var ns = List[Int]()
        var metas = List[_ChunkMeta]()
        var outs = List[Int]()
        var widxs = List[Int]()
        for k in range(len(cols)):
            var c = cols[k]
            _ = t.add_column(self.schema.name(c), self.schema.kind(c), scale=self.schema.scale(c))
            var src = _chunk_source(info.file._get_raw_fd(), info.size, info.path, info.groups[g][c], info.leaves[c], maps)
            bases.append(src[0])
            ns.append(src[1])
            metas.append(info.groups[g][c].copy())
            outs.append(k)
            widxs.append(Int(info.widx.unsafe_ptr() + c) if self.schema.kind(c) == TEXT else 0)
        _decode_group(bases, ns, metas, cols, outs, widxs, False, info.leaves, 0, t, ext=ext)
        _ = maps^
        t.check()
        return t^


# ── a row group read block by block (F-149) ──────────────────────────────────────────────────────────

struct _GroupPlan(Movable):
    """One row group of a scan, opened: the page headers of the columns read are walked and the dictionaries decoded.
    Read only from then on, so any number of `_GroupReader`s (the row-range morsels of the group) can share it."""
    var scan: ParquetScan
    var cols: List[Int]
    var ctxs: List[_Ctx]
    var pages: List[_Page]
    var of_col: List[List[Int]]      # per column read: its data pages, in row order
    var rows: Int
    var decoded: Table               # a CSV scan (F-178): the group's columns, parsed whole when it is opened; the readers copy blocks of it

    def __init__(out self, var scan: ParquetScan, cols: List[Int]):
        self.decoded = Table.empty()
        self.scan = scan^
        self.cols = cols.copy()
        self.ctxs = List[_Ctx]()
        self.pages = List[_Page]()
        self.of_col = List[List[Int]]()
        self.rows = 0


def _open_plan(scan: ParquetScan, g: Int, cols: List[Int]) raises -> _GroupPlan:
    """Opens row group `g`: its headers are walked and its dictionaries decoded here. The dictionaries of the TEXT
    columns in `cols` must have been built (`ParquetScan.ensure_words`)."""
    var plan = _GroupPlan(scan.copy(), cols)
    ref info = scan._info[]
    info.check_unchanged()
    if info.is_csv:
        plan.decoded = scan._read(g, cols, 0)
        plan.rows = plan.decoded.rows()
        return plan^
    var maps = List[_Mapped]()                           # the chunks, while the headers are walked and the dictionaries decoded
    var rows = -1
    for k in range(len(cols)):
        var c = cols[k]
        ref meta = info.groups[g][c]
        var src = _chunk_source(info.file._get_raw_fd(), info.size, info.path, meta, info.leaves[c], maps)
        plan.ctxs.append(_Ctx(c, k, meta.codec, info.leaves[c].repetition == OPTIONAL,
                              Int(info.widx.unsafe_ptr() + c) if scan.schema.kind(c) == TEXT else 0, False, List[Int64](), 0, False))
        if rows < 0:
            rows = meta.values
        elif meta.values != rows:
            raise Error("column `" + info.leaves[c].name + "` has " + String(meta.values) + " rows in this row group, column `" +
                        info.leaves[cols[0]].name + "` has " + String(rows) + " (corrupt file)")
        if meta.group_rows >= 0 and meta.values != meta.group_rows:
            raise Error("column `" + info.leaves[c].name + "` has " + String(meta.values) + " values in a row group of " +
                        String(meta.group_rows) + " rows (corrupt file)")
        _walk_chunk(src[0], src[1], meta, info.leaves[c], plan.ctxs[k], k, 0, plan.pages, False)
        plan.of_col.append(List[Int]())
    plan.rows = max(rows, 0)
    for i in range(len(plan.pages)):
        if plan.pages[i].is_dict:
            var k = plan.pages[i].ctx
            _decode_dict_page(plan.pages[i], plan.ctxs[k], info.leaves[cols[k]], 0)
    _ = maps^                                            # the chunks are let go: pages are mapped one at a time from here
    for i in range(len(plan.pages)):
        ref p = plan.pages[i]
        if p.is_dict:
            continue
        var pt = info.leaves[cols[p.ctx]].ptype
        var dict_coded = p.encoding == ENC_RLE_DICTIONARY or p.encoding == ENC_PLAIN_DICTIONARY
        if not (dict_coded or (p.encoding == ENC_PLAIN and (pt == PT_INT32 or pt == PT_INT64 or pt == PT_FIXED or pt == PT_FLOAT or pt == PT_DOUBLE))):
            p.whole = True                               # text, booleans, delta: decoded whole, copied block by block
        plan.of_col[p.ctx].append(i)
    return plan^


struct _GroupReader(Movable):
    """Rows [r0, r1) of a row group (a row-range morsel), decoded a block of `block` rows at a time by the thread that
    owns it: `next_block` decodes the next rows of each column read into the caller's table, pages one after the other
    (a page is mapped, decompressed and its levels unpacked when its first block is wanted, and dropped when its last
    block has been), so the working set is a block and the page being read per column. Only dictionary-indexed and
    PLAIN fixed-width pages are cut into blocks; a page of another encoding (text, booleans, delta) is decoded whole when
    its first block is wanted and copied block by block. The reader has its own copy of the page descriptors and shares
    the plan's dictionaries (read only); nothing here uses the thread pool. The plan must outlive the reader: that is why the plan and the reader are internal (underscore), used by the scan loops of
    query.mojo, which keep the plan alive across every reader."""
    var plan: Int                    # the address of the `_GroupPlan`
    var scan: ParquetScan
    var cols: List[Int]
    var pages: List[_Page]
    var of_col: List[List[Int]]
    var page_maps: List[_Mapped]     # per page: its bytes, mapped while its blocks are read (the resident set is a page)
    var cur: List[Int]               # per column read: the page of `of_col` being read
    var block: Int
    var at: Int                      # the first row of the next block
    var stop: Int                    # and the end of the range

    def __init__(out self, plan: _GroupPlan, r0: Int, r1: Int, block: Int):
        self.plan = Int(UnsafePointer(to=plan))
        self.scan = plan.scan.copy()
        self.cols = plan.cols.copy()
        self.pages = plan.pages.copy()
        self.of_col = plan.of_col.copy()
        self.page_maps = List[_Mapped]()
        for _ in range(len(plan.pages)):
            self.page_maps.append(_Mapped())
        self.cur = List[Int](length=len(plan.cols), fill=0)
        self.block = block
        self.at = r0
        self.stop = min(r1, plan.rows)

    def next_block(mut self, mut t: Table) raises -> Int:
        """Decodes the next block into rows 0.. of `t`'s columns (in the order of `cols`; each holds at least `block`
        values), sets their validity bitmaps and null counts, and returns the number of rows (0 at the end)."""
        if self.at >= self.stop:
            return 0
        var b0 = self.at
        var b1 = min(self.stop, b0 + self.block)
        ref info = self.scan._info[]
        ref plan = UnsafePointer[_GroupPlan, MutAnyOrigin](unsafe_from_address=self.plan)[]
        if info.is_csv:
            for k in range(len(self.cols)):
                ref col = t.cols[k]
                ref src = plan.decoded.cols[k]
                for j in range(b0, b1):
                    col.values[j - b0] = src.values[j]
                var nulls = 0
                if len(src.validity) > 0:
                    col.validity.clear()
                    col.validity.resize((self.block + 7) // 8, 0xFF)
                    var vptr = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(col.validity.unsafe_ptr()))
                    _copy_bits(vptr, 0, UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(src.validity.unsafe_ptr())), b0, b1 - b0)
                    for j in range(b0, b1):
                        if src.is_null(j):
                            nulls += 1
                if nulls == 0:
                    col.validity.clear()
                col.nulls = nulls
            self.at = b1
            return b1 - b0
        for k in range(len(self.cols)):
            ref col = t.cols[k]
            ref ctx = plan.ctxs[k]
            ref leaf = info.leaves[self.cols[k]]
            if ctx.optional:
                col.validity.clear()
                col.validity.resize((self.block + 7) // 8, 0xFF)
            var vptr = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(col.validity.unsafe_ptr()))
            var base = Int(col.values.unsafe_ptr())
            var nulls = 0
            var ci = self.cur[k]
            while ci < len(self.of_col[k]):
                ref p = self.pages[self.of_col[k][ci]]
                if p.row >= b1:
                    break
                if p.nvals == 0 or p.row + p.nvals <= b0:
                    ci += 1
                    continue
                if p.csize > 0 and self.page_maps[self.of_col[k][ci]].span == 0:      # first block of the page: map its bytes
                    self.page_maps[self.of_col[k][ci]] = _Mapped(info.file._get_raw_fd(), p.foff, p.csize, info.path)
                    p.addr = self.page_maps[self.of_col[k][ci]].data
                if p.whole:
                    if len(p.values) == 0:               # first block of this page: decode all of it
                        p.values = List[Int64](unsafe_uninit_length=p.nvals)
                        _decode_page(p, ctx, leaf, Int(p.values.unsafe_ptr()), 0)
                    var a0 = max(b0, p.row) - p.row
                    var a1 = min(b1, p.row + p.nvals) - p.row
                    for j in range(a0, a1):
                        col.values[p.row + j - b0] = p.values[j]
                    if p.nulls > 0:
                        var bp = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(p.bits.unsafe_ptr()))
                        _copy_bits(vptr, p.row + a0 - b0, bp, a0, a1 - a0)
                        for j in range(a0, a1):
                            if not _get_bit(bp, j):
                                nulls += 1
                    if p.row + p.nvals <= b1:            # the page's last block
                        p.values = List[Int64]()
                        p.bits = List[UInt8]()
                        self.page_maps[self.of_col[k][ci]] = _Mapped()
                        ci += 1
                        continue
                    break
                if len(p.sl.v) == 0:                     # first block of this page: open it and plan a slice per block
                    var cuts = List[Int]()
                    var g = (p.row // self.block + 1) * self.block
                    while g < p.row + p.nvals:
                        cuts.append(g - p.row)
                        g += self.block
                    _prep_page(p, ctx, leaf, cuts)
                var sidx = p.sl.next
                while sidx < len(p.sl.row) and p.row + p.sl.row[sidx] < max(b0, p.row):      # (a range that starts mid-page)
                    sidx += 1
                if sidx >= len(p.sl.row) or p.row + p.sl.row[sidx] != max(b0, p.row):
                    raise Error("internal error: a block does not start at a slice")
                _decode_slice(p, sidx, ctx, leaf, base + (p.row - b0) * 8)
                var r0 = p.sl.row[sidx]
                var r1 = p.sl.row[sidx + 1] if sidx + 1 < len(p.sl.row) else p.nvals
                var cnt = (p.sl.v[sidx + 1] if sidx + 1 < len(p.sl.v) else p.present) - p.sl.v[sidx]
                if r1 - r0 > cnt:
                    nulls += r1 - r0 - cnt
                    _copy_bits(vptr, p.row + r0 - b0, UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(p.bits.unsafe_ptr())),
                               r0, r1 - r0)
                p.sl.next = sidx + 1
                if p.sl.next >= len(p.sl.v):             # the page's last block: let go of it
                    p.keep = List[UInt8]()
                    p.bits = List[UInt8]()
                    self.page_maps[self.of_col[k][ci]] = _Mapped()
                    ci += 1
                else:
                    break
            self.cur[k] = ci
            if ctx.optional and nulls == 0:
                col.validity.clear()
            col.nulls = nulls
        self.at = b1
        return b1 - b0


def scan_parquet(path: String) raises -> ParquetScan:
    """Opens a Parquet file for streaming (`Query(scan)`): reads the footer, no column data and no dictionary. Peak
    memory of a query over it is one row group's columns (plus the dictionaries of the TEXT columns it reads)."""
    var file = _open_file(path)
    var size = _file_size(file, path)
    var fd = file._get_raw_fd()
    if size < 12:
        raise Error("`" + path + "` is not a Parquet file (no PAR1 magic)")
    var head = _Mapped(fd, 0, 4, path)
    var tail = _Mapped(fd, size - 8, 8, path)
    var hp = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=head.data)
    var tp = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=tail.data)
    if hp[0] != 0x50 or hp[1] != 0x41 or hp[2] != 0x52 or hp[3] != 0x31 or \
       tp[4] != 0x50 or tp[5] != 0x41 or tp[6] != 0x52 or tp[7] != 0x31:
        raise Error("`" + path + "` is not a Parquet file (no PAR1 magic)")
    var flen = Int(tp[0]) | (Int(tp[1]) << 8) | (Int(tp[2]) << 16) | (Int(tp[3]) << 24)
    head.release()
    tail.release()
    if flen <= 0 or flen > size - 12:
        raise Error("`" + path + "`: the Parquet footer length is wrong (corrupt file)")
    var fm = _Mapped(fd, size - 8 - flen, flen, path)
    var r = ThriftReader(fm.data, flen)
    var footer = _read_footer(r)
    fm.release()
    var cols = List[_Leaf]()
    var dicts = _schema_table(path, footer.leaves, cols)
    var unused = List[_Leaf]()
    var schema = _schema_table(path, footer.leaves, unused)
    var n_cols = len(cols)
    var groups = List[List[_ChunkMeta]]()
    var rows_of = List[Int]()
    for g in range(len(footer.groups)):
        if len(footer.groups[g]) != n_cols:
            raise Error("`" + path + "`: row group " + String(g) + " has " + String(len(footer.groups[g])) +
                        " columns, the schema " + String(n_cols))
        rows_of.append(footer.groups[g][0].values if n_cols > 0 else 0)
        groups.append(footer.groups[g].copy())
    var info = _ScanInfo(path, file^, size, cols^, groups^, rows_of^, footer.omx.copy(), dicts^)
    info.extra = footer.extra.copy()
    return ParquetScan(info^, schema^.share())


comptime CsvScan = ParquetScan           # F-178: `scan_csv` gives the same type: every ParquetScan call and every Query / SQL / Frame source works


def scan_csv(path: String, options: CsvOptions = CsvOptions()) raises -> ParquetScan:
    """Opens a CSV or TSV file for streaming (csvscan.mojo): the header and the kinds from its first rows (the rules of `read_csv`), one pass
    over the file that finds its row groups and fills `scan.csv_report()`, no column data kept. A query, `Catalog.add_scan`, `Frame(scan)`
    and `save` then work as on a Parquet scan, one row group (`options.group_rows` rows) at a time."""
    var file = _open_file(path)
    var fd = file._get_raw_fd()
    var one = List[UInt8](length=1, fill=0)
    var got = external_call["read", Int](fd, one.unsafe_ptr(), Int(1))
    _ = one^
    if got < 0:                                                       # a directory opens, but cannot be read
        raise Error("couldn't read `" + path + "`: not a regular file")
    var size = _file_size(file, path)
    var src = csv_open(fd, size, path, options)
    var leaves = List[_Leaf]()
    var schema = Table.empty()
    var dicts = Table.empty()
    for c in range(len(src.names)):
        var k = ColumnKind(src.kinds[c])
        var total = 0
        for g in range(len(src.rows)):
            total += src.nulls[g][c]
        var ptype = PT_BYTE_ARRAY if k == TEXT else PT_DOUBLE if k == FLOAT else PT_INT64
        leaves.append(_Leaf(src.names[c], ptype, 0, OPTIONAL if total > 0 else REQUIRED, -1, src.scales[c], 0, -1, 0, 0, False))
        _ = schema.add_column(src.names[c], k, scale=src.scales[c])
        _ = dicts.add_column(src.names[c], k, scale=src.scales[c])
    var groups = List[List[_ChunkMeta]]()
    var rows_of = List[Int]()
    for g in range(len(src.rows)):
        var metas = List[_ChunkMeta]()
        for c in range(len(src.names)):
            metas.append(_ChunkMeta(leaves[c].ptype, UNCOMPRESSED, src.rows[g], 0, 0, 0, src.rows[g], src.nulls[g][c], 0, 0, 0, 0, 0))
        groups.append(metas^)
        rows_of.append(src.rows[g])
    var info = _ScanInfo(path, file^, size, leaves^, groups^, rows_of^, String(), dicts^)
    info.is_csv = True
    info.csv = src^
    return ParquetScan(info^, schema^.share())

