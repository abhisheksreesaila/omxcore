"""zstd and snappy in Mojo (F-180): the two Parquet codecs for builds with no C library to call.

An Apple app links neither libzstd nor libsnappy, and the app ships nothing but its own binary. These are the same
two formats written in Mojo, so a file saved on a Mac is compressed like one saved on Linux and each reads the
other's. Nothing is vendored: the code is written from the format descriptions (RFC 8878 for zstd, the snappy
`format_description.txt`) and tested against libzstd / libsnappy, pyarrow and DuckDB (tests/test_codec_mojo.py,
mojo/tests/test_codec.mojo). No symbol from outside the package is used, so the Mac link check stays as it was.

snappy: raw format (what Parquet stores), a greedy matcher over a hash of four bytes.
zstd, reading: the whole format but dictionaries (all block and literal types, Huffman tables with direct or FSE coded
weights, the three FSE modes and repeat tables, repeat offsets, skippable and concatenated frames). The content
checksum, when a frame has one, is verified (xxHash64). Offsets reach back no further than the frame's start and
its declared window. The input is untrusted: every length is checked against
the input and the stated output size before anything is written.
zstd, writing: one frame, 128 KB blocks, a 1 MB window. The matcher tries the last offset one byte on first, then a
table of four-byte keys (4 candidates per bucket, the key stored beside the position), with a one-step lazy choice
priced in bits; literals are Huffman coded (1 or 4 streams, weights direct or FSE coded) when that is smaller;
sequences are coded with the predefined, a repeated-symbol or a purpose-built FSE table, whichever costs fewer bits,
and with repeat offsets; a block that does not shrink is stored, a block of one byte repeated is a run. It does not
reuse tables between blocks. On the tables omxcore saves, files come out the size of libzstd level 3's or smaller,
at about a third of its speed (mojo/README.md: Build switches).
"""
from std.bit import count_trailing_zeros, count_leading_zeros

comptime Ptr = UnsafePointer[UInt8, MutAnyOrigin]


def _ptr(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


@always_inline
def _load32(p: Ptr, i: Int) -> UInt32:
    return (p + i).bitcast[UInt32]().load[alignment=1]()


@always_inline
def _load64(p: Ptr, i: Int) -> UInt64:
    return (p + i).bitcast[UInt64]().load[alignment=1]()


# ── snappy (raw format, as Parquet stores it) ─────────────────────────────────────────────────────────

comptime SNAPPY_HASH_BITS = 14
comptime SNAPPY_WINDOW = 65535          # copies with a 2-byte offset reach this far back


def snappy_max_compressed_length(n: Int) -> Int:
    return 32 + n + n // 6


def _snappy_literal(dst: Ptr, o_in: Int, src: Ptr, start: Int, length: Int) -> Int:
    var o = o_in
    var m = length - 1
    if m < 60:
        dst[o] = UInt8(m << 2)
        o += 1
    else:
        var nb = 1
        while nb < 4 and (m >> (8 * nb)) != 0:
            nb += 1
        dst[o] = UInt8((59 + nb) << 2)
        o += 1
        for k in range(nb):
            dst[o] = UInt8((m >> (8 * k)) & 0xFF)
            o += 1
    for k in range(length):
        dst[o + k] = src[start + k]
    return o + length


def _snappy_copy(dst: Ptr, o_in: Int, offset: Int, length_in: Int) -> Int:
    var o = o_in
    var length = length_in
    while length >= 68:                       # copies of 64, so the rest is at least 4
        dst[o] = UInt8((63 << 2) | 2)
        dst[o + 1] = UInt8(offset & 0xFF)
        dst[o + 2] = UInt8(offset >> 8)
        o += 3
        length -= 64
    if length > 64:
        dst[o] = UInt8((59 << 2) | 2)
        dst[o + 1] = UInt8(offset & 0xFF)
        dst[o + 2] = UInt8(offset >> 8)
        o += 3
        length -= 60
    if length >= 12 or offset >= 2048:
        dst[o] = UInt8(((length - 1) << 2) | 2)
        dst[o + 1] = UInt8(offset & 0xFF)
        dst[o + 2] = UInt8(offset >> 8)
        o += 3
    else:
        dst[o] = UInt8(((offset >> 8) << 5) | ((length - 4) << 2) | 1)
        dst[o + 1] = UInt8(offset & 0xFF)
        o += 2
    return o


def snappy_compress(src: List[UInt8]) raises -> List[UInt8]:
    """Snappy's raw format: the length as a varint, then literals and copies. A greedy matcher over a hash of four
    bytes, skipping faster the longer it finds nothing (as snappy does)."""
    var n = len(src)
    if n > 2147483647:
        raise Error("snappy: input too large")
    var out = List[UInt8](unsafe_uninit_length=snappy_max_compressed_length(n))
    var op = _ptr(Int(out.unsafe_ptr()))
    var o = 0
    var v = n
    while v >= 0x80:
        op[o] = UInt8((v & 0x7F) | 0x80)
        o += 1
        v >>= 7
    op[o] = UInt8(v)
    o += 1
    var sp = _ptr(Int(src.unsafe_ptr()))
    var table = List[Int32](length=1 << SNAPPY_HASH_BITS, fill=0)      # position + 1, 0 for none (a zero fill is a plain memset)
    var tp = table.unsafe_ptr()
    var i = 0
    var lit = 0                                  # start of the pending literal
    var limit = n - 4                            # a match needs four bytes to load
    var skip = 32
    while i <= limit:
        var cur = _load32(sp, i)
        var h = Int((cur * UInt32(0x1E35A7BD)) >> UInt32(32 - SNAPPY_HASH_BITS))
        var cand = Int(tp[h]) - 1
        tp[h] = Int32(i + 1)
        if cand >= 0 and i - cand <= SNAPPY_WINDOW and _load32(sp, cand) == cur:
            if i > lit:
                o = _snappy_literal(op, o, sp, lit, i - lit)
            var len = _match_len(sp, cand, i, n)
            o = _snappy_copy(op, o, i - cand, len)
            i += len
            lit = i
            skip = 32
            if i >= 1 and i - 1 <= limit:        # index the byte before the next position, as snappy does
                var c2 = _load32(sp, i - 1)
                tp[Int((c2 * UInt32(0x1E35A7BD)) >> UInt32(32 - SNAPPY_HASH_BITS))] = Int32(i)
        else:
            i += skip >> 5
            skip += 1
    if n > lit:
        o = _snappy_literal(op, o, sp, lit, n - lit)
    _ = table^
    out.resize(o, 0)
    return out^


def snappy_decompress(src_addr: Int, n: Int, size: Int) raises -> List[UInt8]:
    """`n` snappy bytes at `src_addr` into exactly `size` bytes; anything else is a corrupt page."""
    var sp = _ptr(src_addr)
    var i = 0
    var stated = 0
    var shift = 0
    while True:
        if i >= n or shift > 28:
            raise Error("snappy: bad length")
        var b = Int(sp[i])
        i += 1
        stated |= (b & 0x7F) << shift
        shift += 7
        if b < 0x80:
            break
    if stated != size:
        raise Error("snappy: wrong length " + String(stated) + " " + String(size) + " " + String(n))
    var out = List[UInt8](unsafe_uninit_length=max(size, 1))
    var op = _ptr(Int(out.unsafe_ptr()))
    var o = 0
    while i < n:
        var tag = Int(sp[i])
        i += 1
        var kind = tag & 3
        if kind == 0:
            var len = (tag >> 2) + 1
            if len > 60:
                var nb = len - 60
                if i + nb > n:
                    raise Error("snappy: truncated")
                len = 1
                for k in range(nb):
                    len += Int(sp[i + k]) << (8 * k)
                i += nb
            if len > n - i or len > size - o:
                raise Error("snappy: literal too long")
            for k in range(len):
                op[o + k] = sp[i + k]
            i += len
            o += len
        else:
            var len = 0
            var offset = 0
            if kind == 1:
                if i + 1 > n:
                    raise Error("snappy: truncated")
                len = 4 + ((tag >> 2) & 7)
                offset = ((tag >> 5) << 8) | Int(sp[i])
                i += 1
            elif kind == 2:
                if i + 2 > n:
                    raise Error("snappy: truncated")
                len = (tag >> 2) + 1
                offset = Int(sp[i]) | (Int(sp[i + 1]) << 8)
                i += 2
            else:
                if i + 4 > n:
                    raise Error("snappy: truncated")
                len = (tag >> 2) + 1
                offset = Int(sp[i]) | (Int(sp[i + 1]) << 8) | (Int(sp[i + 2]) << 16) | (Int(sp[i + 3]) << 24)
                i += 4
            if offset == 0 or offset > o or len > size - o:
                raise Error("snappy: bad copy")
            for k in range(len):                 # byte by byte: a copy may overlap what it writes
                op[o + k] = op[o - offset + k]
            o += len
    if o != size:
        raise Error("snappy: short output")
    out.resize(size, 0)
    return out^


def snappy_decompress(src: List[UInt8], size: Int) raises -> List[UInt8]:
    return snappy_decompress(Int(src.unsafe_ptr()), len(src), size)


# ── zstd (RFC 8878) ───────────────────────────────────────────────────────────────────────────────────
# Tables the format fixes: how a sequence code becomes a length (baseline plus extra bits).

def _ll_default() -> List[Int]:
    return [4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1]


def _ml_default() -> List[Int]:
    return [1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1]


def _of_default() -> List[Int]:
    return [1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]


def _ll_bits() -> List[Int]:
    return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def _ml_bits() -> List[Int]:
    return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def _bases(bits: List[Int], first: Int) -> List[Int]:
    """Each code's baseline: the first code's value, then every code starts where the one before ends."""
    var b = List[Int]()
    var v = first
    for k in bits:
        b.append(v)
        v += 1 << k
    return b^


comptime LL_DEFAULT_LOG = 6
comptime ML_DEFAULT_LOG = 6
comptime OF_DEFAULT_LOG = 5
comptime LL_MAX_LOG = 9
comptime ML_MAX_LOG = 9
comptime OF_MAX_LOG = 8
comptime ZSTD_BLOCK_MAX = 1 << 17
comptime HUF_MAX_BITS = 11


@always_inline
def _highbit(v: Int) -> Int:
    """Index of the highest set bit (v > 0)."""
    return 63 - Int(count_leading_zeros(UInt64(v)))


@fieldwise_init
struct _Fse(Copyable, Movable):
    """An FSE decoding table: for each state its symbol, the bits to read and the baseline of the next state."""
    var log: Int
    var sym: List[UInt8]
    var nbits: List[UInt8]
    var base: List[UInt16]


def _fse_spread(norm: List[Int], log: Int) raises -> List[Int]:
    """The symbol of each table position: symbols with probability < 1 at the end, the rest spread by a fixed step."""
    var size = 1 << log
    var table = List[Int](length=size, fill=0)
    var high = size - 1
    for s in range(len(norm)):
        if norm[s] == -1:
            table[high] = s
            high -= 1
    var step = (size >> 1) + (size >> 3) + 3
    var mask = size - 1
    var pos = 0
    for s in range(len(norm)):
        var c = norm[s]
        for _ in range(c if c > 0 else 0):
            table[pos] = s
            pos = (pos + step) & mask
            while pos > high:
                pos = (pos + step) & mask
    if pos != 0:
        raise Error("zstd: bad FSE table")
    return table^


def _fse_build(norm: List[Int], log: Int) raises -> _Fse:
    var size = 1 << log
    var total = 0
    for c in norm:
        total += 1 if c == -1 else c
    if total != size or log > 15:
        raise Error("zstd: bad FSE table")
    var spread = _fse_spread(norm, log)
    var t = _Fse(log, List[UInt8](length=size, fill=0), List[UInt8](length=size, fill=0), List[UInt16](length=size, fill=0))
    var next = List[Int](length=len(norm), fill=0)
    for s in range(len(norm)):
        next[s] = 1 if norm[s] == -1 else norm[s]
    for u in range(size):
        var s = spread[u]
        var x = next[s]
        next[s] = x + 1
        var nb = log - _highbit(x)
        t.sym[u] = UInt8(s)
        t.nbits[u] = UInt8(nb)
        t.base[u] = UInt16((x << nb) - size)
    return t^


def _fse_single(sym: Int) -> _Fse:
    """RLE mode: one state, always `sym`, no bits."""
    return _Fse(0, [UInt8(sym)], [UInt8(0)], [UInt16(0)])


struct _FwdBits:
    """Bits read from the low end of each byte upward (the FSE table descriptions)."""
    var a: Int
    var n: Int           # bytes
    var pos: Int         # bits consumed

    def __init__(out self, p: Ptr, n: Int):
        self.a = Int(p)
        self.n = n
        self.pos = 0

    def peek(self, k: Int) -> Int:
        var byte = self.pos >> 3
        var v = 0
        for j in range(8):
            if byte + j < self.n:
                v |= Int(_ptr(self.a)[byte + j]) << (8 * j)
        return (v >> (self.pos & 7)) & ((1 << k) - 1)

    def skip(mut self, k: Int):
        self.pos += k

    def bytes_used(self) raises -> Int:
        var used = (self.pos + 7) >> 3
        if used > self.n:
            raise Error("zstd: table description runs past its data")
        return used


def _read_ncount(p: Ptr, n: Int, max_symbol: Int, max_log: Int, mut norm: List[Int], mut used: Int) raises -> Int:
    """Reads an FSE table description: the normalized counts into `norm`, the bytes it took into `used`; returns the
    accuracy log."""
    var r = _FwdBits(p, n)
    var al = 5 + r.peek(4)
    r.skip(4)
    if al > max_log:
        raise Error("zstd: FSE accuracy too large")
    var remaining = (1 << al) + 1
    var threshold = 1 << al
    var nbits = al + 1
    var sym = 0
    var prev0 = False
    norm.clear()
    while remaining > 1 and sym <= max_symbol:
        if prev0:
            var n0 = sym
            while r.peek(16) == 0xFFFF:
                n0 += 24
                r.skip(16)
                if r.pos > 8 * n:
                    raise Error("zstd: table description runs past its data")
            while r.peek(2) == 3:
                n0 += 3
                r.skip(2)
                if r.pos > 8 * n:
                    raise Error("zstd: table description runs past its data")
            n0 += r.peek(2)
            r.skip(2)
            if n0 > max_symbol + 1:
                raise Error("zstd: bad FSE table")
            while sym < n0:
                norm.append(0)
                sym += 1
            if sym > max_symbol:
                break
        var mx = (2 * threshold - 1) - remaining
        var count = 0
        var low = r.peek(nbits) & (threshold - 1)
        if low < mx:
            count = low
            r.skip(nbits - 1)
        else:
            count = r.peek(nbits) & (2 * threshold - 1)
            if count >= threshold:
                count -= mx
            r.skip(nbits)
        count -= 1
        remaining -= -count if count < 0 else count
        norm.append(count)
        sym += 1
        prev0 = count == 0
        if remaining < 1:
            raise Error("zstd: bad FSE table")
        while remaining < threshold:
            nbits -= 1
            threshold >>= 1
        if r.pos > 8 * n + 16:
            raise Error("zstd: table description runs past its data")
    if remaining != 1:
        raise Error("zstd: bad FSE table")
    used = r.bytes_used()
    return al


struct _RevBits:
    """Bits read from the high end of a stream backwards (the last byte holds a 1 marking where the data ends)."""
    var a: Int
    var n: Int
    var pos: Int         # data bits still to read
    var over: Bool

    def __init__(out self, p: Ptr, n: Int) raises:
        self.a = Int(p)
        self.n = n
        self.over = False
        if n < 1 or _ptr(self.a)[n - 1] == 0:
            raise Error("zstd: bad bit stream")
        self.pos = 8 * (n - 1) + _highbit(Int(_ptr(self.a)[n - 1]))

    @always_inline
    def _word(self, byte: Int) -> UInt64:
        if byte + 8 <= self.n:
            return _load64(_ptr(self.a), byte)
        var v = UInt64(0)
        for j in range(8):
            if byte + j < self.n:
                v |= UInt64(_ptr(self.a)[byte + j]) << UInt64(8 * j)
        return v

    @always_inline
    def peek(self, k: Int) -> Int:
        """The next k (<= 32) bits; bits below the start of the stream read as zero."""
        if k == 0:
            return 0
        var at = self.pos - k
        if at >= 0:
            return Int((self._word(at >> 3) >> UInt64(at & 7)) & ((UInt64(1) << UInt64(k)) - 1))
        var have = k + at                            # bits that really exist, the low `have` of the value's high part
        if have <= 0:
            return 0
        return Int(self._word(0) & ((UInt64(1) << UInt64(have)) - 1)) << (-at)

    @always_inline
    def read(mut self, k: Int) -> Int:
        var v = self.peek(k)
        self.pos -= k
        if self.pos < 0:
            self.over = True
        return v

    @always_inline
    def consume(mut self, k: Int):
        self.pos -= k
        if self.pos < 0:
            self.over = True


@fieldwise_init
struct _Huf(Copyable, Movable):
    """A Huffman decoding table of 1 << bits entries: the symbol and the code length of whatever bits start there."""
    var bits: Int
    var sym: List[UInt8]
    var nb: List[UInt8]


def _huf_build(mut weights: List[Int]) raises -> _Huf:
    """`weights` holds every symbol's weight but the last, whose weight is implied by the total."""
    var total = 0
    for w in weights:
        if w > 0:
            total += 1 << (w - 1)
    if total == 0:
        raise Error("zstd: bad Huffman weights")
    var bits = _highbit(total) + 1
    if bits > 12:
        raise Error("zstd: bad Huffman weights")
    var rest = (1 << bits) - total
    if rest & (rest - 1) != 0:
        raise Error("zstd: bad Huffman weights")
    weights.append(_highbit(rest) + 1)
    var size = 1 << bits
    var t = _Huf(bits, List[UInt8](length=size, fill=0), List[UInt8](length=size, fill=0))
    var start = List[Int](length=14, fill=0)           # first table slot of each weight (lowest weight first)
    var count = List[Int](length=14, fill=0)
    for w in weights:
        count[w] += 1
    var next = 0
    for w in range(1, 14):
        start[w] = next
        next += count[w] << (w - 1)
    for s in range(len(weights)):
        var w = weights[s]
        if w == 0:
            continue
        var span = 1 << (w - 1)
        var at = start[w]
        for k in range(span):
            t.sym[at + k] = UInt8(s)
            t.nb[at + k] = UInt8(bits + 1 - w)
        start[w] = at + span
    return t^


def _fse_decode_weights(p: Ptr, n: Int, mut weights: List[Int]) raises:
    """Huffman weights stored as an FSE stream with two interleaved states."""
    var norm = List[Int]()
    var used = 0
    var al = _read_ncount(p, n, 255, 6, norm, used)
    var t = _fse_build(norm, al)
    var rd = _RevBits(p + used, n - used)
    var s1 = rd.read(al)
    var s2 = rd.read(al)
    if rd.over:
        raise Error("zstd: bad Huffman weights")
    while True:
        weights.append(Int(t.sym[s1]))
        s1 = Int(t.base[s1]) + rd.read(Int(t.nbits[s1]))
        if rd.over:
            weights.append(Int(t.sym[s2]))
            break
        weights.append(Int(t.sym[s2]))
        s2 = Int(t.base[s2]) + rd.read(Int(t.nbits[s2]))
        if rd.over:
            weights.append(Int(t.sym[s1]))
            break
        if len(weights) > 255:
            raise Error("zstd: bad Huffman weights")
    if len(weights) > 255:
        raise Error("zstd: bad Huffman weights")
    for w in weights:
        if w > 12:
            raise Error("zstd: bad Huffman weights")


def _huf_read_tree(p: Ptr, n: Int, mut used: Int) raises -> _Huf:
    """The tree description at the start of a compressed literals section; `used` is its length."""
    if n < 1:
        raise Error("zstd: truncated")
    var hdr = Int(p[0])
    var weights = List[Int]()
    if hdr >= 128:
        var nw = hdr - 127
        var nbytes = (nw + 1) // 2
        if 1 + nbytes > n:
            raise Error("zstd: truncated")
        for i in range(nw):
            var b = Int(p[1 + i // 2])
            weights.append((b >> 4) if i % 2 == 0 else (b & 15))
        used = 1 + nbytes
    else:
        if hdr == 0 or 1 + hdr > n:
            raise Error("zstd: bad Huffman header")
        _fse_decode_weights(p + 1, hdr, weights)
        used = 1 + hdr
    return _huf_build(weights)


def _huf_decode_stream(t: _Huf, p: Ptr, n: Int, dst: Ptr, count: Int) raises:
    var rd = _RevBits(p, n)
    var bits = t.bits
    var sp = t.sym.unsafe_ptr()
    var nbp = t.nb.unsafe_ptr()
    for i in range(count):
        var e = rd.peek(bits)
        dst[i] = sp[e]
        rd.consume(Int(nbp[e]))
    if rd.over or rd.pos != 0:
        raise Error("zstd: bad Huffman stream")


struct _ZstdState:
    """What a frame carries from block to block: the last Huffman table, the last sequence tables, the repeat offsets."""
    var huf: _Huf
    var have_huf: Bool
    var ll: _Fse
    var of: _Fse
    var ml: _Fse
    var have_seq: Bool
    var rep1: Int
    var rep2: Int
    var rep3: Int
    var ll_bits: List[Int]
    var ll_base: List[Int]
    var ml_bits: List[Int]
    var ml_base: List[Int]

    def __init__(out self) raises:
        self.ll_bits = _ll_bits()
        self.ll_base = _bases(self.ll_bits, 0)
        self.ml_bits = _ml_bits()
        self.ml_base = _bases(self.ml_bits, 3)
        self.huf = _Huf(0, List[UInt8](), List[UInt8]())
        self.have_huf = False
        self.ll = _fse_build(_ll_default(), LL_DEFAULT_LOG)
        self.of = _fse_build(_of_default(), OF_DEFAULT_LOG)
        self.ml = _fse_build(_ml_default(), ML_DEFAULT_LOG)
        self.have_seq = False
        self.rep1 = 1
        self.rep2 = 4
        self.rep3 = 8


def _read_literals(mut st: _ZstdState, p: Ptr, n: Int, mut lits: List[UInt8], mut used: Int) raises:
    """The literals section of a compressed block into `lits`; `used` is its length in the block."""
    if n < 1:
        raise Error("zstd: truncated")
    var b0 = Int(p[0])
    var kind = b0 & 3
    var fmt = (b0 >> 2) & 3
    if kind <= 1:
        var hdr = 1
        var size = 0
        if fmt == 0 or fmt == 2:
            size = b0 >> 3
        elif fmt == 1:
            if n < 2:
                raise Error("zstd: truncated")
            size = (b0 >> 4) | (Int(p[1]) << 4)
            hdr = 2
        else:
            if n < 3:
                raise Error("zstd: truncated")
            size = (b0 >> 4) | (Int(p[1]) << 4) | (Int(p[2]) << 12)
            hdr = 3
        if size > ZSTD_BLOCK_MAX:
            raise Error("zstd: literals too long")
        lits = List[UInt8](unsafe_uninit_length=size)
        if kind == 0:
            if hdr + size > n:
                raise Error("zstd: truncated")
            for i in range(size):
                lits[i] = p[hdr + i]
            used = hdr + size
        else:
            if hdr + 1 > n:
                raise Error("zstd: truncated")
            var v = p[hdr]
            for i in range(size):
                lits[i] = v
            used = hdr + 1
        return
    var hdr = 3
    var regen = 0
    var comp = 0
    var streams = 4
    if fmt == 0 or fmt == 1:
        if n < 3:
            raise Error("zstd: truncated")
        var v = b0 | (Int(p[1]) << 8) | (Int(p[2]) << 16)
        regen = (v >> 4) & 1023
        comp = (v >> 14) & 1023
        streams = 1 if fmt == 0 else 4
    elif fmt == 2:
        if n < 4:
            raise Error("zstd: truncated")
        var v = b0 | (Int(p[1]) << 8) | (Int(p[2]) << 16) | (Int(p[3]) << 24)
        regen = (v >> 4) & 16383
        comp = (v >> 18) & 16383
        hdr = 4
    else:
        if n < 5:
            raise Error("zstd: truncated")
        var v = b0 | (Int(p[1]) << 8) | (Int(p[2]) << 16) | (Int(p[3]) << 24) | (Int(p[4]) << 32)
        regen = (v >> 4) & 262143
        comp = (v >> 22) & 262143
        hdr = 5
    if regen > ZSTD_BLOCK_MAX or hdr + comp > n:
        raise Error("zstd: bad literals section")
    var body = p + hdr
    var blen = comp
    if kind == 2:
        var tlen = 0
        st.huf = _huf_read_tree(body, blen, tlen)
        st.have_huf = True
        body = body + tlen
        blen -= tlen
    elif not st.have_huf:
        raise Error("zstd: literals reuse a Huffman table that was never sent")
    if blen < 1:
        raise Error("zstd: truncated")
    lits = List[UInt8](unsafe_uninit_length=regen)
    var out = _ptr(Int(lits.unsafe_ptr()))
    if streams == 1:
        _huf_decode_stream(st.huf, body, blen, out, regen)
    else:
        if blen < 6 or regen < 4:
            raise Error("zstd: bad literals section")
        var s1 = Int(body[0]) | (Int(body[1]) << 8)
        var s2 = Int(body[2]) | (Int(body[3]) << 8)
        var s3 = Int(body[4]) | (Int(body[5]) << 8)
        var s4 = blen - 6 - s1 - s2 - s3
        if s4 < 1:
            raise Error("zstd: bad literals section")
        var q = (regen + 3) // 4
        var rest = regen - 3 * q
        if rest < 0:
            raise Error("zstd: bad literals section")
        var at = body + 6
        _huf_decode_stream(st.huf, at, s1, out, q)
        _huf_decode_stream(st.huf, at + s1, s2, out + q, q)
        _huf_decode_stream(st.huf, at + s1 + s2, s3, out + 2 * q, q)
        _huf_decode_stream(st.huf, at + s1 + s2 + s3, s4, out + 3 * q, rest)
    used = hdr + comp


@always_inline
def _copy_match(op: Ptr, o: Int, offset: Int, length: Int):
    var from_ = o - offset
    if offset >= 8:
        var k = 0
        while k + 8 <= length:
            (op + o + k).bitcast[UInt64]().store[alignment=1]((op + from_ + k).bitcast[UInt64]().load[alignment=1]())
            k += 8
        while k < length:
            op[o + k] = op[from_ + k]
            k += 1
    else:
        for k in range(length):
            op[o + k] = op[from_ + k]


def _seq_table(mode: Int, mut cur: _Fse, default_norm: List[Int], default_log: Int, max_symbol: Int, max_log: Int,
               p: Ptr, n: Int, mut used: Int, have_prev: Bool) raises:
    """One of the three sequence tables, as its mode byte says; `used` is how many bytes of the block it took."""
    used = 0
    if mode == 0:
        cur = _fse_build(default_norm, default_log)
    elif mode == 1:
        if n < 1:
            raise Error("zstd: truncated")
        if Int(p[0]) > max_symbol:
            raise Error("zstd: bad sequence symbol")
        cur = _fse_single(Int(p[0]))
        used = 1
    elif mode == 2:
        var norm = List[Int]()
        var al = _read_ncount(p, n, max_symbol, max_log, norm, used)
        cur = _fse_build(norm, al)
    else:
        if not have_prev:
            raise Error("zstd: sequences reuse a table that was never sent")


def _decode_block(mut st: _ZstdState, p: Ptr, n: Int, op: Ptr, o_in: Int, size: Int, frame_start: Int, window: Int) raises -> Int:
    """A compressed block's bytes at p into the output at op from o_in; returns the new output position."""
    var lits = List[UInt8]()
    var used = 0
    _read_literals(st, p, n, lits, used)
    var i = used
    var nlit = len(lits)
    var lp = _ptr(Int(lits.unsafe_ptr()))
    var o = o_in
    if i >= n:
        raise Error("zstd: truncated")
    var nseq = Int(p[i])
    i += 1
    if nseq == 0:
        pass
    elif nseq < 128:
        pass
    elif nseq < 255:
        if i >= n:
            raise Error("zstd: truncated")
        nseq = ((nseq - 128) << 8) + Int(p[i])
        i += 1
    else:
        if i + 2 > n:
            raise Error("zstd: truncated")
        nseq = Int(p[i]) + (Int(p[i + 1]) << 8) + 0x7F00
        i += 2
    var lpos = 0
    if nseq > 0:
        if i >= n:
            raise Error("zstd: truncated")
        var modes = Int(p[i])
        i += 1
        if modes & 3 != 0:
            raise Error("zstd: reserved bits set")
        var took = 0
        _seq_table(modes >> 6, st.ll, _ll_default(), LL_DEFAULT_LOG, 35, LL_MAX_LOG, p + i, n - i, took, st.have_seq)
        i += took
        _seq_table((modes >> 4) & 3, st.of, _of_default(), OF_DEFAULT_LOG, 31, OF_MAX_LOG, p + i, n - i, took, st.have_seq)
        i += took
        _seq_table((modes >> 2) & 3, st.ml, _ml_default(), ML_DEFAULT_LOG, 52, ML_MAX_LOG, p + i, n - i, took, st.have_seq)
        i += took
        st.have_seq = True
        var rd = _RevBits(p + i, n - i)
        var sll = rd.read(st.ll.log)
        var sof = rd.read(st.of.log)
        var sml = rd.read(st.ml.log)
        var llsym = st.ll.sym.unsafe_ptr()
        var llnb = st.ll.nbits.unsafe_ptr()
        var llbase = st.ll.base.unsafe_ptr()
        var ofsym = st.of.sym.unsafe_ptr()
        var ofnb = st.of.nbits.unsafe_ptr()
        var ofbase = st.of.base.unsafe_ptr()
        var mlsym = st.ml.sym.unsafe_ptr()
        var mlnb = st.ml.nbits.unsafe_ptr()
        var mlbase = st.ml.base.unsafe_ptr()
        var llbase_ = st.ll_base.unsafe_ptr()
        var llbits_ = st.ll_bits.unsafe_ptr()
        var mlbase_ = st.ml_base.unsafe_ptr()
        var mlbits_ = st.ml_bits.unsafe_ptr()
        for s in range(nseq):
            var llc = Int(llsym[sll])
            var ofc = Int(ofsym[sof])
            var mlc = Int(mlsym[sml])
            var ofv = (1 << ofc) + rd.read(ofc)
            var mlv = mlbase_[mlc] + rd.read(mlbits_[mlc])
            var llv = llbase_[llc] + rd.read(llbits_[llc])
            if s < nseq - 1:
                sll = Int(llbase[sll]) + rd.read(Int(llnb[sll]))
                sml = Int(mlbase[sml]) + rd.read(Int(mlnb[sml]))
                sof = Int(ofbase[sof]) + rd.read(Int(ofnb[sof]))
            if rd.over:
                raise Error("zstd: sequences run past their stream")
            var off = 0
            if ofv > 3:
                off = ofv - 3
                st.rep3 = st.rep2
                st.rep2 = st.rep1
                st.rep1 = off
            else:
                var idx = ofv + (1 if llv == 0 else 0)
                if idx == 1:
                    off = st.rep1
                elif idx == 2:
                    off = st.rep2
                    st.rep2 = st.rep1
                    st.rep1 = off
                elif idx == 3:
                    off = st.rep3
                    st.rep3 = st.rep2
                    st.rep2 = st.rep1
                    st.rep1 = off
                else:
                    off = st.rep1 - 1
                    if off == 0:
                        raise Error("zstd: bad offset")
                    st.rep3 = st.rep2
                    st.rep2 = st.rep1
                    st.rep1 = off
            if llv > nlit - lpos or llv + mlv > size - o:
                raise Error("zstd: sequence too long")
            for k in range(llv):
                op[o + k] = lp[lpos + k]
            lpos += llv
            o += llv
            if off > o - frame_start or off > window or off == 0:
                raise Error("zstd: offset past the start")
            _copy_match(op, o, off, mlv)
            o += mlv
        if rd.pos != 0:
            raise Error("zstd: sequences do not fill their stream")
    elif i != n:
        raise Error("zstd: bytes after the literals")
    var left = nlit - lpos
    if left > size - o:
        raise Error("zstd: output past its stated size")
    for k in range(left):
        op[o + k] = lp[lpos + k]
    _ = lits^
    return o + left


# ── xxHash64, the zstd content checksum ──────────────────────────────────────────────────────────────

comptime XXP1 = UInt64(11400714785074694791)
comptime XXP2 = UInt64(14029467366897019727)
comptime XXP3 = UInt64(1609587929392839161)
comptime XXP4 = UInt64(9650029242287828579)
comptime XXP5 = UInt64(2870177450012600261)


@always_inline
def _rotl(x: UInt64, r: Int) -> UInt64:
    return (x << UInt64(r)) | (x >> UInt64(64 - r))


@always_inline
def _xx_round(acc: UInt64, v: UInt64) -> UInt64:
    return _rotl(acc + v * XXP2, 31) * XXP1


def _xxh64(p: Ptr, n: Int) -> UInt64:
    var i = 0
    var h = UInt64(0)
    if n >= 32:
        var v1 = XXP1 + XXP2
        var v2 = XXP2
        var v3 = UInt64(0)
        var v4 = UInt64(0) - XXP1
        while i + 32 <= n:
            v1 = _xx_round(v1, _load64(p, i))
            v2 = _xx_round(v2, _load64(p, i + 8))
            v3 = _xx_round(v3, _load64(p, i + 16))
            v4 = _xx_round(v4, _load64(p, i + 24))
            i += 32
        h = _rotl(v1, 1) + _rotl(v2, 7) + _rotl(v3, 12) + _rotl(v4, 18)
        h = (h ^ _xx_round(0, v1)) * XXP1 + XXP4
        h = (h ^ _xx_round(0, v2)) * XXP1 + XXP4
        h = (h ^ _xx_round(0, v3)) * XXP1 + XXP4
        h = (h ^ _xx_round(0, v4)) * XXP1 + XXP4
    else:
        h = XXP5
    h += UInt64(n)
    while i + 8 <= n:
        h = _rotl(h ^ _xx_round(0, _load64(p, i)), 27) * XXP1 + XXP4
        i += 8
    if i + 4 <= n:
        h = _rotl(h ^ (UInt64(_load32(p, i)) * XXP1), 23) * XXP2 + XXP3
        i += 4
    while i < n:
        h = _rotl(h ^ (UInt64(p[i]) * XXP5), 11) * XXP1
        i += 1
    h ^= h >> 33
    h *= XXP2
    h ^= h >> 29
    h *= XXP3
    h ^= h >> 32
    return h


def zstd_decompress(src_addr: Int, n: Int, size: Int) raises -> List[UInt8]:
    """`n` zstd bytes at `src_addr` (one frame, or several in a row, skippable frames ignored) into exactly `size`
    bytes. A frame's content checksum, when it has one, is verified. No dictionaries. `size` comes from a page header
    the file's writer chose: it is refused before any allocation when `n` bytes cannot hold that much (a zstd block of 4
    bytes makes 128 KB at most), and each frame's own stated size must fit it."""
    if size < 0 or size > (n // 4 + 1) * ZSTD_BLOCK_MAX:
        raise Error("zstd: the stated size is impossible for this much input")
    var sp = _ptr(src_addr)
    var out = List[UInt8](unsafe_uninit_length=max(size, 1))
    var op = _ptr(Int(out.unsafe_ptr()))
    var o = 0
    var i = 0
    while i < n:
        if i + 4 > n:
            raise Error("zstd: truncated")
        var magic = Int(_load32(sp, i))
        if magic & 0xFFFFFFF0 == 0x184D2A50:
            if i + 8 > n:
                raise Error("zstd: truncated")
            var skip = Int(_load32(sp, i + 4))
            if skip > n - i - 8:
                raise Error("zstd: truncated")
            i += 8 + skip
            continue
        if magic != 0xFD2FB528:
            raise Error("zstd: not a zstd frame")
        i += 4
        if i >= n:
            raise Error("zstd: truncated")
        var fhd = Int(sp[i])
        i += 1
        if fhd & 8 != 0:
            raise Error("zstd: reserved bit set")
        var single = (fhd >> 5) & 1
        var checksum = (fhd >> 2) & 1
        if fhd & 3 != 0:
            raise Error("zstd: dictionaries are not supported")
        var window = 0
        if single == 0:
            if i >= n:
                raise Error("zstd: truncated")
            var wd = Int(sp[i])
            var base = 1 << (10 + (wd >> 3))
            window = base + (base >> 3) * (wd & 7)
            i += 1                              # the window descriptor
        var fcs_bytes = 0
        var fcs_flag = fhd >> 6
        if fcs_flag == 0:
            fcs_bytes = 1 if single == 1 else 0
        else:
            fcs_bytes = 2 << (fcs_flag - 1)     # 2, 4, 8
        if i + fcs_bytes > n:
            raise Error("zstd: truncated")
        var content = -1
        if fcs_bytes > 0:
            content = 0
            for k in range(fcs_bytes):
                content |= Int(sp[i + k]) << (8 * k)
            if fcs_bytes == 2:
                content += 256
            if content < 0 or content > size - o:
                raise Error("zstd: frame larger than its stated size")
        i += fcs_bytes
        if single == 1:
            window = content                    # a single segment's window is its whole content
        var st = _ZstdState()
        var frame_start = o
        while True:
            if i + 3 > n:
                raise Error("zstd: truncated")
            var bh = Int(sp[i]) | (Int(sp[i + 1]) << 8) | (Int(sp[i + 2]) << 16)
            i += 3
            var last = bh & 1
            var btype = (bh >> 1) & 3
            var bsize = bh >> 3
            if btype == 0:
                if bsize > ZSTD_BLOCK_MAX or bsize > n - i or bsize > size - o:
                    raise Error("zstd: bad raw block")
                for k in range(bsize):
                    op[o + k] = sp[i + k]
                o += bsize
                i += bsize
            elif btype == 1:
                if bsize > ZSTD_BLOCK_MAX or i >= n or bsize > size - o:
                    raise Error("zstd: bad RLE block")
                var v = sp[i]
                for k in range(bsize):
                    op[o + k] = v
                o += bsize
                i += 1
            elif btype == 2:
                if bsize > ZSTD_BLOCK_MAX or bsize > n - i:
                    raise Error("zstd: bad compressed block")
                o = _decode_block(st, sp + i, bsize, op, o, size, frame_start, window)
                i += bsize
            else:
                raise Error("zstd: reserved block type")
            if last == 1:
                break
        if content >= 0 and o - frame_start != content:
            raise Error("zstd: frame shorter than its stated size")
        if checksum == 1:
            if i + 4 > n:
                raise Error("zstd: truncated")
            if _load32(sp, i) != UInt32(_xxh64(op + frame_start, o - frame_start) & 0xFFFFFFFF):
                raise Error("zstd: the content checksum does not match")
            i += 4
    if o != size:
        raise Error("zstd: output does not match its stated size")
    out.resize(size, 0)
    return out^


def zstd_decompress(src: List[UInt8], size: Int) raises -> List[UInt8]:
    return zstd_decompress(Int(src.unsafe_ptr()), len(src), size)


# ── zstd compression ──────────────────────────────────────────────────────────────────────────────────
# A greedy-plus-one-step-lazy matcher over a hash table of four-byte keys, literals Huffman coded, sequences coded with FSE tables
# (predefined, one repeated symbol, or built for the block, whichever costs fewer bits).

struct _BitOut:
    """Bits written from the low end of each byte upward, collected a 32-bit word at a time."""
    var words: List[UInt32]
    var acc: UInt64
    var n: Int                      # bits in `acc` (< 32 between calls)

    def __init__(out self):
        self.words = List[UInt32]()
        self.acc = 0
        self.n = 0

    @always_inline
    def add(mut self, v: Int, k: Int):
        self.acc |= (UInt64(v) & ((UInt64(1) << UInt64(k)) - 1)) << UInt64(self.n)
        self.n += k
        if self.n >= 32:
            self.words.append(UInt32(self.acc & 0xFFFFFFFF))
            self.acc >>= 32
            self.n -= 32

    def close_stream(mut self):
        """Ends a backwards-read stream: a 1 bit (the decoder finds where the data ends by it), zeros to a whole byte."""
        self.add(1, 1)

    def take(mut self) -> List[UInt8]:
        """The bytes written, the last one padded with zero bits."""
        var rest = (self.n + 7) // 8
        var out = List[UInt8](unsafe_uninit_length=4 * len(self.words) + rest)
        for w in range(len(self.words)):
            var x = self.words[w]
            for k in range(4):
                out[4 * w + k] = UInt8((x >> UInt32(8 * k)) & 0xFF)
        for k in range(rest):
            out[4 * len(self.words) + k] = UInt8((self.acc >> UInt64(8 * k)) & 0xFF)
        self.words.clear()
        self.acc = 0
        self.n = 0
        return out^


struct _FseEnc(Copyable, Movable):
    """What encoding with an FSE table needs: for each symbol where its states are and how many bits it costs."""
    var log: Int
    var rle: Bool
    var delta_nb: List[Int]
    var delta_find: List[Int]
    var states: List[Int]

    def __init__(out self, log: Int, rle: Bool, var delta_nb: List[Int], var delta_find: List[Int], var states: List[Int]):
        self.log = log
        self.rle = rle
        self.delta_nb = delta_nb^
        self.delta_find = delta_find^
        self.states = states^

    def init_state(self, sym: Int) -> Int:
        if self.rle:
            return 0
        var nb = (self.delta_nb[sym] + (1 << 15)) >> 16
        var value = (nb << 16) - self.delta_nb[sym]
        return self.states[(value >> nb) + self.delta_find[sym]]

    @always_inline
    def encode(self, mut out: _BitOut, mut state: Int, sym: Int):
        if self.rle:
            return
        var nb = (state + self.delta_nb[sym]) >> 16
        out.add(state, nb)
        state = self.states[(state >> nb) + self.delta_find[sym]]

    def flush(self, mut out: _BitOut, state: Int):
        if self.rle:
            return
        out.add(state, self.log)

    def flush_to(self, mut out: _BitW, state: Int):
        if self.rle:
            return
        out.add(state, self.log)


def _fse_enc_build(norm: List[Int], log: Int) raises -> _FseEnc:
    var size = 1 << log
    var spread = _fse_spread(norm, log)
    var cumul = List[Int](length=len(norm) + 1, fill=0)
    for s in range(len(norm)):
        cumul[s + 1] = cumul[s] + (1 if norm[s] == -1 else (norm[s] if norm[s] > 0 else 0))
    var fill = List[Int](length=len(norm), fill=0)
    for s in range(len(norm)):
        fill[s] = cumul[s]
    var states = List[Int](length=size, fill=0)
    for u in range(size):
        var s = spread[u]
        states[fill[s]] = size + u
        fill[s] += 1
    var delta_nb = List[Int](length=len(norm), fill=0)
    var delta_find = List[Int](length=len(norm), fill=0)
    var total = 0
    for s in range(len(norm)):
        var c = norm[s]
        if c == 0:
            delta_nb[s] = ((log + 1) << 16) - size
        elif c == -1 or c == 1:
            delta_nb[s] = (log << 16) - size
            delta_find[s] = total - 1
            total += 1
        else:
            var maxbits = log - _highbit(c - 1)
            var minplus = c << maxbits
            delta_nb[s] = (maxbits << 16) - minplus
            delta_find[s] = total - c
            total += c
    return _FseEnc(log, False, delta_nb^, delta_find^, states^)


def _fse_enc_rle() -> _FseEnc:
    return _FseEnc(0, True, List[Int](), List[Int](), List[Int]())


def _fse_normalize(counts: List[Int], total: Int, log: Int) -> List[Int]:
    """Counts scaled to sum to 1 << log, every symbol that occurs keeping at least 1."""
    var size = 1 << log
    var norm = List[Int](length=len(counts), fill=0)
    var sum = 0
    var biggest = 0
    for s in range(len(counts)):
        if counts[s] > 0:
            var q = counts[s] * size // total
            norm[s] = q if q > 0 else 1
            sum += norm[s]
            if counts[s] > counts[biggest]:
                biggest = s
    var diff = size - sum
    if diff > 0:
        # give the rest to the symbols that lost the most in rounding, largest first
        var order = List[Int]()
        for s in range(len(counts)):
            if counts[s] > 0:
                order.append(((counts[s] * size) % total) * 4096 + s)      # remainder, then symbol (ascending sorts last)
        sort(order)
        var k = len(order) - 1
        while diff > 0:
            norm[order[k] & 4095] += 1
            diff -= 1
            k -= 1
            if k < 0:
                k = len(order) - 1
    while diff < 0:                                  # rounding up the rare ones overshot: take from the biggest
        for s in range(len(norm)):
            if norm[s] > norm[biggest]:
                biggest = s
        norm[biggest] -= 1
        diff += 1
    return norm^


def _write_ncount(norm: List[Int], log: Int) -> List[UInt8]:
    """The table description the decoder's `_read_ncount` reads."""
    var w = _BitOut()
    var size = 1 << log
    var remaining = size + 1
    var threshold = size
    var nbits = log + 1
    w.add(log - 5, 4)
    var sym = 0
    var prev0 = False
    var alphabet = len(norm)
    while sym < alphabet and remaining > 1:
        if prev0:
            var start = sym
            while sym < alphabet and norm[sym] == 0:
                sym += 1
            if sym == alphabet:
                break
            while sym >= start + 24:
                start += 24
                w.add(0xFFFF, 16)
            while sym >= start + 3:
                start += 3
                w.add(3, 2)
            w.add(sym - start, 2)
        var count = norm[sym]
        sym += 1
        var mx = (2 * threshold - 1) - remaining
        remaining -= -count if count < 0 else count
        count += 1
        if count >= threshold:
            count += mx
        w.add(count, nbits - 1 if count < mx else nbits)          # a short code takes one bit fewer
        prev0 = count == 1
        while remaining < threshold:
            nbits -= 1
            threshold >>= 1
    return w.take()


def _cost(x: Int) -> Int:
    """About 256 * log2(x) for x >= 1 (the bits an FSE symbol costs, for choosing between tables)."""
    var h = _highbit(x)
    var frac = ((x << 8) >> h) - 256 if h <= 8 else (x >> (h - 8)) - 256
    return h * 256 + frac


def _table_cost(counts: List[Int], norm: List[Int], log: Int) -> Int:
    """Estimated bits to code `counts` with the table `norm`; a huge number when a symbol has no state."""
    var bits = 0
    for s in range(len(counts)):
        if counts[s] == 0:
            continue
        if s >= len(norm) or norm[s] == 0:
            return 1 << 60
        var p = 1 if norm[s] == -1 else norm[s]
        bits += counts[s] * ((log << 8) - _cost(p))
    return bits >> 8


# ── Huffman coding of literals ────────────────────────────────────────────────────────────────────────

def _huf_depths(f: List[Int]) -> List[Int]:
    """Code length of each symbol (0 when absent) of an optimal prefix code for the frequencies `f` (two or more
    symbols occur)."""
    var keys = List[Int]()
    for s in range(len(f)):
        if f[s] > 0:
            keys.append((f[s] << 8) | s)
    sort(keys)
    var n = len(keys)
    var w = List[Int](length=2 * n - 1, fill=0)
    var parent = List[Int](length=2 * n - 1, fill=0)
    for k in range(n):
        w[k] = keys[k] >> 8
    var li = 0
    var ii = n
    var nxt = n
    for _ in range(n - 1):
        var a = 0
        if li < n and (ii >= nxt or w[li] <= w[ii]):
            a = li
            li += 1
        else:
            a = ii
            ii += 1
        var b = 0
        if li < n and (ii >= nxt or w[li] <= w[ii]):
            b = li
            li += 1
        else:
            b = ii
            ii += 1
        w[nxt] = w[a] + w[b]
        parent[a] = nxt
        parent[b] = nxt
        nxt += 1
    var depth = List[Int](length=2 * n - 1, fill=0)
    for k in range(2 * n - 3, -1, -1):
        depth[k] = depth[parent[k]] + 1
    var lens = List[Int](length=len(f), fill=0)
    for k in range(n):
        lens[keys[k] & 255] = depth[k]
    return lens^


def _huf_lengths(freq: List[Int]) -> List[Int]:
    """Code lengths of at most HUF_MAX_BITS: while the optimal code is deeper, halve the frequencies (flattening the
    tree) and try again."""
    var f = freq.copy()
    while True:
        var lens = _huf_depths(f)
        var deepest = 0
        for l in lens:
            if l > deepest:
                deepest = l
        if deepest <= HUF_MAX_BITS:
            return lens^
        for s in range(len(f)):
            if f[s] > 0:
                f[s] = (f[s] + 1) >> 1


def _fse_compress_weights(w: List[Int], nw: Int) raises -> List[UInt8]:
    """The first `nw` weights as an FSE stream (table description, then two interleaved states); empty when that
    cannot be done (a single distinct weight)."""
    var counts = List[Int](length=13, fill=0)
    var distinct = 0
    for i in range(nw):
        if counts[w[i]] == 0:
            distinct += 1
        counts[w[i]] += 1
    if distinct < 2:
        return List[UInt8]()
    var log = 6
    var norm = _fse_normalize(counts, nw, log)
    var enc = _fse_enc_build(norm, log)
    var out = _write_ncount(norm, log)
    var bw = _BitOut()
    var st1 = 0
    var st2 = 0
    var i = nw - 1
    if i % 2 == 0:
        st1 = enc.init_state(w[i])
    else:
        st2 = enc.init_state(w[i])
    i -= 1
    if i % 2 == 0:
        st1 = enc.init_state(w[i])
    else:
        st2 = enc.init_state(w[i])
    i -= 1
    while i >= 0:
        if i % 2 == 0:
            enc.encode(bw, st1, w[i])
        else:
            enc.encode(bw, st2, w[i])
        i -= 1
    enc.flush(bw, st2)
    enc.flush(bw, st1)
    bw.close_stream()
    out.extend(bw.take())
    return out^


def _huf_stream(lens: List[Int], codes: List[Int], p: Ptr, start: Int, count: Int) -> List[UInt8]:
    var bw = _BitOut()
    for i in range(start + count - 1, start - 1, -1):
        var s = Int(p[i])
        bw.add(codes[s], lens[s])
    bw.close_stream()
    return bw.take()


def _huf_compress(p: Ptr, n: Int, mut tree_and_streams: List[UInt8]) raises -> Bool:
    """Huffman codes the `n` literals at `p`: the tree description, then one stream (n < 256) or the jump table and
    four. False when the literals do not allow it (fewer than two symbols, or the result is too big a header)."""
    var freq = List[Int](length=256, fill=0)
    for i in range(n):
        freq[Int(p[i])] += 1
    var distinct = 0
    var last = 0
    for s in range(256):
        if freq[s] > 0:
            distinct += 1
            last = s
    if distinct < 2:
        return False
    var lens = _huf_lengths(freq)
    var maxb = 0
    for l in lens:
        if l > maxb:
            maxb = l
    var weight = List[Int](length=256, fill=0)
    var count = List[Int](length=14, fill=0)
    for s in range(256):
        if lens[s] > 0:
            weight[s] = maxb + 1 - lens[s]
            count[weight[s]] += 1
    var start = List[Int](length=14, fill=0)
    var next = 0
    for wv in range(1, 14):
        start[wv] = next
        next += count[wv] << (wv - 1)
    var codes = List[Int](length=256, fill=0)
    for s in range(256):
        if lens[s] > 0:
            var at = start[weight[s]]
            codes[s] = at >> (maxb - lens[s])
            start[weight[s]] = at + (1 << (weight[s] - 1))
    # the tree description: the weights of symbols 0 .. last-1 (the last one's follows from the total)
    var tree = List[UInt8]()
    if last <= 128:
        tree.append(UInt8(127 + last))
        var k = 0
        while k < last:
            var hi = weight[k]
            var lo = weight[k + 1] if k + 1 < last else 0
            tree.append(UInt8((hi << 4) | lo))
            k += 2
    else:
        var packed = _fse_compress_weights(weight, last)
        if len(packed) == 0 or len(packed) >= 128:
            return False
        tree.append(UInt8(len(packed)))
        tree.extend(packed^)
    tree_and_streams = tree^
    if n < 256:
        tree_and_streams.extend(_huf_stream(lens, codes, p, 0, n))
        return True
    var q = (n + 3) // 4
    var s1 = _huf_stream(lens, codes, p, 0, q)
    var s2 = _huf_stream(lens, codes, p, q, q)
    var s3 = _huf_stream(lens, codes, p, 2 * q, q)
    var s4 = _huf_stream(lens, codes, p, 3 * q, n - 3 * q)
    if len(s1) > 65535 or len(s2) > 65535 or len(s3) > 65535:
        return False
    for v in [len(s1), len(s2), len(s3)]:
        tree_and_streams.append(UInt8(v & 0xFF))
        tree_and_streams.append(UInt8(v >> 8))
    tree_and_streams.extend(s1^)
    tree_and_streams.extend(s2^)
    tree_and_streams.extend(s3^)
    tree_and_streams.extend(s4^)
    return True


# ── finding matches ───────────────────────────────────────────────────────────────────────────────────

comptime ZSTD_WINDOW_LOG = 20
comptime ZSTD_WINDOW = 1 << ZSTD_WINDOW_LOG
comptime WAYS = 4                       # candidates looked at per position
comptime HASH_BITS_MAX = 16             # buckets: 64 K, 2 MB
comptime LAZY_MAX = 16                  # a match this long is taken at once; a shorter one gives way to a better one a byte on
comptime MIN_MATCH = 4
comptime FAR_SHORT = 1 << 12           # a match of MIN_MATCH bytes this far back is not worth its offset


@always_inline
def _match_len(p: Ptr, a: Int, b: Int, limit: Int) -> Int:
    """How many bytes from `b` (stopping at `limit`) equal those from `a` (< b)."""
    var l = 0
    while b + l + 8 <= limit:
        var x = _load64(p, a + l) ^ _load64(p, b + l)
        if x != 0:
            return l + Int(count_trailing_zeros(x)) // 8
        l += 8
    while b + l < limit and p[a + l] == p[b + l]:
        l += 1
    return l


struct _Matcher:
    """Where each four-byte key was last seen: a hash of the key picks a bucket of WAYS entries, newest first, each
    holding the position and the key itself, so a candidate that cannot match is turned down without reading the input
    at its position (a read the cache rarely has)."""
    var a: Int                      # address of the input
    var table: List[UInt64]         # per bucket WAYS entries: (position + 1) | key << 32, 0 for none
    var hshift: Int

    def __init__(out self, a: Int, n: Int):
        self.a = a
        var hbits = 8
        while hbits < HASH_BITS_MAX and (1 << hbits) < n:
            hbits += 1
        self.hshift = 32 - hbits
        self.table = List[UInt64](length=WAYS << hbits, fill=0)

    @always_inline
    def _bucket(self, cur: UInt32) -> Int:
        return Int((cur * UInt32(2654435761)) >> UInt32(self.hshift)) * WAYS

    @always_inline
    def _push(mut self, b: Int, i: Int, cur: UInt32):
        var t = self.table.unsafe_ptr()
        comptime for w in range(WAYS - 1, 0, -1):
            t[b + w] = t[b + w - 1]
        t[b] = UInt64(i + 1) | (UInt64(cur) << 32)

    @always_inline
    def insert(mut self, i: Int):
        var cur = _load32(_ptr(self.a), i)
        self._push(self._bucket(cur), i, cur)

    def find(mut self, i: Int, limit: Int, rep1: Int) -> Int:
        """The best match for position i (its bytes up to `limit`), packed as length << 21 | offset; 0 for none. Also
        indexes i."""
        var p = _ptr(self.a)
        var cur = _load32(p, i)
        var b = self._bucket(cur)
        var best_len = 0
        var best_off = 0
        if rep1 <= i and _load32(p, i - rep1) == cur:
            best_len = _match_len(p, i - rep1, i, limit)
            best_off = rep1
        var t = self.table.unsafe_ptr()
        for w in range(WAYS):
            var e = t[b + w]
            if e == 0:
                break
            if UInt32(e >> 32) != cur:
                continue
            var d = i - (Int(e & 0xFFFFFFFF) - 1)
            if d > ZSTD_WINDOW:
                break                                      # the rest are older still
            var l = _match_len(p, i - d, i, limit)
            if l > best_len:
                best_len = l
                best_off = d
        self._push(b, i, cur)
        if best_len < MIN_MATCH or (best_len == MIN_MATCH and best_off >= FAR_SHORT):
            return 0
        return (best_len << 21) | best_off


@fieldwise_init
struct _Seqs(Movable):
    var ll: List[Int]               # literals before each match
    var ml: List[Int]               # match lengths
    var ov: List[Int]               # offset values as the format codes them (1-3: repeats, else offset + 3)
    var lits: List[UInt8]           # every literal, in order


struct _ZEnc:
    """What compression carries from block to block."""
    var rep1: Int
    var rep2: Int
    var rep3: Int
    var ll_bits: List[Int]
    var ll_base: List[Int]
    var ml_bits: List[Int]
    var ml_base: List[Int]
    var ll_lut: List[Int]           # literal length (< 64) -> code
    var ml_lut: List[Int]           # match length - 3 (< 128) -> code

    def __init__(out self):
        self.rep1 = 1
        self.rep2 = 4
        self.rep3 = 8
        self.ll_bits = _ll_bits()
        self.ll_base = _bases(self.ll_bits, 0)
        self.ml_bits = _ml_bits()
        self.ml_base = _bases(self.ml_bits, 3)
        self.ll_lut = List[Int](length=64, fill=0)
        self.ml_lut = List[Int](length=128, fill=0)
        var c = 0
        for v in range(64):
            while c + 1 < 36 and self.ll_base[c + 1] <= v:
                c += 1
            self.ll_lut[v] = c
        c = 0
        for v in range(128):
            while c + 1 < 53 and self.ml_base[c + 1] <= v + 3:
                c += 1
            self.ml_lut[v] = c

    @always_inline
    def ll_code(self, v: Int) -> Int:
        return self.ll_lut[v] if v < 64 else _highbit(v) + 19

    @always_inline
    def ml_code(self, m: Int) -> Int:
        var v = m - 3
        return self.ml_lut[v] if v < 128 else _highbit(v) + 36

    def offset_value(mut self, off: Int, ll: Int) -> Int:
        """The offset as a sequence states it, updating the repeat offsets the way the decoder will."""
        if ll > 0:
            if off == self.rep1:
                return 1
            if off == self.rep2:
                self.rep2 = self.rep1
                self.rep1 = off
                return 2
            if off == self.rep3:
                self.rep3 = self.rep2
                self.rep2 = self.rep1
                self.rep1 = off
                return 3
        else:
            if off == self.rep2:
                self.rep2 = self.rep1
                self.rep1 = off
                return 1
            if off == self.rep3:
                self.rep3 = self.rep2
                self.rep2 = self.rep1
                self.rep1 = off
                return 2
            if self.rep1 > 1 and off == self.rep1 - 1:
                self.rep3 = self.rep2
                self.rep2 = self.rep1
                self.rep1 = off
                return 3
        self.rep3 = self.rep2
        self.rep2 = self.rep1
        self.rep1 = off
        return off + 3


@always_inline
def _gain(len: Int, off: Int, rep1: Int) -> Int:
    """What a match is worth in bits: eight per byte it covers, less what its offset and the sequence around it cost (a
    few bits for the offset used last, which a sequence can repeat)."""
    return len * 8 - (0 if off == rep1 else _highbit(off + 1) + 8)


def _find_sequences(mut z: _ZEnc, mut m: _Matcher, p: Ptr, start: Int, end: Int, mut sq: _Seqs):
    var i = start
    var anchor = start
    var skip_step = 0
    while i + MIN_MATCH <= end:
        var len = 0
        var off = 0
        var q = i + 1
        # a match at the offset used last, one byte on, is the cheapest to code: take it before looking further
        if z.rep1 <= i and i + 1 + MIN_MATCH <= end and _load32(p, i + 1 - z.rep1) == _load32(p, i + 1):
            m.insert(i)
            len = _match_len(p, i + 1 - z.rep1, i + 1, end)
            off = z.rep1
            i += 1
            q = i
        else:
            var found = m.find(i, end, z.rep1)
            if found == 0:
                skip_step = min((i - anchor) >> 8, 3)       # nothing found for a long while: look at fewer positions (but still
                                                            # index enough of them for a later copy of this data to be found)
                i += 1 + skip_step
                continue
            len = found >> 21
            off = found & ((1 << 21) - 1)
            if len < LAZY_MAX and i + 1 + MIN_MATCH <= end:     # lazy: is the next position clearly better?
                var next = m.find(i + 1, end, z.rep1)
                var nlen = next >> 21
                var noff = next & ((1 << 21) - 1)
                if next != 0 and _gain(nlen, noff, z.rep1) > _gain(len, off, z.rep1) + 8:      # one more literal is 8 bits
                    i += 1
                    len = nlen
                    off = noff
                    q = i + 1
                else:
                    q = i + 2                               # i + 1 is indexed already
        var ll = i - anchor
        for k in range(anchor, i):
            sq.lits.append(p[k])
        sq.ll.append(ll)
        sq.ml.append(len)
        sq.ov.append(z.offset_value(off, ll))
        var stop = i + len
        var qlim = min(stop, end - MIN_MATCH + 1)
        if qlim - q > 32:
            for _ in range(8):
                m.insert(q)
                q += 1
            q = qlim - 8
        while q < qlim:
            m.insert(q)
            q += 1
        i = stop
        anchor = i
    for k in range(anchor, end):
        sq.lits.append(p[k])


# ── blocks ────────────────────────────────────────────────────────────────────────────────────────────

def _plain_header(mut out: List[UInt8], kind: Int, n: Int):
    """The header of a raw (kind 0) or one-repeated-byte (kind 1) literals section: 5, 12 or 20 bits of size."""
    if n < 32:
        out.append(UInt8(kind | (n << 3)))
    elif n < 4096:
        out.append(UInt8(kind | (1 << 2) | ((n & 15) << 4)))
        out.append(UInt8(n >> 4))
    else:
        out.append(UInt8(kind | (3 << 2) | ((n & 15) << 4)))
        out.append(UInt8((n >> 4) & 0xFF))
        out.append(UInt8(n >> 12))


def _literals_section(p: Ptr, n: Int) raises -> List[UInt8]:
    """The literals of a block: Huffman coded when that is clearly smaller, else one repeated byte, else raw."""
    var out = List[UInt8]()
    var same = n > 1
    for i in range(1, n):
        if p[i] != p[0]:
            same = False
            break
    if same:
        _plain_header(out, 1, n)
        out.append(p[0])
        return out^
    var plain = (1 if n < 32 else (2 if n < 4096 else 3)) + n
    if n >= 32:
        var packed = List[UInt8]()
        if _huf_compress(p, n, packed):
            var c = len(packed)
            var hdr = 3 if (n < 1024 and c < 1024) else (4 if (n < 16384 and c < 16384) else 5)
            if hdr + c + 1 < plain:
                var fmt = (0 if n < 256 else 1) if hdr == 3 else (2 if hdr == 4 else 3)
                var v = 2 | (fmt << 2) | (n << 4) | (c << (14 if hdr == 3 else (18 if hdr == 4 else 22)))
                for k in range(hdr):
                    out.append(UInt8((v >> (8 * k)) & 0xFF))
                out.extend(packed^)
                return out^
    _plain_header(out, 0, n)
    for i in range(n):
        out.append(p[i])
    return out^


def _pick_table(counts: List[Int], n: Int, max_symbol: Int, default_norm: List[Int], default_log: Int, max_log: Int,
                mut mode: Int, mut desc: List[UInt8]) raises -> _FseEnc:
    """The table for one of the three code streams (`n` codes, `counts` of each symbol): predefined (mode 0), a repeated
    symbol (1) or one made for these codes (2), by the bits each would cost."""
    var distinct = 0
    var top = 0
    for s in range(max_symbol + 1):
        if counts[s] > 0:
            distinct += 1
            top = s
    desc = List[UInt8]()
    if distinct == 1:
        mode = 1
        desc.append(UInt8(top))
        return _fse_enc_rle()
    var predefined = _table_cost(counts, default_norm, default_log)
    var log = max_log
    var by_src = _highbit(n - 1) - 1
    if by_src < log:
        log = by_src
    var min_bits = min(_highbit(n) + 1, _highbit(top) + 2)
    if min_bits > log:
        log = min_bits
    if log < 5:
        log = 5
    if log > max_log:
        log = max_log
    if (1 << log) >= distinct:
        var norm = _fse_normalize(counts, n, log)
        var header = _write_ncount(norm, log)
        var custom = _table_cost(counts, norm, log) + 8 * len(header)
        if custom < predefined:
            mode = 2
            desc = header^
            return _fse_enc_build(norm, log)
    mode = 0
    return _fse_enc_build(default_norm, default_log)


struct _BitW:
    """Bits written from the low end of each byte upward into a buffer the caller sized, 32 at a time."""
    var base: Int                   # the buffer's address
    var pos: Int
    var acc: UInt64
    var n: Int                      # bits in `acc` (< 32 between calls)

    def __init__(out self, base: Int):
        self.base = base
        self.pos = 0
        self.acc = 0
        self.n = 0

    @always_inline
    def add(mut self, v: Int, k: Int):
        self.acc |= (UInt64(v) & ((UInt64(1) << UInt64(k)) - 1)) << UInt64(self.n)
        self.n += k
        if self.n >= 32:
            (_ptr(self.base) + self.pos).bitcast[UInt32]().store[alignment=1](UInt32(self.acc & 0xFFFFFFFF))
            self.pos += 4
            self.acc >>= 32
            self.n -= 32

    def finish(mut self) -> Int:
        """Ends a backwards-read stream (a 1 bit, zeros to a whole byte); the bytes written."""
        self.add(1, 1)
        var rest = (self.n + 7) // 8
        for k in range(rest):
            _ptr(self.base)[self.pos + k] = UInt8((self.acc >> UInt64(8 * k)) & 0xFF)
        return self.pos + rest


comptime IPtr = UnsafePointer[Int, ImmutAnyOrigin]


def _ip(ref l: List[Int]) -> IPtr:
    return IPtr(unsafe_from_address=Int(l.unsafe_ptr()))


@always_inline
def _fse_step(rle: Bool, mut bw: _BitW, mut state: Int, sym: Int, dnb: IPtr,
              dfind: IPtr, states: IPtr):
    if not rle:
        var nb = (state + dnb[sym]) >> 16
        bw.add(state, nb)
        state = states[(state >> nb) + dfind[sym]]


def _sequence_stream(z: _ZEnc, sq: _Seqs, llc: List[UInt8], mlc: List[UInt8], ofc: List[UInt8],
                     ll_enc: _FseEnc, of_enc: _FseEnc, ml_enc: _FseEnc) -> List[UInt8]:
    """The bit stream of the sequences: written last sequence first, the decoder reads it backwards."""
    var nseq = len(sq.ll)
    var out = List[UInt8](unsafe_uninit_length=16 * nseq + 32)             # at most 120 bits a sequence
    var bw = _BitW(Int(out.unsafe_ptr()))
    var lcp = llc.unsafe_ptr()
    var mcp = mlc.unsafe_ptr()
    var ocp = ofc.unsafe_ptr()
    var llp = sq.ll.unsafe_ptr()
    var mlp = sq.ml.unsafe_ptr()
    var ovp = sq.ov.unsafe_ptr()
    var llb = z.ll_base.unsafe_ptr()
    var llx = z.ll_bits.unsafe_ptr()
    var mlb = z.ml_base.unsafe_ptr()
    var mlx = z.ml_bits.unsafe_ptr()
    var l_dnb = _ip(ll_enc.delta_nb)
    var l_dfd = _ip(ll_enc.delta_find)
    var l_st = _ip(ll_enc.states)
    var m_dnb = _ip(ml_enc.delta_nb)
    var m_dfd = _ip(ml_enc.delta_find)
    var m_st = _ip(ml_enc.states)
    var o_dnb = _ip(of_enc.delta_nb)
    var o_dfd = _ip(of_enc.delta_find)
    var o_st = _ip(of_enc.states)
    var l_rle = ll_enc.rle
    var m_rle = ml_enc.rle
    var o_rle = of_enc.rle
    var last = nseq - 1
    var st_ml = ml_enc.init_state(Int(mcp[last]))
    var st_of = of_enc.init_state(Int(ocp[last]))
    var st_ll = ll_enc.init_state(Int(lcp[last]))
    bw.add(llp[last] - llb[Int(lcp[last])], llx[Int(lcp[last])])
    bw.add(mlp[last] - mlb[Int(mcp[last])], mlx[Int(mcp[last])])
    bw.add(ovp[last] - (1 << Int(ocp[last])), Int(ocp[last]))
    for k in range(last - 1, -1, -1):
        var lc = Int(lcp[k])
        var mc = Int(mcp[k])
        var oc = Int(ocp[k])
        _fse_step(o_rle, bw, st_of, oc, o_dnb, o_dfd, o_st)
        _fse_step(m_rle, bw, st_ml, mc, m_dnb, m_dfd, m_st)
        _fse_step(l_rle, bw, st_ll, lc, l_dnb, l_dfd, l_st)
        bw.add(llp[k] - llb[lc], llx[lc])
        bw.add(mlp[k] - mlb[mc], mlx[mc])
        bw.add(ovp[k] - (1 << oc), oc)
    ml_enc.flush_to(bw, st_ml)
    of_enc.flush_to(bw, st_of)
    ll_enc.flush_to(bw, st_ll)
    var size = bw.finish()
    out.resize(size, 0)
    _ = llc
    _ = mlc
    _ = ofc
    return out^


def _sequences_section(z: _ZEnc, sq: _Seqs) raises -> List[UInt8]:
    var nseq = len(sq.ll)
    var out = List[UInt8]()
    if nseq == 0:
        out.append(0)
        return out^
    if nseq < 128:
        out.append(UInt8(nseq))
    elif nseq < 0x7F00:
        out.append(UInt8((nseq >> 8) + 128))
        out.append(UInt8(nseq & 0xFF))
    else:
        out.append(255)
        out.append(UInt8((nseq - 0x7F00) & 0xFF))
        out.append(UInt8((nseq - 0x7F00) >> 8))
    # the codes of every sequence (one byte each) and how often each occurs
    var llc = List[UInt8](unsafe_uninit_length=nseq)
    var mlc = List[UInt8](unsafe_uninit_length=nseq)
    var ofc = List[UInt8](unsafe_uninit_length=nseq)
    var ll_n = List[Int](length=36, fill=0)
    var ml_n = List[Int](length=53, fill=0)
    var of_n = List[Int](length=32, fill=0)
    for k in range(nseq):
        var a = z.ll_code(sq.ll[k])
        var b = z.ml_code(sq.ml[k])
        var c = _highbit(sq.ov[k])
        llc[k] = UInt8(a)
        mlc[k] = UInt8(b)
        ofc[k] = UInt8(c)
        ll_n[a] += 1
        ml_n[b] += 1
        of_n[c] += 1
    var mode_ll = 0
    var mode_of = 0
    var mode_ml = 0
    var desc_ll = List[UInt8]()
    var desc_of = List[UInt8]()
    var desc_ml = List[UInt8]()
    var ll_enc = _pick_table(ll_n, nseq, 35, _ll_default(), LL_DEFAULT_LOG, LL_MAX_LOG, mode_ll, desc_ll)
    var of_enc = _pick_table(of_n, nseq, 31, _of_default(), OF_DEFAULT_LOG, OF_MAX_LOG, mode_of, desc_of)
    var ml_enc = _pick_table(ml_n, nseq, 52, _ml_default(), ML_DEFAULT_LOG, ML_MAX_LOG, mode_ml, desc_ml)
    out.append(UInt8((mode_ll << 6) | (mode_of << 4) | (mode_ml << 2)))
    out.extend(desc_ll^)
    out.extend(desc_of^)
    out.extend(desc_ml^)
    out.extend(_sequence_stream(z, sq, llc, mlc, ofc, ll_enc, of_enc, ml_enc))
    return out^


def _block_header(mut out: List[UInt8], last: Bool, kind: Int, size: Int):
    var v = (1 if last else 0) | (kind << 1) | (size << 3)
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))


def zstd_compress(src: List[UInt8]) raises -> List[UInt8]:
    """One zstd frame holding `src`: blocks of up to 128 KB, each compressed when that makes it smaller, else
    stored (or, for one repeated byte, a run). The window is 1 MB; the frame states the content size."""
    var n = len(src)
    if n > 2147483647:
        raise Error("zstd: input too large")
    var out = List[UInt8]()
    out.append(0x28)
    out.append(0xB5)
    out.append(0x2F)
    out.append(0xFD)
    out.append(0x80)                                           # content size in 4 bytes, window descriptor follows
    out.append(UInt8((ZSTD_WINDOW_LOG - 10) << 3))
    for k in range(4):
        out.append(UInt8((n >> (8 * k)) & 0xFF))
    if n == 0:
        _block_header(out, True, 0, 0)
        return out^
    var sp = _ptr(Int(src.unsafe_ptr()))
    var z = _ZEnc()
    var m = _Matcher(Int(src.unsafe_ptr()), n)
    var pos = 0
    while pos < n:
        var end = min(pos + ZSTD_BLOCK_MAX, n)
        var last = end == n
        var size = end - pos
        var same = size > 1
        for i in range(pos + 1, end):
            if sp[i] != sp[pos]:
                same = False
                break
        if same:
            _block_header(out, last, 1, size)
            out.append(sp[pos])
            pos = end
            continue
        var r1 = z.rep1
        var r2 = z.rep2
        var r3 = z.rep3
        var sq = _Seqs(List[Int](), List[Int](), List[Int](), List[UInt8]())
        _find_sequences(z, m, sp, pos, end, sq)
        var body = _literals_section(_ptr(Int(sq.lits.unsafe_ptr())), len(sq.lits))
        body.extend(_sequences_section(z, sq))
        if len(body) < size:
            _block_header(out, last, 2, len(body))
            out.extend(body^)
        else:
            z.rep1 = r1                                        # a stored block leaves the decoder's offsets alone
            z.rep2 = r2
            z.rep3 = r3
            _block_header(out, last, 0, size)
            for i in range(pos, end):
                out.append(sp[i])
        pos = end
    return out^
