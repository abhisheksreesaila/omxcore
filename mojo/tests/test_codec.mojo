# F-180: zstd and snappy written in Mojo (omxcore/codec.mojo), the codecs of every build with no C library.
# The oracle is the C libraries linked into this test: what the Mojo encoder writes libzstd / libsnappy must decode to the exact
# input, and what they write (zstd levels 1, 3, 9 and 19, which between them use Huffman literals with FSE-coded weights, every
# FSE table mode and repeat offsets) the Mojo decoder must decode to the exact input. The frames the encoder writes are opened
# to prove that the block types and table modes the compression ratio depends on were really used. Corrupt input must raise or
# decode, never crash.
from std.ffi import external_call
from omxcore.codec import (zstd_compress, zstd_decompress, snappy_compress, snappy_decompress, _literals_section, _sequences_section, _ZEnc,
                           _Seqs, _ptr)


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


# kinds of input: what the pages of a Parquet file look like, and what they never do
comptime RANDOM = 0          # incompressible
comptime PERIOD7 = 1         # a short period
comptime MOSTLY_COUNTING = 2
comptime ONE_BYTE = 3        # one byte repeated
comptime UNIFORM_64 = 4      # 64 equally likely byte values: Huffman only
comptime FEW_SYMBOLS = 5     # a few letters, no structure
comptime SKEWED_255 = 6      # byte values 0..254, small ones likelier: Huffman with weights past 128 symbols
comptime WORDS = 7           # words from a small vocabulary
comptime INT64S = 8          # little-endian 64-bit integers that grow
comptime KINDS = 9


def make(kind: Int, n: Int) -> List[UInt8]:
    var d = List[UInt8](length=n, fill=0)
    var s = UInt64(88172645463325252)
    for i in range(n):
        s ^= s << 13
        s ^= s >> 7
        s ^= s << 17
        if kind == RANDOM:
            d[i] = UInt8(s & 0xFF)
        elif kind == PERIOD7:
            d[i] = UInt8(i % 7)
        elif kind == MOSTLY_COUNTING:
            d[i] = UInt8((i // 100) & 0xFF) if (s & 15) != 0 else UInt8(s & 0xFF)
        elif kind == ONE_BYTE:
            d[i] = 5
        elif kind == UNIFORM_64:
            d[i] = UInt8((s >> 11) & 63)
        elif kind == FEW_SYMBOLS:
            d[i] = UInt8(97 + Int((s >> 8) & 7) * Int((s >> 20) & 1) + Int((s >> 30) & 3))
        elif kind == SKEWED_255:
            var r = Int((s >> 20) & 0xFFFF) >> 8
            d[i] = UInt8((r * r) >> 8)
        elif kind == WORDS:
            var w = Int((s >> 24) & 15)
            d[i] = UInt8(97 + (w * 7 + (i % 5)) % 26) if (i % 6) != 5 else UInt8(32)
        else:
            var v = i // 8 * 37 + Int(s & 3)
            d[i] = UInt8((v >> (8 * (i % 8))) & 0xFF)
    return d^


def same(a: List[UInt8], b: List[UInt8], n: Int, what: String) raises:
    expect(len(a) >= n and len(b) >= n, what + ": too short")
    for i in range(n):
        if a[i] != b[i]:
            raise Error("FAILED: " + what + ": byte " + String(i) + " is " + String(Int(a[i])) + ", wanted " + String(Int(b[i])))


# ── the C libraries ──────────────────────────────────────────────────────────────────────────────────

def c_zstd(src: List[UInt8], level: Int) raises -> List[UInt8]:
    var cap = external_call["ZSTD_compressBound", Int](len(src))
    var out = List[UInt8](length=cap + 1, fill=0)
    var n = external_call["ZSTD_compress", Int](out.unsafe_ptr(), cap, src.unsafe_ptr(), len(src), Int32(level))
    expect(external_call["ZSTD_isError", UInt32](n) == 0, "libzstd compresses")
    return List[UInt8](out[:n])


def c_unzstd(src: List[UInt8], n: Int) raises -> List[UInt8]:
    var out = List[UInt8](length=n + 1, fill=0)
    var got = external_call["ZSTD_decompress", Int](out.unsafe_ptr(), n, src.unsafe_ptr(), len(src))
    expect(external_call["ZSTD_isError", UInt32](got) == 0 and got == n, "libzstd decodes a frame of the Mojo encoder: got " + String(got) + ", wanted " + String(n))
    return out^


def c_snappy(src: List[UInt8]) raises -> List[UInt8]:
    var cap = external_call["snappy_max_compressed_length", Int](len(src))
    var out = List[UInt8](length=cap + 1, fill=0)
    var n = cap
    expect(external_call["snappy_compress", Int32](src.unsafe_ptr(), len(src), out.unsafe_ptr(), UnsafePointer(to=n)) == 0, "libsnappy compresses")
    return List[UInt8](out[:n])


def c_unsnappy(src: List[UInt8], n: Int) raises -> List[UInt8]:
    var out = List[UInt8](length=n + 1, fill=0)
    var got = n
    var rc = external_call["snappy_uncompress", Int32](src.unsafe_ptr(), len(src), out.unsafe_ptr(), UnsafePointer(to=got))
    expect(rc == 0 and got == n, "libsnappy decodes a page of the Mojo encoder")
    return out^


# ── what the encoder wrote ───────────────────────────────────────────────────────────────────────────

struct Seen(Movable):
    """Counts of what a frame holds: [raw, run, compressed blocks, Huffman 1 stream, Huffman 4 streams, raw literals, run literals,
    sequences in predefined mode, in repeated-symbol mode, in built-table mode]."""
    var c: List[Int]

    def __init__(out self):
        self.c = List[Int](length=10, fill=0)


def inspect(f: List[UInt8], mut seen: Seen) raises:
    var i = 10                                            # magic, descriptor, window, content size
    while True:
        var bh = Int(f[i]) | (Int(f[i + 1]) << 8) | (Int(f[i + 2]) << 16)
        i += 3
        var last = bh & 1
        var kind = (bh >> 1) & 3
        var size = bh >> 3
        if kind == 0:
            seen.c[0] += 1
            i += size
        elif kind == 1:
            seen.c[1] += 1
            i += 1
        else:
            seen.c[2] += 1
            var b0 = Int(f[i])
            var lk = b0 & 3
            var fmt = (b0 >> 2) & 3
            var j = i
            if lk <= 1:
                var sz = 0
                var hdr = 1
                if fmt == 1:
                    sz = (b0 >> 4) | (Int(f[i + 1]) << 4)
                    hdr = 2
                elif fmt == 3:
                    sz = (b0 >> 4) | (Int(f[i + 1]) << 4) | (Int(f[i + 2]) << 12)
                    hdr = 3
                else:
                    sz = b0 >> 3
                seen.c[5 if lk == 0 else 6] += 1
                j += hdr + (sz if lk == 0 else 1)
            else:
                var comp = 0
                var hdr = 3
                if fmt <= 1:
                    comp = (b0 | (Int(f[i + 1]) << 8) | (Int(f[i + 2]) << 16)) >> 14 & 1023
                    seen.c[3 if fmt == 0 else 4] += 1
                elif fmt == 2:
                    comp = (b0 | (Int(f[i + 1]) << 8) | (Int(f[i + 2]) << 16) | (Int(f[i + 3]) << 24)) >> 18 & 16383
                    hdr = 4
                    seen.c[4] += 1
                else:
                    comp = (b0 | (Int(f[i + 1]) << 8) | (Int(f[i + 2]) << 16) | (Int(f[i + 3]) << 24) | (Int(f[i + 4]) << 32)) >> 22 & 262143
                    hdr = 5
                    seen.c[4] += 1
                j += hdr + comp
            var nseq = Int(f[j])
            if nseq > 0:
                var modes = Int(f[j + 1 + (1 if nseq >= 128 and nseq < 255 else (2 if nseq == 255 else 0))])
                seen.c[7 + (modes >> 6)] += 1
                seen.c[7 + ((modes >> 4) & 3)] += 1
                seen.c[7 + ((modes >> 2) & 3)] += 1
            i += size
        if last == 1:
            break


# ── the tests ────────────────────────────────────────────────────────────────────────────────────────

def check_zstd(src: List[UInt8], what: String, mut seen: Seen) raises:
    var n = len(src)
    var c = zstd_compress(src)
    var back = zstd_decompress(c, n)
    same(back, src, n, what + ": Mojo encode, Mojo decode")
    same(c_unzstd(c, n), src, n, what + ": Mojo encode, libzstd decode")
    inspect(c, seen)
    for level in [1, 3, 9, 19]:
        var theirs = c_zstd(src, level)
        same(zstd_decompress(theirs, n), src, n, what + ": libzstd level " + String(level) + " encode, Mojo decode")


def check_snappy(src: List[UInt8], what: String) raises:
    var n = len(src)
    var c = snappy_compress(src)
    same(snappy_decompress(c, n), src, n, what + ": Mojo encode, Mojo decode")
    same(c_unsnappy(c, n), src, n, what + ": Mojo encode, libsnappy decode")
    same(snappy_decompress(c_snappy(src), n), src, n, what + ": libsnappy encode, Mojo decode")


def main() raises:
    var seen = Seen()
    for kind in range(KINDS):
        for n in [0, 1, 2, 3, 4, 5, 31, 32, 33, 100, 255, 256, 257, 1000, 5000, 65535, 65536, 131071, 131072, 131073, 300000, 1000003]:
            var src = make(kind, n)
            var what = "kind " + String(kind) + ", " + String(n) + " bytes"
            check_zstd(src, what, seen)
            check_snappy(src, what)
    # a block whose literals are one byte repeated, then copied: built by hand (a matcher that finds the copies leaves no such block
    # in ordinary data), so the run-literals path of both coders is tested
    var xs = List[UInt8](length=4, fill=120)
    var body = _literals_section(_ptr(Int(xs.unsafe_ptr())), 4)
    var one = _Seqs(List[Int](), List[Int](), List[Int](), List[UInt8]())
    one.ll.append(4)
    one.ml.append(4)
    one.ov.append(4 + 3)                                  # offset 4
    body.extend(_sequences_section(_ZEnc(), one))
    var frame_xs: List[UInt8] = [0x28, 0xB5, 0x2F, 0xFD, 0x80, 0x50, 8, 0, 0, 0]
    var bh = 1 | (2 << 1) | (len(body) << 3)
    frame_xs.append(UInt8(bh & 255))
    frame_xs.append(UInt8((bh >> 8) & 255))
    frame_xs.append(UInt8((bh >> 16) & 255))
    frame_xs.extend(body^)
    var eight = List[UInt8](length=8, fill=120)
    same(zstd_decompress(frame_xs, 8), eight, 8, "a run-literals block, Mojo decode")
    same(c_unzstd(frame_xs, 8), eight, 8, "a run-literals block, libzstd decode")
    inspect(frame_xs, seen)
    # the encoder really used what the ratio depends on
    expect(seen.c[0] > 0 and seen.c[1] > 0 and seen.c[2] > 0, "stored, run and compressed blocks all occur")
    expect(seen.c[3] > 0 and seen.c[4] > 0 and seen.c[5] > 0 and seen.c[6] > 0, "one- and four-stream Huffman literals, raw and run literals occur: " + String(seen.c[3]) + " " + String(seen.c[4]) + " " + String(seen.c[5]) + " " + String(seen.c[6]))
    expect(seen.c[7] > 0 and seen.c[8] > 0 and seen.c[9] > 0, "predefined, repeated-symbol and built sequence tables all occur")

    # 300 inputs made of pieces of every kind, repeated and damaged copies of their own earlier bytes, in all sizes
    var rs = UInt64(2463534242)
    for trial in range(300):
        var src = List[UInt8]()
        var pieces = 1 + Int(rs % 6)
        for _ in range(pieces):
            rs ^= rs << 13
            rs ^= rs >> 7
            rs ^= rs << 17
            var kind = Int(rs % UInt64(KINDS))
            var plen = Int((rs >> 8) % UInt64(1 << (4 + Int((rs >> 40) % 14))))
            if (rs >> 50) % 3 == 0 and len(src) > 0:                       # a damaged copy of earlier bytes
                var from_ = Int((rs >> 12) % UInt64(len(src)))
                for k in range(min(plen, len(src) - from_)):
                    var b = src[from_ + k]
                    src.append(b ^ UInt8(1 if (rs >> UInt64(k % 40)) % 29 == 0 else 0))
            else:
                src.extend(make(kind, plen))
        var what = "fuzz " + String(trial) + " (" + String(len(src)) + " bytes)"
        check_zstd(src, what, seen)
        check_snappy(src, what)
    # matches that reach back across 128 KB blocks, and past the 1 MB window
    for gap in [1000, 200000, 700000, 1048576, 1500000]:
        var half = make(RANDOM, gap)
        var src = List[UInt8]()
        src.extend(half.copy())
        src.extend(half.copy())
        src.extend(half.copy())
        var c = zstd_compress(src)
        same(c_unzstd(c, len(src)), src, len(src), "a random block of " + String(gap) + " repeated, libzstd decodes")
        same(zstd_decompress(c, len(src)), src, len(src), "a random block of " + String(gap) + " repeated")
        if gap <= 1000000:
            expect(len(c) < len(half) + len(half) // 20, "the repeats of a " + String(gap) + "-byte block cost almost nothing: " + String(len(c)))
        var sn = snappy_compress(src)
        same(snappy_decompress(sn, len(src)), src, len(src), "snappy, gap " + String(gap))

    # sizes: highly repetitive data shrinks to a sliver, structured data below its size, random data grows by a few bytes only
    var big = make(ONE_BYTE, 4000000)
    expect(len(zstd_compress(big)) < 200, "4 MB of one byte: " + String(len(zstd_compress(big))))
    expect(len(snappy_compress(big)) <= len(c_snappy(big)), "4 MB of one byte in snappy is no bigger than libsnappy's: " + String(len(snappy_compress(big))))
    var rnd = make(RANDOM, 1000000)
    expect(len(zstd_compress(rnd)) <= 1000000 + 40, "random data is stored: " + String(len(zstd_compress(rnd))))
    expect(len(snappy_compress(rnd)) <= 1000000 + 1000000 // 6 + 32, "random data in snappy: " + String(len(snappy_compress(rnd))))
    var ints = make(INT64S, 1000000)
    expect(len(zstd_compress(ints)) * 100 < 1000000 * 40, "growing 64-bit integers keep under 40%: " + String(len(zstd_compress(ints))))

    # a page of the wrong stated size, truncated or damaged input: an error or some bytes, never a crash
    var frame = zstd_compress(make(WORDS, 50000))
    var sn = snappy_compress(make(WORDS, 50000))
    var skew = c_zstd(make(SKEWED_255, 50000), 19)             # Huffman literals with FSE-coded weights, from libzstd
    var counting = c_zstd(make(INT64S, 50000), 1)
    var refused = 0
    var s = UInt64(12345)
    for trial in range(8000):
        s ^= s << 13
        s ^= s >> 7
        s ^= s << 17
        var f = frame.copy() if trial % 3 == 0 else (skew.copy() if trial % 3 == 1 else counting.copy())
        var g = sn.copy()
        var at = Int(s % UInt64(len(f)))
        var at2 = Int((s >> 20) % UInt64(len(g)))
        f[at] = f[at] ^ UInt8(1 << Int((s >> 40) & 7))
        g[at2] = g[at2] ^ UInt8(1 << Int((s >> 44) & 7))
        if trial % 4 == 1:
            f.resize(at, 0)
            g.resize(at2, 0)
        try:
            _ = zstd_decompress(f, 50000)
        except:
            refused += 1
        try:
            _ = snappy_decompress(g, 50000)
        except:
            refused += 1
    expect(refused > 2000, "damaged input is mostly refused: " + String(refused) + " of 16000")
    var wrong = False
    try:
        _ = zstd_decompress(frame, 49999)
    except:
        wrong = True
    expect(wrong, "a zstd frame is refused when the page size is not the one it states")
    wrong = False
    try:
        _ = snappy_decompress(sn, 49999)
    except:
        wrong = True
    expect(wrong, "a snappy page is refused when the page size is not the one it states")
    print("ok: zstd and snappy in Mojo; blocks stored/run/compressed", seen.c[0], seen.c[1], seen.c[2], "huffman 1/4 streams", seen.c[3], seen.c[4],
          "sequence tables predefined/rle/built", seen.c[7], seen.c[8], seen.c[9])
