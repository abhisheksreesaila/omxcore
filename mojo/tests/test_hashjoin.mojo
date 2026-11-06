# F-153: the hash join kernel (kernels_aot/hashjoin_cpu.mojo) against a plain dictionary, on every layout it chooses.
from std.collections import Dict
from omxcore.kernels.hashjoin_cpu import (JoinMap, hj_build, hj_probe_unique, hj_probe_pairs, hj_probe_block, MapView, hj_release, MODE_DIRECT, MODE_HASH,
                                          MODE_EMPTY, MODE_BITS, E_BUDGET, hj_probe_select, hj_filter_block, hj_probe_left_block, PM_SEMI, PM_ANTI, PM_ANTI_NA, PM_NONE)
from omxcore.kernels.hashjoin_cpu import IP, LP, BP


def expect(ok: Bool, what: String) raises:
    if not ok:
        raise Error("FAILED: " + what)


struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> UInt64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        var x = self.s
        x ^= x >> 33
        return x

    def below(mut self, n: Int) -> Int:
        return Int(self.next() % UInt64(n))


def make_keys(mut rng: Rng, n: Int, distinct: Int, base: Int64, step: Int64, unique: Bool) -> List[Int64]:
    var out = List[Int64](length=n, fill=0)
    if unique:
        # distinct keys in a shuffled order
        for i in range(n):
            out[i] = base + Int64(i) * step
        for i in range(n - 1, 0, -1):
            var j = rng.below(i + 1)
            var t = out[i]
            out[i] = out[j]
            out[j] = t
    else:
        for i in range(n):
            out[i] = base + Int64(rng.below(distinct)) * step
    return out^


def check(name: String, build: List[Int64], build_ok: List[UInt8], probe: List[Int64], probe_ok: List[UInt8], want_mode: Int, want_unique: Bool, threads: Int) raises:
    var nb = len(build)
    var np = len(probe)
    var truth = Dict[Int64, List[Int]]()
    for i in range(nb):
        if len(build_ok) > 0 and build_ok[i] == 0:
            continue
        if build[i] not in truth:
            truth[build[i]] = List[Int]()
        truth[build[i]].append(i)
    var bok = Int(build_ok.unsafe_ptr()) if len(build_ok) > 0 else 0
    var pok = Int(probe_ok.unsafe_ptr()) if len(probe_ok) > 0 else 0
    var m = hj_build(Int(build.unsafe_ptr()), bok, nb, threads)
    expect(m.mode == want_mode, name + ": layout " + m.kind() + " (wanted mode " + String(want_mode) + ")")
    if want_mode != MODE_EMPTY:
        expect(m.unique == want_unique, name + ": unique is " + String(m.unique))
    # pairs
    var out = List[Int64](length=4, fill=0)
    var rc = hj_probe_pairs(m, Int(probe.unsafe_ptr()), pok, np, threads, 1 << 40, Int(out.unsafe_ptr()))
    expect(rc == 0, name + ": probe_pairs rc " + String(rc))
    var total = 0
    for i in range(np):
        if len(probe_ok) > 0 and probe_ok[i] == 0:
            continue
        if probe[i] in truth:
            total += len(truth[probe[i]])
    expect(Int(out[0]) == total, name + ": pairs " + String(Int(out[0])) + " wanted " + String(total))
    var pr = IP(unsafe_from_address=Int(out[1]))
    var br = IP(unsafe_from_address=Int(out[2]))
    var at = 0
    for i in range(np):
        if len(probe_ok) > 0 and probe_ok[i] == 0:
            continue
        if probe[i] in truth:
            for b in truth[probe[i]]:
                expect(Int(pr[at]) == i and Int(br[at]) == b, name + ": pair " + String(at) + " is (" + String(Int(pr[at])) + ", " + String(Int(br[at])) + ") wanted (" + String(i) + ", " + String(b) + ")")
                at += 1
    expect(at == total, name + ": pairs walked")
    hj_release(Int(out[1]))
    hj_release(Int(out[2]))
    # a budget below the count refuses without allocating
    if total > 0:
        var small = List[Int64](length=4, fill=0)
        var rc2 = hj_probe_pairs(m, Int(probe.unsafe_ptr()), pok, np, threads, total - 1, Int(small.unsafe_ptr()))
        expect(rc2 == E_BUDGET and Int(small[0]) == total and Int(small[1]) == 0, name + ": budget refusal")
    # outer pairs (a LEFT join): a row with no partner or no usable key is one pair with build row -1
    var oo = List[Int64](length=4, fill=0)
    var rco = hj_probe_pairs(m, Int(probe.unsafe_ptr()), pok, np, threads, 1 << 40, Int(oo.unsafe_ptr()), True)
    expect(rco == 0, name + ": outer pairs rc " + String(rco))
    var want_pr = List[Int]()
    var want_br = List[Int]()
    for i in range(np):
        var usable = not (len(probe_ok) > 0 and probe_ok[i] == 0)
        if usable and probe[i] in truth:
            for b in truth[probe[i]]:
                want_pr.append(i)
                want_br.append(b)
        else:
            want_pr.append(i)
            want_br.append(-1)
    expect(Int(oo[0]) == len(want_pr), name + ": outer pairs " + String(Int(oo[0])) + " wanted " + String(len(want_pr)))
    if len(want_pr) > 0:
        var opr = IP(unsafe_from_address=Int(oo[1]))
        var obr = IP(unsafe_from_address=Int(oo[2]))
        for k in range(len(want_pr)):
            expect(Int(opr[k]) == want_pr[k] and Int(obr[k]) == want_br[k], name + ": outer pair " + String(k))
    hj_release(Int(oo[1]))
    hj_release(Int(oo[2]))
    # presence filters (on the regular map and on the presence-only map): semi, anti, null-aware anti, none; whole probe side and the fused block form over every second row
    for pres in [False, True]:
        var mm = hj_build(Int(build.unsafe_ptr()), bok, nb, threads, pres)
        for fm in [PM_SEMI, PM_ANTI, PM_ANTI_NA, PM_NONE]:
            var want_rows = List[Int]()
            for i in range(np):
                var usable = not (len(probe_ok) > 0 and probe_ok[i] == 0)
                var present = usable and probe[i] in truth
                var keep = False
                if fm == PM_SEMI:
                    keep = present
                elif fm == PM_ANTI:
                    keep = not present
                elif fm == PM_ANTI_NA:
                    keep = usable and not present
                if keep:
                    want_rows.append(i)
            var so = List[Int64](length=2, fill=0)
            _ = hj_probe_select(mm, Int(probe.unsafe_ptr()), pok, np, threads, fm, Int(so.unsafe_ptr()))
            expect(Int(so[0]) == len(want_rows), name + ": select(pres=" + String(pres) + ") mode " + String(fm) + " kept " + String(Int(so[0])) + " wanted " + String(len(want_rows)))
            expect((Int(so[1]) == 0) == (len(want_rows) == 0), name + ": select(pres=" + String(pres) + ") mode " + String(fm) + " allocates only when rows are kept")
            if len(want_rows) > 0:
                var got = IP(unsafe_from_address=Int(so[1]))
                for k in range(len(want_rows)):
                    expect(Int(got[k]) == want_rows[k], name + ": select(pres=" + String(pres) + ") mode " + String(fm) + " row " + String(k))
            hj_release(Int(so[1]))
            var bsel = List[Int32](length=np + 1, fill=0)
            var bn = 0
            for i in range(0, np, 2):
                bsel[bn] = Int32(i)
                bn += 1
            var bkept = hj_filter_block(MapView(mm), LP(unsafe_from_address=Int(probe.unsafe_ptr())), BP(unsafe_from_address=pok if pok != 0 else Int(probe.unsafe_ptr())),
                                        pok != 0, 0, IP(unsafe_from_address=Int(bsel.unsafe_ptr())), bn, fm)
            var bq = 0
            for k in range(len(want_rows)):
                if want_rows[k] % 2 == 0:
                    expect(Int(bsel[bq]) == want_rows[k], name + ": filter block(pres=" + String(pres) + ") mode " + String(fm) + " row " + String(want_rows[k]))
                    bq += 1
            expect(bkept == bq, name + ": filter block(pres=" + String(pres) + ") mode " + String(fm) + " kept " + String(bkept) + " wanted " + String(bq))
    # unique lookups
    if want_unique and want_mode != MODE_EMPTY:
        var res = List[Int32](length=np, fill=0)
        hj_probe_unique(m, Int(probe.unsafe_ptr()), pok, np, Int(res.unsafe_ptr()), threads)
        for i in range(np):
            var want = -1
            if not (len(probe_ok) > 0 and probe_ok[i] == 0) and probe[i] in truth:
                want = truth[probe[i]][0]
            expect(Int(res[i]) == want, name + ": unique lookup " + String(i) + ": " + String(Int(res[i])) + " wanted " + String(want))
        # the fused form: every second row selected
        var sel = List[Int32](length=np, fill=0)
        var nsel = 0
        for i in range(0, np, 2):
            sel[nsel] = Int32(i)
            nsel += 1
        var prow = List[Int32](length=np, fill=0)
        var kept = hj_probe_block(MapView(m), LP(unsafe_from_address=Int(probe.unsafe_ptr())), BP(unsafe_from_address=pok if pok != 0 else Int(probe.unsafe_ptr())),
                                  pok != 0, 0, IP(unsafe_from_address=Int(sel.unsafe_ptr())), nsel, IP(unsafe_from_address=Int(prow.unsafe_ptr())))
        var q = 0
        for i in range(0, np, 2):
            var want = -1
            if not (len(probe_ok) > 0 and probe_ok[i] == 0) and probe[i] in truth:
                want = truth[probe[i]][0]
            if want >= 0:
                expect(Int(sel[q]) == i and Int(prow[q]) == want, name + ": block lookup of row " + String(i))
                q += 1
        expect(kept == q, name + ": block kept " + String(kept) + " wanted " + String(q))
        # the same by row: the partner of block row r is prow[r], whatever later compaction of sel does
        var sel2 = List[Int32](length=np, fill=0)
        var nsel2 = 0
        for i in range(0, np, 2):
            sel2[nsel2] = Int32(i)
            nsel2 += 1
        var prow2 = List[Int32](length=np, fill=-5)
        var kept2 = hj_probe_block(MapView(m), LP(unsafe_from_address=Int(probe.unsafe_ptr())), BP(unsafe_from_address=pok if pok != 0 else Int(probe.unsafe_ptr())),
                                   pok != 0, 0, IP(unsafe_from_address=Int(sel2.unsafe_ptr())), nsel2, IP(unsafe_from_address=Int(prow2.unsafe_ptr())), True)
        expect(kept2 == kept, name + ": by-row block kept " + String(kept2))
        for t in range(kept2):
            var row = Int(sel2[t])
            expect(Int(prow2[row]) == truth[probe[row]][0], name + ": by-row partner of row " + String(row))
        # the LEFT form: every row stays, its partner or -1
        var lsel = List[Int32](length=np, fill=0)
        for i in range(np):
            lsel[i] = Int32(i)
        var lrow = List[Int32](length=np, fill=7)
        hj_probe_left_block(MapView(m), LP(unsafe_from_address=Int(probe.unsafe_ptr())), BP(unsafe_from_address=pok if pok != 0 else Int(probe.unsafe_ptr())),
                            pok != 0, 0, IP(unsafe_from_address=Int(lsel.unsafe_ptr())), np, IP(unsafe_from_address=Int(lrow.unsafe_ptr())))
        expect(Int(lsel[np - 1]) == np - 1, name + ": left block leaves every row selected")
        for i in range(np):
            var want = -1
            if not (len(probe_ok) > 0 and probe_ok[i] == 0) and probe[i] in truth:
                want = truth[probe[i]][0]
            expect(Int(lrow[i]) == want, name + ": left block row " + String(i) + ": " + String(Int(lrow[i])) + " wanted " + String(want))
    print("  ok", name, "->", m.kind())


def check_bits(name: String, build: List[Int64], build_ok: List[UInt8], probe: List[Int64], probe_ok: List[UInt8], want_bits: Bool, threads: Int) raises:
    """F-207: `hj_build(bits=True)`: unique keys over a range under 2^26 are one bit each (MODE_BITS), anything else builds as usual; every lookup form agrees with
    a dictionary on which probe rows have a partner."""
    var nb = len(build)
    var np = len(probe)
    var truth = Dict[Int64, Int]()
    for i in range(nb):
        if len(build_ok) > 0 and build_ok[i] == 0:
            continue
        truth[build[i]] = i
    var bok = Int(build_ok.unsafe_ptr()) if len(build_ok) > 0 else 0
    var pok = Int(probe_ok.unsafe_ptr()) if len(probe_ok) > 0 else 0
    var m = hj_build(Int(build.unsafe_ptr()), bok, nb, threads, False, True)
    expect((m.mode == MODE_BITS) == want_bits, name + ": layout " + m.kind())
    var present = List[Bool]()
    var n_present = 0
    for i in range(np):
        var hit = not (len(probe_ok) > 0 and probe_ok[i] == 0) and probe[i] in truth
        present.append(hit)
        n_present += 1 if hit else 0
    # unique lookups: a partner or -1 (the bits map names no row, any non-negative number)
    if m.unique:
        var res = List[Int32](length=np, fill=-9)
        hj_probe_unique(m, Int(probe.unsafe_ptr()), pok, np, Int(res.unsafe_ptr()), threads)
        for i in range(np):
            expect((Int(res[i]) >= 0) == present[i], name + ": unique lookup " + String(i))
        var sel = List[Int32](length=np, fill=0)
        for i in range(np):
            sel[i] = Int32(i)
        var prow = List[Int32](length=np, fill=-5)
        var kept = hj_probe_block(MapView(m), LP(unsafe_from_address=Int(probe.unsafe_ptr())), BP(unsafe_from_address=pok if pok != 0 else Int(probe.unsafe_ptr())),
                                  pok != 0, 0, IP(unsafe_from_address=Int(sel.unsafe_ptr())), np, IP(unsafe_from_address=Int(prow.unsafe_ptr())), True)
        expect(kept == n_present, name + ": block kept " + String(kept) + " wanted " + String(n_present))
        var q = 0
        for i in range(np):
            if present[i]:
                expect(Int(sel[q]) == i and Int(prow[i]) >= 0, name + ": block row " + String(q))
                q += 1
        # semi / anti over the whole probe side and in blocks
        for fm in [PM_SEMI, PM_ANTI]:
            var so = List[Int64](length=2, fill=0)
            _ = hj_probe_select(m, Int(probe.unsafe_ptr()), pok, np, threads, fm, Int(so.unsafe_ptr()))
            expect(Int(so[0]) == (n_present if fm == PM_SEMI else np - n_present), name + ": select mode " + String(fm))
            hj_release(Int(so[1]))
        var lsel = List[Int32](length=np, fill=0)
        for i in range(np):
            lsel[i] = Int32(i)
        var lrow = List[Int32](length=np, fill=7)
        hj_probe_left_block(MapView(m), LP(unsafe_from_address=Int(probe.unsafe_ptr())), BP(unsafe_from_address=pok if pok != 0 else Int(probe.unsafe_ptr())),
                            pok != 0, 0, IP(unsafe_from_address=Int(lsel.unsafe_ptr())), np, IP(unsafe_from_address=Int(lrow.unsafe_ptr())))
        expect(Int(lsel[np - 1]) == np - 1, name + ": left block leaves every row selected")
        for i in range(np):
            expect((Int(lrow[i]) >= 0) == present[i], name + ": left block row " + String(i))
    print("  ok", name, "->", m.kind())


def main() raises:
    var rng = Rng(12345)
    var none_b = List[UInt8]()
    var none_p = List[UInt8]()
    for threads in [1, 4]:
        # direct, unique (dense primary keys), probe with hits and misses
        var b1 = make_keys(rng, 5000, 0, 100, 1, True)
        var p1 = make_keys(rng, 20000, 6500, 0, 1, False)
        check("dense unique t" + String(threads), b1, none_b, p1, none_p, MODE_DIRECT, True, threads)
        # direct, repeated (many to many)
        var b2 = make_keys(rng, 8000, 700, -50, 2, False)
        var p2 = make_keys(rng, 9000, 900, -60, 2, False)
        check("dense repeated t" + String(threads), b2, none_b, p2, none_p, MODE_DIRECT, False, threads)
        # hash, unique: keys spread over a huge range
        var b3 = make_keys(rng, 6000, 0, -4000000000000, 1000003, True)
        var p3 = List[Int64]()
        for i in range(15000):
            p3.append(b3[rng.below(6000)] + Int64(rng.below(2)))
        check("sparse unique t" + String(threads), b3, none_b, p3, none_p, MODE_HASH, True, threads)
        # hash, repeated, enough rows for several partitions and the bitmap
        var b4 = make_keys(rng, 60000, 9000, 1000000, 100003, False)
        var p4 = make_keys(rng, 50000, 12000, 1000000, 100003, False)
        check("sparse repeated t" + String(threads), b4, none_b, p4, none_p, MODE_HASH, False, threads)
        # extreme int64 keys (past 2^53) stay exact
        var b5 = List[Int64]()
        b5.append(Int64.MAX)
        b5.append(Int64.MAX - 1)
        b5.append(Int64.MIN)
        b5.append(9007199254740993)
        b5.append(9007199254740992)
        b5.append(0)
        var p5 = List[Int64]()
        var cands: List[Int64] = [Int64(9007199254740993), Int64(9007199254740992), Int64(9007199254740994), Int64.MAX, Int64.MAX - 1, Int64.MIN + 1, Int64.MIN, Int64(1)]
        for k in cands:
            p5.append(k)
        check("past 2^53 t" + String(threads), b5, none_b, p5, none_p, MODE_HASH, True, threads)
        # a hot key: half the build rows share one key
        var b6 = make_keys(rng, 20000, 500, 0, 1, False)
        for i in range(0, 20000, 2):
            b6[i] = 7
        var p6 = make_keys(rng, 3000, 600, 0, 1, False)
        check("hot key t" + String(threads), b6, none_b, p6, none_p, MODE_DIRECT, False, threads)
        # unusable rows on both sides (NULL keys never match)
        var b7 = make_keys(rng, 7000, 900, 10, 1, False)
        var bok7 = List[UInt8](length=7000, fill=1)
        for i in range(0, 7000, 5):
            bok7[i] = 0
        var p7 = make_keys(rng, 7000, 900, 10, 1, False)
        var pok7 = List[UInt8](length=7000, fill=1)
        for i in range(0, 7000, 3):
            pok7[i] = 0
        check("null keys t" + String(threads), b7, bok7, p7, pok7, MODE_DIRECT, False, threads)
        # nothing to build
        var b8 = make_keys(rng, 100, 10, 0, 1, False)
        var bok8 = List[UInt8](length=100, fill=0)
        check("empty build t" + String(threads), b8, bok8, make_keys(rng, 50, 10, 0, 1, False), none_p, MODE_EMPTY, True, threads)
        # one row each
        var one = List[Int64]()
        one.append(42)
        check("one row t" + String(threads), one, none_b, one, none_p, MODE_DIRECT, True, threads)
    for threads in [1, 4]:
        # F-207 bits: unique keys spread over 140,000 values (too sparse for a direct array at 8x rows is not needed: bits win either way), NULL build and probe keys
        var bb = make_keys(rng, 20000, 0, 1000, 7, True)
        var bbok = List[UInt8](length=20000, fill=1)
        for i in range(0, 20000, 9):
            bbok[i] = 0
        var pb = make_keys(rng, 30000, 22000, 990, 3, False)
        var pbok = List[UInt8](length=30000, fill=1)
        for i in range(0, 30000, 4):
            pbok[i] = 0
        check_bits("bits unique t" + String(threads), bb, bbok, pb, pbok, True, threads)
        check_bits("bits unique, no NULL keys t" + String(threads), bb, none_b, pb, none_p, True, threads)
        # a repeated key among the usable rows: not unique, built as usual
        var br = make_keys(rng, 20000, 5000, 1000, 7, False)
        check_bits("bits, repeated keys t" + String(threads), br, none_b, pb, none_p, False, threads)
        # a repeat only on a row that is not usable does not count
        var bn = make_keys(rng, 5000, 0, 100, 3, True)
        var bnok = List[UInt8](length=5000, fill=1)
        bn.append(bn[0])
        bnok.append(0)
        check_bits("bits, a repeat on an unusable row t" + String(threads), bn, bnok, pb, none_p, True, threads)
        # a range past 2^26 values: the bitmap would be too big, the usual layout
        var bw = make_keys(rng, 3000, 0, -4000000000000, 1000003, True)
        check_bits("bits, a wide range t" + String(threads), bw, none_b, bw, none_p, False, threads)
        # extreme keys: the span overflows, the usual layout
        var bx = List[Int64]()
        bx.append(Int64.MAX)
        bx.append(Int64.MIN)
        check_bits("bits, extreme keys t" + String(threads), bx, none_b, bx, none_p, False, threads)
        # one key: a one-bit map
        var b1k = List[Int64]()
        b1k.append(42)
        check_bits("bits, one key t" + String(threads), b1k, none_b, b1k, none_p, True, threads)
    print("ok test_hashjoin")
