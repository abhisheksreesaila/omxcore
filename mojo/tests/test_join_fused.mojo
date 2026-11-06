# F-204: the builder's join runs the way SQL's does. `QueryResult.notes` say which path (fused: no joined row, only the build-side columns the plan reads;
# materialized: the rows are written, only the columns the plan reads), a Frame is joined as its edits leave it, and every answer is a double loop's.
# (tests/test_f204_builder_join.py compares the builder, SQL over tables and SQL over Frames with DuckDB on random worlds.)
from omxcore import Table, SharedTable, Frame, INT, Query, QueryResult, col, sum, count, count_distinct, max_of, GPU


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


def has_note(r: QueryResult, text: String) -> Bool:
    for n in r.notes:
        if text in n:
            return True
    return False


def facts(n: Int) raises -> Table:
    # id, k (every 11th NULL, 0 .. 49), g (0 .. 3), pad0 .. pad5 (carried by nobody)
    var t = Table.empty()
    var id = t.add_column("id", INT)
    var k = t.add_column("k", INT)
    var g = t.add_column("g", INT)
    var pad = List[Int]()
    for p in range(6):
        pad.append(t.add_column("pad" + String(p), INT))
    for i in range(n):
        t.append(id, Int64(i))
        if i % 11 == 10:
            t.append_null(k)
        else:
            t.append(k, Int64((i * 7) % 50))
        t.append(g, Int64(i % 4))
        for p in range(6):
            t.append(pad[p], Int64(i + p))
    return t^


def dims(copies: Int, hot: Int) raises -> Table:
    # dk 0 .. 39 (`copies` rows each; key 5 has `hot` rows), w = dk * 10 + copy, pad0 .. pad5
    var t = Table.empty()
    var dk = t.add_column("dk", INT)
    var w = t.add_column("w", INT)
    var pad = List[Int]()
    for p in range(6):
        pad.append(t.add_column("pad" + String(p), INT))
    for key in range(40):
        var rows = hot if key == 5 else copies
        for c in range(rows):
            t.append(dk, Int64(key))
            t.append(w, Int64(key * 10 + c))
            for p in range(6):
                t.append(pad[p], Int64(c))
    return t^


def main() raises:
    var n = 3000
    var left = facts(n)^.share()
    # (1) unique keys: fused, one build-side column gathered, by hand
    var right1 = dims(1, 1)^.share()
    var by_hand_count = 0
    var by_hand_sum = 0
    for i in range(n):
        if i % 11 == 10:
            continue
        var key = (i * 7) % 50
        if key < 40:
            by_hand_count += 1
            by_hand_sum += key * 10
    var r1 = Query(left).join(right1, ["k"], ["dk"]).agg(count(), sum(col("w"))).compute()
    expect(r1.counts[0] == by_hand_count and r1.values(1)[0] == by_hand_sum, "unique keys: " + String(r1.counts[0]) + " " + String(r1.values(1)[0]))
    expect(has_note(r1, "join path: fused: 3,000 probe rows look their partner up in blocks of 1,024 and 1 build-side column is gathered"), "fused, one column: " + String(r1.notes[len(r1.notes) - 1]))
    # (2) a key that repeats (3 copies, key 5 has 9): still fused, rows expanded in the front stage
    var right2 = dims(3, 9)^.share()
    var rows2 = 0
    var sum2 = 0
    for i in range(n):
        if i % 11 == 10:
            continue
        var key = (i * 7) % 50
        if key < 40:
            var copies = 9 if key == 5 else 3
            rows2 += copies
            for c in range(copies):
                sum2 += key * 10 + c
    var r2 = Query(left).join(right2, ["k"], ["dk"]).group_by(col("g"), 4).agg(count(), sum(col("w"))).compute()
    var total = 0
    var wsum = 0
    for g in range(4):
        total += r2.counts[g]
        wsum += r2.values(1)[g]
    expect(total == rows2 and wsum == sum2, "repeated keys: " + String(total) + " wanted " + String(rows2))
    expect(has_note(r2, "join path: fused") and has_note(r2, "expanded to its partners"), "repeated keys fuse and expand")
    # (3) a hot key (200 rows): the front stage would need 200 rounds, so the rows are written, and only the columns the plan reads (2 of 14)
    var right3 = dims(1, 200)^.share()
    var rows3 = 0
    for i in range(n):
        if i % 11 == 10:
            continue
        var key = (i * 7) % 50
        if key < 40:
            rows3 += 200 if key == 5 else 1
    var r3 = Query(left).join(right3, ["k"], ["dk"]).agg(count(), max_of(col("w"))).compute()
    expect(r3.counts[0] == rows3, "hot key: " + String(r3.counts[0]) + " wanted " + String(rows3))
    expect(has_note(r3, "join path: materialized") and has_note(r3, "gathered into one table of 1 columns") and has_note(r3, "repeat"), "hot key is materialized with one column: " + String(r3.notes[len(r3.notes) - 1]))
    # (4) a holistic aggregate needs every joined row: materialized, the answer is the distinct count of w
    var r4 = Query(left).join(right1, ["k"], ["dk"]).agg(count_distinct(col("w"))).compute()
    expect(r4.values(0)[0] == 40 and has_note(r4, "a holistic aggregate"), "count_distinct after a join: " + String(r4.values(0)[0]))
    # (5) LEFT, semi, anti: notes name the rule
    var r5 = Query(left).join(right1, ["k"], ["dk"], "left").agg(count()).compute()
    expect(r5.counts[0] == n and has_note(r5, "a row without a partner stays"), "left join keeps every row")
    var r6 = Query(left).join(right1, ["k"], ["dk"], "semi").agg(count()).compute()
    var r7 = Query(left).join(right1, ["k"], ["dk"], "anti").agg(count()).compute()
    expect(r6.counts[0] == by_hand_count and r6.counts[0] + r7.counts[0] == n, "semi and anti split the rows")
    expect(has_note(r6, "semi join: no build-side column is gathered") and has_note(r7, "anti join: no build-side column is gathered"), "semi / anti notes")
    # (6) a Frame on either side: its edits are joined (set one key, remove a row, append a row; the dimension loses key 7 and gains a second row of key 1)
    var lf = Frame(left)
    lf.set(lf.column("k"), 0, Int64(7))                       # row 0: key 0 -> 7
    lf.remove(1)                                              # row 1 (key 7) gone
    _ = lf.append(["3000", "1", "2", "0", "0", "0", "0", "0", "0"])      # key 1, group 2
    var rf = Frame(right1)
    rf.remove(7)                                              # key 7 (dim row 7) gone
    _ = rf.append(["1", "5000", "0", "0", "0", "0", "0", "0"])   # a second partner of key 1: w = 5000
    var f_count = 0
    var f_sum = 0
    for i in range(n + 1):
        if i == 1 or (i < n and i % 11 == 10):
            continue
        var key = 1 if i == n else ((i * 7) % 50 if i != 0 else 7)
        if key >= 40 or key == 7:
            continue
        f_count += 1
        f_sum += key * 10
        if key == 1:
            f_count += 1
            f_sum += 5000
    var r8 = Query(lf).join(rf, ["k"], ["dk"]).agg(count(), sum(col("w"))).compute()
    expect(r8.counts[0] == f_count and r8.values(1)[0] == f_sum, "two Frames: " + String(r8.counts[0]) + " " + String(r8.values(1)[0]) + " wanted " + String(f_count) + " " + String(f_sum))
    expect(has_note(r8, "join path: fused"), "two Frames join fused")
    # (6b) a formula column of a Frame is a column of the join
    var ff = Frame(left)
    _ = ff.formula("g10 = g * 10")
    var g10_sum = 0
    for i in range(n):
        if i % 11 != 10 and (i * 7) % 50 < 40:
            g10_sum += (i % 4) * 10
    var r9 = Query(ff).join(right1, ["k"], ["dk"]).agg(count(), sum(col("g10"))).compute()
    expect(r9.counts[0] == by_hand_count and r9.values(1)[0] == g10_sum, "a formula column through the join: " + String(r9.values(1)[0]) + " wanted " + String(g10_sum))
    # (7) the joined table can still be written
    var jt = Query(left).join(right1, ["k"], ["dk"]).joined_table()
    expect(jt.rows() == by_hand_count and jt.columns() == 9 + 8 and jt.name(9) == "dk" and jt.name(10) == "w" and jt.name(11) == "pad0_1", "joined_table: " + String(jt.rows()) + " " + jt.name(11))
    # (7b) a second join on the first one's rows (the first is written, the second runs fused): w = key * 10 finds r2's row of that key, z = key
    var r2t = Table.empty()
    var r2k = r2t.add_column("rk", INT)
    var r2z = r2t.add_column("z", INT)
    for key in range(40):
        r2t.append(r2k, Int64(key * 10))
        r2t.append(r2z, Int64(key))
    var chain_sum = 0
    for i in range(n):
        if i % 11 != 10 and (i * 7) % 50 < 40:
            chain_sum += (i * 7) % 50
    var rc = Query(left).join(right1, ["k"], ["dk"]).join(r2t^.share(), ["w"], ["rk"]).agg(count(), sum(col("z"))).compute()
    expect(rc.counts[0] == by_hand_count and rc.values(1)[0] == chain_sum, "two joins in a row: " + String(rc.counts[0]) + " " + String(rc.values(1)[0]) + " wanted " + String(by_hand_count) + " " + String(chain_sum))
    expect(has_note(rc, "join path: fused"), "the second join of a chain is fused")
    # (7c) two keys whose ranges multiply past 2^62 are hashed tuples, verified per pair: written, not fused, and still right
    var hl = Table.empty()
    var h1 = hl.add_column("a", INT)
    var h2 = hl.add_column("b", INT)
    var hr = Table.empty()
    var g1 = hr.add_column("a", INT)
    var g2 = hr.add_column("b", INT)
    var gv = hr.add_column("v", INT)
    var big = Int64(1) << 40
    for i in range(500):
        hl.append(h1, Int64(i % 50) * big)
        hl.append(h2, Int64(i % 7) * big + Int64(i % 3))
    for i in range(100):
        hr.append(g1, Int64(i % 50) * big)
        hr.append(g2, Int64(i % 7) * big + Int64(i % 3))
        hr.append(gv, Int64(i))
    var hwant = 0
    var hsum = 0
    for i in range(500):
        for j in range(100):
            if i % 50 == j % 50 and i % 7 == j % 7 and i % 3 == j % 3:
                hwant += 1
                hsum += j
    var rh = Query(hl^.share()).join(hr^.share(), ["a", "b"], ["a", "b"]).agg(count(), sum(col("v"))).compute()
    expect(rh.counts[0] == hwant and rh.values(1)[0] == hsum and hwant > 0, "hashed keys: " + String(rh.counts[0]) + " wanted " + String(hwant))
    expect(has_note(rh, "join path: materialized") and has_note(rh, "hashed tuples"), "hashed keys are written, with the reason")
    # (8) an empty side: nothing matches (inner, semi), everything is unmatched (left, anti); GPU asked for: refused by name, nothing written
    var empty = Table.empty()
    _ = empty.add_column("dk", INT)
    _ = empty.add_column("w", INT)
    var none = empty^.share()
    var e1 = Query(left).join(none, ["k"], ["dk"]).agg(count(), sum(col("w"))).compute()
    expect(e1.counts[0] == 0 and not e1.present(1, 0), "an empty right side, inner")
    var e2 = Query(left).join(none, ["k"], ["dk"], "left").agg(count(), sum(col("w"))).compute()
    expect(e2.counts[0] == n and not e2.present(1, 0), "an empty right side, left")
    var e3 = Query(left).join(none, ["k"], ["dk"], "anti").agg(count()).compute()
    expect(e3.counts[0] == n, "an empty right side, anti keeps every row")
    var e4 = Query(none).join(left, ["dk"], ["k"]).agg(count()).compute()
    expect(e4.counts[0] == 0, "an empty left side")
    var refused = False
    try:
        _ = Query(left).join(right1, ["k"], ["dk"]).agg(count()).compute(GPU)
    except e:
        refused = String(e) == "device GPU: a Query that starts with a join runs on the CPU (the join is fused into the CPU pipeline; there is no DeviceTable of the joined rows)"
    expect(refused, "compute(GPU) after a join is refused with the join-specific text")
    print("ok test_join_fused")
