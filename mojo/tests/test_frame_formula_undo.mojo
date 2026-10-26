# F-179: adding, replacing and removing a formula column is a step in the Frame's single edit log, next to cell edits,
# appends, removes and merges: undo and redo walk it in either direction and give back the exact earlier state (formula
# texts and positions, the values that read them), on a table, an edited frame and a Parquet scan frame.
from omxcore import (Table, SharedTable, Frame, Query, col, sum, INT, DECIMAL, TEXT, write_parquet, read_parquet, scan_parquet, ZSTD, frame_snapshot)


struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed * 0x9E3779B97F4A7C15 + 1

    def next(mut self) -> UInt64:
        self.s ^= self.s << 13
        self.s ^= self.s >> 7
        self.s ^= self.s << 17
        return self.s

    def int(mut self, lo: Int, hi: Int) -> Int:       # [lo, hi]
        return lo + Int(self.next() % UInt64(hi - lo + 1))


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


comptime N = 2_000
comptime GROUP = 704             # three row groups for the scan frame


def make_table() raises -> SharedTable:
    var t = Table.empty()
    var g = t.add_column("region", TEXT)
    var r = t.add_column("revenue", DECIMAL, scale=2)
    var k = t.add_column("cost", DECIMAL, scale=2)
    var q = t.add_column("qty", INT)
    var words: List[String] = ["West", "East", "North"]
    var rng = Rng(3)
    for i in range(N):
        t.append_text(g, words[rng.int(0, 2)])
        if i % 17 == 3:
            t.append_null(r)
        else:
            t.append(r, Int64(rng.int(-500_00, 2_000_00)))
        t.append(k, Int64(rng.int(0, 900_00)))
        t.append(q, Int64(rng.int(1, 9)))
    return t^.share()


def source(kind: Int) raises -> Frame:
    """0: a Frame over the table; 1: the same with edits made before the formula steps; 2: a Frame over a Parquet file."""
    if kind == 2:
        write_parquet("f179_scan.parquet", make_table(), "", ZSTD, GROUP)
        return Frame(scan_parquet("f179_scan.parquet"))
    var f = Frame(make_table())
    if kind == 1:
        f.set(f.column("revenue"), 4, 123_45)
        f.set_text(f.column("region"), 6, "Central")
        f.remove(8)
    return f^


comptime STEPS = 12


def step(mut f: Frame, i: Int) raises:
    """The script both the frame under test and its reference follow: cell edits and formula steps interleaved."""
    if i == 0:
        f.set(f.column("revenue"), 3, 777_00)
    elif i == 1:
        _ = f.formula("margin = revenue - cost")
    elif i == 2:
        f.set(f.column("cost"), 5, 11_00)
    elif i == 3:
        _ = f.formula("double = margin * 2")                       # a formula over a formula
    elif i == 4:
        _ = f.formula("margin = revenue * qty")                    # replaces: `double` now reads the new one
    elif i == 5:
        _ = f.append(["East", "50.00", "20.00", "4"])
    elif i == 6:
        f.set_null(f.column("revenue"), 7)
    elif i == 7:
        f.remove_formula(f.column("double"))                       # the first of the two, so `margin` moves nothing; `double` goes
    elif i == 8:
        _ = f.formula("tax = revenue / 4")
    elif i == 9:
        f.remove(9)
    elif i == 10:
        f.remove_formula(f.column("margin"))                       # the middle one: `tax` moves up
    else:
        f.set_text(f.column("qty"), 2, "9")


def mix(h: UInt64, v: Int64) -> UInt64:
    return (h ^ UInt64(v)) * 0x100000001B3 + 0x9E3779B97F4A7C15


def fingerprint(f: Frame) raises -> UInt64:
    """Everything a person sees: the columns and their formulas, every live cell, and a query over each formula."""
    var h: UInt64 = 14695981039346656037
    h = mix(h, Int64(f.columns()))
    for c in range(f.columns()):
        for b in f.name(c).as_bytes():
            h = mix(h, Int64(b))
        h = mix(h, Int64(f.scale(c)))
        if c >= f.stored():
            for b in f.formula_text(c).as_bytes():
                h = mix(h, Int64(b))
    for c in range(f.stored()):
        h = mix(h, Int64(f.word_count(c)))
    for r in f.changed_rows():
        h = mix(h, Int64(r))
    for row in range(f.rows()):
        if not f.alive(row):
            h = mix(h, -1)
            continue
        for c in range(f.stored()):
            h = mix(h, Int64(1) if f.edited(c, row) else Int64(0))
        for c in range(f.columns()):
            if f.is_null(c, row):
                h = mix(h, -2)
            else:
                h = mix(h, f.get(c, row))
    for c in range(f.stored(), f.columns()):
        var r = Query(f).agg(sum(col(f.name(c)))).compute()
        h = mix(h, Int64(r.values(0)[0]))
    var q = Query(f).group_by(col("qty"), groups=16).agg(sum(col("revenue")), sum(col("cost"))).compute()
    for g in range(len(q.counts)):
        h = mix(h, Int64(q.counts[g]))
        h = mix(h, Int64(q.values(0)[g]))
        h = mix(h, Int64(q.values(1)[g]))
    return h


def reference(kind: Int, steps: Int) raises -> Frame:
    var f = source(kind)
    for i in range(steps):
        step(f, i)
    return f^


def test_every_step_undoes_and_redoes_to_the_exact_earlier_state(kind: Int) raises:
    var want = List[UInt64]()
    for k in range(STEPS + 1):
        want.append(fingerprint(reference(kind, k)))
    for k in range(STEPS):
        expect(want[k] != want[k + 1], "kind " + String(kind) + ": step " + String(k) + " changes what is seen")
    var f = source(kind)
    for i in range(STEPS):
        step(f, i)
    for k in range(STEPS, 0, -1):
        var before = f.edits
        expect(f.undo(), "undo step " + String(k))
        expect(f.edits > before, "the edit counter moves on undo")
        expect(fingerprint(f) == want[k - 1], "kind " + String(kind) + ": undone to " + String(k - 1) + " steps")
    var pre = f.undo()                                             # kind 1 has its own edits below the script
    expect(pre == (kind == 1), "nothing left to undo but the edits made before the script")
    if pre:
        expect(f.redo(), "and they redo")
    for k in range(1, STEPS + 1):
        var before = f.edits
        expect(f.redo(), "redo step " + String(k))
        expect(f.edits > before, "the edit counter moves on redo")
        expect(fingerprint(f) == want[k], "kind " + String(kind) + ": redone to " + String(k) + " steps")
    expect(not f.redo(), "nothing left to redo")


def test_undo_and_redo_in_any_order(kind: Int) raises:
    var want = List[UInt64]()
    for k in range(STEPS + 1):
        want.append(fingerprint(reference(kind, k)))
    var f = source(kind)
    for i in range(STEPS):
        step(f, i)
    var at = STEPS
    var rng = Rng(UInt64(kind + 11))
    for n in range(150):
        if rng.int(0, 1) == 0:
            if at > 0:
                expect(f.undo(), "undo")
                at -= 1
        else:
            expect(f.redo() == (at < STEPS), "redo only below the top")
            if at < STEPS:
                at += 1
        expect(fingerprint(f) == want[at], "kind " + String(kind) + ": walk " + String(n) + " at " + String(at))


def test_a_new_edit_forgets_what_was_undone(kind: Int) raises:
    var f = source(kind)
    for i in range(5):
        step(f, i)
    expect(f.undo() and f.undo(), "two undone")
    _ = f.formula("other = qty + 1")
    expect(not f.redo(), "the formula step cleared redo")
    expect(f.undo(), "back before `other`")
    expect(f.redo(), "`other` can be redone")
    f.set(f.column("qty"), 1, 5)
    expect(not f.redo(), "a cell edit clears redo")
    expect(f.undo() and f.undo() and f.undo() and f.undo(), "four more undone")
    _ = f.formula("m2 = qty")
    _ = f.formula("m3 = qty")
    expect(f.undo(), "m3 undone")
    f.remove_formula(f.column("m2"))
    expect(not f.redo(), "removing a formula clears redo")


def test_exact_values_around_an_added_formula() raises:
    var f = Frame(make_table())
    var rev = f.column("revenue")
    var cost = f.column("cost")
    var base_revenue = f.get(rev, 10)
    var base_cost = f.get(cost, 10)
    var m = f.formula("margin = revenue - cost")
    expect(m == 4 and f.columns() == 5, "the formula is column 4")
    expect(f.get(m, 10) == base_revenue - base_cost, "margin of row 10")
    f.set(rev, 10, 1_000_00)
    expect(f.get(m, 10) == 1_000_00 - base_cost, "margin follows the cell edit")
    expect(f.undo(), "undo the cell edit")
    expect(f.get(m, 10) == base_revenue - base_cost, "margin back to the original after undoing the cell")
    expect(f.undo(), "undo the formula")
    expect(f.columns() == 4 and f.index("margin") < 0, "no formula column")
    expect(f.redo() and f.columns() == 5 and f.formula_text(4) == "margin = revenue - cost", "the formula is back with its text")
    expect(f.get(4, 10) == base_revenue - base_cost, "and its values")
    expect(f.redo() and f.get(4, 10) == 1_000_00 - base_cost, "the cell edit redone: the formula reads it")


def test_replace_remove_and_position() raises:
    var f = Frame(make_table())
    _ = f.formula("a = qty + 1")
    _ = f.formula("b = a * 2")
    _ = f.formula("c = qty * 3")
    var q = f.get(f.column("qty"), 0)
    var before = f.edits
    f.remove_formula(f.column("b"))                                # the middle one; `c` does not read it
    expect(f.edits > before and f.columns() == 6 and f.column("c") == 5, "c moved up")
    expect(f.get(f.column("c"), 0) == q * 3, "c keeps its values")
    expect(f.undo(), "undo the removal")
    expect(f.columns() == 7 and f.formula_text(5) == "b = a * 2" and f.formula_text(6) == "c = qty * 3", "b is back between a and c")
    expect(f.get(5, 0) == (q + 1) * 2, "b reads a again")
    var refused = False
    try:
        f.remove_formula(4)                                        # `a` is read by `b`
    except e:
        refused = "read by the formula `b`" in String(e)
    expect(refused, "a formula another one reads is not removed")
    expect(f.columns() == 7, "the refusal left the frame as it was")
    var scaled = f.formula("a = qty * 1.5")                        # replace: a new scale, `b` follows
    expect(scaled == 4 and f.scale(5) == f.scale(4) and f.scale(4) == 1, "b has a's new scale")
    expect(f.undo(), "undo the replace")
    expect(f.formula_text(4) == "a = qty + 1" and f.scale(4) == 0 and f.scale(5) == 0, "the old a and b's old scale")
    expect(f.get(5, 0) == (q + 1) * 2, "b's old value")


def test_saving_after_undo_writes_the_undone_rows() raises:
    for kind in range(3):
        var f = source(kind)
        for i in range(STEPS):
            step(f, i)
        for _ in range(5):                                         # past the append, the cell edits and a formula removal
            expect(f.undo(), "undone")
        var want = reference(kind, STEPS - 5)
        f.save("f179_saved.parquet", "{}")
        var back = read_parquet("f179_saved.parquet")
        var got = back.take_table().share()
        var alive = 0
        for row in range(want.rows()):
            if not want.alive(row):
                continue
            expect(got.columns() == want.stored(), "stored columns saved")
            for c in range(want.stored()):
                expect(got.is_null(c, alive) == want.is_null(c, row), "kind " + String(kind) + ": null at " + String(c) + "," + String(row))
                if not want.is_null(c, row):
                    if want.base.kind(c) == TEXT:
                        expect(got.text_of(c, Int(got.get(c, alive))) == want.text_of(c, want.get(c, row)), "text at " + String(c) + "," + String(row))
                    else:
                        expect(got.get(c, alive) == want.get(c, row), "value at " + String(c) + "," + String(row))
            alive += 1
        expect(got.rows() == alive, "kind " + String(kind) + ": saved " + String(got.rows()) + " rows, wanted " + String(alive))


def merge_script(mut f: Frame, i: Int) raises:
    var region = f.column("region")
    if i == 0:
        f.set(f.column("cost"), 1, 5_00)
    elif i == 1:
        _ = f.merge(region, [Int64(1)], Int64(0))                  # East reads as West
    elif i == 2:
        _ = f.formula("net = revenue - cost")
    else:
        _ = f.merge(region, [Int64(0)], Int64(2))                  # West (and East with it) reads as North: a repoint


def test_merges_redo_with_formulas_between_them(kind: Int) raises:
    var want = List[UInt64]()
    for k in range(5):
        var r = source(kind)
        for i in range(k):
            merge_script(r, i)
        want.append(fingerprint(r))
    for k in range(4):
        expect(want[k] != want[k + 1], "the merge script step " + String(k) + " changes what is seen")
    var f = source(kind)
    for i in range(4):
        merge_script(f, i)
    expect(fingerprint(f) == want[4], "kind " + String(kind) + ": the script itself")
    for k in range(4, 0, -1):
        expect(f.undo() and fingerprint(f) == want[k - 1], "kind " + String(kind) + ": merge undone to " + String(k - 1))
    for k in range(1, 5):
        expect(f.redo() and fingerprint(f) == want[k], "kind " + String(kind) + ": merge redone to " + String(k))
    expect(f.undo() and f.undo() and f.redo() and f.undo() and f.redo() and f.redo(), "back and forth")
    expect(fingerprint(f) == want[4], "kind " + String(kind) + ": the top again")


def test_a_snapshot_undoes_formula_steps(kind: Int) raises:
    var f = source(kind)
    for i in range(STEPS):
        step(f, i)
    _ = f.undo()
    var want = fingerprint(f)
    var snap = frame_snapshot(f)
    expect(fingerprint(snap) == want, "the snapshot reads as the frame")
    f.set(f.column("qty"), 1, 2)                                   # the frame goes on without it
    expect(snap.redo(), "the snapshot redoes the step the frame had undone")
    expect(fingerprint(snap) == fingerprint(reference(kind, STEPS)), "kind " + String(kind) + ": snapshot redone")
    for _ in range(STEPS):
        expect(snap.undo(), "snapshot undo")
    expect(fingerprint(snap) == fingerprint(reference(kind, 0)), "kind " + String(kind) + ": snapshot undone through the formula steps")


def test_dropping_the_redo_list_frees_the_formula_snapshots() raises:
    var f = source(0)
    for i in range(STEPS):
        step(f, i)
    var held = 0
    for _ in range(STEPS):
        _ = f.undo()
    for s in f.swaps:
        held += len(s)
    expect(held > 0, "undone formula steps keep their lists for redo")
    f.set(f.column("qty"), 0, 3)                                   # a new edit: nothing can be redone
    held = 0
    for s in f.swaps:
        held += len(s)
    expect(held == 0, "no list is held for a step that can no longer be redone, " + String(held) + " left")


def main() raises:
    for kind in range(3):
        test_every_step_undoes_and_redoes_to_the_exact_earlier_state(kind)
        test_undo_and_redo_in_any_order(kind)
        test_a_new_edit_forgets_what_was_undone(kind)
        test_merges_redo_with_formulas_between_them(kind)
        test_a_snapshot_undoes_formula_steps(kind)
    test_dropping_the_redo_list_frees_the_formula_snapshots()
    test_exact_values_around_an_added_formula()
    test_replace_remove_and_position()
    test_saving_after_undo_writes_the_undone_rows()
    print("ok")
