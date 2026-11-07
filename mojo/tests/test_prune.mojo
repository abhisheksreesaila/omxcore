# F-210: dead instructions are dropped from the CPU kernel's code (query.prune_dead, query.cpu_code). The SQL lowering writes guard registers for every
# AND of a filter (OP_NE against zero, then a copy) that nothing reads: 12 of TPC-H Q6's 37 instructions. A register below `keep_below`
# and the predicate register are the only ones read from outside the program; an aggregate, OP_FLAG and an integer op that sets
# the overflow flag are never dropped. What stays is checked here instruction by instruction; that the answers are unchanged is
# what the SQL suites (equal to DuckDB) and test_kernels_from_package check.
from omxcore.query import prune_dead, cpu_code, Program

comptime LOAD_CONST = 2
comptime LOADI = 3
comptime NE = 25
comptime AND = 30
comptime IADD = 15
comptime FLAG = 49
comptime IMAX = 45
comptime ICMP = 46
comptime AGG_ISUM = 53


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


def ins(mut code: List[Int32], op: Int, d: Int, a: Int = 0, b: Int = 0, c: Int = 0):
    code.append(Int32(op))
    code.append(Int32(d))
    code.append(Int32(a))
    code.append(Int32(b))
    code.append(Int32(c))


def codes(code: List[Int32], want: List[Int], n: Int, what: String) raises:
    """The instruction words after the 16-word header are exactly `want` (n instructions)."""
    if len(code) != 16 + len(want) or len(want) != 5 * n:
        raise Error("FAILED: " + what + ": " + String((len(code) - 16) // 5) + " instructions, wanted " + String(n))
    for i in range(len(want)):
        if Int(code[16 + i]) != want[i]:
            raise Error("FAILED: " + what + ": word " + String(i) + " is " + String(Int(code[16 + i])) + ", wanted " + String(want[i]))


def same(dead: List[Bool], want: List[Bool], what: String) raises:
    for i in range(len(want)):
        if dead[i] != want[i]:
            raise Error("FAILED: " + what + ": instruction " + String(i) + " is " + ("dead" if dead[i] else "live") + ", wanted " + ("dead" if want[i] else "live"))


def main() raises:
    # Q6's shape: a comparison feeds the kept predicate register 1; a guard chain (const, NE, copy) that nothing reads is dead
    var a = List[Int32]()
    ins(a, LOADI, 3, 0)
    ins(a, LOAD_CONST, 4, 0)
    ins(a, ICMP, 3, 3, 4, 23)
    ins(a, LOAD_CONST, 5, 1)          # dead: read only by the NE below
    ins(a, NE, 6, 3, 5)               # dead: read only by the copy below
    ins(a, IMAX, 5, 6, 6)             # dead: register 5 is a temporary nobody reads after it
    ins(a, IMAX, 1, 3, 3)             # the predicate register
    same(prune_dead(a, 2, 1), [False, False, False, True, True, True, False], "the guard chain goes")

    # a guard that a later instruction reads stays, with everything it needs
    var b = List[Int32]()
    ins(b, LOAD_CONST, 5, 1)
    ins(b, NE, 6, 3, 5)
    ins(b, IMAX, 5, 6, 6)
    ins(b, AND, 1, 5, 3)
    same(prune_dead(b, 2, 1), [False, False, False, False], "a guard that is read stays")

    # a value overwritten before anyone reads it is dead; a register below keep_below is read from outside, so its last write stays
    var c = List[Int32]()
    ins(c, LOAD_CONST, 9, 0)          # dead: overwritten next
    ins(c, LOAD_CONST, 9, 1)
    ins(c, IMAX, 1, 9, 9)
    ins(c, LOAD_CONST, 2, 2)          # a kept register (keep_below 4): stays although nothing reads it here
    ins(c, LOAD_CONST, 8, 2)          # a temporary nobody reads: dead
    same(prune_dead(c, 4, 1), [True, False, False, False, True], "an overwritten value goes, a kept register stays")

    # OP_FLAG keeps what feeds it; an integer add (it sets the overflow flag) stays although its result is unread; an aggregate stays
    var d = List[Int32]()
    ins(d, LOAD_CONST, 5, 0)
    ins(d, IMAX, 6, 5, 5)
    ins(d, FLAG, 6, 6)
    ins(d, IADD, 9, 5, 5)
    ins(d, AGG_ISUM, 0, 5)
    same(prune_dead(d, 2, -1), [False, False, False, False, False], "FLAG, IADD and an aggregate keep their inputs")

    # cpu_code: the code the kernel gets. A column compared with a constant fuses into one OP_ICMPCOL (56) when nothing reads the two registers after it
    var f = Program(2)
    f.prune_below = 2
    f.pred_reg = 1
    f.emit(LOADI, 3, 0)
    f.emit(LOAD_CONST, 4, 0)
    f.emit(ICMP, 3, 3, 4, 23)
    f.emit(LOAD_CONST, 5, 1)          # a guard chain nothing reads: dropped
    f.emit(NE, 6, 3, 5)
    f.emit(IMAX, 5, 6, 6)
    f.emit(IMAX, 1, 3, 3)
    var fc = cpu_code(f)
    codes(fc, [56, 3, 0, 0, 23, 45, 1, 3, 3, 0], 2, "a fused compare and the kept copy")
    expect(Int(fc[3]) == 2 and Int(fc[4]) == 0 and Int(fc[5]) == 0 and Int(fc[6]) == 1, "the header counts the instructions that stay and names the predicate register")

    var g = Program(2)                 # the compared register is read again later: no fusion
    g.prune_below = 2
    g.pred_reg = 1
    g.emit(LOADI, 3, 0)
    g.emit(LOAD_CONST, 4, 0)
    g.emit(ICMP, 5, 3, 4, 23)
    g.emit(AND, 1, 3, 5)
    codes(cpu_code(g), [3, 3, 0, 0, 0, 2, 4, 0, 0, 0, 46, 5, 3, 4, 23, 30, 1, 3, 5, 0], 4, "x is read later: no fusion")

    var h = Program(2)                 # the constant register is read again later: no fusion
    h.prune_below = 2
    h.pred_reg = 1
    h.emit(LOADI, 3, 0)
    h.emit(LOAD_CONST, 4, 0)
    h.emit(ICMP, 5, 3, 4, 23)
    h.emit(AND, 1, 5, 4)
    codes(cpu_code(h), [3, 3, 0, 0, 0, 2, 4, 0, 0, 0, 46, 5, 3, 4, 23, 30, 1, 5, 4, 0], 4, "y is read later: no fusion")

    var s = Program(2)                 # not adjacent: no fusion
    s.prune_below = 3
    s.pred_reg = 1
    s.emit(LOADI, 3, 0)
    s.emit(LOAD_CONST, 6, 1)
    s.emit(LOAD_CONST, 4, 0)
    s.emit(ICMP, 1, 3, 4, 23)
    s.emit(IMAX, 2, 6, 6)              # register 2 is kept, so the constant in register 6 is read
    codes(cpu_code(s), [3, 3, 0, 0, 0, 2, 6, 1, 0, 0, 2, 4, 0, 0, 0, 46, 1, 3, 4, 23, 45, 2, 6, 6, 0], 5, "an instruction between the three: no fusion")

    var u = Program(2)                 # a program from anywhere but the SQL lowering (prune_below -1): nothing is dropped or fused
    u.pred_reg = 1
    u.emit(LOADI, 3, 0)
    u.emit(LOAD_CONST, 4, 0)
    u.emit(ICMP, 1, 3, 4, 23)
    u.emit(LOAD_CONST, 5, 1)
    codes(cpu_code(u), [3, 3, 0, 0, 0, 2, 4, 0, 0, 0, 46, 1, 3, 4, 23, 2, 5, 1, 0, 0], 4, "no prune_below: the code is as written")

    var b2 = Program(2)                # the compare is in the body: the section counts follow
    b2.prune_below = 2
    b2.pred_reg = 1
    b2.section = 1
    b2.emit(LOAD_CONST, 4, 0)
    b2.emit(LOADI, 3, 1)
    b2.emit(ICMP, 3, 3, 4, 20)
    b2.emit(IMAX, 2, 3, 3)
    b2.section = 2
    b2.emit(AGG_ISUM, 0, 2)
    var bc = cpu_code(b2)
    codes(bc, [56, 3, 1, 0, 20, 45, 2, 3, 3, 0, 53, 0, 2, 0, 0], 3, "a compare in the body fuses too (the constant first)")
    expect(Int(bc[3]) == 0 and Int(bc[4]) == 2 and Int(bc[5]) == 1, "the section counts follow the instructions")

    print("prune ok")
