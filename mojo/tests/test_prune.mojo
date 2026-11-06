# F-210: dead instructions are dropped from the CPU kernel's code (kargs.prune_dead). The SQL lowering writes guard registers for every
# AND of a filter (OP_NE against zero, then a copy) that nothing reads: 12 of TPC-H Q6's 37 instructions. A register below `keep_below`
# and the predicate register are the only ones read from outside the program; an aggregate, OP_FLAG and an integer op that sets
# the overflow flag are never dropped. What stays is checked here instruction by instruction; that the answers are unchanged is
# what the SQL suites (equal to DuckDB) and test_kernels_from_package check.
from omxcore.query import prune_dead

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

    print("prune ok")
