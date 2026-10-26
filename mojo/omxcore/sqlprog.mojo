"""SQL lowering mode (F-150): a bound query as one kernel program, with SQL's rules.

The builder's `_lower` (query.mojo) keeps the Query API's rules: a filter that reads a null cell is false as a whole,
a group is a dense number 0 .. groups-1. SQL needs more, so this module emits the same kernel instructions with its own
rules, on the same `Program` type:

- Three-valued logic. A boolean is a float register holding 1.0 (true), 0.0 (false) or 0.5 (unknown), so AND is the
  minimum, OR the maximum and NOT is 1 - x (Kleene), and WHERE keeps the rows that read exactly 1.0. A comparison is
  unknown only when an operand can be NULL: a column with no nulls compiles to plain 0.0 / 1.0 with no extra work.
- A value is an int64 register plus, when it can be NULL, a float register that is 1.0 where it is (`nul`). Arithmetic
  ORs the `nul` of its operands; `x // 0` and `x % 0` are NULL; `//` and `%` truncate toward zero as DuckDB does.
- A group key is a number in a mixed-radix layout with a NULL slot per nullable key (dense), or a list of key words
  (hashed): NULL is a group of its own.
- Every aggregate skips NULL: it is fed `SELECT(nul, identity, value)` and a count of the NULLs it skipped, so that
  SUM / MIN / MAX / AVG of nothing is NULL, which the finishing code reads off `rows - nulls`.
"""
from std.collections import Dict
from std.memory import ArcPointer
from std.time import perf_counter_ns
from .table import TEXT, BOOL, FLOAT, INT, DECIMAL, DATE, F32_COLUMN, SharedTable
from .parquet import ParquetScan
from .kernels.pipeline_cpu import LP
from .kernels._fpenv import core_count, par_n
from .query import (Program, FOOTER_LIE, OP_LOAD_CONST, OP_LOADI, OP_IADD, OP_ISUB, OP_IMUL, OP_F2I, OP_AND, OP_OR, OP_NOT, OP_ISNULL,
                    OP_IDIV, OP_IMOD, OP_IMIN, OP_IMAX, OP_ICMP, OP_AGG_ISUM, OP_AGG_IMIN, OP_AGG_IMAX, MAX_REGS,
                    CTL_PLACEHOLDER, A_SUM, A_MIN, A_MAX, A_FSUM, A_FMIN, A_FMAX, _frame_mask, _null_mask,
                    OP_LOAD_COL, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_AGG_SUM, OP_AGG_MIN, OP_AGG_MAX, NAN_AWARE)
from .frame import _text_less, text_ranks
from .bind import (BoundQuery, BExpr, B_DIV, B_I2F, B_FDIV, B_FMOD, B_LEAST, B_GREATEST, B_COL, B_LIT, B_NULL, B_ADD, B_SUB, B_MUL,
                   B_IDIV, B_IMOD, B_NEG, B_RESCALE, B_CMP, B_AND, B_OR, B_NOT, B_ISNULL, B_IN, C_LT, C_GE, C_LE, C_GT, C_EQ,
                   C_NE, F_SUM, F_MIN, F_MAX, F_AVG, pow10_checked,
                   B_STR, B_CASE, B_COALESCE, B_DISTINCT, B_CAST, B_DATEPART, B_DATETRUNC, B_ADDMONTHS, B_DATEDIFF, B_BOOLIS, B_TFN, B_CONCAT, DERIVED_BASE)
from .sqlfn import (TFn, CaseTables, parse_text, reverse_is_exact, date_text, decimal_text, date_part, date_trunc_days, add_months, date_diff_days, TF_LOWER, TF_UPPER, TF_LIKE, TF_REVERSE, DP_YEAR, DP_MONTH, DP_DAY, DP_DOW, DP_ISODOW, DP_DOY, DP_QUARTER, DP_EPOCH,
                    DP_DECADE, DP_WEEK, DP_ISOYEAR, DP_CENTURY, DP_MILLENNIUM)
from .planner import PhysicalPlan, G_DENSE, G_HASHED

comptime OP_F32 = 34
comptime OP_SUBF = 11        # float x - y
comptime OP_I2F = 19
comptime OP_SELECT = 35
comptime OP_IN_SET = 37
comptime OP_FLOOR = 33
comptime OP_TAB = 47            # F-156: dst = table[a - lo] (a TEXT function's answer per dictionary word)
comptime OP_ISET = 48           # F-156: dst = 1.0 when the int64 is in a hash set (an IN list of any length)
comptime OP_FLAG = 49           # F-156: the error flag, set where a cast met a value it cannot convert
comptime CONCAT_LIMIT = 1 << 20     # the most pairs of words a || of two columns builds
comptime CAST_TEXT_LIMIT = 1 << 22  # the most values a CAST of a column to text writes the words of



struct DerivedStore(Movable):
    """The dictionaries a query's own TEXT expressions make (a text function's answers, a CASE over words, a constant word), by `src - DERIVED_BASE`.
    Every DictAccess of one query shares it, so the words the program was lowered with are the words the result is read with."""
    var words: List[List[String]]
    var built: List[Bool]
    var index: List[Dict[String, Int]]       # word -> code, made when a word is first looked up (a dictionary of millions is rarely asked)
    var indexed: List[Bool]

    def __init__(out self):
        self.words = List[List[String]]()
        self.built = List[Bool]()
        self.index = List[Dict[String, Int]]()
        self.indexed = List[Bool]()

    def put(mut self, i: Int, var ws: List[String]):
        while len(self.words) <= i:
            self.words.append(List[String]())
            self.built.append(False)
            self.index.append(Dict[String, Int]())
            self.indexed.append(False)
        self.words[i] = ws^
        self.index[i] = Dict[String, Int]()
        self.indexed[i] = False
        self.built[i] = True

    def lookup(mut self, i: Int, w: String) -> Int:
        if not self.indexed[i]:
            var ix = Dict[String, Int]()
            for k in range(len(self.words[i])):
                ix[self.words[i][k]] = k
            self.index[i] = ix^
            self.indexed[i] = True
        var hit = self.index[i].get(w)
        return hit.value() if hit else -1


struct DictAccess(Movable):
    """The words of a source's TEXT columns: a table's or a scan's dictionary, then the words a frame's session added; and the dictionaries the query's own
    TEXT expressions make (a `src` from DERIVED_BASE on)."""
    var table: SharedTable
    var scan: Optional[ParquetScan]
    var extra: List[List[String]]
    var store: ArcPointer[DerivedStore]
    var range_cols: List[Int]            # the lowest and highest value of integer-like columns (the planner's cache), for a CAST of a column to text
    var range_lo: List[Int64]
    var range_hi: List[Int64]
    var range_has: List[Bool]
    var rows: Int                        # how many rows the source holds (a per-day table pays off over this many)
    var range_exact: Bool                # the ranges were found by a pass over the data (False: a Parquet footer's statistics, a hint a row may contradict)

    def __init__(out self, table: SharedTable, scan: Optional[ParquetScan], extra: List[List[String]]):
        self.rows = 0
        self.range_exact = False
        self.table = table
        self.scan = scan.copy()
        self.extra = extra.copy()
        self.store = ArcPointer(DerivedStore())
        self.range_cols = List[Int]()
        self.range_lo = List[Int64]()
        self.range_hi = List[Int64]()
        self.range_has = List[Bool]()

    def set_ranges(mut self, cols: List[Int], lo: List[Int64], hi: List[Int64], has: List[Bool], rows: Int = 0, exact: Bool = True):
        self.rows = rows
        self.range_exact = exact
        self.range_cols = cols.copy()
        self.range_lo = lo.copy()
        self.range_hi = hi.copy()
        self.range_has = has.copy()

    def share(self) -> DictAccess:
        """The same words, and the same derived dictionaries (not a copy of them)."""
        var d = DictAccess(self.table, self.scan, self.extra)
        d.store = self.store
        d.set_ranges(self.range_cols, self.range_lo, self.range_hi, self.range_has, self.rows, self.range_exact)
        return d^

    def range_of(self, c: Int) -> Tuple[Bool, Int64, Int64]:
        for i in range(len(self.range_cols)):
            if self.range_cols[i] == c:
                return (self.range_has[i], self.range_lo[i], self.range_hi[i])
        return (False, Int64(0), Int64(0))

    def word_ptrs(self, c: Int) raises -> Tuple[Int, Int, Int, Int]:
        """Where a dictionary's words are, for a loop over them in parallel: (the address of the base words, how many, the address of the frame's added words, how many)."""
        if c >= DERIVED_BASE:
            var i = c - DERIVED_BASE
            if not self.has_derived(c):
                raise Error("internal: the dictionary of expression " + String(i) + " was read before it was made")
            return (Int(self.store[].words[i].unsafe_ptr()), len(self.store[].words[i]), 0, 0)
        var base = self.scan.value().words_address(c) if self.scan else Int(self.table.dictionary(c).unsafe_ptr())
        var nbase = self.base_count(c)
        if c < len(self.extra) and len(self.extra[c]) > 0:
            return (base, nbase, Int(self.extra[c].unsafe_ptr()), len(self.extra[c]))
        return (base, nbase, 0, 0)

    def has_derived(self, c: Int) -> Bool:
        var i = c - DERIVED_BASE
        return c >= DERIVED_BASE and i < len(self.store[].built) and self.store[].built[i]

    def set_derived(self, c: Int, var words: List[String]):
        self.store[].put(c - DERIVED_BASE, words^)

    def base_count(self, c: Int) raises -> Int:
        if self.scan:
            return self.scan.value().word_count(c)
        return len(self.table.dictionary(c))

    def count(self, c: Int) raises -> Int:
        if c >= DERIVED_BASE:
            if not self.has_derived(c):
                raise Error("internal: the dictionary of expression " + String(c - DERIVED_BASE) + " was read before it was made")
            return len(self.store[].words[c - DERIVED_BASE])
        return self.base_count(c) + (len(self.extra[c]) if c < len(self.extra) else 0)

    def word(self, c: Int, code: Int) raises -> String:
        if c >= DERIVED_BASE:
            return self.store[].words[c - DERIVED_BASE][code]
        var b = self.base_count(c)
        if code < b:
            return self.scan.value().text_of(c, code) if self.scan else self.table.text_of(c, code)
        return self.extra[c][code - b]

    def code_of(self, c: Int, w: String) raises -> Int:
        if c >= DERIVED_BASE:
            return self.store[].lookup(c - DERIVED_BASE, w)
        var k = self.scan.value().code_of(c, w) if self.scan else self.table.code_of(c, w)
        if k >= 0:
            return k
        if c < len(self.extra):
            for i in range(len(self.extra[c])):
                if self.extra[c][i] == w:
                    return self.base_count(c) + i
        return -1


struct V(Copyable, Movable):
    """An emitted expression: its register, the float register that is 1.0 where it is NULL (-1: never), and for a
    boolean whether it can hold the unknown 0.5; `fl`: the register holds a float64 (F-157), not int64 bits."""
    var reg: Int
    var nul: Int
    var unk: Bool
    var fl: Bool

    def __init__(out self, reg: Int, nul: Int, unk: Bool, fl: Bool = False):
        self.reg = reg
        self.nul = nul
        self.unk = unk
        self.fl = fl


struct SqlProgram(Movable):
    var p: Program
    var set_bytes: List[UInt8]
    var set_meta: List[Int64]
    var agg_val_slot: List[Int]      # per aggregate: its kernel slot, -1 for none
    var agg_nul_slot: List[Int]      # per aggregate: the slot counting the NULLs it skipped, -1 when its argument has none
    var agg_nan_slot: List[Int]      # per float min / max: the slot counting the NaNs it left out (NaN is above every number), else -1
    var agg_src: List[Int]           # per aggregate: its holistic value source, -1 for a plain one
    var slot_ops: List[Int]          # per slot: how partial results merge (A_SUM, A_MIN or A_MAX)
    var key_regs: Int                # registers 0 .. key_regs-1 hold the group key (dense: 1)
    var n_src: Int
    var guard_slot: Int              # dense with a footer-stated range: the extra group that rows outside it land in, else -1
    var instr: Int                   # instructions a row runs (predicate, body, aggregates)
    var row_vregs: List[Int]         # F-152, rows: per computed expression, the register holding its value (int64 bits) ...
    var row_nregs: List[Int]         # ... and its NULL flag (1 where NULL), -1 when it can never be NULL

    def __init__(out self, var p: Program):
        self.p = p^
        self.set_bytes = List[UInt8]()
        self.set_meta = List[Int64]()
        self.agg_val_slot = List[Int]()
        self.agg_nul_slot = List[Int]()
        self.agg_nan_slot = List[Int]()
        self.agg_src = List[Int]()
        self.slot_ops = List[Int]()
        self.key_regs = 1
        self.n_src = 0
        self.guard_slot = -1
        self.instr = 0
        self.row_vregs = List[Int]()
        self.row_nregs = List[Int]()


def bkey(rexprs: List[BExpr], e: Int) -> String:
    """A string that is equal for equal row-level expressions: to tell which aggregates read the same argument."""
    ref x = rexprs[e]
    var s = String(x.op) + ":" + String(x.value) + ":" + x.text
    if x.a >= 0:
        s += "(" + bkey(rexprs, x.a)
        if x.b >= 0:
            s += "," + bkey(rexprs, x.b)
        s += ")"
    for it in x.items:
        s += "," + bkey(rexprs, it)
    return s


def collect_columns(q: BoundQuery) -> List[Int]:
    """The source columns the query reads, ascending."""
    var seen = List[Int]()
    for e in q.rexprs:
        if e.op == B_COL:
            var c = Int(e.value)
            var at = len(seen)
            var dup = False
            for s in seen:
                if s == c:
                    dup = True
            if not dup:
                seen.append(c)
    # insertion sort (a handful of columns)
    for i in range(1, len(seen)):
        var j = i
        while j > 0 and seen[j - 1] > seen[j]:
            var t = seen[j]
            seen[j] = seen[j - 1]
            seen[j - 1] = t
            j -= 1
    return seen^


def _words_bool(t: TFn, base: Int, nbase: Int, extra: Int, nextra: Int, out_addr: Int):
    """out[k] = whether the text function (a predicate) holds for word k, in parallel over the words (read-only: no String is made)."""
    var n = nbase + nextra
    var tasks = max(1, min(core_count(), n // 16384))

    def work(i: Int) {imm}:
        var a = n * i // tasks
        var b = n * (i + 1) // tasks
        var bp = UnsafePointer[String, MutAnyOrigin](unsafe_from_address=base)
        var ep = UnsafePointer[String, MutAnyOrigin](unsafe_from_address=extra)
        var outp = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=out_addr)
        for k in range(a, b):
            if k < nbase:
                outp[k] = UInt8(1) if t.to_bool(bp[k]) else UInt8(0)
            else:
                outp[k] = UInt8(1) if t.to_bool(ep[k - nbase]) else UInt8(0)

    if tasks == 1:
        work(0)
    else:
        par_n(work, tasks, tasks)


def _words_int(t: TFn, base: Int, nbase: Int, extra: Int, nextra: Int, out_addr: Int):
    """out[k] = the integer the text function (length, strpos) answers for word k, in parallel over the words."""
    var n = nbase + nextra
    var tasks = max(1, min(core_count(), n // 16384))

    def work(i: Int) {imm}:
        var a = n * i // tasks
        var b = n * (i + 1) // tasks
        var bp = UnsafePointer[String, MutAnyOrigin](unsafe_from_address=base)
        var ep = UnsafePointer[String, MutAnyOrigin](unsafe_from_address=extra)
        var outp = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=out_addr)
        for k in range(a, b):
            if k < nbase:
                outp[k] = Int64(t.to_int(bp[k]))
            else:
                outp[k] = Int64(t.to_int(ep[k - nbase]))

    if tasks == 1:
        work(0)
    else:
        par_n(work, tasks, tasks)


struct TextUnion(Copyable, Movable):
    """The dictionary of a CASE / COALESCE over words: its words once each, and for every value its codes in it (`tabs`), or the code of a constant (`lit`)."""
    var words: List[String]
    var index: Dict[String, Int]
    var tabs: List[List[Int64]]
    var lit: List[Int]
    var identity: List[Bool]

    def __init__(out self):
        self.words = List[String]()
        self.index = Dict[String, Int]()
        self.tabs = List[List[Int64]]()
        self.lit = List[Int]()
        self.identity = List[Bool]()

    def add(mut self, w: String) -> Int:
        var hit = self.index.get(w)
        if hit:
            return hit.value()
        self.index[w] = len(self.words)
        self.words.append(w)
        return len(self.words) - 1

    def is_identity(self, j: Int) -> Bool:
        return j < len(self.identity) and self.identity[j]


struct _Emitter(Movable):
    var p: Program
    var rx: List[BExpr]
    var nullable: List[Bool]
    var pos: List[Int]
    var aliases: List[List[Int]]     # per source column: from, to, from, to ... (a frame's merged TEXT codes)
    var live: List[Bool]
    var kept_next: Int
    var dicts: DictAccess
    var set_bytes: List[UInt8]
    var set_meta: List[Int64]
    var case_tabs: Optional[CaseTables]
    var ents: Dict[Int, List[Int64]]     # F-156: per derived TEXT / INT node, its per-word table (code or answer per code of the dictionary it reads)
    var undecided: Dict[Int, List[UInt8]]    # ... and per node whose function DuckDB does not answer by code points (reverse), the words it is not decided for
    var unions: Dict[Int, TextUnion]     # ... and per CASE / COALESCE over words, its dictionary
    var guard: Int                   # F-156: the register that says where the expression being emitted is live (a CASE branch), -1 for everywhere

    def __init__(out self, var p: Program, var rx: List[BExpr], var nullable: List[Bool], var pos: List[Int],
                 var aliases: List[List[Int]], var dicts: DictAccess):
        self.case_tabs = None
        self.ents = Dict[Int, List[Int64]]()
        self.undecided = Dict[Int, List[UInt8]]()
        self.unions = Dict[Int, TextUnion]()
        self.guard = -1
        self.p = p^
        self.rx = rx^
        self.nullable = nullable^
        self.pos = pos^
        self.aliases = aliases^
        self.live = List[Bool](length=MAX_REGS, fill=False)
        self.kept_next = 0
        self.dicts = dicts^
        self.set_bytes = List[UInt8]()
        self.set_meta = List[Int64]()

    def take_program(deinit self) -> Program:
        return self.p^

    # -- registers
    def tmp(mut self) raises -> Int:
        var r = MAX_REGS - 1
        while r >= self.kept_next:
            if not self.live[r]:
                self.live[r] = True
                return r
            r -= 1
        raise Error("the query needs more than " + String(MAX_REGS) + " registers")

    def free(mut self, r: Int):
        if r >= self.kept_next and r < MAX_REGS:
            self.live[r] = False

    def reserve(mut self, n: Int):
        """Registers 0 .. n-1 are for the group key (and the control block): kept, never temporaries."""
        for i in range(n):
            self.live[i] = True
        self.kept_next = n

    def keep(mut self, r: Int) raises -> Int:
        """Copies register r into one that lives to the end of the program."""
        if self.kept_next >= MAX_REGS or self.live[self.kept_next]:
            raise Error("the query needs more than " + String(MAX_REGS) + " registers")
        var k = self.kept_next
        self.kept_next += 1
        self.live[k] = True
        self.p.emit(OP_IMAX, k, r, r)
        return k

    def ins(mut self, op: Int, d: Int, a: Int = 0, b: Int = 0, c: Int = 0):
        self.p.emit(op, d, a, b, c)

    def const_i(mut self, v: Int64) raises -> Int:
        var r = self.tmp()
        self.p.emit(OP_LOAD_CONST, r, self.p.const(v))
        return r

    def const_f(mut self, x: Float64) raises -> Int:
        var r = self.tmp()
        self.p.emit(OP_LOAD_CONST, r, self.p.const(Int64(x.to_bits())))
        return r

    def or_nul(mut self, a: Int, b: Int) -> Int:
        if a < 0:
            return b
        if b < 0:
            return a
        self.p.emit(OP_OR, a, a, b)
        self.free(b)
        return a

    # -- values
    def value(mut self, e: Int) raises -> V:
        var x = self.rx[e].copy()
        if x.op == B_COL:
            var c = Int(x.value)
            var r = self.tmp()
            if x.kind == FLOAT.value:
                self.p.has_float = True
                self.ins(OP_LOAD_COL, r, self.p.column_f64(self.pos[c]))
            else:
                self.ins(OP_LOADI, r, self.p.column(self.pos[c]))
            if x.kind == TEXT.value and len(self.aliases[c]) > 0:
                var k = 0
                while k < len(self.aliases[c]):
                    var from_c = self.const_i(Int64(self.aliases[c][k]))
                    var hit = self.tmp()
                    self.ins(OP_ICMP, hit, r, from_c, C_EQ)
                    var to_c = self.const_i(Int64(self.aliases[c][k + 1]))
                    self.ins(OP_SELECT, r, hit, to_c, r)
                    self.free(from_c)
                    self.free(hit)
                    self.free(to_c)
                    k += 2
            var nul = -1
            if self.nullable[c]:
                nul = self.tmp()
                self.ins(OP_ISNULL, nul, _null_mask(self.p, self.pos[c]))
            return V(r, nul, False, x.kind == FLOAT.value)
        if x.op == B_STR:                                      # a constant word: a dictionary of one word, code 0
            self.prep_dict(e)
            return V(self.const_i(0), -1, False)
        if x.op == B_CASE:
            return self.case_generic(e, False)
        if x.op == B_COALESCE:
            return self.coalesce_generic(e, False)
        if x.op == B_CAST:
            return self.cast_value(e)
        if x.op == B_TFN:
            return self.tfn_value(e)
        if x.op == B_CONCAT:
            return self.concat_value(e)
        if x.op == B_DATEPART or x.op == B_DATETRUNC or x.op == B_ADDMONTHS or x.op == B_DATEDIFF:
            var by_day = self.day_table(e)
            if by_day:
                return by_day.value().copy()
            if x.op == B_DATEPART:
                return self.datepart_value(x)
            if x.op == B_DATETRUNC:
                return self.datetrunc_value(x)
            if x.op == B_ADDMONTHS:
                return self.addmonths_value(x)
            return self.datediff_value(x)
        if x.op == B_LIT:
            return V(self.const_i(x.value), -1, False, x.kind == FLOAT.value)       # (a float literal's value is its bits: the constant pool holds bits)
        if x.op == B_NULL:
            return V(self.const_i(0), self.const_f(1.0), False)
        if x.op == B_RESCALE:
            var a = self.value(x.a)
            var k = self.const_i(pow10_checked(Int(x.value)))
            self.ins(OP_IMUL, a.reg, a.reg, k)
            self.free(k)
            return a^
        if x.op == B_I2F:                                      # an INT or DECIMAL read as a float: float(v) / 10^scale, as DuckDB casts it
            var a = self.value(x.a)
            self.p.has_float = True
            self.ins(OP_I2F, a.reg, a.reg)
            var real = x.scale == F32_COLUMN                       # cast to a 32-bit float (DuckDB's REAL): rounded after each step
            if real:
                self.ins(OP_F32, a.reg, a.reg)
            if x.value > 0:
                var k = self.const_f(Float64(Int(pow10_checked(Int(x.value)))))
                self.ins(OP_DIV, a.reg, a.reg, k)
                self.free(k)
                if real:
                    self.ins(OP_F32, a.reg, a.reg)
            return V(a.reg, a.nul, False, True)
        if x.op == B_NEG:
            var a = self.value(x.a)
            if a.fl:                                           # times -1.0: the sign of zero, inf and NaN flips too
                var m1 = self.const_f(-1.0)
                self.ins(OP_MUL, a.reg, a.reg, m1)
                self.free(m1)
                return a^
            var z = self.const_i(0)
            self.ins(OP_ISUB, a.reg, z, a.reg)
            self.free(z)
            return a^
        if x.op == B_ADD or x.op == B_SUB or x.op == B_MUL or x.op == B_DIV:
            var a = self.value(x.a)
            var b = self.value(x.b)
            if x.kind == FLOAT.value:
                self.ins(OP_ADD if x.op == B_ADD else OP_SUB if x.op == B_SUB else OP_MUL if x.op == B_MUL else OP_DIV, a.reg, a.reg, b.reg)
                if x.scale == F32_COLUMN:                          # REAL arithmetic: the double result rounded to 32 bits is the 32-bit result
                    self.ins(OP_F32, a.reg, a.reg)
                self.free(b.reg)
                return V(a.reg, self.or_nul(a.nul, b.nul), False, True)
            self.ins(OP_IADD if x.op == B_ADD else OP_ISUB if x.op == B_SUB else OP_IMUL, a.reg, a.reg, b.reg)
            self.free(b.reg)
            return V(a.reg, self.or_nul(a.nul, b.nul), False)
        if x.op == B_IDIV or x.op == B_IMOD:
            return self.divmod(x)
        if (x.op == B_LEAST or x.op == B_GREATEST) and x.kind == FLOAT.value:         # NaN is above every number: least skips it, greatest is it
            var a = self.value(x.a)
            var b = self.value(x.b)
            var pick = self.tmp()
            self.ins(C_LT if x.op == B_LEAST else C_GT, pick, a.reg, b.reg, NAN_AWARE)
            self.ins(OP_SELECT, a.reg, pick, a.reg, b.reg)
            self.free(pick)
            self.free(b.reg)
            return V(a.reg, self.or_nul(a.nul, b.nul), False, True)
        if x.op == B_FDIV or x.op == B_FMOD or x.op == B_LEAST or x.op == B_GREATEST:      # a frame formula's operators: the kernel's own
            var a = self.value(x.a)
            var b = self.value(x.b)
            self.ins(OP_IDIV if x.op == B_FDIV else OP_IMOD if x.op == B_FMOD else OP_IMIN if x.op == B_LEAST else OP_IMAX, a.reg, a.reg, b.reg)
            self.free(b.reg)
            return V(a.reg, self.or_nul(a.nul, b.nul), False)
        raise Error("internal: expression kind " + String(x.op) + " is not a row-level value")

    # ── F-156: tables, text, CASE, CAST, dates ────────────────────────────────────────────────────────────

    def map_case(mut self, w: String, up: Bool) raises -> String:
        if not self.case_tabs:
            self.case_tabs = CaseTables()
        return self.case_tabs.value().map(w, up)

    def add_tab(mut self, entries: List[Int64], lo: Int) -> Int:
        """A table for OP_TAB: `entries` as 8-byte words in the program's set bytes (8-aligned); returns its set number."""
        var at = len(self.set_bytes)
        var off = (at + 7) // 8 * 8
        self.set_bytes.resize(off + 8 * max(len(entries), 1), 0)
        var base = LP(unsafe_from_address=Int(self.set_bytes.unsafe_ptr()) + off)
        for i in range(len(entries)):
            base[i] = entries[i]
        self.set_meta.append(Int64(off))
        self.set_meta.append(Int64(len(entries)))
        self.set_meta.append(Int64(lo))
        return len(self.set_meta) // 3 - 1

    def tab(mut self, reg: Int, entries: List[Int64], lo: Int = 0):
        """reg = entries[reg - lo], in place (0 outside the table)."""
        var s = self.add_tab(entries, lo)
        self.ins(OP_TAB, reg, reg, s)

    def dict_has(self, d: Int) -> Bool:
        return self.dicts.has_derived(d)

    def truth(mut self, cond: Int) raises -> Int:
        """A register that is 1.0 where the condition is true and 0.0 where it is false or unknown."""
        var w = self.boolean(cond)
        if w.unk:
            var one = self.const_f(1.0)
            self.ins(24, w.reg, w.reg, one)
            self.free(one)
        return w.reg

    def blend(mut self, mask: Int, a: V, b: V) raises -> V:
        """mask != 0 ? a : b, value and NULL flag. Frees a and b; the mask stays."""
        var r = self.tmp()
        self.ins(OP_SELECT, r, mask, a.reg, b.reg)
        var nul = -1
        if a.nul >= 0 or b.nul >= 0:
            nul = self.tmp()
            var za = a.nul
            var zb = b.nul
            var zero = -1
            if za < 0 or zb < 0:
                zero = self.const_f(0.0)
            self.ins(OP_SELECT, nul, mask, za if za >= 0 else zero, zb if zb >= 0 else zero)
            if zero >= 0:
                self.free(zero)
        self.free(a.reg)
        self.free(b.reg)
        self.free(a.nul)
        self.free(b.nul)
        return V(r, nul, a.unk or b.unk, a.fl or b.fl)

    def guard_with(mut self, take: Int) raises -> Int:
        """The register that says where an expression is live: the outer guard AND `take` (a fresh register; -1 for no guard)."""
        if take < 0:
            return self.guard
        var g = self.tmp()
        if self.guard >= 0:
            self.ins(OP_AND, g, self.guard, take)
        else:
            self.p.emit(OP_IMAX, g, take, take)
        return g

    def flag_errors(mut self, bad: Int, nul: Int, why: String) raises:
        """Where `bad` is 1.0 on a row that is live (and not NULL) the query fails: the kernel's error flag (OP_FLAG)."""
        var t = self.tmp()
        if nul >= 0:
            var z = self.const_f(0.0)
            self.ins(OP_SELECT, t, nul, z, bad)
            self.free(z)
        else:
            self.p.emit(OP_IMAX, t, bad, bad)
        if self.guard >= 0:
            self.ins(OP_AND, t, t, self.guard)
        self.ins(OP_FLAG, t, t)
        self.free(t)
        self.p.add_conv(why)

    # -- text values
    # -- text values: the dictionary of a derived TEXT node is made once, before anything reads it
    def derive_words(mut self, node: Int, var words: List[String]):
        self.dicts.set_derived(self.rx[node].src, words^)

    def make_tfn(self, x: BExpr) -> TFn:
        var strs = List[String]()
        var ints = List[Int]()
        for it in x.items:
            if self.rx[it].op == B_STR:
                strs.append(self.rx[it].text)
            else:
                ints.append(Int(self.rx[it].value))
        return TFn(Int(x.value) & 0xFF, Int(x.value) >> 8, strs, ints)

    def tfn_word(mut self, t: TFn, w: String) raises -> String:
        if t.func == TF_LOWER or t.func == TF_UPPER:
            return self.map_case(w, t.func == TF_UPPER)
        return t.to_text(w)

    def prep_dict(mut self, e: Int) raises:
        """Makes the dictionary of TEXT node e when it is a derived one (a constant word, a text function's answers, a ||, a CASE over words), and first the
        dictionaries of the nodes it reads. Idempotent: a node used twice is made once."""
        var x = self.rx[e].copy()
        if x.kind != TEXT.value:
            return
        if x.op == B_STR:
            if not self.dicts.has_derived(x.src):
                var ws = List[String]()
                ws.append(x.text)
                self.derive_words(e, ws^)
            return
        if x.op == B_TFN:
            if e in self.ents:
                return
            var t0 = perf_counter_ns()
            self.prep_dict(x.a)
            var t = self.make_tfn(x)
            var d = self.rx[x.a].src
            var n = self.dicts.count(d)
            var ents = List[Int64](length=n, fill=0)
            var words = List[String]()
            var seen = Dict[String, Int]()
            var unk = List[UInt8](length=max(n, 1), fill=0)
            var any_unk = False
            for k in range(n):
                var src_word = self.dicts.word(d, k)
                if t.func == TF_REVERSE and not reverse_is_exact(src_word):
                    unk[k] = 1
                    any_unk = True
                var w = self.tfn_word(t, src_word)
                var hit = seen.get(w)
                if hit:
                    ents[k] = Int64(hit.value())
                else:
                    seen[w] = len(words)
                    ents[k] = Int64(len(words))
                    words.append(w)
            self.derive_words(e, words^)
            self.ents[e] = ents^
            if any_unk:
                self.undecided[e] = unk^
            self.p.text_tabs += 1
            self.p.text_words += n
            self.p.text_ns += Int(perf_counter_ns() - t0)
            return
        if x.op == B_CONCAT:
            if e in self.ents:
                return
            self.prep_concat(e, x)
            return
        if x.op == B_CASE or x.op == B_COALESCE:
            if e in self.unions:
                return
            var vals = List[Int]()
            if x.op == B_CASE:
                var nw = Int(x.value)
                for k in range(nw):
                    vals.append(x.items[2 * k + 1])
                vals.append(x.items[2 * nw])
            else:
                vals = x.items.copy()
            for v in vals:
                self.prep_dict(v)
            var u = self.make_union(vals)
            self.derive_words(e, u.words.copy())
            self.unions[e] = u^
            return
        if x.op == B_CAST:                                         # a boolean as text: 'false' and 'true' are the codes 0 and 1; a number or a date column: its range's words
            if self.rx[x.a].kind == BOOL.value:
                if not self.dicts.has_derived(x.src):
                    var ws = List[String]()
                    ws.append("false")
                    ws.append("true")
                    self.derive_words(e, ws^)
            elif e not in self.ents:
                self.prep_number_text(e, x)
            return

    def prep_concat(mut self, e: Int, x: BExpr) raises:
        var skip = x.value != 0
        var xa = self.rx[x.a].copy()
        var xb = self.rx[x.b].copy()
        var a_lit = xa.op == B_STR
        var b_lit = xb.op == B_STR
        var words = List[String]()
        var seen = Dict[String, Int]()
        var plus = 1 if skip else 0                               # concat(): a NULL word is '' (one more entry)
        if a_lit and b_lit:
            raise Error("internal: a || of two constants reached the program")
        if a_lit != b_lit:                                        # one value and one constant
            var vn = x.b if a_lit else x.a
            self.prep_dict(vn)
            var lit = xa.text if a_lit else xb.text
            var d = self.rx[vn].src
            var n = self.dicts.count(d)
            var ents = List[Int64](length=n + plus, fill=0)
            for k in range(n + plus):
                var w = self.dicts.word(d, k) if k < n else String()
                var s = lit + w if a_lit else w + lit
                var hit = seen.get(s)
                if hit:
                    ents[k] = Int64(hit.value())
                else:
                    seen[s] = len(words)
                    ents[k] = Int64(len(words))
                    words.append(s)
            self.p.text_words += n
            self.derive_words(e, words^)
            self.ents[e] = ents^
            self.p.text_tabs += 1
            return
        self.prep_dict(x.a)
        self.prep_dict(x.b)
        var da = self.rx[x.a].src
        var db = self.rx[x.b].src
        var na = self.dicts.count(da)
        var nb = self.dicts.count(db)
        var total = (na + plus) * (nb + plus)
        if total > CONCAT_LIMIT:
            raise Error("concatenating two columns whose dictionaries hold " + String(na) + " and " + String(nb) + " distinct words needs " + String(total) +
                        " combinations (the most is " + String(CONCAT_LIMIT) + "): omxcore builds the result's words once per pair of words, never per row")
        var ents = List[Int64](length=max(total, 1), fill=0)
        for i in range(na + plus):
            var wa = self.dicts.word(da, i) if i < na else String()
            for j in range(nb + plus):
                var s = wa + (self.dicts.word(db, j) if j < nb else String())
                var hit = seen.get(s)
                var k = i * (nb + plus) + j
                if hit:
                    ents[k] = Int64(hit.value())
                else:
                    seen[s] = len(words)
                    ents[k] = Int64(len(words))
                    words.append(s)
        self.p.text_words += na + nb
        self.derive_words(e, words^)
        self.ents[e] = ents^
        self.p.text_tabs += 1

    def make_union(mut self, vals: List[Int]) raises -> TextUnion:
        """The dictionary of a CASE / COALESCE over words: every constant word and every word of each value's dictionary, once; per value the codes in it."""
        var u = TextUnion()
        for j in range(len(vals)):
            var x = self.rx[vals[j]].copy()
            if x.op == B_NULL:
                u.tabs.append(List[Int64]())
                u.lit.append(-1)
                u.identity.append(False)
                continue
            if x.op == B_STR:
                u.tabs.append(List[Int64]())
                u.lit.append(u.add(x.text))
                u.identity.append(False)
                continue
            var d = x.src
            var n = self.dicts.count(d)
            var tab = List[Int64](length=n, fill=0)
            var identity = True
            for k in range(n):
                var c = u.add(self.dicts.word(d, k))
                tab[k] = Int64(c)
                identity = identity and c == k
            u.tabs.append(tab^)
            u.lit.append(-1)
            u.identity.append(identity)
        return u^

    def tfn_value(mut self, e: Int) raises -> V:
        """A text function over a TEXT value: evaluated on each word of the value's dictionary once, the answers a table the kernel looks the codes up in."""
        var x = self.rx[e].copy()
        if x.kind == TEXT.value:
            self.prep_dict(e)
        elif e not in self.ents:
            self.prep_dict(x.a)
            var t = self.make_tfn(x)
            var d = self.rx[x.a].src
            var n = self.dicts.count(d)
            var ents = List[Int64](length=n, fill=0)
            var wp = self.dicts.word_ptrs(d)
            _words_int(t, wp[0], wp[1], wp[2], wp[3], Int(ents.unsafe_ptr()))
            self.ents[e] = ents^
            self.p.text_tabs += 1
            self.p.text_words += n
        var v = self.value(x.a)
        if e in self.undecided:                                      # words DuckDB answers by grapheme clusters: an error where a row holds one
            var fu = self.tmp()
            self.ins(OP_I2F, fu, v.reg)
            var set_u = self.add_set(self.undecided[e].copy(), 0)
            self.ins(OP_IN_SET, fu, fu, set_u)
            self.flag_errors(fu, v.nul, "reverse() of a word with combining marks, joiners or an emoji sequence: DuckDB reverses by grapheme clusters, omxcore by code points")
            self.free(fu)
        self.tab(v.reg, self.ents[e].copy())
        return V(v.reg, v.nul, False)

    def tfn_bool(mut self, e: Int) raises -> V:
        var x = self.rx[e].copy()
        var t0 = perf_counter_ns()
        self.prep_dict(x.a)
        var t = self.make_tfn(x)
        var d = self.rx[x.a].src
        var n = self.dicts.count(d)
        var v = self.value(x.a)
        var bytes = List[UInt8](length=max(n, 1), fill=0)
        var ilike = t.func == TF_LIKE and (t.flags & 1) != 0
        if ilike:
            for k in range(n):
                var w = self.map_case(self.dicts.word(d, k), False)
                bytes[k] = 1 if t.to_bool(w) else 0
        else:
            var wp = self.dicts.word_ptrs(d)
            _words_bool(t, wp[0], wp[1], wp[2], wp[3], Int(bytes.unsafe_ptr()))
            _ = bytes.unsafe_ptr()                                   # (the list lives to here: the loop above wrote through its address)
        var r = self.tmp()
        self.ins(OP_I2F, r, v.reg)
        self.free(v.reg)
        var set_at = self.add_set(bytes, 0)
        self.ins(OP_IN_SET, r, r, set_at)
        self.p.text_tabs += 1
        self.p.text_words += n
        self.p.text_ns += Int(perf_counter_ns() - t0)
        return self.with_unknown(r, v.nul)

    def concat_value(mut self, e: Int) raises -> V:
        """a || b: each pair of words is one word of the result's dictionary (one side a constant: one entry per word of the other)."""
        var x = self.rx[e].copy()
        self.prep_dict(e)
        var skip = x.value != 0
        var a_lit = self.rx[x.a].op == B_STR
        var b_lit = self.rx[x.b].op == B_STR
        var plus = 1 if skip else 0
        if a_lit != b_lit:
            var vn = x.b if a_lit else x.a
            var n = self.dicts.count(self.rx[vn].src)
            var v = self.value(vn)
            var nul = v.nul
            if skip and v.nul >= 0:                               # a NULL reads as the one more entry
                var nc = self.const_i(Int64(n))
                self.ins(OP_SELECT, v.reg, v.nul, nc, v.reg)
                self.free(nc)
                self.free(v.nul)
                nul = -1
            self.tab(v.reg, self.ents[e].copy())
            return V(v.reg, nul, False)
        var na = self.dicts.count(self.rx[x.a].src)
        var nb = self.dicts.count(self.rx[x.b].src)
        var va = self.value(x.a)
        var vb = self.value(x.b)
        var nul = -1
        if skip:
            if va.nul >= 0:
                var nc = self.const_i(Int64(na))
                self.ins(OP_SELECT, va.reg, va.nul, nc, va.reg)
                self.free(nc)
                self.free(va.nul)
            if vb.nul >= 0:
                var nc2 = self.const_i(Int64(nb))
                self.ins(OP_SELECT, vb.reg, vb.nul, nc2, vb.reg)
                self.free(nc2)
                self.free(vb.nul)
        else:
            nul = self.or_nul(va.nul, vb.nul)
        var m = self.const_i(Int64(nb + plus))
        self.ins(OP_IMUL, va.reg, va.reg, m)
        self.free(m)
        self.ins(OP_IADD, va.reg, va.reg, vb.reg)
        self.free(vb.reg)
        self.tab(va.reg, self.ents[e].copy())
        return V(va.reg, nul, False)

    def branch(mut self, node: Int, j: Int, u: TextUnion, text: Bool, as_bool: Bool) raises -> V:
        """The value of one result of a CASE / COALESCE, in the node's type (a TEXT value in the union's codes)."""
        if as_bool:
            return self.boolean(node)
        var x = self.rx[node].copy()
        if not text or x.op == B_NULL:
            if x.op == B_NULL:
                return V(self.const_i(0), self.const_f(1.0), False, x.kind == FLOAT.value)
            return self.value(node)
        if x.op == B_STR:
            return V(self.const_i(Int64(u.lit[j])), -1, False)
        var v = self.value(node)
        if not u.is_identity(j):
            self.tab(v.reg, u.tabs[j])
            self.p.text_tabs += 1
            self.p.text_words += len(u.tabs[j])
        return v^

    def case_generic(mut self, e: Int, as_bool: Bool) raises -> V:
        var x = self.rx[e].copy()
        var nw = Int(x.value)
        var vals = List[Int]()
        for k in range(nw):
            vals.append(x.items[2 * k + 1])
        vals.append(x.items[2 * nw])
        var text = x.kind == TEXT.value
        if text:
            self.prep_dict(e)
        var u = self.unions[e].copy() if text else TextUnion()
        var saved = self.guard
        var done = -1
        var acc = V(-1, -1, False)
        for k in range(nw):
            var c = self.truth(x.items[2 * k])
            var take = c
            var own_take = False
            if done >= 0:                                          # a row that an earlier WHEN took is not this WHEN's
                var nd = self.tmp()
                self.ins(OP_NOT, nd, done)
                take = self.tmp()
                self.ins(OP_AND, take, c, nd)
                self.free(nd)
                own_take = True
            self.guard = self.guard_with(take)
            var tv = self.branch(vals[k], k, u, text, as_bool)
            self.free(self.guard)
            self.guard = saved
            if k == 0:
                acc = tv.copy()
                done = self.tmp()
                self.p.emit(OP_IMAX, done, c, c)
            else:
                acc = self.blend(take, tv, acc)
                self.ins(OP_OR, done, done, c)
            self.free(c)
            if own_take:
                self.free(take)
        var nd3 = self.tmp()
        self.ins(OP_NOT, nd3, done)
        self.guard = self.guard_with(nd3)
        var ev = self.branch(vals[nw], nw, u, text, as_bool)
        self.free(self.guard)
        self.guard = saved
        self.free(nd3)
        var out = self.blend(done, acc, ev)
        self.free(done)
        return out^

    def coalesce_generic(mut self, e: Int, as_bool: Bool) raises -> V:
        var x = self.rx[e].copy()
        var text = x.kind == TEXT.value
        if text:
            self.prep_dict(e)
        var u = self.unions[e].copy() if text else TextUnion()
        var saved = self.guard
        var acc = self.branch(x.items[0], 0, u, text, as_bool)
        for j in range(1, len(x.items)):
            var null_mask = acc.nul
            var made = False
            if as_bool:                                            # a boolean is NULL where it is unknown (0.5)
                made = True
                null_mask = self.tmp()
                var half = self.const_f(0.5)
                self.ins(24, null_mask, acc.reg, half)
                self.free(half)
            elif acc.nul < 0:
                break                                              # never NULL: the rest is never read
            self.guard = self.guard_with(null_mask)
            var vj = self.branch(x.items[j], j, u, text, as_bool)
            self.free(self.guard)
            self.guard = saved
            var res = self.tmp()
            self.ins(OP_SELECT, res, null_mask, vj.reg, acc.reg)
            var nul = -1
            if not as_bool and vj.nul >= 0:
                nul = self.tmp()
                self.ins(OP_AND, nul, acc.nul, vj.nul)
            self.free(acc.reg)
            self.free(vj.reg)
            if not as_bool:
                self.free(acc.nul)
            self.free(vj.nul)
            if made:
                self.free(null_mask)
            acc = V(res, nul, acc.unk or vj.unk, acc.fl or vj.fl)
        return acc^

    # -- comparisons
    def operand_value(mut self, node: Int) raises -> V:
        """A comparison's operand as a value register: a condition (BOOL, not a column or a constant) is 0 / 1 where it is true / false and NULL where unknown."""
        var x = self.rx[node].copy()
        if x.kind == BOOL.value and x.op != B_COL and x.op != B_LIT:
            var b = self.boolean(node)
            var nul = -1
            var r = b.reg
            if b.unk:
                var half = self.const_f(0.5)
                nul = self.tmp()
                self.ins(24, nul, b.reg, half)
                self.free(half)
                var one = self.const_f(1.0)
                self.ins(24, b.reg, b.reg, one)
                self.free(one)
            var iv = self.tmp()
            self.ins(OP_F2I, iv, r)
            self.free(r)
            return V(iv, nul, False)
        return self.value(node)

    def cmp_raw(mut self, a_node: Int, b_node: Int, op: Int) raises -> Tuple[Int, Int, Int]:
        """a <op> b over definite values: (r, na, nb), r 1.0 where it holds, na / nb the NULL flags of the operands (-1: never NULL)."""
        var lhs = self.rx[a_node].copy()
        if lhs.kind == TEXT.value:
            return self.text_cmp_raw(a_node, b_node, op)
        var a = self.operand_value(a_node)
        var b = self.operand_value(b_node)
        var r = a.reg
        if a.fl:                                               # DuckDB's order: NaN equals itself and is above every number; -0.0 equals 0.0
            self.ins(op, r, a.reg, b.reg, NAN_AWARE)
        else:
            self.ins(OP_ICMP, r, a.reg, b.reg, op)
        self.free(b.reg)
        return (r, a.nul, b.nul)

    def text_cmp_raw(mut self, a_node: Int, b_node: Int, op: Int) raises -> Tuple[Int, Int, Int]:
        var xb = self.rx[b_node].copy()
        self.prep_dict(a_node)
        if xb.op != B_STR:
            self.prep_dict(b_node)
        var d = self.rx[a_node].src
        var v = self.value(a_node)
        var r = v.reg
        if xb.op == B_STR:                                     # a word of the dictionary: its code, or the codes of the words in range
            var lit = xb.text
            if op == C_EQ or op == C_NE:
                var code = self.dicts.code_of(d, lit)
                var k = self.const_i(Int64(code))              # -1: no row holds the word
                self.ins(OP_ICMP, r, r, k, op)
                self.free(k)
            else:
                var n = self.dicts.count(d)
                var bytes = List[UInt8](length=max(n, 1), fill=0)
                for code in range(n):
                    var w = self.dicts.word(d, code)
                    var hit: Bool
                    if op == C_LT:
                        hit = _text_less(w, lit)
                    elif op == C_GT:
                        hit = _text_less(lit, w)
                    elif op == C_LE:
                        hit = not _text_less(lit, w)
                    else:
                        hit = not _text_less(w, lit)
                    bytes[code] = 1 if hit else 0
                var f = self.tmp()
                self.ins(OP_I2F, f, r)
                self.free(r)
                r = f
                var set_at = self.add_set(bytes, 0)
                self.ins(OP_IN_SET, r, r, set_at)
            return (r, v.nul, -1)
        # two values: their codes are in different dictionaries, so one is translated into the other's (equality), or both into the words' common ranks
        var db = self.rx[b_node].src
        var b = self.value(b_node)
        if op == C_EQ or op == C_NE:
            if d != db:
                var nb = self.dicts.count(db)
                var tb = List[Int64](length=nb, fill=0)
                for kb in range(nb):
                    var ca = self.dicts.code_of(d, self.dicts.word(db, kb))
                    tb[kb] = Int64(ca) if ca >= 0 else Int64(-2 - kb)         # a word the other side lacks equals nothing (a negative that is nobody's code)
                self.tab(b.reg, tb)
                self.p.text_tabs += 1
                self.p.text_words += nb
            self.ins(OP_ICMP, r, r, b.reg, op)
        else:
            var union_words = List[String]()
            var at = Dict[String, Int]()
            var na = self.dicts.count(d)
            var ia = List[Int](length=na, fill=0)
            for k in range(na):
                var w = self.dicts.word(d, k)
                var hit = at.get(w)
                if hit:
                    ia[k] = hit.value()
                else:
                    at[w] = len(union_words)
                    ia[k] = len(union_words)
                    union_words.append(w)
            var nb = self.dicts.count(db)
            var ib = List[Int](length=nb, fill=0)
            for k in range(nb):
                var w = self.dicts.word(db, k)
                var hit = at.get(w)
                if hit:
                    ib[k] = hit.value()
                else:
                    at[w] = len(union_words)
                    ib[k] = len(union_words)
                    union_words.append(w)
            var ranks = text_ranks(union_words)
            var ta = List[Int64](length=na, fill=0)
            for k in range(na):
                ta[k] = ranks[ia[k]]
            var tb2 = List[Int64](length=nb, fill=0)
            for k in range(nb):
                tb2[k] = ranks[ib[k]]
            self.tab(r, ta)
            self.tab(b.reg, tb2)
            self.p.text_tabs += 2
            self.p.text_words += na + nb
            self.ins(OP_ICMP, r, r, b.reg, op)
        self.free(b.reg)
        return (r, v.nul, b.nul)

    def distinct_bool(mut self, x: BExpr) raises -> V:
        """a IS [NOT] DISTINCT FROM b: NULL is a value, so two NULLs are not distinct, a NULL and a value are, two values are when they differ."""
        var t = self.cmp_raw(x.a, x.b, C_NE)
        var r = t[0]
        var na = t[1]
        var nb = t[2]
        if na >= 0 or nb >= 0:
            var anyn = self.or_nul_copy(na, nb)
            var differ = self.tmp()                              # exactly one is NULL
            var zero = self.const_f(0.0)
            self.ins(25, differ, na if na >= 0 else zero, nb if nb >= 0 else zero, 0)
            self.free(zero)
            self.ins(OP_SELECT, r, anyn, differ, r)
            self.free(differ)
            self.free(anyn)
            self.free(na)
            self.free(nb)
        if x.value != 0:                                         # IS NOT DISTINCT FROM
            self.ins(OP_NOT, r, r)
        return V(r, -1, False)

    def or_nul_copy(mut self, a: Int, b: Int) raises -> Int:
        """A new register holding a OR b (a and b stay)."""
        var r = self.tmp()
        if a >= 0 and b >= 0:
            self.ins(OP_OR, r, a, b)
        elif a >= 0:
            self.p.emit(OP_IMAX, r, a, a)
        else:
            self.p.emit(OP_IMAX, r, b, b)
        return r

    # -- numbers
    def fconst(mut self, x: Float64) raises -> Int:
        return self.const_f(x)

    def round_half_away(mut self, r: Int) raises:
        """r = the float r rounded to a whole number, halves away from zero (in place; exact: the fraction of a double is exact)."""
        var neg = self.tmp()
        var zero = self.const_f(0.0)
        self.ins(C_LT, neg, r, zero, 0)
        var a = self.tmp()
        self.ins(OP_SUBF, a, zero, r)                            # -r
        self.ins(OP_SELECT, a, neg, a, r)                        # |r|
        var f = self.tmp()
        self.ins(OP_FLOOR, f, a)
        var fr = self.tmp()
        self.ins(OP_SUBF, fr, a, f)
        var half = self.const_f(0.5)
        var up = self.tmp()
        self.ins(C_GE, up, fr, half, 0)
        var one = self.const_f(1.0)
        self.ins(OP_ADD, a, f, one)
        self.ins(OP_SELECT, f, up, a, f)                         # |r| rounded
        self.ins(OP_SUBF, a, zero, f)
        self.ins(OP_SELECT, r, neg, a, f)
        self.free(neg)
        self.free(zero)
        self.free(a)
        self.free(f)
        self.free(fr)
        self.free(half)
        self.free(up)
        self.free(one)

    def round_half_even(mut self, r: Int) raises:
        """r = the float r rounded to a whole number, halves to the even one (in place)."""
        var f = self.tmp()
        self.ins(OP_FLOOR, f, r)
        var fr = self.tmp()
        self.ins(OP_SUBF, fr, r, f)
        var half = self.const_f(0.5)
        var gt = self.tmp()
        self.ins(C_GT, gt, fr, half, 0)
        var eq = self.tmp()
        self.ins(C_EQ, eq, fr, half, 0)
        var h2 = self.const_f(0.5)
        var q = self.tmp()
        self.ins(OP_MUL, q, f, h2)
        self.ins(OP_FLOOR, q, q)
        var two = self.const_f(2.0)
        self.ins(OP_MUL, q, q, two)                              # 2 * floor(f / 2)
        var odd = self.tmp()
        self.ins(C_NE, odd, q, f, 0)
        self.ins(OP_AND, eq, eq, odd)
        self.ins(OP_OR, gt, gt, eq)
        var one = self.const_f(1.0)
        self.ins(OP_ADD, q, f, one)
        self.ins(OP_SELECT, r, gt, q, f)
        self.free(f)
        self.free(fr)
        self.free(half)
        self.free(gt)
        self.free(eq)
        self.free(h2)
        self.free(q)
        self.free(two)
        self.free(odd)
        self.free(one)

    def iabs_into(mut self, r: Int) raises -> Tuple[Int, Int]:
        """(|r|, the register that is 1.0 where r is negative): new registers; r stays."""
        var neg = self.tmp()
        var zero = self.const_i(0)
        self.ins(OP_ICMP, neg, r, zero, C_LT)
        var a = self.tmp()
        self.ins(OP_ISUB, a, zero, r)
        self.ins(OP_SELECT, a, neg, a, r)
        self.free(zero)
        return (a, neg)

    def idiv_round_away(mut self, r: Int, p: Int64) raises:
        """r = r / p rounded to the nearest whole number, halves away from zero (in place; int registers)."""
        var ab = self.iabs_into(r)
        var a = ab[0]
        var neg = ab[1]
        var half = self.const_i(p // 2)
        self.ins(OP_IADD, a, a, half)
        self.free(half)
        var pc = self.const_i(p)
        self.ins(OP_IDIV, a, a, pc)
        self.free(pc)
        var zero = self.const_i(0)
        var na = self.tmp()
        self.ins(OP_ISUB, na, zero, a)
        self.ins(OP_SELECT, r, neg, na, a)
        self.free(zero)
        self.free(na)
        self.free(a)
        self.free(neg)

    def out_of_range(mut self, r: Int, lo: Int64, hi: Int64) raises -> Int:
        """A new register: 1.0 where the int register r is below lo or above hi."""
        var l = self.const_i(lo)
        var bad = self.tmp()
        self.ins(OP_ICMP, bad, r, l, C_LT)
        self.free(l)
        var h = self.const_i(hi)
        var over = self.tmp()
        self.ins(OP_ICMP, over, r, h, C_GT)
        self.free(h)
        self.ins(OP_OR, bad, bad, over)
        self.free(over)
        return bad

    def cast_finish(mut self, v: V, bad: Int, try_: Bool, why: String) raises -> V:
        """A cast that can fail: TRY_CAST makes the failing rows NULL, CAST makes them an error."""
        if bad < 0:
            return v.copy()
        if try_:
            var nul = self.or_nul(v.nul, bad)
            return V(v.reg, nul, False, v.fl)
        self.flag_errors(bad, v.nul, why)
        self.free(bad)
        return v.copy()

    def cast_value(mut self, e: Int) raises -> V:
        var x = self.rx[e].copy()
        var src = self.rx[x.a].copy()
        var fk = src.kind
        var fs = src.scale
        var tk = x.kind
        var ts = x.scale
        var flags = Int(x.value)
        var try_ = (flags & 1) != 0
        var bits = (flags >> 1) & 0x7F
        var prec = flags >> 8
        if fk == TEXT.value:
            return self.cast_from_text(e, x)
        if tk == TEXT.value and fk != BOOL.value:
            return self.cast_number_text(e, x)
        var v = self.value(x.a)
        var bad = -1
        if tk == TEXT.value:                                       # a boolean: 'false' and 'true' are codes 0 and 1
            self.prep_dict(e)
            return v^
        if tk == INT.value or tk == DECIMAL.value:
            if fk == FLOAT.value:
                if ts > 0:
                    var k = self.const_f(Float64(Int(pow10_checked(ts))))
                    self.ins(OP_MUL, v.reg, v.reg, k)
                    self.free(k)
                if tk == DECIMAL.value:
                    self.round_half_away(v.reg)
                else:
                    self.round_half_even(v.reg)
                var limit = Float64(Int(pow10_checked(prec))) if tk == DECIMAL.value else (9223372036854775808.0 if bits >= 64 else Float64(Int(1) << (bits - 1)))
                var lo = self.const_f(-limit)
                var hi = self.const_f(limit)
                var fin = self.tmp()                                # in range (NaN is neither): everything else is a failure
                var ge = self.tmp()
                self.ins(C_GE, ge, v.reg, lo, 0)
                self.ins(C_LT, fin, v.reg, hi, 0)
                self.ins(OP_AND, fin, fin, ge)
                self.free(ge)
                self.free(lo)
                self.free(hi)
                if tk == DECIMAL.value:                              # a DECIMAL(p, s) holds at most 10^p - 1: the limit itself is out too
                    var lo2 = self.const_f(-limit)
                    var gt = self.tmp()
                    self.ins(C_GT, gt, v.reg, lo2, 0)
                    self.ins(OP_AND, fin, fin, gt)
                    self.free(gt)
                    self.free(lo2)
                bad = self.tmp()
                self.ins(OP_NOT, bad, fin)
                self.free(fin)
                var z = self.const_f(0.0)
                self.ins(OP_SELECT, v.reg, bad, z, v.reg)           # a failing row holds 0, so the conversion below reads a number
                self.free(z)
                self.ins(OP_F2I, v.reg, v.reg)
                v.fl = False                                         # int64 bits now: a group key must not read -2 as a NaN
                return self.cast_finish(v, bad, try_, String(self.rx[e].text))
            elif fk == INT.value or fk == DECIMAL.value or fk == BOOL.value:
                var from_scale = fs if fk != BOOL.value else 0
                if ts > from_scale:
                    var m = pow10_checked(ts - from_scale)
                    var cap = (pow10_checked(prec) - 1) // m if tk == DECIMAL.value else Int64.MAX // m
                    var pre = self.out_of_range(v.reg, -cap, cap)     # a value that cannot fit after scaling fails before it overflows the multiplication
                    var z0 = self.const_i(0)
                    self.ins(OP_SELECT, v.reg, pre, z0, v.reg)
                    self.free(z0)
                    bad = pre
                    var k = self.const_i(m)
                    self.ins(OP_IMUL, v.reg, v.reg, k)
                    self.free(k)
                elif ts < from_scale:
                    self.idiv_round_away(v.reg, pow10_checked(from_scale - ts))
            if tk == DECIMAL.value:
                var lim = pow10_checked(prec)
                var b2 = self.out_of_range(v.reg, -(lim - 1), lim - 1)
                bad = b2 if bad < 0 else self.or_nul(bad, b2)
            elif tk == INT.value and bits < 64:
                var lim2 = Int64(1) << Int64(bits - 1)
                var b3 = self.out_of_range(v.reg, -lim2, lim2 - 1)
                bad = b3 if bad < 0 else self.or_nul(bad, b3)
            return self.cast_finish(v, bad, try_, String(self.rx[e].text))
        if tk == FLOAT.value:
            if fk == FLOAT.value:
                var fbad = -1
                if ts == F32_COLUMN and fs != F32_COLUMN:           # a finite double past the largest 32-bit float is a failure (inf and NaN stay)
                    var ab = self.tmp()
                    var zero = self.const_f(0.0)
                    self.ins(OP_SUBF, ab, zero, v.reg)
                    var neg = self.tmp()
                    self.ins(C_LT, neg, v.reg, zero, 0)
                    self.ins(OP_SELECT, ab, neg, ab, v.reg)         # |x|
                    var big = self.const_f(3.4028235677973366e38)   # the midpoint to the next float: what rounds to inf
                    var over = self.tmp()
                    self.ins(C_GE, over, ab, big, 0)
                    var inf = self.const_f(Float64(from_bits=UInt64(0x7FF0000000000000)))
                    var fin = self.tmp()
                    self.ins(C_LT, fin, ab, inf, 0)
                    self.ins(OP_AND, over, over, fin)
                    fbad = over
                    self.free(ab)
                    self.free(zero)
                    self.free(neg)
                    self.free(big)
                    self.free(inf)
                    self.free(fin)
                    var zf = self.const_f(0.0)
                    self.ins(OP_SELECT, v.reg, over, zf, v.reg)
                    self.free(zf)
                    self.ins(OP_F32, v.reg, v.reg)
                return self.cast_finish(V(v.reg, v.nul, False, True), fbad, try_, String(self.rx[e].text))
            raise Error("internal: a cast of " + String(fk) + " to FLOAT reached the program")
        raise Error("internal: a cast from " + String(fk) + " to " + String(tk) + " reached the program")

    def prep_number_text(mut self, e: Int, x: BExpr) raises:
        """The words of CAST(column AS VARCHAR) for an INT, DECIMAL or DATE column: the text of every value from the column's lowest to its highest (known from
        its statistics or one pass), made once; a date outside years 1 to 9999 (DuckDB writes `(BC)` and five digits) is marked, an error where a row holds it."""
        var src = self.rx[x.a].copy()
        var c = Int(src.value)
        var r = self.dicts.range_of(c)
        if not r[0]:
            raise Error("CAST(" + src.text + " AS VARCHAR) needs the lowest and highest value of the column: they are not known here (a file without statistics, or a joined column)")
        var span = Int(r[2]) - Int(r[1]) + 1
        if span > CAST_TEXT_LIMIT:
            raise Error("CAST(" + src.text + " AS VARCHAR): the column spans " + String(span) + " values, and the words of all of them are made once (the most is " + String(CAST_TEXT_LIMIT) + ")")
        var words = List[String]()
        var bad = List[UInt8](length=max(span, 1), fill=0)
        var any_bad = False
        for k in range(span):
            var v = Int(r[1]) + k
            if src.kind == DATE.value:
                if v < -719162 or v > 2932896:
                    words.append(String())
                    bad[k] = 1
                    any_bad = True
                else:
                    words.append(date_text(v))
            else:
                words.append(decimal_text(Int64(v), src.scale))
        self.derive_words(e, words^)
        if any_bad:
            self.undecided[e] = bad^
        var base = List[Int64]()
        base.append(r[1])
        self.ents[e] = base^                                       # (the lowest value: the code of a value is the value less this)
        self.p.text_tabs += 1
        self.p.text_words += span

    def cast_number_text(mut self, e: Int, x: BExpr) raises -> V:
        self.prep_dict(e)
        var src = self.rx[x.a].copy()
        var lo = self.ents[e][0]
        var v = self.value(x.a)
        var lc = self.const_i(lo)
        self.ins(OP_ISUB, v.reg, v.reg, lc)
        self.free(lc)
        if e in self.undecided:
            var fl = self.tmp()
            self.ins(OP_I2F, fl, v.reg)
            var set_at = self.add_set(self.undecided[e].copy(), 0)
            self.ins(OP_IN_SET, fl, fl, set_at)
            self.flag_errors(fl, v.nul, "CAST(" + src.text + " AS VARCHAR) of a date before year 1 or after year 9999 (DuckDB writes those with `(BC)` or five digits)")
            self.free(fl)
        return V(v.reg, v.nul, False)

    def cast_from_text(mut self, e: Int, x: BExpr) raises -> V:
        """A word's value as a number, date or boolean, parsed once per word of the dictionary; a word that does not read as one is an error where a row holds it
        (TRY_CAST: NULL)."""
        self.prep_dict(x.a)
        var d = self.rx[x.a].src
        var n = self.dicts.count(d)
        var flags = Int(x.value)
        var try_ = (flags & 1) != 0
        var bits = (flags >> 1) & 0x7F
        var prec = flags >> 8
        var v = self.value(x.a)
        var ents = List[Int64](length=n, fill=0)
        var badb = List[UInt8](length=max(n, 1), fill=0)
        var unsb = List[UInt8](length=max(n, 1), fill=0)
        var any_bad = False
        var any_unsure = False
        for k in range(n):
            var r = parse_text(self.dicts.word(d, k), x.kind, x.scale, bits, prec)
            ents[k] = r.value
            if r.unsure:
                unsb[k] = 1
                any_unsure = True
            elif not r.ok:
                badb[k] = 1
                any_bad = True
        var fl = -1
        if any_bad:                                                  # the codes of the words that do not read, tested before the codes are replaced by values
            fl = self.tmp()
            self.ins(OP_I2F, fl, v.reg)
            var set_at = self.add_set(badb, 0)
            self.ins(OP_IN_SET, fl, fl, set_at)
        if any_unsure:                                               # words whose meaning in DuckDB is a quirk: an error even for TRY_CAST
            var fu = self.tmp()
            self.ins(OP_I2F, fu, v.reg)
            var set_u = self.add_set(unsb, 0)
            self.ins(OP_IN_SET, fu, fu, set_u)
            self.flag_errors(fu, v.nul, "the word is one of DuckDB's text quirks that omxcore does not read (a date such as `infinity`, a BC date, a decimal with an exponent): " + String(self.rx[e].text))
            self.free(fu)
        self.tab(v.reg, ents)
        self.p.text_tabs += 1
        self.p.text_words += n
        return self.cast_finish(V(v.reg, v.nul, False, x.kind == FLOAT.value), fl, try_, String(self.rx[e].text))

    # -- dates
    def civil(mut self, z: Int) raises -> Tuple[Int, Int, Int]:
        """(year, month, day) registers of the day-number register z (floor divisions: right for dates before 1970). z stays."""
        var t = self.tmp()
        var c = self.const_i(719468)
        self.ins(OP_IADD, t, z, c)
        self.free(c)
        var era = self.tmp()
        var k = self.const_i(146097)
        self.ins(OP_IDIV, era, t, k)
        var doe = self.tmp()
        self.ins(OP_IMUL, doe, era, k)
        self.ins(OP_ISUB, doe, t, doe)
        self.free(k)
        self.free(t)
        var a = self.tmp()
        var b = self.tmp()
        var yoe = self.tmp()
        var k1 = self.const_i(1460)
        self.ins(OP_IDIV, a, doe, k1)
        self.free(k1)
        self.ins(OP_ISUB, yoe, doe, a)
        var k2 = self.const_i(36524)
        self.ins(OP_IDIV, a, doe, k2)
        self.free(k2)
        self.ins(OP_IADD, yoe, yoe, a)
        var k3 = self.const_i(146096)
        self.ins(OP_IDIV, a, doe, k3)
        self.free(k3)
        self.ins(OP_ISUB, yoe, yoe, a)
        var k4 = self.const_i(365)
        self.ins(OP_IDIV, yoe, yoe, k4)                           # yoe
        var doy = self.tmp()
        self.ins(OP_IMUL, a, yoe, k4)
        self.free(k4)
        var k5 = self.const_i(4)
        self.ins(OP_IDIV, b, yoe, k5)
        self.free(k5)
        self.ins(OP_IADD, a, a, b)
        var k6 = self.const_i(100)
        self.ins(OP_IDIV, b, yoe, k6)
        self.free(k6)
        self.ins(OP_ISUB, a, a, b)
        self.ins(OP_ISUB, doy, doe, a)                            # doy
        self.free(doe)
        var y = self.tmp()
        var k7 = self.const_i(400)
        self.ins(OP_IMUL, y, era, k7)
        self.free(k7)
        self.free(era)
        self.ins(OP_IADD, y, y, yoe)                              # year before the month shift
        self.free(yoe)
        var mp = self.tmp()
        var k8 = self.const_i(5)
        self.ins(OP_IMUL, mp, doy, k8)
        self.free(k8)
        var k9 = self.const_i(2)
        self.ins(OP_IADD, mp, mp, k9)
        self.free(k9)
        var k10 = self.const_i(153)
        self.ins(OP_IDIV, mp, mp, k10)                            # mp
        var d = self.tmp()
        self.ins(OP_IMUL, a, mp, k10)
        self.free(k10)
        var k11 = self.const_i(2)
        self.ins(OP_IADD, a, a, k11)
        self.free(k11)
        var k12 = self.const_i(5)
        self.ins(OP_IDIV, a, a, k12)
        self.free(k12)
        self.ins(OP_ISUB, d, doy, a)
        var one = self.const_i(1)
        self.ins(OP_IADD, d, d, one)                              # day
        self.free(doy)
        var m = self.tmp()
        var ten = self.const_i(10)
        var ge = self.tmp()
        self.ins(OP_ICMP, ge, mp, ten, C_GE)
        self.free(ten)
        self.ins(OP_F2I, ge, ge)                                  # 1 where mp >= 10
        var k13 = self.const_i(12)
        self.ins(OP_IMUL, b, ge, k13)
        self.free(k13)
        var three = self.const_i(3)
        self.ins(OP_IADD, m, mp, three)
        self.free(three)
        self.ins(OP_ISUB, m, m, b)                                # month
        self.free(ge)
        self.free(mp)
        var two = self.const_i(2)
        var le2 = self.tmp()
        self.ins(OP_ICMP, le2, m, two, C_LE)
        self.free(two)
        self.ins(OP_F2I, le2, le2)
        self.ins(OP_IADD, y, y, le2)                              # year
        self.free(le2)
        self.free(one)
        self.free(a)
        self.free(b)
        return (y, m, d)

    def days_from_civil_regs(mut self, y: Int, m: Int, d: Int) raises -> Int:
        """The day-number register of the date in registers y, m, d (they stay)."""
        var two = self.const_i(2)
        var le2 = self.tmp()
        self.ins(OP_ICMP, le2, m, two, C_LE)
        self.free(two)
        self.ins(OP_F2I, le2, le2)                                # 1 where m <= 2
        var yy = self.tmp()
        self.ins(OP_ISUB, yy, y, le2)
        var era = self.tmp()
        var k400 = self.const_i(400)
        self.ins(OP_IDIV, era, yy, k400)
        var yoe = self.tmp()
        self.ins(OP_IMUL, yoe, era, k400)
        self.free(k400)
        self.ins(OP_ISUB, yoe, yy, yoe)
        self.free(yy)
        var t = self.tmp()                                        # m - 3 + 12 * (m <= 2)
        var k12 = self.const_i(12)
        self.ins(OP_IMUL, t, le2, k12)
        self.free(k12)
        self.free(le2)
        self.ins(OP_IADD, t, t, m)
        var k3 = self.const_i(3)
        self.ins(OP_ISUB, t, t, k3)
        self.free(k3)
        var doy = self.tmp()
        var k153 = self.const_i(153)
        self.ins(OP_IMUL, doy, t, k153)
        self.free(k153)
        self.free(t)
        var k2 = self.const_i(2)
        self.ins(OP_IADD, doy, doy, k2)
        self.free(k2)
        var k5 = self.const_i(5)
        self.ins(OP_IDIV, doy, doy, k5)
        self.free(k5)
        self.ins(OP_IADD, doy, doy, d)
        var one = self.const_i(1)
        self.ins(OP_ISUB, doy, doy, one)
        self.free(one)
        var doe = self.tmp()
        var k365 = self.const_i(365)
        self.ins(OP_IMUL, doe, yoe, k365)
        self.free(k365)
        var q = self.tmp()
        var k4 = self.const_i(4)
        self.ins(OP_IDIV, q, yoe, k4)
        self.free(k4)
        self.ins(OP_IADD, doe, doe, q)
        var k100 = self.const_i(100)
        self.ins(OP_IDIV, q, yoe, k100)
        self.free(k100)
        self.ins(OP_ISUB, doe, doe, q)
        self.ins(OP_IADD, doe, doe, doy)
        self.free(q)
        self.free(yoe)
        self.free(doy)
        var k146 = self.const_i(146097)
        self.ins(OP_IMUL, era, era, k146)
        self.free(k146)
        self.ins(OP_IADD, era, era, doe)
        self.free(doe)
        var k719 = self.const_i(719468)
        self.ins(OP_ISUB, era, era, k719)
        self.free(k719)
        return era

    def date_operand_value(mut self, node: Int) raises -> V:
        """A DATE operand's day numbers, a NULL row's garbage replaced by 0 so the arithmetic after it reads numbers."""
        var v = self.value(node)
        if v.nul >= 0:
            var z = self.const_i(0)
            self.ins(OP_SELECT, v.reg, v.nul, z, v.reg)
            self.free(z)
        return v^

    def day_leaf(self, e: Int, mut col: Int, mut leaf: Int) -> Bool:
        """Whether expression e is a function of the days of one DATE column and constants (date functions, + and - of days): then its answer can be made once per
        day of the column's range. `col` is the column, `leaf` the node that reads it."""
        var x = self.rx[e].copy()
        if x.op == B_COL:
            if x.kind != DATE.value:
                return False
            if col < 0:
                col = Int(x.value)
                leaf = e
            return col == Int(x.value)
        if x.op == B_LIT:
            return x.kind == INT.value or x.kind == DATE.value
        if x.op == B_DATEPART or x.op == B_DATETRUNC or x.op == B_NEG:
            return self.day_leaf(x.a, col, leaf)
        if x.op == B_ADDMONTHS or x.op == B_DATEDIFF or x.op == B_ADD or x.op == B_SUB:
            return self.day_leaf(x.a, col, leaf) and self.day_leaf(x.b, col, leaf)
        return False

    def day_eval(self, e: Int, day: Int) -> Int64:
        """The value of a day_leaf expression for one day: the same functions the constants are folded with (`sqlfn.mojo`)."""
        var x = self.rx[e].copy()
        if x.op == B_COL:
            return Int64(day)
        if x.op == B_LIT:
            return x.value
        if x.op == B_NEG:
            return -self.day_eval(x.a, day)
        if x.op == B_DATEPART:
            var v = date_part(Int(self.day_eval(x.a, day)), Int(x.value))
            return Int64(Float64(v).to_bits()) if x.kind == FLOAT.value else Int64(v)
        if x.op == B_DATETRUNC:
            return Int64(date_trunc_days(Int(self.day_eval(x.a, day)), Int(x.value)))
        if x.op == B_ADDMONTHS:
            return Int64(add_months(Int(self.day_eval(x.a, day)), Int(self.day_eval(x.b, day))))
        if x.op == B_DATEDIFF:
            return Int64(date_diff_days(Int(self.day_eval(x.a, day)), Int(self.day_eval(x.b, day)), Int(x.value)))
        if x.op == B_ADD:
            return self.day_eval(x.a, day) + self.day_eval(x.b, day)
        return self.day_eval(x.a, day) - self.day_eval(x.b, day)

    def day_table(mut self, e: Int) raises -> Optional[V]:
        """A date function of one DATE column, answered once per day of the column's range (a table the kernel looks the day up in) instead of in civil-date
        arithmetic on every row, when the range is short against the rows (the table costs about what 20 rows do). A footer's range is a hint: a row outside
        it is the footer lying (the query then runs again without statistics)."""
        var col = -1
        var leaf = -1
        if not self.day_leaf(e, col, leaf) or col < 0:
            return None
        var r = self.dicts.range_of(col)
        if not r[0]:
            return None
        var span = Int(r[2]) - Int(r[1]) + 1
        if span <= 0 or span > CAST_TEXT_LIMIT or span * 20 > self.dicts.rows:
            return None
        var lo = r[1]
        var ents = List[Int64](length=span, fill=0)
        for k in range(span):
            ents[k] = self.day_eval(e, Int(lo) + k)
        var v = self.value(leaf)
        var lc = self.const_i(lo)
        self.ins(OP_ISUB, v.reg, v.reg, lc)
        self.free(lc)
        if not self.dicts.range_exact:
            var bad = self.out_of_range(v.reg, 0, Int64(span - 1))
            self.flag_errors(bad, v.nul, FOOTER_LIE + "a date outside the range the footer gave")
            self.free(bad)
        self.tab(v.reg, ents)
        self.p.day_tabs += 1
        self.p.day_days += span
        if self.rx[e].kind == FLOAT.value:
            self.p.has_float = True
        return V(v.reg, v.nul, False, self.rx[e].kind == FLOAT.value)

    def datepart_value(mut self, x: BExpr) raises -> V:
        var v = self.date_operand_value(x.a)
        var part = Int(x.value)
        if part == DP_DOW or part == DP_ISODOW:
            var k = self.const_i(Int64(4 if part == DP_DOW else 3))
            self.ins(OP_IADD, v.reg, v.reg, k)
            self.free(k)
            var seven = self.const_i(7)
            self.ins(OP_IMOD, v.reg, v.reg, seven)
            self.free(seven)
            if part == DP_ISODOW:
                var one = self.const_i(1)
                self.ins(OP_IADD, v.reg, v.reg, one)
                self.free(one)
            return v^
        if part == DP_EPOCH:                                         # seconds, as the FLOAT DuckDB answers
            var k = self.const_i(86400)
            self.ins(OP_IMUL, v.reg, v.reg, k)
            self.free(k)
            self.p.has_float = True
            self.ins(OP_I2F, v.reg, v.reg)
            return V(v.reg, v.nul, False, True)
        if part == DP_WEEK or part == DP_ISOYEAR:                    # ISO week and year: from the Thursday of the date's week, th = z - (z + 3) mod 7 + 3
            var th = self.tmp()
            var k3 = self.const_i(3)
            self.ins(OP_IADD, th, v.reg, k3)
            var k7 = self.const_i(7)
            self.ins(OP_IMOD, th, th, k7)
            self.ins(OP_ISUB, th, v.reg, th)
            self.ins(OP_IADD, th, th, k3)
            self.free(k3)
            var ct = self.civil(th)
            var wk = self.tmp()
            if part == DP_ISOYEAR:
                self.p.emit(OP_IMAX, wk, ct[0], ct[0])
            else:
                var one = self.const_i(1)
                var jan = self.days_from_civil_regs(ct[0], one, one)
                self.ins(OP_ISUB, wk, th, jan)
                self.ins(OP_IDIV, wk, wk, k7)
                self.ins(OP_IADD, wk, wk, one)
                self.free(jan)
                self.free(one)
            self.free(k7)
            self.free(ct[0])
            self.free(ct[1])
            self.free(ct[2])
            self.free(th)
            self.free(v.reg)
            return V(wk, v.nul, False)
        var c = self.civil(v.reg)
        var y = c[0]
        var m = c[1]
        var d = c[2]
        var out = self.tmp()
        if part == DP_YEAR:
            self.p.emit(OP_IMAX, out, y, y)
        elif part == DP_MONTH:
            self.p.emit(OP_IMAX, out, m, m)
        elif part == DP_DAY:
            self.p.emit(OP_IMAX, out, d, d)
        elif part == DP_QUARTER:
            var one = self.const_i(1)
            self.ins(OP_ISUB, out, m, one)
            var three = self.const_i(3)
            self.ins(OP_IDIV, out, out, three)
            self.ins(OP_IADD, out, out, one)
            self.free(one)
            self.free(three)
        elif part == DP_DOY:
            var one = self.const_i(1)
            var jan = self.days_from_civil_regs(y, one, one)
            self.ins(OP_ISUB, out, v.reg, jan)
            self.ins(OP_IADD, out, out, one)
            self.free(jan)
            self.free(one)
        elif part == DP_DECADE:
            var ten = self.const_i(10)
            self.ins(OP_IDIV, out, y, ten)
            self.free(ten)
        elif part == DP_CENTURY or part == DP_MILLENNIUM:
            # (y + 99) // 100 for a positive year, y // 100 below (the years 1 to 9999 in practice)
            var k = self.const_i(Int64(100 if part == DP_CENTURY else 1000))
            var off = self.const_i(Int64(99 if part == DP_CENTURY else 999))
            self.ins(OP_IADD, out, y, off)
            self.ins(OP_IDIV, out, out, k)
            self.free(k)
            self.free(off)
        self.free(y)
        self.free(m)
        self.free(d)
        self.free(v.reg)
        return V(out, v.nul, False)

    def datetrunc_value(mut self, x: BExpr) raises -> V:
        var v = self.date_operand_value(x.a)
        var part = Int(x.value)
        if part == DP_WEEK:                                       # back to Monday: z - (z + 3) mod 7
            var t = self.tmp()
            var three = self.const_i(3)
            self.ins(OP_IADD, t, v.reg, three)
            self.free(three)
            var seven = self.const_i(7)
            self.ins(OP_IMOD, t, t, seven)
            self.free(seven)
            self.ins(OP_ISUB, v.reg, v.reg, t)
            self.free(t)
            return v^
        var c = self.civil(v.reg)
        var y = c[0]
        var m = c[1]
        var d = c[2]
        self.free(v.reg)
        var one = self.const_i(1)
        var ty = y
        var tm = one
        var td = one
        var own_y = False
        var own_m = False
        if part == DP_MONTH:
            tm = m
        elif part == DP_QUARTER:
            tm = self.tmp()
            own_m = True
            self.ins(OP_ISUB, tm, m, one)
            var three = self.const_i(3)
            self.ins(OP_IDIV, tm, tm, three)
            self.ins(OP_IMUL, tm, tm, three)
            self.free(three)
            self.ins(OP_IADD, tm, tm, one)
        elif part == DP_DECADE or part == DP_CENTURY or part == DP_MILLENNIUM:
            ty = self.tmp()
            own_y = True
            if part == DP_DECADE:
                var ten = self.const_i(10)
                self.ins(OP_IDIV, ty, y, ten)
                self.ins(OP_IMUL, ty, ty, ten)
                self.free(ten)
            else:                                                  # DuckDB: the year divided by 100 (1000) and floored, not the calendar century
                var k = self.const_i(Int64(100 if part == DP_CENTURY else 1000))
                self.ins(OP_IDIV, ty, y, k)
                self.ins(OP_IMUL, ty, ty, k)
                self.free(k)
        var res = self.days_from_civil_regs(ty, tm, td)
        if own_y:
            self.free(ty)
        if own_m:
            self.free(tm)
        self.free(one)
        self.free(y)
        self.free(m)
        self.free(d)
        return V(res, v.nul, False)

    def addmonths_value(mut self, x: BExpr) raises -> V:
        var v = self.date_operand_value(x.a)
        var n = self.value(x.b)
        if n.nul >= 0:
            var z = self.const_i(0)
            self.ins(OP_SELECT, n.reg, n.nul, z, n.reg)
            self.free(z)
        var c = self.civil(v.reg)
        var y = c[0]
        var m = c[1]
        var d = c[2]
        self.free(v.reg)
        var total = self.tmp()                                    # y * 12 + (m - 1) + n
        var k12 = self.const_i(12)
        self.ins(OP_IMUL, total, y, k12)
        self.ins(OP_IADD, total, total, m)
        var one = self.const_i(1)
        self.ins(OP_ISUB, total, total, one)
        self.ins(OP_IADD, total, total, n.reg)
        self.free(n.reg)
        var y2 = self.tmp()
        self.ins(OP_IDIV, y2, total, k12)
        var m2 = self.tmp()
        self.ins(OP_IMUL, m2, y2, k12)
        self.ins(OP_ISUB, m2, total, m2)
        self.ins(OP_IADD, m2, m2, one)
        self.free(total)
        self.free(k12)
        # days in the target month: a table of 12, plus 1 for February of a leap year
        var base = List[Int64]()
        for dm in [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]:
            base.append(Int64(dm))
        var dim = self.tmp()
        self.p.emit(OP_IMAX, dim, m2, m2)
        self.tab(dim, base, 1)
        var two = self.const_i(2)
        var feb = self.tmp()
        self.ins(OP_ICMP, feb, m2, two, C_EQ)
        self.free(two)
        var k4 = self.const_i(4)
        var r4 = self.tmp()
        self.ins(OP_IMOD, r4, y2, k4)
        self.free(k4)
        var zero = self.const_i(0)
        var leap = self.tmp()
        self.ins(OP_ICMP, leap, r4, zero, C_EQ)
        var k100 = self.const_i(100)
        self.ins(OP_IMOD, r4, y2, k100)
        self.free(k100)
        var c100 = self.tmp()
        self.ins(OP_ICMP, c100, r4, zero, C_NE)
        var k400 = self.const_i(400)
        self.ins(OP_IMOD, r4, y2, k400)
        self.free(k400)
        var c400 = self.tmp()
        self.ins(OP_ICMP, c400, r4, zero, C_EQ)
        self.ins(OP_OR, c100, c100, c400)
        self.ins(OP_AND, leap, leap, c100)
        self.ins(OP_AND, leap, leap, feb)
        self.ins(OP_F2I, leap, leap)
        self.ins(OP_IADD, dim, dim, leap)
        self.free(feb)
        self.free(r4)
        self.free(zero)
        self.free(leap)
        self.free(c100)
        self.free(c400)
        self.ins(OP_IMIN, d, d, dim)
        self.free(dim)
        var res = self.days_from_civil_regs(y2, m2, d)
        self.free(y)
        self.free(m)
        self.free(d)
        self.free(y2)
        self.free(m2)
        self.free(one)
        return V(res, self.or_nul(v.nul, n.nul), False)

    def total_months(mut self, z: Int) raises -> Int:
        """y * 12 + (m - 1) of the day-number register z (a new register)."""
        var c = self.civil(z)
        var t = self.tmp()
        var k12 = self.const_i(12)
        self.ins(OP_IMUL, t, c[0], k12)
        self.free(k12)
        self.ins(OP_IADD, t, t, c[1])
        var one = self.const_i(1)
        self.ins(OP_ISUB, t, t, one)
        self.free(one)
        self.free(c[0])
        self.free(c[1])
        self.free(c[2])
        return t

    def datediff_value(mut self, x: BExpr) raises -> V:
        var a = self.date_operand_value(x.a)
        var b = self.date_operand_value(x.b)
        var part = Int(x.value)
        var res = self.tmp()
        if part == DP_DAY or part == DP_WEEK:
            self.ins(OP_ISUB, res, b.reg, a.reg)
            if part == DP_WEEK:                                   # whole weeks, truncated toward zero
                var neg = self.tmp()
                var zero = self.const_i(0)
                self.ins(OP_ICMP, neg, res, zero, C_LT)
                var ab = self.tmp()
                self.ins(OP_ISUB, ab, zero, res)
                self.ins(OP_SELECT, ab, neg, ab, res)
                var seven = self.const_i(7)
                self.ins(OP_IDIV, ab, ab, seven)
                self.ins(OP_ISUB, res, zero, ab)
                self.ins(OP_SELECT, res, neg, res, ab)
                self.free(neg)
                self.free(zero)
                self.free(ab)
                self.free(seven)
        else:
            var ma = self.total_months(a.reg)
            var mb = self.total_months(b.reg)
            if part == DP_QUARTER or part == DP_YEAR or part == DP_DECADE or part == DP_CENTURY or part == DP_MILLENNIUM:
                var k = 3 if part == DP_QUARTER else 12 if part == DP_YEAR else 120 if part == DP_DECADE else 1200 if part == DP_CENTURY else 12000
                var kc = self.const_i(Int64(k))
                self.ins(OP_IDIV, ma, ma, kc)
                self.ins(OP_IDIV, mb, mb, kc)
                self.free(kc)
            self.ins(OP_ISUB, res, mb, ma)
            self.free(ma)
            self.free(mb)
        self.free(a.reg)
        self.free(b.reg)
        return V(res, self.or_nul(a.nul, b.nul), False)

    def divmod(mut self, x: BExpr) raises -> V:
        """a // b or a % b truncating toward zero (DuckDB's), from the kernel's floor division: the quotient is one more
        than the floored one when the remainder is not zero and the operands' signs differ. b = 0 gives NULL."""
        var a = self.value(x.a)
        var b = self.value(x.b)
        var zero = self.const_i(0)
        var q = self.tmp()
        self.ins(OP_IDIV, q, a.reg, b.reg)
        var t = self.tmp()
        self.ins(OP_IMUL, t, q, b.reg)
        self.ins(OP_ISUB, t, a.reg, t)                            # t: the floored remainder
        var rz = self.tmp()
        self.ins(OP_ICMP, rz, t, zero, C_NE)
        var na = self.tmp()
        self.ins(OP_ICMP, na, a.reg, zero, C_LT)
        var nb = self.tmp()
        self.ins(OP_ICMP, nb, b.reg, zero, C_LT)
        self.ins(C_NE, na, na, nb)                                # float compare: the signs differ
        self.ins(OP_AND, rz, rz, na)
        self.ins(OP_F2I, rz, rz)
        self.ins(OP_IADD, q, q, rz)                               # the truncated quotient
        var res = q
        if x.op == B_IMOD:
            self.ins(OP_IMUL, t, q, b.reg)
            self.ins(OP_ISUB, q, a.reg, t)
        var dz = self.tmp()
        self.ins(OP_ICMP, dz, b.reg, zero, C_EQ)
        var nul = self.or_nul(a.nul, b.nul)
        nul = self.or_nul(nul, dz)
        self.free(a.reg)
        self.free(b.reg)
        self.free(zero)
        self.free(t)
        self.free(rz)
        self.free(na)
        self.free(nb)
        return V(res, nul, False)

    # -- booleans (Kleene: 1.0 true, 0.0 false, 0.5 unknown)
    def with_unknown(mut self, r: Int, nul: Int) raises -> V:
        """r (a definite boolean) where nul is 0, unknown where it is 1."""
        if nul < 0:
            return V(r, -1, False)
        var h = self.const_f(0.5)
        self.ins(OP_SELECT, r, nul, h, r)
        self.free(h)
        self.free(nul)
        return V(r, -1, True)

    def boolean(mut self, e: Int) raises -> V:
        var x = self.rx[e].copy()
        if x.op == B_CMP:
            return self.compare(x)
        if x.op == B_AND or x.op == B_OR:
            var a = self.boolean(x.a)
            var decided = self.tmp()                                  # where the left side leaves the right one to matter: AND: not false, OR: not true
            var kconst = self.const_f(0.0 if x.op == B_AND else 1.0)
            self.ins(25, decided, a.reg, kconst, 0)
            self.free(kconst)
            var saved_guard = self.guard
            self.guard = self.guard_with(decided)
            var b = self.boolean(x.b)
            self.free(self.guard)
            self.guard = saved_guard
            self.free(decided)
            if not a.unk and not b.unk:
                self.ins(OP_AND if x.op == B_AND else OP_OR, a.reg, a.reg, b.reg)
            else:                                                 # the minimum / the maximum
                var t = self.tmp()
                self.ins(C_LT if x.op == B_AND else C_GT, t, a.reg, b.reg)
                self.ins(OP_SELECT, a.reg, t, a.reg, b.reg)
                self.free(t)
            self.free(b.reg)
            return V(a.reg, -1, a.unk or b.unk)
        if x.op == B_NOT:
            var a = self.boolean(x.a)
            if not a.unk:
                self.ins(OP_NOT, a.reg, a.reg)
            else:
                var one = self.const_f(1.0)
                self.ins(OP_SUBF, a.reg, one, a.reg)
                self.free(one)
            return a^
        if x.op == B_ISNULL:
            var inner = self.rx[x.a].copy()
            if inner.kind == BOOL.value and inner.op != B_COL and inner.op != B_LIT:       # a boolean expression: unknown is its NULL
                var a = self.boolean(x.a)
                var h = self.const_f(0.5)
                self.ins(C_EQ, a.reg, a.reg, h)
                self.free(h)
                return V(a.reg, -1, False)
            var v = self.value(x.a)
            self.free(v.reg)
            if v.nul >= 0:
                return V(v.nul, -1, False)
            return V(self.const_f(0.0), -1, False)
        if x.op == B_IN:
            return self.in_list(x)
        if x.op == B_TFN:
            return self.tfn_bool(e)
        if x.op == B_CASE:
            return self.case_generic(e, True)
        if x.op == B_COALESCE:
            return self.coalesce_generic(e, True)
        if x.op == B_DISTINCT:
            return self.distinct_bool(x)
        if x.op == B_BOOLIS:
            var a = self.boolean(x.a)
            var code = Int(x.value)
            var k = self.const_f(1.0 if code < 2 else 0.0)
            self.ins(24 if code % 2 == 0 else 25, a.reg, a.reg, k)
            self.free(k)
            return V(a.reg, -1, False)
        if x.op == B_CAST:
            return self.cast_bool(e)
        if x.op == B_NULL:
            return V(self.const_f(0.5), -1, True)
        if x.op == B_LIT:
            return V(self.const_f(1.0 if x.value != 0 else 0.0), -1, False)
        if x.op == B_COL:                                          # a BOOL column used as a condition
            var v = self.value(e)
            var r = self.tmp()
            self.ins(OP_I2F, r, v.reg)
            self.free(v.reg)
            return self.with_unknown(r, v.nul)
        raise Error("internal: expression kind " + String(x.op) + " is not a condition")

    def compare(mut self, x: BExpr) raises -> V:
        var t = self.cmp_raw(x.a, x.b, Int(x.value))
        return self.with_unknown(t[0], self.or_nul(t[1], t[2]))

    def cast_bool(mut self, e: Int) raises -> V:
        """CAST(x AS BOOLEAN): true where the number is not zero; a word is read once per dictionary word."""
        var x = self.rx[e].copy()
        var src = self.rx[x.a].copy()
        var r = self.tmp()
        if src.kind == TEXT.value:
            var v = self.cast_from_text(e, x)
            self.ins(OP_I2F, r, v.reg)
            self.free(v.reg)
            return self.with_unknown(r, v.nul)
        var v = self.value(x.a)
        var zero = self.const_i(0) if not v.fl else self.const_f(0.0)
        if v.fl:
            self.ins(25, r, v.reg, zero, 0)                    # != (IEEE: NaN is not zero)
        else:
            self.ins(OP_ICMP, r, v.reg, zero, C_NE)
        self.free(zero)
        self.free(v.reg)
        return self.with_unknown(r, v.nul)

    def add_set(mut self, bytes: List[UInt8], lo: Int) -> Int:
        var off = len(self.set_bytes)
        self.set_bytes.extend(bytes.copy())
        self.set_meta.append(Int64(off))
        self.set_meta.append(Int64(len(bytes)))
        self.set_meta.append(Int64(lo))
        return len(self.set_meta) // 3 - 1

    def int_set(mut self, r: Int, v: V, items: List[Int]) raises:
        """r = 1.0 where the value is one of the constants, by a hash set of int64 keys (a float's bits, with -0.0 as 0.0 and one NaN)."""
        var keys = List[Int64]()
        for it in items:
            var k = self.rx[it].value
            if v.fl:
                var f = Float64(from_bits=UInt64(k))
                if f != f:
                    k = Int64(0x7FF8000000000000)
                elif f == 0.0:
                    k = 0
            keys.append(k)
        var cap = 16
        while cap < 2 * len(keys):
            cap *= 2
        var lg = 0
        while (1 << lg) < cap:
            lg += 1
        var table = List[Int64](length=cap, fill=Int64.MIN)
        var has_min = Int64(0)
        for k in keys:
            if k == Int64.MIN:
                has_min = 1
                continue
            var h = Int((UInt64(k) * UInt64(0x9E3779B97F4A7C15)) >> UInt64(64 - lg))
            while table[h] != Int64.MIN and table[h] != k:
                h = (h + 1) & (cap - 1)
            table[h] = k
        var off = (len(self.set_bytes) + 7) // 8 * 8
        self.set_bytes.resize(off + 8 * cap, 0)
        var base = LP(unsafe_from_address=Int(self.set_bytes.unsafe_ptr()) + off)
        for i in range(cap):
            base[i] = table[i]
        self.set_meta.append(Int64(off))
        self.set_meta.append(Int64(cap))
        self.set_meta.append(has_min)
        var at = len(self.set_meta) // 3 - 1
        var probe = v.reg
        if v.fl:                                                   # -0.0 + 0.0 is +0.0; every NaN is the one NaN
            probe = self.tmp()
            var zero = self.const_f(0.0)
            self.ins(OP_ADD, probe, v.reg, zero)
            self.free(zero)
            var nanf = self.tmp()
            self.ins(25, nanf, probe, probe, 0)
            var canon = self.const_f(Float64(from_bits=UInt64(0x7FF8000000000000)))
            self.ins(OP_SELECT, probe, nanf, canon, probe)
            self.free(canon)
            self.free(nanf)
        self.ins(OP_ISET, r, probe, at)
        if v.fl:
            self.free(probe)
        self.free(v.reg)

    def in_list(mut self, x: BExpr) raises -> V:
        var lhs = self.rx[x.a].copy()
        if lhs.kind == TEXT.value:
            self.prep_dict(x.a)
        var v = self.value(x.a)
        var r: Int
        if lhs.kind == TEXT.value:
            var c = lhs.src
            var n = self.dicts.count(c)
            var bytes = List[UInt8](length=max(n, 1), fill=0)
            for it in x.items:
                var code = self.dicts.code_of(c, self.rx[it].text)
                if code >= 0:
                    bytes[code] = 1
            r = self.tmp()
            self.ins(OP_I2F, r, v.reg)
            self.free(v.reg)
            var set_at = self.add_set(bytes, 0)
            self.ins(OP_IN_SET, r, r, set_at)
        else:
            r = self.tmp()
            if len(x.items) > 8:                                   # a long list: the kernel's hash set
                self.int_set(r, v, x.items)
            else:
                if len(x.items) == 0:
                    self.ins(OP_LOAD_CONST, r, self.p.const(Int64(0)))
                for i in range(len(x.items)):
                    var k = self.const_i(self.rx[x.items[i]].value)
                    if i == 0:
                        if v.fl:
                            self.ins(C_EQ, r, v.reg, k, NAN_AWARE)
                        else:
                            self.ins(OP_ICMP, r, v.reg, k, C_EQ)
                    else:
                        var t = self.tmp()
                        if v.fl:
                            self.ins(C_EQ, t, v.reg, k, NAN_AWARE)
                        else:
                            self.ins(OP_ICMP, t, v.reg, k, C_EQ)
                        self.ins(OP_OR, r, r, t)
                        self.free(t)
                    self.free(k)
                self.free(v.reg)
        if x.value != 0:                                           # the list holds a NULL: a miss is unknown, not false
            var one = self.const_f(1.0)
            var h = self.const_f(0.5)
            self.ins(OP_SELECT, r, r, one, h)
            self.free(one)
            self.free(h)
            if v.nul >= 0:
                var hh = self.const_f(0.5)
                self.ins(OP_SELECT, r, v.nul, hh, r)
                self.free(hh)
                self.free(v.nul)
            return V(r, -1, True)
        return self.with_unknown(r, v.nul)


comptime PHASE_ALL = 0       # the plain aggregates and the holistic ones in one program
comptime PHASE_PLAIN = 1     # hashed groups, pass 1: the plain aggregates, the key words out
comptime PHASE_HOL = 2       # hashed groups, pass 2: only the holistic sources, each row mapped to its group by the hook


def lower_sql(q: BoundQuery, plan: PhysicalPlan, var nullable: List[Bool], var pos: List[Int], nchunk_cols: Int,
              var dicts: DictAccess, var aliases: List[List[Int]], masked: Bool, phase: Int = PHASE_ALL) raises -> SqlProgram:
    """The program for `q` under `plan`. `nullable[c]`: source column c may hold NULLs; `pos[c]`: its place in the chunk
    tables the program runs on; `aliases[c]`: a frame's merged TEXT codes; `masked`: a frame mask hides changed rows."""
    var e = _Emitter(Program(nchunk_cols), q.rexprs.copy(), nullable^, pos^, aliases^, dicts^)
    var sp_aggs = q.aggs.copy()
    # which aggregates are holistic: their values go to the kernel hook, one source per distinct argument
    var holistic = False
    for a in sp_aggs:
        if a.hol:
            holistic = True
    if phase == PHASE_PLAIN:
        holistic = False
    var nkw = 1 if plan.group != G_HASHED else plan.key_words()
    var key_regs = nkw
    var key_base = 0                                              # the first register of the key
    var ctl_reg = 1                                               # the register holding the control block of a holistic query
    if phase == PHASE_HOL:                                        # hashed, pass 2: the control block first, then the key words
        key_base = 1
        ctl_reg = 0
    var out = SqlProgram(Program(0))
    e.p.mat = holistic
    var guarded = False                                           # a dense key whose range a file's footer stated: one extra slot catches rows outside it
    if plan.group == G_DENSE:
        for lay in plan.keys:
            guarded = guarded or lay.guarded
    e.p.groups = (plan.domain + (1 if guarded else 0)) if plan.group == G_DENSE else 1
    out.guard_slot = plan.domain if guarded else -1
    # registers: the key (dense: register 0, and register 1 the control block of a holistic query) or the key words
    e.reserve(key_regs + (1 if holistic else 0))
    e.p.section = 1
    if holistic:
        e.p.ctl_const = len(e.p.consts)
        e.p.consts.append(CTL_PLACEHOLDER)
        var cr = e.tmp()
        e.ins(OP_LOAD_CONST, cr, e.p.ctl_const)
        e.ins(OP_IMAX, ctl_reg, cr, cr)
        e.free(cr)
    # the predicate: the WHERE (unknown is not true), then the frame mask
    e.p.section = 0
    var pred = -1
    var mask = -1
    if masked:                                                    # a changed row of a frame is masked out: an error flag must not fire on its stale values
        mask = e.tmp()
        e.ins(OP_ISNULL, mask, _frame_mask(e.p))
        e.ins(OP_NOT, mask, mask)
        e.guard = mask
    if q.where >= 0:
        var w = e.boolean(q.where)
        if w.unk:
            var one = e.const_f(1.0)
            e.ins(24, w.reg, w.reg, one)                         # == 1.0: unknown and false both drop the row
            e.free(one)
        pred = w.reg
    e.guard = -1
    if masked:
        if pred >= 0:
            e.ins(OP_AND, pred, pred, mask)
            e.free(mask)
        else:
            pred = mask
    if pred >= 0:
        var k = e.keep(pred)
        e.free(pred)
        pred = k
    e.p.pred_reg = pred
    # the group key
    e.p.section = 1
    if plan.group == G_DENSE:
        e.p.keyed = True
        var acc = -1
        var anybad = -1
        for k in range(len(plan.keys)):
            ref lay = plan.keys[k]
            var v = e.value(lay.expr)
            if lay.nullable and v.nul < 0:                           # the planner says the key may be NULL (it is cautious about expressions): this one never is
                v = V(v.reg, e.const_f(0.0), v.unk, v.fl)
            if lay.kind == TEXT.value or lay.kind == BOOL.value:
                if lay.nullable:
                    var one = e.const_i(1)
                    e.ins(OP_IADD, v.reg, v.reg, one)
                    e.free(one)
            else:
                var lo = e.const_i(lay.lo)
                e.ins(OP_ISUB, v.reg, v.reg, lo)
                e.free(lo)
                if lay.guarded:                                   # a value below the range or past it is flagged (NULL is not a value)
                    var zero = e.const_i(0)
                    var bad = e.tmp()
                    e.ins(OP_ICMP, bad, v.reg, zero, C_LT)
                    e.free(zero)
                    var span = e.const_i(Int64(lay.radix - (1 if lay.nullable else 0)))
                    var past = e.tmp()
                    e.ins(OP_ICMP, past, v.reg, span, C_GE)
                    e.free(span)
                    e.ins(OP_OR, bad, bad, past)
                    e.free(past)
                    if lay.nullable:
                        var f0 = e.const_f(0.0)
                        e.ins(OP_SELECT, bad, v.nul, f0, bad)
                        e.free(f0)
                    if anybad < 0:
                        anybad = bad
                    else:
                        e.ins(OP_OR, anybad, anybad, bad)
                        e.free(bad)
                if lay.nullable:
                    var one = e.const_i(1)
                    e.ins(OP_IADD, v.reg, v.reg, one)
                    e.free(one)
            if lay.nullable:
                var z = e.const_i(0)
                e.ins(OP_SELECT, v.reg, v.nul, z, v.reg)         # NULL is slot 0
                e.free(z)
                e.free(v.nul)
            if acc < 0:
                acc = v.reg
            else:
                var rd = e.const_i(Int64(lay.radix))
                e.ins(OP_IMUL, acc, acc, rd)
                e.ins(OP_IADD, acc, acc, v.reg)
                e.free(rd)
                e.free(v.reg)
        if anybad >= 0:
            var dom = e.const_i(Int64(plan.domain))
            e.ins(OP_SELECT, acc, anybad, dom, acc)
            e.free(dom)
            e.free(anybad)
        e.ins(OP_IMAX, 0, acc, acc)
        e.free(acc)
    elif plan.group == G_HASHED:
        e.p.keyed = True
        var at = key_base
        for k in range(len(plan.keys)):
            ref lay = plan.keys[k]
            var v = e.value(lay.expr)
            if lay.nullable and v.nul < 0:                           # the planner says the key may be NULL (it is cautious about expressions): this one never is
                v = V(v.reg, e.const_f(0.0), v.unk, v.fl)
            if v.fl:                                                  # -0.0 is 0.0 and every NaN is one value (F-077): one key word per value
                var zero = e.const_f(0.0)
                e.ins(OP_ADD, v.reg, v.reg, zero)                     # -0.0 + 0.0 is +0.0
                e.free(zero)
                var nanf = e.tmp()
                e.ins(25, nanf, v.reg, v.reg, 0)
                var canon = e.const_f(Float64(from_bits=UInt64(0x7FF8000000000000)))
                e.ins(OP_SELECT, v.reg, nanf, canon, v.reg)
                e.free(canon)
                e.free(nanf)
            if lay.nullable:
                var z = e.const_i(0)
                var ni = e.tmp()
                e.ins(OP_F2I, ni, v.nul)
                e.ins(OP_SELECT, v.reg, v.nul, z, v.reg)         # NULL's value word is 0, its flag word 1
                e.free(z)
                e.free(v.nul)
                e.ins(OP_IMAX, at, v.reg, v.reg)
                e.ins(OP_IMAX, at + 1, ni, ni)
                e.free(ni)
                at += 2
            else:
                e.ins(OP_IMAX, at, v.reg, v.reg)
                at += 1
            e.free(v.reg)
    elif holistic:
        e.p.keyed = True
        e.ins(OP_LOAD_CONST, 0, e.p.const(0))
    # the aggregates, one pass per distinct argument
    var n_aggs = len(sp_aggs)
    for _ in range(n_aggs):
        out.agg_val_slot.append(-1)
        out.agg_nul_slot.append(-1)
        out.agg_nan_slot.append(-1)
        out.agg_src.append(-1)
    var done = List[Bool](length=n_aggs, fill=False)
    var n_slots = 0
    var n_src = 0
    var src_vreg = List[Int]()
    var src_valid = List[Int]()
    var src_float = List[Bool]()
    for i in range(n_aggs):
        if (phase == PHASE_PLAIN and sp_aggs[i].hol) or (phase == PHASE_HOL and not sp_aggs[i].hol):
            done[i] = True                                        # the other pass computes it
        if done[i] or sp_aggs[i].arg < 0:
            continue
        var key = bkey(e.rx, sp_aggs[i].arg)
        var members = List[Int]()
        for j in range(i, n_aggs):
            var other_pass = (phase == PHASE_PLAIN and sp_aggs[j].hol) or (phase == PHASE_HOL and not sp_aggs[j].hol)
            if not done[j] and not other_pass and sp_aggs[j].arg >= 0 and bkey(e.rx, sp_aggs[j].arg) == key:
                members.append(j)
                done[j] = True
        var need_sum = False
        var need_min = False
        var need_max = False
        var need_hol = False
        for j in members:
            var f = sp_aggs[j].func
            var h = sp_aggs[j].hol
            need_sum = need_sum or ((f == F_SUM or f == F_AVG) and not h)
            need_min = need_min or (f == F_MIN and not h)
            need_max = need_max or (f == F_MAX and not h)
            need_hol = need_hol or h
        var v: V
        if not (need_sum or need_min or need_max or need_hol) and e.rx[sp_aggs[i].arg].op == B_COL:       # count(col): its NULL flags only
            var c = Int(e.rx[sp_aggs[i].arg].value)
            var nul = -1
            if e.nullable[c]:
                nul = e.tmp()
                e.ins(OP_ISNULL, nul, _null_mask(e.p, e.pos[c]))
            v = V(-1, nul, False)
        else:
            v = e.value(sp_aggs[i].arg)
        var raw_kept = -1
        var sum_slot = -1
        var min_slot = -1
        var max_slot = -1
        var nul_slot = -1
        if need_sum:
            var m = v.reg
            if v.nul >= 0:
                var z = e.const_i(0)
                m = e.tmp()
                e.ins(OP_SELECT, m, v.nul, z, v.reg)
                e.free(z)
            var kept = e.keep(m)
            if v.nul < 0:
                raw_kept = kept
            else:
                e.free(m)
            sum_slot = n_slots
            e.p.section = 2
            e.ins(OP_AGG_SUM if v.fl else OP_AGG_ISUM, sum_slot, kept)
            e.p.section = 1
            out.slot_ops.append(A_FSUM if v.fl else A_SUM)
            if v.fl:
                e.p.repro = True                                      # a float sum has one defined order (F-064)
            n_slots += 1
        var isn = -1                                                  # float min / max: 1.0 where the value is NaN
        if v.fl and (need_min or need_max):
            isn = e.tmp()
            e.ins(25, isn, v.reg, v.reg, 0)                           # != (IEEE): true only for NaN
        if need_min and v.fl:
            var ident = e.const_f(Float64(from_bits=UInt64(0x7FF0000000000000)))
            var m = e.tmp()
            e.ins(OP_SELECT, m, isn, ident, v.reg)                    # a NaN is left out: it is above every number
            if v.nul >= 0:
                e.ins(OP_SELECT, m, v.nul, ident, m)
            var kept = e.keep(m)
            e.free(m)
            e.free(ident)
            min_slot = n_slots
            e.p.section = 2
            e.ins(OP_AGG_MIN, min_slot, kept)
            e.p.section = 1
            out.slot_ops.append(A_FMIN)
            n_slots += 1
        elif need_min:
            var kept: Int
            if v.nul < 0 and raw_kept >= 0:
                kept = raw_kept
            else:
                var m = v.reg
                if v.nul >= 0:
                    var id = e.const_i(Int64.MAX)
                    m = e.tmp()
                    e.ins(OP_SELECT, m, v.nul, id, v.reg)
                    e.free(id)
                kept = e.keep(m)
                if v.nul >= 0:
                    e.free(m)
                else:
                    raw_kept = kept
            min_slot = n_slots
            e.p.section = 2
            e.ins(OP_AGG_IMIN, min_slot, kept)
            e.p.section = 1
            out.slot_ops.append(A_MIN)
            n_slots += 1
        if need_max and v.fl:
            var ident = e.const_f(Float64(from_bits=UInt64(0xFFF0000000000000)))
            var m = e.tmp()
            e.ins(OP_SELECT, m, isn, ident, v.reg)
            if v.nul >= 0:
                e.ins(OP_SELECT, m, v.nul, ident, m)
            var kept = e.keep(m)
            e.free(m)
            e.free(ident)
            max_slot = n_slots
            e.p.section = 2
            e.ins(OP_AGG_MAX, max_slot, kept)
            e.p.section = 1
            out.slot_ops.append(A_FMAX)
            n_slots += 1
        elif need_max:
            var kept: Int
            if v.nul < 0 and raw_kept >= 0:
                kept = raw_kept
            else:
                var m = v.reg
                if v.nul >= 0:
                    var id = e.const_i(Int64.MIN)
                    m = e.tmp()
                    e.ins(OP_SELECT, m, v.nul, id, v.reg)
                    e.free(id)
                kept = e.keep(m)
                if v.nul >= 0:
                    e.free(m)
                else:
                    raw_kept = kept
            max_slot = n_slots
            e.p.section = 2
            e.ins(OP_AGG_IMAX, max_slot, kept)
            e.p.section = 1
            out.slot_ops.append(A_MAX)
            n_slots += 1
        var nan_slot = -1
        if isn >= 0 and phase != PHASE_HOL:                           # the NaNs a float min / max left out (a NaN is a value: MAX is NaN when there is one)
            var nf = isn
            if v.nul >= 0:
                nf = e.tmp()
                var zf = e.const_i(0)
                e.ins(OP_SELECT, nf, v.nul, zf, isn)                  # a NULL row is not a NaN
                e.free(zf)
            var ni2 = e.tmp()
            e.ins(OP_F2I, ni2, nf)
            var kept_nan = e.keep(ni2)
            e.free(ni2)
            if nf != isn:
                e.free(nf)
            nan_slot = n_slots
            e.p.section = 2
            e.ins(OP_AGG_ISUM, nan_slot, kept_nan)
            e.p.section = 1
            out.slot_ops.append(A_SUM)
            n_slots += 1
        e.free(isn)
        if v.nul >= 0 and phase != PHASE_HOL:                     # the NULLs this argument skipped
            var ni = e.tmp()
            e.ins(OP_F2I, ni, v.nul)
            var kept_n = e.keep(ni)
            e.free(ni)
            nul_slot = n_slots
            e.p.section = 2
            e.ins(OP_AGG_ISUM, nul_slot, kept_n)
            e.p.section = 1
            out.slot_ops.append(A_SUM)
            n_slots += 1
        var src = -1
        if need_hol:
            src = n_src
            n_src += 1
            var vr = v.reg
            var kept_v = raw_kept if (v.nul < 0 and raw_kept >= 0) else e.keep(vr)
            var kept_valid = -1
            if v.nul >= 0:
                var nv = e.tmp()
                e.ins(OP_NOT, nv, v.nul)
                e.ins(OP_F2I, nv, nv)
                kept_valid = e.keep(nv)
                e.free(nv)
            src_vreg.append(kept_v)
            src_valid.append(kept_valid)
            src_float.append(v.fl)
        for j in members:
            var f = sp_aggs[j].func
            out.agg_nul_slot[j] = nul_slot
            out.agg_nan_slot[j] = nan_slot
            if sp_aggs[j].hol:
                out.agg_src[j] = src
            elif f == F_SUM or f == F_AVG:
                out.agg_val_slot[j] = sum_slot
            elif f == F_MIN:
                out.agg_val_slot[j] = min_slot
            elif f == F_MAX:
                out.agg_val_slot[j] = max_slot
            elif sp_aggs[j].hol:
                out.agg_src[j] = src
        e.free(v.reg)
        e.free(v.nul)
    if n_slots == 0:                                              # only count(*): one dummy sum keeps the kernel happy
        var z = e.const_i(0)
        var kz = e.keep(z)
        e.free(z)
        e.p.section = 2
        e.ins(OP_AGG_ISUM, 0, kz)
        out.slot_ops.append(A_SUM)
        n_slots = 1
    e.p.n_slots = n_slots
    e.p.n_src = n_src
    e.p.src_vreg = src_vreg^
    e.p.src_valid = src_valid^
    e.p.src_float = src_float^
    out.n_src = n_src
    out.key_regs = key_regs
    out.set_bytes = e.set_bytes.copy()
    out.set_meta = e.set_meta.copy()
    out.instr = (len(e.p.pred) + len(e.p.body) + len(e.p.aggs)) // 5
    out.p = e^.take_program()
    out.p.set_bytes = out.set_bytes.copy()
    out.p.set_meta = out.set_meta.copy()
    return out^


def lower_rows(q: BoundQuery, computed: List[Int], var nullable: List[Bool], var pos: List[Int], nchunk_cols: Int,
               var dicts: DictAccess, var aliases: List[List[Int]], masked: Bool) raises -> SqlProgram:
    """F-152: the program of a row query. It runs the WHERE (and the frame mask) and computes the expressions `computed` (row-level
    expressions of the query) into registers that live to the end, so the executor's block hook (`rows.mojo`) can write them,
    with the selected rows' numbers. Registers 0 and 1 are the group key (always 0: one group) and the hook's control block."""
    var e = _Emitter(Program(nchunk_cols), q.rexprs.copy(), nullable^, pos^, aliases^, dicts^)
    var out = SqlProgram(Program(0))
    e.p.mat = False
    e.p.groups = 1
    e.reserve(2)
    e.p.section = 1
    e.p.ctl_const = len(e.p.consts)
    e.p.consts.append(CTL_PLACEHOLDER)
    var cr = e.tmp()
    e.ins(OP_LOAD_CONST, cr, e.p.ctl_const)
    e.ins(OP_IMAX, 1, cr, cr)
    e.free(cr)
    e.p.section = 0
    var pred = -1
    var mask = -1
    if masked:                                                    # a changed row of a frame is masked out: an error flag must not fire on its stale values
        mask = e.tmp()
        e.ins(OP_ISNULL, mask, _frame_mask(e.p))
        e.ins(OP_NOT, mask, mask)
        e.guard = mask
    if q.where >= 0:
        var w = e.boolean(q.where)
        if w.unk:
            var one = e.const_f(1.0)
            e.ins(24, w.reg, w.reg, one)
            e.free(one)
        pred = w.reg
    e.guard = -1
    if masked:
        if pred >= 0:
            e.ins(OP_AND, pred, pred, mask)
            e.free(mask)
        else:
            pred = mask
    if pred >= 0:
        var k = e.keep(pred)
        e.free(pred)
        pred = k
    e.p.pred_reg = pred
    e.p.section = 1
    e.p.keyed = True
    e.ins(OP_LOAD_CONST, 0, e.p.const(0))
    for ci in computed:
        var x = e.rx[ci].copy()
        var vreg: Int
        var nreg = -1
        if x.kind == BOOL.value and x.op != B_COL and x.op != B_LIT:        # a condition: 1.0 true, 0.0 false, 0.5 unknown (NULL)
            var b = e.boolean(ci)
            var iv = b.reg
            if b.unk:
                var half = e.const_f(0.5)
                var isn = e.tmp()
                e.ins(24, isn, b.reg, half)
                e.free(half)
                var ni = e.tmp()
                e.ins(OP_F2I, ni, isn)
                e.free(isn)
                nreg = e.keep(ni)
                e.free(ni)
                var one = e.const_f(1.0)
                e.ins(24, b.reg, b.reg, one)
                e.free(one)
            var vi = e.tmp()
            e.ins(OP_F2I, vi, b.reg)
            e.free(b.reg)
            vreg = e.keep(vi)
            e.free(vi)
        else:
            var v = e.value(ci)
            vreg = e.keep(v.reg)
            e.free(v.reg)
            if v.nul >= 0:
                var ni = e.tmp()
                e.ins(OP_F2I, ni, v.nul)
                nreg = e.keep(ni)
                e.free(ni)
                e.free(v.nul)
        out.row_vregs.append(vreg)
        out.row_nregs.append(nreg)
    var z = e.const_i(0)                                          # the kernel wants one aggregate: a dummy sum
    var kz = e.keep(z)
    e.free(z)
    e.p.section = 2
    e.ins(OP_AGG_ISUM, 0, kz)
    out.slot_ops.append(A_SUM)
    e.p.n_slots = 1
    e.p.n_src = 0
    out.key_regs = 1
    out.set_bytes = e.set_bytes.copy()
    out.set_meta = e.set_meta.copy()
    out.instr = (len(e.p.pred) + len(e.p.body) + len(e.p.aggs)) // 5
    out.p = e^.take_program()
    out.p.set_bytes = out.set_bytes.copy()
    out.p.set_meta = out.set_meta.copy()
    return out^
