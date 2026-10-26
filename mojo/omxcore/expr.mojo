"""Column expressions (docs/omxcore-mojo-api.md sections 4 and 5).

An `Expr` is a tree kept flat in post-order (operands before their operator), the order a stack machine and the
kernels' register programs both want. Values are exact Int64 (every column stores Int64, see table.mojo), except FLOAT (F-157): a FLOAT column or a float literal is a
float64, an INT or DECIMAL next to it becomes a float64 too (a DECIMAL as its integer divided by 10^scale, as DuckDB casts it), and
`/` always divides doubles (IEEE: x / 0 is inf, -inf or NaN). Comparisons follow DuckDB's order for floats (NaN equals itself and is
above every number, -0.0 equals 0.0). Comparisons and `&` / `|` / `~` give booleans, used by filters.

    col("revenue") - col("cost")            # arithmetic: + - * // %, unary -, least / greatest
    col("price") * 1.1 / col("qty")          # floats: a float literal, `/`
    col("day") // 7                          # a bin
    (col("region") == 2) & col("day").between(0, 364)
"""

comptime MAX_QUANTILE_BINS = 32      # quantile bins (F-177): every edge is one constant of the kernel program, which holds 64

comptime N_COL = 1
comptime N_LIT = 2
comptime N_LITF = 3          # a float64 literal: value = its bits (F-157)
comptime N_ADD = 10
comptime N_SUB = 11
comptime N_MUL = 12
comptime N_FLOORDIV = 13
comptime N_MOD = 14
comptime N_NEG = 15
comptime N_LEAST = 16
comptime N_GREATEST = 17
comptime N_DIV = 18          # `/`: a division of float64s (F-157)
comptime N_LT = 20
comptime N_LE = 21
comptime N_GT = 22
comptime N_GE = 23
comptime N_EQ = 24
comptime N_NE = 25
comptime N_AND = 30
comptime N_OR = 31
comptime N_NOT = 32
comptime N_SCALE_TO = 34     # no-op marking its operand as having `value` decimal places (internal: a formula's rescaled side)
comptime N_BOOL2INT = 33     # a comparison as the number 1 / 0 (internal: merges remap text codes with it)
comptime N_I2F = 35          # an INT or DECIMAL operand read as a float64 (internal): value = the operand's decimal places


@fieldwise_init
struct Node(Copyable, Movable):
    var op: Int
    var value: Int64        # N_LIT: the literal
    var name: String        # N_COL: the column's name

    def arity(self) -> Int:
        if self.op == N_COL or self.op == N_LIT or self.op == N_LITF:
            return 0
        if self.op == N_NEG or self.op == N_NOT or self.op == N_BOOL2INT or self.op == N_SCALE_TO or self.op == N_I2F:
            return 1
        return 2

    def is_bool(self) -> Bool:
        return self.op >= N_LT and self.op <= N_NOT


struct Expr(Copyable, Movable):
    var nodes: List[Node]

    def __init__(out self, var nodes: List[Node]):
        self.nodes = nodes^

    @implicit
    def __init__(out self, value: Int):
        """An integer literal (so `col("a") + 1` works)."""
        self.nodes = [Node(N_LIT, Int64(value), "")]

    @implicit
    def __init__(out self, value: Float64):
        """A float64 literal (so `col("a") * 1.5` works)."""
        self.nodes = [Node(N_LITF, Int64(value.to_bits()), "")]

    def _bin(self, op: Int, other: Expr) -> Expr:
        var nodes = self.nodes.copy()
        for n in other.nodes:
            nodes.append(n.copy())
        nodes.append(Node(op, 0, ""))
        return Expr(nodes^)

    def _un(self, op: Int) -> Expr:
        var nodes = self.nodes.copy()
        nodes.append(Node(op, 0, ""))
        return Expr(nodes^)

    def __add__(self, other: Expr) -> Expr:
        return self._bin(N_ADD, other)

    def __sub__(self, other: Expr) -> Expr:
        return self._bin(N_SUB, other)

    def __mul__(self, other: Expr) -> Expr:
        return self._bin(N_MUL, other)

    def __truediv__(self, other: Expr) -> Expr:
        """Division of float64s (an INT or DECIMAL side is read as a float64); IEEE: x / 0 is inf, -inf or NaN."""
        return self._bin(N_DIV, other)

    def __floordiv__(self, other: Expr) -> Expr:
        """Floor division, as Mojo's Int64 (x // 0 is 0, as in the app's formulas)."""
        return self._bin(N_FLOORDIV, other)

    def __mod__(self, other: Expr) -> Expr:
        """The remainder with the divisor's sign, as Mojo's Int64 (x % 0 is 0)."""
        return self._bin(N_MOD, other)

    def __neg__(self) -> Expr:
        return self._un(N_NEG)

    def __lt__(self, other: Expr) -> Expr:
        return self._bin(N_LT, other)

    def __le__(self, other: Expr) -> Expr:
        return self._bin(N_LE, other)

    def __gt__(self, other: Expr) -> Expr:
        return self._bin(N_GT, other)

    def __ge__(self, other: Expr) -> Expr:
        return self._bin(N_GE, other)

    def __eq__(self, other: Expr) -> Expr:
        return self._bin(N_EQ, other)

    def __ne__(self, other: Expr) -> Expr:
        return self._bin(N_NE, other)

    def __and__(self, other: Expr) -> Expr:
        return self._bin(N_AND, other)

    def __or__(self, other: Expr) -> Expr:
        return self._bin(N_OR, other)

    def __invert__(self) -> Expr:
        return self._un(N_NOT)

    def between(self, lo: Int, hi: Int) -> Expr:
        """lo <= self <= hi (inclusive, the app's range filter)."""
        return (self >= Expr(lo)) & (self <= Expr(hi))

    def is_bool(self) -> Bool:
        return self.nodes[len(self.nodes) - 1].is_bool()

    def columns(self) -> List[String]:
        """The column names it reads, each once, in first-use order."""
        var out = List[String]()
        for n in self.nodes:
            if n.op == N_COL and n.name not in out:
                out.append(n.name)
        return out^


def col(name: String) -> Expr:
    return Expr([Node(N_COL, 0, name)])


def lit(value: Int) -> Expr:
    return Expr(value)


def lit_float(value: Float64) -> Expr:
    """A float64 literal."""
    return Expr(value)


def lit_f32(value: Float64) -> Expr:
    """A literal rounded to a 32-bit float, to compare with a column a Parquet file stored as FLOAT (32 bits, read widened to float64):
    `col("l_discount") <= lit_f32(0.07)` keeps the rows holding 0.07, which `lit_float(0.07)` (the double above that float) would not."""
    return Expr(Float64(Float32(value)))


def least(a: Expr, b: Expr) -> Expr:
    """The smaller of two values per row (a formula's `min`)."""
    return a._bin(N_LEAST, b)


def greatest(a: Expr, b: Expr) -> Expr:
    """The larger of two values per row (a formula's `max`)."""
    return a._bin(N_GREATEST, b)
