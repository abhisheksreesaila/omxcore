"""Names and types for SQL: the Catalog of sources, the typed expression tree and the binder (F-150).

The binder turns a parsed statement (`sql.mojo`) into a `BoundQuery`:
- row-level expressions (WHERE, aggregate arguments, GROUP BY keys) bound against the source's columns, in `rexprs`;
- the aggregates the query computes (`aggs`), each once, and the group keys (`keys`);
- the expressions that run after aggregation (select items, HAVING, ORDER BY), in `pexprs`, whose leaves are
  group-key and aggregate references.
It checks types the way DuckDB does for the subset (decimal scales aligned, a string literal read as a DATE next to a
DATE) and raises with the position of the offending text. Nothing here reads data: the planner and the executor do.
"""
from .table import SharedTable, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, F32_COLUMN
from .parquet import ParquetScan
from .frame import Frame, Overlay, Formula
from .expr import MAX_QUANTILE_BINS, N_COL, N_LIT, N_LITF, N_DIV, N_I2F, N_ADD, N_SUB, N_MUL, N_FLOORDIV, N_MOD, N_NEG, N_LEAST, N_GREATEST, N_SCALE_TO
from .csv import days_from_civil, _days_in_month, parse_float_text, F_OK, F_RANGE
from .sql import (SqlStatement, parse_sql, sql_error, closest, lower_ascii, X_INT, X_DEC, X_STR, X_NULL, X_TRUE, X_FALSE,
                  X_DATE, X_COL, X_NEG, X_BIN, X_NOT, X_ISNULL, X_BETWEEN, X_IN, X_CALL, X_FLT, EXPLAIN_NONE, SqlExpr, SelectItem,
                  X_SUBIN, X_EXISTS, JK_INNER, JK_LEFT, JK_SEMI, JK_ANTI, JK_ANTI_NA,
                  X_CASE, X_CAST, X_LIKE, X_DISTINCT, X_INTERVAL, X_ISBOOL)
from .sqlfn import (TFn, CaseTables, parse_text, ascii_case, tf_kind, char_count, decimal_text, date_text, date_part, date_trunc_days, date_diff_days, add_months, part_code,
                    DP_DAY, DP_EPOCH, TF_LOWER, TF_UPPER, TF_TRIM, TF_SUBSTR, TF_LENGTH, TF_STARTS, TF_ENDS, TF_CONTAINS, TF_LIKE, TF_STRPOS, TF_LEFT, TF_RIGHT,
                    TF_REVERSE, TF_REPLACE)
from .frame import _text_less

comptime K_NULL = 6          # the type of a bare NULL literal

# ── the catalog ──────────────────────────────────────────────────────────────────────────────────────

comptime SRC_TABLE = 0
comptime SRC_FRAME = 1       # a Frame over a table: its edits as a mask over the shared rows plus the changed rows
comptime SRC_SCAN = 2        # a Parquet file read one row group at a time (also a Frame over one)


struct SqlSource(Copyable, Movable):
    var name: String
    var kind: Int
    var table: SharedTable           # the data (a table, a frame's base); a scan's columns without rows
    var scan: Optional[ParquetScan]
    var overlay: Overlay             # a frame's edits as of registration (empty otherwise)
    var framed: Bool
    var version: Int                 # a frame's edit counter when it was registered: stats are cached per version
    var range_cols: List[Int]        # the planner's cache of integer-like column ranges: column, lowest, highest, has values
    var range_lo: List[Int64]
    var range_hi: List[Int64]
    var range_has: List[Bool]
    var fixed_rows: Int              # F-153: the rows of a joined schema whose table holds none (the probe side's); -1 otherwise

    def __init__(out self, name: String, table: SharedTable):
        self.fixed_rows = -1
        self.name = name
        self.kind = SRC_TABLE
        self.table = table
        self.scan = None
        self.overlay = Overlay(table)
        self.framed = False
        self.version = 0
        self.range_cols = List[Int]()
        self.range_lo = List[Int64]()
        self.range_hi = List[Int64]()
        self.range_has = List[Bool]()

    def columns(self) -> Int:
        return self.table.columns()

    def rows(self) -> Int:
        """Rows in the source as registered (a frame: its base rows plus appended, removed rows included)."""
        if self.fixed_rows >= 0:
            return self.fixed_rows
        if self.scan:
            return self.scan.value().rows()
        return self.table.rows()


struct Catalog(Movable):
    """The named sources a query may read. `add_table`, `add_frame` and `add_scan` register one under a name
    (case does not matter); a frame is snapshotted when added: add it again after more edits."""
    var sources: List[SqlSource]
    var view_names: List[String]     # F-159: named views, see `view`
    var view_sql: List[String]

    def __init__(out self):
        self.sources = List[SqlSource]()
        self.view_names = List[String]()
        self.view_sql = List[String]()

    def copy(self) -> Catalog:
        """A catalog of the same sources (tables and scans are shared handles, a frame's snapshot is copied): what a background job (F-175) works on."""
        var out = Catalog()
        for s in self.sources:
            out.sources.append(s.copy())
        out.view_names = self.view_names.copy()
        out.view_sql = self.view_sql.copy()
        return out^

    def view(mut self, name: String, query: String) raises:
        """F-159: a named view. Any query that reads `name` in FROM reads the result of `query` (a SELECT, WITH or UNION over the catalog's sources
        and the views added before it); the query runs once per query that reads it, as a CTE read once does. A source and a view may not share a name."""
        if self.find(name) >= 0:
            raise Error("`" + name + "` is a source of this catalog: a view needs a name of its own")
        var key = lower_ascii(name)
        for i in range(len(self.view_names)):
            if lower_ascii(self.view_names[i]) == key:
                self.view_sql[i] = query
                return
        self.view_names.append(name)
        self.view_sql.append(query)

    def view_index(self, name: String) -> Int:
        var key = lower_ascii(name)
        for i in range(len(self.view_names)):
            if lower_ascii(self.view_names[i]) == key:
                return i
        return -1

    def view_text(self, i: Int) -> String:
        return self.view_sql[i]

    def without_view(self, i: Int) -> Catalog:
        """A copy of the catalog that has the views defined before view `i` (a view reads only those: no recursion)."""
        var out = Catalog()
        for s in self.sources:
            out.sources.append(s.copy())
        for k in range(i):
            out.view_names.append(self.view_names[k])
            out.view_sql.append(self.view_sql[k])
        return out^

    def row_count(self, name: String) -> Int:
        var i = self.find(name)
        return self.sources[i].rows() if i >= 0 else 0

    def reads_a_view(self, st: SqlStatement) -> Bool:
        """Whether a FROM entry of the statement names a view."""
        if len(self.view_names) == 0:
            return False
        if st.table_kid < 0 and not st.table_is_file and self.view_index(st.table) >= 0:
            return True
        if st.has_right and st.right_kid < 0 and not st.right_is_file and self.view_index(st.right_table) >= 0:
            return True
        for ex in st.extra:
            if ex.kid < 0 and not ex.is_file and self.view_index(ex.name) >= 0:
                return True
        return False

    def _put(mut self, var s: SqlSource):
        var key = lower_ascii(s.name)
        for i in range(len(self.sources)):
            if lower_ascii(self.sources[i].name) == key:
                self.sources[i] = s^
                return
        self.sources.append(s^)

    def add_table(mut self, name: String, table: SharedTable):
        self._put(SqlSource(name, table))

    def add_scan(mut self, name: String, scan: ParquetScan):
        var s = SqlSource(name, scan.schema)
        s.kind = SRC_SCAN
        s.scan = scan.copy()
        self._put(s^)

    def add_frame(mut self, name: String, frame: Frame) raises:
        var s = SqlSource(name, frame.base)
        s.kind = SRC_SCAN if frame.scan else SRC_FRAME
        s.scan = frame.scan.copy()
        s.overlay = frame.overlay()
        s.framed = True
        s.version = frame.edits
        self._put(s^)

    def find(self, name: String) -> Int:
        var key = lower_ascii(name)
        for i in range(len(self.sources)):
            if lower_ascii(self.sources[i].name) == key:
                return i
        return -1

    def names(self) -> List[String]:
        var out = List[String]()
        for s in self.sources:
            out.append(s.name)
        return out^


# ── the typed expression tree ────────────────────────────────────────────────────────────────────────

comptime B_COL = 0           # value: the source column
comptime B_LIT = 1           # value: the stored value (kind INT, DECIMAL, DATE or BOOL)
comptime B_STR = 2           # text: a string literal, still to be matched against a dictionary
comptime B_NULL = 3
comptime B_ADD = 10
comptime B_SUB = 11
comptime B_MUL = 12
comptime B_IDIV = 13         # integer division, truncating toward zero; x // 0 is NULL
comptime B_IMOD = 14         # remainder with the dividend's sign; x % 0 is NULL
comptime B_NEG = 15
comptime B_DIV = 16          # `/`: a FLOAT (the operands are FLOATs before aggregation)
comptime B_I2F = 28          # an INT or DECIMAL as a FLOAT (a = the operand, value = its decimal places), as DuckDB casts: float(v) / 10^scale
comptime B_RESCALE = 17      # a times 10^value: the same number with `value` more decimal places
comptime B_FDIV = 18         # a frame formula's `//`: floor division, x // 0 is 0 (the builder's rule, not SQL's)
comptime B_FMOD = 19         # a frame formula's `%`: the divisor's sign, x % 0 is 0
comptime B_LEAST = 26        # a frame formula's min(a, b)
comptime B_GREATEST = 27     # a frame formula's max(a, b)
comptime B_CMP = 20          # value: 20 <, 21 <=, 22 >, 23 >=, 24 =, 25 <> (the kernel's own codes)
comptime B_AND = 30
comptime B_OR = 31
comptime B_NOT = 32
comptime B_ISNULL = 33
comptime B_IN = 34           # a IN (items), the items literals; value 1: the list holds a NULL
comptime B_AGG = 40          # value: the aggregate's number (after aggregation)
comptime B_KEY = 41          # value: the group key's number (after aggregation)
# F-156: expressions. A node whose value is TEXT and not a column or a literal belongs to a dictionary of its own: `src` = DERIVED_BASE + the node's number.
comptime B_CASE = 50         # items: cond, then, cond, then, ..., else (an ELSE the query left out is a B_NULL); value: the number of WHENs
comptime B_COALESCE = 51     # items: the arguments, all of the node's type
comptime B_DISTINCT = 52     # a, b: two operands of one type; value 1: IS NOT DISTINCT FROM, 0: IS DISTINCT FROM (never NULL)
comptime B_CAST = 53         # a: the operand; kind, scale: the target; value: 1 TRY_CAST | bits << 1 (the integer's width) | precision << 8
comptime B_DATEPART = 54     # a: a DATE; value: the part (DP_*); INT
comptime B_DATETRUNC = 55    # a: a DATE; value: the part; DATE
comptime B_ADDMONTHS = 56    # a: a DATE; b: an INT number of months; DATE (the day clamps to the end of the month)
comptime B_DATEDIFF = 57     # a, b: DATEs; value: the part; INT
comptime B_BOOLIS = 58       # a: a BOOL; value 0: IS TRUE, 1: IS NOT TRUE, 2: IS FALSE, 3: IS NOT FALSE
comptime B_TFN = 60          # a function of a TEXT value and constants (`items`: B_STR / B_LIT nodes): value = TF_* | flags << 8; its kind says what it answers
comptime B_CONCAT = 61       # a || b over TEXT; value 1: concat() (a NULL reads as '')
comptime B_QBIN = 62         # F-177 quantile_bin(a, n): INT 1 .. n; value: n; items: the n - 1 edge literals (the executor fills them from an exact pass: `b` is -1 until then, -2 while it does)
comptime DERIVED_BASE = 1 << 20      # a `src` from here on names the dictionary of node (src - DERIVED_BASE) of the pool it is in

comptime C_LT = 20
comptime C_LE = 21
comptime C_GT = 22
comptime C_GE = 23
comptime C_EQ = 24
comptime C_NE = 25


struct BExpr(Copyable, Movable):
    var op: Int
    var a: Int
    var b: Int
    var items: List[Int]
    var kind: Int                    # ColumnKind value of the result, or K_NULL
    var scale: Int
    var value: Int64
    var text: String
    var src: Int                     # a TEXT value: the source column whose dictionary its codes belong to
    var line: Int
    var col: Int

    def __init__(out self, op: Int, kind: Int, scale: Int, a: Int, b: Int, value: Int64, text: String, line: Int, col: Int):
        self.op = op
        self.src = -1
        self.kind = kind
        self.scale = scale
        self.a = a
        self.b = b
        self.items = List[Int]()
        self.value = value
        self.text = text
        self.line = line
        self.col = col


comptime F_COUNT_STAR = 0
comptime F_COUNT = 1
comptime F_SUM = 2
comptime F_MIN = 3
comptime F_MAX = 4
comptime F_AVG = 5
comptime F_COUNT_DISTINCT = 6
comptime F_MEDIAN = 7
comptime F_QUANTILE = 8
comptime F_STDDEV_SAMP = 9       # F-177: stddev / stddev_samp, stddev_pop, variance / var_samp, var_pop: a FLOAT from the group's count, mean and squared deviations
comptime F_STDDEV_POP = 10
comptime F_VAR_SAMP = 11
comptime F_VAR_POP = 12
comptime F_QEDGE = 13            # internal: edge `p` of `bins` quantile bins of the argument (what quantile_bin(x, n) needs, found by sqlexec.resolve_bins)


struct AggSpec(Copyable, Movable):
    var func: Int
    var arg: Int                     # a row-level expression of `rexprs`, -1 for COUNT(*)
    var p: Float64
    var kind: Int                    # the result type
    var scale: Int
    var name: String                 # as the output names it: sum(revenue), count_star()
    var canon: String
    var hol: Bool                    # needs every value of its group: count(DISTINCT), median, quantile_disc, min / max of TEXT
    var bins: Int                    # F_QEDGE: the number of bins

    def __init__(out self, func: Int, arg: Int, p: Float64, kind: Int, scale: Int, name: String, canon: String, hol: Bool = False, bins: Int = 0):
        self.hol = hol
        self.bins = bins
        self.func = func
        self.arg = arg
        self.p = p
        self.kind = kind
        self.scale = scale
        self.name = name
        self.canon = canon


struct KeySpec(Copyable, Movable):
    var expr: Int                    # a row-level expression
    var name: String
    var canon: String

    def __init__(out self, expr: Int, name: String, canon: String):
        self.expr = expr
        self.name = name
        self.canon = canon


struct SortKey(Copyable, Movable):
    var expr: Int                    # a post-aggregate expression
    var desc: Bool
    var nulls_first: Bool

    def __init__(out self, expr: Int, desc: Bool, nulls_first: Bool):
        self.expr = expr
        self.desc = desc
        self.nulls_first = nulls_first


struct BoundQuery(Movable):
    var source: Int                  # the catalog's source
    var rexprs: List[BExpr]
    var pexprs: List[BExpr]
    var where: Int                   # a row-level boolean, -1 for none
    var keys: List[KeySpec]
    var aggs: List[AggSpec]
    var outputs: List[Int]           # per select item: a post-aggregate expression
    var out_names: List[String]
    var having: Int                  # a post-aggregate boolean, -1 for none
    var order: List[SortKey]
    var limit: Int
    var offset: Int
    var explain: Int
    var text: String                 # the SQL
    var line: Int                    # where its select list starts (for errors that belong to the whole query)
    var col: Int
    var where_text: String
    var having_text: String
    var order_text: List[String]
    var out_text: List[String]
    # F-152: row queries (no aggregate, no GROUP BY) and SELECT DISTINCT
    # F-153: a join of two tables. Columns are numbered over both: the first table's, then the second's (`tab_start`).
    var joined: Bool
    var join_kind: Int               # F-154: JK_INNER, JK_LEFT, JK_SEMI, JK_ANTI, JK_ANTI_NA (a semi / anti join keeps only the first table's columns)
    var sources: List[Int]           # the catalog's sources, one per table of FROM
    var tab_names: List[String]
    var tab_alias: List[String]
    var tab_start: List[Int]
    var tab_cols: List[Int]
    var jk_l: List[Int]              # the equi-join keys: a column of the first table ...
    var jk_r: List[Int]              # ... and the column of the second that must equal it
    var jk_l_names: List[String]
    var jk_r_names: List[String]
    var conj_l: List[Int]            # WHERE / ON conjuncts that read only the first table, only the second, or both
    var conj_r: List[Int]
    var conj_x: List[Int]
    var conj_l_text: List[String]
    var conj_r_text: List[String]
    var conj_x_text: List[String]
    var using_cols: List[String]
    # F-155: a join of three or more tables (inner joins): per table its own conjuncts, the equalities between tables (columns, global numbers), and the
    # conjuncts that read two or more tables with the bit set of the tables they read (bit t = table t)
    var col_names: List[String]      # the name of each global column (the tables' columns one after the other)
    var leaf_conj: List[List[Int]]
    var leaf_conj_text: List[List[String]]
    var edge_a: List[Int]
    var edge_b: List[Int]
    var resid: List[Int]
    var resid_text: List[String]
    var resid_mask: List[Int]
    var rows: Bool                   # a row query: `routs` and `rorder` hold its select list and ORDER BY (expressions of `rexprs`)
    var distinct: Bool               # SELECT DISTINCT (bound as a group-by on the select list)
    var routs: List[Int]
    var rorder: List[SortKey]

    def src_name(self, g: Int) -> String:
        return self.col_names[g] if g < len(self.col_names) else String()

    def __init__(out self):
        self.joined = False
        self.join_kind = JK_INNER
        self.sources = List[Int]()
        self.tab_names = List[String]()
        self.tab_alias = List[String]()
        self.tab_start = List[Int]()
        self.tab_cols = List[Int]()
        self.jk_l = List[Int]()
        self.jk_r = List[Int]()
        self.jk_l_names = List[String]()
        self.jk_r_names = List[String]()
        self.conj_l = List[Int]()
        self.conj_r = List[Int]()
        self.conj_x = List[Int]()
        self.conj_l_text = List[String]()
        self.conj_r_text = List[String]()
        self.conj_x_text = List[String]()
        self.using_cols = List[String]()
        self.col_names = List[String]()
        self.leaf_conj = List[List[Int]]()
        self.leaf_conj_text = List[List[String]]()
        self.edge_a = List[Int]()
        self.edge_b = List[Int]()
        self.resid = List[Int]()
        self.resid_text = List[String]()
        self.resid_mask = List[Int]()
        self.rows = False
        self.distinct = False
        self.routs = List[Int]()
        self.rorder = List[SortKey]()
        self.source = -1
        self.rexprs = List[BExpr]()
        self.pexprs = List[BExpr]()
        self.where = -1
        self.keys = List[KeySpec]()
        self.aggs = List[AggSpec]()
        self.outputs = List[Int]()
        self.out_names = List[String]()
        self.having = -1
        self.order = List[SortKey]()
        self.limit = -1
        self.offset = 0
        self.explain = EXPLAIN_NONE
        self.text = String()
        self.line = 1
        self.col = 1
        self.where_text = String()
        self.having_text = String()
        self.order_text = List[String]()
        self.out_text = List[String]()


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────

def pow10_checked(k: Int) raises -> Int64:
    if k < 0 or k > 18:
        raise Error("a power of ten outside 10^0 .. 10^18")
    var v: Int64 = 1
    for _ in range(k):
        v *= 10
    return v


def kind_name(kind: Int, scale: Int = 0) -> String:
    if kind == K_NULL:
        return "NULL"
    if kind == 1:
        return "DECIMAL(" + String(scale) + ")"
    var names: List[String] = ["INT", "DECIMAL", "TEXT", "DATE", "BOOL", "FLOAT"]
    return names[kind]


def _is_numeric(kind: Int) -> Bool:
    return kind == INT.value or kind == DECIMAL.value


def _is_num(kind: Int) -> Bool:
    """A number the arithmetic takes: INT, DECIMAL or FLOAT."""
    return kind == INT.value or kind == DECIMAL.value or kind == FLOAT.value


def decimal_to_float(v: Int64, scale: Int) -> Float64:
    """A stored decimal as a float64, as DuckDB casts DECIMAL to DOUBLE: the integer as a double, divided by the power of ten."""
    if scale == 0:
        return Float64(v)
    var p = Float64(1.0)
    for _ in range(scale):
        p *= 10.0                     # exact up to 10^22
    return Float64(v) / p


def decimal_to_f32(v: Int64, scale: Int) -> Float64:
    """A stored decimal as a 32-bit float (widened), as DuckDB casts DECIMAL to REAL: the integer as a float32 divided by the power
    of ten in float32 (exact for up to ten places and 24 bits of integer), else the float64 value rounded to float32."""
    if scale <= 10 and v > -16777216 and v < 16777216:
        var p = Float32(1.0)
        for _ in range(scale):
            p *= 10.0
        return Float64(Float32(Float64(v)) / p)
    return Float64(Float32(decimal_to_float(v, scale)))


def date_problem(s: String) -> String:
    """Why 'YYYY-MM-DD' text is not a date, "" when it is one."""
    var b = s.as_bytes()
    if len(b) != 10 or b[4] != UInt8(45) or b[7] != UInt8(45):
        return "write it as YYYY-MM-DD"
    var nums = List[Int]()
    for rng in [(0, 4), (5, 7), (8, 10)]:
        var v = 0
        for i in range(rng[0], rng[1]):
            if not (UInt8(48) <= b[i] <= UInt8(57)):
                return "write it as YYYY-MM-DD"
            v = v * 10 + Int(b[i] - 48)
        nums.append(v)
    if nums[1] < 1 or nums[1] > 12:
        return "there is no month " + String(nums[1])
    var days = _days_in_month(nums[0], nums[1])
    if nums[2] < 1 or nums[2] > days:
        var names: List[String] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        return "the day " + String(nums[2]) + " does not exist: " + names[nums[1] - 1] + " " + String(s[byte=0:4]) + " has " + String(days) + " days"
    return ""


def parse_date_literal(s: String) -> Int:
    """Days since 1970-01-01 of 'YYYY-MM-DD', or Int.MIN when the text is not such a date."""
    var b = s.as_bytes()
    if len(b) != 10 or b[4] != UInt8(45) or b[7] != UInt8(45):
        return Int.MIN
    var nums = List[Int]()
    for rng in [(0, 4), (5, 7), (8, 10)]:
        var v = 0
        for i in range(rng[0], rng[1]):
            if not (UInt8(48) <= b[i] <= UInt8(57)):
                return Int.MIN
            v = v * 10 + Int(b[i] - 48)
        nums.append(v)
    if nums[1] < 1 or nums[1] > 12 or nums[2] < 1 or nums[2] > _days_in_month(nums[0], nums[1]):
        return Int.MIN
    return days_from_civil(nums[0], nums[1], nums[2])


struct _Num(Movable):
    var value: Int64
    var scale: Int
    var ok: Bool

    def __init__(out self):
        self.value = 0
        self.scale = 0
        self.ok = False


def parse_number(text: String) -> _Num:
    """The stored value and scale of a literal such as `-12.50`; ok is False past int64 or 18 decimal places."""
    var out = _Num()
    var b = text.as_bytes()
    var neg = False
    var i = 0
    if len(b) > 0 and b[0] == UInt8(45):
        neg = True
        i = 1
    var v = UInt64(0)
    var scale = 0
    var seen_dot = False
    var limit = UInt64(9223372036854775807) + (UInt64(1) if neg else UInt64(0))
    while i < len(b):
        if b[i] == UInt8(46):
            seen_dot = True
            i += 1
            continue
        var d = UInt64(b[i] - 48)
        if v > (limit - d) // 10:
            return out^
        v = v * 10 + d
        if seen_dot:
            scale += 1
        i += 1
    if scale > 18:
        return out^
    out.value = Int64(0) - Int64(v) if neg else Int64(v)
    out.scale = scale
    out.ok = True
    return out^


# ── the binder ───────────────────────────────────────────────────────────────────────────────────────

struct _Binder(Movable):
    var st: SqlStatement
    var cat_names: List[String]
    var src_names: List[String]       # the source's columns
    var src_kinds: List[Int]
    var src_scales: List[Int]
    var table_name: String
    var table_alias: String
    var q: BoundQuery
    var in_post: Bool
    var key_canons: List[String]
    var alias_expr: List[Int]         # post-aggregate expression of each select item, by item (-1 while unbound)
    var n_stored: Int                 # the source's stored columns; the frame's formula columns follow them
    var formulas: List[Formula]
    var distinct: Bool
    var tab_names: List[String]       # one per table of FROM (F-153)
    var tab_alias: List[String]
    var tab_start: List[Int]
    var tab_cols: List[Int]
    var using_cols: List[String]
    var jkind: Int                    # F-154: the kind of join
    var in_on: Bool                   # binding an ON condition (a semi / anti join's right table is visible there only)
    var inner_first: Bool             # the ON came from a subquery: a bare name means the subquery's table when it has the column
    var cases: Optional[CaseTables]   # F-156: the case-mapping tables, built when a constant needs them

    def __init__(out self, var st: SqlStatement):
        self.cases = None
        self.jkind = JK_INNER
        self.in_on = False
        self.inner_first = False
        self.tab_names = List[String]()
        self.tab_alias = List[String]()
        self.tab_start = List[Int]()
        self.tab_cols = List[Int]()
        self.using_cols = List[String]()
        self.distinct = False
        self.st = st^
        self.cat_names = List[String]()
        self.src_names = List[String]()
        self.src_kinds = List[Int]()
        self.src_scales = List[Int]()
        self.table_name = String()
        self.table_alias = String()
        self.q = BoundQuery()
        self.in_post = False
        self.key_canons = List[String]()
        self.alias_expr = List[Int]()
        self.n_stored = 0
        self.formulas = List[Formula]()

    def take(deinit self) -> BoundQuery:
        return self.q^

    # -- tree building
    def mk(mut self, op: Int, kind: Int, scale: Int, a: Int, b: Int, value: Int64, text: String, ast: Int) -> Int:
        var line = self.st.exprs[ast].line if ast >= 0 else 0
        var col = self.st.exprs[ast].col if ast >= 0 else 0
        if self.in_post:
            self.q.pexprs.append(BExpr(op, kind, scale, a, b, value, text, line, col))
            return len(self.q.pexprs) - 1
        self.q.rexprs.append(BExpr(op, kind, scale, a, b, value, text, line, col))
        return len(self.q.rexprs) - 1

    def at(self, e: Int) -> BExpr:
        return self.q.pexprs[e].copy() if self.in_post else self.q.rexprs[e].copy()

    def err(self, ast: Int, msg: String) -> Error:
        return sql_error(self.st.exprs[ast].line, self.st.exprs[ast].col, msg)

    # -- names
    def lookup(self, ast: Int) -> Tuple[Int, Int]:
        """The column a name means, and how it went: 0 found, 1 unknown table or alias, 2 no such column, 3 ambiguous inside one table,
        4 ambiguous between the two tables (a USING column is not: it means the first table's)."""
        ref x = self.st.exprs[ast]
        var lo = 0
        var hi = len(self.src_names)
        var left_only = self.is_semi_anti() and not self.in_on        # after a semi / anti join only the first table has columns
        if left_only:
            hi = self.tab_start[1]
        if x.text2.byte_length() > 0:
            var q = lower_ascii(x.text2)
            var tab = -1
            if self.inner_first and self.in_on and len(self.tab_names) > 1:
                # a subquery's condition: names resolve innermost scope first. A qualifier is the subquery table's when it is that table's alias, or its name
                # when it has no alias; only then the outer table's (`#0` and `#1` are the binder's own handles for the two sides)
                if q == "#0":
                    tab = 0
                elif q == "#1":
                    tab = 1
                else:
                    for t in [1, 0]:
                        var shown = lower_ascii(self.tab_alias[t]) if self.tab_alias[t].byte_length() > 0 else lower_ascii(self.tab_names[t])
                        if q == shown:
                            tab = t
                            break
            if tab < 0:
                for t in range(len(self.tab_names)):
                    if q == lower_ascii(self.tab_names[t]) or (self.tab_alias[t].byte_length() > 0 and q == lower_ascii(self.tab_alias[t])):
                        tab = t
                        break
            if tab < 0 or (left_only and tab == 1):
                return (-1, 1)
            lo = self.tab_start[tab]
            hi = lo + self.tab_cols[tab]
        elif self.inner_first and self.in_on and len(self.tab_start) > 1:
            # a subquery's condition: a bare name is the subquery's column when its table has one, else the outer query's
            var key1 = lower_ascii(x.text)
            for c in range(self.tab_start[1], self.tab_start[1] + self.tab_cols[1]):
                if lower_ascii(self.src_names[c]) == key1:
                    return (c, 0)
            hi = self.tab_start[1]
        var key = lower_ascii(x.text)
        var found = -1
        var found_tab = -1
        for c in range(lo, hi):
            if lower_ascii(self.src_names[c]) == key:
                var t = 0
                for u in range(len(self.tab_start)):
                    if c >= self.tab_start[u]:
                        t = u
                if found >= 0:
                    if t == found_tab:
                        return (found, 3)
                    for un in self.using_cols:
                        if lower_ascii(un) == key:
                            return (found, 0)
                    return (found, 4)
                found = c
                found_tab = t
        if found < 0:
            return (-1, 2)
        return (found, 0)

    def is_semi_anti(self) -> Bool:
        return self.jkind == JK_SEMI or self.jkind == JK_ANTI or self.jkind == JK_ANTI_NA

    def find_column(self, ast: Int) raises -> Int:
        ref x = self.st.exprs[ast]
        var r = self.lookup(ast)
        if r[1] == 1:
            raise self.err(ast, "unknown table or alias `" + x.text2 + "` in `" + x.text2 + "." + x.text + "`")
        if r[1] == 3:
            var key = lower_ascii(x.text)
            var first = -1
            var second = -1
            for c in range(len(self.src_names)):
                if lower_ascii(self.src_names[c]) == key:
                    if first < 0:
                        first = c
                    elif second < 0:
                        second = c
            raise self.err(ast, "column `" + x.text + "` is ambiguous: `" + self.src_names[first] + "` and `" + self.src_names[second] + "`")
        if r[1] == 4:
            var key4 = lower_ascii(x.text)
            var ta = -1
            var tb = -1
            for c in range(len(self.src_names)):
                if lower_ascii(self.src_names[c]) == key4:
                    var tt = self.table_of(c)
                    if ta < 0:
                        ta = tt
                    elif tb < 0 and tt != ta:
                        tb = tt
            raise self.err(ast, "column `" + x.text + "` is ambiguous: it is in both `" + self.tab_label(ta) + "` and `" + self.tab_label(tb) + "` (write " +
                           self.tab_label(ta) + "." + x.text + " or " + self.tab_label(tb) + "." + x.text + ")")
        if r[1] == 2:
            var hint = closest(x.text, self.src_names)
            var where_ = ("table `" + self.table_name + "`") if len(self.tab_names) < 2 else (" or ".join(self.tab_labels()))
            if x.text2.byte_length() > 0 and len(self.tab_names) > 1:
                where_ = "`" + x.text2 + "`"
            raise self.err(ast, "column `" + x.text + "` not found in " + where_ +
                           ((" (did you mean `" + hint + "`?)") if hint.byte_length() > 0 else ""))
        return r[0]

    def split_and(self, e: Int, mut out: List[Int]):
        """The conjuncts of an AND chain (syntax-tree indices)."""
        ref x = self.st.exprs[e]
        if x.kind == X_BIN and x.text == "AND":
            var a = x.a
            var b = x.b
            self.split_and(a, out)
            self.split_and(b, out)
        else:
            out.append(e)

    def or_terms(self, e: Int, mut out: List[Int]):
        ref x = self.st.exprs[e]
        if x.kind == X_BIN and x.text == "OR":
            var a = x.a
            var b = x.b
            self.or_terms(a, out)
            self.or_terms(b, out)
        else:
            out.append(e)

    def chain(mut self, op: String, parts: List[Int]) -> Int:
        var acc = parts[0]
        for k in range(1, len(parts)):
            self.st.exprs.append(SqlExpr(X_BIN, op, String(), acc, parts[k], -1, List[Int](), False, self.st.exprs[parts[k]].line, self.st.exprs[parts[k]].col))
            acc = len(self.st.exprs) - 1
        return acc

    def factor_or(mut self, c: Int, mut out: List[Int]):
        """`(A AND x) OR (A AND y)` is `A AND (x OR y)`: the conjuncts every disjunct has come out as conjuncts of their own (an equality between two tables
        in all of them becomes a join key: TPC-H Q19), the rest stays an OR."""
        var terms = List[Int]()
        self.or_terms(c, terms)
        var lists = List[List[Int]]()
        for t in terms:
            var l = List[Int]()
            self.split_and(t, l)
            lists.append(l^)
        var texts = List[List[String]]()
        for l in lists:
            var tx = List[String]()
            for y in l:
                tx.append(self.st.render(y))
            texts.append(tx^)
        var common = List[Int]()
        var common_text = List[String]()
        for i in range(len(lists[0])):
            var inall = True
            for k in range(1, len(lists)):
                var found = False
                for z in texts[k]:
                    if z == texts[0][i]:
                        found = True
                if not found:
                    inall = False
            if inall:
                common.append(lists[0][i])
                common_text.append(texts[0][i])
        if len(common) == 0:
            out.append(c)
            return
        var rest = List[Int]()
        var all_have_rest = True
        for k in range(len(lists)):
            var rem = List[Int]()
            for i in range(len(lists[k])):
                var is_common = False
                for z in common_text:
                    if z == texts[k][i]:
                        is_common = True
                if not is_common:
                    rem.append(lists[k][i])
            if len(rem) == 0:
                all_have_rest = False
            else:
                rest.append(self.chain("AND", rem))
        for x in common:
            out.append(x)
        if all_have_rest:
            out.append(self.chain("OR", rest))

    def split_conj(mut self, e: Int, mut out: List[Int]):
        """The conjuncts of a join condition, with the common factors of its ORs pulled out."""
        var tmp = List[Int]()
        self.split_and(e, tmp)
        for c in tmp:
            if self.st.exprs[c].kind == X_BIN and self.st.exprs[c].text == "OR":
                self.factor_or(c, out)
            else:
                out.append(c)

    def sides(self, e: Int) -> Int:
        """Which tables a row-level expression reads: bit 1 the first, bit 2 the second."""
        ref x = self.q.rexprs[e]
        var m = 0
        if x.op == B_COL:
            return 1 << self.table_of(Int(x.value))
        if x.a >= 0:
            m |= self.sides(x.a)
        if x.b >= 0:
            m |= self.sides(x.b)
        for it in x.items:
            m |= self.sides(it)
        return m

    def key_pair_check(self, lc: Int, rc: Int, line: Int, col: Int) raises:
        var lk = self.src_kinds[lc]
        var rk = self.src_kinds[rc]
        var ok = (_is_numeric(lk) and _is_numeric(rk)) or (lk == rk and (lk == DATE.value or lk == BOOL.value or lk == TEXT.value or lk == FLOAT.value))
        if not ok:
            raise sql_error(line, col, "cannot join " + self.tab_label(self.table_of(lc)) + "." + self.src_names[lc] + " (" + kind_name(lk, self.src_scales[lc]) + ") with " +
                           self.tab_label(self.table_of(rc)) + "." + self.src_names[rc] + " (" + kind_name(rk, self.src_scales[rc]) + "): the two sides must be the same kind of value" +
                           (" (a FLOAT is joined with a FLOAT only)" if lk == FLOAT.value or rk == FLOAT.value else ""))

    def bind_where(mut self) raises:
        """WHERE (and, for a join, ON): one table binds the whole condition; two tables split it into conjuncts: an equality between a
        column of each table is an equi-join key, a conjunct that reads one table filters that table before the join, the rest is
        checked on the joined rows."""
        if not self.q.joined:
            if self.st.where >= 0:
                var w = self.value(self.st.where)
                self.q.where = self.boolean(w, self.st.where, "WHERE")
                self.q.where_text = self.st.render(self.st.where)
            return
        var conj = List[Int]()
        if self.st.join_on >= 0:
            if _has_aggregate(self.st, self.st.join_on):
                raise self.err(self.st.join_on, "aggregate functions are not allowed in ON")
            self.split_conj(self.st.join_on, conj)
        var n_on = len(conj)                                  # the first n_on conjuncts are the ON condition's
        if self.st.where >= 0:
            self.split_conj(self.st.where, conj)
        if len(self.tab_names) > 2:
            if len(self.st.join_using) > 0:
                raise sql_error(self.st.right_line, self.st.right_col, "`USING` in a join of more than two tables is not supported yet: write ON")
            self.bind_where_multi(conj)
            return
        # USING (k): the first table's k equals the second's
        for name in self.st.join_using:
            var lc = -1
            var rc = -1
            for c in range(self.tab_start[0], self.tab_start[0] + self.tab_cols[0]):
                if lower_ascii(self.src_names[c]) == lower_ascii(name):
                    lc = c
            for c in range(self.tab_start[1], self.tab_start[1] + self.tab_cols[1]):
                if lower_ascii(self.src_names[c]) == lower_ascii(name):
                    rc = c
            if lc < 0 or rc < 0:
                raise sql_error(self.st.right_line, self.st.right_col, "USING (" + name + "): the column is not in " + ("`" + self.tab_label(0) + "`" if lc < 0 else "`" + self.tab_label(1) + "`"))
            self.key_pair_check(lc, rc, self.st.right_line, self.st.right_col)
            self.q.jk_l.append(lc)
            self.q.jk_r.append(rc)
            self.q.jk_l_names.append(self.src_names[lc])
            self.q.jk_r_names.append(self.src_names[rc])
        var residual = List[Int]()
        for ci in range(len(conj)):
            var c = conj[ci]
            self.in_on = ci < n_on
            ref x = self.st.exprs[c]
            # an equality between a column of each table is a join key: from ON (or USING, or the comma join's WHERE) always; a WHERE equality of a LEFT / SEMI /
            # ANTI join is a filter on the joined rows (it removes a LEFT join's NULL-extended rows), so it is not a key there
            if x.kind == X_BIN and x.text == "=" and self.st.exprs[x.a].kind == X_COL and self.st.exprs[x.b].kind == X_COL and (self.jkind == JK_INNER or ci < n_on):
                var ra = self.lookup(x.a)
                var rb = self.lookup(x.b)
                if ra[1] == 0 and rb[1] == 0 and self.table_of(ra[0]) != self.table_of(rb[0]):
                    var lc = ra[0] if self.table_of(ra[0]) == 0 else rb[0]
                    var rc = rb[0] if self.table_of(ra[0]) == 0 else ra[0]
                    self.key_pair_check(lc, rc, x.line, x.col)
                    self.q.jk_l.append(lc)
                    self.q.jk_r.append(rc)
                    self.q.jk_l_names.append(self.src_names[lc])
                    self.q.jk_r_names.append(self.src_names[rc])
                    continue
            var e = self.boolean(self.value(c), c, "ON" if ci < n_on else "WHERE")
            var m = self.sides(e)
            var from_on = ci < n_on
            var inner = self.jkind == JK_INNER
            if not inner and from_on and (m == 3 or (m == 1 and self.jkind != JK_SEMI) or (m == 0 and self.jkind != JK_SEMI)):
                raise self.err(c, "a condition in ON that is not an equality between a column of each table (or a condition on the second table alone) is not supported in a " +
                               self.join_name() + " yet (F-155)")
            if m == 1:
                self.q.conj_l.append(e)
                self.q.conj_l_text.append(self.st.render(c))
            elif m == 2 and (inner or from_on):
                self.q.conj_r.append(e)
                self.q.conj_r_text.append(self.st.render(c))
            else:
                self.q.conj_x.append(e)
                self.q.conj_x_text.append(self.st.render(c))
        self.in_on = False
        if len(self.q.jk_l) == 0:
            if self.st.from_subquery:
                raise sql_error(self.st.right_line, self.st.right_col, "the subquery needs an equality between one of its columns and a column of the outer query (IN gives it; " +
                                "an EXISTS without a correlated equality is a constant, which is not supported)")
            raise sql_error(self.st.right_line, self.st.right_col, "the join has no equality between a column of `" + self.tab_label(0) + "` and a column of `" + self.tab_label(1) +
                            "`: a join needs one in ON, USING or WHERE (a cross join is not supported)")
        # the conjuncts that read both tables, and the constants, are checked on the joined rows: one AND chain
        if len(self.q.conj_x) > 0:
            var acc = self.q.conj_x[0]
            for k in range(1, len(self.q.conj_x)):
                self.q.rexprs.append(BExpr(B_AND, BOOL.value, 0, acc, self.q.conj_x[k], 0, "", 0, 0))
                acc = len(self.q.rexprs) - 1
            self.q.where = acc
            var t = String()
            for k in range(len(self.q.conj_x_text)):
                t += (" AND " if k > 0 else "") + self.q.conj_x_text[k]
            self.q.where_text = t

    def join_name(self) -> String:
        if self.jkind == JK_LEFT:
            return "LEFT JOIN"
        if self.jkind == JK_SEMI:
            return "SEMI JOIN (or a subquery)"
        return "ANTI JOIN (or NOT IN / NOT EXISTS)"

    def tab_labels(self) -> List[String]:
        var out = List[String]()
        for t in range(len(self.tab_names)):
            out.append("`" + self.tab_label(t) + "`")
        return out^

    def bind_where_multi(mut self, conj: List[Int]) raises:
        """Three or more tables (inner joins): each conjunct is an equality between columns of two tables (an edge of the join graph), a condition on one table
        (it filters that table first), or a condition that reads several (checked as soon as all of them are joined, or last). The tables must be connected by
        equalities: a cross product is not supported."""
        var n = len(self.tab_names)
        for t in range(n):
            self.q.leaf_conj.append(List[Int]())
            self.q.leaf_conj_text.append(List[String]())
        for ci in range(len(conj)):
            var c = conj[ci]
            ref x = self.st.exprs[c]
            if x.kind == X_BIN and x.text == "=" and self.st.exprs[x.a].kind == X_COL and self.st.exprs[x.b].kind == X_COL:
                var ra = self.lookup(x.a)
                var rb = self.lookup(x.b)
                if ra[1] == 0 and rb[1] == 0 and self.table_of(ra[0]) != self.table_of(rb[0]):
                    self.key_pair_check(ra[0], rb[0], x.line, x.col)
                    self.q.edge_a.append(ra[0])
                    self.q.edge_b.append(rb[0])
                    continue
            var e = self.boolean(self.value(c), c, "WHERE")
            var m = self.sides(e)
            var one = m != 0 and (m & (m - 1)) == 0
            if one:
                var t = 0
                while (1 << t) != m:
                    t += 1
                self.q.leaf_conj[t].append(e)
                self.q.leaf_conj_text[t].append(self.st.render(c))
            else:
                self.q.resid.append(e)
                self.q.resid_text.append(self.st.render(c))
                self.q.resid_mask.append(m)
        # connected by equalities
        var seen = List[Bool](length=n, fill=False)
        seen[0] = True
        var grew = True
        while grew:
            grew = False
            for k in range(len(self.q.edge_a)):
                var ta = self.table_of(self.q.edge_a[k])
                var tb = self.table_of(self.q.edge_b[k])
                if seen[ta] != seen[tb]:
                    seen[ta] = True
                    seen[tb] = True
                    grew = True
        for t in range(n):
            if not seen[t]:
                raise sql_error(self.st.right_line, self.st.right_col, "`" + self.tab_label(t) + "` is not joined to the other tables by an equality in ON or WHERE (a cross join is not supported)")

    def tab_label(self, t: Int) -> String:
        return self.tab_alias[t] if self.tab_alias[t].byte_length() > 0 else self.tab_names[t]

    def table_of(self, col: Int) -> Int:
        var t = 0
        for u in range(len(self.tab_start)):
            if col >= self.tab_start[u]:
                t = u
        return t

    def canon(self, e: Int) -> String:
        """The expression with names in lower case, to tell whether a select item is a GROUP BY key."""
        ref x = self.st.exprs[e]
        if x.kind == X_COL:
            var r = self.lookup(e)
            if r[1] == 0:
                return "#" + String(r[0])
            return lower_ascii(x.text2) + "." + lower_ascii(x.text)
        if x.kind == X_CALL:
            var s = lower_ascii(x.text) + "("
            if x.a == 1:
                s += "*"
            elif x.flag:
                s += "distinct "
            for k in range(len(x.items)):
                s += ("," if k > 0 else "") + self.canon(x.items[k])
            return s + ")"
        if x.kind == X_BIN:
            return "(" + self.canon(x.a) + x.text + self.canon(x.b) + ")"
        if x.kind == X_NEG or x.kind == X_NOT:
            return x.text + "(" + self.canon(x.a) + ")"
        if x.kind == X_ISNULL:
            return "isnull" + ("not" if x.flag else "") + "(" + self.canon(x.a) + ")"
        if x.kind == X_BETWEEN:
            return "between" + ("not" if x.flag else "") + "(" + self.canon(x.a) + "," + self.canon(x.b) + "," + self.canon(x.c) + ")"
        if x.kind == X_IN:
            var s = "in" + ("not" if x.flag else "") + "(" + self.canon(x.a)
            for k in range(len(x.items)):
                s += "," + self.canon(x.items[k])
            return s + ")"
        if x.kind == X_STR:
            return "'" + x.text + "'"
        if x.kind == X_CASE:
            var s = "case(" + (self.canon(x.a) if x.a >= 0 else "")
            for k in range(len(x.items)):
                s += "," + self.canon(x.items[k])
            return s + ";" + (self.canon(x.c) if x.c >= 0 else "") + ")"
        if x.kind == X_CAST:
            return ("trycast(" if x.flag else "cast(") + self.canon(x.a) + ":" + x.text + String(x.b) + "," + String(x.c) + ")"
        if x.kind == X_LIKE:
            return "like(" + x.text + ("not" if x.flag else "") + "," + self.canon(x.a) + "," + self.canon(x.b) + "," + (self.canon(x.c) if x.c >= 0 else "") + ")"
        if x.kind == X_DISTINCT:
            return "distinct" + ("not" if x.flag else "") + "(" + self.canon(x.a) + "," + self.canon(x.b) + ")"
        if x.kind == X_INTERVAL:
            return "interval(" + self.canon(x.a) + "," + x.text + ")"
        if x.kind == X_ISBOOL:
            return "isbool" + x.text + ("not" if x.flag else "") + "(" + self.canon(x.a) + ")"
        return x.text

    # -- a frame's formula column: its expression over stored columns, bound as the builder evaluates it
    def formula_column(mut self, fi: Int, ast: Int, depth: Int) raises -> Int:
        if depth > 32:
            raise self.err(ast, "formulas refer to each other in a cycle")
        var stack = List[Int]()
        var nodes = self.formulas[fi].expr.nodes.copy()
        var fname = self.formulas[fi].name
        var fscale = self.formulas[fi].scale
        for n in nodes:
            if n.op == N_COL:
                var found = -1
                for c in range(len(self.src_names)):
                    if self.src_names[c] == n.name:
                        found = c
                        break
                if found < 0:
                    raise self.err(ast, "formula `" + fname + "` reads column `" + n.name + "`, which is not there")
                if found >= self.n_stored:
                    stack.append(self.formula_column(found - self.n_stored, ast, depth + 1))
                else:
                    var node = self.mk(B_COL, self.src_kinds[found], self.src_scales[found], -1, -1, Int64(found), String(self.src_names[found]), ast)
                    if self.src_kinds[found] == TEXT.value:
                        self.q.rexprs[node].src = found
                    stack.append(node)
            elif n.op == N_LIT:
                stack.append(self.mk(B_LIT, INT.value, 0, -1, -1, n.value, "", ast))
            elif n.op == N_LITF:
                stack.append(self.mk(B_LIT, FLOAT.value, 0, -1, -1, n.value, "", ast))
            elif n.op == N_I2F:
                var a = stack.pop()
                stack.append(self.mk(B_I2F, FLOAT.value, 0, a, -1, n.value, "", ast))
            elif n.op == N_SCALE_TO:
                pass                                              # the child already holds the scaled number
            elif n.op == N_NEG:
                var a = stack.pop()
                stack.append(self.mk(B_NEG, self.at(a).kind, 0, a, -1, 0, "", ast))
            else:
                var b = stack.pop()
                var a = stack.pop()
                var op = B_ADD if n.op == N_ADD else B_SUB if n.op == N_SUB else B_MUL if n.op == N_MUL else B_FDIV if n.op == N_FLOORDIV else \
                         B_FMOD if n.op == N_MOD else B_LEAST if n.op == N_LEAST else B_GREATEST if n.op == N_GREATEST else B_DIV if n.op == N_DIV else -1
                if op < 0:
                    raise self.err(ast, "formula `" + fname + "` uses an operator SQL cannot read yet")
                var float_op = self.at(a).kind == FLOAT.value or self.at(b).kind == FLOAT.value or n.op == N_DIV
                stack.append(self.mk(op, FLOAT.value if float_op else INT.value, 0, a, b, 0, "", ast))
        var top = stack.pop()
        self.q.rexprs[top].kind = FLOAT.value if fscale < 0 else DECIMAL.value if fscale > 0 else INT.value
        self.q.rexprs[top].scale = max(fscale, 0)
        return top

    # -- literals
    def literal(mut self, ast: Int) raises -> Int:
        var x = self.st.exprs[ast].copy()
        if x.kind == X_INT or x.kind == X_DEC:
            var n = parse_number(x.text)
            if not n.ok:
                raise self.err(ast, "the number " + x.text + " does not fit: integers must fit 64 bits and decimals 18 places")
            return self.mk(B_LIT, INT.value if x.kind == X_INT else DECIMAL.value, n.scale, -1, -1, n.value, String(x.text), ast)
        if x.kind == X_FLT:
            var f = Float64(0.0)
            var st = parse_float_text(x.text, f)
            if st == F_RANGE:
                raise self.err(ast, "the number " + x.text + " is past the largest float64 (about 1.8e308)")
            return self.mk(B_LIT, FLOAT.value, 0, -1, -1, Int64(f.to_bits()), String(x.text), ast)
        if x.kind == X_STR:
            return self.lit_str(String(x.text), ast)
        if x.kind == X_NULL:
            return self.mk(B_NULL, K_NULL, 0, -1, -1, 0, "", ast)
        if x.kind == X_TRUE or x.kind == X_FALSE:
            return self.mk(B_LIT, BOOL.value, 0, -1, -1, Int64(1 if x.kind == X_TRUE else 0), String(x.text), ast)
        var days = parse_date_literal(x.text)
        if days == Int.MIN:
            raise self.err(ast, "`" + x.text + "` is not a date: " + date_problem(x.text))
        return self.mk(B_LIT, DATE.value, 0, -1, -1, Int64(days), String(x.text), ast)

    # -- coercions between the two sides of an operator
    def rescale(mut self, e: Int, to: Int, ast: Int) raises -> Int:
        var x = self.at(e)
        if x.scale == to:
            return e
        var k = to - x.scale
        if k < 0:
            raise self.err(ast, "internal: a value cannot lose decimal places")
        if x.op == B_LIT:
            var m = pow10_checked(k)
            var v = x.value * m
            if x.value != 0 and v // x.value != m:
                raise self.err(ast, "the number " + x.text + " does not fit at " + String(to) + " decimal places")
            return self.mk(B_LIT, DECIMAL.value if to > 0 else INT.value, to, -1, -1, v, x.text, ast)
        return self.mk(B_RESCALE, DECIMAL.value, to, e, -1, Int64(k), "", ast)

    def string_as_date(mut self, e: Int, ast: Int) raises -> Int:
        var x = self.at(e)
        var days = parse_date_literal(x.text)
        if days == Int.MIN:
            raise self.err(ast, "cannot read '" + x.text + "' as a DATE: " + date_problem(x.text))
        return self.mk(B_LIT, DATE.value, 0, -1, -1, Int64(days), String(x.text), ast)

    def align(mut self, mut a: Int, mut b: Int, ast: Int) raises:
        """Gives two numeric operands the same decimal places (the shorter one is scaled up), as DuckDB does."""
        var x = self.at(a)
        var y = self.at(b)
        if x.kind == FLOAT.value or y.kind == FLOAT.value:
            return
        if x.scale < y.scale:
            a = self.rescale(a, y.scale, ast)
        elif y.scale < x.scale:
            b = self.rescale(b, x.scale, ast)


    def to_float(mut self, e: Int, ast: Int, real: Bool = False) raises -> Int:
        """The operand as a FLOAT. `real`: in a 32-bit context (DuckDB's REAL: the other side is a 32-bit float), so an INT or DECIMAL is
        cast to REAL: a literal is folded to a 32-bit float, anything else is converted and rounded to 32 bits. A FLOAT stays as it is
        (an exponent literal is a DOUBLE and is never re-rounded). After aggregation the evaluator reads INT and DECIMAL as doubles
        itself, so only a literal is folded there and a 32-bit context with anything else raises."""
        var x = self.at(e)
        if x.kind == FLOAT.value:
            return e
        if x.op == B_LIT:
            var f = decimal_to_f32(x.value, x.scale) if real else decimal_to_float(x.value, x.scale)
            return self.mk(B_LIT, FLOAT.value, 0, -1, -1, Int64(f.to_bits()), x.text, ast)
        if self.in_post:
            if real:
                raise self.err(ast, "a 32-bit float (a REAL column, or min / max of one) meeting an INT or DECIMAL value that is not a literal, after aggregation, is not supported: DuckDB does it in 32 bits")
            return e
        if real and x.scale > 10:
            raise self.err(ast, "a DECIMAL with more than 10 places meeting a 32-bit float is not supported")
        return self.mk(B_I2F, FLOAT.value, F32_COLUMN if real else 0, e, -1, Int64(x.scale), "", ast)

    def float_pair(mut self, mut a: Int, mut b: Int, ast: Int) raises -> Bool:
        """Both operands as FLOATs, in DuckDB's type for the pair: a 32-bit float (REAL: a Parquet FLOAT column, min / max of one, its
        negation or arithmetic) next to REAL, INT or DECIMAL is REAL (`l_discount <= 0.07` keeps the rows holding 0.07; `f = d` casts the
        DECIMAL to REAL); next to a DOUBLE (a double column, an exponent literal such as 7e-2) it is DOUBLE. Returns True for REAL."""
        var x = self.at(a)
        var y = self.at(b)
        var xr = x.kind == FLOAT.value and x.scale == F32_COLUMN
        var yr = y.kind == FLOAT.value and y.scale == F32_COLUMN
        var xd = x.kind == FLOAT.value and x.scale != F32_COLUMN
        var yd = y.kind == FLOAT.value and y.scale != F32_COLUMN
        var real = (xr or yr) and not xd and not yd
        a = self.to_float(a, ast, real)
        b = self.to_float(b, ast, real)
        return real

    def compare(mut self, code: Int, left: Int, right: Int, ast: Int) raises -> Int:
        var a = left
        var b = right
        var op = code
        var x = self.at(a)
        var y = self.at(b)
        var opname = self.st.exprs[ast].text
        if x.op == B_NULL or y.op == B_NULL:                 # anything compared with NULL is unknown
            return self.mk(B_NULL, BOOL.value, 0, -1, -1, 0, "", ast)
        # a string next to a DATE is that date
        if x.kind == DATE.value and y.op == B_STR:
            b = self.string_as_date(b, ast)
            y = self.at(b)
        elif y.kind == DATE.value and x.op == B_STR:
            a = self.string_as_date(a, ast)
            x = self.at(a)
        if x.kind == TEXT.value or y.kind == TEXT.value:
            if x.kind != TEXT.value or y.kind != TEXT.value:
                raise self.err(ast, "cannot compare " + kind_name(x.kind, x.scale) + " with " + kind_name(y.kind, y.scale) + " (`" + opname + "`)")
            if x.op == B_STR and y.op == B_STR:                # two constants: answered now (= and <> by bytes, the order ignoring case)
                var l = x.text
                var r = y.text
                var res = (l == r) if code == C_EQ else (l != r) if code == C_NE else _text_less(l, r) if code == C_LT else _text_less(r, l) if code == C_GT else \
                          (not _text_less(r, l)) if code == C_LE else (not _text_less(l, r))
                return self.mk(B_LIT, BOOL.value, 0, -1, -1, Int64(1 if res else 0), "", ast)
            if x.op == B_STR:                                  # the value goes first: `'a' < c` is `c > 'a'`
                var t = a
                a = b
                b = t
                op = C_GT if code == C_LT else C_GE if code == C_LE else C_LT if code == C_GT else C_LE if code == C_GE else code
        elif _is_numeric(x.kind) and _is_numeric(y.kind):
            self.align(a, b, ast)
        elif (x.kind == FLOAT.value and _is_num(y.kind)) or (y.kind == FLOAT.value and _is_num(x.kind)):
            _ = self.float_pair(a, b, ast)
        elif x.kind == y.kind and (x.kind == DATE.value or x.kind == BOOL.value):
            pass
        else:
            raise self.err(ast, "cannot compare " + kind_name(x.kind, x.scale) + " with " + kind_name(y.kind, y.scale) + " (`" + opname + "`)")
        return self.mk(B_CMP, BOOL.value, 0, a, b, Int64(op), "", ast)

    def numeric_result(self, x: BExpr, y: BExpr, ast: Int) raises -> Bool:
        """Both operands are numbers (INT, DECIMAL or, after aggregation, FLOAT)."""
        var ok = _is_num(x.kind) and _is_num(y.kind)
        if not ok and x.kind != K_NULL and y.kind != K_NULL:
            raise self.err(ast, "cannot apply `" + self.st.exprs[ast].text + "` to " + kind_name(x.kind, x.scale) + " and " + kind_name(y.kind, y.scale))
        return ok

    def arith(mut self, op: String, left: Int, right: Int, ast: Int) raises -> Int:
        var a = left
        var b = right
        var x = self.at(a)
        var y = self.at(b)
        if x.op == B_NULL or y.op == B_NULL:
            var other = y.copy() if x.op == B_NULL else x.copy()
            return self.mk(B_NULL, other.kind, other.scale, -1, -1, 0, "", ast)
        # DATE arithmetic: date +- days, date - date
        if x.kind == DATE.value or y.kind == DATE.value:
            if op == "-" and x.kind == DATE.value and y.kind == DATE.value:
                return self.mk(B_SUB, INT.value, 0, a, b, 0, "", ast)
            if (op == "+" and ((x.kind == DATE.value and y.kind == INT.value) or (x.kind == INT.value and y.kind == DATE.value))) or \
               (op == "-" and x.kind == DATE.value and y.kind == INT.value):
                return self.mk(B_ADD if op == "+" else B_SUB, DATE.value, 0, a, b, 0, "", ast)
            raise self.err(ast, "cannot apply `" + op + "` to " + kind_name(x.kind, x.scale) + " and " + kind_name(y.kind, y.scale))
        if not _is_num(x.kind) or not _is_num(y.kind):
            raise self.err(ast, "cannot apply `" + op + "` to " + kind_name(x.kind, x.scale) + " and " + kind_name(y.kind, y.scale))
        if op == "/":                                          # DuckDB's `/` is a division of doubles (of 32-bit floats next to REAL)
            var real_div = self.float_pair(a, b, ast)
            if real_div and self.in_post:
                raise self.err(ast, "arithmetic on 32-bit floats after aggregation is not supported: DuckDB does it in 32 bits")
            return self.mk(B_DIV, FLOAT.value, F32_COLUMN if real_div else 0, a, b, 0, "", ast)
        if op == "//" and (x.kind != INT.value or y.kind != INT.value):
            # DuckDB: `//` of a float or a decimal is a division of doubles (not floored), and a zero divisor is NULL
            var real_div = self.float_pair(a, b, ast)
            if real_div:
                raise self.err(ast, "`//` on a 32-bit float (REAL) is not supported: DuckDB does it in 32 bits")
            var div = self.mk(B_DIV, FLOAT.value, 0, a, b, 0, "", ast)
            var zero = self.mk(B_LIT, FLOAT.value, 0, -1, -1, Int64(Float64(0.0).to_bits()), "", ast)
            var is_zero = self.compare(C_EQ, b, zero, ast)
            var conds: List[Int] = [is_zero]
            var vals: List[Int] = [self.null_of(FLOAT.value, 0, ast)]
            return self.make_case(conds, vals, div, ast)
        if x.kind == FLOAT.value or y.kind == FLOAT.value:
            if op == "%":
                raise self.err(ast, "`%` on a FLOAT value is not supported: the remainder of doubles is C's fmod, which the kernel's floored remainder does not give exactly")
            var real_op = self.float_pair(a, b, ast)
            if real_op and self.in_post:
                raise self.err(ast, "arithmetic on 32-bit floats after aggregation is not supported: DuckDB does it in 32 bits")
            return self.mk(B_ADD if op == "+" else B_SUB if op == "-" else B_MUL, FLOAT.value, F32_COLUMN if real_op else 0, a, b, 0, "", ast)
        if op == "//":
            return self.mk(B_IDIV, INT.value, 0, a, b, 0, "", ast)
        if op == "*":
            var scale = x.scale + y.scale
            if scale > 18:
                raise self.err(ast, "the product has " + String(scale) + " decimal places; the most a value can hold is 18")
            if x.op == B_LIT and y.op == B_LIT:
                var v = x.value * y.value
                if (x.value != 0 and v // x.value != y.value) or (x.value == -1 and y.value == Int64.MIN):
                    raise self.err(ast, "the product passes the int64 range")
                return self.mk(B_LIT, DECIMAL.value if scale > 0 else INT.value, scale, -1, -1, v, "", ast)
            return self.mk(B_MUL, DECIMAL.value if scale > 0 else INT.value, scale, a, b, 0, "", ast)
        self.align(a, b, ast)
        x = self.at(a)
        y = self.at(b)
        var kind = DECIMAL.value if x.scale > 0 else INT.value
        if op == "%":
            return self.mk(B_IMOD, kind, x.scale, a, b, 0, "", ast)
        if x.op == B_LIT and y.op == B_LIT:
            var v = x.value + y.value if op == "+" else x.value - y.value
            var over = ((x.value ^ v) & (y.value ^ v)) < 0 if op == "+" else ((x.value ^ y.value) & (x.value ^ v)) < 0
            if over:
                raise self.err(ast, "the result passes the int64 range")
            return self.mk(B_LIT, kind, x.scale, -1, -1, v, "", ast)
        return self.mk(B_ADD if op == "+" else B_SUB, kind, x.scale, a, b, 0, "", ast)

    def boolean(mut self, e: Int, ast: Int, what: String) raises -> Int:
        var x = self.at(e)
        if x.op == B_NULL and x.kind == K_NULL:
            return self.mk(B_NULL, BOOL.value, 0, -1, -1, 0, "", ast)
        if x.kind != BOOL.value:
            raise self.err(ast, what + " needs a true / false value, found " + kind_name(x.kind, x.scale))
        return e

    def in_list(mut self, ast: Int) raises -> Int:
        var xi = self.st.exprs[ast].copy()
        var a = self.value(xi.a)
        var xa = self.at(a)
        if xa.op == B_NULL:
            return self.mk(B_NULL, BOOL.value, 0, -1, -1, 0, "", ast)
        if xa.op == B_STR:                                         # a constant against a list: answered now
            var found = False
            var unknown = False
            for it in xi.items:
                var lit = self.at(self.value(it))
                if lit.op == B_NULL:
                    unknown = True
                elif lit.op == B_STR and lit.text == xa.text:
                    found = True
            if found:
                return self.mk(B_LIT, BOOL.value, 0, -1, -1, 1, "", ast)
            if unknown:
                return self.mk(B_NULL, BOOL.value, 0, -1, -1, 0, "", ast)
            return self.mk(B_LIT, BOOL.value, 0, -1, -1, 0, "", ast)
        var lits = List[Int]()
        var has_null = False
        var scale = xa.scale
        var any_double = False                                    # one DOUBLE in the list makes the whole list DOUBLE, as DuckDB's common type
        for it in xi.items:
            any_double = any_double or self.st.exprs[it].kind == X_FLT
        for it in xi.items:
            var ik = self.st.exprs[it].kind
            if ik == X_COL or ik == X_CALL:
                raise self.err(it, "IN lists must hold literals, not column references or functions")
            var lit = self.value(it)
            var l = self.at(lit)
            if l.op == B_NULL:
                has_null = True
                continue
            if l.op != B_LIT and l.op != B_STR:
                raise self.err(it, "IN lists must hold literals")
            if xa.kind == DATE.value and l.op == B_STR:
                lit = self.string_as_date(lit, it)
                l = self.at(lit)
            if xa.kind == TEXT.value:
                if l.op != B_STR:
                    raise self.err(it, "cannot compare TEXT with " + kind_name(l.kind, l.scale) + " in an IN list")
            elif xa.kind == FLOAT.value and _is_num(l.kind):
                lit = self.to_float(lit, it, xa.scale == F32_COLUMN and not any_double)
            elif _is_numeric(xa.kind) and l.kind == FLOAT.value:
                raise self.err(it, "an IN list with a FLOAT value needs a FLOAT on the left: " + kind_name(xa.kind, xa.scale) + " IN (a float) compares as doubles")
            elif _is_numeric(xa.kind) and _is_numeric(l.kind):
                scale = max(scale, l.scale)
            elif xa.kind == l.kind and (xa.kind == DATE.value or xa.kind == BOOL.value):
                pass
            else:
                raise self.err(it, "cannot compare " + kind_name(xa.kind, xa.scale) + " with " + kind_name(l.kind, l.scale) + " in an IN list")
            lits.append(lit)
        if _is_numeric(xa.kind) and scale > xa.scale:
            a = self.rescale(a, scale, ast)
        if _is_numeric(xa.kind):
            for k in range(len(lits)):
                lits[k] = self.rescale(lits[k], scale, ast)
        var node = self.mk(B_IN, BOOL.value, 0, a, -1, Int64(1 if has_null else 0), "", ast)
        if self.in_post:
            self.q.pexprs[node].items = lits^
        else:
            self.q.rexprs[node].items = lits^
        return node

    # ── F-156: expressions (CASE, COALESCE, CAST, text and date functions) ─────────────────────────────────

    def set_items(mut self, e: Int, var items: List[Int]):
        if self.in_post:
            self.q.pexprs[e].items = items^
        else:
            self.q.rexprs[e].items = items^

    def set_src(mut self, e: Int, src: Int):
        if self.in_post:
            self.q.pexprs[e].src = src
        else:
            self.q.rexprs[e].src = src

    def retype(mut self, e: Int, kind: Int, scale: Int):
        if self.in_post:
            self.q.pexprs[e].kind = kind
            self.q.pexprs[e].scale = scale
        else:
            self.q.rexprs[e].kind = kind
            self.q.rexprs[e].scale = scale

    def mk_text(mut self, op: Int, a: Int, b: Int, value: Int64, text: String, ast: Int) -> Int:
        """A node whose value is a word of a dictionary of its own (a TEXT function's answer, a CASE over words): `src` says which."""
        var e = self.mk(op, TEXT.value, 0, a, b, value, text, ast)
        self.set_src(e, DERIVED_BASE + e)
        return e

    def lit_str(mut self, s: String, ast: Int) -> Int:
        return self.mk_text(B_STR, -1, -1, 0, s, ast)

    def lit_int(mut self, v: Int, ast: Int) -> Int:
        return self.mk(B_LIT, INT.value, 0, -1, -1, Int64(v), String(v), ast)

    def null_of(mut self, kind: Int, scale: Int, ast: Int) -> Int:
        var e = self.mk(B_NULL, kind, scale, -1, -1, 0, "", ast)
        if kind == TEXT.value:
            self.set_src(e, DERIVED_BASE + e)
        return e

    def common_type(mut self, mut ids: List[Int], ast: Int, what: String) raises -> Tuple[Int, Int]:
        """The type the values of a CASE or COALESCE share, as DuckDB gives it: a DECIMAL with the most decimal places, a FLOAT when one value is (an INT or DECIMAL
        is converted: 32-bit when every float is REAL), a string next to DATEs is a date. Converts `ids` in place; a NULL takes the type. Returns (kind, scale)."""
        var has_text = False
        var has_float = False
        var has_num = False
        var has_date = False
        var has_bool = False
        var any_real = False
        var any_double = False
        var max_scale = 0
        var any_value = False
        var all_str = True
        for id in ids:
            var x = self.at(id)
            if x.op == B_NULL:
                continue
            any_value = True
            if x.kind == TEXT.value:
                has_text = True
                all_str = all_str and x.op == B_STR
            elif x.kind == FLOAT.value:
                has_float = True
                if x.scale == F32_COLUMN:
                    any_real = True
                else:
                    any_double = True
            elif x.kind == INT.value or x.kind == DECIMAL.value:
                has_num = True
                max_scale = max(max_scale, x.scale)
            elif x.kind == DATE.value:
                has_date = True
            elif x.kind == BOOL.value:
                has_bool = True
        if not any_value:
            raise self.err(ast, what + ": every result is NULL, so its type cannot be known")
        if has_text and has_date and all_str:                       # a string next to a date is that date
            for k in range(len(ids)):
                if self.at(ids[k]).op == B_STR:
                    ids[k] = self.string_as_date(ids[k], ast)
            has_text = False
        var kinds = (1 if has_text else 0) + (1 if has_float or has_num else 0) + (1 if has_date else 0) + (1 if has_bool else 0)
        if kinds > 1:
            var names = List[String]()
            for id in ids:
                var x = self.at(id)
                if x.op != B_NULL:
                    var nm = kind_name(x.kind, x.scale)
                    var seen = False
                    for s in names:
                        seen = seen or s == nm
                    if not seen:
                        names.append(nm)
            raise self.err(ast, what + " mixes " + " and ".join(names) + ": cast them to one type")
        if has_float:
            var real = any_real and not any_double
            for k in range(len(ids)):
                var x = self.at(ids[k])
                if x.op == B_NULL:
                    self.retype(ids[k], FLOAT.value, F32_COLUMN if real else 0)
                elif x.kind != FLOAT.value:
                    ids[k] = self.to_float(ids[k], ast, real)
            return (FLOAT.value, F32_COLUMN if real else 0)
        if has_num:
            for k in range(len(ids)):
                var x = self.at(ids[k])
                if x.op == B_NULL:
                    self.retype(ids[k], DECIMAL.value if max_scale > 0 else INT.value, max_scale)
                else:
                    ids[k] = self.rescale(ids[k], max_scale, ast)
            return (DECIMAL.value if max_scale > 0 else INT.value, max_scale)
        var kind = TEXT.value if has_text else DATE.value if has_date else BOOL.value
        for k in range(len(ids)):
            if self.at(ids[k]).op == B_NULL:
                self.retype(ids[k], kind, 0)
                if kind == TEXT.value:
                    self.set_src(ids[k], DERIVED_BASE + ids[k])
        return (kind, 0)

    def make_case(mut self, conds: List[Int], vals: List[Int], else_: Int, ast: Int) raises -> Int:
        var all_vals = vals.copy()
        all_vals.append(else_)
        var t = self.common_type(all_vals, ast, "CASE")
        var items = List[Int]()
        for k in range(len(conds)):
            items.append(conds[k])
            items.append(all_vals[k])
        items.append(all_vals[len(conds)])
        var e: Int
        if t[0] == TEXT.value:
            e = self.mk_text(B_CASE, -1, -1, Int64(len(conds)), "", ast)
        else:
            e = self.mk(B_CASE, t[0], t[1], -1, -1, Int64(len(conds)), "", ast)
        self.set_items(e, items^)
        return e

    def case_expr(mut self, ast: Int) raises -> Int:
        var x = self.st.exprs[ast].copy()
        var operand = -1
        if x.a >= 0:
            operand = self.value(x.a)
        var conds = List[Int]()
        var vals = List[Int]()
        var k = 0
        while k + 1 < len(x.items):
            var w = self.value(x.items[k])
            var cond: Int
            if operand >= 0:
                cond = self.compare(C_EQ, operand, w, x.items[k])
            else:
                cond = self.boolean(w, x.items[k], "CASE WHEN")
            conds.append(cond)
            vals.append(self.value(x.items[k + 1]))
            k += 2
        var else_ = self.value(x.c) if x.c >= 0 else self.null_of(K_NULL, 0, ast)
        return self.make_case(conds, vals, else_, ast)

    def coalesce_expr(mut self, ast: Int) raises -> Int:
        var x = self.st.exprs[ast].copy()
        if len(x.items) == 0:
            raise self.err(ast, x.text.lower() + "() needs at least one argument")
        var ids = List[Int]()
        for it in x.items:
            ids.append(self.value(it))
        if len(ids) == 1:
            return ids[0]
        var t = self.common_type(ids, ast, x.text.lower() + "()")
        var e: Int
        if t[0] == TEXT.value:
            e = self.mk_text(B_COALESCE, -1, -1, 0, "", ast)
        else:
            e = self.mk(B_COALESCE, t[0], t[1], -1, -1, 0, "", ast)
        self.set_items(e, ids^)
        return e

    def nullif_expr(mut self, ast: Int) raises -> Int:
        var x = self.st.exprs[ast].copy()
        if len(x.items) != 2:
            raise self.err(ast, "nullif() takes two arguments, found " + String(len(x.items)))
        var a = self.value(x.items[0])
        var b = self.value(x.items[1])
        if self.at(a).op == B_NULL or self.at(b).op == B_NULL:
            return a
        var cond = self.compare(C_EQ, a, b, ast)
        var xa = self.at(a)
        var nul = self.null_of(xa.kind, xa.scale, ast)
        var items: List[Int] = [cond, nul, a]
        var e: Int
        if xa.kind == TEXT.value:
            e = self.mk_text(B_CASE, -1, -1, 1, "", ast)
        else:
            e = self.mk(B_CASE, xa.kind, xa.scale, -1, -1, 1, "", ast)
        self.set_items(e, items^)
        return e

    def distinct_expr(mut self, ast: Int) raises -> Int:
        """a IS [NOT] DISTINCT FROM b: a comparison that treats NULL as a value (never NULL itself)."""
        var x = self.st.exprs[ast].copy()
        var a = self.value(x.a)
        var b = self.value(x.b)
        var xa = self.at(a)
        var xb = self.at(b)
        if xa.op == B_NULL and xb.op == B_NULL:
            return self.mk(B_LIT, BOOL.value, 0, -1, -1, Int64(1 if x.flag else 0), "", ast)
        if xa.op == B_NULL or xb.op == B_NULL:                      # x IS DISTINCT FROM NULL is x IS NOT NULL
            var other = b if xa.op == B_NULL else a
            var n = self.mk(B_ISNULL, BOOL.value, 0, other, -1, 0, "", ast)
            if x.flag:
                return n
            return self.mk(B_NOT, BOOL.value, 0, n, -1, 0, "", ast)
        var cmp = self.compare(C_EQ, a, b, ast)
        var c = self.at(cmp)
        if c.op == B_LIT:                                           # two constants
            var same = c.value != 0
            return self.mk(B_LIT, BOOL.value, 0, -1, -1, Int64(1 if same == x.flag else 0), "", ast)
        return self.mk(B_DISTINCT, BOOL.value, 0, c.a, c.b, Int64(1 if x.flag else 0), "", ast)

    def isbool_expr(mut self, ast: Int) raises -> Int:
        var x = self.st.exprs[ast].copy()
        var a = self.boolean(self.value(x.a), x.a, "IS " + x.text)
        if self.at(a).op == B_NULL:                                 # NULL IS TRUE is false, NULL IS NOT TRUE is true
            var t = x.flag
            return self.mk(B_LIT, BOOL.value, 0, -1, -1, Int64(1 if t else 0), "", ast)
        var code = (0 if x.text == "TRUE" else 2) + (1 if x.flag else 0)
        return self.mk(B_BOOLIS, BOOL.value, 0, a, -1, Int64(code), "", ast)

    # -- CAST
    def cast_expr(mut self, ast: Int) raises -> Int:
        var x = self.st.exprs[ast].copy()
        var a = self.value(x.a)
        var ax = self.at(a)
        var kind = INT.value
        var scale = 0
        var bits = 64
        var prec = 18
        var tname = x.text
        if tname == "INTEGER":
            bits = 32
        elif tname == "SMALLINT":
            bits = 16
        elif tname == "TINYINT":
            bits = 8
        elif tname == "DECIMAL":
            kind = DECIMAL.value
            prec = x.b if x.b >= 0 else 18
            scale = x.c if x.c >= 0 else (3 if x.b < 0 else 0)
            if prec < 1 or prec > 18:
                raise self.err(ast, "DECIMAL(" + String(prec) + ", " + String(scale) + "): omxcore stores a decimal in an int64, so the precision is 1 to 18")
            if scale > prec:
                raise self.err(ast, "DECIMAL(" + String(prec) + ", " + String(scale) + "): the scale is more than the precision")
        elif tname == "DOUBLE":
            kind = FLOAT.value
        elif tname == "REAL":
            kind = FLOAT.value
            scale = F32_COLUMN
        elif tname == "VARCHAR":
            kind = TEXT.value
        elif tname == "DATE":
            kind = DATE.value
        elif tname == "BOOLEAN":
            kind = BOOL.value
        if ax.op == B_NULL:
            return self.null_of(kind, scale, ast)
        if ax.op == B_STR and kind != TEXT.value:                    # a string constant is read now
            var pr = parse_text(ax.text, kind, scale, bits, prec)
            if pr.ok:
                return self.mk(B_LIT, kind, scale, -1, -1, pr.value, "", ast)
            if x.flag:
                return self.null_of(kind, scale, ast)
            # not folded: the cast stays in the program, where the word is an error for a row that reaches it (a branch no row takes never raises)
        var fk = ax.kind
        var ok = False
        if fk == TEXT.value:
            ok = True
        elif fk == INT.value or fk == DECIMAL.value:
            ok = kind == INT.value or kind == DECIMAL.value or kind == FLOAT.value or kind == BOOL.value or kind == TEXT.value
        elif fk == FLOAT.value:
            ok = kind == INT.value or kind == DECIMAL.value or kind == FLOAT.value or kind == BOOL.value
        elif fk == DATE.value:
            ok = kind == DATE.value or kind == TEXT.value
        elif fk == BOOL.value:
            ok = kind == BOOL.value or kind == INT.value or kind == DECIMAL.value or kind == FLOAT.value or kind == TEXT.value
        if not ok:
            raise self.err(ast, "cannot cast " + kind_name(fk, ax.scale) + " to " + tname + (" (DuckDB refuses it too)" if fk == DATE.value or kind == DATE.value else
                           (": a float's text is its shortest form per row, which omxcore does not make" if fk == FLOAT.value and kind == TEXT.value else "")))
        if kind == TEXT.value and fk != TEXT.value and fk != BOOL.value and ax.op != B_COL:
            raise self.err(ast, "CAST(... AS VARCHAR) of " + kind_name(fk, ax.scale) + " needs a column, not an expression: omxcore writes the text of every value between the column's lowest and highest once, and knows that range for a column only")
        if fk == kind and kind != DECIMAL.value and (kind != INT.value or bits >= 64) and (kind != FLOAT.value):      # the same type: nothing to do
            return a
        if fk == BOOL.value and kind == BOOL.value:
            return a
        if (fk == INT.value or fk == DECIMAL.value or fk == BOOL.value) and kind == FLOAT.value:
            if fk == BOOL.value and ax.op != B_COL and ax.op != B_LIT:
                pass
            else:
                return self.mk(B_I2F, FLOAT.value, scale, a, -1, Int64(ax.scale if fk != BOOL.value else 0), "", ast)
        if fk == BOOL.value and ax.op != B_COL and ax.op != B_LIT:     # a condition as a number or text: CASE WHEN b THEN 1 WHEN NOT b THEN 0 END
            var one: Int
            var zero: Int
            if kind == TEXT.value:
                one = self.lit_str("true", ast)
                zero = self.lit_str("false", ast)
            elif kind == FLOAT.value:
                one = self.mk(B_LIT, FLOAT.value, scale, -1, -1, Int64(Float64(1.0).to_bits()), "", ast)
                zero = self.mk(B_LIT, FLOAT.value, scale, -1, -1, Int64(Float64(0.0).to_bits()), "", ast)
            else:
                var m = pow10_checked(scale)
                one = self.mk(B_LIT, kind, scale, -1, -1, m, "", ast)
                zero = self.mk(B_LIT, kind, scale, -1, -1, 0, "", ast)
            var neg = self.mk(B_NOT, BOOL.value, 0, a, -1, 0, "", ast)
            var conds: List[Int] = [a, neg]
            var vals: List[Int] = [one, zero]
            return self.make_case(conds, vals, self.null_of(kind, scale, ast), ast)
        var value = (1 if x.flag else 0) | (bits << 1) | (prec << 8)
        var text = self.st.render(ast)
        if kind == TEXT.value:
            return self.mk_text(B_CAST, a, -1, Int64(value), text, ast)
        return self.mk(B_CAST, kind, scale, a, -1, Int64(value), text, ast)

    # -- text
    def const_text(mut self, arg: Int, what: String) raises -> String:
        var x = self.at(self.value(arg))
        if x.op != B_STR:
            raise self.err(arg, what + " must be a string constant")
        return x.text

    def const_int(mut self, arg: Int, what: String) raises -> Int:
        var x = self.at(self.value(arg))
        if x.op != B_LIT or x.kind != INT.value:
            raise self.err(arg, what + " must be a whole number constant")
        return Int(x.value)

    def text_operand(mut self, arg: Int, what: String) raises -> Int:
        var e = self.value(arg)
        var x = self.at(e)
        if x.op != B_NULL and x.kind != TEXT.value:
            raise self.err(arg, what + " needs TEXT, found " + kind_name(x.kind, x.scale) + " (CAST it AS VARCHAR first)")
        return e

    def fold_cases(mut self, s: String, up: Bool) raises -> String:
        var ascii = True
        for b in s.as_bytes():
            if b >= UInt8(0x80):
                ascii = False
        if not ascii and not self.cases:
            self.cases = CaseTables()
        if ascii:
            return ascii_case(s, up)
        return self.cases.value().map(s, up)

    def text_function(mut self, ast: Int, func: Int, flags: Int, lo: Int, hi: Int, name: String) raises -> Int:
        """A function of one TEXT value and constants, evaluated once per word of the dictionary the value comes from (or at once when the value is a constant)."""
        var x = self.st.exprs[ast].copy()
        var n = len(x.items) - 1
        if len(x.items) == 0 or n < lo or n > hi:
            raise self.err(ast, name + "() takes " + (String(lo + 1) if lo == hi else String(lo + 1) + " or " + String(hi + 1)) + " argument" + ("s" if hi > 0 else "") + ", found " + String(len(x.items)))
        var a = self.text_operand(x.items[0], name + "()")
        var strs = List[String]()
        var ints = List[Int]()
        var has_len = False
        var args = List[Int]()
        for k in range(1, len(x.items)):
            var ak = x.items[k]
            var wants_int = func == TF_SUBSTR or func == TF_LEFT or func == TF_RIGHT
            if wants_int:
                var v = self.const_int(ak, "the argument " + String(k + 1) + " of " + name + "()")
                ints.append(v)
                args.append(self.lit_int(v, ak))
            else:
                var s = self.const_text(ak, "the argument " + String(k + 1) + " of " + name + "()")
                strs.append(s)
                args.append(self.lit_str(s, ak))
        var ax = self.at(a)
        var result_kind = tf_kind(func)
        if ax.op == B_NULL:
            return self.null_of(result_kind, 0, ast)
        var t = TFn(func, flags, strs, ints)
        if ax.op == B_STR:                                           # all constants: answered now
            if result_kind == TEXT.value:
                return self.lit_str(self.tf_text(t, ax.text), ast)
            if result_kind == INT.value:
                return self.lit_int(t.to_int(ax.text), ast)
            return self.mk(B_LIT, BOOL.value, 0, -1, -1, Int64(1 if t.to_bool(ax.text) else 0), "", ast)
        var e: Int
        if result_kind == TEXT.value:
            e = self.mk_text(B_TFN, a, -1, Int64(func | (flags << 8)), "", ast)
        else:
            e = self.mk(B_TFN, result_kind, 0, a, -1, Int64(func | (flags << 8)), "", ast)
        self.set_items(e, args^)
        return e

    def tf_text(mut self, t: TFn, w: String) raises -> String:
        if t.func == TF_LOWER or t.func == TF_UPPER:
            return self.fold_cases(w, t.func == TF_UPPER)
        return t.to_text(w)

    def like_expr(mut self, ast: Int) raises -> Int:
        var x = self.st.exprs[ast].copy()
        var a = self.text_operand(x.a, x.text + "'s left side")
        var pe = self.value(x.b)
        var px = self.at(pe)
        var esc = String()
        var has_esc = False
        if x.c >= 0:
            esc = self.const_text(x.c, "the ESCAPE of " + x.text)
            has_esc = esc.byte_length() > 0
            if char_count(esc) > 1:
                raise self.err(x.c, "the ESCAPE of " + x.text + " is one character, found '" + esc + "'")
            if esc == "%" or esc == "_":
                raise self.err(x.c, "the ESCAPE character cannot be % or _")
        if px.op == B_NULL or self.at(a).op == B_NULL:
            return self.null_of(BOOL.value, 0, ast)
        if px.op != B_STR:
            raise self.err(x.b, "the pattern of " + x.text + " must be a string constant")
        var pattern = px.text
        var ilike = x.text == "ILIKE"
        if ilike:
            pattern = self.fold_cases(pattern, False)
            if has_esc:
                esc = self.fold_cases(esc, False)
        var strs = List[String]()
        strs.append(pattern)
        strs.append(esc if has_esc else "")
        var t = TFn(TF_LIKE, (1 if ilike else 0) | (2 if has_esc else 0), strs, List[Int]())
        var ax = self.at(a)
        var node: Int
        if ax.op == B_STR:
            var w = self.fold_cases(ax.text, False) if ilike else ax.text
            node = self.mk(B_LIT, BOOL.value, 0, -1, -1, Int64(1 if t.to_bool(w) else 0), "", ast)
        else:
            node = self.mk(B_TFN, BOOL.value, 0, a, -1, Int64(TF_LIKE | (t.flags << 8)), "", ast)
            var args = List[Int]()
            args.append(self.lit_str(pattern, x.b))
            args.append(self.lit_str(esc if has_esc else "", x.b))
            self.set_items(node, args^)
        if x.flag:
            if self.at(node).op == B_LIT:
                return self.mk(B_LIT, BOOL.value, 0, -1, -1, Int64(1 if self.at(node).value == 0 else 0), "", ast)
            return self.mk(B_NOT, BOOL.value, 0, node, -1, 0, "", ast)
        return node

    def concat_operand(mut self, arg: Int, what: String) raises -> Int:
        """A value of `||` / concat(): TEXT, or a constant number, date or boolean read as the text DuckDB would make of it."""
        var e = self.value(arg)
        var x = self.at(e)
        if x.op == B_NULL or x.kind == TEXT.value:
            return e
        if x.op == B_LIT:
            if x.kind == INT.value or x.kind == DECIMAL.value:
                return self.lit_str(decimal_text(x.value, x.scale), arg)
            if x.kind == DATE.value:
                return self.lit_str(date_text(Int(x.value)), arg)
            if x.kind == BOOL.value:
                return self.lit_str("true" if x.value != 0 else "false", arg)
        raise self.err(arg, what + " needs TEXT, found " + kind_name(x.kind, x.scale) + " (CAST it AS VARCHAR first)")

    def concat_values(mut self, ids: List[Int], ast: Int, null_skips: Bool) raises -> Int:
        """a || b || ...: NULL when a value is NULL; concat(a, b, ...) reads a NULL as the empty string."""
        var acc = -1
        for id in ids:
            var x = self.at(id)
            if x.op == B_NULL:
                if null_skips:
                    continue
                return self.null_of(TEXT.value, 0, ast)
            if acc < 0:
                acc = id
                continue
            var xa = self.at(acc)
            if xa.op == B_STR and x.op == B_STR:
                acc = self.lit_str(xa.text + x.text, ast)
            else:
                acc = self.mk_text(B_CONCAT, acc, id, Int64(1 if null_skips else 0), "", ast)
        if acc < 0:
            return self.lit_str("", ast)
        if null_skips and self.at(acc).op != B_STR:                  # concat() of one value: a NULL is ''
            var only = acc
            if self.at(only).op != B_CONCAT:
                acc = self.mk_text(B_CONCAT, only, self.lit_str("", ast), Int64(1), "", ast)
        return acc

    # -- dates
    def date_operand(mut self, arg: Int, what: String) raises -> Int:
        var e = self.value(arg)
        var x = self.at(e)
        if x.op == B_STR:
            return self.string_as_date(e, arg)
        if x.op != B_NULL and x.kind != DATE.value:
            raise self.err(arg, what + " needs a DATE, found " + kind_name(x.kind, x.scale))
        return e

    def part_arg(mut self, arg: Int, what: String, allowed: String) raises -> Int:
        var name = lower_ascii(self.const_text(arg, "the date part of " + what))
        var code = part_code(name)
        if code < 0 or (" " + allowed + " ").find(" " + name + " ") < 0:
            raise self.err(arg, "`" + name + "` is not a date part " + what + " takes (" + allowed.replace(" ", ", ") + ")")
        return code

    def date_part_expr(mut self, ast: Int, part: Int, date_arg: Int, what: String) raises -> Int:
        var d = self.date_operand(date_arg, what)
        var dx = self.at(d)
        if dx.op == B_NULL:
            return self.null_of(INT.value, 0, ast)
        if part == DP_EPOCH:                                         # DuckDB's epoch is a DOUBLE
            if dx.op == B_LIT:
                return self.mk(B_LIT, FLOAT.value, 0, -1, -1, Int64(Float64(date_part(Int(dx.value), part)).to_bits()), "", ast)
            return self.mk(B_DATEPART, FLOAT.value, 0, d, -1, Int64(part), "", ast)
        if dx.op == B_LIT:
            return self.lit_int(date_part(Int(dx.value), part), ast)
        return self.mk(B_DATEPART, INT.value, 0, d, -1, Int64(part), "", ast)

    def scalar_call(mut self, ast: Int) raises -> Int:
        var x = self.st.exprs[ast].copy()
        var name = x.text
        var n = len(x.items)
        if name == "COALESCE" or name == "IFNULL" or name == "NVL":
            if name != "COALESCE" and n != 2:
                raise self.err(ast, name.lower() + "() takes two arguments, found " + String(n))
            return self.coalesce_expr(ast)
        if name == "NULLIF":
            return self.nullif_expr(ast)
        if name == "LOWER" or name == "LCASE":
            return self.text_function(ast, TF_LOWER, 0, 0, 0, "lower")
        if name == "UPPER" or name == "UCASE":
            return self.text_function(ast, TF_UPPER, 0, 0, 0, "upper")
        if name == "TRIM":
            return self.text_function(ast, TF_TRIM, 3, 0, 1, "trim")
        if name == "LTRIM":
            return self.text_function(ast, TF_TRIM, 1, 0, 1, "ltrim")
        if name == "RTRIM":
            return self.text_function(ast, TF_TRIM, 2, 0, 1, "rtrim")
        if name == "SUBSTR" or name == "SUBSTRING":
            return self.text_function(ast, TF_SUBSTR, 0, 1, 2, name.lower())
        if name == "LENGTH" or name == "LEN" or name == "CHAR_LENGTH" or name == "CHARACTER_LENGTH":
            return self.text_function(ast, TF_LENGTH, 0, 0, 0, name.lower())
        if name == "STARTS_WITH" or name == "PREFIX":
            return self.text_function(ast, TF_STARTS, 0, 1, 1, name.lower())
        if name == "ENDS_WITH" or name == "SUFFIX":
            return self.text_function(ast, TF_ENDS, 0, 1, 1, name.lower())
        if name == "CONTAINS":
            return self.text_function(ast, TF_CONTAINS, 0, 1, 1, "contains")
        if name == "STRPOS" or name == "INSTR":
            return self.text_function(ast, TF_STRPOS, 0, 1, 1, name.lower())
        if name == "LEFT":
            return self.text_function(ast, TF_LEFT, 0, 1, 1, "left")
        if name == "RIGHT":
            return self.text_function(ast, TF_RIGHT, 0, 1, 1, "right")
        if name == "REVERSE":
            return self.text_function(ast, TF_REVERSE, 0, 0, 0, "reverse")
        if name == "REPLACE":
            return self.text_function(ast, TF_REPLACE, 0, 2, 2, "replace")
        if name == "CONCAT":
            if n == 0:
                raise self.err(ast, "concat() needs at least one argument")
            var ids = List[Int]()
            for it in x.items:
                ids.append(self.concat_operand(it, "concat()"))
            return self.concat_values(ids, ast, True)
        if name == "EXTRACT" or name == "DATE_PART" or name == "DATEPART":
            if n != 2:
                raise self.err(ast, name.lower() + "() takes a part and a date, found " + String(n) + " argument" + ("" if n == 1 else "s"))
            var part = self.part_arg(x.items[0], name.lower(), "year month day dow dayofweek isodow doy dayofyear quarter epoch decade century millennium week weekofyear isoyear")
            return self.date_part_expr(ast, part, x.items[1], name.lower())
        if name == "YEAR" or name == "MONTH" or name == "DAY" or name == "QUARTER" or name == "DAYOFWEEK" or name == "ISODOW" or name == "DAYOFYEAR" or name == "DECADE" or name == "WEEK" or name == "WEEKOFYEAR" or name == "ISOYEAR":
            if n != 1:
                raise self.err(ast, name.lower() + "() takes one date, found " + String(n) + " arguments")
            return self.date_part_expr(ast, part_code(name.lower()), x.items[0], name.lower())
        if name == "QUANTILE_BIN":
            if n != 2:
                raise self.err(ast, "quantile_bin() takes a value and the number of bins, found " + String(n) + " argument" + ("" if n == 1 else "s"))
            if self.in_post:
                raise self.err(ast, "quantile_bin() bins the rows of a table: it cannot take an aggregate (bin the column, then aggregate per bin)")
            var xv = self.value(x.items[0])
            var xe = self.at(xv)
            if xe.op == B_NULL or xe.op == B_LIT:
                raise self.err(x.items[0], "quantile_bin() needs a column or an expression of one, not a constant")
            if xe.kind != INT.value and xe.kind != DECIMAL.value and xe.kind != FLOAT.value and xe.kind != DATE.value:
                raise self.err(x.items[0], "quantile_bin() needs numbers or dates, found " + kind_name(xe.kind, xe.scale))
            var ne = self.at(self.value(x.items[1]))
            if ne.op != B_LIT or ne.kind != INT.value or ne.value < 2 or ne.value > MAX_QUANTILE_BINS:
                raise self.err(x.items[1], "the number of bins of quantile_bin() is a whole number from 2 to " + String(MAX_QUANTILE_BINS))
            return self.mk(B_QBIN, INT.value, 0, xv, -1, ne.value, self.st.render(x.items[0]), ast)
        if name == "DATE_TRUNC" or name == "DATETRUNC":
            if n != 2:
                raise self.err(ast, "date_trunc() takes a part and a date, found " + String(n) + " argument" + ("" if n == 1 else "s"))
            var part = self.part_arg(x.items[0], "date_trunc", "year quarter month week day decade century millennium")
            var d = self.date_operand(x.items[1], "date_trunc()")
            var dx = self.at(d)
            if dx.op == B_NULL:
                return self.null_of(DATE.value, 0, ast)
            if dx.op == B_LIT:
                return self.mk(B_LIT, DATE.value, 0, -1, -1, Int64(date_trunc_days(Int(dx.value), part)), "", ast)
            if part == DP_DAY:
                return d
            return self.mk(B_DATETRUNC, DATE.value, 0, d, -1, Int64(part), "", ast)
        if name == "DATEDIFF" or name == "DATE_DIFF":
            if n != 3:
                raise self.err(ast, name.lower() + "() takes a part and two dates, found " + String(n) + " arguments")
            var part = self.part_arg(x.items[0], name.lower(), "year quarter month week day decade century millennium")
            var a = self.date_operand(x.items[1], name.lower() + "()")
            var b = self.date_operand(x.items[2], name.lower() + "()")
            var ax = self.at(a)
            var bx = self.at(b)
            if ax.op == B_NULL or bx.op == B_NULL:
                return self.null_of(INT.value, 0, ast)
            if ax.op == B_LIT and bx.op == B_LIT:
                return self.lit_int(date_diff_days(Int(ax.value), Int(bx.value), part), ast)
            return self.mk(B_DATEDIFF, INT.value, 0, a, b, Int64(part), "", ast)
        raise self.unknown_function(ast)

    def date_interval(mut self, ast: Int) raises -> Int:
        """`date + INTERVAL n unit` and `date - INTERVAL n unit` (and `INTERVAL n unit + date`): days and weeks add to the day number; months and years move
        the month and clamp the day to the end of the target month, as DuckDB does (DuckDB's result is a TIMESTAMP at midnight; here it is the DATE)."""
        var x = self.st.exprs[ast].copy()
        var iv_ast = x.b if self.st.exprs[x.b].kind == X_INTERVAL else x.a
        var d_ast = x.a if iv_ast == x.b else x.b
        if iv_ast == x.a and x.text == "-":
            raise self.err(ast, "an interval cannot be subtracted from: write date - INTERVAL ...")
        if self.st.exprs[d_ast].kind == X_INTERVAL:
            raise self.err(ast, "two intervals cannot be added: add them to a date one after the other")
        var iv = self.st.exprs[iv_ast].copy()
        var d = self.date_operand(d_ast, "an interval's date")
        var dx = self.at(d)
        var amount = self.value(iv.a)
        var ax = self.at(amount)
        if ax.op != B_NULL and ax.kind != INT.value:
            raise self.err(iv.a, "the amount of an INTERVAL is a whole number, found " + kind_name(ax.kind, ax.scale))
        if dx.op == B_NULL or ax.op == B_NULL:
            return self.null_of(DATE.value, 0, ast)
        var neg = x.text == "-"
        var unit = iv.text
        if unit == "DAY" or unit == "WEEK":
            if unit == "WEEK":
                amount = self.arith("*", amount, self.lit_int(7, iv.a), iv.a)
                ax = self.at(amount)
            if dx.op == B_LIT and ax.op == B_LIT:
                var days = Int(dx.value) + (-1 if neg else 1) * Int(ax.value)
                return self.mk(B_LIT, DATE.value, 0, -1, -1, Int64(days), "", ast)
            return self.mk(B_SUB if neg else B_ADD, DATE.value, 0, d, amount, 0, "", ast)
        if unit == "YEAR":
            amount = self.arith("*", amount, self.lit_int(12, iv.a), iv.a)
            ax = self.at(amount)
        if neg:
            amount = self.mk(B_NEG, INT.value, 0, amount, -1, 0, "", ast) if ax.op != B_LIT else self.lit_int(-Int(ax.value), ast)
            ax = self.at(amount)
        if dx.op == B_LIT and ax.op == B_LIT:
            return self.mk(B_LIT, DATE.value, 0, -1, -1, Int64(add_months(Int(dx.value), Int(ax.value))), "", ast)
        return self.mk(B_ADDMONTHS, DATE.value, 0, d, amount, 0, "", ast)

    def value(mut self, ast: Int) raises -> Int:
        var x = self.st.exprs[ast].copy()
        if self.in_post:
            var cn = self.canon(ast)
            for k in range(len(self.key_canons)):
                if self.key_canons[k] == cn:
                    var key = self.q.keys[k].copy()
                    var ke = self.q.rexprs[key.expr].copy()
                    var n = self.mk(B_KEY, ke.kind, ke.scale, -1, -1, Int64(k), key.name, ast)
                    self.q.pexprs[n].src = ke.src
                    return n
        var kind = x.kind
        if kind == X_INT or kind == X_DEC or kind == X_FLT or kind == X_STR or kind == X_NULL or kind == X_TRUE or kind == X_FALSE or kind == X_DATE:
            return self.literal(ast)
        if kind == X_COL:
            if self.in_post:
                # a select item's alias (HAVING, ORDER BY); a plain column is only allowed as a group key
                for k in range(len(self.st.items)):
                    if self.st.items[k].has_alias and lower_ascii(self.st.items[k].alias) == lower_ascii(x.text) and x.text2.byte_length() == 0 \
                       and self.alias_expr[k] >= 0:
                        return self.alias_expr[k]
                var c = self.find_column(ast)
                if self.distinct:
                    raise self.err(ast, "for SELECT DISTINCT, ORDER BY expressions must appear in the select list: `" + self.src_names[c] + "` does not")
                raise self.err(ast, "column `" + self.src_names[c] + "` must appear in the GROUP BY clause or be used in an aggregate function")
            var c = self.find_column(ast)
            if c >= self.n_stored:
                return self.formula_column(c - self.n_stored, ast, 0)
            var n = self.mk(B_COL, self.src_kinds[c], self.src_scales[c], -1, -1, Int64(c), String(self.src_names[c]), ast)
            if self.src_kinds[c] == TEXT.value:
                self.q.rexprs[n].src = c
            return n
        if kind == X_NEG:
            var a = self.value(x.a)
            var xa = self.at(a)
            if xa.op == B_NULL:
                return a
            if not _is_num(xa.kind):
                raise self.err(ast, "cannot negate " + kind_name(xa.kind, xa.scale))
            if xa.op == B_LIT and xa.kind == FLOAT.value:
                return self.mk(B_LIT, FLOAT.value, xa.scale, -1, -1, Int64((-Float64(from_bits=UInt64(xa.value))).to_bits()), "-" + xa.text, ast)
            if xa.op == B_LIT:
                if xa.value == Int64.MIN:
                    raise self.err(ast, "the number passes the int64 range")
                return self.mk(B_LIT, xa.kind, xa.scale, -1, -1, -xa.value, "", ast)
            return self.mk(B_NEG, xa.kind, xa.scale, a, -1, 0, "", ast)
        if kind == X_CASE:
            return self.case_expr(ast)
        if kind == X_CAST:
            return self.cast_expr(ast)
        if kind == X_LIKE:
            return self.like_expr(ast)
        if kind == X_DISTINCT:
            return self.distinct_expr(ast)
        if kind == X_ISBOOL:
            return self.isbool_expr(ast)
        if kind == X_INTERVAL:
            raise self.err(ast, "an INTERVAL can only be added to or subtracted from a DATE (date + INTERVAL 3 MONTH)")
        if kind == X_BIN:
            var op = x.text
            if (op == "+" or op == "-") and (self.st.exprs[x.a].kind == X_INTERVAL or self.st.exprs[x.b].kind == X_INTERVAL):
                return self.date_interval(ast)
            if op == "||":
                var parts = List[Int]()
                parts.append(self.concat_operand(x.a, "`||`"))
                parts.append(self.concat_operand(x.b, "`||`"))
                return self.concat_values(parts, ast, False)
            if op == "AND" or op == "OR":
                var l = self.boolean(self.value(x.a), x.a, "`" + op + "`")
                var r = self.boolean(self.value(x.b), x.b, "`" + op + "`")
                return self.mk(B_AND if op == "AND" else B_OR, BOOL.value, 0, l, r, 0, "", ast)
            var a = self.value(x.a)
            var b = self.value(x.b)
            if op == "=" or op == "<>" or op == "<" or op == "<=" or op == ">" or op == ">=":
                return self.compare(C_EQ if op == "=" else C_NE if op == "<>" else C_LT if op == "<" else C_LE if op == "<=" else C_GT if op == ">" else C_GE,
                                    a, b, ast)
            return self.arith(op, a, b, ast)
        if kind == X_NOT:
            var a = self.boolean(self.value(x.a), x.a, "NOT")
            if self.at(a).op == B_NULL:
                return a
            return self.mk(B_NOT, BOOL.value, 0, a, -1, 0, "", ast)
        if kind == X_ISNULL:
            var a = self.value(x.a)
            var n = self.mk(B_ISNULL, BOOL.value, 0, a, -1, 0, "", ast)
            if x.flag:
                return self.mk(B_NOT, BOOL.value, 0, n, -1, 0, "", ast)
            return n
        if kind == X_BETWEEN:
            var v = self.value(x.a)
            var lo = self.value(x.b)
            var hi = self.value(x.c)
            var c1 = self.compare(C_GE, v, lo, ast)
            var c2 = self.compare(C_LE, v, hi, ast)
            var both = self.mk(B_AND, BOOL.value, 0, c1, c2, 0, "", ast)
            if x.flag:
                return self.mk(B_NOT, BOOL.value, 0, both, -1, 0, "", ast)
            return both
        if kind == X_SUBIN or kind == X_EXISTS:
            raise self.err(ast, "a subquery (IN, NOT IN, EXISTS, NOT EXISTS) is supported as a condition of WHERE joined to the others by AND, one per query: " +
                           "it cannot be inside OR, NOT of a longer condition, the select list or HAVING (F-155)")
        if kind == X_IN:
            var n = self.in_list(ast)
            if x.flag and self.at(n).op != B_NULL:
                return self.mk(B_NOT, BOOL.value, 0, n, -1, 0, "", ast)
            return n
        # a function call
        if not _is_aggregate_name(x.text):
            return self.scalar_call(ast)
        if not self.in_post:
            raise self.err(ast, "aggregate function " + x.text.lower() + "() is not allowed here: aggregates belong in the select list, HAVING and ORDER BY")
        return self.aggregate(ast)

    def unknown_function(self, ast: Int) -> Error:
        var name = self.st.exprs[ast].text
        var names = _aggregate_names()
        for sn in _scalar_names():
            names.append(sn)
        var hint = closest(name, names)
        if hint.byte_length() > 0:
            return self.err(ast, "function `" + name.lower() + "` is not supported (did you mean `" + hint + "`?)")
        return self.err(ast, "function `" + name.lower() + "` is not supported: SQL here has the aggregates " + ", ".join(_aggregate_names()) +
                        " and the scalar functions " + ", ".join(_scalar_names()))

    def aggregate(mut self, ast: Int) raises -> Int:
        var x = self.st.exprs[ast].copy()
        var name = x.text
        var func = -1
        var arg = -1
        var p = 0.0
        var nargs = len(x.items)
        if name == "COUNT":
            if x.a == 1:
                if x.flag:
                    raise self.err(ast, "COUNT(DISTINCT *) is not valid")
                func = F_COUNT_STAR
            else:
                if nargs != 1:
                    raise self.err(ast, "count() takes one argument or *, found " + String(nargs))
                func = F_COUNT_DISTINCT if x.flag else F_COUNT
        else:
            if x.a == 1:
                raise self.err(ast, name.lower() + "(*) is not valid: only COUNT takes *")
            if x.flag and name != "MIN" and name != "MAX":
                raise self.err(ast, name.lower() + "(DISTINCT ...) is not supported: only COUNT(DISTINCT ...) is")
            if name == "SUM":
                func = F_SUM
            elif name == "MIN":
                func = F_MIN
            elif name == "MAX":
                func = F_MAX
            elif name == "AVG" or name == "MEAN":
                func = F_AVG
            elif name == "MEDIAN":
                func = F_MEDIAN
            elif name == "STDDEV" or name == "STDDEV_SAMP":
                func = F_STDDEV_SAMP
            elif name == "STDDEV_POP":
                func = F_STDDEV_POP
            elif name == "VARIANCE" or name == "VAR_SAMP":
                func = F_VAR_SAMP
            elif name == "VAR_POP":
                func = F_VAR_POP
            else:
                func = F_QUANTILE
            var want = 2 if func == F_QUANTILE else 1
            if nargs != want:
                raise self.err(ast, name.lower() + "() takes " + String(want) + (" arguments" if want != 1 else " argument") + ", found " + String(nargs))
        if func != F_COUNT_STAR:
            if _has_aggregate(self.st, x.items[0]):
                raise self.err(x.items[0], "aggregate functions cannot be nested")
            var was = self.in_post
            self.in_post = False
            arg = self.value(x.items[0])
            self.in_post = was
        if func == F_QUANTILE:
            var pe = x.items[1]
            if self.st.exprs[pe].kind != X_INT and self.st.exprs[pe].kind != X_DEC:
                raise self.err(pe, "the quantile of quantile_disc(x, p) must be a number between 0 and 1")
            var pn = parse_number(self.st.exprs[pe].text)
            if not pn.ok:
                raise self.err(pe, "the quantile must be a number between 0 and 1")
            p = Float64(Int(pn.value)) / Float64(Int(pow10_checked(pn.scale)))
            if p < 0.0 or p > 1.0:
                raise self.err(pe, "the quantile must be between 0 and 1, found " + self.st.exprs[pe].text)
        var rkind = INT.value
        var rscale = 0
        var src = -1
        if arg >= 0:
            var a = self.q.rexprs[arg].copy()
            if a.op == B_NULL:
                raise self.err(x.items[0], name.lower() + "(NULL) cannot be typed: write a column or a value")
            if a.kind == BOOL.value and a.op != B_COL and a.op != B_LIT:
                raise self.err(x.items[0], name.lower() + "() needs a value, found a comparison")
            if func == F_COUNT or func == F_COUNT_DISTINCT:
                if a.kind == TEXT.value and a.op == B_STR and func == F_COUNT_DISTINCT:
                    raise self.err(x.items[0], name.lower() + "() of a constant word is not supported")
            elif func == F_SUM:
                if a.kind == BOOL.value:                              # DuckDB sums a BOOLEAN as 0 / 1
                    rkind = INT.value
                elif not _is_num(a.kind):
                    raise self.err(x.items[0], "sum() needs numbers, found " + kind_name(a.kind, a.scale))
                else:
                    rkind = a.kind
                    rscale = 0 if a.kind == FLOAT.value else a.scale
            elif func == F_AVG:
                if not _is_num(a.kind):
                    raise self.err(x.items[0], "avg() needs numbers, found " + kind_name(a.kind, a.scale))
                rkind = FLOAT.value
            elif func >= F_STDDEV_SAMP and func <= F_VAR_POP:                  # a float64 whatever the number's kind, as DuckDB
                if not _is_num(a.kind):
                    raise self.err(x.items[0], name.lower() + "() needs numbers, found " + kind_name(a.kind, a.scale))
                rkind = FLOAT.value
            elif func == F_MIN or func == F_MAX or func == F_QUANTILE:
                rkind = a.kind
                rscale = a.scale
                src = a.src
                if a.kind == TEXT.value and a.op == B_STR:
                    raise self.err(x.items[0], name.lower() + "() of a constant word is not supported")
            elif func == F_MEDIAN and a.kind == TEXT.value:             # words have no mean: the lower middle word, in the text order, as DuckDB
                if a.op == B_STR:
                    raise self.err(x.items[0], "median() of a constant word is not supported")
                rkind = TEXT.value
                src = a.src
            elif func == F_MEDIAN and a.kind == DATE.value:
                raise self.err(x.items[0], "median() of a DATE is a TIMESTAMP in DuckDB (the middle of two dates can be half a day), which omxcore has no kind for yet (F-157): use quantile_disc(" +
                               self.st.render(x.items[0]) + ", 0.5)")
            elif func == F_MEDIAN:
                if not _is_num(a.kind):
                    raise self.err(x.items[0], "median() needs numbers, found " + kind_name(a.kind, a.scale))
                if a.kind == INT.value or a.kind == FLOAT.value:
                    rkind = FLOAT.value
                    rscale = a.scale if a.kind == FLOAT.value else 0       # a median of REAL is REAL
                else:
                    if a.scale + 1 > 18:
                        raise self.err(x.items[0], "median() of a DECIMAL with " + String(a.scale) + " places needs " + String(a.scale + 1) + ", past the 18 a value can hold")
                    rkind = DECIMAL.value
                    rscale = a.scale + 1
            # count(1), count('x'): a count of rows
            if func == F_COUNT and (a.op == B_LIT or a.op == B_STR):
                func = F_COUNT_STAR
                arg = -1
        var hol = func == F_COUNT_DISTINCT or func == F_MEDIAN or func == F_QUANTILE or (func >= F_STDDEV_SAMP and func <= F_VAR_POP)
        if (func == F_MIN or func == F_MAX) and rkind == TEXT.value:        # TEXT has no running min: the lowest / highest of its ranked values
            hol = True
            p = 0.0 if func == F_MIN else 1.0
        # dedupe: the same aggregate asked twice is computed once
        var canon = String(func) + ":" + ("*" if arg < 0 else self.canon(x.items[0])) + ":" + String(p)
        var shown: String
        if func == F_COUNT_STAR:
            shown = "count_star()"
        elif func == F_COUNT_DISTINCT:
            shown = "count(DISTINCT " + self.st.render(x.items[0]) + ")"
        elif func == F_QUANTILE:
            shown = "quantile_disc(" + self.st.render(x.items[0]) + ", " + self.st.render(x.items[1]) + ")"
        else:
            shown = (name.lower() if name != "MEAN" else "avg") + "(" + self.st.render(x.items[0]) + ")"
        var idx = -1
        for k in range(len(self.q.aggs)):
            if self.q.aggs[k].canon == canon:
                idx = k
                break
        if idx < 0:
            self.q.aggs.append(AggSpec(func, arg, p, rkind, rscale, shown, canon, hol))
            idx = len(self.q.aggs) - 1
        var n = self.mk(B_AGG, rkind, rscale, -1, -1, Int64(idx), shown, ast)
        self.q.pexprs[n].src = src
        return n


def _aggregate_names() -> List[String]:
    return ["sum", "count", "min", "max", "avg", "median", "quantile_disc", "stddev", "stddev_samp", "stddev_pop", "variance", "var_samp", "var_pop"]


def _scalar_names() -> List[String]:
    return ["coalesce", "ifnull", "nvl", "nullif", "lower", "upper", "trim", "ltrim", "rtrim", "substr", "substring", "length", "starts_with", "ends_with", "contains",
            "strpos", "instr", "left", "right", "reverse", "replace", "concat", "extract", "date_part", "year", "month", "day", "quarter", "dayofweek", "isodow",
            "dayofyear", "date_trunc", "datediff", "date_diff", "week", "weekofyear", "isoyear", "quantile_bin"]


def _is_scalar_name(up: String) -> Bool:
    var low = up.lower()
    for n in _scalar_names():
        if n == low:
            return True
    return up == "LCASE" or up == "UCASE" or up == "LEN" or up == "CHAR_LENGTH" or up == "CHARACTER_LENGTH" or up == "PREFIX" or up == "SUFFIX" or \
           up == "DATEPART" or up == "DATETRUNC" or up == "DECADE" or up == "QUANTILE_BIN"


def _is_aggregate_name(up: String) -> Bool:
    return up == "SUM" or up == "COUNT" or up == "MIN" or up == "MAX" or up == "AVG" or up == "MEAN" or up == "MEDIAN" or up == "QUANTILE_DISC" or
           up == "STDDEV" or up == "STDDEV_SAMP" or up == "STDDEV_POP" or up == "VARIANCE" or up == "VAR_SAMP" or up == "VAR_POP"


def _has_aggregate(st: SqlStatement, e: Int) -> Bool:
    ref x = st.exprs[e]
    if x.kind == X_CALL and _is_aggregate_name(x.text):
        return True
    if x.kind == X_CALL or x.kind == X_IN or x.kind == X_CASE:
        for it in x.items:
            if _has_aggregate(st, it):
                return True
    if x.kind == X_CAST:                                               # (its b and c are the precision and scale)
        return _has_aggregate(st, x.a)
    if x.kind == X_IN or x.kind == X_BIN or x.kind == X_NEG or x.kind == X_NOT or x.kind == X_ISNULL or x.kind == X_BETWEEN or x.kind == X_CASE or \
       x.kind == X_LIKE or x.kind == X_DISTINCT or x.kind == X_INTERVAL or x.kind == X_ISBOOL:
        if x.a >= 0 and _has_aggregate(st, x.a):
            return True
        if x.b >= 0 and _has_aggregate(st, x.b):
            return True
        if x.c >= 0 and _has_aggregate(st, x.c):
            return True
    return False


def _split_and_expr(st: SqlStatement, e: Int, mut out: List[Int]):
    if e < 0:
        return
    ref x = st.exprs[e]
    if x.kind == X_BIN and x.text == "AND":
        var a = x.a
        var b = x.b
        _split_and_expr(st, a, out)
        _split_and_expr(st, b, out)
    else:
        out.append(e)


def _and_chain(mut st: SqlStatement, parts: List[Int]) -> Int:
    """The conjunction of syntax-tree expressions (a new AND node per pair), -1 for none."""
    if len(parts) == 0:
        return -1
    var acc = parts[0]
    for k in range(1, len(parts)):
        st.exprs.append(SqlExpr(X_BIN, "AND", String(), acc, parts[k], -1, List[Int](), False, st.exprs[parts[k]].line, st.exprs[parts[k]].col))
        acc = len(st.exprs) - 1
    return acc


def _lift_subquery(mut st: SqlStatement, catalog: Catalog) raises:
    """`WHERE ... AND x [NOT] IN (SELECT c FROM t [WHERE p])` and `... AND [NOT] EXISTS (SELECT ... FROM t WHERE p)` (F-154) become a semi or
    anti join of the outer table with `t`: ON is the equality `x = c` (IN) and the subquery's conjuncts (the correlated equalities are the join keys,
    the others filter `t`); the rest of the WHERE stays. NOT IN is the null-aware anti join. One such subquery per query, and none beside an explicit
    join: more is multi-way (F-155). A subquery anywhere else is refused when its expression is bound."""
    if len(st.subs) == 0:
        return
    var conj = List[Int]()
    _split_and_expr(st, st.where, conj)
    var at = -1
    var neg = False
    var node = -1
    for i in range(len(conj)):
        var e = conj[i]
        var flip = False
        if st.exprs[e].kind == X_NOT and (st.exprs[st.exprs[e].a].kind == X_SUBIN or st.exprs[st.exprs[e].a].kind == X_EXISTS):
            e = st.exprs[e].a
            flip = True
        if st.exprs[e].kind == X_SUBIN or st.exprs[e].kind == X_EXISTS:
            if at >= 0:
                raise sql_error(st.exprs[e].line, st.exprs[e].col, "more than one subquery in WHERE is not supported yet (F-155)")
            at = i
            node = e
            neg = st.exprs[e].flag != flip
    if at < 0:
        return
    var x = st.exprs[node].copy()
    if st.has_right:
        raise sql_error(x.line, x.col, "a subquery in a query that also joins tables is not supported yet (F-155)")
    var sub = st.subs[x.items[0]].copy()
    if x.kind == X_SUBIN:
        st.join_kind = JK_ANTI_NA if neg else JK_SEMI
    else:
        st.join_kind = JK_ANTI if neg else JK_SEMI
    st.has_right = True
    st.right_table = sub.table
    st.right_alias = sub.alias
    st.right_is_file = sub.is_file
    st.right_line = sub.line
    st.right_col = sub.col
    st.from_subquery = True
    var parts = List[Int]()
    if x.kind == X_SUBIN:
        if sub.sel < 0:
            raise sql_error(sub.line, sub.col, "IN (SELECT *) needs one column: name it")
        # each side of the equality is bound to its own table by the binder's handles `#0` (outer) and `#1` (subquery), whatever the two tables are called
        # (a qualifier the user wrote that names neither table is left to be reported)
        var outer_shown = lower_ascii(st.table_alias) if st.table_alias.byte_length() > 0 else lower_ascii(st.table)
        var inner_shown = lower_ascii(sub.alias) if sub.alias.byte_length() > 0 else lower_ascii(sub.table)
        if st.exprs[x.a].kind == X_COL and (st.exprs[x.a].text2.byte_length() == 0 or lower_ascii(st.exprs[x.a].text2) == outer_shown or
                                            lower_ascii(st.exprs[x.a].text2) == lower_ascii(st.table)):
            st.exprs[x.a].text2 = "#0"
        if st.exprs[sub.sel].kind == X_COL and (st.exprs[sub.sel].text2.byte_length() == 0 or lower_ascii(st.exprs[sub.sel].text2) == inner_shown or
                                                lower_ascii(st.exprs[sub.sel].text2) == lower_ascii(sub.table)):
            st.exprs[sub.sel].text2 = "#1"
        st.exprs.append(SqlExpr(X_BIN, "=", String(), x.a, sub.sel, -1, List[Int](), False, x.line, x.col))
        parts.append(len(st.exprs) - 1)
    _split_and_expr(st, sub.where, parts)
    st.join_on = _and_chain(st, parts)
    var rest = List[Int]()
    for i in range(len(conj)):
        if i != at:
            rest.append(conj[i])
    st.where = _and_chain(st, rest)


def bind_sql(catalog: Catalog, text: String) raises -> BoundQuery:
    """Parses and binds one statement against the catalog."""
    return bind_statement(catalog, parse_sql(text), text)


def bind_statement(catalog: Catalog, var st: SqlStatement, text: String) raises -> BoundQuery:
    """Binds a parsed statement (`text` is what EXPLAIN and the error positions refer to)."""
    _lift_subquery(st, catalog)
    var src = catalog.find(st.table)
    if src < 0:
        var hint = closest(st.table, catalog.names())
        if st.table_is_file:
            raise sql_error(st.table_line, st.table_col, "the file '" + st.table + "' is not open: the `mx` command opens a file named in FROM; with the library, add it to the Catalog under that name")
        raise sql_error(st.table_line, st.table_col, "table `" + st.table + "` not found" +
                        ((" (did you mean `" + hint + "`?)") if hint.byte_length() > 0 else
                         (" (the catalog has " + ", ".join(catalog.names()) + ")") if len(catalog.sources) > 0 else " (the catalog is empty)"))
    var src2 = -1
    if st.has_right:
        src2 = catalog.find(st.right_table)
        if src2 < 0:
            var hint = closest(st.right_table, catalog.names())
            if st.right_is_file:
                raise sql_error(st.right_line, st.right_col, "the file '" + st.right_table + "' is not open: the `mx` command opens a file named in FROM; with the library, add it to the Catalog under that name")
            raise sql_error(st.right_line, st.right_col, "table `" + st.right_table + "` not found" +
                            ((" (did you mean `" + hint + "`?)") if hint.byte_length() > 0 else
                             (" (the catalog has " + ", ".join(catalog.names()) + ")") if len(catalog.sources) > 0 else " (the catalog is empty)"))
        if len(catalog.sources[src].overlay.formulas) > 0 or len(catalog.sources[src2].overlay.formulas) > 0:
            raise sql_error(st.right_line, st.right_col, "a join over a frame's formula columns is not supported yet: bake the formulas first")
        var same = lower_ascii(catalog.sources[src].name) == lower_ascii(catalog.sources[src2].name)
        if same and not st.from_subquery and (st.table_alias.byte_length() == 0 or st.right_alias.byte_length() == 0 or lower_ascii(st.table_alias) == lower_ascii(st.right_alias)):
            raise sql_error(st.right_line, st.right_col, "joining `" + catalog.sources[src].name + "` with itself needs a different alias for each side (FROM t a JOIN t b ...)")
        if not same and not st.from_subquery and st.table_alias.byte_length() > 0 and lower_ascii(st.table_alias) == lower_ascii(st.right_alias):
            raise sql_error(st.right_line, st.right_col, "the alias `" + st.right_alias + "` is used for both tables")
    var extra_src = List[Int]()
    for ex in st.extra:
        var sx = catalog.find(ex.name)
        if sx < 0:
            var hint = closest(ex.name, catalog.names())
            if ex.is_file:
                raise sql_error(ex.line, ex.col, "the file '" + ex.name + "' is not open: the `mx` command opens a file named in FROM; with the library, add it to the Catalog under that name")
            raise sql_error(ex.line, ex.col, "table `" + ex.name + "` not found" +
                            ((" (did you mean `" + hint + "`?)") if hint.byte_length() > 0 else
                             (" (the catalog has " + ", ".join(catalog.names()) + ")") if len(catalog.sources) > 0 else " (the catalog is empty)"))
        if len(catalog.sources[sx].overlay.formulas) > 0:
            raise sql_error(ex.line, ex.col, "a join over a frame's formula columns is not supported yet: bake the formulas first")
        # every table of FROM needs its own name: the same table twice needs two aliases
        var names = List[String]()
        var aliases = List[String]()
        names.append(catalog.sources[src].name)
        aliases.append(st.table_alias)
        names.append(catalog.sources[src2].name)
        aliases.append(st.right_alias)
        for y in range(len(extra_src)):
            names.append(catalog.sources[extra_src[y]].name)
            aliases.append(st.extra[y].alias)
        for y in range(len(names)):
            var same = lower_ascii(names[y]) == lower_ascii(catalog.sources[sx].name)
            if same and (aliases[y].byte_length() == 0 or ex.alias.byte_length() == 0 or lower_ascii(aliases[y]) == lower_ascii(ex.alias)):
                raise sql_error(ex.line, ex.col, "joining `" + names[y] + "` with itself needs a different alias for each side (FROM t a JOIN t b ...)")
            if not same and aliases[y].byte_length() > 0 and lower_ascii(aliases[y]) == lower_ascii(ex.alias):
                raise sql_error(ex.line, ex.col, "the alias `" + ex.alias + "` is used for two tables")
        extra_src.append(sx)
    var b = _Binder(st^)
    b.jkind = b.st.join_kind
    b.inner_first = b.st.from_subquery
    b.q.join_kind = b.st.join_kind
    b.table_name = catalog.sources[src].name
    b.table_alias = b.st.table_alias
    for c in range(catalog.sources[src].table.columns()):
        b.src_names.append(catalog.sources[src].table.name(c))
        b.src_kinds.append(catalog.sources[src].table.kind(c).value)
        b.src_scales.append(catalog.sources[src].table.scale(c))
    b.n_stored = len(b.src_names)
    b.tab_names.append(b.table_name)
    b.tab_alias.append(b.table_alias)
    b.tab_start.append(0)
    b.tab_cols.append(len(b.src_names))
    b.q.sources.append(src)
    if src2 >= 0:
        b.q.joined = True
        b.tab_names.append(catalog.sources[src2].name)
        b.tab_alias.append(b.st.right_alias)
        b.tab_start.append(len(b.src_names))
        b.tab_cols.append(catalog.sources[src2].table.columns())
        for c in range(catalog.sources[src2].table.columns()):
            b.src_names.append(catalog.sources[src2].table.name(c))
            b.src_kinds.append(catalog.sources[src2].table.kind(c).value)
            b.src_scales.append(catalog.sources[src2].table.scale(c))
        b.n_stored = len(b.src_names)
        b.q.sources.append(src2)
        for u in b.st.join_using:
            b.using_cols.append(u)
            b.q.using_cols.append(u)
        for y in range(len(extra_src)):
            ref sx = catalog.sources[extra_src[y]]
            b.tab_names.append(sx.name)
            b.tab_alias.append(b.st.extra[y].alias)
            b.tab_start.append(len(b.src_names))
            b.tab_cols.append(sx.table.columns())
            for c in range(sx.table.columns()):
                b.src_names.append(sx.table.name(c))
                b.src_kinds.append(sx.table.kind(c).value)
                b.src_scales.append(sx.table.scale(c))
            b.n_stored = len(b.src_names)
            b.q.sources.append(extra_src[y])
    b.q.col_names = b.src_names.copy()
    b.q.tab_names = b.tab_names.copy()
    b.q.tab_alias = b.tab_alias.copy()
    b.q.tab_start = b.tab_start.copy()
    b.q.tab_cols = b.tab_cols.copy()
    for f in catalog.sources[src].overlay.formulas:           # a frame's formula columns are columns too
        b.formulas.append(f.copy())
        b.src_names.append(f.name)
        b.src_kinds.append(FLOAT.value if f.scale < 0 else DECIMAL.value if f.scale > 0 else INT.value)
        b.src_scales.append(max(f.scale, 0))
    b.q.source = src
    b.q.text = text
    b.q.line = b.st.items[0].line
    b.q.col = b.st.items[0].col
    b.q.explain = b.st.explain
    b.q.limit = b.st.limit
    b.q.offset = b.st.offset
    for i in range(len(b.st.exprs)):                    # an unknown function is reported as that, before anything else
        if b.st.exprs[i].kind == X_CALL and b.st.exprs[i].c < 0 and not _is_aggregate_name(b.st.exprs[i].text) and not _is_scalar_name(b.st.exprs[i].text):
            raise b.unknown_function(i)
    # F-152: `*` and `t.*` become the source's columns (a frame's formula columns too)
    var items = List[SelectItem]()
    for it in b.st.items:
        if not it.star:
            items.append(it.copy())
            continue
        var only = -1
        if it.alias.byte_length() > 0:
            var qual = lower_ascii(it.alias)
            for t in range(len(b.tab_names)):
                if qual == lower_ascii(b.tab_names[t]) or (b.tab_alias[t].byte_length() > 0 and qual == lower_ascii(b.tab_alias[t])):
                    only = t
            if only < 0:
                raise sql_error(it.line, it.col, "unknown table or alias `" + it.alias + "` in `" + it.alias + ".*`")
        if b.is_semi_anti():                                   # a semi / anti join returns the first table's columns only
            if only == 1:
                raise sql_error(it.line, it.col, "unknown table or alias `" + it.alias + "` in `" + it.alias + ".*`: after a semi or anti join only the first table has columns")
            for c in range(b.tab_start[0], b.tab_start[0] + b.tab_cols[0]):
                b.st.exprs.append(SqlExpr(X_COL, b.src_names[c], String(), -1, -1, -1, List[Int](), False, it.line, it.col))
                items.append(SelectItem(len(b.st.exprs) - 1, String(), False, it.line, it.col))
            continue
        if len(b.tab_names) == 1:
            for nm in b.src_names:
                b.st.exprs.append(SqlExpr(X_COL, nm, String(), -1, -1, -1, List[Int](), False, it.line, it.col))
                items.append(SelectItem(len(b.st.exprs) - 1, String(), False, it.line, it.col))
            continue
        # two tables: the first table's columns, then the second's without a USING column (it is the first's); every name qualified
        for t in range(len(b.tab_names)):
            if only >= 0 and t != only:
                continue
            for c in range(b.tab_start[t], b.tab_start[t] + b.tab_cols[t]):
                var skip = False
                if only < 0 and t == 1:
                    for un in b.using_cols:
                        if lower_ascii(un) == lower_ascii(b.src_names[c]):
                            skip = True
                if skip:
                    continue
                b.st.exprs.append(SqlExpr(X_COL, b.src_names[c], b.tab_label(t), -1, -1, -1, List[Int](), False, it.line, it.col))
                items.append(SelectItem(len(b.st.exprs) - 1, String(), False, it.line, it.col))
    b.st.items = items^
    if len(b.st.items) == 0:
        raise sql_error(1, 1, "the select list is empty")
    # an aggregate query: GROUP BY, or an aggregate somewhere
    var aggregating = len(b.st.group_by) > 0
    for it in b.st.items:
        if _has_aggregate(b.st, it.expr):
            aggregating = True
    if b.st.having >= 0:
        aggregating = True
    for o in b.st.order_by:
        if _has_aggregate(b.st, o.expr):
            aggregating = True
    if b.st.distinct:
        if aggregating:
            raise sql_error(b.st.items[0].line, b.st.items[0].col, "`SELECT DISTINCT` together with aggregates, GROUP BY or HAVING is not supported yet: " +
                            "use GROUP BY alone (it returns the same distinct rows), or DISTINCT without aggregates")
        # SELECT DISTINCT a, b is the group-by of the select list: the same machinery runs it
        for it in b.st.items:
            b.st.group_by.append(it.expr)
        aggregating = True
        b.q.distinct = True
        b.distinct = True
    if not aggregating:
        return bind_rows(b^)
    # WHERE (and a join's ON)
    if b.st.where >= 0 and _has_aggregate(b.st, b.st.where):
        raise b.err(b.st.where, "aggregate functions are not allowed in WHERE: filter the groups with HAVING")
    b.bind_where()
    # GROUP BY: ordinals and aliases name select items
    for g in range(len(b.st.group_by)):
        var ast = b.st.group_by[g]
        var gx = b.st.exprs[ast].copy()
        var target = ast
        if gx.kind == X_INT and not b.distinct:
            var n = Int(parse_number(gx.text).value)
            if n < 1 or n > len(b.st.items):
                raise b.err(ast, "GROUP BY " + gx.text + " is out of range: the select list has " + String(len(b.st.items)) + " items")
            target = b.st.items[n - 1].expr
            if _has_aggregate(b.st, target):
                raise b.err(ast, "GROUP BY " + gx.text + " refers to an aggregate")
        elif gx.kind == X_COL and gx.text2.byte_length() == 0 and not b.distinct:
            var is_column = False
            for nm in b.src_names:
                if lower_ascii(nm) == lower_ascii(gx.text):
                    is_column = True
            if not is_column:
                for it in b.st.items:
                    if it.has_alias and lower_ascii(it.alias) == lower_ascii(gx.text):
                        target = it.expr
                        break
        if _has_aggregate(b.st, target):
            raise b.err(ast, "aggregate functions are not allowed in GROUP BY")
        var cn = b.canon(target)
        var dup = False
        for k in b.key_canons:
            if k == cn:
                dup = True
        if dup:
            continue
        var e = b.value(target)
        var ke = b.q.rexprs[e].copy()
        if ke.op == B_NULL:
            raise b.err(target, "GROUP BY NULL cannot be typed")
        if ke.kind == BOOL.value and ke.op != B_COL:
            raise b.err(target, "grouping by a comparison is not supported yet: group by a column")
        b.q.keys.append(KeySpec(e, b.src_names[Int(ke.value)] if ke.op == B_COL else b.st.render(target), cn))
        b.key_canons.append(cn)
    # select items, HAVING, ORDER BY: after aggregation
    b.in_post = True
    for _ in range(len(b.st.items)):
        b.alias_expr.append(-1)
    for k in range(len(b.st.items)):
        var it = b.st.items[k].copy()
        var e = b.value(it.expr)
        var pe = b.q.pexprs[e].copy()
        if pe.op == B_NULL and pe.kind == K_NULL:
            raise b.err(it.expr, "a bare NULL in the select list cannot be typed")
        b.q.outputs.append(e)
        var nm: String
        if it.has_alias:
            nm = it.alias
        elif pe.op == B_KEY or pe.op == B_AGG:
            nm = pe.text
        else:
            nm = b.st.render(it.expr)
        b.q.out_names.append(nm)
        b.q.out_text.append(b.st.render(it.expr))
        if it.has_alias:
            b.alias_expr[k] = e
    if b.st.having >= 0:
        var h = b.value(b.st.having)
        b.q.having = b.boolean(h, b.st.having, "HAVING")
        b.q.having_text = b.st.render(b.st.having)
    for oi in b.st.order_by:
        var ox = b.st.exprs[oi.expr].copy()
        var e = -1
        if ox.kind == X_INT:
            var n = Int(parse_number(ox.text).value)
            if n < 1 or n > len(b.st.items):
                raise b.err(oi.expr, "ORDER BY " + ox.text + " is out of range: the select list has " + String(len(b.st.items)) + " items")
            e = b.q.outputs[n - 1]
        elif ox.kind == X_COL and ox.text2.byte_length() == 0:
            for k in range(len(b.st.items)):
                if b.st.items[k].has_alias and lower_ascii(b.st.items[k].alias) == lower_ascii(ox.text):
                    e = b.q.outputs[k]
                    break
        if e < 0:
            e = b.value(oi.expr)
        var pe = b.q.pexprs[e].copy()
        if pe.op == B_NULL:
            raise b.err(oi.expr, "ORDER BY a constant NULL does nothing: remove it")
        b.q.order.append(SortKey(e, oi.desc, oi.nulls == 1))
        b.q.order_text.append(b.st.render(oi.expr) + (" DESC" if oi.desc else "") + (" NULLS FIRST" if oi.nulls == 1 else ""))
    return b^.take()


def bind_rows(var b: _Binder) raises -> BoundQuery:
    """A query with no aggregate and no GROUP BY: it returns rows. The select list and ORDER BY are row-level expressions
    (`routs`, `rorder`); WHERE is bound as before."""
    b.q.rows = True
    b.bind_where()
    for k in range(len(b.st.items)):
        var it = b.st.items[k].copy()
        var e = b.value(it.expr)
        var x = b.q.rexprs[e].copy()
        if x.op == B_NULL and x.kind == K_NULL:
            raise b.err(it.expr, "a bare NULL in the select list cannot be typed")
        b.q.routs.append(e)
        var nm: String
        if it.has_alias:
            nm = it.alias
        elif x.op == B_COL:
            nm = x.text
        else:
            nm = b.st.render(it.expr)
        b.q.out_names.append(nm)
        b.q.out_text.append(b.st.render(it.expr))
    for oi in b.st.order_by:
        var ox = b.st.exprs[oi.expr].copy()
        var e = -1
        if ox.kind == X_INT:
            var n = Int(parse_number(ox.text).value)
            if n < 1 or n > len(b.st.items):
                raise b.err(oi.expr, "ORDER BY " + ox.text + " is out of range: the select list has " + String(len(b.st.items)) + " items")
            e = b.q.routs[n - 1]
        elif ox.kind == X_COL and ox.text2.byte_length() == 0:
            for k in range(len(b.st.items)):
                if b.st.items[k].has_alias and lower_ascii(b.st.items[k].alias) == lower_ascii(ox.text):
                    e = b.q.routs[k]
                    break
        if e < 0:
            e = b.value(oi.expr)
        var x = b.q.rexprs[e].copy()
        if x.op == B_NULL:
            raise b.err(oi.expr, "ORDER BY a constant NULL does nothing: remove it")
        b.q.rorder.append(SortKey(e, oi.desc, oi.nulls == 1))
        b.q.order_text.append(b.st.render(oi.expr) + (" DESC" if oi.desc else "") + (" NULLS FIRST" if oi.nulls == 1 else ""))
    return b^.take()


def copy_query(q: BoundQuery) -> BoundQuery:
    """A copy of a bound query (the plan lowers a new one from its nodes and leaves the binder's alone)."""
    var c = BoundQuery()
    c.source = q.source
    c.rexprs = q.rexprs.copy()
    c.pexprs = q.pexprs.copy()
    c.where = q.where
    c.keys = q.keys.copy()
    c.aggs = q.aggs.copy()
    c.outputs = q.outputs.copy()
    c.out_names = q.out_names.copy()
    c.having = q.having
    c.order = q.order.copy()
    c.limit = q.limit
    c.offset = q.offset
    c.explain = q.explain
    c.text = q.text
    c.line = q.line
    c.col = q.col
    c.where_text = q.where_text
    c.having_text = q.having_text
    c.order_text = q.order_text.copy()
    c.out_text = q.out_text.copy()
    c.joined = q.joined
    c.join_kind = q.join_kind
    c.sources = q.sources.copy()
    c.tab_names = q.tab_names.copy()
    c.tab_alias = q.tab_alias.copy()
    c.tab_start = q.tab_start.copy()
    c.tab_cols = q.tab_cols.copy()
    c.jk_l = q.jk_l.copy()
    c.jk_r = q.jk_r.copy()
    c.jk_l_names = q.jk_l_names.copy()
    c.jk_r_names = q.jk_r_names.copy()
    c.conj_l = q.conj_l.copy()
    c.conj_r = q.conj_r.copy()
    c.conj_x = q.conj_x.copy()
    c.conj_l_text = q.conj_l_text.copy()
    c.conj_r_text = q.conj_r_text.copy()
    c.conj_x_text = q.conj_x_text.copy()
    c.using_cols = q.using_cols.copy()
    c.col_names = q.col_names.copy()
    c.leaf_conj = q.leaf_conj.copy()
    c.leaf_conj_text = q.leaf_conj_text.copy()
    c.edge_a = q.edge_a.copy()
    c.edge_b = q.edge_b.copy()
    c.resid = q.resid.copy()
    c.resid_text = q.resid_text.copy()
    c.resid_mask = q.resid_mask.copy()
    c.rows = q.rows
    c.distinct = q.distinct
    c.routs = q.routs.copy()
    c.rorder = q.rorder.copy()
    return c^
