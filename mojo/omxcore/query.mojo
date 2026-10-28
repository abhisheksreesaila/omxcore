"""Queries (docs/omxcore-mojo-api.md section 5), on a small plan representation lowered to the native kernel.

    var r = Query(table).where(col("region") == 2)
                        .group_by(col("carrier"), groups=7)       # a key expression with a known group count
                        .agg(sum(col("revenue")), count())
                        .compute()
    r.sums; r.counts; r.values(i); r.means(i); r.milliseconds

The plan is a list of nodes (Scan, Filter, Aggregate today; Project, Sort, Limit and Join have their kinds
reserved), so the smart planner can later rewrite and extend it without changing this API. `compute` lowers the
plan to one program for the CPU kernel (`kernels/pipeline_cpu.mojo`): filters ANDed into the predicate, the group
key computed per block, every value an exact int64.

Semantics, as the app's stand-in (mojoui frame_gpu.group_sum), plus nulls:
- groups are dense 0 .. groups-1; a row whose key is outside that range, or null, is left out;
- a filter that reads a null cell is false for that row (the whole filter, not only the comparison);
- sum / mean / min / max skip a row whose value reads a null cell; count() counts rows, count(e) rows where e
  reads no null; `present(i, g)`: aggregate i has a value in group g (its min / max / mean are meaningless otherwise);
- values are the stored integers (a DECIMAL(2) literal 5 is 0.05); + - and comparisons refuse columns of different
  scales (decimal-aware formulas: F-136);
- the null rule for `|`: a filter reading a null cell is false as a whole, so `(a > 0) | (b == 1)` drops a row whose
  a is null even when b == 1 (SQL would keep it);
- a total that passes int64 raises (the stand-in wrapped silently);
- `count_distinct(e)`, `median(e)` (DuckDB's median, exact at scale + 1) / `percentile(e, p)` (quantile_disc) and `top_k(e, k)` (F-148) are exact
  per group over a table, a Frame and a scan, ignore nulls, and run on the CPU: the kernel's per-block hook writes the
  (group, value) pairs and holistic.mojo consumes them (`r.values(i)`, `r.present(i, g)`, `r.top(i, g)`).
- `stddev(e)` / `stddev_pop(e)` / `variance(e)` / `var_pop(e)` (F-177) are float64 aggregates folded from the same pairs into a count, a mean and a sum of squared
  deviations per group (`r.floats(i)`; `present` is false below the two values the sample variants need). `Query.quantile_bins(e, n)` finds the edges of
  n equal-count bins (A_QEDGE aggregates, whole-number order statistics) and `QuantileBins.key(e)` is the bin number 0 .. n-1 as a group key.
"""
from std.time import perf_counter_ns
from std.atomic import Atomic
from std.memory import Pointer
from .table import SharedTable, Table, TEXT, FLOAT, INT, DECIMAL, DATE, BOOL
from .parquet import ParquetScan, _GroupPlan, _GroupReader, _open_plan, _copy_bits, _count_bits
from .frame import Frame, Overlay, _eval_row, _eval_row_t
from .expr import (MAX_QUANTILE_BINS, Expr, Node, N_COL, N_LIT, N_LITF, N_DIV, N_I2F, N_ADD, N_SUB, N_MUL, N_FLOORDIV, N_MOD, N_NEG, N_LEAST, N_GREATEST,
                   N_LT, N_LE, N_GT, N_GE, N_EQ, N_NE, N_AND, N_OR, N_NOT, N_BOOL2INT, N_SCALE_TO)
from .kernels._fpenv import core_count, par_n
from .kernels.pipeline_cpu import pipeline_keyed, pipeline_plain, E_OVERFLOW, IP, FP, LP, VEC
from .kernels.pipeline_gpu import run_pipeline_gpu, _choose_mode, scratch_rows, int_word_program_ok, APPLE_GPU, HDR as GPU_HDR
from max.gpu.host import DeviceContext, DeviceBuffer
from .gpu import (DeviceTable, HAS_GPU, NO_GPU_MESSAGE, gpu_available, _ScanContext, _scan_context, _stage_load, _stage_range, _stage_data, _stage_valid,
                  _pinned_prepare, gpu_budget, gpu_capacity, gpu_free_bytes, human_bytes, gpu_fail_piece, INJECTED_OOM, GPU_RETRY_FACTOR_PERCENT)
from .holistic import (HolStates, HolSpec, HolResult, window_blocks, heap_offer, float_key, A_DISTINCT, A_PCT, A_TOPK, A_MEDIAN,
                       A_VAR_POP, A_VAR_SAMP, A_STDDEV_POP, A_STDDEV_SAMP, A_QEDGE, is_spread, spread_needs)
from .joincore import JoinKeys, JoinSide, normalize_keys, join_pairs, select_rows, gather_join, JOIN_BUDGET_BYTES
from .kernels.hashjoin_cpu import hj_build, PM_SEMI, PM_ANTI
from .frame import text_ranks
from .job import JobCtl, STAGE_PLAN, STAGE_AGGREGATE, STAGE_FINISH      # F-175
from .window import WindowFn, Window, W_SUM, W_COUNT, W_COUNT_STAR, W_AVG, W_MIN, W_MAX      # F-168

# plan node kinds
comptime P_SCAN = 0
comptime P_FILTER = 1
comptime P_PROJECT = 2       # reserved
comptime P_AGGREGATE = 3
comptime P_SORT = 4          # reserved
comptime P_LIMIT = 5         # reserved
comptime P_JOIN = 6          # reserved

# aggregate kinds
comptime A_SUM = 0
comptime A_COUNT = 1         # count() with no argument: rows
comptime A_COUNT_OF = 2      # count(e): rows where e reads no null
comptime A_MIN = 3
comptime A_MAX = 4
comptime A_MEAN = 5

# devices
comptime CPU = 0
comptime GPU = 1
comptime AUTO = 2
# AUTO uses the GPU when a DeviceTable is attached and the table has at least this many rows: below it the CPU
# kernel (all cores, no launch or copy-back) answers in about the time a GPU launch plus the result copy take.
comptime AUTO_GPU_MIN_ROWS = 1_000_000

# how a kernel slot merges across partial results (a morsel, a row group, a chunk): the integer ones above, and the float64 ones (F-157)
comptime A_FSUM = 20         # float64 addition, in the order the partial results are folded (chunk order: the sum is reproducible)
comptime A_FMIN = 21         # float64 minimum (NaN never reaches it: the lowering swaps it for the identity and counts it)
comptime A_FMAX = 22

# kernel opcodes (kernels_aot/pipeline_cpu.mojo, pipeline_program.py)
comptime MAGIC = 0x4D585031
comptime OP_LOAD_COL = 1
comptime OP_LOAD_CONST = 2
comptime OP_LOADI = 3
comptime OP_ADD = 10
comptime OP_SUB = 11
comptime OP_MUL = 12
comptime OP_DIV = 13
comptime OP_IADD = 15
comptime OP_ISUB = 16
comptime OP_IMUL = 17
comptime OP_F2I = 18
comptime OP_I2F = 19
comptime OP_AND = 30
comptime OP_OR = 31
comptime OP_NOT = 32
comptime OP_SELECT = 35
comptime OP_ISNULL = 41
comptime OP_IDIV = 42
comptime OP_IMOD = 43
comptime OP_IMIN = 44
comptime OP_IMAX = 45
comptime OP_ICMP = 46
comptime OP_AGG_SUM = 50
comptime OP_AGG_MIN = 51
comptime OP_AGG_MAX = 52
comptime OP_AGG_ISUM = 53
comptime OP_AGG_IMIN = 54
comptime OP_AGG_IMAX = 55
comptime MAX_REGS = 48
comptime MAX_CONSTS = 64
comptime DT_I64 = 1
comptime DT_F64 = 3
comptime NAN_AWARE = 1       # the last operand of a float compare: NaN equals itself and is above every number, as DuckDB
comptime DT_NULLMASK = 7


struct Agg(Copyable, Movable):
    var kind: Int
    var arg: Expr
    var p: Float64              # percentile: the fraction 0 .. 1
    var k: Int                  # top_k: how many values

    def __init__(out self, kind: Int, var arg: Expr, p: Float64 = 0.0, k: Int = 0):
        self.kind = kind
        self.arg = arg^
        self.p = p
        self.k = k

    def over(self, var w: Window) raises -> WindowFn:
        """This aggregate over a window (F-168): `sum(col("v")).over(Window().partition_by(col("g")).order_by(col("day")))`. sum, count, mean, min_of and
        max_of take a window; the others raise by name."""
        var kind = -1
        if self.kind == A_SUM:
            kind = W_SUM
        elif self.kind == A_COUNT:
            kind = W_COUNT_STAR
        elif self.kind == A_COUNT_OF:
            kind = W_COUNT
        elif self.kind == A_MEAN:
            kind = W_AVG
        elif self.kind == A_MIN:
            kind = W_MIN
        elif self.kind == A_MAX:
            kind = W_MAX
        if kind < 0:
            raise Error("this aggregate is not supported as a window function (sum, count, mean, min_of and max_of are)")
        var f = WindowFn(kind)
        f.arg = self.arg.copy()
        return f^.over(w^)


def sum(e: Expr) -> Agg:
    return Agg(A_SUM, e.copy())


def count() -> Agg:
    return Agg(A_COUNT, Expr(0))


def count(e: Expr) -> Agg:
    return Agg(A_COUNT_OF, e.copy())


def mean(e: Expr) -> Agg:
    return Agg(A_MEAN, e.copy())


def min_of(e: Expr) -> Agg:
    return Agg(A_MIN, e.copy())


def max_of(e: Expr) -> Agg:
    return Agg(A_MAX, e.copy())


def count_distinct(e: Expr) -> Agg:
    """The number of different non-null values of e in the group (exact; `r.values(i)`)."""
    return Agg(A_DISTINCT, e.copy())


def percentile(e: Expr, p: Float64) -> Agg:
    """The value at fraction p (0 .. 1) of the group's sorted non-null values: DuckDB's `quantile_disc`, the value at
    index max(ceil(p * n) - 1, 0), an actual value of the column (`r.values(i)`, `r.present(i, g)`)."""
    return Agg(A_PCT, e.copy(), p)


def median(e: Expr) -> Agg:
    """DuckDB's `median` (quantile_cont at 0.5): the middle value, or for an even count the mean of the two middle ones,
    exactly, so the answer is the stored value times 10 (the column's scale plus one decimal place): [1, 2, 3, 4] gives
    25. A TEXT column has no mean: its median is the lower middle word (a code), unscaled. The value `percentile(e, 0.5)`
    returns is the lower middle, unscaled."""
    return Agg(A_MEDIAN, e.copy())


def stddev(e: Expr) -> Agg:
    """The sample standard deviation of the group's non-null values (DuckDB's `stddev` = `stddev_samp`): a float64 (`r.floats(i)`), no value
    (`r.present(i, g)` false) for a group with fewer than two. INT, DECIMAL and FLOAT values are read as float64 as DuckDB casts them
    (`float(v) / 10^scale`); the sum of squared deviations is stable against cancellation (F-177)."""
    return Agg(A_STDDEV_SAMP, e.copy())


def stddev_pop(e: Expr) -> Agg:
    """The population standard deviation (`stddev_pop`): a value for a group of one (0.0), none for an empty one."""
    return Agg(A_STDDEV_POP, e.copy())


def variance(e: Expr) -> Agg:
    """The sample variance (DuckDB's `variance` = `var_samp`); see `stddev`."""
    return Agg(A_VAR_SAMP, e.copy())


def var_pop(e: Expr) -> Agg:
    """The population variance (`var_pop`); see `stddev_pop`."""
    return Agg(A_VAR_POP, e.copy())


def top_k(e: Expr, k: Int) -> Agg:
    """The k largest non-null values of e in the group, largest first, repeats kept (`r.top(i, g)`)."""
    return Agg(A_TOPK, e.copy(), 0.0, k)


@fieldwise_init
struct PlanNode(Copyable, Movable):
    var kind: Int
    var input: Int                # the node below (-1 for a scan)
    var exprs: List[Expr]         # Filter: the predicate; Aggregate: the group key (none for one group)
    var aggs: List[Agg]
    var groups: Int


struct QuantileBins(Copyable, Movable):
    """The edges of equal-count bins (`Query.quantile_bins`): edge k is the value at sorted place ceil(k * N / bins). `key(e)` is the bin of a row, 0 ..
    bins()-1: the number of edges below its value, an `Expr` for `group_by(key, groups=bins())`."""
    var edges: List[Int64]       # stored values (a FLOAT column's are float64 bits)
    var is_float: Bool

    def __init__(out self, var edges: List[Int64], is_float: Bool):
        self.edges = edges^
        self.is_float = is_float

    def bins(self) -> Int:
        return len(self.edges) + 1

    def key(self, e: Expr) -> Expr:
        """The bin of `e` (0 .. bins()-1): the sum over the edges of (e > edge) as 1 or 0."""
        var acc = Expr(0)
        for k in range(len(self.edges)):
            var edge = Expr(Float64(from_bits=UInt64(self.edges[k]))) if self.is_float else Expr(Int(self.edges[k]))
            var term = (e > edge)._un(N_BOOL2INT)
            if k == 0:
                acc = term^
            else:
                acc = acc + term
        return acc^


struct Query(Movable):
    """A plan over one shared table, or over a session's view of it (`Query(frame)`: its edits, appended and removed
    rows, merges and formula columns, copied when the query is built), built by chaining; `compute` runs it."""
    var table: SharedTable
    var nodes: List[PlanNode]
    var framed: Bool
    var overlay: Overlay
    var device: Optional[DeviceTable]
    var scan: Optional[ParquetScan]      # a Parquet file streamed one row group at a time (`Query(scan)`)
    var _morsels: Int                    # a Parquet scan by morsels: MORSELS_AUTO, MORSELS_ON or MORSELS_OFF (F-149)
    var _block_rows: Int                 # ... in blocks of this many rows (0: SCAN_BLOCK_ROWS)
    var _gpu_ops: Int                    # AUTO sends a streamed file to the GPU above this many kernel instructions a row per core per column read
    var _window_rows: Int                # rows per window of a count_distinct / percentile / top_k query (0: as many as
                                         # fit WINDOW_BYTES); tests set it small to cut a table into several windows
    var _threads: Int                    # worker threads the CPU kernel starts (0: one a core); a float64 sum is the same for any number
    var windows: List[WindowFn]          # F-168: window functions over the rows the plan keeps (`window`; `compute_windows` in winquery.mojo runs them)

    def __init__(out self, table: SharedTable):
        self.windows = List[WindowFn]()
        self.table = table
        self.device = None
        self.scan = None
        self._window_rows = 0
        self._threads = 0
        self._morsels = MORSELS_AUTO
        self._block_rows = 0
        self._gpu_ops = AUTO_GPU_OPS_PER_CORE_COLUMN
        self.nodes = [PlanNode(P_SCAN, -1, List[Expr](), List[Agg](), 0)]
        self.framed = False
        self.overlay = Overlay(table)

    def __init__(out self, scan: ParquetScan):
        """A plan over a Parquet file that may be bigger than memory: `compute` reads one row group at a time (only the
        columns the plan uses), runs the kernel on it on the CPU and merges the partial results, so the answer is
        `read_parquet` + the same query, exactly, and the peak memory is one row group plus the result."""
        self.windows = List[WindowFn]()
        self.table = scan.schema
        self.device = None
        self._window_rows = 0
        self._threads = 0
        self._morsels = MORSELS_AUTO
        self._block_rows = 0
        self._gpu_ops = AUTO_GPU_OPS_PER_CORE_COLUMN
        self.scan = scan.copy()
        self.nodes = [PlanNode(P_SCAN, -1, List[Expr](), List[Agg](), 0)]
        self.framed = False
        self.overlay = Overlay(scan.schema)

    def __init__(out self, frame: Frame) raises:
        self.windows = List[WindowFn]()
        self.table = frame.base
        self.device = None
        self._window_rows = 0
        self._threads = 0
        self._morsels = MORSELS_AUTO
        self._block_rows = 0
        self._gpu_ops = AUTO_GPU_OPS_PER_CORE_COLUMN
        self.scan = frame.scan.copy()                    # a Frame over a scan streams the file with the overlay (F-147)
        self.nodes = [PlanNode(P_SCAN, -1, List[Expr](), List[Agg](), 0)]
        self.framed = True
        self.overlay = frame.overlay()

    def _top(self) -> Int:
        return len(self.nodes) - 1

    def where(var self, predicate: Expr) -> Self:
        """Keep rows where `predicate` holds (several `where`s are ANDed)."""
        self.nodes.append(PlanNode(P_FILTER, self._top(), [predicate.copy()], List[Agg](), 0))
        return self^

    def window(var self, *fns: WindowFn) -> Self:
        """Window functions over the rows the plan's `where`s keep (F-168): `Query(t).where(...).window(row_number().over(w), sum(col("v")).over(w))`;
        `compute_windows(query)` (winquery.mojo) returns the source's columns and one more per function."""
        for f in fns:
            self.windows.append(f.copy())
        return self^

    def _aggregate(self) -> Int:
        for i in range(len(self.nodes)):
            if self.nodes[i].kind == P_AGGREGATE:
                return i
        return -1

    def group_by(var self, key: Expr, groups: Int) raises -> Self:
        """Group by an integer key expression with values 0 .. groups-1 (dense: missing groups are zeros).
        `where` filters rows whether it comes before or after (there is no HAVING yet)."""
        var at = self._aggregate()
        if at >= 0 and len(self.nodes[at].exprs) == 0:
            raise Error("group_by must come before agg")
        if at >= 0:
            raise Error("a query has one group_by (and its aggregates); this one has a second")
        self.nodes.append(PlanNode(P_AGGREGATE, self._top(), [key.copy()], List[Agg](), groups))
        return self^

    def agg(var self, *aggs: Agg) -> Self:
        var at = self._aggregate()
        if at < 0:
            self.nodes.append(PlanNode(P_AGGREGATE, self._top(), List[Expr](), List[Agg](), 1))
            at = self._top()
        for a in aggs:
            self.nodes[at].aggs.append(a.copy())
        return self^

    def quantile_bins(var self, e: Expr, bins: Int) raises -> QuantileBins:
        """Equal-count bins of `e` over the rows this query keeps (its `where`s, a Frame's edits, a scan's row groups): the `bins` - 1 edges, found
        exactly. Edge k (k = 1 .. bins-1) is the value at sorted place ceil(k * N / bins) of the N non-null values of `e` (whole-number arithmetic);
        a value's bin is the number of edges below it, 0 .. bins-1, so equal values share a bin and a run of ties across an edge falls in the lower
        bin. It is SQL's `quantile_bin(e, bins)` minus one. Use it as the key of a second query:

            var b = Query(f).where(...).quantile_bins(col("x"), 10)
            var r = Query(f).where(...).group_by(b.key(col("x")), groups=b.bins()).agg(count(), sum(col("y"))).compute()

        The bins are of `e` over the rows that pass this query's filters: give both queries the same `where`s. A row whose `e` is null has no bin
        (the builder leaves a null key out). One pass over the data on the CPU; at most MAX_QUANTILE_BINS (32) bins."""
        if bins < 2 or bins > MAX_QUANTILE_BINS:
            raise Error("quantile_bins: the number of bins is a whole number from 2 to " + String(MAX_QUANTILE_BINS) + ", got " + String(bins))
        if self._aggregate() >= 0:
            raise Error("quantile_bins comes before group_by and agg: it bins the rows that pass the filters")
        var aggs = List[Agg]()
        for k in range(1, bins):
            aggs.append(Agg(A_QEDGE, e.copy(), Float64(k), bins))
        self.nodes.append(PlanNode(P_AGGREGATE, self._top(), List[Expr](), aggs^, 1))
        var r = self.compute(CPU)
        var out = QuantileBins(List[Int64](), r.is_float(0))
        for i in range(bins - 1):
            out.edges.append(Int64(r.values(i)[0]) if r.present(i, 0) else Int64(0))      # (no value at all: no row reaches a bin)
        return out^

    def morsels(var self, mode: Int) -> Self:
        """How a Parquet scan runs: MORSELS_AUTO (by row-range morsels on all cores when the file is big enough),
        MORSELS_ON (always: tests, timing) or MORSELS_OFF (a row group at a time). The answer is the same."""
        self._morsels = mode
        return self^

    def auto_gpu_ops(var self, ops_per_core_column: Int) -> Self:
        """For a Parquet scan on AUTO: the GPU when the plan runs more than this many kernel instructions a row per core
        per column read (default AUTO_GPU_OPS_PER_CORE_COLUMN); 0 sends every file to the GPU (tests, timing)."""
        self._gpu_ops = ops_per_core_column
        return self^

    def threads(var self, n: Int) -> Self:
        """The CPU kernel's worker threads (default: one a core). Only the time changes: a float64 sum has one defined order
        (per 65,536-row morsel in row order, the morsels' totals in order), whatever the number of threads or the block size."""
        self._threads = max(n, 0)
        return self^

    def block_rows(var self, rows: Int) -> Self:
        """The rows a morsel worker decodes and aggregates at a time (default SCAN_BLOCK_ROWS)."""
        self._block_rows = rows
        return self^

    def on_device(var self, device_table: DeviceTable) raises -> Self:
        """Attaches the device copy of this query's table, so `compute(GPU)` (and AUTO, on a big table) can use it.
        Explicit rather than a process-wide registry: Mojo has no safe mutable globals, the app owns the one
        DeviceTable next to its SharedTable and hands it to each session's queries, and a handle that was made
        from another table is refused here, not silently answered with the wrong data."""
        if self.scan:
            raise Error("on_device: a Parquet scan runs on the CPU, one row group at a time")
        if not device_table.mirrors(self.table):
            raise Error("on_device: this DeviceTable was not made from the query's table")
        self.device = device_table
        return self^

    def join(var self, other: SharedTable, left_on: List[String], right_on: List[String], how: String = "inner") raises -> Self:
        """The inner equi-join of this query's table with `other` on `left_on[i] = right_on[i]`, as a new query over the joined rows (this table's
        columns, then `other`'s; a repeated name gets a suffix). The builder's own rules apply from there on (`where`, `group_by`, `agg`).
        The join is the engine's (the same kernel and key rules as SQL): a NULL key never matches, TEXT keys match by bytes whatever the
        dictionaries, DECIMAL scales are aligned (INT is scale 0), FLOAT keys follow DuckDB (-0.0 = 0.0, NaN = NaN), int64 keys past 2^53 are
        exact, and for an inner join the smaller table is the one the hash table is built on. `how`: "inner"; "left" (every row of this table; the
        other's columns are NULL where nothing matches); "semi" (this table's rows that have a partner, this table's columns only); "anti" (the rows
        that have none: a NULL key has none). A left, semi or anti join builds on `other`. The join must come first, over a plain table (bake a
        Frame, read a file, first)."""
        if how != "inner" and how != "left" and how != "semi" and how != "anti":
            raise Error("a `" + how + "` join is not supported: how is inner, left, semi or anti")
        if len(self.nodes) > 1:
            raise Error("join must come first: filters and aggregates come after it")
        if self.framed or self.scan:
            raise Error("join: a Frame or a Parquet scan is not joined by the builder: bake the frame or read the file into a table first (SQL joins them as they are)")
        if len(left_on) == 0 or len(left_on) != len(right_on):
            raise Error("join: `left_on` and `right_on` name the same number of columns, at least one")
        var lcols = List[Int]()
        var rcols = List[Int]()
        for k in range(len(left_on)):
            var l = self.table.index(left_on[k])
            var r = other.index(right_on[k])
            if l < 0:
                raise Error("join: no column `" + left_on[k] + "` in this query's table")
            if r < 0:
                raise Error("join: no column `" + right_on[k] + "` in the other table")
            var lk = self.table.kind(l).value
            var rk = other.kind(r).value
            var numeric = (lk == INT.value or lk == DECIMAL.value) and (rk == INT.value or rk == DECIMAL.value)
            if not (numeric or lk == rk):
                raise Error("join: cannot join `" + left_on[k] + "` with `" + right_on[k] + "`: the two columns are different kinds of value")
            lcols.append(l)
            rcols.append(r)
        var lc = List[Int]()
        for c in range(self.table.columns()):
            lc.append(c)
        var rc = List[Int]()
        for c in range(other.columns()):
            rc.append(c)
        var left = JoinSide(self.table, lc^, lcols.copy(), True)
        var right = JoinSide(other, rc^ if (how == "inner" or how == "left") else List[Int](), rcols.copy(), True)
        var left_builds = how == "inner" and self.table.rows() < other.rows()
        var keys = JoinKeys()
        if left_builds:
            normalize_keys(self.table, other, lcols, rcols, keys, self._threads)
        else:
            normalize_keys(other, self.table, rcols, lcols, keys, self._threads)
        var nb = self.table.rows() if left_builds else other.rows()
        var np = other.rows() if left_builds else self.table.rows()
        var m = hj_build(keys.bk_addr, keys.bo_addr, nb, self._threads, how == "semi" or how == "anti")
        var joined: Table
        if how == "semi" or how == "anti":
            var pick = select_rows(m, keys, np, PM_SEMI if how == "semi" else PM_ANTI, self._threads)
            joined = gather_join(left, right, pick, False, self._threads)
        else:
            var ncols = self.table.columns() + other.columns()
            var pairs = join_pairs(m, keys, np, JOIN_BUDGET_BYTES // (8 * max(ncols, 1)), self._threads, how == "left")
            joined = gather_join(left, right, pairs, left_builds, self._threads, how == "left")
        var out = Query(joined^.share())
        out._threads = self._threads
        out._morsels = self._morsels
        out._block_rows = self._block_rows
        out._window_rows = self._window_rows
        return out^

    def compute(self, device: Int = AUTO, job: JobCtl = JobCtl()) raises -> QueryResult:
        """Runs the plan. `device`: CPU, GPU (raises without a live attached DeviceTable) or AUTO (the GPU when a
        live DeviceTable is attached and the table has `AUTO_GPU_MIN_ROWS` rows or more, else the CPU). The result is
        the same either way, edits of a frame included. `job` (F-175, job.mojo): the background job this runs for, whose cancel
        is checked between row ranges, windows, row groups and blocks and whose progress it advances; a plain call passes none."""
        job.stage(STAGE_PLAN)
        if self.scan:
            var scan_nodes = self.nodes.copy()
            if self.framed:                              # formulas inlined, merges remapped (as below)
                for i in range(len(scan_nodes)):
                    for k in range(len(scan_nodes[i].exprs)):
                        scan_nodes[i].exprs[k] = self.overlay.expand(scan_nodes[i].exprs[k])
                    for k in range(len(scan_nodes[i].aggs)):
                        scan_nodes[i].aggs[k].arg = self.overlay.expand(scan_nodes[i].aggs[k].arg)
            return _compute_scan(self.scan.value(), scan_nodes, device, self.nodes, self.overlay, self.framed,
                                 self._morsels, self._block_rows, self._gpu_ops, self._threads, job)
        var holistic = False
        for node in self.nodes:
            for a in node.aggs:
                holistic = holistic or a.kind >= A_DISTINCT
        var use_gpu = False
        var pieced = False
        var fallback = String()
        if device == GPU:
            comptime if not HAS_GPU:
                raise Error("device GPU: " + NO_GPU_MESSAGE)
            if holistic:
                raise Error("device GPU: count_distinct, percentile, top_k, stddev and variance run on the CPU")
            if not self.device:
                raise Error("device GPU: no DeviceTable is attached (Query.on_device(DeviceTable(table)))")
            if not self.device.value().is_live():
                raise Error("device GPU: the DeviceTable was released")
            use_gpu = True
        elif device == AUTO and self.device and self.device.value().is_live() and self.table.rows() >= AUTO_GPU_MIN_ROWS:
            if holistic:
                fallback = "count_distinct, percentile, top_k, stddev and variance run on the CPU"
            else:
                use_gpu = True
        var t0 = perf_counter_ns()
        var nodes = self.nodes.copy()
        if self.framed:                                  # formulas inlined, merges remapped
            for i in range(len(nodes)):
                for k in range(len(nodes[i].exprs)):
                    nodes[i].exprs[k] = self.overlay.expand(nodes[i].exprs[k])
                for k in range(len(nodes[i].aggs)):
                    nodes[i].aggs[k].arg = self.overlay.expand(nodes[i].aggs[k].arg)
        var prog = _lower(self.table, nodes, self.framed, len(self.overlay.mask) > 0)
        job.check()
        job.stage(STAGE_AGGREGATE)
        job.add_total(self.table.rows())
        if use_gpu and prog.has_float:
            if device == GPU:
                raise Error("device GPU: a FLOAT column or float arithmetic runs on the CPU (a float64 sum has one defined order there)")
            use_gpu = False
            fallback = "a FLOAT column or float arithmetic runs on the CPU (a float64 sum has one defined order there)"
        var hs = HolStates(1, List[HolSpec](), List[List[Int64]]())
        if prog.mat:
            hs = _build_states(self.nodes, nodes, self.table, self.overlay.words, None, -1 if self.framed else self.table.rows(), prog.src_float.copy(), prog.src_scale.copy())
        var r = QueryResult()
        var done = False
        if use_gpu:
            comptime if HAS_GPU:
                # F-189: does the working set fit? A table over the budget has no device copy: AUTO answers on the CPU, GPU
                # uploads row ranges that fit; a resident table's launch needs its buffers (AUTO: the CPU if they do not fit
                # or an earlier query ran out of memory and not much more is free now; GPU: a readable error)
                var dt = self.device.value()
                var ctx = dt.context()
                var usable = gpu_budget(ctx)
                var masked = len(self.overlay.mask) > 0
                var fixed = _gpu_fixed_bytes(ctx, prog, (self.table.rows() + 7) // 8 + 8 if masked else 0)
                var used = _gpu_columns_used(self.table, prog)
                var want = fixed + _range_bytes(self.table.rows(), _count_true(used[0]), _count_true(used[1]), False)
                if not dt.is_resident():
                    if device == AUTO:
                        use_gpu = False
                        fallback = ("the query needs " + human_bytes(want) + " on the GPU (" + String(self.table.rows()) + " rows), over its budget of " +
                                    human_bytes(usable) + ": AUTO answers on the CPU, compute(GPU) runs it in pieces that fit")
                    else:
                        pieced = True
                elif fixed > usable:
                    var why = "the query's buffers need " + human_bytes(fixed) + " (" + String(prog.groups) + " groups), over the GPU budget of " + human_bytes(usable)
                    if device == GPU and fixed > gpu_capacity(ctx):      # (not merely a GPU that other work fills right now: that one is the CPU's)
                        raise Error("device GPU: " + why)
                    use_gpu = False
                    fallback = why^
                elif device == AUTO and dt.oom_free() > 0 and gpu_free_bytes(ctx) * 100 < dt.oom_free() * GPU_RETRY_FACTOR_PERCENT:
                    use_gpu = False
                    fallback = ("an earlier query ran out of GPU memory with " + human_bytes(dt.oom_free()) + " free; AUTO stays on the CPU until " +
                                human_bytes(dt.oom_free() // 100 * GPU_RETRY_FACTOR_PERCENT) + " is free (" + human_bytes(gpu_free_bytes(ctx)) + " now)")
        if use_gpu:
            comptime if HAS_GPU:
                if pieced:
                    r = _run_gpu_pieces(self.device.value(), self.table, prog, self.overlay.mask, fallback, job)
                    done = True
                else:
                    var g = _run_gpu(self.device.value(), self.table, prog, self.overlay.mask, device == GPU, fallback)
                    job.advance(self.table.rows())
                    if g:
                        r = g.take()
                        r.device = GPU
                        r.pieces = 1
                        r.gpu_pieces = 1
                        self.device.value().forget_oom()
                        done = True
                    elif _is_oom(fallback):
                        self.device.value().remember_oom(gpu_free_bytes(self.device.value().context()))
        if not done:
            var mask_addr = Int(self.overlay.mask.unsafe_ptr()) if len(self.overlay.mask) > 0 else 0
            if prog.mat:
                r = _run_mat(self.table, prog, mask_addr, hs, self._window_rows, self._threads, job)
            else:
                r = _run_cpu_job(self.table, prog, mask_addr, self._threads, job)
            r.fallback_reason = fallback^
        job.check()
        job.stage(STAGE_FINISH)
        if self.framed and len(self.overlay.delta) > 0:
            _add_delta(r, self.table, nodes, self.overlay, hs)
        if prog.mat:
            _finish_holistic(r, hs)
        r.milliseconds = Float64(perf_counter_ns() - t0) / 1e6
        return r^


struct QueryResult(Movable):
    var kinds: List[Int]
    var slot_of: List[Int]          # per aggregate: its kernel slot, -1 when none (count(), mean's count)
    var count_slot_of: List[Int]    # per aggregate: the slot counting its non-null rows (count(e), mean), -1 when none
    var slots: List[List[Int]]      # per kernel slot, per group
    var counts: List[Int]           # rows per group (count(*))
    var sums: List[Int]             # the first sum's values (the spec's `r.sums`), else empty
    var milliseconds: Float64
    var device: Int                 # CPU or GPU: where the kernel ran
    var fallback_reason: String     # AUTO chose the GPU and the CPU answered instead: why (else ""); GPU: why a piece ran on the CPU
    var pieces: Int                 # F-189: the row ranges (a table) or row groups (a scan) a GPU query was cut into, 1 for one launch, 0 on the CPU
    var gpu_pieces: Int             # ... and how many of them ran on the GPU (the rest ran on the CPU, `fallback_reason` says why)
    # count_distinct / percentile / top_k (holistic.mojo): per aggregate its index among the holistic ones, else -1
    var hol_of: List[Int]
    var hvals: List[List[Int]]      # per holistic aggregate: per group, the distinct count or the percentile value
    var htop: List[List[Int]]       # per holistic aggregate: top_k values, group after group, largest first
    var htop_off: List[List[Int]]   # per holistic aggregate: group g's values are htop[off[g] : off[g + 1]]
    var hcount: List[List[Int]]     # per holistic aggregate: per group, how many values it saw (non-null rows)
    var agg_float: List[Bool]       # per aggregate: its value is a float64 (`floats(i)`; `values(i)` holds its bits)
    var nan_slot_of: List[Int]      # per float min / max: the slot counting the NaNs the kernel left out, else -1
    var hfloat: List[Bool]          # per holistic aggregate: its source is float64 (median, quantile_disc, top_k give float bits)

    def __init__(out self):
        self.device = CPU
        self.pieces = 0
        self.gpu_pieces = 0
        self.hol_of = List[Int]()
        self.hvals = List[List[Int]]()
        self.htop = List[List[Int]]()
        self.htop_off = List[List[Int]]()
        self.hcount = List[List[Int]]()
        self.agg_float = List[Bool]()
        self.nan_slot_of = List[Int]()
        self.hfloat = List[Bool]()
        self.fallback_reason = String()
        self.kinds = List[Int]()
        self.slot_of = List[Int]()
        self.count_slot_of = List[Int]()
        self.slots = List[List[Int]]()
        self.counts = List[Int]()
        self.sums = List[Int]()
        self.milliseconds = 0.0

    def values(self, i: Int) -> List[Int]:
        """Aggregate i's value per group (a mean: use `means`; a top_k: use `top`). count_distinct: the number of
        different non-null values; percentile / median: the stored value (a TEXT column's code; the value's decimal
        scale is the column's), 0 for a group with none (`present` says)."""
        if self.hol_of[i] >= 0:
            return self.hvals[self.hol_of[i]].copy()
        if self.kinds[i] == A_COUNT:
            return self.counts.copy()
        if self.kinds[i] == A_COUNT_OF:
            return self.slots[self.count_slot_of[i]].copy() if self.count_slot_of[i] >= 0 else self.counts.copy()
        if len(self.agg_float) > i and self.agg_float[i]:             # a float64 aggregate: the bits of `floats(i)`
            var f = self.floats(i)
            var out = List[Int]()
            for g in range(len(f)):
                out.append(Int(Int64(f[g].to_bits())))
            return out^
        return self.slots[self.slot_of[i]].copy()

    def is_float(self, i: Int) -> Bool:
        """Aggregate i's value is a float64 (`floats(i)`): a float sum, mean, min, max, median or quantile (not count_distinct)."""
        if self.hol_of[i] >= 0:
            return is_spread(self.kinds[i]) or (self.hfloat[self.hol_of[i]] and self.kinds[i] != A_DISTINCT)
        return len(self.agg_float) > i and self.agg_float[i]

    def floats(self, i: Int) -> List[Float64]:
        """Aggregate i's value per group as a float64 (F-157): a float sum, min, max (NaN is above every number: MAX is NaN when the
        group holds one, MIN only when it holds nothing else) or mean (NaN for a group with no value), a float median / percentile,
        or any other aggregate's number converted. A group with no value holds 0.0 (`present` says)."""
        var out = List[Float64]()
        if self.hol_of[i] >= 0:
            var h = self.hol_of[i]
            for g in range(len(self.hvals[h])):
                out.append(Float64(from_bits=UInt64(Int64(self.hvals[h][g]))) if self.is_float(i) else Float64(self.hvals[h][g]))
            return out^
        if len(self.agg_float) > i and self.agg_float[i]:
            var s = self.slots[self.slot_of[i]].copy()
            var n = self.slots[self.count_slot_of[i]].copy() if self.count_slot_of[i] >= 0 else self.counts.copy()
            for g in range(len(s)):
                var x = Float64(from_bits=UInt64(Int64(s[g])))
                if self.kinds[i] == A_MEAN:
                    out.append(x / Float64(n[g]) if n[g] > 0 else Float64(from_bits=UInt64(0x7FF8000000000000)))
                elif self.kinds[i] == A_SUM:
                    out.append(x)
                else:
                    var nans = self.slots[self.nan_slot_of[i]][g]
                    var nan = Float64(from_bits=UInt64(0x7FF8000000000000))
                    if self.kinds[i] == A_MAX and nans > 0:
                        out.append(nan)
                    elif self.kinds[i] == A_MIN and n[g] > 0 and n[g] - nans == 0:
                        out.append(nan)
                    else:
                        out.append(x if n[g] > 0 else 0.0)
            return out^
        for v in self.values(i):
            out.append(Float64(v))
        return out^

    def means(self, i: Int) -> List[Float64]:
        """Aggregate i's mean per group: its exact sum over its count (NaN for a group with none)."""
        if len(self.agg_float) > i and self.agg_float[i]:
            return self.floats(i)
        var s = self.slots[self.slot_of[i]].copy()
        var n = self.slots[self.count_slot_of[i]].copy() if self.count_slot_of[i] >= 0 else self.counts.copy()
        var out = List[Float64]()
        for g in range(len(s)):
            out.append(Float64(s[g]) / Float64(n[g]) if n[g] > 0 else Float64.MAX * 0.0 / 0.0)
        return out^

    def top(self, i: Int, g: Int) -> List[Int]:
        """top_k aggregate i's values in group g: its k largest non-null values, largest first, repeats kept (fewer
        than k when the group has fewer; a TEXT column's codes, ordered alphabetically)."""
        var h = self.hol_of[i]
        var out = List[Int]()
        if h < 0 or len(self.htop_off[h]) == 0:                  # not a top_k aggregate
            return out^
        for x in range(self.htop_off[h][g], self.htop_off[h][g + 1]):
            out.append(self.htop[h][x])
        return out^

    def is_empty(self, g: Int) -> Bool:
        """Group g has no rows."""
        return self.counts[g] == 0

    def present(self, i: Int, g: Int) -> Bool:
        """Aggregate i has a value in group g: a row whose value is not null (a min / max / mean over only nulls
        has none, and its number is meaningless). count_distinct, percentile, top_k: the group has a non-null value."""
        if self.hol_of[i] >= 0:
            return self.hcount[self.hol_of[i]][g] >= (spread_needs(self.kinds[i]) if is_spread(self.kinds[i]) else 1)
        if self.count_slot_of[i] >= 0:
            return self.slots[self.count_slot_of[i]][g] > 0
        return self.counts[g] > 0


# ── lowering ─────────────────────────────────────────────────────────────────────────────────────────

struct Program(Movable):
    var pred: List[Int32]
    var body: List[Int32]
    var aggs: List[Int32]
    var consts: List[Int64]          # int64 bits, loaded exactly by LOAD_CONST
    var cols: List[Int]              # table column of each program column (data first, then validity triplets)
    var dtypes: List[Int32]
    var null_of: List[Int]           # table column -> program column whose dtype is the null mask, or -1
    var n_regs: Int
    var pred_reg: Int
    var keyed: Bool
    var groups: Int
    var kinds: List[Int]
    var slot_of: List[Int]
    var count_slot_of: List[Int]
    var n_slots: Int
    var mat: Bool                    # a holistic aggregate (count_distinct, percentile, top_k): the hook materializes pairs
    var src_of: List[Int]            # per aggregate: its holistic value source, -1 for a plain one
    var n_src: Int
    var src_vreg: List[Int]          # per source: the register holding its value, and its validity (-1: no nulls)
    var src_valid: List[Int]
    var ctl_const: Int               # the constant slot patched with the control block's address (`_run_mat`)
    var section: Int                 # where `emit` writes: 0 predicate, 1 body, 2 aggregates
    var next_keep: Int               # registers 1 .. next_keep-1 hold values that must live on (the predicate's
    var temp_top: Int                # running AND, each aggregate's value); temporaries come down from the top
    var set_bytes: List[UInt8]       # F-150 (SQL): the byte sets OP_IN_SET reads and their (offset, length, lowest) triples; the builder has none
    var set_meta: List[Int64]
    var repro: Bool                  # F-157: the program sums float64 (OP_AGG_SUM): the kernel runs in its reproducible mode (F-064)
    var has_float: Bool              # it reads a float64 column, a float literal or converts to float64: it runs on the CPU only
    var agg_float: List[Bool]        # per aggregate: its value is a float64 (sum, mean, min, max of a float expression)
    var nan_slot_of: List[Int]       # per float min / max: the slot counting the NaNs it left out (NaN is above every number), else -1
    var src_float: List[Bool]        # per holistic source: it reads float64 values (the hook turns them into order-preserving keys)
    var src_scale: List[Int]         # per holistic source: the decimal places of its integer values (stddev / variance read float(v) / 10^scale)
    var conv: List[String]           # F-156: the casts that can fail (OP_FLAG): what the error says when one does
    var text_tabs: Int               # F-156: the per-word tables the program carries (a text function answered once per dictionary word)
    var text_words: Int              # ... and how many words they were evaluated on
    var text_ns: Int                 # ... and how long that took
    var day_tabs: Int                # F-156: the per-day tables (a date function of one column, answered once per day of the column's range)
    var day_days: Int                # ... over how many days

    def __init__(out self, n_table_cols: Int):
        self.pred = List[Int32]()
        self.body = List[Int32]()
        self.aggs = List[Int32]()
        self.consts = List[Int64]()
        self.cols = List[Int]()
        self.dtypes = List[Int32]()
        self.null_of = List[Int](length=n_table_cols, fill=-1)
        self.n_regs = MAX_REGS           # register 0: the group key (int), read by `_key_from_reg0`
        self.next_keep = 1
        self.temp_top = MAX_REGS - 1
        self.pred_reg = -1
        self.keyed = False
        self.groups = 1
        self.kinds = List[Int]()
        self.slot_of = List[Int]()
        self.count_slot_of = List[Int]()
        self.n_slots = 0
        self.mat = False
        self.src_of = List[Int]()
        self.n_src = 0
        self.src_vreg = List[Int]()
        self.src_valid = List[Int]()
        self.ctl_const = -1
        self.section = 0
        self.set_bytes = List[UInt8]()
        self.set_meta = List[Int64]()
        self.repro = False
        self.has_float = False
        self.agg_float = List[Bool]()
        self.nan_slot_of = List[Int]()
        self.src_float = List[Bool]()
        self.src_scale = List[Int]()
        self.conv = List[String]()
        self.text_tabs = 0
        self.text_words = 0
        self.text_ns = 0
        self.day_tabs = 0
        self.day_days = 0

    def add_conv(mut self, why: String):
        for c in self.conv:
            if c == why:
                return
        if why.startswith(FOOTER_LIE):                              # a lying footer is reported as itself, first: the query then runs again without statistics
            self.conv.insert(0, why)
        else:
            self.conv.append(why)

    def conv_text(self) -> String:
        return "; ".join(self.conv)

    def emit(mut self, op: Int, d: Int, a: Int = 0, b: Int = 0, c: Int = 0):
        var ins: List[Int32] = [Int32(op), Int32(d), Int32(a), Int32(b), Int32(c)]
        if self.section == 0:
            self.pred.extend(ins^)
        elif self.section == 1:
            self.body.extend(ins^)
        else:
            self.aggs.extend(ins^)

    def reg(mut self) raises -> Int:
        """A temporary register, free again after `reset_temps`."""
        if self.temp_top < self.next_keep:
            raise Error("the query is too big for one program (more than " + String(MAX_REGS) + " registers)")
        self.temp_top -= 1
        return self.temp_top + 1

    def keep(mut self, r: Int, boolean: Bool) raises -> Int:
        """Copies register r into one that lives to the end of the program."""
        if self.next_keep > self.temp_top:
            raise Error("the query is too big for one program (more than " + String(MAX_REGS) + " registers)")
        var k = self.next_keep
        self.next_keep += 1
        self.emit(OP_AND if boolean else OP_IMAX, k, r, r)       # a & a, max(a, a): a copy
        return k

    def reset_temps(mut self):
        self.temp_top = MAX_REGS - 1

    def const(mut self, v: Int64) raises -> Int:
        for k in range(len(self.consts)):
            if self.consts[k] == v:
                return k
        if len(self.consts) >= MAX_CONSTS:
            raise Error("the query has more than " + String(MAX_CONSTS) + " distinct constants")
        self.consts.append(v)
        return len(self.consts) - 1

    def column(mut self, table_col: Int) -> Int:
        for k in range(len(self.cols)):
            if self.cols[k] == table_col and Int(self.dtypes[k]) == DT_I64:
                return k
        self.cols.append(table_col)
        self.dtypes.append(DT_I64)
        return len(self.cols) - 1

    def column_f64(mut self, table_col: Int) -> Int:
        """The program column that reads a FLOAT table column as float64 (OP_LOAD_COL)."""
        for k in range(len(self.cols)):
            if self.cols[k] == table_col and Int(self.dtypes[k]) == DT_F64:
                return k
        self.cols.append(table_col)
        self.dtypes.append(DT_F64)
        return len(self.cols) - 1


def _op_name(op: Int) -> String:
    var names: List[String] = ["+", "-", "*", "//", "%", "-", "least", "greatest", "/", "", "<", "<=", ">", ">=",
                               "==", "!=", "", "", "", "", "&", "|", "~"]
    return names[op - N_ADD] if N_ADD <= op <= N_NOT else "operator " + String(op)


@fieldwise_init
struct _Slot(Copyable, Movable):
    var reg: Int
    var boolean: Bool       # a comparison's 1.0 / 0.0 (float) rather than an int64 value
    var scale: Int          # a column's decimal places; -1 for a literal or a mix that has none
    var fl: Bool            # the register holds a float64 (F-157), not int64 bits


def _to_float(mut p: Program, mut a: _Slot) raises:
    """Reads an int register as a float64, in place: float(v) / 10^scale, as DuckDB casts an INT or DECIMAL to DOUBLE. A value past
    +-2^53 is converted the way DuckDB does it (F-190): the kernel's OP_I2F with 1 + decimal places in its last field does the whole cast."""
    if a.fl:
        return
    p.emit(OP_I2F, a.reg, a.reg, 0, 1 + max(a.scale, 0))
    a.fl = True
    a.scale = -1
    p.has_float = True


def _compile_expr(mut p: Program, e: Expr, t: SharedTable) raises -> Int:
    """Emits e; returns its register: an int register for a value, a float (1.0 / 0.0) for a boolean (see `_compile_typed`)."""
    return _compile_typed(p, e, t).reg


def _compile_typed(mut p: Program, e: Expr, t: SharedTable) raises -> _Slot:
    """Emits e; returns its slot: an int register for a value, a float64 register (`fl`) for a FLOAT value, a float (1.0 / 0.0) for a
    boolean. Raises when an operator gets the wrong kind (a number where a comparison is due, or the reverse) or when + - and
    comparisons meet columns of different decimal scales (decimal-aware formulas are F-136: values are the stored integers). An INT or
    DECIMAL meeting a float64 is converted to one (`_to_float`)."""
    var stack = List[_Slot]()
    for n in e.nodes:
        if n.op == N_COL:
            var c = t.column(n.name)
            var r = p.reg()
            if t.kind(c) == FLOAT:
                p.emit(OP_LOAD_COL, r, p.column_f64(c))
                p.has_float = True
                stack.append(_Slot(r, False, -1, True))
            else:
                p.emit(OP_LOADI, r, p.column(c))
                stack.append(_Slot(r, False, t.scale(c), False))
        elif n.op == N_LIT:
            var r = p.reg()
            p.emit(OP_LOAD_CONST, r, p.const(n.value))
            stack.append(_Slot(r, False, -1, False))
        elif n.op == N_LITF:                             # the constant pool holds bits
            var r = p.reg()
            p.emit(OP_LOAD_CONST, r, p.const(n.value))
            p.has_float = True
            stack.append(_Slot(r, False, -1, True))
        elif n.op == N_I2F:
            var a = stack.pop()
            a.scale = Int(n.value)
            _to_float(p, a)
            stack.append(a^)
        elif n.op == N_BOOL2INT:                         # a comparison as the number 1 / 0 (merges)
            var a = stack.pop()
            if not a.boolean:
                raise Error("internal: int() of a number")
            p.emit(OP_F2I, a.reg, a.reg)                 # in place: an operator's result takes its operand's register
            stack.append(_Slot(a.reg, False, -1, False))
        elif n.op == N_SCALE_TO:                         # a formula's rescaled side: the value is already scaled up
            var a = stack.pop()
            if a.boolean:
                raise Error("internal: a rescaled comparison")
            stack.append(_Slot(a.reg, False, Int(n.value), a.fl))
        elif n.op == N_NEG or n.op == N_NOT:
            var a = stack.pop()
            if a.boolean != (n.op == N_NOT):
                raise Error("`" + _op_name(n.op) + "` needs " + ("a comparison" if n.op == N_NOT else "a number") +
                            ", got " + ("a number" if n.op == N_NOT else "a comparison"))
            p.temp_top = a.reg                           # operands are freed: the result takes the first one's register
            var r = p.reg()
            if n.op == N_NOT:
                p.emit(OP_NOT, r, a.reg)
            elif a.fl:                                   # times -1.0: the sign of zero, inf and NaN flips too
                var m1 = p.reg()
                p.emit(OP_LOAD_CONST, m1, p.const(Int64(Float64(-1.0).to_bits())))
                p.emit(OP_MUL, r, a.reg, m1)
            else:
                var z = p.reg()
                p.emit(OP_LOAD_CONST, z, p.const(0))
                p.emit(OP_ISUB, r, z, a.reg)
            stack.append(_Slot(r, n.op == N_NOT, a.scale, a.fl and n.op == N_NEG))
        else:
            var b = stack.pop()
            var a = stack.pop()
            var logical = n.op == N_AND or n.op == N_OR
            if a.boolean != logical or b.boolean != logical:
                raise Error("`" + _op_name(n.op) + "` needs " + ("comparisons" if logical else "numbers") + " on both sides")
            var compare = n.op >= N_LT and n.op <= N_NE
            var fl = (a.fl or b.fl or n.op == N_DIV) and not logical
            if fl:
                if n.op == N_FLOORDIV or n.op == N_MOD:
                    raise Error("`" + _op_name(n.op) + "` on a FLOAT is not supported: use `/` for a division of floats")
                _to_float(p, a)
                _to_float(p, b)
            elif (n.op == N_ADD or n.op == N_SUB or n.op == N_LEAST or n.op == N_GREATEST or compare) and \
                 a.scale >= 0 and b.scale >= 0 and a.scale != b.scale:
                raise Error("`" + _op_name(n.op) + "` meets values with " + String(a.scale) + " and " + String(b.scale) +
                            " decimal places: the stored integers would not line up (decimal-aware formulas: F-136)")
            p.temp_top = a.reg                           # both operands are freed (a, b are the two newest registers)
            var r = p.reg()
            var scale = a.scale if a.scale >= 0 else b.scale
            if fl:
                if n.op == N_ADD:
                    p.emit(OP_ADD, r, a.reg, b.reg)
                elif n.op == N_SUB:
                    p.emit(OP_SUB, r, a.reg, b.reg)
                elif n.op == N_MUL:
                    p.emit(OP_MUL, r, a.reg, b.reg)
                elif n.op == N_DIV:
                    p.emit(OP_DIV, r, a.reg, b.reg)
                elif n.op == N_LEAST or n.op == N_GREATEST:    # NaN is above every number: least skips it, greatest is it
                    var pick = p.reg()
                    p.emit(N_LT if n.op == N_LEAST else N_GT, pick, a.reg, b.reg, NAN_AWARE)
                    p.emit(OP_SELECT, r, pick, a.reg, b.reg)
                elif compare:
                    p.emit(n.op, r, a.reg, b.reg, NAN_AWARE)  # N_LT .. N_NE are the kernel's OP_LT .. OP_NE
                else:
                    raise Error("unknown expression operator " + String(n.op))
                stack.append(_Slot(r, compare, -1, not compare))
                continue
            if n.op == N_ADD:
                p.emit(OP_IADD, r, a.reg, b.reg)
            elif n.op == N_SUB:
                p.emit(OP_ISUB, r, a.reg, b.reg)
            elif n.op == N_MUL:
                p.emit(OP_IMUL, r, a.reg, b.reg)
                scale = a.scale + b.scale if a.scale >= 0 and b.scale >= 0 else scale
            elif n.op == N_FLOORDIV:
                p.emit(OP_IDIV, r, a.reg, b.reg)
                scale = -1
            elif n.op == N_MOD:
                p.emit(OP_IMOD, r, a.reg, b.reg)
            elif n.op == N_LEAST:
                p.emit(OP_IMIN, r, a.reg, b.reg)
            elif n.op == N_GREATEST:
                p.emit(OP_IMAX, r, a.reg, b.reg)
            elif compare:
                p.emit(OP_ICMP, r, a.reg, b.reg, n.op)      # N_LT .. N_NE are the kernel's OP_LT .. OP_NE
            elif n.op == N_AND:
                p.emit(OP_AND, r, a.reg, b.reg)
            elif n.op == N_OR:
                p.emit(OP_OR, r, a.reg, b.reg)
            else:
                raise Error("unknown expression operator " + String(n.op))
            stack.append(_Slot(r, compare or logical, -1 if compare or logical else scale, False))
    if len(stack) != 1:
        raise Error("malformed expression")
    return stack[0].copy()


def _null_mask(mut p: Program, table_col: Int) -> Int:
    """The program column whose dtype makes OP_ISNULL read `table_col`'s validity bitmap (two more entries hold the
    bitmap's address and its first bit)."""
    if p.null_of[table_col] >= 0:
        return p.null_of[table_col]
    var at = len(p.cols)
    p.cols.append(table_col)                       # the mask column (its own entry is never read)
    p.dtypes.append(Int32(DT_NULLMASK | ((at + 1) << 8)))
    p.cols.append(-1 - table_col)                  # -1 - c: the validity bitmap of column c
    p.dtypes.append(DT_I64)
    p.cols.append(-1_000_000)                      # the bit base (0: one chunk starting at row 0)
    p.dtypes.append(DT_I64)
    p.null_of[table_col] = at
    return at


def _valid_of(mut p: Program, e: Expr, t: SharedTable) raises -> Int:
    """An int register: 1 where every column e reads is non-null, else 0. -1 when e reads no nullable column."""
    var out = -1
    for name in e.columns():
        var c = t.column(name)
        if t.null_count(c) == 0:
            continue
        var isnull = p.reg()
        p.emit(OP_ISNULL, isnull, _null_mask(p, c))
        var i = p.reg()
        p.emit(OP_F2I, i, isnull)
        var one = p.reg()
        p.emit(OP_LOAD_CONST, one, p.const(1))
        var v = p.reg()
        p.emit(OP_ISUB, v, one, i)
        if out < 0:
            out = v
        else:
            var both = p.reg()
            p.emit(OP_IMUL, both, out, v)
            out = both
    return out


def _same_expr(a: Expr, b: Expr) -> Bool:
    if len(a.nodes) != len(b.nodes):
        return False
    for i in range(len(a.nodes)):
        if a.nodes[i].op != b.nodes[i].op or a.nodes[i].value != b.nodes[i].value or a.nodes[i].name != b.nodes[i].name:
            return False
    return True


def _sources(aggs: List[Agg]) -> List[Int]:
    """Per aggregate: -1 for a plain one, else the number of its value source. Holistic aggregates over the same
    expression share a source: the kernel hook materializes its (group, value) pairs once."""
    var out = List[Int](length=len(aggs), fill=-1)
    var first = List[Int]()                  # the aggregate that opened each source
    for i in range(len(aggs)):
        if aggs[i].kind < A_DISTINCT:
            continue
        var s = -1
        for k in range(len(first)):
            if _same_expr(aggs[first[k]].arg, aggs[i].arg):
                s = k
                break
        if s < 0:
            first.append(i)
            s = len(first) - 1
        out[i] = s
    return out^


comptime CTL_REG = 1                     # the register holding the control block's address (see `_mat_hook`)
comptime SRC_WORDS = 11                  # control block words per value source
comptime CTL_PLACEHOLDER = Int64(0x5A5A5A5A5A5A5A5A)


def _lower(t: SharedTable, nodes: List[PlanNode], force_counts: Bool, masked: Bool) raises -> Program:
    """The plan as one kernel program. `force_counts` (a frame's query): every non-count aggregate carries its non-null
    count (an edit can null a cell of a column the shared data has no nulls in). `masked`: the predicate also skips
    the shared rows the session changed (the frame mask, filled in by `_run_cpu`)."""
    var p = Program(t.columns())
    var filters = List[Expr]()
    var agg_node = -1
    for i in range(len(nodes)):
        ref node = nodes[i]
        if node.kind == P_FILTER:
            filters.append(node.exprs[0].copy())
        elif node.kind == P_AGGREGATE:
            agg_node = i
        elif node.kind != P_SCAN:
            raise Error("plan node kind " + String(node.kind) + " is not supported yet")
    if agg_node < 0 or len(nodes[agg_node].aggs) == 0:
        raise Error("a query needs at least one aggregate (.agg(...))")
    ref an = nodes[agg_node]
    p.groups = an.groups
    if p.groups < 1:
        raise Error("groups must be at least 1, got " + String(p.groups))
    p.src_of = _sources(an.aggs)
    for s in p.src_of:
        p.n_src = max(p.n_src, s + 1)
    p.mat = p.n_src > 0
    p.src_vreg = List[Int](length=p.n_src, fill=-1)
    p.src_valid = List[Int](length=p.n_src, fill=-1)
    p.src_float = List[Bool](length=p.n_src, fill=False)
    p.src_scale = List[Int](length=p.n_src, fill=0)
    if p.mat:                                            # register 1: the address of the control block the hook reads
        p.section = 1
        p.ctl_const = len(p.consts)
        p.consts.append(CTL_PLACEHOLDER)
        var cr = p.reg()
        p.emit(OP_LOAD_CONST, cr, p.ctl_const)
        if p.keep(cr, False) != CTL_REG:
            raise Error("internal: the control register moved")
        p.reset_temps()

    # predicate: every filter, each false where it reads a null; the key in range and not null. Each part is ANDed
    # into one kept register as it completes, so its temporaries are free for the next.
    p.section = 0
    var acc = -1
    for f in filters:
        if not f.is_bool():
            raise Error("a filter must be a comparison (==, <, between, &, |, ~), not a value")
        var r = _compile_expr(p, f, t)
        var v = _valid_of(p, f, t)
        if v >= 0:
            var vf = p.reg()
            p.emit(OP_ICMP, vf, v, _const_reg(p, 1), 24)            # valid == 1, as a boolean
            var both = p.reg()
            p.emit(OP_AND, both, r, vf)
            r = both
        acc = _and_into(p, acc, r)
    if len(an.exprs) > 0:
        p.keyed = True
        ref key = an.exprs[0]
        if key.is_bool():
            raise Error("a group key must be a value, not a comparison")
        var kv = _valid_of(p, key, t)                      # a null key: the row is left out
        if kv >= 0:
            var vf = p.reg()
            p.emit(OP_ICMP, vf, kv, _const_reg(p, 1), 24)
            acc = _and_into(p, acc, vf)
        # (the key's range is checked per selected row by `_key_from_reg0`: a key formula is never evaluated on a
        # row the filters left out, so it cannot raise there)
        # the body: the key into register 0, where `_key_from_reg0` reads it per block
        p.section = 1
        var kslot = _compile_typed(p, key, t)
        if kslot.fl:
            raise Error("a group key must be a whole number 0 .. groups-1: a FLOAT has no dense numbering (group by a float with SQL, which hashes it)")
        var kb = kslot.reg
        p.emit(OP_IMAX, 0, kb, kb)
        p.reset_temps()
        p.section = 0
    elif p.groups != 1:
        raise Error("groups=" + String(p.groups) + " needs a group_by key")
    elif p.mat:                                          # no group_by: one group, key 0 in register 0 (the hook reads it)
        p.keyed = True
        p.section = 1
        p.emit(OP_LOAD_CONST, 0, p.const(0))
        p.section = 0
    if masked:                                           # the frame's changed shared rows: the delta adds them
        var isnull = p.reg()
        p.emit(OP_ISNULL, isnull, _frame_mask(p))
        var keep = p.reg()
        p.emit(OP_NOT, keep, isnull)
        acc = _and_into(p, acc, keep)
    p.pred_reg = acc
    p.reset_temps()

    # aggregates: one kernel slot per sum / min / max, one more per count(e) / mean over nullable columns
    for a in an.aggs:
        p.kinds.append(a.kind)
        p.slot_of.append(-1)
        p.count_slot_of.append(-1)
        p.agg_float.append(False)
        p.nan_slot_of.append(-1)
        var i = len(p.kinds) - 1
        if a.kind == A_COUNT:
            continue
        if a.kind >= A_DISTINCT:                         # a holistic aggregate: its value and validity registers, once per source
            var src = p.src_of[i]
            if p.src_vreg[src] < 0:
                if a.arg.is_bool():
                    raise Error("an aggregate needs a value, not a comparison")
                p.section = 1
                var hv = _valid_of(p, a.arg, t)
                if hv >= 0:
                    p.src_valid[src] = p.keep(hv, False)
                var vs = _compile_typed(p, a.arg, t)
                p.src_vreg[src] = p.keep(vs.reg, False)
                p.src_float[src] = vs.fl
                p.src_scale[src] = vs.scale if vs.scale > 0 else 0
                p.reset_temps()
            continue
        p.section = 1
        var valid = _valid_of(p, a.arg, t)
        if valid < 0 and force_counts:
            valid = _const_reg(p, 1)
        if a.kind != A_COUNT:
            if valid >= 0:                               # the rows this aggregate counts (non-null values)
                var kept_valid = p.keep(valid, False)
                p.section = 2
                p.emit(OP_AGG_ISUM, p.n_slots, kept_valid)
                p.section = 1
                p.count_slot_of[i] = p.n_slots
                p.n_slots += 1
            if a.kind == A_COUNT_OF:
                p.reset_temps()
                continue
        var vslot = _compile_typed(p, a.arg, t)
        var v = vslot.reg
        if a.arg.is_bool():
            raise Error("an aggregate needs a value, not a comparison")
        if vslot.fl:                                     # a float64 value (F-157)
            p.agg_float[i] = True
            var vf = -1                                  # 1.0 where the value is non-null
            if valid >= 0:
                vf = p.reg()
                p.emit(OP_I2F, vf, valid)
            if a.kind == A_SUM or a.kind == A_MEAN:      # null rows add 0.0; the kernel's float sum has one defined order (F-064)
                var m = v
                if vf >= 0:
                    var z = _const_reg(p, 0)
                    m = p.reg()
                    p.emit(OP_SELECT, m, vf, v, z)
                var kept = p.keep(m, False)
                p.reset_temps()
                p.section = 2
                p.emit(OP_AGG_SUM, p.n_slots, kept)
                p.repro = True
            else:                                        # min / max: a NaN and a null row are swapped for the identity; the NaNs are counted
                var ident = _const_reg(p, Int64(Float64(from_bits=UInt64(0x7FF0000000000000) if a.kind == A_MIN else UInt64(0xFFF0000000000000)).to_bits()))
                var isn = p.reg()
                p.emit(25, isn, v, v, 0)                 # != (IEEE): 1.0 only for NaN
                var m = p.reg()
                p.emit(OP_SELECT, m, isn, ident, v)
                if vf >= 0:
                    p.emit(OP_SELECT, m, vf, m, ident)
                    var zero = _const_reg(p, 0)
                    p.emit(OP_SELECT, isn, vf, isn, zero)    # a null row is not a NaN
                var ni = p.reg()
                p.emit(OP_F2I, ni, isn)
                var kept = p.keep(m, False)
                var kept_nan = p.keep(ni, False)
                p.reset_temps()
                p.section = 2
                p.emit(OP_AGG_ISUM, p.n_slots, kept_nan)
                p.nan_slot_of[i] = p.n_slots
                p.n_slots += 1
                p.emit(OP_AGG_MIN if a.kind == A_MIN else OP_AGG_MAX, p.n_slots, kept)
            p.slot_of[i] = p.n_slots
            p.n_slots += 1
            continue
        if valid >= 0:                                   # skip null rows: value * valid (+ the identity * isnull)
            var kept = p.reg()
            p.emit(OP_IMUL, kept, v, valid)
            v = kept
            if a.kind == A_MIN or a.kind == A_MAX:
                var one = _const_reg(p, 1)
                var isnull = p.reg()
                p.emit(OP_ISUB, isnull, one, valid)
                var ident = _const_reg(p, Int64.MAX if a.kind == A_MIN else Int64.MIN)
                var fill = p.reg()
                p.emit(OP_IMUL, fill, ident, isnull)
                var both = p.reg()
                p.emit(OP_IADD, both, v, fill)
                v = both
        var op = OP_AGG_IMIN if a.kind == A_MIN else OP_AGG_IMAX if a.kind == A_MAX else OP_AGG_ISUM
        var kept = p.keep(v, False)
        p.reset_temps()
        p.section = 2
        p.emit(op, p.n_slots, kept)
        p.slot_of[i] = p.n_slots
        p.n_slots += 1
    if p.n_slots == 0:                                   # only count(): one dummy sum keeps the kernel happy
        p.section = 1
        var z = p.keep(_const_reg(p, 0), False)
        p.section = 2
        p.emit(OP_AGG_ISUM, 0, z)
        p.n_slots = 1
    return p^


def _and_into(mut p: Program, acc: Int, r: Int) raises -> Int:
    """acc AND r in the predicate's kept register (made from r the first time); temporaries are free after."""
    if acc < 0:
        var k = p.keep(r, True)
        p.reset_temps()
        return k
    p.emit(OP_AND, acc, acc, r)
    p.reset_temps()
    return acc


def _frame_mask(mut p: Program) -> Int:
    """The program column whose OP_ISNULL reads the frame mask (bit cleared: a shared row the session changed)."""
    var at = len(p.cols)
    p.cols.append(-5_000_000)                      # the mask column (never read)
    p.dtypes.append(Int32(DT_NULLMASK | ((at + 1) << 8)))
    p.cols.append(-4_000_000)                      # the mask bitmap's address (`_run_cpu`)
    p.dtypes.append(DT_I64)
    p.cols.append(-1_000_000)                      # bit base 0
    p.dtypes.append(DT_I64)
    return at


def _const_reg(mut p: Program, v: Int64) raises -> Int:
    var r = p.reg()
    p.emit(OP_LOAD_CONST, r, p.const(v))
    return r


# ── running ──────────────────────────────────────────────────────────────────────────────────────────

def _key_from_reg0(chunk: Int, row0: Int, sel: IP, nsel: Int, use_sel: Bool, dst: IP, regs: FP) -> Bool:
    """The group ids of a block's selected rows from register 0 (int64 bits). A key outside 0 .. 2^31-1 becomes -1;
    the kernel's accumulators skip any id outside 0 .. groups-1, so out-of-range keys are left out (the app's rule)."""
    var k = LP(unsafe_from_address=Int(regs))
    for j in range(nsel):
        var v = k[j]
        dst[j] = Int32(v) if v >= 0 and v <= Int64(Int32.MAX) else -1
    return True


@always_inline
def _worker_slot(ctl: LP, regs: FP) -> Int:
    """The calling worker's number (0, 1, 2 ... in the order workers first reach the hook): its register file's address
    is its identity for the whole run, entered in a table in the control block the first time. -1 if the table is full."""
    var table = LP(unsafe_from_address=Int(ctl[4]))
    var counter = LP(unsafe_from_address=Int(ctl[3]))
    var mine = Int64(Int(regs))
    for i in range(min(Int(counter[0]), Int(ctl[5]))):
        if table[i] == mine:
            return i
    var i = Int(Atomic[Scalar[DType.int64]].fetch_add(counter, 1))
    if i >= Int(ctl[5]):
        return -1
    table[i] = mine
    return i


def _mat_hook(chunk: Int, row0: Int, sel: IP, nsel: Int, use_sel: Bool, dst: IP, regs: FP) -> Bool:
    """`_key_from_reg0`, and the materializing of the holistic aggregates' pairs. Register CTL_REG holds the address
    of the control block (Int64 words): [0] sources, [1] first row of the window, [2] groups, [3] worker counter,
    [4] worker table, [5] its size, then per source SRC_WORDS words: [value register, validity register or -1, group
    ids address, values address, per-block counts address, direct, k, heaps address, sizes address, rank address, float].
    For each source the block's selected rows whose group is in range and whose value is not null are written, packed,
    at entry (row0 - window start) of the source's arrays, and their number at the block's place in the counts;
    block b of the window starts at row `start + b * VEC`, so no two blocks (or threads) share an entry. A `direct`
    source (one top_k, few groups) writes nothing: the values go straight into the calling worker's own heaps."""
    var key = LP(unsafe_from_address=Int(regs))
    for j in range(nsel):
        var v = key[j]
        dst[j] = Int32(v) if v >= 0 and v <= Int64(Int32.MAX) else -1
    var ctl = LP(unsafe_from_address=Int(LP(unsafe_from_address=Int(regs + CTL_REG * VEC))[0]))
    var groups = ctl[2]
    var base = row0 - Int(ctl[1])
    var me = -2
    for s in range(Int(ctl[0])):
        var c = ctl + (6 + SRC_WORDS * s)
        var vals = LP(unsafe_from_address=Int(regs + Int(c[0]) * VEC))
        if c[5] != 0:
            if me == -2:
                me = _worker_slot(ctl, regs)
            if me < 0:
                return False
            var k = Int(c[6])
            var heap = LP(unsafe_from_address=Int(c[7])) + (me * Int(groups) + 0) * k
            var size = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(c[8])) + me * Int(groups)
            var rank = LP(unsafe_from_address=Int(c[9])) if c[9] != 0 else key
            var isf = c[10] != 0
            if c[1] < 0:
                for j in range(nsel):
                    var g = key[j]
                    if g >= 0 and g < groups:
                        var v = rank[Int(vals[j])] if c[9] != 0 else float_key(vals[j]) if isf else vals[j]
                        size[Int(g)] = heap_offer(heap + Int(g) * k, size[Int(g)], k, v)
            else:
                var valid = LP(unsafe_from_address=Int(regs + Int(c[1]) * VEC))
                for j in range(nsel):
                    var g = key[j]
                    if g >= 0 and g < groups and valid[j] != 0:
                        var v = rank[Int(vals[j])] if c[9] != 0 else float_key(vals[j]) if isf else vals[j]
                        size[Int(g)] = heap_offer(heap + Int(g) * k, size[Int(g)], k, v)
            continue
        var gp = IP(unsafe_from_address=Int(c[2])) + base
        var vp = LP(unsafe_from_address=Int(c[3])) + base
        var o = 0
        if c[1] < 0:
            for j in range(nsel):
                var g = key[j]
                gp[o] = Int32(g)
                vp[o] = vals[j]
                o += Int((g >= 0) & (g < groups))
        else:
            var valid = LP(unsafe_from_address=Int(regs + Int(c[1]) * VEC))
            for j in range(nsel):
                var g = key[j]
                gp[o] = Int32(g)
                vp[o] = vals[j]
                o += Int((g >= 0) & (g < groups) & (valid[j] != 0))
        if c[10] != 0:                                   # float64 values: held as order-preserving keys from here on
            for q in range(o):
                vp[q] = float_key(vp[q])
        IP(unsafe_from_address=Int(c[4]))[base // VEC] = Int32(o)
    return True


def _run_cpu(t: SharedTable, p: Program, mask_addr: Int, start: Int = 0, end: Int = -1, ctl_addr: Int = 0, workers: Int = 0) raises -> QueryResult:
    """The kernel over rows [start, end) (all by default). `ctl_addr`: the control block of a holistic query
    (`_mat_hook`), whose pairs the kernel materializes while it aggregates."""
    var n = t.rows()
    var stop = n if end < 0 else end
    var code = List[Int32]()
    var header: List[Int] = [MAGIC, p.n_regs, 0, len(p.pred) // 5, len(p.body) // 5, len(p.aggs) // 5,
                             p.pred_reg, len(p.cols), 0, 0, 0, 0, 0, 0, 0, 0]
    for h in header:
        code.append(Int32(h))
    for w in p.pred:
        code.append(w)
    for w in p.body:
        code.append(w)
    for w in p.aggs:
        code.append(w)
    var consts = List[Float64]()
    for v in p.consts:                       # int64 bits in the float64 pool: LOAD_CONST copies them exactly
        consts.append(Float64(from_bits=UInt64(v)))
    if len(consts) == 0:
        consts.append(0.0)
    if ctl_addr != 0:
        consts[p.ctl_const] = Float64(from_bits=UInt64(ctl_addr))
    var set_bytes: List[UInt8] = p.set_bytes.copy() if len(p.set_bytes) > 0 else [0]      # (F-150: a SQL program's IN sets)
    var set_meta: List[Int64] = p.set_meta.copy() if len(p.set_meta) > 0 else [0, 0, 0]
    var tbl: List[Int64] = [0, Int64(n)]
    for c in p.cols:
        if c >= 0:
            tbl.append(Int64(Int(t.values(c).unsafe_ptr())))
        elif c == -1_000_000 or c == -5_000_000:
            tbl.append(0)
        elif c == -4_000_000:
            tbl.append(Int64(mask_addr))
        else:
            tbl.append(Int64(t._table[].cols[-1 - c].validity_address()))
    var dtypes = p.dtypes.copy()
    if len(dtypes) == 0:
        dtypes.append(DT_I64)
    var out_vals = List[Float64](length=p.n_slots * p.groups, fill=0.0)
    var out_cnt = List[Int64](length=p.groups, fill=0)
    var threads = workers if workers > 0 else core_count()
    var rc: Int
    if ctl_addr != 0:
        rc = pipeline_keyed[type_of(_mat_hook)](Int(code.unsafe_ptr()), Int(consts.unsafe_ptr()), Int(set_bytes.unsafe_ptr()),
                                            Int(set_meta.unsafe_ptr()), Int(tbl.unsafe_ptr()), 1, Int(dtypes.unsafe_ptr()),
                                            p.groups, start, stop, Int(out_vals.unsafe_ptr()), Int(out_cnt.unsafe_ptr()),
                                            threads, _mat_hook, p.repro)
    elif p.keyed:
        rc = pipeline_keyed[type_of(_key_from_reg0)](Int(code.unsafe_ptr()), Int(consts.unsafe_ptr()), Int(set_bytes.unsafe_ptr()),
                                            Int(set_meta.unsafe_ptr()), Int(tbl.unsafe_ptr()), 1, Int(dtypes.unsafe_ptr()),
                                            p.groups, start, stop, Int(out_vals.unsafe_ptr()), Int(out_cnt.unsafe_ptr()),
                                            threads, _key_from_reg0, p.repro)
    else:
        rc = pipeline_plain(Int(code.unsafe_ptr()), Int(consts.unsafe_ptr()), Int(set_bytes.unsafe_ptr()),
                            Int(set_meta.unsafe_ptr()), Int(tbl.unsafe_ptr()), 1, Int(dtypes.unsafe_ptr()),
                            0, 1, start, stop, Int(out_vals.unsafe_ptr()), Int(out_cnt.unsafe_ptr()), threads, p.repro)
    _ = code^, consts^, set_bytes^, set_meta^, tbl^, dtypes^      # the kernel read them: keep them past the call
    if rc == E_OVERFLOW:
        raise Error(overflow_message(p.has_float, p.conv_text()))
    if rc != 0:
        raise Error("the CPU kernel failed with code " + String(rc))
    var r = _make_result(p, out_vals, out_cnt)
    _ = out_vals^
    return r^


comptime JOB_RANGE_ROWS = 1 << 21        # F-175: rows a background job's plain query runs between two cancel checks


def _run_cpu_job(t: SharedTable, p: Program, mask_addr: Int, threads: Int, job: JobCtl) raises -> QueryResult:
    """`_run_cpu` over all rows; for a background job over a big table, in row ranges whose partial results are merged (as the
    windows of `_run_mat` and the row groups of a scan are), so a cancel is seen between ranges and progress moves. Integer
    sums, counts, min and max merge exactly, so the answer is the one call's. A query with a float sum or a reproducible plan
    (`p.repro`) keeps its one call: its float64 sum has one defined order, which ranges would change. Any error from the ranges
    (a partial total that passed int64 where the whole does not) is settled by the one call."""
    var n = t.rows()
    var ops = _slot_ops(p)
    var exact = job.active() and not p.repro
    for o in ops:
        exact = exact and o < A_FSUM
    var step = max(JOB_RANGE_ROWS, 8 * p.groups) // MORSEL_ROWS * MORSEL_ROWS
    if not exact or n <= 2 * step:
        var r = _run_cpu(t, p, mask_addr, 0, -1, 0, threads)
        job.advance(n)
        return r^
    var at = 0
    try:
        var acc = QueryResult()
        while at < n:
            job.check()
            var end = min(n, at + step)
            var r = _run_cpu(t, p, mask_addr, at, end, 0, threads)
            if at == 0:
                acc = r^
            else:
                _merge_partial(acc, r, ops)
            job.advance(end - at)
            at = end
        for i in range(len(acc.kinds)):
            if acc.kinds[i] == A_SUM:
                acc.sums = acc.slots[acc.slot_of[i]].copy()
                break
        return acc^
    except e:
        job.check()
        var r = _run_cpu(t, p, mask_addr, 0, -1, 0, threads)
        job.advance(n - at)
        return r^


def _make_result(p: Program, out_vals: List[Float64], out_cnt: List[Int64]) -> QueryResult:
    """The kernel's output (slot words are int64 bits, per slot then group; counts per group) as a QueryResult."""
    var r = QueryResult()
    r.kinds = p.kinds.copy()
    r.slot_of = p.slot_of.copy()
    r.count_slot_of = p.count_slot_of.copy()
    r.agg_float = p.agg_float.copy()
    r.nan_slot_of = p.nan_slot_of.copy()
    r.hol_of = List[Int](length=len(p.kinds), fill=-1)
    var bits = out_vals.unsafe_ptr().bitcast[Int64]()
    for s in range(p.n_slots):
        var vals = List[Int]()
        for g in range(p.groups):
            vals.append(Int(bits[s * p.groups + g]))
        r.slots.append(vals^)
    for g in range(p.groups):
        r.counts.append(Int(out_cnt[g]))
    for i in range(len(p.kinds)):
        if p.kinds[i] == A_SUM:
            r.sums = r.slots[p.slot_of[i]].copy()
            break
    return r^


# ── holistic aggregates: windows of the kernel's pairs into the states (F-148) ─────────────────────

comptime MORSEL_ROWS = 65536
comptime WINDOW_BYTES = 256 * 1024 * 1024       # what a window's pairs may take: 12 bytes a row per value source


def _slot_ops(p: Program) -> List[Int]:
    """How each kernel slot merges across partial results: A_SUM (also for counts), A_MIN or A_MAX."""
    var ops = List[Int](length=p.n_slots, fill=A_SUM)
    for k in range(len(p.aggs) // 5):
        var op = Int(p.aggs[k * 5])
        if op == OP_AGG_IMIN:
            ops[Int(p.aggs[k * 5 + 1])] = A_MIN
        elif op == OP_AGG_IMAX:
            ops[Int(p.aggs[k * 5 + 1])] = A_MAX
        elif op == OP_AGG_SUM:
            ops[Int(p.aggs[k * 5 + 1])] = A_FSUM
        elif op == OP_AGG_MIN:
            ops[Int(p.aggs[k * 5 + 1])] = A_FMIN
        elif op == OP_AGG_MAX:
            ops[Int(p.aggs[k * 5 + 1])] = A_FMAX
    return ops^


def fold_float(op: Int, a: Int, b: Int) -> Int:
    """Two float64 slot words (their bits) combined: A_FSUM adds, A_FMIN / A_FMAX take the smaller / larger."""
    var x = Float64(from_bits=UInt64(Int64(a)))
    var y = Float64(from_bits=UInt64(Int64(b)))
    var r = x + y if op == A_FSUM else (y if y < x else x) if op == A_FMIN else (y if y > x else x)
    return Int(Int64(r.to_bits()))


def slot_identity(op: Int) -> Int:
    """A slot's starting word for its merge op: 0 for sums, the extreme values (as int64, or +-inf as float64 bits) for extremes."""
    if op == A_MIN:
        return Int(Int64.MAX)
    if op == A_MAX:
        return Int(Int64.MIN)
    if op == A_FMIN:
        return Int(Int64(Float64(from_bits=UInt64(0x7FF0000000000000)).to_bits()))
    if op == A_FMAX:
        return Int(Int64(Float64(from_bits=UInt64(0xFFF0000000000000)).to_bits()))
    return 0


def _merge_partial(mut acc: QueryResult, r: QueryResult, ops: List[Int]) raises:
    """Folds the partial result `r` (a window, a row group) into `acc`: sums and counts add (a total past int64
    raises, as one run does), min / max combine."""
    for s in range(len(ops)):
        for x in range(len(acc.slots[s])):
            var a = acc.slots[s][x]
            var b = r.slots[s][x]
            if ops[s] >= A_FSUM:
                acc.slots[s][x] = fold_float(ops[s], a, b)
            elif ops[s] == A_MIN:
                acc.slots[s][x] = min(a, b)
            elif ops[s] == A_MAX:
                acc.slots[s][x] = max(a, b)
            else:
                var y = Int64(a) + Int64(b)
                if ((Int64(a) ^ y) & (Int64(b) ^ y)) < 0:
                    raise Error(OVERFLOW_MESSAGE)
                acc.slots[s][x] = Int(y)
    for x in range(len(acc.counts)):
        acc.counts[x] += r.counts[x]


def _run_mat(t: SharedTable, p: Program, mask_addr: Int, mut hs: HolStates, window_rows: Int = 0, threads: Int = 0, job: JobCtl = JobCtl()) raises -> QueryResult:
    """A query with count_distinct / percentile / top_k over `t`: the kernel runs over windows of rows, aggregating
    the plain aggregates as always (partial results merged) and, through `_mat_hook`, writing the (group, value) pairs
    of each value source, which `hs` takes in (the states keep what later windows must merge with). A window is as many
    rows as keep its pairs under WINDOW_BYTES, a multiple of the morsel size so blocks line up."""
    var n = t.rows()
    var rows = max(MORSEL_ROWS, (window_rows if window_rows > 0 else WINDOW_BYTES // (12 * p.n_src)) // MORSEL_ROWS * MORSEL_ROWS)
    var ops = _slot_ops(p)
    var acc = QueryResult()
    var workers = core_count() + 4                       # register files the hook may meet (the kernel starts at most core_count())
    var direct = List[Int](length=p.n_src, fill=-1)      # per source: its only top_k's number, when the hook feeds it directly
    for s in range(p.n_src):
        direct[s] = hs.direct_topk(s, workers)
    var done = 0
    var first = True
    while first or done < n:
        job.check()                                      # F-175: between windows
        var end = min(n, done + rows)
        var nblk = (end - done + VEC - 1) // VEC
        var gids = List[List[Int32]]()
        var vals = List[List[Int64]]()
        var cnts = List[List[Int32]]()
        var heaps = List[List[Int64]]()                  # per source: workers x groups x k words for a direct one
        var sizes = List[List[Int]]()
        var table = List[Int64](length=workers, fill=0)
        var counter = List[Int64](length=1, fill=0)
        var ctl = List[Int64]()
        ctl.append(Int64(p.n_src))
        ctl.append(Int64(done))
        ctl.append(Int64(p.groups))
        ctl.append(Int64(Int(counter.unsafe_ptr())))
        ctl.append(Int64(Int(table.unsafe_ptr())))
        ctl.append(Int64(workers))
        for s in range(p.n_src):
            var is_direct = direct[s] >= 0
            var kk = hs.specs[direct[s]].k if is_direct else 0
            gids.append(List[Int32](unsafe_uninit_length=1 if is_direct else max(end - done, 1)))
            vals.append(List[Int64](unsafe_uninit_length=1 if is_direct else max(end - done, 1)))
            cnts.append(List[Int32](length=1 if is_direct else max(nblk, 1), fill=0))
            heaps.append(List[Int64](length=workers * p.groups * kk if is_direct else 1, fill=0))
            sizes.append(List[Int](length=workers * p.groups if is_direct else 1, fill=0))
            ctl.append(Int64(p.src_vreg[s]))
            ctl.append(Int64(p.src_valid[s]))
            ctl.append(Int64(Int(gids[s].unsafe_ptr())))
            ctl.append(Int64(Int(vals[s].unsafe_ptr())))
            ctl.append(Int64(Int(cnts[s].unsafe_ptr())))
            ctl.append(Int64(1 if is_direct else 0))
            ctl.append(Int64(kk))
            ctl.append(Int64(Int(heaps[s].unsafe_ptr())))
            ctl.append(Int64(Int(sizes[s].unsafe_ptr())))
            ctl.append(Int64(hs.rank_of(s) if is_direct else 0))
            ctl.append(Int64(1 if p.src_float[s] else 0))
        var rw = _run_cpu(t, p, mask_addr, done, end, Int(ctl.unsafe_ptr()), threads)
        var used_workers = min(Int(counter[0]), workers)
        for s in range(p.n_src):
            if direct[s] >= 0:
                hs.merge_workers(direct[s], Int(heaps[s].unsafe_ptr()), Int(sizes[s].unsafe_ptr()), used_workers)
            else:
                hs.absorb(s, window_blocks(Int(gids[s].unsafe_ptr()), Int(vals[s].unsafe_ptr()), Int(cnts[s].unsafe_ptr()), nblk, VEC))
        _ = ctl^, gids^, vals^, cnts^, heaps^, sizes^, table^, counter^     # the kernel and the states read them by address
        if first:
            acc = rw^
        else:
            _merge_partial(acc, rw, ops)
        first = False
        job.advance(end - done)
        done = end
    for i in range(len(acc.kinds)):                      # `sums` was copied from the first window's slot
        if acc.kinds[i] == A_SUM:
            acc.sums = acc.slots[acc.slot_of[i]].copy()
            break
    return acc^


def _build_states(orig: List[PlanNode], nodes: List[PlanNode], base: SharedTable, extra: List[List[String]],
                  scan: Optional[ParquetScan] = None, max_rows: Int = -1, var floats: List[Bool] = List[Bool](), var scales: List[Int] = List[Int]()) raises -> HolStates:
    """The states for the plan's holistic aggregates (none: an empty set). `nodes` are the plan with a frame's formulas
    and merges expanded (they decide which aggregates share a value source), `orig` the plan as built (to see which
    values are a plain TEXT column, whose percentile / top_k order is the words' alphabetical order, not the codes').
    `extra`: a frame's session words per column (codes after the shared dictionary). `scan`: the words are the
    file's (its dictionaries are built on first use, not kept in `base`)."""
    var at = -1
    for i in range(len(nodes)):
        if nodes[i].kind == P_AGGREGATE:
            at = i
    if at < 0:
        return HolStates(1, List[HolSpec](), List[List[Int64]]())
    var src_of = _sources(nodes[at].aggs)
    var nsrc = 0
    for s in src_of:
        nsrc = max(nsrc, s + 1)
    var ranks = List[List[Int64]]()
    for _ in range(nsrc):
        ranks.append(List[Int64]())
    var specs = List[HolSpec]()
    for i in range(len(src_of)):
        if src_of[i] < 0:
            continue
        ref a = nodes[at].aggs[i]
        specs.append(HolSpec(i, a.kind, a.p, a.k, src_of[i]))
        if is_spread(a.kind) or a.kind == A_QEDGE:         # numbers only (bins also dates): a word or a flag has no spread, and no order worth cutting
            for name in orig[at].aggs[i].arg.columns():
                var c = base.index(name)
                if c >= 0 and (base.kind(c) == TEXT or base.kind(c) == BOOL or (base.kind(c) == DATE and a.kind != A_QEDGE)):
                    raise Error(("quantile bins" if a.kind == A_QEDGE else "stddev / variance") + " need numbers" + (" or dates" if a.kind == A_QEDGE else "") + ": column `" + name + "` is not one")
            continue
        if a.kind == A_DISTINCT or len(ranks[src_of[i]]) > 0:
            continue
        ref oa = orig[at].aggs[i].arg
        for name in oa.columns():
            var c = base.index(name)
            if c >= 0 and base.kind(c) == TEXT:
                if len(oa.nodes) != 1:
                    raise Error("percentile / top_k of text needs the column itself, not an expression over `" + name + "`")
                var words = List[String]()
                if scan:
                    for k in range(scan.value().word_count(c)):
                        words.append(scan.value().text_of(c, k))
                else:
                    for w in base.dictionary(c):
                        words.append(w)
                if c < len(extra):
                    for w in extra[c]:
                        words.append(w)
                ranks[src_of[i]] = text_ranks(words)
    return HolStates(nodes[at].groups, specs^, ranks^, max_rows, floats^, scales^)


def _finish_holistic(mut r: QueryResult, mut hs: HolStates) raises:
    """Puts the states' answers into the result."""
    var res = hs.finish()
    r.hvals = res.vals.copy()
    r.htop = res.top.copy()
    r.htop_off = res.top_off.copy()
    r.hcount = res.counts.copy()
    for h in range(len(hs.specs)):
        r.hol_of[hs.specs[h].agg] = h
        r.hfloat.append(hs.floats[hs.specs[h].src])


# ── running on the GPU ───────────────────────────────────────────────────────────────────────────────

def _reg_fields(op: Int) -> Int:
    """Which of an instruction's operands d, a, b name registers: bit 0 = d, bit 1 = a, bit 2 = b."""
    if op == OP_LOAD_CONST or op == OP_LOADI or op == OP_ISNULL:
        return 1
    if op == OP_F2I or op == OP_NOT:
        return 3
    if op >= OP_AGG_ISUM:
        return 2                                          # an aggregate: d is the slot, a the value register
    return 7


def _compact_registers(pred_reg: Int, mut code: List[Int32]) -> Tuple[Int, Int]:
    """The program uses registers 0 .. 47 (kept ones from the bottom, temporaries from the top). A GPU thread block
    holds its registers in shared memory (256 rows x 8 bytes each), so renumber the registers actually used densely
    in `code` (all instructions, predicate then body then aggregates). Returns (registers used, new predicate register)."""
    var used = List[Bool](length=MAX_REGS, fill=False)
    used[0] = True
    if pred_reg >= 0:
        used[pred_reg] = True
    for i in range(len(code) // 5):
        var f = _reg_fields(Int(code[i * 5]))
        for k in range(3):
            if (f >> k) & 1 != 0:
                used[Int(code[i * 5 + 1 + k])] = True
    var map = List[Int32](length=MAX_REGS, fill=0)
    var n = 0
    for r in range(MAX_REGS):
        if used[r]:
            map[r] = Int32(n)
            n += 1
    for i in range(len(code) // 5):
        var f = _reg_fields(Int(code[i * 5]))
        for k in range(3):
            if (f >> k) & 1 != 0:
                code[i * 5 + 1 + k] = map[Int(code[i * 5 + 1 + k])]
    return (n, Int(map[pred_reg]) if pred_reg >= 0 else -1)


comptime OVERFLOW_MESSAGE = "a sum or an expression passed the int64 range (about 9.2e18)"
comptime FOOTER_LIE = "footer statistics wrong: "              # the prefix `run_sql` reads to run a query again without a file's statistics


def overflow_message(has_float: Bool, conv: String = "") -> String:
    """The kernel's overflow flag: an int64 total or result past its range (an integer past 2^53 read as a double rounds, F-190: it no
    longer raises, so `has_float` changes nothing); F-156: and a CAST that met a value it cannot convert (`conv` names them)."""
    var base = OVERFLOW_MESSAGE
    if conv.startswith(FOOTER_LIE):
        return conv
    if conv.byte_length() > 0:
        return "a value could not be converted (" + conv + "), or: " + base
    return base


struct _QueryBuffers(Movable):
    """The device buffers one GPU query needs. Making them is the only step that can fail for lack of memory, so
    AUTO's fallback wraps exactly this constructor."""
    var code: DeviceBuffer[DType.int32]
    var consts: DeviceBuffer[DType.float64]
    var cols: DeviceBuffer[DType.int64]
    var dtypes: DeviceBuffer[DType.int32]
    var ident: DeviceBuffer[DType.float64]
    var set: DeviceBuffer[DType.int64]
    var vals: DeviceBuffer[DType.float64]
    var cnt: DeviceBuffer[DType.int64]
    var scratch: DeviceBuffer[DType.float64]
    var ovf: DeviceBuffer[DType.int64]
    var mask: DeviceBuffer[DType.uint8]

    def __init__(out self, ctx: DeviceContext, n_code: Int, n_cols: Int, n_dtypes: Int, slots: Int, groups: Int,
                 scratch_words: Int, mask_len: Int) raises:
        self.code = ctx.enqueue_create_buffer[DType.int32](n_code)
        self.consts = ctx.enqueue_create_buffer[DType.float64](MAX_CONSTS)
        self.cols = ctx.enqueue_create_buffer[DType.int64](n_cols)
        self.dtypes = ctx.enqueue_create_buffer[DType.int32](n_dtypes)
        self.ident = ctx.enqueue_create_buffer[DType.float64](slots)
        self.set = ctx.enqueue_create_buffer[DType.int64](4)             # no string sets: a dummy
        self.vals = ctx.enqueue_create_buffer[DType.float64](slots * groups)
        self.cnt = ctx.enqueue_create_buffer[DType.int64](groups)
        self.scratch = ctx.enqueue_create_buffer[DType.float64](scratch_words)
        self.ovf = ctx.enqueue_create_buffer[DType.int64](1)
        self.mask = ctx.enqueue_create_buffer[DType.uint8](max(mask_len, 1))


comptime OOM_PREFIX = "the GPU could not "


def _is_oom(why: String) -> Bool:
    """`fallback` says an allocation failed (as opposed to the kernel declining the program or the size check)."""
    return why.startswith(OOM_PREFIX)


def _gpu_scratch_words(ctx: DeviceContext, p: Program, n_regs: Int, code_len: Int) -> Int:
    """The per-block partial rows the launch's mode merges (`scratch_rows`), in float64 words: the one figure both the allocation
    and the budget check use."""
    var n_blocks = max(ctx.default_device_info.sm_count, 1)
    var mode = _choose_mode(p.groups, p.n_slots, n_regs, code_len, 0, True)
    return max(scratch_rows(mode, p.groups, p.n_slots, n_blocks) * (p.n_slots + 1) * p.groups, 1)


def _gpu_fixed_bytes(ctx: DeviceContext, p: Program, mask_len: Int) -> Int:
    """F-189: the device bytes one launch of `p` takes besides the columns: the sizes `_QueryBuffers` allocates."""
    var ins = p.pred.copy()
    ins.extend(p.body.copy())
    ins.extend(p.aggs.copy())
    var compact = _compact_registers(p.pred_reg, ins)
    var code_len = GPU_HDR + len(ins)
    var words = _gpu_scratch_words(ctx, p, compact[0], code_len)
    return (4 * code_len + 8 * MAX_CONSTS + 8 * max(len(p.cols), 1) + 4 * max(len(p.dtypes), 1) + 8 * p.n_slots + 32
            + 8 * p.n_slots * p.groups + 8 * p.groups + 8 * words + 8 + max(mask_len, 1))


def _gpu_columns_used(t: SharedTable, p: Program) -> Tuple[List[Bool], List[Bool]]:
    """Which table columns the program reads the values of, and which the validity bitmaps of."""
    var data = List[Bool](length=t.columns(), fill=False)
    var valid = List[Bool](length=t.columns(), fill=False)
    for c in p.cols:
        if c == -1_000_000 or c == -5_000_000 or c == -4_000_000:
            continue
        if c >= 0:
            data[c] = True
        else:
            valid[-1 - c] = True
    return (data^, valid^)


def _range_bytes(rows: Int, nd: Int, nv: Int, masked: Bool) -> Int:
    """Device bytes of `rows` rows of `nd` value columns, `nv` bitmaps and a mask (a bitmap slice carries 16 spare bytes)."""
    return nd * 8 * rows + (nv + (1 if masked else 0)) * (rows // 8 + 16)


def _mask_slice(mask: List[UInt8], r0: Int, rows: Int) -> List[UInt8]:
    """Rows [r0, r0 + rows) of the frame mask from bit 0, with 8 spare bytes (always a mask: the program was lowered with one)."""
    if len(mask) == 0:
        return List[UInt8]()
    var out = List[UInt8](length=(rows + 7) // 8 + 8, fill=0xFF)
    var src = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(mask.unsafe_ptr()))
    var dst = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(out.unsafe_ptr()))
    _copy_bits(dst, 0, src, r0, rows)
    return out^


def _run_gpu(dt: DeviceTable, t: SharedTable, p: Program, mask: List[UInt8], strict: Bool, mut fallback: String) raises -> Optional[QueryResult]:
    var data = List[Int]()
    var valid = List[Int]()
    for c in range(t.columns()):
        data.append(dt.data_address(c))
        valid.append(dt.validity_address(c))
    return _run_gpu_cols(dt.context(), data, valid, t.rows(), t, p, mask, strict, fallback, 1)


def _run_gpu_cols(ctx: DeviceContext, data_addr: List[Int], valid_addr: List[Int], n_rows: Int, t: SharedTable, p: Program,
                  mask: List[UInt8], strict: Bool, mut fallback: String, piece: Int = 0) raises -> Optional[QueryResult]:
    """The lowered program on the device table (kernels/pipeline_gpu.mojo, keyed mode: the group id is register 0).
    Same answers as `_run_cpu`: exact int64 totals, the overflow flag raises the same error.

    Two things send the query to the CPU (None, and `fallback` says which): the device cannot hold the query's buffers
    (an allocation failed: F-189, on `GPU` too, the caller reruns the piece on the CPU), or the kernel declined the
    program (needs more shared memory than a block has: -5; on an Apple GPU also more groups than its atomic-free
    modes hold, -6, or an instruction its Int64 word does not run), which `strict` (device=GPU) raises instead. A launch
    error, an illegal address or an int64 overflow is a bug or an answer: it always raises. `piece` (> 0): the number
    `MXFRAME_GPU_FAIL_PIECE` is compared with (a test knob that makes this allocation fail)."""
    var ins = p.pred.copy()
    ins.extend(p.body.copy())
    ins.extend(p.aggs.copy())
    var compact = _compact_registers(p.pred_reg, ins)
    var n_regs = compact[0]
    var code = List[Int32]()
    var header: List[Int] = [MAGIC, n_regs, 0, len(p.pred) // 5, len(p.body) // 5, len(p.aggs) // 5,
                             compact[1], len(p.cols), 0, 0, 0, 0, 0, 0, 0, 0]
    for h in header:
        code.append(Int32(h))
    code.extend(ins^)
    var consts = List[Float64](length=MAX_CONSTS, fill=0.0)
    for k in range(len(p.consts)):                       # int64 bits in the float64 pool, copied exactly
        consts[k] = Float64(from_bits=UInt64(p.consts[k]))
    var A = p.n_slots
    var G = p.groups
    var agg_base = (len(p.pred) + len(p.body)) // 5
    comptime if APPLE_GPU:                               # F-183: the Int64-word kernel runs integer programs only
        if not int_word_program_ok(code, (len(code) - GPU_HDR) // 5, agg_base):
            var why = "the Apple GPU kernel runs integer programs only (this one has an instruction its Int64 word does not run)"
            if strict:
                raise Error("device GPU: " + why)
            fallback = why
            return None
    var n_blocks = max(ctx.default_device_info.sm_count, 1)
    # the per-block partial rows exist only in the modes that merge them (0, 3, 4, and 5 on an Apple GPU): one row of
    # (slots + 1) x groups per block the launch may use (`scratch_rows`). The atomic modes write the output directly.
    var scratch_words = _gpu_scratch_words(ctx, p, n_regs, len(code))
    var made = Optional[_QueryBuffers](None)
    try:
        if piece > 0 and gpu_fail_piece() == piece:
            raise Error(INJECTED_OOM)
        made = _QueryBuffers(ctx, len(code), max(len(p.cols), 1), max(len(p.dtypes), 1), A, G, scratch_words, len(mask))
    except e:
        fallback = OOM_PREFIX + "allocate the query's buffers: " + String(e)
        return None
    var b = made.take()
    ref d_code = b.code
    ref d_consts = b.consts
    ref d_cols = b.cols
    ref d_dt = b.dtypes
    ref d_ident = b.ident
    ref d_set = b.set
    ref d_vals = b.vals
    ref d_cnt = b.cnt
    ref d_scratch = b.scratch
    ref d_ovf = b.ovf
    if len(mask) > 0:
        ctx.enqueue_copy(dst_buf=b.mask, src_ptr=mask.unsafe_ptr())
    var cols = List[Int64]()
    for c in p.cols:
        if c >= 0:
            cols.append(Int64(data_addr[c]))
        elif c == -1_000_000 or c == -5_000_000:
            cols.append(0)
        elif c == -4_000_000:
            cols.append(Int64(Int(b.mask.unsafe_ptr())))
        else:
            cols.append(Int64(valid_addr[-1 - c]))
    if len(cols) == 0:
        cols.append(0)
    var dtypes = p.dtypes.copy()
    if len(dtypes) == 0:
        dtypes.append(DT_I64)
    var ident = List[Float64](length=A, fill=0.0)
    for a in range(len(p.aggs) // 5):
        var op = Int(p.aggs[a * 5])
        var slot = Int(p.aggs[a * 5 + 1])
        if op == OP_AGG_IMIN:
            ident[slot] = Float64(from_bits=UInt64(Int64.MAX))
        elif op == OP_AGG_IMAX:
            ident[slot] = Float64(from_bits=UInt64(Int64.MIN))
    ctx.enqueue_copy(dst_buf=d_code, src_ptr=code.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_consts, src_ptr=consts.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_cols, src_ptr=cols.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_dt, src_ptr=dtypes.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ident, src_ptr=ident.unsafe_ptr())
    ctx.enqueue_memset(d_set, 0)
    ctx.enqueue_memset(d_ovf, 0)
    var rc = run_pipeline_gpu(ctx, Int(d_code.unsafe_ptr()), Int(d_consts.unsafe_ptr()), Int(d_set.unsafe_ptr()),
                              Int(d_set.unsafe_ptr()), Int(d_cols.unsafe_ptr()), Int(d_dt.unsafe_ptr()),
                              -1 if p.keyed else 0, G, A, 0, n_rows, Int(d_vals.unsafe_ptr()),
                              Int(d_cnt.unsafe_ptr()), Int(d_ident.unsafe_ptr()), Int(d_scratch.unsafe_ptr()),
                              n_regs, len(code), agg_base, n_blocks, 1, 0, Int(d_ovf.unsafe_ptr()), True)
    if rc != 0:
        if not strict and rc == -5:
            fallback = "the GPU kernel declined the program (code -5: it needs more shared memory than a block has)"
            return None                                   # the CPU answers
        if not strict and rc == -6:                       # (asked for only on an Apple GPU, where atomics are out)
            fallback = "the GPU kernel declined the program (code -6: its groups or registers do not fit the atomic-free modes)"
            return None
        raise Error("the GPU kernel failed with code " + String(rc) +
                    (" (the query needs more GPU shared memory than a block has: " + String(n_regs) + " registers, " + String(A) + " slots, " + String(G) + " groups)"
                     if rc == -5 else (" (the query's groups or registers do not fit the GPU's atomic-free modes: " + String(n_regs) + " registers, " + String(A) + " slots, " + String(G) + " groups)" if rc == -6 else "")))
    var out_vals = List[Float64](length=A * G, fill=0.0)
    var out_cnt = List[Int64](length=G, fill=0)
    var ovf = List[Int64](length=1, fill=0)
    ctx.enqueue_copy(dst_ptr=out_vals.unsafe_ptr(), src_buf=d_vals)
    ctx.enqueue_copy(dst_ptr=out_cnt.unsafe_ptr(), src_buf=d_cnt)
    ctx.enqueue_copy(dst_ptr=ovf.unsafe_ptr(), src_buf=d_ovf)
    ctx.synchronize()
    _ = code^, consts^, cols^, dtypes^, ident^
    _ = b^   # Mojo frees at last use: the device buffers must outlive the kernel
    if ovf[0] != 0:
        raise Error(OVERFLOW_MESSAGE)
    return _make_result(p, out_vals, out_cnt)


def _count_true(xs: List[Bool]) -> Int:
    var n = 0
    for x in xs:
        if x:
            n += 1
    return n


def _adopt_ext(t: SharedTable, ext: List[Int], rows: Int):
    """A scan's decoded row group lives in the caller's pinned buffers (`ext`), not in `t`: a CPU run gets the values in `t`."""
    for c in range(t._table[].columns()):
        var v = List[Int64](unsafe_uninit_length=rows)
        var src = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=ext[c])
        for i in range(rows):
            v[i] = src[i]
        t._table[].cols[c].values = v^


def _gpu_ranges(addr: Int, ctx: DeviceContext, t: SharedTable, ext: List[Int], p: Program, mask: List[UInt8], rows: Int,
                mut fallback: String, piece_base: Int, job: JobCtl) raises -> QueryResult:
    """F-189: the program over rows [0, rows) of `t` (a whole table, or a scan's decoded row group whose values are in `ext`) in row
    ranges that fit the GPU budget, uploaded and run one after the other on the device `addr` (a `_ScanContext`) is on. Each
    range's partial is merged with `_merge_partial`, as the windows of the CPU's job runs and a scan's row groups are, so the
    answer is the CPU's. A range the device cannot hold (an allocation failed) runs on the CPU, and `fallback` says why;
    a partial total that passes int64 (where the whole may not) is settled by one CPU run of all the rows. The kernel
    declining the program raises (`device=GPU`). `piece_base` >= 0: range n is piece `piece_base` + n for MXFRAME_GPU_FAIL_PIECE.
    The range is a multiple of 64 rows, so a bitmap slice starts on a byte."""
    var usable = gpu_budget(ctx)
    var masked = len(mask) > 0
    var used = _gpu_columns_used(t, p)
    var nd = _count_true(used[0])
    var nv = _count_true(used[1])
    var fixed = _gpu_fixed_bytes(ctx, p, 0)
    var per_row_bits = 64 * nd + nv + (1 if masked else 0)
    var step = rows
    var all_cpu = String()
    if per_row_bits > 0:
        var room = usable - fixed - (nv + (1 if masked else 0)) * 16
        step = room * 8 // per_row_bits // 64 * 64 if room > 0 else 0
        while step > 64 and fixed + _range_bytes(step, nd, nv, masked) > usable:
            step -= 64
        step = min(step, rows)
        if step < 64 and rows > step:
            var why = ("the GPU budget of " + human_bytes(usable) + " leaves no room for 64 rows of the " + String(nd) +
                       " columns this query reads, after its " + human_bytes(fixed) + " of buffers")
            if fixed + _range_bytes(64, nd, nv, masked) > gpu_capacity(ctx):
                raise Error("device GPU: " + why)
            all_cpu = why^                               # (the GPU is merely full right now: all the rows run on the CPU)
            step = rows
    var ops = _slot_ops(p)
    var acc = _make_result(p, List[Float64](length=p.n_slots * p.groups, fill=0.0), List[Int64](length=p.groups, fill=0))
    for sl in range(p.n_slots):
        acc.slots[sl] = List[Int](length=p.groups, fill=slot_identity(ops[sl]))
    var mask_addr = Int(mask.unsafe_ptr()) if masked else 0
    var adopted = False
    var n = 0
    var gpu_n = 0
    var r0 = 0
    try:
        while r0 < rows:
            job.check()
            var m = min(step, rows - r0)
            n += 1
            var why = String()
            var got = Optional[QueryResult](None)
            try:
                if all_cpu.byte_length() > 0:
                    raise Error(all_cpu)
                if piece_base >= 0 and gpu_fail_piece() == piece_base + n:
                    raise Error(INJECTED_OOM)
                _stage_range(addr, t, used[0], used[1], r0, m, ext)
            except e:
                why = OOM_PREFIX + "hold a piece of " + String(m) + " rows: " + String(e)
            if why.byte_length() == 0:
                var data = List[Int]()
                var valid = List[Int]()
                for c in range(t.columns()):
                    data.append(_stage_data(addr, c))
                    valid.append(_stage_valid(addr, c))
                var ms = _mask_slice(mask, r0, m)
                got = _run_gpu_cols(ctx, data, valid, m, t, p, ms, True, why)
                _ = ms^
                _ = data^, valid^
            if got:
                _merge_partial(acc, got.value(), ops)
                gpu_n += 1
            else:
                if fallback.byte_length() == 0:
                    fallback = why^
                if len(ext) > 0 and not adopted:
                    _adopt_ext(t, ext, rows)
                    adopted = True
                _merge_partial(acc, _run_cpu(t, p, mask_addr, r0, r0 + m), ops)
            job.advance(m)
            r0 += m
    except e:
        if String(e) != OVERFLOW_MESSAGE:
            raise e^
        if len(ext) > 0 and not adopted:                 # (a partial total passed int64: all the rows in one CPU run decide)
            _adopt_ext(t, ext, rows)
        var whole = _run_cpu(t, p, mask_addr, 0, rows)
        whole.fallback_reason = "a partial total passed the int64 range; the CPU settled the whole"
        whole.pieces = n
        return whole^
    for i in range(len(acc.kinds)):
        if acc.kinds[i] == A_SUM:
            acc.sums = acc.slots[acc.slot_of[i]].copy()
            break
    acc.device = GPU if gpu_n == n else CPU
    acc.pieces = n
    acc.gpu_pieces = gpu_n
    acc.fallback_reason = fallback.copy()
    return acc^


def _run_gpu_pieces(dt: DeviceTable, t: SharedTable, p: Program, mask: List[UInt8], mut fallback: String, job: JobCtl) raises -> QueryResult:
    """`compute(GPU)` on a table over the GPU budget (no device copy): `_gpu_ranges` over all its rows on the DeviceTable's device."""
    var ctx = dt.context()
    var own = _ScanContext(ctx)
    var r = _gpu_ranges(own.addr, ctx, t, List[Int](), p, mask, t.rows(), fallback, 0, job)
    if r.gpu_pieces < r.pieces and _is_oom(fallback):
        dt.remember_oom(gpu_free_bytes(ctx))
    _ = own^                                                  # (the buffers live to here)
    return r^


# ── a frame's changed and appended rows, on the CPU ──────────────────────────────────────────────────

def _add_delta(mut r: QueryResult, t: SharedTable, nodes: List[PlanNode], o: Overlay, mut hs: HolStates) raises:
    """Adds the session's changed (live) and appended rows to the kernel's result over the unchanged shared rows,
    with the kernel's rules: a filter reading a null is false, a null or out-of-range key leaves the row out, an
    aggregate skips a null value, totals are exact (an overflow raises)."""
    var filters = List[Expr]()
    var keyed = False
    var key = Expr(0)
    var groups = 1
    var aggs = List[Agg]()
    for node in nodes:
        if node.kind == P_FILTER:
            filters.append(node.exprs[0].copy())
        elif node.kind == P_AGGREGATE:
            groups = node.groups
            aggs = node.aggs.copy()
            if len(node.exprs) > 0:
                keyed = True
                key = node.exprs[0].copy()
    var src_of = _sources(aggs)                          # holistic aggregates: the delta rows' pairs per value source
    var pair_g = List[List[Int32]]()
    var pair_v = List[List[Int64]]()
    var first_of = List[Int]()
    for i in range(len(aggs)):
        if src_of[i] == len(first_of):
            first_of.append(i)
            pair_g.append(List[Int32]())
            pair_v.append(List[Int64]())
    for raw in o.delta:
        var keep = True
        for f in filters:
            var v = _eval_row(f, t, raw, o.null_marker)
            if v[1] or v[0] == 0:
                keep = False
                break
        if not keep:
            continue
        var g = 0
        if keyed:
            var k = _eval_row(key, t, raw, o.null_marker)
            if k[1] or k[0] < 0 or k[0] >= Int64(groups):
                continue
            g = Int(k[0])
        r.counts[g] += 1
        for sidx in range(len(first_of)):
            var hv = _eval_row(aggs[first_of[sidx]].arg, t, raw, o.null_marker)
            if not hv[1]:
                pair_g[sidx].append(Int32(g))
                pair_v[sidx].append(float_key(hv[0]) if hs.floats[sidx] else hv[0])
        for i in range(len(aggs)):
            if aggs[i].kind == A_COUNT or aggs[i].kind >= A_DISTINCT:
                continue
            var v = _eval_row_t(aggs[i].arg, t, raw, o.null_marker)
            if v[1]:
                continue
            if r.count_slot_of[i] >= 0:
                r.slots[r.count_slot_of[i]][g] += 1
            if aggs[i].kind == A_COUNT_OF:
                continue
            var s = r.slot_of[i]
            if len(r.agg_float) > i and r.agg_float[i]:                    # a float64 aggregate: the delta rows fold in row order
                var fx = Float64(from_bits=UInt64(Int64(r.slots[s][g])))
                var fv = Float64(from_bits=UInt64(v[0]))
                if aggs[i].kind == A_SUM or aggs[i].kind == A_MEAN:
                    r.slots[s][g] = Int(Int64((fx + fv).to_bits()))
                elif fv != fv:                                              # a NaN is counted, not folded (it is above every number)
                    r.slots[r.nan_slot_of[i]][g] += 1
                elif aggs[i].kind == A_MIN and fv < fx:
                    r.slots[s][g] = Int(v[0])
                elif aggs[i].kind == A_MAX and fv > fx:
                    r.slots[s][g] = Int(v[0])
                continue
            var x = Int64(r.slots[s][g])
            if aggs[i].kind == A_MIN:
                r.slots[s][g] = Int(min(x, v[0]))
            elif aggs[i].kind == A_MAX:
                r.slots[s][g] = Int(max(x, v[0]))
            else:
                var y = x + v[0]
                if ((x ^ y) & (v[0] ^ y)) < 0:
                    raise Error("a sum or an expression passed the int64 range (about 9.2e18)")
                r.slots[s][g] = Int(y)
    for sidx in range(len(first_of)):
        hs.absorb_rows(sidx, pair_g[sidx], pair_v[sidx])
    for i in range(len(r.kinds)):
        if r.kinds[i] == A_SUM:
            r.sums = r.slots[r.slot_of[i]].copy()
            break


# ── a Parquet scan by morsels: row ranges of row groups on every core, blocks of rows (F-149) ────────

comptime SCAN_BLOCK_ROWS = 32768
# AUTO runs a streamed file on the GPU when the file has at least this many rows and the plan costs the CPU more than the
# trip to the device. A row group is decoded on the CPU either way (into pinned memory when the GPU takes it) and then
# copied over the bus: that costs per column read (8 bytes a row each); the CPU kernel costs per instruction of the plan,
# spread over the cores. Measured on 20M rows (F-146 in docs/plan.md), a light plan (28 instructions, 4 columns) and a heavy
# one (90 instructions, 3 columns): 12 cores: the CPU wins both; 4 cores: light is a tie, heavy goes to the GPU by 1.9x;
# 2 cores: light goes to the GPU by 1.25x, heavy by 2.7x. So the GPU is picked when
#     instructions a row > AUTO_GPU_OPS_PER_CORE_COLUMN x columns read x cores.
comptime AUTO_GPU_SCAN_MIN_ROWS = 5_000_000
comptime AUTO_GPU_OPS_PER_CORE_COLUMN = 3
comptime MORSELS_AUTO = 0
comptime MORSELS_ON = 1
comptime MORSELS_OFF = 2
comptime MORSELS_PER_WORKER = 3      # about this many row-range morsels a worker, so the last round is not a long tail


def _use_morsels(mode: Int, rows: Int, block: Int) -> Bool:
    """Morsels unless forbidden; on AUTO only when there is more than a few blocks of rows and more than one core."""
    if mode == MORSELS_ON:
        return rows > 0
    if mode == MORSELS_OFF:
        return False
    return core_count() > 1 and rows >= 8 * block


def _identity_partial(n_slots: Int, groups: Int, ops: List[Int]) -> QueryResult:
    var acc = QueryResult()
    for s in range(n_slots):
        acc.slots.append(List[Int](length=groups, fill=slot_identity(ops[s])))
    acc.counts = List[Int](length=groups, fill=0)
    return acc^


struct _MorselRun(Movable):
    var tgroup: List[Int]            # per morsel (in order: row group, then row range): its row group, as an index of `todo`
    var parts: List[QueryResult]     # its partial result (the layout of `_lower(.., force_counts)`)
    var derrs: List[String]          # its decode error ("" when none)
    var kerrs: List[String]          # its first kernel error (the rest of its blocks are decoded but not aggregated)
    var nulls: List[List[Int]]       # the nulls of each column it read
    var gerrs: List[String]          # per row group: an error opening it

    def __init__(out self):
        self.tgroup = List[Int]()
        self.parts = List[QueryResult]()
        self.derrs = List[String]()
        self.kerrs = List[String]()
        self.nulls = List[List[Int]]()
        self.gerrs = List[String]()


def _morsel_task(plan: _GroupPlan, gfile: Int, r0: Int, r1: Int, used: List[Int], nodes: List[PlanNode], n_slots: Int,
                 groups: Int, ops: List[Int], block: Int, sm: _ScanMask, mut part: QueryResult, mut tally: List[Int],
                 mut kernel_error: String, job: JobCtl = JobCtl()) raises:
    """Rows [r0, r1) of row group `gfile` on one worker: decode block after block, run the kernel (one thread) on each,
    fold the block's partial into the morsel's in block order. A kernel error is kept and the remaining blocks are still
    decoded (not aggregated), so a decode error later in the range is the one raised, as when the group was decoded first.
    `sm` (F-147): the frame mask, of which the morsel takes its rows' slice."""
    var reader = _GroupReader(plan, r0, r1, block)
    var bt = Table.empty()
    for k in range(len(used)):
        _ = bt.add_column(plan.scan.name(used[k]), plan.scan.kind(used[k]), scale=plan.scan.scale(used[k]))
        bt.cols[k].values = List[Int64](unsafe_uninit_length=block)
    var shared = bt^.share()
    part = _identity_partial(n_slots, groups, ops)
    var gm = _range_mask(sm, gfile, r0, r1)          # F-147: the morsel's slice of the frame mask; r0 and a block start at multiples of 8
    var gm_addr = Int(gm.unsafe_ptr()) if len(gm) > 0 else 0
    while True:
        if job.cancelled():                              # F-175: between blocks; the caller raises "cancelled" after the round
            break
        var b0 = reader.at
        var rows = reader.next_block(shared._table[])
        if rows == 0:
            break
        job.advance(rows)
        if kernel_error.byte_length() > 0:
            continue
        try:
            var r = _run_cpu(shared, _lower(shared, nodes, True, gm_addr != 0), gm_addr + (b0 - r0) // 8 if gm_addr != 0 else 0, 0, rows, 0, 1)
            _merge_partial(part, r, ops)
            for k in range(len(used)):
                tally[k] += shared._table[].cols[k].nulls
        except e:
            kernel_error = String(e)
    _ = gm^


def _scan_morsels(scan: ParquetScan, nodes: List[PlanNode], used: List[Int], todo: List[Int], n_slots: Int, groups: Int,
                  ops: List[Int], block: Int, sm: _ScanMask, job: JobCtl = JobCtl()) raises -> _MorselRun:
    """The row groups of `todo`: with at least a group a core, one morsel per group; with fewer, in waves of two groups a
    core: the wave's groups are opened in parallel (headers walked, dictionaries decoded once and then shared read only),
    then cut into row-range morsels of whole blocks (about MORSELS_PER_WORKER a worker) that all the cores take in turn.
    The morsels come back in (group, range) order."""
    var run = _MorselRun()
    var workers = core_count()
    var batch = max(1, 2 * workers)
    for _ in range(len(todo)):
        run.gerrs.append(String())
    if len(todo) >= workers:
        # Every worker has a row group of its own at least: one morsel per group, opened by the worker that runs it (no
        # barrier between opening and aggregating, no row group read twice at a range boundary).
        for i in range(len(todo)):
            run.tgroup.append(i)
            run.parts.append(QueryResult())
            run.derrs.append(String())
            run.kerrs.append(String())
            run.nulls.append(List[Int](length=len(used), fill=0))
        var sp1 = Int(UnsafePointer(to=scan))
        var up1 = Int(UnsafePointer(to=used))
        var tp1 = Int(UnsafePointer(to=todo))
        var np1 = Int(UnsafePointer(to=nodes))
        var op1 = Int(UnsafePointer(to=ops))
        var pa1 = Int(run.parts.unsafe_ptr())
        var de1 = Int(run.derrs.unsafe_ptr())
        var ke1 = Int(run.kerrs.unsafe_ptr())
        var na1 = Int(run.nulls.unsafe_ptr())
        var ge1 = Int(run.gerrs.unsafe_ptr())
        var sm1 = Int(UnsafePointer(to=sm))
        def whole_group(i: Int) {imm}:
            if job.cancelled():
                return
            var plan = _GroupPlan(Pointer[ParquetScan, MutUntrackedOrigin](unsafe_from_address=sp1)[].copy(),
                                 Pointer[List[Int], MutUntrackedOrigin](unsafe_from_address=up1)[])
            try:
                plan = _open_plan(Pointer[ParquetScan, MutUntrackedOrigin](unsafe_from_address=sp1)[],
                                 Pointer[List[Int], MutUntrackedOrigin](unsafe_from_address=tp1)[][i],
                                 Pointer[List[Int], MutUntrackedOrigin](unsafe_from_address=up1)[])
            except e:
                (Pointer[String, MutUntrackedOrigin](unsafe_from_address=ge1) + i)[] = String(e)
                return
            try:
                _morsel_task(plan, Pointer[List[Int], MutUntrackedOrigin](unsafe_from_address=tp1)[][i], 0, plan.rows,
                             Pointer[List[Int], MutUntrackedOrigin](unsafe_from_address=up1)[],
                             Pointer[List[PlanNode], MutUntrackedOrigin](unsafe_from_address=np1)[], n_slots, groups,
                             Pointer[List[Int], MutUntrackedOrigin](unsafe_from_address=op1)[], block,
                             Pointer[_ScanMask, MutUntrackedOrigin](unsafe_from_address=sm1)[],
                             (Pointer[QueryResult, MutUntrackedOrigin](unsafe_from_address=pa1) + i)[],
                             (Pointer[List[Int], MutUntrackedOrigin](unsafe_from_address=na1) + i)[],
                             (Pointer[String, MutUntrackedOrigin](unsafe_from_address=ke1) + i)[], job)
            except e:
                (Pointer[String, MutUntrackedOrigin](unsafe_from_address=de1) + i)[] = String(e)
        par_n(whole_group, len(todo), min(len(todo), workers))
        return run^
    var w0 = 0
    while w0 < len(todo):
        job.check()
        var nb = min(batch, len(todo) - w0)
        var plans = List[_GroupPlan]()
        for _ in range(nb):
            plans.append(_GroupPlan(scan.copy(), used))
        var sp = Int(UnsafePointer(to=scan))
        var up = Int(UnsafePointer(to=used))
        var tp = Int(UnsafePointer(to=todo))
        var pl = Int(plans.unsafe_ptr())
        var ge = Int(run.gerrs.unsafe_ptr())
        var base = w0
        def open_job(i: Int) {imm}:
            if job.cancelled():
                return
            try:
                (Pointer[_GroupPlan, MutUntrackedOrigin](unsafe_from_address=pl) + i)[] = _open_plan(
                    Pointer[ParquetScan, MutUntrackedOrigin](unsafe_from_address=sp)[],
                    Pointer[List[Int], MutUntrackedOrigin](unsafe_from_address=tp)[][base + i],
                    Pointer[List[Int], MutUntrackedOrigin](unsafe_from_address=up)[])
            except e:
                (Pointer[String, MutUntrackedOrigin](unsafe_from_address=ge) + base + i)[] = String(e)
        if nb == 1:
            open_job(0)
        else:
            par_n(open_job, nb, min(nb, workers))
        var tg = List[Int]()                             # the wave's morsels
        var tr0 = List[Int]()
        var tr1 = List[Int]()
        var per_group = max(1, (MORSELS_PER_WORKER * workers + nb - 1) // nb)
        for i in range(nb):
            if run.gerrs[w0 + i].byte_length() > 0:
                continue
            var rows = plans[i].rows
            var blocks = (rows + block - 1) // block
            var n = max(1, min(per_group, blocks))
            for j in range(n):
                var lo = (blocks * j // n) * block
                var hi = min(rows, (blocks * (j + 1) // n) * block)
                if hi > lo:
                    tg.append(i)
                    tr0.append(lo)
                    tr1.append(hi)
        var nt = len(tg)
        var first = len(run.parts)
        for t in range(nt):
            run.tgroup.append(w0 + tg[t])
            run.parts.append(QueryResult())
            run.derrs.append(String())
            run.kerrs.append(String())
            run.nulls.append(List[Int](length=len(used), fill=0))
        var np = Int(UnsafePointer(to=nodes))
        var op = Int(UnsafePointer(to=ops))
        var smp = Int(UnsafePointer(to=sm))
        var pa = Int(run.parts.unsafe_ptr())
        var de = Int(run.derrs.unsafe_ptr())
        var ke = Int(run.kerrs.unsafe_ptr())
        var na = Int(run.nulls.unsafe_ptr())
        var tga = Int(tg.unsafe_ptr())
        var r0a = Int(tr0.unsafe_ptr())
        var r1a = Int(tr1.unsafe_ptr())
        def task(t: Int) {imm}:
            if job.cancelled():
                return
            var gi = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=tga)[t]
            var at = first + t
            try:
                _morsel_task((Pointer[_GroupPlan, MutUntrackedOrigin](unsafe_from_address=pl) + gi)[],
                             Pointer[List[Int], MutUntrackedOrigin](unsafe_from_address=tp)[][w0 + gi],
                             Pointer[Int, MutUntrackedOrigin](unsafe_from_address=r0a)[t],
                             Pointer[Int, MutUntrackedOrigin](unsafe_from_address=r1a)[t],
                             Pointer[List[Int], MutUntrackedOrigin](unsafe_from_address=up)[],
                             Pointer[List[PlanNode], MutUntrackedOrigin](unsafe_from_address=np)[], n_slots, groups,
                             Pointer[List[Int], MutUntrackedOrigin](unsafe_from_address=op)[], block,
                             Pointer[_ScanMask, MutUntrackedOrigin](unsafe_from_address=smp)[],
                             (Pointer[QueryResult, MutUntrackedOrigin](unsafe_from_address=pa) + at)[],
                             (Pointer[List[Int], MutUntrackedOrigin](unsafe_from_address=na) + at)[],
                             (Pointer[String, MutUntrackedOrigin](unsafe_from_address=ke) + at)[], job)
            except e:
                (Pointer[String, MutUntrackedOrigin](unsafe_from_address=de) + at)[] = String(e)
        if nt > 0:
            par_n(task, nt, min(nt, workers))
        _ = tg^
        _ = tr0^
        _ = tr1^
        _ = plans^
        w0 += nb
    return run^


def _run_part_gpu(t: SharedTable, nodes: List[PlanNode], gm: List[UInt8], strict: Bool, ctx_addr: Int,
                  mut fallback: String, rows: Int, ext: List[Int], piece: Int = 0) raises -> Optional[QueryResult]:
    """One decoded row group of a scan on the GPU: its used columns copied into the scan's device buffers (exact int64,
    validity bitmaps beside them), the program run in keyed mode and the group's partial copied back, so the device
    holds one row group of the columns the plan reads. `gm` (F-147): the frame mask slice of the group. None (with
    `fallback` saying why) when the device could not hold the group, an allocation failed, or the kernel declined the program
    and `strict` is off: the caller runs that group on the CPU. F-189: a group over the GPU budget runs in row ranges that fit
    on `GPU` (`strict`: `_gpu_ranges`) and on the CPU on AUTO; an allocation failure is the CPU's on both. The partial has
    the layout of `_run_part`'s; `.device` says whether it all ran on the GPU. `piece`: this group's number for MXFRAME_GPU_FAIL_PIECE."""
    comptime if HAS_GPU:
        var p = _lower(t, nodes, True, len(gm) > 0)
        var ctx = _scan_context(ctx_addr)
        var usable = gpu_budget(ctx)
        var fixed = _gpu_fixed_bytes(ctx, p, len(gm))
        var need = fixed + _range_bytes(rows, t.columns(), t.columns(), False)
        if need > usable:
            if not strict:
                fallback = "a row group of " + String(rows) + " rows needs " + human_bytes(need) + " on the GPU, over its budget of " + human_bytes(usable)
                return None
            return _gpu_ranges(ctx_addr, ctx, t, ext, p, gm, rows, fallback, -1, JobCtl())
        try:
            if piece > 0 and gpu_fail_piece() == piece:
                raise Error(INJECTED_OOM)
            _stage_load(ctx_addr, t, ext, rows)
        except e:
            fallback = OOM_PREFIX + "hold a row group of " + String(rows) + " rows: " + String(e)
            return None
        var data = List[Int]()
        var valid = List[Int]()
        for c in range(t.columns()):
            data.append(_stage_data(ctx_addr, c))
            valid.append(_stage_valid(ctx_addr, c))
        var g = _run_gpu_cols(ctx, data, valid, rows, t, p, gm, strict, fallback)
        if g:
            g.value().device = GPU
        return g^
    else:
        return None


def _run_part_dev(t: SharedTable, nodes: List[PlanNode], mut hs: HolStates, gm: List[UInt8], use_gpu: Bool, strict: Bool,
                  ctx_addr: Int, mut fallback: String, mut gpu_groups: Int, rows: Int, ext: List[Int], threads: Int = 0,
                  piece: Int = 0) raises -> QueryResult:
    """A scan's row group on the GPU when the scan runs there (the CPU when the group falls back, and for the groups after
    one whose allocation failed), else on the CPU. `ext`: the group's values are in these pinned buffers
    (`read_group_into`), not in `t`, which a CPU run then gets them in."""
    if use_gpu and not _is_oom(fallback):
        var g = _run_part_gpu(t, nodes, gm, strict, ctx_addr, fallback, rows, ext, piece)
        if g:
            if g.value().device == GPU:
                gpu_groups += 1
            return g.take()
        _adopt_ext(t, ext, rows)                            # the CPU takes the group: its values go into the table
    elif use_gpu:
        _adopt_ext(t, ext, rows)
    return _run_part(t, nodes, hs, Int(gm.unsafe_ptr()) if len(gm) > 0 else 0, threads)


def _run_part(t: SharedTable, nodes: List[PlanNode], mut hs: HolStates, mask_addr: Int = 0, threads: Int = 0) raises -> QueryResult:
    """One row group of a scan: the plan lowered on its own table, run on the CPU (a holistic plan feeds `hs`).
    `mask_addr` (F-147): the frame mask of this group's rows, 0 for none."""
    var p = _lower(t, nodes, True, mask_addr != 0)
    if p.mat:
        return _run_mat(t, p, mask_addr, hs, 0, threads)
    return _run_cpu(t, p, mask_addr, 0, -1, 0, threads)


# ---- F-147: a frame over a scan. The frame mask is one bitmap over the whole file; a row group (or a block of one)
# takes its slice, copied to start at bit 0, and the changed rows' values are added once after the last group.

struct _ScanMask(Copyable, Movable):
    var addr: Int                    # the frame mask's address (bit i = file row i is unchanged), 0: no row changed
    var starts: List[Int]            # the first file row of each row group

    def __init__(out self):
        self.addr = 0
        self.starts = List[Int]()


def _group_mask(m: _ScanMask, g: Int, rows: Int) -> List[UInt8]:
    """Row group g's slice of the frame mask (empty: nothing in it changed or there is no mask), with 8 spare bytes
    so a kernel reading a word past the last row stays inside."""
    var out = List[UInt8]()
    if m.addr == 0:
        return out^
    out = List[UInt8](length=(rows + 7) // 8 + 8, fill=0xFF)
    var src = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=m.addr)
    var dst = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(out.unsafe_ptr()))
    _copy_bits(dst, 0, src, m.starts[g], rows)
    if _count_bits(dst, rows) == rows:               # nothing in this group changed: no mask, no masked program
        return List[UInt8]()
    return out^


def _range_mask(m: _ScanMask, g: Int, r0: Int, r1: Int) -> List[UInt8]:
    """Rows [r0, r1) of row group g's slice of the frame mask, from bit 0 (empty: none changed there)."""
    var out = List[UInt8]()
    if m.addr == 0:
        return out^
    var rows = r1 - r0
    out = List[UInt8](length=(rows + 7) // 8 + 8, fill=0xFF)
    var src = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=m.addr)
    var dst = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(out.unsafe_ptr()))
    _copy_bits(dst, 0, src, m.starts[g] + r0, rows)
    if _count_bits(dst, rows) == rows:
        return List[UInt8]()
    return out^


# ── a Parquet scan: one row group at a time (F-140) ──────────────────────────────────────────────────

def _compute_scan(scan: ParquetScan, nodes: List[PlanNode], device: Int, orig: List[PlanNode], o: Overlay,
                  framed: Bool, morsels: Int = 0, block_rows: Int = 0, gpu_ops: Int = AUTO_GPU_OPS_PER_CORE_COLUMN, threads: Int = 0,
                  job: JobCtl = JobCtl()) raises -> QueryResult:
    """The plan over a file, a row group at a time: decode the columns the plan reads, run the CPU kernel on that
    group, merge its partial result into the running one (sums and counts add, min / max combine; a sum passing
    int64 raises, as one run does). Each group is lowered on its own table, so its null-free columns skip the null
    machinery; every group is lowered with a non-null count per aggregate (`force_counts`), so the partial results
    all have one layout, and the merged result is laid out at the end as a whole-file run would have (a count slot
    only for an aggregate over a column that holds a null anywhere in the file).

    F-147, a Frame over the scan (`framed`; `orig` is the plan as built, `nodes` with the frame's formulas and merges
    expanded, `o` its overlay): each group (or block of one) skips the rows the frame changed or removed (its slice
    of `o.mask`), and after the last group the frame's changed and appended rows are added (`_add_delta`), exactly as
    `Query.compute` does for a Frame over a table."""
    var t0 = perf_counter_ns()
    var names = List[String]()
    for node in nodes:
        for e in node.exprs:
            for nm in e.columns():
                if nm not in names:
                    names.append(nm)
        for a in node.aggs:
            for nm in a.arg.columns():
                if nm not in names:
                    names.append(nm)
    var used = List[Int]()                   # the file's columns the plan reads, in file order
    for nm in names:
        var c = scan.column(nm)
        var at = len(used)
        while at > 0 and used[at - 1] > c:
            at -= 1
        if at == 0 or used[at - 1] != c:
            used.insert(at, c)
    if len(used) == 0 and scan.columns() > 0:   # count() alone reads no column: a table's row count is its first column's
        used.append(0)
    var shape = _lower(scan.schema, nodes, True, False)         # the per-group layout; also raises plan errors up front
    # ---- F-146: the GPU for a streamed file. Each decoded row group (the pipeline of the loop below: group N+1 decodes
    # on the CPU while group N is uploaded and aggregated) goes to the device and its partial comes back to be merged in
    # row-group order; the device holds one row group of the columns the plan reads. GPU: raises where it cannot run;
    # AUTO: the GPU for a file of at least AUTO_GPU_SCAN_MIN_ROWS rows when a GPU is there, the CPU (with the reason in
    # `fallback_reason`) otherwise, and a group the device cannot hold runs on the CPU.
    var use_gpu = False
    var why = String()
    var ops_row = (len(shape.pred) + len(shape.body) + len(shape.aggs)) // 5
    if device == GPU:
        comptime if not HAS_GPU:
            raise Error("device GPU: " + NO_GPU_MESSAGE)
        if not gpu_available():
            raise Error("device GPU: no GPU found")
        if shape.mat:
            raise Error("device GPU: count_distinct / percentile / top_k run on the CPU")
        if shape.has_float:
            raise Error("device GPU: a FLOAT column or float arithmetic runs on the CPU (a float64 sum has one defined order there)")
        use_gpu = True
    elif device == AUTO and gpu_available():
        if shape.has_float:
            why = "a FLOAT column or float arithmetic runs on the CPU (a float64 sum has one defined order there)"
        elif shape.mat:
            why = "count_distinct / percentile / top_k run on the CPU"
        elif scan.rows() < AUTO_GPU_SCAN_MIN_ROWS and gpu_ops > 0:
            why = "the file has " + String(scan.rows()) + " rows, under AUTO's " + String(AUTO_GPU_SCAN_MIN_ROWS) + " for the GPU"
        elif ops_row <= gpu_ops * len(used) * core_count():
            why = "the plan runs " + String(ops_row) + " kernel instructions a row over " + String(len(used)) + " columns on " + String(core_count()) + " cores: cheaper on the CPU than moving the file to the GPU"
        else:
            use_gpu = True
    var gown = _ScanContext(use_gpu)                          # frees the device context on every way out of this function
    var gctx = gown.addr
    if use_gpu and device == AUTO:
        # F-189: AUTO keeps a file on the CPU when one row group of the columns it reads does not fit the GPU budget (GPU would cut
        # the group into row ranges), or an earlier query on this file ran out of GPU memory and not much more is free now
        comptime if HAS_GPU:
            var ctx = _scan_context(gctx)
            var usable = gpu_budget(ctx)
            var biggest = 0
            for g in range(scan.row_groups()):
                biggest = max(biggest, scan.group_rows(g))
            var need = _gpu_fixed_bytes(ctx, shape, (biggest + 7) // 8 + 8 if len(o.mask) > 0 else 0) + _range_bytes(biggest, len(used), len(used), False)
            var free = gpu_free_bytes(ctx)
            if need > usable:
                why = "a row group of " + String(biggest) + " rows needs " + human_bytes(need) + " on the GPU, over its budget of " + human_bytes(usable) + ": AUTO answers on the CPU"
            elif scan.gpu_oom_free() > 0 and free * 100 < scan.gpu_oom_free() * GPU_RETRY_FACTOR_PERCENT:
                why = ("an earlier query ran out of GPU memory with " + human_bytes(scan.gpu_oom_free()) + " free; AUTO stays on the CPU until " +
                       human_bytes(scan.gpu_oom_free() // 100 * GPU_RETRY_FACTOR_PERCENT) + " is free (" + human_bytes(free) + " now)")
            if why.byte_length() > 0:
                use_gpu = False
                gown = _ScanContext(False)
                gctx = 0
    var gpu_groups = 0
    var ops = _slot_ops(shape)                                  # how each kernel slot merges
    var acc = QueryResult()
    for s in range(shape.n_slots):
        acc.slots.append(List[Int](length=shape.groups, fill=slot_identity(ops[s])))
    acc.counts = List[Int](length=shape.groups, fill=0)
    var nulls = List[Int](length=len(used), fill=0)
    var todo = List[Int]()                   # the row groups with rows, in file order
    for g in range(scan.row_groups()):
        if scan.group_rows(g) > 0:
            todo.append(g)
    var todo_all = todo.copy()
    job.stage(STAGE_AGGREGATE)
    for g in todo:
        job.add_total(scan.group_rows(g))
    job.check()
    for c in used:                           # every dictionary the plan needs is built before groups decode beside the kernel
        scan.ensure_words(c)
    var hs = _build_states(orig, nodes, scan.schema, o.words, Optional[ParquetScan](scan.copy()), -1 if framed else scan.rows(), shape.src_float.copy(), shape.src_scale.copy())    # (text order needs the words)
    var sm = _ScanMask()                                                      # F-147
    if len(o.mask) > 0:
        sm.addr = Int(o.mask.unsafe_ptr())
        var at = 0
        for g in range(scan.row_groups()):
            sm.starts.append(at)
            at += scan.group_rows(g)
    # ---- F-149: morsel-driven scan (the one place the group-at-a-time loop below is bypassed) -----------------------
    # The row groups are cut into row-range morsels that all the cores take (`_scan_morsels`); the morsels' partials come
    # back in (row group, range) order and are merged and raised in that order, a row group at a time, as the loop below
    # would: a group's decode error before its kernel error, an earlier group's error before a later group's. A plan with
    # holistic aggregates (distinct, percentile, top_k) keeps the loop below: its states are fed in row-group order.
    var block = (block_rows if block_rows > 0 else SCAN_BLOCK_ROWS) // 8 * 8      # a multiple of 8: a block's frame mask slice starts on a byte (F-147)
    block = max(block, 64)
    if not shape.mat and not use_gpu and not shape.repro and _use_morsels(morsels, scan.rows(), block):      # (a float64 sum is folded row group by row group, in file order: the one order that does not depend on the cores)
        var run = _scan_morsels(scan, nodes, used, todo, shape.n_slots, shape.groups, ops, block, sm, job)
        job.check()                                      # a cancelled round left partial morsels: none of them is merged
        var ti = 0
        for i in range(len(todo)):
            if run.gerrs[i].byte_length() > 0:
                raise Error(run.gerrs[i])
            var first = ti
            while ti < len(run.tgroup) and run.tgroup[ti] == i:
                ti += 1
            for t in range(first, ti):                   # a decode error of the group comes before a kernel error of it
                if run.derrs[t].byte_length() > 0:
                    raise Error(run.derrs[t])
            for t in range(first, ti):
                if run.kerrs[t].byte_length() > 0:
                    raise Error(run.kerrs[t])
            for t in range(first, ti):
                _merge_partial(acc, run.parts[t], ops)
                for k in range(len(used)):
                    nulls[k] += run.nulls[t][k]
        todo = List[Int]()
    # ---- end F-149 ------------------------------------------------------------------------------------------------
    var cur = Table.empty()
    var ext_cur = List[Int]()                # (GPU) the pinned buffers the group being aggregated was decoded into
    if len(todo) > 0:
        if use_gpu:
            ext_cur = _pinned_prepare(gctx, 0, len(used), scan.group_rows(todo[0]))
            cur = scan.read_group_into(todo[0], used, ext_cur)
        else:
            cur = scan.read_group(todo[0], used)
    for ix in range(len(todo)):
        job.check()                                      # F-175: between row groups
        for k in range(len(used)):
            nulls[k] += cur.null_count(k)
        var part = cur^.share()
        var gm = _group_mask(sm, todo[ix], scan.group_rows(todo[ix]))        # F-147
        var gm_addr = Int(gm.unsafe_ptr()) if len(gm) > 0 else 0
        var r = QueryResult()
        var nxt = Table.empty()
        var pending = String()
        var ext_nxt = List[Int]()
        if use_gpu and ix + 1 < len(todo):       # the next group decodes into the other set of pinned buffers
            ext_nxt = _pinned_prepare(gctx, (ix + 1) % 2, len(used), scan.group_rows(todo[ix + 1]))
        var rows_cur = scan.group_rows(todo[ix])
        var piece_no = ix + 1                    # (MXFRAME_GPU_FAIL_PIECE counts the groups that run, from 1)
        if ix + 1 < len(todo):
            # group ix + 1 decodes while the kernel aggregates group ix (two groups in memory); a serial build runs the
            # two tasks one after the other. The partials are still merged below in row-group order.
            var e0 = String()
            var e1 = String()
            var rp = Int(UnsafePointer(to=r))
            var np = Int(UnsafePointer(to=nxt))
            var e0p = Int(UnsafePointer(to=e0))
            var e1p = Int(UnsafePointer(to=e1))
            var sp = Int(UnsafePointer(to=scan))
            var pp = Int(UnsafePointer(to=part))
            var up = Int(UnsafePointer(to=used))
            var np_nodes = Int(UnsafePointer(to=nodes))
            var hsp = Int(UnsafePointer(to=hs))
            var gmp = Int(UnsafePointer(to=gm))
            var whyp = Int(UnsafePointer(to=why))
            var ggp = Int(UnsafePointer(to=gpu_groups))
            var extc = Int(UnsafePointer(to=ext_cur))
            var extn = Int(UnsafePointer(to=ext_nxt))
            var next_group = todo[ix + 1]
            def both(i: Int) {imm}:
                if i == 0:
                    try:
                        var tbl = Pointer[SharedTable, MutUntrackedOrigin](unsafe_from_address=pp)[]
                        var nds = Pointer[List[PlanNode], MutUntrackedOrigin](unsafe_from_address=np_nodes)[].copy()
                        Pointer[QueryResult, MutUntrackedOrigin](unsafe_from_address=rp)[] = _run_part_dev(
                            tbl, nds, Pointer[HolStates, MutUntrackedOrigin](unsafe_from_address=hsp)[],
                            Pointer[List[UInt8], MutUntrackedOrigin](unsafe_from_address=gmp)[], use_gpu, device == GPU, gctx,
                            Pointer[String, MutUntrackedOrigin](unsafe_from_address=whyp)[],
                            Pointer[Int, MutUntrackedOrigin](unsafe_from_address=ggp)[], rows_cur,
                            Pointer[List[Int], MutUntrackedOrigin](unsafe_from_address=extc)[], threads, piece_no)
                    except e:
                        Pointer[String, MutUntrackedOrigin](unsafe_from_address=e0p)[] = String(e)
                else:
                    try:
                        ref scn = Pointer[ParquetScan, MutUntrackedOrigin](unsafe_from_address=sp)[]
                        if use_gpu:
                            Pointer[Table, MutUntrackedOrigin](unsafe_from_address=np)[] = scn.read_group_into(
                                next_group, Pointer[List[Int], MutUntrackedOrigin](unsafe_from_address=up)[],
                                Pointer[List[Int], MutUntrackedOrigin](unsafe_from_address=extn)[])
                        else:
                            Pointer[Table, MutUntrackedOrigin](unsafe_from_address=np)[] = scn.read_group(
                                next_group, Pointer[List[Int], MutUntrackedOrigin](unsafe_from_address=up)[])
                    except e:
                        Pointer[String, MutUntrackedOrigin](unsafe_from_address=e1p)[] = String(e)
            par_n(both, 2, 2)
            _ = len(ext_cur) + len(ext_nxt)           # (both lists are read by address inside the tasks: alive to here)
            if e0.byte_length() > 0:                 # the earlier group's error comes first
                raise Error(e0)
            pending = e1^                            # the next group's decode error waits until this group is merged
        else:
            r = _run_part_dev(part, nodes, hs, gm, use_gpu, device == GPU, gctx, why, gpu_groups, rows_cur, ext_cur, threads, piece_no)
        _ = part^
        _ = gm^
        cur = nxt^
        ext_cur = ext_nxt^
        _merge_partial(acc, r, ops)
        job.advance(rows_cur)
        if pending.byte_length() > 0:                # only now, after this group's partial was merged (its overflow comes first)
            raise Error(pending)
    job.check()
    job.stage(STAGE_FINISH)
    # the layout of a whole-file run: lowered over the columns with the null counts the file turned out to have
    var proto = Table.empty()
    for k in range(len(used)):
        _ = proto.add_column(scan.name(used[k]), scan.kind(used[k]), scale=scan.scale(used[k]))
        proto.cols[k].nulls = nulls[k]
    var want = _lower(proto^.share(), nodes, framed, False)        # a frame's query always counts non-null rows (F-147)
    var out = QueryResult()
    out.kinds = want.kinds.copy()
    out.slot_of = want.slot_of.copy()
    out.count_slot_of = want.count_slot_of.copy()
    out.agg_float = want.agg_float.copy()
    out.nan_slot_of = want.nan_slot_of.copy()
    for _ in range(want.n_slots):
        out.slots.append(List[Int](length=want.groups, fill=0))
    for i in range(len(want.kinds)):
        if want.slot_of[i] >= 0:
            out.slots[want.slot_of[i]] = acc.slots[shape.slot_of[i]].copy()
        if want.count_slot_of[i] >= 0:
            out.slots[want.count_slot_of[i]] = acc.slots[shape.count_slot_of[i]].copy()
        if want.nan_slot_of[i] >= 0:
            out.slots[want.nan_slot_of[i]] = acc.slots[shape.nan_slot_of[i]].copy()
    out.counts = acc.counts.copy()
    out.hol_of = List[Int](length=len(out.kinds), fill=-1)
    if framed and len(o.delta) > 0:                          # F-147: the frame's changed and appended rows
        _add_delta(out, scan.schema, nodes, o, hs)
    if shape.mat:
        _finish_holistic(out, hs)
    for i in range(len(out.kinds)):
        if out.kinds[i] == A_SUM:
            out.sums = out.slots[out.slot_of[i]].copy()
            break
    if use_gpu:
        comptime if HAS_GPU:
            if _is_oom(why):                                      # F-189: remembered on the file, for AUTO's next query
                scan.remember_gpu_oom(gpu_free_bytes(_scan_context(gctx)))
            elif scan.gpu_oom_free() > 0 and gpu_groups == len(todo_all):
                scan.forget_gpu_oom()
        out.pieces = len(todo_all)
        out.gpu_pieces = gpu_groups
    _ = gown^                                                 # (kept alive to here: the groups above use its address)
    out.device = GPU if use_gpu and gpu_groups == len(todo_all) else CPU
    out.fallback_reason = why^
    out.milliseconds = Float64(perf_counter_ns() - t0) / 1e6
    return out^
