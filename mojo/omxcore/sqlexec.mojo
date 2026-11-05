"""Running SQL (F-150): `sql(catalog, text)` -> `SqlResult` holding a Table.

    var cat = Catalog()
    cat.add_table("orders", orders)              # or add_frame(name, frame), add_scan(name, scan)
    var r = sql(cat, "SELECT region, sum(revenue) FROM orders WHERE qty > 2 GROUP BY region ORDER BY 2 DESC LIMIT 5")
    r.table; r.notes; r.explain; r.ms

The steps: parse and bind (`bind.mojo`), plan from statistics (`planner.mojo`: dense or hashed groups, with the reason),
lower to one kernel program (`sqlprog.mojo`), run it over the source's chunks and merge their partial results,
then finish in the SQL way (`sqleval.mojo` for HAVING / ORDER BY / select expressions), producing an owned `Table`.

Chunks: a table is one chunk; a frame is its shared table with the changed rows masked out plus a small table of
the changed and appended rows; a scan is one chunk per row group. Each chunk's partial result (sums and counts add,
min / max combine; hashed groups merge by key) folds into one.
"""
from std.time import perf_counter_ns
from std.math import exp, log
from .table import Table, SharedTable, Column, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, F32_COLUMN
from .frame import NULL_EDIT, text_ranks, _radix_order
from .job import JobCtl, STAGE_PLAN, STAGE_AGGREGATE, STAGE_FINISH      # F-175
from .query import (Program, QueryResult, _run_cpu, _run_cpu_job, _run_mat, _key_from_reg0, _make_result, _merge_partial, _identity_partial, _ScanMask, _group_mask,
                    OVERFLOW_MESSAGE, overflow_message, CPU, GPU, AUTO, MAGIC, DT_F64, OP_LOAD_CONST, OP_LOADI, OP_IADD, OP_ISUB, OP_IMUL, OP_AND, OP_OR, OP_NOT, OP_SELECT,
                    OP_ISNULL, OP_IDIV, OP_IMOD, OP_IMIN, OP_IMAX, OP_ICMP, OP_AGG_ISUM, OP_AGG_IMIN, OP_AGG_IMAX, OP_F2I, _run_gpu, _run_gpu_pieces, A_MIN, A_MAX, A_FSUM, MORSEL_ROWS, JOB_RANGE_ROWS, WINDOW_BYTES, SRC_WORDS, fold_float, slot_identity)
from .kernels._fpenv import core_count, par_n
from .kernels.pipeline_cpu import IP, LP, FP, BP, VEC, E_OVERFLOW, _no_front, _no_keys, pipeline_keyed
from .kernels.hashagg_cpu import hash_aggregate
from .holistic import (HolStates, HolSpec, HolResult, A_DISTINCT, A_PCT, A_MEDIAN, A_QEDGE, A_VAR_POP, A_VAR_SAMP, A_STDDEV_POP, A_STDDEV_SAMP, window_blocks,
                           float_key)
from .sql import sql_error, parse_sql, SqlStatement, EXPLAIN_NONE, EXPLAIN_ANALYZE
from .bind import (copy_query, Catalog, SqlSource, BoundQuery, SortKey, bind_sql, bind_statement, SRC_TABLE, SRC_FRAME, SRC_SCAN, B_COL, F_COUNT_STAR,
                   F_COUNT, F_SUM, F_MIN, F_MAX, F_AVG, F_COUNT_DISTINCT, F_MEDIAN, F_STDDEV_SAMP, F_STDDEV_POP, F_VAR_SAMP, F_VAR_POP, F_QEDGE, B_QBIN, B_LIT, B_AGG,
                   BExpr, AggSpec, KeySpec, pow10_checked)
from .plan import LogicalPlan, build_plan, executable
from .post import run_post
from .join import (needed_columns, side_query, joined_query, virtual_schema, all_needed, needed_after, table_of_column, filter_query, final_query, count_query,
                   with_probe_filter, final_needed, virtual_schema_multi, star_query)
from .joincore import JoinKeys, JoinSide, KeyPlan, make_key_plan, column_ndv, normalize_keys, unmatchable_not_null, join_pairs, select_rows, gather_join, JoinPairs, JOIN_BUDGET_BYTES, build_budget_error
from .kernels.hashjoin_cpu import hj_build, JoinMap, PM_SEMI, PM_ANTI, PM_ANTI_NA, PM_NONE
from .kargs import KernelRun
from .winsql import plan_windows, finish_windows, point_wide_columns, fix_wide_columns, WIN_TABLE
from .jfront import EXPAND_MAX_REP, pred_reads_probe_only, FusedJoin, StageData, StreamStage, ProbeStream, chunk_join, stream_template, front_spec, jf_front
from std.memory import ArcPointer
from .gpu import DeviceTable, HAS_GPU, gpu_available
from .rows import RowStage, prepare_rows, execute_rows, _Item
from .sink import ResultSink                  # F-167
from .parquet import write_parquet, ZSTD, DEFAULT_ROW_GROUP
from std.memory.alloc import unsafe_alloc
from .sqlprog import lower_rows
from .planner import (plan_sinks, Planner, RulePlanner, PlannerOptions, PhysicalPlan, Stats, ColStat, G_NONE, G_DENSE, G_HASHED, commas, JoinStats, JoinPlan,
                      BUILD_AUTO, BUILD_LEFT, BUILD_RIGHT)
from .sqlprog import SqlProgram, DictAccess, lower_sql, collect_columns, PHASE_ALL, PHASE_PLAIN, PHASE_HOL
from .sqleval import ValCol, float_bits, ld_div, ld_div_wide
from .wide import wide_text
from .exec import (_read_ahead, _table_addresses, _column_range, _is_range_kind, _key_columns, prepare_ranges, _alias_lists, _delta_nulls, make_stats,
                   delta_table, _gather, _sort_keys, _order_rows, _SortCols, _top_rows, _fill_column, _unique_name, _to_table, _ones_mask,
                   _Chunks, _open_chunks, _load_chunk, chunk_rows_total, LIE, _check_footer, _all, _Inputs, _inputs)


struct SqlResult(Movable):
    var table: Table
    var explain: String              # the plan as text (always filled; EXPLAIN returns it as the table too)
    var ms: Float64
    var device: Int                  # CPU (the SQL path runs on the CPU in F-150)
    var fallback_reason: String
    var notes: List[String]          # what the planner decided and what ran, one line each: tests assert them
    var actuals: List[String]        # F-155: what each operator really did (rows, milliseconds), one line each: EXPLAIN ANALYZE prints them
    var rows_written: Int            # F-167: `sql_to_parquet`: the rows written to the file (-1 for a plain `sql`)

    def __init__(out self):
        self.actuals = List[String]()
        self.rows_written = -1
        self.table = Table.empty()
        self.explain = String()
        self.ms = 0.0
        self.device = CPU
        self.fallback_reason = String()
        self.notes = List[String]()

    def take_table(mut self) -> Table:
        """The result table, moved out (the result keeps an empty one)."""
        var t = self.table^
        self.table = Table.empty()
        return t^


# ── partial results ──────────────────────────────────────────────────────────────────────────────────

struct Partial(Movable):
    var slots: List[List[Int]]       # [slot][group]
    var counts: List[Int]            # rows per group
    var kw: List[List[Int64]]        # hashed: [key word][group]

    def __init__(out self):
        self.slots = List[List[Int]]()
        self.counts = List[Int]()
        self.kw = List[List[Int64]]()

    def groups(self) -> Int:
        return len(self.counts)


def _mix(h: UInt64) -> UInt64:
    var x = h ^ (h >> 33)
    x *= 0xFF51AFD7ED558CCD
    x ^= x >> 33
    x *= 0xC4CEB9FE1A85EC53
    x ^= x >> 33
    return x


def _hash_group(kw: List[List[Int64]], g: Int) -> UInt64:
    var h = UInt64(0x9E3779B97F4A7C15)
    for k in range(len(kw)):
        h = _mix(h ^ UInt64(kw[k][g]))
    return h


comptime MERGE_PART_BITS = 8                  # 256 partitions: each is merged by one worker, with no sharing and no locks


struct _PartData(Movable):
    """One partition of a streamed hashed aggregate's groups: their slots, counts and key words, and the table that finds a group by its key words."""
    var slots: List[List[Int]]
    var counts: List[Int]
    var kw: List[List[Int64]]
    var hashes: List[UInt64]
    var table: List[Int32]
    var mask: Int

    def __init__(out self, n_slots: Int, nkw: Int):
        self.slots = List[List[Int]]()
        for _ in range(n_slots):
            self.slots.append(List[Int]())
        self.counts = List[Int]()
        self.kw = List[List[Int64]]()
        for _ in range(nkw):
            self.kw.append(List[Int64]())
        self.hashes = List[UInt64]()
        self.table = List[Int32](length=16, fill=-1)
        self.mask = 15

    def grow(mut self):
        var cap = len(self.table) * 2
        self.table = List[Int32](length=cap, fill=-1)
        self.mask = cap - 1
        for g in range(len(self.counts)):
            var pos = Int(self.hashes[g]) & self.mask
            while self.table[pos] >= 0:
                pos = (pos + 1) & self.mask
            self.table[pos] = Int32(g)

    def fold(mut self, other: Partial, o: Int, h: UInt64, ops: List[Int]) -> Bool:
        """Folds group `o` of `other` (hash `h`) into this partition; False when an integer total passes int64."""
        if 2 * (len(self.counts) + 1) > len(self.table):
            self.grow()
        var pos = Int(h) & self.mask
        var found = -1
        while self.table[pos] >= 0:
            var g = Int(self.table[pos])
            if self.hashes[g] == h:
                var same = True
                for k in range(len(self.kw)):
                    if self.kw[k][g] != other.kw[k][o]:
                        same = False
                        break
                if same:
                    found = g
                    break
            pos = (pos + 1) & self.mask
        if found < 0:
            found = len(self.counts)
            for sl in range(len(self.slots)):
                self.slots[sl].append(other.slots[sl][o])
            self.counts.append(other.counts[o])
            for k in range(len(self.kw)):
                self.kw[k].append(other.kw[k][o])
            self.hashes.append(h)
            self.table[pos] = Int32(found)
            return True
        for sl in range(len(self.slots)):
            var a = self.slots[sl][found]
            var b = other.slots[sl][o]
            if ops[sl] >= A_FSUM:
                self.slots[sl][found] = fold_float(ops[sl], a, b)
            elif ops[sl] == A_MIN:
                self.slots[sl][found] = min(a, b)
            elif ops[sl] == A_MAX:
                self.slots[sl][found] = max(a, b)
            else:
                var x = Int64(a)
                var y = Int64(b)
                var t = x + y
                if ((x ^ t) & (y ^ t)) < 0:
                    return False
                self.slots[sl][found] = Int(t)
        self.counts[found] += other.counts[o]
        return True


struct PartAcc(Movable):
    """The groups of a streamed hashed aggregate (a scan's row groups, a frame's base and delta), merged as each chunk's partial arrives: the first partial is
    kept whole, from the second on the groups live in 256 hash partitions and a partial's groups are routed by their hash and folded by the workers, one
    partition each (no sharing, no locks), so the merge costs the chunk's groups over the workers, not a serial pass over a growing table. Integer totals are
    exact (a total past int64 raises); the groups come back partition by partition."""
    var first: Partial
    var batches: Int
    var parts: List[_PartData]
    var n_slots: Int
    var nkw: Int

    def __init__(out self):
        self.first = Partial()
        self.batches = 0
        self.parts = List[_PartData]()
        self.n_slots = 0
        self.nkw = 0

    def _route(mut self, batch: Partial, ops: List[Int], threads: Int) raises:
        var og = batch.groups()
        if og == 0:
            return
        var np = 1 << MERGE_PART_BITS
        var hashes = List[UInt64](length=og, fill=0)
        var hp = hashes.unsafe_ptr()
        var T = max(1, min(threads if threads > 0 else core_count(), og // 32768 + 1))

        def hash_range(t: Int) {imm}:
            for o in range(og * t // T, og * (t + 1) // T):
                hp[o] = _hash_group(batch.kw, o)

        par_n(hash_range, T, T)
        var start = List[Int](length=np + 1, fill=0)
        for o in range(og):
            start[Int(hashes[o] >> UInt64(64 - MERGE_PART_BITS)) + 1] += 1
        for p in range(np):
            start[p + 1] += start[p]
        var fill = List[Int](length=np, fill=0)
        var order = List[Int32](length=og, fill=0)
        for o in range(og):
            var p = Int(hashes[o] >> UInt64(64 - MERGE_PART_BITS))
            order[start[p] + fill[p]] = Int32(o)
            fill[p] += 1
        var bad = List[Int](length=1, fill=0)
        var badp = bad.unsafe_ptr()
        var pp = self.parts.unsafe_ptr()
        var op = order.unsafe_ptr()
        var sp = start.unsafe_ptr()
        var W = max(1, min(threads if threads > 0 else core_count(), np))

        def fold_parts(w: Int) {imm}:
            for p in range(w, np, W):
                for i in range(sp[p], sp[p + 1]):
                    var o = Int(op[i])
                    if not pp[p].fold(batch, o, hp[o], ops):
                        badp[0] = 1
                        return

        par_n(fold_parts, W, W)
        _ = order^                                                    # (the closure read them through addresses: alive until the workers are done)
        _ = start^
        if bad[0] != 0:
            raise Error(OVERFLOW_MESSAGE)

    def add(mut self, var batch: Partial, ops: List[Int], threads: Int) raises:
        if batch.groups() == 0:
            return
        if self.batches == 0:
            self.n_slots = len(batch.slots)
            self.nkw = len(batch.kw)
            self.first = batch^
            self.batches = 1
            return
        if self.batches == 1:
            for _ in range(1 << MERGE_PART_BITS):
                self.parts.append(_PartData(self.n_slots, self.nkw))
            var f = self.first^
            self.first = Partial()
            self._route(f, ops, threads)
        self._route(batch, ops, threads)
        self.batches += 1

    def finish(mut self) -> Partial:
        if self.batches <= 1:
            var f = self.first^
            self.first = Partial()
            return f^
        var out = Partial()
        var total = 0
        for p in range(len(self.parts)):
            total += len(self.parts[p].counts)
        for sl in range(self.n_slots):
            var col = List[Int]()
            col.reserve(total)
            for p in range(len(self.parts)):
                col.extend(self.parts[p].slots[sl].copy())
            out.slots.append(col^)
        out.counts.reserve(total)
        for p in range(len(self.parts)):
            out.counts.extend(self.parts[p].counts.copy())
        for k in range(self.nkw):
            var col = List[Int64]()
            col.reserve(total)
            for p in range(len(self.parts)):
                col.extend(self.parts[p].kw[k].copy())
            out.kw.append(col^)
        self.parts = List[_PartData]()
        return out^


def _hashed_result(run: KernelRun, res_r: Int, n_slots: Int, nkw: Int) -> Partial:
    """The groups hash_aggregate found, from the addresses it left in the run's result region (the run is a parameter: alive here)."""
    var res = run.i64(res_r)
    var out = Partial()
    var g = Int(res[0])
    var stride = max(g, 1)
    var vals = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(res[1]))
    var cnt = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(res[2]))
    for s in range(n_slots):
        var col = List[Int]()
        col.reserve(g)
        for x in range(g):
            col.append(Int(vals[s * stride + x]))
        out.slots.append(col^)
    for x in range(g):
        out.counts.append(Int(cnt[x]))
    for k in range(nkw):
        var kp = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(res[3 + k]))
        var col = List[Int64]()
        col.reserve(g)
        for x in range(g):
            col.append(kp[x])
        out.kw.append(col^)
        kp.free()
    vals.free()
    cnt.free()
    return out^


def _run_hashed(t: SharedTable, sp: SqlProgram, mask_addr: Int, nkw: Int, threads: Int, fj: Optional[ArcPointer[FusedJoin]],
                row0: Int = 0, rows: Int = -1) raises -> Partial:
    """The kernel with hashed groups (hashagg_cpu.hash_aggregate): each worker numbers the key tuples of its rows in a local
    table, the partitions are merged in parallel, and the result is the groups with their key words, slots and counts."""
    var n_rows = t.rows() if rows < 0 else rows                   # F-175: rows row0 .. row0 + n_rows (a multiple of 8 from row0) of the chunk
    var run = KernelRun(sp.p, sp.set_bytes, sp.set_meta, t, mask_addr, 1, row0, n_rows)
    var res_r = run.region(3 + max(nkw, 1))

    def keyw(c: Int, i0: Int, sl: IP, nsel: Int, use_sel: Bool, dst: LP, rg: FP, wsp: BP) {imm} -> Bool:
        for k in range(nkw):
            var src = LP(unsafe_from_address=Int(rg) + k * VEC * 8)
            var o = dst + k * VEC
            for j in range(nsel):
                o[j] = src[j]
        return True

    var rc: Int
    if fj:
        var fs = front_spec(run, fj.value()[], sp.p)
        fs.keys_a += 8 * row0                                      # the probe keys are indexed by the block's first row
        if fs.ok_a != 0:
            fs.ok_a += row0

        def front(k: Int, i0: Int, n: Int, sel: IP, ws: BP) {imm} -> Int:
            return jf_front(fs, i0, n, sel, ws)

        rc = run.hashed_fused(keyw, front, fs.ws_bytes, nkw, n_rows, threads, res_r)
    else:
        rc = run.hashed(keyw, nkw, n_rows, threads, res_r)
    if rc == -7 or rc == E_OVERFLOW:
        raise Error(overflow_message(sp.p.conv_text()))
    if rc != 0:
        raise Error("the hashed group-by kernel failed with code " + String(rc))
    return _hashed_result(run, res_r, sp.p.n_slots, nkw)


# ── text for EXPLAIN ─────────────────────────────────────────────────────────────────────────────────

def _explain_text(lp: LogicalPlan, notes: List[String]) -> String:
    var out = String("Logical plan\n")
    var lines = lp.tree()
    for i in range(len(lines)):
        out += "  " * (1 + lines[i][0]) + lines[i][1] + "\n"
    out += "Physical plan\n"
    for n in notes:
        out += "  " + n + "\n"
    return out^


def _finish(q: BoundQuery, plan: PhysicalPlan, sp: SqlProgram, acc: Partial, hres: HolResult, hol_of: List[Int],
            dicts: DictAccess, opts: PlannerOptions, mut notes: List[String]) raises -> Table:
    # the groups that exist: only non-empty ones, except the one row of a query with no GROUP BY
    var t_a = perf_counter_ns()
    var gids = List[Int]()
    if plan.group == G_NONE:
        gids.append(0)
    elif plan.group == G_DENSE:
        for g in range(plan.domain):
            if acc.counts[g] > 0:
                gids.append(g)
    else:
        for g in range(acc.groups()):
            gids.append(g)
    var n = len(gids)
    var t_b = perf_counter_ns()
    # group keys
    var keys = List[ValCol]()
    var kwat = 0
    for k in range(len(plan.keys)):
        ref lay = plan.keys[k]
        var vc = ValCol(lay.kind, lay.scale, n)
        vc.src = lay.src if lay.src >= 0 else q.rexprs[lay.expr].src          # a derived TEXT key (lower(s), CASE ...): its own dictionary
        for i in range(n):
            if plan.group == G_DENSE:
                var part = (gids[i] // lay.stride) % lay.radix
                if lay.nullable:
                    if part == 0:
                        vc.set_null(i)
                        continue
                    part -= 1
                vc.v[i] = Int64(part) + (0 if (lay.kind == TEXT.value or lay.kind == BOOL.value) else lay.lo)
            else:
                if lay.nullable and acc.kw[kwat + 1][gids[i]] != 0:
                    vc.set_null(i)
                    continue
                vc.v[i] = acc.kw[kwat][gids[i]]
        kwat += lay.words
        keys.append(vc^)
    # aggregates
    var aggs = List[ValCol]()
    for a in range(len(q.aggs)):
        ref sp_a = q.aggs[a]
        var vc = ValCol(sp_a.kind, sp_a.scale, n)
        var vs = sp.agg_val_slot[a]
        var ns = sp.agg_nul_slot[a]
        var h = hol_of[a]
        for i in range(n):
            var g = gids[i]
            var rows = acc.counts[g]
            var nonnull = rows - (acc.slots[ns][g] if ns >= 0 else 0)
            var f = sp_a.func
            var arg_float = sp_a.arg >= 0 and q.rexprs[sp_a.arg].kind == FLOAT.value
            if sp_a.hol:                                          # the holistic states: count(DISTINCT), median, quantile_disc, min / max of TEXT
                var present = hres.counts[h][g] >= (2 if f == F_STDDEV_SAMP or f == F_VAR_SAMP else 1)       # (the sample variants need two values)
                if f == F_COUNT_DISTINCT:
                    vc.v[i] = Int64(hres.vals[h][g])
                elif not present:
                    vc.set_null(i)
                elif f == F_MEDIAN and sp_a.kind == FLOAT.value and not arg_float:
                    vc.v[i] = float_bits(Float64(hres.vals[h][g]) / 10.0)
                elif f == F_MEDIAN and arg_float and q.rexprs[sp_a.arg].scale == F32_COLUMN:       # median of REAL is a REAL: the interpolation rounded to 32 bits
                    vc.v[i] = float_bits(Float64(Float32(Float64(from_bits=UInt64(Int64(hres.vals[h][g]))))))
                else:
                    vc.v[i] = Int64(hres.vals[h][g])
            elif f == F_COUNT_STAR:
                vc.v[i] = Int64(rows)
            elif f == F_COUNT:
                vc.v[i] = Int64(nonnull)
            elif f == F_SUM or f == F_MIN or f == F_MAX:
                if nonnull == 0:
                    vc.set_null(i)
                elif f == F_SUM and sp.agg_hi_slot[a] >= 0:         # F-192: the total is the sum of the upper halves times 2^32 plus the lower ones
                    vc.set_wide(i, (Int128(Int64(acc.slots[sp.agg_hi_slot[a]][g])) << Int128(32)) + Int128(Int64(acc.slots[vs][g])))
                elif arg_float and f != F_SUM:                       # NaN is above every number: MAX is NaN when there is one, MIN only when there is nothing else
                    var nans = acc.slots[sp.agg_nan_slot[a]][g]
                    if (f == F_MAX and nans > 0) or (f == F_MIN and nonnull - nans == 0):
                        vc.v[i] = Int64(0x7FF8000000000000)
                    else:
                        vc.v[i] = Int64(acc.slots[vs][g])
                else:
                    vc.v[i] = Int64(acc.slots[vs][g])
            elif f == F_AVG:
                if nonnull == 0:
                    vc.set_null(i)
                elif arg_float:                                      # a sum of doubles over the count, in doubles
                    vc.v[i] = float_bits(Float64(from_bits=UInt64(Int64(acc.slots[vs][g]))) / Float64(nonnull))
                else:
                    var den = UInt128(nonnull) * UInt128(Int(pow10_checked(q.rexprs[sp_a.arg].scale)))
                    if sp.agg_hi_slot[a] >= 0:
                        vc.v[i] = float_bits(ld_div_wide((Int128(Int64(acc.slots[sp.agg_hi_slot[a]][g])) << Int128(32)) + Int128(Int64(acc.slots[vs][g])), den))
                    else:
                        vc.v[i] = float_bits(ld_div(Int64(acc.slots[vs][g]), den))
        if sp_a.kind == TEXT.value:
            vc.src = q.rexprs[sp_a.arg].src
        aggs.append(vc^)
    var t_c = perf_counter_ns()
    # HAVING, the select list, ORDER BY, OFFSET and LIMIT: a row query over the groups (post.mojo)
    var table = run_post(q, keys, aggs, dicts, opts, n, notes)
    notes.append("finish: groups " + _ms(Float64(t_b - t_a) / 1e6) + " ms, columns " + _ms(Float64(t_c - t_b) / 1e6) + " ms, then HAVING, ORDER BY and the select list over the groups " +
                 _ms(Float64(perf_counter_ns() - t_c) / 1e6) + " ms")
    return table^


# ── running ──────────────────────────────────────────────────────────────────────────────────────────

def _holistic_states(q: BoundQuery, groups: Int, sp: SqlProgram, dicts: DictAccess, rows: Int) raises -> HolStates:
    var specs = List[HolSpec]()
    var ranks = List[List[Int64]]()
    var scales = List[Int](length=sp.n_src, fill=0)                # the decimal places of a stddev / variance source
    for _ in range(sp.n_src):
        ranks.append(List[Int64]())
    for a in range(len(q.aggs)):
        if sp.agg_src[a] < 0:
            continue
        ref sa = q.aggs[a]
        var kind = A_DISTINCT if sa.func == F_COUNT_DISTINCT else A_MEDIAN if sa.func == F_MEDIAN else A_PCT
        if sa.func == F_STDDEV_SAMP:
            kind = A_STDDEV_SAMP
        elif sa.func == F_STDDEV_POP:
            kind = A_STDDEV_POP
        elif sa.func == F_VAR_SAMP:
            kind = A_VAR_SAMP
        elif sa.func == F_VAR_POP:
            kind = A_VAR_POP
        if sa.func == F_QEDGE:
            kind = A_QEDGE
        specs.append(HolSpec(a, kind, sa.p, sa.bins if sa.func == F_QEDGE else 0, sp.agg_src[a]))
        if kind >= A_VAR_POP and kind <= A_STDDEV_SAMP and q.rexprs[sa.arg].kind == DECIMAL.value:
            scales[sp.agg_src[a]] = q.rexprs[sa.arg].scale
        if sa.func != F_COUNT_DISTINCT and q.rexprs[sa.arg].kind == TEXT.value and len(ranks[sp.agg_src[a]]) == 0:
            var c = q.rexprs[sa.arg].src                       # (a derived TEXT value: its own dictionary)
            var words = List[String]()
            for k in range(dicts.count(c)):
                words.append(dicts.word(c, k))
            ranks[sp.agg_src[a]] = text_ranks(words)
    return HolStates(groups, specs^, ranks^, rows, sp.p.src_float.copy(), scales^)


def _run_dense_fused(t: SharedTable, sp: SqlProgram, mask_addr: Int, threads: Int, fj: ArcPointer[FusedJoin], start: Int = 0, stop: Int = -1) raises -> QueryResult:
    """The dense / global aggregate over the rows the join's front stage lets through (the kernel call of query.mojo's `_run_cpu`, fused)."""
    ref p = sp.p
    var run = KernelRun(p, sp.set_bytes, sp.set_meta, t, mask_addr, p.groups)
    var fs = front_spec(run, fj[], p)

    def front(k: Int, i0: Int, n: Int, sel: IP, ws: BP) {imm} -> Int:
        return jf_front(fs, i0, n, sel, ws)

    var rc: Int
    var end = t.rows() if stop < 0 else stop
    if p.keyed:
        rc = run.fused(_key_from_reg0, front, fs.ws_bytes, True, p.groups, start, end, threads)
    else:
        rc = run.fused(_no_keys, front, fs.ws_bytes, False, 1, start, end, threads)
    if rc == E_OVERFLOW:
        raise Error(overflow_message(p.conv_text()))
    if rc != 0:
        raise Error("the CPU kernel failed with code " + String(rc))
    return _make_result(p, run.out_vals, run.out_cnt)


# ── F-191: the device ─────────────────────────────────────────────────────────────────────────────────────────

struct _Gpu(Movable):
    """Where a single-table aggregate runs: the decision made before it runs (`use`, or `why` it does not), and what happened (`ran`)."""
    var asked: Int                   # PlannerOptions.device
    var use: Bool                    # the builder's GPU programs run this query's aggregate stage on the GPU
    var why: String                  # the CPU: the reason
    var dt: Optional[DeviceTable]
    var uploaded: Bool               # the table was uploaded for this query (no DeviceTable was attached)
    var ran: Bool
    var declined: String             # the GPU was tried and the CPU answered: why

    def __init__(out self, asked: Int):
        self.asked = asked
        self.use = False
        self.why = String()
        self.dt = None
        self.uploaded = False
        self.ran = False
        self.declined = String()

    def on_cpu(self) -> Bool:
        return not self.ran

    def reason(self) -> String:
        """Why the CPU answered (empty when the GPU did or the CPU was asked for)."""
        if self.ran or self.asked == CPU:
            return String()
        return self.declined if self.declined.byte_length() > 0 else self.why

    def note(self) -> String:
        var who = String("GPU asked for") if self.asked == GPU else String("AUTO")
        if self.ran:
            return "device: GPU (" + who + (", the table uploaded for this query" if self.uploaded else ", the attached DeviceTable") + ")"
        if self.asked == CPU:
            return "device: CPU (device CPU was asked for)"
        return "device: CPU (" + who + (": " if self.asked == AUTO else "; ") + self.reason() + ")"


comptime OP_SUBF = 11    # float64 x - y (three-valued logic: 1.0 - unknown), the kernel's OP_SUB
comptime OP_LT = 20      # query.mojo has no names for the kernel's float64 compares OP_LT .. OP_NE (20 .. 25). SQL emits them on 0 / 1 predicate registers and on
comptime OP_NE = 25      # float constants (0.0, 0.5, 1.0) for NULL / unknown handling (three-valued AND / OR); the GPU kernel compares them as float64 like the CPU
                         # (pipeline_gpu.mojo OP_LT .. OP_NE). A FLOAT column and I2F never reach here: `_gpu_blocker` declines them.


def _gpu_op_ok(op: Int) -> Bool:
    """The integer instructions the builder's GPU kernel runs with the CPU's meaning (kernels_aot/pipeline_gpu.mojo)."""
    return (op == OP_LOAD_CONST or op == OP_LOADI or op == OP_IADD or op == OP_ISUB or op == OP_IMUL or op == OP_AND or op == OP_OR or op == OP_NOT or
            op == OP_SELECT or op == OP_ISNULL or op == OP_IDIV or op == OP_IMOD or op == OP_IMIN or op == OP_IMAX or op == OP_ICMP or
            op == OP_AGG_ISUM or op == OP_AGG_IMIN or op == OP_AGG_IMAX or (op >= OP_LT and op <= OP_NE) or op == OP_F2I or op == OP_SUBF)


def _bad_op(ins: List[Int32]) -> Int:
    """The first instruction in `ins` the GPU kernel does not run (-1: none)."""
    for i in range(len(ins) // 5):
        if not _gpu_op_ok(Int(ins[i * 5])):
            return Int(ins[i * 5])
    return -1


def _gpu_blocker(src: SqlSource, q: BoundQuery, plan: PhysicalPlan, sp: SqlProgram, two_pass: Bool, fj: Bool) -> String:
    """Why the GPU programs cannot run this query's aggregate stage ("" when they can)."""
    if q.rows:
        return "a query without aggregates runs on the CPU"
    if fj:
        return "a join runs on the CPU"
    if src.kind != SRC_TABLE:
        return "a " + ("frame" if src.kind == SRC_FRAME else "Parquet or CSV scan") + " runs on the CPU (the GPU runs a table)"
    if plan.group == G_HASHED:
        return "hashed groups run on the CPU (the GPU kernel takes no groups or dense groups)"
    if sp.p.mat or two_pass:
        return "count(DISTINCT), median, quantiles and min / max of TEXT run on the CPU"
    if sp.p.has_float or sp.p.repro:
        return "a FLOAT column or float arithmetic runs on the CPU (a float64 sum has one defined order there)"
    for d in sp.p.dtypes:
        if Int(d) == DT_F64:
            return "a FLOAT column runs on the CPU"
    var bad = max(_bad_op(sp.p.pred), max(_bad_op(sp.p.body), _bad_op(sp.p.aggs)))
    if bad >= 0:
        return "the program uses an instruction the GPU kernel does not run (opcode " + String(bad) + ": text, dates, IN lists, casts or float arithmetic)"
    return String()


def _pick_device(src: SqlSource, q: BoundQuery, plan: PhysicalPlan, sp: SqlProgram, opts: PlannerOptions, two_pass: Bool, fj: Bool) -> _Gpu:
    """CPU never touches the GPU. AUTO runs on the CPU unless a DeviceTable of this table is attached (its data is resident: uploading per query costs more than
    the CPU's answer), the table has `opts.gpu_min_rows` rows (the builder's rule) and the GPU programs run the query. GPU runs what the programs run, uploading
    the table for the query when none is attached. Everything else is the CPU with the reason."""
    var g = _Gpu(opts.device)
    if opts.device == CPU:
        return g^
    if opts.device == AUTO and src.kind == SRC_TABLE and src.rows() < opts.gpu_min_rows:
        g.why = "the table has " + String(src.rows()) + " rows, under the " + String(opts.gpu_min_rows) + " AUTO runs on the GPU"
        return g^
    var blocker = _gpu_blocker(src, q, plan, sp, two_pass, fj)
    if blocker.byte_length() > 0:
        g.why = blocker^
        return g^
    var attached = _attached(src, opts)
    if opts.device == AUTO and not attached:
        g.why = "no table resident on the GPU (attach a DeviceTable, or --device gpu uploads it for the query)"
        return g^
    comptime if HAS_GPU:
        if not gpu_available():
            g.why = "no GPU found"
            return g^
        g.use = True
        g.uploaded = not attached
    else:
        g.why = "no GPU on this build"
    return g^


def _attached(src: SqlSource, opts: PlannerOptions) -> Bool:
    """A live DeviceTable made from this source's table is attached."""
    return Bool(opts.device_table) and src.kind == SRC_TABLE and opts.device_table.value().is_live() and opts.device_table.value().mirrors(src.table)


def _gpu_open(src: SqlSource, opts: PlannerOptions, mut g: _Gpu) raises:
    """The device copy of the table: the attached one, else the table is uploaded for this query (every column: `DeviceTable` takes a whole table). A table that
    does not fit stays unresident (it runs in pieces). A copy uploaded by an earlier run of this query (the 128-bit retry) is released first."""
    g.dt = None
    if _attached(src, opts):
        g.dt = opts.device_table.value()
    else:
        g.dt = DeviceTable(src.table)


def _gpu_chunk(chunk: SharedTable, sp: SqlProgram, mut g: _Gpu, job: JobCtl) raises -> Optional[QueryResult]:
    """The program over the chunk on the GPU, or None (the CPU answers; `g.declined` says why). Same program, same answer."""
    comptime if HAS_GPU:
        var fb = String()
        var dt = g.dt.value()
        var none = List[UInt8]()
        var r = QueryResult()
        if dt.is_resident():
            var got = _run_gpu(dt, chunk, sp.p, none, False, fb)
            if not got:
                g.declined = fb^
                return None
            r = got.take()
        else:
            r = _run_gpu_pieces(dt, chunk, sp.p, none, False, fb, job)
        g.ran = True
        return r^
    return None


def _fold(sp: SqlProgram, plan: PhysicalPlan, chunk: SharedTable, mask_addr: Int, mut hs: HolStates, ops: List[Int], threads: Int,
          mut pacc: PartAcc, mut dense_acc: QueryResult, fj: Optional[ArcPointer[FusedJoin]], mut gpu: _Gpu, job: JobCtl = JobCtl()) raises:
    """Runs the program over one chunk and folds its partial result into the accumulator. `job` (F-175) moves by the chunk's rows."""
    var n = chunk.rows()
    var exact = job.active() and not sp.p.repro
    for o in ops:
        exact = exact and o < A_FSUM
    var step = JOB_RANGE_ROWS // MORSEL_ROWS * MORSEL_ROWS
    exact = exact and n > 2 * step
    if plan.group == G_HASHED:
        if exact:                                                    # F-175: row ranges whose groups are merged, as a scan's row groups are
            var at = 0
            while at < n:
                job.check()
                var end = min(n, at + step)
                pacc.add(_run_hashed(chunk, sp, mask_addr, plan.key_words(), threads, fj, at, end - at), ops, threads)
                job.advance(end - at)
                at = end
            return
        pacc.add(_run_hashed(chunk, sp, mask_addr, plan.key_words(), threads, fj), ops, threads)
        job.advance(n)
        return
    var r: QueryResult
    if fj:
        if exact and not sp.p.mat:
            var at = 0
            r = _identity_partial(sp.p.n_slots, sp.p.groups, ops)
            while at < n:
                job.check()
                var end = min(n, at + step)
                _merge_partial(r, _run_dense_fused(chunk, sp, mask_addr, threads, fj.value(), at, end), ops)
                job.advance(end - at)
                at = end
        else:
            r = _run_dense_fused(chunk, sp, mask_addr, threads, fj.value())
            job.advance(n)
    elif sp.p.mat:
        r = _run_mat(chunk, sp.p, mask_addr, hs, 0, 0, job)
    else:
        var on_gpu = _gpu_chunk(chunk, sp, gpu, job) if (gpu.dt and mask_addr == 0) else Optional[QueryResult](None)    # F-191; the GPU kernel takes no overlay mask: a masked chunk stays on the CPU
        if on_gpu:
            r = on_gpu.take()
            job.advance(n)
        else:
            r = _run_cpu_job(chunk, sp.p, mask_addr, threads, job)          # (row ranges for a big table: a cancel is seen between them)
    _merge_partial(dense_acc, r, ops)


# ── hashed groups with holistic aggregates: a second pass maps each row to its group ─────────────────

def _hmat_hook(chunk: Int, row0: Int, sel: IP, nsel: Int, use_sel: Bool, dst: IP, regs: FP) -> Bool:
    """The group id of each selected row, found by its key words in the table of the groups pass 1 found (register 0 holds
    the address of the control block, registers 1 .. nkw the key words), then the pairs of every holistic source, as
    query.mojo's `_mat_hook` writes them."""
    var ctl = LP(unsafe_from_address=Int(LP(unsafe_from_address=Int(regs))[0]))
    var nsrc = Int(ctl[0])
    var tail = 6 + SRC_WORDS * nsrc
    var nkw = Int(ctl[tail])
    var table = IP(unsafe_from_address=Int(ctl[tail + 1]))
    var mask = UInt64(ctl[tail + 2])
    var kwp = LP(unsafe_from_address=Int(ctl[tail + 3]))
    for j in range(nsel):
        var h = UInt64(0x9E3779B97F4A7C15)
        for k in range(nkw):
            h = _mix(h ^ UInt64(LP(unsafe_from_address=Int(regs + (1 + k) * VEC))[j]))
        var pos = Int(h & mask)
        var gid = -1
        while True:
            var g = Int(table[pos])
            if g < 0:
                break
            var same = True
            for k in range(nkw):
                if LP(unsafe_from_address=Int(kwp[k]))[g] != LP(unsafe_from_address=Int(regs + (1 + k) * VEC))[j]:
                    same = False
                    break
            if same:
                gid = g
                break
            pos = (pos + 1) & Int(mask)
        dst[j] = Int32(gid)
    var groups = ctl[2]
    var base = row0 - Int(ctl[1])
    for s in range(nsrc):
        var c = ctl + (6 + SRC_WORDS * s)
        var vals = LP(unsafe_from_address=Int(regs + Int(c[0]) * VEC))
        var gp = IP(unsafe_from_address=Int(c[2])) + base
        var vp = LP(unsafe_from_address=Int(c[3])) + base
        var o = 0
        if c[1] < 0:
            for j in range(nsel):
                var g = Int64(dst[j])
                gp[o] = Int32(g)
                vp[o] = vals[j]
                o += Int((g >= 0) & (g < groups))
        else:
            var valid = LP(unsafe_from_address=Int(regs + Int(c[1]) * VEC))
            for j in range(nsel):
                var g = Int64(dst[j])
                gp[o] = Int32(g)
                vp[o] = vals[j]
                o += Int((g >= 0) & (g < groups) & (valid[j] != 0))
        if c[10] != 0:
            for q in range(o):
                vp[q] = float_key(vp[q])
        IP(unsafe_from_address=Int(c[4]))[base // VEC] = Int32(o)
    return True


struct _GroupLookup(Movable):
    """The hash table from key words to group number, over the groups of pass 1."""
    var table: List[Int32]
    var kwp: List[Int]
    var mask: Int
    var nkw: Int
    var groups: Int

    def __init__(out self, acc: Partial):
        self.groups = acc.groups()
        self.nkw = len(acc.kw)
        var cap = 16
        while cap < 2 * max(self.groups, 1):
            cap *= 2
        self.table = List[Int32](length=cap, fill=-1)
        self.mask = cap - 1
        self.kwp = List[Int]()
        for k in range(self.nkw):
            self.kwp.append(Int(acc.kw[k].unsafe_ptr()))
        for g in range(self.groups):
            var pos = Int(_hash_group(acc.kw, g) & UInt64(self.mask))
            while self.table[pos] >= 0:
                pos = (pos + 1) & self.mask
            self.table[pos] = Int32(g)


def _absorb_window(run: KernelRun, mut hs: HolStates, s: Int, g_r: Int, v_r: Int, c_r: Int, nblk: Int) raises:
    """The window's pairs of source s into the states (the run is a parameter: the regions stay alive while the states read them)."""
    hs.absorb(s, window_blocks(Int(run.i32(g_r)), Int(run.i64(v_r)), Int(run.i32(c_r)), nblk, VEC))


def _run_hol_hashed(t: SharedTable, sp: SqlProgram, look: _GroupLookup, mask_addr: Int, mut hs: HolStates, threads: Int) raises:
    """Pass 2 over one chunk: the kernel runs the predicate and the key words, `_hmat_hook` finds each row's group and
    writes the holistic sources' (group, value) pairs, which the states take in windows (as query.mojo's `_run_mat`)."""
    ref p = sp.p
    var n = t.rows()
    var rows = max(MORSEL_ROWS, (WINDOW_BYTES // (12 * max(p.n_src, 1))) // MORSEL_ROWS * MORSEL_ROWS)
    var done = 0
    var first = True
    var G = max(look.groups, 1)
    while first or done < n:
        var end = min(n, done + rows)
        var nblk = (end - done + VEC - 1) // VEC
        var run = KernelRun(p, sp.set_bytes, sp.set_meta, t, mask_addr, G)
        var ctl = run.region(6 + SRC_WORDS * p.n_src + 4)
        var cw = run.i64(ctl)
        cw[0] = Int64(p.n_src)
        cw[1] = Int64(done)
        cw[2] = Int64(look.groups)
        var g_r = List[Int]()
        var v_r = List[Int]()
        var c_r = List[Int]()
        for s in range(p.n_src):
            g_r.append(run.bytes_region(4 * max(end - done, 1), False))
            v_r.append(run.region(max(end - done, 1), False))
            c_r.append(run.bytes_region(4 * max(nblk, 1)))
            var w = cw + (6 + SRC_WORDS * s)
            w[0] = Int64(p.src_vreg[s])
            w[1] = Int64(p.src_valid[s])
            w[2] = Int64(Int(run.i32(g_r[s])))
            w[3] = Int64(Int(run.i64(v_r[s])))
            w[4] = Int64(Int(run.i32(c_r[s])))
            w[10] = Int64(1 if p.src_float[s] else 0)
        var tail = cw + (6 + SRC_WORDS * p.n_src)
        tail[0] = Int64(look.nkw)
        tail[1] = Int64(Int(look.table.unsafe_ptr()))
        tail[2] = Int64(look.mask)
        tail[3] = Int64(Int(look.kwp.unsafe_ptr()))
        run.control(ctl)
        var rc = run.keyed(_hmat_hook, G, done, end, threads)
        if rc == E_OVERFLOW:
            raise Error(overflow_message(p.conv_text()))
        if rc != 0:
            raise Error("the CPU kernel failed with code " + String(rc))
        for s in range(p.n_src):
            _absorb_window(run, hs, s, g_r[s], v_r[s], c_r[s], nblk)
        first = False
        done = end


def _lower(src: SqlSource, q: BoundQuery, plan: PhysicalPlan, dicts: DictAccess, inp: _Inputs, phase: Int, wide: Bool = False) raises -> SqlProgram:
    try:
        return lower_sql(q, plan, inp.nullable.copy(), inp.pos.copy(), inp.chunk_cols, dicts.share(),
                         _alias_lists(src), inp.masked, phase, wide)
    except e:
        var msg = String(e)
        if msg.startswith("the query needs more than"):
            raise sql_error(q.line, q.col, msg + " (the select list's aggregates and expressions are too many or too long for one program: about 20 aggregates over different expressions, fewer when an expression is long); split it into two queries")
        if msg.startswith("the query has more than"):
            raise sql_error(q.line, q.col, msg + " (a program holds 64 different numbers); split the query in two")
        raise e^


def _kernel_notes(src: SqlSource, q: BoundQuery, plan: PhysicalPlan, dicts: DictAccess, inp: _Inputs, sp: SqlProgram, mut notes: List[String]):
    var names = String()
    var k = 0
    var read = collect_columns(q)
    for c in read:
        names += (", " if k > 0 else "") + src.table.name(c)
        k += 1
    notes.append("columns read: " + (names if k > 0 else "none") + " (" + String(len(read)) + " of " + String(src.table.columns()) + ")")
    notes.append("kernel: " + String(sp.instr) + " instructions a row" + (", " + String(len(sp.set_meta) // 3) + " sets" if len(sp.set_meta) > 0 else "") +
                 " (" + String(sp.p.n_slots) + (" slot" if sp.p.n_slots == 1 else " slots") + ")")
    if sp.p.day_tabs > 0:
        notes.append("dates: " + String(sp.p.day_tabs) + " per-day " + ("table" if sp.p.day_tabs == 1 else "tables") + " over " + String(sp.p.day_days) +
                     " days (each date function of a column ran once per day of its range, never per row)")
    if sp.p.text_tabs > 0:
        notes.append("text: " + String(sp.p.text_tabs) + " per-word " + ("table" if sp.p.text_tabs == 1 else "tables") + " over " + String(sp.p.text_words) +
                     " dictionary words in " + _ms(Float64(sp.p.text_ns) / 1e6) + " ms (each text function ran once per word, never per row)")
    if plan.group == G_HASHED and plan.holistic:
        notes.append("holistic: a second pass maps each row to its group for count(DISTINCT), median, quantile_disc and min / max of TEXT")


def _execute(src: SqlSource, q: BoundQuery, plan: PhysicalPlan, opts: PlannerOptions, dicts: DictAccess,
             mut notes: List[String], inp: _Inputs, var sp: SqlProgram, mut gpu: _Gpu, trust: Bool = True, fj: Optional[ArcPointer[FusedJoin]] = None) raises -> Table:
    var used = inp.used.copy()
    var scan_mode = inp.scan_mode
    var two_pass = plan.group == G_HASHED and plan.holistic
    var ops = sp.slot_ops.copy()
    var threads = opts.threads
    var rows_total = -1 if src.kind != SRC_TABLE else src.rows()
    var hs = _holistic_states(q, sp.p.groups, sp, dicts, rows_total) if (sp.p.mat and not two_pass) else \
             HolStates(1, List[HolSpec](), List[List[Int64]]())
    var domain = sp.p.groups                                      # (a dense plan with a footer-stated range has one extra, for rows outside it)
    var acc = Partial()
    var pacc = PartAcc()
    var dense_acc = _identity_partial(sp.p.n_slots, domain, ops)
    var ps: Optional[ArcPointer[ProbeStream]] = None                  # F-158: a probe side that arrives in chunks (a scan's row groups, a frame's rows)
    if fj:
        ps = fj.value()[].stream
    var lsrc = ps.value()[].src.copy() if ps else src.copy()           # the source the chunks come from
    var ch: _Chunks
    if ps:
        ch = _open_chunks(lsrc, ps.value()[].used)
        scan_mode = lsrc.kind == SRC_SCAN
    else:
        ch = _open_chunks(src, used)
        if fj:
            ch.override = fj.value()[].probe
    var delta_rows = len(lsrc.overlay.delta)
    if gpu.use:                                                       # F-191
        _gpu_open(src, opts, gpu)
    if fj and not ps and fj.value()[].pred_first and sp.p.pred_reg >= 0 and pred_reads_probe_only(sp.p, fj.value()[].n_probe_cols):
        notes.append("probe filter first: the filter reads only the probe table's columns, so each block of 1,024 rows is filtered before the probe and only the rows that pass are probed")
    var t_run = perf_counter_ns()
    var ns_load = 0
    opts.job.stage(STAGE_AGGREGATE)
    opts.job.add_total(chunk_rows_total(lsrc, ch))
    var batch = List[SharedTable]()                                   # a scan's row groups decoded ahead, a worker each
    var batch_at = 0
    var batch_size = max(1, min(threads if threads > 0 else core_count(), 8))
    var read_ahead = scan_mode and not ch.masked and (not fj or ps) and ch.n() > 1 and batch_size > 1
    for i in range(ch.n()):
        opts.job.check()                                              # F-175: between chunks (a scan's row groups)
        var t_load = perf_counter_ns()
        var chunk: SharedTable
        if read_ahead and i < ch.shared():
            if i >= batch_at + len(batch):
                batch_at = i
                batch = _read_ahead(lsrc, ch, i, min(batch_size, ch.shared() - i), threads)
            chunk = batch[i - batch_at]
            ch.mask_addr = 0
        else:
            chunk = _load_chunk(lsrc, ch, i)
        if trust:
            _check_footer(lsrc, ch, i, chunk)
        ns_load += perf_counter_ns() - t_load
        var fji = fj
        if ps:                                                         # this chunk's own join: its keys, the build sides' maps shared
            fji = ArcPointer(chunk_join(ps.value()[], chunk))
        _fold(sp, plan, chunk, ch.mask_addr, hs, ops, threads, pacc, dense_acc, fji, gpu, opts.job)
    if sp.guard_slot >= 0 and dense_acc.counts[sp.guard_slot] > 0:
        raise Error(LIE + String(dense_acc.counts[sp.guard_slot]) + (" row lies" if dense_acc.counts[sp.guard_slot] == 1 else " rows lie") +
                    " outside the range of a group key that the footer gave")
    var ms_run = Float64(perf_counter_ns() - t_run) / 1e6
    var ms_load = Float64(ns_load) / 1e6
    if lsrc.kind == SRC_FRAME:
        notes.append("ran: frame " + lsrc.name + ": " + commas(lsrc.table.rows()) + " shared rows (the changed ones masked out) + " +
                     commas(delta_rows) + " changed or appended rows" + (" (streamed through the join)" if ps else ""))
    elif scan_mode:
        notes.append("ran: scan " + lsrc.name + ": " + String(len(ch.groups)) + " of " + String(lsrc.scan.value().row_groups()) +
                     " row groups read (" + String(len(used)) + (" column" if len(used) == 1 else " columns") + "), one chunk each" +
                     ((", + " + commas(delta_rows) + " changed or appended rows") if delta_rows > 0 else "") + (" (streamed through the join)" if ps else ""))
    else:
        notes.append("ran: table " + src.name + ": 1 chunk of " + commas(src.table.rows()) + " rows")
    if plan.group == G_HASHED:
        acc = pacc.finish()
    else:
        acc.slots = dense_acc.slots.copy()
        acc.counts = dense_acc.counts.copy()
    var hres = HolResult()
    var hol_of = List[Int](length=len(q.aggs), fill=-1)
    var t_hol = perf_counter_ns()
    if two_pass:
        # pass 2: the groups are known, find each row's group again and feed the holistic states
        var sp2 = _lower(src, q, plan, dicts, inp, PHASE_HOL)
        var look = _GroupLookup(acc)
        var hs2 = _holistic_states(q, max(acc.groups(), 1), sp2, dicts, rows_total)
        var ch2 = _open_chunks(src, used)
        opts.job.add_total(chunk_rows_total(src, ch2))
        for i in range(ch2.n()):
            opts.job.check()
            var chunk = _load_chunk(src, ch2, i)
            _run_hol_hashed(chunk, sp2, look, ch2.mask_addr, hs2, threads)
            opts.job.advance(chunk.rows())
        notes.append("ran: holistic second pass over the " + String(ch2.n()) + (" chunk" if ch2.n() == 1 else " chunks") + " mapped each row to one of " +
                     commas(acc.groups()) + " groups")
        hres = hs2.finish()
        for h in range(len(hs2.specs)):
            hol_of[hs2.specs[h].agg] = h
    elif sp.p.mat:
        hres = hs.finish()
        for h in range(len(hs.specs)):
            hol_of[hs.specs[h].agg] = h
    var t_fin = perf_counter_ns()
    opts.job.check()
    opts.job.stage(STAGE_FINISH)
    var table = _finish(q, plan, sp, acc, hres, hol_of, dicts, opts, notes)
    notes.append("time: " + ("decode " + _ms(ms_load) + " ms, " if scan_mode else "") + "kernel " + _ms(ms_run - ms_load) + " ms, holistic states " + _ms(Float64(t_fin - t_hol) / 1e6) + " ms, finish " + _ms(Float64(perf_counter_ns() - t_fin) / 1e6) + " ms")
    return table^


def _ms(x: Float64) -> String:
    return String(Float64(Int(x * 100.0)) / 100.0)


def _fuse_inputs(mut inp: _Inputs, fj: FusedJoin):
    """The program of a fused join reads the probe table's own columns where the joined column is one, and placeholder columns after them
    (the front stage points those at the gathered build-side values)."""
    var np = fj.n_probe_cols
    for v in range(fj.virtual_columns()):
        inp.pos[v] = fj.vreal[v] if fj.vprobe[v] else np + fj.vpartner[v]
    inp.chunk_cols = np + len(fj.partner)


def _prefill_ranges(mut src: SqlSource, jq: BoundQuery, pt: SharedTable, bt: SharedTable, vprobe: List[Bool], vreal: List[Int]):
    """The value range of each integer-like joined column a group key reads, from the real tables (the schema holds no rows to scan)."""
    var cols = List[Int]()
    for k in jq.keys:
        _key_columns(jq, k.expr, cols)
    for c in cols:
        var r = _column_range(pt if vprobe[c] else bt, vreal[c])
        src.range_cols.append(c)
        src.range_lo.append(r[1])
        src.range_hi.append(r[2])
        src.range_has.append(r[0])


def _side_names(q: BoundQuery, t: Int) -> String:
    return q.tab_alias[t] if q.tab_alias[t].byte_length() > 0 else q.tab_names[t]


def _expand_note(expands: Bool, maxrep: Int) -> String:
    if not expands:
        return String("")
    return " (the build side's keys repeat, up to " + commas(maxrep) + " rows of a key: each probe row is expanded to its partners in rounds inside the front stage, no pairs are written)"


def _kind_name(k: Int) -> String:
    if k == 1:
        return "left"
    if k == 2:
        return "semi"
    if k == 3:
        return "anti"
    if k == 4:
        return "anti (null-aware)"
    return "inner"


def _join_side[P: Planner](mut catalog: Catalog, q: BoundQuery, t: Int, needed: List[Int], planner: P, opts: PlannerOptions, trust: Bool, build: Bool = False) raises -> JoinSide:
    """Side `t` of the join: the source table itself when nothing filters it and it is a plain table, else the result of a row query."""
    opts.job.check()                                              # F-175: between the steps of a join
    ref src = catalog.sources[q.sources[t]]
    var start = q.tab_start[t]
    var conj = q.conj_l.copy() if t == 0 else q.conj_r.copy()
    var keycols = q.jk_l.copy() if t == 0 else q.jk_r.copy()
    if len(conj) == 0 and src.kind == SRC_TABLE and len(src.overlay.mask) == 0 and len(src.overlay.delta) == 0:
        var cols = List[Int]()
        for g in needed:
            cols.append(g - start)
        var kc = List[Int]()
        for g in keycols:
            kc.append(g - start)
        return JoinSide(src.table, cols^, kc^, True)
    var sq = side_query(catalog, q, t, needed)
    sq.is_side = build                                              # F-158: only the side the hash table is built from counts against the join budget
    var r = _run_sql_once(catalog, sq, planner, opts, trust, String())
    var cols = List[Int]()
    for i in range(len(needed)):
        cols.append(i)
    var kc = List[Int]()
    for g in keycols:
        for i in range(len(needed)):
            if needed[i] == g:
                kc.append(i)
    return JoinSide(r.take_table().share(), cols^, kc^, False)


comptime DEFER_PASS_DEN = 2          # ... and when at least 1 row in this many passes its filter (else the few rows that pass are written: cheaper than probing them all)
comptime DEFER_MIN_ROWS = 65536      # a filtered table of at least this many rows is counted first and read unfiltered when it is the probe side of a fused join


def _can_defer(catalog: Catalog, q: BoundQuery, t: Int, allow: Bool) -> Bool:
    """Side `t` has a filter, is a plain table (no edits) of enough rows: its filtered rows need not be written when its join is fused (the pipeline filters)."""
    if not allow:
        return False
    var conj = (q.conj_l.copy() if t == 0 else q.conj_r.copy()) if len(q.sources) == 2 else q.leaf_conj[t].copy()
    if len(conj) == 0:
        return False
    ref src = catalog.sources[q.sources[t]]
    return src.kind == SRC_TABLE and len(src.overlay.mask) == 0 and len(src.overlay.delta) == 0 and src.table.rows() >= DEFER_MIN_ROWS


def _count_side[P: Planner](mut catalog: Catalog, q: BoundQuery, t: Int, planner: P, opts: PlannerOptions, trust: Bool) raises -> Int:
    var r = _run_sql_once(catalog, count_query(catalog, q, t), planner, opts, trust, String())
    return Int(r.table.get(0, 0))


def _any_hol(q: BoundQuery) -> Bool:
    for a in q.aggs:
        if a.hol:
            return True
    return False


def _plain_side(catalog: Catalog, q: BoundQuery, t: Int, needed: List[Int]) -> JoinSide:
    """Side `t` as its source table itself (nothing filtered: the filter, if any, runs in the pipeline)."""
    ref src = catalog.sources[q.sources[t]]
    var start = q.tab_start[t]
    var keycols = q.jk_l.copy() if t == 0 else q.jk_r.copy()
    var cols = List[Int]()
    for g in needed:
        cols.append(g - start)
    var kc = List[Int]()
    for g in keycols:
        kc.append(g - start)
    return JoinSide(src.table, cols^, kc^, True)


def _prefill_ranges_n(mut src: SqlSource, jq: BoundQuery, tabs: List[SharedTable], vtab: List[Int], vreal: List[Int], skip: List[Int] = List[Int]()):
    """`_prefill_ranges` for a star: virtual column c is column `vreal[c]` of table `tabs[vtab[c]]` (`skip[c]` != 0: its range is unknown, a streamed probe's)."""
    var cols = List[Int]()
    for k in jq.keys:
        _key_columns(jq, k.expr, cols)
    for c in cols:
        if len(skip) > 0 and skip[c] != 0:
            continue
        var r = _column_range(tabs[vtab[c]], vreal[c])
        src.range_cols.append(c)
        src.range_lo.append(r[1])
        src.range_hi.append(r[2])
        src.range_has.append(r[0])


def _tail_components(q: BoundQuery, tail: List[Int]) -> List[List[Int]]:
    """The dimensions of a star's tail grouped by the equalities between them: a dimension reached only through another (nation through supplier) is joined to
    it first, and the group probes the fact table as one."""
    var group = List[Int](length=len(tail), fill=-1)
    var n_groups = 0
    for i in range(len(tail)):
        if group[i] < 0:
            group[i] = n_groups
            var grew = True
            while grew:
                grew = False
                for j in range(len(tail)):
                    if group[j] >= 0:
                        continue
                    for k in range(len(q.edge_a)):
                        var ta = table_of_column(q, q.edge_a[k])
                        var tb = table_of_column(q, q.edge_b[k])
                        for i2 in range(len(tail)):
                            if group[i2] == n_groups and ((ta == tail[j] and tb == tail[i2]) or (tb == tail[j] and ta == tail[i2])):
                                group[j] = n_groups
                                grew = True
            n_groups += 1
    var out = List[List[Int]]()
    for g in range(n_groups):
        var l = List[Int]()
        for i in range(len(tail)):
            if group[i] == g:
                l.append(tail[i])
        out.append(l^)
    return out^


def _star_valid(q: BoundQuery, jp: JoinPlan, s0: Int) -> Bool:
    """Steps `s0` to the last can run as one star: the tables they join form dimension groups (each held together by equalities between its tables), every
    group reaches the tables joined before step `s0` by an equality (its keys are columns of the probe table), and a group of one table is a table the planner
    builds the hash table on. A group of several is joined first (small) and then probes as one."""
    var before = 1 << jp.start
    for j in range(s0):
        before |= 1 << jp.steps[j].leaf
    var tail = List[Int]()
    for j in range(s0, len(jp.steps)):
        tail.append(jp.steps[j].leaf)
    var comps = _tail_components(q, tail)
    if len(comps) < 2 and not (len(comps) == 1 and len(comps[0]) >= 2):
        return False
    for c in comps:
        var bits = 0
        for lf in c:
            bits |= 1 << lf
        var reaches = False
        for k in range(len(q.edge_a)):
            var ta = table_of_column(q, q.edge_a[k])
            var tb = table_of_column(q, q.edge_b[k])
            if ((bits >> ta) & 1 != 0 and (before >> tb) & 1 != 0) or ((bits >> tb) & 1 != 0 and (before >> ta) & 1 != 0):
                reaches = True
        if not reaches:
            return False
        for lf in c:                                                  # every table of a dimension group is one the planner builds the hash table on (a hub like the fact table is not)
            for j in range(s0, len(jp.steps)):
                if jp.steps[j].leaf == lf and not jp.steps[j].build_new:
                    return False
    return True


def _can_stream(catalog: Catalog, q: BoundQuery, t: Int, allow: Bool) -> Bool:
    """Side `t` is a Parquet scan or a Frame (edits, appends, removes, merges): it can probe a fused join chunk by chunk (a row group, the shared rows, the changed
    rows) without a copy of its rows being made, for an aggregate or a row query."""
    if not allow:
        return False
    ref src = catalog.sources[q.sources[t]]
    return src.kind == SRC_SCAN or src.kind == SRC_FRAME


def _probe_null_count(src: SqlSource, c: Int) -> Int:
    """Nulls a streamed probe column may hold: what the table or the file's statistics say, the changed rows' nulls too (unknown counts as one)."""
    var n = 0
    if src.kind == SRC_SCAN:
        var k = src.scan.value().null_count(c)
        n = 1 if k < 0 else k
    else:
        n = src.table.null_count(c)
    if n == 0 and _delta_nulls(src, c):
        n = 1
    return n


def _prefill_ranges_stream(mut src: SqlSource, jq: BoundQuery, bt: SharedTable, vstage: List[Int], vreal: List[Int]):
    """The value range of each group-key column that comes from the build side (a probe column's range is unknown: the keys are hashed)."""
    var cols = List[Int]()
    for k in jq.keys:
        _key_columns(jq, k.expr, cols)
    for c in cols:
        if vstage[c] < 0:
            continue
        var r = _column_range(bt, vreal[c])
        src.range_cols.append(c)
        src.range_lo.append(r[1])
        src.range_hi.append(r[2])
        src.range_has.append(r[0])


def _probe_words(src: SqlSource, cols: List[Int], mut words: List[List[String]], mut aliases: List[List[Int]]) raises:
    """For `make_key_plan`: each key column's words by code (the file's or table's, then a Frame's session words) and its merged codes."""
    var dact = DictAccess(src.table, src.scan, src.overlay.words)
    var all_alias = _alias_lists(src)
    for c in cols:
        var ws = List[String]()
        if src.table.kind(c) == TEXT:
            for code in range(dact.count(c)):
                ws.append(dact.word(c, code))
        words.append(ws^)
        aliases.append(all_alias[c].copy())


def _stream_schema_fix(mut vt: Table, src: SqlSource, vstage: List[Int], vreal: List[Int]) raises:
    """The virtual schema of a streamed join: its probe columns hold the source's dictionaries (shared) and may hold NULLs."""
    for v in range(len(vstage)):
        if vstage[v] < 0:
            var c = vreal[v]
            if src.table.kind(c) == TEXT:
                if src.scan:
                    src.scan.value().dictionary_into(vt, v, c)
                else:
                    vt.share_dictionary(v, src.table, c)
            vt.cols[v].nulls = _probe_null_count(src, c)


def _stream_overlay_fix(mut v0: SqlSource, src: SqlSource, vstage: List[Int], vreal: List[Int]):
    """... and a Frame's session words and merged codes ride on the virtual columns that are its columns."""
    for v in range(len(vstage)):
        if vstage[v] >= 0:
            continue
        var c = vreal[v]
        if c < len(src.overlay.words) and len(src.overlay.words[c]) > 0:
            while len(v0.overlay.words) <= v:
                v0.overlay.words.append(List[String]())
            v0.overlay.words[v] = src.overlay.words[c].copy()
        for item in src.overlay.aliases.items():
            if Int(item.key >> 32) == c:
                v0.overlay.aliases[(v << 32) | Int(item.key & 0xFFFFFFFF)] = item.value


def _stream_layout(src: SqlSource, vstage: List[Int], vreal: List[Int], key_cols: List[List[Int]], mut used: List[Int], mut pos: List[Int]):
    """The columns a chunk of the streamed source holds, in order (a scan: the ones the query and the keys read; a table or Frame: all), and where each is."""
    var ncs = src.table.columns()
    pos = List[Int](length=ncs, fill=-1)
    if src.kind == SRC_SCAN:
        var want = List[Bool](length=ncs, fill=False)
        for v in range(len(vstage)):
            if vstage[v] < 0:
                want[vreal[v]] = True
        for kc in key_cols:
            for c in kc:
                want[c] = True
        for c in range(ncs):
            if want[c]:
                pos[c] = len(used)
                used.append(c)
    else:
        for c in range(ncs):
            pos[c] = c
            used.append(c)


def _run_join_query[P: Planner](mut catalog: Catalog, q: BoundQuery, lp: LogicalPlan, planner: P, opts: PlannerOptions, trust: Bool, lie: String,
                                t0: Int) raises -> SqlResult:
    """A query over two tables: each side is reduced to the rows and columns the query needs (a row query over its source, with the filter
    that reads only it), the smaller side is the build side of the hash join (`kernels/hashjoin_cpu.mojo`), the matching pairs are gathered
    into one table, and the rest of the query (the filter that reads both sides, aggregates, ORDER BY, LIMIT) runs over that table."""
    opts.job.check()                                              # F-175: between the steps of a join
    var res = SqlResult()
    if q.explain != EXPLAIN_NONE:
        res.notes.append("join: " + _kind_name(q.join_kind) + ", on " + ", ".join(q.jk_l_names) + " = " + ", ".join(q.jk_r_names) +
                         (" (the build side is the side with fewer rows after its filter: decided when the query runs)" if q.join_kind == 0 else " (the build side is the right table)"))
        res.explain = _explain_text(lp, res.notes)
        var t = Table.empty()
        var c = t.add_column("plan", TEXT)
        for line in res.explain.split("\n"):
            if line.byte_length() > 0:
                t.append_text(c, String(line))
        res.table = t^
        res.ms = Float64(perf_counter_ns() - t0) / 1e6
        return res^
    var kind = q.join_kind
    var semi_anti = kind >= 2
    var lneed = needed_columns(q, 0)
    var rneed = needed_columns(q, 1)                                 # a semi / anti join: the key columns only (the right table has no columns after it)
    var t_sides = perf_counter_ns()
    var any_holistic = False
    for a in q.aggs:
        if a.hol:
            any_holistic = True
    # a big plain table with a filter, on the side that will probe, is counted first: when the join is fused its filter runs in the pipeline and no filtered
    # copy of it is written. A left, semi or anti join probes with its left table; an inner join with the larger side.
    var may0 = _can_defer(catalog, q, 0, opts.fuse and not any_holistic)
    var may1 = kind == 0 and _can_defer(catalog, q, 1, opts.fuse and not any_holistic)
    # a scan or a Frame on the side that probes is not read into a table at all: its chunks go through the fused join one at a time (F-158)
    var str0 = _can_stream(catalog, q, 0, opts.fuse and not any_holistic)
    var str1 = kind == 0 and _can_stream(catalog, q, 1, opts.fuse and not any_holistic)
    var left = _plain_side(catalog, q, 0, lneed)
    var right = _plain_side(catalog, q, 1, rneed)
    var lrows = 0
    var rrows = 0
    var pf0 = False                                                  # F-161: side t's selective filter is left to the pipeline, which runs it before the probe
    var pf1 = False
    if str0:
        lrows = catalog.sources[q.sources[0]].rows()                 # (its rows before the filter: the side is not read to count them)
    elif may0:
        lrows = _count_side[P](catalog, q, 0, planner, opts, trust)
        if lrows * DEFER_PASS_DEN < catalog.sources[q.sources[0]].table.rows():      # a selective filter: writing the few rows that pass beats probing them all ...
            if len(q.conj_x) == 0:
                pf0 = True                                       # ... unless the pipeline runs the filter first and probes only those rows (F-161): nothing is written
            else:
                may0 = False
                left = _join_side[P](catalog, q, 0, lneed, planner, opts, trust)
    else:
        left = _join_side[P](catalog, q, 0, lneed, planner, opts, trust)
        lrows = left.table.rows()
    if str1:
        rrows = catalog.sources[q.sources[1]].rows()
    elif may1:
        rrows = _count_side[P](catalog, q, 1, planner, opts, trust)
        if rrows * DEFER_PASS_DEN < catalog.sources[q.sources[1]].table.rows():
            if len(q.conj_x) == 0:
                pf1 = True
            else:
                may1 = False
                right = _join_side[P](catalog, q, 1, rneed, planner, opts, trust)
    else:
        right = _join_side[P](catalog, q, 1, rneed, planner, opts, trust)
        rrows = right.table.rows()
    var left_builds = kind == 0 and lrows < rrows                    # left, semi and anti joins build on the right table
    if kind == 0 and opts.build_side == BUILD_LEFT:
        left_builds = True
    elif kind == 0 and opts.build_side == BUILD_RIGHT:
        left_builds = False
    var defer_probe = -1                                             # the side whose filter is left to the pipeline, or -1
    var streaming = False                                            # ... and it is a scan or a Frame whose chunks probe one at a time
    if (str0 or may0) and left_builds:
        left = _join_side[P](catalog, q, 0, lneed, planner, opts, trust, True)       # the build side is read
        lrows = left.table.rows()
    elif str0:
        defer_probe = 0
        streaming = True
    elif may0:
        defer_probe = 0
    if (str1 or may1) and not left_builds:
        right = _join_side[P](catalog, q, 1, rneed, planner, opts, trust, True)
        rrows = right.table.rows()
    elif str1:
        defer_probe = 1
        streaming = True
    elif may1:
        defer_probe = 1
    var qe = copy_query(q)
    if defer_probe >= 0:
        qe = with_probe_filter(q, defer_probe)
        if defer_probe == 0:
            lneed = needed_columns(qe, 0)
            left = _plain_side(catalog, qe, 0, lneed)
        else:
            rneed = needed_columns(qe, 1)
            right = _plain_side(catalog, qe, 1, rneed)
    ref built = left if left_builds else right                       # a build side read as a row query (not the source table itself) counts against the budget
    if not built.zero_copy and built.table.rows() * max(len(built.cols), 1) * 8 > (opts.join_budget if opts.join_budget > 0 else JOIN_BUDGET_BYTES):
        raise build_budget_error(catalog.sources[q.sources[0 if left_builds else 1]].name, built.table.rows(), len(built.cols), opts.join_budget if opts.join_budget > 0 else JOIN_BUDGET_BYTES)
    var ms_sides = Float64(perf_counter_ns() - t_sides) / 1e6
    var t_keys = perf_counter_ns()
    var keys = JoinKeys()
    var m = JoinMap()
    var fmode = 0                                                    # the front stage's mode: 0 inner, 1 left, else a presence filter
    var na_note = String()
    var t_build = perf_counter_ns()
    var t_probe = perf_counter_ns()
    var fusable = False
    var expands = False
    var pass_no = 0
    var kplan = KeyPlan()
    var probe_src = SqlSource("", Table.empty().share())             # a streamed probe's source (a copy of the catalog's)
    var probe_keys_src = List[Int]()                                 # ... and its key columns by source number
    if streaming:
        probe_src = catalog.sources[q.sources[defer_probe]].copy()
        for g in (q.jk_l if defer_probe == 0 else q.jk_r):
            probe_keys_src.append(g - q.tab_start[defer_probe])
    while True:
        pass_no += 1
        keys = JoinKeys()
        if streaming:
            # the probe's chunks are keyed one at a time by a plan made once from the build side
            ref bs = left if left_builds else right
            var pw = List[List[String]]()
            var pal = List[List[Int]]()
            _probe_words(probe_src, probe_keys_src, pw, pal)
            kplan = make_key_plan(bs.table, probe_src.table, bs.key_cols, probe_keys_src, pw, pal, keys)
        elif left_builds:
            normalize_keys(left.table, right.table, left.key_cols, right.key_cols, keys, opts.threads)
        else:
            normalize_keys(right.table, left.table, right.key_cols, left.key_cols, keys, opts.threads)
        var nb0 = left.table.rows() if left_builds else right.table.rows()
        fmode = 0
        na_note = String()
        if kind == 1:
            fmode = 1
        elif semi_anti:
            fmode = PM_SEMI if kind == 2 else PM_ANTI
            if kind == 4:                                            # NOT IN: an empty list keeps every row; a NULL in the list keeps none; else a NULL key is dropped
                if rrows == 0:
                    na_note = " (the list is empty: every row is kept, a NULL key too)"
                elif right.table.null_count(right.key_cols[0]) > 0:
                    fmode = PM_NONE
                    na_note = " (the list holds a NULL: no row passes)"
                else:
                    fmode = PM_ANTI_NA
                    na_note = " (a NULL key is dropped)"
        t_build = perf_counter_ns()
        m = hj_build(keys.bk_addr, keys.bo_addr, nb0, opts.threads, semi_anti)
        if fmode == PM_ANTI_NA and not streaming:
            unmatchable_not_null(keys, m.lo, m.hi, left.table, left.key_cols[0], left.table.rows())
        t_probe = perf_counter_ns()
        expands = kind <= 1 and not q.rows and not m.unique and m.maxrep <= EXPAND_MAX_REP        # F-161: an aggregate over repeated build keys expands in the front stage
        fusable = opts.fuse and not keys.verify and not any_holistic and (m.unique or semi_anti or expands)
        if streaming and (kplan.hashed or fmode == PM_ANTI_NA):
            fusable = False                                          # (hashed tuples are verified per pair; NOT IN's unmatchable words are settled over the whole probe side)
        if defer_probe >= 0 and not fusable and pass_no == 1:
            # the join cannot be fused after all (repeated build keys, hashed keys): read the probe side filtered, as before
            if defer_probe == 0:
                lneed = needed_columns(q, 0)
                left = _join_side[P](catalog, q, 0, lneed, planner, opts, trust)
            else:
                rneed = needed_columns(q, 1)
                right = _join_side[P](catalog, q, 1, rneed, planner, opts, trust)
            qe = copy_query(q)
            defer_probe = -1
            streaming = False
            continue
        break
    var nb = left.table.rows() if left_builds else right.table.rows()
    var np = right.table.rows() if left_builds else left.table.rows()
    if streaming:
        np = probe_src.rows()
    if semi_anti:
        right.cols = List[Int]()                                     # nothing of the right table is carried
        rneed = List[Int]()
    var ncols = len(lneed) + len(rneed)
    var budget = (opts.join_budget if opts.join_budget > 0 else JOIN_BUDGET_BYTES) // (8 * max(ncols, 1))
    res.notes.append("join: " + _kind_name(kind) + " on " + ", ".join(q.jk_l_names) + " = " + ", ".join(q.jk_r_names) + ": " + commas(lrows) + " rows of " + _side_names(q, 0) +
                     " and " + commas(rrows) + " rows of " + _side_names(q, 1) + " after their filters" +
                     (" (" + _side_names(q, 0) + " read in place)" if left.zero_copy else "") + (" (" + _side_names(q, 1) + " read in place)" if right.zero_copy else "") + na_note)
    if streaming:
        res.notes.append("join probe stream: " + _side_names(q, defer_probe) + " (" + ("a Parquet scan of " + String(probe_src.scan.value().row_groups()) + " row groups" if probe_src.kind == SRC_SCAN else "a Frame: its shared rows, then its " + commas(len(probe_src.overlay.delta)) + " changed or appended rows") +
                         ") is never read into a table: each chunk's keys are planned from the build side and probe the shared map" +
                         (", its filter runs in the pipeline" if len(q.conj_l if defer_probe == 0 else q.conj_r) > 0 else ""))
    elif defer_probe >= 0:
        res.notes.append("join probe filter: the filter of " + _side_names(q, defer_probe) + " runs in the pipeline (" + commas(left.table.rows() if defer_probe == 0 else right.table.rows()) +
                         " rows read, " + commas(lrows if defer_probe == 0 else rrows) + " pass): no filtered copy of it is written")
    res.notes.append("join build side: " + (_side_names(q, 0) if left_builds else _side_names(q, 1)) + " (" + commas(nb) + " rows, " +
                     ("the smaller side" if (kind == 0 and opts.build_side == BUILD_AUTO) else "forced by the options" if kind == 0 else "the right side: a " + _kind_name(kind) + " join builds on it") + "): " + m.kind())
    res.notes.append("join keys: " + keys.layout)
    res.notes.append("join time: sides " + _ms(ms_sides) + " ms, keys " + _ms(Float64(t_build - t_keys) / 1e6) + " ms, build " + _ms(Float64(t_probe - t_build) / 1e6) + " ms")
    var jq = joined_query(qe, lneed, rneed)
    var cat2 = Catalog()
    var inner: SqlResult
    if fusable and streaming:
        var bside = left.table if left_builds else right.table
        var vstage = List[Int]()
        var vreal_s = List[Int]()
        for c in left.cols:
            vstage.append(-1 if defer_probe == 0 else 0)
            vreal_s.append(c)
        for c in right.cols:
            vstage.append(-1 if defer_probe == 1 else 0)
            vreal_s.append(c)
        var used = List[Int]()
        var pos = List[Int]()
        var kcs = List[List[Int]]()
        kcs.append(probe_keys_src.copy())
        _stream_layout(probe_src, vstage, vreal_s, kcs, used, pos)
        # the virtual schema: the probe's columns (kinds, dictionaries, nullability from the source) then the build side's
        var vt = virtual_schema(left, right, kind == 1)
        _stream_schema_fix(vt, probe_src, vstage, vreal_s)
        cat2.add_table("join", vt^.share())
        cat2.sources[0].fixed_rows = np
        _stream_overlay_fix(cat2.sources[0], probe_src, vstage, vreal_s)
        _prefill_ranges_stream(cat2.sources[0], jq, bside, vstage, vreal_s)
        var maxrep = m.maxrep
        var stages = List[StreamStage]()
        stages.append(StreamStage(bside, ArcPointer(m^), kplan^, probe_keys_src.copy()))
        var ps = ArcPointer(ProbeStream(probe_src.copy(), used^, pos^, stages^, fmode, False, vstage^, vreal_s^))
        var template = ArcPointer(stream_template(ps))
        res.notes.append("join path: fused: " + commas(np) + " probe rows (before the filter) of " + String(len(ps[].used)) + " columns arrive chunk by chunk, " +
                         ("each looks its partner up in blocks of 1,024" if not semi_anti else "each is kept or dropped by whether its key is present in the build side, in blocks of 1,024") +
                         " (no joined row is written, no copy of the probe side)" + (" (a row without a partner stays: its build-side columns are NULL)" if kind == 1 else "") + _expand_note(expands, maxrep))
        inner = _run_sql_once(cat2, jq, planner, opts, trust, lie, template)
    elif fusable:
        # fused: no joined row is written; the pipeline probes block by block (keeps or drops the rows by the join's rule) and gathers the build side's columns into scratch
        var vprobe = List[Bool]()
        var vreal = List[Int]()
        for c in left.cols:
            vprobe.append(not left_builds)
            vreal.append(c)
        for c in right.cols:
            vprobe.append(left_builds)
            vreal.append(c)
        var vt = virtual_schema(left, right, kind == 1)
        cat2.add_table("join", vt^.share())
        cat2.sources[0].fixed_rows = np
        var pt = right.table if left_builds else left.table
        var bt = left.table if left_builds else right.table
        _prefill_ranges(cat2.sources[0], jq, pt, bt, vprobe, vreal)
        var fjv = FusedJoin(pt, bt, vprobe^, vreal^, ArcPointer(m^), keys^, fmode)
        fjv.pred_first = defer_probe >= 0 and (pf0 if defer_probe == 0 else pf1)          # F-161: the selective probe filter runs before the probe
        var fj = ArcPointer(fjv^)
        if semi_anti:
            res.notes.append("join path: fused: " + commas(np) + " probe rows are " +
                             ("kept when their key is present in the build side" if kind == 2 else "dropped when their key is present in the build side" if kind == 3 else "kept when their key is absent from the list") +
                             ", in blocks of 1,024 (" + _kind_name(kind) + " join: no build-side column is gathered, no joined row is written)")
        else:
            res.notes.append("join path: fused: " + commas(np) + " probe rows look their partner up in blocks of 1,024 and " + String(len(fj[].partner)) +
                             " build-side " + ("column is" if len(fj[].partner) == 1 else "columns are") + " gathered into per-worker scratch (no joined row is written)" +
                             (" (a row without a partner stays: its build-side columns are NULL)" if kind == 1 else "") + _expand_note(expands, fj[].map[].maxrep))
        inner = _run_sql_once(cat2, jq, planner, opts, trust, lie, fj)
    else:
        var t_gather = perf_counter_ns()
        var jt: Table
        var n_out: Int
        if semi_anti:
            var pick = select_rows(m, keys, np, fmode, opts.threads)
            n_out = pick.n
            jt = gather_join(left, right, pick, False, opts.threads)
        else:
            var pairs = join_pairs(m, keys, np, budget, opts.threads, kind == 1)
            n_out = pairs.n
            jt = gather_join(left, right, pairs, left_builds, opts.threads, kind == 1)
        var ms_gather = Float64(perf_counter_ns() - t_gather) / 1e6
        var why = String()
        if not opts.fuse:
            why = " (fusing is off in the options)"
        elif any_holistic:
            why = " (a holistic aggregate needs every joined row)"
        elif keys.verify:
            why = " (the keys are hashed tuples, verified per pair)"
        elif not m.unique and not semi_anti and not q.rows and kind <= 1 and m.maxrep > EXPAND_MAX_REP:
            why = " (the build side's keys repeat, one up to " + commas(m.maxrep) + " times: past " + String(EXPAND_MAX_REP) + " the front stage would need a round for each)"
        elif not m.unique and not semi_anti:
            why = " (the build side's keys repeat)"
        if semi_anti:
            res.notes.append("join path: materialized: " + commas(n_out) + " of " + commas(np) + " rows pass the " + _kind_name(kind) + " join and are gathered into one table of " + String(ncols) + " columns" + why)
        else:
            res.notes.append("join path: materialized: " + commas(n_out) + (" pairs (unmatched rows included)" if kind == 1 else " matching pairs") + " gathered into one table of " + String(ncols) + " columns" + why)
        res.notes.append("join time: gather " + _ms(ms_gather) + " ms")
        cat2.add_table("join", jt^.share())
        inner = _run_sql_once(cat2, jq, planner, opts, trust, lie)
    for n in inner.notes:
        res.notes.append(n)
    res.actuals.append("join (" + _kind_name(kind) + "): " + commas(lrows) + " rows of " + _side_names(q, 0) + " and " + commas(rrows) + " rows of " + _side_names(q, 1) +
                       " after their filters: sides " + _ms(ms_sides) + " ms, keys " + _ms(Float64(t_build - t_keys) / 1e6) + " ms, build on " + commas(nb) + " rows " +
                       _ms(Float64(t_probe - t_build) / 1e6) + " ms")
    for a in inner.actuals:
        res.actuals.append(a)
    res.table = inner.take_table()
    res.explain = _explain_text(lp, res.notes)
    res.ms = Float64(perf_counter_ns() - t0) / 1e6
    return res^


struct _Leaf(Movable):
    """One table of a multi-way join after its own filters: the table (the source itself when nothing filters it, else a row query's result), the column of it
    that each needed global column is, and the global number of each of its columns."""
    var table: SharedTable
    var cols: List[Int]              # the table's column of each needed global column (ascending)
    var gcols: List[Int]             # ... those global columns
    var gtable: List[Int]            # the global column of each of the table's columns (-1: not one of the query's)
    var rows: Int
    var base_rows: Int
    var deferred: Bool               # a big plain table whose filter has only been counted: `table` is the unfiltered source (read filtered when it is the last join's probe side)
    var streamed: Bool               # a scan or a Frame, not read: `table` is its schema; it is read filtered unless it is a probe side, which streams

    def __init__(out self, table: SharedTable, var cols: List[Int], var gcols: List[Int], var gtable: List[Int], rows: Int, base_rows: Int, deferred: Bool = False, streamed: Bool = False):
        self.deferred = deferred
        self.streamed = streamed
        self.table = table
        self.cols = cols^
        self.gcols = gcols^
        self.gtable = gtable^
        self.rows = rows
        self.base_rows = base_rows


def _position(g: List[Int], col: Int) -> Int:
    for i in range(len(g)):
        if g[i] == col:
            return i
    return -1


def _label(q: BoundQuery, t: Int) -> String:
    return q.tab_alias[t] if q.tab_alias[t].byte_length() > 0 else q.tab_names[t]


def _read_filtered[P: Planner](mut catalog: Catalog, q: BoundQuery, t: Int, mut leaves: List[_Leaf], planner: P, opts: PlannerOptions, trust: Bool, mut res: SqlResult) raises:
    """A leaf that was only counted: its filtered rows, as a table of the columns later steps read."""
    opts.job.check()                                              # F-175: between the steps of a join
    var t_m = perf_counter_ns()
    var sq = side_query(catalog, q, t, leaves[t].gcols)
    var r = _run_sql_once(catalog, sq, planner, opts, trust, String())
    var tab = r.take_table().share()
    var cols2 = List[Int]()
    for i in range(len(leaves[t].gcols)):
        cols2.append(i)
    var gt = leaves[t].gcols.copy()
    var gc = leaves[t].gcols.copy()
    leaves[t] = _Leaf(tab, cols2^, gc^, gt^, tab.rows(), leaves[t].base_rows, False, False)
    res.actuals.append("read " + _label(q, t) + " filtered: " + commas(tab.rows()) + " rows, " + _ms(Float64(perf_counter_ns() - t_m) / 1e6) + " ms")


def _ndv_after_filter(l: _Leaf, ndv: Int) -> Int:
    """F-202: the distinct values of a column of a leaf that is only counted (its table is unfiltered): `ndv` values spread over `base_rows` rows, `rows` of them kept
    at random, leave ndv x (1 - (1 - rows / base_rows) ^ (base_rows / ndv)) of them."""
    if not l.deferred or l.rows >= l.base_rows or ndv <= 1 or l.base_rows <= 0:
        return ndv
    if l.rows <= 0:
        return 0
    var per = Float64(l.base_rows) / Float64(ndv)
    var keep = Float64(ndv) * (1.0 - exp(per * log(1.0 - Float64(l.rows) / Float64(l.base_rows))))
    return max(1, min(ndv, Int(keep + 0.5)))


def _copy_leaf(l: _Leaf) -> _Leaf:
    return _Leaf(l.table, l.cols.copy(), l.gcols.copy(), l.gtable.copy(), l.rows, l.base_rows, l.deferred, l.streamed)


def _build_cluster[P: Planner](q: BoundQuery, leaves: List[_Leaf], js: JoinStats, cluster: List[Int], keep: List[Int], planner: P, opts: PlannerOptions, threads: Int,
                               budget_bytes: Int, mut res: SqlResult) raises -> _Leaf:
    """The tables of a dimension group joined with each other (the planner orders them), keeping the global columns `keep`: one table to build a star's hash table on."""
    opts.job.check()                                              # F-175: between the steps of a join
    var sub = JoinStats()
    for lf in cluster:
        sub.names.append(js.names[lf])
        sub.base_rows.append(js.base_rows[lf])
        sub.rows.append(js.rows[lf])
    for k in range(len(js.edge_a)):
        var ia = _position(cluster, js.edge_a[k])
        var ib = _position(cluster, js.edge_b[k])
        if ia >= 0 and ib >= 0:
            sub.edge_a.append(ia)
            sub.edge_b.append(ib)
            sub.ndv_a.append(js.ndv_a[k])
            sub.ndv_b.append(js.ndv_b[k])
    var o = opts.copy()
    o.join_order = List[Int]()
    o.written_order = False
    var sp = planner.join_order(sub, o)
    var first = cluster[sp.start]
    var cur_table = leaves[first].table
    var cur_g = leaves[first].gcols.copy()
    var cur_pos = leaves[first].cols.copy()
    var joined = 1 << first
    for i in range(len(sp.steps)):
        var t_step = perf_counter_ns()
        var lf = cluster[sp.steps[i].leaf]
        var lkey = List[Int]()
        var rkey = List[Int]()
        for k in range(len(q.edge_a)):
            var ga = q.edge_a[k]
            var gb = q.edge_b[k]
            var ta = table_of_column(q, ga)
            var tb = table_of_column(q, gb)
            if ta == lf and (joined >> tb) & 1 != 0:
                lkey.append(cur_pos[_position(cur_g, gb)])
                rkey.append(leaves[lf].cols[_position(leaves[lf].gcols, ga)])
            elif tb == lf and (joined >> ta) & 1 != 0:
                lkey.append(cur_pos[_position(cur_g, ga)])
                rkey.append(leaves[lf].cols[_position(leaves[lf].gcols, gb)])
        var stepped = _join_step(q, cur_table, cur_g, cur_pos, leaves[lf], keep, lkey^, rkey^, sp.steps[i].build_new, sp.steps[i].est_rows, _label(q, lf),
                                 "star dimension group, join " + String(i + 1), threads, budget_bytes, res, t_step)
        cur_g = stepped[1].copy()
        cur_table = stepped[0]
        cur_pos = List[Int]()
        for c in range(len(cur_g)):
            cur_pos.append(c)
        joined |= 1 << lf
    var ident = List[Int]()
    for c in range(len(cur_g)):
        ident.append(c)
    return _Leaf(cur_table, ident^, cur_g.copy(), cur_g.copy(), cur_table.rows(), 0, False)


def _run_star[P: Planner](q: BoundQuery, leaves: List[_Leaf], js: JoinStats, jp: JoinPlan, s0: Int, before: Int, probe: SharedTable, probe_gt: List[Int], conj0: List[Int],
                          psrc: SqlSource, streamed: Bool, done: List[Bool], planner: P, opts: PlannerOptions, trust: Bool, lie: String, mut res: SqlResult, mut out: SqlResult) raises -> Bool:
    """Steps `s0` to the last as one star: the probe table (the rows joined so far, or a table read unfiltered whose filter `conj0` runs in the pipeline) meets each
    dimension (a table, or a group of tables joined with each other first) through a map built on the dimension's unique keys, all in one pass over the probe rows
    (no joined row is written, no intermediate table). False (nothing changed but notes) when a dimension's keys are not unique or are hashed: the steps run one by one."""
    var threads = opts.threads
    var budget_bytes = opts.join_budget if opts.join_budget > 0 else JOIN_BUDGET_BYTES
    var t_star = perf_counter_ns()
    var conj = conj0.copy()
    for k in range(len(q.resid)):
        if not done[k]:
            conj.append(q.resid[k])
    var need = final_needed(q, conj)
    var tail = List[Int]()
    for j in range(s0, len(jp.steps)):
        tail.append(jp.steps[j].leaf)
    var comps = _tail_components(q, tail)
    var stages = List[StageData]()
    var sstages = List[StreamStage]()
    var dim_tabs = List[SharedTable]()
    var dim_gt = List[List[Int]]()
    var dim_name = List[String]()
    for cl in comps:
        var bits = 0
        var label = String()
        for lf in cl:
            bits |= 1 << lf
            label += ("+" if label.byte_length() > 0 else "") + _label(q, lf)
        var stage_leaf: _Leaf
        if len(cl) == 1:
            stage_leaf = _copy_leaf(leaves[cl[0]])
        else:
            # the columns the group keeps: what the query reads, the keys to the probe table, and the keys that join its own tables
            var keep = List[Int]()
            for g in range(len(need)):
                if need[g] and (bits >> table_of_column(q, g)) & 1 != 0:
                    keep.append(g)
            for k in range(len(q.edge_a)):
                var ta = table_of_column(q, q.edge_a[k])
                var tb = table_of_column(q, q.edge_b[k])
                if (bits >> ta) & 1 != 0 and _position(keep, q.edge_a[k]) < 0:
                    keep.append(q.edge_a[k])
                if (bits >> tb) & 1 != 0 and _position(keep, q.edge_b[k]) < 0:
                    keep.append(q.edge_b[k])
            stage_leaf = _build_cluster[P](q, leaves, js, cl, keep, planner, opts, threads, budget_bytes, res)
        var lkey = List[Int]()
        var rkey = List[Int]()
        for k in range(len(q.edge_a)):
            var ga = q.edge_a[k]
            var gb = q.edge_b[k]
            var ta = table_of_column(q, ga)
            var tb = table_of_column(q, gb)
            if (bits >> ta) & 1 != 0 and (before >> tb) & 1 != 0:
                lkey.append(_position(probe_gt, gb))
                rkey.append(_position(stage_leaf.gtable, ga))
            elif (bits >> tb) & 1 != 0 and (before >> ta) & 1 != 0:
                lkey.append(_position(probe_gt, ga))
                rkey.append(_position(stage_leaf.gtable, gb))
        var keys = JoinKeys()
        var kplan = KeyPlan()
        if streamed:
            # the probe arrives in chunks: its keys are planned once from the dimension, applied to each chunk
            var pw = List[List[String]]()
            var pal = List[List[Int]]()
            _probe_words(psrc, lkey, pw, pal)
            kplan = make_key_plan(stage_leaf.table, psrc.table, rkey, lkey, pw, pal, keys)
        else:
            normalize_keys(stage_leaf.table, probe, rkey, lkey, keys, threads)
        var m = hj_build(keys.bk_addr, keys.bo_addr, stage_leaf.table.rows(), threads)
        if keys.verify or kplan.hashed or not m.unique:
            res.notes.append("star: " + label + " has " + ("hashed keys" if (keys.verify or kplan.hashed) else "repeated keys") + ": the joins run one by one")
            return False
        res.notes.append("star dimension " + label + ": " + commas(stage_leaf.table.rows()) + " rows, " + m.kind() + ", " + (kplan.layout if streamed else keys.layout))
        if streamed:
            sstages.append(StreamStage(stage_leaf.table, ArcPointer(m^), kplan^, lkey.copy()))
        else:
            stages.append(StageData(ArcPointer(m^), keys^))
        dim_tabs.append(stage_leaf.table)
        dim_gt.append(stage_leaf.gtable.copy())
        dim_name.append(label)
    var tabs = List[SharedTable]()
    var colsets = List[List[Int]]()
    var vglobal = List[Int]()
    var vstage = List[Int]()
    var vreal = List[Int]()
    var vtab = List[Int]()
    tabs.append(probe)
    var pc = List[Int]()
    for c in range(probe.columns()):
        if probe_gt[c] >= 0 and need[probe_gt[c]]:
            pc.append(c)
            vglobal.append(probe_gt[c])
            vstage.append(-1)
            vreal.append(c)
            vtab.append(0)
    colsets.append(pc^)
    var n_dim_cols = 0
    for si in range(len(dim_tabs)):
        tabs.append(dim_tabs[si])
        var dc = List[Int]()
        for c in range(dim_tabs[si].columns()):
            var g = dim_gt[si][c]
            if g >= 0 and need[g]:
                dc.append(c)
                vglobal.append(g)
                vstage.append(si)
                vreal.append(c)
                vtab.append(si + 1)
                n_dim_cols += 1
        colsets.append(dc^)
    var vt = virtual_schema_multi(tabs, colsets)
    var cat2 = Catalog()
    var psrc_rows = probe.rows()
    if streamed:
        _stream_schema_fix(vt, psrc, vstage, vreal)
        psrc_rows = psrc.rows()
    cat2.add_table("join", vt^.share())
    cat2.sources[0].fixed_rows = psrc_rows
    if streamed:
        _stream_overlay_fix(cat2.sources[0], psrc, vstage, vreal)
    var jq = star_query(q, vglobal, conj)
    if streamed:
        var skip = List[Int]()
        for c in range(len(vtab)):
            skip.append(0 if vtab[c] == 0 else 1)
        _prefill_ranges_n(cat2.sources[0], jq, tabs, vtab, vreal, skip)
    else:
        _prefill_ranges_n(cat2.sources[0], jq, tabs, vtab, vreal)
    var names = String()
    for si in range(len(dim_name)):
        names += (", " if si > 0 else "") + dim_name[si]
    res.notes.append("join path: fused: a star: " + commas(psrc_rows) + " probe rows" + (" (a scan or Frame streamed chunk by chunk, never read into a table)" if streamed else "") + " meet " + String(len(dim_name)) + " dimensions (" + names + ") with unique keys in one pass of blocks of 1,024; " +
                     String(n_dim_cols) + " dimension columns are gathered into per-worker scratch (no joined row and no intermediate table is written)" +
                     (" (the probe table's filter runs in the pipeline)" if len(conj0) > 0 else ""))
    if streamed:
        var used = List[Int]()
        var pos = List[Int]()
        var kcs = List[List[Int]]()
        for i in range(len(sstages)):                               # (an index loop: `for st in sstages` over this List crashes the compiler, docs/lessons.md)
            kcs.append(sstages[i].pcols.copy())
        _stream_layout(psrc, vstage, vreal, kcs, used, pos)
        var ps = ArcPointer(ProbeStream(psrc.copy(), used^, pos^, sstages^, 0, True, vstage^, vreal^))
        var template = ArcPointer(stream_template(ps))
        out = _run_sql_once(cat2, jq, planner, opts, trust, lie, template)
    else:
        var fj = ArcPointer(FusedJoin(probe, dim_tabs^, vstage^, vreal^, stages^))
        out = _run_sql_once(cat2, jq, planner, opts, trust, lie, fj)
    res.actuals.append("star (" + names + "): " + commas(psrc_rows) + " probe rows, " + _ms(Float64(perf_counter_ns() - t_star) / 1e6) + " ms including the aggregate")
    return True


def _join_step(q: BoundQuery, cur_table: SharedTable, cur_g: List[Int], cur_pos: List[Int], leaf: _Leaf, keep: List[Int], var lkey: List[Int], var rkey: List[Int],
               build_new: Bool, est: Int, label: String, title: String, threads: Int, budget_bytes: Int, mut res: SqlResult, t_step: Int) raises -> Tuple[SharedTable, List[Int]]:
    """One join of the rows joined so far (`cur_table`, global column of each table column in `cur_g`, the table column of each in `cur_pos`) with a leaf on the key
    columns `lkey` / `rkey` (table columns), keeping the global columns `keep`; the pairs are gathered into a new table. Returns it and the global column of each of its
    columns."""
    var lkeep = List[Int]()                                          # the table columns each side carries on
    var lkeep_g = List[Int]()
    for i in range(len(cur_g)):
        if _position(keep, cur_g[i]) >= 0:
            lkeep.append(cur_pos[i])
            lkeep_g.append(cur_g[i])
    var rkeep = List[Int]()
    var rkeep_g = List[Int]()
    for i in range(len(leaf.gcols)):
        if _position(keep, leaf.gcols[i]) >= 0:
            rkeep.append(leaf.cols[i])
            rkeep_g.append(leaf.gcols[i])
    var left = JoinSide(cur_table, lkeep^, lkey^, False)
    var right = JoinSide(leaf.table, rkeep^, rkey^, False)
    var left_builds = not build_new
    var keys = JoinKeys()
    if left_builds:
        normalize_keys(left.table, right.table, left.key_cols, right.key_cols, keys, threads)
    else:
        normalize_keys(right.table, left.table, right.key_cols, left.key_cols, keys, threads)
    var nb = left.table.rows() if left_builds else right.table.rows()
    var np = right.table.rows() if left_builds else left.table.rows()
    var m = hj_build(keys.bk_addr, keys.bo_addr, nb, threads)
    var build_kind = m.kind()
    var pairs = join_pairs(m, keys, np, budget_bytes // (8 * max(len(lkeep_g) + len(rkeep_g), 1)), threads)
    var jt = gather_join(left, right, pairs, left_builds, threads)
    var out_rows = jt.rows()
    res.notes.append(title + ": " + commas(left.table.rows()) + " rows with " + commas(right.table.rows()) + " rows of " + label + " on " + String(len(left.key_cols)) +
                     (" key (" if len(left.key_cols) == 1 else " keys (") + keys.layout + "), hash table on " + commas(nb) + " rows: " + build_kind + ", " + commas(out_rows) + " rows out (estimated " + commas(est) + ")")
    res.actuals.append(title + " (" + label + "): " + commas(left.table.rows()) + " x " + commas(right.table.rows()) + " -> " + commas(out_rows) + " rows (estimated " +
                       commas(est) + "), build " + commas(nb) + " rows, " + _ms(Float64(perf_counter_ns() - t_step) / 1e6) + " ms")
    var new_g = lkeep_g^
    for g in rkeep_g:
        new_g.append(g)
    return (jt^.share(), new_g^)


def _run_multiway[P: Planner](mut catalog: Catalog, q: BoundQuery, lp: LogicalPlan, planner: P, opts: PlannerOptions, trust: Bool, lie: String,
                              t0: Int) raises -> SqlResult:
    """Three or more tables joined by equalities (inner joins). Each table runs its own filter first (a row query over its source), so the planner works
    from exact row counts; the planner (`Planner.join_order`) picks the order and the build side of each join; every join but the last materializes its pairs
    (each keeps only the columns later steps read, and the conditions across tables are checked as soon as the tables they read are joined); the last is the
    two-table join of `_run_join_query` with the rest of the query (aggregates, ORDER BY, LIMIT), fused when the build side's keys are unique."""
    opts.job.check()                                              # F-175: between the steps of a join
    var res = SqlResult()
    var n = len(q.sources)
    var threads = opts.threads
    var budget_bytes = opts.join_budget if opts.join_budget > 0 else JOIN_BUDGET_BYTES
    var needed = all_needed(q)
    var leaves = List[_Leaf]()
    var js = JoinStats()
    for t in range(n):
        var t_leaf = perf_counter_ns()
        var start = q.tab_start[t]
        var gcols = List[Int]()
        for g in needed:
            if table_of_column(q, g) == t:
                gcols.append(g)
        var src_table = catalog.sources[q.sources[t]].table
        var plain = catalog.sources[q.sources[t]].kind == SRC_TABLE and len(catalog.sources[q.sources[t]].overlay.mask) == 0 and len(catalog.sources[q.sources[t]].overlay.delta) == 0
        var base_rows = src_table.rows()
        var table = src_table
        var cols = List[Int]()
        var gtable = List[Int]()
        var is_deferred = False
        var is_streamed = False
        var counted = -1
        if len(q.leaf_conj[t]) == 0 and plain:
            table = src_table
            for g in gcols:
                cols.append(g - start)
            for c in range(src_table.columns()):
                gtable.append(start + c)
        elif _can_defer(catalog, q, t, opts.fuse and not _any_hol(q)):
            counted = _count_side[P](catalog, q, t, planner, opts, trust)          # its rows are only counted for now
            is_deferred = True                                         # F-202: always: read filtered only if the order does not make it the last join's probe side
            table = src_table
            for g in gcols:
                cols.append(g - start)
            for c in range(src_table.columns()):
                gtable.append(start + c)
        elif _can_stream(catalog, q, t, opts.fuse and not _any_hol(q)):
            # a scan or a Frame: not read now; it streams if it ends up a probe side, else it is read filtered once the order is known
            is_deferred = True
            is_streamed = True
            counted = catalog.sources[q.sources[t]].rows()
            base_rows = counted
            table = src_table
            for g in gcols:
                cols.append(g - start)
            for c in range(src_table.columns()):
                gtable.append(start + c)
        if not (len(q.leaf_conj[t]) == 0 and plain) and not is_deferred:
            cols = List[Int]()
            gtable = List[Int]()
            var sq = side_query(catalog, q, t, gcols)
            var r = _run_sql_once(catalog, sq, planner, opts, trust, String())
            table = r.take_table().share()
            for i in range(len(gcols)):
                cols.append(i)
                gtable.append(gcols[i])
        var rows = counted if is_deferred else table.rows()
        js.names.append(_label(q, t))
        js.base_rows.append(base_rows)
        js.rows.append(rows)
        js.deferred.append(is_deferred and not is_streamed)
        res.actuals.append("scan " + _label(q, t) + ": " + commas(base_rows) + " rows, " + commas(rows) + " after " + ("its filter" if len(q.leaf_conj[t]) > 0 else "no filter") +
                           " (" + String(len(gcols)) + " columns kept" + (", counted: its filter may run in the last join's pipeline" if is_deferred else "") + "), " + _ms(Float64(perf_counter_ns() - t_leaf) / 1e6) + " ms")
        leaves.append(_Leaf(table, cols^, gcols^, gtable^, rows, base_rows, is_deferred, is_streamed))
    var t_ndv = perf_counter_ns()
    for k in range(len(q.edge_a)):
        var ta = table_of_column(q, q.edge_a[k])
        var tb = table_of_column(q, q.edge_b[k])
        js.edge_a.append(ta)
        js.edge_b.append(tb)
        js.ndv_a.append(-1 if leaves[ta].streamed else _ndv_after_filter(leaves[ta], column_ndv(leaves[ta].table, leaves[ta].cols[_position(leaves[ta].gcols, q.edge_a[k])], threads)))
        js.ndv_b.append(-1 if leaves[tb].streamed else _ndv_after_filter(leaves[tb], column_ndv(leaves[tb].table, leaves[tb].cols[_position(leaves[tb].gcols, q.edge_b[k])], threads)))
    for k in range(len(q.edge_a)):
        res.notes.append("statistics: " + _label(q, js.edge_a[k]) + "." + q.src_name(q.edge_a[k]) + " has " + (commas(js.ndv_a[k]) if js.ndv_a[k] >= 0 else String("an unknown number of")) + " distinct values in " + commas(js.rows[js.edge_a[k]]) +
                         " rows, " + _label(q, js.edge_b[k]) + "." + q.src_name(q.edge_b[k]) + " " + (commas(js.ndv_b[k]) if js.ndv_b[k] >= 0 else String("an unknown number")) + " in " + commas(js.rows[js.edge_b[k]]))
    res.actuals.append("distinct counts of the join columns: " + _ms(Float64(perf_counter_ns() - t_ndv) / 1e6) + " ms")
    var jp = planner.join_order(js, opts)
    for nt in jp.notes:
        res.notes.append(nt)
    if q.explain != EXPLAIN_NONE and q.explain != EXPLAIN_ANALYZE:
        res.explain = _explain_text(lp, res.notes)
        var tb = Table.empty()
        var c = tb.add_column("plan", TEXT)
        for line in res.explain.split("\n"):
            if line.byte_length() > 0:
                tb.append_text(c, String(line))
        res.table = tb^
        res.ms = Float64(perf_counter_ns() - t0) / 1e6
        return res^
    # a tail of joins on dimensions with unique keys runs as one star (one pass over the probe rows probing every dimension), when it is at least two joins
    var star_s0 = -1
    if opts.fuse and not _any_hol(q):
        for s0 in range(len(jp.steps) - 1):
            if _star_valid(q, jp, s0):
                star_s0 = s0
                break
    # a leaf that was only counted is read filtered now, unless it is the last join's probe side (its filter then runs in that join's pipeline)
    var last_leaf = jp.steps[len(jp.steps) - 1].leaf
    for t in range(n):
        if leaves[t].deferred and not (t == last_leaf and not jp.steps[len(jp.steps) - 1].build_new) and not (t == jp.start and star_s0 == 0):
            _read_filtered[P](catalog, q, t, leaves, planner, opts, trust, res)
    # the steps
    var done = List[Bool](length=len(q.resid), fill=False)
    var joined = 1 << jp.start
    var cur_table = leaves[jp.start].table
    var cur_g = leaves[jp.start].gcols.copy()
    var cur_pos = leaves[jp.start].cols.copy()
    var si = -1
    while si + 1 < len(jp.steps) - 1:
        si += 1
        if si == star_s0:
            # the probe table: the rows joined so far, or (the star starts at the start table) that table itself
            var sp_table = cur_table
            var sp_gt = cur_g.copy()
            var sp_conj = List[Int]()
            var sp_conj_text = List[String]()
            var sp_src = SqlSource("", Table.empty().share())
            var sp_streamed = False
            if si == 0:
                sp_table = leaves[jp.start].table
                sp_gt = leaves[jp.start].gtable.copy()
                if leaves[jp.start].deferred:
                    sp_conj = q.leaf_conj[jp.start].copy()
                    sp_conj_text = q.leaf_conj_text[jp.start].copy()
                if leaves[jp.start].streamed:
                    sp_src = catalog.sources[q.sources[jp.start]].copy()
                    sp_streamed = True
            var star_out = SqlResult()
            if _run_star[P](q, leaves, js, jp, si, joined, sp_table, sp_gt, sp_conj, sp_src, sp_streamed, done, planner, opts, trust, lie, res, star_out):
                for nt in star_out.notes:
                    res.notes.append(nt)
                for a in star_out.actuals:
                    res.actuals.append(a)
                res.table = star_out.take_table()
                res.explain = _explain_text(lp, res.notes)
                res.ms = Float64(perf_counter_ns() - t0) / 1e6
                return res^
            star_s0 = -1
            if si == 0 and leaves[jp.start].deferred:
                _read_filtered[P](catalog, q, jp.start, leaves, planner, opts, trust, res)
                cur_table = leaves[jp.start].table
                cur_g = leaves[jp.start].gcols.copy()
                cur_pos = leaves[jp.start].cols.copy()
        var t_step = perf_counter_ns()
        ref step = jp.steps[si]
        var lf = step.leaf
        var after = joined | (1 << lf)
        var keep = needed_after(q, after, done)
        var lkey = List[Int]()
        var rkey = List[Int]()
        for k in range(len(q.edge_a)):
            var ga = q.edge_a[k]
            var gb = q.edge_b[k]
            var ta = table_of_column(q, ga)
            var tb = table_of_column(q, gb)
            if ta == lf and (joined >> tb) & 1 != 0:
                lkey.append(cur_pos[_position(cur_g, gb)])
                rkey.append(leaves[lf].cols[_position(leaves[lf].gcols, ga)])
            elif tb == lf and (joined >> ta) & 1 != 0:
                lkey.append(cur_pos[_position(cur_g, ga)])
                rkey.append(leaves[lf].cols[_position(leaves[lf].gcols, gb)])
        var stepped = _join_step(q, cur_table, cur_g, cur_pos, leaves[lf], keep, lkey^, rkey^, step.build_new, step.est_rows, _label(q, lf), "join step " + String(si + 1),
                                 threads, budget_bytes, res, t_step)
        cur_g = stepped[1].copy()
        cur_table = stepped[0]
        cur_pos = List[Int]()
        for i in range(len(cur_g)):
            cur_pos.append(i)
        joined = after
        # the conditions across tables whose tables are all joined now
        var conj = List[Int]()
        var ctext = List[String]()
        for k in range(len(q.resid)):
            if not done[k] and (q.resid_mask[k] & ~joined) == 0:
                conj.append(q.resid[k])
                ctext.append(q.resid_text[k])
                done[k] = True
        if len(conj) > 0:
            var t_f = perf_counter_ns()
            var fq = filter_query(q, conj, cur_g, cur_table)
            var tmp = Catalog()
            tmp.add_table("rows", cur_table)
            var fr = _run_sql_once(tmp, fq, planner, opts, trust, String())
            var before = cur_table.rows()
            cur_table = fr.take_table().share()
            res.notes.append("conditions across tables checked after join step " + String(si + 1) + ": " + " AND ".join(ctext) + ": " + commas(before) + " rows to " + commas(cur_table.rows()))
            res.actuals.append("filter across tables (" + " AND ".join(ctext) + "): " + commas(before) + " -> " + commas(cur_table.rows()) + " rows, " + _ms(Float64(perf_counter_ns() - t_f) / 1e6) + " ms")
    # the last join, with the rest of the query
    ref last = jp.steps[len(jp.steps) - 1]
    var lf = last.leaf
    var key_l = List[Int]()
    var key_r = List[Int]()
    for k in range(len(q.edge_a)):
        var ga = q.edge_a[k]
        var gb = q.edge_b[k]
        var ta = table_of_column(q, ga)
        var tb = table_of_column(q, gb)
        if ta == lf and (joined >> tb) & 1 != 0:
            key_l.append(gb)
            key_r.append(ga)
        elif tb == lf and (joined >> ta) & 1 != 0:
            key_l.append(ga)
            key_r.append(gb)
    var rest = List[Int]()
    for k in range(len(q.resid)):
        if not done[k]:
            rest.append(q.resid[k])
    var rconj = List[Int]()
    var rtext = List[String]()
    if leaves[lf].deferred:
        rconj = q.leaf_conj[lf].copy()
        rtext = q.leaf_conj_text[lf].copy()
    var fq = final_query(q, cur_g, leaves[lf].gtable, cur_table, leaves[lf].table, key_l, key_r, rest, rconj, rtext)
    var cat3 = Catalog()
    cat3.add_table("rows joined so far", cur_table)
    if leaves[lf].deferred:
        var rsrc = catalog.sources[q.sources[lf]].copy()                       # the real source: the last join reads (or streams) it itself
        rsrc.name = _label(q, lf)
        cat3.sources.append(rsrc^)
    else:
        cat3.add_table(_label(q, lf), leaves[lf].table)
    var o2 = opts.copy()
    o2.build_side = BUILD_RIGHT if last.build_new else BUILD_LEFT
    var lp2 = build_plan(fq, "rows joined so far")
    var inner = _run_join_query(cat3, fq, lp2, planner, o2, trust, lie, perf_counter_ns())
    res.notes.append("last join: the rows joined so far (" + commas(cur_table.rows()) + ") with " + _label(q, lf) + " (" + commas(leaves[lf].table.rows()) + " rows), estimated " + commas(last.est_rows) + " rows out")
    for nt in inner.notes:
        res.notes.append(nt)
    for a in inner.actuals:
        res.actuals.append(a)
    res.table = inner.take_table()
    res.explain = _explain_text(lp, res.notes)
    res.ms = Float64(perf_counter_ns() - t0) / 1e6
    return res^


def _sums_integers(q: BoundQuery) -> Bool:
    """Whether the query has a SUM or AVG of INT or DECIMAL values: the aggregates that can hold 128 bits (F-192)."""
    for a in q.aggs:
        if (a.func == F_SUM or a.func == F_AVG) and not a.hol and a.arg >= 0:
            var k = q.rexprs[a.arg].kind
            if k == INT.value or k == DECIMAL.value:
                return True
    return False


def _run_sql_once[P: Planner](mut catalog: Catalog, q0: BoundQuery, planner: P, opts: PlannerOptions, trust: Bool, lie: String,
                              fj: Optional[ArcPointer[FusedJoin]] = None) raises -> SqlResult:
    var t0 = perf_counter_ns()
    var lp = build_plan(q0, catalog.sources[q0.source].name)         # the plan; the executor lowers its query from the nodes
    if q0.joined and len(q0.sources) > 2:
        return _run_multiway(catalog, q0, lp, planner, opts, trust, lie, t0)
    var q = executable(lp, q0)
    if q.joined:
        return _run_join_query(catalog, q, lp, planner, opts, trust, lie, t0)
    prepare_ranges(catalog, q)
    ref src = catalog.sources[q.source]
    var scan = src.scan.copy()
    if scan:
        for e in q.rexprs:
            if e.op == B_COL and e.kind == TEXT.value:
                scan.value().ensure_words(Int(e.value))
    var extra = src.overlay.words.copy()
    var dicts = DictAccess(src.table, scan, extra)
    if trust or src.kind != SRC_SCAN:                        # (a footer's range that was found wrong is not used the second time)
        dicts.set_ranges(src.range_cols, src.range_lo, src.range_hi, src.range_has, src.rows() + (len(src.overlay.delta) if src.kind != SRC_SCAN else 0), src.kind != SRC_SCAN)
    var st = make_stats(src, q, dicts, trust)
    var plan = planner.plan(q, st, opts)
    plan_sinks(lp, q, st, opts, plan)
    var res = SqlResult()
    res.notes = plan.notes.copy()
    if lie.byte_length() > 0:
        res.notes.append("statistics: the footer of `" + src.name + "` cannot be believed (" + lie + "); ran again without them: hashed groups, every column nullable")
    var inp = _inputs(src, q, trust)
    var partner = List[Bool]()
    if fj:                                                           # the fused join: the program's table columns are the probe table's, then placeholders
        for v in fj.value()[].vprobe:
            partner.append(not v)
        _fuse_inputs(inp, fj.value()[])
        if fj.value()[].stream:
            inp.masked = len(fj.value()[].stream.value()[].src.overlay.mask) > 0
    var sp = SqlProgram(Program(0))
    var stage = RowStage(SqlProgram(Program(0)), List[_Item]())
    if q.rows:
        stage = prepare_rows(src, q, plan, dicts, inp, res.notes, True, partner)
    else:
        sp = _lower(src, q, plan, dicts, inp, PHASE_PLAIN if (plan.group == G_HASHED and plan.holistic) else PHASE_ALL)
        _kernel_notes(src, q, plan, dicts, inp, sp, res.notes)
    var gpu = _pick_device(src, q, plan, sp, opts, plan.group == G_HASHED and plan.holistic, Bool(fj))        # F-191
    if q.explain != EXPLAIN_NONE:
        gpu.ran = gpu.use
        res.notes.append(gpu.note())
        res.explain = _explain_text(lp, res.notes)
        var t = Table.empty()
        var c = t.add_column("plan", TEXT)
        for line in res.explain.split("\n"):
            if line.byte_length() > 0:
                t.append_text(c, String(line))
        res.table = t^
        res.ms = Float64(perf_counter_ns() - t0) / 1e6
        return res^
    var notes = res.notes.copy()
    if q.rows:
        res.table = execute_rows(src, q, plan, opts, dicts, notes, inp, stage^, trust, fj, True)
    else:
        try:
            res.table = _execute(src, q, plan, opts, dicts, notes, inp, sp^, gpu, trust, fj)
        except e:
            if String(e) != OVERFLOW_MESSAGE or not _sums_integers(q):
                raise e^
            # F-192: a total passed int64 (or an expression did: then this raises the same message again). DuckDB sums into 128 bits: run again with
            # every integer SUM / AVG split into the sums of its upper and lower halves, which cannot overflow, and join the halves in 128 bits.
            notes = res.notes.copy()
            notes.append("sums: a total passed int64 (about 9.2e18); ran again with 128-bit totals (each value's upper and lower 32 bits summed apart)")
            var sp_wide: SqlProgram
            try:
                sp_wide = _lower(src, q, plan, dicts, inp, PHASE_PLAIN if (plan.group == G_HASHED and plan.holistic) else PHASE_ALL, True)
            except lowering:                                                  # (the split sums need more registers than the query has: the error is the total's)
                raise Error(OVERFLOW_MESSAGE)
            res.table = _execute(src, q, plan, opts, dicts, notes, inp, sp_wide^, gpu, trust, fj)
    notes.append(gpu.note())
    res.device = GPU if gpu.ran else CPU
    res.fallback_reason = gpu.reason()
    res.notes = notes^
    res.explain = _explain_text(lp, res.notes)
    res.ms = Float64(perf_counter_ns() - t0) / 1e6
    return res^


def _unresolved_bins(q: BoundQuery) -> List[Int]:
    """The quantile_bin nodes whose edges are not known yet (`b` is -1 and there are no edge literals)."""
    var out = List[Int]()
    for i in range(len(q.rexprs)):
        if q.rexprs[i].op == B_QBIN and q.rexprs[i].b == -1 and len(q.rexprs[i].items) == 0:
            out.append(i)
    return out^


def _reaches(q: BoundQuery, root: Int, target: Int) -> Bool:
    """Whether row-level expression `target` is `root` or inside it."""
    if root < 0:
        return False
    if root == target:
        return True
    ref x = q.rexprs[root]
    if _reaches(q, x.a, target) or _reaches(q, x.b, target):
        return True
    for it in x.items:
        if _reaches(q, it, target):
            return True
    return False


def _resolve_bins[P: Planner](mut catalog: Catalog, q: BoundQuery, planner: P, opts: PlannerOptions) raises -> BoundQuery:
    """quantile_bin(x, n) needs the n - 1 edges of equal-count bins of x: edge k is the value at sorted place ceil(k * N / n) among the N non-NULL values
    of x that the query's WHERE lets through. They are found by a query of their own (n - 1 exact order statistics of x, one pass over the source, the
    holistic machinery) before the real one is planned, and written into the node as literals of x's own type."""
    if q.joined or q.rows:
        raise Error("quantile_bin() is supported in an aggregate query over one table (GROUP BY quantile_bin(x, n)), not in a join or a row query yet")
    var todo = _unresolved_bins(q)
    for i in todo:                                       # only as (part of) a GROUP BY key: not in WHERE, an aggregate's argument, or another quantile_bin
        var forbidden = List[Int]()
        forbidden.append(q.where)
        for a in q.aggs:
            forbidden.append(a.arg)
        for j in todo:
            forbidden.append(q.rexprs[j].a)
        var keyed = False
        for k in q.keys:
            keyed = keyed or _reaches(q, k.expr, i)
        var bad = not keyed
        for r in forbidden:
            bad = bad or _reaches(q, r, i)
        if bad:
            raise Error("quantile_bin() is supported only in GROUP BY (a key of an aggregate query over one table), not in WHERE, inside an aggregate or another quantile_bin")
    var out = copy_query(q)
    for i in todo:
        var node = q.rexprs[i].copy()
        var n = Int(node.value)
        var x = q.rexprs[node.a].copy()
        var aux = copy_query(q)
        for j in todo:
            aux.rexprs[j].b = -2                                      # (the query's bins are being resolved: the aux query neither resolves nor checks them again)
        aux.keys = List[KeySpec]()
        aux.aggs = List[AggSpec]()
        aux.pexprs = List[BExpr]()
        aux.outputs = List[Int]()
        aux.out_names = List[String]()
        aux.out_text = List[String]()
        aux.order = List[SortKey]()
        aux.order_text = List[String]()
        aux.having = -1
        aux.limit = -1
        aux.offset = 0
        aux.explain = EXPLAIN_NONE
        for k in range(1, n):
            var name = "edge" + String(k)
            aux.aggs.append(AggSpec(F_QEDGE, node.a, Float64(k), x.kind, x.scale, name, "qedge:" + String(k), True, n))
            aux.pexprs.append(BExpr(B_AGG, x.kind, x.scale, -1, -1, Int64(k - 1), name, node.line, node.col))
            aux.outputs.append(k - 1)
            aux.out_names.append(name)
            aux.out_text.append(name)
        var r = run_sql(catalog, aux, planner, opts)
        for k in range(n - 1):                                        # (no value at all: every x is NULL or filtered out, so no row reaches the bin)
            var edge = Int64(0) if r.table.is_null(k, 0) else r.table.get(k, 0)
            out.rexprs.append(BExpr(B_LIT, x.kind, x.scale, -1, -1, edge, "", node.line, node.col))
            out.rexprs[i].items.append(len(out.rexprs) - 1)
    return out^


def run_sql[P: Planner](mut catalog: Catalog, q: BoundQuery, planner: P, opts: PlannerOptions) raises -> SqlResult:
    """Runs the query (after finding the edges of any quantile_bin in it)."""
    if len(_unresolved_bins(q)) > 0:
        var resolved = _resolve_bins(catalog, q, planner, opts)
        return _run_sql_resolved(catalog, resolved, planner, opts)
    return _run_sql_resolved(catalog, q, planner, opts)


def _run_sql_resolved[P: Planner](mut catalog: Catalog, q: BoundQuery, planner: P, opts: PlannerOptions) raises -> SqlResult:
    """Runs the query. A Parquet footer's min / max and null counts are a hint that lets the planner use dense groups and skip NULL
    checks; they are verified as the row groups are read (a null count that is wrong, a row outside a stated range), and when one
    is wrong the query runs again without the statistics, which the notes say."""
    opts.job.stage(STAGE_PLAN)
    if q.explain == EXPLAIN_ANALYZE:
        return _explain_analyze(catalog, q, planner, opts)
    var r: SqlResult
    try:
        r = _run_sql_once(catalog, q, planner, opts, True, String())
    except e:
        var msg = String(e)
        if not msg.startswith(LIE):
            raise e^
        r = _run_sql_once(catalog, q, planner, opts, False, String(msg[byte=LIE.byte_length():]))
    _device_note(r, opts)
    return r^


def _device_note(mut r: SqlResult, opts: PlannerOptions):
    """F-191: a statement the single-table path did not run (a join, a multi-way join) still says which device ran it."""
    for n in r.notes:
        if n.startswith("device: "):                      # (a join's aggregate stage reports its own: the device it ran on is the result's)
            if n.startswith("device: GPU"):
                r.device = GPU
            return
    if opts.device == CPU:
        r.notes.append("device: CPU (device CPU was asked for)")
        return
    var why = String("joins run on the CPU (the GPU runs single-table aggregates)")
    r.notes.append("device: CPU (" + ("GPU asked for; " if opts.device == GPU else "AUTO: ") + why + ")")
    r.fallback_reason = why^


def _explain_analyze[P: Planner](mut catalog: Catalog, q0: BoundQuery, planner: P, opts: PlannerOptions) raises -> SqlResult:
    """EXPLAIN ANALYZE: the query runs, and the plan comes back with what each operator really did (rows, milliseconds) after the planner's decisions."""
    var q = copy_query(q0)
    q.explain = EXPLAIN_NONE
    var r: SqlResult
    try:
        r = _run_sql_once(catalog, q, planner, opts, True, String())
    except e:
        var msg = String(e)
        if not msg.startswith(LIE):
            raise e^
        r = _run_sql_once(catalog, q, planner, opts, False, String(msg[byte=LIE.byte_length():]))
    var text = r.explain.rstrip("\n") + "\nActual\n"
    for a in r.actuals:
        text += "  " + a + "\n"
    for nt in r.notes:
        if nt.startswith("time:") or nt.startswith("join time:") or nt.startswith("ran:"):
            text += "  " + nt + "\n"
    text += "  result: " + commas(r.table.rows()) + " rows in " + _ms(r.ms) + " ms\n"
    var out = SqlResult()
    out.notes = r.notes.copy()
    out.actuals = r.actuals.copy()
    out.explain = text
    out.ms = r.ms
    var t = Table.empty()
    var c = t.add_column("plan", TEXT)
    for line in text.split("\n"):
        if line.byte_length() > 0:
            t.append_text(c, String(line))
    out.table = t^
    return out^


def run_plain[P: Planner](mut catalog: Catalog, var st: SqlStatement, text: String, planner: P, options: PlannerOptions) raises -> SqlResult:
    """Runs one statement that reads only named sources (F-159's sqlquery.mojo takes the rest apart first): window functions, or bind and run."""
    if len(st.wins) > 0:
        return _run_windows(catalog, st^, text, planner, options)
    var q = bind_statement(catalog, st^, text)
    return run_sql(catalog, q, planner, options)


def _run_windows[P: Planner](mut catalog: Catalog, var st: SqlStatement, text: String, planner: P, opts: PlannerOptions) raises -> SqlResult:
    """A query with window functions (F-168, winsql.mojo): the query without them runs as usual, the window columns are added to its rows, and the
    select list, DISTINCT, ORDER BY and LIMIT run over that result. Window functions run on the CPU, whatever the device asked for."""
    var t0 = perf_counter_ns()
    var explain = st.explain
    if explain == EXPLAIN_ANALYZE:
        raise sql_error(st.items[0].line, st.items[0].col, "EXPLAIN ANALYZE of a query with window functions is not supported yet (EXPLAIN is)")
    var wp = plan_windows(st^, catalog)
    var inner = SqlStatement(String())
    swap(inner, wp.inner)
    var iq = bind_statement(catalog, inner^, text)
    iq.explain = explain
    var r = run_sql(catalog, iq, planner, opts)
    if explain != EXPLAIN_NONE:                                          # the window step first, then the inner query's plan
        var lines = List[String]()
        lines.append("Window functions (after the query below, on the CPU)")
        for g in wp.groups:
            var sorts = String("  sort by ")
            for p in g.part:
                sorts += "partition column " + String(p) + ", "
            sorts += String(len(g.order)) + " order key" + ("" if len(g.order) == 1 else "s")
            lines.append(sorts)
        for dsc in wp.descriptions:
            lines.append("  " + dsc)
        lines.append("Query")
        var shown = Table.empty()
        var c = shown.add_column("plan", TEXT)
        for l in lines:
            shown.append_text(c, l)
        for k in range(r.table.rows()):
            shown.append_text(c, "  " + r.table.text_of(0, Int(r.table.get(0, k))))
        r.table = shown^
        r.explain = ""
        for k in range(r.table.rows()):
            r.explain += r.table.text_of(0, Int(r.table.get(0, k))) + "\n"
        return r^
    var rows = Table.empty()
    swap(rows, r.table)
    var notes = r.notes.copy()
    var t_w = perf_counter_ns()
    finish_windows(wp, rows, opts.threads, opts.job, notes)
    notes.append("window: " + _ms(Float64(perf_counter_ns() - t_w) / 1e6) + " ms")
    var outer = SqlStatement(String())
    swap(outer, wp.outer)
    var wide_cols = point_wide_columns(rows, outer)                  # F-192: totals past int64 pass through the outer query as ranks
    if len(wide_cols.out) > 0:
        notes.append("totals past int64 in " + String(len(wide_cols.out)) + " output column(s) go through the outer query as ranks")
    var cat2 = Catalog()
    cat2.add_table(WIN_TABLE, rows^.share())
    var oq = bind_statement(cat2, outer^, text)
    var out = run_sql(cat2, oq, planner, opts)
    fix_wide_columns(out.table, wide_cols)
    for nt in out.notes:
        notes.append(nt)
    out.notes = notes^
    out.fallback_reason = "window functions run on the CPU"
    out.ms = Float64(perf_counter_ns() - t0) / 1e6
    return out^
