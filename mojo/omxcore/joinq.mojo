"""The builder's join (F-204): `Query.join` no longer writes the joined rows. It keeps both tables, the keys and the schema of the joined columns, and
`compute` runs the plan the way SQL runs a two-table join: the same key normalization, the same hash table (`hj_build`), and, when the build side's
keys are unique, are expanded (repeats up to EXPAND_MAX_REP), or the join is a semi / anti one, the same fused front stage (`jfront.mojo`) that probes
block by block and gathers only the build-side columns the program reads. No joined row exists, and the build side is never gathered beside the probe
side, so a fact table against a dimension no longer meets the join memory budget.

The program is the builder's own (`_lower` over the schema of the joined columns, so every rule of the builder holds after a join): its column numbers
are then pointed at the probe table's columns and at the placeholders the front stage fills (`_fuse_pos`), as `sqlexec._fuse_inputs` does for SQL.
When the join cannot be fused (a holistic aggregate needs every joined row, the keys are hashed tuples, a key repeats more than EXPAND_MAX_REP times, or
the device is asked for) the rows are written as before, but only the columns the plan reads.
"""
from std.time import perf_counter_ns
from std.memory import ArcPointer
from .table import Table, SharedTable, TEXT
from .bind import Catalog, SqlSource, SRC_TABLE
from .query import (Query, QueryResult, LazyJoin, Program, GPU, A_DISTINCT, _lower, _key_from_reg0, _make_result, overflow_message)
from .job import JobCtl, STAGE_AGGREGATE
from .joincore import JoinKeys, JoinSide, normalize_keys, join_pairs, select_rows, gather_join, JOIN_BUDGET_BYTES, commas
from .join import virtual_schema
from .jfront import EXPAND_MAX_REP, FusedJoin, front_spec, jf_front
from .kargs import KernelRun
from .kernels.pipeline_cpu import IP, BP, E_OVERFLOW, _no_keys
from .kernels.hashjoin_cpu import hj_build, JoinMap, PM_SEMI, PM_ANTI
from .sqlquery import sql


def how_kind(how: String) -> Int:
    """0 inner, 1 left, 2 semi, 3 anti (the `JK_*` numbers of SQL)."""
    if how == "left":
        return 1
    if how == "semi":
        return 2
    if how == "anti":
        return 3
    return 0


def realize_source(var src: SqlSource) raises -> SharedTable:
    """The rows a source reads as a table: a plain table is itself, a Frame is its edited rows (a row query of the engine over the source, so the shared
    rows, the changed and appended rows, merges and formula columns come out as the SQL engine reads them)."""
    if src.kind == SRC_TABLE and not src.framed:
        return src.table
    var name = src.name
    var cat = Catalog()
    cat._put(src^)
    var r = sql(cat, "SELECT * FROM " + name)
    return r.take_table().share()


def _carried(lt: SharedTable, rt: SharedTable, kind: Int, used: List[Bool]) -> Tuple[List[Int], List[Int]]:
    """The columns of each table the joined columns `used` marks (the right table's are not carried by a semi / anti join)."""
    var lc = List[Int]()
    var rc = List[Int]()
    var at = 0
    for c in range(lt.columns()):
        if used[at]:
            lc.append(c)
        at += 1
    if kind <= 1:
        for c in range(rt.columns()):
            if used[at]:
                rc.append(c)
            at += 1
    return (lc^, rc^)


def joined_schema(lt: SharedTable, rt: SharedTable, kind: Int) raises -> Table:
    """The joined columns (no rows): this table's columns, then the other's (not for a semi / anti join), a repeated name suffixed `_1`."""
    var all_l = List[Int]()
    for c in range(lt.columns()):
        all_l.append(c)
    var all_r = List[Int]()
    if kind <= 1:
        for c in range(rt.columns()):
            all_r.append(c)
    return virtual_schema(JoinSide(lt, all_l^, List[Int](), True), JoinSide(rt, all_r^, List[Int](), True), kind == 1)


def materialize_join(jn: LazyJoin, used: List[Bool], threads: Int, mut note: String) raises -> Table:
    """The joined rows with the columns `used` marks, under the schema's names (every column when all are marked): the hash join as before, one table written."""
    var kind = how_kind(jn.how)
    var cols = _carried(jn.left, jn.right, kind, used)
    var left = JoinSide(jn.left, cols[0].copy(), jn.lkeys.copy(), True)
    var right = JoinSide(jn.right, cols[1].copy(), jn.rkeys.copy(), True)
    var names = List[String]()
    for v in range(len(used)):
        if used[v]:
            names.append(jn.names[v])
    var left_builds = kind == 0 and jn.left.rows() < jn.right.rows()
    var keys = JoinKeys()
    if left_builds:
        normalize_keys(jn.left, jn.right, jn.lkeys, jn.rkeys, keys, threads)
    else:
        normalize_keys(jn.right, jn.left, jn.rkeys, jn.lkeys, keys, threads)
    var nb = jn.left.rows() if left_builds else jn.right.rows()
    var np = jn.right.rows() if left_builds else jn.left.rows()
    var m = hj_build(keys.bk_addr, keys.bo_addr, nb, threads, kind >= 2)
    var out: Table
    var n_out = 0
    if kind >= 2:
        var pick = select_rows(m, keys, np, PM_SEMI if kind == 2 else PM_ANTI, threads)
        n_out = pick.n
        out = gather_join(left, right, pick, False, threads, False, names)
    else:
        var pairs = join_pairs(m, keys, np, JOIN_BUDGET_BYTES // (8 * max(len(names), 1)), threads, kind == 1)
        n_out = pairs.n
        out = gather_join(left, right, pairs, left_builds, threads, kind == 1, names)
    note = commas(n_out) + (" rows" if kind >= 2 else " matching pairs" if kind == 0 else " pairs (unmatched rows included)") + " gathered into one table of " + String(len(names)) + " columns"
    return out^


def _mark(jn: LazyJoin, e_names: List[String], mut used: List[Bool]):
    for name in e_names:
        for v in range(len(jn.names)):
            if jn.names[v] == name:
                used[v] = True
                break


def _fuse_pos(mut p: Program, fj: FusedJoin):
    """Points the program's columns (numbered over the joined schema) at the probe table's columns, and at the placeholders after them that the front stage
    fills with the partner's values: a data column `c`, a null-mask entry `c` and a validity column `-1 - c` all follow the column."""
    var np = fj.n_probe_cols
    var pos = List[Int]()
    for v in range(fj.virtual_columns()):
        pos.append(fj.vreal[v] if fj.vprobe[v] else np + fj.vpartner[v])
    for i in range(len(p.cols)):
        var c = p.cols[i]
        if c >= 0:
            p.cols[i] = pos[c]
        elif c > -1_000_000:                                          # (the program's own markers, -1_000_000 and below, name no column)
            p.cols[i] = -1 - pos[-1 - c]


def compute_joined(q: Query, device: Int, job: JobCtl) raises -> QueryResult:
    """`Query.compute` of a query that starts with a join."""
    ref jn = q.jn.value()
    var t0 = perf_counter_ns()
    var kind = how_kind(jn.how)
    var semi_anti = kind >= 2
    var vt = q.base                                                  # the joined schema, no rows
    var prog = _lower(vt, q.nodes, False, False)                      # a bad plan fails here, before any key is built
    var used = List[Bool](length=len(jn.names), fill=False)
    for node in q.nodes:
        for e in node.exprs:
            _mark(jn, e.columns(), used)
        for a in node.aggs:
            _mark(jn, a.arg.columns(), used)
    var left_builds = kind == 0 and jn.left.rows() < jn.right.rows()
    var why = String()
    var keys = JoinKeys()
    var m = JoinMap()
    var fusable = not prog.mat and device != GPU
    if prog.mat:
        why = " (a holistic aggregate needs every joined row)"
    elif device == GPU:
        why = " (the GPU was asked for: a joined query has no DeviceTable)"
    var expands = False
    if fusable:
        if left_builds:
            normalize_keys(jn.left, jn.right, jn.lkeys, jn.rkeys, keys, q._threads)
        else:
            normalize_keys(jn.right, jn.left, jn.rkeys, jn.lkeys, keys, q._threads)
        var nb = jn.left.rows() if left_builds else jn.right.rows()
        var bits = kind == 0                                         # F-207: no build-side column is read: unique keys are one bit each
        for v in range(0 if left_builds else jn.left.columns(), jn.left.columns() if left_builds else len(used)):
            if used[v]:
                bits = False
        m = hj_build(keys.bk_addr, keys.bo_addr, nb, q._threads, semi_anti, bits)
        expands = kind <= 1 and not m.unique and m.maxrep <= EXPAND_MAX_REP
        if keys.verify:
            why = " (the keys are hashed tuples, verified per pair)"
        elif not (m.unique or semi_anti or expands):
            why = " (the build side's keys repeat, one up to " + commas(m.maxrep) + " times: past " + String(EXPAND_MAX_REP) + " the front stage would need a round for each)"
        fusable = why.byte_length() == 0
    var np = (jn.right.rows() if left_builds else jn.left.rows())
    var side_note = "join: " + jn.how + " on " + jn.key_text + ": " + commas(jn.left.rows()) + " rows and " + commas(jn.right.rows()) + " rows"
    if not fusable:
        if device == GPU:
            raise Error("device GPU: a Query that starts with a join runs on the CPU (the join is fused into the CPU pipeline; there is no DeviceTable of the joined rows)")
        var gathered = String()
        var jt = materialize_join(jn, used, q._threads, gathered)
        var inner = Query(jt^.share())
        inner.nodes = q.nodes.copy()
        inner._threads = q._threads
        inner._morsels = q._morsels
        inner._block_rows = q._block_rows
        inner._window_rows = q._window_rows
        var r = inner.compute(device, job)
        r.notes.insert(0, "join path: materialized: " + gathered + why)
        r.notes.insert(0, side_note)
        r.milliseconds = Float64(perf_counter_ns() - t0) / 1e6
        return r^
    # a joined column the plan does not read is never looked at: it is declared a column of the probe table (its number is not used), so the front stage
    # gathers (and keeps scratch for) only the build-side columns the program reads
    var vprobe = List[Bool]()
    var vreal = List[Int]()
    var v = 0
    for c in range(jn.left.columns()):
        vprobe.append(not left_builds or not used[v])
        vreal.append(c if used[v] else 0)
        v += 1
    if not semi_anti:
        for c in range(jn.right.columns()):
            vprobe.append(left_builds or not used[v])
            vreal.append(c if used[v] else 0)
            v += 1
    var pt = jn.right if left_builds else jn.left
    var bt = jn.left if left_builds else jn.right
    var fmode = 0 if kind == 0 else 1 if kind == 1 else PM_SEMI if kind == 2 else PM_ANTI
    var fj = ArcPointer(FusedJoin(pt, bt, vprobe^, vreal^, ArcPointer(m^), keys^, fmode))
    _fuse_pos(prog, fj[])
    job.stage(STAGE_AGGREGATE)
    job.add_total(np)
    var run = KernelRun(prog, prog.set_bytes, prog.set_meta, pt, 0, prog.groups)
    var fs = front_spec(run, fj[], prog)

    def front(k: Int, i0: Int, n: Int, sel: IP, ws: BP) {imm} -> Int:
        return jf_front(fs, i0, n, sel, ws)

    var rc: Int
    if prog.keyed:
        rc = run.fused(_key_from_reg0, front, fs.ws_bytes, True, prog.groups, 0, np, q._threads)
    else:
        rc = run.fused(_no_keys, front, fs.ws_bytes, False, 1, 0, np, q._threads)
    if rc == E_OVERFLOW:
        raise Error(overflow_message(prog.conv_text()))
    if rc != 0:
        raise Error("the CPU kernel failed with code " + String(rc))
    job.advance(np)
    var r = _make_result(prog, run.out_vals, run.out_cnt)
    r.notes.append(side_note)
    r.notes.append("join build side: " + ("the left table" if left_builds else "the right table") + " (" + commas(jn.left.rows() if left_builds else jn.right.rows()) + " rows): " + fj[].map[].kind())
    if semi_anti:
        r.notes.append("join path: fused: " + commas(np) + " probe rows are " + ("kept when their key is present in the build side" if kind == 2 else "dropped when their key is present in the build side") +
                       ", in blocks of 1,024 (" + jn.how + " join: no build-side column is gathered, no joined row is written)")
    else:
        r.notes.append("join path: fused: " + commas(np) + " probe rows look their partner up in blocks of 1,024 and " + String(len(fj[].partner)) + " build-side " +
                       ("column is" if len(fj[].partner) == 1 else "columns are") + " gathered into per-worker scratch (no joined row is written)" +
                       (" (a row without a partner stays: its build-side columns are NULL)" if kind == 1 else "") +
                       (" (the build side's keys repeat, up to " + commas(fj[].map[].maxrep) + " rows of a key: each probe row is expanded to its partners in rounds inside the front stage, no pairs are written)" if expands else ""))
    r.milliseconds = Float64(perf_counter_ns() - t0) / 1e6
    return r^
