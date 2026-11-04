"""The stage after aggregation, for totals past int64 (F-192).

`post.mojo` runs HAVING, the select list and ORDER BY through the kernel, whose registers are int64. A total past int64 (a SUM of
DECIMAL values, a scalar subquery's answer, a total times a constant) does not fit one, so when the stage meets such a number it asks this
module to take those expressions out of the kernel's hands first:

  - every expression made of + - * negation, a rescale and comparisons over totals, keys and constants (`B_ADD` ... `B_CMP`) that reads a
    total or a wide constant (`B_WLIT`) is evaluated here, per group, in 128 bits (`Int128`; a result past DuckDB's 38 digits is an error);
  - a BOOL result (HAVING a comparison) becomes an ordinary BOOL column of the groups' table; a number that fits int64 becomes an ordinary
    column of its kind; a number that does not becomes a column of ranks, the place of the value among the distinct values (NULL
    stays NULL), which the kernel carries through HAVING, ORDER BY and LIMIT as it would any integer (equal values have equal ranks and
    the ranks' order is the values' order) and `fix_wide` turns back into the 128-bit values in the finished table;
  - a number meeting a FLOAT (`/`, AVG) is converted to a double here, the way a DECIMAL(38) is cast.

What the kernel cannot be given (a wide number inside CASE, COALESCE, a CAST ...) is refused with the int64 message, as before.
"""
from .table import Table, INT, DECIMAL, BOOL, FLOAT
from .bind import (BoundQuery, BExpr, B_COL, B_LIT, B_NULL, B_WLIT, B_ADD, B_SUB, B_MUL, B_NEG, B_RESCALE, B_CMP, B_ISNULL, B_DIV, B_I2F, B_AND, B_OR, B_NOT,
                   C_LT, C_LE, C_GT, C_GE, C_EQ, C_NE, decimal_to_float)
from .sqleval import ValCol, float_bits
from .query import OVERFLOW_MESSAGE, WIDE_LITERAL_MESSAGE
from .wide import pow10_wide, wide_make, wide_lo, wide_hi, fits_int64, wide_mul, wide_add, wide_sub, wide_to_float, within_decimal


def _is_number(kind: Int) -> Bool:
    return kind == INT.value or kind == DECIMAL.value


struct WCol(Movable):
    """The values of one expression for every group, in 128 bits."""
    var v: List[Int128]
    var nul: List[UInt8]             # empty: none is NULL

    def __init__(out self, n: Int):
        self.v = List[Int128](length=n, fill=Int128(0))
        self.nul = List[UInt8]()

    def is_null(self, i: Int) -> Bool:
        return len(self.nul) > 0 and self.nul[i] != 0

    def set_null(mut self, i: Int):
        if len(self.nul) == 0:
            self.nul = List[UInt8](length=len(self.v), fill=0)
        self.nul[i] = 1


struct WideRewrite(Movable):
    """What the rewrite added to the stage: columns for the groups' table (after the keys and aggregates), and for each rank column the
    expression node it stands for and the distinct values its ranks index."""
    var hidden: List[ValCol]
    var base: Int                    # the groups' table position of the first hidden column
    var ptr_node: List[Int]
    var ptr_vals: List[List[Int128]]

    def __init__(out self, base: Int):
        self.hidden = List[ValCol]()
        self.base = base
        self.ptr_node = List[Int]()
        self.ptr_vals = List[List[Int128]]()


struct _Scan(Movable):
    """Per node of the original pool: can the evaluator here take it (`ok`), and does it read a total or a wide constant (`leaf`)."""
    var ok: List[Int]                # -1 not looked at, 0 no, 1 yes
    var leaf: List[Bool]
    var visited: List[Bool]

    def __init__(out self, n: Int):
        self.ok = List[Int](length=n, fill=-1)
        self.leaf = List[Bool](length=n, fill=False)
        self.visited = List[Bool](length=n, fill=False)


def _scan(q0: List[BExpr], i: Int, nk: Int, mut s: _Scan):
    if s.ok[i] >= 0:
        return
    s.ok[i] = 0
    ref x = q0[i]
    var good = False
    var lf = False
    if x.op == B_COL:
        good = _is_number(x.kind)
        lf = Int(x.value) >= nk
    elif x.op == B_WLIT:
        good = True
        lf = True
    elif x.op == B_LIT:
        good = _is_number(x.kind)
    elif x.op == B_NULL:
        good = True
    elif x.op == B_ADD or x.op == B_SUB or x.op == B_MUL:
        if _is_number(x.kind):
            _scan(q0, x.a, nk, s)
            _scan(q0, x.b, nk, s)
            good = s.ok[x.a] == 1 and s.ok[x.b] == 1
            lf = s.leaf[x.a] or s.leaf[x.b]
    elif x.op == B_NEG or x.op == B_RESCALE:
        if _is_number(x.kind):
            _scan(q0, x.a, nk, s)
            good = s.ok[x.a] == 1
            lf = s.leaf[x.a]
    elif x.op == B_ISNULL:
        _scan(q0, x.a, nk, s)
        good = s.ok[x.a] == 1
        lf = s.leaf[x.a]
    elif x.op == B_CMP:
        _scan(q0, x.a, nk, s)
        _scan(q0, x.b, nk, s)
        good = s.ok[x.a] == 1 and s.ok[x.b] == 1 and (_is_number(q0[x.a].kind) or q0[x.a].op == B_NULL) and (_is_number(q0[x.b].kind) or q0[x.b].op == B_NULL)
        lf = s.leaf[x.a] or s.leaf[x.b]
    s.ok[i] = 1 if good else 0
    s.leaf[i] = lf


def _eval(q0: List[BExpr], i: Int, nk: Int, keys: List[ValCol], aggs: List[ValCol], n: Int) raises -> WCol:
    ref x = q0[i]
    var out = WCol(n)
    if x.op == B_COL:
        var col = Int(x.value)
        if col < nk:
            for r in range(n):
                if keys[col].is_null(r):
                    out.set_null(r)
                else:
                    out.v[r] = keys[col].wide_at(r)
        else:
            for r in range(n):
                if aggs[col - nk].is_null(r):
                    out.set_null(r)
                else:
                    out.v[r] = aggs[col - nk].wide_at(r)
        return out^
    if x.op == B_LIT or x.op == B_WLIT:
        var c = wide_make(x.value, x.hi) if x.op == B_WLIT else Int128(x.value)
        for r in range(n):
            out.v[r] = c
        return out^
    if x.op == B_NULL:
        for r in range(n):
            out.set_null(r)
        return out^
    if x.op == B_ISNULL:
        var a = _eval(q0, x.a, nk, keys, aggs, n)
        for r in range(n):
            out.v[r] = Int128(1 if a.is_null(r) else 0)
        return out^
    if x.op == B_NEG or x.op == B_RESCALE:
        var a = _eval(q0, x.a, nk, keys, aggs, n)
        var m = pow10_wide(Int(x.value)) if x.op == B_RESCALE else Int128(-1)
        for r in range(n):
            if a.is_null(r):
                out.set_null(r)
            else:
                out.v[r] = wide_mul(a.v[r], m)
        return out^
    var a = _eval(q0, x.a, nk, keys, aggs, n)
    var b = _eval(q0, x.b, nk, keys, aggs, n)
    var sa = q0[x.a].scale
    var sb = q0[x.b].scale
    if x.op == B_CMP:
        var ma = Int128(1)
        var mb = Int128(1)
        if sa < sb:
            ma = pow10_wide(sb - sa)
        elif sb < sa:
            mb = pow10_wide(sa - sb)
        for r in range(n):
            if a.is_null(r) or b.is_null(r):
                out.set_null(r)
                continue
            var l = wide_mul(a.v[r], ma)
            var rr = wide_mul(b.v[r], mb)
            var c = Int(x.value)
            var res = l < rr if c == C_LT else l <= rr if c == C_LE else l > rr if c == C_GT else l >= rr if c == C_GE else l == rr if c == C_EQ else l != rr
            out.v[r] = Int128(1 if res else 0)
        return out^
    var ma = Int128(1)
    var mb = Int128(1)
    if x.op != B_MUL:
        if sa < sb:
            ma = pow10_wide(sb - sa)
        elif sb < sa:
            mb = pow10_wide(sa - sb)
    for r in range(n):
        if a.is_null(r) or b.is_null(r):
            out.set_null(r)
        elif x.op == B_MUL:
            out.v[r] = wide_mul(a.v[r], b.v[r])
        elif x.op == B_ADD:
            out.v[r] = wide_add(wide_mul(a.v[r], ma), wide_mul(b.v[r], mb))
        else:
            out.v[r] = wide_sub(wide_mul(a.v[r], ma), wide_mul(b.v[r], mb))
    return out^


def distinct_sorted(values: List[Int128], nul: List[UInt8]) -> List[Int128]:
    var all = List[Int128]()
    for i in range(len(values)):
        if len(nul) == 0 or nul[i] == 0:
            all.append(values[i])
    sort(all)
    var out = List[Int128]()
    for v in all:
        if len(out) == 0 or out[len(out) - 1] != v:
            out.append(v)
    return out^


def rank_of(distinct: List[Int128], v: Int128) -> Int:
    var lo = 0
    var hi = len(distinct) - 1
    while lo < hi:
        var mid = (lo + hi) // 2
        if distinct[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _hidden_node(mut q: BoundQuery, mut rw: WideRewrite, kind: Int, scale: Int, var col: ValCol, proto: BExpr) -> Int:
    """A new node reading a new hidden column."""
    rw.hidden.append(col^)
    q.rexprs.append(BExpr(B_COL, kind, scale, -1, -1, Int64(rw.base + len(rw.hidden) - 1), String(), proto.line, proto.col))
    return len(q.rexprs) - 1


def _materialize(mut q: BoundQuery, i: Int, w: WCol, can_point: Bool, mut rw: WideRewrite) raises:
    """Node `i` of the working pool becomes a read of a hidden column holding `w`: a BOOL column, a column of its kind when every value fits
    int64, else (the node is a root: the select list, ORDER BY) a column of ranks."""
    var n = len(w.v)
    var kind = q.rexprs[i].kind
    var scale = q.rexprs[i].scale
    if kind == BOOL.value:
        scale = 0
    var col = ValCol(kind, scale, n)
    col.nul = w.nul.copy()
    var all_fit = True
    for r in range(n):
        if not w.is_null(r) and not fits_int64(w.v[r]):
            all_fit = False
            break
    if kind == BOOL.value or all_fit:
        for r in range(n):
            if not w.is_null(r):
                col.v[r] = Int64(w.v[r])
        var proto = q.rexprs[i].copy()
        q.rexprs[i] = BExpr(B_COL, kind, scale, -1, -1, Int64(rw.base + len(rw.hidden)), String(), proto.line, proto.col)
        rw.hidden.append(col^)
        return
    if not can_point:
        raise Error(OVERFLOW_MESSAGE)
    var distinct = distinct_sorted(w.v, w.nul)
    for r in range(n):
        if not w.is_null(r):
            col.v[r] = Int64(rank_of(distinct, w.v[r]))
    var proto = q.rexprs[i].copy()
    q.rexprs[i] = BExpr(B_COL, kind, scale, -1, -1, Int64(rw.base + len(rw.hidden)), String(), proto.line, proto.col)
    rw.hidden.append(col^)
    rw.ptr_node.append(i)
    rw.ptr_vals.append(distinct^)


def _is_op(x: BExpr) -> Bool:
    return x.op == B_ADD or x.op == B_SUB or x.op == B_MUL or x.op == B_NEG or x.op == B_RESCALE or x.op == B_CMP or x.op == B_ISNULL


def _visit(mut q: BoundQuery, q0: List[BExpr], i: Int, can_point: Bool, nk: Int, keys: List[ValCol], aggs: List[ValCol], n: Int,
           mut s: _Scan, mut rw: WideRewrite) raises:
    """Top down from a root of the stage (a select item, a sort key, HAVING): the first expression the evaluator here can take and that reads a
    total is evaluated whole; anything else hands its operands on."""
    if s.visited[i]:
        return
    s.visited[i] = True
    var x = q.rexprs[i].copy()
    if x.op == B_COL and Int(x.value) >= nk and Int(x.value) < nk + len(aggs):      # a total read as it is
        if aggs[Int(x.value) - nk].is_wide():
            var w = _eval(q0, i, nk, keys, aggs, n)
            _materialize(q, i, w, can_point, rw)
        return
    if s.ok[i] == 1 and s.leaf[i] and (_is_op(x) or x.op == B_WLIT):
        var w = _eval(q0, i, nk, keys, aggs, n)
        _materialize(q, i, w, can_point, rw)
        return
    # (an expression the evaluator cannot take: its operands may still be)
    if x.a >= 0:
        _visit(q, q0, x.a, False, nk, keys, aggs, n, s, rw)
    if x.b >= 0:
        _visit(q, q0, x.b, False, nk, keys, aggs, n, s, rw)
    for it in x.items:
        _visit(q, q0, it, False, nk, keys, aggs, n, s, rw)


def _float_operand(mut q: BoundQuery, q0: List[BExpr], operand: Int, nk: Int, keys: List[ValCol], aggs: List[ValCol], n: Int, s: _Scan,
                   mut rw: WideRewrite) raises -> Int:
    """The node to read where a FLOAT meets the number `operand`: when it reads a total (or is one that does not fit int64), a new column holding
    the value as a double; else the operand itself (the stage's own shadow columns convert it)."""
    if s.ok[operand] != 1 or not s.leaf[operand]:
        return operand
    var x = q0[operand].copy()
    if x.op == B_COL and not aggs[Int(x.value) - nk].is_wide():
        return operand
    var w = _eval(q0, operand, nk, keys, aggs, n)
    var col = ValCol(FLOAT.value, 0, n)
    col.nul = w.nul.copy()
    for r in range(n):
        if not w.is_null(r):
            col.v[r] = float_bits(decimal_to_float(Int64(w.v[r]), x.scale) if fits_int64(w.v[r]) else wide_to_float(w.v[r], x.scale))
    return _hidden_node(q, rw, FLOAT.value, 0, col^, x)


def wide_prepass(mut q: BoundQuery, nk: Int, keys: List[ValCol], aggs: List[ValCol], n: Int, mut rw: WideRewrite) raises:
    """Rewrites the working pool `q` (post.mojo's `q2`: its select list `routs`, `rorder`, `where` and the nodes under them) as described above."""
    var q0 = q.rexprs.copy()
    var s = _Scan(3 * len(q0) + 8)                                    # (room for the nodes reading the columns made here: at most one per operand)
    for i in range(len(q0)):
        _scan(q0, i, nk, s)
    # a FLOAT meeting a total
    for i in range(len(q0)):
        var x = q0[i].copy()
        var float_ctx = x.op == B_DIV or ((x.op == B_ADD or x.op == B_SUB or x.op == B_MUL) and x.kind == FLOAT.value)
        if x.op == B_CMP and (q0[x.a].kind == FLOAT.value or q0[x.b].kind == FLOAT.value):
            float_ctx = True
        if float_ctx:
            var a = _float_operand(q, q0, x.a, nk, keys, aggs, n, s, rw)
            var b = _float_operand(q, q0, x.b, nk, keys, aggs, n, s, rw)
            q.rexprs[i].a = a
            q.rexprs[i].b = b
    for r in q.routs:
        _visit(q, q0, r, True, nk, keys, aggs, n, s, rw)
    for k in q.rorder:
        _visit(q, q0, k.expr, True, nk, keys, aggs, n, s, rw)
    if q.where >= 0:
        _visit(q, q0, q.where, False, nk, keys, aggs, n, s, rw)
    # nothing the kernel will read may still be a total that does not fit it: a rank column may only be a select item or a sort key itself (a
    # parent that is not a plain read of it, such as a CASE arm, would compute with ranks)
    var is_rank = List[Bool](length=len(q.rexprs), fill=False)
    for p in rw.ptr_node:
        is_rank[p] = True
    var seen = List[Bool](length=len(q.rexprs), fill=False)
    var todo = List[Int]()
    for r in q.routs:
        todo.append(r)
    for k in q.rorder:
        todo.append(k.expr)
    if q.where >= 0:
        todo.append(q.where)
    while len(todo) > 0:
        var i = todo.pop()
        if seen[i]:
            continue
        seen[i] = True
        ref x = q.rexprs[i]
        if x.op == B_WLIT:
            raise Error(WIDE_LITERAL_MESSAGE + " (" + x.text + ")")
        if x.op == B_COL and Int(x.value) >= nk and Int(x.value) < nk + len(aggs) and aggs[Int(x.value) - nk].is_wide():
            raise Error(OVERFLOW_MESSAGE)
        var kids = List[Int]()
        if x.a >= 0:
            kids.append(x.a)
        if x.b >= 0:
            kids.append(x.b)
        for it in x.items:
            kids.append(it)
        for c in kids:
            if is_rank[c]:
                raise Error(OVERFLOW_MESSAGE)
            todo.append(c)


def fix_wide(mut table: Table, q: BoundQuery, rw: WideRewrite) raises:
    """The finished table's rank columns turned back into the 128-bit values they stand for."""
    for j in range(len(q.routs)):
        for p in range(len(rw.ptr_node)):
            if rw.ptr_node[p] != q.routs[j]:
                continue
            ref col = table.cols[j]
            var rows = len(col.values)
            col.high = List[Int64](length=rows, fill=0)
            for r in range(rows):
                if col.is_null(r):
                    continue
                var v = rw.ptr_vals[p][Int(col.values[r])]
                col.values[r] = wide_lo(v)
                col.high[r] = wide_hi(v)
            break
