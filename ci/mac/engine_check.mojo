# F-182 / F-186: omxcore on this machine's GPU. Reads the orders CSV (ci/mac/orders.py), uploads it once, runs each
# query with device=GPU and prints `Q <id> ran=<gpu|cpu> <answer>` in ci/mac/orders.py's format (pyarrow's answers are
# its `R` lines; scripts/check_apple_gpu.py compares them). A query that raises prints `Q <id> ran=error <message>`.
#   mojo build -I build ci/mac/engine_check.mojo -o engine_check && ./engine_check orders.csv
from std.sys import argv
from omxcore import (read_csv, CsvOptions, Query, QueryResult, SharedTable, Expr, Agg, col, lit, sum, count, min_of,
                     max_of, GPU, DeviceTable, gpu_available, gpu_device)


def _flat(r: QueryResult, n: Int) -> String:
    var s = String()
    for i in range(n):
        if i > 0:
            s += ","
        s += String(r.values(i)[0])
    return s


def _grouped(r: QueryResult, n: Int, t: SharedTable, key: String) raises -> String:
    var c = t.column(key)
    var words = t.dictionary(c)
    var order = List[Int]()                      # groups sorted by their word
    for g in range(len(words)):
        var at = len(order)
        for k in range(len(order)):
            if words[g] < words[order[k]]:
                at = k
                break
        order.insert(at, g)
    var s = String()
    for g in order:
        if r.counts[g] == 0:
            continue
        if s.byte_length() > 0:
            s += "|"
        s += words[g] + "="
        for i in range(n):
            if i > 0:
                s += ","
            s += String(r.values(i)[g])
    return s


def _top5(r: QueryResult, t: SharedTable, key: String) raises -> String:
    var c = t.column(key)
    var words = t.dictionary(c)
    var v = r.values(0)
    var order = List[Int]()                      # largest sum first, ties by word
    for g in range(len(words)):
        if r.counts[g] == 0:
            continue
        var at = len(order)
        for k in range(len(order)):
            var o = order[k]
            if v[g] > v[o] or (v[g] == v[o] and words[g] < words[o]):
                at = k
                break
        order.insert(at, g)
    var s = String()
    for k in range(min(5, len(order))):
        if k > 0:
            s += "|"
        s += words[order[k]] + "=" + String(v[order[k]])
    return s


def _ran(r: QueryResult) -> String:
    return "gpu" if r.device == GPU else "cpu"


def main() raises:
    var available = gpu_available()
    print("ENGINE AVAILABLE:", available)
    if not available:
        print("ENGINE DEVICE: none")
        return
    print("ENGINE DEVICE:", gpu_device())
    var csv = read_csv(String(argv()[1]), CsvOptions(decimal_columns=["amount"]))
    var t = csv.take_table().share()
    print("ENGINE ROWS:", t.rows())
    var dt = DeviceTable(t)
    var big = (col("qty") > lit(10)) & (col("amount") < lit(500_000))
    var ids: List[String] = ["count", "sums", "minmax", "by-status", "by-status-minmax", "where", "where-by-region",
                             "nulls-by-status", "top5-customer", "top5-region-where", "many-groups", "pivot30"]
    for qi in range(len(ids)):
        var id = ids[qi]
        try:
            var q = Query(t).on_device(dt)
            var line = String()
            var r: QueryResult
            if id == "count":
                r = q^.agg(count()).compute(GPU)
                line = _flat(r, 1)
            elif id == "sums":
                r = q^.agg(sum(col("qty")), sum(col("amount"))).compute(GPU)
                line = _flat(r, 2)
            elif id == "minmax":
                r = q^.agg(min_of(col("qty")), max_of(col("qty")), min_of(col("amount")), max_of(col("amount"))).compute(GPU)
                line = _flat(r, 4)
            elif id == "by-status":
                var g = len(t.dictionary(t.column("status")))
                r = q^.group_by(col("status"), groups=g).agg(count(), sum(col("qty")), sum(col("amount"))).compute(GPU)
                line = _grouped(r, 3, t, "status")
            elif id == "by-status-minmax":
                var g = len(t.dictionary(t.column("status")))
                r = q^.group_by(col("status"), groups=g).agg(min_of(col("qty")), max_of(col("qty")), min_of(col("amount")),
                                                            max_of(col("amount"))).compute(GPU)
                line = _grouped(r, 4, t, "status")
            elif id == "where":
                r = q^.where(big.copy()).agg(count(), sum(col("amount"))).compute(GPU)
                line = _flat(r, 2)
            elif id == "where-by-region":
                var g = len(t.dictionary(t.column("region")))
                r = q^.where(big.copy()).group_by(col("region"), groups=g).agg(count(), sum(col("amount")), max_of(col("qty"))).compute(GPU)
                line = _grouped(r, 3, t, "region")
            elif id == "nulls-by-status":
                var g = len(t.dictionary(t.column("status")))
                r = q^.group_by(col("status"), groups=g).agg(count(col("discount")), sum(col("discount")), max_of(col("discount"))).compute(GPU)
                line = _grouped(r, 3, t, "status")
            elif id == "top5-customer":
                var g = len(t.dictionary(t.column("customer")))
                r = q^.group_by(col("customer"), groups=g).agg(sum(col("amount"))).compute(GPU)
                line = _top5(r, t, "customer")
            elif id == "top5-region-where":
                var g = len(t.dictionary(t.column("region")))
                r = q^.where(big.copy()).group_by(col("region"), groups=g).agg(sum(col("amount"))).compute(GPU)
                line = _top5(r, t, "region")
            elif id == "many-groups":
                r = q^.group_by(col("id") % 5000, groups=5000).agg(sum(col("amount")), count()).compute(GPU)
                for g in range(5000):
                    if r.counts[g] > 0:
                        if line.byte_length() > 0:
                            line += "|"
                        line += String(g) + "=" + String(r.values(0)[g]) + "," + String(r.values(1)[g])
            else:
                var g = len(t.dictionary(t.column("status")))
                var p = q^.group_by(col("status"), groups=g)
                for i in range(30):
                    var a = sum(col("qty") + i) if i % 3 == 0 else (min_of(col("amount") * (i + 1)) if i % 3 == 1 else max_of(col("amount") - i))
                    p = p^.agg(a^)
                r = p^.compute(GPU)
                line = _grouped(r, 30, t, "status")
            print("Q", id, "ran=" + _ran(r), line)
        except e:
            print("Q", id, "ran=error", String(e))
    dt.release()
