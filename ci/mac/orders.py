"""F-186: the orders data for the Apple GPU check, and pyarrow's answers to the engine's queries.

    python ci/mac/orders.py <out.csv> [rows]     writes the CSV and prints one `R <id> <answer>` line per query

Columns: id, qty (INT 1..50), amount (DECIMAL, two places), discount (INT, about 5% empty = null), status (5 words),
region (12 words), customer (300 words). The answers use the same text as ci/mac/engine_check.mojo's `Q` lines:
a list of values, or `word=values` per group sorted by word, or the top five groups by a sum (largest first, ties by
word). Money is in cents (the DECIMAL's stored integers).
"""

import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

STATUS = ["cancelled", "delivered", "pending", "returned", "shipped"]
REGION = ["Atlantic", "Central", "East", "Gulf", "Lakes", "Midwest", "Mountain", "North", "Pacific", "Plains", "South", "West"]
CUSTOMER = [f"c{i:03d}" for i in range(300)]


def make(rows):
    rng = np.random.default_rng(20261024)
    return {
        "id": np.arange(rows, dtype=np.int64),
        "qty": rng.integers(1, 51, rows, dtype=np.int64),
        "amount": rng.integers(1, 1_000_000, rows, dtype=np.int64),       # cents: 0.01 .. 9999.99
        "discount": rng.integers(0, 30, rows, dtype=np.int64),
        "discount_null": rng.random(rows) < 0.05,
        "status": rng.integers(0, len(STATUS), rows),
        "region": rng.integers(0, len(REGION), rows),
        "customer": rng.integers(0, len(CUSTOMER), rows),
    }


def write_csv(d, path):
    rows = len(d["id"])
    cents = d["amount"]
    with open(path, "w") as f:
        f.write("id,qty,amount,discount,status,region,customer\n")
        for i in range(rows):
            disc = "" if d["discount_null"][i] else str(d["discount"][i])
            f.write(f"{d['id'][i]},{d['qty'][i]},{cents[i] // 100}.{cents[i] % 100:02d},{disc},"
                    f"{STATUS[d['status'][i]]},{REGION[d['region'][i]]},{CUSTOMER[d['customer'][i]]}\n")


def table(d):
    return pa.table({
        "id": d["id"], "qty": d["qty"], "amount": d["amount"],
        "discount": pa.array(d["discount"], mask=d["discount_null"]),
        "status": pa.array([STATUS[i] for i in d["status"]]),
        "region": pa.array([REGION[i] for i in d["region"]]),
        "customer": pa.array([CUSTOMER[i] for i in d["customer"]]),
    })


def flat(t, specs):
    out = []
    for c, f in specs:
        if f == "count_all":
            out.append(t.num_rows)
        elif f == "count":
            out.append(pc.count(t[c]).as_py())
        else:
            out.append(getattr(pc, f)(t[c]).as_py())
    return ",".join(str(v) for v in out)


def grouped(t, key, specs):
    aggs = [(c, f) if f != "count_all" else ([], "count_all") for c, f in specs]
    g = t.group_by(key).aggregate(aggs)
    names = [f"{c}_{f}" if f != "count_all" else "count_all" for c, f in specs]
    rows = sorted(zip(g[key].to_pylist(), *[g[n].to_pylist() for n in names]))
    return "|".join(r[0] + "=" + ",".join(str(v) for v in r[1:]) for r in rows)


def top5(t, key, col):
    g = t.group_by(key).aggregate([(col, "sum")])
    rows = sorted(zip(g[key].to_pylist(), g[f"{col}_sum"].to_pylist()), key=lambda r: (-r[1], r[0]))[:5]
    return "|".join(f"{k}={v}" for k, v in rows)


def answers(t):
    big = pc.and_(pc.greater(t["qty"], 10), pc.less(t["amount"], 500_000))
    w = t.filter(big)
    return [
        ("count", flat(t, [(None, "count_all")])),
        ("sums", flat(t, [("qty", "sum"), ("amount", "sum")])),
        ("minmax", flat(t, [("qty", "min"), ("qty", "max"), ("amount", "min"), ("amount", "max")])),
        ("by-status", grouped(t, "status", [(None, "count_all"), ("qty", "sum"), ("amount", "sum")])),
        ("by-status-minmax", grouped(t, "status", [("qty", "min"), ("qty", "max"), ("amount", "min"), ("amount", "max")])),
        ("where", flat(w, [(None, "count_all"), ("amount", "sum")])),
        ("where-by-region", grouped(w, "region", [(None, "count_all"), ("amount", "sum"), ("qty", "max")])),
        ("nulls-by-status", grouped(t, "status", [("discount", "count"), ("discount", "sum"), ("discount", "max")])),
        ("top5-customer", top5(t, "customer", "amount")),
        ("top5-region-where", top5(w, "region", "amount")),
        ("many-groups", many(t)),
        ("pivot30", pivot(t)),
    ]


def many(t):
    """5000 groups (id % 5000): sum of amount and count; on an Apple GPU the many-groups mode (5). Groups in key order."""
    k = pc.subtract(t["id"], pc.multiply(pc.divide(t["id"], 5000), 5000))      # id % 5000 (ids are >= 0)
    g = pa.table({"k": k, "amount": t["amount"]}).group_by("k").aggregate([("amount", "sum"), ([], "count_all")])
    rows = sorted(zip(g["k"].to_pylist(), g["amount_sum"].to_pylist(), g["count_all"].to_pylist()))
    return "|".join(f"{k}={a},{c}" for k, a, c in rows)


def pivot(t):
    """30 aggregates over status (a big program: 64-thread narrow blocks on an Apple GPU): for i in 0..29,
    sum(qty + i) if i % 3 == 0, min(amount * (i + 1)) if i % 3 == 1, else max(amount - i)."""
    out = []
    for name in sorted(set(t["status"].to_pylist())):
        sub = t.filter(pc.equal(t["status"], name))
        vals = []
        for i in range(30):
            if i % 3 == 0:
                vals.append(pc.sum(pc.add(sub["qty"], i)).as_py())
            elif i % 3 == 1:
                vals.append(pc.min(pc.multiply(sub["amount"], i + 1)).as_py())
            else:
                vals.append(pc.max(pc.subtract(sub["amount"], i)).as_py())
        out.append(name + "=" + ",".join(str(v) for v in vals))
    return "|".join(out)


def main():
    path = sys.argv[1]
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    d = make(rows)
    write_csv(d, path)
    for qid, a in answers(table(d)):
        print("R", qid, a)


if __name__ == "__main__":
    main()
