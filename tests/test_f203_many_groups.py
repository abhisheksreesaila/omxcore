"""F-203: GROUP BY with very many groups (a near-unique id), HAVING and ORDER BY an aggregate LIMIT over them. Every answer is DuckDB's, through the real
`mx` command (tests/mx_oracle.py), over shapes the hashed group path meets: near-unique keys (a wide range, so the planner hashes them), skewed keys, a NULL
key group, TEXT keys, two-column keys, HAVING over every aggregate kind (and 128-bit sums), ORDER BY an aggregate with LIMIT, and random worlds."""
import random
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import oracle_connection
from mx_oracle import check, mx_binary, run_mx

ROWS = 40000


@pytest.fixture(scope="module")
def mx():
    return mx_binary()


def _world(rnd, rows):
    uid, skew, nul, txt, a, b, d, wide, f = [], [], [], [], [], [], [], [], []
    ids = rnd.sample(range(1, 10**12), rows // 3)                  # ~3 rows per id, ids spread over a 10^12 range: hashed groups
    for i in range(rows):
        uid.append(ids[i % len(ids)] if rnd.random() > 0.002 else None)       # a NULL key group
        skew.append(int(rnd.paretovariate(1.2)) % 5000)                       # a few hot keys and a long tail
        nul.append(None if rnd.random() < 0.01 else rnd.randrange(0, 20000))
        txt.append("k%06d" % rnd.randrange(0, 15000))
        a.append(rnd.randrange(0, 20000))
        b.append(rnd.randrange(0, 4))
        d.append(Decimal(rnd.randrange(0, 5000)).scaleb(-2) if rnd.random() > 0.05 else None)
        wide.append(Decimal(rnd.randrange(10**16, 10**18 - 1)).scaleb(-2))   # sums of these pass int64 within a group
        f.append(rnd.random() * 100 if rnd.random() > 0.05 else None)
    return pa.table({"uid": pa.array(uid, pa.int64()), "skew": pa.array(skew, pa.int64()), "nul": pa.array(nul, pa.int64()),
                     "txt": pa.array(txt, pa.string()), "a": pa.array(a, pa.int64()), "b": pa.array(b, pa.int32()),
                     "d": pa.array(d, pa.decimal128(12, 2)), "wide": pa.array(wide, pa.decimal128(18, 2)), "f": pa.array(f, pa.float64())})


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    path = tmp_path_factory.mktemp("f203") / "t.parquet"
    pq.write_table(_world(random.Random(203), ROWS), path, row_group_size=7000)
    con = oracle_connection()
    con.execute(f"create table t as select * from read_parquet('{path}')")
    return {"files": {"t": str(path)}, "con": con}


QUERIES = [
    # near-unique keys
    "select uid from t group by uid having sum(a) > 30000 order by uid",
    "select uid, sum(a) s from t group by uid having sum(a) > 30000 order by s desc, uid limit 20",
    "select uid, count(*) c from t group by uid having count(*) >= 5 order by uid",
    "select uid, count(*) c from t group by uid order by c desc, uid limit 7",
    "select count(*) from (select uid from t group by uid having sum(a) > 20000) x",
    "select uid, sum(a) s from t group by uid having sum(a) > 1000000000 order by uid",          # nothing survives
    "select uid, sum(a) s from t group by uid having sum(a) >= 0 order by uid limit 5",           # everything survives
    # NULL key group
    "select nul, sum(a), count(*) from t group by nul having sum(a) > 40000 order by nul",
    "select nul, count(*) c from t group by nul having nul is null",
    "select nul, count(*) c from t group by nul having count(*) > 1 order by c desc, nul limit 10",
    # skewed keys
    "select skew, sum(a) s, count(*) c from t group by skew having count(*) > 20 order by skew",
    "select skew, count(*) c from t group by skew order by c desc, skew limit 10",
    # text keys
    "select txt, sum(a) s from t group by txt having sum(a) > 30000 order by txt",
    "select txt, count(*) c from t group by txt order by c desc, txt limit 5",
    # two-column keys
    "select uid, b, sum(a) s from t group by uid, b having sum(a) > 15000 order by uid, b",
    "select txt, b, count(*) c from t group by txt, b having count(*) > 3 order by txt, b",
    "select skew, b, sum(d) s from t group by skew, b order by s desc nulls last, skew, b limit 8",
    # HAVING over every aggregate kind
    "select nul, count(d) from t group by nul having count(d) > 2 order by nul",
    "select nul, sum(d) from t group by nul having sum(d) > 200 order by nul",
    "select nul, avg(d) from t group by nul having avg(d) > 26 order by nul",
    "select nul, min(a) from t group by nul having min(a) < 1500 order by nul",
    "select nul, max(a) from t group by nul having max(a) > 19990 order by nul",
    "select nul, sum(f) from t group by nul having sum(f) > 500 order by nul",
    "select nul, avg(f), min(f), max(f) from t group by nul having max(f) > 99.9 order by nul",
    "select nul, sum(a), count(*), min(d), max(d), avg(a) from t group by nul having sum(a) > 30000 and count(*) > 2 order by nul",
    "select nul, sum(a) from t group by nul having sum(a) > 30000 or count(*) > 4 order by nul",
    "select nul, sum(a) from t group by nul having not (sum(a) < 30000) order by nul",
    "select nul, sum(a) from t group by nul having nul > 100 and sum(a) > 20000 order by nul",
    "select nul, sum(a) from t group by nul having sum(a) > count(*) * 15000 order by nul",
    "select nul, sum(a) from t group by nul having sum(d) is null order by nul",
    "select nul, count(distinct a) from t group by nul having count(distinct a) > 3 order by nul",
    # 128-bit sums
    "select skew, sum(wide) s from t group by skew having sum(wide) > 90000000000000000.00 order by skew",
    "select uid, sum(wide) s from t group by uid having sum(wide) > 1000000000000000.00 order by uid",
    "select b, sum(wide) s, avg(wide) from t group by b having sum(wide) > 0 order by b",
    # ORDER BY an aggregate with LIMIT
    "select uid, sum(d) s from t group by uid order by s desc nulls last, uid limit 10",
    "select uid, avg(a) s from t group by uid order by s, uid limit 10",
    "select uid, sum(a) s from t group by uid having count(*) > 2 order by s desc, uid limit 10 offset 3",
    # a select list that is not just the key
    "select uid, sum(a) * 2 + 1, count(*) from t group by uid having sum(a) > 30000 order by uid",
    "select uid % 100 k, sum(a) from t group by uid % 100 having sum(a) > 100000 order by k",
    "select sum(a) from t group by uid having sum(a) > 30000 order by 1",
    # filters before the group
    "select uid, sum(a) from t where b > 1 group by uid having sum(a) > 20000 order by uid",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_many_groups_equal_duckdb(mx, world, sql):
    check(mx, sql, world["con"], world["files"], ordered=True)


@pytest.mark.parametrize("seed", range(6))
def test_random_worlds(mx, tmp_path, seed):
    rnd = random.Random(seed * 7919 + 1)
    path = tmp_path / "r.parquet"
    pq.write_table(_world(rnd, 3000 + seed * 700), path, row_group_size=rnd.choice([500, 1100, 4096]))
    con = oracle_connection()
    con.execute(f"create table t as select * from read_parquet('{path}')")
    cut = rnd.randrange(1, 40000)
    for sql in (f"select uid, sum(a) s, count(*) c from t group by uid having sum(a) > {cut} order by uid",
                f"select uid, b, avg(d) from t group by uid, b having max(a) > {cut // 3} order by uid, b",
                f"select txt, min(f), max(f) from t group by txt having sum(a) > {cut} order by txt",
                f"select uid, sum(a) s from t group by uid order by s desc, uid limit {rnd.randrange(1, 50)}",
                f"select skew, sum(wide) from t group by skew having sum(wide) > {cut}000000000000.00 order by skew"):
        check(mx, sql, con, {"t": str(path)}, ordered=True)
