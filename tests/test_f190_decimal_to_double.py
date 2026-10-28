"""F-190: an INT or DECIMAL past 2^53 read as a DOUBLE rounds to the nearest double, equal to DuckDB bit for bit (it raised "int64 range exceeded").

DuckDB's rule (probed here, bit for bit, not assumed): a DECIMAL whose storage is an int64 (precision <= 18) reads as a double
- up to +-2^53 (the integer is exact): float(raw) / 10^scale;
- past it: the whole part and the fraction are converted apart, float(raw / 10^scale) + float(raw % 10^scale) / 10^scale, the division
  truncating toward zero (so the plain quotient and the correctly rounded value are both wrong by an ulp in many cases);
a BIGINT is converted as it is. `test_the_rule_...` checks that formula against DuckDB on the adversarial values at every scale, so the
Mojo engine's rule (kernels_aot/pipeline_cpu.mojo `_decimal_f64`, bind.mojo `decimal_to_float`) is known to be DuckDB's.

The engine runs through the real `mx` command (tests/mx_oracle.py); TPC-H Q14 runs over DuckDB's own dbgen at SF1 (~/.cache/omxcore-gate).
"""
import os
import random
import struct
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from decimal import Decimal

from mx_oracle import check, mx_binary, run_mx

TWO53 = 1 << 53
TWO62 = 1 << 62
LIMIT18 = 10**18 - 1                      # the largest DECIMAL(18, s) storage integer


@pytest.fixture(scope="module")
def mx():
    return mx_binary()


def _raws():
    """Storage integers around 2^53, 2^59 (the DECIMAL(18) ceiling is ~2^59.8) and, for BIGINT, 2^62 and the ends of int64, both signs."""
    base = [0, 1, 7, TWO53 - 1, TWO53, TWO53 + 1, TWO53 + 2, TWO53 + 3, TWO53 * 3 + 1, TWO53 * 5 + 3, (1 << 58) + 12345, 10**17 + 1, 10**18 - 1,
            999_999_999_999_999_937, 123_456_789_012_345_678]
    rnd = random.Random(190)
    base += [rnd.randrange(TWO53, LIMIT18) for _ in range(120)]
    base += [rnd.randrange(1, TWO53) for _ in range(20)]
    return sorted(set(base + [-v for v in base]))


BIG = [v for v in (TWO62 + 12345, TWO62 + 1, (1 << 63) - 1, (1 << 63) - 2, (1 << 63) - 1025, (1 << 63) - 513, 9_007_199_254_740_993 * 1000) if v < (1 << 63)]
BIG = sorted(set(BIG + [-v for v in BIG] + [-(1 << 63) + 1]))


@pytest.fixture(scope="module")
def wide(tmp_path_factory):
    """One row per probe value; a BIGINT column and a DECIMAL(18, s) column for every scale 0..18."""
    d = tmp_path_factory.mktemp("f190")
    raws = _raws()
    cols = {"id": pa.array(range(len(raws) + len(BIG)), pa.int64()), "b": pa.array(raws + BIG, pa.int64())}
    for s in range(0, 19):
        vals = [Decimal(v).scaleb(-s) for v in raws] + [None] * len(BIG)
        cols[f"d{s}"] = pa.array(vals, pa.decimal128(18, s))
    path = d / "wide.parquet"
    pq.write_table(pa.table(cols), path, row_group_size=97)
    con = duckdb.connect()
    con.execute(f"create table t as select * from read_parquet('{path}')")
    return {"path": path, "con": con, "raws": raws}


def _bits(x):
    return struct.pack("<d", x)


def duckdb_double(raw, scale):
    p = 10**scale
    if abs(raw) <= TWO53:
        return float(raw) / float(p)
    q = abs(raw) // p
    x = float(q) + float(abs(raw) - q * p) / float(p)
    return -x if raw < 0 else x


def test_the_rule_is_split_past_2_53(wide):
    """DuckDB's DECIMAL -> DOUBLE on every probe value at every scale, bit for bit; the BIGINT is float(raw)."""
    con = wide["con"]
    raws = wide["raws"]
    for s in range(0, 19):
        got = [r[0] for r in con.execute(f"select cast(d{s} as double) from t where d{s} is not null order by id").fetchall()]
        want = [duckdb_double(v, s) for v in raws]
        assert [_bits(x) for x in got] == [_bits(x) for x in want], f"scale {s}"
    got = [r[0] for r in con.execute("select cast(b as double) from t order by id").fetchall()]
    assert [_bits(x) for x in got] == [_bits(float(v)) for v in raws + BIG]


def test_cast_and_arithmetic_equal_duckdb_at_every_scale(mx, wide):
    con, path = wide["con"], {"t": wide["path"]}
    for s in range(0, 19):
        check(mx, f"SELECT id, CAST(d{s} AS DOUBLE) AS c, d{s} * 1.5e0 AS m, d{s} / 3 AS q, d{s} + 0.25e0 AS a FROM t WHERE d{s} IS NOT NULL ORDER BY id", con, path, ordered=True)
    check(mx, "SELECT id, CAST(b AS DOUBLE) AS c, b * 1.0e0 AS m, b / 7 AS q, b + 0.5e0 AS a FROM t ORDER BY id", con, path, ordered=True)


def test_min_and_max_of_big_values_equal_duckdb(mx, wide):
    con, path = wide["con"], {"t": wide["path"]}
    check(mx, "SELECT min(d2 * 1.0e0), max(d2 * 1.0e0), min(b + 0.0e0), max(b + 0.0e0) FROM t", con, path)


def test_the_probes_of_the_review(mx, tmp_path):
    """sum(x)/3 and 100.00*sum(x) worked, 100.00*sum(x)/2.0 raised: a sum whose scaled raw integer passes 2^53."""
    n = 5000
    t = pa.table({"x": pa.array([Decimal("1234567.89") + i for i in range(n)], pa.decimal128(15, 2)),
                  "y": pa.array([Decimal("98765.43") + i for i in range(n)], pa.decimal128(15, 2))})
    path = tmp_path / "x.parquet"
    pq.write_table(t, path)
    con = duckdb.connect()
    con.execute(f"create table t as select * from read_parquet('{path}')")
    for sql in ["SELECT sum(x) / 3 FROM t", "SELECT 100.00 * sum(x) FROM t", "SELECT 100.00 * sum(x) / 2.0 FROM t",
                "SELECT 100.00 * sum(x) / sum(y) FROM t", "SELECT 100.00 * sum(x * 1000) / sum(y * 1000) FROM t"]:
        check(mx, sql, con, {"t": path})


def test_a_sum_that_really_passes_int64_still_raises(mx, tmp_path):
    t = pa.table({"v": pa.array([Decimal(10**17 - 1)] * 100, pa.decimal128(18, 0))})
    path = tmp_path / "o.parquet"
    pq.write_table(t, path)
    out = run_mx(mx, ["sql", "SELECT sum(v) / 2.0 FROM t"], {"t": path}, expect_exit=1)
    assert out is None or "int64 range" in str(out)


# ── TPC-H Q14 at SF1, over DuckDB's own dbgen ─────────────────────────────────────────────────────────

GATE_DATA = Path(os.environ.get("OMXCORE_GATE_DATA", Path.home() / ".cache" / "omxcore-gate"))
TABLES = ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]


def tpch_dir(sf="1"):
    """Typed TPC-H Parquet written once by DuckDB's tpch extension (DECIMAL(15,2), DATE, ...), as scripts/gate_common.py does on the gate branch."""
    d = GATE_DATA / f"tpch_sf{sf}"
    if all((d / f"{t}.parquet").exists() for t in TABLES):
        return d
    tmp = d.with_name(d.name + f".partial{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    tmpdb = tmp / "dbgen.duckdb"
    con = duckdb.connect(str(tmpdb))
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={sf})")
    for t in TABLES:
        con.execute(f"COPY {t} TO '{tmp / (t + '.parquet')}' (FORMAT parquet)")
    con.close()
    tmpdb.unlink()
    try:
        tmp.rename(d)
    except OSError:                                     # another worker finished first
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()
    return d


def test_tpch_q14_at_sf1_equals_duckdb(mx):
    d = tpch_dir("1")
    files = {"lineitem": d / "lineitem.parquet", "part": d / "part.parquet"}
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    q14 = con.execute("SELECT query FROM tpch_queries() WHERE query_nr = 14").fetchone()[0].strip().rstrip(";")
    for name, path in files.items():
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
    res = check(mx, q14, con, files)
    assert res["row_count"] == 1


def test_aggregates_read_as_doubles_follow_the_same_rule(mx, tmp_path):
    """Review: the shadow column of `sum(d) * 1.5e0` / `sum(d) / 3` (post.mojo) converted with the plain quotient: an ulp off for group sums past 2^53."""
    rnd = random.Random(7)
    rows = [(g, rnd.randrange(10**15, 4 * 10**16)) for g in range(300) for _ in range(12)]
    cols = {"g": pa.array([g for g, _ in rows], pa.int64())}
    for s in (0, 1, 2, 5, 9):
        cols[f"d{s}"] = pa.array([Decimal(v).scaleb(-s) for _, v in rows], pa.decimal128(18, s))
    path = tmp_path / "grp.parquet"
    pq.write_table(pa.table(cols), path)
    con = duckdb.connect()
    con.execute(f"create table t as select * from read_parquet('{path}')")
    for s in (0, 1, 2, 5, 9):
        check(mx, f"SELECT g, sum(d{s}) * 1.5e0 AS a, sum(d{s}) / 3 AS b, avg(d{s}) * 1.0e0 AS c, max(d{s}) + 0.5e0 AS m FROM t GROUP BY g", con, {"t": path})


def test_window_sums_and_averages_read_as_doubles_follow_the_same_rule(mx, tmp_path):
    """Review 2: `sum(d) OVER (...) * 1.5e0`, `/ 3`, `avg(d) OVER (...) * 1.0e0` over window sums past 2^53 (inside int64), bit for bit.
    The result goes out as Parquet so no JSON text rounds it."""
    import subprocess
    rnd = random.Random(11)
    n = 600
    cols = {"g": pa.array([i % 5 for i in range(n)], pa.int64()), "o": pa.array(range(n), pa.int64())}
    scales = (0, 2, 5, 9)
    for s in scales:
        vals = [rnd.randrange(10**15, 4 * 10**16) if i % 4 else rnd.randrange(10**3, 10**9) for i in range(n)]
        cols[f"d{s}"] = pa.array([Decimal(v).scaleb(-s) for v in vals], pa.decimal128(18, s))
    src = tmp_path / "w.parquet"
    pq.write_table(pa.table(cols), src)
    con = duckdb.connect()
    con.execute(f"create table t as select * from read_parquet('{src}')")
    for s in scales:
        for sql in [
            f"SELECT o, sum(d{s}) OVER (ORDER BY o ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) * 1.5e0 AS a, "
            f"avg(d{s}) OVER (ORDER BY o ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING) * 1.0e0 AS b FROM t ORDER BY o",
            f"SELECT o, sum(d{s}) OVER (PARTITION BY g ORDER BY o ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) / 3 AS a, "
            f"avg(d{s}) OVER (PARTITION BY g ORDER BY o ROWS BETWEEN 6 PRECEDING AND 1 FOLLOWING) * 1.0e0 AS b FROM t ORDER BY o",
        ]:
            out = tmp_path / "o.parquet"
            p = subprocess.run([str(mx), "sql", sql, "-t", f"t={src}", "--format", "parquet", "--out", str(out), "--max-rows", "1000000"], capture_output=True)
            assert p.returncode == 0, (sql, p.stdout[-300:], p.stderr[-300:])
            got = pq.read_table(out)
            want = con.execute(sql).fetchall()
            assert got.num_rows == len(want) == n
            for ci in (1, 2):
                g = got.column(ci).to_pylist()
                assert [_bits(x) for x in g] == [_bits(float(r[ci])) for r in want], f"{sql} column {ci}"
