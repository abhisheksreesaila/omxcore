"""F-212: a WHERE or CASE over nullable columns (comparison, AND, OR, NOT, IS [NOT] NULL) equals DuckDB on every device, and runs on the GPU.

The condition is three-valued: a comparison with a NULL is unknown, NOT unknown is unknown, FALSE AND unknown is false, TRUE OR unknown is true, and only a row
where the whole condition is true is kept. The engine lowers such a condition to integer instructions (known-true / known-false masks, no 0.5 constant and
no float compare), which an Apple GPU's Int64 word runs; the first query below was refused there ("the Apple GPU kernel runs integer programs only").
Each condition is run on the CPU and on the GPU (skipped without one) and compared with DuckDB's answer, so a wrong NOT or a NULL kept as true shows."""
import duckdb
import pytest

from mx_oracle import check, mx_binary, run_mx

CONDITIONS = [
    "qty > 3",
    "qty IS NULL",
    "qty IS NOT NULL AND qty <> 2",
    "NOT (qty > 3)",
    "NOT (qty > 3) OR qty IS NULL",
    "qty > 3 OR price < 0",
    "qty > 3 AND price < 0",
    "NOT (qty > 3 AND price < 0)",
    "NOT (qty > 3 OR price < 0)",
    "NOT (NOT (qty > 3))",
    "(qty > 3 OR other < 0) AND NOT (qty < 0 AND other < 0)",
    "qty + other > 5",
    "qty * 2 = other",
    "qty <> other",
    "qty IS NULL OR other IS NULL",
    "NOT (qty IS NULL) AND other IS NOT NULL",
    "TRUE AND qty > 3",
    "FALSE OR qty > 3",
]


@pytest.fixture(scope="module")
def mx():
    return mx_binary()


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """400 rows; qty, other, price, s (text) and d (date) each hold NULLs (every 7th, 5th, 11th, 6th and 9th row, offset so that every combination of
    unknowns occurs)."""
    path = tmp_path_factory.mktemp("f212") / "t.csv"
    rows = ["g,qty,other,price,s,d"]
    s = 4321
    for i in range(400):
        s = (s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        v = (s >> 33) % 1000
        qty = "" if i % 7 == 2 else str(v % 11 - 3)
        other = "" if i % 5 == 1 else str((v // 7) % 13 - 4)
        price = "" if i % 11 == 6 else str(v * 7 - 2000)
        s_ = "" if i % 6 == 3 else "abcde"[v % 5]
        d = "" if i % 9 == 4 else f"2024-0{v % 9 + 1}-1{v % 10}"
        rows.append(f"{i % 4},{qty},{other},{price},{s_},{d}")
    path.write_text("\n".join(rows) + "\n")
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t AS SELECT * FROM read_csv('{path}')")
    return path, con


@pytest.fixture(scope="module")
def has_gpu(mx, data):
    return run_mx(mx, ["sql", "--device", "gpu", "SELECT sum(price) FROM t"], files={"t": data[0]})["device"] == "gpu"


@pytest.mark.parametrize("cond", CONDITIONS)
def test_a_where_over_nullable_columns_equals_duckdb(mx, data, cond):
    path, con = data
    sql = f"SELECT g, count(*), sum(price), sum(qty), min(other) FROM t WHERE {cond} GROUP BY g ORDER BY g"
    assert con.execute(f"SELECT count(*) FROM t WHERE {cond}").fetchone()[0] not in (0, 400), "the condition keeps some rows and drops others"
    check(mx, sql, con, files={"t": path}, ordered=True, extra=("--device", "cpu"))


@pytest.mark.parametrize("cond", CONDITIONS)
def test_the_same_where_runs_on_the_gpu_and_equals_duckdb(mx, data, has_gpu, cond):
    if not has_gpu:
        pytest.skip("no GPU on this machine")
    path, con = data
    sql = f"SELECT g, count(*), sum(price), sum(qty), min(other) FROM t WHERE {cond} GROUP BY g ORDER BY g"
    res = check(mx, sql, con, files={"t": path}, ordered=True, extra=("--device", "gpu"))
    assert res["device"] == "gpu", "\n".join(res["notes"])


@pytest.mark.parametrize("cond", ["qty > 3", "NOT (qty > 3 AND price < 0)", "qty IS NULL OR other > 2"])
def test_a_case_with_a_nullable_condition_and_no_else_equals_duckdb(mx, data, has_gpu, cond):
    path, con = data
    sql = f"SELECT g, count(CASE WHEN {cond} THEN 1 END), sum(CASE WHEN {cond} THEN price END), max(CASE WHEN {cond} THEN qty ELSE 0 END) FROM t GROUP BY g ORDER BY g"
    for dev in ("cpu", "gpu") if has_gpu else ("cpu",):
        res = check(mx, sql, con, files={"t": path}, ordered=True, extra=("--device", dev))
        assert res["device"] == dev, "\n".join(res["notes"])


TEXT_DATE_CONDITIONS = [
    "s = 'a'",
    "s = 'a' OR s IS NULL",
    "NOT (s = 'b') AND qty > 0",
    "s < 'c' OR d > DATE '2024-05-01'",
    "NOT (d >= DATE '2024-03-01' AND s <> 'e')",
    "d IS NULL OR (s IS NOT NULL AND d < DATE '2024-06-15')",
]


@pytest.mark.parametrize("cond", TEXT_DATE_CONDITIONS)
def test_a_where_over_nullable_text_and_date_columns_equals_duckdb(mx, data, has_gpu, cond):
    """Text and date comparisons take the same known-true / known-false lowering; the answer equals DuckDB on each device (the device that ran it is
    not asserted: a text compare may stay on the CPU)."""
    path, con = data
    sql = f"SELECT g, count(*), sum(price), min(qty) FROM t WHERE {cond} GROUP BY g ORDER BY g"
    assert con.execute(f"SELECT count(*) FROM t WHERE {cond}").fetchone()[0] not in (0, 400), "the condition keeps some rows and drops others"
    for dev in ("cpu", "gpu") if has_gpu else ("cpu",):
        check(mx, sql, con, files={"t": path}, ordered=True, extra=("--device", dev))
