"""F-191: `mx sql --device cpu|gpu|auto` and the JSON `device` field. DuckDB is the oracle for the values; the notes and `device` say which device ran.

AUTO (the default) answers on the CPU unless the table has `--gpu-min-rows` rows (1,000,000 by default); `--device cpu` never touches the GPU;
`--device gpu` runs the single-table integer / DECIMAL aggregates the GPU programs run and says why anything else ran on the CPU. The tables are tiny
(the GPUs are shared): the threshold is lowered with `--gpu-min-rows 0` instead of building a big table. The GPU checks skip when the machine has no GPU.
"""
import duckdb
import pytest

from mx_oracle import check, mx_binary, run_mx


@pytest.fixture(scope="module")
def mx():
    return mx_binary()


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    path = tmp_path_factory.mktemp("f191") / "t.csv"
    rows = ["region,qty,price,f"]
    s = 12345
    for i in range(400):
        s = (s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        v = (s >> 33) % 1000
        qty = "" if i % 13 == 4 else str(v % 11 - 3)
        rows.append(f"{['west', 'east', 'north', 'south'][(s >> 20) % 4]},{qty},{v * 7 - 2000},{v / 3 + 0.123456789012}")
    path.write_text("\n".join(rows) + "\n")
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t AS SELECT * FROM read_csv('{path}')")
    return path, con


AGG = "SELECT region, sum(qty), count(*), min(price), max(price) FROM t WHERE qty > 1 GROUP BY region ORDER BY region"


def notes_of(res):
    return "\n".join(res["notes"])


@pytest.fixture(scope="module")
def has_gpu(mx, data):
    res = run_mx(mx, ["sql", "--device", "gpu", "SELECT sum(price) FROM t"], files={"t": data[0]})
    return res["device"] == "gpu"


def test_default_is_auto_and_a_small_table_runs_on_the_cpu_saying_why(mx, data):
    path, con = data
    res = check(mx, AGG, con, files={"t": path}, ordered=True)
    assert res["device"] == "cpu"
    assert "device: CPU (AUTO: the table has 400 rows, under the 1000000 AUTO runs on the GPU)" in res["notes"]


def test_device_cpu_never_touches_the_gpu(mx, data):
    path, con = data
    res = check(mx, AGG, con, files={"t": path}, ordered=True, extra=("--device", "cpu", "--gpu-min-rows", "0"))
    assert res["device"] == "cpu"
    assert "device: CPU (device CPU was asked for)" in res["notes"]


def test_device_gpu_runs_the_aggregates_and_equals_duckdb(mx, data, has_gpu):
    if not has_gpu:
        pytest.skip("no GPU on this machine")
    path, con = data
    for sql in [AGG,
                "SELECT sum(price * qty), count(qty), count(*) FROM t",
                "SELECT region, sum(CASE WHEN qty > 4 THEN price ELSE 0 END), max(qty) FROM t GROUP BY region ORDER BY region",
                "SELECT qty, sum(price) FROM t WHERE price BETWEEN 0 AND 3000 AND region <> 'west' GROUP BY qty ORDER BY qty"]:
        res = check(mx, sql, con, files={"t": path}, ordered=True, extra=("--device", "gpu"))
        assert res["device"] == "gpu", notes_of(res)
        assert "device: GPU (GPU asked for, the table uploaded for this query)" in res["notes"]


def test_auto_goes_to_the_gpu_at_its_row_threshold(mx, data, has_gpu):
    if not has_gpu:
        pytest.skip("no GPU on this machine")
    path, con = data
    on = check(mx, AGG, con, files={"t": path}, ordered=True, extra=("--gpu-min-rows", "400"))
    off = check(mx, AGG, con, files={"t": path}, ordered=True, extra=("--gpu-min-rows", "401"))
    assert (on["device"], off["device"]) == ("gpu", "cpu"), notes_of(on) + notes_of(off)


def test_what_the_gpu_programs_do_not_run_stays_on_the_cpu_with_the_reason(mx, data):
    path, con = data
    res = check(mx, "SELECT region, max(f) FROM t GROUP BY region ORDER BY region", con, files={"t": path}, ordered=True, extra=("--device", "gpu"))
    assert res["device"] == "cpu"
    assert any(n.startswith("device: CPU (GPU asked for; a FLOAT") for n in res["notes"]), notes_of(res)
    res = check(mx, AGG, con, files={"t": path}, ordered=True, extra=("--device", "gpu", "--group", "hashed"))
    assert res["device"] == "cpu"
    assert any(n.startswith("device: CPU (GPU asked for; hashed groups") for n in res["notes"]), notes_of(res)
