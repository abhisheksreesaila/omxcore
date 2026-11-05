"""F-160: `COPY (query) TO 'file.parquet'` through the real `mx sql`: the file holds the rows DuckDB gives for the query (kinds, scales and values), the answer is
the one row `Count`, the options reach the writer (codec, row group size), and what it cannot write is refused by name.
"""
import pyarrow.parquet as pq
import pytest

from conftest import oracle_connection
from mx_oracle import assert_equal, duck_kind, mx_binary, run_mx


@pytest.fixture(scope="module")
def mx():
    return mx_binary()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("copy")
    con = oracle_connection()
    con.execute("""create table t as select i::BIGINT as id,
        case when i % 11 = 0 then null else (i % 7)::BIGINT end as k,
        case when i % 13 = 0 then null else ((i * 37) % 50 - 10)::BIGINT end as v,
        case when i % 17 = 0 then null else (['West', 'east', 'North', 'south', 'Apple'])[1 + (i % 5)] end as s,
        case when i % 19 = 0 then null else ((i * 7) % 1000 - 300) * 0.01 end::DECIMAL(10,2) as d,
        date '2020-01-01' + (i % 400)::INTEGER as day from range(300) r(i)""")
    con.execute("create table u as select i::BIGINT as k, (['a (b', 'c)', 'd'])[1 + (i % 3)] as name from range(7) r(i)")
    path = d / "t.parquet"
    con.execute(f"copy t to '{path}' (format parquet, row_group_size 64)")
    upath = d / "u.parquet"
    con.execute(f"copy u to '{upath}' (format parquet)")
    return {"con": con, "dir": d, "t": path, "u": upath}


def copy_equals_duckdb(mx, world, name, query, extra_files=None, ordered=True, options=""):
    out = world["dir"] / f"{name}.parquet"
    out.unlink(missing_ok=True)
    files = {"t": world["t"], "u": world["u"], **(extra_files or {})}
    res = run_mx(mx, ["sql", f"COPY ({query}) TO '{out}'{options}"], files)
    con = world["con"]
    cur = con.execute(query)
    want = cur.fetchall()
    assert res["columns"][0]["name"] == "Count" and res["rows"] == [[len(want)]], res
    back = oracle_connection()
    got_cur = back.execute(f"select * from read_parquet('{out}')")
    got = got_cur.fetchall()
    for a, b in zip(got_cur.description, cur.description):
        assert a[0] == b[0] and duck_kind(a[1]) == duck_kind(b[1]), f"{name}: column {a} vs DuckDB's {b}"
    assert_equal(got, want, ordered, name)
    return out


def test_filter_and_projection_with_every_kind(mx, world):
    copy_equals_duckdb(mx, world, "plain", "SELECT id, v, s, d, day FROM t WHERE id % 3 = 0 ORDER BY id")


def test_aggregate_with_order_by(mx, world):
    copy_equals_duckdb(mx, world, "agg", "SELECT k, count(*) AS n, sum(v) AS sv, min(d) AS lo FROM t GROUP BY k ORDER BY k NULLS LAST")


def test_cte_join_and_a_file_named_in_from(mx, world):
    q = ("WITH big AS (SELECT id, k, v FROM t WHERE v > 20) "
         "SELECT big.id, u.name, big.v FROM big JOIN u ON big.k = u.k ORDER BY big.id")
    copy_equals_duckdb(mx, world, "cte", q)
    f = world["t"]
    out = world["dir"] / "fromfile.parquet"
    res = run_mx(mx, ["sql", f"COPY (SELECT count(*) AS n, sum(v) AS sv FROM '{f}' WHERE id < 100) TO '{out}'"])
    assert res["rows"] == [[1]]
    assert oracle_connection().execute(f"select n, sv from read_parquet('{out}')").fetchall() == world["con"].execute("select count(*), sum(v) from t where id < 100").fetchall()


def test_parentheses_and_quotes_inside_the_query(mx, world):
    q = "SELECT id, name FROM t JOIN u ON t.k = u.k WHERE name <> ')' AND name <> 'it''s (' ORDER BY id LIMIT 40"
    copy_equals_duckdb(mx, world, "quotes", q)


def test_options_reach_the_writer(mx, world):
    out = copy_equals_duckdb(mx, world, "opts", "SELECT id, s FROM t ORDER BY id", options=" (FORMAT PARQUET, COMPRESSION snappy, ROW_GROUP_SIZE 64);")
    md = pq.ParquetFile(out).metadata
    assert md.num_rows == 300 and md.num_row_groups == 5
    assert md.row_group(0).column(0).compression == "SNAPPY"
    out = copy_equals_duckdb(mx, world, "dflt", "SELECT id FROM t ORDER BY id")
    assert pq.ParquetFile(out).metadata.row_group(0).column(0).compression == "ZSTD"


def test_an_empty_result_writes_an_empty_file(mx, world):
    out = world["dir"] / "empty.parquet"
    res = run_mx(mx, ["sql", f"COPY (SELECT id, s FROM t WHERE id < 0) TO '{out}'"], {"t": world["t"]})
    assert res["rows"] == [[0]]
    assert oracle_connection().execute(f"select count(*) from read_parquet('{out}')").fetchall() == [(0,)]


@pytest.mark.parametrize("sql,message", [
    ("COPY (SELECT id FROM t) TO '{d}/x.csv'", "COPY writes Parquet only"),
    ("COPY (SELECT id FROM t) TO '{d}/x.parquet' (FORMAT CSV)", "FORMAT CSV is not supported"),
    ("COPY (SELECT id FROM t) TO '{d}/x.parquet' (HEADER)", "needs a value"),
    ("COPY (SELECT id FROM t) TO '{d}/x.parquet' (DELIMITER ',')", "the COPY option DELIMITER is not supported"),
    ("COPY (SELECT id FROM t) TO '{d}/x.parquet' (ROW_GROUP_SIZE 7)", "multiple of 8"),
    ("COPY SELECT id FROM t TO '{d}/x.parquet'", "COPY takes a query in parentheses"),
    ("COPY (SELECT id FROM t) '{d}/x.parquet'", "needs TO"),
    ("COPY (SELECT nope FROM t) TO '{d}/x.parquet'", "nope"),
    ("COPY (SELECT id FROM t) TO '{d}/no/such/dir/x.parquet'", "no/such/dir"),
])
def test_refusals_name_the_problem_and_write_nothing(mx, world, sql, message):
    d = world["dir"]
    res = run_mx(mx, ["sql", sql.format(d=d)], {"t": world["t"]}, expect_exit=1)
    assert message in res["error"]["message"], res
    assert not (d / "x.parquet").exists() and not (d / "x.csv").exists()
