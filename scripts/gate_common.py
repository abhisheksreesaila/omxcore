"""F-162: what the M8 gate shares between its tests (tests/test_mx_tpch_gate.py, tests/test_mx_random_gate.py) and its scoreboard
(scripts/gate_scoreboard.py): typed TPC-H Parquet, the 22 queries as DuckDB writes them, the random suite as SQL, the DuckDB oracle.

Data: DuckDB's own dbgen (the `tpch` extension), written once per scale factor to ~/.cache/omxcore-gate/tpch_sf<SF>/<table>.parquet
with the types TPC-H says (DECIMAL(15,2), DATE, BIGINT / INTEGER, VARCHAR). `OMXCORE_GATE_DATA` moves the folder. Never /tmp.
The random suite's filter constants come from the quantiles of these same files (scripts/bench_random/gen.py shapes the queries).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE_DATA = Path(os.environ.get("OMXCORE_GATE_DATA", Path.home() / ".cache" / "omxcore-gate"))
TABLES = ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]
DATE_COLUMNS = {"l_shipdate", "l_commitdate", "l_receiptdate", "o_orderdate"}


def _fmt(sf) -> str:
    return str(sf)


def data_dir(sf) -> Path:
    return GATE_DATA / f"tpch_sf{_fmt(sf)}"


def ensure_tpch(sf) -> Path:
    """The folder of typed TPC-H Parquet for scale factor `sf`, generated with DuckDB's dbgen when it is not there yet."""
    d = data_dir(sf)
    if all((d / f"{t}.parquet").exists() for t in TABLES):
        return d
    import duckdb
    tmp = d.with_name(d.name + f".partial{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    tmpdb = tmp / "dbgen.duckdb"                       # on disk, beside the output: SF10 does not fit a memory database
    con = duckdb.connect(str(tmpdb))
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={sf})")
    for t in TABLES:
        con.execute(f"COPY {t} TO '{tmp / (t + '.parquet')}' (FORMAT parquet)")
    con.close()
    tmpdb.unlink()
    try:
        tmp.rename(d)
    except OSError:                                    # another worker finished first
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()
    return d


def table_paths(sf) -> dict:
    d = ensure_tpch(sf)
    return {t: str(d / f"{t}.parquet") for t in TABLES}


def queries() -> dict:
    """{1..22: SQL text} as DuckDB's tpch extension writes them (without the trailing semicolon)."""
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    return {n: q.strip().rstrip(";") for n, q in con.execute("SELECT query_nr, query FROM tpch_queries() ORDER BY 1").fetchall()}


def duck_views(paths: dict):
    """A DuckDB connection with one view per table over the Parquet files (the same files `mx -t name=path` reads)."""
    import duckdb
    con = duckdb.connect()
    for name, path in paths.items():
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
    return con


# ── the random suite as SQL over the typed data ────────────────────────────────────────────────────

def _random_gen():
    sys.path.insert(0, str(ROOT / "scripts" / "bench_random"))
    import gen
    return gen


def random_stats(sf=1):
    """Quantiles of the numeric filter columns and distinct values of the text ones, from the typed files (the shape gen.stats gives)."""
    import numpy as np
    gen = _random_gen()
    paths = table_paths(sf)
    con = duck_views(paths)
    qs = [round(float(q), 6) for q in np.linspace(0.02, 0.98, 25)]
    out = {}
    for t, cols in gen.FILTERS.items():
        for c in cols:
            if c in gen.STRING_COLS:
                out[c] = sorted(r[0] for r in con.execute(f"SELECT DISTINCT {c} FROM {t}").fetchall())
            else:
                x = f"date_diff('day', DATE '1970-01-01', {c})" if c in DATE_COLUMNS else f"CAST({c} AS DOUBLE)"
                out[c] = [float(r[0]) for r in con.execute(f"SELECT unnest(quantile_cont({x}, {qs}::DOUBLE[])) FROM {t}").fetchall()]
    return out


def random_sql(seed: int, stats: dict) -> str:
    """The seed's random query as SQL over the typed tables: a date filter's day number becomes a DATE literal."""
    import datetime
    gen = _random_gen()
    spec = gen.make(seed, stats)
    epoch = datetime.date(1970, 1, 1)
    real = gen._sql_filter

    def typed(f):
        if f["col"] in DATE_COLUMNS:
            day = lambda n: f"DATE '{epoch + datetime.timedelta(days=int(n))}'"
            c, op, v = f["col"], f["op"], f["v"]
            if op == "between":
                return f"{c} >= {day(v[0])} AND {c} <= {day(v[1])}"
            return f"{c} {dict(eq='=', ne='<>', lt='<', ge='>=')[op]} {day(v)}"
        return real(f)

    gen._sql_filter = typed
    try:
        return gen.sql(spec)
    finally:
        gen._sql_filter = real


def random_tables(sql_text: str) -> list:
    """The tables a generated query names (so a run registers only those)."""
    import re
    return [t for t in TABLES if re.search(rf"\b{t}\b", sql_text)]
