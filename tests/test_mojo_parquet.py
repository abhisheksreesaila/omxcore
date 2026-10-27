"""F-135 (M4): Parquet written by the Python-free Mojo package reads back in pyarrow and DuckDB with every value,
type, null and the "omx" metadata, for each codec and several row-group sizes."""
import json
import os
import struct
import subprocess
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parent.parent
WORDS = ["West", "East", "Nörth", "", "South-East"]


@pytest.fixture(scope="module")
def writer(tmp_path_factory):
    out = tmp_path_factory.mktemp("mojo") / "pq_write_sample"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_write_sample.mojo"), str(out)],
                   check=True, capture_output=True, text=True)
    return out


def expected(n):
    rows = []
    for i in range(n):
        rev = None if i % 13 == 4 else (i * 1_000_003 + (1 << 53) if i % 97 == 0 else i * 101 - 50_000)
        rows.append({
            "qty": i * 7 - 3000,
            "revenue": rev,
            "region": None if i % 17 == 2 else WORDS[i % 5],
            "day": 18_000 + i % 400,
            "paid": None if i % 11 == 6 else (i % 3 == 0),
        })
    return rows


@pytest.mark.parametrize("codec", ["zstd", "snappy", "none"])
@pytest.mark.parametrize("n,group", [(0, 1024), (1, 8), (1000, 1024), (5000, 1024)])
def test_pyarrow_and_duckdb_read_what_mojo_wrote(writer, tmp_path, codec, n, group):
    path = tmp_path / "t.parquet"
    subprocess.run([str(writer), str(path), codec, str(n), str(group)], check=True)
    t = pq.read_table(path)
    md = pq.read_metadata(path)
    assert md.num_row_groups == max(1, -(-n // group))
    assert json.loads(t.schema.metadata[b"omx"]) == {"formats": {"revenue": "money"}}
    assert t.schema.field("qty").type == pa.int64()
    assert t.schema.field("revenue").type == pa.decimal128(18, 2)
    assert t.schema.field("region").type == pa.string()
    assert t.schema.field("day").type == pa.date32()
    assert t.schema.field("paid").type == pa.bool_()
    want = expected(n)
    got = t.to_pylist()
    for g, w in zip(got, want):
        assert g["qty"] == w["qty"] and g["region"] == w["region"] and g["paid"] == w["paid"]
        assert (g["revenue"] is None) == (w["revenue"] is None)
        if w["revenue"] is not None:
            assert int(g["revenue"].scaleb(2)) == w["revenue"]          # the exact scaled integer
        assert (g["day"] - __import__("datetime").date(1970, 1, 1)).days == w["day"]
    assert len(got) == n
    con = duckdb.connect()
    s = con.execute(f"SELECT count(*), sum(qty), count(revenue), count(region), sum(CASE WHEN paid THEN 1 ELSE 0 END) "
                    f"FROM read_parquet('{path}')").fetchone()
    assert s[0] == n
    assert (s[1] or 0) == sum(w["qty"] for w in want)
    assert s[2] == sum(w["revenue"] is not None for w in want)
    assert s[3] == sum(w["region"] is not None for w in want)
    assert (s[4] or 0) == sum(bool(w["paid"]) for w in want)


# ── reading files other writers made ─────────────────────────────────────────────────────────────────

import datetime as _dt
import decimal as _dec
import random as _random


@pytest.fixture(scope="module")
def dumper(tmp_path_factory):
    out = tmp_path_factory.mktemp("mojo") / "pq_dump"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_dump.mojo"), str(out)],
                   check=True, capture_output=True, text=True)
    return out


def mojo_read(dumper, path):
    """(columns [(name, kind, scale)], rows of stored values with TEXT as the word)."""
    out = subprocess.run([str(dumper), str(path)], check=True, capture_output=True).stdout.decode("utf-8")
    lines = out.split("\n")
    head = json.loads(lines[0])
    words = {}
    rows = []
    for line in lines[1:]:
        if line.startswith("WORDS "):
            _, c, rest = line.split(" ", 2)
            words[int(c)] = rest.split("\x1f")[:-1]
        elif line:
            rows.append(json.loads(line))
    for r in rows:
        for c, v in enumerate(r):
            if isinstance(v, dict):
                r[c] = words[c][v["word"]]
    return [tuple(c) for c in head["columns"]], rows


def stored(v, t):
    """A pyarrow value as omxcore stores it."""
    if v is None:
        return None
    if pa.types.is_decimal(t):
        return int(v.scaleb(t.scale))
    if pa.types.is_date(t):
        return (v - _dt.date(1970, 1, 1)).days
    if pa.types.is_timestamp(t):
        return (v.replace(tzinfo=None) - _dt.datetime(1970, 1, 1)) // _dt.timedelta(days=1)
    if pa.types.is_boolean(t):
        return int(v)
    return v


def sample(n, seed=3):
    r = _random.Random(seed)
    def maybe(v, k):
        return None if r.random() < k else v
    return pa.table({
        "i64": pa.array([maybe(r.randint(-2**62, 2**62), 0.1) for _ in range(n)], pa.int64()),
        "i32": pa.array([r.randint(-2**31, 2**31 - 1) for _ in range(n)], pa.int32()),
        "dec2": pa.array([maybe(_dec.Decimal(r.randint(-10**9, 10**9)).scaleb(-2), 0.2) for _ in range(n)], pa.decimal128(12, 2)),
        "dec4": pa.array([_dec.Decimal(r.randint(-10**17, 10**17)).scaleb(-4) for _ in range(n)], pa.decimal128(18, 4)),
        "d": pa.array([maybe(_dt.date(1960, 1, 1) + _dt.timedelta(days=r.randint(0, 40000)), 0.05) for _ in range(n)], pa.date32()),
        "ts_us": pa.array([_dt.datetime(1950, 1, 1) + _dt.timedelta(seconds=r.randint(0, 3 * 10**9)) for _ in range(n)], pa.timestamp("us")),
        "ts_ms": pa.array([_dt.datetime(2020, 1, 1) + _dt.timedelta(seconds=r.randint(-10**8, 10**8)) for _ in range(n)], pa.timestamp("ms")),
        "ts_ns": pa.array([_dt.datetime(2001, 1, 1) + _dt.timedelta(seconds=r.randint(0, 10**8)) for _ in range(n)], pa.timestamp("ns")),
        "b": pa.array([maybe(r.random() < 0.5, 0.1) for _ in range(n)], pa.bool_()),
        "s": pa.array([maybe(r.choice(["West", "East", "", "Ünïcode ✓", "x" * 300]), 0.1) for _ in range(n)], pa.string()),
        "s_many": pa.array([f"key-{r.randint(0, n)}" for _ in range(n)], pa.string()),
    })


def check_equal(dumper, path, table):
    cols, rows = mojo_read(dumper, path)
    assert [c[0] for c in cols] == table.column_names
    want = table.to_pylist()
    assert len(rows) == len(want)
    for i, (g, w) in enumerate(zip(rows, want)):
        for c, name in enumerate(table.column_names):
            assert g[c] == stored(w[name], table.schema.field(name).type), (i, name, g[c], w[name])


@pytest.mark.parametrize("opts", [
    dict(compression="zstd"),
    dict(compression="snappy"),
    dict(compression="none"),
    dict(compression="zstd", use_dictionary=False),
    dict(compression="snappy", data_page_version="2.0"),
    dict(compression="zstd", data_page_version="2.0", use_dictionary=False),
    dict(compression="none", use_dictionary=False, column_encoding={"i64": "DELTA_BINARY_PACKED", "i32": "DELTA_BINARY_PACKED"}),
    dict(compression="zstd", row_group_size=777, data_page_size=4096),
])
def test_mojo_reads_what_pyarrow_wrote(dumper, tmp_path, opts):
    t = sample(5000)
    path = tmp_path / "p.parquet"
    pq.write_table(t, path, **opts)
    check_equal(dumper, path, t)


def test_mojo_reads_what_duckdb_wrote(dumper, tmp_path):
    t = sample(3000, seed=9)
    path = tmp_path / "d.parquet"
    con = duckdb.connect()
    con.register("t", t)
    con.execute(f"COPY t TO '{path}' (FORMAT parquet)")
    back = pq.read_table(path)            # DuckDB's own types for the same values (its timestamps, decimals)
    check_equal(dumper, path, back)


def test_round_trip_keeps_the_omx_metadata(writer, dumper, tmp_path):
    path = tmp_path / "rt.parquet"
    subprocess.run([str(writer), str(path), "zstd", "2000", "512"], check=True)
    cols, rows = mojo_read(dumper, path)
    assert cols == [("qty", "INT", 0), ("revenue", "DECIMAL", 2), ("region", "TEXT", 0), ("day", "DATE", 0), ("paid", "BOOL", 0)]
    want = expected(2000)
    for g, w in zip(rows, want):
        assert g == [w["qty"], w["revenue"], w["region"], w["day"], None if w["paid"] is None else int(w["paid"])]


def test_unsupported_columns_raise_readably(dumper, tmp_path):
    path = tmp_path / "f.parquet"
    pq.write_table(pa.table({"x": pa.array([1.5, 2.5])}), path)       # (DOUBLE is the FLOAT kind since F-157: tests/test_mojo_float_parquet.py)
    assert mojo_read(dumper, path)[0] == [("x", "FLOAT", 0)]
    pq.write_table(pa.table({"x": pa.array([1.5, 2.5], pa.float16())}), path)
    p = subprocess.run([str(dumper), str(path)], capture_output=True, text=True)
    assert p.returncode != 0 and "column `x`" in (p.stdout + p.stderr)


# ── readers' own corner cases: what other writers produce ────────────────────────────────────────────

I64 = [-2**63, 2**63 - 1, 0, -2**63, 2**63 - 1, 1, -1]
I32 = [-2**31, 2**31 - 1, 0, -2**31, 5, 2**31 - 1, 0]


def write_and_check(dumper, tmp_path, table, **opts):
    path = tmp_path / "w.parquet"
    pq.write_table(table, path, **opts)
    check_equal(dumper, path, table)


def test_small_and_unsigned_integers_keep_their_values(dumper, tmp_path):
    t = pa.table({
        "i8": pa.array([-128, 127, None], pa.int8()), "i16": pa.array([-32768, 32767, 7], pa.int16()),
        "u8": pa.array([255, 0, 7], pa.uint8()), "u16": pa.array([65535, 0, 7], pa.uint16()),
        "u32": pa.array([3_000_000_000, 2**32 - 1, None], pa.uint32()),
        "u64": pa.array([0, 2**63 - 1, 5], pa.uint64()),
    })
    for opts in (dict(), dict(use_dictionary=False), dict(data_page_version="2.0"),
                 dict(use_dictionary=False, column_encoding={"u32": "DELTA_BINARY_PACKED", "u64": "DELTA_BINARY_PACKED"})):
        write_and_check(dumper, tmp_path, t, **opts)


def test_uint64_past_int64_raises_naming_the_column(dumper, tmp_path):
    for opts in (dict(), dict(use_dictionary=False)):
        path = tmp_path / "u64.parquet"
        pq.write_table(pa.table({"big": pa.array([1, 2**63 + 5], pa.uint64())}), path, **opts)
        refuses(dumper, path, "column `big` is UINT_64 and holds 9223372036854775813, past the int64 range")


def test_text_columns_of_every_shape(dumper, tmp_path):
    for t in (
        pa.table({"s": pa.array([None] * 13, pa.string())}),               # all null: no words at all
        pa.table({"s": pa.array(["w"] * 13)}),                              # one word
        pa.table({"s": pa.array([""] * 11)}),                               # only empty strings
        pa.table({"s": pa.array(["x", None, "", "x"], pa.large_string())}),
        pa.table({"s": pa.array(["a", "b", "a", None]).dictionary_encode()}),
    ):
        for opts in (dict(), dict(use_dictionary=False), dict(data_page_version="2.0")):
            write_and_check(dumper, tmp_path, t, **opts)


def test_all_null_int_column(dumper, tmp_path):
    for opts in (dict(), dict(data_page_version="2.0"), dict(use_dictionary=False)):
        write_and_check(dumper, tmp_path, pa.table({"n": pa.array([None] * 13, pa.int64())}), **opts)


def test_delta_extremes_and_a_single_value(dumper, tmp_path):
    enc = {"i": "DELTA_BINARY_PACKED", "j": "DELTA_BINARY_PACKED"}
    t = pa.table({"i": pa.array(I64, pa.int64()), "j": pa.array(I32, pa.int32())})
    write_and_check(dumper, tmp_path, t, use_dictionary=False, column_encoding=enc)
    write_and_check(dumper, tmp_path, pa.table({"i": pa.array([42], pa.int64())}), use_dictionary=False,
                    column_encoding={"i": "DELTA_BINARY_PACKED"})
    write_and_check(dumper, tmp_path, pa.table({"i": pa.array(range(0, 33_000, 1000), pa.int64())}),
                    use_dictionary=False, column_encoding={"i": "DELTA_BINARY_PACKED"})
    write_and_check(dumper, tmp_path, pa.table({"i": pa.array(I64, pa.int64())}), data_page_version="2.0")


def test_v2_page_of_an_optional_column_without_nulls(dumper, tmp_path):
    t = pa.table({"i": pa.array([1, 2, 3], pa.int64()), "b": pa.array([True, False, True])})
    write_and_check(dumper, tmp_path, t, data_page_version="2.0", compression="zstd")


def test_file_with_a_page_index_and_small_pages(dumper, tmp_path):
    write_and_check(dumper, tmp_path, pa.table({"i": pa.array(range(100), pa.int64())}),
                    write_page_index=True, data_page_size=64, write_batch_size=10)
    t = pa.table({"i": pa.array([None if k % 3 == 0 else k for k in range(13)], pa.int64())})
    write_and_check(dumper, tmp_path, t, data_page_version="1.0", data_page_size=16, write_batch_size=5)


def test_timestamp_in_a_zone_keeps_its_utc_day(dumper, tmp_path):
    # 2020-01-01 23:30 UTC is 18,262 days after 1970; the zone (isAdjustedToUTC) is only a display hint
    t = pa.table({"t": pa.array([_dt.datetime(2020, 1, 1, 23, 30, tzinfo=_dt.timezone.utc)], pa.timestamp("us", tz="Asia/Tokyo"))})
    path = tmp_path / "tz.parquet"
    pq.write_table(t, path)
    assert mojo_read(dumper, path)[1] == [[18262]]


def refuses(dumper, path, *words):
    p = subprocess.run([str(dumper), str(path)], capture_output=True, text=True)
    assert p.returncode == 1, (p.returncode, p.stderr[-300:])       # a normal error exit, never a signal
    for w in words:
        assert w in p.stdout + p.stderr, (w, p.stdout + p.stderr)


def test_columns_without_a_kind_raise_readably(dumper, tmp_path):
    path = tmp_path / "x.parquet"
    pq.write_table(pa.table({"b": pa.array([b"\xff\xfe", b"ok"], pa.binary())}), path)
    refuses(dumper, path, "column `b` is BINARY (BYTE_ARRAY without a string type)")
    pq.write_table(pa.table({"t": pa.array([_dt.time(1, 2, 3)], pa.time32("ms"))}), path)
    refuses(dumper, path, "column `t` is a TIME")
    pq.write_table(pa.table({"t": pa.array([_dt.time(1, 2, 3)], pa.time64("us"))}), path)
    refuses(dumper, path, "column `t` is a TIME")
    pq.write_table(pa.table({"d": pa.array([_dec.Decimal("1.5")], pa.decimal128(30, 1))}), path)
    refuses(dumper, path, "column `d`: DECIMAL(30, 1) has more digits")


def test_invalid_utf8_in_a_text_column_raises(dumper, tmp_path):
    bad = pa.array([b"ok", b"\xff\xfe"], pa.binary()).view(pa.string())      # a string column holding invalid UTF-8
    for opts in (dict(), dict(use_dictionary=False)):
        path = tmp_path / "u.parquet"
        pq.write_table(pa.table({"s": bad}), path, **opts)
        refuses(dumper, path, "column `s`", "not valid UTF-8")


def test_definition_levels_may_be_bit_packed(dumper, tmp_path):
    # the deprecated BIT_PACKED level encoding: most significant bit first, no length prefix
    body = bytes([0b10100000]) + struct.pack("<qq", 5, 9)                    # levels 1,0,1 (3 rows): values 5, 9
    path = tmp_path / "bp.parquet"
    evil_build(path, body, 3, len(body), len(body), rep=1, def_enc=4)
    assert mojo_read(dumper, path)[1] == [[5], [None], [9]]


# ── corrupt and hostile files: a readable error and a normal exit, never a crash ─────────────────────



def uv(v):
    o = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            o.append(b | 0x80)
        else:
            o.append(b)
            return bytes(o)


def zz(v):
    return uv(((v << 1) ^ (v >> 63)) & ((1 << 64) - 1))


class Compact:
    """A thrift compact-protocol writer, just enough to hand-make Parquet metadata."""
    def __init__(s):
        s.b = bytearray()
        s.last = [0]

    def f(s, fid, t):
        d = fid - s.last[-1]
        if 0 < d <= 15:
            s.b.append((d << 4) | t)
        else:
            s.b.append(t)
            s.b += zz(fid)
        s.last[-1] = fid

    def i32(s, fid, v):
        s.f(fid, 5)
        s.b += zz(v)

    def i64(s, fid, v):
        s.f(fid, 6)
        s.b += zz(v)

    def bin(s, fid, v, rawlen=None):
        s.f(fid, 8)
        s.b += uv(len(v) if rawlen is None else rawlen)
        s.b += v

    def st(s, fid):
        s.f(fid, 12)
        s.last.append(0)

    def lst(s, fid, et, n):
        s.f(fid, 9)
        s.b += bytes([(n << 4) | et]) if n < 15 else bytes([0xF0 | et]) + uv(n)

    def el(s):
        s.last.append(0)

    def end(s):
        s.b.append(0)
        s.last.pop()


def evil_build(path, body, nvals, usize, csize, ptype=2, rep=0, type_length=None, decimal=False, enc=0,
               name=b"x", namelen=None, footer_extra=None, def_enc=3, v2=None, chunk_values=None):
    """One column, one row group, one data page: `body` is the page, the sizes and counts are whatever we say."""
    h = Compact()
    h.i32(1, 3 if v2 else 0)
    h.i32(2, usize)
    h.i32(3, csize)
    if v2:
        h.st(8)
        h.i32(1, nvals)
        h.i32(2, 0)
        h.i32(3, nvals)
        h.i32(4, enc)
        h.i32(5, v2["def_len"])
        h.i32(6, 0)
        h.end()
    else:
        h.st(5)
        h.i32(1, nvals)
        h.i32(2, enc)
        h.i32(3, def_enc)
        h.i32(4, 3)
        h.end()
    h.b.append(0)
    data = b"PAR1"
    off = len(data)
    data += bytes(h.b) + body
    m = Compact()
    m.i32(1, 1)
    m.lst(2, 12, 2)
    m.el(); m.bin(4, b"schema"); m.i32(5, 1); m.end()
    m.el(); m.i32(1, ptype)
    if type_length is not None:
        m.i32(2, type_length)
    m.i32(3, rep)
    m.bin(4, name, namelen)
    if decimal:
        m.i32(6, 5); m.i32(7, 2); m.i32(8, 18)
    m.end()
    m.i64(3, nvals)
    m.lst(4, 12, 1)
    m.el(); m.lst(1, 12, 1)
    m.el(); m.i64(2, off)
    m.st(3); m.i32(1, ptype); m.lst(2, 5, 1); m.b += zz(0); m.lst(3, 8, 1); m.b += uv(1) + b"x"
    m.i32(4, 0); m.i64(5, nvals if chunk_values is None else chunk_values)
    m.i64(6, len(body)); m.i64(7, len(body)); m.i64(9, off); m.end(); m.end()
    m.i64(2, len(body)); m.i64(3, nvals); m.end()
    if footer_extra:
        footer_extra(m)
    m.b.append(0)
    data += bytes(m.b) + struct.pack("<i", len(m.b)) + b"PAR1"
    Path(path).write_bytes(data)


OK3 = struct.pack("<qqq", 1, 2, 3)
RUN_2_40 = uv((1 << 40) << 1) + b"\x01"                                       # an RLE run of 2^40 ones


def unknown_field(kind):
    def add(m):
        if kind == "binary_len_max":                                           # a skipped binary of 2^64-1 bytes
            m.f(99, 8); m.b += uv((1 << 64) - 1)
        elif kind == "list_size_huge":
            m.f(99, 9); m.b += bytes([0xF5]) + uv(1 << 40)
        elif kind == "binary_past_end":
            m.f(99, 8); m.b += uv(1 << 30)
        elif kind == "struct_bomb":                                            # 100,000 nested structs
            m.f(99, 12); m.b += bytes([0x1C]) * 100_000
        elif kind == "double_past_end":
            m.f(99, 7)
    return add


CORRUPT = {
    "uncompressed page claims more bytes than it has": dict(body=OK3, nvals=3, usize=200_000_000, csize=24),
    "uncompressed sizes disagree": dict(body=OK3, nvals=3, usize=16, csize=24),
    "negative compressed size": dict(body=OK3, nvals=3, usize=24, csize=-5),
    "negative uncompressed size": dict(body=OK3, nvals=3, usize=-24, csize=24),
    "column name length 2^64-1": dict(body=OK3, nvals=3, usize=24, csize=24, namelen=(1 << 64) - 1),
    "v1 level length past the page": dict(body=struct.pack("<I", 0x7FFFFFF0) + b"\x03\x01" + OK3, nvals=3,
                                          usize=30, csize=30, rep=1),
    "v2 level length past the page": dict(body=b"\x03\x01" + OK3, nvals=3, usize=26, csize=26, rep=1,
                                          v2=dict(def_len=2_000_000_000)),
    "v2 negative level length": dict(body=OK3, nvals=3, usize=24, csize=24, rep=1, v2=dict(def_len=-4)),
    "dictionary index run longer than the page": dict(body=b"\x08" + uv(((1 << 57) << 1) | 1) + b"\x00" * 8,
                                                      nvals=3, usize=17, csize=17, enc=8),
    "dictionary index width 65": dict(body=b"\x41" + uv(3 << 1) + b"\x00" * 9, nvals=3, usize=11, csize=11, enc=8),
    "fixed-length decimal short of its values": dict(body=struct.pack(">q", 7), nvals=1000, usize=8, csize=8,
                                                     ptype=7, type_length=8, decimal=True),
    "fixed-length decimal with length zero": dict(body=struct.pack(">q", 7), nvals=1, usize=8, csize=8,
                                                  ptype=7, type_length=0, decimal=True),
    "delta total of 2^40": dict(body=uv(128) + uv(4) + uv(1 << 40) + zz(0) + zz(0) + b"\x00" * 4, nvals=3,
                                usize=12, csize=12, enc=5),
    "delta with no miniblocks": dict(body=uv(128) + uv(0) + uv(3) + zz(0) + zz(0), nvals=3, usize=6, csize=6, enc=5),
    "delta miniblock width 65": dict(body=uv(128) + uv(4) + uv(3) + zz(0) + zz(0) + bytes([65, 0, 0, 0]) + b"\xff" * 300,
                                     nvals=3, usize=312, csize=312, enc=5),
    "more values in the page than in the column chunk": dict(body=OK3, nvals=3, usize=24, csize=24, chunk_values=2),
    "column name that is not UTF-8": dict(body=OK3, nvals=3, usize=24, csize=24, name=b"\xff\xfe"),
    "unknown physical type 9": dict(body=OK3, nvals=3, usize=24, csize=24, ptype=9),
    "no physical type": dict(body=OK3, nvals=3, usize=24, csize=24, ptype=-1),
    "skipped binary of 2^64-1 bytes": dict(body=OK3, nvals=3, usize=24, csize=24, footer_extra=unknown_field("binary_len_max")),
    "skipped list of 2^40 elements": dict(body=OK3, nvals=3, usize=24, csize=24, footer_extra=unknown_field("list_size_huge")),
    "skipped binary past the footer": dict(body=OK3, nvals=3, usize=24, csize=24, footer_extra=unknown_field("binary_past_end")),
    "skipped double past the footer": dict(body=OK3, nvals=3, usize=24, csize=24, footer_extra=unknown_field("double_past_end")),
    "100,000 nested structs": dict(body=OK3, nvals=3, usize=24, csize=24, footer_extra=unknown_field("struct_bomb")),
}


@pytest.mark.parametrize("what", sorted(CORRUPT))
def test_corrupt_file_raises_instead_of_crashing(dumper, tmp_path, what):
    path = tmp_path / "evil.parquet"
    evil_build(path, **CORRUPT[what])
    refuses(dumper, path, "corrupt")


def test_runs_longer_than_the_values_still_needed_are_cut_not_allocated(dumper, tmp_path):
    # definition levels as one RLE run of 2^40 ones, three rows: reads as three present values
    body = struct.pack("<I", len(RUN_2_40)) + RUN_2_40 + OK3
    path = tmp_path / "run.parquet"
    evil_build(path, body, 3, len(body), len(body), rep=1)
    assert mojo_read(dumper, path)[1] == [[1], [2], [3]]


def test_truncated_files_raise(dumper, tmp_path):
    good = tmp_path / "good.parquet"
    pq.write_table(pa.table({"i": pa.array(range(50), pa.int64())}), good)
    raw = good.read_bytes()
    for cut in (0, 3, 11, 12, len(raw) // 2, len(raw) - 9, len(raw) - 1):
        path = tmp_path / f"cut{cut}.parquet"
        path.write_bytes(raw[:cut])
        refuses(dumper, path)


def test_random_byte_flips_never_crash_the_reader(dumper, tmp_path):
    good = tmp_path / "good.parquet"
    t = sample(60, seed=5).select(["i64", "dec2", "d", "b", "s"])
    pq.write_table(t, good, compression="snappy")
    raw = bytearray(good.read_bytes())
    r = _random.Random(11)
    for _ in range(150):
        b = bytearray(raw)
        for _ in range(r.randint(1, 3)):
            b[r.randrange(len(b))] = r.randrange(256)
        path = tmp_path / "flip.parquet"
        path.write_bytes(bytes(b))
        p = subprocess.run([str(dumper), str(path)], capture_output=True, timeout=60)
        assert p.returncode in (0, 1), (p.returncode, p.stderr[-300:])      # read it or refused it; no signal


# ── writing: refusals, empty dictionaries, row-group sizes ───────────────────────────────────────────

@pytest.fixture(scope="module")
def edge_writer(tmp_path_factory):
    out = tmp_path_factory.mktemp("mojo") / "pq_write_edge"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_write_edge.mojo"), str(out)],
                   check=True, capture_output=True, text=True)
    return out


def test_decimal_past_18_digits_is_refused_naming_the_column(edge_writer, tmp_path):
    path = tmp_path / "d.parquet"
    p = subprocess.run([str(edge_writer), "decimal_overflow", str(path)], capture_output=True, text=True)
    assert p.returncode == 1
    assert "column `money`: 1000000000000000000 does not fit DECIMAL(18, 2)" in p.stdout + p.stderr
    assert not path.exists()


def test_table_without_columns_is_refused(edge_writer, tmp_path):
    path = tmp_path / "none.parquet"
    p = subprocess.run([str(edge_writer), "no_columns", str(path)], capture_output=True, text=True)
    assert p.returncode == 1
    assert "a table with no columns cannot be written as Parquet" in p.stdout + p.stderr
    assert not path.exists()


def test_all_null_text_without_words_in_row_groups_of_8(edge_writer, tmp_path):
    path = tmp_path / "n.parquet"
    subprocess.run([str(edge_writer), "null_text", str(path)], check=True)
    t = pq.read_table(path)
    assert pq.read_metadata(path).num_row_groups == 2
    assert t.column("id").to_pylist() == list(range(13))
    assert t.column("name").to_pylist() == [None] * 13
    assert t.schema.field("name").type == pa.string()
    con = duckdb.connect()
    assert con.execute(f"SELECT count(*), count(name) FROM read_parquet('{path}')").fetchone() == (13, 0)


def test_a_refused_write_leaves_the_existing_file_and_no_temp_file(edge_writer, tmp_path):
    path = tmp_path / "keep.omx"
    p = subprocess.run([str(edge_writer), "atomic", str(path)], capture_output=True, text=True, check=True)
    assert "refused: column `money`" in p.stdout
    assert pq.read_table(path).column("qty").to_pylist() == [41, 42]
    assert sorted(x.name for x in tmp_path.iterdir()) == ["keep.omx"]  # no temp file left, whatever its name


def test_footer_names_the_writer_and_uncompressed_row_group_size(writer, tmp_path):
    path = tmp_path / "m.parquet"
    subprocess.run([str(writer), str(path), "zstd", "3000", "1024"], check=True)
    md = pq.read_metadata(path)
    assert md.created_by == "omxcore version 0.6.0"
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        assert rg.total_byte_size == sum(rg.column(c).total_uncompressed_size for c in range(rg.num_columns))


# ── F-140: a query streamed over row groups equals read_parquet + the same query ─────────────────────

SCAN_WORDS = ["West", "East", "Nörth", "", "South-East", "Far", "Central", "Zed"]
SCAN_ROWS, SCAN_GROUP = 36_000, 4_000


def scan_table(n=SCAN_ROWS, group=SCAN_GROUP, many_words=False, seed=11):
    """region (string; each row group draws from a different window of words, so row-group dictionaries differ from
    each other and from the file's), carrier int64 -1..9 (null for the whole of row group 5), day date32, revenue
    decimal(18, 2) (null for the whole of row group 2), cost int64."""
    r = _random.Random(seed)
    region, carrier, day, revenue, cost = [], [], [], [], []
    for i in range(n):
        rg = i // group
        if many_words:
            region.append(None if r.random() < 0.03 else f"w{r.randint(0, 2999)}")
        else:
            start = (rg * 3) % 8
            region.append(None if r.random() < 0.05 else SCAN_WORDS[(start + r.randint(0, 2)) % 8])
        carrier.append(None if rg == 5 or r.random() < 0.03 else r.randint(-1, 9))
        day.append(_dt.date(1970, 1, 1) + _dt.timedelta(days=r.randint(9000, 15000)))
        revenue.append(None if rg == 2 or r.random() < 0.1 else
                       _dec.Decimal(5_000_000_000_000_000 + r.randint(0, 99) if i % 400 == 0 else r.randint(-10**6, 10**6)).scaleb(-2))
        cost.append(r.randint(-5000, 5000))
    return pa.table({
        "region": pa.array(region, pa.string()), "carrier": pa.array(carrier, pa.int64()),
        "day": pa.array(day, pa.date32()), "revenue": pa.array(revenue, pa.decimal128(18, 2)),
        "cost": pa.array(cost, pa.int64()),
    })


@pytest.fixture(scope="module")
def prober(tmp_path_factory):
    out = tmp_path_factory.mktemp("mojo") / "pq_scan_probe"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_scan_probe.mojo"), str(out)],
                   check=True, capture_output=True, text=True)
    return out


def probe(prober, path, mode, query):
    p = subprocess.run([str(prober), str(path), mode, query], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(p.stdout.strip().split("\n")[-1])


SCAN_QUERIES = ["by_region", "filtered", "pivot", "bins", "global"]


def same_answer(a, b):
    """Every number a caller can read, exactly (not the time or the memory)."""
    for k in ("words", "kinds", "slot_of", "count_slot_of", "counts", "slots", "present"):
        assert a[k] == b[k], k
    assert a["device"] == b["device"] == 0


def agg_values(res, i, g):
    kind, so, cso = res["kinds"][i], res["slot_of"][i], res["count_slot_of"][i]
    if kind == 1:                                       # count()
        return res["counts"][g]
    if kind == 2:                                       # count(e)
        return res["slots"][cso][g] if cso >= 0 else res["counts"][g]
    return res["slots"][so][g]


def check_against_duckdb(path, res, query):
    """The query's numbers per group, from DuckDB on the same file."""
    words = res["words"].split("|")
    cents = "CAST(revenue * 100 AS BIGINT)"
    day = "(CAST(day AS DATE) - DATE '1970-01-01')"
    con = duckdb.connect()
    src = f"read_parquet('{path}')"
    if query == "by_region":
        rows = con.execute(f"SELECT region, sum({cents}), count(*), count(revenue), min(cost), max(cost), min({cents}), max({cents}) "
                           f"FROM {src} WHERE region IS NOT NULL GROUP BY region").fetchall()
        for region, s, n, nv, lo, hi, rlo, rhi in rows:
            g = words.index(region)
            assert res["counts"][g] == n
            assert agg_values(res, 1, g) == n and agg_values(res, 2, g) == nv
            assert agg_values(res, 3, g) == lo and agg_values(res, 4, g) == hi
            if nv:
                assert agg_values(res, 0, g) == int(s)
                assert agg_values(res, 6, g) == rlo and agg_values(res, 7, g) == rhi
                assert agg_values(res, 5, g) == int(s)                       # mean's sum
            assert res["present"][6][g] == (1 if nv else 0)
        assert sum(res["counts"]) == sum(r[2] for r in rows)
    elif query == "filtered":
        rows = con.execute(f"SELECT carrier, sum({cents}), count(*), max(cost) FROM {src} "
                           f"WHERE region = 'East' AND {day} BETWEEN 10000 AND 14000 AND carrier BETWEEN 0 AND 7 GROUP BY carrier").fetchall()
        got = {c: (agg_values(res, 0, c), res["counts"][c], agg_values(res, 2, c)) for c in range(8) if res["counts"][c]}
        assert got == {c: (int(s) if s is not None else 0, n, hi) for c, s, n, hi in rows}
    elif query == "pivot":
        rows = con.execute(f"SELECT region, carrier, sum(cost), count(revenue) FROM {src} WHERE region IS NOT NULL "
                           f"AND carrier BETWEEN 0 AND 7 GROUP BY region, carrier").fetchall()
        got = {g: (agg_values(res, 0, g), agg_values(res, 1, g)) for g in range(len(res["counts"])) if res["counts"][g]}
        assert got == {words.index(r) * 8 + c: (s, nv) for r, c, s, nv in rows}
    elif query == "bins":
        rows = con.execute(f"SELECT least({day} // 30, 99), count(*), sum(cost) FROM {src} WHERE {day} IS NOT NULL GROUP BY 1").fetchall()
        got = {g: (res["counts"][g], agg_values(res, 1, g)) for g in range(100) if res["counts"][g]}
        assert got == {b: (n, s) for b, n, s in rows}
    elif query == "global":
        s, n, rlo, rhi = con.execute(f"SELECT sum(cost), count(*), min({cents}), max({cents}) FROM {src}").fetchone()
        assert (agg_values(res, 0, 0), agg_values(res, 1, 0), agg_values(res, 2, 0), agg_values(res, 3, 0)) == (s, n, rlo, rhi)


PYARROW_FILES = {
    "zstd-dict-v1": dict(compression="zstd", use_dictionary=True, data_page_version="1.0"),
    "snappy-dict-v2": dict(compression="snappy", use_dictionary=True, data_page_version="2.0"),
    "none-plain-v1": dict(compression="none", use_dictionary=False, data_page_version="1.0"),
    "zstd-plain-v2": dict(compression="zstd", use_dictionary=False, data_page_version="2.0"),
    "small-pages": dict(compression="snappy", use_dictionary=True, data_page_size=2_000),
}


def write_with_empty_groups(path, table, **opts):
    """Row groups of SCAN_GROUP rows, with an empty row group before the first, in the middle and after the last."""
    w = pq.ParquetWriter(path, table.schema, **opts)
    w.write_table(table.slice(0, 0))
    for lo in range(0, table.num_rows, SCAN_GROUP):
        w.write_table(table.slice(lo, SCAN_GROUP))
        if lo == 3 * SCAN_GROUP:
            w.write_table(table.slice(0, 0))
    w.write_table(table.slice(0, 0))
    w.close()


@pytest.mark.parametrize("name", sorted(PYARROW_FILES))
def test_streamed_query_equals_read_parquet_on_pyarrow_files(prober, tmp_path, name):
    path = tmp_path / "t.parquet"
    write_with_empty_groups(path, scan_table(), **PYARROW_FILES[name])
    assert pq.read_metadata(path).num_row_groups == 12                   # 9 + 3 empty
    for q in SCAN_QUERIES:
        scanned, read = probe(prober, path, "scan", q), probe(prober, path, "read", q)
        same_answer(scanned, read)
        check_against_duckdb(path, scanned, q)


def test_text_that_falls_back_to_plain_pages_streams_the_same_codes(prober, tmp_path):
    """3,000 distinct words with a 1 KB dictionary limit: each chunk starts dictionary-encoded and finishes PLAIN."""
    path = tmp_path / "t.parquet"
    pq.write_table(scan_table(many_words=True), path, row_group_size=SCAN_GROUP, dictionary_pagesize_limit=1024)
    encodings = pq.read_metadata(path).row_group(0).column(0).encodings
    assert "PLAIN" in encodings and "RLE_DICTIONARY" in encodings
    for q in ("by_region", "big"):
        scanned, read = probe(prober, path, "scan", q), probe(prober, path, "read", q)
        same_answer(scanned, read)
    check_against_duckdb(path, probe(prober, path, "scan", "by_region"), "by_region")


@pytest.mark.parametrize("codec", ["zstd", "snappy", "none"])
def test_streamed_query_equals_read_parquet_on_duckdb_files(prober, tmp_path, codec):
    src = tmp_path / "src.parquet"
    pq.write_table(scan_table(), src)
    path = tmp_path / "t.parquet"
    duckdb.connect().execute(f"COPY (SELECT * FROM read_parquet('{src}')) TO '{path}' "
                             f"(FORMAT parquet, ROW_GROUP_SIZE {SCAN_GROUP}, COMPRESSION {'uncompressed' if codec == 'none' else codec})")
    assert pq.read_metadata(path).num_row_groups == 9
    for q in SCAN_QUERIES:
        scanned, read = probe(prober, path, "scan", q), probe(prober, path, "read", q)
        same_answer(scanned, read)
        check_against_duckdb(path, scanned, q)


def test_a_file_with_no_rows_and_a_file_that_is_not_parquet(prober, tmp_path):
    path = tmp_path / "e.parquet"
    pq.write_table(scan_table(0), path)
    same_answer(probe(prober, path, "scan", "global"), probe(prober, path, "read", "global"))
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not a parquet file at all")
    p = subprocess.run([str(prober), str(bad), "scan", "global"], capture_output=True, text=True)
    assert p.returncode != 0 and "not a Parquet file" in p.stdout + p.stderr


def test_a_truncated_file_raises_instead_of_crashing(prober, tmp_path):
    path = tmp_path / "t.parquet"
    pq.write_table(scan_table(), path, row_group_size=SCAN_GROUP)
    data = path.read_bytes()
    for cut in (len(data) // 2, len(data) - 100, 20):
        bad = tmp_path / f"cut{cut}.parquet"
        bad.write_bytes(data[:cut])
        p = subprocess.run([str(prober), str(bad), "scan", "by_region"], capture_output=True, text=True)
        assert p.returncode == 1 and "Unhandled exception" in p.stdout + p.stderr, (cut, p.returncode)


def test_random_byte_flips_never_crash_a_scan(prober, tmp_path):
    path = tmp_path / "t.parquet"
    pq.write_table(scan_table(8_000), path, row_group_size=2_000)
    data = bytearray(path.read_bytes())
    r = _random.Random(5)
    for k in range(60):
        flipped = bytearray(data)
        for _ in range(3):
            flipped[r.randrange(len(flipped))] ^= 1 << r.randrange(8)
        bad = tmp_path / "flip.parquet"
        bad.write_bytes(flipped)
        p = subprocess.run([str(prober), str(bad), "scan", "by_region"], capture_output=True, text=True)
        assert p.returncode in (0, 1), (k, p.returncode)                 # an answer or a readable error, never a signal


def test_an_unused_unique_id_text_column_costs_nothing(prober, tmp_path):
    """500k unique 30-byte ids that the writer's dictionary limit pushed to PLAIN pages, never read by the query: the
    stream stays near one row group, where the whole-file read holds the ids."""
    path = tmp_path / "hc.parquet"
    n, g = 2_000_000, 250_000
    w = pq.ParquetWriter(path, pa.schema([("t", pa.string()), ("v", pa.int64())]), compression="none")
    for lo in range(0, n, g):
        ids = pa.array([f"id-{i:012d}-padpadpad" for i in range(lo, lo + g)])
        w.write_table(pa.table([ids, pa.array(range(lo, lo + g), pa.int64())], schema=w.schema))
    w.close()
    scanned, read = probe(prober, path, "scan", "sum_v"), probe(prober, path, "read", "sum_v")
    same_answer(scanned, read)
    assert scanned["slots"][0][0] == n * (n - 1) // 2 and scanned["counts"][0] == n
    assert scanned["peak_kb"] < 90_000 and scanned["peak_kb"] * 4 < read["peak_kb"], (scanned["peak_kb"], read["peak_kb"])


def test_the_serial_build_decodes_the_same_numbers_as_the_parallel_one(prober, tmp_path, monkeypatch):
    """F-141: pages decode in parallel; the serial build (no thread pool) runs them in a loop. Many small pages, nulls,
    dictionary and plain text: every answer is identical, streamed and whole-file."""
    monkeypatch.setenv("MXFRAME_BUILD_FLAGS", "-D MXFRAME_SERIAL")
    serial = tmp_path / "pq_scan_probe_serial"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_scan_probe.mojo"), str(serial)],
                   check=True, capture_output=True, text=True)
    path = tmp_path / "t.parquet"
    pq.write_table(scan_table(), path, row_group_size=SCAN_GROUP, data_page_size=1_500, compression="snappy")
    for q in SCAN_QUERIES:
        for mode in ("scan", "read"):
            same_answer(probe(prober, path, mode, q), probe(serial, path, mode, q))


# ── F-141 review: every cell through read and scan, claims a page cannot back ───────────────────────

@pytest.fixture(scope="module")
def cells(tmp_path_factory):
    out = tmp_path_factory.mktemp("mojo") / "pq_cells"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_cells.mojo"), str(out)],
                   check=True, capture_output=True, text=True)
    return out


def cell_text(v, t):
    if v is None:
        return "N"
    if pa.types.is_decimal(t):
        return str(int(v.scaleb(t.scale)))
    if pa.types.is_date(t):
        return str((v - _dt.date(1970, 1, 1)).days)
    if pa.types.is_boolean(t):
        return str(int(v))
    return str(v)


def page_edge_table(n, k, seed):
    """Nulls on both edges of every k-row page (and some at random), a null-free column, int32, decimals of both
    widths, bool, dictionary text, text that falls back from dictionary to PLAIN mid-column, an all-null column."""
    r = _random.Random(seed)

    def edge(i):
        return i % k == 0 or i % k == k - 1 or r.random() < 0.15

    words = [f"w{j}" for j in range(40)]
    return pa.table({
        "i64": pa.array([None if edge(i) else r.randint(-2**62, 2**62) for i in range(n)], pa.int64()),
        "i32": pa.array([None if i % 9 == 3 else r.randint(-2**31, 2**31 - 1) for i in range(n)], pa.int32()),
        "req": pa.array([r.randint(-10**12, 10**12) for i in range(n)], pa.int64()),
        "dec": pa.array([None if edge(i) else _dec.Decimal(r.randint(-10**9, 10**9)).scaleb(-2) for i in range(n)], pa.decimal128(12, 2)),
        "dec18": pa.array([None if i % 5 == 0 else _dec.Decimal(r.randint(-10**17, 10**17)).scaleb(-4) for i in range(n)], pa.decimal128(18, 4)),
        "d": pa.array([None if edge(i) else _dt.date(1960, 1, 1) + _dt.timedelta(days=r.randint(0, 40000)) for i in range(n)], pa.date32()),
        "b": pa.array([None if edge(i) else r.random() < 0.5 for i in range(n)], pa.bool_()),
        "s": pa.array([None if edge(i) else r.choice(words) for i in range(n)], pa.string()),
        "fb": pa.array([None if i % 7 == 1 else (r.choice(words) if i < n // 3 else f"u{i}-{r.randint(0, 10**6)}") for i in range(n)], pa.string()),
        "allnull": pa.array([None] * n, pa.int64()),
    })


@pytest.mark.parametrize("k", [1, 3, 7, 13])
@pytest.mark.parametrize("version", ["1.0", "2.0"])
@pytest.mark.parametrize("codec", ["NONE", "ZSTD"])
@pytest.mark.parametrize("dict_limit", [256, 1 << 20])
def test_every_cell_through_read_and_scan_with_odd_pages(cells, tmp_path, k, version, codec, dict_limit):
    """Pages of k rows (nulls on their edges), v1 and v2, a dictionary that overflows mid-column, two row groups."""
    n = 1500
    t = page_edge_table(n, k, k * 31 + dict_limit)
    path = tmp_path / "t.parquet"
    pq.write_table(t, path, row_group_size=997, data_page_size=1, write_batch_size=k, data_page_version=version,
                   compression=codec, dictionary_pagesize_limit=dict_limit)
    want = "".join("\t".join(cell_text(t.column(c)[i].as_py(), t.schema.field(c).type) for c in range(t.num_columns)) + "\n"
                   for i in range(n))
    for mode in ("read", "scan"):
        got = subprocess.run([str(cells), str(path), mode], capture_output=True, text=True, check=True).stdout
        assert got == want, (mode, next(i for i, (a, b) in enumerate(zip(got.split("\n"), want.split("\n"))) if a != b))


def multi_page_file(path, pages, ptype=2, rep=0, text=False, group_rows=None, name=b"x"):
    """One INT64 column `x`, one row group, one chunk made of `pages`: (kind, nvals, encoding, body) with kind 0 a data
    page and 2 a dictionary page. Sizes are the body's."""
    data = b"PAR1"
    off = len(data)
    chunk = b""
    total = 0
    for kind, nvals, enc, body in pages:
        h = Compact()
        h.i32(1, kind)
        h.i32(2, len(body))
        h.i32(3, len(body))
        if kind == 2:
            h.st(7); h.i32(1, nvals); h.i32(2, enc); h.end()
        else:
            h.st(5); h.i32(1, nvals); h.i32(2, enc); h.i32(3, 3); h.i32(4, 3); h.end()
            total += nvals
        h.b.append(0)
        chunk += bytes(h.b) + body
    data += chunk
    m = Compact()
    m.i32(1, 1); m.lst(2, 12, 2)
    m.el(); m.bin(4, b"schema"); m.i32(5, 1); m.end()
    m.el(); m.i32(1, ptype); m.i32(3, rep); m.bin(4, name)
    if text:
        m.i32(6, 0)                                           # converted type UTF8
    m.end()
    m.i64(3, total); m.lst(4, 12, 1)
    m.el(); m.lst(1, 12, 1)
    m.el(); m.i64(2, off)
    m.st(3); m.i32(1, ptype); m.lst(2, 5, 1); m.b += zz(0); m.lst(3, 8, 1); m.b += uv(len(name)) + name
    m.i32(4, 0); m.i64(5, total); m.i64(6, len(chunk)); m.i64(7, len(chunk)); m.i64(9, off); m.end(); m.end()
    m.i64(2, len(chunk)); m.i64(3, total if group_rows is None else group_rows); m.end(); m.b.append(0)
    data += bytes(m.b) + struct.pack("<i", len(m.b)) + b"PAR1"
    Path(path).write_bytes(data)


def refused_by_both(cells, path, message):
    """read_parquet and the scan raise `message` (a normal error exit) under a 32 GB address-space cap, so a build that
    allocates what the file claims would abort instead."""
    import resource

    def cap():
        resource.setrlimit(resource.RLIMIT_AS, (32 << 30, 32 << 30))

    for mode in ("read", "scan"):
        p = subprocess.run([str(cells), str(path), mode], capture_output=True, text=True, preexec_fn=cap)
        assert p.returncode == 1, (mode, p.returncode, p.stderr[-300:])
        assert message in p.stdout + p.stderr, (mode, p.stdout + p.stderr)


def footer_counts_file(path, groups):
    """One INT64 column; every row group's chunk points at the same 3-value PLAIN page and states `values` / `rows`
    from `groups` (the footer is what lies, the page is fine)."""
    h = Compact(); h.i32(1, 0); h.i32(2, 24); h.i32(3, 24)
    h.st(5); h.i32(1, 3); h.i32(2, 0); h.i32(3, 3); h.i32(4, 3); h.end(); h.b.append(0)
    data = b"PAR1"; off = len(data); chunk = bytes(h.b) + struct.pack("<qqq", 1, 2, 3); data += chunk
    m = Compact(); m.i32(1, 1); m.lst(2, 12, 2)
    m.el(); m.bin(4, b"schema"); m.i32(5, 1); m.end()
    m.el(); m.i32(1, 2); m.i32(3, 0); m.bin(4, b"x"); m.end()
    m.i64(3, sum(g[1] for g in groups) & ((1 << 63) - 1)); m.lst(4, 12, len(groups))
    for vals, rows in groups:
        m.el(); m.lst(1, 12, 1)
        m.el(); m.i64(2, off)
        m.st(3); m.i32(1, 2); m.lst(2, 5, 1); m.b += zz(0); m.lst(3, 8, 1); m.b += uv(1) + b"x"
        m.i32(4, 0); m.i64(5, vals); m.i64(6, len(chunk)); m.i64(7, len(chunk)); m.i64(9, off); m.end(); m.end()
        m.i64(2, len(chunk)); m.i64(3, rows); m.end()
    m.b.append(0)
    data += bytes(m.b) + struct.pack("<i", len(m.b)) + b"PAR1"
    Path(path).write_bytes(data)


@pytest.mark.parametrize("groups", [[(-5, -5)], [(-(1 << 40), 3), (3, 3)], [(1 << 62, 1 << 62)] * 2,
                                    [(1 << 62, 1 << 62)] * 4],
                         ids=["negative", "negative then good", "sum wraps x2", "sum wraps x4"])
def test_footer_row_counts_that_are_negative_or_wrap_are_refused(cells, tmp_path, groups):
    """read_parquet sums the footer's counts to size its columns: a negative count or a sum past int64 is refused (a
    normal error exit, never an abort) under a 32 GB address-space cap, as the scan already was."""
    import resource

    def cap():
        resource.setrlimit(resource.RLIMIT_AS, (32 << 30, 32 << 30))

    path = tmp_path / "counts.parquet"
    footer_counts_file(path, groups)
    for mode in ("read", "scan"):
        p = subprocess.run([str(cells), str(path), mode], capture_output=True, text=True, preexec_fn=cap)
        assert p.returncode == 1, (mode, p.returncode, p.stderr[-300:])


I31 = 2**31 - 1


def bomb_file(path, npages, per, usize, codec):
    """One INT64 column of `npages` header-only pages (no bodies), each claiming `per` rows and `usize` bytes."""
    data = b"PAR1"
    off = len(data)
    chunk = b""
    for _ in range(npages):
        h = Compact(); h.i32(1, 0); h.i32(2, usize); h.i32(3, 0)
        h.st(5); h.i32(1, per); h.i32(2, 0); h.i32(3, 3); h.i32(4, 3); h.end(); h.b.append(0)
        chunk += bytes(h.b)
    data += chunk
    total = npages * per
    m = Compact(); m.i32(1, 1); m.lst(2, 12, 2)
    m.el(); m.bin(4, b"schema"); m.i32(5, 1); m.end()
    m.el(); m.i32(1, 2); m.i32(3, 0); m.bin(4, b"x"); m.end()
    m.i64(3, total); m.lst(4, 12, 1)
    m.el(); m.lst(1, 12, 1)
    m.el(); m.i64(2, off)
    m.st(3); m.i32(1, 2); m.lst(2, 5, 1); m.b += zz(0); m.lst(3, 8, 1); m.b += uv(1) + b"x"
    m.i32(4, codec); m.i64(5, total); m.i64(6, len(chunk)); m.i64(7, len(chunk)); m.i64(9, off); m.end(); m.end()
    m.i64(2, len(chunk)); m.i64(3, total); m.end(); m.b.append(0)
    data += bytes(m.b) + struct.pack("<i", len(m.b)) + b"PAR1"
    Path(path).write_bytes(data)


def test_row_counts_beyond_the_machines_memory_are_refused_before_anything_is_sized(cells, tmp_path):
    """The row counts are consistent (pages = chunk = row group = file) and the bytes are tiny, so only the memory the
    claim would need can refuse it: 4,096 empty pages of 2^23 rows (275 GB), a snappy chunk of 64 pages claiming 2^31-1
    rows and bytes each, a single page of 2^31-1 rows in 24 bytes (the last needs 16 GB: only a small machine refuses it)."""
    path = tmp_path / "bomb.parquet"
    bomb_file(path, 4096, 1 << 23, 0, 0)
    refused_by_both(cells, path, "claims 34359738368 rows")
    refused_by_both(cells, path, "more than this machine's memory")
    bomb_file(path, 64, I31, I31, 1)
    refused_by_both(cells, path, "claims 137438953408 rows")
    bomb_file(path, 64, I31, 0, 0)
    refused_by_both(cells, path, "more than this machine's memory")


def test_a_row_group_whose_pages_disagree_with_its_row_count_is_refused(cells, tmp_path):
    path = tmp_path / "rows.parquet"
    multi_page_file(path, [(0, 3, 0, struct.pack("<qqq", 1, 2, 3))])        # num_rows 3, one page of 3: fine
    out = subprocess.run([str(cells), str(path), "read"], capture_output=True, text=True, check=True).stdout
    assert out == "1\n2\n3\n"
    multi_page_file(path, [(0, 3, 0, struct.pack("<qqq", 1, 2, 3))], group_rows=4)     # the row group says 4
    refused_by_both(cells, path, "column `x` has 3 values in a row group of 4 rows (corrupt file)")


def test_a_second_dictionary_page_is_refused(cells, tmp_path):
    path = tmp_path / "two_dicts.parquet"
    d = struct.pack("<qq", 10, 20)
    multi_page_file(path, [(2, 2, 0, d), (2, 2, 0, d), (0, 3, 8, bytes([1]) + uv((1 << 1) | 1) + bytes([0b010]))])
    refused_by_both(cells, path, "column `x`: a chunk has two dictionary pages (corrupt file)")
    multi_page_file(path, [(2, 2, 0, d), (0, 3, 8, bytes([1]) + uv((1 << 1) | 1) + bytes([0b010]))])
    for mode in ("read", "scan"):                                     # one dictionary page: the indices 0, 1, 0 read
        out = subprocess.run([str(cells), str(path), mode], capture_output=True, text=True, check=True).stdout
        assert out == "10\n20\n10\n"


def test_a_dictionary_index_past_the_dictionary_says_so(cells, tmp_path):
    path = tmp_path / "index.parquet"
    d = struct.pack("<qq", 10, 20)
    # width 2, one bit-packed group of 8 indices: 3, 0, 0, ... (3 is past the two words)
    multi_page_file(path, [(2, 2, 0, d), (0, 3, 8, bytes([2]) + uv((1 << 1) | 1) + bytes([0b11, 0]))])
    refused_by_both(cells, path, "column `x`: a dictionary index is out of range (corrupt file)")


def test_the_first_error_in_page_order_is_the_one_raised(cells, tmp_path):
    """TEXT under read_parquet: its PLAIN pages add words and so run serially, its dictionary-indexed pages run in
    parallel. A bad dictionary index page that comes BEFORE a truncated PLAIN page in the file is the error reported
    (the serial page used to raise first, whatever the file order), and the other way round."""
    path = tmp_path / "order.parquet"
    words = struct.pack("<I", 1) + b"a" + struct.pack("<I", 1) + b"b"
    bad_index = bytes([2]) + uv((1 << 1) | 1) + bytes([0b11, 0])             # index 3 of a 2-word dictionary
    short_plain = struct.pack("<I", 1) + b"a"                                # 1 word of the 3 it claims
    multi_page_file(path, [(2, 2, 0, words), (0, 3, 8, bad_index), (0, 3, 0, short_plain)], ptype=6, text=True)
    p = subprocess.run([str(cells), str(path), "read"], capture_output=True, text=True)
    assert p.returncode == 1 and "a dictionary index is out of range" in p.stdout + p.stderr, p.stdout + p.stderr
    multi_page_file(path, [(2, 2, 0, words), (0, 3, 0, short_plain), (0, 3, 8, bad_index)], ptype=6, text=True)
    p = subprocess.run([str(cells), str(path), "read"], capture_output=True, text=True)
    assert p.returncode == 1 and "a Parquet page ends early" in p.stdout + p.stderr, p.stdout + p.stderr


# ── F-142: text pages decoded in parallel, words coded in file order ─────────────────────────────────

def text_table(n, seed=5):
    """uid: unique ids with nulls (every row of a group misses the word index), rep: 60k words repeated across pages
    with nulls, fb: a few words, then unique ones (dictionary fallback mid-column), small: eight words (dictionary)."""
    r = _random.Random(seed)
    vocab = [f"vocab-{j:06d}-padding" for j in range(60_000)]
    return pa.table({
        "uid": pa.array([None if r.random() < 0.05 else f"id-{i:09d}-{r.randint(0, 10**6)}" for i in range(n)], pa.string()),
        "rep": pa.array([None if r.random() < 0.1 else r.choice(vocab) for _ in range(n)], pa.string()),
        "fb": pa.array([None if i % 11 == 1 else (f"few{r.randint(0, 9)}" if i < n // 4 else f"u{i}-{r.randint(0, 10**6)}") for i in range(n)], pa.string()),
        "small": pa.array([f"s{r.randint(0, 7)}" for _ in range(n)], pa.string()),
    })


def first_seen_codes(column):
    codes, out = {}, []
    for v in column.to_pylist():
        if v is None:
            out.append("N")
        else:
            out.append(str(codes.setdefault(v, len(codes))))
    return out


@pytest.mark.parametrize("version", ["1.0", "2.0"])
@pytest.mark.parametrize("codec", ["NONE", "SNAPPY"])
def test_text_words_and_codes_through_the_two_phase_decode(cells, tmp_path, version, codec):
    """Every group misses 100k+ words at once (the parallel dedupe runs), pages are small, words repeat across pages
    and groups: the words AND the codes (first seen, in file order) equal pyarrow's values and a serial first-seen
    numbering, through read_parquet and the scan, and the serial build gives the same bytes."""
    n = 240_000
    t = text_table(n)
    path = tmp_path / "t.parquet"
    pq.write_table(t, path, row_group_size=120_000, data_page_size=64 * 1024, data_page_version=version, compression=codec)
    want_words = "".join("\t".join(cell_text(t.column(c)[i].as_py(), t.schema.field(c).type) for c in range(4)) + "\n" for i in range(n))
    cols = [first_seen_codes(t.column(c)) for c in range(4)]
    want_codes = "".join("\t".join(cols[c][i] for c in range(4)) + "\n" for i in range(n))
    import os
    env = dict(os.environ)
    serial = tmp_path / "pq_cells_serial"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_cells.mojo"), str(serial)],
                   check=True, capture_output=True, text=True, env={**env, "MXFRAME_BUILD_FLAGS": "-D MXFRAME_SERIAL"})
    for binary in (cells, serial):
        for mode in ("read", "scan"):
            assert subprocess.run([str(binary), str(path), mode], capture_output=True, text=True, check=True).stdout == want_words
            assert subprocess.run([str(binary), str(path), mode, "codes"], capture_output=True, text=True, check=True).stdout == want_codes


# ── F-143: the next row group decodes while the kernel runs; errors and sums still follow row-group order ──

def test_a_kernel_error_in_an_earlier_group_beats_a_decode_error_in_the_next(prober, tmp_path):
    """Group 1's sum overflows (the kernel, task 0) while group 2 is being decoded beside it and is corrupt (task 1):
    the error of the earlier group is the one raised, as when the groups ran one after the other."""
    big = 2**62
    t = pa.table({"v": pa.array([1, 2, 3, big, big, 7, 8, 9], pa.int64())})
    path = tmp_path / "t.parquet"
    pq.write_table(t, path, row_group_size=3, compression="NONE", use_dictionary=False)       # groups: [1 2 3] [big big 7] [8 9]
    md = pq.read_metadata(path)
    assert md.num_row_groups == 3
    raw = bytearray(path.read_bytes())
    off = md.row_group(2).column(0).data_page_offset
    raw[off:off + 8] = b"\xff" * 8                                                          # group 2's page header
    path.write_bytes(raw)
    p = subprocess.run([str(prober), str(path), "scan", "sum_v"], capture_output=True, text=True)
    assert p.returncode == 1 and "a sum or an expression passed the int64 range" in p.stdout + p.stderr, p.stdout + p.stderr


def test_a_sum_whose_partials_pass_int64_raises_the_same_streamed_and_whole(prober, tmp_path):
    big = 2**62
    path = tmp_path / "t.parquet"
    pq.write_table(pa.table({"v": pa.array([big, big, big, big, -big, 5], pa.int64())}), path, row_group_size=2,
                   compression="NONE", use_dictionary=False)
    outs = [subprocess.run([str(prober), str(path), mode, "sum_v"], capture_output=True, text=True) for mode in ("scan", "read")]
    assert [o.returncode for o in outs] == [1, 1]
    assert all("a sum or an expression passed the int64 range" in o.stdout + o.stderr for o in outs)


def test_an_overflow_while_merging_a_group_beats_the_decode_error_of_the_next(prober, tmp_path):
    """Groups [2^62, 2^62-1] (the partial fits), [1, 0] (adding it to the running total overflows) and a corrupt third:
    the group that is merged before the third is reached reports its overflow, not the third's corrupt page."""
    big = 2**62
    path = tmp_path / "t.parquet"
    pq.write_table(pa.table({"v": pa.array([big, big - 1, 1, 0, 8, 9], pa.int64())}), path, row_group_size=2,
                   compression="NONE", use_dictionary=False)
    md = pq.read_metadata(path)
    raw = bytearray(path.read_bytes())
    off = md.row_group(2).column(0).data_page_offset
    raw[off:off + 8] = b"\xff" * 8
    path.write_bytes(raw)
    p = subprocess.run([str(prober), str(path), "scan", "sum_v"], capture_output=True, text=True)
    assert p.returncode == 1 and "a sum or an expression passed the int64 range" in p.stdout + p.stderr, p.stdout + p.stderr


# ── F-145: big pages split by row range, dictionary pages decoded as parallel jobs ───────────────────

def big_page_table(n, seed=9):
    """One column per kind of splittable page: a dictionary-index page (low cardinality, with nulls), PLAIN int64 / int32
    with nulls, PLAIN decimals of both widths, a dictionary text column, a required column and an all-null one."""
    r = _random.Random(seed)
    words = [f"w{j}" for j in range(30)]
    return pa.table({
        "lc": pa.array([None if r.random() < 0.07 else r.randint(0, 9) for _ in range(n)], pa.int64()),
        "lc_req": pa.array([r.randint(-3, 400) for _ in range(n)], pa.int64()),
        "i64": pa.array([None if r.random() < 0.2 else r.randint(-2**62, 2**62) for _ in range(n)], pa.int64()),
        "i32": pa.array([None if i % 5 == 0 or (i // 7) % 11 == 0 else r.randint(-2**31, 2**31 - 1) for i in range(n)], pa.int32()),
        "dec18": pa.array([_dec.Decimal(r.randint(-10**17, 10**17)).scaleb(-2) for _ in range(n)], pa.decimal128(18, 2)),
        "dec12": pa.array([None if r.random() < 0.05 else _dec.Decimal(r.randint(-10**9, 10**9)).scaleb(-2) for _ in range(n)], pa.decimal128(12, 2)),
        "txt": pa.array([None if r.random() < 0.1 else r.choice(words) for _ in range(n)], pa.string()),
        "allnull": pa.array([None] * n, pa.int64()),
    })


@pytest.mark.parametrize("version", ["1.0", "2.0"])
@pytest.mark.parametrize("codec", ["NONE", "SNAPPY"])
def test_big_single_page_columns_slice_by_row_range_and_decode_the_same(cells, tmp_path, version, codec):
    """700k-row pages (one page a column) are cut into slices at run boundaries and byte offsets, with nulls on either side
    of the cuts: every cell equals pyarrow's, through read_parquet and the scan, in the parallel and the serial build."""
    n = 700_000
    t = big_page_table(n)
    path = tmp_path / "t.parquet"
    pq.write_table(t, path, row_group_size=n, data_page_size=256 * 1024 * 1024, data_page_version=version, compression=codec,
                   use_dictionary=["lc", "lc_req", "txt"])
    md = pq.read_metadata(path).row_group(0)
    assert md.column(0).encodings.count("RLE_DICTIONARY") and "PLAIN" in md.column(2).encodings
    want = "".join("\t".join(cell_text(t.column(c)[i].as_py(), t.schema.field(c).type) for c in range(t.num_columns)) + "\n"
                   for i in range(n))
    serial = tmp_path / "pq_cells_serial"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_cells.mojo"), str(serial)],
                   check=True, capture_output=True, text=True, env={**__import__("os").environ, "MXFRAME_BUILD_FLAGS": "-D MXFRAME_SERIAL"})
    for binary in (cells, serial):
        for mode in ("read", "scan"):
            got = subprocess.run([str(binary), str(path), mode], capture_output=True, text=True, check=True).stdout
            assert got == want, (binary.name, mode)


# ── F-149: morsel-driven streamed scans (a worker per row group, blocks of rows) ─────────────────────

def probe_morsels(prober, path, query, rows, mode="1"):
    env = {**__import__("os").environ, "MXFRAME_SCAN_MORSELS": mode, "MXFRAME_MORSEL_ROWS": str(rows)}
    p = subprocess.run([str(prober), str(path), "scan", query], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(p.stdout.strip().split("\n")[-1])


@pytest.mark.parametrize("name", ["zstd-dict-v1", "snappy-dict-v2", "none-plain-v1", "small-pages"])
@pytest.mark.parametrize("rows", [1000, 3001])
def test_morsels_equal_read_parquet_and_duckdb(prober, tmp_path, name, rows):
    """Blocks of 1,000 and 3,001 rows cut across pages of every size, nulls (a whole row group of them), empty row groups,
    dictionary fallback to PLAIN text: the answer of every query equals read_parquet's and DuckDB's."""
    path = tmp_path / "t.parquet"
    write_with_empty_groups(path, scan_table(), **PYARROW_FILES[name])
    for q in SCAN_QUERIES:
        got = probe_morsels(prober, path, q, rows)
        same_answer(got, probe(prober, path, "read", q))
        check_against_duckdb(path, got, q)


def test_morsels_cut_pages_inside_runs_and_over_nulls(prober, tmp_path):
    """Dictionary-index pages with long repeated runs and short bit-packed ones, nulls everywhere, and blocks that start in
    the middle of a run: the groups sum to what read_parquet gives."""
    r = _random.Random(3)
    n = 60_000
    vals = []
    while len(vals) < n:                                              # runs of 1..3000 equal values, then noise
        vals += [r.randint(0, 6)] * r.randint(1, 3000) if r.random() < 0.5 else [r.randint(0, 6) for _ in range(r.randint(1, 40))]
    vals = vals[:n]
    t = pa.table({"region": pa.array([f"w{v}" if r.random() > 0.1 else None for v in vals], pa.string()),
                  "carrier": pa.array([None if r.random() < 0.2 else v for v in vals], pa.int64()),
                  "day": pa.array([_dt.date(1970, 1, 1) + _dt.timedelta(days=9000 + v) for v in vals], pa.date32()),
                  "revenue": pa.array([None if r.random() < 0.1 else _dec.Decimal(r.randint(-10**6, 10**6)).scaleb(-2) for _ in vals],
                                      pa.decimal128(18, 2)),
                  "cost": pa.array([r.randint(-5000, 5000) for _ in vals], pa.int64())})
    path = tmp_path / "t.parquet"
    pq.write_table(t, path, row_group_size=15_000, data_page_size=20_000, compression="snappy")
    for rows in (700, 4096):
        for q in SCAN_QUERIES:
            same_answer(probe_morsels(prober, path, q, rows), probe(prober, path, "read", q))


def test_a_decode_error_later_in_a_morsel_beats_an_earlier_blocks_kernel_error(prober, tmp_path):
    """One chunk: block 1 (64 rows, all 2^62) overflows the sum in the kernel, a later page is corrupt (a bit width of 65):
    the decode error is reported whatever the block size (the group is decoded before it is aggregated), and the serial
    scan, the whole-file read path and the morsel scan with 64-row or 100,000-row blocks all say the same."""
    path = tmp_path / "t.parquet"
    d = struct.pack("<qq", 2**62, 1)
    good = bytes([1]) + uv((8 << 1) | 1) + bytes(8)                     # 64 indices, all 0
    bad = bytes([65]) + uv((1 << 1) | 1) + bytes(9)                     # a bit width of 65
    multi_page_file(path, [(2, 2, 0, d), (0, 64, 8, good), (0, 8, 8, bad)], name=b"v")
    for env_rows, mode in (("64", "1"), ("100000", "1"), ("", "0")):
        env = {**__import__("os").environ, "MXFRAME_SCAN_MORSELS": mode}
        if env_rows:
            env["MXFRAME_MORSEL_ROWS"] = env_rows
        p = subprocess.run([str(prober), str(path), "scan", "sum_v"], capture_output=True, text=True, env=env)
        assert p.returncode == 1 and "a bit width of 65" in p.stdout + p.stderr, (env_rows, mode, p.stdout + p.stderr)


@pytest.mark.parametrize("group", [1_200, 12_000, 36_000])
def test_morsels_with_many_few_and_one_row_groups(prober, tmp_path, group):
    """30 row groups (a group a core or more: one morsel per group), 3 (cut into row ranges of whole blocks) and 1 (all
    ranges of one group): the same answer as read_parquet and DuckDB, blocks of 1,000 rows."""
    path = tmp_path / "t.parquet"
    pq.write_table(scan_table(), path, row_group_size=group, compression="snappy")
    for q in SCAN_QUERIES:
        got = probe_morsels(prober, path, q, 1000)
        same_answer(got, probe(prober, path, "read", q))
        check_against_duckdb(path, got, q)


# ── F-147: editing a streamed file (Frame over a scan) ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def editor(tmp_path_factory):
    out = tmp_path_factory.mktemp("mojo") / "pq_edit_probe"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_edit_probe.mojo"), str(out)],
                   check=True, capture_output=True, text=True)
    return out


def edit_script(rows):
    """The probe's fixed edits (mojo/tools/pq_edit_probe.mojo), replayed here independently of the engine."""
    sets = {}
    for k in range(1000):
        sets[(k * 1000003) % rows] = k * 3 - 1500
    fresh = {(k * 7919 + 5) % rows for k in range(100)}
    removed = {(k * 999983 + 17) % rows for k in range(200)}
    return sets, fresh, removed


def run_edit_probe(editor, path, out, cache_mib, *extra):
    p = subprocess.run([str(editor), str(path), str(out), str(cache_mib), *extra], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(p.stdout.strip().split("\n")[-1])


def test_an_edited_streamed_file_equals_an_outside_replay_and_its_memory_stays_bounded(editor, tmp_path):
    """A file several times bigger than the frame's cache: edits, a query, a grid-style walk and a save through a Frame over
    a scan; the query, the walk and the saved file equal numpy/pyarrow/DuckDB replaying the same edits, and the peak resident
    memory of every phase stays a fraction of the file."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from bench_scan_memory import write_file
    import numpy as np
    rows, group, cache_mib = 12_000_000, 250_000, 4
    path, out = tmp_path / "big.parquet", tmp_path / "edited.parquet"
    write_file(str(path), rows, group)
    size_kb = path.stat().st_size // 1024
    res = run_edit_probe(editor, path, out, cache_mib)
    assert res["rows"] == rows and res["row_groups"] == rows // group

    # the replay
    orig = pq.read_table(path, columns=["region", "cost"], read_dictionary=["region"])
    cost = orig.column("cost").to_numpy().copy()
    region = orig.column("region").unify_dictionaries()                  # row groups carry different dictionaries
    dictionary = region.chunk(0).dictionary.to_pylist()
    codes = np.concatenate([c.indices.to_numpy() for c in region.chunks]).astype(np.int64)
    words = dictionary + ["Fresh", "Appended"]
    sets, fresh, removed = edit_script(rows)
    for row, v in sets.items():
        cost[row] = v
    for row in fresh:
        codes[row] = len(dictionary)
    alive = np.ones(rows, bool)
    alive[list(removed)] = False
    cost_all = np.concatenate([cost[alive], np.full(50, 77, np.int64)])
    codes_all = np.concatenate([codes[alive], np.full(50, len(dictionary) + 1, np.int64)])
    assert len(cost_all) == rows - len(removed) + 50

    # the query over the frame, by word
    got = {w: (s, n) for w, s, n in zip(res["words"], res["sums"], res["counts"]) if n}
    want = {}
    for k, w in enumerate(words):
        sel = codes_all == k
        if sel.any():
            want[w] = (int(cost_all[sel].sum()), int(sel.sum()))
    assert got == want

    # the walk (a row every 997 through get / is_null)
    walk = np.arange(0, rows, 997)
    walk = walk[alive[walk]]
    assert res["get_sum"] == int(cost[walk].sum())

    # the saved file, read by pyarrow and DuckDB: every cost, in order, and the metadata
    saved = pq.read_table(out)
    assert saved.num_rows == len(cost_all)
    assert np.array_equal(saved.column("cost").to_numpy(), cost_all)
    assert json.loads(saved.schema.metadata[b"omx"]) == {"edited": True}
    con = duckdb.connect()
    rows_db = con.execute(f"SELECT region, sum(cost), count(*) FROM read_parquet('{out}') GROUP BY region").fetchall()
    assert {r: (int(s), n) for r, s, n in rows_db} == want
    assert saved.column("region").to_pylist()[-50:] == ["Appended"] * 50

    # the memory: every phase under a third of the file (the process alone is ~20 MB; 83 MB at the save, 332 MB file), the cache within its limit
    assert res["cache_held"] <= cache_mib * 2**20
    for phase in ("edit_peak_kb", "query_peak_kb", "get_peak_kb", "save_peak_kb"):
        assert res[phase] * 3 < size_kb, (phase, res[phase], size_kb)
    assert res["save_peak_kb"] < 120_000, res["save_peak_kb"]


def small_edit_file(path, day_type=None, revenue_type=None, metadata=None, rows=2000):
    """The probe's five columns (region, carrier, day, revenue, cost) in 500-row groups, with the types asked for."""
    day_type = day_type or pa.date32()
    revenue_type = revenue_type or pa.decimal128(18, 2)
    r = _random.Random(5)
    days = [_dt.date(2020, 1, 1) + _dt.timedelta(days=i % 300) for i in range(rows)]
    if pa.types.is_timestamp(day_type):
        day = pa.array([_dt.datetime(d.year, d.month, d.day, 13, 45) for d in days], day_type)
    else:
        day = pa.array(days, day_type)
    t = pa.table({"region": pa.array([WORDS[i % 3] for i in range(rows)], pa.string()),
                  "carrier": pa.array([i % 9 for i in range(rows)], pa.int64()), "day": day,
                  "revenue": pa.array([_dec.Decimal(r.randint(-10**6, 10**6)).scaleb(-2) for _ in range(rows)], revenue_type),
                  "cost": pa.array([r.randint(-5000, 5000) for _ in range(rows)], pa.int64())})
    if metadata:
        t = t.replace_schema_metadata(metadata)
    pq.write_table(t, path, row_group_size=500, compression="snappy")
    return t


def test_saving_keeps_foreign_key_value_metadata(editor, tmp_path):
    path, out = tmp_path / "kv.parquet", tmp_path / "kv.out.parquet"
    small_edit_file(path, metadata={"other": "keep me", "omx": "{\"old\": 1}"})
    run_edit_probe(editor, path, out, 4)
    md = pq.read_table(out).schema.metadata
    assert md[b"other"] == b"keep me"
    assert json.loads(md[b"omx"]) == {"edited": True}              # omx is the caller's, not the source's


def test_a_type_change_that_keeps_every_value_saves_without_asking(editor, tmp_path):
    """DECIMAL(9, 2) is held as DECIMAL(18, 2): no value is lost, so save writes it (pandas / pyarrow write decimals at
    the data's precision, so refusing would refuse most foreign decimal files)."""
    path, out = tmp_path / "d.parquet", tmp_path / "d.out.parquet"
    small_edit_file(path, revenue_type=pa.decimal128(9, 2))
    run_edit_probe(editor, path, out, 4)
    saved = pq.read_table(out)
    assert saved.schema.field("revenue").type == pa.decimal128(18, 2)
    assert len(saved) > 0


@pytest.mark.parametrize("column,kind,lost", [
    ("day", pa.timestamp("us"), "time of day"),
])
def test_a_save_that_would_lose_what_reading_narrowed_is_refused_unless_allowed(editor, tmp_path, column, kind, lost):
    path, out = tmp_path / "n.parquet", tmp_path / "n.out.parquet"
    small_edit_file(path, **{"day_type" if column == "day" else "revenue_type": kind})
    p = subprocess.run([str(editor), str(path), str(out), "4"], capture_output=True, text=True)
    said = p.stdout + p.stderr
    assert p.returncode != 0
    assert f"column `{column}`" in said and lost in said and "allow_narrowing=True" in said, said
    assert not out.exists() and not list(tmp_path.glob("*.tmp"))        # nothing written, no temp file left
    run_edit_probe(editor, path, out, 4, "allow")
    saved = pq.read_table(out)
    assert saved.schema.field(column).type == (pa.date32() if column == "day" else pa.decimal128(18, 2))


@pytest.fixture(scope="module")
def edger(tmp_path_factory):
    out = tmp_path_factory.mktemp("mojo") / "pq_edge_probe"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_edge_probe.mojo"), str(out)],
                   check=True, capture_output=True, text=True)
    return out


EDGE_ROWS = [0, 1, 999, 1000, 1001, 4095, 4096, 4097, 13336, 13337, 13338, 26673, 26674, 26675, 40010]


@pytest.mark.parametrize("mode,block", [(0, 0), (1, 1000), (1, 4096), (1, 64), (2, 0)])
def test_frame_edits_at_group_and_block_edges_of_a_pyarrow_file_with_odd_row_groups(edger, tmp_path, mode, block):
    """Three row groups of 13,337 rows (not a multiple of 8: the second and third start mid-byte), edits and removals on
    either side of every group start and block edge, queried under every scan loop: three groups are fewer than the
    cores of any machine that has more than three, so MORSELS_ON takes the row-range path there whatever the box, and the
    group path and AUTO give what numpy replaying the same edits gives."""
    t = scan_table(40_011)
    path = tmp_path / "t.parquet"
    pq.write_table(t, path, row_group_size=13_337, compression="snappy")
    assert [pq.read_metadata(path).row_group(i).num_rows for i in range(3)] == [13_337, 13_337, 13_337]
    p = subprocess.run([str(edger), str(path), str(mode), str(block)], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    got = json.loads(p.stdout.strip().split("\n")[-1])
    region = t.column("region").to_pylist()
    cost = t.column("cost").to_pylist()
    alive = [True] * len(region)
    for i, row in enumerate(EDGE_ROWS):
        cost[row] = 10_000_000 + i
        if i % 2 == 0:
            region[row] = "Edge"
    for i, row in enumerate(EDGE_ROWS):
        if i % 4 == 1:
            alive[row] = False
    want = {}
    for r, c, a in zip(region, cost, alive):
        if a and r is not None:
            s, n = want.get(r, (0, 0))
            want[r] = (s + c, n + 1)
    assert {w: (s, n) for w, s, n in zip(got["words"], got["sums"], got["counts"]) if n} == want


# ── F-146: the GPU for a streamed file ───────────────────────────────────────────────────────────────

def _has_gpu():
    import shutil
    return shutil.which("nvidia-smi") is not None and subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def probe_device(prober, path, query, device, **extra):
    env = {**__import__("os").environ, "MXFRAME_SCAN_DEVICE": device, **extra}
    p = subprocess.run([str(prober), str(path), "scan", query], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(p.stdout.strip().split("\n")[-1])


def same_numbers(a, b):
    for k in ("words", "kinds", "slot_of", "count_slot_of", "counts", "slots", "present"):
        assert a[k] == b[k], k


@pytest.mark.skipif(not _has_gpu(), reason="no GPU")
@pytest.mark.parametrize("name", ["zstd-dict-v1", "snappy-dict-v2", "none-plain-v1", "small-pages"])
def test_the_gpu_scan_equals_the_cpu_scan_and_duckdb_on_pyarrow_files(prober, tmp_path, name):
    """Nulls (a whole row group of them), empty row groups, dictionary fallback to PLAIN text, every codec and page shape:
    each query on the GPU gives, number for number, what the CPU scan, read_parquet and DuckDB give."""
    path = tmp_path / "t.parquet"
    write_with_empty_groups(path, scan_table(), **PYARROW_FILES[name])
    for q in SCAN_QUERIES:
        got = probe_device(prober, path, q, "gpu")
        assert got["device"] == 1 and got["reason"] == "", (q, got["device"], got["reason"])
        same_numbers(got, probe_device(prober, path, q, "cpu"))
        same_numbers(got, probe(prober, path, "read", q))
        check_against_duckdb(path, got, q)


@pytest.mark.skipif(not _has_gpu(), reason="no GPU")
def test_auto_on_a_small_file_is_the_cpu_and_says_why(prober, tmp_path):
    path = tmp_path / "t.parquet"
    pq.write_table(scan_table(), path, row_group_size=SCAN_GROUP)
    got = probe_device(prober, path, "by_region", "auto")
    assert got["device"] == 0 and "under AUTO's" in got["reason"], (got["device"], got["reason"])


def _process_gpu_mib(pid):
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    for line in out.strip().splitlines():
        p, mem = [x.strip() for x in line.split(",")]
        if p == str(pid):
            return int(mem)
    return 0


@pytest.mark.skipif(not _has_gpu(), reason="no GPU")
def test_gpu_scans_that_raise_give_their_device_context_back(tmp_path):
    """Every way out of a GPU scan frees its DeviceContext: the GPU memory the process holds after 1,000 scans that raise (an
    overflow in the kernel) is what it holds after one (the ~512 MiB CUDA context of the process is there either way; a scan
    that kept its context grew by ~0.15 MiB each, 234 MiB over 1,500). nvidia-smi is the observer: the leak is invisible to
    the process's own free-memory reading, the CUDA context being shared inside it."""
    import time
    probe = tmp_path / "gpu_leak_probe"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/gpu_leak_probe.mojo"), str(probe)],
                   check=True, capture_output=True, text=True)
    path = tmp_path / "ov.parquet"
    pq.write_table(pa.table({"v": pa.array([2**62] * 8 + [1] * 1000, pa.int64())}), path, row_group_size=200)
    held = {}
    for n in (1, 1000):
        pr = subprocess.Popen([str(probe), str(path), str(n)], stdout=subprocess.PIPE, text=True)
        assert pr.stdout.readline().strip() == f"failed {n}"
        time.sleep(0.5)
        held[n] = _process_gpu_mib(pr.pid)
        pr.wait()
    if held[1] == 0:
        pytest.skip("nvidia-smi cannot see this process's GPU memory (container, MIG or driver permissions)")
    assert held[1000] - held[1] < 60, f"GPU MiB held after 1 failed scan {held[1]}, after 1000: {held[1000]}"


# ── F-188: the pages a save makes at once follow a byte budget, not the thread count ─────────────────────

@pytest.fixture(scope="module", params=["libzstd", "mojo-codec"])
def budget_probe(request, tmp_path_factory):
    out = tmp_path_factory.mktemp("mojo") / "pq_write_budget_probe"
    env = dict(os.environ)
    if request.param == "mojo-codec":
        env["MXFRAME_BUILD_FLAGS"] = "-D MXFRAME_NO_CODECS"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_write_budget_probe.mojo"), str(out)],
                   check=True, capture_output=True, text=True, env=env)
    return out


def run_budget_probe(probe, tmp_path, rows, threads, budget, data):
    path = tmp_path / f"b{threads}_{budget}_{data}.parquet"
    p = subprocess.run([str(probe), str(rows), str(path), str(threads), str(budget), data], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    res = json.loads(p.stdout.strip().split("\n")[-1])
    res["growth_kb"] = res["peak_kb"] - res["before_kb"]
    return res


@pytest.mark.parametrize("data", ["random", "compressible"])
def test_a_save_with_many_threads_holds_pages_within_its_budget(budget_probe, tmp_path, data):
    """A 125 MB table in one row group saved on 24 threads: the memory the save adds stays within the budget it was given (plus 2 MB), for
    incompressible and for compressible values alike (the coder's match tables and buffers are counted in a page's cost), with libzstd and with
    the Mojo coder; the bytes written do not depend on the budget."""
    rows = 2_000_000
    MB = 2**20
    files = {}
    for budget in (1 * MB, 4 * MB, 16 * MB, 64 * MB):
        res = run_budget_probe(budget_probe, tmp_path, rows, 24, budget, data)
        assert res["table_kb"] == 125000
        assert res["growth_kb"] * 1024 <= budget + 2 * MB, (budget, res)
        files[budget] = (tmp_path / f"b24_{budget}_{data}.parquet").read_bytes()
    one = run_budget_probe(budget_probe, tmp_path, rows, 1, 4 * MB, data)
    assert one["growth_kb"] * 1024 <= 4 * MB + 2 * MB, one
    assert len(set(files.values())) == 1 and files[1 * MB] == (tmp_path / f"b1_{4 * MB}_{data}.parquet").read_bytes()


def test_without_a_bound_a_save_holds_the_pages_of_its_row_group(budget_probe, tmp_path):
    """The measure sees what the budget bounds: with a 1 GB budget the same save adds over 100 MB."""
    big = run_budget_probe(budget_probe, tmp_path, 2_000_000, 24, 1024 * 2**20, "random")
    assert big["growth_kb"] * 1024 > 100 * 2**20, big
