"""F-180: zstd and snappy written in Mojo (mojo/omxcore/codec.mojo), the codecs of every build with no C library (macOS, iOS).

The recoder is built with -D MXFRAME_NO_CODECS, the switch every non-Linux target has by default: it links neither libzstd nor
libsnappy. It reads a file written by pyarrow (so the Mojo decoder meets what another writer produces: dictionary pages, data
page versions 1 and 2, nulls) and writes it back with the Mojo encoder. pyarrow and DuckDB must then read the same values
back, from a file whose column chunks say ZSTD or SNAPPY, and the file must really be smaller where the data allows."""
import datetime
import os
import random
import subprocess
from decimal import Decimal
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def recode(tmp_path_factory):
    """pq_recode built the way an Apple app builds the package: the Mojo codecs, no C library."""
    out = tmp_path_factory.mktemp("mojo-codec") / "pq_recode"
    env = dict(os.environ, MXFRAME_BUILD_FLAGS="-D MXFRAME_NO_CODECS")
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_recode.mojo"), str(out)],
                   check=True, capture_output=True, text=True, env=env)
    # the build really has no codec symbol to call
    assert b"ZSTD_compress" not in subprocess.run(["nm", "-D", str(out)], capture_output=True).stdout
    assert b"snappy_compress" not in subprocess.run(["nm", "-D", str(out)], capture_output=True).stdout
    return out


WORDS = ["West", "East", "North", "South-East", "Nörth", "", "a longer region name that repeats"]


def schema_table(n, seed=1, nulls=True, kind="mixed"):
    """A table of every column type omxcore stores, with the types it reads back (so Table.equals compares like with like)."""
    rnd = random.Random(seed)

    def nullable(f, every):
        return [None if nulls and i % every == 2 else f(i) for i in range(n)]

    if kind == "random":                      # nothing for a codec to find
        ints = [rnd.randrange(-(1 << 63), 1 << 63) for _ in range(n)]
        dbl = [rnd.random() * 1e6 for _ in range(n)]
        txt = [os.urandom(12).hex() for _ in range(n)]
        dec = [Decimal(rnd.randrange(-10**15, 10**15)).scaleb(-2) for _ in range(n)]
        day = [datetime.date(1970, 1, 1) + datetime.timedelta(days=rnd.randrange(0, 40000)) for _ in range(n)]
        flag = [rnd.random() < 0.5 for _ in range(n)]
    elif kind == "repetitive":                # a codec should shrink it to a sliver
        ints = [7] * n
        dbl = [1.5] * n
        txt = ["constant text"] * n
        dec = [Decimal("12.34")] * n
        day = [datetime.date(2024, 5, 6)] * n
        flag = [True] * n
    else:                                     # like a table an app saves
        ints = [i * 7 - 3000 for i in range(n)]
        dbl = [(i % 1000) * 0.25 for i in range(n)]
        txt = [WORDS[(i * 31 + i // 7) % len(WORDS)] for i in range(n)]
        dec = [Decimal(i * 101 - 50_000).scaleb(-2) for i in range(n)]
        day = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i % 400) for i in range(n)]
        flag = [i % 3 == 0 for i in range(n)]
    return pa.table({
        "id": pa.array(ints, pa.int64()),
        "ratio": pa.array(nullable(lambda i: dbl[i], 13), pa.float64()),
        "region": pa.array(nullable(lambda i: txt[i], 17), pa.string()),
        "money": pa.array(nullable(lambda i: dec[i], 19), pa.decimal128(18, 2)),
        "day": pa.array(nullable(lambda i: day[i], 11), pa.date32()),
        "paid": pa.array(nullable(lambda i: flag[i], 7), pa.bool_()),
    })


CASES = {
    "mixed with nulls": lambda: schema_table(5000),
    "no nulls": lambda: schema_table(3000, nulls=False),
    "empty table": lambda: schema_table(0),
    "one row": lambda: schema_table(1),
    "two rows": lambda: schema_table(2),
    "incompressible": lambda: schema_table(4000, kind="random"),
    "highly repetitive": lambda: schema_table(20000, kind="repetitive"),
    "large pages": lambda: schema_table(1_200_000, nulls=False),
}
SOURCES = {
    "uncompressed": dict(compression="none"),
    "zstd v1": dict(compression="zstd", data_page_version="1.0"),
    "zstd v2 no dictionary": dict(compression="zstd", data_page_version="2.0", use_dictionary=False),
    "snappy v2": dict(compression="snappy", data_page_version="2.0"),
    "snappy small pages": dict(compression="snappy", data_page_size=2000),
}


def run(recode, src, dst, codec, group=1 << 20):
    subprocess.run([str(recode), str(src), str(dst), codec, str(group)], check=True, capture_output=True)


def same_table(got, want):
    """Equal in names, types and every value (a column without nulls comes back REQUIRED: nullability is not compared)."""
    assert [(f.name, f.type) for f in got.schema] == [(f.name, f.type) for f in want.schema]
    return got.cast(want.schema).equals(want)


def duck_rows(path):
    return duckdb.sql(f"select * from read_parquet('{path}')").arrow().read_all()


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("case", [c for c in CASES if c != "large pages"])
def test_pyarrow_and_duckdb_read_what_the_mojo_codecs_wrote(recode, tmp_path, case, source):
    table = CASES[case]()
    src = tmp_path / "src.parquet"
    pq.write_table(table, src, **SOURCES[source])
    for codec, name in (("zstd", "ZSTD"), ("snappy", "SNAPPY")):
        dst = tmp_path / f"{codec}.parquet"
        run(recode, src, dst, codec, group=1024)
        got = pq.read_table(dst)
        assert same_table(got, table), f"pyarrow reads other values from the {codec} file"
        if table.num_rows:
            assert {pq.read_metadata(dst).row_group(g).column(c).compression
                    for g in range(pq.read_metadata(dst).num_row_groups) for c in range(table.num_columns)} == {name}
        if table.num_rows:                                    # DuckDB reads a file with no rows differently; skip
            assert same_table(duck_rows(dst), table), f"DuckDB reads other values from the {codec} file"


@pytest.mark.parametrize("source", ["uncompressed", "zstd v1", "snappy v2"])
def test_large_pages(recode, tmp_path, source):
    """1.2 million rows in one row group: pages of megabytes, matches across 128 KB blocks."""
    table = CASES["large pages"]()
    src = tmp_path / "src.parquet"
    pq.write_table(table, src, row_group_size=1 << 21, **SOURCES[source])
    for codec in ("zstd", "snappy"):
        dst = tmp_path / f"{codec}.parquet"
        run(recode, src, dst, codec, group=1 << 21)
        assert pq.read_metadata(dst).num_row_groups == 1
        assert same_table(pq.read_table(dst), table)
        assert same_table(duck_rows(dst), table)


def test_sizes(recode, tmp_path):
    """Repetitive data shrinks to a sliver, structured data well below its plain size, random data is not made bigger."""
    sizes = {}
    for kind in ("repetitive", "mixed", "random"):
        table = schema_table(20000, kind=kind, nulls=False)
        src = tmp_path / f"{kind}.parquet"
        pq.write_table(table, src, compression="none")
        plain = tmp_path / f"{kind}-plain.parquet"                # the same file saved with no codec: pyarrow's own is dictionary coded
        run(recode, src, plain, "none")
        sizes[kind, "plain"] = plain.stat().st_size
        for codec in ("zstd", "snappy"):
            dst = tmp_path / f"{kind}-{codec}.parquet"
            run(recode, src, dst, codec)
            sizes[kind, codec] = dst.stat().st_size
    assert sizes["repetitive", "zstd"] * 50 < sizes["repetitive", "plain"], sizes
    assert sizes["repetitive", "snappy"] * 20 < sizes["repetitive", "plain"], sizes
    assert sizes["mixed", "zstd"] * 3 < sizes["mixed", "plain"], sizes
    assert sizes["mixed", "snappy"] * 2 < sizes["mixed", "plain"], sizes
    assert sizes["random", "zstd"] < sizes["random", "plain"] * 1.01, sizes
    assert sizes["random", "snappy"] < sizes["random", "plain"] * 1.01, sizes


def test_the_mojo_codecs_make_files_like_the_c_libraries(recode, tmp_path):
    """On Linux the default build uses libzstd / libsnappy: the same table saved by either reads back identically and the
    Mojo file is within a quarter of the C file in size."""
    c_build = tmp_path / "pq_recode_c"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_recode.mojo"), str(c_build), "--no-package"],
                   check=True, capture_output=True, text=True)
    table = schema_table(50000, nulls=False)
    src = tmp_path / "src.parquet"
    pq.write_table(table, src, compression="none")
    for codec in ("zstd", "snappy"):
        theirs, ours = tmp_path / f"c-{codec}.parquet", tmp_path / f"m-{codec}.parquet"
        run(c_build, src, theirs, codec)
        run(recode, src, ours, codec)
        assert pq.read_table(theirs).equals(pq.read_table(ours))
        assert ours.stat().st_size < theirs.stat().st_size * 1.25, (codec, ours.stat().st_size, theirs.stat().st_size)


def test_a_c_codec_file_opens_on_the_no_codec_build_and_back(recode, tmp_path):
    """A file saved on Linux (libzstd) opens on a Mac build and a file saved on the Mac build opens on Linux: the formats are
    the real ones, not a private dialect. Round trip through both builds ends where it began."""
    c_build = tmp_path / "pq_recode_c"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_recode.mojo"), str(c_build), "--no-package"],
                   check=True, capture_output=True, text=True)
    table = schema_table(7000)
    src = tmp_path / "src.parquet"
    pq.write_table(table, src, compression="none")
    a, b, c = tmp_path / "a.parquet", tmp_path / "b.parquet", tmp_path / "c.parquet"
    run(c_build, src, a, "zstd")          # libzstd writes
    run(recode, a, b, "snappy")           # the Mojo codecs read zstd and write snappy
    run(c_build, b, c, "zstd")            # libsnappy reads what the Mojo snappy wrote
    assert same_table(pq.read_table(c), table)
