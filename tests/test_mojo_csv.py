"""F-134: the Mojo CSV reader (mojo/omxcore/csv.mojo) against pyarrow on awkward files.

A small Mojo program (mojo/tools/csv_dump.mojo), built against the Python-free package, prints the parsed table as
JSON lines; pyarrow's reader is the oracle. Run with `pixi run pytest tests/test_mojo_csv.py`.
"""
import os
import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pv
import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
DUMP = BUILD / f"csv_dump{os.environ.get('PYTEST_XDIST_WORKER', '')}"   # one per xdist worker: never rebuilt while another runs it


@pytest.fixture(scope="session")
def dump_tool():
    lib = ROOT / ".pixi/envs/default/lib"
    subprocess.run(["bash", str(ROOT / "scripts/build_mojo_pkg.sh")], check=True, capture_output=True, cwd=ROOT)
    subprocess.run(["mojo", "build", "--disable-warnings", str(ROOT / "mojo/tools/csv_dump.mojo"), "-I", str(BUILD),
                    f"-Xlinker=-L{lib}", "-Xlinker=-lzstd", "-Xlinker=-lsnappy", "-Xlinker=-rpath", f"-Xlinker={lib}",
                    "-o", str(DUMP)], check=True, capture_output=True, cwd=ROOT)
    return DUMP


def mojo_read(path, *extra):
    out = subprocess.run([str(DUMP), str(path), *map(str, extra)], check=True, capture_output=True, text=True).stdout
    lines = out.split("\n")
    return json.loads(lines[0]), [json.loads(x) for x in lines[1:] if x]


def arrow_read(path):
    """pyarrow's view in the same shape: stored integers (decimals scaled, dates as days), text as strings."""
    table = pv.read_csv(path)
    cols = []
    for name in table.column_names:
        col = table[name]
        t = col.type
        if pa.types.is_integer(t) or pa.types.is_boolean(t):
            vals = [None if v is None else int(v) for v in col.to_pylist()]
        elif pa.types.is_floating(t):
            places = max([len(repr(v).split(".")[1].rstrip("0")) if "." in repr(v) else 0
                          for v in col.to_pylist() if v is not None] or [0])
            scale = 2 if places <= 2 else 4
            vals = [None if v is None else int(round(v * 10 ** scale)) for v in col.to_pylist()]
        elif pa.types.is_date(t) or pa.types.is_timestamp(t):
            vals = [None if v is None else v.toordinal() - 719163 for v in col.cast(pa.date32()).to_pylist()]
        else:
            vals = [None if v == "" else v for v in col.to_pylist()]  # an empty text cell is a null here
        cols.append(vals)
    return table, [list(r) for r in zip(*cols)] if cols else []


AWKWARD = {
    "plain.csv": "id,price,when,paid,city\n1,2.50,2024-01-05,true,Oslo\n2,10,2024-02-29,false,Rome\n3,,2023-12-31,true,\n",
    "quotes.csv": 'a,b,c\n"x,y","say ""hi""",1\n"line1\nline2","",2\n"",z,3\n',
    "crlf.csv": "a,b\r\n1,x\r\n2,y\r\n",
    "nonl.csv": "a,b\n1,x\n2,y",
    "bom.csv": "﻿id,name\n1,a\n2,b\n",
    "unicode.csv": "name,n\nZürich,1\n東京,2\nZürich,3\n",
    "stray_quote.csv": 'a,b\n5" pipe,x\n6,y\n7,z\n',
    "zones.csv": "t\n2023-01-01T23:30:00-05:00\n2023-01-01T00:30:00+05:00\n2023-01-01T12:00:00+00:00\n"
                 "2023-12-31T20:00:00-04:00\n2024-03-01T01:00:00+02:00\n",
    "negatives.csv": "a,b\n-1,-2.5\n3,0.25\n9007199254740993,1.5\n",
}


@pytest.mark.parametrize("name", sorted(AWKWARD))
@pytest.mark.parametrize("threads,chunk", [(1, 4194304), (4, 5)])
def test_awkward_files_equal_pyarrow(dump_tool, tmp_path, name, threads, chunk):
    path = tmp_path / name
    path.write_text(AWKWARD[name], encoding="utf8", newline="")
    head, rows = mojo_read(path, threads, chunk)
    table, expected = arrow_read(path)
    assert [h[0] for h in head] == table.column_names
    assert rows == expected


def test_generated_file_equals_pyarrow(dump_tool, tmp_path):
    path = tmp_path / "gen.csv"
    cities = ["Oslo", "Rome", '"Quito, EC"', "Bern"]
    with open(path, "w") as f:
        f.write("id,price,when,paid,city,qty\n")
        for i in range(60_000):
            f.write(f"{i},{(i * 37) % 100000 / 100:.2f},2024-{i % 12 + 1:02d}-{i % 28 + 1:02d},"
                    f"{'true' if i % 2 else 'false'},{cities[i % 4]},{'' if i % 10 == 0 else i % 50}\n")
    head, rows = mojo_read(path, 4, 50_000)
    _, expected = arrow_read(path)
    assert rows == expected
    assert [h[1] for h in head] == ["INT", "DECIMAL", "DATE", "BOOL", "TEXT", "INT"]
