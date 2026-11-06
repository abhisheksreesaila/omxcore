"""F-157: the CSV reader's FLOAT kind. A column becomes FLOAT when the first rows show an exponent, more than 4 decimal places,
or nan / inf beside numbers, or when the caller names it; the value is the correctly rounded float64 (compared with Python's float())."""
import os
import json
import random
import struct
import subprocess
from pathlib import Path

import pytest

from f157_data import bits

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
DUMP = BUILD / f"csv_dump{os.environ.get('PYTEST_XDIST_WORKER', '')}"   # one per xdist worker: never rebuilt while another runs it


@pytest.fixture(scope="module")
def dump_tool():
    lib = ROOT / ".pixi/envs/default/lib"
    subprocess.run(["bash", str(ROOT / "scripts/build_mojo_pkg.sh")], check=True, capture_output=True, cwd=ROOT)
    subprocess.run(["mojo", "build", "--disable-warnings", str(ROOT / "mojo/tools/csv_dump.mojo"), "-I", str(BUILD),
                    f"-Xlinker=-L{lib}", "-Xlinker=-lzstd", "-Xlinker=-lsnappy", "-Xlinker=-rpath", f"-Xlinker={lib}",
                    "-o", str(DUMP)], check=True, capture_output=True, cwd=ROOT)
    return DUMP


def dump(path, *extra):
    out = subprocess.run([str(DUMP), str(path), *map(str, extra)], check=True, capture_output=True, text=True).stdout
    lines = out.split("\n")
    return json.loads(lines[0]), [json.loads(x) for x in lines[1:] if x]


def write(tmp_path, header, cells):
    path = tmp_path / "f.csv"
    path.write_text(header + "\n" + "\n".join(cells) + "\n", encoding="utf8")
    return path


def test_the_rule_exponent_or_more_than_four_places_or_asked(dump_tool, tmp_path):
    cases = {
        "ints": (["1", "2", "3"], "INT", 0),
        "two places": (["1.5", "2.25", "3"], "DECIMAL", 2),
        "four places": (["1.5", "2.2501", "3"], "DECIMAL", 4),
        "four places and zeros": (["1.5", "2.250000", "3"], "DECIMAL", 2),
        "five places": (["1.5", "2.25001", "3"], "FLOAT", 0),
        "nine places": (["1.5", "2.123456789", "3"], "FLOAT", 0),
        "ten places": (["1.5", "2.1234567891", "3"], "FLOAT", 0),
        "nine places that do not fit at their scale": (["1.5", "2.123456789", "99999999999"], "FLOAT", 0),
        "an exponent": (["1", "2e3", "3"], "FLOAT", 0),
        "an upper case exponent": (["1.5E-3", "2", ""], "FLOAT", 0),
        "nan beside numbers": (["1.5", "nan", "3"], "FLOAT", 0),
        "inf beside numbers": (["1", "-Inf", "+infinity"], "FLOAT", 0),
        "only the word nan": (["nan", "NaN"], "TEXT", 0),
        "words": (["abc", "1e5"], "TEXT", 0),
        "an integer past int64": (["123456789012345678901234", "1"], "FLOAT", 0),
        "a decimal past int64": (["123456789012345678901234.5", "1"], "FLOAT", 0),
        "only a wide integer": (["123456789012345678901234"], "FLOAT", 0),
    }
    for name, (cells, kind, scale) in cases.items():
        head, rows = dump(write(tmp_path, "x", cells))
        assert head == [["x", kind, scale]], (name, head)


def test_asked_columns_are_float_whatever_the_rows_show(dump_tool, tmp_path):
    path = write(tmp_path, "a,b,c", ["1,2.5,x", "2,3,y", "3,,z"])
    head, rows = dump(path, 1, 4194304, 100000, "a,b")
    assert head == [["a", "FLOAT", 0], ["b", "FLOAT", 0], ["c", "TEXT", 0]]
    assert rows == [[bits(1.0), bits(2.5), "x"], [bits(2.0), bits(3.0), "y"], [bits(3.0), None, "z"]]


def test_a_cell_that_is_not_a_number_is_a_null_in_a_float_column(dump_tool, tmp_path):
    path = write(tmp_path, "a", ["1.5", "abc", "1e999", "-1e-999", "2"])
    head, rows = dump(path, 1, 4194304, 100000, "a")
    assert head == [["a", "FLOAT", 0]]
    assert rows == [[bits(1.5)], [None], [None], [bits(-0.0)], [bits(2.0)]]


def test_a_value_past_the_window_with_an_exponent_in_a_decimal_column_is_a_failed_cell(dump_tool, tmp_path):
    path = write(tmp_path, "a", ["1.5", "2.5", "3e2"])
    head, rows = dump(path, 1, 4194304, 2)                                # infer_rows = 2
    assert head == [["a", "DECIMAL", 2]]
    assert rows == [[150], [250], [None]]


def test_every_double_is_read_correctly_rounded(dump_tool, tmp_path):
    r = random.Random(5)
    cells = []
    for _ in range(40000):
        k = r.random()
        if k < 0.25:
            s = repr(struct.unpack("<d", struct.pack("<Q", r.getrandbits(64)))[0])      # any bit pattern, shortest repr
        elif k < 0.5:
            s = f"{r.uniform(-1, 1) * 10 ** r.randint(-40, 40):.{r.randint(0, 24)}e}"   # long mantissas, big exponents
        elif k < 0.7:
            s = f"{r.uniform(-1e6, 1e6):.{r.randint(5, 20)}f}"                          # many places
        elif k < 0.85:
            s = str(r.randint(0, 10 ** r.randint(1, 30))) + "." + str(r.randint(0, 10 ** r.randint(1, 30)))
        else:
            m = r.randint(2 ** 52, 2 ** 53)
            s = repr(float(m) + 0.5) + "e" + str(r.randint(-5, 5))                      # near halfway cases
        if "nan" in s or "inf" in s:
            continue
        cells.append(s)
    path = write(tmp_path, "x", ["1e0"] + cells)                                       # an exponent in the window makes it FLOAT
    for threads, chunk in ((1, 4194304), (4, 4096)):
        head, rows = dump(path, threads, chunk)
        assert head == [["x", "FLOAT", 0]]
        got = [row[0] for row in rows]
        want = [bits(float(c)) if abs(float(c)) != float("inf") else None for c in ["1e0"] + cells]
        assert got == want


def test_specials_and_signs(dump_tool, tmp_path):
    cells = ["-0.0", "0", "-0", "nan", "NaN", "-nan", "inf", "-Infinity", "+1.5", "5e-324", "2.2250738585072014e-308",
             "4.9406564584124654e-324", "1.7976931348623157e308", "1.7976931348623159e308", ".5", "5.", "-.5e1", "00012.50e+01"]
    path = write(tmp_path, "x", cells)
    head, rows = dump(path)
    assert head == [["x", "FLOAT", 0]]
    nan = bits(float("nan"))
    inf = bits(float("inf"))
    assert [r[0] for r in rows] == [bits(-0.0), bits(0.0), bits(-0.0), nan, nan, nan, inf, bits(-float("inf")), bits(1.5),
                                    bits(5e-324), bits(2.2250738585072014e-308), bits(5e-324), bits(1.7976931348623157e308),
                                    None, bits(0.5), bits(5.0), bits(-5.0), bits(125.0)]
