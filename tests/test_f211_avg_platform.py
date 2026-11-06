"""F-211: AVG of an integer is divided the way DuckDB divides it on THIS platform, and the formula is pinned on both.

DuckDB divides its HUGEINT total in a C++ `long double`: an x87 80-bit value on x86-64 (the Linux machines), a plain double on arm64
(Apple Silicon) and Windows. Measured, not assumed:
- x86-64, DuckDB 1.5.1: `_ld_div` (the total as a 64-bit-mantissa value, divided there, rounded to a double);
- arm64, DuckDB 1.5.6 on an Apple M4: 106 of the 108 differing cases of 400 random BIGINT groups were `double(total) / n` (the other two had totals past
  2^64, the HUGEINT cast `upper * 2^64 + lower` first), 0 were the correctly rounded quotient this engine used there before.
The pinned bits below were produced by DuckDB itself on each machine from the same rows (cases where the two formulas differ), so the
test holds on either machine whichever formula that machine's DuckDB uses, and the live tests compare with the oracle on the machine they run on."""
import numpy as np
import pytest
from fractions import Fraction

import omxcore.np_arrow as npa
from conftest import oracle_connection

# (HUGEINT total, row count, DuckDB 1.5.1 on x86-64, DuckDB 1.5.6 on arm64), as float64 bits
CASES = [
    (18796004328173193453, 15, 0x43B163C9C1B92A1E, 0x43B163C9C1B92A1F),
    (4917837881205379468, 39, 0x437BFFDD55084A2F, 0x437BFFDD55084A2E),
    (-13838771659498541754, 24, 0xC3A001197567ADF2, 0xC3A001197567ADF1),
    (-22133353314417395071, 38, 0xC3A02A9A80F81BCE, 0xC3A02A9A80F81BCD),
]


def _f(bits):
    return np.array([bits], dtype=np.uint64).view(np.float64)[0].item()


HAS_X87 = np.finfo(np.longdouble).nmant >= 63


@pytest.mark.parametrize("total,n,x86,arm", CASES)
def test_the_division_without_an_x87_is_what_duckdb_on_arm64_gave(monkeypatch, total, n, x86, arm):
    monkeypatch.setattr(npa, "_X87", False)
    assert npa._ld_div(total, n) == _f(arm)


@pytest.mark.skipif(not HAS_X87, reason="needs numpy's x87 long double (x86-64)")
@pytest.mark.parametrize("total,n,x86,arm", CASES)
def test_the_division_with_an_x87_is_what_duckdb_on_x86_gave(monkeypatch, total, n, x86, arm):
    monkeypatch.setattr(npa, "_X87", True)
    assert npa._ld_div(total, n) == _f(x86)


def test_the_two_formulas_differ_on_these_cases():
    """The cases are not vacuous: each differs between the platforms by one ulp."""
    assert all(x86 != arm and abs(x86 - arm) < 64 for _, _, x86, arm in CASES)


def _groups(seed, groups, per):
    r = np.random.default_rng(seed)
    return [[int(v) for v in r.integers(-(1 << 62), 1 << 62, per)] for _ in range(groups)]


def test_the_vectorised_division_of_int64_totals_is_this_platforms_duckdb():
    """Groups of BIGINTs whose total fits int64 (between 2^53 and 2^63): `int_mean_of_totals` equals DuckDB's AVG on this machine, and the groups really
    are ones where double(total) / n and the correctly rounded quotient differ (so a wrong formula cannot pass)."""
    con = oracle_connection()
    rows = [vals for vals in _groups(211, 400, 6) if abs(sum(vals)) < (1 << 63)]
    con.execute("create table t(g integer, v bigint)")
    con.executemany("insert into t values (?, ?)", [(i, v) for i, vals in enumerate(rows) for v in vals])
    want = dict(con.execute("select g, avg(v) from t group by g").fetchall())
    tot = np.array([sum(v) for v in rows], dtype=np.int64)
    cnt = np.array([len(v) for v in rows], dtype=np.int64)
    got = npa.int_mean_of_totals(tot, cnt)
    assert got.tolist() == [want[i] for i in range(len(rows))]
    assert sum(float(Fraction(int(t), int(c))) != float(t) / int(c) for t, c in zip(tot, cnt)) >= 3


def test_the_vectorised_division_without_an_x87_is_what_duckdb_on_arm64_gave(monkeypatch):
    """The same pinned bits through `int_mean_of_totals` (a total that fits int64, both signs and a count of 0), run on any machine."""
    monkeypatch.setattr(npa, "_X87", False)
    tot = np.array([4917837881205379468, -7, 7, -(1 << 62) - 12345, 0], dtype=np.int64)
    cnt = np.array([39, 2, 3, 6, 0], dtype=np.int64)
    got = npa.int_mean_of_totals(tot, cnt)
    assert got.tolist() == [_f(0x437BFFDD55084A2E), -3.5, 7 / 3, -float((1 << 62) + 12345) / 6, 0.0]
