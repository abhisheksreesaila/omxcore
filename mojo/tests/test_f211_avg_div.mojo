# F-211: AVG's division is the platform's. DuckDB divides a HUGEINT total in a C++ `long double`: an x87 80-bit value on x86-64 (the total
# is upper * 2^64 + lower in 64 bits of mantissa, divided in 64 bits, then rounded to a double), a plain double on arm64 and Windows (the
# total is DuckDB's HUGEINT-to-double cast, divided by count * 10^scale as doubles). The two differ in the last bit for about 1 average in 2
# (arm64: the total rounds to a double first) and 1 in 2,000 (x86: the correctly rounded quotient, mostly). Every case below was run through
# DuckDB 1.5.1 on x86-64 Linux and DuckDB 1.5.6 on an Apple M4 (the same rows on both, ~/.cache/f211/gen_cases.py): the bits each gave are
# the expected values, so the test pins both formulas on whichever machine runs it, and `avg_div` must agree with this machine's DuckDB.
from std.sys import CompilationTarget
from omxcore.wide import parse_wide
from omxcore.sqleval import avg_div, avg_div_x87, avg_div_double


def fail(msg: String) raises:
    raise Error(msg)


def _case(total: String, n: Int, scale: Int, x86: UInt64, arm: UInt64) raises:
    var w = parse_wide(total)
    var num = w.value
    if not w.ok:
        fail("cannot read " + total)
    var want_x86 = Float64(from_bits=x86)
    var want_arm = Float64(from_bits=arm)
    var what = " (total " + total + ", " + String(n) + " rows, scale " + String(scale) + ")"
    if avg_div_x87(num, n, scale) != want_x86:
        fail("the x87 formula gave " + String(avg_div_x87(num, n, scale)) + ", DuckDB on x86-64 gave " + String(want_x86) + what)
    if avg_div_double(num, n, scale) != want_arm:
        fail("the double formula gave " + String(avg_div_double(num, n, scale)) + ", DuckDB on arm64 gave " + String(want_arm) + what)
    comptime if CompilationTarget.is_x86():
        if avg_div(num, n, scale) != want_x86:
            fail("avg_div on x86-64 gave " + String(avg_div(num, n, scale)) + ", DuckDB gave " + String(want_x86) + what)
    else:
        if avg_div(num, n, scale) != want_arm:
            fail("avg_div off x86 gave " + String(avg_div(num, n, scale)) + ", DuckDB gave " + String(want_arm) + what)


def main() raises:
    # BIGINT totals (scale 0), past 2^53, past 2^63 and 2^64, both signs
    _case("18796004328173193453", 15, 0, 0x43B163C9C1B92A1E, 0x43B163C9C1B92A1F)
    _case("4917837881205379468", 39, 0, 0x437BFFDD55084A2F, 0x437BFFDD55084A2E)
    _case("-13838771659498541754", 24, 0, 0xC3A001197567ADF2, 0xC3A001197567ADF1)
    _case("-22133353314417395071", 38, 0, 0xC3A02A9A80F81BCE, 0xC3A02A9A80F81BCD)
    # DECIMAL(18, s) and DECIMAL(38, s): the total over count * 10^s
    _case("-2402317952048052700", 35, 2, 0xC303820BF73E8878, 0xC303820BF73E8879)
    _case("3981764578035373963", 37, 2, 0x430E960A662B3109, 0x430E960A662B310A)
    _case("3886900181356454190", 19, 2, 0x431D1258FDDE20F2, 0x431D1258FDDE20F3)
    _case("2133558934406808184", 13, 9, 0x41A3908ADC1005A1, 0x41A3908ADC1005A0)
    _case("565424162883621530", 16, 9, 0x4180D9D811711A83, 0x4180D9D811711A82)
    _case("920543302662353902188955268100168846", 2, 2, 0x46EC5DCE651C740B, 0x46EC5DCE651C740A)
    _case("5336336221648829673375280449350192801", 37, 2, 0x46D1C6F2B8CC5288, 0x46D1C6F2B8CC5289)
    _case("-1633974985940738680332544576017029401", 34, 2, 0xC6B7B1C89139BC02, 0xC6B7B1C89139BC01)
    _case("216765072367148763388482972378690982", 13, 6, 0x45CAF04D717FB975, 0x45CAF04D717FB974)
    _case("981488941722454878993373155792106837", 21, 6, 0x45E2E08D243137C5, 0x45E2E08D243137C4)
    # scales past 10^27 are not exact in a 64-bit mantissa: DuckDB on x86 rounds the power of ten
    _case("379834077384666650781989696195494038", 14, 25, 0x41E436D42117EC7E, 0x41E436D42117EC7E)
    _case("-203220816755982520033511973753405239", 20, 25, 0xC1CE484249E3D42C, 0xC1CE484249E3D42E)
    _case("-5761195004335980423961485449693874993", 39, 25, 0xC20B83FAA7174D5A, 0xC20B83FAA7174D59)
    _case("157466328032436613841552271633429334", 3, 38, 0x3F413313E1C2122F, 0x3F413313E1C2122E)
    _case("-216903238109763768664790777584388603", 3, 38, 0xBF47B10DDB5704F6, 0xBF47B10DDB5704F7)
    _case("-3095360351231709745167924889994064394", 5, 38, 0xBF795B70EF4AC112, 0xBF795B70EF4AC113)
    print("avg_div: ok")
