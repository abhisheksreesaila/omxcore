# F-191: SQL honours the device. `PlannerOptions.device` is AUTO by default: the CPU unless AUTO's rule (a table of `gpu_min_rows` rows or
# more) picks the GPU; CPU never touches the GPU; GPU runs the single-table integer / DECIMAL aggregates the builder's GPU programs run and
# says what fell back. The oracle is the same query on the CPU (every cell equal). Tiny tables only: the GPUs are shared (`gpu_min_rows`
# is lowered instead of making a table big). The device-free checks run on any machine; the GPU ones skip with no GPU.
from omxcore import (Table, SharedTable, Catalog, PlannerOptions, GROUP_HASHED, sql, SqlResult, INT, DECIMAL, TEXT, FLOAT, CPU, GPU, AUTO,
                     gpu_available, DeviceTable)
from omxcore.gpu import GPU_BUDGET_ENV
from std.os import getenv, setenv, unsetenv


def fail(msg: String) raises:
    raise Error(msg)


def rows_of(t: Table) raises -> String:
    var out = String()
    for r in range(t.rows()):
        for c in range(t.columns()):
            var s = String()
            if t.is_null(c, r):
                s = "NULL"
            elif t.kind(c).value == TEXT.value:
                s = t.text_of(c, Int(t.get(c, r)))
            elif t.kind(c).value == FLOAT.value:
                s = String(Float64(from_bits=UInt64(t.get(c, r))))
            else:
                s = String(t.get(c, r))
            out += (" " if c > 0 else "") + s
        out += "\n"
    return out^


def notes_of(r: SqlResult) -> String:
    var all = String()
    for n in r.notes:
        all += n + "\n"
    return all^


def has_note(r: SqlResult, prefix: String) raises:
    for n in r.notes:
        if n.startswith(prefix):
            return
    fail("no note starts with `" + prefix + "`; the notes are:\n" + notes_of(r))


def table() raises -> SharedTable:
    var t = Table.empty()
    var region = t.add_column("region", TEXT)
    var qty = t.add_column("qty", INT)
    var price = t.add_column("price", DECIMAL, scale=2)
    var f = t.add_column("f", FLOAT)
    var regions: List[String] = ["West", "east", "North", "South"]
    var s = UInt64(12345)
    for i in range(300):
        s = s * 6364136223846793005 + 1442695040888963407
        var v = Int64((s >> 33) % 1000)
        t.append_text(region, regions[Int((s >> 20) % 4)])
        if i % 17 == 5:
            t.append_null(qty)
        else:
            t.append(qty, v % 11 - 3)
        t.append(price, v * 7 - 2000)
        t.append_float(f, Float64(Int(v)) / 4.0)
    return t^.share()


def table_shared(mut cat: Catalog) raises -> SharedTable:
    return cat.sources[0].table


def run(cat: Catalog, text: String, device: Int, min_rows: Int = 0, hashed: Bool = False) raises -> SqlResult:
    var c = cat.copy()
    var o = PlannerOptions()
    o.device = device
    o.gpu_min_rows = min_rows
    if hashed:
        o.group = GROUP_HASHED
    return sql(c, text, o)


def same_as_cpu(cat: Catalog, text: String, min_rows: Int = 0) raises:
    var want = run(cat, text, CPU)
    var got = run(cat, text, GPU, min_rows)
    if got.device != GPU:
        fail("`" + text + "` did not run on the GPU: " + got.fallback_reason + "\n" + notes_of(got))
    if rows_of(got.table) != rows_of(want.table):
        fail(text + "\n--- gpu ---\n" + rows_of(got.table) + "--- cpu ---\n" + rows_of(want.table))
    has_note(got, "device: GPU")
    var auto = run(cat, text, AUTO, min_rows)
    if auto.device != GPU or rows_of(auto.table) != rows_of(want.table):
        fail("AUTO with gpu_min_rows=" + String(min_rows) + " on `" + text + "`:\n" + notes_of(auto))


def main() raises:
    var cat = Catalog()
    cat.add_table("t", table())

    # default: AUTO, and AUTO answers on the CPU for a table under its row threshold, saying why
    if PlannerOptions().device != AUTO:
        fail("the default device is not AUTO")
    var q = "SELECT region, sum(qty), count(*), min(price), max(price) FROM t WHERE qty > 3 GROUP BY region ORDER BY region"
    var small = run(cat, q, AUTO, 1_000_000)
    if small.device != CPU:
        fail("AUTO ran a 300-row table on the GPU")
    has_note(small, "device: CPU (AUTO: the table has 300 rows, under the 1000000 AUTO runs on the GPU)")
    # CPU never touches the GPU
    var cpu = run(cat, q, CPU)
    if cpu.device != CPU:
        fail("device CPU reported " + String(cpu.device))
    has_note(cpu, "device: CPU (device CPU was asked for)")
    # a query the GPU programs do not run says so, whatever the device
    var fl = run(cat, "SELECT region, sum(f) FROM t GROUP BY region ORDER BY region", GPU)
    if fl.device != CPU:
        fail("a FLOAT sum ran on the GPU")
    has_note(fl, "device: CPU (GPU asked for; ")
    if not (fl.fallback_reason.byte_length() > 0):
        fail("no fallback_reason for a FLOAT sum on --device gpu")

    if not gpu_available():
        print("test_sql_gpu: no GPU, GPU checks skipped")
        return
    same_as_cpu(cat, q)
    same_as_cpu(cat, "SELECT sum(price * qty), count(qty), count(*) FROM t")
    same_as_cpu(cat, "SELECT region, sum(CASE WHEN qty > 4 THEN price ELSE 0 END), max(qty) FROM t GROUP BY region ORDER BY region")
    same_as_cpu(cat, "SELECT qty, sum(price), avg(qty) FROM t WHERE price BETWEEN 0 AND 3000 AND region <> 'West' GROUP BY qty ORDER BY qty")
    # hashed groups stay on the CPU, with the reason
    var h = run(cat, q, GPU, 0, True)
    if h.device != CPU:
        fail("hashed groups ran on the GPU")
    has_note(h, "device: CPU (GPU asked for; hashed groups")
    if rows_of(h.table) != rows_of(cpu.table):
        fail("hashed groups gave other rows")
    # a join: the joined rows are built on the CPU, their aggregate stage runs on the GPU and gives the CPU's rows
    same_as_cpu(cat, "SELECT a.region, sum(b.qty) FROM t a JOIN t b ON a.qty = b.qty GROUP BY a.region ORDER BY a.region")
    # AUTO's threshold is a number of rows: 300 rows run on the GPU at 300, on the CPU at 301
    var edge = run(cat, q, AUTO, 300)
    var over = run(cat, q, AUTO, 301)
    if edge.device != GPU or over.device != CPU:
        fail("AUTO at the threshold: " + String(edge.device) + " / " + String(over.device))
    # a DeviceTable already on the device is used as it is (not uploaded again)
    var c2 = cat.copy()
    var o = PlannerOptions()
    o.device = GPU
    o.device_table = DeviceTable(table_shared(c2))
    var attached = sql(c2, q, o)
    if attached.device != GPU:
        fail("attached DeviceTable: " + attached.fallback_reason)
    has_note(attached, "device: GPU (GPU asked for, the attached DeviceTable)")
    if rows_of(attached.table) != rows_of(cpu.table):
        fail("attached DeviceTable gave other rows")
    # a total past int64 is settled in 128 bits as on the CPU (the GPU raises the overflow, the query runs again split)
    var w = Table.empty()
    var k = w.add_column("k", INT)
    var v = w.add_column("v", INT)
    for i in range(6):
        w.append(k, Int64(i % 2))
        w.append(v, Int64(4_000_000_000_000_000_000))
    var wc = Catalog()
    wc.add_table("w", w^.share())
    var wq = "SELECT k, sum(v) FROM w GROUP BY k ORDER BY k"
    var wcpu = run(wc, wq, CPU)
    var wgpu = run(wc, wq, GPU)
    if wgpu.device != GPU or rows_of(wgpu.table) != rows_of(wcpu.table):
        fail("wide sums on the GPU:\n" + notes_of(wgpu) + rows_of(wgpu.table) + "--- cpu ---\n" + rows_of(wcpu.table))
    # a table over the (faked) GPU budget runs in pieces and gives the CPU's rows
    var big = Table.empty()
    var bk = big.add_column("k", INT)
    var bv = big.add_column("v", INT)
    var bp = big.add_column("p", DECIMAL, scale=2)
    for i in range(60_000):
        big.append(bk, Int64(i % 7))
        big.append(bv, Int64(i % 1013))
        big.append(bp, Int64(i % 977) * 3)
    var bc = Catalog()
    bc.add_table("b", big^.share())
    var bq = "SELECT k, sum(v), count(*), max(p) FROM b WHERE v > 10 GROUP BY k ORDER BY k"
    var saved = getenv(GPU_BUDGET_ENV, "")
    _ = setenv(GPU_BUDGET_ENV, "1M")
    var bcpu = run(bc, bq, CPU)
    var bgpu = run(bc, bq, GPU)
    if saved.byte_length() > 0:
        _ = setenv(GPU_BUDGET_ENV, saved)
    else:
        _ = unsetenv(GPU_BUDGET_ENV)
    if bgpu.device != GPU or rows_of(bgpu.table) != rows_of(bcpu.table):
        fail("pieces:\n" + notes_of(bgpu) + rows_of(bgpu.table) + "--- cpu ---\n" + rows_of(bcpu.table))
    print("test_sql_gpu: ok")
