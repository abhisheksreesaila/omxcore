# F-178 test tool (tests/test_csv_scan_memory.py): runs one SQL string over a CSV file, either read whole or streamed with scan_csv, and
# prints one JSON line with the answer and the process's peak memory (VmHWM: the resident set at its highest, file pages included).
#
#   csv_scan_probe <table|scan> <path.csv> "<SQL>" [group_rows] [block_bytes]
#
# {"rows": N, "checksum": X, "row_groups": G, "open_ms": .., "query_ms": .., "vm_hwm_kb": .., "rss_anon_kb": .., "report_failed": F}
# checksum: the sum of every INT / DECIMAL / DATE / BOOL cell of the result (a TEXT cell adds the byte length of its word, so a wrong group
# or a wrong total changes it).
from std.sys import argv
from std.time import perf_counter_ns
from omxcore import Catalog, sql, read_csv, scan_csv, CsvOptions, TEXT, INT, DECIMAL, DATE, BOOL


def status_kb(key: String) raises -> Int:
    var text = String()
    with open("/proc/self/status", "r") as fh:
        text = fh.read()
    for line in text.split("\n"):
        if line.startswith(key):
            var digits = String()
            for b in line.as_bytes():
                if b >= 48 and b <= 57:
                    digits += chr(Int(b))
            return Int(digits)
    return -1


def main() raises:
    var args = argv()
    var mode = String(args[1])
    var path = String(args[2])
    var text = String(args[3])
    var options = CsvOptions()
    if len(args) > 4:
        options.group_rows = Int(String(args[4]))
    if len(args) > 5:
        options.block_bytes = Int(String(args[5]))
    if mode == "groups":                                     # decode every row group's first column and nothing else
        var scan2 = scan_csv(path, options)
        var one: List[Int] = [0]
        var total = 0
        for g in range(scan2.row_groups()):
            var part = scan2.read_group(g, one)
            total += part.rows()
        print("groups " + String(total) + " hwm " + String(status_kb("VmHWM:")) + " anon " + String(status_kb("RssAnon:")))
        return
    var cat = Catalog()
    var t0 = perf_counter_ns()
    var groups = 0
    var failed = 0
    if mode == "scan":
        var scan = scan_csv(path, options)
        groups = scan.row_groups()
        failed = scan.csv_report().failed_count()
        cat.add_scan("t", scan^)
    else:
        var r = read_csv(path, options)
        failed = 0
        for c in r.columns:
            failed += c.failed_count
        cat.add_table("t", r.take_table().share())
    var t1 = perf_counter_ns()
    var open_hwm = status_kb("VmHWM:")
    var res = sql(cat, text)
    var t2 = perf_counter_ns()
    var sumcheck = 0
    for c in range(res.table.columns()):
        var k = res.table.kind(c)
        for r in range(res.table.rows()):
            if res.table.is_null(c, r):
                continue
            if k == TEXT:
                sumcheck += res.table.text_of(c, Int(res.table.get(c, r))).byte_length()
            else:
                sumcheck += Int(res.table.get(c, r))
    print('{"rows": ' + String(res.table.rows()) + ', "checksum": ' + String(sumcheck) + ', "row_groups": ' + String(groups) +
          ', "open_ms": ' + String(Float64(t1 - t0) / 1e6) + ', "query_ms": ' + String(Float64(t2 - t1) / 1e6) +
          ', "open_hwm_kb": ' + String(open_hwm) + ', "vm_hwm_kb": ' + String(status_kb("VmHWM:")) + ', "rss_anon_kb": ' + String(status_kb("RssAnon:")) +
          ', "report_failed": ' + String(failed) + '}')
