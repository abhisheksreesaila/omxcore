# F-200: a job's thread and its caller ask for the same TEXT dictionary of one scan at once (`ParquetScan.ensure_words` builds it
# under a hand-made lock). One round is cheap; many rounds, each on a fresh scan, make a race a near certainty (F-199's lesson).
from std.atomic import Atomic
from std.memory.alloc import unsafe_alloc
from omxcore import JOBS_ARE_THREADED
from omxcore import (Table, TEXT, INT, write_parquet, scan_parquet, ParquetScan, JobCtl, JobBody, start_job, JOB_DONE, UNCOMPRESSED)

comptime WORDS = 1500


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


def make_file(path: String) raises:
    var t = Table.empty()
    var id = t.add_column("id", INT)
    var a = t.add_column("a", TEXT)
    var b = t.add_column("b", TEXT)
    for w in range(WORDS):
        _ = t.add_word(a, "alpha-" + String(w))
        _ = t.add_word(b, "beta-" + String(w))
    for i in range(3000):
        t.append(id, Int64(i))
        t.append(a, Int64(i % WORDS))
        t.append(b, Int64((i * 7) % WORDS))
    write_parquet(path, t, "{}", UNCOMPRESSED, 1000)


def digest(scan: ParquetScan) raises -> Int:
    """What a reader of both dictionaries sees: word counts, every word's text, and its code looked up by the text."""
    var h = 0
    var names: List[String] = ["a", "b"]
    var stems: List[String] = ["alpha-", "beta-"]
    for k in range(2):
        var c = scan.index(names[k])
        var n = scan.word_count(c)
        expect(n == WORDS, "word count " + String(n))
        h += n
        for w in range(WORDS):
            var text = scan.text_of(c, w)
            expect(text == stems[k] + String(w), "word " + String(w) + ": " + text)
            h += scan.code_of(c, text) * (w + 1) + text.byte_length()
    return h


struct Reader(JobBody):
    comptime Out = Int
    var scan: ParquetScan
    var gate: Int                    # the address of one word: the caller sets it, then both threads ask for the dictionary together

    def __init__(out self, var scan: ParquetScan, gate: Int):
        self.scan = scan^
        self.gate = gate

    def run(mut self, ctl: JobCtl) raises -> Int:
        var g = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=self.gate)
        while Atomic[Scalar[DType.int64]].load(g) == 0:
            pass
        return digest(self.scan)


def main() raises:
    comptime if not JOBS_ARE_THREADED:
        # The race is between the caller and a job's thread. Where a job runs inside start_job (the default macOS build) there is no
        # second thread, and the Reader would wait for a gate the caller only sets after start_job returns: it never would. The
        # threaded build of this test is what scripts/mojo_test.sh runs on macOS (-D MXFRAME_JOB_THREADS).
        print("skipped: jobs do not run on threads in this build, so there is nothing to race")
        return
    make_file("dict_race.parquet")
    var want = digest(scan_parquet("dict_race.parquet"))
    var gate = unsafe_alloc[Int64](1)
    for round in range(600):
        gate[0] = 0
        var g = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(gate))
        var scan = scan_parquet("dict_race.parquet")
        var job = start_job(Reader(scan.copy(), Int(gate)))
        Atomic[Scalar[DType.int64]].store(g, 1)
        var mine = digest(scan)
        expect(job.wait() == JOB_DONE, "the job finishes, round " + String(round))
        expect(mine == want, "caller's digest " + String(mine) + " want " + String(want) + ", round " + String(round))
        var theirs = job.result()
        expect(theirs == want, "job's digest " + String(theirs) + " want " + String(want) + ", round " + String(round))
    print("ok")
