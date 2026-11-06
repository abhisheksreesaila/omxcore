# F-175: the job handle itself (mojo/omxcore/job.mojo) with made-up work: state, result, error, cancel, progress, callback,
# dropped handles, many jobs at once, and no memory left behind. The real work (queries, reads, saves) is test_jobs.mojo.
from std.atomic import Atomic
from std.ffi import external_call, c_int
from std.sys import CompilationTarget
from std.memory.alloc import unsafe_alloc
from omxcore import (Job, JobCtl, JobBody, JobDone, start_job, start_job_then, JOB_RUNNING, JOB_DONE, JOB_FAILED, JOB_CANCELLED, STAGE_AGGREGATE)
from omxcore.kernels._fpenv import SERIAL


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


def pause(us: Int):
    external_call["usleep", c_int](Int32(us))


def rss_kb() raises -> Int:
    """Resident memory of this process, in KB: /proc/self/statm on Linux, Mach task_info on macOS (F-210: jobs run on threads there too, so the test reaches it)."""
    comptime if CompilationTarget.is_linux():
        var f = open("/proc/self/statm", "r")
        var text = f.read()
        f.close()
        var parts = text.split(" ")
        return Int(String(parts[1])) * 4
    else:                                          # macOS: task_info(MACH_TASK_BASIC_INFO).resident_size
        var info = unsafe_alloc[Int64](8)
        for i in range(8):
            info[i] = 0
        var count = unsafe_alloc[Int32](1)
        count[0] = 12
        var task = external_call["mach_task_self", Int32]()
        _ = external_call["task_info", Int32](task, Int32(20), info, count)
        var kb = Int(info[1]) // 1024
        info.free()
        count.free()
        return kb


struct Counter(Movable):
    """Counts bodies that were freed: the address of one Int64 that outlives the jobs."""
    var at: Int

    def __init__(out self):
        self.at = Int(unsafe_alloc[Int64](1))
        Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=self.at)[] = 0

    def value(self) -> Int:
        return Int(Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=self.at)[])


struct Steps(JobBody):
    """`n` steps of `us` microseconds, each a cancel check and one row done; the result is the sum of the steps' numbers,
    carried in a ballast list so a leak would show in the memory."""
    comptime Out = Int
    var n: Int
    var us: Int
    var fail_at: Int
    var ballast: List[Int64]
    var freed: Int                          # the address of a counter bumped when this body is freed

    def __init__(out self, n: Int, us: Int, freed: Int, fail_at: Int = -1, ballast: Int = 0):
        self.n = n
        self.us = us
        self.fail_at = fail_at
        self.ballast = List[Int64](length=ballast, fill=1)
        self.freed = freed

    def run(mut self, ctl: JobCtl) raises -> Int:
        ctl.stage(STAGE_AGGREGATE)
        ctl.add_total(self.n)
        var s = 0
        for i in range(self.n):
            ctl.check()
            if i == self.fail_at:
                raise Error("step " + String(i) + " failed")
            if self.us > 0:
                pause(self.us)
            s += i
            ctl.advance(1)
        return s

    def __del__(deinit self):
        if self.freed != 0:
            _ = Atomic[Scalar[DType.int64]].fetch_add(Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=self.freed), 1)


struct Note(JobDone):
    """A callback that records the state it was called with, and how many times."""
    var at: Int

    def __init__(out self, at: Int):
        self.at = at

    def done(mut self, state: Int):
        var p = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=self.at)
        p[] = Int64(state)
        _ = Atomic[Scalar[DType.int64]].fetch_add(p + 1, 1)


def test_result_and_progress() raises:
    var c = Counter()
    var job = start_job(Steps(100, 200, c.at))
    var last = 0
    var seen_running = False
    while job.poll() == JOB_RUNNING:
        seen_running = True
        var p = job.progress()
        expect(p.done >= last, "progress only grows: " + String(p.done) + " after " + String(last))
        expect(p.done <= max(p.total, p.done) and p.stage != "", "progress names its stage")
        last = p.done
        pause(500)
    expect(job.poll() == JOB_DONE, "the job is done")
    var p = job.progress()
    expect(p.done == 100 and p.total == 100 and p.stage == "aggregating", "progress ends at the total: " + String(p.done) + "/" + String(p.total) + " " + p.stage)
    comptime if not SERIAL:
        expect(seen_running, "the caller saw it run (it was off the caller's thread)")
    expect(job.result() == 4950, "the result is the sum 0..99")
    var again = False
    try:
        _ = job.result()
    except e:
        again = String(e) == "the result was already taken"
    expect(again, "the result comes out once")
    _ = job^
    expect(c.value() == 1, "the body was freed once")


def test_failure() raises:
    var c = Counter()
    var job = start_job(Steps(50, 0, c.at, fail_at=7))
    expect(job.wait() == JOB_FAILED, "a failing job is JOB_FAILED")
    var msg = String()
    try:
        _ = job.result()
    except e:
        msg = String(e)
    expect(msg == "step 7 failed", "result() raises the job's own error: `" + msg + "`")


def test_cancel() raises:
    var c = Counter()
    var job = start_job(Steps(1_000_000, 100, c.at))
    pause(5000)
    var before = job.progress().done
    job.cancel()
    expect(job.poll() == JOB_CANCELLED, "cancel() answers at once")
    expect(job.wait() == JOB_CANCELLED, "and it stays cancelled when the work stops")
    var msg = String()
    try:
        _ = job.result()
    except e:
        msg = String(e)
    expect(msg == "cancelled", "a cancelled job has no result: `" + msg + "`")
    var p = job.progress()
    expect(p.done < 1_000_000 and p.done >= before, "the work stopped part way: " + String(p.done))
    _ = job^
    expect(c.value() == 1, "the body was freed")


struct Committer(JobBody):
    """A save's shape: work, the point of no return (`commit`), then the rename, which takes a while."""
    comptime Out = Int
    var work_us: Int
    var after_us: Int

    def __init__(out self, work_us: Int, after_us: Int):
        self.work_us = work_us
        self.after_us = after_us

    def run(mut self, ctl: JobCtl) raises -> Int:
        ctl.add_total(2)
        pause(self.work_us)
        ctl.check()
        ctl.commit()
        ctl.advance(1)                      # "committed": the test cancels after seeing this
        pause(self.after_us)
        ctl.advance(1)
        return 7


def test_cancel_after_commit_does_nothing() raises:
    var late = start_job(Committer(0, 100_000))
    while late.progress().done < 1:
        pause(200)
    late.cancel()                           # the file is being moved into place: too late to stop it
    expect(late.poll() == JOB_RUNNING, "a cancel after the point of no return changes nothing")
    expect(late.result() == 7 and late.poll() == JOB_DONE, "the job ends done with its result")
    var early = start_job(Committer(100_000, 0))
    pause(2000)
    early.cancel()                          # before the commit: the work stops there
    expect(early.wait() == JOB_CANCELLED and early.progress().done == 0, "a cancel before the commit stops the job before it")


def test_cancel_after_done_is_nothing() raises:
    var job = start_job(Steps(10, 0, 0))
    _ = job.wait()
    job.cancel()
    expect(job.poll() == JOB_DONE and job.result() == 45, "cancelling a finished job changes nothing")


def test_callback() raises:
    var rec = unsafe_alloc[Int64](2)
    rec[0] = -1
    rec[1] = 0
    var job = start_job_then(Steps(20, 100, 0), Note(Int(rec)))
    _ = job.wait()
    expect(rec[0] == JOB_DONE and rec[1] == 1, "the callback ran once with JOB_DONE: " + String(rec[0]) + " x" + String(rec[1]))
    var bad = start_job_then(Steps(20, 0, 0, fail_at=3), Note(Int(rec)))
    _ = bad.wait()
    expect(rec[0] == JOB_FAILED and rec[1] == 2, "... with JOB_FAILED for a failure")
    var slow = start_job_then(Steps(1_000_000, 100, 0), Note(Int(rec)))
    pause(2000)
    slow.cancel()
    _ = slow.wait()
    expect(rec[0] == JOB_CANCELLED and rec[1] == 3, "... and with JOB_CANCELLED once the work stopped")
    rec.free()


def test_drop_cancels_and_frees() raises:
    var c = Counter()
    var rec = unsafe_alloc[Int64](2)
    rec[0] = -1
    rec[1] = 0
    var before = rss_kb()
    for _ in range(300):
        var job = start_job_then(Steps(1_000_000, 50, c.at, ballast=100_000), Note(Int(rec)))
        pause(300)
        _ = job^                                   # dropped while running: cancels, frees when the work stops
    var waited = 0
    while c.value() < 300 and waited < 5000:       # (the threads end on their own)
        pause(1000)
        waited += 1
    expect(c.value() == 300, "every dropped job's body was freed: " + String(c.value()))
    expect(rec[1] == 0, "a dropped handle's callback is skipped")
    var grown = rss_kb() - before
    expect(grown < 40_000, "memory after 300 dropped jobs of 800 KB: grew " + String(grown) + " KB")
    rec.free()


def test_many_at_once() raises:
    var jobs = List[Job[Int]]()
    for k in range(16):
        jobs.append(start_job(Steps(200 + k, 100, 0)))
    var total = 0
    for k in range(16):
        total += jobs[k].result() - (199 + k) * (200 + k) // 2
    expect(total == 0, "sixteen jobs at once each give their own sum")


def main() raises:
    test_result_and_progress()
    test_failure()
    comptime if not SERIAL:
        test_cancel()
        test_cancel_after_commit_does_nothing()
        test_callback()
        test_drop_cancels_and_frees()
    test_cancel_after_done_is_nothing()
    test_many_at_once()
    print("test_job ok")
