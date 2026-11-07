"""Background jobs (F-175, docs/omxcore-mojo-api.md "Background jobs"): run any query, read or save off the caller's thread.

A `Job[T]` is the handle of one running piece of work that will produce a `T`:

    var job = compute_job(query)          # jobs.mojo: compute_job, sql_job, read_csv_job, read_parquet_job, save_job, write_parquet_job
    while job.poll() == JOB_RUNNING:      # JOB_RUNNING, JOB_DONE, JOB_FAILED, JOB_CANCELLED
        draw(job.progress())              # rows done of rows, and the stage
    var r = job.result()                  # waits; raises the job's error if it failed, "cancelled" if it was cancelled

`cancel()` is answered at once (`poll()` says JOB_CANCELLED from then on) and the work stops at its next check, between
morsels, row groups, blocks or windows; a cancelled job never hands out a partial result, and a cancelled save leaves the
old file in place (the writer saves to a temporary file and renames it). Dropping the handle cancels the job and frees
everything the job owns: the inputs when the work ends, the result with the last of the two owners (the handle and the thread).

The work itself reads a `JobCtl` (a copyable word: the address of the job's control block, 0 for a plain call). Engine code
takes `ctl: JobCtl = JobCtl()` and calls `ctl.check()` (raises when cancelled), `ctl.add_total(n)` and `ctl.advance(n)`
(rows done, one atomic add per block, never per row) and `ctl.stage(s)`.

Memory ownership: the job's body (a struct, `JobBody`) owns its inputs; the control block and the result slot are reference
counted by hand (`W_REFS`: the handle and the thread). A job holds no address of the caller's memory.

Threads: a raw `pthread_create` thread per job (Linux, and macOS since F-210). The serial build (`-D MXFRAME_SERIAL`,
`kernels/_fpenv.mojo`) has no async runtime and, by default, no job threads: the job runs to its end inside `compute_job` and the handle
comes back done (or failed); `cancel()` then has nothing to stop; `-D MXFRAME_JOB_THREADS` gives it threads anyway (see `THREADS`). The completion callback runs on the job's thread (on the
caller's thread in the serial build) with the final state and the token given at the start; it is skipped when the handle was
dropped first.
"""
from std.atomic import Atomic
from std.ffi import external_call, c_int
from std.memory.alloc import unsafe_alloc
from std.sys import is_defined
from .kernels._fpenv import SERIAL, start_pool

# A job gets a thread of its own in the default build (Linux, macOS). The serial build (-D MXFRAME_SERIAL) runs a job inside its
# start call, unless the app opts in with -D MXFRAME_JOB_THREADS: the job then gets a pthread whose kernels still run in a loop (no async
# runtime), at the price of three libSystem calls the app's stub must list: pthread_create, pthread_join, pthread_detach.
comptime JOB_STACK_BYTES = 16 * 1024 * 1024        # a job thread's stack
comptime THREADS = (not SERIAL) or is_defined["MXFRAME_JOB_THREADS"]()
comptime JOBS_ARE_THREADED = THREADS     # public: an app can assert it (False: a job runs inside its start call and its callback runs before the handle is returned)

comptime JOB_RUNNING = 0
comptime JOB_DONE = 1
comptime JOB_FAILED = 2
comptime JOB_CANCELLED = 3

comptime CANCELLED_MESSAGE = "cancelled"

# the stages `progress().stage` names
comptime STAGE_START = 0
comptime STAGE_PLAN = 1
comptime STAGE_READ = 2
comptime STAGE_AGGREGATE = 3
comptime STAGE_WRITE = 4
comptime STAGE_FINISH = 5
comptime STAGE_SORT = 6

# the control block: one Int64 per word, every access atomic
comptime W_STATE = 0
comptime W_CANCEL = 1
comptime W_DONE = 2
comptime W_TOTAL = 3
comptime W_STAGE = 4
comptime W_REFS = 5
comptime W_ABANDONED = 6
comptime W_BYTES = 7                 # progress counts bytes, not rows (a CSV file's rows are not known before they are parsed)
comptime WORDS = 8

comptime CANCEL_NONE = 0             # W_CANCEL: nobody asked
comptime CANCEL_ASKED = 1            # cancel() or a dropped handle
comptime CANCEL_COMMITTED = 2        # a save passed its point of no return: a later cancel() does nothing

comptime _A = Atomic[Scalar[DType.int64]]
comptime _P = Pointer[Int64, MutUntrackedOrigin]


def stage_name(s: Int) -> String:
    if s == STAGE_START:
        return "starting"
    if s == STAGE_PLAN:
        return "planning"
    if s == STAGE_READ:
        return "reading"
    if s == STAGE_AGGREGATE:
        return "aggregating"
    if s == STAGE_WRITE:
        return "writing"
    if s == STAGE_FINISH:
        return "finishing"
    if s == STAGE_SORT:
        return "sorting"
    return "stage " + String(s)


def state_name(s: Int) -> String:
    if s == JOB_RUNNING:
        return "running"
    if s == JOB_DONE:
        return "done"
    if s == JOB_FAILED:
        return "failed"
    return "cancelled"


struct JobCtl(ImplicitlyCopyable):
    """What running work sees of its job. `JobCtl()` (no job) is never cancelled and counts nothing: every function that
    takes one runs unchanged as a plain call."""
    var words: Int

    def __init__(out self):
        self.words = 0

    def __init__(out self, words: Int):
        self.words = words

    @always_inline
    def active(self) -> Bool:
        return self.words != 0

    @always_inline
    def cancelled(self) -> Bool:
        if self.words == 0:
            return False
        return _A.load(_P(unsafe_from_address=self.words) + W_CANCEL) == CANCEL_ASKED

    def check(self) raises:
        """Raises when the job was cancelled: the work unwinds, freeing what it owns, and returns no result."""
        if self.cancelled():
            raise Error(CANCELLED_MESSAGE)

    def commit(self) raises:
        """The point of no return of a save (just before its file is renamed into place): raises if a cancel got in first, else
        from here on `cancel()` changes nothing and the job ends done. A cancelled save therefore always leaves the old file."""
        if self.words != 0:
            var expect = Int64(CANCEL_NONE)
            if not _A.compare_exchange(_P(unsafe_from_address=self.words) + W_CANCEL, expect, CANCEL_COMMITTED):
                if expect == CANCEL_ASKED:
                    raise Error(CANCELLED_MESSAGE)

    def stage(self, s: Int):
        if self.words != 0:
            _A.store(_P(unsafe_from_address=self.words) + W_STAGE, Int64(s))

    def add_total(self, n: Int):
        """The work grew by `n` rows (or bytes): called when a source is opened, so the total is known as soon as it can be."""
        if self.words != 0:
            _ = _A.fetch_add(_P(unsafe_from_address=self.words) + W_TOTAL, Int64(n))

    def advance(self, n: Int):
        """`n` more rows (or bytes) are done: one atomic add, called at block boundaries."""
        if self.words != 0:
            _ = _A.fetch_add(_P(unsafe_from_address=self.words) + W_DONE, Int64(n))

    def count_bytes(self):
        """Progress counts bytes from here on (a CSV file)."""
        if self.words != 0:
            _A.store(_P(unsafe_from_address=self.words) + W_BYTES, 1)


struct JobProgress(Copyable, Movable):
    """`done` of `total` rows (bytes when `bytes`), and the stage. `total` can grow while the job finds more work (a join reads
    its sides one after another); `done` only grows, and equals `total` when the job is done."""
    var done: Int
    var total: Int
    var stage: String
    var bytes: Bool

    def __init__(out self, done: Int, total: Int, stage: String, bytes: Bool):
        self.done = done
        self.total = total
        self.stage = stage
        self.bytes = bytes


trait JobBody(Movable, Deinitable):
    """The work of a job: it owns its inputs, `run` produces the result."""
    comptime Out: Movable & Deinitable

    def run(mut self, ctl: JobCtl) raises -> Self.Out:
        ...


struct _Out[T: Movable & Deinitable](Movable):
    var value: Optional[Self.T]
    var error: String

    def __init__(out self):
        self.value = None
        self.error = String()


trait JobDone(Movable, Deinitable):
    """A completion callback: a struct the app writes (it can hold whatever it needs, a token, an address) and hands to
    `start_job_then`. `done(state)` runs once when the work has ended, on the job's thread (the caller's in the serial
    build), with JOB_DONE, JOB_FAILED or JOB_CANCELLED; it is skipped when the handle was dropped first."""

    def done(mut self, state: Int):
        ...


struct NoCallback(JobDone):
    def __init__(out self):
        pass

    def done(mut self, state: Int):
        pass


struct _Task[B: JobBody, CB: JobDone](Movable):
    var body: Self.B
    var words: Int
    var out: Int                                  # the address of the result slot
    var cb: Self.CB

    def __init__(out self, var body: Self.B, words: Int, out_at: Int, var cb: Self.CB):
        self.body = body^
        self.words = words
        self.out = out_at
        self.cb = cb^


def _release[T: Movable & Deinitable](words: Int, out_at: Int):
    """One owner less; the last one frees the control block and the result slot."""
    var w = _P(unsafe_from_address=words)
    if _A.fetch_add(w + W_REFS, -1) == 1:
        var p = Pointer[_Out[T], MutUntrackedOrigin](unsafe_from_address=out_at)
        _ = p.take_pointee()
        p.free()
        w.free()


def _finish_state(w: _P, state: Int64) -> Int64:
    """Running becomes `state`, unless `cancel()` got there first; the state it ends in."""
    var expect = Int64(JOB_RUNNING)
    if _A.compare_exchange(w + W_STATE, expect, state):
        return state
    return expect


def _run_task[B: JobBody, CB: JobDone](tp: Pointer[_Task[B, CB], MutUntrackedOrigin]):
    """Runs the body, publishes its result and state, calls the callback, frees the body. Used by the thread and by the serial build."""
    var words = tp[].words
    var w = _P(unsafe_from_address=words)
    var out = Pointer[_Out[B.Out], MutUntrackedOrigin](unsafe_from_address=tp[].out)
    var ctl = JobCtl(words)
    var final = Int64(JOB_FAILED)
    try:
        var v = tp[].body.run(ctl)
        if ctl.cancelled():
            final = _finish_state(w, JOB_CANCELLED)
        else:
            if _A.load(w + W_DONE) < _A.load(w + W_TOTAL):         # (a LIMIT that stopped early, a skipped range: done is done)
                _A.store(w + W_DONE, _A.load(w + W_TOTAL))
            out[].value = v^
            final = _finish_state(w, JOB_DONE)
    except e:
        if ctl.cancelled():
            final = _finish_state(w, JOB_CANCELLED)
        else:
            out[].error = String(e)
            final = _finish_state(w, JOB_FAILED)
    if _A.load(w + W_ABANDONED) == 0:
        tp[].cb.done(Int(final))
    _ = tp.take_pointee()                         # the body and its inputs: freed as soon as the work ends
    tp.free()
    _release[B.Out](words, Int(out))


def _entry[B: JobBody, CB: JobDone](arg: Int) abi("C") -> Int:
    comptime if not SERIAL:
        start_pool()                              # a thread of ours has no async runtime yet
    _run_task[B, CB](Pointer[_Task[B, CB], MutUntrackedOrigin](unsafe_from_address=arg))
    return 0


struct Job[T: Movable & Deinitable](Movable):
    """The handle of a background job that produces a `T`. Dropping it cancels the job (see the module doc)."""
    var words: Int
    var out: Int
    var tid: Int
    var threaded: Bool
    var joined: Bool
    var taken: Bool

    def __init__(out self, words: Int, out_at: Int, tid: Int, threaded: Bool):
        self.words = words
        self.out = out_at
        self.tid = tid
        self.threaded = threaded
        self.joined = False
        self.taken = False

    def poll(self) -> Int:
        """JOB_RUNNING, JOB_DONE, JOB_FAILED or JOB_CANCELLED."""
        return Int(_A.load(_P(unsafe_from_address=self.words) + W_STATE))

    def wait(mut self) -> Int:
        """Blocks until the job's thread has ended; returns the state. After `cancel()` it waits for the work to stop at its next check."""
        comptime if THREADS:
            if self.threaded and not self.joined:
                _ = external_call["pthread_join", c_int](self.tid, Int(0))
                self.joined = True
        return self.poll()

    def result(mut self) raises -> Self.T:
        """Waits, then moves the result out (once). Raises the job's error when it failed, "cancelled" when it was."""
        var s = self.wait()
        if s == JOB_CANCELLED:
            raise Error(CANCELLED_MESSAGE)
        var slot = Pointer[_Out[Self.T], MutUntrackedOrigin](unsafe_from_address=self.out)
        if s == JOB_FAILED:
            raise Error(slot[].error)
        if self.taken:
            raise Error("the result was already taken")
        self.taken = True
        return slot[].value.take()

    def cancel(self):
        """Asks the work to stop at its next check. The state is JOB_CANCELLED from this call on, unless the job had already ended
        (or a save had passed `commit()`: its file is being moved into place and the job ends done)."""
        var w = _P(unsafe_from_address=self.words)
        var expect = Int64(CANCEL_NONE)
        if _A.compare_exchange(w + W_CANCEL, expect, CANCEL_ASKED) or expect == CANCEL_ASKED:
            _ = _finish_state(w, JOB_CANCELLED)

    def progress(self) -> JobProgress:
        var w = _P(unsafe_from_address=self.words)
        var total = Int(_A.load(w + W_TOTAL))
        var done = Int(_A.load(w + W_DONE))
        return JobProgress(min(done, total) if total > 0 else done, total, stage_name(Int(_A.load(w + W_STAGE))), _A.load(w + W_BYTES) != 0)

    def __del__(deinit self):
        var w = _P(unsafe_from_address=self.words)
        _A.store(w + W_ABANDONED, 1)
        var expect = Int64(CANCEL_NONE)
        _ = _A.compare_exchange(w + W_CANCEL, expect, CANCEL_ASKED)
        comptime if THREADS:
            if self.threaded and not self.joined:
                _ = external_call["pthread_detach", c_int](self.tid)
        _release[Self.T](self.words, self.out)


def start_job[B: JobBody](var body: B) -> Job[B.Out]:
    """`start_job_then` without a callback."""
    return start_job_then(body^, NoCallback())


def start_job_then[B: JobBody, CB: JobDone](var body: B, var cb: CB) -> Job[B.Out]:
    """Starts `body` on a thread of its own (the serial build runs it here and now). `cb.done(state)` is called once when the
    work has ended, on the job's thread, unless the handle was dropped by then."""
    var words = unsafe_alloc[Int64](WORDS)
    for i in range(WORDS):
        words[i] = 0
    words[W_REFS] = 2                              # the handle and the work (the thread, or the call in the serial build)
    var slot = unsafe_alloc[_Out[B.Out]](1)
    slot.unsafe_write(_Out[B.Out]())
    var task = unsafe_alloc[_Task[B, CB]](1)
    task.unsafe_write(_Task[B, CB](body^, Int(words), Int(slot), cb^))
    comptime if not THREADS:
        _run_task[B, CB](task)
        return Job[B.Out](Int(words), Int(slot), 0, False)
    else:
        var tid = Int(0)
        # a stack of our own size: Apple's threads get 512 KB, and the SQL parser and the planner recurse
        var attr = unsafe_alloc[UInt8](256)
        for i in range(256):
            attr[i] = 0
        _ = external_call["pthread_attr_init", c_int](attr)
        _ = external_call["pthread_attr_setstacksize", c_int](attr, Int(JOB_STACK_BYTES))
        var rc = external_call["pthread_create", c_int](Pointer(to=tid), attr, _entry[B, CB], Int(task))
        _ = external_call["pthread_attr_destroy", c_int](attr)
        attr.free()
        if rc != 0:                                # no thread to be had: run it here
            _run_task[B, CB](task)
            return Job[B.Out](Int(words), Int(slot), 0, False)
        return Job[B.Out](Int(words), Int(slot), tid, True)
