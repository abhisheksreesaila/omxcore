"""Independent tasks on every thread the build has (F-187): `run_tasks(task, n, threads)` runs `task.run(0)` .. `task.run(n - 1)`,
each once, in any order, and returns when all are done. A task writes only its own slot of a result the caller made.

Which threads:
  - Linux and any build with the thread pool: `par_n` (kernels/_fpenv.mojo), the pool every kernel uses.
  - a serial build (`MXFRAME_SERIAL`, and every target but Linux) has no async runtime, so by default the tasks run in a loop
    on the calling thread: a plain Mac build stays serial.
  - a serial build with `-D MXFRAME_JOB_THREADS` (F-175) has pthreads: they are used here too, with the same libSystem calls
    the jobs already need (pthread_create, pthread_join, pthread_attr_init / _setstacksize / _destroy) and sysconf, which the
    package uses already. No async runtime is added.
`threads` caps the workers (0: the cores the build can use; 1: the calling thread only). The result never depends on it.
"""
from std.atomic import Atomic
from std.ffi import external_call, c_int
from std.memory.alloc import unsafe_alloc
from std.sys import CompilationTarget
from .kernels._fpenv import SERIAL, par_n, core_count
from .job import THREADS

comptime _A = Atomic[Scalar[DType.int64]]
comptime _P = Pointer[Int64, MutUntrackedOrigin]
comptime FAN_STACK_BYTES = 16 * 1024 * 1024        # a worker's stack (Apple's threads get 512 KB)
comptime FAN_MAX_THREADS = 16                      # pthread fan-out only: workers are made per call


trait ParTask(Copyable, Movable, Deinitable):
    """One piece of work in `n` independent parts."""

    def run(self, i: Int):
        ...


struct _Fan[T: ParTask](Movable):
    var task: Self.T
    var next: Int            # the address of the shared counter: the next task to take
    var n: Int

    def __init__(out self, var task: Self.T, next: Int, n: Int):
        self.task = task^
        self.next = next
        self.n = n


def _fan_loop[T: ParTask](fan: Pointer[_Fan[T], MutUntrackedOrigin]):
    var counter = _P(unsafe_from_address=fan[].next)
    while True:
        var i = Int(_A.fetch_add(counter, 1))
        if i >= fan[].n:
            return
        fan[].task.run(i)


def _fan_entry[T: ParTask](arg: Int) abi("C") -> Int:
    _fan_loop[T](Pointer[_Fan[T], MutUntrackedOrigin](unsafe_from_address=arg))
    return 0


def _cores() -> Int:
    """The cores of a build with pthreads and no pool: sysconf(_SC_NPROCESSORS_ONLN)."""
    comptime if CompilationTarget.is_linux():
        var n = external_call["sysconf", Int](Int32(84))
        return max(1, n)
    else:
        var n = external_call["sysconf", Int](Int32(58))             # Darwin's _SC_NPROCESSORS_ONLN
        return max(1, n)


def run_tasks[T: ParTask](task: T, n: Int, threads: Int = 0):
    """Runs `task.run(i)` for i in 0 ..< n on up to `threads` threads (0: all the build has); see the module doc."""
    if n <= 0:
        return
    comptime if not SERIAL:
        var workers = max(1, min(n, core_count() if threads <= 0 else threads))
        if workers == 1:
            for i in range(n):
                task.run(i)
            return

        def work(i: Int) {imm}:
            task.run(i)

        par_n(work, n, workers)
    elif not THREADS:
        for i in range(n):
            task.run(i)
    else:
        var workers = max(1, min(n, min(FAN_MAX_THREADS, _cores()) if threads <= 0 else threads))
        if workers == 1:
            for i in range(n):
                task.run(i)
            return
        var counter = unsafe_alloc[Int64](1)
        counter[0] = 0
        var fan = unsafe_alloc[_Fan[T]](1)
        fan.unsafe_write(_Fan[T](task.copy(), Int(counter), n))
        var attr = unsafe_alloc[UInt8](256)
        for i in range(256):
            attr[i] = 0
        _ = external_call["pthread_attr_init", c_int](attr)
        _ = external_call["pthread_attr_setstacksize", c_int](attr, Int(FAN_STACK_BYTES))
        var tids = unsafe_alloc[Int](workers)
        var started = 0
        for _ in range(workers - 1):                  # the calling thread is the last worker
            var tid = Int(0)
            var rc = external_call["pthread_create", c_int](Pointer(to=tid), attr, _fan_entry[T], Int(fan))
            if rc != 0:
                break                                 # no more threads to be had: the rest of the work falls to the ones running
            tids[started] = tid
            started += 1
        _ = external_call["pthread_attr_destroy", c_int](attr)
        attr.free()
        _fan_loop[T](Pointer[_Fan[T], MutUntrackedOrigin](unsafe_from_address=Int(fan)))
        for k in range(started):
            _ = external_call["pthread_join", c_int](tids[k], Int(0))
        tids.free()
        _ = fan.take_pointee()
        fan.free()
        counter.free()
