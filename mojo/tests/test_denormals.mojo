# F-210 (after F-082): a kernel task reads a denormal float as a value, not as zero, on every CPU. Mojo's `parallelize` runs its
# tasks with flush-to-zero on (x86 MXCSR 0x9FE0; arm64 FPCR.FZ, measured on an M4), so every kernel launches its workers
# through kernels/_fpenv.mojo's `par`, which resets the control register at the start of each task. The numbers come
# from memory the compiler cannot see through, and the sum of two of them is the check: 5e-324 + 0.0 is 5e-324, and 0.0 if flushed.
from std.memory.alloc import unsafe_alloc
from omxcore.kernels._fpenv import par, par_n, sync_par, core_count


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


def main() raises:
    var vals = unsafe_alloc[Float64](2)
    vals[0] = 5e-324                      # the smallest denormal
    vals[1] = 0.0
    var n = 4 * core_count() + 3
    var seen = unsafe_alloc[Int64](n)
    for i in range(n):
        seen[i] = -1

    def task(i: Int) {mut seen, imm vals}:
        var s = vals[0] + vals[1]
        seen[i] = 1 if s > 0.0 else 0

    par(task, n)
    for i in range(n):
        expect(seen[i] == 1, "par: task " + String(i) + " read 5e-324 + 0.0 as zero")
    for i in range(n):
        seen[i] = -1
    par_n(task, n, 2)
    for i in range(n):
        expect(seen[i] == 1, "par_n: task " + String(i) + " read 5e-324 + 0.0 as zero")
    for i in range(n):
        seen[i] = -1
    sync_par(task, n)
    for i in range(n):
        expect(seen[i] == 1, "sync_par: task " + String(i) + " read 5e-324 + 0.0 as zero")
    print("denormals ok:", n, "tasks, 3 launchers")
    vals.free()
    seen.free()
