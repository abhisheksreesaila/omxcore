"""A profiler for when the `perf` tool is not installed (it is not on this machine) but the kernel allows user-space counters
(`/proc/sys/kernel/perf_event_paranoid` <= 2): it calls perf_event_open itself. Linux x86_64 only.

    python3 scripts/perf_sample.py counters ./binary [args]   # cycles, instructions (IPC), branch and cache misses of the whole run
    python3 scripts/perf_sample.py samples  ./binary [args]   # where the cycles go: hottest functions and hottest 16-byte blocks

For `samples` the binary must keep its symbols (a Mojo `mojo build` output does); look at a hot block with
`objdump -d --no-show-raw-insn -M intel --start-address=0x... --stop-address=0x... ./binary`. Set LD_LIBRARY_PATH as the binary needs.
Found with it (F-187): the zstd encoder ran at IPC 2.7, so it was instruction-bound, not waiting for memory or mispredicting
branches (1.6% of branches): prefetching and a smaller table did nothing; replacing List indexing and appends by raw pointers did.
"""
import bisect
import collections
import ctypes
import mmap
import os
import struct
import subprocess
import sys
import time

libc = ctypes.CDLL(None, use_errno=True)
SYS_PERF_EVENT_OPEN = 298
HARDWARE = 0
COUNTERS = [("cycles", 0), ("instructions", 1), ("branches", 4), ("branch-misses", 5), ("cache-refs", 2), ("cache-misses", 3)]


def open_event(config, pid, period=0, sample_ip=False):
    attr = bytearray(128)
    struct.pack_into("IIQQQQ", attr, 0, HARDWARE, 128, config, period, 1 if sample_ip else 0, 0)    # type, size, config, period, sample_type
    # flags: disabled, inherit (counters only), exclude_kernel, exclude_hv, enable_on_exec
    flags = (1 << 0) | (0 if sample_ip else 1 << 1) | (1 << 5) | (1 << 6) | (1 << 12)
    struct.pack_into("Q", attr, 40, flags)
    buf = (ctypes.c_char * 128).from_buffer(attr)
    return libc.syscall(SYS_PERF_EVENT_OPEN, ctypes.byref(buf), pid, -1, -1, 0)


def start(argv):
    """Forks the program, stopped until `go()`; returns (pid, pipe to release it)."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(w)
        os.read(r, 1)
        os.execv(argv[0], argv)
    return pid, w


def counters(argv):
    pid, go = start(argv)
    fds = {name: open_event(cfg, pid) for name, cfg in COUNTERS}
    os.write(go, b"x")
    os.waitpid(pid, 0)
    got = {name: struct.unpack("Q", os.read(fd, 8))[0] for name, fd in fds.items() if fd >= 0}
    for name, v in got.items():
        print("%-14s %16d" % (name, v))
    if "cycles" in got and "instructions" in got:
        print("IPC %.2f" % (got["instructions"] / got["cycles"]))
    if "branches" in got and "branch-misses" in got:
        print("branch miss rate %.2f%%" % (100.0 * got["branch-misses"] / got["branches"]))


def samples(argv, period=200000):
    pid, go = start(argv)
    fd = open_event(0, pid, period=period, sample_ip=True)
    if fd < 0:
        sys.exit("perf_event_open failed (errno %d): is perf_event_paranoid > 2?" % ctypes.get_errno())
    pages, page = 1 + 4096, mmap.PAGESIZE
    ring = mmap.mmap(fd, pages * page, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    os.write(go, b"x")
    exe, base = os.path.realpath(argv[0]), 0
    for _ in range(400):                                                        # the load address, once the program has been exec'd
        try:
            hits = [line for line in open("/proc/%d/maps" % pid) if exe in line]
        except OSError:
            break
        if hits:
            base = int(hits[0].split("-")[0], 16)
            break
        time.sleep(0.002)
    ips, tail, size = [], 0, (pages - 1) * page

    def drain():
        nonlocal tail
        head = struct.unpack_from("Q", ring, 1024)[0]                         # data_head
        while tail < head:
            kind, _misc, n = struct.unpack_from("IHH", ring, page + tail % size)
            if kind == 9:                                                       # PERF_RECORD_SAMPLE
                ips.append(struct.unpack_from("Q", ring, page + (tail + 8) % size)[0])
            tail += n
        struct.pack_into("Q", ring, 1032, tail)                                 # data_tail

    while True:
        done, _status = os.waitpid(pid, os.WNOHANG)
        drain()
        if done:
            break
        time.sleep(0.01)
    drain()
    syms = []
    for line in subprocess.run(["nm", "-n", "--defined-only", argv[0]], capture_output=True, text=True).stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[1] in "tTwW":
            syms.append((int(parts[0], 16), parts[2]))
    addrs = [a for a, _ in syms]
    by_fn, by_block = collections.Counter(), collections.Counter()
    for ip in ips:
        off = ip - base
        k = bisect.bisect_right(addrs, off) - 1
        by_fn[syms[k][1][:100] if k >= 0 else "?"] += 1
        by_block[off >> 4] += 1
    print("%d samples" % len(ips))
    for name, n in by_fn.most_common(12):
        print("%6.2f%% %s" % (100.0 * n / len(ips), name))
    print("--- hottest 16-byte blocks")
    for b, n in by_block.most_common(25):
        k = bisect.bisect_right(addrs, b << 4) - 1
        print("%6.2f%% 0x%x %s+0x%x" % (100.0 * n / len(ips), b << 4, syms[k][1][:50], (b << 4) - syms[k][0]))


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("counters", "samples"):
        sys.exit(__doc__)
    (counters if sys.argv[1] == "counters" else samples)(sys.argv[2:])
