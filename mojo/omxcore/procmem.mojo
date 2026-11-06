"""This process's memory, for the tests and tools that measure it (F-209). Not part of the API (not exported from the package):
`from omxcore.procmem import peak_kb`.

Linux reads /proc/self/status (VmHWM, VmRSS, RssAnon) and can restart the peak (clear_refs 5). macOS has getrusage (the peak,
in bytes: ru_maxrss) and mach's task_info (the resident size now); it has no anonymous / file split and no way to restart the
peak, so `anon_kb` is -1 and `reset_peak` returns False there. A test that needs either skips on macOS, saying so."""
from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
from std.sys import CompilationTarget


def _status_kb(key: String) raises -> Int:
    """One kB figure of /proc/self/status (Linux), or -1."""
    var text = String()
    with open("/proc/self/status", "r") as f:
        text = f.read()
    for line in text.split("\n"):
        if line.startswith(key):
            var digits = String()
            for b in line.as_bytes():
                if b >= 48 and b <= 57:
                    digits += chr(Int(b))
            return Int(digits)
    return -1


def peak_kb() raises -> Int:
    """The most resident memory this process has held, in kB (file pages included)."""
    comptime if CompilationTarget.is_linux():
        return _status_kb("VmHWM:")
    else:
        var ru = unsafe_alloc[Int64](18)                 # struct rusage: 144 bytes, ru_maxrss is the fifth long (bytes on macOS)
        for i in range(18):
            ru[i] = 0
        _ = external_call["getrusage", Int32](Int32(0), ru)
        var kb = Int(ru[4]) // 1024
        ru.free()
        return kb


def rss_kb() raises -> Int:
    """The memory this process holds now, in kB."""
    comptime if CompilationTarget.is_linux():
        return _status_kb("VmRSS:")
    else:
        var info = unsafe_alloc[UInt64](12)              # mach_task_basic_info: virtual_size, resident_size, resident_size_max, ...
        for i in range(12):
            info[i] = 0
        var count = unsafe_alloc[UInt32](1)
        count[0] = 12                                    # in 4-byte words
        var task = external_call["mach_task_self", UInt32]()
        var rc = external_call["task_info", Int32](task, Int32(20), info, count)   # MACH_TASK_BASIC_INFO
        var kb = Int(info[1]) // 1024 if rc == 0 else -1
        info.free()
        count.free()
        return kb


def anon_kb() raises -> Int:
    """The anonymous (not file-backed) part of the resident memory, in kB; -1 where the OS does not say (macOS)."""
    comptime if CompilationTarget.is_linux():
        return _status_kb("RssAnon:")
    else:
        return -1


def reset_peak() -> Bool:
    """Restart the peak from the current size (Linux); False where the OS cannot (macOS), so `peak_kb` stays the process's all-time peak."""
    comptime if CompilationTarget.is_linux():
        try:
            with open("/proc/self/clear_refs", "w") as f:
                f.write("5")
            return True
        except:
            return False
    else:
        return False
