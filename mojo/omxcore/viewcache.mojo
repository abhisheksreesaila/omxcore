"""The result cache of a catalog's named views (F-160).

A view's result is kept under a key made of its canonical plan text and the versions of everything it reads (`views.view_key`): a source's identity
and edit counter, a file's size and modification time, the keys of the views upstream of it. The same key means the same answer, so a read of the view
finds it here instead of running its SQL; an edit to an input changes the keys of the views below it and of no other, so exactly those run again.

Bounded by bytes: when the results kept pass `budget`, the least recently used go first; a result bigger than the budget is not kept. A catalog and every
copy of it (a query's working copy, a background job's) share one cache, so the table is guarded by a spin lock, held only while the lists change (never
while a view runs). `runs` counts the views that really ran and `hits` the reads served from here: tests count those, not time.
"""
from std.atomic import Atomic
from std.memory import Pointer
from std.ffi import external_call
from .table import SharedTable, TEXT
from .job import THREADS

comptime DEFAULT_CACHE_BYTES = 256 << 20


def table_bytes(t: SharedTable) -> Int:
    """About what a table holds in memory: 8 bytes (16 for a wide column) and a validity byte a value, and each dictionary word."""
    var n = 0
    for c in range(t.columns()):
        n += t.rows() * (17 if t.is_wide(c) else 9)       # (a total past int64, F-192, keeps a second word a value)
        if t.kind(c) == TEXT:
            var words = t.dictionary(c)
            for w in range(len(words)):
                n += words[w].byte_length() + 24
    return n


struct CacheEntry(Movable):
    var key: String
    var table: SharedTable
    var inputs: List[SharedTable]      # the tables the key names by identity: held, so an address in a key is never reused while the entry lives
    var bytes: Int
    var used: Int                      # the tick of the last read

    def __init__(out self, var key: String, table: SharedTable, var inputs: List[SharedTable], bytes: Int, used: Int):
        self.key = key^
        self.table = table
        self.inputs = inputs^
        self.bytes = bytes
        self.used = used


struct ResultCache(Movable):
    var entries: List[CacheEntry]
    var budget: Int
    var bytes: Int
    var tick: Int
    var runs: Int
    var hits: Int
    var lock: List[Int64]

    def __init__(out self, budget: Int = DEFAULT_CACHE_BYTES):
        self.entries = List[CacheEntry]()
        self.budget = budget
        self.bytes = 0
        self.tick = 0
        self.runs = 0
        self.hits = 0
        self.lock = List[Int64](length=1, fill=0)

    def _lock(self) -> Pointer[Int64, MutUntrackedOrigin]:
        return Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(self.lock.unsafe_ptr()))

    def acquire(self):
        var l = self._lock()
        while True:
            var free = Int64(0)
            if Atomic[Scalar[DType.int64]].compare_exchange(l, free, 1):
                break
            comptime if THREADS:
                _ = external_call["sched_yield", Int32]()

    def release(self):
        Atomic[Scalar[DType.int64]].store(self._lock(), 0)

    def find(mut self, key: String) -> Optional[SharedTable]:
        """The kept result under `key` (counted as a hit and made the most recent), or None."""
        self.acquire()
        var out = Optional[SharedTable]()
        for i in range(len(self.entries)):
            if self.entries[i].key == key:
                self.tick += 1
                self.entries[i].used = self.tick
                self.hits += 1
                out = self.entries[i].table
                break
        self.release()
        return out

    def put(mut self, var key: String, table: SharedTable, var inputs: List[SharedTable]):
        """Keeps a result that was computed (counted as a run), then drops the least recently used past the budget."""
        self.acquire()
        self.runs += 1
        var bytes = table_bytes(table)
        if bytes <= self.budget:
            for i in range(len(self.entries)):
                if self.entries[i].key == key:
                    self.bytes -= self.entries[i].bytes
                    _ = self.entries.pop(i)
                    break
            self.tick += 1
            self.entries.append(CacheEntry(key^, table, inputs^, bytes, self.tick))
            self.bytes += bytes
            self._trim()
        self.release()

    def _trim(mut self):
        while self.bytes > self.budget and len(self.entries) > 0:
            var oldest = 0
            for i in range(1, len(self.entries)):
                if self.entries[i].used < self.entries[oldest].used:
                    oldest = i
            self.bytes -= self.entries[oldest].bytes
            _ = self.entries.pop(oldest)

    def set_budget(mut self, budget: Int):
        self.acquire()
        self.budget = budget
        self._trim()
        self.release()

    def clear(mut self):
        self.acquire()
        self.entries = List[CacheEntry]()
        self.bytes = 0
        self.release()
