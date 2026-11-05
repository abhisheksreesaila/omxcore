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
from std.memory.alloc import unsafe_alloc
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


struct _CacheState(Movable):
    """What the lock guards. It lives on the heap and is reached only through a raw pointer made after the lock is taken (see `ResultCache`)."""
    var entries: List[CacheEntry]
    var budget: Int
    var bytes: Int
    var tick: Int
    var runs: Int
    var hits: Int

    def __init__(out self, budget: Int):
        self.entries = List[CacheEntry]()
        self.budget = budget
        self.bytes = 0
        self.tick = 0
        self.runs = 0
        self.hits = 0


struct ResultCache(Movable):
    """Two heap blocks: the lock word and the `_CacheState` it guards, held here as addresses (F-199).

    The state is NOT a field of this struct on purpose. Two threads call these methods on the same cache at once, and a method that takes `mut self`
    (or reads fields through `self`) tells the compiler it is the only one touching them: it may then keep `self.entries` in registers or move its
    loads and stores across the lock's atomic operations, whatever fences stand there. That made `two_jobs_share_the_cache` crash about 3 runs in 150:
    a thread worked on a list another thread had grown and freed. A pointer made from an address after the lock is taken carries no such promise, so
    the lock orders every access to the state."""
    var state: Int
    var lock: Int

    def __init__(out self, budget: Int = DEFAULT_CACHE_BYTES):
        var st = unsafe_alloc[_CacheState](1)
        st.unsafe_write(_CacheState(budget))
        self.state = Int(st)
        var l = unsafe_alloc[Int64](1)
        l[0] = 0
        self.lock = Int(l)

    def __del__(deinit self):
        var st = Pointer[_CacheState, MutAnyOrigin](unsafe_from_address=self.state)
        _ = st.take_pointee()
        st.free()
        Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=self.lock).free()

    def _state(self) -> Pointer[_CacheState, MutAnyOrigin]:
        return Pointer[_CacheState, MutAnyOrigin](unsafe_from_address=self.state)

    def acquire(self):
        var l = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=self.lock)
        while True:
            var free = Int64(0)
            if Atomic[Scalar[DType.int64]].compare_exchange(l, free, 1):
                break
            comptime if THREADS:
                _ = external_call["sched_yield", Int32]()

    def release(self):
        Atomic[Scalar[DType.int64]].store(Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=self.lock), 0)

    def runs(self) -> Int:
        """How many views really ran (a result `put` here)."""
        self.acquire()
        var n = self._state()[].runs
        self.release()
        return n

    def hits(self) -> Int:
        """How many reads were served from here."""
        self.acquire()
        var n = self._state()[].hits
        self.release()
        return n

    def find(self, key: String) -> Optional[SharedTable]:
        """The kept result under `key` (counted as a hit and made the most recent), or None."""
        self.acquire()
        var st = self._state()
        var out = Optional[SharedTable]()
        for i in range(len(st[].entries)):
            if st[].entries[i].key == key:
                st[].tick += 1
                st[].entries[i].used = st[].tick
                st[].hits += 1
                out = st[].entries[i].table
                break
        self.release()
        return out

    def put(self, var key: String, table: SharedTable, var inputs: List[SharedTable]):
        """Keeps a result that was computed (counted as a run), then drops the least recently used past the budget."""
        self.acquire()
        var st = self._state()
        st[].runs += 1
        var bytes = table_bytes(table)
        if bytes <= st[].budget:
            for i in range(len(st[].entries)):
                if st[].entries[i].key == key:
                    st[].bytes -= st[].entries[i].bytes
                    _ = st[].entries.pop(i)
                    break
            st[].tick += 1
            st[].entries.append(CacheEntry(key^, table, inputs^, bytes, st[].tick))
            st[].bytes += bytes
            self._trim()
        self.release()

    def _trim(self):
        var st = self._state()
        while st[].bytes > st[].budget and len(st[].entries) > 0:
            var oldest = 0
            for i in range(1, len(st[].entries)):
                if st[].entries[i].used < st[].entries[oldest].used:
                    oldest = i
            st[].bytes -= st[].entries[oldest].bytes
            _ = st[].entries.pop(oldest)

    def set_budget(self, budget: Int):
        self.acquire()
        self._state()[].budget = budget
        self._trim()
        self.release()

    def clear(self):
        self.acquire()
        var st = self._state()
        st[].entries = List[CacheEntry]()
        st[].bytes = 0
        self.release()
