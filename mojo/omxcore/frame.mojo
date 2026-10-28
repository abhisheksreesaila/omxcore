"""A session's view of a shared table: its edits, rows, merges and formula columns (docs/omxcore-mojo-api.md
sections 3, 4 and 6).

    var f = Frame(shared)
    f.set(c, row, 1250); f.set_text(c, row, "West")       # typed text is parsed by the column's kind
    var row = f.append(["12.50", "East"]); f.remove(row)
    f.undo(); f.redo(); f.edits                            # one log for sets, appends, removes, merges and formula steps
    var m = f.formula("margin = revenue - cost")           # decimal-aware, never materialised
    f.get(m, row); f.permutation(c, descending=False)
    Query(f).group_by(col("region"), groups=n).agg(sum(col("margin"))).compute()   # includes the edits

Nothing is copied from the shared table: an edit is a sparse cell, an appended row lives in the frame, a removed
row is a tombstone. Formula columns follow the stored ones (index stored + k); their expressions are inlined
where they are used, so an aggregation over a formula runs in the kernel like any expression. A merge folds text
values into one (three spellings of a customer): every reference to the column reads the merged code.

A Frame also sits over a Parquet file that is not in memory (F-147): `Frame(scan_parquet(path))` has this same API,
with the file's rows as the base. Edits are the same sparse cells; `get` / `is_null` read the row group that holds the
row through a bounded cache of decoded column chunks (`set_cache_limit`, 128 MiB by default); `Query(frame)` streams
the file with the overlay applied; `permutation` streams the column (O(rows) keys in memory); `bake` is refused (it
would hold the file) and `save(path)` streams the edited file out, a row group at a time.
"""
from std.collections import Dict, Set
from std.memory import ArcPointer
from .job import JobCtl, STAGE_WRITE, STAGE_SORT      # F-175
from .table import Table, SharedTable, Column, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .expr import (Expr, Node, col, least, greatest, N_COL, N_LIT, N_LITF, N_DIV, N_I2F, N_ADD, N_SUB, N_MUL, N_FLOORDIV, N_MOD, N_NEG,
                   N_LEAST, N_GREATEST, N_LT, N_LE, N_GT, N_GE, N_EQ, N_NE, N_AND, N_OR, N_NOT, N_BOOL2INT, N_SCALE_TO)
from .csv import _parse_int, _parse_decimal, _parse_date, _parse_bool, _parse_float, F_OK, F_RANGE, F_NAN_BITS
from .holistic import float_key
from .sortspill import SpillSort, sort_budget_bytes
from .bind import SortKey, decimal_to_float
from .sqleval import ValCol
from .sqlprog import DictAccess
from .parquet import ParquetScan, ParquetWriter, write_parquet, ZSTD, DEFAULT_ROW_GROUP, STREAM_WRITE_BUDGET

comptime SET = 0
comptime APPEND = 1
comptime REMOVE = 2
comptime MERGE = 3
comptime REPOINT = 4
comptime FORMULA = 5                # a formula added, replaced or removed: `row` indexes `swaps`, the list of formulas the step left behind
comptime NULL_EDIT = Int64(0x7FF8000000000001)   # marks an EDITED cell or an APPENDED value that is null (never a base cell: those are null by the
                                                 # validity bitmap, so a base INT / DECIMAL / DATE / FLOAT value may be any int64, this one too). `set`
                                                 # refuses it as a value. A quiet NaN's bits, so a FLOAT cell edited to exactly this NaN is stored as the
                                                 # canonical NaN. In the rows of an `Overlay` a query reads, nulls are marked with `Overlay.null_marker`,
                                                 # which is this value unless some changed row holds it as data.
comptime CACHE_BYTES = 128 * 1024 * 1024  # the default bound of a scan frame's cache of decoded column chunks


@fieldwise_init
struct Edit(Copyable, Movable):
    var kind: Int
    var col: Int
    var row: Int
    var old: Int64           # SET: the value before (NULL_EDIT: null); MERGE / REPOINT: the merged code
    var new: Int64
    var had: Bool            # SET: the cell was already edited before this edit


@fieldwise_init
struct Formula(Copyable, Movable):
    var name: String
    var text: String
    var expr: Expr           # references other columns by name (formulas are expanded where used)
    var scale: Int


def _pow10(k: Int) -> Int64:
    var v: Int64 = 1
    for _ in range(k):
        v *= 10
    return v


struct _ChunkCache(Movable):
    """What a Frame over a scan keeps of the file: decoded column chunks (one column of one row group) up to `limit`
    bytes, least recently used out first (the newest chunk is always kept, so a chunk bigger than the limit still
    reads), and the base values of the rows the session touched (`rows`, filled by `Frame.overlay`). A chunk lives in a
    slot of `tables`; `slot_of` finds a key's slot and a doubly linked list through the slots keeps the use order, so a
    lookup, a use and an eviction are all O(1)."""
    var slot_of: Dict[Int, Int]      # (group << 20) | column -> slot
    var tables: List[Table]          # per slot: one column (an empty table in a free slot)
    var keys: List[Int]              # per slot: its key
    var newer: List[Int]             # per slot: the next more recently used slot, -1 for the newest
    var older: List[Int]             # per slot: the next less recently used slot, -1 for the oldest
    var free: List[Int]              # slots with no chunk
    var newest: Int
    var oldest: Int
    var held: Int                    # bytes in `tables`
    var limit: Int
    var loads: Int                   # chunks decoded so far
    var rows: Dict[Int, List[Int64]] # touched row -> its stored values from the file, then one 0 / 1 per column: the cell is null

    def __init__(out self):
        self.slot_of = Dict[Int, Int]()
        self.tables = List[Table]()
        self.keys = List[Int]()
        self.newer = List[Int]()
        self.older = List[Int]()
        self.free = List[Int]()
        self.newest = -1
        self.oldest = -1
        self.held = 0
        self.limit = CACHE_BYTES
        self.loads = 0
        self.rows = Dict[Int, List[Int64]]()

    def _unlink(mut self, i: Int):
        var n = self.newer[i]
        var o = self.older[i]
        if o >= 0:
            self.newer[o] = n
        else:
            self.oldest = n
        if n >= 0:
            self.older[n] = o
        else:
            self.newest = o

    def _push_newest(mut self, i: Int):
        self.older[i] = self.newest
        self.newer[i] = -1
        if self.newest >= 0:
            self.newer[self.newest] = i
        else:
            self.oldest = i
        self.newest = i

    def evict(mut self, keep: Int):
        """Drops least recently used chunks until the rest fit `keep` bytes."""
        while self.oldest >= 0 and self.held > keep:
            var i = self.oldest
            self._unlink(i)
            self.held -= _chunk_bytes(self.tables[i])
            _ = self.slot_of.pop(self.keys[i], 0)
            self.tables[i] = Table.empty()
            self.free.append(i)

    def fetch(mut self, scan: ParquetScan, g: Int, c: Int) raises -> Int:
        """The slot in `tables` of column c of row group g, decoded now if it is not held."""
        var key = (g << 20) | c
        var hit = self.slot_of.get(key)
        if hit:
            var i = hit.value()
            if self.newest != i:
                self._unlink(i)
                self._push_newest(i)
            return i
        var t = scan.read_group(g, [c])
        var size = _chunk_bytes(t)
        self.evict(self.limit - size)
        var i: Int
        if len(self.free) > 0:
            i = self.free.pop()
            self.tables[i] = t^
            self.keys[i] = key
        else:
            i = len(self.tables)
            self.tables.append(t^)
            self.keys.append(key)
            self.newer.append(-1)
            self.older.append(-1)
        self.slot_of[key] = i
        self._push_newest(i)
        self.held += size
        self.loads += 1
        return i


def _chunk_bytes(t: Table) -> Int:
    return len(t.cols[0].values) * 8 + len(t.cols[0].validity)


struct Frame(Movable):
    var base: SharedTable                 # the shared table; over a scan, its schema (columns, no rows)
    var scan: Optional[ParquetScan]       # set: the base rows are this file's, read a row group at a time
    var nbase: Int                        # the base's rows
    var starts: List[Int]                 # a scan's first row of each row group, then its total
    var cache: ArcPointer[_ChunkCache]
    var cells: Dict[Int, Int64]           # (row << 20) | col -> edited value (NULL_EDIT: set to null)
    var added: List[List[Int64]]          # appended rows, per row a value per stored column (NULL_EDIT: null)
    var removed: Set[Int]
    var words: List[List[String]]         # per stored column: words this session added (codes after the dictionary)
    var word_codes: List[Dict[String, Int]]
    var aliases: Dict[Int, Int64]         # (col << 32) | code -> the code it was merged into (one step deep)
    var log: List[Edit]
    var edits: Int                        # moves on every edit and undo: hosts re-aggregate when it moves
    var formulas: List[Formula]
    var redo_log: List[Edit]              # undone steps, the next one to redo last; any new edit empties it
    var redo_rows: List[List[Int64]]      # the appended rows that undo took back, the next to redo last
    var swaps: List[List[Formula]]        # per formula step: the formulas on the other side of it (undo and redo swap them in)

    def __init__(out self, base: SharedTable):
        self.base = base
        self.scan = None
        self.nbase = base.rows()
        self.starts = List[Int]()
        self.cache = ArcPointer(_ChunkCache())
        self.cells = Dict[Int, Int64]()
        self.added = List[List[Int64]]()
        self.removed = Set[Int]()
        self.words = List[List[String]]()
        self.word_codes = List[Dict[String, Int]]()
        for _ in range(base.columns()):
            self.words.append(List[String]())
            self.word_codes.append(Dict[String, Int]())
        self.aliases = Dict[Int, Int64]()
        self.log = List[Edit]()
        self.edits = 0
        self.formulas = List[Formula]()
        self.redo_log = List[Edit]()
        self.redo_rows = List[List[Int64]]()
        self.swaps = List[List[Formula]]()

    def __init__(out self, scan: ParquetScan):
        """A session over a Parquet file that stays on disk: the same API as `Frame(shared)`."""
        self = Self(scan.schema)
        self.scan = scan.copy()
        self.nbase = scan.rows()
        var at = 0
        for g in range(scan.row_groups()):
            self.starts.append(at)
            at += scan.group_rows(g)
        self.starts.append(at)

    # ── shape ───────────────────────────────────────────────────────────────────────────────────────

    def stored(self) -> Int:
        return self.base.columns()

    def columns(self) -> Int:
        return self.stored() + len(self.formulas)

    def rows(self) -> Int:
        """Base rows plus appended rows (removed rows keep their numbers: ask `alive`)."""
        return self.nbase + len(self.added)

    def alive(self, row: Int) -> Bool:
        return 0 <= row < self.rows() and row not in self.removed

    def name(self, c: Int) -> String:
        return self.base.name(c) if c < self.stored() else self.formulas[c - self.stored()].name

    def kind(self, c: Int) -> ColumnKind:
        if c < self.stored():
            return self.base.kind(c)
        var fs = self.formulas[c - self.stored()].scale
        return FLOAT if fs < 0 else DECIMAL if fs > 0 else INT

    def scale(self, c: Int) -> Int:
        """A stored column's scale (a DECIMAL's places; a FLOAT's 0, or 32 for a column the file stored as 32-bit floats), a formula's
        places (0 for a FLOAT formula)."""
        return self.base.scale(c) if c < self.stored() else max(self.formulas[c - self.stored()].scale, 0)

    def index(self, name: String) -> Int:
        var c = self.base.index(name)
        if c >= 0:
            return c
        for k in range(len(self.formulas)):
            if self.formulas[k].name == name:
                return self.stored() + k
        return -1

    def column(self, name: String) raises -> Int:
        var c = self.index(name)
        if c < 0:
            raise Error("column `" + name + "` not found")
        return c

    # ── values ──────────────────────────────────────────────────────────────────────────────────────

    def _raw(self, c: Int, row: Int) raises -> Int64:
        """A stored column's value as edited (NULL_EDIT for null), before merges. Only for an edit's undo record and `raw_row`: a base value
        equal to NULL_EDIT is indistinguishable here, so everything that decides on nullness uses `_cell`."""
        var cell = self._cell(c, row)
        return NULL_EDIT if cell[1] else cell[0]

    def _group_of(self, row: Int) -> Int:
        """The row group of a scan that holds `row`."""
        var lo = 0
        var hi = len(self.starts) - 2
        while lo < hi:
            var mid = (lo + hi + 1) // 2
            if self.starts[mid] <= row:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _base_cell(self, c: Int, row: Int) raises -> Tuple[Int64, Bool]:
        """(value, null) of a base row from the file (or shared table): nulls come from the validity bitmap, never from the value. A scan: from
        the rows the session touched, else from the cache, which decodes the column chunk of the row's group on a miss."""
        if not self.scan:
            return (Int64(0), True) if self.base.is_null(c, row) else (self.base.get(c, row), False)
        ref cache = self.cache[]
        if row in cache.rows:
            return (cache.rows[row][c], cache.rows[row][self.stored() + c] != 0)
        var g = self._group_of(row)
        var i = cache.fetch(self.scan.value(), g, c)
        ref t = cache.tables[i]
        var r = row - self.starts[g]
        return (Int64(0), True) if t.is_null(0, r) else (t.get(0, r), False)

    def _cell(self, c: Int, row: Int) raises -> Tuple[Int64, Bool]:
        """(value, null) of a stored column's cell as edited, before merges."""
        var hit = self.cells.get((row << 20) | c)
        if hit:
            return (Int64(0), True) if hit.value() == NULL_EDIT else (hit.value(), False)
        if row < self.nbase:
            return self._base_cell(c, row)
        var v = self.added[row - self.nbase][c]
        return (Int64(0), True) if v == NULL_EDIT else (v, False)

    def set_cache_limit(mut self, bytes: Int):
        """Bounds the decoded column chunks a scan frame keeps for `get` and `is_null` (8 bytes a value plus the null
        bits; the chunk just read is always kept, so one chunk bigger than the limit still reads)."""
        ref cache = self.cache[]
        cache.limit = bytes
        cache.evict(bytes)

    def cache_held(self) -> Int:
        """Bytes of decoded column chunks the cache holds now."""
        return self.cache[].held

    def cache_loads(self) -> Int:
        """Column chunks decoded so far (a chunk read again after eviction counts again)."""
        return self.cache[].loads

    def _shared_words(self, c: Int) raises -> Int:
        """The words of TEXT column c in the shared table or the file (a file's are built on first use)."""
        if self.scan:
            return self.scan.value().word_count(c)
        return len(self.base.dictionary(c))

    def edited(self, c: Int, row: Int) -> Bool:
        """Does the cell hold a session edit (set, set_null or set_text) that undo has not taken back?"""
        return ((row << 20) | c) in self.cells

    def word_count(self, c: Int) raises -> Int:
        """A TEXT column's words: the shared dictionary's (a file's, built on first use) plus this session's (codes run
        0 up to this)."""
        return self._shared_words(c) + len(self.words[c])

    def is_null(self, c: Int, row: Int) raises -> Bool:
        if c < self.stored():
            return self._cell(c, row)[1]
        return self.evaluate(self.program(c), row)[1]

    def get(self, c: Int, row: Int) raises -> Int64:
        """The value as this session sees it: edited, merged; a formula evaluated (0 for a null: ask `is_null`)."""
        if c < self.stored():
            var cell = self._cell(c, row)
            return 0 if cell[1] else self.alias(c, cell[0])
        return self.evaluate(self.program(c), row)[0]

    def get_float(self, c: Int, row: Int) raises -> Float64:
        """A FLOAT cell (or a FLOAT formula's value) as a float64 (0.0 for a null: ask `is_null`)."""
        return Float64(from_bits=UInt64(self.get(c, row)))

    def text_of(self, c: Int, code: Int64) raises -> String:
        """The word of a TEXT code: the shared dictionary's, then this session's."""
        var n = self._shared_words(c)
        var k = Int(code)
        if k < n:
            return self.scan.value().text_of(c, k) if self.scan else self.base.text_of(c, k)
        return self.words[c][k - n] if k - n < len(self.words[c]) else "?"

    def code_of(self, c: Int, word: String) raises -> Int:
        var k = self.scan.value().code_of(c, word) if self.scan else self.base.code_of(c, word)
        if k >= 0:
            return k
        var mine = self.word_codes[c].get(word, -1)
        return self._shared_words(c) + mine if mine >= 0 else -1

    def parse(mut self, c: Int, text: String) raises -> Int64:
        """A stored value from typed text (TEXT: its code, added if new; empty: null)."""
        var s = String(text.strip())
        if s.byte_length() == 0:
            return NULL_EDIT
        var kind = self.base.kind(c)
        if kind == TEXT:
            var k = self.code_of(c, s)
            if k >= 0:
                return Int64(k)
            self.word_codes[c][s] = len(self.words[c])
            self.words[c].append(s)
            return Int64(self._shared_words(c) + len(self.words[c]) - 1)
        var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(s.unsafe_ptr()))
        var n = s.byte_length()
        var out: Int64 = 0
        var places = 0
        var ok: Bool
        if kind == FLOAT:
            var x = Float64(0.0)
            var st = _parse_float(p, n, x)
            _ = s.byte_length()
            if st == F_RANGE:
                raise Error("column `" + self.base.name(c) + "`: `" + s + "` is past the largest float64 (about 1.8e308)")
            if st != F_OK:
                raise Error("column `" + self.base.name(c) + "`: `" + s + "` isn't a number")
            return Int64(x.to_bits())
        if kind == DATE:
            ok = _parse_date(p, n, out)
        elif kind == BOOL:
            ok = _parse_bool(p, n, out)
        elif kind == DECIMAL:
            ok = _parse_decimal(p, n, self.base.scale(c), out, places)
        else:
            ok = _parse_int(p, n, out)
        _ = s.byte_length()                             # keep s alive past the parse (p points into it)
        if not ok:
            raise Error("column `" + self.base.name(c) + "`: `" + s + "` isn't " +
                        ("a date" if kind == DATE else "true or false" if kind == BOOL else "a number"))
        if out == NULL_EDIT:
            raise Error("column `" + self.base.name(c) + "`: `" + s + "` is reserved (it marks an empty cell)")
        return out

    # ── edits ───────────────────────────────────────────────────────────────────────────────────────

    def set(mut self, c: Int, row: Int, value: Int64) raises:
        """Sets a stored cell (a TEXT code, a scaled DECIMAL, days for a DATE, a FLOAT's bits). The value 9221120237041090561
        (0x7FF8000000000001, `NULL_EDIT`) is reserved as the mark of a null edit: use `set_null`. A base cell may hold it as data.""" 
        if value == NULL_EDIT:
            raise Error("the value " + String(value) + " is reserved for null; use set_null")
        self._set(c, row, value)

    def set_float(mut self, c: Int, row: Int, value: Float64) raises:
        """Sets a FLOAT cell. The bits are kept as given (-0.0, a NaN's payload), except the one NaN that marks a null edit."""
        self._check_cell(c, row)
        if self.base.kind(c) != FLOAT:
            raise Error("column `" + self.base.name(c) + "` is " + String(self.base.kind(c)) + ", not FLOAT")
        var bits = Int64(value.to_bits())
        self._set(c, row, Int64(F_NAN_BITS) if bits == NULL_EDIT else bits)

    def _check_cell(self, c: Int, row: Int) raises:
        if c < 0 or c >= self.stored():
            raise Error("only stored columns are edited (a formula follows its inputs)")
        if not self.alive(row):
            raise Error("row " + String(row) + " does not exist")

    def _set(mut self, c: Int, row: Int, value: Int64) raises:
        self._check_cell(c, row)
        if self.base.kind(c) == TEXT and value != NULL_EDIT and \
                (value < 0 or value >= Int64(self._shared_words(c) + len(self.words[c]))):
            raise Error("column `" + self.base.name(c) + "`: code " + String(value) + " is not one of its words")
        if self.base.kind(c) == BOOL and value != NULL_EDIT and value != 0 and value != 1:
            raise Error("column `" + self.base.name(c) + "` is BOOL: " + String(value) + " is not 0 or 1")
        var key = (row << 20) | c
        var had = key in self.cells
        var old = self._raw(c, row)
        self.cells[key] = value
        self._record(Edit(SET, c, row, old, value, had))

    def set_null(mut self, c: Int, row: Int) raises:
        self._set(c, row, NULL_EDIT)

    def set_text(mut self, c: Int, row: Int, text: String) raises:
        """Sets a cell from typed text (parsed by the column's kind; empty text makes it null)."""
        self._check_cell(c, row)                            # before `parse` can add a word to a column
        self._set(c, row, self.parse(c, text))

    def append(mut self, values: List[String]) raises -> Int:
        """A new row from typed values (one per stored column; missing or empty ones are null); returns its number."""
        var r = List[Int64]()
        for c in range(self.stored()):
            r.append(self.parse(c, values[c]) if c < len(values) else NULL_EDIT)
        self.added.append(r^)
        var row = self.rows() - 1
        self._record(Edit(APPEND, -1, row, 0, 0, False))
        return row

    def remove(mut self, row: Int):
        if self.alive(row):
            self.removed.add(row)
            self._record(Edit(REMOVE, -1, row, 0, 0, False))

    def _record(mut self, e: Edit):
        """A new step: it goes on the log, the edit counter moves, and what was undone can no longer be redone."""
        self.log.append(e.copy())
        self._forget_redo()
        self.edits += 1

    def _forget_redo(mut self):
        """What was undone can no longer be redone: free the formula lists its steps held."""
        for e in self.redo_log:
            if e.kind == FORMULA:
                self.swaps[e.row] = List[Formula]()
        self.redo_log.clear()
        self.redo_rows.clear()

    @staticmethod
    def _is_merge(e: Edit) -> Bool:
        return e.kind == MERGE or e.kind == REPOINT

    def undo(mut self) -> Bool:
        """Takes back the last edit (a whole merge at once; a formula added, replaced or removed); False when there is nothing to undo."""
        if len(self.log) == 0:
            return False
        var e = self.log.pop()
        if e.kind == SET:
            var key = (e.row << 20) | e.col
            if e.had:
                self.cells[key] = e.old
            else:
                _ = self.cells.pop(key, 0)
            self.redo_log.append(e^)
        elif Self._is_merge(e):
            var number = e.row
            self.redo_log.append(self._unmerge_step(e))
            while len(self.log) > 0 and Self._is_merge(self.log[len(self.log) - 1]) and self.log[len(self.log) - 1].row == number:
                self.redo_log.append(self._unmerge_step(self.log.pop()))
        elif e.kind == APPEND:
            self.redo_rows.append(self.added.pop())
            self.redo_log.append(e^)
        elif e.kind == FORMULA:
            swap(self.formulas, self.swaps[e.row])
            self.redo_log.append(e^)
        else:
            self.removed.discard(e.row)
            self.redo_log.append(e^)
        self.edits += 1
        return True

    def redo(mut self) -> Bool:
        """Puts back the step the last `undo` took back; False when there is none (a new edit forgets what was undone)."""
        if len(self.redo_log) == 0:
            return False
        var e = self.redo_log.pop()
        if e.kind == SET:
            self.cells[(e.row << 20) | e.col] = e.new
            self.log.append(e^)
        elif Self._is_merge(e):
            var number = e.row
            self.log.append(self._unmerge_step(e, True))
            while len(self.redo_log) > 0 and Self._is_merge(self.redo_log[len(self.redo_log) - 1]) and \
                    self.redo_log[len(self.redo_log) - 1].row == number:
                self.log.append(self._unmerge_step(self.redo_log.pop(), True))
        elif e.kind == APPEND:
            self.added.append(self.redo_rows.pop())
            self.log.append(e^)
        elif e.kind == FORMULA:
            swap(self.formulas, self.swaps[e.row])
            self.log.append(e^)
        else:
            self.removed.add(e.row)
            self.log.append(e^)
        self.edits += 1
        return True

    def changed_rows(self) -> List[Int]:
        """Rows that differ from the shared data (edited, appended or removed), each once, ascending."""
        var seen = Set[Int]()
        for key in self.cells.keys():
            seen.add(key >> 20)
        for i in range(len(self.added)):
            seen.add(self.nbase + i)
        for row in self.removed:
            seen.add(row)
        var out = List[Int]()
        for r in seen:
            out.append(r)
        sort(out)
        return out^

    # ── merges ──────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _alias_key(c: Int, code: Int64) -> Int:
        return (c << 32) | Int(code)

    def alias(self, c: Int, code: Int64) -> Int64:
        """The code a text value reads as: itself, or the one it was merged into (never a chain)."""
        if len(self.aliases) == 0 or self.base.kind(c) != TEXT:   # only text codes merge; a number is a value
            return code
        var hit = self.aliases.get(Self._alias_key(c, code))
        return hit.value() if hit else code

    def _unmerge_step(mut self, e: Edit, forward: Bool = False) -> Edit:
        """Undoes one merge step (or, `forward`, redoes it); returns the step as the other direction needs it."""
        var key = Self._alias_key(e.col, e.old)
        if e.kind == MERGE:
            if forward:
                self.aliases[key] = e.new
            else:
                _ = self.aliases.pop(key, 0)
            return e.copy()
        var now = self.aliases.get(key).value()                        # a repoint swaps the target with the one it names
        self.aliases[key] = e.new
        return Edit(REPOINT, e.col, e.row, e.old, now, False)

    def merge(mut self, c: Int, codes: List[Int64], into: Int64) raises -> Int:
        """Folds the text values `codes` of column c into `into`: every row reads as `into` from now on. One undo
        step however many rows; returns how many values were merged."""
        if self.base.kind(c) != TEXT:
            raise Error("column `" + self.base.name(c) + "` is not TEXT: only text values merge")
        var number = 0
        for e in self.log:
            if e.kind == MERGE or e.kind == REPOINT:
                number = max(number, e.row + 1)
        var target = self.alias(c, into)
        var count = 0
        for code in codes:
            if code == target or self.alias(c, code) == target:
                continue
            var moved = List[Int64]()
            for item in self.aliases.items():
                if (item.key >> 32) == c and item.value == code:
                    moved.append(Int64(item.key & 0xFFFFFFFF))
            for m in moved:
                self.aliases[Self._alias_key(c, m)] = target
                self.log.append(Edit(REPOINT, c, number, m, code, False))
            var was = self.aliases.get(Self._alias_key(c, code))
            self.aliases[Self._alias_key(c, code)] = target
            if was:
                self.log.append(Edit(REPOINT, c, number, code, was.value(), False))
            else:
                self.log.append(Edit(MERGE, c, number, code, target, False))
            count += 1
        if count > 0:
            self._forget_redo()
            self.edits += 1
        return count

    # ── formulas ────────────────────────────────────────────────────────────────────────────────────

    def formula(mut self, text: String) raises -> Int:
        """Adds (or replaces) a formula column from "name = expression"; returns its column index. Column
        expressions only: + - * / // % min max, unary minus, numbers, parentheses. Decimal-aware as the app's:
        + - % min max bring both sides to one scale, * adds scales (then back to 2 places past 2), / gives 2 places
        (the numerator pre-scaled), // gives a whole number."""
        var parsed = self._parse_formula(text)
        var name = parsed[0]
        var saved = self.formulas.copy()
        try:
            var at = -1
            for k in range(len(self.formulas)):
                if self.formulas[k].name == name:
                    at = k
            if at < 0:
                self.formulas.append(Formula(name, text, parsed[1].copy(), parsed[2]))
                _ = self.program(self.columns() - 1)         # a cycle raises here
                self._record_formulas(saved)
                return self.columns() - 1
            var pending = self.dependants(self.stored() + at)    # their scales and factors were built on the old one
            self.formulas[at] = Formula(name, text, parsed[1].copy(), parsed[2])
            _ = self.program(self.stored() + at)             # a cycle raises here
            while len(pending) > 0:                          # each after the formulas it reads
                var later = List[String]()
                for dep in pending:
                    var j = self.index(dep) - self.stored()
                    var ready = True
                    for r in self.formulas[j].expr.columns():
                        if r != dep and r in pending:
                            ready = False
                    if not ready:
                        later.append(dep)
                        continue
                    var again = self._parse_formula(self.formulas[j].text)
                    self.formulas[j] = Formula(dep, self.formulas[j].text, again[1].copy(), again[2])
                if len(later) == len(pending):
                    raise Error("formulas refer to each other in a cycle")
                pending = later^
            self._record_formulas(saved)
            return self.stored() + at
        except e:
            self.formulas = saved^                           # a refused formula leaves the frame as it was
            raise e^

    def _record_formulas(mut self, before: List[Formula]):
        self.swaps.append(before.copy())
        self._record(Edit(FORMULA, -1, len(self.swaps) - 1, 0, 0, False))

    def remove_formula(mut self, m: Int) raises:
        """Removes the formula column m (the columns after it move up by one); one undo step. Refused while another formula reads it."""
        if m < self.stored() or m >= self.columns():
            raise Error("only a formula column is removed")
        var readers = self.dependants(m)
        if len(readers) > 0:
            raise Error("`" + self.name(m) + "` is read by the formula `" + readers[0] + "`: remove that first")
        var saved = self.formulas.copy()
        _ = self.formulas.pop(m - self.stored())
        self._record_formulas(saved)

    def _parse_formula(self, text: String) raises -> Tuple[String, Expr, Int]:
        var eq = text.find("=")
        if eq < 0:
            raise Error("a formula looks like: margin = revenue - cost")
        var name = String(String(text[byte=:eq]).strip())
        if name.byte_length() == 0:
            raise Error("the formula needs a name")
        if self.base.index(name) >= 0:
            raise Error("`" + name + "` is a stored column; a formula needs a new name")
        var body = String(text[byte=eq + 1:])
        var p = _FormulaParser(body, self)
        var r = p.expr()
        if p.peek() != "":
            raise Error("unexpected `" + p.peek() + "` in the formula")
        return (name, r[0].copy(), r[1])

    def formula_text(self, m: Int) -> String:
        return self.formulas[m - self.stored()].text

    def program(self, c: Int) raises -> Expr:
        """Column c as an expression over stored columns: formulas inlined, merges applied."""
        return self.expand(col(self.name(c)))

    def expand(self, e: Expr) raises -> Expr:
        """e with every formula reference inlined and every merged text column remapped."""
        return _expand(e, self.base, self.formulas, self.aliases, 0)

    def depends(self, m: Int, c: Int) raises -> Bool:
        """Does column m (a formula) read column c, directly or through other formulas?"""
        var target = self.name(c)
        for name in self._reads(m, 0):
            if name == target:
                return True
        return False

    def _reads(self, m: Int, depth: Int) raises -> List[String]:
        if depth > 32:
            raise Error("formulas refer to each other in a cycle")
        var out = List[String]()
        if m < self.stored():
            return out^
        for name in self.formulas[m - self.stored()].expr.columns():
            out.append(name)
            var k = self.index(name)
            if k >= self.stored():
                out.extend(self._reads(k, depth + 1))
        return out^

    def dependants(self, c: Int) raises -> List[String]:
        """The formulas that read column c (directly or not)."""
        var out = List[String]()
        for k in range(len(self.formulas)):
            if self.depends(self.stored() + k, c):
                out.append(self.formulas[k].name)
        return out^

    # ── evaluating one row (the CPU side of queries over edits, and `get` on a formula) ──────────────

    def evaluate(self, e: Expr, row: Int) raises -> Tuple[Int64, Bool]:
        """(value, null) of an expression over stored columns (as `expand` gives) on one row."""
        var raw = List[Int64](length=self.stored(), fill=0)
        var nulls = List[Int](length=self.stored(), fill=0)
        for name in e.columns():                             # only the columns the expression reads (over a scan each is a chunk)
            var c = self.base.column(name)
            var cell = self._cell(c, row)
            raw[c] = cell[0]
            nulls[c] = 1 if cell[1] else 0
        var marker = pick_marker(raw, nulls)
        for c in range(self.stored()):
            if nulls[c] != 0:
                raw[c] = marker
        return _eval_row(e, self.base, raw, marker)

    def raw_row(self, row: Int) raises -> List[Int64]:
        """The row's stored values as edited (NULL_EDIT for null), before merges."""
        var out = List[Int64]()
        for c in range(self.stored()):
            out.append(self._raw(c, row))
        return out^

    def _load_base_rows(self, rows: List[Int]) raises:
        """A scan: puts the file's values of the base rows `rows` (ascending) into the touched-rows memo, one pass per
        row group and column, each chunk decoded for the purpose and let go (not through the cache)."""
        if not self.scan:
            return
        ref cache = self.cache[]
        var i = 0
        while i < len(rows) and rows[i] < self.nbase:
            var g = self._group_of(rows[i])
            var need = List[Int]()
            while i < len(rows) and rows[i] < self.starts[g + 1]:
                if rows[i] not in cache.rows:
                    need.append(rows[i])
                i += 1
            if len(need) == 0:
                continue
            var vals = List[List[Int64]]()
            for _ in range(len(need)):
                vals.append(List[Int64](length=2 * self.stored(), fill=0))
            for c in range(self.stored()):
                var t = self.scan.value().read_group(g, [c])
                for k in range(len(need)):
                    var r = need[k] - self.starts[g]
                    if t.is_null(0, r):
                        vals[k][self.stored() + c] = 1
                    else:
                        vals[k][c] = t.get(0, r)
            for k in range(len(need)):
                cache.rows[need[k]] = vals[k].copy()

    def overlay(self) raises -> Overlay:
        """What a query needs of this session, copied: its formulas and merges (to expand expressions), the shared
        rows it changed (a bitmap the kernel skips) and the values of its changed and appended rows."""
        var o = Overlay(self.base)
        o.formulas = self.formulas.copy()
        o.aliases = self.aliases.copy()
        o.words = self.words.copy()
        var n = self.nbase
        var changed = self.changed_rows()
        if len(changed) > 0:
            if changed[0] < n:                               # appended rows alone leave the shared rows untouched
                o.mask = List[UInt8](length=(n + 7) // 8, fill=0xFF)
            var live = List[Int]()
            for row in changed:
                if row < n:
                    o.mask[row // 8] &= ~(UInt8(1) << UInt8(row % 8))
                if self.alive(row):
                    live.append(row)
            self._load_base_rows(live)                       # a scan: the file's rows, a chunk at a time, once
            var rows_v = List[List[Int64]]()
            var rows_n = List[List[Int]]()
            for row in live:
                var vs = List[Int64]()
                var ns = List[Int]()
                for c in range(self.stored()):
                    var cell = self._cell(c, row)
                    vs.append(cell[0])
                    ns.append(1 if cell[1] else 0)
                rows_v.append(vs^)
                rows_n.append(ns^)
            var all_v = List[Int64]()
            var all_n = List[Int]()
            for k in range(len(rows_v)):
                all_v.extend(rows_v[k].copy())
                all_n.extend(rows_n[k].copy())
            o.null_marker = pick_marker(all_v, all_n)
            for k in range(len(rows_v)):
                for c in range(self.stored()):
                    if rows_n[k][c] != 0:
                        rows_v[k][c] = o.null_marker
                o.delta.append(rows_v[k].copy())
                o.delta_rows.append(live[k])
        return o^

    # ── sort ────────────────────────────────────────────────────────────────────────────────────────

    def permutation(self, c: Int, descending: Bool = False, job: JobCtl = JobCtl(), budget: Int = 0, spill_dir: String = "", run_rows: Int = 0) raises -> List[Int]:
        """The live rows in the column's order, stable (ties keep row order); text alphabetical (ignoring ASCII case, bytes break ties; merges
        applied), nulls last either way. The data never moves. A radix sort over Int64 keys; a column with no edits
        is read straight from the shared values. `job` (F-175, `permutation_job`): a cancel is seen every 1M rows of the key pass and of each
        radix pass; progress counts the key pass's rows, then the sort's element steps.

        F-167: `budget` (bytes, 0: SORT_BUDGET_DEFAULT) bounds the sort's own memory, about 56 bytes a live row (the keys, the row numbers and the radix
        buffers). A frame with more rows than that allows is sorted in runs of (key, row) pairs that fit: each is sorted, written to a temporary file
        in `spill_dir` ("": TMPDIR, else a cache folder, else /tmp) and freed, then the runs are merged into the list of row numbers (8 bytes a row,
        the answer). The order is exactly the in-memory one. `run_rows` > 0 is a test's override: rows per run, always spilling."""
        job.stage(STAGE_SORT)
        job.add_total(self.rows())
        var n = self.nbase
        var text = c < self.stored() and self.base.kind(c) == TEXT
        var isfloat = self.kind(c) == FLOAT
        var rank = List[Int64]()
        if text:                                             # each code's place among the words, alphabetically
            rank = _text_ranks(self, c)
        var keys = List[Int64]()
        var rows = List[Int]()
        var nulls = List[Int]()
        var limit = sort_budget_bytes(budget)
        var spilling = run_rows > 0 or self.rows() * PERM_BYTES_PER_ROW > limit
        var cap = run_rows if run_rows > 0 else max(1024, limit // PERM_RUN_BYTES_PER_ROW)
        var spill = Optional[SpillSort]()
        if not spilling:
            keys.reserve(self.rows())
            rows.reserve(self.rows())
        if c >= self.stored():
            var prog = self.program(c)
            for row in range(self.rows()):
                if row & (JOB_SORT_STEP - 1) == 0:
                    job.check()
                    job.advance(min(JOB_SORT_STEP, self.rows() - row))
                if self.alive(row):
                    var v = self.evaluate(prog, row)
                    if v[1]:
                        nulls.append(row)
                    else:
                        keys.append(float_key(v[0]) if isfloat else v[0])
                        rows.append(row)
                        if spilling and len(keys) >= cap:
                            _perm_run(spill, keys, rows, descending, spill_dir, job)
        else:
            var edited = False
            for key in self.cells.keys():
                if (key & 0xFFFFF) == c:
                    edited = True
                    break
            var merged = text and len(self.aliases) > 0
            var has_removed = len(self.removed) > 0
            # The rows come in segments: the shared table whole, or a scan's row groups one at a time (the column's
            # chunk decoded for the pass and let go), then the appended rows.
            var segments = (len(self.starts) - 1 if self.scan else 1) + 1
            for seg in range(segments):
                var lo = n
                var hi = self.rows()
                var tbl = self.base
                var tc = c
                if seg < segments - 1:
                    lo = self.starts[seg] if self.scan else 0
                    hi = self.starts[seg + 1] if self.scan else n
                    if self.scan:
                        if hi == lo:
                            continue
                        tbl = self.scan.value().read_group(seg, [c])^.share()
                        tc = 0
                var some_nulls = tbl.null_count(tc) > 0
                var vals = tbl.values(tc)
                for row in range(lo, hi):
                    if row & (JOB_SORT_STEP - 1) == 0:
                        job.check()
                        job.advance(min(JOB_SORT_STEP, hi - row))
                    if has_removed and row in self.removed:
                        continue
                    var raw: Int64
                    var is_null = False
                    var hit = Optional[Int64](None)
                    if edited:
                        hit = self.cells.get((row << 20) | c)
                    if hit:
                        raw = hit.value()
                        is_null = raw == NULL_EDIT
                    elif row >= n:
                        raw = self.added[row - n][c]
                        is_null = raw == NULL_EDIT
                    elif some_nulls and tbl.is_null(tc, row - lo):
                        raw = 0
                        is_null = True
                    else:
                        raw = vals[row - lo]
                    if is_null:
                        nulls.append(row)
                        continue
                    if merged:
                        raw = self.alias(c, raw)
                    keys.append(rank[Int(raw)] if text else float_key(raw) if isfloat else raw)
                    rows.append(row)
                    if spilling and len(keys) >= cap:
                        _perm_run(spill, keys, rows, descending, spill_dir, job)
        if spill:                                             # F-167: the last pairs are a run too; the runs merge into the answer
            _perm_run(spill, keys, rows, descending, spill_dir, job)
            var dicts = DictAccess(Table.empty().share(), Optional[ParquetScan](), List[List[String]]())
            spill.value().begin_merge(dicts, limit, 0, -1)
            var merged = List[Int]()
            merged.reserve(self.rows())
            while True:
                var batch = spill.value().merge_batch(1 << 16, dicts)
                if len(batch[1].v) == 0:
                    break
                for v in batch[1].v:
                    merged.append(Int(v))
            merged.extend(nulls^)
            return merged^
        var out = _radix_order_job(keys, rows, descending, job)
        out.extend(nulls^)
        return out^

    # ── saving ──────────────────────────────────────────────────────────────────────────────────────

    def bake(self) raises -> Table:
        """A new table with this session's edits applied (removed rows gone, merges folded, session words added);
        formulas are not stored (their texts go in the document's metadata). Over a scan it is refused: the table
        would hold the whole file; `save(path)` streams it out instead."""
        if self.scan:
            raise Error("a Frame over a streamed file cannot be baked into memory: save it to a new file with `save(path)`")
        var t = Table.empty()
        for c in range(self.stored()):
            _ = t.add_column(self.base.name(c), self.base.kind(c), scale=self.base.scale(c))
            if self.base.kind(c) == TEXT:
                for k in range(self.word_count(c)):
                    _ = t.add_word(c, self.text_of(c, Int64(k)))
        for row in range(self.rows()):
            if not self.alive(row):
                continue
            for c in range(self.stored()):
                var cell = self._cell(c, row)
                if cell[1]:
                    t.append_null(c)
                else:
                    t.append(c, self.alias(c, cell[0]))
        return t^

    def save(self, path: String, metadata: String = "", compression: Int = ZSTD, allow_narrowing: Bool = False, job: JobCtl = JobCtl(), budget: Int = 0) raises:
        """Writes the edited rows to a new Parquet file (`metadata` under the key "omx"; the file replaces `path` by
        rename, so `path` may be the file this frame reads). A frame over a scan streams: one row group of the file at
        a time (all its columns decoded, edits applied, encoded and written; appended rows follow in groups of
        DEFAULT_ROW_GROUP), so the memory is one row group, the file's dictionaries and the edits, whatever the size.
        The saved row groups are the source's, less the removed rows. A frame over a table writes `bake()`. Formulas
        are not stored. The file's other key-value metadata is written back (not Arrow's schema, which describes the
        source's types). A column whose type reading changed but whose every value it kept (DECIMAL(9, 2) as DECIMAL(18, 2), ENUM
        or JSON as text, a narrower or unsigned integer as int64: `scan.type_change(c)`) is written as held. Only a loss
        of values (a TIMESTAMP held as its UTC day: `scan.narrowing(c)`) makes `save` raise, before writing anything and
        naming the column, unless `allow_narrowing`.

        `job` (F-175, job.mojo `save_job`): the background job this saves for. A cancel is seen between row groups and the old file
        stays (the temporary file is removed); progress counts the rows written.

        `budget` (F-188): bytes of pages being compressed at once (bodies and compressed bytes); 0: STREAM_WRITE_BUDGET for a frame over a scan,
        WRITE_BUDGET for a table. The threads (as many as the build has) work inside it; the file is the same bytes for any budget."""
        if not self.scan:
            write_parquet(path, self.bake(), metadata, compression, DEFAULT_ROW_GROUP, job, budget=budget)
            return
        ref scan = self.scan.value()
        var stored = self.stored()
        if not allow_narrowing:
            for c in range(stored):
                var lost = scan.narrowing(c)
                if lost.byte_length() > 0:
                    raise Error("column `" + self.base.name(c) + "` is " + lost + ". Nothing was written; pass allow_narrowing=True to save anyway")
        var out = Table.empty()                              # one row group's rows at a time; its words are the file's
        var null_edit = List[Bool](length=stored, fill=False)
        for item in self.cells.items():
            if item.value == NULL_EDIT:
                null_edit[item.key & 0xFFFFF] = True
        for row in self.added:
            for c in range(stored):
                if row[c] == NULL_EDIT:
                    null_edit[c] = True
        var optional = List[Bool]()
        var remap = List[List[Int64]]()                      # per column: a merged text code -> the code it reads as
        for c in range(stored):
            _ = out.add_column(self.base.name(c), self.base.kind(c), scale=self.base.scale(c))
            optional.append(scan.nullable(c) or null_edit[c])
            var map = List[Int64]()
            if self.base.kind(c) == TEXT:
                var words = self.word_count(c)
                for k in range(words):
                    _ = out.add_word(c, self.text_of(c, Int64(k)))
                if _has_aliases(self.aliases, c):
                    for k in range(words):
                        map.append(self.alias(c, Int64(k)))
            remap.append(map^)
        var all = List[Int]()
        for c in range(stored):
            all.append(c)
        var w = ParquetWriter(path, out, optional, compression)
        w.budget = budget if budget > 0 else STREAM_WRITE_BUDGET
        try:
            var changed = self.changed_rows()                # ascending; the rows whose values are not the file's
            var ci = 0
            job.stage(STAGE_WRITE)
            job.add_total(self.nbase + len(self.added))
            for g in range(scan.row_groups()):
                job.check()                                  # F-175: between row groups
                var n = scan.group_rows(g)
                if n == 0:
                    continue
                var lo = self.starts[g]
                var tbl = scan.read_group(g, all)
                _clear_rows(out)
                var r = 0
                while r < n:
                    var stop = n                             # the rows lo + r .. lo + stop are the file's
                    if ci < len(changed) and changed[ci] < lo + n:
                        stop = changed[ci] - lo
                    for c in range(stored):
                        _push_run(out.cols[c], tbl.cols[c], r, stop, remap[c])
                    r = stop
                    if r < n:                                # row lo + r is edited, removed or both
                        var row = lo + r
                        ci += 1
                        if self.alive(row):
                            for c in range(stored):
                                var hit = self.cells.get((row << 20) | c)
                                if hit:
                                    _push_value(out, c, hit.value(), remap[c])
                                else:
                                    _push(out, c, tbl, c, r, remap[c])
                        r += 1
                w.write_group(out)
                job.advance(n)
            _clear_rows(out)
            for row in range(self.nbase, self.rows()):
                if not self.alive(row):
                    continue
                for c in range(stored):
                    var hit = self.cells.get((row << 20) | c)        # an appended row can be edited too
                    _push_value(out, c, hit.value() if hit else self.added[row - self.nbase][c], remap[c])
                if out.rows() >= DEFAULT_ROW_GROUP:
                    job.check()
                    job.advance(out.rows())
                    w.write_group(out)
                    _clear_rows(out)
            if out.rows() > 0 or w.group_count() == 0:
                job.advance(out.rows())
                w.write_group(out)
            job.commit()                                     # (raises when cancelled: the temporary file goes, the old one stays)
            w.finish(metadata, scan.extra_metadata())
        except e:
            w.abort()
            raise e^


def _clear_rows(mut t: Table):
    """Empties every column's values and nulls, keeping the columns and their words."""
    for c in range(len(t.cols)):
        t.cols[c].values.clear()
        t.cols[c].validity.clear()
        t.cols[c].nulls = 0


def _push_value(mut t: Table, c: Int, raw: Int64, remap: List[Int64]) raises:
    """Appends a stored value as edited (NULL_EDIT: null) to column c, a merged text code as the code it reads as."""
    if raw == NULL_EDIT:
        t.append_null(c)
    else:
        t.append(c, remap[Int(raw)] if len(remap) > 0 else raw)


def _push(mut t: Table, c: Int, src: Table, sc: Int, row: Int, remap: List[Int64]) raises:
    """Appends row `row` of `src`'s column sc to column c of t (the file's value, untouched by edits)."""
    if src.cols[sc].is_null(row):
        t.append_null(c)
    else:
        var v = src.cols[sc].values[row]
        t.append(c, remap[Int(v)] if len(remap) > 0 else v)


def _push_run(mut dst: Column, src: Column, lo: Int, hi: Int, remap: List[Int64]):
    """Appends rows lo..hi of `src` (the file's values; text codes already the file-wide ones) to `dst`, a merged text
    code as the code it reads as. A column with no nulls so far is copied as it is."""
    if src.nulls == 0 and len(dst.validity) == 0 and len(remap) == 0:
        dst.values.reserve(len(dst.values) + hi - lo)
        for i in range(lo, hi):
            dst.values.append(src.values[i])
        return
    for i in range(lo, hi):
        if src.is_null(i):
            dst.push(0, False)
        else:
            dst.push(remap[Int(src.values[i])] if len(remap) > 0 else src.values[i], True)


def _ascii_lower(b: UInt8) -> UInt8:
    return b + 32 if UInt8(65) <= b <= UInt8(90) else b


def _text_ranks(f: Frame, c: Int) raises -> List[Int64]:
    """Every code's place among the column's words (shared and session), alphabetical ignoring ASCII case with
    bytes as the tie-break: rank[code], computed once."""
    var words = List[String]()
    for k in range(f.word_count(c)):
        words.append(f.text_of(c, Int64(k)))
    return text_ranks(words)


def text_ranks(words: List[String]) -> List[Int64]:
    """Every code's place among `words` (code order), alphabetical ignoring ASCII case with bytes as the tie-break:
    rank[code]. The order of `permutation` and of a text percentile / top_k."""
    var n = len(words)
    var order = List[Int](length=n, fill=0)
    for k in range(n):
        order[k] = k
    var tmp = List[Int](length=n, fill=0)
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = min(lo + width, n)
            var hi = min(lo + 2 * width, n)
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                if _text_less(words[order[j]], words[order[i]]):
                    tmp[k] = order[j]
                    j += 1
                else:
                    tmp[k] = order[i]
                    i += 1
                k += 1
            while i < mid:
                tmp[k] = order[i]
                i += 1
                k += 1
            while j < hi:
                tmp[k] = order[j]
                j += 1
                k += 1
            lo += 2 * width
        swap(order, tmp)
        width *= 2
    var rank = List[Int64](length=n, fill=0)
    for i in range(n):
        rank[order[i]] = Int64(i)
    return rank^


def _text_less(a: String, b: String) -> Bool:
    """Alphabetical ignoring ASCII case ("apple" before "Zebra"); words equal that way fall back to byte order."""
    var x = a.as_bytes()
    var y = b.as_bytes()
    for i in range(min(len(x), len(y))):
        if _ascii_lower(x[i]) != _ascii_lower(y[i]):
            return _ascii_lower(x[i]) < _ascii_lower(y[i])
    if len(x) != len(y):
        return len(x) < len(y)
    for i in range(len(x)):
        if x[i] != y[i]:
            return x[i] < y[i]
    return False


def _radix_order(keys: List[Int64], rows: List[Int], descending: Bool) -> List[Int]:
    """`_radix_order_job` for a plain call (no job: it cannot raise)."""
    try:
        return _radix_order_job(keys, rows, descending, JobCtl())
    except:
        return List[Int]()


comptime PERM_BYTES_PER_ROW = 56        # F-167: a permutation in memory: keys, row numbers, and the radix passes' two buffers and row orders (measured)
comptime PERM_RUN_BYTES_PER_ROW = 160   # a spilled run: the pairs, their columns, the sort of them and the buffer being written


def _perm_run(mut spill: Optional[SpillSort], mut keys: List[Int64], mut rows: List[Int], descending: Bool, dir: String, job: JobCtl) raises:
    """F-167: the (key, row) pairs so far become a sorted run on disk; the lists are emptied. The keys are already Int64 in sort order (text
    ranks, float keys), the rows ascend, so ties keep the row order: the merge's last key is the row number."""
    var n = len(keys)
    if n == 0:
        return
    var kc = ValCol(INT.value, 0, n)
    var rc = ValCol(INT.value, 0, n)
    var seq = List[Int64](length=n, fill=0)
    for i in range(n):
        kc.v[i] = keys[i]
        rc.v[i] = Int64(rows[i])
        seq[i] = Int64(rows[i])
    keys = List[Int64]()
    rows = List[Int]()
    var cols = List[ValCol]()
    cols.append(kc^)
    cols.append(rc^)
    if not spill:
        var order = List[SortKey]()
        order.append(SortKey(0, descending, False))
        var key_col: List[Int] = [0]
        spill = SpillSort(dir, order, key_col, cols, job)
    var dicts = DictAccess(Table.empty().share(), Optional[ParquetScan](), List[List[String]]())
    spill.value().add_run(cols, seq, 0, dicts)


comptime JOB_SORT_STEP = 1 << 20      # F-175: elements a sort handles between two cancel checks (and one progress add)


def _radix_order_job(keys: List[Int64], rows: List[Int], descending: Bool, job: JobCtl) raises -> List[Int]:
    """`rows` ordered by `keys` (same length), stable. LSD radix sort, 16 bits a pass, only the passes the keys'
    range needs; descending flips the keys, so equal keys still keep their order."""
    var n = len(keys)
    var u = List[UInt64](length=n, fill=0)
    var lo = UInt64.MAX
    var hi = UInt64(0)
    for i in range(n):
        if i & (JOB_SORT_STEP - 1) == 0:
            job.check()
            job.advance(min(JOB_SORT_STEP, n - i))
        var k = keys[i].cast[DType.uint64]() ^ (UInt64(1) << 63)      # signed order as unsigned order
        if descending:
            k = ~k
        u[i] = k
        lo = min(lo, k)
        hi = max(hi, k)
    var idx = List[Int](length=n, fill=0)
    for i in range(n):
        u[i] -= lo
        idx[i] = rows[i]
    var span = hi - lo if n > 0 else UInt64(0)
    var u2 = List[UInt64](length=n, fill=0)
    var i2 = List[Int](length=n, fill=0)
    var counts = List[Int](length=65537, fill=0)
    var shift = 0
    var passes = 0
    while passes < 4 and (span >> UInt64(16 * passes)) != 0:
        passes += 1
    job.add_total(2 * n * passes)
    while shift < 64 and (span >> UInt64(shift)) != 0:
        for d in range(65537):
            counts[d] = 0
        for i in range(n):
            if i & (JOB_SORT_STEP - 1) == 0:
                job.check()
                job.advance(min(JOB_SORT_STEP, n - i))
            counts[Int((u[i] >> UInt64(shift)) & 0xFFFF) + 1] += 1
        for d in range(65536):
            counts[d + 1] += counts[d]
        for i in range(n):
            if i & (JOB_SORT_STEP - 1) == 0:
                job.check()
                job.advance(min(JOB_SORT_STEP, n - i))
            var d = Int((u[i] >> UInt64(shift)) & 0xFFFF)
            var at = counts[d]
            counts[d] = at + 1
            u2[at] = u[i]
            i2[at] = idx[i]
        swap(u, u2)
        swap(idx, i2)
        shift += 16
    return idx^


# ── the formula parser (the app's grammar and decimal rules) ──────────────────────────────────────────

struct _FormulaParser:
    var text: String
    var pos: Int
    var names: List[String]
    var scales: List[Int]

    def __init__(out self, text: String, f: Frame) raises:
        self.text = text
        self.pos = 0
        self.names = List[String]()
        self.scales = List[Int]()
        for c in range(f.columns()):
            self.names.append(f.name(c))
            self.scales.append(-1 if f.kind(c) == FLOAT else f.scale(c))      # -1: a float64

    def peek(mut self) -> String:
        while self.pos < self.text.byte_length() and String(self.text[byte=self.pos:self.pos + 1]) == " ":
            self.pos += 1
        if self.pos >= self.text.byte_length():
            return ""
        return String(self.text[byte=self.pos:self.pos + 1])

    def expr(mut self) raises -> Tuple[Expr, Int]:
        var left = self.term()
        while self.peek() == "+" or self.peek() == "-":
            var op = self.peek()
            self.pos += 1
            var right = self.term()
            var a = _align(left^, right^)
            left = ((a[0] + a[1]) if op == "+" else (a[0] - a[1]), a[2])
        return left^

    def term(mut self) raises -> Tuple[Expr, Int]:
        var left = self.factor()
        while True:
            var c = self.peek()
            if c == "*":
                self.pos += 1
                var right = self.factor()
                if left[1] < 0 or right[1] < 0:
                    var fa = _align(left^, right^)
                    left = (fa[0] * fa[1], -1)
                    continue
                var e = left[0] * right[0]
                var s = left[1] + right[1]
                if s > 2:                                    # keep money at cents
                    e = e // Expr(Int(_pow10(s - 2)))
                    s = 2
                left = (e^, s)
            elif c == "/":
                self.pos += 1
                if self.pos < self.text.byte_length() and String(self.text[byte=self.pos:self.pos + 1]) == "/":
                    self.pos += 1
                    var right = self.factor()
                    if left[1] < 0 or right[1] < 0:
                        raise Error("`//` on a FLOAT is not supported in a formula: use `/`")
                    var a = _align(left^, right^)
                    left = (a[0] // a[1], 0)
                else:
                    var right = self.factor()
                    if left[1] < 0 or right[1] < 0:           # a division of float64s (IEEE: x / 0 is inf, -inf or NaN)
                        var fa = _align(left^, right^)
                        left = (fa[0] / fa[1], -1)
                        continue
                    var k = 2 - left[1] + right[1]           # (a * 10^k) // b has 2 decimal places
                    var num = left[0] * Expr(Int(_pow10(k))) if k > 0 else left[0].copy()
                    var den = right[0] * Expr(Int(_pow10(-k))) if k < 0 else right[0].copy()
                    left = (num // den, 2)
            elif c == "%":
                self.pos += 1
                var right = self.factor()
                if left[1] < 0 or right[1] < 0:
                    raise Error("`%` on a FLOAT is not supported in a formula")
                var a = _align(left^, right^)
                left = (a[0] % a[1], a[2])
            else:
                return left^

    def factor(mut self) raises -> Tuple[Expr, Int]:
        var c = self.peek()
        if c == "":
            raise Error("the formula ends early")
        if c == "-":
            self.pos += 1
            var f = self.factor()
            return (-f[0], f[1])
        if c == "(":
            self.pos += 1
            var e = self.expr()
            if self.peek() != ")":
                raise Error("missing ) in the formula")
            self.pos += 1
            return e^
        var o = ord(c)
        if (ord("0") <= o <= ord("9")) or c == ".":
            var start = self.pos
            while self.pos < self.text.byte_length():
                var ch = ord(String(self.text[byte=self.pos:self.pos + 1]))
                if (ord("0") <= ch <= ord("9")) or ch == ord("."):
                    self.pos += 1
                else:
                    break
            if self.pos + 1 < self.text.byte_length() and (String(self.text[byte=self.pos:self.pos + 1]) == "e" or String(self.text[byte=self.pos:self.pos + 1]) == "E"):
                var q = self.pos + 1                          # an exponent makes a float64 literal: 1.5e-3
                var sg = String(self.text[byte=q:q + 1])
                if (sg == "+" or sg == "-") and q + 1 < self.text.byte_length():
                    q += 1
                if q < self.text.byte_length() and ord("0") <= ord(String(self.text[byte=q:q + 1])) <= ord("9"):
                    while q < self.text.byte_length() and ord("0") <= ord(String(self.text[byte=q:q + 1])) <= ord("9"):
                        q += 1
                    var ftext = String(self.text[byte=start:q])
                    self.pos = q
                    var fv = Float64(0.0)
                    var fp = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(ftext.unsafe_ptr()))
                    var st = _parse_float(fp, ftext.byte_length(), fv)
                    _ = ftext.byte_length()
                    if st != F_OK:
                        raise Error("`" + ftext + "` isn't a number the formula can hold")
                    return (Expr(fv), -1)
            var num = String(self.text[byte=start:self.pos])
            var dot = num.find(".")
            var scale = 0 if dot < 0 else num.byte_length() - dot - 1
            var out: Int64 = 0
            var places = 0
            var np = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(num.unsafe_ptr()))
            var ok = _parse_decimal(np, num.byte_length(), scale, out, places)
            _ = num.byte_length()
            if not ok:
                raise Error("`" + num + "` isn't a number")
            return (Expr(Int(out)), scale)
        if c == "_" or (ord("a") <= o <= ord("z")) or (ord("A") <= o <= ord("Z")):
            var start = self.pos
            while self.pos < self.text.byte_length():
                var ch = ord(String(self.text[byte=self.pos:self.pos + 1]))
                if ch == ord("_") or (ord("a") <= ch <= ord("z")) or (ord("A") <= ch <= ord("Z")) or (ord("0") <= ch <= ord("9")):
                    self.pos += 1
                else:
                    break
            var word = String(self.text[byte=start:self.pos])
            if word == "min" or word == "max":
                if self.peek() != "(":
                    raise Error(word + " needs (a, b)")
                self.pos += 1
                var a = self.expr()
                if self.peek() != ",":
                    raise Error(word + " needs two values: " + word + "(a, b)")
                self.pos += 1
                var b = self.expr()
                if self.peek() != ")":
                    raise Error("missing ) after " + word + "(a, b")
                self.pos += 1
                var al = _align(a^, b^)
                return ((least(al[0], al[1]) if word == "min" else greatest(al[0], al[1])), al[2])
            for k in range(len(self.names)):
                if self.names[k] == word:
                    return (col(word), self.scales[k])
            raise Error("column `" + word + "` not found")
        raise Error("unexpected `" + c + "` in the formula")


def _scaled_up(e: Expr, places: Int, scale: Int) -> Expr:
    """e times 10^places, marked as having `scale` decimal places (a column times a literal would keep the column's
    scale, and the query compiler's check would refuse the + - min max that follows)."""
    var out = e * Expr(Int(_pow10(places)))
    out.nodes.append(Node(N_SCALE_TO, Int64(scale), ""))
    return out^


def _floaty(e: Expr, scale: Int) -> Expr:
    """An INT or DECIMAL expression (with `scale` decimal places) read as a float64."""
    var out = e.copy()
    out.nodes.append(Node(N_I2F, Int64(max(scale, 0)), ""))
    return out^


def _align(var a: Tuple[Expr, Int], var b: Tuple[Expr, Int]) -> Tuple[Expr, Expr, Int]:
    """Both sides at one scale: the one with fewer decimal places is multiplied up. A float64 side (scale -1) makes both float64s."""
    if a[1] < 0 or b[1] < 0:
        return (a[0].copy() if a[1] < 0 else _floaty(a[0], a[1]), b[0].copy() if b[1] < 0 else _floaty(b[0], b[1]), -1)
    if a[1] < b[1]:
        return (_scaled_up(a[0], b[1] - a[1], b[1]), b[0].copy(), b[1])
    if b[1] < a[1]:
        return (a[0].copy(), _scaled_up(b[0], a[1] - b[1], a[1]), a[1])
    return (a[0].copy(), b[0].copy(), a[1])



# ── shared with queries: expanding expressions and evaluating rows ───────────────────────────────────

struct Overlay(Copyable, Movable):
    """A session's changes as a query sees them (copied when the query is built, see `Frame.overlay`)."""
    var base: SharedTable
    var formulas: List[Formula]
    var aliases: Dict[Int, Int64]
    var mask: List[UInt8]                # empty: no shared row changed; else bit 1 = the shared row is unchanged
    var delta: List[List[Int64]]         # the live changed and appended rows' stored values (`null_marker`: null)
    var delta_rows: List[Int]            # the frame's row number of each delta row (ascending): where a query puts it among the shared rows
    var words: List[List[String]]        # per stored column: the words the session added (codes after the dictionary)
    var null_marker: Int64               # in `delta`, a null cell holds this value (NULL_EDIT unless a changed row holds it as data)

    def __init__(out self, base: SharedTable):
        self.base = base
        self.null_marker = NULL_EDIT
        self.words = List[List[String]]()
        self.formulas = List[Formula]()
        self.aliases = Dict[Int, Int64]()
        self.mask = List[UInt8]()
        self.delta = List[List[Int64]]()
        self.delta_rows = List[Int]()

    def expand(self, e: Expr) raises -> Expr:
        return _expand(e, self.base, self.formulas, self.aliases, 0)


def _has_aliases(aliases: Dict[Int, Int64], c: Int) -> Bool:
    for k in aliases.keys():
        if (k >> 32) == c:
            return True
    return False


def _remapped(base: SharedTable, aliases: Dict[Int, Int64], c: Int) -> Expr:
    """Column c with its merges applied, in exact integer arithmetic: c + sum_i int(c == a_i) * (t_i - a_i)."""
    var e = col(base.name(c))
    for item in aliases.items():
        if (item.key >> 32) == c:
            var a = Int(item.key & 0xFFFFFFFF)
            var flag = Expr((col(base.name(c)) == Expr(a)).nodes.copy())
            flag.nodes.append(Node(N_BOOL2INT, 0, ""))
            e = e + flag * Expr(Int(item.value) - a)
    return e^


def _expand(e: Expr, base: SharedTable, formulas: List[Formula], aliases: Dict[Int, Int64], depth: Int) raises -> Expr:
    """e with every formula reference inlined and every merged text column remapped."""
    if depth > 32:
        raise Error("formulas refer to each other in a cycle")
    var nodes = List[Node]()
    for n in e.nodes:
        if n.op != N_COL:
            nodes.append(n.copy())
            continue
        var c = base.index(n.name)
        if c >= 0:
            if base.kind(c) == TEXT and _has_aliases(aliases, c):
                nodes.extend(_remapped(base, aliases, c).nodes.copy())
            else:
                nodes.append(n.copy())
            continue
        var found = False
        for f in formulas:
            if f.name == n.name:
                nodes.extend(_expand(f.expr, base, formulas, aliases, depth + 1).nodes.copy())
                found = True
                break
        if not found:
            raise Error("column `" + n.name + "` not found")
    return Expr(nodes^)


def _overflow() -> Error:
    return Error("a sum or an expression passed the int64 range (about 9.2e18)")


def pick_marker(vals: List[Int64], nulls: List[Int]) -> Int64:
    """A value to mark null cells with among rows whose non-null cells are `vals` (`nulls[i]` 1 where cell i is null): NULL_EDIT, or the
    next value up that no non-null cell holds."""
    var m = NULL_EDIT
    var clash = True
    while clash:
        clash = False
        for i in range(len(vals)):
            if nulls[i] == 0 and vals[i] == m:
                clash = True
                break
        if clash:
            m += 1
    return m


def _eval_row(e: Expr, base: SharedTable, raw: List[Int64], marker: Int64 = NULL_EDIT) raises -> Tuple[Int64, Bool]:
    """(value, null) of an expression over stored columns on one row's values (see `_eval_row_t`; a FLOAT result is its bits)."""
    var r = _eval_row_t(e, base, raw, marker)
    return (r[0], r[1])


def _fcmp(a: Float64, b: Float64) -> Int:
    """DuckDB's order of doubles: NaN equals itself and is above every number (-0.0 equals 0.0, as IEEE)."""
    if a != a:
        return 0 if b != b else 1
    if b != b:
        return -1
    return -1 if a < b else 1 if a > b else 0


def _eval_row_t(e: Expr, base: SharedTable, raw: List[Int64], marker: Int64 = NULL_EDIT) raises -> Tuple[Int64, Bool, Bool]:
    """(value, null, is a float64) of an expression over stored columns on one row's values (NULL_EDIT: null), with the kernel's
    semantics: exact int64 (overflow raises), Mojo's floor // and %, x // 0 = x % 0 = 0. A null operand makes the result null,
    through comparisons and logic too (a filter reading a null is false as a whole). A FLOAT column or float literal makes a
    float64 (the value is its bits): an INT or DECIMAL beside it is converted as DuckDB casts it (float(v) / 10^scale; past 2^53 it
    raises, as the kernel does), `/` divides doubles (IEEE), comparisons use DuckDB's order of doubles."""
    var vals = List[Int64]()
    var nulls = List[Bool]()
    var fls = List[Bool]()
    var scales = List[Int]()
    for n in e.nodes:
        if n.op == N_COL:
            var c = base.column(n.name)
            var v = raw[c]
            vals.append(0 if v == marker else v)
            nulls.append(v == marker)
            fls.append(base.kind(c) == FLOAT)
            scales.append(-1 if base.kind(c) == FLOAT else base.scale(c))
        elif n.op == N_LIT or n.op == N_LITF:
            vals.append(n.value)
            nulls.append(False)
            fls.append(n.op == N_LITF)
            scales.append(-1)
        elif n.op == N_NEG or n.op == N_NOT or n.op == N_BOOL2INT or n.op == N_SCALE_TO or n.op == N_I2F:
            var a = vals.pop()
            var an = nulls.pop()
            var af = fls.pop()
            var asc = scales.pop()
            if n.op == N_I2F:
                if not af:
                    a = _to_float_bits(a, Int(n.value))
                vals.append(a)
                nulls.append(an)
                fls.append(True)
                scales.append(-1)
                continue
            if n.op == N_NEG and af:
                vals.append(Int64((-Float64(from_bits=UInt64(a))).to_bits()))
            else:
                if n.op == N_NEG and a == Int64.MIN:
                    raise _overflow()
                vals.append(-a if n.op == N_NEG else Int64(a == 0) if n.op == N_NOT else a)
            nulls.append(an)
            fls.append(af and (n.op == N_NEG or n.op == N_SCALE_TO))
            scales.append(Int(n.value) if n.op == N_SCALE_TO else asc if n.op == N_NEG else -1)
        else:
            var b = vals.pop()
            var a = vals.pop()
            var bn = nulls.pop()
            var an = nulls.pop()
            var bf = fls.pop()
            var af = fls.pop()
            var bs = scales.pop()
            var as_ = scales.pop()
            var v: Int64 = 0
            var logical = n.op == N_AND or n.op == N_OR
            var compare = n.op >= N_LT and n.op <= N_NE
            if (af or bf or n.op == N_DIV) and not logical:
                if n.op == N_FLOORDIV or n.op == N_MOD:
                    raise Error("`//` and `%` on a FLOAT are not supported")
                if not an and not af:
                    a = _to_float_bits(a, as_ if as_ > 0 else 0)
                if not bn and not bf:
                    b = _to_float_bits(b, bs if bs > 0 else 0)
                var x = Float64(from_bits=UInt64(a))
                var y = Float64(from_bits=UInt64(b))
                var out_fl = True
                if n.op == N_ADD:
                    v = Int64((x + y).to_bits())
                elif n.op == N_SUB:
                    v = Int64((x - y).to_bits())
                elif n.op == N_MUL:
                    v = Int64((x * y).to_bits())
                elif n.op == N_DIV:
                    v = Int64((x / y).to_bits())
                elif n.op == N_LEAST:
                    v = a if _fcmp(x, y) <= 0 else b
                elif n.op == N_GREATEST:
                    v = a if _fcmp(x, y) >= 0 else b
                else:
                    var c3 = _fcmp(x, y)
                    out_fl = False
                    v = Int64((c3 < 0) if n.op == N_LT else (c3 <= 0) if n.op == N_LE else (c3 > 0) if n.op == N_GT else
                              (c3 >= 0) if n.op == N_GE else (c3 == 0) if n.op == N_EQ else (c3 != 0))
                vals.append(v)
                nulls.append(an or bn)
                fls.append(out_fl)
                scales.append(-1)
                continue
            if n.op == N_ADD:
                v = a + b
                if ((a ^ v) & (b ^ v)) < 0:
                    raise _overflow()
            elif n.op == N_SUB:
                v = a - b
                if ((a ^ b) & (a ^ v)) < 0:
                    raise _overflow()
            elif n.op == N_MUL:
                v = a * b
                if (a == -1 and b == Int64.MIN) or (b == -1 and a == Int64.MIN) or (a != 0 and v // a != b):
                    raise _overflow()
            elif n.op == N_FLOORDIV:
                if b == -1 and a == Int64.MIN:
                    raise _overflow()
                v = a // b if b != 0 else 0
            elif n.op == N_MOD:
                v = a % b if b != 0 and b != -1 else 0
            elif n.op == N_LEAST:
                v = min(a, b)
            elif n.op == N_GREATEST:
                v = max(a, b)
            elif n.op == N_LT:
                v = Int64(a < b)
            elif n.op == N_LE:
                v = Int64(a <= b)
            elif n.op == N_GT:
                v = Int64(a > b)
            elif n.op == N_GE:
                v = Int64(a >= b)
            elif n.op == N_EQ:
                v = Int64(a == b)
            elif n.op == N_NE:
                v = Int64(a != b)
            elif n.op == N_AND:
                v = Int64(a != 0 and b != 0)
            else:
                v = Int64(a != 0 or b != 0)
            vals.append(v)
            nulls.append(an or bn)
            fls.append(False)
            var sc = as_ if as_ >= 0 else bs
            if n.op == N_MUL:
                sc = as_ + bs if as_ >= 0 and bs >= 0 else sc
            if n.op == N_FLOORDIV or compare or logical:
                sc = -1
            scales.append(sc)
    return (vals[0], nulls[0], fls[0])


def _to_float_bits(a: Int64, scale: Int) raises -> Int64:
    """An INT or DECIMAL value as the bits of a float64, as the kernel's conversion does (DuckDB's cast, also past 2^53, F-190)."""
    return Int64(decimal_to_float(a, scale).to_bits())
