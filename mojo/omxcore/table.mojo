"""Tables and columns (docs/omxcore-mojo-api.md section 1).

Every value is an Int64 underneath, column-major and contiguous, so the app borrows a column as `Span[Int64]` and
the kernels read it as an int64 column with no conversion:
  INT      the integer
  DECIMAL  the value times 10^scale (scale 2: cents), so sums stay exact
  TEXT     a dictionary code (0, 1, ... in first-seen order); the words are the column's dictionary
  DATE     days since 1970-01-01
  BOOL     0 or 1
  FLOAT    a float64's bits (F-157): `append_float`, `get_float`; -0.0, NaN payloads and subnormals are kept as given
A null is a cleared bit in the column's validity bitmap (Arrow's layout: bit i of byte i // 8, least significant
first); its value slot holds 0. A column with no nulls has no bitmap.

A `Table` is built by one owner (`add_column`, `append*`), then `share()`d: a `SharedTable` is a read-only handle
that any number of sessions copy without copying the columns.
"""
from std.collections import Dict
from std.memory import ArcPointer


struct ColumnKind(Equatable, ImplicitlyCopyable, Writable):
    var value: Int

    def __init__(out self, value: Int):
        self.value = value

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    def write_to(self, mut writer: Some[Writer]):
        var names: List[String] = ["INT", "DECIMAL", "TEXT", "DATE", "BOOL", "FLOAT"]
        writer.write(names[self.value] if 0 <= self.value < len(names) else "kind " + String(self.value))


comptime INT = ColumnKind(0)
comptime DECIMAL = ColumnKind(1)
comptime TEXT = ColumnKind(2)
comptime DATE = ColumnKind(3)
comptime BOOL = ColumnKind(4)
comptime FLOAT = ColumnKind(5)     # float64 (F-157): the value slot holds the float64's bits; the validity layout is the others'
comptime F32_COLUMN = 32           # a FLOAT's `scale` when the file stored 32-bit floats (read widened to float64): SQL compares such a column with a
                                   # literal in 32 bits, as DuckDB does (`l_discount <= 0.07` keeps the rows holding 0.07). 0 for a float64 column.


struct WordDict(Movable):
    """A TEXT column's dictionary: the words by code and the code of each word. Columns hold it through an `ArcPointer`, so a table made from another's
    columns (a join's gather, a query's result) shares the dictionary instead of copying it; the first `add_word` on a shared one copies it first."""
    var words: List[String]            # by code
    var codes: Dict[String, Int]       # word -> code

    def __init__(out self):
        self.words = List[String]()
        self.codes = Dict[String, Int]()


struct Column(Movable):
    var name: String
    var kind: ColumnKind
    var scale: Int
    var values: List[Int64]
    var high: List[Int64]              # F-192: a total past int64 (a SUM, a window SUM): the upper 64 bits of each 128-bit value, `values` holding the lower 64; empty while every value fits int64
    var validity: List[UInt8]          # empty: no nulls
    var nulls: Int
    var dict: ArcPointer[WordDict]     # TEXT: the dictionary (shared between columns that hold the same words)

    def __init__(out self, name: String, kind: ColumnKind, scale: Int):
        self.name = name
        self.kind = kind
        self.scale = scale
        self.values = List[Int64]()
        self.high = List[Int64]()
        self.validity = List[UInt8]()
        self.nulls = 0
        self.dict = ArcPointer(WordDict())

    def push(mut self, value: Int64, valid: Bool):
        var i = len(self.values)
        self.values.append(value if valid else 0)
        if not valid and len(self.validity) == 0:      # first null: every earlier row was valid
            for _ in range((i + 8) // 8):
                self.validity.append(0xFF)
        if len(self.validity) > 0:
            if i // 8 >= len(self.validity):
                self.validity.append(0xFF)
            if not valid:
                self.validity[i // 8] &= ~(UInt8(1) << UInt8(i % 8))
                self.nulls += 1

    def validity_address(self) -> Int:
        """The validity bitmap's address (Arrow layout), 0 when the column has no nulls (for the kernels)."""
        return Int(self.validity.unsafe_ptr()) if len(self.validity) > 0 else 0

    def is_null(self, row: Int) -> Bool:
        if len(self.validity) == 0:
            return False
        return (self.validity[row // 8] >> UInt8(row % 8)) & 1 == 0


struct Table(Movable):
    var cols: List[Column]
    var by_name: Dict[String, Int]

    def __init__(out self):
        self.cols = List[Column]()
        self.by_name = Dict[String, Int]()

    @staticmethod
    def empty() -> Table:
        """A table with no columns ("New blank sheet")."""
        return Table()

    # ── shape and metadata ──────────────────────────────────────────────────────────────────────

    def add_column(mut self, name: String, kind: ColumnKind, scale: Int = 0) raises -> Int:
        """Adds an empty column; returns its index. A DECIMAL's scale is its number of decimal places."""
        if name in self.by_name:
            raise Error("column `" + name + "` already exists")
        if kind.value < 0 or kind.value > 5:
            raise Error("column `" + name + "`: unknown kind " + String(kind.value))
        if kind == FLOAT:
            if scale != 0 and scale != F32_COLUMN:
                raise Error("column `" + name + "`: a FLOAT's scale is 0 (float64) or " + String(F32_COLUMN) + " (a column the file stored as 32-bit floats)")
        elif scale < 0 or scale > 18:
            raise Error("column `" + name + "`: scale " + String(scale) + " is outside 0..18")
        elif kind != DECIMAL and scale != 0:
            raise Error("column `" + name + "`: only a DECIMAL has a scale")
        self.by_name[name] = len(self.cols)
        self.cols.append(Column(name, kind, scale))
        return len(self.cols) - 1

    def rows(self) -> Int:
        return len(self.cols[0].values) if len(self.cols) > 0 else 0

    def columns(self) -> Int:
        return len(self.cols)

    def name(self, c: Int) -> String:
        return self.cols[c].name

    def kind(self, c: Int) -> ColumnKind:
        return self.cols[c].kind

    def scale(self, c: Int) -> Int:
        return self.cols[c].scale

    def index(self, name: String) -> Int:
        """The column's index, or -1 when there is none."""
        return self.by_name.get(name, -1)

    def column(self, name: String) raises -> Int:
        """The column's index; raises "column `x` not found" when there is none."""
        var c = self.index(name)
        if c < 0:
            raise Error("column `" + name + "` not found")
        return c

    def check(self) raises:
        """Raises when the columns do not all have the same number of rows."""
        for c in range(1, len(self.cols)):
            if len(self.cols[c].values) != self.rows():
                raise Error("column `" + self.cols[c].name + "` has " + _rows(len(self.cols[c].values)) +
                            ", column `" + self.cols[0].name + "` has " + _rows(self.rows()))

    # ── building ────────────────────────────────────────────────────────────────────────────────

    def append(mut self, c: Int, value: Int64) raises:
        """Appends a stored value (scaled for a DECIMAL, a code for TEXT, days for DATE, 0 / 1 for BOOL). A TEXT
        code must be one of the column's words (`add_word`), a BOOL 0 or 1."""
        ref col = self.cols[c]
        if col.kind == TEXT and (value < 0 or value >= Int64(len(col.dict[].words))):
            raise Error("column `" + col.name + "`: code " + String(value) + " is not one of its " +
                        String(len(col.dict[].words)) + " words")
        if col.kind == BOOL and value != 0 and value != 1:
            raise Error("column `" + col.name + "` is BOOL: " + String(value) + " is not 0 or 1")
        col.push(value, True)

    def append_float(mut self, c: Int, value: Float64) raises:
        """Appends a FLOAT value (its bits are stored as they are: -0.0 and every NaN payload keep their bits)."""
        if self.cols[c].kind != FLOAT:
            raise Error("column `" + self.cols[c].name + "` is " + String(self.cols[c].kind) + ", not FLOAT")
        self.cols[c].push(Int64(value.to_bits()), True)

    def append_null(mut self, c: Int):
        self.cols[c].push(0, False)

    def add_word(mut self, c: Int, word: String) raises -> Int:
        """A TEXT column's code for `word`, adding it to the dictionary when new (no row is added): load a
        dictionary once, then `append` codes."""
        if self.cols[c].kind != TEXT:
            raise Error("column `" + self.cols[c].name + "` is " + String(self.cols[c].kind) + ", not TEXT")
        var code = self.cols[c].dict[].codes.get(word, -1)
        if code < 0:
            if self.cols[c].dict.count() > 1:                       # shared with another column: this one gets a copy before it changes
                var own = WordDict()
                own.words = self.cols[c].dict[].words.copy()
                own.codes = self.cols[c].dict[].codes.copy()
                self.cols[c].dict = ArcPointer(own^)
            code = len(self.cols[c].dict[].words)
            self.cols[c].dict[].words.append(word)
            self.cols[c].dict[].codes[word] = code
        return code

    def append_text(mut self, c: Int, word: String) raises:
        """Appends a TEXT value: an existing word keeps its code, a new word gets the next one."""
        var code = self.add_word(c, word)
        self.cols[c].push(Int64(code), True)

    # ── reading ─────────────────────────────────────────────────────────────────────────────────

    def get(self, c: Int, row: Int) -> Int64:
        """The stored value (0 for a null: ask `is_null`). Not range-checked (the hot path): c and row must exist."""
        return self.cols[c].values[row]

    def get_float(self, c: Int, row: Int) -> Float64:
        """A FLOAT cell (a null reads 0.0: ask `is_null`)."""
        return Float64(from_bits=UInt64(self.cols[c].values[row]))

    def is_wide(self, c: Int) -> Bool:
        """Whether the column holds a total past int64 (F-192): `get` is then only the lower 64 bits; `get_high` the upper ones."""
        return len(self.cols[c].high) > 0

    def get_high(self, c: Int, row: Int) -> Int64:
        """The upper 64 bits of a cell: its sign extension for a column that is not wide."""
        return self.cols[c].high[row] if len(self.cols[c].high) > 0 else self.cols[c].values[row] >> 63

    def any_wide(self) -> Bool:
        for c in range(len(self.cols)):
            if len(self.cols[c].high) > 0:
                return True
        return False

    def is_null(self, c: Int, row: Int) -> Bool:
        return self.cols[c].is_null(row)

    def null_count(self, c: Int) -> Int:
        return self.cols[c].nulls

    def values(self, c: Int) -> Span[Int64, origin_of(self)]:
        """The column's values, read-only and without a copy, valid until the column next changes."""
        var p = Pointer[Int64, origin_of(self)](unsafe_from_address=Int(self.cols[c].values.unsafe_ptr()))
        return Span[Int64, origin_of(self)](unsafe_ptr=p, length=len(self.cols[c].values))

    def dictionary(self, c: Int) -> Span[String, origin_of(self)]:
        """A TEXT column's words, by code, read-only."""
        var p = Pointer[String, origin_of(self)](unsafe_from_address=Int(self.cols[c].dict[].words.unsafe_ptr()))
        return Span[String, origin_of(self)](unsafe_ptr=p, length=len(self.cols[c].dict[].words))

    def share_dictionary(mut self, c: Int, other: SharedTable, oc: Int):
        """TEXT column `c` takes column `oc` of `other` as its dictionary (no word is copied: the two share it until one of them adds a word)."""
        self.cols[c].dict = other._table[].cols[oc].dict

    def text_of(self, c: Int, code: Int) -> String:
        return self.cols[c].dict[].words[code]

    def code_of(self, c: Int, word: String) -> Int:
        """The word's code, or -1 when the column has never held it."""
        return self.cols[c].dict[].codes.get(word, -1)

    def share(deinit self) -> SharedTable:
        """Hands the table over as a read-only table every session can hold (no copy)."""
        return SharedTable(self^)


struct SharedTable(ImplicitlyCopyable):
    """A read-only handle on a shared table. Copies share the columns; the table lives while any handle does.
    Everything public reads: `_table` is internal (Mojo has no private fields), never write through it."""
    var _table: ArcPointer[Table]

    def __init__(out self, var table: Table):
        self._table = ArcPointer(table^)

    def rows(self) -> Int:
        return self._table[].rows()

    def columns(self) -> Int:
        return self._table[].columns()

    def name(self, c: Int) -> String:
        return self._table[].name(c)

    def kind(self, c: Int) -> ColumnKind:
        return self._table[].kind(c)

    def scale(self, c: Int) -> Int:
        return self._table[].scale(c)

    def index(self, name: String) -> Int:
        return self._table[].index(name)

    def column(self, name: String) raises -> Int:
        return self._table[].column(name)

    def get(self, c: Int, row: Int) -> Int64:
        return self._table[].get(c, row)

    def get_float(self, c: Int, row: Int) -> Float64:
        return self._table[].get_float(c, row)

    def is_wide(self, c: Int) -> Bool:
        return self._table[].is_wide(c)

    def get_high(self, c: Int, row: Int) -> Int64:
        return self._table[].get_high(c, row)

    def is_null(self, c: Int, row: Int) -> Bool:
        return self._table[].is_null(c, row)

    def null_count(self, c: Int) -> Int:
        return self._table[].null_count(c)

    def values(self, c: Int) -> Span[Int64, origin_of(self)]:
        """The column's values, read-only, without a copy; the span lives no longer than this handle. Do not reassign
        or drop the handle while using its spans (if it held the last reference, the columns are freed)."""
        var p = Pointer[Int64, origin_of(self)](unsafe_from_address=Int(self._table[].cols[c].values.unsafe_ptr()))
        return Span[Int64, origin_of(self)](unsafe_ptr=p, length=len(self._table[].cols[c].values))

    def dictionary(self, c: Int) -> Span[String, origin_of(self)]:
        """A TEXT column's words, by code, read-only."""
        var p = Pointer[String, origin_of(self)](unsafe_from_address=Int(self._table[].cols[c].dict[].words.unsafe_ptr()))
        return Span[String, origin_of(self)](unsafe_ptr=p, length=len(self._table[].cols[c].dict[].words))

    def text_of(self, c: Int, code: Int) -> String:
        return self._table[].text_of(c, code)

    def code_of(self, c: Int, word: String) -> Int:
        return self._table[].code_of(c, word)


def _rows(n: Int) -> String:
    return String(n) + (" row" if n == 1 else " rows")
