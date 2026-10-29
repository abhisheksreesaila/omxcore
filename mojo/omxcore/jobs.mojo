"""The jobs (F-175, job.mojo has the handle): every long piece of work an app starts, off its own thread.

    compute_job(query, device)               -> Job[QueryResult]    the builder's Query.compute
    sql_job(catalog, text, options)          -> Job[SqlResult]      sql(catalog, text)
    read_csv_job(path, options)              -> Job[CsvResult]      read_csv
    scan_csv_job(path, options)              -> Job[ParquetScan]    scan_csv (F-178: the pass that finds a CSV's row groups)
    read_parquet_job(path)                   -> Job[ParquetResult]  read_parquet
    save_job(frame, path, ...)               -> Job[Int]            Frame.save (the rows written)
    write_parquet_job(table, path, ...)      -> Job[Int]            write_parquet

Each takes its inputs by value or copies them (the job owns what it works on): a `Query` is moved in, a `Catalog` and a `Frame`
are snapshotted when the job starts, so the caller may go on editing the frame or the catalog's sources; a `SharedTable` is
shared read only. The result of a job is what the plain call returns, value for value. Each takes a `JobDone` callback (job.mojo) as its
last argument, `NoCallback()` by default. Docs: docs/omxcore-mojo-api.md section 8c.
"""
from .job import Job, JobCtl, JobBody, JobDone, NoCallback, start_job_then
from .query import Query, QueryResult, AUTO
from .bind import Catalog
from .planner import PlannerOptions, RulePlanner
from .sqlexec import SqlResult
from .sqlquery import sql_with
from .csv import read_csv, CsvOptions, CsvResult
from .parquet import ParquetResult, ParquetScan, read_parquet, scan_csv, write_parquet, ZSTD, DEFAULT_ROW_GROUP
from .table import SharedTable
from .frame import Frame, _ChunkCache
from std.memory import ArcPointer


struct QueryJob(JobBody):
    comptime Out = QueryResult
    var query: Query
    var device: Int

    def __init__(out self, var query: Query, device: Int):
        self.query = query^
        self.device = device

    def run(mut self, ctl: JobCtl) raises -> QueryResult:
        return self.query.compute(self.device, ctl)


def compute_job[CB: JobDone = NoCallback](var query: Query, device: Int = AUTO, var done: CB = NoCallback()) -> Job[QueryResult]:
    """`query.compute(device)` on a thread of its own. Moves the query in; build it again for another run."""
    return start_job_then(QueryJob(query^, device), done^)


struct SqlJob(JobBody):
    comptime Out = SqlResult
    var catalog: Catalog
    var text: String
    var options: PlannerOptions

    def __init__(out self, var catalog: Catalog, var text: String, var options: PlannerOptions):
        self.catalog = catalog^
        self.text = text^
        self.options = options^

    def run(mut self, ctl: JobCtl) raises -> SqlResult:
        self.options.job = ctl
        return sql_with(self.catalog, self.text, RulePlanner(), self.options)


def sql_job[CB: JobDone = NoCallback](catalog: Catalog, text: String, options: PlannerOptions = PlannerOptions(),
                                      var done: CB = NoCallback()) -> Job[SqlResult]:
    """`sql(catalog, text, options)` on a thread of its own. The job works on a copy of the catalog: the caller may go on using
    (and re-registering frames in) its own."""
    return start_job_then(SqlJob(catalog.copy(), text, options.copy()), done^)


struct CsvJob(JobBody):
    comptime Out = CsvResult
    var path: String
    var options: CsvOptions

    def __init__(out self, var path: String, var options: CsvOptions):
        self.path = path^
        self.options = options^

    def run(mut self, ctl: JobCtl) raises -> CsvResult:
        return read_csv(self.path, self.options, ctl)


def read_csv_job[CB: JobDone = NoCallback](path: String, options: CsvOptions = CsvOptions(), var done: CB = NoCallback()) -> Job[CsvResult]:
    """`read_csv(path, options)` on a thread of its own; its progress counts the file's bytes."""
    return start_job_then(CsvJob(path, options.copy()), done^)


struct ScanCsvJob(JobBody):
    comptime Out = ParquetScan
    var path: String
    var options: CsvOptions

    def __init__(out self, var path: String, var options: CsvOptions):
        self.path = path^
        self.options = options^

    def run(mut self, ctl: JobCtl) raises -> ParquetScan:
        return scan_csv(self.path, self.options, ctl)


def scan_csv_job[CB: JobDone = NoCallback](path: String, options: CsvOptions = CsvOptions(), var done: CB = NoCallback()) -> Job[ParquetScan]:
    """`scan_csv(path, options)` on a thread of its own: opening a big CSV reads all of it once, so it takes a while. Its progress counts the file's
    bytes; a cancel is seen between row groups (about `options.group_rows` rows)."""
    return start_job_then(ScanCsvJob(path, options.copy()), done^)


struct ParquetJob(JobBody):
    comptime Out = ParquetResult
    var path: String

    def __init__(out self, var path: String):
        self.path = path^

    def run(mut self, ctl: JobCtl) raises -> ParquetResult:
        return read_parquet(self.path, ctl)


def read_parquet_job[CB: JobDone = NoCallback](path: String, var done: CB = NoCallback()) -> Job[ParquetResult]:
    """`read_parquet(path)` on a thread of its own."""
    return start_job_then(ParquetJob(path), done^)


def frame_snapshot(f: Frame) -> Frame:
    """A frame of its own with `f`'s edits as they are now (cells, appended and removed rows, words, merges, log and what was undone, formulas): the
    caller may go on editing and reading `f` while a job works on the snapshot. A frame over a file shares the file (a
    `ParquetScan` is a shared handle), not its chunk cache."""
    var out = Frame(f.base)
    out.scan = f.scan.copy()
    out.nbase = f.nbase
    out.starts = f.starts.copy()
    out.cache = ArcPointer(_ChunkCache())
    out.cells = f.cells.copy()
    out.added = f.added.copy()
    out.removed = f.removed.copy()
    out.words = f.words.copy()
    out.word_codes = f.word_codes.copy()
    out.aliases = f.aliases.copy()
    out.log = f.log.copy()
    out.edits = f.edits
    out.formulas = f.formulas.copy()
    out.redo_log = f.redo_log.copy()
    out.redo_rows = f.redo_rows.copy()
    out.swaps = f.swaps.copy()
    return out^


struct SaveJob(JobBody):
    comptime Out = Int                          # the rows written
    var frame: Frame
    var path: String
    var metadata: String
    var compression: Int
    var allow_narrowing: Bool

    def __init__(out self, var frame: Frame, var path: String, var metadata: String, compression: Int, allow_narrowing: Bool):
        self.frame = frame^
        self.path = path^
        self.metadata = metadata^
        self.compression = compression
        self.allow_narrowing = allow_narrowing

    def run(mut self, ctl: JobCtl) raises -> Int:
        self.frame.save(self.path, self.metadata, self.compression, self.allow_narrowing, ctl)
        return self.frame.rows() - len(self.frame.removed)               # (`rows()` keeps the numbers of removed rows)


def save_job[CB: JobDone = NoCallback](frame: Frame, path: String, metadata: String = "", compression: Int = ZSTD,
                                       allow_narrowing: Bool = False, var done: CB = NoCallback()) -> Job[Int]:
    """`frame.save(path, ...)` on a thread of its own, over a snapshot of the frame taken now (later edits are not in the file).
    A cancel leaves the old file; the result is the rows written."""
    return start_job_then(SaveJob(frame_snapshot(frame), path, metadata, compression, allow_narrowing), done^)


struct WriteJob(JobBody):
    comptime Out = Int
    var table: SharedTable
    var path: String
    var metadata: String
    var compression: Int
    var row_group_rows: Int

    def __init__(out self, table: SharedTable, var path: String, var metadata: String, compression: Int, row_group_rows: Int):
        self.table = table
        self.path = path^
        self.metadata = metadata^
        self.compression = compression
        self.row_group_rows = row_group_rows

    def run(mut self, ctl: JobCtl) raises -> Int:
        write_parquet(self.path, self.table, self.metadata, self.compression, self.row_group_rows, ctl)
        return self.table.rows()


def write_parquet_job[CB: JobDone = NoCallback](table: SharedTable, path: String, metadata: String = "", compression: Int = ZSTD,
                                                row_group_rows: Int = DEFAULT_ROW_GROUP, var done: CB = NoCallback()) -> Job[Int]:
    """`write_parquet(path, table, ...)` on a thread of its own; a cancel leaves the old file; the result is the rows written."""
    return start_job_then(WriteJob(table, path, metadata, compression, row_group_rows), done^)


struct PermutationJob(JobBody):
    comptime Out = List[Int]
    var frame: Frame
    var column: Int
    var descending: Bool
    var budget: Int
    var spill_dir: String

    def __init__(out self, var frame: Frame, column: Int, descending: Bool, budget: Int, spill_dir: String):
        self.frame = frame^
        self.column = column
        self.descending = descending
        self.budget = budget
        self.spill_dir = spill_dir

    def run(mut self, ctl: JobCtl) raises -> List[Int]:
        return self.frame.permutation(self.column, self.descending, ctl, self.budget, self.spill_dir)


def permutation_job[CB: JobDone = NoCallback](frame: Frame, column: Int, descending: Bool = False, var done: CB = NoCallback(), budget: Int = 0,
                                              spill_dir: String = "") -> Job[List[Int]]:
    """`frame.permutation(column, descending)` (a sort) on a thread of its own, over a snapshot of the frame taken now. The cancel checks
    are inside the key pass and every radix pass (every 1M elements)."""
    return start_job_then(PermutationJob(frame_snapshot(frame), column, descending, budget, spill_dir), done^)
