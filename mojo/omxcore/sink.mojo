"""Where a row query's ordered rows may go instead of a result table (F-167).

`sql_to_parquet` puts one of these in `PlannerOptions.sink`. A sorting row query that outgrows the sort budget merges its spilled runs
straight into a Parquet file through it, and never holds the result; it says so by storing the rows written at `rows_at`. A query that
did not stream leaves -1 there and returns its table as usual, and the caller writes that table.
"""


struct ResultSink(Copyable, Movable):
    var path: String
    var metadata: String
    var compression: Int
    var row_group_rows: Int
    var rows_at: Int              # the address of one Int64 the caller owns: -1 nothing was streamed; else the rows written to `path`

    def __init__(out self, path: String, metadata: String, compression: Int, row_group_rows: Int, rows_at: Int):
        self.path = path
        self.metadata = metadata
        self.compression = compression
        self.row_group_rows = row_group_rows
        self.rows_at = rows_at
