"""Window functions from the builder (F-168): `compute_windows(query)`.

    var w = Window().partition_by(col("region")).order_by(col("day")).rows(-2, CURRENT_ROW)
    var q = Query(orders).where(col("qty") > 0).window(row_number().over(Window().partition_by(col("region")).order_by(col("day"))),
                                                          sum(col("price")).over(w).alias("moving"))
    var r = compute_windows(q)            # r.table: the source's columns for the rows `where` keeps, then one column per function
    r.device; r.fallback_reason; r.notes; r.ms

The query's rows are those its `where`s keep, over a table, a Frame (its edits and formula columns) or a Parquet scan (which materialises the columns the
functions and the source's other columns need: the result holds them all). The plan is written as SQL (the expressions as text, columns quoted) and runs
through the SQL path, so the answers are the SQL path's (`winsql.mojo`, `window.mojo`) and the builder has no second implementation. `device`: the CPU runs
windows; AUTO answers on the CPU with `fallback_reason` saying so, and GPU raises. A Query with a group_by or an aggregate takes no windows (write that in SQL).
"""
from .table import Table
from .query import Query, CPU, GPU, AUTO
from .bind import Catalog
from .planner import PlannerOptions
from .sqlexec import SqlResult
from .sqlquery import sql

comptime SOURCE_NAME = "__query"


def window_sql(q: Query) raises -> String:
    """The SQL text of a Query with `window(...)` functions: `SELECT *, f(...) OVER (...) AS name, ... FROM __query [WHERE ...]`."""
    if len(q.windows) == 0:
        raise Error("the query has no window functions: add some with Query.window(...)")
    return q.view_sql(SOURCE_NAME)


def compute_windows(q: Query, device: Int = AUTO) raises -> SqlResult:
    """Runs a Query's window functions (see the module doc): `.table` holds the rows' source columns and the functions' columns. `device` AUTO / CPU run on
    the CPU (`fallback_reason` says so under AUTO); GPU raises."""
    if device == GPU:
        raise Error("window functions run on the CPU only: use device=CPU or AUTO")
    var text = window_sql(q)
    var cat = Catalog()
    cat._put(q.view_source(SOURCE_NAME))
    var opts = PlannerOptions()
    opts.threads = q._threads
    var r = sql(cat, text, opts)
    r.fallback_reason = "window functions run on the CPU" if device == AUTO else String()
    r.notes.append("builder: " + text)
    return r^
