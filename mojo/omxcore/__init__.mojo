"""omxcore: a compute engine (CPU and GPU) with persistence to open Parquet files, usable with no Python (docs/omxcore-mojo-api.md, docs/briefs/mojo-api.md).

Build: `bash scripts/build_mojo_pkg.sh` -> build/omxcore.mojoc (and build/mxframe.mojoc, the same source under the old name, for one stable release). Tests: `pixi run mojo-test`.
"""
from .table import Table, SharedTable, Column, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, F32_COLUMN
from .expr import Expr, col, lit, lit_float, lit_f32, least, greatest
from .query import (Query, QueryResult, Agg, sum, count, mean, min_of, max_of, count_distinct, percentile, median, top_k,
                    stddev, stddev_pop, variance, var_pop,
                    CPU, GPU, AUTO, MORSELS_AUTO, MORSELS_ON, MORSELS_OFF)
from .parquet import write_parquet, read_parquet, scan_parquet, ParquetResult, ParquetScan, ParquetWriter, UNCOMPRESSED, SNAPPY, ZSTD
from .csv import read_csv, CsvOptions, CsvResult, CsvColumn, FailedRow
from .frame import Frame, Formula
from .floatfmt import float_text
from .gpu import DeviceTable, gpu_available, gpu_device
from .sql import parse_sql, SqlStatement
from .bind import Catalog, SqlSource, BoundQuery, bind_sql, SRC_TABLE, SRC_FRAME, SRC_SCAN
from .planner import (Planner, RulePlanner, PlannerOptions, PhysicalPlan, KeyLayout, Stats, ColStat, GROUP_AUTO, GROUP_DENSE, GROUP_HASHED, BUILD_AUTO, BUILD_LEFT, BUILD_RIGHT, JoinStats, JoinPlan, JoinStep, rule_join_order,
                      G_NONE, G_DENSE, G_HASHED)
from .plan import LogicalPlan, PlanNode, build_plan
from .sqlexec import sql, sql_with, SqlResult
from .blocks import range_aggregate, read_block, RangeResult, RangeAgg, MORSEL
# F-175: background jobs (jobs.mojo holds the start functions)
from .job import (Job, JobCtl, JobProgress, JobBody, JobDone, NoCallback, start_job, start_job_then, JOB_RUNNING, JOB_DONE, JOB_FAILED, JOB_CANCELLED, CANCELLED_MESSAGE,
                  STAGE_START, STAGE_PLAN, STAGE_READ, STAGE_AGGREGATE, STAGE_WRITE, STAGE_FINISH, STAGE_SORT, state_name, JOBS_ARE_THREADED)
from .jobs import compute_job, sql_job, read_csv_job, read_parquet_job, save_job, write_parquet_job, permutation_job, frame_snapshot
