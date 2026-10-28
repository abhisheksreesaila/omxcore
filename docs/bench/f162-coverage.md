# F-162 coverage: TPC-H 22 and the random suite through `mx sql` against DuckDB (2026-10-28)

What `mx sql` (built from main at 63a62ea, before F-159) can run today, and whether the answer equals DuckDB's. Correctness only: no timing here (the scoreboard is `scripts/gate_scoreboard.py`, run on a quiet night).

**Setup.** DuckDB 1.5.1's `tpch` extension (`dbgen`) wrote SF1 as typed Parquet (DECIMAL(15,2), DATE, BIGINT / INTEGER, VARCHAR) into `~/.cache/omxcore-gate/tpch_sf1/` (`scripts/gate_common.py`, generated on first use, about 9 s). The 22 queries are the texts DuckDB's `tpch_queries()` returns (Q15 already a `WITH`; no query edited). `mx sql --max-rows 0 "<SQL>" -t name=path` for the tables a query names, and the same text through DuckDB over the same files. Compared (`tests/gate_compare.py`): column kinds and DECIMAL scales, INT / DECIMAL / DATE / TEXT exactly, FLOAT to a relative 1e-9, rows in order when the query has `ORDER BY`, as a multiset when it has none.

Run it: `pixi run test tests/test_mx_tpch_gate.py tests/test_mx_random_gate.py -n 8` (about 1 minute). `--runxfail` shows each gap's real error.

## TPC-H SF1: 8 of 22 run and equal DuckDB

| Q | result | engine error today | group |
|---:|---|---|---|
| 1 | equal | | |
| 2 | fails | a scalar subquery is not supported yet (F-159) | F-159 |
| 3 | equal | | |
| 4 | equal | | |
| 5 | equal | | |
| 6 | equal | | |
| 7 | fails | a subquery in FROM is not supported yet (F-159) | F-159 |
| 8 | fails | a subquery in FROM is not supported yet (F-159) | F-159 |
| 9 | fails | a subquery in FROM is not supported yet (F-159) | F-159 |
| 10 | equal | | |
| 11 | fails | a scalar subquery is not supported yet (F-159) | F-159 |
| 12 | equal | | |
| 13 | fails | a subquery in FROM is not supported yet (F-159) | F-159 |
| 14 | fails | a sum or an expression passed the int64 range (about 9.2e18), or an integer past 2^53 met a float64 (it is not exact as a double) | other gap |
| 15 | fails | `WITH` (common table expressions) is not supported yet (F-159) | F-159 |
| 16 | fails | a subquery in a query that also joins tables is not supported yet (F-155) | F-155 |
| 17 | fails | a scalar subquery is not supported yet (F-159) | F-159 |
| 18 | fails | `GROUP` in a subquery is not supported yet (F-159) | F-159 |
| 19 | equal | | |
| 20 | fails | a scalar subquery is not supported yet (F-159) | F-159 |
| 21 | fails | more than one subquery in WHERE is not supported yet (F-155) | F-155 |
| 22 | fails | a subquery in FROM is not supported yet (F-159) | F-159 |

Equal: Q1, Q3, Q4, Q5, Q6, Q10, Q12, Q19 (8). Every failure is an error from the engine, none a wrong answer.

**Grouped**
- **Needs F-159 (CTE, derived table, scalar subquery, `GROUP` in a subquery): 11 queries.** Q2, Q11, Q17, Q20 (scalar subquery); Q7, Q8, Q9, Q13, Q22 (derived table in FROM); Q15 (`WITH`); Q18 (`GROUP BY` inside an `IN` subquery). Q2/Q11/Q17/Q20 may also need the correlated form; Q22 needs `NOT EXISTS` next to a scalar subquery. Whether these reach "equal" after F-159 is for the xfails to say.
- **Needs F-155 (subqueries in a query that also joins; several subqueries in one WHERE): 2 queries.** Q16 (`NOT IN` subquery plus a join), Q21 (two `EXISTS` / `NOT EXISTS` with `<>`).
- **Other gap: 1 query (Q14), a bug, not a missing feature (F-190).** `/` is typed FLOAT (`bind.mojo`), so nothing scales up inside int64. The failure is the cast of a large DECIMAL to DOUBLE: `OP_I2F` (`query.mojo`) rejects the raw DECIMAL integer when it is past 2^53 and raises "a sum or an expression passed the int64 range ... integer past 2^53 met a float64", where DuckDB rounds to the nearest double. Isolated: `sum(l_extendedprice) / 3` and `100.00 * sum(l_extendedprice)` alone work; `100.00 * sum(l_extendedprice) / 2.0` fails (raw sum about 2.3e17 > 2^53); the Q14 shape works when filtered to `l_orderkey < 100000`. Any DECIMAL aggregate with a raw value past 2^53 divided or cast to DOUBLE will hit it (Q8 is the same shape, behind F-159 for now).

## Random suite: 80 of 80 seeds equal DuckDB

`scripts/bench_random/gen.py` (the F-044 / F-058 generator: filters, joins along the foreign keys, group-by, HAVING, top-N, COUNT(DISTINCT), SELECT DISTINCT), seeds 0-79, rendered as SQL over the typed tables (`gate_common.random_sql`: date filters become `DATE '…'` literals; the constants come from the quantiles of the typed SF1 files). All 80 equal DuckDB, FLOAT to 1e-9. No gap, so none is xfailed. The comparison itself is tested to fail on a wrong answer (`test_the_comparison_sees_a_different_answer`).

**Ties at a LIMIT cut.** The comparison is exact and ordered, with no tie handling. Q3, Q10 and Q18 order by revenue / price and cut at a LIMIT; if two rows ever tied at the cut and the engines picked different ones, it would show as a false alarm (a row differing), not a hidden bug. None does at SF1 today.

**Timing note.** The scoreboard runs `mx` with `--stream never`, so a file is never streamed inside the clock.

## Not covered here

- **The repo's FLOAT Parquet** (`data/tpch_sf1`: FLOAT columns, dates as day numbers, the data of the 2026-10-17 scoreboard). The ticket's second variant is not in the tests: TPC-H text has `DATE '1994-01-01'` literals that cannot compare with an INTEGER column, and DuckDB would need typed views over the raw files (`tests/tpch_duckdb_sql.py`) while `mx` reads the raw FLOAT, which makes the check about float32 rounding rather than the engine. The typed dbgen data is the exact-answer variant.
- **EXPLAINs** (the ticket records them, does not gate on them): not done in this harness.
- **omxplorer's goal checks** (second F-162 check): not touched.
- **`--device gpu|auto`**: `mx sql` reports `device: CPU (SQL runs on the CPU)` for every device, so the scoreboard's `gpu` and `auto` columns time the same CPU engine until the CLI honours them. `--devices cpu` is enough for a night until then.
- **SF10 and timings**: the scoreboard does the SF10 runs; the test data is SF1.
