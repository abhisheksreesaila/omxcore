# Night scoreboard 2026-11-06 (partial, CPU)

`scripts/gate_scoreboard.py --sf 1,10 --devices cpu --repeats 3 --random 100`, started at 00:03 PST after 5 quiet minutes, then other projects' builds and agents started (load 11 to 13 for most runs); the run was stopped at its 2-hour limit before SF10's random suite finished and before the correctness check, so this summary is computed from the per-run log (median of 3, both engines under the same load). Main at dc37fa9 (F-201 to F-205).

| set | queries | omxcore faster | within 10% | geo. mean DuckDB / omxcore | 2026-11-05 baseline (under load) |
|---|---:|---:|---:|---:|---:|
| random, SF1 | 100 | 84 | 87 | 1.58x | not measured |
| random, SF10 | 59 of 100 | 38 | 39 | 1.26x | not measured |
| TPC-H, SF1 | 22 | 7 | 8 | 0.76x | 0.41x (4 faster) |
| TPC-H, SF10 | 22 | 3 | 3 | 0.51x | 0.23x (0 faster) |

Worst random queries (omxcore ms / DuckDB ms): SF1 r14 82/10 (3-key GROUP BY with TEXT keys, ORDER BY sum LIMIT), r64 80/16 (COUNT DISTINCT over a join), r68 33/12 (COUNT DISTINCT by l_orderkey), r40 22/13 (COUNT DISTINCT); SF10 r11 82/10 (COUNT DISTINCT by p_partkey), r1 576/193 (lineitem x orders x supplier aggregate), r12 27/12 (COUNT DISTINCT over a join), r24 98/44 (ORDER BY ... LIMIT over a filtered scan), r9 141/64 (COUNT DISTINCT over a join). COUNT(DISTINCT) per group is the most common loss.

Worst TPC-H SF10: Q5 481/77, Q11 102/18, Q17 323/66, Q21 1329/359, Q2 90/26, Q8 204/76, Q7 246/97, Q20 161/67.
