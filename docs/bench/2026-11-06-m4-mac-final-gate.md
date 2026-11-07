# M8 gate scoreboard: 2026-11-06

omxcore `mx sql` (cpu, gpu, auto) against DuckDB 1.5.6, engine-only first runs: a fresh process per (engine, query, run), the tables loaded outside the clock, no warm-up, the median of 3 run(s); omxcore runs with `--stream never` (the file is read whole before the query, never streamed inside its clock), DuckDB queries preloaded tables. Typed TPC-H Parquet (DuckDB dbgen: DECIMAL, DATE), queries as DuckDB's tpch extension writes them. Started 2026-11-06 17:11; load average before each run: median 2.6, max 4.1 (NOT QUIET: re-run on an idle machine). Command: `scripts/gate_scoreboard.py --sf 1 --devices cpu,gpu,auto --repeats 3 --random 50 --out docs/bench/2026-11-06-m4-mac-final-gate.md`.

Device as the CLI reported it: `device: CPU (AUTO: a join runs on the CPU)`; `device: CPU (AUTO: a query without aggregates runs on the CPU)`; `device: CPU (AUTO: count(DISTINCT), median, quantiles and min / max of TEXT run on the CPU)`; `device: CPU (AUTO: no table resident on the GPU (attach a DeviceTable, or --device gpu uploads it for the query))`; `device: CPU (AUTO: the program uses an instruction the GPU kernel does not run (opcode 19: text, dates, IN lists, casts or float arithmetic))`; `device: CPU (AUTO: the table has 10000 rows, under the 1000000 AUTO runs on the GPU)`; `device: CPU (AUTO: the table has 118274 rows, under the 1000000 AUTO runs on the GPU)`; `device: CPU (AUTO: the table has 150000 rows, under the 1000000 AUTO runs on the GPU)`; `device: CPU (AUTO: the table has 200000 rows, under the 1000000 AUTO runs on the GPU)`; `device: CPU (AUTO: the table has 214377 rows, under the 1000000 AUTO runs on the GPU)`; `device: CPU (AUTO: the table has 2603 rows, under the 1000000 AUTO runs on the GPU)`; `device: CPU (AUTO: the table has 319404 rows, under the 1000000 AUTO runs on the GPU)`; `device: CPU (AUTO: the table has 412 rows, under the 1000000 AUTO runs on the GPU)`; `device: CPU (AUTO: the table has 57218 rows, under the 1000000 AUTO runs on the GPU)`; `device: CPU (AUTO: the table has 5924 rows, under the 1000000 AUTO runs on the GPU)`; `device: CPU (AUTO: the table has 6088 rows, under the 1000000 AUTO runs on the GPU)`; `device: CPU (AUTO: the table has 6384 rows, under the 1000000 AUTO runs on the GPU)`; `device: CPU (AUTO: the table has 642 rows, under the 1000000 AUTO runs on the GPU)`; `device: CPU (AUTO: the table has 73089 rows, under the 1000000 AUTO runs on the GPU)`; `device: CPU (AUTO: the table has 800000 rows, under the 1000000 AUTO runs on the GPU)`; `device: CPU (GPU asked for; a join runs on the CPU)`; `device: CPU (GPU asked for; a query without aggregates runs on the CPU)`; `device: CPU (GPU asked for; count(DISTINCT), median, quantiles and min / max of TEXT run on the CPU)`; `device: CPU (GPU asked for; hashed groups run on the CPU (the GPU kernel takes no groups or dense groups))`; `device: CPU (GPU asked for; the GPU kernel declined the program (code -6: its groups or registers do not fit the atomic-free modes))`; `device: CPU (GPU asked for; the program uses an instruction the GPU kernel does not run (opcode 19: text, dates, IN lists, casts or float arithmetic))`; `device: CPU (device CPU was asked for)`; `device: GPU (GPU asked for, the table uploaded for this query)`. Where it says the SQL runs on the CPU, the `gpu` and `auto` columns time the same CPU engine.

## TPC-H SF1

| query | DuckDB ms | omxcore cpu ms | ratio cpu | omxcore gpu ms | ratio gpu | omxcore auto ms | ratio auto |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 21.6 | 18.0 | 1.20x | 317.3 | 0.07x | 17.8 | 1.21x |
| Q2 | 5.0 | 9.2 | 0.54x | 9.1 | 0.54x | 9.2 | 0.54x |
| Q3 | 8.7 | 20.4 | 0.43x | 86.1 | 0.10x | 20.5 | 0.43x |
| Q4 | 9.5 | 11.7 | 0.82x | 62.2 | 0.15x | 11.8 | 0.81x |
| Q5 | 8.4 | 19.2 | 0.43x | 60.0 | 0.14x | 19.3 | 0.43x |
| Q6 | 3.3 | 4.4 | 0.74x | 128.6 | 0.03x | 4.1 | 0.80x |
| Q7 | 9.3 | 22.0 | 0.42x | 52.3 | 0.18x | 21.4 | 0.44x |
| Q8 | 7.6 | 32.3 | 0.24x | 103.8 | 0.07x | 27.8 | 0.27x |
| Q9 | 51.4 | 54.0 | 0.95x | 80.3 | 0.64x | 48.2 | 1.07x |
| Q10 | 17.7 | 14.4 | 1.22x | 66.5 | 0.27x | 14.7 | 1.21x |
| Q11 | 3.0 | 8.6 | 0.36x | 8.5 | 0.36x | 8.6 | 0.36x |
| Q12 | 13.2 | 16.5 | 0.80x | 16.7 | 0.79x | 16.5 | 0.80x |
| Q13 | 32.4 | 32.8 | 0.99x | 61.9 | 0.52x | 32.9 | 0.99x |
| Q14 | 7.0 | 6.8 | 1.03x | 32.9 | 0.21x | 6.7 | 1.05x |
| Q15 | 4.7 | 5.4 | 0.87x | 149.4 | 0.03x | 5.3 | 0.89x |
| Q16 | 11.1 | 15.0 | 0.74x | 15.1 | 0.74x | 14.9 | 0.75x |
| Q17 | 6.7 | 17.0 | 0.40x | 46.2 | 0.15x | 17.0 | 0.40x |
| Q18 | 21.7 | 37.9 | 0.57x | 38.3 | 0.57x | 37.4 | 0.58x |
| Q19 | 23.5 | 10.5 | 2.24x | 10.4 | 2.27x | 10.8 | 2.19x |
| Q20 | 7.4 | 11.1 | 0.67x | 37.1 | 0.20x | 10.7 | 0.70x |
| Q21 | 26.1 | 54.7 | 0.48x | 116.5 | 0.22x | 54.4 | 0.48x |
| Q22 | 7.6 | 12.3 | 0.61x | 33.7 | 0.22x | 12.3 | 0.61x |

| device | queries | equal DuckDB | faster than DuckDB | within 10% | geo. mean DuckDB / omxcore | 90% gate |
|---|---:|---:|---:|---:|---:|---|
| cpu | 22 | 22 | 4 | 6 | 0.67x | NOT MET (4/22, needs 20) |
| gpu | 22 | 22 | 1 | 1 | 0.23x | NOT MET (1/22, needs 20) |
| auto | 22 | 22 | 5 | 6 | 0.68x | NOT MET (5/22, needs 20) |

## random SF1

| query | DuckDB ms | omxcore cpu ms | ratio cpu | omxcore gpu ms | ratio gpu | omxcore auto ms | ratio auto |
|---|---:|---:|---:|---:|---:|---:|---:|
| r0 | 2.3 | 0.5 | 4.79x | 0.5 | 4.79x | 0.5 | 4.80x |
| r1 | 22.5 | 17.8 | 1.27x | 43.2 | 0.52x | 17.9 | 1.26x |
| r2 | 15.9 | 5.5 | 2.89x | 206.5 | 0.08x | 7.2 | 2.22x |
| r3 | 9.4 | 22.2 | 0.42x | 22.6 | 0.42x | 22.1 | 0.42x |
| r4 | 5.8 | 4.7 | 1.23x | 53.4 | 0.11x | 4.8 | 1.22x |
| r5 | 9.0 | 4.6 | 1.96x | 4.6 | 1.96x | 4.6 | 1.97x |
| r6 | 4.9 | 0.8 | 6.38x | 0.8 | 6.48x | 0.8 | 6.47x |
| r7 | 6.2 | 5.1 | 1.20x | 141.4 | 0.04x | 4.7 | 1.30x |
| r8 | 8.0 | 3.6 | 2.24x | 3.5 | 2.26x | 3.6 | 2.22x |
| r9 | 6.6 | 6.8 | 0.97x | 31.7 | 0.21x | 7.1 | 0.92x |
| r10 | 3.2 | 0.6 | 5.51x | 0.6 | 5.48x | 0.6 | 5.53x |
| r11 | 2.4 | 1.0 | 2.42x | 1.0 | 2.44x | 1.0 | 2.41x |
| r12 | 1.7 | 1.9 | 0.92x | 1.9 | 0.90x | 1.9 | 0.90x |
| r13 | 5.1 | 4.3 | 1.19x | 4.3 | 1.17x | 4.4 | 1.17x |
| r14 | 5.3 | 9.1 | 0.59x | 41.3 | 0.13x | 9.1 | 0.59x |
| r15 | 50.2 | 31.6 | 1.59x | 57.2 | 0.88x | 32.4 | 1.55x |
| r16 | 2.0 | 0.4 | 5.29x | 0.4 | 5.26x | 0.4 | 5.29x |
| r17 | 1.7 | 0.6 | 2.91x | 24.2 | 0.07x | 0.6 | 2.86x |
| r18 | 20.4 | 19.5 | 1.04x | 59.7 | 0.34x | 19.3 | 1.05x |
| r19 | 36.8 | 11.0 | 3.34x | 10.9 | 3.37x | 11.0 | 3.33x |
| r20 | 6.9 | 4.7 | 1.47x | 59.4 | 0.12x | 4.6 | 1.49x |
| r21 | 2.7 | 0.7 | 3.73x | 0.7 | 3.73x | 0.7 | 3.71x |
| r22 | 5.8 | 3.5 | 1.67x | 27.1 | 0.22x | 3.4 | 1.70x |
| r23 | 20.0 | 11.4 | 1.76x | 11.5 | 1.74x | 11.4 | 1.75x |
| r24 | 5.6 | 2.9 | 1.91x | 2.9 | 1.89x | 2.9 | 1.92x |
| r25 | 5.8 | 4.4 | 1.32x | 4.3 | 1.34x | 4.4 | 1.32x |
| r26 | 4.8 | 3.7 | 1.29x | 59.6 | 0.08x | 3.7 | 1.29x |
| r27 | 6.4 | 3.6 | 1.77x | 3.6 | 1.77x | 3.7 | 1.75x |
| r28 | 23.8 | 7.5 | 3.19x | 130.7 | 0.18x | 7.5 | 3.16x |
| r29 | 14.6 | 4.7 | 3.14x | 30.2 | 0.49x | 4.7 | 3.10x |
| r30 | 4.0 | 0.9 | 4.59x | 43.0 | 0.09x | 0.9 | 4.41x |
| r31 | 3.0 | 0.4 | 7.22x | 29.0 | 0.10x | 0.4 | 7.24x |
| r32 | 3.7 | 1.2 | 3.02x | 40.4 | 0.09x | 1.2 | 3.02x |
| r33 | 7.3 | 5.1 | 1.43x | 5.0 | 1.46x | 5.1 | 1.44x |
| r34 | 2.0 | 1.2 | 1.64x | 39.1 | 0.05x | 1.2 | 1.61x |
| r35 | 1.8 | 1.3 | 1.38x | 40.1 | 0.04x | 1.3 | 1.36x |
| r36 | 2.1 | 2.9 | 0.72x | 117.9 | 0.02x | 2.9 | 0.73x |
| r37 | 19.7 | 7.6 | 2.59x | 7.6 | 2.58x | 7.5 | 2.62x |
| r38 | 0.9 | 0.4 | 2.17x | 0.4 | 2.21x | 0.4 | 2.17x |
| r39 | 6.4 | 4.7 | 1.35x | 40.5 | 0.16x | 4.7 | 1.35x |
| r40 | 7.6 | 7.2 | 1.05x | 7.4 | 1.03x | 7.3 | 1.04x |
| r41 | 1.7 | 1.0 | 1.72x | 1.0 | 1.74x | 1.0 | 1.73x |
| r42 | 64.1 | 41.3 | 1.55x | 235.9 | 0.27x | 41.1 | 1.56x |
| r43 | 24.0 | 19.0 | 1.26x | 19.2 | 1.25x | 19.1 | 1.26x |
| r44 | 5.6 | 5.5 | 1.01x | 4.8 | 1.16x | 5.0 | 1.11x |
| r45 | 2.9 | 0.6 | 4.50x | 0.6 | 4.51x | 0.6 | 4.49x |
| r46 | 2.8 | 0.5 | 6.12x | 27.4 | 0.10x | 0.5 | 6.06x |
| r47 | 5.2 | 5.0 | 1.04x | 5.1 | 1.02x | 5.2 | 0.99x |
| r48 | 1.9 | 1.4 | 1.33x | 39.7 | 0.05x | 1.4 | 1.34x |
| r49 | 2.0 | 1.3 | 1.48x | 42.3 | 0.05x | 1.3 | 1.48x |

| device | queries | equal DuckDB | faster than DuckDB | within 10% | geo. mean DuckDB / omxcore | 90% gate |
|---|---:|---:|---:|---:|---:|---|
| cpu | 50 | 50 | 45 | 47 | 1.89x | PASS (45/50, needs 45) |
| gpu | 50 | 50 | 23 | 23 | 0.49x | NOT MET (23/50, needs 45) |
| auto | 50 | 50 | 44 | 46 | 1.88x | NOT MET (44/50, needs 45) |

## omxcore failures and differences (0)


## Verdict

- TPC-H SF1 cpu: NOT MET (4/22 faster than DuckDB)
- TPC-H SF1 gpu: NOT MET (1/22 faster than DuckDB)
- TPC-H SF1 auto: NOT MET (5/22 faster than DuckDB)
- random SF1 cpu: PASS (45/50 faster than DuckDB)
- random SF1 gpu: NOT MET (23/50 faster than DuckDB)
- random SF1 auto: NOT MET (44/50 faster than DuckDB)

**The 90% gate: NOT MET**

## Queries run

- Q1: `SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty, sum(l_extendedprice) AS sum_base_price, sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price, sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, avg(l_quantity) AS avg_qty, avg(l_extendedprice) AS avg_price, avg(l_discount) AS avg_disc, count(*) AS count_order FROM lineitem WHERE l_shipdate <= CAST('1998-09-02' A`
- Q2: `SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment FROM part, supplier, partsupp, nation, region WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND p_size = 15 AND p_type LIKE '%BRASS' AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'EUROPE' AND ps_supplycost = ( SELECT min(ps_supplycost) FROM partsupp, supplier, nation, region WHER`
- Q3: `SELECT l_orderkey, sum(l_extendedprice * (1 - l_discount)) AS revenue, o_orderdate, o_shippriority FROM customer, orders, lineitem WHERE c_mktsegment = 'BUILDING' AND c_custkey = o_custkey AND l_orderkey = o_orderkey AND o_orderdate < CAST('1995-03-15' AS date) AND l_shipdate > CAST('1995-03-15' AS date) GROUP BY l_orderkey, o_orderdate, o_shippriority ORDER BY revenue DESC, o_orderdate LIMIT 10`
- Q4: `SELECT o_orderpriority, count(*) AS order_count FROM orders WHERE o_orderdate >= CAST('1993-07-01' AS date) AND o_orderdate < CAST('1993-10-01' AS date) AND EXISTS ( SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate) GROUP BY o_orderpriority ORDER BY o_orderpriority`
- Q5: `SELECT n_name, sum(l_extendedprice * (1 - l_discount)) AS revenue FROM customer, orders, lineitem, supplier, nation, region WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND l_suppkey = s_suppkey AND c_nationkey = s_nationkey AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'ASIA' AND o_orderdate >= CAST('1994-01-01' AS date) AND o_orderdate < CAST('1995-01-01' AS`
- Q6: `SELECT sum(l_extendedprice * l_discount) AS revenue FROM lineitem WHERE l_shipdate >= CAST('1994-01-01' AS date) AND l_shipdate < CAST('1995-01-01' AS date) AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24`
- Q7: `SELECT supp_nation, cust_nation, l_year, sum(volume) AS revenue FROM ( SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation, extract(year FROM l_shipdate) AS l_year, l_extendedprice * (1 - l_discount) AS volume FROM supplier, lineitem, orders, customer, nation n1, nation n2 WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey AND c_custkey = o_custkey AND s_nationkey = n1.n_nationkey AND c`
- Q8: `SELECT o_year, sum( CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) / sum(volume) AS mkt_share FROM ( SELECT extract(year FROM o_orderdate) AS o_year, l_extendedprice * (1 - l_discount) AS volume, n2.n_name AS nation FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region WHERE p_partkey = l_partkey AND s_suppkey = l_suppkey AND l_orderkey = o_orderkey AND o_custkey = c_c`
- Q9: `SELECT nation, o_year, sum(amount) AS sum_profit FROM ( SELECT n_name AS nation, extract(year FROM o_orderdate) AS o_year, l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount FROM part, supplier, lineitem, partsupp, orders, nation WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey AND ps_partkey = l_partkey AND p_partkey = l_partkey AND o_orderkey = l_orderkey AND s_natio`
- Q10: `SELECT c_custkey, c_name, sum(l_extendedprice * (1 - l_discount)) AS revenue, c_acctbal, n_name, c_address, c_phone, c_comment FROM customer, orders, lineitem, nation WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND o_orderdate >= CAST('1993-10-01' AS date) AND o_orderdate < CAST('1994-01-01' AS date) AND l_returnflag = 'R' AND c_nationkey = n_nationkey GROUP BY c_custkey, c_name, c_acc`
- Q11: `SELECT ps_partkey, sum(ps_supplycost * ps_availqty) AS value FROM partsupp, supplier, nation WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY' GROUP BY ps_partkey HAVING sum(ps_supplycost * ps_availqty) > ( SELECT sum(ps_supplycost * ps_availqty) * 0.0001000000 FROM partsupp, supplier, nation WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'G`
- Q12: `SELECT l_shipmode, sum( CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count, sum( CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count FROM orders, lineitem WHERE o_orderkey = l_orderkey AND l_shipmode IN ('MAIL', 'SHIP') AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate AND `
- Q13: `SELECT c_count, count(*) AS custdist FROM ( SELECT c_custkey, count(o_orderkey) FROM customer LEFT OUTER JOIN orders ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%' GROUP BY c_custkey) AS c_orders (c_custkey, c_count) GROUP BY c_count ORDER BY custdist DESC, c_count DESC`
- Q14: `SELECT 100.00 * sum( CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM lineitem, part WHERE l_partkey = p_partkey AND l_shipdate >= date '1995-09-01' AND l_shipdate < CAST('1995-10-01' AS date)`
- Q15: `WITH revenue AS ( SELECT l_suppkey AS supplier_no, sum(l_extendedprice * (1 - l_discount)) AS total_revenue FROM lineitem WHERE l_shipdate >= CAST('1996-01-01' AS date) AND l_shipdate < CAST('1996-04-01' AS date) GROUP BY supplier_no ) SELECT s_suppkey, s_name, s_address, s_phone, total_revenue FROM supplier, revenue WHERE s_suppkey = supplier_no AND total_revenue = ( SELECT max(total_revenue) FRO`
- Q16: `SELECT p_brand, p_type, p_size, count(DISTINCT ps_suppkey) AS supplier_cnt FROM partsupp, part WHERE p_partkey = ps_partkey AND p_brand <> 'Brand#45' AND p_type NOT LIKE 'MEDIUM POLISHED%' AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9) AND ps_suppkey NOT IN ( SELECT s_suppkey FROM supplier WHERE s_comment LIKE '%Customer%Complaints%') GROUP BY p_brand, p_type, p_size ORDER BY supplier_cnt DESC, p_br`
- Q17: `SELECT sum(l_extendedprice) / 7.0 AS avg_yearly FROM lineitem, part WHERE p_partkey = l_partkey AND p_brand = 'Brand#23' AND p_container = 'MED BOX' AND l_quantity < ( SELECT 0.2 * avg(l_quantity) FROM lineitem WHERE l_partkey = p_partkey)`
- Q18: `SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity) FROM customer, orders, lineitem WHERE o_orderkey IN ( SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING sum(l_quantity) > 300) AND c_custkey = o_custkey AND o_orderkey = l_orderkey GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice ORDER BY o_totalprice DESC, o_orderdate LIMIT 100`
- Q19: `SELECT sum(l_extendedprice * (1 - l_discount)) AS revenue FROM lineitem, part WHERE (p_partkey = l_partkey AND p_brand = 'Brand#12' AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') AND l_quantity >= 1 AND l_quantity <= 1 + 10 AND p_size BETWEEN 1 AND 5 AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON') OR (p_partkey = l_partkey AND p_brand = 'Brand#23' AND `
- Q20: `SELECT s_name, s_address FROM supplier, nation WHERE s_suppkey IN ( SELECT ps_suppkey FROM partsupp WHERE ps_partkey IN ( SELECT p_partkey FROM part WHERE p_name LIKE 'forest%') AND ps_availqty > ( SELECT 0.5 * sum(l_quantity) FROM lineitem WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey AND l_shipdate >= CAST('1994-01-01' AS date) AND l_shipdate < CAST('1995-01-01' AS date))) AND s_nation`
- Q21: `SELECT s_name, count(*) AS numwait FROM supplier, lineitem l1, orders, nation WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey AND o_orderstatus = 'F' AND l1.l_receiptdate > l1.l_commitdate AND EXISTS ( SELECT * FROM lineitem l2 WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey) AND NOT EXISTS ( SELECT * FROM lineitem l3 WHERE l3.l_orderkey = l1.l_orderkey AND l3.l_`
- Q22: `SELECT cntrycode, count(*) AS numcust, sum(c_acctbal) AS totacctbal FROM ( SELECT substring(c_phone FROM 1 FOR 2) AS cntrycode, c_acctbal FROM customer WHERE substring(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17') AND c_acctbal > ( SELECT avg(c_acctbal) FROM customer WHERE c_acctbal > 0.00 AND substring(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')) AND `
- r0: `SELECT c_custkey, c_acctbal FROM customer WHERE (c_mktsegment IN ('BUILDING', 'FURNITURE', 'MACHINERY')) AND (c_acctbal < 3166.7) ORDER BY c_acctbal ASC, c_custkey ASC LIMIT 200`
- r1: `SELECT l_linestatus, SUM(l_extendedprice * (1 - l_discount)) AS a0, AVG(l_quantity) AS a1, MAX(l_extendedprice * (1 - l_discount)) AS a2 FROM lineitem JOIN orders ON l_orderkey = o_orderkey JOIN supplier ON l_suppkey = s_suppkey WHERE (l_extendedprice < 10826.64) AND (o_orderpriority IN ('4-NOT SPECIFIED', '5-LOW')) AND (l_shipinstruct = 'TAKE BACK RETURN') GROUP BY l_linestatus ORDER BY a0 DESC, `
- r2: `SELECT l_linestatus, l_shipmode, AVG(l_extendedprice) AS a0, AVG(l_extendedprice) AS a1, COUNT(l_tax) AS a2 FROM lineitem WHERE TRUE GROUP BY l_linestatus, l_shipmode`
- r3: `SELECT s_nationkey, p_size, SUM(p_size) AS a0, AVG(l_tax) AS a1 FROM lineitem JOIN supplier ON l_suppkey = s_suppkey JOIN part ON l_partkey = p_partkey WHERE TRUE GROUP BY s_nationkey, p_size`
- r4: `SELECT o_orderpriority, o_orderstatus, SUM(o_totalprice) AS a0, COUNT(c_acctbal) AS a1, MIN(o_totalprice) AS a2 FROM orders JOIN customer ON o_custkey = c_custkey WHERE (o_totalprice < 144409.04) AND (o_orderdate >= DATE '1994-07-06' AND o_orderdate <= DATE '1995-04-20') AND (c_acctbal < 6265.79) GROUP BY o_orderpriority, o_orderstatus HAVING COUNT(c_acctbal) > 10`
- r5: `SELECT p_container, SUM(s_acctbal) AS a0, AVG(ps_availqty) AS a1, AVG(ps_supplycost) AS a2, SUM(ps_supplycost) AS a3 FROM partsupp JOIN part ON ps_partkey = p_partkey JOIN supplier ON ps_suppkey = s_suppkey JOIN nation ON s_nationkey = n_nationkey WHERE TRUE GROUP BY p_container ORDER BY a0 DESC, p_container ASC LIMIT 10`
- r6: `SELECT p_mfgr, p_size, SUM(p_size) AS a0 FROM part WHERE (p_size < 43) AND (p_mfgr IN ('Manufacturer#1', 'Manufacturer#2', 'Manufacturer#3', 'Manufacturer#5')) GROUP BY p_mfgr, p_size`
- r7: `SELECT MAX(l_extendedprice * (1 - l_discount)) AS a0 FROM lineitem WHERE (l_quantity < 24.0) AND (l_extendedprice >= 5087.01 AND l_extendedprice <= 19458.72) AND (l_returnflag <> 'A')`
- r8: `SELECT p_type, MAX(ps_availqty) AS a0, MAX(s_acctbal) AS a1 FROM partsupp JOIN part ON ps_partkey = p_partkey JOIN supplier ON ps_suppkey = s_suppkey WHERE TRUE GROUP BY p_type ORDER BY a0 DESC, p_type ASC`
- r9: `SELECT ps_partkey, COUNT(DISTINCT p_container) AS n FROM partsupp JOIN part ON ps_partkey = p_partkey WHERE (p_mfgr = 'Manufacturer#3') GROUP BY ps_partkey ORDER BY n DESC, ps_partkey ASC`
- r10: `SELECT c_mktsegment, c_nationkey, SUM(c_acctbal) AS a0, SUM(c_acctbal) AS a1, MAX(c_acctbal) AS a2 FROM customer WHERE (c_acctbal >= 5817.34) AND (c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'FURNITURE', 'MACHINERY')) GROUP BY c_mktsegment, c_nationkey ORDER BY a0 DESC, c_mktsegment ASC, c_nationkey ASC LIMIT 10`
- r11: `SELECT p_partkey, COUNT(DISTINCT p_size) AS n FROM part WHERE (p_mfgr IN ('Manufacturer#1', 'Manufacturer#2', 'Manufacturer#4', 'Manufacturer#5')) AND (p_brand IN ('Brand#13', 'Brand#43')) AND (p_size >= 3 AND p_size <= 13) GROUP BY p_partkey ORDER BY n DESC, p_partkey ASC LIMIT 100`
- r12: `SELECT c_nationkey, COUNT(DISTINCT o_orderpriority) AS n FROM orders JOIN customer ON o_custkey = c_custkey WHERE (o_orderdate >= DATE '1997-05-27') AND (o_orderpriority IN ('1-URGENT', '2-HIGH', '3-MEDIUM', '5-LOW')) AND (o_orderstatus <> 'O') GROUP BY c_nationkey HAVING COUNT(DISTINCT o_orderpriority) > 1 ORDER BY n DESC, c_nationkey ASC`
- r13: `SELECT o_orderstatus, COUNT(c_acctbal) AS a0, SUM(c_acctbal) AS a1 FROM orders JOIN customer ON o_custkey = c_custkey JOIN nation ON c_nationkey = n_nationkey WHERE (n_name <> 'RUSSIA') GROUP BY o_orderstatus HAVING COUNT(c_acctbal) > 20`
- r14: `SELECT o_custkey, o_orderpriority, o_orderstatus, SUM(o_totalprice) AS a0, MIN(o_totalprice) AS a1, MIN(o_totalprice) AS a2 FROM orders WHERE (o_orderpriority = '4-NOT SPECIFIED') AND (o_totalprice >= 166120.85) GROUP BY o_custkey, o_orderpriority, o_orderstatus ORDER BY a0 DESC, o_custkey ASC, o_orderpriority ASC, o_orderstatus ASC LIMIT 10`
- r15: `SELECT o_orderstatus, l_shipinstruct, l_linestatus, MAX(o_totalprice) AS a0, MAX(l_discount) AS a1, SUM(p_size) AS a2 FROM lineitem JOIN supplier ON l_suppkey = s_suppkey JOIN orders ON l_orderkey = o_orderkey JOIN part ON l_partkey = p_partkey WHERE (l_quantity < 43.0) GROUP BY o_orderstatus, l_shipinstruct, l_linestatus ORDER BY a0 DESC, o_orderstatus ASC, l_shipinstruct ASC, l_linestatus ASC LI`
- r16: `SELECT MIN(p_size) AS a0, COUNT(p_size) AS a1, SUM(p_size) AS a2 FROM part WHERE (p_container IN ('MED BAG', 'SM CAN')) AND (p_type <> 'MEDIUM BRUSHED COPPER') AND (p_size < 19)`
- r17: `SELECT DISTINCT n_name, c_mktsegment FROM customer JOIN nation ON c_nationkey = n_nationkey WHERE (c_acctbal < -784.52) AND (c_mktsegment <> 'HOUSEHOLD')`
- r18: `SELECT p_brand, l_shipinstruct, COUNT(l_quantity) AS a0, MIN(p_size) AS a1, SUM(o_totalprice) AS a2 FROM lineitem JOIN part ON l_partkey = p_partkey JOIN supplier ON l_suppkey = s_suppkey JOIN orders ON l_orderkey = o_orderkey WHERE (o_totalprice >= 177206.8 AND o_totalprice <= 212358.47) GROUP BY p_brand, l_shipinstruct ORDER BY a0 DESC, p_brand ASC, l_shipinstruct ASC LIMIT 10`
- r19: `SELECT l_shipmode, MIN(l_discount) AS a0, MIN(l_extendedprice * (1 - l_discount)) AS a1, AVG(l_extendedprice) AS a2, SUM(l_quantity) AS a3 FROM lineitem WHERE (l_shipinstruct IN ('COLLECT COD', 'DELIVER IN PERSON', 'NONE', 'TAKE BACK RETURN')) GROUP BY l_shipmode ORDER BY a0 ASC, l_shipmode ASC`
- r20: `SELECT o_orderpriority, MAX(c_acctbal) AS a0, MAX(o_totalprice) AS a1, SUM(c_acctbal) AS a2 FROM orders JOIN customer ON o_custkey = c_custkey JOIN nation ON c_nationkey = n_nationkey WHERE (o_orderdate >= DATE '1995-07-25') GROUP BY o_orderpriority`
- r21: `SELECT c_mktsegment, COUNT(c_acctbal) AS a0, MAX(c_acctbal) AS a1 FROM customer JOIN nation ON c_nationkey = n_nationkey WHERE (c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'FURNITURE', 'MACHINERY')) AND (c_acctbal >= 7136.99) GROUP BY c_mktsegment ORDER BY a0 DESC, c_mktsegment ASC LIMIT 10`
- r22: `SELECT o_orderpriority, SUM(c_acctbal) AS a0, AVG(o_totalprice) AS a1, AVG(c_acctbal) AS a2 FROM orders JOIN customer ON o_custkey = c_custkey WHERE (c_mktsegment = 'FURNITURE') GROUP BY o_orderpriority`
- r23: `SELECT l_returnflag, s_nationkey, COUNT(l_tax) AS a0 FROM lineitem JOIN supplier ON l_suppkey = s_suppkey WHERE (l_shipinstruct IN ('DELIVER IN PERSON', 'TAKE BACK RETURN')) AND (s_acctbal >= -787.58) GROUP BY l_returnflag, s_nationkey`
- r24: `SELECT l_orderkey, l_linenumber, l_quantity, l_discount FROM lineitem WHERE (l_receiptdate < DATE '1997-11-15') ORDER BY l_quantity ASC, l_orderkey ASC, l_linenumber ASC LIMIT 10`
- r25: `SELECT l_orderkey, l_linenumber, l_tax, l_quantity FROM lineitem WHERE (l_extendedprice >= 5087.01) AND (l_quantity >= 37.0) ORDER BY l_tax DESC, l_orderkey ASC, l_linenumber ASC LIMIT 10`
- r26: `SELECT o_orderpriority, SUM(c_acctbal) AS a0, MAX(c_acctbal) AS a1 FROM orders JOIN customer ON o_custkey = c_custkey JOIN nation ON c_nationkey = n_nationkey WHERE (o_orderdate >= DATE '1993-06-16' AND o_orderdate <= DATE '1994-10-10') AND (c_mktsegment IN ('BUILDING', 'HOUSEHOLD')) AND (c_acctbal >= -784.52 AND c_acctbal <= 1429.5) GROUP BY o_orderpriority ORDER BY a0 DESC, o_orderpriority ASC`
- r27: `SELECT o_shippriority, COUNT(DISTINCT c_mktsegment) AS n FROM orders JOIN customer ON o_custkey = c_custkey WHERE TRUE GROUP BY o_shippriority ORDER BY n DESC, o_shippriority ASC LIMIT 20`
- r28: `SELECT l_shipinstruct, l_linestatus, SUM(l_extendedprice * (1 - l_discount)) AS a0, SUM(l_extendedprice * (1 - l_discount)) AS a1 FROM lineitem WHERE (l_receiptdate < DATE '1997-08-11') GROUP BY l_shipinstruct, l_linestatus ORDER BY a0 DESC, l_shipinstruct ASC, l_linestatus ASC LIMIT 10`
- r29: `SELECT DISTINCT l_shipinstruct FROM lineitem JOIN supplier ON l_suppkey = s_suppkey WHERE (l_discount >= 0.05) AND (l_shipmode = 'AIR')`
- r30: `SELECT DISTINCT o_orderpriority, o_orderstatus FROM orders WHERE (o_orderpriority = '4-NOT SPECIFIED')`
- r31: `SELECT p_mfgr, p_brand, SUM(p_size) AS a0, COUNT(p_size) AS a1 FROM part WHERE TRUE GROUP BY p_mfgr, p_brand HAVING COUNT(p_size) > 2`
- r32: `SELECT o_orderstatus, MIN(o_totalprice) AS a0 FROM orders WHERE (o_orderpriority <> '4-NOT SPECIFIED') GROUP BY o_orderstatus`
- r33: `SELECT n_name, MIN(c_acctbal) AS a0, AVG(o_totalprice) AS a1, SUM(o_totalprice) AS a2, MIN(c_acctbal) AS a3 FROM orders JOIN customer ON o_custkey = c_custkey JOIN nation ON c_nationkey = n_nationkey WHERE (c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'FURNITURE')) AND (c_acctbal >= 3166.7) GROUP BY n_name`
- r34: `SELECT DISTINCT ps_suppkey FROM partsupp WHERE (ps_availqty >= 4605)`
- r35: `SELECT DISTINCT ps_suppkey FROM partsupp WHERE (ps_availqty >= 3409) AND (ps_supplycost >= 61.36 AND ps_supplycost <= 260.84)`
- r36: `SELECT SUM(l_quantity) AS a0, SUM(l_extendedprice) AS a1, COUNT(l_tax) AS a2 FROM lineitem WHERE TRUE`
- r37: `SELECT l_linestatus, l_returnflag, MAX(l_quantity) AS a0 FROM lineitem WHERE (l_extendedprice < 65867.5) AND (l_returnflag IN ('A', 'N', 'R')) GROUP BY l_linestatus, l_returnflag`
- r38: `SELECT c_custkey, c_acctbal FROM customer WHERE TRUE ORDER BY c_acctbal ASC, c_custkey ASC LIMIT 50`
- r39: `SELECT o_shippriority, o_orderpriority, o_orderstatus, COUNT(o_totalprice) AS a0, MIN(o_totalprice) AS a1, SUM(c_acctbal) AS a2 FROM orders JOIN customer ON o_custkey = c_custkey JOIN nation ON c_nationkey = n_nationkey WHERE (o_orderstatus = 'P') GROUP BY o_shippriority, o_orderpriority, o_orderstatus`
- r40: `SELECT l_suppkey, COUNT(DISTINCT l_returnflag) AS n FROM lineitem WHERE (l_receiptdate >= DATE '1994-06-16') AND (l_discount >= 0.06 AND l_discount <= 0.07) GROUP BY l_suppkey ORDER BY n DESC, l_suppkey ASC LIMIT 100`
- r41: `SELECT ps_partkey, ps_suppkey, ps_supplycost, ps_availqty FROM partsupp WHERE (ps_supplycost >= 380.85 AND ps_supplycost <= 580.1) ORDER BY ps_supplycost ASC, ps_partkey ASC, ps_suppkey ASC LIMIT 200`
- r42: `SELECT l_suppkey, l_returnflag, l_shipmode, SUM(l_extendedprice * (1 - l_discount)) AS a0, SUM(l_quantity) AS a1, COUNT(l_tax) AS a2, SUM(l_tax) AS a3 FROM lineitem WHERE (l_receiptdate < DATE '1998-05-27') AND (l_discount < 0.09) GROUP BY l_suppkey, l_returnflag, l_shipmode ORDER BY a0 ASC, l_suppkey ASC, l_returnflag ASC, l_shipmode ASC LIMIT 100`
- r43: `SELECT o_custkey, o_shippriority, o_orderstatus, AVG(o_totalprice) AS a0, AVG(o_totalprice) AS a1, MAX(o_totalprice) AS a2, SUM(o_totalprice) AS a3 FROM orders WHERE (o_orderstatus IN ('F', 'O', 'P')) GROUP BY o_custkey, o_shippriority, o_orderstatus`
- r44: `SELECT l_orderkey, l_linenumber, l_discount, l_tax FROM lineitem WHERE (l_receiptdate < DATE '1993-02-19') AND (l_shipdate >= DATE '1992-04-18' AND l_shipdate <= DATE '1993-02-03') AND (l_quantity < 49.0) ORDER BY l_discount ASC, l_orderkey ASC, l_linenumber ASC LIMIT 200`
- r45: `SELECT c_mktsegment, SUM(c_acctbal) AS a0, MIN(c_acctbal) AS a1, MIN(c_acctbal) AS a2, AVG(c_acctbal) AS a3 FROM customer JOIN nation ON c_nationkey = n_nationkey WHERE TRUE GROUP BY c_mktsegment ORDER BY a0 ASC, c_mktsegment ASC LIMIT 100`
- r46: `SELECT c_mktsegment, c_nationkey, SUM(c_acctbal) AS a0 FROM customer WHERE TRUE GROUP BY c_mktsegment, c_nationkey`
- r47: `SELECT MIN(l_tax) AS a0 FROM lineitem WHERE (l_extendedprice >= 48251.5) AND (l_returnflag IN ('A', 'N', 'R')) AND (l_shipdate >= DATE '1992-04-18' AND l_shipdate <= DATE '1993-02-03')`
- r48: `SELECT DISTINCT ps_suppkey FROM partsupp WHERE (ps_availqty < 1802) AND (ps_supplycost >= 141.43 AND ps_supplycost <= 300.97) ORDER BY ps_suppkey ASC`
- r49: `SELECT ps_suppkey, COUNT(ps_availqty) AS a0, COUNT(ps_availqty) AS a1, AVG(ps_supplycost) AS a2, AVG(ps_supplycost) AS a3 FROM partsupp WHERE (ps_availqty >= 6603 AND ps_availqty <= 7798) AND (ps_supplycost >= 61.36 AND ps_supplycost <= 260.84) GROUP BY ps_suppkey ORDER BY a0 DESC, ps_suppkey ASC LIMIT 10`

## Tests on the same clean M4 clone (main 6d316be)

`pixi run mojo-test`: 49 ok, 0 FAIL. `pixi run test -n 4`: 20,398 passed, 9,838 skipped, 5 xfailed, 3 failed: `test_tpch_duckdb[sf1-4-gpu]` and `[sf1-13-gpu]` (the Python path's GPU loads the x86 kernel library, F-214) and `test_coproc.py::test_prepare_overlaps_with_compute` (a sleep-timing bound, fails alone at the parent too). Was 34 failed at F-209's start, 256 failed before F-209.
