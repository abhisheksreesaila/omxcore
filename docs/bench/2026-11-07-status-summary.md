# omxcore vs DuckDB: where we stand (2026-11-07, paused here)

Every answer below equals DuckDB. Ratio = geometric mean of DuckDB time / omxcore time (above 1x = omxcore faster). Engine-only, fresh process per run, median of 3.

## Linux box (24-core CPU, 2x NVIDIA RTX 3090), DuckDB 1.5.1, CPU device

Latest: `docs/bench/2026-11-06-night-partial.md` (machine loaded 11-13, run cut short; before F-206/F-207/F-210's CPU gains).

| set | omxcore faster | ratio | 2026-11-05 baseline |
|---|---:|---:|---:|
| random SF1 | 84 / 100 | 1.58x | not measured |
| random SF10 | 38 / 59 (partial) | 1.26x | not measured |
| TPC-H SF1 | 7 / 22 | 0.76x | 0.41x (4 / 22) |
| TPC-H SF10 | 3 / 22 | 0.51x | 0.23x (0 / 22) |

Join work (F-201..F-205, SF10, under load): Q18 8308 -> ~1092 ms, Q22 1464 -> 114, Q7 1529 -> 244, Q10 923 -> 201 (beats DuckDB).

NVIDIA 3090 GPU device: **not measured on the current engine**. Both GPUs were held by another process (llama-server, ~21 GB each) all week. Last GPU numbers are from F-058 (2026-10-17, older engine): AUTO beat DuckDB on 10 / 22 TPC-H at SF10; SF10 total 2.75 s vs DuckDB 4.83 s (`docs/bench/f058-report.md`). First thing to run when the GPUs are free: `pixi run python scripts/gate_scoreboard.py --sf 1,10 --devices cpu,gpu,auto --repeats 3 --random 100`.

## M4 Mac (10 cores, 16 GB), DuckDB 1.5.6

Clean run at main 6d316be, `docs/bench/2026-11-06-m4-mac-final-gate.md` (load ~2.6).

| set | start of 2026-11-06 (cpu) | cpu now | auto now | gpu now |
|---|---:|---:|---:|---:|
| random SF1 (50) | 11 / 50, 0.59x | **45 / 50, 1.89x** | 44 / 50, 1.88x | 23 / 50, 0.49x |
| TPC-H SF1 | 0 / 22, 0.22x | 4 / 22, 0.67x | 5 / 22, 0.68x | 1 / 22, 0.23x |

Why it moved: the Mac build ran on one core (F-210: all cores now), arm64 denormal flag, arm64 kernels vectorized, dead SQL instructions pruned (that also sped Linux: Q6 14.1 -> 7.5 ms). Mac tests: mojo 49 ok; pytest 3 failed (2 fixed by F-214, 1 timing test that fails on the parent too).

Apple GPU: slower than the M4's CPU. Not the copy: with data already on the GPU, a 6M-row scan is 9.6-12 ms GPU vs 2.0 ms CPU (SF10 Q6 721 vs 85 ms). The GPU kernel is an interpreter with registers in threadgroup memory: F-215 (a new Apple kernel) is the fix. AUTO on a Mac picks the CPU.

## Footprint

mx 5.6 MB + kernels 3.5 MB + runtime 3.3 MB, about 13 MB, vs DuckDB's 60 MB library.

## Paused (owner 2026-11-07: good stopping point)

- F-213 (GPU uploads only the columns a query reads; Apple AUTO cost rule): branch `f-213-apple-gpu-in-place` at 8d3ee4a, review fixes committed, needs re-review, suites, merge.
- F-161 builder `Query.join` fused: branch `worktree-agent-adf35e1a8b23d3267` at 67da0f7, WIP, not built.
- F-161 hash-probe memory latency (radix / prefetch): not started (no commits).
- F-215 fast Apple GPU kernel: todo. F-208 median(DECIMAL) scale: todo.
- Nightly 11pm task paused.
