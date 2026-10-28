# omxcore (Mojo): what happens when data is bigger than the GPU (F-189)

F-056 solved this in the Python front end (`gpu_budget.py`, `docs/research/gpu-out-of-core.md`). Apps call the Mojo
engine, which had none of it. This note records what the engine did before F-189 and what the MAX runtime does on
a real out-of-memory, measured on this box (2x RTX 3090, 24 GB each, MAX/Mojo 1.1.0).

## 1. What the MAX runtime does on a failed allocation (probe, GPU 0, idle)

`DeviceContext.enqueue_create_buffer` goes through MAX's device-context memory manager: one pool, sized when the
context is created (default 90% of the driver's free memory, 22.8 GB of 25.2 GB here;
`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT` changes it). Measured:

| probe | result |
|---|---|
| `ctx.get_memory_info()` | `(free, total)` are **the pool's** (22817013760 both, at the start), not the driver's: it is 90% of what the driver had free when the context was made. With the percent set to 25 it reports 6442450944. After 22 x 1 GiB buffers `free` was 0 (pool exhausted). |
| one buffer bigger than the pool but smaller than the card (23.9 GB) | **succeeds** (the manager asks the driver for the part the pool lacks). So the pool is a budget, not a limit. |
| 1 GiB buffers in a loop | the 23rd (past the card) raises an `Error`: `CUDA call failed: CUDA_ERROR_OUT_OF_MEMORY (out of memory)`. The error is catchable. Nothing else is damaged: a small allocation made afterwards works, and dropping the buffers returns `free` to its start value. |
| stderr | the runtime prints a multi-line `oom log` (pool and free-list state) for every failure. Harmless, noisy: a fallback that is expected should not be tried often. |
| a kernel that cannot get memory | (F-053 lesson) a kernel-side buffer taken from an empty pool printed `oom log` and returned zeros without an exception. omxcore allocates every device buffer itself, with `enqueue_create_buffer`, before it launches, so a failure surfaces as the catchable `Error` above, never as a silent wrong answer. |

So: a real out-of-memory is a **raised `Error` at the allocation**, recoverable; `get_memory_info()[0]` is a usable
"how much may I take" figure (it already leaves the 10% headroom and, being the pool, drops as omxcore's own
buffers fill it).

## 2. What the engine did before F-189

Measured with the `test_today_*` cases of `mojo/tests/test_gpu_oom.mojo` as committed in `de9648e` (replaced by the
piece tests in the next commit; a fake 1 MB GPU via the test knob that this ticket adds, tables of 7 to 26 MB):

- `DeviceTable(table)` uploaded every column whole, with no budget: `bytes()` was the whole table. On a real card
  smaller than the table the constructor raised the `Error` above part-way through, leaving the first columns
  uploaded until the exception unwound.
- `Query.compute(GPU)` on that table ran the whole thing in one launch (it can only run on an attached
  `DeviceTable`, so it never met a table bigger than the card: it could not even be set up). With the table
  resident, a query's own buffers (`_QueryBuffers`: code, constants, group table, per-block scratch, mask) were the
  only later allocation: AUTO caught their failure and answered on the CPU (`fallback_reason` "could not allocate");
  **`compute(GPU)` re-raised it**.
- `compute(AUTO)` chose the GPU on table size and a live `DeviceTable` alone (>= 1,000,000 rows): it never asked
  whether the working set fits.
- A GPU scan over a Parquet or CSV file (F-146) already went a row group at a time, one set of reused device
  buffers, so a **file** bigger than the card ran. Nothing looked at the size of one group, though: a group bigger
  than the card failed at `_stage_load`; on `GPU` that raised, on `AUTO` that group ran on the CPU ("could not hold
  a row group"). An allocation failure was not remembered: every later group, and every later query, tried again.
- No test could make a small GPU: filling the real device is opt-in (`MXFRAME_TEST_FILL_GPU=1`), because the
  allocator pools and the piece that crosses a margin takes everything left.

## 3. What F-189 does about it

See `docs/omxcore-mojo-api.md`, the GPU section. In short: a budget per device (`get_memory_info` free, capped by
`MXFRAME_GPU_MEM_BUDGET`, 90% usable); `DeviceTable` of a table that does not fit holds no device copy and
`compute(GPU)` uploads row ranges that fit in turn and merges the partials like the CPU's windows do; AUTO
answers a too-big table on the CPU and says how big it is; an allocation failure reruns the piece on the CPU and AUTO
remembers it until 1.5x more is free (`MXFRAME_GPU_FAIL_PIECE` injects the failure in tests).
