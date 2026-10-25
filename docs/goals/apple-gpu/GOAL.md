# Goal: omxcore's queries on Apple's GPU

**End goal (owner, 2026-10-24):** "ensure that we step into the Apple GPU": `gpu_available()` is true on an Apple Silicon
Mac and the app's queries run on its GPU (Metal), with the same answers as pyarrow, while NVIDIA does not change.
Spec: abhisheksreesaila/omxcode `docs/plans/omxcore-apple-gpu.md`. Tickets F-181..F-186 in `docs/plan.md`.

## Milestones (each checked by `pixi run python scripts/check_apple_gpu.py`)
- **M1 Runner premises.** The macos-26 job's probe prints `PROBE DEVICE: <name> · metal` and `ok` for each premise
  the engine relies on: a kernel writes through an integer address, through an address it reads from a buffer, with
  dynamic shared memory (16 and 32 KB), 64-bit integer + - * // %, and 32-bit warp shuffles. (F-181)
- **M2 The kernel compiles for Metal.** On Linux, the package's pipeline kernels built for `metal:1` (`--emit asm`)
  hold no `double`, no `atomicrmw` and no inline assembly in their AIR; the NVIDIA PTX equals the baseline from
  3a33697 (`ci/mac/nvidia_ptx.sha`). (F-183, F-184, F-185)
- **M3 The engine sees the Apple GPU.** On the runner the engine (built from this commit) prints
  `ENGINE DEVICE: <name> · metal` from `gpu_device()` and `ENGINE AVAILABLE: True`. (F-182)
- **M4 Done means.** On the runner, the 1M-row orders queries (count, sum, min, max of INT and DECIMAL, overall and by
  a TEXT column; a WHERE; ORDER BY an aggregate with LIMIT) run with `device=GPU`, report that they ran on the GPU, and
  equal pyarrow's answers: at least 8 `Q` lines, every one `ran=gpu` and `omx == ref`. (F-184, F-186)
- **M5 NVIDIA unchanged.** Locally: `pixi run mojo-test` passes, and the GPU Mojo tests pass on the RTX 3090. (F-186)

## Out of bounds
No PyPI publish, no repo rename, no benchmarks beyond a light A/B on GPU 0 (quiet machine rule), no force-push, no
paid services. Push only to abhisheksreesaila/omxcore; issues on the app repo only for the STABLE hand-off.

## Assumptions (instead of asking)
- Apple GPU on by default on macOS (owner said so); `-D MXFRAME_NO_GPU` remains the opt-out.
- The interpreter's word is a compile-time parameter: Int64 on Apple, Float64 (unchanged) on NVIDIA. Float programs
  already stay on the CPU in the Mojo API (`has_float`), so the Apple kernel needs integer ops only.
- Scope is the Mojo package (`kernels_aot/pipeline_gpu.mojo`, `mojo/omxcore/gpu.mojo`, `query.mojo`); the Python
  wheel's AOT GPU kernels (`kernels_aot_gpu.mojo`, `_atomic.mojo`) are Linux only and stay as they are.
- The runner's virtual GPU answers correctness only; speed is measured on the owner's Mac later.

## Budget
One night (2026-10-24/25), about 150 turns. Report in `docs/goals/apple-gpu/report.md`.
