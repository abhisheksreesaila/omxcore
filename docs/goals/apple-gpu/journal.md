# Journal (Apple GPU)

## 2026-10-24 evening
- Read the spec; its file:line references hold at 3a33697.
- Linux `mojo build --target-accelerator metal:1` of omxcode's gpuprobe: parses, lowers to AIR, then fails at
  "Metal Compiler failed to compile metallib" (Apple's toolchain is Mac-only). `--emit asm` writes the kernel's AIR as a
  sidecar `.ll` (`target triple = "air64-apple-macosx15.0.0"`): a local check for double / atomics / asm.
- `std.sys.info.is_apple_gpu()` inside a kernel resolves per target: one source gave `store i64 7` in the Metal AIR
  and 8 on the RTX 3090.
- Consequence for the app: a Mac build whose GPU path is on needs Apple's Metal toolchain at build time, so it must be
  built on a Mac (or a macOS runner); a Linux cross-build of the Mac app must pass `-D MXFRAME_NO_GPU`.
- The pipeline kernel: modes 1 and 2 use 64-bit atomics; 0, 3 and 4 write per-block rows merged in block order.
- Open premise (the riskiest): the engine passes device addresses as integers and reads column addresses out of a
  buffer. On Metal that needs every buffer resident and its GPU address valid in the kernel. First probe on the runner.

## Run 1 on macos-26 (run 40171201386, c1fe1ea)
- `PROBE DEVICE: Apple Paravirtual device · metal` (Apple M1 (Virtual), macOS 26.6.2). Every premise **wrong**: an
  integer-address write left 1023 of 1024 slots untouched; shared memory, int64 maths and shuffles wrong too. All
  of those passed buffers as Int64 addresses (`unsafe_ptr()` cast to an integer); omxcode's probe, which passes
  typed pointers, gave `RESULT: ok` on the same runner. Hypothesis: on Metal `DeviceBuffer.unsafe_ptr()` on the host
  is not the address the kernel sees (MAX translates typed pointer arguments when it binds them), and/or a buffer used
  only through an address is not resident. Probe v2 separates the two.

## Run 2 (40171534745, ae3afeb)
- typed-pointer write ok; the host's `unsafe_ptr()` is the address the kernel sees (`same`, unified memory);
  shared memory 16 / 32 KB, int64 maths and 32-lane shuffles ok.
- Every pointer made from an integer wrong, even of a buffer bound as an argument (`typed-roundtrip`), so not
  residency: in AIR `Pointer(unsafe_from_address=)` is a generic `ptr`, which Metal reads as thread-private memory;
  device memory is `ptr addrspace(1)`. Fix: kernels make such pointers with `address_space=KAS`, GLOBAL when
  `is_apple_gpu()` (a module-level alias resolves per kernel target: AIR `ptr addrspace(1)`, NVIDIA PTX unchanged).
- `ENGINE AVAILABLE: False`: the runner's virtual M1 is not `is_apple_silicon()`; HAS_GPU now keys on
  `is_macos() and has_apple_gpu_accelerator()` (D-07 revised).
- Kernel word: `pipeline_kernel[RPT, B, W]`, `_run_int` beside the untouched float64 `_run`; atomic modes compiled for
  float64 only; the NVIDIA merge kernel kept verbatim as its own function (a parameterised merge laid its loop out
  differently in PTX). NVIDIA PTX fingerprint = baseline 3bbe74fdd7263978 (6 kernels). Metal AIR: no double, no
  atomicrmw, no asm. Remaining generic `load ptr` in AIR are stack slots (the shuffle's), which is fine.
- NVIDIA: mojo-test gpu / gpu_scan / query / holistic_query / kernels_from_package ok; pytest -k "gpu or pipeline"
  9879 passed.

## Run 4 (40173257990, fd3ca30): every milestone passes
- `scripts/check_apple_gpu.py`: M1 premises, M2 Metal AIR clean + NVIDIA PTX = baseline, M3 `ENGINE DEVICE: Apple
  Paravirtual device · metal`, M4 10/10 orders queries on the Apple GPU equal pyarrow, M5 mojo-test 26 ok.
- Known limit: more than 4096 (slots + 1) x groups decline on an Apple GPU (-6: AUTO answers on the CPU). Next: mode 5.

## Review round 1 (NEEDS WORK) and mode 5 (2026-10-25, worktree f-184-mode5)
- Mode 5: many groups on the Int64 word, mode 4's ordered walk with the block's table in its device-memory scratch
  row, group g owned by thread g % B. Narrow blocks are 64 threads on Apple (NARROW_B), so a 44-aggregate pivot runs.
- Keeping NVIDIA's PTX byte-identical needed *additive* comptime blocks: wrapping an existing `if mode == 4:` in
  `comptime if W == F64:` (or `x and helper()` that folds to True) swapped two predicate registers in two kernels.
- `test_gpu` under `-D OMXCORE_APPLE_GPU_SHAPE`: ok (two cases Apple-aware: 44 aggregates now fit and equal the CPU;
  3M groups decline to the CPU on AUTO with a reason).
- Light NVIDIA A/B on GPU 0 (1M orders, 15 runs, median ms, base 3a33697 vs new): 0.171/0.170, 0.248/0.248,
  0.165/0.164, 0.112/0.112; a second round equal within noise (one 0.27 vs 0.25 with the Apple-shape suite loading the box).
