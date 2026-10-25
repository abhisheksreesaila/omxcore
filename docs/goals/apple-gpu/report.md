# Report: omxcore on Apple's GPU (autopilot, night of 2026-10-24)

**End goal:** `gpu_available()` true on Apple Silicon and the app's queries on its GPU (Metal), same answers as
pyarrow, NVIDIA unchanged. **Where it got:** every milestone passes (`pixi run python scripts/check_apple_gpu.py`):

```
(runner report: run 40177467956 at 5d4a5f3)
PASS M1 runner premises: Apple Paravirtual device · metal
PASS M2 kernel compiles for Metal: 7 Metal kernels clean, NVIDIA PTX = baseline
PASS M3 engine sees the Apple GPU: Apple Paravirtual device · metal
PASS M4 done means: 12/12 equal pyarrow on the GPU
PASS M5 NVIDIA unchanged (mojo-test): 27 ok
PASS M6 Apple's plan on this GPU: 7 suites ok; orders: 12/12 equal pyarrow under Apple's plan
```

What runs on an Apple GPU: integer programs (INT, DECIMAL, TEXT keys, DATE; count, sum, min, max, filters, NULLs),
up to about a million (slots + 1) x groups, pivots up to ~44 aggregates. Float programs and bigger group counts are
answered by the CPU (AUTO says why). NVIDIA: PTX byte-identical to 3a33697, timings equal within noise.

## 30-second demo
```sh
pixi run python scripts/check_apple_gpu.py --quick          # M1-M4 from the newest macos-26 run, M2 locally
gh run download <run id> -n apple-gpu-report -D out && cat out/engine.txt   # the engine's lines on the Mac's GPU
# on the owner's Mac (native build: the GPU is on by default)
xcodebuild -downloadComponent MetalToolchain
cd ci/mac && pixi run bash run.sh     # OMX_ACCEL=none on a real Mac; prints ENGINE DEVICE: Apple M… · metal
```

## Decisions worth a second look
- **D-02:** Int64 word on Apple only (the spec said everywhere): NVIDIA keeps Float64, so its PTX is pinned and
  identical; Apple runs integer programs, float programs go to the CPU (float32 on Apple is a later ticket).
- **D-07:** the GPU turns on for a macOS build whose accelerator is an Apple GPU. A Linux cross-build of the Mac app
  keeps no GPU (Metal kernels need Apple's toolchain at build time), so **omxcode must build its Mac binary on a Mac**
  (or a macos runner) to get the GPU.
- **D-09:** pointers from integer addresses are in the device address space on Apple (Metal has no generic one); found
  by the runner probe, the riskiest premise of the night.

## What's next
Real-silicon check on the owner's Mac (speed and correctness on an M-series), then float32 on Apple.

## Incident
A worktree commit (c40e15b) carried a `.pixi` symlink that replaced the main checkout's pixi environment on merge;
fixed in 956dd16, environment reinstalled from `pixi.lock`, lesson recorded.
