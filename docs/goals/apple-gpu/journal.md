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
