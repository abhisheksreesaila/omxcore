#!/usr/bin/env bash
# Build the committed kernels_aot libraries (F-008, F-066). Ahead-of-time, nothing compiles at run time.
#
# Usage:  bash scripts/build_pipeline.sh [target]      (default: all)
#   cpu     libmxpipeline_cpu.so and libmxexec_cpu.so (the executor imports pipeline_cpu.mojo)
#   exec    libmxexec_cpu.so
#   gpu     libmxpipeline_gpu.so
#   keys    libmxgroupkeys_cpu.so
#   join    libmxjoin_cpu.so
#   hashjoin libmxhashjoin_cpu.so (the hash join kernel's C ABI: hashjoin_lib.mojo over hashjoin_cpu.mojo)
#   aot     libmxkernels_aot.so and libmxkernels_aot_gpu.so (Mojo's default target, as in release.sh)
#   all     every library
#
# After each build it sets the relocatable RUNPATH (set_runpath.sh) and records the hashes of the
# Mojo sources the library was built from (record_build_hash.py); tests/test_kernel_hashes.py and CI
# fail when a source changed without its library being rebuilt. Re-record after merging branches
# that edited kernels_aot/*.mojo by running this with no argument, then commit the .so files and
# their .sources.json files.
#
# On macOS: no x86-64 level, no RUNPATH step (ELF only), and libmxkernels_aot_gpu is skipped, not built (Metal has no float64).
#
# All but `aot` are built for x86-64-v3 (AVX2 + FMA + BMI2, Haswell / Zen1 and newer), not the
# build host's native CPU, so a wheel does not SIGILL on an older machine. Measured on SF10 Q1
# (60M rows, 24 threads): 51 ms vs 56 ms native, i.e. no cost.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WHICH="${1:-all}"
BUILT=()

# build <source stem> <library stem> [mojo flags...]
build() {
    local src="$1" lib="$2"; shift 2
    pixi run mojo build "kernels_aot/$src.mojo" -I kernels_aot "$@" --emit shared-lib \
        -o "kernels_aot/$lib.so"
    if [[ "$(uname)" == "Linux" ]]; then bash scripts/set_runpath.sh "kernels_aot/$lib.so"; fi   # ELF RUNPATH: Linux only
    pixi run python scripts/record_build_hash.py "kernels_aot/$lib.so"
    BUILT+=("kernels_aot/$lib.so")
}
if [[ "$(uname -m)" == "x86_64" ]]; then V3=(--target-cpu x86-64-v3); else V3=(); fi   # an x86-64 level means nothing on arm64

case "$WHICH" in all|cpu|exec|gpu|keys|join|hashjoin|aot) ;; *) echo "unknown target: $WHICH" >&2; exit 2 ;; esac

if [[ "$WHICH" == "all" || "$WHICH" == "cpu" ]]; then
    build pipeline_cpu libmxpipeline_cpu "${V3[@]}"
fi
if [[ "$WHICH" == "all" || "$WHICH" == "cpu" || "$WHICH" == "exec" ]]; then
    # F-030 native executor: one call runs the whole plan (imports pipeline_cpu.mojo)
    build executor_cpu libmxexec_cpu "${V3[@]}"
fi
if [[ "$WHICH" == "all" || "$WHICH" == "gpu" ]]; then
    build pipeline_gpu libmxpipeline_gpu "${V3[@]}"
fi
if [[ "$WHICH" == "all" || "$WHICH" == "keys" ]]; then
    build groupkeys_cpu libmxgroupkeys_cpu "${V3[@]}"
fi
if [[ "$WHICH" == "all" || "$WHICH" == "join" ]]; then
    build joinsemi_cpu libmxjoin_cpu "${V3[@]}"
fi
if [[ "$WHICH" == "all" || "$WHICH" == "hashjoin" ]]; then
    build hashjoin_lib libmxhashjoin_cpu "${V3[@]}"
fi
if [[ "$WHICH" == "all" || "$WHICH" == "aot" ]]; then
    build kernels_aot libmxkernels_aot
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "skipped libmxkernels_aot_gpu: its kernels are float64 and Metal has no float64 (the Python path's GPU is CUDA only)"
    else
        build kernels_aot_gpu libmxkernels_aot_gpu
    fi
fi
echo "built: ${BUILT[*]}"
