#!/usr/bin/env bash
# Run pytest on this checkout from any directory name (git worktrees included).
# The repo root is the `omxcore` package, and pytest names it after its directory,
# so tests run through a symlink called `omxcore` (and `mxframe`, the forwarding module in compat/).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
# Test temp files (generated TPC-H data, pytest tmp_path) go to disk, not the RAM /tmp, which
# several agents' suites filled until tests failed with "Disk quota exceeded".
export TMPDIR="${MXFRAME_TEST_TMPDIR:-$HOME/.cache/mxframe-test-tmp}"
mkdir -p "$TMPDIR"
shim="${TMPDIR:-/tmp}/omxcore-pytest-$(printf '%s' "$root" | { sha1sum 2> /dev/null || shasum; } | cut -c1-10)"
mkdir -p "$shim" && ln -sfn "$root" "$shim/omxcore" && ln -sfn "$root/compat/mxframe" "$shim/mxframe"
# The committed kernels_aot/*.so are x86-64 ELF files: a Mac cannot load them. Build this machine's libraries into
# build/kernels-<arch> (untracked; rebuilt when a kernels_aot source changed, and it skips what Metal cannot build, see
# build_pipeline.sh) and point the loaders at them with the MXFRAME_*_LIB variables: the committed files stay as they are.
if [ "$(uname)" = "Darwin" ]; then
  kout="$root/build/kernels-$(uname -m)"
  want="$(cat "$root"/kernels_aot/*.mojo | shasum | cut -c1-40)"
  if [ "$(cat "$kout/.sources" 2> /dev/null)" != "$want" ]; then
    echo "run_tests.sh: building the kernel libraries for this Mac into $kout (scripts/build_pipeline.sh)"
    MXFRAME_KERNEL_OUT="$kout" bash "$root/scripts/build_pipeline.sh"
    echo "$want" > "$kout/.sources"
  fi
  export MXFRAME_PIPELINE_CPU_LIB="$kout/libmxpipeline_cpu.so" MXFRAME_PIPELINE_GPU_LIB="$kout/libmxpipeline_gpu.so" \
         MXFRAME_EXEC_CPU_LIB="$kout/libmxexec_cpu.so" MXFRAME_GROUPKEYS_LIB="$kout/libmxgroupkeys_cpu.so" \
         MXFRAME_JOIN_LIB="$kout/libmxjoin_cpu.so" MXFRAME_HASHJOIN_LIB="$kout/libmxhashjoin_cpu.so" \
         MXFRAME_AOT_LIB="$kout/libmxkernels_aot.so"
fi
# Test paths (`tests/test_x.py`, `tests/test_x.py::name`) run alone, through the shim;
# with none, the whole tests directory runs. Other arguments pass through.
args=() paths=()
for a in "$@"; do
  case "$a" in
    -*) args+=("$a") ;;
    tests/*|./tests/*|"$root"/tests/*) paths+=("$shim/omxcore/${a#"$root"/}") ;;
    *) args+=("$a") ;;
  esac
done
[ ${#paths[@]} -eq 0 ] && paths=("$shim/omxcore/tests")
# a test stuck for 15 minutes dumps every thread's stack (a full-suite run hung once under load, F-070)
cd "$shim/omxcore" && exec pytest -o faulthandler_timeout=900 "${paths[@]}" ${args[@]+"${args[@]}"}
