#!/usr/bin/env bash
# Build one Mojo program against the package (build/omxcore.mojoc), with the codec libraries linked.
# Usage: bash scripts/mojo_build.sh <source.mojo> <output binary> [--no-package]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
export TMPDIR="${MXFRAME_TEST_TMPDIR:-$HOME/.cache/mxframe-test-tmp}"   # compiler temporaries off the shared /tmp tmpfs
mkdir -p "$TMPDIR" "$root/build"
[ "${3:-}" = "--no-package" ] || bash "$root/scripts/build_mojo_pkg.sh" > "$root/build/pkg-build.log" 2>&1 \
  || { grep -E "error:" -A4 "$root/build/pkg-build.log" | head -40; exit 1; }
lib="$root/.pixi/envs/default/lib"
mojo build --disable-warnings ${MXFRAME_BUILD_FLAGS:-} "$1" -I "$root/build" -Xlinker -L"$lib" -Xlinker -lzstd -Xlinker -lsnappy \
  -Xlinker -rpath -Xlinker "$lib" -o "$2"
