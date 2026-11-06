#!/usr/bin/env bash
# F-151: builds the `mx` command line (mojo/cli/mx.mojo) into build/mx, against the Mojo package, and checks that it never needs Python.
# Usage: pixi run bash scripts/build_mx.sh [output path]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
export TMPDIR="${MXFRAME_TEST_TMPDIR:-$HOME/.cache/mxframe-test-tmp}"
mkdir -p "$TMPDIR" "$root/build"
out="${1:-$root/build/mx}"
bash "$root/scripts/build_mojo_pkg.sh" > "$root/build/pkg-build.log" 2>&1 \
  || { grep -E "error:" -A4 "$root/build/pkg-build.log" | head -40; echo "FAIL (package build)"; exit 1; }
lib="$root/.pixi/envs/default/lib"
mojo build --disable-warnings "$root/mojo/cli/mx.mojo" -I "$root/build" -Xlinker -L"$lib" -Xlinker -lzstd -Xlinker -lsnappy \
  -Xlinker -rpath -Xlinker "$lib" -o "$out"
# The same guard as scripts/mojo_test.sh: no libpython, no Mojo Python interop symbols.
# ldd is GNU only; macOS lists a binary's libraries with otool -L.
libs_of() { if command -v ldd > /dev/null; then ldd "$1"; else otool -L "$1"; fi; }
if libs_of "$out" | grep -i python > /dev/null \
   || nm -D "$out" 2>/dev/null | grep KGEN_CompilerRT_Python > /dev/null \
   || strings "$out" | grep -E "^Py_Initialize" > /dev/null; then
  echo "FAIL: mx uses Python (libpython linked or Mojo Python interop)"; rm -f "$out"; exit 1
fi
echo "built $out"
