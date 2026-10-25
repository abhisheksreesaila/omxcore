#!/usr/bin/env bash
# F-181: the Apple GPU check on a macos-26 runner (.github/workflows/apple-gpu.yml). Writes out/report.txt:
# the probe's PROBE lines, pyarrow's R lines and the engine's ENGINE / Q lines (scripts/check_apple_gpu.py reads them).
# The runner's GPU is virtual: MAX needs M1's Metal family named (--target-accelerator metal:1). Its timings mean nothing.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="$root/out"; mkdir -p "$out"
cd "$root/ci/mac"
accel=(--target-accelerator "${OMX_ACCEL:-metal:1}"); [ "${OMX_ACCEL:-}" = none ] && accel=()
{ sw_vers; sysctl -n machdep.cpu.brand_string; } > "$out/macos.txt" 2>&1
pixi run mojo --version > "$out/mojo-version.txt" 2>&1

echo "== probe"
pixi run mojo run --disable-warnings "${accel[@]}" probe_metal.mojo > "$out/probe.txt" 2>&1 || echo "PROBE EXIT: $?" >> "$out/probe.txt"
cat "$out/probe.txt"

echo "== package"   # scripts/build_mojo_pkg.sh's staging, without flock / GNU sed
stage="$root/build/stage/omxcore"; rm -rf "$root/build/stage"; mkdir -p "$stage/kernels"
cp "$root"/mojo/omxcore/*.mojo "$stage/"
cp "$root"/kernels_aot/*.mojo "$stage/kernels/"
echo '"""The native kernels (copied from kernels_aot/)."""' > "$stage/kernels/__init__.mojo"
for f in "$root"/kernels_aot/*.mojo; do
  m="$(basename "$f" .mojo)"
  perl -pi -e "s/^from $m import/from .$m import/" "$stage"/kernels/*.mojo
done
pixi run mojo precompile --disable-warnings "$stage" -o "$root/build/omxcore.mojoc" > "$out/package.txt" 2>&1 \
  && echo "PACKAGE: ok" >> "$out/package.txt" || echo "PACKAGE: failed" >> "$out/package.txt"
tail -3 "$out/package.txt"

echo "== data and pyarrow's answers"
pixi run python orders.py "$root/build/orders.csv" "${OMX_ROWS:-1000000}" > "$out/ref.txt" 2>&1
tail -3 "$out/ref.txt"

echo "== engine"
if pixi run mojo build --disable-warnings "${accel[@]}" -I "$root/build" engine_check.mojo -o "$root/build/engine_check" > "$out/engine-build.txt" 2>&1; then
  "$root/build/engine_check" "$root/build/orders.csv" > "$out/engine.txt" 2>&1 || echo "ENGINE EXIT: $?" >> "$out/engine.txt"
else
  echo "ENGINE BUILD: failed" > "$out/engine.txt"; grep -E "error" -A3 "$out/engine-build.txt" | head -60 >> "$out/engine.txt"
fi
cat "$out/engine.txt"
cat "$out/probe.txt" "$out/package.txt" "$out/ref.txt" "$out/engine.txt" > "$out/report.txt"
mkdir -p "$out/crashes" && cp ~/Library/Logs/DiagnosticReports/*engine_check* "$out/crashes/" 2>/dev/null
exit 0
