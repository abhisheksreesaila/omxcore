#!/usr/bin/env bash
# F-138b: does the Mojo package link into the omxplorer Mac build, which has no GPU, no Python, no Mojo async
# runtime and no zstd / snappy?
#
# It compiles mojo/tools/mac_link_probe.mojo (a small program over the whole package: CSV, queries on CPU / AUTO /
# GPU, a frame's edits, Parquet save and open) as an object for arm64-apple-macosx14.0, builds the app's runtime
# stand-ins (mojoui/hosts/mojo_rt.mojo), and
#   1. lists the object's undefined symbols (llvm-nm -u) and fails on any that is not an Apple system function or one
#      of the stand-ins: no AsyncRT, no Tracy, no zstd / snappy, nothing from Python or CUDA;
#   2. links them as the app does (ld64.lld, -e _mojo_entry, -dead_strip, the app's apple/sdk/macos/*.tbd), which
#      fails on any symbol the tbd files do not export. The app's libSystem stub lists its functions by hand, so the
#      link uses a copy with the package's file-system calls added;
#   3. prints those calls and the symbols outside the app's `ALLOWED` set (docs/goals/05-check.py): what F-138c must
#      add to the app's stub and its check.
#
# Usage:  pixi run bash scripts/check_mac_link.sh        (MXAPP_UI=<path> if the app repo is not ~/Projects/mxapp/mojo-ui)
# The app repo is only read.
set -euo pipefail
export LC_ALL=C                                   # comm needs the same sort order as sort
root="$(cd "$(dirname "$0")/.." && pwd)"
app="${MXAPP_UI:-$HOME/Projects/mxapp/mojo-ui}"
out="$root/build/mac-link"
triple=arm64-apple-macosx14.0
[ -f "$app/scripts/build.sh" ] || { echo "FAIL: the app repo is not at $app (set MXAPP_UI)"; exit 2; }

if command -v ld64.lld > /dev/null; then lld=(ld64.lld)
elif [ -x "$app/.pixi/envs/default/bin/ld64.lld" ]; then lld=("$app/.pixi/envs/default/bin/ld64.lld")
else lld=(lld -flavor darwin); fi

mkdir -p "$out"; rm -f "$out"/*.o "$out/probe"
bash "$root/scripts/build_mojo_pkg.sh" > "$out/pkg-build.log" 2>&1 || { tail -20 "$out/pkg-build.log"; echo "FAIL (package build)"; exit 1; }

# The switches no GPU and no codecs are on by default for every target but Linux. Serial kernels are not (F-210): a Mac build
# runs queries on every core with Mojo's async runtime, which the app's link does not have, so the app builds with
# -D MXFRAME_SERIAL and this check builds the probe the same way. The default build is checked at the end.
mojo build --disable-warnings "$root/mojo/tools/mac_link_probe.mojo" -I "$root/build" --target-triple $triple --emit object -D MXFRAME_SERIAL \
  -o "$out/probe.o" > "$out/probe-build.log" 2>&1 || { tail -30 "$out/probe-build.log"; echo "FAIL (probe build)"; exit 1; }
mojo build --disable-warnings "$app/mojoui/hosts/mojo_rt.mojo" --target-triple $triple --emit object \
  -o "$out/mojo_rt.o" > "$out/rt-build.log" 2>&1 || { tail -30 "$out/rt-build.log"; echo "FAIL (mojo_rt build)"; exit 1; }

defined="$(llvm-nm -j --defined-only "$out/probe.o" "$out/mojo_rt.o" | grep -v ':$' | sort -u)"
undefined="$(llvm-nm -u -j "$out/probe.o" | grep -v ':$' | sort -u)"
missing="$(comm -23 <(echo "$undefined") <(echo "$defined"))"      # not satisfied by the stand-ins: must be Apple's

echo "undefined in the package object, not defined by the stand-ins:"; echo "$missing" | sed 's/^/  /'
bad="$(echo "$missing" | grep -E 'AsyncRT|Tracy|KGEN|ZSTD|snappy|^_Py|python|cuda|DeviceContext|DeviceBuffer' || true)"
if [ -n "$bad" ]; then echo "FAIL: symbols this build must not need:"; echo "$bad" | sed 's/^/  /'; exit 1; fi

# The app's apple/sdk/macos/libSystem.tbd lists, by hand, the libSystem functions the app uses. The package adds the
# file-system, process and memory-size (sysconf) calls below (all POSIX / libc, in libSystem on macOS).
libc_ok=" _close _lseek _lstat _mkdir _mmap _munmap _open _read _rename _stat _strerror _unlink _getpid _sysconf ___error _memset_pattern16 _strtod _strtod_l _newlocale _freelocale "
have="$(grep -o '_[A-Za-z0-9_]*' "$app/apple/sdk/macos/libSystem.tbd" | sort -u)"
need="$(comm -23 <(echo "$missing") <(echo "$have") | tr '\n' ' ')"
for s in $need; do
  case "$libc_ok" in *" $s "*) ;; *) echo "FAIL: $s is not a libSystem function the package is known to need"; exit 1 ;; esac
done
sdk="$out/sdk"; rm -rf "$sdk"; mkdir -p "$sdk"; cp "$app"/apple/sdk/macos/*.tbd "$sdk/"
extra="$(echo $need | sed 's/ /, /g')"
sed -i "s/symbols:         \[ /symbols:         [ $extra, /" "$sdk/libSystem.tbd"
"${lld[@]}" -arch arm64 -e _mojo_entry -dead_strip -no_adhoc_codesign -platform_version macos 14.0 26.0 \
  -o "$out/probe" "$out/probe.o" "$out/mojo_rt.o" "$sdk"/*.tbd \
  || { echo "FAIL: the link failed"; exit 1; }

allowed="$(python3 -c '
import re, sys
src = open(sys.argv[1]).read()
block = re.search(r"ALLOWED = \{(.*?)\n\}", src, re.S).group(1)
print("\n".join(sorted(re.findall(r"\"(_\w+)\"", block))))
' "$app/docs/goals/05-check.py")"
echo; echo "linked $out/probe ($(stat -c %s "$out/probe") bytes); libraries it needs:"
llvm-otool -L "$out/probe" | tail -n +2 | sed 's/ (.*//; s/^\s*/  /'
echo; echo "to add to the app's apple/sdk/macos/libSystem.tbd (F-138c):"; echo "  $need"
echo "to add to ALLOWED in the app's check (the package needs these beyond the current set):"
echo "  $(comm -23 <(echo "$missing") <(echo "$allowed") | tr '\n' ' ')"
# F-175: jobs with threads on this build (-D MXFRAME_JOB_THREADS): the same object plus pthread calls and nothing else. The app adds the
# listed functions to its stub (and ALLOWED) only if it turns the switch on; the default build above runs a job inside its start call.
mojo build --disable-warnings "$root/mojo/tools/mac_link_probe.mojo" -I "$root/build" --target-triple $triple --emit object -D MXFRAME_SERIAL -D MXFRAME_JOB_THREADS \
  -o "$out/probe-threads.o" > "$out/probe-threads-build.log" 2>&1 || { tail -30 "$out/probe-threads-build.log"; echo "FAIL (probe build with job threads)"; exit 1; }
added="$(comm -13 <(echo "$undefined") <(llvm-nm -u -j "$out/probe-threads.o" | grep -v ':$' | sort -u) | tr '\n' ' ')"
for s in $added; do
  case " _pthread_create _pthread_join _pthread_detach _pthread_attr_init _pthread_attr_setstacksize _pthread_attr_destroy _sched_yield " in *" $s "*) ;; *) echo "FAIL: job threads add $s"; exit 1 ;; esac
done
echo; echo "with job threads (-D MXFRAME_JOB_THREADS) the app's libSystem stub and ALLOWED also need:"; echo "  $added"
# F-210: the default build (no -D MXFRAME_SERIAL) runs every kernel on all the cores through Mojo's async runtime: the same object plus
# the runtime's own entry points and nothing else. An app that links that runtime (it ships Mojo's libraries) can use it; the app's link
# stub has none, so it builds with -D MXFRAME_SERIAL.
mojo build --disable-warnings "$root/mojo/tools/mac_link_probe.mojo" -I "$root/build" --target-triple $triple --emit object \
  -o "$out/probe-pool.o" > "$out/probe-pool-build.log" 2>&1 || { tail -30 "$out/probe-pool-build.log"; echo "FAIL (probe build without -D MXFRAME_SERIAL)"; exit 1; }
pool="$(comm -13 <(echo "$undefined") <(llvm-nm -u -j "$out/probe-pool.o" | grep -v ':$' | sort -u) | tr '\n' ' ')"
for s in $pool; do
  case "$s" in *AsyncRT*|*KGEN_CompilerRT*|_pthread_*|_sched_yield) ;; *) echo "FAIL: the thread pool adds $s"; exit 1 ;; esac   # the pool, and the job threads that come with it
done
echo; echo "without -D MXFRAME_SERIAL (the default: every core) the object also needs Mojo's async runtime, which the app does not link:"; echo "  $pool"
echo; echo "OK: the package links for $triple with -D MXFRAME_SERIAL and no GPU, Python, async runtime or codec symbol"
