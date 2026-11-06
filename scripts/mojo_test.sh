#!/usr/bin/env bash
# Build the package, then build and run every mojo/tests/test_*.mojo against it (each has a main() that raises on
# failure). Usage: pixi run mojo-test [test files...]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
# Mojo's compiler writes its temporaries to TMPDIR; /tmp is a shared tmpfs that other projects fill (builds failed
# with "Disk quota exceeded"), so use a disk-backed directory, as scripts/run_tests.sh does.
export TMPDIR="${MXFRAME_TEST_TMPDIR:-$HOME/.cache/mxframe-test-tmp}"
mkdir -p "$TMPDIR" "$root/build"
python3 "$root/scripts/check_list_lifetimes.py" || { echo "FAIL (a list whose address a kernel reads is freed too early: see scripts/check_list_lifetimes.py)"; exit 1; }
# Each run gets a private working directory: the tests write their scratch files by relative path, so two runs at
# once (agents in several worktrees) never read each other's half-written files.
rundir="$(mktemp -d "$TMPDIR/mojo-test-run.XXXXXX")"
trap 'rm -rf "$rundir"' EXIT
bash "$root/scripts/build_mojo_pkg.sh" > "$root/build/pkg-build.log" 2>&1 || { grep -E "error:" -A4 "$root/build/pkg-build.log" | head -40; echo "FAIL (package build)"; exit 1; }
lib="$root/.pixi/envs/default/lib"
out="$rundir/bin"   # per run: two runs in one checkout never rebuild a binary the other is running (failures print their log)
mkdir -p "$out"
# ldd is GNU only; macOS lists a binary's libraries with otool -L.
libs_of() { if command -v ldd > /dev/null; then ldd "$1"; else otool -L "$1"; fi; }
files=("$@")
[ ${#files[@]} -eq 0 ] && files=("$root"/mojo/tests/test_*.mojo)
fail=0
# A test that hangs (a spin on a gate nobody sets, a join that never ends) must fail with a message, not block the suite:
# every run is capped at MOJO_TEST_TIMEOUT seconds (default 900, as run_tests.sh's faulthandler). macOS has no `timeout`:
# perl's alarm survives exec and kills the program the same way.
limited() { local secs="$1"; shift
  if command -v timeout > /dev/null; then timeout "$secs" "$@"
  elif command -v gtimeout > /dev/null; then gtimeout "$secs" "$@"
  else perl -e 'alarm shift; exec @ARGV or die "exec: $!"' "$secs" "$@"; fi; }
# check_test <source> <name> [extra build flags...]: build one test against the package and run it in the private run directory.
check_test() {
  local t="$1" name="$2"; shift 2
  # MOJO_TEST_FLAGS: extra build flags, e.g. "-D OMXCORE_APPLE_GPU_SHAPE" (the Apple GPU's plan on this machine's GPU)
  # shellcheck disable=SC2086
  if ! mojo build --disable-warnings ${MOJO_TEST_FLAGS:-} "$@" "$t" -I "$root/build" -Xlinker -L"$lib" -Xlinker -lzstd -Xlinker -lsnappy -Xlinker -rpath -Xlinker "$lib" \
       -o "$out/$name" > "$out/$name.build.log" 2>&1; then
    echo "FAIL (build) $name"; tail -20 "$out/$name.build.log"; fail=1; return
  fi
  # The package must never need Python (brief: no libpython). ldd alone is not enough: Mojo's Python interop
  # dlopens libpython at run time, so look for the interop's runtime symbols in the binary too.
  if libs_of "$out/$name" | grep -i python > /dev/null \
     || nm -D "$out/$name" 2>/dev/null | grep KGEN_CompilerRT_Python > /dev/null \
     || strings "$out/$name" | grep -E "^Py_Initialize" > /dev/null; then
    echo "FAIL $name uses Python (libpython linked or Mojo Python interop)"; fail=1; return
  fi
  local rc=0
  mkdir -p "$rundir/$name"   # its own scratch directory: the threaded run of a test must not find the first run's files
  (cd "$rundir/$name" && limited "${MOJO_TEST_TIMEOUT:-900}" "$out/$name") > "$out/$name.log" 2>&1 || rc=$?
  if [ $rc -eq 0 ]; then echo "ok   $name"
  elif [ $rc -eq 124 ] || [ $rc -eq 142 ]; then   # timeout's 124, or SIGALRM (128 + 14) from the perl fallback
    echo "FAIL $name timed out after ${MOJO_TEST_TIMEOUT:-900}s (hung: a wait that never ends; MOJO_TEST_TIMEOUT changes the limit)"; tail -20 "$out/$name.log"; fail=1
  else echo "FAIL $name"; tail -20 "$out/$name.log"; fail=1; fi
}
for t in "${files[@]}"; do
  check_test "$t" "$(basename "$t" .mojo)"
done
# The default macOS build runs a job inside its start call (no job threads), so the job tests cannot see a cancel arrive or
# two threads race there: they say so and pass. The same tests built with -D MXFRAME_JOB_THREADS (what an app that wants
# background jobs on a Mac builds with, F-175) are the ones that check cancelling and racing, so run those here too.
if [ $# -eq 0 ] && [ "$(uname)" != "Linux" ]; then
  for name in test_job test_jobs test_sort_spill test_permutation_spill test_parquet_dict_race; do
    check_test "$root/mojo/tests/$name.mojo" "${name}_threads" -D MXFRAME_JOB_THREADS
  done
fi

# F-138b: the same package built the way an app with no GPU, thread pool or codecs builds it (what every non-Linux
# target gets by default, forced here with -D). Its inputs are a zstd and a snappy file from the normal build.
if [ $# -eq 0 ]; then
  sw="$root/mojo/tests/switches/test_switches"
  if mojo build --disable-warnings "$root/mojo/tools/pq_write_sample.mojo" -I "$root/build" -Xlinker -L"$lib" -Xlinker -lzstd -Xlinker -lsnappy \
       -Xlinker -rpath -Xlinker "$lib" -o "$out/pq_write_sample" > "$out/pq_write_sample.build.log" 2>&1 \
     && "$out/pq_write_sample" "$out/switches-zstd.parquet" zstd 500 200 && "$out/pq_write_sample" "$out/switches-snappy.parquet" snappy 500 200 \
     && mojo build --disable-warnings "$sw.mojo" -I "$root/build" -D MXFRAME_NO_GPU -D MXFRAME_SERIAL -D MXFRAME_NO_CODECS \
       -o "$out/test_switches" > "$out/test_switches.build.log" 2>&1; then
    if nm "$out/test_switches" | grep -E "ZSTD_|snappy_|AsyncRT_Device" > /dev/null; then
      echo "FAIL test_switches links a codec or GPU symbol (the Linux main() still brings the CPU runtime; the Mac check proves none)"; fail=1
    elif (cd "$rundir" && "$out/test_switches" "$out/switches-zstd.parquet" "$out/switches-snappy.parquet") > "$out/test_switches.log" 2>&1; then
      echo "ok   test_switches (no GPU, serial, no codecs)"
    else echo "FAIL test_switches"; tail -20 "$out/test_switches.log"; fail=1; fi
    # F-175: the same build with job threads (-D MXFRAME_JOB_THREADS): a job on its own pthread whose kernels run in a loop
    if mojo build --disable-warnings "$sw.mojo" -I "$root/build" -D MXFRAME_NO_GPU -D MXFRAME_SERIAL -D MXFRAME_NO_CODECS -D MXFRAME_JOB_THREADS \
         -o "$out/test_switches_threads" > "$out/test_switches_threads.build.log" 2>&1; then
      if nm "$out/test_switches_threads" | grep -E "ZSTD_|snappy_|AsyncRT_Device" > /dev/null; then
        echo "FAIL test_switches_threads links a codec or GPU symbol"; fail=1
      elif (cd "$rundir" && "$out/test_switches_threads" "$out/switches-zstd.parquet" "$out/switches-snappy.parquet") > "$out/test_switches_threads.log" 2>&1; then
        echo "ok   test_switches_threads (serial kernels, jobs on threads)"
      else echo "FAIL test_switches_threads"; tail -20 "$out/test_switches_threads.log"; fail=1; fi
    else echo "FAIL (build) test_switches_threads"; tail -20 "$out/test_switches_threads.build.log"; fail=1; fi
  else echo "FAIL (build) test_switches"; tail -20 "$out/test_switches.build.log"; fail=1; fi
fi
# The Mojo example in docs/reference.md compiles, runs and prints what that page shows.
if [ $# -eq 0 ]; then
  printf 'region,qty,revenue\nWest,3,12.50\nEast,1,4.00\nWest,5,20.25\nNorth,4,9.99\nEast,7,30.00\n' > "$out/readme-orders.csv"
  want=$'West 3275 2\nEast 3000 1\nNorth 999 1\nnet 14324 edits 2'
  if mojo build --disable-warnings "$root/mojo/tools/readme_example.mojo" -I "$root/build" -Xlinker -L"$lib" -Xlinker -lzstd \
       -Xlinker -lsnappy -Xlinker -rpath -Xlinker "$lib" -o "$out/readme_example" > "$out/readme_example.build.log" 2>&1 \
     && [ "$("$out/readme_example" "$out/readme-orders.csv")" == "$want" ]; then
    echo "ok   readme_example"
  else echo "FAIL readme_example (docs/reference.md shows its output)"; tail -5 "$out/readme_example.build.log"; fail=1; fi
fi
exit $fail
