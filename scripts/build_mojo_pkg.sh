#!/usr/bin/env bash
# Build the Python-free Mojo package build/omxcore.mojoc (F-131, renamed F-165), and for one stable release the same
# source again as build/mxframe.mojoc, so an app that still writes `from mxframe import ...` keeps building from the
# command in STABLE.md. (A .mojoc is named by its package directory; a forwarding package would have to re-export
# every module and name of the first one, so building the one source under two names is the plain way.)
#
# The API lives in mojo/omxcore/. The kernels stay in kernels_aot/ (one source: the Python path builds its .so
# files from them with flat imports, `from pipeline_cpu import ...`). They are copied into the package as
# omxcore/kernels/ with those imports made package-relative, because a .mojoc does not carry modules that were
# only on the -I path, and a shared library that re-imports @export functions does not export them.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$root/build"
# Many builds may run at once (pytest -n workers, mojo-test beside pytest): one at a time behind a lock, staged in
# a private directory and published by rename, so a reader never sees a half-written package; skipped when the
# sources and the compiler are unchanged.
# macOS has neither flock nor sha1sum: a mkdir lock (atomic) and shasum take their place there.
work="$(mktemp -d "$root/build/pkg.XXXXXX")"
if command -v flock > /dev/null; then
  exec 9>"$root/build/.omxcore-pkg.lock"
  flock 9
  trap 'rm -rf "$work"' EXIT
else
  # The lock directory holds the owner's pid: a lock whose owner is dead (killed build) is cleared, and a wait of
  # MXFRAME_LOCK_WAIT seconds (default 1800) ends with a message instead of blocking forever.
  lockdir="$root/build/.omxcore-pkg.lockdir"
  waited=0
  until mkdir "$lockdir" 2> /dev/null; do
    owner="$(cat "$lockdir/pid" 2> /dev/null || true)"
    if [ -n "$owner" ] && ! kill -0 "$owner" 2> /dev/null; then
      echo "build_mojo_pkg.sh: clearing the lock of dead build $owner" >&2
      rm -rf "$lockdir"; continue
    fi
    if [ "$waited" -ge "${MXFRAME_LOCK_WAIT:-1800}" ]; then
      echo "build_mojo_pkg.sh: waited ${waited}s for $lockdir (held by pid ${owner:-unknown}); remove it if no build runs" >&2; exit 1
    fi
    sleep 1; waited=$((waited + 1))
  done
  echo $$ > "$lockdir/pid"
  trap 'rm -rf "$work" "$lockdir"' EXIT
fi
sha1() { if command -v sha1sum > /dev/null; then sha1sum; else shasum; fi; }
stamp="$( { mojo --version 2>/dev/null; cat "$root"/mojo/omxcore/*.mojo "$root"/kernels_aot/*.mojo "$0"; } | sha1 | cut -c1-40)"
# build_one <package name>: stage the sources as <name>/, precompile, publish build/<name>.mojoc
build_one() {
  local name="$1" out="$root/build/$1.mojoc"
  if [ -f "$out" ] && [ "$(cat "$out.stamp" 2>/dev/null)" = "$stamp" ]; then
    echo "build/$name.mojoc is up to date"
    return
  fi
  local stage="$work/$name"
  mkdir -p "$stage/kernels"
  cp "$root"/mojo/omxcore/*.mojo "$stage/"
  cp "$root"/kernels_aot/*.mojo "$stage/kernels/"
  echo '"""The native kernels (copied from kernels_aot/ by scripts/build_mojo_pkg.sh: edit them there)."""' > "$stage/kernels/__init__.mojo"
  for f in "$root"/kernels_aot/*.mojo; do
    m="$(basename "$f" .mojo)"
    sed -i.bak -E "s/^from $m import/from .$m import/" "$stage"/kernels/*.mojo   # -i.bak: GNU and BSD sed both take it
  done
  rm -f "$stage"/kernels/*.bak
  mojo precompile --disable-warnings "$stage" -o "$work/$name.mojoc"
  mv -f "$work/$name.mojoc" "$out"
  echo "$stamp" > "$out.stamp"
  echo "built build/$name.mojoc"
}
build_one omxcore
build_one mxframe
