#!/usr/bin/env bash
# Build the Mojo kernel package from source.
# Usage:  bash scripts/build_kernels.sh
#
# Produces  mxframe/kernels.mojoc  from  mxframe/kernels/*.mojo

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_ROOT/kernels"
OUT_PKG="$REPO_ROOT/kernels.mojoc"
export TMPDIR="${MXFRAME_TEST_TMPDIR:-$HOME/.cache/mxframe-test-tmp}"   # not the shared /tmp tmpfs
mkdir -p "$TMPDIR"
STAGE_DIR="$(mktemp -d)"

cleanup() {
	rm -rf "$STAGE_DIR"
}
trap cleanup EXIT

echo "── Staging kernels in $STAGE_DIR ──"
cp "$SRC_DIR/__init__.mojo" "$STAGE_DIR/__init__.mojo"
cp "$SRC_DIR/_atomic.mojo" "$STAGE_DIR/_atomic.mojo"
cp "$SRC_DIR/group_sum.mojo" "$STAGE_DIR/group_sum.mojo"
cp "$SRC_DIR/group_min.mojo" "$STAGE_DIR/group_min.mojo"
cp "$SRC_DIR/group_max.mojo" "$STAGE_DIR/group_max.mojo"
cp "$SRC_DIR/group_count.mojo" "$STAGE_DIR/group_count.mojo"
cp "$SRC_DIR/sort_indices.mojo" "$STAGE_DIR/sort_indices.mojo"
cp "$SRC_DIR/unique_mask.mojo" "$STAGE_DIR/unique_mask.mojo"
cp "$SRC_DIR/join_count.mojo" "$STAGE_DIR/join_count.mojo"
cp "$SRC_DIR/join_scatter.mojo" "$STAGE_DIR/join_scatter.mojo"
cp "$SRC_DIR/filter_gather.mojo" "$STAGE_DIR/filter_gather.mojo"
cp "$SRC_DIR/join_count_left.mojo" "$STAGE_DIR/join_count_left.mojo"
cp "$SRC_DIR/join_scatter_left.mojo" "$STAGE_DIR/join_scatter_left.mojo"
cp "$SRC_DIR/group_composite.mojo" "$STAGE_DIR/group_composite.mojo"
cp "$SRC_DIR/masked_global_agg.mojo" "$STAGE_DIR/masked_global_agg.mojo"

echo "── Building kernels from staged subset ──"
if command -v mojo >/dev/null 2>&1; then
	mojo package "$STAGE_DIR" -o "$OUT_PKG"
elif command -v pixi >/dev/null 2>&1; then
	pixi run mojo package "$STAGE_DIR" -o "$OUT_PKG"
else
	echo "Error: neither 'mojo' nor 'pixi' found in PATH" >&2
	exit 127
fi
echo "── Built: $OUT_PKG ($(du -h "$OUT_PKG" | cut -f1)) ──"
