# F-185: the GPU launch plan agrees with the buffers the query allocates, on every target's limits. Host code only
# (no GPU needed): for a grid of program sizes, the mode `_choose_mode` picks fits the target's shared memory
# (SMEM_LIMIT: 32 KB on an Apple GPU, 48 KB elsewhere), uses no 64-bit atomics on an Apple GPU (modes 0, 3, 4, 5), and
# never launches more blocks than `scratch_rows` gives per-block rows for.
# Run also with `MOJO_TEST_FLAGS="-D OMXCORE_APPLE_GPU_SHAPE" pixi run mojo-test mojo/tests/test_gpu_plan.mojo`.
from omxcore.kernels.pipeline_gpu import (_choose_mode, _narrow, _smem_bytes, _grid_blocks, scratch_rows, SMEM_LIMIT,
                                          APPLE_GPU)
from omxcore.kernels.kernels_aot_gpu import group_shared_bytes, GROUP_SMEM_LIMIT, SHARED_MAX_GROUPS


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


def test_plans_fit_and_scratch_covers_the_grid() raises:
    var cores: List[Int] = [1, 8, 10, 40, 82]
    var rows: List[Int] = [1, 1000, 1_000_000, 50_000_000]
    var plans = 0
    var declined = 0
    var modes = List[Int](length=6, fill=0)
    var narrows = 0
    for groups in [1, 2, 5, 7, 41, 300, 2000, 5000, 8192, 10_000, 12_288, 100_000, 3_000_000]:
        for slots in [0, 1, 3, 6, 12]:
            for regs in [2, 6, 12, 23, 40]:
                var code_ints = 16 + 5 * (regs * 2 + slots)
                var mode = _choose_mode(groups, slots, regs, code_ints, 0, True)
                if mode < 0:
                    declined += 1
                    continue
                plans += 1
                modes[mode] += 1
                var narrow = _narrow(mode, groups, slots, regs, code_ints)
                if narrow:
                    narrows += 1
                var what = "groups " + String(groups) + " slots " + String(slots) + " regs " + String(regs) + " mode " + String(mode)
                expect(_smem_bytes(mode, groups, slots, regs, code_ints, narrow) <= SMEM_LIMIT, "shared memory fits: " + what)
                comptime if APPLE_GPU:
                    expect(mode == 0 or mode == 3 or mode == 4 or mode == 5, "no atomic mode on an Apple GPU: " + what)
                for c in cores:
                    var sr = scratch_rows(mode, groups, slots, c)
                    for n in rows:
                        var b = _grid_blocks(mode, groups, slots, regs, code_ints, narrow, n, c)
                        expect(b >= 1, "at least one block: " + what)
                        if mode == 0 or mode == 3 or mode == 4 or mode == 5:
                            expect(b <= sr, "scratch has a row per block (" + String(b) + " > " + String(sr) + "): " + what)
    expect(plans > 0, "some plans")
    expect(narrows > 0, "some plans use narrow blocks")
    comptime if APPLE_GPU:
        expect(modes[5] > 0 and modes[4] > 0 and modes[0] > 0, "Apple's plan reaches modes 0, 4 and 5")
        expect(modes[1] == 0 and modes[2] == 0, "and never an atomic mode")
    print("ok", plans, "plans", declined, "declined", "apple" if APPLE_GPU else "")


def test_the_group_kernels_shared_table_fits_the_target() raises:
    """F-195 (omxcore#19): `run_group`'s shared path asked for groups x 4 bytes up to 12288 groups (48 KB), past an
    Apple GPU's 32 KB. Its table is now capped by the target; past the cap the global path runs (0 bytes)."""
    comptime if APPLE_GPU:
        expect(GROUP_SMEM_LIMIT == 32 * 1024 and SHARED_MAX_GROUPS == 8192, "Apple: 32 KB, 8192 groups")
    else:
        expect(GROUP_SMEM_LIMIT == 48 * 1024 and SHARED_MAX_GROUPS == 12288, "NVIDIA: 48 KB, 12288 groups as before")
    for ng in [1, 41, 1000, 8191, 8192, 8193, 10_000, 12_288, 12_289, 100_000]:
        var b = group_shared_bytes(ng, 10 * ng)
        expect(b <= GROUP_SMEM_LIMIT, String(ng) + " groups ask for " + String(b) + " bytes")
        expect(b == (ng * 4 if ng <= SHARED_MAX_GROUPS else 0), String(ng) + " groups: shared up to the cap, global past it")
    expect(group_shared_bytes(1000, 3999) == 0, "too few rows per group: the global path")
    print("ok group kernel within", GROUP_SMEM_LIMIT, "bytes", "apple" if APPLE_GPU else "")


def main() raises:
    test_plans_fit_and_scratch_covers_the_grid()
    test_the_group_kernels_shared_table_fits_the_target()
