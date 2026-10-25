# F-185: the GPU launch plan agrees with the buffers the query allocates, on every target's limits. Host code only
# (no GPU needed): for a grid of program sizes, the mode `_choose_mode` picks fits the target's shared memory
# (SMEM_LIMIT: 32 KB on an Apple GPU, 48 KB elsewhere), uses no 64-bit atomics on an Apple GPU (modes 0, 3, 4, 5), and
# never launches more blocks than `scratch_rows` gives per-block rows for.
# Run also with `MOJO_TEST_FLAGS="-D OMXCORE_APPLE_GPU_SHAPE" pixi run mojo-test mojo/tests/test_gpu_plan.mojo`.
from omxcore.kernels.pipeline_gpu import (_choose_mode, _narrow, _smem_bytes, _grid_blocks, scratch_rows, SMEM_LIMIT,
                                          APPLE_GPU)


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
    for groups in [1, 2, 5, 7, 41, 300, 2000, 5000, 100_000, 3_000_000]:
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


def main() raises:
    test_plans_fit_and_scratch_covers_the_grid()
