# M2 (docs/goals/apple-gpu/GOAL.md): a program that instantiates every GPU kernel the package launches (the pipeline
# kernel in each block shape, its init and merge kernels), so `mojo build --emit asm` writes each one's device code
# beside it: `.ptx` for NVIDIA, `.ll` (AIR) for `--target-accelerator metal:1`. scripts/check_apple_gpu.py reads them.
# Built only, never run.
from omxcore import read_csv, CsvOptions, Query, col, lit, sum, count, GPU, DeviceTable


def main() raises:
    var csv = read_csv("orders.csv", CsvOptions())
    var t = csv.take_table().share()
    var dt = DeviceTable(t)
    var r = Query(t).on_device(dt).where(col("qty") > lit(1)).group_by(col("status"), groups=5).agg(count(), sum(col("qty"))).compute(GPU)
    print(r.values(0)[0])
