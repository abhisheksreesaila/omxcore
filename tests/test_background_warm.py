"""import omxcore starts a CPU-only warm-up thread (kernel libraries + engine modules)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(code, env_extra=None):
    import os
    env = dict(os.environ, **(env_extra or {}))
    prog = f"import sys; sys.path.insert(0, {str(ROOT)!r}); import conftest\n" + code
    return subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, check=True, env=env).stdout.strip()


def test_warm_loads_cpu_libraries_without_touching_the_gpu():
    out = _run("""
import omxcore, time
assert omxcore._warm_thread is not None
omxcore._warm_thread.join(30)
if sys.platform == 'darwin':          # no /proc/self/maps: ask dyld for the loaded images
    import ctypes
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    maps = '\\n'.join(dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count()))
else:
    maps = open('/proc/self/maps').read()
print('libmxkernels_aot.so' in maps, 'libmxpipeline_cpu.so' in maps, 'libcuda' in maps,
      'omxcore.custom_ops' in sys.modules, 'pandas' in sys.modules)
""")
    assert out.split() == ["True", "True", "False", "True", "False"]


def test_warm_can_be_disabled():
    out = _run("import omxcore; print(omxcore._warm_thread is None, 'omxcore.custom_ops' in sys.modules)",
               {"MXFRAME_WARM": "0"})
    assert out.split() == ["True", "False"]


def test_read_parquet_does_not_import_pandas(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq
    p = tmp_path / "a.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3]}), p)
    out = _run(f"import omxcore as mx; mx.read_parquet({str(p)!r}, columns=['x']).compute(device='cpu'); print('pandas' in sys.modules)",
               {"MXFRAME_WARM": "0"})
    assert out == "False"


def test_fork_right_after_import_does_not_deadlock():
    out = _run("""
import os, signal, time
import numpy as np, pyarrow as pa
import omxcore
from omxcore import LazyFrame, col
ok = 0
for delay in (0.001, 0.005, 0.01, 0.02):
    pid = os.fork()
    if pid == 0:
        signal.alarm(20)
        t = pa.table({"k": np.arange(100, dtype=np.int32) % 3, "v": np.ones(100)})
        LazyFrame(t).groupby("k").agg(col("v").sum()).compute(device="cpu")
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    ok += int(os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0)
print(ok)
""")
    assert out.splitlines()[-1] == "4"


def test_import_defers_heavy_modules():
    out = _run("import omxcore; print(*[m in sys.modules for m in "
               "('sqlglot', 'pyarrow.parquet', 'pyarrow.csv', 'pyarrow.compute', 'pandas')])",
               {"MXFRAME_WARM": "0"})
    assert out.split() == ["False"] * 5
