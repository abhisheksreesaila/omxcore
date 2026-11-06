"""F-036: the shipped kernel libraries find the Mojo runtime from the environment they are
installed in, never from the machine that built them."""
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

KERNELS = Path(__file__).resolve().parent.parent / "kernels_aot"
LIBS = sorted(KERNELS.glob("*.so"))
RUNTIME = ("libKGENCompilerRTShared.so", "libAsyncRTMojoBindings.so",
           "libMSupportGlobals.so", "libAsyncRTRuntimeGlobals.so")


linux_only = pytest.mark.skipif(sys.platform != "linux", reason="the shipped libraries' ELF RUNPATH: macOS dylibs are out of scope (scripts/set_runpath.sh)")


def runpath(path: Path) -> list:
    """The RUNPATH / RPATH entries of an ELF64 little-endian file (reads the dynamic section)."""
    data = path.read_bytes()
    (shoff,) = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    dynamic = next(s for s in sections if s[1] == 6)                  # SHT_DYNAMIC
    strtab = sections[dynamic[6]]                                      # its sh_link
    out = []
    for off in range(dynamic[4], dynamic[4] + dynamic[5], 16):
        tag, val = struct.unpack_from("<qQ", data, off)
        if tag in (15, 29):                                            # DT_RPATH, DT_RUNPATH
            start = strtab[4] + val
            out += data[start:data.index(b"\0", start)].decode().split(":")
    return out


def test_every_registered_library_is_there():
    """The list lives once, in scripts/record_build_hash.py (a new kernel library is registered there; a hand-kept
    list here went stale when libmxhashjoin_cpu.so arrived)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("rbh", KERNELS.parent / "scripts" / "record_build_hash.py")
    rbh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rbh)
    assert [p.name for p in LIBS] == sorted(rbh.LIBRARIES)


@linux_only
@pytest.mark.parametrize("lib", LIBS, ids=lambda p: p.name)
def test_runpath_is_relative_to_the_library(lib):
    assert runpath(lib) == ["$ORIGIN/../.pixi/envs/default/lib", "$ORIGIN/../../modular/lib",
                            "$ORIGIN/../../../.."]


def _env_lib() -> Path:
    import sys
    return Path(sys.prefix) / "lib"


@linux_only
@pytest.mark.parametrize("layout", ["venv", "pip_modular"])
def test_a_library_loads_in_a_foreign_interpreter_from_an_installed_layout(tmp_path, layout):
    """Copy a library and the Mojo runtime into the layout pip or conda would make, then dlopen
    it from the system Python (no conda RPATH of its own): it loads only if the RUNPATH works."""
    system_python = shutil.which("python3", path="/usr/bin")
    if system_python is None or not all((_env_lib() / n).exists() for n in RUNTIME):
        pytest.skip("needs /usr/bin/python3 and the Mojo runtime in this environment")
    site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    kernels = site / "omxcore" / "kernels_aot"
    kernels.mkdir(parents=True)
    runtime_dir = tmp_path / "venv" / "lib" if layout == "venv" else site / "modular" / "lib"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for n in RUNTIME:
        shutil.copy(_env_lib() / n, runtime_dir / n)
    lib = kernels / "libmxexec_cpu.so"
    shutil.copy(KERNELS / lib.name, lib)
    code = f"import ctypes; ctypes.CDLL({str(lib)!r}); print('loaded')"
    r = subprocess.run([system_python, "-I", "-c", code], capture_output=True, text=True, env={})
    assert (r.stdout.strip(), r.returncode) == ("loaded", 0), r.stderr


@linux_only
def test_the_same_copy_fails_without_the_runtime_so_the_check_means_something(tmp_path):
    system_python = shutil.which("python3", path="/usr/bin")
    if system_python is None:
        pytest.skip("needs /usr/bin/python3")
    kernels = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "omxcore" / "kernels_aot"
    kernels.mkdir(parents=True)
    shutil.copy(KERNELS / "libmxexec_cpu.so", kernels)
    code = f"import ctypes; ctypes.CDLL({str(kernels / 'libmxexec_cpu.so')!r})"
    r = subprocess.run([system_python, "-I", "-c", code], capture_output=True, text=True, env={})
    assert r.returncode != 0 and "libKGENCompilerRTShared.so" in r.stderr


@linux_only
def test_a_checkout_prefers_its_own_pixi_env_over_a_runtime_planted_next_to_it(tmp_path):
    """A checkout at /tmp/mxframe must not load /tmp/modular/lib (writable by anyone) when its
    own pixi env has the runtime."""
    system_python = shutil.which("python3", path="/usr/bin")
    if system_python is None or not all((_env_lib() / n).exists() for n in RUNTIME):
        pytest.skip("needs /usr/bin/python3 and the Mojo runtime in this environment")
    repo = tmp_path / "omxcore"
    (repo / "kernels_aot").mkdir(parents=True)
    own = repo / ".pixi" / "envs" / "default" / "lib"
    planted = tmp_path / "modular" / "lib"
    own.mkdir(parents=True)
    planted.mkdir(parents=True)
    for n in RUNTIME:
        shutil.copy(_env_lib() / n, own / n)
    (planted / "libKGENCompilerRTShared.so").write_bytes(b"not an ELF file: the attacker's code")
    lib = repo / "kernels_aot" / "libmxexec_cpu.so"
    shutil.copy(KERNELS / lib.name, lib)
    code = f"import ctypes; ctypes.CDLL({str(lib)!r}); print('loaded')"
    r = subprocess.run([system_python, "-I", "-c", code], capture_output=True, text=True, env={})
    assert (r.stdout.strip(), r.returncode) == ("loaded", 0), r.stderr
