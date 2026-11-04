"""Shared pytest setup: make `import omxcore` resolve to this checkout (works in git worktrees)."""
import atexit
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
import scratch  # noqa: E402

scratch.use_for_tempfile()   # F-194: tempfile, pytest's tmp_path and child processes use ~/.cache, never the shared /tmp
_shim = scratch.shim_dir(_ROOT, "omxcore-shim")
for _name, _target in (("omxcore", _ROOT), ("mxframe", _ROOT / "compat" / "mxframe")):   # mxframe: the forwarding module
    if not (_shim / _name).exists():
        (_shim / _name).symlink_to(_target)
sys.path.insert(0, str(_shim))

TPCH_DATA = Path(os.environ.get("MXFRAME_TPCH_DATA", "/home/abhisheksreesaila/Projects/mxframe/data"))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs TPC-H Parquet data on disk / runs long")

# Tests must never write observations into the user's real calibration file.
if "MXFRAME_CALIBRATION" not in os.environ:
    _calib = scratch.root() / f"mxframe-test-calibration-{os.getpid()}.json"
    os.environ["MXFRAME_CALIBRATION"] = str(_calib)
    atexit.register(lambda: _calib.unlink(missing_ok=True))

_litter_before = scratch.litter()


def pytest_sessionfinish(session, exitstatus):
    """A run that leaves a new /tmp/mxframe-* or /tmp/omxcore-* entry fails (the controller only, not xdist workers)."""
    if hasattr(session.config, "workerinput"):
        return
    new = scratch.litter() - _litter_before
    if new:
        print(f"\nF-194: this run left entries in /tmp: {sorted(new)}")
        session.exitstatus = 1


import pytest  # noqa: E402


def _mx_env():
    return {k: v for k, v in os.environ.items() if k.startswith("MXFRAME_")}


@pytest.fixture(autouse=True)
def _no_env_leaks():
    """A test that leaves an MXFRAME_* variable changed fails here, and the variable is put back. Leaks broke later
    tests on the same xdist worker twice (`MXFRAME_HOST_MEM_BUDGET`: small in-memory queries streamed and failed
    with a 0-byte budget). Use monkeypatch.setenv, or restore in a finally. Declared before the other autouse
    fixtures, so it checks after their monkeypatch has undone its changes."""
    before = _mx_env()
    yield
    after = _mx_env()
    if after != before:
        for k in after.keys() - before.keys():
            del os.environ[k]
        os.environ.update(before)
        changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
        pytest.fail(f"the test left environment variables changed: {changed} "
                    f"(was {[before.get(k) for k in changed]}, now {[after.get(k) for k in changed]})")


@pytest.fixture(autouse=True)
def _strict_native(monkeypatch):
    """An unexpected exception in a native hook is raised in the tests, not turned into a quiet fallback
    (a test that wants the fallback unsets it)."""
    monkeypatch.setenv("MXFRAME_STRICT_NATIVE", "1")
