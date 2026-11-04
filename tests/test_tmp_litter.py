"""F-194: tests and scripts leave nothing in /tmp (a shared tmpfs with a per-user quota)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import scratch  # noqa: E402


def _root_in_fresh_process(**env):
    e = {k: v for k, v in os.environ.items() if k != "MXFRAME_TEST_TMPDIR"}
    e.update(env)
    code = f"import sys; sys.path.insert(0, {str(ROOT / 'scripts')!r}); import scratch; print(scratch.root())"
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e, check=True).stdout.strip()


def test_scratch_root_defaults_to_the_cache_folder_and_follows_the_override(tmp_path):
    assert _root_in_fresh_process() == str(Path.home() / ".cache" / "mxframe-test-tmp")
    assert _root_in_fresh_process(MXFRAME_TEST_TMPDIR=str(tmp_path / "x")) == str(tmp_path / "x")


def test_tempfile_and_pytest_scratch_live_under_the_scratch_root(tmp_path):
    assert Path(tempfile.gettempdir()) == scratch.root()
    assert os.environ["TMPDIR"] == str(scratch.root())          # subprocesses (and the Mojo compiler) inherit it
    assert scratch.root() in tmp_path.parents


def test_import_shims_are_under_the_scratch_root_not_tmp():
    assert scratch.shim_dir(ROOT).parent == scratch.root()
    assert Path("/tmp") not in scratch.shim_dir(ROOT).parents


def test_litter_check_names_exactly_the_new_entries(tmp_path):
    (tmp_path / "mxframe-old").mkdir()
    before = scratch.litter(tmp_path)
    assert before == {"mxframe-old"}
    (tmp_path / "omxcore-src-1").mkdir()
    (tmp_path / "mxframe-shim-2").mkdir()
    (tmp_path / "unrelated").mkdir()
    assert scratch.litter(tmp_path) - before == {"omxcore-src-1", "mxframe-shim-2"}
