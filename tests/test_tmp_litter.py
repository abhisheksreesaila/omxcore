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


def test_litter_owned_by_this_run_is_told_from_another_process():
    mine_by_hash = "omxcore-shim-" + scratch.shim_dir(ROOT).name.split("-")[-1]
    names = {"mxframe-x-4242", "mxframe-x-14242", mine_by_hash, "omxcore-shim-0000000000"}
    assert scratch.owned(names, ROOT, pid=4242) == {"mxframe-x-4242", mine_by_hash}


def _session(tmp_path, body):
    """A tiny pytest session that loads this repo's conftest hook; returns (exit code, output)."""
    (tmp_path / "probe_plugin.py").write_text(
        "import importlib.util\n"
        f"s = importlib.util.spec_from_file_location('repo_conftest', {str(ROOT / 'tests' / 'conftest.py')!r})\n"
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
        "pytest_sessionfinish = m.pytest_sessionfinish\n")
    (tmp_path / "test_probe.py").write_text(body)
    env = {k: v for k, v in os.environ.items() if k != "MXFRAME_LITTER_CHECK"}
    env["PYTHONPATH"] = str(tmp_path)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "probe_plugin", str(tmp_path / "test_probe.py")],
                       capture_output=True, text=True, env=env, cwd=tmp_path)
    return r.returncode, r.stdout


def test_a_session_that_leaves_an_entry_in_tmp_fails_and_a_clean_one_passes(tmp_path):
    code, out = _session(tmp_path, "def test_ok():\n    assert 1 + 1 == 2\n")
    assert code == 0, out
    probe = "/tmp/mxframe-f194-probe-{pid}"
    body = ("import os\n"
            "def test_leaves_litter():\n"
            f"    os.mkdir({probe!r}.format(pid=os.getpid()))\n")
    try:
        code, out = _session(tmp_path, body)
        assert code != 0 and "MXFRAME_LITTER_CHECK=0" in out and "mxframe-f194-probe-" in out, out
    finally:
        for p in Path("/tmp").glob("mxframe-f194-probe-*"):
            p.rmdir()
