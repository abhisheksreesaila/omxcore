"""Where tests and scripts put scratch (F-194). Never /tmp: it is a shared tmpfs with a per-user quota.

`MXFRAME_TEST_TMPDIR` moves the folder; the default is ~/.cache/mxframe-test-tmp (as scripts/run_tests.sh, mojo_test.sh
and build_mx.sh already use for TMPDIR)."""
import hashlib
import os
import re
import tempfile
from pathlib import Path


def root() -> Path:
    d = Path(os.environ.get("MXFRAME_TEST_TMPDIR") or Path.home() / ".cache" / "mxframe-test-tmp")
    d.mkdir(parents=True, exist_ok=True)
    return d


def use_for_tempfile() -> Path:
    """Point `tempfile` (so pytest's tmp_path too) and child processes (TMPDIR) at the scratch folder."""
    d = root()
    os.environ["TMPDIR"] = str(d)
    tempfile.tempdir = str(d)
    return d


def shim_dir(checkout: Path, kind: str = "shim") -> Path:
    """A folder for `omxcore` -> this checkout links, so `import omxcore` resolves to it (git worktrees included)."""
    d = root() / f"{kind}-{hashlib.sha1(str(checkout).encode()).hexdigest()[:10]}"
    d.mkdir(exist_ok=True)
    return d


def owned(names, checkout: Path, pid: int | None = None) -> set[str]:
    """The names this run could have made: they hold its pid or the checkout hash that names its shims."""
    pid = os.getpid() if pid is None else pid
    h = hashlib.sha1(str(checkout).encode()).hexdigest()[:10]
    return {n for n in names if h in n or re.search(rf"(?<!\d){pid}(?!\d)", n)}


def litter(tmp: str | Path = "/tmp") -> set[str]:
    """The `mxframe-*` / `omxcore-*` entries in `tmp`: a run must not add any."""
    t = Path(tmp)
    return {p.name for pat in ("mxframe-*", "omxcore-*") for p in t.glob(pat)}
