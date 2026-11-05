"""Parent side: launch one worker subprocess per (engine, query, sf), with a timeout, and
turn whatever it printed (or how it died) into a result record."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
PREFIX = "@@RESULT "


def shim_dir() -> str:
    """A dir with `omxcore` -> this checkout, so subprocesses import THIS checkout's code."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    import scratch
    d = scratch.shim_dir(REPO_ROOT, "bench-shim")
    link = d / "omxcore"
    if link.is_symlink() and Path(os.readlink(link)) != REPO_ROOT:
        link.unlink()
    if not link.exists():
        link.symlink_to(REPO_ROOT)
    return str(d)


def resolve_gpu(spec: str) -> tuple[str, str]:
    """`--gpu` is an nvidia-smi index (or already a GPU-... UUID). CUDA_VISIBLE_DEVICES numbers
    devices in its own order (fastest first unless CUDA_DEVICE_ORDER=PCI_BUS_ID), which need not
    match nvidia-smi, so pass the UUID. Returns (value for CUDA_VISIBLE_DEVICES, description)."""
    spec = str(spec).strip()
    if spec.startswith(("GPU-", "MIG-")):
        return spec, f"UUID {spec}"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    for line in out.strip().splitlines():
        idx, uuid, name = [x.strip() for x in line.split(",", 2)]
        if idx == spec:
            return uuid, f"nvidia-smi index {idx} ({name}), {uuid}"
    return spec, f"CUDA index {spec} (nvidia-smi lookup failed; may not match nvidia-smi's numbering)"


def worker_env(gpu: str) -> dict:
    """`gpu` is the CUDA_VISIBLE_DEVICES value (see resolve_gpu)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([shim_dir(), str(SCRIPTS_DIR)])
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def parse_output(text: str) -> dict:
    merged: dict = {}
    for line in text.splitlines():
        if line.startswith(PREFIX):
            try:
                merged.update(json.loads(line[len(PREFIX):]))
            except json.JSONDecodeError:
                pass
    return merged


def _tail(text: str, n: int = 3) -> str:
    lines = [l for l in text.strip().splitlines() if l.strip() and not l.startswith(PREFIX)]
    return " | ".join(lines[-n:])[:300]


def run_one(engine: str, query: int, data_dir: str, result_path: str, timeout: float,
            gpu: str = "1", mode: str = "user-felt") -> dict:
    """Returns a dict with import_ms/first_ms/repeat_ms/rows/error/mxframe_path (no correctness)."""
    cmd = [sys.executable, "-m", "tpch_user.worker", "--engine", engine, "--query", str(query),
           "--data-dir", data_dir, "--result", result_path,
           "--skip-repeat-over-ms", str(timeout * 1000 * 0.4), "--mode", mode]
    with tempfile.TemporaryFile("w+") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=worker_env(gpu),
                             cwd=str(REPO_ROOT), start_new_session=True)
        timed_out = False
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            rc = p.wait()
        log.seek(0)
        text = log.read()
    rec = parse_output(text)
    rec.pop("engine", None)
    rec.pop("query", None)
    rec.pop("repeat_reloaded", None)
    rec["mode"] = mode
    if timed_out:
        if "first_ms" in rec and "error" not in rec:
            rec["repeat_ms"] = None
            rec["repeat_note"] = "repeat timed out"
        else:
            rec["error"] = f"TIMEOUT after {timeout:.0f}s"
    elif "first_ms" not in rec and "error" not in rec:
        if rc is not None and rc < 0:
            try:
                sig = signal.Signals(-rc).name
            except ValueError:
                sig = str(-rc)
            rec["error"] = f"CRASH signal {sig}: {_tail(text)}"
        else:
            rec["error"] = f"CRASH rc={rc}: {_tail(text)}"
    return rec


def check_mxframe_path() -> str:
    """Import omxcore the way a worker does and return where it came from (must be REPO_ROOT)."""
    r = subprocess.run([sys.executable, "-c",
                        "import os, omxcore; print(os.path.dirname(os.path.abspath(omxcore.__file__)))"],
                       capture_output=True, text=True, env=worker_env("-1"), cwd=str(REPO_ROOT), timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"cannot import omxcore in a worker: {r.stderr.strip()[-300:]}")
    return r.stdout.strip().splitlines()[-1]


def warm_page_cache(data_dir: str, tables: list[str]) -> None:
    """Read the Parquet files once so no engine pays the cold-disk cost (all read from page cache)."""
    for name in tables:
        path = os.path.join(data_dir, f"{name}.parquet")
        if os.path.exists(path):
            with open(path, "rb") as f:
                while f.read(64 << 20):
                    pass
