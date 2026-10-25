"""The Apple GPU goal's checker (docs/goals/apple-gpu/GOAL.md): one PASS / FAIL line per milestone.

    pixi run python scripts/check_apple_gpu.py            all milestones (M5 runs the Mojo suites: ~10 minutes)
    pixi run python scripts/check_apple_gpu.py --quick    M1-M4 only
    pixi run python scripts/check_apple_gpu.py --ptx-baseline <omxcore build dir>   prints the NVIDIA PTX fingerprint

M1, M3, M4 read the newest report of the `apple-gpu` workflow (macos-26 runner) for this repo's main branch; M2 builds
ci/mac/kernels_emit.mojo here, for Metal (AIR) and for this machine's NVIDIA GPU (PTX).
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache" / "omxcore-apple-gpu"
ENV = dict(os.environ, TMPDIR=str(Path.home() / ".cache" / "mxframe-test-tmp"))


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=kw.pop("cwd", ROOT), env=ENV, capture_output=True, text=True, **kw)


def report():
    """The newest apple-gpu run's report.txt (downloaded once per run id), and the commit it ran."""
    r = run(["gh", "run", "list", "--workflow", "apple-gpu.yml", "--branch", "main", "-L", "5",
             "--json", "databaseId,headSha,status,conclusion"])
    if r.returncode != 0:
        return None, "gh run list failed: " + r.stderr.strip()[:200]
    runs = [x for x in json.loads(r.stdout) if x["status"] == "completed"]
    if not runs:
        return None, "no completed apple-gpu run on main"
    rid, sha = runs[0]["databaseId"], runs[0]["headSha"][:7]
    d = CACHE / str(rid)
    if not (d / "report.txt").exists():
        shutil.rmtree(d, ignore_errors=True)
        r = run(["gh", "run", "download", str(rid), "-n", "apple-gpu-report", "-D", str(d)])
        if r.returncode != 0:
            return None, f"run {rid} ({sha}): no report artifact"
    return (d / "report.txt").read_text(errors="replace"), f"run {rid} at {sha}"


def m1(text):
    dev = re.search(r"^PROBE DEVICE: (.+ · metal)$", text, re.M)
    want = ["typed-write", "int-address", "address-in-buffer", "learned-address-not-arg", "shared-16kb", "shared-32kb",
            "int64-maths", "shuffle-32"]
    bad = [w for w in want if not re.search(rf"^PROBE {w}: ok$", text, re.M)]
    if dev and not bad:
        return True, dev.group(1)
    return False, ("no Metal device line; " if not dev else "") + ("not ok: " + ", ".join(bad) if bad else "")


def m3(text):
    dev = re.search(r"^ENGINE DEVICE: (.+ · metal)$", text, re.M)
    ok = re.search(r"^ENGINE AVAILABLE: True$", text, re.M)
    return bool(dev and ok), dev.group(1) if dev else "no ENGINE DEVICE line naming metal"


def m4(text):
    refs = dict(re.findall(r"^R (\S+) (.*)$", text, re.M))
    got = re.findall(r"^Q (\S+) ran=(\S+) (.*)$", text, re.M)
    if len(refs) < 8:
        return False, f"{len(refs)} reference answers (want >= 8)"
    bad = []
    for qid in refs:
        mine = [g for g in got if g[0] == qid]
        if not mine:
            bad.append(f"{qid}: missing")
        elif mine[0][1] != "gpu":
            bad.append(f"{qid}: ran={mine[0][1]} {mine[0][2][:80]}")
        elif mine[0][2] != refs[qid]:
            bad.append(f"{qid}: {mine[0][2][:60]} != {refs[qid][:60]}")
    return not bad, f"{len(refs) - len(bad)}/{len(refs)} equal pyarrow on the GPU" + ("; " + "; ".join(bad) if bad else "")


def build_package(src):
    r = run(["bash", "scripts/build_mojo_pkg.sh"], cwd=src)
    if r.returncode != 0:
        raise RuntimeError("package build failed: " + (r.stdout + r.stderr)[-600:])
    return Path(src) / "build"


def emit(build_dir, metal):
    """Builds kernels_emit.mojo with --emit asm; returns {file name: text} of the device-code sidecars."""
    out = Path(tempfile.mkdtemp(dir=CACHE))
    cmd = ["mojo", "build", "--disable-warnings", "--emit", "asm", "-I", str(build_dir), "ci/mac/kernels_emit.mojo",
           "-o", str(out / "k.s")]
    if metal:
        cmd[2:2] = ["--target-accelerator", "metal:1"]
    r = run(["pixi", "run"] + cmd)
    files = {p.name: p.read_text(errors="replace") for p in out.iterdir() if p.suffix in (".ptx", ".ll")}
    shutil.rmtree(out, ignore_errors=True)
    if r.returncode != 0 or not files:
        raise RuntimeError("emit failed: " + (r.stdout + r.stderr)[-800:])
    return files


def ptx_fingerprint(files):
    """The kernels' PTX with their (parameter-hashed) symbol names blanked, as one hash over the sorted bodies."""
    bodies = sorted(re.sub(r"omxcore_kernels_pipeline_gpu\w+", "K", t) for n, t in files.items() if n.endswith(".ptx"))
    return hashlib.sha256("\n\0\n".join(bodies).encode()).hexdigest()[:16] + f" ({len(bodies)} kernels)"


def m2():
    build = build_package(ROOT)
    air = emit(build, metal=True)
    bad = []
    for name, text in air.items():
        body = "\n".join(l for l in text.splitlines() if not l.startswith(("target datalayout", "declare")))
        for what, pat in [("double", r"\bdouble\b"), ("atomicrmw", r"\batomicrmw\b"), ("inline asm", r"\basm\b")]:
            if re.search(pat, body):
                bad.append(f"{name[:40]}: {what}")
    want = (ROOT / "ci/mac/nvidia_ptx.sha").read_text().strip()
    have = ptx_fingerprint(emit(build, metal=False))
    if have != want:
        bad.append(f"NVIDIA PTX changed: {have} != baseline {want}")
    return not bad, f"{len(air)} Metal kernels clean, NVIDIA PTX = baseline" if not bad else "; ".join(bad[:6])


def m5():
    r = run(["pixi", "run", "mojo-test"])
    fails = [l for l in r.stdout.splitlines() if l.startswith("FAIL")]
    oks = [l for l in r.stdout.splitlines() if l.startswith("ok")]
    return r.returncode == 0, f"{len(oks)} ok" + (", " + "; ".join(fails[:4]) if fails else "")


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[1] == "--ptx-baseline":
        print(ptx_fingerprint(emit(Path(sys.argv[2]), metal=False)))
        return 0
    quick = "--quick" in sys.argv
    text, where = report()
    results = []
    for name, fn in [("M1 runner premises", lambda: m1(text)), ("M2 kernel compiles for Metal", m2),
                     ("M3 engine sees the Apple GPU", lambda: m3(text)), ("M4 done means", lambda: m4(text)),
                     ("M5 NVIDIA unchanged (mojo-test)", m5)]:
        if quick and name.startswith("M5"):
            continue
        if text is None and name[:2] in ("M1", "M3", "M4"):
            results.append((name, False, where))
            continue
        try:
            ok, why = fn()
        except Exception as e:  # a failed build is a FAIL with its reason
            ok, why = False, str(e)[-400:]
        results.append((name, ok, why))
    print(f"(runner report: {where})")
    for name, ok, why in results:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {why}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
