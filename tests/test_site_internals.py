"""F-050: docs/site/internals.html names only things that exist.

The page marks what it names so this test can find it: layer data lives in one JSON block
(`<script id="layers-data">`), and the walkthrough, tables and diagram carry `data-path` (a repo file),
`data-sym` (`file::name`, the name must appear in that file), `data-env` (an environment variable that
the code reads) and `data-label` (a provenance "path" value). On top of the marked ones, every file-like
token and every MXFRAME_* token in the page text is checked too, so an unmarked mention cannot rot.
"""
from html.parser import HTMLParser
from pathlib import Path
import json
import re

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "site" / "internals.html"
SEARCH_DIRS = ("", "kernels_aot", "scripts", "tests", "docs", "kernels")
CODE_GLOBS = ("*.py", "kernels_aot/*.mojo", "kernels_aot/*.py", "scripts/*.sh")
# env vars that only the test-suite reads
TEST_ONLY = {"MXFRAME_TEST_DEVICE", "MXFRAME_TPCH_DATA", "MXFRAME_TPCH_SF10", "MXFRAME_FUZZ_SEEDS",
             "MXFRAME_FUZZ_LARGE_SEEDS"}


class _Marked(HTMLParser):
    def __init__(self):
        super().__init__()
        self.marks = {"path": [], "sym": [], "env": [], "label": []}
        self.json = None
        self._in_json = False
        self._buf = []
        self.text = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        for k in self.marks:
            if f"data-{k}" in a:
                self.marks[k].append(a[f"data-{k}"])
        if tag == "script":
            self._skip += 1
            if a.get("id") == "layers-data":
                self._in_json = True

    def handle_endtag(self, tag):
        if tag == "script":
            self._skip -= 1
            if self._in_json:
                self.json = json.loads("".join(self._buf))
                self._in_json = False

    def handle_data(self, data):
        if self._in_json:
            self._buf.append(data)
        elif not self._skip:
            self.text.append(data)


@pytest.fixture(scope="module")
def page():
    assert PAGE.is_file(), "docs/site/internals.html is missing"
    p = _Marked()
    p.feed(PAGE.read_text())
    return p


def _resolve(token):
    for d in SEARCH_DIRS:
        cand = ROOT / d / token
        if cand.exists():
            return cand
    return None


def _code_text():
    out = []
    for g in CODE_GLOBS:
        out += [p.read_text(errors="ignore") for p in ROOT.glob(g)]
    return "\n".join(out)


def _layer_files(page):
    return [f for layer in page.json["layers"] for f in layer["files"]]


def test_page_is_wired_like_the_others(page):
    html = PAGE.read_text()
    assert 'class="oq-root oq-night oq-family-dev oq-app-core oq-top" data-theme="dark"' in html
    assert 'href="site.css"' in html and 'src="site.js"' in html and '<main class="main" id="main">' in html
    assert html.index("<h1") < html.index('class="read"'), "calm first (2026-10-20): the title, then the how-to-read box"


def test_layer_stack_has_the_seven_layers(page):
    names = [layer["name"] for layer in page.json["layers"]]
    assert len(names) == 7
    for needle in ("LazyFrame", "validation", "ptimizer", "Native", "Fused", "AOT", "auto"):
        assert any(needle in n for n in names), needle
    for layer in page.json["layers"]:
        for key in ("does", "files", "seam_in", "seam_out", "declines", "next", "see"):
            assert layer.get(key), f"{layer['name']}: {key} is empty"


def test_every_layer_file_exists(page):
    files = _layer_files(page)
    assert "native_exec.py" in files and "kernels_aot/executor_cpu.mojo" in files
    for f in files:
        assert (ROOT / f).is_file(), f"layer names missing file {f}"


def test_every_marked_path_exists(page):
    assert page.marks["path"], "the page marks the repo files it names"
    for f in page.marks["path"]:
        assert (ROOT / f).exists(), f"page names missing path {f}"


def test_every_marked_symbol_exists_in_its_file(page):
    syms = page.marks["sym"] + [s for layer in page.json["layers"] for s in layer.get("symbols", [])]
    assert len(syms) >= 15
    for s in syms:
        f, name = s.split("::")
        assert (ROOT / f).is_file(), f"{s}: no such file"
        assert re.search(rf"\b{re.escape(name)}\b", (ROOT / f).read_text()), f"{name} not in {f}"


def test_every_provenance_label_is_set_by_the_code(page):
    labels = page.marks["label"] + [x for layer in page.json["layers"] for x in layer.get("labels", [])]
    assert {"native", "cpu_aot"} <= set(labels) and any(x.startswith("fused_pipeline_") for x in labels)
    code = _code_text()
    for lab in labels:
        pat = f'"{lab}"' if not lab.endswith("_") else f'"{lab}'  # prefix: f"fused_pipeline_{used}"
        assert pat in code or f'f"{lab}' in code, f"no code sets provenance path {lab}"


def test_every_env_var_is_read_by_the_code(page):
    code = _code_text()
    tests = "\n".join(p.read_text(errors="ignore") for p in [*(ROOT / "tests").glob("*.py"), *(ROOT / "scripts").glob("*.py")]
                      if p.name != Path(__file__).name)
    envs = set(page.marks["env"])
    for must in ("MXFRAME_NATIVE", "MXFRAME_FUSED_PIPELINE", "MXFRAME_CPU_THREADS", "MXFRAME_EXEC_CPU_LIB"):
        assert must in envs, must
    for v in envs:
        assert v in (tests if v in TEST_ONLY else code), f"{v} is not read anywhere"


def test_env_table_covers_every_variable_the_code_reads(page):
    """Anything new that reads MXFRAME_* has to be added to the page's table."""
    in_code = set(re.findall(r"MXFRAME_[A-Z0-9_]+", _code_text()))
    in_code -= {"MXFRAME_TPCH_DATA"}            # tests only, listed below it
    missing = in_code - set(page.marks["env"])
    assert not missing, f"internals.html env table lacks {sorted(missing)}"


def test_every_token_in_the_text_exists(page):
    text = " ".join(page.text) + " " + json.dumps(page.json)
    files = set(re.findall(r"[\w./-]+\.(?:py|mojo|sh|so|mojoc)\b", text))
    files = {f for f in files if not f.startswith(("http", "docs/site/"))}
    assert len(files) >= 20
    for f in sorted(files):
        assert _resolve(f.strip("/")) is not None, f"page mentions {f}, which does not exist"
    for v in set(re.findall(r"MXFRAME_[A-Z0-9_]+", text)):
        assert v in page.marks["env"], f"{v} is mentioned but not in the env table"


def test_github_links_point_at_the_blob_path(page):
    base = "https://github.com/abhisheksreesaila/omxcore/blob/main/"
    html = PAGE.read_text()
    assert base in html or "BLOB" in html
    assert "const BLOB" in html and base in html


def test_stack_story_matches_the_dispatch_code(page):
    """Joins run first and non-aggregate plans call native helpers from the AOT path; the page says so."""
    text = " ".join(page.text) + " " + json.dumps(page.json)
    assert "never half-run" not in text and "is never half-run" not in text
    custom = (ROOT / "custom_ops.py").read_text()
    # order in compute: joins are materialised before the native/fused tries
    assert custom.index("self._materialize_joins(plan)") < custom.index("_native_exec.try_execute(self, plan")
    assert custom.index("_native_exec.try_execute(self, plan") < custom.index("try_fused_aggregate(self, plan")
    # the AOT path calls these native helpers, which the layer text names
    for fn in ("try_filter", "try_filter_topk", "try_topk", "try_distinct"):
        assert f"native_exec.{fn}(" in custom, fn
        assert re.search(rf"\bdef {fn}\b", (ROOT / "native_exec.py").read_text()), fn
    join_src = (ROOT / "join_engine.py").read_text()
    assert "native_exec.try_filter(" in join_src and "native_exec.try_gather(" in join_src
    aot = next(l for l in page.json["layers"] if l["id"] == "aot")
    assert "join" in aot["does"] and "try_filter_topk" in aot["does"]


def test_thread_default_text_matches_the_code(page):
    html = PAGE.read_text()
    assert "hyperthreads included" in html
    aot = (ROOT / "aot_kernels.py").read_text()
    assert "sched_getaffinity" in aot and "physical" in (ROOT / "pipeline_kernels.py").read_text()


def test_native_decline_list_covers_the_common_reasons(page):
    native = next(l for l in page.json["layers"] if l["id"] == "native")["declines"]
    assert "for example" in native
    for needle in ("one table", "bytecode compiler", "HAVING", "sort is by an expression"):
        assert needle in native, needle
    src = (ROOT / "native_exec.py").read_text()
    for reason in ("input is not a scan of one table", "expression not supported by the bytecode compiler",
                   "HAVING other than", "sort by an expression"):
        assert reason in src, reason


def test_optimizer_arrow_does_not_claim_a_decline():
    html = PAGE.read_text()
    assert "never declines" in html
    assert 'i === 3 ? " if it declines"' not in html
