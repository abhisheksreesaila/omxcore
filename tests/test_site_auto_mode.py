"""F-052: docs/site/auto-mode.html ("How device=\"auto\" decides").

The page quotes thresholds and replays the planner's rules in the browser, so these tests pin it to
the code: every constant on the page is read back from the module that defines it, every bench file
it names exists, and the claims in "Honest limits" are checked against the source they describe.
"""
import ast
import csv
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from statistics import median

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
PAGE = DOCS / "site" / "auto-mode.html"
BENCH = DOCS / "bench"


def _page_text():
    return PAGE.read_text()


def _rules():
    m = re.search(r'<script type="application/json" id="rules">(.*?)</script>', _page_text(), re.S)
    assert m, "the page must carry its constants in <script type=application/json id=rules>"
    return json.loads(m.group(1))


def _module_constants(path):
    """Module-level NAME = <constant expression> assignments, evaluated without builtins."""
    out = {}
    for node in ast.parse(path.read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        value = getattr(node, "value", None)
        for t in targets:
            if isinstance(t, ast.Name) and value is not None:
                try:
                    out[t.id] = eval(compile(ast.Expression(value), str(path), "eval"), {"__builtins__": {}})
                except Exception:
                    pass
    return out


def _default_link():
    """link_gbps / link_latency_ms literals inside default_calibration()."""
    tree = ast.parse((ROOT / "calibration.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "default_calibration")
    found = {}
    for call in ast.walk(fn):
        if isinstance(call, ast.Call):
            for kw in call.keywords:
                if kw.arg in ("link_gbps", "link_latency_ms"):
                    found[kw.arg] = ast.literal_eval(kw.value)
    return found


# ── the page exists and is wired to its data ────────────────────────────────

def test_page_exists_and_opens_with_a_read_box():
    text = _page_text()
    assert 'class="oq-root oq-night oq-family-dev oq-app-core oq-top" data-theme="dark"' in text
    assert 'href="site.css"' in text and 'src="site.js"' in text and '<main class="main" id="main">' in text
    assert text.index("<h1") < text.index('class="read"'), "calm first (2026-10-20): the title, then the how-to-read box"


def test_every_bench_file_named_exists():
    names = set(re.findall(r"bench/([\w.-]+\.(?:csv|json))", _page_text()))
    for needed in ("final-sf10.csv", "final-sf1.csv", "engine-sf10.csv", "m2s-engine.csv", "auto-coproc.json"):
        assert needed in names, f"the page should read bench/{needed}"
    for n in names:
        assert (BENCH / n).is_file(), n
    for i in (1, 2, 3):
        assert f"final-sf1-extra{i}.csv" in _page_text() or "final-sf1-extra" in _page_text()


def test_page_has_no_typed_results():
    """The measured results come from bench/ at runtime: no result tables or ms values in the markup."""
    html = re.sub(r"<script.*?</script>", "", _page_text(), flags=re.S)
    html = re.sub(r"<(style|pre|code)\b.*?</\1>", "", html, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", html)
    for bad in ("22/22", "21/22", "353 ms", "177 ms"):
        assert bad not in text, f"{bad!r} is typed on the page; read it from bench/ instead"


# ── constants on the page == constants in the code ──────────────────────────

def test_rules_constants_match_the_code():
    rules = _rules()["constants"]
    assert rules, "no constants listed"
    for name, spec in rules.items():
        consts = _module_constants(ROOT / spec["file"])
        assert name in consts, f"{name} is not a module-level constant of {spec['file']}"
        assert consts[name] == pytest.approx(spec["value"]), f"{name}: page says {spec['value']}, {spec['file']} says {consts[name]}"


def test_rules_cover_the_thresholds_the_explainer_uses():
    rules = _rules()["constants"]
    for name in ("SMALL_QUERY_ROWS", "COLD_GPU_START_MS", "COLD_NATIVE_MS_PER_MROW", "GPU_WIN_MARGIN", "PREFER_CPU_MARGIN", "JOIN_GPU_SAFETY", "IDLE_BEFORE_BACKGROUND_S",
                 "MIN_UPLOAD_BYTES", "MIN_SHARE", "DEFAULT_GPU_ALPHA_MS", "RESIDENT_MEM_FRACTION", "FAILED_RETRY_S",
                 "_DEFAULT_GPU_INIT_MS"):
        assert name in rules, name


def test_default_cost_fits_and_link_match_calibration():
    data = _rules()
    fits = _module_constants(ROOT / "calibration.py")["_DEFAULT_FITS"]
    assert data["fits"], "the explainer prices with the built-in default fits"
    for k, v in data["fits"].items():
        assert list(fits[k]) == pytest.approx(v), k
    link = _default_link()
    assert data["link"]["gbps"] == link["link_gbps"]
    assert data["link"]["latency_ms"] == link["link_latency_ms"]


def test_page_prices_the_cpu_side_with_every_native_fit():
    """The explainer's CPU column is the native executor's: every native_* default fit is on the page."""
    fits = _module_constants(ROOT / "calibration.py")["_DEFAULT_FITS"]
    native = {k for k in fits if k.startswith("native_")}
    assert native and native <= set(_rules()["fits"])


class _Spans(HTMLParser):
    def __init__(self):
        super().__init__()
        self.found, self._open = [], None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "data-const" in a:
            self._open = [a["data-const"], a.get("data-fmt", "plain"), ""]

    def handle_data(self, data):
        if self._open is not None:
            self._open[2] += data

    def handle_endtag(self, tag):
        if self._open is not None:
            self.found.append(tuple(self._open))
            self._open = None


def _fmt(value, fmt):
    if fmt == "ms":
        return f"{value * 1000:g} ms"
    if fmt == "s":
        return f"{value:g} s"
    if fmt == "pct":
        return f"{value * 100:g}%"
    if fmt == "rows":
        return f"{int(value):,}"
    if fmt == "mib":
        return f"{int(value) >> 20} MiB"
    if fmt == "x":
        return f"{value:g}"
    return f"{value:g}"


def test_thresholds_quoted_in_prose_match_the_constants():
    rules = _rules()["constants"]
    p = _Spans()
    p.feed(_page_text())
    assert len(p.found) >= 8, "prose should quote its thresholds through data-const spans"
    for name, fmt, text in p.found:
        assert name in rules, f"data-const={name} is not in the rules block"
        assert text.strip() == _fmt(rules[name]["value"], fmt), f"{name}: page text {text!r}"


# ── the measured results ────────────────────────────────────────────────────

def _first(rows):
    d = {}
    for r in rows:
        if r["engine"] in ("mx-cpu", "mx-gpu", "mx-auto") and r["first_ms"]:
            d.setdefault((r["engine"], int(r["query"])), []).append(float(r["first_ms"]))
    return d


def _within(d, agg=median):
    n = 0
    for q in range(1, 23):
        m = {e: agg(d[(e, q)]) for e in ("mx-cpu", "mx-gpu", "mx-auto")}
        n += m["mx-auto"] <= 1.10 * min(m["mx-cpu"], m["mx-gpu"])
    return n


def _csv(name):
    return list(csv.DictReader(open(BENCH / name)))


def test_headline_counts_reproduce_from_the_csvs():
    """22/22 at SF10, 21/22 at SF1 on the median of 4 runs: the numbers the page computes at runtime."""
    assert _within(_first(_csv("final-sf10.csv"))) == 22
    runs = ["final-sf1.csv"] + [f"final-sf1-extra{i}.csv" for i in (1, 2, 3)]
    assert _within(_first(sum((_csv(f) for f in runs), []))) == 21
    assert sorted(_within(_first(_csv(f)), min) for f in runs) == [17, 18, 20, 21]


def test_coproc_numbers_are_cited_and_found_in_their_source():
    data = json.loads((BENCH / "auto-coproc.json").read_text())

    def has(text, n):
        return re.search(rf"(?<![\d.]){re.escape(str(n))}(?![\d])", text) is not None

    for block in ("fused_stage", "session", "repeat_speedup"):
        src = ROOT / data[block]["source"]
        assert src.is_file(), f"{block}: source {src} missing"
        text = src.read_text()
        values = []
        for row in data[block].get("rows", []):
            values += [v for k, v in row.items() if isinstance(v, (int, float))]
        values += [data[block]["factor"]] if "factor" in data[block] else []
        assert values
        for v in values:
            assert has(text, v), f"{block}: {v} not found in {data[block]['source']}"


# ── honest limits: each claim is tied to the code it describes ──────────────

def test_limit_native_executor_declines_only_under_a_split_policy():
    """The site says the policy (set only for a predicted split or gpu_fraction) is what makes it decline."""
    assert "CPU+GPU co-processing policy active" in (ROOT / "native_exec.py").read_text()
    assert "if decision.split or forced" in (ROOT / "lazy_frame.py").read_text()


def test_limit_auto_counts_distinct_on_codes_unless_steered_to_the_gpu():
    src = (ROOT / "lazy_frame.py").read_text()
    assert 'cd_codes = device == "cpu" or codes_decision is not None' in src
    assert 'if prefer == "gpu" or "gpu" in (placement or {}).values():' in src

def test_limit_whole_query_cpu_price_is_native_but_the_split_uses_the_older_kernel():
    assert _module_constants(ROOT / "auto_mode.py")["NATIVE_CPU_FACTOR"] == 1.0, "revisit the site's limits text"
    src = (ROOT / "auto_mode.py").read_text()
    assert 'calib.estimate("native_join", "cpu"' in src and 'calib.estimate("native_group", "cpu"' in src
    assert '"native_join@cpu:g8000000"' in (ROOT / "calibration.py").read_text()


def test_limit_placer_and_residency_tracker_are_not_on_the_query_path():
    skip = {"auto_mode.py", "residency.py"}
    for f in ROOT.glob("*.py"):
        if f.name in skip or f.name in ("setup.py", "_modidx.py"):
            continue
        text = f.read_text()
        assert "Placer(" not in text and "get_residency(" not in text and "ResidencyTracker(" not in text, f.name


def test_no_gpu_operator_is_blocked_today():
    """The explainer's 'is any operator blocked' rule says the list is empty."""
    assert _module_constants(ROOT / "auto_mode.py")["KNOWN_BROKEN_GPU"] == {}


def test_limit_auto_places_the_whole_query_not_each_operator():
    assert "Phase 1 places the whole" in (ROOT / "lazy_frame.py").read_text()


# ── the old page points to the new one ──────────────────────────────────────

def test_old_page_is_a_short_note_linking_to_the_new_one():
    old = (DOCS / "auto-mode.html").read_text()
    assert 'href="site/auto-mode.html"' in old
    assert len(old.splitlines()) < 40


def test_benchmark_pass_margin_is_its_own_constant():
    """'Within 10%' is the benchmark's criterion (F-009), not the planner's GPU_WIN_MARGIN."""
    html = _page_text()
    assert re.search(r"const BENCH_PASS_MARGIN = 0\.10;", html)
    start = html.index("function perQuery")
    assert "BENCH_PASS_MARGIN" in html[start:start + 900]
    for m in re.finditer(r"ratio <= 1 \+ ([\w.]+)", html):
        assert m.group(1) == "BENCH_PASS_MARGIN"
    assert "X(1 + K.GPU_WIN_MARGIN)" not in html
    assert "within 10%" in (DOCS / "plan.md").read_text()
