"""The docs site in docs/site/ (F-047..F-052): every page loads the shared CSS and JS, opens with a
"how to read this" box, and every local link, script, image and data file it names exists."""
from html.parser import HTMLParser
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "docs" / "site"
PAGES = sorted(SITE.glob("*.html"))


class _Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs, self.classes, self.theme = [], [], None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html":
            self.theme = a.get("data-theme")
        for key in ("href", "src"):
            if a.get(key):
                self.refs.append(a[key])
        self.classes += (a.get("class") or "").split()


def _parse(path):
    p = _Page()
    p.feed(path.read_text())
    return p


def _local(ref):
    return not re.match(r"^(https?:|mailto:|#|data:)", ref)


def test_site_has_index_and_shared_assets():
    assert (SITE / "index.html").is_file()
    assert (SITE / "site.css").is_file() and (SITE / "site.js").is_file()
    css = (SITE / "site.css").read_text()
    for series, token in (("--mx", "--oq-accent"), ("--dk", "--oq-data-4"), ("--pl", "--oq-data-1")):
        assert re.search(re.escape(series) + r":\s*var\(" + re.escape(token) + r"\)", css), \
            f"{series} (a chart series) must read the omxqua token {token}"


def test_bench_link_reaches_benchmark_files():
    assert (SITE / "bench" / "m2s-engine.csv").is_file()


LANDING = {"story.html", "deck.html"}   # the story and its slides: their own frame instead of the docs top bar, the same tokens (below)


@pytest.mark.parametrize("page", [p for p in PAGES if p.name not in LANDING], ids=lambda p: p.name)
def test_page_shell(page):
    p = _parse(page)
    assert p.theme == "dark", "night is the default theme (omxcore blue on dark, as on omxlabs.tech)"
    assert "site.css" in p.refs and "site.js" in p.refs
    assert "read" in p.classes, 'every page opens with a "how to read this" box'


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_local_links_exist(page):
    for ref in filter(_local, _parse(page).refs):
        target = (page.parent / ref.split("#")[0].split("?")[0]).resolve()
        assert target.exists(), f"{page.name} links to missing {ref}"


def test_nav_pages_exist():
    js = (SITE / "site.js").read_text()
    for href in re.findall(r'\["([\w-]+\.html)"', js):
        assert (SITE / href).is_file(), f"nav lists missing page {href}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_data_files_named_in_pages_exist(page):
    text = page.read_text()
    for name in re.findall(r'bench/([\w.-]+\.(?:csv|json))', text):
        assert (SITE / "bench" / name).is_file(), f"{page.name} reads missing bench/{name}"


def test_pages_workflow_copies_bench_files():
    wf = (ROOT / ".github" / "workflows" / "pages.yml").read_text()
    assert "docs/site" in wf and "-L" in wf, "deploy must dereference the bench link"


def test_csv_files_the_pages_read_have_no_quotes():
    """site.js loadCSV splits on bare commas; a quoted field with a comma would be torn apart."""
    bench = ROOT / "docs" / "bench"
    names = set()
    for f in [*SITE.glob("*.html"), *SITE.glob("*.js"), *(SITE / "js").glob("*.js")]:
        names |= {Path(n).name for n in re.findall(r"bench/([\w.-]+\.csv)", f.read_text())}
    assert len(names) >= 10
    for n in sorted(names):
        assert (bench / n).is_file(), n
        assert '"' not in (bench / n).read_text(), f"{n} has a quote: site.js loadCSV cannot parse quoted fields"


def test_pages_workflow_publishes_the_kernel_visualizer():
    """site.js links visualize/omxcore_pipeline.html; it lives outside docs/site and is copied in at deploy."""
    wf = (ROOT / ".github" / "workflows" / "pages.yml").read_text()
    assert "cp -r visualize _site/visualize" in wf
    assert (ROOT / "visualize" / "omxcore_pipeline.html").is_file()
    assert "visualize/omxcore_pipeline.html" in (SITE / "site.js").read_text()


OMXQUA = ["omxqua/src/fonts.css", "omxqua/src/tokens.css", "omxqua/src/components.css", "omxqua/src/omxqua.js"]


def test_vendored_omxqua_is_complete():
    """omxqua ships with the site (bin/omxqua-sync docs/site/ --ref v0.1.4): nothing loads from a CDN."""
    for ref in OMXQUA:
        assert (SITE / ref).is_file(), ref
    assert (SITE / "omxqua" / "VERSION").read_text().startswith("ref: v0.1.4")
    for font in re.findall(r"url\(\.\./fonts/([\w.-]+)\)", (SITE / "omxqua/src/fonts.css").read_text()):
        assert (SITE / "omxqua/fonts" / font).is_file(), font


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_page_loads_omxqua_in_order_with_the_dev_family(page):
    """The four files in the documented order, root classes oq-root + oq-night + oq-family-dev + oq-app-core (+ oq-top for the pages
    with the shell), no font service and no omxqua CDN."""
    text = page.read_text()
    at = [text.index(f'"{ref}"') for ref in OMXQUA]
    assert at == sorted(at), f"{page.name}: omxqua files out of order"
    root = re.search(r"<html[^>]*class=\"([^\"]*)\"", text).group(1).split()
    assert {"oq-root", "oq-night", "oq-family-dev", "oq-app-core"} <= set(root), root   # omxcore blue (brand.css), replacing teal (2026-10-20)
    if page.name != "deck.html":
        assert "oq-top" in root, "the shell is Top islands"
    assert "googleapis" not in text and "jsdelivr.net/gh/" not in text, page.name
    assert "Space+Mono" not in text


COLOUR = re.compile(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(\s*\d|\bhsla?\(")


def _style_and_script(text):
    """The CSS and JS of a page (not its links like href="#journey")."""
    parts = re.findall(r"<style>(.*?)</style>", text, re.S) + re.findall(r"<script>(.*?)</script>", text, re.S)
    parts += re.findall(r'style="([^"]*)"', text)
    return "\n".join(parts)


@pytest.mark.parametrize("path", [*PAGES, SITE / "site.css", SITE / "site.js", SITE / "js" / "bench.js"], ids=lambda p: p.name)
def test_no_page_defines_its_own_colour(path):
    """Every colour is an omxqua --oq-* token. A literal hex, rgb() or hsl() in a page's CSS or JS would be a second palette
    (charts read tokens through cssVar; they never carry a colour of their own); no gradients, blur or glass either."""
    text = path.read_text()
    code = _style_and_script(text) if path.suffix == ".html" else text
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    assert not COLOUR.findall(code), f"{path.name}: {COLOUR.findall(code)}"
    for banned in ("gradient(", "backdrop-filter", "blur("):
        assert banned not in code, f"{path.name}: {banned}"


def test_every_link_that_looks_like_a_button_springs():
    """omxqua.js squashes `button, a.cta, a.more, [data-press], ...` on press; a link styled as a button or a card is none
    of those, so it must carry data-press or it doesn't spring (owner, 2026-10-20: "the buttons aren't bouncy")."""
    pages = [*PAGES, SITE / "site.js"]
    for page in pages:
        for tag in re.findall(r"<a\b[^>]*>", page.read_text()):
            cls = re.search(r'class="([^"]*)"', tag)
            if cls and {"btn", "card", "pill", "dk-brand"} & set(cls.group(1).split()):
                assert "data-press" in tag, f"{page.name}: {tag}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_page_wears_the_omxcore_brand(page):
    """omxcore's brand from omxlabs.tech: brand.css (the blue and the chip mark) after omxqua's components and before
    site.css, and the chip favicon."""
    text = page.read_text()
    assert text.index('"omxqua/src/components.css"') < text.index('"brand.css"') < text.index('"site.css"' if page.name != "deck.html" else '"site.js"')
    assert '<link rel="icon" href="favicon-core.svg" type="image/svg+xml">' in text
    assert (SITE / "favicon-core.svg").is_file()


def test_brand_blue_and_mark_come_from_omxlabs():
    css = (SITE / "brand.css").read_text()
    assert "--ox-core: #4aa8ff" in css and ".oq-night.oq-app-core { --oq-accent: var(--ox-core); --oq-on-accent: #0b0b0d; }" in css
    assert "prefers-reduced-motion" in css
    assert 'class="lm lm-core"' in (SITE / "site.js").read_text()
