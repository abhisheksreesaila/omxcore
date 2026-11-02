// omxcore docs site: the omxqua shell (Top islands on desktop, tab bar and a "More" sheet on the phone), the light / night
// choice shared by every page, tooltip, small SVG and CSV helpers. Every page includes this file (in <head>) and wraps its
// content as <div class="screen"><main class="main">…</main></div>; the islands are built here. Pages are listed once, here.
// A page can set window.SHELL_ITEMS = [{href: "#journey", label: "Journey"}, …] to list its own sections instead (the story does).
const PAGES = [
  ["story.html", "The story", "book", "Story"],
  ["engine.html", "Engine", "cpu"],
  ["benchmarks.html", "Benchmarks", "chart"],
  ["auto-mode.html", "CPU or GPU", "zap"],
  ["internals.html", "Internals", "layers"],
  ["how-it-got-fast.html", "How it got fast", "gauge"],
];
const REPO = "https://github.com/abhisheksreesaila/omxcore";
const ICONS = {  // Lucide-style strokes (ISC)
  book: '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
  cpu: '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
  chart: '<path d="M3 3v18h18"/><path d="M18 17V9M13 17V5M8 17v-3"/>',
  zap: '<path d="M13 2 3 14h9l-1 8 10-12h-9z"/>',
  layers: '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
  gauge: '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
  dots: '<circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/><circle cx="5" cy="12" r="1"/>',
  home: '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M9 22V12h6v10"/>',
  moon: '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9z"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/>',
  out: '<path d="M15 3h6v6M10 14 21 3M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  flag: '<path d="M4 22V4a1 1 0 0 1 .4-.8A6 6 0 0 1 8 2c3 0 5 2 8 2a6 6 0 0 0 4-1v12a6 6 0 0 1-4 1c-3 0-5-2-8-2a6 6 0 0 0-4 1"/>',
  file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
};
// omxcore's mark (omxlabs/assets/marks.js "core"): a chip with nine cores; brand.css lights them in a wave on hover
const CORE_MARK = '<span class="pm"><svg class="lm lm-core" viewBox="0 0 24 24" aria-hidden="true"><rect class="s" x="5" y="5" width="14" height="14" rx="2.5"/><path class="s pin" d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/><circle class="f d" style="--i:0" cx="9" cy="9" r="1.2"/><circle class="f d" style="--i:1" cx="12" cy="9" r="1.2"/><circle class="f d" style="--i:2" cx="15" cy="9" r="1.2"/><circle class="f d" style="--i:1" cx="9" cy="12" r="1.2"/><circle class="f d" style="--i:2" cx="12" cy="12" r="1.2"/><circle class="f d" style="--i:3" cx="15" cy="12" r="1.2"/><circle class="f d" style="--i:2" cx="9" cy="15" r="1.2"/><circle class="f d" style="--i:3" cx="12" cy="15" r="1.2"/><circle class="f d" style="--i:4" cx="15" cy="15" r="1.2"/></svg></span>';
const ico = (n) => `<svg class="ico" viewBox="0 0 24 24" aria-hidden="true">${ICONS[n]}</svg>`;

// ---------- light / night: oq-night by default (omxcore's brand is blue on dark), oq-light when chosen; one choice for every page ("mx-theme") ----------
function applyTheme(t) {
  const r = document.documentElement;
  r.classList.toggle("oq-night", t !== "light"); r.classList.toggle("oq-light", t === "light");
  r.dataset.theme = t === "light" ? "light" : "dark";
}
applyTheme(localStorage.getItem("mx-theme"));

function toggleTheme() {
  const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(t);
  localStorage.setItem("mx-theme", t);
  syncThemeButton();
  document.dispatchEvent(new Event("themechange"));
}
function syncThemeButton() {
  const btn = document.getElementById("theme-btn");
  if (!btn) return;
  const dark = document.documentElement.dataset.theme === "dark";
  btn.setAttribute("aria-pressed", dark);
  btn.innerHTML = ico(dark ? "sun" : "moon") + `<span class="t">${dark ? "Light" : "Night"}</span>`;
}

// ---------- the shell ----------
function shellItems() {
  if (window.SHELL_ITEMS) return window.SHELL_ITEMS.map((it, i) => ({ ...it, icon: it.icon || "flag", on: false, i }));
  const here = location.pathname.split("/").pop() || "index.html";
  return PAGES.map(([href, label, icon, short], i) => ({ href, label, short: short || label, icon, on: href === here, i }));
}

function renderShell() {
  const screen = document.querySelector(".screen");
  if (!screen || screen.querySelector(".topnav")) return;
  const items = shellItems(), here = location.pathname.split("/").pop() || "index.html";
  const sections = !!window.SHELL_ITEMS;
  const btn = (it) => `<button type="button" data-i="${it.i}" data-href="${it.href}"${it.on ? ' class="on" aria-current="page"' : ""}>${it.label}</button>`;
  // desktop: brand island · nav pill · the right island (the Beacon's place; nothing here is live, so it holds the theme choice)
  const brand = `<a class="brand-isl" href="${sections ? "#top" : "index.html"}">${CORE_MARK}omxcore</a>`;
  const topnav = `<nav class="topnav" id="topnav" aria-label="${sections ? "On this page" : "Pages"}"><i class="tb-thumb"></i>` +
    items.map((it) => btn(it)).join("") +
    `<button type="button" class="more" data-more aria-label="Show all pages">+<span>${items.length - 4}</span></button></nav>`;
  const right = `<div class="isl-r"><span class="badge ok"><i></i>open source · Apache-2.0</span>` +
    `<button type="button" class="btn text" id="theme-btn" aria-label="Switch light and night" aria-pressed="false"></button></div>`;
  // phone: the tab bar holds up to 4 pages and a More sheet holds the rest (never a hamburger)
  const tabItems = sections ? items.slice(0, 5) : items.slice(0, 3), rest = sections ? [] : items.slice(3);
  const moreOn = rest.some((it) => it.on) || (!sections && here === "index.html");
  const tabbar = `<nav class="tabbar" aria-label="${sections ? "On this page" : "Pages"}"><i class="tb-thumb"></i>` +
    tabItems.map((it) => `<button type="button" data-href="${it.href}"${it.on ? ' class="on" aria-current="page"' : ""}>${ico(it.icon)}${it.short || it.label}</button>`).join("") +
    (rest.length ? `<button type="button" data-sheet${moreOn ? ' class="on"' : ""} aria-haspopup="dialog">${ico("dots")}More</button>` : "") + `</nav>`;
  const sheet = rest.length ? `<div class="sheet-wrap" hidden><div class="scrim"></div><div class="sheet" role="dialog" aria-modal="true" aria-label="More pages"><div class="grab"></div>` +
    `<div style="display:flex;align-items:center;margin-bottom:4px"><div class="lab" style="flex:1">More pages</div><button type="button" class="xbtn" data-close aria-label="Close">${ico("x")}</button></div>` +
    rest.map((it) => `<a class="mi${it.on ? " on" : ""}" href="${it.href}"${it.on ? ' aria-current="page"' : ""}>${ico(it.icon)}${it.label}</a>`).join("") +
    `<a class="mi${here === "index.html" ? " on" : ""}" href="index.html">${ico("home")}Overview</a><hr>` +
    `<a class="mi" href="${REPO}/blob/main/docs/reference.md">${ico("file")}API${ico("out")}</a><a class="mi" href="${REPO}">${ico("out")}GitHub</a></div></div>` : "";
  screen.insertAdjacentHTML("afterbegin", brand + topnav + right + tabbar + sheet);
  syncThemeButton();
  document.getElementById("theme-btn").onclick = toggleTheme;

  const nav = document.getElementById("topnav"), tb = screen.querySelector(".tabbar"), wrap = screen.querySelector(".sheet-wrap");
  const go = (b) => { const h = b.dataset.href; if (h) { if (h[0] === "#") { document.querySelector(h)?.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }); history.replaceState(null, "", h); } else location.href = h; } };
  nav.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    if (b.matches("[data-more]")) return Omxqua.openTopnav(nav, true);
    go(b);
  });
  nav.addEventListener("pointerenter", () => Omxqua.holdTopnav(nav));
  nav.addEventListener("pointerleave", () => Omxqua.foldTopnavLater(nav));
  tb.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    if (b.matches("[data-sheet]")) { wrap.hidden = false; const s = wrap.querySelector(".sheet"); if (!matchMedia("(prefers-reduced-motion: reduce)").matches) s.animate([{ transform: "translateY(40px)", opacity: 0 }, { transform: "none", opacity: 1 }], { duration: 320, easing: "cubic-bezier(.2,0,0,1)" }); wrap.querySelector("[data-close]").focus(); return; }
    go(b);
  });
  if (wrap) {
    const close = () => { wrap.hidden = true; tb.querySelector("[data-sheet]")?.focus(); };
    wrap.addEventListener("click", (e) => { if (e.target.matches(".scrim, [data-close], [data-close] *")) close(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !wrap.hidden) close(); });
  }

  // place the pill and the thumbs once the fonts are in, and again when the window crosses the phone breakpoint
  const place = () => {
    for (const el of [nav, tb]) { const t = el.firstElementChild, tr = t.style.transition; t.style.transition = "none"; if (el === nav) Omxqua.layoutTopnav(nav, false); else Omxqua.placeThumb(tb); requestAnimationFrame(() => { if (el === nav) Omxqua.placeThumb(nav); t.offsetWidth; t.style.transition = tr; }); }
  };
  (document.fonts ? document.fonts.ready : Promise.resolve()).then(() => { place(); new ResizeObserver(place).observe(screen); });

  if (sections) {  // the thumb follows the section in view
    const targets = items.map((it) => document.querySelector(it.href)).filter(Boolean);
    const io = new IntersectionObserver((es) => {
      const e = es.filter((x) => x.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0]; if (!e) return;
      const i = targets.indexOf(e.target);
      nav.querySelectorAll("button[data-i]").forEach((b) => b.classList.toggle("on", +b.dataset.i === i));
      [...tb.querySelectorAll("button")].forEach((b, k) => b.classList.toggle("on", k === i));
      Omxqua.layoutTopnav(nav, true); Omxqua.placeThumb(tb);
    }, { rootMargin: "-30% 0px -60% 0px" });
    targets.forEach((t) => io.observe(t));
  }

  const main = document.querySelector("main");
  if (main && !sections && !main.querySelector(".foot-links"))
    main.insertAdjacentHTML("beforeend", `<p class="foot-links">` +
      `<a href="story.html">The story</a> · <a href="index.html">Overview</a> · ` +
      `<a href="visualize/omxcore_pipeline.html">Kernel visualizer ↗</a> · ` +  // copied in by pages.yml
      `<a href="${REPO}/blob/main/docs/reference.md">API ↗</a> · <a href="${REPO}">GitHub ↗</a> · ` +
      `<a href="${REPO}/blob/main/docs/omxcore-mojo-api.md">Mojo API ↗</a> · <a href="${REPO}/blob/main/CHANGELOG.md">Changelog ↗</a> · ` +
      `omxcore is Apache-2.0</p>`);
}

// colour token as a value (charts are SVG and need real colours)
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// the omxqua ink or surface that reads best (WCAG contrast) on a data colour, for a label drawn over a bar
function textOn(fill) {
  const cx = (textOn.cx ||= document.createElement("canvas").getContext("2d", { willReadFrequently: true }));
  const lum = (c) => { cx.clearRect(0, 0, 1, 1); cx.fillStyle = c; cx.fillRect(0, 0, 1, 1); const d = cx.getImageData(0, 0, 1, 1).data;
    return [0, 1, 2].map((i) => { const v = d[i] / 255; return v <= .03928 ? v / 12.92 : Math.pow((v + .055) / 1.055, 2.4); }).reduce((a, v, i) => a + v * [.2126, .7152, .0722][i], 0); };
  const ratio = (a, b) => (Math.max(a, b) + .05) / (Math.min(a, b) + .05), f = lum(fill), ink = cssVar("--oq-ink"), paper = cssVar("--oq-surface-1");
  return ratio(f, lum(ink)) >= ratio(f, lum(paper)) ? ink : paper;
}

const tipEl = () => document.querySelector(".tip") || document.body.appendChild(Object.assign(document.createElement("div"), { className: "tip", hidden: true }));
function hover(el, html) {
  el.addEventListener("mousemove", (e) => {
    const t = tipEl();
    t.innerHTML = typeof html === "function" ? html() : html;
    t.hidden = false;
    t.style.left = Math.min(e.clientX + 14, innerWidth - t.offsetWidth - 8) + "px";
    t.style.top = e.clientY + 14 + "px";
  });
  el.addEventListener("mouseleave", () => (tipEl().hidden = true));
}

const SVGNS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  return e;
}

// minimal CSV parser: header row, comma separated, no quoted fields (tests/test_docs_site.py checks that no CSV a page reads has a quote)
async function loadCSV(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  const [head, ...rows] = (await res.text()).trim().split(/\r?\n/);
  const cols = head.split(",");
  return rows.map((line) => Object.fromEntries(line.split(",").map((v, i) => [cols[i], v])));
}

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

// show a load error where the chart would be, with the file that failed
function showError(el, path, err) {
  el.innerHTML = `<div class="status err">Couldn't load <a href="${path}">${path}</a> (${err.message}). ` +
    `The source files are in <a href="${REPO}/tree/main/docs/bench">docs/bench</a>.</div>`;
}

const fmtMs = (ms) => (ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : Math.round(ms) + " ms");

document.addEventListener("DOMContentLoaded", renderShell);
