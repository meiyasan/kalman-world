"""Full-text search for the documentation site.

Doxygen's own search box finds *names*: classes, functions, files, page titles.
It does not look inside the text, so "thermometer" or "cooling rate" finds
nothing even though a page explains both. This script adds the missing half,
after Doxygen has run:

1. it reads every generated page and writes ``search/fulltext-index.js``, one
   entry per section (page, heading, anchor, text);
2. it appends a small script to ``search/search.js`` — a file every page already
   loads — that searches that index as you type and lists the matching sections,
   with the words highlighted, under Doxygen's own results.

Nothing is fetched from the network and no template is changed, so the site
still works from ``file://`` and the doxygen-awesome theme stays untouched.

    python tools/site_search.py var/build/public/docs/<version>
    python tools/site_search.py --newest var/build/public/docs     # the build just made

``make docs`` and ``make pages`` run it for you.
"""
from __future__ import annotations
import html
import json
import os
import re
import sys
from html.parser import HTMLParser

MARKER = "/* kalman-framework full-text search */"

# Pages that are listings, not text: the built-in search already covers them.
SKIP = re.compile(r"(_source\.html|-members\.html|_dep\.html|_incl\.html|^dir_|^globals|^functions|^namespacemembers"
                  r"|^annotated|^classes|^files|^hierarchy|^inherits|^graph_legend|^namespaces|^pages|^search|^index_)")
# File names Doxygen gives the pages it generates from code: classes, namespaces,
# files ("_8" is the encoded dot of an extension), groups, examples.
REFERENCE_PAGE = re.compile(r"^(class|struct|union|namespace|group__|dir_|concept)|_8[a-z]+(-example)?\.html$")
HEADINGS = {"h1", "h2", "h3", "h4"}
DROP = {"script", "style", "svg", "map"}                 # never text
VOID = {"br", "img", "hr", "input", "meta", "link", "area", "col", "wbr"}
MAX_GUIDE = 4000                                         # characters kept per section of a guide page
MAX_REFERENCE = 1200                                     # ... and of a class / namespace / file page


class Page(HTMLParser):
    """Collect the title and the text of ``div.contents``, split at headings."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.sections = [{"heading": "", "anchor": "", "text": []}]
        self._stack = []                                 # (tag, kind) — kind: None, "contents", "title", "heading", "drop"
        self._contents = 0
        self._title = self._heading = self._drop = 0
        self._anchor = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in VOID:
            return
        cls = (a.get("class") or "").split()
        kind = None
        if tag in DROP or "memtitle-permalink" in cls or "permalink" in cls or "lineno" in cls:
            kind = "drop"; self._drop += 1
        elif tag == "div" and "title" in cls and not self._contents:
            kind = "title"; self._title += 1
        elif tag == "div" and "contents" in cls:
            kind = "contents"; self._contents += 1
        elif self._contents and (tag in HEADINGS or (tag == "h2" and "memtitle" in cls)):
            kind = "heading"; self._heading += 1
            anchor = a.get("id") or self._anchor
            self.sections.append({"heading": "", "anchor": anchor or "", "text": []})
            self._anchor = ""
        if self._contents and tag == "a" and a.get("id") and kind != "heading":
            if self._heading:                            # the anchor sits inside the heading
                self.sections[-1]["anchor"] = self.sections[-1]["anchor"] or a["id"]
            else:                                        # ... or just before it
                self._anchor = a["id"]
        self._stack.append((tag, kind))

    def handle_endtag(self, tag):
        # A closing tag with no opener (an auto-closed <p>, say) must not unwind
        # the page: only close back to the matching opener, if there is one.
        if not any(t == tag for t, _ in self._stack):
            return
        while self._stack:
            t, kind = self._stack.pop()
            if kind == "drop": self._drop -= 1
            elif kind == "title": self._title -= 1
            elif kind == "contents": self._contents -= 1
            elif kind == "heading": self._heading -= 1
            if t == tag:
                break

    def handle_data(self, data):
        if self._drop:
            return
        if self._title:
            self.title += data
        elif self._heading:
            self.sections[-1]["heading"] += data
        elif self._contents:
            self.sections[-1]["text"].append(data)


def squeeze(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_index(site: str) -> list[dict]:
    entries = []
    for name in sorted(os.listdir(site)):
        if not name.endswith(".html") or SKIP.search(name):
            continue
        with open(os.path.join(site, name), encoding="utf-8", errors="replace") as fh:
            page = Page()
            try:
                page.feed(fh.read())
            except Exception:                            # a malformed page is skipped, not fatal
                continue
        raw_title = squeeze(page.title) or name
        title = re.sub(r"\s*(Class|Struct|Namespace|File) Reference.*$", "", raw_title)
        # A written page: a Markdown guide, the front page, or a \page in a source
        # file (the libraries' pages). Everything else Doxygen generated from code.
        written = (name.startswith(("md_", "index"))
                   or (title == raw_title and not REFERENCE_PAGE.search(name)))
        # What kind of page: a guide (2) answers a question, a reference page (1)
        # documents one thing, a test (0) is rarely what a reader is looking for.
        if written:
            kind = 2
        elif title.startswith(("tests.", "tests::")) or name.startswith("test") or "_1_1test__" in name:
            kind = 0
        else:
            kind = 1
        for section in page.sections:
            text = squeeze(" ".join(section["text"]))
            heading = squeeze(section["heading"]).rstrip("¶").strip()
            if len(text) < 40 and not heading:
                continue
            limit = MAX_GUIDE if written else MAX_REFERENCE
            entries.append({"p": name, "t": title, "h": heading, "a": section["anchor"],
                            "k": kind, "x": text[:limit]})
    return entries


# The browser side. Plain JavaScript, no library: every word of the query must
# occur in a section; a hit in the page title or the heading counts for more.
SCRIPT = MARKER + r"""
(function () {
  if (window.__kfFullText) return; window.__kfFullText = true;
  var index = null, loading = false, timer = null, panel = null;
  var here = (document.currentScript && document.currentScript.src) || '';
  var base = here.replace(/search\.js(\?.*)?$/, '');

  function fold(s) { return s.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, ''); }
  function esc(s) { return s.replace(/[&<>"]/g, function (c) { return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]; }); }

  function load(done) {
    if (index) return done();
    if (loading) return;
    loading = true;
    var s = document.createElement('script');
    s.src = base + 'fulltext-index.js';
    s.onload = function () {
      index = (window.KF_FULLTEXT || []).map(function (e) {
        return {e: e, title: fold(e.t), heading: fold(e.h), text: fold(e.x)};
      });
      done();
    };
    document.head.appendChild(s);
  }

  function search(query) {
    var words = fold(query).split(/[^a-z0-9_:.+-]+/).filter(function (w) { return w.length > 1; });
    if (!words.length) return [];
    var hits = [];
    for (var i = 0; i < index.length; i++) {
      var d = index[i], score = 0, ok = true;
      for (var j = 0; j < words.length; j++) {
        var w = words[j], inT = d.title.indexOf(w) >= 0, inH = d.heading.indexOf(w) >= 0, at = d.text.indexOf(w);
        if (!inT && !inH && at < 0) { ok = false; break; }
        score += (inT ? 6 : 0) + (inH ? 10 : 0);
        var n = 0; while (at >= 0 && n < 8) { n++; at = d.text.indexOf(w, at + w.length); }
        score += n;
      }
      if (ok && words.length > 1 && d.text.indexOf(words.join(' ')) >= 0) score += 12;   // the exact phrase
      if (ok) {
        if (d.e.k === 2) score += 25;                    // a guide page first
        else if (d.e.k === 0) score *= 0.3;              // a test last
        hits.push({d: d, score: score});
      }
    }
    hits.sort(function (a, b) { return b.score - a.score; });
    return hits.slice(0, 12).map(function (h) { return {e: h.d.e, folded: h.d.text, words: words}; });
  }

  function snippet(hit) {
    var text = hit.e.x, at = -1;
    for (var j = 0; j < hit.words.length && at < 0; j++) at = hit.folded.indexOf(hit.words[j]);
    var from = Math.max(0, at - 60), part = text.slice(from, from + 190);
    var out = esc((from > 0 ? '… ' : '') + part + (from + 190 < text.length ? ' …' : ''));
    hit.words.forEach(function (w) {
      out = out.replace(new RegExp('(' + w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig'), '<mark>$1</mark>');
    });
    return out;
  }

  function ensurePanel() {
    var win = document.getElementById('MSearchResultsWindow');
    if (!win) return null;
    if (!panel) {
      panel = document.createElement('div');
      panel.id = 'kf-fulltext';
      var css = document.createElement('style');
      css.textContent =
        '#MSearchResultsWindow{min-width:min(520px,92vw);max-height:78vh;overflow-y:auto}' +
        '#MSearchResultsWindow.kf-hits #NoMatches{display:none !important}' +
        '#kf-fulltext{font-family:var(--font-family,sans-serif);font-size:13px;padding:6px 10px 10px;' +
        'border-top:1px solid var(--separator-color,#dedede);color:var(--page-foreground-color,#2f4153)}' +
        '#kf-fulltext .kf-head{font-size:11px;text-transform:uppercase;letter-spacing:.04em;opacity:.65;margin:4px 0 6px}' +
        '#kf-fulltext a{display:block;padding:6px 8px;margin:0 -8px;border-radius:var(--border-radius-medium,6px);' +
        'text-decoration:none;color:inherit}' +
        '#kf-fulltext a:hover,#kf-fulltext a:focus{background:var(--side-nav-background,#f3f3f3)}' +
        '#kf-fulltext .kf-title{font-weight:600;color:var(--primary-color,#1779c4)}' +
        '#kf-fulltext .kf-snip{display:block;color:var(--page-foreground-color,#2f4153);opacity:.85;' +
        'line-height:1.35;margin-top:2px;font-weight:400}' +
        '#kf-fulltext mark{background:rgba(255,213,79,.55);color:inherit;border-radius:2px;padding:0 1px}';
      document.head.appendChild(css);
    }
    if (panel.parentNode !== win) win.appendChild(panel);
    return win;
  }

  function render(query) {
    var win = ensurePanel();
    if (!win) return;
    if (!query.trim()) { panel.innerHTML = ''; return; }
    var hits = search(query);
    var html = '<div class="kf-head">In the text' + (hits.length ? '' : ': nothing found') + '</div>';
    hits.forEach(function (h) {
      var where = esc(h.e.t) + (h.e.h && h.e.h !== h.e.t ? ' › ' + esc(h.e.h) : '');
      html += '<a href="' + base + '../' + h.e.p + (h.e.a ? '#' + h.e.a : '') + '"><span class="kf-title">' +
              where + '</span><span class="kf-snip">' + snippet(h) + '</span></a>';
    });
    panel.innerHTML = html;
    win.classList.toggle('kf-hits', hits.length > 0);    // hides Doxygen's "No Matches" line
    if (hits.length) {
      // Show the window even when no name matched. Doxygen places it under the
      // search box only when it has results of its own, so do the same here.
      win.style.display = 'block';
      var box = document.getElementById('MSearchField');
      if (box && !win.style.top && win.offsetParent) {
        var b = box.getBoundingClientRect(), p = win.offsetParent.getBoundingClientRect();
        win.style.top = Math.round(b.bottom - p.top + 8) + 'px';
        win.style.left = '10px';
      }
    }
  }

  // Doxygen rebuilds its search box after the page has loaded, which drops any
  // listener put on the element itself; so listen on the document instead.
  function isBox(t) { return t && t.id === 'MSearchField'; }
  function go(box) {
    clearTimeout(timer);
    timer = setTimeout(function () { load(function () { render(box.value); }); }, 320);
  }
  document.addEventListener('input', function (ev) { if (isBox(ev.target)) go(ev.target); }, true);
  document.addEventListener('keyup', function (ev) { if (isBox(ev.target)) go(ev.target); }, true);
  document.addEventListener('focusin', function (ev) {
    if (!isBox(ev.target)) return;
    ev.target.setAttribute('placeholder', 'Search names and text');
    load(function () {});
  }, true);
})();
"""


def install(site: str) -> tuple[int, int]:
    search_dir = os.path.join(site, "search")
    search_js = os.path.join(search_dir, "search.js")
    if not os.path.isfile(search_js):
        sys.exit(f"{site} is not a Doxygen site with the search engine on (no search/search.js).")

    entries = build_index(site)
    payload = "window.KF_FULLTEXT = " + json.dumps(entries, ensure_ascii=False, separators=(",", ":")) + ";\n"
    with open(os.path.join(search_dir, "fulltext-index.js"), "w", encoding="utf-8") as fh:
        fh.write(payload)

    with open(search_js, encoding="utf-8") as fh:
        source = fh.read()
    # Idempotent: replace an earlier copy, and keep what other tools appended after it.
    before, found, after = source.partition(MARKER)
    tail = after.split("\n/* kalman-framework", 1) if found else [""]
    kept = "\n\n/* kalman-framework" + tail[1].rstrip() if len(tail) > 1 else ""
    source = before.rstrip() + kept + "\n\n" + SCRIPT
    with open(search_js, "w", encoding="utf-8") as fh:
        fh.write(source)
    tidy_navigation(site)
    return len(entries), len({e["p"] for e in entries}), len(payload.encode("utf-8"))


def tidy_navigation(site: str, title: str = "Introduction"):
    """Name the front page's entry in the side navigation.

    Doxygen puts the project name at the root of the tree and, when the README's
    first heading differs from it, the front page under it as a second entry
    saying nearly the same thing; that entry becomes ``title``. When the two are
    the same ("Kalman World"), Doxygen makes no second entry and nothing changes."""
    path = os.path.join(site, "navtreedata.js")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    fixed = re.sub(r'(\[\s*"[^"]*",\s*"index\.html",\s*\[\s*\[\s*")[^"]*(",\s*"index\.html")',
                   lambda m: m.group(1) + title + m.group(2), source, count=1)
    if fixed != source:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(fixed)


def newest(root: str) -> str:
    """The most recently built version under ``root`` (a directory of sites)."""
    sites = [os.path.join(root, d) for d in os.listdir(root)
             if os.path.isfile(os.path.join(root, d, "search", "search.js"))
             and not os.path.islink(os.path.join(root, d))]
    if not sites:
        sys.exit(f"no Doxygen site with a search engine under {root}.")
    return max(sites, key=lambda d: os.path.getmtime(os.path.join(d, "index.html")))


def main():
    args = sys.argv[1:]
    pick_newest = "--newest" in args
    args = [a for a in args if a != "--newest"]
    if len(args) != 1 or args[0] in ("-h", "--help"):
        sys.exit(__doc__)
    site = args[0]
    if not os.path.isdir(site):
        sys.exit(f"{site} is not a directory.")
    if pick_newest:
        site = newest(site)
    count, pages, size = install(site)
    print(f"full-text search: {count} sections of {pages} pages indexed, {size / 1e6:.1f} MB "
          f"({os.path.join(site, 'search', 'fulltext-index.js')})")


if __name__ == "__main__":
    main()
