"""The documentation site's full-text search index (tools/site_search.py).

The index is built from the pages Doxygen generated; these tests feed the
builder a small hand-written site, so they need neither Doxygen nor a build.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

import site_search

PAGE = """<html><head><title>site: {title}</title><script>var hidden = "thermometer in a script";</script></head>
<body><div class="header"><div class="headertitle"><div class="title">{title}</div></div></div>
<div class="contents">
<p>Coffee cools toward the room temperature.<p>An unclosed paragraph, then a stray closer.</p></p>
<h2 class="doxsection"><a class="anchor" id="physics"></a>The physics</h2>
<p>The filter learns the cooling rate from the thermometer alone.</p>
<a class="anchor" id="code"></a><h2>The model</h2>
<div class="fragment"><span class="lineno">   12</span>out(0) = room;</div>
</div><div id="nav">navigation text that is not content</div></body></html>"""


class TestSiteSearchIndex(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.site = self.tmp.name
        os.mkdir(os.path.join(self.site, "search"))
        with open(os.path.join(self.site, "search", "search.js"), "w") as fh:
            fh.write("var searchBox = 1;\n")
        self.write("md_docs_guide.html", "A guide")
        self.write("class_kalman_1_1_model.html", "Kalman::Model Class Reference")
        self.write("classtests_1_1test__x.html", "tests.test_x.TestX Class Reference")
        self.write("thing_8py_source.html", "a source listing")
        self.write("globals.html", "a listing")
        self.write("libkalmanapps.html", "libKalmanApps — the application models")
        self.write("_storm_8h.html", "Storm.h File Reference")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, title):
        with open(os.path.join(self.site, name), "w", encoding="utf-8") as fh:
            fh.write(PAGE.format(title=title))

    def test_sections_carry_their_heading_anchor_and_text(self):
        entries = [e for e in site_search.build_index(self.site) if e["p"] == "md_docs_guide.html"]
        by_heading = {e["h"]: e for e in entries}
        self.assertEqual(set(by_heading), {"", "The physics", "The model"})
        self.assertEqual(by_heading["The physics"]["a"], "physics")       # anchor inside the heading
        self.assertEqual(by_heading["The model"]["a"], "code")            # anchor just before it
        self.assertIn("cooling rate from the thermometer", by_heading["The physics"]["x"])
        self.assertIn("stray closer", by_heading[""]["x"], "a stray </p> does not end the page")
        text = " ".join(e["x"] for e in entries)
        self.assertNotIn("in a script", text)
        self.assertNotIn("navigation text", text)
        self.assertNotIn("12", by_heading["The model"]["x"], "line numbers are not text")

    def test_listings_are_left_out_and_pages_are_ranked_by_kind(self):
        entries = site_search.build_index(self.site)
        pages = {e["p"]: e for e in entries}
        self.assertNotIn("thing_8py_source.html", pages)
        self.assertNotIn("globals.html", pages)
        self.assertEqual(pages["md_docs_guide.html"]["k"], 2)             # a guide
        self.assertEqual(pages["class_kalman_1_1_model.html"]["k"], 1)    # a reference page
        self.assertEqual(pages["classtests_1_1test__x.html"]["k"], 0)     # a test
        self.assertEqual(pages["class_kalman_1_1_model.html"]["t"], "Kalman::Model")
        self.assertEqual(pages["libkalmanapps.html"]["k"], 2)             # a \page written in a header
        self.assertEqual(pages["_storm_8h.html"]["k"], 1)                 # a file page

    def test_install_writes_the_index_and_is_idempotent(self):
        first = site_search.install(self.site)
        second = site_search.install(self.site)
        self.assertEqual(first, second)
        with open(os.path.join(self.site, "search", "search.js")) as fh:
            source = fh.read()
        self.assertTrue(source.startswith("var searchBox = 1;"), "Doxygen's own script is kept")
        self.assertEqual(source.count(site_search.MARKER), 1, "the search script is installed once")
        with open(os.path.join(self.site, "search", "fulltext-index.js"), encoding="utf-8") as fh:
            payload = fh.read()
        entries = json.loads(payload[len("window.KF_FULLTEXT = "):].rstrip().rstrip(";"))
        self.assertEqual(len(entries), first[0])

    def test_newest_picks_the_latest_build(self):
        root = tempfile.mkdtemp()
        for i, version in enumerate(("old", "new")):
            d = os.path.join(root, version, "search")
            os.makedirs(d)
            open(os.path.join(d, "search.js"), "w").close()
            index = os.path.join(root, version, "index.html")
            open(index, "w").close()
            os.utime(index, (1000 + i, 1000 + i))
        self.assertEqual(os.path.basename(site_search.newest(root)), "new")


if __name__ == "__main__":
    unittest.main()
