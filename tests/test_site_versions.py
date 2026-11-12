"""Version switching on the documentation site (tools/site_versions.py)."""
import json
import os
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

import site_search
import site_versions


def build(root, name, mtime, search=True):
    os.makedirs(os.path.join(root, name, "search"))
    index = os.path.join(root, name, "index.html")
    open(index, "w").close()
    os.utime(index, (mtime, mtime))
    if search:
        with open(os.path.join(root, name, "search", "search.js"), "w") as fh:
            fh.write("var searchBox = 1;\n")


class TestSiteVersions(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        for i, name in enumerate(["1.0.2", "1.0.10", "1.0.9", "v2.0", "1.0.2-3-gabc1234", "05f5b82", "latest"]):
            build(self.root, name, 1000 + i)
        os.makedirs(os.path.join(self.root, "not-a-site"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_releases_come_first_by_number_then_development_builds_by_date(self):
        listing = site_versions.versions(self.root)
        self.assertEqual([v["name"] for v in listing],
                         ["v2.0", "1.0.10", "1.0.9", "1.0.2", "05f5b82", "1.0.2-3-gabc1234"])
        self.assertEqual([v["kind"] for v in listing], ["release"] * 4 + ["dev"] * 2)

    def test_install_writes_both_lists_and_one_copy_of_the_script(self):
        site_versions.install(self.root)
        site_versions.install(self.root)
        with open(os.path.join(self.root, "versions.json")) as fh:
            self.assertEqual(json.load(fh)[0], {"name": "v2.0", "kind": "release"})
        with open(os.path.join(self.root, "metadata.txt")) as fh:
            listed = fh.read().split()
        self.assertEqual(len(listed), 6)
        self.assertNotIn("latest", listed)
        with open(os.path.join(self.root, "1.0.9", "search", "search.js")) as fh:
            source = fh.read()
        self.assertTrue(source.startswith("var searchBox = 1;"))
        self.assertEqual(source.count(site_versions.MARKER), 1)

    def test_the_search_and_version_scripts_coexist_in_either_order(self):
        site = os.path.join(self.root, "1.0.9")
        for first, second in ((site_search.install, lambda s: site_versions.install(self.root)),
                              (lambda s: site_versions.install(self.root), site_search.install)):
            for _ in range(2):
                first(site)
                second(site)
            with open(os.path.join(site, "search", "search.js")) as fh:
                source = fh.read()
            self.assertEqual(source.count(site_versions.MARKER), 1)
            self.assertEqual(source.count(site_search.MARKER), 1)
            self.assertTrue(source.startswith("var searchBox = 1;"))

    def test_prune_keeps_the_releases_and_the_newest_development_build(self):
        gone = site_versions.prune_dev(self.root)
        self.assertEqual(gone, ["1.0.2-3-gabc1234"])
        self.assertEqual([v["name"] for v in site_versions.versions(self.root)],
                         ["v2.0", "1.0.10", "1.0.9", "1.0.2", "05f5b82"])

    def test_an_earlier_artifact_adds_only_the_versions_that_are_missing(self):
        archive = os.path.join(self.root, "pages.zip")
        with zipfile.ZipFile(archive, "w") as z:
            for version in ("0.9.0", "1.0.2", "latest"):
                z.writestr(f"public/{version}/index.html", f"old {version}")
                z.writestr(f"public/{version}/search/search.js", "var searchBox = 1;\n")
            z.writestr("public/index.html", "redirect")
            z.writestr("public/metadata.txt", "0.9.0\n")
        added = site_versions.merge_archive(archive, self.root)
        self.assertEqual(added, ["0.9.0"])
        with open(os.path.join(self.root, "0.9.0", "index.html")) as fh:
            self.assertEqual(fh.read(), "old 0.9.0")
        self.assertEqual(os.path.getsize(os.path.join(self.root, "1.0.2", "index.html")), 0, "kept, not replaced")

    def test_restore_outside_ci_does_nothing(self):
        saved = {k: os.environ.pop(k, None) for k in ("CI_API_V4_URL", "CI_PROJECT_ID", "CI_JOB_TOKEN")}
        try:
            self.assertEqual(site_versions.restore(self.root), [])
        finally:
            os.environ.update({k: v for k, v in saved.items() if v is not None})


if __name__ == "__main__":
    unittest.main()
