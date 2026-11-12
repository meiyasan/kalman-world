"""Version switching for the documentation site.

Each build of the documentation lands in its own folder under the site root,
named after the version (``1.0.2`` for a release, ``1.0.2-11-g64fefe7`` for a
build between releases). The theme has a version menu, which reads
``metadata.txt`` in the site root. This script makes that menu usable:

1. it writes ``metadata.txt`` (the theme's list) and ``versions.json`` (the same
   list in display order, each entry marked ``release`` or ``dev``);
2. it appends a small script to every version's ``search/search.js`` that
   - lists releases first, newest first, by version number (so 1.0.10 comes
     after 1.0.9, not before 1.0.2), then the builds between releases;
   - marks the newest release ``latest`` and the others ``dev``;
   - on a switch, opens the *same page* in the other version when it exists
     there, and that version's front page otherwise.

With ``--prune-dev`` it first removes every build between releases except the
newest, which is what a published site wants: one entry per release plus the
current development build.

With ``--restore`` it first brings back the versions published earlier. GitLab
Pages serves one deployment per project, so a pipeline that publishes only what
it just built would erase every other version. The earlier ones are taken from
the artifacts of the ``pages`` job of the default branch and of each release
tag, and a version already present is left alone.

    python tools/site_versions.py var/build/public/docs
    python tools/site_versions.py --restore var/build/public/docs       # CI, before the build
    python tools/site_versions.py --prune-dev var/build/public/docs     # CI, after it

``make docs`` and ``make pages`` run it. The menu needs the site to be served
over http (``python -m http.server`` in the site root is enough): a browser
does not let a page opened from ``file://`` read the list.
"""
from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

MARKER = "/* kalman-framework version menu */"
RELEASE = re.compile(r"^v?(\d+)\.(\d+)(?:\.(\d+))?$")

SCRIPT = MARKER + r"""
(function () {
  if (window.__kfVersions) return; window.__kfVersions = true;
  var here = (document.currentScript && document.currentScript.src) || '';
  var root = here.replace(/[^\/]+\/search\/search\.js(\?.*)?$/, '');      // the site root, with its slash
  var list = null, busy = false;

  function current() {
    var rest = location.href.slice(root.length);
    return rest.split('/')[0];
  }
  function page() {                                  // path of this page inside its version
    var rest = location.href.slice(root.length).split(/[?#]/)[0];
    return rest.split('/').slice(1).join('/') || 'index.html';
  }
  function label(v, i) {
    if (v.kind === 'release') return v.name + (i === 0 ? ' (latest)' : '');
    return v.name + ' (dev)';
  }
  function signature(select) {
    return Array.prototype.map.call(select.options, function (o) { return o.value + '=' + o.text; }).join('|');
  }
  function fill() {
    var select = document.getElementById('MSearchVersionSelect');
    if (!select || !list || busy) return;
    var first = list.findIndex(function (v) { return v.kind === 'release'; });
    var want = list.map(function (v, i) { return v.name + '=' + label(v, i - first); }).join('|');
    if (signature(select) === want) return;
    busy = true;
    select.onchange = null;                          // the theme's handler; ours is on the document
    while (select.options.length) select.remove(0);
    var now = current();
    list.forEach(function (v, i) {
      var option = new Option(label(v, i - first), v.name);
      if (v.name === now) option.selected = true;
      select.add(option);
    });
    var box = document.getElementById('MSearchVersion');
    if (box) box.style.display = list.length > 1 ? 'block' : 'none';
    busy = false;
  }
  function go(name) {
    var same = root + name + '/' + page();
    fetch(same, {method: 'HEAD'}).then(function (r) {
      location.href = r.ok ? same + location.hash : root + name + '/index.html';
    }).catch(function () { location.href = root + name + '/index.html'; });
  }

  document.addEventListener('change', function (ev) {
    if (ev.target && ev.target.id === 'MSearchVersionSelect' && list) {
      ev.stopImmediatePropagation();
      go(ev.target.value);
    }
  }, true);

  function start() {
    fetch(root + 'versions.json').then(function (r) { return r.ok ? r.json() : null; }).then(function (data) {
      if (!data || !data.length) return;
      list = data;
      fill();
      // The theme fills the same menu from metadata.txt a moment later, in
      // alphabetical order; put ours back whenever that happens.
      var n = 0, timer = setInterval(function () { fill(); if (++n > 20) clearInterval(timer); }, 300);
    }).catch(function () {});
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
"""


def release_key(name: str):
    m = RELEASE.match(name)
    return tuple(int(g or 0) for g in m.groups()) if m else None


def versions(root: str) -> list[dict]:
    """The built versions under ``root`` in display order: releases, newest
    first by number; then the builds between releases, newest first by date."""
    found = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.islink(path) or not os.path.isfile(os.path.join(path, "index.html")):
            continue
        if name == "latest":                          # a copy of the newest version, not a version
            continue
        found.append((name, release_key(name), os.path.getmtime(os.path.join(path, "index.html"))))
    releases = sorted((f for f in found if f[1]), key=lambda f: f[1], reverse=True)
    dev = sorted((f for f in found if not f[1]), key=lambda f: f[2], reverse=True)
    return ([{"name": n, "kind": "release"} for n, _, _ in releases]
            + [{"name": n, "kind": "dev"} for n, _, _ in dev])


def prune_dev(root: str) -> list[str]:
    """Remove every build between releases except the newest; return what went."""
    dev = [v["name"] for v in versions(root) if v["kind"] == "dev"]
    for name in dev[1:]:
        shutil.rmtree(os.path.join(root, name))
    return dev[1:]


def merge_archive(archive: str, root: str) -> list[str]:
    """Unpack the version folders of a ``pages`` artifact (a zip holding
    ``public/<version>/...``) into ``root``; return the versions added. A
    version that ``root`` already has is not touched."""
    added = []
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory() as tmp:
        names = {n.split("/")[1] for n in z.namelist()
                 if n.startswith("public/") and n.count("/") >= 2 and n.split("/")[1]}
        wanted = sorted(v for v in names if v != "latest" and f"public/{v}/index.html" in z.namelist()
                        and not os.path.exists(os.path.join(root, v)))
        for version in wanted:
            prefix = f"public/{version}/"
            z.extractall(tmp, [n for n in z.namelist() if n.startswith(prefix)])
            shutil.move(os.path.join(tmp, "public", version), os.path.join(root, version))
            added.append(version)
    return added


def restore(root: str) -> list[str]:
    """Bring back the versions earlier pipelines published (GitLab CI only)."""
    env = os.environ
    api, project, token = env.get("CI_API_V4_URL"), env.get("CI_PROJECT_ID"), env.get("CI_JOB_TOKEN")
    if not (api and project and token):
        print("versions: not in GitLab CI -- nothing to restore")
        return []
    refs = [env.get("CI_DEFAULT_BRANCH", "main")]
    try:
        tags = subprocess.run(["git", "tag", "--list"], capture_output=True, text=True, check=True).stdout.split()
        refs += sorted((t for t in tags if release_key(t)), key=release_key, reverse=True)
    except (OSError, subprocess.CalledProcessError):
        pass
    os.makedirs(root, exist_ok=True)
    added = []
    for ref in refs:
        url = f"{api}/projects/{project}/jobs/artifacts/{urllib.request.quote(ref, safe='')}/download?job=pages"
        request = urllib.request.Request(url, headers={"JOB-TOKEN": token})
        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    shutil.copyfileobj(response, tmp)
                tmp.flush()
                added += merge_archive(tmp.name, root)
            except (urllib.error.URLError, zipfile.BadZipFile, OSError) as error:
                print(f"versions: no earlier site for {ref} ({error})")
    return added


def install(root: str) -> list[dict]:
    listing = versions(root)
    with open(os.path.join(root, "metadata.txt"), "w", encoding="utf-8") as fh:
        fh.write("".join(v["name"] + "\n" for v in sorted(listing, key=lambda v: v["name"])))
    with open(os.path.join(root, "versions.json"), "w", encoding="utf-8") as fh:
        json.dump(listing, fh)
    for v in listing:
        search_js = os.path.join(root, v["name"], "search", "search.js")
        if not os.path.isfile(search_js):
            continue                                  # a version built without the search engine
        with open(search_js, encoding="utf-8") as fh:
            source = fh.read()
        if MARKER in source:
            before, _, after = source.partition(MARKER)
            # Replace an earlier copy and keep what other tools appended after it.
            tail = after.split("\n/* kalman-framework", 1)
            source = before.rstrip() + ("\n\n/* kalman-framework" + tail[1] if len(tail) > 1 else "")
        with open(search_js, "w", encoding="utf-8") as fh:
            fh.write(source.rstrip() + "\n\n" + SCRIPT)
    return listing


def main():
    args = sys.argv[1:]
    prune = "--prune-dev" in args
    bring_back = "--restore" in args
    args = [a for a in args if a not in ("--prune-dev", "--restore")]
    if len(args) != 1 or args[0] in ("-h", "--help"):
        sys.exit(__doc__)
    root = args[0]
    if bring_back:
        added = restore(root)
        print(f"versions: restored {len(added)} earlier version(s)" + (": " + ", ".join(added) if added else ""))
        return
    if not os.path.isdir(root):
        sys.exit(f"{root} is not a directory.")
    if prune:
        gone = prune_dev(root)
        if gone:
            print(f"versions: removed {len(gone)} older development build(s): {', '.join(gone)}")
    listing = install(root)
    names = ", ".join(f"{v['name']}{'' if v['kind'] == 'release' else ' (dev)'}" for v in listing)
    print(f"versions: {len(listing)} listed in {os.path.join(root, 'versions.json')}: {names}")


if __name__ == "__main__":
    main()
