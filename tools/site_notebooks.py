"""Publish the notebooks on the documentation site, with their interactive players.

A notebook viewed on GitLab or GitHub is rendered without JavaScript: the
viewer removes scripts from the saved outputs, so the play/scrub players that
``model.plot(..., interactive=True)`` saves show nothing there. A web page
served from the documentation site has no such limit. This script converts
every notebook under ``share/notebooks`` to a standalone HTML page, with its
saved outputs as they are (nothing is executed), and writes an index:

    <site>/notebooks/index.html
    <site>/notebooks/00_Kalman101.html ...

    python tools/site_notebooks.py var/build/public/docs/<version>
    python tools/site_notebooks.py --newest var/build/public/docs

``make docs`` runs it after Doxygen. It needs ``nbconvert``; without it the
step is skipped with a notice and the rest of the site is unaffected.
"""
from __future__ import annotations
import html
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = os.path.join(REPO, "share", "notebooks")

INDEX = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Notebooks</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 52rem; margin: 3rem auto; padding: 0 1.2rem; color: #2f4153; line-height: 1.5; }}
  h1 {{ font-size: 1.7rem; }}
  a {{ color: #1779c4; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  li {{ margin: 0.9rem 0; list-style: none; }}
  ul {{ padding: 0; }}
  .what {{ display: block; opacity: 0.8; font-size: 0.93rem; }}
  .tag {{ font-size: 0.75rem; border: 1px solid #1779c4; color: #1779c4; border-radius: 3px;
          padding: 0 0.35rem; margin-left: 0.5rem; vertical-align: middle; }}
  @media (prefers-color-scheme: dark) {{ body {{ background: #1c1d1f; color: #d2dbde; }} }}
</style></head><body>
<p><a href="../index.html">&larr; documentation</a></p>
<h1>Notebooks</h1>
<p>Each page is a notebook as it was last run, figures and interactive players included.
The files themselves are in <code>share/notebooks</code>.</p>
<ul>
{items}
</ul>
</body></html>
"""


def describe(path: str) -> tuple[str, str, bool]:
    """(title, first sentence, has an interactive player) of a notebook."""
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    title, intro, interactive = os.path.basename(path), "", False
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and title.endswith(".ipynb"):
            m = re.search(r"^#\s+(.+)$", source, re.M)
            if m:
                title = m.group(1).strip()
                rest = source[m.end():].strip().split("\n\n")[0]
                rest = re.sub(r"[*_`$]|\[([^\]]*)\]\([^)]*\)", r"\1", rest)
                intro = re.split(r"(?<=[.!?])\s", " ".join(rest.split()))[0][:220]
        for out in cell.get("outputs", []):
            if "<script" in "".join(out.get("data", {}).get("text/html", [])):
                interactive = True
    return title, intro, interactive


def publish(site: str) -> int:
    try:
        from nbconvert import HTMLExporter
    except ImportError:
        print("notebooks: nbconvert is not installed -- the notebook pages are skipped")
        return 0
    out = os.path.join(site, "notebooks")
    os.makedirs(out, exist_ok=True)
    exporter = HTMLExporter()
    items = []
    for name in sorted(f for f in os.listdir(NOTEBOOKS) if f.endswith(".ipynb")):
        path = os.path.join(NOTEBOOKS, name)
        body, _ = exporter.from_filename(path)
        page = name[:-len(".ipynb")] + ".html"
        with open(os.path.join(out, page), "w", encoding="utf-8") as fh:
            fh.write(body)
        title, intro, interactive = describe(path)
        tag = '<span class="tag">interactive</span>' if interactive else ""
        items.append(f'  <li><a href="{html.escape(page)}">{html.escape(title)}</a>{tag}'
                     f'<span class="what">{html.escape(intro)}</span></li>')
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(INDEX.format(items="\n".join(items)))
    return len(items)


def newest(root: str) -> str:
    sites = [os.path.join(root, d) for d in os.listdir(root)
             if os.path.isfile(os.path.join(root, d, "index.html"))
             and not os.path.islink(os.path.join(root, d))]
    if not sites:
        sys.exit(f"no documentation site under {root}.")
    return max(sites, key=lambda d: os.path.getmtime(os.path.join(d, "index.html")))


def main():
    args = sys.argv[1:]
    pick = "--newest" in args
    args = [a for a in args if a != "--newest"]
    if len(args) != 1 or args[0] in ("-h", "--help"):
        sys.exit(__doc__)
    site = newest(args[0]) if pick else args[0]
    if not os.path.isdir(site):
        sys.exit(f"{site} is not a directory.")
    count = publish(site)
    if count:
        print(f"notebooks: {count} pages written to {os.path.join(site, 'notebooks')}")


if __name__ == "__main__":
    main()
