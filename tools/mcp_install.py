#!/usr/bin/env python3
"""
Register the Kalman MCP server with any MCP-capable client — one command for
all of them.

    python tools/mcp_install.py --list                 # what's installed here
    python tools/mcp_install.py --all                  # configure every detected client
    python tools/mcp_install.py claude-desktop cursor  # or pick some
    python tools/mcp_install.py --all --dry-run        # show the diff, write nothing
    python tools/mcp_install.py --all --remove         # unregister everywhere

Each client keeps its MCP servers in a JSON (or TOML) file under a
client-specific key — ``mcpServers`` for most, ``servers`` for VS Code, a
``[mcp_servers.*]`` table for Codex.  This script knows those locations, edits
only our own entry, and leaves the rest of the file untouched (a timestamped
``.bak`` is written before any change).

Auth: pass ``--token`` (mint one with ``make api-token``) and/or
``--jwt-secret`` to embed credentials in the launch environment; without them
the server runs with auth disabled, which is fine for local use.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import shutil
import sys
from pathlib import Path

SERVER_NAME = "kalman"
REPO = Path(__file__).resolve().parent.parent
HOME = Path.home()
OS = {"Darwin": "mac", "Windows": "win"}.get(platform.system(), "linux")

APPDATA = Path(os.environ.get("APPDATA", HOME / "AppData" / "Roaming"))
SUPPORT = HOME / "Library" / "Application Support"          # macOS
XDG = Path(os.environ.get("XDG_CONFIG_HOME", HOME / ".config"))


def _p(mac: Path, win: Path, linux: Path) -> Path:
    return {"mac": mac, "win": win, "linux": linux}[OS]


# name -> (config path, container key, format)
#   container key "" means the servers live at the document root
CLIENTS: dict[str, tuple[Path, str, str]] = {
    "claude-desktop": (
        _p(SUPPORT / "Claude" / "claude_desktop_config.json",
           APPDATA / "Claude" / "claude_desktop_config.json",
           XDG / "Claude" / "claude_desktop_config.json"),
        "mcpServers", "json"),
    "claude-code": (HOME / ".claude.json", "mcpServers", "json"),
    "cursor": (HOME / ".cursor" / "mcp.json", "mcpServers", "json"),
    "windsurf": (HOME / ".codeium" / "windsurf" / "mcp_config.json", "mcpServers", "json"),
    "vscode": (
        _p(SUPPORT / "Code" / "User" / "mcp.json",
           APPDATA / "Code" / "User" / "mcp.json",
           XDG / "Code" / "User" / "mcp.json"),
        "servers", "json"),
    "vscode-insiders": (
        _p(SUPPORT / "Code - Insiders" / "User" / "mcp.json",
           APPDATA / "Code - Insiders" / "User" / "mcp.json",
           XDG / "Code - Insiders" / "User" / "mcp.json"),
        "servers", "json"),
    "cline": (
        _p(SUPPORT / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev"
           / "settings" / "cline_mcp_settings.json",
           APPDATA / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev"
           / "settings" / "cline_mcp_settings.json",
           XDG / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev"
           / "settings" / "cline_mcp_settings.json"),
        "mcpServers", "json"),
    "gemini-cli": (HOME / ".gemini" / "settings.json", "mcpServers", "json"),
    "codex": (HOME / ".codex" / "config.toml", "mcp_servers", "toml"),
    "opencode": (HOME / ".opencode" / "config.json", "mcpServers", "json"),
    "kiro": (HOME / ".kiro" / "settings" / "mcp.json", "mcpServers", "json"),
    "trae": (HOME / ".trae" / "mcp.json", "mcpServers", "json"),
}

# A project-local config, handy for committing alongside the repo.
PROJECT_TARGETS = {
    "project-vscode": (REPO / ".vscode" / "mcp.json", "servers", "json"),
    "project-mcp": (REPO / ".mcp.json", "mcpServers", "json"),
}


def server_entry(python: str, token: str | None, secret: str | None,
                 transport: str, portable: bool = False) -> dict:
    """The launch spec every client understands.

    ``portable`` emits machine-independent values (``python`` from PATH,
    ``PYTHONPATH="."``) suitable for a config committed to the repo; the
    default embeds absolute paths, which is what a per-user config wants."""
    if portable:
        python = "python"
    env = {"PYTHONPATH": "." if portable else str(REPO)}
    if secret:
        env["KALMAN_JWT_SECRET"] = secret
    if token:
        env["KALMAN_MCP_TOKEN"] = token
    if transport != "stdio":
        env["KALMAN_MCP_TRANSPORT"] = transport
    return {"command": python, "args": ["-m", "kalman.mcp"], "env": env}


# ---------------------------------------------------------------------------
# JSON / TOML editing (surgical — only our own entry is touched)
# ---------------------------------------------------------------------------
def _load(path: Path, fmt: str):
    if not path.exists() or path.stat().st_size == 0:
        return {}
    text = path.read_text()
    if fmt == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise SystemExit(f"! {path} is not valid JSON ({e}); fix or move it first.")
    return text                                   # toml handled as text


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = path.with_suffix(path.suffix + f".bak-{stamp}")
    shutil.copy2(path, dst)
    return dst


def _toml_block(entry: dict, key: str) -> str:
    env = "\n".join(f'{k} = "{v}"' for k, v in entry["env"].items())
    args = ", ".join(f'"{a}"' for a in entry["args"])
    return (f'\n[{key}.{SERVER_NAME}]\n'
            f'command = "{entry["command"]}"\n'
            f'args = [{args}]\n'
            f'[{key}.{SERVER_NAME}.env]\n{env}\n')


def apply(path: Path, key: str, fmt: str, entry: dict | None,
          dry_run: bool) -> str:
    """Add (entry) or remove (entry=None) our server. Returns a status word."""
    if fmt == "toml":
        text = _load(path, fmt) or ""
        marker = f"[{key}.{SERVER_NAME}]"
        present = marker in text
        if entry is None:
            if not present:
                return "absent"
            lines, out, skip = text.splitlines(True), [], False
            for ln in lines:
                if ln.strip().startswith(f"[{key}.{SERVER_NAME}"):
                    skip = True
                elif skip and ln.lstrip().startswith("["):
                    skip = False
                if not skip:
                    out.append(ln)
            new = "".join(out)
        else:
            if present:                            # replace the existing block
                lines, out, skip = text.splitlines(True), [], False
                for ln in lines:
                    if ln.strip().startswith(f"[{key}.{SERVER_NAME}"):
                        skip = True
                    elif skip and ln.lstrip().startswith("["):
                        skip = False
                    if not skip:
                        out.append(ln)
                text = "".join(out)
            new = text.rstrip() + "\n" + _toml_block(entry, key)
        if dry_run:
            return "would-update" if entry else "would-remove"
        path.parent.mkdir(parents=True, exist_ok=True)
        _backup(path)
        path.write_text(new)
        return "updated" if entry else "removed"

    doc = _load(path, fmt)
    if not isinstance(doc, dict):
        raise SystemExit(f"! {path} has an unexpected shape.")
    container = doc if not key else doc.setdefault(key, {})
    if not isinstance(container, dict):
        raise SystemExit(f"! {path}: '{key}' is not an object.")
    if entry is None:
        if SERVER_NAME not in container:
            return "absent"
        if dry_run:
            return "would-remove"
        container.pop(SERVER_NAME)
    else:
        if container.get(SERVER_NAME) == entry:
            return "already-current"
        if dry_run:
            return "would-update"
        container[SERVER_NAME] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup(path)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return "removed" if entry is None else "updated"


# ---------------------------------------------------------------------------
def _detect(targets: dict) -> list[str]:
    """Clients whose config file (or its parent directory) already exists."""
    found = []
    for name, (path, *_rest) in targets.items():
        if path.exists() or path.parent.exists():
            found.append(name)
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="mcp_install", description="Register the Kalman MCP server with MCP clients.")
    ap.add_argument("clients", nargs="*", help="client names (default: --all)")
    ap.add_argument("--all", action="store_true", help="every detected client")
    ap.add_argument("--list", action="store_true", help="list clients and detection status")
    ap.add_argument("--remove", action="store_true", help="unregister instead of register")
    ap.add_argument("--dry-run", action="store_true", help="print what would change")
    ap.add_argument("--python", default=sys.executable, help="interpreter to launch the server")
    ap.add_argument("--token", help="JWT for the server's gated tools (see: make api-token)")
    ap.add_argument("--jwt-secret", help="KALMAN_JWT_SECRET to enable auth in the server")
    ap.add_argument("--transport", default="stdio", choices=("stdio", "sse", "streamable-http"))
    ap.add_argument("--project", action="store_true",
                    help="also write repo-local configs (.vscode/mcp.json, .mcp.json)")
    ap.add_argument("--portable", action="store_true",
                    help="machine-independent paths (python from PATH, PYTHONPATH=.) "
                         "- use for a config committed to the repo")
    args = ap.parse_args(argv)

    targets = dict(CLIENTS)
    if args.project:
        targets.update(PROJECT_TARGETS)

    if args.list:
        print(f"platform: {OS}   repo: {REPO}")
        for name, (path, key, fmt) in sorted(targets.items()):
            state = "installed" if path.exists() else (
                "available" if path.parent.exists() else "not found")
            has = ""
            if path.exists():
                try:
                    doc = _load(path, fmt)
                    present = (SERVER_NAME in (doc.get(key, {}) if key else doc)
                               if fmt == "json" else f"[{key}.{SERVER_NAME}]" in doc)
                    has = "  <- kalman registered" if present else ""
                except SystemExit:
                    has = "  (unreadable)"
            print(f"  {name:18s} {state:10s} {path}{has}")
        return 0

    chosen = args.clients or (_detect(targets) if args.all else [])
    if not chosen:
        ap.error("name at least one client, or pass --all (see --list)")
    unknown = [c for c in chosen if c not in targets]
    if unknown:
        ap.error(f"unknown client(s): {', '.join(unknown)} (see --list)")

    entry = None if args.remove else server_entry(
        args.python, args.token, args.jwt_secret, args.transport, args.portable)

    print(f"{'Removing' if args.remove else 'Registering'} '{SERVER_NAME}'"
          f"{' (dry run)' if args.dry_run else ''} — repo {REPO}")
    rc = 0
    for name in chosen:
        path, key, fmt = targets[name]
        try:
            status = apply(path, key, fmt, entry, args.dry_run)
        except SystemExit as e:
            print(f"  {name:18s} ERROR {e}"); rc = 1; continue
        print(f"  {name:18s} {status:14s} {path}")
    if not args.remove and not args.dry_run:
        print("\nRestart the client(s) to pick up the change. Verify with:\n"
              f"  {args.python} -m kalman.mcp   # should start and wait on stdio")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
