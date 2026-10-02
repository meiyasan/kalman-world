"""
The shared registry behind :mod:`kalman.api` — one SQLite file that every
process reading the same state directory sees.

Before this, authored models and loaded plugins lived in a dict inside each
process. The REST server and the MCP server are two processes, so a model
registered through one was invisible to the other, and a restart lost it. Now
the registry is on disk:

  - ``models``        authored KalmanSpecs (name -> spec, dimensions, owner)
  - ``plugins``       compiled model libraries loaded through the API, so every
                      process (and every restart) loads them again
  - ``sources``       named measurement sources an admin or author set up once
                      and everybody fetches by name
  - ``measurements``  rows pushed into a ``push`` source by a device, a
                      simulation or a game engine
  - ``events``        an append-only log of all of the above; ``GET /events``
                      streams it, so clients subscribe instead of polling the
                      catalog

The file is ``$KALMAN_STATE_DIR/registry.sqlite`` (default ``~/.kalman/state``).
WAL mode lets one writer and many readers work at the same time, across
processes and threads. Values are JSON text; nothing here executes code.
"""
from __future__ import annotations
import contextlib
import json
import os
import sqlite3
import threading
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    name    TEXT PRIMARY KEY,
    spec    TEXT NOT NULL,
    info    TEXT NOT NULL,
    owner   TEXT,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS plugins (
    name    TEXT PRIMARY KEY,
    path    TEXT NOT NULL,
    models  TEXT NOT NULL,
    owner   TEXT,
    loaded  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    name    TEXT PRIMARY KEY,
    spec    TEXT NOT NULL,
    owner   TEXT,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS measurements (
    seq      INTEGER PRIMARY KEY AUTOINCREMENT,
    source   TEXT NOT NULL,
    row      TEXT NOT NULL,
    t        REAL,
    owner    TEXT,
    received REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS measurements_by_source ON measurements(source, seq);
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    type    TEXT NOT NULL,
    payload TEXT NOT NULL,
    ts      REAL NOT NULL
);
"""

#: Rows kept per push source, and events kept in the log. Old rows go first.
MAX_ROWS = int(os.environ.get("KALMAN_SOURCE_MAX_ROWS", "100000"))
MAX_EVENTS = int(os.environ.get("KALMAN_MAX_EVENTS", "10000"))


class ConflictError(Exception):
    """A name is already taken by a different definition (HTTP 409)."""


_ready: set = set()
_ready_lock = threading.Lock()


def state_dir() -> str:
    return os.path.abspath(os.path.expanduser(
        os.environ.get("KALMAN_STATE_DIR") or "~/.kalman/state"))


def path() -> str:
    return os.path.join(state_dir(), "registry.sqlite")


@contextlib.contextmanager
def _db():
    """One connection per call: sqlite3 connections must not cross threads, and
    FastAPI runs sync routes on a thread pool. Commits on success."""
    p = path()
    with _ready_lock:
        if p not in _ready:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            con = sqlite3.connect(p, timeout=30)
            try:
                con.execute("PRAGMA journal_mode=WAL")
                con.executescript(_SCHEMA)
                con.commit()
            finally:
                con.close()
            try:
                os.chmod(p, 0o600)            # owner-only, like the credentials file
            except OSError:
                pass
            _ready.add(p)
    con = sqlite3.connect(p, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _emit(con, type_: str, payload: dict) -> int:
    cur = con.execute("INSERT INTO events(type, payload, ts) VALUES (?,?,?)",
                      (type_, json.dumps(payload), time.time()))
    eid = cur.lastrowid
    if MAX_EVENTS > 0:
        con.execute("DELETE FROM events WHERE id <= ?", (eid - MAX_EVENTS,))
    return eid


def emit(type_: str, payload: dict) -> int:
    with _db() as con:
        return _emit(con, type_, payload)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def _model(row) -> dict:
    return {"name": row["name"], "spec": json.loads(row["spec"]), **json.loads(row["info"]),
            "owner": row["owner"], "created": row["created"], "updated": row["updated"]}


def get_model(name: str) -> dict | None:
    with _db() as con:
        row = con.execute("SELECT * FROM models WHERE name=?", (name,)).fetchone()
    return _model(row) if row else None


def model_names() -> list[str]:
    with _db() as con:
        return [r[0] for r in con.execute("SELECT name FROM models ORDER BY name")]


def list_models() -> list[dict]:
    with _db() as con:
        return [_model(r) for r in con.execute("SELECT * FROM models ORDER BY name")]


def put_model(name: str, spec: dict, info: dict, owner: str | None, *,
              replace: bool = False, may_replace=lambda rec: True) -> tuple[dict, str]:
    """Insert or update a model. Returns ``(record, action)`` where action is
    ``created``, ``replaced`` or ``unchanged``.

    The same spec under the same name is a no-op, so a client can register on
    every start. A different spec needs ``replace=True`` and ``may_replace(old)``
    true; otherwise :class:`ConflictError` — a shared registry must not let one
    author silently overwrite another's model."""
    now = time.time()
    with _db() as con:
        row = con.execute("SELECT * FROM models WHERE name=?", (name,)).fetchone()
        if row is not None:
            old = _model(row)
            if _same(old["spec"], spec):
                return old, "unchanged"
            if not replace:
                raise ConflictError(f"model {name!r} already exists with a different spec "
                                    f"(owner {old['owner'] or 'unknown'}); pass replace=true "
                                    "to update it, or choose another name.")
            may_replace(old)                  # raises AuthError when not allowed
            con.execute("UPDATE models SET spec=?, info=?, updated=? WHERE name=?",
                        (json.dumps(spec), json.dumps(info), now, name))
            action = "replaced"
        else:
            con.execute("INSERT INTO models VALUES (?,?,?,?,?,?)",
                        (name, json.dumps(spec), json.dumps(info), owner, now, now))
            action = "created"
        _emit(con, f"model.{action}", {"name": name, "owner": owner, **info})
        row = con.execute("SELECT * FROM models WHERE name=?", (name,)).fetchone()
    return _model(row), action


def delete_model(name: str, *, may_delete=lambda rec: True, by: str | None = None) -> dict:
    with _db() as con:
        row = con.execute("SELECT * FROM models WHERE name=?", (name,)).fetchone()
        if row is None:
            raise KeyError(f"no registered model named {name!r}.")
        rec = _model(row)
        may_delete(rec)
        con.execute("DELETE FROM models WHERE name=?", (name,))
        _emit(con, "model.deleted", {"name": name, "by": by})
    return rec


# ---------------------------------------------------------------------------
# plugins
# ---------------------------------------------------------------------------
def put_plugin(name: str, plugin_path: str, models: list, owner: str | None) -> bool:
    """Record a loaded plugin. Returns True when it was not recorded before."""
    with _db() as con:
        row = con.execute("SELECT models FROM plugins WHERE name=?", (name,)).fetchone()
        if row is not None and sorted(json.loads(row[0])) == sorted(models):
            return False
        con.execute("INSERT OR REPLACE INTO plugins VALUES (?,?,?,?,?)",
                    (name, plugin_path, json.dumps(sorted(models)), owner, time.time()))
        _emit(con, "plugin.loaded", {"name": name, "models": sorted(models), "owner": owner})
    return True


def list_plugins() -> list[dict]:
    with _db() as con:
        return [{"name": r["name"], "path": r["path"], "models": json.loads(r["models"]),
                 "owner": r["owner"], "loaded": r["loaded"]}
                for r in con.execute("SELECT * FROM plugins ORDER BY name")]


# ---------------------------------------------------------------------------
# sources and pushed measurements
# ---------------------------------------------------------------------------
def _source(row) -> dict:
    return {"name": row["name"], **json.loads(row["spec"]), "owner": row["owner"],
            "created": row["created"], "updated": row["updated"]}


def get_source(name: str) -> dict | None:
    with _db() as con:
        row = con.execute("SELECT * FROM sources WHERE name=?", (name,)).fetchone()
    return _source(row) if row else None


def list_sources() -> list[dict]:
    with _db() as con:
        out = []
        for r in con.execute("SELECT * FROM sources ORDER BY name"):
            rec = _source(r)
            if rec.get("kind") == "push":
                n, last = con.execute("SELECT COUNT(*), MAX(seq) FROM measurements "
                                      "WHERE source=?", (rec["name"],)).fetchone()
                rec["rows"], rec["last"] = n, last or 0
            out.append(rec)
        return out


def put_source(name: str, spec: dict, owner: str | None, *, replace: bool = False,
               may_replace=lambda rec: True) -> tuple[dict, str]:
    now = time.time()
    with _db() as con:
        row = con.execute("SELECT * FROM sources WHERE name=?", (name,)).fetchone()
        if row is not None:
            old = _source(row)
            if _same(json.loads(row["spec"]), spec):
                return old, "unchanged"
            if not replace:
                raise ConflictError(f"source {name!r} already exists with a different "
                                    "definition; pass replace=true to update it.")
            may_replace(old)
            con.execute("UPDATE sources SET spec=?, updated=? WHERE name=?",
                        (json.dumps(spec), now, name))
            action = "replaced"
        else:
            con.execute("INSERT INTO sources VALUES (?,?,?,?,?)",
                        (name, json.dumps(spec), owner, now, now))
            action = "created"
        _emit(con, f"source.{action}", {"name": name, "kind": spec.get("kind"), "owner": owner})
        row = con.execute("SELECT * FROM sources WHERE name=?", (name,)).fetchone()
    return _source(row), action


def delete_source(name: str, *, may_delete=lambda rec: True, by: str | None = None) -> dict:
    with _db() as con:
        row = con.execute("SELECT * FROM sources WHERE name=?", (name,)).fetchone()
        if row is None:
            raise KeyError(f"no source named {name!r}.")
        rec = _source(row)
        may_delete(rec)
        con.execute("DELETE FROM sources WHERE name=?", (name,))
        con.execute("DELETE FROM measurements WHERE source=?", (name,))
        _emit(con, "source.deleted", {"name": name, "by": by})
    return rec


def append_measurements(source: str, rows: list, times: list | None,
                        owner: str | None) -> dict:
    now = time.time()
    with _db() as con:
        first = last = None
        for i, row in enumerate(rows):
            t = None if times is None else times[i]
            cur = con.execute("INSERT INTO measurements(source, row, t, owner, received) "
                              "VALUES (?,?,?,?,?)", (source, json.dumps(row), t, owner, now))
            first = first or cur.lastrowid
            last = cur.lastrowid
        if MAX_ROWS > 0:
            con.execute("DELETE FROM measurements WHERE source=? AND seq NOT IN "
                        "(SELECT seq FROM measurements WHERE source=? "
                        " ORDER BY seq DESC LIMIT ?)", (source, source, MAX_ROWS))
        out = {"source": source, "first": first, "last": last, "count": len(rows)}
        _emit(con, "measurement.received", {**out, "owner": owner})
    return out


def read_measurements(source: str, since: int = 0, limit: int = 10000) -> list[dict]:
    with _db() as con:
        return [{"seq": r["seq"], "row": json.loads(r["row"]), "t": r["t"],
                 "received": r["received"]}
                for r in con.execute("SELECT seq, row, t, received FROM measurements "
                                     "WHERE source=? AND seq>? ORDER BY seq LIMIT ?",
                                     (source, int(since), int(limit)))]


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
def events(since: int = 0, limit: int = 100) -> list[dict]:
    with _db() as con:
        return [{"id": r["id"], "type": r["type"], "payload": json.loads(r["payload"]),
                 "ts": r["ts"]}
                for r in con.execute("SELECT * FROM events WHERE id>? ORDER BY id LIMIT ?",
                                     (int(since), int(limit)))]


def last_event() -> int:
    with _db() as con:
        return con.execute("SELECT COALESCE(MAX(id), 0) FROM events").fetchone()[0]
