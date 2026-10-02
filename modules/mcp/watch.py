"""Tell connected MCP clients when the shared registry changes.

A client that subscribed to one of our resources gets a
``notifications/resources/updated`` for it when somebody — this server, the
REST API, another agent, a Unity scene — registers a model, loads a plugin,
sets up a source or pushes measurements. The client then reads the resource
again; the notification itself carries no registry content, so it needs no
scope check of its own.

Both ways to subscribe are served, because both are in use:

  - ``resources/subscribe`` (MCP up to 2025-11-25): the watcher keeps the
    subscribed sessions per URI and notifies each one;
  - ``subscriptions/listen`` (MCP 2026-07-28): the watcher publishes to the
    SDK's subscription bus, which owns the streams.

  ============================  =======================================
  registry event                resource reported as updated
  ============================  =======================================
  ``model.*``, ``plugin.*``     ``catalog://models``
  ``source.*``                  ``catalog://sources``
  ``measurement.received``      ``catalog://sources``
  every event                   ``registry://events``
  ============================  =======================================

The watcher polls ``backend.events`` — the registry file in-process, or the
REST API's ``GET /events`` — because the change may come from any process. It
is one task per server however many sessions are open: the SDK enters the
lifespan once per session on stdio and SSE, and once per server on streamable
HTTP.

A client that does neither calls ``list_events`` instead.
"""
from __future__ import annotations
import asyncio
import os
import sys
from contextlib import asynccontextmanager

MODELS, SOURCES, EVENTS = "catalog://models", "catalog://sources", "registry://events"

#: resources/subscribe connections kept per resource; the oldest go first.
MAX_SUBSCRIBERS = 1024


def uris_for(event_type: str) -> set:
    """The resources a registry event of this type makes stale."""
    family = event_type.split(".", 1)[0]
    out = {EVENTS}
    if family in ("model", "plugin"):
        out.add(MODELS)
    elif family in ("source", "measurement"):
        out.add(SOURCES)
    return out


class RegistryWatcher:

    def __init__(self, backend, bus, *, token: str | None = None, poll: float | None = None):
        self.backend, self.bus, self.token = backend, bus, token
        self.poll = float(os.environ.get("KALMAN_EVENTS_POLL", "1.0")) if poll is None else poll
        self._users = 0
        self._task: asyncio.Task | None = None
        # resources/subscribe: uri -> {connection key -> session}. The SDK makes a
        # new session object for every request, all on the same connection, so a
        # weak reference would die with the request; one is kept per connection
        # and dropped when a send to it fails.
        self._subscribers: dict = {}

    def attach(self, server) -> bool:
        """Serve ``resources/subscribe`` / ``unsubscribe`` on ``server``, for
        clients older than ``subscriptions/listen``. Registering the handlers
        is also what makes the server advertise ``resources.subscribe``."""
        lowlevel = getattr(server, "_lowlevel_server", None)
        if lowlevel is None or not hasattr(lowlevel, "add_request_handler"):
            return False
        import mcp.types as types

        def key(session):
            return id(getattr(session, "_connection", session))

        async def subscribe(ctx, params):
            held = self._subscribers.setdefault(str(params.uri), {})
            held[key(ctx.session)] = ctx.session
            while len(held) > MAX_SUBSCRIBERS:           # oldest first
                held.pop(next(iter(held)))
            return types.EmptyResult()

        async def unsubscribe(ctx, params):
            self._subscribers.get(str(params.uri), {}).pop(key(ctx.session), None)
            return types.EmptyResult()

        lowlevel.add_request_handler("resources/subscribe", types.SubscribeRequestParams, subscribe)
        lowlevel.add_request_handler("resources/unsubscribe", types.UnsubscribeRequestParams,
                                     unsubscribe)
        return True

    async def _notify(self, uri: str):
        from mcp.server.subscriptions import ResourceUpdated
        await self.bus.publish(ResourceUpdated(uri=uri))        # subscriptions/listen
        for k, session in list(self._subscribers.get(uri, {}).items()):   # resources/subscribe
            try:
                await session.send_resource_updated(uri)
            except Exception:                    # the connection is gone
                self._subscribers[uri].pop(k, None)

    @asynccontextmanager
    async def lifespan(self, _app=None):
        self._users += 1
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())
        try:
            yield {}
        finally:
            self._users -= 1
            if self._users == 0 and self._task is not None:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
                self._task = None

    async def _events(self, since: int) -> dict:
        if self.backend.kind == "local":
            # Our own registry file. No scope check: nothing read here leaves
            # the process, only "this resource changed".
            return await asyncio.to_thread(self.backend.api.events, since, 100)
        return await asyncio.to_thread(self.backend.events, since, 100, self.token)

    async def _run(self):
        from ..api.auth import AuthError
        cursor, delay = None, self.poll
        while True:
            try:
                if cursor is None:                       # start from now, not from history
                    cursor = (await self._events(-1))["last"]
                res = await self._events(cursor)
                stale = set()
                for e in res["events"]:
                    stale |= uris_for(e["type"])
                cursor = res["last"]
                for uri in sorted(stale):
                    await self._notify(uri)
                delay = self.poll
            except asyncio.CancelledError:
                raise
            except AuthError as e:
                # Without a token that may read events there is nothing to watch.
                print(f"kalman.mcp: change notifications are off ({e}) Set KALMAN_MCP_TOKEN "
                      "to a token with the read scope.", file=sys.stderr)
                return
            except Exception:                            # the API is down: wait, then look again
                delay = min(delay * 2, 30.0)
            await asyncio.sleep(delay)
