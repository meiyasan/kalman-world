"""FastAPI app over the :mod:`kalman.api` facade — one thin adapter, no logic.

Each route declares a required **scope**; a ``Authorization: Bearer <JWT>``
header is verified by the same :func:`kalman.api.auth.authorize` the MCP server
uses.  FastAPI auto-generates the OpenAPI schema at ``/docs`` — a
machine-discoverable contract for any non-MCP agent.  ``fastapi`` is imported
lazily so the core never depends on it.

This is the registry's front door: models, plugins and measurement sources are
registered here, everybody lists them here, and ``GET /events/stream``
announces each change as it happens. The MCP server can run as a client of this
API (``KALMAN_API_URL``), so agents and HTTP clients such as Unity share one
registry and one set of runs.

Errors map to status codes: 400 a bad request, 401 no or bad token (or a wrong
password), 403 a token without the scope (or not the owner), 404 an unknown
name or run, 409 a name taken by a different definition, 429 too many failed
logins.
"""
import asyncio
import json
import os
import time


def _cors_origins() -> list[str]:
    raw = os.environ.get("KALMAN_CORS_ORIGINS", "")
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app():
    from fastapi import Body, FastAPI, HTTPException, Depends, Header, Query, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    from starlette.concurrency import run_in_threadpool
    from .. import api as facade
    from ..api import store
    from ..api import auth
    from ..api.auth import authorize, AuthError, enabled

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app):
        # Plugins recorded by any process load here too, before the first request.
        await run_in_threadpool(facade._sync_plugins)
        yield

    app = FastAPI(title="Kalman API", version="1.1", lifespan=lifespan,
                  description="Introspect, execute, author and share Kalman models.")

    # Browsers (a Unity WebGL build, a web dashboard) need CORS; native clients
    # do not. Off unless KALMAN_CORS_ORIGINS lists the allowed origins ("*" for
    # any). The token travels in a header, never in a cookie, so credentials
    # mode stays off and "*" cannot leak a session.
    origins = _cors_origins()
    if origins:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                           allow_methods=["GET", "POST", "DELETE"],
                           allow_headers=["Authorization", "Content-Type", "Last-Event-ID"])

    # ---- errors -> status codes, once for every route ----
    def _err(status, headers=None):
        async def handler(request, exc):
            msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            return JSONResponse(status_code=status, content={"detail": str(msg)},
                                headers=headers)
        return handler

    app.add_exception_handler(facade.ConflictError, _err(409))
    app.add_exception_handler(KeyError, _err(404))
    app.add_exception_handler(ValueError, _err(400))
    app.add_exception_handler(AuthError, _err(403))   # a valid token, but not allowed
    app.add_exception_handler(auth.InvalidToken, _err(401, {"WWW-Authenticate": "Bearer"}))

    def _token(authorization):
        if authorization and authorization.lower().startswith("bearer "):
            return authorization.split(None, 1)[1].strip()
        return None

    def require(scope: str):
        def dep(authorization: str = Header(default=None)):
            token = _token(authorization)
            try:
                authorize(token, scope)
            except auth.InvalidToken as e:     # none, forged, expired, revoked
                raise HTTPException(status_code=401, detail=str(e),
                                    headers={"WWW-Authenticate": "Bearer"})
            except AuthError as e:             # a good token without the scope
                raise HTTPException(status_code=403, detail=str(e))
            return token
        return dep

    def need(body: dict, *keys):
        missing = [k for k in keys if k not in body]
        if missing:
            raise ValueError(f"missing field(s) {missing} in the request body.")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "auth": enabled()}

    @app.get("/whoami")
    def whoami(authorization: str = Header(default=None)):
        token = _token(authorization)
        try:
            return facade.whoami(token)
        except AuthError as e:
            raise HTTPException(status_code=401, detail=str(e),
                                headers={"WWW-Authenticate": "Bearer"})

    # ---- tokens ----
    @app.post("/token")
    def token(body: dict, request: Request):
        """``{user, password, scopes?, exp_hours?}`` -> ``{access_token, ...}``.
        The scopes must be in the account's grant; see ``make api-user``."""
        need(body, "user", "password")
        try:
            return auth.login(body["user"], body["password"], body.get("scopes"),
                              exp_hours=body.get("exp_hours"),
                              client=request.client.host if request.client else "")
        except auth.RateLimited as e:
            raise HTTPException(status_code=429, detail=str(e),
                                headers={"Retry-After": str(int(e.retry_after) + 1)})
        except auth.BadCredentials as e:
            raise HTTPException(status_code=401, detail=str(e),
                                headers={"WWW-Authenticate": "Bearer"})

    @app.post("/token/revoke")
    def revoke(body: dict = Body(default=None), authorization: str = Header(default=None)):
        """``{}`` revokes the presented token (logout); ``{user}`` revokes every
        token of that user issued until now (the user's own, or admin)."""
        tok = _token(authorization)
        try:
            auth.claims(tok)
        except AuthError as e:
            raise HTTPException(status_code=401, detail=str(e),
                                headers={"WWW-Authenticate": "Bearer"})
        return auth.revoke(tok, user=(body or {}).get("user"))

    # ---- read ----
    @app.get("/catalog")
    def catalog(_=Depends(require("read"))):
        return facade.catalog()

    @app.get("/describe/{name}")
    def describe(name: str, _=Depends(require("read"))):
        return facade.describe(name)

    @app.get("/diagnostics/{run_id}")
    def diagnostics(run_id: str, _=Depends(require("read"))):
        return facade.diagnostics(run_id)

    @app.get("/plot/{run_id}")
    def plot(run_id: str, _=Depends(require("read"))):
        return Response(content=facade.render_plot(run_id), media_type="image/png")

    @app.get("/providers")
    def providers(_=Depends(require("read"))):
        return facade.providers()

    @app.get("/providers/{name}")
    def describe_provider(name: str, _=Depends(require("read"))):
        return facade.describe_provider(name)

    @app.get("/models")
    def models(_=Depends(require("read"))):
        return facade.models()

    @app.get("/plugins")
    def plugins(_=Depends(require("read"))):
        return facade.plugins()

    @app.get("/sources")
    def sources(_=Depends(require("read"))):
        return facade.sources()

    @app.get("/sources/{name}")
    def describe_source(name: str, _=Depends(require("read"))):
        return facade.describe_source(name)

    # ---- subscribe ----
    @app.get("/events")
    def events(since: int = 0, limit: int = 100, _=Depends(require("read"))):
        return facade.events(since, limit)

    @app.get("/events/stream")
    async def events_stream(request: Request, since: int | None = None,
                            timeout: float | None = Query(default=None, ge=0),
                            last_event_id: str | None = Header(default=None),
                            _=Depends(require("read"))):
        """Server-Sent Events: every registry change after ``since`` (or the
        ``Last-Event-ID`` a reconnecting client sends; default: from now).
        ``timeout`` closes the stream after that many seconds."""
        if since is None:
            since = int(last_event_id) if (last_event_id or "").isdigit() \
                else await run_in_threadpool(store.last_event)
        poll = float(os.environ.get("KALMAN_EVENTS_POLL", "0.5"))

        async def gen():
            cursor, start, quiet = since, time.monotonic(), time.monotonic()
            yield "retry: 3000\n\n"
            while True:
                evs = await run_in_threadpool(store.events, cursor, 100)
                for e in evs:
                    cursor = e["id"]
                    yield (f"id: {e['id']}\nevent: {e['type']}\n"
                           f"data: {json.dumps(e)}\n\n")
                if evs:
                    quiet = time.monotonic()
                elif time.monotonic() - quiet > 15:    # keep proxies from closing it
                    yield ": keep-alive\n\n"
                    quiet = time.monotonic()
                if timeout is not None and time.monotonic() - start >= timeout:
                    return
                if await request.is_disconnected():
                    return
                if not evs:
                    await asyncio.sleep(poll)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    # ---- send_measurement ----
    @app.post("/filter")
    def filter_(body: dict, tok=Depends(require("send_measurement"))):
        """``{spec, data}`` or ``{spec, source, query?}``: measure and filter."""
        need(body, "spec")
        return facade.run_filter(body["spec"], body.get("data"), smooth=body.get("smooth", False),
                                 source=body.get("source"), query=body.get("query"),
                                 credential=tok)

    @app.post("/forecast")
    def forecast(body: dict, tok=Depends(require("send_measurement"))):
        need(body, "spec", "x0")
        return facade.forecast(body["spec"], body["x0"], body.get("P0"),
                               int(body.get("steps", 1)))

    @app.post("/osse")
    def osse(body: dict, tok=Depends(require("send_measurement"))):
        need(body, "spec")
        return facade.run_osse(body["spec"], **body.get("options", {}))

    @app.post("/control")
    def control(body: dict, tok=Depends(require("send_measurement"))):
        need(body, "model", "controller", "duration")
        return facade.run_control(body["model"], body["controller"], float(body["duration"]))

    @app.post("/fetch")
    def fetch(body: dict, tok=Depends(require("send_measurement"))):
        """``{provider: {kind, init}}`` or ``{source: <name>}``, plus ``query``."""
        provider = body.get("provider") or ({"source": body["source"]} if "source" in body else None)
        if provider is None:
            raise ValueError("give provider={kind, init} or source=<name>.")
        return facade.fetch(provider, body.get("query", {}), credential=tok)

    @app.get("/sources/{name}/measurements")
    def read_measurements(name: str, since: int = 0, limit: int = 10000,
                          tok=Depends(require("send_measurement"))):
        return facade.fetch({"source": name}, {"since": since, "limit": limit}, credential=tok)

    @app.post("/sources/{name}/measurements")
    def push_measurements(name: str, body: dict, tok=Depends(require("send_measurement"))):
        """``{rows: [[..], ..], time?: [..]}`` into a push source."""
        need(body, "rows")
        return facade.push_measurements(name, body["rows"], body.get("time"), credential=tok)

    # ---- write_models ----
    @app.post("/models")
    def register(body: dict, tok=Depends(require("write_models"))):
        need(body, "spec")
        return facade.register_model(body["spec"], body.get("name"),
                                     replace=bool(body.get("replace", False)), credential=tok)

    @app.delete("/models/{name}")
    def delete_model(name: str, tok=Depends(require("write_models"))):
        return facade.delete_model(name, credential=tok)

    @app.post("/sources")
    def register_source(body: dict, tok=Depends(require("write_models"))):
        need(body, "name", "provider")
        return facade.register_source(body["name"], body["provider"],
                                      replace=bool(body.get("replace", False)), credential=tok)

    @app.delete("/sources/{name}")
    def delete_source(name: str, tok=Depends(require("write_models"))):
        return facade.delete_source(name, credential=tok)

    @app.post("/plugins")
    def load_plugin(body: dict, tok=Depends(require("load_plugins"))):
        need(body, "name")
        return facade.load_plugin(body["name"], credential=tok)

    @app.post("/scaffold")
    def scaffold(body: dict, tok=Depends(require("write_models"))):
        need(body, "spec")
        return {"source": facade.scaffold(body["spec"], body.get("class_name"), credential=tok)}

    return app
