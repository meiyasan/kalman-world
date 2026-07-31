"""FastAPI app over the :mod:`kalman.api` facade — one thin adapter, no logic.

Each route declares a required **scope**; a ``Authorization: Bearer <JWT>``
header is verified by the same :func:`kalman.api.auth.authorize` the MCP server
uses.  FastAPI auto-generates the OpenAPI schema at ``/docs`` — a
machine-discoverable contract for any non-MCP agent.  ``fastapi`` is imported
lazily so the core never depends on it.
"""
from __future__ import annotations


def create_app():
    from fastapi import FastAPI, HTTPException, Depends, Header
    from fastapi.responses import Response
    from .. import api as facade
    from ..api.auth import authorize, AuthError

    app = FastAPI(title="Kalman API", version="1.0",
                  description="Introspect, execute, and author Kalman models.")

    def require(scope: str):
        def dep(authorization: str = Header(default=None)):
            token = None
            if authorization and authorization.lower().startswith("bearer "):
                token = authorization.split(None, 1)[1]
            try:
                authorize(token, scope)
            except AuthError as e:
                raise HTTPException(status_code=401 if not token else 403, detail=str(e))
            return token
        return dep

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    # ---- read ----
    @app.get("/catalog")
    def catalog(_=Depends(require("read"))):
        return facade.catalog()

    @app.get("/describe/{name}")
    def describe(name: str, _=Depends(require("read"))):
        try:
            return facade.describe(name)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/diagnostics/{run_id}")
    def diagnostics(run_id: str, _=Depends(require("read"))):
        try:
            return facade.diagnostics(run_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/plot/{run_id}")
    def plot(run_id: str, _=Depends(require("read"))):
        try:
            return Response(content=facade.render_plot(run_id), media_type="image/png")
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    # ---- send_measurement ----
    @app.post("/filter")
    def filter_(body: dict, tok=Depends(require("send_measurement"))):
        return facade.run_filter(body["spec"], body["data"], smooth=body.get("smooth", False))

    @app.post("/forecast")
    def forecast(body: dict, tok=Depends(require("send_measurement"))):
        m = facade.make_model(body["spec"])
        cone = m.forecast_cone(body["x0"], body.get("P0"), int(body["steps"]))
        return facade.jsonify(cone)

    @app.post("/osse")
    def osse(body: dict, tok=Depends(require("send_measurement"))):
        return facade.run_osse(body["spec"], **body.get("options", {}))

    @app.post("/control")
    def control(body: dict, tok=Depends(require("send_measurement"))):
        return facade.run_control(body["model"], body["controller"], float(body["duration"]))

    @app.post("/fetch")
    def fetch(body: dict, tok=Depends(require("send_measurement"))):
        return facade.fetch(body["provider"], body.get("query", {}), credential=tok)

    # ---- write_models ----
    @app.post("/models")
    def register(body: dict, tok=Depends(require("write_models"))):
        return facade.register_model(body["spec"], body.get("name"), credential=tok)

    @app.post("/scaffold")
    def scaffold(body: dict, tok=Depends(require("write_models"))):
        return {"source": facade.scaffold(body["spec"], body.get("class_name"), credential=tok)}

    return app
