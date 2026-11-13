#!/usr/bin/env bash
# The Kalman servers as a background service on this machine.
#
#   tools/serve.sh start|stop|restart|status|logs|run
#
# What runs is chosen by BACKEND:
#   python  (default) the REST API of kalman.rest, every model and every verb
#   cxx     kalman-serve, the C++ server (C++ models; see docs/18_Net.md)
#   docker  the rest and mcp services of docker-compose.yml
#
# HOST and PORT say where it listens (127.0.0.1:8000). With MCP=1 the MCP server
# is started beside it over HTTP (MCP_TRANSPORT, sse by default, on MCP_PORT,
# 8765), as a client of that REST API; without it MCP clients start their own
# over stdio (make mcp). PYTHON names the interpreter.
#
# The process ids are kept in var/run and the output in var/log. `run` stays in
# the foreground. A service started here does not come back after a reboot.
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD
RUN=$ROOT/var/run
LOG=$ROOT/var/log
BACKEND=${BACKEND:-python}
HOST=${HOST:-${KALMAN_REST_HOST:-127.0.0.1}}
PORT=${PORT:-${KALMAN_REST_PORT:-8000}}
MCP=${MCP:-0}
MCP_PORT=${MCP_PORT:-${KALMAN_MCP_PORT:-8765}}
MCP_TRANSPORT=${MCP_TRANSPORT:-sse}
URL="http://$HOST:$PORT"

say()  { printf '%s\n' "$*"; }
fail() { printf 'serve: %s\n' "$*" >&2; exit 1; }

# The interpreter itself, not a launcher such as "conda run": the service is
# started and stopped by its process id.
python_path() {
    local py=${PYTHON:-}
    if [ -z "$py" ] || [ "$py" = "python3" ]; then
        local name; name=$(basename "$ROOT")
        for dir in "$ROOT/.venv" "$HOME/.conda/envs/$name" "${CONDA_ROOT:-/nonexistent}/envs/$name" \
                   "$HOME/.conda/envs/${ENVNAME:-}" "${CONDA_ROOT:-/nonexistent}/envs/${ENVNAME:-}"; do
            if [ -x "$dir/bin/python" ]; then py="$dir/bin/python"; break; fi
        done
    fi
    py=$(${py:-python3} -c 'import sys; print(sys.executable)' 2>/dev/null) || return 1
    printf '%s' "$py"
}

alive() {   # NAME: true when its recorded process still runs
    local pid; pid=$(cat "$RUN/$1.pid" 2>/dev/null) || return 1
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

healthy() { curl -fs -m 2 "${1:-$URL}/healthz" >/dev/null 2>&1; }

listener() {   # PORT: "command pid" of whoever listens on it, or nothing
    lsof -nP -iTCP:"$1" -sTCP:LISTEN 2>/dev/null | awk 'NR == 2 { print $1 " (pid " $2 ")" }'
}

launch() {   # NAME COMMAND...: start detached from this terminal, record the process id
    local name=$1; shift
    mkdir -p "$RUN" "$LOG"
    printf '\n=== %s: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG/$name.log"
    # A session of its own, so closing the terminal or ending make does not stop it.
    nohup "$PY" -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' "$@" \
        >> "$LOG/$name.log" 2>&1 < /dev/null &
    echo $! > "$RUN/$name.pid"
}

where() {   # NAME: the address it was started on (it may differ from what is asked for now)
    cat "$RUN/$1.url" 2>/dev/null || echo "an unknown address"
}

halt() {   # NAME
    local name=$1 pid
    if ! alive "$name"; then rm -f "$RUN/$name.pid" "$RUN/$name.url"; return 1; fi
    pid=$(cat "$RUN/$name.pid")
    kill "$pid" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.25
    done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
    rm -f "$RUN/$name.pid" "$RUN/$name.url"
    return 0
}

rest_command() {   # the command of the REST server, in REST_CMD
    case $BACKEND in
        python)
            "$PY" -c 'import kalman.rest, uvicorn, fastapi' 2>/dev/null \
                || fail "$PY cannot import kalman.rest with FastAPI and uvicorn (pip install -e '.[rest]'), or give PYTHON=."
            REST_CMD=("$PY" -m kalman.rest) ;;
        cxx)
            local bin
            for bin in "$ROOT/var/install/bin/kalman-serve" "$ROOT/var/build/bin/kalman-serve"; do
                [ -x "$bin" ] && break
            done
            [ -x "$bin" ] || fail "kalman-serve is not built (make build install)."
            REST_CMD=("$bin" --rest --host "$HOST" --port "$PORT") ;;
        *) fail "BACKEND is python, cxx or docker, not '$BACKEND'." ;;
    esac
}

start() {
    if [ "$BACKEND" = docker ]; then
        docker compose up -d rest mcp || exit 1
        say "REST API   http://localhost:8000   (docker)"
        say "MCP server http://localhost:8765   (docker)"
        return
    fi
    PY=$(python_path) || fail "no Python found: give PYTHON=/path/to/python."
    if alive rest; then
        say "REST API   $(where rest)   already running (pid $(cat "$RUN/rest.pid"))"
        [ "$(where rest)" = "$URL" ] || fail "it runs at $(where rest), not at $URL: make serve-stop first to move it."
    else
        local held; held=$(listener "$PORT")
        [ -n "$held" ] && fail "port $PORT is taken by $held. Stop it, or give another: make serve PORT=8001"
        rest_command
        KALMAN_REST_HOST=$HOST KALMAN_REST_PORT=$PORT launch rest "${REST_CMD[@]}"
        echo "$URL" > "$RUN/rest.url"
        for _ in $(seq 1 80); do
            healthy && break
            alive rest || break
            sleep 0.25
        done
        if ! healthy; then
            halt rest
            tail -n 15 "$LOG/rest.log" >&2
            fail "the REST API did not come up; the lines above are the end of var/log/rest.log."
        fi
        say "REST API   $URL   started (pid $(cat "$RUN/rest.pid"), $BACKEND, log var/log/rest.log)"
    fi
    if [ "$MCP" = 1 ]; then
        if alive mcp; then
            say "MCP server $(where mcp)   already running (pid $(cat "$RUN/mcp.pid"))"
        else
            local held; held=$(listener "$MCP_PORT")
            [ -n "$held" ] && fail "port $MCP_PORT is taken by $held. Give another: make serve MCP=1 MCP_PORT=8766"
            "$PY" -c 'import mcp' 2>/dev/null || fail "$PY has no MCP SDK (make mcp installs it)."
            KALMAN_MCP_TRANSPORT=$MCP_TRANSPORT KALMAN_MCP_HOST=$HOST KALMAN_MCP_PORT=$MCP_PORT KALMAN_API_URL=$URL \
                launch mcp "$PY" -m kalman.mcp
            echo "http://$HOST:$MCP_PORT" > "$RUN/mcp.url"
            for _ in $(seq 1 40); do
                [ -n "$(listener "$MCP_PORT")" ] && break
                alive mcp || break
                sleep 0.25
            done
            if [ -z "$(listener "$MCP_PORT")" ]; then
                halt mcp
                tail -n 15 "$LOG/mcp.log" >&2
                fail "the MCP server did not come up; the lines above are the end of var/log/mcp.log."
            fi
            say "MCP server http://$HOST:$MCP_PORT   started (pid $(cat "$RUN/mcp.pid"), $MCP_TRANSPORT, log var/log/mcp.log)"
        fi
    fi
}

stop() {
    if [ "$BACKEND" = docker ]; then docker compose stop rest mcp; return; fi
    local any=0
    if halt mcp; then say "MCP server stopped"; any=1; fi
    if halt rest; then say "REST API   stopped"; any=1; fi
    [ $any = 1 ] || say "nothing was running"
}

status() {
    if [ "$BACKEND" = docker ]; then docker compose ps rest mcp; return; fi
    local code=0 held
    if alive rest; then
        if healthy "$(where rest)"; then say "REST API   $(where rest)   running (pid $(cat "$RUN/rest.pid")), healthy"
        else say "REST API   $(where rest)   running (pid $(cat "$RUN/rest.pid")), but /healthz does not answer"; code=1; fi
    else
        held=$(listener "$PORT")
        say "REST API   not running${held:+ (port $PORT is held by $held, not started here)}"
        code=3
    fi
    if alive mcp; then say "MCP server $(where mcp)   running (pid $(cat "$RUN/mcp.pid"))"
    else say "MCP server not running (clients start their own over stdio: make mcp)"; fi
    return $code
}

case ${1:-} in
    start)   start ;;
    stop)    stop ;;
    restart) stop; start ;;
    status)  status ;;
    logs)    if [ "$BACKEND" = docker ]; then docker compose logs -f rest mcp
             else mkdir -p "$LOG"; touch "$LOG/rest.log"; tail -n 40 -f "$LOG"/rest.log $( [ -f "$LOG/mcp.log" ] && echo "$LOG/mcp.log" ); fi ;;
    run)     PY=$(python_path) || fail "no Python found: give PYTHON=/path/to/python."
             [ "$BACKEND" = docker ] && exec docker compose up rest mcp
             rest_command
             KALMAN_REST_HOST=$HOST KALMAN_REST_PORT=$PORT exec "${REST_CMD[@]}" ;;
    *)       say "usage: tools/serve.sh start|stop|restart|status|logs|run   (BACKEND=python|cxx|docker HOST= PORT= MCP=1)"; exit 2 ;;
esac
