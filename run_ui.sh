#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-command launcher for the Rakhshai graphical UI.
#
# Note: this script's messages are in English because most terminals do not
# render right-to-left Persian text correctly. Persian is shown only in the
# web UI (app.py), where it renders properly.
#
# Usage:
#   ./run_ui.sh                 # run on http://127.0.0.1:7860
#   ./run_ui.sh --share         # create a temporary public link
#   ./run_ui.sh --port 8080     # choose a custom port
#   ./run_ui.sh stop            # stop the server and free the port
#   ./run_ui.sh restart         # stop, then start again
#   ./run_ui.sh status          # show whether it is running
#   ./run_ui.sh check           # list optional packages and whether each is installed
#
# Press Ctrl + C while running for a clean shutdown (the port is freed).
# If the port is left busy by a previous instance, it is freed automatically.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
DEFAULT_PORT=7860

# --- Help -----------------------------------------------------------------
print_help() {
  cat <<'EOF'
Rakhshai graphical UI — launcher

  ./run_ui.sh [options]     run the UI
  ./run_ui.sh stop          stop the server and free the port
  ./run_ui.sh restart       stop, then start again
  ./run_ui.sh status        show running status
  ./run_ui.sh check         list optional packages and whether each is installed
  ./run_ui.sh help          show this help

Run options (passed through to app.py):
  --port <n>      choose a port (default 7860)
  --host <addr>   choose a host (default 127.0.0.1)
  --share         create a temporary public link

Press Ctrl + C while running to stop.
EOF
}

# --- Resolve the port from the arguments (if --port was given) ------------
detect_port() {
  local port="$DEFAULT_PORT" prev=""
  for a in "$@"; do
    [ "$prev" = "--port" ] && port="$a"
    case "$a" in --port=*) port="${a#--port=}" ;; esac
    prev="$a"
  done
  printf '%s' "$port"
}

# --- PIDs listening on a given port ---------------------------------------
pids_on_port() {
  lsof -ti "tcp:$1" -sTCP:LISTEN 2>/dev/null || true
}

# --- Free a port (stop a running or suspended instance) -------------------
free_port() {
  local port="$1" pids
  pids="$(pids_on_port "$port")"
  if [ -z "$pids" ]; then
    return 0
  fi

  echo "🧹 Freeing port $port (stopping previous instance: $(echo "$pids" | tr '\n' ' '))…"
  # Resume first in case the process was suspended, then ask it to stop (SIGTERM).
  # shellcheck disable=SC2086
  kill -CONT $pids 2>/dev/null || true
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true

  # Wait a few seconds for it to exit.
  for _ in 1 2 3 4 5 6; do
    [ -z "$(pids_on_port "$port")" ] && break
    sleep 0.4
  done

  # Still alive? Force it (SIGKILL).
  pids="$(pids_on_port "$port")"
  if [ -n "$pids" ]; then
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
    sleep 0.3
  fi

  if [ -z "$(pids_on_port "$port")" ]; then
    echo "✅ Port $port is free."
  else
    echo "⚠️  Could not fully free port $port." >&2
    return 1
  fi
}

# --- Show status ----------------------------------------------------------
show_status() {
  local port="$1" pids
  pids="$(pids_on_port "$port")"
  if [ -n "$pids" ]; then
    echo "🟢 UI is running on port $port (PID: $(echo "$pids" | tr '\n' ' '))."
    echo "   URL: http://127.0.0.1:$port"
  else
    echo "⚪ Nothing is running on port $port."
  fi
}

# --- Check optional/complementary packages --------------------------------
check_deps() {
  if ! command -v "$PY" >/dev/null 2>&1; then
    echo "❌ Python not found. Please install Python 3.10+." >&2
    return 1
  fi
  echo "Optional packages — install only the ones you need:"
  echo ""
  # Each row: import_name|pip_spec|what it enables
  local rows=(
    "sklearn|scikit-learn>=1.2|ML metrics, TF-IDF summarization, document graphs"
    "stanza|stanza>=1.6|advanced Persian NLP: POS, lemma, dependency parsing"
    "faiss|faiss-cpu>=1.7.4|fast vector similarity search"
  )
  local row import rest pip desc name
  for row in "${rows[@]}"; do
    import="${row%%|*}"; rest="${row#*|}"
    pip="${rest%%|*}";   desc="${rest#*|}"
    name="${pip%%>=*}"
    echo "  • ${name} — ${desc}"
    if "$PY" -c "import ${import}" >/dev/null 2>&1; then
      echo "      ✅ installed"
    else
      echo "      ❌ not installed   →  ${PY} -m pip install \"${pip}\""
    fi
  done

  # Stanza also needs its Persian language model.
  if "$PY" -c "import stanza" >/dev/null 2>&1; then
    echo "  • stanza Persian model (fa)"
    if [ -d "${HOME}/stanza_resources/fa" ]; then
      echo "      ✅ installed"
    else
      echo "      ❌ not installed   →  ${PY} -c \"import stanza; stanza.download('fa')\""
    fi
  fi

  echo ""
  echo "Install everything at once:  ${PY} -m pip install -e \".[all]\""
}

# --- Separate the subcommand from the arguments ---------------------------
SUB="start"
case "${1:-}" in
  stop|status|restart) SUB="$1"; shift ;;
  check|deps)          SUB="check"; shift ;;
  start)               SUB="start"; shift ;;
  help|-h|--help)      print_help; exit 0 ;;
esac

PORT="$(detect_port "$@")"

case "$SUB" in
  stop)
    free_port "$PORT"
    exit 0
    ;;
  status)
    show_status "$PORT"
    exit 0
    ;;
  check)
    check_deps
    exit 0
    ;;
  restart)
    free_port "$PORT" || true
    ;;
esac

# --- Check and install prerequisites --------------------------------------
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "❌ Python not found. Please install Python 3.10+." >&2
  exit 1
fi

echo "🔎 Checking prerequisites…"

# Install the package itself (editable) if it is missing.
if ! "$PY" -c "import rakhshai_graph_nlp" >/dev/null 2>&1; then
  echo "📦 Installing the rakhshai-graph-nlp package …"
  "$PY" -m pip install -e . >/dev/null
fi

# Install the UI dependency if it is missing.
if ! "$PY" -c "import gradio" >/dev/null 2>&1; then
  echo "📦 Installing the UI dependency …"
  "$PY" -m pip install "gradio>=4.0" >/dev/null
fi

# If the port is still busy from a previous instance (e.g. suspended with
# Ctrl+Z), free it before starting.
if [ -n "$(pids_on_port "$PORT")" ]; then
  echo "ℹ️  Port $PORT is already in use; stopping the previous instance."
  free_port "$PORT" || true
fi

# --- Run the UI with a clean shutdown -------------------------------------
echo "🚀 Starting the Rakhshai UI …"
echo "   To stop: Ctrl + C   (or from another terminal: ./run_ui.sh stop)"

APP_PID=""
cleanup() {
  trap - INT TERM EXIT
  if [ -n "$APP_PID" ] && kill -0 "$APP_PID" 2>/dev/null; then
    echo ""
    echo "🛑 Shutting down the UI …"
    kill "$APP_PID" 2>/dev/null || true
    wait "$APP_PID" 2>/dev/null || true
  fi
  free_port "$PORT" >/dev/null 2>&1 || true
  echo "👋 UI stopped."
}
trap cleanup INT TERM EXIT

"$PY" app.py "$@" &
APP_PID=$!
wait "$APP_PID"
