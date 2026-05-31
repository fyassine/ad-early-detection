#!/usr/bin/env bash
set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_DIR="${SCRIPT_DIR}/app"
FRONTEND_DIR="${SCRIPT_DIR}/frontend"

: "${DATA_ROOT:=${REPO_ROOT}/DATA}"
: "${DASHBOARD_CACHE_ROOT:=${SCRIPT_DIR}/.cache}"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
PORT=8050
SERVER_LOG_DIR="${SCRIPT_DIR}/logs/server"
LOG_KEEP=20

# ── colours ──────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; BOLD='\033[1m'; RESET='\033[0m'
else
  GREEN=''; YELLOW=''; RED=''; BOLD=''; RESET=''
fi
ok()   { echo -e "${GREEN}✓${RESET}  $*"; }
step() { echo -e "${YELLOW}→${RESET}  $*"; }
err()  { echo -e "${RED}✗${RESET}  $*" >&2; }

# ── flags ────────────────────────────────────────────────────────────────────
OPT_CLEAN_PY=0
OPT_CLEAN_GELSTM=0
OPT_REBUILD=0
OPT_BG=0
OPT_NO_START=0

usage() {
  cat <<EOF
${BOLD}Usage:${RESET} $(basename "$0") [OPTIONS]

Restart the fMRI dashboard server on port ${PORT}.

${BOLD}Options:${RESET}
  --clean-py      Delete app/__pycache__ and *.pyc files
  --clean-gelstm  Delete .cache/gelstm/predictions_*.pkl
  --rebuild       Rebuild the Vite frontend (npm run build)
  --full          All three of the above
  --bg            Run server in background (nohup); tails log
  --no-start      Kill the old server only, do not start a new one
  -h, --help      Show this help

${BOLD}Examples:${RESET}
  ./restart.sh                   # kill + restart (foreground)
  ./restart.sh --full --bg       # full clean rebuild, background
  ./restart.sh --no-start        # just kill whatever is on port ${PORT}

${BOLD}Environment:${RESET}
  DATA_ROOT              ${DATA_ROOT}
  DASHBOARD_CACHE_ROOT   ${DASHBOARD_CACHE_ROOT}
  VENV                   ${VENV_PYTHON}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean-py)     OPT_CLEAN_PY=1 ;;
    --clean-gelstm) OPT_CLEAN_GELSTM=1 ;;
    --rebuild)      OPT_REBUILD=1 ;;
    --full)         OPT_CLEAN_PY=1; OPT_CLEAN_GELSTM=1; OPT_REBUILD=1 ;;
    --bg)           OPT_BG=1 ;;
    --no-start)     OPT_NO_START=1 ;;
    -h|--help)      usage; exit 0 ;;
    *) err "Unknown option: $1"; usage; exit 1 ;;
  esac
  shift
done

# ── step 1: kill old server ───────────────────────────────────────────────────
step "Looking for processes on port ${PORT}…"

get_port_pids() {
  # ss preferred; lsof as fallback
  if command -v ss &>/dev/null; then
    ss -tlnp 2>/dev/null | { grep ":${PORT} \|:${PORT}$" || true; } | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' || true
  elif command -v lsof &>/dev/null; then
    lsof -ti :"${PORT}" 2>/dev/null || true
  fi
}

PIDS=$(get_port_pids)

if [[ -z "${PIDS}" ]]; then
  ok "No process found on port ${PORT}"
else
  for pid in ${PIDS}; do
    step "Sending SIGTERM to PID ${pid}…"
    kill "${pid}" 2>/dev/null || true
  done

  # wait up to 10 s for port to free
  for i in $(seq 1 10); do
    sleep 1
    REMAINING=$(get_port_pids)
    if [[ -z "${REMAINING}" ]]; then
      ok "Port ${PORT} is free"
      break
    fi
    if [[ "${i}" -eq 10 ]]; then
      step "Still alive after 10 s — sending SIGKILL…"
      for pid in ${REMAINING}; do
        kill -9 "${pid}" 2>/dev/null || true
      done
      sleep 1
      if [[ -n "$(get_port_pids)" ]]; then
        err "Could not free port ${PORT} — aborting"
        exit 1
      fi
      ok "Forcefully killed. Port ${PORT} is free"
    fi
  done
fi

[[ "${OPT_NO_START}" -eq 1 ]] && exit 0

# ── step 2: clear Python caches ──────────────────────────────────────────────
if [[ "${OPT_CLEAN_PY}" -eq 1 ]]; then
  step "Clearing Python bytecode caches…"
  find "${APP_DIR}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
  find "${APP_DIR}" -name "*.pyc" -delete 2>/dev/null || true
  ok "Python caches cleared"
fi

# ── step 3: clear GELSTM predictions cache ───────────────────────────────────
if [[ "${OPT_CLEAN_GELSTM}" -eq 1 ]]; then
  step "Clearing GELSTM predictions cache…"
  GELSTM_CACHE="${DASHBOARD_CACHE_ROOT}/gelstm/predictions_*.pkl"
  # shellcheck disable=SC2086
  FOUND=$(ls ${GELSTM_CACHE} 2>/dev/null | wc -l)
  if [[ "${FOUND}" -gt 0 ]]; then
    # shellcheck disable=SC2086
    rm -f ${GELSTM_CACHE}
    ok "Removed ${FOUND} GELSTM prediction file(s)"
  else
    ok "No GELSTM prediction files found (nothing to clear)"
  fi
fi

# ── step 4: rebuild frontend ─────────────────────────────────────────────────
if [[ "${OPT_REBUILD}" -eq 1 ]]; then
  step "Building Vite frontend…"
  # Find a Node.js >= 16 (required by Vite). Check common locations.
  _find_node() {
    for candidate in \
        /tmp/node/bin/node \
        "${HOME}/.nvm/versions/node/$(ls "${HOME}/.nvm/versions/node/" 2>/dev/null | sort -rV | head -1)/bin/node" \
        "$(which node 2>/dev/null)"; do
      [[ -x "${candidate}" ]] || continue
      local ver; ver=$("${candidate}" -e 'process.exit(parseInt(process.versions.node)<16?1:0)' 2>/dev/null && echo ok || echo old)
      [[ "${ver}" == ok ]] && echo "${candidate}" && return 0
    done
    return 1
  }
  NODE16=$(_find_node) || {
    err "No Node.js >= 16 found. Try: export PATH=/path/to/node16+/bin:\$PATH"
    exit 1
  }
  step "Using Node.js $("${NODE16}" --version) at ${NODE16}"
  if [[ ! -d "${FRONTEND_DIR}/node_modules" ]]; then
    step "node_modules not found — running npm install first…"
    (cd "${FRONTEND_DIR}" && "${NODE16}" "$(dirname "${NODE16}")/../lib/node_modules/npm/bin/npm-cli.js" install 2>/dev/null \
      || PATH="$(dirname "${NODE16}"):$PATH" npm install)
  fi
  (cd "${FRONTEND_DIR}" && "${NODE16}" ./node_modules/vite/bin/vite.js build)
  ok "Frontend built"
fi

# ── step 5: verify venv ───────────────────────────────────────────────────────
if [[ ! -x "${VENV_PYTHON}" ]]; then
  err "Project venv not found at: ${VENV_PYTHON}"
  err "Run: python3 -m venv ${REPO_ROOT}/.venv && pip install -r requirements.txt"
  exit 1
fi

# ── step 6: start server ─────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Starting dashboard…${RESET}"
echo -e "  DATA_ROOT            = ${DATA_ROOT}"
echo -e "  DASHBOARD_CACHE_ROOT = ${DASHBOARD_CACHE_ROOT}"
echo -e "  Python               = ${VENV_PYTHON}"
echo ""

SERVER_CMD=(
  env
  "DATA_ROOT=${DATA_ROOT}"
  "DASHBOARD_CACHE_ROOT=${DASHBOARD_CACHE_ROOT}"
  "${VENV_PYTHON}" -m uvicorn app.main:app
  --host 0.0.0.0
  --port "${PORT}"
)

if [[ "${OPT_BG}" -eq 1 ]]; then
  mkdir -p "${SERVER_LOG_DIR}"
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${SERVER_LOG_DIR}/server_${TS}.log"
  # Rotate: keep only the LOG_KEEP newest server_*.log files.
  ls -1t "${SERVER_LOG_DIR}"/server_*.log 2>/dev/null \
    | tail -n +"$((LOG_KEEP + 1))" | xargs -r rm -f || true
  # Refresh latest.log symlink (relative target so the link survives moves).
  ln -sf "server_${TS}.log" "${SERVER_LOG_DIR}/latest.log"

  cd "${SCRIPT_DIR}"
  nohup "${SERVER_CMD[@]}" >"${LOG_FILE}" 2>&1 &
  BG_PID=$!
  # brief wait to catch immediate startup failures
  sleep 2
  if ! kill -0 "${BG_PID}" 2>/dev/null; then
    err "Server exited immediately — check ${LOG_FILE}"
    tail -20 "${LOG_FILE}"
    exit 1
  fi
  ok "Server started in background (PID ${BG_PID})"
  echo -e "  Log:  ${LOG_FILE}"
  echo -e "  Latest: ${SERVER_LOG_DIR}/latest.log  (tail -f ${SERVER_LOG_DIR}/latest.log)"
  echo -e "  URL:  http://localhost:${PORT}"
  echo ""
  step "Tailing log (Ctrl-C to detach — server keeps running)…"
  tail -f "${LOG_FILE}"
else
  _exit_code=0
  trap '_exit_code=$?' EXIT
  cd "${SCRIPT_DIR}"
  ok "Server starting — press Ctrl-C to stop"
  echo -e "  URL:  http://localhost:${PORT}"
  echo ""
  "${SERVER_CMD[@]}" || true
  if [[ "${_exit_code}" -ne 0 && "${_exit_code}" -ne 130 ]]; then
    err "Server exited with code ${_exit_code}"
  fi
fi
