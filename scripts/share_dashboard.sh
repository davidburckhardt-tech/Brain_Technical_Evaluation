#!/usr/bin/env bash
# Rebuilds the dashboard and exposes it over a temporary public ngrok tunnel,
# so it can be opened from any device (phone, tablet, another laptop) to
# record the screen - useful when the Brain unit itself is terminal-only,
# with no GUI/browser to view results in.
#
# Requires ngrok installed and authenticated (`ngrok config add-authtoken
# <token>` once, from a free https://ngrok.com account) - without that,
# `ngrok http` fails with an auth error.
#
# Ctrl+C stops both the local server and the tunnel.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
VENV_DIR="${PROJECT_ROOT}/.venv"
DASHBOARD_DIR="${PROJECT_ROOT}/dashboard"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

PORT="${DASHBOARD_PORT:-8080}"
LOG_DIR="${PROJECT_ROOT}/results"
HTTP_LOG="${LOG_DIR}/dashboard_http.log"
NGROK_LOG="${LOG_DIR}/dashboard_ngrok.log"
mkdir -p "${LOG_DIR}"

if ! command -v ngrok >/dev/null 2>&1; then
  echo "Error: ngrok is not installed. Install it with 'brew install ngrok', then" >&2
  echo "run 'ngrok config add-authtoken <token>' once (free account at https://ngrok.com)." >&2
  exit 1
fi

echo "==> Rebuilding dashboard from current results..."
"${VENV_DIR}/bin/python3" "${PROJECT_ROOT}/scripts/build_dashboard.py"

echo "==> Serving ${DASHBOARD_DIR} on http://localhost:${PORT} ..."
(cd "${DASHBOARD_DIR}" && python3 -m http.server "${PORT}" >"${HTTP_LOG}" 2>&1) &
HTTP_PID=$!

echo "==> Starting ngrok tunnel to port ${PORT}..."
ngrok http "${PORT}" --log=stdout >"${NGROK_LOG}" 2>&1 &
NGROK_PID=$!

cleanup() {
  echo ""
  echo "==> Stopping tunnel and local server..."
  kill -TERM "${NGROK_PID}" 2>/dev/null || true
  kill -TERM "${HTTP_PID}" 2>/dev/null || true
  wait "${NGROK_PID}" 2>/dev/null || true
  wait "${HTTP_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Waiting for ngrok's public URL..."
PUBLIC_URL=""
for i in $(seq 1 30); do
  PUBLIC_URL="$(curl -sf http://127.0.0.1:4040/api/tunnels 2>/dev/null \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['tunnels'][0]['public_url'])" 2>/dev/null || true)"
  if [[ -n "${PUBLIC_URL}" ]]; then
    break
  fi
  sleep 1
done

if [[ -z "${PUBLIC_URL}" ]]; then
  echo "Error: ngrok did not report a public URL in time. Check ${NGROK_LOG}" >&2
  echo "(a common cause is a missing/invalid authtoken - see the note at the top of this script)." >&2
  exit 1
fi

echo ""
echo "=================================================================="
echo " Dashboard available at: ${PUBLIC_URL}"
echo " Open that URL from any device to view/record the results."
echo " Press Ctrl+C here to stop sharing."
echo "=================================================================="
echo ""

wait "${NGROK_PID}"
