#!/usr/bin/env bash
set -Eeuo pipefail

SUPERVISOR_CONFIG="${CAPTIONER_SUPERVISOR_CONFIG:-/etc/supervisor/conf.d/ultimate-captioner.conf}"
SERVER_PORT="${GRADIO_SERVER_PORT:-7861}"
READY_TIMEOUT="${CAPTIONER_APP_READY_TIMEOUT:-300}"
AUTO_DOWNLOAD="${CAPTIONER_AUTO_DOWNLOAD_MODELS:-true}"
RESTART_AFTER_DOWNLOAD="${CAPTIONER_RESTART_AFTER_MODEL_DOWNLOAD:-true}"

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

wait_for_app() {
  local deadline=$((SECONDS + READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 3 "http://127.0.0.1:${SERVER_PORT}/" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

if wait_for_app; then
  telegram-notify "🟢 Pod started and Ultimate Image Captioner is up on port ${SERVER_PORT}." || true
else
  telegram-notify "🔴 Pod started, but Ultimate Image Captioner did not become ready within ${READY_TIMEOUT}s. Check /workspace/logs/captioner.log." || true
  exit 1
fi

if ! is_true "$AUTO_DOWNLOAD"; then
  echo "[captioner-bootstrap] Automatic model download disabled"
  exit 0
fi

set +e
download-captioner-models
DOWNLOAD_RC=$?
set -e

if (( DOWNLOAD_RC != 0 )); then
  echo "[captioner-bootstrap] Model download failed with rc=${DOWNLOAD_RC}" >&2
  exit "$DOWNLOAD_RC"
fi

if is_true "$RESTART_AFTER_DOWNLOAD"; then
  echo "[captioner-bootstrap] Restarting app after model download"
  supervisorctl -c "$SUPERVISOR_CONFIG" restart captioner
  if wait_for_app; then
    telegram-notify "🔄 Model download finished; Ultimate Image Captioner restarted and is ready." || true
  else
    telegram-notify "⚠️ Model download finished, but the app did not become ready after restart. Check /workspace/logs/captioner.log." || true
    exit 1
  fi
fi
