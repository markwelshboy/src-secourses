#!/usr/bin/env bash
set -Eeuo pipefail

SUPERVISOR_CONFIG="${CAPTIONER_SUPERVISOR_CONFIG:-/etc/supervisor/conf.d/ultimate-captioner.conf}"
INTERVAL="${TELEGRAM_NAG_INTERVAL_SECONDS:-1800}"

if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || (( INTERVAL < 60 )); then
  echo "[captioner-nag] Invalid TELEGRAM_NAG_INTERVAL_SECONDS=${INTERVAL}; using 1800" >&2
  INTERVAL=1800
fi

while true; do
  sleep "$INTERVAL"

  APP_STATUS="$(supervisorctl -c "$SUPERVISOR_CONFIG" status captioner 2>/dev/null || echo 'captioner UNKNOWN')"
  UPTIME="$(uptime -p 2>/dev/null || echo 'uptime unknown')"
  GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ', ' - || true)"
  [[ -n "$GPU" ]] || GPU="unknown GPU"

  telegram-notify "💸 Pod is still running — remember to stop it when finished.
App: ${APP_STATUS}
Uptime: ${UPTIME}
GPU: ${GPU}" || true
done
