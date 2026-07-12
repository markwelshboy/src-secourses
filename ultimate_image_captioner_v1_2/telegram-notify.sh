#!/usr/bin/env bash
set -uo pipefail

SECRETS_FILE="${TELEGRAM_ENV_FILE:-/root/.secrets/telegram.env}"
if [[ -f "$SECRETS_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$SECRETS_FILE"
fi

BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
CHAT_ID="${TELEGRAM_CHAT_ID:-}"
NAME="${TELEGRAM_NAME:-ultimate-image-captioner}"

if [[ -z "$BOT_TOKEN" || -z "$CHAT_ID" ]]; then
  [[ "${TELEGRAM_DEBUG:-false}" == "true" ]] && \
    echo "[telegram-notify] TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not configured; skipping" >&2
  exit 0
fi

if (( $# > 0 )); then
  MESSAGE="$*"
else
  MESSAGE="$(cat)"
fi

HOST="$(hostname 2>/dev/null || echo unknown)"
TEXT="[${NAME}] ${MESSAGE}
Host: ${HOST}"

if ! curl -fsS --max-time 15 \
  -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
  --data-urlencode "chat_id=${CHAT_ID}" \
  --data-urlencode "text=${TEXT}" \
  -d "disable_web_page_preview=true" \
  >/dev/null; then
  echo "[telegram-notify] Telegram send failed" >&2
  exit 1
fi
