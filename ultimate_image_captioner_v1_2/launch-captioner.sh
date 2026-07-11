#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="${CAPTIONER_WORKSPACE_DIR:-/workspace/Ultimate_Image_Captioner_Pro}"
PYTHON_BIN="${CAPTIONER_PYTHON:-/opt/venv/bin/python}"
SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
SERVER_PORT="${GRADIO_SERVER_PORT:-7861}"

export HF_HOME="${HF_HOME:-/workspace}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

if [[ ! -f "$APP_DIR/app.py" ]]; then
  echo "[launch-captioner] app.py not found at $APP_DIR" >&2
  exit 1
fi

args=(
  "$PYTHON_BIN"
  app.py
  --server-name "$SERVER_NAME"
  --server-port "$SERVER_PORT"
  --no-inbrowser
)

case "${GRADIO_SHARE:-false}" in
  1|true|TRUE|yes|YES)
    args+=(--share)
    ;;
esac

cd "$APP_DIR"
echo "[launch-captioner] ${args[*]} $*"
exec "${args[@]}" "$@"
