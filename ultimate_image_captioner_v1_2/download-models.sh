#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${CAPTIONER_PYTHON:-/opt/venv/bin/python}"
DOWNLOADER="${CAPTIONER_DOWNLOADER:-/workspace/HF_model_downloader.py}"
DOWNLOADER_SOURCE="${CAPTIONER_DOWNLOADER_SOURCE:-/opt/ultimate-image-captioner-tools/HF_model_downloader.py}"
APP_DIR="${CAPTIONER_WORKSPACE_DIR:-/workspace/Ultimate_Image_Captioner_Pro}"
STARTED_AT="$(date +%s)"

notify_finished() {
  local rc=$?
  trap - EXIT
  local elapsed=$(( $(date +%s) - STARTED_AT ))
  if (( rc == 0 )); then
    telegram-notify "✅ Model download/verification finished successfully in ${elapsed}s."
  else
    telegram-notify "❌ Model download failed after ${elapsed}s with exit code ${rc}. Check the pod logs."
  fi
  exit "$rc"
}
trap notify_finished EXIT

mkdir -p /workspace "$APP_DIR"

if [[ ! -f "$DOWNLOADER" ]]; then
  if [[ ! -f "$DOWNLOADER_SOURCE" ]]; then
    echo "[download-captioner-models] Downloader is missing from the image: $DOWNLOADER_SOURCE" >&2
    exit 1
  fi
  install -m 0644 "$DOWNLOADER_SOURCE" "$DOWNLOADER"
fi

export HF_HOME="${HF_HOME:-/workspace}"

# The downloader's default target is relative to its own file location.
# Keeping it at /workspace/HF_model_downloader.py therefore targets:
#   /workspace/Ultimate_Image_Captioner_Pro
cd /workspace

telegram-notify "⬇️ Model download/verification started."
echo "[download-captioner-models] Downloader: $DOWNLOADER"
echo "[download-captioner-models] Target:     $APP_DIR"
"$PYTHON_BIN" "$DOWNLOADER"
