#!/usr/bin/env bash
set -Eeuo pipefail

SUPERVISOR_CONFIG="${CAPTIONER_SUPERVISOR_CONFIG:-/etc/supervisor/conf.d/ultimate-captioner.conf}"
APP_DIR="${CAPTIONER_WORKSPACE_DIR:-/workspace/Ultimate_Image_Captioner_Pro}"
LOG_FILE="${CAPTIONER_LOG_FILE:-/workspace/logs/captioner.log}"
PYTHON_BIN="${CAPTIONER_PYTHON:-/opt/venv/bin/python}"
PROGRAM_NAME="captioner"

supervisor() {
  supervisorctl -c "$SUPERVISOR_CONFIG" "$@"
}

status() {
  supervisor status "$PROGRAM_NAME"
}

services() {
  supervisor status
}

logs() {
  local lines="${1:-200}"
  mkdir -p "$(dirname "$LOG_FILE")"
  touch "$LOG_FILE"
  tail -n "$lines" -F "$LOG_FILE"
}

service_logs() {
  local name="${1:?service name required}"
  local lines="${2:-200}"
  local file="/workspace/logs/${name}.log"
  touch "$file"
  tail -n "$lines" -F "$file"
}

doctor() {
  "$PYTHON_BIN" - "$APP_DIR" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

app = Path(sys.argv[1])
qwen = app / "model_files_qwen3_vl3_8b_instruct"
nested = app / "Ultimate_Image_Captioner_Pro"
nested_qwen = nested / "model_files_qwen3_vl3_8b_instruct"

print(f"Application: {app}")
print(f"app.py:      {'OK' if (app / 'app.py').is_file() else 'MISSING'}")
print(f"Python:      {sys.executable}")

try:
    import torch
    print(f"CUDA:        {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'})")
except Exception as exc:
    print(f"CUDA check:  ERROR: {exc}")

processor_markers = (
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.json",
)


def report_model_dir(label: str, path: Path) -> None:
    print(f"\n{label}: {path}")
    if not path.is_dir():
        print("  MISSING")
        return
    files = sorted(p.name for p in path.iterdir() if p.is_file())
    print(f"  top-level files: {len(files)}")
    print(f"  processor markers: {[name for name in processor_markers if (path / name).is_file()] or 'NONE'}")
    for name in files[:20]:
        print(f"  - {name}")
    if len(files) > 20:
        print(f"  ... {len(files) - 20} more")


report_model_dir("Expected Qwen directory", qwen)
report_model_dir("Nested Qwen directory", nested_qwen)

if nested.is_dir():
    print("\nWARNING: Nested download layout detected.")
    print("Run: captionerctl repair-download-layout")

if qwen.is_dir():
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(qwen, trust_remote_code=True, local_files_only=True)
        print("\nAutoProcessor local load: OK")
    except Exception as exc:
        print(f"\nAutoProcessor local load: FAILED: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
PY
}

repair_download_layout() {
  local nested="$APP_DIR/Ultimate_Image_Captioner_Pro"
  if [[ ! -d "$nested" ]]; then
    echo "[captionerctl] No nested download directory found at $nested"
    return 0
  fi

  echo "[captionerctl] Merging nested downloader output into $APP_DIR"
  echo "[captionerctl] Existing application files will not be overwritten."
  rsync -a --ignore-existing --info=progress2 "$nested/" "$APP_DIR/"
  echo
  echo "[captionerctl] Merge complete. The nested directory has been retained for safety:"
  echo "  $nested"
  echo "[captionerctl] Run 'captionerctl doctor', then remove it manually after verification."
}

usage() {
  cat <<'EOF_USAGE'
Usage: captionerctl COMMAND

Commands:
  status                         Show app status
  services                       Show app, bootstrap, and nag services
  start                          Start the app
  stop                           Stop the app
  restart                        Restart the app
  logs [LINES]                   Follow the persistent app log (default 200 lines)
  bootstrap-logs [LINES]         Follow automatic download/bootstrap log
  nag-logs [LINES]               Follow Telegram nag log
  download                       Run model download/verification manually
  telegram-test [MESSAGE]        Send a Telegram test message
  doctor                         Check CUDA and local model/processor files
  repair-download-layout         Merge an accidentally nested downloader target
EOF_USAGE
}

command="${1:-status}"
shift || true

case "$command" in
  status) status ;;
  services) services ;;
  start|stop|restart) supervisor "$command" "$PROGRAM_NAME" ;;
  logs) logs "${1:-200}" ;;
  bootstrap-logs) service_logs captioner-bootstrap "${1:-200}" ;;
  nag-logs) service_logs captioner-nag "${1:-200}" ;;
  download) download-captioner-models ;;
  telegram-test) telegram-notify "${*:-Test message from Ultimate Image Captioner}" ;;
  doctor) doctor ;;
  repair-download-layout|repair) repair_download_layout ;;
  help|-h|--help) usage ;;
  *)
    echo "Unknown command: $command" >&2
    usage >&2
    exit 2
    ;;
esac
