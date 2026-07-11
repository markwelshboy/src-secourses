#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_REPO_URL="${RUNTIME_REPO_URL:-https://github.com/markwelshboy/pod-runtime.git}"
RUNTIME_DIR="${RUNTIME_DIR:-/workspace/pod-runtime}"
IMAGE_APP_DIR="${CAPTIONER_IMAGE_DIR:-/opt/Ultimate_Image_Captioner_Pro}"
WORKSPACE_APP_DIR="${CAPTIONER_WORKSPACE_DIR:-/workspace/Ultimate_Image_Captioner_Pro}"
DOWNLOADER_SOURCE="${CAPTIONER_DOWNLOADER_SOURCE:-/opt/ultimate-image-captioner-tools/HF_model_downloader.py}"
DOWNLOADER_TARGET="${CAPTIONER_DOWNLOADER:-/workspace/HF_model_downloader.py}"
SUPERVISOR_CONFIG="${CAPTIONER_SUPERVISOR_CONFIG:-/etc/supervisor/conf.d/ultimate-captioner.conf}"
CAPTIONER_LOG="${CAPTIONER_LOG_FILE:-/workspace/logs/captioner.log}"
TELEGRAM_ENV_FILE="${TELEGRAM_ENV_FILE:-/root/.secrets/telegram.env}"

export POD_RUNTIME_DIR="$RUNTIME_DIR"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TEMP="${TEMP:-$TMPDIR}"
export TMP="${TMP:-$TMPDIR}"
export TELEGRAM_ENV_FILE

log() {
  printf '[captioner-start] %s\n' "$*"
}

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

sync_application() {
  if [[ ! -f "$IMAGE_APP_DIR/app.py" ]]; then
    log "FATAL: baked application not found at $IMAGE_APP_DIR"
    exit 1
  fi

  mkdir -p "$WORKSPACE_APP_DIR"

  # Refresh image-owned application code without deleting persistent state.
  # Preserve a legacy nested downloader target so it can be repaired safely.
  rsync -a --delete \
    --exclude='.git/' \
    --exclude='outputs/' \
    --exclude='presets/' \
    --exclude='model_files_*/' \
    --exclude='.cache/' \
    --exclude='Ultimate_Image_Captioner_Pro/' \
    "$IMAGE_APP_DIR/" "$WORKSPACE_APP_DIR/"

  mkdir -p \
    "$WORKSPACE_APP_DIR/outputs" \
    "$WORKSPACE_APP_DIR/presets" \
    "$WORKSPACE_APP_DIR/model_files_beta_one" \
    "$WORKSPACE_APP_DIR/model_files_qwen3_vl3_8b_instruct" \
    /workspace/logs \
    "$TMPDIR"

  # Seed newly shipped presets while preserving user changes and additions.
  if [[ -d "$IMAGE_APP_DIR/presets" ]]; then
    rsync -a --ignore-existing "$IMAGE_APP_DIR/presets/" "$WORKSPACE_APP_DIR/presets/"
  fi

  if [[ -f "$DOWNLOADER_SOURCE" ]]; then
    install -m 0644 "$DOWNLOADER_SOURCE" "$DOWNLOADER_TARGET"
    log "Installed model downloader at $DOWNLOADER_TARGET"
  else
    log "WARNING: baked model downloader not found at $DOWNLOADER_SOURCE"
  fi

  if [[ -d "$WORKSPACE_APP_DIR/Ultimate_Image_Captioner_Pro" ]]; then
    log "WARNING: nested downloader target detected at $WORKSPACE_APP_DIR/Ultimate_Image_Captioner_Pro"
    log "         Run 'captionerctl repair-download-layout', then 'captionerctl doctor'."
  fi
}

sync_pod_runtime() {
  mkdir -p /workspace

  if [[ -d "$RUNTIME_DIR/.git" ]]; then
    log "Updating pod-runtime in $RUNTIME_DIR"
    git -C "$RUNTIME_DIR" pull --rebase --autostash || true
  else
    log "Cloning pod-runtime into $RUNTIME_DIR"
    rm -rf "$RUNTIME_DIR"
    git clone --depth 1 "$RUNTIME_REPO_URL" "$RUNTIME_DIR" || {
      log "WARNING: pod-runtime clone failed; continuing without shell customizations"
      return 0
    }
  fi

  local tmp=/root/.bashrc.captioner.tmp
  if [[ -f "$RUNTIME_DIR/.bashrc" ]]; then
    cp "$RUNTIME_DIR/.bashrc" "$tmp"
    sed -i "s|REPO_ROOT=<CHANGEME>|REPO_ROOT=\"$RUNTIME_DIR\"|" "$tmp"
    install -m 0644 "$tmp" /root/.bashrc
    rm -f "$tmp"
  fi

  local file
  for file in .bash_functions .bash_aliases .bash_prompt .git-qol.sh; do
    [[ -f "$RUNTIME_DIR/$file" ]] && install -m 0644 "$RUNTIME_DIR/$file" "/root/$file"
  done
}

persist_telegram_environment() {
  mkdir -p "$(dirname "$TELEGRAM_ENV_FILE")"
  umask 077
  {
    printf '# Generated %s\n' "$(date -Is)"
    printf 'export TELEGRAM_BOT_TOKEN=%q\n' "${TELEGRAM_BOT_TOKEN:-}"
    printf 'export TELEGRAM_CHAT_ID=%q\n' "${TELEGRAM_CHAT_ID:-}"
    printf 'export TELEGRAM_NAME=%q\n' "${TELEGRAM_NAME:-ultimate-image-captioner}"
  } > "$TELEGRAM_ENV_FILE"
  chmod 0600 "$TELEGRAM_ENV_FILE"
  umask 022

  if [[ -n "${TELEGRAM_BOT_TOKEN:-}" && -n "${TELEGRAM_CHAT_ID:-}" ]]; then
    log "Telegram notifications enabled as ${TELEGRAM_NAME:-ultimate-image-captioner}"
  else
    log "Telegram notifications disabled; TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not both set"
  fi
}

install_authorized_keys() {
  install -d -m 0700 /root/.ssh

  local key_material=""
  if [[ -n "${SSH_PUBLIC_KEY_B64:-}" ]]; then
    key_material="$(printf '%s' "$SSH_PUBLIC_KEY_B64" | base64 -d)"
  elif [[ -n "${SSH_PUBLIC_KEY_FILE:-}" && -f "${SSH_PUBLIC_KEY_FILE}" ]]; then
    key_material="$(cat "${SSH_PUBLIC_KEY_FILE}")"
  elif [[ -n "${SSH_PUBLIC_KEY:-}" ]]; then
    key_material="$SSH_PUBLIC_KEY"
  elif [[ -n "${RUNPOD_SSH_PUBLIC_KEY:-}" ]]; then
    key_material="$RUNPOD_SSH_PUBLIC_KEY"
  elif [[ -n "${PUBLIC_KEY:-}" ]]; then
    key_material="$PUBLIC_KEY"
  elif [[ -n "${SSH_AUTHORIZED_KEYS:-}" ]]; then
    key_material="$SSH_AUTHORIZED_KEYS"
  fi

  if [[ -n "$key_material" ]]; then
    printf '%s\n' "$key_material" > /root/.ssh/authorized_keys
    chmod 0600 /root/.ssh/authorized_keys
    log "Installed SSH public key material supplied at runtime"
  elif [[ -s /root/.ssh/authorized_keys ]]; then
    chmod 0600 /root/.ssh/authorized_keys
    log "Using existing /root/.ssh/authorized_keys"
  else
    log "WARNING: no SSH public key supplied; SSH will run but key login will not be available"
  fi
}

start_ssh() {
  mkdir -p /run/sshd /var/run/sshd /etc/ssh/sshd_config.d
  ssh-keygen -A
  # Unlock the root account for public-key auth only; password auth remains disabled.
  passwd -d root >/dev/null 2>&1 || true

  cat > /etc/ssh/sshd_config.d/99-ultimate-captioner.conf <<'SSHD'
PermitRootLogin prohibit-password
PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes
UsePAM no
X11Forwarding no
AllowTcpForwarding yes
GatewayPorts no
ClientAliveInterval 60
ClientAliveCountMax 3
SSHD

  /usr/sbin/sshd -D -e &
  SSHD_PID=$!
  export SSHD_PID
  log "OpenSSH started (pid $SSHD_PID)"
}

configure_supervisor() {
  mkdir -p /run /workspace/logs "$(dirname "$SUPERVISOR_CONFIG")"
  touch "$CAPTIONER_LOG"

  cat > "$SUPERVISOR_CONFIG" <<EOF_SUPERVISOR
[unix_http_server]
file=/run/ultimate-captioner-supervisor.sock
chmod=0700

[supervisord]
nodaemon=true
logfile=/workspace/logs/supervisord.log
logfile_maxbytes=20MB
logfile_backups=2
pidfile=/run/ultimate-captioner-supervisord.pid
childlogdir=/workspace/logs

[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///run/ultimate-captioner-supervisor.sock

[program:captioner]
command=/usr/local/bin/launch-captioner
directory=$WORKSPACE_APP_DIR
user=root
autostart=false
autorestart=true
startsecs=3
startretries=3
stopsignal=TERM
stopwaitsecs=20
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile=$CAPTIONER_LOG
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=3

[program:captioner-bootstrap]
command=/usr/local/bin/captioner-bootstrap
directory=/workspace
user=root
autostart=false
autorestart=false
startsecs=0
startretries=0
redirect_stderr=true
stdout_logfile=/workspace/logs/captioner-bootstrap.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=2

[program:captioner-nag]
command=/usr/local/bin/captioner-nag
directory=/workspace
user=root
autostart=false
autorestart=true
startsecs=1
startretries=3
stopsignal=TERM
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile=/workspace/logs/captioner-nag.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=2
EOF_SUPERVISOR
}

start_supervisor() {
  /usr/bin/supervisord -n -c "$SUPERVISOR_CONFIG" &
  SUPERVISOR_PID=$!
  export SUPERVISOR_PID

  local attempt
  for attempt in {1..50}; do
    if supervisorctl -c "$SUPERVISOR_CONFIG" status >/dev/null 2>&1; then
      log "Supervisor started (pid $SUPERVISOR_PID)"
      return 0
    fi
    sleep 0.1
  done

  log "FATAL: supervisor did not become ready"
  return 1
}

cleanup() {
  local rc=$?
  if [[ -n "${SUPERVISOR_PID:-}" ]]; then
    kill "$SUPERVISOR_PID" 2>/dev/null || true
  fi
  if [[ -n "${SSHD_PID:-}" ]]; then
    kill "$SSHD_PID" 2>/dev/null || true
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

sync_application
sync_pod_runtime
persist_telegram_environment
install_authorized_keys
start_ssh
configure_supervisor
start_supervisor

if (( $# > 0 )); then
  log "Executing custom container command: $*"
  "$@"
  exit $?
fi

if [[ -n "${TELEGRAM_BOT_TOKEN:-}" && -n "${TELEGRAM_CHAT_ID:-}" ]] && \
   is_true "${TELEGRAM_NAG_ENABLED:-true}"; then
  supervisorctl -c "$SUPERVISOR_CONFIG" start captioner-nag
  log "Telegram pod nag enabled every ${TELEGRAM_NAG_INTERVAL_SECONDS:-1800}s"
fi

case "${CAPTIONER_AUTO_START:-true}" in
  1|true|TRUE|yes|YES)
    log "Starting Ultimate Image Captioner on port ${GRADIO_SERVER_PORT:-7861}"
    supervisorctl -c "$SUPERVISOR_CONFIG" start captioner
    supervisorctl -c "$SUPERVISOR_CONFIG" start captioner-bootstrap
    log "App log: $CAPTIONER_LOG"
    log "Controls: captionerctl status|restart|logs|doctor"
    ;;
  *)
    log "CAPTIONER_AUTO_START is disabled; use 'captionerctl start' when ready"
    ;;
esac

wait "$SUPERVISOR_PID"
