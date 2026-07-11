#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_REPO_URL="${RUNTIME_REPO_URL:-https://github.com/markwelshboy/pod-runtime.git}"
RUNTIME_DIR="${RUNTIME_DIR:-/workspace/pod-runtime}"
IMAGE_APP_DIR="${CAPTIONER_IMAGE_DIR:-/opt/Ultimate_Image_Captioner_Pro}"
WORKSPACE_APP_DIR="${CAPTIONER_WORKSPACE_DIR:-/workspace/Ultimate_Image_Captioner_Pro}"

export POD_RUNTIME_DIR="$RUNTIME_DIR"

log() {
  printf '[captioner-start] %s\n' "$*"
}

sync_application() {
  if [[ ! -f "$IMAGE_APP_DIR/app.py" ]]; then
    log "FATAL: baked application not found at $IMAGE_APP_DIR"
    exit 1
  fi

  mkdir -p "$WORKSPACE_APP_DIR"

  # Refresh image-owned application code without deleting persistent state.
  rsync -a --delete \
    --exclude='.git/' \
    --exclude='outputs/' \
    --exclude='presets/' \
    --exclude='model_files_*/' \
    --exclude='.cache/' \
    "$IMAGE_APP_DIR/" "$WORKSPACE_APP_DIR/"

  mkdir -p \
    "$WORKSPACE_APP_DIR/outputs" \
    "$WORKSPACE_APP_DIR/presets" \
    "$WORKSPACE_APP_DIR/model_files_beta_one" \
    "$WORKSPACE_APP_DIR/model_files_qwen3_vl3_8b_instruct" \
    /workspace/logs

  # Seed newly shipped presets while preserving user changes and additions.
  if [[ -d "$IMAGE_APP_DIR/presets" ]]; then
    rsync -a --ignore-existing "$IMAGE_APP_DIR/presets/" "$WORKSPACE_APP_DIR/presets/"
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

cleanup() {
  local rc=$?
  if [[ -n "${SSHD_PID:-}" ]]; then
    kill "$SSHD_PID" 2>/dev/null || true
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

sync_application
sync_pod_runtime
install_authorized_keys
start_ssh

if (( $# > 0 )); then
  log "Executing custom container command: $*"
  "$@"
  exit $?
fi

case "${CAPTIONER_AUTO_START:-true}" in
  1|true|TRUE|yes|YES)
    log "Launching Ultimate Image Captioner on port ${GRADIO_SERVER_PORT:-7861}"
    /usr/local/bin/launch-captioner
    ;;
  *)
    log "CAPTIONER_AUTO_START is disabled; keeping the container alive for SSH"
    tail -f /dev/null
    ;;
esac
