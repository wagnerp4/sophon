#!/usr/bin/env bash
# Disposable QEMU VM used by sophon's `vm` sandbox mode. Runs inside WSL/Linux.
set -euo pipefail

VM_DIR="${SOPHON_VM_DIR:-$HOME/.local/share/sophon/vm}"
SSH_PORT="${SOPHON_VM_SSH_PORT:-2222}"
MEM_MB="${SOPHON_VM_MEM_MB:-2048}"
CPUS="${SOPHON_VM_CPUS:-2}"
DISK_SIZE="${SOPHON_VM_DISK:-20G}"
IMAGE_URL="${SOPHON_VM_IMAGE_URL:-https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img}"
LOCAL_BASE_CANDIDATES=("$HOME/vms/qemu-lab/jammy-server-cloudimg-amd64.img")
VM_USER="sophon"

BASE="$VM_DIR/base.img"
DISK="$VM_DIR/disk.qcow2"
SEED="$VM_DIR/seed.iso"
KEY="$VM_DIR/id_ed25519"
PIDFILE="$VM_DIR/qemu.pid"
SERIAL="$VM_DIR/serial.log"

die() { echo "error: $*" >&2; exit 1; }

need() {
  command -v "$1" >/dev/null 2>&1 || die "$1 not found (sudo apt install -y $2)"
}

kvm_usable() { [[ -r /dev/kvm && -w /dev/kvm ]]; }

running() {
  [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

ssh_base() {
  ssh -i "$KEY" -p "$SSH_PORT" \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR -o BatchMode=yes -o ConnectTimeout=5 \
    "$VM_USER@127.0.0.1" "$@"
}

ssh_ready() { ssh_base true >/dev/null 2>&1; }

write_seed() {
  local tmp
  tmp="$(mktemp -d)"
  cat >"$tmp/meta-data" <<EOF
instance-id: sophon-$(date +%s)
local-hostname: sophon-sandbox
EOF
  cat >"$tmp/user-data" <<EOF
#cloud-config
users:
  - name: $VM_USER
    shell: /bin/bash
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - $(cat "$KEY.pub")
ssh_pwauth: false
package_update: false
EOF
  xorriso -as mkisofs -quiet -output "$SEED" -volid cidata -joliet -rock \
    "$tmp/user-data" "$tmp/meta-data" 2>/dev/null
  rm -rf "$tmp"
}

new_disk() {
  rm -f "$DISK"
  qemu-img create -q -f qcow2 -F qcow2 -b "$BASE" "$DISK" "$DISK_SIZE"
  write_seed
}

cmd_setup() {
  need qemu-system-x86_64 qemu-system-x86
  need qemu-img qemu-utils
  need xorriso xorriso
  need ssh openssh-client
  mkdir -p "$VM_DIR"
  [[ -f "$KEY" ]] || ssh-keygen -q -t ed25519 -N "" -C sophon-sandbox -f "$KEY"
  if [[ ! -f "$BASE" ]]; then
    local found=""
    for cand in "${LOCAL_BASE_CANDIDATES[@]}"; do
      [[ -f "$cand" ]] && found="$cand" && break
    done
    if [[ -n "$found" ]]; then
      echo "base image: linking $found"
      ln "$found" "$BASE" 2>/dev/null || cp "$found" "$BASE"
    else
      echo "base image: downloading $IMAGE_URL"
      curl -fL --progress-bar -o "$BASE.part" "$IMAGE_URL"
      mv "$BASE.part" "$BASE"
    fi
  fi
  [[ -f "$DISK" ]] || new_disk
  echo "setup ok: $VM_DIR"
  kvm_usable || echo "note: /dev/kvm not usable by $(id -un); VM will use slow TCG emulation. Fix: sudo usermod -aG kvm $(id -un), then 'wsl --shutdown'."
}

cmd_up() {
  [[ -f "$DISK" ]] || cmd_setup
  if running; then
    echo "already running (pid $(cat "$PIDFILE"))"
    return 0
  fi
  local accel="tcg" cpu="max"
  if kvm_usable; then accel="kvm"; cpu="host"; fi
  : >"$SERIAL"
  qemu-system-x86_64 \
    -name sophon-sandbox \
    -machine q35,accel="$accel" -cpu "$cpu" \
    -m "$MEM_MB" -smp "$CPUS" \
    -drive file="$DISK",format=qcow2,if=virtio \
    -drive file="$SEED",format=raw,media=cdrom,readonly=on \
    -nic user,model=virtio-net-pci,hostfwd=tcp:127.0.0.1:"$SSH_PORT"-:22 \
    -display none -serial file:"$SERIAL" \
    -daemonize -pidfile "$PIDFILE"
  echo "started (pid $(cat "$PIDFILE"), accel=$accel, ssh port $SSH_PORT)"
  if [[ "${1:-}" == "--wait" ]]; then cmd_wait "${2:-900}"; fi
}

cmd_wait() {
  local limit="${1:-900}" waited=0
  until ssh_ready; do
    running || die "VM exited during boot; see $SERIAL"
    (( waited >= limit )) && die "ssh not ready after ${limit}s; see $SERIAL"
    sleep 5; waited=$((waited + 5))
  done
  echo "ready after ~${waited}s"
}

cmd_status() {
  if ! running; then
    echo "state: stopped"
    [[ -f "$DISK" ]] || echo "disk: missing (run setup)"
    return 0
  fi
  if ssh_ready; then
    echo "state: ready (pid $(cat "$PIDFILE"), ssh 127.0.0.1:$SSH_PORT)"
    ssh_base 'echo "guest: $(. /etc/os-release; echo $PRETTY_NAME), kernel $(uname -r), up $(uptime -p)"'
  else
    echo "state: booting (pid $(cat "$PIDFILE")); tail: $(tail -c 300 "$SERIAL" | tr -d '\r' | tail -n 1)"
  fi
  kvm_usable && echo "accel: kvm" || echo "accel: tcg (slow; add yourself to the kvm group)"
}

cmd_down() {
  running || { echo "not running"; rm -f "$PIDFILE"; return 0; }
  local pid; pid="$(cat "$PIDFILE")"
  ssh_base 'sudo poweroff' >/dev/null 2>&1 || true
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
  rm -f "$PIDFILE"
  echo "stopped"
}

cmd_reset() {
  cmd_down
  new_disk
  echo "disk reset to a clean base image"
}

# exec-b64 <timeout_s> <cwd_b64> <command_b64>; base64 keeps quoting intact through wsl.exe and ssh.
cmd_exec_b64() {
  local timeout_s="$1" cwd_b64="$2" cmd_b64="$3"
  running || die "VM is not running (/vm up)"
  ssh_ready || die "VM is still booting (/vm status)"
  ssh_base "mkdir -p ~/work; cd -- \"\$(echo $cwd_b64 | base64 -d)\" 2>/dev/null || cd ~; timeout -k 5 ${timeout_s} bash -lc \"\$(echo $cmd_b64 | base64 -d)\""
}

cmd_ssh_hint() {
  echo "ssh -i $KEY -p $SSH_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $VM_USER@127.0.0.1"
}

case "${1:-status}" in
  setup) cmd_setup ;;
  up) shift; cmd_up "$@" ;;
  wait) shift; cmd_wait "$@" ;;
  status) cmd_status ;;
  down) cmd_down ;;
  reset) cmd_reset ;;
  exec-b64) shift; cmd_exec_b64 "$@" ;;
  ssh) shift; if [[ $# -gt 0 ]]; then ssh_base "$@"; else cmd_ssh_hint; fi ;;
  *) die "usage: sandbox-vm.sh setup|up [--wait [s]]|wait [s]|status|down|reset|ssh [cmd]|exec-b64 T CWD CMD" ;;
esac
