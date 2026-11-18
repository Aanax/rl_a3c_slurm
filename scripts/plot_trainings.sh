#!/usr/bin/env bash
# Plot Pong training curves for logs/<experiment> folders on ki.
# The figure matches the last Pong cell in plot_trainings.ipynb:
# mean of reward_mean_10 across runs, with a std band.
# Only runs that reach --min-frames (default 2000000) are included.
# The x-axis ends at the shortest included run.
#
# Usage:
#   scripts/plot_trainings.sh EXPERIMENT [EXPERIMENT ...] [--baseline NAME] [--min-frames N] [--out PATH]
#
# EXPERIMENT is the folder under logs/ on ki. A logs/ prefix is optional.
# The baseline line defaults to Hierarchial_v2_train_steps_4_100.
# The png defaults to plots/train_curves.png.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-ki}"
REMOTE_ROOT="${REMOTE_ROOT:-/s/ls4/users/aamore/rl_a3c_pytorch}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa2}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH_CONTROL_PATH="/tmp/plot_trainings_${USER}_%r@%h:%p"
RSYNC_RSH="ssh -i ${SSH_KEY} -o BatchMode=yes -o ConnectTimeout=90 -o ClearAllForwardings=yes -o ControlMaster=auto -o ControlPersist=15m -o ControlPath=${SSH_CONTROL_PATH}"

say() {
  printf '%s\n' "$*" >&2
}

usage() {
  say "Usage: $0 EXPERIMENT [EXPERIMENT ...] [--baseline NAME] [--min-frames N] [--out PATH]"
  say "  EXPERIMENT    logs folder name on ${REMOTE_HOST} (one line per name)"
  say "  --baseline    line to compare against (default: Hierarchial_v2_train_steps_4_100)"
  say "  --min-frames  drop runs shorter than this (default: 2000000)"
  say "  --out         png path (default: plots/train_curves.png)"
}

ssh_ki() {
  local attempt status
  for attempt in 1 2 3; do
    status=0
    ssh -i "${SSH_KEY}" -o BatchMode=yes -o ConnectTimeout=90 -o ClearAllForwardings=yes \
      -o ControlMaster=auto -o ControlPersist=15m -o ControlPath="${SSH_CONTROL_PATH}" \
      "${REMOTE_HOST}" "$@" || status=$?
    if [ "$status" -eq 0 ]; then
      return 0
    fi
    if [ "$status" -ne 255 ]; then
      return "$status"
    fi
    say "ssh connection failed (attempt ${attempt})"
    sleep 5
  done
  return 255
}

require_experiment_name() {
  local name="$1"
  if [[ ! "$name" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Experiment name must be letters, digits, '.', '_' or '-': ${name}" >&2
    exit 1
  fi
}

normalize_name() {
  local name="$1"
  name="${name%/}"
  name="${name#./}"
  name="${name#logs/}"
  printf '%s\n' "$name"
}

sync_logs() {
  local name="$1"
  local local_dir="${ROOT}/logs/${name}"
  require_experiment_name "$name"
  if ! ssh_ki "test -d '${REMOTE_ROOT}/logs/${name}'"; then
    echo "No ${REMOTE_ROOT}/logs/${name} on ${REMOTE_HOST}" >&2
    exit 1
  fi
  mkdir -p "${local_dir}"
  say "Syncing logs/${name}"
  rsync -a \
    --include='*log*' \
    --exclude='*' \
    -e "${RSYNC_RSH}" \
    "${REMOTE_HOST}:${REMOTE_ROOT}/logs/${name}/" \
    "${local_dir}/"
}

main() {
  local baseline="Hierarchial_v2_train_steps_4_100"
  local min_frames="2000000"
  local out="${ROOT}/plots/train_curves.png"
  local experiments=()
  local name

  while [ "$#" -gt 0 ]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --baseline)
        baseline="${2:?--baseline needs a folder name}"
        shift 2
        ;;
      --min-frames)
        min_frames="${2:?--min-frames needs a number}"
        shift 2
        ;;
      --out)
        out="${2:?--out needs a path}"
        shift 2
        ;;
      *)
        experiments+=("$1")
        shift
        ;;
    esac
  done

  if [ "${#experiments[@]}" -eq 0 ]; then
    usage >&2
    exit 1
  fi
  if ! [[ "$min_frames" =~ ^[0-9]+$ ]]; then
    echo "--min-frames must be a non-negative integer: ${min_frames}" >&2
    exit 1
  fi

  baseline="$(normalize_name "$baseline")"
  require_experiment_name "$baseline"

  local normalized=()
  for name in "${experiments[@]}"; do
    name="$(normalize_name "$name")"
    require_experiment_name "$name"
    normalized+=("$name")
  done

  local python_bin
  if [ -x "${ROOT}/.venv/bin/python" ]; then
    python_bin="${ROOT}/.venv/bin/python"
  else
    python_bin="$(command -v python3)"
  fi

  trap 'ssh -i "${SSH_KEY}" -o ControlPath="${SSH_CONTROL_PATH}" -O exit "${REMOTE_HOST}" 2>/dev/null || true' EXIT

  sync_logs "$baseline"
  local synced=" ${baseline} "
  local added=" "
  local exp_args=()
  for name in "${normalized[@]}"; do
    if [[ "$added" == *" ${name} "* ]]; then
      continue
    fi
    added="${added}${name} "
    if [[ "$synced" != *" ${name} "* ]]; then
      sync_logs "$name"
      synced="${synced}${name} "
    fi
    exp_args+=("${ROOT}/logs/${name}")
  done

  mkdir -p "$(dirname "$out")"
  mkdir -p /tmp/mplconfig-plot
  say "Drawing ${out}"
  (
    cd "$ROOT"
    MPLCONFIGDIR=/tmp/mplconfig-plot "$python_bin" src/plot_train_curves.py \
      --baseline "${ROOT}/logs/${baseline}" \
      --min-frames "$min_frames" \
      --out "$out" \
      "${exp_args[@]}"
  )
}

main "$@"
