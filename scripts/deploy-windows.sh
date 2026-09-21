#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SYNC_VENV_FLAG=()
if [[ "${1:-}" == "--sync-venv" ]]; then
  SYNC_VENV_FLAG=(--sync-venv)
elif [[ "${1:-}" == "--skip-venv" ]]; then
  SYNC_VENV_FLAG=(--skip-venv)
elif [[ "${1:-}" == "--profile-only" ]]; then
  SYNC_VENV_FLAG=(--profile-only)
fi

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m cli.host.windows_deploy "${SYNC_VENV_FLAG[@]}"
