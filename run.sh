#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

if [[ -f "$PROJECT_ROOT/my_env_name/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/my_env_name/bin/activate"
fi

python "$PROJECT_ROOT/run_main_with_dashboard.py" "$@"
