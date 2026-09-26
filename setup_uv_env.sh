#!/usr/bin/env bash
# Create a Python 3.12 environment with the locked dependencies and install this repository.
# Usage: bash setup_uv_env.sh   (set UV_VENV=/path/to/env to choose the target; default .venv-uv/)
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
UV_VENV="${UV_VENV:-$PROJECT_ROOT/.venv-uv}"
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
# Never infer the target from an activated environment.
unset VIRTUAL_ENV
if [[ ! -x "$UV_VENV/bin/python" ]]; then
    uv venv "$UV_VENV" --python 3.12
fi
"$UV_VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
uv pip sync --python "$UV_VENV/bin/python" "$PROJECT_ROOT/requirements.txt"
uv pip install --python "$UV_VENV/bin/python" --no-deps -e "$PROJECT_ROOT"
uv pip check --python "$UV_VENV/bin/python"
echo "Activate with: source \"$UV_VENV/bin/activate\""
