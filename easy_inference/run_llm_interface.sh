#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
    echo "ERROR: .venv not found in $ROOT_DIR"
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python easy_inference/llm_interface.py "$@"
