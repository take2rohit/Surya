#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
    echo "ERROR: .venv not found in $ROOT_DIR"
    exit 1
fi

source .venv/bin/activate
python easy_inference/llm_interface.py "$@"
