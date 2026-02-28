#!/bin/bash
# Run the Surya Chat UI
cd "$(dirname "$0")"
source .venv/bin/activate
python chat_ui.py
