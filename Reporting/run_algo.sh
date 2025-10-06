#!/bin/bash

# Script to run algo.py using the virtual environment in parent directory
# This ensures we use the shared venv with yfinance installed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PARENT_DIR/venv"

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Run algo.py with the parent directory's venv
echo "Running algo.py with venv from parent directory..."
"$VENV_PATH/bin/python" "$SCRIPT_DIR/algo.py" "$@"
