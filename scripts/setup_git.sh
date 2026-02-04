#!/bin/bash
set -e

# Ensure we are in the repo root
cd "$(dirname "$0")/.."

echo "Installing git filters..."
if command -v nbstripout >/dev/null 2>&1; then
    nbstripout --install
    echo "nbstripout filter installed successfully."
else
    # Try running from the venv if not in path
    if [ -f ".venv/bin/nbstripout" ]; then
        .venv/bin/nbstripout --install
        echo "nbstripout filter installed successfully (via .venv)."
    else
        echo "Error: nbstripout not found. Please run 'pip install -e \".[dev]\"' first."
        exit 1
    fi
fi
