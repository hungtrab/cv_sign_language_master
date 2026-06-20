#!/usr/bin/env bash
# Step 1: Create venv and install dependencies
set -e

echo "=== Step 1: Setup ==="

if [ ! -d ".venv" ]; then
    echo "Creating venv..."
    python3 -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/pip install -e ".[dev]" -q

echo "Done. Activate with: source .venv/bin/activate"
