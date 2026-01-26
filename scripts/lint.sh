#!/bin/bash
# Lint code using ruff

set -e

echo "Linting code with ruff..."

# Check if we're in a uv environment
if command -v uv &> /dev/null; then
    uv run ruff check .
    uv run ruff format --check .
else
    echo "uv not found, trying direct ruff..."
    if command -v ruff &> /dev/null; then
        ruff check .
        ruff format --check .
    else
        echo "Error: ruff not found. Please install ruff or run setup.sh first."
        exit 1
    fi
fi

echo "Linting complete!"