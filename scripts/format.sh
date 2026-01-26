#!/bin/bash
# Format code using ruff

set -e

echo "Formatting code with ruff..."

# Check if we're in a uv environment
if command -v uv &> /dev/null; then
    uv run ruff format .
    uv run ruff check --fix .
else
    echo "uv not found, trying direct ruff..."
    if command -v ruff &> /dev/null; then
        ruff format .
        ruff check --fix .
    else
        echo "Error: ruff not found. Please install ruff or run setup.sh first."
        exit 1
    fi
fi

echo "Code formatting complete!"