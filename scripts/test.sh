#!/bin/bash
# Run tests

set -e

echo "Running tests..."

# Check if we're in a uv environment
if command -v uv &> /dev/null; then
    uv run pytest "$@"
else
    echo "uv not found, trying direct pytest..."
    if command -v pytest &> /dev/null; then
        pytest "$@"
    else
        echo "Error: pytest not found. Please install pytest or run setup.sh first."
        exit 1
    fi
fi

echo "Tests complete!"