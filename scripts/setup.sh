#!/bin/bash
# Setup script for IDP Extractor development environment

set -e

echo "Setting up IDP Extractor development environment..."

# Install uv via pip if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv via pip..."
    pip install uv
fi

# Sync dependencies
echo "Installing dependencies..."
uv sync

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
uv run pre-commit install

echo "Setup complete! You can now use:"
echo "  uv run python -m idp_extractor ..."
echo "  ./scripts/format.sh"
echo "  ./scripts/lint.sh"
echo "  ./scripts/test.sh"