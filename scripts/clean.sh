#!/bin/bash
# Clean script for IDP Extractor

set -e

echo "Cleaning up build artifacts and cache files..."

# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove build artifacts
rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true

# Remove coverage reports
rm -rf htmlcov/ .coverage .coverage.* coverage.xml 2>/dev/null || true

# Remove pytest cache
rm -rf .pytest_cache/ 2>/dev/null || true

# Remove mypy cache
rm -rf .mypy_cache/ 2>/dev/null || true

# Remove ruff cache
rm -rf .ruff_cache/ 2>/dev/null || true

# Remove uv cache (optional, uncomment if needed)
# rm -rf ~/.cache/uv/ 2>/dev/null || true

echo "Cleanup complete!"