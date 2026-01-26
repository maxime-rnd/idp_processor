#!/bin/bash
# Make all scripts executable
# Run this script once after cloning the repository

echo "Making scripts executable..."

# Find all .sh files in scripts directory and make them executable
find scripts/ -name "*.sh" -type f -exec chmod +x {} \;

echo "All scripts are now executable!"
echo ""
echo "You can now run:"
echo "  ./scripts/setup.sh"
echo "  ./scripts/clean.sh"
echo "  ./scripts/format.sh"
echo "  ./scripts/lint.sh"
echo "  ./scripts/test.sh"
echo "  ./scripts/run_folder.sh <folder_path>"