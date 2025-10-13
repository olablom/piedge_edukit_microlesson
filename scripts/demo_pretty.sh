#!/bin/bash
# PiEdge EduKit - Pretty Demo Script
# Runs a complete demo with visualizations

set -euo pipefail

echo "🎨 PiEdge EduKit - Pretty Demo"
echo "==============================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Run 'bash scripts/setup_venv.sh' first."
    exit 1
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows Git Bash
    source .venv/Scripts/activate
    PYTHON_CMD=".venv/Scripts/python"
else
    # macOS/Linux
    source .venv/bin/activate
    PYTHON_CMD=".venv/bin/python"
fi

echo "🚀 Starting demo..."

# Run all notebooks
echo "📓 Running all notebooks..."
$PYTHON_CMD scripts/run_notebooks.py

# Generate HTML reports
echo "📊 Generating HTML reports..."
$PYTHON_CMD -m jupyter nbconvert --to html --output-dir reports/nbhtml notebooks/*.ipynb

# Run validation
echo "✅ Running validation..."
$PYTHON_CMD scripts/nb_validate.py

# Run sanity check
echo "🔍 Running sanity check..."
$PYTHON_CMD scripts/nb_sanity_check.py

echo ""
echo "🎉 Demo completed successfully!"
echo "==============================="
echo "📁 Check the following directories for results:"
echo "   - reports/nbexec/     (executed notebooks)"
echo "   - reports/nbhtml/     (HTML reports)"
echo "   - models/             (trained models)"
echo "   - reports/            (plots and metrics)"
echo ""
echo "🌐 Open reports/nbhtml/ in your browser to view the results!"
