#!/usr/bin/env bash
set -euo pipefail

# Ensure we're in the repo root (where this script lives in scripts/)
cd "$(dirname "$0")/.."

# Activate virtual environment (Git Bash on Windows or Unix)
if [ -f .venv/Scripts/activate ]; then
  # Windows venv (Git Bash)
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
elif [ -f .venv/bin/activate ]; then
  # Unix venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "Error: .venv not found. Run scripts/setup_venv.sh first." >&2
  exit 1
fi

NB_PATH="notebooks/00_run_everything.ipynb"
if [ ! -f "$NB_PATH" ]; then
  echo "Error: $NB_PATH not found." >&2
  exit 1
fi

echo "Clearing outputs in $NB_PATH ..."
python -m jupyter nbconvert --clear-output --inplace "$NB_PATH"

echo "Opening Jupyter Lab with $NB_PATH ..."
python -m jupyter lab "$NB_PATH"


