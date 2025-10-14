#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3.12 -m venv .venv

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

echo "Activated .venv. To use later: source .venv/bin/activate"

