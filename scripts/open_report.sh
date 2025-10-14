#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

REPORT="notebooks/run_everything_REPORT.html"
if [ ! -f "$REPORT" ]; then
  echo "Error: $REPORT not found. Generate it first (run main.py or run_notebook_headless.sh)." >&2
  exit 1
fi

case "${OS:-}" in
  Windows_NT)
    # On Git Bash / Windows, use 'start' via cmd
    cmd.exe /C start "" "$(cygpath -w "$REPORT")" >/dev/null 2>&1 || true
    ;;
  *)
    # macOS or Linux
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$REPORT" >/dev/null 2>&1 || true
    elif command -v open >/dev/null 2>&1; then
      open "$REPORT" >/dev/null 2>&1 || true
    else
      echo "Open the file manually: $REPORT"
    fi
    ;;
esac

echo "Opened: $REPORT"


