#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root
cd "$(dirname "$0")/.."

# 0) Ensure venv is active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "⚠️  .venv is not active. Run: source .venv/Scripts/activate (Git Bash)"; exit 1
fi

# 1) Clean outputs
echo "Cleaning models/, reports/, progress/, artifacts/ ..."
rm -rf models/ reports/ progress/ artifacts/
mkdir -p models reports progress artifacts

# 2) Ensure package (editable)
pip install -e .

NB="notebooks/00_run_everything.ipynb"
if [ ! -f "$NB" ]; then
  echo "Error: $NB not found." >&2
  exit 1
fi

# 3) Execute the notebook headlessly (fresh)
echo "Executing notebook headlessly: $NB"
python -m jupyter nbconvert \
  --to notebook --execute \
  --ExecutePreprocessor.timeout=1800 \
  --output notebooks/00_run_everything.ipynb \
  notebooks/00_run_everything.ipynb

# 4) Print receipt summary to terminal
if [[ -f progress/receipt.json ]]; then
  echo -e "\nReceipt summary:"
  python - <<'PY'
import json; r=json.load(open("progress/receipt.json"))
status = r.get("status") or ("PASS" if r.get("pass") else "FAIL")
print(("✅ VERIFY: PASS" if str(status).upper()=="PASS" else "❌ VERIFY: FAIL"))
m=r.get("metrics",{})
for k in ("fp32_mean_ms","int8_mean_ms","speedup_pct","mae"):
    if k in m: print(f"  {k}: {m[k]}")
PY
else
  echo "❌ No progress/receipt.json produced"
fi

# 5) Produce a clean HTML report (no code cells)
python -m jupyter nbconvert \
  --to html --no-input --no-prompt \
  --output run_everything_REPORT.html \
  --output-dir notebooks \
  notebooks/00_run_everything.ipynb

echo -e "\n✅ Done. Open report: notebooks/run_everything_REPORT.html"


