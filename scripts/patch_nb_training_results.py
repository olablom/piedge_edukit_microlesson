#!/usr/bin/env python3
# filename: scripts/patch_nb_training_results.py
"""
Patch the 'Training results (Experiment 1)' cell in
notebooks/01_training_and_export.ipynb to avoid formatting strings
as floats (:.3f). Makes printing robust across missing/str values.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    nb_path = Path("notebooks/01_training_and_export.ipynb")
    if not nb_path.exists():
        print("[WARN] Notebook not found:", nb_path)
        return 0

    data = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])

    safe_cell = (
        "# Show training results (safe)\n"
        "import json, os\n\n"
        "def fmt_float(v):\n"
        "    try:\n"
        "        return '{:.3f}'.format(float(v))\n"
        "    except Exception:\n"
        "        return str(v)\n\n"
        "if os.path.exists('./models_exp1/training_info.json'):\n"
        "    with open('./models_exp1/training_info.json', 'r') as f:\n"
        "        info = json.load(f)\n\n"
        "    print('Training results (Experiment 1):')\n"
        "    print('Final accuracy:', fmt_float(info.get('final_accuracy', 'N/A')))\n"
        "    print('Final loss:', fmt_float(info.get('final_loss', 'N/A')))\n"
        "    print('Epochs:', info.get('epochs', 'N/A'))\n"
        "    print('Batch size:', info.get('batch_size', 'N/A'))\n"
        "else:\n"
        "    print('Training info missing')\n"
    )

    updated = False
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "Training results (Experiment 1)" in src and "final_accuracy" in src:
            cell["source"] = [line for line in safe_cell.splitlines(keepends=True)]
            updated = True
            break

    if updated:
        nb_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print("[UPDATED] Patched training results cell")
    else:
        print("[OK] No matching training results cell found – nothing changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
