#!/usr/bin/env python3
import json
import pathlib

# Direct character replacements for Swedish characters
char_replacements = {"å": "a", "ä": "a", "ö": "o", "Å": "A", "Ä": "A", "Ö": "O"}

for nb_path in pathlib.Path("notebooks").glob("*.ipynb"):
    if ".bak" in str(nb_path):
        continue
    print(f"Processing {nb_path}...")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    changed = False

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", [])
            if isinstance(source, list):
                text = "".join(source)
            else:
                text = source

            original_text = text
            for swedish, english in char_replacements.items():
                text = text.replace(swedish, english)

            if text != original_text:
                cell["source"] = [text]
                changed = True

    if changed:
        nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[FIXED] {nb_path}")
    else:
        print(f"[OK] {nb_path}")
