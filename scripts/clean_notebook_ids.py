#!/usr/bin/env python3
import json
from pathlib import Path


def clean_notebook(path: Path):
    """Remove 'id' fields from notebook cells to make it nbconvert-compatible."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if "id" in cell:
            del cell["id"]
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Cleaned {path.name}")


def main():
    for nb_path in Path("notebooks").glob("*.ipynb"):
        if not nb_path.name.endswith(".bak"):
            clean_notebook(nb_path)


if __name__ == "__main__":
    main()
