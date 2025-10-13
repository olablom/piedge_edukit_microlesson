# scripts/fix_notebook_celltypes.py
import json
import re
import shutil
from pathlib import Path

nb_path = Path("notebooks/01_training_and_export.ipynb")
bak = nb_path.with_suffix(".ipynb.bak_autofix")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

PY_KEYWORDS = {
    "def",
    "class",
    "import",
    "from",
    "for",
    "while",
    "if",
    "elif",
    "else",
    "try",
    "except",
    "with",
    "return",
    "yield",
    "lambda",
    "async",
    "await",
    "pass",
    "break",
    "continue",
}
CODE_SIGNS = re.compile(
    r"(^%|^!|^\s*#\s*pyright|^\s*from\s|^\s*import\s|^\s*def\s|^\s*class\s)|(=|:=|\(|\)|\[|\]|\{|\}|:)\s*$|\b(torch|numpy|plt|sklearn|onnx|onnxruntime|pandas)\b",
    re.I | re.M,
)


def looks_like_code(src: str) -> bool:
    s = src.strip()
    if not s:
        return True
    if CODE_SIGNS.search(s):
        return True
    # Count keywords; if many, treat as code
    kw = sum(1 for w in re.findall(r"[A-Za-z_]+", s) if w in PY_KEYWORDS)
    return kw >= 2


changed = 0
for c in nb.get("cells", []):
    if c.get("cell_type") == "code":
        # Join to string for heuristics
        src = "".join(c.get("source", []))
        if not looks_like_code(src):
            c["cell_type"] = "markdown"
            # Remove execution artifacts if any
            c.pop("outputs", None)
            c.pop("execution_count", None)
            changed += 1

if changed:
    shutil.copyfile(nb_path, bak)
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[FIXED] Converted {changed} mis-typed code cells to markdown")
    print(f"[BACKUP] {bak}")
else:
    print("[OK] No suspicious code cells found")
