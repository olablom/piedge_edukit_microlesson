import re
import shutil
from pathlib import Path

import nbformat as nbf

NB = Path("notebooks/01_training_and_export.ipynb")
bak = NB.with_suffix(".ipynb.bak_sanitize")

# Heuristik: känns det som kod?
KW = {
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
LIB_HINT = re.compile(r"\b(torch|numpy|onnx|onnxruntime|pandas|matplotlib|sklearn)\b", re.I)
CODE_LINE = re.compile(r"^\s*(%|!|#|from\s|import\s|def\s|class\s|@|\w+\s*=\s*|print\(|for\s|\(|\{|\[)")


def looks_like_code(src: str) -> bool:
    s = src.strip()
    if not s:
        return True

    # Check for obvious markdown patterns
    if re.search(r"^#+\s", s, re.M):  # Headers
        return False
    if re.search(r"\*\*.*\*\*", s):  # Bold text
        return False
    if re.search(r"^\s*[-*]\s", s, re.M):  # Bullet points
        return False
    if re.search(r"^\s*\d+\.\s", s, re.M):  # Numbered lists
        return False
    if "Learning Goals" in s or "Concepts" in s or "Common Pitfalls" in s:
        return False

    if LIB_HINT.search(s):
        return True
    for line in s.splitlines()[:20]:  # kika på början
        if CODE_LINE.search(line):
            return True
    # nyckelordcount
    words = re.findall(r"[A-Za-z_]+", s)
    if sum(w in KW for w in words) >= 2:
        return True
    return False


nb = nbf.read(NB, as_version=4)
changed = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        src = cell.source or ""
        if not looks_like_code(src):
            # gör om till markdown
            new = nbf.v4.new_markdown_cell(src)
            new.metadata = {k: v for k, v in cell.metadata.items() if k != "execution"}
            nb.cells[i] = new
            changed += 1
    elif cell.cell_type == "markdown":
        # ibland hamnar ```python runt prosa – ta bort trippel-backticks som kapslar in ren text
        if "```" in cell.source and not looks_like_code(cell.source):
            cell.source = cell.source.replace("```python", "```").replace("```", "")

if changed:
    shutil.copyfile(NB, bak)
    nbf.write(nb, NB)
    print(f"[FIXED] Converted {changed} code cells -> markdown")
    print(f"[BACKUP] {bak}")
else:
    print("[OK] No suspicious code cells found")
