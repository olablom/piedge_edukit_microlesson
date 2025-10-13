# scripts/notebook_final_cleanup.py
import json
import shutil
from pathlib import Path

REPLACEMENTS = {
    # terminologi
    "onNX": "ONNX",
    "Onnx": "ONNX",
    "onNx": "ONNX",
    "FakeData is syntetiska": "FakeData is synthetic",
    "Preprocessing contract": "Preprocessing contract",
    # svenska -> engelska (vanliga rubriker/fraser)
    "Varför": "Why",
    "Syfte": "Purpose",
    "Lärandemål": "Learning goals",
    "Körläge": "Run mode",
    "Förklaring": "Explanation",
    "Varning": "Warning",
    "OBS": "Note",
    "Träningsresultat": "Training results",
    "nar ": "when ",
    "jamfor": "compare",
    # typos/engelska småfix
    "ptoerns": "patterns",
    "Fltoen": "Flatten",
    "Stakest": "Start",
    "epand": "epoch",
    "inferens": "inference",
    "nodvandigt": "necessary",
    "tranings": "training",
    "tranings-": "training-",
    "Classer": "classes",
    # unicode som kan bråka i terminalutskrifter
    "→": "->",
    "×": "x",
}

CODE_HARD_HINTS = tuple(
    [
        "import ",
        "from ",
        "def ",
        "class ",
        "@",
        "for ",
        "while ",
        "if ",
        "try:",
        "except",
        "with ",
        "return",
        "yield",
        "print(",
        "torch.",
        "numpy",
        "nn.",
        "plt.",
        "%",
        "!",
        "# ruff:",
    ]
)


def looks_like_code(text: str) -> bool:
    s = (text or "").lstrip()
    if not s:
        return True
    return any(s.startswith(p) for p in CODE_HARD_HINTS)


def de_swedify_markdown(text: str) -> str:
    # mild ersättning: byt bara i markdown
    out = text
    for k, v in REPLACEMENTS.items():
        out = out.replace(k, v)
    # translitera åäö i prosa om de råkat smyga in
    out = (
        out.replace("å", "a").replace("Å", "A").replace("ä", "a").replace("Ä", "A").replace("ö", "o").replace("Ö", "O")
    )
    return out


def process(nb_path: Path) -> int:
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    changed = 0
    for cell in data.get("cells", []):
        src_list = cell.get("source", [])
        src = "".join(src_list)

        # 1) code->markdown om det uppenbart inte är kod
        if cell.get("cell_type") == "code" and not looks_like_code(src):
            cell["cell_type"] = "markdown"
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
            changed += 1

        # 2) ersättningar i markdown
        if cell.get("cell_type") == "markdown" and src:
            new_src = de_swedify_markdown(src)
            if new_src != src:
                cell["source"] = new_src
                changed += 1

    if changed:
        bak = nb_path.with_suffix(".ipynb.bak_final")
        shutil.copyfile(nb_path, bak)
        nb_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return changed


def main():
    base = Path("notebooks")
    total = 0
    for nb in base.glob("*.ipynb"):
        if ".ipynb_checkpoints" in str(nb):
            continue
        total += process(nb)
    print(f"[cleanup] edits: {total}")


if __name__ == "__main__":
    main()
