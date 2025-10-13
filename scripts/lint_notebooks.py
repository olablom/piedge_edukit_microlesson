#!/usr/bin/env python3
import json
import os
import pathlib
import re
import sys

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

SWEDISH_CHARS = re.compile(r"[åäöÅÄÖ]")
# Lägg till/ta bort ord efter behov
SWEDISH_WORDS = re.compile(
    r"\b("
    r"Sammanfattning|Reflektionsfrågor|Nästa steg|Varför|Innan du kör|"
    r"Kör|Fortsätt|Lärandemål|Syfte|Notera|Obs|Förutsättningar|"
    r"Tidsbudget|Framgångskriterier|Förklaring"
    r")\b",
    re.IGNORECASE,
)


def check_notebook(path: pathlib.Path):
    bad = []
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue
        txt = "".join(cell.get("source", []))
        hits = []
        if SWEDISH_CHARS.search(txt):
            hits.append("å/ä/ö")
        if SWEDISH_WORDS.search(txt):
            hits.append("svenska ord")
        if hits:
            excerpt = txt.strip().splitlines()[:3]
            # Clean excerpt for Windows console
            clean_excerpt = []
            for line in excerpt:
                try:
                    clean_excerpt.append(line.encode("ascii", "replace").decode("ascii"))
                except:
                    clean_excerpt.append(line)
            bad.append((i, ", ".join(hits), " | ".join(clean_excerpt)))
    return bad


def main(files):
    failures = []
    for f in files:
        p = pathlib.Path(f)
        if p.suffix != ".ipynb":
            continue
        # Skip checkpoint files
        if ".ipynb_checkpoints" in str(p):
            continue
        res = check_notebook(p)
        if res:
            for idx, kind, ex in res:
                failures.append(f"{p}:{idx}: {kind} -> {ex}")
    if failures:
        print("Swedish detected in markdown cells:")
        for failure in failures:
            print(failure)
        sys.exit(1)


if __name__ == "__main__":
    files = sys.argv[1:] or []
    if not files:
        # fallback: lint alla spårade notebooks
        import subprocess

        out = subprocess.check_output(["git", "ls-files", "*.ipynb"], text=True)
        files = [x for x in out.splitlines() if x.strip()]
    main(files)
