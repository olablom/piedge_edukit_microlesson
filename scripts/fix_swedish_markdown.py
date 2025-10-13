#!/usr/bin/env python3
"""
Fix Swedish text in Jupyter notebook markdown cells.
Only affects markdown cells, leaves code cells untouched.
"""

import json
import pathlib
import re

# Swedish to English translations for common terms
REPLACEMENTS = [
    # Common Swedish terms
    (r"\bReflektionsfrågor\b", "Reflection Questions"),
    (r"\bSammanfattning\b", "Summary"),
    (r"\bNästa steg\b", "Next steps"),
    (r"\bFörstå vad som händer\b", "Understanding what happens"),
    (r"\bMål\b", "Goal"),
    (r"\bTips\b", "Tip"),
    (r"\bVad är\b", "What is"),
    (r"\bvarför\b", "why"),
    (r"\buseer\b", "use"),
    (r"\bFakeData\b", "FakeData"),  # Keep as is
    (r"\bExperimentera\b", "Experiment"),
    (r"\bTräning\b", "Training"),
    (r"\bhyperparametrar\b", "hyperparameters"),
    (r"\bepochs\b", "epochs"),  # Keep as is
    (r"\bbatch_size\b", "batch_size"),  # Keep as is
    (r"\bantal\b", "number of"),
    (r"\bgenomgångar\b", "passes"),
    (r"\bdatasetet\b", "dataset"),
    (r"\bbilder\b", "images"),
    (r"\bträningssteg\b", "training step"),
    (r"\bförtränade\b", "pretrained"),
    (r"\bvikter\b", "weights"),
    (r"\bSnabb\b", "Quick"),
    (r"\bträning\b", "training"),
    (r"\bingen\b", "no"),
    (r"\bpretrained\b", "pretrained"),  # Keep as is
    (r"\bVisa\b", "Show"),
    (r"\bträningsresultat\b", "training results"),
    (r"\bExperiment\b", "Experiment"),  # Keep as is
    (r"\bResultaten\b", "Results"),
    (r"\bvisas\b", "shown"),
    (r"\bovan\b", "above"),
    (r"\bträningsloopen\b", "training loop"),
    (r"\bDu kan\b", "You can"),
    (r"\bjämföra\b", "compare"),
    (r"\baccuracy\b", "accuracy"),  # Keep as is
    (r"\bloss\b", "loss"),  # Keep as is
    (r"\bmellan\b", "between"),
    (r"\bolika\b", "different"),
    (r"\bexperiment\b", "experiments"),
    (r"\bVad händer\b", "What happens"),
    (r"\böverfitting\b", "overfitting"),
    (r"\bhöjer\b", "increase"),
    (r"\bkan\b", "can"),
    (r"\blära sig\b", "learn"),
    (r"\bträningsdata\b", "training data"),
    (r"\bför bra\b", "too well"),
    (r"\bdåligt\b", "poorly"),
    (r"\bgeneralisera\b", "generalize"),
    (r"\bnya\b", "new"),
    (r"\bdata\b", "data"),  # Keep as is
    (r"\bDetta kallas\b", "This is called"),
    (r"\bKör\b", "Run"),
    (r"\bsamma\b", "same"),
    (r"\bmen\b", "but"),
    (r"\bmed\b", "with"),
    (r"\bvs\b", "vs"),  # Keep as is
    (r"\bvalideringsdata\b", "validation data"),
    (r"\bVarför\b", "Why"),
    (r"\bexporterar\b", "export"),
    (r"\btill\b", "to"),
    (r"\bONNX\b", "ONNX"),  # Keep as is
    (r"\bPi/edge\b", "Pi/edge"),  # Keep as is
    (r"\bär\b", "is"),
    (r"\bett\b", "a"),
    (r"\bstandardformat\b", "standard format"),
    (r"\bsom\b", "that"),
    (r"\bfungerar\b", "works"),
    (r"\bpå\b", "on"),
    (r"\bmånga\b", "many"),
    (r"\bplattformar\b", "platforms"),
    (r"\bCPU\b", "CPU"),  # Keep as is
    (r"\bGPU\b", "GPU"),  # Keep as is
    (r"\bmobil\b", "mobile"),
    (r"\bedge\b", "edge"),  # Keep as is
    (r"\bDet\b", "It"),
    (r"\bgör\b", "makes"),
    (r"\bmodellen\b", "model"),
    (r"\bportabel\b", "portable"),
    (r"\boptimerad\b", "optimized"),
    (r"\binference\b", "inference"),  # Keep as is
    (r"\bFördelar\b", "Advantages"),
    (r"\bSnabbare\b", "Faster"),
    (r"\bän\b", "than"),
    (r"\bPyTorch\b", "PyTorch"),  # Keep as is
    (r"\bMindre\b", "Less"),
    (r"\bminnesusening\b", "memory usage"),
    (r"\bFungerar\b", "Works"),
    (r"\bRaspberry Pi\b", "Raspberry Pi"),  # Keep as is
    (r"\bStöd\b", "Support"),
    (r"\bför\b", "for"),
    (r"\bkvantisering\b", "quantization"),
    (r"\bINT8\b", "INT8"),  # Keep as is
    (r"\bYour own\b", "Your own"),  # Keep as is
    (r"\bTask\b", "Task"),  # Keep as is
    (r"\bTräna\b", "Train"),
    (r"\ben\b", "a"),
    (r"\bmodell\b", "model"),
    (r"\bandra\b", "other"),
    (r"\binställningar\b", "settings"),
    (r"\bSuggestions\b", "Suggestions"),  # Keep as is
    (r"\bÖka\b", "Increase"),
    (r"\btill\b", "to"),
    (r"\bÄndra\b", "Change"),
    (r"\bTesta\b", "Test"),
    (r"\bCode to modify\b", "Code to modify"),  # Keep as is
    (r"\bÄndra dessa värden\b", "Change these values"),
    (r"\bTräningen\b", "Training"),
    (r"\bkörs\b", "runs"),
    (r"\bautomatiskt\b", "automatically"),
    (r"\bi\b", "in"),
    (r"\bnästa\b", "next"),
    (r"\bcell\b", "cell"),
    (r"\byou have nu lärt dig\b", "you have now learned"),
    (r"\bVad\b", "What"),
    (r"\bHur\b", "How"),
    (r"\bVarför\b", "Why"),
    (r"\bONNX-export\b", "ONNX export"),
    (r"\bviktigt\b", "important"),
    (r"\bedge deployment\b", "edge deployment"),  # Keep as is
    (r"\bNext step\b", "Next step"),  # Keep as is
    (r"\bGo to\b", "Go to"),  # Keep as is
    (r"\bför att\b", "to"),
    (r"\bförstå\b", "understand"),
    (r"\bvi\b", "we"),
    (r"\bmäter\b", "measure"),
    (r"\bmodellens\b", "model's"),
    (r"\bprestanda\b", "performance"),
    (r"\bViktiga\b", "Important"),
    (r"\bbegrepp\b", "concepts"),
    (r"\bEpochs\b", "Epochs"),  # Keep as is
    (r"\bBatch size\b", "Batch size"),  # Keep as is
    (r"\bPretrained weights\b", "Pretrained weights"),  # Keep as is
    (r"\bImageNet\b", "ImageNet"),  # Keep as is
    (r"\bStandardformat\b", "Standard format"),
]


def fix_notebook(notebook_path):
    """Fix Swedish text in a single notebook."""
    print(f"Processing {notebook_path}...")

    # Read notebook
    with open(notebook_path, encoding="utf-8") as f:
        nb = json.load(f)

    changed = False

    # Process each cell
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            # Get markdown source as string
            source = cell.get("source", [])
            if isinstance(source, list):
                text = "".join(source)
            else:
                text = source

            # Apply replacements
            original_text = text
            for pattern, replacement in REPLACEMENTS:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

            # Update if changed
            if text != original_text:
                cell["source"] = [text]
                changed = True

    # Save if changed
    if changed:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"[FIXED] {notebook_path}")
        return True
    else:
        print(f"[OK] No changes needed for {notebook_path}")
        return False


def main():
    """Fix all notebooks in the project."""
    # Find all .ipynb files (excluding .bak files)
    notebook_files = []
    for pattern in ["notebooks/*.ipynb", "*.ipynb"]:
        notebook_files.extend(pathlib.Path(".").glob(pattern))

    # Filter out .bak files
    notebook_files = [f for f in notebook_files if not f.name.endswith(".bak")]

    if not notebook_files:
        print("No notebooks found")
        return

    print(f"Found {len(notebook_files)} notebooks to process")

    fixed_count = 0
    for notebook_file in notebook_files:
        if fix_notebook(notebook_file):
            fixed_count += 1

    print(f"\n[SUMMARY] Fixed {fixed_count} out of {len(notebook_files)} notebooks")


if __name__ == "__main__":
    main()
