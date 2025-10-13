#!/usr/bin/env python3
"""
PiEdge EduKit - Notebook Validator
Validates notebooks for Swedish text, preprocessing contracts, and ONNX exports
"""

import json
import re
from pathlib import Path

# Swedish patterns to detect
SV_PATTERNS = [
    r"\bär\b", r"\boch\b", r"\bing(?:en|a)\b", r"\bmedvetet\b", r"\butan\b", 
    r"[åäöÅÄÖ]"
]

# ONNX export pattern
ONNX_RE = re.compile(r"torch\.\s*onnx\s*\.?\s*export", re.I)

def has_swedish_text(text):
    """Check if text contains Swedish words/characters"""
    return any(re.search(pattern, text, flags=re.I) for pattern in SV_PATTERNS)

def has_preprocessing_contract(text):
    """Check if text contains preprocessing contract"""
    return ("Preprocessing Contract" in text and 
            re.search(r"64\s*[x×]\s*64", text) and 
            re.search(r"\bno\s+normaliz", text, re.I))

def has_onnx_export(text):
    """Check if text contains ONNX export code"""
    return bool(ONNX_RE.search(text))

def validate_notebook(notebook_path):
    """Validate a single notebook"""
    print(f"📓 Validating: {notebook_path}")
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read notebook: {e}")
        return False
    
    issues = []
    
    # Extract all text from markdown cells
    markdown_text = ""
    code_text = ""
    
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            markdown_text += "".join(cell.get("source", []))
        elif cell.get("cell_type") == "code":
            code_text += "".join(cell.get("source", []))
    
    # Check for Swedish text
    if has_swedish_text(markdown_text):
        issues.append("Swedish text found in markdown")
    
    # Check for preprocessing contract
    if not has_preprocessing_contract(markdown_text):
        issues.append("Missing or incomplete preprocessing contract")
    
    # Check for ONNX export (only in training notebook)
    if "01_training_and_export" in str(notebook_path):
        if not has_onnx_export(code_text):
            issues.append("Missing ONNX export code")
    
    if issues:
        print(f"❌ Issues found: {', '.join(issues)}")
        return False
    else:
        print("✅ All checks passed")
        return True

def main():
    print("🔍 PiEdge EduKit - Notebook Validation")
    print("=" * 50)
    
    # Find all notebooks
    notebook_dir = Path("notebooks")
    notebooks = list(notebook_dir.glob("*.ipynb"))
    
    if not notebooks:
        print("❌ No notebooks found in notebooks/ directory")
        sys.exit(1)
    
    # Validate each notebook
    success_count = 0
    for notebook in sorted(notebooks):
        if validate_notebook(notebook):
            success_count += 1
        print()
    
    print("=" * 50)
    print(f"📊 Results: {success_count}/{len(notebooks)} notebooks passed validation")
    
    if success_count == len(notebooks):
        print("🎉 All notebooks passed validation!")
    else:
        print("⚠️  Some notebooks failed validation. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
