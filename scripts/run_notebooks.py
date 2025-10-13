#!/usr/bin/env python3
"""
PiEdge EduKit - Notebook Runner
Executes all notebooks in sequence and saves executed copies
"""

import subprocess
import os
import sys
from pathlib import Path

def run_notebook(notebook_path, output_dir="reports/nbexec"):
    """Run a single notebook and save executed copy"""
    print(f"📓 Running: {notebook_path}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run notebook
    cmd = [
        "python", "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--output-dir", output_dir,
        "--output", f"{Path(notebook_path).stem}.executed.ipynb",
        str(notebook_path)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Completed: {notebook_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {notebook_path}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 PiEdge EduKit - Running all notebooks")
    print("=" * 50)
    
    # List of notebooks to run in order
    notebooks = [
        "notebooks/01_training_and_export.ipynb",
        "notebooks/02_latency_benchmark.ipynb", 
        "notebooks/03_quantization.ipynb",
        "notebooks/04_evaluate_and_verify.ipynb"
    ]
    
    # Check if notebooks exist
    for notebook in notebooks:
        if not Path(notebook).exists():
            print(f"❌ Notebook not found: {notebook}")
            sys.exit(1)
    
    # Run notebooks
    success_count = 0
    for notebook in notebooks:
        if run_notebook(notebook):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {success_count}/{len(notebooks)} notebooks completed successfully")
    
    if success_count == len(notebooks):
        print("🎉 All notebooks completed successfully!")
    else:
        print("⚠️  Some notebooks failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
